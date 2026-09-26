//! Reranking via a decoder model formatted as a yes/no judgment, following
//! Qwen3-Reranker's own real, published recipe rather than a BERT-style
//! cross-encoder.
//!
//! `docs/keel-integration.md`'s "3. `/v1/rerank`" section flagged Keel's own
//! `bge-reranker-v2-m3` as a true cross-encoder (bidirectional, a
//! classification head over `[CLS]`) that infero's decoder-only forward pass
//! cannot run at all -- and said to confirm feasibility before committing to
//! encoder support. The real finding: Qwen released a *decoder* reranker
//! alongside Qwen3-Embedding, same architecture family infero already runs,
//! that produces a comparable relevance score by reading the logits of the
//! literal tokens "yes" and "no" at the position right after a fixed
//! instruction prompt. This is that recipe, not a new architecture.

use anyhow::{Context, Result, ensure};

use crate::{BatchItem, BatchItemKind, Model};

/// Same batch-size reasoning as `crate::embed::MAX_BATCH` -- a hard cap this
/// endpoint's own HTTP layer enforces before ever reaching the model.
pub const MAX_BATCH: usize = 64;

/// Qwen3-Reranker's own fixed system+user prefix, verbatim from its real,
/// published model card (`Qwen/Qwen3-Reranker-0.6B`'s README, "Using
/// Transformers" section) -- not re-derived, since a reranker's score is
/// only meaningful if the prompt matches what it was tuned against exactly.
pub const PREFIX: &str = "<|im_start|>system\nJudge whether the Document meets the requirements \
based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \
\"no\".<|im_end|>\n<|im_start|>user\n";

/// Same source: the fixed suffix that puts the model at the exact position
/// its own logits recipe reads (right after an empty `<think></think>` --
/// this checkpoint reasons about the judgment inside those tags when
/// generating for real, but the score only ever reads the very next token,
/// so a pre-closed empty think block is what the model card itself uses to
/// skip straight to the answer position).
pub const SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// The instruction Qwen3-Reranker's own default prompt uses when a caller
/// does not supply one of its own task-specific instructions.
pub const DEFAULT_INSTRUCTION: &str = "Given a web search query, retrieve relevant passages that \
answer the query";

/// One (query, document) pair's body text, in the exact field order and
/// labels the model card's own `format_instruction` uses. Everything else
/// (the fixed prefix/suffix) is shared across a whole batch.
pub fn format_pair(instruction: &str, query: &str, document: &str) -> String {
    format!("<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}")
}

/// A relevance score per (query, document) pair, in the same order the
/// caller's `bodies` arrived in.
///
/// `prefix_tokens`/`suffix_tokens` are tokenized once by the caller (see
/// `format_pair`'s own doc comment for why the boundary matters) and shared
/// across every pair in the batch; `bodies` is each pair's own
/// `format_pair`-built text, tokenized per pair. `token_true_id`/
/// `token_false_id` are the vocabulary ids for the bare words `"yes"`/`"no"`
/// (not a chat-template special token), looked up once by the caller.
pub fn rerank_batch(
    model: &mut Model,
    prefix_tokens: &[u32],
    bodies: &[Vec<u32>],
    suffix_tokens: &[u32],
    token_true_id: u32,
    token_false_id: u32,
) -> Result<Vec<f32>> {
    ensure!(!bodies.is_empty(), "rerank_batch needs at least one pair");
    ensure!(
        bodies.len() <= MAX_BATCH,
        "{} pairs exceeds the {MAX_BATCH}-pair batch cap",
        bodies.len()
    );

    let sequences: Vec<Vec<u32>> = bodies
        .iter()
        .enumerate()
        .map(|(i, body)| {
            ensure!(!body.is_empty(), "pair {i} tokenized to nothing");
            let mut seq = Vec::with_capacity(prefix_tokens.len() + body.len() + suffix_tokens.len());
            seq.extend_from_slice(prefix_tokens);
            seq.extend_from_slice(body);
            seq.extend_from_slice(suffix_tokens);
            Ok(seq)
        })
        .collect::<Result<_>>()?;

    let vocab_size = model.config().vocab_size;
    let batch_tokens = model.batch_tokens();
    let mut out: Vec<f32> = Vec::with_capacity(bodies.len());

    let mut start = 0usize;
    while start < sequences.len() {
        let mut end = start;
        let mut group_tokens = 0usize;
        while end < sequences.len() {
            let next_len = sequences[end].len();
            if end > start && group_tokens + next_len > batch_tokens {
                break;
            }
            group_tokens += next_len;
            end += 1;
        }
        rerank_group(
            model,
            &sequences[start..end],
            vocab_size,
            token_true_id,
            token_false_id,
            &mut out,
        )?;
        start = end;
    }

    debug_assert_eq!(out.len(), bodies.len());
    Ok(out)
}

fn rerank_group(
    model: &mut Model,
    group: &[Vec<u32>],
    vocab_size: usize,
    token_true_id: u32,
    token_false_id: u32,
    out: &mut Vec<f32>,
) -> Result<()> {
    let n_slots: usize = group.iter().map(Vec::len).sum();
    let mut pool = model
        .new_pool(n_slots, group.len())
        .context("allocating the rerank batch's throwaway KV pool")?;

    let seqs: Vec<_> = group
        .iter()
        .map(|_| pool.alloc().context("throwaway rerank pool ran out of sequence rows"))
        .collect::<Result<_>>()?;
    let items: Vec<BatchItem<'_>> = seqs
        .iter()
        .zip(group)
        .map(|(&seq, tokens)| BatchItem::new(seq, tokens, BatchItemKind::Prefill))
        .collect();

    let n_rows = model.forward_batch_device(&items, &mut pool)?;
    debug_assert_eq!(n_rows, group.len());
    let logits = model.logits_host()?;
    ensure!(
        logits.len() == group.len() * vocab_size,
        "logits_host returned {} floats for {} rows of width {vocab_size}",
        logits.len(),
        group.len()
    );

    for row in logits.chunks_exact(vocab_size) {
        let true_logit = row[token_true_id as usize] as f64;
        let false_logit = row[token_false_id as usize] as f64;
        // log-sum-exp rather than exp-then-divide: these are real model
        // logits, not bounded probabilities, and a large enough logit
        // overflows `f64::exp` into infinity long before it overflows this
        // subtraction.
        let m = true_logit.max(false_logit);
        let true_score = (true_logit - m).exp();
        let false_score = (false_logit - m).exp();
        out.push((true_score / (true_score + false_score)) as f32);
    }
    Ok(())
}
