//! Text embeddings: a batch of independent prefill-only passes that stop at
//! `output_norm`, using last-token pooling and a server-owned L2
//! normalization.
//!
//! Not a new forward-pass code path -- `forward_batch_device` already
//! computes exactly this vector for every item that sets `wants_logits`
//! (see [`Model::hidden_states_host`]'s own doc comment); this module is the
//! plumbing that builds a throwaway batch of one-off sequences, reads that
//! vector back instead of running `lm_head` on it, and normalizes it.
//! Written for Qwen3-Embedding-style dense Qwen3 decoders -- the same
//! architecture family (`crates/model/src/config.rs`'s `SUPPORTED` list)
//! infero already runs for generation.

use anyhow::{Context, Result, ensure};

use crate::{BatchItem, BatchItemKind, Model};

/// One embedding request's hard cap on how many texts it may carry.
///
/// Not a device limit -- `Model::batch_tokens` is what actually bounds one
/// forward pass, and a request under this cap can still need several passes
/// if its texts are long. This cap exists for the same reason the doc that
/// specified it gives: a caller that loops single texts through this
/// endpoint gets a clear rejection instead of quietly paying per-call
/// overhead 32-64 times over.
pub const MAX_BATCH: usize = 64;

/// L2-normalized, one row per input text, in the same order.
///
/// Takes already-tokenized input rather than a `Tokenizer` and raw text:
/// `infero-model` does not otherwise depend on `infero-tokenizer` (tokenizing
/// is the caller's job -- see `infero-server`'s chat-completions path), and
/// an embedding input is not a conversational turn, so the caller's own
/// plain (non-chat-template) encode is what belongs here, not a second
/// tokenizing convention owned by this crate. Texts are batched into as few
/// forward passes as `Model::batch_tokens` allows, each pass its own
/// throwaway `KvPool`: nothing here is meant to survive past this call,
/// unlike the server's real continuous-batching pool.
pub fn embed_batch(model: &mut Model, token_lists: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
    ensure!(!token_lists.is_empty(), "embed_batch needs at least one text");
    ensure!(
        token_lists.len() <= MAX_BATCH,
        "{} texts exceeds the {MAX_BATCH}-text batch cap",
        token_lists.len()
    );
    for (i, tokens) in token_lists.iter().enumerate() {
        ensure!(!tokens.is_empty(), "text {i} tokenized to nothing");
    }

    let d_model = model.config().d_model;
    let batch_tokens = model.batch_tokens();
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(token_lists.len());

    // Greedily group texts into passes that fit `batch_tokens`, splitting a
    // single text too wide for one pass into its own (necessarily undersized
    // by this grouping, but that is `forward_batch_rows`'s own limit to
    // enforce) group rather than silently truncating it.
    let mut start = 0usize;
    while start < token_lists.len() {
        let mut end = start;
        let mut group_tokens = 0usize;
        while end < token_lists.len() {
            let next_len = token_lists[end].len();
            if end > start && group_tokens + next_len > batch_tokens {
                break;
            }
            group_tokens += next_len;
            end += 1;
        }
        embed_group(model, &token_lists[start..end], d_model, &mut out)?;
        start = end;
    }

    debug_assert_eq!(out.len(), token_lists.len());
    Ok(out)
}

fn embed_group(
    model: &mut Model,
    group: &[Vec<u32>],
    d_model: usize,
    out: &mut Vec<Vec<f32>>,
) -> Result<()> {
    let n_slots: usize = group.iter().map(Vec::len).sum();
    let mut pool = model
        .new_pool(n_slots, group.len())
        .context("allocating the embedding batch's throwaway KV pool")?;

    let seqs: Vec<_> = group
        .iter()
        .map(|_| pool.alloc().context("throwaway embedding pool ran out of sequence rows"))
        .collect::<Result<_>>()?;
    let items: Vec<BatchItem<'_>> = seqs
        .iter()
        .zip(group)
        .map(|(&seq, tokens)| BatchItem::new(seq, tokens, BatchItemKind::Prefill))
        .collect();

    let n_rows = model.forward_batch_device(&items, &mut pool)?;
    debug_assert_eq!(n_rows, group.len());
    let hidden = model.hidden_states_host()?;
    ensure!(
        hidden.len() == group.len() * d_model,
        "hidden_states_host returned {} floats for {} rows of width {d_model}",
        hidden.len(),
        group.len()
    );

    for row in hidden.chunks_exact(d_model) {
        out.push(normalize(row)?);
    }
    Ok(())
}

/// L2-normalize one row, and check the result rather than trust the model's
/// own config to have already done it.
///
/// The real bug this guards against: BGE-M3's `modules.json` ships its own
/// `Normalize` module, so a caller-side `normalize: false` silently did
/// nothing -- the vectors were unit-length either way, from a guarantee that
/// lives in an upstream repo's config the caller never sees. Doing the
/// division here, and asserting its own result, means this endpoint's
/// normalization guarantee does not depend on whether the *next* model this
/// server loads happens to carry the same module.
fn normalize(row: &[f32]) -> Result<Vec<f32>> {
    let norm = (row.iter().map(|x| (*x as f64).powi(2)).sum::<f64>()).sqrt();
    ensure!(
        norm.is_finite() && norm > 1e-6,
        "embedding norm {norm} is degenerate -- refusing to divide by it \
         (a zero-ish vector is a bug to surface, not a result to return)"
    );
    let out: Vec<f32> = row.iter().map(|x| (*x as f64 / norm) as f32).collect();
    let check = (out.iter().map(|x| (*x as f64).powi(2)).sum::<f64>()).sqrt();
    ensure!(
        (check - 1.0).abs() < 1e-3,
        "post-normalize norm {check} is not 1.0 -- normalization bug, not float noise"
    );
    Ok(out)
}
