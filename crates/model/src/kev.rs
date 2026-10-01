//! Kev-style decision-model scoring: a classification head riding a decoder
//! backbone, not text generation. Upstream: github.com/jaredpalmer/kev
//! (Apache-2.0), an open-weights alternative to TypeSafe's "Jev" decision
//! model, built on the same Qwen3.5/3.8 (GatedDeltaNet-hybrid) checkpoints
//! infero already runs.
//!
//! This ports Kev's real packing format (`kev/model.py::encode`, the row
//! form only -- see [`Encoded`]'s doc comment) and its pointer-head math
//! (`kev/model.py::PointerHead`). It does not port Kev's own CUDA-graph
//! serving internals (a hand-rolled "state bank" + bucketed graph replay):
//! the one thing that machinery exists for -- running the state once and
//! having every question continue from it -- is already a few lines on top
//! of infero's existing `KvPool::fork` (built for, and tested by, MTP's
//! speculative tree-draft verification; this is its second real caller).
//!
//! A question never goes through sampling or the vocab projection at all:
//! see [`Model::hidden_states_at`] for the readout this needed, the one
//! genuinely new engine capability this feature required.

use anyhow::{Context, Result, ensure};

use crate::{BatchItem, BatchItemKind, KvPool, Model, SeqId};

/// The five Qwen special tokens Kev repurposes as packing delimiters --
/// reused rather than added so no new embedding rows are needed (a LoRA
/// adapter, or for a full-weight checkpoint training itself, gives them
/// their packing meaning). Resolved once per loaded tokenizer by the caller
/// (`crates/server` owns the real `Tokenizer`; this crate does not depend on
/// it, the same "already-tokenized input" boundary `embed`/`rerank` use).
///
/// Kev's own names: `state` = `<|fim_prefix|>`, `q` = `<|fim_middle|>`,
/// `opt` = `<|box_start|>`, `opt_close` = `<|box_end|>`,
/// `decide` = `<|fim_suffix|>` (`kev/model.py`'s `SPECIAL` list, in that
/// order).
#[derive(Debug, Clone, Copy)]
pub struct SpecialTokens {
    pub state: u32,
    pub q: u32,
    pub opt: u32,
    pub opt_close: u32,
    pub decide: u32,
}

/// One question's already-tokenized pieces. Tokenizing free text is the
/// caller's job; this is the boundary.
pub struct QuestionInput {
    pub instr: Vec<u32>,
    pub options: Vec<Vec<u32>>,
}

/// One question's branch after packing: its own tokens only, meant to run as
/// a causal continuation of the shared state (see [`decide_batch`]) -- not
/// offset by the state's length, which is whatever sequence slot the
/// continuation actually lands in.
///
/// `kev/model.py`'s `rows_of()` returns exactly this shape (minus the state
/// tokens, which that function also returns but this module never packs
/// into one sequence with the branch -- see [`Encoded`]).
#[derive(Debug)]
pub struct Branch {
    pub ids: Vec<u32>,
    /// Index into `ids` of the final `<decide>` token (always `ids.len() - 1`
    /// by construction, kept explicit so a caller never has to assume it).
    pub decide_idx: usize,
    /// Index into `ids` of each option's closing `</opt>` token, in option
    /// order.
    pub opt_idx: Vec<usize>,
}

/// One packed record: the state's tokens, once, and every question's own
/// branch.
///
/// This is `kev/model.py`'s row form only -- the packed single-sequence
/// block-causal form (`branch_mask_batch`, `encode`'s `seg`/`pos`/`opt`
/// arrays) is never built, because that form exists for attention-only
/// backbones and every Kev checkpoint infero targets is the hybrid
/// (GatedDeltaNet) kind, which `kev.model.is_hybrid` says cannot honour a
/// packed mask at all -- `DecisionModel.rows_form` returns `True`
/// unconditionally for one. Building the packed form's bookkeeping for a
/// code path nothing here will ever take is exactly the kind of unvalidated
/// schema surface Phase 0 of this project's own auto-fusion investigation
/// flagged as a real risk; this module does not repeat it.
#[derive(Debug)]
pub struct Encoded {
    pub state_ids: Vec<u32>,
    pub branches: Vec<Branch>,
}

/// Why a record didn't fit. Mirrors `kev.model.ContextOverflow`'s two real
/// causes (not its exact wording -- this is a library type, not an HTTP
/// body; `crates/server`'s route handler is what renders a 422).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeError {
    /// The state alone (including its own `<state>` token) is over the
    /// limit.
    StateTooLong { tokens: usize, max: usize },
    /// One question's branch plus the (already-fitting) state is over the
    /// limit. `question` is its index in the input slice.
    BranchTooLong {
        question: usize,
        tokens: usize,
        max: usize,
    },
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncodeError::StateTooLong { tokens, max } => {
                write!(f, "state is {tokens} tokens, over the {max}-token limit")
            }
            EncodeError::BranchTooLong {
                question,
                tokens,
                max,
            } => write!(
                f,
                "question {question}'s row (state + its branch) is {tokens} tokens, over the {max}-token limit"
            ),
        }
    }
}

impl std::error::Error for EncodeError {}

/// Port of `kev.model.encode`'s row-building half: `[<state> state...]` then,
/// per question, `<q> instr <opt> opt0 </opt> <opt> opt1 </opt> ... <decide>`
/// (`kev/model.py`'s own docstring on `encode`, lines 87-100).
///
/// `max_state`/`max_branch` are `kev.model`'s own serving limits
/// (`SERVE_MAX_STATE`/`SERVE_MAX_BRANCH`) or training limits
/// (`MAX_STATE`/`MAX_BRANCH`), the caller's choice -- this function does not
/// pick a default, unlike Kev's own `encode()`, because infero has no single
/// "the" Kev checkpoint with one fixed context the way a hand-run script
/// does.
pub fn encode(
    special: &SpecialTokens,
    state: &[u32],
    questions: &[QuestionInput],
    max_state: usize,
    max_branch: usize,
) -> std::result::Result<Encoded, EncodeError> {
    let state_tokens = state.len() + 1; // the <state> token itself counts, same as kev's state_tokens
    if state_tokens > max_state {
        return Err(EncodeError::StateTooLong {
            tokens: state_tokens,
            max: max_state,
        });
    }
    let mut state_ids = Vec::with_capacity(state_tokens);
    state_ids.push(special.state);
    state_ids.extend_from_slice(state);

    let mut branches = Vec::with_capacity(questions.len());
    for (qi, q) in questions.iter().enumerate() {
        let mut ids = Vec::with_capacity(1 + q.instr.len() + q.options.iter().map(|o| o.len() + 2).sum::<usize>() + 1);
        ids.push(special.q);
        ids.extend_from_slice(&q.instr);
        let mut opt_idx = Vec::with_capacity(q.options.len());
        for opt in &q.options {
            ids.push(special.opt);
            ids.extend_from_slice(opt);
            ids.push(special.opt_close);
            opt_idx.push(ids.len() - 1);
        }
        ids.push(special.decide);
        let decide_idx = ids.len() - 1;

        let row_tokens = state_tokens + ids.len();
        if row_tokens > max_branch {
            return Err(EncodeError::BranchTooLong {
                question: qi,
                tokens: row_tokens,
                max: max_branch,
            });
        }
        branches.push(Branch {
            ids,
            decide_idx,
            opt_idx,
        });
    }
    Ok(Encoded {
        state_ids,
        branches,
    })
}

/// The pointer-readout head: scores each option's hidden state against the
/// question's `<decide>` hidden state. Port of `kev.model.PointerHead`
/// (`q`/`k`: `nn.Linear(d, dp)`; `forward`: `(k(h_opts) @ q(h_decide)) *
/// scale`, divided by `temperature` at inference).
///
/// Pure host-side f32 math, deliberately: `dp` (head capacity) is 256 on
/// every released Kev checkpoint and a question rarely has more than a
/// handful of options, so the two projections are a `[K, d] x [d, 256]`
/// matmul at most a few hundred K wide -- cheap enough on the CPU that a new
/// CUDA kernel (and the sanitizer/verification overhead every other kernel
/// in this codebase paid) buys nothing here. If a checkpoint with a much
/// larger option count or `dp` ever makes this a real cost, move it to the
/// GPU then, with a real measurement to justify it.
pub struct PointerHead {
    /// `[dp, d]`, row-major -- PyTorch `nn.Linear`'s own weight layout
    /// (`out_features x in_features`), so these are exactly `head.pt`'s
    /// `q.weight`/`k.weight` tensors, untransposed.
    pub q_weight: Vec<f32>,
    pub q_bias: Vec<f32>,
    pub k_weight: Vec<f32>,
    pub k_bias: Vec<f32>,
    pub d: usize,
    pub dp: usize,
    /// Calibration temperature fitted on held-out data
    /// (`scripts/calibrate_checkpoint.py` upstream); divides the logits
    /// before softmax. `1.0` for an uncalibrated checkpoint.
    pub temperature: f32,
}

impl PointerHead {
    pub fn new(q_weight: Vec<f32>, q_bias: Vec<f32>, k_weight: Vec<f32>, k_bias: Vec<f32>, d: usize, dp: usize, temperature: f32) -> Result<Self> {
        ensure!(q_weight.len() == dp * d, "q_weight is {} floats, expected {dp}*{d}", q_weight.len());
        ensure!(q_bias.len() == dp, "q_bias is {} floats, expected {dp}", q_bias.len());
        ensure!(k_weight.len() == dp * d, "k_weight is {} floats, expected {dp}*{d}", k_weight.len());
        ensure!(k_bias.len() == dp, "k_bias is {} floats, expected {dp}", k_bias.len());
        ensure!(temperature.is_finite() && temperature > 0.0, "a non-positive or non-finite temperature ({temperature}) would divide logits by zero or flip their sign");
        Ok(Self {
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            d,
            dp,
            temperature,
        })
    }

    fn project(&self, weight: &[f32], bias: &[f32], x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.d);
        let mut out = vec![0f32; self.dp];
        for (o, slot) in out.iter_mut().enumerate() {
            let row = &weight[o * self.d..(o + 1) * self.d];
            let mut acc = bias[o];
            for i in 0..self.d {
                acc += row[i] * x[i];
            }
            *slot = acc;
        }
        out
    }

    /// Probabilities over one question's options, given the `<decide>`
    /// hidden state and each option's `</opt>` hidden state (`d`-wide f32
    /// each, in option order). Port of `kev.model.PointerHead.forward` +
    /// the `F.softmax(z, -1)` its callers apply (`kev/model.py`'s `probs`,
    /// `_readout`).
    pub fn probs(&self, h_decide: &[f32], h_opts: &[&[f32]]) -> Result<Vec<f32>> {
        ensure!(!h_opts.is_empty(), "a question needs at least one option");
        ensure!(h_decide.len() == self.d, "h_decide is {} floats, expected {}", h_decide.len(), self.d);
        let scale = 1.0 / (self.dp as f32).sqrt();
        let q = self.project(&self.q_weight, &self.q_bias, h_decide);
        let mut logits = Vec::with_capacity(h_opts.len());
        for (i, h) in h_opts.iter().enumerate() {
            ensure!(h.len() == self.d, "option {i}'s hidden state is {} floats, expected {}", h.len(), self.d);
            let k = self.project(&self.k_weight, &self.k_bias, h);
            let dot: f32 = k.iter().zip(&q).map(|(a, b)| a * b).sum();
            logits.push((dot * scale) / self.temperature);
        }
        Ok(softmax(&logits))
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|x| x / sum).collect()
}

/// A loaded Kev checkpoint's own extra piece -- the backbone is loaded the
/// normal way (plain safetensors; the Rust side never merges a LoRA
/// adapter itself, see `docs/kev-decision-model.md`'s "what this needed"
/// section for why that's an offline step, not engine code).
pub struct KevHead {
    pub head: PointerHead,
    /// The base checkpoint this head was trained against
    /// (`head.pt`'s/`kev_head.json`'s `base`), carried through for a
    /// sanity check at load time, not acted on here.
    pub base: String,
}

impl KevHead {
    /// Loads `<dir>/kev_head.safetensors` + `<dir>/kev_head.json`, the
    /// sidecar `scripts/merge_kev_lora.py` (this repo's own offline
    /// conversion of Kev's `head.pt`, a raw torch.save pickle this crate
    /// does not parse) produces alongside a merged backbone.
    pub fn load(dir: impl AsRef<std::path::Path>, d_model: usize) -> Result<Self> {
        let dir = dir.as_ref();
        let file = infero_safetensors::File::open(dir.join("kev_head.safetensors"))
            .with_context(|| format!("opening {}/kev_head.safetensors", dir.display()))?;
        let meta_text = std::fs::read_to_string(dir.join("kev_head.json"))
            .with_context(|| format!("reading {}/kev_head.json", dir.display()))?;
        let meta: serde_json::Value = serde_json::from_str(&meta_text).context("parsing kev_head.json")?;
        let get = |name: &str| -> Result<Vec<f32>> {
            file.get(name)
                .with_context(|| format!("kev_head.safetensors has no tensor named {name}"))?
                .to_f32()
        };
        let head_dim = meta["head_dim"].as_u64().context("kev_head.json: head_dim")? as usize;
        let temperature = meta["temperature"].as_f64().context("kev_head.json: temperature")? as f32;
        let base = meta["base"].as_str().unwrap_or("unknown").to_string();
        let head = PointerHead::new(get("q.weight")?, get("q.bias")?, get("k.weight")?, get("k.bias")?, d_model, head_dim, temperature)?;
        Ok(Self { head, base })
    }
}

/// One question's score request: its branch (already encoded) and how many
/// options it has -- the caller already has `Branch`, this just groups what
/// [`decide_batch`] needs per question without re-deriving it.
pub struct ScoreQuestion<'a> {
    pub branch: &'a Branch,
}

/// Score every question of one record against one model, state computed
/// once.
///
/// The mechanism: prefill the state as an ordinary sequence, then
/// `KvPool::fork` it onto a fresh, empty sequence per question -- forking
/// shares the state's KV slots by reference and (for a GatedDeltaNet hybrid)
/// copies its small recurrent/conv state outright, exactly the cost
/// `GdnState::fork`'s own doc comment measures (about 0.6 ms for a
/// four-branch fork on infero's 27B). Every forked branch then runs as a
/// plain causal continuation **in one shared forward pass** -- no new
/// attention masking, no new batching concept, because a forked sequence is
/// architecturally indistinguishable from any other in-flight sequence once
/// its prefix is in place, and `forward_batch_device` already takes an
/// arbitrary multi-sequence `&[BatchItem]`.
///
/// This did NOT batch originally -- an earlier version forked and ran one
/// question at a time, in a loop. Real production traffic (Keel, 10-40
/// questions a record) measured that version's cost as ~25ms fixed +
/// ~14ms/question: the fixed cost of a `forward_batch_device` call (several
/// hundred kernel launches for one decoder pass) paid once *per question*
/// instead of once per record. Batching every forked branch of a group into
/// one call pays that fixed cost once per group instead.
///
/// Branches are grouped to fit `model.batch_tokens()` (same "greedily pack
/// until the token budget" idiom `embed_batch` uses), not assumed to fit in
/// one pass -- a record with many long branches still works, just over more
/// than one forward call.
///
/// Every branch's own tokens want a hidden-state readout at more than one
/// position (`<decide>` plus every `</opt>`), scattered through the branch
/// rather than trailing -- not what `wants_logits`/`forward_batch_rows`'s
/// `tail` (last N) can express, and not worth bending that shared,
/// capacity-bounded (`max_logit_rows`) machinery for. See
/// [`Model::hidden_states_at`]'s own doc comment for why this reads
/// `act.x` directly instead; that accessor is itself a flat multi-item
/// gather, so one call already covers a whole group's worth of picks.
///
/// Returns one `Vec<f32>` of probabilities per question, in `questions`'
/// order. The caller is responsible for freeing `state_seq` once every
/// question here (and anything else that forked from it) is done reading;
/// this function frees each question's own forked sequence itself before
/// returning.
pub fn decide_batch(
    model: &mut Model,
    pool: &mut KvPool,
    state_seq: SeqId,
    head: &PointerHead,
    questions: &[ScoreQuestion<'_>],
) -> Result<Vec<Vec<f32>>> {
    let dev = model.device().clone();
    let d = model.config().d_model;
    let batch_tokens = model.batch_tokens();
    let mut out: Vec<Option<Vec<f32>>> = (0..questions.len()).map(|_| None).collect();

    // Fork every branch up front -- cheap pool bookkeeping plus one small
    // GdnState copy each, no forward pass yet -- so a later group's fork
    // never has to wait on an earlier group's forward call.
    let mut branch_seqs = Vec::with_capacity(questions.len());
    for q in questions {
        let branch_seq = pool
            .alloc()
            .context("no free sequence slot for this question's forked branch")?;
        pool.fork(&dev, state_seq, branch_seq)
            .context("forking the shared state onto this question's branch")?;
        branch_seqs.push(branch_seq);
    }

    // Greedily group questions into passes of at most `batch_tokens` new
    // tokens total (a forked branch's own appended tokens; the state it
    // continues from doesn't count against this call's budget).
    let mut start = 0usize;
    while start < questions.len() {
        let mut end = start;
        let mut group_tokens = 0usize;
        while end < questions.len() {
            let next_len = questions[end].branch.ids.len();
            if end > start && group_tokens + next_len > batch_tokens {
                break;
            }
            group_tokens += next_len;
            end += 1;
        }

        let items: Vec<BatchItem<'_>> = (start..end)
            .map(|i| BatchItem::without_logits(branch_seqs[i], &questions[i].branch.ids, BatchItemKind::Prefill))
            .collect();
        model
            .forward_batch_device(&items, pool)
            .context("running this group of forked branches")?;

        // Absolute positions within this call's flat token layout: item i's
        // tokens start right after items start..i's, in the same order
        // `forward_batch_rows` itself lays them out (see its own `logit_rows`
        // construction).
        let mut picks = Vec::with_capacity(group_tokens + (end - start));
        let mut pick_counts = Vec::with_capacity(end - start);
        let mut base = 0i32;
        for i in start..end {
            let b = questions[i].branch;
            pick_counts.push(1 + b.opt_idx.len());
            picks.push(base + b.decide_idx as i32);
            picks.extend(b.opt_idx.iter().map(|&o| base + o as i32));
            base += b.ids.len() as i32;
        }
        let hidden = model
            .hidden_states_at(&picks, group_tokens)
            .context("reading back this group's <decide>/</opt> hidden states")?;

        let mut rows = hidden.chunks_exact(d);
        for (i, &n_picks) in (start..end).zip(&pick_counts) {
            let h_decide = rows.next().context("hidden_states_at returned fewer rows than expected")?;
            let h_opts: Vec<&[f32]> = (1..n_picks)
                .map(|_| rows.next().context("hidden_states_at returned fewer rows than expected"))
                .collect::<Result<_>>()?;
            ensure!(
                h_opts.len() == questions[i].branch.opt_idx.len(),
                "got {} option hidden states for {} options",
                h_opts.len(),
                questions[i].branch.opt_idx.len()
            );
            out[i] = Some(head.probs(h_decide, &h_opts)?);
        }

        start = end;
    }

    for seq in branch_seqs {
        pool.free(seq);
    }
    out.into_iter()
        .enumerate()
        .map(|(i, p)| p.with_context(|| format!("question {i} was never scored (internal grouping bug)")))
        .collect()
}

/// One record end to end: encode, run the state once, fork+score every
/// question, clean up. The one-shot-job convenience wrapper
/// [`embed_batch`](crate::embed::embed_batch)/[`rerank_batch`](crate::rerank::rerank_batch)
/// each are for their own endpoint -- this is `crates/server`'s real call
/// site, a throwaway `KvPool` sized for exactly this record (same reasoning
/// as `embed_group`'s: nothing here is meant to survive past this call,
/// unlike the server's real continuous-batching pool).
pub fn decide_record(
    model: &mut Model,
    special: &SpecialTokens,
    head: &PointerHead,
    state: &[u32],
    questions: &[QuestionInput],
    max_state: usize,
    max_branch: usize,
) -> Result<Vec<Vec<f32>>> {
    let enc = encode(special, state, questions, max_state, max_branch).map_err(|e| anyhow::anyhow!("{e}"))?;
    // `decide_batch` now forks and holds every branch at once (so a group's
    // forward pass can batch them together -- see its own doc comment), so
    // this pool has to size for all of them live simultaneously: the state's
    // own tokens once, plus every branch's own tokens (each branch owns its
    // own slots past the shared, borrowed state prefix).
    anyhow::ensure!(!enc.branches.is_empty(), "a decision record needs at least one question");
    let total_branch_tokens: usize = enc.branches.iter().map(|b| b.ids.len()).sum();
    let n_slots = enc.state_ids.len() + total_branch_tokens;
    let max_seqs = 1 + enc.branches.len();
    let mut pool = model
        .new_pool(n_slots, max_seqs)
        .context("allocating the decision record's throwaway KV pool")?;

    let state_seq = pool.alloc().context("no free sequence slot for the shared state")?;
    let item = BatchItem::without_logits(state_seq, &enc.state_ids, BatchItemKind::Prefill);
    model.forward_batch_device(std::slice::from_ref(&item), &mut pool).context("running the shared state")?;

    let score_questions: Vec<ScoreQuestion<'_>> = enc.branches.iter().map(|b| ScoreQuestion { branch: b }).collect();
    let probs = decide_batch(model, &mut pool, state_seq, head, &score_questions);
    pool.free(state_seq);
    probs
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPECIAL: SpecialTokens = SpecialTokens {
        state: 1000,
        q: 1001,
        opt: 1002,
        opt_close: 1003,
        decide: 1004,
    };

    #[test]
    fn encode_packs_state_once_and_every_branch_in_order() {
        let state = vec![10, 11, 12];
        let questions = vec![
            QuestionInput {
                instr: vec![20, 21],
                options: vec![vec![30], vec![31, 32]],
            },
            QuestionInput {
                instr: vec![40],
                options: vec![vec![50]],
            },
        ];
        let enc = encode(&SPECIAL, &state, &questions, 100, 100).unwrap();

        assert_eq!(enc.state_ids, vec![1000, 10, 11, 12]);
        assert_eq!(enc.branches.len(), 2);

        let b0 = &enc.branches[0];
        assert_eq!(b0.ids, vec![1001, 20, 21, 1002, 30, 1003, 1002, 31, 32, 1003, 1004]);
        assert_eq!(b0.decide_idx, b0.ids.len() - 1);
        assert_eq!(b0.ids[b0.decide_idx], 1004);
        // The two </opt> positions, in option order.
        assert_eq!(b0.opt_idx, vec![5, 9]);
        assert_eq!(b0.ids[5], 1003);
        assert_eq!(b0.ids[9], 1003);

        let b1 = &enc.branches[1];
        assert_eq!(b1.ids, vec![1001, 40, 1002, 50, 1003, 1004]);
        assert_eq!(b1.opt_idx, vec![4]);
        assert_eq!(b1.decide_idx, 5);
    }

    #[test]
    fn encode_rejects_a_state_over_the_limit() {
        let state = vec![1, 2, 3, 4, 5];
        let questions = vec![];
        // +1 for the <state> token itself.
        let err = encode(&SPECIAL, &state, &questions, 5, 100).unwrap_err();
        assert_eq!(
            err,
            EncodeError::StateTooLong {
                tokens: 6,
                max: 5
            }
        );
    }

    #[test]
    fn encode_rejects_a_branch_over_the_limit() {
        let state = vec![1, 2];
        let questions = vec![QuestionInput {
            instr: vec![1, 2, 3, 4, 5],
            options: vec![vec![1]],
        }];
        // state_ids = 3 tokens; this branch = q + 5 instr + (opt+1+close) + decide = 1+5+3+1 = 10; row = 13.
        let err = encode(&SPECIAL, &state, &questions, 100, 12).unwrap_err();
        assert_eq!(
            err,
            EncodeError::BranchTooLong {
                question: 0,
                tokens: 13,
                max: 12
            }
        );
    }

    #[test]
    fn encode_rejects_nothing_right_at_the_limit() {
        let state = vec![1, 2];
        let questions = vec![QuestionInput {
            instr: vec![1, 2, 3, 4, 5],
            options: vec![vec![1]],
        }];
        encode(&SPECIAL, &state, &questions, 100, 13).unwrap();
    }

    fn head_with_identity_like_weights(d: usize, dp: usize) -> PointerHead {
        // q/k project onto the first `dp` input dims unchanged (an
        // [dp, d] matrix that is the dp x dp identity in its first dp
        // columns and zero elsewhere), zero bias, temperature 1 -- so the
        // head's output is a plain, hand-checkable dot product of the
        // first `dp` coordinates of each input.
        let mut w = vec![0f32; dp * d];
        for i in 0..dp {
            w[i * d + i] = 1.0;
        }
        PointerHead::new(w.clone(), vec![0.0; dp], w, vec![0.0; dp], d, dp, 1.0).unwrap()
    }

    #[test]
    fn pointer_head_picks_the_closer_option_by_dot_product() {
        let head = head_with_identity_like_weights(4, 2);
        let decide = vec![1.0, 0.0, 9.0, 9.0]; // only the first 2 coords matter
        let close = vec![1.0, 0.0, -5.0, -5.0]; // dot = 1.0
        let far = vec![-1.0, 0.0, 5.0, 5.0]; // dot = -1.0
        let probs = head.probs(&decide, &[&far, &close]).unwrap();
        assert_eq!(probs.len(), 2);
        assert!(probs[1] > probs[0], "the closer option should win: {probs:?}");
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "probabilities must sum to 1: {probs:?}");
    }

    #[test]
    fn pointer_head_is_uniform_on_a_tie() {
        let head = head_with_identity_like_weights(3, 3);
        let decide = vec![1.0, 1.0, 1.0];
        let a = vec![2.0, 2.0, 2.0];
        let b = vec![2.0, 2.0, 2.0];
        let probs = head.probs(&decide, &[&a, &b]).unwrap();
        assert!((probs[0] - 0.5).abs() < 1e-6);
        assert!((probs[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn pointer_head_temperature_flattens_without_changing_the_argmax() {
        let hot = head_with_identity_like_weights(2, 2);
        let cool = PointerHead::new(hot.q_weight.clone(), hot.q_bias.clone(), hot.k_weight.clone(), hot.k_bias.clone(), 2, 2, 8.0).unwrap();

        let decide = vec![3.0, 0.0];
        let lo = vec![1.0, 0.0];
        let hi = vec![5.0, 0.0];
        let p_hot = hot.probs(&decide, &[&lo, &hi]).unwrap();
        let p_cool = cool.probs(&decide, &[&lo, &hi]).unwrap();

        assert!(p_hot[1] > p_hot[0]);
        assert!(p_cool[1] > p_cool[0], "temperature must not flip the argmax");
        assert!(
            (p_cool[1] - p_cool[0]).abs() < (p_hot[1] - p_hot[0]).abs(),
            "a higher temperature must flatten the distribution: hot {p_hot:?} cool {p_cool:?}"
        );
    }

    #[test]
    fn pointer_head_rejects_a_non_positive_temperature() {
        assert!(PointerHead::new(vec![0.0; 2], vec![0.0], vec![0.0; 2], vec![0.0], 2, 1, 0.0).is_err());
        assert!(PointerHead::new(vec![0.0; 2], vec![0.0], vec![0.0; 2], vec![0.0], 2, 1, -1.0).is_err());
    }

    #[test]
    fn pointer_head_rejects_mismatched_shapes() {
        assert!(PointerHead::new(vec![0.0; 3], vec![0.0; 2], vec![0.0; 4], vec![0.0; 2], 2, 2, 1.0).is_err());
    }
}
