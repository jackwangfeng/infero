//! Speculative decoding: draft, verify, accept, and put back what was rejected.
//!
//! The design is `notes/qwen3.5-mtp.md`'s, and the part of it that needs care is
//! not the acceptance rule — that is fifteen lines in [`crate::qwen35_mtp`] — but
//! undoing a rejected token's effect on three different kinds of memory:
//!
//! | memory | how it is rolled back | cost |
//! |---|---|---|
//! | the KV cache | truncate the sequence; the next step overwrites | free |
//! | the conv window | restore the pre-step taps, re-run the accepted rows | 120 KiB a layer |
//! | the recurrent state | journal the update's inputs, replay the accepted prefix | 41 KiB a token a layer |
//!
//! The KV cache is easy because it is append-only: a verification pass writes
//! `k + 1` slots and dropping the sequence's length to `accepted + 1` leaves the
//! rest to be overwritten. The other two are in-place, and the two entries above
//! are the same idea at two prices.
//!
//! **Why journal-and-replay rather than a snapshot.** The recurrent state is 3
//! MiB per sequence per linear layer, 147 MiB over the 27B's 48 of them, and it
//! is updated in place — by the time the logits come back and say how many
//! candidates survived, all `k + 1` have been folded into the same numbers.
//! Copying it out before the step and back on rejection is 294 MiB of copies per
//! sequence per step against a step that is already bandwidth-bound reading 27
//! GB of weights; at a batch of 32 that is 9.4 GiB a step, a third again on top
//! of the work. vLLM's alternative is `k + 1` resident state slots per sequence,
//! 441 MiB at `k = 2`, 13.8 GiB at batch 32 — memory this engine does not have.
//! What this module does instead is record, per candidate token, the inputs the
//! recurrence consumed, and afterwards replay only the accepted prefix into the
//! persistent state. That is exact rather than approximate, and it is exact by
//! construction: the replay is the same kernel over the same inputs, so it walks
//! the same trajectory the unspeculated decode would have.
//!
//! The journal holds the recurrence's **inputs** — the post-convolution,
//! post-l2norm packed `[q | k | v]` row, plus `g` and `beta` — rather than the
//! notes' `(k_t, delta_t, g_t)` update terms. The two are the same size to
//! within a rounding error (41.3 KiB against 48.2 KiB a token a layer on the
//! 27B; the inputs carry `q` and `beta` and not `delta`) and the same exactness,
//! and the inputs need no new kernel to apply: replaying them *is*
//! `gdn_delta_rule`. Storing `delta` would need a commit kernel that does the
//! rank-one update with a given delta, which no kernel here does.
//!
//! **The one thing this cannot do yet.** For the replay to start from the
//! pre-step state, the verification pass must not write the persistent state at
//! all. `gdn_delta_rule` always stores what it computed. So the verification pass
//! runs the recurrence on a *working copy* of the layer's state
//! ([`GdnRollback::state_scratch`]), which costs one 3 MiB copy per layer per
//! step — half of the snapshot-restore the notes reject, and the commit is still
//! the journal replay rather than a copy back. Closing the gap is one flag on one
//! kernel; [`GdnRollback::KERNEL_WANTED`] says exactly which, and
//! `crates/model/tests/spec.rs` measures what it would save.

use anyhow::{Context, Result};
use tuili_gpu::{Buf, View};
use tuili_gpu::Device;
use tuili_kernels::gdn::SeqLayout;

use crate::config::LinearAttnConfig;
use crate::qwen35_mtp::{Accepted, accept_greedy};
use crate::{BatchItem, KvPool, Model, SeqId};

/// What the drafter needs to know about the tokens the target just confirmed.
///
/// The MTP head's slot `i` fuses the hidden state of token `t_i` with the
/// *embedding of `t_{i+1}`* and predicts `t_{i+2}`, so a draft round needs three
/// things lined up: which of the last pass's tokens are confirmed, what position
/// each of them had, and which token follows each. Getting the third one off by
/// one is the mistake that costs acceptance rate and nothing else — no crash, no
/// shape error — so the arithmetic lives here, next to the acceptance rule that
/// produced it, rather than in the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftFeed {
    /// Which of the last forward pass's tokens to read hidden states for.
    pub rows: std::ops::Range<usize>,
    /// The target position of each row's own token.
    pub positions: Vec<usize>,
    /// For each row, the token that follows it.
    pub shifted: Vec<u32>,
}

impl DraftFeed {
    /// The feed after a prefill: every prompt token, each followed by the next,
    /// and the last followed by the token the prefill's logits chose.
    ///
    /// This is vLLM's shift, `[a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3,
    /// next]`, with the freshly sampled token in the final slot. Priming the head
    /// over the whole prompt rather than only its last token is not an
    /// optimization: the drafter attends over its own cache, and a cache with
    /// holes in it is a different model.
    pub fn after_prefill(prompt: &[u32], pending: u32) -> Self {
        assert!(!prompt.is_empty(), "a prefill of no tokens");
        let mut shifted: Vec<u32> = prompt[1..].to_vec();
        shifted.push(pending);
        Self {
            rows: 0..prompt.len(),
            positions: (0..prompt.len()).collect(),
            shifted,
        }
    }
}

/// What one verification step produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecOutcome {
    /// The tokens to append to the sequence, in order. Between 1 and `k + 1` of
    /// them, and never zero — which is what keeps speculation from livelocking.
    pub tokens: Vec<u32>,
    /// How many of the drafts survived, in `0..=k`.
    pub accepted: usize,
    /// How many were offered, so that a mean acceptance length can be taken
    /// over steps without the caller tracking `k` separately.
    pub drafted: usize,
    /// What to hand the drafter for the next round.
    pub feed: DraftFeed,
}

impl SpecOutcome {
    /// Tokens emitted per verification step — the number the throughput gain is
    /// proportional to. `1 + accepted`, by construction.
    pub fn acceptance_length(&self) -> usize {
        self.tokens.len()
    }
}

/// The per-step journal that lets a rejected candidate be un-run.
///
/// One of these per model that has linear-attention blocks; `None` on a pure
/// attention model, where a verification pass has nothing in-place to undo.
pub struct GdnRollback {
    la: LinearAttnConfig,
    n_linear: usize,
    max_seqs: usize,
    /// Candidate tokens one step may carry, `k + 1`.
    cap: usize,

    /// `[n_linear, cap, conv_channels]` — the packed row **after** the causal
    /// convolution and **after** `q`/`k` were l2-normalized and `q` scaled.
    ///
    /// This capture point is the whole correctness of the replay, and the two
    /// wrong ones both run: journalling the pre-convolution row replays a
    /// recurrence over unfiltered inputs, and journalling between the
    /// convolution and the normalization replays it over unnormalized keys.
    /// `crates/model/tests/spec.rs` requires both to give a different state.
    qkv: Buf<f32>,
    /// `[n_linear, cap, value_heads]` each, as `gdn_gate_decay` produced them.
    g: Buf<f32>,
    beta: Buf<f32>,
    /// `[n_linear, cap, conv_channels]` — the **pre**-convolution row, which is
    /// what re-advancing the window over the accepted prefix needs.
    qkv_pre: Buf<f32>,
    /// `[n_linear, max_seqs, conv_channels * (conv_k - 1)]` — the convolution
    /// taps as they stood before the verification pass.
    ///
    /// This is the notes' widened conv window, stored as a separate buffer
    /// rather than as extra columns on the pool's. `conv_state_shape =
    /// (conv_dim, conv_kernel_size - 1 + num_spec)` needs `gdn_conv` to index a
    /// window wider than the kernel it convolves with; keeping the pre-step taps
    /// and the candidates' inputs holds exactly the same history — the union is
    /// the wide window — and needs no kernel change. 120 KiB a layer a sequence
    /// against the widened window's 200 KiB at `k = 2`.
    conv_pre: Buf<f32>,
    /// A working copy of one layer's recurrent state, so that the verification
    /// pass leaves the persistent one alone. See the module note.
    state_scratch: Buf<f32>,
    /// Somewhere for the replay's outputs to go. The replay is run for its
    /// effect on the state; the readouts were already computed by the pass being
    /// replayed and are thrown away here.
    out_scratch: Buf<f32>,
    conv_out_scratch: Buf<f32>,

    /// Set for the duration of a verification pass.
    armed: bool,
    /// Rows the armed pass carried.
    rows: usize,
    /// Which sequence slot the armed pass is for.
    slot: usize,
    /// Copies *issued from the host*, for the report — this is the cost the
    /// kernel change below would remove.
    ///
    /// Not the number the device performed: a captured CUDA graph replays these
    /// copies without running this code, so after the second pass at a given
    /// shape the counter stops moving while the copies keep happening. It answers
    /// "did the journal arm" and "what shape is the staging", not "how many
    /// bytes moved this step".
    state_copies: u64,
}

impl GdnRollback {
    /// The kernel change that would make this free, stated precisely enough to
    /// implement without reading this file.
    ///
    /// > `gdn_delta_rule_f32` and `gdn_delta_rule_reg128_f32` in
    /// > `crates/kernels/src/cu/gdn.cu` take one `float* state`, which they both
    /// > read and write. Give them a second pointer, `float* state_out`, and
    /// > write the final state there instead: `state_out == state` reproduces
    /// > today's behaviour, and `state_out == nullptr` computes the readouts from
    /// > `state` and stores nothing. The register variant already loads S on
    /// > entry and stores it once on exit, so this is one guarded store; the
    /// > reference variant writes S every token and would need its two passes to
    /// > accumulate into a scratch instead, or simply not to be the one used for
    /// > a verification pass.
    ///
    /// With that, a verification step reads the persistent state, computes the
    /// candidates' outputs, and leaves it untouched, and this journal's replay
    /// commits exactly the accepted prefix. The 3 MiB-a-layer working copy below
    /// disappears; nothing else here changes.
    pub const KERNEL_WANTED: &'static str =
        "gdn_delta_rule with a separate state_out pointer (nullptr = do not store)";

    pub fn new(
        dev: &Device,
        la: LinearAttnConfig,
        layer_is_linear: &[bool],
        max_seqs: usize,
        cap: usize,
    ) -> Result<Self> {
        anyhow::ensure!(cap > 0, "a journal with room for no candidates");
        let n_linear = layer_is_linear.iter().filter(|l| **l).count();
        let stream = dev.stream();
        let width = la.conv_channels();
        let heads = la.value_heads;
        let state = la.value_heads * la.key_head_dim * la.value_head_dim;
        let conv = width * (la.conv_kernel - 1);
        let this = Self {
            la,
            n_linear,
            max_seqs,
            cap,
            qkv: stream.alloc_zeros::<f32>(n_linear * cap * width)?,
            g: stream.alloc_zeros::<f32>(n_linear * cap * heads)?,
            beta: stream.alloc_zeros::<f32>(n_linear * cap * heads)?,
            qkv_pre: stream.alloc_zeros::<f32>(n_linear * cap * width)?,
            conv_pre: stream.alloc_zeros::<f32>(n_linear * max_seqs * conv)?,
            state_scratch: stream.alloc_zeros::<f32>(max_seqs * state)?,
            out_scratch: stream.alloc_zeros::<f32>(cap * la.value_dim())?,
            conv_out_scratch: stream.alloc_zeros::<f32>(cap * width)?,
            armed: false,
            rows: 0,
            slot: 0,
            state_copies: 0,
        };
        tracing::info!(
            linear_layers = n_linear,
            candidates = cap,
            journal_kib = (this.journal_bytes()) >> 10,
            working_state_kib = (this.state_scratch.len() * 4) >> 10,
            "speculative rollback journal allocated"
        );
        Ok(this)
    }

    /// What the journal itself costs, excluding the working state copy.
    pub fn journal_bytes(&self) -> usize {
        (self.qkv.len() + self.g.len() + self.beta.len() + self.qkv_pre.len() + self.conv_pre.len())
            * 4
    }

    /// 3 MiB-a-layer copies issued so far. Zero once the kernel above exists.
    pub fn state_copies(&self) -> u64 {
        self.state_copies
    }

    pub fn is_armed(&self) -> bool {
        self.armed
    }

    /// One layer's state, for a caller checking what a pass left behind.
    pub fn working_state(&self) -> View<'_, f32> {
        self.state_scratch.as_view()
    }

    /// Begin recording a verification pass over `rows` candidates.
    ///
    /// [`Model::verify_draft`] does this; it is public so that
    /// `crates/model/tests/gdn_rollback.rs` can drive the same code against the
    /// kernels directly. A test that reimplemented the replay would be checking
    /// its own copy of the algorithm, which is the failure this codebase has
    /// already paid for once.
    pub fn arm(&mut self, slot: usize, rows: usize) -> Result<()> {
        anyhow::ensure!(
            rows <= self.cap,
            "a verification pass of {rows} candidates against a journal built \
             for {}",
            self.cap
        );
        anyhow::ensure!(slot < self.max_seqs, "sequence slot {slot} is past the pool");
        self.armed = true;
        self.rows = rows;
        self.slot = slot;
        Ok(())
    }

    pub fn disarm(&mut self) {
        self.armed = false;
    }

    fn conv_span(&self, ordinal: usize) -> std::ops::Range<usize> {
        let n = self.la.conv_channels() * (self.la.conv_kernel - 1) * self.max_seqs;
        ordinal * n..(ordinal + 1) * n
    }

    fn row_span(&self, ordinal: usize, width: usize, rows: usize) -> std::ops::Range<usize> {
        let base = ordinal * self.cap * width;
        base..base + rows * width
    }
}

/// The buffers `linear_attention` hands the journal, named so the capture point
/// is legible at the call site.
pub struct GdnTap<'a> {
    /// The input projection's packed row, before the convolution.
    pub pre_conv: View<'a, f32>,
    /// The same row after the convolution *and* after `q`/`k` were
    /// l2-normalized — what the recurrence actually consumes.
    pub post_conv: View<'a, f32>,
    pub g: View<'a, f32>,
    pub beta: View<'a, f32>,
}

impl GdnRollback {
    /// Copy this layer's convolution taps out before anything overwrites them.
    pub fn save_conv(
        &mut self,
        dev: &Device,
        ordinal: usize,
        conv: &View<'_, f32>,
    ) -> Result<()> {
        let span = self.conv_span(ordinal);
        anyhow::ensure!(
            conv.len() == span.len(),
            "layer {ordinal}'s conv window is {} floats, the journal reserves {}",
            conv.len(),
            span.len()
        );
        dev.stream()
            .memcpy_dtod(conv, &mut self.conv_pre.slice_mut(span))?;
        Ok(())
    }

    /// Copy this layer's recurrent state into the working buffer the pass will
    /// advance, leaving the persistent one at its pre-step value.
    pub fn stage_state(
        &mut self,
        dev: &Device,
        state: &View<'_, f32>,
    ) -> Result<()> {
        anyhow::ensure!(
            state.len() == self.state_scratch.len(),
            "a layer's state is {} floats, the working copy holds {}",
            state.len(),
            self.state_scratch.len()
        );
        dev.stream()
            .memcpy_dtod(state, &mut self.state_scratch.as_view_mut())?;
        self.state_copies += 1;
        Ok(())
    }

    pub fn state_scratch_mut(&mut self) -> tuili_gpu::ViewMut<'_, f32> {
        self.state_scratch.as_view_mut()
    }

    /// Put one layer's convolution window and recurrent state where an
    /// unspeculated decode of the first `keep` candidates would have left them.
    ///
    /// Both halves are a replay rather than a restore, and both use the kernel
    /// the forward pass used:
    ///
    /// * the convolution window is rewound to its pre-step taps and then walked
    ///   forward over the accepted rows' *pre*-convolution inputs, which is the
    ///   notes' widened window with the extra columns kept somewhere else;
    /// * the recurrence starts from the persistent state — which the pass left
    ///   alone, having run on the working copy — and consumes the accepted rows'
    ///   journalled inputs in order.
    ///
    /// The outputs of both go to scratch. They were already computed correctly by
    /// the pass being replayed, for exactly these tokens, from exactly this
    /// state; recomputing them is the price of not having a commit kernel, and it
    /// is `keep` tokens of arithmetic against a step that just read the whole
    /// model.
    #[allow(clippy::too_many_arguments)]
    pub fn replay_layer(
        &mut self,
        dev: &Device,
        kern: &tuili_kernels::Kernels,
        ordinal: usize,
        keep: usize,
        seqs: &SeqLayout<'_>,
        conv_w: &View<'_, f32>,
        state: &mut tuili_gpu::ViewMut<'_, f32>,
        conv: &mut tuili_gpu::ViewMut<'_, f32>,
    ) -> Result<()> {
        anyhow::ensure!(keep <= self.rows, "keeping {keep} of {} rows", self.rows);
        let la = self.la;
        let width = la.conv_channels();
        let heads = la.value_heads;
        let span = self.conv_span(ordinal);
        dev.stream().memcpy_dtod(&self.conv_pre.slice(span), conv)?;
        if keep == 0 {
            return Ok(());
        }
        let rows = self.row_span(ordinal, width, keep);
        kern.gdn_conv(
            &mut self.conv_out_scratch.slice_mut(..keep * width),
            &self.qkv_pre.slice(rows.clone()),
            conv,
            conv_w,
            seqs,
            width,
            la.conv_kernel,
        )?;
        let gr = self.row_span(ordinal, heads, keep);
        kern.gdn_delta_rule(
            &mut self.out_scratch.slice_mut(..keep * la.value_dim()),
            state,
            &self.qkv.slice(rows),
            &self.g.slice(gr.clone()),
            &self.beta.slice(gr),
            seqs,
            heads,
            la.key_heads,
            la.key_head_dim,
            la.value_head_dim,
            (width, 0, la.key_dim(), 2 * la.key_dim()),
                    la.v_heads_tiled,
        )?;
        Ok(())
    }

    /// Record one layer's candidate rows.
    pub fn record(&mut self, dev: &Device, ordinal: usize, tap: GdnTap<'_>) -> Result<()> {
        let rows = self.rows;
        let width = self.la.conv_channels();
        let heads = self.la.value_heads;
        // The tap has to be exactly the armed pass's rows. A shorter one would
        // leave the tail of the journal holding the previous step's numbers,
        // which replays a state that never existed; a longer one would be
        // truncated by the copy. Neither is visible downstream.
        anyhow::ensure!(
            tap.pre_conv.len() == rows * width
                && tap.post_conv.len() == rows * width
                && tap.g.len() == rows * heads
                && tap.beta.len() == rows * heads,
            "layer {ordinal}: the tap is {}/{} row-widths and {}/{} gates against \
             an armed pass of {rows} rows",
            tap.pre_conv.len(),
            tap.post_conv.len(),
            tap.g.len(),
            tap.beta.len(),
        );
        let stream = dev.stream();
        stream.memcpy_dtod(
            &tap.pre_conv,
            &mut self.qkv_pre.slice_mut(self.row_span(ordinal, width, rows)),
        )?;
        stream.memcpy_dtod(
            &tap.post_conv,
            &mut self.qkv.slice_mut(self.row_span(ordinal, width, rows)),
        )?;
        stream.memcpy_dtod(
            &tap.g,
            &mut self.g.slice_mut(self.row_span(ordinal, heads, rows)),
        )?;
        stream.memcpy_dtod(
            &tap.beta,
            &mut self.beta.slice_mut(self.row_span(ordinal, heads, rows)),
        )?;
        Ok(())
    }
}

impl Model {
    /// Make room to undo a rejected candidate, for a model that needs it.
    ///
    /// Only the recurrent state and the convolution window need anything: the KV
    /// cache rolls back for free and the drafter's own cache is overwritten in
    /// place. So on a pure attention model this allocates nothing and
    /// [`Model::verify_draft`] works either way — which is what lets the
    /// greedy-equivalence test run on a model small enough to iterate on.
    pub fn enable_speculation(&mut self, k: usize, pool: &KvPool) -> Result<()> {
        anyhow::ensure!(k > 0, "speculating no tokens");
        anyhow::ensure!(
            k + 1 <= self.max_logit_rows,
            "verifying {k} drafts needs {} logit rows and the model was built \
             for {}",
            k + 1,
            self.max_logit_rows
        );
        self.gdn_rollback = match self.cfg.linear_attn {
            Some(la) => Some(GdnRollback::new(
                &self.dev,
                la,
                &self.layer_kinds,
                pool.max_seqs(),
                k + 1,
            )?),
            None => None,
        };
        Ok(())
    }

    /// The journal, for a caller reporting what rollback cost.
    pub fn gdn_rollback(&self) -> Option<&GdnRollback> {
        self.gdn_rollback.as_ref()
    }

    /// Drop the journal.
    ///
    /// [`Model::verify_draft`] still works without it, and on a pure attention
    /// model that is the whole story. On a model with recurrent state it is
    /// *wrong* — the rejected candidates stay folded into the state — and it
    /// exists so that `tests/spec_gdn.rs` can show the journal is what makes the
    /// output match, rather than asserting it.
    pub fn disable_speculation(&mut self) {
        self.gdn_rollback = None;
    }

    /// Verify `k` drafted tokens after `pending`, accept a prefix, and undo the
    /// rest.
    ///
    /// `pending` is the token the previous step settled on and which has not been
    /// through the model yet; `draft` is the proposal for what follows it. One
    /// forward pass over `1 + draft.len()` tokens produces one prediction per
    /// candidate plus the bonus, and the greedy rule takes the longest prefix
    /// that the target model would have produced on its own — so the tokens this
    /// returns are, token for token, what unspeculated greedy decoding would
    /// have returned. That property is the point: speculation becomes a pure
    /// throughput change, and a bad drafter costs speed and never quality.
    ///
    /// Afterwards the sequence holds `accepted + 1` new tokens: the accepted
    /// drafts plus `pending` itself. The last token in [`SpecOutcome::tokens`] is
    /// the next step's `pending` and is deliberately *not* in the cache.
    pub fn verify_draft(
        &mut self,
        seq: SeqId,
        pool: &mut KvPool,
        pending: u32,
        draft: &[u32],
    ) -> Result<SpecOutcome> {
        let n = draft.len() + 1;
        anyhow::ensure!(
            n <= self.max_logit_rows,
            "verifying {} candidates needs {n} logit rows, the model was built \
             for {}",
            draft.len(),
            self.max_logit_rows
        );
        let len_before = pool.len(seq);
        let mut candidates = Vec::with_capacity(n);
        candidates.push(pending);
        candidates.extend_from_slice(draft);

        if let Some(r) = self.gdn_rollback.as_mut() {
            r.arm(seq.0, n)?;
        }
        let rows = {
            let item = BatchItem::new(seq, &candidates);
            // Every candidate's logits, not just the last: `logits[j]` is the
            // target's own prediction for the token after candidate `j`, and the
            // acceptance rule reads all of them.
            let r = self.forward_batch_rows(std::slice::from_ref(&item), pool, &[n]);
            // A failed pass leaves the journal armed, and the next step would
            // then commit rows it never recorded.
            if let (true, Some(j)) = (r.is_err(), self.gdn_rollback.as_mut()) {
                j.disarm();
            }
            r?
        };
        anyhow::ensure!(rows == n, "asked for {n} logit rows and got {rows}");

        let vocab = self.cfg.vocab_size;
        let logits = self.logits_host()?;
        let target_argmax: Vec<u32> = (0..n)
            .map(|j| crate::mtp::argmax(&logits[j * vocab..(j + 1) * vocab]))
            .collect();
        let accepted = accept_greedy(draft, &target_argmax);

        self.settle(seq, pool, len_before, &accepted)?;
        // The next draft round's input. `keep = accepted + 1` rows of this pass
        // survive — the token the previous step settled on, plus the drafts that
        // matched — and the token following the last of them is the one this step
        // just emitted, which is not in the cache yet.
        let keep = accepted.accepted + 1;
        let mut shifted: Vec<u32> = candidates[1..keep].to_vec();
        shifted.push(*accepted.tokens.last().expect("a step emits at least one"));
        let feed = DraftFeed {
            rows: 0..keep,
            positions: (len_before..len_before + keep).collect(),
            shifted,
        };
        debug_assert_eq!(feed.shifted.len(), feed.positions.len());
        Ok(SpecOutcome {
            tokens: accepted.tokens,
            accepted: accepted.accepted,
            drafted: draft.len(),
            feed,
        })
    }

    /// Verify a sampled draft, preserving the request's own distribution.
    ///
    /// The rule is rejection sampling: accept draft `j` with probability
    /// `min(1, p_target / p_draft)`, and on the first rejection emit a token
    /// drawn from the residual `(p_target - p_draft)+`. That is what makes
    /// speculation a pure speedup rather than a different sampler — the output
    /// distribution is the target's, exactly, whatever the drafter does.
    ///
    /// Both probabilities come from [`crate::Sampler::distribution`], so the
    /// transformation the request asked for is applied once, in one place. The
    /// history matters and grows: position `j`'s distribution is conditioned on
    /// the tokens before it, drafts included, and the repetition penalty reads
    /// that window. Scoring every position against the prompt's window would
    /// silently mis-measure both sides.
    pub fn verify_draft_sampled(
        &mut self,
        seq: SeqId,
        pool: &mut KvPool,
        pending: u32,
        draft: &[crate::mtp::Drafted],
        sampler: &mut crate::Sampler,
        history: &[u32],
    ) -> Result<SpecOutcome> {
        let n = draft.len() + 1;
        anyhow::ensure!(
            n <= self.max_logit_rows,
            "verifying {} candidates needs {n} logit rows, the model was built \
             for {}",
            draft.len(),
            self.max_logit_rows
        );
        anyhow::ensure!(
            !sampler.params().is_greedy(),
            "a greedy request takes `verify_draft`, whose acceptance rule is \
             exact rather than a ratio"
        );
        let len_before = pool.len(seq);
        let mut candidates = Vec::with_capacity(n);
        candidates.push(pending);
        candidates.extend(draft.iter().map(|d| d.token));

        if let Some(r) = self.gdn_rollback.as_mut() {
            r.arm(seq.0, n)?;
        }
        let rows = {
            let item = BatchItem::new(seq, &candidates);
            let r = self.forward_batch_rows(std::slice::from_ref(&item), pool, &[n]);
            if let (true, Some(j)) = (r.is_err(), self.gdn_rollback.as_mut()) {
                j.disarm();
            }
            r?
        };
        anyhow::ensure!(rows == n, "asked for {n} logit rows and got {rows}");

        let vocab = self.cfg.vocab_size;
        // Copied out because building each row's distribution borrows the
        // sampler mutably while the logits live in `self`.
        // Walk the draft, extending the window as tokens are accepted.
        let mut window: Vec<u32> = history.to_vec();
        window.push(pending);

        // Every row's penalty window is known before the loop runs, because row
        // `j`'s distribution is only ever consulted when rows `0..j` were all
        // accepted — so its window is the history plus the first `j` drafted
        // tokens, whatever the acceptance test decides. That is what lets all
        // `n` distributions come from one device call.
        let mut win_owned: Vec<Vec<u32>> = Vec::with_capacity(n);
        {
            let mut w = window.clone();
            for j in 0..n {
                win_owned.push(w.clone());
                if let Some(d) = draft.get(j) {
                    w.push(d.token);
                }
            }
        }
        let sp = sampler.params().clone();
        let row_specs: Vec<crate::RowSample> = (0..n)
            .map(|_| crate::RowSample {
                temperature: sp.temperature,
                top_p: sp.top_p,
                top_k: sp.top_k as u32,
                rep_penalty: sp.repetition_penalty,
                // Unused: every draw this function needs comes from `sampler`,
                // in the order the host path took them, so a seed reproduces the
                // same stream. The device only supplies the distributions.
                rnd: 0.0,
            })
            .collect();
        let win_refs: Vec<&[u32]> = win_owned.iter().map(|w| w.as_slice()).collect();
        let device_dists = self.survivors_on_device(&row_specs, &win_refs)?;

        // The host fallback keeps the whole vocabulary; the device path returns
        // the nucleus, which is what both the acceptance test and the residual
        // are over. 2.98 MB and three full-vocabulary passes against a kilobyte.
        let logits: Vec<f32> = if device_dists.is_some() {
            Vec::new()
        } else {
            self.logits_host()?.to_vec()
        };
        let mut tokens: Vec<u32> = Vec::with_capacity(n);
        let mut accepted = 0usize;
        for (j, d) in draft.iter().enumerate() {
            let token = d.token;
            // Both draws come out before the distribution borrows the sampler.
            // Taking them unconditionally also keeps the generator's sequence
            // independent of which branch runs, so a seed reproduces the same
            // stream whether a draft was accepted or not.
            let draw = sampler.next_draw();
            let residual_draw = sampler.next_draw();
            // Normalized already on the device path, so the normalizer is one —
            // `pick` and `draw_residual` both take weights and a total, and
            // `w / 1.0` is `w`.
            let (dist, total): (&[(u32, f32)], f64) = match &device_dists {
                Some(d) => (&d[j], 1.0),
                None => {
                    let row = &logits[j * vocab..(j + 1) * vocab];
                    let (dd, tt) = sampler.distribution(row, &window);
                    (dd, tt)
                }
            };
            let p_target = dist
                .iter()
                .find(|(t, _)| *t == token)
                .map(|(_, w)| *w as f64 / total)
                .unwrap_or(0.0);
            let p_draft = d
                .q
                .iter()
                .find(|(t, _)| *t == token)
                .map(|(_, w)| *w)
                .unwrap_or(0.0);
            // A drafted token outside the target's truncated support has
            // probability zero there and is always rejected — which is correct
            // and is also how top-k and top-p keep speculation from smuggling
            // in tokens the request excluded.
            if p_draft > 0.0 && p_target / p_draft as f64 >= draw {
                tokens.push(token);
                window.push(token);
                accepted += 1;
                continue;
            }
            // Rejected: draw from the residual (p_target - p_draft)+, which is
            // what makes the composition exact. The draft's mass sits entirely
            // on `token`, so the residual is the target with that one entry
            // reduced.
            let recovered = Self::draw_residual(dist, total, &d.q, residual_draw);
            tokens.push(recovered);
            break;
        }
        if accepted == draft.len() {
            // Every draft survived, so the target's own extra row is a free
            // token — the bonus that makes k accepted drafts worth k + 1.
            let draw = sampler.next_draw();
            let (dist, total): (&[(u32, f32)], f64) = match &device_dists {
                Some(d) => (&d[draft.len()], 1.0),
                None => {
                    let row = &logits[draft.len() * vocab..(draft.len() + 1) * vocab];
                    let (dd, tt) = sampler.distribution(row, &window);
                    (dd, tt)
                }
            };
            tokens.push(crate::Sampler::pick(dist, total, draw));
        }
        debug_assert!(!tokens.is_empty(), "a step has to emit something");

        let acc = crate::qwen35_mtp::Accepted {
            tokens: tokens.clone(),
            accepted,
        };
        self.settle(seq, pool, len_before, &acc)?;
        let keep = accepted + 1;
        let mut shifted: Vec<u32> = candidates[1..keep].to_vec();
        shifted.push(*tokens.last().expect("a step emits at least one"));
        let feed = DraftFeed {
            rows: 0..keep,
            positions: (len_before..len_before + keep).collect(),
            shifted,
        };
        Ok(SpecOutcome {
            tokens,
            accepted,
            drafted: draft.len(),
            feed,
        })
    }

    /// Sample from `(p_target - q)+`, normalized, where `q` is the drafter's
    /// whole distribution.
    ///
    /// Public so a test can drive the rule this engine actually uses rather than
    /// a transcription of it. A test that reimplements the rule and agrees with
    /// itself is the failure mode that cost this project a day already.
    pub fn draw_residual(
        dist: &[(u32, f32)],
        total: f64,
        q: &[(u32, f32)],
        draw: f64,
    ) -> u32 {
        match Self::residual(dist, total, q) {
            Some((residual, sum)) => crate::Sampler::pick(&residual, sum, draw),
            // The target's support is entirely covered by the draft's, which can
            // only happen when the two agree completely. Any token of that
            // shared support is a correct draw; the first is the most likely.
            None => dist.first().map(|(t, _)| *t).unwrap_or(0),
        }
    }

    /// `(p - q)+` over the target's support, unnormalized, with its own total.
    ///
    /// `None` when nothing is left, which needs `p`'s support to sit entirely
    /// inside `q`'s with no more mass anywhere — only reachable when the two
    /// distributions agree.
    ///
    /// Split out from [`Self::draw_residual`] because multi-candidate acceptance
    /// needs the residual as a *distribution* to test the next candidate
    /// against, not a draw from it.
    pub fn residual(
        dist: &[(u32, f32)],
        total: f64,
        q: &[(u32, f32)],
    ) -> Option<(Vec<(u32, f32)>, f64)> {
        let q_of = |tok: u32| {
            q.iter()
                .find(|(t, _)| *t == tok)
                .map(|(_, w)| *w as f64)
                .unwrap_or(0.0)
        };
        let mut residual: Vec<(u32, f32)> = Vec::with_capacity(dist.len());
        let mut sum = 0.0f64;
        for &(t, w) in dist {
            // `q` is subtracted at every token, not only the drafted one. See
            // `Drafted::q`.
            let r = w as f64 / total - q_of(t);
            if r > 0.0 {
                residual.push((t, r as f32));
                sum += r;
            }
        }
        if residual.is_empty() || sum <= 0.0 {
            return None;
        }
        Some((residual, sum))
    }

    /// Accept one of several candidates for the same position, or draw from what
    /// is left of the target.
    ///
    /// The one-candidate rule with the residual carried forward. Reject `x_i` and
    /// the target becomes `norm((p - q)+)`; the next candidate is tested against
    /// *that*, not against the original `p`. With the candidates drawn i.i.d.
    /// from `q`, the composition is exactly `p` — the same argument the
    /// one-candidate case rests on, applied down the list.
    ///
    /// This is what a tree draft needs and what "take whichever branch accepted"
    /// is not: picking the best of several branches conditions on the outcome and
    /// biases what comes out, however natural it looks.
    ///
    /// `draws` supplies one uniform per candidate and `final_draw` the fallback.
    /// Taking them all up front keeps the generator's sequence independent of how
    /// many candidates were tested, so a seed reproduces the same stream.
    ///
    /// Returns which candidate was accepted, if any, and the token to emit.
    pub fn accept_multi(
        p: &[(u32, f32)],
        p_total: f64,
        q: &[(u32, f32)],
        candidates: &[u32],
        draws: &[f64],
        final_draw: f64,
    ) -> (Option<usize>, u32) {
        debug_assert!(draws.len() >= candidates.len());
        // Normalized once, so the loop below can keep replacing it with a
        // residual that is also normalized-with-a-total.
        let mut cur: Vec<(u32, f32)> = p
            .iter()
            .map(|(t, w)| (*t, (*w as f64 / p_total) as f32))
            .collect();
        let mut cur_total = 1.0f64;
        for (i, &x) in candidates.iter().enumerate() {
            let q_x = q
                .iter()
                .find(|(t, _)| *t == x)
                .map(|(_, w)| *w as f64)
                .unwrap_or(0.0);
            let p_x = cur
                .iter()
                .find(|(t, _)| *t == x)
                .map(|(_, w)| *w as f64 / cur_total)
                .unwrap_or(0.0);
            // A candidate outside the target's remaining support has probability
            // zero there and is always rejected, which is how top-k and top-p
            // keep speculation from smuggling in tokens the request excluded.
            if q_x > 0.0 && p_x / q_x >= draws[i] {
                return (Some(i), x);
            }
            match Self::residual(&cur, cur_total, q) {
                Some((r, t)) => {
                    cur = r;
                    cur_total = t;
                }
                // Nothing left to subtract from: the remaining target and the
                // draft agree, so any token of the shared support is a correct
                // draw and there is no point testing further candidates.
                None => {
                    return (None, cur.first().map(|(t, _)| *t).unwrap_or(0));
                }
            }
        }
        (None, crate::Sampler::pick(&cur, cur_total, final_draw))
    }

    /// Roll every kind of per-sequence memory back to the accepted prefix.
    fn settle(
        &mut self,
        seq: SeqId,
        pool: &mut KvPool,
        len_before: usize,
        accepted: &Accepted,
    ) -> Result<()> {
        // `accepted + 1` entries survive: the accepted drafts and the token the
        // previous step had already settled on. Everything past that was written
        // by this pass and is dropped.
        let keep = accepted.accepted + 1;
        pool.truncate(seq, len_before + keep);
        if self.gdn_rollback.is_some() {
            self.commit_recurrent(seq, pool, keep)?;
        }
        Ok(())
    }

    /// Replay the accepted prefix into the persistent recurrent state and the
    /// convolution window.
    ///
    /// Runs the same kernels the forward pass ran, over the same inputs, in the
    /// same order — which is what makes the result the state an unspeculated
    /// decode of those tokens would have left, exactly rather than nearly.
    fn commit_recurrent(&mut self, seq: SeqId, pool: &mut KvPool, keep: usize) -> Result<()> {
        let dev = self.dev.clone();
        let mut r = self
            .gdn_rollback
            .take()
            .context("no rollback journal to commit")?;
        let res = (|| -> Result<()> {
            anyhow::ensure!(r.armed, "committing a journal that was never armed");
            anyhow::ensure!(
                keep <= r.rows,
                "keeping {keep} of a {}-candidate pass",
                r.rows
            );
            anyhow::ensure!(
                r.slot == seq.0,
                "the journal was armed for slot {} and is being committed for {}",
                r.slot,
                seq.0
            );
            // One row per sequence slot: this one contributes the accepted
            // prefix, starting at the journal's row 0, and every other slot
            // contributes nothing and its blocks exit.
            let mut spans = vec![(0i32, 0i32); pool.max_seqs()];
            spans[seq.0] = (0, keep as i32);
            pool.set_gdn_layout(&dev, &spans)?;

            let n_seqs = pool.max_seqs();
            for ordinal in 0..r.n_linear {
                let conv_w = pool_conv_weight(&self.w, ordinal)?;
                let (first, ntok, mut state, mut conv) = pool.gdn_parts(ordinal);
                let seqs = SeqLayout {
                    first_token: &first,
                    n_tokens: &ntok,
                    n_seqs,
                    total_tokens: keep,
                };
                r.replay_layer(
                    &dev,
                    &self.kern,
                    ordinal,
                    keep,
                    &seqs,
                    &conv_w,
                    &mut state,
                    &mut conv,
                )?;
            }
            Ok(())
        })();
        r.disarm();
        self.gdn_rollback = Some(r);
        res
    }
}

/// The convolution weight of the `ordinal`-th linear layer.
///
/// The journal indexes layers by their linear ordinal, the way the state pool
/// does; the weights are indexed by model layer. Walking the layers to find the
/// n-th linear one keeps the two numberings from being conflated at the call
/// site.
fn pool_conv_weight(w: &crate::Weights, ordinal: usize) -> Result<View<'_, f32>> {
    let mut seen = 0usize;
    for l in &w.layers {
        if let Some(g) = l.gdn.as_ref() {
            if seen == ordinal {
                return Ok(g.conv1d.as_view());
            }
            seen += 1;
        }
    }
    anyhow::bail!("no linear layer with ordinal {ordinal}")
}
