//! The multi-token-prediction head, on the device.
//!
//! The host reference in [`crate::qwen35_mtp`] is the specification; this is the
//! same computation over the real kernels, and
//! `crates/model/tests/qwen35_mtp_device.rs` holds the two side by side. Read
//! the reference first — every ordering decision is documented there, with the
//! second reading that also runs.
//!
//! What the head is, in one block:
//!
//! ```text
//! e = rms_norm(embed[shifted_ids], pre_fc_norm_embedding)
//! h = rms_norm(target_final_hidden, pre_fc_norm_hidden)
//! x = fc @ [e | h]                     // embedding in the low half
//! x = full_attention_block(x)          // mtp.layers.0, gate + partial rope
//! o = rms_norm(x, mtp.norm)
//! logits = lm_head @ o                 // the *text model's* lm_head
//! ```
//!
//! Three things about the shape of this file:
//!
//! **It owns its own KV cache.** The head's layer is a full-attention block, so
//! drafting needs somewhere to put `k` and `v` — one layer's worth, `kv_heads *
//! d_head * 2` per token, a sixteenth of what the 27B's sixteen attention layers
//! cost. It is a separate allocation from the text model's pool rather than a
//! row in it, because it is one layer against sixty-four and because rolling the
//! drafter back is a different operation on a different length: the drafter's
//! slot `p` holds the pair `(h_p, emb(t_{p+1}))`, one behind the target's.
//!
//! **It touches no recurrent state.** The head has no GatedDeltaNet layer —
//! `notes/qwen3.5-mtp.md` establishes this three independent ways and
//! [`crate::weights::load_mtp`] refuses a checkpoint that disagrees — so a draft
//! step is pure attention and there is nothing to roll back on the draft side.
//! All the state trouble is on the verification side; see [`crate::spec`].
//!
//! **The positions are the hidden state's, not the embedded token's.** Slot `i`
//! carries `h_i` and `emb(t_{i+1})` and uses rope position `i`. vLLM passes the
//! target's positions to the drafter unchanged and adds one per subsequent draft
//! step. This cannot be pinned by a single forward's numbers — shifting every
//! position by a constant leaves a self-consistent attention output invariant,
//! and the capture measures that invariance at 4.5e-07 relative — so it is read
//! off the reference implementation and asserted structurally instead: the
//! drafter's cache length after `n` rows is what a one-behind cache would be.

use anyhow::{Context, Result};
use infero_gpu::{Buf, View, ViewMut};
use half::f16;
use infero_gpu::Device;
use infero_kernels::{AttnDims, BatchLayout, Kernels, WeightType};

use crate::weights::{Matrix, MtpWeights, mrope_axis_table};

/// Everything about the head's one block that is a number rather than a tensor.
///
/// A copy of the relevant slice of [`crate::Config`] rather than a borrow of it,
/// so that a test can build a head at shapes no checkpoint has.
#[derive(Debug, Clone, Copy)]
pub struct HeadDims {
    pub d_model: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub d_head: usize,
    /// How many of each head's dimensions rotate — 64 of 256 on the 27B.
    pub rotary_dim: usize,
    pub d_ff: usize,
    pub eps: f32,
    pub rope_theta: f32,
    /// The vocabulary, which sizes the logits buffer.
    pub vocab: usize,
    /// See `Config::mrope_section`. `None` for every model without M-RoPE.
    pub mrope_section: Option<[usize; 3]>,
}

impl HeadDims {
    pub fn d_attn(&self) -> usize {
        self.heads * self.d_head
    }

    pub fn d_kv(&self) -> usize {
        self.kv_heads * self.d_head
    }

    pub fn from_config(cfg: &crate::Config) -> Self {
        Self {
            d_model: cfg.d_model,
            heads: cfg.n_heads,
            kv_heads: cfg.n_kv_heads,
            d_head: cfg.d_head,
            rotary_dim: cfg.rotary_dim,
            d_ff: cfg.d_ff,
            eps: cfg.rms_eps,
            rope_theta: cfg.rope_theta,
            vocab: cfg.vocab_size,
            mrope_section: cfg.mrope_section,
        }
    }
}

/// The head's weights, activations and its own KV cache.
pub struct MtpHead {
    dev: Device,
    w: MtpWeights,
    dims: HeadDims,
    max_tokens: usize,
    max_seq: usize,

    /// `[max_tokens, d_model]` — the gathered embedding of the shifted ids.
    emb: Buf<f32>,
    /// `[max_tokens, d_model]` — whatever is playing the part of the hidden
    /// state this step: the text model's final hidden on the first draft step,
    /// the head's own output on the ones after it.
    hidden_in: Buf<f32>,
    /// `[max_tokens, 2 * d_model]` — `[e | h]`, the only place the concat order
    /// is expressed.
    cat: Buf<f32>,
    /// The residual stream.
    x: Buf<f32>,
    xb: Buf<f32>,
    q: Buf<f32>,
    k: Buf<f32>,
    v: Buf<f32>,
    /// Doubles as the `q_proj` output (`2 * d_attn` wide) and the FFN gate.
    gate: Buf<f32>,
    up: Buf<f32>,
    ffn: Buf<f32>,
    /// The attention output gate, de-interleaved out of `q_proj`'s output.
    attn_gate: Buf<f32>,
    attn: Buf<f32>,
    scores: Buf<f32>,
    proj: Buf<f32>,
    /// `[max_tokens, d_model]` — the head's output, after `mtp.norm`. This is
    /// both what `lm_head` scores and what the next draft step feeds back in.
    out: Buf<f32>,
    logits: Buf<f32>,
    x16: Buf<f16>,

    ids: Buf<i32>,
    positions: Buf<i32>,
    /// `[max_tokens, 3]`, token-major `[T, H, W]` -- fed to the rope kernel
    /// instead of `positions` whenever `run` is given a real `mrope` slice
    /// (only `prime`/`step` ever are; `step_from_own_output`/`step_tree` are
    /// always decode-phase and stay on the scalar `positions` path, which is
    /// exactly right there -- see `MtpHead::run`'s doc comment).
    mrope_positions: Buf<i32>,
    slots: Buf<i32>,
    seq_of: Buf<i32>,
    /// All ones: the head rotates every pair at the base frequency.
    freqs: Buf<f32>,
    /// Which axis of `mrope_positions` each rope frequency reads, from
    /// `HeadDims::mrope_section` -- all zeros for a model without one, which
    /// makes `pos_stride: 1` (`positions`, not `mrope_positions`) the only
    /// path that buffer's contents can affect. See
    /// `Kernels::rope_qk_partial`'s doc comment.
    mrope_axis: Buf<i32>,

    /// The drafter's own single-sequence KV cache, `[kv_heads, max_seq, d_head]`.
    kc: Buf<f16>,
    vc: Buf<f16>,
    /// `[branches, max_seq]` — position to slot, one row a branch.
    ///
    /// One branch is the linear draft and the table is the identity. A tree
    /// draft forks: every branch shares the prefix's slots and owns the slots
    /// past it, which is what lets siblings sit at the same position without
    /// overwriting each other's keys. See [`Self::fork`].
    slot_table: Buf<i32>,
    /// Branches the table holds, and so the widest tree level this head can run.
    branches: usize,
    /// Slots the shared prefix occupies. Positions below it map to themselves in
    /// every branch; above it each branch has its own.
    fork_at: usize,
    /// One row quantized to q8_1, for the integer vocabulary mat-vec.
    q8_1: Buf<u8>,
    /// How far the drafter's cache reaches, in positions.
    len: usize,
    /// Rows the last [`MtpHead::step`] produced.
    rows: usize,
    logits_host: Vec<f32>,
    /// Device-sampler state for a sampled draft, one row wide. Allocated on
    /// first use; a greedy draft never touches it.
    samp: Option<DraftSampleBufs>,
}

impl MtpHead {
    /// `max_tokens` bounds one draft step's rows; `max_seq` the drafter's cache.
    pub fn new(
        dev: &Device,
        w: MtpWeights,
        dims: HeadDims,
        max_tokens: usize,
        max_seq: usize,
        // Widest tree level this head will be asked for. One is the linear
        // draft; the slot table and the cache grow with it.
        branches: usize,
    ) -> Result<Self> {
        anyhow::ensure!(max_tokens > 0 && max_seq > 0, "an empty MTP head");
        anyhow::ensure!(branches > 0, "a head with no branches");
        let stream = dev.stream();
        let (d, da, dkv) = (dims.d_model, dims.d_attn(), dims.d_kv());
        let t = max_tokens;
        let alloc = |n: usize, what: &str| -> Result<Buf<f32>> {
            stream
                .alloc_zeros::<f32>(n)
                .with_context(|| format!("allocating the MTP head's {what}"))
        };
        Ok(Self {
            dev: dev.clone(),
            dims,
            max_tokens,
            max_seq,
            emb: alloc(t * d, "embedding rows")?,
            hidden_in: alloc(t * d, "hidden rows")?,
            cat: alloc(t * 2 * d, "fc input")?,
            x: alloc(t * d, "residual")?,
            xb: alloc(t * d, "normalized residual")?,
            q: alloc(t * da, "queries")?,
            k: alloc(t * dkv, "keys")?,
            v: alloc(t * dkv, "values")?,
            gate: alloc(t * (2 * da).max(dims.d_ff), "q_proj output / ffn gate")?,
            up: alloc(t * dims.d_ff, "ffn up")?,
            ffn: alloc(t * dims.d_ff, "ffn hidden")?,
            attn_gate: alloc(t * da, "attention output gate")?,
            attn: alloc(t * da, "attention output")?,
            scores: alloc(dims.heads * t * max_seq, "attention scores")?,
            proj: alloc(t * d, "projection")?,
            out: alloc(t * d, "head output")?,
            // One row, not `t`: `logits_row` projects a single row on demand and
            // every draft reads exactly one. At a vocabulary of 248320 the
            // difference is 993 KiB a row against nothing gained.
            logits: alloc(dims.vocab, "draft logits")?,
            x16: stream.alloc_zeros::<f16>(t * (2 * d).max(dims.d_ff).max(da))?,
            ids: stream.alloc_zeros::<i32>(t)?,
            positions: stream.alloc_zeros::<i32>(t)?,
            slots: stream.alloc_zeros::<i32>(t)?,
            seq_of: stream.alloc_zeros::<i32>(t)?,
            freqs: stream.clone_htod(&vec![1.0f32; dims.rotary_dim / 2])?,
            mrope_axis: stream.clone_htod(&mrope_axis_table(dims.rotary_dim, dims.mrope_section))?,
            mrope_positions: stream.alloc_zeros::<i32>(t * 3)?,
            kc: stream.alloc_zeros::<f16>(dims.kv_heads * max_seq * dims.d_head)?,
            vc: stream.alloc_zeros::<f16>(dims.kv_heads * max_seq * dims.d_head)?,
            slot_table: stream.alloc_zeros::<i32>(branches * max_seq)?,
            branches,
            fork_at: 0,
            // The activation of whichever projection is being quantized,
            // sized for the widest single row `matmul` ever passes it: not
            // `d_model` but `d_ff`, which `w_down` and the FC's `2 * d_model`
            // cat both exceed it by. See `matmul`'s own `has_mmvq` branch.
            q8_1: stream.alloc_zeros::<u8>(Kernels::q8_1_bytes((2 * d).max(da).max(dims.d_ff)))?,
            len: 0,
            rows: 0,
            w,
            logits_host: vec![0.0; dims.vocab],
            samp: None,
        })
    }

    pub fn dims(&self) -> HeadDims {
        self.dims
    }

    /// Device bytes held: the weights plus everything above.
    pub fn bytes(&self) -> usize {
        let acts = self.emb.len()
            + self.hidden_in.len()
            + self.cat.len()
            + self.x.len()
            + self.xb.len()
            + self.q.len()
            + self.k.len()
            + self.v.len()
            + self.gate.len()
            + self.up.len()
            + self.ffn.len()
            + self.attn_gate.len()
            + self.attn.len()
            + self.scores.len()
            + self.proj.len()
            + self.out.len()
            + self.logits.len();
        self.w.device_bytes + acts * 4 + (self.kc.len() + self.vc.len()) * 2
    }

    /// How many positions the drafter's cache holds.
    pub fn cache_len(&self) -> usize {
        self.len
    }

    /// Drop the drafter's cache back to `len` positions.
    ///
    /// The drafter's coordinate system is one behind the target's — slot `p`
    /// holds `(h_p, emb(t_{p+1}))` — so a caller rolling back a rejected draft
    /// has to convert. [`crate::spec`] does; nothing here does it silently.
    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    pub fn reset(&mut self) {
        self.len = 0;
        self.rows = 0;
    }

    /// One draft step over `shifted_ids.len()` rows.
    ///
    /// * `embed` is the text model's embedding matrix — the head has none.
    /// * `shifted_ids[i]` is the token *after* the one whose hidden state is
    ///   `hidden[i]`: vLLM's `[a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3,
    ///   next]` shift, with the freshly sampled token in the last slot.
    /// * `positions[i]` is the position of the *hidden state's* token.
    /// * `hidden` is `[rows, d_model]`, the text model's output **after**
    ///   `model.language_model.norm`.
    /// * `mrope`: see [`Self::run`]'s doc comment.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        kern: &Kernels,
        embed: &Matrix,
        shifted_ids: &[u32],
        positions: &[usize],
        hidden: &View<'_, f32>,
        mrope: Option<&[i32]>,
    ) -> Result<()> {
        let n = shifted_ids.len();
        anyhow::ensure!(
            n == positions.len(),
            "{n} shifted ids against {} positions",
            positions.len()
        );
        anyhow::ensure!(
            hidden.len() >= n * self.dims.d_model,
            "the hidden states hold {} floats, {n} rows of {} needed",
            hidden.len(),
            self.dims.d_model
        );
        anyhow::ensure!(
            n <= self.max_tokens,
            "{n} rows into a head built for {}; call `prime` for a feed that \
             may be wider than one step",
            self.max_tokens
        );
        self.dev
            .stream()
            .memcpy_dtod(hidden, &mut self.hidden_in.slice_mut(..n * self.dims.d_model))?;
        // One branch: `step` is the linear draft's own entry point.
        self.run(kern, embed, shifted_ids, positions, &vec![0; shifted_ids.len()], 0, mrope)
    }

    /// [`MtpHead::step`] over a feed of any width, in chunks.
    ///
    /// Priming after a prefill covers the whole prompt — `DraftFeed::after_prefill`
    /// says why, and it is not optional: the drafter attends over its own cache,
    /// and a cache with holes is a different model. But three of this head's
    /// buffers are `max_tokens` wide in a way that cannot absorb a prompt.
    /// `scores` is `heads * max_tokens * max_seq`, which at 2048 rows and 8192
    /// slots is 1.6 TB. So the head stays narrow and the feed is split.
    ///
    /// This is a chunked prefill, and `run` already permits it: it asserts
    /// `positions[0] <= len` and appends at absolute positions, so a chunk
    /// starting where the cache ends is exactly the legal case. What it does not
    /// do is renumber rows, which is the return value — the last chunk's index of
    /// the final token, since `rows` only ever describes the most recent chunk.
    /// A caller that keeps using `feed.rows.len() - 1` reads a row the last chunk
    /// never wrote.
    /// `mrope`: `Some`, token-major `[T,H,W]` (`3 * shifted_ids.len()`
    /// entries) over the *whole* feed, when priming after a prefill that may
    /// have spliced in an image -- sliced per chunk below the same way
    /// `positions`/`shifted_ids` are. See [`Self::run`]'s doc comment for why
    /// this is the one place besides `step` that takes a real one.
    #[allow(clippy::too_many_arguments)]
    pub fn prime(
        &mut self,
        kern: &Kernels,
        embed: &Matrix,
        shifted_ids: &[u32],
        positions: &[usize],
        hidden: &View<'_, f32>,
        mrope: Option<&[i32]>,
    ) -> Result<usize> {
        let n = shifted_ids.len();
        anyhow::ensure!(n > 0, "priming the drafter with no rows");
        anyhow::ensure!(
            n == positions.len(),
            "{n} shifted ids against {} positions",
            positions.len()
        );
        if let Some(m) = mrope {
            anyhow::ensure!(
                m.len() == 3 * n,
                "{} mrope entries for {n} rows, expected {}",
                m.len(),
                3 * n
            );
        }
        let d = self.dims.d_model;
        anyhow::ensure!(
            hidden.len() >= n * d,
            "the hidden states hold {} floats, {n} rows of {d} needed",
            hidden.len(),
        );
        let width = self.max_tokens;
        let mut last_row = 0;
        let mut start = 0;
        while start < n {
            let end = (start + width).min(n);
            let chunk = hidden.slice(start * d..end * d);
            self.step(
                kern,
                embed,
                &shifted_ids[start..end],
                &positions[start..end],
                &chunk,
                mrope.map(|m| &m[3 * start..3 * end]),
            )?;
            last_row = end - start - 1;
            start = end;
        }
        Ok(last_row)
    }

    /// The same, feeding the head its **own** previous output as the hidden
    /// state.
    ///
    /// This is what makes a `k`-token draft possible from a one-layer head:
    /// `spec_step_idx % mtp_num_hidden_layers` is always zero, so step two
    /// re-enters the same layer with `mtp.norm`'s output — not the layer's
    /// pre-norm output, and not the text model's hidden state again — paired with
    /// the embedding of the token step one drafted.
    pub fn step_from_own_output(
        &mut self,
        kern: &Kernels,
        embed: &Matrix,
        drafted: u32,
        position: usize,
        row: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            row < self.rows,
            "row {row} is past the {} the last step produced",
            self.rows
        );
        let d = self.dims.d_model;
        let src = self.out.slice(row * d..(row + 1) * d);
        // A device-to-device copy rather than a view swap: `run` writes `out`,
        // so reading it as an input would alias.
        let mut dst = self.hidden_in.slice_mut(..d);
        self.dev.stream().memcpy_dtod(&src, &mut dst)?;
        self.run(kern, embed, &[drafted], &[position], &[0], 0, None)
    }

    /// One draft step's kernels: four small uploads, then eighteen launches.
    ///
    /// **A CUDA graph over this buys nothing.** Eighteen kernels at one row looks
    /// like launch latency, and capturing them — keyed on `(rows, kv bucket)`,
    /// bucketed at 64 the way the text side's decode graphs are — measured 4.11
    /// ms against 4.19 at k=2. The capture was correct (all six of the head's
    /// tests passed with it, reference output and acceptance length included);
    /// it was simply nothing to gain, because the draft has been within 0.75 ms
    /// of its byte bound since the vocabulary projection moved to `mmvq`. The
    /// 120 GB/s I thought this ran at came from subtracting an assumed
    /// `lm_head` time from the total, and the assumption was wrong.
    /// One tree level: several rows, each continuing its own branch from its own
    /// parent's output.
    ///
    /// `src_rows[i]` is the row of the previous level whose output feeds row `i`
    /// — siblings share a parent and so share a source row. `positions[i]` is
    /// where row `i` sits in its branch's sequence, and `branch_of[i]` which
    /// branch that is, which together pick the slot its key goes to.
    ///
    /// The whole level in one call is the point: a draft pass costs what its
    /// weights cost and rows are nearly free, so a level of `B^L` branches is the
    /// same price as one. Calling this once a branch instead would multiply the
    /// draft's cost by the tree's width and give the whole scheme back.
    pub fn step_tree(
        &mut self,
        kern: &Kernels,
        embed: &Matrix,
        tokens: &[u32],
        positions: &[usize],
        src_rows: &[usize],
        branch_of: &[usize],
        tail: usize,
    ) -> Result<()> {
        let n = tokens.len();
        anyhow::ensure!(
            n == positions.len() && n == src_rows.len() && n == branch_of.len(),
            "{n} tokens against {} positions, {} source rows and {} branches",
            positions.len(),
            src_rows.len(),
            branch_of.len()
        );
        anyhow::ensure!(n > 0 && n <= self.max_tokens, "a level of {n} rows");
        for (i, r) in src_rows.iter().enumerate() {
            anyhow::ensure!(
                *r < self.rows,
                "row {i} continues from row {r}, past the {} the last level \
                 produced",
                self.rows
            );
        }
        for b in branch_of {
            anyhow::ensure!(
                *b < self.branches,
                "branch {b} of a head forked {} ways",
                self.branches
            );
        }
        let d = self.dims.d_model;
        // Gather the parents' outputs into consecutive rows. A copy rather than a
        // view because `run` writes `out`, so reading it as an input would alias
        // — the same reason `step_from_own_output` copies.
        for (i, r) in src_rows.iter().enumerate() {
            let src = self.out.slice(r * d..(r + 1) * d);
            let mut dst = self.hidden_in.slice_mut(i * d..(i + 1) * d);
            self.dev.stream().memcpy_dtod(&src, &mut dst)?;
        }
        self.run(kern, embed, tokens, positions, branch_of, tail, None)
    }

    /// Point every branch at the same prefix and its own slots past it.
    ///
    /// Branch `b`, position `p` maps to `p` while `p < base` and to
    /// `base + b * tail + (p - base)` above it. So siblings can sit at the same
    /// position without overwriting each other's keys, which is the whole
    /// mechanism a tree draft needs — the drafter's attention already reads its
    /// cache through this table and through `seq_of`, because it shares
    /// `BatchLayout` with the text model's batched path. Only the sequence count
    /// was pinned at one.
    ///
    /// `tail` is how far past the prefix a branch may reach, so `depth - 1` for a
    /// tree of that depth. The table is filled to `base + branches * tail` and no
    /// further: the attention never reads past `kv_len`.
    ///
    /// Uploaded once a draft rather than once a level — at a 200-token prefix and
    /// eight branches that is 6.5 KB, which is why this is a host-side build and
    /// not a kernel.
    pub fn fork(&mut self, base: usize, tail: usize) -> Result<()> {
        anyhow::ensure!(
            base + self.branches * tail <= self.max_seq,
            "a prefix of {base} and {} branches of {tail} want {} slots, the \
             drafter's cache holds {}",
            self.branches,
            base + self.branches * tail,
            self.max_seq
        );
        let width = base + tail;
        let mut table = vec![0i32; self.branches * self.max_seq];
        for b in 0..self.branches {
            for p in 0..width {
                table[b * self.max_seq + p] = if p < base {
                    p as i32
                } else {
                    (base + b * tail + (p - base)) as i32
                };
            }
        }
        self.dev
            .stream()
            .memcpy_htod(&table, &mut self.slot_table.slice_mut(..))?;
        self.fork_at = base;
        Ok(())
    }

    /// Which slot branch `b` writes for `position`, matching [`Self::fork`].
    ///
    /// The host needs this to tell `run` where a row's key goes; the table tells
    /// the *kernel* where to read it. Both have to agree, so they are one
    /// formula in one place.
    fn slot_of(&self, branch: usize, position: usize, tail: usize) -> usize {
        if position < self.fork_at {
            position
        } else {
            self.fork_at + branch * tail + (position - self.fork_at)
        }
    }

    /// `branch_of` names the tree branch each row belongs to, and `tail` how far
    /// past the fork point a branch may reach. All zeros and any `tail` is the
    /// linear draft, where a slot equals its position.
    ///
    /// `mrope`: `Some`, token-major `[T,H,W]` (`3 * shifted_ids.len()`
    /// entries), only when this row set can be mid-image -- which in
    /// practice means only `step`/`prime`, priming the drafter right after a
    /// prefill that may have spliced one in. `step_from_own_output` and
    /// `step_tree` are always decode-phase continuations of an already-primed
    /// cache, never mid-image, so they pass `None` and get the scalar
    /// `positions` path -- correct because a decode-phase position's three
    /// axes are always equal, which `crates/kernels/tests/mrope.rs`'s
    /// `equal_axes_reduce_to_the_scalar_case` proves is bit-identical to
    /// giving the kernel real equal-valued axes anyway.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        kern: &Kernels,
        embed: &Matrix,
        shifted_ids: &[u32],
        positions: &[usize],
        branch_of: &[usize],
        tail: usize,
        mrope: Option<&[i32]>,
    ) -> Result<()> {
        let n = shifted_ids.len();
        let dims = self.dims;
        let (d, da, dkv) = (dims.d_model, dims.d_attn(), dims.d_kv());
        anyhow::ensure!(
            n > 0 && n <= self.max_tokens,
            "a draft step of {n} rows, against the {} this head was built for",
            self.max_tokens
        );
        let last = *positions.last().unwrap();
        anyhow::ensure!(
            last + 1 <= self.max_seq,
            "position {last} is past the {} slots the drafter's cache holds",
            self.max_seq
        );
        // Consecutive *within a branch*. A tree level puts one row per branch at
        // the same position, which is the point of forking; what would still be
        // wrong is a gap along one branch, since the slot mapping and the cache
        // length both assume a branch is contiguous.
        for (i, w) in positions.windows(2).enumerate() {
            if branch_of[i] != branch_of[i + 1] {
                continue;
            }
            anyhow::ensure!(
                w[1] == w[0] + 1,
                "branch {}'s rows must be consecutive positions, got {w:?}",
                branch_of[i]
            );
        }
        anyhow::ensure!(
            positions[0] <= self.len,
            "a draft step at position {} with only {} positions cached would \
             leave a hole in the drafter's history",
            positions[0],
            self.len
        );
        anyhow::ensure!(
            embed.k == d && embed.n >= 1,
            "the embedding matrix is [{}, {}], expected rows of {d}",
            embed.n,
            embed.k
        );

        let stream = self.dev.stream().clone();
        let ids: Vec<i32> = shifted_ids.iter().map(|t| *t as i32).collect();
        // Slot equals position: the drafter is one sequence with a cache of its
        // own, so there is no pool to share and no indirection to get wrong.
        let pos: Vec<i32> = positions.iter().map(|p| *p as i32).collect();
        // The slot a row's key goes to has to match what `fork` told the kernel
        // to read, so both come from `slot_of`.
        let slots: Vec<i32> = positions
            .iter()
            .zip(branch_of)
            .map(|(p, b)| self.slot_of(*b, *p, tail) as i32)
            .collect();
        let seqs: Vec<i32> = branch_of.iter().map(|b| *b as i32).collect();
        stream.memcpy_htod(&ids, &mut self.ids.slice_mut(..n))?;
        stream.memcpy_htod(&pos, &mut self.positions.slice_mut(..n))?;
        stream.memcpy_htod(&slots, &mut self.slots.slice_mut(..n))?;
        stream.memcpy_htod(&seqs, &mut self.seq_of.slice_mut(..n))?;
        // `pos_stride` is a property of this *head*, not of this call:
        // `self.mrope_axis` holds the head's real, possibly non-zero axis
        // map whenever `dims.mrope_section` is set, regardless of whether
        // this particular row set carries an `mrope` array. Making
        // `pos_stride` follow `mrope.is_some()` instead of `mrope_axis`'s own
        // shape was a real bug caught by `tests/mtp_mrope.rs`: a decode-phase
        // call (`mrope: None`) on a model with M-RoPE would fall back to
        // `pos_stride: 1` while `mrope_axis` still named axis 1 or 2 for some
        // frequencies, reading `self.positions[token + 1]` /
        // `[token + 2]` -- a different token's position entirely, since
        // `self.positions` is only `n` long, one value a token. Broadcasting
        // `[p, p, p]` here for the `None` case is what `Acts::mrope_positions`
        // does on the target model's decode path, for the identical reason.
        let pos_stride = if self.dims.mrope_section.is_some() { 3 } else { 1 };
        if pos_stride == 3 {
            let triples: Vec<i32> = match mrope {
                Some(m) => {
                    anyhow::ensure!(
                        m.len() == 3 * n,
                        "{} mrope entries for {n} rows, expected {}",
                        m.len(),
                        3 * n
                    );
                    m.to_vec()
                }
                None => positions.iter().flat_map(|&p| [p as i32; 3]).collect(),
            };
            stream.memcpy_htod(&triples, &mut self.mrope_positions.slice_mut(..n * 3))?;
        }
        kern.write_slot_table(
            &mut self.slot_table.as_view_mut(),
            &self.seq_of.slice(..n),
            &self.positions.slice(..n),
            &self.slots.slice(..n),
            self.max_seq,
            n,
        )?;

        kern.gather_rows(
            &mut self.emb.slice_mut(..n * d),
            &embed.view(None)?,
            embed.ty,
            &self.ids.slice(..n),
            n,
            d,
        )?;

        // `[e | h]`, embedding first. One `rms_norm` launch per row per half:
        // the kernel writes rows of `d` back to back and the destination here is
        // strided by `2 * d`, so the alternative is normalizing into two buffers
        // and copying them together — the same number of launches, plus the
        // copies. The ordering decision lives here and nowhere else.
        for t in 0..n {
            let base = t * 2 * d;
            kern.rms_norm(
                &mut self.cat.slice_mut(base..base + d),
                &self.emb.slice(t * d..(t + 1) * d),
                &self.w.pre_fc_norm_embedding.as_view(),
                1,
                d,
                dims.eps,
            )?;
            kern.rms_norm(
                &mut self.cat.slice_mut(base + d..base + 2 * d),
                &self.hidden_in.slice(t * d..(t + 1) * d),
                &self.w.pre_fc_norm_hidden.as_view(),
                1,
                d,
                dims.eps,
            )?;
        }
        matmul(
            kern,
            &mut self.x.slice_mut(..n * d),
            &self.w.fc,
            &mut self.q8_1,
            &mut self.x16,
            &self.cat.slice(..n * 2 * d),
            n,
        )?;

        // ---- the decoder layer, pre-norm with two residuals ----------------
        let l = self.w.layer.attn();
        kern.rms_norm(
            &mut self.xb.slice_mut(..n * d),
            &self.x.slice(..n * d),
            &self.w.layer.attn_norm.as_view(),
            n,
            d,
            dims.eps,
        )?;
        // The query and its gate interleave per head — 512 columns a head, q
        // first — so `q_proj`'s output lands in `gate` and is de-interleaved.
        // Splitting it down the middle instead also runs and is a different
        // model; `tests/qwen35_mtp.rs` pins the layout past head 0, where the
        // two readings stop coinciding.
        anyhow::ensure!(
            l.output_gate,
            "the MTP head's q_proj is not gated; this forward pass applies a \
             sigmoid gate before o_proj and there is nothing to apply"
        );
        matmul(
            kern,
            &mut self.gate.slice_mut(..n * 2 * da),
            &l.wq,
            &mut self.q8_1,
            &mut self.x16,
            &self.xb.slice(..n * d),
            n,
        )?;
        {
            let (q, ag) = (&mut self.q, &mut self.attn_gate);
            kern.split_interleaved(
                &mut q.slice_mut(..n * da),
                &mut ag.slice_mut(..n * da),
                &self.gate.slice(..n * 2 * da),
                n,
                dims.heads,
                dims.d_head,
            )?;
        }
        matmul(
            kern,
            &mut self.k.slice_mut(..n * dkv),
            &l.wk,
            &mut self.q8_1,
            &mut self.x16,
            &self.xb.slice(..n * d),
            n,
        )?;
        matmul(
            kern,
            &mut self.v.slice_mut(..n * dkv),
            &l.wv,
            &mut self.q8_1,
            &mut self.x16,
            &self.xb.slice(..n * d),
            n,
        )?;

        // Per-head q/k norms, before the rotary. The weights were given their
        // `+1` at load — see `weights::norm_offset` — so the plain kernel is
        // right here and adding the offset again would square the mistake.
        let qn = l
            .q_norm
            .as_ref()
            .context("the MTP head has no q_norm; Qwen3.5's blocks all do")?;
        kern.qk_norm(
            &mut self.q.slice_mut(..n * da),
            &qn.as_view(),
            n,
            dims.heads,
            dims.d_head,
            da,
            0,
            dims.eps,
        )?;
        let kn = l
            .k_norm
            .as_ref()
            .context("the MTP head has no k_norm; Qwen3.5's blocks all do")?;
        kern.qk_norm(
            &mut self.k.slice_mut(..n * dkv),
            &kn.as_view(),
            n,
            dims.kv_heads,
            dims.d_head,
            dkv,
            0,
            dims.eps,
        )?;

        {
            let rope_positions = if pos_stride == 3 {
                self.mrope_positions.slice(..n * 3)
            } else {
                self.positions.slice(..n)
            };
            let (q, k) = (&mut self.q, &mut self.k);
            kern.rope_qk_partial(
                &mut q.slice_mut(..n * da),
                &mut k.slice_mut(..n * dkv),
                &rope_positions,
                &self.freqs.as_view(),
                &self.mrope_axis.as_view(),
                pos_stride,
                n,
                dims.heads,
                dims.kv_heads,
                dims.d_head,
                dims.rotary_dim,
                dims.rope_theta,
                1.0,
                false,
            )?;
        }

        {
            let (kc, vc) = (&mut self.kc, &mut self.vc);
            kern.store_kv2(
                &mut kc.as_view_mut(),
                &mut vc.as_view_mut(),
                &self.k.slice(..n * dkv),
                &self.v.slice(..n * dkv),
                &self.slots.slice(..n),
                dims.kv_heads,
                dims.d_head,
                self.max_seq,
                n,
            )?;
        }

        // The drafter's cache now reaches one past the last row's position.
        self.len = self.len.max(last + 1);
        let kv_len = self.len;
        let attn_dims = AttnDims {
            n_heads: dims.heads,
            n_kv_heads: dims.kv_heads,
            d_head: dims.d_head,
            n_slots: self.max_seq,
            n_tokens: n,
        };
        let score_len = dims.heads * n * kv_len;
        {
            let seq_of = self.seq_of.slice(..n);
            let positions_v = self.positions.slice(..n);
            let table = self.slot_table.as_view();
            let batch = BatchLayout {
                seq_of: &seq_of,
                positions: &positions_v,
                slot_table: &table,
                table_stride: self.max_seq,
            };
            kern.attn_scores(
                &mut self.scores.slice_mut(..score_len),
                &self.q.slice(..n * da),
                &self.kc.as_view(),
                batch,
                attn_dims,
                kv_len,
                1.0 / (dims.d_head as f32).sqrt(),
            )?;
            kern.attn_softmax(
                &mut self.scores.slice_mut(..score_len),
                dims.heads,
                n,
                kv_len,
            )?;
            let batch = BatchLayout {
                seq_of: &seq_of,
                positions: &positions_v,
                slot_table: &table,
                table_stride: self.max_seq,
            };
            kern.attn_output(
                &mut self.attn.slice_mut(..n * da),
                &self.scores.slice(..score_len),
                &self.vc.as_view(),
                batch,
                attn_dims,
                kv_len,
                None,
            )?;
        }

        // Sigmoid, not silu: `output_gate_type: "swish"` is in the config and no
        // implementation reads it.
        kern.sigmoid_gate(
            &mut self.attn.slice_mut(..n * da),
            &self.attn_gate.slice(..n * da),
            n * da,
        )?;
        matmul(
            kern,
            &mut self.proj.slice_mut(..n * d),
            &l.wo,
            &mut self.q8_1,
            &mut self.x16,
            &self.attn.slice(..n * da),
            n,
        )?;
        kern.add_assign(
            &mut self.x.slice_mut(..n * d),
            &self.proj.slice(..n * d),
            n * d,
        )?;

        kern.rms_norm(
            &mut self.xb.slice_mut(..n * d),
            &self.x.slice(..n * d),
            &self.w.layer.ffn_norm.as_view(),
            n,
            d,
            dims.eps,
        )?;
        matmul(
            kern,
            &mut self.gate.slice_mut(..n * dims.d_ff),
            &self.w.layer.dense().w_gate,
            &mut self.q8_1,
            &mut self.x16,
            &self.xb.slice(..n * d),
            n,
        )?;
        matmul(
            kern,
            &mut self.up.slice_mut(..n * dims.d_ff),
            &self.w.layer.dense().w_up,
            &mut self.q8_1,
            &mut self.x16,
            &self.xb.slice(..n * d),
            n,
        )?;
        kern.silu_mul(
            &mut self.ffn.slice_mut(..n * dims.d_ff),
            &self.gate.slice(..n * dims.d_ff),
            &self.up.slice(..n * dims.d_ff),
            n * dims.d_ff,
        )?;
        matmul(
            kern,
            &mut self.proj.slice_mut(..n * d),
            &self.w.layer.dense().w_down,
            &mut self.q8_1,
            &mut self.x16,
            &self.ffn.slice(..n * dims.d_ff),
            n,
        )?;
        kern.add_assign(
            &mut self.x.slice_mut(..n * d),
            &self.proj.slice(..n * d),
            n * d,
        )?;

        // The head's own final norm, a different tensor from the text model's.
        kern.rms_norm(
            &mut self.out.slice_mut(..n * d),
            &self.x.slice(..n * d),
            &self.w.norm.as_view(),
            n,
            d,
            dims.eps,
        )?;
        self.rows = n;
        Ok(())
    }

    /// The head's output for one row, which is what `lm_head` consumes.
    pub fn output_row(&self, row: usize) -> View<'_, f32> {
        let d = self.dims.d_model;
        self.out.slice(row * d..(row + 1) * d)
    }

    /// Every row the last step produced.
    pub fn output(&self) -> View<'_, f32> {
        self.out.slice(..self.rows * self.dims.d_model)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// How far the drafter's own cache reaches, in positions.
    ///
    /// Observable because chunked priming's whole job is to leave this where a
    /// single wide step would have: `rows` describes only the last chunk, so it
    /// cannot answer whether the earlier ones landed.
    /// Lanes the head's cache was forked for, and so the widest tree level it
    /// can run.
    pub fn branch_count(&self) -> usize {
        self.branches
    }

    pub fn cached(&self) -> usize {
        self.len
    }

    /// `head @ out[row]`, brought back to the host.
    ///
    /// `head` is the text model's `lm_head`. The head has none of its own —
    /// `tie_word_embeddings` is false on this checkpoint, so `lm_head` and the
    /// embedding are different tensors and the drafter wants the former.
    /// The vocabulary projection for one row, left on the device.
    ///
    /// A sampled draft does not need the logits on the host: the device sampler
    /// returns the token and the truncated distribution the acceptance rule
    /// composes with, which is a kilobyte where these are 993 KB. Returns the
    /// row width.
    pub fn logits_row_device(
        &mut self,
        kern: &Kernels,
        head: &Matrix,
        row: usize,
    ) -> Result<usize> {
        let d = self.dims.d_model;
        anyhow::ensure!(row < self.rows, "row {row} of {}", self.rows);
        anyhow::ensure!(
            head.k == d,
            "the vocabulary projection contracts over {} where the head is {d} \
             wide",
            head.k
        );
        let n = head.n;
        if self.logits_host.len() < n {
            self.logits_host = vec![0.0; n];
        }
        {
            let src = self.out.slice(row * d..(row + 1) * d);
            let mut out = self.logits.slice_mut(..n);
            // The same kernel the text model's own vocabulary projection takes,
            // not `matmul`'s `gemv`. This is the largest matrix a draft touches
            // — 1288 MiB on the 27B against the head's own 476 — so which
            // mat-vec runs on it decides the draft's cost, and the integer one
            // is what `Model::forward_batch_rows` picks at one row for exactly
            // this reason. Going through `gemv` here left the draft at 4.65 ms
            // against a 1.68 ms byte bound.
            if Kernels::has_mmvq(head.ty) && d.is_multiple_of(32) {
                let bytes = Kernels::q8_1_bytes(d);
                kern.quantize_q8_1(&mut self.q8_1.slice_mut(..bytes), &src, d)?;
                kern.mmvq(
                    &mut out,
                    &head.view(None)?,
                    head.ty,
                    &self.q8_1.slice(..bytes),
                    d,
                    n,
                )?;
            } else {
                matmul(kern, &mut out, head, &mut self.q8_1, &mut self.x16, &src, 1)?;
            }
        }
        Ok(n)
    }

    /// The same, copied to the host. Callers that need only a token or a
    /// truncated distribution should take the device path.
    pub fn logits_row(&mut self, kern: &Kernels, head: &Matrix, row: usize) -> Result<&[f32]> {
        let n = self.logits_row_device(kern, head, row)?;
        let stream = self.dev.stream().clone();
        stream.memcpy_dtoh(&self.logits.slice(..n), &mut self.logits_host[..n])?;
        self.dev.synchronize()?;
        Ok(&self.logits_host[..n])
    }

    /// The greedy draft token for one row.
    pub fn draft_row(&mut self, kern: &Kernels, head: &Matrix, row: usize) -> Result<u32> {
        let logits = self.logits_row(kern, head, row)?;
        Ok(argmax(logits))
    }

    /// One drafted token, sampled the way the request asked for, with the
    /// probability the draft assigned it.
    ///
    /// The probability is the other half of the acceptance rule, and it has to
    /// come from the same transformation the target's will — temperature, top-k,
    /// top-p, repetition penalty — which is why this borrows the request's own
    /// sampler rather than taking a temperature.
    ///
    /// Drafting by argmax and accepting stochastically is also valid: the draft
    /// is then a point mass, the ratio `p_target / p_draft` becomes
    /// `p_target(t)`, and the rule still preserves the target's distribution
    /// exactly. It just accepts rarely, because a point mass is a poor proposal
    /// — at temperature 0.7 the target's own top token typically carries 0.3 to
    /// 0.7 of the mass, so roughly half the drafts die. Sampling the draft at
    /// the request's temperature is what makes the ratio close to one whenever
    /// the two models agree.
    /// The drafter's own distribution for one row, truncated the way the request
    /// asked for.
    ///
    /// Split from the sampling so that a tree node can draw several candidates
    /// from it. See [`Self::draft_row_candidates`].
    fn draft_row_dist(
        &mut self,
        kern: &Kernels,
        head: &Matrix,
        row: usize,
        sampler: &mut crate::Sampler,
        history: &[u32],
    ) -> Result<(u32, Vec<(u32, f32)>)> {
        anyhow::ensure!(
            !sampler.params().is_greedy(),
            "draft_row_sampled needs a sampling distribution; a greedy request \
             takes the greedy acceptance rule"
        );
        let n = self.logits_row_device(kern, head, row)?;
        let sp = sampler.params().clone();
        let draw = sampler.next_draw();

        // The penalty window, run-length encoded the way the kernel wants it:
        // sorted unique ids with counts. Tens of tokens, so it stays on the host
        // and rides along in the same transfer as the parameters.
        let win = sp.repetition_window.min(history.len());
        let mut w: Vec<u32> = history[history.len() - win..].to_vec();
        w.sort_unstable();
        let mut ptok: Vec<i32> = Vec::with_capacity(w.len());
        let mut pcnt: Vec<i32> = Vec::with_capacity(w.len());
        let mut i = 0usize;
        while i < w.len() {
            let t = w[i];
            let mut c = 0i32;
            while i < w.len() && w[i] == t {
                c += 1;
                i += 1;
            }
            if (t as usize) < n {
                ptok.push(t as i32);
                pcnt.push(c);
            }
        }

        let top_k = sp.top_k.max(1);
        let stride = top_k.max(ptok.len()).max(1);
        let fits = matches!(&self.samp, Some(b) if b.stride >= stride);
        if !fits {
            let st = self.dev.stream().clone();
            let cand = Kernels::SAMPLE_SPLITS * stride;
            self.samp = Some(DraftSampleBufs {
                params: st.alloc_zeros::<f32>(4)?,
                pen_tok: st.alloc_zeros::<i32>(stride)?,
                pen_cnt: st.alloc_zeros::<i32>(stride)?,
                pen_len: st.alloc_zeros::<i32>(1)?,
                rnd: st.alloc_zeros::<f64>(1)?,
                out: st.alloc_zeros::<u32>(1)?,
                cand_v: st.alloc_zeros::<f32>(cand)?,
                cand_i: st.alloc_zeros::<i32>(cand)?,
                surv_id: st.alloc_zeros::<u32>(stride)?,
                surv_p: st.alloc_zeros::<f32>(stride)?,
                surv_len: st.alloc_zeros::<i32>(1)?,
                stride,
            });
        }
        let b = self.samp.as_mut().unwrap();
        let stride = b.stride;
        let stream = self.dev.stream().clone();
        let params = [
            sp.temperature,
            sp.top_p,
            f32::from_bits(top_k as u32),
            sp.repetition_penalty,
        ];
        let plen = [ptok.len() as i32];
        ptok.resize(stride, 0);
        pcnt.resize(stride, 0);
        stream.memcpy_htod(&params, &mut b.params.slice_mut(..4))?;
        stream.memcpy_htod(&ptok, &mut b.pen_tok.slice_mut(..stride))?;
        stream.memcpy_htod(&pcnt, &mut b.pen_cnt.slice_mut(..stride))?;
        stream.memcpy_htod(&plen, &mut b.pen_len.slice_mut(..1))?;
        stream.memcpy_htod(&[draw], &mut b.rnd.slice_mut(..1))?;

        {
            let (pv, tv, cv2, lv, rv) = (
                b.params.slice(..4),
                b.pen_tok.slice(..stride),
                b.pen_cnt.slice(..stride),
                b.pen_len.slice(..1),
                b.rnd.slice(..1),
            );
            let mut out_v = b.out.slice_mut(..1);
            let mut cav = b.cand_v.slice_mut(..Kernels::SAMPLE_SPLITS * top_k);
            let mut cai = b.cand_i.slice_mut(..Kernels::SAMPLE_SPLITS * top_k);
            let mut id_v = b.surv_id.slice_mut(..stride);
            let mut p_v = b.surv_p.slice_mut(..stride);
            let mut len_v = b.surv_len.slice_mut(..1);
            kern.sample_rows_split(
                &mut out_v,
                &mut cav,
                &mut cai,
                &self.logits.slice(..n),
                &pv,
                &tv,
                &cv2,
                &lv,
                &rv,
                1,
                n,
                stride,
                top_k,
                Some(infero_kernels::Survivors {
                    id: &mut id_v,
                    p: &mut p_v,
                    len: &mut len_v,
                    stride,
                }),
            )?;
        }

        let mut tok_out = [0u32; 1];
        let mut len_out = [0i32; 1];
        let mut ids = vec![0u32; stride];
        let mut ps = vec![0f32; stride];
        stream.memcpy_dtoh(&b.out.slice(..1), &mut tok_out)?;
        stream.memcpy_dtoh(&b.surv_len.slice(..1), &mut len_out)?;
        stream.memcpy_dtoh(&b.surv_id.slice(..stride), &mut ids)?;
        stream.memcpy_dtoh(&b.surv_p.slice(..stride), &mut ps)?;
        self.dev.synchronize()?;
        let token = tok_out[0];
        let keep = (len_out[0].max(0) as usize).min(stride);
        anyhow::ensure!(
            keep > 0,
            "the device sampler kept no survivors, so there is no `q` to accept \
             against"
        );
        // The *whole* distribution, normalized, not just the sampled token's
        // probability.
        //
        // The acceptance rule's residual is `(p_target - q)+` at every token,
        // and `q` is this. Keeping only `q(drafted)` and subtracting it at that
        // one token treats the drafter as a point mass, which it is not once it
        // has sampled — and the composition then does not reproduce the target's
        // distribution. Measured, that error put one token 0.0745 away from its
        // probability, about a hundred standard errors.
        //
        // The cost is the truncated support, so top_k pairs — 40 for a typical
        // request, against a 248320-entry vocabulary.
        let q: Vec<(u32, f32)> = ids[..keep]
            .iter()
            .copied()
            .zip(ps[..keep].iter().copied())
            .collect();
        Ok((token, q))
    }

    /// One drafted token from a row, with the distribution it came from.
    pub fn draft_row_sampled(
        &mut self,
        kern: &Kernels,
        head: &Matrix,
        row: usize,
        sampler: &mut crate::Sampler,
        history: &[u32],
    ) -> Result<(u32, Vec<(u32, f32)>)> {
        let (token, q) = self.draft_row_dist(kern, head, row, sampler, history)?;
        anyhow::ensure!(
            q.iter().any(|(t, w)| *t == token && *w > 0.0),
            "the draft sampled {token}, which carries no weight in its own \
             distribution"
        );
        Ok((token, q))
    }

    /// `n` candidates from one row, drawn i.i.d. from the same distribution.
    ///
    /// Siblings in a tree are exactly this: one parent, one `q`, several draws.
    /// Sharing the distribution is not an optimization — the multi-candidate
    /// acceptance rule is stated for candidates i.i.d. from a single `q`, and
    /// that is what makes the composition reproduce the target.
    ///
    /// Duplicates are allowed and are not a bug: i.i.d. draws from a peaked `q`
    /// often repeat, and the acceptance rule handles a repeat correctly — the
    /// second copy is tested against a residual that has already had `q`
    /// subtracted once, so it is much less likely to pass. Removing them would
    /// change the proposal and break the guarantee.
    pub fn draft_row_candidates(
        &mut self,
        kern: &Kernels,
        head: &Matrix,
        row: usize,
        n: usize,
        sampler: &mut crate::Sampler,
        history: &[u32],
    ) -> Result<(Vec<u32>, Vec<(u32, f32)>)> {
        anyhow::ensure!(n > 0, "a node with no candidates");
        // The device's own draw is the first candidate rather than a discarded
        // one, so a node of `n` candidates consumes exactly `n` uniforms and a
        // width of one reproduces the linear draft's stream token for token.
        let (first, q) = self.draft_row_dist(kern, head, row, sampler, history)?;
        let mut cands = Vec::with_capacity(n);
        cands.push(first);
        for _ in 1..n {
            cands.push(crate::Sampler::pick(&q, 1.0, sampler.next_draw()));
        }
        Ok((cands, q))
    }
}

/// A drafted tree: nodes in breadth-first order, with one `q` per internal node.
///
/// Siblings share a `q` because they are i.i.d. draws from it, which is both what
/// the multi-candidate acceptance rule is stated for and why the bookkeeping is
/// per *parent* rather than per node.
///
/// `parent[i]` indexes `nodes`, or `None` for a level-one node whose parent is
/// the root — the position the target has already reached. `q_of[i]` indexes
/// `qs`: the distribution node `i` was drawn from, shared with its siblings.
pub struct TreeDraft {
    /// Breadth-first: level one first, then level two, and so on.
    pub nodes: Vec<TreeNode>,
    /// One per internal node, the root's first.
    pub qs: Vec<Vec<(u32, f32)>>,
    /// Candidates a node fans out to, one entry a level. `[2, 1]` is two
    /// candidates for the next token and one after that, which is the shape a
    /// verification pass can afford: two root-to-leaf paths of three rows each,
    /// six in all, against the eight where the FP8 kernel's cost steps up.
    pub widths: Vec<usize>,
    /// Root-to-leaf paths, and so sequences a verification pass needs.
    pub leaves: usize,
}

pub struct TreeNode {
    pub token: u32,
    /// Index into `TreeDraft::nodes`, or `None` for a child of the root.
    pub parent: Option<usize>,
    /// Index into `TreeDraft::qs`.
    pub q_of: usize,
    /// The drafter row this node's own output landed in, which its children
    /// continue from.
    pub row: usize,
    /// Which forked branch of the drafter's cache this node lives on.
    pub lane: usize,
}

impl TreeDraft {
    /// The path from the root down to `node`, tokens in order.
    pub fn path(&self, node: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let mut cur = Some(node);
        while let Some(i) = cur {
            out.push(self.nodes[i].token);
            cur = self.nodes[i].parent;
        }
        out.reverse();
        out
    }
}

/// One drafted token and the distribution it came from.
///
/// The distribution is here rather than a single probability because the
/// acceptance rule's residual is `(p_target - q)+` at every token. Carrying only
/// `q(token)` and subtracting it at that one place treats a sampled draft as a
/// point mass; the composition then does not reproduce the target's
/// distribution, and the error is large — a hundred standard errors on the
/// largest bin in simulation, not a rounding difference.
pub struct Drafted {
    pub token: u32,
    /// The drafter's normalized distribution over its truncated support.
    pub q: Vec<(u32, f32)>,
}

impl crate::Model {
    /// Load the checkpoint's MTP head and keep it beside the text model.
    ///
    /// Returns false when the checkpoint has no head, which is not an error — it
    /// is most of them.
    ///
    /// `max_draft_rows` bounds **one chunk**, not the feed: priming after a
    /// prefill covers the whole prompt, and [`MtpHead::prime`] splits it. It has
    /// to be at least `k + 1`, the width of a verification round's feed, since
    /// that one is not split usefully — and larger only trades launches for the
    /// `heads * rows * max_seq` score buffer.
    pub fn load_mtp_head(
        &mut self,
        dir: impl AsRef<std::path::Path>,
        max_draft_rows: usize,
    ) -> anyhow::Result<bool> {
        let shards = infero_safetensors::Shards::open_dir(dir.as_ref())?;
        let Some(w) = crate::weights::load_mtp(&self.dev, &shards, &self.cfg)? else {
            return Ok(false);
        };
        let head = MtpHead::new(
            &self.dev,
            w,
            HeadDims::from_config(&self.cfg),
            max_draft_rows.max(1),
            self.max_seq,
            // One branch for now: the tree draft sets this from its widest
            // level once `draft_tree` exists. See `MtpHead::fork`.
            1,
        )?;
        self.install_mtp_head(head)?;
        Ok(true)
    }

    /// The same, from llama.cpp's sidecar GGUF.
    ///
    /// Separate from [`Self::load_mtp_head`] rather than a branch inside it
    /// because the two take different things: that one a checkpoint *directory*
    /// whose head sits among the text model's shards, this one the path to a
    /// second file. llama.cpp ships the head as `mtp-<model>.gguf`, a standalone
    /// 65-block model, and the text model's own file stops at `blk.63`.
    ///
    /// The head is loaded against `self.cfg`, which came from the text model:
    /// the two files agree on every shape that matters here, and requiring them
    /// to is the point — a sidecar built for a different `d_model` should fail
    /// loudly at `install_mtp_head` rather than draft nonsense.
    pub fn load_mtp_head_gguf(
        &mut self,
        path: impl AsRef<std::path::Path>,
        max_draft_rows: usize,
    ) -> anyhow::Result<bool> {
        let f = infero_gguf::Gguf::open(path.as_ref())?;
        let Some(w) = crate::weights::load_mtp_gguf(&self.dev, &f, &self.cfg)? else {
            return Ok(false);
        };
        let head = MtpHead::new(
            &self.dev,
            w,
            HeadDims::from_config(&self.cfg),
            max_draft_rows.max(1),
            self.max_seq,
            1,
        )?;
        self.install_mtp_head(head)?;
        Ok(true)
    }

    /// Attach a head built elsewhere — the path a test with synthetic weights
    /// takes, and the reason [`MtpHead`] does not reach into `Model` itself.
    pub fn install_mtp_head(&mut self, head: MtpHead) -> anyhow::Result<()> {
        anyhow::ensure!(
            head.dims().d_model == self.cfg.d_model,
            "the head is {} wide and the text model {}",
            head.dims().d_model,
            self.cfg.d_model
        );
        tracing::info!(mib = head.bytes() >> 20, "mtp head installed");
        self.mtp = Some(head);
        // The head's second input, captured from every pass — one row per token,
        // not per sampled row, because the drafter needs a history to attend to.
        self.mtp_hidden = Some(
            self.dev
                .stream()
                .alloc_zeros::<f32>(self.batch_tokens() * self.cfg.d_model)?,
        );
        Ok(())
    }

    pub fn has_mtp_head(&self) -> bool {
        self.mtp.is_some()
    }

    /// The head, for a caller that wants to drive it directly.
    pub fn mtp_head_mut(&mut self) -> Option<&mut MtpHead> {
        self.mtp.as_mut()
    }

    pub fn mtp_head(&self) -> Option<&MtpHead> {
        self.mtp.as_ref()
    }

    /// Draft `k` tokens with the head, from the hidden states the last forward
    /// pass produced.
    ///
    /// `feed` comes out of [`crate::spec::SpecOutcome`], or from
    /// [`crate::spec::DraftFeed::after_prefill`] for the first round. Steps after
    /// the first feed the head its own output and add one to the position, which
    /// is what makes `k > 1` possible from a one-layer head.
    /// A tree of drafts: `branch` candidates a node, `depth` levels deep.
    ///
    /// One drafter pass a level, not one a branch. A level of `B^L` rows costs
    /// what one row costs — the pass reads the head's weights either way and rows
    /// are nearly free — so a two-wide three-deep tree is the price of a linear
    /// draft of three, and offers fourteen candidates instead of three.
    ///
    /// Siblings are i.i.d. draws from their parent's own distribution, which is
    /// what [`crate::Model::accept_multi`] is stated for. The `q` is kept once a
    /// parent rather than once a node for the same reason.
    ///
    /// `branch = 1` is the linear draft, and produces the same tokens from the
    /// same seed as [`Self::draft_with_head_sampled`] does — the head consumes
    /// one uniform a candidate either way.
    pub fn draft_tree(
        &mut self,
        widths: &[usize],
        feed: &crate::spec::DraftFeed,
        sampler: &mut crate::Sampler,
        history: &[u32],
    ) -> anyhow::Result<TreeDraft> {
        let depth = widths.len();
        anyhow::ensure!(depth > 0, "a tree of no levels");
        anyhow::ensure!(widths.iter().all(|w| *w > 0), "a level of no candidates");
        let d = self.cfg.d_model;
        let hidden_rows = feed.rows.clone();
        let (positions, shifted_ids) = (&feed.positions, &feed.shifted);
        let rows = hidden_rows.len();
        anyhow::ensure!(
            rows == positions.len() && rows == shifted_ids.len(),
            "{rows} hidden rows against {} positions and {} ids",
            positions.len(),
            shifted_ids.len()
        );
        // Leaves, and so the lanes the head had to be forked for. Also the
        // sequences a verification pass will need, since every root-to-leaf path
        // is its own — the recurrence cannot fork.
        let leaves: usize = widths.iter().product();
        // The lanes in flight at the deepest level the drafter runs, which is one
        // level short of the leaves.
        let widest: usize = widths[..depth - 1].iter().product::<usize>().max(1);
        let mut head = self
            .mtp
            .take()
            .context("this model has no MTP head; call load_mtp_head first")?;
        let res = (|| -> anyhow::Result<TreeDraft> {
            anyhow::ensure!(
                head.branch_count() >= widest,
                "a tree of {widths:?} needs {widest} lanes, the head was built \
                 with {}",
                head.branch_count()
            );
            let hidden = self
                .mtp_hidden
                .as_ref()
                .context("no captured hidden states")?
                .slice(hidden_rows.start * d..hidden_rows.end * d);
            head.truncate(positions[0]);
            let root_row =
                head.prime(&self.kern, &self.w.token_embd, shifted_ids, positions, &hidden, feed.mrope.as_deref())?;
            let lm = self.w.output.as_ref().unwrap_or(&self.w.token_embd);
            let base = positions[rows - 1] + 1;
            // Every lane shares the prefix and owns `depth` slots past it, which
            // is as far as any branch reaches.
            head.fork(base, depth)?;

            let mut nodes: Vec<TreeNode> = Vec::new();
            let mut qs: Vec<Vec<(u32, f32)>> = Vec::new();

            // Level one: the root's own row, `branch` draws from one `q`.
            let (cands, q) = head.draft_row_candidates(
                &self.kern,
                lm,
                root_row,
                widths[0],
                sampler,
                history,
            )?;
            qs.push(q);
            for (lane, tok) in cands.into_iter().enumerate() {
                nodes.push(TreeNode {
                    token: tok,
                    parent: None,
                    q_of: 0,
                    // Filled in when the level runs; the root's children have not
                    // been through the drafter yet.
                    row: usize::MAX,
                    lane,
                });
            }

            // Each further level: one `step_tree` for the whole level, then one
            // `q` a node and `branch` draws from it.
            let mut level: Vec<usize> = (0..nodes.len()).collect();
            for l in 1..depth {
                let tokens: Vec<u32> = level.iter().map(|i| nodes[*i].token).collect();
                let src_rows: Vec<usize> = level
                    .iter()
                    .map(|i| match nodes[*i].parent {
                        Some(p) => nodes[p].row,
                        None => root_row,
                    })
                    .collect();
                let lanes: Vec<usize> = level.iter().map(|i| nodes[*i].lane).collect();
                let pos = vec![base + l - 1; level.len()];
                head.step_tree(
                    &self.kern,
                    &self.w.token_embd,
                    &tokens,
                    &pos,
                    &src_rows,
                    &lanes,
                    depth,
                )?;
                // `step_tree` lays the level out in the order it was given, so
                // node `level[j]`'s output is row `j`.
                for (j, i) in level.iter().enumerate() {
                    nodes[*i].row = j;
                }

                let mut next: Vec<usize> = Vec::new();
                for (j, i) in level.iter().enumerate() {
                    // The window a node's distribution is penalized against is
                    // its own path, not the tree's — a sibling's tokens are a
                    // different continuation entirely.
                    let mut window = history.to_vec();
                    let mut path = Vec::new();
                    let mut cur = Some(*i);
                    while let Some(c) = cur {
                        path.push(nodes[c].token);
                        cur = nodes[c].parent;
                    }
                    path.reverse();
                    window.extend_from_slice(&path);

                    let (cands, q) = head.draft_row_candidates(
                        &self.kern,
                        lm,
                        j,
                        widths[l],
                        sampler,
                        &window,
                    )?;
                    let q_of = qs.len();
                    qs.push(q);
                    for (b, tok) in cands.into_iter().enumerate() {
                        next.push(nodes.len());
                        nodes.push(TreeNode {
                            token: tok,
                            parent: Some(*i),
                            q_of,
                            row: usize::MAX,
                            // A child stays on its parent's lane at width one and
                            // fans out otherwise, so that every root-to-leaf path
                            // has a lane to itself at the widest level.
                            lane: nodes[*i].lane * widths[l] + b,
                        });
                    }
                }
                level = next;
            }
            Ok(TreeDraft { nodes, qs, widths: widths.to_vec(), leaves })
        })();
        self.mtp = Some(head);
        res
    }

    /// A draft plus the probability the drafter gave each token.
    ///
    /// The sampled counterpart of [`Model::draft_with_head`]. The history grows
    /// as the draft does: token `j`'s distribution is conditioned on the drafts
    /// before it, and the repetition penalty reads that window, so passing a
    /// stale history would score every draft against the wrong distribution.
    pub fn draft_with_head_sampled(
        &mut self,
        k: usize,
        feed: &crate::spec::DraftFeed,
        sampler: &mut crate::Sampler,
        history: &[u32],
    ) -> anyhow::Result<Vec<Drafted>> {
        anyhow::ensure!(k > 0, "a draft of no tokens");
        let d = self.cfg.d_model;
        let hidden_rows = feed.rows.clone();
        let (positions, shifted_ids) = (&feed.positions, &feed.shifted);
        let rows = hidden_rows.len();
        anyhow::ensure!(
            rows == positions.len() && rows == shifted_ids.len(),
            "{rows} hidden rows against {} positions and {} ids",
            positions.len(),
            shifted_ids.len()
        );
        let mut head = self
            .mtp
            .take()
            .context("this model has no MTP head; call load_mtp_head first")?;
        let res = (|| -> anyhow::Result<Vec<Drafted>> {
            let hidden = self
                .mtp_hidden
                .as_ref()
                .context("no captured hidden states")?
                .slice(hidden_rows.start * d..hidden_rows.end * d);
            head.truncate(positions[0]);
            // `prime`, not `step`: after a prefill the feed is the whole prompt
            // and the head is built for one draft step's width. The row it
            // returns is the final token's index within the last chunk.
            let mut row =
                head.prime(&self.kern, &self.w.token_embd, shifted_ids, positions, &hidden, feed.mrope.as_deref())?;
            let lm = self.w.output.as_ref().unwrap_or(&self.w.token_embd);
            let mut drafted = Vec::with_capacity(k);
            // The window the repetition penalty reads, extended per draft.
            let mut window: Vec<u32> = history.to_vec();
            let mut position = positions[rows - 1];
            let (mut token, mut q) =
                head.draft_row_sampled(&self.kern, lm, row, sampler, &window)?;
            drafted.push(Drafted { token, q });
            for _ in 1..k {
                window.push(token);
                position += 1;
                head.step_from_own_output(&self.kern, &self.w.token_embd, token, position, row)?;
                row = 0;
                (token, q) = head.draft_row_sampled(&self.kern, lm, row, sampler, &window)?;
                drafted.push(Drafted { token, q });
            }
            Ok(drafted)
        })();
        self.mtp = Some(head);
        res
    }

    pub fn draft_with_head(
        &mut self,
        k: usize,
        feed: &crate::spec::DraftFeed,
    ) -> anyhow::Result<Vec<u32>> {
        anyhow::ensure!(k > 0, "a draft of no tokens");
        let d = self.cfg.d_model;
        let hidden_rows = feed.rows.clone();
        let (positions, shifted_ids) = (&feed.positions, &feed.shifted);
        let rows = hidden_rows.len();
        anyhow::ensure!(
            rows == positions.len() && rows == shifted_ids.len(),
            "{rows} hidden rows against {} positions and {} ids",
            positions.len(),
            shifted_ids.len()
        );
        let mut head = self
            .mtp
            .take()
            .context("this model has no MTP head; call load_mtp_head first")?;
        let res = (|| -> anyhow::Result<Vec<u32>> {
            let hidden = self
                .mtp_hidden
                .as_ref()
                .context("no captured hidden states")?
                .slice(hidden_rows.start * d..hidden_rows.end * d);
            // Roll the drafter's own cache back to the confirmed extent before
            // re-feeding it. The rejected tokens of the previous round were
            // written at positions at or after this one; they are about to be
            // overwritten, and the causal mask ignores anything past the last row
            // regardless, but the length has to come back or `kv_len` grows
            // without bound. This is the drafter's coordinate system, one behind
            // the target's — see `MtpHead::truncate`.
            head.truncate(positions[0]);
            let mut row =
                head.prime(&self.kern, &self.w.token_embd, shifted_ids, positions, &hidden, feed.mrope.as_deref())?;
            let lm = self.w.output.as_ref().unwrap_or(&self.w.token_embd);
            let mut drafted = Vec::with_capacity(k);
            let mut position = positions[rows - 1];
            let mut token = head.draft_row(&self.kern, lm, row)?;
            drafted.push(token);
            for _ in 1..k {
                position += 1;
                head.step_from_own_output(&self.kern, &self.w.token_embd, token, position, row)?;
                row = 0;
                token = head.draft_row(&self.kern, lm, row)?;
                drafted.push(token);
            }
            Ok(drafted)
        })();
        self.mtp = Some(head);
        res
    }
}

/// `out[t, :] = w · x[t, :]` for the head's f16 matrices.
///
/// A local copy of the one decision [`crate::Model::matmul_pre`] makes for this
/// weight type — the mat-vec for a handful of rows, cuBLAS above that — rather
/// than a call into it, because the head's activations are its own and threading
/// `Model`'s scratch through here would tie the drafter to the text model's
/// buffers for no gain.
fn matmul(
    kern: &Kernels,
    out: &mut ViewMut<'_, f32>,
    w: &Matrix,
    q8_1: &mut Buf<u8>,
    x16: &mut Buf<f16>,
    x: &View<'_, f32>,
    n_tokens: usize,
) -> Result<()> {
    anyhow::ensure!(
        x.len() >= n_tokens * w.k,
        "a matmul over {n_tokens} rows of {} wants {} floats, the activation \
         holds {}",
        w.k,
        n_tokens * w.k,
        x.len()
    );
    let weights = w.view(None)?;
    // One draft step is one token, and a Q8_0 head is bandwidth-bound there
    // exactly like the vocabulary mat-vec below is -- same kernel, same
    // reason. Going through `gemv` here left a k=4 verify round's draft phase
    // at 9.36 ms; `mmvq` is what the vocabulary step already takes for the
    // identical reason ("Going through `gemv` here left the draft at 4.65 ms
    // against a 1.68 ms byte bound"), just never extended to the head's own
    // seven projections. `q8_1` is sized for the widest single row across all
    // of them, not just `d_model`, so this covers `w_down`'s `d_ff` and the
    // FC's `2 * d_model` too.
    if n_tokens == 1 && Kernels::has_mmvq(w.ty) && w.k.is_multiple_of(32) {
        let bytes = Kernels::q8_1_bytes(w.k);
        kern.quantize_q8_1(&mut q8_1.slice_mut(..bytes), x, w.k)?;
        return kern.mmvq(out, &weights, w.ty, &q8_1.slice(..bytes), w.k, w.n);
    }
    // Block-scaled FP8, which is what the head's seven projections are in this
    // checkpoint. Same two paths `Model::matmul_pre` takes and for the same
    // reasons: at one token a mat-vec is the right kernel shape, and at a few
    // tokens reading each weight once beats expanding the matrix.
    if w.ty == infero_kernels::WeightType::F8E4M3 {
        // Tensor cores whenever they will take the shape, which is every token
        // count up to eight and any `k` that is a multiple of the scale block.
        // The table is on the same branch in `Model::matmul_pre`.
        if kern.mma_f8_block(out, &weights, x, w.k, w.n, n_tokens, false)? {
            return Ok(());
        }
        if n_tokens == 1 {
            return kern.mmv_f8_block(out, &weights, x, w.k, w.n, false);
        }
        if kern.mmv_f8_block_batch(out, &weights, x, w.k, w.n, n_tokens, false)? {
            return Ok(());
        }
        // Wider than the batched mat-vec goes: expand and use the GEMM, which
        // is what a prime over a whole prompt does.
        let n_x = n_tokens * w.k;
        kern.to_f16(&mut x16.slice_mut(..n_x), x, n_x)?;
        let mut w16 = kern
            .device()
            .stream()
            .alloc_zeros::<f16>(w.elements())
            .context("staging the head's FP8 matrix as f16")?;
        kern.dequant_f8_block_to_f16(&mut w16.as_view_mut(), &weights, w.k, w.n)?;
        return kern.gemm_f16(out, &x16.slice(..n_x), &w16.as_view(), n_tokens, w.k, w.n);
    }
    // A quantized head, which is what the GGUF sidecar ships: `blk.64`'s seven
    // projections are Q8_0. There is no GEMM at that layout, and there does not
    // need to be — `gemv` tiles its rows eight to a block along `grid.y`, so it
    // takes a whole prime's feed as readily as one draft step's. Before this the
    // wide path asserted f16 and a Q8_0 head could not be primed at all.
    if n_tokens <= 4 || w.ty != WeightType::F16 {
        return kern.gemv(out, &weights, w.ty, x, w.k, w.n, n_tokens);
    }
    let n_x = n_tokens * w.k;
    kern.to_f16(&mut x16.slice_mut(..n_x), x, n_x)?;
    anyhow::ensure!(
        w.ty == WeightType::F16,
        "the MTP head expects f16 matrices; {:?} has no GEMM here",
        w.ty
    );
    // Safety: the range holds exactly `k * n` f16 values written by the loader,
    // and f16 has no invalid bit patterns.
    let view = unsafe { weights.transmute::<f16>(w.elements()) }
        .context("an f16 weight buffer is misaligned")?;
    kern.gemm_f16(out, &x16.slice(..n_x), &view, n_tokens, w.k, w.n)
}

/// The first index of the largest value, which is the tie-break a greedy sampler
/// has to agree on for speculation to be exact.
pub fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// One row's device-sampler state for a draft.
///
/// The draft used to sample on the host: 993 KB of logits copied back per draft
/// token and a walk over the whole vocabulary in Rust, measured at 0.709 ms of a
/// 2.249 ms draft with the wall clock and the kernel sum taken in the same run.
/// At `k = 2` that is 1.42 ms a round.
///
/// Doing it on the device needed two things. `sample_rows_f32` already had the
/// token and the nucleus it drew from, so it now writes them out; and its top-k
/// scanned the vocabulary once per survivor, which is forty passes over 248320
/// tokens and measured 5.99 ms — hence `sample_rows_split`, whose candidate
/// buffers are what `cand_v`/`cand_i` are.
struct DraftSampleBufs {
    params: Buf<f32>,
    pen_tok: Buf<i32>,
    pen_cnt: Buf<i32>,
    pen_len: Buf<i32>,
    rnd: Buf<f64>,
    out: Buf<u32>,
    cand_v: Buf<f32>,
    cand_i: Buf<i32>,
    surv_id: Buf<u32>,
    surv_p: Buf<f32>,
    surv_len: Buf<i32>,
    /// Survivor entries and the penalty window's pitch.
    stride: usize,
}
