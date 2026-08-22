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
use cudarc::driver::{CudaSlice, CudaView, CudaViewMut};
use half::f16;
use tuili_cuda::Device;
use tuili_kernels::{AttnDims, BatchLayout, Kernels, WeightType};

use crate::weights::{Matrix, MtpWeights};

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
    emb: CudaSlice<f32>,
    /// `[max_tokens, d_model]` — whatever is playing the part of the hidden
    /// state this step: the text model's final hidden on the first draft step,
    /// the head's own output on the ones after it.
    hidden_in: CudaSlice<f32>,
    /// `[max_tokens, 2 * d_model]` — `[e | h]`, the only place the concat order
    /// is expressed.
    cat: CudaSlice<f32>,
    /// The residual stream.
    x: CudaSlice<f32>,
    xb: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    /// Doubles as the `q_proj` output (`2 * d_attn` wide) and the FFN gate.
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    ffn: CudaSlice<f32>,
    /// The attention output gate, de-interleaved out of `q_proj`'s output.
    attn_gate: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    scores: CudaSlice<f32>,
    proj: CudaSlice<f32>,
    /// `[max_tokens, d_model]` — the head's output, after `mtp.norm`. This is
    /// both what `lm_head` scores and what the next draft step feeds back in.
    out: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    x16: CudaSlice<f16>,

    ids: CudaSlice<i32>,
    positions: CudaSlice<i32>,
    slots: CudaSlice<i32>,
    seq_of: CudaSlice<i32>,
    /// All ones: the head rotates every pair at the base frequency.
    freqs: CudaSlice<f32>,

    /// The drafter's own single-sequence KV cache, `[kv_heads, max_seq, d_head]`.
    kc: CudaSlice<f16>,
    vc: CudaSlice<f16>,
    slot_table: CudaSlice<i32>,
    /// How far the drafter's cache reaches, in positions.
    len: usize,
    /// Rows the last [`MtpHead::step`] produced.
    rows: usize,
    logits_host: Vec<f32>,
}

impl MtpHead {
    /// `max_tokens` bounds one draft step's rows; `max_seq` the drafter's cache.
    pub fn new(
        dev: &Device,
        w: MtpWeights,
        dims: HeadDims,
        max_tokens: usize,
        max_seq: usize,
    ) -> Result<Self> {
        anyhow::ensure!(max_tokens > 0 && max_seq > 0, "an empty MTP head");
        let stream = dev.stream();
        let (d, da, dkv) = (dims.d_model, dims.d_attn(), dims.d_kv());
        let t = max_tokens;
        let alloc = |n: usize, what: &str| -> Result<CudaSlice<f32>> {
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
            logits: alloc(t * dims.vocab, "draft logits")?,
            x16: stream.alloc_zeros::<f16>(t * (2 * d).max(dims.d_ff).max(da))?,
            ids: stream.alloc_zeros::<i32>(t)?,
            positions: stream.alloc_zeros::<i32>(t)?,
            slots: stream.alloc_zeros::<i32>(t)?,
            seq_of: stream.alloc_zeros::<i32>(t)?,
            freqs: stream.clone_htod(&vec![1.0f32; dims.rotary_dim / 2])?,
            kc: stream.alloc_zeros::<f16>(dims.kv_heads * max_seq * dims.d_head)?,
            vc: stream.alloc_zeros::<f16>(dims.kv_heads * max_seq * dims.d_head)?,
            slot_table: stream.alloc_zeros::<i32>(max_seq)?,
            len: 0,
            rows: 0,
            w,
            logits_host: vec![0.0; dims.vocab],
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
    pub fn step(
        &mut self,
        kern: &Kernels,
        embed: &Matrix,
        shifted_ids: &[u32],
        positions: &[usize],
        hidden: &CudaView<'_, f32>,
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
        self.dev
            .stream()
            .memcpy_dtod(hidden, &mut self.hidden_in.slice_mut(..n * self.dims.d_model))?;
        self.run(kern, embed, shifted_ids, positions)
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
        self.run(kern, embed, &[drafted], &[position])
    }

    fn run(
        &mut self,
        kern: &Kernels,
        embed: &Matrix,
        shifted_ids: &[u32],
        positions: &[usize],
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
        for w in positions.windows(2) {
            anyhow::ensure!(
                w[1] == w[0] + 1,
                "the drafter's rows must be consecutive positions, got {w:?}"
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
        stream.memcpy_htod(&ids, &mut self.ids.slice_mut(..n))?;
        stream.memcpy_htod(&pos, &mut self.positions.slice_mut(..n))?;
        stream.memcpy_htod(&pos, &mut self.slots.slice_mut(..n))?;
        stream.memcpy_htod(&vec![0i32; n], &mut self.seq_of.slice_mut(..n))?;
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
            &mut self.x16,
            &self.xb.slice(..n * d),
            n,
        )?;
        matmul(
            kern,
            &mut self.v.slice_mut(..n * dkv),
            &l.wv,
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
            let (q, k) = (&mut self.q, &mut self.k);
            kern.rope_qk_partial(
                &mut q.slice_mut(..n * da),
                &mut k.slice_mut(..n * dkv),
                &self.positions.slice(..n),
                &self.freqs.as_view(),
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
            &self.w.layer.w_gate,
            &mut self.x16,
            &self.xb.slice(..n * d),
            n,
        )?;
        matmul(
            kern,
            &mut self.up.slice_mut(..n * dims.d_ff),
            &self.w.layer.w_up,
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
            &self.w.layer.w_down,
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
    pub fn output_row(&self, row: usize) -> CudaView<'_, f32> {
        let d = self.dims.d_model;
        self.out.slice(row * d..(row + 1) * d)
    }

    /// Every row the last step produced.
    pub fn output(&self) -> CudaView<'_, f32> {
        self.out.slice(..self.rows * self.dims.d_model)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// `head @ out[row]`, brought back to the host.
    ///
    /// `head` is the text model's `lm_head`. The head has none of its own —
    /// `tie_word_embeddings` is false on this checkpoint, so `lm_head` and the
    /// embedding are different tensors and the drafter wants the former.
    pub fn logits_row(&mut self, kern: &Kernels, head: &Matrix, row: usize) -> Result<&[f32]> {
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
            matmul(kern, &mut out, head, &mut self.x16, &src, 1)?;
        }
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
}

impl crate::Model {
    /// Load the checkpoint's MTP head and keep it beside the text model.
    ///
    /// Returns false when the checkpoint has no head, which is not an error — it
    /// is most of them. `max_draft_rows` bounds one draft step's width: the first
    /// step of a round covers every token the verification pass confirmed, so
    /// `k + 1` is enough for speculative decoding and more only helps a caller
    /// that wants to prime the drafter over a whole prompt.
    pub fn load_mtp_head(
        &mut self,
        dir: impl AsRef<std::path::Path>,
        max_draft_rows: usize,
    ) -> anyhow::Result<bool> {
        let shards = tuili_safetensors::Shards::open_dir(dir.as_ref())?;
        let Some(w) = crate::weights::load_mtp(&self.dev, &shards, &self.cfg)? else {
            return Ok(false);
        };
        let head = MtpHead::new(
            &self.dev,
            w,
            HeadDims::from_config(&self.cfg),
            max_draft_rows.max(1),
            self.max_seq,

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
                .alloc_zeros::<f32>(crate::MAX_BATCH_TOKENS * self.cfg.d_model)?,
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

    /// Draft `k` tokens with the head, from the hidden states the last forward
    /// pass produced.
    ///
    /// `feed` comes out of [`crate::spec::SpecOutcome`], or from
    /// [`crate::spec::DraftFeed::after_prefill`] for the first round. Steps after
    /// the first feed the head its own output and add one to the position, which
    /// is what makes `k > 1` possible from a one-layer head.
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
            head.step(&self.kern, &self.w.token_embd, shifted_ids, positions, &hidden)?;
            let lm = self.w.output.as_ref().unwrap_or(&self.w.token_embd);
            let mut drafted = Vec::with_capacity(k);
            let mut row = rows - 1;
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
/// buffers for no gain. Everything the head owns is f16, so there is no integer
/// path to choose between.
fn matmul(
    kern: &Kernels,
    out: &mut CudaViewMut<'_, f32>,
    w: &Matrix,
    x16: &mut CudaSlice<f16>,
    x: &CudaView<'_, f32>,
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
    if n_tokens <= 4 {
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
