//! Launchers for the GatedDeltaNet block and the gated-attention output gate.
//!
//! In a file of their own so that this work and the rotary work do not collide
//! in `lib.rs`; the kernels themselves are in `cu/gdn.cu`.
//!
//! The one thing worth reading before using any of these: the recurrent state
//! is *in-place and persistent*. Every other buffer in this engine is either an
//! activation that dies at the end of the step or an append-only cache. A
//! GatedDeltaNet state is read and rewritten by the same launch, carries across
//! steps, and is per sequence. That has three consequences the callers have to
//! respect — the state buffer's address must be stable across a CUDA graph
//! capture and its replays, a sequence's tokens must reach the kernel
//! contiguous and in order, and anything that rewinds the sequence (a rejected
//! speculative draft, a cancelled request) has to restore the state rather than
//! just moving a length counter.

use anyhow::{Context, Result};
use cudarc::driver::{CudaView, CudaViewMut, LaunchConfig, PushKernelArg};

use crate::{Kernels, REDUCE_BLOCK, gdn_src};

/// How the tokens of a batch map onto sequences.
///
/// The GatedDeltaNet kernels need this and the attention kernels do not: a
/// state update is sequential within a sequence, so the kernel walks a
/// sequence's tokens in order and cannot be handed an arbitrary permutation.
/// `first_token[s]` and `n_tokens[s]` are device arrays because the kernel
/// indexes with them; keeping them on the host would mean one launch per
/// sequence.
pub struct SeqLayout<'a> {
    pub first_token: &'a CudaView<'a, i32>,
    pub n_tokens: &'a CudaView<'a, i32>,
    pub n_seqs: usize,
    pub total_tokens: usize,
}

impl Kernels {
    /// Depthwise causal convolution with a carried window, plus SiLU.
    ///
    /// `x` and `out` are `[total_tokens, channels]`; `state` is
    /// `[n_seqs, channels, k - 1]`, oldest tap first, and is advanced in place.
    /// `w` is `[channels, k]`.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_conv(
        &self,
        out: &mut CudaViewMut<'_, f32>,
        x: &CudaView<'_, f32>,
        state: &mut CudaViewMut<'_, f32>,
        w: &CudaView<'_, f32>,
        seqs: &SeqLayout<'_>,
        channels: usize,
        k: usize,
    ) -> Result<()> {
        // `win[8]` in the kernel bounds the carried window; the checkpoint uses
        // k = 4. Refuse rather than overrun.
        anyhow::ensure!(
            (2..=8).contains(&k),
            "conv kernel width {k} is outside the range the kernel's register \
             window covers (2..=8)"
        );
        debug_assert!(out.len() >= seqs.total_tokens * channels);
        debug_assert!(x.len() >= seqs.total_tokens * channels);
        debug_assert!(state.len() >= seqs.n_seqs * channels * (k - 1));
        debug_assert!(w.len() >= channels * k);

        let f = self.dev.kernels().get("tuili_gdn", gdn_src(), "gdn_conv_f32")?;
        const BLOCK: u32 = 128;
        let cfg = LaunchConfig {
            grid_dim: (
                (channels as u32).div_ceil(BLOCK),
                seqs.n_seqs as u32,
                1,
            ),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (c, kk) = (channels as i32, k as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out)
            .arg(x)
            .arg(state)
            .arg(w)
            .arg(seqs.first_token)
            .arg(seqs.n_tokens)
            .arg(&c)
            .arg(&kk);
        self.dev.profile().time("gdn_conv", self.dev.stream(), || {
            unsafe { b.launch(cfg) }.context("gdn_conv")?;
            Ok(())
        })?;
        Ok(())
    }

    /// `beta = sigmoid(b)` and `g = -exp(A_log) * softplus(a + dt_bias)`.
    ///
    /// `a` and `b` are `[n_tokens, heads]`; `a_log` and `dt_bias` are `[heads]`.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_gate_decay(
        &self,
        beta: &mut CudaViewMut<'_, f32>,
        g: &mut CudaViewMut<'_, f32>,
        a: &CudaView<'_, f32>,
        b_in: &CudaView<'_, f32>,
        a_log: &CudaView<'_, f32>,
        dt_bias: &CudaView<'_, f32>,
        n_tokens: usize,
        heads: usize,
    ) -> Result<()> {
        let n = n_tokens * heads;
        debug_assert!(beta.len() >= n && g.len() >= n);
        debug_assert!(a.len() >= n && b_in.len() >= n);
        debug_assert!(a_log.len() >= heads && dt_bias.len() >= heads);

        let f = self
            .dev
            .kernels()
            .get("tuili_gdn", gdn_src(), "gdn_gate_decay_f32")?;
        const BLOCK: u32 = 256;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (nt, h) = (n_tokens as i32, heads as i32);
        let mut bl = self.dev.stream().launch_builder(&f);
        bl.arg(beta)
            .arg(g)
            .arg(a)
            .arg(b_in)
            .arg(a_log)
            .arg(dt_bias)
            .arg(&nt)
            .arg(&h);
        self.dev
            .profile()
            .time("gdn_gate_decay", self.dev.stream(), || {
                unsafe { bl.launch(cfg) }.context("gdn_gate_decay")?;
                Ok(())
            })?;
        Ok(())
    }

    /// L2-normalize each key head's row of `q` and `k` in place, scaling `q`
    /// by `1/sqrt(dk)`.
    ///
    /// `qkv` is `[n_tokens, stride]` — the packed row the input projection
    /// produced — and `q_off`/`k_off` locate q and k inside it. The scale
    /// belongs to `q` alone; putting it on both, or on neither and folding it
    /// into the readout, changes the result.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_qk_l2norm(
        &self,
        qkv: &mut CudaViewMut<'_, f32>,
        n_tokens: usize,
        key_heads: usize,
        dk: usize,
        stride: usize,
        q_off: usize,
        k_off: usize,
        eps: f32,
    ) -> Result<()> {
        debug_assert!(qkv.len() >= n_tokens * stride);
        debug_assert!(q_off + key_heads * dk <= stride);
        debug_assert!(k_off + key_heads * dk <= stride);
        let f = self
            .dev
            .kernels()
            .get("tuili_gdn", gdn_src(), "gdn_qk_l2norm_f32")?;
        let cfg = LaunchConfig {
            grid_dim: ((n_tokens * key_heads) as u32, 1, 1),
            block_dim: (REDUCE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (kh, d) = (key_heads as i32, dk as i32);
        let (st, qo, ko) = (stride as i32, q_off as i32, k_off as i32);
        let scale = (dk as f32).sqrt().recip();
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(qkv)
            .arg(&kh)
            .arg(&d)
            .arg(&st)
            .arg(&qo)
            .arg(&ko)
            .arg(&eps)
            .arg(&scale);
        self.dev
            .profile()
            .time("gdn_qk_l2norm", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("gdn_qk_l2norm")?;
                Ok(())
            })?;
        Ok(())
    }

    /// The gated delta rule, advancing `state` in place.
    ///
    /// `qkv` is `[total_tokens, stride]` holding q, k and v at `q_off`, `k_off`
    /// and `v_off`, with q and k already normalized and q already scaled. `out`
    /// is `[total_tokens, heads, dv]`. `g` and `beta` are
    /// `[total_tokens, heads]`. `state` is `[n_seqs, heads, dk, dv]`.
    ///
    /// `key_heads` may be smaller than `heads`; the kernel expands them the way
    /// `repeat_interleave` does, so value head `h` reads key head
    /// `h / (heads / key_heads)`.
    ///
    /// One block a (head, sequence) pair, `dv` threads. `dv` past 1024 would
    /// need a second dimension of work per thread; the checkpoint uses 128.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_delta_rule(
        &self,
        out: &mut CudaViewMut<'_, f32>,
        state: &mut CudaViewMut<'_, f32>,
        qkv: &CudaView<'_, f32>,
        g: &CudaView<'_, f32>,
        beta: &CudaView<'_, f32>,
        seqs: &SeqLayout<'_>,
        heads: usize,
        key_heads: usize,
        dk: usize,
        dv: usize,
        offsets: (usize, usize, usize, usize),
    ) -> Result<()> {
        let (stride, q_off, k_off, v_off) = offsets;
        anyhow::ensure!(
            dv <= 1024,
            "the delta-rule kernel gives each of dv threads one column of the \
             state; dv is {dv}, past the 1024-thread block limit"
        );
        anyhow::ensure!(
            key_heads > 0 && heads.is_multiple_of(key_heads),
            "{heads} value heads do not divide into {key_heads} key heads, so \
             the repeat_interleave expansion is not defined"
        );
        let t = seqs.total_tokens;
        debug_assert!(out.len() >= t * heads * dv);
        debug_assert!(state.len() >= seqs.n_seqs * heads * dk * dv);
        debug_assert!(qkv.len() >= t * stride);
        debug_assert!(q_off + key_heads * dk <= stride);
        debug_assert!(k_off + key_heads * dk <= stride);
        debug_assert!(v_off + heads * dv <= stride);
        debug_assert!(g.len() >= t * heads && beta.len() >= t * heads);

        let f = self
            .dev
            .kernels()
            .get("tuili_gdn", gdn_src(), "gdn_delta_rule_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (heads as u32, seqs.n_seqs as u32, 1),
            block_dim: (dv.max(32) as u32, 1, 1),
            // q and k for the current token, shared by every thread.
            shared_mem_bytes: (2 * dk * std::mem::size_of::<f32>()) as u32,
        };
        let (h, kh) = (heads as i32, key_heads as i32);
        let (a, b_) = (dk as i32, dv as i32);
        let (st, qo, ko, vo) = (stride as i32, q_off as i32, k_off as i32, v_off as i32);
        let mut bl = self.dev.stream().launch_builder(&f);
        bl.arg(out)
            .arg(state)
            .arg(qkv)
            .arg(g)
            .arg(beta)
            .arg(seqs.first_token)
            .arg(seqs.n_tokens)
            .arg(&h)
            .arg(&kh)
            .arg(&a)
            .arg(&b_)
            .arg(&st)
            .arg(&qo)
            .arg(&ko)
            .arg(&vo);
        self.dev
            .profile()
            .time("gdn_delta_rule", self.dev.stream(), || {
                unsafe { bl.launch(cfg) }.context("gdn_delta_rule")?;
                Ok(())
            })?;
        Ok(())
    }

    /// `out = rms_norm(x, weight) * silu(z)`, over rows of `dv`.
    ///
    /// Normalize first, gate second. The other order runs and is a different
    /// model.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_gated_rmsnorm(
        &self,
        out: &mut CudaViewMut<'_, f32>,
        x: &CudaView<'_, f32>,
        z: &CudaView<'_, f32>,
        weight: &CudaView<'_, f32>,
        rows: usize,
        dv: usize,
        eps: f32,
    ) -> Result<()> {
        debug_assert!(out.len() >= rows * dv && x.len() >= rows * dv);
        debug_assert!(z.len() >= rows * dv && weight.len() >= dv);
        let f = self
            .dev
            .kernels()
            .get("tuili_gdn", gdn_src(), "gdn_gated_rmsnorm_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (REDUCE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let d = dv as i32;
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(x).arg(z).arg(weight).arg(&d).arg(&eps);
        self.dev
            .profile()
            .time("gdn_gated_rmsnorm", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("gdn_gated_rmsnorm")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Split a `[tokens, heads, 2 * head_dim]` projection into its value and
    /// its gate.
    ///
    /// `q` and `gate` are each `[tokens, heads, head_dim]`. The query and the
    /// gate interleave per head, so this is a strided gather rather than a
    /// split down the middle — the split down the middle also runs.
    pub fn split_interleaved(
        &self,
        q: &mut CudaViewMut<'_, f32>,
        gate: &mut CudaViewMut<'_, f32>,
        src: &CudaView<'_, f32>,
        n_tokens: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        let n = n_tokens * heads * head_dim;
        debug_assert!(q.len() >= n && gate.len() >= n);
        debug_assert!(src.len() >= 2 * n);
        let f = self
            .dev
            .kernels()
            .get("tuili_gdn", gdn_src(), "split_interleaved_f32")?;
        const BLOCK: u32 = 256;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (h, hd) = (heads as i32, head_dim as i32);
        let count = n as i64;
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(q).arg(gate).arg(src).arg(&h).arg(&hd).arg(&count);
        self.dev
            .profile()
            .time("split_interleaved", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("split_interleaved")?;
                Ok(())
            })?;
        Ok(())
    }

    /// `x *= sigmoid(gate)`, in place.
    ///
    /// The output gate of Qwen3.5's full-attention layers, applied before
    /// `o_proj`. Sigmoid, not silu — the reference implementation does not read
    /// config's `output_gate_type: "swish"`.
    pub fn sigmoid_gate(
        &self,
        x: &mut CudaViewMut<'_, f32>,
        gate: &CudaView<'_, f32>,
        n: usize,
    ) -> Result<()> {
        debug_assert!(x.len() >= n && gate.len() >= n);
        let f = self
            .dev
            .kernels()
            .get("tuili_gdn", gdn_src(), "sigmoid_gate_f32")?;
        const BLOCK: u32 = 256;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let count = n as i64;
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(x).arg(gate).arg(&count);
        self.dev
            .profile()
            .time("sigmoid_gate", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("sigmoid_gate")?;
                Ok(())
            })?;
        Ok(())
    }
}
