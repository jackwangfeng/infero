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
use tuili_gpu::{View, ViewMut, LaunchConfig, KernelArg};

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
    pub first_token: &'a View<'a, i32>,
    pub n_tokens: &'a View<'a, i32>,
    pub n_seqs: usize,
    pub total_tokens: usize,
}

/// Which delta-rule kernel to run.
///
/// The three differ only in where the recurrent state lives while a chunk of
/// tokens is being consumed, and that is the whole performance story: the state
/// is `dk * dv` f32 — 64 KiB a head at this checkpoint's 128 by 128 — and it
/// does not change size with the chunk, so a version that keeps it in global
/// memory rereads and rewrites all of it every token, forever.
///
/// Microseconds a launch at the 27B's shape — 48 value heads, 16 key heads,
/// `dk = dv = 128` — from `examples/gdn_delta_bench.rs`, on an RTX A4000
/// (sm_86, 48 SMs) and an RTX PRO 6000 Blackwell (sm_120, 188 SMs):
///
/// | | 1 token, 1 seq | 1 token, 32 seqs | 512 tokens, 1 seq |
/// |---|---|---|---|
/// | `Global` sm_86  | 73.4 | 1047 | 17814 |
/// | `Shared` sm_86  | 62.1 | 1757 |  3237 |
/// | `Reg` sm_86     | 18.6 |  522 |   588 |
/// | `Global` sm_120 | 75.6 |  180 | 21798 |
/// | `Shared` sm_120 | 63.8 |  560 |  3393 |
/// | `Reg` sm_120    |  8.0 |  137 |   378 |
///
/// The sm_86 32-sequence column is the clean one: `Global` and `Reg` both run
/// at ~388 GB/s, 87% of that card's peak, and the 2.0x is exactly the ratio of
/// the bytes they are obliged to move. The other columns are larger than 2x
/// because at 48 blocks `Global` cannot saturate DRAM and at 512 tokens it
/// streams the state 512 times — and the sm_120 32-sequence column is *smaller*
/// than 2x because that card's 128 MB L2 holds the 96 MiB state, so its rereads
/// never reach DRAM. See the note in `cu/gdn.cu` for the rest of that.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DeltaVariant {
    /// `Reg` when `dk == dv == 128`, `Global` otherwise. What every caller
    /// should use.
    Auto,
    /// The state stays in global memory and is streamed twice a token. Correct
    /// for any `dk`/`dv`, and the fallback for shapes `Reg` is not instantiated
    /// for. Only reachable by name now that the checkpoint's shape takes the
    /// fast path, which is why the tests ask for it explicitly.
    Global,
    /// The state lives in registers for the whole chunk — loaded once, stored
    /// once, two threads a column — so a token moves q, k, v and the output
    /// rather than the state. Requires `dk == dv == 128`, and is launched with
    /// `2 * dv` threads. See the note above `gdn_delta_rule_reg_body` for the
    /// four choices inside it and what each alternative measured.
    Reg,
    /// The state lives in dynamic shared memory, loaded once and stored once.
    /// Saves the same traffic as `Reg` and loses to it everywhere, and loses to
    /// `Global` at 32 sequences: `(dk * dv + 2 * dk) * 4` bytes a block is one
    /// resident block an SM, so every barrier stalls the whole SM. Needs the
    /// opt-in dynamic-shared attribute past 48 KiB, which at 128 by 128 it is.
    Shared,
}

impl DeltaVariant {
    /// `Auto` resolved against the head dims; the others pass through.
    fn resolve(self, dk: usize, dv: usize) -> Self {
        match self {
            // 128 is what `gdn_delta_rule_reg128_f32` is instantiated for. A
            // second instantiation is a one-line change, but every one costs
            // NVRTC time on a cold cache for a shape no checkpoint here uses.
            Self::Auto if dk == 128 && dv == 128 => Self::Reg,
            Self::Auto => Self::Global,
            other => other,
        }
    }
}

impl Kernels {
    /// Registers a thread, static shared bytes, and *spill* bytes a thread for
    /// one of the GatedDeltaNet kernels.
    ///
    /// The third number is the one that matters for the register-blocked delta
    /// rule: 128 floats of state a thread only live in registers if every loop
    /// over `dk` unrolls, and a dynamically indexed local array compiles fine,
    /// runs fine, and puts the state back in the DRAM the whole exercise was
    /// about. Non-zero here means the optimization did not happen.
    pub fn gdn_kernel_registers(&self, name: &str) -> Result<(i32, i32, i32)> {
        let f = self.dev.kernels().get("tuili_gdn", gdn_src(), name)?;
        Ok((f.num_regs()?, f.shared_size_bytes()?, f.local_size_bytes()?))
    }

    /// Blocks an SM the driver will make resident for a GatedDeltaNet kernel at
    /// a given block size and dynamic shared request.
    pub fn gdn_occupancy_blocks(&self, name: &str, threads: u32, dynamic: usize) -> Result<u32> {
        let f = self.dev.kernels().get("tuili_gdn", gdn_src(), name)?;
        if dynamic > 48 * 1024 {
            tuili_gpu::set_max_dynamic_shared(&f, dynamic as u32)?;
        }
        Ok(f.occupancy_max_active_blocks_per_multiprocessor(threads, dynamic, None)?)
    }

    /// Depthwise causal convolution with a carried window, plus SiLU.
    ///
    /// `x` and `out` are `[total_tokens, channels]`; `state` is
    /// `[n_seqs, channels, k - 1]`, oldest tap first, and is advanced in place.
    /// `w` is `[channels, k]`.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_conv(
        &self,
        out: &mut ViewMut<'_, f32>,
        x: &View<'_, f32>,
        state: &mut ViewMut<'_, f32>,
        w: &View<'_, f32>,
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
        beta: &mut ViewMut<'_, f32>,
        g: &mut ViewMut<'_, f32>,
        a: &View<'_, f32>,
        b_in: &View<'_, f32>,
        a_log: &View<'_, f32>,
        dt_bias: &View<'_, f32>,
        n_tokens: usize,
        heads: usize,
        // `stride` is the row pitch of `a` and `b`: `heads` when they are their
        // own buffers, `2 * heads` when they are halves of one stacked
        // projection. See the kernel.
        stride: usize,
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
        let (nt, h, st) = (n_tokens as i32, heads as i32, stride as i32);
        let mut bl = self.dev.stream().launch_builder(&f);
        bl.arg(beta)
            .arg(g)
            .arg(a)
            .arg(b_in)
            .arg(a_log)
            .arg(dt_bias)
            .arg(&nt)
            .arg(&h)
            .arg(&st);
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
        qkv: &mut ViewMut<'_, f32>,
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
    /// One block a (head, sequence) pair. The fallback gives a thread a column
    /// of the state, so `dv` past 1024 would need a second dimension of work
    /// per thread; the register version gives a column to two threads, so its
    /// ceiling is half that. The checkpoint uses 128 either way.
    ///
    /// Which of the three kernels runs is [`DeltaVariant::Auto`]'s choice; see
    /// that type for what the others cost.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_delta_rule(
        &self,
        out: &mut ViewMut<'_, f32>,
        state: &mut ViewMut<'_, f32>,
        qkv: &View<'_, f32>,
        g: &View<'_, f32>,
        beta: &View<'_, f32>,
        seqs: &SeqLayout<'_>,
        heads: usize,
        key_heads: usize,
        dk: usize,
        dv: usize,
        offsets: (usize, usize, usize, usize),
    ) -> Result<()> {
        self.gdn_delta_rule_variant(
            out,
            state,
            qkv,
            g,
            beta,
            seqs,
            heads,
            key_heads,
            dk,
            dv,
            offsets,
            DeltaVariant::Auto,
        )
    }

    /// The gated delta rule, with the kernel named rather than chosen.
    ///
    /// Exists so the tests can hold the fallback to the same standard as the
    /// path a served step takes: at the checkpoint's `dk = dv = 128` every
    /// caller gets [`DeltaVariant::Reg`], and without a way to ask for
    /// [`DeltaVariant::Global`] by name the version that covers every other
    /// shape would go unexercised.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_delta_rule_variant(
        &self,
        out: &mut ViewMut<'_, f32>,
        state: &mut ViewMut<'_, f32>,
        qkv: &View<'_, f32>,
        g: &View<'_, f32>,
        beta: &View<'_, f32>,
        seqs: &SeqLayout<'_>,
        heads: usize,
        key_heads: usize,
        dk: usize,
        dv: usize,
        offsets: (usize, usize, usize, usize),
        variant: DeltaVariant,
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

        let chosen = variant.resolve(dk, dv);
        anyhow::ensure!(
            chosen != DeltaVariant::Reg || (dk == 128 && dv == 128),
            "the register-blocked delta rule is instantiated for dk = dv = 128 \
             and was asked for {dk}x{dv}; use DeltaVariant::Auto, which falls \
             back on its own"
        );
        // Shared holds q and k for the token being consumed. The register
        // version double-buffers them so it needs one barrier a token instead
        // of two; the shared version puts the whole state after them.
        let f32_size = std::mem::size_of::<f32>();
        let (name, threads, shared) = match chosen {
            // `R = 2` threads a column: 2 * dv threads, 4 * dk floats of
            // shared. Both are the kernel's, not the caller's, choice — see
            // the note above `gdn_delta_rule_reg_body`.
            DeltaVariant::Reg => ("gdn_delta_rule_reg128_f32", 2 * dv, 4 * dk * f32_size),
            DeltaVariant::Shared => (
                "gdn_delta_rule_smem_f32",
                dv.max(32),
                (2 * dk + dk * dv) * f32_size,
            ),
            _ => ("gdn_delta_rule_f32", dv.max(32), 2 * dk * f32_size),
        };
        let f = self.dev.kernels().get("tuili_gdn", gdn_src(), name)?;
        // Past 48 KiB a block the dynamic size is opt-in, and a launch that
        // asks for more without it fails with an invalid-value error rather
        // than falling back to something smaller.
        if shared > 48 * 1024 {
            tuili_gpu::set_max_dynamic_shared(&f, shared as u32).with_context(|| {
                format!(
                    "the shared-memory delta rule wants {shared} bytes a block \
                     for a {dk}x{dv} state, which this device will not give it"
                )
            })?;
        }
        let cfg = LaunchConfig {
            grid_dim: (heads as u32, seqs.n_seqs as u32, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: shared as u32,
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
        out: &mut ViewMut<'_, f32>,
        x: &View<'_, f32>,
        z: &View<'_, f32>,
        weight: &View<'_, f32>,
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
        q: &mut ViewMut<'_, f32>,
        gate: &mut ViewMut<'_, f32>,
        src: &View<'_, f32>,
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
        x: &mut ViewMut<'_, f32>,
        gate: &View<'_, f32>,
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
