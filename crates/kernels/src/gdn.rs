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
use infero_gpu::{View, ViewMut, LaunchConfig, KernelArg};

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

/// Which of kernel 2's three real variants [`Kernels::gdn_chunk_split3_delta_rule`]
/// runs -- see each kernel's own doc comment in `gdn.cu` for what each real,
/// measured fix did to the whole 3-kernel pipeline's real benchmark result.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GdnChunkStateVariant {
    /// `gdn_chunk_state_f32` -- fully synchronous, 48 blocks (`heads`).
    Plain,
    /// `gdn_chunk_state_pipelined_f32` -- `cp.async` double-buffered, same
    /// 48-block grid.
    Pipelined,
    /// `gdn_chunk_state_pipelined_split4_f32` -- pipelined, and `GDN_DV`'s
    /// 128 columns split 4 ways (192 blocks instead of 48), legal because
    /// the state recurrence is column-independent.
    PipelinedSplit4,
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
    /// `Reg`'s same register-resident state and thread layout, but a chunk of
    /// up to 64 tokens is processed with block-wide parallel matrix ops
    /// (`gdn_chunk_delta_rule_f32`) instead of one token at a time — see the
    /// comment on that kernel for the algorithm (checked against vLLM's
    /// vendored `flash-linear-attention` reference, not re-derived from the
    /// paper). Requires `dk == dv == 128`, like `Reg`. Not reachable through
    /// `Auto` yet — only by name, until it's benchmarked end to end.
    Chunk,
}

impl DeltaVariant {
    /// `Auto` resolved against the head dims; the others pass through.
    fn resolve(self, dk: usize, dv: usize) -> Self {
        match self {
            // 128 is what `gdn_delta_rule_reg128_f32` is instantiated for. A
            // second instantiation is a one-line change, but every one costs
            // NVRTC time on a cold cache for a shape no checkpoint here uses.
            // Ported to Metal too now: `simd_shuffle_xor` for the partner
            // reduction is the direct analogue of `__shfl_xor_sync`, R = 2
            // keeps the same register count viable there, and
            // `gdn_reg128_check.rs` matches the global kernel to f32 noise
            // (1e-7 to 1e-9) at every token count checked while running
            // 2.3-4.2x faster from eight tokens up -- on an M4 Max, which is
            // a different register budget than either CUDA card the note
            // above was tuned against, so the win was re-measured rather than
            // assumed to carry over.
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
    #[cfg(feature = "cuda")]
    pub fn gdn_kernel_registers(&self, name: &str) -> Result<(i32, i32, i32)> {
        let f = self.dev.kernels().get("infero_gdn", gdn_src(), name)?;
        Ok((f.num_regs()?, f.shared_size_bytes()?, f.local_size_bytes()?))
    }

    /// Blocks an SM the driver will make resident for a GatedDeltaNet kernel at
    /// a given block size and dynamic shared request.
    #[cfg(feature = "cuda")]
    pub fn gdn_occupancy_blocks(&self, name: &str, threads: u32, dynamic: usize) -> Result<u32> {
        let f = self.dev.kernels().get("infero_gdn", gdn_src(), name)?;
        if dynamic > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, dynamic as u32)?;
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

        let f = self.dev.kernels().get("infero_gdn", gdn_src(), "gdn_conv_f32")?;
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

    /// [`Self::gdn_conv`], but for the one-sequence prefill case: also splits
    /// the token dimension across blocks (`gdn_conv_chunked_f32`) instead of
    /// leaving every channel's whole token loop to a single thread. `channels
    /// / 128` blocks (80 at this checkpoint) against this GPU's 188 SMs
    /// measured 8.28% achieved occupancy for the unsplit kernel -- most SMs
    /// idle for the entire call. A chunk past the first bootstraps its
    /// window by re-reading the `k - 1` raw inputs immediately before it
    /// (already resident in `x`, no cross-block dependency), so this is
    /// splitting a per-channel loop that was never really sequential across
    /// chunks, not a new synchronization primitive.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_conv_prefill(
        &self,
        out: &mut ViewMut<'_, f32>,
        x: &View<'_, f32>,
        state: &mut ViewMut<'_, f32>,
        w: &View<'_, f32>,
        seqs: &SeqLayout<'_>,
        channels: usize,
        k: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            (2..=8).contains(&k),
            "conv kernel width {k} is outside the range the kernel's register \
             window covers (2..=8)"
        );
        anyhow::ensure!(seqs.n_seqs == 1, "gdn_conv_prefill is scoped to one sequence a call, got {}", seqs.n_seqs);
        debug_assert!(out.len() >= seqs.total_tokens * channels);
        debug_assert!(x.len() >= seqs.total_tokens * channels);
        debug_assert!(state.len() >= channels * (k - 1));
        debug_assert!(w.len() >= channels * k);

        const BLOCK: u32 = 128;
        let channel_blocks = (channels as u32).div_ceil(BLOCK);
        // Oversubscribe generously (this GPU has 188 SMs; others have fewer)
        // rather than tune to one card, and keep chunks no smaller than 32
        // tokens so the `k - 1`-tap re-read at each chunk's start stays a
        // rounding error against the chunk's own work.
        const TARGET_BLOCKS: u32 = 512;
        const MIN_CHUNK: usize = 32;
        let n_chunks = (TARGET_BLOCKS.div_ceil(channel_blocks.max(1)).max(1) as usize).min(seqs.total_tokens.max(1));
        let chunk_len = seqs.total_tokens.div_ceil(n_chunks.max(1)).max(MIN_CHUNK);
        let n_chunks = seqs.total_tokens.div_ceil(chunk_len).max(1);

        let f = self.dev.kernels().get("infero_gdn", gdn_src(), "gdn_conv_chunked_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (channel_blocks, n_chunks as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (c, kk, cl) = (channels as i32, k as i32, chunk_len as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out)
            .arg(x)
            .arg(state)
            .arg(w)
            .arg(seqs.first_token)
            .arg(seqs.n_tokens)
            .arg(&c)
            .arg(&kk)
            .arg(&cl);
        self.dev.profile().time("gdn_conv", self.dev.stream(), || {
            unsafe { b.launch(cfg) }.context("gdn_conv_chunked")?;
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
            .get("infero_gdn", gdn_src(), "gdn_gate_decay_f32")?;
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
            .get("infero_gdn", gdn_src(), "gdn_qk_l2norm_f32")?;
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
        v_tiled: bool,
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
            v_tiled,
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
        // Whether the checkpoint stores V heads tiled rather than grouped by
        // key head; see `LinearAttnConfig::v_heads_tiled`.
        v_tiled: bool,
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
        anyhow::ensure!(
            chosen != DeltaVariant::Chunk || (dk == 128 && dv == 128),
            "the chunked delta rule is instantiated for dk = dv = 128 and was \
             asked for {dk}x{dv}"
        );
        // Shared holds q and k for the token being consumed. The register
        // version double-buffers them so it needs one barrier a token instead
        // of two; the shared version puts the whole state after them.
        let f32_size = std::mem::size_of::<f32>();
        let (name, threads, shared) = match chosen {
            // `R = 2` threads a column: 2 * dv threads, `4 * dk + 32` floats
            // of shared -- the `+ 32` is bank-conflict padding (`4 * R *
            // PAD`, R = 2 and PAD = 4 in the kernel). All of this is the
            // kernel's, not the caller's, choice — see the note above
            // `gdn_delta_rule_reg_body`.
            //
            // `n_seqs == 1` (a solo prefill, the reason this variant exists
            // at all) launches exactly `heads` blocks -- 48 on this
            // checkpoint, 25.5% of a 188-SM part, independent of anything
            // per-block occupancy fixes. The `split4` kernel quarters `dv`
            // across four times as many blocks instead, at the cost of a
            // redundant per-block q/k reload (see its own comment); worth it
            // exactly when the launch would otherwise leave most SMs with
            // nothing to do, which is only true at `n_seqs == 1` -- at
            // `n_seqs > 1` the plain kernel already launches `heads *
            // n_seqs` blocks, quartering further would just multiply
            // redundant reloads for no occupancy gain the device still had
            // room to give for free.
            DeltaVariant::Reg if seqs.n_seqs == 1 => {
                ("gdn_delta_rule_reg128_split4_f32", dv / 2, (4 * dk + 32) * f32_size)
            }
            DeltaVariant::Reg => ("gdn_delta_rule_reg128_f32", 2 * dv, (4 * dk + 32) * f32_size),
            DeltaVariant::Shared => (
                "gdn_delta_rule_smem_f32",
                dv.max(32),
                (2 * dk + dk * dv) * f32_size,
            ),
            // Must match `gdn_chunk_delta_rule_f32`'s shared-memory layout in
            // `gdn.cu` exactly: `sk`+`sq`+`sv` (3 * 32 * `GDN_ROW_PAD` floats)
            // + `sgc`+`sbeta`+`sbg` (3 * 32 floats) + `sA` (32 *
            // `GDN_A_STRIDE` floats) + `sW`+`sD` (2 * 32 * `GDN_ROW_PAD`
            // floats). Everything is `float`, not `__half` -- see the
            // comment atop the kernel for why half precision measurably
            // failed the reference comparison here (the forward-substitution
            // inverse is recursive, so storage rounding compounds instead of
            // staying a fixed error) and why the chunk length is 32, not the
            // reference's 64: buying back the shared-memory room `float`
            // everywhere costs. The row strides are padded past
            // `dk`/`GDN_CHUNK` (both exact multiples of the 32-way
            // shared-memory bank count) to avoid worst-case bank conflicts
            // on every cross-row access -- see the kernel comment on
            // `GDN_ROW_PAD`/`GDN_A_STRIDE`.
            DeltaVariant::Chunk => {
                const GDN_CHUNK: usize = 32;
                let row_pad = dk + 4;
                let a_stride = GDN_CHUNK + 1;
                let kqv = 3 * GDN_CHUNK * row_pad * f32_size;
                let gc_beta_bg = 3 * GDN_CHUNK * f32_size;
                let a_mat = GDN_CHUNK * a_stride * f32_size;
                let w = GDN_CHUNK * row_pad * f32_size;
                let d = GDN_CHUNK * row_pad * f32_size;
                ("gdn_chunk_delta_rule_f32", 2 * dv, kqv + gc_beta_bg + a_mat + w + d)
            }
            _ => ("gdn_delta_rule_f32", dv.max(32), 2 * dk * f32_size),
        };
        let f = self.dev.kernels().get("infero_gdn", gdn_src(), name)?;
        // Past 48 KiB a block the dynamic size is opt-in, and a launch that
        // asks for more without it fails with an invalid-value error rather
        // than falling back to something smaller.
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared as u32).with_context(|| {
                format!(
                    "the shared-memory delta rule wants {shared} bytes a block \
                     for a {dk}x{dv} state, which this device will not give it"
                )
            })?;
        }
        let col_groups = if name == "gdn_delta_rule_reg128_split4_f32" { 4 } else { 1 };
        let cfg = LaunchConfig {
            grid_dim: (heads as u32, seqs.n_seqs as u32, col_groups),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: shared as u32,
        };
        let (h, kh) = (heads as i32, key_heads as i32);
        let (a, b_) = (dk as i32, dv as i32);
        let (st, qo, ko, vo) = (stride as i32, q_off as i32, k_off as i32, v_off as i32);
        let vt = i32::from(v_tiled);
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
            .arg(&vo)
            .arg(&vt);
        self.dev
            .profile()
            .time("gdn_delta_rule", self.dev.stream(), || {
                unsafe { bl.launch(cfg) }.context("gdn_delta_rule")?;
                Ok(())
            })?;
        Ok(())
    }

    /// A separate, isolated kernel computing the per-chunk affine-recurrence
    /// pair `(A(c), B(c))` such that `S(c+1) = A(c)@S(c) + B(c)` exactly --
    /// the enabling step for a real parallel-scan solve of kernel 2's own
    /// sequential recurrence, not the scan itself. See `gdn_chunk_ab_f32`'s
    /// own doc comment in `gdn.cu` for the full derivation, why it's
    /// verified rather than speculative, AND why this reads `w`/`u` back
    /// from an already-run [`Self::gdn_chunk_uw_only`] call instead of being
    /// fused into that kernel's own body -- a fused version produced a
    /// real, confirmed-wrong `B` that this isolated design does not
    /// reproduce. `a`/`b` must be sized `n_chunks*heads*128*128` each
    /// (`f32`); `w`/`u` must already hold kernel 1's own output for the same
    /// chunks (sized `n_chunks*heads*32*128` each).
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_chunk_ab(
        &self,
        a: &mut ViewMut<'_, f32>,
        b: &mut ViewMut<'_, f32>,
        w: &View<'_, f32>,
        u: &View<'_, f32>,
        qkv: &View<'_, f32>,
        g: &View<'_, f32>,
        seqs: &SeqLayout<'_>,
        heads: usize,
        key_heads: usize,
        dk: usize,
        dv: usize,
        offsets: (usize, usize, usize, usize),
        v_tiled: bool,
    ) -> Result<()> {
        anyhow::ensure!(seqs.n_seqs == 1, "gdn_chunk_ab: single sequence only");
        anyhow::ensure!(dk == 128 && dv == 128, "gdn_chunk_ab is instantiated for dk = dv = 128, got {dk}x{dv}");
        let (stride, _q_off, k_off, _v_off) = offsets;
        const GDN_CHUNK: usize = 32;
        const GDN_DK: usize = 128;
        const ROW_PAD: usize = GDN_DK + 4;
        let n_chunks = seqs.total_tokens.div_ceil(GDN_CHUNK).max(1);
        let f32_size = std::mem::size_of::<f32>();

        let f = self.dev.kernels().get("infero_gdn", gdn_src(), "gdn_chunk_ab_f32")?;
        let shared = (3 * GDN_CHUNK * ROW_PAD + 2 * GDN_CHUNK) * f32_size;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared as u32)?;
        }
        let cfg = LaunchConfig {
            grid_dim: (heads as u32, n_chunks as u32, 1),
            block_dim: (2 * dv as u32, 1, 1),
            shared_mem_bytes: shared as u32,
        };
        let (h, kh) = (heads as i32, key_heads as i32);
        let (dka, dva) = (dk as i32, dv as i32);
        let (st, ko) = (stride as i32, k_off as i32);
        let vt = i32::from(v_tiled);
        let mut bl = self.dev.stream().launch_builder(&f);
        bl.arg(&mut *a)
            .arg(&mut *b)
            .arg(w)
            .arg(u)
            .arg(qkv)
            .arg(g)
            .arg(seqs.first_token)
            .arg(seqs.n_tokens)
            .arg(&h)
            .arg(&kh)
            .arg(&dka)
            .arg(&dva)
            .arg(&st)
            .arg(&ko)
            .arg(&vt);
        self.dev
            .profile()
            .time("gdn_chunk_ab", self.dev.stream(), || {
                unsafe { bl.launch(cfg) }.context("gdn_chunk_ab")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Combines two affine-recurrence pairs, `(a2,b2)` applied after
    /// `(a1,b1)`, into one: `(a2@a1, a2@b1+b2)`. The one real new piece of
    /// GEMM engineering the group-scan needs -- see `gdn_ab_combine_f32`'s
    /// own doc comment in `gdn.cu` for why a naive both-operands-in-shared
    /// design doesn't fit this GPU's shared-memory ceiling, and the row-
    /// streaming technique used instead. All four matrices are `128x128`
    /// (`a1`/`a2`/`a_out`: `DK*DK`; `b1`/`b2`/`b_out`: `DK*DV`). Standalone
    /// single-block launch -- not yet wired into a real group-scan grid.
    pub fn gdn_ab_combine(
        &self,
        a_out: &mut ViewMut<'_, f32>,
        b_out: &mut ViewMut<'_, f32>,
        a1: &View<'_, f32>,
        b1: &View<'_, f32>,
        a2: &View<'_, f32>,
        b2: &View<'_, f32>,
    ) -> Result<()> {
        const GDN_DK: usize = 128;
        let f32_size = std::mem::size_of::<f32>();
        let f = self.dev.kernels().get("infero_gdn", gdn_src(), "gdn_ab_combine_f32")?;
        let shared = (2 * GDN_DK * f32_size) as u32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: shared,
        };
        let mut bl = self.dev.stream().launch_builder(&f);
        bl.arg(&mut *a_out).arg(&mut *b_out).arg(a1).arg(b1).arg(a2).arg(b2);
        self.dev
            .profile()
            .time("gdn_ab_combine", self.dev.stream(), || {
                unsafe { bl.launch(cfg) }.context("gdn_ab_combine")?;
                Ok(())
            })?;
        Ok(())
    }

    /// The group-local scan: `n_groups` blocks a head instead of 1, each
    /// walking its own `group_size`-chunk range sequentially via
    /// `gdn_ab_combine_f32`'s own row-streaming combine, reused rather than
    /// reimplemented. See `gdn_group_scan_f32`'s own doc comment in `gdn.cu`
    /// for the full design and why it reuses `prefix_a`/`prefix_b`
    /// (required output, not just scratch) as its row-streaming source
    /// instead of new register-to-shared staging.
    ///
    /// `prefix_a`/`prefix_b` (sized `n_chunks*heads*128*128` /
    /// `n_chunks*heads*128*128`) receive every chunk's own incoming prefix
    /// transform, needed by the not-yet-built final correction pass.
    /// `group_a`/`group_b` (sized `n_groups*heads*128*128`) receive each
    /// group's own total transform, needed by the not-yet-built cross-group
    /// combine. `n_groups = n_chunks.div_ceil(group_size)`.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_group_scan(
        &self,
        prefix_a: &mut ViewMut<'_, f32>,
        prefix_b: &mut ViewMut<'_, f32>,
        group_a: &mut ViewMut<'_, f32>,
        group_b: &mut ViewMut<'_, f32>,
        a_in: &View<'_, f32>,
        b_in: &View<'_, f32>,
        heads: usize,
        n_chunks: usize,
        group_size: usize,
    ) -> Result<()> {
        const GDN_DK: usize = 128;
        let f32_size = std::mem::size_of::<f32>();
        let n_groups = n_chunks.div_ceil(group_size).max(1);
        let f = self.dev.kernels().get("infero_gdn", gdn_src(), "gdn_group_scan_f32")?;
        let shared = (2 * GDN_DK * f32_size) as u32;
        let cfg = LaunchConfig {
            grid_dim: (heads as u32, n_groups as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: shared,
        };
        let (h, nc, gs) = (heads as i32, n_chunks as i32, group_size as i32);
        let mut bl = self.dev.stream().launch_builder(&f);
        bl.arg(&mut *prefix_a)
            .arg(&mut *prefix_b)
            .arg(&mut *group_a)
            .arg(&mut *group_b)
            .arg(a_in)
            .arg(b_in)
            .arg(&h)
            .arg(&nc)
            .arg(&gs);
        self.dev
            .profile()
            .time("gdn_group_scan", self.dev.stream(), || {
                unsafe { bl.launch(cfg) }.context("gdn_group_scan")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Diagnostic-only standalone wrapper for the plain, unmodified
    /// `gdn_chunk_uw_f32` (kernel 1 alone, no `A`/`B` extension) -- built to
    /// check whether a real, direct-against-a-host-reference bug found
    /// while testing `gdn_chunk_uw_ab_f32` also affects this already-
    /// shipped kernel, since no existing test checks `W`/`U` directly (only
    /// the full 3-kernel pipeline's final output, which may not surface the
    /// same issue).
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_chunk_uw_only(
        &self,
        w: &mut ViewMut<'_, f32>,
        u: &mut ViewMut<'_, f32>,
        qkv: &View<'_, f32>,
        g: &View<'_, f32>,
        beta: &View<'_, f32>,
        seqs: &SeqLayout<'_>,
        heads: usize,
        key_heads: usize,
        dk: usize,
        dv: usize,
        offsets: (usize, usize, usize, usize),
        v_tiled: bool,
    ) -> Result<()> {
        anyhow::ensure!(seqs.n_seqs == 1, "gdn_chunk_uw_only: single sequence only");
        anyhow::ensure!(dk == 128 && dv == 128, "gdn_chunk_uw_only is instantiated for dk = dv = 128, got {dk}x{dv}");
        let (stride, _q_off, k_off, v_off) = offsets;
        const GDN_CHUNK: usize = 32;
        const GDN_DK: usize = 128;
        const ROW_PAD: usize = GDN_DK + 4;
        const A_STRIDE: usize = GDN_CHUNK + 1;
        let n_chunks = seqs.total_tokens.div_ceil(GDN_CHUNK).max(1);
        let f32_size = std::mem::size_of::<f32>();

        let f = self.dev.kernels().get("infero_gdn", gdn_src(), "gdn_chunk_uw_f32")?;
        let shared = (2 * GDN_CHUNK * ROW_PAD + 3 * GDN_CHUNK + GDN_CHUNK * A_STRIDE) * f32_size;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared as u32)?;
        }
        let cfg = LaunchConfig {
            grid_dim: (heads as u32, n_chunks as u32, 1),
            block_dim: (2 * dv as u32, 1, 1),
            shared_mem_bytes: shared as u32,
        };
        let (h, kh) = (heads as i32, key_heads as i32);
        let (dka, dva) = (dk as i32, dv as i32);
        let (st, ko, vo) = (stride as i32, k_off as i32, v_off as i32);
        let vt = i32::from(v_tiled);
        let mut bl = self.dev.stream().launch_builder(&f);
        bl.arg(&mut *w)
            .arg(&mut *u)
            .arg(qkv)
            .arg(g)
            .arg(beta)
            .arg(seqs.first_token)
            .arg(seqs.n_tokens)
            .arg(&h)
            .arg(&kh)
            .arg(&dka)
            .arg(&dva)
            .arg(&st)
            .arg(&ko)
            .arg(&vo)
            .arg(&vt);
        self.dev
            .profile()
            .time("gdn_chunk_uw_only", self.dev.stream(), || {
                unsafe { bl.launch(cfg) }.context("gdn_chunk_uw_only")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Three-kernel split of the chunked delta rule -- see the doc comment
    /// on `gdn_chunk_uw_f32` in `gdn.cu` for the full architecture and why
    /// it's a real, independent re-derivation of SGLang's own real
    /// production Blackwell GDN kernel structure (which cannot be ported
    /// directly: it needs `tcgen05`, which sm_120a does not have).
    ///
    /// Single sequence only for this first pass (`seqs.n_seqs` must be 1) --
    /// proving the architecture is correct and measuring its real cost
    /// against `gdn_delta_rule_reg128_f32`, not yet matching that kernel's
    /// multi-sequence/incremental-call generality. `dk`/`dv` must be 128,
    /// same restriction `DeltaVariant::Chunk` already has.
    ///
    /// `k2` selects kernel 2's own three variants -- see each kernel's doc
    /// comment in `gdn.cu` for why kernel 2, not kernels 1 or 3, was the
    /// whole pipeline's real remaining bottleneck, and what each of these
    /// three real, measured fixes did about it.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_chunk_split3_delta_rule(
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
        v_tiled: bool,
        k2: GdnChunkStateVariant,
    ) -> Result<()> {
        anyhow::ensure!(seqs.n_seqs == 1, "gdn_chunk_split3_delta_rule: single sequence only");
        anyhow::ensure!(
            dk == 128 && dv == 128,
            "gdn_chunk_split3_delta_rule is instantiated for dk = dv = 128, got {dk}x{dv}"
        );
        let (stride, q_off, k_off, v_off) = offsets;
        const GDN_CHUNK: usize = 32;
        const GDN_DK: usize = 128;
        const ROW_PAD: usize = GDN_DK + 4;
        const A_STRIDE: usize = GDN_CHUNK + 1;
        let n_chunks = seqs.total_tokens.div_ceil(GDN_CHUNK).max(1);
        let f32_size = std::mem::size_of::<f32>();

        let stream = self.dev.stream();
        let mut w_buf = stream.alloc_zeros::<f32>(n_chunks * heads * GDN_CHUNK * GDN_DK)?;
        let mut u_buf = stream.alloc_zeros::<f32>(n_chunks * heads * GDN_CHUNK * dv)?;
        let mut delta_buf = stream.alloc_zeros::<f32>(n_chunks * heads * GDN_CHUNK * dv)?;
        let mut s_before_buf = stream.alloc_zeros::<f32>(n_chunks * heads * GDN_DK * dv)?;

        let (h, kh) = (heads as i32, key_heads as i32);
        let (a, b_) = (dk as i32, dv as i32);
        let (st, qo, ko, vo) = (stride as i32, q_off as i32, k_off as i32, v_off as i32);
        let vt = i32::from(v_tiled);

        // Kernel 1: parallel (head, chunk), no state dependency.
        {
            let f = self.dev.kernels().get("infero_gdn", gdn_src(), "gdn_chunk_uw_f32")?;
            let shared = (3 * GDN_CHUNK * ROW_PAD + 3 * GDN_CHUNK + GDN_CHUNK * A_STRIDE) * f32_size;
            if shared > 48 * 1024 {
                infero_gpu::set_max_dynamic_shared(&f, shared as u32)?;
            }
            let cfg = LaunchConfig {
                grid_dim: (heads as u32, n_chunks as u32, 1),
                block_dim: (2 * dv as u32, 1, 1),
                shared_mem_bytes: shared as u32,
            };
            let mut bl = self.dev.stream().launch_builder(&f);
            bl.arg(&mut w_buf)
                .arg(&mut u_buf)
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
                .arg(&ko)
                .arg(&vo)
                .arg(&vt);
            self.dev
                .profile()
                .time("gdn_chunk_uw", self.dev.stream(), || {
                    unsafe { bl.launch(cfg) }.context("gdn_chunk_uw")?;
                    Ok(())
                })?;
        }

        // Kernel 2: sequential over chunks. Three variants -- see
        // `GdnChunkStateVariant`'s own doc comment and each kernel's in
        // `gdn.cu`. Only `PipelinedSplit4` uses more than one block a head
        // (`heads * 4` instead of `heads`, its whole point).
        {
            let (name, col_groups, block_threads) = match k2 {
                GdnChunkStateVariant::Plain => ("gdn_chunk_state_f32", 1, 2 * dv),
                GdnChunkStateVariant::Pipelined => ("gdn_chunk_state_pipelined_f32", 1, 2 * dv),
                GdnChunkStateVariant::PipelinedSplit4 => {
                    ("gdn_chunk_state_pipelined_split4_f32", 4, dv / 2)
                }
            };
            let f = self.dev.kernels().get("infero_gdn", gdn_src(), name)?;
            let shared = if matches!(k2, GdnChunkStateVariant::Plain) {
                (2 * GDN_CHUNK * ROW_PAD + GDN_CHUNK) * f32_size
            } else {
                (4 * GDN_CHUNK * ROW_PAD + GDN_CHUNK) * f32_size
            };
            if shared > 48 * 1024 {
                infero_gpu::set_max_dynamic_shared(&f, shared as u32)?;
            }
            let cfg = LaunchConfig {
                grid_dim: (heads as u32, col_groups as u32, 1),
                block_dim: (block_threads as u32, 1, 1),
                shared_mem_bytes: shared as u32,
            };
            let nc = n_chunks as i32;
            let mut bl = self.dev.stream().launch_builder(&f);
            bl.arg(&mut delta_buf)
                .arg(&mut s_before_buf)
                .arg(&mut *state)
                .arg(&w_buf)
                .arg(&u_buf)
                .arg(qkv)
                .arg(g)
                .arg(seqs.first_token)
                .arg(seqs.n_tokens)
                .arg(&h)
                .arg(&kh)
                .arg(&a)
                .arg(&b_)
                .arg(&st)
                .arg(&ko)
                .arg(&vt)
                .arg(&nc);
            self.dev
                .profile()
                .time("gdn_chunk_state", self.dev.stream(), || {
                    unsafe { bl.launch(cfg) }.context("gdn_chunk_state")?;
                    Ok(())
                })?;
        }

        // Kernel 3: parallel (head, chunk) again, no cross-chunk dependency.
        {
            let f = self.dev.kernels().get("infero_gdn", gdn_src(), "gdn_chunk_output_f32")?;
            let shared = (2 * GDN_CHUNK * ROW_PAD + GDN_CHUNK + GDN_CHUNK * A_STRIDE) * f32_size;
            if shared > 48 * 1024 {
                infero_gpu::set_max_dynamic_shared(&f, shared as u32)?;
            }
            let cfg = LaunchConfig {
                grid_dim: (heads as u32, n_chunks as u32, 1),
                block_dim: (2 * dv as u32, 1, 1),
                shared_mem_bytes: shared as u32,
            };
            let mut bl = self.dev.stream().launch_builder(&f);
            bl.arg(&mut *out)
                .arg(&delta_buf)
                .arg(&s_before_buf)
                .arg(qkv)
                .arg(g)
                .arg(seqs.first_token)
                .arg(seqs.n_tokens)
                .arg(&h)
                .arg(&kh)
                .arg(&a)
                .arg(&b_)
                .arg(&st)
                .arg(&qo)
                .arg(&ko)
                .arg(&vt);
            self.dev
                .profile()
                .time("gdn_chunk_output", self.dev.stream(), || {
                    unsafe { bl.launch(cfg) }.context("gdn_chunk_output")?;
                    Ok(())
                })?;
        }
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
            .get("infero_gdn", gdn_src(), "gdn_gated_rmsnorm_f32")?;
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
            .get("infero_gdn", gdn_src(), "split_interleaved_f32")?;
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
            .get("infero_gdn", gdn_src(), "sigmoid_gate_f32")?;
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

    /// Two `memcpy_dtod`s as one launch — a verification pass's conv window
    /// and recurrent-state working copy, staged before it runs. See the note
    /// on `gdn_rollback_stage2_f32` for why this exists.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_rollback_stage2(
        &self,
        dst0: &mut ViewMut<'_, f32>,
        src0: &View<'_, f32>,
        dst1: &mut ViewMut<'_, f32>,
        src1: &View<'_, f32>,
    ) -> Result<()> {
        let (n0, n1) = (src0.len(), src1.len());
        debug_assert!(dst0.len() >= n0 && dst1.len() >= n1);
        let f = self
            .dev
            .kernels()
            .get("infero_gdn", gdn_src(), "gdn_rollback_stage2_f32")?;
        const BLOCK: u32 = 256;
        let total = (n0 + n1) as u32;
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(BLOCK).max(1), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (c0, c1) = (n0 as i64, n1 as i64);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(dst0).arg(src0).arg(&c0).arg(dst1).arg(src1).arg(&c1);
        self.dev
            .profile()
            .time("gdn_rollback_stage2", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("gdn_rollback_stage2")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Four `memcpy_dtod`s as one launch — a verification pass's journal tap
    /// (pre-conv, post-conv, gate, beta), recorded after it runs. See the
    /// note on `gdn_rollback_stage2_f32` for why this exists.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_rollback_record4(
        &self,
        dst0: &mut ViewMut<'_, f32>,
        src0: &View<'_, f32>,
        dst1: &mut ViewMut<'_, f32>,
        src1: &View<'_, f32>,
        dst2: &mut ViewMut<'_, f32>,
        src2: &View<'_, f32>,
        dst3: &mut ViewMut<'_, f32>,
        src3: &View<'_, f32>,
    ) -> Result<()> {
        let (n0, n1, n2, n3) = (src0.len(), src1.len(), src2.len(), src3.len());
        debug_assert!(
            dst0.len() >= n0 && dst1.len() >= n1 && dst2.len() >= n2 && dst3.len() >= n3
        );
        let f = self
            .dev
            .kernels()
            .get("infero_gdn", gdn_src(), "gdn_rollback_record4_f32")?;
        const BLOCK: u32 = 256;
        let total = (n0 + n1 + n2 + n3) as u32;
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(BLOCK).max(1), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (c0, c1, c2, c3) = (n0 as i64, n1 as i64, n2 as i64, n3 as i64);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(dst0)
            .arg(src0)
            .arg(&c0)
            .arg(dst1)
            .arg(src1)
            .arg(&c1)
            .arg(dst2)
            .arg(src2)
            .arg(&c2)
            .arg(dst3)
            .arg(src3)
            .arg(&c3);
        self.dev
            .profile()
            .time("gdn_rollback_record4", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("gdn_rollback_record4")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Sequential reference for [`Self::gdn_pp_pipelined_probe`] -- see
    /// `gdn_pp_sequential_ref`'s doc comment in `cu/gdn.cu`.
    pub fn gdn_pp_sequential_ref(&self, out: &mut ViewMut<'_, f32>) -> Result<()> {
        let f = self.dev.kernels().get("infero_gdn", gdn_src(), "gdn_pp_sequential_ref")?;
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out);
        unsafe { b.launch(cfg) }.context("gdn_pp_sequential_ref")?;
        Ok(())
    }

    /// Isolated toy probe: does pipelining GatedDeltaNet's own state-advance
    /// and output-compute stages across two physical warps (state races
    /// ahead uninterrupted, output trails one timestep behind) actually
    /// overlap on real hardware? See `gdn_pp_pipelined_probe`'s doc comment
    /// in `cu/gdn.cu` for the traced dependency-graph argument this tests.
    pub fn gdn_pp_pipelined_probe(&self, out: &mut ViewMut<'_, f32>) -> Result<()> {
        let f = self.dev.kernels().get("infero_gdn", gdn_src(), "gdn_pp_pipelined_probe")?;
        let shared = 2 * 32 * 64 * 4;
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (64, 1, 1), shared_mem_bytes: shared as u32 };
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out);
        unsafe { b.launch(cfg) }.context("gdn_pp_pipelined_probe")?;
        Ok(())
    }

    /// Same as [`Self::gdn_pp_pipelined_probe`], batching `GDN_PP_BATCH`
    /// (4) timesteps per handoff round instead of 1 -- see
    /// `gdn_pp_pipelined_batched_probe`'s doc comment in `cu/gdn.cu`.
    pub fn gdn_pp_pipelined_batched_probe(&self, out: &mut ViewMut<'_, f32>) -> Result<()> {
        let f = self.dev.kernels().get("infero_gdn", gdn_src(), "gdn_pp_pipelined_batched_probe")?;
        let shared = 2 * 32 * 4 * 64 * 4;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared as u32)?;
        }
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (64, 1, 1), shared_mem_bytes: shared as u32 };
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out);
        unsafe { b.launch(cfg) }.context("gdn_pp_pipelined_batched_probe")?;
        Ok(())
    }
}
