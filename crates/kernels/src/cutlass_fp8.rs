//! Rust binding for the AOT-compiled CUTLASS SM120 blockwise-scaled
//! FP8->bf16 GEMM (`cutlass/fp8_bw_gemm.cu`). Only built behind the
//! `cutlass` feature -- see that file's header for the memory layout it
//! needs, which differs from every other FP8 kernel in this crate on both
//! scale grids, and from [`crate::WeightType::F8E4M3`]'s quant byte layout
//! too (that one's [`crate::fp8::ROW_GROUP`]-interleaved for
//! `mma_e4m3_block`'s access pattern; CUTLASS wants plain `[n,k]` row-major).

use anyhow::{Context, Result};
use cudarc::driver::{DevicePtr, DevicePtrMut};
use infero_gpu::{Buf, KernelArg, LaunchConfig, View, ViewMut};

use crate::attn_backend;
use crate::fp8::{FP8_BLOCK, ROW_GROUP};
use crate::{Kernels, fp8_src};

/// [`Kernels::mma_e4m3_cutlass_sfa_f32out`]'s crossover, in tokens, between
/// the small-M tile (`fp8_bw_gemm.cu`'s `small_m` namespace, `<64,128,128>`)
/// and the plain, prefill-tuned tile (`<128,128,128>`). 64 because that is
/// the small tile's own M, past which it needs a second M-tile iteration the
/// wide tile would not -- confirmed still the right crossover for `small_m`
/// itself by the same real sweep that measured [`SWAP_AB_MAX_TOKENS`]
/// below (`examples/swap_ab_vs_small_m_bench.rs`, real qwen38-27b-fp8 FFN
/// gate-projection shape, K=5120 N=17408, `bw`'s SM120 hardware): `small_m`
/// still clearly beats the plain tile at 48 and 64 tokens (0.027ms vs
/// 0.043ms, ~1.6x), so this threshold itself did not move.
const SMALL_M_MAX_TOKENS: usize = 64;

/// Crossover, in tokens, between the operand-swapped small-M tile
/// (`small_m_swap`, `<128,32,128>`) and the plain (non-swapped) small-M tile
/// (`small_m`, `<64,128,128>`) -- both cover `n_tokens <= SMALL_M_MAX_TOKENS`,
/// this decides which of the two. Real, measured (not vLLM's own `M <= 64`
/// threshold, which this hardware/kernel does NOT share -- see below),
/// via `examples/swap_ab_vs_small_m_bench.rs` on `bw`'s SM120 hardware at
/// the real qwen38-27b-fp8 FFN gate-projection shape (K=5120, N=17408):
///
/// ```text
/// tokens   small_m(ms)   swap(ms)   swap/small_m
///      1        0.0536     0.0349          0.650
///      8        0.0513     0.0308          0.600
///     16        0.0451     0.0267          0.591   <- real batch=16 decode shape
///     24        0.0431     0.0247          0.573
///     32        0.0268     0.0226          0.844
///     48        0.0268     0.0423          1.580   <- swap now LOSES
///     64        0.0272     0.0415          1.522
/// ```
///
/// vLLM's own real dispatch (`sm120_blockwise_fp8_config_swapab`,
/// `cutlass_gemm_blockwise_sm120_fp8_dispatch`) swaps for the whole
/// `M <= 64` range -- the measurement above shows infero's own kernel does
/// NOT share that crossover: swap wins clearly through 32 tokens and loses
/// badly at 48 and past it. 32 is the highest measured point still a real
/// win (1.19x), not an extrapolation past what was actually timed.
const SWAP_AB_MAX_TOKENS: usize = 32;

/// A [`crate::WeightType::F8E4M3`] matrix's precomputed CUTLASS-side state:
/// the scale grid transposed from `[n/128,k/128]` to `[k/128,n/128]`, and --
/// only when the matrix's own storage is still [`crate::fp8::ROW_GROUP`]-
/// interleaved (`already_plain: false` at [`Kernels::prepare_cutlass_weight`])
/// -- a one-time un-repacked copy of the quants too. Build once at
/// weight-load time and hold onto it; both passes are `O(n*k)`, the same
/// order as the GEMM itself they feed, so redoing them on every forward call
/// (which an earlier version of this path did) ate most of the win.
///
/// `already_plain: true` (the `cutlass` feature's unified-format path, see
/// [`Kernels::mmv_f8_plain`]'s doc comment) means the matrix's *own* device
/// buffer already is what CUTLASS wants, so `quants` stays `None` and
/// [`Kernels::mma_e4m3_cutlass`] reads straight from it -- no second copy of
/// the weights at all, not even a budgeted one.
pub struct CutlassWeight {
    quants: Option<Buf<u8>>,
    scale_t: Buf<f32>,
    k: usize,
    n: usize,
}

/// When `already_plain` is `false`, a [`CutlassWeight`] also caches a
/// *second* copy of the matrix's quant bytes next to the
/// [`crate::fp8::ROW_GROUP`]-interleaved one `mma_e4m3_block` still needs for
/// decode-batch traffic -- caching one for every FP8 matrix a model has is
/// not a rounding error, it is close to doubling that model's FP8 weight
/// footprint, and finding that out from a `CUDA_ERROR_OUT_OF_MEMORY` against
/// a nearly-full card is how this constant came to exist. Unset (or `0`),
/// no matrix gets that second copy cached and every call falls back to
/// `mma_e4m3_block` -- opt in explicitly with
/// `INFERO_CUTLASS_WEIGHT_BUDGET_MIB` once the caller's VRAM budget for it
/// is known, not by default. Irrelevant (and not consulted) when
/// `already_plain` is `true`: there is nothing to budget.
fn cutlass_weight_budget_bytes() -> usize {
    static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("INFERO_CUTLASS_WEIGHT_BUDGET_MIB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|mib| mib << 20)
            .unwrap_or(0)
    })
}

static CUTLASS_WEIGHT_USED_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// A grow-only scratch buffer, reused across calls instead of
/// `cudaMalloc`/`cudaFree`-ing fresh each time. `mma_e4m3_cutlass` used to
/// allocate its workspace/padding/output buffers fresh on every call --
/// harmless at the token counts the kernel-level benchmark used, but real
/// churn at the 34560-call-a-prefill rate a chunked forward pass produces
/// (one call a chunk a FFN matrix a layer). One `Scratch` per distinct
/// buffer role (workspace, padded activations, bf16 output); a single
/// `Mutex` each because this crate's forward pass issues these calls
/// sequentially on one stream, never concurrently -- see
/// [`crate::fp8::pad_rows`]'s neighbor `prepare_cutlass_weight`'s own
/// single-stream assumption for the same reasoning.
struct Scratch {
    buf: std::sync::Mutex<Option<Buf<u8>>>,
}
impl Scratch {
    const fn new() -> Self {
        Self { buf: std::sync::Mutex::new(None) }
    }
    /// Runs `f` with a `ViewMut<u8>` over at least `bytes` of scratch,
    /// zeroed only the first time (or after growing) -- callers that need
    /// zeroed memory every call (like a padded activation buffer whose tail
    /// rows must read as zero) must zero their own slice, not rely on this.
    fn with<T>(
        &self,
        stream: &std::sync::Arc<infero_gpu::Stream>,
        bytes: usize,
        f: impl FnOnce(&mut ViewMut<'_, u8>) -> Result<T>,
    ) -> Result<T> {
        let mut guard = self.buf.lock().unwrap();
        if guard.as_ref().is_none_or(|b| b.len() < bytes) {
            *guard = Some(stream.alloc_zeros::<u8>(bytes.max(1))?);
        }
        let buf = guard.as_mut().unwrap();
        f(&mut buf.slice_mut(0..bytes.max(1)))
    }
}
static CUTLASS_WORKSPACE: Scratch = Scratch::new();
static CUTLASS_A_PAD: Scratch = Scratch::new();
static CUTLASS_D_PAD: Scratch = Scratch::new();

impl Kernels {
    /// Prepares a [`crate::WeightType::F8E4M3`] matrix's [`CutlassWeight`].
    /// Call once per weight matrix, not per forward pass.
    ///
    /// `already_plain` must be `true` iff `w` is *not*
    /// [`crate::fp8::ROW_GROUP`]-interleaved (i.e. the caller loaded it with
    /// [`crate::fp8::pad_rows`], not [`crate::fp8::repack_rows`]) -- getting
    /// this wrong silently corrupts every matmul through the returned
    /// weight, not just this method.
    ///
    /// Refuses (an `Err`, meant as "stay on `mma_e4m3_block`" for a
    /// non-`already_plain` caller, not a hard failure) once
    /// [`cutlass_weight_budget_bytes`]'s VRAM budget for the un-repacked
    /// quants copy is spent -- see its doc comment. Never refuses when
    /// `already_plain` is `true`: there's no second quants copy to budget,
    /// only the small transposed scale grid.
    pub fn prepare_cutlass_weight(
        &self,
        w: &View<'_, u8>,
        k: usize,
        n: usize,
        already_plain: bool,
    ) -> Result<CutlassWeight> {
        anyhow::ensure!(
            k.is_multiple_of(FP8_BLOCK) && n.is_multiple_of(FP8_BLOCK),
            "the CUTLASS GEMM's tile is {FP8_BLOCK}; got k={k} n={n}"
        );
        let stream = self.dev.stream();
        let groups = k / FP8_BLOCK;
        let n_blocks = n / FP8_BLOCK;
        let n_padded_rows = n.next_multiple_of(ROW_GROUP);
        let scale_byte_offset = (n_padded_rows * k) as i32;

        let quants = if already_plain {
            None
        } else {
            let bytes = n * k;
            let budget = cutlass_weight_budget_bytes();
            let used = CUTLASS_WEIGHT_USED_BYTES.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
            if used + bytes > budget {
                CUTLASS_WEIGHT_USED_BYTES.fetch_sub(bytes, std::sync::atomic::Ordering::Relaxed);
                anyhow::bail!(
                    "CUTLASS weight cache budget ({} MiB) spent; set INFERO_CUTLASS_WEIGHT_BUDGET_MIB \
                     higher to cache more matrices (each on top of that matrix's existing VRAM copy) \
                     -- or load weights with the unified plain layout to avoid the second copy entirely",
                    budget >> 20
                );
            }
            let mut q = stream.alloc_zeros::<u8>(n * k)?;
            let f = self.dev.kernels().get("infero_fp8", fp8_src(), "unrepack_rows_e4m3")?;
            let total = (n_padded_rows.div_ceil(ROW_GROUP) * (k / 4) * ROW_GROUP) as u32;
            const BLOCK: u32 = 256;
            let cfg = LaunchConfig {
                grid_dim: (total.div_ceil(BLOCK), 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let (ki, ni) = (k as i32, n as i32);
            let mut b = stream.launch_builder(&f);
            b.arg(&mut q).arg(w).arg(&ki).arg(&ni);
            unsafe { b.launch(cfg) }.context("unrepack_rows_e4m3")?;
            Some(q)
        };

        let mut scale_t = stream.alloc_zeros::<f32>(groups * n_blocks)?;
        {
            let f = self.dev.kernels().get("infero_fp8", fp8_src(), "transpose_scale_b_f32")?;
            const BLOCK: u32 = 128;
            let cfg = LaunchConfig {
                grid_dim: ((groups as u32).div_ceil(BLOCK), n_blocks as u32, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let (nb, kb) = (n_blocks as i32, groups as i32);
            let mut b = stream.launch_builder(&f);
            b.arg(&mut scale_t).arg(w).arg(&scale_byte_offset).arg(&nb).arg(&kb);
            unsafe { b.launch(cfg) }.context("transpose_scale_b_f32")?;
        }

        self.dev.stream().synchronize().context("preparing a CUTLASS weight")?;
        Ok(CutlassWeight { quants, scale_t, k, n })
    }
}

mod ffi {
    use std::ffi::c_void;
    unsafe extern "C" {
        pub fn infero_cutlass_fp8_bw_gemm_workspace(m: i32, n: i32, k: i32) -> usize;
        #[allow(clippy::too_many_arguments)]
        pub fn infero_cutlass_fp8_bw_gemm(
            a: *const c_void,
            b: *const c_void,
            sfa: *const f32,
            sfb: *const f32,
            d: *mut c_void,
            workspace: *mut c_void,
            m: i32,
            n: i32,
            k: i32,
            stream: cudarc::driver::sys::CUstream,
        ) -> i32;
        pub fn infero_cutlass_fp8_bw_gemm_f32out_workspace(m: i32, n: i32, k: i32) -> usize;
        #[allow(clippy::too_many_arguments)]
        pub fn infero_cutlass_fp8_bw_gemm_f32out(
            a: *const c_void,
            b: *const c_void,
            sfa: *const f32,
            sfb: *const f32,
            d: *mut f32,
            workspace: *mut c_void,
            m: i32,
            n: i32,
            k: i32,
            accum: i32,
            stream: cudarc::driver::sys::CUstream,
        ) -> i32;

        // SM120 only, same wide `<128,128,128>` tile as the plain entry point
        // above but with CUTLASS's `StreamKScheduler` in place of the default
        // persistent scheduler -- see `fp8_bw_gemm.cu`'s `stream_k` namespace
        // comment for why this exists and why the "no sm120 stream-K
        // scheduler exists" prior verdict was wrong.
        pub fn infero_cutlass_fp8_bw_gemm_f32out_stream_k_workspace(m: i32, n: i32, k: i32) -> usize;
        #[allow(clippy::too_many_arguments)]
        pub fn infero_cutlass_fp8_bw_gemm_f32out_stream_k(
            a: *const c_void,
            b: *const c_void,
            sfa: *const f32,
            sfb: *const f32,
            d: *mut f32,
            workspace: *mut c_void,
            m: i32,
            n: i32,
            k: i32,
            accum: i32,
            stream: cudarc::driver::sys::CUstream,
        ) -> i32;

        // Hopper/Blackwell-datacenter kernel bodies (real, distinctly-typed
        // `GemmKernel`s, not the SM120 kernel above recompiled — see
        // `fp8_bw_gemm.cu`'s own `sm90`/`sm100` namespace comment). Same
        // signature shape as the SM120 entry points above; which one gets
        // called is decided in Rust by `Kernels::caps`, never at the C level.
        pub fn infero_cutlass_fp8_bw_gemm_f32out_workspace_sm90(m: i32, n: i32, k: i32) -> usize;
        #[allow(clippy::too_many_arguments)]
        pub fn infero_cutlass_fp8_bw_gemm_f32out_sm90(
            a: *const c_void,
            b: *const c_void,
            sfa: *const f32,
            sfb: *const f32,
            d: *mut f32,
            workspace: *mut c_void,
            m: i32,
            n: i32,
            k: i32,
            accum: i32,
            stream: cudarc::driver::sys::CUstream,
        ) -> i32;
        pub fn infero_cutlass_fp8_bw_gemm_f32out_workspace_sm100(m: i32, n: i32, k: i32) -> usize;
        #[allow(clippy::too_many_arguments)]
        pub fn infero_cutlass_fp8_bw_gemm_f32out_sm100(
            a: *const c_void,
            b: *const c_void,
            sfa: *const f32,
            sfb: *const f32,
            d: *mut f32,
            workspace: *mut c_void,
            m: i32,
            n: i32,
            k: i32,
            accum: i32,
            stream: cudarc::driver::sys::CUstream,
        ) -> i32;

        // SM120 only, small-M tile (`<64,128,128>` against the plain entry
        // point's `<128,128,128>`) -- see `fp8_bw_gemm.cu`'s `small_m`
        // namespace comment for why this exists (decode-shaped `n_tokens`
        // wastes most of a 128-wide M tile) and why it is SM120-only for now
        // (no SM90/SM100 hardware to verify a second small-M body against).
        pub fn infero_cutlass_fp8_bw_gemm_f32out_small_m_workspace(m: i32, n: i32, k: i32) -> usize;
        #[allow(clippy::too_many_arguments)]
        pub fn infero_cutlass_fp8_bw_gemm_f32out_small_m(
            a: *const c_void,
            b: *const c_void,
            sfa: *const f32,
            sfb: *const f32,
            d: *mut f32,
            workspace: *mut c_void,
            m: i32,
            n: i32,
            k: i32,
            accum: i32,
            stream: cudarc::driver::sys::CUstream,
        ) -> i32;

        // Same operand-swapped small-M tile as `_small_m_swap` above, with
        // `StreamKScheduler` in place of the default scheduler -- see
        // `fp8_bw_gemm.cu`'s `small_m_swap_stream_k` namespace comment.
        pub fn infero_cutlass_fp8_bw_gemm_f32out_small_m_swap_stream_k_workspace(m: i32, n: i32, k: i32) -> usize;
        #[allow(clippy::too_many_arguments)]
        pub fn infero_cutlass_fp8_bw_gemm_f32out_small_m_swap_stream_k(
            a: *const c_void,
            b: *const c_void,
            sfa: *const f32,
            sfb: *const f32,
            d: *mut f32,
            workspace: *mut c_void,
            m: i32,
            n: i32,
            k: i32,
            accum: i32,
            stream: cudarc::driver::sys::CUstream,
        ) -> i32;

        // SM120 only, operand-swapped small-M tile (`<128,32,128>`, weight
        // and activation swapped into the mainloop's A/B slots) -- see
        // `fp8_bw_gemm.cu`'s `small_m_swap` namespace comment for why this
        // exists (verified against vLLM's own real dispatch source, not
        // guessed) and why it is SM120-only for the same reason `small_m` is.
        pub fn infero_cutlass_fp8_bw_gemm_f32out_small_m_swap_workspace(m: i32, n: i32, k: i32) -> usize;
        #[allow(clippy::too_many_arguments)]
        pub fn infero_cutlass_fp8_bw_gemm_f32out_small_m_swap(
            a: *const c_void,
            b: *const c_void,
            sfa: *const f32,
            sfb: *const f32,
            d: *mut f32,
            workspace: *mut c_void,
            m: i32,
            n: i32,
            k: i32,
            accum: i32,
            stream: cudarc::driver::sys::CUstream,
        ) -> i32;
    }
}

/// Which real, distinctly-compiled CUTLASS FP8 GEMM kernel body this GPU's
/// compute capability maps to — `None` means no working kernel exists for
/// this hardware at all (yet), and the caller must fall back to
/// `mma_e4m3_block` rather than attempt a mismatched entry point.
///
/// Only SM120 (this crate's original target) has ever been executed on real
/// hardware. SM90/SM100 are real, separately-typed CUTLASS kernel bodies
/// (see `fp8_bw_gemm.cu`'s `sm90`/`sm100` namespaces, added in `62edc2f`) —
/// compile-verified and linked, but their *correctness on real hardware* is
/// unverified, since no SM90/SM100 GPU exists on any box this project has
/// access to. Routing to them is a real, tested decision (see this file's
/// `#[cfg(test)] mod gemm_arch_tests`); the kernel body's own numerical
/// correctness once it actually runs is a separate, still-open claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GemmArchTier {
    Sm90,
    Sm100,
    Sm120,
}

fn gemm_arch_tier(caps: attn_backend::HardwareCaps) -> Option<GemmArchTier> {
    match caps.arch {
        90 => Some(GemmArchTier::Sm90),
        100 => Some(GemmArchTier::Sm100),
        120 => Some(GemmArchTier::Sm120),
        _ => None,
    }
}

#[cfg(test)]
mod gemm_arch_tests {
    use super::*;

    #[test]
    fn routes_each_known_arch_to_its_own_kernel_body() {
        assert_eq!(
            gemm_arch_tier(attn_backend::HardwareCaps { arch: 90, sm_count: 1 }),
            Some(GemmArchTier::Sm90)
        );
        assert_eq!(
            gemm_arch_tier(attn_backend::HardwareCaps { arch: 100, sm_count: 1 }),
            Some(GemmArchTier::Sm100)
        );
        assert_eq!(
            gemm_arch_tier(attn_backend::HardwareCaps { arch: 120, sm_count: 1 }),
            Some(GemmArchTier::Sm120)
        );
    }

    #[test]
    fn refuses_hardware_with_no_working_kernel_body() {
        // Ampere: no FP8 tensor-core hardware at all. Ada (89): CUTLASS has
        // no blockwise-scaling config for it in this project's vendored
        // checkout yet (see `62edc2f`'s own commit message). Both must come
        // back `None`, not silently route to a mismatched entry point.
        assert_eq!(gemm_arch_tier(attn_backend::HardwareCaps { arch: 80, sm_count: 1 }), None);
        assert_eq!(gemm_arch_tier(attn_backend::HardwareCaps { arch: 89, sm_count: 1 }), None);
        assert_eq!(gemm_arch_tier(attn_backend::HardwareCaps { arch: 121, sm_count: 1 }), None);
    }
}

impl Kernels {
    /// [`Kernels::mma_e4m3_block`], routed through the AOT CUTLASS GEMM
    /// instead of the hand-written tensor-core kernel -- ~10x its measured
    /// TFLOPS on the shapes this model uses (see the project memory this
    /// came out of). `cw` must already be prepared (build once at load time
    /// with [`Kernels::prepare_cutlass_weight`]); what remains per call is
    /// transposing the *activation* scale, `O(n_tokens*k/128)` against an
    /// `O(n*k*n_tokens)` GEMM. No `n_tokens` padding: CUTLASS accepts any
    /// `M` here, not just multiples of 128.
    ///
    /// `w` is the matrix's own device buffer, same one every other FP8
    /// kernel here reads -- used directly as the quants operand when `cw`
    /// was built `already_plain` (nothing to duplicate), ignored in favor of
    /// `cw`'s own un-repacked copy otherwise.
    ///
    /// Same contract as `mma_e4m3_block`: same `xq`/`xs` layouts in,
    /// `Ok(false)` if the shape is not one this path handles (caller falls
    /// back to `mma_e4m3_block`), never a wrong answer.
    #[allow(clippy::too_many_arguments)]
    pub fn mma_e4m3_cutlass(
        &self,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        cw: &CutlassWeight,
        xq: &View<'_, u8>,
        xs: &View<'_, f32>,
        k: usize,
        n: usize,
        n_tokens: usize,
        accum: bool,
    ) -> Result<bool> {
        anyhow::ensure!(
            cw.k == k && cw.n == n,
            "CutlassWeight is [{}, {}], called with k={k} n={n}",
            cw.n,
            cw.k
        );
        if !k.is_multiple_of(FP8_BLOCK) || !n.is_multiple_of(FP8_BLOCK) || n_tokens == 0 {
            return Ok(false);
        }
        let stream = self.dev.stream();
        let groups = k / FP8_BLOCK;
        // No padding: CUTLASS's `can_implement`/correctness hold for any M,
        // not just multiples of 128 (verified across n_tokens 1..129, see
        // [[project-infero-perf-gap]]).
        let m_pad = n_tokens;

        // Transpose + pad the activation scale: [n_tokens,groups] -> [groups,m_pad].
        // `mma_e4m3_cutlass_sfa` skips this for callers (the unified-layout
        // path) whose quantizer wrote the transposed layout directly —
        // see `Kernels::quantize_act_e4m3_cutlass`.
        let mut sfa_t = stream.alloc_zeros::<f32>(groups * m_pad)?;
        {
            let f = self
                .dev
                .kernels()
                .get("infero_fp8", fp8_src(), "transpose_pad_scale_a_f32")?;
            const BLOCK: u32 = 128;
            let cfg = LaunchConfig {
                grid_dim: ((m_pad as u32).div_ceil(BLOCK), groups as u32, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let (nt, gr, mp) = (n_tokens as i32, groups as i32, m_pad as i32);
            let mut b = stream.launch_builder(&f);
            b.arg(&mut sfa_t).arg(xs).arg(&nt).arg(&gr).arg(&mp);
            self.dev
                .profile()
                .time("cutlass_transpose_sfa", stream, || {
                    unsafe { b.launch(cfg) }.context("transpose_pad_scale_a_f32")?;
                    Ok(())
                })?;
        }
        self.mma_e4m3_cutlass_sfa(out, w, cw, xq, &sfa_t.as_view(), k, n, n_tokens, m_pad, accum)
    }

    /// Same as [`Self::mma_e4m3_cutlass`], but `sfa_t` is already in the
    /// transposed `[groups, m_pad]` layout — the caller's own quantizer wrote
    /// it directly (see [`Kernels::quantize_act_e4m3_cutlass`]), so there is
    /// no separate `[n_tokens, groups]` scale to transpose here. `m_pad` is
    /// whatever width the caller actually built `sfa_t` at — pass `n_tokens`
    /// itself for no padding at all; CUTLASS's `can_implement`/correctness
    /// were verified to accept any `M`, not just multiples of 128 (see
    /// [[project-infero-perf-gap]]'s CUTLASS-M-alignment entry), so callers
    /// no longer need to round up.
    #[allow(clippy::too_many_arguments)]
    pub fn mma_e4m3_cutlass_sfa(
        &self,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        cw: &CutlassWeight,
        xq: &View<'_, u8>,
        sfa_t: &View<'_, f32>,
        k: usize,
        n: usize,
        n_tokens: usize,
        m_pad: usize,
        accum: bool,
    ) -> Result<bool> {
        anyhow::ensure!(
            cw.k == k && cw.n == n,
            "CutlassWeight is [{}, {}], called with k={k} n={n}",
            cw.n,
            cw.k
        );
        if !k.is_multiple_of(FP8_BLOCK) || !n.is_multiple_of(FP8_BLOCK) || n_tokens == 0 {
            return Ok(false);
        }
        anyhow::ensure!(m_pad >= n_tokens, "m_pad {m_pad} is narrower than n_tokens {n_tokens}");
        let stream = self.dev.stream();
        debug_assert!(sfa_t.len() >= (k / FP8_BLOCK) * m_pad);

        // `m_pad == n_tokens` (the common case now that CUTLASS is known to
        // accept any M) needs no activation padding at all -- `xq` goes
        // straight in as `a`, skipping `CUTLASS_A_PAD`'s memset+memcpy
        // entirely. A caller that still wants real padding (`m_pad >
        // n_tokens`) gets the old copy-into-scratch path.
        let d_pad_bytes = m_pad * n * 2; // bf16, held as raw bits
        let ws_bytes = unsafe { ffi::infero_cutlass_fp8_bw_gemm_workspace(m_pad as i32, n as i32, k as i32) };

        let run_gemm = |a_ptr, stream: &_| -> Result<()> {
            CUTLASS_D_PAD.with(stream, d_pad_bytes, |d_pad_view| {
                CUTLASS_WORKSPACE.with(stream, ws_bytes.max(1), |ws_view| {
                    let (b_ptr, _rb) = match &cw.quants {
                        Some(q) => q.device_ptr(stream),
                        None => w.device_ptr(stream),
                    };
                    let (sfa_ptr, _rsfa) = sfa_t.device_ptr(stream);
                    let (sfb_ptr, _rsfb) = cw.scale_t.device_ptr(stream);
                    let (d_ptr, _rd) = d_pad_view.device_ptr_mut(stream);
                    let (ws_ptr, _rws) = ws_view.device_ptr_mut(stream);
                    let status = self
                        .dev
                        .profile()
                        .time("cutlass_fp8_gemm", stream, || {
                            let st = unsafe {
                                ffi::infero_cutlass_fp8_bw_gemm(
                                    a_ptr as *const std::ffi::c_void,
                                    b_ptr as *const std::ffi::c_void,
                                    sfa_ptr as *const f32,
                                    sfb_ptr as *const f32,
                                    d_ptr as *mut std::ffi::c_void,
                                    ws_ptr as *mut std::ffi::c_void,
                                    m_pad as i32,
                                    n as i32,
                                    k as i32,
                                    stream.cu_stream(),
                                )
                            };
                            Ok(st)
                        })?;
                    // These `SyncOnDrop` guards borrow their buffers
                    // mutably; drop them now that the launch is submitted,
                    // or the `as_view()` read-back below can't borrow
                    // `d_pad_view` again.
                    drop((_rb, _rsfa, _rsfb, _rd, _rws));
                    anyhow::ensure!(status == 0, "CUTLASS GEMM returned status {status}");

                    // Upconvert bf16 -> f32 into `out`, discarding any padded rows.
                    let f = self
                        .dev
                        .kernels()
                        .get("infero_fp8", fp8_src(), "bf16_store_or_accum_f32")?;
                    const BLOCK: u32 = 128;
                    let cfg = LaunchConfig {
                        grid_dim: ((n as u32).div_ceil(BLOCK), n_tokens as u32, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let (nt, ni, mp, acc) = (n_tokens as i32, n as i32, m_pad as i32, i32::from(accum));
                    // `d_pad_view` is `CUTLASS_D_PAD`'s raw byte scratch;
                    // the GEMM wrote `m_pad*n` bf16 values (raw bits) into
                    // it, so reinterpret rather than copy to read them back.
                    let d_pad_u16 = unsafe { d_pad_view.as_view().transmute::<u16>(m_pad * n) }
                        .context("CUTLASS output scratch too small to reinterpret as bf16")?;
                    let mut b = stream.launch_builder(&f);
                    b.arg(out).arg(&d_pad_u16).arg(&nt).arg(&ni).arg(&mp).arg(&acc);
                    self.dev
                        .profile()
                        .time("cutlass_bf16_store", stream, || {
                            unsafe { b.launch(cfg) }.context("bf16_store_or_accum_f32")?;
                            Ok(())
                        })
                })
            })
        };

        if m_pad == n_tokens {
            let (a_ptr, _ra) = xq.device_ptr(stream);
            let result = run_gemm(a_ptr, stream);
            drop(_ra);
            result?;
        } else {
            let a_pad_bytes = m_pad * k;
            CUTLASS_A_PAD.with(stream, a_pad_bytes, |a_pad_bytes_view| {
                stream
                    .memset_zeros(&mut a_pad_bytes_view.slice_mut(n_tokens * k..m_pad * k))
                    .context("clearing the CUTLASS activation pad tail")?;
                stream
                    .memcpy_dtod(&xq.slice(0..n_tokens * k), &mut a_pad_bytes_view.slice_mut(0..n_tokens * k))
                    .context("padding activations for the CUTLASS GEMM")?;
                let (a_ptr, _ra) = a_pad_bytes_view.device_ptr(stream);
                let result = run_gemm(a_ptr, stream);
                drop(_ra);
                result
            })?;
        }

        Ok(true)
    }

    /// [`Self::mma_e4m3_cutlass_sfa`], but CUTLASS's own epilogue writes `out`
    /// (f32) directly -- no bf16 scratch, no separate
    /// `bf16_store_or_accum_f32` kernel afterward. Only for `n_tokens` (no
    /// padding); a caller that still needs `m_pad > n_tokens` should use
    /// [`Self::mma_e4m3_cutlass_sfa`] instead. `accum` maps straight to
    /// CUTLASS's own `beta` (1.0 to add into `out`'s existing contents, 0.0
    /// to overwrite), the same semantics `bf16_store_or_accum_f32`'s own
    /// `accum` flag had.
    #[allow(clippy::too_many_arguments)]
    pub fn mma_e4m3_cutlass_sfa_f32out(
        &self,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        cw: &CutlassWeight,
        xq: &View<'_, u8>,
        sfa_t: &View<'_, f32>,
        k: usize,
        n: usize,
        n_tokens: usize,
        accum: bool,
    ) -> Result<bool> {
        anyhow::ensure!(
            cw.k == k && cw.n == n,
            "CutlassWeight is [{}, {}], called with k={k} n={n}",
            cw.n,
            cw.k
        );
        if !k.is_multiple_of(FP8_BLOCK) || !n.is_multiple_of(FP8_BLOCK) || n_tokens == 0 {
            return Ok(false);
        }
        // No working CUTLASS kernel body exists for this hardware at all
        // (e.g. Ampere has no FP8 tensor cores; Ada has no blockwise-scaling
        // config in this project's vendored CUTLASS checkout yet) -- same
        // "caller falls back to `mma_e4m3_block`" contract as the shape
        // checks just above, not a crash or a mismatched-entry-point launch.
        let Some(tier) = gemm_arch_tier(self.caps) else {
            return Ok(false);
        };
        debug_assert!(sfa_t.len() >= (k / FP8_BLOCK) * n_tokens);
        debug_assert!(out.len() >= n_tokens * n);

        // Three real, distinctly-compiled kernel bodies (see `fp8_bw_gemm.cu`'s
        // `sm90`/`sm100` namespaces, `62edc2f`) share this exact signature --
        // picking the function pointer pair here is the whole dispatch, the
        // call site below is otherwise identical for all three.
        #[allow(clippy::type_complexity)]
        let (workspace_fn, gemm_fn): (
            unsafe extern "C" fn(i32, i32, i32) -> usize,
            unsafe extern "C" fn(
                *const std::ffi::c_void,
                *const std::ffi::c_void,
                *const f32,
                *const f32,
                *mut f32,
                *mut std::ffi::c_void,
                i32,
                i32,
                i32,
                i32,
                cudarc::driver::sys::CUstream,
            ) -> i32,
        ) = match tier {
            GemmArchTier::Sm90 => {
                (ffi::infero_cutlass_fp8_bw_gemm_f32out_workspace_sm90, ffi::infero_cutlass_fp8_bw_gemm_f32out_sm90)
            }
            GemmArchTier::Sm100 => {
                (ffi::infero_cutlass_fp8_bw_gemm_f32out_workspace_sm100, ffi::infero_cutlass_fp8_bw_gemm_f32out_sm100)
            }
            // Operand-swapped small-M tile (`<128,32,128>`, `fp8_bw_gemm.cu`'s
            // `small_m_swap` namespace) -- the technique is vLLM's own real
            // choice at this quantization scheme (verified against vLLM's
            // current GitHub source, 2026-09-08, not guessed; see that
            // namespace's own comment), but the THRESHOLD is not: vLLM swaps
            // for the whole `M <= 64` range, while a real sweep on this
            // hardware/kernel (`examples/swap_ab_vs_small_m_bench.rs`, see
            // `SWAP_AB_MAX_TOKENS`'s own doc comment for the numbers) shows
            // swap loses badly past 32 tokens here -- do not widen this past
            // `SWAP_AB_MAX_TOKENS` without a fresh measurement backing it.
            // FFN down-projection's own exact shape (K=17408,N=5120) showed a
            // real, measured ~33% kernel-level win from swapping
            // `small_m_swap`'s scheduler to `StreamKScheduler`
            // (`examples/swap_ab_vs_small_m_bench.rs`, `small_m_swap_stream_k`
            // namespace -- `ncu` found this tile only fills 40 of 188 SMs a
            // wave, 21%, real idle capacity stream-K's own K-splitting can
            // redistribute into) -- but a real batch=16 end-to-end A/B (two
            // samples each side: baseline 573.89/585.70 tok/s vs with this
            // dispatch wired in, 573.35/586.18 tok/s) found ~0% effect,
            // indistinguishable from noise: this one projection is too small
            // a share of a decode step's total kernel time for a 33% win on
            // it alone to clear the noise floor, the same "real kernel win,
            // doesn't survive contact with end-to-end" pattern this whole
            // investigation has hit before (`gemv_f16_ksplit`, `decoupled6`,
            // etc. -- see project memory). NOT wired in; `small_m_swap_stream_k`
            // stays real, tested, available infra (correctness- and
            // memcheck/racecheck-clean, see `cutlass_fp8_gemm.rs`'s
            // `the_small_m_swap_stream_k_gemm_matches_small_m_swap`), same
            // status as those earlier kernels.
            GemmArchTier::Sm120 if n_tokens <= SWAP_AB_MAX_TOKENS => (
                ffi::infero_cutlass_fp8_bw_gemm_f32out_small_m_swap_workspace,
                ffi::infero_cutlass_fp8_bw_gemm_f32out_small_m_swap,
            ),
            // Plain (non-swapped) small-M tile (`<64,128,128>`) -- still a
            // real, measured win over the wide default tile through
            // `SMALL_M_MAX_TOKENS` (see that constant's own doc comment for
            // the numbers), just no longer the best choice below
            // `SWAP_AB_MAX_TOKENS` where swap wins instead.
            GemmArchTier::Sm120 if n_tokens <= SMALL_M_MAX_TOKENS => {
                (ffi::infero_cutlass_fp8_bw_gemm_f32out_small_m_workspace, ffi::infero_cutlass_fp8_bw_gemm_f32out_small_m)
            }
            GemmArchTier::Sm120 => (ffi::infero_cutlass_fp8_bw_gemm_f32out_workspace, ffi::infero_cutlass_fp8_bw_gemm_f32out),
        };
        self.mma_e4m3_cutlass_sfa_f32out_with(workspace_fn, gemm_fn, out, w, cw, xq, sfa_t, k, n, n_tokens, accum)?;
        Ok(true)
    }

    /// Benchmarking-only entry point: bypasses [`Self::mma_e4m3_cutlass_sfa_f32out`]'s
    /// own `SMALL_M_MAX_TOKENS`-keyed dispatch and forces one specific SM120
    /// kernel body, so a probe can time the small-M tile against the
    /// operand-swapped one head to head at the same real shape without
    /// rebuilding the crate twice. Not meant for any call site outside
    /// `examples/`; the real dispatch above is what production code path
    /// uses. Panics (via the underlying `unsafe extern "C"` call) rather than
    /// falling back if `caps` is not SM120 -- a probe caller is expected to
    /// have already checked that itself, the way `gemm_vs_vllm_probe.rs`
    /// does.
    #[allow(clippy::too_many_arguments)]
    pub fn mma_e4m3_cutlass_sfa_f32out_bench(
        &self,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        cw: &CutlassWeight,
        xq: &View<'_, u8>,
        sfa_t: &View<'_, f32>,
        k: usize,
        n: usize,
        n_tokens: usize,
        accum: bool,
        tile: BenchTile,
    ) -> Result<()> {
        #[allow(clippy::type_complexity)]
        let (workspace_fn, gemm_fn): (
            unsafe extern "C" fn(i32, i32, i32) -> usize,
            unsafe extern "C" fn(
                *const std::ffi::c_void,
                *const std::ffi::c_void,
                *const f32,
                *const f32,
                *mut f32,
                *mut std::ffi::c_void,
                i32,
                i32,
                i32,
                i32,
                cudarc::driver::sys::CUstream,
            ) -> i32,
        ) = match tile {
            BenchTile::SmallM => {
                (ffi::infero_cutlass_fp8_bw_gemm_f32out_small_m_workspace, ffi::infero_cutlass_fp8_bw_gemm_f32out_small_m)
            }
            BenchTile::SmallMSwap => (
                ffi::infero_cutlass_fp8_bw_gemm_f32out_small_m_swap_workspace,
                ffi::infero_cutlass_fp8_bw_gemm_f32out_small_m_swap,
            ),
            BenchTile::Default => (ffi::infero_cutlass_fp8_bw_gemm_f32out_workspace, ffi::infero_cutlass_fp8_bw_gemm_f32out),
            BenchTile::StreamK => (
                ffi::infero_cutlass_fp8_bw_gemm_f32out_stream_k_workspace,
                ffi::infero_cutlass_fp8_bw_gemm_f32out_stream_k,
            ),
            BenchTile::SmallMSwapStreamK => (
                ffi::infero_cutlass_fp8_bw_gemm_f32out_small_m_swap_stream_k_workspace,
                ffi::infero_cutlass_fp8_bw_gemm_f32out_small_m_swap_stream_k,
            ),
        };
        self.mma_e4m3_cutlass_sfa_f32out_with(workspace_fn, gemm_fn, out, w, cw, xq, sfa_t, k, n, n_tokens, accum)
    }

    #[allow(clippy::too_many_arguments)]
    fn mma_e4m3_cutlass_sfa_f32out_with(
        &self,
        #[allow(clippy::type_complexity)] workspace_fn: unsafe extern "C" fn(i32, i32, i32) -> usize,
        #[allow(clippy::type_complexity)] gemm_fn: unsafe extern "C" fn(
            *const std::ffi::c_void,
            *const std::ffi::c_void,
            *const f32,
            *const f32,
            *mut f32,
            *mut std::ffi::c_void,
            i32,
            i32,
            i32,
            i32,
            cudarc::driver::sys::CUstream,
        ) -> i32,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        cw: &CutlassWeight,
        xq: &View<'_, u8>,
        sfa_t: &View<'_, f32>,
        k: usize,
        n: usize,
        n_tokens: usize,
        accum: bool,
    ) -> Result<()> {
        let stream = self.dev.stream();
        let ws_bytes = unsafe { workspace_fn(n_tokens as i32, n as i32, k as i32) };

        CUTLASS_WORKSPACE.with(stream, ws_bytes.max(1), |ws_view| {
            let (a_ptr, _ra) = xq.device_ptr(stream);
            let (b_ptr, _rb) = match &cw.quants {
                Some(q) => q.device_ptr(stream),
                None => w.device_ptr(stream),
            };
            let (sfa_ptr, _rsfa) = sfa_t.device_ptr(stream);
            let (sfb_ptr, _rsfb) = cw.scale_t.device_ptr(stream);
            let (d_ptr, _rd) = out.device_ptr_mut(stream);
            let (ws_ptr, _rws) = ws_view.device_ptr_mut(stream);
            let acc = i32::from(accum);
            let status = self
                .dev
                .profile()
                .time("cutlass_fp8_gemm_f32out", stream, || {
                    let st = unsafe {
                        gemm_fn(
                            a_ptr as *const std::ffi::c_void,
                            b_ptr as *const std::ffi::c_void,
                            sfa_ptr as *const f32,
                            sfb_ptr as *const f32,
                            d_ptr as *mut f32,
                            ws_ptr as *mut std::ffi::c_void,
                            n_tokens as i32,
                            n as i32,
                            k as i32,
                            acc,
                            stream.cu_stream(),
                        )
                    };
                    Ok(st)
                })?;
            drop((_ra, _rb, _rsfa, _rsfb, _rd, _rws));
            anyhow::ensure!(status == 0, "CUTLASS f32-output GEMM returned status {status}");
            Ok(())
        })
    }
}

/// Which SM120 tile [`Kernels::mma_e4m3_cutlass_sfa_f32out_bench`] forces --
/// benchmarking-only, see that method's own doc comment.
#[derive(Clone, Copy, Debug)]
pub enum BenchTile {
    SmallM,
    SmallMSwap,
    Default,
    StreamK,
    SmallMSwapStreamK,
}
