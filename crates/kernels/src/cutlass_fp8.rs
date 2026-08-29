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

use crate::fp8::{FP8_BLOCK, ROW_GROUP};
use crate::{Kernels, fp8_src};

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
    }
}

impl Kernels {
    /// [`Kernels::mma_e4m3_block`], routed through the AOT CUTLASS GEMM
    /// instead of the hand-written tensor-core kernel -- ~10x its measured
    /// TFLOPS on the shapes this model uses (see the project memory this
    /// came out of). `cw` must already be prepared (build once at load time
    /// with [`Kernels::prepare_cutlass_weight`]); what remains per call is
    /// transposing/padding the *activation* scale and padding `n_tokens` up
    /// to CUTLASS's 128-row minimum, both `O(n_tokens*k/128)` or smaller
    /// against an `O(n*k*n_tokens)` GEMM.
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
        let m_pad = n_tokens.next_multiple_of(128);

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
        self.mma_e4m3_cutlass_sfa(out, w, cw, xq, &sfa_t.as_view(), k, n, n_tokens, accum)
    }

    /// Same as [`Self::mma_e4m3_cutlass`], but `sfa_t` is already in the
    /// transposed-and-padded `[groups, m_pad]` layout (`m_pad =
    /// n_tokens.next_multiple_of(128)`) — the caller's own quantizer wrote
    /// it directly (see [`Kernels::quantize_act_e4m3_cutlass`]), so there is
    /// no separate `[n_tokens, groups]` scale to transpose here.
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
        let m_pad = n_tokens.next_multiple_of(128);
        debug_assert!(sfa_t.len() >= (k / FP8_BLOCK) * m_pad);

        // Pad activations: [n_tokens,k] -> [m_pad,k], zero rows past
        // n_tokens. `CUTLASS_A_PAD` is reused across calls (see its doc
        // comment), so the tail past `n_tokens` -- which a *fresh*
        // `alloc_zeros` would already be zero, but reused scratch might
        // still hold a previous call's rows -- needs an explicit clear
        // whenever this call is narrower than the last one that grew it.
        let a_pad_bytes = m_pad * k;
        let d_pad_bytes = m_pad * n * 2; // bf16, held as raw bits
        let ws_bytes = unsafe { ffi::infero_cutlass_fp8_bw_gemm_workspace(m_pad as i32, n as i32, k as i32) };

        CUTLASS_A_PAD.with(stream, a_pad_bytes, |a_pad_bytes_view| {
            if m_pad > n_tokens {
                stream
                    .memset_zeros(&mut a_pad_bytes_view.slice_mut(n_tokens * k..m_pad * k))
                    .context("clearing the CUTLASS activation pad tail")?;
            }
            stream
                .memcpy_dtod(&xq.slice(0..n_tokens * k), &mut a_pad_bytes_view.slice_mut(0..n_tokens * k))
                .context("padding activations for the CUTLASS GEMM")?;

            CUTLASS_D_PAD.with(stream, d_pad_bytes, |d_pad_view| {
                CUTLASS_WORKSPACE.with(stream, ws_bytes.max(1), |ws_view| {
                    let (a_ptr, _ra) = a_pad_bytes_view.device_ptr(stream);
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
                    drop((_ra, _rb, _rsfa, _rsfb, _rd, _rws));
                    anyhow::ensure!(status == 0, "CUTLASS GEMM returned status {status}");

                    // 6. Upconvert bf16 -> f32 into `out`, discarding the padded rows.
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
        })?;

        Ok(true)
    }
}
