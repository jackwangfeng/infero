//! Rust binding for the AOT-compiled CUTLASS SM120 NVFP4 (W4A4) blockscaled
//! GEMM (`cutlass/fp4_bw_gemm.cu`). Only built behind the `cutlass` feature,
//! mirroring `cutlass_fp8.rs`'s own structure (`CutlassWeight` ->
//! [`CutlassFp4Weight`], `prepare_cutlass_weight` ->
//! [`Kernels::prepare_cutlass_fp4_weight`], `mod ffi`) -- see that file's own
//! doc comments for the parts this one does not repeat.
//!
//! Two real differences from the FP8 file, both load-bearing, and both
//! resolutions of this task's own brief (which pointed at real code rather
//! than specifying an answer):
//!
//! 1. **`sfa`/`sfb`'s element type.** `cutlass_fp8.rs`'s own
//!    `mma_e4m3_cutlass_sfa_f32out` takes `sfa_t: &View<'_, f32>`
//!    (`cutlass_fp8.rs:687`, C ABI `*const f32` at `cutlass_fp8.rs:248`) --
//!    real, plain `f32`, because `fp8_bw_gemm.cu`'s blockwise scaling is a
//!    SOFTWARE scale (`ScaleConfig`'s scale values are plain
//!    `ElementAccumulator = float`, matching `crate::fp8::scale_grid`'s own
//!    plain-f32 128x128 grid) multiplied in by the epilogue after the
//!    tensor-core MMA runs, not consumed BY the MMA instruction itself. This
//!    task's brief stub signature (`sfa: &View<f32>`) copies that shape, but
//!    NVFP4's block-scaled MMA is a genuinely different mechanism: the scale
//!    factor is consumed NATIVELY by the tensor-core instruction alongside
//!    the 4-bit data, and CUTLASS types that native scale factor as
//!    `cutlass::nv_float4_t<cutlass::float_e2m1_t>::ScaleFactorType =
//!    cutlass::float_ue4m3_t` (`include/cutlass/float_subbyte.h:510-517`,
//!    fetched via `gh api` this session) -- a real, distinct, byte-sized
//!    8-bit float, not `float`. This is exactly what
//!    [`crate::Kernels::quantize_act_e2m1_cutlass`] (Task 5) already
//!    produces as raw `u8` scale bytes. So every `sfa`/`sfb` in this file is
//!    `&View<'_, u8>`, not `&View<'_, f32>` -- the brief's own stub was a
//!    copy-paste of the FP8 shape, not a real NVFP4 requirement.
//! 2. **The scale layout needs a real repack, unlike FP8's plain transpose.**
//!    `quantize_act_e2m1_cutlass`'s own doc comment says its `xq_scale`
//!    output is deliberately left in the same plain `[rows, blocks]`
//!    row-major layout `dequant_f4e2m1`'s `scale` argument uses, "deferring
//!    any CUTLASS-required swizzle/repack to this task" -- so this file adds
//!    a call to `swizzle_sf_e2m1` (`cu/fp4.cu`, NVRTC-compiled like every
//!    other kernel in this crate except the CUTLASS GEMM body itself,
//!    exactly the split `fp8.cu`'s `unrepack_rows_e4m3`/
//!    `transpose_scale_b_f32` already use for CUTLASS's FP8 path) -- a real,
//!    dual-sourced repack kernel; see its own doc comment for the exact
//!    offset formula and both real sources it was cross-checked against
//!    (CUTLASS's own `Sm1xxBlockScaledConfig` CuTe layout algebra, and
//!    vLLM's own real `cvt_quant_to_fp4_get_sf_out_offset` CUDA kernel).
//!
//! A third real gap in the brief's own stub, found (not specified) while
//! reading `fp4.rs::dequant_f4e2m1_row`: NVFP4's real two-level weight scale
//! (`value = e2m1_value * block_scale * weight_scale_2`) has a per-tensor
//! `weight_scale_2` factor that is NOT baked into the per-block scale bytes
//! `swizzle_sf_e2m1` repacks -- and CUTLASS's block-scaled MMA only ever
//! consumes the per-block factor during the tensor-core instruction itself
//! (confirmed against vLLM's own real NVFP4 CUTLASS dispatch,
//! `csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_kernels.cu`,
//! fetched this session: it passes a separate real `alpha` tensor alongside
//! `A_sf`/`B_sf`, never folded into either). So `weight_scale_2` has to be
//! applied as this GEMM's own `alpha` epilogue scalar -- [`CutlassFp4Weight`]
//! caches it at weight-load time (the caller already has to extract it from
//! `WeightType::F4E2M1`'s own trailing scalars for `dequant_f4e2m1`'s
//! existing `scale2` parameter, so this adds no new extraction burden), and
//! [`Kernels::mma_e2m1_cutlass_sfa_f32out`]'s own public signature stays
//! exactly the shape the brief specifies (no new visible parameter) by
//! reading it from `cw` internally.
//!
//! **CORRECTION 1 (found during the end-to-end NVFP4 garbage-output
//! investigation -- real logits ~30-50x too small at every prompt):** the
//! claim originally written here -- that `quantize_act_e2m1_cutlass`'s own
//! `input_scale` "needs no equivalent post-multiply" because its per-block
//! scale bytes "already fully absorb that factor" -- was WRONG. Absorbing
//! `input_scale` into the per-block scale byte at *encode* time (exactly
//! what `scale_f32 = input_scale * vec_max / 6` does) is precisely why a
//! *decode*-time correction is still needed: CUTLASS's native MMA only ever
//! computes `code * sf_byte_raw`, which mechanically equals `x_true *
//! input_scale`, not `x_true`. This correction made `alpha` divide by
//! `input_scale` (`cw.scale2 / cw.input_scale`) -- which fixed lm_head's
//! visible logit magnitude, but see Correction 2: it was still wrong, and it
//! left the real bug (an all-zero FFN) in place.
//!
//! **CORRECTION 2 (the real fix, found by whole-branch code review): the
//! quantizer's `global_scale` argument and `alpha`'s formula had each other's
//! roles.** Real, fetched vLLM source (`modelopt.py`'s `apply_weights`) uses
//! the checkpoint's single `input_scale` scalar (call it `S`) two different
//! ways, and Correction 1 conflated them:
//!
//!   - `layer.alpha = S * layer.weight_global_scale` -- a **MULTIPLY** by the
//!     raw `S`, not a divide.
//!   - `layer.input_global_scale_inv = 1.0 / S` -- the quantizer's own
//!     `global_scale` argument is `S`'s **reciprocal**, never `S` itself.
//!
//! Correction 1 passed the raw `cw.input_scale()` (`S`) as the quantizer's
//! `global_scale` at both real call sites (`Model::matmul_pre`, the lm_head
//! dispatch) -- correct for `alpha`'s role, wrong for the quantizer's -- and
//! then computed `alpha = cw.scale2 / cw.input_scale`, applying `S`'s
//! reciprocal role a second time on top. The quantizer's own per-block scale
//! byte is `S_arg * vec_max / 6`; feeding it `S` directly (order `1e-3` on
//! this checkpoint's real FFN tensors) drives that byte to e4m3's underflow
//! floor for any block whose `vec_max` is not enormous -- provably, for this
//! checkpoint's real `gate_proj` (`input_scale≈0.0014`, `amax≈3.763`, below
//! the `amax<3.969` all-zero threshold this format's math implies): **every
//! FFN activation block in all 64 layers quantized to exactly zero.** The
//! FFN sub-layer silently contributed nothing to any forward pass; only
//! lm_head (`input_scale≈0.02167`, clear of that floor) looked correct,
//! which is why the garbage-output investigation's lm_head-only ground-truth
//! checks never caught it. `alpha`'s algebra cannot rescue an operand that
//! is already zero, which is why Correction 1 alone did not fix generation.
//! Fixed now: both quantizer call sites pass `1.0 / cw.input_scale()`, and
//! [`Kernels::mma_e2m1_cutlass_sfa_f32out`] computes `cw.scale2 *
//! cw.input_scale` (a multiply) as `alpha`. See that function's own doc
//! comment, and `fp4_quantize_act.rs`'s real-magnitude regression test, for
//! the rest of the derivation.

use anyhow::{Context, Result};
use cudarc::driver::{DevicePtr, DevicePtrMut};
use infero_gpu::{Buf, KernelArg, LaunchConfig, View, ViewMut};

use crate::fp4::F4E2M1_BLOCK;
use crate::{Kernels, fp4_src};

/// CUTLASS's own real operand alignment for this GEMM's `K` axis:
/// `16 * 8 / sizeof_bits(e2m1) = 32` elements (`fp4_bw_gemm.cu`'s own
/// `AlignmentA`/`AlignmentB`, 128 bits / 4-bit elements) -- not this format's
/// own 16-element block size, though 32 is a multiple of it, so any `k`
/// satisfying this is automatically block-aligned too.
pub const FP4_GEMM_K_ALIGN: usize = 32;

/// Threads a block for [`swizzle_sf_e2m1`]'s row-dimension tiling, matching
/// `fp4.rs`'s `FP4_DEQUANT_BLOCK`/`FP4_QUANT_BLOCK`.
const FP4_SWIZZLE_BLOCK: u32 = 256;

/// Padded byte size of the swizzled scale buffer CUTLASS's canonical
/// `Sm1xxBlockScaledConfig<16>` layout needs for `rows` (M for the
/// activation SFA, N for the weight SFB) and `blocks`
/// (`k.div_ceil(F4E2M1_BLOCK)`) -- `rows.div_ceil(128) * blocks.div_ceil(4) *
/// 512` bytes. See `cu/fp4.cu`'s `swizzle_sf_e2m1` doc comment for the real,
/// dual-sourced derivation (CUTLASS's own CuTe `SfAtom` layout algebra, cross-
/// checked against vLLM's own real `computeSwizzledSFShape`, which returns
/// this exact same `(round_up(rows,128), round_up(blocks,4))` shape).
fn swizzled_sf_bytes(rows: usize, blocks: usize) -> usize {
    rows.div_ceil(128) * blocks.div_ceil(4) * 512
}

/// A grow-only scratch buffer, reused across calls instead of
/// `cudaMalloc`/`cudaFree`-ing fresh each time -- same reasoning and same
/// single-stream assumption as `cutlass_fp8.rs`'s own `Scratch` (this
/// crate's forward pass issues these calls sequentially on one stream, never
/// concurrently). Duplicated here rather than reused from `cutlass_fp8.rs`
/// (whose `Scratch` is module-private) to avoid introducing a shared
/// cross-module abstraction this task was not asked to build.
struct Scratch {
    buf: std::sync::Mutex<Option<Buf<u8>>>,
}
impl Scratch {
    const fn new() -> Self {
        Self { buf: std::sync::Mutex::new(None) }
    }
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
static CUTLASS_FP4_WORKSPACE: Scratch = Scratch::new();
static CUTLASS_FP4_SFA: Scratch = Scratch::new();

/// A [`crate::WeightType::F4E2M1`] matrix's precomputed CUTLASS-side state:
/// the weight's own per-block scale grid repacked into CUTLASS's real
/// swizzled layout (see this module's own doc comment), plus the
/// checkpoint's `weight_scale_2` scalar, cached here for use (divided by the
/// activation's own `input_scale`) as the GEMM's `alpha` -- see this
/// module's doc comment (the "CORRECTION" paragraph) and
/// [`Kernels::mma_e2m1_cutlass_sfa_f32out`]'s own doc comment for why BOTH
/// factors are needed. `F4E2M1`'s own on-disk quant bytes need no repack or
/// second copy at all -- unlike `F8E4M3`, that format has no
/// interleaved-vs-plain distinction ([`crate::WeightType::F4E2M1`]'s own doc
/// comment: always plain), so [`Kernels::mma_e2m1_cutlass_sfa_f32out`] reads
/// the weight's own device buffer directly.
pub struct CutlassFp4Weight {
    scale_sfb: Buf<u8>,
    scale2: f32,
    /// The checkpoint's `input_scale` scalar. Read back both by the forward
    /// pass's own `quantize_act_e2m1_cutlass` call (which needs it as that
    /// kernel's own `input_scale` argument) AND, since the fix described in
    /// this module's own "CORRECTION" doc comment, by
    /// [`Kernels::mma_e2m1_cutlass_sfa_f32out`] itself, which divides it into
    /// `scale2` to form the GEMM's real `alpha` -- it is NOT a pure
    /// activation-side value with no bearing on this GEMM, despite what an
    /// earlier version of this doc comment claimed.
    input_scale: f32,
    k: usize,
    n: usize,
}

impl CutlassFp4Weight {
    /// The checkpoint's `input_scale` scalar (see the field's own doc
    /// comment for why it lives here rather than on `Matrix`).
    pub fn input_scale(&self) -> f32 {
        self.input_scale
    }
}

impl Kernels {
    /// Prepares a [`crate::WeightType::F4E2M1`] matrix's [`CutlassFp4Weight`].
    /// Call once per weight matrix, not per forward pass.
    ///
    /// `w` is the matrix's own device buffer, holding (per
    /// [`crate::WeightType::F4E2M1`]'s own layout) `n * k.div_ceil(2)` packed
    /// quant bytes, then `n * k.div_ceil(F4E2M1_BLOCK)` f8_e4m3 block-scale
    /// bytes, then two trailing f32 scalars this function does not read
    /// itself (the caller extracts them the same way it already does for
    /// [`Kernels::dequant_f4e2m1`]'s own `scale2` parameter). `scale2` is
    /// the checkpoint's `weight_scale_2`; `input_scale` is cached as-is on
    /// the returned value (see [`CutlassFp4Weight::input_scale`]'s own doc
    /// comment).
    pub fn prepare_cutlass_fp4_weight(
        &self,
        w: &View<'_, u8>,
        k: usize,
        n: usize,
        scale2: f32,
        input_scale: f32,
    ) -> Result<CutlassFp4Weight> {
        anyhow::ensure!(
            k.is_multiple_of(FP4_GEMM_K_ALIGN),
            "the CUTLASS NVFP4 GEMM's own operand alignment is {FP4_GEMM_K_ALIGN} elements along k; got k={k}"
        );
        let stream = self.dev.stream();
        let blocks = k.div_ceil(F4E2M1_BLOCK);
        let quant_bytes = n * k.div_ceil(2);
        let scale_bytes = n * blocks;
        anyhow::ensure!(
            w.len() >= quant_bytes + scale_bytes,
            "F4E2M1 weight buffer holds {} bytes, need at least {} for its own quants + scale grid",
            w.len(),
            quant_bytes + scale_bytes
        );

        let sfb_bytes = swizzled_sf_bytes(n, blocks);
        let mut scale_sfb = stream.alloc_zeros::<u8>(sfb_bytes)?;
        let sf_flat = w.slice(quant_bytes..quant_bytes + scale_bytes);
        launch_swizzle(self, &mut scale_sfb.as_view_mut(), &sf_flat, n, blocks)?;

        self.dev.stream().synchronize().context("preparing a CUTLASS NVFP4 weight")?;
        Ok(CutlassFp4Weight { scale_sfb, scale2, input_scale, k, n })
    }
}

/// Launches `swizzle_sf_e2m1` (`cu/fp4.cu`), repacking `sf_flat`'s plain
/// `[rows, blocks]` row-major bytes into `sf_swizzled`'s CUTLASS canonical
/// layout. `sf_swizzled` must already hold at least
/// [`swizzled_sf_bytes`]`(rows, blocks)` bytes; the launch covers that whole
/// padded extent, so callers do not need to pre-zero it.
fn launch_swizzle(
    kernels: &Kernels,
    sf_swizzled: &mut ViewMut<'_, u8>,
    sf_flat: &View<'_, u8>,
    rows: usize,
    blocks: usize,
) -> Result<()> {
    let stream = kernels.dev.stream();
    let f = kernels.dev.kernels().get("infero_fp4", fp4_src(), "swizzle_sf_e2m1")?;
    let rows_padded = rows.div_ceil(128) * 128;
    let blocks_padded = blocks.div_ceil(4) * 4;
    let cfg = LaunchConfig {
        grid_dim: ((rows_padded as u32).div_ceil(FP4_SWIZZLE_BLOCK), blocks_padded as u32, 1),
        block_dim: (FP4_SWIZZLE_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let (ri, bi) = (rows as i32, blocks as i32);
    let mut b = stream.launch_builder(&f);
    b.arg(sf_swizzled).arg(sf_flat).arg(&ri).arg(&bi);
    kernels
        .dev
        .profile()
        .time("cutlass_fp4_swizzle_sf", stream, || {
            unsafe { b.launch(cfg) }.context("swizzle_sf_e2m1")?;
            Ok(())
        })
}

mod ffi {
    use std::ffi::c_void;
    unsafe extern "C" {
        pub fn infero_cutlass_fp4_bw_gemm_f32out_workspace(m: i32, n: i32, k: i32) -> usize;
        #[allow(clippy::too_many_arguments)]
        pub fn infero_cutlass_fp4_bw_gemm_f32out(
            a: *const c_void,
            b: *const c_void,
            sfa: *const c_void,
            sfb: *const c_void,
            d: *mut f32,
            workspace: *mut c_void,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            accum: i32,
            stream: cudarc::driver::sys::CUstream,
        ) -> i32;
    }
}

impl Kernels {
    /// The NVFP4 (W4A4) counterpart of
    /// [`crate::Kernels::mma_e4m3_cutlass_sfa_f32out`]: CUTLASS's own
    /// epilogue writes `out` (f32) directly, no bf16/f32 scratch, no
    /// separate store/upconvert kernel afterward. `cw` must already be
    /// prepared (build once at load time with
    /// [`Kernels::prepare_cutlass_fp4_weight`]).
    ///
    /// `xq` is [`Kernels::quantize_act_e2m1_cutlass`]'s own packed e2m1
    /// output, used as-is (no padding: CUTLASS accepts any `n_tokens`, the
    /// same "no `M`-multiple-of-anything" contract
    /// `mma_e4m3_cutlass_sfa_f32out`'s own doc comment establishes for FP8,
    /// not re-verified here since this box cannot run NVFP4 tensor cores at
    /// all -- flagged as unverified in this task's own report). `sfa` is
    /// that same call's own `xq_scale` output, in ITS plain `[n_tokens,
    /// k.div_ceil(F4E2M1_BLOCK)]` row-major layout -- this function repacks
    /// it into CUTLASS's swizzled layout internally (see this module's own
    /// doc comment for why that repack belongs here, not in Task 5).
    ///
    /// `accum` maps to CUTLASS's own `beta` (1.0 to add into `out`'s
    /// existing contents, 0.0 to overwrite), the same convention
    /// `mma_e4m3_cutlass_sfa_f32out`'s own `accum` uses. CUTLASS's own
    /// `alpha` is `weight_scale_2 / input_scale` (`cw.scale2 /
    /// cw.input_scale`, computed fresh on every call rather than cached on
    /// `cw` since `input_scale` is per-activation-quantize-call in spirit
    /// even though this GEMM's own `cw` happens to be the one place both
    /// live) -- see this module's own doc comment for the full derivation of
    /// why BOTH factors are needed, not `weight_scale_2` alone.
    ///
    /// `Ok(false)` (never a wrong answer) if this GPU has no NVFP4 tensor
    /// cores (`!self.dev.caps().fp4`, i.e. below sm_120) or the shape is one
    /// this path does not handle -- caller must fall back to a dequantize
    /// path (no NVFP4 mat-vec kernel exists in this crate as of this task;
    /// see the task-6 report for this gap).
    #[allow(clippy::too_many_arguments)]
    pub fn mma_e2m1_cutlass_sfa_f32out(
        &self,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        cw: &CutlassFp4Weight,
        xq: &View<'_, u8>,
        sfa: &View<'_, u8>,
        k: usize,
        n: usize,
        n_tokens: usize,
        accum: bool,
    ) -> Result<bool> {
        anyhow::ensure!(
            cw.k == k && cw.n == n,
            "CutlassFp4Weight is [{}, {}], called with k={k} n={n}",
            cw.n,
            cw.k
        );
        if !k.is_multiple_of(FP4_GEMM_K_ALIGN) || n_tokens == 0 {
            return Ok(false);
        }
        // No NVFP4 tensor cores below sm_120 -- same "caller falls back"
        // contract as the shape check above, not a crash or a mismatched
        // launch. See `crates/cuda/src/backend.rs`'s `Caps::fp4`.
        if !self.dev.caps().fp4 {
            return Ok(false);
        }

        let blocks = k.div_ceil(F4E2M1_BLOCK);
        debug_assert!(sfa.len() >= n_tokens * blocks);
        debug_assert!(xq.len() >= n_tokens * k.div_ceil(2));
        debug_assert!(out.len() >= n_tokens * n);

        // GEMM `alpha`: `weight_scale_2 * input_scale` -- a MULTIPLY, not a
        // divide, and this is the second and final correction to this
        // formula (the first, `cw.scale2 / cw.input_scale`, was itself
        // wrong -- see below). Real, fetched vLLM source
        // (`modelopt.py`'s `apply_weights`) is unambiguous:
        //
        //   input_global_scale = layer.input_scale.max()      # == cw.input_scale, the raw checkpoint scalar
        //   layer.alpha = input_global_scale * layer.weight_global_scale   # MULTIPLY
        //   layer.input_global_scale_inv = 1.0 / input_global_scale        # fed to the ACTIVATION QUANTIZER, not alpha
        //
        // The two scalars vLLM computes from the checkpoint's single
        // `input_scale` play different roles: `input_global_scale` (the raw
        // value) feeds `alpha` directly; its reciprocal feeds the quantizer.
        // The previous version of this fix conflated them -- it left the
        // quantizer call site (`Model::matmul_pre`, the lm_head dispatch)
        // passing the raw `cw.input_scale()` as the quantizer's
        // `global_scale`, correct only for `alpha`'s role, and then divided
        // by it here too, which is `alpha`'s role applied a second time.
        // Concretely: `quantize_act_e2m1_cutlass`'s per-block scale byte is
        // `input_arg * vec_max / 6` where `input_arg` is whatever the caller
        // passes -- with the caller (wrongly) passing the raw `input_scale`
        // instead of its reciprocal, every real checkpoint's calibrated
        // `input_scale` (order `1e-3`) drove that scale byte to e4m3's
        // underflow floor for any activation block whose `vec_max` wasn't
        // enormous -- provably: `gate_proj`'s real `input_scale≈0.0014`
        // (`amax≈3.763`) is below this checkpoint's own zero-everything
        // threshold of `input_scale < 15.75/16128*... ` i.e. `amax < 3.969`,
        // so EVERY FFN activation block in all 64 layers quantized to
        // exactly zero -- the FFN sub-layer contributed nothing, silently,
        // while lm_head (whose `input_scale≈0.02167` stays well clear of
        // that floor) looked fine. `alpha`'s own algebra could not rescue
        // zeroed operands, which is why the first alpha fix alone did not
        // fix end-to-end generation. See `Model::matmul_pre`'s and the
        // lm_head dispatch's own call sites (both now pass
        // `1.0 / cw.input_scale()`), and `fp4_quantize_act.rs`'s new
        // real-magnitude regression test, for the other half of this fix.
        let alpha = cw.scale2 * cw.input_scale;
        debug_assert!(
            alpha.is_finite(),
            "NVFP4 GEMM alpha is non-finite (scale2={}, input_scale={}) -- a zero/garbage \
             input_scale would silently produce Inf/NaN logits instead of a loud failure",
            cw.scale2,
            cw.input_scale
        );

        let stream = self.dev.stream();
        let sfa_bytes = swizzled_sf_bytes(n_tokens, blocks);
        let ws_bytes = unsafe { ffi::infero_cutlass_fp4_bw_gemm_f32out_workspace(n_tokens as i32, n as i32, k as i32) };

        CUTLASS_FP4_SFA.with(stream, sfa_bytes, |sfa_swizzled| {
            launch_swizzle(self, sfa_swizzled, sfa, n_tokens, blocks)?;

            CUTLASS_FP4_WORKSPACE.with(stream, ws_bytes.max(1), |ws_view| {
                let (a_ptr, _ra) = xq.device_ptr(stream);
                let (b_ptr, _rb) = w.device_ptr(stream);
                let sfa_swizzled_view = sfa_swizzled.as_view();
                let (sfa_ptr, _rsfa) = sfa_swizzled_view.device_ptr(stream);
                let (sfb_ptr, _rsfb) = cw.scale_sfb.device_ptr(stream);
                let (d_ptr, _rd) = out.device_ptr_mut(stream);
                let (ws_ptr, _rws) = ws_view.device_ptr_mut(stream);
                let acc = i32::from(accum);
                let status = self
                    .dev
                    .profile()
                    .time("cutlass_fp4_gemm_f32out", stream, || {
                        let st = unsafe {
                            ffi::infero_cutlass_fp4_bw_gemm_f32out(
                                a_ptr as *const std::ffi::c_void,
                                b_ptr as *const std::ffi::c_void,
                                sfa_ptr as *const std::ffi::c_void,
                                sfb_ptr as *const std::ffi::c_void,
                                d_ptr as *mut f32,
                                ws_ptr as *mut std::ffi::c_void,
                                n_tokens as i32,
                                n as i32,
                                k as i32,
                                alpha,
                                acc,
                                stream.cu_stream(),
                            )
                        };
                        Ok(st)
                    })?;
                drop((_ra, _rb, _rsfa, _rsfb, _rd, _rws));
                anyhow::ensure!(status == 0, "CUTLASS NVFP4 GEMM returned status {status}");
                Ok(())
            })
        })?;
        Ok(true)
    }
}
