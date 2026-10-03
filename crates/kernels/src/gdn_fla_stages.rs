//! The rest of FLA's real GDN prefill pipeline, AOT-compiled and FFI-called
//! the same way `gdn_triton_aot.rs` already does for kernel-2 (the sequential
//! state-pass) -- see that file's own doc comment for the "Route B" pattern
//! this extends, and this module's own comment on each stage for where its
//! compiled artifact and winning autotune config came from.
//!
//! Five more real Triton kernels, in their real pipeline order:
//!
//!   1. `chunk_local_cumsum_scalar_kernel` (cumsum.py) -- chunk-local cumsum
//!      of the raw per-token gate `g`.
//!   2. `chunk_scaled_dot_kkt_fwd_kernel` (chunk_scaled_dot_kkt.py) -- builds
//!      the chunk-local system matrix `A = tril(diag(beta) @ K @ Kᵀ, -1)`.
//!   3. `merge_16x16_to_64x64_inverse_kernel` (solve_tril.py) -- inverts
//!      `I + A`; this is the real BT=64 dispatch target, not a generic
//!      "solve_tril" kernel (see `solve_tril`'s own Python dispatch).
//!   4. `recompute_w_u_fwd_kernel` (wy_fast.py) -- reconstructs `W`/`U` from
//!      the inverted system matrix.
//!   5. `chunk_fwd_kernel_o` (chunk_o.py) -- the output pass (kernel-3
//!      equivalent): the causal intra-chunk term plus the cross-chunk
//!      history term reading kernel-2's own `h` output.
//!
//! Kernel-2 itself (`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`) is
//! already bound in `gdn_triton_aot.rs` and reused here unchanged.
//!
//! `l2norm_fwd` is deliberately NOT part of this pipeline: infero already
//! does the equivalent normalization itself, in place, before any of these
//! stages run (`Kernels::gdn_qk_l2norm`, `gdn_qk_l2norm_f32` in `gdn.cu`) --
//! same eps convention, same "scale lands on q only" convention as FLA's own
//! default. Adding a second call here would double-apply it. Callers of
//! [`Kernels::gdn_full_triton_pipeline`] MUST have already run
//! `gdn_qk_l2norm` on `qkv` (the existing 3-kernel-split callers already do).
//!
//! # The real layout-conversion step
//!
//! All 5 stages above, plus kernel-2, want FLA's own real `(B=1, T, heads,
//! dim)` bf16 tensor layout with a chunk-local system matrix and separate
//! `q`/`k`/`v`/`beta`/`g` tensors -- not this engine's packed `[total_tokens,
//! stride]` f32 row, and not this engine's persistent `[heads, dk, dv]` f32
//! state (which is `S[k][v]`; FLA's `h0`/`ht` are `S[v][k]`, the transpose --
//! see `gdn.cu`'s new `gdn_fla_transpose_hab_to_hba_f32` for how that was
//! confirmed and where the transpose actually happens). Five small new
//! kernels in `gdn.cu` (`gdn_fla_extract_qk_f32_to_bf16`,
//! `gdn_fla_extract_v_f32_to_bf16`, `gdn_fla_cast_f32_to_bf16`,
//! `gdn_fla_cast_bf16_to_f32`, `gdn_fla_transpose_hab_to_hba_f32`) do this
//! real conversion work; see `gdn_full_triton_pipeline`'s own body for where
//! each runs. These are real launches with real bandwidth cost -- included
//! in this pipeline's own benchmark, not hidden from it.
//!
//! # Scope (matches `gdn_triton_aot.rs`'s own fixed shape)
//!
//! Fixed to `B=1` (one sequence a call, like `gdn_chunk_ab`/`gdn_scan_finish`
//! elsewhere in this crate), `H=48, Hg=16, K=128, V=128, BT=64`,
//! `IS_VARLEN=0`, and `v_tiled=false` (FLA's own `v` layout has no tiled-heads
//! reading; a tiled `v` would need a different extraction kernel, not
//! attempted here). Every intermediate FLA-layout buffer lives in
//! [`GdnFlaScratch`], caller-owned and reused across calls -- the same
//! convention `gdn_chunk_split3_delta_rule`'s own `w_buf`/`u_buf`/
//! `delta_buf`/`s_before_buf` and `GdnActs`'s (`crates/model/src/lib.rs`)
//! load-time allocation already use, and for the same reason: `stream.
//! alloc_zeros` on all fifteen of these is a real memset, not free, and this
//! module's own benchmark example measures it. `GdnFlaScratch` allocates once
//! for a `max_tokens` bound and any call with `t <= max_tokens` reuses it.

use anyhow::Result;
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{DevicePtr, DevicePtrMut};
use infero_gpu::{Buf, Device, KernelArg, LaunchConfig, View, ViewMut};

use crate::gdn::SeqLayout;
use crate::gdn_triton_aot::{self, BT, H, HG, K, V};
use crate::{Kernels, gdn_src};

/// Caller-owned scratch for [`Kernels::gdn_full_triton_pipeline`], sized once
/// for up to `max_tokens` tokens and reused across calls -- see this module's
/// own doc comment for why. Every field here is exactly the buffer
/// `gdn_full_triton_pipeline` used to `stream.alloc_zeros` fresh each call;
/// see that function's own body for what each one holds and in which stage.
///
/// `h_bf` is the one field sized off chunk count (`max_tokens.div_ceil(BT)`)
/// rather than token count directly -- kernel-2's own per-chunk history
/// output. `h0_f32`/`ht_f32` are the two fields NOT sized off `max_tokens` at
/// all: they hold one `[H,V,K]` state, the same fixed size regardless of how
/// many tokens a call processes.
pub struct GdnFlaScratch {
    max_tokens: usize,
    q_bf: Buf<u16>,
    k_bf: Buf<u16>,
    v_bf: Buf<u16>,
    beta_bf: Buf<u16>,
    g_raw_bf: Buf<u16>,
    g_cs: Buf<f32>,
    a_mat: Buf<f32>,
    ai_mat: Buf<u16>,
    w_bf: Buf<u16>,
    u_bf: Buf<u16>,
    h_bf: Buf<u16>,
    v_new_bf: Buf<u16>,
    h0_f32: Buf<f32>,
    ht_f32: Buf<f32>,
    o_bf: Buf<u16>,
}

impl GdnFlaScratch {
    /// Allocates every intermediate buffer [`Kernels::gdn_full_triton_pipeline`]
    /// needs, sized for any single call with `t <= max_tokens`. Call once (load
    /// time, the same convention `GdnActs` uses in `crates/model/src/lib.rs`)
    /// and reuse across calls/variants.
    pub fn new(dev: &Device, max_tokens: usize) -> Result<Self> {
        anyhow::ensure!(max_tokens > 0, "GdnFlaScratch: max_tokens must be > 0");
        let stream = dev.stream();
        let max_nt = max_tokens.div_ceil(BT);
        Ok(Self {
            max_tokens,
            q_bf: stream.alloc_zeros::<u16>(max_tokens * HG * K)?,
            k_bf: stream.alloc_zeros::<u16>(max_tokens * HG * K)?,
            v_bf: stream.alloc_zeros::<u16>(max_tokens * H * V)?,
            beta_bf: stream.alloc_zeros::<u16>(max_tokens * H)?,
            g_raw_bf: stream.alloc_zeros::<u16>(max_tokens * H)?,
            g_cs: stream.alloc_zeros::<f32>(max_tokens * H)?,
            a_mat: stream.alloc_zeros::<f32>(max_tokens * H * BT)?,
            ai_mat: stream.alloc_zeros::<u16>(max_tokens * H * BT)?,
            w_bf: stream.alloc_zeros::<u16>(max_tokens * H * K)?,
            u_bf: stream.alloc_zeros::<u16>(max_tokens * H * V)?,
            h_bf: stream.alloc_zeros::<u16>(max_nt * H * V * K)?,
            v_new_bf: stream.alloc_zeros::<u16>(max_tokens * H * V)?,
            h0_f32: stream.alloc_zeros::<f32>(H * V * K)?,
            ht_f32: stream.alloc_zeros::<f32>(H * V * K)?,
            o_bf: stream.alloc_zeros::<u16>(max_tokens * H * V)?,
        })
    }

    /// The bound this scratch was sized for -- a call with more tokens than
    /// this must allocate a bigger `GdnFlaScratch`, not reuse this one.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }
}

mod ffi {
    use cudarc::driver::sys::{CUdeviceptr, CUstream};

    unsafe extern "C" {
        // cumsum.py, `chunk_local_cumsum_scalar_kernel`. `s`: bf16 raw g
        // `[T,H]`. `o`: f32 cumsum'd g `[T,H]`, this pipeline's `g` from here
        // on.
        pub fn cumsum_88a339d3_0d1d2d3d(
            stream: CUstream,
            s: CUdeviceptr,
            o: CUdeviceptr,
            cu_seqlens: CUdeviceptr,
            chunk_indices: CUdeviceptr,
            t: i32,
        ) -> i32;

        // chunk_scaled_dot_kkt.py, `chunk_scaled_dot_kkt_fwd_kernel`. `k`:
        // bf16 `[T,Hg,K]`. `beta`: bf16 `[T,H]`. `g`: f32 cumsum'd `[T,H]`.
        // `a`: f32 output `[T,H,BT]`.
        pub fn kkt_2e3078a5_0d1d2d3d4d5d(
            stream: CUstream,
            k: CUdeviceptr,
            beta: CUdeviceptr,
            g: CUdeviceptr,
            a: CUdeviceptr,
            cu_seqlens: CUdeviceptr,
            chunk_indices: CUdeviceptr,
            t: i32,
        ) -> i32;

        // solve_tril.py, `merge_16x16_to_64x64_inverse_kernel` -- the real
        // BT=64 dispatch target, not a generic "solve_tril" symbol. `a`: f32
        // in `[T,H,BT]`. `ai`: bf16 out `[T,H,BT]`.
        pub fn solve_tril64_ce06b0ca_0d1d2d3d(
            stream: CUstream,
            a: CUdeviceptr,
            ai: CUdeviceptr,
            cu_seqlens: CUdeviceptr,
            chunk_indices: CUdeviceptr,
            t: i32,
        ) -> i32;

        // wy_fast.py, `recompute_w_u_fwd_kernel`. `k`/`v`/`beta`: bf16. `w`
        // out `[T,H,K]` bf16. `u` out `[T,H,V]` bf16. `a`: `Ai` from
        // solve_tril, bf16 `[T,H,BT]`. `g`: f32 cumsum'd `[T,H]`.
        #[allow(clippy::too_many_arguments)]
        pub fn wu_58f79eea_0d1d2d3d4d5d6d7d8d(
            stream: CUstream,
            k: CUdeviceptr,
            v: CUdeviceptr,
            beta: CUdeviceptr,
            w: CUdeviceptr,
            u: CUdeviceptr,
            a: CUdeviceptr,
            g: CUdeviceptr,
            cu_seqlens: CUdeviceptr,
            chunk_indices: CUdeviceptr,
            t: i32,
        ) -> i32;

        // chunk_o.py, `chunk_fwd_kernel_o` -- kernel-3 equivalent. `q`/`k`:
        // bf16 `[T,Hg,K]`. `v`: kernel-2's own `v_new` output, bf16
        // `[T,H,V]`. `h`: kernel-2's own `h` output, bf16 `[NT,H,V,K]`. `g`:
        // f32 cumsum'd `[T,H]`. `o`: bf16 output `[T,H,V]`. `scale` (`K**-0.5`)
        // is baked in as a compile-time literal at this artifact's own
        // AOT-compile phase -- NOT a runtime argument (a real ABI bug in
        // Triton's stock C-stub generator for scalar-float runtime args made
        // that the only safe choice; see this project's own AOT-compile
        // report for `chunk_o` for the full finding).
        #[allow(clippy::too_many_arguments)]
        pub fn chunk_o_4401e9ec_0d1d2d3d4d5d6d7d(
            stream: CUstream,
            q: CUdeviceptr,
            k: CUdeviceptr,
            v: CUdeviceptr,
            h: CUdeviceptr,
            g: CUdeviceptr,
            o: CUdeviceptr,
            cu_seqlens: CUdeviceptr,
            chunk_indices: CUdeviceptr,
            t: i32,
        ) -> i32;
    }
}

/// A `nullptr`-valued `CUdeviceptr` -- every stage's `cu_seqlens`/
/// `chunk_indices` args, always: `IS_VARLEN=0` is baked into every one of
/// these compiled artifacts, matching kernel-2's own convention.
const NULL_PTR: CUdeviceptr = 0;

macro_rules! check {
    ($name:literal, $status:expr) => {
        anyhow::ensure!($status == 0, concat!("Triton-AOT ", $name, " launch returned CUresult {}"), $status);
    };
}

impl Kernels {
    /// The full FLA GDN prefill pipeline, real AOT-compiled Triton machine
    /// code end to end, one sequence a call. See this module's own doc
    /// comment for the stage order, the layout-conversion step, and this
    /// entry point's fixed scope.
    ///
    /// `qkv`/`g`/`beta`/`out`/`state`/`offsets`/`seqs` follow
    /// [`Kernels::gdn_delta_rule`]'s own convention exactly -- `g` here is
    /// the RAW, pre-cumsum per-token gate (this pipeline's own cumsum stage
    /// produces the cumulative one internally), and `q`/`k` inside `qkv`
    /// must already be L2-normalized and q-scaled (`gdn_qk_l2norm`), same as
    /// every other GDN entry point in this crate.
    ///
    /// `scratch` is every intermediate FLA-layout buffer this pipeline needs,
    /// caller-owned -- see [`GdnFlaScratch`]'s own doc comment. It must have
    /// been allocated with `max_tokens >= seqs.total_tokens`; a smaller
    /// scratch is a bug at the call site, not something this function resizes
    /// for you (same contract as `gdn_chunk_split3_delta_rule`'s own
    /// `w_buf`/`u_buf`/`delta_buf`/`s_before_buf`).
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_full_triton_pipeline(
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
        scratch: &mut GdnFlaScratch,
    ) -> Result<()> {
        anyhow::ensure!(seqs.n_seqs == 1, "gdn_full_triton_pipeline: single sequence only (B=1 compiled artifacts)");
        anyhow::ensure!(
            !v_tiled,
            "gdn_full_triton_pipeline: v_tiled=true is not supported by the compiled FLA stages"
        );
        anyhow::ensure!(
            heads == H && key_heads == HG && dk == K && dv == V,
            "gdn_full_triton_pipeline is compiled for H={H}, Hg={HG}, K={K}, V={V}; got heads={heads}, \
             key_heads={key_heads}, dk={dk}, dv={dv}"
        );
        let (stride, q_off, k_off, v_off) = offsets;
        let t = seqs.total_tokens;
        anyhow::ensure!(t > 0, "gdn_full_triton_pipeline: empty sequence");
        anyhow::ensure!(
            t <= scratch.max_tokens(),
            "gdn_full_triton_pipeline: t={t} exceeds scratch's max_tokens={}",
            scratch.max_tokens()
        );
        let nt = t.div_ceil(BT);
        let stream = self.dev.stream();

        debug_assert!(qkv.len() >= t * stride);
        debug_assert!(g.len() >= t * heads && beta.len() >= t * heads);
        debug_assert!(out.len() >= t * heads * dv);
        debug_assert!(state.len() >= heads * dk * dv);

        // ---- scratch, FLA's own real layout, caller-owned -- see
        // `GdnFlaScratch`'s own doc comment ----
        let q_bf = &mut scratch.q_bf;
        let k_bf = &mut scratch.k_bf;
        let v_bf = &mut scratch.v_bf;
        let beta_bf = &mut scratch.beta_bf;
        let g_raw_bf = &mut scratch.g_raw_bf;
        let g_cs = &mut scratch.g_cs;
        let a_mat = &mut scratch.a_mat;
        let ai_mat = &mut scratch.ai_mat;
        let w_bf = &mut scratch.w_bf;
        let u_bf = &mut scratch.u_bf;
        let h_bf = &mut scratch.h_bf;
        let v_new_bf = &mut scratch.v_new_bf;
        let h0_f32 = &mut scratch.h0_f32;
        let ht_f32 = &mut scratch.ht_f32;
        let o_bf = &mut scratch.o_bf;

        // ---- pack: infero's own layout -> FLA's own real layout ----
        self.dev.profile().time("gdn_fla_pack", stream, || {
            // q: infero's own `gdn_qk_l2norm` already scaled it by
            // `1/sqrt(dk)`; undo that here since `chunk_o` bakes in its own
            // `K**-0.5` (see `gdn.cu`'s own comment on this kernel for why
            // double-scaling q was this pipeline's first real bug). k: no
            // scale of its own either way.
            self.launch_extract_qk(qkv, q_bf, stride, q_off, HG * K, t * HG * K, (dk as f32).sqrt())?;
            self.launch_extract_qk(qkv, k_bf, stride, k_off, HG * K, t * HG * K, 1.0)?;
            self.launch_extract_v(qkv, v_bf, stride, v_off, H * V, t * H * V)?;
            self.launch_cast_f32_to_bf16(beta, beta_bf, t * H)?;
            self.launch_cast_f32_to_bf16(g, g_raw_bf, t * H)?;
            self.launch_transpose(state, h0_f32, heads, dk, dv)?;
            Ok(())
        })?;

        // ---- the 5 real AOT-compiled FLA stages, in real pipeline order ----
        //
        // Every pointer below is extracted with `.0` and no bound guard name:
        // `device_ptr[_mut]` returns `(CUdeviceptr, guard)`, and the guard is
        // a live borrow of the owning `CudaSlice` -- several buffers here
        // (`a_mat`, `ai_mat`, `w_bf`, `u_bf`, `h_bf`, `v_new_bf`, `ht_f32`)
        // are written by one stage and read by a later one, a genuine
        // mutable-then-shared reuse the borrow checker will not allow if the
        // first guard is still alive. The underlying device pointer (a plain
        // `u64`) stays valid regardless -- it is the same allocation for the
        // buffer's whole lifetime -- so dropping each guard immediately
        // (`.0` drops the unnamed `.1` temporary at the end of its own `let`
        // statement) is correct, not just a way to silence the compiler.
        let g_raw_p = g_raw_bf.device_ptr(stream).0;
        let g_cs_p = g_cs.device_ptr_mut(stream).0;
        self.dev.profile().time("gdn_fla_cumsum", stream, || {
            let status = unsafe {
                ffi::cumsum_88a339d3_0d1d2d3d(stream.cu_stream(), g_raw_p, g_cs_p, NULL_PTR, NULL_PTR, t as i32)
            };
            check!("cumsum", status);
            Ok(())
        })?;

        let k_p = k_bf.device_ptr(stream).0;
        let beta_bf_p = beta_bf.device_ptr(stream).0;
        let a_p = a_mat.device_ptr_mut(stream).0;
        self.dev.profile().time("gdn_fla_kkt", stream, || {
            let status = unsafe {
                ffi::kkt_2e3078a5_0d1d2d3d4d5d(
                    stream.cu_stream(),
                    k_p,
                    beta_bf_p,
                    g_cs_p,
                    a_p,
                    NULL_PTR,
                    NULL_PTR,
                    t as i32,
                )
            };
            check!("kkt", status);
            Ok(())
        })?;

        let a_ro = a_mat.device_ptr(stream).0;
        let ai_p = ai_mat.device_ptr_mut(stream).0;
        self.dev.profile().time("gdn_fla_solve_tril", stream, || {
            let status =
                unsafe { ffi::solve_tril64_ce06b0ca_0d1d2d3d(stream.cu_stream(), a_ro, ai_p, NULL_PTR, NULL_PTR, t as i32) };
            check!("solve_tril64", status);
            Ok(())
        })?;

        let v_p = v_bf.device_ptr(stream).0;
        let w_p = w_bf.device_ptr_mut(stream).0;
        let u_p = u_bf.device_ptr_mut(stream).0;
        let ai_ro = ai_mat.device_ptr(stream).0;
        self.dev.profile().time("gdn_fla_wy_fast", stream, || {
            let status = unsafe {
                ffi::wu_58f79eea_0d1d2d3d4d5d6d7d8d(
                    stream.cu_stream(),
                    k_p,
                    v_p,
                    beta_bf_p,
                    w_p,
                    u_p,
                    ai_ro,
                    g_cs_p,
                    NULL_PTR,
                    NULL_PTR,
                    t as i32,
                )
            };
            check!("wy_fast", status);
            Ok(())
        })?;

        // kernel-2 (already bound in `gdn_triton_aot.rs`): its own real "v"
        // input is wy_fast's `u` output, not the raw value -- matches that
        // module's own probe.
        let w_ro = w_bf.device_ptr(stream).0;
        let u_ro = u_bf.device_ptr(stream).0;
        let h0_ro = h0_f32.device_ptr(stream).0;
        let h_p = h_bf.device_ptr_mut(stream).0;
        let v_new_p = v_new_bf.device_ptr_mut(stream).0;
        let ht_p = ht_f32.device_ptr_mut(stream).0;
        self.dev.profile().time("gdn_fla_kernel2", stream, || unsafe {
            gdn_triton_aot::launch_chunk_gated_delta_rule_fwd_h(
                stream, k_p, u_ro, w_ro, v_new_p, g_cs_p, h_p, h0_ro, ht_p, t as i32,
            )
        })?;

        let q_p = q_bf.device_ptr(stream).0;
        let v_new_ro = v_new_bf.device_ptr(stream).0;
        let h_ro = h_bf.device_ptr(stream).0;
        let o_p = o_bf.device_ptr_mut(stream).0;
        self.dev.profile().time("gdn_fla_chunk_o", stream, || {
            let status = unsafe {
                ffi::chunk_o_4401e9ec_0d1d2d3d4d5d6d7d(
                    stream.cu_stream(),
                    q_p,
                    k_p,
                    v_new_ro,
                    h_ro,
                    g_cs_p,
                    o_p,
                    NULL_PTR,
                    NULL_PTR,
                    t as i32,
                )
            };
            check!("chunk_o", status);
            Ok(())
        })?;

        // ---- unpack: o (bf16, same [T,H,V] shape/order as `out`) and the
        // updated state (FLA's `S[v][k]` -> this engine's own `S[k][v]`) ----
        self.dev.profile().time("gdn_fla_unpack", stream, || {
            self.launch_cast_bf16_to_f32(o_bf, out, t * heads * dv)?;
            self.launch_transpose_into(ht_f32, state, heads, dv, dk)?;
            Ok(())
        })?;

        Ok(())
    }

    /// `gdn_fla_extract_qk_f32_to_bf16`: gathers one of q/k's `[T,Hg,K]` bf16
    /// tensor out of the packed `[T,stride]` f32 row at column offset `off`.
    fn launch_extract_qk(
        &self,
        qkv: &View<'_, f32>,
        dst: &mut cudarc::driver::CudaSlice<u16>,
        stride: usize,
        off: usize,
        hg_k: usize,
        n: usize,
        scale: f32,
    ) -> Result<()> {
        let f = self.dev.kernels().get("infero_gdn", gdn_src(), "gdn_fla_extract_qk_f32_to_bf16")?;
        let mut b = self.dev.stream().launch_builder(&f);
        let (st, of, hk, nn) = (stride as i32, off as i32, hg_k as i32, n as i64);
        b.arg(qkv).arg(dst).arg(&st).arg(&of).arg(&hk).arg(&nn).arg(&scale);
        unsafe { b.launch(Self::flat_cfg(n)) }?;
        Ok(())
    }

    /// `gdn_fla_extract_v_f32_to_bf16`: same gather for v's `[T,H,V]`.
    fn launch_extract_v(
        &self,
        qkv: &View<'_, f32>,
        dst: &mut cudarc::driver::CudaSlice<u16>,
        stride: usize,
        off: usize,
        h_v: usize,
        n: usize,
    ) -> Result<()> {
        let f = self.dev.kernels().get("infero_gdn", gdn_src(), "gdn_fla_extract_v_f32_to_bf16")?;
        let mut b = self.dev.stream().launch_builder(&f);
        let (st, of, hv, nn) = (stride as i32, off as i32, h_v as i32, n as i64);
        b.arg(qkv).arg(dst).arg(&st).arg(&of).arg(&hv).arg(&nn);
        unsafe { b.launch(Self::flat_cfg(n)) }?;
        Ok(())
    }

    /// `gdn_fla_cast_f32_to_bf16`: flat elementwise cast, no gather (`beta`,
    /// the raw gate).
    fn launch_cast_f32_to_bf16(
        &self,
        src: &View<'_, f32>,
        dst: &mut cudarc::driver::CudaSlice<u16>,
        n: usize,
    ) -> Result<()> {
        let f = self.dev.kernels().get("infero_gdn", gdn_src(), "gdn_fla_cast_f32_to_bf16")?;
        let mut b = self.dev.stream().launch_builder(&f);
        let nn = n as i64;
        b.arg(src).arg(dst).arg(&nn);
        unsafe { b.launch(Self::flat_cfg(n)) }?;
        Ok(())
    }

    /// `gdn_fla_cast_bf16_to_f32`: the inverse cast, for the final output.
    fn launch_cast_bf16_to_f32(
        &self,
        src: &cudarc::driver::CudaSlice<u16>,
        dst: &mut ViewMut<'_, f32>,
        n: usize,
    ) -> Result<()> {
        let f = self.dev.kernels().get("infero_gdn", gdn_src(), "gdn_fla_cast_bf16_to_f32")?;
        let mut b = self.dev.stream().launch_builder(&f);
        let nn = n as i64;
        b.arg(src).arg(dst).arg(&nn);
        unsafe { b.launch(Self::flat_cfg(n)) }?;
        Ok(())
    }

    /// `gdn_fla_transpose_hab_to_hba_f32`: infero's own persistent state
    /// (`ViewMut`, read-only here despite the mutable type -- the caller
    /// holds it as `&mut ViewMut` because it also writes the updated state
    /// back later via [`Self::launch_transpose_into`]) -> FLA's own `h0`
    /// layout (a fresh scratch buffer).
    fn launch_transpose(
        &self,
        src: &mut ViewMut<'_, f32>,
        dst: &mut cudarc::driver::CudaSlice<f32>,
        heads: usize,
        a_dim: usize,
        b_dim: usize,
    ) -> Result<()> {
        let f = self.dev.kernels().get("infero_gdn", gdn_src(), "gdn_fla_transpose_hab_to_hba_f32")?;
        let mut b = self.dev.stream().launch_builder(&f);
        let (h, a, bd) = (heads as i32, a_dim as i32, b_dim as i32);
        b.arg(src).arg(dst).arg(&h).arg(&a).arg(&bd);
        unsafe { b.launch(Self::flat_cfg(heads * a_dim * b_dim)) }?;
        Ok(())
    }

    /// Same kernel, the other direction: FLA's own final `ht` (a scratch
    /// buffer, read-only here) -> infero's own persistent state (`ViewMut`,
    /// written in place).
    fn launch_transpose_into(
        &self,
        src: &cudarc::driver::CudaSlice<f32>,
        dst: &mut ViewMut<'_, f32>,
        heads: usize,
        a_dim: usize,
        b_dim: usize,
    ) -> Result<()> {
        let f = self.dev.kernels().get("infero_gdn", gdn_src(), "gdn_fla_transpose_hab_to_hba_f32")?;
        let mut b = self.dev.stream().launch_builder(&f);
        let (h, a, bd) = (heads as i32, a_dim as i32, b_dim as i32);
        b.arg(src).arg(dst).arg(&h).arg(&a).arg(&bd);
        unsafe { b.launch(Self::flat_cfg(heads * a_dim * b_dim)) }?;
        Ok(())
    }

    /// 256 threads a block, enough blocks to cover `n` at that width but
    /// capped at 65535 -- every one of these glue kernels loops grid-stride
    /// internally (see `gdn.cu`), so a capped grid is still correct for any
    /// `n`, just fewer threads doing more iterations each.
    fn flat_cfg(n: usize) -> LaunchConfig {
        const BLOCK: u32 = 256;
        let blocks = (n as u64).div_ceil(BLOCK as u64).clamp(1, 65535) as u32;
        LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 }
    }
}
