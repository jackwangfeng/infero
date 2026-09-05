//! Rust side of the torch-free FlashAttention-2 FFI backend. See
//! `crates/kernels/src/cu_vendor/flash_attn2_shim.cu` for the CUDA side and
//! `docs/superpowers/specs/2026-09-05-pluggable-attention-backend-design.md`
//! for the design this implements.

use crate::attn_backend::{AttentionBackend, AttnCallCtx, HardwareCaps};
use crate::{AttnDims, KvQuant};
use anyhow::{ensure, Result};
use cudarc::driver::{DevicePtr, DevicePtrMut};
use infero_gpu::Buf;
use std::sync::Mutex;

// Same FFI boundary shape `cutlass_fp8.rs` already uses for its own AOT
// kernel (`crates/kernels/src/cutlass_fp8.rs`): a `CUstream` from
// `cudarc::driver::sys`, obtained at the call site via `stream.cu_stream()`.
unsafe extern "C" {
    fn infero_flash_attn2_fwd_causal_f16(
        q: *const std::ffi::c_void,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        out: *mut std::ffi::c_void,
        lse_scratch: *mut std::ffi::c_void,
        seqlen_q: i32,
        seqlen_k: i32,
        n_heads: i32,
        n_kv_heads: i32,
        d_head: i32,
        kv_n_slots: i32,
        softmax_scale: f32,
        stream: cudarc::driver::sys::CUstream,
    ) -> i32;
}

/// A torch-free FFI shim around Dao-AILab/flash-attention's real CUDA
/// forward kernel, scoped to one case: causal, fp16, `d_head == 256`, a
/// single physically-contiguous sequence (see the shim's own doc comment for
/// the full list of what's out of scope this pass — varlen, paged KV,
/// dropout, rotary, alibi, softcap, split-KV).
///
/// Priority 0 — numerically *lower* than
/// [`super::attn_backend::InferoHandRolled`]'s 100 (that field's own doc
/// comment explains the swap: this used to be the reverse, before real
/// benchmarking found this backend actually wins the real production shape).
/// `min_by_key` picks the lowest priority among backends whose `supports()`
/// returns true, so this only actually wins a call when its own `supports()`
/// below also returns true for it — which is gated to the specific
/// `n_tokens` regime this backend has been measured to win, not "whenever
/// eligible" in some broader sense.
#[derive(Default)]
pub struct FlashAttn2Ffi {
    /// Reused across calls rather than freshly `alloc_zeros`'d every call —
    /// a real 30552-token prefill makes 64 of these (4 chunks × 16 layers),
    /// and a fresh ~100MB allocation each time was real, measured overhead
    /// (see `Scratch::ensure_capacity`'s own doc comment). `Mutex` rather
    /// than `RefCell`: `AttentionBackend: Send + Sync` requires this type stay
    /// `Sync`, and `RefCell` isn't, even though the one caller
    /// (`Model::attention()`, one sequential per-layer forward pass, no
    /// concurrent access to a single `Model`) never actually contends this
    /// lock — an uncontended `Mutex` lock is not a real cost worth avoiding
    /// for that.
    scratch: Mutex<Scratch>,
}

#[derive(Default)]
struct Scratch {
    q16: Option<Buf<half::f16>>,
    out16: Option<Buf<half::f16>>,
    lse: Option<Buf<f32>>,
}

impl Scratch {
    /// Grows (never shrinks) each buffer to at least `n_qo`/`n_lse` elements,
    /// reusing the existing allocation when it's already big enough. A real
    /// prefill's chunks vary in size (the last chunk of a 30552-token run at
    /// `batch_tokens=8192` is only 5976 tokens), so this can't just allocate
    /// once at a fixed size — but in practice the first call already
    /// establishes the max (chunks before the last one are always the full
    /// `batch_tokens`), so this reallocates at most once per model load, not
    /// once per call.
    fn ensure_capacity(
        &mut self,
        stream: &std::sync::Arc<infero_gpu::Stream>,
        n_qo: usize,
        n_lse: usize,
    ) -> Result<()> {
        if self.q16.as_ref().is_none_or(|b| b.len() < n_qo) {
            self.q16 = Some(stream.alloc_zeros::<half::f16>(n_qo)?);
        }
        if self.out16.as_ref().is_none_or(|b| b.len() < n_qo) {
            self.out16 = Some(stream.alloc_zeros::<half::f16>(n_qo)?);
        }
        if self.lse.as_ref().is_none_or(|b| b.len() < n_lse) {
            self.lse = Some(stream.alloc_zeros::<f32>(n_lse)?);
        }
        Ok(())
    }
}

impl AttentionBackend for FlashAttn2Ffi {
    fn name(&self) -> &'static str {
        "flash_attn2"
    }

    fn priority(&self) -> u32 {
        0
    }

    /// FA2's own real floor (`_is_fa2_supported`, `vllm_flash_attn`): compute
    /// capability >= 8.0. `d_head == 256` is this shim's own, narrower scope
    /// (one fixed `Kernel_traits` instantiation, one KV layout it addresses).
    ///
    /// `dims.n_tokens >= FA2_ROW_THRESHOLD` is the part that decides whether
    /// this backend actually wins by default rather than merely being
    /// eligible: two real, controlled measurements this session found the
    /// direction between this backend and `InferoHandRolled`'s tuned kernels
    /// flips with query-row count — at 1024 rows, `ws4`/`decoupled6` beat FA2
    /// by ~10-15x (a clean CUDA-event comparison); at this checkpoint's real
    /// 8192-row production chunk (`CUTLASS_BATCH_TOKENS`), FA2 wins by
    /// ~2.7-3x (a real end-to-end `prefill_profile` comparison, 4798ms vs
    /// 5891ms). There is no third measured point between those two, so
    /// `FA2_ROW_THRESHOLD = 4096` is a judgment call — roughly the
    /// geometric midpoint, chosen to comfortably clear 1024 (where infero
    /// wins) while comfortably admitting 8192 (where FA2 wins) — not a
    /// measured crossover. `dims.n_tokens` here is `Model`'s real, configured
    /// `batch_tokens()` at load time (the only row count `prefill_run` ever
    /// actually uses for this checkpoint), not a per-call value — selection
    /// happens once, per the trait's own contract.
    fn supports(&self, caps: &HardwareCaps, dims: &AttnDims, kv_quant: KvQuant) -> bool {
        const FA2_ROW_THRESHOLD: usize = 4096;
        caps.at_least(8, 0)
            && !kv_quant.is_quantized()
            && dims.d_head == 256
            && dims.n_tokens >= FA2_ROW_THRESHOLD
    }

    fn prefill(&self, ctx: &mut AttnCallCtx<'_>) -> Result<()> {
        ensure!(ctx.run_base == 0, "flash_attn2 backend: only run_base=0 supported this pass");
        ensure!(
            ctx.dims.d_head == 256,
            "flash_attn2 backend: only d_head=256 supported this pass (got {})",
            ctx.dims.d_head
        );

        // Resolve the physical KV-cache base slot for this run and verify
        // it is contiguous. infero's pool is a general per-token
        // (sequence, logical position) -> physical slot table (see
        // `crates/model/src/cache.rs`'s own doc comment: "one pool ...
        // keeps a table mapping its logical positions onto physical
        // slots"), which FA2's own kernel has no concept of at all -- it
        // expects a flat `[n_kv_heads, n_slots, d_head]` region starting at
        // logical position 0. This pass's scope (a single freshly-prefilled
        // sequence, `Model::attention()`'s own `prefill_run` precondition)
        // means the physical slots for positions `0..kv_len` SHOULD already
        // be contiguous in practice, but that is this call's own
        // assumption to verify, not something to assume silently -- a
        // violation fails loudly here rather than producing silently wrong
        // attention output.
        let sid = ctx.stream.clone_dtoh(&ctx.batch.seq_of.slice(0..1))?[0];
        let table_offset = sid as usize * ctx.batch.table_stride;
        let slots = ctx.stream.clone_dtoh(&ctx.batch.slot_table.slice(table_offset..table_offset + ctx.kv_len))?;
        let base_slot = slots[0];
        for (i, &s) in slots.iter().enumerate() {
            ensure!(
                s == base_slot + i as i32,
                "flash_attn2 backend: KV slots for this run are not physically contiguous \
                 (slot[{i}]={s}, expected {}) -- this backend only supports a single, freshly \
                 prefilled sequence in this pass",
                base_slot + i as i32
            );
        }

        let d_head = ctx.dims.d_head;
        // `ctx.q` is infero's real activation dtype, f32 -- every hand-rolled
        // prefill kernel in `ops.cu` takes Q as f32 and converts per-element
        // internally while reading it. This shim's kernel has no such
        // conversion (its `Element` type is a fixed `cutlass::half_t`,
        // matching K/V, which really are f16 in the KV pool) -- it needs a
        // real f16 buffer, not a raw-byte reinterpretation of f32 data
        // (which is what silently happened here before this fix: the same
        // memory, read 2 bytes at a time instead of 4, decodes as
        // essentially-random fp16 bit patterns, including frequent NaNs --
        // this was the real cause of this backend's wrong-output bug, not
        // any launch/params-fill logic).
        let n_q_elems = ctx.run_tokens * ctx.dims.n_heads * d_head;
        let n_out_elems = n_q_elems;
        let n_lse_elems = ctx.dims.n_heads * ctx.run_tokens;

        // Reused across calls (see `Scratch`'s own doc comment) rather than a
        // fresh ~100MB `alloc_zeros` every one of a real prefill's 64
        // layer*chunk calls. The lock is held for the rest of this function —
        // every device pointer below is derived from a buffer inside it.
        let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        scratch.ensure_capacity(ctx.stream, n_q_elems.max(n_out_elems), n_lse_elems)?;
        // Destructure once, up front, into three independent `&mut` field
        // borrows -- calling `scratch.q16.as_mut()` and then
        // `scratch.out16.as_mut()` separately re-borrows the WHOLE
        // `MutexGuard<Scratch>` each time (its `DerefMut` gives one `&mut
        // Scratch`, not per-field access), which conflicts once any of the
        // three needs to stay borrowed past the point the next one starts
        // (exactly the case here: all three pointers must stay valid through
        // one kernel launch below). This pattern splits the borrow disjointly
        // in one step, so the compiler can see all three fields never alias.
        let Scratch { q16, out16, lse } = &mut *scratch;
        let q16 = q16.as_mut().expect("ensure_capacity just populated this");
        let out16 = out16.as_mut().expect("ensure_capacity just populated this");
        let lse = lse.as_mut().expect("ensure_capacity just populated this");

        ctx.kern.to_f16(&mut q16.slice_mut(..n_q_elems), ctx.q, n_q_elems)?;
        let q16_view = q16.slice(..n_q_elems);
        let (q_ptr, _g1) = q16_view.device_ptr(ctx.stream);
        // The kernel's `Element` type is `cutlass::half_t` on the OUTPUT side
        // too (`convert_type<Element>(acc_o)` right before the final write in
        // `flash_fwd_kernel.h`) -- it writes f16, not f32. `ctx.out` is
        // infero's real f32 activation buffer, same class of dtype mismatch
        // as Q above, just on the write side: pointing the kernel at
        // `ctx.out`'s raw f32 memory directly (as this backend did before
        // the correctness fix landed) makes it write 2-byte fp16 elements
        // into a buffer whose real element stride is 4 bytes, so every
        // read-back element is built from two half-overlapping fp16 writes
        // -- explains the otherwise-mysterious ~2^-13-scale,
        // deterministic-but-wrong values this test was seeing even after
        // the acc_o register value was confirmed correct via
        // `print_tensor`. Write to a real f16 buffer, convert back with
        // `Kernels::from_f16` (a device-side kernel, same shape as the Q
        // conversion above) -- the first version of this fix went through
        // the host (`clone_dtoh`, a CPU-side `f32::from` loop,
        // `memcpy_htod_sync`) for every one of a real prefill's 64
        // layer*chunk calls, which measured as ~14.4s of a real 18.6s
        // 30552-token run -- the actual dominant cost of this backend, far
        // larger than the FFI call or the kernel itself. Fixed here.
        let mut out16_view = out16.slice_mut(..n_out_elems);
        let (out_ptr, _g2) = out16_view.device_ptr_mut(ctx.stream);
        let (k_base_ptr, _g3) = ctx.k_cache.device_ptr(ctx.stream);
        let (v_base_ptr, _g4) = ctx.v_cache.device_ptr(ctx.stream);
        // `k_cache`/`v_cache` cover the WHOLE per-layer pool
        // (`[n_kv_heads, n_slots, d_head]`); offset to this run's first
        // physical slot within kv_head 0 -- the shim's own `k_head_stride`
        // (using `dims.n_slots`, the pool's real per-head stride, not
        // `kv_len`) reaches every other head correctly from there.
        let elem_bytes = std::mem::size_of::<half::f16>() as u64;
        let k_ptr = k_base_ptr + (base_slot as u64) * (d_head as u64) * elem_bytes;
        let v_ptr = v_base_ptr + (base_slot as u64) * (d_head as u64) * elem_bytes;

        // FA2 always writes a log-sum-exp per (head, query row) even though
        // this call never reads it back -- point it at real, reused scratch
        // rather than null (the kernel does not check for that).
        let mut lse_view = lse.slice_mut(..n_lse_elems);
        let (lse_ptr, _g5) = lse_view.device_ptr_mut(ctx.stream);

        let rc = unsafe {
            infero_flash_attn2_fwd_causal_f16(
                q_ptr as *const std::ffi::c_void,
                k_ptr as *const std::ffi::c_void,
                v_ptr as *const std::ffi::c_void,
                out_ptr as *mut std::ffi::c_void,
                lse_ptr as *mut std::ffi::c_void,
                ctx.run_tokens as i32,
                ctx.kv_len as i32,
                ctx.dims.n_heads as i32,
                ctx.dims.n_kv_heads as i32,
                d_head as i32,
                ctx.dims.n_slots as i32,
                ctx.scale,
                ctx.stream.cu_stream(),
            )
        };
        // `_g2` (and the rest) hold `SyncOnDrop`-style guards whose `Drop`
        // impl is presumably what actually waits for the launch above to be
        // safe to reuse the buffer -- drop it explicitly here (rather than
        // relying on NLL, which doesn't end the borrow until end of scope
        // for a `Drop` type) so `out16` is free to read back below.
        drop(_g2);
        ensure!(rc == 0, "flash_attn2 fwd failed, code {rc}");
        ctx.kern.from_f16(ctx.out, &out16.slice(..n_out_elems), n_out_elems)?;
        Ok(())
    }
}
