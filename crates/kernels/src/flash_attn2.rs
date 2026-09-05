//! Rust side of the torch-free FlashAttention-2 FFI backend. See
//! `crates/kernels/src/cu_vendor/flash_attn2_shim.cu` for the CUDA side and
//! `docs/superpowers/specs/2026-09-05-pluggable-attention-backend-design.md`
//! for the design this implements.

use crate::attn_backend::{AttentionBackend, AttnCallCtx, HardwareCaps};
use crate::{AttnDims, KvQuant};
use anyhow::{ensure, Result};
use cudarc::driver::{DevicePtr, DevicePtrMut};

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
/// Priority 100 — always loses to [`super::attn_backend::InferoHandRolled`]'s
/// priority 0 when both support a shape, since this backend exists to cover
/// hardware infero has no tuned kernel for yet, not to compete on hardware
/// that already has one. Known to be slower than infero's own tuned kernels
/// on sm_120a (this vendor kernel runs here only via generic sm_80 PTX
/// forward compatibility, with zero Blackwell-specific tuning) — see the
/// design doc's non-goals.
pub struct FlashAttn2Ffi;

impl AttentionBackend for FlashAttn2Ffi {
    fn name(&self) -> &'static str {
        "flash_attn2"
    }

    fn priority(&self) -> u32 {
        100
    }

    fn supports(&self, caps: &HardwareCaps, dims: &AttnDims, kv_quant: KvQuant) -> bool {
        // FA2's own real floor (`_is_fa2_supported`, `vllm_flash_attn`):
        // compute capability >= 8.0. Everything else here is this specific
        // shim's own, narrower scope (one fixed Kernel_traits instantiation,
        // one KV layout it can address).
        caps.at_least(8, 0) && !kv_quant.is_quantized() && dims.d_head == 256
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
        let mut q16 = ctx.stream.alloc_zeros::<half::f16>(n_q_elems)?;
        ctx.kern.to_f16(&mut q16.as_view_mut(), ctx.q, n_q_elems)?;
        let q16_view = q16.as_view();
        let (q_ptr, _g1) = q16_view.device_ptr(ctx.stream);
        // The kernel's `Element` type is `cutlass::half_t` on the OUTPUT side
        // too (`convert_type<Element>(acc_o)` right before the final write in
        // `flash_fwd_kernel.h`) -- it writes f16, not f32. `ctx.out` is
        // infero's real f32 activation buffer, same class of dtype mismatch
        // as Q above, just on the write side: pointing the kernel at
        // `ctx.out`'s raw f32 memory directly (as this backend did before
        // this fix) makes it write 2-byte fp16 elements into a buffer whose
        // real element stride is 4 bytes, so every read-back element is
        // built from two half-overlapping fp16 writes -- explains the
        // otherwise-mysterious ~2^-13-scale, deterministic-but-wrong values
        // this test was seeing even after the acc_o register value was
        // confirmed correct via `print_tensor`. Write to a real f16 buffer
        // and convert on the host afterward -- correctness-only for this
        // pass (see the design doc's non-goals: this backend does not need
        // to be fast), not a device-side kernel.
        let n_out_elems = ctx.run_tokens * ctx.dims.n_heads * d_head;
        let mut out16 = ctx.stream.alloc_zeros::<half::f16>(n_out_elems)?;
        let mut out16_view = out16.as_view_mut();
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
        // this call never reads it back -- allocate real scratch rather
        // than pass null (the kernel does not check for that).
        let mut lse = ctx.stream.alloc_zeros::<f32>(ctx.dims.n_heads * ctx.run_tokens)?;
        let (lse_ptr, _g5) = lse.device_ptr_mut(ctx.stream);

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
        let out16_host = ctx.stream.clone_dtoh(&out16)?;
        let out32_host: Vec<f32> = out16_host.iter().map(|v| f32::from(*v)).collect();
        let (ctx_out_ptr, _g6) = ctx.out.device_ptr_mut(ctx.stream);
        unsafe {
            cudarc::driver::result::memcpy_htod_sync(ctx_out_ptr, &out32_host)
                .map_err(|e| anyhow::anyhow!("flash_attn2: writing converted output failed: {e}"))?;
        }
        Ok(())
    }
}
