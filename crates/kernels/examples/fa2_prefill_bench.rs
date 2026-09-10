//! Real head-to-head timing of `FlashAttn2Ffi::prefill` against infero's own
//! `attn_prefill_decoupled6_f16acc`, swept across single-shot prefill lengths,
//! at the real Qwen3.8-27B family's GQA shape (24 q-heads, 4 kv-heads,
//! d_head=256 -- shared by both the FP8 production checkpoint and the NVFP4
//! one). Real production logs show FP8's own real launch selects
//! `flash_attn2` as its attention backend and an earlier real investigation
//! (`docs/superpowers/plans/2026-09-05-mixed-batch-attention-dispatch-split.md`)
//! measured it winning over `decoupled6` across a real 332-7562 row sweep --
//! this checks whether that holds at the much longer lengths the NVFP4
//! investigation's own 27295-token prefill test used, to find the real
//! crossover rather than assume one extreme generalizes.
//!
//! Deliberately bypasses the full model (no KV pool, no weights) -- only
//! Q/K/V buffers for the current sweep point -- so it fits on a
//! memory-constrained shared GPU.
//!
//!   INFERO_ATTN_MMA=1 cargo run --release --features cuda -p infero-kernels \
//!     --example fa2_prefill_bench --features flash_attn2

use anyhow::Result;
use half::f16;
use infero_cuda::Device;
use infero_kernels::attn_backend::{AttentionBackend, AttnCallCtx};
use infero_kernels::flash_attn2::FlashAttn2Ffi;
use infero_kernels::{AttnDims, BatchLayout, Kernels};

fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect()
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::WARN).init();

    let dev = Device::new(0)?;
    let k = Kernels::new(dev.clone());
    let stream = dev.stream().clone();

    let (n_heads, n_kv_heads, d_head) = (24usize, 4usize, 256usize);
    let scale = 1.0 / (d_head as f32).sqrt();

    // Single-shot prefill at each length: run_tokens == kv_len (a fresh
    // sequence's own first, whole-prompt pass) -- the real shape this
    // sweep's own real precedent (332-7562 rows) and the NVFP4 investigation's
    // own 27295-token test both instantiate as one contiguous run.
    let lengths = [512usize, 1024, 2048, 4096, 8192, 16384, 27295];

    for &len in &lengths {
        let kv_elems = n_kv_heads * len * d_head;
        let kh: Vec<f16> = pseudo_random(kv_elems, 0x82).into_iter().map(f16::from_f32).collect();
        let vh: Vec<f16> = pseudo_random(kv_elems, 0x93).into_iter().map(f16::from_f32).collect();
        let dk = stream.clone_htod(&kh)?;
        let dv = stream.clone_htod(&vh)?;

        let positions: Vec<i32> = (0..len as i32).collect();
        let table: Vec<i32> = (0..len as i32).collect();
        let dpos = stream.clone_htod(&positions)?;
        let dtable = stream.clone_htod(&table)?;
        let seq_of = vec![0i32; len];
        let dseq = stream.clone_htod(&seq_of)?;

        let q = pseudo_random(len * n_heads * d_head, 0x71);
        let dq = stream.clone_htod(&q)?;

        let dims = AttnDims { n_heads, n_kv_heads, d_head, n_slots: len, n_tokens: len };
        let batch = BatchLayout {
            seq_of: &dseq.as_view(),
            positions: &dpos.as_view(),
            slot_table: &dtable.as_view(),
            table_stride: len,
        };

        let mut out = stream.alloc_zeros::<f32>(len * n_heads * d_head)?;
        let mut part = stream.alloc_zeros::<f32>(1)?;
        let fa2 = FlashAttn2Ffi::default();

        let reps = if len <= 4096 { 10 } else { 3 };

        // warm up + time flash_attn2
        for _ in 0..2 {
            let mut ctx = AttnCallCtx {
                out: &mut out.as_view_mut(),
                q: &dq.as_view(),
                k_cache: &dk.as_view(),
                v_cache: &dv.as_view(),
                batch,
                dims,
                run_base: 0,
                run_tokens: len,
                kv_len: len,
                scale,
                partial: &mut part.as_view_mut(),
                stream: &stream,
                kern: &k,
            };
            fa2.prefill(&mut ctx)?;
        }
        dev.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            let mut ctx = AttnCallCtx {
                out: &mut out.as_view_mut(),
                q: &dq.as_view(),
                k_cache: &dk.as_view(),
                v_cache: &dv.as_view(),
                batch,
                dims,
                run_base: 0,
                run_tokens: len,
                kv_len: len,
                scale,
                partial: &mut part.as_view_mut(),
                stream: &stream,
                kern: &k,
            };
            fa2.prefill(&mut ctx)?;
        }
        dev.synchronize()?;
        let fa2_ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

        // warm up + time decoupled6
        for _ in 0..2 {
            k.attn_prefill_decoupled6_f16acc(
                &mut out.as_view_mut(),
                &dq.as_view(),
                &dk.as_view(),
                &dv.as_view(),
                batch,
                dims,
                0,
                len,
                len,
                scale,
            )?;
        }
        dev.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            k.attn_prefill_decoupled6_f16acc(
                &mut out.as_view_mut(),
                &dq.as_view(),
                &dk.as_view(),
                &dv.as_view(),
                batch,
                dims,
                0,
                len,
                len,
                scale,
            )?;
        }
        dev.synchronize()?;
        let infero_ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

        println!(
            "len={len:>6}: flash_attn2={fa2_ms:>9.4}ms  decoupled6={infero_ms:>9.4}ms  \
             ratio(decoupled6/fa2)={:.3}x  winner={}",
            infero_ms / fa2_ms,
            if fa2_ms < infero_ms { "flash_attn2" } else { "decoupled6" }
        );
    }
    Ok(())
}
