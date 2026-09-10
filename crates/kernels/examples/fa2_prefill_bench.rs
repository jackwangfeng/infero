//! Real head-to-head timing of `FlashAttn2Ffi::prefill` against infero's own
//! `attn_prefill_decoupled6_f16acc`, at the exact 14-chunk causal-prefill
//! shape a real 27295-token request produces on the NVFP4 investigation's
//! target checkpoint (24 q-heads, 4 kv-heads, d_head=256). Deliberately
//! bypasses the full model (no KV pool, no weights) -- only Q/K/V buffers
//! sized for this one benchmark -- so it fits on a memory-constrained shared
//! GPU where loading the real checkpoint alongside FA2's own KV-pool
//! requirements does not.
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
    const TOTAL: usize = 27295;
    const CHUNK: usize = 2048;

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < TOTAL {
        let end = (start + CHUNK).min(TOTAL);
        chunks.push((start, end));
        start = end;
    }
    println!("chunks: {chunks:?}");

    // One big K/V cache sized for the whole sequence, identity slot table
    // (contiguous, single fresh sequence -- FA2's own real precondition).
    let kv_elems = n_kv_heads * TOTAL * d_head;
    let kh: Vec<f16> = pseudo_random(kv_elems, 0x82).into_iter().map(f16::from_f32).collect();
    let vh: Vec<f16> = pseudo_random(kv_elems, 0x93).into_iter().map(f16::from_f32).collect();
    let dk = stream.clone_htod(&kh)?;
    let dv = stream.clone_htod(&vh)?;

    let positions: Vec<i32> = (0..TOTAL as i32).collect();
    let table: Vec<i32> = (0..TOTAL as i32).collect();
    let dpos = stream.clone_htod(&positions)?;
    let dtable = stream.clone_htod(&table)?;

    // Q for the whole sequence, contiguous, so each chunk just slices it.
    let q_all = pseudo_random(TOTAL * n_heads * d_head, 0x71);
    let dq_all = stream.clone_htod(&q_all)?;

    let scale = 1.0 / (d_head as f32).sqrt();

    let run = |backend_name: &str, use_fa2: bool| -> Result<f64> {
        let mut out = stream.alloc_zeros::<f32>(CHUNK * n_heads * d_head)?;
        let mut part = stream.alloc_zeros::<f32>(1)?;
        let fa2 = FlashAttn2Ffi::default();

        dev.synchronize()?;
        let t0 = std::time::Instant::now();
        for &(start, end) in &chunks {
            let run_tokens = end - start;
            let kv_len = end;
            let dims = AttnDims { n_heads, n_kv_heads, d_head, n_slots: kv_len, n_tokens: run_tokens };

            let seq_of = vec![0i32; run_tokens];
            let dseq = stream.clone_htod(&seq_of)?;
            let vseq = dseq.as_view();
            let vpos_full = dpos.as_view();
            let vtable_full = dtable.as_view();
            let batch = BatchLayout {
                seq_of: &vseq,
                positions: &vpos_full.slice(start..end),
                slot_table: &vtable_full,
                table_stride: TOTAL,
            };

            let q_lo = start * n_heads * d_head;
            let q_hi = end * n_heads * d_head;
            let out_len = run_tokens * n_heads * d_head;

            if use_fa2 {
                let mut ctx = AttnCallCtx {
                    out: &mut out.slice_mut(..out_len),
                    q: &dq_all.as_view().slice(q_lo..q_hi),
                    k_cache: &dk.as_view(),
                    v_cache: &dv.as_view(),
                    batch,
                    dims,
                    run_base: 0,
                    run_tokens,
                    kv_len,
                    scale,
                    partial: &mut part.as_view_mut(),
                    stream: &stream,
                    kern: &k,
                };
                fa2.prefill(&mut ctx)?;
            } else {
                k.attn_prefill_decoupled6_f16acc(
                    &mut out.slice_mut(..out_len),
                    &dq_all.as_view().slice(q_lo..q_hi),
                    &dk.as_view(),
                    &dv.as_view(),
                    batch,
                    dims,
                    0,
                    run_tokens,
                    kv_len,
                    scale,
                )?;
            }
        }
        dev.synchronize()?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("{backend_name}: {ms:.2} ms");
        Ok(ms)
    };

    let fa2_ms = run("flash_attn2", true)?;
    let infero_ms = run("attn_prefill_decoupled6_f16acc", false)?;

    println!("SUMMARY flash_attn2_ms={fa2_ms:.2} infero_ms={infero_ms:.2} speedup={:.2}x", infero_ms / fa2_ms);
    Ok(())
}
