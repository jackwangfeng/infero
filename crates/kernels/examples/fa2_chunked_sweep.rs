//! Real head-to-head timing of `FlashAttn2Ffi::prefill` vs
//! `attn_prefill_decoupled6_f16acc` in the REAL chunked-continuation shape
//! (run_tokens fixed at CHUNK, kv_len growing chunk-over-chunk) that a long
//! prompt actually produces once it exceeds the model's own `batch_tokens`
//! -- as opposed to `fa2_prefill_bench.rs`'s single-shot sweep
//! (run_tokens == kv_len), which only matches how a request *shorter than*
//! `batch_tokens` gets processed. Real question: does FP8 production's own
//! validated "flash_attn2 wins" finding (measured at `batch_tokens=8192`,
//! same GQA/d_head=256 shape) still hold once kv_len grows past run_tokens
//! across several sequential chunks -- the shape a request *longer than*
//! `batch_tokens` actually takes, for either checkpoint.
//!
//!   INFERO_ATTN_MMA=1 cargo run --release --features cuda -p infero-kernels \
//!     --example fa2_chunked_sweep --features flash_attn2

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

fn run_sweep(k: &Kernels, dev: &infero_cuda::Device, chunk: usize, total: usize) -> Result<()> {
    let stream = dev.stream().clone();
    let (n_heads, n_kv_heads, d_head) = (24usize, 4usize, 256usize);
    let scale = 1.0 / (d_head as f32).sqrt();

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < total {
        let end = (start + chunk).min(total);
        chunks.push((start, end));
        start = end;
    }

    let kv_elems = n_kv_heads * total * d_head;
    let kh: Vec<f16> = pseudo_random(kv_elems, 0x82).into_iter().map(f16::from_f32).collect();
    let vh: Vec<f16> = pseudo_random(kv_elems, 0x93).into_iter().map(f16::from_f32).collect();
    let dk = stream.clone_htod(&kh)?;
    let dv = stream.clone_htod(&vh)?;

    let positions: Vec<i32> = (0..total as i32).collect();
    let table: Vec<i32> = (0..total as i32).collect();
    let dpos = stream.clone_htod(&positions)?;
    let dtable = stream.clone_htod(&table)?;

    let q_all = pseudo_random(total * n_heads * d_head, 0x71);
    let dq_all = stream.clone_htod(&q_all)?;

    let run = |use_fa2: bool| -> Result<f64> {
        let mut out = stream.alloc_zeros::<f32>(chunk * n_heads * d_head)?;
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
            let batch = BatchLayout {
                seq_of: &dseq.as_view(),
                positions: &dpos.as_view().slice(start..end),
                slot_table: &dtable.as_view(),
                table_stride: total,
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
                    kern: k,
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
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    };

    let fa2_ms = run(true)?;
    let infero_ms = run(false)?;
    println!(
        "chunk={chunk:>6} total={total:>6} ({} chunks): flash_attn2={fa2_ms:>9.3}ms  \
         decoupled6={infero_ms:>9.3}ms  ratio(decoupled6/fa2)={:.3}x  winner={}",
        chunks.len(),
        infero_ms / fa2_ms,
        if fa2_ms < infero_ms { "flash_attn2" } else { "decoupled6" }
    );
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::WARN).init();
    let dev = Device::new(0)?;
    let k = Kernels::new(dev.clone());

    for &chunk in &[2048usize, 8192] {
        run_sweep(&k, &dev, chunk, 27295)?;
    }
    Ok(())
}
