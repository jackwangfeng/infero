//! Real end-to-end timing: does the decoupled-role design (register
//! math validated by the regcheck probes, correctness proven against the
//! reference) actually deliver more occupancy/throughput than `ws4` at the
//! real chunked usage pattern -- 30552 tokens (this checkpoint's own
//! prompt length), 1024-token batches, 16 full-attention layers?
//!
//!   INFERO_ATTN_MMA=1 cargo run --release --features cuda -p infero-kernels \
//!     --example attn_decoupled_vs_ws4_bench

use anyhow::Result;
use half::f16;
use infero_gpu::Device;
use infero_kernels::{AttnDims, BatchLayout, Kernels};

const N_HEADS: usize = 24;
const N_KV_HEADS: usize = 4;
const D_HEAD: usize = 256;
const TOTAL_TOKENS: usize = 30552;
const BATCH_TOKENS: usize = 1024;
const N_LAYERS: usize = 16;

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
    let k = Kernels::new(Device::new(0)?);
    let stream = k.device().stream().clone();

    let n_slots = TOTAL_TOKENS + 128;
    let n_tokens = TOTAL_TOKENS;
    let q = pseudo_random(n_tokens * N_HEADS * D_HEAD, 0x71);
    let kv_elems = N_KV_HEADS * n_slots * D_HEAD;
    let kh: Vec<f16> = pseudo_random(kv_elems, 0x82).into_iter().map(f16::from_f32).collect();
    let vh: Vec<f16> = pseudo_random(kv_elems, 0x93).into_iter().map(f16::from_f32).collect();
    let seq_of = vec![0i32; n_tokens];
    let positions: Vec<i32> = (0..n_tokens as i32).collect();
    let table: Vec<i32> = (0..n_slots as i32).collect();
    let table_stride = n_slots;

    let dq = stream.clone_htod(&q)?;
    let dk = stream.clone_htod(&kh)?;
    let dv = stream.clone_htod(&vh)?;
    let dpos = stream.clone_htod(&positions)?;
    let dseq = stream.clone_htod(&seq_of)?;
    let dtable = stream.clone_htod(&table)?;
    let (vseq, vpos, vtable) = (dseq.as_view(), dpos.as_view(), dtable.as_view());
    let batch = BatchLayout { seq_of: &vseq, positions: &vpos, slot_table: &vtable, table_stride };

    let scale = 1.0 / (D_HEAD as f32).sqrt();
    let dims = AttnDims { n_heads: N_HEADS, n_kv_heads: N_KV_HEADS, d_head: D_HEAD, n_slots, n_tokens };

    let mut out_ws4 = stream.alloc_zeros::<f32>(n_tokens * N_HEADS * D_HEAD)?;
    let mut out_dc = stream.alloc_zeros::<f32>(n_tokens * N_HEADS * D_HEAD)?;
    let mut out_dc2 = stream.alloc_zeros::<f32>(n_tokens * N_HEADS * D_HEAD)?;
    let mut out_dc2f = stream.alloc_zeros::<f32>(n_tokens * N_HEADS * D_HEAD)?;
    let mut part = stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(N_HEADS, D_HEAD, BATCH_TOKENS))?;

    let run_ws4 = |out: &mut infero_gpu::Buf<f32>, part: &mut infero_gpu::Buf<f32>| -> Result<f64> {
        k.device().synchronize()?;
        let t0 = std::time::Instant::now();
        for _layer in 0..N_LAYERS {
            let mut base = 0usize;
            while base < TOTAL_TOKENS {
                let run_tokens = BATCH_TOKENS.min(TOTAL_TOKENS - base);
                let kv_len = base + run_tokens;
                k.attn_prefill_ws4(
                    &mut out.as_view_mut(), &dq.as_view(), &dk.as_view(), &dv.as_view(),
                    batch, dims, base, run_tokens, kv_len, scale, &mut part.as_view_mut(),
                )?;
                base += run_tokens;
            }
        }
        k.device().synchronize()?;
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    };

    let run_decoupled = |out: &mut infero_gpu::Buf<f32>| -> Result<f64> {
        k.device().synchronize()?;
        let t0 = std::time::Instant::now();
        for _layer in 0..N_LAYERS {
            let mut base = 0usize;
            while base < TOTAL_TOKENS {
                let run_tokens = BATCH_TOKENS.min(TOTAL_TOKENS - base);
                let kv_len = base + run_tokens;
                k.attn_prefill_decoupled(
                    &mut out.as_view_mut(), &dq.as_view(), &dk.as_view(), &dv.as_view(),
                    batch, dims, base, run_tokens, kv_len, scale,
                )?;
                base += run_tokens;
            }
        }
        k.device().synchronize()?;
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    };

    let run_decoupled2 = |out: &mut infero_gpu::Buf<f32>| -> Result<f64> {
        k.device().synchronize()?;
        let t0 = std::time::Instant::now();
        for _layer in 0..N_LAYERS {
            let mut base = 0usize;
            while base < TOTAL_TOKENS {
                let run_tokens = BATCH_TOKENS.min(TOTAL_TOKENS - base);
                let kv_len = base + run_tokens;
                k.attn_prefill_decoupled2(
                    &mut out.as_view_mut(), &dq.as_view(), &dk.as_view(), &dv.as_view(),
                    batch, dims, base, run_tokens, kv_len, scale,
                )?;
                base += run_tokens;
            }
        }
        k.device().synchronize()?;
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    };

    let run_decoupled2f = |out: &mut infero_gpu::Buf<f32>| -> Result<f64> {
        k.device().synchronize()?;
        let t0 = std::time::Instant::now();
        for _layer in 0..N_LAYERS {
            let mut base = 0usize;
            while base < TOTAL_TOKENS {
                let run_tokens = BATCH_TOKENS.min(TOTAL_TOKENS - base);
                let kv_len = base + run_tokens;
                k.attn_prefill_decoupled2_f16acc(
                    &mut out.as_view_mut(), &dq.as_view(), &dk.as_view(), &dv.as_view(),
                    batch, dims, base, run_tokens, kv_len, scale,
                )?;
                base += run_tokens;
            }
        }
        k.device().synchronize()?;
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    };

    // Warmup, then best of several timed reps each.
    run_ws4(&mut out_ws4, &mut part)?;
    run_decoupled(&mut out_dc)?;
    run_decoupled2(&mut out_dc2)?;
    run_decoupled2f(&mut out_dc2f)?;

    const REPEATS: usize = 3;
    let mut best_ws4 = f64::MAX;
    let mut best_dc = f64::MAX;
    let mut best_dc2 = f64::MAX;
    let mut best_dc2f = f64::MAX;
    for _ in 0..REPEATS {
        best_ws4 = best_ws4.min(run_ws4(&mut out_ws4, &mut part)?);
        best_dc = best_dc.min(run_decoupled(&mut out_dc)?);
        best_dc2 = best_dc2.min(run_decoupled2(&mut out_dc2)?);
        best_dc2f = best_dc2f.min(run_decoupled2f(&mut out_dc2f)?);
    }

    println!(
        "attn_prefill_ws4         best: {best_ws4:.2} ms total ({N_LAYERS} layers x {TOTAL_TOKENS} tokens)"
    );
    println!(
        "attn_prefill_decoupled   best: {best_dc:.2} ms total, {:.3}x  (T=1)",
        best_ws4 / best_dc
    );
    println!(
        "attn_prefill_decoupled2  best: {best_dc2:.2} ms total, {:.3}x  (T=2)",
        best_ws4 / best_dc2
    );
    println!(
        "attn_prefill_decoupled2f best: {best_dc2f:.2} ms total, {:.3}x  (T=2, fp16 PV accum, 3 blocks/SM)",
        best_ws4 / best_dc2f
    );
    Ok(())
}
