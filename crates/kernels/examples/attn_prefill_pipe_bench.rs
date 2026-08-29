//! Head-to-head timing of `attn_prefill` vs `attn_prefill_pipe` at the real
//! chunked usage pattern: 30552 tokens (Qwen3.8-27B-FP8's own prompt length
//! from this session's benchmarking) processed in growing-`kv_len` chunks,
//! matching how `Model::forward_batch_device` actually calls this kernel --
//! not one call over the whole sequence, which would blow the `partial`
//! buffer's size past available VRAM (`attn_partial_floats` scales with
//! `run_tokens`).
//!
//!   INFERO_ATTN_MMA=1 cargo run --release --features cuda -p infero-kernels \
//!     --example attn_prefill_pipe_bench

use anyhow::Result;
use half::f16;
use infero_gpu::Device;
use infero_kernels::{AttnDims, BatchLayout, Kernels};

const N_HEADS: usize = 24;
const N_KV_HEADS: usize = 4;
const D_HEAD: usize = 256;
const TOTAL_TOKENS: usize = 30552;
const BATCH_TOKENS: usize = 1024;
const N_LAYERS: usize = 16; // this checkpoint's full-attention layer count

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

    let mut out = stream.alloc_zeros::<f32>(n_tokens * N_HEADS * D_HEAD)?;
    let mut part = stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(N_HEADS, D_HEAD, BATCH_TOKENS))?;

    let run_one = |variant: u32, out: &mut infero_gpu::Buf<f32>, part: &mut infero_gpu::Buf<f32>| -> Result<f64> {
        k.device().synchronize()?;
        let t0 = std::time::Instant::now();
        for _layer in 0..N_LAYERS {
            let mut base = 0usize;
            while base < TOTAL_TOKENS {
                let run_tokens = BATCH_TOKENS.min(TOTAL_TOKENS - base);
                let kv_len = base + run_tokens;
                match variant {
                    0 => k.attn_prefill(
                        &mut out.as_view_mut(), &dq.as_view(), &dk.as_view(), &dv.as_view(),
                        batch, dims, base, run_tokens, kv_len, scale, &mut part.as_view_mut(),
                    )?,
                    1 => k.attn_prefill_pipe(
                        &mut out.as_view_mut(), &dq.as_view(), &dk.as_view(), &dv.as_view(),
                        batch, dims, base, run_tokens, kv_len, scale, &mut part.as_view_mut(),
                    )?,
                    2 => k.attn_prefill_natv(
                        &mut out.as_view_mut(), &dq.as_view(), &dk.as_view(), &dv.as_view(),
                        batch, dims, base, run_tokens, kv_len, scale, &mut part.as_view_mut(),
                    )?,
                    _ => k.attn_prefill_pipev(
                        &mut out.as_view_mut(), &dq.as_view(), &dk.as_view(), &dv.as_view(),
                        batch, dims, base, run_tokens, kv_len, scale, &mut part.as_view_mut(),
                    )?,
                }
                base += run_tokens;
            }
        }
        k.device().synchronize()?;
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    };

    // Warmup all four, then take the best of a few timed reps each.
    for v in 0..4 {
        run_one(v, &mut out, &mut part)?;
    }

    let mut best = [f64::MAX; 4];
    for _ in 0..3 {
        for v in 0..4u32 {
            best[v as usize] = best[v as usize].min(run_one(v, &mut out, &mut part)?);
        }
    }
    println!("attn_prefill (mma)  best: {:.2} ms total ({N_LAYERS} layers x {TOTAL_TOKENS} tokens)", best[0]);
    println!("attn_prefill_pipe   best: {:.2} ms total, {:.3}x", best[1], best[0] / best[1]);
    println!("attn_prefill_natv   best: {:.2} ms total, {:.3}x", best[2], best[0] / best[2]);
    println!("attn_prefill_pipev  best: {:.2} ms total, {:.3}x", best[3], best[0] / best[3]);
    Ok(())
}
