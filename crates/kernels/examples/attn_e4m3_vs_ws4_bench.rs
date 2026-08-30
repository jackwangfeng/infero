//! The real question this session's whole e4m3-attention investigation has
//! been building toward: does the full pipeline (one-shot K quantize, once
//! a chunk, amortized over every later chunk's causal re-reads, feeding
//! `attn_prefill_e4m3k_f32`) beat `attn_prefill_ws4_f32` at the real chunked
//! usage pattern -- not a bare-MMA probe, not a full-tile probe with
//! synthetic data, a real correctness-tested kernel run the way the model
//! actually calls attention. Same shape and chunking loop as
//! `attn_prefill_pipe_bench.rs`: 30552 tokens (this checkpoint's own real
//! prompt length), growing `kv_len`, 16 layers (this checkpoint's real
//! full-attention layer count).
//!
//! The e4m3 path's quantize cost is charged once a chunk, on just that
//! chunk's *new* tokens -- not once a K-tile-read the way the earlier,
//! conclusively-negative on-the-fly probes did -- since the whole point of
//! this session's amortization argument was that a one-shot quantize,
//! cached and re-read by every later chunk's causal attention, is cheap
//! relative to the QK^T saving it unlocks. This benchmark is where that
//! argument either cashes out or doesn't.
//!
//!   INFERO_ATTN_MMA=1 cargo run --release --features cuda -p infero-kernels \
//!     --example attn_e4m3_vs_ws4_bench

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
    let k_f32 = pseudo_random(TOTAL_TOKENS * N_KV_HEADS * D_HEAD, 0x82);
    let kh: Vec<f16> = k_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let vh: Vec<f16> = pseudo_random(kv_elems, 0x93).into_iter().map(f16::from_f32).collect();
    let seq_of = vec![0i32; n_tokens];
    let positions: Vec<i32> = (0..n_tokens as i32).collect();
    let table: Vec<i32> = (0..n_slots as i32).collect();
    let table_stride = n_slots;

    let dq = stream.clone_htod(&q)?;
    let dk_f32 = stream.clone_htod(&k_f32)?;
    let dk = stream.clone_htod(&kh)?;
    let dv = stream.clone_htod(&vh)?;
    let dpos = stream.clone_htod(&positions)?;
    let dseq = stream.clone_htod(&seq_of)?;
    let dtable = stream.clone_htod(&table)?;
    let (vseq, vpos, vtable) = (dseq.as_view(), dpos.as_view(), dtable.as_view());
    let batch = BatchLayout { seq_of: &vseq, positions: &vpos, slot_table: &vtable, table_stride };

    let scale = 1.0 / (D_HEAD as f32).sqrt();
    let dims_ws4 = AttnDims { n_heads: N_HEADS, n_kv_heads: N_KV_HEADS, d_head: D_HEAD, n_slots, n_tokens };
    let dims_e4m3 = AttnDims { n_heads: N_HEADS, n_kv_heads: N_KV_HEADS, d_head: D_HEAD, n_slots: 0, n_tokens };

    let mut out = stream.alloc_zeros::<f32>(n_tokens * N_HEADS * D_HEAD)?;
    let mut part = stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(N_HEADS, D_HEAD, BATCH_TOKENS))?;
    // V for the e4m3 path shares `dv`'s own contiguous, position-major
    // layout (this benchmark's `v` was never staged into the paged,
    // slot-table-indexed shape `ws4`'s own `dv` above technically needs --
    // both use the same flat buffer here since `n_slots == n_tokens` for a
    // single fresh sequence, the only case this validation kernel supports).
    let mut out2 = stream.alloc_zeros::<f32>(n_tokens * N_HEADS * D_HEAD)?;
    let mut part2 = stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(N_HEADS, D_HEAD, BATCH_TOKENS))?;
    let mut kq = stream.alloc_zeros::<u8>(TOTAL_TOKENS * N_KV_HEADS * D_HEAD)?;
    let mut kscale = stream.alloc_zeros::<f32>(TOTAL_TOKENS * N_KV_HEADS)?;

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
                    batch, dims_ws4, base, run_tokens, kv_len, scale, &mut part.as_view_mut(),
                )?;
                base += run_tokens;
            }
        }
        k.device().synchronize()?;
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    };

    let run_e4m3 = |out: &mut infero_gpu::Buf<f32>, part: &mut infero_gpu::Buf<f32>, kq: &mut infero_gpu::Buf<u8>, kscale: &mut infero_gpu::Buf<f32>| -> Result<f64> {
        k.device().synchronize()?;
        let t0 = std::time::Instant::now();
        for _layer in 0..N_LAYERS {
            let mut base = 0usize;
            while base < TOTAL_TOKENS {
                let run_tokens = BATCH_TOKENS.min(TOTAL_TOKENS - base);
                let kv_len = base + run_tokens;
                // One-shot quantize of just this chunk's *new* tokens --
                // every layer re-quantizes its own K (a real forward pass
                // has 16 distinct full-attention layers' worth of K, not
                // one shared buffer), charged in full to this timing. No
                // sync between calls -- an earlier version synced here to
                // time the quantize call alone and that sync corrupted the
                // *total* too: it drained the previous iteration's still-
                // in-flight attn_prefill_e4m3k call and misattributed that
                // time to quantize, while also breaking the async queue's
                // own overlap. Removed; see the isolated
                // quantize_k_e4m3 measurement in this file's memory entry
                // for the real per-call cost (~4 us, not milliseconds).
                let k_off = base * N_KV_HEADS * D_HEAD;
                let k_end = kv_len * N_KV_HEADS * D_HEAD;
                let ks_off = base * N_KV_HEADS;
                let ks_end = kv_len * N_KV_HEADS;
                k.quantize_k_e4m3(
                    &mut kq.slice_mut(k_off..k_end),
                    &mut kscale.slice_mut(ks_off..ks_end),
                    &dk_f32.slice(k_off..k_end),
                    run_tokens,
                    N_KV_HEADS,
                    D_HEAD,
                )?;
                k.attn_prefill_e4m3k(
                    &mut out.as_view_mut(), &dq.as_view(),
                    &kq.slice(0..k_end), &kscale.slice(0..ks_end), &dv.as_view(),
                    dims_e4m3, base, run_tokens, kv_len, scale,
                    &mut part.as_view_mut(),
                )?;
                base += run_tokens;
            }
        }
        k.device().synchronize()?;
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    };

    // Warmup, then best of a few timed reps each.
    run_ws4(&mut out, &mut part)?;
    run_e4m3(&mut out2, &mut part2, &mut kq, &mut kscale)?;

    let mut best_ws4 = f64::MAX;
    let mut best_e4m3 = f64::MAX;
    for _ in 0..3 {
        best_ws4 = best_ws4.min(run_ws4(&mut out, &mut part)?);
        best_e4m3 = best_e4m3.min(run_e4m3(&mut out2, &mut part2, &mut kq, &mut kscale)?);
    }

    println!("attn_prefill_ws4        best: {best_ws4:.2} ms total ({N_LAYERS} layers x {TOTAL_TOKENS} tokens)");
    println!("attn_prefill_e4m3k+quant best: {best_e4m3:.2} ms total, {:.3}x", best_ws4 / best_e4m3);
    Ok(())
}
