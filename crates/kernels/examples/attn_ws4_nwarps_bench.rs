//! Does shrinking ws4's block (fewer consumer warps, same per-warp work,
//! same per-thread register count) so more than one block's worth of
//! registers fit an SM -- FlashAttention-2's own H100 lever for hdim256,
//! per its `kernel_traits.h`/`flash_fwd_launch_template.h` -- actually pay
//! off here, at the real, smaller memory-retraffic cost (`7/NWARPS`x more
//! blocks, each still streaming the full causal K/V range) than the
//! decoupled-role kernel's role-fragmentation tax (7x, for an unrelated
//! reason)?
//!
//! Real 30552-token/16-layer/1024-batch shape, same as every other real
//! attention bench this session.
//!
//!   INFERO_ATTN_MMA=1 cargo run --release --features cuda -p infero-kernels \
//!     --example attn_ws4_nwarps_bench

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

    let mut out = stream.alloc_zeros::<f32>(n_tokens * N_HEADS * D_HEAD)?;
    let mut part = stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(N_HEADS, D_HEAD, BATCH_TOKENS))?;

    let run_nw = |nwarps: usize, out: &mut infero_gpu::Buf<f32>, part: &mut infero_gpu::Buf<f32>| -> Result<f64> {
        k.device().synchronize()?;
        let t0 = std::time::Instant::now();
        for _layer in 0..N_LAYERS {
            let mut base = 0usize;
            while base < TOTAL_TOKENS {
                let run_tokens = BATCH_TOKENS.min(TOTAL_TOKENS - base);
                let kv_len = base + run_tokens;
                k.attn_prefill_ws4_nw(
                    &mut out.as_view_mut(), &dq.as_view(), &dk.as_view(), &dv.as_view(),
                    batch, dims, base, run_tokens, kv_len, scale, &mut part.as_view_mut(), nwarps,
                )?;
                base += run_tokens;
            }
        }
        k.device().synchronize()?;
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    };

    // Correctness cross-check first: every NWARPS must produce the exact
    // same attention output for the same inputs (this only changes how many
    // rows a block covers, not the math), on a small single-chunk shape
    // where the whole output fits comfortably for a host round-trip.
    {
        let small_tokens = 2048usize;
        let mut out_small = stream.alloc_zeros::<f32>(small_tokens * N_HEADS * D_HEAD)?;
        let mut part_small = stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(N_HEADS, D_HEAD, small_tokens))?;
        let mut reference: Option<Vec<f32>> = None;
        for &nwarps in &[7usize, 4, 3, 2, 1] {
            k.attn_prefill_ws4_nw(
                &mut out_small.as_view_mut(), &dq.as_view(), &dk.as_view(), &dv.as_view(),
                batch, dims, 0, small_tokens, small_tokens, scale, &mut part_small.as_view_mut(), nwarps,
            )?;
            k.device().synchronize()?;
            let got = stream.clone_dtoh(&out_small)?;
            match &reference {
                None => reference = Some(got),
                Some(want) => {
                    let max_diff = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                    println!("NWARPS={nwarps} vs NWARPS=7 reference: max abs diff {max_diff:.6}");
                    assert!(max_diff < 1e-3, "NWARPS={nwarps} disagrees with the NWARPS=7 reference");
                }
            }
        }
        println!("correctness check passed for all NWARPS values.\n");
    }

    const REPEATS: usize = 3;
    for &nwarps in &[7usize, 4, 3, 2, 1] {
        run_nw(nwarps, &mut out, &mut part)?; // warmup
        let mut best = f64::MAX;
        for _ in 0..REPEATS {
            best = best.min(run_nw(nwarps, &mut out, &mut part)?);
        }
        println!(
            "NWARPS={nwarps} best: {best:.2} ms total ({N_LAYERS} layers x {TOTAL_TOKENS} tokens)"
        );
    }
    Ok(())
}
