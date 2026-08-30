//! Does e4m3's real per-instruction throughput win (measured standalone at
//! 1.958x in `mma_fp8_vs_f16_probe`) survive being embedded in `ws4`'s real
//! per-tile dependency chain -- QK^T -> online-softmax bookkeeping (two
//! shuffle reductions, an `expf` loop, a correction, an `o[]` rescale) -> PV?
//! This session's own `ncu` numbers on the real, deployed `attn_prefill_ws4`
//! found that scalar chain, not tensor-core issue rate, to be its dominant
//! stall -- so a bare-MMA-loop win is necessary but not sufficient for a
//! rewrite to pay off end to end. Reproduces `ws4`'s exact consumer-side
//! per-tile shape at this checkpoint's real d_head=256, WK=48, one resident
//! synthetic K/V tile reused every iteration (no producer, no cp.async --
//! that pipeline overlap is orthogonal to what's measured here). PV stays
//! `mma_f16` in both variants, isolating the comparison to QK^T.
//!
//!     cargo run --release -p infero-kernels --example attn_full_tile_probe

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::Kernels;

const OUTER_ITERS: usize = 200_000;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let dev = Device::new(0)?;
    let sm_count = dev.sm_count();
    let k = Kernels::new(dev.clone());
    let stream = k.device().stream().clone();

    // One warp a block, `ws4`'s own consumer-warp population order of
    // magnitude (7 consumer warps a block, many blocks) rather than a
    // single-warp best case.
    let blocks = sm_count as usize * 8;
    let mut out = stream.alloc_zeros::<f32>(blocks)?;
    let scale = 1.0f32 / (256f32).sqrt();

    let name = dev.name();
    let arch = dev.arch();
    println!(
        "device: {name} (sm_{arch}, {sm_count} SMs), {blocks} blocks x 1 warp, {OUTER_ITERS} outer iters"
    );

    k.attn_full_tile_f16_probe(&mut out.as_view_mut(), blocks, 64, scale)?;
    k.attn_full_tile_e4m3_probe(&mut out.as_view_mut(), blocks, 64, scale)?;
    k.device().synchronize()?;

    const REPEATS: usize = 5;
    let mut f16_best = f64::INFINITY;
    for _ in 0..REPEATS {
        let t = std::time::Instant::now();
        k.attn_full_tile_f16_probe(&mut out.as_view_mut(), blocks, OUTER_ITERS, scale)?;
        k.device().synchronize()?;
        f16_best = f16_best.min(t.elapsed().as_secs_f64());
    }
    let mut e4m3_best = f64::INFINITY;
    for _ in 0..REPEATS {
        let t = std::time::Instant::now();
        k.attn_full_tile_e4m3_probe(&mut out.as_view_mut(), blocks, OUTER_ITERS, scale)?;
        k.device().synchronize()?;
        e4m3_best = e4m3_best.min(t.elapsed().as_secs_f64());
    }

    let f16_per_tile = f16_best * 1e9 / (OUTER_ITERS * blocks) as f64;
    let e4m3_per_tile = e4m3_best * 1e9 / (OUTER_ITERS * blocks) as f64;

    println!("\nf16  full tile (QK^T+softmax+PV): {:.3} ms total, {:.3} ns/tile", f16_best * 1e3, f16_per_tile);
    println!("e4m3-QK^T full tile (QK^T+softmax+PV): {:.3} ms total, {:.3} ns/tile", e4m3_best * 1e3, e4m3_per_tile);
    println!(
        "\nfull-tile speedup (e4m3 QK^T + f16 PV, vs all-f16): {:.3}x  (bare-MMA-only speedup was 1.958x)",
        f16_best / e4m3_best
    );

    // Sanity check: does wall time scale linearly with outer_iters? If the
    // compiler hoisted loop-invariant work (sk/sv/qa[1..] never change
    // inside the loop, only qa[0].x[0] does via the feedback trick) instead
    // of genuinely re-running the tile's dependency chain every iteration,
    // a smaller iters count would take a disproportionately large share of
    // the big-iters time, and this ratio would come out well under 10x.
    let small_iters = OUTER_ITERS / 10;
    let t = std::time::Instant::now();
    k.attn_full_tile_f16_probe(&mut out.as_view_mut(), blocks, small_iters, scale)?;
    k.device().synchronize()?;
    let f16_small = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    k.attn_full_tile_e4m3_probe(&mut out.as_view_mut(), blocks, small_iters, scale)?;
    k.device().synchronize()?;
    let e4m3_small = t.elapsed().as_secs_f64();
    println!(
        "\nlinearity check (iters/10 vs iters, expect ~10x): f16 {:.3}x, e4m3 {:.3}x",
        f16_best / f16_small,
        e4m3_best / e4m3_small
    );
    Ok(())
}
