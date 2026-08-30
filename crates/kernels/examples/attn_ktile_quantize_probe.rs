//! What would `ws4`'s producer newly pay to convert a resident K tile from
//! `__half` to e4m3 (per-key scale) before an e4m3 QK^T could use it? The
//! full-tile probe (`attn_full_tile_probe`) measured the QK^T+softmax+PV
//! per-tile saving once K is *already* e4m3 -- 4.99 ns/tile at this
//! checkpoint's real d_head=256/WK=48 shape (8.29 f16 vs 3.30 e4m3). This
//! measures the other side of the ledger: if this quantization step costs
//! more than that per tile, it eats the whole saving before a real kernel
//! ever gets to keep any of it. Two designs: v1 (one warp, five
//! `__shfl_xor_sync` steps a key) and v2 (one thread a key, a plain
//! sequential loop, no cross-lane communication at all).
//!
//!     cargo run --release -p infero-kernels --example attn_ktile_quantize_probe

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::Kernels;

const OUTER_ITERS: usize = 200_000;
// From attn_full_tile_probe's last real measurement on this card.
const FULL_TILE_SAVING_NS: f64 = 8.291 - 3.298;

fn bench(
    label: &str,
    blocks: usize,
    mut run: impl FnMut(usize) -> Result<()>,
) -> Result<f64> {
    run(64)?;
    const REPEATS: usize = 5;
    let mut best = f64::INFINITY;
    for _ in 0..REPEATS {
        let t = std::time::Instant::now();
        run(OUTER_ITERS)?;
        best = best.min(t.elapsed().as_secs_f64());
    }
    let small_iters = OUTER_ITERS / 10;
    let t = std::time::Instant::now();
    run(small_iters)?;
    let small = t.elapsed().as_secs_f64();
    let per_tile_ns = best * 1e9 / (OUTER_ITERS * blocks) as f64;
    println!("\n{label}: {:.3} ms total, {per_tile_ns:.4} ns/tile-unit", best * 1e3);
    println!("  linearity check (iters/10 vs iters, expect ~10x): {:.3}x", best / small);
    Ok(per_tile_ns)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let dev = Device::new(0)?;
    let sm_count = dev.sm_count();
    let k = Kernels::new(dev.clone());
    let stream = k.device().stream().clone();

    let blocks = sm_count as usize * 8;
    let mut out = stream.alloc_zeros::<f32>(blocks)?;

    println!("device: {} (sm_{}, {sm_count} SMs), {blocks} blocks", dev.name(), dev.arch());

    let v1_per_tile = bench("v1 (1 warp, shuffle reduction)", blocks, |iters| {
        k.attn_ktile_e4m3_quantize_probe(&mut out.as_view_mut(), blocks, iters)?;
        k.device().synchronize()?;
        Ok(())
    })?;
    let v2_per_tile = bench("v2 (1 thread/key, sequential, no shuffles)", blocks, |iters| {
        k.attn_ktile_e4m3_quantize_v2_probe(&mut out.as_view_mut(), blocks, iters)?;
        k.device().synchronize()?;
        Ok(())
    })?;

    println!(
        "\nfull-tile QK^T saving available (from attn_full_tile_probe): {FULL_TILE_SAVING_NS:.3} ns/tile"
    );
    for (label, per_tile_ns) in [("v1", v1_per_tile), ("v2", v2_per_tile)] {
        if per_tile_ns < FULL_TILE_SAVING_NS {
            println!(
                "{label}: quantization cost ({per_tile_ns:.3} ns/tile) is LESS than the saving -- net positive by {:.3} ns/tile ({:.1}% kept)",
                FULL_TILE_SAVING_NS - per_tile_ns,
                100.0 * (FULL_TILE_SAVING_NS - per_tile_ns) / FULL_TILE_SAVING_NS
            );
        } else {
            println!(
                "{label}: quantization cost ({per_tile_ns:.3} ns/tile) EXCEEDS the saving -- net negative by {:.3} ns/tile",
                per_tile_ns - FULL_TILE_SAVING_NS
            );
        }
    }
    Ok(())
}
