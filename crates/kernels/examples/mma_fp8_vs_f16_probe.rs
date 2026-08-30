//! How much does `mma.m16n8k32.e4m3` really buy over `mma.m16n8k16.f16` on
//! this card, at the exact instruction count `attn_prefill_mma_ws4_f32`'s own
//! QK^T loop issues for this checkpoint's d_head=256 (six 8-key sub-tiles
//! times `d_head/16 = 16` `mma_f16` calls, against the same six times
//! `d_head/32 = 8` `mma_e4m3` calls)?
//!
//! This is step one of evaluating an e4m3 QK^T/PV attention rewrite -- a
//! register-resident, memory-traffic-free micro-benchmark of the tensor-core
//! instruction itself, deliberately built before touching the real kernel
//! (which would also need a new e4m3 KV-cache format end to end, a
//! production-quality-risk change this probe says nothing about). See the
//! doc comment on `mma_f16_throughput_probe`/`mma_e4m3_throughput_probe` in
//! `ops.cu`.
//!
//!     cargo run --release -p infero-kernels --example mma_fp8_vs_f16_probe

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::Kernels;

// `attn_prefill_mma_ws4_f32`'s own per-K-tile QK^T instruction counts at this
// checkpoint's d_head=256: six 8-key sub-tiles, `d_head/16` f16 steps or
// `d_head/32` e4m3 steps each.
const D_HEAD: usize = 256;
const SUB_TILES: usize = 6;
const F16_KSTEPS: usize = D_HEAD / 16;
const E4M3_KSTEPS: usize = D_HEAD / 32;
// How many K-tiles' worth of QK^T to run back to back, so the loop is long
// enough to time cleanly.
const OUTER_ITERS: usize = 20_000;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let dev = Device::new(0)?;
    let sm_count = dev.sm_count();
    let k = Kernels::new(dev.clone());
    let stream = k.device().stream().clone();

    // One warp a block, enough blocks to actually fill the SMs the way
    // `ws4`'s real consumer-warp population does, not a single-warp
    // best-case latency number.
    let blocks = sm_count as usize * 8;
    let mut out = stream.alloc_zeros::<f32>(blocks)?;

    let f16_reps = SUB_TILES * F16_KSTEPS * OUTER_ITERS;
    let e4m3_reps = SUB_TILES * E4M3_KSTEPS * OUTER_ITERS;

    println!(
        "device: {} (sm_{}, {sm_count} SMs), {blocks} blocks x 1 warp",
        dev.name(),
        dev.arch()
    );
    println!(
        "f16  : {SUB_TILES} sub-tiles x {F16_KSTEPS} ksteps x {OUTER_ITERS} outer = {f16_reps} mma.m16n8k16 calls/warp"
    );
    println!(
        "e4m3 : {SUB_TILES} sub-tiles x {E4M3_KSTEPS} ksteps x {OUTER_ITERS} outer = {e4m3_reps} mma.m16n8k32 calls/warp"
    );

    // Warm up (first-launch JIT/context cost), then time.
    k.mma_f16_throughput_probe(&mut out.as_view_mut(), blocks, 64)?;
    k.mma_e4m3_throughput_probe(&mut out.as_view_mut(), blocks, 64)?;
    k.device().synchronize()?;

    const REPEATS: usize = 5;
    let mut f16_best = f64::INFINITY;
    for _ in 0..REPEATS {
        let t = std::time::Instant::now();
        k.mma_f16_throughput_probe(&mut out.as_view_mut(), blocks, f16_reps)?;
        k.device().synchronize()?;
        f16_best = f16_best.min(t.elapsed().as_secs_f64());
    }
    let mut e4m3_best = f64::INFINITY;
    for _ in 0..REPEATS {
        let t = std::time::Instant::now();
        k.mma_e4m3_throughput_probe(&mut out.as_view_mut(), blocks, e4m3_reps)?;
        k.device().synchronize()?;
        e4m3_best = e4m3_best.min(t.elapsed().as_secs_f64());
    }

    let f16_per_call = f16_best * 1e9 / (f16_reps * blocks) as f64;
    let e4m3_per_call = e4m3_best * 1e9 / (e4m3_reps * blocks) as f64;

    println!("\nf16  QK^T-equivalent: {:.3} ms total, {:.3} ns/mma-call", f16_best * 1e3, f16_per_call);
    println!("e4m3 QK^T-equivalent: {:.3} ms total, {:.3} ns/mma-call", e4m3_best * 1e3, e4m3_per_call);
    println!(
        "\nsame-QK^T-coverage speedup (e4m3 vs f16, matched instruction ratio): {:.3}x",
        f16_best / e4m3_best
    );
    Ok(())
}
