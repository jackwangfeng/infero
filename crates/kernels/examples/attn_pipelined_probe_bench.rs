//! Does software-pipelining `ws4`'s per-tile loop -- issuing tile `i+1`'s
//! QK^T before tile `i`'s softmax+PV finishes, via a second independent
//! accumulator set -- actually let the warp scheduler overlap the SFU-bound
//! softmax scalar chain with tensor-core MMA work, on real sm_120a hardware
//! and the ordinary synchronous `mma.sync` this kernel family already uses?
//!
//! Context: `ws4`'s own `ncu` numbers show only 32% compute throughput,
//! dominated by a dependent scalar chain (online-softmax bookkeeping)
//! between K-tile iterations -- not memory latency, not tensor-core issue
//! rate. FlashAttention-3 closes an equivalent gap on Hopper via exactly
//! this kind of pipelining (either across two warpgroups -- "ping-pong" --
//! or, as here, within one warp/warpgroup, at the cost of a duplicated
//! accumulator). FA3 gets there through `wgmma`'s async completion; this
//! GPU's sm_120a has no `wgmma` (confirmed: `mma.sync.aligned` is the only
//! MMA path on this architecture), so the question is whether the *same*
//! overlap shows up from pure instruction reordering, with no new hardware
//! primitive, just software pipelining plus the warp scheduler's own
//! ordinary latency hiding across independent execution units.
//!
//!     cargo run --release -p infero-kernels --example attn_pipelined_probe_bench

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

    let blocks = sm_count as usize * 8;
    let mut out = stream.alloc_zeros::<f32>(blocks)?;
    let scale = 1.0f32 / (256f32).sqrt();

    let name = dev.name();
    let arch = dev.arch();
    println!(
        "device: {name} (sm_{arch}, {sm_count} SMs), {blocks} blocks x 1 warp, {OUTER_ITERS} outer iters"
    );

    k.attn_full_tile_f16_probe(&mut out.as_view_mut(), blocks, 64, scale)?;
    k.attn_full_tile_pipelined_probe(&mut out.as_view_mut(), blocks, 64, scale)?;
    k.device().synchronize()?;

    const REPEATS: usize = 5;
    let mut seq_best = f64::INFINITY;
    for _ in 0..REPEATS {
        let t = std::time::Instant::now();
        k.attn_full_tile_f16_probe(&mut out.as_view_mut(), blocks, OUTER_ITERS, scale)?;
        k.device().synchronize()?;
        seq_best = seq_best.min(t.elapsed().as_secs_f64());
    }
    let mut pipe_best = f64::INFINITY;
    for _ in 0..REPEATS {
        let t = std::time::Instant::now();
        k.attn_full_tile_pipelined_probe(&mut out.as_view_mut(), blocks, OUTER_ITERS, scale)?;
        k.device().synchronize()?;
        pipe_best = pipe_best.min(t.elapsed().as_secs_f64());
    }

    let seq_per_tile = seq_best * 1e9 / (OUTER_ITERS * blocks) as f64;
    let pipe_per_tile = pipe_best * 1e9 / (OUTER_ITERS * blocks) as f64;

    println!("\nsequential   (QK^T -> softmax+PV, in order): {:.3} ms total, {seq_per_tile:.3} ns/tile", seq_best * 1e3);
    println!("pipelined    (next QK^T issued before softmax+PV): {:.3} ms total, {pipe_per_tile:.3} ns/tile", pipe_best * 1e3);
    println!("\nsoftware-pipelining speedup: {:.3}x", seq_best / pipe_best);

    // Same linearity guard as the other full-tile probes: rules out the
    // compiler hoisting loop-invariant work instead of genuinely re-running
    // the dependency chain every iteration.
    let small_iters = OUTER_ITERS / 10;
    let t = std::time::Instant::now();
    k.attn_full_tile_f16_probe(&mut out.as_view_mut(), blocks, small_iters, scale)?;
    k.device().synchronize()?;
    let seq_small = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    k.attn_full_tile_pipelined_probe(&mut out.as_view_mut(), blocks, small_iters, scale)?;
    k.device().synchronize()?;
    let pipe_small = t.elapsed().as_secs_f64();
    println!(
        "\nlinearity check (iters/10 vs iters, expect ~10x): sequential {:.3}x, pipelined {:.3}x",
        seq_best / seq_small,
        pipe_best / pipe_small
    );
    Ok(())
}
