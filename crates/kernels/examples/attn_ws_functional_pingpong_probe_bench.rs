//! Does real cross-warp concurrency -- two DIFFERENT physical warps, one
//! doing QK^T/PV-shaped tensor-core work continuously, the other doing a
//! softmax-shaped dependent scalar chain one tile behind -- actually
//! overlap on sm_120a, where `attn_full_tile_pipelined_probe` already
//! showed single-warp instruction reordering does not (0.600x, because
//! `mma.sync` blocks its own issuing warp regardless of source order)?
//!
//! `ws4`'s own consumer warps are already independent of each other, but
//! all 7 hit the same per-tile barrier and run the same QK^T-then-softmax-
//! then-PV sequence at the same rate -- they likely reach the softmax
//! dependent chain in near lockstep with nothing else resident to fill the
//! gap. This probe checks whether deliberately staggering two roles by one
//! tile, so a tensor-core-only warp and a softmax-only warp are never idle
//! at the same time, produces measurable overlap -- correctness first
//! (checksum must match a sequential single-warp reference exactly, since
//! every quantity here is a deterministic function of iteration index),
//! then timing.
//!
//!     cargo run --release -p infero-kernels --example attn_ws_functional_pingpong_probe_bench

use anyhow::{bail, Result};
use infero_cuda::Device;
use infero_kernels::Kernels;

const REPEATS: usize = 200;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let dev = Device::new(0)?;
    let k = Kernels::new(dev.clone());
    let stream = k.device().stream().clone();

    let mut out_seq = stream.alloc_zeros::<f32>(2)?;
    let mut out_pp = stream.alloc_zeros::<f32>(2)?;

    k.attn_ws_functional_pingpong_sequential_ref(&mut out_seq.as_view_mut())?;
    k.attn_ws_functional_pingpong_probe(&mut out_pp.as_view_mut())?;
    k.device().synchronize()?;

    let seq_host = stream.clone_dtoh(&out_seq)?;
    let pp_host = stream.clone_dtoh(&out_pp)?;
    println!("sequential checksum: mma_side={:.6} softmax_side={:.6}", seq_host[0], seq_host[1]);
    println!("pingpong   checksum: mma_side={:.6} softmax_side={:.6}", pp_host[0], pp_host[1]);

    let bits_match = |a: f32, b: f32| a.to_bits() == b.to_bits() || (a - b).abs() <= 1e-3;
    let mma_diff = (seq_host[0] - pp_host[0]).abs();
    let softmax_diff = (seq_host[1] - pp_host[1]).abs();
    if !bits_match(seq_host[0], pp_host[0]) || !bits_match(seq_host[1], pp_host[1]) {
        bail!(
            "checksum mismatch -- the cross-warp handoff protocol has a bug: \
             mma_side diff={mma_diff:.6}, softmax_side diff={softmax_diff:.6}"
        );
    }
    println!("checksums match -- the handoff protocol is correct.\n");

    let mut seq_best = f64::INFINITY;
    for _ in 0..REPEATS {
        let t = std::time::Instant::now();
        k.attn_ws_functional_pingpong_sequential_ref(&mut out_seq.as_view_mut())?;
        k.device().synchronize()?;
        seq_best = seq_best.min(t.elapsed().as_secs_f64());
    }
    let mut pp_best = f64::INFINITY;
    for _ in 0..REPEATS {
        let t = std::time::Instant::now();
        k.attn_ws_functional_pingpong_probe(&mut out_pp.as_view_mut())?;
        k.device().synchronize()?;
        pp_best = pp_best.min(t.elapsed().as_secs_f64());
    }

    println!("sequential (1 warp, QK^T -> softmax -> PV in order): {:.3} us", seq_best * 1e6);
    println!("pingpong   (2 warps, MMA role || softmax role, 1 tile offset): {:.3} us", pp_best * 1e6);
    println!("\ncross-warp overlap speedup: {:.3}x", seq_best / pp_best);
    Ok(())
}
