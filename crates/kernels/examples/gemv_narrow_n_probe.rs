//! Is `gemv`'s real cost at GDN's own `in_proj_a`/`in_proj_b` shape
//! (K=5120, N=48, F16 after the loader's bf16->f16 upconvert) bandwidth or
//! launch/reduction overhead? Real profile: 48 layers, 48 launches/step,
//! 42.8us each -- but 48*5120*2 bytes (F16) is under 0.5 MiB a layer, which
//! at any real GPU bandwidth should take a small fraction of a microsecond.
//! If per-call cost stays ~flat as N grows, it's overhead, not bandwidth.
//!
//! Measured: flat at ~24.6us from N=48 through N=512, only starting to
//! scale with N around N=1024 -- confirms bandwidth is not the constraint
//! at this shape. But `examples/launch_overhead.rs`'s own real floor on
//! this box is ~2.45us/launch with no sync, ~8.61us with one -- an order
//! of magnitude below the measured 24.6us, so this isn't pure CUDA launch
//! overhead either. Something inside this specific kernel/dispatch path
//! (the block-level reduction, the per-call `format!`-based kernel-name
//! lookup, or something `ncu` would show and this box's `ERR_NVGPUCTRPERM`
//! blocks seeing) costs the extra ~15-20us. Real, quantified, unexplained --
//! not chased further without a profiler that works here.
//!
//! Separately: batching this across GDN's own 48 layers (the fix this
//! finding would otherwise suggest) is not available at all -- each layer's
//! input is that layer's own hidden state, which does not exist until the
//! *previous* layer has finished (an ordinary transformer's residual
//! stream), so the 48 calls are genuinely sequential, not 48 independent
//! units of work a batched kernel could fuse the way `stacked3` fuses
//! Q/K/V (which really do share one input).
//!
//!   cargo run --release -p infero-kernels --features cuda --example gemv_narrow_n_probe

use std::time::Instant;

use anyhow::Result;
use infero_gpu::Device;
use infero_kernels::{Kernels, WeightType};

fn pseudo_random_f16(n: usize, seed: u64) -> Vec<half::f16> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            half::f16::from_f32((((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0) * 3.0)
        })
        .collect()
}

fn pseudo_random_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0) * 3.0
        })
        .collect()
}

fn bench(k: &Kernels, k_dim: usize, n_dim: usize, reps: usize) -> Result<()> {
    let stream = k.device().stream().clone();
    let w: Vec<half::f16> = pseudo_random_f16(n_dim * k_dim, 0xF16F);
    let w_bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    let d_w = stream.clone_htod(&w_bytes)?;
    let x: Vec<f32> = pseudo_random_f32(k_dim, 0xACE0);
    let d_x = stream.clone_htod(&x)?;
    let mut d_out = stream.alloc_zeros::<f32>(n_dim)?;

    for _ in 0..3 {
        k.gemv(&mut d_out.as_view_mut(), &d_w.as_view(), WeightType::F16, &d_x.as_view(), k_dim, n_dim, 1)?;
    }
    k.device().synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        k.gemv(&mut d_out.as_view_mut(), &d_w.as_view(), WeightType::F16, &d_x.as_view(), k_dim, n_dim, 1)?;
    }
    k.device().synchronize()?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
    let bytes = (n_dim * k_dim * 2) as f64; // F16 = 2 bytes/elem
    let gbps = bytes / (ms / 1000.0) / 1e9;
    println!("K={k_dim:6} N={n_dim:6}  {ms:9.5} ms  {gbps:8.2} GB/s  ({:.1} us/call)", ms * 1000.0);
    Ok(())
}

fn main() -> Result<()> {
    let k = Kernels::new(Device::new(0)?);
    println!("-- real GDN in_proj_a/b shape (K=5120, N=48) vs larger N, same K --");
    for n in [48, 96, 256, 512, 1024, 4096, 17408] {
        bench(&k, 5120, n, 500)?;
    }
    Ok(())
}
