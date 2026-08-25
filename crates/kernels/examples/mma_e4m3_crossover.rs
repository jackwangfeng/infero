//! Where does `mma_e4m3_block` (native e4m3xe4m3, plus a `quantize_act_e4m3`
//! launch) stop being cheaper per call than `mma_f8_block` (the older W8A16
//! path, weight widened to f16)? `matmul_pre` took the new kernel at every
//! `n_tokens >= 2`, tuned only against batch=16/32 decode throughput — this
//! sweeps small token counts too, which is what a k=2 speculative verify
//! step (`n_tokens = k + 1 = 3`) actually runs at, to find the real
//! crossover instead of guessing one.
//!
//!     cargo run --release -p tuili-kernels --example mma_e4m3_crossover
//!
//! Shape defaults to the fused FFN gate+up on the 27B: k = d_model = 5120,
//! n = 2 * d_ff = 34816, both multiples of `mma_e4m3_block`'s `K_TILE = 256`.

use anyhow::{Context, Result};
use tuili_cuda::backend::{CAPTURE_RELAXED, GraphFlags};
use tuili_cuda::Device;
use tuili_kernels::Kernels;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

const EAGER_ITERS: usize = 500;
const GRAPH_REPS: usize = 64;
const GRAPH_REPLAYS: usize = 30;

fn pseudo_random_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 56) as u8
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
            ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect()
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let k_dim = env_usize("MMA_K", 5120);
    let n_dim = env_usize("MMA_N", 34816);
    let dev = Device::new(0)?;
    println!("device: {} (sm_{}, {} SMs)", dev.name(), dev.arch(), dev.sm_count());
    println!("shape: k={k_dim} n={n_dim}");
    let k = Kernels::new(dev.clone());
    let stream = dev.stream().clone();

    let w_bytes = tuili_kernels::fp8::fp8_bytes(k_dim, n_dim);
    let weights = pseudo_random_bytes(w_bytes, 1);
    let d_w = stream.clone_htod(&weights)?;

    let max_tokens = env_usize("MMA_MAX_TOKENS", 32);
    let x = pseudo_random_f32(max_tokens * k_dim, 2);
    let d_x = stream.clone_htod(&x)?;
    let mut d_out = stream.alloc_zeros::<f32>(max_tokens * n_dim)?;

    let scale_cols = k_dim.div_ceil(tuili_kernels::fp8::ACT_QUANT_GROUP);
    let mut d_xq = stream.alloc_zeros::<u8>(max_tokens * k_dim)?;
    let mut d_xs = stream.alloc_zeros::<f32>(max_tokens * scale_cols)?;

    println!(
        "{:>6} {:>10} {:>10} {:>8}   {:>10} {:>10} {:>8}",
        "tokens", "eag f8", "eag e4m3", "e4m3/f8", "gr f8", "gr e4m3", "e4m3/f8"
    );
    for &n_tokens in &[1usize, 2, 3, 4, 6, 8, 12, 16, 24, 32] {
        if n_tokens > max_tokens {
            continue;
        }
        let xv = d_x.slice(..n_tokens * k_dim);
        let mut ov = d_out.slice_mut(..n_tokens * n_dim);

        for _ in 0..5 {
            let ran = k.mma_f8_block(&mut ov, &d_w.as_view(), &xv, k_dim, n_dim, n_tokens, false)?;
            anyhow::ensure!(ran, "mma_f8_block declined this shape at {n_tokens} tokens");
        }
        dev.synchronize()?;
        let t = std::time::Instant::now();
        for _ in 0..EAGER_ITERS {
            k.mma_f8_block(&mut ov, &d_w.as_view(), &xv, k_dim, n_dim, n_tokens, false)?;
        }
        dev.synchronize()?;
        let f8_us = t.elapsed().as_secs_f64() * 1e6 / EAGER_ITERS as f64;

        for _ in 0..5 {
            k.quantize_act_e4m3(
                &mut d_xq.slice_mut(..n_tokens * k_dim),
                &mut d_xs.slice_mut(..n_tokens * scale_cols),
                &xv,
                k_dim,
                n_tokens,
            )?;
            let ran = k.mma_e4m3_block(
                &mut ov,
                &d_w.as_view(),
                &d_xq.slice(..n_tokens * k_dim),
                &d_xs.slice(..n_tokens * scale_cols),
                k_dim,
                n_dim,
                n_tokens,
                false,
            )?;
            anyhow::ensure!(ran, "mma_e4m3_block declined this shape at {n_tokens} tokens");
        }
        dev.synchronize()?;
        let t = std::time::Instant::now();
        for _ in 0..EAGER_ITERS {
            k.quantize_act_e4m3(
                &mut d_xq.slice_mut(..n_tokens * k_dim),
                &mut d_xs.slice_mut(..n_tokens * scale_cols),
                &xv,
                k_dim,
                n_tokens,
            )
            .context("quantize_act_e4m3")?;
            k.mma_e4m3_block(
                &mut ov,
                &d_w.as_view(),
                &d_xq.slice(..n_tokens * k_dim),
                &d_xs.slice(..n_tokens * scale_cols),
                k_dim,
                n_dim,
                n_tokens,
                false,
            )
            .context("mma_e4m3_block")?;
        }
        dev.synchronize()?;
        let e4m3_us = t.elapsed().as_secs_f64() * 1e6 / EAGER_ITERS as f64;

        // The live server never calls these eagerly — decode steps run inside
        // a captured CUDA graph, replayed once a round. Graph replay drops
        // CPU-side per-launch dispatch to near nothing, which is exactly the
        // cost the eager numbers above spend an extra `quantize_act_e4m3`
        // launch on; a kernel that wins eager by hiding two dispatches behind
        // async queuing can still lose once dispatch itself is free and only
        // GPU execution time remains. `mma_f8_graph_bench.rs` asked the same
        // question for `mma_f8_block` alone; this is the two-kernel version.
        stream.begin_capture(CAPTURE_RELAXED)?;
        let cap_res = (|| -> Result<()> {
            for _ in 0..GRAPH_REPS {
                k.mma_f8_block(&mut ov, &d_w.as_view(), &xv, k_dim, n_dim, n_tokens, false)?;
            }
            Ok(())
        })();
        let graph = stream.end_capture(GraphFlags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
        cap_res?;
        let f8_graph = graph?.context("f8 capture produced no graph")?;
        f8_graph.upload()?;
        let t = std::time::Instant::now();
        for _ in 0..GRAPH_REPLAYS {
            f8_graph.launch()?;
        }
        dev.synchronize()?;
        let f8_graph_us =
            t.elapsed().as_secs_f64() * 1e6 / (GRAPH_REPLAYS * GRAPH_REPS) as f64;

        stream.begin_capture(CAPTURE_RELAXED)?;
        let cap_res = (|| -> Result<()> {
            for _ in 0..GRAPH_REPS {
                k.quantize_act_e4m3(
                    &mut d_xq.slice_mut(..n_tokens * k_dim),
                    &mut d_xs.slice_mut(..n_tokens * scale_cols),
                    &xv,
                    k_dim,
                    n_tokens,
                )?;
                k.mma_e4m3_block(
                    &mut ov,
                    &d_w.as_view(),
                    &d_xq.slice(..n_tokens * k_dim),
                    &d_xs.slice(..n_tokens * scale_cols),
                    k_dim,
                    n_dim,
                    n_tokens,
                    false,
                )?;
            }
            Ok(())
        })();
        let graph = stream.end_capture(GraphFlags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
        cap_res?;
        let e4m3_graph = graph?.context("e4m3 capture produced no graph")?;
        e4m3_graph.upload()?;
        let t = std::time::Instant::now();
        for _ in 0..GRAPH_REPLAYS {
            e4m3_graph.launch()?;
        }
        dev.synchronize()?;
        let e4m3_graph_us =
            t.elapsed().as_secs_f64() * 1e6 / (GRAPH_REPLAYS * GRAPH_REPS) as f64;

        println!(
            "{n_tokens:>6} {f8_us:>10.2} {e4m3_us:>10.2} {:>7.3}x | graph: {f8_graph_us:>10.2} {e4m3_graph_us:>10.2} {:>7.3}x",
            e4m3_us / f8_us,
            e4m3_graph_us / f8_graph_us,
        );
    }
    Ok(())
}
