//! `mma_f8_block` timed two ways: eager (one launch, one sync each) and
//! captured in a CUDA graph and replayed, the way the live server actually
//! runs a decode step. `ncu` needs `INFERO_NO_GRAPH=1` to profile at all —
//! graph capture does not survive its replay mechanism — so every stall
//! reason it names was measured in a mode the server never uses. This
//! answers the question that leaves open: does the eager-mode stall this
//! session's two fixes (epilogue coalescing, staging balance) removed
//! actually cost anything once the kernel runs inside a graph, or does the
//! graph already hide it?
//!
//!     cargo run --release -p infero-kernels --example mma_f8_graph_bench
//!
//! Shape is GatedDeltaNet's fused `in_proj_qkv`+`in_proj_z` on the 27B:
//! k = d_model = 5120, n = conv_channels + value_dim = 16384, the widest
//! fusion this session's earlier commits built and the one `ncu` profiled
//! throughout. Data is pseudo-random, not real FP8 codes or scales — this
//! measures timing, not numerics, which `tests/fp8_matvec.rs` already covers.

use anyhow::{Context, Result};
use infero_cuda::backend::{CAPTURE_RELAXED, GraphFlags};
use infero_cuda::Device;
use infero_kernels::Kernels;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

// Repetitions captured into one graph, and how many times that graph is
// replayed. The product is what the per-call average divides by.
const GRAPH_REPS: usize = 64;
const GRAPH_REPLAYS: usize = 30;
const EAGER_ITERS: usize = 200;

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
    // Defaults are GatedDeltaNet's fused `in_proj_qkv`+`in_proj_z` on the
    // 27B; MMA_F8_K/N/TOKENS override for other shapes in the same model --
    // e.g. 5120/5120/16 for a narrow `out_proj`-shaped matrix, or
    // 5120/34816/16 for the fused FFN gate+up.
    let k_dim = env_usize("MMA_F8_K", 5120);
    let n_dim = env_usize("MMA_F8_N", 16384);
    let n_tokens = env_usize("MMA_F8_TOKENS", 16);
    let dev = Device::new(0)?;
    println!("device: {} (sm_{}, {} SMs)", dev.name(), dev.arch(), dev.sm_count());
    println!("shape: k={k_dim} n={n_dim} tokens={n_tokens}");
    let k = Kernels::new(dev.clone());
    let stream = dev.stream().clone();

    let w_bytes = infero_kernels::fp8::fp8_bytes(k_dim, n_dim);
    let weights = pseudo_random_bytes(w_bytes, 1);
    let x = pseudo_random_f32(n_tokens * k_dim, 2);

    let d_w = stream.clone_htod(&weights)?;
    let d_x = stream.clone_htod(&x)?;
    let mut d_out = stream.alloc_zeros::<f32>(n_tokens * n_dim)?;

    // Warm-up: also confirms the shape is eligible (`ran` is `false` past the
    // kernel's own token/`k`-alignment limits, which would make every timing
    // below measure nothing).
    for _ in 0..5 {
        let ran = k.mma_f8_block(&mut d_out.as_view_mut(), &d_w.as_view(), &d_x.as_view(), k_dim, n_dim, n_tokens, false)?;
        anyhow::ensure!(ran, "mma_f8_block declined this shape — bench is measuring nothing");
    }
    dev.synchronize()?;

    // Eager: one launch, one host-side wait for the whole run, same as the
    // other kernel benches in this crate.
    let t = std::time::Instant::now();
    for _ in 0..EAGER_ITERS {
        k.mma_f8_block(&mut d_out.as_view_mut(), &d_w.as_view(), &d_x.as_view(), k_dim, n_dim, n_tokens, false)?;
    }
    dev.synchronize()?;
    let eager_us = t.elapsed().as_secs_f64() * 1e6 / EAGER_ITERS as f64;
    println!("eager:  {eager_us:.2} us/call ({EAGER_ITERS} calls, no graph)");

    // Captured: `GRAPH_REPS` launches recorded once, replayed `GRAPH_REPLAYS`
    // times. Timed the same way the model's own graph path is — the capture
    // itself is excluded, only replay.
    stream.begin_capture(CAPTURE_RELAXED)?;
    let cap_res = (|| -> Result<()> {
        for _ in 0..GRAPH_REPS {
            k.mma_f8_block(&mut d_out.as_view_mut(), &d_w.as_view(), &d_x.as_view(), k_dim, n_dim, n_tokens, false)?;
        }
        Ok(())
    })();
    let graph = stream.end_capture(GraphFlags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
    cap_res?;
    let graph = graph?.context("stream capture produced no graph")?;
    graph.upload()?;

    let t = std::time::Instant::now();
    for _ in 0..GRAPH_REPLAYS {
        graph.launch()?;
    }
    dev.synchronize()?;
    let graph_us = t.elapsed().as_secs_f64() * 1e6 / (GRAPH_REPLAYS * GRAPH_REPS) as f64;
    println!(
        "graph:  {graph_us:.2} us/call ({GRAPH_REPLAYS} replays of {GRAPH_REPS} calls each)"
    );
    println!(
        "graph/eager: {:.3}x",
        graph_us / eager_us
    );
    Ok(())
}
