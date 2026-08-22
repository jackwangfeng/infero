//! What does a marginal row cost the batched FP8 mat-vec, and why?
//!
//! The batched mat-vec reads the weights once whatever the row count, so a
//! second row adds no DRAM traffic and it costs 2.25 ms of a decode step
//! anyway. Three explanations have been tried end to end and all three were
//! wrong — the arithmetic is an order of magnitude too small, the request count
//! is what the row-interleave already fixed, and halving the activation bytes
//! measured *slower*. Their obituaries are on `fp8::BATCH_KERNELS`.
//!
//! So: measure the kernel on its own, at the 27B's widest projection, and take
//! pieces away. `TUILI_FP8_STRIP` selects what the kernel skips:
//!
//! ```text
//!   (unset)   everything
//!   fma       the multiply-accumulate, keeping the loads
//!   reduce    the per-row-per-token reduction, keeping the loop
//!   both      loop over the weights and write nothing
//! ```
//!
//! What each answer would mean. If stripping `fma` flattens the row curve, the
//! arithmetic is the cost after all and the end-to-end estimate was wrong. If
//! stripping `reduce` flattens it, it is the twelve shuffle chains a block runs
//! at three tokens against four at one. If neither does, the cost is in the
//! loads the strip leaves behind, and the next thing to vary is the block shape.
//!
//! **The weights have to come from DRAM.** One 85 MiB projection reused forty
//! times sits in this card's 128 MB L2 after the first rep, and then the kernel
//! is not reading memory at all — the first version of this probe reported 3597
//! GB/s with the reduction stripped, which is twice this card's DRAM bandwidth
//! and should have been the tell. A decode step streams 29.6 GB and caches none
//! of it, so the probe rotates through enough copies to evict.
//!
//!     cargo run --release -p tuili-kernels --example fp8_row_cost

use anyhow::Result;
use tuili_cuda::Device;
use tuili_kernels::Kernels;
use tuili_kernels::fp8::{FP8_BLOCK, fp8_bytes, repack_rows, scale_grid};

/// The 27B's widest projection: `down_proj` is `[5120, 17408]`, and `gate`/`up`
/// are `[17408, 5120]`. Taking the second shape means many output rows over a
/// short contraction, which is the shape the row-interleave is about.
const N: usize = 17408;
const K: usize = 5120;

/// Distinct copies of the matrix to rotate through, so that a rep reads DRAM
/// rather than L2. 85 MiB each, so four is 340 MB against a 128 MB L2.
const COPIES: usize = 4;

/// Enough repetitions that a single launch's latency is not the measurement,
/// and few enough that the whole sweep runs in seconds.
const REPS: usize = 40;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let dev = Device::new(
        std::env::var("TUILI_DEVICE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
    )?;
    let k = Kernels::new(dev.clone());
    k.warm_up()?;
    let stream = dev.stream().clone();

    // Deterministic bytes that avoid E4M3's two NaN patterns, and scales that
    // are not one, so nothing can be constant-folded and the numbers stay real.
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let quants: Vec<u8> = (0..N * K)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let b = (s >> 24) as u8;
            if b == 0x7F || b == 0xFF { 0x38 } else { b }
        })
        .collect();
    let scales: Vec<f32> = (0..scale_grid(K, N))
        .map(|i| 0.3 + 0.4 * (i % 7) as f32)
        .collect();
    let mut buf = repack_rows(&quants, K, N)?;
    for v in &scales {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(buf.len(), fp8_bytes(K, N));
    // The same bytes in `COPIES` places. Identical contents keep the answer
    // comparable across reps; distinct *addresses* are what defeats L2.
    let weights: Vec<_> = (0..COPIES)
        .map(|_| stream.clone_htod(&buf))
        .collect::<Result<Vec<_>, _>>()?;
    println!(
        "  {} MiB resident, against a 128 MB L2",
        (COPIES * buf.len()) >> 20
    );

    let strip = std::env::var("TUILI_FP8_STRIP").unwrap_or_default();
    println!(
        "  [{N}, {K}] FP8, {} MiB of weights, block {FP8_BLOCK}, strip={:?}",
        (N * K) >> 20,
        if strip.is_empty() { "none" } else { &strip }
    );
    println!("  {:>6}  {:>9}  {:>9}  {:>10}", "tokens", "ms", "GB/s", "marginal");

    let mut prev = 0.0f64;
    for n_tokens in [1usize, 2, 3, 4, 8] {
        let x: Vec<f32> = (0..n_tokens * K)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 97.0)
            .collect();
        let d_x = stream.clone_htod(&x)?;
        let mut d_out = stream.alloc_zeros::<f32>(n_tokens * N)?;

        // One untimed call, then the sweep with a single drain at the end: this
        // is a kernel's throughput, and the harness must not measure a launch.
        run(&k, &mut d_out, &weights[0], &d_x, n_tokens)?;
        dev.synchronize()?;
        let t0 = std::time::Instant::now();
        for rep in 0..REPS {
            run(&k, &mut d_out, &weights[rep % COPIES], &d_x, n_tokens)?;
        }
        dev.synchronize()?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / REPS as f64;

        // The weights are read once whatever the row count, so this is the rate
        // the *weights* move at and nothing else.
        let gbs = (N * K) as f64 / (ms / 1000.0) / 1e9;
        let marginal = if n_tokens == 1 {
            String::from("-")
        } else {
            format!("{:+.3} ms", (ms - prev) / (n_tokens - 1) as f64)
        };
        println!("  {n_tokens:>6}  {ms:>9.3}  {gbs:>9.0}  {marginal:>10}");
        if n_tokens == 1 {
            prev = ms;
        }
    }
    Ok(())
}

fn run(
    k: &Kernels,
    out: &mut cudarc::driver::CudaSlice<f32>,
    w: &cudarc::driver::CudaSlice<u8>,
    x: &cudarc::driver::CudaSlice<f32>,
    n_tokens: usize,
) -> Result<()> {
    if n_tokens == 1 {
        k.mmv_f8_block(&mut out.as_view_mut(), &w.as_view(), &x.as_view(), K, N, false)?;
        return Ok(());
    }
    let ran = k.mmv_f8_block_batch(
        &mut out.as_view_mut(),
        &w.as_view(),
        &x.as_view(),
        K,
        N,
        n_tokens,
        false,
    )?;
    anyhow::ensure!(ran, "the batched mat-vec declined {n_tokens} tokens");
    Ok(())
}
