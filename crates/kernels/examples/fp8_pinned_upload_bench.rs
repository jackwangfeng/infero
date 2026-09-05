//! Real H2D upload-bandwidth comparison: a plain (pageable) `Vec<u8>` versus
//! a `PinnedHostSlice<u8>` allocated via `infero_kernels::fp8::pinned`,
//! filled identically, uploaded through the same `Stream::memcpy_htod` call.
//! This isolates the ONE variable the pinned-allocation work targets (H2D
//! transfer rate), independent of the repack/pad/concat CPU-side cost those
//! functions already share with their plain `Vec`-returning counterparts.
//!
//!   cargo run --release --features cuda -p infero-kernels --example fp8_pinned_upload_bench

use anyhow::{Context, Result};
use infero_gpu::Device;
use infero_kernels::fp8::pinned::pad_rows_pinned;
use std::time::Instant;

fn main() -> Result<()> {
    let dev = Device::new(0)?;
    let stream = dev.stream();

    // ~170 MiB, matching this file's own doc comment on `concat`'s original
    // bottleneck ("gate/up's quants alone run ~170 MiB a layer on the 27B").
    let k = 5120usize;
    let n = 17408usize; // FFN intermediate width on the real checkpoint this session profiled
    let quants: Vec<u8> = (0..(k * n)).map(|i| (i % 251) as u8).collect();
    let bytes = quants.len();
    println!("payload: {bytes} bytes ({:.1} MiB)", bytes as f64 / (1024.0 * 1024.0));

    const REPS: usize = 10;

    // Pageable path: plain Vec, the same shape `pad_rows` already returns,
    // uploaded via the identical `memcpy_htod` call the pinned path uses.
    let pageable = infero_kernels::fp8::pad_rows(&quants, k, n).context("pad_rows")?;
    let padded_len = pageable.len();
    let mut dev_buf = stream.alloc_zeros::<u8>(padded_len)?;
    stream.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..REPS {
        stream.memcpy_htod(&pageable, &mut dev_buf.slice_mut(..padded_len))?;
    }
    stream.synchronize()?;
    let pageable_elapsed = t0.elapsed();
    let pageable_ms = pageable_elapsed.as_secs_f64() * 1000.0 / REPS as f64;
    let pageable_gbps = (padded_len as f64 * REPS as f64)
        / pageable_elapsed.as_secs_f64()
        / 1e9;
    println!("pageable Vec<u8>: {pageable_ms:.3} ms/upload, {pageable_gbps:.2} GB/s");

    // Pinned path: same bytes, same padded length, allocated via the real
    // pinned-allocation primitive this session's weight-load work already
    // uses for layer-offload staging (`pack_layer`'s `alloc_pinned`).
    let mut pinned = pad_rows_pinned(&dev, &quants, k, n).context("pad_rows_pinned")?;
    assert_eq!(pinned.as_mut_slice()?.len(), padded_len);
    let t0 = Instant::now();
    for _ in 0..REPS {
        stream.memcpy_htod(&pinned, &mut dev_buf.slice_mut(..padded_len))?;
    }
    stream.synchronize()?;
    let pinned_elapsed = t0.elapsed();
    let pinned_ms = pinned_elapsed.as_secs_f64() * 1000.0 / REPS as f64;
    let pinned_gbps = (padded_len as f64 * REPS as f64) / pinned_elapsed.as_secs_f64() / 1e9;
    println!("pinned PinnedHostSlice<u8>: {pinned_ms:.3} ms/upload, {pinned_gbps:.2} GB/s");

    println!(
        "speedup: {:.2}x ({pageable_ms:.3} ms -> {pinned_ms:.3} ms)",
        pageable_ms / pinned_ms
    );

    Ok(())
}
