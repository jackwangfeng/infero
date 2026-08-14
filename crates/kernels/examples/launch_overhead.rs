//! Where does a decode step's time actually go?
//!
//! A single-token forward pass issues roughly 500 kernels. If the per-launch
//! cost is tens of microseconds, no amount of kernel tuning matters — the fix
//! is to launch less. This measures the floor.
//!
//!     cargo run --release -p tuili-kernels --example launch_overhead

use anyhow::Result;
use tuili_cuda::Device;
use tuili_kernels::Kernels;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();

    let k = Kernels::new(Device::new(0)?);
    k.warm_up()?;
    let stream = k.device().stream().clone();

    // Small buffers: the point is the launch, not the work.
    let a = stream.alloc_zeros::<f32>(1024)?;
    let b = stream.alloc_zeros::<f32>(1024)?;
    let mut out = stream.alloc_zeros::<f32>(1024)?;

    const N: usize = 5000;

    // Enqueue only: how long the CPU spends submitting.
    let t = std::time::Instant::now();
    for _ in 0..N {
        k.add(&mut out.as_view_mut(), &a.as_view(), &b.as_view(), 1024)?;
    }
    let submit = t.elapsed();
    k.device().synchronize()?;
    let total = t.elapsed();

    println!("{N} trivial launches");
    println!(
        "  cpu submit : {:>8.2} ms   ({:.2} us / launch)",
        submit.as_secs_f64() * 1e3,
        submit.as_secs_f64() * 1e6 / N as f64
    );
    println!(
        "  wall total : {:>8.2} ms   ({:.2} us / launch)",
        total.as_secs_f64() * 1e3,
        total.as_secs_f64() * 1e6 / N as f64
    );

    // The same, but synchronizing every time — what a per-token host readback
    // costs if it lands in the middle of the pipeline.
    let t = std::time::Instant::now();
    for _ in 0..500 {
        k.add(&mut out.as_view_mut(), &a.as_view(), &b.as_view(), 1024)?;
        k.device().synchronize()?;
    }
    let synced = t.elapsed();
    println!(
        "  with sync  : {:>8.2} us / launch",
        synced.as_secs_f64() * 1e6 / 500.0
    );

    // A 600 KB device-to-host copy, the per-token logits readback.
    let logits = stream.alloc_zeros::<f32>(151_936)?;
    let mut host = vec![0.0f32; 151_936];
    k.device().synchronize()?;
    let t = std::time::Instant::now();
    for _ in 0..200 {
        stream.memcpy_dtoh(&logits, &mut host)?;
        k.device().synchronize()?;
    }
    println!(
        "\nlogits readback (600 KiB + sync): {:.1} us",
        t.elapsed().as_secs_f64() * 1e6 / 200.0
    );

    // The ceiling on CPU offload: streaming a layer's weights in is a
    // page-locked host-to-device copy, so this is what "how slow is offload"
    // ultimately reduces to.
    const MIB: usize = 64;
    let bytes = MIB << 20;
    let ctx = k.device().context().clone();
    // Safety: written once below, and dropped at the end of main.
    let mut pinned = unsafe { ctx.alloc_pinned::<u8>(bytes) }?;
    pinned.as_mut_slice()?.fill(7);
    let mut dst = stream.alloc_zeros::<u8>(bytes)?;

    for _ in 0..3 {
        stream.memcpy_htod(&pinned, &mut dst)?;
    }
    k.device().synchronize()?;
    let reps = 20;
    let t = std::time::Instant::now();
    for _ in 0..reps {
        stream.memcpy_htod(&pinned, &mut dst)?;
    }
    k.device().synchronize()?;
    let secs = t.elapsed().as_secs_f64();
    println!(
        "pinned host->device: {:.1} GB/s ({MIB} MiB x {reps})",
        (bytes * reps) as f64 / secs / 1e9
    );

    let mut pageable = vec![7u8; bytes];
    pageable[0] = 1;
    k.device().synchronize()?;
    let t = std::time::Instant::now();
    for _ in 0..reps {
        stream.memcpy_htod(pageable.as_slice(), &mut dst)?;
    }
    k.device().synchronize()?;
    println!(
        "pageable host->device: {:.1} GB/s",
        (bytes * reps) as f64 / t.elapsed().as_secs_f64() / 1e9
    );
    Ok(())
}
