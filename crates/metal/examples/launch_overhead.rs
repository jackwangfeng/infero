//! Where does a Metal decode step's "issue" time actually go?
//!
//! The scheduler's own per-step timing splits a decode step into issue (CPU
//! encodes ~880 dispatches into the open batch), sample (reads results back),
//! and advance (bookkeeping, which is where the wait for the GPU actually
//! lands since neither of the other two synchronises). A clean single-request
//! run measured issue_ms=7.2 against a 73.5ms step -- about 10%, small next
//! to the memory-bound GPU work, but not nothing. This isolates that 7.2ms:
//! is it the Metal driver's per-dispatch encode cost, or Rust-side overhead
//! (`Vec<Arg>` per launch, a `Vec<u8>` heap allocation per scalar argument)
//! this crate's own `LaunchBuilder` adds on top of it?
//!
//! The twin of `tuili-kernels/examples/launch_overhead.rs`, which measures
//! the same thing on CUDA.
//!
//!     cargo run --release -p tuili-metal --example launch_overhead

use anyhow::Result;
use tuili_kernels::Kernels;
use tuili_metal::Device;

fn main() -> Result<()> {
    let dev = Device::new(0)?;
    let k = Kernels::new(dev.clone());
    let stream = dev.stream();

    let a = stream.alloc_zeros::<f32>(1024)?;
    let b = stream.alloc_zeros::<f32>(1024)?;
    let mut out = stream.alloc_zeros::<f32>(1024)?;

    const N: usize = 5000;

    // Warm: first call compiles/caches the pipeline state.
    k.add(&mut out.as_view_mut(), &a.as_view(), &b.as_view(), 1024)?;
    dev.synchronize()?;

    // Enqueue only: how long the CPU spends encoding into the open batch.
    // `add` at 1024 elements is one dispatch, three buffer args, one scalar --
    // representative of the smaller elementwise kernels (rms_norm, add_assign,
    // silu_mul) that make up most of a decode step's ~880.
    let t = std::time::Instant::now();
    for _ in 0..N {
        k.add(&mut out.as_view_mut(), &a.as_view(), &b.as_view(), 1024)?;
    }
    let submit = t.elapsed();
    dev.synchronize()?;
    let total = t.elapsed();

    println!("{N} trivial launches (add, 1024 elements, batched)");
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

    // The same, but synchronizing every time -- forces one command buffer a
    // launch, same as the empty-dispatch number in `bandwidth.rs` but through
    // the real `Kernels::add` path rather than a hand-rolled launch.
    dev.synchronize()?;
    let t = std::time::Instant::now();
    for _ in 0..500 {
        k.add(&mut out.as_view_mut(), &a.as_view(), &b.as_view(), 1024)?;
        dev.synchronize()?;
    }
    let synced = t.elapsed();
    println!(
        "  with sync  : {:>8.2} us / launch",
        synced.as_secs_f64() * 1e6 / 500.0
    );

    // A launch with more scalar args than `add`'s one (rms_norm takes
    // n_tokens, d, eps -- three), to see whether `Arg::Bytes`'s per-scalar
    // heap allocation shows up as the count goes up.
    dev.synchronize()?;
    let t = std::time::Instant::now();
    for _ in 0..N {
        k.rms_norm(&mut out.as_view_mut(), &a.as_view(), &b.as_view(), 1, 1024, 1e-6)?;
    }
    let rms_submit = t.elapsed();
    dev.synchronize()?;
    println!("\n{N} rms_norm launches (3 scalar args instead of add's 1)");
    println!(
        "  cpu submit : {:>8.2} ms   ({:.2} us / launch)",
        rms_submit.as_secs_f64() * 1e3,
        rms_submit.as_secs_f64() * 1e6 / N as f64
    );

    // `GdnState::reset` calls `memset_zeros` 96 times (48 linear layers x
    // {recurrent, conv}) on every new sequence's admission, and
    // `memset_zeros` unconditionally synchronises first. Nothing is queued
    // between these calls, so if `synchronize` were free when there is
    // nothing to wait for, this loop would cost nothing. `queued_ms` on a
    // real request is ~1.9s regardless of prompt length, which does not fit
    // a per-token compute cost -- this isolates whether it fits a per-call
    // synchronize cost instead.
    dev.synchronize()?;
    let t = std::time::Instant::now();
    for _ in 0..96 {
        dev.synchronize()?;
    }
    let idle_sync = t.elapsed();
    println!("\n96 synchronize() calls with nothing queued");
    println!(
        "  total : {:>8.3} ms   ({:.2} us / call)",
        idle_sync.as_secs_f64() * 1e3,
        idle_sync.as_secs_f64() * 1e6 / 96.0
    );

    Ok(())
}
