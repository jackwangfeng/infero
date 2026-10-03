//! Standalone verification: how long does `Kernels::silu_mul` (the plain,
//! non-split f32 SwiGLU kernel the NVFP4 checkpoint's FFN dispatch actually
//! uses) take at T=8192, d_ff=17408 -- the shape an adversarial review is
//! trying to confirm a claimed 938.2 us/call number for.
//!
//!     cargo run --release -p infero-kernels --example silu_mul_verify_bench

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::Kernels;

const T: usize = 8192;
const D_FF: usize = 17408;

fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
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
    let dev = Device::new(0)?;
    println!("device: {} (sm_{}, {} SMs)", dev.name(), dev.arch(), dev.sm_count());
    let k = Kernels::new(dev.clone());
    let stream = dev.stream().clone();

    let n = T * D_FF;
    let gate = pseudo_random(n, 0xa1);
    let up = pseudo_random(n, 0xa2);
    let d_gate = stream.clone_htod(&gate)?;
    let d_up = stream.clone_htod(&up)?;
    let mut d_out = stream.alloc_zeros::<f32>(n)?;

    // Warm up.
    for _ in 0..5 {
        k.silu_mul(&mut d_out.as_view_mut(), &d_gate.as_view(), &d_up.as_view(), n)?;
    }
    dev.context().synchronize()?;

    let reps = 50;
    let start = std::time::Instant::now();
    for _ in 0..reps {
        k.silu_mul(&mut d_out.as_view_mut(), &d_gate.as_view(), &d_up.as_view(), n)?;
    }
    dev.context().synchronize()?;
    let elapsed = start.elapsed();
    let us_per_call = elapsed.as_secs_f64() * 1e6 / reps as f64;
    println!("infero silu_mul_f32: T={T} d_ff={D_FF}: {us_per_call:.1} us/call ({reps} reps)");
    let bytes_moved = (n * 4 * 3) as f64; // read gate+up, write out, all f32
    let gbps = bytes_moved / (us_per_call / 1e6) / 1e9;
    println!("implied bandwidth: {gbps:.1} GB/s");
    Ok(())
}
