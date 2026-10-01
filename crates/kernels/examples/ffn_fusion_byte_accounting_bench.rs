//! Standalone, isolated measurement for the FFN down_proj silu_mul+quantize
//! fusion investigation (measurement/design phase only -- no fused kernel
//! here). Benchmarks the REAL kernels this codebase's down_proj/NVFP4 path
//! dispatches today (`silu_mul_f16` + `quantize_act_e2m1_cutlass_f16`) at the
//! real production shape, plus the plain f32 variants for comparison, and
//! reports wall-clock time + implied bandwidth for each so the fusion's
//! savings can be computed from real measurement rather than estimated.
//!
//! Does not touch the live server process or any shared state: allocates its
//! own buffers, runs in its own process.
//!
//!     CUDA_VISIBLE_DEVICES=3 cargo run --release -p infero-kernels \
//!         --example ffn_fusion_byte_accounting_bench

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::Kernels;
use half::f16;

const T: usize = 8192; // real chunk size per the production batch_tokens log
const D_FF: usize = 17408; // this checkpoint's real d_ff

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

fn bench<F: FnMut() -> Result<()>>(name: &str, reps: usize, mut f: F, dev: &Device) -> Result<f64> {
    for _ in 0..5 {
        f()?;
    }
    dev.context().synchronize()?;
    let start = std::time::Instant::now();
    for _ in 0..reps {
        f()?;
    }
    dev.context().synchronize()?;
    let elapsed = start.elapsed();
    let us = elapsed.as_secs_f64() * 1e6 / reps as f64;
    println!("{name}: {us:.1} us/call ({reps} reps)");
    Ok(us)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let dev = Device::new(0)?;
    println!("device: {} (sm_{}, {} SMs)", dev.name(), dev.arch(), dev.sm_count());
    let k = Kernels::new(dev.clone());
    let stream = dev.stream().clone();

    let n = T * D_FF;
    let blocks_per_row = D_FF.div_ceil(16);
    let gate = pseudo_random(n, 0xa1);
    let up = pseudo_random(n, 0xa2);
    let d_gate = stream.clone_htod(&gate)?;
    let d_up = stream.clone_htod(&up)?;
    let mut d_ffn_f32 = stream.alloc_zeros::<f32>(n)?;
    let mut d_ffn_f16 = stream.alloc_zeros::<f16>(n)?;
    let mut d_xq = stream.alloc_zeros::<u8>(n.div_ceil(2))?;
    let mut d_xs = stream.alloc_zeros::<u8>(T * blocks_per_row)?;
    let reps = 50;

    println!("\n=== T={T} d_ff={D_FF} n={n} (elements) ===\n");

    // 1. silu_mul_f16: TODAY's real down_proj dispatch (down_is_nvfp4==true).
    //    Reads gate+up (f32), writes BOTH ffn (f32, dead for down_proj's own
    //    consumption -- see lib.rs:7244's x_f16 short-circuit) and ffn_f16.
    let us_silu_f16 = bench(
        "silu_mul_f16        (read gate+up f32, write ffn f32 [dead]+ffn_f16)",
        reps,
        || {
            k.silu_mul_f16(
                &mut d_ffn_f32.as_view_mut(),
                &mut d_ffn_f16.as_view_mut(),
                &d_gate.as_view(),
                &d_up.as_view(),
                n,
            )
        },
        &dev,
    )?;
    let bytes_silu_f16 = (n * 4 * 2 + n * 4 + n * 2) as f64; // read gate+up f32, write ffn f32 + ffn_f16
    println!(
        "  bytes moved: {:.0} MiB, implied BW: {:.1} GB/s\n",
        bytes_silu_f16 / 1048576.0,
        bytes_silu_f16 / (us_silu_f16 / 1e6) / 1e9
    );

    // 2. plain silu_mul (f32 only) -- what the non-NVFP4-down path uses, for
    //    comparison against #1.
    let us_silu_f32 = bench(
        "silu_mul             (read gate+up f32, write ffn f32 only)",
        reps,
        || {
            k.silu_mul(
                &mut d_ffn_f32.as_view_mut(),
                &d_gate.as_view(),
                &d_up.as_view(),
                n,
            )
        },
        &dev,
    )?;
    let bytes_silu_f32 = (n * 4 * 3) as f64;
    println!(
        "  bytes moved: {:.0} MiB, implied BW: {:.1} GB/s\n",
        bytes_silu_f32 / 1048576.0,
        bytes_silu_f32 / (us_silu_f32 / 1e6) / 1e9
    );

    // 3. quantize_act_e2m1_cutlass_f16: TODAY's real down_proj quantizer
    //    dispatch (reads the ffn_f16 shadow copy #1 wrote).
    let us_quant_f16 = bench(
        "quantize_act_e2m1_cutlass_f16 (read ffn_f16, write xq+xs)",
        reps,
        || {
            k.quantize_act_e2m1_cutlass_f16(
                &mut d_xq.as_view_mut(),
                &mut d_xs.as_view_mut(),
                &d_ffn_f16.as_view(),
                1.0,
                D_FF,
                T,
            )
        },
        &dev,
    )?;
    let bytes_quant_f16 = (n * 2 + n / 2 + T * blocks_per_row) as f64;
    println!(
        "  bytes moved: {:.0} MiB, implied BW: {:.1} GB/s\n",
        bytes_quant_f16 / 1048576.0,
        bytes_quant_f16 / (us_quant_f16 / 1e6) / 1e9
    );

    // 4. quantize_act_e2m1_cutlass (f32 variant) -- what gate/up's own
    //    (unrelated, already-deduped) quantize call uses, and what down_proj
    //    would use if the f16-shadow experiment were reverted instead of
    //    fused.
    let us_quant_f32 = bench(
        "quantize_act_e2m1_cutlass     (read ffn f32, write xq+xs)",
        reps,
        || {
            k.quantize_act_e2m1_cutlass(
                &mut d_xq.as_view_mut(),
                &mut d_xs.as_view_mut(),
                &d_ffn_f32.as_view(),
                1.0,
                D_FF,
                T,
            )
        },
        &dev,
    )?;
    let bytes_quant_f32 = (n * 4 + n / 2 + T * blocks_per_row) as f64;
    println!(
        "  bytes moved: {:.0} MiB, implied BW: {:.1} GB/s\n",
        bytes_quant_f32 / 1048576.0,
        bytes_quant_f32 / (us_quant_f32 / 1e6) / 1e9
    );

    println!("=== Summary (one layer, one T={T} chunk) ===");
    let today_us = us_silu_f16 + us_quant_f16;
    let today_bytes = bytes_silu_f16 + bytes_quant_f16;
    println!(
        "TODAY (silu_mul_f16 + quantize_act_e2m1_cutlass_f16): {today_us:.1} us/layer/chunk, \
         {:.0} MiB moved",
        today_bytes / 1048576.0
    );
    let alt_us = us_silu_f32 + us_quant_f32;
    let alt_bytes = bytes_silu_f32 + bytes_quant_f32;
    println!(
        "ALT   (plain silu_mul + quantize_act_e2m1_cutlass, no f16 shadow): {alt_us:.1} us/layer/chunk, \
         {:.0} MiB moved  (net-neutral check vs TODAY)",
        alt_bytes / 1048576.0
    );
    // The fused kernel this session is scoping (not built here): reads
    // gate+up once (same as silu_mul_f16/silu_mul's own read), writes only
    // xq+xs directly -- no ffn/ffn_f16 intermediate at all.
    let bytes_fused = (n * 4 * 2 + n / 2 + T * blocks_per_row) as f64;
    println!(
        "FUSED (hypothetical: read gate+up, write xq+xs only): {:.0} MiB moved, \
         {:.0} MiB ({:.1}% of TODAY) eliminated",
        bytes_fused / 1048576.0,
        (today_bytes - bytes_fused) / 1048576.0,
        100.0 * (today_bytes - bytes_fused) / today_bytes
    );
    Ok(())
}
