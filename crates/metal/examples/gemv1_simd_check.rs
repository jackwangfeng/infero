//! Correctness and speed of `gemv1_simd_q4_K` -- a batch=1-only Q4_K gemv
//! modeled on llama.cpp's Metal `kernel_mul_mv_q4_K_f32`, which owns rows
//! per simdgroup and reduces with a bare `simd_sum` instead of a
//! threadgroup-wide barrier -- against the deployed `gemv1_q4_K`, which puts
//! one row on a whole threadgroup and reduces with `BLOCK_SUM`.
//!
//! `gemv1_q4_K` measures 92-145 GB/s on an M4 Max against its 546 GB/s peak;
//! the barrier every one of a decode step's ~190 Q4_K launches pays despite
//! never needing to combine results across more than one row is the leading
//! suspect (see the doc comment on `gemv1_simd_q4_K` in quant.metal).
//!
//!     cargo run --release -p infero-metal --example gemv1_simd_check

use anyhow::Result;
use infero_metal::{Device, LaunchConfig};

const COMMON: &str = include_str!("../../kernels/src/msl/common.metal");
const QUANT: &str = include_str!("../../kernels/src/msl/quant.metal");

fn ms(mut f: impl FnMut() -> Result<()>, iters: usize) -> Result<f64> {
    f()?;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        f()?;
    }
    Ok(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as i32 as f32) / (1i64 << 24) as f32
    }
}

fn bench_shape(
    dev: &Device,
    quant: &str,
    label: &str,
    k: usize,
    n: usize,
    rng: &mut Rng,
) -> Result<()> {
    let s = dev.stream();
    let nb = k / 256;
    let block_bytes = 144usize;
    let n_blocks = n * nb;
    let mut w_bytes_host = vec![0u8; n_blocks * block_bytes];
    for blk in 0..n_blocks {
        let base = blk * block_bytes;
        let d = half::f16::from_f32(0.01 + rng.next_f32().abs() * 0.05);
        let dmin = half::f16::from_f32(0.01 + rng.next_f32().abs() * 0.05);
        w_bytes_host[base..base + 2].copy_from_slice(&d.to_bits().to_le_bytes());
        w_bytes_host[base + 2..base + 4].copy_from_slice(&dmin.to_bits().to_le_bytes());
        for byte in &mut w_bytes_host[base + 4..base + block_bytes] {
            *byte = (rng.next_f32().to_bits() & 0xFF) as u8;
        }
    }
    let d_w = s.clone_htod(&w_bytes_host)?;
    let x_host: Vec<f32> = (0..k).map(|_| rng.next_f32()).collect();
    let d_x = s.clone_htod(&x_host)?;
    let mut out_old = s.alloc_zeros::<f32>(n)?;
    let mut out_new = s.alloc_zeros::<f32>(n)?;
    let (ki, ni, nti) = (k as i32, n as i32, 1i32);

    const GEMV_BLOCK_MAX: u32 = 128;
    let block = ((nb * 8) as u32).next_multiple_of(32).clamp(32, GEMV_BLOCK_MAX);
    let f_old = dev.kernels().get("quant", quant, "gemv1_q4_K")?;
    let t_old = ms(
        || {
            let mut b = s.launch_builder(&f_old);
            b.arg(&out_old.as_view_mut())
                .arg(&d_w.as_view())
                .arg(&d_x.as_view())
                .arg(&ki)
                .arg(&ni)
                .arg(&nti);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (n as u32, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                })?
            };
            s.synchronize()
        },
        30,
    )?;

    const ROWS: u32 = 2;
    const SGS: u32 = 4;
    let f_new = dev.kernels().get("quant", quant, "gemv1_simd_q4_K")?;
    let t_new = ms(
        || {
            let mut b = s.launch_builder(&f_new);
            b.arg(&out_new.as_view_mut())
                .arg(&d_w.as_view())
                .arg(&d_x.as_view())
                .arg(&ki)
                .arg(&ni)
                .arg(&nti);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: ((n as u32).div_ceil(ROWS * SGS), 1, 1),
                    block_dim: (32, SGS, 1),
                    shared_mem_bytes: 0,
                })?
            };
            s.synchronize()
        },
        30,
    )?;

    let a = s.clone_dtoh(&out_old)?;
    let b = s.clone_dtoh(&out_new)?;
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for row in 0..n {
        let (va, vb) = (a[row], b[row]);
        let ad = (va - vb).abs();
        max_abs = max_abs.max(ad);
        max_rel = max_rel.max(ad / va.abs().max(1e-6));
    }
    println!(
        "{label:24} k={k:6} n={n:6}  old {t_old:7.4}ms  simd {t_new:7.4}ms  speedup {:.2}x  \
         max_abs_diff {max_abs:.3e}  max_rel_diff {max_rel:.3e}",
        t_old / t_new,
    );
    Ok(())
}

fn main() -> Result<()> {
    let dev = Device::new(0)?;
    let quant = format!("{COMMON}\n{QUANT}");
    let mut rng = Rng(0x243F6A8885A308D3);

    bench_shape(&dev, &quant, "ffn_gate/up", 5120, 17408, &mut rng)?;
    bench_shape(&dev, &quant, "ffn_down", 17408, 5120, &mut rng)?;
    bench_shape(&dev, &quant, "small (attn-ish)", 5120, 1024, &mut rng)?;
    Ok(())
}
