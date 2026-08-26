//! Correctness and speed of `gemv_mma_shared16_q4_K` (a doubled row tile,
//! 16 rows instead of 8, decode still serial on simdgroup 0) against the
//! deployed `gemv_mma_shared_q4_K`.
//!
//! See the new kernel's doc comment in quant.metal for why this is being
//! asked before a much larger rewrite: llama.cpp's own Q4_K prefill matmul
//! uses a 64-row-by-32-token tile against tuili's 8-by-32, cooperatively
//! dequantized by all 128 threads in the threadgroup rather than serialized
//! on one simdgroup. This checks the cheaper half of that hypothesis first
//! -- does merely widening the row tile help even without also making the
//! decode cooperative -- before deciding whether the full rewrite is worth
//! the much larger risk this session already took once on a kernel that
//! won in isolation and lost in the real pipeline.
//!
//!     cargo run --release -p tuili-metal --example gemv_mma_shared16_check

use anyhow::Result;
use tuili_metal::{Device, LaunchConfig};

const COMMON: &str = include_str!("../../kernels/src/msl/common.metal");
const QUANT: &str = include_str!("../../kernels/src/msl/quant.metal");
const MMA_SHARED_TOKGROUPS: u32 = 4;

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

fn bench(dev: &Device, quant: &str, label: &str, k: usize, n: usize, rng: &mut Rng) -> Result<()> {
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

    let f_old = dev.kernels().get("quant", quant, "gemv_mma_shared_q4_K")?;
    let f_new = dev.kernels().get("quant", quant, "gemv_mma_shared16_q4_K")?;
    let f_32 = dev.kernels().get("quant", quant, "gemv_mma_shared32_q4_K")?;
    let f_coop = dev.kernels().get("quant", quant, "gemv_mma_coop32_q4_K")?;

    for &n_tokens in &[32usize, 48, 63, 90, 128] {
        let x_host: Vec<f32> = (0..n_tokens.next_multiple_of(8) * k)
            .map(|i| if i < n_tokens * k { rng.next_f32() } else { 0.0 })
            .collect();
        let d_x = s.clone_htod(&x_host)?;
        let mut out_old = s.alloc_zeros::<f32>(n * n_tokens)?;
        let mut out_new = s.alloc_zeros::<f32>(n * n_tokens)?;
        let (ki, ni, nti) = (k as i32, n as i32, n_tokens as i32);

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
                        grid_dim: (
                            (n as u32).div_ceil(8),
                            (n_tokens as u32).div_ceil(8 * MMA_SHARED_TOKGROUPS),
                            1,
                        ),
                        block_dim: (32, MMA_SHARED_TOKGROUPS, 1),
                        shared_mem_bytes: 0,
                    })?
                };
                s.synchronize()
            },
            15,
        )?;

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
                        grid_dim: (
                            (n as u32).div_ceil(16),
                            (n_tokens as u32).div_ceil(8 * MMA_SHARED_TOKGROUPS),
                            1,
                        ),
                        block_dim: (32, MMA_SHARED_TOKGROUPS, 1),
                        shared_mem_bytes: 0,
                    })?
                };
                s.synchronize()
            },
            15,
        )?;

        let mut out_32 = s.alloc_zeros::<f32>(n * n_tokens)?;
        let t_32 = ms(
            || {
                let mut b = s.launch_builder(&f_32);
                b.arg(&out_32.as_view_mut())
                    .arg(&d_w.as_view())
                    .arg(&d_x.as_view())
                    .arg(&ki)
                    .arg(&ni)
                    .arg(&nti);
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: (
                            (n as u32).div_ceil(32),
                            (n_tokens as u32).div_ceil(8 * MMA_SHARED_TOKGROUPS),
                            1,
                        ),
                        block_dim: (32, MMA_SHARED_TOKGROUPS, 1),
                        shared_mem_bytes: 0,
                    })?
                };
                s.synchronize()
            },
            15,
        )?;

        let mut out_coop = s.alloc_zeros::<f32>(n * n_tokens)?;
        let t_coop = ms(
            || {
                let mut b = s.launch_builder(&f_coop);
                b.arg(&out_coop.as_view_mut())
                    .arg(&d_w.as_view())
                    .arg(&d_x.as_view())
                    .arg(&ki)
                    .arg(&ni)
                    .arg(&nti);
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: (
                            (n as u32).div_ceil(32),
                            (n_tokens as u32).div_ceil(8 * MMA_SHARED_TOKGROUPS),
                            1,
                        ),
                        block_dim: (32, MMA_SHARED_TOKGROUPS, 1),
                        shared_mem_bytes: 0,
                    })?
                };
                s.synchronize()
            },
            15,
        )?;

        let a = s.clone_dtoh(&out_old)?;
        let b = s.clone_dtoh(&out_new)?;
        let c = s.clone_dtoh(&out_32)?;
        let d = s.clone_dtoh(&out_coop)?;
        let mut max_abs = 0.0f32;
        let mut max_abs_32 = 0.0f32;
        let mut max_abs_coop = 0.0f32;
        for i in 0..n * n_tokens {
            max_abs = max_abs.max((a[i] - b[i]).abs());
            max_abs_32 = max_abs_32.max((a[i] - c[i]).abs());
            max_abs_coop = max_abs_coop.max((a[i] - d[i]).abs());
        }
        println!(
            "{label:16} n_tokens={n_tokens:4}  8-row {t_old:7.3}ms  16-row {t_new:7.3}ms  \
             32-row {t_32:7.3}ms  coop32 {t_coop:7.3}ms  16x {:.2}  32x {:.2}  coopx {:.2}  \
             diff16 {max_abs:.1e}  diff32 {max_abs_32:.1e}  diffcoop {max_abs_coop:.1e}",
            t_old / t_new,
            t_old / t_32,
            t_old / t_coop,
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let dev = Device::new(0)?;
    let quant = format!("{COMMON}\n{QUANT}");
    let mut rng = Rng(0x3243F6A8885A308D);

    bench(&dev, &quant, "ffn_gate/up", 5120, 17408, &mut rng)?;
    bench(&dev, &quant, "ffn_down", 17408, 5120, &mut rng)?;
    Ok(())
}
