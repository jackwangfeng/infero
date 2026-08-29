//! Correctness and speed of `gemv_mma_multisg_q4_K` against both the
//! deployed single-simdgroup `gemv_mma_q4_K` and the GEMM path
//! (`dequant_q4_K_f16_vec` + `gemm_f16`, MPS) that wins at prefill's actual
//! token counts today.
//!
//! `gemm_f16_overhead.rs` found MPS's GFLOPS climbing hard with token count
//! -- 15% of this GPU's peak at 20, 76% by 512 -- and a real end-to-end A/B
//! (forcing every Q4_K matmul onto `gemv_mma_q4_K` instead of GEMM) measured
//! that kernel *losing* to GEMM at 61 tokens, 908ms against 850ms for the
//! whole request. One simdgroup a threadgroup is the suspect: it leaves the
//! other simdgroup slots a core can run concurrently idle every dispatch,
//! which is a plausible reason MPS -- which keeps many independent tiles in
//! flight -- wins. `gemv_mma_multisg_q4_K` gives four simdgroups their own
//! tile each, decoding independent rows with no data sharing between them.
//!
//!     cargo run --release -p infero-metal --example gemv_mma_multisg_check

use anyhow::Result;
use infero_metal::{Device, LaunchConfig};

const COMMON: &str = include_str!("../../kernels/src/msl/common.metal");
const QUANT: &str = include_str!("../../kernels/src/msl/quant.metal");
const MMA_SIMDGROUPS: u32 = 4;

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

fn main() -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();
    let quant = format!("{COMMON}\n{QUANT}");

    let k = 5120usize;
    let n = 17408usize;
    let nb = k / 256;
    let block_bytes = 144usize;
    let n_blocks = n * nb;

    let mut rng = Rng(0xABCDEF0123456789);
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

    let f_old = dev.kernels().get("quant", &quant, "gemv_mma_q4_K")?;
    let f_multisg = dev.kernels().get("quant", &quant, "gemv_mma_multisg_q4_K")?;
    let f_shared = dev.kernels().get("quant", &quant, "gemv_mma_shared_q4_K")?;
    const MMA_SHARED_TOKGROUPS: u32 = 4;

    for &n_tokens in &[8usize, 20, 32, 53, 61, 90, 128] {
        // Padded to a multiple of eight tokens: every kernel here may
        // overread its last (possibly partial) tile by up to seven tokens,
        // same as the deployed single-simdgroup kernel already does, and
        // this is the buffer-padding a real caller is expected to provide.
        let x_host: Vec<f32> = (0..n_tokens.next_multiple_of(8) * k)
            .map(|i| if i < n_tokens * k { rng.next_f32() } else { 0.0 })
            .collect();
        let d_x = s.clone_htod(&x_host)?;
        let mut out_old = s.alloc_zeros::<f32>(n * n_tokens)?;
        let mut out_multisg = s.alloc_zeros::<f32>(n * n_tokens)?;
        let mut out_shared = s.alloc_zeros::<f32>(n * n_tokens)?;
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
                        grid_dim: ((n as u32).div_ceil(8), (n_tokens as u32).div_ceil(8), 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    })?
                };
                s.synchronize()
            },
            10,
        )?;

        let t_multisg = ms(
            || {
                let mut b = s.launch_builder(&f_multisg);
                b.arg(&out_multisg.as_view_mut())
                    .arg(&d_w.as_view())
                    .arg(&d_x.as_view())
                    .arg(&ki)
                    .arg(&ni)
                    .arg(&nti);
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: ((n as u32).div_ceil(8 * MMA_SIMDGROUPS), (n_tokens as u32).div_ceil(8), 1),
                        block_dim: (32, MMA_SIMDGROUPS, 1),
                        shared_mem_bytes: 0,
                    })?
                };
                s.synchronize()
            },
            10,
        )?;

        let t_shared = ms(
            || {
                let mut b = s.launch_builder(&f_shared);
                b.arg(&out_shared.as_view_mut())
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
            10,
        )?;

        let a = s.clone_dtoh(&out_old)?;
        let b = s.clone_dtoh(&out_multisg)?;
        let c = s.clone_dtoh(&out_shared)?;
        let mut max_abs = 0.0f32;
        let mut max_abs_shared = 0.0f32;
        for i in 0..n * n_tokens {
            max_abs = max_abs.max((a[i] - b[i]).abs());
            max_abs_shared = max_abs_shared.max((a[i] - c[i]).abs());
        }
        println!(
            "n_tokens={n_tokens:4}  single-sg {t_old:7.3}ms  multi-sg(4) {t_multisg:7.3}ms ({:.2}x, diff {max_abs:.1e})  shared(4) {t_shared:7.3}ms ({:.2}x, diff {max_abs_shared:.1e})",
            t_old / t_multisg,
            t_old / t_shared,
        );
    }
    Ok(())
}
