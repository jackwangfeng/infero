//! Where the Q8_0 gemv-vs-GEMM crossover actually is, independent of Q4_K's.
//!
//! `GEMM_THRESHOLD_DEFAULT` (16, non-CUDA) was measured "through the 27B,
//! one chunk each" -- a whole-model sweep dominated by Q4_K FFN bytes, not a
//! per-weight-type measurement -- and Q4_K later got its own override
//! (`TUILI_Q4K_MMA_MAX`) once that stopped being good enough for it
//! specifically. Q8_0 (every GDN and attention projection: `in_proj_qkv`,
//! `in_proj_ba`, `out_proj`, `wq`/`wk`/`wv`/`wo`) still shares the Q4_K-tuned
//! knob and has never been checked on its own. If its real crossover is
//! higher than 16, every prefill chunk between 16 and that point is paying
//! MPS's low-M inefficiency (`gemm_f16_overhead.rs`: 15% of peak at 20
//! tokens) for matrices that would have been faster on the plain batched
//! `gemv_q8_0` mat-vec.
//!
//!     cargo run --release -p tuili-metal --example gemv_q8_0_threshold_check

use anyhow::Result;
use tuili_metal::{Device, LaunchConfig};

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

fn bench(dev: &Device, quant: &str, label: &str, k: usize, n: usize, rng: &mut Rng) -> Result<()> {
    let s = dev.stream();
    let nb = k / 32;
    let block_bytes = 34usize; // block_q8_0: half d + 32 x i8
    let n_blocks = n * nb;
    let mut w_bytes_host = vec![0u8; n_blocks * block_bytes];
    for blk in 0..n_blocks {
        let base = blk * block_bytes;
        let d = half::f16::from_f32(0.01 + rng.next_f32().abs() * 0.05);
        w_bytes_host[base..base + 2].copy_from_slice(&d.to_bits().to_le_bytes());
        for byte in &mut w_bytes_host[base + 2..base + block_bytes] {
            *byte = (rng.next_f32().to_bits() & 0xFF) as u8;
        }
    }
    let d_w = s.clone_htod(&w_bytes_host)?;

    for &n_tokens in &[8usize, 16, 24, 32, 48, 63, 90, 128] {
        // Padded to a multiple of eight tokens: gemv_mma_q8_0's 8x8 tiling
        // may read up to seven tokens past n_tokens on its last tile, same
        // as gemv_mma_q4_K already does -- the buffer padding a real caller
        // is expected to provide.
        let x_host: Vec<f32> = (0..n_tokens.next_multiple_of(8) * k)
            .map(|i| if i < n_tokens * k { rng.next_f32() } else { 0.0 })
            .collect();
        let d_x = s.clone_htod(&x_host)?;
        let mut out_gemv = s.alloc_zeros::<f32>(n * n_tokens)?;
        let mut out_gemm = s.alloc_zeros::<f32>(n * n_tokens)?;
        let mut out_mma = s.alloc_zeros::<f32>(n * n_tokens)?;
        let (ki, ni, nti) = (k as i32, n as i32, n_tokens as i32);

        // The batched gemv, GEMV_TOKENS = 8 a threadgroup, grid.y covers the rest.
        const GEMV_BLOCK_MAX: u32 = 128;
        let per_block = 32 / 8; // Q8_0_PER_THREAD = 8
        let chunks = (k / 32) * per_block;
        let block = (chunks as u32).next_multiple_of(32).clamp(32, GEMV_BLOCK_MAX);
        let f_gemv = dev.kernels().get("quant", quant, "gemv_q8_0")?;
        let t_gemv = ms(
            || {
                let mut b = s.launch_builder(&f_gemv);
                b.arg(&out_gemv.as_view_mut())
                    .arg(&d_w.as_view())
                    .arg(&d_x.as_view())
                    .arg(&ki)
                    .arg(&ni)
                    .arg(&nti);
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: (n as u32, (n_tokens as u32).div_ceil(8), 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    })?
                };
                s.synchronize()
            },
            15,
        )?;

        // dequant_to_f16 (Q8_0, vectorized) + gemm_f16 (MPS), the path
        // `matmul_pre` takes above `gemm_threshold()`.
        let f_dq = dev.kernels().get("quant", quant, "dequant_q8_0_f16_vec")?;
        let mut w16 = s.alloc_zeros::<half::f16>(n * k)?;
        let n_elements = (n * k) as u32;
        let t_total = ms(
            || {
                let mut b = s.launch_builder(&f_dq);
                let ne = n_elements;
                b.arg(&w16.as_view_mut()).arg(&d_w.as_view()).arg(&ne);
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: ((n_elements / 32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })?
                };
                let x16: Vec<half::f16> = x_host.iter().map(|&v| half::f16::from_f32(v)).collect();
                let d_x16 = s.clone_htod(&x16)?;
                tuili_metal::backend::gemm_f16_to_f32(
                    dev,
                    &mut out_gemm.as_view_mut(),
                    &d_x16.as_view(),
                    &w16.as_view(),
                    n_tokens,
                    k,
                    n,
                )?;
                s.synchronize()
            },
            15,
        )?;

        let f_mma = dev.kernels().get("quant", quant, "gemv_mma_q8_0")?;
        let t_mma = ms(
            || {
                let mut b = s.launch_builder(&f_mma);
                b.arg(&out_mma.as_view_mut())
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
            15,
        )?;

        // Random +-0.5 activations dot a random-signed-byte weight row average
        // out near zero over k = 5120 terms, so a relative-error metric here
        // blows up on plain rounding noise around that cancellation; absolute
        // error is the meaningful check for this random fixture.
        let a = s.clone_dtoh(&out_gemv)?;
        let b = s.clone_dtoh(&out_gemm)?;
        let c = s.clone_dtoh(&out_mma)?;
        let mut max_abs_gemm = 0.0f32;
        let mut max_abs_mma = 0.0f32;
        for i in 0..n * n_tokens {
            max_abs_gemm = max_abs_gemm.max((a[i] - b[i]).abs());
            max_abs_mma = max_abs_mma.max((a[i] - c[i]).abs());
        }
        println!(
            "{label:16} n_tokens={n_tokens:4}  gemv {t_gemv:7.3}ms  gemm(dequant+mps) {t_total:7.3}ms  \
             mma {t_mma:7.3}ms  mma-vs-gemm {:.2}x  mma-vs-gemv {:.2}x  \
             diff(gemm) {max_abs_gemm:.2e}  diff(mma) {max_abs_mma:.2e}",
            t_total / t_mma,
            t_gemv / t_mma,
        );
    }
    Ok(())
}

/// Just the dequant kernel, old (one thread a element) against vectorized
/// (one thread a 32-element block), independent of the gemv-vs-GEMM question
/// -- this is the fixed cost every GEMM-path call pays regardless of where
/// the crossover sits.
fn dequant_only(dev: &Device, quant: &str, k: usize, n: usize, rng: &mut Rng) -> Result<()> {
    let s = dev.stream();
    let nb = k / 32;
    let block_bytes = 34usize;
    let n_blocks = n * nb;
    let mut w_bytes_host = vec![0u8; n_blocks * block_bytes];
    for blk in 0..n_blocks {
        let base = blk * block_bytes;
        let d = half::f16::from_f32(0.01 + rng.next_f32().abs() * 0.05);
        w_bytes_host[base..base + 2].copy_from_slice(&d.to_bits().to_le_bytes());
        for byte in &mut w_bytes_host[base + 2..base + block_bytes] {
            *byte = (rng.next_f32().to_bits() & 0xFF) as u8;
        }
    }
    let d_w = s.clone_htod(&w_bytes_host)?;
    let n_elements = (n * k) as u32;
    let mut w16_old = s.alloc_zeros::<half::f16>(n * k)?;
    let mut w16_new = s.alloc_zeros::<half::f16>(n * k)?;

    let f_old = dev.kernels().get("quant", quant, "dequant_q8_0_f16")?;
    let t_old = ms(
        || {
            let mut b = s.launch_builder(&f_old);
            let ne = n_elements;
            b.arg(&w16_old.as_view_mut()).arg(&d_w.as_view()).arg(&ne);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (n_elements.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?
            };
            s.synchronize()
        },
        20,
    )?;

    let f_new = dev.kernels().get("quant", quant, "dequant_q8_0_f16_vec")?;
    let t_new = ms(
        || {
            let mut b = s.launch_builder(&f_new);
            let ne = n_elements;
            b.arg(&w16_new.as_view_mut()).arg(&d_w.as_view()).arg(&ne);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: ((n_elements / 32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?
            };
            s.synchronize()
        },
        20,
    )?;

    let a = s.clone_dtoh(&w16_old)?;
    let b = s.clone_dtoh(&w16_new)?;
    let mut max_abs = 0.0f32;
    for i in 0..n * k {
        max_abs = max_abs.max((a[i].to_f32() - b[i].to_f32()).abs());
    }
    println!(
        "dequant only  k={k:6} n={n:6}  old {t_old:7.4}ms  vec {t_new:7.4}ms  speedup {:.2}x  \
         max_abs_diff {max_abs:.3e}",
        t_old / t_new,
    );
    Ok(())
}

fn main() -> Result<()> {
    let dev = Device::new(0)?;
    let quant = format!("{COMMON}\n{QUANT}");
    let mut rng = Rng(0xD1B54A32D192ED03);

    dequant_only(&dev, &quant, 5120, 10240, &mut rng)?;
    dequant_only(&dev, &quant, 5120, 6144, &mut rng)?;
    bench(&dev, &quant, "attn_qkv", 5120, 10240, &mut rng)?;
    bench(&dev, &quant, "attn_gate/z", 5120, 6144, &mut rng)?;
    Ok(())
}
