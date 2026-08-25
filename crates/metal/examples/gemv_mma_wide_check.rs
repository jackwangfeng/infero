//! Correctness and speed of `gemv_mma_wide_q4_K` against the deployed
//! `gemv_mma_q4_K`.
//!
//! `gemv_mma_q4_K` covers eight tokens a threadgroup; a wider batch is
//! `grid.y` threadgroups, each re-fetching and re-decoding the same weight
//! row from scratch. `gemv_mma_wide_q4_K` decodes a row once and MMAs it
//! against up to eight token-tiles before moving on, so it should compute
//! the identical answer -- same dequantisation, same accumulation order
//! within a tile -- faster past eight tokens. This checks both: the wide
//! kernel's output against the deployed one's, bit-for-bit tolerance aside,
//! and the wall time each takes at token counts spanning the gap between
//! `TUILI_MMA_MIN` (8) and `GEMM_THRESHOLD_DEFAULT` (48) where prefill
//! currently lives.
//!
//!     cargo run --release -p tuili-metal --example gemv_mma_wide_check

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

// A tiny xorshift so the fixture is deterministic without pulling in `rand`.
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

    // A real FFN shape: k = d_model, n = ffn width.
    let k = 5120usize;
    let n = 17408usize;
    let nb = k / 256;
    let block_bytes = 144usize; // block_q4_K: 2 halfs + 12 + 128
    let w_bytes = n * nb * block_bytes;

    let mut rng = Rng(0x9E3779B97F4A7C15);
    // Structured-random blocks, not fully random bytes: `d`/`dmin` are `half`,
    // and a random bit pattern there is a live NaN/Inf risk that would make a
    // real mismatch invisible (NaN != NaN, so a diff against it reads as
    // "no worse"). Scales and quant nibbles have no such trap -- every byte
    // value is a valid 6-bit or 4-bit field -- so those stay random.
    let n_blocks = n * nb;
    let mut w_bytes_host = vec![0u8; w_bytes];
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

    for &n_tokens in &[8usize, 16, 32, 48, 53, 64] {
        let x_host: Vec<f32> = (0..n_tokens * k).map(|_| rng.next_f32()).collect();
        let d_x = s.clone_htod(&x_host)?;
        let mut out_old = s.alloc_zeros::<f32>(n * n_tokens)?;
        let mut out_new = s.alloc_zeros::<f32>(n * n_tokens)?;

        let (ki, ni, nti) = (k as i32, n as i32, n_tokens as i32);

        // Old: grid.y = ceil(n_tokens / 8), one weight re-decode a group.
        let f_old = dev.kernels().get("quant", &quant, "gemv_mma_q4_K")?;
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

        // New: grid.y = ceil(n_tokens / 64), up to eight tiles decoded once.
        let f_new = dev.kernels().get("quant", &quant, "gemv_mma_wide_q4_K")?;
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
                        grid_dim: ((n as u32).div_ceil(8), (n_tokens as u32).div_ceil(64), 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    })?
                };
                s.synchronize()
            },
            10,
        )?;

        let a = s.clone_dtoh(&out_old)?;
        let b = s.clone_dtoh(&out_new)?;
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut worst = (0usize, 0usize);
        for row in 0..n {
            for t in 0..n_tokens {
                let idx = t * n + row;
                let (va, vb) = (a[idx], b[idx]);
                let ad = (va - vb).abs();
                if ad > max_abs {
                    max_abs = ad;
                    worst = (row, t);
                }
                let rel = ad / va.abs().max(1e-6);
                max_rel = max_rel.max(rel);
            }
        }
        println!(
            "n_tokens={n_tokens:3}  old {t_old:7.3}ms  new {t_new:7.3}ms  speedup {:.2}x  \
             max_abs_diff {max_abs:.3e}  max_rel_diff {max_rel:.3e}  worst=(row={},tok={})",
            t_old / t_new,
            worst.0,
            worst.1
        );
    }
    Ok(())
}
