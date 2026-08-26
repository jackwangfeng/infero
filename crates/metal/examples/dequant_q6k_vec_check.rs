//! Correctness and speed of `dequant_q6_K_f16_vec` against the generic
//! one-thread-a-element `dequant_q6_K_f16`.
//!
//! Same fix as `dequant_q4k_vec_check.rs` checked for Q4_K, applied to a
//! block with no clean 32-element group: `dequant_q6_K_f16_vec` gives one
//! thread the same four elements (`l`, `l+32`, `l+64`, `l+96`) that
//! `GEMV_BODY_Q6_K` already gives one thread, instead of re-deriving `qh[l]`
//! and the nibble unpacks fresh in four separate threads.
//!
//!     cargo run --release -p tuili-metal --example dequant_q6k_vec_check

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

fn main() -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();
    let quant = format!("{COMMON}\n{QUANT}");

    // The real attn_output (o_proj) shape on this checkpoint.
    let k = 5120usize;
    let n = 6144usize;
    let nb = k / 256;
    let block_bytes = 128 + 64 + 16 + 2; // ql + qh + scales + d (half)
    let n_blocks = n * nb;

    let mut rng = Rng(0x2545F4914F6CDD1D);
    let mut w_bytes_host = vec![0u8; n_blocks * block_bytes];
    for blk in 0..n_blocks {
        let base = blk * block_bytes;
        for byte in &mut w_bytes_host[base..base + 128 + 64] {
            *byte = (rng.next_f32().to_bits() & 0xFF) as u8;
        }
        for byte in &mut w_bytes_host[base + 192..base + 192 + 16] {
            // scales are signed int8; keep them in a sane range like a real
            // checkpoint's, not the full byte range.
            *byte = ((rng.next_f32() * 32.0) as i8) as u8;
        }
        let d = half::f16::from_f32(0.01 + rng.next_f32().abs() * 0.05);
        w_bytes_host[base + 208..base + 210].copy_from_slice(&d.to_bits().to_le_bytes());
    }
    let d_w = s.clone_htod(&w_bytes_host)?;
    let n_elements = (n * k) as u32;

    let mut w16_old = s.alloc_zeros::<half::f16>(n * k)?;
    let mut w16_new = s.alloc_zeros::<half::f16>(n * k)?;

    let f_old = dev.kernels().get("quant", &quant, "dequant_q6_K_f16")?;
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

    let f_new = dev.kernels().get("quant", &quant, "dequant_q6_K_f16_vec")?;
    let t_new = ms(
        || {
            let mut b = s.launch_builder(&f_new);
            let ne = n_elements;
            b.arg(&w16_new.as_view_mut()).arg(&d_w.as_view()).arg(&ne);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: ((n_elements / 4).div_ceil(256), 1, 1),
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
