//! Correctness and speed of `embed_row_q4_K` against the deployed
//! `gather_rows_q4_K` (the generic one-thread-a-element `GATHER_KERNEL`
//! macro) for the token embedding table.
//!
//! `embed_row_q4_K` already exists in quant.metal -- its doc comment says
//! why it should exist ("a batch of one token wants exactly one of its
//! 248320 rows") -- but nothing in `Kernels::gather_rows` ever calls it;
//! `gather_rows` always dispatches `gather_rows_{ty.suffix()}`, the generic
//! macro, regardless of `ty`. For Q4_K that macro re-reads and re-unpacks
//! `q4k_scale_min` fresh for every one of a 32-element group's elements --
//! the same redundancy `dequant_q4_K_f16_vec` (6acfeb2) fixed for the whole-
//! matrix dequant, just never ported to the embedding gather.
//!
//!     cargo run --release -p infero-metal --example embed_row_check

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
    fn next_u32(&mut self, bound: u32) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % bound as u64) as u32
    }
}

fn main() -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();
    let quant = format!("{COMMON}\n{QUANT}");

    // The real embedding table shape: d_model = 5120, vocab = 248320.
    let k = 5120usize;
    let vocab = 248320usize;
    let nb = k / 256;
    let block_bytes = 144usize;
    let n_blocks = vocab * nb;

    let mut rng = Rng(0x9E3779B97F4A7C15);
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

    for &n_tokens in &[1usize, 8, 63, 128] {
        let rows_host: Vec<i32> = (0..n_tokens).map(|_| rng.next_u32(vocab as u32) as i32).collect();
        let d_rows = s.clone_htod(&rows_host)?;
        let mut out_old = s.alloc_zeros::<f32>(n_tokens * k)?;
        let mut out_new = s.alloc_zeros::<f32>(n_tokens * k)?;
        let ki = k as i32;

        const ELEMENTWISE_BLOCK: u32 = 256;
        let f_old = dev.kernels().get("quant", &quant, "gather_rows_q4_K")?;
        let t_old = ms(
            || {
                let mut b = s.launch_builder(&f_old);
                b.arg(&out_old.as_view_mut()).arg(&d_w.as_view()).arg(&d_rows.as_view()).arg(&ki);
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: ((k as u32).div_ceil(ELEMENTWISE_BLOCK), n_tokens as u32, 1),
                        block_dim: (ELEMENTWISE_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })?
                };
                s.synchronize()
            },
            20,
        )?;

        let f_new = dev.kernels().get("quant", &quant, "embed_row_q4_K")?;
        let block = 128u32;
        let t_new = ms(
            || {
                let mut b = s.launch_builder(&f_new);
                b.arg(&out_new.as_view_mut()).arg(&d_w.as_view()).arg(&d_rows.as_view()).arg(&ki);
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: (1, n_tokens as u32, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    })?
                };
                s.synchronize()
            },
            20,
        )?;

        let a = s.clone_dtoh(&out_old)?;
        let b = s.clone_dtoh(&out_new)?;
        let mut max_abs = 0.0f32;
        for i in 0..n_tokens * k {
            max_abs = max_abs.max((a[i] - b[i]).abs());
        }
        println!(
            "n_tokens={n_tokens:4}  old {t_old:7.4}ms  new {t_new:7.4}ms  speedup {:.2}x  max_abs_diff {max_abs:.3e}",
            t_old / t_new,
        );
    }
    Ok(())
}
