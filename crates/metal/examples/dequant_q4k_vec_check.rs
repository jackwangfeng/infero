//! Correctness and speed of `dequant_q4_K_f16_vec` against the deployed
//! `dequant_q4_K_f16`.
//!
//! The deployed kernel is one thread a element: every one of a 32-wide
//! group's threads unpacks the same `q4k_scale_min` and reads the same
//! block's `d`/`dmin` independently, and writes one `half` at a time. The
//! vector version gives one thread the whole group -- unpack once, two
//! vectorised `uint4` reads, four `half4` stores -- the same restructuring
//! `GEMV_BODY_Q4_K` already uses for the mat-vec. This checks the output
//! matches (it is the same arithmetic, so any difference is a real bug) and
//! times both across the tensor sizes dequant_to_f16 actually runs at during
//! prefill: `profile()` reports it at ~496 launches for a single 53-token
//! prompt, meaning every large Q4_K matrix in the model pays this once a
//! prefill call regardless of how many tokens that call carries.
//!
//!     cargo run --release -p tuili-metal --example dequant_q4k_vec_check

use anyhow::Result;
use tuili_metal::{Device, LaunchConfig};

const COMMON: &str = include_str!("../../kernels/src/msl/common.metal");
const QUANT: &str = include_str!("../../kernels/src/msl/quant.metal");
const ELEMENTWISE_BLOCK: u32 = 256;

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

    let block_bytes = 144usize;
    let mut rng = Rng(0xC0FFEE1234567);

    // Real prefill shapes: ffn_gate/up (5120 x 17408) and ffn_down (17408 x
    // 5120) are the two distinct sizes the 27B's dequant calls actually hit.
    for (label, rows, k) in [("ffn_gate/up", 17408usize, 5120usize), ("ffn_down", 5120, 17408)] {
        let nb_per_row = k / 256;
        let n_elements = rows * k;
        let n_blocks = rows * nb_per_row;
        let w_bytes = n_blocks * block_bytes;

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
        let mut out_old = s.alloc_zeros::<half::f16>(n_elements)?;
        let mut out_new = s.alloc_zeros::<half::f16>(n_elements)?;
        let n_u64 = n_elements as u64;

        let f_old = dev.kernels().get("quant", &quant, "dequant_q4_K_f16")?;
        let t_old = ms(
            || {
                let mut b = s.launch_builder(&f_old);
                b.arg(&out_old.as_view_mut()).arg(&d_w.as_view()).arg(&n_u64);
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: ((n_elements as u32).div_ceil(ELEMENTWISE_BLOCK), 1, 1),
                        block_dim: (ELEMENTWISE_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })?
                };
                s.synchronize()
            },
            10,
        )?;

        let f_new = dev.kernels().get("quant", &quant, "dequant_q4_K_f16_vec")?;
        let n_groups = (n_elements / 32) as u32;
        let t_new = ms(
            || {
                let mut b = s.launch_builder(&f_new);
                b.arg(&out_new.as_view_mut()).arg(&d_w.as_view()).arg(&n_u64);
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: (n_groups.div_ceil(ELEMENTWISE_BLOCK), 1, 1),
                        block_dim: (ELEMENTWISE_BLOCK, 1, 1),
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
        let mut worst = 0usize;
        for i in 0..n_elements {
            let (va, vb) = (a[i].to_f32(), b[i].to_f32());
            let ad = (va - vb).abs();
            if ad > max_abs {
                max_abs = ad;
                worst = i;
            }
        }
        let bytes = (w_bytes + n_elements * 2) as f64;
        println!(
            "{label:12} n={n_elements:10}  old {t_old:7.3}ms {:6.1}GB/s  new {t_new:7.3}ms {:6.1}GB/s  speedup {:.2}x  max_abs_diff {max_abs:.3e} @ {worst}",
            bytes / (t_old / 1e3) / 1e9,
            bytes / (t_new / 1e3) / 1e9,
            t_old / t_new,
        );
    }
    Ok(())
}
