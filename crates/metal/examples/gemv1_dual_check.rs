//! Correctness and speed of `gemv1_dual_q4_K` (one dispatch across two
//! separate weight matrices) against two separate `gemv1_q4_K` calls, the
//! current decode path -- swept across the three real gate/up-shaped,
//! qkv/z-shaped, and wk/wv-shaped pairs this session considered fusing.
//!
//! Born from a constraint discovered trying to fuse gate/up the way
//! `in_proj_ba`/`w_kv` already do: `stacked2_gguf`'s physical byte
//! concatenation costs real VRAM (6.4 GiB for gate+up across every layer,
//! confirmed to exhaust this machine's free VRAM and corrupt the KV pool --
//! see the doc comments on `in_proj_qz` and `w_gate_up` in weights.rs).
//! `gemv1_dual_q4_K` gets the same one-launch-instead-of-two win without
//! moving a single byte: it takes both matrices' existing pointers and
//! picks between them by row index inside one dispatch.
//!
//! Not worth shipping anywhere tried, though the picture is noisier than a
//! single run of any of these suggests -- rerun several times each:
//! gate/up (17408 + 17408 rows) came back 1.01-1.19x; qkv/z-shaped
//! (10240 + 6144) 0.92-1.17x, essentially noise around parity; wk/wv-shaped
//! (1024 + 1024, the one pair this session actually fused, via
//! `stacked2_gguf`'s VRAM-costing copy) 0.48-0.96x, never clearly ahead and
//! sometimes a real regression. The general shape holds even though the
//! exact numbers do not: bigger matrices see a small, inconsistent win
//! (streaming already dwarfs one launch's overhead, so removing a launch
//! buys little); the smallest pair -- the one case a launch-count cut
//! should matter most for -- never wins and sometimes loses outright, on a
//! machine too noisy at this scale to say why with confidence. A VRAM-free
//! fusion mechanism that does not carry `stacked2_gguf`'s memory cost
//! sounds like it should be a strictly better version of `w_kv`; measured,
//! it is not one anywhere tried, and specifically not at `w_kv`'s own
//! shape. Not wired into `Kernels::gemv` on the strength of this.
//!
//!     cargo run --release -p tuili-metal --example gemv1_dual_check

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

fn rand_q4k_bytes(rng: &mut Rng, n: usize, nb: usize) -> Vec<u8> {
    let block_bytes = 144usize;
    let mut out = vec![0u8; n * nb * block_bytes];
    for blk in 0..n * nb {
        let base = blk * block_bytes;
        let d = half::f16::from_f32(0.01 + rng.next_f32().abs() * 0.05);
        let dmin = half::f16::from_f32(0.01 + rng.next_f32().abs() * 0.05);
        out[base..base + 2].copy_from_slice(&d.to_bits().to_le_bytes());
        out[base + 2..base + 4].copy_from_slice(&dmin.to_bits().to_le_bytes());
        for byte in &mut out[base + 4..base + block_bytes] {
            *byte = (rng.next_f32().to_bits() & 0xFF) as u8;
        }
    }
    out
}

fn bench(dev: &Device, quant: &str, label: &str, k: usize, n1: usize, n2: usize, rng: &mut Rng) -> Result<()> {
    let s = dev.stream();
    let nb = k / 256;

    let w1_bytes = rand_q4k_bytes(rng, n1, nb);
    let w2_bytes = rand_q4k_bytes(rng, n2, nb);
    let d_w1 = s.clone_htod(&w1_bytes)?;
    let d_w2 = s.clone_htod(&w2_bytes)?;
    let x_host: Vec<f32> = (0..k).map(|_| rng.next_f32()).collect();
    let d_x = s.clone_htod(&x_host)?;

    let mut out1_old = s.alloc_zeros::<f32>(n1)?;
    let mut out2_old = s.alloc_zeros::<f32>(n2)?;
    let mut out1_new = s.alloc_zeros::<f32>(n1)?;
    let mut out2_new = s.alloc_zeros::<f32>(n2)?;

    const GEMV_BLOCK_MAX: u32 = 128;
    let block = ((nb * 8) as u32).next_multiple_of(32).clamp(32, GEMV_BLOCK_MAX);
    let (ki, n1i, n2i, one) = (k as i32, n1 as i32, n2 as i32, 1i32);

    let f_old = dev.kernels().get("quant", quant, "gemv1_q4_K")?;
    let t_old = ms(
        || {
            {
                let mut b = s.launch_builder(&f_old);
                b.arg(&out1_old.as_view_mut())
                    .arg(&d_w1.as_view())
                    .arg(&d_x.as_view())
                    .arg(&ki)
                    .arg(&n1i)
                    .arg(&one);
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: (n1 as u32, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    })?
                };
            }
            {
                let mut b = s.launch_builder(&f_old);
                b.arg(&out2_old.as_view_mut())
                    .arg(&d_w2.as_view())
                    .arg(&d_x.as_view())
                    .arg(&ki)
                    .arg(&n2i)
                    .arg(&one);
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: (n2 as u32, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    })?
                };
            }
            s.synchronize()
        },
        30,
    )?;

    let f_new = dev.kernels().get("quant", quant, "gemv1_dual_q4_K")?;
    let t_new = ms(
        || {
            let mut b = s.launch_builder(&f_new);
            b.arg(&out1_new.as_view_mut())
                .arg(&out2_new.as_view_mut())
                .arg(&d_w1.as_view())
                .arg(&d_w2.as_view())
                .arg(&d_x.as_view())
                .arg(&ki)
                .arg(&n1i)
                .arg(&n2i);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: ((n1 + n2) as u32, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                })?
            };
            s.synchronize()
        },
        30,
    )?;

    let a1 = s.clone_dtoh(&out1_old)?;
    let a2 = s.clone_dtoh(&out2_old)?;
    let b1 = s.clone_dtoh(&out1_new)?;
    let b2 = s.clone_dtoh(&out2_new)?;
    let mut max_abs = 0.0f32;
    for i in 0..n1 {
        max_abs = max_abs.max((a1[i] - b1[i]).abs());
    }
    for i in 0..n2 {
        max_abs = max_abs.max((a2[i] - b2[i]).abs());
    }
    println!(
        "{label:24} n1={n1:6} n2={n2:6}  two-calls {t_old:7.4}ms  dual {t_new:7.4}ms  \
         speedup {:.2}x  max_abs_diff {max_abs:.3e}",
        t_old / t_new,
    );
    Ok(())
}

fn main() -> Result<()> {
    let dev = Device::new(0)?;
    let quant = format!("{COMMON}\n{QUANT}");
    let mut rng = Rng(0xBF58476D1CE4E5B9);

    bench(&dev, &quant, "gate/up shape", 5120, 17408, 17408, &mut rng)?;
    bench(&dev, &quant, "qkv/z shape", 5120, 10240, 6144, &mut rng)?;
    bench(&dev, &quant, "wk/wv shape", 5120, 1024, 1024, &mut rng)?;
    Ok(())
}
