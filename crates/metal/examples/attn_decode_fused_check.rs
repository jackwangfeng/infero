//! Correctness and speed of `attn_decode_fused_f32` (scores + softmax +
//! output fused into one dispatch, threadgroup-memory-staged) against the
//! three separate kernels it replaces, and against an f64 CPU reference.
//!
//! `decode_attention()`'s own comment says the CUDA fused kernel
//! (`attn_decode_gqa_f32`) is "worth porting, and not worth blocking on" --
//! this is not that kernel (no register-pipelined K/V prefetch, no chunked
//! occupancy tuning, no MMA variant), it is the simplest fusion that
//! removes the score row's device-memory round trip by reusing each of the
//! three kernels' exact math and order rather than re-deriving flash
//! attention's algebra. See the kernel's own doc comment in ops.metal.
//!
//! Two references rather than one: the three-kernel path is what production
//! actually runs today, so matching it byte-for-bit is the real bar: an f64
//! CPU reference catches a wrong *algorithm*, not the specific rounding this
//! GPU path already commits to.
//!
//!     cargo run --release -p tuili-metal --example attn_decode_fused_check

use anyhow::Result;
use tuili_metal::{Device, LaunchConfig};

const COMMON: &str = include_str!("../../kernels/src/msl/common.metal");
const OPS: &str = include_str!("../../kernels/src/msl/ops.metal");

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

/// f64 attention over one token, every head, no masking beyond `kv_len`
/// (every key is real -- this test's `positions[0] = kv_len - 1`).
fn cpu_reference(
    q: &[f32],
    k: &[half::f16],
    v: &[half::f16],
    n_heads: usize,
    n_kv_heads: usize,
    d_head: usize,
    kv_len: usize,
    scale: f32,
) -> Vec<f32> {
    let group = n_heads / n_kv_heads;
    let mut out = vec![0.0f32; n_heads * d_head];
    for h in 0..n_heads {
        let kv_head = h / group;
        let qr = &q[h * d_head..(h + 1) * d_head];
        let mut scores = vec![0.0f64; kv_len];
        for j in 0..kv_len {
            let kr = &k[(kv_head * kv_len + j) * d_head..(kv_head * kv_len + j + 1) * d_head];
            let mut acc = 0.0f64;
            for i in 0..d_head {
                acc += qr[i] as f64 * kr[i].to_f64();
            }
            scores[j] = acc * scale as f64;
        }
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
        let sum: f64 = exps.iter().sum();
        let weights: Vec<f64> = exps.iter().map(|e| e / sum).collect();
        for i in 0..d_head {
            let mut acc = 0.0f64;
            for j in 0..kv_len {
                let vr = v[(kv_head * kv_len + j) * d_head + i].to_f64();
                acc += weights[j] * vr;
            }
            out[h * d_head + i] = acc as f32;
        }
    }
    out
}

fn main() -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();
    let common = format!("{COMMON}\n{OPS}");

    let n_heads = 24usize;
    let n_kv_heads = 4usize;
    let d_head = 256usize;
    let group = n_heads / n_kv_heads;
    let n_slots = 300usize;
    let scale = 1.0f32 / (d_head as f32).sqrt();

    let mut rng = Rng(0x9E3779B97F4A7C15);

    for &kv_len in &[16usize, 63, 128, 256] {
        let q_host: Vec<f32> = (0..n_heads * d_head).map(|_| rng.next_f32()).collect();
        let k_host: Vec<half::f16> = (0..n_kv_heads * n_slots * d_head)
            .map(|_| half::f16::from_f32(rng.next_f32()))
            .collect();
        let v_host: Vec<half::f16> = (0..n_kv_heads * n_slots * d_head)
            .map(|_| half::f16::from_f32(rng.next_f32()))
            .collect();
        // One sequence, one token, identity slot mapping: slot_table[j] = j.
        let slot_table_host: Vec<i32> = (0..kv_len as i32).collect();
        let seq_of_host = [0i32];
        let positions_host = [(kv_len - 1) as i32];

        let d_q = s.clone_htod(&q_host)?;
        let d_k = s.clone_htod(&k_host)?;
        let d_v = s.clone_htod(&v_host)?;
        let d_slot_table = s.clone_htod(&slot_table_host)?;
        let d_seq_of = s.clone_htod(&seq_of_host)?;
        let d_positions = s.clone_htod(&positions_host)?;

        // ---- three-kernel path ----
        let mut scores = s.alloc_zeros::<f32>(n_heads * kv_len)?;
        let mut out_old = s.alloc_zeros::<f32>(n_heads * d_head)?;
        let (table_stride, nh, nkvh, dh, ns, kl) = (
            kv_len as i32,
            n_heads as i32,
            n_kv_heads as i32,
            d_head as i32,
            n_slots as i32,
            kv_len as i32,
        );

        let f_scores = dev.kernels().get("ops", &common, "attn_scores_f32")?;
        let f_softmax = dev.kernels().get("ops", &common, "attn_softmax_f32")?;
        let f_output = dev.kernels().get("ops", &common, "attn_output_f32")?;
        let t_old = ms(
            || {
                {
                    let mut b = s.launch_builder(&f_scores);
                    b.arg(&scores.as_view_mut())
                        .arg(&d_q.as_view())
                        .arg(&d_k.as_view())
                        .arg(&d_seq_of.as_view())
                        .arg(&d_positions.as_view())
                        .arg(&d_slot_table.as_view())
                        .arg(&table_stride)
                        .arg(&nh)
                        .arg(&nkvh)
                        .arg(&dh)
                        .arg(&ns)
                        .arg(&kl)
                        .arg(&scale);
                    unsafe {
                        b.launch(LaunchConfig {
                            grid_dim: ((kv_len as u32).div_ceil(4), n_heads as u32, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        })?
                    };
                }
                {
                    let mut b = s.launch_builder(&f_softmax);
                    b.arg(&scores.as_view_mut()).arg(&kl);
                    unsafe {
                        b.launch(LaunchConfig {
                            grid_dim: (n_heads as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })?
                    };
                }
                {
                    let mut b = s.launch_builder(&f_output);
                    b.arg(&out_old.as_view_mut())
                        .arg(&scores.as_view())
                        .arg(&d_v.as_view())
                        .arg(&d_seq_of.as_view())
                        .arg(&d_positions.as_view())
                        .arg(&d_slot_table.as_view())
                        .arg(&table_stride)
                        .arg(&nh)
                        .arg(&nkvh)
                        .arg(&dh)
                        .arg(&ns)
                        .arg(&kl);
                    unsafe {
                        b.launch(LaunchConfig {
                            grid_dim: (n_heads as u32, 1, 1),
                            block_dim: ((d_head as u32).next_multiple_of(32), 1, 1),
                            shared_mem_bytes: 0,
                        })?
                    };
                }
                s.synchronize()
            },
            30,
        )?;

        // ---- fused kernel ----
        let mut out_new = s.alloc_zeros::<f32>(n_heads * d_head)?;
        let f_fused = dev.kernels().get("ops", &common, "attn_decode_fused_f32")?;
        // Phase 1 wants roughly one SIMD group an eight keys so it does not
        // serialize kv_len across a fixed, small simdgroup count; phase 3
        // wants at least d_head threads. Whichever is wider, capped at 1024.
        let sg_for_scores = (kv_len as u32).div_ceil(8).max(1);
        let block = (sg_for_scores * 32)
            .max((d_head as u32).next_multiple_of(32))
            .min(1024);
        let _ = group;
        let t_new = ms(
            || {
                let mut b = s.launch_builder(&f_fused);
                b.arg(&out_new.as_view_mut())
                    .arg(&d_q.as_view())
                    .arg(&d_k.as_view())
                    .arg(&d_v.as_view())
                    .arg(&d_seq_of.as_view())
                    .arg(&d_positions.as_view())
                    .arg(&d_slot_table.as_view())
                    .arg(&table_stride)
                    .arg(&nh)
                    .arg(&nkvh)
                    .arg(&dh)
                    .arg(&ns)
                    .arg(&kl)
                    .arg(&scale);
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: (n_heads as u32, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: (kv_len as u32) * 4,
                    })?
                };
                s.synchronize()
            },
            30,
        )?;

        let a = s.clone_dtoh(&out_old)?;
        let b_ = s.clone_dtoh(&out_new)?;
        let cpu = cpu_reference(&q_host, &k_host, &v_host, n_heads, n_kv_heads, d_head, kv_len, scale);
        let mut max_abs_vs_old = 0.0f32;
        let mut max_abs_vs_cpu = 0.0f32;
        for i in 0..n_heads * d_head {
            max_abs_vs_old = max_abs_vs_old.max((a[i] - b_[i]).abs());
            max_abs_vs_cpu = max_abs_vs_cpu.max((b_[i] - cpu[i]).abs());
        }
        println!(
            "kv_len={kv_len:5}  3-kernel {t_old:7.4}ms  fused {t_new:7.4}ms  speedup {:.2}x  \
             diff-vs-3kernel {max_abs_vs_old:.3e}  diff-vs-cpu-f64 {max_abs_vs_cpu:.3e}",
            t_old / t_new,
        );
    }
    Ok(())
}
