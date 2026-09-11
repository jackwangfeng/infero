//! Q5_K vs Q4_K throughput at the real shape the new format actually landed
//! on: `Qwen3.8-27B-Q4_K_M.gguf`'s own `attn_k`/`attn_v` tensors, which
//! llama.cpp's Q4_K_M strategy bumped to Q5_K (K=d_model=5120,
//! N=n_kv_heads*d_head=4*256=1024 -- this is the exact tensor that used to
//! loud-fail with "weight type Q5_K is not implemented" before this format
//! was added). Q5_K is `Q4_K` plus one extra bit per weight (a `qh` byte
//! array and a few more ALU ops per dot-product term), so the question this
//! answers is how much that extra bit actually costs against the format it
//! was inserted right next to in every dispatch table.
//!
//!   cargo run --release -p infero-kernels --example q5k_vs_q4k_bench

use std::time::Instant;

use anyhow::Result;
use infero_gpu::Device;
use infero_kernels::{Kernels, WeightType};

const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12;

fn pseudo_random_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        })
        .collect()
}

fn pseudo_random_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect()
}

/// One real `block_q4_K` (144 bytes: `d`, `dmin`, `scales[12]`, `qs[128]`).
fn q4_k_row_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_blocks * 144);
    for i in 0..n_blocks {
        out.extend_from_slice(&half::f16::from_f32(0.04).to_le_bytes());
        out.extend_from_slice(&half::f16::from_f32(0.015).to_le_bytes());
        out.extend_from_slice(&pseudo_random_bytes(K_SCALE_SIZE, seed + i as u64 * 7 + 1));
        out.extend_from_slice(&pseudo_random_bytes(QK_K / 2, seed + i as u64 * 7 + 2));
    }
    out
}

/// One real `block_q5_K` (176 bytes: `d`, `dmin`, `scales[12]`, `qh[32]`,
/// `qs[128]`), same layout `q5_k.rs`'s own `BlockQ5K` verifies.
fn q5_k_row_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_blocks * 176);
    for i in 0..n_blocks {
        out.extend_from_slice(&half::f16::from_f32(0.04).to_le_bytes());
        out.extend_from_slice(&half::f16::from_f32(0.015).to_le_bytes());
        out.extend_from_slice(&pseudo_random_bytes(K_SCALE_SIZE, seed + i as u64 * 7 + 1));
        out.extend_from_slice(&pseudo_random_bytes(QK_K / 8, seed + i as u64 * 7 + 3));
        out.extend_from_slice(&pseudo_random_bytes(QK_K / 2, seed + i as u64 * 7 + 2));
    }
    out
}

fn time_ms<F: FnMut() -> Result<()>>(k: &Kernels, reps: usize, mut f: F) -> Result<f64> {
    for _ in 0..5 {
        f()?;
    }
    k.device().synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        f()?;
    }
    // Without this, `elapsed()` measures async launch-queueing overhead, not
    // real GPU execution time -- and once enough unsynchronized launches
    // queue up, CPU-side launch calls start blocking on driver-level
    // backpressure, which looks exactly like a kernel-side cliff but is not
    // one (caught by isolating one shape with a per-call sync in
    // `q5k_t16_isolate.rs`, which found no such cliff at all).
    k.device().synchronize()?;
    Ok(t0.elapsed().as_secs_f64() * 1000.0 / reps as f64)
}

fn main() -> Result<()> {
    let k = Kernels::new(Device::new(0)?);
    let stream = k.device().stream().clone();

    // The real `attn_k`/`attn_v` shape in Qwen3.8-27B: K=d_model=5120,
    // N=n_kv_heads*d_head=4*256=1024.
    let (kk, n) = (5120usize, 1024usize);
    let nb_per_row = kk / QK_K;
    let reps = 200;

    println!("shape: K={kk} N={n} (real Qwen3.8-27B attn_k/attn_v), {reps} reps, warmup 5\n");

    // --- decode (n_tokens=1): gemv and single-row mmvq ---
    let d_q4 = stream.clone_htod(&q4_k_row_bytes(nb_per_row * n, 0xA5A5))?;
    let d_q5 = stream.clone_htod(&q5_k_row_bytes(nb_per_row * n, 0xA5A5))?;
    let x = pseudo_random_f32(kk, 0x1234);
    let d_x = stream.clone_htod(&x)?;
    let mut d_out = stream.alloc_zeros::<f32>(n)?;

    let ms_gemv_q4 = time_ms(&k, reps, || {
        k.gemv(&mut d_out.as_view_mut(), &d_q4.as_view(), WeightType::Q4K, &d_x.as_view(), kk, n, 1)
    })?;
    let ms_gemv_q5 = time_ms(&k, reps, || {
        k.gemv(&mut d_out.as_view_mut(), &d_q5.as_view(), WeightType::Q5K, &d_x.as_view(), kk, n, 1)
    })?;
    println!(
        "gemv   (n_tokens=1): Q4_K {ms_gemv_q4:.4} ms   Q5_K {ms_gemv_q5:.4} ms   ({:+.1}%)",
        100.0 * (ms_gemv_q5 / ms_gemv_q4 - 1.0)
    );

    if k.device().caps().int_tensor_gemm {
        let bytes = Kernels::q8_1_bytes(kk);
        let mut d_q8 = stream.alloc_zeros::<u8>(bytes)?;
        k.quantize_q8_1(&mut d_q8.as_view_mut(), &d_x.as_view(), kk)?;

        let ms_mmvq_q4 = time_ms(&k, reps, || {
            k.mmvq(&mut d_out.as_view_mut(), &d_q4.as_view(), WeightType::Q4K, &d_q8.as_view(), kk, n)
        })?;
        let ms_mmvq_q5 = time_ms(&k, reps, || {
            k.mmvq(&mut d_out.as_view_mut(), &d_q5.as_view(), WeightType::Q5K, &d_q8.as_view(), kk, n)
        })?;
        println!(
            "mmvq   (n_tokens=1): Q4_K {ms_mmvq_q4:.4} ms   Q5_K {ms_mmvq_q5:.4} ms   ({:+.1}%)",
            100.0 * (ms_mmvq_q5 / ms_mmvq_q4 - 1.0)
        );

        // --- batch decode (n_tokens=16, this server's own default max_seqs shape) ---
        for &n_tokens in &[8usize, 16, 32] {
            let xb = pseudo_random_f32(n_tokens * kk, 0x5678 + n_tokens as u64);
            let d_xb = stream.clone_htod(&xb)?;
            let bytes_b = Kernels::q8_1_bytes(n_tokens * kk);
            let mut d_q8b = stream.alloc_zeros::<u8>(bytes_b)?;
            k.quantize_q8_1(&mut d_q8b.as_view_mut(), &d_xb.as_view(), n_tokens * kk)?;
            let mut d_outb = stream.alloc_zeros::<f32>(n * n_tokens)?;

            let ms_batch_q4 = time_ms(&k, reps, || {
                k.mmvq_batch(&mut d_outb.as_view_mut(), &d_q4.as_view(), WeightType::Q4K, &d_q8b.as_view(), kk, n, n_tokens)
            })?;
            let ms_batch_q5 = time_ms(&k, reps, || {
                k.mmvq_batch(&mut d_outb.as_view_mut(), &d_q5.as_view(), WeightType::Q5K, &d_q8b.as_view(), kk, n, n_tokens)
            })?;
            println!(
                "mmvq_batch (n_tokens={n_tokens:2}): Q4_K {ms_batch_q4:.4} ms   Q5_K {ms_batch_q5:.4} ms   ({:+.1}%)",
                100.0 * (ms_batch_q5 / ms_batch_q4 - 1.0)
            );
        }
    } else {
        println!("skipping mmvq/mmvq_batch: no int8 tensor-core dp4a support on this device");
    }

    Ok(())
}
