//! `tq_attn_decode_f32` in isolation, for `ncu` to profile without a live
//! server's CUDA-graph-captured decode step in the way.
//!
//!     cargo run --release -p tuili-kernels --example tq_attn_decode_bench
//!
//! Shapes match the 27B: 24 query heads, 4 KV heads (group 6), 256-wide
//! heads. Data is pseudo-random, not real TurboQuant codes — this measures
//! the kernel's timing and occupancy, not its numerics, which the model-level
//! tests already cover.

use anyhow::Result;
use half::f16;
use tuili_cuda::Device;
use tuili_kernels::{AttnDims, BatchLayout, Kernels};

const N_HEADS: usize = 24;
const N_KV_HEADS: usize = 4;
const D_HEAD: usize = 256;
const N_SLOTS: usize = 8192;
const KV_LEN: usize = 2048;
const K_BITS: u8 = 4;
const V_BITS: u8 = 4;

fn pseudo_random_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 56) as u8
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

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let dev = Device::new(0)?;
    println!("device: {} (sm_{}, {} SMs)", dev.name(), dev.arch(), dev.sm_count());
    let k = Kernels::new(dev.clone());
    let stream = dev.stream().clone();

    let group = N_HEADS / N_KV_HEADS;
    let per_byte_k = 8 / K_BITS as usize;
    let per_byte_v = 8 / V_BITS as usize;

    let q_rot = pseudo_random_f32(N_HEADS * D_HEAD, 1);
    let q_qjl = pseudo_random_f32(N_HEADS * D_HEAD, 2);
    let k_codes = pseudo_random_bytes(N_KV_HEADS * N_SLOTS * D_HEAD / per_byte_k, 3);
    let k_signs = pseudo_random_bytes(N_KV_HEADS * N_SLOTS * D_HEAD / 8, 4);
    let k_scale: Vec<f16> = pseudo_random_f32(N_KV_HEADS * N_SLOTS, 5)
        .into_iter()
        .map(|v| f16::from_f32(v.abs() + 0.01))
        .collect();
    let k_gamma: Vec<f16> = pseudo_random_f32(N_KV_HEADS * N_SLOTS, 6)
        .into_iter()
        .map(|v| f16::from_f32(v.abs() + 0.01))
        .collect();
    let v_codes = pseudo_random_bytes(N_KV_HEADS * N_SLOTS * D_HEAD / per_byte_v, 7);
    let v_scale: Vec<f16> = pseudo_random_f32(N_KV_HEADS * N_SLOTS, 8)
        .into_iter()
        .map(|v| f16::from_f32(v.abs() + 0.01))
        .collect();
    let k_levels = pseudo_random_f32(1 << K_BITS, 9);
    let v_levels = pseudo_random_f32(1 << V_BITS, 10);

    let seq_of = vec![0i32];
    let positions = vec![(KV_LEN - 1) as i32];
    let slot_table: Vec<i32> = (0..KV_LEN as i32).collect();

    let d_q_rot = stream.clone_htod(&q_rot)?;
    let d_q_qjl = stream.clone_htod(&q_qjl)?;
    let d_k_codes = stream.clone_htod(&k_codes)?;
    let d_k_signs = stream.clone_htod(&k_signs)?;
    let d_k_scale = stream.clone_htod(&k_scale)?;
    let d_k_gamma = stream.clone_htod(&k_gamma)?;
    let d_v_codes = stream.clone_htod(&v_codes)?;
    let d_v_scale = stream.clone_htod(&v_scale)?;
    let d_k_levels = stream.clone_htod(&k_levels)?;
    let d_v_levels = stream.clone_htod(&v_levels)?;
    let d_seq_of = stream.clone_htod(&seq_of)?;
    let d_positions = stream.clone_htod(&positions)?;
    let d_slot_table = stream.clone_htod(&slot_table)?;
    let mut d_out = stream.alloc_zeros::<f32>(N_HEADS * D_HEAD)?;
    let mut d_partial =
        stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(N_HEADS, D_HEAD, 1))?;

    let dims = AttnDims {
        n_heads: N_HEADS,
        n_kv_heads: N_KV_HEADS,
        d_head: D_HEAD,
        n_slots: N_SLOTS,
        n_tokens: 1,
    };
    let batch = BatchLayout {
        seq_of: &d_seq_of.as_view(),
        positions: &d_positions.as_view(),
        slot_table: &d_slot_table.as_view(),
        table_stride: KV_LEN,
    };

    // Warm-up, then a timed loop — `dev.profile()` needs `TUILI_PROFILE`/
    // `TUILI_METAL_PROFILE`-style opt-in on this backend too, so this just
    // brackets with a host clock and a synchronize instead.
    for _ in 0..3 {
        k.tq_attn_decode(
            &mut d_out.as_view_mut(),
            &d_q_rot.as_view(),
            &d_q_qjl.as_view(),
            &d_k_codes.as_view(),
            &d_k_signs.as_view(),
            &d_k_scale.as_view(),
            &d_k_gamma.as_view(),
            &d_v_codes.as_view(),
            &d_v_scale.as_view(),
            batch,
            &d_k_levels.as_view(),
            K_BITS,
            &d_v_levels.as_view(),
            V_BITS,
            dims,
            KV_LEN,
            1.0 / (D_HEAD as f32).sqrt(),
            1.0,
            &mut d_partial.as_view_mut(),
        )?;
    }
    stream.synchronize()?;

    let iters = 200;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        k.tq_attn_decode(
            &mut d_out.as_view_mut(),
            &d_q_rot.as_view(),
            &d_q_qjl.as_view(),
            &d_k_codes.as_view(),
            &d_k_signs.as_view(),
            &d_k_scale.as_view(),
            &d_k_gamma.as_view(),
            &d_v_codes.as_view(),
            &d_v_scale.as_view(),
            batch,
            &d_k_levels.as_view(),
            K_BITS,
            &d_v_levels.as_view(),
            V_BITS,
            dims,
            KV_LEN,
            1.0 / (D_HEAD as f32).sqrt(),
            1.0,
            &mut d_partial.as_view_mut(),
        )?;
    }
    stream.synchronize()?;
    let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("tq_attn_decode_f32: {us:.2} us a call, kv_len={KV_LEN}, group={group}");
    Ok(())
}
