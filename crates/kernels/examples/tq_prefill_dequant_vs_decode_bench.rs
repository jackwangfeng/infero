//! `tq_attn_decode` vs. `tq_dequant_kv`+`attn_prefill_ws4`, across prefill
//! run lengths, to find the real crossover on this codebase's own kernels
//! -- see `.superpowers/sdd/2026-09-06-tq4-prefill-dequant-dispatch/`
//! and `TQ_DEQUANT_THRESHOLD`'s doc comment in `crates/model/src/lib.rs`.
//!
//! Shapes match the 27B: 24 query heads, 4 KV heads (group 6), 256-wide
//! heads, same as `tq_attn_decode_bench.rs`. Data is pseudo-random, not real
//! TurboQuant codes or a real QJL rotation -- this measures the two paths'
//! timing, not their numerics, which the model-level tests already cover.
//!
//! Both paths are measured at `qjl_scale = 1.0`, i.e. with the QJL
//! correction term's extra `O(d^2)` per-key matvec actually paid for on both
//! sides -- this plan's whole `Tq4` config always has it on, and the
//! dequant path's own cost (folding the term onto the key once, per
//! `tq_dequant_kv`'s doc comment) is exactly the cost this benchmark exists
//! to weigh against `tq_attn_decode`'s per-call cost of the same term.
//!
//! `attn_prefill_ws4` requires `INFERO_ATTN_MMA=1` in the environment
//! *before this process starts* (its gate reads the env var once into a
//! `OnceLock`), so run this as:
//!
//!     INFERO_ATTN_MMA=1 cargo run --release -p infero-kernels --example tq_prefill_dequant_vs_decode_bench

use anyhow::Result;
use half::f16;
use infero_cuda::Device;
use infero_gpu::Buf;
use infero_kernels::{AttnDims, BatchLayout, Kernels};

const N_HEADS: usize = 24;
const N_KV_HEADS: usize = 4;
const D_HEAD: usize = 256;
const N_SLOTS: usize = 8192;
const KV_LEN: usize = 2048;
const K_BITS: u8 = 4;
const V_BITS: u8 = 4;
const RUN_LENGTHS: &[usize] = &[
    1, 2, 3, 4, 6, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
];

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
    println!(
        "device: {} (sm_{}, {} SMs)",
        dev.name(),
        dev.arch(),
        dev.sm_count()
    );
    let k = Kernels::new(dev.clone());
    let stream = dev.stream().clone();

    let max_n = RUN_LENGTHS.iter().copied().max().unwrap_or(1);
    let per_byte_k = 8 / K_BITS as usize;
    let per_byte_v = 8 / V_BITS as usize;

    // Shared TurboQuant pool state (packed codes over N_SLOTS), same shapes
    // as tq_attn_decode_bench.rs, reused by both paths.
    let q_rot = pseudo_random_f32(max_n * N_HEADS * D_HEAD, 1);
    let q_qjl = pseudo_random_f32(max_n * N_HEADS * D_HEAD, 2);
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
    // `qjl_t` is `d_head x d_head`, column-major (`DeviceTables::qjl_t`'s own
    // layout) -- numerics don't matter here, only that it's a real d*d f32
    // buffer the kernel reads through.
    let qjl_t = pseudo_random_f32(D_HEAD * D_HEAD, 11);

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
    let d_qjl_t = stream.clone_htod(&qjl_t)?;
    let d_slot_table = stream.clone_htod(&slot_table)?;

    let mut d_out = stream.alloc_zeros::<f32>(max_n * N_HEADS * D_HEAD)?;
    let mut d_partial =
        stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(N_HEADS, D_HEAD, max_n))?;

    // Compact dequant scratch, [N_KV_HEADS, KV_LEN, D_HEAD] f16 -- same
    // layout `Model`'s long-run dispatch path builds in crates/model/src/lib.rs.
    let dq_len = N_KV_HEADS * KV_LEN * D_HEAD;
    let mut d_dequant_k = stream.alloc_zeros::<f16>(dq_len)?;
    let mut d_dequant_v = stream.alloc_zeros::<f16>(dq_len)?;

    let attn_scale = 1.0 / (D_HEAD as f32).sqrt();
    let qjl_scale = 1.0f32; // real Tq4 cost: QJL correction term always on.

    println!("run_tokens | tq_attn_decode (us) | dequant+ws4 (us) | speedup");
    for &n in RUN_LENGTHS {
        // --- tq_attn_decode: dims.n_tokens = n, all in one sequence, this
        // run's tokens sitting at the tail of a KV_LEN-long cache. ---
        let seq_of = vec![0i32; n];
        let positions: Vec<i32> = ((KV_LEN - n) as i32..KV_LEN as i32).collect();
        let d_seq_of = stream.clone_htod(&seq_of)?;
        let d_positions = stream.clone_htod(&positions)?;
        let decode_batch = BatchLayout {
            seq_of: &d_seq_of.as_view(),
            positions: &d_positions.as_view(),
            slot_table: &d_slot_table.as_view(),
            table_stride: KV_LEN,
        };
        let decode_dims = AttnDims {
            n_heads: N_HEADS,
            n_kv_heads: N_KV_HEADS,
            d_head: D_HEAD,
            n_slots: N_SLOTS,
            n_tokens: n,
        };

        let run_decode = |out: &mut Buf<f32>, partial: &mut Buf<f32>| -> Result<()> {
            k.tq_attn_decode(
                &mut out.as_view_mut(),
                &d_q_rot.as_view(),
                &d_q_qjl.as_view(),
                &d_k_codes.as_view(),
                &d_k_signs.as_view(),
                &d_k_scale.as_view(),
                &d_k_gamma.as_view(),
                &d_v_codes.as_view(),
                &d_v_scale.as_view(),
                decode_batch,
                &d_k_levels.as_view(),
                K_BITS,
                &d_v_levels.as_view(),
                V_BITS,
                decode_dims,
                KV_LEN,
                attn_scale,
                qjl_scale,
                &mut partial.as_view_mut(),
            )
        };

        for _ in 0..3 {
            run_decode(&mut d_out, &mut d_partial)?;
        }
        stream.synchronize()?;
        let iters = 200;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            run_decode(&mut d_out, &mut d_partial)?;
        }
        stream.synchronize()?;
        let decode_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

        // --- dequant + ws4: dequantize the whole KV_LEN span once, then
        // serve this run's n query tokens with the ordinary tile kernel.
        // Both calls are inside the timed loop, same as production pays
        // for every long-run dispatch. ---
        let identity_seq_of = vec![0i32; n];
        let identity_slots: Vec<i32> = (0..KV_LEN as i32).collect();
        let ws4_positions: Vec<i32> = ((KV_LEN - n) as i32..KV_LEN as i32).collect();
        let d_identity_seq_of = stream.clone_htod(&identity_seq_of)?;
        let d_identity_slots = stream.clone_htod(&identity_slots)?;
        let d_ws4_positions = stream.clone_htod(&ws4_positions)?;
        let ws4_batch = BatchLayout {
            seq_of: &d_identity_seq_of.as_view(),
            positions: &d_ws4_positions.as_view(),
            slot_table: &d_identity_slots.as_view(),
            table_stride: KV_LEN,
        };
        let ws4_dims = AttnDims {
            n_heads: N_HEADS,
            n_kv_heads: N_KV_HEADS,
            d_head: D_HEAD,
            n_slots: KV_LEN,
            n_tokens: n,
        };

        let run_dequant_ws4 = |out: &mut Buf<f32>,
                               dequant_k: &mut Buf<f16>,
                               dequant_v: &mut Buf<f16>,
                               partial: &mut Buf<f32>|
         -> Result<()> {
            k.tq_dequant_kv(
                &mut dequant_k.as_view_mut(),
                &mut dequant_v.as_view_mut(),
                &d_k_codes.as_view(),
                &d_k_signs.as_view(),
                &d_k_scale.as_view(),
                &d_k_gamma.as_view(),
                &d_v_codes.as_view(),
                &d_v_scale.as_view(),
                &d_slot_table.as_view(),
                &d_k_levels.as_view(),
                K_BITS,
                &d_v_levels.as_view(),
                V_BITS,
                &d_qjl_t.as_view(),
                qjl_scale,
                N_KV_HEADS,
                D_HEAD,
                N_SLOTS,
                KV_LEN,
            )?;
            k.attn_prefill_ws4(
                &mut out.as_view_mut(),
                &d_q_rot.as_view(),
                &dequant_k.as_view(),
                &dequant_v.as_view(),
                ws4_batch,
                ws4_dims,
                0,
                n,
                KV_LEN,
                attn_scale,
                &mut partial.as_view_mut(),
            )
        };

        for _ in 0..3 {
            run_dequant_ws4(
                &mut d_out,
                &mut d_dequant_k,
                &mut d_dequant_v,
                &mut d_partial,
            )?;
        }
        stream.synchronize()?;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            run_dequant_ws4(
                &mut d_out,
                &mut d_dequant_k,
                &mut d_dequant_v,
                &mut d_partial,
            )?;
        }
        stream.synchronize()?;
        let dequant_ws4_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

        println!(
            "{n:>10} | {decode_us:>20.2} | {dequant_ws4_us:>17.2} | {:.2}x",
            decode_us / dequant_ws4_us
        );
    }
    Ok(())
}
