//! Direct, head-to-head decode-time attention kernel benchmark: infero's
//! `attn_decode` (the real kernel the production dispatch path picks for
//! every decode step, see `crates/model/src/lib.rs`'s `attention()` --
//! `INFERO_DECODE_ATTN` gate) against vLLM's own real decode-time kernel
//! (`flash_attn_varlen_func`, paged KV cache, `fa_version=2` on this
//! Blackwell RTX PRO 6000 box -- see `scripts/vllm_attn_decode_kernel_bench.py`
//! for that side).
//!
//! Shape matches the real Qwen3.8-27B-NVFP4 checkpoint (24 query heads, 4 KV
//! heads, 256-wide heads, GQA group 6) at a real decode batch (B=16) and a
//! realistic mid-generation kv_len. Data is pseudo-random -- this measures
//! the kernel's timing, not its numerics, same convention as every other
//! attention bench in this crate.
//!
//!     INFERO_ATTN_MMA=1 cargo run --release --features cuda -p infero-kernels \
//!       --example attn_decode_vs_vllm_bench

use anyhow::Result;
use half::f16;
use infero_gpu::Device;
use infero_kernels::{AttnDims, BatchLayout, Kernels};

const N_HEADS: usize = 24;
const N_KV_HEADS: usize = 4;
const D_HEAD: usize = 256;
const KV_LEN: usize = 2048;
const BATCH: usize = 16;

fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
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

    // One contiguous kv_len-slot pool per sequence, flattened -- mirrors the
    // paged cache's per-sequence contiguity closely enough for a kernel-time
    // (not eviction-behavior) comparison.
    let n_slots = BATCH * KV_LEN;
    let group = N_HEADS / N_KV_HEADS;

    let q = pseudo_random(BATCH * N_HEADS * D_HEAD, 0x71);
    let kv_elems = N_KV_HEADS * n_slots * D_HEAD;
    let kh: Vec<f16> = pseudo_random(kv_elems, 0x82).into_iter().map(f16::from_f32).collect();
    let vh: Vec<f16> = pseudo_random(kv_elems, 0x93).into_iter().map(f16::from_f32).collect();

    let seq_of: Vec<i32> = (0..BATCH as i32).collect();
    let positions = vec![(KV_LEN - 1) as i32; BATCH];
    // slot_table[seq * table_stride + pos] -> pool slot; contiguous per seq.
    let slot_table: Vec<i32> = (0..(BATCH * KV_LEN) as i32).collect();

    let dq = stream.clone_htod(&q)?;
    let dk = stream.clone_htod(&kh)?;
    let dv = stream.clone_htod(&vh)?;
    let d_seq_of = stream.clone_htod(&seq_of)?;
    let d_positions = stream.clone_htod(&positions)?;
    let d_slot_table = stream.clone_htod(&slot_table)?;
    let mut d_out = stream.alloc_zeros::<f32>(BATCH * N_HEADS * D_HEAD)?;
    let mut d_partial =
        stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(N_HEADS, D_HEAD, BATCH))?;

    let dims = AttnDims {
        n_heads: N_HEADS,
        n_kv_heads: N_KV_HEADS,
        d_head: D_HEAD,
        n_slots,
        n_tokens: BATCH,
    };
    let batch = BatchLayout {
        seq_of: &d_seq_of.as_view(),
        positions: &d_positions.as_view(),
        slot_table: &d_slot_table.as_view(),
        table_stride: KV_LEN,
    };

    anyhow::ensure!(
        k.decode_attention(&dims, KV_LEN),
        "shape not eligible for attn_decode on this build -- check INFERO_ATTN_MMA / d_head/group support"
    );

    for _ in 0..5 {
        k.attn_decode(
            &mut d_out.as_view_mut(),
            None,
            &dq.as_view(),
            &dk.as_view(),
            &dv.as_view(),
            batch,
            dims,
            KV_LEN,
            1.0 / (D_HEAD as f32).sqrt(),
            &mut d_partial.as_view_mut(),
        )?;
    }
    stream.synchronize()?;

    let iters = 500;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        k.attn_decode(
            &mut d_out.as_view_mut(),
            None,
            &dq.as_view(),
            &dk.as_view(),
            &dv.as_view(),
            batch,
            dims,
            KV_LEN,
            1.0 / (D_HEAD as f32).sqrt(),
            &mut d_partial.as_view_mut(),
        )?;
    }
    stream.synchronize()?;
    let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!(
        "attn_decode: {us:.2} us/call, B={BATCH} kv_len={KV_LEN} n_heads={N_HEADS} \
         n_kv_heads={N_KV_HEADS} d_head={D_HEAD} group={group} (INFERO_ATTN_MMA={:?})",
        std::env::var("INFERO_ATTN_MMA").ok()
    );
    Ok(())
}
