//! Minimal harness for `attn_prefill_split` correctness/hazard checking
//! under compute-sanitizer -- mirrors `attn_prefill_ws4_only.rs`'s
//! reasoning: the full 16-layer benchmark is too slow to sanitize usefully.
//! Covers both the single-chunk direct-write path (small `kv_len`) and the
//! multi-chunk `attn_ms_reduce_f32`/`attn_pv_sum_reduce_f32` path (large
//! `kv_len`), since those are genuinely different code paths.

use anyhow::Result;
use half::f16;
use infero_gpu::Device;
use infero_kernels::{AttnDims, BatchLayout, Kernels};

const N_HEADS: usize = 24;
const N_KV_HEADS: usize = 4;
const D_HEAD: usize = 256;

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

fn run(k: &Kernels, total_tokens: usize) -> Result<()> {
    let stream = k.device().stream().clone();
    let n_slots = total_tokens + 128;
    let n_tokens = total_tokens;
    let q = pseudo_random(n_tokens * N_HEADS * D_HEAD, 0x71);
    let kv_elems = N_KV_HEADS * n_slots * D_HEAD;
    let kh: Vec<f16> = pseudo_random(kv_elems, 0x82).into_iter().map(f16::from_f32).collect();
    let vh: Vec<f16> = pseudo_random(kv_elems, 0x93).into_iter().map(f16::from_f32).collect();
    let seq_of = vec![0i32; n_tokens];
    let positions: Vec<i32> = (0..n_tokens as i32).collect();
    let table: Vec<i32> = (0..n_slots as i32).collect();
    let table_stride = n_slots;

    let dq = stream.clone_htod(&q)?;
    let dk = stream.clone_htod(&kh)?;
    let dv = stream.clone_htod(&vh)?;
    let dpos = stream.clone_htod(&positions)?;
    let dseq = stream.clone_htod(&seq_of)?;
    let dtable = stream.clone_htod(&table)?;
    let (vseq, vpos, vtable) = (dseq.as_view(), dpos.as_view(), dtable.as_view());
    let batch = BatchLayout { seq_of: &vseq, positions: &vpos, slot_table: &vtable, table_stride };

    let scale = 1.0 / (D_HEAD as f32).sqrt();
    let dims = AttnDims { n_heads: N_HEADS, n_kv_heads: N_KV_HEADS, d_head: D_HEAD, n_slots, n_tokens };

    let mut out = stream.alloc_zeros::<f32>(n_tokens * N_HEADS * D_HEAD)?;
    let mut ms = stream.alloc_zeros::<f32>(Kernels::attn_ms_floats(N_HEADS, total_tokens))?;
    let mut part = stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(N_HEADS, D_HEAD, total_tokens))?;
    k.attn_prefill_split(
        &mut out.as_view_mut(), &dq.as_view(), &dk.as_view(), &dv.as_view(),
        batch, dims, 0, total_tokens, total_tokens, scale, &mut ms.as_view_mut(), &mut part.as_view_mut(),
    )?;
    k.device().synchronize()?;
    println!("attn_prefill_split: {total_tokens} tokens, ok");
    Ok(())
}

fn main() -> Result<()> {
    let k = Kernels::new(Device::new(0)?);
    run(&k, 32)?; // single-chunk direct-write path
    run(&k, 8192)?; // multi-chunk reduce path
    Ok(())
}
