//! Minimal, single-variant harness for `attn_prefill_ws4` correctness/hazard
//! checking under compute-sanitizer -- the full `attn_prefill_pipe_bench`
//! (6 variants x 16 layers x ~30 chunks x several reps) is far too slow
//! under memcheck/racecheck instrumentation for a quick check. A handful of
//! chunks at the real shape is enough to exercise every code path (multiple
//! `n_blk` values, the tail partial-tile case, both `single`/multi-chunk
//! branches) without the full prefill's wall-clock cost.

use anyhow::Result;
use half::f16;
use infero_gpu::Device;
use infero_kernels::{AttnDims, BatchLayout, Kernels};

const N_HEADS: usize = 24;
const N_KV_HEADS: usize = 4;
const D_HEAD: usize = 256;
const TOTAL_TOKENS: usize = 4096;
const BATCH_TOKENS: usize = 1024;

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
    let k = Kernels::new(Device::new(0)?);
    let stream = k.device().stream().clone();

    let n_slots = TOTAL_TOKENS + 128;
    let n_tokens = TOTAL_TOKENS;
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
    let mut part = stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(N_HEADS, D_HEAD, BATCH_TOKENS))?;

    let mut base = 0usize;
    while base < TOTAL_TOKENS {
        let run_tokens = BATCH_TOKENS.min(TOTAL_TOKENS - base);
        let kv_len = base + run_tokens;
        k.attn_prefill_ws4(
            &mut out.as_view_mut(), &dq.as_view(), &dk.as_view(), &dv.as_view(),
            batch, dims, base, run_tokens, kv_len, scale, &mut part.as_view_mut(),
        )?;
        base += run_tokens;
    }
    k.device().synchronize()?;
    println!("attn_prefill_ws4: {TOTAL_TOKENS} tokens, {} chunks, ok", TOTAL_TOKENS.div_ceil(BATCH_TOKENS));
    Ok(())
}
