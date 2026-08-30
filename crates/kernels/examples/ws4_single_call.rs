//! One single, moderate-kv_len attn_prefill_ws4 launch, minimal buffers --
//! for ncu profiling (warp state / barrier stall timeline) without needing
//! the full 30552-token sweep's memory footprint. Written specifically
//! because GPU3 is nearly full tonight (production server + another user's
//! process) and the usual benchmark's buffers don't fit alongside ncu's own
//! instrumentation overhead.
use anyhow::Result;
use half::f16;
use infero_gpu::Device;
use infero_kernels::{AttnDims, BatchLayout, Kernels};

const N_HEADS: usize = 24;
const N_KV_HEADS: usize = 4;
const D_HEAD: usize = 256;
const KV_LEN: usize = 4096;
const RUN_TOKENS: usize = 1024;
const RUN_BASE: usize = KV_LEN - RUN_TOKENS;

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

    let n_slots = KV_LEN + 128;
    // q/out are full-sequence buffers (ws4's own convention: run_base
    // offsets into them), even though only the tail RUN_TOKENS slice is
    // ever written/read for this one call.
    let q = pseudo_random(KV_LEN * N_HEADS * D_HEAD, 0x71);
    let kv_elems = N_KV_HEADS * n_slots * D_HEAD;
    let kh: Vec<f16> = pseudo_random(kv_elems, 0x82).into_iter().map(f16::from_f32).collect();
    let vh: Vec<f16> = pseudo_random(kv_elems, 0x93).into_iter().map(f16::from_f32).collect();
    let seq_of = vec![0i32; KV_LEN];
    let positions: Vec<i32> = (0..KV_LEN as i32).collect();
    let table: Vec<i32> = (0..n_slots as i32).collect();

    let dq = stream.clone_htod(&q)?;
    let dk = stream.clone_htod(&kh)?;
    let dv = stream.clone_htod(&vh)?;
    let dpos = stream.clone_htod(&positions)?;
    let dseq = stream.clone_htod(&seq_of)?;
    let dtable = stream.clone_htod(&table)?;
    let (vseq, vpos, vtable) = (dseq.as_view(), dpos.as_view(), dtable.as_view());
    let batch = BatchLayout { seq_of: &vseq, positions: &vpos, slot_table: &vtable, table_stride: n_slots };

    let scale = 1.0 / (D_HEAD as f32).sqrt();
    let dims = AttnDims { n_heads: N_HEADS, n_kv_heads: N_KV_HEADS, d_head: D_HEAD, n_slots, n_tokens: KV_LEN };

    let mut out = stream.alloc_zeros::<f32>(KV_LEN * N_HEADS * D_HEAD)?;
    let mut part = stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(N_HEADS, D_HEAD, RUN_TOKENS))?;

    k.attn_prefill_ws4(
        &mut out.as_view_mut(), &dq.as_view(), &dk.as_view(), &dv.as_view(),
        batch, dims, RUN_BASE, RUN_TOKENS, KV_LEN, scale, &mut part.as_view_mut(),
    )?;
    k.device().synchronize()?;
    println!("ws4 single call done: kv_len={KV_LEN} run_tokens={RUN_TOKENS} run_base={RUN_BASE}");
    Ok(())
}
