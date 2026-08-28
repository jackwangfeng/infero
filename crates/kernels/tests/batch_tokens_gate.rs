//! Qwen3.8-27B-FP8's shape — 24 heads over 4 KV heads, `d_head` 256 — is the
//! one `infero_model`'s `needs_score_buffer` was written against: both fused
//! attention kernels take it, so its prefill chunk should never shrink with
//! `--ctx`. This pins the two gates that decision rests on, independent of
//! any model file. See `crates/model/tests/batch_tokens.rs` for the
//! `batch_tokens_for` side of the same fix.

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::{AttnDims, Kernels};

const QWEN38_27B: AttnDims = AttnDims {
    n_heads: 24,
    n_kv_heads: 4,
    d_head: 256,
    n_slots: 0,
    n_tokens: 0,
};

fn kernels_or_skip() -> Result<Option<Kernels>> {
    match Device::new(0) {
        Ok(dev) => Ok(Some(Kernels::new(dev))),
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            Ok(None)
        }
    }
}

#[test]
fn qwen38_shape_takes_the_fused_dense_kernel() -> Result<()> {
    let Some(kern) = kernels_or_skip()? else {
        return Ok(());
    };
    // The gate on CUDA does not read `kv_len`, so any value here stands for
    // every context length this model could be configured with.
    assert!(kern.decode_attention(&QWEN38_27B, 131072));
    Ok(())
}

#[test]
fn qwen38_shape_takes_the_fused_turboquant_kernel() -> Result<()> {
    let Some(kern) = kernels_or_skip()? else {
        return Ok(());
    };
    assert!(kern.tq_decode_attention(&QWEN38_27B));
    Ok(())
}

#[test]
fn a_group_ratio_outside_turboquants_unrolled_range_falls_back() -> Result<()> {
    let Some(kern) = kernels_or_skip()? else {
        return Ok(());
    };
    // `tq_attn_decode` unrolls the group loop up to 8; a shape this lopsided
    // is what the three-kernel fallback still exists for.
    let wide_group = AttnDims {
        n_heads: 32,
        n_kv_heads: 1,
        ..QWEN38_27B
    };
    assert!(!kern.tq_decode_attention(&wide_group));
    Ok(())
}
