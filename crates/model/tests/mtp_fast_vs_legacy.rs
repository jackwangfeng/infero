//! Does the new non-materializing attention path
//! (`attn_prefill_decoupled6_f16acc`, gated by `INFERO_ATTN_MMA=1` +
//! `d_head==256`) produce the same logits as the legacy
//! `attn_scores`/`attn_softmax`/`attn_output` triple it replaces?
//!
//! `MtpHead::new`'s `use_fast_attn` decision reads `INFERO_ATTN_MMA` through
//! `Kernels::prefill_attention`, which caches the env var in a process-wide
//! `OnceLock` -- so the two paths can't be exercised from the same test
//! binary. Instead this one test is run twice, once per process, with the
//! shape held identical (24 heads / 4 kv heads / d_head=256, the real
//! qwen38-27b-fp8 attention shape, so the exact-256 gate on
//! `attn_prefill_decoupled6_f16acc` is actually exercised):
//!
//!   cargo test -p infero-model --test mtp_fast_vs_legacy -- --nocapture            # legacy (env unset)
//!   INFERO_ATTN_MMA=1 cargo test -p infero-model --test mtp_fast_vs_legacy -- --nocapture  # fast path
//!
//! Each run prints its row-0 logits as one comma-separated line prefixed
//! `LOGITS:` for a caller to diff across the two invocations.

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::Kernels;
use infero_model::mtp::{HeadDims, MtpHead};
use infero_model::weights::{AttnWeights, DenseFfn, Layer, Matrix, MtpWeights};

/// Real qwen38-27b-fp8 attention shape (24 heads / 4 kv / d_head 256) so the
/// fast kernel's exact `d_head == 256` gate is actually exercised; `d_model`
/// and `vocab` shrunk since only the attention shape matters here.
fn dims() -> HeadDims {
    HeadDims {
        d_model: 64,
        heads: 24,
        kv_heads: 4,
        d_head: 256,
        rotary_dim: 64,
        d_ff: 128,
        eps: 1e-6,
        rope_theta: 10_000.0,
        vocab: 32,
        mrope_section: None,
    }
}

fn synth(dev: &Device, dims: HeadDims) -> Result<(MtpWeights, Matrix)> {
    let seed = std::cell::Cell::new(0x1234_5678u32);
    let next = move || {
        seed.set(seed.get().wrapping_mul(1_664_525).wrapping_add(1_013_904_223));
        (seed.get() >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    };
    let (d, da, dkv) = (dims.d_model, dims.d_attn(), dims.d_kv());
    let m = |k: usize, n: usize| -> Result<Matrix> {
        let v: Vec<half::f16> = (0..k * n).map(|_| half::f16::from_f32(next() * 0.3)).collect();
        Matrix::upload_f16(dev, &v, k, n)
    };
    let vec1 = |n: usize| -> Result<infero_model::weights::Vector> {
        let v: Vec<f32> = (0..n).map(|_| 1.0 + next() * 0.1).collect();
        Ok(dev.stream().clone_htod(&v)?)
    };
    let w = MtpWeights {
        fc: m(2 * d, d)?,
        pre_fc_norm_embedding: vec1(d)?,
        pre_fc_norm_hidden: vec1(d)?,
        norm: vec1(d)?,
        layer: Layer {
            attn_norm: vec1(d)?,
            attn: Some(AttnWeights {
                wq: m(d, 2 * da)?,
                wk: m(d, dkv)?,
                wv: m(d, dkv)?,
                wo: m(da, d)?,
                bq: None,
                bk: None,
                bv: None,
                bo: None,
                q_norm: Some(vec1(dims.d_head)?),
                k_norm: Some(vec1(dims.d_head)?),
                w_qkv: None,
                w_kv: None,
                output_gate: true,
            }),
            gdn: None,
            ffn_norm: vec1(d)?,
            dense: Some(DenseFfn {
                w_gate: m(d, dims.d_ff)?,
                w_up: m(d, dims.d_ff)?,
                w_down: m(dims.d_ff, d)?,
                w_gate_up: None,
            }),
            moe: None,
            blob: None,
        },
        device_bytes: 0,
    };
    let embed = m(d, dims.vocab)?;
    Ok((w, embed))
}

const T: usize = 6;

#[test]
fn prints_row0_logits_for_cross_process_diff() -> Result<()> {
    let Ok(dev) = Device::new(0) else {
        eprintln!("no CUDA device; skipping");
        return Ok(());
    };
    let kern = Kernels::new(dev.clone());
    kern.warm_up()?;
    let dm = dims();
    let d = dm.d_model;
    let (w, embed) = synth(&dev, dm)?;
    let mut head = MtpHead::new(&dev, &kern, w, dm, T, 128, 1)?;

    let ids: Vec<u32> = (0..T as u32).map(|i| (i * 7 + 3) % dm.vocab as u32).collect();
    let positions: Vec<usize> = (0..T).collect();
    let hidden_host: Vec<f32> = (0..T * d).map(|i| ((i * 37 % 101) as f32 - 50.0) / 97.0).collect();
    let hidden = dev.stream().clone_htod(&hidden_host)?;

    head.step(&kern, &embed, &ids, &positions, &hidden.as_view(), None)?;

    let env = std::env::var("INFERO_ATTN_MMA").unwrap_or_default();
    eprintln!("mode: INFERO_ATTN_MMA={env:?} d_head={}", dm.d_head);
    // Every row, not just row 0: row r attends causally over r+1 keys, so
    // later rows are the only ones that actually exercise the multi-key
    // causal window the fast kernel must get right, not just the trivial
    // single-key case row 0 reduces to.
    for r in 0..T {
        let row = head.logits_row(&kern, &embed, r)?.to_vec();
        let line: Vec<String> = row.iter().map(|v| format!("{v:.6}")).collect();
        eprintln!("LOGITS[{r}]:{}", line.join(","));
    }
    Ok(())
}
