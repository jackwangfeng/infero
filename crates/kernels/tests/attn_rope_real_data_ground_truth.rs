//! Ground-truth check for RoPE (rotary position embedding) applied to
//! regular attention's Q/K, using REAL, captured activations from a real
//! forward pass on the RadixArk NVFP4 checkpoint (via
//! `INFERO_PROBE=<layer>`/`INFERO_PROBE_DUMP=<dir>`,
//! `crates/model/src/lib.rs`'s `Model::attention`), rather than synthetic
//! data.
//!
//! By this point in the investigation, every matmul (`lm_head`, `o_proj`,
//! GDN's `in_proj_qz`/`out_proj`, Q/K/V's own projections structurally),
//! every GDN internal step (gating range, core recurrence, gated_rmsnorm),
//! attention's own score/softmax/combine mechanism, and the final norm are
//! all independently confirmed correct against real captured data, and a
//! full 64-layer `after_ffn` RMS scan shows smooth, monotonic growth with
//! zero layer-specific anomalies. RoPE -- applied to Q/K in place, between
//! the projection and the attention mechanism, and never itself
//! independently checked (every attention-mechanism ground-truth check so
//! far took infero's own post-RoPE Q/K as a GIVEN input) -- is the one
//! remaining computation.
//!
//! Reuses infero's own PRE-EXISTING, already-validated host reference
//! (`infero_model::qwen35::{rope_tables, apply_partial_rope}`, checked
//! against the real device kernels in `crates/kernels/tests/
//! partial_rope.rs` at this checkpoint's own real shape -- `head_dim=256`,
//! `rotary_dim=64`, `theta=10_000_000.0`, per that file's own constants)
//! rather than a fresh, unvalidated reimplementation.
//!
//! Skips (does not fail) when `INFERO_ATTN_PROBE_DIR` isn't set -- this
//! needs real dump files from a real `bw` run, not synthetic data.

use anyhow::{Context, Result};
use infero_model::qwen35::{apply_partial_rope, rope_tables};

// This checkpoint's real values, already established in this investigation
// (`crates/kernels/tests/partial_rope.rs`'s own constants, confirmed there
// as "the real 256 and 64" for head_dim/rotary_dim; theta alongside them).
const D_HEAD: usize = 256;
const ROTARY_DIM: usize = 64;
const THETA: f32 = 10_000_000.0;

fn read_f32(path: &str) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    anyhow::ensure!(
        bytes.len().is_multiple_of(4),
        "{path}: {} bytes, not a multiple of 4",
        bytes.len()
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect())
}

/// Runs the real, already-validated host reference over one tensor
/// (`Q` or `K`), `[t_len, heads, D_HEAD]` row-major, in place.
fn reference(x: &mut [f32], positions: &[u32], heads: usize) {
    let (cos, sin) = rope_tables(THETA, ROTARY_DIM, positions);
    apply_partial_rope(x, &cos, &sin, positions.len(), heads, D_HEAD, ROTARY_DIM);
}

fn compare(name: &str, got: &[f32], want: &[f32]) {
    let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let floor = 3e-2 * peak.max(f32::MIN_POSITIVE);
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let r = (g - w).abs() / (floor + 3e-2 * w.abs());
        if r > worst {
            worst = r;
            worst_at = i;
        }
    }
    println!(
        "{name}: worst relative error {worst:.4} at element {worst_at} (host reference {}, real device {})",
        got[worst_at], want[worst_at]
    );
    assert!(
        worst <= 1.0,
        "{name} disagrees with the ALREADY-VALIDATED host reference by {worst:.1}x tolerance \
         at element {worst_at}: host reference {}, real device {} -- a real bug in RoPE's real \
         kernel (rope_qk_partial/rope_qk_packed_partial, cu/*.cu) or this checkpoint's real \
         theta/rotary_dim/position values, not a hypothesis",
        got[worst_at],
        want[worst_at]
    );
}

#[test]
fn attn_q_and_k_match_the_rope_host_reference_on_real_captured_data() -> Result<()> {
    let Ok(dir) = std::env::var("INFERO_ATTN_PROBE_DIR") else {
        eprintln!(
            "skipping: set INFERO_ATTN_PROBE_DIR to the real probe dump directory \
             (e.g. /tmp/attn_probe from this investigation's own bw run) -- only \
             present when run against real captured data"
        );
        return Ok(());
    };
    // This model's real per-layer head counts -- fill in from the real
    // config.json (`num_attention_heads`/`num_key_value_heads`), the same
    // values the attention-mechanism ground-truth check (Addendum 9) used.
    // Left as an environment override rather than hardcoded so this test
    // doesn't need a source edit once the real numbers are confirmed.
    let n_heads: usize = std::env::var("INFERO_ATTN_N_HEADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .context("set INFERO_ATTN_N_HEADS to this model's real num_attention_heads")?;
    let n_kv_heads: usize = std::env::var("INFERO_ATTN_N_KV_HEADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .context("set INFERO_ATTN_N_KV_HEADS to this model's real num_key_value_heads")?;
    let da = n_heads * D_HEAD;
    let kv_dim = n_kv_heads * D_HEAD;

    let q_pre = read_f32(&format!("{dir}/eng.attn_q_prerope.f32"))?;
    let q_post = read_f32(&format!("{dir}/eng.attn_q_final.f32"))?;
    anyhow::ensure!(q_pre.len() == q_post.len(), "attn_q_prerope/attn_q_final length mismatch");
    anyhow::ensure!(q_pre.len().is_multiple_of(da), "attn_q_prerope: {} not a multiple of da={da}", q_pre.len());
    let t_len = q_pre.len() / da;
    println!("real captured data: t_len={t_len} (real prompt tokens this attention layer processed)");

    // Real positions for this investigation's single fresh, non-multiturn
    // request (prefill of a brand-new sequence, no prior KV-cache content):
    // position i is exactly token i's own index, 0..t_len. Flagged
    // explicitly rather than silently assumed -- if this investigation's
    // real request pattern ever changes (multi-turn, resumed generation),
    // this assumption stops holding and needs a real position dump instead.
    let positions: Vec<u32> = (0..t_len as u32).collect();

    let mut q = q_pre.clone();
    reference(&mut q, &positions, n_heads);
    compare("Q", &q, &q_post);

    // K: either unpacked (attn_k_prerope/attn_k_final) or still inside the
    // packed [q|k|v] row (attn_qkv_prerope_packed/attn_qkv_packed) --
    // de-interleave on the host side either way, same reasoning as the
    // attention-mechanism check (Addendum 9).
    let k_pre_path = format!("{dir}/eng.attn_k_prerope.f32");
    if std::path::Path::new(&k_pre_path).exists() {
        let k_pre = read_f32(&k_pre_path)?;
        let k_post = read_f32(&format!("{dir}/eng.attn_k_final.f32"))?;
        anyhow::ensure!(k_pre.len() == t_len * kv_dim, "attn_k_prerope: {} elements, want {}", k_pre.len(), t_len * kv_dim);
        let mut k = k_pre.clone();
        reference(&mut k, &positions, n_kv_heads);
        compare("K", &k, &k_post);
    } else {
        let fused_w = da + 2 * kv_dim;
        let packed_pre = read_f32(&format!("{dir}/eng.attn_qkv_prerope_packed.f32"))?;
        let packed_post = read_f32(&format!("{dir}/eng.attn_qkv_packed.f32"))?;
        anyhow::ensure!(packed_pre.len() == t_len * fused_w, "attn_qkv_prerope_packed: {} elements, want {}", packed_pre.len(), t_len * fused_w);
        let mut k_pre = vec![0.0f32; t_len * kv_dim];
        let mut k_post = vec![0.0f32; t_len * kv_dim];
        for t in 0..t_len {
            k_pre[t * kv_dim..(t + 1) * kv_dim]
                .copy_from_slice(&packed_pre[t * fused_w + da..t * fused_w + da + kv_dim]);
            k_post[t * kv_dim..(t + 1) * kv_dim]
                .copy_from_slice(&packed_post[t * fused_w + da..t * fused_w + da + kv_dim]);
        }
        let mut k = k_pre.clone();
        reference(&mut k, &positions, n_kv_heads);
        compare("K (de-interleaved from the packed row)", &k, &k_post);
    }
    Ok(())
}
