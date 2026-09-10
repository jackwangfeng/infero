//! Ground-truth check for GatedDeltaNet's `gated_delta_rule` recurrence
//! using REAL, captured activations from a real forward pass on the
//! RadixArk NVFP4 checkpoint (via `INFERO_PROBE=<layer>`/
//! `INFERO_PROBE_DUMP=<dir>`, `crates/model/src/lib.rs`'s
//! `Model::linear_attention`), rather than synthetic data.
//!
//! Every matmul in this checkpoint's forward pass (`lm_head`, regular
//! attention's `o_proj`, GDN's `in_proj_qz` fusion, GDN's `out_proj`) has
//! now been independently ground-truth-checked and confirmed correct given
//! its own real input -- yet real vLLM inference on the exact same
//! checkpoint produces coherent output while infero's does not (a real,
//! systematic bias toward low-vocab-id digit tokens, not random noise).
//! The one real, non-matmul computation in the whole 64-layer pipeline no
//! check has touched yet is the gated-delta-rule recurrence itself.
//!
//! Rather than writing a NEW, unvalidated reimplementation of that
//! recurrence (a real risk: a bug in a fresh reimplementation would prove
//! nothing), this feeds infero's own PRE-EXISTING, already-validated host
//! reference (`infero_model::qwen35::gated_delta_rule`, checked against a
//! vendored `flash-linear-attention` reference per `gated_delta.rs`'s own
//! doc comment) the REAL q/k/v/g/beta values this checkpoint's own forward
//! pass produced, and compares its answer against the REAL, real-hardware
//! device-computed `gdn_core` output for the exact same real data. If they
//! disagree, that is a real, precisely-located bug in the device kernel
//! (`gdn_delta_rule_f32`/`gdn_delta_rule_reg128_f32`, `cu/gdn.cu`) or in
//! how this checkpoint's real config (head counts, `v_heads_tiled`) feeds
//! it -- not a hypothesis, a direct disagreement with an independently
//! trusted reference. If they agree, the recurrence itself is cleared too.
//!
//! No new probe round needed: this reuses the exact same
//! `eng.gdn_qk_normed.f32`/`eng.gdn_g.f32`/`eng.gdn_beta.f32`/
//! `eng.gdn_core.f32` dump files the earlier gating-sanity-check round
//! already captured (see `task8-lmhead-rootcause-report.md`'s Addendum 6),
//! since `gdn_qk_normed` already carries q, k AND v (the whole packed row)
//! and `gdn_core` is exactly the value to compare against.
//!
//! Skips (does not fail) when `INFERO_GDN_PROBE_DIR` isn't set -- this
//! needs real dump files from a real `bw` run, not synthetic data, the same
//! convention this project's other real-checkpoint-gated tests use.

use anyhow::{Context, Result};
use infero_model::qwen35::gated_delta_rule;

// This checkpoint's real GDN shape (RadixArk/Qwen3.8-27B-NVFP4), already
// established earlier in this investigation (`crates/kernels/src/cu/gdn.cu`'s
// own sm120 timing-table comment: "48 value heads, 16 key heads, dk = dv =
// 128"; `in_proj_z`'s real `value_dim=6144=48*128` and `in_proj_qkv`'s real
// `width=10240=2*2048+6144` with `key_dim=2048=16*128`, both confirmed via
// direct checkpoint header reads in this same investigation).
const KEY_HEADS: usize = 16;
const VAL_HEADS: usize = 48;
const DK: usize = 128;
const DV: usize = 128;
const KEY_DIM: usize = KEY_HEADS * DK; // 2048
const VAL_DIM: usize = VAL_HEADS * DV; // 6144
const WIDTH: usize = 2 * KEY_DIM + VAL_DIM; // 10240, matches `acts.qkv_conv`'s real row width
const REP: usize = VAL_HEADS / KEY_HEADS; // 3

fn read_f32(path: &str) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    anyhow::ensure!(
        bytes.len().is_multiple_of(4),
        "{path}: {} bytes, not a multiple of 4 (not a raw f32 dump?)",
        bytes.len()
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect())
}

/// Expands `key_heads`-wide q/k to `key_heads * rep` (= value-heads) width
/// the way real Hugging Face `repeat_interleave` does: value head `h` reads
/// key head `h / rep`, i.e. each key head's row is repeated `rep` times
/// CONSECUTIVELY -- matching `cu/gdn.cu`'s own real, non-tiled
/// (`v_heads_tiled == false`, confirmed this checkpoint's real, safetensors/
/// HF-format convention, `crates/model/src/config.rs:704`) `khead = head /
/// (heads / key_heads)` grouping exactly.
fn repeat_interleave(x: &[f32], t_len: usize, key_heads: usize, rep: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; t_len * key_heads * rep * d];
    for t in 0..t_len {
        for kh in 0..key_heads {
            let src = &x[(t * key_heads + kh) * d..(t * key_heads + kh + 1) * d];
            for r in 0..rep {
                let h = kh * rep + r;
                out[(t * key_heads * rep + h) * d..(t * key_heads * rep + h + 1) * d]
                    .copy_from_slice(src);
            }
        }
    }
    out
}

#[test]
fn gdn_core_matches_the_host_reference_on_real_captured_data() -> Result<()> {
    let Ok(dir) = std::env::var("INFERO_GDN_PROBE_DIR") else {
        eprintln!(
            "skipping: set INFERO_GDN_PROBE_DIR to the real probe dump directory \
             (e.g. /tmp/gdn_probe2 from this investigation's own bw run) -- only \
             present when run against real captured data"
        );
        return Ok(());
    };

    let packed = read_f32(&format!("{dir}/eng.gdn_qk_normed.f32"))?;
    let g = read_f32(&format!("{dir}/eng.gdn_g.f32"))?;
    let beta = read_f32(&format!("{dir}/eng.gdn_beta.f32"))?;
    let want = read_f32(&format!("{dir}/eng.gdn_core.f32"))?;

    anyhow::ensure!(
        packed.len().is_multiple_of(WIDTH),
        "eng.gdn_qk_normed.f32: {} elements, not a multiple of WIDTH={WIDTH}",
        packed.len()
    );
    let t_len = packed.len() / WIDTH;
    anyhow::ensure!(
        g.len() == t_len * VAL_HEADS,
        "eng.gdn_g.f32: {} elements, want {} ({t_len} tokens * {VAL_HEADS} heads)",
        g.len(),
        t_len * VAL_HEADS
    );
    anyhow::ensure!(
        beta.len() == t_len * VAL_HEADS,
        "eng.gdn_beta.f32: {} elements, want {}",
        beta.len(),
        t_len * VAL_HEADS
    );
    anyhow::ensure!(
        want.len() == t_len * VAL_DIM,
        "eng.gdn_core.f32: {} elements, want {} ({t_len} tokens * {VAL_DIM})",
        want.len(),
        t_len * VAL_DIM
    );
    println!("real captured data: t_len={t_len} (real prompt tokens this GDN layer processed)");

    // Slice q/k (key-head width, already l2-normed+scaled by the real
    // device `gdn_qk_l2norm` kernel) and v (untouched by that kernel, still
    // whatever the real conv+SiLU produced) out of the packed row.
    let mut q_small = vec![0.0f32; t_len * KEY_DIM];
    let mut k_small = vec![0.0f32; t_len * KEY_DIM];
    let mut v = vec![0.0f32; t_len * VAL_DIM];
    for t in 0..t_len {
        let row = &packed[t * WIDTH..(t + 1) * WIDTH];
        q_small[t * KEY_DIM..(t + 1) * KEY_DIM].copy_from_slice(&row[0..KEY_DIM]);
        k_small[t * KEY_DIM..(t + 1) * KEY_DIM].copy_from_slice(&row[KEY_DIM..2 * KEY_DIM]);
        v[t * VAL_DIM..(t + 1) * VAL_DIM].copy_from_slice(&row[2 * KEY_DIM..WIDTH]);
    }

    let q = repeat_interleave(&q_small, t_len, KEY_HEADS, REP, DK);
    let k = repeat_interleave(&k_small, t_len, KEY_HEADS, REP, DK);

    let mut state = vec![0.0f32; VAL_HEADS * DK * DV];
    // `gated_delta_rule` re-applies l2norm + the `1/sqrt(dk)` q-scale
    // internally (it is designed to take RAW q/k, per its own doc comment).
    // Feeding it already-normalized q/k here is a real mathematical no-op,
    // not an approximation: re-normalizing an already-unit-direction vector
    // by the same `dk` reproduces it exactly (`k`: already unit norm, so
    // `k/||k|| == k`; `q`: already has norm `1/sqrt(dk)`, so normalizing it
    // to unit length and rescaling by `1/sqrt(dk)` again returns the exact
    // same vector) -- confirmed algebraically, not assumed, before relying
    // on it here.
    let got = gated_delta_rule(&q, &k, &v, &g, &beta, &mut state, t_len, VAL_HEADS, DK, DV, 1e-6);

    // Same tolerance shape this project's other real-hardware-vs-host-
    // reference comparisons use (a relative-error floor plus a relative
    // term against the peak magnitude).
    let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let floor = 3e-2 * peak.max(f32::MIN_POSITIVE);
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for (i, (&g_, &w)) in got.iter().zip(want.iter()).enumerate() {
        let r = (g_ - w).abs() / (floor + 3e-2 * w.abs());
        if r > worst {
            worst = r;
            worst_at = i;
        }
    }
    let (t, rest) = (worst_at / VAL_DIM, worst_at % VAL_DIM);
    let (h, j) = (rest / DV, rest % DV);
    println!(
        "worst relative error {worst:.4} at element {worst_at} (token {t}, head {h}, col {j}): \
         host reference {}, real device {}",
        got[worst_at], want[worst_at]
    );
    assert!(
        worst <= 1.0,
        "gdn_core disagrees with the ALREADY-VALIDATED host reference by {worst:.1}x tolerance \
         at token {t}, head {h}, col {j}: host reference {}, real device {} -- a real bug in \
         gdn_delta_rule_f32/gdn_delta_rule_reg128_f32 (cu/gdn.cu) or this checkpoint's real \
         config feeding it, not a hypothesis",
        got[worst_at],
        want[worst_at]
    );
    Ok(())
}
