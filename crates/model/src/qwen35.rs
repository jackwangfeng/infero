//! Host-side reference for the two Qwen3.5 block types.
//!
//! This is deliberately the slow, obvious version: plain `f32`, explicit
//! shapes, one loop per thing the reference implementation does. It exists to
//! be *read* as the spec and to be the thing CUDA kernels are checked against.
//!
//! Why a reference at all, rather than going straight to kernels: the
//! bf16-as-f16 embedding bug survived nine component-level A/Bs because each
//! one asked "did this stage do its job" about a stage that was doing its job
//! perfectly on nonsense. The layout decisions here — where q ends and k
//! begins, whether the attention gate interleaves per head, which transpose
//! the recurrence takes — all have a second reading that runs to completion
//! and produces fluent garbage. So each one is pinned against a capture of the
//! reference implementation on the real checkpoint
//! (`tools/capture_qwen35_layers.py`).
//!
//! Names follow `transformers.models.qwen3_5.modeling_qwen3_5`.

/// SiLU, a.k.a. swish: `x * sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `log(1 + exp(x))`, computed so that a large `x` does not overflow before the
/// log brings it back. `dt_bias` reaches +19 in the real checkpoint, so this
/// matters: `exp(19)` is fine but `a + dt_bias` is unbounded in principle.
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// Depthwise causal convolution over time, one weight row per channel.
///
/// `x` is `[t_len, channels]` row-major — token-major, which is how activations
/// already sit in infero. The reference transposes to `[channels, t_len]` and
/// calls `F.conv1d` with `padding = k - 1` then truncates to the first `t_len`
/// outputs; that combination is exactly a causal window, so this reads the
/// window directly and skips the transpose.
///
/// The direction is the part to get right. Output `t` looks at inputs
/// `t-(k-1) ..= t`, and weight index `j` pairs with input `t - (k-1) + j`, so
/// `w[k-1]` multiplies the *current* token. Reversing that runs fine and shifts
/// the whole model one token into the future.
pub fn depthwise_causal_conv1d(
    x: &[f32],
    w: &[f32],
    t_len: usize,
    channels: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), t_len * channels);
    assert_eq!(w.len(), channels * k);
    let mut out = vec![0.0f32; t_len * channels];
    for t in 0..t_len {
        for c in 0..channels {
            let mut acc = 0.0f32;
            for j in 0..k {
                // The input this tap reads; taps before the start are zero,
                // which is what the left padding means.
                let src = t as isize - (k as isize - 1) + j as isize;
                if src >= 0 {
                    acc += w[c * k + j] * x[src as usize * channels + c];
                }
            }
            out[t * channels + c] = acc;
        }
    }
    out
}

/// Continue a depthwise causal convolution from a saved window.
///
/// `state` is `[channels, k-1]`, oldest first: the tail of the tokens already
/// consumed. Returns the outputs for the new tokens and leaves `state` holding
/// the window for whatever comes next. Decode calls this with `t_len == 1`.
pub fn depthwise_causal_conv1d_update(
    x: &[f32],
    state: &mut [f32],
    w: &[f32],
    t_len: usize,
    channels: usize,
    k: usize,
) -> Vec<f32> {
    let hist = k - 1;
    assert_eq!(state.len(), channels * hist);
    let mut out = vec![0.0f32; t_len * channels];
    for t in 0..t_len {
        for c in 0..channels {
            let mut acc = 0.0f32;
            for j in 0..k {
                let src = t as isize - hist as isize + j as isize;
                let v = if src >= 0 {
                    x[src as usize * channels + c]
                } else {
                    // Negative reaches back into the saved window; -hist is its
                    // first (oldest) slot.
                    state[c * hist + (hist as isize + src) as usize]
                };
                acc += w[c * k + j] * v;
            }
            out[t * channels + c] = acc;
        }
    }
    // Refresh the window with the last `hist` tokens seen, drawing from the old
    // state when the new chunk is shorter than the window.
    let mut fresh = vec![0.0f32; channels * hist];
    for c in 0..channels {
        for s in 0..hist {
            // Position of this slot counted backwards from the end.
            let back = hist - s; // 1..=hist tokens before the next one
            let idx = t_len as isize - back as isize;
            fresh[c * hist + s] = if idx >= 0 {
                x[idx as usize * channels + c]
            } else {
                state[c * hist + (hist as isize + idx) as usize]
            };
        }
    }
    state.copy_from_slice(&fresh);
    out
}

/// L2-normalize each row of `rows` values, matching the FLA convention the
/// reference cites: `eps` is added to the sum of squares, not to the norm.
///
/// Two things about that `eps`. It is a *literal* `1e-6` in the reference —
/// `l2norm(query, dim=-1, eps=1e-6)` inside both delta-rule paths — and not
/// `rms_norm_eps`. The two coincide on this checkpoint, which is exactly how
/// they would come to be conflated; callers should pass `1e-6`, and
/// `lib.rs`'s `gdn_qk_l2norm` does.
///
/// And the placement is now pinned rather than believed: at unit norm the three
/// readings (eps on the sum of squares, on the mean, or added to the root) agree
/// to 3e-8, well inside any tolerance a test would use, so the recurrence check
/// could not see it. `cross_check_against_transformers` in
/// `tools/capture_qwen35_layers.py` therefore also compares against
/// `modeling_qwen3_5.l2norm` at an RMS near `sqrt(eps)`, where the sum-form and
/// the root-form land 167% and 8% away.
pub fn l2norm_rows(x: &mut [f32], row_len: usize, eps: f32) {
    for row in x.chunks_mut(row_len) {
        let ss: f32 = row.iter().map(|v| v * v).sum();
        let inv = (ss + eps).sqrt().recip();
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

/// RMSNorm with a learned gain, over the last dimension.
///
/// `gain_offset` is the trap. Qwen3.5 has two RMSNorm classes and they differ by
/// exactly this:
///
/// - `Qwen3_5RMSNorm` initializes its weight to **zeros** and computes
///   `normalized * (1 + weight)`. Every regular norm in the text model is one of
///   these: `input_layernorm`, `post_attention_layernorm`, the final `norm`, and
///   the per-head `q_norm` and `k_norm`. Pass `1.0`.
/// - `Qwen3_5RMSNormGated` initializes to **ones** and computes
///   `weight * normalized`. Only the GatedDeltaNet output norm
///   (`linear_attn.norm`) is one of these. Pass `0.0`.
///
/// Reading the weights cannot settle which is which. The two populations do
/// separate on average — an `input_layernorm` centred at 0.036 would be
/// annihilated by the plain form — but they overlap: some trained `q_norm`
/// deltas exceed 0.5 while some `linear_attn.norm` gains fall below 1.5. Only
/// the consuming class decides, so this takes the offset as an argument rather
/// than guessing from the data.
///
/// Getting it wrong on `q_norm` scales every query by roughly 0.23 instead of
/// 1.23 and inverts the sign wherever the delta is negative, which is a model
/// that talks fluently and means something else.
pub fn rms_norm_rows(x: &[f32], w: &[f32], row_len: usize, eps: f32, gain_offset: f32) -> Vec<f32> {
    assert_eq!(w.len(), row_len);
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks(row_len) {
        let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / row_len as f32;
        let inv = (mean_sq + eps).sqrt().recip();
        for (v, g) in row.iter().zip(w) {
            out.push((gain_offset + g) * (v * inv));
        }
    }
    out
}

/// The gated delta rule, one token at a time.
///
/// Shapes: `q`, `k` are `[t_len, heads, dk]`; `v` is `[t_len, heads, dv]`;
/// `g` and `beta` are `[t_len, heads]`; `state` is `[heads, dk, dv]` and is
/// updated in place. Returns the output, `[t_len, heads, dv]`.
///
/// `q` and `k` arrive already repeated out to `heads` — the checkpoint has 16
/// key heads against 48 value heads and the reference expands them with
/// `repeat_interleave`, so head `h` of the recurrence uses key head `h / 3`.
/// Doing that expansion with a *stride* instead (head `h` -> key head
/// `h % 16`) also runs, and gives a different model.
///
/// The l2 normalization and the `1/sqrt(dk)` scale happen here, inside, because
/// the reference does them inside its kernel (`use_qk_l2norm_in_kernel=True`)
/// and the scale lands on `q` only.
pub fn gated_delta_rule(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    t_len: usize,
    heads: usize,
    dk: usize,
    dv: usize,
    eps: f32,
) -> Vec<f32> {
    assert_eq!(q.len(), t_len * heads * dk);
    assert_eq!(k.len(), t_len * heads * dk);
    assert_eq!(v.len(), t_len * heads * dv);
    assert_eq!(g.len(), t_len * heads);
    assert_eq!(beta.len(), t_len * heads);
    assert_eq!(state.len(), heads * dk * dv);

    let mut qn = q.to_vec();
    let mut kn = k.to_vec();
    l2norm_rows(&mut qn, dk, eps);
    l2norm_rows(&mut kn, dk, eps);
    let scale = (dk as f32).sqrt().recip();
    for value in qn.iter_mut() {
        *value *= scale;
    }

    let mut out = vec![0.0f32; t_len * heads * dv];
    for t in 0..t_len {
        for h in 0..heads {
            let s = &mut state[h * dk * dv..(h + 1) * dk * dv];
            let qh = &qn[(t * heads + h) * dk..(t * heads + h + 1) * dk];
            let kh = &kn[(t * heads + h) * dk..(t * heads + h + 1) * dk];
            let vh = &v[(t * heads + h) * dv..(t * heads + h + 1) * dv];
            let decay = g[t * heads + h].exp();
            let b = beta[t * heads + h];

            // S *= exp(g)
            for value in s.iter_mut() {
                *value *= decay;
            }
            // kv_mem = kᵀ S, contracting the key axis.
            let mut kv_mem = vec![0.0f32; dv];
            for (i, &ki) in kh.iter().enumerate() {
                if ki == 0.0 {
                    continue;
                }
                for (j, acc) in kv_mem.iter_mut().enumerate() {
                    *acc += s[i * dv + j] * ki;
                }
            }
            // delta = (v - kv_mem) * beta, then S += k ⊗ delta.
            let mut delta = vec![0.0f32; dv];
            for j in 0..dv {
                delta[j] = (vh[j] - kv_mem[j]) * b;
            }
            for (i, &ki) in kh.iter().enumerate() {
                if ki == 0.0 {
                    continue;
                }
                for (j, &dj) in delta.iter().enumerate() {
                    s[i * dv + j] += ki * dj;
                }
            }
            // o = qᵀ S, the same contraction as kv_mem but with q.
            let o = &mut out[(t * heads + h) * dv..(t * heads + h + 1) * dv];
            for (i, &qi) in qh.iter().enumerate() {
                if qi == 0.0 {
                    continue;
                }
                for (j, acc) in o.iter_mut().enumerate() {
                    *acc += s[i * dv + j] * qi;
                }
            }
        }
    }
    out
}

/// Cosine and sine tables for partial rotary embeddings.
///
/// `rotary_dim` is `int(head_dim * partial_rotary_factor)` — 64 of 256 on the
/// 27B — and the frequency exponent is divided by `rotary_dim`, **not** by
/// `head_dim`. So this is not the leading slice of the full-width table; it is a
/// different table that compresses the same frequency span into fewer dims.
/// Using the full-width table's prefix runs fine and degrades long-range
/// retrieval while everything nearby stays right.
///
/// Returns `(cos, sin)`, each `[positions.len(), rotary_dim]`, with the
/// half-duplicated layout that `rotate_half` expects: column `i` and column
/// `i + rotary_dim/2` carry the same frequency.
pub fn rope_tables(theta: f32, rotary_dim: usize, positions: &[u32]) -> (Vec<f32>, Vec<f32>) {
    assert!(rotary_dim.is_multiple_of(2));
    let half = rotary_dim / 2;
    let inv: Vec<f64> = (0..half)
        .map(|i| (theta as f64).powf(-((2 * i) as f64 / rotary_dim as f64)))
        .collect();
    let mut cos = vec![0.0f32; positions.len() * rotary_dim];
    let mut sin = vec![0.0f32; positions.len() * rotary_dim];
    for (t, &p) in positions.iter().enumerate() {
        for i in 0..half {
            // f64 for both the frequency and the angle, on purpose.
            //
            // The two obvious f32 formulations of `theta^(-2i/rot)` differ by
            // exactly one ulp — 1.2e-7 relative. At position 130000 that ulp
            // amplifies into a 2.5e-3 error in the cosine, because the angle is
            // then ~7.9e4 and f32's ulp there is 0.0078 rad. So the rope table
            // at this model's 262144-token context is simply not reproducible
            // across implementations in f32; the reference's own table is one
            // arbitrary point in a 2.5e-3-wide cloud.
            //
            // f64 is the implementation-independent answer and agrees with the
            // reference to 3e-7 at ordinary positions, so that is what this
            // computes. `the_far_position_gap_is_precision_not_layout` records
            // how far the reference drifts, and how much larger a real layout
            // error would be.
            let angle = p as f64 * inv[i];
            let (s, c) = (angle.sin() as f32, angle.cos() as f32);
            cos[t * rotary_dim + i] = c;
            cos[t * rotary_dim + i + half] = c;
            sin[t * rotary_dim + i] = s;
            sin[t * rotary_dim + i + half] = s;
        }
    }
    (cos, sin)
}

/// Apply partial rotary embeddings in place to `[t_len, heads, head_dim]`.
///
/// Only the first `rotary_dim` of each head rotates; the rest passes through
/// untouched. Pairing is `(i, i + rotary_dim/2)` — the non-interleaved
/// `rotate_half` convention, not the adjacent-pair one some models use.
pub fn apply_partial_rope(
    x: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    t_len: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) {
    assert!(rotary_dim <= head_dim);
    assert_eq!(cos.len(), t_len * rotary_dim);
    let half = rotary_dim / 2;
    for t in 0..t_len {
        for h in 0..heads {
            let base = (t * heads + h) * head_dim;
            for i in 0..half {
                let (a, b) = (x[base + i], x[base + i + half]);
                let (c0, s0) = (cos[t * rotary_dim + i], sin[t * rotary_dim + i]);
                let (c1, s1) = (cos[t * rotary_dim + i + half], sin[t * rotary_dim + i + half]);
                // rotate_half puts -x2 where x1 was, so the first half takes
                // `-b * sin` and the second takes `+a * sin`.
                x[base + i] = a * c0 - b * s0;
                x[base + i + half] = b * c1 + a * s1;
            }
        }
    }
}

/// Split `q_proj`'s output into the query and its gate.
///
/// The output is `[t_len, heads * 2 * head_dim]`, and the reference reads it as
/// `[t_len, heads, 2 * head_dim]` before splitting the last axis. So within one
/// head's `2 * head_dim` values the query comes first and the gate second — the
/// two interleave per head. Reading it as `[all queries | all gates]` is the
/// other plausible layout, it runs to completion, and it is wrong.
pub fn split_q_and_gate(
    qg: &[f32],
    t_len: usize,
    heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(qg.len(), t_len * heads * 2 * head_dim);
    let mut q = Vec::with_capacity(t_len * heads * head_dim);
    let mut gate = Vec::with_capacity(t_len * heads * head_dim);
    for t in 0..t_len {
        for h in 0..heads {
            let base = (t * heads + h) * 2 * head_dim;
            q.extend_from_slice(&qg[base..base + head_dim]);
            gate.extend_from_slice(&qg[base + head_dim..base + 2 * head_dim]);
        }
    }
    (q, gate)
}

/// Causal attention with grouped key/value heads, in `f32`.
///
/// Returns `[t_len, heads * head_dim]`. `positions` gives each token's absolute
/// position so a decode step attends over history it cannot see in `k`; here it
/// is only used for the causal mask, and `k`/`v` are the full history.
///
/// The `1/sqrt(head_dim)` lands on the scores, which is `Qwen3_5Attention`'s
/// `self.scaling = self.head_dim**-0.5` passed into `eager_attention_forward` as
/// `matmul(q, kᵀ) * scaling`. Scaling `q` instead is the same function up to
/// rounding and needs no test; `1/head_dim` or `1/sqrt(d_model)` are not, and
/// `check_gated_attention_against_reference` in
/// `tools/capture_qwen35_layers.py` measures the first of those at 22% of peak.
/// The key expansion is `repeat_kv`, i.e. `repeat_interleave` — head `h` uses kv
/// head `h / group`, not `h % kv_heads`.
pub fn causal_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t_len: usize,
    kv_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let group = heads / kv_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut out = vec![0.0f32; t_len * heads * head_dim];
    let mut scores = vec![0.0f32; kv_len];
    for t in 0..t_len {
        // The query at index `t` sits at absolute position
        // `kv_len - t_len + t`, so it may attend up to and including that.
        let limit = kv_len - t_len + t;
        for h in 0..heads {
            let kvh = h / group;
            let qh = &q[(t * heads + h) * head_dim..(t * heads + h + 1) * head_dim];
            let mut max = f32::NEG_INFINITY;
            for (s, score) in scores.iter_mut().enumerate().take(limit + 1) {
                let kh = &k[(s * kv_heads + kvh) * head_dim..(s * kv_heads + kvh + 1) * head_dim];
                let dot: f32 = qh.iter().zip(kh).map(|(a, b)| a * b).sum();
                *score = dot * scale;
                max = max.max(*score);
            }
            let mut denom = 0.0f32;
            for score in scores.iter_mut().take(limit + 1) {
                *score = (*score - max).exp();
                denom += *score;
            }
            let o = &mut out[(t * heads + h) * head_dim..(t * heads + h + 1) * head_dim];
            for (s, &p) in scores.iter().enumerate().take(limit + 1) {
                let w = p / denom;
                let vh = &v[(s * kv_heads + kvh) * head_dim..(s * kv_heads + kvh + 1) * head_dim];
                for (acc, &val) in o.iter_mut().zip(vh) {
                    *acc += w * val;
                }
            }
        }
    }
    out
}
