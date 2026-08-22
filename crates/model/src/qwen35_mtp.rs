//! Host-side reference for the Qwen3.5 MTP (multi-token prediction) head.
//!
//! Same contract as [`crate::qwen35`]: plain `f32`, explicit shapes, one loop
//! per thing the reference does, written to be *read* as the spec. See
//! `notes/qwen3.5-mtp.md` for the prose version and
//! `tools/capture_qwen35_mtp.py` for the oracle this is pinned against.
//!
//! What the head is
//! ----------------
//! One full-attention decoder layer — the same block type as the target model's
//! layers 3, 7, …, 63, output gate and partial rope included — wrapped in four
//! tensors of glue:
//!
//! ```text
//! e = rms_norm(embedding_of(next_token),  pre_fc_norm_embedding)
//! h = rms_norm(target_final_hidden_state, pre_fc_norm_hidden)
//! x = fc @ concat([e, h])            // fc is [5120, 10240]
//! x = full_attention_decoder_layer(x)
//! out = rms_norm(x, mtp.norm)
//! logits = lm_head @ out             // the *target model's* lm_head
//! ```
//!
//! Three things in that block have a second reading that runs to completion:
//!
//! 1. **The concat order.** `[e, h]` and `[h, e]` are the same shape and `fc`
//!    eats either. Getting it backwards costs nothing at runtime and produces
//!    drafts that are grammatical and wrong, so the acceptance rate quietly
//!    collapses towards chance and speculative decoding becomes a slowdown that
//!    looks like a scheduling problem.
//!
//! 2. **Which norm goes on which input.** Also same shape, also silent. On this
//!    checkpoint the two weights are genuinely different — `pre_fc_norm_hidden`
//!    spans `[-0.375, +0.455]` while `pre_fc_norm_embedding` is negative
//!    everywhere, `[-0.750, -0.186]` — so swapping them is a real change and
//!    still not a crash.
//!
//! 3. **`Qwen3_5RMSNorm` is `(1 + weight) * normalized`, not `weight *`.** This
//!    is the one that would have been inherited by accident, because
//!    [`crate::qwen35::rms_norm_rows`] implements the plain form — correctly,
//!    for the one norm in this model that wants it, the *gated* output norm
//!    inside GatedDeltaNet (`Qwen3_5RMSNormGated`, whose weight is
//!    one-initialized and sits at 0.87 ± 0.07 in the checkpoint). Every other
//!    norm in Qwen3.5 — `input_layernorm`, `post_attention_layernorm`, `q_norm`,
//!    `k_norm`, `model.norm`, and all four of the MTP head's — is
//!    `Qwen3_5RMSNorm`, whose weight is *zero*-initialized and stored as a
//!    deviation from unity. Reading a deviation as a gain scales
//!    `pre_fc_norm_embedding` by roughly -0.46 instead of +0.54: sign flipped,
//!    magnitude in the right ballpark, no NaN, no crash.
//!
//!    Both `transformers.models.qwen3_5.Qwen3_5RMSNorm` and vLLM's
//!    `GemmaRMSNorm` (which vLLM aliases as `Qwen3_5RMSNorm`) agree on the
//!    offset form to the bit; the capture runs both and refuses to write if they
//!    do not.
//!
//! What the head is *not*: it is not a linear-attention layer. The checkpoint
//! settles this without any interpretation — `mtp.layers.0.*` carries
//! `self_attn.{q,k,v,o}_proj` and `self_attn.{q,k}_norm` and carries no
//! `conv1d`, no `A_log`, no `dt_bias`, no `in_proj_*` — and vLLM's
//! `Qwen3_5MultiTokenPredictor` constructs it with a literal
//! `layer_type="full_attention"`. Which means the head owns one extra
//! full-attention KV cache and touches no recurrent state at all. That is the
//! single most consequential fact for scheduling, so it gets checked twice.

use crate::qwen35::{
    apply_partial_rope, causal_attention, sigmoid, silu, split_q_and_gate,
};

/// RMSNorm with a **unit-offset** learned gain: `(1 + w) * x / rms(x)`.
///
/// This is `Qwen3_5RMSNorm`. Not to be confused with
/// [`crate::qwen35::rms_norm_rows`], which is `w * x / rms(x)` and is the right
/// formula for exactly one norm in this architecture — `Qwen3_5RMSNormGated`,
/// the GatedDeltaNet output norm. The two classes differ only in this `1.0 +`
/// and in how their weights are initialized, so a checkpoint tells you which is
/// which: a `Qwen3_5RMSNorm` weight sits near zero, a `Qwen3_5RMSNormGated`
/// weight sits near one.
///
/// `eps` is added to the mean square, inside the square root — not to the rms.
pub fn rms_norm_offset_rows(x: &[f32], w: &[f32], row_len: usize, eps: f32) -> Vec<f32> {
    assert_eq!(w.len(), row_len);
    assert!(x.len().is_multiple_of(row_len));
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks(row_len) {
        let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / row_len as f32;
        let inv = (mean_sq + eps).sqrt().recip();
        for (v, g) in row.iter().zip(w) {
            out.push((1.0 + g) * (v * inv));
        }
    }
    out
}

/// `y[t, o] = sum_i x[t, i] * w[o, i]`, with no bias.
///
/// Row-major `[t_len, in_dim]` times row-major `[out_dim, in_dim]`, which is how
/// every weight in this checkpoint is stored: output-major, so a row of `w` is
/// contiguous and is the thing that produces one output column.
pub fn linear(x: &[f32], w: &[f32], t_len: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    assert_eq!(x.len(), t_len * in_dim);
    assert_eq!(w.len(), out_dim * in_dim);
    let mut out = vec![0.0f32; t_len * out_dim];
    for t in 0..t_len {
        let xt = &x[t * in_dim..(t + 1) * in_dim];
        for o in 0..out_dim {
            let wo = &w[o * in_dim..(o + 1) * in_dim];
            out[t * out_dim + o] = xt.iter().zip(wo).map(|(a, b)| a * b).sum();
        }
    }
    out
}

/// SwiGLU: `down(silu(gate(x)) * up(x))`.
///
/// `silu` lands on the `gate_proj` branch and not on `up_proj`; the mirror image
/// runs and gives a different model. The activation is `silu` and not `gelu`
/// because `hidden_act` says `silu`.
pub fn swiglu_mlp(
    x: &[f32],
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    t_len: usize,
    d_model: usize,
    d_ff: usize,
) -> Vec<f32> {
    let g = linear(x, gate_w, t_len, d_model, d_ff);
    let u = linear(x, up_w, t_len, d_model, d_ff);
    let mut h = vec![0.0f32; t_len * d_ff];
    for i in 0..h.len() {
        h[i] = silu(g[i]) * u[i];
    }
    linear(&h, down_w, t_len, d_ff, d_model)
}

/// Shapes and scalars for one Qwen3.5 full-attention block. On the 27B:
/// `d_model = 5120`, `heads = 24`, `kv_heads = 4`, `head_dim = 256`,
/// `rotary_dim = 64`, `d_ff = 17408`, `rope_theta = 1e7`, `eps = 1e-6`.
#[derive(Clone, Copy, Debug)]
pub struct BlockDims {
    pub d_model: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub d_ff: usize,
    pub eps: f32,
}

impl BlockDims {
    /// `heads * head_dim` — 6144 on the 27B, and *not* `d_model`. `o_proj` is
    /// `[d_model, d_attn]`, so the attention width and the residual width are
    /// different numbers and confusing them is a shape error rather than a
    /// silent one. Small mercy.
    pub fn d_attn(&self) -> usize {
        self.heads * self.head_dim
    }
    /// `kv_heads * head_dim` — 1024 on the 27B.
    pub fn d_kv(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

/// The weights of one full-attention decoder layer, in checkpoint layout.
pub struct BlockWeights<'a> {
    /// `[d_model]`, a `Qwen3_5RMSNorm` deviation.
    pub input_layernorm: &'a [f32],
    /// `[heads * 2 * head_dim, d_model]` — q and its gate, interleaved per head.
    pub q_proj: &'a [f32],
    /// `[kv_heads * head_dim, d_model]`.
    pub k_proj: &'a [f32],
    /// `[kv_heads * head_dim, d_model]`.
    pub v_proj: &'a [f32],
    /// `[d_model, heads * head_dim]`.
    pub o_proj: &'a [f32],
    /// `[head_dim]`, per-head, a deviation.
    pub q_norm: &'a [f32],
    /// `[head_dim]`, per-head, a deviation.
    pub k_norm: &'a [f32],
    /// `[d_model]`, a deviation.
    pub post_attention_layernorm: &'a [f32],
    /// `[d_ff, d_model]`.
    pub gate_proj: &'a [f32],
    /// `[d_ff, d_model]`.
    pub up_proj: &'a [f32],
    /// `[d_model, d_ff]`.
    pub down_proj: &'a [f32],
}

/// Everything the head owns. The embedding and the `lm_head` are deliberately
/// absent: the head borrows the target model's, which is what
/// `mtp_use_dedicated_embeddings = False` means and what the checkpoint shows by
/// shipping no `mtp.embed_tokens` and no `mtp.lm_head`.
pub struct MtpWeights<'a> {
    /// `[d_model]`. Applies to the **token embedding**.
    pub pre_fc_norm_embedding: &'a [f32],
    /// `[d_model]`. Applies to the **target model's final hidden state**.
    pub pre_fc_norm_hidden: &'a [f32],
    /// `[d_model, 2 * d_model]`. Columns `0..d_model` multiply the embedding
    /// half; columns `d_model..2*d_model` multiply the hidden half.
    pub fc: &'a [f32],
    pub layer: BlockWeights<'a>,
    /// `[d_model]`, the head's own final norm — a separate tensor from the
    /// target model's `model.language_model.norm`.
    pub norm: &'a [f32],
}

/// The intermediate tensors, in the order the head produces them. Named to match
/// `tools/capture_qwen35_mtp.py`'s dumps so the test reads as a diff.
pub struct MtpStages {
    /// `[t_len, d_model]`, `rms_norm(embedding, pre_fc_norm_embedding)`.
    pub emb_normed: Vec<f32>,
    /// `[t_len, d_model]`, `rms_norm(hidden, pre_fc_norm_hidden)`.
    pub hidden_normed: Vec<f32>,
    /// `[t_len, d_model]`, the fused representation entering the decoder layer.
    pub fc_out: Vec<f32>,
    /// `[t_len, d_model]`, the decoder layer's output including both residuals.
    pub layer_out: Vec<f32>,
    /// `[t_len, d_model]`, what the target model's `lm_head` consumes.
    pub output: Vec<f32>,
}

/// Fuse the next token's embedding with the target model's final hidden state.
///
/// This is the whole MTP-specific idea and the whole place it can go wrong
/// silently, so it is one small function with one big comment.
///
/// * `embeds` is `[t_len, d_model]`: at slot `i`, the embedding of token
///   `t_{i+1}` — the token *after* the one whose hidden state sits in `hidden`.
/// * `hidden` is `[t_len, d_model]`: at slot `i`, the target model's hidden
///   state for token `t_i`, taken **after** `model.language_model.norm`.
/// * The concat is `[normalized embedding, normalized hidden]`, embedding first.
///
/// Evidence for the order, since "I read the code" is what the nine failed A/Bs
/// had: `tools/capture_qwen35_mtp.py` parses vLLM's
/// `Qwen3_5MultiTokenPredictor.forward` into a canonical string and requires it
/// to be exactly
/// `cat[pre_fc_norm_embedding(EMBEDDING)|pre_fc_norm_hidden(TARGET_HIDDEN)]@-1`,
/// does the same to `Qwen3NextMultiTokenPredictor.forward` as a second witness,
/// and then measures on real text that this composition drafts the target
/// model's own next token far more often than the swapped one does.
pub fn fuse_embedding_and_hidden(
    embeds: &[f32],
    hidden: &[f32],
    w: &MtpWeights,
    t_len: usize,
    d_model: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(embeds.len(), t_len * d_model);
    assert_eq!(hidden.len(), t_len * d_model);
    assert_eq!(w.fc.len(), d_model * 2 * d_model);

    let e = rms_norm_offset_rows(embeds, w.pre_fc_norm_embedding, d_model, eps);
    let h = rms_norm_offset_rows(hidden, w.pre_fc_norm_hidden, d_model, eps);
    let cat = concat_embedding_then_hidden(&e, &h, t_len, d_model);
    let fused = linear(&cat, w.fc, t_len, 2 * d_model, d_model);
    (e, h, fused)
}

/// `[normalized embedding | normalized hidden]`, row by row — the ordering
/// decision, on its own, so that a test can exercise *this function* rather than
/// rebuild the concatenation alongside it and end up checking its own copy.
///
/// A real kernel will not materialize this at all; it will compute
/// `fc[:, :d] @ e + fc[:, d:] @ h` and skip the copy entirely. That is fine and
/// equivalent, but writing it that way here would bury the ordering inside an
/// index expression instead of leaving it on the surface where it can be read.
pub fn concat_embedding_then_hidden(
    e: &[f32],
    h: &[f32],
    t_len: usize,
    d_model: usize,
) -> Vec<f32> {
    assert_eq!(e.len(), t_len * d_model);
    assert_eq!(h.len(), t_len * d_model);
    let mut cat = vec![0.0f32; t_len * 2 * d_model];
    for t in 0..t_len {
        cat[t * 2 * d_model..t * 2 * d_model + d_model]
            .copy_from_slice(&e[t * d_model..(t + 1) * d_model]);
        cat[t * 2 * d_model + d_model..(t + 1) * 2 * d_model]
            .copy_from_slice(&h[t * d_model..(t + 1) * d_model]);
    }
    cat
}

/// One Qwen3.5 full-attention decoder layer, pre-norm with two residuals.
///
/// `positions[i]` is the absolute position of slot `i`, used only for rope and
/// for the causal mask. `k` and `v` here are computed from `x` alone, so this is
/// the whole-sequence form; a decode step would splice the cached history in
/// before calling [`causal_attention`].
///
/// The order of operations is the part worth reading:
///
/// ```text
/// h  = rms_norm(x, input_layernorm)
/// q|gate = q_proj(h)                   // interleaved per head, q first
/// q  = rope(rms_norm(q, q_norm))       // norm before rope
/// k  = rope(rms_norm(k, k_norm))
/// a  = attention(q, k, v) * sigmoid(gate)   // gate before o_proj, sigmoid not silu
/// x  = x + o_proj(a)
/// x  = x + mlp(rms_norm(x, post_attention_layernorm))
/// ```
///
/// `sigmoid` and not `silu`: `config.output_gate_type` says `"swish"` and the
/// implementation never reads it. See `notes/qwen3.5-architecture.md`.
pub fn full_attention_layer(
    x: &[f32],
    w: &BlockWeights,
    cos: &[f32],
    sin: &[f32],
    t_len: usize,
    dims: BlockDims,
) -> Vec<f32> {
    let (d, nh, nkv, hd) = (dims.d_model, dims.heads, dims.kv_heads, dims.head_dim);
    assert_eq!(x.len(), t_len * d);

    let h = rms_norm_offset_rows(x, w.input_layernorm, d, dims.eps);

    let qg = linear(&h, w.q_proj, t_len, d, nh * 2 * hd);
    let (q, gate) = split_q_and_gate(&qg, t_len, nh, hd);
    let k = linear(&h, w.k_proj, t_len, d, dims.d_kv());
    let v = linear(&h, w.v_proj, t_len, d, dims.d_kv());

    // The q/k norms are per *head*, width head_dim, and they are the offset
    // form like every other Qwen3_5RMSNorm. `rms_norm_offset_rows` walks rows of
    // `head_dim`, which is exactly one head of one token, so no reshape is
    // needed — the tensors are already `[t_len, heads, head_dim]` row-major.
    let mut q = rms_norm_offset_rows(&q, w.q_norm, hd, dims.eps);
    let mut k = rms_norm_offset_rows(&k, w.k_norm, hd, dims.eps);

    apply_partial_rope(&mut q, cos, sin, t_len, nh, hd, dims.rotary_dim);
    apply_partial_rope(&mut k, cos, sin, t_len, nkv, hd, dims.rotary_dim);

    let mut ctx = causal_attention(&q, &k, &v, t_len, t_len, nh, nkv, hd);
    for (c, g) in ctx.iter_mut().zip(&gate) {
        *c *= sigmoid(*g);
    }

    let attn = linear(&ctx, w.o_proj, t_len, dims.d_attn(), d);
    let mut resid: Vec<f32> = x.iter().zip(&attn).map(|(a, b)| a + b).collect();

    let normed = rms_norm_offset_rows(&resid, w.post_attention_layernorm, d, dims.eps);
    let mlp = swiglu_mlp(
        &normed,
        w.gate_proj,
        w.up_proj,
        w.down_proj,
        t_len,
        d,
        dims.d_ff,
    );
    for (r, m) in resid.iter_mut().zip(&mlp) {
        *r += m;
    }
    resid
}

/// The whole head: fuse, one decoder layer, final norm.
///
/// The caller supplies the rope tables so that the position convention stays
/// visible at the call site. It matters and it is not the obvious one: slot `i`
/// carries the hidden state of token `t_i` and the embedding of token `t_{i+1}`,
/// and the position it uses is **`i`, the hidden state's position**, not `i + 1`,
/// the embedded token's. vLLM passes the target model's positions to the drafter
/// unchanged (`llm_base_proposer.set_inputs_first_pass` rotates the ids and
/// leaves `target_positions` alone) and then increments by one per subsequent
/// draft step. So the drafter's KV cache slot `p` holds the pair
/// `(h_p, emb(t_{p+1}))` at rope position `p`, one behind the target's.
pub fn mtp_head(
    embeds: &[f32],
    hidden: &[f32],
    w: &MtpWeights,
    cos: &[f32],
    sin: &[f32],
    t_len: usize,
    dims: BlockDims,
) -> MtpStages {
    let (emb_normed, hidden_normed, fc_out) =
        fuse_embedding_and_hidden(embeds, hidden, w, t_len, dims.d_model, dims.eps);
    let layer_out = full_attention_layer(&fc_out, &w.layer, cos, sin, t_len, dims);
    let output = rms_norm_offset_rows(&layer_out, w.norm, dims.d_model, dims.eps);
    MtpStages {
        emb_normed,
        hidden_normed,
        fc_out,
        layer_out,
        output,
    }
}

// ------------------------------------------------------------- acceptance rule

/// What a verification step emitted, and how much of the draft survived.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Accepted {
    /// The tokens to append. Always at least one — the rule cannot make a step
    /// produce nothing, which is what keeps speculation from being able to
    /// livelock.
    pub tokens: Vec<u32>,
    /// How many *draft* tokens were accepted, in `0..=draft.len()`. The number
    /// of state and KV entries to keep is `accepted + 1`: the accepted drafts
    /// plus the one token the target itself produced.
    pub accepted: usize,
}

/// Greedy acceptance, as vLLM's `rejection_greedy_sample_kernel` does it.
///
/// `draft[j]` is the head's proposal for slot `j`; `target_argmax[j]` is the
/// target model's own argmax at the same slot, from the same verification
/// forward pass. `target_argmax` must be one longer than `draft`: its last entry
/// is the *bonus* token, the target's prediction from the position after the
/// whole draft, which is only usable if every draft token was accepted.
///
/// The rule:
///
/// * scan left to right; while `draft[j] == target_argmax[j]`, emit it;
/// * at the first mismatch, emit `target_argmax[j]` and stop;
/// * if nothing mismatched, emit the bonus token too.
///
/// So a step emits between 1 and `k + 1` tokens, and the sequence it produces is
/// bit-identical to what unspeculated greedy decoding would have produced. That
/// exactness is the property worth protecting: it means speculative decoding can
/// be switched on and off without changing outputs, which in turn means a
/// regression in the draft head shows up as a throughput change and never as a
/// quality change.
pub fn accept_greedy(draft: &[u32], target_argmax: &[u32]) -> Accepted {
    assert_eq!(
        target_argmax.len(),
        draft.len() + 1,
        "the verification pass produces one more prediction than there are \
         draft tokens; the extra one is the bonus token"
    );
    let mut tokens = Vec::with_capacity(draft.len() + 1);
    for (j, &d) in draft.iter().enumerate() {
        if d != target_argmax[j] {
            tokens.push(target_argmax[j]);
            return Accepted {
                tokens,
                accepted: j,
            };
        }
        tokens.push(d);
    }
    tokens.push(target_argmax[draft.len()]);
    Accepted {
        tokens,
        accepted: draft.len(),
    }
}

/// Stochastic acceptance, as vLLM's `rejection_random_sample_kernel` does it.
///
/// Accept `draft[j]` when `p_target(draft[j]) / p_draft(draft[j]) >= u_j` for a
/// fresh `u_j ~ U(0, 1)`; on rejection emit a token drawn from the normalized
/// residual `max(0, p_target - p_draft)` instead, which is what makes the whole
/// scheme exactly sample from the target distribution. `recovered[j]` is that
/// draw, supplied by the caller because the sampling is the caller's business;
/// this function is only the accept/reject bookkeeping.
///
/// A zero draft probability is rejected rather than dividing — vLLM guards the
/// same way, and the ratio is `+inf` there, which would accept a token the draft
/// model considers impossible.
pub fn accept_stochastic(
    draft: &[u32],
    p_target: &[f32],
    p_draft: &[f32],
    uniform: &[f32],
    recovered: &[u32],
    bonus: u32,
) -> Accepted {
    let k = draft.len();
    assert_eq!(p_target.len(), k);
    assert_eq!(p_draft.len(), k);
    assert_eq!(uniform.len(), k);
    assert_eq!(recovered.len(), k);
    let mut tokens = Vec::with_capacity(k + 1);
    for j in 0..k {
        let ok = p_draft[j] > 0.0 && p_target[j] / p_draft[j] >= uniform[j];
        if !ok {
            tokens.push(recovered[j]);
            return Accepted {
                tokens,
                accepted: j,
            };
        }
        tokens.push(draft[j]);
    }
    tokens.push(bonus);
    Accepted { tokens, accepted: k }
}

// --------------------------------------------- rolling back a recurrent state

/// The per-token record a GatedDeltaNet layer needs in order to undo, or rather
/// to *not do*, a rejected token's state update.
///
/// The problem this solves: the recurrent state `S[h]` is `[dk, dv]` and is
/// updated in place, so a verification pass over `k + 1` candidate tokens has
/// already folded all `k + 1` of them into `S` by the time the logits come back
/// and say only `n` were accepted. Unlike a KV cache, there is nothing to
/// truncate — the rejected tokens are mixed into the same numbers as the
/// accepted ones.
///
/// The naive fix is to snapshot `S` before the step and restore it on rejection,
/// which costs 3 MiB per sequence per linear layer, 147 MiB per sequence over
/// all 48, copied twice per decode step. That is not the memory that hurts, it
/// is the bandwidth: 294 MiB of copies per sequence per step against a step that
/// is already bandwidth-bound reading 27 GB of weights for the whole batch. At
/// batch 32 the copies are 9 GiB per step. Prohibitive.
///
/// What this does instead: the recurrence's per-token update is exactly
///
/// ```text
/// S <- S * exp(g_t)          // scalar per head
/// S <- S + k_t (x) delta_t   // rank one per head
/// ```
///
/// and `k_t`, `delta_t`, `g_t` are all computed during the forward pass anyway.
/// `k_t` is `[heads, dk]`, `delta_t` is `[heads, dv]`, `g_t` is `[heads]` — for
/// the 27B that is 48x128 + 48x128 + 48 floats, about 48 KiB per token per
/// layer, against 3 MiB for the state. So journal those, run the recurrence
/// *without* committing to `S`, and afterwards replay only the accepted prefix
/// into `S`. Cost: 147 MiB of state (unchanged, no snapshot) plus
/// `(k + 1) * 48 KiB * 48` of journal — 6.9 MiB per sequence at `k = 2` — and a
/// replay whose arithmetic is `n` rank-one updates per head, which is the same
/// work the forward pass already did once.
///
/// Why not invert the recurrence and step backwards: because it is numerically
/// hopeless. With `k` unit-normalized, recovering the pre-update state needs a
/// division by `1 - beta_t`, and `beta_t = sigmoid(b_t)` sits arbitrarily close
/// to 1. One token near `beta = 0.999` destroys three digits.
#[derive(Clone, Debug)]
pub struct DeltaJournalEntry {
    /// `[heads, dk]`, the l2-normalized key actually used by the recurrence.
    pub k: Vec<f32>,
    /// `[heads, dv]`, `(v_t - k_t^T S) * beta_t`, as computed in the forward.
    pub delta: Vec<f32>,
    /// `[heads]`, the log decay — non-positive.
    pub g: Vec<f32>,
}

/// Apply the first `accepted` journal entries to `state`, in order.
///
/// `state` is `[heads, dk, dv]` and holds the value from *before* the
/// verification step. After this it holds the value it would have had if only
/// the accepted tokens had ever been fed to the layer — exactly, not
/// approximately, because the journal stores the update terms rather than trying
/// to reconstruct them.
pub fn replay_accepted(
    state: &mut [f32],
    journal: &[DeltaJournalEntry],
    accepted: usize,
    heads: usize,
    dk: usize,
    dv: usize,
) {
    assert_eq!(state.len(), heads * dk * dv);
    assert!(
        accepted <= journal.len(),
        "cannot replay {accepted} tokens from a journal of {}",
        journal.len()
    );
    for entry in &journal[..accepted] {
        assert_eq!(entry.k.len(), heads * dk);
        assert_eq!(entry.delta.len(), heads * dv);
        assert_eq!(entry.g.len(), heads);
        for h in 0..heads {
            let s = &mut state[h * dk * dv..(h + 1) * dk * dv];
            let decay = entry.g[h].exp();
            for value in s.iter_mut() {
                *value *= decay;
            }
            let kh = &entry.k[h * dk..(h + 1) * dk];
            let dh = &entry.delta[h * dv..(h + 1) * dv];
            for (i, &ki) in kh.iter().enumerate() {
                if ki == 0.0 {
                    continue;
                }
                for (j, &dj) in dh.iter().enumerate() {
                    s[i * dv + j] += ki * dj;
                }
            }
        }
    }
}
