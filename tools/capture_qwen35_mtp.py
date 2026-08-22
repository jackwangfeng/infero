#!/usr/bin/env python3
"""Capture the Qwen3.5 MTP (multi-token prediction) head, stage by stage, from a
real checkpoint, and refuse to write anything unless the reference
implementations agree with what is being captured.

Why this file is shaped the way it is
-------------------------------------
The MTP head is four tensors of glue (`fc`, `pre_fc_norm_embedding`,
`pre_fc_norm_hidden`, `norm`) wrapped around one ordinary full-attention decoder
layer. Every one of those glue decisions has a second reading that runs to
completion and produces plausible-but-wrong drafts:

  * `torch.cat([embedding, hidden])` vs `cat([hidden, embedding])` — same shape,
    same dtype, `fc` consumes either. Wrong drafts, no crash.
  * `pre_fc_norm_embedding` on the embedding vs on the hidden state — same
    shape. Wrong drafts, no crash.
  * `Qwen3_5RMSNorm` is the **`(1 + weight)`** variant, not `weight *`. On this
    checkpoint `mtp.pre_fc_norm_embedding.weight` is negative everywhere
    (mean -0.46, max -0.19), so the plain reading flips the sign of every
    embedding dimension and halves it. Runs fine.

So this script does three separate kinds of checking before it writes a byte:

1. **Mechanical source reads of the reference.** `transformers` does not
   implement the MTP head at all (`modeling_qwen3_5.py` carries
   `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`), so the reference is
   vLLM's `Qwen3_5MultiTokenPredictor`. Rather than transcribe its `forward`
   and hope, we parse its AST and reduce it to a canonical string describing
   what reaches `self.fc`. The same reduction is applied to
   `Qwen3NextMultiTokenPredictor` as an independent witness, because Qwen3-Next
   is the same mechanism and an accidental edit to one is unlikely to hit both.

2. **Numeric agreement between two independent RMSNorm implementations.**
   transformers' `Qwen3_5RMSNorm` and vLLM's `GemmaRMSNorm` are separate code,
   and both are run here on a real checkpoint weight and compared against the
   f32 formula this capture uses — and against the plain `weight *` reading,
   which must *disagree*, or the check proves nothing.

3. **An end-to-end behavioural check that needs no reference at all.** The head
   exists to predict token t+2 from (hidden state of t, embedding of t+1). So
   run the whole 64-layer target model on real text, and count how often the
   head's argmax equals the target model's own argmax one position later. A
   correct composition agrees most of the time. Every wrong composition —
   swapped concat halves, swapped norms, plain-`w` norm, pre-final-norm hidden
   state — is also measured here, and must score worse. That is the check that
   cannot be satisfied by a self-consistent misreading.

The activation-scale trap
------------------------
`tools/capture_qwen35_layers.py` records why random input is not good enough:
activations land orders of magnitude below `rms_norm_eps`, every RMS denominator
becomes `sqrt(eps)`, and the norm degenerates into a constant scale that cannot
tell one formulation from another. The MTP head has *four* RMSNorms in it, so it
is more exposed to this than either block type, and it consumes the target
model's **final** hidden state — there is no shortcut prefix. This script
therefore runs all `num_hidden_layers` layers for real, streaming one layer at a
time so the peak memory is one layer rather than sixty-four, and then asserts
that for every tensor entering a norm, and every row of it, `eps` accounts for
less than `MAX_EPS_SHARE` of the RMS denominator. `--prefix-layers` can truncate
the stack for a fast smoke run, but the manifest records that it was truncated
and the Rust test refuses such a capture.

    python3 capture_qwen35_mtp.py <model-dir> <out-dir> [--tokens 16]
"""

import argparse
import ast
import inspect
import itertools
import json
import os
import sys
import textwrap

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture_qwen35_layers import load_f32, read_index  # noqa: E402

# How much of an RMSNorm's denominator `eps` is allowed to account for.
#
# `mean(x^2)` compared against `eps` is the quantity that matters, but the
# readable form of it is `eps / (mean(x^2) + eps)`: the fraction of the
# denominator that comes from the epsilon rather than from the data. At 1.0 the
# norm is a constant `1/sqrt(eps)` scale and the capture blesses anything; the
# existing GatedDeltaNet capture found itself at essentially 1.0 with random
# input. Anything under a percent leaves the norm doing real, per-token work.
#
# This threshold is looser than it looks like it should be for one honest
# reason: the raw embedding rows of this checkpoint have RMS ~0.014, some 60x
# below the final hidden state, so `mean(x^2)` there is 2e-4 no matter how real
# the text is. That is the model, not a defect in the capture — and 0.5% of the
# denominator is not domination.
MAX_EPS_SHARE = 0.01

# The head must at least beat this agreement rate against the target model's own
# next-token argmax. vLLM measures a mean acceptance length of 2.0 on this
# checkpoint, which for a two-token draft implies a per-token acceptance around
# 0.62; a first-token greedy agreement well under 0.4 means the composition is
# wrong, not that the head is weak.
MIN_TOP1_AGREEMENT = 0.40

# What vLLM's MTP forward must reduce to. Written out rather than derived so that
# a change on either side is a loud failure.
EXPECTED_FC_INPUT = (
    "cat[pre_fc_norm_embedding(EMBEDDING)|pre_fc_norm_hidden(TARGET_HIDDEN)]@-1"
)

# The reduction resolves local rebindings, so the embedding argument comes back
# as whichever expression produced it. Collapse the two spellings the references
# use — `inputs_embeds` when the caller passed embeddings in, and
# `self.embed_input_ids(input_ids)` when it passed ids — onto one name, because
# the decision under test is *which* of the two inputs each norm and each concat
# half gets, not how the embedding was spelled.
_CANONICAL = [
    ("embed_input_ids(input_ids)", "EMBEDDING"),
    ("get_input_embeddings(input_ids)", "EMBEDDING"),
    ("inputs_embeds", "EMBEDDING"),
    ("hidden_states", "TARGET_HIDDEN"),
]


# --------------------------------------------------------------- f32 formulas
#
# These two are the only arithmetic this file transcribes, and both are checked
# against the reference below before use.


def rms_norm_offset(x, w, eps):
    """`Qwen3_5RMSNorm`: normalize, then scale by **(1 + weight)**.

    The offset is the whole point. These weights are zero-initialized and stored
    as deviations from unity, so `weight * x` is a different function — and on
    this checkpoint a sign-flipping one for `pre_fc_norm_embedding`.
    """
    x = x.float()
    var = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(var + eps)) * (1.0 + w.float())


def rms_norm_plain(x, w, eps):
    """The wrong reading, kept so the checks can show they discriminate.

    This *is* the right formula for `Qwen3_5RMSNormGated` (the GatedDeltaNet
    output norm, which is one-initialized), which is exactly why the confusion
    is available.
    """
    x = x.float()
    var = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(var + eps)) * w.float()


# ------------------------------------------------------- reference AST reading


def _describe(node, env):
    """Reduce an expression to a canonical string, resolving local variables."""
    if isinstance(node, ast.Name):
        return env.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Call):
        fn = node.func
        args = [_describe(a, env) for a in node.args]
        if isinstance(fn, ast.Attribute) and fn.attr == "cat":
            # torch.cat([a, b], dim=d) -> cat[a|b]@d
            elts = node.args[0].elts if isinstance(node.args[0], (ast.List, ast.Tuple)) else []
            dim = "?"
            for kw in node.keywords:
                if kw.arg == "dim":
                    dim = _describe(kw.value, env)
            joined = "|".join(_describe(e, env) for e in elts)
            return f"cat[{joined}]@{dim}"
        if isinstance(fn, ast.Attribute):
            return f"{fn.attr}({', '.join(args)})"
        return f"{_describe(fn, env)}({', '.join(args)})"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return "-" + _describe(node.operand, env)
    if isinstance(node, ast.Subscript):
        return f"{_describe(node.value, env)}[..]"
    if isinstance(node, ast.Tuple):
        return "(" + ", ".join(_describe(e, env) for e in node.elts) + ")"
    return type(node).__name__


def _walk_assignments(body, env, seen):
    """Interpret a statement list in order, recording what each name holds.

    Order matters and `ast.walk` does not preserve it: vLLM's forward rebinds
    `hidden_states` three times in four lines, so an unordered scan cannot tell
    which binding `torch.cat` saw. For the same reason `seen` accumulates every
    descriptor ever produced: `hidden_states = self.fc(...)` is overwritten two
    lines later, so the final environment no longer mentions `fc` at all.
    """
    for stmt in body:
        if isinstance(stmt, ast.Assign):
            desc = _describe(stmt.value, env)
            seen.append(desc)
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name):
                    env[tgt.id] = desc
                elif isinstance(tgt, ast.Tuple):
                    for i, el in enumerate(tgt.elts):
                        if isinstance(el, ast.Name):
                            env[el.id] = f"{desc}#{i}"
        elif isinstance(stmt, (ast.If, ast.With, ast.For, ast.While)):
            # Only the taken branch of `if get_pp_group().is_first_rank` matters
            # for single-GPU semantics; the else branch reads intermediate
            # tensors from a previous pipeline stage.
            _walk_assignments(stmt.body, env, seen)
        elif isinstance(stmt, ast.Return):
            # Several returns exist (the pipeline-parallel early exits come
            # first in the source); the one that matters is the last.
            env["__return__"] = _describe(stmt.value, env)
    return env


def reduce_mtp_forward(fn):
    """What this MTP forward feeds to `self.fc`, as a canonical string."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    seen = []
    env = _walk_assignments(tree.body[0].body, {}, seen)
    fc_inputs = [v for v in seen if v.startswith("fc(")]
    if not fc_inputs:
        raise SystemExit(f"{fn.__qualname__}: no call to self.fc found; the "
                         f"reference has been restructured and this capture's "
                         f"reading of it is stale")
    inner = sorted(fc_inputs, key=len)[-1][len("fc("):-1]
    for src, dst in _CANONICAL:
        inner = inner.replace(src, dst)
    return inner, env


def check_draft_composition():
    """Read the concat order and the norm assignment out of the reference.

    This is the check that pins the one decision with no runtime symptom. Both
    tensors are `[T, 5120]` of the same dtype, so swapping them or swapping the
    two norms produces a head that runs at full speed and drafts nonsense.
    """
    witnesses = {}
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor
    from vllm.model_executor.models.qwen3_next_mtp import (
        Qwen3NextMultiTokenPredictor,
    )

    for name, cls in (
        ("qwen3_5", Qwen3_5MultiTokenPredictor),
        ("qwen3_next", Qwen3NextMultiTokenPredictor),
    ):
        inner, env = reduce_mtp_forward(cls.forward)
        witnesses[name] = inner
        print(f"  {name:<11} fc input = {inner}")
        if inner != EXPECTED_FC_INPUT:
            raise SystemExit(
                f"{name}: the reference feeds `{inner}` to fc, this capture "
                f"expects `{EXPECTED_FC_INPUT}`. Do not 'fix' the expectation "
                f"— the reference is the authority; re-derive the capture."
            )
        ret = env.get("__return__", "")
        if "norm(" not in ret:
            raise SystemExit(f"{name}: forward returns `{ret}`, expected the "
                             f"final `self.norm(...)`")
        print(f"  {name:<11} returns  = {ret}")

    # And the decoder layer inside the head must be full attention. Read it out
    # of __init__ rather than believing the tensor names alone.
    # `@support_torch_compile` replaces `__init__` with its own wrapper, so
    # getsource on the method returns the decorator, not the model. Read the
    # class body.
    init_src = inspect.getsource(Qwen3_5MultiTokenPredictor)
    if 'layer_type="full_attention"' not in init_src:
        raise SystemExit("the reference no longer builds the MTP decoder layer "
                         "with layer_type=\"full_attention\"; re-derive")
    print("  layer_type  = full_attention (from Qwen3_5MultiTokenPredictor.__init__)")

    # And it reuses the base model's embedding and lm_head rather than shipping
    # its own: `load_weights` rewrites `mtp.` -> `model.` and separately pulls
    # `embed_tokens` / `lm_head` out of the *target* checkpoint.
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP
    lw = inspect.getsource(Qwen3_5MTP.load_weights)
    for needle in ('name.startswith("mtp.")', 'name.replace("mtp.", "model.")',
                   '["embed_tokens", "lm_head"]'):
        if needle not in lw:
            raise SystemExit(f"Qwen3_5MTP.load_weights no longer contains "
                             f"`{needle}`; the embedding/lm_head sharing story "
                             f"has changed and this capture is stale")
    print("  weight remap= mtp.* -> model.*, embed_tokens/lm_head from the base "
          "checkpoint")
    return witnesses


def check_rms_norm_form(headers, eps):
    """Two independent implementations of `Qwen3_5RMSNorm`, one f32 formula.

    Uses a real checkpoint weight rather than a random one: the whole reason the
    plain reading is dangerous is what these particular numbers are.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm
        vllm_norm = GemmaRMSNorm(5120, eps=eps)

    w = load_f32(headers, "mtp.pre_fc_norm_embedding.weight")
    tfms_norm = Qwen3_5RMSNorm(5120, eps=eps)
    with torch.no_grad():
        tfms_norm.weight.copy_(w)
        vllm_norm.weight.copy_(w)

    torch.manual_seed(11)
    x = torch.randn(7, 5120) * 3.0
    mine = rms_norm_offset(x, w, eps)
    d_tfms = (tfms_norm(x) - mine).abs().max().item()
    d_vllm = (vllm_norm(x) - mine).abs().max().item()
    wrong = rms_norm_plain(x, w, eps)
    d_wrong = (wrong - mine).abs().max().item() / mine.abs().max().item()

    print(f"  (1+w) form vs transformers Qwen3_5RMSNorm: Δ={d_tfms:.2e}")
    print(f"  (1+w) form vs vLLM GemmaRMSNorm:           Δ={d_vllm:.2e}")
    print(f"  plain `w *` form differs by:               {d_wrong:.2%} of peak")
    print(f"  weight stats: mean={w.mean():+.4f} min={w.min():+.4f} "
          f"max={w.max():+.4f}  (all-negative => `w *` flips every sign)")
    if d_tfms > 2e-5 or d_vllm > 2e-5:
        raise SystemExit("the (1+weight) RMSNorm formula disagrees with the "
                         "reference implementations; fix it before capturing")
    if d_wrong < 0.5:
        raise SystemExit("the plain `w *` reading gives nearly the same answer "
                         "here, so this check does not discriminate and the "
                         "capture would bless either form")

    # The fused add-then-norm that vLLM's residual stream uses must be exactly
    # norm(x + residual) — the alternative (norm(x) + residual) also runs.
    r = torch.randn(7, 5120) * 3.0
    out, res = vllm_norm(x.clone(), r.clone())
    d_fused = (out - rms_norm_offset(x + r, w, eps)).abs().max().item()
    d_res = (res - (x + r)).abs().max().item()
    print(f"  fused add-norm == norm(x+residual):        Δ={d_fused:.2e} "
          f"(residual passthrough Δ={d_res:.2e})")
    if d_fused > 2e-5 or d_res > 2e-5:
        raise SystemExit("vLLM's fused_add_rms_norm is not norm(x+residual); "
                         "the residual placement in this capture is wrong")
    return {"d_transformers": d_tfms, "d_vllm": d_vllm,
            "plain_form_relative_difference": d_wrong}


def check_tensor_inventory(headers, cfg):
    """The checkpoint's own evidence for what the head is and is not."""
    mtp = sorted(k for k in headers if k.startswith("mtp."))
    layer = [k for k in mtp if k.startswith("mtp.layers.0.")]
    facts = {}

    facts["num_mtp_tensors"] = len(mtp)
    # A linear-attention layer would carry conv1d / A_log / dt_bias / in_proj_*.
    linear_markers = [k for k in layer
                      if any(m in k for m in ("linear_attn", "conv1d", "A_log",
                                              "dt_bias", "in_proj"))]
    attn_markers = [k for k in layer if ".self_attn." in k]
    if linear_markers:
        raise SystemExit(f"the MTP layer carries GatedDeltaNet tensors "
                         f"{linear_markers}; it is not a full-attention layer "
                         f"and this capture is built on the wrong assumption")
    if not attn_markers:
        raise SystemExit("the MTP layer has no self_attn tensors at all")
    facts["layer_is_full_attention"] = True

    # Dedicated embeddings / lm_head: the config says no, the checkpoint must
    # agree, or `mtp_use_dedicated_embeddings` is not the switch we think it is.
    dedicated = [k for k in mtp if "embed_tokens" in k or "lm_head" in k]
    if cfg.get("mtp_use_dedicated_embeddings") is not False:
        raise SystemExit("mtp_use_dedicated_embeddings is not False; this "
                         "capture only covers the shared-embedding case")
    if dedicated:
        raise SystemExit(f"config says mtp_use_dedicated_embeddings=False but "
                         f"the checkpoint ships {dedicated}")
    for need in ("lm_head.weight", "model.language_model.embed_tokens.weight",
                 "model.language_model.norm.weight"):
        if need not in headers:
            raise SystemExit(f"the base checkpoint has no {need}, so the head "
                             f"cannot be reusing it")
    facts["reuses_base_embedding_and_lm_head"] = True
    if cfg.get("tie_word_embeddings"):
        raise SystemExit("tie_word_embeddings is true; lm_head would then be "
                         "the embedding and this capture's separate lm_head "
                         "check is meaningless")
    facts["tie_word_embeddings"] = False

    # The one shape that encodes the concat: fc takes 2 * hidden_size.
    fc_shape = headers["mtp.fc.weight"][2]["shape"]
    if fc_shape != [cfg["hidden_size"], 2 * cfg["hidden_size"]]:
        raise SystemExit(f"mtp.fc.weight is {fc_shape}, expected "
                         f"[{cfg['hidden_size']}, {2 * cfg['hidden_size']}]")
    facts["fc_shape"] = fc_shape
    print(f"  {len(mtp)} mtp.* tensors, layer is full attention, fc is {fc_shape}")
    return facts


# ---------------------------------------------------------------- the target


def load_layer(headers, cfg_obj, layer_idx, prefix, dtype):
    """One `Qwen3_5DecoderLayer` with real weights, the reference's own class."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    layer = m.Qwen3_5DecoderLayer(cfg_obj, layer_idx).to(dtype).eval()
    missing = []
    with torch.no_grad():
        for name, param in layer.named_parameters():
            key = prefix + name
            if key not in headers:
                missing.append(name)
                continue
            w = load_f32(headers, key)
            param.copy_(w.reshape(param.shape))
    if missing:
        raise SystemExit(f"{prefix}: no weights for {missing}")
    return layer


def rope_tables(theta, rot, positions, dtype):
    """The partial-rope tables, `rotate_half` layout, exactly as the notes say.

    The exponent is normalized by `rot`, not by `head_dim`; see
    notes/qwen3.5-architecture.md for why the other reading is silent.
    """
    inv = 1.0 / (theta ** (torch.arange(0, rot, 2, dtype=torch.float64) / rot))
    freqs = positions.double()[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype)[None], emb.sin().to(dtype)[None]


def check_rope_tables_against_reference(cfg, cfg_obj):
    """`rope_tables` above, against `Qwen3_5TextRotaryEmbedding`.

    The table is the one thing in this file that the behavioural check cannot
    settle. It feeds the reference's own decoder layers, so an error in it
    degrades the target model — but gracefully: the top-1 agreement stays
    plausible while long-range attention rots, which is the signature the
    head_dim-vs-rot normalization mistake produces. So compare against the
    reference's `inv_freq` directly and require the wrong reading to be visible.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    ref = m.Qwen3_5TextRotaryEmbedding(cfg_obj)
    rot = int(cfg["head_dim"] * cfg["partial_rotary_factor"])
    theta = cfg["rope_theta"]
    if ref.inv_freq.shape[0] != rot // 2:
        raise SystemExit(
            f"the reference builds {ref.inv_freq.shape[0]} rotary frequencies, "
            f"this capture builds rot//2 = {rot // 2}. partial_rotary_factor or "
            f"head_dim is being read from the wrong place.")

    # A 2-D `position_ids` is expanded into three identical rows, so
    # `apply_interleaved_mrope` is a no-op and what comes back is the plain
    # partial table this capture builds. (The interleaving is pinned separately,
    # in tools/capture_qwen35_vision.py, straight out of the reference method.)
    pos = torch.tensor([[0, 1, 2, 5, 31]])
    cos_ref, sin_ref = ref(torch.zeros(1, pos.shape[1], 1), pos)
    cos, sin = rope_tables(theta, rot, pos[0], torch.float32)
    d_inv = (1.0 / (theta ** (torch.arange(0, rot, 2, dtype=torch.float64) / rot))
             - ref.inv_freq.double()).abs().max().item()
    d_cos = (cos[0] - cos_ref[0]).abs().max().item()
    d_sin = (sin[0] - sin_ref[0]).abs().max().item()
    wrong = 1.0 / (theta ** (torch.arange(0, rot, 2, dtype=torch.float64)
                             / cfg["head_dim"]))
    sep = (wrong - ref.inv_freq.double()).abs().max().item()
    print(f"  rope inv_freq vs Qwen3_5TextRotaryEmbedding: Δ={d_inv:.2e}  "
          f"cos Δ={d_cos:.2e}  sin Δ={d_sin:.2e}")
    print(f"  (normalizing the exponent by head_dim={cfg['head_dim']} instead of "
          f"rot={rot} moves inv_freq by {sep:.3e})")
    if d_inv > 1e-7 or d_cos > 2e-6 or d_sin > 2e-6:
        raise SystemExit("the rope table disagrees with the reference's; fix it "
                         "before capturing anything")
    if sep < 1e-2:
        raise SystemExit("the head_dim-normalized table is indistinguishable "
                         "here; this check is decorative")
    return {"d_inv_freq": d_inv, "d_cos": d_cos, "d_sin": d_sin,
            "head_dim_normalized_separation": sep}


def causal_mask(t_len, dtype):
    """Additive causal mask.

    `eager_attention_forward` applies no mask at all when it is given `None`, so
    the attention is *non-causal* by default. Omitting this does not fail; it
    lets every token see the future, and the captured hidden states would then
    be from a model that cannot exist at decode time.
    """
    m = torch.zeros(t_len, t_len, dtype=dtype)
    m.masked_fill_(torch.triu(torch.ones(t_len, t_len, dtype=torch.bool), 1),
                   float("-inf"))
    return m[None, None]


def run_target(model_dir, headers, cfg, cfg_obj, ids, n_layers, report):
    """The full target stack, streaming one layer at a time.

    Returns the hidden state before and after `model.language_model.norm`. Both,
    because "which one does the head consume" is a real question with two
    plausible answers, and the capture should be able to settle it rather than
    assume it.
    """
    dtype = torch.bfloat16
    t_len = ids.shape[1]
    emb_w = load_f32(headers, "model.language_model.embed_tokens.weight")
    h = torch.nn.functional.embedding(ids, emb_w.to(dtype))

    rot = int(cfg["head_dim"] * cfg["partial_rotary_factor"])
    pe = rope_tables(cfg["rope_theta"], rot, torch.arange(t_len), dtype)
    mask = causal_mask(t_len, dtype)

    with torch.no_grad():
        for i in range(n_layers):
            layer = load_layer(headers, cfg_obj, i,
                               f"model.language_model.layers.{i}.", dtype)
            kind = cfg_obj.layer_types[i]
            out = layer(h, position_embeddings=pe,
                        attention_mask=None if kind == "linear_attention" else mask)
            h = out[0] if isinstance(out, tuple) else out
            del layer
            if i % 8 == 0 or i == n_layers - 1:
                print(f"    layer {i:>2} ({kind:<16}) hidden RMS "
                      f"{h.float().pow(2).mean().sqrt().item():.4f}")
        pre_norm = h[0].float().clone()
        norm_w = load_f32(headers, "model.language_model.norm.weight")
        final = rms_norm_offset(pre_norm, norm_w, cfg["rms_norm_eps"])

    report("target hidden entering model.norm", pre_norm)
    return pre_norm, final


# ------------------------------------------------------------------ the head


class MtpHead(torch.nn.Module):
    """The MTP head, assembled out of the reference implementations' own classes.

    Nothing here is a transcription of the head's arithmetic except the order in
    which the four glue pieces are applied, and that order is what
    `check_draft_composition` reads out of vLLM's AST. The decoder layer is
    `transformers.models.qwen3_5.Qwen3_5DecoderLayer` verbatim, so the output
    gate, the per-head q/gate interleave, the partial rope and the SwiGLU MLP
    are the reference's own code rather than mine.
    """

    def __init__(self, headers, cfg, cfg_obj, full_attention_layer_idx):
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

        eps = cfg["rms_norm_eps"]
        d = cfg["hidden_size"]

        def norm(name):
            n = Qwen3_5RMSNorm(d, eps=eps)
            with torch.no_grad():
                n.weight.copy_(load_f32(headers, f"mtp.{name}.weight"))
            return n

        self.pre_fc_norm_embedding = norm("pre_fc_norm_embedding")
        self.pre_fc_norm_hidden = norm("pre_fc_norm_hidden")
        self.norm = norm("norm")
        self.fc = torch.nn.Linear(2 * d, d, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(load_f32(headers, "mtp.fc.weight"))
        # `layer_types[full_attention_layer_idx]` must be "full_attention" so
        # that the class builds a Qwen3_5Attention rather than a GatedDeltaNet.
        self.layer = load_layer(headers, cfg_obj, full_attention_layer_idx,
                               "mtp.layers.0.", torch.float32)
        self.taps = {}
        self._install_taps()

    def _install_taps(self):
        """Tap the decoder layer's submodules so every stage inside it is
        captured as the reference's own tensor rather than as a recomputation.

        None of the big projections can be dumped — `q_proj` alone is 251 MB in
        f32 — so the test cannot check the head's layer end to end from weights.
        What it can do is check every *transition*: the two RMSNorms, the
        per-head q/gate split, the partial rope, the attention, the sigmoid gate,
        and both residual adds. All of those are pinned by these taps plus a
        handful of probe rows, and none of them needs a matrix.

        `q_norm`'s *input* is worth singling out: it is the reference's own `q`
        after `view(T, heads, 2*head_dim)` and `chunk(2, dim=-1)`. Dumping it
        next to the raw `q_proj` output lets a test rediscover the interleave
        from the reference rather than inherit this script's reading of it.
        """
        sa = self.layer.self_attn

        def tap(name, mod, want_input=False):
            def fn(_m, inputs, output):
                if want_input:
                    self.taps[name + "_in"] = inputs[0].detach().float()
                self.taps[name + "_out"] = (
                    output[0] if isinstance(output, tuple) else output
                ).detach().float()
            mod.register_forward_hook(fn)

        tap("pre_attn_norm", self.layer.input_layernorm)
        tap("q_proj", sa.q_proj)
        tap("q_norm", sa.q_norm, want_input=True)
        tap("k_norm", sa.k_norm, want_input=True)
        tap("v_proj", sa.v_proj)
        tap("o_proj", sa.o_proj, want_input=True)
        tap("post_attn_norm", self.layer.post_attention_layernorm)
        tap("mlp", self.layer.mlp)

    def forward(self, inputs_embeds, hidden_states, position_embeddings, mask):
        e = self.pre_fc_norm_embedding(inputs_embeds)
        h = self.pre_fc_norm_hidden(hidden_states)
        fused = self.fc(torch.cat([e, h], dim=-1))
        out = self.layer(fused[None], position_embeddings=position_embeddings,
                         attention_mask=mask)
        out = out[0] if isinstance(out, tuple) else out
        # Everything returned is `[t_len, hidden]`; the leading batch axis only
        # exists because the reference's decoder layer wants it.
        stages = {"emb_normed": e, "hidden_normed": h,
                  "fc_out": fused, "layer_out": out[0]}
        # Every tap carries the decoder layer's leading batch axis of 1.
        for k, v in self.taps.items():
            stages["layer." + k] = v[0]
        return self.norm(out)[0], stages


# ------------------------------------------------------- the acceptance rule
#
# `qwen35_mtp.rs`'s `accept_greedy` and `accept_stochastic` are transcriptions of
# vLLM's `rejection_greedy_sample_kernel` and `rejection_random_sample_kernel`,
# and nothing checked them. They are not floating-point arithmetic, which is why
# they were easy to overlook, but they are exactly as silent when wrong: an
# off-by-one in where the bonus token goes, or accepting on `>` instead of `>=`,
# changes the emitted sequence and nothing crashes. In the greedy case a wrong
# rule breaks the property the whole design rests on — that speculation emits
# bit-identically what unspeculated greedy decoding would have.
#
# Both kernels take plain pointers, so they can be launched directly on a small
# hand-built battery without any of vLLM's scheduling machinery. They are triton,
# so this needs a GPU; when there is none the arrays are simply absent and the
# Rust test skips rather than passing vacuously.


def acceptance_battery(seed=20260822):
    """Cases that exercise every branch of the rule, on GPU, via vLLM's kernels.

    Returns a dict of arrays, or None if the kernels cannot be launched here.
    """
    if not torch.cuda.is_available():
        print("  !! no GPU: vLLM's rejection kernels are triton and cannot run, "
              "so accept_greedy / accept_stochastic stay unchecked here")
        return None
    try:
        from vllm.v1.sample.rejection_sampler import (
            PLACEHOLDER_TOKEN_ID,
            rejection_greedy_sample_kernel,
            rejection_random_sample_kernel,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  !! cannot import vLLM's rejection sampler ({e}); "
              f"accept_greedy / accept_stochastic stay unchecked")
        return None

    dev = "cuda"
    vocab, max_spec = 16, 4
    gen = torch.Generator().manual_seed(seed)
    # Draft lengths 1..max_spec, and for each length one case that rejects at
    # every position plus one that accepts everything — so "emit the bonus" and
    # "emit the target's own token and stop" both occur, at both ends.
    lens, drafts, targets = [], [], []
    for length in range(1, max_spec + 1):
        for reject_at in list(range(length)) + [None]:
            lens.append(length)
            d = torch.randint(0, vocab, (length,), generator=gen)
            t = d.clone()
            if reject_at is not None:
                t[reject_at] = (d[reject_at] + 1 + int(
                    torch.randint(0, vocab - 1, (1,), generator=gen))) % vocab
                # Beyond the first mismatch the target's tokens are arbitrary and
                # must be ignored; make them differ so a rule that keeps reading
                # past the rejection is visible.
                for j in range(reject_at + 1, length):
                    t[j] = (d[j] + 3) % vocab
            drafts.append(d)
            targets.append(t)
    batch = len(lens)
    n = sum(lens)
    cu = torch.tensor(list(itertools.accumulate(lens)), dtype=torch.int32,
                      device=dev)
    draft = torch.cat(drafts).to(torch.int32).to(dev)
    targ = torch.cat(targets).to(torch.int32).to(dev)
    bonus = torch.arange(1000, 1000 + batch, dtype=torch.int32, device=dev)

    greedy = torch.full((batch, max_spec + 1), PLACEHOLDER_TOKEN_ID,
                        dtype=torch.int32, device=dev)
    rejection_greedy_sample_kernel[(batch,)](
        greedy, cu, draft, targ, bonus, None, max_spec, None, None,
        SYNTHETIC_MODE=False)

    # The stochastic rule. Probabilities are built so the ratio straddles the
    # uniform draw in both directions, and one row has p_draft == 0 exactly —
    # the case vLLM guards against dividing by, where the ratio would be +inf
    # and would accept a token the draft model considers impossible.
    dp = torch.rand(n, vocab, generator=gen) + 1e-3
    tp = torch.rand(n, vocab, generator=gen) + 1e-3
    dcpu = draft.cpu().long()
    dp[3, dcpu[3]] = 0.0
    dp = (dp / dp.sum(-1, keepdim=True)).to(dev)
    tp = (tp / tp.sum(-1, keepdim=True)).to(dev)
    uniform = torch.rand(n, generator=gen).to(dev)
    recovered = torch.randint(0, vocab, (n,), generator=gen).to(torch.int32).to(dev)
    is_greedy = torch.zeros(batch, dtype=torch.bool, device=dev)
    random_out = torch.full((batch, max_spec + 1), PLACEHOLDER_TOKEN_ID,
                            dtype=torch.int32, device=dev)
    rejection_random_sample_kernel[(batch,)](
        random_out, cu, draft, dp, tp, bonus, recovered, uniform, is_greedy,
        max_spec, vocab, None, NO_DRAFT_PROBS=False, SYNTHETIC_MODE=False)

    idx = torch.arange(n, device=dev)
    p_draft = dp[idx, draft.long()]
    p_target = tp[idx, draft.long()]
    accepted_g = (greedy != PLACEHOLDER_TOKEN_ID).sum(-1)
    accepted_r = (random_out != PLACEHOLDER_TOKEN_ID).sum(-1)
    print(f"  vLLM's rejection kernels on {batch} cases, {n} draft tokens: "
          f"greedy emits {accepted_g.tolist()}")
    print(f"  {'':>4}stochastic emits {accepted_r.tolist()}, "
          f"{int((p_draft == 0).sum())} zero draft probability")
    # The battery has to contain rejections and full acceptances, or it proves
    # nothing about either branch.
    lens_t = torch.tensor(lens, device=dev)
    if not ((accepted_g == lens_t + 1).any() and (accepted_g <= lens_t).any()):
        raise SystemExit("the acceptance battery has no full acceptance or no "
                         "rejection; it cannot pin the rule")

    return {
        "accept.num_draft": torch.tensor(lens, dtype=torch.float32),
        "accept.draft": draft.float().cpu(),
        "accept.target_argmax": targ.float().cpu(),
        "accept.bonus": bonus.float().cpu(),
        "accept.greedy_out": greedy.float().cpu(),
        "accept.p_draft": p_draft.float().cpu(),
        "accept.p_target": p_target.float().cpu(),
        "accept.uniform": uniform.float().cpu(),
        "accept.recovered": recovered.float().cpu(),
        "accept.random_out": random_out.float().cpu(),
        "accept.placeholder": torch.tensor([float(PLACEHOLDER_TOKEN_ID)]),
    }


# ------------------------------------------------ a whole layer, small enough
#
# The taps above pin every *transition* inside the head's decoder layer, but not
# the transitions' composition, and not the MLP at all: `q_proj` alone is 251 MB
# in f32, so the real layer's weights cannot be dumped for a Rust test to run
# end to end. The pieces that go unchecked as a result are exactly the ones with
# a mirror image that runs — `silu` on `gate_proj` rather than on `up_proj`, the
# two residual adds, whether the second norm sees the post-attention stream.
#
# So build the same class at a size that fits: `Qwen3_5DecoderLayer` from the
# reference, random weights, 40 wide. Every weight, the input and the output get
# dumped, and the Rust `full_attention_layer` then answers to the library
# directly instead of to a decomposition of it. 40x40 matrices are not a
# numerical stand-in for 5120x5120 ones and are not meant to be — the real
# checkpoint's activations are what the stagewise checks are for. This is for
# layout and composition, where size is irrelevant.


SYNTH = {"d_model": 40, "heads": 6, "kv_heads": 2, "head_dim": 8,
         "rotary_dim": 4, "d_ff": 56, "tokens": 9, "rope_theta": 1e7}


def synthetic_layer_capture(cfg):
    """One small `Qwen3_5DecoderLayer` and one small `Qwen3_5MLP`, with weights.

    Returns (arrays, config). Nothing here is a transcription: the tensors come
    out of the reference's own modules, and the only thing this file decides is
    which of them to write down.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    s = dict(SYNTH)
    hd, nh = s["head_dim"], s["heads"]
    obj = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=s["d_model"], intermediate_size=s["d_ff"],
        num_hidden_layers=2, num_attention_heads=nh,
        num_key_value_heads=s["kv_heads"], head_dim=hd,
        rms_norm_eps=cfg["rms_norm_eps"], hidden_act=cfg["hidden_act"],
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": s["rope_theta"],
                         "partial_rotary_factor": s["rotary_dim"] / hd,
                         "mrope_section": [1, 1, 0], "mrope_interleaved": True},
    )
    obj._attn_implementation = "eager"
    layer = m.Qwen3_5DecoderLayer(obj, 1).to(torch.float32).eval()
    gen = torch.Generator().manual_seed(20260822)
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.4)

    if type(layer.self_attn).__name__ != "Qwen3_5Attention":
        raise SystemExit("layer_types[1]='full_attention' did not build a "
                         "Qwen3_5Attention")
    # `ACT2FN[...]` hands back a fresh module each time, so identity says
    # nothing; compare what it computes. The Rust `swiglu_mlp` hard-codes silu.
    probe = torch.linspace(-8, 8, 401)
    d_act = (layer.mlp.act_fn(probe)
             - torch.nn.functional.silu(probe)).abs().max().item()
    d_gelu = (layer.mlp.act_fn(probe)
              - torch.nn.functional.gelu(probe)).abs().max().item()
    if d_act > 1e-6:
        raise SystemExit(f"the MLP activation is not silu (Δ={d_act:.2e}); "
                         f"hidden_act says {cfg['hidden_act']!r}")
    if d_gelu < 1e-2:
        raise SystemExit("silu and gelu are indistinguishable on this probe")

    T = s["tokens"]
    x = torch.randn(T, s["d_model"], generator=gen)
    pe = rope_tables(s["rope_theta"], s["rotary_dim"], torch.arange(T),
                     torch.float32)
    with torch.no_grad():
        out = layer(x[None], position_embeddings=pe,
                    attention_mask=causal_mask(T, torch.float32))
        out = (out[0] if isinstance(out, tuple) else out)[0]
        # The MLP on its own, on its own input, so a failure localizes. The
        # mirror image — silu on up_proj instead of gate_proj — must not
        # reproduce it, or the dump is not evidence about the orientation.
        mx = torch.randn(T, s["d_model"], generator=gen)
        my = layer.mlp(mx)
        mirror = layer.mlp.down_proj(layer.mlp.gate_proj(mx)
                                    * layer.mlp.act_fn(layer.mlp.up_proj(mx)))
    sep = (mirror - my).abs().max().item() / my.abs().max().item()
    print(f"  synthetic layer: {s['d_model']} wide, {T} tokens; the mirrored "
          f"SwiGLU (silu on up_proj) differs by {sep:.3e} of peak")
    if sep < 1e-2:
        raise SystemExit("silu on gate_proj and silu on up_proj give nearly the "
                         "same answer on this input; the dump would bless both")

    sa = layer.self_attn
    arrays = {
        "synth.x": x, "synth.out": out,
        "synth.rope_cos": pe[0][0], "synth.rope_sin": pe[1][0],
        "synth.mlp_x": mx, "synth.mlp_y": my,
        "synth.w.input_layernorm": layer.input_layernorm.weight,
        "synth.w.post_attention_layernorm": layer.post_attention_layernorm.weight,
        "synth.w.q_norm": sa.q_norm.weight, "synth.w.k_norm": sa.k_norm.weight,
        "synth.w.q_proj": sa.q_proj.weight, "synth.w.k_proj": sa.k_proj.weight,
        "synth.w.v_proj": sa.v_proj.weight, "synth.w.o_proj": sa.o_proj.weight,
        "synth.w.gate_proj": layer.mlp.gate_proj.weight,
        "synth.w.up_proj": layer.mlp.up_proj.weight,
        "synth.w.down_proj": layer.mlp.down_proj.weight,
    }
    s["eps"] = cfg["rms_norm_eps"]
    s["mirrored_swiglu_separation"] = sep
    return arrays, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--prefix-layers", type=int, default=None,
                    help="run only the first N target layers. A truncated "
                         "prefix is NOT a valid oracle for the head — the head "
                         "consumes the final hidden state — and the manifest "
                         "records it so the Rust test can refuse. For smoke "
                         "runs only.")
    ap.add_argument("--text", default=None)
    ap.add_argument("--dump-layer-weights", action="store_true",
                    help="also dump the head's whole decoder layer and fc, 1.7 GB "
                         "in f32. Off by default because the stage-by-stage "
                         "checks pin every transition without it; on when you "
                         "want the Rust reference's `full_attention_layer` and "
                         "`mtp_head` exercised end to end rather than "
                         "transition by transition.")
    args = ap.parse_args()

    raw = json.load(open(os.path.join(args.model_dir, "config.json")))
    cfg = dict(raw["text_config"])
    # rope_theta lives only inside rope_parameters; reading it from the top
    # level silently substitutes a default frequency base.
    rp = cfg.get("rope_parameters", {})
    cfg["rope_theta"] = rp.get("rope_theta", cfg.get("rope_theta"))
    cfg["partial_rotary_factor"] = rp.get(
        "partial_rotary_factor", cfg.get("partial_rotary_factor", 1.0))
    assert cfg["rope_theta"] is not None, "no rope_theta anywhere in the config"

    torch.set_num_threads(min(96, os.cpu_count() or 8))
    headers = read_index(args.model_dir)

    print("== cross-checks against the reference implementations")
    witnesses = check_draft_composition()
    norm_check = check_rms_norm_form(headers, cfg["rms_norm_eps"])
    inventory = check_tensor_inventory(headers, cfg)

    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    cfg_obj = Qwen3_5TextConfig(**raw["text_config"])
    cfg_obj._attn_implementation = "eager"
    full_idx = cfg_obj.layer_types.index("full_attention")
    print(f"  the head's decoder layer is built as layer_types[{full_idx}] = "
          f"{cfg_obj.layer_types[full_idx]}")
    rope_check = check_rope_tables_against_reference(cfg, cfg_obj)
    synth_arrays, synth_cfg = synthetic_layer_capture(cfg)
    accept_arrays = acceptance_battery()
    if accept_arrays:
        synth_arrays.update(accept_arrays)

    # ---- tokens. Real text, so the hidden states are on the model's manifold;
    # the top-1 agreement check below is only meaningful on text the model can
    # actually predict.
    text = args.text or (
        "The GatedDeltaNet recurrence keeps a fixed-size state per head, so the "
        "cost of a decode step does not grow with the length of the context. "
        "That is the property speculative decoding has to preserve: a rejected "
        "draft must leave the state exactly where an unspeculated decode would "
        "have left it, and the state is updated in place."
    )
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    ids = tok(text, return_tensors="pt").input_ids[:, : args.tokens]
    t_len = ids.shape[1]
    print(f"\n== target model, {t_len} real tokens, "
          f"{args.prefix_layers or cfg['num_hidden_layers']} layers")

    eps_reports = []

    def report(what, x):
        eps = cfg["rms_norm_eps"]
        # Per-row, not global: one row inside the eps regime is enough to make
        # that row's norm a constant, and averaging hides it.
        ms = x.float().pow(2).mean(-1)
        share = (eps / (ms + eps)).max().item()
        eps_reports.append({"tensor": what, "min_mean_square": ms.min().item(),
                            "worst_eps_share": share})
        flag = "" if share < MAX_EPS_SHARE else "   <-- eps DOMINATES"
        print(f"    {what:<44} min mean(x^2)={ms.min().item():.3e}  "
              f"eps is {share:.3%} of the worst row's denominator{flag}")

    n_layers = args.prefix_layers or cfg["num_hidden_layers"]
    truncated = n_layers != cfg["num_hidden_layers"]
    pre_norm, final_hidden = run_target(args.model_dir, headers, cfg, cfg_obj,
                                        ids, n_layers, report)

    # ---- the target's own next-token predictions, which are both the shifted
    # input to the head and the thing the head's drafts are graded against.
    lm_head_w = load_f32(headers, "lm_head.weight")
    with torch.no_grad():
        target_logits = final_hidden @ lm_head_w.T
    target_argmax = target_logits.argmax(-1)

    # Slot i of the head sees (hidden state of token i, embedding of token i+1)
    # and predicts token i+2. vLLM builds the shifted ids exactly this way:
    # rotate the target ids left by one and put the freshly sampled token in the
    # last slot.
    shifted = torch.cat([ids[0, 1:], target_argmax[-1:]])
    emb_w = load_f32(headers, "model.language_model.embed_tokens.weight")
    inputs_embeds = torch.nn.functional.embedding(shifted, emb_w)
    report("embedding entering pre_fc_norm_embedding", inputs_embeds)
    report("hidden entering pre_fc_norm_hidden", final_hidden)

    print("\n== the MTP head")
    head = MtpHead(headers, cfg, cfg_obj, full_idx).eval()

    rot = int(cfg["head_dim"] * cfg["partial_rotary_factor"])
    theta = cfg["rope_theta"]
    mask = causal_mask(t_len, torch.float32)

    # vLLM leaves the drafter's positions equal to the target's: slot i keeps the
    # position of the token whose *hidden state* it carries, one less than the
    # position of the token it embeds
    # (`llm_base_proposer.set_inputs_first_pass` rotates the ids and passes
    # `target_positions` straight through). Run the +1 alternative too — not
    # because it is a candidate error this capture can catch, but to *measure*
    # that it cannot: see the assertion below.
    positions = torch.arange(t_len)
    pe = rope_tables(theta, rot, positions, torch.float32)
    pe_plus1 = rope_tables(theta, rot, positions + 1, torch.float32)

    def relative_l2(a, b):
        return ((a - b).double().pow(2).sum() / b.double().pow(2).sum()).sqrt().item()

    stages = {}
    with torch.no_grad():
        out, mid = head(inputs_embeds, final_hidden, pe, mask)
        stages.update(mid)
        stages["output"] = out
        stages["draft_argmax"] = (out @ lm_head_w.T).argmax(-1)

        out_p1, _ = head(inputs_embeds, final_hidden, pe_plus1, mask)
        alt_pos_argmax = (out_p1 @ lm_head_w.T).argmax(-1)

        # Slot i drafts token i+2, which the target predicts from slot i+1. Two
        # numbers per variant: how often it drafts what the target would have
        # sampled, and how far its hidden output sits from the reference
        # composition's. The first is what speculative decoding cares about; the
        # second is what says whether this capture can tell the two apart at all.
        def score(name, argmax, output):
            n = t_len - 1
            hit = (argmax[:n] == target_argmax[1:1 + n]).float().mean().item()
            l2 = relative_l2(output, stages["output"])
            print(f"    {name:<30} top-1 {hit:>6.1%}   output ΔL2 {l2:.2e}")
            return hit, l2

        print("    variant                        agreement       divergence")
        agree, _ = score("the reference composition", stages["draft_argmax"], out)
        agree_pos, l2_pos = score("positions shifted +1", alt_pos_argmax, out_p1)

        d = cfg["hidden_size"]
        eps = cfg["rms_norm_eps"]
        w_emb = load_f32(headers, "mtp.pre_fc_norm_embedding.weight")
        w_hid = load_f32(headers, "mtp.pre_fc_norm_hidden.weight")
        fc_w = head.fc.weight

        def run_variant(cat_first, cat_second):
            fused = torch.cat([cat_first, cat_second], dim=-1) @ fc_w.T
            o = head.layer(fused[None], position_embeddings=pe,
                           attention_mask=mask)
            o = o[0] if isinstance(o, tuple) else o
            o = head.norm(o)[0]
            return (o @ lm_head_w.T).argmax(-1), o

        e_n, h_n = stages["emb_normed"], stages["hidden_normed"]
        agree_swap_cat, l2_swap_cat = score("concat halves swapped",
                                            *run_variant(h_n, e_n))
        agree_swap_norm, l2_swap_norm = score("the two pre_fc norms swapped",
            *run_variant(rms_norm_offset(inputs_embeds, w_hid, eps),
                         rms_norm_offset(final_hidden, w_emb, eps)))
        agree_plain, l2_plain = score("plain `w *` instead of (1+w)",
            *run_variant(rms_norm_plain(inputs_embeds, w_emb, eps),
                         rms_norm_plain(final_hidden, w_hid, eps)))
        agree_prenorm, l2_prenorm = score("pre-model.norm hidden state",
            *run_variant(e_n, rms_norm_offset(pre_norm, w_hid, eps)))

    behaviour = {
        "top1_agreement": agree,
        "top1_agreement_positions_plus_1": agree_pos,
        "top1_agreement_concat_swapped": agree_swap_cat,
        "top1_agreement_norms_swapped": agree_swap_norm,
        "top1_agreement_plain_rmsnorm": agree_plain,
        "top1_agreement_pre_final_norm_hidden": agree_prenorm,
        "divergence_positions_plus_1": l2_pos,
        "divergence_concat_swapped": l2_swap_cat,
        "divergence_norms_swapped": l2_swap_norm,
        "divergence_plain_rmsnorm": l2_plain,
        "divergence_pre_final_norm_hidden": l2_prenorm,
    }
    if not truncated:
        if agree < MIN_TOP1_AGREEMENT:
            raise SystemExit(
                f"the head agrees with the target only {agree:.1%} of the time. "
                f"A correctly composed MTP head lifts decode from 44 to 89 "
                f"tok/s on this checkpoint, which needs far better than that. "
                f"Something in the composition is wrong; refusing to write a "
                f"capture that would bless it.")

        # The three readings that are outright errors have to be visibly worse,
        # both in what they draft and in what they compute. If they were not,
        # this capture would be blessing whichever one happened to be coded.
        for name, hit, l2 in (
            ("concat halves swapped", agree_swap_cat, l2_swap_cat),
            ("the two pre_fc norms swapped", agree_swap_norm, l2_swap_norm),
            ("plain `w *` RMSNorm", agree_plain, l2_plain),
        ):
            if hit >= agree or l2 < 1e-2:
                raise SystemExit(
                    f"the variant `{name}` drafts as well as the reference "
                    f"composition ({hit:.1%} vs {agree:.1%}) or computes nearly "
                    f"the same thing (ΔL2 {l2:.2e}). This capture cannot tell "
                    f"them apart, so it is not evidence for either.")

        # Feeding the head the hidden state from *before* the target's final norm
        # is a fourth plausible reading, and it is a different computation — but
        # it is not one this behavioural check can rule out: on this text it
        # drafts about as well. That is not surprising, since `pre_fc_norm_hidden`
        # renormalizes anyway and only the per-channel gain of `model.norm`
        # survives. So it is pinned numerically instead: the capture dumps both
        # hidden states, the two normalized versions differ by ΔL2 below, and the
        # Rust test asserts which one the reference's `hidden_normed` matches.
        if l2_prenorm < 1e-2:
            raise SystemExit(
                f"using the pre-final-norm hidden state changes the head's "
                f"output by only ΔL2 {l2_prenorm:.2e}, so nothing distinguishes "
                f"the two and this capture cannot pin which one the head reads")
        print(f"    note: the pre-final-norm hidden state drafts about as well "
              f"({agree_prenorm:.1%} vs {agree:.1%}) but is a different "
              f"computation (ΔL2 {l2_prenorm:.2e}); the Rust test pins it "
              f"numerically, not behaviourally")

        # And the position convention is *not* pinned by this capture, on
        # purpose, because it cannot be: shifting every position by a constant
        # must leave a self-contained attention pass unchanged — that is what
        # "rotary embeddings encode relative position" means, and
        # notes/qwen3.5-architecture.md leans on the same invariant. So assert
        # the invariance rather than pretend to a discrimination. The convention
        # only becomes observable across draft steps, where it fixes the offset
        # between a new token and the drafter's cached history, and there it is
        # settled by reading vLLM rather than by any number here.
        if l2_pos > 1e-4:
            raise SystemExit(
                f"shifting every rope position by one changed the head's output "
                f"by ΔL2 {l2_pos:.2e}. A uniform shift must be invisible to a "
                f"single self-contained pass; anything above phase noise means "
                f"the frequency table or the rotate_half pairing is wrong.")
        print(f"    rope shift invariance holds (ΔL2 {l2_pos:.2e} under a "
              f"uniform +1), so the position convention is fixed by the "
              f"reference's source, not by this capture")

        bad = [r for r in eps_reports if r["worst_eps_share"] >= MAX_EPS_SHARE]
        if bad:
            raise SystemExit(
                f"eps accounts for more than {MAX_EPS_SHARE:.0%} of the RMS "
                f"denominator for {[r['tensor'] for r in bad]}. In that regime "
                f"the norm degenerates towards a constant 1/sqrt(eps) scale and "
                f"the capture cannot discriminate its formulation. Use real "
                f"activations.")

    # ------------------------------------------------------------------ write
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {"arrays": {}, "config": {k: cfg[k] for k in (
        "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
        "rms_norm_eps", "partial_rotary_factor", "intermediate_size",
        "vocab_size", "num_hidden_layers", "mtp_num_hidden_layers",
    )}}
    manifest["config"]["rope_theta"] = cfg["rope_theta"]
    manifest["config"]["tokens"] = t_len
    manifest["verified_against_reference"] = True
    manifest["prefix_layers_run"] = n_layers
    manifest["prefix_truncated"] = truncated
    manifest["fc_input_reduction"] = witnesses
    manifest["rms_norm_check"] = norm_check
    manifest["rope_table_check"] = rope_check
    manifest["synth_config"] = synth_cfg
    manifest["acceptance_from_vllm_kernels"] = accept_arrays is not None
    manifest["tensor_inventory"] = inventory
    manifest["behaviour"] = behaviour
    manifest["eps_headroom"] = eps_reports
    manifest["token_ids"] = ids[0].tolist()
    manifest["shifted_token_ids"] = shifted.tolist()

    def dump(name, t):
        t = t.detach().contiguous().float()
        with open(os.path.join(args.out_dir, name + ".f32"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        manifest["arrays"][name] = list(t.shape)
        flat = t.reshape(-1)
        print(f"  {name:<32} {str(list(t.shape)):<16} "
              f"[{flat.min():+.5f}, {flat.max():+.5f}] "
              f"nonfinite={int((~flat.isfinite()).sum())}")

    print("\n== writing")
    dump("target.pre_norm_hidden", pre_norm)
    dump("target.final_hidden", final_hidden)
    dump("inputs_embeds", inputs_embeds)
    dump("target.argmax", target_argmax.float())
    dump("shifted_ids", shifted.float())

    dump("w.pre_fc_norm_embedding", load_f32(headers, "mtp.pre_fc_norm_embedding.weight"))
    dump("w.pre_fc_norm_hidden", load_f32(headers, "mtp.pre_fc_norm_hidden.weight"))
    dump("w.norm", load_f32(headers, "mtp.norm.weight"))
    dump("w.input_layernorm", load_f32(headers, "mtp.layers.0.input_layernorm.weight"))
    dump("w.post_attention_layernorm",
         load_f32(headers, "mtp.layers.0.post_attention_layernorm.weight"))
    dump("w.q_norm", load_f32(headers, "mtp.layers.0.self_attn.q_norm.weight"))
    dump("w.k_norm", load_f32(headers, "mtp.layers.0.self_attn.k_norm.weight"))

    dump("rope_cos", pe[0][0])
    dump("rope_sin", pe[1][0])
    for name in sorted(synth_arrays):
        dump(name, synth_arrays[name])
    dump("emb_normed", stages["emb_normed"])
    dump("hidden_normed", stages["hidden_normed"])
    dump("fc_out", stages["fc_out"])
    for name in sorted(k for k in stages if k.startswith("layer.")):
        dump(name, stages[name])
    dump("layer_out", stages["layer_out"])
    dump("output", stages["output"])
    dump("draft_argmax", stages["draft_argmax"].float())
    dump("alt_pos.draft_argmax", alt_pos_argmax.float())

    # Whole rows of `fc`, so a test can recompute a few outputs from the two
    # captured halves and find out which half of the weight multiplies which
    # input. The full matrix is 210 MB; eight rows are 320 KB and settle it.
    fc_rows = [0, 1, 2, d // 2, d - 2, d - 1]
    dump("fc_probe_rows", torch.tensor(fc_rows, dtype=torch.float32))
    dump("fc_probe_w", head.fc.weight[fc_rows])

    # Same idea for q_proj: the per-head q/gate interleave has a second reading
    # that agrees at head 0 and diverges after, so probe past the first head.
    nh, hd = cfg["num_attention_heads"], cfg["head_dim"]
    q_w = load_f32(headers, "mtp.layers.0.self_attn.q_proj.weight")
    probe = sorted({r for h in (0, 1, nh - 1) for dd in (0, 1, hd - 1)
                    for r in (h * 2 * hd + dd, h * 2 * hd + hd + dd,
                              h * hd + dd, nh * hd + h * hd + dd)})
    dump("q_proj_probe_rows", torch.tensor(probe, dtype=torch.float32))
    dump("q_proj_probe_w", q_w[probe])

    # lm_head rows for the drafted ids plus the logits they produce, so the
    # "which lm_head" question has a numeric answer and not only an index-file
    # one. The full lm_head is 5 GB.
    ids_probe = torch.unique(torch.cat([stages["draft_argmax"],
                                        target_argmax])).long()
    dump("lm_head_probe_ids", ids_probe.float())
    dump("lm_head_probe_w", lm_head_w[ids_probe])
    dump("draft_logits_probe", stages["output"] @ lm_head_w[ids_probe].T)

    if args.dump_layer_weights:
        print("  (full weights: lets the Rust reference run the whole head)")
        dump("w.fc", head.fc.weight)
        for name, tensor in (
            ("q_proj", "self_attn.q_proj"), ("k_proj", "self_attn.k_proj"),
            ("v_proj", "self_attn.v_proj"), ("o_proj", "self_attn.o_proj"),
            ("gate_proj", "mlp.gate_proj"), ("up_proj", "mlp.up_proj"),
            ("down_proj", "mlp.down_proj"),
        ):
            dump("w." + name, load_f32(headers, f"mtp.layers.0.{tensor}.weight"))
    manifest["has_layer_weights"] = bool(args.dump_layer_weights)

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {len(manifest['arrays'])} arrays to {args.out_dir}")
    if truncated:
        print("!! prefix was truncated; this capture is a smoke test, not an "
              "oracle, and the Rust test will refuse it")


if __name__ == "__main__":
    sys.exit(main())
