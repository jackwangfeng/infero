#!/usr/bin/env python3
"""Capture torch's stage-by-stage output for one GatedDeltaNet layer and one
gated-attention layer of a real Qwen3.5 checkpoint.

This exists because local self-consistency is not evidence. The bf16-as-f16
embedding bug passed nine component-level A/Bs — every stage did its job, on
faithfully-processed nonsense. The only checks that found it compared against
something outside the implementation. So before writing a single line of the
Rust or CUDA GatedDeltaNet, capture what the reference implementation actually
produces for a fixed input, at every stage, and make the port answer to that.

Writes a directory of raw little-endian f32 arrays plus a manifest.json giving
each array's shape, so a Rust test can read them with no dependencies.

    python3 capture_qwen35_layers.py <model-dir> <out-dir> [--tokens N]
"""

import argparse
import json
import os
import struct
import sys

import torch
import torch.nn.functional as F

E4M3_MAX = 448.0


def read_index(model_dir):
    """Map tensor name -> (shard path, dtype, shape, byte range)."""
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    want = idx["weight_map"]
    headers = {}
    for shard in sorted(set(want.values())):
        path = os.path.join(model_dir, shard)
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            headers[k] = (path, 8 + n, v)
    return headers


def load_raw(headers, name):
    """One tensor, as torch, without loading its whole shard."""
    path, base, meta = headers[name]
    start, end = meta["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(base + start)
        buf = fh.read(end - start)
    dt = meta["dtype"]
    if dt == "BF16":
        t = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16)
    elif dt == "F16":
        t = torch.frombuffer(bytearray(buf), dtype=torch.float16)
    elif dt == "F32":
        t = torch.frombuffer(bytearray(buf), dtype=torch.float32)
    elif dt == "F8_E4M3":
        t = torch.frombuffer(bytearray(buf), dtype=torch.float8_e4m3fn)
    else:
        raise SystemExit(f"{name}: unhandled dtype {dt}")
    return t.reshape(meta["shape"])


def load_f32(headers, name, block=128):
    """A tensor as f32, dequantizing FP8 against its block scale grid.

    The scale indexing is the part worth being explicit about: a transposed or
    row-only index does not fail, it mis-scales every tile past the first and
    reads as a model that loads and then talks nonsense.
    """
    _, _, meta = headers[name]
    q = load_raw(headers, name)
    if meta["dtype"] != "F8_E4M3":
        return q.float()
    scales = load_f32(headers, name + "_scale_inv")
    rows, cols = q.shape
    gr, gc = scales.shape
    assert (gr, gc) == (-(-rows // block), -(-cols // block)), (
        f"{name}: quants {rows}x{cols} imply a {-(-rows//block)}x{-(-cols//block)} "
        f"scale grid at block {block}, got {gr}x{gc}"
    )
    out = q.float()
    # Expand the grid to the quant shape rather than looping, then multiply.
    full = scales.repeat_interleave(block, 0).repeat_interleave(block, 1)
    return out * full[:rows, :cols]


def check_fp8_dequant_against_vllm(chk, headers, names):
    """`load_f32`'s block dequantization, against vLLM's `block_dequant`.

    This is the one transcription in this file that sits *underneath* everything
    else: every capture, every Rust test and the whole comparison chain rests on
    these weights being the weights. And its failure mode is the one that started
    all of this — the bf16-as-f16 embedding bug — because a mis-scaled tile is
    faithfully processed nonsense. Specifically, a scale grid indexed
    `[col_tile][row_tile]` instead of `[row_tile][col_tile]` produces a tensor of
    the right shape whose first tile is correct, which is exactly enough to make
    a spot check pass.

    vLLM's `block_dequant` is separate code with the same semantics (it lives in
    `int8_utils` but is dtype-agnostic: cast, then multiply tile `[j][i]`), so it
    is an independent oracle for the indexing. The transposed reading has to
    disagree, or this check says only that some multiplication happened.
    """
    from vllm.model_executor.layers.quantization.utils.int8_utils import block_dequant

    block = 128
    checked = 0
    for name in names:
        if headers[name][2]["dtype"] != "F8_E4M3":
            continue
        q = load_raw(headers, name)
        scales = load_f32(headers, name + "_scale_inv")
        if scales.shape[0] == scales.shape[1]:
            # A square scale grid cannot distinguish the two indexings, so it is
            # useless for this check; skip rather than pretend.
            continue
        mine = load_f32(headers, name)
        ref = block_dequant(q.clone(), scales, [block, block])
        peak = ref.abs().max()
        chk(f"load_f32 dequant of {name.split('.')[-2]} vs vLLM block_dequant",
            (mine - ref).abs().max() / peak, 0)
        # The two indexings that run and mis-scale every tile past the first: a
        # transposed lookup, and a row-only one that ignores the column tile.
        rows, cols = q.shape
        gr, gc = scales.shape
        for label, pick in (
            ("transposed scale lookup",
             lambda j, i: scales[min(i, gr - 1)][min(j, gc - 1)]),
            ("row-only scale lookup", lambda j, i: scales[j][0]),
        ):
            wrong = q.float()
            for i in range(gc):
                for j in range(gr):
                    wrong[j * block:min((j + 1) * block, rows),
                          i * block:min((i + 1) * block, cols)] *= pick(j, i)
            sep = (wrong - ref).abs().max().item() / float(peak)
            print(f"  (a {label} on {name.split('.')[-2]} is off by "
                  f"{sep:.3e} of peak; grid is {gr}x{gc})")
            if sep < 1e-2:
                raise SystemExit(f"a {label} reproduces the reference "
                                 f"dequantization; this check does not "
                                 f"discriminate")
        checked += 1
    if not checked:
        print("  !! no non-square FP8 scale grid among the probed tensors, so "
              "the dequantization's scale indexing is unchecked")
    return checked


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def rms_norm(x, w, eps=1e-6, gain_offset=0.0):
    """RMSNorm with a learned gain, and the offset that decides which class.

    `Qwen3_5RMSNorm` stores its weight as a delta from one and computes
    `normalized * (1 + w)`; `Qwen3_5RMSNormGated` stores a gain and computes
    `w * normalized`. Pass 1.0 for the former — every regular norm in the text
    model, including q_norm and k_norm — and 0.0 for the latter, which is only
    `linear_attn.norm`.

    An earlier version of this file had no offset, so it and
    `qwen35.rs::rms_norm_rows` agreed with each other and both got q_norm and
    k_norm wrong. Neither was checked against the library, which is the one
    stage of this capture that was not — see cross_check_against_transformers.
    """
    v = x.float().pow(2).mean(-1, keepdim=True)
    return (gain_offset + w) * (x.float() * torch.rsqrt(v + eps))


def recurrent_gated_delta(q, k, v, g, beta):
    """The reference recurrence, transcribed from
    transformers.models.qwen3_5.modeling_qwen3_5.torch_recurrent_gated_delta_rule.

    q, k: [T, H, Dk]   v: [T, H, Dv]   g, beta: [T, H]
    Returns (out [T, H, Dv], final state [H, Dk, Dv]).
    """
    T, H, dk = k.shape
    dv = v.shape[-1]
    q = l2norm(q.float()) * (dk ** -0.5)
    k = l2norm(k.float())
    v = v.float()
    S = torch.zeros(H, dk, dv, dtype=torch.float32)
    out = torch.zeros(T, H, dv, dtype=torch.float32)
    for t in range(T):
        S = S * g[t].exp().view(H, 1, 1)
        kv_mem = (S * k[t].unsqueeze(-1)).sum(dim=-2)          # kᵀS  -> [H, dv]
        delta = (v[t] - kv_mem) * beta[t].unsqueeze(-1)         # [H, dv]
        S = S + k[t].unsqueeze(-1) * delta.unsqueeze(-2)        # outer product
        out[t] = (S * q[t].unsqueeze(-1)).sum(dim=-2)           # qᵀS
    return out, S


# --------------------------------------------------------- transcribed blocks
#
# Everything this file computes for itself lives in the four functions below, so
# that `cross_check_against_transformers` can run *this* code against the
# reference's own modules. A cross-check that exercises a second copy of the
# arithmetic is the same mistake as a capture that agrees with the port it is
# supposed to be checking — one level up, and just as invisible.


def rope_tables(theta, rot, positions):
    """Partial-rope cos/sin, `[T, rot]` each, in the `rotate_half` layout.

    The exponent is normalized by `rot`, not by `head_dim`: this is a different
    table, not the leading slice of the full-width one. Checked against
    `Qwen3_5TextRotaryEmbedding` below, which is where the `int(head_dim *
    partial_rotary_factor)` and the `rope_parameters` lookup are settled too.
    """
    inv = 1.0 / (theta ** (torch.arange(0, rot, 2, dtype=torch.float32) / rot))
    freqs = positions[:, None].float() * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)              # [T, rot]
    return emb.cos(), emb.sin()


def apply_partial_rope(t, cos, sin):
    """`[T, heads, head_dim]` in, the first `cos.shape[-1]` dims rotated.

    Pairing is `(i, i + rot/2)` — `rotate_half`, not the adjacent-pair form.
    """
    rot = cos.shape[-1]
    r, keep = t[..., :rot], t[..., rot:]
    h = rot // 2
    rotated = torch.cat([-r[..., h:], r[..., :h]], dim=-1)
    return torch.cat([r * cos[:, None, :] + rotated * sin[:, None, :], keep], dim=-1)


def gated_delta_net_stages(x, w, d):
    """One `Qwen3_5GatedDeltaNet` block, stage by stage, in f32.

    `x` is `[T, hidden]`, `w` maps tensor suffixes to f32 weights, `d` carries
    `nk / nv / dk / dv / ksz / eps`. Returns every intermediate, keyed as the
    dumps are named.
    """
    T = x.shape[0]
    nk, nv, dk, dv, ksz = d["nk"], d["nv"], d["dk"], d["dv"], d["ksz"]
    key_dim, val_dim = nk * dk, nv * dv
    s = {}

    qkv = x @ w["in_proj_qkv"].T
    s["z"] = x @ w["in_proj_z"].T
    s["a"] = x @ w["in_proj_a"].T
    s["b"] = x @ w["in_proj_b"].T
    s["qkv_pre_conv"] = qkv

    # Depthwise causal conv, then silu. `causal_conv1d_fn` is exactly
    # `conv1d(padding=k-1)[..., :T]` followed by `ACT2FN[hidden_act]`, and the
    # left padding plus the truncation is what makes it causal; reversing the
    # taps runs and shifts the model one token into the future.
    conv_out = F.conv1d(qkv.T.unsqueeze(0), w["conv1d"].unsqueeze(1), None,
                        padding=ksz - 1, groups=qkv.shape[-1])[:, :, :T]
    qkv_c = F.silu(conv_out).squeeze(0).T
    s["qkv_post_conv"] = qkv_c

    q, k, v = torch.split(qkv_c, [key_dim, key_dim, val_dim], dim=-1)
    q = q.reshape(T, nk, dk)
    k = k.reshape(T, nk, dk)
    v = v.reshape(T, nv, dv)

    s["beta"] = torch.sigmoid(s["b"])
    s["g"] = -w["A_log"].exp() * F.softplus(s["a"] + w["dt_bias"])

    rep = nv // nk
    q = q.repeat_interleave(rep, dim=1)
    k = k.repeat_interleave(rep, dim=1)

    core, state = recurrent_gated_delta(q, k, v, s["g"], s["beta"])
    s["core_attn_out"], s["final_state"] = core, state

    # Qwen3_5RMSNormGated: a plain gain, no offset, and it normalizes *before*
    # it gates. The gate is silu, not sigmoid.
    normed = rms_norm(core.reshape(-1, dv), w["norm"], d["eps"], gain_offset=0.0)
    gated = normed * F.silu(s["z"].reshape(-1, dv).float())
    s["after_gated_norm"] = gated
    s["output"] = gated.reshape(T, val_dim) @ w["out_proj"].T
    return s


def gated_attention_stages(x, w, d, cos, sin):
    """One `Qwen3_5Attention` block, stage by stage, in f32.

    `x` is `[T, hidden]`; `d` carries `nh / nkv / hd / eps`. The decisions this
    encodes, each of which has a second reading that runs to completion:

    * `q_proj`'s output is viewed `[T, heads, 2 * head_dim]` and split on the
      LAST axis, so q and its gate interleave per head.
    * `q_norm` / `k_norm` are `Qwen3_5RMSNorm` — the `(1 + w)` offset form — and
      run per head, before rope.
    * `1/sqrt(head_dim)` multiplies the *scores*, not q or k.
    * The keys are expanded to the query head count by `repeat_kv`, which is
      `repeat_interleave`, not a stride.
    * The output gate is `sigmoid` and lands before `o_proj`.
      `config.output_gate_type` says `"swish"` and is never read.
    """
    T = x.shape[0]
    nh, nkv, hd = d["nh"], d["nkv"], d["hd"]
    s = {}

    qg = (x @ w["q_proj"].T).reshape(T, nh, 2 * hd)
    q, gate = qg[..., :hd], qg[..., hd:]
    s["q_pre_norm"] = q.contiguous()
    s["gate"] = gate.reshape(T, nh * hd).contiguous()

    k = (x @ w["k_proj"].T).reshape(T, nkv, hd)
    v = (x @ w["v_proj"].T).reshape(T, nkv, hd)
    q = rms_norm(q, w["q_norm"], d["eps"], gain_offset=1.0)
    k = rms_norm(k, w["k_norm"], d["eps"], gain_offset=1.0)
    s["q_post_norm"], s["k_post_norm"] = q, k

    q, k = apply_partial_rope(q, cos, sin), apply_partial_rope(k, cos, sin)
    s["q_post_rope"], s["k_post_rope"] = q, k

    grp = nh // nkv
    kk = k.repeat_interleave(grp, dim=1)
    vv = v.repeat_interleave(grp, dim=1)
    scores = torch.einsum("thd,shd->hts", q, kk) * (hd ** -0.5)
    mask = torch.triu(torch.full((T, T), float("-inf")), 1)
    probs = torch.softmax(scores + mask, dim=-1)
    ctx = torch.einsum("hts,shd->thd", probs, vv).reshape(T, nh * hd)
    s["attn_out_pre_gate"] = ctx

    ctx = ctx * torch.sigmoid(gate.reshape(T, nh * hd))
    s["attn_out_post_gate"] = ctx
    s["output"] = ctx @ w["o_proj"].T
    return s


# ------------------------------------------------------------- cross-checking


def _small_text_config(nh, nkv, hd, d_model, nk, nv, dk, dv, ksz, rot_factor):
    """A `Qwen3_5TextConfig` small enough to instantiate both block types.

    Tiny on purpose: the layout questions do not get easier at 5120 wide, and a
    check that runs in a second is a check that runs.
    """
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=d_model, intermediate_size=3 * d_model,
        num_hidden_layers=4, num_attention_heads=nh, num_key_value_heads=nkv,
        head_dim=hd, rms_norm_eps=1e-6, hidden_act="silu",
        linear_conv_kernel_dim=ksz, linear_key_head_dim=dk,
        linear_value_head_dim=dv, linear_num_key_heads=nk,
        linear_num_value_heads=nv,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 1e7,
                         "partial_rotary_factor": rot_factor,
                         "mrope_section": [hd // 8, hd // 8, hd // 8],
                         "mrope_interleaved": True},
    )
    cfg._attn_implementation = "eager"
    return cfg


def _randomize(mod, seed):
    """Random weights, so the check is about the arithmetic and not the data."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in mod.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
    return mod


def check_rope_table_against_reference(chk, cfg_obj, theta, rot):
    """The partial-rope table, against `Qwen3_5TextRotaryEmbedding`.

    This is the one stage of the attention capture that a hook cannot see and
    that has a famous silent misreading: normalizing the frequency exponent by
    `head_dim` instead of by `rot`. Both give a `[T, rot]` table, both run, and
    the wrong one degrades only the long-distance dimensions — which reads as a
    context-length problem rather than a config bug. So compare against the
    reference's own `inv_freq` *and* record that the head_dim-normalized reading
    is far outside the tolerance.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    ref = m.Qwen3_5TextRotaryEmbedding(cfg_obj)
    hd = cfg_obj.head_dim
    assert ref.inv_freq.shape[0] == rot // 2, (
        f"the reference builds {ref.inv_freq.shape[0]} frequencies, this capture "
        f"assumes rot//2 = {rot // 2}; partial_rotary_factor is being read from "
        f"the wrong place")

    # `position_ids` given as [batch, seq] is expanded to three identical rows,
    # so `apply_interleaved_mrope` is a no-op and what comes back is the plain
    # partial table this capture builds. (The interleaving itself is pinned in
    # tools/capture_qwen35_vision.py, where it is read out of the reference.)
    pos = torch.tensor([[0, 1, 2, 7, 4095, 130000]])
    cos_ref, sin_ref = ref(torch.zeros(1, pos.shape[1], 1), pos)
    cos, sin = rope_tables(theta, rot, pos[0])
    # f32 at position 130000 is a 2.5e-3-wide cloud; compare in f64 through the
    # frequency instead, then the angles at ordinary positions.
    chk("rope inv_freq vs Qwen3_5TextRotaryEmbedding",
        (1.0 / (theta ** (torch.arange(0, rot, 2, dtype=torch.float64) / rot))
         - ref.inv_freq.double()).abs().max(), 1e-7)
    chk("rope cos vs Qwen3_5TextRotaryEmbedding (near positions)",
        (cos[:4] - cos_ref[0, :4]).abs().max(), 2e-6)
    chk("rope sin vs Qwen3_5TextRotaryEmbedding (near positions)",
        (sin[:4] - sin_ref[0, :4]).abs().max(), 2e-6)

    # The mistake that matters, measured rather than assumed away.
    wrong = 1.0 / (theta ** (torch.arange(0, rot, 2, dtype=torch.float64) / hd))
    sep = (wrong - ref.inv_freq.double()).abs().max().item()
    print(f"  (normalizing the exponent by head_dim={hd} instead of rot={rot} "
          f"moves inv_freq by {sep:.3e})")
    if sep < 1e-2:
        raise SystemExit("the head_dim-normalized frequency table is "
                         "indistinguishable here; this check is decorative")


def check_gated_attention_against_reference(chk):
    """The whole gated-attention interior, against `Qwen3_5Attention`.

    One check, eight decisions: the per-head q/gate interleave, that q_norm and
    k_norm are the offset `Qwen3_5RMSNorm` applied per head before rope, the
    partial rope table and its `rotate_half` pairing, the `1/sqrt(head_dim)`
    scale landing on the scores, the softmax and its causal mask, the
    `repeat_kv` key expansion, and that the output gate is `sigmoid` applied
    before `o_proj`.

    Every one of those was previously believed on the strength of having read
    the source. `gated_attention_stages` is the code the capture runs, so this
    is a check on the capture and not on a copy of it.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    d_model, nh, nkv, hd, rot = 40, 6, 2, 8, 4
    cfg = _small_text_config(nh, nkv, hd, d_model, 2, 6, 6, 6, 4, rot / hd)
    attn = _randomize(m.Qwen3_5Attention(cfg, 3).eval(), 4242)

    assert type(attn.q_norm).__name__ == "Qwen3_5RMSNorm", type(attn.q_norm)
    assert type(attn.k_norm).__name__ == "Qwen3_5RMSNorm", type(attn.k_norm)
    chk("Qwen3_5Attention.scaling == head_dim**-0.5",
        abs(attn.scaling - hd ** -0.5), 0)
    assert attn.q_proj.bias is None and attn.o_proj.bias is None, (
        "the reference attention has projection biases on this config; the "
        "capture and the Rust reference are both bias-free")

    T = 9
    torch.manual_seed(99)
    x = torch.randn(T, d_model)
    positions = torch.arange(T)
    cos, sin = rope_tables(1e7, rot, positions)

    mask = torch.zeros(T, T)
    mask.masked_fill_(torch.triu(torch.ones(T, T, dtype=torch.bool), 1),
                      float("-inf"))
    with torch.no_grad():
        ref_out, _ = attn(x[None], position_embeddings=(cos[None], sin[None]),
                          attention_mask=mask[None, None])

    w = {"q_proj": attn.q_proj.weight, "k_proj": attn.k_proj.weight,
         "v_proj": attn.v_proj.weight, "o_proj": attn.o_proj.weight,
         "q_norm": attn.q_norm.weight, "k_norm": attn.k_norm.weight}
    d = {"nh": nh, "nkv": nkv, "hd": hd, "eps": cfg.rms_norm_eps}
    with torch.no_grad():
        mine = gated_attention_stages(x, w, d, cos, sin)
    peak = ref_out.abs().max().item()
    chk("gated_attention_stages vs Qwen3_5Attention",
        (mine["output"] - ref_out[0]).abs().max() / peak, 1e-5)

    # And the readings that also run must not reproduce it, or the check above
    # says only that some arithmetic ran.
    with torch.no_grad():
        alts = {}
        # [all q | all gate] instead of interleaved per head.
        flat = x @ attn.q_proj.weight.T
        alt = torch.cat([flat[:, : nh * hd].reshape(T, nh, hd),
                         flat[:, nh * hd:].reshape(T, nh, hd)], dim=-1)
        alts["q/gate in halves, not per head"] = _alt_output(
            alt.reshape(T, nh * 2 * hd), w, d, cos, sin, nh, nkv, hd, x)
        # silu instead of sigmoid on the gate.
        gate = mine["gate"]
        alts["silu gate instead of sigmoid"] = (
            mine["attn_out_pre_gate"] * F.silu(gate)) @ w["o_proj"].T
        # The plain RMSNorm reading of q_norm / k_norm.
        d0 = dict(d)
        alts["plain `w *` q/k norm"] = _alt_norm_output(x, w, d0, cos, sin,
                                                        nh, nkv, hd)
        # 1/sqrt(head_dim) on q instead of on the scores is the *same* function,
        # so it is deliberately not listed: scaling q before the dot product and
        # scaling the dot product are identical up to f32 rounding. What is not
        # identical is scaling by 1/sqrt(d_model) or by 1/d — check that.
        kk = mine["k_post_rope"].repeat_interleave(nh // nkv, dim=1)
        vv = (x @ w["v_proj"].T).reshape(T, nkv, hd).repeat_interleave(
            nh // nkv, dim=1)
        sc = torch.einsum("thd,shd->hts", mine["q_post_rope"], kk) * (1.0 / hd)
        sc = sc + torch.triu(torch.full((T, T), float("-inf")), 1)
        ctx = torch.einsum("hts,shd->thd", torch.softmax(sc, -1), vv)
        alts["1/head_dim instead of 1/sqrt(head_dim)"] = (
            ctx.reshape(T, nh * hd) * torch.sigmoid(gate)) @ w["o_proj"].T

    for name, alt in alts.items():
        sep = (alt - ref_out[0]).abs().max().item() / peak
        print(f"  (the reading `{name}` is off by {sep:.3e} of peak)")
        if sep < 1e-2:
            raise SystemExit(
                f"the alternative reading `{name}` reproduces "
                f"Qwen3_5Attention to {sep:.2e}; this check cannot tell the two "
                f"apart and is not evidence about the choice")


def _alt_output(qg_flat, w, d, cos, sin, nh, nkv, hd, x):
    """The `[all q | all gate]` reading, carried through to the output."""
    T = x.shape[0]
    qg = qg_flat.reshape(T, nh, 2 * hd)
    q, gate = qg[..., :hd], qg[..., hd:]
    k = (x @ w["k_proj"].T).reshape(T, nkv, hd)
    v = (x @ w["v_proj"].T).reshape(T, nkv, hd)
    q = rms_norm(q, w["q_norm"], d["eps"], gain_offset=1.0)
    k = rms_norm(k, w["k_norm"], d["eps"], gain_offset=1.0)
    q, k = apply_partial_rope(q, cos, sin), apply_partial_rope(k, cos, sin)
    grp = nh // nkv
    sc = torch.einsum("thd,shd->hts", q, k.repeat_interleave(grp, dim=1))
    sc = sc * (hd ** -0.5) + torch.triu(torch.full((T, T), float("-inf")), 1)
    ctx = torch.einsum("hts,shd->thd", torch.softmax(sc, -1),
                       v.repeat_interleave(grp, dim=1)).reshape(T, nh * hd)
    return (ctx * torch.sigmoid(gate.reshape(T, nh * hd))) @ w["o_proj"].T


def _alt_norm_output(x, w, d, cos, sin, nh, nkv, hd):
    """q_norm / k_norm read as the plain `w *` form."""
    T = x.shape[0]
    qg = (x @ w["q_proj"].T).reshape(T, nh, 2 * hd)
    q, gate = qg[..., :hd], qg[..., hd:]
    k = (x @ w["k_proj"].T).reshape(T, nkv, hd)
    v = (x @ w["v_proj"].T).reshape(T, nkv, hd)
    q = rms_norm(q, w["q_norm"], d["eps"], gain_offset=0.0)
    k = rms_norm(k, w["k_norm"], d["eps"], gain_offset=0.0)
    q, k = apply_partial_rope(q, cos, sin), apply_partial_rope(k, cos, sin)
    grp = nh // nkv
    sc = torch.einsum("thd,shd->hts", q, k.repeat_interleave(grp, dim=1))
    sc = sc * (hd ** -0.5) + torch.triu(torch.full((T, T), float("-inf")), 1)
    ctx = torch.einsum("hts,shd->thd", torch.softmax(sc, -1),
                       v.repeat_interleave(grp, dim=1)).reshape(T, nh * hd)
    return (ctx * torch.sigmoid(gate.reshape(T, nh * hd))) @ w["o_proj"].T


def check_gated_delta_net_against_reference(chk):
    """The whole GatedDeltaNet interior, against `Qwen3_5GatedDeltaNet`.

    Covers what the recurrence check alone does not: that the convolution is
    left-padded and truncated (causal, current token on the last tap) with
    `silu` after it, where `[q | k | v]` splits inside the packed row,
    `beta = sigmoid(b)`, `g = -exp(A_log) * softplus(a + dt_bias)`, that the 16
    key heads expand to 48 value heads by `repeat_interleave` rather than by a
    stride, and that the output norm gates with `silu(z)` after normalizing.

    The reference takes its chunked prefill path here (no cache), so the
    tolerance is the chunk-vs-recurrence gap the existing check measures at
    ~5e-8 relative, not zero.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    d_model, nk, nv, dk, dv, ksz = 40, 2, 6, 6, 6, 4
    cfg = _small_text_config(6, 2, 8, d_model, nk, nv, dk, dv, ksz, 0.5)
    gdn = _randomize(m.Qwen3_5GatedDeltaNet(cfg, 0).eval(), 777)
    with torch.no_grad():
        # A_log is log of a positive number in the real init; a random negative
        # weight there would make exp(A_log) meaningless rather than merely
        # different.
        gen = torch.Generator().manual_seed(778)
        gdn.A_log.copy_(torch.log(torch.rand(nv, generator=gen) * 15.5 + 0.5))
        gdn.dt_bias.copy_(torch.randn(nv, generator=gen))

    assert type(gdn.norm).__name__ == "Qwen3_5RMSNormGated", type(gdn.norm)
    assert gdn.activation == "silu", gdn.activation
    assert gdn.conv1d.bias is None, "the reference conv1d has a bias here"

    T = 11
    torch.manual_seed(555)
    x = torch.randn(T, d_model)
    with torch.no_grad():
        ref_out = gdn(x[None])[0]

    w = {"in_proj_qkv": gdn.in_proj_qkv.weight, "in_proj_z": gdn.in_proj_z.weight,
         "in_proj_a": gdn.in_proj_a.weight, "in_proj_b": gdn.in_proj_b.weight,
         "conv1d": gdn.conv1d.weight.squeeze(1), "A_log": gdn.A_log,
         "dt_bias": gdn.dt_bias, "norm": gdn.norm.weight,
         "out_proj": gdn.out_proj.weight}
    d = {"nk": nk, "nv": nv, "dk": dk, "dv": dv, "ksz": ksz,
         "eps": cfg.rms_norm_eps}
    with torch.no_grad():
        mine = gated_delta_net_stages(x, w, d)
    peak = ref_out.abs().max().item()
    chk("gated_delta_net_stages vs Qwen3_5GatedDeltaNet",
        (mine["output"] - ref_out).abs().max() / peak, 2e-3)

    # The alternatives that also run.
    key_dim, val_dim = nk * dk, nv * dv
    with torch.no_grad():
        alts = {}
        qkv_c = mine["qkv_post_conv"]
        z, g, beta = mine["z"], mine["g"], mine["beta"]
        q0, k0, v0 = torch.split(qkv_c, [key_dim, key_dim, val_dim], dim=-1)
        q0 = q0.reshape(T, nk, dk)
        k0 = k0.reshape(T, nk, dk)
        v0 = v0.reshape(T, nv, dv)
        rep = nv // nk

        def finish(q, k, v, g_, beta_, gate_first=False):
            core, _ = recurrent_gated_delta(q, k, v, g_, beta_)
            flat = core.reshape(-1, dv)
            if gate_first:
                out = rms_norm(flat * F.silu(z.reshape(-1, dv).float()),
                               w["norm"], d["eps"], gain_offset=0.0)
            else:
                out = rms_norm(flat, w["norm"], d["eps"], gain_offset=0.0) \
                    * F.silu(z.reshape(-1, dv).float())
            return out.reshape(T, val_dim) @ w["out_proj"].T

        # Modular head expansion rather than repeat_interleave.
        idx = torch.arange(nv) % nk
        alts["head expansion by stride, not repeat_interleave"] = finish(
            q0[:, idx], k0[:, idx], v0, g, beta)
        # Gate before normalize.
        alts["gate before normalize"] = finish(
            q0.repeat_interleave(rep, 1), k0.repeat_interleave(rep, 1), v0,
            g, beta, gate_first=True)
        # sigmoid instead of silu on z.
        core, _ = recurrent_gated_delta(q0.repeat_interleave(rep, 1),
                                        k0.repeat_interleave(rep, 1), v0, g, beta)
        alts["sigmoid(z) instead of silu(z)"] = (
            rms_norm(core.reshape(-1, dv), w["norm"], d["eps"], gain_offset=0.0)
            * torch.sigmoid(z.reshape(-1, dv).float())
        ).reshape(T, val_dim) @ w["out_proj"].T
        # The offset RMSNorm reading of the *gated* norm.
        alts["offset (1+w) reading of the gated norm"] = (
            rms_norm(core.reshape(-1, dv), w["norm"], d["eps"], gain_offset=1.0)
            * F.silu(z.reshape(-1, dv).float())
        ).reshape(T, val_dim) @ w["out_proj"].T
        # beta as silu instead of sigmoid, and g without the softplus.
        alts["beta = silu(b) instead of sigmoid(b)"] = finish(
            q0.repeat_interleave(rep, 1), k0.repeat_interleave(rep, 1), v0,
            g, F.silu(mine["b"]))
        # `dt_bias` dropped from the softplus argument. Not "softplus removed
        # entirely", which overflows exp(g) into NaN and so measures nothing.
        alts["dt_bias dropped from g"] = finish(
            q0.repeat_interleave(rep, 1), k0.repeat_interleave(rep, 1), v0,
            -w["A_log"].exp() * F.softplus(mine["a"]), beta)
        # Reversed convolution taps: same shape, one token of look-ahead.
        rev = F.conv1d(mine["qkv_pre_conv"].T.unsqueeze(0),
                       w["conv1d"].flip(-1).unsqueeze(1), None,
                       padding=ksz - 1, groups=2 * key_dim + val_dim)[:, :, :T]
        rq, rk, rv = torch.split(F.silu(rev).squeeze(0).T,
                                 [key_dim, key_dim, val_dim], dim=-1)
        alts["reversed convolution taps"] = finish(
            rq.reshape(T, nk, dk).repeat_interleave(rep, 1),
            rk.reshape(T, nk, dk).repeat_interleave(rep, 1),
            rv.reshape(T, nv, dv), g, beta)

    for name, alt in alts.items():
        sep = (alt - ref_out).abs().max().item() / peak
        print(f"  (the reading `{name}` is off by {sep:.3e} of peak)")
        if sep < 1e-2:
            raise SystemExit(
                f"the alternative reading `{name}` reproduces "
                f"Qwen3_5GatedDeltaNet to {sep:.2e}; this check cannot tell the "
                f"two apart and is not evidence about the choice")

    # The decode path's convolution window, against `causal_conv1d_update`.
    #
    # `qwen35.rs::depthwise_causal_conv1d_update` is pinned in the Rust tests by
    # a property — splitting a sequence must reproduce the single-call answer,
    # which does force the window to carry the right values in the right order.
    # But nothing checked the *convention*, and the reference states one: the
    # state is the last `k-1` inputs, newest last, and the update overwrites it
    # in place. A window that is right for this implementation and shifted by one
    # relative to the reference would pass the property and, at decode time, read
    # a token late — three stale taps being three tokens of the previous
    # conversation, which is short enough to read as a bad sample.
    conv_dim = 2 * key_dim + val_dim
    with torch.no_grad():
        whole = F.silu(F.conv1d(mine["qkv_pre_conv"].T.unsqueeze(0),
                                w["conv1d"].unsqueeze(1), None,
                                padding=ksz - 1, groups=conv_dim)[:, :, :T])
        # Feed the same sequence one token at a time through the reference's
        # cached-decode kernel, starting from a zero window.
        state = torch.zeros(1, conv_dim, ksz)
        step_out = []
        for t in range(T):
            step_out.append(m.causal_conv1d_update(
                mine["qkv_pre_conv"][t][None, :, None], state,
                w["conv1d"], None, "silu"))
        stepped = torch.cat(step_out, dim=-1)
    chk("causal_conv1d_update stepped == conv1d whole-sequence",
        (stepped - whole).abs().max() / whole.abs().max(), 2e-6)
    # And the window the reference left behind is the last k-1 inputs, newest
    # last — the layout `depthwise_causal_conv1d_update` keeps.
    tail = mine["qkv_pre_conv"][T - (ksz - 1):].T                 # [C, k-1]
    chk("the conv state is the last k-1 inputs, newest last",
        (state[0, :, -(ksz - 1):] - tail).abs().max(), 0)
    sep = (state[0, :, -(ksz - 1):] - tail.flip(-1)).abs().max().item()
    print(f"  (a window stored oldest-last instead differs by {sep:.3e})")
    if sep < 1e-2:
        raise SystemExit("the two window orders agree on this input; the "
                         "convention is not pinned")


def cross_check_against_transformers(cfg, headers=None):
    """Check the transcription above against the library it was transcribed from.

    Without this the capture is only my reading of the reference, and my reading
    is the thing under test. Calls transformers' own functions on random input
    and requires agreement.
    """
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    except Exception as e:  # noqa: BLE001
        print(f"!! cannot import the reference module ({e}); "
              f"the capture is unverified — do not trust it as an oracle")
        return False

    torch.manual_seed(7)
    T, H, dk, dv = 9, 6, 16, 16
    q = torch.randn(1, T, H, dk)
    k = torch.randn(1, T, H, dk)
    v = torch.randn(1, T, H, dv)
    g = -torch.rand(1, T, H) * 0.5
    beta = torch.rand(1, T, H)

    ref_out, ref_state = m.torch_recurrent_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=None,
        output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    mine_out, mine_state = recurrent_gated_delta(q[0], k[0], v[0], g[0], beta[0])

    d_out = (ref_out[0].float() - mine_out).abs().max().item()
    d_st = (ref_state[0].float() - mine_state).abs().max().item()
    ok = d_out < 2e-5 and d_st < 2e-5
    print(f"cross-check vs transformers: out Δ={d_out:.2e}  state Δ={d_st:.2e}  "
          f"{'agree' if ok else 'DISAGREE'}")

    # And the chunked path, which is what prefill actually uses, has to agree
    # with the recurrence too — otherwise a prefill/decode split silently
    # produces two different models.
    try:
        ch_out, ch_state = m.torch_chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=None,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
        d_ch = (ch_out[0].float() - mine_out).abs().max().item()
        print(f"chunked vs recurrent:       out Δ={d_ch:.2e}  "
              f"{'agree' if d_ch < 2e-3 else 'DISAGREE'}")
        ok = ok and d_ch < 2e-3
    except Exception as e:  # noqa: BLE001
        print(f"   (chunked path not runnable here: {e})")

    # The norms. This check was missing, and its absence is exactly why the
    # offset form went unnoticed: the recurrence was checked against the library
    # and the norms were not, so `rms_norm` here and `rms_norm_rows` in Rust
    # agreed with each other on a reading neither had confirmed.
    torch.manual_seed(11)
    x = torch.randn(4, 64)
    w = torch.randn(64) * 0.3

    plain = m.Qwen3_5RMSNormGated(64, eps=1e-6)
    with torch.no_grad():
        plain.weight.copy_(w)
    # The gated class needs a gate; a gate of large positive values makes
    # silu(gate) ~= gate, so divide it back out to isolate the norm.
    gate = torch.full_like(x, 30.0)
    ref_plain = plain(x, gate) / torch.nn.functional.silu(gate)
    mine_plain = rms_norm(x, w, 1e-6, gain_offset=0.0)
    d_plain = (ref_plain - mine_plain).abs().max().item()

    offset = m.Qwen3_5RMSNorm(64, eps=1e-6)
    with torch.no_grad():
        offset.weight.copy_(w)
    ref_offset = offset(x)
    mine_offset = rms_norm(x, w, 1e-6, gain_offset=1.0)
    d_offset = (ref_offset - mine_offset).abs().max().item()

    print(f"Qwen3_5RMSNormGated (plain w):  Δ={d_plain:.2e}")
    print(f"Qwen3_5RMSNorm      ((1+w)  ):  Δ={d_offset:.2e}")
    # And the two forms must differ, or the offset is not pinned by anything.
    spread = (ref_offset - ref_plain).abs().max().item()
    print(f"the two forms differ by {spread:.2e}, so the offset matters")
    ok = ok and d_plain < 2e-5 and d_offset < 2e-5 and spread > 1e-2
    if not (d_plain < 2e-5 and d_offset < 2e-5):
        print("DISAGREE on the norms")

    # Where the epsilon goes. The two checks above run at unit RMS, where
    # `mean(x^2)` is a million times eps and all three placements — eps on the
    # mean of squares (the reference), on the *sum* of squares, or added to the
    # root — agree to 5e-7. That is inside the tolerance, so those checks say
    # nothing about the placement. Drive the norm at an RMS near sqrt(eps)
    # instead, which is the regime a near-dead channel actually lands in, and the
    # three separate by tens of percent.
    torch.manual_seed(13)
    tiny = torch.randn(4, 64) * 1e-3          # mean(x^2) ~= eps
    ref_tiny = offset(tiny)
    mine_tiny = rms_norm(tiny, w, 1e-6, gain_offset=1.0)
    peak_tiny = ref_tiny.abs().max().item()
    d_tiny = (ref_tiny - mine_tiny).abs().max().item() / peak_tiny
    ms = tiny.pow(2).mean(-1, keepdim=True)
    on_sum = (1.0 + w) * (tiny * torch.rsqrt(tiny.pow(2).sum(-1, keepdim=True) + 1e-6))
    on_root = (1.0 + w) * (tiny / (ms.sqrt() + 1e-6))
    d_sum = (on_sum - ref_tiny).abs().max().item() / peak_tiny
    d_root = (on_root - ref_tiny).abs().max().item() / peak_tiny
    print(f"eps on the mean of squares at RMS~sqrt(eps): Δ={d_tiny:.2e} of peak "
          f"(on the sum instead: {d_sum:.1%}; added to the root: {d_root:.1%})")
    ok = ok and d_tiny < 2e-5
    if d_sum < 1e-2 or d_root < 1e-2:
        raise SystemExit("the epsilon placements are indistinguishable even at "
                         "RMS ~ sqrt(eps); this check is decorative")

    # Same question for the l2 normalization inside the recurrence, whose eps is
    # a literal 1e-6 in the reference (`l2norm(x, dim=-1, eps=1e-6)`) and is
    # *not* `rms_norm_eps` — they coincide on this checkpoint. The recurrence
    # check above runs at unit norm, where the placement is invisible.
    row = torch.randn(5, 16) * 1e-4
    ref_l2 = m.l2norm(row)
    d_l2 = (ref_l2 - l2norm(row)).abs().max().item() / ref_l2.abs().max().item()
    alt_norm = row / (row.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6)
    alt_mean = row * torch.rsqrt(row.pow(2).mean(-1, keepdim=True) + 1e-6)
    s_norm = (alt_norm - ref_l2).abs().max().item() / ref_l2.abs().max().item()
    s_mean = (alt_mean - ref_l2).abs().max().item() / ref_l2.abs().max().item()
    print(f"l2norm eps on the sum of squares: Δ={d_l2:.2e} of peak "
          f"(added to the norm: {s_norm:.1%}; on the mean: {s_mean:.1%})")
    ok = ok and d_l2 < 2e-6
    if s_norm < 1e-2 or s_mean < 1e-2:
        raise SystemExit("the l2norm epsilon placements are indistinguishable; "
                         "this check is decorative")

    # The two block interiors, against the reference's own module classes. These
    # cover every stage a forward hook cannot see and that nothing above touched:
    # the rope table and its application, the attention scale and softmax, the
    # GQA key expansion, the sigmoid output gate, the causal convolution, beta
    # and g, and the 16-to-48 head expansion.
    def chk(name, delta, tol):
        nonlocal ok
        dd = float(torch.as_tensor(delta).detach())
        print(f"  {name:<52} Δ={dd:.3e}  tol={tol:.0e}  "
              f"{'ok' if dd <= tol else 'DISAGREE'}")
        ok = ok and dd <= tol

    rot = int(cfg["head_dim"] * cfg["partial_rotary_factor"])
    print("-- the partial rope table")
    check_rope_table_against_reference(chk, _checkpoint_text_config(cfg),
                                      cfg["rope_theta"], rot)
    print("-- the gated attention block")
    check_gated_attention_against_reference(chk)
    print("-- the GatedDeltaNet block")
    check_gated_delta_net_against_reference(chk)

    if headers is not None:
        print("-- the FP8 block dequantization every weight above came through")
        try:
            check_fp8_dequant_against_vllm(chk, headers, [
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                "model.language_model.layers.0.linear_attn.out_proj.weight",
                "model.language_model.layers.3.self_attn.q_proj.weight",
                "model.language_model.layers.3.self_attn.o_proj.weight",
                "model.language_model.layers.3.mlp.down_proj.weight",
            ])
        except ImportError as e:
            print(f"  !! vLLM not importable ({e}); the FP8 dequantization's "
                  f"scale indexing is unchecked, and every number in this "
                  f"capture rests on it")

    if not ok:
        raise SystemExit("the transcription disagrees with the reference; "
                         "fix it before capturing anything")
    return True


def _checkpoint_text_config(cfg):
    """The checkpoint's own text config as a `Qwen3_5TextConfig`.

    Used only so the rope check runs against *this model's* theta and partial
    factor rather than a synthetic pair — the whole point of that check is that
    `rope_theta` lives under `rope_parameters` and the exponent is normalized by
    `int(head_dim * partial_rotary_factor)`.
    """
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(**{k: v for k, v in cfg.items()
                                if k not in ("rope_theta", "partial_rotary_factor")})


def capture_linear(headers, cfg, x, dump):
    """GatedDeltaNet, stage by stage."""
    p = "model.language_model.layers.0.linear_attn."
    nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    dk, dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    key_dim, val_dim = nk * dk, nv * dv
    ksz = cfg["linear_conv_kernel_dim"]
    T = x.shape[0]

    w_qkv = load_f32(headers, p + "in_proj_qkv.weight")
    w_z = load_f32(headers, p + "in_proj_z.weight")
    w_a = load_f32(headers, p + "in_proj_a.weight")
    w_b = load_f32(headers, p + "in_proj_b.weight")
    conv_w = load_f32(headers, p + "conv1d.weight").squeeze(1)   # [10240, 4]
    a_log = load_f32(headers, p + "A_log")
    dt_bias = load_f32(headers, p + "dt_bias")
    norm_w = load_f32(headers, p + "norm.weight")
    w_out = load_f32(headers, p + "out_proj.weight")

    dump("linear.w_conv", conv_w)
    dump("linear.A_log", a_log)
    dump("linear.dt_bias", dt_bias)
    dump("linear.norm_w", norm_w)

    # Rows straddling both split boundaries, so a test can confirm where q ends
    # and k begins rather than inheriting this capture's own split.
    bnd = [0, key_dim - 1, key_dim, 2 * key_dim - 1, 2 * key_dim,
           2 * key_dim + val_dim - 1]
    dump("linear.qkv_boundary_rows", torch.tensor(bnd, dtype=torch.float32))
    dump("linear.qkv_boundary_w", w_qkv[bnd])

    # The arithmetic lives in `gated_delta_net_stages`, which
    # `check_gated_delta_net_against_reference` runs against
    # `Qwen3_5GatedDeltaNet` itself. So the stages dumped here are the stages
    # that were checked, rather than a second spelling of them.
    s = gated_delta_net_stages(x, {
        "in_proj_qkv": w_qkv, "in_proj_z": w_z, "in_proj_a": w_a,
        "in_proj_b": w_b, "conv1d": conv_w, "A_log": a_log,
        "dt_bias": dt_bias, "norm": norm_w, "out_proj": w_out,
    }, {"nk": nk, "nv": nv, "dk": dk, "dv": dv, "ksz": ksz,
        "eps": cfg["rms_norm_eps"]})
    for name in ("qkv_pre_conv", "z", "a", "b", "qkv_post_conv", "beta", "g",
                 "core_attn_out", "final_state", "after_gated_norm", "output"):
        dump("linear." + name, s[name])


def capture_full(headers, cfg, x, dump, pos_offset=0, tag="full"):
    """The gated attention layer, stage by stage.

    `pos_offset` shifts the rope positions. At positions 0..T the low-frequency
    dims barely move, so a bug in the tail of the frequency table is invisible —
    and that bug is exactly the one that shows up as long-context retrieval
    degrading while everything nearby stays correct. Capturing a second set at a
    six-figure position gives the tail some signal.
    """
    p = "model.language_model.layers.3.self_attn."
    dump = lambda n, t, _d=dump: _d(n.replace("full.", tag + "."), t)  # noqa: E731
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg["head_dim"]
    T = x.shape[0]

    w_q = load_f32(headers, p + "q_proj.weight")
    w_k = load_f32(headers, p + "k_proj.weight")
    w_v = load_f32(headers, p + "v_proj.weight")
    w_o = load_f32(headers, p + "o_proj.weight")
    qn = load_f32(headers, p + "q_norm.weight")
    kn = load_f32(headers, p + "k_norm.weight")
    dump("full.q_norm_w", qn)
    dump("full.k_norm_w", kn)

    # The layout trap: view to [T, heads, 2*head_dim] and split the LAST dim,
    # so q and the gate interleave per head. [all q | all gate] also "works".
    # A layout test needs the weight rows, not just the split result. Under the
    # interleaved reading, q[t, h, d] uses row h*2*hd + d and gate[t, h, d] uses
    # row h*2*hd + hd + d; under [all q | all gate] it would be h*hd + d and
    # nh*hd + h*hd + d. Dumping these specific rows lets a test distinguish them
    # instead of trusting whichever split the capture happened to make.
    probe_rows = []
    for h in (0, 1, nh - 1):
        for d in (0, 1, hd - 1):
            probe_rows += [h * 2 * hd + d, h * 2 * hd + hd + d,
                           h * hd + d, nh * hd + h * hd + d]
    probe_rows = sorted(set(probe_rows))
    dump("full.q_proj_probe_rows", torch.tensor(probe_rows, dtype=torch.float32))
    dump("full.q_proj_probe_w", w_q[probe_rows])

    # partial RoPE: only the first int(head_dim * factor) dims rotate, and the
    # frequency table is normalized by THAT width, not by head_dim.
    rot = int(hd * cfg["partial_rotary_factor"])
    pos = torch.arange(T, dtype=torch.float32) + pos_offset
    cos, sin = rope_tables(cfg["rope_theta"], rot, pos)
    dump("full.rope_cos", cos)
    dump("full.rope_sin", sin)

    # As with the linear block: the arithmetic is `gated_attention_stages`, which
    # `check_gated_attention_against_reference` runs against `Qwen3_5Attention`
    # itself, so what is dumped here is what was checked.
    s = gated_attention_stages(x, {
        "q_proj": w_q, "k_proj": w_k, "v_proj": w_v, "o_proj": w_o,
        "q_norm": qn, "k_norm": kn,
    }, {"nh": nh, "nkv": nkv, "hd": hd, "eps": cfg["rms_norm_eps"]}, cos, sin)
    for name in ("q_pre_norm", "gate", "q_post_norm", "k_post_norm",
                 "q_post_rope", "k_post_rope", "attn_out_pre_gate",
                 "attn_out_post_gate", "output"):
        dump("full." + name, s[name])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--tokens", type=int, default=12)
    ap.add_argument("--real-input", metavar="DIR",
                    help="use activations dumped by qwen35_real_prefix.py as the "
                         "block inputs instead of synthetic noise. Strongly "
                         "preferred: random input runs the block far below "
                         "rms_norm_eps, where the gated norm does nothing and "
                         "the capture cannot discriminate its formulation.")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model_dir, "config.json")))["text_config"]
    # rope_theta and partial_rotary_factor live under `rope_parameters`, not at
    # the text_config top level. Reading them from the wrong place does not
    # fail — it silently substitutes a default frequency base.
    rp = cfg.get("rope_parameters", {})
    cfg["rope_theta"] = rp.get("rope_theta", cfg.get("rope_theta"))
    cfg["partial_rotary_factor"] = rp.get(
        "partial_rotary_factor", cfg.get("partial_rotary_factor", 1.0))
    assert cfg["rope_theta"] is not None, "no rope_theta anywhere in the config"
    headers = read_index(args.model_dir)
    verified = cross_check_against_transformers(cfg, headers)
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = {"config": {k: cfg[k] for k in (
        "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
        "rms_norm_eps", "partial_rotary_factor",
        "linear_num_key_heads", "linear_num_value_heads",
        "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim",
    )}, "arrays": {}}
    manifest["config"]["rope_theta"] = cfg["rope_theta"]
    manifest["config"]["tokens"] = args.tokens

    def dump(name, t):
        t = t.detach().contiguous().float()
        with open(os.path.join(args.out_dir, name + ".f32"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        manifest["arrays"][name] = list(t.shape)
        flat = t.reshape(-1)
        print(f"  {name:<28} {list(t.shape)}  "
              f"[{flat.min():+.5f}, {flat.max():+.5f}]  "
              f"nonfinite={int((~flat.isfinite()).sum())}")

    # The input scale is not cosmetic. Both blocks receive
    # `input_layernorm(hidden_states)`, whose per-element distribution is
    # `unit-RMS noise * layernorm weight`. An earlier version of this script used
    # randn * 0.02, which put mean(core_attn_out^2) around 6e-12 — five orders
    # below rms_norm_eps of 1e-6. The eps then dominated every RMS denominator,
    # so the gated norm degenerated to a constant 1/sqrt(eps) scale and the
    # capture could not tell "normalize then gate" from "gate then normalize".
    # A capture in a regime where the operation under test does nothing blesses
    # anything. So build the input the way the model actually produces it.
    torch.manual_seed(20260822)
    ln = load_f32(headers, "model.language_model.layers.0.input_layernorm.weight")
    dump("input_layernorm_weight", ln)

    def real(layer):
        meta = json.load(open(os.path.join(args.real_input, "real_prefix.json")))
        name = f"real_input.layer{layer}"
        shape = meta["arrays"][name]
        raw = open(os.path.join(args.real_input, name + ".f32"), "rb").read()
        t = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape)
        return t[: args.tokens].clone()

    if args.real_input:
        x_lin, x_full = real(0), real(3)
        manifest["input_source"] = "real activations from qwen35_real_prefix.py"
    else:
        x_lin = torch.randn(args.tokens, cfg["hidden_size"], dtype=torch.float32) * ln
        x_full = x_lin
        manifest["input_source"] = "synthetic noise * input_layernorm weight"
    dump("input", x_lin)
    dump("input_full", x_full)

    print("== GatedDeltaNet (layer 0)")
    capture_linear(headers, cfg, x_lin, dump)
    print("== gated attention (layer 3), positions 0..T")
    capture_full(headers, cfg, x_full, dump)
    print("== gated attention (layer 3), positions 130000..")
    capture_full(headers, cfg, x_full, dump, pos_offset=130000, tag="full_far")

    manifest["verified_against_transformers"] = verified
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {len(manifest['arrays'])} arrays to {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
