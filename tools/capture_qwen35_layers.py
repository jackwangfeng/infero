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


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def rms_norm(x, w, eps=1e-6):
    v = x.float().pow(2).mean(-1, keepdim=True)
    return w * (x.float() * torch.rsqrt(v + eps))


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


def cross_check_against_transformers(cfg):
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

    if not ok:
        raise SystemExit("the transcription disagrees with the reference; "
                         "fix it before capturing anything")
    return True


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

    qkv = x @ w_qkv.T                                     # [T, 10240]
    z = x @ w_z.T                                         # [T, 6144]
    a = x @ w_a.T                                         # [T, 48]
    b = x @ w_b.T
    dump("linear.qkv_pre_conv", qkv)
    dump("linear.z", z)
    dump("linear.a", a)
    dump("linear.b", b)

    # Depthwise causal conv: pad kernel-1 on the left, no bias, then silu.
    cd = qkv.shape[-1]
    conv_in = qkv.T.unsqueeze(0)                          # [1, C, T]
    conv_out = F.conv1d(conv_in, conv_w.unsqueeze(1), None, padding=ksz - 1, groups=cd)
    conv_out = conv_out[:, :, :T]
    qkv_c = F.silu(conv_out).squeeze(0).T                 # [T, C]
    dump("linear.qkv_post_conv", qkv_c)

    q, k, v = torch.split(qkv_c, [key_dim, key_dim, val_dim], dim=-1)
    q = q.reshape(T, nk, dk)
    k = k.reshape(T, nk, dk)
    v = v.reshape(T, nv, dv)

    beta = torch.sigmoid(b)
    g = -a_log.exp() * F.softplus(a + dt_bias)
    dump("linear.beta", beta)
    dump("linear.g", g)

    rep = nv // nk
    q = q.repeat_interleave(rep, dim=1)
    k = k.repeat_interleave(rep, dim=1)

    core, state = recurrent_gated_delta(q, k, v, g, beta)
    dump("linear.core_attn_out", core)
    dump("linear.final_state", state)

    normed = rms_norm(core.reshape(-1, dv), norm_w, cfg["rms_norm_eps"])
    gated = normed * F.silu(z.reshape(-1, dv).float())
    dump("linear.after_gated_norm", gated)

    out = gated.reshape(T, val_dim) @ w_out.T
    dump("linear.output", out)


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

    qg = (x @ w_q.T).reshape(T, nh, 2 * hd)
    q, gate = qg[..., :hd], qg[..., hd:]
    dump("full.q_pre_norm", q.contiguous())
    dump("full.gate", gate.reshape(T, nh * hd).contiguous())

    k = (x @ w_k.T).reshape(T, nkv, hd)
    v = (x @ w_v.T).reshape(T, nkv, hd)
    q = rms_norm(q, qn, cfg["rms_norm_eps"])
    k = rms_norm(k, kn, cfg["rms_norm_eps"])
    dump("full.q_post_norm", q)
    dump("full.k_post_norm", k)

    # partial RoPE: only the first int(head_dim * factor) dims rotate, and the
    # frequency table is normalized by THAT width, not by head_dim.
    rot = int(hd * cfg["partial_rotary_factor"])
    theta = cfg["rope_theta"]
    inv = 1.0 / (theta ** (torch.arange(0, rot, 2, dtype=torch.float32) / rot))
    pos = torch.arange(T, dtype=torch.float32) + pos_offset
    freqs = pos[:, None] * inv[None, :]                  # [T, rot/2]
    emb = torch.cat([freqs, freqs], dim=-1)              # [T, rot], rotate_half layout
    cos, sin = emb.cos(), emb.sin()
    dump("full.rope_cos", cos)
    dump("full.rope_sin", sin)

    def rope(t):
        r, keep = t[..., :rot], t[..., rot:]
        h = rot // 2
        rotated = torch.cat([-r[..., h:], r[..., :h]], dim=-1)
        return torch.cat([r * cos[:, None, :] + rotated * sin[:, None, :], keep], dim=-1)

    q, k = rope(q), rope(k)
    dump("full.q_post_rope", q)
    dump("full.k_post_rope", k)

    # Causal attention with GQA.
    grp = nh // nkv
    kk = k.repeat_interleave(grp, dim=1)
    vv = v.repeat_interleave(grp, dim=1)
    scores = torch.einsum("thd,shd->hts", q, kk) * (hd ** -0.5)
    mask = torch.triu(torch.full((T, T), float("-inf")), 1)
    probs = torch.softmax(scores + mask, dim=-1)
    ctx = torch.einsum("hts,shd->thd", probs, vv).reshape(T, nh * hd)
    dump("full.attn_out_pre_gate", ctx)

    ctx = ctx * torch.sigmoid(gate.reshape(T, nh * hd))
    dump("full.attn_out_post_gate", ctx)
    dump("full.output", ctx @ w_o.T)


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
    verified = cross_check_against_transformers(cfg)
    os.makedirs(args.out_dir, exist_ok=True)
    headers = read_index(args.model_dir)

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
