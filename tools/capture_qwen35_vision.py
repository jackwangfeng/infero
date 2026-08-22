#!/usr/bin/env python3
"""Capture the Qwen3.5 vision tower's stage-by-stage output on a real checkpoint.

Same reason as `capture_qwen35_layers.py`: local self-consistency is not
evidence. The bf16-as-f16 embedding bug passed nine component-level A/Bs because
every stage did its job on faithfully-processed nonsense, and the vision tower
is a richer source of that failure mode than the text side. Which of the 1536
numbers in a patch is `(c, t, y, x)`; whether `qkv` is `[all q | all k | all v]`
or interleaved per head; whether the merger normalizes before or after it
shuffles 2x2 blocks; whether vision RoPE splits h/w in blocks or interleaves
them the way the text side's mRoPE does — every one of those has a second
reading that runs to completion and produces a fluent caption of the wrong
image.

So this runs the reference implementation's *own module classes* on the real
BF16 vision weights, hooks the endpoints of every stage, and dumps what came
out. Where a stage's interior cannot be hooked (the q/k/v split and the rotary
application live inside `Qwen3_5VisionAttention.forward`), the capture dumps the
hooked endpoints on both sides and `cross_check_against_transformers` requires
that this file's transcription of the interior reproduces them. A Rust test can
then re-derive the interior the same way and be answering to the reference
rather than to me.

Everything runs in f32 with f32 weights (the vision tower is entirely BF16 in
the checkpoint, so f32 holds it exactly). That is deliberate: the point of this
capture is to pin layout, and bf16 rounding noise only makes the tolerances
looser and the layout conclusions weaker.

Writes raw little-endian f32 arrays plus a manifest.json giving each array's
shape, so a Rust test can read them with no dependencies.

    /home/jeff/vllm312/bin/python capture_qwen35_vision.py <model-dir> <out-dir>
"""

import argparse
import json
import math
import os
import struct
import sys

import torch
import torch.nn.functional as F

# ------------------------------------------------------------------ checkpoint


def read_index(model_dir):
    """Map tensor name -> (shard path, header end, safetensors metadata)."""
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
    path, base, meta = headers[name]
    start, end = meta["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(base + start)
        buf = fh.read(end - start)
    dt = meta["dtype"]
    torch_dt = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "F8_E4M3": torch.float8_e4m3fn,
    }.get(dt)
    if torch_dt is None:
        raise SystemExit(f"{name}: unhandled dtype {dt}")
    return torch.frombuffer(bytearray(buf), dtype=torch_dt).reshape(meta["shape"])


def load_f32(headers, name, block=128):
    """A tensor as f32, dequantizing FP8 against its 128x128 block scale grid.

    The vision tower is BF16 throughout in this checkpoint — `quantization_config
    .modules_to_not_convert` lists every `visual.*` linear — so the FP8 branch is
    never taken here. It stays because "the vision tower happens to be BF16 in
    the checkpoint I looked at" is not something to hard-code: a later export
    that quantizes the tower would otherwise load garbage silently.
    """
    _, _, meta = headers[name]
    q = load_raw(headers, name)
    if meta["dtype"] != "F8_E4M3":
        return q.float()
    scales = load_f32(headers, name + "_scale_inv")
    rows, cols = q.shape
    gr, gc = scales.shape
    assert (gr, gc) == (-(-rows // block), -(-cols // block)), (
        f"{name}: quants {rows}x{cols} imply a {-(-rows // block)}x"
        f"{-(-cols // block)} scale grid at block {block}, got {gr}x{gc}"
    )
    full = scales.repeat_interleave(block, 0).repeat_interleave(block, 1)
    return q.float() * full[:rows, :cols]


# ------------------------------------------------------- the transcribed parts
#
# These are the pieces a CUDA port has to get right and that no forward hook can
# observe from outside. Each one is checked against the reference below.


def layer_norm(x, w, b, eps=1e-6):
    """LayerNorm, not RMSNorm: the mean is subtracted and there is a bias.

    Every norm in the vision tower is this. Every norm in the text tower is
    RMSNorm. Swapping them runs.
    """
    mu = x.mean(-1, keepdim=True)
    var = (x - mu).pow(2).mean(-1, keepdim=True)
    return (x - mu) * torch.rsqrt(var + eps) * w + b


def gelu_tanh(x):
    """`gelu_pytorch_tanh` — what `vision_config.hidden_act` names, used by the
    27 block MLPs."""
    return x * 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def gelu_erf(x):
    """Exact GELU — what `nn.GELU()` is, and what the *merger* uses. The tower
    has two different GELUs in it and the config names only one of them."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def vision_rope_table(position_ids, head_dim, theta=10000.0):
    """Vision RoPE: cos/sin of shape [N, head_dim] for `position_ids` [N, 2].

    Three separate traps live in these four lines.

    1. The frequency table has `head_dim // 2` = 36 slots, and the exponent is
       normalized by 36, not by head_dim. Same shape of error as the text side's
       partial rope: using the leading 18 frequencies of a 72-wide schedule runs
       and quietly changes every angle.
    2. `theta` is 10000 here. The text side is 1e7. Nothing complains.
    3. The axis layout is *blocked*, not interleaved: dims [0, 18) take the h
       position, dims [18, 36) take w, and `cat((emb, emb))` copies both blocks
       into [36, 72). The text side's mRoPE for the same checkpoint interleaves
       its three axes by `index % 3`. Two schemes, one model; using either one
       in the other's place runs to completion.
    """
    dim = head_dim // 2
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    # [N, 2, dim/2] -> [N, dim]: h against every frequency, then w.
    freqs = (position_ids.float().unsqueeze(-1) * inv).flatten(1)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_vision_rope(x, cos, sin):
    """`rotate_half` pairing: dim `i` pairs with dim `i + head_dim/2`.

    `x` is [N, heads, head_dim]; cos/sin are [N, head_dim]. The whole head
    rotates — there is no partial-rotary factor on the vision side.
    """
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos.unsqueeze(-2) + rot * sin.unsqueeze(-2)


def split_qkv(qkv, heads):
    """`qkv` [N, 3*dim] -> q, k, v each [N, heads, head_dim].

    The reference spelling is
    `qkv(h).reshape(seq, 3, heads, -1).permute(1, 0, 2, 3).unbind(0)`.
    The 3 sits *before* the head axis, so the output is
    `[all q | all k | all v]` in blocks of `dim`.

    This is the exact opposite of the text side, where `q_proj`'s output is
    `view(..., heads, 2 * head_dim)` and the query and its gate interleave
    within each head. Same model, two conventions, and both readings of either
    tensor produce three correctly-shaped tensors.
    """
    n = qkv.shape[0]
    return qkv.reshape(n, 3, heads, -1).permute(1, 0, 2, 3).unbind(0)


def segment_attention(q, k, v, cu_seqlens, scaling):
    """Non-causal attention, restricted to each `cu_seqlens` segment.

    q/k/v are [N, heads, head_dim]. Two things to get right:

    - It is *not* causal. `Qwen3_5VisionAttention.is_causal = False`. A causal
      mask here runs and blinds every patch to everything below and right of it.
    - The segments are per *frame*, not per image: `get_vision_cu_seqlens` uses
      `repeat_interleave(h * w, t)`, so a t-frame video is t independent
      attention blocks. Letting attention span frames runs and mixes them.
    """
    out = torch.zeros_like(q)
    for a, b in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        qs = q[a:b].transpose(0, 1)  # [heads, L, hd]
        ks = k[a:b].transpose(0, 1)
        vs = v[a:b].transpose(0, 1)
        scores = (qs @ ks.transpose(-1, -2)) * scaling
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(qs.dtype)
        out[a:b] = (probs @ vs).transpose(0, 1)
    return out


def vision_cu_seqlens(grid_thw):
    """Attention segment boundaries: one segment per frame of each entry."""
    lens = []
    for t, h, w in grid_thw.tolist():
        lens += [h * w] * t
    cu = [0]
    for length in lens:
        cu.append(cu[-1] + length)
    return torch.tensor(cu, dtype=torch.int32)


def vision_position_ids(grid_thw, merge):
    """(h, w) index per patch, in spatial-merge-block order.

    Patch `p` of a frame decodes as
    `p = ((block_row * blocks_w + block_col) * merge + in_row) * merge + in_col`
    with `row = block_row * merge + in_row`, `col = block_col * merge + in_col`.
    Raster order — `p = row * w + col` — is the obvious alternative, it runs, and
    it silently transposes the position field *and* makes the merger average
    four patches that are a row apart instead of a 2x2 block.
    """
    out = []
    for t, h, w in grid_thw.tolist():
        ids = torch.empty(h * w, 2, dtype=torch.long)
        p = 0
        for br in range(h // merge):
            for bc in range(w // merge):
                for ir in range(merge):
                    for ic in range(merge):
                        ids[p, 0] = br * merge + ir
                        ids[p, 1] = bc * merge + ic
                        p += 1
        out.append(ids.repeat(t, 1))
    return torch.cat(out, 0)


def pos_embed_taps(grid_thw, side, merge, align_corners=True):
    """Bilinear resample of the learned `side x side` position grid.

    Returns (indices [N, 4], weights [N, 4]) into the flattened `pos_embed`
    table. `align_corners=True` is what `Qwen3_5VisionModel.__init__` sets, so
    the source coordinate is `index * (side - 1) / (size - 1)`. The library
    helper's own *default* is `align_corners=False`, which would give
    `(index + 0.5) * side / size - 0.5` — a different, plausible, silently
    running resample.
    """
    idx_out, w_out = [], []
    for t, h, w in grid_thw.tolist():
        for _ in range(t):
            for p in range(h * w):
                in_col = p % merge
                in_row = (p // merge) % merge
                block_col = (p // (merge * merge)) % (w // merge)
                block_row = p // (merge * merge * (w // merge))
                row = block_row * merge + in_row
                col = block_col * merge + in_col
                taps = []
                weights = []
                for index, size in ((row, h), (col, w)):
                    if align_corners:
                        src = index * (side - 1) / max(size - 1, 1)
                    else:
                        src = (index + 0.5) * side / size - 0.5
                    fl = math.floor(src)
                    t0 = min(max(fl, 0), side - 1)
                    t1 = min(max(fl + 1, 0), side - 1)
                    d0 = abs(src - fl)
                    taps.append((t0, t1))
                    weights.append((max(1.0 - d0, 0.0), max(1.0 - abs(src - fl - 1), 0.0)))
                (h0, h1), (w0, w1) = taps
                (hw0, hw1), (ww0, ww1) = weights
                idx_out.append([h0 * side + w0, h0 * side + w1, h1 * side + w0, h1 * side + w1])
                w_out.append([hw0 * ww0, hw0 * ww1, hw1 * ww0, hw1 * ww1])
    return torch.tensor(idx_out, dtype=torch.long), torch.tensor(w_out, dtype=torch.float32)


def patch_slot(c, t, y, x, temporal_patch, patch):
    """Where component `(c, t, y, x)` of a patch sits in its 1536-wide row.

    The processor's `patchify` permutes to
    `[batch, gh/m, gw/m, m, m, channel, patch, patch]`, unsqueezes a temporal
    axis between `channel` and the two `patch` axes, and flattens the last four.
    So the stride order is c > t > y > x. Any other ordering of those four —
    `(t, c, y, x)`, or `(c, t, x, y)` — feeds the Conv3d a transposed patch and
    the tower runs and describes a scrambled image.
    """
    return ((c * temporal_patch + t) * patch + y) * patch + x


def patchify(image, patch, merge, temporal_patch):
    """`[C, H, W]` -> `[gh*gw, C*T*P*P]`, in spatial-merge-block order.

    A still image's `temporal_patch` slots are the *same* frame repeated, not
    zeros and not two consecutive frames: `patchify` does
    `unsqueeze(6).expand(..., temporal_patch_size, ...)`. So the Conv3d's two
    temporal taps both see the same pixels and effectively act as their sum.
    Feeding one tap and zeroing the other runs, and halves the patch embedding.
    """
    c_n, h_px, w_px = image.shape
    gh, gw = h_px // patch, w_px // patch
    out = torch.zeros(gh * gw, c_n * temporal_patch * patch * patch)
    for p in range(gh * gw):
        in_col = p % merge
        in_row = (p // merge) % merge
        block_col = (p // (merge * merge)) % (gw // merge)
        block_row = p // (merge * merge * (gw // merge))
        row, col = block_row * merge + in_row, block_col * merge + in_col
        tile = image[:, row * patch:(row + 1) * patch, col * patch:(col + 1) * patch]
        for c in range(c_n):
            for t in range(temporal_patch):
                for y in range(patch):
                    for x in range(patch):
                        out[p, patch_slot(c, t, y, x, temporal_patch, patch)] = tile[c, y, x]
    return out, gh, gw


def smart_resize(height, width, factor, min_pixels, max_pixels):
    """The dynamic-resolution rule. `factor` is `patch_size * merge_size` = 32,
    *not* `patch_size`: the grid must be even in both axes so the merger's 2x2
    blocks are whole."""
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


# ------------------------------------------------------------------ the tower


def build_tower(model_dir, headers, dtype=torch.float32):
    """The reference `Qwen3_5VisionModel`, with the checkpoint's real weights."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    raw = json.load(open(os.path.join(model_dir, "config.json")))
    cfg = Qwen3_5Config(**{k: v for k, v in raw.items() if k != "quantization_config"})
    vcfg = cfg.vision_config
    vcfg._attn_implementation = "eager"
    model = m.Qwen3_5VisionModel(vcfg).to(dtype).eval()

    prefix = "model.visual."
    loaded = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            key = prefix + name
            if key not in headers:
                raise SystemExit(f"no checkpoint tensor for {key}")
            w = load_f32(headers, key)
            assert w.shape == param.shape, f"{key}: {list(w.shape)} vs {list(param.shape)}"
            param.copy_(w.to(dtype))
            loaded += 1
    print(f"  loaded {loaded} vision tensors into the reference module")
    return cfg, vcfg, model


def procedural_image(height, width, seed):
    """A deterministic uint8 RGB image with real spatial structure.

    Structure matters. Flat or uniform-noise input makes neighbouring patches
    either identical or independent, and both regimes hide layout errors: with
    identical patches every permutation of the patch order gives the same
    answer, and with independent noise the position embedding contributes
    nothing that the rest of the tower can be seen to use. Sinusoids at several
    scales plus a little noise give patches that differ from their neighbours in
    a way that depends on *where* they are.
    """
    g = torch.Generator().manual_seed(seed)
    ys = torch.linspace(0, 1, height).view(-1, 1)
    xs = torch.linspace(0, 1, width).view(1, -1)
    chans = []
    for c, (fy, fx, ph) in enumerate([(3.0, 5.0, 0.0), (7.0, 2.0, 1.1), (11.0, 13.0, 2.3)]):
        base = torch.sin(2 * math.pi * (fy * ys + ph)) * torch.cos(2 * math.pi * fx * xs)
        base = base + 0.5 * torch.sin(2 * math.pi * (17.0 * ys + 19.0 * xs) + c)
        base = base + 0.15 * torch.randn(height, width, generator=g)
        chans.append(base)
    img = torch.stack(chans, 0)
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).round().clamp(0, 255).to(torch.uint8)


def preprocess(ip, image_hw, seed, longest_edge):
    """Run the reference image processor and return (pixel_values, grid_thw)."""
    h, w = image_hw
    img = procedural_image(h, w, seed)
    out = ip.preprocess(
        images=[img.permute(1, 2, 0).numpy()],
        size={"shortest_edge": 256, "longest_edge": longest_edge},
        return_tensors="pt",
    )
    return out["pixel_values"].float(), out["image_grid_thw"]


# -------------------------------------------------------------- cross-checking


def check_module_constants(chk, vcfg, model):
    """What the reference's vision modules actually are, read off the instance.

    Every constant below is hard-coded somewhere — in this file's `layer_norm`
    default, in the manifest, or in `VisionDims::QWEN35_27B` on the Rust side —
    and every one of them is silent if wrong:

    * The epsilon. `Qwen3_5VisionBlock` writes `nn.LayerNorm(hidden, eps=1e-6)`
      as a literal; there is no `layer_norm_eps` in `vision_config` to read, so
      a port that reaches for `nn.LayerNorm`'s own default gets 1e-5 and a 0.4%
      error wherever a row's variance is small. That is a rounding-looking bug.
    * The norm *class*. RMSNorm is what the text tower uses, an RMSNorm-shaped
      transcription of a LayerNorm runs, and the difference is a mean and a bias.
    * The merger norm's width. 1152 means `use_postshuffle_norm=False` — the norm
      runs per patch, before the 2x2 grouping. 4608 would mean the other order.
      This is the checkpoint's own evidence, in a shape.
    * Which GELU is where: `nn.GELU()` (exact) in the merger, the tanh form in
      the 27 blocks. The config names only the latter.
    * The attention scale, and that it is not causal.
    * A bias on every projection. The text tower has none anywhere; loading the
      vision tower with the text loader drops 12 tensors per block.

    Returns the epsilon the tower actually uses, so nothing downstream needs to
    assume it.
    """
    hidden = vcfg.hidden_size
    heads = vcfg.num_heads
    head_dim = hidden // heads
    wide = hidden * vcfg.spatial_merge_size ** 2

    epsilons = set()
    for tag, norm, width in (("blocks.0.norm1", model.blocks[0].norm1, hidden),
                             ("blocks.0.norm2", model.blocks[0].norm2, hidden),
                             ("merger.norm", model.merger.norm, hidden)):
        if not isinstance(norm, torch.nn.LayerNorm):
            raise SystemExit(f"{tag} is a {type(norm).__name__}, not an "
                             f"nn.LayerNorm; this capture and the Rust "
                             f"reference both centre and add a bias")
        if norm.bias is None:
            raise SystemExit(f"{tag} has no bias")
        if tuple(norm.normalized_shape) != (width,):
            raise SystemExit(
                f"{tag} normalizes over {tuple(norm.normalized_shape)}, not "
                f"({width},). For merger.norm that is the difference between "
                f"use_postshuffle_norm False and True — the norm running per "
                f"patch or over the grouped 2x2 block.")
        epsilons.add(norm.eps)
    if len(epsilons) != 1:
        raise SystemExit(f"the tower's LayerNorms disagree on eps: {epsilons}")
    eps = epsilons.pop()
    print(f"  every vision LayerNorm: nn.LayerNorm with a bias, eps={eps:g}, "
          f"merger.norm over {hidden} (per patch, pre-shuffle)")

    # The last norm of the 27 blocks too, in case the depth-0 block is special.
    last = model.blocks[-1]
    if not isinstance(last.norm1, torch.nn.LayerNorm) or last.norm1.eps != eps:
        raise SystemExit("the last block's norm1 differs from the first's")

    if not isinstance(model.merger.act_fn, torch.nn.GELU):
        raise SystemExit(f"merger.act_fn is {type(model.merger.act_fn).__name__}, "
                         f"not nn.GELU")
    if getattr(model.merger.act_fn, "approximate", "none") != "none":
        raise SystemExit(f"merger.act_fn is nn.GELU(approximate="
                         f"{model.merger.act_fn.approximate!r}), i.e. the tanh "
                         f"form, not the exact one this capture assumes")
    if vcfg.hidden_act != "gelu_pytorch_tanh":
        raise SystemExit(f"vision_config.hidden_act is {vcfg.hidden_act!r}, not "
                         f"gelu_pytorch_tanh")

    attn = model.blocks[0].attn
    chk("vision attention scaling == head_dim**-0.5",
        abs(attn.scaling - head_dim ** -0.5), 0)
    if attn.is_causal:
        raise SystemExit("Qwen3_5VisionAttention.is_causal is True; this "
                         "capture's segment_attention applies no causal mask")

    biased = {
        "patch_embed.proj": model.patch_embed.proj,
        "blocks.0.attn.qkv": attn.qkv,
        "blocks.0.attn.proj": attn.proj,
        "blocks.0.mlp.linear_fc1": model.blocks[0].mlp.linear_fc1,
        "blocks.0.mlp.linear_fc2": model.blocks[0].mlp.linear_fc2,
        "merger.linear_fc1": model.merger.linear_fc1,
        "merger.linear_fc2": model.merger.linear_fc2,
    }
    missing = [n for n, mod in biased.items() if mod.bias is None]
    if missing:
        raise SystemExit(f"these vision projections have no bias: {missing}. "
                         f"The whole tower is supposed to be biased; a loader "
                         f"that drops them reads as fluent nonsense.")
    print(f"  every vision projection carries a bias ({len(biased)} checked)")

    if (model.merger.linear_fc1.in_features, model.merger.linear_fc1.out_features) \
            != (wide, wide):
        raise SystemExit(
            f"merger.linear_fc1 is {model.merger.linear_fc1.in_features} -> "
            f"{model.merger.linear_fc1.out_features}, expected {wide} -> {wide}")
    if model.merger.linear_fc2.out_features != vcfg.out_hidden_size:
        raise SystemExit(
            f"merger.linear_fc2 outputs {model.merger.linear_fc2.out_features}, "
            f"not out_hidden_size={vcfg.out_hidden_size}. The class default is "
            f"3584 (the 9B); a loader that falls back to it builds a merger "
            f"whose output does not fit the language model.")
    return eps


def cross_check_against_transformers(model_dir, vcfg, model, ip, hooked):
    """Require every transcription above to agree with the reference.

    Without this the capture is my reading of the reference, and my reading is
    the thing under test. Each check either calls the library's own function or
    reproduces a tensor the library itself produced, and every one of them has
    to pass before anything is written.
    """
    from transformers.activations import ACT2FN
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        smart_resize as ref_smart_resize,
    )
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.vision_utils import (
        get_vision_cu_seqlens,
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )

    torch.manual_seed(11)
    checks = []

    def chk(name, delta, tol):
        d = float(torch.as_tensor(delta).detach())
        checks.append((name, d, tol))
        print(f"  {name:<44} Δ={d:.3e}  tol={tol:.0e}  "
              f"{'ok' if d <= tol else 'DISAGREE'}")

    merge = vcfg.spatial_merge_size
    heads = vcfg.num_heads
    head_dim = vcfg.hidden_size // heads

    # 0. What the modules *are*, read off the instantiated reference rather than
    # assumed from the notes. Every one of these is a constant this file or the
    # Rust reference hard-codes, and every one of them is silent if wrong: an
    # eps of 1e-5 instead of 1e-6, an RMSNorm where a LayerNorm was expected, a
    # merger norm over 4608 instead of 1152, the tanh GELU in the merger, a
    # missing bias on a projection.
    eps = check_module_constants(chk, vcfg, model)

    # 1. LayerNorm against torch's — using the tower's *own* norm module, so
    # this is a check on the reference's configuration and not only on the
    # formula, and the tower's *own* first-block input, so the mean it subtracts
    # is a real one. A `randn` row has a mean of ~0 by construction, which makes
    # the centring look optional: the "biased but not centred" reading below
    # separates by 2% on noise and by 100% on a real activation.
    x = hooked["img"]["hidden_in"]
    ref_ln = model.blocks[0].norm1
    with torch.no_grad():
        ref_out = ref_ln(x)
    chk("layer_norm vs the tower's own nn.LayerNorm",
        (layer_norm(x, ref_ln.weight.detach(), ref_ln.bias.detach(), eps)
         - ref_out).abs().max(), 2e-5)
    print(f"  (the norm1 input's row means run "
          f"[{float(x.mean(-1).min()):+.3f}, {float(x.mean(-1).max()):+.3f}], so "
          f"the mean subtraction is doing something)")

    # 1b. And the readings that also run. An RMSNorm-shaped transcription of a
    # LayerNorm — no mean subtraction, no bias — is the text tower's habit
    # carried across, and it runs.
    with torch.no_grad():
        gw, gb = ref_ln.weight.detach(), ref_ln.bias.detach()
        ctr0 = x - x.mean(-1, keepdim=True)
        peak_ln = ref_out.abs().max()
        for name, wrong in (
            ("no mean subtraction and no bias (RMSNorm-shaped)",
             x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gw),
            ("centred but no bias",
             ctr0 * torch.rsqrt(ctr0.pow(2).mean(-1, keepdim=True) + eps) * gw),
            ("biased but not centred",
             x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gw + gb),
        ):
            sep = (wrong - ref_out).abs().max() / peak_ln
            print(f"  (the reading `{name}` is off by {float(sep):.3e} of peak)")
            if float(sep) < 1e-2:
                raise SystemExit(f"the reading `{name}` reproduces the reference "
                                 f"LayerNorm; this check does not discriminate")

    # 1c. Where the epsilon goes. At unit variance every placement agrees to
    # 5e-7, which is inside the tolerance above, so the check so far says
    # nothing about it. Drive the norm at a variance near eps instead.
    tiny = torch.randn(9, vcfg.hidden_size) * (eps ** 0.5)
    with torch.no_grad():
        ref_tiny = ref_ln(tiny)
        peak_tiny = ref_tiny.abs().max()
        chk("layer_norm at variance ~ eps",
            (layer_norm(tiny, gw, gb, eps) - ref_tiny).abs().max() / peak_tiny, 2e-5)
        ctr = tiny - tiny.mean(-1, keepdim=True)
        var = ctr.pow(2).mean(-1, keepdim=True)
        on_root = ctr / (var.sqrt() + eps) * gw + gb
        on_sum = ctr * torch.rsqrt(ctr.pow(2).sum(-1, keepdim=True) + eps) * gw + gb
        for name, wrong in (("eps added to the standard deviation", on_root),
                            ("eps on the sum of squares", on_sum)):
            sep = (wrong - ref_tiny).abs().max() / peak_tiny
            print(f"  (the reading `{name}` is off by {float(sep):.3e} of peak)")
            if float(sep) < 1e-2:
                raise SystemExit(f"`{name}` is indistinguishable even at variance "
                                 f"~ eps; this check is decorative")

    # 2/3. Both GELUs.
    y = torch.randn(4096) * 3
    chk("gelu_tanh vs ACT2FN[gelu_pytorch_tanh]",
        (gelu_tanh(y) - ACT2FN[vcfg.hidden_act](y)).abs().max(), 1e-5)
    chk("gelu_tanh vs the blocks' own act_fn",
        (gelu_tanh(y) - model.blocks[0].mlp.act_fn(y)).abs().max(), 1e-5)
    chk("gelu_erf vs nn.GELU", (gelu_erf(y) - torch.nn.GELU()(y)).abs().max(), 1e-5)
    chk("gelu_erf vs the merger's own act_fn",
        (gelu_erf(y) - model.merger.act_fn(y)).abs().max(), 1e-5)
    # And record that they are *not* the same function, so the doc's claim that
    # the tower uses two different GELUs is a claim with content.
    two_gelus = (gelu_tanh(y) - gelu_erf(y)).abs().max()
    print(f"  (the two GELUs differ by at most {float(two_gelus):.2e} over |x|<~12)")
    if float(two_gelus) < 1e-5:
        raise SystemExit("the two GELUs agree to f32 resolution here, so "
                         "'the merger uses the exact one' is not a claim this "
                         "capture can support")

    # 4. Vision rope table against the reference module.
    pids = torch.randint(0, 60, (23, 2))
    ref_freqs = model.rotary_pos_emb(pids)
    ref_emb = torch.cat((ref_freqs, ref_freqs), dim=-1)
    cos, sin = vision_rope_table(pids, head_dim)
    chk("vision_rope_table cos vs Qwen3_5VisionRotaryEmbedding",
        (cos - ref_emb.cos()).abs().max(), 1e-5)
    chk("vision_rope_table sin vs Qwen3_5VisionRotaryEmbedding",
        (sin - ref_emb.sin()).abs().max(), 1e-5)
    assert model.rotary_pos_emb.dim == head_dim // 2, (
        f"the reference builds its frequency table over {model.rotary_pos_emb.dim} "
        f"dims, not head_dim//2 = {head_dim // 2}")
    assert model.rotary_pos_emb.theta == 10000.0, (
        f"vision rope theta is {model.rotary_pos_emb.theta}, not 10000")

    # 5. rotate_half application against apply_rotary_pos_emb_vision.
    q = torch.randn(23, heads, head_dim)
    k = torch.randn(23, heads, head_dim)
    rq, rk = m.apply_rotary_pos_emb_vision(q, k, cos, sin)
    chk("apply_vision_rope q vs apply_rotary_pos_emb_vision",
        (apply_vision_rope(q, cos, sin) - rq).abs().max(), 1e-5)
    chk("apply_vision_rope k vs apply_rotary_pos_emb_vision",
        (apply_vision_rope(k, cos, sin) - rk).abs().max(), 1e-5)

    # 6. Geometry: position ids, cu_seqlens, interpolation taps.
    probe_grid = torch.tensor([[1, 6, 8], [2, 4, 10], [1, 2, 12]])
    chk("vision_position_ids vs get_vision_position_ids",
        (vision_position_ids(probe_grid, merge)
         - get_vision_position_ids(probe_grid, merge)).abs().max().float(), 0)
    chk("vision_cu_seqlens vs get_vision_cu_seqlens",
        (vision_cu_seqlens(probe_grid) - get_vision_cu_seqlens(probe_grid))
        .abs().max().float(), 0)
    side = model.num_grid_per_side
    ref_idx, ref_w = get_vision_interpolation_indices_and_weights(
        probe_grid, num_grid_per_side=side, mode=model.interpolation_mode,
        align_corners=model.interpolation_align_corners, spatial_merge_size=merge)
    mine_idx, mine_w = pos_embed_taps(probe_grid, side, merge,
                                      model.interpolation_align_corners)
    chk("pos_embed_taps indices vs vision_utils",
        (mine_idx - ref_idx).abs().max().float(), 0)
    # Indices must match exactly; the weights only to f32 rounding, because the
    # reference computes `index * (side - 1) / (size - 1)` in f32 tensors and
    # this file computes it in Python floats.
    chk("pos_embed_taps weights vs vision_utils", (mine_w - ref_w).abs().max(), 2e-5)

    # 6b. And the taps really do reproduce a bilinear resample — an oracle that
    # does not come from the same file as the taps.
    table = torch.randn(side * side, 3)
    for t_, h_, w_ in probe_grid.tolist():
        one = torch.tensor([[1, h_, w_]])
        idx, wt = pos_embed_taps(one, side, merge, model.interpolation_align_corners)
        got = (table[idx] * wt[:, :, None]).sum(1)
        ref = F.interpolate(table.view(1, side, side, 3).permute(0, 3, 1, 2),
                            size=(h_, w_), mode="bilinear", align_corners=True)
        ref = ref[0].permute(1, 2, 0).reshape(-1, 3)
        # `ref` is in raster order; reorder it into merge-block order.
        pos = vision_position_ids(one, merge)
        ref = ref[pos[:, 0] * w_ + pos[:, 1]]
        chk(f"pos_embed_taps == F.interpolate on {h_}x{w_}",
            (got - ref).abs().max(), 5e-5)

    # 6c. The patch layout, against the processor's own `patchify`. The probe
    # image is `arange`, so every pixel is a distinct number and the comparison
    # locates each one exactly rather than agreeing by luck.
    patch, tpatch = vcfg.patch_size, vcfg.temporal_patch_size
    gh, gw = 6, 8
    probe = torch.arange(3 * gh * patch * gw * patch, dtype=torch.float32).reshape(
        3, gh * patch, gw * patch)
    ref_px, r_gh, r_gw = ip.patchify(probe[None], patch_size=patch, merge_size=merge,
                                     temporal_patch_size=tpatch)
    assert (r_gh, r_gw) == (gh, gw)
    mine_px, _, _ = patchify(probe, patch, merge, tpatch)
    chk("patchify vs Qwen2VLImageProcessor.patchify",
        (mine_px - ref_px[0]).abs().max(), 0)
    # Raster patch order must not reproduce it.
    raster = torch.stack([
        probe[:, (p // gw) * patch:(p // gw + 1) * patch,
              (p % gw) * patch:(p % gw + 1) * patch].reshape(-1)
        for p in range(gh * gw)])
    raster = raster.repeat_interleave(1, 0)
    sep = (raster[:, 0] - ref_px[0, :, 0]).abs().max()
    print(f"  (raster patch order differs from block order by {float(sep):.3e})")
    if float(sep) == 0:
        raise SystemExit("raster and merge-block patch order agree on this grid; "
                         "pick a grid where they do not")

    # 7. smart_resize against the library's.
    worst = 0
    for h_, w_ in [(300, 400), (150, 500), (64, 64), (4000, 3000), (256, 256),
                   (33, 33), (1024, 1024), (200, 4000), (8000, 60)]:
        a = smart_resize(h_, w_, 32, 65536, 16777216)
        b = ref_smart_resize(h_, w_, factor=32, min_pixels=65536, max_pixels=16777216)
        worst = max(worst, abs(a[0] - b[0]), abs(a[1] - b[1]))
    chk("smart_resize vs qwen2_vl.smart_resize", float(worst), 0)

    # 8. The patch-embed weight really is `[out, c, t, y, x]` flattened in that
    # order. Evidence, not a reshape: drive the reference Conv3d with one-hot
    # inputs and read off which flat slot lands on which weight column.
    patch_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size ** 2
    eye = torch.eye(patch_dim)
    with torch.no_grad():
        resp = model.patch_embed(eye) - model.patch_embed.proj.bias
    w_flat = model.patch_embed.proj.weight.reshape(vcfg.hidden_size, patch_dim)
    chk("patch_embed one-hot response vs weight.view(out, c*t*y*x)",
        (resp - w_flat.T).abs().max(), 2e-5)

    # 9. The q/k/v split and the whole attention interior, checked against the
    # reference's own endpoints: `attn.qkv`'s output in, `attn.proj`'s input out.
    for tag, h in hooked.items():
        qkv = h["b0.qkv"]
        cos, sin = h["rope_cos"], h["rope_sin"]
        q, k, v = split_qkv(qkv, heads)
        q, k = apply_vision_rope(q, cos, sin), apply_vision_rope(k, cos, sin)
        cu = [int(c) for c in h["cu_seqlens"].tolist()]
        got = segment_attention(q, k, v, cu, head_dim ** -0.5)
        chk(f"[{tag}] split_qkv+rope+segment_attention vs attn.proj input",
            (got.reshape(qkv.shape[0], -1) - h["b0.attn_pre_proj"]).abs().max(), 2e-4)

        # And the same pipeline with the interior read the other plausible way
        # must *not* reproduce it, or the check above is not evidence about
        # layout — only that some arithmetic ran.
        qi = qkv.reshape(qkv.shape[0], heads, 3, head_dim)[:, :, 0]
        ki = qkv.reshape(qkv.shape[0], heads, 3, head_dim)[:, :, 1]
        vi = qkv.reshape(qkv.shape[0], heads, 3, head_dim)[:, :, 2]
        qi, ki = apply_vision_rope(qi, cos, sin), apply_vision_rope(ki, cos, sin)
        other = segment_attention(qi, ki, vi, cu, head_dim ** -0.5)
        sep = (other.reshape(qkv.shape[0], -1) - h["b0.attn_pre_proj"]).abs().max()
        print(f"  [{tag}] per-head-interleaved qkv reading is off by "
              f"{float(sep):.3e} (must be large)")
        if float(sep) < 1e-2:
            raise SystemExit(
                f"[{tag}] reading qkv as per-head-interleaved reproduced the "
                f"reference to {float(sep):.2e}; this capture cannot tell the "
                f"two layouts apart and is worthless as an oracle for them")

    # 10. The interleaved mRoPE axis map is read straight out of the reference
    # method by `mrope_axis_map`, so there is nothing to cross-check — but assert
    # the invariant that makes it meaningful: the counts the axis map produces
    # have to be the config's `mrope_section`.
    axis, section, half, _ = mrope_axis_map(model_dir)
    counts = [int((axis == a).sum()) for a in range(3)]
    chk("mrope axis counts vs config mrope_section",
        float(max(abs(a - b) for a, b in zip(counts, list(section)))), 0)
    assert half == sum(section), (
        f"{half} rotary frequencies but mrope_section sums to {sum(section)}")
    del Qwen3_5TextConfig

    ok = all(d <= t for _, d, t in checks)
    if not ok:
        bad = [n for n, d, t in checks if d > t]
        raise SystemExit("the transcription disagrees with the reference at: "
                         + ", ".join(bad) + "; fix it before capturing anything")
    return True


def mrope_axis_map(model_dir):
    """Which of T/H/W drives each of the 32 rotary frequencies, straight from
    `Qwen3_5TextRotaryEmbedding.apply_interleaved_mrope`.

    Feed it a `freqs` whose value along axis `a` is the constant `a`, and the
    tensor it returns *is* the axis assignment. No transcription involved.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    raw = json.load(open(os.path.join(model_dir, "config.json")))["text_config"]
    cfg = Qwen3_5TextConfig(**raw)
    rot = m.Qwen3_5TextRotaryEmbedding(cfg)
    half = rot.inv_freq.shape[0]
    marker = torch.arange(3, dtype=torch.float32).view(3, 1, 1, 1).expand(3, 1, 1, half).clone()
    axis = rot.apply_interleaved_mrope(marker, rot.mrope_section).reshape(-1)
    return axis, rot.mrope_section, half, cfg


def llm_position_ids(model_dir, cfg, input_ids, mm_token_type_ids, image_grid_thw):
    """3-D text-side positions for a spliced sequence, via the reference's own
    `Qwen3_5Model.get_rope_index`.

    The full model is 27B and will not be instantiated here, but `get_rope_index`
    only reaches `self.config` and `self.get_vision_position_ids`, so binding the
    unbound methods to a stub runs the library's code without the weights.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    class Stub:
        pass

    stub = Stub()
    stub.config = cfg
    stub.get_vision_position_ids = m.Qwen3_5Model.get_vision_position_ids.__get__(stub)
    pos, delta = m.Qwen3_5Model.get_rope_index.__get__(stub)(
        input_ids=input_ids, mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
    )
    return pos, delta


# ------------------------------------------------------------------- capturing


def run_tower(model, vcfg, pixel_values, grid_thw):
    """Run the reference tower, hooking the endpoint of every stage.

    Returns a dict of tensors. Everything in it came out of a reference module;
    nothing in it is computed here.
    """
    from transformers.vision_utils import (
        get_vision_attention_seqlens,
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )

    got = {}
    handles = []

    def out_hook(name):
        def fn(_m, _i, o):
            got[name] = (o[0] if isinstance(o, tuple) else o).detach().float().clone()
        return fn

    def in_hook(name, argno=0):
        def fn(_m, i, _o):
            got[name] = i[argno].detach().float().clone()
        return fn

    blk = model.blocks[0]
    handles += [
        model.patch_embed.register_forward_hook(out_hook("patch_embed_out")),
        blk.norm1.register_forward_hook(out_hook("b0.norm1_out")),
        blk.attn.qkv.register_forward_hook(out_hook("b0.qkv")),
        blk.attn.proj.register_forward_hook(in_hook("b0.attn_pre_proj")),
        blk.attn.register_forward_hook(out_hook("b0.attn_out")),
        blk.norm2.register_forward_hook(in_hook("b0.resid1")),
        blk.norm2.register_forward_hook(out_hook("b0.norm2_out")),
        blk.mlp.linear_fc1.register_forward_hook(out_hook("b0.fc1_out")),
        blk.mlp.linear_fc2.register_forward_hook(in_hook("b0.act_out")),
        blk.mlp.register_forward_hook(out_hook("b0.mlp_out")),
        blk.register_forward_hook(out_hook("b0.out")),
        model.merger.norm.register_forward_hook(out_hook("merger.norm_out")),
        model.merger.linear_fc1.register_forward_hook(out_hook("merger.fc1_out")),
        model.merger.linear_fc2.register_forward_hook(in_hook("merger.act_out")),
    ]
    # The rotary tables are passed to each block as a kwarg, so grab them there
    # rather than recomputing.
    def pre_hook(_m, _args, kwargs):
        cos, sin = kwargs["position_embeddings"]
        got["rope_cos"] = cos.detach().float().clone()
        got["rope_sin"] = sin.detach().float().clone()
        return None

    handles.append(blk.register_forward_pre_hook(pre_hook, with_kwargs=True))

    with torch.no_grad():
        out = model(pixel_values, grid_thw=grid_thw, return_dict=True)
    for h in handles:
        h.remove()

    got["last_hidden"] = out.last_hidden_state.detach().float().clone()
    got["image_embeds"] = out.pooler_output.detach().float().clone()

    merge = vcfg.spatial_merge_size
    cu, _ = get_vision_attention_seqlens(grid_thw, vcfg)
    got["cu_seqlens"] = cu.float()
    got["position_ids"] = get_vision_position_ids(grid_thw, merge).float()
    idx, wts = get_vision_interpolation_indices_and_weights(
        grid_thw, num_grid_per_side=model.num_grid_per_side,
        mode=model.interpolation_mode,
        align_corners=model.interpolation_align_corners, spatial_merge_size=merge)
    got["interp_indices"] = idx.float()
    got["interp_weights"] = wts.float()
    with torch.no_grad():
        got["pos_embeds"] = (model.pos_embed(idx) * wts[:, :, None]).sum(1).float()
    got["hidden_in"] = got["patch_embed_out"] + got["pos_embeds"]
    got["pixel_values"] = pixel_values.float()
    got["grid_thw"] = grid_thw.float()
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    args = ap.parse_args()

    torch.set_num_threads(min(32, os.cpu_count() or 8))
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoImageProcessor

    headers = read_index(args.model_dir)
    print("building the reference vision tower on CPU")
    cfg, vcfg, model = build_tower(args.model_dir, headers)
    merge = vcfg.spatial_merge_size
    heads = vcfg.num_heads
    head_dim = vcfg.hidden_size // heads
    print(f"  depth={vcfg.depth} hidden={vcfg.hidden_size} heads={heads} "
          f"head_dim={head_dim} inter={vcfg.intermediate_size} "
          f"out_hidden={vcfg.out_hidden_size} patch={vcfg.patch_size} "
          f"tpatch={vcfg.temporal_patch_size} merge={merge} "
          f"pos_grid={model.num_grid_per_side}^2")

    ip = AutoImageProcessor.from_pretrained(args.model_dir)

    # Small grids on purpose. The layout questions do not get easier at 16k
    # tokens, and a capture a Rust test can hold in memory is a capture that
    # gets run.
    px_a, grid_a = preprocess(ip, (300, 400), 20260822, 12288)   # -> [1, 6, 8]
    px_c, grid_c = preprocess(ip, (150, 500), 20260823, 12288)   # -> [1, 2, 12]
    print(f"  image A grid {grid_a.tolist()} patches {px_a.shape}")
    print(f"  image C grid {grid_c.tolist()} patches {px_c.shape}")

    groups = {}
    # One image alone.
    groups["img"] = (px_a, grid_a)
    # Two images of different shapes packed into one call: pins that attention
    # does not leak across images and that the merger's grouping is per-image.
    groups["pack"] = (torch.cat([px_a, px_c], 0), torch.cat([grid_a, grid_c], 0))
    # A two-frame "video" whose frames differ: pins that attention does not leak
    # across frames either. Two *identical* frames would not discriminate —
    # attending over a duplicated key set gives the same answer.
    px_b = px_a.flip(0).contiguous() * 0.7
    groups["vid"] = (torch.cat([px_a, px_b], 0),
                     torch.tensor([[2, int(grid_a[0, 1]), int(grid_a[0, 2])]]))

    hooked = {}
    for tag, (px, grid) in groups.items():
        print(f"== running the reference tower on group '{tag}' grid {grid.tolist()}")
        hooked[tag] = run_tower(model, vcfg, px, grid)

    print("== cross-checking the transcriptions against the reference")
    verified = cross_check_against_transformers(args.model_dir, vcfg, model, ip, hooked)

    # Is any normalization degenerate? LayerNorm's eps is 1e-6 on the variance,
    # so a stage whose variance lands near 1e-6 normalizes to a constant and the
    # capture would bless any formulation of it. Report, do not assume.
    print("== normalization inputs: variance vs eps=1e-6")
    h = hooked["img"]
    worst = None
    for name in ("hidden_in", "b0.resid1", "last_hidden"):
        x = h[name]
        var = (x - x.mean(-1, keepdim=True)).pow(2).mean(-1)
        lo = float(var.min())
        print(f"  {name:<22} row variance min {lo:.3e} median "
              f"{float(var.median()):.3e} max {float(var.max()):.3e}")
        worst = lo if worst is None else min(worst, lo)
    if worst < 1e-3:
        raise SystemExit(
            f"the smallest LayerNorm input variance is {worst:.3e}, within three "
            f"orders of eps=1e-6; at that scale the norm degenerates to a "
            f"constant scale and this capture cannot discriminate its "
            f"formulation. Use an input with more structure.")

    axis, section, half, text_cfg = mrope_axis_map(args.model_dir)
    print(f"== text-side mRoPE: section {list(section)} over {half} frequencies")
    print(f"   axis per frequency index: {axis.long().tolist()}")

    # A spliced sequence: text, <|vision_start|>, image pads, <|vision_end|>,
    # text. The ids come from the config, not from memory.
    n_tok_a = int(grid_a.prod()) // (merge * merge)
    ids = ([9000, 9001, 9002]
           + [cfg.vision_start_token_id]
           + [cfg.image_token_id] * n_tok_a
           + [cfg.vision_end_token_id]
           + [9003, 9004])
    types = ([0, 0, 0] + [0] + [1] * n_tok_a + [0] + [0, 0])
    pos3, delta = llm_position_ids(args.model_dir, cfg,
                                  torch.tensor([ids]), torch.tensor([types]), grid_a)
    print(f"== splice: {len(ids)} tokens, {n_tok_a} image tokens, "
          f"rope delta {delta.reshape(-1).tolist()}")

    manifest = {
        "config": {
            "depth": vcfg.depth,
            "hidden_size": vcfg.hidden_size,
            "num_heads": heads,
            "head_dim": head_dim,
            "intermediate_size": vcfg.intermediate_size,
            "out_hidden_size": vcfg.out_hidden_size,
            "in_channels": vcfg.in_channels,
            "patch_size": vcfg.patch_size,
            "temporal_patch_size": vcfg.temporal_patch_size,
            "spatial_merge_size": merge,
            "num_position_embeddings": vcfg.num_position_embeddings,
            "num_grid_per_side": model.num_grid_per_side,
            # Read off the module, not written down: `Qwen3_5VisionBlock` spells
            # `nn.LayerNorm(hidden, eps=1e-6)` as a literal and `vision_config`
            # has no field for it, so a port that takes nn.LayerNorm's own
            # default gets 1e-5. check_module_constants refuses if the three
            # norms disagree.
            "layer_norm_eps": float(model.blocks[0].norm1.eps),
            "vision_rope_theta": float(model.rotary_pos_emb.theta),
            "vision_rope_dim": int(model.rotary_pos_emb.dim),
            "image_token_id": cfg.image_token_id,
            "video_token_id": cfg.video_token_id,
            "vision_start_token_id": cfg.vision_start_token_id,
            "vision_end_token_id": cfg.vision_end_token_id,
            "mrope_half": half,
            "smart_resize_factor": vcfg.patch_size * merge,
            "min_pixels": int(ip.size.shortest_edge),
            "max_pixels": int(ip.size.longest_edge),
        },
        "arrays": {},
    }

    def dump(name, t):
        t = torch.as_tensor(t).detach().contiguous().float()
        with open(os.path.join(args.out_dir, name + ".f32"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        manifest["arrays"][name] = list(t.shape)
        flat = t.reshape(-1)
        print(f"  {name:<34} {str(list(t.shape)):<16} "
              f"[{flat.min():+.5f}, {flat.max():+.5f}] "
              f"nonfinite={int((~flat.isfinite()).sum())}")

    print("== patchify probe (from the reference image processor)")
    # An `arange` image: every pixel is a distinct number, so a test can locate
    # each one in the flattened patch instead of agreeing by coincidence.
    pgh, pgw = 6, 8
    probe = torch.arange(
        3 * pgh * vcfg.patch_size * pgw * vcfg.patch_size, dtype=torch.float32
    ).reshape(3, pgh * vcfg.patch_size, pgw * vcfg.patch_size)
    ref_px, _, _ = ip.patchify(probe[None], patch_size=vcfg.patch_size,
                               merge_size=merge,
                               temporal_patch_size=vcfg.temporal_patch_size)
    dump("patchify.probe_image", probe)
    dump("patchify.probe_pixels", ref_px[0])
    dump("patchify.probe_grid", torch.tensor([1, pgh, pgw], dtype=torch.float32))

    print("== weights and small tables")
    patch_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size ** 2
    dump("patch_embed.w_flat",
         model.patch_embed.proj.weight.reshape(vcfg.hidden_size, patch_dim))
    dump("patch_embed.bias", model.patch_embed.proj.bias)
    # One-hot slots spread across the (c, t, y, x) grid, plus their responses:
    # evidence for the flat ordering that does not come from a reshape.
    slots = sorted({0, 1, patch_dim - 1}
                   | {((c * vcfg.temporal_patch_size + t) * vcfg.patch_size + y)
                      * vcfg.patch_size + x
                      for c in (0, 1, 2) for t in (0, 1)
                      for y in (0, 1, 15) for x in (0, 1, 15)})
    eye = torch.zeros(len(slots), patch_dim)
    for i, s in enumerate(slots):
        eye[i, s] = 1.0
    with torch.no_grad():
        dump("patch_embed.onehot_slots", torch.tensor(slots, dtype=torch.float32))
        dump("patch_embed.onehot_out", model.patch_embed(eye))
    dump("pos_embed.table", model.pos_embed.weight)

    blk = model.blocks[0]
    dump("b0.norm1.weight", blk.norm1.weight)
    dump("b0.norm1.bias", blk.norm1.bias)
    dump("b0.norm2.weight", blk.norm2.weight)
    dump("b0.norm2.bias", blk.norm2.bias)
    dump("b0.qkv.bias", blk.attn.qkv.bias)
    dump("b0.proj.bias", blk.attn.proj.bias)
    dump("b0.fc1.bias", blk.mlp.linear_fc1.bias)
    dump("b0.fc2.bias", blk.mlp.linear_fc2.bias)
    # Rows straddling the q/k/v boundaries and a few per-head probes, so a test
    # can locate q, k and v in the weight instead of inheriting this file's
    # split. Under `[all q | all k | all v]` component `s` of head `h` dim `d`
    # is row `s*dim + h*head_dim + d`; under a per-head interleaving it would be
    # row `h*3*head_dim + s*head_dim + d`.
    dim = vcfg.hidden_size
    rows = set()
    for s in (0, 1, 2):
        rows |= {s * dim, s * dim + dim - 1}
        for hh in (0, 1, heads - 1):
            for dd in (0, 1, head_dim - 1):
                rows |= {s * dim + hh * head_dim + dd,
                         hh * 3 * head_dim + s * head_dim + dd}
    rows = sorted(r for r in rows if r < 3 * dim)
    dump("b0.qkv.probe_rows", torch.tensor(rows, dtype=torch.float32))
    dump("b0.qkv.probe_w", blk.attn.qkv.weight[rows])

    dump("merger.norm.weight", model.merger.norm.weight)
    dump("merger.norm.bias", model.merger.norm.bias)
    dump("merger.fc1.bias", model.merger.linear_fc1.bias)
    dump("merger.fc2.bias", model.merger.linear_fc2.bias)
    mrows = sorted({0, 1, 1151, 1152, 1153, 2304, 3456, 4607})
    dump("merger.fc1.probe_rows", torch.tensor(mrows, dtype=torch.float32))
    dump("merger.fc1.probe_w", model.merger.linear_fc1.weight[mrows])

    print("== per-group activations")
    full = ("pixel_values", "grid_thw", "cu_seqlens", "position_ids",
            "interp_indices", "interp_weights", "pos_embeds", "patch_embed_out",
            "hidden_in", "rope_cos", "rope_sin", "b0.norm1_out", "b0.qkv",
            "b0.attn_pre_proj", "b0.attn_out", "b0.resid1", "b0.norm2_out",
            "b0.fc1_out", "b0.act_out", "b0.mlp_out", "b0.out", "last_hidden",
            "merger.norm_out", "merger.fc1_out", "merger.act_out", "image_embeds")
    slim = ("grid_thw", "cu_seqlens", "position_ids", "hidden_in", "rope_cos",
            "b0.qkv", "b0.attn_pre_proj", "b0.out", "last_hidden", "image_embeds")
    for tag, h in hooked.items():
        for name in (full if tag == "img" else slim):
            dump(f"{tag}.{name}", h[name])

    print("== text-side splice")
    dump("mrope.axis_of_index", axis)
    dump("mrope.section", torch.tensor(list(section), dtype=torch.float32))
    dump("splice.input_ids", torch.tensor(ids, dtype=torch.float32))
    dump("splice.mm_token_type_ids", torch.tensor(types, dtype=torch.float32))
    dump("splice.grid_thw", grid_a.float())
    dump("splice.position_ids", pos3[:, 0].float())
    dump("splice.rope_delta", delta.reshape(-1).float())

    print("== smart_resize table (from the reference function)")
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        smart_resize as ref_smart_resize,
    )
    # Aspect ratios stay under 200:1 — the reference raises above that rather
    # than resizing, which is itself worth knowing and is noted in the spec.
    cases = [(300, 400), (150, 500), (64, 64), (4000, 3000), (256, 256),
             (33, 33), (1024, 1024), (200, 4000), (8000, 60), (12000, 9000)]
    tbl = []
    for h_, w_ in cases:
        hb, wb = ref_smart_resize(h_, w_, factor=vcfg.patch_size * merge,
                                  min_pixels=int(ip.size.shortest_edge),
                                  max_pixels=int(ip.size.longest_edge))
        tbl.append([h_, w_, hb, wb])
    dump("smart_resize.cases", torch.tensor(tbl, dtype=torch.float32))

    manifest["verified_against_transformers"] = verified
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {len(manifest['arrays'])} arrays to {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
