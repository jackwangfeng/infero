#!/usr/bin/env python3
"""Run the real first few layers of a Qwen3.5 checkpoint on CPU and dump the
actual activations entering a GatedDeltaNet layer and a gated-attention layer.

Why this exists on top of the synthetic capture: with random input the whole
block runs five orders of magnitude below `rms_norm_eps`, so the gated RMSNorm
degenerates to a constant `1/sqrt(eps)` scale and the capture cannot tell
"normalize then gate" from "gate then normalize". Random vectors decorrelate
every projection and give up a factor of sqrt(d) in magnitude; real hidden
states do not. A capture taken in a regime where the operation under test does
nothing will bless any implementation of it.

The prefix runs with the reference implementation's own module classes — not a
transcription — so the activations are what the model actually produces. Only
the weight loading is ours, because the checkpoint is FP8 with block scales and
that path wants a GPU in transformers.

    python3 qwen35_real_prefix.py <model-dir> <out-dir> [--layers 4] [--tokens 12]
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture_qwen35_layers import load_f32, read_index  # noqa: E402


def build_prefix(model_dir, n_layers):
    """Embedding plus the first `n_layers` decoder layers, real weights, bf16."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    raw = json.load(open(os.path.join(model_dir, "config.json")))
    cfg = Qwen3_5TextConfig(**raw["text_config"])
    cfg._attn_implementation = "eager"
    headers = read_index(model_dir)

    emb = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        emb.weight.copy_(load_f32(headers, "model.language_model.embed_tokens.weight"))

    layers = []
    for i in range(n_layers):
        layer = m.Qwen3_5DecoderLayer(cfg, i).to(torch.bfloat16).eval()
        prefix = f"model.language_model.layers.{i}."
        missing = []
        with torch.no_grad():
            for name, param in layer.named_parameters():
                key = prefix + name
                if key not in headers:
                    missing.append(name)
                    continue
                w = load_f32(headers, key)
                if w.shape != param.shape:
                    # conv1d is [C, 1, K] in the checkpoint and in the module;
                    # anything else disagreeing is a real mismatch.
                    w = w.reshape(param.shape)
                param.copy_(w)
        if missing:
            raise SystemExit(f"layer {i}: no weights for {missing}")
        layers.append(layer)
        print(f"  layer {i} ({cfg.layer_types[i]}) loaded")
    return cfg, emb, layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--layers", type=int, default=4,
                    help="how many decoder layers to run; 4 reaches the first "
                         "full-attention layer (index 3)")
    ap.add_argument("--tokens", type=int, default=12)
    args = ap.parse_args()

    torch.set_num_threads(min(64, os.cpu_count() or 8))
    os.makedirs(args.out_dir, exist_ok=True)

    print("building the real prefix on CPU")
    cfg, emb, layers = build_prefix(args.model_dir, args.layers)

    # Real token ids. A fixed arbitrary sequence inside the vocabulary is enough
    # — what matters is that the hidden states are the embedding's own output and
    # then genuinely processed, not that the text means anything.
    torch.manual_seed(20260822)
    ids = torch.randint(0, cfg.vocab_size, (1, args.tokens))

    captured = {}

    def hook(idx):
        def fn(_mod, inputs, output):
            captured[idx] = output.detach().float()[0].clone()
        return fn

    handles = [l.input_layernorm.register_forward_hook(hook(i))
               for i, l in enumerate(layers)]

    rot = int(cfg.head_dim * cfg.rope_parameters.get("partial_rotary_factor", 1.0))
    theta = cfg.rope_parameters["rope_theta"]
    pos = torch.arange(args.tokens)
    inv = 1.0 / (theta ** (torch.arange(0, rot, 2, dtype=torch.float32) / rot))
    freqs = pos[:, None].float() * inv[None, :]
    embt = torch.cat([freqs, freqs], dim=-1)
    pe = (embt.cos().to(torch.bfloat16)[None], embt.sin().to(torch.bfloat16)[None])

    with torch.no_grad():
        h = emb(ids)
        for i, layer in enumerate(layers):
            out = layer(h, position_embeddings=pe, attention_mask=None)
            h = out[0] if isinstance(out, tuple) else out
            print(f"  after layer {i}: hidden RMS "
                  f"{h.float().pow(2).mean().sqrt().item():.4f}")
    for hd in handles:
        hd.remove()

    manifest = {}
    for idx, act in captured.items():
        name = f"real_input.layer{idx}"
        with open(os.path.join(args.out_dir, name + ".f32"), "wb") as fh:
            fh.write(act.contiguous().numpy().tobytes())
        manifest[name] = list(act.shape)
        rms = act.pow(2).mean().sqrt().item()
        print(f"  {name}: shape {list(act.shape)}  RMS {rms:.4f}  "
              f"type {cfg.layer_types[idx]}")

    meta = {
        "arrays": manifest,
        "token_ids": ids[0].tolist(),
        "layer_types": cfg.layer_types[: args.layers],
    }
    with open(os.path.join(args.out_dir, "real_prefix.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nwrote {len(manifest)} real activation tensors to {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
