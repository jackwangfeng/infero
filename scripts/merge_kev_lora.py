#!/usr/bin/env python3
"""Merge a Kev LoRA adapter into its frozen Qwen3.5 base, offline, so infero's
existing (unmodified) safetensors loader can load the result like any other
plain checkpoint. Mirrors kev/checkpoint.py's own LoadOptions.merge semantics:
the delta is computed in fp32 from the fp32 adapter and added in fp32, with
one rounding to the base's storage dtype -- not peft's merge_and_unload()
(not installed in this venv, and this is simple enough not to need it).

Also converts head.pt's PointerHead (q/k Linear weights + bias, calibration
temperature) into a small safetensors + JSON sidecar, since head.pt is a
raw torch.save pickle and infero's Rust loader only reads safetensors/gguf.

Usage:
    python3 scripts/merge_kev_lora.py <base-checkpoint-dir> <kev-adapter-dir> <out-dir>

<base-checkpoint-dir>: a plain HF download of the base model named in
<kev-adapter-dir>/adapter_config.json's "base_model_name_or_path" (e.g.
Qwen/Qwen3.5-0.8B-Base), as a safetensors directory -- every *.safetensors
file in it is read and re-written shard for shard (no index.json needed;
infero_safetensors::Shards::open_dir globs the directory the same way, so
this just has to produce valid shards, not any particular shard count).
<kev-adapter-dir>: a Kev checkpoint directory (e.g. a
`huggingface-cli download jaredpalmer/kev-4b` snapshot) carrying
adapter_config.json/adapter_model.safetensors/head.pt.

For a full-weight Kev checkpoint (e.g. kev-27b, no LoRA adapter at all --
check its training_config.json for "weights": "full"), skip this script
entirely: point infero straight at the checkpoint directory, and run only
`convert_head(checkpoint_dir, checkpoint_dir)` (see this file's __main__) to
get its kev_head.safetensors/kev_head.json sidecar.

Real run, 2026-10-01 (jaredpalmer/kev-0.8b + Qwen/Qwen3.5-0.8B-Base): 186/186
LoRA targets matched and merged; the result loaded and scored correctly
through infero's real /v1/systemone endpoint (see docs/keel-integration.md's
"5. /v1/systemone" section).
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main(base_dir, adapter_dir, out_dir):
    base_dir, adapter_dir, out_dir = Path(base_dir), Path(adapter_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    r, alpha = adapter_cfg["r"], adapter_cfg["lora_alpha"]
    scale = alpha / r
    print(f"lora r={r} alpha={alpha} scale={scale}")

    # name translation: adapter tensors are "base_model.model.layers.N.X.lora_A/B.weight";
    # the base checkpoint's own tensors are "model.language_model.layers.N.X.weight".
    lora = {}
    with safe_open(adapter_dir / "adapter_model.safetensors", framework="pt") as f:
        for k in f.keys():
            assert k.startswith("base_model.model.layers.") and k.endswith((".lora_A.weight", ".lora_B.weight"))
            rest = k[len("base_model.model.layers."):]
            layer, _, tail = rest.partition(".")
            proj, _ab = tail.rsplit(".", 1)    # tail = "linear_attn.in_proj_a.lora_A.weight", _ab = "weight"
            proj, kind = proj.rsplit(".", 1)   # proj = "linear_attn.in_proj_a", kind = "lora_A"
            lora.setdefault((int(layer), proj), {})[kind] = f.get_tensor(k).float()

    print(f"{len(lora)} (layer, proj) targets to merge")

    shards = sorted(base_dir.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no .safetensors files in {base_dir}")
    print(f"{len(shards)} base shard(s): {[s.name for s in shards]}")

    merged_count = 0
    for shard in shards:
        merged = {}
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                hit = None
                if k.startswith("model.language_model.layers.") and k.endswith(".weight"):
                    rest = k[len("model.language_model.layers."):-len(".weight")]
                    layer, _, proj = rest.partition(".")
                    if (int(layer), proj) in lora:
                        hit = (int(layer), proj)
                if hit is not None:
                    ab = lora.pop(hit)
                    delta = scale * (ab["lora_B"] @ ab["lora_A"])
                    assert delta.shape == t.shape, f"{k}: base {t.shape} vs lora delta {delta.shape}"
                    merged_t = (t.float() + delta).to(t.dtype)
                    merged[k] = merged_t.contiguous()
                    merged_count += 1
                else:
                    merged[k] = t
        save_file(merged, out_dir / shard.name, metadata={"format": "pt"})
        print(f"wrote {out_dir / shard.name} ({len(merged)} tensors)")
    print(f"merged {merged_count} tensors across {len(shards)} shard(s)")
    if lora:
        raise SystemExit(f"{len(lora)} lora targets never matched a base tensor: {list(lora)[:5]}")

    (out_dir / "config.json").write_text((base_dir / "config.json").read_text())
    for extra in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json", "special_tokens_map.json", "added_tokens.json"):
        src = base_dir / extra
        if src.exists():
            (out_dir / extra).write_bytes(src.read_bytes())
    print(f"wrote merged checkpoint to {out_dir}")


def convert_head(kev_dir, out_dir):
    kev_dir, out_dir = Path(kev_dir), Path(out_dir)
    d = torch.load(kev_dir / "head.pt", map_location="cpu", weights_only=False)
    head = d["head"]
    save_file(
        {k: v.contiguous() for k, v in head.items()},
        out_dir / "kev_head.safetensors",
    )
    meta = {
        "base": d["base"],
        "head_dim": d["head_dim"],
        "temperature": d["temperature"],
        "option_isolation": d["option_isolation"],
        "weights": d.get("weights", "lora"),
    }
    (out_dir / "kev_head.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_dir / 'kev_head.safetensors'} and kev_head.json: {meta}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    base_dir, adapter_dir, out_dir = sys.argv[1:4]
    main(base_dir, adapter_dir, out_dir)
    convert_head(adapter_dir, out_dir)
