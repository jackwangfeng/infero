#!/usr/bin/env python3
"""Capture reference logits from Hugging Face for the forward-pass test.

Coherent output is weak evidence: a subtly wrong RoPE base or a transposed
projection still produces fluent text. This records what the reference
implementation actually predicts, so the Rust engine can be held to it.

    python scripts/make_logits_fixtures.py Qwen/Qwen2.5-0.5B-Instruct
"""

import json
import pathlib
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    "人工智能是",
]

TOP_K = 20


def main() -> int:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
    out_path = pathlib.Path(__file__).resolve().parent.parent / (
        "crates/model/tests/fixtures/qwen2.5-0.5b-instruct-logits.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_id)
    # float32 on CPU: this is the reference, so precision matters more than
    # speed, and it keeps the numbers independent of any GPU kernel.
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()

    cases = []
    for prompt in PROMPTS:
        ids = tok.encode(prompt, add_special_tokens=False)
        with torch.no_grad():
            out = model(torch.tensor([ids]))
        logits = out.logits[0, -1].float()
        top = torch.topk(logits, TOP_K)
        cases.append(
            {
                "prompt": prompt,
                "ids": ids,
                "top_ids": top.indices.tolist(),
                "top_logits": [round(v, 4) for v in top.values.tolist()],
                "mean": round(logits.mean().item(), 6),
                "std": round(logits.std().item(), 6),
                "argmax": int(logits.argmax()),
                "argmax_piece": tok.decode([int(logits.argmax())]),
            }
        )
        print(f"{prompt[:40]!r:45} -> {cases[-1]['argmax_piece']!r}")

    out_path.write_text(
        json.dumps({"model": model_id, "top_k": TOP_K, "cases": cases},
                   ensure_ascii=False, indent=1)
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
