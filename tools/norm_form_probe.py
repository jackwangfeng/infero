#!/usr/bin/env python3
"""Which RMSNorm form does each norm tensor in a checkpoint use?

`Qwen3_5RMSNorm` initializes its weight to *zeros* and computes
`normalized * (1 + weight)`, while `Qwen3RMSNorm` initializes to ones and
computes `weight * normalized`. Two conventions in one file: Qwen3.5's regular
norms use the offset form and `Qwen3_5RMSNormGated` — the GatedDeltaNet output
norm — uses the plain one.

That distinction is invisible to a test that shares the mistake. **The source is
the only authority here**: every `Qwen3_5RMSNorm` uses the offset form and every
`Qwen3_5RMSNormGated` uses the plain one, so which form a tensor wants follows
from which class consumes it, not from the tensor.

This script exists to *corroborate* that reading, not to decide it — and it is
worth knowing that it cannot decide it. The two populations do separate on
average, but they overlap: some trained `q_norm` deltas exceed 0.5 while some
`linear_attn.norm` gains fall below 1.5. A mean-based classifier gets those
wrong. What the output is good for is a sanity check that the whole
`input_layernorm` population sits near zero, which the plain form would
annihilate — that much is decisive, and it was the first hint that something
was off.

    python3 norm_form_probe.py <model-dir>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture_qwen35_layers import load_f32, read_index  # noqa: E402


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    headers = read_index(sys.argv[1])
    norms = sorted(n for n in headers if n.endswith("norm.weight") or "layernorm" in n)
    print(f"{'tensor':<64} {'mean':>8} {'min':>8} {'max':>8}  implied")
    verdicts = {}
    for name in norms:
        t = load_f32(headers, name).float()
        mean = t.mean().item()
        # Not a verdict — a hint. See the module docstring: the populations
        # overlap and the source decides.
        form = "near 0" if abs(mean) < 0.5 else "near 1"
        verdicts.setdefault(form, []).append(name)
        short = name.replace("model.language_model.", "").replace("model.", "")
        print(f"{short:<64} {mean:>8.4f} {t.min():>8.4f} {t.max():>8.4f}  {form}")

    print()
    print("Grouped by where the mean lands. This is NOT the classification:")
    print("the source is, and it says every Qwen3_5RMSNorm is (1 + w) while")
    print("Qwen3_5RMSNormGated — only linear_attn.norm — is plain w.")
    print("q_norm and k_norm appearing in both groups below is exactly why a")
    print("mean-based reading cannot be trusted.")
    for form, names in verdicts.items():
        print(f"mean {form}: {len(names)} tensors")
        # Print the distinct shapes of names, collapsing layer indices.
        import re
        kinds = sorted({re.sub(r"\.\d+\.", ".N.", n) for n in names})
        for k in kinds:
            print(f"    {k.replace('model.language_model.', '')}")


if __name__ == "__main__":
    main()
