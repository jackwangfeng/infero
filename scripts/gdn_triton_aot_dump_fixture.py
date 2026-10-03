#!/usr/bin/env python3
"""Dumps a small, real Python-computed GDN correctness fixture to raw binary
files `crates/kernels/examples/gdn_triton_aot_probe.rs` reads directly (no
`torch`/`ctypes` needed on the Rust side -- just a directory of contiguous,
native-byte-order tensors).

Not vendored into the repo as fixture *data* (see
`crates/kernels/src/gdn_triton_aot.rs`'s own doc comment for why the
compiled artifact itself isn't either) -- run this on the box that has the
real `vllm.third_party.flash_linear_attention` install (`bw`, in
`/home/jeff/vllm-venv`) to regenerate it:

    /home/jeff/vllm-venv/bin/python3 scripts/gdn_triton_aot_dump_fixture.py \\
        --out /tmp/gdn_triton_aot_fixture

Small shape by design (B=1, T=256, so NT=4): this only needs to prove the
Rust-side FFI call reproduces a real reference bit-for-bit (or very close,
per dtype), not to be representative of a full-size layer -- the benchmark
side of the probe uses its own larger synthetic T=4096 shape and does not
need a reference (correctness at that shape was already established via
this same repo's AOT-compile phase, which found the raw launcher and the
normal Triton JIT path bit-identical at T=4096 too -- see that phase's own
report for the exact comparison).
"""

import argparse
import pathlib

import torch
from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)

B, T, HG, K, H, V, BT = 1, 256, 16, 128, 48, 128, 64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory to write the fixture into")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = "cuda"
    torch.cuda.init()

    k = (torch.randn(B, T, HG, K, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    w = (torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    u = (torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    g = (torch.randn(B, T, H, device=device, dtype=torch.float32) * -0.01).contiguous()
    initial_state = (torch.randn(B, H, V, K, device=device, dtype=torch.float32) * 0.1).contiguous()

    h_ref, v_new_ref, final_state_ref = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=None,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=BT,
        save_new_value=True,
        cu_seqlens=None,
    )
    torch.cuda.synchronize()

    tensors = {
        "k": k,
        "w": w,
        "u": u,
        "g": g,
        "initial_state": initial_state,
        "h_ref": h_ref.contiguous(),
        "v_new_ref": v_new_ref.contiguous(),
        "final_state_ref": final_state_ref.contiguous(),
    }
    for name, t in tensors.items():
        path = out / f"{name}.bin"
        t_cpu = t.cpu()
        # numpy has no bfloat16 -- reinterpret the same 2 bytes/element as
        # int16 first (a bit-pattern view, not a value conversion) so
        # `.tobytes()` still gets the real raw bf16 bytes Rust re-parses.
        if t_cpu.dtype == torch.bfloat16:
            t_cpu = t_cpu.view(torch.int16)
        path.write_bytes(t_cpu.numpy().tobytes())
        print(f"{name}: {tuple(t.shape)} {t.dtype} -> {path} ({path.stat().st_size} bytes)")

    print(f"\nshape: B={B} T={T} Hg={HG} K={K} H={H} V={V} BT={BT} NT={(T + BT - 1) // BT}")
    print("(these constants are also hardcoded in gdn_triton_aot_probe.rs -- keep both in sync)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
