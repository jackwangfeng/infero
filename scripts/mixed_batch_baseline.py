#!/usr/bin/env python3
"""Capture token-level baseline transcripts for the mixed-batch dispatch
split (docs/superpowers/plans/2026-09-05-mixed-batch-attention-dispatch-split.md,
Task 1). Re-run this same script, unmodified, against the POST-change binary
in Task 7 and diff the saved JSON files token-for-token."""
import json
import sys
import time
import threading
import urllib.request

BASE = "http://127.0.0.1:8301"
SEED = 42424242
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "docs/superpowers/plans/mixed_batch_baseline"

def chat(messages, max_tokens=64, seed=SEED):
    body = json.dumps({
        "model": "qwen38-27b-fp8",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "logprobs": False,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())

def save(name, obj):
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/{name}.json", "w") as f:
        json.dump(obj, f, indent=2)
    print(f"saved {OUT_DIR}/{name}.json")

# Scenario A: pure single-sequence prefill+decode -- exercises the existing
# single_seq_run path, must be untouched by this change.
resp_a = chat([{"role": "user", "content": "Count from 1 to 20, one number per line."}], max_tokens=80)
save("scenario_a_single_seq", resp_a)

# Scenario B: engineered mixed decode+prefill batch. Fire a long prompt
# (forces multi-chunk prefill, several batch_tokens=8192-sized chunks) on one
# connection, and a fraction of a second later fire a short prompt on a
# second connection whose first decode step should land in the SAME
# scheduler batch as one of the long prompt's later prefill chunks.
long_prompt = "Please summarize this list in detail, one sentence per item: " + \
    ", ".join(f"item number {i} is about topic {i%7}" for i in range(4000))
results = {}
def run_long():
    results["long"] = chat([{"role": "user", "content": long_prompt}], max_tokens=40)
def run_short():
    time.sleep(0.05)  # let the long prompt's first chunk get scheduled first
    results["short"] = chat([{"role": "user", "content": "What is 2+2? Answer with just the number."}], max_tokens=8)
t1 = threading.Thread(target=run_long)
t2 = threading.Thread(target=run_short)
t1.start(); t2.start()
t1.join(); t2.join()
save("scenario_b_mixed_decode_prefill", results)

# Scenario C: two simultaneous prefills (two prompts admitted close together,
# both still chunk-prefilling in the same batch).
prompt_x = "Explain photosynthesis in exactly three sentences."
prompt_y = "Explain how a car engine works in exactly three sentences."
results_c = {}
def run_x():
    results_c["x"] = chat([{"role": "user", "content": prompt_x}], max_tokens=60)
def run_y():
    results_c["y"] = chat([{"role": "user", "content": prompt_y}], max_tokens=60)
tx = threading.Thread(target=run_x)
ty = threading.Thread(target=run_y)
tx.start(); ty.start()
tx.join(); ty.join()
save("scenario_c_two_prefills", results_c)

print("done")
