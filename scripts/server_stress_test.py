#!/usr/bin/env python3
"""Real, varied load test for a running infero server, driven over its real
HTTP API -- no mocks. Built to catch the class of bug found 2026-09-05: the
FA2 attention backend's KV-slot-contiguity assumption, which only broke under
real multi-turn / concurrent traffic, never under the single-shot
`prefill_profile`-style benchmarks this session otherwise relied on all day.

Usage:
    python3 scripts/server_stress_test.py [--base-url http://127.0.0.1:8301] [--passes 5]

Exit code 0 iff every category passed on every pass. Prints a per-category,
per-pass pass/fail table and any real error bodies seen, not just a verdict.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.request
import urllib.error

MODEL = "qwen38-27b-fp8"


def chat(base_url: str, messages: list[dict], max_tokens: int = 32, timeout: int = 120) -> tuple[int, dict | str]:
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw
    except Exception as e:  # noqa: BLE001 - real network/timeout failures are real results here
        return -1, f"{type(e).__name__}: {e}"


def content_of(resp: dict) -> str:
    try:
        return resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""


def ok(status: int, resp) -> bool:
    return status == 200 and isinstance(resp, dict) and bool(content_of(resp).strip())


# ---------------------------------------------------------------------------
# Test categories. Each returns (passed: bool, detail: str).
# ---------------------------------------------------------------------------

def t_single_turn_short(base_url: str) -> tuple[bool, str]:
    status, resp = chat(base_url, [{"role": "user", "content": "What is the capital of France?"}])
    return ok(status, resp), f"status={status} content={content_of(resp) if isinstance(resp, dict) else resp!r}"


def t_single_turn_medium(base_url: str) -> tuple[bool, str]:
    # ~2000 words of filler + a real question, enough to be a real multi-token
    # prompt without needing an artificially constructed >8192-token blob.
    filler = ("The quick brown fox jumps over the lazy dog. " * 250)
    status, resp = chat(base_url, [{"role": "user", "content": filler + "\n\nWhat is 7 times 6?"}])
    return ok(status, resp), f"status={status} content={content_of(resp) if isinstance(resp, dict) else resp!r}"


def t_single_turn_long_chunked(base_url: str) -> tuple[bool, str]:
    # Real prefill chunking is CUTLASS_BATCH_TOKENS=8192 -- force more than one
    # chunk with a real, long, repetitive-but-real-token prompt.
    filler = ("The quick brown fox jumps over the lazy dog. " * 2200)  # ~9-10k tokens
    status, resp = chat(base_url, [{"role": "user", "content": filler + "\n\nWhat animal jumped over the dog?"}], max_tokens=48)
    return ok(status, resp), f"status={status} prompt_tokens={resp.get('usage', {}).get('prompt_tokens') if isinstance(resp, dict) else '?'} content={content_of(resp) if isinstance(resp, dict) else resp!r}"


def t_multiturn(base_url: str, n_turns: int) -> tuple[bool, str]:
    messages: list[dict] = [{"role": "user", "content": "Remember this secret number: 42. Just acknowledge briefly."}]
    for i in range(n_turns):
        status, resp = chat(base_url, messages, max_tokens=40)
        if not ok(status, resp):
            return False, f"turn {i+1}/{n_turns} failed: status={status} body={resp!r}"
        reply = content_of(resp)
        messages.append({"role": "assistant", "content": reply})
        if i < n_turns - 1:
            messages.append({"role": "user", "content": f"Turn {i+2}: say a random short fact."})
    # Final turn: ask for the secret back, a real content-sensibility check.
    # max_tokens=64, not 40: by turn 10 there's a lot of accumulated fake
    # history to reference before this checkpoint's verbose reasoning
    # preamble reaches the actual "42" -- 40 flaked real (2026-09-05, 1/5
    # passes truncated mid-reasoning, never reached the answer) on an
    # otherwise-correct, uncontaminated model, same class of false failure
    # as `t_retire_and_reuse`'s original max_tokens=16.
    messages.append({"role": "user", "content": "What was the secret number I told you earlier?"})
    status, resp = chat(base_url, messages, max_tokens=64)
    if not ok(status, resp):
        return False, f"final recall turn failed: status={status} body={resp!r}"
    final = content_of(resp)
    recalled = "42" in final
    return recalled, f"{n_turns} turns ok, recall_turn_status={status} recalled_42={recalled} content={final!r}"


def t_concurrent(base_url: str, n: int) -> tuple[bool, str]:
    prompts = [f"Count from 1 to 5, then tell me a fact about the number {i}." for i in range(n)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(chat, base_url, [{"role": "user", "content": p}], 40, 180) for p in prompts]
        results = [f.result() for f in futs]
    failures = [(i, s, r) for i, (s, r) in enumerate(results) if not ok(s, r)]
    if failures:
        return False, f"{len(failures)}/{n} concurrent requests failed: " + "; ".join(
            f"[{i}] status={s} body={r!r}" for i, s, r in failures[:3]
        )
    return True, f"all {n} concurrent requests ok"


def t_retire_and_reuse(base_url: str) -> tuple[bool, str]:
    # Finish a short conversation, then start a fresh one -- exercises the
    # pool's free-slot reuse path (the same class of state the contiguity bug
    # lived in) via real HTTP traffic, not the tp_generate.rs bypass tool.
    #
    # max_tokens=32, not 16: this checkpoint's chat template always emits a
    # reasoning preamble ("User asks: ... Need answer concise. Final: X.
    # </think>\n\n<answer>") before the real answer -- confirmed by manual
    # reproduction (2026-09-05) that 16 tokens routinely truncates mid-preamble,
    # never reaching "Tokyo" at all, on an otherwise perfectly correct,
    # uncontaminated response (verified separately with 80 tokens). That was
    # a false failure in this harness, not a real pool-corruption bug -- don't
    # reintroduce it by shrinking this back down.
    status1, resp1 = chat(base_url, [{"role": "user", "content": "Say 'first conversation done'."}], max_tokens=16)
    if not ok(status1, resp1):
        return False, f"first conversation failed: status={status1} body={resp1!r}"
    status2, resp2 = chat(base_url, [{"role": "user", "content": "What is the capital of Japan?"}], max_tokens=32)
    if not ok(status2, resp2):
        return False, f"second (post-retire) conversation failed: status={status2} body={resp2!r}"
    return "Tokyo" in content_of(resp2) or "tokyo" in content_of(resp2).lower(), (
        f"first={content_of(resp1)!r} second={content_of(resp2)!r}"
    )


def t_tool_calling(base_url: str) -> tuple[bool, str]:
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
        "tools": tools,
        "max_tokens": 100,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status, parsed = resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        return False, f"status={e.code} body={raw}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    if status != 200:
        return False, f"status={status} body={parsed!r}"
    msg = parsed.get("choices", [{}])[0].get("message", {})
    has_call = bool(msg.get("tool_calls"))
    has_text = bool((msg.get("content") or "").strip())
    return has_call or has_text, f"status={status} tool_calls={msg.get('tool_calls')!r} content={msg.get('content')!r}"


CATEGORIES: list[tuple[str, "callable"]] = [
    ("single_turn_short", t_single_turn_short),
    ("single_turn_medium", t_single_turn_medium),
    ("single_turn_long_chunked", t_single_turn_long_chunked),
    ("multiturn_2", lambda u: t_multiturn(u, 2)),
    ("multiturn_5", lambda u: t_multiturn(u, 5)),
    ("multiturn_10", lambda u: t_multiturn(u, 10)),
    ("concurrent_at_capacity_2", lambda u: t_concurrent(u, 2)),
    ("concurrent_over_capacity_4", lambda u: t_concurrent(u, 4)),
    ("retire_and_reuse", t_retire_and_reuse),
    ("tool_calling", t_tool_calling),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8301")
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--only", default=None, help="comma-separated category names to run, default all")
    args = ap.parse_args()

    names = set(args.only.split(",")) if args.only else None
    cats = [(n, f) for n, f in CATEGORIES if names is None or n in names]

    results: dict[str, list[tuple[bool, str]]] = {n: [] for n, _ in cats}
    for p in range(1, args.passes + 1):
        print(f"=== pass {p}/{args.passes} ===")
        for name, fn in cats:
            t0 = time.time()
            try:
                passed, detail = fn(args.base_url)
            except Exception as e:  # noqa: BLE001 - a harness-level exception is a real failure to report
                passed, detail = False, f"harness exception: {type(e).__name__}: {e}"
            dt = time.time() - t0
            results[name].append((passed, detail))
            print(f"  [{'PASS' if passed else 'FAIL'}] {name} ({dt:.1f}s): {detail[:200]}")

    print("\n=== summary ===")
    all_ok = True
    for name, runs in results.items():
        n_pass = sum(1 for p, _ in runs if p)
        n_total = len(runs)
        flaky = 0 < n_pass < n_total
        status = "STABLE-PASS" if n_pass == n_total else ("FLAKY" if flaky else "STABLE-FAIL")
        if status != "STABLE-PASS":
            all_ok = False
        print(f"  {name}: {n_pass}/{n_total} passed -- {status}")
        if status != "STABLE-PASS":
            for i, (p, d) in enumerate(runs):
                if not p:
                    print(f"      pass {i+1} detail: {d[:300]}")

    print(f"\noverall: {'ALL STABLE' if all_ok else 'REAL FAILURES OR FLAKINESS FOUND'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
