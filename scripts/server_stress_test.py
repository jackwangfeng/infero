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
    # max_tokens=200, not 64: 64 itself still flaked real at n_turns=2
    # (2026-09-05, reproduced 6/6 truncated mid-reasoning at max_tokens=64,
    # `finish_reason=length`, never reaching "42") on an otherwise-correct,
    # uncontaminated model -- this checkpoint's reasoning preamble length
    # before a recall answer is highly variable (confirmed to run 150-300+
    # tokens even at n_turns=2, not proportional to conversation length the
    # way the original max_tokens=64 fix assumed), same class of false
    # failure as `t_retire_and_reuse`'s original max_tokens=16. Verified at
    # max_tokens=300 that the model always resolves correctly to "42" once
    # given enough room; 200 keeps real margin above that.
    messages.append({"role": "user", "content": "What was the secret number I told you earlier?"})
    status, resp = chat(base_url, messages, max_tokens=200)
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


# A real 64x64 solid-red PNG and a real 2s solid-blue 64x64 h.264 mp4,
# generated once (PIL / ffmpeg) and embedded so this script has zero extra
# runtime dependencies. Added 2026-09-05 to close a real 0%-coverage gap
# (`qwen35_vision_image.rs`) found by this session's own coverage measurement
# -- vision/video had never been exercised by any test all session.
TEST_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAX0lEQVR4nO3PQQ0AIBDAMMC/50MEj4ZkVbDtWX87OuBVA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA9oFUoUBf3Xr7AgAAAAASUVORK5CYII="
)
TEST_VIDEO_B64 = (
    "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAAAwttZGF0AAACrQYF//+p3EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NCByMzEwOCAzMWUxOWY5IC0gSC4yNjQvTVBFRy00IEFWQyBjb2RlYyAtIENvcHlsZWZ0IDIwMDMtMjAyMyAtIGh0dHA6Ly93d3cudmlkZW9sYW4ub3JnL3gyNjQuaHRtbCAtIG9wdGlvbnM6IGNhYmFjPTEgcmVmPTMgZGVibG9jaz0xOjA6MCBhbmFseXNlPTB4MzoweDExMyBtZT1oZXggc3VibWU9NyBwc3k9MSBwc3lfcmQ9MS4wMDowLjAwIG1peGVkX3JlZj0xIG1lX3JhbmdlPTE2IGNocm9tYV9tZT0xIHRyZWxsaXM9MSA4eDhkY3Q9MSBjcW09MCBkZWFkem9uZT0yMSwxMSBmYXN0X3Bza2lwPTEgY2hyb21hX3FwX29mZnNldD0tMiB0aHJlYWRzPTIgbG9va2FoZWFkX3RocmVhZHM9MSBzbGljZWRfdGhyZWFkcz0wIG5yPTAgZGVjaW1hdGU9MSBpbnRlcmxhY2VkPTAgYmx1cmF5X2NvbXBhdD0wIGNvbnN0cmFpbmVkX2ludHJhPTAgYmZyYW1lcz0zIGJfcHlyYW1pZD0yIGJfYWRhcHQ9MSBiX2JpYXM9MCBkaXJlY3Q9MSB3ZWlnaHRiPTEgb3Blbl9nb3A9MCB3ZWlnaHRwPTIga2V5aW50PTI1MCBrZXlpbnRfbWluPTIgc2NlbmVjdXQ9NDAgaW50cmFfcmVmcmVzaD0wIHJjX2xvb2thaGVhZD00MCByYz1jcmYgbWJ0cmVlPTEgY3JmPTIzLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAAoZYiEABX//uzPfgU3IDyL9ZQIdLVudeOY06aFdOh0hhIVsUAt6pJMwwAAAApBmiNsQS/+tSvvAAAACEGeQXiCfwExAAAACAGeYmpBLwF7AAADYm1vb3YAAABsbXZoZAAAAAAAAAAAAAAAAAAAA+gAAAfQAAEAAAEAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAKMdHJhawAAAFx0a2hkAAAAAwAAAAAAAAAAAAAAAQAAAAAAAAfQAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAABAAAAAQAAAAAAAJGVkdHMAAAAcZWxzdAAAAAAAAAABAAAH0AAAQAAAAQAAAAACBG1kaWEAAAAgbWRoZAAAAAAAAAAAAAAAAAAAQAAAAIAAVcQAAAAAAC1oZGxyAAAAAAAAAAB2aWRlAAAAAAAAAAAAAAAAVmlkZW9IYW5kbGVyAAAAAa9taW5mAAAAFHZtaGQAAAABAAAAAAAAAAAAAAAkZGluZgAAABxkcmVmAAAAAAAAAAEAAAAMdXJsIAAAAAEAAAFvc3RibAAAAL9zdHNkAAAAAAAAAAEAAACvYXZjMQAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAABAAEAASAAAAEgAAAAAAAAAARVMYXZjNjAuMzEuMTAyIGxpYngyNjQAAAAAAAAAAAAAABj//wAAADVhdmNDAWQACv/hABhnZAAKrNlEJsBEAAADAAQAAAMAEDxIllgBAAZo6+PLIsD9+PgAAAAAEHBhc3AAAAABAAAAAQAAABRidHJ0AAAAAAAADAwAAAwMAAAAGHN0dHMAAAAAAAAAAQAAAAQAACAAAAAAFHN0c3MAAAAAAAAAAQAAAAEAAAAoY3R0cwAAAAAAAAADAAAAAQAAQAAAAAABAACAAAAAAAIAACAAAAAAHHN0c2MAAAAAAAAAAQAAAAEAAAAEAAAAAQAAACRzdHN6AAAAAAAAAAAAAAAEAAAC3QAAAA4AAAAMAAAADAAAABRzdGNvAAAAAAAAAAEAAAAwAAAAYnVkdGEAAABabWV0YQAAAAAAAAAhaGRscgAAAAAAAAAAbWRpcmFwcGwAAAAAAAAAAAAAAAAtaWxzdAAAACWpdG9vAAAAHWRhdGEAAAABAAAAAExhdmY2MC4xNi4xMDA="
)


def t_vision_image(base_url: str) -> tuple[bool, str]:
    status, resp = chat(base_url, [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What color is this image? Reply with just the color name."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TEST_IMAGE_B64}"}},
        ],
    }], max_tokens=40)
    passed = ok(status, resp) and "red" in content_of(resp).lower()
    return passed, f"status={status} content={content_of(resp) if isinstance(resp, dict) else resp!r}"


def t_vision_multiturn(base_url: str) -> tuple[bool, str]:
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What color is this image? Reply with just the color name."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TEST_IMAGE_B64}"}},
        ],
    }]
    status, resp = chat(base_url, messages, max_tokens=40)
    if not ok(status, resp):
        return False, f"turn 1 failed: status={status} body={resp!r}"
    messages.append({"role": "assistant", "content": content_of(resp)})
    messages.append({"role": "user", "content": "What color did I just show you? One word."})
    # max_tokens=200, not 60: same class of false failure as t_multiturn's
    # recall turn -- reproduced 3/3 truncated at max_tokens=60
    # (`finish_reason=length`, 2026-09-05) on an otherwise-correct model that
    # always resolves to "Red" once given enough room (verified at
    # max_tokens=300); this checkpoint's cross-turn image recall goes through
    # the same long, variable reasoning preamble as the text case.
    status, resp = chat(base_url, messages, max_tokens=200)
    passed = ok(status, resp) and "red" in content_of(resp).lower()
    return passed, f"status={status} content={content_of(resp) if isinstance(resp, dict) else resp!r}"


def t_video(base_url: str) -> tuple[bool, str]:
    status, resp = chat(base_url, [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What color is this video? One word."},
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{TEST_VIDEO_B64}"}},
        ],
    }], max_tokens=40)
    passed = ok(status, resp) and "blue" in content_of(resp).lower()
    return passed, f"status={status} content={content_of(resp) if isinstance(resp, dict) else resp!r}"


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
    ("vision_image", t_vision_image),
    ("vision_multiturn", t_vision_multiturn),
    ("video", t_video),
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
