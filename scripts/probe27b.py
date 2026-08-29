#!/usr/bin/env python3
"""Decode speed and multi-turn continuity against a running infero server.

The multi-turn check is the one that matters for this model: 48 of its 64
blocks carry recurrent state that is overwritten in place, so a second turn
that cannot recall the first would mean the state is not surviving between
requests — and that failure looks like a model with no memory rather than like
a crash.
"""

import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8098/v1/chat/completions"


def gen(messages, max_tokens=160, temperature=0.7):
    body = json.dumps(
        {
            "model": "q",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    start = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=900))
    elapsed = time.time() - start
    return d["choices"][0]["message"].get("content") or "", d["usage"], elapsed


def main():
    print("=== decode speed, three samples ===")
    for _ in range(3):
        _, u, el = gen([{"role": "user", "content": "数到二十，用逗号分隔。"}], 160)
        out = u["completion_tokens"]
        rate = out / el if el else 0.0
        print(f"  {out:>4} tok / {el:5.1f}s = {rate:5.1f} tok/s   prompt {u['prompt_tokens']}")

    print()
    print("=== multi-turn: the second turn has to recall the first ===")
    history = [{"role": "user", "content": "记住这个数字：73942。只回复 OK。"}]
    first, _, _ = gen(history, 60)
    print("  turn 1:", first.strip()[:100].replace("\n", " "))
    history += [
        {"role": "assistant", "content": first},
        {"role": "user", "content": "我刚让你记的数字是多少？"},
    ]
    second, _, _ = gen(history, 80)
    print("  turn 2:", second.strip()[:160].replace("\n", " "))
    ok = "73942" in second
    print("  recalled:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
