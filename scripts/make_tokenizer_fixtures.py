#!/usr/bin/env python3
"""Generate golden tokenizer fixtures from Hugging Face.

Our tokenizer is built from the GGUF vocab, not from tokenizer.json, so the
only way to know it agrees with the reference implementation is to check it
against one. This writes the expected ids to a JSON file that the Rust test
suite asserts against; it is not run during `cargo test`.

    python scripts/make_tokenizer_fixtures.py Qwen/Qwen2.5-0.5B-Instruct
"""

import json
import pathlib
import sys

from transformers import AutoTokenizer

CASES = [
    "Hello world",
    "hello",
    " hello",
    "Hello, world!",
    "don't stop believin'",
    "2024-08-12",
    "The quick brown fox jumps over the lazy dog.",
    "你好，世界！",
    "日本語のテキストもトークン化する",
    "emoji: 🦀 rust 🚀 ship",
    "def main():\n    print('hi')\n",
    "a  b   c",
    "\n\n\n",
    "   leading spaces",
    "trailing   ",
    "CamelCaseIdentifier_and_snake_case",
    "1234567890",
    "<|im_start|>user\nhi<|im_end|>\n",
    "mixed 中文 and English 混合 text",
    "x" * 200,
    "$1,234.56 (USD) — 50% off!",
    "https://example.com/path?a=1&b=2#frag",
    "\t\ttabbed\tindent",
    "ünïcödé àccénts",
    "",
]

CHATS = [
    [{"role": "user", "content": "hi"}],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ],
]


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
    out_path = pathlib.Path(__file__).resolve().parent.parent / (
        "crates/tokenizer/tests/fixtures/qwen2.5-0.5b-instruct.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model)

    fixtures = {
        "model": model,
        "vocab_size": len(tok),
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
        # add_special_tokens=False: GGUF models carry their own bos policy and
        # the chat template already emits every marker it wants.
        "encode": [
            {"text": t, "ids": tok.encode(t, add_special_tokens=False)} for t in CASES
        ],
        "chat": [
            {
                "messages": msgs,
                "prompt": tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                ),
            }
            for msgs in CHATS
        ],
    }

    out_path.write_text(json.dumps(fixtures, ensure_ascii=False, indent=1))
    print(f"wrote {out_path} ({len(CASES)} encode cases, {len(CHATS)} chat cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
