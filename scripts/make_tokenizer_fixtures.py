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

try:
    from transformers import AutoTokenizer
except Exception as e:  # torch missing or broken -- fall back to `tokenizers`
    AutoTokenizer = None
    _TRANSFORMERS_ERR = e

# `tokenizers` is the reference implementation that `transformers` wraps: it
# reads `tokenizer.json` and applies the same pre-tokenizer and merges. It does
# not render chat templates, so the fallback writes no `chat` cases -- which is
# fine for the question fixtures usually exist to answer, whether the split and
# the merge order agree.
from tokenizers import Tokenizer as HfTokenizer  # noqa: E402

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
    # Cases that tell the Qwen splits apart from the GPT-2 one they fall back to
    # when `tokenizer.ggml.pre` is a name this reader does not know.
    #
    # GPT-2 allows only a *space* before a word (` ?\p{L}+`); every Qwen allows
    # any single non-letter non-digit, so the quote or bracket joins the word.
    '("hello") [world] {and} <more>',
    "don't--stop, really·now",
    # GPT-2 takes digit runs, Qwen takes one digit at a time.
    "v3.8 has 248320 tokens and 27000000000 parameters",
    # Combining marks: `\p{M}` counts as a letter in the Qwen3.5 split and as
    # punctuation in Qwen2's, which cuts these words in half.
    "cafe\u0301 nai\u0308ve re\u0301sume\u0301",          # NFD Latin
    "\u0939\u093f\u0928\u094d\u0926\u0940 \u092d\u093e\u0937\u093e",  # Devanagari with matras
    "\u0627\u0644\u0639\u064e\u0631\u064e\u0628\u0650\u064a\u064e\u0651\u0629",  # Arabic with harakat
    "\u0e40\u0e01\u0e34\u0e14 \u0e02\u0e49\u0e2d\u0e04\u0e27\u0e32\u0e21",        # Thai with vowel signs
    "\u05e9\u05b8\u05dc\u05d5\u05b9\u05dd",                                  # Hebrew with niqqud
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
    # Named after the model rather than fixed, so a second checkpoint's fixtures
    # do not overwrite the first's.
    name = sys.argv[2] if len(sys.argv) > 2 else model.split("/")[-1].lower()
    out_path = pathlib.Path(__file__).resolve().parent.parent / (
        f"crates/tokenizer/tests/fixtures/{name}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if AutoTokenizer is not None:
        tok = AutoTokenizer.from_pretrained(model)
        vocab_size = len(tok)
        bos, eos = tok.bos_token_id, tok.eos_token_id
        encode = lambda t: tok.encode(t, add_special_tokens=False)
        decode = lambda ids: tok.decode(ids, skip_special_tokens=False)
        chat = [
            {
                "messages": m,
                "prompt": tok.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True
                ),
            }
            for m in CHATS
        ]
    else:
        print(f"transformers unavailable ({_TRANSFORMERS_ERR}); using `tokenizers`")
        path = pathlib.Path(model)
        path = path / "tokenizer.json" if path.is_dir() else path
        raw = HfTokenizer.from_file(str(path))
        cfg_path = pathlib.Path(model) / "tokenizer_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        vocab_size = raw.get_vocab_size(with_added_tokens=True)
        specials = {t: i for i, t in raw.get_vocab().items()} if False else {}
        del specials
        v = raw.get_vocab()
        bos = v.get(cfg.get("bos_token") or "", None)
        eos = v.get(
            cfg.get("eos_token")["content"]
            if isinstance(cfg.get("eos_token"), dict)
            else (cfg.get("eos_token") or ""),
            None,
        )
        encode = lambda t: raw.encode(t, add_special_tokens=False).ids
        decode = lambda ids: raw.decode(ids, skip_special_tokens=False)
        # The chat template is a Jinja string in `tokenizer_config.json`, which
        # is what `apply_chat_template` renders. Rendering it directly keeps the
        # fallback's fixtures complete rather than quietly dropping the chat
        # cases and leaving that test passing over an empty list.
        tmpl = cfg.get("chat_template")
        if tmpl:
            import jinja2

            env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
            env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(
                RuntimeError(m)
            )
            j = env.from_string(tmpl)
            chat = [
                {
                    "messages": m,
                    "prompt": j.render(messages=m, add_generation_prompt=True),
                }
                for m in CHATS
            ]
        else:
            chat = []

    fixtures = {
        "model": model,
        "vocab_size": vocab_size,
        "bos_token_id": bos,
        "eos_token_id": eos,
        # add_special_tokens=False: GGUF models carry their own bos policy and
        # the chat template already emits every marker it wants.
        # `decoded` is what the reference gives back, which is not always the
        # input: the NFC normalizer composes, so a decomposed accent encodes to
        # the composed form's ids and decodes to the composed form. The
        # round-trip tests assert against this rather than against `text`.
        "encode": [
            {"text": t, "ids": encode(t), "decoded": decode(encode(t))} for t in CASES
        ],
        "chat": chat,
    }

    out_path.write_text(json.dumps(fixtures, ensure_ascii=False, indent=1))
    print(f"wrote {out_path} ({len(CASES)} encode cases, {len(CHATS)} chat cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
