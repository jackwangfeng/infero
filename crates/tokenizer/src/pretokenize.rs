//! Pre-tokenization: the regex split that runs before BPE.
//!
//! Which pattern a model wants is recorded in `tokenizer.ggml.pre`. Getting it
//! wrong doesn't crash — it silently produces a different token sequence than
//! the model was trained on, so the patterns here are transcribed from
//! llama.cpp's table rather than approximated.

use anyhow::{Context, Result};
use fancy_regex::Regex;

const GPT2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

const QWEN2: &str = concat!(
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

// Qwen3.5 / Qwen3.8, read off `tokenizer.json` in `Qwen/Qwen3.8-27B` rather
// than guessed from the family name.
//
// `QWEN2` with one change: combining marks count as letters. A decomposed
// accent, Devanagari matras, Arabic and Hebrew diacritics, Thai vowel signs --
// under `QWEN2` each of those is a `\p{M}` that falls to the punctuation
// alternative and cuts the word it belongs to in half.
//
// llama.cpp writes `qwen35` into `tokenizer.ggml.pre` for these checkpoints,
// which matched nothing here, and the unknown-name fallback is `GPT2`. That is
// the wrong neighbour to fall back to: `GPT2` takes digits in runs where every
// Qwen takes them one at a time, and only allows a *space* before a word where
// Qwen allows any single non-letter -- so `("hello` chunks differently. This is
// what the 27B has been serving with.
const QWEN35: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+",
    r"|\p{N}",
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

const LLAMA3: &str = concat!(
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

// The same split as `LLAMA3`, spelled the way `tokenizer.json` spells it: one
// case-insensitive group instead of the expanded alternation llama.cpp writes.
// Recognising it keeps a Hugging Face checkpoint on the shared, tested pattern
// rather than on the fallback that compiles whatever the file contains.
const LLAMA3_HF: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

pub struct PreTokenizer {
    re: Regex,
    kind: &'static str,
}

impl PreTokenizer {
    pub fn new(pre: &str) -> Result<Self> {
        Self::from_name(pre)
    }

    /// From a `tokenizer.json` `pre_tokenizer` node.
    ///
    /// The node is normally a `Sequence` whose first `Split` carries the exact
    /// regex the model was trained with, which is more specific than the name
    /// GGUF records. Match on that pattern and fall back to the name-based
    /// table when it is something else.
    pub fn from_json(node: &serde_json::Value) -> Result<Self> {
        fn find_split(v: &serde_json::Value) -> Option<&str> {
            if v["type"] == "Split" {
                return v["pattern"]["Regex"].as_str();
            }
            v["pretokenizers"]
                .as_array()?
                .iter()
                .find_map(find_split)
        }
        let name = match find_split(node) {
            Some(p) if p == LLAMA3 || p == LLAMA3_HF => "llama3",
            Some(p) if p == QWEN35 => "qwen35",
            Some(p) if p == QWEN2 => "qwen2",
            Some(p) if p == GPT2 => "gpt2",
            Some(p) => {
                tracing::warn!(
                    pattern = p,
                    "unrecognised pre-tokenizer regex; using it directly"
                );
                let re = Regex::new(p).context("compiling the model's own split pattern")?;
                return Ok(Self { kind: "custom", re });
            }
            None => "default",
        };
        Self::from_name(name)
    }

    fn from_name(pre: &str) -> Result<Self> {
        let (kind, pattern) = match pre {
            "qwen35" | "qwen3" => ("qwen35", QWEN35),
            "qwen2" => ("qwen2", QWEN2),
            "llama3" | "llama-v3" | "llama-bpe" => ("llama3", LLAMA3),
            "default" | "gpt-2" | "gpt2" | "olmo" | "jais" => ("gpt2", GPT2),
            other => {
                // Falling back is better than refusing to load, but the token
                // stream may not match the model's training, so say so loudly.
                tracing::warn!(
                    pre = other,
                    "unknown tokenizer.ggml.pre; falling back to the gpt2 split"
                );
                ("gpt2", GPT2)
            }
        };
        let re = Regex::new(pattern)
            .with_context(|| format!("compiling the {kind} pre-tokenizer pattern"))?;
        Ok(Self { re, kind })
    }

    pub fn kind(&self) -> &'static str {
        self.kind
    }

    /// Split `text` into pre-tokens.
    ///
    /// The pattern is total over any input in practice; if a byte range somehow
    /// matches nothing, it is emitted as its own piece so no input is dropped.
    pub fn split<'t>(&self, text: &'t str) -> Vec<&'t str> {
        let mut out = Vec::with_capacity(text.len() / 4 + 1);
        let mut cursor = 0usize;

        for m in self.re.find_iter(text) {
            let Ok(m) = m else { break };
            if m.start() > cursor {
                out.push(&text[cursor..m.start()]);
            }
            if !m.as_str().is_empty() {
                out.push(m.as_str());
            }
            cursor = m.end();
        }
        if cursor < text.len() {
            out.push(&text[cursor..]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen2_splits_like_llama_cpp() {
        let p = PreTokenizer::new("qwen2").unwrap();
        assert_eq!(p.split("Hello world"), vec!["Hello", " world"]);
        assert_eq!(p.split("don't"), vec!["don", "'t"]);
        // qwen2 emits digits one at a time.
        assert_eq!(p.split("2024"), vec!["2", "0", "2", "4"]);
        assert_eq!(p.split("a\n\nb"), vec!["a", "\n\n", "b"]);
    }

    #[test]
    fn llama3_groups_digits_in_threes() {
        let p = PreTokenizer::new("llama3").unwrap();
        assert_eq!(p.split("2024"), vec!["202", "4"]);
    }

    #[test]
    fn nothing_is_dropped() {
        let p = PreTokenizer::new("qwen2").unwrap();
        for text in ["", " ", "  \t ", "你好，世界", "🦀🦀", "a  b", "!!!???"] {
            assert_eq!(
                p.split(text).concat(),
                text,
                "round trip failed for {text:?}"
            );
        }
    }

    #[test]
    fn unknown_pre_falls_back() {
        assert_eq!(PreTokenizer::new("something-new").unwrap().kind(), "gpt2");
    }
}
