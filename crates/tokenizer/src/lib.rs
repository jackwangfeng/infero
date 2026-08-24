//! Byte-level BPE tokenizer built straight from a GGUF file.
//!
//! No `tokenizer.json` and no Python: the vocabulary, merge list, special-token
//! ids and chat template all live in the model file's metadata, which is the
//! whole point of GGUF being self-contained.

mod bpe;
mod bytelevel;
mod chat;
mod pretokenize;

use std::sync::Mutex;

use anyhow::{Context, Result};
use rustc_hash::{FxHashMap, FxHashSet};
use tuili_gguf::Gguf;

use bpe::Bpe;
use bytelevel::ByteLevel;
use pretokenize::PreTokenizer;

pub use chat::{ChatMessage, ChatTemplate, ContentPart, ToolCall, ToolCallFunction};

/// GGUF token type tags.
mod token_type {
    pub const CONTROL: i64 = 3;
    pub const USER_DEFINED: i64 = 4;
    pub const BYTE: i64 = 6;
}

/// Whether this vocabulary family was trained through an NFC normalizer.
///
/// Inference, and only needed for a GGUF: the format has no normalizer field.
/// Every Qwen `tokenizer.json` declares NFC; GPT-2 and Llama-3 declare none,
/// and adding one there would change tokenisation for text that those models
/// tokenise correctly today.
fn nfc_for(pre_kind: &str) -> bool {
    matches!(pre_kind, "qwen2" | "qwen35")
}

pub struct Tokenizer {
    /// id -> piece, still in the byte-level alphabet.
    pieces: Vec<String>,
    piece_ids: FxHashMap<String, u32>,
    /// Raw byte -> its single-character token id. All of encoding starts here.
    byte_ids: [u32; 256],
    bpe: Bpe,
    bytelevel: ByteLevel,
    pre: PreTokenizer,
    /// Whether to compose the input to NFC before splitting it.
    ///
    /// Both Qwen `tokenizer.json` files declare `"normalizer": {"type": "NFC"}`,
    /// and the model was trained through it, so `cafe` + U+0301 has to become
    /// `café` before the split sees it or the bytes handed to BPE are not the
    /// bytes it has merges for.
    ///
    /// A GGUF does not record this. `tokenizer.ggml.*` has a `pre` and a
    /// `model` and no normalizer field, because llama.cpp does not normalize for
    /// byte-level BPE at all -- so for a GGUF it has to be inferred from the
    /// vocabulary family, which is what `nfc_for` does. Reading it off
    /// `tokenizer.json` when there is one is strictly better and that path does.
    nfc: bool,

    specials: Vec<(String, u32)>,
    special_ids: FxHashSet<u32>,

    bos: Option<u32>,
    eos: Option<u32>,
    pad: Option<u32>,
    eog: FxHashSet<u32>,
    add_bos_by_default: bool,

    chat_template: Option<ChatTemplate>,

    /// Pre-token -> ids. Prompts repeat words heavily and BPE is the hot loop.
    cache: Mutex<FxHashMap<String, Vec<u32>>>,
}

impl Tokenizer {
    pub fn from_gguf(f: &Gguf) -> Result<Self> {
        let model = f.str("tokenizer.ggml.model").unwrap_or("gpt2");
        if model != "gpt2" {
            anyhow::bail!(
                "tokenizer model `{model}` is not supported yet (only byte-level BPE / `gpt2`)"
            );
        }

        let pieces: Vec<String> = f
            .str_array("tokenizer.ggml.tokens")
            .context("model has no tokenizer.ggml.tokens")?
            .to_vec();
        anyhow::ensure!(!pieces.is_empty(), "empty vocabulary");

        let types = f.int_array("tokenizer.ggml.token_type").unwrap_or_default();
        if !types.is_empty() && types.len() != pieces.len() {
            anyhow::bail!(
                "token_type has {} entries but the vocab has {}",
                types.len(),
                pieces.len()
            );
        }

        let borrowed: FxHashMap<&str, u32> = pieces
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i as u32))
            .collect();

        let merges = f.str_array("tokenizer.ggml.merges").unwrap_or_default();
        let bpe = Bpe::new(merges.iter().map(String::as_str), &borrowed);
        anyhow::ensure!(!bpe.is_empty(), "model has no usable BPE merges");

        let bytelevel = ByteLevel::new();
        let mut byte_ids = [u32::MAX; 256];
        let mut missing = Vec::new();
        for b in 0..=255u8 {
            let s = bytelevel.encode(&[b]);
            match borrowed.get(s.as_str()) {
                Some(&id) => byte_ids[b as usize] = id,
                None => missing.push(b),
            }
        }
        if !missing.is_empty() {
            anyhow::bail!(
                "vocabulary is missing single-byte tokens for {} byte values (e.g. {:#04x}); \
                 this is not a byte-level BPE vocab",
                missing.len(),
                missing[0]
            );
        }
        drop(borrowed);

        let mut specials: Vec<(String, u32)> = Vec::new();
        for (id, piece) in pieces.iter().enumerate() {
            let is_special = types
                .get(id)
                .is_some_and(|&t| t == token_type::CONTROL || t == token_type::USER_DEFINED);
            // A byte-fallback token looks like a control token textually but
            // must still take part in normal merging.
            let is_byte = types.get(id).is_some_and(|&t| t == token_type::BYTE);
            if is_special && !is_byte {
                specials.push((piece.clone(), id as u32));
            }
        }
        // Longest first, so `<|im_start|>` wins over a hypothetical `<|im`.
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
        let special_ids: FxHashSet<u32> = specials.iter().map(|(_, id)| *id).collect();

        let piece_ids: FxHashMap<String, u32> = pieces
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();

        let pre = PreTokenizer::new(f.str("tokenizer.ggml.pre").unwrap_or("default"))?;

        let bos = f.u32("tokenizer.ggml.bos_token_id").ok();
        let eos = f.u32("tokenizer.ggml.eos_token_id").ok();
        let pad = f.u32("tokenizer.ggml.padding_token_id").ok();
        let add_bos_by_default = f.bool("tokenizer.ggml.add_bos_token").unwrap_or(false);

        let mut eog: FxHashSet<u32> = eos.into_iter().collect();
        if let Ok(id) = f.u32("tokenizer.ggml.eot_token_id") {
            eog.insert(id);
        }
        // Chat-tuned models stop on the turn marker even when `eos` points at
        // the base model's `<|endoftext|>`.
        for name in [
            "<|im_end|>",
            "<|endoftext|>",
            "<|eot_id|>",
            "<|end_of_text|>",
        ] {
            if let Some(&id) = piece_ids.get(name) {
                eog.insert(id);
            }
        }

        let chat_template = compile_template(f.str("tokenizer.chat_template").ok(), &pieces, bos, eos);

        tracing::info!(
            vocab = pieces.len(),
            merges = bpe.len(),
            specials = specials.len(),
            chat_template = chat_template.is_some(),
            "tokenizer ready"
        );

        Ok(Self {
            pieces,
            piece_ids,
            byte_ids,
            bpe,
            bytelevel,
            nfc: nfc_for(pre.kind()),
            pre,
            specials,
            special_ids,
            bos,
            eos,
            pad,
            eog,
            add_bos_by_default,
            chat_template,
            cache: Mutex::new(FxHashMap::default()),
        })
    }

    /// From a Hugging Face `tokenizer.json` plus its `tokenizer_config.json`,
    /// as an AWQ checkpoint ships them.
    ///
    /// Same byte-level BPE as the GGUF path — the vocabulary and merges are
    /// identical, only the container differs. `added_tokens` carries what GGUF
    /// records as a token type, and the split pattern comes from the
    /// pre-tokenizer's own regex rather than from a name, which is the one
    /// thing a `tokenizer.json` states more precisely than GGUF metadata does.
    pub fn from_hf_dir(dir: impl AsRef<std::path::Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let read = |name: &str| -> Result<serde_json::Value> {
            let path = dir.join(name);
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
        };
        let tj = read("tokenizer.json")?;
        let cj = read("tokenizer_config.json").unwrap_or(serde_json::Value::Null);

        let model = &tj["model"];
        anyhow::ensure!(
            model["type"].as_str().unwrap_or("BPE") == "BPE",
            "tokenizer model `{}` is not supported yet (only byte-level BPE)",
            model["type"].as_str().unwrap_or("?")
        );
        let vocab = model["vocab"]
            .as_object()
            .context("tokenizer.json has no model.vocab")?;

        // `added_tokens` may sit past the end of `vocab`, so size from both.
        let added = tj["added_tokens"].as_array().cloned().unwrap_or_default();
        let n = vocab
            .values()
            .filter_map(|v| v.as_u64())
            .chain(added.iter().filter_map(|t| t["id"].as_u64()))
            .max()
            .context("empty vocabulary")? as usize
            + 1;
        let mut pieces = vec![String::new(); n];
        for (piece, id) in vocab {
            if let Some(i) = id.as_u64() {
                pieces[i as usize] = piece.clone();
            }
        }
        for t in &added {
            if let (Some(i), Some(c)) = (t["id"].as_u64(), t["content"].as_str()) {
                pieces[i as usize] = c.to_string();
            }
        }

        let borrowed: FxHashMap<&str, u32> = pieces
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i as u32))
            .collect();

        // Merges are either "a b" or ["a", "b"], depending on the version that
        // wrote the file.
        let merges: Vec<String> = model["merges"]
            .as_array()
            .context("tokenizer.json has no model.merges")?
            .iter()
            .filter_map(|m| match m {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Array(p) if p.len() == 2 => {
                    Some(format!("{} {}", p[0].as_str()?, p[1].as_str()?))
                }
                _ => None,
            })
            .collect();
        let bpe = Bpe::new(merges.iter().map(String::as_str), &borrowed);
        anyhow::ensure!(!bpe.is_empty(), "tokenizer has no usable BPE merges");

        let bytelevel = ByteLevel::new();
        let mut byte_ids = [u32::MAX; 256];
        let mut missing = Vec::new();
        for b in 0..=255u8 {
            match borrowed.get(bytelevel.encode(&[b]).as_str()) {
                Some(&id) => byte_ids[b as usize] = id,
                None => missing.push(b),
            }
        }
        anyhow::ensure!(
            missing.is_empty(),
            "vocabulary is missing single-byte tokens for {} byte values (e.g. {:#04x}); \
             this is not a byte-level BPE vocab",
            missing.len(),
            missing.first().copied().unwrap_or(0)
        );
        drop(borrowed);

        // Every added token is matched atomically, flagged or not. The `special`
        // flag decides what `decode` may skip, not what `encode` can see — in
        // Hugging Face the whole `added_tokens` list goes into the trie that runs
        // before the BPE. Filtering on the flag here splits Qwen3.5's `<think>`,
        // which ships as `"special": false`, into `<` + `think` + `>`; since its
        // chat template appends `<think>` to every generation prompt, the model
        // then never sees the marker it was tuned to continue from. The GGUF
        // reader already takes `USER_DEFINED` next to `CONTROL` for this reason.
        let mut specials: Vec<(String, u32)> = added
            .iter()
            .filter_map(|t| Some((t["content"].as_str()?.to_string(), t["id"].as_u64()? as u32)))
            .collect();
        // Longest first, so `<|im_start|>` wins over a hypothetical `<|im`.
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
        // Only the flagged ones are "special" to callers.
        let special_ids: FxHashSet<u32> = added
            .iter()
            .filter(|t| t["special"].as_bool().unwrap_or(true))
            .filter_map(|t| t["id"].as_u64())
            .map(|id| id as u32)
            .collect();

        let piece_ids: FxHashMap<String, u32> = pieces
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();

        let pre = PreTokenizer::from_json(&tj["pre_tokenizer"])?;
        // Declared, so no inference needed. Only NFC is recognised: it is what
        // every byte-level BPE checkpoint this reader has seen uses, and a
        // different one silently ignored would be worse than a refusal.
        let declared_nfc = match tj["normalizer"]["type"].as_str() {
            None => None,
            Some("NFC") => Some(true),
            Some(other) => anyhow::bail!(
                "tokenizer.json declares a `{other}` normalizer; this reader \
                 implements NFC only, and applying the wrong one changes the \
                 bytes BPE merges over"
            ),
        };

        // `tokenizer_config.json` names its special tokens; look the strings up.
        let id_of = |v: &serde_json::Value| -> Option<u32> {
            let s = match v {
                serde_json::Value::String(s) => s.as_str(),
                other => other["content"].as_str()?,
            };
            piece_ids.get(s).copied()
        };
        let bos = id_of(&cj["bos_token"]);
        let eos = id_of(&cj["eos_token"]);
        let pad = id_of(&cj["pad_token"]);
        let add_bos_by_default = cj["add_bos_token"].as_bool().unwrap_or(bos.is_some());

        let mut eog: FxHashSet<u32> = eos.into_iter().collect();
        // Chat-tuned models stop on the turn marker even when `eos` points at
        // the base model's `<|end_of_text|>`.
        for name in [
            "<|im_end|>",
            "<|endoftext|>",
            "<|eot_id|>",
            "<|end_of_text|>",
        ] {
            if let Some(&id) = piece_ids.get(name) {
                eog.insert(id);
            }
        }

        let chat_template = compile_template(cj["chat_template"].as_str(), &pieces, bos, eos);

        tracing::info!(
            vocab = pieces.len(),
            merges = bpe.len(),
            specials = specials.len(),
            chat_template = chat_template.is_some(),
            "tokenizer ready"
        );

        Ok(Self {
            pieces,
            piece_ids,
            byte_ids,
            bpe,
            bytelevel,
            nfc: declared_nfc.unwrap_or_else(|| nfc_for(pre.kind())),
            pre,
            specials,
            special_ids,
            bos,
            eos,
            pad,
            eog,
            add_bos_by_default,
            chat_template,
            cache: Mutex::new(FxHashMap::default()),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// Which pre-tokenizer split this vocabulary uses (`"qwen2"`, `"gpt2"`, …).
    pub fn pretokenizer(&self) -> &'static str {
        self.pre.kind()
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.bos
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.eos
    }

    pub fn pad_id(&self) -> Option<u32> {
        self.pad
    }

    /// True for any token that should stop generation.
    pub fn is_eog(&self, id: u32) -> bool {
        self.eog.contains(&id)
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special_ids.contains(&id)
    }

    pub fn token_to_id(&self, piece: &str) -> Option<u32> {
        self.piece_ids.get(piece).copied()
    }

    /// The raw vocabulary entry, in the byte-level alphabet. Use [`decode`] to
    /// get readable text.
    pub fn id_to_piece(&self, id: u32) -> Option<&str> {
        self.pieces.get(id as usize).map(String::as_str)
    }

    pub fn chat_template(&self) -> Option<&ChatTemplate> {
        self.chat_template.as_ref()
    }

    /// Tokenize `text`.
    ///
    /// `add_bos` follows the model's own `add_bos_token` flag when `None`.
    /// `parse_special` controls whether markers like `<|im_start|>` in the
    /// input become their single control token (what a chat template needs) or
    /// are tokenized as ordinary text (what untrusted user text wants).
    pub fn encode(&self, text: &str, add_bos: Option<bool>, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 8);

        if add_bos.unwrap_or(self.add_bos_by_default)
            && let Some(bos) = self.bos
        {
            out.push(bos);
        }

        if parse_special {
            let mut rest = text;
            while !rest.is_empty() {
                match self.next_special(rest) {
                    Some((at, len, id)) => {
                        self.encode_ordinary(&rest[..at], &mut out);
                        out.push(id);
                        rest = &rest[at + len..];
                    }
                    None => {
                        self.encode_ordinary(rest, &mut out);
                        break;
                    }
                }
            }
        } else {
            self.encode_ordinary(text, &mut out);
        }

        out
    }

    /// Earliest special token in `text`, as (byte offset, byte length, id).
    ///
    /// Linear in the number of special tokens; there are a few dozen, and
    /// `find` on each is memchr-fast, so this beats building a combined regex.
    fn next_special(&self, text: &str) -> Option<(usize, usize, u32)> {
        let mut best: Option<(usize, usize, u32)> = None;
        for (piece, id) in &self.specials {
            if let Some(at) = text.find(piece.as_str()) {
                // Ties go to the longer match; `specials` is length-sorted so
                // the first hit at a given offset already is the longest.
                let better = match best {
                    None => true,
                    Some((b_at, b_len, _)) => at < b_at || (at == b_at && piece.len() > b_len),
                };
                if better {
                    best = Some((at, piece.len(), *id));
                }
                if at == 0 && best.is_some_and(|(b, _, _)| b == 0) {
                    // Can't do better than a longest match at offset 0.
                    break;
                }
            }
        }
        best
    }

    fn encode_ordinary(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        // Per ordinary segment rather than over the whole input, which is where
        // Hugging Face applies it too: added tokens are matched on the raw text
        // first and only the text between them is normalised. Normalising first
        // would let a composition change the bytes of a marker.
        //
        // `is_nfc_quick` is a table lookup per character and says Yes for all
        // ASCII, so text that is already composed -- nearly all of it -- pays
        // nothing and never allocates.
        if self.nfc && !matches!(unicode_normalization::is_nfc_quick(text.chars()), unicode_normalization::IsNormalized::Yes) {
            let composed: String = unicode_normalization::UnicodeNormalization::nfc(text.chars()).collect();
            for piece in self.pre.split(&composed) {
                self.encode_pretoken(piece, out);
            }
            return;
        }
        for piece in self.pre.split(text) {
            self.encode_pretoken(piece, out);
        }
    }

    fn encode_pretoken(&self, piece: &str, out: &mut Vec<u32>) {
        if piece.is_empty() {
            return;
        }

        // Whole pre-token already a token? Very common for short words.
        let alphabet = self.bytelevel.encode(piece.as_bytes());
        if let Some(&id) = self.piece_ids.get(&alphabet) {
            out.push(id);
            return;
        }

        if let Some(hit) = self.cache.lock().unwrap().get(&alphabet) {
            out.extend_from_slice(hit);
            return;
        }

        let mut ids: Vec<u32> = piece
            .as_bytes()
            .iter()
            .map(|&b| self.byte_ids[b as usize])
            .collect();
        self.bpe.merge(&mut ids);

        out.extend_from_slice(&ids);
        // Unbounded growth would be a leak on adversarial input; long
        // pre-tokens are also the ones least likely to repeat.
        if alphabet.len() <= 128 {
            self.cache.lock().unwrap().insert(alphabet, ids);
        }
    }

    /// Turn ids back into text.
    ///
    /// Decoding is byte-oriented: a token can end mid-codepoint (CJK text
    /// routinely does), so the bytes are accumulated and only then interpreted
    /// as UTF-8. Streaming callers should use [`Detokenizer`] instead.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes = Vec::with_capacity(ids.len() * 3);
        for &id in ids {
            if skip_special && self.special_ids.contains(&id) {
                continue;
            }
            if let Some(piece) = self.pieces.get(id as usize) {
                self.bytelevel.decode_into(piece, &mut bytes);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Bytes for one token, without the UTF-8 interpretation.
    pub fn token_bytes(&self, id: u32, out: &mut Vec<u8>) {
        if let Some(piece) = self.pieces.get(id as usize) {
            self.bytelevel.decode_into(piece, out);
        }
    }

    /// A streaming decoder for this tokenizer.
    pub fn detokenizer(&self) -> Detokenizer<'_> {
        Detokenizer {
            tok: self,
            pending: Vec::new(),
        }
    }

    /// Streaming decode against caller-owned state.
    ///
    /// [`Detokenizer`] borrows the tokenizer, which a scheduler holding many
    /// concurrent sequences behind an `Arc` cannot do. These two let it keep
    /// the byte buffer itself.
    pub fn stream_push(&self, id: u32, pending: &mut Vec<u8>) -> String {
        self.token_bytes(id, pending);
        take_complete(pending)
    }

    /// Flush a trailing partial sequence at end of generation.
    pub fn stream_finish(&self, pending: &mut Vec<u8>) -> String {
        if pending.is_empty() {
            return String::new();
        }
        let s = String::from_utf8_lossy(pending).into_owned();
        pending.clear();
        s
    }
}

/// Emit whatever prefix of `pending` is now valid UTF-8, keeping the rest.
fn take_complete(pending: &mut Vec<u8>) -> String {
    match std::str::from_utf8(pending) {
        Ok(s) => {
            let s = s.to_string();
            pending.clear();
            s
        }
        Err(e) => {
            let good = e.valid_up_to();
            if good == 0 && e.error_len().is_some() {
                // Genuinely invalid, not a truncation: emit a replacement
                // char rather than stalling the stream forever.
                let bad = e.error_len().unwrap();
                pending.drain(..bad);
                return "\u{fffd}".to_string();
            }
            let s = String::from_utf8_lossy(&pending[..good]).into_owned();
            pending.drain(..good);
            s
        }
    }
}

/// Emits text as tokens arrive, holding back bytes that would split a
/// multi-byte character.
pub struct Detokenizer<'a> {
    tok: &'a Tokenizer,
    pending: Vec<u8>,
}

impl Detokenizer<'_> {
    /// Feed one token; returns whatever text is now complete (often empty).
    pub fn push(&mut self, id: u32) -> String {
        self.tok.stream_push(id, &mut self.pending)
    }

    /// Flush any trailing partial sequence at end of generation.
    pub fn finish(&mut self) -> String {
        self.tok.stream_finish(&mut self.pending)
    }
}

/// Compile a chat template, warning rather than failing when it does not.
///
/// A model that cannot render a chat turn still serves `/v1/completions`, so a
/// broken template is a degraded server rather than no server.
fn compile_template(
    src: Option<&str>,
    pieces: &[String],
    bos: Option<u32>,
    eos: Option<u32>,
) -> Option<ChatTemplate> {
    let text = |id: Option<u32>| -> String {
        id.and_then(|i| pieces.get(i as usize))
            .cloned()
            .unwrap_or_default()
    };
    match ChatTemplate::with_tokens(src?, &text(bos), &text(eos)) {
        Ok(t) => Some(t),
        Err(e) => {
            tracing::warn!(error = %e, "chat template failed to compile; /v1/chat will not work");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal byte-level BPE `tokenizer.json`: one token per byte value, plus
    /// two added tokens — one flagged `special`, one not.
    ///
    /// Qwen3.5 ships `<think>` and `</think>` as the second kind, and its chat
    /// template appends `<think>` to every generation prompt, so whether an
    /// unflagged added token survives encoding decides whether the model is
    /// prompted with the marker it was trained on or with three ordinary pieces.
    fn synthetic_hf_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("tuili-tok-{name}"));
        std::fs::create_dir_all(&dir).unwrap();
        let bl = ByteLevel::new();
        let mut vocab = serde_json::Map::new();
        for b in 0u8..=255 {
            vocab.insert(bl.encode(&[b]), serde_json::json!(b as u64));
        }
        // One real merge, so the BPE is non-empty; `th` is what `<think>` would
        // fall apart into if the added token were missed.
        vocab.insert("th".into(), serde_json::json!(258u64));
        let tj = serde_json::json!({
            "model": { "type": "BPE", "vocab": vocab, "merges": [["t", "h"]] },
            "added_tokens": [
                { "id": 256, "content": "<|marked|>", "special": true },
                { "id": 257, "content": "<think>", "special": false },
            ],
        });
        std::fs::write(dir.join("tokenizer.json"), tj.to_string()).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
        dir
    }

    /// Hugging Face matches every entry of `added_tokens` atomically; the
    /// `special` flag says whether `decode` may skip it, not whether `encode`
    /// sees it. Reading the flag as "is this a token at all" splits `<think>`
    /// into `<`, `think`, `>` and feeds the model a prompt it was never tuned
    /// on. The GGUF reader already takes `USER_DEFINED` alongside `CONTROL`,
    /// so this is also the two readers agreeing.
    #[test]
    fn an_added_token_is_one_id_even_when_it_is_not_flagged_special() {
        let dir = synthetic_hf_dir("added-unflagged");
        let tok = Tokenizer::from_hf_dir(&dir).unwrap();

        assert_eq!(
            tok.encode("<|marked|>", Some(false), true),
            vec![256],
            "a flagged added token should be its own id"
        );
        assert_eq!(
            tok.encode("<think>", Some(false), true),
            vec![257],
            "an unflagged added token is still one token to Hugging Face"
        );
        // And the flag still decides what `is_special` reports, which is what
        // callers use to hide markers rather than to tokenize them.
        assert!(tok.is_special(256));
        assert!(
            !tok.is_special(257),
            "`special: false` must stay unflagged even though it now encodes atomically"
        );
    }
}
