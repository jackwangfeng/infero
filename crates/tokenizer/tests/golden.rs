//! Check the GGUF-derived tokenizer against Hugging Face's output.
//!
//! Fixtures are produced by `scripts/make_tokenizer_fixtures.py`. Skipped when
//! the model file isn't downloaded.

use std::path::PathBuf;

use serde::Deserialize;
use infero_gguf::Gguf;
use infero_tokenizer::{ChatMessage, Tokenizer};

#[derive(Deserialize)]
struct Fixtures {
    vocab_size: usize,
    eos_token_id: u32,
    encode: Vec<EncodeCase>,
    chat: Vec<ChatCase>,
}

#[derive(Deserialize)]
struct EncodeCase {
    text: String,
    ids: Vec<u32>,
    /// What the reference decodes these ids back to, which is not always
    /// `text`: the NFC normalizer composes, so a decomposed accent comes back
    /// composed. Asserting the round trip against `text` would be asserting
    /// that the tokenizer does *not* normalise.
    decoded: String,
}

#[derive(Deserialize)]
struct ChatCase {
    messages: Vec<Msg>,
    prompt: String,
}

#[derive(Deserialize)]
struct Msg {
    role: String,
    content: String,
}

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Every checkpoint with both a GGUF and a fixture, paired.
///
/// Two rather than one, and the second is not redundant: Qwen2.5 and Qwen3.8
/// differ in their pre-tokenizer by exactly one thing -- whether a combining
/// mark counts as a letter -- so running the same cases through both is what
/// distinguishes "the split is right" from "the split is close enough for
/// English". `tokenizer.ggml.pre` is `qwen2` in one file and `qwen35` in the
/// other, and until the second name existed here it fell through to the GPT-2
/// split.
const MODELS: &[(&str, &str)] = &[
    (
        "models/qwen2.5-0.5b-instruct-fp16.gguf",
        "qwen2.5-0.5b-instruct",
    ),
    ("models/Qwen3.8-27B-Q4_K_M.gguf", "qwen3.8-27b"),
];

fn workspace_models() -> Vec<(String, Tokenizer, Fixtures)> {
    // A single override still works, and then it is paired with whichever
    // fixture the caller names.
    if let Ok(p) = std::env::var("INFERO_TEST_GGUF") {
        let name = std::env::var("INFERO_TEST_FIXTURE")
            .unwrap_or_else(|_| "qwen2.5-0.5b-instruct".to_string());
        return open(&PathBuf::from(p), &name).into_iter().collect();
    }
    MODELS
        .iter()
        .filter_map(|(gguf, fixture)| open(&workspace().join(gguf), fixture))
        .collect()
}

fn open(model: &PathBuf, fixture: &str) -> Option<(String, Tokenizer, Fixtures)> {
    if !model.exists() {
        eprintln!("skipping: {} not downloaded", model.display());
        return None;
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("{fixture}.json"));
    let raw = std::fs::read_to_string(&path).expect("reading fixtures");
    let fixtures: Fixtures = serde_json::from_str(&raw).expect("parsing fixtures");
    let gguf = Gguf::open(model).expect("opening model");
    let tok = Tokenizer::from_gguf(&gguf).expect("building tokenizer");
    Some((fixture.to_string(), tok, fixtures))
}

#[test]
fn encodes_identically_to_huggingface() {
    for (name, tok, fx) in workspace_models() {
        let mut failures = Vec::new();
        for case in &fx.encode {
            // parse_special = false: HF's `encode(add_special_tokens=False)`
            // still splits out added tokens, which is the behaviour we want.
            let got = tok.encode(&case.text, Some(false), true);
            if got != case.ids {
                failures.push(format!(
                    "  {:?}\n    want {:?}\n    got  {:?}",
                    case.text, case.ids, got
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "{name}: {} of {} encode cases differ:\n{}",
            failures.len(),
            fx.encode.len(),
            failures.join("\n")
        );
    }
}

#[test]
fn decode_roundtrips_every_case() {
    for (name, tok, fx) in workspace_models() {
        for case in &fx.encode {
            let ids = tok.encode(&case.text, Some(false), true);
            assert_eq!(tok.decode(&ids, false), case.decoded, "{name} roundtrip {:?}", case.text);
        }
    }
}

#[test]
fn streaming_decode_matches_batch_decode() {
    for (name, tok, fx) in workspace_models() {
        for case in &fx.encode {
            let ids = tok.encode(&case.text, Some(false), true);
            let mut de = tok.detokenizer();
            let mut streamed = String::new();
            for &id in &ids {
                streamed.push_str(&de.push(id));
            }
            streamed.push_str(&de.finish());
            assert_eq!(streamed, case.decoded, "{name} streaming {:?}", case.text);
        }
    }
}

#[test]
fn chat_template_matches_huggingface() {
    for (name, tok, fx) in workspace_models() {
        let template = tok.chat_template().expect("model has a chat template");
        for case in &fx.chat {
            let msgs: Vec<ChatMessage> = case
                .messages
                .iter()
                .map(|m| ChatMessage::new(&m.role, &m.content))
                .collect();
            let got = template.render(&msgs, true).expect("rendering");
            assert_eq!(got, case.prompt, "{name} chat template");
        }
    }
}

#[test]
fn vocab_and_special_ids_agree() {
    for (name, tok, fx) in workspace_models() {
        // GGUF pads the vocab out to the embedding matrix's row count, so it is
        // larger than HF's `len(tokenizer)`. The trailing entries are unused
        // placeholders; every real token must still be there.
        assert!(
            tok.vocab_size() >= fx.vocab_size,
            "{name}: gguf vocab {} < hf vocab {}",
            tok.vocab_size(),
            fx.vocab_size
        );
        assert_eq!(tok.eos_id(), Some(fx.eos_token_id), "{name} eos");
        assert!(tok.is_eog(fx.eos_token_id), "{name} eog");
        assert!(tok.is_special(tok.token_to_id("<|im_start|>").unwrap()), "{name}");
    }
}

#[test]
fn special_tokens_are_atomic_when_parsed() {
    for (name, tok, _fx) in workspace_models() {
        let im_start = tok.token_to_id("<|im_start|>").unwrap();
        let parsed = tok.encode("<|im_start|>user", Some(false), true);
        assert_eq!(parsed[0], im_start, "{name}");

        // With parsing off the marker is just text, and must not collapse to
        // the control token -- that is what keeps user input from forging a
        // turn.
        let literal = tok.encode("<|im_start|>user", Some(false), false);
        assert!(
            !literal.contains(&im_start),
            "{name}: special token leaked: {literal:?}"
        );
        assert_eq!(tok.decode(&literal, false), "<|im_start|>user", "{name}");
    }
}
