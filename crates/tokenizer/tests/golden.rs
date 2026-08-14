//! Check the GGUF-derived tokenizer against Hugging Face's output.
//!
//! Fixtures are produced by `scripts/make_tokenizer_fixtures.py`. Skipped when
//! the model file isn't downloaded.

use std::path::PathBuf;

use serde::Deserialize;
use tuili_gguf::Gguf;
use tuili_tokenizer::{ChatMessage, Tokenizer};

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

fn load() -> Option<(Tokenizer, Fixtures)> {
    let model = std::env::var("TUILI_TEST_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace().join("models/qwen2.5-0.5b-instruct-q8_0.gguf"));
    if !model.exists() {
        eprintln!("skipping: {} not downloaded", model.display());
        return None;
    }

    let gguf = Gguf::open(&model).expect("opening model");
    let tok = Tokenizer::from_gguf(&gguf).expect("building tokenizer");

    let fixture_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen2.5-0.5b-instruct.json");
    let raw = std::fs::read_to_string(&fixture_path).expect("reading fixtures");
    let fixtures: Fixtures = serde_json::from_str(&raw).expect("parsing fixtures");

    Some((tok, fixtures))
}

macro_rules! setup {
    () => {
        match load() {
            Some(x) => x,
            None => return,
        }
    };
}

#[test]
fn encodes_identically_to_huggingface() {
    let (tok, fx) = setup!();

    let mut failures = Vec::new();
    for case in &fx.encode {
        // parse_special = false: HF's `encode(add_special_tokens=False)` still
        // splits out added tokens, which is the same behaviour we want here.
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
        "{} of {} encode cases differ:\n{}",
        failures.len(),
        fx.encode.len(),
        failures.join("\n")
    );
}

#[test]
fn decode_roundtrips_every_case() {
    let (tok, fx) = setup!();
    for case in &fx.encode {
        let ids = tok.encode(&case.text, Some(false), true);
        assert_eq!(
            tok.decode(&ids, false),
            case.text,
            "roundtrip {:?}",
            case.text
        );
    }
}

#[test]
fn streaming_decode_matches_batch_decode() {
    let (tok, fx) = setup!();
    for case in &fx.encode {
        let ids = tok.encode(&case.text, Some(false), true);
        let mut de = tok.detokenizer();
        let mut streamed = String::new();
        for &id in &ids {
            streamed.push_str(&de.push(id));
        }
        streamed.push_str(&de.finish());
        assert_eq!(streamed, case.text, "streaming {:?}", case.text);
    }
}

#[test]
fn chat_template_matches_huggingface() {
    let (tok, fx) = setup!();
    let template = tok.chat_template().expect("model has a chat template");

    for case in &fx.chat {
        let msgs: Vec<ChatMessage> = case
            .messages
            .iter()
            .map(|m| ChatMessage::new(&m.role, &m.content))
            .collect();
        let got = template.render(&msgs, true).expect("rendering");
        assert_eq!(got, case.prompt);
    }
}

#[test]
fn vocab_and_special_ids_agree() {
    let (tok, fx) = setup!();
    // GGUF pads the vocab out to the embedding matrix's row count, so it is
    // larger than HF's `len(tokenizer)`. The trailing entries are unused
    // placeholders; every real token must still be there.
    assert!(
        tok.vocab_size() >= fx.vocab_size,
        "gguf vocab {} < hf vocab {}",
        tok.vocab_size(),
        fx.vocab_size
    );
    assert_eq!(tok.eos_id(), Some(fx.eos_token_id));
    assert!(tok.is_eog(fx.eos_token_id));
    assert!(tok.is_special(tok.token_to_id("<|im_start|>").unwrap()));
}

#[test]
fn special_tokens_are_atomic_when_parsed() {
    let (tok, fx) = setup!();
    let _ = fx;
    let im_start = tok.token_to_id("<|im_start|>").unwrap();

    let parsed = tok.encode("<|im_start|>user", Some(false), true);
    assert_eq!(parsed[0], im_start);

    // With parsing off the marker is just text, and must not collapse to the
    // control token — that is what keeps user input from forging a turn.
    let literal = tok.encode("<|im_start|>user", Some(false), false);
    assert!(
        !literal.contains(&im_start),
        "special token leaked: {literal:?}"
    );
    assert_eq!(tok.decode(&literal, false), "<|im_start|>user");
}
