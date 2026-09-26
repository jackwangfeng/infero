//! `docs/keel-integration.md`'s "2. Schema-constrained decoding" section,
//! run against a real small GGUF model: the output must be genuinely
//! parseable JSON matching the schema's required fields and types, not just
//! "usually looks like JSON" -- checked with `serde_json::from_str`, not a
//! human reading the completion.

use std::path::PathBuf;

use anyhow::Result;
use infero_cuda::Device;
use infero_gguf::Gguf;
use infero_model::json_grammar::Schema;
use infero_model::{Model, SamplingParams};
use infero_tokenizer::{ChatMessage, Tokenizer};

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn model_path() -> Option<PathBuf> {
    let p = std::env::var("INFERO_TEST_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace().join("models/qwen2.5-0.5b-instruct-q8_0.gguf"));
    p.exists().then_some(p)
}

fn vocab_bytes(tokenizer: &Tokenizer) -> Vec<Vec<u8>> {
    let mut buf = Vec::new();
    (0..tokenizer.vocab_size() as u32)
        .map(|id| {
            buf.clear();
            tokenizer.token_bytes(id, &mut buf);
            buf.clone()
        })
        .collect()
}

fn generate(schema_json: &str, prompt: &str, max_tokens: usize) -> Option<Result<serde_json::Value>> {
    let path = model_path()?;
    eprintln!("(if this panics with a missing file, set INFERO_TEST_GGUF or place a model at {})", path.display());
    let gguf = Gguf::open(&path).expect("opening the gguf");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("building tokenizer");
    let mut model = Model::load(Device::new(0).expect("cuda device"), &gguf, 2048).expect("loading model");

    let template = tokenizer.chat_template().expect("model has a chat template");
    let rendered = template.render(&[ChatMessage::user(prompt)], true).expect("rendering chat template");
    let prompt_tokens = tokenizer.encode(&rendered, Some(false), true);

    let schema = Schema::parse(&serde_json::from_str(schema_json).unwrap()).expect("parsing schema");
    let bytes = vocab_bytes(&tokenizer);
    let ids = infero_model::constrained::generate_json(
        &mut model,
        &bytes,
        &prompt_tokens,
        schema,
        max_tokens,
        SamplingParams::greedy(),
    );
    Some(ids.map(|ids| {
        let text = tokenizer.decode(&ids, true);
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("output was not valid JSON: {e}\n{text:?}"))
    }))
}

#[test]
fn extracted_fields_are_present_and_correctly_typed() {
    let Some(result) = generate(
        r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},
            "city":{"type":"string"}},"required":["name","age","city"]}"#,
        "Extract the name and age from: John is 34 years old and lives in Paris.",
        100,
    ) else {
        eprintln!("skipping: no local GGUF fixture available");
        return;
    };
    let v = result.expect("generate_json");
    assert!(v.get("name").and_then(|x| x.as_str()).is_some(), "{v}");
    assert!(v.get("age").and_then(|x| x.as_i64()).is_some(), "{v}");
    assert!(v.get("city").and_then(|x| x.as_str()).is_some(), "{v}");
}

#[test]
fn enum_and_array_fields_are_respected() {
    let allowed = ["positive", "negative", "neutral", "mixed"];
    let Some(result) = generate(
        r#"{"type":"object","properties":{"sentiment":{"type":"string",
            "enum":["positive","negative","neutral","mixed"]},
            "keywords":{"type":"array","items":{"type":"string"}}},
            "required":["sentiment","keywords"]}"#,
        "Classify the sentiment and list up to 3 keywords: The battery life is amazing but the \
         screen is way too dim.",
        150,
    ) else {
        eprintln!("skipping: no local GGUF fixture available");
        return;
    };
    let v = result.expect("generate_json");
    let sentiment = v.get("sentiment").and_then(|x| x.as_str()).expect("sentiment field");
    assert!(allowed.contains(&sentiment), "sentiment {sentiment:?} not one of {allowed:?}");
    let keywords = v.get("keywords").and_then(|x| x.as_array()).expect("keywords array");
    assert!(!keywords.is_empty());
    assert!(keywords.iter().all(|k| k.is_string()));
}

#[test]
fn every_primitive_type_composes_in_one_schema() {
    let Some(result) = generate(
        r#"{"type":"object","properties":{"name":{"type":"string"},"legs":{"type":"integer"},
            "can_fly":{"type":"boolean"},"diet":{"type":"string",
            "enum":["carnivore","herbivore","omnivore"]}},
            "required":["name","legs","can_fly","diet"]}"#,
        "Describe a fictional animal with a name, leg count, whether it can fly, and its diet.",
        200,
    ) else {
        eprintln!("skipping: no local GGUF fixture available");
        return;
    };
    let v = result.expect("generate_json");
    assert!(v.get("name").and_then(|x| x.as_str()).is_some(), "{v}");
    assert!(v.get("legs").and_then(|x| x.as_i64()).is_some(), "{v}");
    assert!(v.get("can_fly").and_then(|x| x.as_bool()).is_some(), "{v}");
    let diet = v.get("diet").and_then(|x| x.as_str()).expect("diet field");
    assert!(["carnivore", "herbivore", "omnivore"].contains(&diet), "{v}");
}
