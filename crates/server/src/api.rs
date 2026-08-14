//! OpenAI-compatible request and response bodies.
//!
//! Only the fields that change what the engine does are honoured; the rest are
//! accepted and ignored so that existing clients work unmodified.

use serde::{Deserialize, Serialize};
use tuili_model::SamplingParams;

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// The newer name for `max_tokens`; both are accepted.
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<StopField>,
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: PromptField,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<StopField>,
    /// Whether to prepend the model's BOS token, if it has one.
    #[serde(default = "default_true")]
    pub echo_bos: bool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum PromptField {
    Text(String),
    /// OpenAI allows a batch of prompts; we serve the first and ignore the
    /// rest rather than silently returning one completion for all of them.
    Batch(Vec<String>),
}

impl PromptField {
    pub fn first(&self) -> &str {
        match self {
            PromptField::Text(s) => s,
            PromptField::Batch(v) => v.first().map(String::as_str).unwrap_or(""),
        }
    }

    pub fn extra_count(&self) -> usize {
        match self {
            PromptField::Text(_) => 0,
            PromptField::Batch(v) => v.len().saturating_sub(1),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopField {
    One(String),
    Many(Vec<String>),
}

impl StopField {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopField::One(s) => vec![s],
            StopField::Many(v) => v,
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Message {
    pub role: String,
    /// Content may arrive as a plain string or as OpenAI's content-part array.
    #[serde(default)]
    pub content: Option<Content>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
}

impl Message {
    /// Flatten the content into plain text, which is all a text model can use.
    pub fn text(&self) -> String {
        match &self.content {
            None => String::new(),
            Some(Content::Text(s)) => s.clone(),
            Some(Content::Parts(parts)) => parts
                .iter()
                .filter(|p| p.kind == "text")
                .filter_map(|p| p.text.as_deref())
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

// ---- responses ----------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ResponseMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize, Default)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: &'static str,
    pub logprobs: Option<()>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChunkChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    pub fn new(prompt: usize, completion: usize) -> Self {
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

#[derive(Debug, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
    // Not part of the OpenAI schema, but the first thing anyone wants to know.
    pub quantization: String,
    pub context_length: usize,
    pub max_seq: usize,
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub code: Option<&'static str>,
}

// ---- parameter mapping --------------------------------------------------

/// Sampling knobs shared by both endpoints.
pub struct Knobs {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub seed: Option<u64>,
    pub repetition_penalty: Option<f32>,
    /// OpenAI's frequency penalty, folded into the repetition penalty when no
    /// explicit one is given.
    pub frequency_penalty: Option<f32>,
}

impl Knobs {
    pub fn into_params(self) -> SamplingParams {
        let d = SamplingParams::default();
        SamplingParams {
            temperature: self.temperature.unwrap_or(d.temperature).clamp(0.0, 4.0),
            top_p: self.top_p.unwrap_or(d.top_p).clamp(0.0, 1.0),
            // top_k = 0 means "no limit" in most clients.
            top_k: match self.top_k {
                Some(0) | None => d.top_k,
                Some(k) => k,
            },
            repetition_penalty: self
                .repetition_penalty
                // frequency_penalty runs 0..2 additively in OpenAI's model;
                // approximating it as a multiplicative penalty is the closest
                // this sampler can get.
                .or_else(|| {
                    self.frequency_penalty
                        .map(|f| 1.0 + f.clamp(0.0, 2.0) * 0.5)
                })
                .unwrap_or(d.repetition_penalty)
                .clamp(1.0, 2.0),
            repetition_window: d.repetition_window,
            seed: self.seed,
        }
    }
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A request id in OpenAI's shape. Uniqueness only has to hold within a
/// process lifetime, so a counter beats pulling in a uuid dependency.
pub fn request_id(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{:016x}{:08x}", now_secs(), n as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_parts_flatten_to_text() {
        let m: Message = serde_json::from_str(
            r#"{"role":"user","content":[{"type":"text","text":"a"},{"type":"image_url"},{"type":"text","text":"b"}]}"#,
        )
        .unwrap();
        assert_eq!(m.text(), "ab");
    }

    #[test]
    fn plain_string_content_still_parses() {
        let m: Message = serde_json::from_str(r#"{"role":"user","content":"hello"}"#).unwrap();
        assert_eq!(m.text(), "hello");
    }

    #[test]
    fn stop_accepts_a_string_or_a_list() {
        let one: StopField = serde_json::from_str(r#""END""#).unwrap();
        assert_eq!(one.into_vec(), vec!["END"]);
        let many: StopField = serde_json::from_str(r#"["a","b"]"#).unwrap();
        assert_eq!(many.into_vec(), vec!["a", "b"]);
    }

    #[test]
    fn top_k_zero_means_unlimited_not_empty() {
        let p = Knobs {
            temperature: None,
            top_p: None,
            top_k: Some(0),
            seed: None,
            repetition_penalty: None,
            frequency_penalty: None,
        }
        .into_params();
        assert!(p.top_k > 1, "top_k collapsed to {}", p.top_k);
    }

    #[test]
    fn temperature_zero_survives_as_greedy() {
        let p = Knobs {
            temperature: Some(0.0),
            top_p: None,
            top_k: None,
            seed: None,
            repetition_penalty: None,
            frequency_penalty: None,
        }
        .into_params();
        assert!(p.is_greedy());
    }

    #[test]
    fn a_batch_prompt_reports_what_it_dropped() {
        let p: PromptField = serde_json::from_str(r#"["a","b","c"]"#).unwrap();
        assert_eq!(p.first(), "a");
        assert_eq!(p.extra_count(), 2);
    }

    #[test]
    fn request_ids_do_not_repeat() {
        let a = request_id("chatcmpl");
        let b = request_id("chatcmpl");
        assert_ne!(a, b);
        assert!(a.starts_with("chatcmpl-"));
    }
}
