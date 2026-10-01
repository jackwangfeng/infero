//! OpenAI-compatible request and response bodies.
//!
//! Only the fields that change what the engine does are honoured; the rest are
//! accepted and ignored so that existing clients work unmodified.

use serde::{Deserialize, Serialize};
use infero_model::SamplingParams;

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
    /// Extra variables for the model's own chat template, as vLLM and the rest
    /// of the OpenAI ecosystem spell it. Qwen3.5 needs
    /// `{"enable_thinking": false}` here to produce a non-thinking turn: its
    /// template treats an *undefined* `enable_thinking` as on.
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Value>,
    /// OpenAI-shaped function definitions, passed through to the template
    /// untouched — `chat_template.jinja` does `tool | tojson` on each one
    /// itself, so there is nothing to translate on the way in.
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    /// Only `"auto"`/absent and `"none"` are honoured. Qwen3.5's template has
    /// no lever to force or forbid a specific function — it is always "if you
    /// choose to call one" — so `"required"` or a forced-function object is
    /// refused rather than silently treated as `"auto"`.
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// OpenAI's own field name and shape for schema-constrained decoding
    /// (`docs/keel-integration.md`'s "2. Schema-constrained decoding"
    /// section). Only `{"type": "json_schema", "json_schema": {"schema":
    /// {...}}}` diverts to the constrained path
    /// (`crate::routes::generate_constrained`); a bare `{"type":
    /// "json_object"}` or `{"type": "text"}` is accepted and ignored, same
    /// as this module's own "accept the rest, honour what changes
    /// behaviour" convention.
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: JsonSchemaSpec },
}

#[derive(Debug, Deserialize)]
pub struct JsonSchemaSpec {
    #[allow(dead_code)]
    #[serde(default)]
    pub name: Option<String>,
    pub schema: serde_json::Value,
    #[allow(dead_code)]
    #[serde(default)]
    pub strict: Option<bool>,
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
    /// An assistant turn's function calls, when replaying history that made
    /// one. Absent on every other role.
    #[serde(default)]
    pub tool_calls: Option<Vec<InToolCall>>,
    /// Which call a `role: "tool"` message answers. Accepted for API
    /// compatibility; Qwen3.5's template does not read it — it matches a
    /// tool result to its call by turn order, not by id.
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct InToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: InToolCallFunction,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct InToolCallFunction {
    pub name: String,
    /// A JSON string on the wire, per OpenAI's spec — `infero_tokenizer`'s
    /// `ToolCallFunction::arguments` wants the parsed object instead; see
    /// `routes::to_chat_message`.
    pub arguments: String,
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
    #[serde(default)]
    pub image_url: Option<ImageUrl>,
    #[serde(default)]
    pub video_url: Option<VideoUrl>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ImageUrl {
    pub url: String,
}

/// Not part of OpenAI's schema (video input is this server's own extension,
/// same as `video_url` itself) -- `fps` mirrors vLLM's per-request
/// `media_io_kwargs["video"]["fps"]` and Gemini's `videoMetadata.fps`: how
/// densely to sample this one clip, overriding the server's own
/// `--video-target-fps` default. `None` when the caller does not care and
/// wants that default.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct VideoUrl {
    pub url: String,
    #[serde(default)]
    pub fps: Option<f64>,
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

    /// The `image_url.url` of every image part, in the order they appear.
    ///
    /// Kept separate from [`Self::text`] rather than folded into one pass over
    /// `content`, because the two callers want different things: routing
    /// builds the model-facing message from *all* parts in order (text and
    /// image interleaved, for a template that cares), while this is only for
    /// deciding whether there is an image to fetch at all and what its source
    /// is.
    pub fn image_urls(&self) -> Vec<&str> {
        match &self.content {
            Some(Content::Parts(parts)) => parts
                .iter()
                .filter_map(|p| p.image_url.as_ref())
                .map(|u| u.url.as_str())
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Every video part's `video_url`, in the order they appear -- the whole
    /// object rather than just the url, since a caller may also have set
    /// `fps`. Same reasoning as [`Self::image_urls`].
    pub fn video_urls(&self) -> Vec<&VideoUrl> {
        match &self.content {
            Some(Content::Parts(parts)) => parts.iter().filter_map(|p| p.video_url.as_ref()).collect(),
            _ => Vec::new(),
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
    /// `null` rather than `""` when a turn is nothing but tool calls — an
    /// empty string reads as "the model said nothing" where OpenAI's own
    /// convention is "there was nothing to say here".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OutToolCall>>,
}

#[derive(Debug, Serialize, Clone)]
pub struct OutToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: OutToolCallFunction,
}

#[derive(Debug, Serialize, Clone)]
pub struct OutToolCallFunction {
    pub name: String,
    /// A JSON string, matching OpenAI's own wire shape — the inverse of
    /// `InToolCallFunction::arguments`.
    pub arguments: String,
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
    /// Every call in one delta rather than split across chunks: this
    /// server's own scan does not learn a call is complete until the whole
    /// `</tool_call>` has arrived, so there is nothing to stream
    /// incrementally the way token-by-token content is. `index` on each
    /// entry is still the position OpenAI's shape expects.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ToolCallDelta {
    pub index: usize,
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: OutToolCallFunction,
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

/// See `docs/keel-integration.md`'s "The contract between them is two
/// endpoints" section -- this is that request shape, not OpenAI's
/// `input`/`data[].embedding` one. `normalize` is accepted (so an existing
/// caller's request body still parses) but not honoured as a toggle: the
/// documented bug this endpoint exists partly to avoid repeating is a
/// `normalize: false` that silently did nothing because the *model's own*
/// config already normalized regardless of the request, so this server
/// normalizes unconditionally and asserts the result itself (see
/// `infero_model::embed::normalize`) rather than making that guarantee
/// something a request field can turn off.
#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    #[allow(dead_code)]
    pub model: Option<String>,
    pub texts: Vec<String>,
    #[allow(dead_code)]
    #[serde(default)]
    pub normalize: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingsResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub dim: usize,
    pub model: String,
    pub model_version: String,
}

/// See `docs/keel-integration.md`'s "3. `/v1/rerank`" section for this
/// request shape. `instruction` is not part of that documented contract --
/// Qwen3-Reranker's own real recipe takes one, and exposing it costs nothing
/// a caller that never sets it would notice, since it defaults to the same
/// instruction the model card itself defaults to
/// (`infero_model::rerank::DEFAULT_INSTRUCTION`).
#[derive(Debug, Deserialize)]
pub struct RerankRequest {
    #[allow(dead_code)]
    pub model: Option<String>,
    pub query: String,
    pub documents: Vec<String>,
    pub top_k: Option<usize>,
    #[serde(default)]
    pub instruction: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RerankResult {
    pub index: usize,
    pub score: f32,
}

/// `POST /v1/systemone`: Kev-style decision-model scoring, not text
/// generation -- see `infero_model::kev`'s module doc for the mechanism and
/// `docs/kev-decision-model.md` for the full walkthrough. Field shapes
/// mirror TypeSafe's own wire format exactly (`kev/api.py`'s `SystemOneRequest`
/// upstream), the one both Jev and every Kev backend (torch/MLX/vLLM) answer
/// to, so a client written against any of them needs no adapter here.
///
/// One real, documented deviation from upstream: a `choice` question's
/// `criteria` object is read back out in **sorted key order**, not the
/// order the request JSON had them in -- this workspace's `serde_json` is
/// not built with the `preserve_order` feature, so `serde_json::Map`'s
/// iteration order is its `BTreeMap` backing's, not insertion order. This
/// changes which order the model sees the options in (and so, in
/// principle, could change which one it favors on a near-tie) but not the
/// correctness of the answer: every probability in the response is still
/// keyed by its own option name, not a positional index.
#[derive(Debug, Deserialize)]
pub struct SystemOneRequest {
    pub state: serde_json::Value,
    #[serde(default = "default_kev_model")]
    pub model: String,
    pub questions: std::collections::BTreeMap<String, SystemOneQuestion>,
}

fn default_kev_model() -> String {
    "kev-latest".to_string()
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SystemOneQuestion {
    /// Yes/no. `criteria` is `{"false": ..., "true": ...}`, both optional
    /// descriptions (`kev/api.py`'s `Noul`).
    Noul {
        #[serde(default)]
        instructions: serde_json::Value,
        #[serde(default)]
        criteria: Option<serde_json::Map<String, serde_json::Value>>,
    },
    /// Multiple choice. `criteria` maps each option's name to an optional
    /// description (`kev/api.py`'s `Choice`).
    Choice {
        #[serde(default)]
        instructions: serde_json::Value,
        criteria: serde_json::Map<String, serde_json::Value>,
    },
    /// An ordered rating scale. `criteria` is the level descriptions, lowest
    /// to highest (`kev/api.py`'s `Score`).
    Score {
        #[serde(default)]
        instructions: serde_json::Value,
        criteria: Vec<serde_json::Value>,
    },
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum SystemOneAnswer {
    Noul {
        #[serde(rename = "type")]
        kind: &'static str,
        noul: f32,
    },
    Choice {
        #[serde(rename = "type")]
        kind: &'static str,
        choice: String,
        confidence: f32,
        probabilities: std::collections::BTreeMap<String, f32>,
    },
    Score {
        #[serde(rename = "type")]
        kind: &'static str,
        score: f32,
        legend: std::collections::BTreeMap<String, serde_json::Value>,
        probabilities: std::collections::BTreeMap<String, f32>,
        confidence: f32,
    },
}

#[derive(Debug, Serialize)]
pub struct SystemOneResponse {
    pub model: String,
    pub model_version: String,
    pub answers: std::collections::BTreeMap<String, SystemOneAnswer>,
}

#[derive(Debug, Serialize)]
pub struct RerankResponse {
    pub results: Vec<RerankResult>,
}

/// Flatten `str | object | array | number | bool | null` into text, field
/// names kept as labels -- port of `kev/api.py`'s `render`. What a `state`
/// or an option's description actually gets tokenized as.
pub fn kev_render(v: &serde_json::Value, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match v {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(_) | serde_json::Value::Bool(_) => v.to_string(),
        serde_json::Value::Array(items) => items
            .iter()
            .map(|x| format!("{pad}- {}", kev_render(x, indent + 1).trim_start()))
            .collect::<Vec<_>>()
            .join("\n"),
        serde_json::Value::Object(map) => map
            .iter()
            .map(|(k, x)| {
                if x.is_object() || x.is_array() {
                    format!("{pad}{k}:\n{}", kev_render(x, indent + 1))
                } else {
                    format!("{pad}{k}: {}", kev_render(x, indent))
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Port of `kev/api.py`'s `option_text`: a name alone, or `"name: description"`.
pub fn kev_option_text(name: &str, desc: Option<&serde_json::Value>) -> String {
    match desc {
        None => name.to_string(),
        Some(serde_json::Value::Null) => name.to_string(),
        Some(serde_json::Value::String(s)) if s.is_empty() => name.to_string(),
        Some(v) => format!("{name}: {}", kev_render(v, 0)),
    }
}

/// `(p_max - 1/K) / (1 - 1/K)`: 0 at uniform, 1 at certainty. Port of
/// `kev/api.py`'s `choice_confidence`.
pub fn choice_confidence(p: &[f32]) -> f32 {
    let k = p.len();
    if k <= 1 {
        return 1.0;
    }
    let sum: f32 = p.iter().sum();
    let max = if sum.abs() < 1e-9 {
        1.0 / k as f32
    } else {
        p.iter().copied().fold(f32::MIN, f32::max) / sum
    };
    (max - 1.0 / k as f32) / (1.0 - 1.0 / k as f32)
}

/// Port of `kev/api.py`'s `score_confidence`: `1` when all mass sits on one
/// level, `0` at uniform or anything as spread.
pub fn score_confidence(p: &[f32]) -> f32 {
    let l = p.len();
    if l <= 1 {
        return 1.0;
    }
    let sum: f32 = p.iter().sum();
    let norm: Vec<f32> = if sum.abs() < 1e-9 {
        vec![1.0 / l as f32; l]
    } else {
        p.iter().map(|x| x / sum).collect()
    };
    let mode = norm
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mean = (l - 1) as f32 / 2.0;
    let d: f32 = (0..l).map(|i| (i as f32 - mean).abs()).sum::<f32>() / l as f32;
    let spread: f32 = norm.iter().enumerate().map(|(i, &pi)| pi * (i as f32 - mode as f32).abs()).sum();
    (1.0 - spread / d).max(0.0)
}

/// 4 decimal places -- `kev/api.py`'s `round_prob`'s own tolerance
/// reasoning (255 options x 0.00005 rounding error < TypeSafe's 0.02
/// sum-of-probabilities tolerance).
pub fn round_prob(x: f32) -> f32 {
    (x * 10000.0).round() / 10000.0
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
