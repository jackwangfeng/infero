//! HTTP handlers.

use std::convert::Infallible;
use std::sync::Arc;

use anyhow::Result;
use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures::Stream;
use tokio::sync::mpsc;
use tuili_tokenizer::ChatMessage;

use crate::api::*;
use crate::engine::{self, Engine, Event, FinishReason, Request};

pub fn router(engine: Arc<Engine>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(engine)
}

/// Anything that can go wrong, rendered as OpenAI's error envelope.
struct ApiError {
    status: StatusCode,
    message: String,
    code: Option<&'static str>,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            code: Some("invalid_request_error"),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            code: None,
        }
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        ApiError::internal(format!("{e:#}"))
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let kind = if self.status.is_client_error() {
            "invalid_request_error"
        } else {
            "internal_error"
        };
        tracing::warn!(status = %self.status, message = %self.message, "request rejected");
        (
            self.status,
            Json(ErrorBody {
                error: ErrorDetail {
                    message: self.message,
                    kind,
                    code: self.code,
                },
            }),
        )
            .into_response()
    }
}

async fn health(State(engine): State<Arc<Engine>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "model": engine.info.id,
        "path": engine.info.path,
        "quantization": engine.info.quant,
        "weights_mib": engine.info.weights_mib,
        "kv_quant": engine.info.kv_quant,
        "kv_cache_mib": engine.info.kv_cache_mib,
        "kv_bits_per_channel": engine.info.kv_bits_per_channel,
        "vram_mib": engine.info.vram_mib,
        "offloaded_mib": engine.info.offloaded_mib,
        "offloaded_layers": engine.info.offloaded_layers,
        "max_seqs": engine.info.max_seqs,
        "kv_slots": engine.info.kv_slots,
        "context_length": engine.info.context_length,
        "max_seq": engine.info.max_seq,
        "queue_depth": engine.queue_depth(),
        "requests_served": engine.requests_served(),
    }))
}

async fn models(State(engine): State<Arc<Engine>>) -> impl IntoResponse {
    Json(ModelList {
        object: "list",
        data: vec![ModelCard {
            id: engine.info.id.clone(),
            object: "model",
            created: now_secs(),
            owned_by: "tuili",
            quantization: engine.info.quant.clone(),
            context_length: engine.info.context_length,
            max_seq: engine.info.max_seq,
        }],
    })
}

async fn chat_completions(
    State(engine): State<Arc<Engine>>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }

    let template = engine.tokenizer().chat_template().ok_or_else(|| {
        ApiError::bad_request("this model has no chat template; use /v1/completions instead")
    })?;

    let messages: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(|m| ChatMessage::new(&m.role, m.text()))
        .collect();
    let prompt = template
        .render(&messages, true)
        .map_err(|e| ApiError::bad_request(format!("chat template failed: {e:#}")))?;

    // parse_special = true: the template's own markers must become control
    // tokens. Message *content* was already escaped into the template output,
    // so a user writing "<|im_start|>" cannot forge a turn boundary here —
    // the template quotes it as literal text.
    let tokens = engine.tokenizer().encode(&prompt, Some(false), true);

    let params = Knobs {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        seed: req.seed,
        repetition_penalty: req.repetition_penalty,
        frequency_penalty: req.frequency_penalty.or(req.presence_penalty),
    }
    .into_params();

    let max_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(512)
        .clamp(1, engine.info.max_seq);
    let stop = req.stop.map(StopField::into_vec).unwrap_or_default();

    let rx = engine.submit(Request {
        prompt: tokens,
        params,
        max_tokens,
        stop,
        events: dummy_sender(),
    })?;

    let model = engine.info.id.clone();
    if req.stream {
        Ok(Sse::new(chat_stream(rx, model))
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let (text, reason, prompt_tokens, completion_tokens) =
            engine::collect(rx).await.map_err(ApiError::from)?;
        Ok(Json(ChatResponse {
            id: request_id("chatcmpl"),
            object: "chat.completion",
            created: now_secs(),
            model,
            choices: vec![ChatChoice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant",
                    content: text,
                },
                finish_reason: reason.as_str(),
            }],
            usage: Usage::new(prompt_tokens, completion_tokens),
        })
        .into_response())
    }
}

async fn completions(
    State(engine): State<Arc<Engine>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    let dropped = req.prompt.extra_count();
    if dropped > 0 {
        tracing::warn!(
            dropped,
            "batched prompts are not supported; serving the first"
        );
    }
    let prompt = req.prompt.first();
    if prompt.is_empty() {
        return Err(ApiError::bad_request("prompt must not be empty"));
    }

    // parse_special = false: raw completion input is untrusted text, and a
    // literal "<|im_end|>" in it must not become a control token.
    let tokens = engine.tokenizer().encode(prompt, Some(req.echo_bos), false);

    let params = Knobs {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        seed: req.seed,
        repetition_penalty: req.repetition_penalty,
        frequency_penalty: None,
    }
    .into_params();

    let max_tokens = req.max_tokens.unwrap_or(256).clamp(1, engine.info.max_seq);
    let stop = req.stop.map(StopField::into_vec).unwrap_or_default();

    let rx = engine.submit(Request {
        prompt: tokens,
        params,
        max_tokens,
        stop,
        events: dummy_sender(),
    })?;

    let model = engine.info.id.clone();
    if req.stream {
        Ok(Sse::new(completion_stream(rx, model))
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let (text, reason, prompt_tokens, completion_tokens) =
            engine::collect(rx).await.map_err(ApiError::from)?;
        Ok(Json(CompletionResponse {
            id: request_id("cmpl"),
            object: "text_completion",
            created: now_secs(),
            model,
            choices: vec![CompletionChoice {
                index: 0,
                text,
                finish_reason: reason.as_str(),
                logprobs: None,
            }],
            usage: Usage::new(prompt_tokens, completion_tokens),
        })
        .into_response())
    }
}

fn chat_stream(
    mut rx: mpsc::UnboundedReceiver<Event>,
    model: String,
) -> impl Stream<Item = Result<SseEvent, Infallible>> {
    let id = request_id("chatcmpl");
    async_stream(move |yielder| async move {
        // OpenAI's first chunk carries the role and no content.
        yielder
            .send(chunk_event(
                &id,
                &model,
                Delta {
                    role: Some("assistant"),
                    content: None,
                },
                None,
            ))
            .await;

        let mut reason = FinishReason::Stop;
        while let Some(ev) = rx.recv().await {
            match ev {
                Event::Text(text) if text.is_empty() => {}
                Event::Text(text) => {
                    yielder
                        .send(chunk_event(
                            &id,
                            &model,
                            Delta {
                                role: None,
                                content: Some(text),
                            },
                            None,
                        ))
                        .await;
                }
                Event::Done { reason: r, .. } => {
                    reason = r;
                    break;
                }
                Event::Failed(message) => {
                    yielder.send(error_event(&message)).await;
                    yielder.send(done_event()).await;
                    return;
                }
            }
        }

        yielder
            .send(chunk_event(
                &id,
                &model,
                Delta::default(),
                Some(reason.as_str()),
            ))
            .await;
        yielder.send(done_event()).await;
    })
}

fn completion_stream(
    mut rx: mpsc::UnboundedReceiver<Event>,
    model: String,
) -> impl Stream<Item = Result<SseEvent, Infallible>> {
    let id = request_id("cmpl");
    async_stream(move |yielder| async move {
        let mut reason = FinishReason::Stop;
        while let Some(ev) = rx.recv().await {
            match ev {
                Event::Text(text) if text.is_empty() => {}
                Event::Text(text) => {
                    let chunk = CompletionChunk {
                        id: id.clone(),
                        object: "text_completion",
                        created: now_secs(),
                        model: model.clone(),
                        choices: vec![CompletionChunkChoice {
                            index: 0,
                            text,
                            finish_reason: None,
                        }],
                    };
                    yielder.send(json_event(&chunk)).await;
                }
                Event::Done { reason: r, .. } => {
                    reason = r;
                    break;
                }
                Event::Failed(message) => {
                    yielder.send(error_event(&message)).await;
                    yielder.send(done_event()).await;
                    return;
                }
            }
        }

        let chunk = CompletionChunk {
            id: id.clone(),
            object: "text_completion",
            created: now_secs(),
            model: model.clone(),
            choices: vec![CompletionChunkChoice {
                index: 0,
                text: String::new(),
                finish_reason: Some(reason.as_str()),
            }],
        };
        yielder.send(json_event(&chunk)).await;
        yielder.send(done_event()).await;
    })
}

fn chunk_event(
    id: &str,
    model: &str,
    delta: Delta,
    finish_reason: Option<&'static str>,
) -> SseEvent {
    json_event(&ChatChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created: now_secs(),
        model: model.to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta,
            finish_reason,
        }],
    })
}

fn json_event<T: serde::Serialize>(value: &T) -> SseEvent {
    match serde_json::to_string(value) {
        Ok(s) => SseEvent::default().data(s),
        // Serializing our own types cannot fail in practice; if it somehow
        // does, tell the client rather than dropping the frame.
        Err(e) => SseEvent::default()
            .data(serde_json::json!({"error": {"message": e.to_string()}}).to_string()),
    }
}

fn error_event(message: &str) -> SseEvent {
    SseEvent::default().data(
        serde_json::json!({"error": {"message": message, "type": "internal_error"}}).to_string(),
    )
}

/// The sentinel every OpenAI-compatible client waits for.
fn done_event() -> SseEvent {
    SseEvent::default().data("[DONE]")
}

/// A placeholder sender; [`Engine::submit`] replaces it with the real one.
fn dummy_sender() -> mpsc::UnboundedSender<Event> {
    mpsc::unbounded_channel().0
}

// A tiny generator helper so the stream bodies above read top to bottom
// instead of as a state machine.
mod stream_helper {
    use super::*;
    use tokio::sync::mpsc;

    pub struct Yielder(pub mpsc::Sender<SseEvent>);

    impl Yielder {
        pub async fn send(&self, event: SseEvent) {
            // A closed receiver means the client hung up; the worker notices
            // the same thing and stops generating.
            let _ = self.0.send(event).await;
        }
    }
}

use stream_helper::Yielder;

fn async_stream<F, Fut>(body: F) -> impl Stream<Item = Result<SseEvent, Infallible>>
where
    F: FnOnce(Yielder) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let (tx, rx) = mpsc::channel::<SseEvent>(32);
    tokio::spawn(async move { body(Yielder(tx)).await });
    tokio_stream::wrappers::ReceiverStream::new(rx).map(Ok)
}

use futures::StreamExt;
