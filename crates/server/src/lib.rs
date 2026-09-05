//! infero's HTTP layer: an OpenAI-compatible API over a GGUF model on CUDA.
//!
//! Split out from the binary so the endpoints can be exercised by integration
//! tests against a real model.

pub mod api;
pub mod auth;
pub mod engine;
pub mod metrics;
pub mod routes;
pub mod prefix;
pub mod scheduler;
pub mod tool_call;
pub mod vision;
pub mod video;
pub mod stop;
#[cfg(feature = "nccl")]
pub mod tp;

pub use engine::{Engine, ModelInfo};
