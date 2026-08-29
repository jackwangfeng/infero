//! infero's HTTP layer: an OpenAI-compatible API over a GGUF model on CUDA.
//!
//! Split out from the binary so the endpoints can be exercised by integration
//! tests against a real model.

pub mod api;
pub mod engine;
pub mod routes;
pub mod prefix;
pub mod scheduler;
pub mod tool_call;
pub mod vision;
pub mod stop;

pub use engine::{Engine, ModelInfo};
