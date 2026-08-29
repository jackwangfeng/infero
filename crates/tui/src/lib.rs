//! Terminal client for a infero server.
//!
//! Split into a library so the HTTP and wrapping layers can be tested without
//! a terminal attached.

pub mod app;
pub mod client;
pub mod ui;
pub mod wrap;
