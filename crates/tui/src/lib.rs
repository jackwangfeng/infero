//! Terminal client for a tuili server.
//!
//! Split into a library so the HTTP and wrapping layers can be tested without
//! a terminal attached.

pub mod app;
pub mod client;
pub mod ui;
pub mod wrap;
