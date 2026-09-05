//! Prometheus text-exposition metrics and liveness bookkeeping.
//!
//! Hand-formatted rather than pulling in a `prometheus` crate: this crate has
//! no such dependency today, and a handful of counters/gauges in the
//! exposition format (`# HELP` / `# TYPE` / `name value`) don't need one.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::engine::Engine;

/// Shared counters the HTTP layer and the worker thread both write to.
///
/// Everything here is a real, load-bearing count fed by an actual request
/// outcome or worker event — nothing is a placeholder metric with no
/// source. There is deliberately no latency histogram: nothing in
/// `Scheduler`/`Worker` tracks per-request timing today, and fabricating one
/// from a coarse wall-clock wrap in the HTTP handler would mislabel queueing
/// time as inference time. Add one when the scheduler itself times a step.
#[derive(Default)]
pub struct Metrics {
    pub requests_received: AtomicU64,
    pub requests_ok: AtomicU64,
    pub requests_error: AtomicU64,
    pub prompt_tokens: AtomicU64,
    pub completion_tokens: AtomicU64,
    /// Set once, right after the worker thread's main loop returns (normal
    /// exit or otherwise) -- `false` for the entire life of a healthy
    /// process. This is what `/health/live` actually checks; it is not a
    /// heartbeat/timeout guess.
    pub worker_stopped: AtomicBool,
}

impl Metrics {
    pub fn record_received(&self) {
        self.requests_received.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_ok(&self, prompt_tokens: usize, completion_tokens: usize) {
        self.requests_ok.fetch_add(1, Ordering::Relaxed);
        self.prompt_tokens.fetch_add(prompt_tokens as u64, Ordering::Relaxed);
        self.completion_tokens.fetch_add(completion_tokens as u64, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.requests_error.fetch_add(1, Ordering::Relaxed);
    }
}

/// Render the full `/metrics` body for one `Engine`.
///
/// Real, current values every time -- `queue_depth`/`requests_served`/
/// `prefix_cache_stats` already read live atomics on `Engine`, and GPU memory
/// is a real `cuMemGetInfo` call on the same device handle the model runs on,
/// not a cached snapshot.
pub fn render(engine: &Engine) -> String {
    let m = &engine.metrics;
    let mut out = String::new();

    push_gauge(&mut out, "infero_up", "1 if the process is serving.", 1);
    push_gauge(
        &mut out,
        "infero_worker_alive",
        "1 if the inference worker thread's main loop is still running.",
        if m.worker_stopped.load(Ordering::Relaxed) { 0 } else { 1 },
    );
    push_gauge(
        &mut out,
        "infero_queue_depth",
        "Sequences currently admitted (in flight) on the worker.",
        engine.queue_depth(),
    );
    push_counter(
        &mut out,
        "infero_requests_received_total",
        "Requests submitted to the engine.",
        m.requests_received.load(Ordering::Relaxed),
    );
    push_counter(
        &mut out,
        "infero_requests_completed_total",
        "Sequences the worker has retired (success or failure), from the scheduler's own count.",
        engine.requests_served(),
    );
    push_counter(
        &mut out,
        "infero_requests_ok_total",
        "Requests the HTTP layer returned a successful response for.",
        m.requests_ok.load(Ordering::Relaxed),
    );
    push_counter(
        &mut out,
        "infero_requests_error_total",
        "Requests the HTTP layer returned an error for.",
        m.requests_error.load(Ordering::Relaxed),
    );
    push_counter(
        &mut out,
        "infero_prompt_tokens_total",
        "Prompt tokens processed, summed across completed requests.",
        m.prompt_tokens.load(Ordering::Relaxed),
    );
    push_counter(
        &mut out,
        "infero_completion_tokens_total",
        "Completion tokens generated, summed across completed requests.",
        m.completion_tokens.load(Ordering::Relaxed),
    );

    if let Some((lookups, hits, tokens_saved)) = engine.prefix_cache_stats() {
        push_counter(&mut out, "infero_prefix_cache_lookups_total", "Prefix-cache lookups.", lookups);
        push_counter(&mut out, "infero_prefix_cache_hits_total", "Prefix-cache hits.", hits);
        push_counter(
            &mut out,
            "infero_prefix_cache_tokens_saved_total",
            "Tokens served from a cached prefix instead of recomputed.",
            tokens_saved,
        );
    }

    match engine.mem_info() {
        Ok((free, total)) => {
            push_gauge(&mut out, "infero_gpu_memory_total_bytes", "Total device memory.", total as u64);
            push_gauge(
                &mut out,
                "infero_gpu_memory_used_bytes",
                "Device memory in use (total - free, this process's device).",
                (total - free) as u64,
            );
        }
        Err(e) => {
            // A real query failure (e.g. mid-shutdown) is worth a comment
            // line in the scrape rather than silently omitting the metric or
            // panicking the handler.
            out.push_str(&format!("# gpu memory query failed: {e:#}\n"));
        }
    }

    out
}

fn push_counter(out: &mut String, name: &str, help: &str, value: u64) {
    out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} counter\n{name} {value}\n"));
}

fn push_gauge(out: &mut String, name: &str, help: &str, value: u64) {
    out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} gauge\n{name} {value}\n"));
}

pub fn new_metrics() -> Arc<Metrics> {
    Arc::new(Metrics::default())
}
