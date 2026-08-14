//! Per-kernel wall-clock accounting.
//!
//! Exists because guessing at which kernel is slow has a poor track record in
//! this project: three rounds of tuning the float mat-vec bought 1.8x, and one
//! look at the actual algorithm bought 9x. Enabled by `TUILI_PROFILE`, off it
//! costs an atomic load per launch.
//!
//! Timing is by CUDA events around each launch, so it measures device time
//! rather than submission time — and it serializes the stream, which is exactly
//! what you do not want in production and exactly what you need to attribute
//! milliseconds.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use cudarc::driver::{CudaContext, CudaEvent, CudaStream};

/// One kernel's accumulated cost.
#[derive(Default, Clone, Copy)]
pub struct Entry {
    pub launches: u64,
    pub millis: f64,
}

pub struct Profile {
    enabled: AtomicBool,
    entries: Mutex<HashMap<&'static str, Entry>>,
    /// Reused rather than created per launch; event creation is not free.
    events: Mutex<Option<(CudaEvent, CudaEvent)>>,
}

impl Profile {
    pub fn new(ctx: &std::sync::Arc<CudaContext>) -> Result<Self> {
        let enabled = std::env::var_os("TUILI_PROFILE").is_some();
        let events = if enabled {
            // Timing events, unlike the disable-timing ones used elsewhere.
            Some((
                ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?,
                ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?,
            ))
        } else {
            None
        };
        if enabled {
            tracing::warn!("TUILI_PROFILE set: kernels are timed and the stream serializes");
        }
        Ok(Self {
            enabled: AtomicBool::new(enabled),
            entries: Mutex::new(HashMap::new()),
            events: Mutex::new(events),
        })
    }

    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Time `launch` and attribute it to `name`.
    ///
    /// The caller passes a closure rather than a pair of start/stop calls so a
    /// kernel that returns early cannot leave a dangling start event.
    pub fn time<T>(
        &self,
        name: &'static str,
        stream: &CudaStream,
        launch: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        if !self.enabled() {
            return launch();
        }
        let guard = self.events.lock().unwrap();
        let Some((start, stop)) = guard.as_ref() else {
            return launch();
        };
        start.record(stream)?;
        let out = launch()?;
        stop.record(stream)?;
        stop.synchronize()?;
        let ms = start.elapsed_ms(stop)? as f64;
        drop(guard);

        let mut entries = self.entries.lock().unwrap();
        let e = entries.entry(name).or_default();
        e.launches += 1;
        e.millis += ms;
        Ok(out)
    }

    pub fn snapshot(&self) -> Vec<(&'static str, Entry)> {
        let mut v: Vec<_> = self
            .entries
            .lock()
            .unwrap()
            .iter()
            .map(|(k, e)| (*k, *e))
            .collect();
        v.sort_by(|a, b| b.1.millis.total_cmp(&a.1.millis));
        v
    }

    pub fn reset(&self) {
        self.entries.lock().unwrap().clear();
    }

    /// A table, slowest first, with each kernel's share of the total.
    pub fn report(&self) -> String {
        let rows = self.snapshot();
        let total: f64 = rows.iter().map(|(_, e)| e.millis).sum();
        if rows.is_empty() {
            return "no kernels timed (set TUILI_PROFILE)".into();
        }
        let mut out = format!(
            "{:<22} {:>10} {:>10} {:>9} {:>8}\n",
            "kernel", "total ms", "launches", "us each", "share"
        );
        for (name, e) in &rows {
            out += &format!(
                "{name:<22} {:>10.2} {:>10} {:>9.1} {:>7.1}%\n",
                e.millis,
                e.launches,
                e.millis * 1000.0 / e.launches as f64,
                100.0 * e.millis / total
            );
        }
        out += &format!("{:<22} {total:>10.2}\n", "total");
        out
    }
}
