//! Per-kernel timing, same shape as the CUDA/Metal sides.
//!
//! A kernel runs synchronously inside `Function::launch` on this backend, so
//! there is no event/queue overhead to correct for the way Metal's own
//! comment describes -- a host clock around the call is already the exact
//! cost, not an approximation of it.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;

use crate::device::Stream;

#[derive(Default, Clone, Copy)]
pub struct Entry {
    pub launches: u64,
    pub millis: f64,
}

pub struct Profile {
    enabled: AtomicBool,
    entries: Mutex<HashMap<&'static str, Entry>>,
}

impl Profile {
    pub fn new() -> Self {
        Self {
            enabled: AtomicBool::new(std::env::var("INFERO_CPU_PROFILE").is_ok()),
            entries: Mutex::new(HashMap::new()),
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    pub fn time<T>(&self, name: &'static str, _stream: Stream, launch: impl FnOnce() -> Result<T>) -> Result<T> {
        if !self.enabled() {
            return launch();
        }
        let t = std::time::Instant::now();
        let out = launch()?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let mut e = self.entries.lock().unwrap();
        let slot = e.entry(name).or_default();
        slot.launches += 1;
        slot.millis += ms;
        Ok(out)
    }

    pub fn snapshot(&self) -> Vec<(&'static str, Entry)> {
        let mut v: Vec<_> = self.entries.lock().unwrap().iter().map(|(k, v)| (*k, *v)).collect();
        v.sort_by(|a, b| b.1.millis.total_cmp(&a.1.millis));
        v
    }

    pub fn reset(&self) {
        self.entries.lock().unwrap().clear();
    }

    pub fn report(&self) -> String {
        let mut s = String::new();
        for (name, e) in self.snapshot() {
            s.push_str(&format!(
                "  {name:<28} {:>8} launches  {:>9.3} ms  {:>7.1} us each\n",
                e.launches,
                e.millis,
                e.millis * 1e3 / e.launches.max(1) as f64
            ));
        }
        s
    }
}

impl Default for Profile {
    fn default() -> Self {
        Self::new()
    }
}
