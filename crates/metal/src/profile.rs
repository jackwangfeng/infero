//! Per-kernel timing, with the same shape as the CUDA side's.
//!
//! The CUDA version brackets a launch with two events and reads their delta on
//! the device. Metal has GPU timestamps too, but they come off a command buffer
//! rather than a stream, and this backend submits one command buffer per launch
//! -- so the honest cheap version is a host clock around a synchronize. That
//! makes an enabled profile *change* what it measures, by serialising launches
//! that would otherwise overlap. Which is why it is off unless asked for, and
//! why the disabled path is a direct call with nothing added.

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
            // Its own switch, not the CUDA side's `INFERO_PROFILE`.
            //
            // There the two are the same variable because there they cost the
            // same: CUDA brackets a launch with events and reads them later, so
            // asking for a per-kernel table costs a few microseconds a launch
            // and the coarse timings in the scheduler stay honest alongside it.
            //
            // Here a per-kernel table costs a `synchronize` a launch, and a
            // 27B decode step runs about 500 of them. That floor -- 130 us a
            // launch, measured -- is larger than most of the kernels it is
            // measuring, and it inflates the scheduler's own draft-versus-verify
            // split by 66 ms a round, which is exactly the quantity that split
            // exists to attribute. One variable for both made the cheap
            // measurement impossible to take.
            enabled: AtomicBool::new(std::env::var("INFERO_METAL_PROFILE").is_ok()),
            entries: Mutex::new(HashMap::new()),
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Time `launch` and attribute it to `name`.
    ///
    /// The closure form is the CUDA side's, and for its reason: a kernel that
    /// returns early cannot leave a dangling start event -- or here, an
    /// unmatched clock.
    pub fn time<T>(
        &self,
        name: &'static str,
        stream: Stream,
        launch: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        if !self.enabled() {
            return launch();
        }
        let t = std::time::Instant::now();
        let out = launch()?;
        stream.synchronize()?;
        let ms = t.elapsed().as_secs_f64() * 1e3;

        let mut e = self.entries.lock().unwrap();
        let slot = e.entry(name).or_default();
        slot.launches += 1;
        slot.millis += ms;
        Ok(out)
    }

    pub fn snapshot(&self) -> Vec<(&'static str, Entry)> {
        let mut v: Vec<_> = self
            .entries
            .lock()
            .unwrap()
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect();
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
