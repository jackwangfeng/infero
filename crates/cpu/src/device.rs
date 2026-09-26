use std::sync::Arc;

use anyhow::{Result, anyhow};

use crate::profile::Profile;

/// What the host is allowed to assume about this "device".
///
/// Every capability that gates a hand-written tensor-core-style kernel reads
/// false here, the same way it does on Metal -- there is no integer
/// tensor-core GEMM, no FP8 matmul, no TMA on a CPU, so `infero-model`'s
/// dispatch routes around all of them through the paths it already has for
/// hardware that lacks them.
#[derive(Debug, Clone, Copy)]
pub struct Caps {
    pub int_tensor_gemm: bool,
    pub fp8: bool,
    pub tma: bool,
    /// Not a real SIMD width -- this backend has no warp/simdgroup concept.
    /// Kept at 32 only because a couple of shared reduction-size constants
    /// read it; nothing here dispatches per-lane.
    pub simd_width: u32,
    pub max_threads_per_group: u32,
    pub working_set_bytes: u64,
}

/// A "device": really just a thread-count and a kernel dispatcher. Cloning is
/// cheap and shares everything, matching `infero_cuda::Device`/
/// `infero_metal::Device`.
#[derive(Clone)]
pub struct Device {
    inner: Arc<Inner>,
}

struct Inner {
    caps: Caps,
    threads: usize,
    profile: Profile,
}

impl Device {
    /// `ordinal` is accepted and checked rather than ignored, matching
    /// Metal's own reasoning: this backend has exactly one "device" (the
    /// host's own CPU), so `--device N` for `N != 0` is a caller error worth
    /// surfacing rather than silently ignoring.
    pub fn new(ordinal: usize) -> Result<Self> {
        if ordinal != 0 {
            return Err(anyhow!("this backend has one device; --device {ordinal} does not exist"));
        }
        let threads = std::env::var("INFERO_CPU_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
        // Approximate: total RAM this process could plausibly grow into.
        // There is no VRAM-style hard ceiling on this backend -- it comes
        // from `/proc/meminfo` when that exists, and a generous guess
        // otherwise (macOS/other Unix, where a small model is the whole
        // point of this backend and precision here buys little).
        let (_, total) = mem_info_bytes();
        Ok(Self {
            inner: Arc::new(Inner {
                caps: Caps {
                    int_tensor_gemm: false,
                    fp8: false,
                    tma: false,
                    simd_width: 32,
                    max_threads_per_group: 1024,
                    working_set_bytes: total,
                },
                threads,
                profile: Profile::new(),
            }),
        })
    }

    pub fn caps(&self) -> &Caps {
        &self.inner.caps
    }

    pub fn name(&self) -> &str {
        "CPU"
    }

    pub fn stream(&self) -> Stream {
        Stream { dev: self.clone() }
    }

    pub fn profile(&self) -> &Profile {
        &self.inner.profile
    }

    /// Threads to spend on a data-parallel loop -- this backend's answer to
    /// `multiProcessorCount`/`sm_count`. `INFERO_CPU_THREADS` overrides the
    /// autodetected count, the same escape hatch `INFERO_METAL_CORES` is on
    /// the Metal side.
    pub fn sm_count(&self) -> u32 {
        self.inner.threads as u32
    }

    /// `attn_backend::HardwareCaps`'s compute-capability floor, read as `0`
    /// here: there is no compute capability to report, and `0` makes every
    /// `HardwareCaps::at_least(major, minor)` check false, which is the
    /// right answer -- the CUDA-arch-gated paths it guards (MMA tiles, FA2
    /// eligibility) don't apply to this backend regardless.
    pub fn arch(&self) -> u32 {
        0
    }

    pub fn synchronize(&self) -> Result<()> {
        // Every kernel already ran to completion inline by the time
        // `Function::launch` returned -- see `launch.rs`'s own doc comment.
        Ok(())
    }

    pub fn context(&self) -> crate::compat::Context {
        crate::compat::Context
    }

    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let (free, total) = mem_info_bytes();
        Ok((free as usize, total as usize))
    }

    pub fn working_set_bytes(&self) -> u64 {
        self.inner.caps.working_set_bytes
    }
}

/// (free, total) physical memory in bytes, best-effort.
fn mem_info_bytes() -> (u64, u64) {
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        let mut total = None;
        let mut free = None;
        let mut cached = None;
        for line in s.lines() {
            let mut parts = line.split_whitespace();
            let Some(key) = parts.next() else { continue };
            let Some(kib) = parts.next().and_then(|v| v.parse::<u64>().ok()) else { continue };
            match key {
                "MemTotal:" => total = Some(kib * 1024),
                "MemAvailable:" => free = Some(kib * 1024),
                "MemFree:" => cached = cached.or(Some(kib * 1024)),
                _ => {}
            }
        }
        if let Some(t) = total {
            return (free.or(cached).unwrap_or(t / 2), t);
        }
    }
    // No `/proc/meminfo` (non-Linux): a conservative fixed guess rather than
    // a hard failure -- this backend's whole point is running a small model
    // somewhere without a real accelerator, and the number here only feeds
    // an advisory KV-pool sizing heuristic, never a hard allocation ceiling.
    let guess = 8u64 << 30;
    (guess / 2, guess)
}

/// A handle for submitting work. Real work happens synchronously inside
/// `Function::launch`, so this carries nothing beyond the device handle --
/// its only job is to give call sites `dev.stream().launch_builder(&f)` and
/// `dev.stream().alloc_zeros(...)`, matching the CUDA/Metal spelling.
#[derive(Clone)]
pub struct Stream {
    pub(crate) dev: Device,
}

impl Stream {
    pub fn device(&self) -> &Device {
        &self.dev
    }

    pub fn synchronize(&self) -> Result<()> {
        self.dev.synchronize()
    }

    pub fn wait(&self, _e: &crate::compat::Event) -> Result<()> {
        anyhow::bail!("events are a CUDA-only path on this backend")
    }

    pub fn begin_capture(&self, _mode: crate::compat::CaptureMode) -> Result<()> {
        anyhow::bail!("graph capture is a CUDA-only path on this backend")
    }

    pub fn end_capture(&self, _flags: crate::compat::GraphFlags) -> Result<Option<crate::compat::Graph>> {
        anyhow::bail!("graph capture is a CUDA-only path on this backend")
    }
}
