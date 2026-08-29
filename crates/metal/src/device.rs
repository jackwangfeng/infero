use std::sync::Arc;

use anyhow::{Result, anyhow};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice};

use crate::launch::Batch;
use crate::profile::Profile;
use crate::msl::Modules;

/// What the host is allowed to assume about this GPU.
///
/// The CUDA side answers the same questions from `arch()`: `>= 80` means the
/// integer tensor cores exist, `>= 89` FP8, `>= 90` TMA. Apple GPUs have none
/// of those, so every capability that gates a hand-written tensor-core kernel
/// reads false here and the dispatch in `infero-model` routes around it through
/// the path it already has for pre-Ampere cards.
#[derive(Debug, Clone, Copy)]
pub struct Caps {
    /// `mmq.cu`: the integer tensor-core GEMM. False on Apple: there is no
    /// `mma.sync` equivalent that takes s8 operands, and `simdgroup_matrix` is
    /// a different shape with a different fragment layout.
    pub int_tensor_gemm: bool,
    /// `fp8.cu`: block-scaled FP8 mat-vec. False on Apple: no FP8 matmul.
    pub fp8: bool,
    /// `cp.async.bulk.tensor`. False on Apple: no equivalent global->local DMA.
    pub tma: bool,
    /// Threads in a SIMD group. 32 on Apple, same as a CUDA warp -- which is
    /// why the shuffle reductions port directly.
    pub simd_width: u32,
    /// Ceiling for a threadgroup, from the pipeline state.
    pub max_threads_per_group: u32,
    /// `recommendedMaxWorkingSetSize`, the practical VRAM budget. On a 36 GB
    /// M4 Max this is about 27 GB unless `iogpu.wired_limit_mb` is raised.
    pub working_set_bytes: u64,
}

/// A Metal device, its queue, and the compiled-module cache.
///
/// Cloning is cheap and shares everything, matching `infero_cuda::Device`.
#[derive(Clone)]
pub struct Device {
    inner: Arc<Inner>,
}

struct Inner {
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    modules: Modules,
    caps: Caps,
    name: String,
    cores: u32,
    batch: Batch,
    gemm: crate::gemm::GemmCache,
    profile: Profile,
}

// SAFETY: `MTLDevice`, `MTLCommandQueue` and `MTLBuffer` are documented as safe
// to use from multiple threads; it is the command *encoders* that are not, and
// those never escape the launch builder that creates them. The scheduler shares
// one `Device` across the request threads, so without this the whole engine
// would have to be single-threaded for a reason Metal does not actually impose.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Device {
    /// The system default GPU. Metal has no device ordinals the way CUDA does --
    /// an Apple Silicon machine has exactly one -- so `ordinal` is accepted and
    /// checked rather than ignored, to keep the caller's `--device N` honest.
    pub fn new(ordinal: usize) -> Result<Self> {
        if ordinal != 0 {
            return Err(anyhow!(
                "this machine has one GPU; --device {ordinal} does not exist"
            ));
        }
        let raw = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| anyhow!("no Metal device: MTLCreateSystemDefaultDevice returned nil"))?;
        let queue = raw
            .newCommandQueue()
            .ok_or_else(|| anyhow!("newCommandQueue failed"))?;

        let name = raw.name().to_string();
        let caps = Caps {
            int_tensor_gemm: false,
            fp8: false,
            tma: false,
            simd_width: 32,
            // The real ceiling is per-pipeline (a kernel with many registers
            // gets less), so this is the device-wide maximum and any kernel
            // that needs the true number asks its pipeline state.
            max_threads_per_group: 1024,
            working_set_bytes: raw.recommendedMaxWorkingSetSize(),
        };

        Ok(Self {
            inner: Arc::new(Inner {
                modules: Modules::new(raw.clone()),
                raw,
                queue,
                caps,
                name,
                cores: std::env::var("INFERO_METAL_CORES")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(32),
                batch: Batch::default(),
                gemm: Default::default(),
                profile: Profile::new(),
            }),
        })
    }

    pub fn caps(&self) -> &Caps {
        &self.inner.caps
    }

    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// The compiled-module cache, keyed the same way the NVRTC one is: a module
    /// name plus its source, so a source change is a cache miss rather than a
    /// stale hit.
    pub fn kernels(&self) -> &Modules {
        &self.inner.modules
    }

    /// The single command queue. Named `stream()` to match the CUDA side; the
    /// ordering guarantee is the same, because dispatches encoded serially into
    /// one queue observe each other's writes.
    /// A handle for submitting work.
    ///
    /// Owned rather than borrowed, and cheap: `Device` is an `Arc` inside, so
    /// this is one relaxed increment. Borrowing instead made `dev.stream()`
    /// hold a shared borrow of the `Model` that owns the device, which collides
    /// with the `&mut self` closure the graph-capture path builds -- a conflict
    /// cudarc does not have because it hands back a reference to an `Arc` the
    /// caller already owns.
    pub fn stream(&self) -> Stream {
        Stream { dev: self.clone() }
    }

    pub(crate) fn raw(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.inner.raw
    }

    pub(crate) fn raw_queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.inner.queue
    }

    pub(crate) fn batch(&self) -> &Batch {
        &self.inner.batch
    }

    pub(crate) fn gemm_cache(&self) -> &crate::gemm::GemmCache {
        &self.inner.gemm
    }

    pub fn profile(&self) -> &Profile {
        &self.inner.profile
    }

    /// Parallel units, for the grid heuristics that aim at "enough blocks to
    /// fill the machine".
    ///
    /// Metal exposes no core count -- there is no `MTLDevice` property for it,
    /// and the CUDA `multiProcessorCount` it stands in for has no counterpart.
    /// So this is a *default*, overridable with `INFERO_METAL_CORES`, and the
    /// eight call sites that read it all use it the same way: `sm_count() * k`
    /// as a target block count. Being wrong by a factor of two changes a tile
    /// count, not an answer.
    pub fn sm_count(&self) -> u32 {
        self.inner.cores
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream().synchronize()
    }

    /// The context the CUDA path hangs streams, events and pinned allocations
    /// off. Metal has no separate context object, so this is a handle whose
    /// only job is to carry those constructors -- and to refuse the two of
    /// them that have no counterpart. See `compat.rs`.
    pub fn context(&self) -> crate::compat::Context {
        crate::compat::Context
    }

    /// Free and total bytes, in `cuMemGetInfo`'s order.
    ///
    /// "Total" is the working-set budget rather than installed RAM: on unified
    /// memory the GPU shares the machine's DRAM with everything else, so the
    /// number that bounds an allocation is the one Metal recommends, not the 36
    /// GB the box has. "Free" subtracts what this device has already been
    /// allocated, which `currentAllocatedSize` reports.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let total = self.inner.caps.working_set_bytes as usize;
        let used = self.inner.raw.currentAllocatedSize();
        Ok((total.saturating_sub(used), total))
    }

    /// Bytes the GPU can hold before the OS starts paging.
    pub fn working_set_bytes(&self) -> u64 {
        self.inner.caps.working_set_bytes
    }
}

/// A handle for submitting work. Borrowed rather than owned so that the call
/// sites read `dev.stream().launch_builder(&f)` exactly as they do on CUDA.
#[derive(Clone)]
pub struct Stream {
    pub(crate) dev: Device,
}

impl Stream {
    pub fn device(&self) -> &Device {
        &self.dev
    }

    pub fn wait(&self, _e: &crate::compat::Event) -> anyhow::Result<()> {
        anyhow::bail!("events are a CUDA-only path on this backend")
    }

    pub fn begin_capture(&self, _mode: crate::compat::CaptureMode) -> anyhow::Result<()> {
        anyhow::bail!("graph capture is a CUDA-only path on this backend")
    }

    pub fn end_capture(
        &self,
        _flags: crate::compat::GraphFlags,
    ) -> anyhow::Result<Option<crate::compat::Graph>> {
        anyhow::bail!("graph capture is a CUDA-only path on this backend")
    }
}
