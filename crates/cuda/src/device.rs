use std::sync::Arc;

use anyhow::{Context, Result};
use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaStream};

use crate::nvrtc::KernelCache;

/// A single GPU plus the handles every kernel launch needs.
///
/// Cloning is cheap and shares the same context, stream and cuBLAS handle, so
/// the model layers can each hold one.
#[derive(Clone)]
pub struct Device {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: Arc<CudaBlas>,
    kernels: Arc<KernelCache>,
    profile: Arc<crate::Profile>,
    arch: u32,
    sm_count: u32,
    name: String,
}

impl Device {
    pub fn new(ordinal: usize) -> Result<Self> {
        // Must happen before the first cudarc call in the process.
        crate::loader::preload();

        let ctx =
            CudaContext::new(ordinal).with_context(|| format!("opening CUDA device {ordinal}"))?;
        // A created stream rather than `default_stream()`, whose handle is the
        // legacy null stream: CUDA refuses to capture that one, and capturing
        // the decode loop into a graph is worth more than the null stream's
        // implicit synchronisation with everything else in the process.
        let stream = ctx.new_stream().context("creating the compute stream")?;
        let blas = CudaBlas::new(stream.clone()).context("creating cuBLAS handle")?;

        let major = ctx.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )?;
        let minor = ctx.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )?;
        let arch = (major * 10 + minor) as u32;
        let sm_count = ctx.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )? as u32;
        let name = ctx.name().unwrap_or_else(|_| "unknown".into());

        // cudarc otherwise tracks every buffer's last reader and writer and
        // inserts cross-stream events on each launch. Those events belong to
        // streams outside a capture, so they turn any attempt to record the
        // decode loop into a graph into CUDA_ERROR_STREAM_CAPTURE_ISOLATION.
        //
        // Safety: this hands cross-stream ordering back to us, and there is
        // exactly one place that needs it — the CPU-offload weight streaming in
        // `tuili-model`, which already synchronises its copy stream against the
        // compute stream with explicit events rather than relying on this.
        unsafe { ctx.disable_event_tracking() };

        let kernels = Arc::new(KernelCache::new(ctx.clone(), arch)?);
        let profile = Arc::new(crate::Profile::new(&ctx)?);

        tracing::info!(device = %name, arch = arch, sms = sm_count, "cuda device ready");

        Ok(Self {
            ctx,
            stream,
            blas: Arc::new(blas),
            kernels,
            profile,
            arch,
            sm_count,
            name,
        })
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn blas(&self) -> &CudaBlas {
        &self.blas
    }

    pub fn kernels(&self) -> &KernelCache {
        &self.kernels
    }

    /// Per-kernel timing, active only under `TUILI_PROFILE`.
    pub fn profile(&self) -> &crate::Profile {
        &self.profile
    }

    /// Streaming multiprocessors on this device.
    ///
    /// A kernel's grid has to be at least this many blocks before the GPU is
    /// even nominally busy, and the tile shapes in `tuili-kernels` were sized
    /// against a 48-SM card. On a 188-SM one the same matrix produces the same
    /// number of blocks and leaves most of the machine idle, so anything that
    /// reports throughput should report this too.
    pub fn sm_count(&self) -> u32 {
        self.sm_count
    }

    /// Compute capability as a two-digit number, e.g. `86` for an RTX A4000.
    pub fn arch(&self) -> u32 {
        self.arch
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().context("stream sync")?;
        Ok(())
    }

    /// Free and total device memory in bytes.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        Ok(cudarc::driver::result::mem_get_info()?)
    }
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("name", &self.name)
            .field("arch", &self.arch)
            .finish()
    }
}
