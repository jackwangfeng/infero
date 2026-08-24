use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Mutex;

use anyhow::{Result, anyhow};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

use crate::buffer::{Elem, View, ViewMut};
use crate::device::Stream;

/// Grid geometry, with CUDA's field names so the 160 call sites in
/// `tuili-kernels` construct it unchanged.
///
/// `grid_dim` is threadgroups, `block_dim` is threads per threadgroup --
/// which is exactly what `dispatchThreadgroups:threadsPerThreadgroup:` takes,
/// so the translation is the identity and not a reinterpretation. (Metal also
/// offers `dispatchThreads:`, which takes a *total* thread count and does the
/// division itself. Using it would silently change the meaning of every grid
/// this engine computes, so it is deliberately not used.)
#[derive(Debug, Clone, Copy)]
pub struct LaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

/// A compiled kernel, ready to dispatch.
#[derive(Clone)]
pub struct Function {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    name: String,
}

// SAFETY: pipeline states are thread-safe Metal objects; see `device.rs`.
unsafe impl Send for Function {}
unsafe impl Sync for Function {}

impl std::fmt::Debug for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Function({})", self.name)
    }
}

impl Function {
    pub(crate) fn new(
        pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        name: String,
    ) -> Self {
        Self { pipeline, name }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Threads this particular kernel may have in a threadgroup. Lower than the
    /// device maximum when the kernel is register-hungry, which is the closest
    /// Metal gets to answering `cuFuncGetAttribute(MAX_THREADS_PER_BLOCK)`.
    pub fn max_threads_per_group(&self) -> u32 {
        self.pipeline.maxTotalThreadsPerThreadgroup() as u32
    }

    /// cudarc's name for the same ceiling.
    pub fn max_threads_per_block(&self) -> Result<i32> {
        Ok(self.max_threads_per_group() as i32)
    }

    /// The SIMD width this kernel executes at -- 32 on every Apple GPU so far,
    /// and the number the ported shuffle reductions assume.
    pub fn thread_execution_width(&self) -> u32 {
        self.pipeline.threadExecutionWidth() as u32
    }
}

/// One argument, resolved to what Metal needs to bind it.
///
/// Public because it appears in `KernelArg`, but opaque in practice: callers
/// build one by passing a view or a scalar to `.arg()`, never by name.
pub enum Arg {
    /// A window into a buffer: bound with its byte offset.
    Buffer {
        buf: Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
        offset: usize,
    },
    /// A null pointer, for the kernels whose second output is optional.
    ///
    /// CUDA spells this as a `u64` zero in the packed argument array, because
    /// there a pointer *is* eight bytes. Metal binds buffers by index, so the
    /// equivalent is `setBuffer(nil, ...)`, which MSL sees as a null pointer and
    /// `if (hout)` tests exactly as the CUDA kernel does.
    Nil,
    /// A scalar, copied into the command buffer. CUDA passes these in the
    /// packed argument array; Metal's `setBytes:` is the same idea, and both
    /// cap out well above the handful of ints and floats these kernels take.
    Bytes(Vec<u8>),
}

/// Accumulates arguments in call order, then dispatches.
///
/// The `.arg()` chain is the whole reason this crate mirrors cudarc's shape:
/// argument *position* becomes the MSL `[[buffer(n)]]` index, so a kernel
/// signature that lists its parameters in the same order as the CUDA one needs
/// no change at the call site.
pub struct LaunchBuilder {
    stream: Stream,
    func: Function,
    args: Vec<Arg>,
}

impl Stream {
    pub fn launch_builder(&self, f: &Function) -> LaunchBuilder {
        LaunchBuilder {
            stream: self.clone(),
            func: f.clone(),
            args: Vec::with_capacity(8),
        }
    }

    /// Wait for everything submitted so far.
    pub fn synchronize(&self) -> Result<()> {
        let last = self.dev.take_last_commit();
        if let Some(cb) = last {
            cb.waitUntilCompleted();
            if let Some(err) = cb.error() {
                return Err(anyhow!("command buffer failed: {err}"));
            }
        }
        Ok(())
    }
}

impl LaunchBuilder {
    pub fn arg<A: KernelArg>(&mut self, a: &A) -> &mut Self {
        self.args.push(a.to_arg());
        self
    }

    /// Encode and submit.
    ///
    /// `unsafe` to match the CUDA side: nothing here checks that the kernel's
    /// parameter list agrees with the arguments pushed, and a mismatch is
    /// undefined behaviour on the GPU rather than a Rust type error.
    ///
    /// One command buffer per launch, which is the simple thing and not the
    /// fast thing: a batched encoder that holds many dispatches would amortise
    /// the submit, and `engine.rs` already notes that the draft step is
    /// launch-bound. Left for a measurement rather than a guess.
    pub unsafe fn launch(&mut self, cfg: LaunchConfig) -> Result<()> {
        let queue = self.stream.dev.raw_queue();
        let cb = queue
            .commandBuffer()
            .ok_or_else(|| anyhow!("commandBuffer() returned nil"))?;
        let enc = cb
            .computeCommandEncoder()
            .ok_or_else(|| anyhow!("computeCommandEncoder() returned nil"))?;

        enc.setComputePipelineState(&self.func.pipeline);
        for (i, a) in self.args.iter().enumerate() {
            match a {
                Arg::Buffer { buf, offset } => unsafe {
                    enc.setBuffer_offset_atIndex(Some(buf), *offset, i);
                },
                Arg::Nil => unsafe {
                    enc.setBuffer_offset_atIndex(None, 0, i);
                },
                Arg::Bytes(b) => unsafe {
                    let p = NonNull::new(b.as_ptr() as *mut c_void)
                        .ok_or_else(|| anyhow!("null scalar argument"))?;
                    enc.setBytes_length_atIndex(p, b.len(), i);
                },
            }
        }
        if cfg.shared_mem_bytes > 0 {
            // The dynamic shared-memory request. CUDA passes one number at
            // launch; Metal binds it per index, and every ported kernel that
            // wants it declares exactly one `threadgroup` array, so index 0.
            // SAFETY: the length is a byte count the kernel's `threadgroup`
            // declaration is sized against; index 0 because every ported kernel
            // that asks for dynamic shared memory declares exactly one array.
            unsafe {
                enc.setThreadgroupMemoryLength_atIndex(cfg.shared_mem_bytes as usize, 0);
            }
        }

        let grid = MTLSize {
            width: cfg.grid_dim.0 as usize,
            height: cfg.grid_dim.1 as usize,
            depth: cfg.grid_dim.2 as usize,
        };
        let group = MTLSize {
            width: cfg.block_dim.0 as usize,
            height: cfg.block_dim.1 as usize,
            depth: cfg.block_dim.2 as usize,
        };
        let limit = self.func.max_threads_per_group();
        let threads = cfg.block_dim.0 * cfg.block_dim.1.max(1) * cfg.block_dim.2.max(1);
        if threads > limit {
            return Err(anyhow!(
                "{}: {threads} threads a group exceeds this kernel's limit of {limit}",
                self.func.name
            ));
        }

        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, group);
        enc.endEncoding();
        cb.commit();
        // `TUILI_METAL_SYNC` waits for every dispatch, which is the crudest
        // possible ordering guarantee. If a run is correct with it and wrong
        // without, the problem is between command buffers rather than inside a
        // kernel.
        static SYNC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *SYNC.get_or_init(|| std::env::var_os("TUILI_METAL_SYNC").is_some()) {
            cb.waitUntilCompleted();
            if let Some(e) = cb.error() {
                return Err(anyhow!("{}: {e}", self.func.name));
            }
        }
        self.stream.dev.remember_commit(cb);
        Ok(())
    }
}

/// Something that can be bound as a kernel argument.
///
/// Mirrors cudarc's `PushKernelArg`, and for the same reason: the call sites
/// mix device windows and host scalars in one `.arg()` chain.
pub trait KernelArg {
    fn to_arg(&self) -> Arg;
}

impl<T: Elem> KernelArg for View<'_, T> {
    fn to_arg(&self) -> Arg {
        Arg::Buffer {
            buf: self.retained_buf(),
            offset: self.byte_offset(),
        }
    }
}

impl<T: Elem> KernelArg for ViewMut<'_, T> {
    fn to_arg(&self) -> Arg {
        Arg::Buffer {
            buf: self.retained_buf(),
            offset: self.byte_offset(),
        }
    }
}

/// The marker `b_args` passes where a kernel's optional output is absent.
#[derive(Debug, Clone, Copy)]
pub struct NullBuffer;

impl KernelArg for NullBuffer {
    fn to_arg(&self) -> Arg {
        Arg::Nil
    }
}

macro_rules! scalar_arg {
    ($($t:ty),*) => {$(
        impl KernelArg for $t {
            fn to_arg(&self) -> Arg {
                Arg::Bytes(self.to_ne_bytes().to_vec())
            }
        }
    )*};
}
scalar_arg!(i32, u32, f32, i64, u64);

/// Tracks the most recent submission so `synchronize()` has something to wait
/// on. One serial queue means the last buffer finishing implies all of them
/// did, so this is a single slot rather than a list.
#[derive(Default)]
pub(crate) struct LastCommit {
    slot: Mutex<Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
}

// SAFETY: command buffers are not thread-safe to *encode* into, but this only
// ever stores an already-committed one and hands it back to be waited on.
unsafe impl Send for LastCommit {}
unsafe impl Sync for LastCommit {}

impl LastCommit {
    pub(crate) fn set(&self, cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>) {
        *self.slot.lock().unwrap() = Some(cb);
    }

    pub(crate) fn take(&self) -> Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        self.slot.lock().unwrap().take()
    }
}
