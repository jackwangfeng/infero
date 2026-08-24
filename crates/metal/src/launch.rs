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
        self.dev.batch().wait()
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
        let limit = self.func.max_threads_per_group();
        let threads = cfg.block_dim.0 * cfg.block_dim.1.max(1) * cfg.block_dim.2.max(1);
        if threads > limit {
            return Err(anyhow!(
                "{}: {threads} threads a group exceeds this kernel's limit of {limit}",
                self.func.name
            ));
        }
        let dev = self.stream.dev.clone();
        dev.batch().encode(dev.raw_queue(), |enc| {
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

        // `TUILI_METAL_TRACE=<kernel>` reports the geometry a dispatch was
        // actually given, which is the only way to tell a wrong grid from a
        // wrong kernel when the numbers come out partially right.
        static TRACE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        if let Some(want) = TRACE.get_or_init(|| std::env::var("TUILI_METAL_TRACE").ok()) {
            if want == "*" || *want == self.func.name {
                eprintln!(
                    "  dispatch {:<24} groups {:?} threads {:?} smem {} args {}",
                    self.func.name, cfg.grid_dim, cfg.block_dim, cfg.shared_mem_bytes,
                    self.args.len()
                );
            }
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, group);
            Ok(())
        })?;
        // `TUILI_METAL_SYNC` waits for every dispatch, which is the crudest
        // possible ordering guarantee and defeats the batching on purpose. If a
        // run is correct with it and wrong without, the problem is between
        // dispatches rather than inside a kernel.
        static SYNC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *SYNC.get_or_init(|| std::env::var_os("TUILI_METAL_SYNC").is_some()) {
            dev.batch().wait()?;
        }
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

/// One command buffer, open across many dispatches.
///
/// A command buffer per launch costs 17.9 us to create, encode and submit --
/// measured, and a 27B decode step issues about 880 of them, so 15.7 ms of a
/// step went to submission rather than arithmetic. Encoding them all into one
/// buffer moves that to a single submit.
///
/// The ordering guarantee is unchanged and this is why: an encoder created
/// without `MTLDispatchTypeConcurrent` is serial, so Metal inserts the barrier
/// between consecutive dispatches itself. That is the same semantics a CUDA
/// stream gives, and it is what the engine's kernels assume -- every one of them
/// reads a buffer the previous one wrote.
///
/// The buffer is committed by `synchronize`, which every host-visible read
/// already calls, and by `flush` when the dispatch count reaches `CAP` so a
/// long prefill does not hold an unbounded encoder open.
pub(crate) struct Batch {
    open: Mutex<Option<Open>>,
    /// The last committed buffer, for `synchronize` to wait on after a flush.
    last: Mutex<Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
}

struct Open {
    cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    dispatches: usize,
}

/// Dispatches one command buffer holds before it is submitted on its own.
///
/// A decode step is about 880 and wants to be one buffer. A 256-token prefill
/// is far more, and holding every one of those open would keep the GPU idle
/// until the last was encoded -- so the cap exists to start the machine working
/// while the host is still describing what to do.
const CAP: usize = 1024;

// SAFETY: a command encoder is not thread-safe, which is exactly what the mutex
// is for: every path that touches the open one holds it. The scheduler shares
// one `Device` across request threads and they serialise here rather than the
// engine having to be single-threaded.
unsafe impl Send for Batch {}
unsafe impl Sync for Batch {}

impl Default for Batch {
    fn default() -> Self {
        Self {
            open: Mutex::new(None),
            last: Mutex::new(None),
        }
    }
}

impl Batch {
    /// Encode one dispatch, opening a buffer if none is open.
    ///
    /// The closure runs with the encoder held, so it cannot escape and the lock
    /// covers exactly the encoding.
    pub(crate) fn encode(
        &self,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        f: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>) -> Result<()>,
    ) -> Result<()> {
        let mut guard = self.open.lock().unwrap();
        if guard.is_none() {
            let cb = queue
                .commandBuffer()
                .ok_or_else(|| anyhow!("commandBuffer() returned nil"))?;
            let enc = cb
                .computeCommandEncoder()
                .ok_or_else(|| anyhow!("computeCommandEncoder() returned nil"))?;
            *guard = Some(Open {
                cb,
                enc,
                dispatches: 0,
            });
        }
        let open = guard.as_mut().unwrap();
        f(&open.enc)?;
        open.dispatches += 1;
        let full = open.dispatches >= CAP;
        drop(guard);
        if full {
            self.commit()?;
        }
        Ok(())
    }

    /// Encode something that wants the *command buffer* rather than a compute
    /// encoder -- MPS kernels, which bring their own.
    ///
    /// The open compute encoder is ended and a fresh one opened on the same
    /// buffer afterwards, so the MPS pass lands between two compute passes of
    /// one submission. Encoders within a command buffer execute in creation
    /// order, which is the ordering guarantee the caller needs: the GEMM sees
    /// the dequantised weights the dispatch before it wrote, and the dispatch
    /// after it sees the GEMM's output.
    ///
    /// Splitting the encoder is not free -- it ends a pass and starts another --
    /// which is a reason to reach a GEMM at a token count where it earns that,
    /// and not below.
    pub(crate) fn encode_on_buffer(
        &self,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        f: impl FnOnce(&ProtocolObject<dyn MTLCommandBuffer>) -> Result<()>,
    ) -> Result<()> {
        let mut guard = self.open.lock().unwrap();
        let cb = match guard.take() {
            Some(open) => {
                open.enc.endEncoding();
                open.cb
            }
            None => queue
                .commandBuffer()
                .ok_or_else(|| anyhow!("commandBuffer() returned nil"))?,
        };
        f(&cb)?;
        let enc = cb
            .computeCommandEncoder()
            .ok_or_else(|| anyhow!("computeCommandEncoder() returned nil"))?;
        *guard = Some(Open {
            cb,
            enc,
            dispatches: 0,
        });
        Ok(())
    }

    /// End the open encoder and submit, without waiting.
    pub(crate) fn commit(&self) -> Result<()> {
        let Some(open) = self.open.lock().unwrap().take() else {
            return Ok(());
        };
        open.enc.endEncoding();
        open.cb.commit();
        *self.last.lock().unwrap() = Some(open.cb);
        Ok(())
    }

    /// Submit whatever is open and wait for everything submitted.
    pub(crate) fn wait(&self) -> Result<()> {
        self.commit()?;
        let last = self.last.lock().unwrap().take();
        if let Some(cb) = last {
            cb.waitUntilCompleted();
            if let Some(err) = cb.error() {
                return Err(anyhow!("command buffer failed: {err}"));
            }
        }
        Ok(())
    }
}
