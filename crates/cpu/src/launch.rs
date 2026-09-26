//! Kernel "launch", which on this backend is a plain function call.
//!
//! This crate does not know what any kernel computes -- unlike CUDA (NVRTC
//! compiles real `.cu` source) and Metal (compiles real `.metal` source),
//! there is no third shading language to write GGML-style dequantization and
//! attention math in, so nothing here parses `src` at all. Instead
//! `Function::launch` looks its own `name` up in a dispatcher that
//! `infero-kernels` registers exactly once, at process start, via
//! [`set_dispatcher`] -- see `crates/kernels/src/cpu/mod.rs`'s own doc
//! comment for why the numerical implementations live there rather than
//! here (this crate cannot depend on that one; that one already depends on
//! this one).
//!
//! Because there is no async queue to submit into, a "launch" runs to
//! completion, on the calling thread, before `launch()` returns. That is
//! what makes `Device::synchronize` a no-op and is the one respect in which
//! this backend is simpler than either GPU one.

use anyhow::{Result, anyhow};

use crate::buffer::{Elem, View, ViewMut};
use crate::device::Stream;

/// Grid geometry, with CUDA's field names so the ~160 call sites in
/// `infero-kernels` construct it unchanged. Read by the registered
/// dispatcher, not by this crate -- most CPU kernel bodies only care about
/// `grid_dim` (how many independent rows/items there are to loop, in
/// parallel via `rayon`, over) and ignore `block_dim`/`shared_mem_bytes`,
/// which describe a GPU thread-block's internal shape this backend has no
/// equivalent for.
#[derive(Debug, Clone, Copy)]
pub struct LaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

/// A "compiled kernel" -- really just the name a dispatch will be looked up
/// by. `src`/`module` (see `get`, below) exist only so call sites match the
/// CUDA/Metal shape; this backend never reads them.
#[derive(Clone, Debug)]
pub struct Function {
    name: String,
}

impl Function {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// This backend has no per-kernel register/occupancy ceiling to report;
    /// answering with `max_threads_per_group` (the same number every kernel
    /// gets) keeps callers that only use this for a sanity `assert!` happy
    /// without inventing a fake one.
    pub fn max_threads_per_group(&self) -> u32 {
        1024
    }

    pub fn max_threads_per_block(&self) -> Result<i32> {
        Ok(self.max_threads_per_group() as i32)
    }

    pub fn thread_execution_width(&self) -> u32 {
        32
    }
}

/// A named "kernel cache" -- in practice just a name pass-through, since
/// there is nothing to compile. Mirrors `infero_metal::msl::Modules`'s own
/// `get` signature so `dev.kernels().get(module, src, name)` call sites need
/// no change.
pub struct Modules;

impl Modules {
    pub fn get(&self, _module: &'static str, _src: &str, name: &str) -> Result<Function> {
        Ok(Function { name: name.to_string() })
    }
}

impl crate::device::Device {
    pub fn kernels(&self) -> Modules {
        Modules
    }
}

/// One argument, resolved to what a native dispatch needs to reconstruct a
/// typed slice or scalar. `Buffer` carries a raw pointer plus enough shape
/// information (`elem_size`, `len`) for the dispatcher to rebuild
/// `std::slice::from_raw_parts[_mut]` -- sound exactly when the dispatcher's
/// own kernel-specific code reconstructs it as the *same* element type the
/// real CUDA kernel this is standing in for takes at that argument
/// position, which is the same "argument order and types must agree with
/// the kernel's real parameter list" contract `LaunchBuilder::launch`
/// already carries on every other backend -- unchecked, and undefined
/// behaviour on a mismatch there too.
pub enum Arg {
    Buffer { ptr: *mut u8, elem_size: usize, len: usize },
    Nil,
    Bytes(Vec<u8>),
}

impl Arg {
    /// # Safety
    /// The caller must know this argument really is a buffer of `T`,
    /// matching the position it was bound at.
    pub unsafe fn as_slice<T: Elem>(&self) -> &[T] {
        match self {
            Arg::Buffer { ptr, len, elem_size } => {
                debug_assert_eq!(*elem_size, std::mem::size_of::<T>(), "element size mismatch");
                unsafe { std::slice::from_raw_parts(*ptr as *const T, *len) }
            }
            _ => panic!("expected a buffer argument"),
        }
    }

    /// # Safety
    /// As `as_slice`, plus: this must be the only live view of this window
    /// for the duration of the borrow (the same aliasing rule every other
    /// backend's `ViewMut` already enforces at the type level, temporarily
    /// unenforced here by the type erasure `Arg` itself is for).
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn as_mut_slice<T: Elem>(&self) -> &mut [T] {
        match self {
            Arg::Buffer { ptr, len, elem_size } => {
                debug_assert_eq!(*elem_size, std::mem::size_of::<T>(), "element size mismatch");
                unsafe { std::slice::from_raw_parts_mut(*ptr as *mut T, *len) }
            }
            _ => panic!("expected a buffer argument"),
        }
    }

    /// Whether this is the `NullBuffer` marker for a kernel's absent
    /// optional output.
    pub fn is_nil(&self) -> bool {
        matches!(self, Arg::Nil)
    }

    /// # Safety
    /// The caller must know this argument really is a scalar `T`.
    pub unsafe fn as_scalar<T: Copy>(&self) -> T {
        match self {
            Arg::Bytes(b) => {
                debug_assert_eq!(b.len(), std::mem::size_of::<T>(), "scalar size mismatch");
                unsafe { std::ptr::read_unaligned(b.as_ptr() as *const T) }
            }
            _ => panic!("expected a scalar argument"),
        }
    }
}

/// The function every backend registers exactly once: given a kernel name,
/// its bound arguments in call order, and the launch geometry, do the real
/// work. `infero-kernels`'s `crates/kernels/src/cpu/mod.rs` is the one real
/// implementation of this signature.
pub type Dispatch = fn(name: &str, args: &[Arg], cfg: LaunchConfig) -> Result<()>;

static DISPATCH: std::sync::OnceLock<Dispatch> = std::sync::OnceLock::new();

/// Register the kernel dispatcher. Idempotent: a second call with the same
/// function pointer is a no-op; a second call with a *different* one is a
/// caller bug (there is exactly one real dispatcher in this workspace) and
/// panics rather than silently keeping whichever registered first.
pub fn set_dispatcher(f: Dispatch) {
    match DISPATCH.set(f) {
        Ok(()) => {}
        Err(_) if DISPATCH.get() == Some(&f) => {}
        Err(_) => panic!("infero_cpu::set_dispatcher called twice with different dispatchers"),
    }
}

/// Accumulates arguments in call order, then dispatches.
pub struct LaunchBuilder {
    func: Function,
    args: Vec<Arg>,
}

impl Stream {
    pub fn launch_builder(&self, f: &Function) -> LaunchBuilder {
        LaunchBuilder { func: f.clone(), args: Vec::with_capacity(8) }
    }
}

impl LaunchBuilder {
    pub fn arg<A: KernelArg>(&mut self, a: &A) -> &mut Self {
        self.args.push(a.to_arg());
        self
    }

    /// Run the kernel. `unsafe` to match the GPU backends: nothing here
    /// checks that the arguments pushed agree with what the registered
    /// dispatcher's `name` branch expects to find at each position.
    pub unsafe fn launch(&mut self, cfg: LaunchConfig) -> Result<()> {
        let f = DISPATCH
            .get()
            .ok_or_else(|| anyhow!("no CPU kernel dispatcher registered (infero_cpu::set_dispatcher)"))?;
        f(&self.func.name, &self.args, cfg)
    }
}

/// Something that can be bound as a kernel argument. Mirrors cudarc's
/// `PushKernelArg`.
pub trait KernelArg {
    fn to_arg(&self) -> Arg;
}

impl<T: Elem> KernelArg for View<'_, T> {
    fn to_arg(&self) -> Arg {
        Arg::Buffer { ptr: self.raw_ptr(), elem_size: std::mem::size_of::<T>(), len: self.len() }
    }
}

impl<T: Elem> KernelArg for ViewMut<'_, T> {
    fn to_arg(&self) -> Arg {
        Arg::Buffer { ptr: self.raw_ptr(), elem_size: std::mem::size_of::<T>(), len: self.len() }
    }
}

/// A whole buffer, bound the same as a view over all of it -- mirrors
/// cudarc's `PushKernelArg` for `CudaSlice<T>`, which some call sites (e.g.
/// `gdn.rs`'s `gdn_scan_split_delta_rule`) pass directly rather than through
/// `.as_view()`.
impl<T: Elem> KernelArg for crate::buffer::Buf<T> {
    fn to_arg(&self) -> Arg {
        self.as_view().to_arg()
    }
}

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
