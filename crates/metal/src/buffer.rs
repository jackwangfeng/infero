use std::marker::PhantomData;
use std::ops::{Bound, RangeBounds};
use std::sync::Arc;

use anyhow::{Result, anyhow};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::device::Stream;

/// An element a kernel can hold. Blanket-implemented for the plain-old-data
/// types the engine moves: `f32`, `f16`, `i32`, `u32`, `u8`.
///
/// The bound exists to keep `transmute` honest -- reinterpreting a byte buffer
/// as f16 is sound because f16 has no invalid bit patterns, and the trait is
/// where that claim is recorded.
pub unsafe trait Elem: Copy + 'static {}
unsafe impl Elem for f32 {}
unsafe impl Elem for half::f16 {}
unsafe impl Elem for i32 {}
unsafe impl Elem for u32 {}
unsafe impl Elem for u8 {}
unsafe impl Elem for i8 {}

struct Raw {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    bytes: usize,
}

// SAFETY: `MTLBuffer` is thread-safe; see `device.rs`. Unified memory means
// `contents()` is a plain host pointer into the same physical pages the GPU
// reads, so there is no staging buffer and no separate host copy to keep
// coherent -- which is the one place this backend is genuinely simpler than the
// CUDA one.
unsafe impl Send for Raw {}
unsafe impl Sync for Raw {}

/// An owned device allocation of `len` elements.
pub struct Buf<T: Elem> {
    raw: Arc<Raw>,
    len: usize,
    _t: PhantomData<T>,
}

impl<T: Elem> Buf<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn slice<R: RangeBounds<usize>>(&self, r: R) -> View<'_, T> {
        let (off, len) = resolve(&r, self.len);
        View {
            raw: &self.raw,
            off,
            len,
            _t: PhantomData,
        }
    }

    pub fn slice_mut<R: RangeBounds<usize>>(&mut self, r: R) -> ViewMut<'_, T> {
        let (off, len) = resolve(&r, self.len);
        ViewMut {
            raw: &self.raw,
            off,
            len,
            _t: PhantomData,
        }
    }

    pub fn as_view(&self) -> View<'_, T> {
        self.slice(..)
    }

    pub fn as_view_mut(&mut self) -> ViewMut<'_, T> {
        self.slice_mut(..)
    }

    /// Reinterpret the allocation as another element type.
    ///
    /// # Safety
    /// The caller guarantees the bytes are a valid `U` sequence. Every `Elem`
    /// here is plain data with no invalid bit patterns, so the real obligation
    /// is only that `n * size_of::<U>()` fits.
    pub unsafe fn transmute<U: Elem>(&self, n: usize) -> Result<View<'_, U>> {
        if n * std::mem::size_of::<U>() > self.raw.bytes {
            return Err(anyhow!(
                "transmute to {n} x {} exceeds the {} byte allocation",
                std::mem::size_of::<U>(),
                self.raw.bytes
            ));
        }
        Ok(View {
            raw: &self.raw,
            off: 0,
            len: n,
            _t: PhantomData,
        })
    }

    /// Read the whole allocation back to the host.
    ///
    /// Unified memory makes this a `memcpy` from a pointer the GPU also sees,
    /// so the only thing that has to happen first is that the work which wrote
    /// it has completed -- which is the caller's business, exactly as it is on
    /// CUDA.
    pub fn to_vec(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.len);
        // SAFETY: `contents()` is valid for `bytes` and `len * size_of::<T>()`
        // was checked at allocation.
        unsafe {
            let src = self.raw.buf.contents().as_ptr() as *const T;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), self.len);
            out.set_len(self.len);
        }
        out
    }
}

macro_rules! view_common {
    ($name:ident) => {
        impl<'a, T: Elem> $name<'a, T> {
            pub fn len(&self) -> usize {
                self.len
            }

            pub fn is_empty(&self) -> bool {
                self.len == 0
            }

            /// Byte offset for `setBuffer:offset:atIndex:`.
            pub(crate) fn byte_offset(&self) -> usize {
                self.off * std::mem::size_of::<T>()
            }

            pub(crate) fn raw_buf(&self) -> &ProtocolObject<dyn MTLBuffer> {
                &self.raw.buf
            }

            /// A retained handle for the launch builder, which outlives the
            /// view: arguments are collected first and bound at dispatch.
            pub(crate) fn retained_buf(
                &self,
            ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
                self.raw.buf.clone()
            }

            pub fn slice<R: RangeBounds<usize>>(&self, r: R) -> View<'a, T> {
                let (off, len) = resolve(&r, self.len);
                View {
                    raw: self.raw,
                    off: self.off + off,
                    len,
                    _t: PhantomData,
                }
            }
        }
    };
}

/// A read-only window into an allocation. Mirrors `cudarc::CudaView`.
#[derive(Clone, Copy)]
pub struct View<'a, T: Elem> {
    raw: &'a Arc<Raw>,
    off: usize,
    len: usize,
    _t: PhantomData<T>,
}

/// A writable window. Mirrors `cudarc::CudaViewMut`.
pub struct ViewMut<'a, T: Elem> {
    raw: &'a Arc<Raw>,
    off: usize,
    len: usize,
    _t: PhantomData<T>,
}

view_common!(View);
view_common!(ViewMut);

impl<'a, T: Elem> ViewMut<'a, T> {
    pub fn slice_mut<R: RangeBounds<usize>>(&mut self, r: R) -> ViewMut<'_, T> {
        let (off, len) = resolve(&r, self.len);
        ViewMut {
            raw: self.raw,
            off: self.off + off,
            len,
            _t: PhantomData,
        }
    }

    pub fn as_view(&self) -> View<'_, T> {
        View {
            raw: self.raw,
            off: self.off,
            len: self.len,
            _t: PhantomData,
        }
    }
}

fn resolve<R: RangeBounds<usize>>(r: &R, cap: usize) -> (usize, usize) {
    let start = match r.start_bound() {
        Bound::Included(&s) => s,
        Bound::Excluded(&s) => s + 1,
        Bound::Unbounded => 0,
    };
    let end = match r.end_bound() {
        Bound::Included(&e) => e + 1,
        Bound::Excluded(&e) => e,
        Bound::Unbounded => cap,
    };
    let end = end.min(cap);
    (start, end.saturating_sub(start))
}

impl Stream<'_> {
    /// A zeroed allocation. `MTLResourceOptions::StorageModeShared` puts it in
    /// the unified pool, which is the only sensible mode on Apple Silicon: a
    /// private-mode buffer would need a blit to read back and buy nothing,
    /// because there is no separate device memory to be private from.
    pub fn alloc_zeros<T: Elem>(&self, n: usize) -> Result<Buf<T>> {
        let bytes = n * std::mem::size_of::<T>();
        // Metal rejects a zero-length buffer; the engine does ask for empty
        // scratch in a few places, so round up rather than fail.
        let alloc = bytes.max(std::mem::size_of::<T>());
        let buf = self
            .dev
            .raw()
            .newBufferWithLength_options(alloc, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("allocating {} MiB failed", alloc >> 20))?;
        // SAFETY: `contents()` is valid for `alloc` bytes and shared storage is
        // host-visible.
        unsafe {
            std::ptr::write_bytes(buf.contents().as_ptr() as *mut u8, 0, alloc);
        }
        Ok(Buf {
            raw: Arc::new(Raw { buf, bytes: alloc }),
            len: n,
            _t: PhantomData,
        })
    }

    /// Host to device. A `memcpy` into unified memory.
    pub fn memcpy_stod<T: Elem>(&self, src: &[T]) -> Result<Buf<T>> {
        let mut b = self.alloc_zeros::<T>(src.len())?;
        self.copy_into(&mut b.as_view_mut(), src)?;
        Ok(b)
    }

    pub fn copy_into<T: Elem>(&self, dst: &mut ViewMut<'_, T>, src: &[T]) -> Result<()> {
        if src.len() > dst.len() {
            return Err(anyhow!(
                "copying {} elements into a {} element window",
                src.len(),
                dst.len()
            ));
        }
        // SAFETY: bounds checked above; shared storage is host-visible.
        unsafe {
            let p = dst.raw_buf().contents().as_ptr() as *mut u8;
            let p = p.add(dst.byte_offset()) as *mut T;
            std::ptr::copy_nonoverlapping(src.as_ptr(), p, src.len());
        }
        Ok(())
    }

    /// Device to host.
    pub fn memcpy_dtov<T: Elem>(&self, src: &View<'_, T>) -> Result<Vec<T>> {
        let mut out = Vec::with_capacity(src.len());
        // SAFETY: as above.
        unsafe {
            let p = src.raw_buf().contents().as_ptr() as *const u8;
            let p = p.add(src.byte_offset()) as *const T;
            std::ptr::copy_nonoverlapping(p, out.as_mut_ptr(), src.len());
            out.set_len(src.len());
        }
        Ok(out)
    }
}
