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
// The device sampler draws in f64: a uniform in f32 quantises visibly at the
// tail of a 248320-wide nucleus.
unsafe impl Elem for f64 {}

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

    /// Two non-overlapping windows, split at `mid`.
    pub fn split_at(&self, mid: usize) -> (View<'_, T>, View<'_, T>) {
        (self.slice(..mid), self.slice(mid..))
    }

    pub fn split_at_mut(&mut self, mid: usize) -> (ViewMut<'_, T>, ViewMut<'_, T>) {
        let (lo, hi) = (
            resolve(&(..mid), self.len),
            resolve(&(mid..), self.len),
        );
        (
            ViewMut { raw: &self.raw, off: lo.0, len: lo.1, _t: PhantomData },
            ViewMut { raw: &self.raw, off: hi.0, len: hi.1, _t: PhantomData },
        )
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
    /// Unified memory makes this a `memcpy` from a pointer the GPU also sees.
    /// It does **not** synchronise -- there is no stream here to synchronise on
    /// -- so the caller must have waited. `Stream::memcpy_dtoh` is the ordered
    /// one and is what the engine takes; this is for tests and examples that
    /// have just called `synchronize` themselves.
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

            /// Reinterpret this window as another element type.
            ///
            /// # Safety
            /// As `Buf::transmute`: every `Elem` is plain data, so the real
            /// obligation is that the byte count fits.
            pub unsafe fn transmute<U: Elem>(&self, n: usize) -> Result<View<'a, U>> {
                let want = n * std::mem::size_of::<U>();
                let have = self.len * std::mem::size_of::<T>();
                if want > have {
                    return Err(anyhow!(
                        "transmute to {n} x {} exceeds this {have} byte window",
                        std::mem::size_of::<U>()
                    ));
                }
                // The element offset scales with the size change; a window that
                // does not start on a `U` boundary is a caller error and is
                // rejected rather than silently rounded.
                let byte_off = self.off * std::mem::size_of::<T>();
                if byte_off % std::mem::size_of::<U>() != 0 {
                    return Err(anyhow!(
                        "window starts {byte_off} bytes in, which is not a multiple of {}",
                        std::mem::size_of::<U>()
                    ));
                }
                Ok(View {
                    raw: self.raw,
                    off: byte_off / std::mem::size_of::<U>(),
                    len: n,
                    _t: PhantomData,
                })
            }

            /// Two non-overlapping windows, split at `mid`.
            pub fn split_at(&self, mid: usize) -> (View<'a, T>, View<'a, T>) {
                (self.slice(..mid), self.slice(mid..))
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

    pub fn split_at_mut(&mut self, mid: usize) -> (ViewMut<'_, T>, ViewMut<'_, T>) {
        let mid = mid.min(self.len);
        (
            ViewMut { raw: self.raw, off: self.off, len: mid, _t: PhantomData },
            ViewMut {
                raw: self.raw,
                off: self.off + mid,
                len: self.len - mid,
                _t: PhantomData,
            },
        )
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

/// Anything a host-to-device copy can write into.
///
/// cudarc reaches the same generality through `DevicePtrMut`; the engine's call
/// sites pass an owned `Buf` in some places and a `ViewMut` in others, and both
/// spellings should work without the caller inserting a conversion.
pub trait CopyDst<T: Elem> {
    fn as_dst(&mut self) -> ViewMut<'_, T>;
}

impl<T: Elem> CopyDst<T> for Buf<T> {
    fn as_dst(&mut self) -> ViewMut<'_, T> {
        self.as_view_mut()
    }
}

impl<T: Elem> CopyDst<T> for ViewMut<'_, T> {
    fn as_dst(&mut self) -> ViewMut<'_, T> {
        ViewMut {
            raw: self.raw,
            off: self.off,
            len: self.len,
            _t: PhantomData,
        }
    }
}

/// Anything a device-to-host copy can read from.
pub trait CopySrc<T: Elem> {
    fn as_src(&self) -> View<'_, T>;
}

impl<T: Elem> CopySrc<T> for Buf<T> {
    fn as_src(&self) -> View<'_, T> {
        self.as_view()
    }
}

impl<T: Elem> CopySrc<T> for View<'_, T> {
    fn as_src(&self) -> View<'_, T> {
        *self
    }
}

impl<T: Elem> CopySrc<T> for ViewMut<'_, T> {
    fn as_src(&self) -> View<'_, T> {
        self.as_view()
    }
}

impl Stream {
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

    /// Host to device, into a window that already exists.
    pub fn memcpy_htod<T: Elem, D: CopyDst<T>>(&self, src: &[T], dst: &mut D) -> Result<()> {
        self.copy_into(&mut dst.as_dst(), src)
    }

    /// Device to host, into a caller-owned slice.
    /// Device to host, into a caller-owned slice.
    ///
    /// Synchronises first, because cudarc's is stream-ordered: it queues behind
    /// the launches already submitted. Reading `contents()` without waiting
    /// looks like it works -- unified memory means the pointer is valid -- and
    /// returns whatever the GPU had written by that instant. Which is most of
    /// the answer, most of the time, and produced fluent nonsense from the real
    /// engine while every kernel test passed.
    pub fn memcpy_dtoh<T: Elem, S: CopySrc<T>>(&self, src: &S, dst: &mut [T]) -> Result<()> {
        self.synchronize()?;
        let src = src.as_src();
        if src.len() > dst.len() {
            return Err(anyhow!(
                "reading {} elements into a {} element slice",
                src.len(),
                dst.len()
            ));
        }
        // SAFETY: bounds checked; shared storage is host-visible.
        unsafe {
            let p = src.raw_buf().contents().as_ptr() as *const u8;
            std::ptr::copy_nonoverlapping(
                p.add(src.byte_offset()) as *const T,
                dst.as_mut_ptr(),
                src.len(),
            );
        }
        Ok(())
    }

    /// Device to host, allocating the destination.
    pub fn clone_dtoh<T: Elem, S: CopySrc<T>>(&self, src: &S) -> Result<Vec<T>> {
        self.memcpy_dtov(&src.as_src())
    }

    /// Device to device.
    ///
    /// A plain `memmove` in unified memory. Overlap is permitted because the
    /// engine's uses -- sliding a KV window, adopting a forked sequence's
    /// slots -- can genuinely overlap, and CUDA's `memcpy_dtod` allows it too.
    pub fn memcpy_dtod<T: Elem, S: CopySrc<T>, D: CopyDst<T>>(
        &self,
        src: &S,
        dst: &mut D,
    ) -> Result<()> {
        self.synchronize()?;
        let src = src.as_src();
        let mut dst = dst.as_dst();
        if src.len() > dst.len() {
            return Err(anyhow!(
                "copying {} elements into a {} element window",
                src.len(),
                dst.len()
            ));
        }
        // SAFETY: both windows are inside live allocations; `copy` (not
        // `copy_nonoverlapping`) because the ranges may overlap.
        unsafe {
            let sp = (src.raw_buf().contents().as_ptr() as *const u8).add(src.byte_offset());
            let dp = (dst.raw_buf().contents().as_ptr() as *mut u8).add(dst.byte_offset());
            std::ptr::copy(sp as *const T, dp as *mut T, src.len());
        }
        Ok(())
    }

    /// Host to device, by cudarc's name for it.
    pub fn clone_htod<T: Elem>(&self, src: &[T]) -> Result<Buf<T>> {
        self.memcpy_stod(src)
    }

    /// Zero a window. Unified memory makes this a host `write_bytes`, which is
    /// why it takes no kernel: the pages the GPU reads are the ones written.
    pub fn memset_zeros<T: Elem, D: CopyDst<T>>(&self, dst: &mut D) -> Result<()> {
        self.synchronize()?;
        let mut dst = dst.as_dst();
        // SAFETY: the window is within an allocation and shared storage is
        // host-visible.
        unsafe {
            let p = dst.raw_buf().contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(
                p.add(dst.byte_offset()),
                0,
                dst.len() * std::mem::size_of::<T>(),
            );
        }
        Ok(())
    }

    /// Host to device. A `memcpy` into unified memory.
    pub fn memcpy_stod<T: Elem>(&self, src: &[T]) -> Result<Buf<T>> {
        let mut b = self.alloc_zeros::<T>(src.len())?;
        self.copy_into(&mut b.as_view_mut(), src)?;
        Ok(b)
    }

    /// Host to device. Stream-ordered for the same reason the read is: writing
    /// a buffer a queued kernel has not finished reading would change its input
    /// underneath it.
    pub fn copy_into<T: Elem>(&self, dst: &mut ViewMut<'_, T>, src: &[T]) -> Result<()> {
        self.synchronize()?;
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

    /// Device to host. Stream-ordered, as above.
    pub fn memcpy_dtov<T: Elem>(&self, src: &View<'_, T>) -> Result<Vec<T>> {
        self.synchronize()?;
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
