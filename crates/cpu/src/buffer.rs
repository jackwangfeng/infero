//! "Device" memory, which on this backend is just memory.
//!
//! Mirrors `infero_metal::buffer` almost exactly -- unified memory means the
//! Metal backend already reduced every one of these operations to a plain
//! host pointer, so there is nothing left to fake for CPU beyond dropping
//! the `MTLBuffer` wrapper around it.

use std::marker::PhantomData;
use std::ops::{Bound, RangeBounds};
use std::sync::Arc;

use anyhow::{Result, anyhow};

/// An element a kernel can hold. See `infero_metal::buffer::Elem`'s own doc
/// comment for the safety argument; it applies unchanged here.
pub unsafe trait Elem: Copy + 'static {}
unsafe impl Elem for f32 {}
unsafe impl Elem for half::f16 {}
unsafe impl Elem for i32 {}
unsafe impl Elem for u32 {}
unsafe impl Elem for u8 {}
unsafe impl Elem for i8 {}
unsafe impl Elem for f64 {}
unsafe impl Elem for i64 {}
unsafe impl Elem for u64 {}

#[allow(dead_code)]
enum Keep {
    Owned(Vec<u8>),
    /// A whole mapped file -- see `map_file` below.
    Mapped(memmap2::Mmap),
}

struct Raw {
    ptr: *mut u8,
    bytes: usize,
    /// Whatever the allocation actually is; `ptr` points into it. Never read
    /// again after construction, only kept alive.
    #[allow(dead_code)]
    keep: Keep,
}

// SAFETY: every access through `ptr` goes through a `View`/`ViewMut` that
// either only reads or is the sole `&mut` window an `unsafe` block acts
// through, matching the discipline `infero_metal::buffer::Raw` documents for
// the same reason (there, the buffer is genuinely GPU-visible; here it is
// genuinely a plain allocation, but the caller-side contract is identical).
unsafe impl Send for Raw {}
unsafe impl Sync for Raw {}

/// An owned allocation of `len` elements.
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
        View { raw: &self.raw, off, len, _t: PhantomData }
    }

    pub fn slice_mut<R: RangeBounds<usize>>(&mut self, r: R) -> ViewMut<'_, T> {
        let (off, len) = resolve(&r, self.len);
        ViewMut { raw: &self.raw, off, len, _t: PhantomData }
    }

    pub fn as_view(&self) -> View<'_, T> {
        self.slice(..)
    }

    pub fn as_view_mut(&mut self) -> ViewMut<'_, T> {
        self.slice_mut(..)
    }

    pub fn split_at(&self, mid: usize) -> (View<'_, T>, View<'_, T>) {
        (self.slice(..mid), self.slice(mid..))
    }

    pub fn split_at_mut(&mut self, mid: usize) -> (ViewMut<'_, T>, ViewMut<'_, T>) {
        let (lo, hi) = (resolve(&(..mid), self.len), resolve(&(mid..), self.len));
        (
            ViewMut { raw: &self.raw, off: lo.0, len: lo.1, _t: PhantomData },
            ViewMut { raw: &self.raw, off: hi.0, len: hi.1, _t: PhantomData },
        )
    }

    /// # Safety
    /// As `infero_metal::buffer::Buf::transmute`: every `Elem` is plain data
    /// with no invalid bit patterns, so the obligation is only that the byte
    /// count fits.
    pub unsafe fn transmute<U: Elem>(&self, n: usize) -> Result<View<'_, U>> {
        if n * std::mem::size_of::<U>() > self.raw.bytes {
            return Err(anyhow!(
                "transmute to {n} x {} exceeds the {} byte allocation",
                std::mem::size_of::<U>(),
                self.raw.bytes
            ));
        }
        Ok(View { raw: &self.raw, off: 0, len: n, _t: PhantomData })
    }

    pub fn to_vec(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.len);
        // SAFETY: `raw.ptr` is valid for `raw.bytes`, and `len * size_of::<T>()`
        // was checked at allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(self.raw.ptr as *const T, out.as_mut_ptr(), self.len);
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

            pub(crate) fn byte_offset(&self) -> usize {
                self.off * std::mem::size_of::<T>()
            }

            /// The raw pointer a kernel dispatch binds -- see `launch.rs`.
            pub(crate) fn raw_ptr(&self) -> *mut u8 {
                // SAFETY of the *use* of this pointer is the launch site's
                // obligation (same contract cudarc/Metal already carry); this
                // accessor itself does no dereference.
                unsafe { self.raw.ptr.add(self.byte_offset()) }
            }

            pub unsafe fn transmute<U: Elem>(&self, n: usize) -> Result<View<'a, U>> {
                let want = n * std::mem::size_of::<U>();
                let have = self.len * std::mem::size_of::<T>();
                if want > have {
                    return Err(anyhow!(
                        "transmute to {n} x {} exceeds this {have} byte window",
                        std::mem::size_of::<U>()
                    ));
                }
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

            pub fn split_at(&self, mid: usize) -> (View<'a, T>, View<'a, T>) {
                (self.slice(..mid), self.slice(mid..))
            }

            pub fn slice<R: RangeBounds<usize>>(&self, r: R) -> View<'a, T> {
                let (off, len) = resolve(&r, self.len);
                View { raw: self.raw, off: self.off + off, len, _t: PhantomData }
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
        ViewMut { raw: self.raw, off: self.off + off, len, _t: PhantomData }
    }

    pub fn as_view(&self) -> View<'_, T> {
        View { raw: self.raw, off: self.off, len: self.len, _t: PhantomData }
    }

    pub fn split_at_mut(&mut self, mid: usize) -> (ViewMut<'_, T>, ViewMut<'_, T>) {
        let mid = mid.min(self.len);
        (
            ViewMut { raw: self.raw, off: self.off, len: mid, _t: PhantomData },
            ViewMut { raw: self.raw, off: self.off + mid, len: self.len - mid, _t: PhantomData },
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
        ViewMut { raw: self.raw, off: self.off, len: self.len, _t: PhantomData }
    }
}

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

impl crate::device::Stream {
    pub fn alloc_zeros<T: Elem>(&self, n: usize) -> Result<Buf<T>> {
        let bytes = (n * std::mem::size_of::<T>()).max(1);
        let mut v = vec![0u8; bytes];
        let ptr = v.as_mut_ptr();
        Ok(Buf {
            raw: Arc::new(Raw { ptr, bytes, keep: Keep::Owned(v) }),
            len: n,
            _t: PhantomData,
        })
    }

    pub fn memcpy_htod<T: Elem, D: CopyDst<T>>(&self, src: &[T], dst: &mut D) -> Result<()> {
        self.copy_into(&mut dst.as_dst(), src)
    }

    pub fn memcpy_dtoh<T: Elem, S: CopySrc<T>>(&self, src: &S, dst: &mut [T]) -> Result<()> {
        let src = src.as_src();
        if src.len() > dst.len() {
            return Err(anyhow!("reading {} elements into a {} element slice", src.len(), dst.len()));
        }
        // SAFETY: bounds checked; plain host memory.
        unsafe {
            std::ptr::copy_nonoverlapping(src.raw_ptr() as *const T, dst.as_mut_ptr(), src.len());
        }
        Ok(())
    }

    pub fn clone_dtoh<T: Elem, S: CopySrc<T>>(&self, src: &S) -> Result<Vec<T>> {
        self.memcpy_dtov(&src.as_src())
    }

    pub fn memcpy_dtod<T: Elem, S: CopySrc<T>, D: CopyDst<T>>(&self, src: &S, dst: &mut D) -> Result<()> {
        let src = src.as_src();
        let dst = dst.as_dst();
        if src.len() > dst.len() {
            return Err(anyhow!("copying {} elements into a {} element window", src.len(), dst.len()));
        }
        // SAFETY: both windows are inside live allocations; `copy` (not
        // `copy_nonoverlapping`) because the ranges may overlap, matching
        // `infero_metal::buffer::Stream::memcpy_dtod`.
        unsafe {
            std::ptr::copy(src.raw_ptr() as *const T, dst.raw_ptr() as *mut T, src.len());
        }
        Ok(())
    }

    pub fn clone_htod<T: Elem>(&self, src: &[T]) -> Result<Buf<T>> {
        self.memcpy_stod(src)
    }

    pub fn memset_zeros<T: Elem, D: CopyDst<T>>(&self, dst: &mut D) -> Result<()> {
        let dst = dst.as_dst();
        // SAFETY: the window is within an allocation.
        unsafe {
            std::ptr::write_bytes(dst.raw_ptr(), 0, dst.len() * std::mem::size_of::<T>());
        }
        Ok(())
    }

    pub fn memcpy_stod<T: Elem>(&self, src: &[T]) -> Result<Buf<T>> {
        let mut b = self.alloc_zeros::<T>(src.len())?;
        self.copy_into(&mut b.as_view_mut(), src)?;
        Ok(b)
    }

    pub fn copy_into<T: Elem>(&self, dst: &mut ViewMut<'_, T>, src: &[T]) -> Result<()> {
        if src.len() > dst.len() {
            return Err(anyhow!("copying {} elements into a {} element window", src.len(), dst.len()));
        }
        // SAFETY: bounds checked above.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst.raw_ptr() as *mut T, src.len());
        }
        Ok(())
    }

    pub fn memcpy_dtov<T: Elem>(&self, src: &View<'_, T>) -> Result<Vec<T>> {
        let mut out = Vec::with_capacity(src.len());
        // SAFETY: as above.
        unsafe {
            std::ptr::copy_nonoverlapping(src.raw_ptr() as *const T, out.as_mut_ptr(), src.len());
            out.set_len(src.len());
        }
        Ok(out)
    }
}

/// A whole file, aliased into "device" memory without a copy -- here, an
/// ordinary `mmap`, since device memory and host memory are the same thing.
pub fn map_file(_dev: &crate::Device, path: &std::path::Path) -> Result<Buf<u8>> {
    use anyhow::Context;
    let file = std::fs::File::open(path).with_context(|| format!("opening {} to map it", path.display()))?;
    // SAFETY: opened read-only; the mapping is owned by the buffer's `keep`
    // for as long as the buffer lives.
    let map = unsafe { memmap2::Mmap::map(&file) }.with_context(|| format!("mapping {}", path.display()))?;
    let len = map.len();
    anyhow::ensure!(len > 0, "{} is empty", path.display());
    let ptr = map.as_ptr() as *mut u8;
    Ok(Buf {
        raw: Arc::new(Raw { ptr, bytes: len, keep: Keep::Mapped(map) }),
        len,
        _t: PhantomData,
    })
}
