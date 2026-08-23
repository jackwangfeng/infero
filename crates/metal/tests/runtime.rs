//! The device layer, on real hardware.
//!
//! These tests do not know what a transformer is. They check the four things
//! the rest of the port rests on: a device opens, MSL compiles at run time, a
//! dispatch lands, and a view's offset reaches the kernel as the window the
//! host asked for. The last one is the interesting one -- it is the only place
//! where the CUDA and Metal argument models genuinely differ, because CUDA
//! passes an already-offset pointer where Metal passes a buffer plus a byte
//! offset, and getting it wrong would read the right length from the wrong
//! place.

use anyhow::Result;
use tuili_metal::{Device, LaunchConfig};

/// `add_f32` from `crates/kernels/src/cu/ops.cu`, transliterated.
///
/// Kept next to the CUDA original in spirit: same parameter order, so the
/// `.arg()` chain at the call site is identical, and `[[buffer(n)]]` indices
/// follow that order rather than being chosen.
const OPS: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add_f32(device float* out          [[buffer(0)]],
                    device const float* a      [[buffer(1)]],
                    device const float* b      [[buffer(2)]],
                    constant int& n            [[buffer(3)]],
                    uint3 tgid  [[threadgroup_position_in_grid]],
                    uint3 tid   [[thread_position_in_threadgroup]],
                    uint3 tgdim [[threads_per_threadgroup]]) {
    const int i = int(tgid.x * tgdim.x + tid.x);
    if (i < n) out[i] = a[i] + b[i];
}

kernel void fill_iota_f32(device float* out [[buffer(0)]],
                          constant int& n   [[buffer(1)]],
                          uint3 tgid  [[threadgroup_position_in_grid]],
                          uint3 tid   [[thread_position_in_threadgroup]],
                          uint3 tgdim [[threads_per_threadgroup]]) {
    const int i = int(tgid.x * tgdim.x + tid.x);
    if (i < n) out[i] = float(i);
}
"#;

const BLOCK: u32 = 256;

fn elementwise(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(BLOCK).max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[test]
fn a_device_opens_and_reports_itself() -> Result<()> {
    let dev = Device::new(0)?;
    let caps = *dev.caps();
    eprintln!(
        "  {} | simd {} | working set {:.1} GiB",
        dev.name(),
        caps.simd_width,
        caps.working_set_bytes as f64 / (1u64 << 30) as f64
    );

    // The reductions ported from `common.cuh` assume a 32-lane shuffle. If a
    // future Apple GPU changes this, every one of them is wrong and it should
    // fail here rather than in the numbers.
    assert_eq!(caps.simd_width, 32, "the ported reductions assume 32 lanes");

    // Nothing hand-written for tensor cores, FP8 or TMA exists on this backend,
    // and the dispatch in `tuili-model` reads exactly these to route around it.
    assert!(!caps.int_tensor_gemm);
    assert!(!caps.fp8);
    assert!(!caps.tma);

    assert!(caps.working_set_bytes > (1u64 << 30), "implausible VRAM budget");
    Ok(())
}

#[test]
fn a_second_device_is_an_error_rather_than_the_first_one() {
    // `--device 1` on a machine with one GPU should say so, not silently serve
    // device 0 and leave the operator thinking they picked something.
    assert!(Device::new(1).is_err());
}

#[test]
fn a_dispatch_adds_two_vectors() -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();

    let n = 4096usize;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (2 * i) as f32).collect();

    let da = s.memcpy_stod(&a)?;
    let db = s.memcpy_stod(&b)?;
    let mut dout = s.alloc_zeros::<f32>(n)?;

    let f = dev.kernels().get("tuili_ops", OPS, "add_f32")?;
    let n_i = n as i32;
    let mut lb = s.launch_builder(&f);
    lb.arg(&dout.as_view_mut())
        .arg(&da.as_view())
        .arg(&db.as_view())
        .arg(&n_i);
    unsafe { lb.launch(elementwise(n as u32))? };
    s.synchronize()?;

    let got = dout.to_vec();
    for i in 0..n {
        let want = a[i] + b[i];
        assert_eq!(got[i], want, "element {i}");
    }
    Ok(())
}

#[test]
fn a_view_offset_binds_the_window_the_host_asked_for() -> Result<()> {
    // CUDA hands the kernel a pointer that already includes the offset; Metal
    // hands it a buffer and a byte offset. This is the one asymmetry in the
    // argument model, so it gets its own test: write into the middle of a
    // buffer through a slice, and require the untouched ends to stay untouched.
    let dev = Device::new(0)?;
    let s = dev.stream();

    let n = 1024usize;
    let (lo, hi) = (256usize, 768usize);

    let mut buf = s.alloc_zeros::<f32>(n)?;
    let f = dev.kernels().get("tuili_ops", OPS, "fill_iota_f32")?;
    let len_i = (hi - lo) as i32;
    {
        let win = buf.slice_mut(lo..hi);
        let mut lb = s.launch_builder(&f);
        lb.arg(&win).arg(&len_i);
        unsafe { lb.launch(elementwise(len_i as u32))? };
    }
    s.synchronize()?;

    let got = buf.to_vec();
    for i in 0..lo {
        assert_eq!(got[i], 0.0, "wrote before the window at {i}");
    }
    for i in lo..hi {
        assert_eq!(got[i], (i - lo) as f32, "window element {i}");
    }
    for i in hi..n {
        assert_eq!(got[i], 0.0, "wrote past the window at {i}");
    }
    Ok(())
}

#[test]
fn a_missing_kernel_names_itself() -> Result<()> {
    let dev = Device::new(0)?;
    let err = dev
        .kernels()
        .get("tuili_ops", OPS, "no_such_kernel")
        .unwrap_err()
        .to_string();
    assert!(err.contains("no_such_kernel"), "unhelpful error: {err}");
    Ok(())
}

#[test]
fn a_compile_error_surfaces_rather_than_aborting() -> Result<()> {
    let dev = Device::new(0)?;
    let err = dev
        .kernels()
        .get("broken", "kernel void x(device float* p) { this is not MSL }", "x")
        .unwrap_err()
        .to_string();
    assert!(err.contains("broken"), "unhelpful error: {err}");
    Ok(())
}

#[test]
fn a_module_recompiles_when_its_source_changes() -> Result<()> {
    // The NVRTC cache is keyed on source because `TUILI_FP8_STRIP` prepends
    // defines that change what the kernel does; a stripped build must never be
    // served from a cache entry built from the serving source. Same rule here.
    let dev = Device::new(0)?;
    let a = "kernel void probe(device float* p [[buffer(0)]]) { p[0] = 1.0f; }";
    let b = "kernel void probe(device float* p [[buffer(0)]]) { p[0] = 2.0f; }";

    let s = dev.stream();
    let one = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut out = s.alloc_zeros::<f32>(1)?;
    for (src, want) in [(a, 1.0f32), (b, 2.0f32)] {
        let f = dev.kernels().get("probe_mod", src, "probe")?;
        let mut lb = s.launch_builder(&f);
        lb.arg(&out.as_view_mut());
        unsafe { lb.launch(one)? };
        s.synchronize()?;
        assert_eq!(out.to_vec()[0], want, "stale pipeline for changed source");
    }
    Ok(())
}
