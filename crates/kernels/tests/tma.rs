//! Is TMA reachable from this engine's NVRTC path?
//!
//! Every way of buying memory-level parallelism inside the current GEMM has now
//! measured negative — a wider row group spills at 255 registers, depth-three
//! register prefetch loses, and the `cp.async` weight ring is monotonically
//! worse with depth on both cards — because registers and shared memory are one
//! resource seen twice, and the latency count wants about 940 KB in flight per
//! SM against the ~276 KB either can hold at a useful block count.
//!
//! The way out is the one CUTLASS and Marlin take: `cp.async.bulk.tensor` moves
//! bytes without holding registers, and an `mbarrier` pipeline lets one block
//! per SM use the whole shared budget. That is a different kernel, and before
//! writing one it is worth knowing whether the mechanism works *here* — runtime
//! NVRTC at `compute_120`, a descriptor built by `cuTensorMapEncodeTiled` and
//! passed by value, no CUTLASS, no offline `nvcc`.
//!
//! So this loads one tile and adds it up. If it passes, the architecture is
//! available and the GEMM rewrite is a matter of work; if it fails, that is
//! days saved.

use anyhow::{Context, Result};
use cudarc::driver::{sys, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg};
use tuili_cuda::Device;

/// `CUtensorMap` is 128 opaque bytes that must reach the kernel by value.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct TmaDesc([u8; 128]);
unsafe impl DeviceRepr for TmaDesc {}

const INNER: usize = 512;
const ROWS: usize = 64;
const BOX_INNER: usize = 128;
const BOX_ROWS: usize = 8;
const TILE_BYTES: usize = BOX_INNER * BOX_ROWS;

const SRC: &str = r#"
struct __align__(64) TmaDesc { unsigned char bytes[128]; };

extern "C" __global__ void tma_tile_sum(float* __restrict__ out,
                                        const __grid_constant__ TmaDesc desc,
                                        int bx, int by, int tile_bytes) {
    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char* tile = smem;
    unsigned long long* mbar = (unsigned long long*)(void*)(smem + 4096);

    const unsigned int tile_s = (unsigned int)__cvta_generic_to_shared(tile);
    const unsigned int mbar_s = (unsigned int)__cvta_generic_to_shared(mbar);

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(mbar_s));
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
                mbar_s),
            "r"(tile_bytes));
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global"
            ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
                tile_s),
            "l"(&desc), "r"(bx), "r"(by), "r"(mbar_s)
            : "memory");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "WAIT:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n"
            "@p bra.uni DONE;\n"
            "bra.uni WAIT;\n"
            "DONE:\n"
            "}\n" ::"r"(mbar_s)
            : "memory");
    }
    __syncthreads();

    float s = 0.0f;
    for (int i = threadIdx.x; i < tile_bytes; i += blockDim.x) {
        s += (float)tile[i];
    }
    atomicAdd(out, s);
}
"#;

#[test]
fn tma_bulk_tensor_loads_a_tile() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    if dev.arch() < 90 {
        eprintln!("skipping: TMA needs sm_90 or newer, this is sm_{}", dev.arch());
        return Ok(());
    }
    let stream = dev.stream();
    // Every byte a one, so the tile's sum is its size.
    let host = vec![1u8; INNER * ROWS];
    let src = stream.clone_htod(&host)?;

    let mut desc = TmaDesc([0u8; 128]);
    let global_dim = [INNER as u64, ROWS as u64];
    let global_strides = [INNER as u64];
    let box_dim = [BOX_INNER as u32, BOX_ROWS as u32];
    let elem_strides = [1u32, 1];
    let (dptr, _sync) = src.device_ptr(stream);
    let ptr = dptr as *mut std::ffi::c_void;
    let r = unsafe {
        sys::cuTensorMapEncodeTiled(
            (&mut desc as *mut TmaDesc).cast(),
            sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
            2,
            ptr,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            elem_strides.as_ptr(),
            sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
            sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };
    anyhow::ensure!(
        r == sys::CUresult::CUDA_SUCCESS,
        "cuTensorMapEncodeTiled: {r:?}"
    );

    let f = dev
        .kernels()
        .get("tuili_tma_probe", SRC, "tma_tile_sum")
        .context("compiling the TMA probe")?;
    let mut out = stream.alloc_zeros::<f32>(1)?;
    let (bx, by, tb) = (0i32, 0i32, TILE_BYTES as i32);
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 4096 + 8,
    };
    let mut b = stream.launch_builder(&f);
    b.arg(&mut out).arg(&desc).arg(&bx).arg(&by).arg(&tb);
    unsafe { b.launch(cfg) }.context("launching the TMA probe")?;
    let got = stream.clone_dtoh(&out)?;
    dev.synchronize()?;
    eprintln!(
        "  tile of {TILE_BYTES} bytes summed to {} (want {TILE_BYTES})",
        got[0]
    );
    assert_eq!(
        got[0], TILE_BYTES as f32,
        "the TMA copy did not deliver the tile"
    );
    Ok(())
}
