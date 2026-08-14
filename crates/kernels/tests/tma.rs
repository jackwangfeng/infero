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

/// What TMA can pull through one fat block an SM, against the 1345 GB/s the
/// `cp.async` probe reaches with many thin ones.
///
/// This is the number that decides whether the GEMM is worth rewriting. The
/// current kernel is at 1164 GB/s of weights on `gate_up` and the roofline is
/// 1792; the elimination work says the missing part is in-flight bytes, and TMA
/// is the only way left to hold them — a descriptor-driven copy engine that does
/// not spend registers, and a deep `mbarrier` pipeline in a block that owns its
/// SM's whole shared budget.
///
/// Reads a `gate_up`-shaped quant plane — 28672 rows of 2048 bytes, the layout
/// `mmq_load_w_q4_g128t` sees — through a six-stage pipeline, over four cycled
/// buffers so nothing is L2-resident.
const PIPE_SRC: &str = r#"
struct __align__(64) TmaDesc { unsigned char bytes[128]; };

#define TMA_BOX_IN BOX_IN_   /* u32 elements */
#define TMA_BOX_ROWS BOX_ROWS_
#define TMA_TILE (TMA_BOX_IN * 4 * TMA_BOX_ROWS)
#define TMA_STAGES STAGES_

extern "C" __global__ void tma_stream_sum(float* __restrict__ out,
                                          const __grid_constant__ TmaDesc desc,
                                          int row_tiles, int col_tiles) {
    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char* tiles = smem;
    unsigned long long* mbar =
        (unsigned long long*)(void*)(smem + TMA_STAGES * TMA_TILE);

    const unsigned int mbar0 = (unsigned int)__cvta_generic_to_shared(mbar);
    if (threadIdx.x == 0) {
#pragma unroll
        for (int s = 0; s < TMA_STAGES; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(
                mbar0 + (unsigned)(s * 8)));
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    }
    __syncthreads();

    // The flattened (row tile, column tile) list, striped across the grid.
    const int total = row_tiles * col_tiles;
    const int per = (total + (int)gridDim.x - 1) / (int)gridDim.x;
    const int begin = per * (int)blockIdx.x;
    const int end = min(total, begin + per);

    // Fill the pipeline, then consume one and refill it.
    int issued = begin;
    for (int s = 0; s < TMA_STAGES && issued < end; ++s, ++issued) {
        if (threadIdx.x == 0) {
            const unsigned int t =
                (unsigned int)__cvta_generic_to_shared(tiles + s * TMA_TILE);
            const unsigned int m = mbar0 + (unsigned)(s * 8);
            const int bx = (issued % col_tiles) * TMA_BOX_IN;
            const int by = (issued / col_tiles) * TMA_BOX_ROWS;
            asm volatile(
                "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
                    m),
                "r"(TMA_TILE));
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global"
                ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
                    t),
                "l"(&desc), "r"(bx), "r"(by), "r"(m)
                : "memory");
        }
    }

    float acc = 0.0f;
    int phase[TMA_STAGES];
#pragma unroll
    for (int s = 0; s < TMA_STAGES; ++s) phase[s] = 0;

    for (int i = begin; i < end; ++i) {
        const int s = (i - begin) % TMA_STAGES;
        const unsigned int m = mbar0 + (unsigned)(s * 8);
        if (threadIdx.x == 0) {
            asm volatile(
                "{\n.reg .pred p;\n"
                "WAITP:\n"
                "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                "@p bra.uni DONEP;\n"
                "bra.uni WAITP;\n"
                "DONEP:\n}\n" ::"r"(m),
                "r"(phase[s])
                : "memory");
        }
        __syncthreads();
        const uint4* t4 = (const uint4*)(const void*)(tiles + s * TMA_TILE);
#pragma unroll 4
        for (int e = threadIdx.x; e < TMA_TILE / 16; e += (int)blockDim.x) {
            const uint4 v = t4[e];
            acc += (float)(v.x ^ v.w);
        }
        __syncthreads();
        phase[s] ^= 1;
        if (issued < end) {
            if (threadIdx.x == 0) {
                const unsigned int t =
                    (unsigned int)__cvta_generic_to_shared(tiles + s * TMA_TILE);
                const int bx = (issued % col_tiles) * TMA_BOX_IN;
                const int by = (issued / col_tiles) * TMA_BOX_ROWS;
                asm volatile(
                    "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
                        m),
                    "r"(TMA_TILE));
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global"
                    ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], "
                    "[%4];\n" ::"r"(t),
                    "l"(&desc), "r"(bx), "r"(by), "r"(m)
                    : "memory");
            }
            ++issued;
        }
    }
    if (acc == 1.2345e-30f) out[0] = acc;
}
"#;

#[test]
fn tma_streams_weights_against_the_cp_async_probe() -> Result<()> {
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
    // `gate_up`'s quant plane: 28672 rows of k/128 * 64 bytes.
    let (rows, row_bytes) = (28672usize, 4096 / 128 * 64);
    let bytes = rows * row_bytes;
    let pools: Vec<_> = (0..4)
        .map(|_| stream.alloc_zeros::<u8>(bytes))
        .collect::<Result<Vec<_>, _>>()?;

    // u32 elements, so a box is 512 bytes wide rather than 256 — four times the
    // copy granularity, which is the only knob the box limits leave (boxDim is
    // capped at 256 elements in each dimension).
    // `boxDim` is capped at 256 elements per dimension, so the copy granularity
    // and the pipeline depth are the only knobs, and the shared budget couples
    // them: this card refuses a block past about 100 KB.
    eprintln!(
        "\n  gate_up's quant plane, {} MiB, {} blocks of 256 threads\n  \
         (cp.async probe 1345, the kernel 1164, roofline 1792)",
        bytes >> 20,
        dev.sm_count()
    );
    for &(box_in, box_rows, stages) in &[
        (64usize, 64usize, 6usize),
        (128, 64, 3),
        (128, 32, 6),
        (256, 32, 3),
        (256, 16, 6),
        (256, 64, 1),
    ] {
    let tile = box_in * 4 * box_rows;
    let shared = stages * tile + stages * 8;
    let (col_tiles, row_tiles) = (row_bytes / (box_in * 4), rows / box_rows);

    let mut descs = Vec::new();
    for p in &pools {
        let mut d = TmaDesc([0u8; 128]);
        let global_dim = [(row_bytes / 4) as u64, rows as u64];
        let global_strides = [row_bytes as u64];
        let box_dim = [box_in as u32, box_rows as u32];
        let elem_strides = [1u32, 1];
        let (dptr, _sync) = p.device_ptr(stream);
        let r = unsafe {
            sys::cuTensorMapEncodeTiled(
                (&mut d as *mut TmaDesc).cast(),
                sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
                2,
                dptr as *mut std::ffi::c_void,
                global_dim.as_ptr(),
                global_strides.as_ptr(),
                box_dim.as_ptr(),
                elem_strides.as_ptr(),
                sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
                sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            )
        };
        anyhow::ensure!(
            r == sys::CUresult::CUDA_SUCCESS,
            "cuTensorMapEncodeTiled: {r:?}"
        );
        descs.push(d);
    }

    let src = PIPE_SRC
        .replace("BOX_IN_", &box_in.to_string())
        .replace("BOX_ROWS_", &box_rows.to_string())
        .replace("STAGES_", &stages.to_string());
    let module: &'static str = Box::leak(
        format!("tuili_tma_pipe_{box_in}_{box_rows}_{stages}").into_boxed_str(),
    );
    let f = match dev
        .kernels()
        .get(module, Box::leak(src.into_boxed_str()), "tma_stream_sum")
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!("  box {box_in}x{box_rows}, {stages} stages: {e}");
            continue;
        }
    };
    if let Err(e) = tuili_cuda::set_max_dynamic_shared(&f, shared as u32) {
        eprintln!("  box {box_in}x{box_rows}, {stages} stages: {} KiB refused ({e})", shared >> 10);
        continue;
    }
    let mut out = stream.alloc_zeros::<f32>(1)?;
    let blocks = dev.sm_count();
    let (rt, ct) = (row_tiles as i32, col_tiles as i32);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: shared as u32,
    };
    let mut run = |i: usize| -> Result<()> {
        let mut b = stream.launch_builder(&f);
        b.arg(&mut out).arg(&descs[i % 4]).arg(&rt).arg(&ct);
        unsafe { b.launch(cfg) }.context("tma_stream_sum")?;
        Ok(())
    };
    for i in 0..4 {
        run(i)?;
    }
    dev.synchronize()?;
    let t0 = std::time::Instant::now();
    for i in 0..20 {
        run(i)?;
    }
    dev.synchronize()?;
    let secs = t0.elapsed().as_secs_f64() / 20.0;
    eprintln!(
        "  box {:>4}x{box_rows:<3} {stages} stages of {:>3} KiB ({:>3} KiB shared)  \
         {:>6.1} us  {:>5.0} GB/s",
        box_in * 4,
        tile >> 10,
        shared >> 10,
        secs * 1e6,
        bytes as f64 / secs / 1e9
    );
    }
    Ok(())
}
