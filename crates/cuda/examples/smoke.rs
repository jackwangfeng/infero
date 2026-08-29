//! End-to-end check that the GPU path works: NVRTC compiles, a kernel runs on
//! half precision data, and cuBLAS is callable.
//!
//!     cargo run -p infero-cuda --example smoke

use anyhow::Result;
use cudarc::cublas::{Gemm, GemmConfig};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use half::f16;
use infero_cuda::Device;

const SRC: &str = r#"
#include <cuda_fp16.h>

extern "C" __global__ void scale_add_f16(const __half* x, const __half* y,
                                         __half* out, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2half(alpha * __half2float(x[i]) + __half2float(y[i]));
}
"#;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "debug".into()),
        )
        .init();

    let dev = Device::new(0)?;
    let (free, total) = dev.mem_info()?;
    println!(
        "device : {} (sm_{}), {:.1}/{:.1} GiB free",
        dev.name(),
        dev.arch(),
        free as f64 / (1 << 30) as f64,
        total as f64 / (1 << 30) as f64,
    );

    nvrtc_half_kernel(&dev)?;
    cublas_identity(&dev)?;

    println!("smoke: ok");
    Ok(())
}

/// Proves NVRTC found the CUDA headers and the driver accepted our PTX.
fn nvrtc_half_kernel(dev: &Device) -> Result<()> {
    const N: usize = 4096;
    let stream = dev.stream();

    let x: Vec<f16> = (0..N).map(|i| f16::from_f32(i as f32)).collect();
    let y: Vec<f16> = (0..N).map(|i| f16::from_f32((N - i) as f32)).collect();

    let dx = stream.clone_htod(&x)?;
    let dy = stream.clone_htod(&y)?;
    let mut dout = stream.alloc_zeros::<f16>(N)?;

    let f = dev.kernels().get("smoke_scale_add", SRC, "scale_add_f16")?;

    let alpha = 2.0f32;
    let n = N as i32;
    let mut launch = stream.launch_builder(&f);
    launch.arg(&dx).arg(&dy).arg(&mut dout).arg(&alpha).arg(&n);
    unsafe { launch.launch(LaunchConfig::for_num_elems(N as u32))? };

    let got = stream.clone_dtoh(&dout)?;
    dev.synchronize()?;

    for i in 0..N {
        let want = alpha * x[i].to_f32() + y[i].to_f32();
        let err = (got[i].to_f32() - want).abs();
        // f16 has ~11 bits of mantissa; allow a relative slip.
        assert!(
            err <= want.abs() * 1e-3 + 1e-2,
            "index {i}: got {} want {want}",
            got[i].to_f32()
        );
    }
    println!("nvrtc  : scale_add_f16 over {N} elements ok");
    Ok(())
}

/// Proves libcublas.so.13 loaded and the handle is bound to our stream.
fn cublas_identity(dev: &Device) -> Result<()> {
    const N: usize = 3;
    let stream = dev.stream();

    // Column-major, but multiplying by the identity is layout-agnostic.
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let eye: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let da = stream.clone_htod(&a)?;
    let de = stream.clone_htod(&eye)?;
    let mut dc = stream.alloc_zeros::<f32>(N * N)?;

    let cfg = GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m: N as i32,
        n: N as i32,
        k: N as i32,
        alpha: 1.0f32,
        lda: N as i32,
        ldb: N as i32,
        beta: 0.0f32,
        ldc: N as i32,
    };
    unsafe { dev.blas().gemm(cfg, &da, &de, &mut dc)? };

    let got = stream.clone_dtoh(&dc)?;
    dev.synchronize()?;
    assert_eq!(got, a, "A * I should be A");
    println!("cublas : {N}x{N} sgemm ok");
    Ok(())
}
