//! Only active behind the `cutlass` feature. Every other kernel in this crate
//! is compiled at runtime by NVRTC (see `infero_cuda::nvrtc`); CUTLASS's
//! template depth is not a realistic JIT target, so its one kernel is
//! AOT-compiled here with `nvcc` instead and linked as a static archive.
//!
//! Needs a real CUDA Toolkit (not just the driver/nvrtc/cublas `.so`s
//! `vendor/cuda` normally provides) and a checkout of NVIDIA/cutlass — see
//! `resolve_nvcc`/`resolve_cutlass_dir` for how those are found.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_CUTLASS").is_none() {
        return;
    }
    println!("cargo:rerun-if-changed=src/cutlass/fp8_bw_gemm.cu");
    println!("cargo:rerun-if-env-changed=INFERO_NVCC");
    println!("cargo:rerun-if-env-changed=INFERO_CUTLASS_DIR");

    let nvcc = resolve_nvcc();
    let cutlass_dir = resolve_cutlass_dir();
    let cutlass_include = cutlass_dir.join("include");
    let cutlass_util = cutlass_dir.join("tools/util/include");
    for p in [&cutlass_include, &cutlass_util] {
        if !p.is_dir() {
            panic!(
                "CUTLASS checkout at {} is missing {} -- set INFERO_CUTLASS_DIR to a checkout \
                 with both `include/` and `tools/util/include/` (sparse-checkout is fine)",
                cutlass_dir.display(),
                p.display()
            );
        }
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let obj = out_dir.join("fp8_bw_gemm.o");
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let src = manifest.join("src/cutlass/fp8_bw_gemm.cu");

    let status = Command::new(&nvcc)
        .args([
            "-std=c++17",
            "-O3",
            "-c",
            "--expt-relaxed-constexpr",
            "-DNDEBUG",
            "-Xcompiler",
            "-fPIC",
            "-gencode",
            "arch=compute_120a,code=sm_120a",
        ])
        .arg("-I")
        .arg(&cutlass_include)
        .arg("-I")
        .arg(&cutlass_util)
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", nvcc.display()));
    if !status.success() {
        panic!("nvcc failed compiling {}", src.display());
    }

    let ar = nvcc
        .parent()
        .map(|p| p.join("../bin/ar"))
        .filter(|p| p.is_file())
        .unwrap_or_else(|| PathBuf::from("ar"));
    let archive = out_dir.join("libinfero_cutlass_fp8.a");
    let status = Command::new(&ar)
        .args(["rcs"])
        .arg(&archive)
        .arg(&obj)
        .status()
        .unwrap_or_else(|e| panic!("failed to run ar: {e}"));
    if !status.success() {
        panic!("ar failed archiving {}", obj.display());
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=infero_cutlass_fp8");

    // The generated object uses `<<<...>>>` kernel launches, which need the
    // CUDA Runtime's launch trampoline -- static, so a shipped binary needs
    // no `libcudart.so` on the target machine (matches the rpath-baking this
    // crate's sibling `cuda/build.rs` already does for the driver side).
    let cuda_lib = nvcc
        .parent()
        .and_then(|bin| bin.parent())
        .map(|root| root.join("lib64"))
        .filter(|p| p.is_dir());
    if let Some(lib) = cuda_lib {
        println!("cargo:rustc-link-search=native={}", lib.display());
    }
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=rt");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=pthread");
    // The compiled object is C++ (CUTLASS's `CUTLASS_CHECK` et al touch
    // `std::cerr`); Rust's own link line doesn't pull in libstdc++.
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn resolve_nvcc() -> PathBuf {
    if let Ok(p) = std::env::var("INFERO_NVCC") {
        return PathBuf::from(p);
    }
    for candidate in [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12.8/bin/nvcc",
        "/usr/local/cuda-13/bin/nvcc",
    ] {
        if Path::new(candidate).is_file() {
            return PathBuf::from(candidate);
        }
    }
    panic!(
        "no nvcc found for the `cutlass` feature -- set INFERO_NVCC to its full path \
         (a plain driver/NVRTC install, e.g. vendor/cuda, does not include it; a full \
         CUDA Toolkit does)"
    );
}

fn resolve_cutlass_dir() -> PathBuf {
    if let Ok(p) = std::env::var("INFERO_CUTLASS_DIR") {
        return PathBuf::from(p);
    }
    panic!(
        "the `cutlass` feature needs a NVIDIA/cutlass checkout -- set INFERO_CUTLASS_DIR \
         (a sparse checkout of just `include/` and `tools/util/` is enough, see \
         crates/kernels/src/cutlass/fp8_bw_gemm.cu's header for which example it tracks)"
    );
}
