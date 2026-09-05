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
    let have_cutlass = std::env::var_os("CARGO_FEATURE_CUTLASS").is_some();
    let have_flash_attn2 = std::env::var_os("CARGO_FEATURE_FLASH_ATTN2").is_some();
    if !have_cutlass && !have_flash_attn2 {
        return;
    }
    println!("cargo:rerun-if-env-changed=INFERO_NVCC");
    let nvcc = resolve_nvcc();
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    if have_cutlass {
        println!("cargo:rerun-if-changed=src/cutlass/fp8_bw_gemm.cu");
        println!("cargo:rerun-if-env-changed=INFERO_CUTLASS_DIR");
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
        let src = manifest.join("src/cutlass/fp8_bw_gemm.cu");
        let obj = out_dir.join("fp8_bw_gemm.o");
        aot_compile(
            &nvcc,
            &src,
            &obj,
            &["arch=compute_120a,code=sm_120a"],
            &[],
            &[&cutlass_include, &cutlass_util],
        );
        archive_and_link(&nvcc, &out_dir, &[obj], "infero_cutlass_fp8");
    }

    if have_flash_attn2 {
        println!("cargo:rerun-if-changed=src/cu_vendor/flash_attn2_shim.cu");
        println!("cargo:rerun-if-env-changed=INFERO_FLASH_ATTN_DIR");
        println!("cargo:rerun-if-env-changed=INFERO_CUTLASS_DIR");
        let cutlass_dir = resolve_cutlass_dir();
        let cutlass_include = cutlass_dir.join("include");
        let fa_dir = resolve_flash_attn_dir();
        let fa_src = fa_dir.join("csrc/flash_attn/src");
        if !fa_src.is_dir() {
            panic!(
                "flash-attention checkout at {} is missing {} -- set INFERO_FLASH_ATTN_DIR to a \
                 checkout of Dao-AILab/flash-attention",
                fa_dir.display(),
                fa_src.display()
            );
        }
        let src = manifest.join("src/cu_vendor/flash_attn2_shim.cu");
        let obj = out_dir.join("flash_attn2_shim.o");
        // sm_80, not sm_120a: this vendor kernel has no Blackwell-specific
        // tuning to target, and this matches what vLLM's own bundled FA2
        // build actually ships and runs on sm_120 via PTX forward
        // compatibility (confirmed via `cuobjdump` earlier this session) --
        // compiling natively for sm_80 here is the real target, not a
        // shortcut.
        aot_compile(
            &nvcc,
            &src,
            &obj,
            // Both a real sm_80 cubin AND embedded sm_80 PTX -- the PTX is
            // what lets the driver JIT this onto sm_120 at load time (the
            // real mechanism confirmed this session via `cuobjdump` on
            // vLLM's own bundled FA2 .so: cubin-only for one arch does NOT
            // run on a newer arch at all, `cudaErrorNoKernelImageForDevice`
            // -- hit and fixed here, not guessed).
            &["arch=compute_80,code=sm_80", "arch=compute_80,code=compute_80"],
            &["FLASHATTENTION_DISABLE_DROPOUT"],
            &[&fa_src, &cutlass_include],
        );
        archive_and_link(&nvcc, &out_dir, &[obj], "infero_flash_attn2");
    }
}

fn aot_compile(
    nvcc: &Path,
    src: &Path,
    obj: &Path,
    gencode: &[&str],
    defines: &[&str],
    includes: &[&Path],
) {
    let mut cmd = Command::new(nvcc);
    cmd.args(["-std=c++17", "-O3", "-c", "--expt-relaxed-constexpr", "-DNDEBUG", "-Xcompiler", "-fPIC"]);
    for g in gencode {
        cmd.args(["-gencode", g]);
    }
    for d in defines {
        cmd.arg(format!("-D{d}"));
    }
    for i in includes {
        cmd.arg("-I").arg(i);
    }
    cmd.arg(src).arg("-o").arg(obj);
    let status = cmd.status().unwrap_or_else(|e| panic!("failed to run {}: {e}", nvcc.display()));
    if !status.success() {
        panic!("nvcc failed compiling {}", src.display());
    }
}

/// Archives `objs` into `lib<name>.a` in `out_dir`, links it statically, and
/// links the same CUDA-runtime/C++-runtime dependencies every AOT-compiled
/// kernel in this crate needs (a `<<<...>>>` launch needs `cudart`'s
/// trampoline; CUTLASS/FA2's own `CUTLASS_CHECK`-style macros touch
/// `std::cerr`, so Rust's link line needs `libstdc++` explicitly).
fn archive_and_link(nvcc: &Path, out_dir: &Path, objs: &[PathBuf], name: &str) {
    let ar = nvcc
        .parent()
        .map(|p| p.join("../bin/ar"))
        .filter(|p| p.is_file())
        .unwrap_or_else(|| PathBuf::from("ar"));
    let archive = out_dir.join(format!("lib{name}.a"));
    let status = Command::new(&ar)
        .args(["rcs"])
        .arg(&archive)
        .args(objs)
        .status()
        .unwrap_or_else(|e| panic!("failed to run ar: {e}"));
    if !status.success() {
        panic!("ar failed archiving into {}", archive.display());
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={name}");

    // Static, so a shipped binary needs no `libcudart.so` on the target
    // machine (matches the rpath-baking this crate's sibling `cuda/build.rs`
    // already does for the driver side).
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
        "the `cutlass`/`flash_attn2` feature needs a NVIDIA/cutlass checkout -- set \
         INFERO_CUTLASS_DIR (a sparse checkout of just `include/` and `tools/util/` is enough, \
         see crates/kernels/src/cutlass/fp8_bw_gemm.cu's header for which example it tracks)"
    );
}

fn resolve_flash_attn_dir() -> PathBuf {
    if let Ok(p) = std::env::var("INFERO_FLASH_ATTN_DIR") {
        return PathBuf::from(p);
    }
    panic!(
        "the `flash_attn2` feature needs a Dao-AILab/flash-attention checkout -- set \
         INFERO_FLASH_ATTN_DIR (e.g. /tmp/flash_attn_src on `bw`)"
    );
}
