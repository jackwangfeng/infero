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
    let have_nccl = std::env::var_os("CARGO_FEATURE_NCCL").is_some();
    let have_triton_aot = std::env::var_os("CARGO_FEATURE_TRITON_AOT").is_some();
    if have_nccl {
        // NCCL ships its own prebuilt `.so` -- no AOT compile of our own
        // source, just a link-search/link-lib, so this doesn't need `nvcc`
        // and runs regardless of whether cutlass/flash_attn2 are enabled.
        println!("cargo:rerun-if-env-changed=INFERO_NCCL_DIR");
        let nccl_dir = resolve_nccl_dir();
        let nccl_lib = nccl_dir.join("lib/x86_64-linux-gnu");
        let nccl_lib = if nccl_lib.is_dir() { nccl_lib } else { nccl_dir.join("lib") };
        println!("cargo:rustc-link-search=native={}", nccl_lib.display());
        println!("cargo:rustc-link-lib=dylib=nccl");
    }
    if !have_cutlass && !have_flash_attn2 && !have_triton_aot {
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
        // Real per-architecture CUTLASS kernel bodies (SM90/SM100 in
        // addition to this box's own SM120) live in the same translation
        // unit, each behind its own `cutlass::arch::SmXX` tag -- CUTLASS's
        // own `__CUDA_ARCH__`-gated kernel implementations make it safe to
        // list several `-gencode` targets against one source file (each
        // architecture-tagged kernel only produces real device code for the
        // target(s) it's actually valid for). Only sm_120a is
        // execution-verified on this box; sm_90a/sm_100a are compile-verified
        // against the real vendored CUTLASS headers, not run.
        aot_compile(
            &nvcc,
            &src,
            &obj,
            &[
                "arch=compute_90a,code=sm_90a",
                "arch=compute_100a,code=sm_100a",
                "arch=compute_120a,code=sm_120a",
            ],
            &[],
            &[&cutlass_include, &cutlass_util],
        );
        archive_and_link(&nvcc, &out_dir, &[obj], "infero_cutlass_fp8");

        // NVFP4 (W4A4) blockscaled GEMM -- SM120 only, matching this plan's
        // own stated Global Constraint ("SM120 only" for v1), unlike the FP8
        // GEMM above which also opportunistically compiles SM90/SM100 bodies.
        // No SM90/SM100 NVFP4 instantiations here: out of this plan's scope,
        // not attempted.
        println!("cargo:rerun-if-changed=src/cutlass/fp4_bw_gemm.cu");
        let src4 = manifest.join("src/cutlass/fp4_bw_gemm.cu");
        let obj4 = out_dir.join("fp4_bw_gemm.o");
        aot_compile(&nvcc, &src4, &obj4, &["arch=compute_120a,code=sm_120a"], &[], &[&cutlass_include, &cutlass_util]);
        archive_and_link(&nvcc, &out_dir, &[obj4], "infero_cutlass_fp4");
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

    if have_triton_aot {
        // Not vendored into this repo -- see gdn_triton_aot.rs's own doc
        // comment for what this directory holds and how it's produced.
        //
        // Originally one `gdn_h.<hash>.c` (kernel-2 alone). `gdn_fla_stages.rs`
        // adds 5 more real Triton-AOT-compiled kernels (cumsum, kkt,
        // solve_tril64, wy_fast, chunk_o) -- each `triton.tools.compile`'s own
        // `.c` file, with its own hash-derived name, in the SAME directory
        // (see that module's own doc comment for where they come from and
        // `INFERO_TRITON_AOT_DIR`'s own doc comment below for the expected
        // directory layout). All of them get compiled and archived together
        // now, not just the one kernel-2 file.
        println!("cargo:rerun-if-env-changed=INFERO_TRITON_AOT_DIR");
        let aot_dir = resolve_triton_aot_dir();
        let srcs = find_triton_aot_cs(&aot_dir);
        // No device code to compile here (Triton already produced each
        // cubin, embedded as a byte array in each `src`) -- a plain C
        // compiler, not `nvcc`, and `cuda.h` from the same Toolkit checkout
        // `nvcc` itself lives under (`resolve_nvcc`'s own doc comment: a full
        // Toolkit, not just the driver/NVRTC `vendor/cuda` normally
        // provides).
        let cc = resolve_cc();
        let cuda_include = nvcc.parent().and_then(|bin| bin.parent()).map(|root| root.join("include"));
        let mut objs = Vec::new();
        for src in &srcs {
            println!("cargo:rerun-if-changed={}", src.display());
            let stem = src.file_stem().unwrap().to_string_lossy();
            let obj = out_dir.join(format!("gdn_triton_aot_{stem}.o"));
            compile_c(&cc, src, cuda_include.as_deref(), &obj);
            objs.push(obj);
        }
        // Reuses `archive_and_link` (finds `ar` next to `nvcc`, links
        // `cudart_static` alongside) even though these objects need none of
        // that -- harmless when both `cutlass`-family and `triton_aot`
        // features are enabled in the same build, and not worth a second
        // near-identical function for the one real difference below.
        archive_and_link(&nvcc, &out_dir, &objs, "infero_gdn_triton_aot");
        // The one real difference: these objects' own undefined symbols
        // (`cuModuleLoadData`, `cuLaunchKernel`, ...) are the CUDA *driver*
        // API, which `cudart_static` does not provide.
        println!("cargo:rustc-link-lib=dylib=cuda");
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

fn resolve_nccl_dir() -> PathBuf {
    if let Ok(p) = std::env::var("INFERO_NCCL_DIR") {
        return PathBuf::from(p);
    }
    panic!(
        "the `nccl` feature needs INFERO_NCCL_DIR set to a prefix containing \
         include/nccl.h and lib/libnccl.so (e.g. /usr after \
         `apt install libnccl2 libnccl-dev`)"
    );
}

fn resolve_triton_aot_dir() -> PathBuf {
    if let Ok(p) = std::env::var("INFERO_TRITON_AOT_DIR") {
        return PathBuf::from(p);
    }
    panic!(
        "the `triton_aot` feature needs INFERO_TRITON_AOT_DIR set to the directory \
         `python3 -m triton.tools.compile` wrote its output into (a generated \
         `gdn_h.<hash>.c`/`.h` pair, e.g. /home/jeff/infero-gdn-bench/aot on `bw`) -- see \
         crates/kernels/src/gdn_triton_aot.rs's own doc comment for how that artifact is \
         produced and its own README_repro.sh for the exact triton.tools.compile invocation"
    );
}

/// Finds every `*.c` file `triton.tools.compile` wrote into `dir` -- one a
/// kernel (kernel-2's own `gdn_h.<hash>.c`, plus `gdn_fla_stages.rs`'s
/// `cumsum.<hash>.c`/`kkt.<hash>.c`/`solve_tril64.<hash>.c`/`wu.<hash>.c`/
/// `chunk_o.<hash>.c`). Each hash depends on that kernel's own exact
/// signature/config, so this globs rather than hardcoding names (the `extern
/// "C"` symbol names in `gdn_triton_aot.rs`/`gdn_fla_stages.rs` still
/// hardcode the *current* hashes, since that's the whole point of these being
/// fixed compiled artifacts -- a regenerated hash needs updating there too,
/// not just here). Every `.c` in the directory is expected to be one of these
/// generated files -- nothing else belongs there.
fn find_triton_aot_cs(dir: &Path) -> Vec<PathBuf> {
    let mut matches: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("reading INFERO_TRITON_AOT_DIR {}: {e}", dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "c"))
        .collect();
    matches.sort();
    if matches.is_empty() {
        panic!(
            "no *.c found in {} -- run that directory's own README_repro.sh (and \
             gdn_fla_stages.rs's aot2/compile_all.sh counterpart) first",
            dir.display()
        );
    }
    matches
}

/// A plain host C compiler for `triton_aot`'s generated `.c` file -- no
/// device code, so no `nvcc` needed for this compile step (only for locating
/// `cuda.h`/`ar`, via `resolve_nvcc`).
fn resolve_cc() -> PathBuf {
    if let Ok(p) = std::env::var("INFERO_CC") {
        return PathBuf::from(p);
    }
    PathBuf::from("cc")
}

fn compile_c(cc: &Path, src: &Path, cuda_include: Option<&Path>, obj: &Path) {
    let mut cmd = Command::new(cc);
    cmd.args(["-c", "-fPIC", "-O2"]);
    if let Some(inc) = cuda_include {
        cmd.arg("-I").arg(inc);
    }
    cmd.arg(src).arg("-o").arg(obj);
    let status = cmd.status().unwrap_or_else(|e| panic!("failed to run {}: {e}", cc.display()));
    if !status.success() {
        panic!("{} failed compiling {}", cc.display(), src.display());
    }
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
