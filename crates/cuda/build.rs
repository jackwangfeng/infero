use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=TUILI_CUDA_DIR");

    let dir = resolve_cuda_dir();
    let lib = dir.join("lib");
    let include = dir.join("include");

    if !lib.is_dir() || !include.is_dir() {
        panic!(
            "CUDA userspace not found at {}\nrun ./scripts/setup-cuda.sh (or set TUILI_CUDA_DIR)",
            dir.display()
        );
    }

    // cudarc dlopens libcublas/libnvrtc at runtime. Baking an rpath into every
    // binary that links this crate means neither `cargo run` nor the shipped
    // binary needs LD_LIBRARY_PATH set by hand.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());

    // NVRTC compiles kernels at runtime and needs the CUDA headers on its
    // include path (cuda_fp16.h, cuda_bf16.h, ...).
    println!("cargo:rustc-env=TUILI_CUDA_INCLUDE={}", include.display());
    println!("cargo:rustc-env=TUILI_CUDA_LIB={}", lib.display());
}

fn resolve_cuda_dir() -> PathBuf {
    if let Ok(d) = std::env::var("TUILI_CUDA_DIR") {
        return PathBuf::from(d);
    }
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor = workspace_root(&manifest).join("vendor/cuda");
    // Symlinks: canonicalize so the rpath survives a moved workspace.
    vendor.canonicalize().unwrap_or(vendor)
}

fn workspace_root(manifest: &Path) -> PathBuf {
    // crates/cuda -> crates -> <root>
    manifest
        .ancestors()
        .find(|p| p.join("Cargo.toml").is_file() && p.join("scripts").is_dir())
        .unwrap_or(manifest)
        .to_path_buf()
}
