//! Make the vendored CUDA shared objects loadable without `LD_LIBRARY_PATH`.
//!
//! There is no system CUDA install here — the libraries come from the pip
//! wheels that `scripts/setup-cuda.sh` links into `vendor/cuda`. cudarc finds
//! them with `dlopen("libcublas.so.13")`, and libnvrtc in turn does the same
//! for its builtins helper. Neither lookup consults an rpath we control, and
//! `LD_LIBRARY_PATH` cannot be changed from inside a running process.
//!
//! What does work: dlopen dedupes by soname. Opening each file ourselves by
//! absolute path with `RTLD_GLOBAL` registers it, so every later lookup by
//! bare name resolves to the object we already loaded.

use std::path::Path;
use std::sync::Once;

use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};

/// Load order matters: a library's `DT_NEEDED` dependencies are resolved
/// through the normal search path, which is exactly what is broken here, so
/// dependencies have to already be resident.
const LOAD_ORDER: &[&str] = &[
    "libnvJitLink.so",
    "libcudart.so",
    "libnvrtc-builtins.so",
    "libnvrtc.so",
    "libcublasLt.so",
    "libcublas.so",
];

/// Preload the vendored CUDA libraries. Idempotent and safe to call from
/// anywhere; the first call does the work.
pub fn preload() {
    static ONCE: Once = Once::new();
    ONCE.call_once(preload_once);
}

fn preload_once() {
    let dir = Path::new(crate::CUDA_LIB_DIR);
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(dir = %dir.display(), error = %e, "cuda lib dir unreadable");
            return;
        }
    };

    let names: Vec<String> = entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();

    let mut loaded = 0usize;
    for prefix in LOAD_ORDER {
        for name in &names {
            // `libnvrtc.alt.so.13` is the older-PTX-ISA build; loading it would
            // claim a different soname and is never what we want.
            if !name.starts_with(prefix) || name.contains(".alt.") {
                continue;
            }
            let path = dir.join(name);
            match unsafe { Library::open(Some(&path), RTLD_NOW | RTLD_GLOBAL) } {
                Ok(lib) => {
                    // Deliberately leaked: these must outlive every CUDA call,
                    // and there is no point in the bookkeeping to unload them.
                    std::mem::forget(lib);
                    loaded += 1;
                    tracing::trace!(lib = %name, "preloaded");
                }
                Err(e) => tracing::debug!(lib = %name, error = %e, "preload failed"),
            }
        }
    }

    tracing::debug!(count = loaded, dir = %dir.display(), "cuda libraries preloaded");
}
