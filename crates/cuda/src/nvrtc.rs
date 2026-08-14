use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::{CompileOptions, Ptx};

/// A launchable kernel. Just cudarc's function handle under a shorter name.
pub type Kernel = CudaFunction;

/// Let a kernel ask for more than 48 KiB of shared memory per block.
///
/// That cap is on *static* `__shared__` arrays and is not negotiable; the way
/// past it is `extern __shared__` plus this opt-in, which exists because the
/// extra comes out of the same store as L1. sm_86 allows 100 KiB per block of
/// the 128 KiB unified store.
pub fn set_max_dynamic_shared(func: &Kernel, bytes: u32) -> Result<()> {
    func.set_attribute(
        cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        bytes as i32,
    )
    .with_context(|| format!("kernel refused a {bytes}-byte dynamic shared request"))?;
    Ok(())
}

/// Compiles `.cu` sources with NVRTC and remembers the result.
///
/// Two layers of caching: PTX is written to `~/.cache/tuili/ptx` keyed by
/// (source, arch, options) so a restart doesn't pay for NVRTC again, and loaded
/// modules/functions are kept in memory so a launch is a hash lookup.
pub struct KernelCache {
    ctx: Arc<CudaContext>,
    arch: u32,
    dir: PathBuf,
    /// Keyed by module label. Sources are compile-time constants, so the label
    /// identifies them; the source hash only names the on-disk PTX.
    modules: Mutex<HashMap<&'static str, Arc<CudaModule>>>,
    /// Two levels so the hot lookup can borrow `&str` instead of allocating a
    /// key string on every launch.
    functions: Mutex<HashMap<&'static str, HashMap<String, Kernel>>>,
}

impl KernelCache {
    pub fn new(ctx: Arc<CudaContext>, arch: u32) -> Result<Self> {
        crate::loader::preload();
        let dir = cache_dir();
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating ptx cache dir {}", dir.display()))?;
        Ok(Self {
            ctx,
            arch,
            dir,
            modules: Mutex::new(HashMap::new()),
            functions: Mutex::new(HashMap::new()),
        })
    }

    /// Look up `func_name` in `src`, compiling and loading the module if needed.
    ///
    /// `label` is only used in log lines and cache filenames; correctness comes
    /// from the source hash, so two labels with identical source share a module.
    pub fn get(&self, label: &'static str, src: &str, func_name: &str) -> Result<Kernel> {
        // This runs on every launch — a decode step issues hundreds — so the
        // hit path must be one hash lookup. Hashing `src` here instead would
        // charge several microseconds per launch for a multi-kilobyte source,
        // which on a small model is most of the step.
        if let Some(f) = self
            .functions
            .lock()
            .unwrap()
            .get(label)
            .and_then(|m| m.get(func_name))
        {
            return Ok(f.clone());
        }

        let module = self.load_module(label, src)?;
        let func = module
            .load_function(func_name)
            .with_context(|| format!("kernel `{func_name}` not found in module `{label}`"))?;

        // Ask for the largest shared-memory share of the unified L1/shared
        // store. On sm_86 that store is 128 KiB and the driver's default split
        // is not the maximum, so a kernel wanting 32 KiB per block can end up
        // with one resident block per SM where it could have three. Advisory:
        // the driver may ignore it, and a kernel that fits either way does not
        // care.
        let _ = func.set_attribute(
            cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
            100,
        );

        self.functions
            .lock()
            .unwrap()
            .entry(label)
            .or_default()
            .insert(func_name.to_string(), func.clone());
        Ok(func)
    }

    fn load_module(&self, label: &'static str, src: &str) -> Result<Arc<CudaModule>> {
        if let Some(m) = self.modules.lock().unwrap().get(label) {
            return Ok(m.clone());
        }

        // Only on a miss, and only to name the cache file: the hash is what
        // makes an edited kernel recompile instead of loading stale PTX.
        let key = self.source_key(src);
        let path = self
            .dir
            .join(format!("{label}-sm{}-{key:016x}.ptx", self.arch));
        let ptx = match std::fs::read_to_string(&path) {
            Ok(text) => {
                tracing::debug!(kernel = label, "ptx cache hit");
                Ptx::from_src(text)
            }
            Err(_) => {
                let started = std::time::Instant::now();
                let ptx = cudarc::nvrtc::compile_ptx_with_opts(src, self.compile_options())
                    .with_context(|| format!("nvrtc failed to compile `{label}`"))?;
                tracing::debug!(
                    kernel = label,
                    ms = started.elapsed().as_millis(),
                    "nvrtc compiled"
                );
                // Best-effort: a failed write only costs a recompile next boot.
                let _ = std::fs::write(&path, ptx.to_src());
                ptx
            }
        };

        let module = self
            .ctx
            .load_module(ptx)
            .with_context(|| format!("loading module `{label}`"))?;
        self.modules.lock().unwrap().insert(label, module.clone());
        Ok(module)
    }

    fn compile_options(&self) -> CompileOptions {
        CompileOptions {
            include_paths: vec![crate::CUDA_INCLUDE_DIR.to_string()],
            options: vec![
                format!("--gpu-architecture=compute_{}", self.arch),
                "--std=c++17".to_string(),
                "-default-device".to_string(),
            ],
            use_fast_math: Some(true),
            ..Default::default()
        }
    }

    /// FNV-1a over the source plus everything that changes codegen.
    fn source_key(&self, src: &str) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |bytes: &[u8]| {
            for b in bytes {
                h ^= *b as u64;
                h = h.wrapping_mul(0x1000_0000_01b3);
            }
        };
        eat(src.as_bytes());
        eat(&self.arch.to_le_bytes());
        eat(env!("CARGO_PKG_VERSION").as_bytes());
        h
    }
}

fn cache_dir() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("tuili/ptx")
}
