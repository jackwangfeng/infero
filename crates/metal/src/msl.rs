use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use anyhow::{Result, anyhow};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLFunction, MTLLibrary};

use crate::launch::Function;

/// Runtime-compiled MSL, cached the way the NVRTC side caches PTX.
///
/// Two levels, because Metal has two: a *library* per module (one
/// `newLibraryWithSource:` per `common.metal` + `ops.metal` string), and a
/// *pipeline state* per kernel inside it. CUDA gets away with one level because
/// `CUfunction` is just a lookup in the loaded module; Metal has to build a
/// pipeline object, which is where the real compilation to machine code
/// happens, so that is the one worth caching hardest.
pub struct Modules {
    dev: Retained<ProtocolObject<dyn MTLDevice>>,
    libs: Mutex<HashMap<&'static str, Lib>>,
    /// Keyed by module, kernel name **and source hash**. Dropping the hash
    /// here was a real bug: the library cache invalidates on a source change
    /// but the pipeline built from the old library would still be served, so
    /// a recompiled kernel silently ran the previous code.
    pipelines: Mutex<HashMap<(&'static str, String, u64), Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
}

struct Lib {
    lib: Retained<ProtocolObject<dyn MTLLibrary>>,
    /// Hash of the source this was built from. A module whose source changed --
    /// which happens when a `#define` is prepended to strip pieces out for a
    /// measurement -- must not hit the cache built from the other version.
    src_hash: u64,
}

// SAFETY: as in `device.rs` -- libraries and pipeline states are thread-safe
// Metal objects, and the mutexes guard the maps rather than the objects.
unsafe impl Send for Modules {}
unsafe impl Sync for Modules {}

fn hash_of(s: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

impl Modules {
    pub(crate) fn new(dev: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            dev,
            libs: Mutex::new(HashMap::new()),
            pipelines: Mutex::new(HashMap::new()),
        }
    }

    /// Compile `src` as `module` if needed, then hand back `name` from it.
    ///
    /// Signature matches the CUDA side's `kernels().get(module, src, name)`.
    pub fn get(&self, module: &'static str, src: &str, name: &str) -> Result<Function> {
        let key = (module, name.to_string(), hash_of(src));
        if let Some(p) = self.pipelines.lock().unwrap().get(&key) {
            return Ok(Function::new(p.clone(), name.to_string()));
        }

        let lib = self.library(module, src)?;
        let ns_name = NSString::from_str(name);
        let func: Retained<ProtocolObject<dyn MTLFunction>> = lib
            .newFunctionWithName(&ns_name)
            .ok_or_else(|| anyhow!("module {module} has no kernel named {name}"))?;

        let pipeline = self
            .dev
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| anyhow!("{module}::{name}: pipeline: {e}"))?;

        self.pipelines
            .lock()
            .unwrap()
            .insert(key, pipeline.clone());
        Ok(Function::new(pipeline, name.to_string()))
    }

    fn library(
        &self,
        module: &'static str,
        src: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
        let want = hash_of(src);
        {
            let libs = self.libs.lock().unwrap();
            if let Some(l) = libs.get(module) {
                if l.src_hash == want {
                    return Ok(l.lib.clone());
                }
            }
        }

        let ns_src = NSString::from_str(src);
        let started = std::time::Instant::now();
        let lib = self
            .dev
            .newLibraryWithSource_options_error(&ns_src, None)
            .map_err(|e| {
                // The module name goes in the outermost message: `anyhow`'s
                // Display prints only that, and a compiler diagnostic with no
                // module attached is the same problem NVRTC has when its log
                // arrives without a source name.
                anyhow!(
                    "compiling MSL module {module} ({} bytes): {e}",
                    src.len()
                )
            })?;
        tracing::debug!(
            module,
            bytes = src.len(),
            ms = started.elapsed().as_millis(),
            "compiled MSL module"
        );

        self.libs.lock().unwrap().insert(
            module,
            Lib {
                lib: lib.clone(),
                src_hash: want,
            },
        );
        Ok(lib)
    }
}
