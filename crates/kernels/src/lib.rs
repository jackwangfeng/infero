//! The GPU operator set: everything a decoder-only transformer needs, and
//! nothing else.
//!
//! Kernels are plain CUDA C compiled by NVRTC at startup (see `infero-cuda`).
//! Weights stay in their GGUF block encoding on the device and are decoded
//! inside the kernel that consumes them, which is the whole reason a 7B model
//! fits in 5 GB.
//!
//! Shapes follow ggml's convention: a linear weight is `[n_out, k]` row-major,
//! so `out[t, r] = dot(w[r, :], x[t, :])`.

pub mod attn_backend;
pub mod awq;
#[cfg(feature = "nccl")]
mod cu_vendor;
#[cfg(feature = "cutlass")]
pub mod cutlass_fp8;
#[cfg(feature = "cutlass")]
pub use cutlass_fp8::CutlassWeight;
#[cfg(feature = "flash_attn2")]
pub mod flash_attn2;
pub mod fp8;
#[cfg(feature = "nccl")]
pub mod tp;
pub mod gdn;
pub mod turboquant;
pub mod vision;
mod weight;

use anyhow::{Context, Result};
use infero_gpu::{View, ViewMut, LaunchConfig, KernelArg};
use half::f16;
use infero_gpu::Device;

pub use attn_backend::{AttentionBackend, AttnCallCtx, HardwareCaps};
pub use BatchLayout as Batch;
pub use turboquant::{Codebook, DeviceTables as TqTables, KvQuant};
pub use weight::WeightType;

/// Widest MMQ block: four warps, eight weight rows each.
const MMQ_MAX_ROWS: u32 = 32;
/// Tokens per MMQ tile, fixed by the `m16n8k32` shape.
const MMQ_M: u32 = 16;
/// Padded byte stride of the f16 activation ring, and the one number the host
/// has to keep in step with `mmq.cu`: it sizes the dynamic shared request.
const MMQ_XF_STRIDE: u32 = 256 * 2 + 32;
/// The same for the `ldmatrix` shapes, which need 16 mod 128 where the scalar
/// gather needs 32. See `MMQ_XL_STRIDE` in `mmq.cu`.
const MMQ_XL_STRIDE: u32 = 256 * 2 + 16;
/// And for the Q8_1 ring: eight 36-byte blocks a row, padded. Matches
/// `MMQ_XA_STRIDE` in `mmq.cu`.
const MMQ_XA_STRIDE: u32 = 8 * 36 + 16;
/// The swizzled f16 ring needs no padding at all; see `MMQ_XK_STRIDE`.
const MMQ_XK_STRIDE: u32 = 256 * 2;

const COMMON_CUH: &str = include_str!("cu/common.cuh");

// The Metal twins. File for file where one exists; a stub where it does not,
// so that a kernel this backend has not got reports itself from the lookup as
// a missing name rather than as a compiler error about a file full of CUDA.
const COMMON_METAL: &str = include_str!("msl/common.metal");
const OPS_METAL: &str = include_str!("msl/ops.metal");
const QUANT_METAL: &str = include_str!("msl/quant.metal");
const GDN_METAL: &str = include_str!("msl/gdn.metal");
const MMVQ_METAL: &str = include_str!("msl/mmvq.metal");
const SAMPLE_METAL: &str = include_str!("msl/sample.metal");
const UNIMPLEMENTED_METAL: &str = include_str!("msl/unimplemented.metal");
const OPS_CU: &str = include_str!("cu/ops.cu");
const QUANT_CU: &str = include_str!("cu/quant.cu");
const TURBOQUANT_CU: &str = include_str!("cu/turboquant.cu");
const MMVQ_CU: &str = include_str!("cu/mmvq.cu");
const MMA_CUH: &str = include_str!("cu/mma.cuh");
const MMQ_CU: &str = include_str!("cu/mmq.cu");
const SAMPLE_CU: &str = include_str!("cu/sample.cu");
const GDN_CU: &str = include_str!("cu/gdn.cu");
const FP8_CU: &str = include_str!("cu/fp8.cu");
const VISION_CU: &str = include_str!("cu/vision.cu");
const MOE_CU: &str = include_str!("cu/moe.cu");

/// Threads per block for the reduction kernels. 256 keeps eight warps busy
/// without pushing occupancy off a cliff on sm_86.
const REDUCE_BLOCK: u32 = 256;

/// Tokens at which the Q4_K mat-vec switches to the matrix units. See the note
/// at the switch itself for the measurements behind the default.
fn mma_min() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("INFERO_MMA_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| *v > 0)
            .unwrap_or(8)
    })
}

/// Registers per thread in the fused norm; must match `RMS_REGS` in mmvq.cu.
const RMS_REGS: u32 = 8;
const ELEMENTWISE_BLOCK: u32 = 256;
/// Warps per block in the attention score kernel; each warp does one score.
const SCORE_WARPS: u32 = 4;
/// Tokens one mat-vec block serves, matching `GEMV_TOKENS` in `quant.cu`.
const GEMV_TOKENS_PER_BLOCK: u32 = 8;
/// `sizeof(block_q8_1)`: a `half2` scale/sum pair plus 32 int8 quants.
pub const Q8_1_BLOCK_BYTES: usize = 36;
/// Above this many tokens, `dequant_to_f16` + cuBLAS beats the tensor-core
/// GEMM: MMQ re-reads the weights once per 16- or 32-token tile while the
/// dequant path pays for its f16 copy exactly once, so the two cross over.
/// Measured at 96 on an A4000; see `mmq_tiles` for the companion threshold.
pub const MMQ_MAX_TOKENS: usize = 96;

#[cfg(feature = "cuda")]
fn ops_src() -> &'static str {
    static SRC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    // The MMA helpers too: `attn_decode_mma_f32` uses the same `m16n8k16`
    // fragments the GEMM does, and `mma.cuh` is where their layout is pinned.
    SRC.get_or_init(|| format!("{COMMON_CUH}\n{MMA_CUH}\n{OPS_CU}"))
}
#[cfg(not(feature = "cuda"))]
fn ops_src() -> &'static str {
    static SRC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SRC.get_or_init(|| format!("{COMMON_METAL}\n{OPS_METAL}"))
}

/// The GatedDeltaNet unit. Separate from `ops_src` so that a change to the
/// linear-attention kernels does not force every other kernel to recompile.
#[cfg(feature = "cuda")]
fn gdn_src() -> &'static str {
    static SRC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    // MMA_CUH: `gdn_chunk_uw_mma_f32` uses the same `m16n8k16` f16 tensor-core
    // fragments the attention/GEMM kernels do, for the chunk-local K·Kᵀ
    // system-matrix product -- see that kernel's own doc comment in `gdn.cu`.
    SRC.get_or_init(|| format!("{COMMON_CUH}\n{MMA_CUH}\n{GDN_CU}"))
}
#[cfg(not(feature = "cuda"))]
fn gdn_src() -> &'static str {
    static SRC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SRC.get_or_init(|| format!("{COMMON_METAL}\n{GDN_METAL}"))
}

/// The Qwen3.5 vision tower. Separate from `ops_src` for the same reason `gdn`
/// is, and for a second one: it reverses nearly every convention `ops.cu` is
/// built around (LayerNorm not RMSNorm, `[all q | all k | all v]` not per-head
/// interleaving, bidirectional not causal, two blocked rotary axes not three
/// interleaved), so keeping the two apart keeps a reader from picking the wrong
/// one by name.
#[cfg(feature = "cuda")]
fn vision_src() -> &'static str {
    static SRC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SRC.get_or_init(|| format!("{COMMON_CUH}\n{VISION_CU}"))
}
#[cfg(not(feature = "cuda"))]
fn vision_src() -> &'static str {
    static SRC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SRC.get_or_init(|| UNIMPLEMENTED_METAL.to_string())
}

/// The block-scaled FP8 unit.
#[cfg(feature = "cuda")]
fn fp8_src() -> &'static str {
    static SRC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    // `INFERO_FP8_STRIP` prepends `#define`s that take pieces out of the mat-vec,
    // for `examples/fp8_row_cost.rs` to attribute the marginal row's cost. The
    // defines change the source, so they change the NVRTC cache key — a stripped
    // build cannot be confused with a serving one.
    SRC.get_or_init(|| {
        format!(
            "{COMMON_CUH}\n{MMA_CUH}\n{}\n{FP8_CU}",
            crate::fp8::strip_flags()
        )
    })
}
#[cfg(not(feature = "cuda"))]
fn fp8_src() -> &'static str {
    static SRC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SRC.get_or_init(|| UNIMPLEMENTED_METAL.to_string())
}

#[cfg(feature = "cuda")]
fn sample_src() -> &'static str {
    static SRC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SRC.get_or_init(|| format!("{COMMON_CUH}\n{SAMPLE_CU}"))
}
#[cfg(not(feature = "cuda"))]
fn sample_src() -> &'static str {
    static SRC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SRC.get_or_init(|| format!("{COMMON_METAL}\n{SAMPLE_METAL}"))
}

#[cfg(feature = "cuda")]
fn quant_src() -> &'static str {
    static SRC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SRC.get_or_init(|| format!("{COMMON_CUH}\n{QUANT_CU}"))
}
#[cfg(not(feature = "cuda"))]
fn quant_src() -> &'static str {
    static SRC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SRC.get_or_init(|| format!("{COMMON_METAL}\n{QUANT_METAL}"))
}

#[cfg(feature = "cuda")]
fn mmq_src() -> &'static str {
    static SRC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SRC.get_or_init(|| format!("{COMMON_CUH}\n{MMA_CUH}\n{MMQ_CU}"))
}
#[cfg(not(feature = "cuda"))]
fn mmq_src() -> &'static str {
    static SRC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SRC.get_or_init(|| UNIMPLEMENTED_METAL.to_string())
}

#[cfg(feature = "cuda")]
fn mmvq_src() -> &'static str {
    static SRC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SRC.get_or_init(|| format!("{COMMON_CUH}\n{MMVQ_CU}"))
}
#[cfg(not(feature = "cuda"))]
fn mmvq_src() -> &'static str {
    static SRC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SRC.get_or_init(|| format!("{COMMON_METAL}\n{MMVQ_METAL}"))
}

/// The MoE kernels compile against `mmvq.cu`, not beside it: `mmvq_moe` is the
/// dense mat-vec with the weight base moved, so it uses the same `tq_dot_*`
/// devices functions and the same block layout. Duplicating those would let the
/// two drift, and a drifted dot product is a wrong answer rather than a
/// compile error.
///
/// CUDA-only: there is no Metal `MOE_CU` counterpart yet, matching the design
/// doc's own AWQ-first scope for the sparse path.
#[cfg(feature = "cuda")]
fn moe_src() -> &'static str {
    static SRC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SRC.get_or_init(|| format!("{COMMON_CUH}\n{MMVQ_CU}\n{MOE_CU}"))
}

#[cfg(feature = "cuda")]
fn tq_src() -> &'static str {
    static SRC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SRC.get_or_init(|| format!("{COMMON_CUH}\n{TURBOQUANT_CU}"))
}

#[cfg(not(feature = "cuda"))]
fn tq_src() -> &'static str {
    static SRC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SRC.get_or_init(|| UNIMPLEMENTED_METAL.to_string())
}

/// Threads for a kernel that walks one vector of `d` per block.
fn per_vector_block(d: usize) -> u32 {
    (d as u32).next_multiple_of(32).clamp(32, 1024)
}

fn elementwise(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(ELEMENTWISE_BLOCK).max(1), 1, 1),
        block_dim: (ELEMENTWISE_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub struct Kernels {
    dev: Device,
    /// One `CUtensorMap` per weight plane, keyed by its device pointer and
    /// shape. Building one is a host call of a few microseconds; a decode step
    /// wants 128 of them, so they are built once and kept.
    tma: std::sync::Mutex<std::collections::HashMap<(u64, usize, usize), TmaDesc>>,
}

/// A `CUtensorMap` is 128 opaque bytes that reach the kernel by value.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct TmaDesc([u8; 128]);
#[cfg(feature = "cuda")]
unsafe impl cudarc::driver::DeviceRepr for TmaDesc {}

/// Launch one of the register-resident norms, with or without the f16 copy.
///
/// The kernel's `__half*` is nullable, and a launch says "no second output" with
/// a zero of pointer width — kernel parameters are untyped bytes at this
/// boundary, so that is the same eight bytes a slice would have written. It is
/// worth the small ugliness: the FP8 projections read f32, and producing an f16
/// copy for them would be 10 KB a row written and never read.
///
/// `label` follows the caller rather than the kernel, so a profile still
/// separates the plain norm from the f16-writing one.
fn b_args(
    k: &Kernels,
    f: &infero_gpu::Function,
    cfg: LaunchConfig,
    out: &mut ViewMut<'_, f32>,
    h_out: Option<&mut ViewMut<'_, f16>>,
    x: &View<'_, f32>,
    weight: &View<'_, f32>,
    d: i32,
    eps: f32,
    label: &'static str,
) -> Result<()> {
    let mut b = k.device().stream().launch_builder(f);
    match h_out {
        Some(h) => {
            b.arg(out).arg(h).arg(x).arg(weight).arg(&d).arg(&eps);
        }
        None => {
            b.arg(out)
                .arg(&infero_gpu::NULL_BUFFER)
                .arg(x)
                .arg(weight)
                .arg(&d)
                .arg(&eps);
        }
    }
    k.device().profile().time(label, k.device().stream(), || {
        unsafe { b.launch(cfg) }.context(label)?;
        Ok(())
    })?;
    Ok(())
}

/// Device buffers for the surviving distribution a sampled row drew from.
///
/// Speculative decoding needs `q` as numbers, not just the token: the acceptance
/// test is `min(1, p(x)/q(x))` and a rejection draws from the normalized
/// `(p - q)+`. The support is the nucleus, a few dozen entries, so reading it
/// back is a kilobyte — against the 993 KB of logits the host would need to
/// reconstruct it, which measured a third of a draft's time.
pub struct Survivors<'a> {
    pub id: &'a mut ViewMut<'a, u32>,
    pub p: &'a mut ViewMut<'a, f32>,
    pub len: &'a mut ViewMut<'a, i32>,
    /// Entries a row. The kernel reports the true `keep` in `len` even when it
    /// exceeds this, so truncation is visible rather than silent.
    pub stride: usize,
}

/// Block size for the register-resident norms: enough threads that
/// `blockDim * RMS_REGS >= d`, rounded to a warp.
fn rms_block(d: usize) -> u32 {
    (d as u32).div_ceil(RMS_REGS).next_multiple_of(32).clamp(32, 1024)
}

/// Whether a row of `d` fits the register-resident norm at all.
fn rms_fits(d: usize) -> bool {
    d <= 1024 * RMS_REGS as usize
}

impl Kernels {
    pub fn new(dev: Device) -> Self {
        Self {
            dev,
            tma: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// Compile every kernel now instead of on first use, so the first token
    /// isn't charged for NVRTC.
    pub fn warm_up(&self) -> Result<()> {
        let started = std::time::Instant::now();
        for name in [
            "rms_norm_f32",
            "add_f32",
            "add_assign_f32",
            "add_bias_f32",
            "silu_mul_f32",
            "take_rows_f32",
            "f32_to_f16",
            "rope_neox_f32",
            "rope_norm_f32",
            "store_kv_f16",
            "write_slot_table",
            "attn_scores_f32",
            "attn_softmax_f32",
            "attn_output_f32",
            "silu_mul_split_f16_f32",
            "attn_flash_reduce_f16_f32",
            "rope_qk_packed_f32",
            "store_kv2_packed_f16",
            "qk_norm_f32",
        ] {
            self.dev.kernels().get("infero_ops", ops_src(), name)?;
        }
        // The GatedDeltaNet unit compiles separately, so warm it separately.
        // Skipping this would push a first-token latency spike into whichever
        // linear-attention layer ran first.
        for name in [
            "gdn_conv_f32",
            "gdn_gate_decay_f32",
            "gdn_qk_l2norm_f32",
            "gdn_delta_rule_f32",
            "gdn_gated_rmsnorm_f32",
            "sigmoid_gate_f32",
            "split_interleaved_f32",
        ] {
            self.dev.kernels().get("infero_gdn", gdn_src(), name)?;
        }
        // The FP8 unit, warmed the same way and for the same reason -- and
        // skipped where the hardware has no FP8 matmul. That is the same fact
        // `matmul` reads before it dispatches to these, so warming them anyway
        // would turn a capability the engine already routes around into a
        // startup failure.
        if self.dev.caps().fp8 {
        for name in [
            "mmv_f8_block_f32",
            // One per token width; see `fp8::BATCH_KERNELS`, which is what
            // dispatches, and which this list has to stay in step with — a name
            // missing here is a first-request NVRTC stall rather than an error,
            // so nothing would report it.
            "mmv_f8_block_batch2_f32",
            "mmv_f8_block_batch4_f32",
            "mmv_f8_block_batch8_f32",
            "mmv_f8_block_batch16_f32",
            "dequant_f8_block_f16",
        ] {
            self.dev.kernels().get("infero_fp8", fp8_src(), name)?;
        }
        }
        // And the vision tower, which is its own translation unit again. A
        // multimodal request pays for these once at startup instead of stalling
        // the first image behind NVRTC.
        //
        // CUDA-only for now, and for a different reason than FP8: nothing about
        // Apple hardware prevents these, `vision.cu` simply has no MSL twin
        // yet. Gating on the feature rather than on a capability says which of
        // the two it is.
        #[cfg(feature = "cuda")]
        for name in [
            "vision_layer_norm_f32",
            "vision_gelu_tanh_f32",
            "vision_gelu_erf_f32",
            "vision_rope_tables_f32",
            "vision_qkv_rope_f32",
            "vision_attn_f32",
            "vision_patchify_f32",
            "vision_add_pos_embed_f32",
            "vision_splice_f32",
        ] {
            self.dev.kernels().get("infero_vision", vision_src(), name)?;
        }
        // Every weight type's gemv, row gather and dequantisation.
        //
        // CUDA-only: this backend has MSL for a subset of the types, and
        // warm-up is an optimisation -- it moves a first-token compile stall to
        // startup. Warming a type the file will never contain would turn that
        // optimisation into a startup failure, so the Metal path pays the stall
        // for whichever types its checkpoint actually uses.
        #[cfg(feature = "cuda")]
        for ty in WeightType::ALL {
            // The transposed AWQ layout is read by the tensor-core GEMM and by
            // the prefill dequantization, and by nothing else yet — the
            // mat-vec and the float path still expect packed blocks.
            if ty == WeightType::Q4G128T {
                // No `gemv` or `gather_rows` for this layout: the float path
                // and the embedding gather never see it. The prefill
                // dequantization and the mat-vec do.
                self.dev.kernels().get(
                    "infero_quant",
                    quant_src(),
                    "dequant_q4_g128t_f16",
                )?;
                continue;
            }
            // The split Q8_0 layout is the batched vocab projection and
            // nothing else: it has a mat-vec and a dequantization, but it is
            // never an embedding table, so there is no row gather for it.
            let prefixes: &[&str] = if ty == WeightType::Q8_0S {
                &["gemv", "dequant"]
            } else {
                &["gemv", "gather_rows", "dequant"]
            };
            for prefix in prefixes {
                let name = match *prefix {
                    "dequant" => format!("dequant_{}_f16", ty.suffix()),
                    _ => format!("{prefix}_{}", ty.suffix()),
                };
                self.dev.kernels().get("infero_quant", quant_src(), &name)?;
            }
        }
        self.dev
            .kernels()
            .get("infero_mmvq", mmvq_src(), "quantize_q8_1_f32")?;
        for ty in WeightType::ALL.iter().filter(|t| Self::has_mmvq(**t)) {
            // Every width the dispatch can reach, not just the single-token
            // one: a missing kernel is otherwise a 500 on whichever request
            // happens to leave two sequences running.
            for tag in ["", "t1", "t2", "t4", "t8", "t16"] {
                let name = format!("mmvq{tag}_{}", ty.suffix());
                self.dev.kernels().get("infero_mmvq", mmvq_src(), &name)?;
            }
        }
        // CUDA-only from here: `mmq.cu` and `turboquant.cu` have no MSL
        // twins. The dispatch already routes around the first through
        // `caps().int_tensor_gemm`, and the second is only reached when
        // `--kv-quant` asks for a compressed cache.
        #[cfg(feature = "cuda")]
        {
            // The tensor-core GEMM, both tile widths. CUDA-only: `mmq.cu` has no
            // MSL twin, and `caps().int_tensor_gemm` already keeps the dispatch
            // away from it.
            //
            // Leaving these out made the
            // first request after startup pay for NVRTC — 20 tok/s against 27 on
            // every request after it.
            if self.dev.caps().int_tensor_gemm {
                for tag in ["", "2"] {
                    for ty in WeightType::ALL
                        .iter()
                        .filter(|t| Self::has_mmq(**t) && **t != WeightType::Q4G128T)
                    {
                        let name = format!("mmq{tag}_{}", ty.suffix());
                        self.dev.kernels().get("infero_mmq", mmq_src(), &name)?;
                    }
                }
            }
            for name in [
                "tq_matvec",
                "tq_store_v",
                "tq_store_k",
                "tq_attn_scores",
                "tq_attn_output",
            ] {
                self.dev.kernels().get("infero_turboquant", tq_src(), name)?;
            }
        }
        tracing::debug!(ms = started.elapsed().as_millis(), "kernels compiled");
        Ok(())
    }

    // ---- normalization and elementwise ----------------------------------

    /// `out[t, :] = rms_norm(x[t, :]) * weight`
    /// Per-head RMS norm in place, over the `d_head` lane of each head.
    ///
    /// `row_stride` and `offset` locate the heads inside `buf`: the fused QKV
    /// path leaves k packed in the `[q | k | v]` row, so it is normalized where
    /// it lies rather than after a scatter. Qwen3 applies this to q and k
    /// before the rotary; models without the weights skip the call entirely.
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm(
        &self,
        buf: &mut ViewMut<'_, f32>,
        weight: &View<'_, f32>,
        n_tokens: usize,
        n_heads: usize,
        d_head: usize,
        row_stride: usize,
        offset: usize,
        eps: f32,
    ) -> Result<()> {
        debug_assert!(weight.len() >= d_head);
        debug_assert!(buf.len() >= (n_tokens - 1) * row_stride + offset + n_heads * d_head);
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "qk_norm_f32")?;
        let cfg = LaunchConfig {
            grid_dim: ((n_tokens * n_heads) as u32, 1, 1),
            block_dim: (REDUCE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (nh, dh) = (n_heads as i32, d_head as i32);
        let (rs, off) = (row_stride as i32, offset as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(buf)
            .arg(weight)
            .arg(&nh)
            .arg(&dh)
            .arg(&rs)
            .arg(&off)
            .arg(&eps);
        self.dev.profile().time("qk_norm", self.dev.stream(), || {
            unsafe { b.launch(cfg) }.context("qk_norm")?;
            Ok(())
        })?;
        Ok(())
    }

    pub fn rms_norm(
        &self,
        out: &mut ViewMut<'_, f32>,
        x: &View<'_, f32>,
        weight: &View<'_, f32>,
        n_tokens: usize,
        d: usize,
        eps: f32,
    ) -> Result<()> {
        debug_assert!(out.len() >= n_tokens * d && x.len() >= n_tokens * d);
        // The register-resident kernel whenever the row fits it, which is every
        // `d_model` in these models. `rms_norm_f32` reads the row, reduces, then
        // reads it *again* to scale; this one keeps it in registers across the
        // reduction and reads once. It was written for the f16-writing path and
        // the only thing stopping the f32 callers from having it was that its
        // second output was mandatory. Measured on the 27B: 14.1 us a launch
        // against 5.4, 140 launches a decode step.
        if rms_fits(d) {
            return self.rms_norm_f16(out, None, x, weight, n_tokens, d, eps);
        }
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "rms_norm_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (n_tokens as u32, 1, 1),
            block_dim: (REDUCE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (d_i, eps_f) = (d as i32, eps);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(x).arg(weight).arg(&d_i).arg(&eps_f);
        self.dev.profile().time("rms_norm", self.dev.stream(), || {
            unsafe { b.launch(cfg) }.context("rms_norm")?;
            Ok(())
        })?;
        Ok(())
    }

    /// `out = a + b`, elementwise. Aliasing `out` with either input is fine.
    pub fn add(
        &self,
        out: &mut ViewMut<'_, f32>,
        a: &View<'_, f32>,
        b_in: &View<'_, f32>,
        n: usize,
    ) -> Result<()> {
        let f = self.dev.kernels().get("infero_ops", ops_src(), "add_f32")?;
        let n_i = n as i32;
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(a).arg(b_in).arg(&n_i);
        self.dev.profile().time("add", self.dev.stream(), || {
            unsafe { b.launch(elementwise(n as u32)) }.context("add")?;
            Ok(())
        })?;
        Ok(())
    }

    /// `out += b`, in place.
    pub fn add_assign(
        &self,
        out: &mut ViewMut<'_, f32>,
        b_in: &View<'_, f32>,
        n: usize,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "add_assign_f32")?;
        let n_i = n as i32;
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(b_in).arg(&n_i);
        self.dev
            .profile()
            .time("add_assign", self.dev.stream(), || {
                // Four elements a thread; see `add_assign_f32`.
                unsafe { b.launch(elementwise((n as u32).div_ceil(4))) }.context("add_assign")?;
                Ok(())
            })?;
        Ok(())
    }

    /// `out[t, j] += bias[j]`
    pub fn add_bias(
        &self,
        out: &mut ViewMut<'_, f32>,
        bias: &View<'_, f32>,
        n_cols: usize,
        n_rows: usize,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "add_bias_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (
                (n_cols as u32).div_ceil(ELEMENTWISE_BLOCK).max(1),
                n_rows as u32,
                1,
            ),
            block_dim: (ELEMENTWISE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (c, r) = (n_cols as i32, n_rows as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(bias).arg(&c).arg(&r);
        self.dev.profile().time("add_bias", self.dev.stream(), || {
            unsafe { b.launch(cfg) }.context("add_bias")?;
            Ok(())
        })?;
        Ok(())
    }

    /// `out = silu(gate) * up`, the SwiGLU non-linearity.
    pub fn silu_mul(
        &self,
        out: &mut ViewMut<'_, f32>,
        gate: &View<'_, f32>,
        up: &View<'_, f32>,
        n: usize,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "silu_mul_f32")?;
        let n_i = n as i32;
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(gate).arg(up).arg(&n_i);
        self.dev.profile().time("silu_mul", self.dev.stream(), || {
            unsafe { b.launch(elementwise(n as u32)) }.context("silu_mul")?;
            Ok(())
        })?;
        Ok(())
    }

    /// Scatter a fused `q ++ k ++ v` result into three tensors.
    ///
    /// `fused` is `[tokens][d + 2 * kv_dim]`, which is what one matmul against
    /// the stacked weight produces.
    pub fn split_qkv(
        &self,
        q: &mut ViewMut<'_, f32>,
        k: &mut ViewMut<'_, f32>,
        v: &mut ViewMut<'_, f32>,
        fused: &View<'_, f32>,
        d: usize,
        kv_dim: usize,
        n_tokens: usize,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "split_qkv_f32")?;
        let total = n_tokens * (d + 2 * kv_dim);
        let (d_i, kv_i, t_i) = (d as i32, kv_dim as i32, total as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(q).arg(k).arg(v).arg(fused).arg(&d_i).arg(&kv_i).arg(&t_i);
        self.dev
            .profile()
            .time("split_qkv", self.dev.stream(), || {
                unsafe { b.launch(elementwise(total as u32)) }.context("split_qkv")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Scatter a fused `a ++ b` result into two tensors — [`Kernels::split_qkv`]
    /// with two outputs instead of three.
    ///
    /// `fused` is `[tokens][width_a + width_b]`, what one matmul against a
    /// stacked weight produces.
    pub fn split2(
        &self,
        a: &mut ViewMut<'_, f32>,
        b: &mut ViewMut<'_, f32>,
        fused: &View<'_, f32>,
        width_a: usize,
        width_b: usize,
        n_tokens: usize,
    ) -> Result<()> {
        let f = self.dev.kernels().get("infero_ops", ops_src(), "split2_f32")?;
        let total = n_tokens * (width_a + width_b);
        let (a_i, b_i, t_i) = (width_a as i32, width_b as i32, total as i32);
        let mut bld = self.dev.stream().launch_builder(&f);
        bld.arg(a).arg(b).arg(fused).arg(&a_i).arg(&b_i).arg(&t_i);
        self.dev.profile().time("split2", self.dev.stream(), || {
            unsafe { bld.launch(elementwise(total as u32)) }.context("split2")?;
            Ok(())
        })?;
        Ok(())
    }

    /// [`Kernels::silu_mul`] over one fused `gate ++ up` row.
    ///
    /// `xy` is `[tokens][2 * d_ff]`, which is what a single matmul against the
    /// concatenated weight produces.
    pub fn silu_mul_split(
        &self,
        out: &mut ViewMut<'_, f32>,
        xy: &View<'_, f32>,
        d_ff: usize,
        total: usize,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "silu_mul_split_f32")?;
        let (d_i, t_i) = (d_ff as i32, total as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(xy).arg(&d_i).arg(&t_i);
        self.dev.profile().time("silu_mul_split", self.dev.stream(), || {
            unsafe { b.launch(elementwise(total as u32)) }.context("silu_mul_split")?;
            Ok(())
        })?;
        Ok(())
    }

    /// [`Self::silu_mul_split`] also writing the f16 copy `down_proj` reads.
    ///
    /// One launch instead of two: the separate `to_f16` over the same row was
    /// 1.2 us a layer in the trace, against a 512 KB f16 write here that the
    /// kernel is already positioned to do.
    pub fn silu_mul_split_f16(
        &self,
        out: &mut ViewMut<'_, f32>,
        hout: &mut ViewMut<'_, f16>,
        xy: &View<'_, f32>,
        d_ff: usize,
        total: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            hout.len() >= total,
            "f16 scratch holds {} of {total} elements",
            hout.len()
        );
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "silu_mul_split_f16_f32")?;
        let (d_i, t_i) = (d_ff as i32, total as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(hout).arg(xy).arg(&d_i).arg(&t_i);
        self.dev.profile().time("silu_mul_split_f16", self.dev.stream(), || {
            unsafe { b.launch(elementwise(total as u32)) }.context("silu_mul_split_f16")?;
            Ok(())
        })?;
        Ok(())
    }

    /// `out[r, :] = x[rows[r], :]`
    /// Threads per sampling block; must match `SAMPLE_BLOCK` in `sample.cu`.
    const SAMPLE_BLOCK: u32 = 256;

    /// The largest `top_k` the device sampler will take.
    ///
    /// Survivors live in shared memory and are found by that many block-wide
    /// passes, so the bound is what keeps both in hand. Past it the host path
    /// runs instead, which is why it is a public predicate rather than an
    /// assertion.
    pub const SAMPLE_MAX_TOP_K: usize = 256;

    /// Whether a batch can be sampled on the device at all.
    ///
    /// The vocabulary bitset is dynamic shared memory, so a large enough
    /// vocabulary rules the kernel out on a card whose limit is 48 KiB without
    /// an opt-in this does not take.
    pub fn can_sample_on_device(vocab: usize, max_top_k: usize) -> bool {
        // The limit is the backend's threadgroup/shared budget, and Apple's is
        // 32 KiB against a CUDA card's 48. That is not a detail: the bitset is
        // `vocab / 32` words, so 151936 tokens fit on both and 248320 fit on
        // neither once the reduction scratch is added -- Qwen3.8 lands in the
        // gap. `sample_rows_split`, whose bitset covers one sixty-fourth of the
        // vocabulary, is what serves the ones this excludes, and the caller
        // reaching for it does not consult this.
        let budget = if cfg!(feature = "cuda") { 48 * 1024 } else { 32 * 1024 };
        max_top_k <= Self::SAMPLE_MAX_TOP_K && Self::sample_shared(vocab) <= budget
    }

    fn sample_shared(vocab: usize) -> u32 {
        let words = vocab.div_ceil(32) as u32;
        // The bitset, then the reduction scratch, then the survivors.
        words * 4 + Self::SAMPLE_BLOCK * 4 * 4
    }

    /// Slices a row's vocabulary takes on the greedy path.
    ///
    /// 32 rows a step at one block each is 2% of a 188-SM card, so the scan runs
    /// at 94 GB/s. Thirty-two slices make it 1024 blocks, which fills the device
    /// at every batch width this engine serves.
    pub const ARGMAX_SPLITS: usize = 32;

    /// Vocabulary slices [`Self::sample_rows_split`] fans out over. Has to match
    /// `SAMPLE_SPLITS` in the kernel, which lays out the candidate buffer.
    pub const SAMPLE_SPLITS: usize = 64;

    /// [`Self::sample_rows`] when every row is greedy: the vocabulary scan split
    /// across the device and the winners reduced by a second kernel.
    ///
    /// `pv`/`pi` hold `n_rows * ARGMAX_SPLITS` slice winners. The answer is the
    /// single-block kernel's, token for token — both passes order candidates
    /// with `samp_better`, so the lowest index still wins a tie.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_rows_greedy(
        &self,
        out: &mut ViewMut<'_, u32>,
        pv: &mut ViewMut<'_, f32>,
        pi: &mut ViewMut<'_, i32>,
        logits: &View<'_, f32>,
        params: &View<'_, f32>,
        pen_tok: &View<'_, i32>,
        pen_cnt: &View<'_, i32>,
        pen_len: &View<'_, i32>,
        n_rows: usize,
        vocab: usize,
        pen_stride: usize,
    ) -> Result<()> {
        let splits = Self::ARGMAX_SPLITS;
        anyhow::ensure!(
            pv.len() >= n_rows * splits && pi.len() >= n_rows * splits,
            "argmax scratch holds {} of {} slots",
            pv.len().min(pi.len()),
            n_rows * splits
        );
        let chunk = vocab.div_ceil(splits);
        let part_shared = ((chunk.div_ceil(32) + 1) * 4) as u32 + Self::SAMPLE_BLOCK * 2 * 4;
        let f = self
            .dev
            .kernels()
            .get("infero_sample", sample_src(), "argmax_partial_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (splits as u32, n_rows as u32, 1),
            block_dim: (Self::SAMPLE_BLOCK, 1, 1),
            shared_mem_bytes: part_shared,
        };
        let (v, ps, sp) = (vocab as i32, pen_stride as i32, splits as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        // Reborrowed, so the second pass can still read what this one writes.
        b.arg(&mut *pv)
            .arg(&mut *pi)
            .arg(logits)
            .arg(params)
            .arg(pen_tok)
            .arg(pen_cnt)
            .arg(pen_len)
            .arg(&v)
            .arg(&ps)
            .arg(&sp);
        self.dev
            .profile()
            .time("argmax_partial", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("argmax_partial")?;
                Ok(())
            })?;

        let g = self
            .dev
            .kernels()
            .get("infero_sample", sample_src(), "argmax_combine_f32")?;
        let cfg2 = LaunchConfig {
            grid_dim: (n_rows as u32, 1, 1),
            block_dim: (Self::SAMPLE_BLOCK, 1, 1),
            shared_mem_bytes: Self::SAMPLE_BLOCK * 2 * 4,
        };
        let pvv = pv.as_view();
        let piv = pi.as_view();
        let mut b2 = self.dev.stream().launch_builder(&g);
        b2.arg(out).arg(&pvv).arg(&piv).arg(&sp);
        self.dev
            .profile()
            .time("argmax_combine", self.dev.stream(), || {
                unsafe { b2.launch(cfg2) }.context("argmax_combine")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Sample one token per row without the logits ever leaving the device.
    ///
    /// `pen_tok` and `pen_cnt` are each row's repetition window as sorted
    /// unique ids and their counts, padded to `pen_stride`; `rnd` is one
    /// uniform draw per row, taken from that sequence's own generator on the
    /// host so that seeding stays reproducible.
    #[allow(clippy::too_many_arguments)]
    /// Where [`Self::sample_rows`] writes the distribution it drew from.
    ///
    /// `stride` bounds one row's entries; the kernel writes `min(keep, stride)`
    /// and reports `keep`, so a caller that sized `stride` below its `top_k`
    /// will see the truncation in `len` rather than silently losing mass.
    /// Splits a row's vocabulary across blocks, so the top-k is one pass.
    ///
    /// [`Self::sample_rows`] scans the vocabulary once per survivor. At a 248320
    /// -token vocabulary and `top_k = 40` that is forty passes in one block, and
    /// it measured 5.99 ms against 0.71 for the host doing the same work — which
    /// is why a speculative draft still copies its logits back. This is the same
    /// answer `sample_rows_greedy` uses for `k = 1`: each of `SAMPLE_SPLITS`
    /// blocks emits its slice's top-k, then one block merges. A token in the
    /// global top-k is in its own slice's top-k, so nothing is lost.
    ///
    /// `cand_v`/`cand_i` need `n_rows * SAMPLE_SPLITS * top_k` entries.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_rows_split(
        &self,
        out: &mut ViewMut<'_, u32>,
        cand_v: &mut ViewMut<'_, f32>,
        cand_i: &mut ViewMut<'_, i32>,
        logits: &View<'_, f32>,
        params: &View<'_, f32>,
        pen_tok: &View<'_, i32>,
        pen_cnt: &View<'_, i32>,
        pen_len: &View<'_, i32>,
        rnd: &View<'_, f64>,
        n_rows: usize,
        vocab: usize,
        pen_stride: usize,
        top_k: usize,
        survivors: Option<Survivors<'_>>,
    ) -> Result<()> {
        let cand_k = top_k.max(1);
        debug_assert!(cand_v.len() >= n_rows * Self::SAMPLE_SPLITS * cand_k);
        let (v, ps, ck) = (vocab as i32, pen_stride as i32, cand_k as i32);
        // Stage one: the penalty bitset covers a slice, not the vocabulary.
        let per = vocab.div_ceil(Self::SAMPLE_SPLITS);
        let words = per.div_ceil(32);
        let sh1 = (words * 4 + 256 * 4 * 2) as u32;
        let f1 = self
            .dev
            .kernels()
            .get("infero_sample", sample_src(), "sample_topk_partial_f32")?;
        let cfg1 = LaunchConfig {
            grid_dim: (n_rows as u32, Self::SAMPLE_SPLITS as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: sh1,
        };
        let mut b1 = self.dev.stream().launch_builder(&f1);
        b1.arg(&mut *cand_v)
            .arg(&mut *cand_i)
            .arg(logits)
            .arg(params)
            .arg(pen_tok)
            .arg(pen_cnt)
            .arg(pen_len)
            .arg(&v)
            .arg(&ps)
            .arg(&ck);
        self.dev
            .profile()
            .time("sample_topk_partial", self.dev.stream(), || {
                unsafe { b1.launch(cfg1) }.context("sample_topk_partial")?;
                Ok(())
            })?;

        // Stage two: merge, then the same tail `sample_rows` runs.
        let f2 = self
            .dev
            .kernels()
            .get("infero_sample", sample_src(), "sample_rows_topk_f32")?;
        let cfg2 = LaunchConfig {
            grid_dim: (n_rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (256 * 4 * 4) as u32,
        };
        let sstride = survivors.as_ref().map_or(0, |x| x.stride) as i32;
        let mut b2 = self.dev.stream().launch_builder(&f2);
        // The candidates are read-only here; the mutable views were stage one's.
        let cv = cand_v.as_view();
        let ci = cand_i.as_view();
        b2.arg(out)
            .arg(&cv)
            .arg(&ci)
            .arg(params)
            .arg(rnd)
            .arg(&v)
            .arg(&ck);
        match survivors {
            Some(x) => {
                b2.arg(x.id).arg(x.p).arg(x.len);
            }
            None => {
                b2.arg(&infero_gpu::NULL_BUFFER).arg(&infero_gpu::NULL_BUFFER).arg(&infero_gpu::NULL_BUFFER);
            }
        }
        b2.arg(&sstride);
        self.dev
            .profile()
            .time("sample_rows_topk", self.dev.stream(), || {
                unsafe { b2.launch(cfg2) }.context("sample_rows_topk")?;
                Ok(())
            })?;
        Ok(())
    }

    pub fn sample_rows(
        &self,
        out: &mut ViewMut<'_, u32>,
        logits: &View<'_, f32>,
        params: &View<'_, f32>,
        pen_tok: &View<'_, i32>,
        pen_cnt: &View<'_, i32>,
        pen_len: &View<'_, i32>,
        rnd: &View<'_, f64>,
        n_rows: usize,
        vocab: usize,
        pen_stride: usize,
        // Optional: the surviving distribution each draw was made from — see
        // `Survivors` and the kernel.
        survivors: Option<Survivors<'_>>,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_sample", sample_src(), "sample_rows_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (n_rows as u32, 1, 1),
            block_dim: (Self::SAMPLE_BLOCK, 1, 1),
            shared_mem_bytes: Self::sample_shared(vocab),
        };
        let v = vocab as i32;
        let ps = pen_stride as i32;
        let mut b = self.dev.stream().launch_builder(&f);
        let sstride = survivors.as_ref().map_or(0, |v| v.stride) as i32;
        b.arg(out)
            .arg(logits)
            .arg(params)
            .arg(pen_tok)
            .arg(pen_cnt)
            .arg(pen_len)
            .arg(rnd)
            .arg(&v)
            .arg(&ps);
        match survivors {
            Some(v) => {
                b.arg(v.id).arg(v.p).arg(v.len);
            }
            None => {
                b.arg(&infero_gpu::NULL_BUFFER).arg(&infero_gpu::NULL_BUFFER).arg(&infero_gpu::NULL_BUFFER);
            }
        }
        b.arg(&sstride);
        self.dev
            .profile()
            .time("sample_rows", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("sample_rows")?;
                Ok(())
            })?;
        Ok(())
    }

    pub fn take_rows(
        &self,
        out: &mut ViewMut<'_, f32>,
        x: &View<'_, f32>,
        rows: &View<'_, i32>,
        n_rows: usize,
        d: usize,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "take_rows_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (
                (d as u32).div_ceil(ELEMENTWISE_BLOCK).max(1),
                n_rows as u32,
                1,
            ),
            block_dim: (ELEMENTWISE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let d_i = d as i32;
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(x).arg(rows).arg(&d_i);
        self.dev
            .profile()
            .time("take_rows", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("take_rows")?;
                Ok(())
            })?;
        Ok(())
    }

    pub fn to_f16(
        &self,
        out: &mut ViewMut<'_, f16>,
        x: &View<'_, f32>,
        n: usize,
    ) -> Result<()> {
        self.to_f16_inner(out, x, n, false)
    }

    /// [`Self::to_f16`] writing k in the order an `ldmatrix`-loaded A fragment
    /// pairs with a weight word read straight out of an AWQ pack.
    ///
    /// See `f32_to_f16_kperm` in `ops.cu`: the alternative is repacking the
    /// weights, and the activations are a thousandth of the bytes.
    pub fn to_f16_kperm(
        &self,
        out: &mut ViewMut<'_, f16>,
        x: &View<'_, f32>,
        n: usize,
    ) -> Result<()> {
        self.to_f16_inner(out, x, n, true)
    }

    /// The inverse of [`Self::to_f16`] -- for a caller holding real f16 data
    /// from outside this crate's own kernels (a vendor FFI kernel's own
    /// output, e.g.) that needs it back in f32 without a host round-trip.
    pub fn from_f16(&self, out: &mut ViewMut<'_, f32>, x: &View<'_, f16>, n: usize) -> Result<()> {
        let f = self.dev.kernels().get("infero_ops", ops_src(), "f16_to_f32")?;
        let n_i = n as i32;
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(x).arg(&n_i);
        let threads = (n as u32).div_ceil(4);
        self.dev.profile().time("from_f16", self.dev.stream(), || {
            unsafe { b.launch(elementwise(threads)) }.context("from_f16")?;
            Ok(())
        })?;
        Ok(())
    }

    fn to_f16_inner(
        &self,
        out: &mut ViewMut<'_, f16>,
        x: &View<'_, f32>,
        n: usize,
        kperm: bool,
    ) -> Result<()> {
        let name = if kperm { "f32_to_f16_kperm" } else { "f32_to_f16" };
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), name)?;
        let n_i = n as i32;
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(x).arg(&n_i);
        // `f32_to_f16` takes four elements a thread — see the note on it — while
        // the k-permuted variant still takes one. One wrapper, two grids.
        let threads = if kperm { n as u32 } else { (n as u32).div_ceil(4) };
        self.dev.profile().time("to_f16", self.dev.stream(), || {
            unsafe { b.launch(elementwise(threads)) }.context("to_f16")?;
            Ok(())
        })?;
        Ok(())
    }

    // ---- positional -----------------------------------------------------

    /// Rotary embeddings applied in place to `x[n_tokens, n_heads, d_head]`.
    ///
    /// `interleaved` selects the pairing: false pairs `i` with `i + d/2`
    /// (NeoX, what Qwen2 wants), true pairs `2i` with `2i+1` (what llama-family
    /// GGUFs want, their Q and K having been permuted to suit).
    #[allow(clippy::too_many_arguments)]
    /// Rotary embeddings for Q and K in one launch.
    ///
    /// See [`Kernels::rope`] for the conventions; this differs only in doing
    /// both tensors at once, which at a batch of one turns two grids of eight
    /// and thirty-two blocks into one of forty.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_qk(
        &self,
        q: &mut ViewMut<'_, f32>,
        k: &mut ViewMut<'_, f32>,
        positions: &View<'_, i32>,
        freq_factors: &View<'_, f32>,
        mrope_axis: &View<'_, i32>,
        pos_stride: usize,
        n_tokens: usize,
        n_heads: usize,
        n_kv_heads: usize,
        d_head: usize,
        theta_base: f32,
        freq_scale: f32,
        interleaved: bool,
    ) -> Result<()> {
        self.rope_qk_partial(
            q,
            k,
            positions,
            freq_factors,
            mrope_axis,
            pos_stride,
            n_tokens,
            n_heads,
            n_kv_heads,
            d_head,
            d_head,
            theta_base,
            freq_scale,
            interleaved,
        )
    }

    /// [`Self::rope_qk`] rotating only the first `rotary_dim` of each head.
    ///
    /// `rotary_dim == d_head` is the full-width case and reduces to exactly the
    /// launch and the arithmetic that shipped before this existed. Below that,
    /// dimensions `[rotary_dim, d_head)` are not addressed at all — both
    /// tensors are rotated in place, so their tails keep their bits — and the
    /// frequency exponent is normalized by `rotary_dim` rather than `d_head`,
    /// which makes the table a compression of the full frequency span into
    /// fewer dimensions rather than its leading slice.
    /// `mrope_axis`/`pos_stride`: `positions` normally holds one value a
    /// token (`pos_stride == 1`) and every entry of `mrope_axis` is `0`, which
    /// reduces the read to exactly `positions[token]` -- the plain-position
    /// arithmetic, unchanged. A model with Qwen3.5-style M-RoPE instead sets
    /// `pos_stride == 3` (`positions` holds a time/height/width triple a
    /// token) and fills `mrope_axis[i]` with which of the three frequency `i`
    /// reads, from `qwen35_vision::interleaved_mrope_axis`. Both `positions`
    /// and `mrope_axis` must be real, non-null buffers either way — Metal has
    /// no null buffer argument.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_qk_partial(
        &self,
        q: &mut ViewMut<'_, f32>,
        k: &mut ViewMut<'_, f32>,
        positions: &View<'_, i32>,
        freq_factors: &View<'_, f32>,
        mrope_axis: &View<'_, i32>,
        pos_stride: usize,
        n_tokens: usize,
        n_heads: usize,
        n_kv_heads: usize,
        d_head: usize,
        rotary_dim: usize,
        theta_base: f32,
        freq_scale: f32,
        interleaved: bool,
    ) -> Result<()> {
        anyhow::ensure!(
            d_head.is_multiple_of(2),
            "d_head {d_head} must be even for rope"
        );
        anyhow::ensure!(
            rotary_dim.is_multiple_of(2) && rotary_dim >= 2 && rotary_dim <= d_head,
            "rotary_dim {rotary_dim} must be even and in 2..={d_head}"
        );
        anyhow::ensure!(
            freq_factors.len() >= rotary_dim / 2,
            "freq_factors holds {} entries, short of the {} pairs that rotate",
            freq_factors.len(),
            rotary_dim / 2
        );
        anyhow::ensure!(
            pos_stride == 1 || pos_stride == 3,
            "pos_stride {pos_stride} must be 1 (plain position) or 3 (mRoPE T/H/W)"
        );
        anyhow::ensure!(
            mrope_axis.len() >= rotary_dim / 2,
            "mrope_axis holds {} entries, short of the {} frequencies that rotate",
            mrope_axis.len(),
            rotary_dim / 2
        );
        anyhow::ensure!(
            positions.len() >= n_tokens * pos_stride,
            "positions holds {} entries, short of {n_tokens} tokens x pos_stride {pos_stride}",
            positions.len()
        );
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "rope_qk_f32")?;
        let half = (rotary_dim / 2) as u32;
        let block = half.clamp(1, 128);
        let cfg = LaunchConfig {
            grid_dim: (
                half.div_ceil(block),
                (n_heads + n_kv_heads) as u32,
                n_tokens as u32,
            ),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (h, kh, dh, rd, il, ps) = (
            n_heads as i32,
            n_kv_heads as i32,
            d_head as i32,
            rotary_dim as i32,
            i32::from(interleaved),
            pos_stride as i32,
        );
        // Argument order matches the kernel's parameter order exactly, which
        // for both backends puts `mrope_axis`/`pos_stride` last -- see the
        // comment on the CUDA kernel for why they aren't next to `positions`
        // where they'd read more naturally.
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(q)
            .arg(k)
            .arg(positions)
            .arg(freq_factors)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&rd)
            .arg(&theta_base)
            .arg(&freq_scale)
            .arg(&il)
            .arg(mrope_axis)
            .arg(&ps);
        self.dev.profile().time("rope_qk", self.dev.stream(), || {
            unsafe { b.launch(cfg) }.context("rope_qk")?;
            Ok(())
        })?;
        Ok(())
    }

    /// [`Self::rope_qk`] reading out of the stacked projection's output row.
    ///
    /// `packed` holds `q`, `k` and `v` a `stride` apart per token, which is what
    /// the fused `qkv` matmul writes. `q` lands in its own buffer because
    /// attention reads it contiguously; `k` is rotated in place for
    /// [`Self::store_kv2_packed`] to read next to `v`. Saves the unpacking copy
    /// — 1.5 MB and a launch a layer.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_qk_packed(
        &self,
        q_dst: &mut ViewMut<'_, f32>,
        packed: &mut ViewMut<'_, f32>,
        stride: usize,
        q_off: usize,
        k_off: usize,
        positions: &View<'_, i32>,
        freq_factors: &View<'_, f32>,
        mrope_axis: &View<'_, i32>,
        pos_stride: usize,
        n_tokens: usize,
        n_heads: usize,
        n_kv_heads: usize,
        d_head: usize,
        theta_base: f32,
        freq_scale: f32,
        interleaved: bool,
    ) -> Result<()> {
        self.rope_qk_packed_partial(
            q_dst,
            packed,
            stride,
            q_off,
            k_off,
            positions,
            freq_factors,
            mrope_axis,
            pos_stride,
            n_tokens,
            n_heads,
            n_kv_heads,
            d_head,
            d_head,
            theta_base,
            freq_scale,
            interleaved,
        )
    }

    /// [`Self::rope_qk_packed`] rotating only the first `rotary_dim` of each
    /// head.
    ///
    /// `k` is rotated in place, so its unrotated tail stays where it is. `q`
    /// is not: it is read out of the packed row and written to `q_dst`, so the
    /// tail has to be *copied*, and this launch carries
    /// `d_head - rotary_dim` extra lanes per q head to do it. Omitting that
    /// copy is the one new way to be wrong here that still runs — `q_dst` would
    /// keep the previous layer's values past `rotary_dim`, which on the 27B is
    /// three quarters of every query.
    #[allow(clippy::too_many_arguments)]
    /// `mrope_axis`/`pos_stride`: see [`Self::rope_qk_partial`]'s doc comment.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_qk_packed_partial(
        &self,
        q_dst: &mut ViewMut<'_, f32>,
        packed: &mut ViewMut<'_, f32>,
        stride: usize,
        q_off: usize,
        k_off: usize,
        positions: &View<'_, i32>,
        freq_factors: &View<'_, f32>,
        mrope_axis: &View<'_, i32>,
        pos_stride: usize,
        n_tokens: usize,
        n_heads: usize,
        n_kv_heads: usize,
        d_head: usize,
        rotary_dim: usize,
        theta_base: f32,
        freq_scale: f32,
        interleaved: bool,
    ) -> Result<()> {
        anyhow::ensure!(
            d_head.is_multiple_of(2),
            "d_head {d_head} must be even for rope"
        );
        anyhow::ensure!(
            rotary_dim.is_multiple_of(2) && rotary_dim >= 2 && rotary_dim <= d_head,
            "rotary_dim {rotary_dim} must be even and in 2..={d_head}"
        );
        anyhow::ensure!(
            freq_factors.len() >= rotary_dim / 2,
            "freq_factors holds {} entries, short of the {} pairs that rotate",
            freq_factors.len(),
            rotary_dim / 2
        );
        anyhow::ensure!(
            pos_stride == 1 || pos_stride == 3,
            "pos_stride {pos_stride} must be 1 (plain position) or 3 (mRoPE T/H/W)"
        );
        anyhow::ensure!(
            mrope_axis.len() >= rotary_dim / 2,
            "mrope_axis holds {} entries, short of the {} frequencies that rotate",
            mrope_axis.len(),
            rotary_dim / 2
        );
        anyhow::ensure!(
            positions.len() >= n_tokens * pos_stride,
            "positions holds {} entries, short of {n_tokens} tokens x pos_stride {pos_stride}",
            positions.len()
        );
        anyhow::ensure!(
            packed.len() >= (n_tokens - 1) * stride + k_off + n_kv_heads * d_head,
            "packed qkv holds {} elements, short for {n_tokens} rows of {stride}",
            packed.len()
        );
        anyhow::ensure!(
            q_dst.len() >= n_tokens * n_heads * d_head,
            "q_dst holds {} elements, short of {n_tokens} x {n_heads} x {d_head}",
            q_dst.len()
        );
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "rope_qk_packed_f32")?;
        // The rotating lanes plus the ones that carry q's untouched tail
        // across. Equal to `d_head / 2` when the whole head rotates, which is
        // the grid this kernel always had.
        let lanes = (rotary_dim / 2 + (d_head - rotary_dim)) as u32;
        let block = lanes.clamp(1, 128);
        let cfg = LaunchConfig {
            grid_dim: (
                lanes.div_ceil(block),
                (n_heads + n_kv_heads) as u32,
                n_tokens as u32,
            ),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (st, qo, ko) = (stride as i32, q_off as i32, k_off as i32);
        let (h, kh, dh, rd, il, ps) = (
            n_heads as i32,
            n_kv_heads as i32,
            d_head as i32,
            rotary_dim as i32,
            i32::from(interleaved),
            pos_stride as i32,
        );
        // Same reason as `rope_qk_partial`: `mrope_axis`/`pos_stride` last so
        // this one `.arg()` sequence matches both backends' parameter order.
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(q_dst)
            .arg(packed)
            .arg(&st)
            .arg(&qo)
            .arg(&ko)
            .arg(positions)
            .arg(freq_factors)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&rd)
            .arg(&theta_base)
            .arg(&freq_scale)
            .arg(&il)
            .arg(mrope_axis)
            .arg(&ps);
        self.dev.profile().time("rope_qk", self.dev.stream(), || {
            unsafe { b.launch(cfg) }.context("rope_qk_packed")?;
            Ok(())
        })?;
        Ok(())
    }

    /// [`Self::store_kv2`] reading `k` and `v` out of the same packed row.
    #[allow(clippy::too_many_arguments)]
    pub fn store_kv2_packed(
        &self,
        k_cache: &mut ViewMut<'_, f16>,
        v_cache: &mut ViewMut<'_, f16>,
        packed: &View<'_, f32>,
        stride: usize,
        k_off: usize,
        v_off: usize,
        slots: &View<'_, i32>,
        n_kv_heads: usize,
        d_head: usize,
        n_slots: usize,
        n_tokens: usize,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "store_kv2_packed_f16")?;
        let block = (d_head as u32).clamp(1, 256);
        let cfg = LaunchConfig {
            grid_dim: (
                (d_head as u32).div_ceil(block),
                2 * n_kv_heads as u32,
                n_tokens as u32,
            ),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (st, ko, vo) = (stride as i32, k_off as i32, v_off as i32);
        let (kh, dh, ms, nt) = (
            n_kv_heads as i32,
            d_head as i32,
            n_slots as i32,
            n_tokens as i32,
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(k_cache)
            .arg(v_cache)
            .arg(packed)
            .arg(&st)
            .arg(&ko)
            .arg(&vo)
            .arg(slots)
            .arg(&kh)
            .arg(&dh)
            .arg(&ms)
            .arg(&nt);
        self.dev.profile().time("store_kv", self.dev.stream(), || {
            unsafe { b.launch(cfg) }.context("store_kv2_packed")?;
            Ok(())
        })?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        x: &mut ViewMut<'_, f32>,
        positions: &View<'_, i32>,
        freq_factors: &View<'_, f32>,
        n_tokens: usize,
        n_heads: usize,
        d_head: usize,
        theta_base: f32,
        freq_scale: f32,
        interleaved: bool,
    ) -> Result<()> {
        anyhow::ensure!(
            d_head.is_multiple_of(2),
            "d_head {d_head} must be even for rope"
        );
        let name = if interleaved {
            "rope_norm_f32"
        } else {
            "rope_neox_f32"
        };
        let f = self.dev.kernels().get("infero_ops", ops_src(), name)?;
        let half = (d_head / 2) as u32;
        let block = half.clamp(1, 128);
        let cfg = LaunchConfig {
            grid_dim: (half.div_ceil(block), n_heads as u32, n_tokens as u32),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (h, dh) = (n_heads as i32, d_head as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(x)
            .arg(positions)
            .arg(freq_factors)
            .arg(&h)
            .arg(&dh)
            .arg(&theta_base)
            .arg(&freq_scale);
        self.dev
            .profile()
            .time("rope_neox", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("rope_neox")?;
                Ok(())
            })?;
        Ok(())
    }

    // ---- attention ------------------------------------------------------

    /// Scatter `src[n_tokens, n_kv_heads, d_head]` into a
    /// `[n_kv_heads, n_slots, d_head]` f16 pool at the given physical slots.
    #[allow(clippy::too_many_arguments)]
    /// Append this step's keys *and* values to the pool in one launch.
    ///
    /// See [`Kernels::store_kv`]; `blockIdx.y` covers both halves, which halves
    /// the launches a decode step spends here.
    #[allow(clippy::too_many_arguments)]
    pub fn store_kv2(
        &self,
        k_cache: &mut ViewMut<'_, f16>,
        v_cache: &mut ViewMut<'_, f16>,
        k_src: &View<'_, f32>,
        v_src: &View<'_, f32>,
        slots: &View<'_, i32>,
        n_kv_heads: usize,
        d_head: usize,
        n_slots: usize,
        n_tokens: usize,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "store_kv2_f16")?;
        let block = (d_head as u32).clamp(1, 256);
        let cfg = LaunchConfig {
            grid_dim: (
                (d_head as u32).div_ceil(block),
                2 * n_kv_heads as u32,
                n_tokens as u32,
            ),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (kh, dh, ms, nt) = (
            n_kv_heads as i32,
            d_head as i32,
            n_slots as i32,
            n_tokens as i32,
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(k_cache)
            .arg(v_cache)
            .arg(k_src)
            .arg(v_src)
            .arg(slots)
            .arg(&kh)
            .arg(&dh)
            .arg(&ms)
            .arg(&nt);
        self.dev.profile().time("store_kv", self.dev.stream(), || {
            unsafe { b.launch(cfg) }.context("store_kv2")?;
            Ok(())
        })?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn store_kv(
        &self,
        cache: &mut ViewMut<'_, f16>,
        src: &View<'_, f32>,
        slots: &View<'_, i32>,
        n_kv_heads: usize,
        d_head: usize,
        n_slots: usize,
        n_tokens: usize,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "store_kv_f16")?;
        let block = (d_head as u32).clamp(1, 256);
        let cfg = LaunchConfig {
            grid_dim: (
                (d_head as u32).div_ceil(block),
                n_kv_heads as u32,
                n_tokens as u32,
            ),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (kh, dh, ms, nt) = (
            n_kv_heads as i32,
            d_head as i32,
            n_slots as i32,
            n_tokens as i32,
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(cache)
            .arg(src)
            .arg(slots)
            .arg(&kh)
            .arg(&dh)
            .arg(&ms)
            .arg(&nt);
        self.dev.profile().time("store_kv", self.dev.stream(), || {
            unsafe { b.launch(cfg) }.context("store_kv")?;
            Ok(())
        })?;
        Ok(())
    }

    /// Record where each of the batch's tokens landed in the pool.
    #[allow(clippy::too_many_arguments)]
    pub fn write_slot_table(
        &self,
        table: &mut ViewMut<'_, i32>,
        seq_of: &View<'_, i32>,
        positions: &View<'_, i32>,
        slots: &View<'_, i32>,
        table_stride: usize,
        n_tokens: usize,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "write_slot_table")?;
        let (stride, n) = (table_stride as i32, n_tokens as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(table)
            .arg(seq_of)
            .arg(positions)
            .arg(slots)
            .arg(&stride)
            .arg(&n);
        self.dev
            .profile()
            .time("write_slot_table", self.dev.stream(), || {
                unsafe { b.launch(elementwise(n_tokens as u32)) }.context("write_slot_table")?;
                Ok(())
            })?;
        Ok(())
    }

    /// `scores[h, t, j] = q[t, h] · k_cache[h/gqa, j] * scale`, causally masked
    /// against `positions[t]`.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_scores(
        &self,
        scores: &mut ViewMut<'_, f32>,
        q: &View<'_, f32>,
        k_cache: &View<'_, f16>,
        batch: BatchLayout<'_>,
        dims: AttnDims,
        kv_len: usize,
        scale: f32,
    ) -> Result<()> {
        // One K fetch for every query head that shares it, when there is more
        // than one. `d_head` above 128 would not fit the four registers the
        // grouped kernel holds it in.
        let group = dims.n_heads / dims.n_kv_heads.max(1);
        // The GQA-shaped score kernels -- one warp serving a whole query group
        // instead of one key -- are CUDA-only for now. Saying false takes the
        // plain per-key kernel, which is ported and correct.
        let gqa = cfg!(feature = "cuda")
            && group > 1
            && dims.d_head <= 4 * 32
            && dims.d_head.is_multiple_of(32);
        let f = self.dev.kernels().get(
            "infero_ops",
            ops_src(),
            // Strided, and measured against the contiguous alternative rather
            // than assumed. `attn_scores_gqa_v4_f32` gives each lane a
            // contiguous run — the change that took `attn_output` from 86.2 us
            // to 36.0 — and here it *loses*: 45.1 us against 43.6 at a batch of
            // 32, and 46.9 before the query load was hoisted out of the guard.
            //
            // The two kernels are not in the same situation. `attn_output`
            // spends each V element once, so its load width sets its
            // instruction count. This one reads a K row once into registers and
            // spends it on all four query heads in the group, so K's width
            // barely registers, and the query access it would replace is
            // already perfectly coalesced — a warp's strided read is 128
            // consecutive floats. There is nothing to win and a branch to lose.
            //
            // `INFERO_ATTN_V1` still selects the older `attn_output`, which is
            // the one where the width mattered.
            // Two keys a warp, which doubles what the score loop has in
            // flight: 43.4 us a layer down to 33.9 at a batch of 32, or 762
            // GB/s of K up to 990. `INFERO_ATTN_X2=0` restores one key a warp.
            match (gqa, !std::env::var("INFERO_ATTN_X2").is_ok_and(|v| v == "0")) {
                (true, true) => "attn_scores_gqa_v4_f32",
                (true, false) => "attn_scores_gqa_f32",
                (false, _) => "attn_scores_f32",
            },
        )?;
        let cfg = LaunchConfig {
            grid_dim: (
                {
                    // The `x2` variant gives a warp two keys, so it needs half
                    // the blocks to cover the range.
                    let per = if gqa
                        && !std::env::var("INFERO_ATTN_X2").is_ok_and(|v| v == "0")
                    {
                        SCORE_WARPS * 2
                    } else {
                        SCORE_WARPS
                    };
                    (kv_len as u32).div_ceil(per).max(1)
                },
                if gqa { dims.n_kv_heads as u32 } else { dims.n_heads as u32 },
                dims.n_tokens as u32,
            ),
            block_dim: (SCORE_WARPS * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        let group_i = group as i32;
        let (stride, h, kh, dh, ms, kl) = (
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            kv_len as i32,
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(scores)
            .arg(q)
            .arg(k_cache)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ms)
            .arg(&kl)
            .arg(&scale);
        if gqa {
            b.arg(&group_i);
        }
        self.dev
            .profile()
            .time("attn_scores", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_scores")?;
                Ok(())
            })?;
        Ok(())
    }

    /// In-place softmax over the kv axis of `scores[n_heads, n_tokens, kv_len]`.
    pub fn attn_softmax(
        &self,
        scores: &mut ViewMut<'_, f32>,
        n_heads: usize,
        n_tokens: usize,
        kv_len: usize,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_softmax_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (n_heads as u32, n_tokens as u32, 1),
            block_dim: (REDUCE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let kl = kv_len as i32;
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(scores).arg(&kl);
        self.dev
            .profile()
            .time("attn_softmax", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_softmax")?;
                Ok(())
            })?;
        Ok(())
    }

    /// `out[t, h, :] = sum_j scores[h, t, j] * v_cache[h/gqa, j, :]`
    /// How many KV chunks the split-K attention output should use, and 0 when
    /// the plain kernel is the better choice.
    ///
    /// The plain kernel makes `n_heads * n_tokens` blocks. That is plenty once
    /// a batch is wide, and far too few for a single sequence — 32 blocks on a
    /// 48-SM device leaves two thirds of it idle. Chunking the KV range buys
    /// blocks; it also costs a second pass over the partials, so it is only
    /// worth it when the grid is actually short.
    fn attn_chunks(&self, dims: &AttnDims, kv_len: usize) -> (u32, u32) {
        // One chunk on a backend without the split kernels. They are a grid
        // recovery for shapes where a layer's V cache has left L2 -- an
        // optimisation, and `attn_output_f32` covers every shape without them,
        // walking the whole key range in one threadgroup instead of several
        // walking chunks that a third launch then reduces.
        if !cfg!(feature = "cuda") {
            return (1, 0);
        }
        // Counted ungrouped on purpose. The grouped value kernel gives a block
        // to each (KV head, token) and reads each V row once for the whole
        // query group — a quarter of the traffic at Llama-3.1's 32-over-8 —
        // and pairing it with the split to buy the grid back was measured:
        // 387 tok/s against 406 at a batch of eight.
        //
        // The reason it loses is that the traffic it saves is not DRAM. A
        // layer's V cache at 256 tokens of context is 590 KB and sits in L2,
        // so the four reads are L2 reads, while the partial sums the split
        // writes and the reduce pass that consumes them are new traffic and a
        // third launch. `attn_output_gqa_split_f32` stays for the shapes where
        // the context is long enough for V to leave L2; this counter decides
        // when it runs, and at these sizes it should not.
        let blocks = (dims.n_heads * dims.n_tokens) as u32;
        let want = self.dev.sm_count() * 4;
        // The gate counts blocks and nothing else, and a block is not the only
        // thing that can be short of work: `attn_output_f32` walks the whole
        // key range in one loop, so its latency is `kv_len` dependent loads
        // however many blocks there are. `INFERO_ATTN_SPLIT=1` chunks anyway.
        static ALWAYS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let always = *ALWAYS.get_or_init(|| std::env::var_os("INFERO_ATTN_SPLIT").is_some());
        if (blocks >= want && !always) || kv_len <= 128 {
            return (0, 0);
        }
        let chunks = if always && blocks >= want {
            // Enough blocks already; the split is for the chain, not the grid.
            4
        } else {
            want.div_ceil(blocks).clamp(2, 32)
        };
        let chunk = (kv_len as u32).div_ceil(chunks).next_multiple_of(32);
        let chunks = (kv_len as u32).div_ceil(chunk.max(1));
        if chunks < 2 { (0, 0) } else { (chunks, chunk) }
    }

    /// Floats of scratch the split attention paths need for their partials.
    ///
    /// The first `32 * n_heads * d_head * n_tokens` hold the per-chunk weighted
    /// sums, which is all `attn_output` uses; the tail is the flash path's
    /// per-chunk `{max, denominator}` pair.
    pub fn attn_partial_floats(n_heads: usize, d_head: usize, n_tokens: usize) -> usize {
        32 * n_heads * n_tokens * (d_head + 2)
    }

    /// Floats of scratch `attn_prefill_split` needs for `(m, l)`: up to 32
    /// chunks' worth of per-chunk partials (same 32-chunk cap
    /// `attn_partial_floats` and `Self::prefill_chunks` already use) plus one
    /// slot for the merged result `attn_ms_reduce_f32` writes, run-relative
    /// (not absolute-token) indexed, matching `attn_partial_floats`' own
    /// convention.
    pub fn attn_ms_floats(n_heads: usize, run_tokens: usize) -> usize {
        33 * n_heads * run_tokens * 2
    }

    /// How the fused attention kernel would split this shape, and `None` when
    /// it declines the work.
    ///
    /// It keeps the score row in shared memory, so a chunk has to fit there;
    /// 2048 keys is 8 KB, comfortably inside the 48 KB every block gets. It is
    /// also worth using only while the grid is short — once a batch is wide
    /// enough to fill the device on its own, the separate score kernel's
    /// grouped-query reuse reads each key once for four heads instead of once
    /// per head, and that is the bigger effect.
    /// Heads the fused attention grid will span: `n_kv_heads` when the grouped
    /// kernel is the one that will run, `n_heads` otherwise.
    ///
    /// This repeats the `gqa` predicate in [`Self::attn_flash`] rather than
    /// sharing it because the split has to be chosen before the launch is
    /// built. The two must agree; a mismatch shows up as a kernel starved of
    /// blocks, which is what it was.
    fn flash_grid_heads(dims: &AttnDims) -> usize {
        let group = dims.n_heads / dims.n_kv_heads.max(1);
        let lanes = (dims.d_head as u32).next_multiple_of(32).clamp(32, 1024);
        let subs = std::env::var("INFERO_ATTN_SUBS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(4)
            .clamp(1, 1024 / lanes.max(1));
        let grouped = std::env::var("INFERO_ATTN_GQA").is_ok_and(|v| v != "0")
            && group > 1
            && group <= 8
            && lanes * subs >= group as u32 * 32
            && dims.d_head.is_multiple_of(8)
            && (lanes * subs).is_multiple_of((dims.d_head / 8) as u32);
        if grouped { dims.n_kv_heads } else { dims.n_heads }
    }

    fn attn_flash_split(&self, dims: &AttnDims, kv_len: usize) -> Option<(u32, u32)> {
        const MAX_CHUNK: u32 = 2048;
        if dims.d_head > 1024 || kv_len == 0 {
            return None;
        }
        // Blocks the launch will actually make, which is not `n_heads *
        // n_tokens` when the grouped kernel runs: that one gives a block to
        // each (KV head, token). Deciding the split from the ungrouped count
        // told it there were 1024 blocks where there were 256, so it split not
        // at all and the grouped kernel ran at 1.4 blocks per SM.
        let blocks = (Self::flash_grid_heads(dims) * dims.n_tokens) as u32;
        let want = self.dev.sm_count() * 4;
        if blocks >= want {
            // Enough blocks already, so the *split* buys nothing. The fused
            // kernel does two other things — keeps the score matrix out of HBM
            // and costs two launches a layer instead of three — and the guess
            // was that those would pay at any batch. `INFERO_FLASH_WIDE=1`
            // measures it and they do not: on an A4000, 379 tok/s against 405
            // at a batch of eight and 730 against 852 at thirty-two; on a
            // Blackwell RTX PRO 6000, 1057.8 against 1091.9 and 1990.5 against
            // 3500.0. One block per (head, token) with the scores in global
            // beats one fused block per (head, token) with them in shared,
            // because the fused block holds a chunk of scores in shared memory
            // and that is what caps its occupancy.
            //
            // Re-run on Blackwell after `attn_output_v4_f32` cut the unfused
            // path's weighted sum from 85.6 us to 35.5, on the theory that the
            // old comparison was against a crippled baseline. It was not the
            // baseline: the gap widened.
            //
            // That is a verdict on *this* fused kernel, not on fusing. vLLM's
            // `flash_attn_varlen_func` was timed at the same shape — batch 32,
            // 512 of history, 32 query heads over 8 KV heads of 128 — and takes
            // 58.1 us against these three kernels' 85.8, which is 1156 GB/s of
            // KV against 782. So a fused decode attention is worth 0.89 ms a
            // step here; it is the largest single piece of the remaining gap.
            // See `scripts/flash_attn_bandwidth.py`.
            //
            // Two things are wrong with `attn_flash_f32`, and only one of them
            // has been fixed. Its K and V loads were two bytes a thread, the
            // same defect `attn_output_f32` had; widening them took the fused
            // path from 1990 tok/s to 3180, a 60% gain that confirms the
            // diagnosis. What remains is that it is not grouped-query aware:
            // one block per *query* head means 32 blocks read the 8 KV heads,
            // so K and V each cross the bus four times. The unfused path only
            // pays that on V, because `attn_scores_gqa_f32` holds a K row for
            // the whole group — which is exactly why it still wins, 85.8 us
            // against the fused 108.2 + 9.2. vLLM pays it on neither, and that
            // is the whole of the 620 GB/s against 1156.
            //
            // The grouped kernel was then given the same treatment — eight
            // halves a thread on both K and V, and the key range split across
            // the block — and it is *worse*: 184.4 us a layer, against the
            // ungrouped 108.2 and the unfused 85.8. Grouping divides the grid
            // by the group as well as the traffic, and one block per (KV head,
            // token) is 256 blocks on 188 SMs. Four times less KV read does not
            // pay for one and a half blocks an SM.
            //
            // `attn_flash_split` was then taught which kernel it is splitting
            // for — see `flash_grid_heads` — since it had been reading 1024
            // blocks where the grouped launch makes 256, and so returning a
            // single chunk. That is worth 10% (2511 tok/s to 2756) and still
            // leaves the fused path 21% behind the three kernels.
            //
            // Five measured shapes of this kernel, at a batch of 32:
            //
            // | fused, ungrouped, 2-byte loads      | 1990 tok/s |
            // | fused, ungrouped, wide loads        | 3180 |
            // | fused, grouped, split blind         | 2511 |
            // | fused, grouped, split aware         | 2756 |
            // | fused, grouped, K held across group | 2777 |
            // | fused, grouped, 256-thread block    | 2676 |
            // | fused, grouped, 512-thread block    | 2380 |
            // | unfused three kernels               | 3489 |
            //
            // The last three are the sharpest. Giving a warp to each *key* and
            // holding that row across the group — exactly what
            // `attn_scores_gqa_f32` does, and why the unfused path wins on K —
            // bought 21 tok/s. Then, on the theory that the score phase was
            // starved because the block was pinned to `group * 32` threads
            // while the unfused kernel spreads that phase over the whole grid,
            // the block was unpinned: wider is *worse*, monotonically. More
            // warps per block cost more resident blocks per SM than the extra
            // parallelism is worth, and the softmax phase leaves all but
            // `group` of them idle anyway.
            //
            // Every one of those changes moved the number the way its diagnosis
            // said it would, and none came close, so what is left is not
            // another defect in this kernel. The decomposition itself is the
            // problem: a block pinned to a (head, token, chunk) with its scores
            // in shared memory. FlashAttention's decode path schedules query
            // heads, KV heads and key chunks against the machine as one
            // problem, and matching it means writing that rather than repairing
            // this. Anyone starting should read `vllm_flash_attn` rather than
            // this file.
            //
            // The gate stays; the switch stays so the result is re-runnable.
            static WIDE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let on = *WIDE
                .get_or_init(|| std::env::var_os("INFERO_FLASH_WIDE").is_some());
            return if on {
                Some((1, (kv_len as u32).next_multiple_of(32).min(MAX_CHUNK)))
            } else {
                None
            };
        }
        let want = std::env::var("INFERO_ATTN_WANT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(want);
        let chunks = want.div_ceil(blocks).clamp(1, 32);
        // Round the chunk *down* to a warp: rounding up is what collapsed a
        // six-way split into a four-way one, and the split is the only source
        // of parallelism this kernel has at a batch of one.
        let chunk = ((kv_len as u32) / chunks / 32 * 32).clamp(32, MAX_CHUNK);
        let chunks = (kv_len as u32).div_ceil(chunk).min(32);
        // A chunk has to cover the range once the count is capped.
        let chunk = (kv_len as u32).div_ceil(chunks).next_multiple_of(32);
        Some((chunks, chunk))
    }

    /// Whether [`Self::attn_flash`] would take this shape.
    pub fn flash_attention(&self, dims: &AttnDims, kv_len: usize) -> bool {
        // Read once: this is asked per layer per step, and `std::env::var`
        // takes the environment lock and allocates every time.
        // The split flash path (`attn_flash_f32` plus its reduce) is CUDA-only
        // for now. False takes the unsplit scores/softmax/output sequence,
        // which is ported: one threadgroup walks a whole key range instead of
        // several walking chunks, so a long context loses parallelism rather
        // than correctness.
        if !cfg!(feature = "cuda") {
            return false;
        }
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| !std::env::var("INFERO_NO_FLASH_ATTN").is_ok_and(|v| v != "0"))
            && self.attn_flash_split(dims, kv_len).is_some()
    }

    /// The KV cache read at attention's grid and access pattern, and nothing
    /// else. See the kernel's comment: this is the ceiling both attention
    /// paths are measured against.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_kv_probe(
        &self,
        sink: &mut ViewMut<'_, f32>,
        k_cache: &View<'_, f16>,
        v_cache: &View<'_, f16>,
        batch: BatchLayout<'_>,
        dims: AttnDims,
        kv_len: usize,
    ) -> Result<()> {
        let (n_chunks, chunk) = self.decode_chunks(&dims, kv_len);
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_kv_probe_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (dims.n_kv_heads as u32, dims.n_tokens as u32, n_chunks),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let (stride, kh, dh, ms, kl, ck) = (
            batch.table_stride as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            kv_len as i32,
            chunk as i32,
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(sink)
            .arg(k_cache)
            .arg(v_cache)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(&kh)
            .arg(&dh)
            .arg(&ms)
            .arg(&kl)
            .arg(&ck);
        self.dev
            .profile()
            .time("attn_kv_probe", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_kv_probe")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Whether [`Self::attn_decode`] takes this shape.
    ///
    /// On CUDA: a real query group, a `d_head` its lanes divide evenly, and
    /// a group narrow enough that `group * 32` is a legal block.
    ///
    /// On Metal: `attn_decode_fused_f32` -- scores, softmax and the V-weighted
    /// sum in one dispatch instead of `attn_decode_gqa_f32`'s heavily tuned
    /// CUDA original (register-pipelined K/V prefetch, chunked occupancy, an
    /// MMA variant -- see its own "worth porting, and not worth blocking on"
    /// comment, which this is not that port of). It stages the whole score
    /// row in threadgroup memory rather than chunking, so `kv_len` is capped
    /// at what fits: `ATTN_DECODE_FUSED_MAX_KV` in ops.metal. Measured
    /// (`examples/attn_decode_fused_check.rs`): byte-exact against the
    /// unfused path at every `kv_len` tried, 0.95-2.21x, a clean win at the
    /// short end and roughly parity by 256 -- never a regression in that
    /// range, so this is not gated any tighter than the memory cap itself.
    pub fn decode_attention(&self, dims: &AttnDims, kv_len: usize) -> bool {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *OFF.get_or_init(|| std::env::var("INFERO_NO_DECODE_ATTN").is_ok_and(|v| v != "0")) {
            return false;
        }
        if !cfg!(feature = "cuda") {
            const ATTN_DECODE_FUSED_MAX_KV: usize = 8192;
            return dims.n_heads > 0
                && dims.n_kv_heads > 0
                && dims.n_heads.is_multiple_of(dims.n_kv_heads)
                && dims.d_head > 0
                && kv_len <= ATTN_DECODE_FUSED_MAX_KV;
        }
        let group = dims.n_heads / dims.n_kv_heads.max(1);
        group > 1
            && group <= 16
            && dims.n_heads == group * dims.n_kv_heads
            && dims.d_head.is_multiple_of(32)
            && dims.d_head / 32 <= 8
            && dims.d_head.is_multiple_of(8)
    }

    /// Whether [`Self::tq_attn_decode`] would take this shape, independent of
    /// `kv_len` -- mirrors the group-ratio check `tq_attn_decode` asserts
    /// internally, so a caller can decide once, at load time, whether a
    /// TurboQuant-quantized model will ever fall back to the score-
    /// materializing three-kernel path (`tq_attn_scores`/`attn_softmax`/
    /// `tq_attn_output`).
    pub fn tq_decode_attention(&self, dims: &AttnDims) -> bool {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *OFF.get_or_init(|| std::env::var("INFERO_TQ_DECODE_ATTN").is_ok_and(|v| v == "0")) {
            return false;
        }
        let group = dims.n_heads / dims.n_kv_heads.max(1);
        group >= 1 && group <= 8 && dims.n_heads == group * dims.n_kv_heads
    }

    /// How the key range is cut up for [`Self::attn_decode`].
    ///
    /// The kernel's grid is one block per (KV head, token, chunk), and the
    /// first two are 256 blocks at Llama-3.1's shape and a batch of 32 — 1.4
    /// per SM on a 188-SM card. The chunk count is what buys the grid back;
    /// it costs a partial buffer and a combine pass, which is why it is the
    /// smallest count that fills the device rather than the largest.
    fn decode_chunks(&self, dims: &AttnDims, kv_len: usize) -> (u32, u32) {
        let blocks = (dims.n_kv_heads * dims.n_tokens).max(1) as u32;
        let want = std::env::var("INFERO_DECODE_WANT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(self.dev.sm_count() * 4);
        let chunks = want.div_ceil(blocks).clamp(1, 16);
        // Whole tiles: a chunk shorter than a tile wastes the block it runs in.
        let chunk = ((kv_len as u32).div_ceil(chunks)).next_multiple_of(32);
        let chunks = (kv_len as u32).div_ceil(chunk.max(32)).max(1);
        (chunks, chunk.max(32))
    }

    /// Scores, softmax and the weighted sum in one pass, with the key range
    /// tiled through shared memory and the score row kept inside a warp.
    ///
    /// See the comment on `attn_decode_gqa_f32`. Replaces `attn_scores` +
    /// `attn_softmax` + `attn_output` for grouped-query shapes; `partial` is
    /// [`Self::attn_partial_floats`] long and the combine pass is the fused
    /// path's.
    #[allow(clippy::too_many_arguments)]
    /// `hout`, when given, receives the f16 copy of the output that the output
    /// projection is about to read — written by the combine rather than by a
    /// separate `to_f16` over the f32. Returns whether it was written: the
    /// single-chunk path stores straight from the attention kernel and has no
    /// combine to fold it into, so the caller has to be told which happened
    /// rather than assume.
    pub fn attn_decode(
        &self,
        out: &mut ViewMut<'_, f32>,
        hout: Option<&mut ViewMut<'_, f16>>,
        q: &View<'_, f32>,
        k_cache: &View<'_, f16>,
        v_cache: &View<'_, f16>,
        batch: BatchLayout<'_>,
        dims: AttnDims,
        kv_len: usize,
        scale: f32,
        partial: &mut ViewMut<'_, f32>,
    ) -> Result<bool> {
        anyhow::ensure!(
            self.decode_attention(&dims, kv_len),
            "attn_decode: unsupported shape"
        );

        if !cfg!(feature = "cuda") {
            let f = self
                .dev
                .kernels()
                .get("infero_ops", ops_src(), "attn_decode_fused_f32")?;
            let sg_for_scores = (kv_len as u32).div_ceil(2).max(1);
            let block = (sg_for_scores * 32)
                .max((dims.d_head as u32).next_multiple_of(32))
                .min(1024);
            let cfg = LaunchConfig {
                grid_dim: (dims.n_heads as u32, dims.n_tokens as u32, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: (kv_len as u32) * 4,
            };
            let (stride, h, kh, dh, ns, kl) = (
                batch.table_stride as i32,
                dims.n_heads as i32,
                dims.n_kv_heads as i32,
                dims.d_head as i32,
                dims.n_slots as i32,
                kv_len as i32,
            );
            let mut b = self.dev.stream().launch_builder(&f);
            b.arg(&mut *out)
                .arg(q)
                .arg(k_cache)
                .arg(v_cache)
                .arg(batch.seq_of)
                .arg(batch.positions)
                .arg(batch.slot_table)
                .arg(&stride)
                .arg(&h)
                .arg(&kh)
                .arg(&dh)
                .arg(&ns)
                .arg(&kl)
                .arg(&scale);
            self.dev
                .profile()
                .time("attn_decode_fused", self.dev.stream(), || {
                    unsafe { b.launch(cfg) }.context("attn_decode_fused")?;
                    Ok(())
                })?;
            // `hout` (the f16 output copy) is CUDA-only here: decode is
            // always one token, and `wo_f16`'s own condition at the call
            // site requires `n > 1`, so this path never actually needs to
            // write one -- there is nothing to test by adding it.
            return Ok(false);
        }

        let group = dims.n_heads / dims.n_kv_heads;
        let (n_chunks, chunk) = self.decode_chunks(&dims, kv_len);
        let ms_off = (32 * dims.n_heads * dims.n_tokens * dims.d_head) as i32;
        anyhow::ensure!(
            n_chunks <= 32,
            "attn_decode: {n_chunks} chunks past the partial buffer's 32"
        );

        // `INFERO_ATTN_MMA=1` runs the tensor-core decomposition instead; see
        // `attn_decode_mma_f32`. Opt-in rather than auto-selected: the softmax
        // weights go through f16 on their way into the value product where the
        // scalar kernel keeps them in f32, which is ~6e-4 relative on an output
        // element -- small, but the chunk count depends on batch width, so a
        // batched and a solo decode can round a greedy token differently. See
        // the longer comment on `attn_decode_mma_f32` in `ops.cu`.
        //
        // `d_head` up to 256 (Qwen3.8-27B-FP8's shape) was blocked here for a
        // while after a real bug: a partial key tile (`n < ATTN_MMA_TILE`, true
        // on the last tile whenever `kv_len` is not a multiple of 64, which is
        // most of the time) left `svt`'s padding rows holding whatever an
        // earlier, unrelated kernel's shared memory had in it. The softmax
        // weight for those keys is correctly masked to zero, but IEEE 754 does
        // not let a zero weight save a `mma.sync` accumulator from a NaN or Inf
        // operand (`0 * NaN = NaN`), so the corruption was in the *values*, not
        // an address -- invisible to both `compute-sanitizer memcheck`
        // (nothing here reads or writes out of bounds) and `racecheck` (no
        // hazard, just uninitialized shared memory used as if it were zero).
        // It always surfaced several kernels and sometimes several layers
        // later, once the NaN reached something that used it as an index.
        // Fixed in `ops.cu` by zeroing `svt`'s tail explicitly whenever a tile
        // is partial.
        const ATTN_MMA_MAX_D_HEAD: usize = 256;
        static MMA_ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let mma_env =
            *MMA_ENV.get_or_init(|| std::env::var("INFERO_ATTN_MMA").as_deref() == Ok("1"));
        let mma = mma_env
            && dims.d_head.is_multiple_of(16)
            && dims.d_head <= ATTN_MMA_MAX_D_HEAD
            && group <= 8;
        let f = self.dev.kernels().get(
            "infero_ops",
            ops_src(),
            if mma { "attn_decode_mma_f32" } else { "attn_decode_gqa_f32" },
        )?;
        // Query rows, and one 16-key tile each of K and V — or, for the MMA
        // shape, sixteen f16 query rows, a 64-key tile, and the transposed V.
        let shared = if mma {
            // Must match `ATTN_MMA_TILE` in `ops.cu`: sixteen f16 query rows, a
            // key tile of K, and V transposed into a `d_head x (tile + pad)`
            // block. At a 64-key tile that is 37.8 KB against the default
            // kernel's 10.5, which is where its blocks an SM go.
            // 64 measured against 32 on a Blackwell, in `bwidth_attn.rs` at
            // batch 32 and 512 of history: the small tile is 66.1 us a layer
            // against the default kernel's 57.4, and the large one loses by the
            // same 15% in the served engine. Halving the shared memory does not
            // rescue this path — at decode only `group` of the sixteen M rows
            // are live, so the tensor cores spend three quarters of their work
            // on padding, and the V transpose is on top of that.
            const T: usize = 64;
            (16 * (dims.d_head + 8) * 2
                + T * (dims.d_head + 8) * 2
                + dims.d_head * (T + 2) * 2) as u32
        } else {
            // Query rows as f32, one 16-key tile each of K and V, and room for
            // the half copy of Q that `ATTN_DECODE_H2` uses — a kilobyte, always
            // reserved so the host does not have to know which way the kernel
            // was compiled.
            (group * dims.d_head * 4
                + 2 * 16 * (dims.d_head + 8) * 2
                + group * dims.d_head * 2) as u32
        };
        // Past the 48 KiB static default -- true for the MMA path once
        // `d_head` reaches 256 (74 KiB here) -- a launch needs the kernel
        // opted in explicitly or the driver refuses it.
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared)?;
        }
        let single = i32::from(n_chunks == 1);
        let cfg = LaunchConfig {
            grid_dim: (dims.n_kv_heads as u32, dims.n_tokens as u32, n_chunks),
            block_dim: (if mma { 128 } else { group as u32 * 32 }, 1, 1),
            shared_mem_bytes: shared,
        };
        let (stride, h, kh, dh, ms, ck, kl, gi) = (
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            chunk as i32,
            kv_len as i32,
            group as i32,
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(&mut *partial)
            .arg(&ms_off)
            .arg(&mut *out)
            .arg(&single)
            .arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ms)
            .arg(&scale)
            .arg(&ck)
            .arg(&kl)
            .arg(&gi);
        self.dev
            .profile()
            .time("attn_decode", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_decode")?;
                Ok(())
            })?;
        if single == 1 {
            return Ok(false);
        }

        let total = (dims.n_tokens * dims.n_heads * dims.d_head) as u32;
        let (nt, nc) = (dims.n_tokens as i32, n_chunks as i32);
        let part = partial.as_view();
        let name = if hout.is_some() {
            "attn_flash_reduce_f16_f32"
        } else {
            "attn_flash_reduce_f32"
        };
        let r = self.dev.kernels().get("infero_ops", ops_src(), name)?;
        let mut rb = self.dev.stream().launch_builder(&r);
        let wrote = match hout {
            Some(h16) => {
                anyhow::ensure!(
                    h16.len() >= total as usize,
                    "attn_decode: f16 scratch holds {} of {total} elements",
                    h16.len()
                );
                rb.arg(out).arg(h16);
                true
            }
            None => {
                rb.arg(out);
                false
            }
        };
        rb.arg(&part)
            .arg(&ms_off)
            .arg(&h)
            .arg(&dh)
            .arg(&nt)
            .arg(&nc);
        self.dev
            .profile()
            .time("attn_decode_reduce", self.dev.stream(), || {
                unsafe { rb.launch(elementwise(total)) }.context("attn_decode_reduce")?;
                Ok(())
            })?;
        Ok(wrote)
    }

    /// Whether [`Self::attn_prefill`] would take this shape.
    ///
    /// The same tensor-core gate as [`Self::attn_decode`]'s MMA branch
    /// (`INFERO_ATTN_MMA=1`, `d_head` a multiple of 16 up to 256, `group <=
    /// 8`), which already guarantees two tokens' query groups fit one
    /// sixteen-row MMA tile (`group * 2 <= 16`) — the minimum this kernel's
    /// query tiling needs to buy anything over one token a block.
    pub fn prefill_attention(&self, dims: &AttnDims) -> bool {
        if !cfg!(feature = "cuda") {
            return false;
        }
        static MMA_ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*MMA_ENV.get_or_init(|| std::env::var("INFERO_ATTN_MMA").as_deref() == Ok("1")) {
            return false;
        }
        let group = dims.n_heads / dims.n_kv_heads.max(1);
        group >= 1
            && group <= 8
            && dims.n_heads == group * dims.n_kv_heads
            && dims.d_head.is_multiple_of(16)
            && dims.d_head <= 256
    }

    /// How the key range is cut up for [`Self::attn_prefill`] — the same
    /// fill-the-device reasoning as [`Self::decode_chunks`], against the
    /// tile grid rather than one block a token.
    fn prefill_chunks(&self, n_kv_heads: usize, n_tiles: usize, kv_len: usize) -> (u32, u32) {
        let blocks = (n_kv_heads * n_tiles).max(1) as u32;
        let want = std::env::var("INFERO_DECODE_WANT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(self.dev.sm_count() * 4);
        let chunks = want.div_ceil(blocks).clamp(1, 16);
        let chunk = ((kv_len as u32).div_ceil(chunks)).next_multiple_of(32);
        let chunks = (kv_len as u32).div_ceil(chunk.max(32)).max(1);
        (chunks, chunk.max(32))
    }

    /// A query-tiled tensor-core attention pass for a contiguous run of one
    /// sequence's tokens — [`Self::attn_decode`]'s MMA path routed a wide
    /// prefill through a kernel built to answer one token a block, which
    /// pays for the whole causal K/V range again for every token in the
    /// run even though each only adds one key to its predecessor's range.
    /// See `attn_prefill_mma_f32` in `ops.cu` for the shape this buys back.
    ///
    /// `run_base`/`run_tokens` describe the slice of `q`/`out` (and of
    /// `batch`) this call covers: every token in `[run_base, run_base +
    /// run_tokens)` must share one `seq_of` and increase `positions` by
    /// exactly one from its predecessor, or a tile will read one sequence's
    /// K/V slots for another's rows. Building that slice — splitting a batch
    /// at sequence boundaries and gaps in `positions` — is the caller's job;
    /// this call trusts it was done. `partial` must hold
    /// [`Self::attn_partial_floats`]`(n_heads, d_head, run_tokens)` floats.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill(
        &self,
        out: &mut ViewMut<'_, f32>,
        q: &View<'_, f32>,
        k_cache: &View<'_, f16>,
        v_cache: &View<'_, f16>,
        batch: BatchLayout<'_>,
        dims: AttnDims,
        run_base: usize,
        run_tokens: usize,
        kv_len: usize,
        scale: f32,
        partial: &mut ViewMut<'_, f32>,
    ) -> Result<()> {
        anyhow::ensure!(self.prefill_attention(&dims), "attn_prefill: unsupported shape");
        anyhow::ensure!(run_tokens >= 1, "attn_prefill: empty run");
        let group = dims.n_heads / dims.n_kv_heads;
        let tpw = (16 / group).max(1);
        // `T` (the K/V tile width) must match `ATTN_MMA_PF_TILE` in `ops.cu` and
        // be a multiple of `ATTN_MMA_WK` (16). Shrunk from `attn_decode`'s 64 to
        // 32 so a block's Q staging (`NWARPS` 16-row slabs) can grow past
        // `attn_decode`'s 4 warps without the two together exceeding this GPU's
        // flat 101376-byte dynamic-shared-memory ceiling -- see the comment on
        // `ATTN_MMA_PF_TILE`. Each extra warp is another independent MMA
        // accumulator living entirely in that warp's own registers (never
        // shared memory), so this is a safe amortization knob, not the
        // per-subgroup-accumulator-in-shared-memory design that turned out to
        // be unfixable (see the attn-prefill-rewrite-deadend memory).
        const T: usize = 32;
        const NWARPS: usize = 7;
        // `ATTN_MMA_KPAD` (shared with attn_decode): row stride in bytes
        // must stay a multiple of 16 for the kernel's `uint4` shared-memory
        // accesses, i.e. `d_head + KPAD` a multiple of 8 halfwords -- 8 is
        // the largest value smaller than attn_decode's that still clears
        // that bar (tried 4: `CUDA_ERROR_MISALIGNED_ADDRESS`, caught by
        // `attn_prefill_matches_the_three_kernels` before it ever reached
        // the server). Do not shrink this without re-deriving the alignment.
        const KPAD: usize = 8;
        const VPAD: usize = 2;
        // NWARPS=8 (VPAD=0 to fit) measured within noise of 7 on the real
        // 30552-token prefill (11.1-11.3s both ways) -- one more warp isn't
        // worth the tighter shared-memory margin and lost bank-conflict
        // padding. 7 is the settled value.
        let tile_tokens = NWARPS * tpw;
        let n_tiles = run_tokens.div_ceil(tile_tokens);
        let (n_chunks, chunk) = self.prefill_chunks(dims.n_kv_heads, n_tiles, kv_len);
        anyhow::ensure!(
            n_chunks <= 32,
            "attn_prefill: {n_chunks} chunks past the partial buffer's 32"
        );
        let ms_off = (32 * dims.n_heads * run_tokens * dims.d_head) as i32;

        let f = self.dev.kernels().get("infero_ops", ops_src(), "attn_prefill_mma_f32")?;
        // `sq` is `NWARPS` sixteen-row slabs, one a warp; `sk`/`svt` are the
        // block-shared K/V tile at width `T`.
        let shared = (NWARPS * 16 * (dims.d_head + KPAD) * 2
            + T * (dims.d_head + KPAD) * 2
            + dims.d_head * (T + VPAD) * 2) as u32;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared)?;
        }
        let single = i32::from(n_chunks == 1);
        let cfg = LaunchConfig {
            grid_dim: (dims.n_kv_heads as u32, n_tiles as u32, n_chunks),
            block_dim: (NWARPS as u32 * 32, 1, 1),
            shared_mem_bytes: shared,
        };
        let (stride, h, kh, dh, ns, ck, kl, gi, tp, rb, rt) = (
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            chunk as i32,
            kv_len as i32,
            group as i32,
            tpw as i32,
            run_base as i32,
            run_tokens as i32,
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(&mut *partial)
            .arg(&ms_off)
            .arg(&mut *out)
            .arg(&single)
            .arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ns)
            .arg(&scale)
            .arg(&ck)
            .arg(&kl)
            .arg(&gi)
            .arg(&tp)
            .arg(&rb)
            .arg(&rt);
        self.dev
            .profile()
            .time("attn_prefill", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_prefill")?;
                Ok(())
            })?;
        if single == 1 {
            return Ok(());
        }

        let total = (run_tokens * dims.n_heads * dims.d_head) as u32;
        let (nt, nc) = (run_tokens as i32, n_chunks as i32);
        let part = partial.as_view();
        let r = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_flash_reduce_f32")?;
        // `attn_flash_reduce_f32` indexes `out` from its own token 0, same as
        // `partial`'s — this kernel's tokens start at `run_base`, so the
        // reduce pass gets a window onto `out` rather than the whole buffer,
        // or it would normalize into someone else's rows.
        let run_elems = run_tokens * dims.n_heads * dims.d_head;
        let out_off = run_base * dims.n_heads * dims.d_head;
        let mut out_run = out.slice_mut(out_off..out_off + run_elems);
        let mut rb2 = self.dev.stream().launch_builder(&r);
        rb2.arg(&mut out_run)
            .arg(&part)
            .arg(&ms_off)
            .arg(&h)
            .arg(&dh)
            .arg(&nt)
            .arg(&nc);
        self.dev
            .profile()
            .time("attn_prefill_reduce", self.dev.stream(), || {
                unsafe { rb2.launch(elementwise(total)) }.context("attn_prefill_reduce")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Same as [`Self::attn_prefill`], but `attn_prefill_mma_pipe_f32`
    /// double-buffers the K half of each key-tile's staging through
    /// `cp.async` one `ATTN_MMA_WK`-wide block ahead of what's being computed
    /// on — see that kernel's doc comment in `ops.cu` for why V stays
    /// synchronous and single-buffered, and why the tile width shrinks from
    /// `T`(32) to `WK`(16) to pay for it. Kept alongside `attn_prefill`
    /// rather than replacing it until this is measured, not just correct —
    /// same reasoning as `gdn_delta_rule_smem_f32` and the chunked GDN kernel
    /// staying in tree next to `reg128`.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_pipe(
        &self,
        out: &mut ViewMut<'_, f32>,
        q: &View<'_, f32>,
        k_cache: &View<'_, f16>,
        v_cache: &View<'_, f16>,
        batch: BatchLayout<'_>,
        dims: AttnDims,
        run_base: usize,
        run_tokens: usize,
        kv_len: usize,
        scale: f32,
        partial: &mut ViewMut<'_, f32>,
    ) -> Result<()> {
        anyhow::ensure!(self.prefill_attention(&dims), "attn_prefill_pipe: unsupported shape");
        anyhow::ensure!(run_tokens >= 1, "attn_prefill_pipe: empty run");
        let group = dims.n_heads / dims.n_kv_heads;
        let tpw = (16 / group).max(1);
        const NWARPS: usize = 7;
        const KPAD: usize = 8;
        const VPAD: usize = 2;
        // The staging/compute granularity — `ATTN_MMA_WK` in `ops.cu`, not
        // `attn_prefill`'s `T` — since this kernel processes one `WK`-wide
        // block per iteration rather than `T`-wide with an inner sub-loop.
        const WK: usize = 16;
        let tile_tokens = NWARPS * tpw;
        let n_tiles = run_tokens.div_ceil(tile_tokens);
        let (n_chunks, chunk) = self.prefill_chunks(dims.n_kv_heads, n_tiles, kv_len);
        anyhow::ensure!(
            n_chunks <= 32,
            "attn_prefill_pipe: {n_chunks} chunks past the partial buffer's 32"
        );
        let ms_off = (32 * dims.n_heads * run_tokens * dims.d_head) as i32;

        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_prefill_mma_pipe_f32")?;
        // `sq` is `NWARPS` sixteen-row slabs; `sk0`+`sk1` are two `WK`-wide
        // K buffers (their combined size equals `attn_prefill`'s single
        // `T`-wide one, `T` == `2 * WK`); `svt` is a single `WK`-wide V
        // buffer, half `attn_prefill`'s `T`-wide one — net, this uses *less*
        // shared memory than `attn_prefill`, not more.
        let shared = (NWARPS * 16 * (dims.d_head + KPAD) * 2
            + 2 * WK * (dims.d_head + KPAD) * 2
            + dims.d_head * (WK + VPAD) * 2) as u32;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared)?;
        }
        let single = i32::from(n_chunks == 1);
        let cfg = LaunchConfig {
            grid_dim: (dims.n_kv_heads as u32, n_tiles as u32, n_chunks),
            block_dim: (NWARPS as u32 * 32, 1, 1),
            shared_mem_bytes: shared,
        };
        let (stride, h, kh, dh, ns, ck, kl, gi, tp, rb, rt) = (
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            chunk as i32,
            kv_len as i32,
            group as i32,
            tpw as i32,
            run_base as i32,
            run_tokens as i32,
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(&mut *partial)
            .arg(&ms_off)
            .arg(&mut *out)
            .arg(&single)
            .arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ns)
            .arg(&scale)
            .arg(&ck)
            .arg(&kl)
            .arg(&gi)
            .arg(&tp)
            .arg(&rb)
            .arg(&rt);
        self.dev
            .profile()
            .time("attn_prefill_pipe", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_prefill_pipe")?;
                Ok(())
            })?;
        if single == 1 {
            return Ok(());
        }

        let total = (run_tokens * dims.n_heads * dims.d_head) as u32;
        let (nt, nc) = (run_tokens as i32, n_chunks as i32);
        let part = partial.as_view();
        let r = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_flash_reduce_f32")?;
        let run_elems = run_tokens * dims.n_heads * dims.d_head;
        let out_off = run_base * dims.n_heads * dims.d_head;
        let mut out_run = out.slice_mut(out_off..out_off + run_elems);
        let mut rb2 = self.dev.stream().launch_builder(&r);
        rb2.arg(&mut out_run)
            .arg(&part)
            .arg(&ms_off)
            .arg(&h)
            .arg(&dh)
            .arg(&nt)
            .arg(&nc);
        self.dev
            .profile()
            .time("attn_prefill_reduce", self.dev.stream(), || {
                unsafe { rb2.launch(elementwise(total)) }.context("attn_prefill_reduce")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Same as [`Self::attn_prefill`], but `attn_prefill_mma_natv_f32` stages
    /// V in its natural `[key][dim]` layout instead of pre-transposing it —
    /// a synchronous copy either way, this isolates whether that layout
    /// change (a prerequisite for pipelining V through `cp.async`, which
    /// cannot transpose) costs more in the PV product's now-unpacked
    /// per-MMA reads than it saves in staging. See the kernel's doc comment
    /// in `ops.cu` for the exact trade.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_natv(
        &self,
        out: &mut ViewMut<'_, f32>,
        q: &View<'_, f32>,
        k_cache: &View<'_, f16>,
        v_cache: &View<'_, f16>,
        batch: BatchLayout<'_>,
        dims: AttnDims,
        run_base: usize,
        run_tokens: usize,
        kv_len: usize,
        scale: f32,
        partial: &mut ViewMut<'_, f32>,
    ) -> Result<()> {
        anyhow::ensure!(self.prefill_attention(&dims), "attn_prefill_natv: unsupported shape");
        anyhow::ensure!(run_tokens >= 1, "attn_prefill_natv: empty run");
        let group = dims.n_heads / dims.n_kv_heads;
        let tpw = (16 / group).max(1);
        const T: usize = 32;
        const NWARPS: usize = 7;
        const KPAD: usize = 8;
        let tile_tokens = NWARPS * tpw;
        let n_tiles = run_tokens.div_ceil(tile_tokens);
        let (n_chunks, chunk) = self.prefill_chunks(dims.n_kv_heads, n_tiles, kv_len);
        anyhow::ensure!(
            n_chunks <= 32,
            "attn_prefill_natv: {n_chunks} chunks past the partial buffer's 32"
        );
        let ms_off = (32 * dims.n_heads * run_tokens * dims.d_head) as i32;

        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_prefill_mma_natv_f32")?;
        // `sq` is `NWARPS` sixteen-row slabs; `sk`/`sv` are both `T`-wide,
        // `krow`-wide K/V tiles now — `sv` no longer needs `d_head`-wide rows
        // the way the transposed `svt` did, so this kernel actually uses
        // *less* shared memory than `attn_prefill`, not more.
        let shared = (NWARPS * 16 * (dims.d_head + KPAD) * 2 + 2 * T * (dims.d_head + KPAD) * 2) as u32;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared)?;
        }
        let single = i32::from(n_chunks == 1);
        let cfg = LaunchConfig {
            grid_dim: (dims.n_kv_heads as u32, n_tiles as u32, n_chunks),
            block_dim: (NWARPS as u32 * 32, 1, 1),
            shared_mem_bytes: shared,
        };
        let (stride, h, kh, dh, ns, ck, kl, gi, tp, rb, rt) = (
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            chunk as i32,
            kv_len as i32,
            group as i32,
            tpw as i32,
            run_base as i32,
            run_tokens as i32,
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(&mut *partial)
            .arg(&ms_off)
            .arg(&mut *out)
            .arg(&single)
            .arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ns)
            .arg(&scale)
            .arg(&ck)
            .arg(&kl)
            .arg(&gi)
            .arg(&tp)
            .arg(&rb)
            .arg(&rt);
        self.dev
            .profile()
            .time("attn_prefill_natv", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_prefill_natv")?;
                Ok(())
            })?;
        if single == 1 {
            return Ok(());
        }

        let total = (run_tokens * dims.n_heads * dims.d_head) as u32;
        let (nt, nc) = (run_tokens as i32, n_chunks as i32);
        let part = partial.as_view();
        let r = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_flash_reduce_f32")?;
        let run_elems = run_tokens * dims.n_heads * dims.d_head;
        let out_off = run_base * dims.n_heads * dims.d_head;
        let mut out_run = out.slice_mut(out_off..out_off + run_elems);
        let mut rb2 = self.dev.stream().launch_builder(&r);
        rb2.arg(&mut out_run)
            .arg(&part)
            .arg(&ms_off)
            .arg(&h)
            .arg(&dh)
            .arg(&nt)
            .arg(&nc);
        self.dev
            .profile()
            .time("attn_prefill_reduce", self.dev.stream(), || {
                unsafe { rb2.launch(elementwise(total)) }.context("attn_prefill_reduce")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Same as [`Self::attn_prefill_natv`], but both K *and* V are
    /// double-buffered through `cp.async` one `ATTN_MMA_WK`-wide block
    /// ahead — `attn_prefill_pipe` could only pipeline K (V's transposed
    /// layout blocked it); `attn_prefill_natv`'s natural V layout is a plain
    /// copy, so `cp.async` can stage it the same way. Tests whether
    /// prefetching *both* operands, not just half, is what the earlier
    /// K-only attempt was missing.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_pipev(
        &self,
        out: &mut ViewMut<'_, f32>,
        q: &View<'_, f32>,
        k_cache: &View<'_, f16>,
        v_cache: &View<'_, f16>,
        batch: BatchLayout<'_>,
        dims: AttnDims,
        run_base: usize,
        run_tokens: usize,
        kv_len: usize,
        scale: f32,
        partial: &mut ViewMut<'_, f32>,
    ) -> Result<()> {
        anyhow::ensure!(self.prefill_attention(&dims), "attn_prefill_pipev: unsupported shape");
        anyhow::ensure!(run_tokens >= 1, "attn_prefill_pipev: empty run");
        let group = dims.n_heads / dims.n_kv_heads;
        let tpw = (16 / group).max(1);
        const NWARPS: usize = 7;
        const KPAD: usize = 8;
        const WK: usize = 16;
        let tile_tokens = NWARPS * tpw;
        let n_tiles = run_tokens.div_ceil(tile_tokens);
        let (n_chunks, chunk) = self.prefill_chunks(dims.n_kv_heads, n_tiles, kv_len);
        anyhow::ensure!(
            n_chunks <= 32,
            "attn_prefill_pipev: {n_chunks} chunks past the partial buffer's 32"
        );
        let ms_off = (32 * dims.n_heads * run_tokens * dims.d_head) as i32;

        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_prefill_mma_pipev_f32")?;
        // `sq` is `NWARPS` sixteen-row slabs; `sk0`/`sk1`/`sv0`/`sv1` are
        // four `WK`-wide, `krow`-wide buffers — comfortably less than
        // `attn_prefill`'s single `T`-wide K tile plus its `d_head`-wide
        // transposed V tile, since `T == 2 * WK` and natural-layout V is
        // `krow`-wide rather than `d_head`-wide.
        let shared = (NWARPS * 16 * (dims.d_head + KPAD) * 2 + 4 * WK * (dims.d_head + KPAD) * 2) as u32;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared)?;
        }
        let single = i32::from(n_chunks == 1);
        let cfg = LaunchConfig {
            grid_dim: (dims.n_kv_heads as u32, n_tiles as u32, n_chunks),
            block_dim: (NWARPS as u32 * 32, 1, 1),
            shared_mem_bytes: shared,
        };
        let (stride, h, kh, dh, ns, ck, kl, gi, tp, rb, rt) = (
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            chunk as i32,
            kv_len as i32,
            group as i32,
            tpw as i32,
            run_base as i32,
            run_tokens as i32,
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(&mut *partial)
            .arg(&ms_off)
            .arg(&mut *out)
            .arg(&single)
            .arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ns)
            .arg(&scale)
            .arg(&ck)
            .arg(&kl)
            .arg(&gi)
            .arg(&tp)
            .arg(&rb)
            .arg(&rt);
        self.dev
            .profile()
            .time("attn_prefill_pipev", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_prefill_pipev")?;
                Ok(())
            })?;
        if single == 1 {
            return Ok(());
        }

        let total = (run_tokens * dims.n_heads * dims.d_head) as u32;
        let (nt, nc) = (run_tokens as i32, n_chunks as i32);
        let part = partial.as_view();
        let r = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_flash_reduce_f32")?;
        let run_elems = run_tokens * dims.n_heads * dims.d_head;
        let out_off = run_base * dims.n_heads * dims.d_head;
        let mut out_run = out.slice_mut(out_off..out_off + run_elems);
        let mut rb2 = self.dev.stream().launch_builder(&r);
        rb2.arg(&mut out_run)
            .arg(&part)
            .arg(&ms_off)
            .arg(&h)
            .arg(&dh)
            .arg(&nt)
            .arg(&nc);
        self.dev
            .profile()
            .time("attn_prefill_reduce", self.dev.stream(), || {
                unsafe { rb2.launch(elementwise(total)) }.context("attn_prefill_reduce")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Same as [`Self::attn_prefill_pipev`], but with real producer/consumer
    /// warp specialization instead of every warp both loading and computing:
    /// one dedicated warp issues `cp.async` for K and V and does nothing
    /// else, signaling readiness to the `NWARPS` compute-only consumer warps
    /// via named barriers (`bar.sync`/`bar.arrive`) instead of a block-wide
    /// `__syncthreads()` — so it can be loading block b+1 while consumers
    /// are still computing on block b, with no rendezvous forcing either
    /// side to wait on the other except for a specific buffer slot's
    /// free/ready signal. See the kernel's doc comment in `ops.cu` for the
    /// barrier protocol (validated in isolation before this kernel was
    /// written) and why this needs one extra warp rather than converting an
    /// existing compute one.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_ws(
        &self,
        out: &mut ViewMut<'_, f32>,
        q: &View<'_, f32>,
        k_cache: &View<'_, f16>,
        v_cache: &View<'_, f16>,
        batch: BatchLayout<'_>,
        dims: AttnDims,
        run_base: usize,
        run_tokens: usize,
        kv_len: usize,
        scale: f32,
        partial: &mut ViewMut<'_, f32>,
    ) -> Result<()> {
        anyhow::ensure!(self.prefill_attention(&dims), "attn_prefill_ws: unsupported shape");
        anyhow::ensure!(run_tokens >= 1, "attn_prefill_ws: empty run");
        let group = dims.n_heads / dims.n_kv_heads;
        let tpw = (16 / group).max(1);
        // consumer/compute warp count -- matches attn_prefill's own.
        //
        // Measured, not assumed: a smaller value shrinks `sq` and the block
        // (fewer threads a block, in principle room for a second resident
        // block an SM against the 245-register/92 KiB-shared occupancy
        // ceiling this kernel otherwise hits -- see the note on
        // `attn_prefill_mma_ws_f32` in `ops.cu`). Swept 7/5/4/3/2 on the
        // real 30552-token benchmark instead of trusting that theory:
        // 2190/2935/3639/4709/6731 ms -- monotonically *worse*, not better.
        // The producer's K/V staging is a per-tile fixed cost that does not
        // shrink with `NWARPS`, and a smaller `NWARPS` means fewer query
        // rows share it (`tile_tokens = NWARPS * tpw`); that cost dominates
        // long before occupancy's theoretical benefit could show up. Do not
        // retry this lever without also shrinking the K/V staging cost per
        // tile, not just the tile.
        const NWARPS: usize = 7;
        const KPAD: usize = 8;
        const WK: usize = 16;
        let tile_tokens = NWARPS * tpw;
        let n_tiles = run_tokens.div_ceil(tile_tokens);
        let (n_chunks, chunk) = self.prefill_chunks(dims.n_kv_heads, n_tiles, kv_len);
        anyhow::ensure!(
            n_chunks <= 32,
            "attn_prefill_ws: {n_chunks} chunks past the partial buffer's 32"
        );
        let ms_off = (32 * dims.n_heads * run_tokens * dims.d_head) as i32;

        let f = self.dev.kernels().get("infero_ops", ops_src(), "attn_prefill_mma_ws_f32")?;
        // `sq` is `NWARPS` (consumer-only) sixteen-row slabs; `sk0`/`sk1`/
        // `sv0`/`sv1` are four `WK`-wide, `krow`-wide buffers -- identical
        // shared-memory footprint to `attn_prefill_pipev`'s, since the
        // producer warp adds threads (`NWARPS + 1` total) but no new shared
        // state during the K/V loop.
        let kv_shared = NWARPS * 16 * (dims.d_head + KPAD) * 2 + 4 * WK * (dims.d_head + KPAD) * 2;
        // The kernel *can* reuse this same region afterward (dead by then)
        // to stage the `single`-path output for a coalesced write instead
        // of the MMA C-fragment's own scattered one -- see its doc comment
        // in `ops.cu`. `tpw * group` is at most 16, and at that width (a
        // wide `group`, few kv heads -- `(16, 2)` at this model's `d_head`
        // is the real example `attn_prefill_matches_the_three_kernels`
        // exercises) the staging buffer alone is past this GPU's ~100 KiB
        // dynamic-shared ceiling, past what even `set_max_dynamic_shared`
        // can grant -- confirmed by trying, not assumed (`CUDA_ERROR_INVALID_VALUE`,
        // "kernel refused a 114688-byte dynamic shared request"). Staging
        // is opt-in per launch (`use_out_stage`) for exactly this reason:
        // when it would not fit, the kernel falls back to the scattered
        // write rather than the launch failing outright.
        const MAX_DYNAMIC_SHARED: usize = 100 * 1024;
        // Matches the kernel's own `stage_row`: widened by 4 columns once
        // `d_head` is a multiple of 32, so the per-row stride stops lining
        // up with the bank count (see the kernel's comment on the 3.5-way
        // shared-store conflict this fixes).
        let stage_row = if dims.d_head % 32 == 0 { dims.d_head + 4 } else { dims.d_head };
        let out_stage_shared = NWARPS * tpw * group * stage_row * 4;
        let use_out_stage = out_stage_shared <= MAX_DYNAMIC_SHARED;
        let shared = if use_out_stage { kv_shared.max(out_stage_shared) } else { kv_shared } as u32;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared)?;
        }
        let single = i32::from(n_chunks == 1);
        let cfg = LaunchConfig {
            grid_dim: (dims.n_kv_heads as u32, n_tiles as u32, n_chunks),
            block_dim: ((NWARPS + 1) as u32 * 32, 1, 1),
            shared_mem_bytes: shared,
        };
        let (stride, h, kh, dh, ns, ck, kl, gi, tp, rb, rt, uos) = (
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            chunk as i32,
            kv_len as i32,
            group as i32,
            tpw as i32,
            run_base as i32,
            run_tokens as i32,
            i32::from(use_out_stage),
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(&mut *partial)
            .arg(&ms_off)
            .arg(&mut *out)
            .arg(&single)
            .arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ns)
            .arg(&scale)
            .arg(&ck)
            .arg(&kl)
            .arg(&gi)
            .arg(&tp)
            .arg(&rb)
            .arg(&rt)
            .arg(&uos);
        self.dev
            .profile()
            .time("attn_prefill_ws", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_prefill_ws")?;
                Ok(())
            })?;
        if single == 1 {
            return Ok(());
        }

        let total = (run_tokens * dims.n_heads * dims.d_head) as u32;
        let (nt, nc) = (run_tokens as i32, n_chunks as i32);
        let part = partial.as_view();
        let r = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_flash_reduce_f32")?;
        let run_elems = run_tokens * dims.n_heads * dims.d_head;
        let out_off = run_base * dims.n_heads * dims.d_head;
        let mut out_run = out.slice_mut(out_off..out_off + run_elems);
        let mut rb2 = self.dev.stream().launch_builder(&r);
        rb2.arg(&mut out_run)
            .arg(&part)
            .arg(&ms_off)
            .arg(&h)
            .arg(&dh)
            .arg(&nt)
            .arg(&nc);
        self.dev
            .profile()
            .time("attn_prefill_reduce", self.dev.stream(), || {
                unsafe { rb2.launch(elementwise(total)) }.context("attn_prefill_reduce")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Dispatches `attn_prefill_mma_ws4_f32`: no `sq` (Q is loaded straight
    /// into `qa[]`'s registers from global memory instead of staged through
    /// shared memory first), so the *entire* 99 KiB dynamic-shared budget
    /// goes to a 48-key-wide K/V double buffer at `NWARPS=7` -- unlike
    /// `attn_prefill_ws3` (reverted), this does not trade `NWARPS` away to
    /// fit a wider tile. See the kernel's own doc comment in `ops.cu` for
    /// the full byte accounting and why `sq` never needed to exist.
    pub fn attn_prefill_ws4(
        &self,
        out: &mut ViewMut<'_, f32>,
        q: &View<'_, f32>,
        k_cache: &View<'_, f16>,
        v_cache: &View<'_, f16>,
        batch: BatchLayout<'_>,
        dims: AttnDims,
        run_base: usize,
        run_tokens: usize,
        kv_len: usize,
        scale: f32,
        partial: &mut ViewMut<'_, f32>,
    ) -> Result<()> {
        anyhow::ensure!(self.prefill_attention(&dims), "attn_prefill_ws4: unsupported shape");
        anyhow::ensure!(run_tokens >= 1, "attn_prefill_ws4: empty run");
        // Folds `__expf`'s own internal multiply-by-log2(e) into this
        // multiply, which was already happening before every exponentiation
        // regardless -- `attn_prefill_mma_ws4_f32` now calls `exp2f`, not
        // `__expf`; see that kernel's own doc comment in `ops.cu`.
        let scale = scale * std::f32::consts::LOG2_E;
        let group = dims.n_heads / dims.n_kv_heads;
        let tpw = (16 / group).max(1);
        const NWARPS: usize = 7;
        const KPAD: usize = 8;
        const WK: usize = 48;
        let tile_tokens = NWARPS * tpw;
        let n_tiles = run_tokens.div_ceil(tile_tokens);
        let (n_chunks, chunk) = self.prefill_chunks(dims.n_kv_heads, n_tiles, kv_len);
        anyhow::ensure!(
            n_chunks <= 32,
            "attn_prefill_ws4: {n_chunks} chunks past the partial buffer's 32"
        );
        let ms_off = (32 * dims.n_heads * run_tokens * dims.d_head) as i32;

        let f = self.dev.kernels().get("infero_ops", ops_src(), "attn_prefill_mma_ws4_f32")?;
        // No `sq` term -- the K/V double buffer is the whole shared-memory
        // footprint during the main loop.
        let kv_shared = 4 * WK * (dims.d_head + KPAD) * 2;
        const MAX_DYNAMIC_SHARED: usize = 100 * 1024;
        let stage_row = if dims.d_head % 32 == 0 { dims.d_head + 4 } else { dims.d_head };
        let out_stage_shared = NWARPS * tpw * group * stage_row * 4;
        let use_out_stage = out_stage_shared <= MAX_DYNAMIC_SHARED;
        let shared = if use_out_stage { kv_shared.max(out_stage_shared) } else { kv_shared } as u32;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared)?;
        }
        let single = i32::from(n_chunks == 1);
        let cfg = LaunchConfig {
            grid_dim: (dims.n_kv_heads as u32, n_tiles as u32, n_chunks),
            block_dim: ((NWARPS + 1) as u32 * 32, 1, 1),
            shared_mem_bytes: shared,
        };
        let (stride, h, kh, dh, ns, ck, kl, gi, tp, rb, rt, uos) = (
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            chunk as i32,
            kv_len as i32,
            group as i32,
            tpw as i32,
            run_base as i32,
            run_tokens as i32,
            i32::from(use_out_stage),
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(&mut *partial)
            .arg(&ms_off)
            .arg(&mut *out)
            .arg(&single)
            .arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ns)
            .arg(&scale)
            .arg(&ck)
            .arg(&kl)
            .arg(&gi)
            .arg(&tp)
            .arg(&rb)
            .arg(&rt)
            .arg(&uos);
        self.dev
            .profile()
            .time("attn_prefill_ws4", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_prefill_ws4")?;
                Ok(())
            })?;
        if single == 1 {
            return Ok(());
        }

        let total = (run_tokens * dims.n_heads * dims.d_head) as u32;
        let (nt, nc) = (run_tokens as i32, n_chunks as i32);
        let part = partial.as_view();
        let r = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_flash_reduce_f32")?;
        let run_elems = run_tokens * dims.n_heads * dims.d_head;
        let out_off = run_base * dims.n_heads * dims.d_head;
        let mut out_run = out.slice_mut(out_off..out_off + run_elems);
        let mut rb2 = self.dev.stream().launch_builder(&r);
        rb2.arg(&mut out_run)
            .arg(&part)
            .arg(&ms_off)
            .arg(&h)
            .arg(&dh)
            .arg(&nt)
            .arg(&nc);
        self.dev
            .profile()
            .time("attn_prefill_reduce", self.dev.stream(), || {
                unsafe { rb2.launch(elementwise(total)) }.context("attn_prefill_reduce")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Diagnostic only: bulk-synchronous counterpart to [`Self::attn_prefill_ws4`]
    /// -- see `attn_prefill_mma_bulk48_f32`'s own doc comment in `ops.cu` for
    /// the hypothesis this tests (no producer/consumer role split, no named
    /// barriers, every warp free-runs its own QK/softmax/PV after a single
    /// `__syncthreads()`-gated cooperative tile load, FA2-style). Same WK4=48
    /// tile and `NWARPS=7` as `ws4`, so `tile_tokens` and grid dims match
    /// exactly -- the only difference from `ws4`'s launch is no `+1` for a
    /// producer warp.
    ///
    /// Real result: 0.851x vs `ws4` (slower) at the checkpoint's real
    /// 30552-token/16-layer shape -- see the kernel's own doc comment for
    /// the measured breakdown and the leading (unconfirmed, `ncu`-blocked)
    /// explanation.
    pub fn attn_prefill_bulk48(
        &self,
        out: &mut ViewMut<'_, f32>,
        q: &View<'_, f32>,
        k_cache: &View<'_, f16>,
        v_cache: &View<'_, f16>,
        batch: BatchLayout<'_>,
        dims: AttnDims,
        run_base: usize,
        run_tokens: usize,
        kv_len: usize,
        scale: f32,
        partial: &mut ViewMut<'_, f32>,
    ) -> Result<()> {
        anyhow::ensure!(self.prefill_attention(&dims), "attn_prefill_bulk48: unsupported shape");
        anyhow::ensure!(run_tokens >= 1, "attn_prefill_bulk48: empty run");
        let scale = scale * std::f32::consts::LOG2_E;
        let group = dims.n_heads / dims.n_kv_heads;
        let tpw = (16 / group).max(1);
        const NWARPS: usize = 7;
        const KPAD: usize = 8;
        const WK: usize = 48;
        let tile_tokens = NWARPS * tpw;
        let n_tiles = run_tokens.div_ceil(tile_tokens);
        let (n_chunks, chunk) = self.prefill_chunks(dims.n_kv_heads, n_tiles, kv_len);
        anyhow::ensure!(
            n_chunks <= 32,
            "attn_prefill_bulk48: {n_chunks} chunks past the partial buffer's 32"
        );
        let ms_off = (32 * dims.n_heads * run_tokens * dims.d_head) as i32;

        let f = self.dev.kernels().get("infero_ops", ops_src(), "attn_prefill_mma_bulk48_f32")?;
        let kv_shared = 4 * WK * (dims.d_head + KPAD) * 2;
        const MAX_DYNAMIC_SHARED: usize = 100 * 1024;
        let stage_row = if dims.d_head % 32 == 0 { dims.d_head + 4 } else { dims.d_head };
        let out_stage_shared = NWARPS * tpw * group * stage_row * 4;
        let use_out_stage = out_stage_shared <= MAX_DYNAMIC_SHARED;
        let shared = if use_out_stage { kv_shared.max(out_stage_shared) } else { kv_shared } as u32;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared)?;
        }
        let single = i32::from(n_chunks == 1);
        let cfg = LaunchConfig {
            grid_dim: (dims.n_kv_heads as u32, n_tiles as u32, n_chunks),
            block_dim: (NWARPS as u32 * 32, 1, 1),
            shared_mem_bytes: shared,
        };
        let (stride, h, kh, dh, ns, ck, kl, gi, tp, rb, rt, uos) = (
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            chunk as i32,
            kv_len as i32,
            group as i32,
            tpw as i32,
            run_base as i32,
            run_tokens as i32,
            i32::from(use_out_stage),
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(&mut *partial)
            .arg(&ms_off)
            .arg(&mut *out)
            .arg(&single)
            .arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ns)
            .arg(&scale)
            .arg(&ck)
            .arg(&kl)
            .arg(&gi)
            .arg(&tp)
            .arg(&rb)
            .arg(&rt)
            .arg(&uos);
        self.dev
            .profile()
            .time("attn_prefill_bulk48", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_prefill_bulk48")?;
                Ok(())
            })?;
        if single == 1 {
            return Ok(());
        }

        let total = (run_tokens * dims.n_heads * dims.d_head) as u32;
        let (nt, nc) = (run_tokens as i32, n_chunks as i32);
        let part = partial.as_view();
        let r = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_flash_reduce_f32")?;
        let run_elems = run_tokens * dims.n_heads * dims.d_head;
        let out_off = run_base * dims.n_heads * dims.d_head;
        let mut out_run = out.slice_mut(out_off..out_off + run_elems);
        let mut rb2 = self.dev.stream().launch_builder(&r);
        rb2.arg(&mut out_run)
            .arg(&part)
            .arg(&ms_off)
            .arg(&h)
            .arg(&dh)
            .arg(&nt)
            .arg(&nc);
        self.dev
            .profile()
            .time("attn_prefill_reduce", self.dev.stream(), || {
                unsafe { rb2.launch(elementwise(total)) }.context("attn_prefill_reduce")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Diagnostic only: `ws4`'s exact producer/7-consumer structure, but a
    /// SINGLE K/V buffer (`WK4=96`, double `ws4`'s 48) instead of a double
    /// buffer -- the same 101,376-byte ceiling spent on tile width instead
    /// of pipeline depth. Motivated by reading FlashAttention-2's own real
    /// source (`flash_fwd_kernel.h`): FA2 never double-buffers K/V either,
    /// hiding load latency via same-warp software-pipelined prefetch
    /// instead. See `attn_prefill_mma_ws5_singlebuf_regcheck_f32`'s own doc
    /// comment in `ops.cu` for why this is a genuinely different axis than
    /// `ws3`'s already-reverted wider-tile attempt (that one cut `NWARPS`
    /// to fit; this one doesn't need to -- confirmed via `kernel_registers`
    /// at exactly 255, identical to `ws4`, before ever building the
    /// correctness test below).
    pub fn attn_prefill_ws5_singlebuf(
        &self,
        out: &mut ViewMut<'_, f32>,
        q: &View<'_, f32>,
        k_cache: &View<'_, f16>,
        v_cache: &View<'_, f16>,
        batch: BatchLayout<'_>,
        dims: AttnDims,
        run_base: usize,
        run_tokens: usize,
        kv_len: usize,
        scale: f32,
        partial: &mut ViewMut<'_, f32>,
    ) -> Result<()> {
        anyhow::ensure!(self.prefill_attention(&dims), "attn_prefill_ws5_singlebuf: unsupported shape");
        anyhow::ensure!(run_tokens >= 1, "attn_prefill_ws5_singlebuf: empty run");
        let scale = scale * std::f32::consts::LOG2_E;
        let group = dims.n_heads / dims.n_kv_heads;
        let tpw = (16 / group).max(1);
        const NWARPS: usize = 7;
        const KPAD: usize = 8;
        const WK: usize = 96;
        let tile_tokens = NWARPS * tpw;
        let n_tiles = run_tokens.div_ceil(tile_tokens);
        let (n_chunks, chunk) = self.prefill_chunks(dims.n_kv_heads, n_tiles, kv_len);
        anyhow::ensure!(
            n_chunks <= 32,
            "attn_prefill_ws5_singlebuf: {n_chunks} chunks past the partial buffer's 32"
        );
        let ms_off = (32 * dims.n_heads * run_tokens * dims.d_head) as i32;

        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_prefill_mma_ws5_singlebuf_regcheck_f32")?;
        // Single K/V buffer -- half of `ws4`'s `4 * WK * (d_head+KPAD) * 2`.
        let kv_shared = 2 * WK * (dims.d_head + KPAD) * 2;
        let shared = kv_shared as u32;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared)?;
        }
        let single = i32::from(n_chunks == 1);
        let cfg = LaunchConfig {
            grid_dim: (dims.n_kv_heads as u32, n_tiles as u32, n_chunks),
            block_dim: ((NWARPS + 1) as u32 * 32, 1, 1),
            shared_mem_bytes: shared,
        };
        let (stride, h, kh, dh, ns, ck, kl, gi, tp, rb, rt) = (
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            chunk as i32,
            kv_len as i32,
            group as i32,
            tpw as i32,
            run_base as i32,
            run_tokens as i32,
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(&mut *partial)
            .arg(&ms_off)
            .arg(&mut *out)
            .arg(&single)
            .arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ns)
            .arg(&scale)
            .arg(&ck)
            .arg(&kl)
            .arg(&gi)
            .arg(&tp)
            .arg(&rb)
            .arg(&rt);
        self.dev
            .profile()
            .time("attn_prefill_ws5_singlebuf", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_prefill_ws5_singlebuf")?;
                Ok(())
            })?;
        if single == 1 {
            return Ok(());
        }

        let total = (run_tokens * dims.n_heads * dims.d_head) as u32;
        let (nt, nc) = (run_tokens as i32, n_chunks as i32);
        let part = partial.as_view();
        let r = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_flash_reduce_f32")?;
        let run_elems = run_tokens * dims.n_heads * dims.d_head;
        let out_off = run_base * dims.n_heads * dims.d_head;
        let mut out_run = out.slice_mut(out_off..out_off + run_elems);
        let mut rb2 = self.dev.stream().launch_builder(&r);
        rb2.arg(&mut out_run)
            .arg(&part)
            .arg(&ms_off)
            .arg(&h)
            .arg(&dh)
            .arg(&nt)
            .arg(&nc);
        self.dev
            .profile()
            .time("attn_prefill_reduce", self.dev.stream(), || {
                unsafe { rb2.launch(elementwise(total)) }.context("attn_prefill_reduce")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Diagnostic only: [`Self::attn_prefill_ws4`] with `NWARPS` exposed as a
    /// runtime parameter instead of the hardcoded compile-time `7` --
    /// otherwise byte-for-byte identical, calling the exact same, already-
    /// correctness-tested `attn_prefill_mma_ws4_f32` kernel (which already
    /// reads `nwarps` from `blockDim.x` at runtime, so this needs zero
    /// kernel-side changes). Checks a real, FlashAttention-2-inspired lever
    /// this session hadn't tried: shrinking the block (fewer consumer warps,
    /// same per-warp work, same per-thread register count) so more than one
    /// block's worth of registers fit an SM, at a real but much smaller
    /// memory-retraffic cost (`7/NWARPS`x more blocks, each still streaming
    /// the full causal K/V range) than the decoupled-role kernel's 7x tax
    /// (which came from an 8-way *role* fragmentation, not this simple
    /// block-shrink). See FA2's own `kernel_traits.h`/`flash_fwd_launch_
    /// template.h`: its H100 config does exactly this (kBlockM 128->64,
    /// warps 8->4) to get 2 CTAs/SM at hdim256, with no accumulator
    /// register trick at all.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_ws4_nw(
        &self,
        out: &mut ViewMut<'_, f32>,
        q: &View<'_, f32>,
        k_cache: &View<'_, f16>,
        v_cache: &View<'_, f16>,
        batch: BatchLayout<'_>,
        dims: AttnDims,
        run_base: usize,
        run_tokens: usize,
        kv_len: usize,
        scale: f32,
        partial: &mut ViewMut<'_, f32>,
        nwarps: usize,
    ) -> Result<()> {
        anyhow::ensure!(self.prefill_attention(&dims), "attn_prefill_ws4_nw: unsupported shape");
        anyhow::ensure!(run_tokens >= 1, "attn_prefill_ws4_nw: empty run");
        // See `attn_prefill_ws4`'s own comment: folds `__expf`'s internal
        // log2(e) multiply into this existing one.
        let scale = scale * std::f32::consts::LOG2_E;
        let group = dims.n_heads / dims.n_kv_heads;
        let tpw = (16 / group).max(1);
        const KPAD: usize = 8;
        const WK: usize = 48;
        let tile_tokens = nwarps * tpw;
        let n_tiles = run_tokens.div_ceil(tile_tokens);
        let (n_chunks, chunk) = self.prefill_chunks(dims.n_kv_heads, n_tiles, kv_len);
        anyhow::ensure!(
            n_chunks <= 32,
            "attn_prefill_ws4_nw: {n_chunks} chunks past the partial buffer's 32"
        );
        let ms_off = (32 * dims.n_heads * run_tokens * dims.d_head) as i32;

        let f = self.dev.kernels().get("infero_ops", ops_src(), "attn_prefill_mma_ws4_f32")?;
        let kv_shared = 4 * WK * (dims.d_head + KPAD) * 2;
        const MAX_DYNAMIC_SHARED: usize = 100 * 1024;
        let stage_row = if dims.d_head % 32 == 0 { dims.d_head + 4 } else { dims.d_head };
        let out_stage_shared = nwarps * tpw * group * stage_row * 4;
        let use_out_stage = out_stage_shared <= MAX_DYNAMIC_SHARED;
        let shared = if use_out_stage { kv_shared.max(out_stage_shared) } else { kv_shared } as u32;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared)?;
        }
        let single = i32::from(n_chunks == 1);
        let cfg = LaunchConfig {
            grid_dim: (dims.n_kv_heads as u32, n_tiles as u32, n_chunks),
            block_dim: ((nwarps + 1) as u32 * 32, 1, 1),
            shared_mem_bytes: shared,
        };
        let (stride, h, kh, dh, ns, ck, kl, gi, tp, rb, rt, uos) = (
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            chunk as i32,
            kv_len as i32,
            group as i32,
            tpw as i32,
            run_base as i32,
            run_tokens as i32,
            i32::from(use_out_stage),
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(&mut *partial)
            .arg(&ms_off)
            .arg(&mut *out)
            .arg(&single)
            .arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ns)
            .arg(&scale)
            .arg(&ck)
            .arg(&kl)
            .arg(&gi)
            .arg(&tp)
            .arg(&rb)
            .arg(&rt)
            .arg(&uos);
        self.dev
            .profile()
            .time("attn_prefill_ws4_nw", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_prefill_ws4_nw")?;
                Ok(())
            })?;
        if single == 1 {
            return Ok(());
        }
        let total = (run_tokens * dims.n_heads * dims.d_head) as u32;
        let (nt, nc) = (run_tokens as i32, n_chunks as i32);
        let part = partial.as_view();
        let r = self.dev.kernels().get("infero_ops", ops_src(), "attn_flash_reduce_f32")?;
        let run_elems = run_tokens * dims.n_heads * dims.d_head;
        let out_off = run_base * dims.n_heads * dims.d_head;
        let mut out_run = out.slice_mut(out_off..out_off + run_elems);
        let mut rb2 = self.dev.stream().launch_builder(&r);
        rb2.arg(&mut out_run).arg(&part).arg(&ms_off).arg(&h).arg(&dh).arg(&nt).arg(&nc);
        self.dev
            .profile()
            .time("attn_prefill_ws4_nw_reduce", self.dev.stream(), || {
                unsafe { rb2.launch(elementwise(total)) }.context("attn_prefill_ws4_nw_reduce")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Validation-only, single-chunk: the decoupled-role design the
    /// `attn_prefill_mma_ws4_qkt_only_regcheck_f32` / `..._pvonly_...` /
    /// `..._decoupled_regcheck_f32` register-count probes (in `ops.cu`)
    /// found does not cost more than it buys, unlike every warp-
    /// fragmentation scheme tried before it. 4 warps/block (memory
    /// producer, QK^T-only, 2 PV-consumers for each d_head half) covering
    /// one 16-row tile, instead of `ws4`'s 8 warps covering 7 tiles' worth
    /// -- `n_tiles` grows 7x to compensate. No `partial`/chunking yet
    /// (deliberately, see `attn_prefill_decoupled_f32`'s doc comment) --
    /// not the real `ws4` replacement while that's true.
    pub fn attn_prefill_decoupled(
        &self,
        out: &mut ViewMut<'_, f32>,
        q: &View<'_, f32>,
        k_cache: &View<'_, f16>,
        v_cache: &View<'_, f16>,
        batch: BatchLayout<'_>,
        dims: AttnDims,
        run_base: usize,
        run_tokens: usize,
        kv_len: usize,
        scale: f32,
    ) -> Result<()> {
        anyhow::ensure!(self.prefill_attention(&dims), "attn_prefill_decoupled: unsupported shape");
        anyhow::ensure!(run_tokens >= 1, "attn_prefill_decoupled: empty run");
        let group = dims.n_heads / dims.n_kv_heads;
        let tpw = (16 / group).max(1);
        const KPAD: usize = 8;
        // 32, not ws4's 48 -- see `attn_prefill_decoupled_f32`'s doc
        // comment in `ops.cu`: ws4's K/V double buffer at WK=48 already
        // spends this GPU's exact 99 KiB shared-memory ceiling, leaving no
        // room for this design's extra score buffer.
        const WK: usize = 32;
        let n_tiles = run_tokens.div_ceil(tpw);

        let f = self.dev.kernels().get("infero_ops", ops_src(), "attn_prefill_decoupled_f32")?;
        let kv_shared = 4 * WK * (dims.d_head + KPAD) * 2;
        // Lane-indexed, not key-indexed (see `attn_prefill_decoupled_f32`'s
        // `sc0` doc comment): 32 lanes x 4 floats per nt sub-tile, x2
        // double-buffered stages.
        let score_shared = 2 * (WK / 8) * 32 * 4 * 4;
        let shared = (kv_shared + score_shared) as u32;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared)?;
        }
        let cfg = LaunchConfig {
            grid_dim: (dims.n_kv_heads as u32, n_tiles as u32, 1),
            block_dim: (4 * 32, 1, 1),
            shared_mem_bytes: shared,
        };
        let (stride, h, kh, dh, ns, kl, gi, tp, rb, rt) = (
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            kv_len as i32,
            group as i32,
            tpw as i32,
            run_base as i32,
            run_tokens as i32,
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(&mut *out)
            .arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ns)
            .arg(&scale)
            .arg(&kl)
            .arg(&gi)
            .arg(&tp)
            .arg(&rb)
            .arg(&rt);
        self.dev
            .profile()
            .time("attn_prefill_decoupled", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_prefill_decoupled")?;
                Ok(())
            })?;
        Ok(())
    }

    /// T=2 generalization of [`Self::attn_prefill_decoupled`] -- see
    /// `attn_prefill_decoupled2_f32`'s doc comment in `ops.cu` for the
    /// register/memory-tradeoff math this exists to actually measure
    /// instead of just reason about.
    pub fn attn_prefill_decoupled2(
        &self,
        out: &mut ViewMut<'_, f32>,
        q: &View<'_, f32>,
        k_cache: &View<'_, f16>,
        v_cache: &View<'_, f16>,
        batch: BatchLayout<'_>,
        dims: AttnDims,
        run_base: usize,
        run_tokens: usize,
        kv_len: usize,
        scale: f32,
    ) -> Result<()> {
        anyhow::ensure!(self.prefill_attention(&dims), "attn_prefill_decoupled2: unsupported shape");
        anyhow::ensure!(run_tokens >= 1, "attn_prefill_decoupled2: empty run");
        let group = dims.n_heads / dims.n_kv_heads;
        let tpw = (16 / group).max(1);
        const KPAD: usize = 8;
        const WK: usize = 32;
        const TG: usize = 2;
        let n_big_tiles = run_tokens.div_ceil(TG * tpw);

        let f = self.dev.kernels().get("infero_ops", ops_src(), "attn_prefill_decoupled2_f32")?;
        let kv_shared = 4 * WK * (dims.d_head + KPAD) * 2;
        // 2 groups x 2 stages x (lane-indexed score buffer per T=1's own
        // sizing) -- see `attn_prefill_decoupled2_f32`'s `sc` doc comment.
        let score_shared = TG * 2 * (WK / 8) * 32 * 4 * 4;
        let shared = (kv_shared + score_shared) as u32;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared)?;
        }
        let cfg = LaunchConfig {
            grid_dim: (dims.n_kv_heads as u32, n_big_tiles as u32, 1),
            block_dim: ((1 + 3 * TG) as u32 * 32, 1, 1),
            shared_mem_bytes: shared,
        };
        let (stride, h, kh, dh, ns, kl, gi, tp, rb, rt) = (
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            kv_len as i32,
            group as i32,
            tpw as i32,
            run_base as i32,
            run_tokens as i32,
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(&mut *out)
            .arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ns)
            .arg(&scale)
            .arg(&kl)
            .arg(&gi)
            .arg(&tp)
            .arg(&rb)
            .arg(&rt);
        self.dev
            .profile()
            .time("attn_prefill_decoupled2", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_prefill_decoupled2")?;
                Ok(())
            })?;
        Ok(())
    }

    /// [`Self::attn_prefill_decoupled2`], with the PV-consumer role's `o[]`
    /// accumulated in fp16 instead of fp32 -- unlike `ws4`'s own fp16-
    /// accumulate experiment (which only captured 21 of a naive 64-register
    /// saving), this role has no `qa[]`/QK^T competing for live ranges, so
    /// it captures the full naive 32-register saving (128 -> 96), even
    /// combined into the real multi-role kernel (confirmed via
    /// `kernel_registers` before ever launching it). Same launch shape as
    /// T=2 otherwise -- only the register count differs, which is exactly
    /// what determines blocks/SM.
    pub fn attn_prefill_decoupled2_f16acc(
        &self,
        out: &mut ViewMut<'_, f32>,
        q: &View<'_, f32>,
        k_cache: &View<'_, f16>,
        v_cache: &View<'_, f16>,
        batch: BatchLayout<'_>,
        dims: AttnDims,
        run_base: usize,
        run_tokens: usize,
        kv_len: usize,
        scale: f32,
    ) -> Result<()> {
        anyhow::ensure!(self.prefill_attention(&dims), "attn_prefill_decoupled2_f16acc: unsupported shape");
        anyhow::ensure!(run_tokens >= 1, "attn_prefill_decoupled2_f16acc: empty run");
        let group = dims.n_heads / dims.n_kv_heads;
        let tpw = (16 / group).max(1);
        const KPAD: usize = 8;
        const WK: usize = 32;
        const TG: usize = 2;
        let n_big_tiles = run_tokens.div_ceil(TG * tpw);

        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_prefill_decoupled2_f16acc_regcheck_f32")?;
        let kv_shared = 4 * WK * (dims.d_head + KPAD) * 2;
        let score_shared = TG * 2 * (WK / 8) * 32 * 4 * 4;
        let shared = (kv_shared + score_shared) as u32;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared)?;
        }
        let cfg = LaunchConfig {
            grid_dim: (dims.n_kv_heads as u32, n_big_tiles as u32, 1),
            block_dim: ((1 + 3 * TG) as u32 * 32, 1, 1),
            shared_mem_bytes: shared,
        };
        let (stride, h, kh, dh, ns, kl, gi, tp, rb, rt) = (
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            kv_len as i32,
            group as i32,
            tpw as i32,
            run_base as i32,
            run_tokens as i32,
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(&mut *out)
            .arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ns)
            .arg(&scale)
            .arg(&kl)
            .arg(&gi)
            .arg(&tp)
            .arg(&rb)
            .arg(&rt);
        self.dev
            .profile()
            .time("attn_prefill_decoupled2_f16acc", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_prefill_decoupled2_f16acc")?;
                Ok(())
            })?;
        Ok(())
    }

    /// T=3 generalization of [`Self::attn_prefill_decoupled2_f16acc`] --
    /// lower memory tax (7/T: 2.33x instead of T=2's 3.5x) at 96 registers
    /// x 320 threads/block = 30,720 regs/block -> 2 resident blocks/SM (20
    /// resident warps). See `attn_prefill_decoupled3_f16acc_regcheck_f32`'s
    /// doc comment in `ops.cu` for the named-barrier-ID-budget constraint
    /// (16 IDs total, 0-15) this required working around.
    pub fn attn_prefill_decoupled3_f16acc(
        &self,
        out: &mut ViewMut<'_, f32>,
        q: &View<'_, f32>,
        k_cache: &View<'_, f16>,
        v_cache: &View<'_, f16>,
        batch: BatchLayout<'_>,
        dims: AttnDims,
        run_base: usize,
        run_tokens: usize,
        kv_len: usize,
        scale: f32,
    ) -> Result<()> {
        anyhow::ensure!(self.prefill_attention(&dims), "attn_prefill_decoupled3_f16acc: unsupported shape");
        anyhow::ensure!(run_tokens >= 1, "attn_prefill_decoupled3_f16acc: empty run");
        let group = dims.n_heads / dims.n_kv_heads;
        let tpw = (16 / group).max(1);
        const KPAD: usize = 8;
        const WK: usize = 32;
        const TG: usize = 3;
        let n_big_tiles = run_tokens.div_ceil(TG * tpw);

        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_prefill_decoupled3_f16acc_regcheck_f32")?;
        let kv_shared = 4 * WK * (dims.d_head + KPAD) * 2;
        let score_shared = TG * 2 * (WK / 8) * 32 * 4 * 4;
        let shared = (kv_shared + score_shared) as u32;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared)?;
        }
        let cfg = LaunchConfig {
            grid_dim: (dims.n_kv_heads as u32, n_big_tiles as u32, 1),
            block_dim: ((1 + 3 * TG) as u32 * 32, 1, 1),
            shared_mem_bytes: shared,
        };
        let (stride, h, kh, dh, ns, kl, gi, tp, rb, rt) = (
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            kv_len as i32,
            group as i32,
            tpw as i32,
            run_base as i32,
            run_tokens as i32,
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(&mut *out)
            .arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ns)
            .arg(&scale)
            .arg(&kl)
            .arg(&gi)
            .arg(&tp)
            .arg(&rb)
            .arg(&rt);
        self.dev
            .profile()
            .time("attn_prefill_decoupled3_f16acc", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_prefill_decoupled3_f16acc")?;
                Ok(())
            })?;
        Ok(())
    }

    /// T=4 generalization of [`Self::attn_prefill_decoupled3_f16acc`] --
    /// lower memory tax (7/T: 1.75x) but only 1 resident block/SM (13
    /// resident warps, down from T=3's 20) since 416 threads x 96
    /// registers = 39,936 regs/block leaves no room for a 2nd. Also
    /// switches the score-ready/free barriers from per-group to shared
    /// across all T groups (a real synchronization redesign, not just a
    /// renumbering) because the per-group scheme's 4 + 4*T named-barrier
    /// IDs would need 20, over the hardware's fixed 16-ID budget. See
    /// `attn_prefill_decoupled4_f16acc_regcheck_f32`'s doc comment in
    /// `ops.cu` for why the shared-barrier substitution is safe (every
    /// warp already loops the same shared `n_blk` bound, so groups never
    /// ran at independent paces to begin with).
    pub fn attn_prefill_decoupled4_f16acc(
        &self,
        out: &mut ViewMut<'_, f32>,
        q: &View<'_, f32>,
        k_cache: &View<'_, f16>,
        v_cache: &View<'_, f16>,
        batch: BatchLayout<'_>,
        dims: AttnDims,
        run_base: usize,
        run_tokens: usize,
        kv_len: usize,
        scale: f32,
    ) -> Result<()> {
        anyhow::ensure!(self.prefill_attention(&dims), "attn_prefill_decoupled4_f16acc: unsupported shape");
        anyhow::ensure!(run_tokens >= 1, "attn_prefill_decoupled4_f16acc: empty run");
        let group = dims.n_heads / dims.n_kv_heads;
        let tpw = (16 / group).max(1);
        const KPAD: usize = 8;
        const WK: usize = 32;
        const TG: usize = 4;
        let n_big_tiles = run_tokens.div_ceil(TG * tpw);

        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_prefill_decoupled4_f16acc_regcheck_f32")?;
        let kv_shared = 4 * WK * (dims.d_head + KPAD) * 2;
        let score_shared = TG * 2 * (WK / 8) * 32 * 4 * 4;
        let shared = (kv_shared + score_shared) as u32;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared)?;
        }
        let cfg = LaunchConfig {
            grid_dim: (dims.n_kv_heads as u32, n_big_tiles as u32, 1),
            block_dim: ((1 + 3 * TG) as u32 * 32, 1, 1),
            shared_mem_bytes: shared,
        };
        let (stride, h, kh, dh, ns, kl, gi, tp, rb, rt) = (
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            kv_len as i32,
            group as i32,
            tpw as i32,
            run_base as i32,
            run_tokens as i32,
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(&mut *out)
            .arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ns)
            .arg(&scale)
            .arg(&kl)
            .arg(&gi)
            .arg(&tp)
            .arg(&rb)
            .arg(&rt);
        self.dev
            .profile()
            .time("attn_prefill_decoupled4_f16acc", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_prefill_decoupled4_f16acc")?;
                Ok(())
            })?;
        Ok(())
    }

    /// T=5 generalization of [`Self::attn_prefill_decoupled4_f16acc`] --
    /// motivated directly by T=4's own real result (0.701x -> 0.848x, the
    /// biggest single-step gain in this family): unlike T=2->T=3 or
    /// T=3->T=4, T=4->T=5 is a genuine double win on the register math --
    /// 512 threads x 96 registers = 49,152 regs/block, still 1 resident
    /// block/SM but 16 resident warps (up from T=4's 13), tax down to
    /// 7/5 = 1.4x. Same shared score-barrier scheme as T=4 (still just 8
    /// named-barrier IDs, independent of T).
    pub fn attn_prefill_decoupled5_f16acc(
        &self,
        out: &mut ViewMut<'_, f32>,
        q: &View<'_, f32>,
        k_cache: &View<'_, f16>,
        v_cache: &View<'_, f16>,
        batch: BatchLayout<'_>,
        dims: AttnDims,
        run_base: usize,
        run_tokens: usize,
        kv_len: usize,
        scale: f32,
    ) -> Result<()> {
        anyhow::ensure!(self.prefill_attention(&dims), "attn_prefill_decoupled5_f16acc: unsupported shape");
        anyhow::ensure!(run_tokens >= 1, "attn_prefill_decoupled5_f16acc: empty run");
        let group = dims.n_heads / dims.n_kv_heads;
        let tpw = (16 / group).max(1);
        const KPAD: usize = 8;
        const WK: usize = 32;
        const TG: usize = 5;
        let n_big_tiles = run_tokens.div_ceil(TG * tpw);

        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_prefill_decoupled5_f16acc_regcheck_f32")?;
        let kv_shared = 4 * WK * (dims.d_head + KPAD) * 2;
        let score_shared = TG * 2 * (WK / 8) * 32 * 4 * 4;
        let shared = (kv_shared + score_shared) as u32;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared)?;
        }
        let cfg = LaunchConfig {
            grid_dim: (dims.n_kv_heads as u32, n_big_tiles as u32, 1),
            block_dim: ((1 + 3 * TG) as u32 * 32, 1, 1),
            shared_mem_bytes: shared,
        };
        let (stride, h, kh, dh, ns, kl, gi, tp, rb, rt) = (
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            kv_len as i32,
            group as i32,
            tpw as i32,
            run_base as i32,
            run_tokens as i32,
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(&mut *out)
            .arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ns)
            .arg(&scale)
            .arg(&kl)
            .arg(&gi)
            .arg(&tp)
            .arg(&rb)
            .arg(&rt);
        self.dev
            .profile()
            .time("attn_prefill_decoupled5_f16acc", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_prefill_decoupled5_f16acc")?;
                Ok(())
            })?;
        Ok(())
    }

    /// T=6 generalization -- the practical ceiling of this family at 96
    /// registers: 608 threads x 96 registers = 58,368 regs/block, still 1
    /// resident block/SM, 19 resident warps (up from T=5's 16), tax down
    /// to 7/6 = 1.167x. T=7 would need 67,584 registers/block, over the
    /// SM's entire 65,536-register file -- an outright launch failure, not
    /// a spill -- so this is as far as this exact register count goes.
    pub fn attn_prefill_decoupled6_f16acc(
        &self,
        out: &mut ViewMut<'_, f32>,
        q: &View<'_, f32>,
        k_cache: &View<'_, f16>,
        v_cache: &View<'_, f16>,
        batch: BatchLayout<'_>,
        dims: AttnDims,
        run_base: usize,
        run_tokens: usize,
        kv_len: usize,
        scale: f32,
    ) -> Result<()> {
        anyhow::ensure!(self.prefill_attention(&dims), "attn_prefill_decoupled6_f16acc: unsupported shape");
        anyhow::ensure!(run_tokens >= 1, "attn_prefill_decoupled6_f16acc: empty run");
        let group = dims.n_heads / dims.n_kv_heads;
        let tpw = (16 / group).max(1);
        const KPAD: usize = 8;
        const WK: usize = 32;
        const TG: usize = 6;
        let n_big_tiles = run_tokens.div_ceil(TG * tpw);

        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_prefill_decoupled6_f16acc_regcheck_f32")?;
        let kv_shared = 4 * WK * (dims.d_head + KPAD) * 2;
        let score_shared = TG * 2 * (WK / 8) * 32 * 4 * 4;
        let shared = (kv_shared + score_shared) as u32;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared)?;
        }
        let cfg = LaunchConfig {
            grid_dim: (dims.n_kv_heads as u32, n_big_tiles as u32, 1),
            block_dim: ((1 + 3 * TG) as u32 * 32, 1, 1),
            shared_mem_bytes: shared,
        };
        let (stride, h, kh, dh, ns, kl, gi, tp, rb, rt) = (
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            kv_len as i32,
            group as i32,
            tpw as i32,
            run_base as i32,
            run_tokens as i32,
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(&mut *out)
            .arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ns)
            .arg(&scale)
            .arg(&kl)
            .arg(&gi)
            .arg(&tp)
            .arg(&rb)
            .arg(&rt);
        self.dev
            .profile()
            .time("attn_prefill_decoupled6_f16acc", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_prefill_decoupled6_f16acc")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Validation-only: `ws4` with e4m3 QK^T against a plain contiguous,
    /// single-sequence, single-chunk K (already quantized by
    /// [`Self::quantize_k_e4m3`]) and V, no paged pool, no `BatchLayout`.
    /// See `attn_prefill_e4m3k_f32`'s doc comment in `ops.cu` for why this
    /// exists and why it is not the real `ws4` replacement: that needs a
    /// persistent e4m3 shadow cache spanning prefill chunks, deliberately
    /// not built yet. `kq`/`kscale`/`v` are `[position, kv_head, d_head]`
    /// (`kscale` `[position, kv_head]`), matching what
    /// `Self::quantize_k_e4m3` produces directly.
    #[allow(clippy::too_many_arguments)]
    /// Same chunk/grid.z/partial-buffer mechanism as [`Self::attn_prefill_ws4`]
    /// (real chunking via [`Self::prefill_chunks`], `attn_flash_reduce_f32`
    /// combines chunks when there's more than one) -- added after this
    /// kernel's own real end-to-end benchmark found it 14.5% slower than
    /// `ws4` at the real 30552-token chunked shape, and losing this exact
    /// parallelism (this kernel had none before) was one of two diagnosed
    /// causes. `partial` must be sized via [`Self::attn_partial_floats`].
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_e4m3k(
        &self,
        out: &mut ViewMut<'_, f32>,
        q: &View<'_, f32>,
        kq: &View<'_, u8>,
        kscale: &View<'_, f32>,
        v: &View<'_, f16>,
        dims: AttnDims,
        run_base: usize,
        run_tokens: usize,
        kv_len: usize,
        scale: f32,
        partial: &mut ViewMut<'_, f32>,
    ) -> Result<()> {
        anyhow::ensure!(dims.d_head == 256, "attn_prefill_e4m3k: only this checkpoint's d_head=256 is supported");
        anyhow::ensure!(run_tokens >= 1, "attn_prefill_e4m3k: empty run");
        let group = dims.n_heads / dims.n_kv_heads;
        let tpw = (16 / group).max(1);
        const NWARPS: usize = 7;
        const KPAD: usize = 8;
        // e4m3 K is one byte an element, half `ws4`'s `__half` -- this
        // kernel's shared-memory footprint at the same tile width is
        // smaller, so 64 fits the same 99 KiB budget with room to spare
        // (must match `ATTN_E4M3_WK` in `ops.cu`).
        const WK: usize = 64;
        let tile_tokens = NWARPS * tpw;
        let n_tiles = run_tokens.div_ceil(tile_tokens);
        let (n_chunks, chunk) = self.prefill_chunks(dims.n_kv_heads, n_tiles, kv_len);
        anyhow::ensure!(
            n_chunks <= 32,
            "attn_prefill_e4m3k: {n_chunks} chunks past the partial buffer's 32"
        );
        let ms_off = (32 * dims.n_heads * run_tokens * dims.d_head) as i32;

        let f = self.dev.kernels().get("infero_ops", ops_src(), "attn_prefill_e4m3k_f32")?;
        let krow = dims.d_head + KPAD;
        let shared = (2 * WK * dims.d_head + 2 * WK * krow * 2 + 2 * WK * 4) as u32;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared)?;
        }
        let single = i32::from(n_chunks == 1);
        let cfg = LaunchConfig {
            grid_dim: (dims.n_kv_heads as u32, n_tiles as u32, n_chunks),
            block_dim: ((NWARPS + 1) as u32 * 32, 1, 1),
            shared_mem_bytes: shared,
        };
        let (h, kh, dh) = (dims.n_heads as i32, dims.n_kv_heads as i32, dims.d_head as i32);
        let (ck, kl, gi, tp) = (chunk as i32, kv_len as i32, group as i32, tpw as i32);
        let (rb, rt) = (run_base as i32, run_tokens as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(&mut *partial).arg(&ms_off).arg(&mut *out).arg(&single)
            .arg(q).arg(kq).arg(kscale).arg(v)
            .arg(&h).arg(&kh).arg(&dh).arg(&scale).arg(&ck).arg(&kl).arg(&gi).arg(&tp).arg(&rb).arg(&rt);
        self.dev
            .profile()
            .time("attn_prefill_e4m3k", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_prefill_e4m3k")?;
                Ok(())
            })?;
        if single == 1 {
            return Ok(());
        }

        let total = (run_tokens * dims.n_heads * dims.d_head) as u32;
        let (nt, nc) = (run_tokens as i32, n_chunks as i32);
        let part = partial.as_view();
        let r = self.dev.kernels().get("infero_ops", ops_src(), "attn_flash_reduce_f32")?;
        let run_elems = run_tokens * dims.n_heads * dims.d_head;
        let out_off = run_base * dims.n_heads * dims.d_head;
        let mut out_run = out.slice_mut(out_off..out_off + run_elems);
        let mut rb2 = self.dev.stream().launch_builder(&r);
        rb2.arg(&mut out_run).arg(&part).arg(&ms_off).arg(&h).arg(&dh).arg(&nt).arg(&nc);
        self.dev
            .profile()
            .time("attn_prefill_e4m3k_reduce", self.dev.stream(), || {
                unsafe { rb2.launch(elementwise(total)) }.context("attn_prefill_e4m3k_reduce")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Option 3's two-kernel split (see `attn_prefill_stats_f32`'s doc
    /// comment in `ops.cu` for the full design and reasoning). Runs the
    /// stats pass (`attn_prefill_stats_f32`: QK^T + exact online-softmax
    /// `(m, l)`, no `V`, no `o[]`, a 96-key tile) into `ms_scratch`'s
    /// per-chunk region, reduces chunks with `attn_ms_reduce_f32`, then runs
    /// the PV pass (`attn_prefill_pv_f32`: recomputes QK^T against the
    /// now-exact `(m, l)`, no running softmax state, a 48-key tile) into
    /// `out` directly (single chunk) or `partial`'s per-chunk region
    /// followed by `attn_pv_sum_reduce_f32` (multiple chunks).
    ///
    /// Real chunking now (`Self::prefill_chunks`, the same heuristic every
    /// other prefill kernel here uses) instead of the single-mega-chunk
    /// workaround this function used before the cross-chunk reduce kernels
    /// existed -- that workaround gave up grid.z parallelism entirely,
    /// confounding this path's own real cost with an avoidable one.
    ///
    /// Both kernels use the *same* `NWARPS` (hence the same `tile_tokens`
    /// and grid shape) deliberately: `ms_scratch`/`partial` are indexed
    /// relative to `run_base` by `tile * tile_tokens + local0 + row_j`, and
    /// that only names the same token in both kernels if their tiling
    /// agrees. Do not give them different `NWARPS` without also changing the
    /// indexing to something tile-shape-independent.
    pub fn attn_prefill_split(
        &self,
        out: &mut ViewMut<'_, f32>,
        q: &View<'_, f32>,
        k_cache: &View<'_, f16>,
        v_cache: &View<'_, f16>,
        batch: BatchLayout<'_>,
        dims: AttnDims,
        run_base: usize,
        run_tokens: usize,
        kv_len: usize,
        scale: f32,
        ms_scratch: &mut ViewMut<'_, f32>,
        partial: &mut ViewMut<'_, f32>,
    ) -> Result<()> {
        anyhow::ensure!(self.prefill_attention(&dims), "attn_prefill_split: unsupported shape");
        anyhow::ensure!(run_tokens >= 1, "attn_prefill_split: empty run");
        let group = dims.n_heads / dims.n_kv_heads;
        let tpw = (16 / group).max(1);
        const NWARPS: usize = 7;
        const KPAD: usize = 8;
        let tile_tokens = NWARPS * tpw;
        let n_tiles = run_tokens.div_ceil(tile_tokens);
        let (n_chunks, chunk) = self.prefill_chunks(dims.n_kv_heads, n_tiles, kv_len);
        anyhow::ensure!(
            n_chunks <= 32,
            "attn_prefill_split: {n_chunks} chunks past the scratch buffers' 32"
        );

        let (stride, h, kh, dh, ns, ck, kl, gi, tp, rb, rt) = (
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            chunk as i32,
            kv_len as i32,
            group as i32,
            tpw as i32,
            run_base as i32,
            run_tokens as i32,
        );
        let grid_dim = (dims.n_kv_heads as u32, n_tiles as u32, n_chunks);
        let block_dim = ((NWARPS + 1) as u32 * 32, 1, 1);
        let ms_partial_len = (n_chunks as usize) * dims.n_heads * run_tokens * 2;
        let ms_total_len = 33 * dims.n_heads * run_tokens * 2;

        // Stats pass: K only, no `sq`, 96-key tile. Writes chunk-indexed
        // partials into the front of `ms_scratch`.
        {
            const WKS: usize = 96;
            let f = self.dev.kernels().get("infero_ops", ops_src(), "attn_prefill_stats_f32")?;
            let shared = (2 * WKS * (dims.d_head + KPAD) * 2) as u32;
            if shared > 48 * 1024 {
                infero_gpu::set_max_dynamic_shared(&f, shared)?;
            }
            let mut ms_partial = ms_scratch.slice_mut(0..ms_partial_len);
            let mut b = self.dev.stream().launch_builder(&f);
            b.arg(&mut ms_partial)
                .arg(q)
                .arg(k_cache)
                .arg(batch.seq_of)
                .arg(batch.positions)
                .arg(batch.slot_table)
                .arg(&stride)
                .arg(&h)
                .arg(&kh)
                .arg(&dh)
                .arg(&ns)
                .arg(&scale)
                .arg(&ck)
                .arg(&kl)
                .arg(&gi)
                .arg(&tp)
                .arg(&rb)
                .arg(&rt);
            let cfg = LaunchConfig { grid_dim, block_dim, shared_mem_bytes: shared };
            self.dev
                .profile()
                .time("attn_prefill_stats", self.dev.stream(), || {
                    unsafe { b.launch(cfg) }.context("attn_prefill_stats")?;
                    Ok(())
                })?;
        }

        // Merge chunk partials into the exact global (m, l), stored right
        // after the partial region so both live in one caller-owned buffer.
        {
            let total = (run_tokens * dims.n_heads) as u32;
            let (rh, rt2, nc) = (h, run_tokens as i32, n_chunks as i32);
            // `split_at_mut`, not two separate `slice`/`slice_mut` calls: the
            // borrow checker can't otherwise see that the two ranges are
            // disjoint within one `&mut ViewMut`. The caller's buffer is
            // sized exactly `attn_ms_floats` (`ms_total_len`), so the second
            // half is already exactly the final-slot region.
            let (ms_partial, mut ms_final) = ms_scratch.split_at_mut(ms_partial_len);
            let ms_partial = ms_partial.as_view();
            let f = self.dev.kernels().get("infero_ops", ops_src(), "attn_ms_reduce_f32")?;
            let mut b = self.dev.stream().launch_builder(&f);
            b.arg(&mut ms_final).arg(&ms_partial).arg(&rh).arg(&rt2).arg(&nc);
            self.dev
                .profile()
                .time("attn_ms_reduce", self.dev.stream(), || {
                    unsafe { b.launch(elementwise(total)) }.context("attn_ms_reduce")?;
                    Ok(())
                })?;
        }

        // PV pass: K and V, 48-key tile, no running softmax state. Single
        // chunk writes `out` directly; multiple chunks write `partial`'s
        // per-chunk region for `attn_pv_sum_reduce_f32` below.
        let single = i32::from(n_chunks == 1);
        {
            const WKP: usize = 48;
            let f = self.dev.kernels().get("infero_ops", ops_src(), "attn_prefill_pv_f32")?;
            let shared = (4 * WKP * (dims.d_head + KPAD) * 2) as u32;
            if shared > 48 * 1024 {
                infero_gpu::set_max_dynamic_shared(&f, shared)?;
            }
            let ms_final = ms_scratch.slice(ms_partial_len..ms_total_len);
            let mut b = self.dev.stream().launch_builder(&f);
            b.arg(&mut *out)
                .arg(&mut *partial)
                .arg(&single)
                .arg(&ms_final)
                .arg(q)
                .arg(k_cache)
                .arg(v_cache)
                .arg(batch.seq_of)
                .arg(batch.positions)
                .arg(batch.slot_table)
                .arg(&stride)
                .arg(&h)
                .arg(&kh)
                .arg(&dh)
                .arg(&ns)
                .arg(&scale)
                .arg(&ck)
                .arg(&kl)
                .arg(&gi)
                .arg(&tp)
                .arg(&rb)
                .arg(&rt);
            let cfg = LaunchConfig { grid_dim, block_dim, shared_mem_bytes: shared };
            self.dev
                .profile()
                .time("attn_prefill_pv", self.dev.stream(), || {
                    unsafe { b.launch(cfg) }.context("attn_prefill_pv")?;
                    Ok(())
                })?;
        }
        if single == 1 {
            return Ok(());
        }

        {
            let total = (run_tokens * dims.n_heads * dims.d_head) as u32;
            let (nt, nc) = (run_tokens as i32, n_chunks as i32);
            let out_off = run_base * dims.n_heads * dims.d_head;
            let run_elems = run_tokens * dims.n_heads * dims.d_head;
            let mut out_run = out.slice_mut(out_off..out_off + run_elems);
            let part = partial.as_view();
            let f = self.dev.kernels().get("infero_ops", ops_src(), "attn_pv_sum_reduce_f32")?;
            let mut b = self.dev.stream().launch_builder(&f);
            b.arg(&mut out_run).arg(&part).arg(&h).arg(&dh).arg(&nt).arg(&nc).arg(&0i32);
            self.dev
                .profile()
                .time("attn_pv_sum_reduce", self.dev.stream(), || {
                    unsafe { b.launch(elementwise(total)) }.context("attn_pv_sum_reduce")?;
                    Ok(())
                })?;
        }
        Ok(())
    }

    /// Launches `attn_ws_pair_probe_f32`, a standalone isolated validation
    /// of a second (consumer-warp-pair) producer/consumer handoff being
    /// considered for `attn_prefill_mma_ws_f32`'s own occupancy problem.
    /// Its first version passed only because a since-removed debug store
    /// happened to change codegen enough to hide a real missing-`"memory"`-
    /// clobber bug (see the kernel's own comment) -- kept as a permanent
    /// regression test for exactly that pitfall
    /// (`attn_ws_pair_probe_matches_closed_form` in `tests/ops.rs`) rather
    /// than deleted now that it passes cleanly.
    pub fn attn_ws_pair_probe(&self, out: &mut ViewMut<'_, f32>, iters: i32) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_ws_pair_probe_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 16,
        };
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(&mut *out).arg(&iters);
        unsafe { b.launch(cfg) }.context("attn_ws_pair_probe")?;
        Ok(())
    }

    /// Scores, softmax and the weighted sum in one pass over the KV range.
    ///
    /// Replaces `attn_scores` + `attn_softmax` + `attn_output` for the shapes
    /// [`Self::flash_attention`] accepts. The score matrix never reaches HBM,
    /// and a layer's attention costs two launches instead of three — at a batch
    /// of one, where each of those kernels is latency rather than bandwidth,
    /// that is most of the cost.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_flash(
        &self,
        out: &mut ViewMut<'_, f32>,
        q: &View<'_, f32>,
        k_cache: &View<'_, f16>,
        v_cache: &View<'_, f16>,
        batch: BatchLayout<'_>,
        dims: AttnDims,
        kv_len: usize,
        scale: f32,
        partial: &mut ViewMut<'_, f32>,
    ) -> Result<()> {
        let (n_chunks, chunk) = self
            .attn_flash_split(&dims, kv_len)
            .context("attn_flash called for a shape it does not take")?;
        let ms_off = (32 * dims.n_heads * dims.n_tokens * dims.d_head) as i32;
        // Stack four groups of `d_head` threads when they fit: the value loop
        // is the kernel's cost, and one group per block leaves the memory
        // pipeline with too few independent addresses to work on.
        let lanes = (dims.d_head as u32).next_multiple_of(32).clamp(32, 1024);
        let subs = std::env::var("INFERO_ATTN_SUBS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(4)
            .clamp(1, 1024 / lanes.max(1));
        let block = lanes * subs;
        let (stride, h, kh, dh, ms, nc, ck) = (
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            n_chunks as i32,
            chunk as i32,
        );

        // The grouped variant needs one warp per query head in the group and
        // one thread per head dimension, and holds the group's accumulators in
        // a fixed eight registers.
        let group = dims.n_heads / dims.n_kv_heads.max(1);
        // The block only has to be wide enough to hold the group's rows and to
        // divide `d_head / 8` for the value loop; it does *not* have to be
        // exactly `group * 32`. Pinning it there left four warps to score a
        // whole chunk of keys — the phase that the unfused `attn_scores_gqa_f32`
        // spreads over the entire grid — which is what made the fused kernel
        // slower than the three it replaces.
        let gqa = std::env::var("INFERO_ATTN_GQA").is_ok_and(|v| v != "0")
            && group > 1
            && group <= 8
            && block >= group as u32 * 32
            && dims.d_head.is_multiple_of(8)
            && block.is_multiple_of((dims.d_head / 8) as u32);
        let f = self.dev.kernels().get(
            "infero_ops",
            ops_src(),
            if gqa { "attn_flash_gqa_f32" } else { "attn_flash_f32" },
        )?;
        let cfg = LaunchConfig {
            grid_dim: (
                if gqa { dims.n_kv_heads as u32 } else { dims.n_heads as u32 },
                dims.n_tokens as u32,
                n_chunks,
            ),
            block_dim: (block, 1, 1),
            // The scores — one row per query head when grouped — the chunk's
            // slot indices, and the per-group partial sums.
            shared_mem_bytes: if gqa {
                // A weight row per query head, the chunk's slots, and the
                // value reduction — `block * 8` floats, as below.
                chunk * 4 * (group as u32 + 1) + block * 8 * 4
            } else {
                // The value loop gives each thread eight halves, so the block
                // covers `blockDim.x / (d_head / 8)` slices of the key range
                // and the reduction holds that many rows of `d_head` — which
                // is `block * 8` floats however `d_head` divides up.
                chunk * 8 + block * 8 * 4
            },
        };
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(&mut *partial)
            .arg(&ms_off)
            .arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ms)
            .arg(&scale)
            .arg(&ck);
        self.dev
            .profile()
            .time("attn_flash", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_flash")?;
                Ok(())
            })?;

        let r = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_flash_reduce_f32")?;
        let part = partial.as_view();
        let total = (dims.n_tokens * dims.n_heads * dims.d_head) as u32;
        let nt = dims.n_tokens as i32;
        let mut rb = self.dev.stream().launch_builder(&r);
        rb.arg(out)
            .arg(&part)
            .arg(&ms_off)
            .arg(&h)
            .arg(&dh)
            .arg(&nt)
            .arg(&nc);
        self.dev
            .profile()
            .time("attn_flash_reduce", self.dev.stream(), || {
                unsafe { rb.launch(elementwise(total)) }.context("attn_flash_reduce")?;
                Ok(())
            })?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attn_output(
        &self,
        out: &mut ViewMut<'_, f32>,
        scores: &View<'_, f32>,
        v_cache: &View<'_, f16>,
        batch: BatchLayout<'_>,
        dims: AttnDims,
        kv_len: usize,
        partial: Option<&mut ViewMut<'_, f32>>,
    ) -> Result<()> {
        let (stride, h, kh, dh, ms, kl) = (
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            kv_len as i32,
        );
        let block = (dims.d_head as u32).next_multiple_of(32).min(1024);
        let (n_chunks, chunk) = match partial {
            Some(_) => self.attn_chunks(&dims, kv_len),
            None => (0, 0),
        };

        if n_chunks >= 2 {
            let part = partial.unwrap();
            // Read each V row once for the whole query group when the group is
            // real and small enough to hold running sums for. The split is
            // what makes this affordable: on its own the grouped kernel is a
            // quarter of the grid, which is why the unsplit one below is off.
            let group = dims.n_heads / dims.n_kv_heads.max(1);
            let gqa = cfg!(feature = "cuda") && group > 1 && group <= 8;
            let f = self.dev.kernels().get(
                "infero_ops",
                ops_src(),
                if gqa {
                    "attn_output_gqa_split_f32"
                } else {
                    "attn_output_split_f32"
                },
            )?;
            let cfg = LaunchConfig {
                grid_dim: (
                    if gqa {
                        dims.n_kv_heads as u32
                    } else {
                        dims.n_heads as u32
                    },
                    dims.n_tokens as u32,
                    n_chunks,
                ),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            let (nc, ck, gi) = (n_chunks as i32, chunk as i32, group as i32);
            let mut b = self.dev.stream().launch_builder(&f);
            b.arg(&mut *part)
                .arg(scores)
                .arg(v_cache)
                .arg(batch.seq_of)
                .arg(batch.positions)
                .arg(batch.slot_table)
                .arg(&stride)
                .arg(&h)
                .arg(&kh)
                .arg(&dh)
                .arg(&ms)
                .arg(&kl)
                .arg(&nc)
                .arg(&ck);
            if gqa {
                b.arg(&gi);
            }
            self.dev
                .profile()
                .time("attn_output", self.dev.stream(), || {
                    unsafe { b.launch(cfg) }.context("attn_output_split")?;
                    Ok(())
                })?;

            let part_view = part.as_view();
            let r = self
                .dev
                .kernels()
                .get("infero_ops", ops_src(), "attn_output_reduce_f32")?;
            let total = (dims.n_tokens * dims.n_heads * dims.d_head) as u32;
            let (nt, ncr) = (dims.n_tokens as i32, n_chunks as i32);
            let mut rb = self.dev.stream().launch_builder(&r);
            rb.arg(out)
                .arg(&part_view)
                .arg(&h)
                .arg(&dh)
                .arg(&nt)
                .arg(&ncr);
            self.dev
                .profile()
                .time("attn_reduce", self.dev.stream(), || {
                    unsafe { rb.launch(elementwise(total)) }.context("attn_reduce")?;
                    Ok(())
                })?;
            return Ok(());
        }

        // Grouping the query heads that share a V row is a loss here, unlike on
        // the score side. It cuts V traffic fourfold and the grid with it — from
        // 1024 blocks to 256 at a batch of 32 — and the parallelism is worth
        // more: 68.0 ms per step against 64.7. The score kernel keeps its `j`
        // dimension when it groups, so it does not pay that.
        let gqa = false;
        let group_i = 0i32;
        // Eight halves per thread and sixteen key slices per block, which is a
        // load width and a count of loads in flight rather than a different
        // sum. `INFERO_ATTN_V1` puts the two-byte version back for A/B.
        // `attn_output_v4_f32` -- eight halves a thread, sixteen key slices a
        // block -- and the GQA variant are CUDA-only for now; false takes
        // `attn_output_f32`, which is ported.
        let wide = cfg!(feature = "cuda")
            && dims.d_head.is_multiple_of(8)
            && dims.d_head >= 8
            && std::env::var_os("INFERO_ATTN_V1").is_none();
        let (lanes, slices) = if wide {
            let l = dims.d_head / 8;
            // A power of two, so the reduction across slices is a clean tree.
            (l as u32, (256 / l.max(1)).max(1).next_power_of_two() as u32)
        } else {
            (0, 0)
        };
        let f = self.dev.kernels().get(
            "infero_ops",
            ops_src(),
            match (wide, gqa) {
                (true, _) => "attn_output_v4_f32",
                (false, true) => "attn_output_gqa_f32",
                (false, false) => "attn_output_f32",
            },
        )?;
        let cfg = LaunchConfig {
            grid_dim: (
                if gqa { dims.n_kv_heads as u32 } else { dims.n_heads as u32 },
                dims.n_tokens as u32,
                1,
            ),
            block_dim: (if wide { lanes * slices } else { block }, 1, 1),
            shared_mem_bytes: if wide {
                slices * dims.d_head as u32 * 4
            } else {
                0
            },
        };
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out)
            .arg(scores)
            .arg(v_cache)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ms)
            .arg(&kl);
        if gqa {
            b.arg(&group_i);
        }
        self.dev
            .profile()
            .time("attn_output", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("attn_output")?;
                Ok(())
            })?;
        Ok(())
    }

    // ---- weights --------------------------------------------------------

    /// `out[t, :] = dequant(w[rows[t], :])`, the embedding lookup.
    pub fn gather_rows(
        &self,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        ty: WeightType,
        rows: &View<'_, i32>,
        n_tokens: usize,
        k: usize,
    ) -> Result<()> {
        // Q4_K only, on Metal: `embed_row_q4_K` unpacks a 32-element group's
        // `q4k_scale_min` once and spends it on the whole group, the same fix
        // `dequant_q4_K_f16_vec` made for the whole-matrix dequant -- the
        // generic `gather_rows_q4_K` below re-reads and re-unpacks it fresh
        // for every element, since it is one instantiation of a macro shared
        // with every other weight type and has no per-type specialisation.
        // Every token this engine embeds takes this path: `token_embd.weight`
        // is Q4_K on a Q4_K_M checkpoint, and every step -- prefill or decode
        // -- starts by gathering one row a token. Measured byte-exact against
        // `gather_rows_q4_K` at the real embedding shape (k = 5120,
        // vocab = 248320), 1.0-2.9x depending on token count, never a
        // regression (`examples/embed_row_check.rs`).
        if !cfg!(feature = "cuda") && ty == WeightType::Q4K {
            let f = self.dev.kernels().get("infero_quant", quant_src(), "embed_row_q4_K")?;
            let cfg = LaunchConfig {
                grid_dim: (1, n_tokens as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            let k_i = k as i32;
            let mut b = self.dev.stream().launch_builder(&f);
            b.arg(out).arg(w).arg(rows).arg(&k_i);
            return self.dev.profile().time("gather_rows", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("gather_rows")?;
                Ok(())
            });
        }
        let name = format!("gather_rows_{}", ty.suffix());
        let f = self.dev.kernels().get("infero_quant", quant_src(), &name)?;
        let cfg = LaunchConfig {
            grid_dim: (
                (k as u32).div_ceil(ELEMENTWISE_BLOCK).max(1),
                n_tokens as u32,
                1,
            ),
            block_dim: (ELEMENTWISE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let k_i = k as i32;
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(w).arg(rows).arg(&k_i);
        self.dev
            .profile()
            .time("gather_rows", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("gather_rows")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Decode a whole weight matrix into f16, for the cuBLAS prefill path.
    pub fn dequant_to_f16(
        &self,
        out: &mut ViewMut<'_, f16>,
        w: &View<'_, u8>,
        ty: WeightType,
        n_elements: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            out.len() >= n_elements,
            "dequant scratch holds {} elements, need {n_elements}",
            out.len()
        );
        // Metal only: one thread a 32-element group instead of one a element,
        // so the group's `q4k_scale_min` unpack happens once instead of
        // thirty-two times and the writes go out as `half4`s. Measured on an
        // M4 Max (`examples/dequant_q4k_vec_check.rs`): identical output,
        // 5.6-5.9x faster, 40 GB/s to 235 -- this kernel was compute-bound on
        // redundant scalar work, not the memory traffic it looks like on
        // paper. `dequant_to_f16` is ~half of a prefill call's cost (measured
        // on a 53-token prompt, `INFERO_METAL_PROFILE=1`), so this is not a
        // rounding error on that number.
        // `Q6_K` earns a smaller multiple: `dequant_q6_K_f16_vec` gives one
        // thread the same four elements `GEMV_BODY_Q6_K` already gives one
        // thread (see its doc comment for why four rather than a clean
        // thirty-two -- a Q6_K block has no single contiguous group the way
        // Q4_K and Q8_0 do), so its group size is four, not thirty-two.
        let vec_group: Option<u64> = if cfg!(feature = "cuda") {
            None
        } else {
            match ty {
                WeightType::Q4K if n_elements.is_multiple_of(32) => Some(32),
                WeightType::Q8_0 if n_elements.is_multiple_of(32) => Some(32),
                WeightType::Q6K if n_elements.is_multiple_of(256) => Some(4),
                _ => None,
            }
        };
        let name = match (ty, vec_group.is_some()) {
            (WeightType::Q4K, true) => "dequant_q4_K_f16_vec".to_string(),
            (WeightType::Q8_0, true) => "dequant_q8_0_f16_vec".to_string(),
            (WeightType::Q6K, true) => "dequant_q6_K_f16_vec".to_string(),
            _ => format!("dequant_{}_f16", ty.suffix()),
        };
        let f = self.dev.kernels().get("infero_quant", quant_src(), &name)?;
        let work_items = match vec_group {
            Some(g) => (n_elements as u64) / g,
            None => n_elements as u64,
        };
        let cfg = LaunchConfig {
            grid_dim: (
                work_items.div_ceil(ELEMENTWISE_BLOCK as u64) as u32,
                1,
                1,
            ),
            block_dim: (ELEMENTWISE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let n = n_elements as u64;
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(w).arg(&n);
        self.dev
            .profile()
            .time("dequant_to_f16", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("dequant_to_f16")?;
                Ok(())
            })?;
        Ok(())
    }

    /// The descriptor for a Q4_G128T quant plane: `n` rows of `row_bytes`,
    /// tiled 128 bytes by `rows` and swizzled so the fragment read is
    /// conflict-free — see `the_128b_swizzle_is_undone_by_xor_with_the_row`.
    ///
    /// Cached: the same matrix is asked for every step, and a graph capture must
    /// not build one (it is a host call, and its result is baked into the
    /// launch as a by-value argument).
    #[cfg(feature = "cuda")]
    fn tma_desc(&self, ptr: u64, n: usize, row_bytes: usize, rows: usize) -> Result<TmaDesc> {
        // `rows` is the box height and belongs in the key: two variants with
        // different row groups over the same matrix need different descriptors.
        let key = (ptr, n, row_bytes * 1024 + rows);
        if let Some(d) = self.tma.lock().unwrap().get(&key) {
            return Ok(*d);
        }
        use cudarc::driver::sys;
        let mut d = TmaDesc([0u8; 128]);
        // u32 elements, so a 128-byte box is 32 of them — `boxDim` is capped at
        // 256 elements a dimension and the innermost box has to be a multiple of
        // 16 bytes.
        let global_dim = [(row_bytes / 4) as u64, n as u64];
        let global_strides = [row_bytes as u64];
        let box_dim = [32u32, rows as u32];
        let elem_strides = [1u32, 1];
        let r = unsafe {
            sys::cuTensorMapEncodeTiled(
                (&mut d as *mut TmaDesc).cast(),
                sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
                2,
                ptr as *mut std::ffi::c_void,
                global_dim.as_ptr(),
                global_strides.as_ptr(),
                box_dim.as_ptr(),
                elem_strides.as_ptr(),
                sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            )
        };
        anyhow::ensure!(
            r == sys::CUresult::CUDA_SUCCESS,
            "cuTensorMapEncodeTiled for a {n}x{row_bytes} plane: {r:?}"
        );
        self.tma.lock().unwrap().insert(key, d);
        Ok(d)
    }

    /// Bytes a `k`-element row occupies once quantized to Q8_1.
    pub fn q8_1_bytes(k: usize) -> usize {
        k.div_ceil(32) * Q8_1_BLOCK_BYTES
    }

    /// Whether the integer mat-vec has a dot product for this encoding.
    ///
    /// The rest still go through the float path; adding one is a matter of
    /// porting its `vec_dot_*_q8_1` from llama.cpp. The split Q8_0 layout is
    /// absent on purpose: it exists for the batched vocab projection, and a
    /// single row never reaches it.
    pub fn has_mmvq(ty: WeightType) -> bool {
        // The integer mat-vec is CUDA-only for now. It rests on `__dp4a` --
        // four int8 products retired in one instruction -- and the sensible
        // Metal equivalent is a different formulation rather than a
        // transliteration, so `mmvq.cu`'s dot products are not ported.
        //
        // Saying false here is not a stub: `matmul` reads exactly this before
        // it dispatches, and answers with the float `gemv` family, which is
        // ported and correct. It decodes each weight to f32 instead of
        // retiring four at a time, so a quantized decode step costs more
        // instructions for the same bytes -- the first thing to fix when this
        // backend's mat-vecs are worth optimising.
        if !cfg!(feature = "cuda") {
            return false;
        }
        matches!(
            ty,
            WeightType::Q8_0
                | WeightType::Q4K
                | WeightType::Q6K
                | WeightType::Q4G128
                | WeightType::Q4G128T
        )
    }

    /// Quantize one activation row to Q8_1, the form the integer mat-vec wants.
    pub fn quantize_q8_1(
        &self,
        out: &mut ViewMut<'_, u8>,
        x: &View<'_, f32>,
        k: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            k.is_multiple_of(32),
            "Q8_1 needs a multiple of 32 elements, got {k}"
        );
        anyhow::ensure!(
            out.len() >= Self::q8_1_bytes(k),
            "q8_1 buffer holds {} bytes, need {}",
            out.len(),
            Self::q8_1_bytes(k)
        );
        let f = self
            .dev
            .kernels()
            .get("infero_mmvq", mmvq_src(), "quantize_q8_1_f32")?;
        let k_i = k as i32;
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(x).arg(&k_i);
        self.dev
            .profile()
            .time("quantize_q8_1", self.dev.stream(), || {
                unsafe { b.launch(elementwise(k as u32)) }.context("quantize_q8_1")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Tokens one multi-token mat-vec block serves. Chosen per call.
    ///
    /// Rounded up to the smallest template that covers `n_tokens` in one
    /// block, not down to the largest one under it: `mmvqt{T}`'s grid is
    /// `(n, ceil(n_tokens / T), 1)`, so picking a `T` below `n_tokens` (the
    /// previous rule, e.g. `T=4` for five tokens) buys a second row of blocks
    /// that re-streams the *entire* weight matrix a second time to cover one
    /// leftover token. That second pass is exactly the cost this kernel
    /// exists to avoid -- a speculative verify pass at `k=4` (five rows)
    /// measured 88.33 ms with `T=4` against 55.74 ms once dispatch got as far
    /// as this kernel at all, and `T=8` (one block, three lanes idle rather
    /// than a second launch) took that to correctness parity with `k=3`'s
    /// four rows, which already landed on a one-block `T=4`. Idle lanes in a
    /// bandwidth-bound kernel cost nothing; a second weight stream costs
    /// everything.
    fn mmvq_t(n_tokens: usize) -> u32 {
        match n_tokens {
            0..=1 => 1,
            2 => 2,
            3..=4 => 4,
            5..=8 => 8,
            _ => 16,
        }
    }

    /// `out[t, r] = dot(w[r, :], x[t, :])`, streaming each weight once and
    /// spending it on `T` tokens.
    ///
    /// The single-token mat-vec reaches 93% of this card's streaming-read
    /// ceiling because it never stages weights through shared memory. This
    /// keeps that property and amortizes the read across a batch, which is the
    /// thing the tensor-core GEMM was supposed to do and does at a quarter of
    /// the rate.
    #[allow(clippy::too_many_arguments)]
    pub fn mmvq_batch(
        &self,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        ty: WeightType,
        x_q8_1: &View<'_, u8>,
        k: usize,
        n: usize,
        n_tokens: usize,
    ) -> Result<()> {
        anyhow::ensure!(Self::has_mmvq(ty), "no integer mat-vec for {ty}");
        let t = Self::mmvq_t(n_tokens);
        let name = format!("mmvqt{t}_{}", ty.suffix());
        let f = self.dev.kernels().get("infero_mmvq", mmvq_src(), &name)?;
        let slices = match ty {
            WeightType::Q8_0 => k / 8,
            WeightType::Q4K => k / 16,
            WeightType::Q6K => k / 8,
            // One 32-weight quarter of a group per thread.
            WeightType::Q4G128 | WeightType::Q4G128T => k / 32,
            _ => unreachable!("guarded above"),
        };
        let block = (slices as u32).next_multiple_of(32).clamp(32, REDUCE_BLOCK);
        let cfg = LaunchConfig {
            grid_dim: (n as u32, (n_tokens as u32).div_ceil(t), 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (k_i, n_i, t_i) = (k as i32, n as i32, n_tokens as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(w).arg(x_q8_1).arg(&k_i).arg(&n_i).arg(&t_i);
        self.dev.profile().time("mmvq_batch", self.dev.stream(), || {
            unsafe { b.launch(cfg) }.context("mmvq_batch")?;
            Ok(())
        })?;
        Ok(())
    }

    /// Whether an expert block's encoding has a MoE mat-vec.
    pub fn has_mmvq_moe(ty: WeightType) -> bool {
        matches!(
            ty,
            WeightType::Q4G128 | WeightType::Q4G128T | WeightType::Q8_0
        )
    }

    /// Softmax, top-k and the combine weights, one block per token.
    ///
    /// `ids` and `weights` come out `[n_tokens, k]`, in descending order of
    /// router logit.
    ///
    /// CUDA-only, matching `moe_src`'s own note: MoE is AWQ-first and has no
    /// Metal kernel yet.
    #[cfg(feature = "cuda")]
    pub fn moe_topk(
        &self,
        ids: &mut ViewMut<'_, i32>,
        weights: &mut ViewMut<'_, f32>,
        logits: &View<'_, f32>,
        n_experts: usize,
        k: usize,
        n_tokens: usize,
        norm_topk_prob: bool,
    ) -> Result<()> {
        anyhow::ensure!(
            k > 0 && k <= n_experts,
            "routing to {k} of {n_experts} experts"
        );
        let f = self
            .dev
            .kernels()
            .get("infero_moe", moe_src(), "moe_topk_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (n_tokens as u32, 1, 1),
            block_dim: (per_vector_block(n_experts), 1, 1),
            shared_mem_bytes: (n_experts * 4) as u32,
        };
        let (ne, k_i, norm) = (n_experts as i32, k as i32, norm_topk_prob as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(ids).arg(weights).arg(logits).arg(&ne).arg(&k_i).arg(&norm);
        self.dev.profile().time("moe_topk", self.dev.stream(), || {
            unsafe { b.launch(cfg) }.context("moe_topk")?;
            Ok(())
        })?;
        Ok(())
    }

    /// The integer mat-vec against every (token, active expert) pair.
    ///
    /// `w_all` is the whole concatenated block and `stride` the bytes between
    /// experts; `expert_ids` is one entry per slot and `out` comes out
    /// `[n_slots, n]`. One launch, not one per slot — at top-8 and 48 layers a
    /// loop would be 1152 launches a token.
    ///
    /// `y_group` is how many consecutive slots share an activation row:
    /// `n_active` for `gate` and `up`, which read the token's residual, and 1
    /// for `down`, which reads each slot's own SwiGLU product. This is what
    /// makes one launch serve decode and prefill alike.
    /// CUDA-only; see `moe_topk`.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn mmvq_moe(
        &self,
        out: &mut ViewMut<'_, f32>,
        w_all: &View<'_, u8>,
        ty: WeightType,
        expert_ids: &View<'_, i32>,
        x_q8_1: &View<'_, u8>,
        k: usize,
        n: usize,
        n_slots: usize,
        stride: usize,
        y_group: usize,
    ) -> Result<()> {
        anyhow::ensure!(Self::has_mmvq_moe(ty), "no MoE mat-vec for {ty}");
        anyhow::ensure!(
            y_group > 0 && n_slots.is_multiple_of(y_group),
            "{n_slots} slots do not group into activation rows of {y_group}"
        );
        let name = format!("mmvq_moe_{}", ty.suffix());
        let f = self.dev.kernels().get("infero_moe", moe_src(), &name)?;
        let slices = match ty {
            WeightType::Q8_0 => k / 8,
            WeightType::Q4G128 | WeightType::Q4G128T => k / 32,
            _ => unreachable!("guarded above"),
        };
        let block = (slices as u32).next_multiple_of(32).clamp(32, REDUCE_BLOCK);
        let cfg = LaunchConfig {
            grid_dim: (n as u32, n_slots as u32, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (k_i, n_i, stride_i, yg) =
            (k as i32, n as i32, stride as i64, y_group as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out)
            .arg(w_all)
            .arg(expert_ids)
            .arg(x_q8_1)
            .arg(&k_i)
            .arg(&n_i)
            .arg(&stride_i)
            .arg(&yg);
        self.dev.profile().time("mmvq_moe", self.dev.stream(), || {
            unsafe { b.launch(cfg) }.context("mmvq_moe")?;
            Ok(())
        })?;
        Ok(())
    }

    /// `out[t] = sum_a weights[t, a] * partials[t, a]`.
    ///
    /// CUDA-only; see `moe_topk`.
    #[cfg(feature = "cuda")]
    pub fn moe_combine(
        &self,
        out: &mut ViewMut<'_, f32>,
        partials: &View<'_, f32>,
        weights: &View<'_, f32>,
        d: usize,
        k: usize,
        n_tokens: usize,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_moe", moe_src(), "moe_combine_f32")?;
        let total = d * n_tokens;
        let (d_i, k_i, total_i) = (d as i32, k as i32, total as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out)
            .arg(partials)
            .arg(weights)
            .arg(&d_i)
            .arg(&k_i)
            .arg(&total_i);
        self.dev.profile().time("moe_combine", self.dev.stream(), || {
            unsafe { b.launch(elementwise(total as u32)) }.context("moe_combine")?;
            Ok(())
        })?;
        Ok(())
    }

    /// RMS norm that also writes its output as f16.
    ///
    /// The counterpart of [`Self::rms_norm_q8_1`] for the f16-operand GEMM,
    /// which is the Q4_G128 default. Saves the separate `to_f16` launch and
    /// its re-read, and stops producing a Q8_1 buffer nothing reads.
    #[allow(clippy::too_many_arguments)]
    /// [`Self::rms_norm_f16`] with the residual add that always precedes it.
    ///
    /// `x` is updated in place: it is the residual stream, and the next
    /// sublayer adds to it. See the kernel's comment for what the fusion saves.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rms_norm_f16(
        &self,
        out: &mut ViewMut<'_, f32>,
        h_out: Option<&mut ViewMut<'_, f16>>,
        x: &mut ViewMut<'_, f32>,
        b: &View<'_, f32>,
        weight: &View<'_, f32>,
        n_tokens: usize,
        d: usize,
        eps: f32,
    ) -> Result<()> {
        anyhow::ensure!(
            rms_fits(d),
            "fused norm needs d <= 1024 * {RMS_REGS}, got {d}"
        );
        let label = if h_out.is_some() { "add_rms_norm_f16" } else { "add_rms_norm" };
        let f = self
            .dev
            .kernels()
            .get("infero_mmvq", mmvq_src(), "add_rms_norm_f16_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (n_tokens as u32, 1, 1),
            block_dim: (rms_block(d), 1, 1),
            shared_mem_bytes: 0,
        };
        let (d_i, eps_f) = (d as i32, eps);
        let mut bl = self.dev.stream().launch_builder(&f);
        match h_out {
            Some(h) => {
                bl.arg(out).arg(h);
            }
            None => {
                bl.arg(out).arg(&infero_gpu::NULL_BUFFER);
            }
        }
        bl.arg(&mut *x).arg(b).arg(weight).arg(&d_i).arg(&eps_f);
        self.dev.profile().time(label, self.dev.stream(), || {
            unsafe { bl.launch(cfg) }.context(label)?;
            Ok(())
        })?;
        Ok(())
    }

    pub fn rms_norm_f16(
        &self,
        out: &mut ViewMut<'_, f32>,
        h_out: Option<&mut ViewMut<'_, f16>>,
        x: &View<'_, f32>,
        weight: &View<'_, f32>,
        n_tokens: usize,
        d: usize,
        eps: f32,
    ) -> Result<()> {
        anyhow::ensure!(
            rms_fits(d),
            "fused norm needs d <= 1024 * {RMS_REGS}, got {d}"
        );
        let label = if h_out.is_some() { "rms_norm_f16" } else { "rms_norm" };
        let f = self
            .dev
            .kernels()
            .get("infero_mmvq", mmvq_src(), "rms_norm_f16_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (n_tokens as u32, 1, 1),
            block_dim: (rms_block(d), 1, 1),
            shared_mem_bytes: 0,
        };
        let (d_i, eps_f) = (d as i32, eps);
        b_args(self, &f, cfg, out, h_out, x, weight, d_i, eps_f, label)
    }

    /// RMS norm that also writes its output's Q8_1 form.
    ///
    /// Saves the separate `quantize_q8_1` launch and its re-read of the
    /// normalized vector. `d` must be a multiple of 32, which every model
    /// dimension is.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_q8_1(
        &self,
        out: &mut ViewMut<'_, f32>,
        q_out: &mut ViewMut<'_, u8>,
        x: &View<'_, f32>,
        weight: &View<'_, f32>,
        n_tokens: usize,
        d: usize,
        eps: f32,
    ) -> Result<()> {
        anyhow::ensure!(d.is_multiple_of(32), "fused norm needs d % 32 == 0");
        // The kernel holds its row in `RMS_REGS` registers per thread, so the
        // block has to be wide enough to cover `d` — and no wider, or the tail
        // threads sit idle through the reduction.
        let block = (d as u32).div_ceil(RMS_REGS).next_multiple_of(32).clamp(32, 1024);
        anyhow::ensure!(
            (block as usize) * (RMS_REGS as usize) >= d,
            "fused norm needs d <= 1024 * {RMS_REGS}, got {d}"
        );
        let f = self
            .dev
            .kernels()
            .get("infero_mmvq", mmvq_src(), "rms_norm_q8_1_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (n_tokens as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (d_i, eps_f) = (d as i32, eps);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(q_out).arg(x).arg(weight).arg(&d_i).arg(&eps_f);
        self.dev
            .profile()
            .time("rms_norm_q8_1", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("rms_norm_q8_1")?;
                Ok(())
            })?;
        Ok(())
    }

    /// `out[r] = dot(w[r, :], x)` with both sides held as integers.
    ///
    /// `x_q8_1` must be the activation row already through
    /// [`Kernels::quantize_q8_1`]. Single row only — above that the answer is a
    /// tensor-core GEMM rather than a wider mat-vec.
    /// Which weight types have a tensor-core GEMM.
    ///
    /// The K-quants plus Q8_0. The legacy block-32 types are absent because a
    /// Q4_K_M build only falls back to them for awkward row lengths, which the
    /// shape check rejects anyway.
    pub fn has_mmq(ty: WeightType) -> bool {
        matches!(
            ty,
            WeightType::Q8_0
                | WeightType::Q8_0S
                | WeightType::Q4K
                | WeightType::Q6K
                | WeightType::Q4G128
                | WeightType::Q4G128T
        )
    }

    /// `out[t, r] = dot(w[r, :], x[t, :])` on the integer tensor cores.
    ///
    /// Needs `x` pre-quantized by [`Self::quantize_q8_1`] and an Ampere or
    /// newer device. Reads each weight once for a whole 16-token tile, which is
    /// what separates it from [`Self::mmvq`].
    #[allow(clippy::too_many_arguments)]
    pub fn mmq(
        &self,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        ty: WeightType,
        x_q8_1: &View<'_, u8>,
        k: usize,
        n: usize,
        n_tokens: usize,
    ) -> Result<()> {
        anyhow::ensure!(Self::has_mmq(ty), "no tensor-core gemm for {ty}");
        anyhow::ensure!(
            self.dev.caps().int_tensor_gemm,
            "the tensor-core integer gemm has no implementation on the {} backend",
            infero_gpu::BACKEND
        );
        anyhow::ensure!(k.is_multiple_of(32), "mmq needs k divisible by 32, got {k}");
        if matches!(ty, WeightType::Q4K | WeightType::Q6K) {
            anyhow::ensure!(
                k.is_multiple_of(256),
                "mmq {ty} needs k divisible by 256, got {k}"
            );
        }
        // The direct-B pipeline reads its weight fragments straight from
        // global instead of expanding every nibble into a shared int8 tile;
        // see `mmq.cu`. Only Q4_G128 has one, whose layout puts a fragment in
        // a single contiguous sector.
        if ty == WeightType::Q4G128T {
            // One kernel reads this layout, and it is the one the layout was
            // made for. `INFERO_MMQ_VARIANT` still selects a shape.
            let v = match std::env::var("INFERO_MMQ_VARIANT") {
                Ok(s)
                    if s.starts_with("mmqz")
                        || s.starts_with("mmqy")
                        || s.starts_with("mmqc") =>
                {
                    Box::leak(s.into_boxed_str())
                }
                _ => "mmqy1w8s2",
            };
            return self.mmq_variant(v, out, w, ty, x_q8_1, k, n, n_tokens);
        }
        let variant = if ty == WeightType::Q8_0S {
            // The default `mmq` name, which resolves the shape and the launch
            // from one place — `mmq_kernel_name` and `mmq_warps` — and so cannot
            // disagree with itself. Naming a shape explicitly here can: two
            // separate mismatches were found and fixed today (the warp count,
            // then `per_block`), and until both were fixed every explicitly
            // named plain shape was timed with the wrong grid. With them fixed,
            // `mmq` and `mmqw8_2` measure the same thing on the vocab
            // projection — 478 us against 468, 1167 GB/s against 1192 — because
            // they *are* the same kernel, and no other instantiated shape comes
            // close (`mmqw8` alone 809, `mmqw2` 852, `mmqw1` 1247).
            // Only the plain family's spellings: `mmq`, `mmq<tiles>` and
            // `mmqw<warps>[_<tiles>]`. An `mmqy`-style name pinned for the
            // layer matmuls has no instantiation here, and accepting it landed
            // as `mmqy1w8s2_q8_0s not found` at the first decode step.
            match std::env::var("INFERO_MMQ_VARIANT") {
                Ok(v)
                    if v == "mmq"
                        || v.strip_prefix("mmq").is_some_and(|r| {
                            r.chars().next().is_some_and(|c| c.is_ascii_digit())
                        })
                        || v.starts_with("mmqw") =>
                {
                    Box::leak(v.into_boxed_str())
                }
                _ => "mmq",
            }
        } else if ty != WeightType::Q4G128 {
            "mmq"
        } else {
            // `mmqp` is the epilogue probe: right pipeline, wrong answer. It
            // prices the per-group scale application and nothing else.
            match std::env::var("INFERO_MMQ_VARIANT").as_deref() {
                Ok("staged") => "mmq",
                Ok("probe") => "mmqp",
                Ok("direct") => "mmqd",
                Ok("striped") => "mmqs",
                // `mmqx<nblk>w<warps>` picks a wide-tile shape by name, and
                // `mmqa<nblk>w<warps>s<stages>` the same tile behind a
                // `cp.async` ring buffer.
                Ok(v)
                    if v.starts_with("mmqx")
                        || v.starts_with("mmqa")
                        || v.starts_with("mmqr")
                        || v.starts_with("mmqsr")
                        || v.starts_with("mmqb")
                        || v.starts_with("mmql") =>
                {
                    Box::leak(v.to_string().into_boxed_str())
                }
                _ => "mmqd",
            }
        };
        self.mmq_variant(variant, out, w, ty, x_q8_1, k, n, n_tokens)
    }

    /// [`Self::mmq`] with the kernel name spelled out, for the attribution
    /// probes in `mmq.cu`. `variant` is `"mmq"`, `"mmq_stage_only"` or
    /// `"mmq_mma_only"`; the last two compute nothing usable.
    #[allow(clippy::too_many_arguments)]
    pub fn mmq_variant(
        &self,
        variant: &str,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        ty: WeightType,
        x_q8_1: &View<'_, u8>,
        k: usize,
        n: usize,
        n_tokens: usize,
    ) -> Result<()> {
        // Four token tiles exist only for the f16 shapes; every other family is
        // instantiated at one and two, and the Q8_0 vocab projection goes
        // through one of them.
        let tiles = if (variant.starts_with("mmqf") && !variant.starts_with("mmqfp"))
            || variant.starts_with("mmqg")
            || variant.starts_with("mmqz")
            || variant.starts_with("mmqy")
            || variant.starts_with("mmqk")
            || variant.starts_with("mmqc")
            || variant.starts_with("mmqt")
            || variant.starts_with("mmqe")
        {
            Self::mmq_tiles(n_tokens)
        } else {
            Self::mmq_tiles(n_tokens).min(2)
        };
        // The plain integer family names its shape too, in two spellings the
        // instantiations grew into: `mmq<tiles>` (four warps) and
        // `mmqw<warps>[_<tiles>]`. The name is the only place that shape
        // appears, so the launch has to read it. Without this the grid was
        // built for four warps and whatever `mmq_tiles` said while the kernel
        // had been compiled for eight and one — which is how `mmqw8` came to
        // measure 213 GB/s of wrong answers, and why the one- and two-warp
        // shapes refused to launch at all (128 threads into
        // `__launch_bounds__(32)`).
        let plain = variant.strip_prefix("mmqw").and_then(|rest| {
            let (w, t) = match rest.split_once('_') {
                Some((w, t)) => (w, t.parse::<u32>().ok()?),
                None => (rest, 1),
            };
            Some((w.parse::<u32>().ok()?, t))
        });
        let plain = plain.or_else(|| match variant {
            "mmq2" => Some((MMQ_MAX_ROWS / 8, 2)),
            _ => None,
        });
        let tiles = plain.map_or(tiles, |(_, t)| t);
        // A wide-tile name carries its own shape: `<nblk>w<warps>`, and the
        // `cp.async` variant appends `s<stages>`, which only the kernel needs.
        // `mmqsr` before `mmqs`, which is an exact name rather than a prefix.
        let wide_named = ["mmqsr", "mmqx", "mmqa", "mmqr", "mmqfp", "mmqf", "mmqg", "mmqzg", "mmqz", "mmqy", "mmqk", "mmqc", "mmqt", "mmqm", "mmqnm", "mmqnx", "mmqnh", "mmqnr", "mmqe", "mmqb", "mmql", "mmqna", "mmqne"]
            .iter()
            .find_map(|p| variant.strip_prefix(p).map(|shape| (*p, shape)));
        let wide_shape = wide_named.map(|(_, shape)| shape);
        let wide = wide_shape.map(|shape| {
            let (nblk, rest) = shape
                .split_once('w')
                .expect("wide-tile name needs <nblk>w<warps>");
            let warps = rest.split_once('s').map_or(rest, |(w, _)| w);
            (
                nblk.parse::<u32>().expect("nblk"),
                warps.parse::<u32>().expect("warps"),
            )
        });
        if variant.starts_with("mmqe")
            || variant.starts_with("mmqa")
            || variant.starts_with("mmqr")
            || variant.starts_with("mmqsr")
            || variant.starts_with("mmqb")
            || variant.starts_with("mmql")
            || variant.starts_with("mmqn")
        {
            // A tile row is 288 bytes of `block_q8_1` and `cp.async.cg` copies
            // 16 at a time, so the row stride `(k / 32) * 36` has to be a
            // multiple of 16 for the source addresses to be aligned at all.
            anyhow::ensure!(
                k.is_multiple_of(128),
                "{variant} needs k divisible by 128, got {k}"
            );
        }
        let warps = match (variant, wide) {
            (_, Some((_, w))) => w,
            (_, None) if plain.is_some() => plain.unwrap().0,
            ("mmq", _) => self.mmq_warps(n, n_tokens),
            _ => MMQ_MAX_ROWS / 8,
        };
        let rows_per_block = wide.map_or(warps * 8, |(nblk, w)| nblk * w * 8);
        let name = if variant == "mmq" {
            self.mmq_kernel_name(ty, n, n_tokens)
        } else if let Some((prefix, shape)) = wide_named {
            // The wide-tile kernels are named by their own shape rather than
            // derived from `mmq_kernel_name`: the rows a block covers is
            // warps * nblk * 8, and the token tiles are part of the name too,
            // so the launch grid and the kernel agree on both.
            let t = match tiles {
                4 => "_4",
                2 => "_2",
                _ => "",
            };
            // The f16 families are instantiated once, under the packed type's
            // suffix; which layout they read is the kernel's business, not the
            // name's.
            let suffix = if ty == WeightType::Q4G128T {
                "q4_g128"
            } else {
                ty.suffix()
            };
            format!("{prefix}{shape}{t}_{suffix}")
        } else if variant == "mmqs" {
            format!("mmqs{}", self.mmq_kernel_name(ty, n, n_tokens).trim_start_matches("mmq"))
        } else if variant == "mmqp" {
            format!("mmqp{}", self.mmq_kernel_name(ty, n, n_tokens).trim_start_matches("mmq"))
        } else if variant == "mmqd" {
            format!("mmqd{}", self.mmq_kernel_name(ty, n, n_tokens).trim_start_matches("mmq"))
        } else {
            format!("{variant}_{}", ty.suffix())
        };
        if variant == "mmq2r" {
            let f = self
                .dev
                .kernels()
                .get("infero_mmq", mmq_src(), "mmq2r_q4_K")?;
            let cfg = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(128), (n_tokens as u32).div_ceil(16), 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let (k_i, n_i, t_i) = (k as i32, n as i32, n_tokens as i32);
            let mut b = self.dev.stream().launch_builder(&f);
            b.arg(out).arg(w).arg(x_q8_1).arg(&k_i).arg(&n_i).arg(&t_i);
            self.dev.profile().time("mmq2r", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("mmq2r")?;
                Ok(())
            })?;
            return Ok(());
        }

        if variant == "mmq_readonly" {
            let f = self
                .dev
                .kernels()
                .get("infero_mmq", mmq_src(), "mmq_readonly_q4_K")?;
            let cfg = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(64), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let (k_i, n_i, t_i) = (k as i32, n as i32, n_tokens as i32);
            let mut b = self.dev.stream().launch_builder(&f);
            b.arg(out).arg(w).arg(x_q8_1).arg(&k_i).arg(&n_i).arg(&t_i);
            unsafe { b.launch(cfg) }.context("mmq_readonly")?;
            return Ok(());
        }

        // The weights-in-A variant has a fixed 64-row, 32-token tile.
        if variant == "mmqw" {
            let f = self
                .dev
                .kernels()
                .get("infero_mmq", mmq_src(), &format!("mmqw_{}", ty.suffix()))?;
            let cfg = LaunchConfig {
                grid_dim: (
                    (n as u32).div_ceil(64),
                    (n_tokens as u32).div_ceil(32),
                    1,
                ),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let (k_i, n_i, t_i) = (k as i32, n as i32, n_tokens as i32);
            let mut b = self.dev.stream().launch_builder(&f);
            b.arg(out).arg(w).arg(x_q8_1).arg(&k_i).arg(&n_i).arg(&t_i);
            self.dev.profile().time("mmqw", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("mmqw")?;
                Ok(())
            })?;
            return Ok(());
        }
        // Token rows a block covers, which sizes `grid.y`. The plain family's
        // explicitly named shapes carry their tile count in the name — parsed
        // into `plain` above — and leaving them at `MMQ_M` launched twice the
        // blocks for a two-tile kernel, half of them past `n_tokens` and doing
        // nothing. The answer stayed right and the *timing* did not: it is why
        // `mmqw8_2` measures 818 us on the vocab projection against the same
        // kernel's 468 through the `mmq` path, and why any comparison in this
        // file that named a plain shape explicitly was reading a grid twice the
        // size it needed. Same class of mismatch as the warps one above, found
        // the same way — a sweep whose numbers did not fit the shape.
        let per_block = if variant == "mmq" || wide.is_some() || plain.is_some() {
            tiles * MMQ_M
        } else {
            MMQ_M
        };
        // Split k when the row grid alone leaves the device idle. Only the
        // direct Q4_G128 kernel reads `gridDim.z`; everything else launches
        // with one slice and stores rather than accumulates.
        let blocks = (n as u32).div_ceil(rows_per_block) * (n_tokens as u32).div_ceil(per_block);
        let splits = if variant == "mmqd" {
            // Aim well past one block per SM. The row grid alone gives 64
            // blocks for a 4096-row projection — 1.3 per SM — and measured
            // against that, splitting k sixteen ways is worth 18% at eight
            // tokens and 17% at sixteen. Three ways, which is what targeting
            // `sm_count * 4` produced, was worth nothing: the device wants
            // enough concurrent blocks to hide the weight loads, not merely
            // enough to be busy.
            let want = self.dev.sm_count() * 16;
            let tiles_k = (k as u32).div_ceil(256).max(1);
            Self::mmq_splits()
                .unwrap_or_else(|| want.div_ceil(blocks.max(1)).clamp(1, 16))
                .min(tiles_k)
        } else {
            1
        };
        // The striped schedule sizes its grid from the device rather than the
        // matrix and partitions the flattened (row group, k chunk) list across
        // it. Runs that straddle a row-group boundary accumulate, so `out` has
        // to start at zero — but only the boundaries do, which is the whole
        // difference from splitting every slice.
        // The f16 kernels carry the striped partition inside them, and their
        // activation ring is `extern __shared__` — 8.5 KB a stage a token tile,
        // which is past the static cap at four stages and two tiles.
        let f16 = (variant.starts_with("mmqf") && !variant.starts_with("mmqfp"))
            || variant.starts_with("mmqg")
            || variant.starts_with("mmqz")
            || variant.starts_with("mmqy")
            || variant.starts_with("mmqk")
            || variant.starts_with("mmqc")
            || variant.starts_with("mmqt")
            || variant.starts_with("mmqm")
            || variant.starts_with("mmqnm")
            || variant.starts_with("mmqnx")
            || variant.starts_with("mmqnh")
            || variant.starts_with("mmqnr");
        // `mmqe_*` takes Q8_1 activations but stages them in the same
        // `extern __shared__` ring, so it needs the request too — at the
        // narrower stride, which is the point of it.
        let e8 = variant.starts_with("mmqe");
        let dyn_shared = if f16 || e8 {
            // The digits after the first `s`, and only those: a name may carry
            // further suffixes (`...s2x2`) that are not the stage count.
            let stages: u32 = wide_shape
                .and_then(|s| s.split_once('s'))
                .and_then(|(_, st)| {
                    let digits: String = st.chars().take_while(|c| c.is_ascii_digit()).collect();
                    digits.parse().ok()
                })
                .expect("mmqf name needs s<stages>");
            let stride = if variant.starts_with("mmqk") {
                MMQ_XK_STRIDE
            } else if e8 {
                MMQ_XA_STRIDE
            } else if variant.starts_with("mmqm") {
                MMQ_XL_STRIDE
            } else {
                MMQ_XF_STRIDE
            };
            let rows = if variant.starts_with("mmqnr") { tiles * 8 } else { tiles * 16 };
            let rows = if variant.starts_with("mmqg") { 16 } else { rows };
            let act = stages * rows * stride;
            if variant.starts_with("mmqc") {
                // Plus the weight ring: two 128-blocks a k-tile, `mrows` rows
                // of an 80-byte padded stride. See `MMQ_BSH_STRIDE`.
                act + stages * 2 * rows_per_block * 80
            } else if variant.starts_with("mmqt") {
                // The TMA ring lands dense — both 128-blocks of a k-tile in one
                // 128-byte row, swizzled rather than padded — plus one barrier a
                // stage, plus 128 for the alignment the kernel rounds up to,
                // since the shared base is only 16-byte aligned and
                // `cp.async.bulk.tensor` wants 128.
                act + 128 + stages * rows_per_block * 128 + stages * 8
            } else {
                act
            }
        } else {
            0
        };
        let striped =
            variant == "mmqs" || variant.starts_with("mmqsr")
                || variant.starts_with("mmqb")
                || variant.starts_with("mmql")
                || variant.starts_with("mmqn")
                || f16;
        let stripe_blocks = if striped {
            // `mmqs`'s 4 was measured on its own shape. `mmqsr` is the same
            // partition around the register-pipelined loop, and 4 happens to be
            // the *worst* point in its sweep — see the table in `mmq.cu`. The
            // two ends of that sweep want opposite numbers, and the split is
            // the token-tile count: at one tile a block holds half the
            // arithmetic and wants twice the blocks.
            let default_bps = if f16 || e8 {
                // Swept for this kernel rather than inherited: it holds eight
                // warps to `mmql_*`'s four and twice the shared memory a
                // stage, so it wants a shorter grid at both tile counts.
                // Interleaved three ways against the numbers below, the
                // ordering held every round — 334 GB/s against 305 at one
                // token tile, 220 against 211 at two.
                //
                // Two, not four, since the weight prefetch landed. The prefetch
                // gives a block a k-tile of weights in flight while it computes
                // the tile before it, so a block covering more of the flattened
                // partition now amortizes what it used to only lengthen — and
                // the optimum moved with it. A step's matmuls, twenty steps,
                // the same binary, two runs each: **91.15 ms at two against
                // 96.39 at four**, and 94.27 at one. It wins at every batch
                // width — 80.8 against 82.0 at eight tokens, 82.0 against 83.8
                // at sixteen — and the runs are reproducible to 0.01 ms, which
                // is what makes a 5.4% difference readable at all here.
                //
                // The curve is still not monotonic at the low end (three
                // measures 100.1, worse than either side of it) and nothing
                // here explains that; it stays a measurement.
                if tiles == 2 { 2 } else { 24 }
            } else if variant.starts_with("mmqsr")
                || variant.starts_with("mmqb")
                || variant.starts_with("mmql")
                || variant.starts_with("mmqn")
            {
                if tiles == 2 { 8 } else { 48 }
            } else {
                4
            };
            let bps = std::env::var("INFERO_MMQ_BPS")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(default_bps);
            let want = self.dev.sm_count() * bps;
            let n_tiles = (n as u32).div_ceil(rows_per_block);
            let k_tiles = (k as u32).div_ceil(256).max(1);
            let total = n_tiles * k_tiles;
            // One whole row group a block, when there are enough row groups to
            // fill the device with them.
            //
            // The partition's runs straddle row-group boundaries, and a
            // straddling run cannot store — it has to `atomicAdd`, which is a
            // read-modify-write of the output and needs that output zeroed
            // first. Both costs are invisible in a kernel profile: the memsets
            // are 128 a step and 170 MB, and every one of `gate_up`'s 917k
            // outputs is accumulated rather than stored, because `iters` is 10
            // against a `k_tiles` of 16 and so *no* row group comes out whole.
            //
            // Sizing the grid to the row groups instead makes `iters` exactly
            // `k_tiles`, every run whole, every output a store — no atomics and
            // no memset.
            //
            // **And it loses**, by 15%: a step's matmuls take 111.9 ms against
            // 97.3 with `gate_up` down from 752 blocks to 224. The atomics and
            // the memset together are worth less than the block count they buy,
            // which is the same trade every other row-tile experiment on this
            // kernel has made and lost. Off by default; `INFERO_MMQ_ALIGNED=1`
            // re-runs it.
            let aligned = std::env::var("INFERO_MMQ_ALIGNED").is_ok_and(|v| v == "1")
                && n_tiles >= self.dev.sm_count()
                && n_tokens <= per_block as usize;
            // Never hand a block fewer than two of the flattened units. At the
            // old ceiling of `total` every row group is split into `k_tiles`
            // pieces and every output goes through `atomicAdd`, which is what
            // the partition exists to avoid — and on the narrow matrices that
            // is exactly what a device-sized grid asks for. `attn_k` is 16 row
            // groups by 16 k chunks, so `sm_count * 24` clamps to 256 and
            // `iters` lands at 1.
            //
            // Measured in GB/s of weights at eight tokens, this rule against
            // the ceiling it replaces: `attn_k` 161 -> 186, `attn_q` 238 ->
            // 259, `ffn_gate` unchanged at 334 because its grid was never the
            // binding term. Fitting a constant per shape does better still —
            // 187 and 285 — and is overfitting to three matrices.
            // `INFERO_MMQ_BLOCKS` pins the count, for testing whether the
            // partition's *balance* matters rather than its size. At 376 blocks
            // `gate_up`'s 3584 units come out 9.53 to a block — so some get ten
            // and some nine — and 376 blocks over 188 SMs holding 2.9 each is
            // 0.69 of a wave, which exposes that 10% in full instead of
            // averaging it away. A count that divides the units evenly would
            // not. It is also what the non-monotonic bps sweep looks like.
            if let Some(b) = std::env::var("INFERO_MMQ_BLOCKS")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
            {
                b.clamp(1, total.max(1))
            } else if aligned {
                n_tiles
            } else {
                want.clamp(1, (total / 2).max(1))
            }
        } else {
            0
        };
        // A run that covers a whole row group stores; only a straddling one
        // accumulates, and only that needs the output zeroed first.
        let accumulates = splits > 1
            || (striped && stripe_blocks * (k as u32).div_ceil(256).max(1) != {
                let n_tiles = (n as u32).div_ceil(rows_per_block);
                n_tiles * (k as u32).div_ceil(256).max(1)
            });
        if accumulates {
            // The slices accumulate into `out`, so it has to start at zero.
            //
            // `INFERO_MMQ_NO_ZERO=1` skips it and computes the wrong answer on
            // purpose: 130 of a step's ~420 graph nodes are these memsets, and
            // their cost is their bytes *plus* a node transition each. Their
            // execution time is 0.12 ms a step in a trace; this prices what
            // removing them would actually be worth. Same idea as `mmqp`.
            if !std::env::var("INFERO_MMQ_NO_ZERO").is_ok_and(|v| v == "1") {
                self.dev.stream().memset_zeros(out)?;
            }
        }
        let f = self.dev.kernels().get("infero_mmq", mmq_src(), &name)?;
        if dyn_shared > 0 {
            infero_gpu::set_max_dynamic_shared(&f, dyn_shared)?;
        }
        let cfg = LaunchConfig {
            grid_dim: (
                if striped {
                    stripe_blocks
                } else {
                    (n as u32).div_ceil(rows_per_block)
                },
                (n_tokens as u32).div_ceil(per_block),
                splits,
            ),
            block_dim: (warps * 32, 1, 1),
            shared_mem_bytes: dyn_shared,
        };
        let (k_i, n_i, t_i) = (k as i32, n as i32, n_tokens as i32);
        // The TMA family needs a descriptor for the quant plane, by value.
        // Built here because this is the only place that knows the shape; the
        // cache makes it a lookup after the first step, and a capture replays
        // the value it was handed.
        // `mmqt*` needs a `CUtensorMap`, which exists only on CUDA and only
        // from sm_90. A backend without TMA never selects one of those
        // variants -- `mmq_f16_variant_for` gates on the capability -- so this
        // is an unreachable branch on Metal rather than a missing feature.
        #[cfg(feature = "cuda")]
        let desc = if variant.starts_with("mmqt") {
            use cudarc::driver::DevicePtr;
            let (ptr, _sync) = w.device_ptr(self.dev.stream());
            Some(self.tma_desc(ptr, n, (k / 128) * 64, rows_per_block as usize)?)
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let desc: Option<TmaDesc> = {
            anyhow::ensure!(
                !variant.starts_with("mmqt"),
                "{variant} needs a CUtensorMap, which this backend has no equivalent of"
            );
            None
        };
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(w).arg(x_q8_1).arg(&k_i).arg(&n_i).arg(&t_i);
        // Only the `mmqt*` variants take a descriptor, and only CUDA has them.
        #[cfg(feature = "cuda")]
        if let Some(d) = desc.as_ref() {
            b.arg(d);
        }
        #[cfg(not(feature = "cuda"))]
        debug_assert!(desc.is_none());
        self.dev.profile().time("mmq", self.dev.stream(), || {
            unsafe { b.launch(cfg) }.context("mmq")?;
            Ok(())
        })?;
        Ok(())
    }

    /// Blocks an SM the driver will resident, for a given block size and
    /// dynamic shared request. The answer that settles which resource binds.
    #[cfg(feature = "cuda")]
    pub fn occupancy_blocks(
        &self,
        module: &'static str,
        name: &str,
        threads: u32,
        dynamic: usize,
    ) -> Result<u32> {
        let src = match module {
            "infero_mmq" => mmq_src(),
            "infero_mmvq" => mmvq_src(),
            "infero_ops" => ops_src(),
            _ => quant_src(),
        };
        let f = self.dev.kernels().get(module, src, name)?;
        if dynamic > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, dynamic as u32)?;
        }
        Ok(f.occupancy_max_active_blocks_per_multiprocessor(threads, dynamic, None)?)
    }

    /// Registers per thread and static shared bytes per block.
    ///
    /// `kernel_limits` answers whether registers cap the *block size*, which is
    /// a weaker question than it looks: a kernel can use far more registers and
    /// still allow 128 threads while fitting fewer blocks on the SM. This is
    /// the number that settles it — 65536 registers per SM on sm_86 divided by
    /// `regs * block_threads` is the resident block count.
    #[cfg(feature = "cuda")]
    pub fn kernel_registers(&self, module: &'static str, name: &str) -> Result<(i32, i32)> {
        let src = match module {
            "infero_mmq" => mmq_src(),
            "infero_mmvq" => mmvq_src(),
            "infero_ops" => ops_src(),
            "infero_fp8" => fp8_src(),
            _ => quant_src(),
        };
        let f = self.dev.kernels().get(module, src, name)?;
        Ok((f.num_regs()?, f.shared_size_bytes()?))
    }

    /// Walk a weight matrix with the GEMM's own access pattern, at four bytes
    /// a load or at sixteen, and read nothing else.
    ///
    /// `wide` picks the sixteen-byte pattern. See `mmq_bw_probe_w4` in
    /// `mmq.cu`: this exists to price the weight repack before doing it.
    pub fn mmq_bw_probe(
        &self,
        wide: bool,
        sink: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        nb: usize,
        n: usize,
        blocks: u32,
    ) -> Result<()> {
        let name = match wide {
            // `INFERO_MMQ_PROBE=coalesced` reads the same bytes as one 512-byte
            // run a warp instead of eight 64-byte ones; see the kernel.
            true if std::env::var("INFERO_MMQ_PROBE").as_deref() == Ok("coalesced") => {
                "mmq_bw_probe_c16"
            }
            // With the scale read the kernel also pays for, row-major or
            // block-major; see the kernels.
            true if std::env::var("INFERO_MMQ_PROBE").as_deref() == Ok("scales") => {
                "mmq_bw_probe_s16"
            }
            true if std::env::var("INFERO_MMQ_PROBE").as_deref() == Ok("scales_bm") => {
                "mmq_bw_probe_sc16"
            }
            // A write stream beside the read one; see the kernel.
            true if std::env::var("INFERO_MMQ_PROBE").as_deref() == Ok("rw") => {
                "mmq_bw_probe_rw16"
            }
            true => "mmq_bw_probe_w16",
            false => "mmq_bw_probe_w4",
        };
        let f = self.dev.kernels().get("infero_mmq", mmq_src(), name)?;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let (nb_i, n_i) = (nb as i32, n as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(sink).arg(w).arg(&nb_i).arg(&n_i);
        unsafe { b.launch(cfg) }.context(name)?;
        Ok(())
    }

    /// Read `bytes` of device memory and discard it.
    ///
    /// The reference point for every bandwidth claim about the quantized
    /// kernels: it is what this card gives a kernel that does nothing but read.
    pub fn stream_read_probe(
        &self,
        sink: &mut ViewMut<'_, f32>,
        src: &View<'_, u8>,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_mmq", mmq_src(), "stream_read_probe")?;
        let n_vec = (src.len() / 16) as i32;
        let cfg = LaunchConfig {
            grid_dim: (2048, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(sink).arg(src).arg(&n_vec);
        unsafe { b.launch(cfg) }.context("stream_read_probe")?;
        Ok(())
    }

    /// Blocks the tensor-core GEMM launches for this shape, and how many the
    /// device can hold at once. A ratio well under 1 means the grid, not the
    /// kernel, is the limit.
    pub fn mmq_grid(&self, n: usize, n_tokens: usize) -> (u32, u32, u32) {
        let tiles = Self::mmq_tiles(n_tokens);
        let warps = self.mmq_warps(n, n_tokens);
        let blocks = (n as u32).div_ceil(warps * 8)
            * (n_tokens as u32).div_ceil(tiles * MMQ_M);
        // Shared memory is what caps blocks per SM: about 16 KB at four warps
        // and one tile, halving the count each time either doubles.
        let per_sm = (6 * 4 / warps.max(1)).min(16) / if tiles == 1 { 1 } else { 2 };
        (blocks, self.dev.sm_count() * per_sm.max(1), warps)
    }

    /// How many 16-token tiles share one staging of the weight tile.
    /// How many 16-token tiles share one staging of the weight tile.
    ///
    /// More tiles means fewer passes over the weights but more shared memory,
    /// and shared memory is what caps blocks per SM. Since the staging is
    /// latency-bound rather than bandwidth-bound, losing resident warps can
    /// cost more than the saved pass — so the thresholds are measured, not
    /// derived. `INFERO_MMQ_TILES` overrides them for re-measuring.
    fn mmq_tiles(n_tokens: usize) -> u32 {
        if let Some(t) = Self::tiles_override() {
            return t;
        }
        // Measured on an A4000 against `blk.0.ffn_gate.weight` of
        // Llama-3.1-8B Q4_K_M: at 16 tokens one tile beats two by 18% because
        // it keeps six blocks per SM instead of four, and by 64 tokens two
        // tiles win by 15% because the saved weight pass finally outweighs
        // that. Four tiles never won at any width.
        // Four tiles was tried on the theory that 64 tokens to one pass over
        // the weights would finally pay, and it does not: on Blackwell at 64
        // tokens it measured 478 GB/s against two tiles' 1149. The ring is
        // four deep in tokens as well as stages, so the shared request grows
        // with it and the block count per SM falls faster than the saved
        // weight pass earns.
        if n_tokens <= 16 { 1 } else { 2 }
    }

    /// The f16-operand kernel `INFERO_MMQ_VARIANT` selects, if it selects one.
    ///
    /// The model has to ask, because this path wants a different activation
    /// buffer: f16 rather than Q8_1. Only Q4_G128 has such a kernel.
    /// The f16-operand kernel is the Q4_G128 default, unless
    /// `INFERO_MMQ_VARIANT` names something else.
    ///
    /// Measured on every Q4_G128 shape a Llama-3.1-8B step touches, in GB/s of
    /// weights against `mmqd`, which held this slot before:
    ///
    /// | | 8 tokens | 32 tokens |
    /// | --- | --- | --- |
    /// | `ffn_gate`, 4096x14336 | 237 -> 331 | 93 -> 214 |
    /// | `ffn_down`, 14336x4096 | 159 -> 341 | 75 -> 227 |
    /// | `attn_q`, 4096x4096 | 180 -> 236 | 87 -> 171 |
    /// | `attn_k`, 4096x1024 | 167 -> 158 | 72 -> 115 |
    ///
    /// Ahead everywhere except `attn_k` at eight tokens, where it gives up 5%
    /// on a matrix that is 3.8% of a layer's weights — 0.2% of the step, against
    /// 2.3x on `ffn_down`.
    pub fn mmq_f16_variant() -> Option<&'static str> {
        Self::mmq_f16_variant_for(WeightType::Q4G128)
    }

    /// The f16-operand kernel for a weight layout.
    ///
    /// The two Q4_G128 layouts need different kernels and the name does not say
    /// so — `mmqz_*` reads the transposed one, `mmqf_*` the packed one — which
    /// is a mistake waiting to happen, and did: routing transposed weights
    /// through `mmqf1w8s2` produces fluent-looking garbage rather than an
    /// error, and `batch_bench` times it happily because it never looks at what
    /// comes out.
    /// [`Self::mmq_f16_variant_for`], widened for matrices that reward it.
    ///
    /// A block covering more rows re-reads the activations fewer times, which
    /// is what this kernel spends its batch-32 time on: for `ffn_gate` the
    /// activations are 58.7 MiB against 31 of weights at 32 tokens. Two
    /// row-blocks pays for that only once the matrix is wide enough to keep the
    /// grid full — measured at 32 tokens, in microseconds, one against two:
    /// `attn_k` 8.2/8.7, `qkv` 16.8/18.3, `ffn_gate` 27.0/27.8, `ffn_down`
    /// 26.9/28.3, all losses; `gate_up` at 28672 wide, 45.5 against 41.6, a
    /// repeatable 8.6%. The threshold sits between the two widest shapes a
    /// Llama-family layer has, which is as much as one model can say.
    pub fn mmq_f16_variant_for_shape(ty: WeightType, _n: usize) -> Option<&'static str> {
        let v = Self::mmq_f16_variant_for(ty)?;
        // An explicitly pinned variant is pinned: without this the shape rule
        // below silently overrides it on the widest matrix, so pinning
        // `mmqy1w8s2` to measure the rule measured the rule.
        if std::env::var_os("INFERO_MMQ_VARIANT").is_some() {
            return Some(v);
        }
        // One shape for every width. `gate_up` used to get `mmqy2w8s2` — twice
        // the rows a block, half the activation re-reads — on the strength of a
        // step's matmuls taking 96.1 ms against `mmqy1w8s2`'s 98.7. That
        // reversed, the way the NBLK sweep in `mmq.cu` reversed the f16 path's
        // first conclusion, and for the same reason: it was measured before the
        // weight prefetch and the grid constant that followed it.
        //
        // Re-measured on the Blackwell at 32 tokens, isolated (GB/s of
        // weights): 1164 for `mmqy1w8s2` against 1082 for `mmqy2w8s2`, 53.6 us
        // against 57.7. In the served engine, one binary, `INFERO_MMQ_VARIANT`
        // pinning the narrow shape for every matrix: **layers 5.898 ms against
        // 5.988 and 4828 tok/s against 4772**.
        //
        // Halving the activation traffic is not the trade it looks like. Those
        // re-reads are L2 hits — `each_projection_at_its_own_shape` puts every
        // shape above this card's DRAM peak once they are counted — and what a
        // wider row group actually costs is blocks. `mmqy4w8s2` halves them
        // again and loses 27%.
        Some(v)
    }

    pub fn mmq_f16_variant_for(ty: WeightType) -> Option<&'static str> {
        static V: std::sync::OnceLock<Option<&'static str>> = std::sync::OnceLock::new();
        static VT: std::sync::OnceLock<Option<&'static str>> = std::sync::OnceLock::new();
        match ty {
            WeightType::Q4G128T => *VT.get_or_init(|| {
                match std::env::var("INFERO_MMQ_VARIANT") {
                    // `mmqc` is the weight-ring family, which reads the same
                    // f16 activations and so belongs on this path too — it was
                    // unreachable from the model before, and measuring it
                    // silently ran the integer fallback instead.
                    Ok(v)
                        if v.starts_with("mmqz")
                            || v.starts_with("mmqy")
                            || v.starts_with("mmqc") =>
                    {
                        Some(&*Box::leak(v.into_boxed_str()))
                    }
                    Ok(_) => None,
                    Err(_) => Some("mmqy1w8s2"),
                }
            }),
            _ => *V.get_or_init(|| match std::env::var("INFERO_MMQ_VARIANT") {
                Ok(v) if v.starts_with("mmqf") => Some(&*Box::leak(v.into_boxed_str())),
                // Any other explicit name means the caller wants an integer
                // kernel, and those take Q8_1 activations.
                Ok(_) => None,
                Err(_) => Some("mmqf1w8s2"),
            }),
        }
    }

    /// `INFERO_MMQ_SPLITS` pins the k-split factor.
    fn mmq_splits() -> Option<u32> {
        static V: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
        *V.get_or_init(|| {
            std::env::var("INFERO_MMQ_SPLITS")
                .ok()
                .and_then(|v| v.parse().ok())
        })
    }

    fn tiles_override() -> Option<u32> {
        static O: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
        *O.get_or_init(|| {
            std::env::var("INFERO_MMQ_TILES").ok().and_then(|v| v.parse().ok())
        })
    }

    fn warps_override() -> Option<u32> {
        static O: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
        *O.get_or_init(|| {
            std::env::var("INFERO_MMQ_WARPS").ok().and_then(|v| v.parse().ok())
        })
    }

    /// Warps per block, hence weight rows per block.
    ///
    /// Eight, which is a borrowed number rather than a derived one: llama.cpp's
    /// tuned Ampere table asks for 256 threads and 128 output rows per block for
    /// Q4_K at every batch width, against the 128 threads and 32 rows this
    /// kernel started with. Wide blocks reuse each staged activation fragment
    /// across four times as many weight rows, which is the traffic that grows
    /// with batch. Measured here at 64 rows — 128 does not fit the 48 KB static
    /// shared-memory limit with this tile layout — it is worth 10% at one token
    /// and 2% at 32.
    ///
    /// Note what the same table says about occupancy: 1. They deliberately run
    /// one block per SM and make it wide, which is the opposite of the instinct
    /// that narrower blocks would help here — an instinct this kernel already
    /// tested and measured at exactly zero.
    ///
    /// Chosen per matrix, because the trade runs both ways: a wide block is
    /// free reuse on a 14336-row projection (224 blocks, plenty) and is pure
    /// loss on a 1024-row one, where it halves an already-short grid from 32
    /// blocks to 16. Measured end to end, a flat 8 gained 2% at batch 32 and
    /// lost 5% at batch 8 for exactly that reason.
    ///
    /// `INFERO_MMQ_WARPS` overrides, for re-measuring on another device.
    fn mmq_warps(&self, n: usize, _n_tokens: usize) -> u32 {
        if let Some(w) = Self::warps_override() {
            return w;
        }
        let wide = (n as u32).div_ceil(64);
        if wide >= self.dev.sm_count() * 2 { 8 } else { 4 }
    }

    fn mmq_kernel_name(&self, ty: WeightType, n: usize, n_tokens: usize) -> String {
        let warps = self.mmq_warps(n, n_tokens);
        // This family is instantiated at one and two token tiles only.
        let tiles = Self::mmq_tiles(n_tokens).min(2);
        let w = if warps == 4 {
            String::new()
        } else {
            format!("w{warps}")
        };
        let t = if tiles == 1 { String::new() } else { tiles.to_string() };
        let sep = if !w.is_empty() && !t.is_empty() { "_" } else { "" };
        format!("mmq{w}{sep}{t}_{}", ty.suffix())
    }

    /// What the driver says a kernel's occupancy is limited to.
    ///
    /// `max_threads_per_block` is derived from the register count, so a value
    /// below the launch's block size means registers are the constraint and a
    /// value at or above it means shared memory or the launch shape is. Worth
    /// asking before cutting either: the batch-32 GEMM spends 54% of its time
    /// in the MMA pipeline against an instruction count that should be
    /// negligible, which is what unhidden shared-memory latency looks like.
    #[cfg(feature = "cuda")]
    pub fn kernel_limits(&self, module: &'static str, name: &str) -> Result<(i32, i32)> {
        let src = match module {
            "infero_mmq" => mmq_src(),
            "infero_mmvq" => mmvq_src(),
            "infero_ops" => ops_src(),
            _ => quant_src(),
        };
        let f = self.dev.kernels().get(module, src, name)?;
        Ok((f.max_threads_per_block()?, f.binary_version()?))
    }

    /// Load an A fragment two ways — cooperatively with `ldmatrix` and with the
    /// scalar gather it would replace — and hand both back for comparison.
    pub fn ldmatrix_probe(
        &self,
        out: &mut ViewMut<'_, i32>,
        a: &View<'_, i8>,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_mmq", mmq_src(), "ldmatrix_a_probe")?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(a).arg(out);
        unsafe { b.launch(cfg) }.context("ldmatrix_a_probe")?;
        Ok(())
    }

    /// [`Self::ldmatrix_probe`] for the B operand.
    pub fn ldmatrix_b_probe(
        &self,
        out: &mut ViewMut<'_, i32>,
        b_in: &View<'_, i8>,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_mmq", mmq_src(), "ldmatrix_b_probe")?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(b_in).arg(out);
        unsafe { b.launch(cfg) }.context("ldmatrix_b_probe")?;
        Ok(())
    }

    /// One `mma.m16n8k32.s8` on a 16x32 by 32x8 pair of int8 tiles.
    /// One `mma.m16n8k32.s8` on a 16x32 by 32x8 pair of int8 tiles.
    ///
    /// Only [`crate`]'s tests call this. It exists so the fragment layouts the
    /// real kernel depends on are proven against a CPU reference instead of
    /// trusted.
    pub fn mma_s8_probe(
        &self,
        d: &mut ViewMut<'_, i32>,
        a: &View<'_, i8>,
        b_in: &View<'_, i8>,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_mmq", mmq_src(), "mma_s8_probe")?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut bb = self.dev.stream().launch_builder(&f);
        bb.arg(a).arg(b_in).arg(d);
        unsafe { bb.launch(cfg) }.context("mma_s8_probe")?;
        Ok(())
    }

    /// [`Self::mmq_variant`] for the f16-operand kernels, whose activations are
    /// plain f16 rather than Q8_1.
    ///
    /// Separate entry point on purpose: the whole point of the f16 path is that
    /// there is no activation quantization, so routing it through a parameter
    /// named `x_q8_1` would be a lie about what the buffer holds. The model
    /// still dispatches the Q8_1 path; this is what the A/B measures against
    /// it before that changes.
    #[allow(clippy::too_many_arguments)]
    pub fn mmq_f16(
        &self,
        variant: &str,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        x_f16: &View<'_, half::f16>,
        k: usize,
        n: usize,
        n_tokens: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            variant.starts_with("mmqf") || variant.starts_with("mmqg")
                || variant.starts_with("mmqz") || variant.starts_with("mmqy")
                || variant.starts_with("mmqk") || variant.starts_with("mmqc")
                || variant.starts_with("mmqt") || variant.starts_with("mmqm")
                || variant.starts_with("mmqnm") || variant.starts_with("mmqnx")
                || variant.starts_with("mmqnh") || variant.starts_with("mmqnr"),
            "mmq_f16 is for the f16-operand kernels, got {variant}"
        );
        anyhow::ensure!(
            k.is_multiple_of(128),
            "{variant} needs k divisible by 128, got {k}"
        );
        // The launcher takes activations as bytes and the kernel casts; only
        // the element type differs from the Q8_1 path.
        // Safe in the only sense that matters here: the kernel reads this
        // buffer through `cp.async` as raw bytes and never as `u8` values, and
        // an f16 slice is exactly twice as many bytes with no padding.
        let bytes = unsafe { x_f16.transmute::<u8>(x_f16.len() * 2) }
            .context("f16 activations do not reinterpret as bytes")?;
        self.mmq_variant(
            variant,
            out,
            w,
            WeightType::Q4G128,
            &bytes,
            k,
            n,
            n_tokens,
        )
    }

    /// One `mma.m16n8k16.f16` on a 16x16 by 16x8 pair of f16 tiles.
    ///
    /// The f16 counterpart of [`Self::mma_s8_probe`], and there for the same
    /// reason: the f16-operand GEMM builds its fragments by hand.
    pub fn mma_f16_probe(
        &self,
        d: &mut ViewMut<'_, f32>,
        a: &View<'_, half::f16>,
        b_in: &View<'_, half::f16>,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_mmq", mmq_src(), "mma_f16_probe")?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut bb = self.dev.stream().launch_builder(&f);
        bb.arg(a).arg(b_in).arg(d);
        unsafe { bb.launch(cfg) }.context("mma_f16_probe")?;
        Ok(())
    }

    /// Validates the register<->MMA-fragment bridge a tensor-core version of
    /// `gdn_chunk_state_f32`'s state-advance would need, before any real
    /// kernel commits to it -- see `gdn_state_bridge_probe`'s own doc comment
    /// in `mma.cuh` for exactly what this proves.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_state_bridge_probe(
        &self,
        pred_out: &mut ViewMut<'_, f32>,
        roundtrip_out: &mut ViewMut<'_, f32>,
        s_in: &View<'_, f32>,
        w_in: &View<'_, half::f16>,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_mmq", mmq_src(), "gdn_state_bridge_probe")?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(s_in).arg(w_in).arg(pred_out).arg(roundtrip_out);
        unsafe { b.launch(cfg) }.context("gdn_state_bridge_probe")?;
        Ok(())
    }

    /// One `mma.m16n8k32.e4m3` on a 16x32 by 8x32 pair of e4m3 tiles.
    ///
    /// The e4m3 counterpart of [`Self::mma_s8_probe`]: same fragment layout,
    /// same register counts, an `s8` operand's byte reinterpreted as `e4m3`.
    /// See `mma_e4m3` in `mma.cuh`.
    pub fn mma_e4m3_probe(
        &self,
        d: &mut ViewMut<'_, f32>,
        a: &View<'_, u8>,
        b_in: &View<'_, u8>,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_mmq", mmq_src(), "mma_e4m3_probe")?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut bb = self.dev.stream().launch_builder(&f);
        bb.arg(a).arg(b_in).arg(d);
        unsafe { bb.launch(cfg) }.context("mma_e4m3_probe")?;
        Ok(())
    }

    /// One 128-weight Q4_G128 block through the `lop3` dequantization, laid
    /// out by logical k. Tests only; see `mmq_deq4_f16_probe` in `mmq.cu`.
    pub fn deq4_f16_probe(
        &self,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_mmq", mmq_src(), "mmq_deq4_f16_probe")?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(w).arg(out);
        unsafe { b.launch(cfg) }.context("mmq_deq4_f16_probe")?;
        Ok(())
    }

    /// Two mat-vecs that share one activation, in one launch.
    ///
    /// Back to back these kernels reach 328 GB/s where one alone reaches 392,
    /// and a CUDA graph does not close the gap: what costs is each kernel
    /// draining before the next can start. Merging the FFN's gate and up
    /// projections, and Q/K/V through [`Kernels::mmvq_fused3`], removes
    /// ninety-six of those drains from a decode step.
    ///
    /// Both matrices must share `k` and the weight type.
    #[allow(clippy::too_many_arguments)]
    pub fn mmvq_fused2(
        &self,
        out0: &mut ViewMut<'_, f32>,
        out1: &mut ViewMut<'_, f32>,
        w0: &View<'_, u8>,
        w1: &View<'_, u8>,
        ty: WeightType,
        x_q8_1: &View<'_, u8>,
        k: usize,
        n0: usize,
        n1: usize,
    ) -> Result<()> {
        let f = self.fused_kernel(ty, "mmvqf2_")?;
        let cfg = self.fused_cfg(ty, k, n0 + n1);
        let (k_i, a, b_n) = (k as i32, n0 as i32, n1 as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out0)
            .arg(out1)
            .arg(w0)
            .arg(w1)
            .arg(x_q8_1)
            .arg(&k_i)
            .arg(&a)
            .arg(&b_n);
        self.dev.profile().time("mmvq_fused", self.dev.stream(), || {
            unsafe { b.launch(cfg) }.context("mmvq_fused2")?;
            Ok(())
        })?;
        Ok(())
    }

    /// Three mat-vecs that share one activation. See [`Kernels::mmvq_fused2`].
    ///
    /// All three must share `k` and the weight type — a Q4_K_M file gives its
    /// first layer a Q6_K V projection between two Q4_K siblings, so the caller
    /// checks rather than assuming.
    #[allow(clippy::too_many_arguments)]
    pub fn mmvq_fused3(
        &self,
        out0: &mut ViewMut<'_, f32>,
        out1: &mut ViewMut<'_, f32>,
        out2: &mut ViewMut<'_, f32>,
        w0: &View<'_, u8>,
        w1: &View<'_, u8>,
        w2: &View<'_, u8>,
        ty: WeightType,
        x_q8_1: &View<'_, u8>,
        k: usize,
        ns: [usize; 3],
    ) -> Result<()> {
        let f = self.fused_kernel(ty, "mmvqf3_")?;
        let cfg = self.fused_cfg(ty, k, ns.iter().sum());
        let k_i = k as i32;
        let n: Vec<i32> = ns.iter().map(|v| *v as i32).collect();
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out0)
            .arg(out1)
            .arg(out2)
            .arg(w0)
            .arg(w1)
            .arg(w2)
            .arg(x_q8_1)
            .arg(&k_i)
            .arg(&n[0])
            .arg(&n[1])
            .arg(&n[2]);
        self.dev.profile().time("mmvq_fused", self.dev.stream(), || {
            unsafe { b.launch(cfg) }.context("mmvq_fused3")?;
            Ok(())
        })?;
        Ok(())
    }

    fn fused_kernel(
        &self,
        ty: WeightType,
        prefix: &str,
    ) -> Result<infero_gpu::Function> {
        anyhow::ensure!(Self::has_mmvq(ty), "no integer mat-vec for {ty}");
        let name = format!("{prefix}{}", ty.suffix());
        self.dev.kernels().get("infero_mmvq", mmvq_src(), &name)
    }

    /// Same block shape as [`Kernels::mmvq`], over the concatenated rows.
    fn fused_cfg(&self, ty: WeightType, k: usize, rows: usize) -> LaunchConfig {
        let slices = match ty {
            WeightType::Q8_0 | WeightType::Q6K => k / 8,
            WeightType::Q4G128 | WeightType::Q4G128T => k / 32,
            _ => k / 16,
        };
        LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (
                (slices as u32).next_multiple_of(32).clamp(32, REDUCE_BLOCK),
                1,
                1,
            ),
            shared_mem_bytes: 0,
        }
    }

    pub fn mmvq(
        &self,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        ty: WeightType,
        x_q8_1: &View<'_, u8>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        self.mmvq_inner(out, w, ty, x_q8_1, k, n, false)
    }

    /// [`Kernels::mmvq`] adding into `out` instead of overwriting it.
    ///
    /// The output and down projections both feed straight back into the
    /// residual stream, and folding that add into the projection saves a
    /// kernel and three passes over the vector per layer.
    #[allow(clippy::too_many_arguments)]
    pub fn mmvq_add(
        &self,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        ty: WeightType,
        x_q8_1: &View<'_, u8>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        self.mmvq_inner(out, w, ty, x_q8_1, k, n, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn mmvq_inner(
        &self,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        ty: WeightType,
        x_q8_1: &View<'_, u8>,
        k: usize,
        n: usize,
        accum: bool,
    ) -> Result<()> {
        anyhow::ensure!(Self::has_mmvq(ty), "no integer mat-vec for {ty}");
        // Rows per block for the warp-per-row shape; 0 is the block-per-row
        // one, which measured level with every setting of it and is the
        // default. Read once — this is asked 225 times a step.
        static ROWS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
        let rows = *ROWS.get_or_init(|| {
            std::env::var("INFERO_MMVQ_ROWS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0)
        });
        let warped = rows > 0 && (n as u32).is_multiple_of(rows);
        let name = format!("{}{}", if warped { "mmvqw_" } else { "mmvq_" }, ty.suffix());
        let f = self.dev.kernels().get("infero_mmvq", mmvq_src(), &name)?;
        // One slice per thread, same shape as the float path.
        let slices = match ty {
            WeightType::Q8_0 => k / 8,
            WeightType::Q4K => k / 16,
            WeightType::Q6K => k / 8,
            // One 32-weight quarter of a group per thread.
            WeightType::Q4G128 | WeightType::Q4G128T => k / 32,
            _ => unreachable!("guarded above"),
        };
        let block = (slices as u32).next_multiple_of(32).clamp(32, REDUCE_BLOCK);
        let cfg = if warped {
            LaunchConfig {
                grid_dim: ((n as u32) / rows, 1, 1),
                block_dim: (32, rows, 1),
                shared_mem_bytes: 0,
            }
        } else {
            LaunchConfig {
                grid_dim: (n as u32, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            }
        };
        let (k_i, n_i, acc) = (k as i32, n as i32, i32::from(accum));
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(w).arg(x_q8_1).arg(&k_i).arg(&n_i).arg(&acc);
        self.dev.profile().time("mmvq", self.dev.stream(), || {
            unsafe { b.launch(cfg) }.context("mmvq")?;
            Ok(())
        })?;
        Ok(())
    }

    /// `out[t, r] = dot(w[r, :], x[t, :])` decoding `w` on the fly.
    ///
    /// Reads the weights once per token, so it is the right kernel for one or
    /// a few tokens and the wrong one for a long prefill — see
    /// [`Kernels::gemm_f16`].
    #[allow(clippy::too_many_arguments)]
    /// The Q4_K mat-vec on the matrix units: eight rows and eight tokens a
    /// simdgroup, one simdgroup a threadgroup.
    fn gemv_mma(
        &self,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        x: &View<'_, f32>,
        k: usize,
        n: usize,
        n_tokens: usize,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_quant", quant_src(), "gemv_mma_q4_K")?;
        let cfg = LaunchConfig {
            grid_dim: (
                (n as u32).div_ceil(8),
                (n_tokens as u32).div_ceil(8).max(1),
                1,
            ),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (k_i, n_i, nt_i) = (k as i32, n as i32, n_tokens as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(w).arg(x).arg(&k_i).arg(&n_i).arg(&nt_i);
        self.dev.profile().time("gemv_mma", self.dev.stream(), || {
            unsafe { b.launch(cfg) }.context("gemv_mma")?;
            Ok(())
        })
    }

    /// `gemv_mma_q4_K`'s tiling, for Q8_0. See the kernel's own doc comment
    /// in quant.metal for why Q8_0 had no matrix-unit path before this.
    fn gemv_mma_q8(
        &self,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        x: &View<'_, f32>,
        k: usize,
        n: usize,
        n_tokens: usize,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_quant", quant_src(), "gemv_mma_q8_0")?;
        let cfg = LaunchConfig {
            grid_dim: (
                (n as u32).div_ceil(8),
                (n_tokens as u32).div_ceil(8).max(1),
                1,
            ),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (k_i, n_i, nt_i) = (k as i32, n as i32, n_tokens as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(w).arg(x).arg(&k_i).arg(&n_i).arg(&nt_i);
        self.dev
            .profile()
            .time("gemv_mma_q8", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("gemv_mma_q8")?;
                Ok(())
            })
    }

    /// `gemv_mma_q8_0` widened to 32 rows a threadgroup with cooperative
    /// decode across all 128 threads, the same fix
    /// `gemv_mma_coop32_q4_K`'s doc comment traces for Q4_K, applied here
    /// directly rather than through an intermediate serial-decode widening
    /// step -- `gemv_mma_q8_0` never had one to begin with. Loses to
    /// `gemv_mma_q8_0` below 24 tokens (0.75-0.92x, `examples/
    /// gemv_q8_0_threshold_check.rs`): a 32-token-wide tile mostly empty of
    /// real tokens is the same underfill `gemv_mma_shared_q4_K` loses to
    /// below 32; wins from 24 up, 1.14-1.62x, growing with token count.
    fn gemv_mma_coop_q8(
        &self,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        x: &View<'_, f32>,
        k: usize,
        n: usize,
        n_tokens: usize,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_quant", quant_src(), "gemv_mma_coop_q8_0")?;
        let cfg = LaunchConfig {
            grid_dim: (
                (n as u32).div_ceil(32),
                (n_tokens as u32).div_ceil(32).max(1),
                1,
            ),
            block_dim: (32, 4, 1),
            shared_mem_bytes: 0,
        };
        let (k_i, n_i, nt_i) = (k as i32, n as i32, n_tokens as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(w).arg(x).arg(&k_i).arg(&n_i).arg(&nt_i);
        self.dev
            .profile()
            .time("gemv_mma_coop_q8", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("gemv_mma_coop_q8")?;
                Ok(())
            })
    }

    /// `gemv_mma_shared_q4_K`'s tiling widened to 32 rows a threadgroup,
    /// with the fix a naive widening needs to actually pay off: all 128
    /// threads decode cooperatively (one row a group of four lanes, no
    /// longer confined to simdgroup 0's 32) instead of one simdgroup
    /// decoding for the other three to wait on. See the kernel's own doc
    /// comment in quant.metal, and the two it supersedes
    /// (`gemv_mma_shared16_q4_K`, which found that widening to 16 rows wins
    /// even with the old serial decode; `gemv_mma_shared32_q4_K`, which
    /// found that widening further to 32 rows *without* fixing the decode
    /// loses to 16) for the trail that led here.
    ///
    /// Measured against the original 8-row `gemv_mma_shared_q4_K` at every
    /// token count it is ever actually dispatched at (32-128,
    /// `examples/gemv_mma_shared16_check.rs`): 1.24-2.36x, byte-exact,
    /// beating both `gemv_mma_shared16_q4_K` (1.12-1.19x there) and
    /// `gemv_mma_shared32_q4_K` (0.94-1.06x) at every one of them. Replaces
    /// `gemv_mma_shared16` outright in the `gemv` dispatch; all three
    /// superseded kernels stay in quant.metal undeleted.
    fn gemv_mma_coop32(
        &self,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        x: &View<'_, f32>,
        k: usize,
        n: usize,
        n_tokens: usize,
    ) -> Result<()> {
        const TOKGROUPS: u32 = 4;
        let f = self
            .dev
            .kernels()
            .get("infero_quant", quant_src(), "gemv_mma_coop32_q4_K")?;
        let cfg = LaunchConfig {
            grid_dim: (
                (n as u32).div_ceil(32),
                (n_tokens as u32).div_ceil(8 * TOKGROUPS).max(1),
                1,
            ),
            block_dim: (32, TOKGROUPS, 1),
            shared_mem_bytes: 0,
        };
        let (k_i, n_i, nt_i) = (k as i32, n as i32, n_tokens as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(w).arg(x).arg(&k_i).arg(&n_i).arg(&nt_i);
        self.dev
            .profile()
            .time("gemv_mma_coop32", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("gemv_mma_coop32")?;
                Ok(())
            })
    }

    pub fn gemv(
        &self,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        ty: WeightType,
        x: &View<'_, f32>,
        k: usize,
        n: usize,
        n_tokens: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            k.is_multiple_of(ty.block_size()),
            "{ty:?} needs k ({k}) to be a multiple of {}",
            ty.block_size()
        );
        // A one-token specialisation where the backend has one.
        //
        // `GEMV_SPREAD`'s trip count must be a compile-time constant or the
        // accumulator leaves registers, so the batched kernel runs eight
        // iterations with a predicate and at one token seven are dead. Measured
        // on an M4 Max, one token, against the batched kernel at the same shape:
        //
        //   output.weight Q6_K   61.9 -> 145.0 GB/s
        //   ffn_down      Q4_K   41.8 ->  95.8
        //   ffn_gate/up   Q4_K   41.2 ->  92.2
        //   attn_qkv      Q8_0   73.7 -> 133.3
        //
        // Decode is always one token, so this is the decode path. The CUDA side
        // has no `gemv1_*`: there `#pragma unroll` on a compile-time trip count
        // already lets the compiler drop the dead iterations, which is the same
        // fix by a different mechanism.
        //
        // `gemv2_*` and `gemv4_*` exist for the same reason and were added for a
        // narrower case: speculation's verification pass is `k + 1` rows, so at
        // the default `k = 1` every round runs a two-row matmul. Sending that
        // through the eight-token kernel cost 2.3x -- six dead iterations a
        // weight element -- and it turned a 1.76-token acceptance into a
        // *slowdown*, 8.4 tok/s down to 4.0. A round is only worth running if
        // its wide pass costs about what the narrow one it replaces did.
        //
        // The group width is the smallest specialisation that covers the rows,
        // and `token0 = tgid.y * T` in `GEMV_PROLOGUE` means the grid follows
        // from it without the kernel knowing which one it is.
        // Rows one threadgroup owns.
        //
        // Four for a batch, one for a decode step, and the asymmetry is
        // measured rather than chosen. `out[n] = W . x` re-reads the whole
        // activation for every output row, so the activation traffic is
        // `n * k * 4` against the weights' `n * k * 4.5/8` -- 7.1x -- and four
        // rows sharing one load divides it by four. But at one token that
        // traffic is not yet binding: Q4_K measures 1.50 TB/s of combined
        // weight-plus-activation traffic there, and the four block reductions
        // and the extra registers cost 7%. At two tokens it is 1.81 TB/s, which
        // is the wall, and sharing wins:
        //
        //   ffn_gate/up Q4_K   1 tok  0.26 -> 0.28 ms   (four rows loses)
        //                      2 tok  0.41 -> 0.35      1.17x
        //                      4 tok  0.68 -> 0.46      1.48x
        //
        // The gain growing with the token count is the signature of activation
        // traffic being what is amortised, and it is why the same change did
        // nothing when first tried at one token.
        //
        // Q4_K only so far: Q8_0 already amortises (1.04x at two tokens, so
        // there is nothing to win) and Q6_K does not (1.74x) but is one launch
        // a step against 448.
        let rows = if cfg!(feature = "cuda") || n_tokens < 2 || ty != WeightType::Q4K {
            1
        } else {
            4
        };
        // Tokens one group carries. Capped at four rather than eight for the
        // multi-row path, because `gemv_q4_K` at eight tokens measures 3.95 ms
        // where two four-token launches measure 0.92 -- the wide kernel is
        // *worse than looping* past four, which is also why four concurrent
        // sequences cost 3.23x a step instead of ~1.1x.
        let per = if cfg!(feature = "cuda") {
            GEMV_TOKENS_PER_BLOCK as usize
        } else {
            match n_tokens {
                0 | 1 => 1,
                2 => 2,
                3 | 4 => 4,
                _ if rows > 1 => 4,
                _ => GEMV_TOKENS_PER_BLOCK as usize,
            }
        };
        // The matrix units, for a batch wide enough to fill an 8x8 tile.
        //
        // Measured against the scalar kernel, Q4_K, as a multiple of the
        // one-token scalar cost -- the MMA kernel's cost is flat in the token
        // count, which is the whole property:
        //
        //   tokens   scalar        mma
        //        2   1.30 1.26   1.73 2.12
        //        4   1.88 2.01   1.57 2.10
        //        8   3.66 4.20   1.66 2.33
        //
        // So it loses at two -- which is what a speculative verification pass
        // is, and why speculation cannot be rescued this way -- and wins from
        // eight, by 1.8x to 2.2x. `INFERO_MMA_MIN` moves the line.
        let mma = !cfg!(feature = "cuda")
            && ty == WeightType::Q4K
            && n_tokens >= mma_min()
            && n as u32 >= 8;
        if mma {
            // This threshold used to be 32: `gemv_mma_shared_q4_K`'s four
            // simdgroups sat mostly idle below it (0.40-0.91x, examples/
            // gemv_mma_multisg_check.rs) because its decode was serial on
            // one of them regardless of token count, so a mostly-empty
            // token tile bought nothing to offset that fixed cost.
            // `gemv_mma_coop32_q4_K`'s decode is cooperative across all 128
            // threads instead, and that changes the shape of this trade
            // completely: measured down to eight tokens (examples/
            // gemv_mma_shared16_check.rs), it beats the single-simdgroup
            // `gemv_mma_q4_K` everywhere in that range too, 1.29-2.05x
            // instead of losing. Lowered to match `mma_min()`'s own floor
            // rather than re-guessing a new one -- there is no token count
            // left where the plain kernel wins back. Overridable for
            // re-measuring rather than arguing about it, same as
            // `INFERO_MMA_MIN` above.
            let shared_min: usize = std::env::var("INFERO_MMA_SHARED_MIN")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|v| *v > 0)
                .unwrap_or(8);
            if n_tokens >= shared_min {
                return self.gemv_mma_coop32(out, w, x, k, n, n_tokens);
            }
            return self.gemv_mma(out, w, x, k, n, n_tokens);
        }
        // Q8_0's matrix-unit path. Unlike Q4_K's, there is no crossover
        // against the scalar kernel to weigh: `gemv_mma_q8_0` beats
        // `gemv_q8_0` at every token count measured, 3.4x at eight tokens
        // widening to 8.3x at 128 (`gemv_q8_0_threshold_check.rs`), because
        // the scalar kernel re-reads the whole activation once a token with
        // no batching to amortise it against, the same shape the Q4_K
        // scalar kernel has. There is a crossover between the two MMA
        // kernels, though, the same shape `gemv_mma_shared_q4_K`'s own
        // 32-token floor has: `gemv_mma_coop_q8_0`'s 32-token-wide tile
        // loses to the single-tile `gemv_mma_q8_0` below 24 tokens
        // (0.75-0.92x, mostly-empty tile) and wins from 24 up
        // (1.14-1.62x). `INFERO_Q8_0_COOP_MIN` moves the line.
        if !cfg!(feature = "cuda") && ty == WeightType::Q8_0 && n_tokens >= 8 && n as u32 >= 32 {
            let coop_min: usize = std::env::var("INFERO_Q8_0_COOP_MIN")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|v| *v > 0)
                .unwrap_or(24);
            if n_tokens >= coop_min {
                return self.gemv_mma_coop_q8(out, w, x, k, n, n_tokens);
            }
            return self.gemv_mma_q8(out, w, x, k, n, n_tokens);
        }
        if !cfg!(feature = "cuda") && ty == WeightType::Q8_0 && n_tokens >= 8 && n as u32 >= 8 {
            return self.gemv_mma_q8(out, w, x, k, n, n_tokens);
        }
        let name = if rows > 1 {
            format!("gemv{per}x{rows}_{}", ty.suffix())
        } else if per == GEMV_TOKENS_PER_BLOCK as usize {
            format!("gemv_{}", ty.suffix())
        } else {
            format!("gemv{per}_{}", ty.suffix())
        };
        let f = self.dev.kernels().get("infero_quant", quant_src(), &name)?;
        // Size the block to the work rather than to a constant: an oversized
        // block idles most of its threads and still pays for the block-wide
        // reduction, which for the vocab projection is the difference between
        // one warp and eight.
        // Capped well below `REDUCE_BLOCK`, and the cap is the point.
        //
        // Sizing the group to the work is right as far as it goes -- an
        // oversized group idles threads and still pays for the block-wide
        // reduction -- but "to the work" was reading as "one unit of work a
        // thread", and at that width the kernel is a launch and a reduction with
        // nothing in between. Measured on an M4 Max, the same kernel and the
        // same answer at five group widths:
        //
        //   ffn_gate/up Q4_K   32:190.2  64:183.2  128:183.2  160:130.2  256:132.0
        //   ffn_down    Q4_K   32:176.5  64:175.0  128:187.3  160:184.0  256:183.7
        //   attn_qkv    Q8_0   32:209.7  64:217.7  128:221.1  160:196.1  256:188.1
        //   output.w    Q6_K   32:154.5  64:152.4  128:150.7  160:149.8  256:145.4
        //
        // `ffn_gate/up` is the shape this rule chose 160 for, because Q4_K at
        // k = 5120 is exactly 160 work items -- and 160 is the worst column in
        // its row by 1.46x. Every one of these four is fastest at or below 128,
        // so 128 is the cap; two of them peak there and the other two give up
        // 3-4% against their own best, which is not worth a per-shape table.
        //
        // A thread looping four or five times over its own chunk hides the
        // reduction behind real work. One doing a single chunk cannot.
        const GEMV_BLOCK_MAX: u32 = 64;
        let block = (ty.gemv_work_items(k) as u32)
            .next_multiple_of(32)
            .clamp(32, GEMV_BLOCK_MAX);
        let cfg = LaunchConfig {
            grid_dim: (
                (n as u32).div_ceil(rows as u32),
                (n_tokens as u32).div_ceil(per as u32).max(1),
                1,
            ),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (k_i, n_i, nt_i) = (k as i32, n as i32, n_tokens as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(w).arg(x).arg(&k_i).arg(&n_i).arg(&nt_i);
        self.dev.profile().time("gemv", self.dev.stream(), || {
            unsafe { b.launch(cfg) }.context("gemv")?;
            Ok(())
        })?;
        Ok(())
    }

    // ---- TurboQuant KV cache ------------------------------------------
    //
    // The paper's estimator is evaluated entirely in the rotated basis, so a
    // cached key is never rotated back: the query is rotated once and the
    // QJL projection is folded into the same basis. See `cu/turboquant.cu`.

    /// `out[v] = M · x[v]`. Safe to call with `out` aliasing `x`.
    pub fn tq_matvec(
        &self,
        out: &mut ViewMut<'_, f32>,
        x: &View<'_, f32>,
        mat: &View<'_, f32>,
        d: usize,
        n_vec: usize,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_turboquant", tq_src(), "tq_matvec")?;
        let cfg = LaunchConfig {
            grid_dim: (n_vec as u32, 1, 1),
            block_dim: (per_vector_block(d), 1, 1),
            shared_mem_bytes: 0,
        };
        let (d_i, n_i) = (d as i32, n_vec as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(x).arg(mat).arg(&d_i).arg(&n_i);
        self.dev
            .profile()
            .time("tq_matvec", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("tq_matvec")?;
                Ok(())
            })?;
        Ok(())
    }

    /// TurboQuant_mse over rotated value vectors.
    #[allow(clippy::too_many_arguments)]
    pub fn tq_store_v(
        &self,
        codes: &mut ViewMut<'_, u8>,
        scale: &mut ViewMut<'_, f16>,
        src: &View<'_, f32>,
        slots: &View<'_, i32>,
        levels: &View<'_, f32>,
        bits: u8,
        n_kv_heads: usize,
        d: usize,
        n_slots: usize,
        n_tokens: usize,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_turboquant", tq_src(), "tq_store_v")?;
        let cfg = LaunchConfig {
            grid_dim: (n_kv_heads as u32, n_tokens as u32, 1),
            block_dim: (per_vector_block(d), 1, 1),
            shared_mem_bytes: 0,
        };
        let (bits_i, kh, d_i, ms, nt) = (
            bits as i32,
            n_kv_heads as i32,
            d as i32,
            n_slots as i32,
            n_tokens as i32,
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(codes)
            .arg(scale)
            .arg(src)
            .arg(slots)
            .arg(levels)
            .arg(&bits_i)
            .arg(&kh)
            .arg(&d_i)
            .arg(&ms)
            .arg(&nt);
        self.dev
            .profile()
            .time("tq_store_v", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("tq_store_v")?;
                Ok(())
            })?;
        Ok(())
    }

    /// TurboQuant_prod over rotated key vectors: MSE codes, QJL signs of the
    /// residual, and the two norms the estimator needs.
    #[allow(clippy::too_many_arguments)]
    pub fn tq_store_k(
        &self,
        codes: &mut ViewMut<'_, u8>,
        signs: &mut ViewMut<'_, u8>,
        scale: &mut ViewMut<'_, f16>,
        gamma: &mut ViewMut<'_, f16>,
        src: &View<'_, f32>,
        qjl: &View<'_, f32>,
        slots: &View<'_, i32>,
        levels: &View<'_, f32>,
        bits: u8,
        n_kv_heads: usize,
        d: usize,
        n_slots: usize,
        n_tokens: usize,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_turboquant", tq_src(), "tq_store_k")?;
        let cfg = LaunchConfig {
            grid_dim: (n_kv_heads as u32, n_tokens as u32, 1),
            block_dim: (per_vector_block(d), 1, 1),
            shared_mem_bytes: 0,
        };
        let (bits_i, kh, d_i, ms, nt) = (
            bits as i32,
            n_kv_heads as i32,
            d as i32,
            n_slots as i32,
            n_tokens as i32,
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(codes)
            .arg(signs)
            .arg(scale)
            .arg(gamma)
            .arg(src)
            .arg(qjl)
            .arg(slots)
            .arg(levels)
            .arg(&bits_i)
            .arg(&kh)
            .arg(&d_i)
            .arg(&ms)
            .arg(&nt);
        self.dev
            .profile()
            .time("tq_store_k", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("tq_store_k")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Attention logits from a quantized key cache, using the two-stage
    /// unbiased inner product estimator.
    #[allow(clippy::too_many_arguments)]
    pub fn tq_attn_scores(
        &self,
        scores: &mut ViewMut<'_, f32>,
        q_rot: &View<'_, f32>,
        q_qjl: &View<'_, f32>,
        codes: &View<'_, u8>,
        signs: &View<'_, u8>,
        scale: &View<'_, f16>,
        gamma: &View<'_, f16>,
        batch: BatchLayout<'_>,
        levels: &View<'_, f32>,
        bits: u8,
        dims: AttnDims,
        kv_len: usize,
        attn_scale: f32,
        qjl_scale: f32,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_turboquant", tq_src(), "tq_attn_scores")?;
        let cfg = LaunchConfig {
            grid_dim: (
                (kv_len as u32).div_ceil(SCORE_WARPS).max(1),
                dims.n_heads as u32,
                dims.n_tokens as u32,
            ),
            block_dim: (SCORE_WARPS * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (bits_i, stride, h, kh, dh, ms, kl) = (
            bits as i32,
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            kv_len as i32,
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(scores)
            .arg(q_rot)
            .arg(q_qjl)
            .arg(codes)
            .arg(signs)
            .arg(scale)
            .arg(gamma)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(levels)
            .arg(&bits_i)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ms)
            .arg(&kl)
            .arg(&attn_scale)
            .arg(&qjl_scale);
        self.dev
            .profile()
            .time("tq_attn_scores", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("tq_attn_scores")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Weighted sum over a quantized value cache. The result is still in the
    /// rotated basis; apply `Πᵀ` with [`Kernels::tq_matvec`] afterwards.
    #[allow(clippy::too_many_arguments)]
    pub fn tq_attn_output(
        &self,
        out: &mut ViewMut<'_, f32>,
        scores: &View<'_, f32>,
        codes: &View<'_, u8>,
        scale: &View<'_, f16>,
        batch: BatchLayout<'_>,
        levels: &View<'_, f32>,
        bits: u8,
        dims: AttnDims,
        kv_len: usize,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_turboquant", tq_src(), "tq_attn_output")?;
        let cfg = LaunchConfig {
            grid_dim: (dims.n_heads as u32, dims.n_tokens as u32, 1),
            block_dim: (per_vector_block(dims.d_head), 1, 1),
            shared_mem_bytes: 0,
        };
        let (bits_i, stride, h, kh, dh, ms, kl) = (
            bits as i32,
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            kv_len as i32,
        );
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out)
            .arg(scores)
            .arg(codes)
            .arg(scale)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(levels)
            .arg(&bits_i)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ms)
            .arg(&kl);
        self.dev
            .profile()
            .time("tq_attn_output", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("tq_attn_output")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Scores, softmax and the weighted value sum over a quantized cache in
    /// one launch — [`Kernels::tq_attn_scores`], [`Kernels::attn_softmax`] and
    /// [`Kernels::tq_attn_output`] fused the way [`Kernels::attn_decode`]
    /// fuses their dense counterparts, and for the same reason: the
    /// three-kernel path writes the whole score row to HBM and reads it back
    /// twice. Still in the rotated basis; the caller applies `Πᵀ` afterwards,
    /// same as the unfused path.
    #[allow(clippy::too_many_arguments)]
    pub fn tq_attn_decode(
        &self,
        out: &mut ViewMut<'_, f32>,
        q_rot: &View<'_, f32>,
        q_qjl: &View<'_, f32>,
        k_codes: &View<'_, u8>,
        k_signs: &View<'_, u8>,
        k_scale: &View<'_, f16>,
        k_gamma: &View<'_, f16>,
        v_codes: &View<'_, u8>,
        v_scale: &View<'_, f16>,
        batch: BatchLayout<'_>,
        k_levels: &View<'_, f32>,
        k_bits: u8,
        v_levels: &View<'_, f32>,
        v_bits: u8,
        dims: AttnDims,
        kv_len: usize,
        attn_scale: f32,
        qjl_scale: f32,
        partial: &mut ViewMut<'_, f32>,
    ) -> Result<()> {
        let group = dims.n_heads / dims.n_kv_heads.max(1);
        anyhow::ensure!(
            group >= 1 && group <= 8 && dims.n_heads == group * dims.n_kv_heads,
            "tq_attn_decode: group {group} out of the fixed-array range this kernel unrolls"
        );
        // Chunked over the key range for the same reason `attn_decode` is:
        // `n_kv_heads * n_tokens` blocks is four at this model's shape, and a
        // device with 188 SMs runs the other 184 idle for the kernel's whole
        // duration otherwise -- `ncu` measured 0.74% compute throughput and
        // 16.7% occupancy before this was added, on a grid `ncu` itself
        // flagged as sized for four SMs. `attn_partial_floats` is enough
        // scratch either way; the caller already sizes `partial` for it.
        let (n_chunks, chunk) = self.decode_chunks(&dims, kv_len);
        let ms_off = (32 * dims.n_heads * dims.n_tokens * dims.d_head) as i32;
        let f = self
            .dev
            .kernels()
            .get("infero_turboquant", tq_src(), "tq_attn_decode_f32")?;
        let block = (SCORE_WARPS * 32).max(per_vector_block(dims.d_head));
        let tile = block / 32;
        let cfg = LaunchConfig {
            grid_dim: (dims.n_kv_heads as u32, dims.n_tokens as u32, n_chunks),
            block_dim: (block, 1, 1),
            shared_mem_bytes: tile * group as u32 * 4,
        };
        let (kb, vb, stride, h, kh, dh, ms, kl, g, cw) = (
            k_bits as i32,
            v_bits as i32,
            batch.table_stride as i32,
            dims.n_heads as i32,
            dims.n_kv_heads as i32,
            dims.d_head as i32,
            dims.n_slots as i32,
            kv_len as i32,
            group as i32,
            chunk as i32,
        );
        // `partial` is only borrowed as a view for the decode kernel, so the
        // combine pass below can still take it as `&mut` afterwards.
        let part_ro = partial.as_view();
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(&part_ro)
            .arg(&ms_off)
            .arg(q_rot)
            .arg(q_qjl)
            .arg(k_codes)
            .arg(k_signs)
            .arg(k_scale)
            .arg(k_gamma)
            .arg(v_codes)
            .arg(v_scale)
            .arg(batch.seq_of)
            .arg(batch.positions)
            .arg(batch.slot_table)
            .arg(&stride)
            .arg(k_levels)
            .arg(&kb)
            .arg(v_levels)
            .arg(&vb)
            .arg(&h)
            .arg(&kh)
            .arg(&dh)
            .arg(&ms)
            .arg(&kl)
            .arg(&attn_scale)
            .arg(&qjl_scale)
            .arg(&g)
            .arg(&cw);
        self.dev
            .profile()
            .time("tq_attn_decode", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("tq_attn_decode")?;
                Ok(())
            })?;
        drop(part_ro);

        // The combine pass is `attn_flash_reduce_f32` itself: a chunk's
        // unnormalized sum and `{max, denominator}` pair mean the same thing
        // whether the values behind them came from an `f16` read or a
        // TurboQuant unpack, so the arithmetic that reduces them across
        // chunks does not need a second copy.
        let r = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_flash_reduce_f32")?;
        let part = partial.as_view();
        let total = (dims.n_tokens * dims.n_heads * dims.d_head) as u32;
        let (nt, nc) = (dims.n_tokens as i32, n_chunks as i32);
        let mut rb = self.dev.stream().launch_builder(&r);
        rb.arg(out).arg(&part).arg(&ms_off).arg(&h).arg(&dh).arg(&nt).arg(&nc);
        self.dev
            .profile()
            .time("tq_attn_decode_reduce", self.dev.stream(), || {
                unsafe { rb.launch(elementwise(total)) }.context("tq_attn_decode_reduce")?;
                Ok(())
            })?;
        Ok(())
    }

    /// `c[n_tokens, n] = a[n_tokens, k] · bᵀ` with f16 inputs and f32 output,
    /// accumulating in f32.
    ///
    /// `b` is the weight matrix in ggml layout, `[n, k]` row-major, already
    /// dequantized. cuBLAS is column-major, so both operands are handed over
    /// transposed-in-place and no data is moved.
    /// Not available on this backend.
    ///
    /// The CUDA path hands both operands to cuBLAS transposed-in-place. The
    /// Metal counterpart is `MPSMatrixMultiplication`, which is a library call
    /// of the same shape and is simply not wired up yet -- so this fails loudly
    /// rather than silently taking a slower path, because the dispatch only
    /// reaches it above `GEMM_THRESHOLD` tokens and a silent fallback there
    /// would look like a mysterious prefill regression.
    #[cfg(not(feature = "cuda"))]
    /// `c = a * b^T` through `MPSMatrixMultiplication`.
    ///
    /// The whole implementation is in the backend, because reaching MPS needs
    /// the raw `MTLBuffer` behind a view and that is not something a neutral
    /// caller should be able to do. See `infero_metal::gemm`.
    ///
    /// One difference from the cuBLAS path worth carrying at the call site:
    /// cuBLAS is asked for an f32 accumulator with f16 operands, and MPS
    /// requires all three matrices to share a type, so this accumulates in f16.
    /// Over `k = 17408` that is a real precision difference. It is a prefill
    /// path -- the decode mat-vec accumulates in f32 and does not come here --
    /// and prefill feeds attention rather than a sampler, but it is not nothing.
    pub fn gemm_f16(
        &self,
        c: &mut ViewMut<'_, f32>,
        a: &View<'_, f16>,
        b: &View<'_, f16>,
        n_tokens: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        self.dev.profile().time("gemm_f16", self.dev.stream(), || {
            infero_gpu::gemm_f16_to_f32(&self.dev, c, a, b, n_tokens, k, n)
        })
    }

    #[cfg(feature = "cuda")]
    pub fn gemm_f16(
        &self,
        c: &mut ViewMut<'_, f32>,
        a: &View<'_, f16>,
        b: &View<'_, f16>,
        n_tokens: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        use cudarc::cublas::sys;
        use cudarc::driver::{DevicePtr, DevicePtrMut};

        let stream = self.dev.stream();
        let (a_ptr, _ra) = a.device_ptr(stream);
        let (b_ptr, _rb) = b.device_ptr(stream);
        let (c_ptr, _rc) = c.device_ptr_mut(stream);

        let alpha = 1.0f32;
        let beta = 0.0f32;

        self.dev.profile().time("gemm_f16", self.dev.stream(), || {
            unsafe {
                cudarc::cublas::result::gemm_ex(
                    *self.dev.blas().handle(),
                    sys::cublasOperation_t::CUBLAS_OP_T,
                    sys::cublasOperation_t::CUBLAS_OP_N,
                    n as i32,
                    n_tokens as i32,
                    k as i32,
                    &alpha as *const f32 as *const _,
                    b_ptr as *const _,
                    sys::cudaDataType::CUDA_R_16F,
                    k as i32,
                    a_ptr as *const _,
                    sys::cudaDataType::CUDA_R_16F,
                    k as i32,
                    &beta as *const f32 as *const _,
                    c_ptr as *mut _,
                    sys::cudaDataType::CUDA_R_32F,
                    n as i32,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
            }
            .context("cublasGemmEx")?;
            Ok(())
        })?;
        Ok(())
    }

    /// A tight, register-resident, memory-traffic-free loop of `reps` back to
    /// back `mma.sync.m16n8k16.f16` instructions, one warp a block. See the
    /// doc comment on `mma_f16_throughput_probe`/`mma_e4m3_throughput_probe`
    /// in `ops.cu` for why this exists: measuring the real per-instruction
    /// issue rate on this card before considering an e4m3 QK^T/PV attention
    /// kernel, rather than trusting `mma_e4m3`'s "roughly double" hardware-
    /// spec comment. `out` gets one f32 a block (the folded accumulator, kept
    /// only so the compiler can't treat the whole loop as dead code).
    pub fn mma_f16_throughput_probe(
        &self,
        out: &mut ViewMut<'_, f32>,
        blocks: usize,
        reps: usize,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "mma_f16_throughput_probe")?;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let r = reps as i32;
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(&r);
        unsafe { b.launch(cfg) }.context("mma_f16_throughput_probe")?;
        Ok(())
    }

    /// The `mma.sync.m16n8k32.e4m3` counterpart of
    /// [`Self::mma_f16_throughput_probe`], same shape, same purpose.
    pub fn mma_e4m3_throughput_probe(
        &self,
        out: &mut ViewMut<'_, f32>,
        blocks: usize,
        reps: usize,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "mma_e4m3_throughput_probe")?;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let r = reps as i32;
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(&r);
        unsafe { b.launch(cfg) }.context("mma_e4m3_throughput_probe")?;
        Ok(())
    }

    /// One warp, `ws4`'s exact d_head=256/WK=48 consumer-side per-tile
    /// instruction shape (QK^T -> online-softmax bookkeeping -> PV) run
    /// `outer_iters` times against one resident, synthetic K/V tile. See
    /// `attn_full_tile_f16_probe`/`attn_full_tile_e4m3_probe` in `ops.cu`.
    pub fn attn_full_tile_f16_probe(
        &self,
        out: &mut ViewMut<'_, f32>,
        blocks: usize,
        outer_iters: usize,
        scale: f32,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_full_tile_f16_probe")?;
        const WK: usize = 48;
        const KROW: usize = 256 + 8;
        let shared = (2 * WK * KROW * 2) as u32;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared)?;
        }
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: shared,
        };
        let iters = outer_iters as i32;
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(&iters).arg(&scale);
        unsafe { b.launch(cfg) }.context("attn_full_tile_f16_probe")?;
        Ok(())
    }

    /// Same per-tile work as [`Self::attn_full_tile_f16_probe`], reordered
    /// so tile `i+1`'s QK^T is issued before tile `i`'s softmax+PV finishes
    /// (a second, independent accumulator set, no register dependency
    /// between them) -- the software-pipelining pattern FlashAttention-3
    /// uses to overlap its SFU-bound softmax with tensor-core MMA, minus
    /// the `wgmma` async completion this GPU's sm_120a doesn't have. See
    /// `attn_full_tile_pipelined_probe` in `ops.cu`.
    pub fn attn_full_tile_pipelined_probe(
        &self,
        out: &mut ViewMut<'_, f32>,
        blocks: usize,
        outer_iters: usize,
        scale: f32,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_full_tile_pipelined_probe")?;
        const WK: usize = 48;
        const KROW: usize = 256 + 8;
        let shared = (2 * WK * KROW * 2) as u32;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared)?;
        }
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: shared,
        };
        let iters = outer_iters as i32;
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(&iters).arg(&scale);
        unsafe { b.launch(cfg) }.context("attn_full_tile_pipelined_probe")?;
        Ok(())
    }

    /// Two physical warps, one doing QK^T/PV-shaped tensor-core busywork
    /// continuously, the other doing a softmax-shaped dependent scalar
    /// chain one tile behind, handed off through shared memory -- real
    /// cross-warp concurrency, unlike [`Self::attn_full_tile_pipelined_probe`]'s
    /// single-warp reordering (found dead: `mma.sync` blocks its own
    /// issuing warp regardless of source order). See
    /// `attn_ws_functional_pingpong_probe` in `ops.cu`.
    pub fn attn_ws_functional_pingpong_probe(&self, out: &mut ViewMut<'_, f32>) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_ws_functional_pingpong_probe")?;
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out);
        unsafe { b.launch(cfg) }.context("attn_ws_functional_pingpong_probe")?;
        Ok(())
    }

    /// Sequential single-warp reference for
    /// [`Self::attn_ws_functional_pingpong_probe`] -- identical arithmetic,
    /// no cross-warp handoff. Its checksum must match exactly.
    pub fn attn_ws_functional_pingpong_sequential_ref(&self, out: &mut ViewMut<'_, f32>) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_ws_functional_pingpong_sequential_ref")?;
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out);
        unsafe { b.launch(cfg) }.context("attn_ws_functional_pingpong_sequential_ref")?;
        Ok(())
    }

    /// Isolates each role's own cost -- see `attn_pp_mma_only_ref` /
    /// `attn_pp_softmax_only_ref` in `ops.cu` for why.
    pub fn attn_pp_mma_only_ref(&self, out: &mut ViewMut<'_, f32>) -> Result<()> {
        let f = self.dev.kernels().get("infero_ops", ops_src(), "attn_pp_mma_only_ref")?;
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out);
        unsafe { b.launch(cfg) }.context("attn_pp_mma_only_ref")?;
        Ok(())
    }

    pub fn attn_pp_softmax_only_ref(&self, out: &mut ViewMut<'_, f32>) -> Result<()> {
        let f = self.dev.kernels().get("infero_ops", ops_src(), "attn_pp_softmax_only_ref")?;
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out);
        unsafe { b.launch(cfg) }.context("attn_pp_softmax_only_ref")?;
        Ok(())
    }

    /// The e4m3-QK^T counterpart of [`Self::attn_full_tile_f16_probe`]; PV
    /// stays `mma_f16` in both, isolating the comparison to QK^T only.
    pub fn attn_full_tile_e4m3_probe(
        &self,
        out: &mut ViewMut<'_, f32>,
        blocks: usize,
        outer_iters: usize,
        scale: f32,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_ops", ops_src(), "attn_full_tile_e4m3_probe")?;
        const WK: usize = 48;
        const KROW: usize = 256 + 8;
        // V stays f16 (2 bytes/elem); K is e4m3 (1 byte/elem).
        let shared = (WK * KROW * 2 + WK * KROW) as u32;
        if shared > 48 * 1024 {
            infero_gpu::set_max_dynamic_shared(&f, shared)?;
        }
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: shared,
        };
        let iters = outer_iters as i32;
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(&iters).arg(&scale);
        unsafe { b.launch(cfg) }.context("attn_full_tile_e4m3_probe")?;
        Ok(())
    }
}

/// The shape parameters every attention kernel needs.
#[derive(Debug, Clone, Copy)]
pub struct AttnDims {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub d_head: usize,
    /// Token slots in the shared KV pool, not a per-sequence limit.
    pub n_slots: usize,
    pub n_tokens: usize,
}

/// Where each token in the batch came from and where its history lives.
///
/// A batch mixes sequences freely: `seq_of` says which sequence a token
/// belongs to, `positions` gives its absolute position within that sequence
/// (which is also its causal mask), and `slot_table` maps a sequence's logical
/// positions onto physical pool slots. Nothing here assumes the tokens are
/// contiguous, ordered, or the same length.
#[derive(Clone, Copy)]
pub struct BatchLayout<'a> {
    pub seq_of: &'a View<'a, i32>,
    pub positions: &'a View<'a, i32>,
    pub slot_table: &'a View<'a, i32>,
    /// Row length of `slot_table`, i.e. the longest sequence it can describe.
    pub table_stride: usize,
}
