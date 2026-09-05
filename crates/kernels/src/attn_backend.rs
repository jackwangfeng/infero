//! A pluggable standard-attention (non-GDN) prefill backend.
//!
//! Two implementations exist: [`InferoHandRolled`], wrapping this crate's own
//! tuned kernels (unchanged), and — behind the `flash_attn2` feature —
//! `crate::flash_attn2::FlashAttn2Ffi`, a torch-free FFI shim around
//! Dao-AILab/flash-attention's CUDA source. See
//! `docs/superpowers/specs/2026-09-05-pluggable-attention-backend-design.md`
//! for the full design and its rationale (GDN is explicitly out of scope,
//! selection is runtime-automatic with no required second env-var gate,
//! backend eligibility for vendor kernels is limited to dense/unquantized KV).

use crate::{AttnDims, BatchLayout, KvQuant, View, ViewMut};
use anyhow::Result;
use half::f16;
use infero_gpu::Device;

/// One-time device capability probe, cheap enough to call at `Model` load
/// time and cache — not per forward call. Wraps [`Device::arch`]/`sm_count`
/// rather than re-querying the driver, since `Device` already resolves both
/// at construction.
#[derive(Debug, Clone, Copy)]
pub struct HardwareCaps {
    /// Compute capability as a two-digit number, e.g. `120` for sm_120a —
    /// same convention as `Device::arch()`.
    pub arch: u32,
    pub sm_count: u32,
}

impl HardwareCaps {
    pub fn probe(dev: &Device) -> Self {
        Self { arch: dev.arch(), sm_count: dev.sm_count() }
    }

    /// `arch >= major*10 + minor`, the same comparison vLLM's own
    /// `DeviceCapability` uses for backend floors (e.g. FA2's real
    /// `>= (8, 0)` floor is `at_least(8, 0)` here).
    pub fn at_least(&self, major: u32, minor: u32) -> bool {
        self.arch >= major * 10 + minor
    }
}

/// The canonical, infero-native call shape every attention backend receives.
/// This is infero's own KV-pool layout, not a lowest-common-denominator one
/// — a backend that wants a different physical layout (e.g. contiguous fp16
/// blocks) repacks internally in its own `prefill`, on its own time budget.
pub struct AttnCallCtx<'a> {
    pub out: &'a mut ViewMut<'a, f32>,
    pub q: &'a View<'a, f32>,
    pub k_cache: &'a View<'a, f16>,
    pub v_cache: &'a View<'a, f16>,
    pub batch: BatchLayout<'a>,
    pub dims: AttnDims,
    pub run_base: usize,
    pub run_tokens: usize,
    pub kv_len: usize,
    pub scale: f32,
    /// `attn_prefill_ws4`'s multi-chunk partial-reduction scratch.
    /// `FlashAttn2Ffi` ignores this (its shim never chunks a run) — it
    /// exists so `InferoHandRolled` can reuse `Model`'s own persistent
    /// `attn_partial` buffer instead of allocating a fresh one per call.
    pub partial: &'a mut ViewMut<'a, f32>,
    /// `InferoHandRolled` ignores this (its kernels reach the stream via its
    /// own `&Kernels` handle) — it exists for `FlashAttn2Ffi`, which has no
    /// `Kernels` of its own to pull one from.
    pub stream: &'a std::sync::Arc<infero_gpu::Stream>,
    /// `InferoHandRolled` ignores this too (it already holds its own
    /// `&Kernels`) — `FlashAttn2Ffi` needs it for one thing this trait
    /// doesn't otherwise expose: converting `q` (infero's real activation
    /// dtype, f32) into the f16 buffer its vendored kernel actually expects,
    /// via the existing `Kernels::to_f16`, rather than duplicating that
    /// kernel or reinterpreting f32 bytes as f16 (which is wrong, not just
    /// imprecise — see `flash_attn2.rs`'s own comment on this field's use).
    pub kern: &'a crate::Kernels,
}

/// One implementation of standard (non-GDN) attention's prefill path.
/// Decode-step attention is out of scope for this trait in this pass — it
/// keeps using infero's existing kernels unconditionally regardless of which
/// backend is selected for prefill.
pub trait AttentionBackend: Send + Sync {
    fn name(&self) -> &'static str;

    /// Lower wins when multiple backends are eligible for the same call.
    fn priority(&self) -> u32;

    /// Checked once at `Model` load time. A backend that returns `true`
    /// here and then fails at call time has a bug in `supports`, not a
    /// normal fallback condition — selection happens once, not per call.
    fn supports(&self, caps: &HardwareCaps, dims: &AttnDims, kv_quant: KvQuant) -> bool;

    fn prefill(&self, ctx: &mut AttnCallCtx<'_>) -> Result<()>;
}

/// Picks the highest-priority backend whose `supports()` returns true.
/// `forced`, when set, must name an eligible backend or this errors loudly
/// rather than silently falling back to a different one — a forced choice
/// that can't run is a config error to surface, not paper over.
pub fn select_backend<'a>(
    backends: &'a [Box<dyn AttentionBackend>],
    caps: &HardwareCaps,
    dims: &AttnDims,
    kv_quant: KvQuant,
    forced: Option<&str>,
) -> Result<&'a dyn AttentionBackend> {
    if let Some(name) = forced {
        return backends
            .iter()
            .find(|b| b.name() == name)
            .filter(|b| b.supports(caps, dims, kv_quant))
            .map(|b| b.as_ref())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "INFERO_ATTN_BACKEND={name} does not support this shape/kv_quant \
                     (dims={dims:?}, kv_quant={kv_quant:?})"
                )
            });
    }
    backends
        .iter()
        .filter(|b| b.supports(caps, dims, kv_quant))
        .min_by_key(|b| b.priority())
        .map(|b| b.as_ref())
        .ok_or_else(|| {
            anyhow::anyhow!("no attention backend supports dims={dims:?} kv_quant={kv_quant:?}")
        })
}

/// Wraps this crate's own tuned kernels behind [`AttentionBackend`],
/// behavior-preserving: identical to `Model::attention()`'s own
/// `prefill_run` branch (the `attn_prefill_decoupled6_f16acc`/
/// `attn_prefill_ws4` choice, including the `INFERO_PREFILL_T6=0` rollback
/// escape hatch). Priority 0 — always wins over any vendor backend that
/// also claims a shape, since the entire point of this crate's own kernels
/// is to beat a generic vendor kernel on hardware they've been tuned for.
pub struct InferoHandRolled<'k> {
    kern: &'k crate::Kernels,
}

impl<'k> InferoHandRolled<'k> {
    pub fn new(kern: &'k crate::Kernels) -> Self {
        Self { kern }
    }
}

impl AttentionBackend for InferoHandRolled<'_> {
    fn name(&self) -> &'static str {
        "handrolled"
    }

    fn priority(&self) -> u32 {
        0
    }

    fn supports(&self, _caps: &HardwareCaps, dims: &AttnDims, _kv_quant: KvQuant) -> bool {
        self.kern.prefill_attention(dims)
    }

    fn prefill(&self, ctx: &mut AttnCallCtx<'_>) -> Result<()> {
        if ctx.dims.d_head == 256 && !std::env::var("INFERO_PREFILL_T6").is_ok_and(|v| v == "0") {
            self.kern.attn_prefill_decoupled6_f16acc(
                ctx.out,
                ctx.q,
                ctx.k_cache,
                ctx.v_cache,
                ctx.batch,
                ctx.dims,
                ctx.run_base,
                ctx.run_tokens,
                ctx.kv_len,
                ctx.scale,
            )
        } else {
            self.kern.attn_prefill_ws4(
                ctx.out,
                ctx.q,
                ctx.k_cache,
                ctx.v_cache,
                ctx.batch,
                ctx.dims,
                ctx.run_base,
                ctx.run_tokens,
                ctx.kv_len,
                ctx.scale,
                ctx.partial,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probes_a_real_device_capability() {
        let dev = Device::new(0).expect("no CUDA device for this test");
        let caps = HardwareCaps::probe(&dev);
        assert!(caps.arch >= 50, "implausible compute capability arch: {}", caps.arch);
        let major = caps.arch / 10;
        let minor = caps.arch % 10;
        assert!(caps.at_least(major, minor));
        assert!(!caps.at_least(major + 1, 0));
    }
}
