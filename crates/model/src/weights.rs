//! Uploading GGUF tensors, to the device or to pinned host memory.
//!
//! A layer is either **resident** — its matrices live in VRAM for the process
//! lifetime — or **offloaded**, in which case its seven big matrices are packed
//! into one page-locked host buffer and DMA'd into a device staging slot just
//! before the layer runs. Compute never leaves the GPU either way; offloading
//! trades PCIe bandwidth for VRAM, not GPU work for CPU work.
//!
//! Norms and biases stay resident regardless. They are a few kilobytes per
//! layer, and streaming them would add descriptors to the transfer without
//! saving anything worth measuring.

use std::sync::Arc;

use anyhow::{Context, Result};
use infero_gpu::{Buf, View};
use infero_gpu::PinnedHostSlice;
use infero_gpu::Device;
use infero_gguf::{GgmlType, Gguf, TensorInfo};
use infero_kernels::WeightType;

use crate::config::Config;
use crate::qwen35_vision::interleaved_mrope_axis;

/// Matrices inside a layer blob start on this boundary, which satisfies every
/// ggml block type's alignment and keeps each sub-copy DMA-friendly.
const BLOB_ALIGN: usize = 256;

/// Cumulative time this process has spent in the FP8 host-side repack step
/// (`fp8::repack_rows`/`pad_rows`) across every tensor `load_awq` has loaded,
/// reported alongside the total load time so a slow load's breakdown is
/// visible in the log rather than needing a profiler to find.
static REPACK_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Same, for the host-to-device upload (`clone_htod`) that follows it.
static UPLOAD_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
// TEMPORARY profiling, not for commit.
static GDN_BLOCK_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ATTN_BLOCK_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FFN_BLOCK_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static STACK_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether FP8 weights load as plain `[n,k]` row-major (`fp8::pad_rows`,
/// CUTLASS's and [`infero_kernels::Kernels::mmv_f8_plain`]'s native layout)
/// instead of [`infero_kernels::fp8::repack_rows`]'s `ROW_GROUP`-interleaved
/// one. Compiles to a hard `false` without the `cutlass` feature: nothing
/// non-CUTLASS reads plain layout, so this must never be true when the
/// kernels that would misread it (`mma_e4m3_block`, `mmv_f8_block`) are the
/// only ones available -- that would silently corrupt every FP8 matmul, not
/// fail loudly.
#[cfg(feature = "cutlass")]
pub(crate) fn fp8_unified_layout() -> bool {
    static U: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *U.get_or_init(|| std::env::var("INFERO_FP8_UNIFIED").as_deref() == Ok("1"))
}
#[cfg(not(feature = "cutlass"))]
pub(crate) fn fp8_unified_layout() -> bool {
    false
}

/// Where a matrix's bytes live.
enum Storage {
    /// In VRAM, for the process lifetime.
    Device(Buf<u8>),
    /// A window into the checkpoint file, aliased into device memory.
    ///
    /// Unified memory only, and it is the difference between an 18 GiB
    /// checkpoint taking 24 s to load and taking none: the pages the GPU reads
    /// *are* the file's pages, so there is nothing to upload. The `Arc` is what
    /// keeps the mapping alive -- one per checkpoint, shared by every matrix in
    /// it.
    Mapped { file: Arc<Buf<u8>>, offset: usize },
    /// In the owning layer's host blob, at this byte offset. The same offset
    /// addresses it inside the staging buffer once the layer is transferred.
    Streamed { offset: usize },
}

/// Holds a lazily-built [`infero_kernels::CutlassWeight`] for a resident
/// matrix, off the `cutlass` feature. `()` when the feature is off, so every
/// `Matrix { .. }` literal can write `cutlass_weight: Default::default()`
/// without a `#[cfg]` at each of the several construction sites.
#[cfg(feature = "cutlass")]
type CutlassSlot = std::sync::OnceLock<Option<infero_kernels::CutlassWeight>>;
#[cfg(not(feature = "cutlass"))]
type CutlassSlot = ();

/// A 2-D weight matrix, still in its GGUF block encoding.
pub struct Matrix {
    pub ty: WeightType,
    /// Elements per row (ggml `ne0`), the contraction dimension.
    pub k: usize,
    /// Number of rows (ggml `ne1`), the output dimension.
    pub n: usize,
    pub n_bytes: usize,
    storage: Storage,
    cutlass_weight: CutlassSlot,
}

impl Matrix {
    pub fn elements(&self) -> usize {
        self.k * self.n
    }

    pub fn is_resident(&self) -> bool {
        matches!(self.storage, Storage::Device(_))
    }

    /// This matrix's [`infero_kernels::CutlassWeight`], built and cached on
    /// first use. Only for resident, [`WeightType::F8E4M3`] matrices --
    /// `None` otherwise, including for offloaded ones: their bytes live in a
    /// per-layer staging buffer that a different layer's DMA overwrites
    /// between calls, so caching a repack of it would go stale silently. The
    /// repack itself is a one-time `O(n*k)` cost the same order as reading
    /// the matrix once, negligible next to loading the checkpoint.
    #[cfg(feature = "cutlass")]
    pub fn cutlass_weight(&self, kern: &infero_kernels::Kernels) -> Option<&infero_kernels::CutlassWeight> {
        if self.ty != WeightType::F8E4M3 || !self.is_resident() {
            return None;
        }
        self.cutlass_weight
            .get_or_init(|| {
                let view = self.view(None).ok()?;
                kern.prepare_cutlass_weight(&view, self.k, self.n, fp8_unified_layout()).ok()
            })
            .as_ref()
    }

    /// A device view of this matrix.
    ///
    /// `stage` must be the staging buffer currently holding this matrix's
    /// layer, and is unused for a resident matrix.
    pub fn view<'a>(&'a self, stage: Option<&'a Buf<u8>>) -> Result<View<'a, u8>> {
        match &self.storage {
            Storage::Device(d) => Ok(d.as_view()),
            Storage::Mapped { file, offset } => {
                anyhow::ensure!(
                    offset + self.n_bytes <= file.len(),
                    "matrix wants {}..{} of a {}-byte checkpoint",
                    offset,
                    offset + self.n_bytes,
                    file.len()
                );
                Ok(file.slice(*offset..offset + self.n_bytes))
            }
            Storage::Streamed { offset } => {
                let stage =
                    stage.context("an offloaded matrix was used without its layer being staged")?;
                anyhow::ensure!(
                    offset + self.n_bytes <= stage.len(),
                    "staging buffer holds {} bytes, matrix wants {}..{}",
                    stage.len(),
                    offset,
                    offset + self.n_bytes
                );
                Ok(stage.slice(*offset..offset + self.n_bytes))
            }
        }
    }
}

/// One projection of every expert in one layer, concatenated.
///
/// Expert `e` starts at `e * stride` and every expert has the same shape, so a
/// kernel reaches its weights by offsetting a single base pointer. Three of
/// these hold a sparse FFN's whole layer where the checkpoint spells it as
/// `n_experts * 3` separate tensors — 384 device allocations a layer on
/// Qwen3-30B-A3B, against three.
///
/// The layout is deliberately the one GGUF already uses for `*_exps` tensors
/// (`[k, n, n_expert]`, one tensor per projection per layer), so a GGUF reader
/// fills this the same way and neither the kernels nor the forward pass learn
/// that a second checkpoint format exists.
pub struct Experts {
    pub ty: WeightType,
    /// The contraction dimension of *one* expert.
    pub k: usize,
    /// The output dimension of *one* expert.
    pub n: usize,
    pub n_experts: usize,
    /// Bytes per expert, which is also the offset multiplier.
    pub stride: usize,
    storage: Storage,
}

impl Experts {
    pub fn n_bytes(&self) -> usize {
        self.stride * self.n_experts
    }

    /// A device view of the whole block, for a kernel that indexes experts
    /// itself. `stride` tells it how far apart they are.
    pub fn view<'a>(&'a self, stage: Option<&'a Buf<u8>>) -> Result<View<'a, u8>> {
        match &self.storage {
            Storage::Device(d) => Ok(d.as_view()),
            // No loader ever puts an expert block here today — `load_awq`'s
            // `expert_projection` always uploads to `Storage::Device`, and
            // there is no GGUF `_exps` reader yet to alias from a mapped
            // checkpoint — but `Storage` is the same enum `Matrix` uses, so
            // this match has to be exhaustive over what the type allows, not
            // just what today's loaders produce.
            Storage::Mapped { file, offset } => {
                let end = offset + self.n_bytes();
                anyhow::ensure!(
                    end <= file.len(),
                    "expert block wants {}..{} of a {}-byte checkpoint",
                    offset,
                    end,
                    file.len()
                );
                Ok(file.slice(*offset..end))
            }
            Storage::Streamed { offset } => {
                let stage = stage
                    .context("an offloaded expert block was used without its layer being staged")?;
                let end = offset + self.n_bytes();
                anyhow::ensure!(
                    end <= stage.len(),
                    "staging buffer holds {} bytes, expert block wants {}..{}",
                    stage.len(),
                    offset,
                    end
                );
                Ok(stage.slice(*offset..end))
            }
        }
    }

    /// A device view of one expert, for the per-expert GEMM the prefill path
    /// loops over.
    pub fn view_of<'a>(
        &'a self,
        e: usize,
        stage: Option<&'a Buf<u8>>,
    ) -> Result<View<'a, u8>> {
        anyhow::ensure!(
            e < self.n_experts,
            "expert {e} of {}",
            self.n_experts
        );
        let whole = self.view(stage)?;
        let at = e * self.stride;
        Ok(whole.slice(at..at + self.stride))
    }
}

/// The sparse half of a block: a router and the experts it selects between.
pub struct MoeWeights {
    /// `mlp.gate`. Excluded from quantization in Qwen3-MoE's AWQ export, so it
    /// arrives as f16 and stays that way — it is `[n_experts, d_model]`, which
    /// is 128 rows here and not worth a quantized kernel.
    pub router: Matrix,
    pub gate: Experts,
    pub up: Experts,
    pub down: Experts,
}

/// A 1-D parameter — norm gains and biases — always held as f32 on the device.
pub type Vector = Buf<f32>;

/// The host-side table for [`Weights::mrope_axis`], uploaded by both loaders.
///
/// `rotary_dim / 2` is `0` only when `rotary_dim` is `0`, which the config
/// loaders already refuse, but `mrope_axis.len() >= rotary_dim / 2` is a
/// buffer-size invariant `rope_qk_partial` enforces at every call, and Metal
/// takes no null buffers -- so this is `.max(1)` rather than exactly zero for
/// a model with no rotary width at all (there are none today, but nothing
/// upstream promises there won't be).
pub(crate) fn mrope_axis_table(rotary_dim: usize, section: Option<[usize; 3]>) -> Vec<i32> {
    let half = (rotary_dim / 2).max(1);
    match section {
        Some(s) => (0..half).map(|i| interleaved_mrope_axis(i, s) as i32).collect(),
        None => vec![0i32; half],
    }
}

impl Matrix {
    /// Upload `[n, k]` f16 values, row-major, as a resident matrix.
    ///
    /// For a caller that has the numbers rather than a checkpoint: the reference
    /// weights a capture carries, chiefly. The loaders build their matrices
    /// through the private paths above and do not go through this.
    pub fn upload_f16(dev: &Device, halves: &[half::f16], k: usize, n: usize) -> Result<Self> {
        anyhow::ensure!(
            halves.len() == k * n,
            "{} halves for a [{n}, {k}] matrix",
            halves.len()
        );
        // Safety: f16 is a transparent u16, so these are already the
        // little-endian halves the device expects, and the view does not outlive
        // `halves`.
        let raw =
            unsafe { std::slice::from_raw_parts(halves.as_ptr() as *const u8, halves.len() * 2) };
        Ok(Self {
            ty: WeightType::F16,
            k,
            n,
            n_bytes: raw.len(),
            storage: Storage::Device(dev.stream().clone_htod(raw)?),
                cutlass_weight: Default::default(),
        })
    }
}

/// One offloaded layer's matrices, packed contiguously in page-locked memory.
///
/// One blob per layer means one DMA per layer: the transfer the prefetch has to
/// hide is a single large contiguous copy rather than seven scattered ones.
pub struct LayerBlob {
    host: PinnedHostSlice<u8>,
    pub bytes: usize,
}

impl LayerBlob {
    pub fn host(&self) -> &PinnedHostSlice<u8> {
        &self.host
    }
}

/// The softmax-attention half of a block.
///
/// Its own struct because Qwen3.5 has blocks that do not have one: 48 of its 64
/// layers mix with a recurrence instead. Before that every model infero loaded
/// had exactly one kind of block, so these fields sat directly on `Layer` and
/// the forward pass could reach them unconditionally.
pub struct AttnWeights {
    pub wq: Matrix,
    pub wk: Matrix,
    pub wv: Matrix,
    pub wo: Matrix,
    pub bq: Option<Vector>,
    pub bk: Option<Vector>,
    pub bv: Option<Vector>,
    pub bo: Option<Vector>,
    /// Qwen3 normalizes each head of `q` and `k` with its own learned
    /// `[d_head]` weight, before the rotary. Absent on llama and qwen2, where
    /// the attention biases play the role these replaced.
    pub q_norm: Option<Vector>,
    pub k_norm: Option<Vector>,
    /// `q`, `k` and `v` stacked along `n`, under `INFERO_FUSE_FFN`. One matmul
    /// and a scatter instead of three; see `stacked` in `load_awq`.
    pub w_qkv: Option<Matrix>,
    /// `k` and `v` stacked along `n`, GGUF-only: `w_qkv` needs `wq` the same
    /// width as `wk`/`wv`, which a gated `wq` (`output_gate`, Qwen3.5) never
    /// is, but `wk` and `wv` are always the same shape as each other
    /// regardless. One matmul instead of two on the pair `output_gate`
    /// otherwise leaves fully unfused. See `stacked2_gguf`.
    pub w_kv: Option<Matrix>,
    /// True when `wq` produces `2 * d_attn` columns: a query and a gate
    /// interleaved per head, which Qwen3.5's attention blocks carry and
    /// nothing before them did.
    pub output_gate: bool,
}

/// The GatedDeltaNet half of a block.
///
/// Field names follow the checkpoint's tensor names rather than being
/// translated, because the mapping is the thing most likely to be got wrong and
/// a reader should be able to check it against `notes/qwen3.5-architecture.md`
/// without a glossary.
pub struct GdnWeights {
    /// `[d_model, 2 * key_dim + value_dim]` — q, k and v in one projection.
    pub in_proj_qkv: Matrix,
    /// `[d_model, value_dim]` — the output gate, value-shaped.
    pub in_proj_z: Matrix,
    /// `[d_model, value_heads]` — the per-head decay input.
    pub in_proj_a: Matrix,
    /// `[d_model, value_heads]` — the per-head write strength.
    pub in_proj_b: Matrix,
    /// `in_proj_qkv` and `in_proj_z` stacked along the output, when both are
    /// FP8 over the same `k` and land on a whole [`infero_kernels::fp8::FP8_BLOCK`]
    /// each — every GQA shape seen so far. One launch instead of two on a pair
    /// that, run separately, are 640 and 384 blocks against 188 SMs: 56% and
    /// 34% achieved occupancy by `ncu`, neither register- nor shared-memory-
    /// limited (`ncu --set full` put both at 100% theoretical) — just not
    /// enough blocks to fill the device for the kernel's whole duration.
    /// Stacked, the pair is 1024 blocks, matching the FFN's own ~82-84%.
    pub in_proj_qz: Option<Matrix>,
    /// `in_proj_a` and `in_proj_b` stacked along the output, when they are the
    /// same dense type. One launch instead of two on matrices whose cost is all
    /// launch. `a` occupies columns `[0, value_heads)` of each row and `b` the
    /// rest, which is why the gate kernel takes a stride.
    pub in_proj_ba: Option<Matrix>,
    /// `[conv_channels, conv_k]`, depthwise, no bias.
    pub conv1d: Vector,
    pub a_log: Vector,
    pub dt_bias: Vector,
    /// `[value_head_dim]`, the gated RMSNorm's gain.
    pub norm: Vector,
    /// `[value_dim, d_model]`.
    pub out_proj: Matrix,
}

pub struct Layer {
    pub attn_norm: Vector,
    /// The mixer. Exactly one of these is `Some`; which one is decided by the
    /// checkpoint's `layer_types`, not by a stride, because Qwen3.5 states the
    /// pattern explicitly and a future model need not repeat every fourth.
    pub attn: Option<AttnWeights>,
    pub gdn: Option<GdnWeights>,
    pub ffn_norm: Vector,
    /// The feed-forward half. Exactly one of these is `Some`, decided by
    /// `MoeConfig::is_sparse` — grouped rather than left as five optional
    /// matrices because "some of the dense FFN and some of the sparse one" is
    /// not a layer any checkpoint describes, and three separate `Option`s that
    /// must agree is the shape this file already warns about elsewhere.
    pub dense: Option<DenseFfn>,
    pub moe: Option<MoeWeights>,
    /// Present when this layer's matrices are streamed rather than resident.
    pub blob: Option<LayerBlob>,
}

/// The dense feed-forward half of a block.
pub struct DenseFfn {
    pub w_gate: Matrix,
    pub w_up: Matrix,
    pub w_down: Matrix,
    /// `gate` and `up` stacked along `n`, under `INFERO_FUSE_FFN`. One matmul
    /// instead of two; see `stacked` in `load_awq`.
    pub w_gate_up: Option<Matrix>,
}

impl Layer {
    pub fn is_offloaded(&self) -> bool {
        self.blob.is_some()
    }

    /// The dense feed-forward half, for a layer the caller has established has
    /// one.
    ///
    /// Panics for the same reason [`Self::attn`] does: reaching here on a
    /// sparse layer means the FFN dispatch is wrong, which is a bug in the
    /// caller rather than a condition to recover from.
    pub fn dense(&self) -> &DenseFfn {
        self.dense
            .as_ref()
            .expect("this layer's FFN is a mixture of experts, not a dense one")
    }

    /// The attention half, for a layer the caller has already established has
    /// one.
    ///
    /// Panics rather than returning an error: reaching here on a
    /// linear-attention layer means the block dispatch is wrong, which is a bug
    /// in this file's caller and not a condition to recover from. The message
    /// names the layer kind so the dispatch is the first place to look.
    pub fn attn(&self) -> &AttnWeights {
        self.attn.as_ref().expect(
            "asked for the attention weights of a linear-attention layer; \
             the block dispatch did not consult Layer::is_linear",
        )
    }

    /// True for a GatedDeltaNet block.
    pub fn is_linear(&self) -> bool {
        self.gdn.is_some()
    }
}

/// The multi-token-prediction head, when a checkpoint carries one.
///
/// Shaped like a [`Layer`] plus four tensors of glue, which is what it is: one
/// full-attention decoder block, the same kind as the text model's layers 3, 7,
/// …, 63, wrapped in `fc` and three norms. See `crates/model/src/qwen35_mtp.rs`
/// for the host reference this is the device counterpart of, and
/// `notes/qwen3.5-mtp.md` for why each of these is the tensor it is.
///
/// Deliberately *not* holding an embedding or a vocabulary projection: the head
/// borrows the text model's. That is what `mtp_use_dedicated_embeddings = false`
/// means, and it is what the checkpoint shows by shipping no `mtp.embed_tokens`
/// and no `mtp.lm_head`. [`load_mtp`] checks both statements and refuses when
/// they disagree, rather than trusting the config over the tensors or the other
/// way round.
pub struct MtpWeights {
    /// `[2 * d_model, d_model]` — `k = 2 * d_model`, so the low half of every
    /// row multiplies the **embedding** and the high half the hidden state.
    pub fc: Matrix,
    /// Applies to the token embedding.
    pub pre_fc_norm_embedding: Vector,
    /// Applies to the text model's final hidden state.
    pub pre_fc_norm_hidden: Vector,
    /// The head's own final norm — a different tensor from the text model's
    /// `model.language_model.norm`.
    pub norm: Vector,
    /// The one decoder layer, in the same shape the text model's blocks use so
    /// that the forward pass can be read against `Model::attention`.
    pub layer: Layer,
    pub device_bytes: usize,
}

pub struct Weights {
    pub token_embd: Matrix,
    pub layers: Vec<Layer>,
    pub output_norm: Vector,
    /// Absent when the model ties the output projection to the embeddings.
    pub output: Option<Matrix>,
    /// The same matrix in [`WeightType::Q8_0S`], for the batched path.
    ///
    /// Only the AWQ loader builds it, because only there does infero choose the
    /// vocab projection's layout. Held *as well as* `output`: the batch-1
    /// mat-vec reads the packed form, and teaching it the split one is a
    /// separate change from proving the split one is faster. 532 MiB on an 8B
    /// model, so it is gated on there being room.
    pub output_split: Option<Matrix>,
    /// Per-dimension RoPE frequency divisors, `d_head / 2` of them. All ones
    /// unless the file carries `rope_freqs.weight`.
    pub rope_freqs: Vector,
    /// Which of a token's `pos_stride` position values each of the
    /// `rotary_dim / 2` rope frequencies reads. All zeros -- read the one
    /// scalar position every model before Qwen3.5 has -- unless
    /// `cfg.mrope_section` is set, in which case this is
    /// `interleaved_mrope_axis(i, section)` per frequency `i`. See
    /// `Kernels::rope_qk_partial`'s doc comment and `notes/mrope-and-video.md`.
    pub mrope_axis: Buf<i32>,
    /// Weight bytes held in VRAM.
    pub device_bytes: usize,
    /// Weight bytes held in page-locked host memory.
    pub host_bytes: usize,
    /// Largest single layer blob, which sizes the staging buffers.
    pub max_blob_bytes: usize,
}

impl Weights {
    /// Load with the first `n_gpu_layers` blocks resident and the rest
    /// offloaded. Embeddings, the output projection and all norms stay
    /// resident: the vocab projection is touched once per token and the norms
    /// are negligible.
    pub fn load(dev: &Device, f: &Gguf, cfg: &Config, n_gpu_layers: usize) -> Result<Self> {
        Self::load_sharded(dev, f, cfg, n_gpu_layers, None)
    }

    /// [`Self::load`] plus optional tensor-parallel sharding: `(tp_rank,
    /// tp_size)`, `None` meaning today's exact unsharded behavior. `cfg` is
    /// expected to already be sharded (`Config::shard_for_tp` called before
    /// this) -- this function shards the WEIGHT BYTES to match; the two
    /// disagreeing (e.g. `cfg.n_heads` already halved but the Q/K/V weights
    /// read in full) would silently build a `Matrix` shaped for the wrong
    /// number of heads.
    ///
    /// Scoped to the non-linear-attention (standard Q/K/V/O + gate/up/down)
    /// resident-layer path for this pass -- a GDN (`is_linear`) block's own
    /// sharding needs the same treatment but isn't exercised by the current
    /// tensor-parallel validation target and isn't implemented here yet; an
    /// offloaded (`i >= n_gpu_layers`) layer under sharding fails loudly
    /// rather than silently loading unsharded weights, for the same reason.
    pub fn load_sharded(
        dev: &Device,
        f: &Gguf,
        cfg: &Config,
        n_gpu_layers: usize,
        shard: Option<(usize, usize)>,
    ) -> Result<Self> {
        let started = std::time::Instant::now();
        let n_gpu_layers = n_gpu_layers.min(cfg.n_layers);
        let mut device_bytes = 0usize;
        let mut host_bytes = 0usize;
        let mut max_blob_bytes = 0usize;

        // Ask the backend whether it can alias the file instead of copying out
        // of it. Unified memory can; a discrete card cannot, and answers `None`
        // rather than failing. Mapped once here and shared by every matrix, so
        // the mapping's lifetime is the weights'.
        //
        // The offloaded path is untouched: a streamed layer is copied into
        // page-locked host memory by `pack_layer`, which is a different question
        // from where the resident ones live.
        let mapped = infero_gpu::map_file(dev, f.path())?.map(Arc::new);
        let mapped = mapped.as_ref();
        if mapped.is_some() {
            tracing::info!("checkpoint aliased into device memory; no upload");
        }

        let token_embd = upload_matrix(dev, f, mapped, "token_embd.weight", &mut device_bytes)?;
        let output_norm = upload_vector(dev, f, "output_norm.weight", &mut device_bytes)?;
        let output = if cfg.tied_embeddings {
            None
        } else {
            Some(upload_matrix(dev, f, mapped, "output.weight", &mut device_bytes)?)
        };

        // Llama 3.1 ships these precomputed; everything else wants no scaling.
        let rope_freqs = match f.get_tensor("rope_freqs.weight") {
            Some(info) if info.n_elements == cfg.d_head / 2 => {
                tracing::info!(dims = info.n_elements, "using rope frequency scaling");
                upload_vector(dev, f, "rope_freqs.weight", &mut device_bytes)?
            }
            Some(info) => {
                anyhow::bail!(
                    "rope_freqs.weight has {} entries, expected d_head/2 = {}",
                    info.n_elements,
                    cfg.d_head / 2
                );
            }
            None => dev.stream().clone_htod(&vec![1.0f32; cfg.d_head / 2])?,
        };
        // No GGUF conversion carries a Qwen3.5 vision tower (`cfg.mrope_section`
        // is always `None` from this loader -- see `Config::from_gguf`), so
        // this is always the all-scalar table.
        let mrope_axis = dev
            .stream()
            .clone_htod(&mrope_axis_table(cfg.rotary_dim, cfg.mrope_section))?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let t = |s: &str| format!("blk.{i}.{s}");
            // Which mixer this block has is read off the file rather than
            // inferred from an interval: Qwen3.5 states an explicit 64-element
            // `layer_types`, and a block that carries `ssm_a` is a linear one
            // whatever a stride would say.
            let is_linear = f.get_tensor(&t("ssm_a")).is_some();
            let names: Vec<String> = if is_linear {
                vec![
                    t("attn_qkv.weight"),
                    t("attn_gate.weight"),
                    t("ssm_alpha.weight"),
                    t("ssm_beta.weight"),
                    t("ssm_out.weight"),
                    t("ffn_gate.weight"),
                    t("ffn_up.weight"),
                    t("ffn_down.weight"),
                ]
            } else {
                vec![
                    t("attn_q.weight"),
                    t("attn_k.weight"),
                    t("attn_v.weight"),
                    t("attn_output.weight"),
                    t("ffn_gate.weight"),
                    t("ffn_up.weight"),
                    t("ffn_down.weight"),
                ]
            };
            // llama.cpp names the pre-FFN norm `ffn_norm` for a llama-style
            // block and `post_attention_norm` for Qwen3.5. The same norm in the
            // same place; only the spelling moved.
            let ffn_norm_name = if f.get_tensor(&t("ffn_norm.weight")).is_some() {
                t("ffn_norm.weight")
            } else {
                t("post_attention_norm.weight")
            };

            let (matrices, blob) = if i < n_gpu_layers {
                let mut m = Vec::with_capacity(names.len());
                match (shard, is_linear) {
                    (None, _) => {
                        for name in &names {
                            m.push(upload_matrix(dev, f, mapped, name, &mut device_bytes)?);
                        }
                    }
                    (Some((tp_rank, tp_size)), true) => {
                        // GDN's own head-count fields are already divided by
                        // `tp_size` (`Config::shard_for_tp` runs once, before
                        // any weight loads) -- `key_dim()`/`value_dim()` below
                        // are each rank's own (post-shard) width, and the
                        // *full* on-disk segment width is that times
                        // `tp_size`. Q and K share `key_heads`; V uses
                        // `value_heads`, a different ratio (GQA-style) -- a
                        // single uniform row-range across the combined
                        // `[q|k|v]` tensor would not land on head
                        // boundaries, so each segment is sharded
                        // independently by its own head count.
                        //
                        // The V segment additionally needs `v_heads_tiled`
                        // handling: for a real GGUF checkpoint (confirmed on
                        // `Qwen3.8-27B-Q8_0.gguf`), value heads are stored
                        // tiled -- `[G0_v0, G1_v0, ..., G0_v1, G1_v1, ...]`,
                        // value head `h`'s real row is `h % key_heads_full`,
                        // not `h` -- so a plain contiguous row-range (the
                        // first version of this code) silently mixed value
                        // heads belonging to OTHER ranks' key heads into
                        // this rank's shard while dropping some of its own.
                        // Confirmed live: this produced coherent-but-not-
                        // bit-exact TP=2 output on the real 27B checkpoint,
                        // not a crash -- see `shard_tiled_value_rows`'s own
                        // doc comment. Q/K rows have no such tiling (key
                        // heads are the thing being tiled *by*, not the
                        // thing tiled), so they stay a plain contiguous
                        // shard.
                        let la = cfg.linear_attn.as_ref().with_context(|| {
                            format!("layer {i}: GDN block but Config has no linear_attn")
                        })?;
                        let kd = la.key_dim();
                        let vd = la.value_dim();
                        let full_kd = kd * tp_size;
                        let key_heads_full = la.key_heads * tp_size;
                        let heads_per_key = la.heads_per_key();
                        let vhd = la.value_head_dim;

                        let shard3 = |info: &TensorInfo| -> Result<Vec<u8>> {
                            let mut bytes =
                                f.tensor_shard(info, (tp_rank * kd)..((tp_rank + 1) * kd))?;
                            bytes.extend(f.tensor_shard(
                                info,
                                (full_kd + tp_rank * kd)..(full_kd + (tp_rank + 1) * kd),
                            )?);
                            bytes.extend(shard_tiled_value_rows(
                                f, info, key_heads_full, heads_per_key, vhd, 2 * full_kd,
                                tp_rank, tp_size,
                            )?);
                            Ok(bytes)
                        };
                        let upload_bytes = |bytes: Vec<u8>,
                                             ty: WeightType,
                                             k: usize,
                                             n: usize,
                                             total: &mut usize|
                         -> Result<Matrix> {
                            let n_bytes = bytes.len();
                            *total += n_bytes;
                            Ok(Matrix {
                                ty,
                                k,
                                n,
                                n_bytes,
                                storage: Storage::Device(dev.stream().clone_htod(&bytes)?),
                                cutlass_weight: Default::default(),
                            })
                        };
                        let upload_value_rows = |name: &str, row_span: usize, total: &mut usize| -> Result<Matrix> {
                            let info = f.tensor(name)?.clone();
                            let ty = WeightType::from_ggml(info.ty).with_context(|| format!("tensor {name}"))?;
                            // `dims[0]` is ggml's fastest axis == in_features
                            // (`k`) for any linear-style weight, regardless
                            // of which axis *we* shard -- unsharded here,
                            // only the value-head output axis (`n`) is.
                            let full_k = info.dims[0] as usize;
                            let bytes = shard_tiled_value_rows(
                                f, &info, key_heads_full, heads_per_key, row_span, 0, tp_rank, tp_size,
                            )?;
                            let n_bytes = bytes.len();
                            *total += n_bytes;
                            Ok(Matrix {
                                ty,
                                k: full_k,
                                n: la.value_heads * row_span,
                                n_bytes,
                                storage: Storage::Device(dev.stream().clone_htod(&bytes)?),
                                cutlass_weight: Default::default(),
                            })
                        };

                        // 0: attn_qkv.weight -- three-segment shard above.
                        let qkv_info = f.tensor(&names[0])?.clone();
                        let qkv_ty = WeightType::from_ggml(qkv_info.ty)
                            .with_context(|| format!("tensor {}", names[0]))?;
                        let d_model = qkv_info.dims[0] as usize;
                        m.push(upload_bytes(
                            shard3(&qkv_info)?,
                            qkv_ty,
                            d_model,
                            2 * kd + vd,
                            &mut device_bytes,
                        )?);
                        // 1: attn_gate.weight -- value-shaped, single segment,
                        // `v_heads_tiled`-aware (row_span = value_head_dim).
                        m.push(upload_value_rows(&names[1], vhd, &mut device_bytes)?);
                        // 2, 3: ssm_alpha.weight / ssm_beta.weight -- one
                        // scalar a head (`n = value_heads`, not `value_dim`),
                        // `v_heads_tiled`-aware (row_span = 1: one row a head,
                        // not `head_dim` rows, unlike the gate/qkv tensors
                        // above).
                        m.push(upload_value_rows(&names[2], 1, &mut device_bytes)?);
                        m.push(upload_value_rows(&names[3], 1, &mut device_bytes)?);
                        // 4: ssm_out.weight -- row-parallel (input =
                        // value_dim), `v_heads_tiled`-aware column sharding.
                        {
                            let info = f.tensor(&names[4])?.clone();
                            let ty = WeightType::from_ggml(info.ty)
                                .with_context(|| format!("tensor {}", names[4]))?;
                            let full_n = info.shape()[0] as usize;
                            let bytes = shard_tiled_value_cols(
                                &f, &info, key_heads_full, heads_per_key, vhd, tp_rank, tp_size,
                            )?;
                            let n_bytes = bytes.len();
                            device_bytes += n_bytes;
                            m.push(Matrix {
                                ty,
                                k: vd,
                                n: full_n,
                                n_bytes,
                                storage: Storage::Device(dev.stream().clone_htod(&bytes)?),
                                cutlass_weight: Default::default(),
                            });
                        }
                        // 5, 6, 7: dense FFN gate/up/down -- ordinary
                        // Megatron-style sharding, identical to the
                        // standard-attention branch's own FFN handling (no
                        // value-head tiling concern -- FFN's hidden
                        // dimension isn't head-shaped at all).
                        for (name, axis) in names[5..8]
                            .iter()
                            .zip([ShardAxis::Output, ShardAxis::Output, ShardAxis::Input])
                        {
                            m.push(upload_matrix_sharded(
                                dev, f, name, axis, tp_rank, tp_size, &mut device_bytes,
                            )?);
                        }
                    }
                    (Some((tp_rank, tp_size)), false) => {
                        // Standard non-linear block: [q, k, v, o, gate, up, down].
                        // Column-parallel (shard the output dim, `n`): q/k/v/gate/up.
                        // Row-parallel (shard the input dim, `k`): o/down.
                        use ShardAxis::{Input, Output};
                        let axes = [Output, Output, Output, Input, Output, Output, Input];
                        anyhow::ensure!(
                            names.len() == axes.len(),
                            "layer {i}: expected {} tensors for the standard-attention shard \
                             pattern, got {}",
                            axes.len(),
                            names.len()
                        );
                        for (name, axis) in names.iter().zip(axes) {
                            m.push(upload_matrix_sharded(
                                dev,
                                f,
                                name,
                                axis,
                                tp_rank,
                                tp_size,
                                &mut device_bytes,
                            )?);
                        }
                    }
                }
                (m, None)
            } else if shard.is_some() {
                anyhow::bail!(
                    "tensor-parallel sharding for an offloaded layer ({i} >= n_gpu_layers={n_gpu_layers}) \
                     is not yet implemented"
                );
            } else {
                let (m, blob) = pack_layer(dev, f, &names)
                    .with_context(|| format!("packing layer {i} into host memory"))?;
                host_bytes += blob.bytes;
                max_blob_bytes = max_blob_bytes.max(blob.bytes);
                (m, Some(blob))
            };
            let mut matrices = matrices.into_iter();

            let (attn, gdn) = if is_linear {
                let qkv = matrices.next().unwrap();
                let z = matrices.next().unwrap();
                let a = matrices.next().unwrap();
                let b = matrices.next().unwrap();
                let out = matrices.next().unwrap();
                (
                    None,
                    Some(GdnWeights {
                        in_proj_qkv: qkv,
                        in_proj_z: z,
                        in_proj_a: a,
                        in_proj_b: b,
                        // `a` and `b` are `value_heads` rows each -- 48 against
                        // a 5120 contraction -- so unstacked they are two
                        // gemv launches whose bytes are a rounding error next
                        // to their dispatch cost. See `stacked2_gguf`.
                        // Disabled under sharding, same precedent as `w_kv`
                        // (`dde80b3`): this fusion reads both tensors' bytes
                        // at their *full*, unsharded width -- it has no
                        // `shard` parameter and never went through the
                        // per-rank row-range path above, so fusing it here
                        // would silently apply a rank's kernel to a
                        // doubled-width buffer sized for the correctly
                        // sharded `in_proj_a`/`in_proj_b` pair.
                        in_proj_ba: if shard.is_some() {
                            None
                        } else {
                            stacked2_gguf(
                                dev, f, &t("ssm_alpha.weight"), &t("ssm_beta.weight"),
                                &mut device_bytes)?
                        },
                        // Tried stacking these with `stacked2_gguf`, the same
                        // trick `in_proj_ba` uses -- `qkv` and `z` are the
                        // same Q8_0 type over the same `k`, and `split2`
                        // does not require equal widths. Produced complete
                        // garbage output and, root-caused after also hitting
                        // it on `w_gate_up` (see its own comment in this
                        // file): not a logic bug in the stacking or the
                        // split at all, but VRAM. `stacked2_gguf` copies the
                        // two tensors' bytes host-side and re-uploads the
                        // concatenation as a new `Storage::Device` buffer --
                        // real, additional VRAM, unlike the checkpoint's own
                        // `Storage::Mapped` aliasing, which costs nothing
                        // because the GPU reads the file's own pages. `a`/`b`
                        // are 48 rows each and the fused pair across all
                        // forty-eight GDN layers is under 300 KiB total, so
                        // this never mattered there. `qkv`/`z` are 10240 and
                        // 6144 rows: 89 MiB a layer, 4.3 GiB across the
                        // model. This machine had 6.4 GiB free after the
                        // checkpoint's own mapping; that allocation nearly
                        // exhausted it, and the pool sizer that runs next
                        // shrank the KV cache to fit what was left -- 4096
                        // slots, kv_mib=1024, against the unmodified
                        // baseline's 19968 slots and kv_mib=4992 -- small
                        // enough to wrap or starve mid-generation, which is
                        // what actually produced the garbage. Not a mechanism
                        // this loader can use for anything wider than a few
                        // hundred rows without giving the VRAM back some
                        // other way first.
                        in_proj_qz: None,
                        conv1d: match shard {
                            None => upload_vector(dev, f, &t("ssm_conv1d.weight"), &mut device_bytes)?,
                            Some((tp_rank, tp_size)) => {
                                // Same three-segment [q|k|v] channel layout
                                // as `attn_qkv.weight` above (`conv_channels
                                // == 2*key_dim + value_dim`), just with
                                // `conv_k` (kernel taps) as the fast/inner
                                // axis instead of `d_model`. Q/K rows stay a
                                // plain contiguous shard; the V segment needs
                                // the same `v_heads_tiled` tiling as
                                // `attn_qkv.weight`'s V segment (see
                                // `shard_tiled_value_rows`'s doc comment --
                                // this exact bug was live on the real 27B
                                // checkpoint, found post-hoc, not by review).
                                let la = cfg.linear_attn.as_ref().unwrap();
                                let kd = la.key_dim();
                                let full_kd = kd * tp_size;
                                let key_heads_full = la.key_heads * tp_size;
                                let heads_per_key = la.heads_per_key();
                                let info = f.tensor(&t("ssm_conv1d.weight"))?.clone();
                                let mut bytes =
                                    f.tensor_shard(&info, (tp_rank * kd)..((tp_rank + 1) * kd))?;
                                bytes.extend(f.tensor_shard(
                                    &info,
                                    (full_kd + tp_rank * kd)..(full_kd + (tp_rank + 1) * kd),
                                )?);
                                bytes.extend(shard_tiled_value_rows(
                                    f, &info, key_heads_full, heads_per_key, la.value_head_dim,
                                    2 * full_kd, tp_rank, tp_size,
                                )?);
                                let host = to_f32(&bytes, &info)?;
                                device_bytes += host.len() * 4;
                                dev.stream().clone_htod(&host)?
                            }
                        },
                        a_log: match shard {
                            None => upload_a_log(dev, f, &t("ssm_a"), &mut device_bytes)?,
                            Some((tp_rank, tp_size)) => {
                                let la = cfg.linear_attn.as_ref().unwrap();
                                let key_heads_full = la.key_heads * tp_size;
                                let heads_per_key = la.heads_per_key();
                                let info = f.tensor(&t("ssm_a"))?.clone();
                                let bytes = shard_tiled_value_rows(
                                    f, &info, key_heads_full, heads_per_key, 1, 0, tp_rank, tp_size,
                                )?;
                                let raw = to_f32(&bytes, &info)?;
                                let host: Vec<f32> = raw
                                    .iter()
                                    .map(|a| {
                                        debug_assert!(*a < 0.0, "ssm_a: {a} is not -exp(A_log)");
                                        (-a).ln()
                                    })
                                    .collect();
                                device_bytes += host.len() * 4;
                                dev.stream().clone_htod(&host)?
                            }
                        },
                        dt_bias: match shard {
                            None => upload_vector(dev, f, &t("ssm_dt.bias"), &mut device_bytes)?,
                            Some((tp_rank, tp_size)) => {
                                let la = cfg.linear_attn.as_ref().unwrap();
                                let key_heads_full = la.key_heads * tp_size;
                                let heads_per_key = la.heads_per_key();
                                let info = f.tensor(&t("ssm_dt.bias"))?.clone();
                                let bytes = shard_tiled_value_rows(
                                    f, &info, key_heads_full, heads_per_key, 1, 0, tp_rank, tp_size,
                                )?;
                                let host = to_f32(&bytes, &info)?;
                                device_bytes += host.len() * 4;
                                dev.stream().clone_htod(&host)?
                            }
                        },
                        // The one norm in this architecture that is a plain
                        // gain: `Qwen3_5RMSNormGated`. llama.cpp's converter
                        // adds one to every *other* norm weight and leaves this
                        // one alone, which is why nothing is undone here.
                        norm: upload_vector(
                            dev, f, &t("ssm_norm.weight"), &mut device_bytes)?,
                        out_proj: out,
                    }),
                )
            } else {
                (
                    Some(AttnWeights {
                        wq: matrices.next().unwrap(),
                        wk: matrices.next().unwrap(),
                        wv: matrices.next().unwrap(),
                        wo: matrices.next().unwrap(),
                        // Qwen2 carries QKV biases; Llama does not. Q/K/V's
                        // biases belong to their own column-parallel
                        // (Output-axis) matrices and must be sharded the
                        // same way -- see `upload_optional_vector_sharded`'s
                        // own doc comment for the real bug this was found
                        // fixing (shapes checked out, values were silently
                        // wrong on every rank but rank 0).
                        bq: upload_optional_vector_sharded(dev, f, &t("attn_q.bias"), shard, &mut device_bytes)?,
                        bk: upload_optional_vector_sharded(dev, f, &t("attn_k.bias"), shard, &mut device_bytes)?,
                        bv: upload_optional_vector_sharded(dev, f, &t("attn_v.bias"), shard, &mut device_bytes)?,
                        // Row-parallel (attn_output's own bias) -- NOT
                        // sharded, applies to the full-width, already-
                        // all-reduced output; see the sharded fn's doc
                        // comment for why these two cases differ.
                        bo: upload_optional_vector(
                            dev, f, &t("attn_output.bias"), &mut device_bytes)?,
                        // Qwen3's per-head q/k norms; llama.cpp names them this
                        // way. `d_head`-wide (one shared gain applied
                        // identically to every head), not `n_heads*d_head`,
                        // so unlike the biases above these are correctly
                        // replicated in full on every rank as-is -- no
                        // sharding needed (verified: absent on the real
                        // validation checkpoint, but confirmed from the
                        // shape this normalization actually needs, not left
                        // unchecked).
                        q_norm: upload_optional_vector(
                            dev, f, &t("attn_q_norm.weight"), &mut device_bytes)?,
                        k_norm: upload_optional_vector(
                            dev, f, &t("attn_k_norm.weight"), &mut device_bytes)?,
                        w_qkv: None,
                        // `stacked2_gguf` reads `attn_k.weight`/`attn_v.weight`
                        // straight out of the GGUF file, at their full,
                        // unsharded width -- it has no `shard` parameter and
                        // does not go through `upload_matrix_sharded`. Under
                        // TP that is a real, load-bearing mismatch, not just a
                        // missed optimization: whenever a layer's K and V
                        // happen to share type and `k` (true for every layer
                        // on most checkpoints, and true here whenever a
                        // Q4_K_M file's mixed-precision layout gives a layer
                        // uniform-type K/V), this builds a
                        // `w.n = full_kv_dim_k + full_kv_dim_v`-wide matrix --
                        // double the per-rank `kv_dim` every other sharded
                        // tensor in this layer uses -- and the caller
                        // (`attention`'s fused-`w_kv` branch) writes/splits it
                        // against buffers sized for the correctly-sharded
                        // `kv_dim`. The resulting garbage K/V is silent: no
                        // shape assertion catches it because `matmul_pre`
                        // writes `n_tokens * w.n` elements into a scratch
                        // buffer that is usually oversized enough to absorb
                        // the overrun without tripping memcheck. This is what
                        // made llama-3.1-8b-instruct-q4_k_m produce coherent
                        // output through its Q6_K-V layers and degenerate
                        // repeated-token garbage from the first Q4_K-V layer
                        // on -- Q4_K/Q6_K mixing is exactly what usually keeps
                        // this condition false. Disabled outright under
                        // sharding rather than reimplemented against
                        // pre-sharded bytes: it is a decode-step launch-count
                        // optimization (see its own doc comment above), and
                        // the per-matrix fallback below is already correct
                        // and already TP-sharded.
                        w_kv: if shard.is_some() {
                            None
                        } else {
                            stacked2_gguf(
                                dev, f, &t("attn_k.weight"), &t("attn_v.weight"),
                                &mut device_bytes)?
                        },
                        output_gate: cfg.attn_output_gate,
                    }),
                    None,
                )
            };

            layers.push(Layer {
                attn_norm: upload_vector(dev, f, &t("attn_norm.weight"), &mut device_bytes)?,
                attn,
                gdn,
                ffn_norm: upload_vector(dev, f, &ffn_norm_name, &mut device_bytes)?,
                dense: Some(DenseFfn {
                    w_gate: matrices.next().unwrap(),
                    w_up: matrices.next().unwrap(),
                    w_down: matrices.next().unwrap(),
                    // Tried this with `stacked2_gguf`, the same trick
                    // `w_kv` uses two fields up -- gate and up are the same
                    // Q4_K type over the same `k`, and would have been the
                    // single biggest launch-count cut this session found,
                    // one pair a layer on all sixty-four rather than forty-
                    // eight or sixteen of them. VRAM, not logic, is why it
                    // is not here: `stacked2_gguf` re-uploads its
                    // concatenation as a new `Storage::Device` buffer,
                    // real VRAM on top of the checkpoint's own zero-cost
                    // `Storage::Mapped` aliasing, and gate+up fused across
                    // every layer is 6.4 GiB -- this machine's entire free
                    // VRAM after the checkpoint's own mapping. The pool
                    // sizer that runs after weight loading shrank the KV
                    // cache to fit what was left (4096 slots against the
                    // unmodified baseline's 19968), small enough to wrap or
                    // starve mid-generation and produce exactly the
                    // complete-garbage-output symptom that gave this away.
                    // See `in_proj_qz`'s own comment above for the same
                    // finding at GDN's smaller (4.3 GiB) scale. Not a
                    // mechanism this loader can use above a few hundred
                    // rows without giving the VRAM back some other way
                    // first -- freeing it from elsewhere, or aliasing
                    // instead of copying, neither of which this session
                    // attempted.
                    w_gate_up: None,
                }),
                // No GGUF reader for `*_exps` yet; `Config::from_gguf` records
                // the expert counts so that a MoE GGUF fails on the missing
                // `blk.0.ffn_gate.weight` above rather than here.
                moe: None,
                blob,
            });
        }

        dev.synchronize()?;
        tracing::info!(
            gpu_layers = n_gpu_layers,
            offloaded = cfg.n_layers - n_gpu_layers,
            vram_mib = device_bytes / (1 << 20),
            host_mib = host_bytes / (1 << 20),
            ms = started.elapsed().as_millis(),
            "weights loaded"
        );

        let this = Self {
            token_embd,
            layers,
            output_norm,
            output,
            // A GGUF file's vocab projection comes in whatever the file chose;
            // the split layout is only for the one infero quantizes itself.
            output_split: None,
            rope_freqs,
            mrope_axis,
            device_bytes,
            host_bytes,
            max_blob_bytes,
        };
        this.check_shapes(cfg)?;
        Ok(this)
    }

    pub fn n_offloaded(&self) -> usize {
        self.layers.iter().filter(|l| l.is_offloaded()).count()
    }

    /// Catch a config/tensor mismatch here rather than as silent garbage
    /// several kernels later.
    fn check_shapes(&self, cfg: &Config) -> Result<()> {
        let d = cfg.d_model;
        // The attention block is not square once `n_heads * d_head` stops
        // equalling `d_model`: q widens the residual to `d_attn` and o narrows
        // it back. This check is the only place a width mistake is caught
        // against the actual tensor, so it has to know the difference.
        let da = cfg.d_attn();
        let kv_dim = cfg.d_kv();

        anyhow::ensure!(
            self.token_embd.k == d && self.token_embd.n == cfg.vocab_size,
            "token_embd is [{}, {}], expected [{d}, {}]",
            self.token_embd.k,
            self.token_embd.n,
            cfg.vocab_size
        );

        for (i, l) in self.layers.iter().enumerate() {
            let expect = |m: &Matrix, k: usize, n: usize, what: &str| -> Result<()> {
                anyhow::ensure!(
                    m.k == k && m.n == n,
                    "layer {i} {what} is [{}, {}], expected [{k}, {n}]",
                    m.k,
                    m.n
                );
                Ok(())
            };
            if let Some(g) = &l.gdn {
                // A GatedDeltaNet block. Its widths come from the linear
                // dimensions, not from the attention ones, and checking it
                // against `d_attn` would pass on some of them by coincidence.
                let la = cfg.linear_attn.context(
                    "a block has GatedDeltaNet weights but the config gives no \
                     linear-attention dimensions to check them against",
                )?;
                let (key_dim, val_dim) = (la.key_dim(), la.value_dim());
                expect(&g.in_proj_qkv, d, la.conv_channels(), "in_proj_qkv")?;
                expect(&g.in_proj_z, d, val_dim, "in_proj_z")?;
                expect(&g.in_proj_a, d, la.value_heads, "in_proj_a")?;
                expect(&g.in_proj_b, d, la.value_heads, "in_proj_b")?;
                expect(&g.out_proj, val_dim, d, "out_proj")?;
                // The 1-D parameters, whose lengths encode the head counts.
                for (v, want, what) in [
                    (&g.conv1d, la.conv_channels() * la.conv_kernel, "conv1d"),
                    (&g.a_log, la.value_heads, "A_log"),
                    (&g.dt_bias, la.value_heads, "dt_bias"),
                    (&g.norm, la.value_head_dim, "norm"),
                ] {
                    anyhow::ensure!(
                        v.len() == want,
                        "layer {i} {what} has {} elements, expected {want}",
                        v.len()
                    );
                }
                let _ = key_dim;
            } else {
                let a = l.attn();
                // A gated q projection is twice as wide: a query and its gate
                // interleaved per head.
                let q_cols = if a.output_gate { 2 * da } else { da };
                expect(&a.wq, d, q_cols, "attn_q")?;
                expect(&a.wk, d, kv_dim, "attn_k")?;
                expect(&a.wv, d, kv_dim, "attn_v")?;
                expect(&a.wo, da, d, "attn_output")?;
            }
            match (&l.dense, &l.moe) {
                (Some(f), None) => {
                    expect(&f.w_gate, d, cfg.d_ff, "ffn_gate")?;
                    expect(&f.w_up, d, cfg.d_ff, "ffn_up")?;
                    expect(&f.w_down, cfg.d_ff, d, "ffn_down")?;
                }
                (None, Some(m)) => {
                    let moe = cfg
                        .moe
                        .as_ref()
                        .context("a layer has expert weights but the config is not sparse")?;
                    let dff = moe.d_ff_expert;
                    // The router's width is the check that catches an expert
                    // count read from the wrong field: 128 rows against a
                    // config that says 64 loads, routes to experts that are
                    // there, and silently never selects half of them.
                    expect(&m.router, d, moe.n_experts, "mlp.gate")?;
                    for (e, name) in [(&m.gate, "gate"), (&m.up, "up")] {
                        anyhow::ensure!(
                            e.k == d && e.n == dff && e.n_experts == moe.n_experts,
                            "expert {name} is [{}, {}] x{} where the config wants \
                             [{dff}, {d}] x{}",
                            e.n,
                            e.k,
                            e.n_experts,
                            moe.n_experts
                        );
                    }
                    anyhow::ensure!(
                        m.down.k == dff && m.down.n == d && m.down.n_experts == moe.n_experts,
                        "expert down is [{}, {}] x{} where the config wants \
                         [{d}, {dff}] x{}",
                        m.down.n,
                        m.down.k,
                        m.down.n_experts,
                        moe.n_experts
                    );
                }
                (Some(_), Some(_)) => anyhow::bail!(
                    "a layer has both a dense FFN and experts; the loader must \
                     pick one from `MoeConfig::is_sparse`"
                ),
                (None, None) => anyhow::bail!("a layer has no FFN at all"),
            }
        }
        Ok(())
    }

    /// The encoding most of the model is stored in, for reporting.
    pub fn dominant_type(&self) -> WeightType {
        let mut totals: std::collections::HashMap<WeightType, usize> = Default::default();
        for l in &self.layers {
            // A block's matrices depend on which mixer it has. Reaching for the
            // attention ones unconditionally is what the `attn()` accessor
            // panics about, and this loop runs over every layer.
            let mixer: Vec<&Matrix> = match (&l.attn, &l.gdn) {
                (Some(a), _) => vec![&a.wq, &a.wk, &a.wv, &a.wo],
                (_, Some(g)) => vec![
                    &g.in_proj_qkv,
                    &g.in_proj_z,
                    &g.in_proj_a,
                    &g.in_proj_b,
                    &g.out_proj,
                ],
                _ => vec![],
            };
            let ffn: Vec<&Matrix> = match &l.dense {
                Some(f) => vec![&f.w_gate, &f.w_up, &f.w_down],
                None => vec![],
            };
            for m in mixer.into_iter().chain(ffn) {
                *totals.entry(m.ty).or_default() += m.n_bytes;
            }
            // The experts are most of a sparse model's bytes, so leaving them
            // out would report the attention projections' encoding as the
            // model's.
            if let Some(m) = &l.moe {
                for e in [&m.gate, &m.up, &m.down] {
                    *totals.entry(e.ty).or_default() += e.n_bytes();
                }
            }
        }
        totals
            .into_iter()
            .max_by_key(|&(_, n)| n)
            .map(|(t, _)| t)
            .unwrap_or(self.token_embd.ty)
    }
}

/// One if this norm weight is stored as a delta from one, zero if it is a gain.
///
/// Qwen3.5 stores most of its norm weights as a *delta from one*:
/// `Qwen3_5RMSNorm` initializes to zeros and computes
/// `normalized * (1 + weight)`, where every other model infero loads initializes
/// to ones and computes `weight * normalized`. Adding the one here, at load,
/// means every norm kernel stays as it is — the alternative was a variant of
/// `rms_norm` and `qk_norm` each.
///
/// The exception is `linear_attn.norm`, which is `Qwen3_5RMSNormGated` and does
/// use the plain form. Two conventions in one checkpoint.
///
/// Which form a tensor wants follows from the class that consumes it, not from
/// the tensor: the two populations overlap. An `input_layernorm` centred at
/// 0.036 would be annihilated by the plain form, but some trained `q_norm`
/// deltas exceed 0.5 and some gated gains fall below 1.5, so a mean-based guess
/// gets those wrong. Hence a name rule, not a data rule.
///
/// A whitelist, not a blacklist. This same loader path reads the attention
/// biases and the GatedDeltaNet's `A_log`, `dt_bias` and `conv1d.weight`, none
/// of which are norms; a blacklist that forgot one of those would add one to a
/// bias or to a decay exponent. A whitelist that forgets a norm leaves that
/// norm on the old convention, which is bad but confined.
///
/// The MTP head adds three names and no new rule. Its `input_layernorm`,
/// `post_attention_layernorm`, `q_norm` and `k_norm` are already matched by the
/// suffixes above — they are the same classes in the same decoder block — and
/// `notes/qwen3.5-mtp.md` measures all four of the head's own norms as the
/// offset form, so `pre_fc_norm_embedding`, `pre_fc_norm_hidden` and `mtp.norm`
/// are listed here by their exact names. `mtp.norm.weight` cannot be matched by
/// the `norm.weight` suffix for the same reason `model.norm.weight` is not:
/// `linear_attn.norm.weight` ends in it too and is the one gain in the model.
fn norm_offset(arch: &str, name: &str) -> f32 {
    const OFFSET_FORM: &[&str] = &[
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    ];
    const OFFSET_FORM_EXACT: &[&str] = &[
        // The final norm before the vocabulary projection.
        "model.norm.weight",
        "model.language_model.norm.weight",
        // The MTP head's three. Its fourth and fifth — the decoder layer's two
        // — are covered by the suffixes; see the note above.
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight",
    ];
    let is_offset_form = OFFSET_FORM.iter().any(|suffix| name.ends_with(suffix))
        || OFFSET_FORM_EXACT.contains(&name);
    if arch == "qwen3_5" && is_offset_form {
        1.0
    } else {
        0.0
    }
}

/// Load the multi-token-prediction head, if the checkpoint has one.
///
/// Returns `None` when neither the config nor the tensors mention a head, and
/// an error when only one of them does — a checkpoint that names
/// `mtp_num_hidden_layers` and ships no `mtp.*` would otherwise build a drafter
/// out of nothing, and one that ships the tensors under a config that does not
/// mention them means this loader is reading a layout it has not been shown.
///
/// The head's tensors live under **`mtp.`**, not `model.mtp.`; vLLM rewrites the
/// prefix to `model.` at load time, which is where a reader looking for
/// `model.mtp.layers.0` gets the idea. Everything else about the block is the
/// text model's full-attention layout, so it goes through the same code.
///
/// One thing in here is not uniform: `mtp.fc` is BF16 while the rest of the head
/// is FP8. The dispatch is on the tensor's own dtype rather than on a list of
/// exceptions, which agrees with the checkpoint's
/// `quantization_config.modules_to_not_convert` (it names `mtp.fc`) and with
/// vLLM's special case, and keeps working if a future export quantizes it.
pub fn load_mtp(
    dev: &Device,
    w: &infero_safetensors::Shards,
    cfg: &Config,
) -> Result<Option<MtpWeights>> {
    let present = w.get("mtp.fc.weight").is_some();
    match (cfg.mtp_layers, present) {
        (0, false) => return Ok(None),
        (0, true) => anyhow::bail!(
            "this checkpoint ships `mtp.*` tensors but its config does not say \
             `mtp_num_hidden_layers`; the head's depth decides how the draft \
             loop indexes its layers and guessing it is not safe"
        ),
        (n, false) => anyhow::bail!(
            "config says mtp_num_hidden_layers = {n} but there is no \
             `mtp.fc.weight`; there is nothing to build the drafter out of"
        ),
        (n, true) => anyhow::ensure!(
            n == 1,
            "this loader builds a one-layer MTP head, the config says {n}. The \
             draft loop indexes `spec_step_idx % mtp_num_hidden_layers`, which \
             is only the identity at one layer"
        ),
    }
    anyhow::ensure!(
        !cfg.mtp_dedicated_embeddings,
        "config says the MTP head has dedicated embeddings; this loader has the \
         head share the text model's `embed_tokens` and `lm_head`, which is what \
         `mtp_use_dedicated_embeddings = false` means"
    );
    // The other half of the sharing claim, from the tensors rather than the
    // config. Absence is part of the spec here: a checkpoint that shipped these
    // would want them used, and using the text model's instead would be a
    // silently different drafter.
    for name in ["mtp.embed_tokens.weight", "mtp.lm_head.weight"] {
        anyhow::ensure!(
            w.get(name).is_none(),
            "the checkpoint carries {name}, so the head does not share the text \
             model's — but the config says it does. One of the two is being \
             misread"
        );
    }
    // And the head's layer is a full-attention block, which the checkpoint
    // settles without interpretation: it has q/k/v/o and no recurrence. This is
    // the single most consequential fact for scheduling — a drafter that ran a
    // GatedDeltaNet layer would advance recurrent state on every speculative
    // token — so it is checked here as well as in the tests.
    for absent in [
        "mtp.layers.0.linear_attn.in_proj_qkv.weight",
        "mtp.layers.0.linear_attn.conv1d.weight",
        "mtp.layers.0.linear_attn.A_log",
        "mtp.layers.0.linear_attn.dt_bias",
    ] {
        anyhow::ensure!(
            w.get(absent).is_none(),
            "the MTP head carries {absent}: its layer is not the full-attention \
             block this drafter assumes, and drafting would touch recurrent state"
        );
    }

    let started = std::time::Instant::now();
    let mut bytes = 0usize;
    let vector = |name: &str, total: &mut usize| -> Result<Vector> {
        let t = w.tensor(name)?;
        let mut v = t.to_f32()?;
        let off = norm_offset(&cfg.arch, name);
        if off != 0.0 {
            for x in v.iter_mut() {
                *x += off;
            }
        }
        *total += v.len() * 4;
        Ok(dev.stream().clone_htod(&v)?)
    };
    let projection = |name: &str, total: &mut usize| -> Result<Matrix> {
        let t = w.tensor(&format!("{name}.weight"))?;
        anyhow::ensure!(
            t.shape.len() == 2,
            "{name}.weight has shape {:?}, expected a matrix",
            t.shape
        );
        // Row-major `[out, in]`, as torch writes it, which is the layout the
        // f16 GEMM and the mat-vec both want: `k` is the contraction dimension
        // and one row of the weight is contiguous.
        let (n, k) = (t.shape[0], t.shape[1]);
        if t.dtype == infero_safetensors::Dtype::F8E4M3 {
            // Kept as FP8, like the text model's projections, and for the same
            // two reasons. This used to dequantize to f16 here — the strategy
            // the text side abandoned — which doubled the head's bytes from
            // 405 MiB to 810 and sent it through `gemv_f16` instead of the FP8
            // mat-vec. A draft step reads the head once, so its bytes are its
            // cost: 5.12 ms measured against a 2.0 ms byte bound, and half of
            // those bytes did not need to exist.
            let scales_t = w
                .tensor(&format!("{name}.weight_scale_inv"))
                .with_context(|| {
                    format!("{name}.weight is FP8, which is meaningless without its block scales")
                })?;
            let scales = scales_t.to_f32()?;
            let want = infero_kernels::fp8::scale_grid(k, n);
            anyhow::ensure!(
                scales.len() == want,
                "{name}'s scale grid has {} entries; an [{n}, {k}] matrix wants {want}",
                scales.len()
            );
            let mut bytes = Vec::with_capacity(infero_kernels::fp8::fp8_bytes(k, n));
            bytes.extend_from_slice(&infero_kernels::fp8::repack_rows(t.data, k, n)?);
            for v in &scales {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            *total += bytes.len();
            return Ok(Matrix {
                ty: WeightType::F8E4M3,
                k,
                n,
                n_bytes: bytes.len(),
                storage: Storage::Device(dev.stream().clone_htod(&bytes)?),
                cutlass_weight: Default::default(),
            });
        }
        // `mtp.fc` lands here — BF16 in this checkpoint. Not a special case in
        // the code, only in the checkpoint.
        let halves = t.to_f16()?;
        anyhow::ensure!(halves.len() == k * n, "{name}: {} halves for {k}x{n}", halves.len());
        // Safety: f16 is a transparent u16, so these are already the
        // little-endian halves the device expects, and the view does not outlive
        // `halves`.
        let raw = unsafe {
            std::slice::from_raw_parts(halves.as_ptr() as *const u8, halves.len() * 2)
        };
        *total += raw.len();
        Ok(Matrix {
            ty: WeightType::F16,
            k,
            n,
            n_bytes: raw.len(),
            storage: Storage::Device(dev.stream().clone_htod(raw)?),
                cutlass_weight: Default::default(),
        })
    };

    let l = "mtp.layers.0";
    let wq = projection(&format!("{l}.self_attn.q_proj"), &mut bytes)?;
    // Twice the attention width means q_proj carries a gate interleaved with
    // the query, per head. Read off the shape, like the text model's blocks,
    // because the shape is what the rest of the code has to agree with.
    let output_gate = wq.n == 2 * cfg.d_attn();
    anyhow::ensure!(
        wq.n == cfg.d_attn() || output_gate,
        "the MTP head's q_proj has {} columns; expected {} for a plain query or \
         {} for a query and its gate",
        wq.n,
        cfg.d_attn(),
        2 * cfg.d_attn()
    );
    let layer = Layer {
        attn_norm: vector(&format!("{l}.input_layernorm.weight"), &mut bytes)?,
        attn: Some(AttnWeights {
            wq,
            wk: projection(&format!("{l}.self_attn.k_proj"), &mut bytes)?,
            wv: projection(&format!("{l}.self_attn.v_proj"), &mut bytes)?,
            wo: projection(&format!("{l}.self_attn.o_proj"), &mut bytes)?,
            bq: None,
            bk: None,
            bv: None,
            bo: None,
            q_norm: Some(vector(&format!("{l}.self_attn.q_norm.weight"), &mut bytes)?),
            k_norm: Some(vector(&format!("{l}.self_attn.k_norm.weight"), &mut bytes)?),
            w_qkv: None,
            w_kv: None,
            output_gate,
        }),
        gdn: None,
        ffn_norm: vector(&format!("{l}.post_attention_layernorm.weight"), &mut bytes)?,
        dense: Some(DenseFfn {
            w_gate: projection(&format!("{l}.mlp.gate_proj"), &mut bytes)?,
            w_up: projection(&format!("{l}.mlp.up_proj"), &mut bytes)?,
            w_down: projection(&format!("{l}.mlp.down_proj"), &mut bytes)?,
            w_gate_up: None,
        }),
        moe: None,
        blob: None,
    };

    let fc = projection("mtp.fc", &mut bytes)?;
    anyhow::ensure!(
        fc.k == 2 * cfg.d_model && fc.n == cfg.d_model,
        "mtp.fc is [{}, {}], expected [{}, {}] — it consumes the embedding and \
         the hidden state concatenated and produces one residual row",
        fc.n,
        fc.k,
        cfg.d_model,
        2 * cfg.d_model
    );
    let head = MtpWeights {
        fc,
        pre_fc_norm_embedding: vector("mtp.pre_fc_norm_embedding.weight", &mut bytes)?,
        pre_fc_norm_hidden: vector("mtp.pre_fc_norm_hidden.weight", &mut bytes)?,
        norm: vector("mtp.norm.weight", &mut bytes)?,
        layer,
        device_bytes: bytes,
    };
    tracing::info!(
        vram_mib = bytes >> 20,
        gated = output_gate,
        ms = started.elapsed().as_millis(),
        "mtp head loaded"
    );
    Ok(Some(head))
}

/// The tile an FP8 checkpoint's `weight_scale_inv` covers, from
/// `quantization_config.weight_block_size`.
const FP8_BLOCK: usize = 128;

/// Describe a tensor without moving its bytes anywhere.
fn describe(f: &Gguf, name: &str) -> Result<(WeightType, usize, usize, usize)> {
    let info = f.tensor(name)?;
    anyhow::ensure!(
        info.dims.len() == 2,
        "{name} has {} dimensions, expected 2",
        info.dims.len()
    );
    let ty = WeightType::from_ggml(info.ty).with_context(|| format!("tensor {name}"))?;
    let (k, n) = (info.dims[0] as usize, info.dims[1] as usize);
    anyhow::ensure!(
        k.is_multiple_of(ty.block_size()),
        "{name}: row length {k} is not a multiple of {}'s block size {}",
        ty,
        ty.block_size()
    );
    Ok((ty, k, n, info.n_bytes))
}

/// Two same-shaped GGUF tensors, stacked along the output the way `load_awq`'s
/// `stacked2` stacks a safetensors pair — one launch instead of two wherever
/// the model code finds this `Some`.
///
/// `stacked2` only takes F16/F32 because it assumes nothing about a block's
/// layout; here the two tensors are read straight off the mapped file, so any
/// encoding works as long as its blocks are row-scoped, which every ggml type
/// this loader supports is (a block never spans two rows) — row-major means
/// the stack is byte concatenation and nothing has to be interleaved. Returns
/// `None` when the two are not the same type over the same `k`, in which case
/// the caller keeps its two launches.
fn stacked2_gguf(dev: &Device, f: &Gguf, a: &str, b: &str, total: &mut usize) -> Result<Option<Matrix>> {
    let (ty, k, n_a, _) = describe(f, a)?;
    let (ty_b, k_b, n_b, _) = describe(f, b)?;
    if ty != ty_b || k != k_b {
        return Ok(None);
    }
    let mut bytes = f.tensor_data(a)?.to_vec();
    bytes.extend_from_slice(f.tensor_data(b)?);
    *total += bytes.len();
    Ok(Some(Matrix {
        ty,
        k,
        n: n_a + n_b,
        n_bytes: bytes.len(),
        storage: Storage::Device(dev.stream().clone_htod(&bytes)?),
                cutlass_weight: Default::default(),
    }))
}

/// Load an AWQ checkpoint, repacking every quantized matrix on the way in.
///
/// Everything stays resident: an AWQ file has no offload story yet, and the
/// point of reading one is speed rather than fitting a model that does not.
///
/// Two things differ from the GGUF path beyond the tensor names. The
/// projections arrive in AWQ's transposed, column-packed layout and are
/// repacked to [`WeightType::Q4G128`] here, once, so the mat-vec sees the
/// output-major rows it wants. And the vocabulary projection arrives as `f16` —
/// 1.05 GB on an 8B model, a fifth of a decode step, which the float mat-vec
/// reads at 141 GB/s against the integer path's 366 — so it is quantized to
/// Q8_0 on the way in. Eight bits is not a meaningful loss for a projection
/// whose output is fed to an argmax over 128k logits.
pub fn load_awq(
    dev: &Device,
    w: &infero_safetensors::Shards,
    cfg: &Config,
    freq_factors: &[f32],
) -> Result<Weights> {
    use infero_kernels::awq::{AwqTensor, quantize_f16_to_q8_0};

    let started = std::time::Instant::now();
    let mut device_bytes = 0usize;

    let upload = |bytes: &[u8], ty: WeightType, k: usize, n: usize, total: &mut usize| -> Result<Matrix> {
        *total += bytes.len();
        let __t0 = std::time::Instant::now();
        let storage = Storage::Device(dev.stream().clone_htod(bytes)?);
        UPLOAD_NS.fetch_add(__t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(Matrix {
            ty,
            k,
            n,
            n_bytes: bytes.len(),
            storage,
                cutlass_weight: Default::default(),
        })
    };
    let arch = cfg.arch.clone();
    let norm_offset = move |name: &str| -> f32 { norm_offset(&arch, name) };
    let vector = |name: &str, total: &mut usize| -> Result<Vector> {
        let t = w.tensor(name)?;
        // `to_f32` rather than `as_f16`: Qwen3's AWQ export stores the norm
        // weights as BF16 even though the AWQ scales are F16, and an F16-only
        // read rejects the checkpoint on `model.norm.weight`.
        let mut v = t.to_f32()?;
        let off = norm_offset(name);
        if off != 0.0 {
            for x in v.iter_mut() {
                *x += off;
            }
        }
        *total += v.len() * 4;
        Ok(dev.stream().clone_htod(&v)?)
    };
    // The same, for a tensor a checkpoint may not carry. Qwen2 and Qwen3 put a
    // bias on q, k and v; Llama does not, and asking for one there is not an
    // error. The GGUF loader has always read these — the AWQ path passed `None`
    // and produced fluent-looking nonsense on a Qwen checkpoint, because a
    // missing bias is not a crash, just a wrong answer.
    let optional_vector = |name: &str, total: &mut usize| -> Result<Option<Vector>> {
        match w.tensor(name) {
            Ok(t) => {
                let mut v = t.to_f32()?;
                let off = norm_offset(name);
                if off != 0.0 {
                    for x in v.iter_mut() {
                        *x += off;
                    }
                }
                *total += v.len() * 4;
                Ok(Some(dev.stream().clone_htod(&v)?))
            }
            Err(_) => Ok(None),
        }
    };
    // A quantized projection's bytes, before they reach the device: AWQ's three
    // tensors in, one packed matrix out. Split from the upload so that
    // projections which are stacked into one matrix — see `fuse_ffn` below —
    // can be concatenated in the layout they will be read in.
    let projection_bytes = |prefix: &str| -> Result<(Vec<u8>, WeightType, usize, usize)> {
        // FP8 and plain-float exports name the matrix `{prefix}.weight`; AWQ
        // splits it into qweight/qzeros/scales. Check for the single tensor
        // first, because its absence is the cheap question.
        //
        // Note the transposed convention between the two. AWQ stores
        // `[in_features, out_features / 8]`, so `k` is dimension 0. Everything
        // else stores output-major `[out_features, in_features]`, so `k` is
        // dimension 1. Reading one with the other's convention gives a matrix of
        // plausible size and wrong meaning.
        if let Some(t) = w.get(&format!("{prefix}.weight")) {
            let (n, k) = (t.shape[0], t.shape[1]);
            if t.dtype == infero_safetensors::Dtype::F8E4M3 {
                // Keep the FP8 bytes and carry the scale grid with them, rather
                // than expanding here. Expanding is correct and was the first
                // version; it doubles what a decode step has to read, and the
                // profiler put the resulting f16 GEMM at 75% of a step.
                //
                // One buffer: `n * k` quants then the grid as f32, which is the
                // layout `WeightType::F8E4M3` documents and both FP8 kernels
                // read. A `Matrix` stays a single allocation, so the offload
                // blob path and `Matrix::view` need no special case.
                let scales_t = w
                    .tensor(&format!("{prefix}.weight_scale_inv"))
                    .with_context(|| format!("{prefix} is FP8 but has no scale grid"))?;
                let scales = scales_t.to_f32()?;
                let want = infero_kernels::fp8::scale_grid(k, n);
                anyhow::ensure!(
                    scales.len() == want,
                    "{prefix}'s scale grid has {} entries; an [{n}, {k}] matrix \
                     at block {} wants {want}",
                    scales.len(),
                    infero_kernels::fp8::FP8_BLOCK,
                );
                // Permuted, not copied: every FP8 kernel reads four interleaved
                // rows as one 16-byte load, which is what took the batched
                // mat-vec off a request-per-row-per-token. The permutation lives
                // in `fp8::repack_rows` so that this and `tests/fp8_matvec.rs`
                // cannot drift apart.
                //
                // Unless `fp8_unified_layout()` is on, in which case this is
                // `fp8::pad_rows` instead -- plain `[n,k]` row-major, no
                // permutation, which is both `Kernels::mmv_f8_plain`'s and
                // CUTLASS's native layout. See that function's doc comment
                // for why this is safe to prefer now.
                //
                // `repack_rows`/`pad_rows` already hand back a `padded *
                // k`-byte `Vec`, filled in parallel — appending it into a
                // second, freshly `with_capacity`'d buffer instead of just
                // using it looked harmless but wasn't: that second buffer's
                // pages are unmapped until this exact `extend_from_slice`
                // first touches them, so the copy pays for every one of the
                // 43 GB checkpoint's page faults on a single thread.
                // `repack_rows`'s own allocation pays the same fault cost
                // already, just spread over sixteen threads — 3.6 s measured
                // against this copy's 25.7 s of the 27B's ~60 s load. Reusing
                // it and only growing it for the scale tail turns that
                // second full pass into nothing.
                let __t0 = std::time::Instant::now();
                let mut bytes = if fp8_unified_layout() {
                    infero_kernels::fp8::pad_rows(t.data, k, n)?
                } else {
                    infero_kernels::fp8::repack_rows(t.data, k, n)?
                };
                REPACK_NS.fetch_add(__t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                bytes.reserve_exact(scales.len() * 4);
                for v in &scales {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                debug_assert_eq!(bytes.len(), infero_kernels::fp8::fp8_bytes(k, n));
                return Ok((bytes, WeightType::F8E4M3, k, n));
            }
            let halves: Vec<half::f16> = if false {
                // Block-scaled FP8. The scale grid is 128x128 and
                // `dequant_f8_to_f16` validates that the grid matches the
                // quants, which is the check that catches a transposed or
                // row-only index — neither of which fails on its own, they just
                // mis-scale every tile past the first.
                let scales = w
                    .tensor(&format!("{prefix}.weight_scale_inv"))
                    .with_context(|| format!("{prefix} is FP8 but has no scale grid"))?;
                t.dequant_f8_to_f16(&scales, 128)
                    .with_context(|| format!("dequantizing {prefix}"))?
            } else {
                t.to_f16()
                    .with_context(|| format!("converting {prefix} to f16"))?
                    .into_owned()
            };
            // Safety: f16 is a transparent u16, so these are already the
            // little-endian halves the device wants.
            let bytes = unsafe {
                std::slice::from_raw_parts(halves.as_ptr() as *const u8, halves.len() * 2)
            }
            .to_vec();
            return Ok((bytes, WeightType::F16, k, n));
        }
        let qw = w.tensor(&format!("{prefix}.qweight"))?;
        let (k, n) = (qw.shape[0], qw.shape[1] * 8);
        // The scales are BF16 in Qwen3's AWQ export, not F16 — bound to a local
        // so the converted halves outlive the borrow `AwqTensor` takes.
        let scales_t = w.tensor(&format!("{prefix}.scales"))?;
        let scales = scales_t.to_f16()?;
        let packed = AwqTensor {
            qweight: qw.as_i32()?,
            qzeros: w.tensor(&format!("{prefix}.qzeros"))?.as_i32()?,
            scales: scales.as_ref(),
            in_features: k,
            out_features: n,
        }
        .repack()
        .with_context(|| format!("repacking {prefix}"))?;
        // The transposed layout, which the f16 tensor-core GEMM reads as one
        // aligned 16-byte fragment per lane rather than four four-byte words.
        // Worth 11% on the GEMM at 32 tokens and 5.8% on the decode step, with
        // the mat-vec level at a batch of one. `INFERO_AWQ_PACKED=1` keeps the
        // old blocks, which is how the two are A/B-ed; `transposable` rejects a
        // row length whose stride would not land the quants on 16 bytes, and
        // every real projection width passes it.
        static PACKED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*PACKED.get_or_init(|| std::env::var_os("INFERO_AWQ_PACKED").is_some())
            && infero_kernels::awq::transposable(k)
        {
            let t = infero_kernels::awq::transpose_words(&packed, k, n);
            return Ok((t, WeightType::Q4G128T, k, n));
        }
        Ok((packed, WeightType::Q4G128, k, n))
    };
    let projection = |prefix: &str, total: &mut usize| -> Result<Matrix> {
        let (bytes, ty, k, n) = projection_bytes(prefix)?;
        upload(&bytes, ty, k, n, total)
    };
    // Same as `projection`, but also hands back the pre-upload bytes and their
    // shape/type -- for a caller that is about to fuse this projection with a
    // sibling one (`stacked`/`stacked_fp8_2`/`stacked2`/`stacked3`) and would
    // otherwise pay `projection_bytes`'s real cost (the FP8 repack, or the AWQ
    // unpack/transpose) a second time for bytes it already has in hand. On the
    // 27B this was 10 of the FFN block's 14 seconds -- gate/up's own read
    // fetched and repacked twice each, once standalone and once again inside
    // `stacked_fp8_2`, every one of 64 layers.
    let projection_with_bytes =
        |prefix: &str, total: &mut usize| -> Result<(Matrix, Vec<u8>, WeightType, usize, usize)> {
            let (bytes, ty, k, n) = projection_bytes(prefix)?;
            let m = upload(&bytes, ty, k, n, total)?;
            Ok((m, bytes, ty, k, n))
        };

    // `gate` and `up` as one matrix, and `q`/`k`/`v` as another, which is what
    // vLLM's `MergedColumnParallelLinear` and `QKVParallelLinear` amount to. A
    // matmul's efficiency here rises steeply with its width — 4096x14336
    // reaches 1154 GB/s where 4096x28672 reaches 1368 — because a narrow one
    // cannot fill the device.
    //
    // On by default since it was measured end to end rather than on the GEMM
    // alone: a batch-32 step's matmuls fall from 110.5 ms to 96.8 over twenty
    // steps and its launches from 225 to 129, worth 4.4% of the served
    // throughput on a Blackwell RTX PRO 6000. `INFERO_FUSE_FFN=0` puts the three
    // narrow matmuls back.
    //
    // It costs VRAM: the stacked copies are held *as well as* the originals,
    // 2 GiB on an 8B model, because at a batch of one the integer mat-vec runs
    // instead and reads the originals. Dropping them means teaching the mat-vec
    // to take a column range of a stacked matrix — the scales of a Q4_G128T
    // matrix live past all of its quants, so a sub-matrix is two disjoint byte
    // ranges rather than one. Worth doing; not worth blocking the throughput on.
    //
    // So the default is conditional rather than unconditional: the stacked
    // copies are the whole of the attention and FFN projections again, and a
    // card that cannot spare that would rather have the KV cache. Whatever the
    // decision, it is logged — a throughput number that moved by 4% because the
    // loader quietly declined is the kind of thing that costs a day.
    let fuse_ffn = match std::env::var("INFERO_FUSE_FFN").as_deref() {
        Ok("0") => false,
        Ok(_) => true,
        Err(_) => {
            // What the stacked copies will cost: `q`+`k`+`v` and `gate`+`up`
            // again, which is every projection but `o` and `down`.
            let mut extra = 0usize;
            for i in 0..cfg.n_layers {
                for m in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                          "mlp.gate_proj", "mlp.up_proj"] {
                    // `qweight` is four bits an element; the repacked form adds
                    // a scale and a zero per group of 128, so 68 bytes where
                    // the pack has 64.
                    let n = format!("model.layers.{i}.{m}.qweight");
                    extra += w.tensor(&n).map(|t| t.data.len() * 17 / 16).unwrap_or(0);
                }
            }
            let free = dev.mem_info().map(|(f, _)| f).unwrap_or(0);
            // Leave the KV cache room to be worth having: the pool is what
            // decides how many sequences can run at once, and a batch that
            // narrows costs far more than 4%.
            let room = extra * 3 < free;
            tracing::info!(
                extra_mib = extra >> 20,
                free_mib = free >> 20,
                fused = room,
                "fused projections"
            );
            room
        }
    };
    // Every `stacked*` helper below takes each input as an already-fetched
    // `(bytes, type, k, n)` piece — the caller has always already called
    // `projection_bytes`/`projection_with_bytes` on the same name to build
    // that input's own standalone `Matrix`, so re-deriving it here a second
    // time (the shape this file had until this pass) means paying the FP8
    // repack or AWQ unpack/transpose twice for the same source bytes. On the
    // 27B, gate/up's own read alone was 10 of the FFN block's 14 seconds this
    // way, 64 times over — a caller now fetches once and hands the same
    // bytes to both its own `Matrix` and whichever of these builds the fused
    // one.
    type Piece<'a> = (&'a [u8], WeightType, usize, usize);
    let stacked = |a: Piece<'_>, b: Piece<'_>, total: &mut usize| -> Result<Option<Matrix>> {
        if !fuse_ffn {
            return Ok(None);
        }
        let (ba, ty_a, k, n_a) = a;
        let (bb, ty_b, k_b, n_b) = b;
        // Only the transposed layout stacks: the packed one keeps its scales
        // inside each block, so appending rows is appending bytes and there is
        // nothing to gain by doing it here rather than in the kernel.
        if ty_a != WeightType::Q4G128T || ty_b != ty_a || k_b != k {
            return Ok(None);
        }
        let c = infero_kernels::awq::concat_t(ba, n_a, bb, n_b, k);
        Ok(Some(upload(&c, ty_a, k, n_a + n_b, total)?))
    };
    let stacked3 = |a: Piece<'_>, b: Piece<'_>, cc: Piece<'_>, total: &mut usize| -> Result<Option<Matrix>> {
        if !fuse_ffn {
            return Ok(None);
        }
        let (ba, ty, k, n_a) = a;
        let (bb, ty_b, k_b, n_b) = b;
        let (bc, ty_c, k_c, n_c) = cc;
        if ty_b != ty || ty_c != ty || k_b != k || k_c != k {
            return Ok(None);
        }
        // FP8's per-block scale grid means a stack is only free of interior
        // padding when every piece's row count already lands on a scale
        // block; q/k/v pass this on every GQA shape seen so far (head_dim a
        // divisor of `FP8_BLOCK`), but a checkpoint that does not still gets
        // a correct, merely unfused, load rather than a wrong one.
        if ty == WeightType::F8E4M3 {
            if !n_a.is_multiple_of(infero_kernels::fp8::FP8_BLOCK)
                || !n_b.is_multiple_of(infero_kernels::fp8::FP8_BLOCK)
                || !n_c.is_multiple_of(infero_kernels::fp8::FP8_BLOCK)
            {
                return Ok(None);
            }
            let abc = infero_kernels::fp8::concat3(ba, n_a, bb, n_b, bc, n_c, k);
            return Ok(Some(upload(&abc, ty, k, n_a + n_b + n_c, total)?));
        }
        if ty != WeightType::Q4G128T {
            return Ok(None);
        }
        let ab = infero_kernels::awq::concat_t(ba, n_a, bb, n_b, k);
        let abc = infero_kernels::awq::concat_t(&ab, n_a + n_b, bc, n_c, k);
        Ok(Some(upload(&abc, ty, k, n_a + n_b + n_c, total)?))
    };
    // Two FP8 matrices over the same `k`, stacked the way `stacked3` stacks
    // three. GatedDeltaNet's `in_proj_qkv`/`in_proj_z` pair is not the
    // Q4G128T-only shape `stacked` (the two-input closure above) expects, and
    // is FP8 where that closure only takes dense F16/F32 — hence its own
    // closure rather than a third case bolted onto either.
    let stacked_fp8_2 = |a: Piece<'_>, b: Piece<'_>, total: &mut usize| -> Result<Option<Matrix>> {
        if !fuse_ffn {
            return Ok(None);
        }
        let (ba, ty, k, n_a) = a;
        let (bb, ty_b, k_b, n_b) = b;
        if ty != WeightType::F8E4M3
            || ty_b != ty
            || k_b != k
            || !n_a.is_multiple_of(infero_kernels::fp8::FP8_BLOCK)
            || !n_b.is_multiple_of(infero_kernels::fp8::FP8_BLOCK)
        {
            return Ok(None);
        }
        let ab = infero_kernels::fp8::concat2(ba, n_a, bb, n_b, k);
        Ok(Some(upload(&ab, ty, k, n_a + n_b, total)?))
    };

    // Two same-shaped projections of one input, stacked along the output.
    //
    // `in_proj_a` and `in_proj_b` are `value_heads` rows — 48 against a 5120
    // contraction — so their bytes want 0.34 us and each launch measured 14.2,
    // twice a layer and 96 times a decode step for 1.36 ms. Row-major means the
    // stack is byte concatenation and nothing has to be interleaved. Returns
    // `None` when the two are not the same dense type over the same `k`, in
    // which case the caller keeps its two launches.
    let stacked2 = |a: Piece<'_>, b: Piece<'_>, total: &mut usize| -> Result<Option<Matrix>> {
        let (ba, ty, k, n_a) = a;
        let (bb, ty_b, k_b, n_b) = b;
        if ty != ty_b || k != k_b || !matches!(ty, WeightType::F16 | WeightType::F32) {
            return Ok(None);
        }
        let mut out = Vec::with_capacity(ba.len() + bb.len());
        out.extend_from_slice(ba);
        out.extend_from_slice(bb);
        Ok(Some(upload(&out, ty, k, n_a + n_b, total)?))
    };

    // Where the text model sits. A multimodal export nests it under
    // `language_model`, so the same tensor is
    // `model.language_model.embed_tokens.weight` there and
    // `model.embed_tokens.weight` everywhere else. Probed rather than derived
    // from the architecture name: the nesting is a property of how the
    // checkpoint was written.
    //
    // The layer prefix below is derived from this rather than probed separately.
    // Probing them independently is how the first attempt at the 27B got the
    // layers right and the embedding wrong, and failed one tensor into the load.
    let stem = ["model.language_model", "model"]
        .into_iter()
        .find(|s| w.get(&format!("{s}.embed_tokens.weight")).is_some())
        .context(
            "found no embedding under `model.embed_tokens.weight` or \
             `model.language_model.embed_tokens.weight`; the checkpoint's tensor \
             names are not ones this loader recognises",
        )?;
    tracing::info!(stem, "text model tensors");

    let embd = w.tensor(&format!("{stem}.embed_tokens.weight"))?;
    // `to_f16` rather than `embd.data`: this uploaded the mapping's bytes and
    // labelled them `F16`, which reinterprets bf16 bit patterns as halves when
    // the checkpoint stores BF16 — no error, just a wrong number for every
    // token. Qwen3's AWQ export writes every float as BF16; Llama-3.1 and
    // Qwen2.5 write F16, which is why the raw upload had never been wrong
    // before and why both of those models kept working while Qwen3 produced
    // degenerate repetition.
    //
    // The embedding is the first operation in the forward pass, so this is the
    // one place where being wrong is invisible downstream: the block that
    // follows normalizes the magnitude away, leaving a residual stream whose
    // RMS climbs smoothly through all 36 layers while carrying nonsense. A
    // magnitude curve cannot detect it; only comparing values can.
    let embd_halves = embd.to_f16()?;
    // Safety: f16 is a transparent u16, so these are already the little-endian
    // halves the device expects; the view does not outlive `embd_halves`.
    let embd_bytes = unsafe {
        std::slice::from_raw_parts(embd_halves.as_ptr() as *const u8, embd_halves.len() * 2)
    };
    let token_embd = upload(
        embd_bytes,
        WeightType::F16,
        embd.shape[1],
        embd.shape[0],
        &mut device_bytes,
    )?;
    let output_norm = vector(&format!("{stem}.norm.weight"), &mut device_bytes)?;
    let output = if cfg.tied_embeddings {
        None
    } else {
        let h = w.tensor("lm_head.weight")?;
        let (n, k) = (h.shape[0], h.shape[1]);
        // `INFERO_LM_HEAD=f16` keeps the matrix as it came, which prices the Q8_0
        // path against twice the bytes. On a card with a small L2 the quantized
        // path reads 558 MB at 90 GB/s, and the question is whether that is the
        // bytes or the layout: a Q8_0 block is 34 bytes, so its quants are only
        // ever halfword-aligned and `mmq_load_w_q8_0` reads them two at a time.
        // `to_f16` rather than `h.data` / `as_f16`: Qwen3's AWQ export stores
        // lm_head as BF16. Uploading its bytes as `WeightType::F16` would
        // reinterpret bf16 bit patterns as halves — no error, just wrong
        // numbers everywhere the vocab projection is read.
        let halves = h.to_f16()?;
        // The two working control checkpoints both store lm_head as F16, so
        // `to_f16` returns a borrow for them and its BF16 branch was never
        // exercised by a model known to produce sane output. Log the first few
        // converted values so they can be checked against the file directly.
        if std::env::var_os("INFERO_LM_HEAD_PROBE").is_some() {
            let head: Vec<f32> = halves.iter().take(8).map(|x| f32::from(*x)).collect();
            tracing::info!(
                dtype = ?h.dtype,
                n = halves.len(),
                ?head,
                "lm_head probe: first 8 converted values"
            );
        }
        if std::env::var("INFERO_LM_HEAD").as_deref() == Ok("f16") {
            tracing::info!(mib = (halves.len() * 2) >> 20, "vocab projection kept f16");
            // Safety: f16 is a transparent u16, so the halves are already the
            // little-endian byte layout the device wants; the view does not
            // outlive `halves`.
            let bytes = unsafe {
                std::slice::from_raw_parts(halves.as_ptr() as *const u8, halves.len() * 2)
            };
            Some(upload(bytes, WeightType::F16, k, n, &mut device_bytes)?)
        } else {
            let q = quantize_f16_to_q8_0(halves.as_ref(), k).context("quantizing lm_head")?;
            tracing::info!(
                from_mib = h.data.len() >> 20,
                to_mib = q.len() >> 20,
                "vocab projection quantized to Q8_0"
            );
            Some(upload(&q, WeightType::Q8_0, k, n, &mut device_bytes)?)
        }
    };
    // And the split layout for the batched path, when there is room for both.
    // `INFERO_LM_HEAD=packed` keeps only the packed one, which is the A/B.
    let output_split = match (&output, std::env::var("INFERO_LM_HEAD").as_deref()) {
        (Some(o), Ok("packed")) => {
            let _ = o;
            None
        }
        (Some(o), _) if o.ty == WeightType::Q8_0 => {
            let h = w.tensor("lm_head.weight")?;
            let (n, k) = (h.shape[0], h.shape[1]);
            let free = dev.mem_info().map(|(f, _)| f).unwrap_or(0);
            let want = n * k * 17 / 16;
            if want * 3 < free {
                let q = infero_kernels::awq::quantize_f16_to_q8_0_split(h.to_f16()?.as_ref(), k)
                    .context("quantizing lm_head, split")?;
                tracing::info!(mib = q.len() >> 20, "vocab projection also split");
                Some(upload(&q, WeightType::Q8_0S, k, n, &mut device_bytes)?)
            } else {
                tracing::info!(want_mib = want >> 20, free_mib = free >> 20,
                               "no room for the split vocab projection");
                None
            }
        }
        _ => None,
    };
    device_bytes += freq_factors.len() * 4;
    let rope_freqs = dev.stream().clone_htod(freq_factors)?;
    let mrope_axis = dev
        .stream()
        .clone_htod(&mrope_axis_table(cfg.rotary_dim, cfg.mrope_section))?;

    // Where the decoder layers live. A multimodal checkpoint nests the text
    // model under `language_model`, so the same layer is
    // `model.language_model.layers.0` there and `model.layers.0` everywhere
    // else. Probe rather than branch on the architecture name: the prefix is a
    // property of how the checkpoint was exported, not of the model.
    let layer_prefix = format!("{stem}.layers");
    anyhow::ensure!(
        [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "linear_attn.in_proj_qkv.weight",
        ]
        .iter()
        .any(|leaf| w.get(&format!("{layer_prefix}.0.{leaf}")).is_some()),
        "the embedding is under `{stem}` but there is no layer 0 under \
         `{layer_prefix}`; this checkpoint splits the text model across two \
         prefixes and the loader assumes one"
    );
    tracing::info!(prefix = layer_prefix, "decoder layers");

    let dense_ffn = |p: &str, total: &mut usize| -> Result<DenseFfn> {
        let (w_gate, gate_bytes, gate_ty, gate_k, gate_n) =
            projection_with_bytes(&format!("{p}.mlp.gate_proj"), total)?;
        let (w_up, up_bytes, up_ty, up_k, up_n) =
            projection_with_bytes(&format!("{p}.mlp.up_proj"), total)?;
        let __t_stack = std::time::Instant::now();
        let a = (gate_bytes.as_slice(), gate_ty, gate_k, gate_n);
        let b = (up_bytes.as_slice(), up_ty, up_k, up_n);
        // `stacked` only takes dense F16/F32; FP8's disjoint quant+scale
        // layout is `stacked_fp8_2`'s case instead. Exactly one of the two
        // can match a given `ty`, so trying both in order is safe.
        let w_gate_up = match stacked(a, b, total)? {
            Some(m) => Some(m),
            None => stacked_fp8_2(a, b, total)?,
        };
        STACK_NS.fetch_add(__t_stack.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let w_down = projection(&format!("{p}.mlp.down_proj"), total)?;
        Ok(DenseFfn { w_gate, w_up, w_gate_up, w_down })
    };

    // One projection of every expert, concatenated in expert order.
    //
    // The checkpoint spells this as `n_experts` separate tensors and they are
    // read one at a time, so the only copy is into the buffer that gets
    // uploaded — but that buffer is the whole projection for the layer, 115 MiB
    // on Qwen3-30B-A3B, which is why it is built and dropped per projection
    // rather than per layer.
    let expert_projection = |p: &str, leaf: &str, n_experts: usize, total: &mut usize| -> Result<Experts> {
        let mut bytes: Vec<u8> = Vec::new();
        let mut shape: Option<(WeightType, usize, usize, usize)> = None;
        for e in 0..n_experts {
            let (b, ty, k, n) = projection_bytes(&format!("{p}.mlp.experts.{e}.{leaf}"))?;
            match shape {
                None => {
                    bytes.reserve(b.len() * n_experts);
                    shape = Some((ty, k, n, b.len()));
                }
                // Every expert has to encode to the same number of bytes or the
                // stride is a lie, and a wrong stride reads the tail of one
                // expert as the head of the next. That produces plausible
                // activations, so it has to be an error here rather than a
                // surprise later.
                Some((ty0, k0, n0, len0)) => anyhow::ensure!(
                    (ty, k, n, b.len()) == (ty0, k0, n0, len0),
                    "expert {e}'s {leaf} is {ty} [{n}, {k}] in {} bytes where \
                     expert 0 is {ty0} [{n0}, {k0}] in {len0}",
                    b.len()
                ),
            }
            bytes.extend_from_slice(&b);
        }
        let (ty, k, n, stride) = shape.context("a sparse layer with no experts")?;
        *total += bytes.len();
        Ok(Experts {
            ty,
            k,
            n,
            n_experts,
            stride,
            storage: Storage::Device(dev.stream().clone_htod(&bytes)?),
        })
    };

    let expert_block = |p: &str, total: &mut usize| -> Result<MoeWeights> {
        let n_experts = cfg
            .moe
            .as_ref()
            .context("a sparse layer with no MoeConfig")?
            .n_experts;
        Ok(MoeWeights {
            // `mlp.gate`, not `mlp.gate_proj` — the router and an expert's gate
            // projection are one character apart in the checkpoint and mean
            // entirely different things. Qwen3-MoE's AWQ export lists
            // `mlp.gate` in `modules_to_not_convert`, so this one comes through
            // `projection_bytes`'s unquantized branch.
            router: projection(&format!("{p}.mlp.gate"), total)?,
            gate: expert_projection(p, "gate_proj", n_experts, total)?,
            up: expert_projection(p, "up_proj", n_experts, total)?,
            down: expert_projection(p, "down_proj", n_experts, total)?,
        })
    };

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let p = format!("{layer_prefix}.{i}");
        // Which mixer this block has, decided by which tensors exist.
        //
        // `text_config.layer_types` says the same thing, and reading it would
        // work. The tensors are the stronger signal: if the config and the
        // weights ever disagree, this way fails with a missing tensor at the
        // layer in question, where trusting the config would slice a projection
        // that is not there — or worse, find one of the right size and mean
        // something else by it. Deriving the pattern from
        // `full_attention_interval` instead would additionally bake in a stride
        // that this checkpoint happens to have and the next need not.
        let is_linear = w
            .get(&format!("{p}.linear_attn.in_proj_qkv.weight"))
            .is_some();

        // Sparse or dense, decided the same way as the mixer: by which tensors
        // exist. `MoeConfig::is_sparse` says the same thing and is checked
        // against this, because a config that disagrees with the checkpoint is
        // the case worth failing loudly on rather than resolving in favour of
        // either.
        let has_experts = w
            .get(&format!("{p}.mlp.experts.0.gate_proj.qweight"))
            .or_else(|| w.get(&format!("{p}.mlp.experts.0.gate_proj.weight")))
            .is_some();
        let sparse = match (&cfg.moe, has_experts) {
            (Some(m), true) => {
                anyhow::ensure!(
                    m.is_sparse(i),
                    "layer {i} has expert weights but the config's \
                     decoder_sparse_step / mlp_only_layers make it dense"
                );
                true
            }
            (Some(m), false) => {
                anyhow::ensure!(
                    !m.is_sparse(i),
                    "the config makes layer {i} sparse but it has no \
                     `mlp.experts.0.gate_proj`"
                );
                false
            }
            (None, true) => anyhow::bail!(
                "layer {i} has expert weights but the config names no \
                 num_experts; a sparse checkpoint read as dense would answer \
                 from a fraction of its parameters"
            ),
            (None, false) => false,
        };

        let __t_attn = std::time::Instant::now();
        let attn = if is_linear {
            None
        } else {
            let (wq, q_bytes, q_ty, q_k, q_n) =
                projection_with_bytes(&format!("{p}.self_attn.q_proj"), &mut device_bytes)?;
            // `q_proj` producing twice the attention width means it carries a
            // gate interleaved with the query, per head. Detected from the
            // shape rather than from the config's `attn_output_gate`, because
            // the shape is what the rest of the code has to agree with.
            let output_gate = wq.n == 2 * cfg.d_attn();
            anyhow::ensure!(
                wq.n == cfg.d_attn() || output_gate,
                "layer {i} q_proj has {} columns; expected {} for a plain query \
                 or {} for a query and its gate",
                wq.n,
                cfg.d_attn(),
                2 * cfg.d_attn(),
            );
            let (wk, k_bytes, k_ty, k_k, k_n) =
                projection_with_bytes(&format!("{p}.self_attn.k_proj"), &mut device_bytes)?;
            let (wv, v_bytes, v_ty, v_k, v_n) =
                projection_with_bytes(&format!("{p}.self_attn.v_proj"), &mut device_bytes)?;
            // The fused QKV stack assumes three same-shaped projections of
            // one input; a gated q_proj is twice as wide as the stack
            // expects, so leave it unfused rather than mis-slice it.
            let w_qkv = if output_gate {
                None
            } else {
                stacked3(
                    (q_bytes.as_slice(), q_ty, q_k, q_n),
                    (k_bytes.as_slice(), k_ty, k_k, k_n),
                    (v_bytes.as_slice(), v_ty, v_k, v_n),
                    &mut device_bytes,
                )?
            };
            Some(AttnWeights {
                wq,
                wk,
                wv,
                wo: projection(&format!("{p}.self_attn.o_proj"), &mut device_bytes)?,
                bq: optional_vector(&format!("{p}.self_attn.q_proj.bias"), &mut device_bytes)?,
                bk: optional_vector(&format!("{p}.self_attn.k_proj.bias"), &mut device_bytes)?,
                bv: optional_vector(&format!("{p}.self_attn.v_proj.bias"), &mut device_bytes)?,
                bo: optional_vector(&format!("{p}.self_attn.o_proj.bias"), &mut device_bytes)?,
                q_norm: optional_vector(
                    &format!("{p}.self_attn.q_norm.weight"), &mut device_bytes)?,
                k_norm: optional_vector(
                    &format!("{p}.self_attn.k_norm.weight"), &mut device_bytes)?,
                w_qkv,
                // The GGUF loader's narrower fallback for a gated `wq`
                // (`stacked2_gguf` on `wk`/`wv` alone) is GGUF-specific --
                // AWQ's `stacked2` only takes F16/F32, and an AWQ
                // checkpoint's `k_proj`/`v_proj` are typically quantized, so
                // it would return `None` here regardless. Left unattempted
                // rather than adding an unused code path.
                w_kv: None,
                output_gate,
            })
        };
        ATTN_BLOCK_NS.fetch_add(__t_attn.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);

        let __t_gdn = std::time::Instant::now();
        let gdn = if is_linear {
            let l = format!("{p}.linear_attn");
            let (in_proj_qkv, qkv_bytes, qkv_ty, qkv_k, qkv_n) =
                projection_with_bytes(&format!("{l}.in_proj_qkv"), &mut device_bytes)?;
            let (in_proj_z, z_bytes, z_ty, z_k, z_n) =
                projection_with_bytes(&format!("{l}.in_proj_z"), &mut device_bytes)?;
            let (in_proj_a, a_bytes, a_ty, a_k, a_n) =
                projection_with_bytes(&format!("{l}.in_proj_a"), &mut device_bytes)?;
            let (in_proj_b, b_bytes, b_ty, b_k, b_n) =
                projection_with_bytes(&format!("{l}.in_proj_b"), &mut device_bytes)?;
            let in_proj_ba = stacked2(
                (a_bytes.as_slice(), a_ty, a_k, a_n),
                (b_bytes.as_slice(), b_ty, b_k, b_n),
                &mut device_bytes,
            )?;
            let in_proj_qz = stacked_fp8_2(
                (qkv_bytes.as_slice(), qkv_ty, qkv_k, qkv_n),
                (z_bytes.as_slice(), z_ty, z_k, z_n),
                &mut device_bytes,
            )?;
            Some(GdnWeights {
                in_proj_qkv,
                in_proj_z,
                in_proj_a,
                in_proj_b,
                in_proj_ba,
                in_proj_qz,
                conv1d: vector(&format!("{l}.conv1d.weight"), &mut device_bytes)?,
                a_log: vector(&format!("{l}.A_log"), &mut device_bytes)?,
                dt_bias: vector(&format!("{l}.dt_bias"), &mut device_bytes)?,
                norm: vector(&format!("{l}.norm.weight"), &mut device_bytes)?,
                out_proj: projection(&format!("{l}.out_proj"), &mut device_bytes)?,
            })
        } else {
            None
        };
        GDN_BLOCK_NS.fetch_add(__t_gdn.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);

        let __t_ffn = std::time::Instant::now();
        let dense_val = if sparse { None } else { Some(dense_ffn(&p, &mut device_bytes)?) };
        FFN_BLOCK_NS.fetch_add(__t_ffn.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);

        layers.push(Layer {
            attn_norm: vector(&format!("{p}.input_layernorm.weight"), &mut device_bytes)?,
            attn,
            gdn,
            ffn_norm: vector(
                &format!("{p}.post_attention_layernorm.weight"),
                &mut device_bytes,
            )?,
            dense: dense_val,
            moe: if sparse {
                Some(expert_block(&p, &mut device_bytes)?)
            } else {
                None
            },
            blob: None,
        });
    }

    tracing::info!(
        layers = cfg.n_layers,
        vram_mib = device_bytes >> 20,
        ms = started.elapsed().as_millis(),
        repack_ms = REPACK_NS.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000,
        upload_ms = UPLOAD_NS.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000,
        attn_ms = ATTN_BLOCK_NS.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000,
        gdn_ms = GDN_BLOCK_NS.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000,
        ffn_ms = FFN_BLOCK_NS.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000,
        stack_ms = STACK_NS.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000,
        "awq weights loaded"
    );
    Ok(Weights {
        token_embd,
        layers,
        output_norm,
        output,
        output_split,
        rope_freqs,
        mrope_axis,
        device_bytes,
        host_bytes: 0,
        max_blob_bytes: 0,
    })
}

fn upload_matrix(
    dev: &Device,
    f: &Gguf,
    mapped: Option<&Arc<Buf<u8>>>,
    name: &str,
    total: &mut usize,
) -> Result<Matrix> {
    let (ty, k, n, n_bytes) = describe(f, name)?;
    *total += n_bytes;
    // Aliased when the backend can alias, copied when it cannot. The two
    // produce the same `View` for every caller above this, which is why the
    // engine does not know which one it got.
    let storage = match mapped {
        Some(file) => Storage::Mapped {
            file: Arc::clone(file),
            offset: f.file_offset(f.tensor(name)?),
        },
        None => {
            let bytes = f.tensor_data(name)?;
            Storage::Device(
                dev.stream()
                    .clone_htod(bytes)
                    .with_context(|| format!("uploading {name} ({} MiB)", bytes.len() >> 20))?,
            )
        }
    };
    Ok(Matrix {
        ty,
        k,
        n,
        n_bytes,
        storage,
        cutlass_weight: Default::default(),
    })
}

/// Which dimension of a `[k, n]` (ggml `[in, out]`) linear weight a
/// tensor-parallel rank shards.
#[derive(Clone, Copy)]
enum ShardAxis {
    /// Column-parallel: shard `n` (the output dim). Each rank computes its
    /// own slice of the output independently -- no communication needed
    /// until whatever consumes it (Q/K/V, GDN's input projection, FFN
    /// gate/up).
    Output,
    /// Row-parallel: shard `k` (the input/contraction dim). Each rank's
    /// result is a partial sum over its own slice of the input and needs an
    /// `ncclAllReduce` with the other ranks before it's the real answer
    /// (attention/GDN output projections, FFN down).
    Input,
}

/// Shard a value-head-indexed *row* range, respecting GGUF's real value-head
/// tiling (`LinearAttnConfig::v_heads_tiled`) rather than assuming a
/// contiguous grouping.
///
/// llama.cpp reorders a checkpoint's value heads to `[G0_v0, G1_v0, ...,
/// G0_v1, G1_v1, ...]` -- value head `h`'s real row is `h % key_heads_full`,
/// not `h` itself, whenever one key head serves more than one value head
/// (`heads_per_key() > 1`, GQA-style). A naive contiguous
/// `(tp_rank*rank_span)..((tp_rank+1)*rank_span)` shard of the un-permuted
/// row index silently mixes value heads belonging to OTHER ranks' key heads
/// into this rank's shard while dropping some of this rank's own -- wrong
/// data, not a crash, which is exactly why this bug (found live, on the
/// real `Qwen3.8-27B-Q8_0.gguf` checkpoint under TP=2) produced
/// coherent-but-not-bit-exact output rather than garbage: the same failure
/// shape `v_heads_tiled`'s own doc comment already describes for reading
/// the tiling flag backwards.
///
/// `key_heads_full`/`heads_per_key` describe the *pre-shard* checkpoint (the
/// caller must multiply the already-divided `Config::shard_for_tp` count
/// back up by `tp_size`); `row_span` is how many raw tensor rows one value
/// head occupies: `value_head_dim` for a real per-head vector (`attn_qkv`'s
/// V segment, `attn_gate`, `conv1d`'s V channels), or `1` for a
/// scalar-a-head tensor (`ssm_alpha`/`ssm_beta`/`ssm_a`/`ssm_dt.bias`).
///
/// Returns this rank's `heads_per_key` groups' rows concatenated in tile
/// order -- exactly what a real, standalone `v_heads_tiled` checkpoint at
/// this rank's own (smaller) head counts would look like, so nothing
/// downstream needs to know sharding happened.
///
/// `row_offset` is a starting row for the value-head axis when it isn't the
/// tensor's own row 0 -- the combined `[q|k|v]` tensors (`attn_qkv.weight`,
/// `ssm_conv1d.weight`) put Q/K first, so V's tiling arithmetic needs
/// `2*full_kd` added to every row it computes; pass `0` for a standalone
/// value-shaped tensor (`attn_gate.weight`, `ssm_alpha`/`ssm_beta`/`ssm_a`/
/// `ssm_dt.bias`).
fn shard_tiled_value_rows(
    f: &Gguf,
    info: &TensorInfo,
    key_heads_full: usize,
    heads_per_key: usize,
    row_span: usize,
    row_offset: usize,
    tp_rank: usize,
    tp_size: usize,
) -> Result<Vec<u8>> {
    anyhow::ensure!(
        key_heads_full % tp_size == 0,
        "key_heads {key_heads_full} does not divide tp_size {tp_size}"
    );
    let key_heads_rank = key_heads_full / tp_size;
    let mut bytes = Vec::new();
    for tile in 0..heads_per_key {
        let tile_base = row_offset + tile * key_heads_full * row_span;
        let start = tile_base + tp_rank * key_heads_rank * row_span;
        let end = start + key_heads_rank * row_span;
        bytes.extend(f.tensor_shard(info, start..end)?);
    }
    Ok(bytes)
}

/// Column-range counterpart of [`shard_tiled_value_rows`], for the one
/// GDN tensor whose value-head axis is the *contraction* dimension
/// (row-parallel sharding -- `ssm_out.weight`'s input width is
/// `value_dim`). Same tiling reasoning, via
/// [`Gguf::tensor_shard_cols_multi`] since the `heads_per_key` column bands
/// must be interleaved per row, not concatenated end to end.
fn shard_tiled_value_cols(
    f: &Gguf,
    info: &TensorInfo,
    key_heads_full: usize,
    heads_per_key: usize,
    col_span: usize,
    tp_rank: usize,
    tp_size: usize,
) -> Result<Vec<u8>> {
    anyhow::ensure!(
        key_heads_full % tp_size == 0,
        "key_heads {key_heads_full} does not divide tp_size {tp_size}"
    );
    let key_heads_rank = key_heads_full / tp_size;
    let ranges: Vec<std::ops::Range<usize>> = (0..heads_per_key)
        .map(|tile| {
            let tile_base = tile * key_heads_full * col_span;
            let start = tile_base + tp_rank * key_heads_rank * col_span;
            start..(start + key_heads_rank * col_span)
        })
        .collect();
    f.tensor_shard_cols_multi(info, &ranges)
}

/// Like [`upload_matrix`], but reads only this rank's `1/tp_size` slice of
/// `name`'s bytes directly off disk (via [`Gguf::tensor_shard`]/
/// [`Gguf::tensor_shard_cols`]) rather than the whole tensor -- never
/// materializes the full weight, on this rank or any other. Always copies
/// (ignores the `mapped`-file-aliasing optimization `upload_matrix` uses
/// when available): a shard is a real subset of the file's bytes, not a
/// contiguous region `map_file`'s whole-file aliasing can express.
fn upload_matrix_sharded(
    dev: &Device,
    f: &Gguf,
    name: &str,
    axis: ShardAxis,
    tp_rank: usize,
    tp_size: usize,
    total: &mut usize,
) -> Result<Matrix> {
    let ty = WeightType::from_ggml(f.tensor(name)?.ty).with_context(|| format!("tensor {name}"))?;
    let info = f.tensor(name)?.clone();
    let (full_k, full_n) = (info.dims[0] as usize, info.dims[1] as usize);
    let (bytes, k, n) = match axis {
        ShardAxis::Output => {
            anyhow::ensure!(
                full_n % tp_size == 0,
                "{name}: output width {full_n} does not divide tp_size {tp_size}"
            );
            let shard_n = full_n / tp_size;
            let range = (tp_rank * shard_n)..((tp_rank + 1) * shard_n);
            (f.tensor_shard(&info, range)?, full_k, shard_n)
        }
        ShardAxis::Input => {
            anyhow::ensure!(
                full_k % tp_size == 0,
                "{name}: input width {full_k} does not divide tp_size {tp_size}"
            );
            let shard_k = full_k / tp_size;
            let range = (tp_rank * shard_k)..((tp_rank + 1) * shard_k);
            (f.tensor_shard_cols(&info, range)?, shard_k, full_n)
        }
    };
    let n_bytes = bytes.len();
    *total += n_bytes;
    let storage = Storage::Device(
        dev.stream()
            .clone_htod(&bytes)
            .with_context(|| format!("uploading sharded {name} ({} MiB)", n_bytes >> 20))?,
    );
    Ok(Matrix {
        ty,
        k,
        n,
        n_bytes,
        storage,
        cutlass_weight: Default::default(),
    })
}

/// Copy a layer's matrices into one page-locked blob and describe each one's
/// place inside it.
fn pack_layer(dev: &Device, f: &Gguf, names: &[String]) -> Result<(Vec<Matrix>, LayerBlob)> {
    let mut described = Vec::with_capacity(names.len());
    let mut offsets = Vec::with_capacity(names.len());
    let mut total = 0usize;
    for name in names {
        let d = describe(f, name)?;
        offsets.push(total);
        total += d.3.next_multiple_of(BLOB_ALIGN);
        described.push(d);
    }

    // Safety: the allocation is fully written below before any read, and the
    // handle owns the memory for as long as the weights live.
    let mut host = unsafe { dev.context().alloc_pinned::<u8>(total) }
        .with_context(|| format!("allocating {} MiB of pinned host memory", total >> 20))?;
    {
        let dst = host.as_mut_slice()?;
        // Padding is never read, but leaving it uninitialized would make the
        // DMA copy indeterminate bytes into VRAM.
        dst.fill(0);
        for (name, &offset) in names.iter().zip(&offsets) {
            let src = f.tensor_data(name)?;
            dst[offset..offset + src.len()].copy_from_slice(src);
        }
    }

    let matrices = described
        .into_iter()
        .zip(&offsets)
        .map(|((ty, k, n, n_bytes), &offset)| Matrix {
            ty,
            k,
            n,
            n_bytes,
            storage: Storage::Streamed { offset },
                cutlass_weight: Default::default(),
        })
        .collect();

    Ok((matrices, LayerBlob { host, bytes: total }))
}

fn upload_optional_vector(
    dev: &Device,
    f: &Gguf,
    name: &str,
    total: &mut usize,
) -> Result<Option<Vector>> {
    match f.get_tensor(name) {
        Some(_) => Ok(Some(upload_vector(dev, f, name, total)?)),
        None => Ok(None),
    }
}

/// Like [`upload_optional_vector`], but for a bias that belongs to a
/// column-parallel (Output-axis-sharded) matrix -- Q/K/V's own biases,
/// `attn_q.bias`/`attn_k.bias`/`attn_v.bias` on a checkpoint that carries
/// them (Qwen2 does; confirmed present on the real validation checkpoint:
/// `attn_q.bias` 896 elements, `attn_k.bias`/`attn_v.bias` 128 each).
///
/// Found the hard way: `check_shapes` only validates the *matrices* -- a
/// bias vector's length is never checked against anything, so loading one
/// unsharded (this function's non-sharded sibling, used unconditionally
/// before this fix) compiles, loads, and passes every shape check while
/// silently adding the WRONG slice of the full bias to a sharded matmul's
/// output on every rank but the one whose shard happens to start at offset
/// 0 -- real, deterministic, values-only-wrong output, exactly the kind of
/// bug this file's own `check_shapes` cannot catch by construction. A
/// row-parallel matrix's bias (`attn_output.bias`, absent on this
/// checkpoint but real on others) is the opposite case and must NOT be
/// sharded here -- it applies to the row-parallel projection's OUTPUT,
/// which is full-width and only correct after the all-reduce sums every
/// rank's partial contribution, so it stays on `upload_optional_vector`.
fn upload_optional_vector_sharded(
    dev: &Device,
    f: &Gguf,
    name: &str,
    shard: Option<(usize, usize)>,
    total: &mut usize,
) -> Result<Option<Vector>> {
    let Some((tp_rank, tp_size)) = shard else {
        return upload_optional_vector(dev, f, name, total);
    };
    let Some(info) = f.get_tensor(name) else {
        return Ok(None);
    };
    let info = info.clone();
    anyhow::ensure!(info.dims.len() == 1, "{name}: expected a 1-D bias, got {:?}", info.dims);
    let full_n = info.dims[0] as usize;
    anyhow::ensure!(
        full_n % tp_size == 0,
        "{name}: length {full_n} does not divide tp_size {tp_size}"
    );
    let shard_n = full_n / tp_size;
    let host = to_f32(f.data(&info), &info).with_context(|| format!("decoding {name} ({})", info.ty))?;
    let start = tp_rank * shard_n;
    let host_shard = &host[start..start + shard_n];
    *total += host_shard.len() * 4;
    Ok(Some(dev.stream().clone_htod(host_shard)?))
}

/// Norm gains and biases are tiny, so they are converted on the host and kept
/// in f32 regardless of how the file stores them.
fn upload_vector(dev: &Device, f: &Gguf, name: &str, total: &mut usize) -> Result<Vector> {
    let info = f.tensor(name)?;
    let host =
        to_f32(f.data(info), info).with_context(|| format!("decoding {name} ({})", info.ty))?;
    *total += host.len() * 4;
    Ok(dev.stream().clone_htod(&host)?)
}

/// `ssm_a` back into the `A_log` the recurrence wants.
///
/// llama.cpp's converter stores `-exp(A_log)`, not `A_log`: this checkpoint's
/// `blk.0.ssm_a` spans [-0.3376, -0.0038], which is exactly `-exp` of the
/// reference's [-5.5625, -1.0859]. `gdn_gate_decay` computes
/// `-exp(a_log) * softplus(...)`, so feeding it the file's value directly turns
/// a decay of 0.03 into 0.97 and every linear block remembers everything.
///
/// Inverting here rather than adding a kernel variant keeps one gate kernel for
/// both loaders, which is the same trade the safetensors path makes for the
/// norm gains.
fn upload_a_log(dev: &Device, f: &Gguf, name: &str, total: &mut usize) -> Result<Vector> {
    let info = f.tensor(name)?;
    let raw = to_f32(f.data(info), info).with_context(|| format!("decoding {name}"))?;
    let host: Vec<f32> = raw
        .iter()
        .map(|a| {
            debug_assert!(*a < 0.0, "{name}: {a} is not -exp(A_log)");
            (-a).ln()
        })
        .collect();
    *total += host.len() * 4;
    Ok(dev.stream().clone_htod(&host)?)
}

fn to_f32(bytes: &[u8], info: &TensorInfo) -> Result<Vec<f32>> {
    Ok(match info.ty {
        GgmlType::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        GgmlType::F16 => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect(),
        GgmlType::BF16 => bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect(),
        other => anyhow::bail!("1-D tensors must be F32, F16 or BF16, got {other}"),
    })
}

// ----------------------------------------------------------------- the vision tower

/// The vision tower's 333 tensors, owned.
///
/// [`infero_kernels::vision::VisionWeights`] is all borrowed views, so something
/// has to hold the allocations; this is it. Everything is f16 or f32 with no
/// quantized path, and that is not an omission: the whole tower sits in the
/// checkpoint's `modules_to_not_convert`, so there are no `weight_scale_inv`
/// tensors to read and a block-dequantizing branch here would be dead code
/// pretending to be generality.
pub struct VisionTower {
    pub shape: infero_kernels::vision::VisionShape,
    pub cfg: crate::config::VisionConfig,
    patch_embed_w: Buf<half::f16>,
    patch_embed_b: Vector,
    pos_embed: Vector,
    blocks: Vec<VisionBlock>,
    merger_norm_w: Vector,
    merger_norm_b: Vector,
    merger_fc1_w: Buf<half::f16>,
    merger_fc1_b: Vector,
    merger_fc2_w: Buf<half::f16>,
    merger_fc2_b: Vector,
    pub device_bytes: usize,
}

struct VisionBlock {
    norm1_w: Vector,
    norm1_b: Vector,
    norm2_w: Vector,
    norm2_b: Vector,
    qkv_w: Buf<half::f16>,
    qkv_b: Vector,
    proj_w: Buf<half::f16>,
    proj_b: Vector,
    fc1_w: Buf<half::f16>,
    fc1_b: Vector,
    fc2_w: Buf<half::f16>,
    fc2_b: Vector,
}

impl VisionTower {
    /// Borrowed views in the shape the kernels take.
    pub fn weights(&self) -> infero_kernels::vision::VisionWeights<'_> {
        infero_kernels::vision::VisionWeights {
            patch_embed_w: self.patch_embed_w.as_view(),
            patch_embed_b: self.patch_embed_b.as_view(),
            pos_embed: self.pos_embed.as_view(),
            blocks: self
                .blocks
                .iter()
                .map(|b| infero_kernels::vision::VisionBlockWeights {
                    norm1_w: b.norm1_w.as_view(),
                    norm1_b: b.norm1_b.as_view(),
                    norm2_w: b.norm2_w.as_view(),
                    norm2_b: b.norm2_b.as_view(),
                    qkv_w: b.qkv_w.as_view(),
                    qkv_b: b.qkv_b.as_view(),
                    proj_w: b.proj_w.as_view(),
                    proj_b: b.proj_b.as_view(),
                    fc1_w: b.fc1_w.as_view(),
                    fc1_b: b.fc1_b.as_view(),
                    fc2_w: b.fc2_w.as_view(),
                    fc2_b: b.fc2_b.as_view(),
                })
                .collect(),
            merger_norm_w: self.merger_norm_w.as_view(),
            merger_norm_b: self.merger_norm_b.as_view(),
            merger_fc1_w: self.merger_fc1_w.as_view(),
            merger_fc1_b: self.merger_fc1_b.as_view(),
            merger_fc2_w: self.merger_fc2_w.as_view(),
            merger_fc2_b: self.merger_fc2_b.as_view(),
        }
    }
}

/// Load `model.visual.*`, or `None` when the checkpoint has no tower.
///
/// The two claims worth stating, because both were settled by reading the
/// checkpoint rather than the class:
///
/// * **`patch_embed.proj.weight` is `[1152, 3, 2, 16, 16]` and wants to be
///   `[1152, 1536]`.** It is a `Conv3d` whose kernel equals its stride over an
///   input already cut into patches, so the flatten is a free view and the
///   patch embedding is a GEMM. Treating it as a convolution is wasted work,
///   not a different answer.
/// * **Every bias is real.** The text tower has none at all — `attention_bias:
///   false`, bias-free MLPs — so a loader written from that habit drops twelve
///   tensors a block. Dropping only `patch_embed.proj.bias` already moves the
///   patch embedding by 3.05 out of a peak of 3.15. This is the AWQ loader
///   dropping Qwen's QKV bias again, and it reads as fluent nonsense.
///
/// The deepstack mergers are deliberately not loaded: `deepstack_visual_indexes`
/// is empty and the tensors do not exist, whatever `modules_to_not_convert`
/// lists.
pub fn load_vision(
    dev: &Device,
    w: &infero_safetensors::Shards,
    cfg: &Config,
) -> Result<Option<VisionTower>> {
    let present = w.get("model.visual.patch_embed.proj.weight").is_some();
    let Some(vc) = cfg.vision else {
        anyhow::ensure!(
            !present,
            "this checkpoint ships `model.visual.*` tensors but its config has \
             no `vision_config`; the tower's depth and hidden size decide 333 \
             tensors' shapes and guessing them is not safe"
        );
        return Ok(None);
    };
    anyhow::ensure!(
        present,
        "config describes a vision tower of depth {} but there is no \
         `model.visual.patch_embed.proj.weight`",
        vc.depth
    );
    // `deepstack_visual_indexes` is empty on this checkpoint. If it were not,
    // the tower would emit extra feature streams that the text side has to
    // consume at named layers, and loading only the trunk would quietly drop
    // them.
    anyhow::ensure!(
        w.get("model.visual.deepstack_merger_list.0.norm.weight").is_none(),
        "this checkpoint carries deepstack mergers; the tower here emits one \
         feature stream and splicing only that would drop the rest"
    );

    let started = std::time::Instant::now();
    let mut bytes = 0usize;
    let vector = |name: &str, want: usize, total: &mut usize| -> Result<Vector> {
        let t = w.tensor(name)?;
        let v = t.to_f32()?;
        anyhow::ensure!(
            v.len() == want,
            "{name} holds {} floats, expected {want}",
            v.len()
        );
        *total += v.len() * 4;
        Ok(dev.stream().clone_htod(&v)?)
    };
    // `[rows, cols]` after flattening every trailing dimension, which is what
    // makes `proj.weight`'s five dimensions a two-dimensional GEMM operand
    // without a copy.
    let matrix =
        |name: &str, rows: usize, cols: usize, total: &mut usize| -> Result<Buf<half::f16>> {
            let t = w.tensor(name)?;
            let elems: usize = t.shape.iter().product();
            anyhow::ensure!(
                t.shape[0] == rows && elems == rows * cols,
                "{name} has shape {:?}, which is not [{rows}, {cols}] however it \
                 is flattened",
                t.shape
            );
            let halves = t.to_f16()?;
            anyhow::ensure!(halves.len() == rows * cols, "{name}: {} halves", halves.len());
            *total += halves.len() * 2;
            Ok(dev.stream().clone_htod(halves.as_ref())?)
        };

    let (d, h4) = (vc.hidden, 4 * vc.hidden);
    let patch_dim = vc.in_channels * vc.temporal_patch * vc.patch * vc.patch;
    let mut blocks = Vec::with_capacity(vc.depth);
    for i in 0..vc.depth {
        let p = format!("model.visual.blocks.{i}");
        blocks.push(VisionBlock {
            norm1_w: vector(&format!("{p}.norm1.weight"), d, &mut bytes)?,
            norm1_b: vector(&format!("{p}.norm1.bias"), d, &mut bytes)?,
            norm2_w: vector(&format!("{p}.norm2.weight"), d, &mut bytes)?,
            norm2_b: vector(&format!("{p}.norm2.bias"), d, &mut bytes)?,
            // `[3 * hidden, hidden]`, whole-q-then-whole-k-then-whole-v rows —
            // not the text side's per-head interleaving.
            qkv_w: matrix(&format!("{p}.attn.qkv.weight"), 3 * d, d, &mut bytes)?,
            qkv_b: vector(&format!("{p}.attn.qkv.bias"), 3 * d, &mut bytes)?,
            proj_w: matrix(&format!("{p}.attn.proj.weight"), d, d, &mut bytes)?,
            proj_b: vector(&format!("{p}.attn.proj.bias"), d, &mut bytes)?,
            fc1_w: matrix(&format!("{p}.mlp.linear_fc1.weight"), vc.intermediate, d, &mut bytes)?,
            fc1_b: vector(&format!("{p}.mlp.linear_fc1.bias"), vc.intermediate, &mut bytes)?,
            fc2_w: matrix(&format!("{p}.mlp.linear_fc2.weight"), d, vc.intermediate, &mut bytes)?,
            fc2_b: vector(&format!("{p}.mlp.linear_fc2.bias"), d, &mut bytes)?,
        });
    }

    let tower = VisionTower {
        shape: infero_kernels::vision::VisionShape {
            depth: vc.depth,
            hidden: vc.hidden,
            heads: vc.heads,
            intermediate: vc.intermediate,
            out_hidden: vc.out_hidden,
            in_channels: vc.in_channels,
            patch: vc.patch,
            temporal_patch: vc.temporal_patch,
            merge: vc.merge,
            eps: crate::config::VisionConfig::EPS,
            rope_theta: crate::config::VisionConfig::ROPE_THETA,
        },
        cfg: vc,
        patch_embed_w: matrix(
            "model.visual.patch_embed.proj.weight",
            vc.hidden,
            patch_dim,
            &mut bytes,
        )?,
        patch_embed_b: vector("model.visual.patch_embed.proj.bias", d, &mut bytes)?,
        pos_embed: vector(
            "model.visual.pos_embed.weight",
            vc.position_embeddings * d,
            &mut bytes,
        )?,
        blocks,
        // `[hidden]`, not `[4 * hidden]`: the merger normalizes each patch
        // before it groups them. A post-shuffle norm would make this 4608 wide
        // and the shape check here is what settles it.
        merger_norm_w: vector("model.visual.merger.norm.weight", d, &mut bytes)?,
        merger_norm_b: vector("model.visual.merger.norm.bias", d, &mut bytes)?,
        merger_fc1_w: matrix("model.visual.merger.linear_fc1.weight", h4, h4, &mut bytes)?,
        merger_fc1_b: vector("model.visual.merger.linear_fc1.bias", h4, &mut bytes)?,
        merger_fc2_w: matrix(
            "model.visual.merger.linear_fc2.weight",
            vc.out_hidden,
            h4,
            &mut bytes,
        )?,
        merger_fc2_b: vector("model.visual.merger.linear_fc2.bias", vc.out_hidden, &mut bytes)?,
        device_bytes: 0,
    };
    let tower = VisionTower { device_bytes: bytes, ..tower };
    tracing::info!(
        tensors = 3 + 6 + 12 * vc.depth,
        vram_mib = bytes >> 20,
        ms = started.elapsed().as_millis(),
        "vision tower loaded"
    );
    Ok(Some(tower))
}

/// The MTP head out of llama.cpp's sidecar GGUF.
///
/// `mtp-Qwen3.8-27B-Q8_0.gguf` is a standalone file rather than a fragment: the
/// head is `blk.64` of a 65-block model, and the file repackages `token_embd`
/// and `output` so llama.cpp can open it alone. Those two are copies of the text
/// model's, not dedicated ones — the head shares, which is what
/// `mtp_use_dedicated_embeddings = false` means — so they are read past rather
/// than loaded.
///
/// That is the one place this differs from [`load_mtp`]. There, the *absence* of
/// `mtp.embed_tokens` is the evidence for sharing, and the loader refuses a
/// checkpoint that ships one. Here the file's self-containment destroys that
/// evidence: a repackaged copy and a dedicated head's own embedding look
/// identical from the tensor list. So the claim has to be carried by the
/// metadata instead, and it is the weaker of the two checks. Worth saying rather
/// than papering over.
///
/// Everything else maps without interpretation. `blk.64` is shaped exactly like
/// one of the text model's sixteen attention blocks — output gate packed into
/// the high half of `attn_q`, q/k norms per head, `post_attention_norm` before
/// the FFN — so the layer loads through the same names at index `n_layers`. The
/// four tensors with no text-model counterpart are the ones under `nextn.`.
pub fn load_mtp_gguf(dev: &Device, f: &Gguf, cfg: &Config) -> Result<Option<MtpWeights>> {
    let t = |s: &str| format!("blk.{}.{s}", cfg.n_layers);
    if f.get_tensor(&t("nextn.eh_proj.weight")).is_none() {
        return Ok(None);
    }
    // The head's depth, from the sidecar rather than from `cfg`: the main
    // checkpoint's metadata does not mention the head at all, so the file
    // carrying the tensors is the only one that can say how many layers they
    // are. `cfg.mtp_layers` stays zero on this path and is not consulted.
    let depth = f
        .usize(&f.akey("nextn_predict_layers")?)
        .context(
            "the sidecar carries `nextn.eh_proj` but no `nextn_predict_layers`; \
             the draft loop indexes layers by it and guessing is not safe",
        )?;
    anyhow::ensure!(
        depth == 1,
        "this loader builds a one-layer MTP head, the sidecar says {depth}. The \
         draft loop indexes `spec_step_idx % nextn_predict_layers`, which is only \
         the identity at one layer"
    );
    // The head's layer is a full-attention block, which the file settles without
    // interpretation. Same check as the safetensors path and for the same
    // reason: a drafter that ran a GatedDeltaNet layer would advance recurrent
    // state on every speculative token.
    anyhow::ensure!(
        f.get_tensor(&t("ssm_a")).is_none(),
        "blk.{} carries `ssm_a`, so it is a linear-attention block rather than \
         the full-attention one this drafter assumes; drafting would touch \
         recurrent state",
        cfg.n_layers
    );

    let started = std::time::Instant::now();
    let mut bytes = 0usize;
    let mapped = infero_gpu::map_file(dev, f.path())?.map(Arc::new);
    let mapped = mapped.as_ref();
    let layer = Layer {
        attn_norm: upload_vector(dev, f, &t("attn_norm.weight"), &mut bytes)?,
        attn: Some(AttnWeights {
            wq: upload_matrix(dev, f, mapped, &t("attn_q.weight"), &mut bytes)?,
            wk: upload_matrix(dev, f, mapped, &t("attn_k.weight"), &mut bytes)?,
            wv: upload_matrix(dev, f, mapped, &t("attn_v.weight"), &mut bytes)?,
            wo: upload_matrix(dev, f, mapped, &t("attn_output.weight"), &mut bytes)?,
            bq: upload_optional_vector(dev, f, &t("attn_q.bias"), &mut bytes)?,
            bk: upload_optional_vector(dev, f, &t("attn_k.bias"), &mut bytes)?,
            bv: upload_optional_vector(dev, f, &t("attn_v.bias"), &mut bytes)?,
            bo: upload_optional_vector(dev, f, &t("attn_output.bias"), &mut bytes)?,
            q_norm: upload_optional_vector(dev, f, &t("attn_q_norm.weight"), &mut bytes)?,
            k_norm: upload_optional_vector(dev, f, &t("attn_k_norm.weight"), &mut bytes)?,
            w_qkv: None,
            w_kv: None,
            output_gate: cfg.attn_output_gate,
        }),
        gdn: None,
        ffn_norm: upload_vector(dev, f, &t("post_attention_norm.weight"), &mut bytes)?,
        dense: Some(DenseFfn {
            w_gate: upload_matrix(dev, f, mapped, &t("ffn_gate.weight"), &mut bytes)?,
            w_up: upload_matrix(dev, f, mapped, &t("ffn_up.weight"), &mut bytes)?,
            w_down: upload_matrix(dev, f, mapped, &t("ffn_down.weight"), &mut bytes)?,
            w_gate_up: None,
        }),
        // The MTP head's own block is always a full-attention, dense-FFN one —
        // enforced above by refusing a sidecar whose layer carries `ssm_a` —
        // so there is nothing sparse here for a MoE checkpoint's drafter to
        // load.
        moe: None,
        blob: None,
    };
    let w = MtpWeights {
        // `[k = 2 * d_model, n = d_model]`, embedding in the low half. The name
        // is llama.cpp's evidence for the order: `eh_proj` projects `[e | h]`,
        // and the conversion writes the HF `fc` through untransposed.
        //
        // If that order were wrong the drafts would simply stop being accepted —
        // speculation verifies every token against the target model, so a broken
        // head costs throughput and never correctness. The acceptance rate is
        // the test, and it is a sharp one: near zero if this is backwards.
        fc: upload_matrix(dev, f, mapped, &t("nextn.eh_proj.weight"), &mut bytes)?,
        pre_fc_norm_embedding: upload_vector(dev, f, &t("nextn.enorm.weight"), &mut bytes)?,
        pre_fc_norm_hidden: upload_vector(dev, f, &t("nextn.hnorm.weight"), &mut bytes)?,
        norm: upload_vector(dev, f, &t("nextn.shared_head_norm.weight"), &mut bytes)?,
        layer,
        device_bytes: bytes,
    };
    dev.synchronize()?;
    tracing::info!(
        mib = bytes >> 20,
        ms = started.elapsed().as_millis(),
        "MTP head loaded from sidecar"
    );
    Ok(Some(w))
}
