//! The transformer: weight loading, the forward pass, and the KV cache.
//!
//! One decoder block, in the order the kernels run:
//!
//! ```text
//!   x  ──► rms_norm ──► q,k,v = W·x + b ──► rope ──► store kv
//!   │                                          │
//!   │                              attention over the cache
//!   │                                          │
//!   └──────────────────► + ◄── W_o · attn ─────┘
//!                        │
//!                        ├──► rms_norm ──► silu(W_g·x) * (W_u·x) ──► W_d·
//!                        │                                            │
//!                        └──────────────────► + ◄──────────────────────┘
//! ```
//!
//! Prefill and decode run the same code; the only difference is how many
//! tokens go in at once, which decides whether each projection uses the
//! dequant-fused mat-vec or cuBLAS.

mod cache;
pub mod config;
mod sampling;
pub mod weights;

use anyhow::{Context, Result};
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, CudaView, CudaViewMut};
use half::f16;
use std::sync::Arc;
use tuili_cuda::Device;
use tuili_gguf::Gguf;
use tuili_kernels::{AttnDims, BatchLayout, Kernels, KvQuant, TqTables};

pub use cache::{KvPool, SeqId};
pub use config::Config;
pub use sampling::{Sampler, SamplingParams};
pub use tuili_kernels::KvQuant as KvCacheQuant;
pub use weights::Weights;

use weights::Matrix;

/// Tokens one forward pass may carry, summed over every sequence in the batch.
///
/// Bounds the attention score buffer, which is the one activation that grows
/// with both batch size and context length.
pub const MAX_BATCH_TOKENS: usize = 256;

/// Kept for callers that still think in terms of a single sequence's prefill.
pub const PREFILL_CHUNK: usize = MAX_BATCH_TOKENS;

/// Default ceiling on sequences that may ask for logits in one pass.
///
/// Only a default: the real limit is chosen at load time from the caller's
/// concurrency, because it sizes a `rows * vocab` buffer on both sides of the
/// bus and a 128k-vocabulary model spends 0.5 MiB per row. Hard-coding it meant
/// `--max-seqs 64` started fine and then failed every request once the batch
/// grew past 32.
pub const DEFAULT_MAX_LOGIT_ROWS: usize = 32;

/// A pool holding exactly one sequence.
///
/// Continuous batching wants a shared pool and explicit sequence ids; a script
/// generating one completion does not. This is the latter, and it is what
/// [`Model::forward`] operates on.
pub struct Session {
    pool: KvPool,
    seq: SeqId,
}

impl Session {
    pub fn len(&self) -> usize {
        self.pool.len(self.seq)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Tokens that still fit.
    pub fn remaining(&self) -> usize {
        self.pool.headroom(self.seq).min(self.pool.free_slots())
    }

    pub fn max_seq(&self) -> usize {
        self.pool.max_seq()
    }

    pub fn bytes(&self) -> usize {
        self.pool.bytes()
    }

    /// Forget the conversation, returning every slot to the pool.
    pub fn clear(&mut self) {
        self.pool.truncate(self.seq, 0);
    }

    pub fn truncate(&mut self, len: usize) {
        self.pool.truncate(self.seq, len);
    }

    pub fn pool(&self) -> &KvPool {
        &self.pool
    }
}

/// One sequence's contribution to a batch.
pub struct BatchItem<'a> {
    pub seq: SeqId,
    /// Appended at the sequence's current length.
    pub tokens: &'a [u32],
    /// False for a mid-prompt chunk, whose logits nobody will read.
    pub wants_logits: bool,
}

impl<'a> BatchItem<'a> {
    pub fn new(seq: SeqId, tokens: &'a [u32]) -> Self {
        Self {
            seq,
            tokens,
            wants_logits: true,
        }
    }

    pub fn without_logits(seq: SeqId, tokens: &'a [u32]) -> Self {
        Self {
            seq,
            tokens,
            wants_logits: false,
        }
    }
}

/// Above this many tokens a projection goes through cuBLAS instead of the
/// mat-vec. The mat-vec re-reads the weights once per token, so the crossover
/// is early.
const GEMM_THRESHOLD: usize = 4;
/// Up to this many tokens, a weight type with no tensor-core GEMM repeats the
/// integer mat-vec once per token instead of taking the float path. Measured
/// against Llama-3.1-8B Q4_K_M, whose Q6_K matrices are the ones affected.
const MMVQ_REPEAT_MAX: usize = 12;
/// Granularity at which decode graphs are captured. A captured graph fixes
/// every launch parameter including `kv_len`, so it is rounded up to a bucket
/// and re-captured only when the bucket changes; attention already masks
/// `j > positions[token]`, so a longer `kv_len` costs a little wasted work and
/// changes no result.
const GRAPH_KV_BUCKET: usize = 64;

/// The bucket, overridable so the trade can be measured: coarser buckets mean
/// fewer captures and more masked KV read per step, finer ones the reverse.
fn graph_kv_bucket() -> usize {
    std::env::var("TUILI_KV_BUCKET")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(GRAPH_KV_BUCKET)
}

/// How a captured graph is instantiated, and whether it is uploaded up front.
///
/// A replay is not free. Traced under load at a batch of 32, the gap between
/// the embedding gather and the first layer's norm — which is exactly the
/// window a graph launch has to fill — had a median of 721 us on *every* step,
/// against vLLM's 14 us in the same place. That is 1.3 us for each of the
/// step's 549 nodes, which is what a graph launch costs when the driver has to
/// set the executable up again rather than replay one it has already staged.
///
/// It is not the instantiation. `TUILI_GRAPH_MODE` was added to price the two
/// alternatives and they are the same number: `autofree` 8.46 ms a step,
/// `plain` plus an explicit `upload()` 8.60, and `INSTANTIATE_FLAG_UPLOAD` is
/// rejected outright by the driver (`CUDA_ERROR_INVALID_VALUE`) because it
/// needs the `WithParams` form. Most of the 721 us is the node-level tracing
/// that measured it: the same server runs a 7.71 ms step without `nsys` and an
/// 8.72 ms step under it. Dropping the graph entirely costs 0.8 ms a step
/// (`TUILI_NO_GRAPH=1`: 9.30 against 8.49), so the graph is paying — it just
/// is not free.
///
/// The switch stays so the result is re-runnable; the default is what it has
/// always been.
fn graph_instantiate_flags() -> cudarc::driver::sys::CUgraphInstantiate_flags {
    use cudarc::driver::sys::CUgraphInstantiate_flags as F;
    static PLAIN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *PLAIN.get_or_init(|| std::env::var("TUILI_GRAPH_MODE").as_deref() == Ok("plain")) {
        // The enum has no zero variant; instantiate takes the raw value.
        return unsafe { std::mem::transmute::<u32, F>(0) };
    }
    F::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
}

/// A captured graph that can travel with the [`Model`] that owns it.
///
/// CUDA is explicit that graph objects are not internally synchronized and must
/// not be touched concurrently from two threads. That is a statement about
/// sharing, not about which thread holds them: a `Model` is moved into the
/// inference worker once and used only from there, so a graph has exactly one
/// owner for its whole life. `CudaGraph` is `!Send` only because it wraps raw
/// handles.
struct SendGraph(cudarc::driver::CudaGraph);

// Safety: as above — one owner, moved rather than shared, and every `Model`
// method takes `&mut self`.
unsafe impl Send for SendGraph {}

/// What is known about a decode shape.
enum GraphSlot {
    /// Seen once and executed normally, which is what makes every kernel it
    /// needs resident. Capture cannot load a module, so this pass has to happen
    /// before the recording one — and it has to be a *different* step, because
    /// capture records without executing: warming and capturing in the same
    /// step would apply the layers once for the warm-up and once for the
    /// replay.
    Warm,
    Ready(SendGraph),
}

/// Device-side activations, allocated once and reused for every forward pass.
struct Activations {
    /// The residual stream, `[chunk, d_model]`.
    x: CudaSlice<f32>,
    /// Normalized copy of `x` feeding the projections.
    xb: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    proj: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    ffn: CudaSlice<f32>,
    /// `[n_heads, chunk, max_seq]`
    scores: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    token_ids: CudaSlice<i32>,
    positions: CudaSlice<i32>,
    /// Per token: which sequence row it belongs to.
    seq_of: CudaSlice<i32>,
    /// Per token: the pool slot its key/value go to.
    slots: CudaSlice<i32>,
    /// Batch rows whose logits are wanted.
    logit_rows: CudaSlice<i32>,
    attn_partial: CudaSlice<f32>,
    /// Those rows, gathered and normalized.
    head_in: CudaSlice<f32>,
}

/// Double-buffered staging for offloaded layers.
///
/// One slot is being filled by the copy stream while the other is being read
/// by the compute stream, so a layer's transfer hides behind the previous
/// layer's arithmetic. The events are the only synchronization: `ready[s]`
/// says the copy has landed, `consumed[s]` says the compute stream is done
/// reading and the slot may be overwritten.
struct Offload {
    copy_stream: Arc<CudaStream>,
    stage: [CudaSlice<u8>; 2],
    ready: [CudaEvent; 2],
    consumed: [CudaEvent; 2],
    /// Which layer each slot currently holds.
    resident: [Option<usize>; 2],
    bytes_in_flight: usize,
    /// Layers transferred so far, for reporting.
    transfers: u64,
}

impl Offload {
    fn new(dev: &Device, blob_bytes: usize) -> Result<Self> {
        let ctx = dev.context();
        let stream = dev.stream();
        Ok(Self {
            copy_stream: ctx
                .new_stream()
                .context("creating the weight copy stream")?,
            stage: [
                stream.alloc_zeros::<u8>(blob_bytes)?,
                stream.alloc_zeros::<u8>(blob_bytes)?,
            ],
            ready: [ctx.new_event(None)?, ctx.new_event(None)?],
            consumed: [ctx.new_event(None)?, ctx.new_event(None)?],
            resident: [None, None],
            bytes_in_flight: blob_bytes * 2,
            transfers: 0,
        })
    }
}

/// Working space for the TurboQuant path.
///
/// The rotation is applied to keys, values and queries on the way in, and
/// undone once on the attention output — never on a cached vector.
struct TqBuffers {
    tables: TqTables,
    k_rot: CudaSlice<f32>,
    v_rot: CudaSlice<f32>,
    q_rot: CudaSlice<f32>,
    /// `S'·(Π·q)`, the query side of the QJL inner product.
    q_qjl: CudaSlice<f32>,
    /// The attention output before `Πᵀ` maps it back.
    acc_rot: CudaSlice<f32>,
}

/// Staging for the cuBLAS path: a dequantized weight matrix and f16 inputs.
struct Scratch {
    w16: CudaSlice<f16>,
    x16: CudaSlice<f16>,
    /// The activation row in Q8_1, for the integer mat-vec.
    q8_1: CudaSlice<u8>,
}

/// One row's sampling parameters plus its own uniform draw.
///
/// `top_k` is carried as its raw bits through an `f32` array so the whole
/// parameter block is one upload; the kernel reads it back as an `int`.
#[derive(Debug, Clone, Copy)]
pub struct RowSample {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub rep_penalty: f32,
    pub rnd: f64,
}

pub struct Model {
    dev: Device,
    kern: Kernels,
    cfg: Config,
    w: Weights,
    act: Activations,
    scratch: Scratch,
    tq: Option<TqBuffers>,
    offload: Option<Offload>,
    kv_quant: KvQuant,
    /// False when `TUILI_NO_MMVQ` is set, forcing decode through the float
    /// mat-vec. Read once at load; the point is to be able to A/B the integer
    /// path's accuracy against a reference that shares everything else.
    use_mmvq: bool,
    /// False when `TUILI_NO_MMQ` is set, sending batches back through
    /// `dequant_to_f16` + cuBLAS. Separate from `use_mmvq` so the tensor-core
    /// GEMM can be A/B'd without also disabling the batch-1 mat-vec.
    use_mmq: bool,
    /// Decode graphs by (tokens, kv bucket). A step issues roughly 700 kernel
    /// launches; replaying one graph removes that cost.
    graphs: std::collections::HashMap<(u64, usize, usize), GraphSlot>,
    /// Cleared by `TUILI_NO_GRAPH`, for measuring what the graphs are worth.
    use_graph: bool,
    max_logit_rows: usize,
    max_seq: usize,
    logits_host: Vec<f32>,
    /// Rows the last forward pass left in `act.logits`.
    logit_rows: usize,
    /// Device buffers for [`Model::sample_on_device`], allocated on first use.
    samp: Option<SampleBufs>,
    /// Device-time attribution for a step's three phases; see `PhaseEvents`.
    phase_ev: Option<PhaseEvents>,
}

/// Device-side scratch for sampling. Sized once, at the batch and vocabulary
/// the model was built for.
struct SampleBufs {
    params: cudarc::driver::CudaSlice<f32>,
    pen_tok: cudarc::driver::CudaSlice<i32>,
    pen_cnt: cudarc::driver::CudaSlice<i32>,
    pen_len: cudarc::driver::CudaSlice<i32>,
    rnd: cudarc::driver::CudaSlice<f64>,
    out: cudarc::driver::CudaSlice<u32>,
    stride: usize,
}

/// Where a step's *GPU* time goes, under `TUILI_PHASE_EVENTS`.
///
/// `StepPhases` below marks host timestamps, which say when work was issued and
/// not when it ran. `Profile` says when each kernel ran but charges an event
/// pair per launch — 3.44 us against kernels that take two, which is why its
/// microseconds cannot be summed. This is the middle: four events a step, on the
/// stream, so the spans are device time and the overhead is four records.
///
/// It exists to attribute the 0.4 ms a step that subtracting per-kernel
/// estimates from the wall clock could not.
struct PhaseEvents {
    ev: Vec<cudarc::driver::CudaEvent>,
    /// Accumulated spans and the step count, so one line covers many steps.
    sums: [f64; 3],
    steps: u64,
}

impl PhaseEvents {
    fn new(dev: &Device) -> Result<Option<Self>> {
        if std::env::var_os("TUILI_PHASE_EVENTS").is_none() {
            return Ok(None);
        }
        let mut ev = Vec::new();
        for _ in 0..4 {
            ev.push(
                dev.context()
                    .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?,
            );
        }
        Ok(Some(Self { ev, sums: [0.0; 3], steps: 0 }))
    }
}

/// Where a forward pass spent its wall clock, under `TUILI_STEP_TIMING`.
///
/// `forward_batch` ends with a device synchronise, so these are GPU times, not
/// launch times: the prologue's mark lands before any of the layer work has
/// been waited on, so read them as "work issued between the marks", and only
/// the last one as a settled total.
pub(crate) struct StepPhases {
    t0: Option<std::time::Instant>,
    marks: std::cell::RefCell<Vec<(usize, f64)>>,
}

impl StepPhases {
    pub(crate) fn start() -> Self {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on = *ON.get_or_init(|| std::env::var_os("TUILI_STEP_TIMING").is_some());
        Self {
            t0: on.then(std::time::Instant::now),
            marks: std::cell::RefCell::new(Vec::new()),
        }
    }

    pub(crate) fn mark(&self, which: usize) {
        if let Some(t0) = self.t0 {
            self.marks
                .borrow_mut()
                .push((which, t0.elapsed().as_secs_f64() * 1e3));
        }
    }

    /// Whether timing is on at all.
    pub(crate) fn timing(&self) -> bool {
        self.t0.is_some()
    }

    /// Milliseconds since the step began.
    pub(crate) fn since_start(&self) -> f64 {
        self.t0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e3)
    }

    pub(crate) fn report(&self) {
        let Some(t0) = self.t0 else { return };
        // One line every 64 steps: enough to see a shift, quiet enough to leave
        // on during a benchmark.
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if !N.fetch_add(1, std::sync::atomic::Ordering::Relaxed).is_multiple_of(64) {
            return;
        }
        let m = self.marks.borrow();
        let at = |i: usize| m.iter().find(|(k, _)| *k == i).map(|(_, v)| *v).unwrap_or(0.0);
        let (pre, pro, lay) = (at(2), at(0), at(1));
        let total = t0.elapsed().as_secs_f64() * 1e3;
        tracing::warn!(
            entry_ms = format!("{pre:.2}"),
            prologue_ms = format!("{:.2}", pro - pre),
            layers_ms = format!("{:.2}", lay - pro),
            head_ms = format!("{:.2}", total - lay),
            total_ms = format!("{total:.2}"),
            "forward phases"
        );
    }
}

impl Model {
    /// Load a GGUF file onto `dev` with a dense f16 KV cache, fully resident.
    pub fn load(dev: Device, f: &Gguf, max_seq: usize) -> Result<Self> {
        Self::load_quantized(dev, f, max_seq, KvQuant::F16)
    }

    /// As [`Model::load`], with a choice of KV cache encoding.
    pub fn load_quantized(
        dev: Device,
        f: &Gguf,
        max_seq: usize,
        kv_quant: KvQuant,
    ) -> Result<Self> {
        Self::load_with(dev, f, max_seq, kv_quant, usize::MAX)
    }

    /// The full constructor.
    ///
    /// `n_gpu_layers` blocks stay in VRAM; the rest live in page-locked host
    /// memory and are streamed in a layer at a time. Pass `usize::MAX` to keep
    /// everything resident.
    pub fn load_with(
        dev: Device,
        f: &Gguf,
        max_seq: usize,
        kv_quant: KvQuant,
        n_gpu_layers: usize,
    ) -> Result<Self> {
        Self::load_full(dev, f, max_seq, kv_quant, n_gpu_layers, DEFAULT_MAX_LOGIT_ROWS)
    }

    /// [`Self::load_with`] plus the number of sequences that may take logits in
    /// one pass — normally the server's `--max-seqs`.
    pub fn load_full(
        dev: Device,
        f: &Gguf,
        max_seq: usize,
        kv_quant: KvQuant,
        n_gpu_layers: usize,
        max_logit_rows: usize,
    ) -> Result<Self> {
        let max_logit_rows = max_logit_rows.clamp(1, MAX_BATCH_TOKENS);
        let cfg = Config::from_gguf(f)?;
        tracing::info!("{cfg}");

        let kern = Kernels::new(dev.clone());
        kern.warm_up()?;

        let w = Weights::load(&dev, f, &cfg, n_gpu_layers)?;
        Self::from_parts(dev, kern, cfg, w, max_seq, kv_quant, max_logit_rows)
    }

    /// An AWQ checkpoint directory, as Hugging Face ships one.
    ///
    /// The quantized projections are repacked into [`WeightType::Q4G128`] and
    /// the `f16` vocabulary projection into Q8_0 on the way in; see
    /// [`weights::load_awq`]. Everything stays resident — there is no offload
    /// path for this format, and the reason to read one is speed.
    pub fn load_awq(
        dev: Device,
        dir: impl AsRef<std::path::Path>,
        max_seq: usize,
        kv_quant: KvQuant,
        max_logit_rows: usize,
    ) -> Result<Self> {
        let max_logit_rows = max_logit_rows.clamp(1, MAX_BATCH_TOKENS);
        let dir = dir.as_ref();
        let shards = tuili_safetensors::Shards::open_dir(dir)?;
        let json = shards.json("config.json")?;
        let name = dir
            .file_name()
            .map_or("unnamed", |s| s.to_str().unwrap_or("unnamed"));
        let cfg = Config::from_hf(&json, name)?;
        tracing::info!("{cfg}");

        let kern = Kernels::new(dev.clone());
        kern.warm_up()?;

        let freqs = cfg.rope_freq_factors(&json);
        let w = weights::load_awq(&dev, &shards, &cfg, &freqs)?;
        Self::from_parts(dev, kern, cfg, w, max_seq, kv_quant, max_logit_rows)
    }

    /// Everything after the weights are in VRAM, which is the same whichever
    /// container they came out of.
    fn from_parts(
        dev: Device,
        kern: Kernels,
        cfg: Config,
        w: Weights,
        max_seq: usize,
        kv_quant: KvQuant,
        max_logit_rows: usize,
    ) -> Result<Self> {
        let max_seq = max_seq.min(cfg.context_length).max(1);
        let offload = if w.n_offloaded() > 0 {
            Some(Offload::new(&dev, w.max_blob_bytes)?)
        } else {
            None
        };
        let act = Activations::new(&dev, &cfg, max_seq, max_logit_rows)?;
        let scratch = Scratch {
            w16: dev
                .stream()
                .alloc_zeros::<f16>(cfg.max_layer_weight_elements())?,
            x16: dev
                .stream()
                .alloc_zeros::<f16>(MAX_BATCH_TOKENS * cfg.d_ff.max(cfg.d_model))?,
            q8_1: dev.stream().alloc_zeros::<u8>(
                MAX_BATCH_TOKENS * Kernels::q8_1_bytes(cfg.d_ff.max(cfg.d_model)),
            )?,
        };

        let tq = if kv_quant.is_quantized() {
            let chunk = MAX_BATCH_TOKENS;
            let kv_dim = cfg.n_kv_heads * cfg.d_head;
            Some(TqBuffers {
                tables: TqTables::new(&dev, cfg.d_head, kv_quant)?,
                k_rot: dev.stream().alloc_zeros::<f32>(chunk * kv_dim)?,
                v_rot: dev.stream().alloc_zeros::<f32>(chunk * kv_dim)?,
                q_rot: dev.stream().alloc_zeros::<f32>(chunk * cfg.d_model)?,
                q_qjl: dev.stream().alloc_zeros::<f32>(chunk * cfg.d_model)?,
                acc_rot: dev.stream().alloc_zeros::<f32>(chunk * cfg.d_model)?,
            })
        } else {
            None
        };

        let use_mmvq = std::env::var_os("TUILI_NO_MMVQ").is_none();
        if !use_mmvq {
            tracing::warn!("TUILI_NO_MMVQ set: decode will use the float mat-vec");
        }
        // Per-kernel timing records and synchronises CUDA events, which is
        // illegal on a stream that is capturing. The two tools answer different
        // questions anyway: a graph hides launch cost, and profiling measures
        // it, so asking for one turns the other off.
        let use_graph = std::env::var_os("TUILI_NO_GRAPH").is_none() && !dev.profile().enabled();
        let use_mmq = std::env::var_os("TUILI_NO_MMQ").is_none();
        if !use_mmq {
            tracing::warn!("TUILI_NO_MMQ set: batches will use dequant + cuBLAS");
        }
        let logits_host = vec![0.0; max_logit_rows * cfg.vocab_size];
        dev.synchronize()?;

        let (free, total) = dev.mem_info()?;
        tracing::info!(
            quant = %w.dominant_type(),
            vram_mib = (w.device_bytes + offload.as_ref().map_or(0, |o| o.bytes_in_flight))
                / (1 << 20),
            offloaded_mib = w.host_bytes / (1 << 20),
            offloaded_layers = w.n_offloaded(),
            gpu_free_mib = free / (1 << 20),
            gpu_total_mib = total / (1 << 20),
            "model ready"
        );

        Ok(Self {
            dev: dev.clone(),
            kern,
            cfg,
            w,
            act,
            scratch,
            tq,
            offload,
            kv_quant,
            use_mmvq,
            use_mmq,
            graphs: std::collections::HashMap::new(),
            use_graph,
            max_logit_rows,
            max_seq,
            logit_rows: 0,
            samp: None,
            phase_ev: PhaseEvents::new(&dev)?,
            logits_host,
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    pub fn kv_quant(&self) -> KvQuant {
        self.kv_quant
    }

    /// How many blocks are streamed from host memory rather than resident.
    pub fn n_offloaded_layers(&self) -> usize {
        self.w.n_offloaded()
    }

    /// (VRAM, pinned host) bytes held by weights, staging included.
    pub fn weight_bytes(&self) -> (usize, usize) {
        (
            self.w.device_bytes + self.offload.as_ref().map_or(0, |o| o.bytes_in_flight),
            self.w.host_bytes,
        )
    }

    /// Layer transfers issued since load, one per offloaded layer per forward.
    pub fn weight_transfers(&self) -> u64 {
        self.offload.as_ref().map_or(0, |o| o.transfers)
    }

    /// Make sure layer `layer` is in a staging slot, and start the next
    /// layer's transfer behind it. Returns the slot, or `None` when the layer
    /// is resident.
    ///
    /// Slot parity follows the layer index, so consecutive offloaded layers
    /// alternate and a prefetch never targets the slot being read.
    fn stage_layer(&mut self, layer: usize) -> Result<Option<usize>> {
        if !self.w.layers[layer].is_offloaded() {
            return Ok(None);
        }
        let slot = layer % 2;
        if self.offload.as_ref().unwrap().resident[slot] != Some(layer) {
            self.issue_transfer(layer)?;
        }
        // The compute stream must not run ahead of the DMA.
        let off = self.offload.as_ref().unwrap();
        self.dev.stream().wait(&off.ready[slot])?;

        // Start the next offloaded layer now, so its transfer overlaps this
        // layer's arithmetic.
        let next = layer + 1;
        if next < self.w.layers.len() && self.w.layers[next].is_offloaded() {
            self.issue_transfer(next)?;
        }
        Ok(Some(slot))
    }

    fn issue_transfer(&mut self, layer: usize) -> Result<()> {
        let slot = layer % 2;
        let blob = self.w.layers[layer]
            .blob
            .as_ref()
            .context("issue_transfer called for a resident layer")?;
        let off = self.offload.as_mut().context("no staging buffers")?;

        // Never overwrite a slot the compute stream is still reading. Waiting
        // on an event that was never recorded is a no-op, which is what makes
        // the first pass work without special-casing it.
        off.copy_stream.wait(&off.consumed[slot])?;
        off.copy_stream
            .memcpy_htod(blob.host(), &mut off.stage[slot].slice_mut(..blob.bytes))
            .with_context(|| format!("streaming layer {layer} weights"))?;
        off.ready[slot].record(&off.copy_stream)?;
        off.resident[slot] = Some(layer);
        off.transfers += 1;
        Ok(())
    }

    /// Signal that the compute stream is done with a staging slot.
    fn release_layer(&mut self, slot: Option<usize>) -> Result<()> {
        if let (Some(slot), Some(off)) = (slot, self.offload.as_ref()) {
            off.consumed[slot].record(self.dev.stream())?;
        }
        Ok(())
    }

    /// A pool sized for `max_seqs` concurrent sequences of up to `max_seq`
    /// tokens, with `n_slots` tokens of shared capacity.
    pub fn new_pool(&self, n_slots: usize, max_seqs: usize) -> Result<KvPool> {
        KvPool::new(
            &self.dev,
            &self.cfg,
            n_slots,
            max_seqs,
            self.max_seq,
            self.kv_quant,
        )
    }

    /// A pool holding exactly one sequence, for single-stream use.
    pub fn new_session(&self) -> Result<Session> {
        let mut pool = self.new_pool(self.max_seq, 1)?;
        let seq = pool.alloc().context("fresh pool had no sequence rows")?;
        Ok(Session { pool, seq })
    }

    /// Run one sequence's tokens, splitting them across passes if needed.
    ///
    /// Returns the logits for the final token. A thin wrapper over
    /// [`Model::forward_batch`] with a batch of one.
    pub fn forward(&mut self, tokens: &[u32], session: &mut Session) -> Result<&[f32]> {
        anyhow::ensure!(!tokens.is_empty(), "forward needs at least one token");
        let seq = session.seq;
        anyhow::ensure!(
            session.pool.headroom(seq) >= tokens.len(),
            "context overflow: {} cached + {} new > {}",
            session.pool.len(seq),
            tokens.len(),
            session.pool.max_seq()
        );

        let n_chunks = tokens.len().div_ceil(MAX_BATCH_TOKENS);
        for (i, chunk) in tokens.chunks(MAX_BATCH_TOKENS).enumerate() {
            let last = i + 1 == n_chunks;
            let item = if last {
                BatchItem::new(seq, chunk)
            } else {
                BatchItem::without_logits(seq, chunk)
            };
            self.forward_batch(std::slice::from_ref(&item), &mut session.pool)?;
        }
        Ok(&self.logits_host[..self.cfg.vocab_size])
    }

    /// Run a batch of sequences in one pass.
    ///
    /// Every item is appended at its sequence's current length, so a batch can
    /// freely mix a fresh prompt, a mid-prompt chunk and a dozen single decode
    /// tokens. Returns the logits of the items that asked for them, in order,
    /// as `n * vocab_size` values.
    pub fn forward_batch_device(
        &mut self,
        items: &[BatchItem<'_>],
        pool: &mut KvPool,
    ) -> Result<usize> {
        let phase = crate::StepPhases::start();
        anyhow::ensure!(!items.is_empty(), "empty batch");
        let n_tokens: usize = items.iter().map(|i| i.tokens.len()).sum();
        anyhow::ensure!(n_tokens > 0, "batch carries no tokens");
        anyhow::ensure!(
            n_tokens <= MAX_BATCH_TOKENS,
            "batch of {n_tokens} tokens exceeds the {MAX_BATCH_TOKENS} a pass can carry"
        );
        let n_logit_rows = items.iter().filter(|i| i.wants_logits).count();
        anyhow::ensure!(
            n_logit_rows <= self.max_logit_rows,
            "{n_logit_rows} sequences want logits, the limit is {}",
            self.max_logit_rows
        );
        {
            // A sequence appearing twice would have its second slice indexed
            // against a length the first slice already moved.
            let mut seen: Vec<usize> = items.iter().map(|i| i.seq.0).collect();
            seen.sort_unstable();
            let unique = seen.len();
            seen.dedup();
            anyhow::ensure!(
                seen.len() == unique,
                "a sequence appears twice in one batch"
            );
        }

        let (d, n_layers, vocab_size, rms_eps) = {
            let c = &self.cfg;
            (c.d_model, c.n_layers, c.vocab_size, c.rms_eps)
        };

        // Lay the batch out flat and claim pool slots for every new token.
        let mut token_ids = Vec::with_capacity(n_tokens);
        let mut seq_of = Vec::with_capacity(n_tokens);
        let mut positions = Vec::with_capacity(n_tokens);
        let mut slots = Vec::with_capacity(n_tokens);
        let mut logit_rows = Vec::with_capacity(n_logit_rows);
        let mut kv_len = 0usize;

        for item in items {
            let start = pool.len(item.seq);
            let taken = pool.extend(item.seq, item.tokens.len())?;
            for (k, (&tok, &slot)) in item.tokens.iter().zip(&taken).enumerate() {
                token_ids.push(tok as i32);
                seq_of.push(item.seq.0 as i32);
                positions.push((start + k) as i32);
                slots.push(slot);
            }
            kv_len = kv_len.max(start + item.tokens.len());
            if item.wants_logits {
                logit_rows.push((token_ids.len() - 1) as i32);
            }
        }

        phase.mark(2);
        let stream = self.dev.stream().clone();
        if let Some(pe) = &self.phase_ev {
            pe.ev[0].record(&stream)?;
        }
        stream.memcpy_htod(&token_ids, &mut self.act.token_ids.slice_mut(..n_tokens))?;
        stream.memcpy_htod(&seq_of, &mut self.act.seq_of.slice_mut(..n_tokens))?;
        stream.memcpy_htod(&positions, &mut self.act.positions.slice_mut(..n_tokens))?;
        stream.memcpy_htod(&slots, &mut self.act.slots.slice_mut(..n_tokens))?;
        if n_logit_rows > 0 {
            stream.memcpy_htod(
                &logit_rows,
                &mut self.act.logit_rows.slice_mut(..n_logit_rows),
            )?;
        }

        // Record where the new tokens landed before anything reads the table.
        {
            let stride = pool.table_stride();
            self.kern.write_slot_table(
                &mut pool.slot_table_mut().as_view_mut(),
                &self.act.seq_of.slice(..n_tokens),
                &self.act.positions.slice(..n_tokens),
                &self.act.slots.slice(..n_tokens),
                stride,
                n_tokens,
            )?;
        }

        self.kern.gather_rows(
            &mut self.act.x.slice_mut(..n_tokens * d),
            &self.w.token_embd.view(None)?,
            self.w.token_embd.ty,
            &self.act.token_ids.slice(..n_tokens),
            n_tokens,
            d,
        )?;

        phase.mark(0);
        let dims = AttnDims {
            n_heads: self.cfg.n_heads,
            n_kv_heads: self.cfg.n_kv_heads,
            d_head: self.cfg.d_head,
            n_slots: pool.n_slots(),
            n_tokens,
        };

        // Offloaded layers are excluded: their staging waits on copy-stream
        // events, which a capture cannot record.
        // The pool is part of the key: a graph holds that pool's device
        // pointers, and replaying it against another pool would read the wrong
        // KV cache — which is exactly what a fresh `Session` per sequence does.
        let key = (pool.id(), n_tokens, kv_len.next_multiple_of(graph_kv_bucket()));
        let graphable = self.use_graph && self.offload.is_none() && key.2 <= self.max_seq;

        match self.graphs.get(&key) {
            Some(GraphSlot::Ready(g)) if graphable => g.0.launch()?,
            slot => {
                let record = graphable && matches!(slot, Some(GraphSlot::Warm));
                let stream = self.dev.stream().clone();
                if record {
                    stream.begin_capture(
                        cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                    )?;
                }
                let kv = if graphable { key.2 } else { kv_len };
                let res = (|| -> Result<()> {
                    for layer in 0..n_layers {
                        let s = self.stage_layer(layer)?;
                        self.attention(layer, n_tokens, kv, dims, pool, s)?;
                        self.feed_forward(layer, n_tokens, s)?;
                        self.release_layer(s)?;
                    }
                    Ok(())
                })();
                if record {
                    // End the capture before reporting, or a failure inside it
                    // surfaces as STREAM_CAPTURE_INVALIDATED and hides the
                    // cause.
                    let graph = stream.end_capture(graph_instantiate_flags());
                    res?;
                    let graph = graph?.context("stream capture produced no graph")?;
                    if phase.timing() {
                        tracing::warn!(
                            ms = format!("{:.2}", phase.since_start()),
                            n_tokens,
                            kv = key.2,
                            "graph captured"
                        );
                    }
                    // Stage the executable before it is ever replayed. Free
                    // under the `UPLOAD` flag, which has already done it.
                    graph.upload()?;
                    // The capture recorded without executing, so this is the
                    // step's only execution.
                    graph.launch()?;
                    self.graphs.insert(key, GraphSlot::Ready(SendGraph(graph)));
                } else {
                    res?;
                    if graphable {
                        self.graphs.insert(key, GraphSlot::Warm);
                    }
                }
            }
        }

        if let Some(pe) = &self.phase_ev {
            pe.ev[1].record(self.dev.stream())?;
        }
        phase.mark(1);
        if n_logit_rows == 0 {
            phase.report();
            self.logit_rows = 0;
            return Ok(0);
        }

        // Only the rows someone asked for reach the vocab projection.
        self.kern.take_rows(
            &mut self.act.head_in.slice_mut(..n_logit_rows * d),
            &self.act.x.slice(..n_tokens * d),
            &self.act.logit_rows.slice(..n_logit_rows),
            n_logit_rows,
            d,
        )?;
        self.kern.rms_norm(
            &mut self.act.xb.slice_mut(..n_logit_rows * d),
            &self.act.head_in.slice(..n_logit_rows * d),
            &self.w.output_norm.as_view(),
            n_logit_rows,
            d,
            rms_eps,
        )?;
        // The batched path prefers the split layout when the loader built one:
        // same values, same order, but a row's quants are contiguous so the tile
        // loader reads sixteen bytes at a time instead of two. See
        // `mmq_load_w_q8_0s`. The mat-vec keeps the packed form.
        let packed_head = self.w.output.as_ref().unwrap_or(&self.w.token_embd);
        let head = match self.w.output_split.as_ref() {
            Some(sp) if n_logit_rows > 1 => sp,
            _ => packed_head,
        };
        // Never cuBLAS here: the vocab projection is by far the largest matrix,
        // and dequantizing it for a handful of rows would cost more than the
        // mat-vec itself.
        //
        // One row takes the integer mat-vec, more take the tensor-core GEMM,
        // the same split the layer projections use.
        //
        // This kernel was previously pinned to the GEMM at every row count to
        // keep the logits independent of batch width. That rationale does not
        // survive contact with the rest of the forward pass: `matmul` already
        // switches between the two kernels at the same boundary, so a solo
        // decode and a batched one differ well before they reach this
        // projection. Holding one matrix invariant bought nothing end to end
        // and cost 1.36 ms per token — the GEMM fills 16 token slots and at one
        // row fifteen of them are zeros, so it runs at 171 GB/s where the
        // mat-vec reaches 369 on the same 431 MiB of weights.
        let head_int = self.use_mmvq
            && Kernels::has_mmvq(head.ty)
            && d.is_multiple_of(32);
        let head_mmq = n_logit_rows > 1
            && self.use_mmq
            && Kernels::has_mmq(head.ty)
            && self.kern.device().arch() >= 80
            && Self::mmq_shape_ok(head);
        if head_int && n_logit_rows == 1 {
            let bytes = Kernels::q8_1_bytes(d);
            self.kern.quantize_q8_1(
                &mut self.scratch.q8_1.slice_mut(..bytes),
                &self.act.xb.slice(..d),
                d,
            )?;
            self.kern.mmvq(
                &mut self.act.logits.slice_mut(..vocab_size),
                &head.view(None)?,
                head.ty,
                &self.scratch.q8_1.slice(..bytes),
                d,
                vocab_size,
            )?;
        } else if head_mmq {
            let bytes = Kernels::q8_1_bytes(d);
            let total = n_logit_rows * bytes;
            self.kern.quantize_q8_1(
                &mut self.scratch.q8_1.slice_mut(..total),
                &self.act.xb.slice(..n_logit_rows * d),
                n_logit_rows * d,
            )?;
            self.kern.mmq(
                &mut self.act.logits.slice_mut(..n_logit_rows * vocab_size),
                &head.view(None)?,
                head.ty,
                &self.scratch.q8_1.slice(..total),
                d,
                vocab_size,
                n_logit_rows,
            )?;
        } else {
            self.kern.gemv(
                &mut self.act.logits.slice_mut(..n_logit_rows * vocab_size),
                &head.view(None)?,
                head.ty,
                &self.act.xb.slice(..n_logit_rows * d),
                d,
                vocab_size,
                n_logit_rows,
            )?;
        }

        if let Some(pe) = &self.phase_ev {
            pe.ev[2].record(self.dev.stream())?;
        }
        phase.report();
        self.logit_rows = n_logit_rows;
        Ok(n_logit_rows)
    }

    /// One token per row, sampled where the logits already are.
    ///
    /// `rows` carries each sequence's parameters and its own uniform draw —
    /// taken on the host from that sequence's `StdRng`, so seeding and
    /// reproducibility are exactly what they were — and `windows` its
    /// repetition window, already truncated by the caller. Only the sampled
    /// ids come back.
    ///
    /// Returns `None` when the batch is outside what the device sampler
    /// covers, which leaves the caller on the host path rather than silently
    /// sampling differently.
    pub fn sample_on_device(
        &mut self,
        rows: &[RowSample],
        windows: &[&[u32]],
    ) -> Result<Option<Vec<u32>>> {
        let vocab = self.cfg.vocab_size;
        let n = rows.len();
        anyhow::ensure!(n == windows.len(), "{n} rows against {} windows", windows.len());
        if n == 0 || n != self.logit_rows {
            return Ok(None);
        }
        let max_k = rows.iter().map(|r| r.top_k as usize).max().unwrap_or(1);
        if !Kernels::can_sample_on_device(vocab, max_k.max(1)) {
            return Ok(None);
        }

        // Each window as sorted unique ids with counts. The kernel binary
        // searches this on a bitset hit, and the counts are what let it
        // reproduce the host's non-greedy path, which penalizes a token once
        // per occurrence rather than once per token.
        let stride = windows.iter().map(|w| w.len()).max().unwrap_or(0).max(1);
        let mut tok = vec![0i32; n * stride];
        let mut cnt = vec![0i32; n * stride];
        let mut len = vec![0i32; n];
        let mut scratch: Vec<u32> = Vec::new();
        for (i, w) in windows.iter().enumerate() {
            scratch.clear();
            scratch.extend_from_slice(w);
            scratch.sort_unstable();
            let mut m = 0usize;
            let mut j = 0usize;
            while j < scratch.len() {
                let t = scratch[j];
                let mut c = 0i32;
                while j < scratch.len() && scratch[j] == t {
                    c += 1;
                    j += 1;
                }
                if (t as usize) < vocab {
                    tok[i * stride + m] = t as i32;
                    cnt[i * stride + m] = c;
                    m += 1;
                }
            }
            len[i] = m as i32;
        }

        let mut params = vec![0f32; n * 4];
        let mut rnd = vec![0f64; n];
        for (i, r) in rows.iter().enumerate() {
            params[i * 4] = r.temperature;
            params[i * 4 + 1] = r.top_p;
            params[i * 4 + 2] = f32::from_bits(r.top_k);
            params[i * 4 + 3] = r.rep_penalty;
            rnd[i] = r.rnd;
        }

        let stream = self.dev.stream().clone();
        let fits = matches!(&self.samp, Some(b) if b.stride >= stride && b.pen_len.len() >= n);
        if !fits {
            self.samp = Some(SampleBufs {
                params: stream.alloc_zeros::<f32>(self.max_logit_rows * 4)?,
                pen_tok: stream.alloc_zeros::<i32>(self.max_logit_rows * stride)?,
                pen_cnt: stream.alloc_zeros::<i32>(self.max_logit_rows * stride)?,
                pen_len: stream.alloc_zeros::<i32>(self.max_logit_rows)?,
                rnd: stream.alloc_zeros::<f64>(self.max_logit_rows)?,
                out: stream.alloc_zeros::<u32>(self.max_logit_rows)?,
                stride,
            });
        }
        let b = self.samp.as_mut().unwrap();
        let stride = b.stride;
        if stride != tok.len() / n {
            // Re-lay the rows at the buffer's stride rather than reallocating
            // for every window length the batch happens to have.
            let mut t2 = vec![0i32; n * stride];
            let mut c2 = vec![0i32; n * stride];
            let old = tok.len() / n;
            for i in 0..n {
                let m = len[i] as usize;
                t2[i * stride..i * stride + m].copy_from_slice(&tok[i * old..i * old + m]);
                c2[i * stride..i * stride + m].copy_from_slice(&cnt[i * old..i * old + m]);
            }
            tok = t2;
            cnt = c2;
        }
        stream.memcpy_htod(&params, &mut b.params.slice_mut(..n * 4))?;
        stream.memcpy_htod(&tok, &mut b.pen_tok.slice_mut(..n * stride))?;
        stream.memcpy_htod(&cnt, &mut b.pen_cnt.slice_mut(..n * stride))?;
        stream.memcpy_htod(&len, &mut b.pen_len.slice_mut(..n))?;
        stream.memcpy_htod(&rnd, &mut b.rnd.slice_mut(..n))?;

        let (params_v, tok_v, cnt_v, len_v, rnd_v) = (
            b.params.slice(..n * 4),
            b.pen_tok.slice(..n * stride),
            b.pen_cnt.slice(..n * stride),
            b.pen_len.slice(..n),
            b.rnd.slice(..n),
        );
        let mut out_v = b.out.slice_mut(..n);
        self.kern.sample_rows(
            &mut out_v,
            &self.act.logits.slice(..n * vocab),
            &params_v,
            &tok_v,
            &cnt_v,
            &len_v,
            &rnd_v,
            n,
            vocab,
            stride,
        )?;
        if let Some(pe) = &self.phase_ev {
            pe.ev[3].record(&stream)?;
        }
        let mut host = vec![0u32; n];
        stream.memcpy_dtoh(&self.samp.as_ref().unwrap().out.slice(..n), &mut host)?;
        self.dev.synchronize()?;
        // The events are settled now, so the spans can be read without a wait
        // of their own. One line every 64 steps.
        if let Some(pe) = &mut self.phase_ev {
            let spans = [
                pe.ev[0].elapsed_ms(&pe.ev[1])? as f64,
                pe.ev[1].elapsed_ms(&pe.ev[2])? as f64,
                pe.ev[2].elapsed_ms(&pe.ev[3])? as f64,
            ];
            for (a, b) in pe.sums.iter_mut().zip(spans) {
                *a += b;
            }
            pe.steps += 1;
            if pe.steps.is_multiple_of(64) {
                let k = pe.steps as f64;
                tracing::warn!(
                    layers_ms = format!("{:.3}", pe.sums[0] / k),
                    vocab_ms = format!("{:.3}", pe.sums[1] / k),
                    sample_ms = format!("{:.3}", pe.sums[2] / k),
                    gpu_ms = format!("{:.3}", pe.sums.iter().sum::<f64>() / k),
                    steps = pe.steps,
                    "phase events"
                );
            }
        }
        Ok(Some(host))
    }

    /// The logits the last [`Model::forward_batch_device`] produced, on the
    /// host. The 16 MiB copy the device sampler exists to avoid, for callers
    /// that fall back to sampling here.
    pub fn logits_host(&mut self) -> Result<&[f32]> {
        let n = self.logit_rows * self.cfg.vocab_size;
        if n == 0 {
            return Ok(&[]);
        }
        let stream = self.dev.stream().clone();
        stream.memcpy_dtoh(&self.act.logits.slice(..n), &mut self.logits_host[..n])?;
        self.dev.synchronize()?;
        Ok(&self.logits_host[..n])
    }

    /// The forward pass, with the logits brought back to the host.
    ///
    /// At a batch of 32 over a 128256-entry vocabulary this copy is 16 MiB and
    /// measured 2.19 ms of a 12.18 ms step. [`Model::sample_on_device`] is the
    /// path that does not pay it; this one remains for callers that want the
    /// logits themselves.
    pub fn forward_batch(&mut self, items: &[BatchItem<'_>], pool: &mut KvPool) -> Result<&[f32]> {
        self.forward_batch_device(items, pool)?;
        self.logits_host()
    }

    fn attention(
        &mut self,
        layer: usize,
        n: usize,
        kv_len: usize,
        dims: AttnDims,
        pool: &mut KvPool,
        slot: Option<usize>,
    ) -> Result<()> {
        let stage = slot.map(|s| &self.offload.as_ref().unwrap().stage[s]);
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let kv_dim = cfg.n_kv_heads * cfg.d_head;
        let l = &self.w.layers[layer];
        let table_stride = pool.table_stride();
        // Set below, where the decode combine either writes the output
        // projection's f16 activation or reports that it did not.
        let mut attn_f16 = false;
        // Whether q/k/v stayed in the stacked projection's output row.
        let mut packed_qkv = false;

        // One Q8_1 form for all three projections, produced by the norm that
        // writes their input. They read the same normalized residual, so
        // quantizing per matrix did the same work three times.
        let want_q = self.use_mmvq
            && [&l.wq, &l.wk, &l.wv]
                .iter()
                .all(|w| Kernels::has_mmvq(w.ty) && w.k == d);
        // Only when this group will actually take the f16 GEMM: at one token
        // the mat-vec runs instead and that path wants Q8_1.
        let want_h = self.use_mmq
            && n > 1
            && n <= tuili_kernels::MMQ_MAX_TOKENS
            && Kernels::mmq_f16_variant_for(l.wq.ty).is_some()
            && [&l.wq, &l.wk, &l.wv].iter().all(|w| {
                matches!(
                    w.ty,
                    tuili_kernels::WeightType::Q4G128
                        | tuili_kernels::WeightType::Q4G128T
                ) && w.k == d
            });
        let eps = cfg.rms_eps;
        let (shared, shared_f16) = if self.attn_norm_takes_residual(layer, n) {
            // The previous layer's FFN left its output in `proj` for this.
            self.kern.add_rms_norm_f16(
                &mut self.act.xb.slice_mut(..n * d),
                &mut self.scratch.x16.slice_mut(..n * d),
                &mut self.act.x.slice_mut(..n * d),
                &self.act.proj.slice(..n * d),
                &self.w.layers[layer].attn_norm.as_view(),
                n,
                d,
                eps,
            )?;
            (None, true)
        } else {
            Self::norm_for_group(
                &self.kern,
                &mut self.scratch,
                &mut self.act.xb,
                &self.act.x.slice(..n * d),
                &self.w.layers[layer].attn_norm.as_view(),
                n,
                d,
                eps,
                want_q,
                want_h,
            )?
        };

        // All three read the same Q8_1 activation, so one launch covers them.
        // Two hundred and twenty-five mat-vecs back to back run at 328 GB/s
        // where one alone runs at 392 — each drains before the next can start
        // — and merging this group and the FFN's removes ninety-six of those.
        if Self::fusable(&[&l.wq, &l.wk, &l.wv], shared, n, self.use_mmvq) {
            let bytes = Kernels::q8_1_bytes(d);
            let (q, k_, v) = (&mut self.act.q, &mut self.act.k, &mut self.act.v);
            self.kern.mmvq_fused3(
                &mut q.slice_mut(..d),
                &mut k_.slice_mut(..kv_dim),
                &mut v.slice_mut(..kv_dim),
                &l.wq.view(stage)?,
                &l.wk.view(stage)?,
                &l.wv.view(stage)?,
                l.wq.ty,
                &self.scratch.q8_1.slice(..bytes),
                d,
                [l.wq.n, l.wk.n, l.wv.n],
            )?;
            for (bias, out, cols) in [
                (&l.bq, &mut self.act.q, d),
                (&l.bk, &mut self.act.k, kv_dim),
                (&l.bv, &mut self.act.v, kv_dim),
            ] {
                if let Some(b) = bias {
                    self.kern
                        .add_bias(&mut out.slice_mut(..n * cols), &b.as_view(), cols, n)?;
                }
            }
        } else if let Some(w) = l.w_qkv.as_ref().filter(|_| n > 1 && want_h) {
            // One matmul for all three, then a scatter. Separately they cost
            // 14.7 + 8.5 + 8.5 us a layer at a batch of 32 because the two
            // narrow ones cannot fill the device; stacked they cost 16.7. The
            // scatter is what buys that without a row stride in `rope_qk`,
            // `store_kv` and `attn_scores`.
            //
            // `act.gate` is the staging buffer: it is the FFN's, already wide
            // enough, and nothing in this block will read it before the FFN
            // writes it again.
            let fused_w = d + 2 * kv_dim;
            Self::matmul_pre(
                &self.kern,
                &mut self.scratch,
                &mut self.act.gate.slice_mut(..n * fused_w),
                w,
                stage,
                &self.act.xb.slice(..n * d),
                n,
                self.use_mmvq,
                self.use_mmq,
                shared,
                shared_f16,
            )?;
            // Nothing has to be unpacked: `rope_qk_packed` reads `q` and `k`
            // out of this row and `store_kv2_packed` reads `k` and `v`, so the
            // scatter into three buffers is 1.5 MB and a launch a layer for an
            // index change. Biases and the quantized KV path still want the
            // unpacked form, and neither is on this model's decode path.
            packed_qkv = l.bq.is_none()
                && l.bk.is_none()
                && l.bv.is_none()
                && self.tq.is_none();
            if !packed_qkv {
                self.kern.split_qkv(
                    &mut self.act.q.slice_mut(..n * d),
                    &mut self.act.k.slice_mut(..n * kv_dim),
                    &mut self.act.v.slice_mut(..n * kv_dim),
                    &self.act.gate.slice(..n * fused_w),
                    d,
                    kv_dim,
                    n,
                )?;
            }
            for (bias, out, cols) in [
                (&l.bq, &mut self.act.q, d),
                (&l.bk, &mut self.act.k, kv_dim),
                (&l.bv, &mut self.act.v, kv_dim),
            ] {
                if let Some(b) = bias {
                    self.kern
                        .add_bias(&mut out.slice_mut(..n * cols), &b.as_view(), cols, n)?;
                }
            }
        } else {
        for (w, bias, out, cols) in [
            (&l.wq, &l.bq, &mut self.act.q, d),
            (&l.wk, &l.bk, &mut self.act.k, kv_dim),
            (&l.wv, &l.bv, &mut self.act.v, kv_dim),
        ] {
            Self::matmul_pre(
                &self.kern,
                &mut self.scratch,
                &mut out.slice_mut(..n * cols),
                w,
                stage,
                &self.act.xb.slice(..n * d),
                n,
                self.use_mmvq,
                self.use_mmq,
                shared,
                shared_f16,
            )?;
            if let Some(b) = bias {
                self.kern
                    .add_bias(&mut out.slice_mut(..n * cols), &b.as_view(), cols, n)?;
            }
        }
        }

        let fused_w = d + 2 * kv_dim;
        if packed_qkv {
            let (q, packed) = (&mut self.act.q, &mut self.act.gate);
            self.kern.rope_qk_packed(
                &mut q.slice_mut(..n * d),
                &mut packed.slice_mut(..n * fused_w),
                fused_w,
                0,
                d,
                &self.act.positions.slice(..n),
                &self.w.rope_freqs.as_view(),
                n,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.d_head,
                cfg.rope_theta,
                cfg.rope_freq_scale,
                cfg.interleaved_rope,
            )?;
        } else {
            let (q, k) = (&mut self.act.q, &mut self.act.k);
            self.kern.rope_qk(
                &mut q.slice_mut(..n * d),
                &mut k.slice_mut(..n * kv_dim),
                &self.act.positions.slice(..n),
                &self.w.rope_freqs.as_view(),
                n,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.d_head,
                cfg.rope_theta,
                cfg.rope_freq_scale,
                cfg.interleaved_rope,
            )?;
        }

        let score_len = cfg.n_heads * n * kv_len;
        let attn_scale = cfg.attn_scale();
        let (n_heads, n_kv_heads, d_head) = (cfg.n_heads, cfg.n_kv_heads, cfg.d_head);

        // These come from the activations, so they can be held across the
        // stores; the slot table lives in the pool and is taken afterwards.
        let seq_of = self.act.seq_of.slice(..n);
        let batch_positions = self.act.positions.slice(..n);

        match self.tq.as_mut() {
            None => {
                if packed_qkv {
                    let packed = self.act.gate.slice(..n * fused_w);
                    let (kc, vc) = pool.dense_mut(layer);
                    self.kern.store_kv2_packed(
                        &mut kc.as_view_mut(),
                        &mut vc.as_view_mut(),
                        &packed,
                        fused_w,
                        d,
                        d + kv_dim,
                        &self.act.slots.slice(..n),
                        n_kv_heads,
                        d_head,
                        dims.n_slots,
                        n,
                    )?;
                } else {
                    let (kc, vc) = pool.dense_mut(layer);
                    self.kern.store_kv2(
                        &mut kc.as_view_mut(),
                        &mut vc.as_view_mut(),
                        &self.act.k.slice(..n * kv_dim),
                        &self.act.v.slice(..n * kv_dim),
                        // Physical slots, not logical positions: two sequences
                        // in one batch both start at position 0 and would
                        // otherwise write over each other.
                        &self.act.slots.slice(..n),
                        n_kv_heads,
                        d_head,
                        dims.n_slots,
                        n,
                    )?;
                }

                let table = pool.slot_table().as_view();
                let batch = BatchLayout {
                    seq_of: &seq_of,
                    positions: &batch_positions,
                    slot_table: &table,
                    table_stride,
                };
                let (attn_out, partial) = (&mut self.act.attn, &mut self.act.attn_partial);
                // Whether the output projection will read its activation as
                // f16. When it will, the combine writes that copy instead of a
                // `f32_to_f16` launch reading the f32 back — the other half of
                // what `silu_mul_split_f16_f32` does for the SwiGLU product.
                // The combine only exists on the split-chunk path, so
                // `attn_decode` reports whether it actually wrote it.
                let wo_f16 = self.use_mmq
                    && n > 1
                    && n <= tuili_kernels::MMQ_MAX_TOKENS
                    && matches!(
                        l.wo.ty,
                        tuili_kernels::WeightType::Q4G128
                            | tuili_kernels::WeightType::Q4G128T
                    )
                    && Kernels::mmq_f16_variant_for_shape(l.wo.ty, l.wo.n).is_some()
                    && Self::mmq_shape_ok(&l.wo);
                // The fused decode kernel measures *level* with the three it
                // replaces on a microbenchmark, and wins by 4.7% in the served
                // engine — 4150 tok/s to 4344 at 32 clients. The difference is
                // the thing a microbenchmark cannot see: the three kernels
                // write a score matrix to HBM and read it twice more, 5.4 MB a
                // layer, and in a real step that traffic competes with the
                // weight stream rather than with itself. Per kernel, under the
                // profile, 77.5 us a layer against 66.2.
                //
                // So a kernel is not slow or fast on its own, and neither
                // number here was wrong. `TUILI_DECODE_ATTN=0` restores the
                // three.
                if !std::env::var("TUILI_DECODE_ATTN").is_ok_and(|v| v == "0")
                    && self.kern.decode_attention(&dims)
                {
                    let mut h16 = self.scratch.x16.slice_mut(..n * d);
                    attn_f16 = self.kern.attn_decode(
                        &mut attn_out.slice_mut(..n * d),
                        wo_f16.then_some(&mut h16),
                        &self.act.q.slice(..n * d),
                        &pool.dense(layer).0.as_view(),
                        &pool.dense(layer).1.as_view(),
                        batch,
                        dims,
                        kv_len,
                        attn_scale,
                        &mut partial.as_view_mut(),
                    )?;
                } else if self.kern.flash_attention(&dims, kv_len) {
                    self.kern.attn_flash(
                        &mut attn_out.slice_mut(..n * d),
                        &self.act.q.slice(..n * d),
                        &pool.dense(layer).0.as_view(),
                        &pool.dense(layer).1.as_view(),
                        batch,
                        dims,
                        kv_len,
                        attn_scale,
                        &mut partial.as_view_mut(),
                    )?;
                } else {
                    self.kern.attn_scores(
                        &mut self.act.scores.slice_mut(..score_len),
                        &self.act.q.slice(..n * d),
                        &pool.dense(layer).0.as_view(),
                        batch,
                        dims,
                        kv_len,
                        attn_scale,
                    )?;
                    self.kern.attn_softmax(
                        &mut self.act.scores.slice_mut(..score_len),
                        n_heads,
                        n,
                        kv_len,
                    )?;
                    self.kern.attn_output(
                        &mut attn_out.slice_mut(..n * d),
                        &self.act.scores.slice(..score_len),
                        &pool.dense(layer).1.as_view(),
                        batch,
                        dims,
                        kv_len,
                        Some(&mut partial.as_view_mut()),
                    )?;
                }
            }
            Some(tq) => {
                let quant = pool.quant();
                let (k_bits, v_bits) = (quant.k_mse_bits(), quant.v_bits());
                let n_kv_vecs = n * n_kv_heads;
                let n_q_vecs = n * n_heads;

                // Into the rotated basis: keys, values and queries alike.
                self.kern.tq_matvec(
                    &mut tq.k_rot.slice_mut(..n * kv_dim),
                    &self.act.k.slice(..n * kv_dim),
                    &tq.tables.rotation.as_view(),
                    d_head,
                    n_kv_vecs,
                )?;
                self.kern.tq_matvec(
                    &mut tq.v_rot.slice_mut(..n * kv_dim),
                    &self.act.v.slice(..n * kv_dim),
                    &tq.tables.rotation.as_view(),
                    d_head,
                    n_kv_vecs,
                )?;
                self.kern.tq_matvec(
                    &mut tq.q_rot.slice_mut(..n * d),
                    &self.act.q.slice(..n * d),
                    &tq.tables.rotation.as_view(),
                    d_head,
                    n_q_vecs,
                )?;
                self.kern.tq_matvec(
                    &mut tq.q_qjl.slice_mut(..n * d),
                    &tq.q_rot.slice(..n * d),
                    &tq.tables.qjl.as_view(),
                    d_head,
                    n_q_vecs,
                )?;

                {
                    let (codes, signs, scale, gamma) = pool.tq_key_mut(layer);
                    self.kern.tq_store_k(
                        &mut codes.as_view_mut(),
                        &mut signs.as_view_mut(),
                        &mut scale.as_view_mut(),
                        &mut gamma.as_view_mut(),
                        &tq.k_rot.slice(..n * kv_dim),
                        &tq.tables.qjl.as_view(),
                        &self.act.slots.slice(..n),
                        &tq.tables.k_levels.as_view(),
                        k_bits,
                        n_kv_heads,
                        d_head,
                        dims.n_slots,
                        n,
                    )?;
                }
                {
                    let (codes, scale) = pool.tq_value_mut(layer);
                    self.kern.tq_store_v(
                        &mut codes.as_view_mut(),
                        &mut scale.as_view_mut(),
                        &tq.v_rot.slice(..n * kv_dim),
                        &self.act.slots.slice(..n),
                        &tq.tables.v_levels.as_view(),
                        v_bits,
                        n_kv_heads,
                        d_head,
                        dims.n_slots,
                        n,
                    )?;
                }

                let table = pool.slot_table().as_view();
                let batch = BatchLayout {
                    seq_of: &seq_of,
                    positions: &batch_positions,
                    slot_table: &table,
                    table_stride,
                };
                {
                    let (codes, signs, scale, gamma) = pool.tq_key(layer);
                    self.kern.tq_attn_scores(
                        &mut self.act.scores.slice_mut(..score_len),
                        &tq.q_rot.slice(..n * d),
                        &tq.q_qjl.slice(..n * d),
                        &codes.as_view(),
                        &signs.as_view(),
                        &scale.as_view(),
                        &gamma.as_view(),
                        batch,
                        &tq.tables.k_levels.as_view(),
                        k_bits,
                        dims,
                        kv_len,
                        attn_scale,
                        quant.qjl_scale(),
                    )?;
                }
                self.kern.attn_softmax(
                    &mut self.act.scores.slice_mut(..score_len),
                    n_heads,
                    n,
                    kv_len,
                )?;
                {
                    let (codes, scale) = pool.tq_value(layer);
                    self.kern.tq_attn_output(
                        &mut tq.acc_rot.slice_mut(..n * d),
                        &self.act.scores.slice(..score_len),
                        &codes.as_view(),
                        &scale.as_view(),
                        batch,
                        &tq.tables.v_levels.as_view(),
                        v_bits,
                        dims,
                        kv_len,
                    )?;
                }
                // Back out of the rotated basis, once, on the output.
                self.kern.tq_matvec(
                    &mut self.act.attn.slice_mut(..n * d),
                    &tq.acc_rot.slice(..n * d),
                    &tq.tables.rotation_t.as_view(),
                    d_head,
                    n_q_vecs,
                )?;
            }
        }

        // Straight into the residual stream: this projection's result is only
        // ever added to it, and the mat-vec can do that itself.
        if l.bo.is_none() && Self::residual_fusable(&l.wo, n, self.use_mmvq) {
            let bytes = Kernels::q8_1_bytes(d);
            self.kern.quantize_q8_1(
                &mut self.scratch.q8_1.slice_mut(..bytes),
                &self.act.attn.slice(..d),
                d,
            )?;
            self.kern.mmvq_add(
                &mut self.act.x.slice_mut(..d),
                &l.wo.view(stage)?,
                l.wo.ty,
                &self.scratch.q8_1.slice(..bytes),
                d,
                l.wo.n,
            )?;
            return Ok(());
        }

        Self::matmul_pre(
            &self.kern,
            &mut self.scratch,
            &mut self.act.proj.slice_mut(..n * d),
            &l.wo,
            stage,
            &self.act.attn.slice(..n * d),
            n,
            self.use_mmvq,
            self.use_mmq,
            None,
            attn_f16,
        )?;
        if let Some(b) = &l.bo {
            self.kern
                .add_bias(&mut self.act.proj.slice_mut(..n * d), &b.as_view(), d, n)?;
        }
        if !self.ffn_norm_takes_residual(layer, n) {
            self.kern.add_assign(
                &mut self.act.x.slice_mut(..n * d),
                &self.act.proj.slice(..n * d),
                n * d,
            )?;
        }
        Ok(())
    }

    /// Whether this layer's FFN norm will add the attention residual itself.
    ///
    /// The add and the norm that follows it are one kernel when the norm is the
    /// f16-writing one — see `Kernels::add_rms_norm_f16`. Both sides of that
    /// have to agree: `attention` skips its `add_assign` exactly when
    /// `feed_forward` is going to do it, and a disagreement is a residual
    /// silently dropped or applied twice. So it is one predicate, asked twice.
    /// Whether this layer's *attention* norm will add the previous layer's FFN
    /// output itself, the way `ffn_norm_takes_residual` absorbs attention's.
    ///
    /// That leaves `add_assign` with nothing to do between two layers: the
    /// residual stream is updated inside the norm that was going to read it
    /// anyway. One launch and one 512 KB round trip a layer, 1.3 us in the
    /// trace. The first layer has no pending residual and the last one has no
    /// successor to absorb it, so both ends still add explicitly.
    ///
    /// Both sides read this: `feed_forward` skips its add exactly when the next
    /// layer's `attention` will do it. `TUILI_FUSE_RESIDUAL=0` turns off this
    /// one and the FFN one together.
    fn attn_norm_takes_residual(&self, layer: usize, n: usize) -> bool {
        if layer == 0 || layer >= self.cfg.n_layers {
            return false;
        }
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *OFF.get_or_init(|| std::env::var("TUILI_FUSE_RESIDUAL").as_deref() == Ok("0")) {
            return false;
        }
        let l = &self.w.layers[layer];
        let prev = &self.w.layers[layer - 1];
        let d = self.cfg.d_model;
        // The consumer: this layer's q/k/v group has to be on the f16 path,
        // because the fused add-and-norm is the kernel that writes f16.
        let consumer = self.use_mmq
            && n > 1
            && n <= tuili_kernels::MMQ_MAX_TOKENS
            && Kernels::mmq_f16_variant_for(l.wq.ty).is_some()
            && [&l.wq, &l.wk, &l.wv].iter().all(|w| {
                matches!(
                    w.ty,
                    tuili_kernels::WeightType::Q4G128 | tuili_kernels::WeightType::Q4G128T
                ) && w.k == d
            });
        // The producer: the previous layer's `down` has to have left its output
        // in `proj` rather than adding itself into the stream, which is what the
        // single-token mat-vec path does.
        consumer && !Self::residual_fusable(&prev.w_down, n, self.use_mmvq)
    }

    fn ffn_norm_takes_residual(&self, layer: usize, n: usize) -> bool {
        let l = &self.w.layers[layer];
        let d = self.cfg.d_model;
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *OFF.get_or_init(|| std::env::var("TUILI_FUSE_RESIDUAL").as_deref() == Ok("0")) {
            return false;
        }
        self.use_mmq
            && n > 1
            && n <= tuili_kernels::MMQ_MAX_TOKENS
            && Kernels::mmq_f16_variant_for(l.w_gate.ty).is_some()
            && [&l.w_gate, &l.w_up].iter().all(|w| {
                matches!(
                    w.ty,
                    tuili_kernels::WeightType::Q4G128 | tuili_kernels::WeightType::Q4G128T
                ) && w.k == d
            })
    }

    fn feed_forward(&mut self, layer: usize, n: usize, slot: Option<usize>) -> Result<()> {
        let stage = slot.map(|s| &self.offload.as_ref().unwrap().stage[s]);
        let cfg = &self.cfg;
        let (d, d_ff) = (cfg.d_model, cfg.d_ff);
        let l = &self.w.layers[layer];

        // `gate` and `up` share the normalized residual, as `q`/`k`/`v` do.
        let want_q = self.use_mmvq
            && [&l.w_gate, &l.w_up]
                .iter()
                .all(|w| Kernels::has_mmvq(w.ty) && w.k == d);
        let want_h = self.use_mmq
            && n > 1
            && n <= tuili_kernels::MMQ_MAX_TOKENS
            && Kernels::mmq_f16_variant_for(l.w_gate.ty).is_some()
            && [&l.w_gate, &l.w_up].iter().all(|w| {
                matches!(
                    w.ty,
                    tuili_kernels::WeightType::Q4G128
                        | tuili_kernels::WeightType::Q4G128T
                ) && w.k == d
            });
        // When the norm is the f16-writing one it also adds the attention
        // residual, which `attention` left in `proj` for it. See
        // `ffn_norm_takes_residual`.
        let (shared, shared_f16) = if want_h {
            self.kern.add_rms_norm_f16(
                &mut self.act.xb.slice_mut(..n * d),
                &mut self.scratch.x16.slice_mut(..n * d),
                &mut self.act.x.slice_mut(..n * d),
                &self.act.proj.slice(..n * d),
                &self.w.layers[layer].ffn_norm.as_view(),
                n,
                d,
                self.cfg.rms_eps,
            )?;
            (None, true)
        } else {
            Self::norm_for_group(
                &self.kern,
                &mut self.scratch,
                &mut self.act.xb,
                &self.act.x.slice(..n * d),
                &l.ffn_norm.as_view(),
                n,
                d,
                cfg.rms_eps,
                want_q,
                want_h,
            )?
        };

        // One matmul for both, when the loader stacked them. Seven projections a
        // layer cost 127.4 us where four cost 104.0, because the narrow ones
        // cannot fill the device — `attn_k` reaches 261 GB/s against
        // `gate_up`'s 1368. This is the FFN half of what vLLM gets from
        // `MergedColumnParallelLinear`.
        let stacked = l.w_gate_up.as_ref().filter(|_| n > 1 && want_h);
        // Whether `down` will read its activation as f16, which is the only
        // case where writing that copy early is worth anything. Mirrors what
        // `matmul_pre` decides for itself; claiming it when the buffer was not
        // written would hand the GEMM a stale one.
        let ffn_f16 = stacked.is_some()
            && self.use_mmq
            && n > 1
            && n <= tuili_kernels::MMQ_MAX_TOKENS
            && matches!(
                l.w_down.ty,
                tuili_kernels::WeightType::Q4G128 | tuili_kernels::WeightType::Q4G128T
            )
            && Kernels::mmq_f16_variant_for_shape(l.w_down.ty, l.w_down.n).is_some()
            && Self::mmq_shape_ok(&l.w_down);
        if let Some(gu) = stacked {
            Self::matmul_pre(
                &self.kern,
                &mut self.scratch,
                &mut self.act.gate.slice_mut(..n * 2 * d_ff),
                gu,
                stage,
                &self.act.xb.slice(..n * d),
                n,
                self.use_mmvq,
                self.use_mmq,
                shared,
                shared_f16,
            )?;
            // `down` is the only reader of this product, and above one token it
            // takes f16 — so the f16 copy can be written here, out of the
            // register the value is already in, rather than by a `to_f16`
            // launch that reads the f32 back. Same numbers, one launch fewer,
            // and one 512 KB round trip fewer. The trace put those conversions
            // at 2.4 us a layer across the two that survived the fused norm.
            if ffn_f16 {
                self.kern.silu_mul_split_f16(
                    &mut self.act.ffn.slice_mut(..n * d_ff),
                    &mut self.scratch.x16.slice_mut(..n * d_ff),
                    &self.act.gate.slice(..n * 2 * d_ff),
                    d_ff,
                    n * d_ff,
                )?;
            } else {
                self.kern.silu_mul_split(
                    &mut self.act.ffn.slice_mut(..n * d_ff),
                    &self.act.gate.slice(..n * 2 * d_ff),
                    d_ff,
                    n * d_ff,
                )?;
            }
        } else if Self::fusable(&[&l.w_gate, &l.w_up], shared, n, self.use_mmvq) {
            let bytes = Kernels::q8_1_bytes(d);
            let (gate, up) = (&mut self.act.gate, &mut self.act.up);
            self.kern.mmvq_fused2(
                &mut gate.slice_mut(..d_ff),
                &mut up.slice_mut(..d_ff),
                &l.w_gate.view(stage)?,
                &l.w_up.view(stage)?,
                l.w_gate.ty,
                &self.scratch.q8_1.slice(..bytes),
                d,
                l.w_gate.n,
                l.w_up.n,
            )?;
        } else {
            Self::matmul_pre(
                &self.kern,
                &mut self.scratch,
                &mut self.act.gate.slice_mut(..n * d_ff),
                &l.w_gate,
                stage,
                &self.act.xb.slice(..n * d),
                n,
                self.use_mmvq,
                self.use_mmq,
                shared,
                shared_f16,
            )?;
            Self::matmul_pre(
                &self.kern,
                &mut self.scratch,
                &mut self.act.up.slice_mut(..n * d_ff),
                &l.w_up,
                stage,
                &self.act.xb.slice(..n * d),
                n,
                self.use_mmvq,
                self.use_mmq,
                shared,
                shared_f16,
            )?;
        }
        // The stacked branch already applied it, over the two halves of one
        // row rather than over two tensors.
        if stacked.is_none() {
            self.kern.silu_mul(
                &mut self.act.ffn.slice_mut(..n * d_ff),
                &self.act.gate.slice(..n * d_ff),
                &self.act.up.slice(..n * d_ff),
                n * d_ff,
            )?;
        }
        // As with the output projection, straight into the residual stream.
        if Self::residual_fusable(&l.w_down, n, self.use_mmvq) {
            let bytes = Kernels::q8_1_bytes(d_ff);
            self.kern.quantize_q8_1(
                &mut self.scratch.q8_1.slice_mut(..bytes),
                &self.act.ffn.slice(..d_ff),
                d_ff,
            )?;
            self.kern.mmvq_add(
                &mut self.act.x.slice_mut(..d),
                &l.w_down.view(stage)?,
                l.w_down.ty,
                &self.scratch.q8_1.slice(..bytes),
                d_ff,
                l.w_down.n,
            )?;
            return Ok(());
        }

        Self::matmul_pre(
            &self.kern,
            &mut self.scratch,
            &mut self.act.proj.slice_mut(..n * d),
            &l.w_down,
            stage,
            &self.act.ffn.slice(..n * d_ff),
            n,
            self.use_mmvq,
            self.use_mmq,
            None,
            ffn_f16,
        )?;
        if !self.attn_norm_takes_residual(layer + 1, n) {
            self.kern.add_assign(
                &mut self.act.x.slice_mut(..n * d),
                &self.act.proj.slice(..n * d),
                n * d,
            )?;
        }
        Ok(())
    }

    /// Normalize into `xb` and, when the group that follows will want it,
    /// produce the Q8_1 form in the same launch.
    #[allow(clippy::too_many_arguments)]
    fn norm_for_group(
        kern: &Kernels,
        scratch: &mut Scratch,
        act_xb: &mut CudaSlice<f32>,
        x: &CudaView<'_, f32>,
        weight: &CudaView<'_, f32>,
        n_tokens: usize,
        d: usize,
        eps: f32,
        want_q8_1: bool,
        want_f16: bool,
    ) -> Result<(Option<usize>, bool)> {
        // The f16-operand GEMM is the Q4_G128 default and takes activations
        // unquantized, so for those groups the Q8_1 half of the fused norm is
        // a buffer nobody reads and every projection then pays its own
        // `to_f16` over the same row. The profile put those at 1.8% and 3.6%
        // of a batch-32 step. One pass, both outputs, shared by the group.
        if want_f16 {
            kern.rms_norm_f16(
                &mut act_xb.slice_mut(..n_tokens * d),
                &mut scratch.x16.slice_mut(..n_tokens * d),
                x,
                weight,
                n_tokens,
                d,
                eps,
            )?;
            return Ok((None, true));
        }
        // The fusion trades a launch for parallelism: the standalone quantizer
        // spreads `d` elements over many blocks, while the fused one does that
        // work inside the single block that computed the norm. `TUILI_NO_FUSED_NORM`
        // exists to measure which way that trade actually falls.
        static SEPARATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let separate =
            *SEPARATE.get_or_init(|| std::env::var_os("TUILI_NO_FUSED_NORM").is_some());
        if separate && want_q8_1 && d.is_multiple_of(32) {
            kern.rms_norm(
                &mut act_xb.slice_mut(..n_tokens * d),
                x,
                weight,
                n_tokens,
                d,
                eps,
            )?;
            let bytes = n_tokens * Kernels::q8_1_bytes(d);
            kern.quantize_q8_1(
                &mut scratch.q8_1.slice_mut(..bytes),
                &act_xb.slice(..n_tokens * d),
                n_tokens * d,
            )?;
            return Ok((Some(bytes), false));
        }
        if !want_q8_1 || !d.is_multiple_of(32) {
            kern.rms_norm(
                &mut act_xb.slice_mut(..n_tokens * d),
                x,
                weight,
                n_tokens,
                d,
                eps,
            )?;
            return Ok((None, false));
        }
        let bytes = n_tokens * Kernels::q8_1_bytes(d);
        kern.rms_norm_q8_1(
            &mut act_xb.slice_mut(..n_tokens * d),
            &mut scratch.q8_1.slice_mut(..bytes),
            x,
            weight,
            n_tokens,
            d,
            eps,
        )?;
        Ok((Some(bytes), false))
    }

    /// Whether a weight's row length suits the tensor-core GEMM.
    ///
    /// Q4_K super-blocks are 256 wide and the tile loop assumes a whole one per
    /// step; the block-32 types only need the MMA's K extent. Qwen2.5-0.5B's
    /// 896-wide rows fall on the wrong side of the Q4_K check, which is why its
    /// K-quant build is mostly Q5_0 to begin with.
    fn mmq_shape_ok(w: &Matrix) -> bool {
        match w.ty {
            // A Q4_G128 tile is two 128-weight blocks, so it wants the same
            // 256-wide step the K-quants do.
            tuili_kernels::WeightType::Q4K
            | tuili_kernels::WeightType::Q6K
            | tuili_kernels::WeightType::Q4G128
            | tuili_kernels::WeightType::Q4G128T => w.k.is_multiple_of(256),
            _ => w.k.is_multiple_of(32),
        }
    }

    /// `out[t, :] = w · x[t, :]`, picking the integer mat-vec, the float one,
    /// or cuBLAS by batch size.
    #[allow(clippy::too_many_arguments)]
    fn matmul(
        kern: &Kernels,
        scratch: &mut Scratch,
        out: &mut CudaViewMut<'_, f32>,
        w: &Matrix,
        stage: Option<&CudaSlice<u8>>,
        x: &CudaView<'_, f32>,
        n_tokens: usize,
        use_mmvq: bool,
        use_mmq: bool,
    ) -> Result<()> {
        Self::matmul_pre(
            kern, scratch, out, w, stage, x, n_tokens, use_mmvq, use_mmq, None, false,
        )
    }

    /// Whether a projection whose result only ever lands in the residual
    /// stream can add itself there, saving the separate `add_assign`.
    ///
    /// Only the single-token integer path: the GEMM writes a tile at a time
    /// and has no accumulate mode, and above one token the saved kernel is a
    /// smaller share of the step anyway.
    fn residual_fusable(w: &Matrix, n_tokens: usize, use_mmvq: bool) -> bool {
        n_tokens == 1
            && use_mmvq
            && Kernels::has_mmvq(w.ty)
            && w.k.is_multiple_of(32)
    }

    /// Whether a group of projections that share one activation can go
    /// through a single fused mat-vec.
    ///
    /// They have to agree on the inner dimension and the weight type — a
    /// Q4_K_M file gives its first layer a Q6_K V projection between two Q4_K
    /// siblings — and the activation has to be quantized already, which is
    /// what `norm_for_group` does for exactly these groups.
    fn fusable(ws: &[&Matrix], pre_quantized: Option<usize>, n_tokens: usize, use_mmvq: bool) -> bool {
        pre_quantized.is_some()
            && n_tokens == 1
            && use_mmvq
            && ws
                .iter()
                .all(|w| Kernels::has_mmvq(w.ty) && w.k.is_multiple_of(32))
            && ws.windows(2).all(|p| p[0].ty == p[1].ty && p[0].k == p[1].k)
    }

    /// [`Self::matmul`] with the activation optionally already in `scratch.q8_1`.
    #[allow(clippy::too_many_arguments)]
    fn matmul_pre(
        kern: &Kernels,
        scratch: &mut Scratch,
        out: &mut CudaViewMut<'_, f32>,
        w: &Matrix,
        stage: Option<&CudaSlice<u8>>,
        x: &CudaView<'_, f32>,
        n_tokens: usize,
        use_mmvq: bool,
        use_mmq: bool,
        pre_quantized: Option<usize>,
        pre_f16: bool,
    ) -> Result<()> {
        let weights = w.view(stage)?;
        let int_x = use_mmvq && Kernels::has_mmvq(w.ty) && w.k.is_multiple_of(32);
        // Whether *this matrix* gets the tensor-core GEMM, not just its type:
        // Q4_K rows that are not a multiple of 256 have the type but not the
        // shape, and they should fall to the mat-vec rather than the float path.
        let mmq_ok = use_mmq
            && Kernels::has_mmq(w.ty)
            && kern.device().arch() >= 80
            && Self::mmq_shape_ok(w);

        // One token is the decode step, and it is bandwidth-bound: the integer
        // path retires four weights per instruction instead of decoding each to
        // a float. Above one token the weights are reused across rows and the
        // answer is a GEMM, not a wider mat-vec.
        if int_x && n_tokens == 1 {
            let bytes = Kernels::q8_1_bytes(w.k);
            if pre_quantized.is_none() {
                kern.quantize_q8_1(&mut scratch.q8_1.slice_mut(..bytes), x, w.k)?;
            }
            return kern.mmvq(out, &weights, w.ty, &scratch.q8_1.slice(..bytes), w.k, w.n);
        }

        // Two or three tokens are the awkward middle: the tensor-core GEMM pays
        // for a full 16-token tile whatever it is given, while the mat-vec can
        // stream the weights once and spend them on a handful of tokens without
        // staging anything through shared memory. Measured on a 31.5 MiB Q4_K
        // projection: at two tokens 120 us against the GEMM's 182, at four they
        // are level, and by eight the GEMM is ahead 222 to 368.
        if int_x && (2..=3).contains(&n_tokens) {
            let bytes = Kernels::q8_1_bytes(w.k);
            if pre_quantized.is_none() {
                kern.quantize_q8_1(
                    &mut scratch.q8_1.slice_mut(..n_tokens * bytes),
                    &x.slice(..n_tokens * w.k),
                    n_tokens * w.k,
                )?;
            }
            return kern.mmvq_batch(
                out,
                &weights,
                w.ty,
                &scratch.q8_1.slice(..n_tokens * bytes),
                w.k,
                w.n,
                n_tokens,
            );
        }

        // Above one token the integer tensor cores run the GEMM straight off the
        // quantized weights: one read of the quantized bytes per token tile,
        // versus the dequant path's read-write-read of a full f16 copy. That
        // gap was 79% of a batch-32 decode step. Past `MMQ_MAX_TOKENS` the
        // per-tile re-reads add up and cuBLAS wins instead.
        if mmq_ok && n_tokens > 1 && n_tokens <= tuili_kernels::MMQ_MAX_TOKENS {
            // The f16-operand kernels take activations unquantized, so they
            // need a different buffer and a different launcher. Off unless
            // `TUILI_MMQ_VARIANT` names one; the default path below is
            // untouched.
            //
            // This re-converts per matmul where the Q8_1 path can hand the
            // same quantized activation to several projections, so the number
            // it measures is if anything pessimistic.
            if matches!(
                w.ty,
                tuili_kernels::WeightType::Q4G128 | tuili_kernels::WeightType::Q4G128T
            ) {
                if let Some(v) = Kernels::mmq_f16_variant_for_shape(w.ty, w.n) {
                    let n = n_tokens * w.k;
                    // The projections sharing an input share the conversion:
                    // the fused norm wrote it once for the whole group.
                    if !pre_f16 {
                        kern.to_f16(
                            &mut scratch.x16.slice_mut(..n),
                            &x.slice(..n),
                            n,
                        )?;
                    }
                    return kern.mmq_f16(
                        v,
                        out,
                        &weights,
                        &scratch.x16.slice(..n),
                        w.k,
                        w.n,
                        n_tokens,
                    );
                }
            }
            let bytes = Kernels::q8_1_bytes(w.k);
            let total = n_tokens * bytes;
            if pre_quantized.is_none() {
                // Blocks never straddle a row because `k` is a multiple of 32,
                // so one launch quantizes every token.
                kern.quantize_q8_1(
                    &mut scratch.q8_1.slice_mut(..total),
                    &x.slice(..n_tokens * w.k),
                    n_tokens * w.k,
                )?;
            }
            return kern.mmq(
                out,
                &weights,
                w.ty,
                &scratch.q8_1.slice(..total),
                w.k,
                w.n,
                n_tokens,
            );
        }

        // A matrix with an integer mat-vec but no tensor-core GEMM does better
        // repeating the mat-vec per token than taking the float path even once:
        // the float `gemv` decodes one weight per thread and runs an order of
        // magnitude below the memory bound. The profile had one such matrix at
        // 670us against the integer mat-vec's 51us, so a handful of repeated
        // passes still wins.
        if int_x && !mmq_ok && n_tokens > 1 && n_tokens <= MMVQ_REPEAT_MAX {
            let bytes = Kernels::q8_1_bytes(w.k);
            if pre_quantized.is_none() {
                kern.quantize_q8_1(
                    &mut scratch.q8_1.slice_mut(..n_tokens * bytes),
                    &x.slice(..n_tokens * w.k),
                    n_tokens * w.k,
                )?;
            }
            for t in 0..n_tokens {
                kern.mmvq(
                    &mut out.slice_mut(t * w.n..(t + 1) * w.n),
                    &weights,
                    w.ty,
                    &scratch.q8_1.slice(t * bytes..(t + 1) * bytes),
                    w.k,
                    w.n,
                )?;
            }
            return Ok(());
        }

        if n_tokens <= GEMM_THRESHOLD {
            return kern.gemv(out, &weights, w.ty, x, w.k, w.n, n_tokens);
        }

        let n_x = n_tokens * w.k;
        kern.to_f16(&mut scratch.x16.slice_mut(..n_x), x, n_x)?;

        if w.ty == tuili_kernels::WeightType::F16 {
            // Already f16 on the device: reinterpret rather than copy.
            //
            // Safety: the range holds exactly `k * n` f16 values copied from
            // the GGUF file, and f16 has no invalid bit patterns.
            let view = unsafe { weights.transmute::<f16>(w.elements()) }
                .context("f16 weight buffer is misaligned")?;
            return kern.gemm_f16(out, &scratch.x16.slice(..n_x), &view, n_tokens, w.k, w.n);
        }

        kern.dequant_to_f16(
            &mut scratch.w16.slice_mut(..w.elements()),
            &weights,
            w.ty,
            w.elements(),
        )?;
        kern.gemm_f16(
            out,
            &scratch.x16.slice(..n_x),
            &scratch.w16.slice(..w.elements()),
            n_tokens,
            w.k,
            w.n,
        )
    }
}

impl Activations {

    fn new(
        dev: &Device,
        cfg: &Config,
        max_seq: usize,
        max_logit_rows: usize,
    ) -> Result<Self> {
        let stream = dev.stream();
        let chunk = MAX_BATCH_TOKENS;
        let d = cfg.d_model;
        let kv_dim = cfg.n_kv_heads * cfg.d_head;

        let alloc_f32 = |n: usize, what: &str| -> Result<CudaSlice<f32>> {
            stream
                .alloc_zeros::<f32>(n)
                .with_context(|| format!("allocating {what} ({} MiB)", (n * 4) >> 20))
        };

        Ok(Self {
            x: alloc_f32(chunk * d, "residual")?,
            xb: alloc_f32(chunk * d, "normalized")?,
            q: alloc_f32(chunk * d, "queries")?,
            k: alloc_f32(chunk * kv_dim, "keys")?,
            v: alloc_f32(chunk * kv_dim, "values")?,
            attn: alloc_f32(chunk * d, "attention output")?,
            proj: alloc_f32(chunk * d.max(1), "projection")?,
            // Twice `d_ff`: under `TUILI_FUSE_FFN` one matmul writes `gate` and
            // `up` into a single row of this, and `silu_mul_split` reads the
            // two halves. The unfused path uses the first half only.
            gate: alloc_f32(chunk * cfg.d_ff * 2, "ffn gate")?,
            up: alloc_f32(chunk * cfg.d_ff, "ffn up")?,
            ffn: alloc_f32(chunk * cfg.d_ff, "ffn hidden")?,
            scores: alloc_f32(cfg.n_heads * chunk * max_seq, "attention scores")?,
            logits: alloc_f32(max_logit_rows * cfg.vocab_size, "logits")?,
            token_ids: stream.alloc_zeros::<i32>(chunk)?,
            positions: stream.alloc_zeros::<i32>(chunk)?,
            seq_of: stream.alloc_zeros::<i32>(chunk)?,
            slots: stream.alloc_zeros::<i32>(chunk)?,
            logit_rows: stream.alloc_zeros::<i32>(max_logit_rows)?,
            // Partial sums for the split-K attention output. Sized for the
            // widest split the kernel will pick; it only allocates once.
            attn_partial: alloc_f32(
                Kernels::attn_partial_floats(cfg.n_heads, cfg.d_head, chunk),
                "attention partials",
            )?,
            head_in: alloc_f32(max_logit_rows * d, "logit rows")?,
        })
    }
}
