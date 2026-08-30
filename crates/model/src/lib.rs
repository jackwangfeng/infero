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
pub mod gdn_state;
pub mod mtp;
pub mod qwen35;
pub mod spec;
pub mod qwen35_mtp;
pub mod qwen35_vision;
/// Image preprocessing: resize, normalize, patchify.
///
/// `tests/qwen35_vision_image.rs` reaches this with `#[path]`, which compiled
/// the file into the test binary and not into the library — so it was checked
/// against Pillow and unreachable from the forward pass at the same time.
pub mod qwen35_vision_image;
mod sampling;
pub mod weights;

use anyhow::{Context, Result};
use infero_gpu::{Buf, View, ViewMut};
use infero_gpu::{Event as CudaEvent, OwnedStream as CudaStream};
use half::f16;
use std::sync::Arc;
use infero_gpu::Device;
use infero_gguf::Gguf;
use infero_kernels::{AttnDims, BatchLayout, Kernels, KvQuant, TqTables};

pub use cache::{KvPool, SeqId};
pub use config::Config;
pub use sampling::{Sampler, SamplingParams};
pub use infero_kernels::KvQuant as KvCacheQuant;
pub use weights::Weights;

use weights::Matrix;

/// Baseline ceiling on the tokens one forward pass may carry, summed over the
/// batch -- a floor under [`batch_ceiling`], not the final number.
///
/// It bounds the attention score buffer -- `n_heads * chunk * max_seq`, the one
/// activation that grows with both batch size and context length, 201 MiB at 256
/// tokens and a 8192 context on this model -- so raising it is a memory trade
/// against a real number, not a free knob.
///
/// Higher on Metal, and for a reason that does not apply to CUDA. Prefill
/// dequantises each weight to f16 once *per chunk*, so a 2561-token prompt in
/// 256-token chunks does it ten times; a bigger chunk divides that. cuBLAS is
/// reached at four tokens there and the same argument would apply, but it is
/// untested on that hardware and 256 is what is tuned.
#[cfg(feature = "cuda")]
pub const MAX_BATCH_TOKENS: usize = 256;
#[cfg(not(feature = "cuda"))]
pub const MAX_BATCH_TOKENS: usize = 1024;

/// What one attention score buffer is allowed to cost.
///
/// The buffer is `n_heads * chunk * max_seq` floats, so the chunk and the
/// context multiply. This is the budget that decides which one gives.
const SCORE_BUDGET: usize = 1 << 30;

/// The ceiling `batch_tokens_for` clamps to, raised for a server that has to
/// admit many sequences at once.
///
/// `MAX_BATCH_TOKENS` alone was measured against one long prompt, and a flat
/// 256 starves concurrent short ones instead: sixteen 60-token chat prompts
/// admitted in the same instant, on a small `--ctx` where the score budget
/// would happily allow thousands, still only fit four to a pass, so the other
/// twelve queue through three extra prefill-sized steps before their first
/// token -- 1.2 s of pure admission latency that has nothing to do with
/// decoding. `max_logit_rows` is what the caller's `--max-seqs` (and
/// speculation's row count) already resolved to, so multiplying it by the
/// same 64-token GEMM floor `batch_tokens_for` clamps down to says: give every
/// concurrent sequence room for at least one GEMM-sized chunk in the same
/// pass, rather than assume the one-long-prompt number covers it.
///
/// This only ever raises the ceiling, never lowers it, and the score budget in
/// `batch_tokens_for` still clamps the result on top -- a large `max_seq`
/// shrinks that budget's own number well below either ceiling regardless, so
/// a big `--max-seqs` cannot reopen the OOM `MAX_BATCH_TOKENS` was tuned to
/// avoid.
fn batch_ceiling(max_logit_rows: usize) -> usize {
    MAX_BATCH_TOKENS.max(max_logit_rows.max(1) * 64)
}

/// Tokens one forward pass carries, given the context it has to hold scores for.
///
/// Not a constant, because the two things it trades against move independently.
/// A bigger chunk amortises prefill's per-chunk dequantisation -- measured on the
/// 27B, a 2561-token prompt: 31.2 s at 256, 25.0 at 512, 22.7 at 1024 -- and a
/// bigger context multiplies the score buffer by the same chunk. At 24 heads and
/// a 8192 context, 1024 tokens is 0.75 GiB; at the 262144 context this model
/// advertises it would be 24 GiB, which does not start at all. Raising the chunk
/// without this bound would have quietly cut the largest usable context by four.
///
/// The floor is 64 rather than 1: below `GEMM_THRESHOLD` a chunk takes the
/// mat-vec, and a chunk small enough to do that has given up the thing chunks
/// are being enlarged for.
///
/// `INFERO_BATCH_TOKENS` overrides it outright, budget included, because the
/// person measuring is entitled to a worse setting than this picks.
///
/// The budget only applies when `needs_score_buffer` is true. A model whose
/// attention shape always takes a fused kernel (see [`needs_score_buffer`])
/// never materializes the `n_heads * chunk * max_seq` buffer this trades
/// against, so throttling the chunk to protect it would only be shrinking
/// prefill for no memory saved.
///
/// `fp8_ceiling` is separate from that budget and applies regardless of
/// `needs_score_buffer`: for the legacy interleaved-layout FP8 tensor-core
/// GEMM (`mma_e4m3_block`) it's
/// [`infero_kernels::fp8::MMA_MAX_TOKENS_FP8`][mma], the width past which
/// that kernel declines the shape and every matmul in the chunk falls to the
/// expand-then-dequantize path instead -- a correctness-preserving but much
/// slower kernel that raising this function's other ceilings (a large
/// `--max-seqs`, or a fused attention kernel freeing `needs_score_buffer`)
/// could otherwise walk the chunk straight into. `None` for a model with no
/// FP8 weights, which that path never applies to. The CUTLASS/unified-layout
/// path's caller passes its own, separately measured ceiling instead:
/// `mma_e4m3_cutlass` doesn't decline a wide `M` the way `mma_e4m3_block`
/// does (it just tiles more of it), so `MMA_MAX_TOKENS_FP8` would only
/// throttle the chunk there for a constraint that doesn't exist on the
/// kernel actually running -- but "uncapped" isn't strictly better either:
/// on a 30552-token prefill, capped at 256 (the legacy ceiling, before this
/// distinction existed) was 11.1s, uncapped (~2048, this server's
/// `batch_ceiling`) was 9.0s, and 1024 was the measured optimum at 8.7s.
///
/// [mma]: infero_kernels::fp8::MMA_MAX_TOKENS_FP8
pub fn batch_tokens_for(
    n_heads: usize,
    max_seq: usize,
    max_logit_rows: usize,
    needs_score_buffer: bool,
    fp8_ceiling: Option<usize>,
) -> usize {
    let ceiling = batch_ceiling(max_logit_rows);
    if let Some(n) = std::env::var("INFERO_BATCH_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
    {
        return n.min(ceiling);
    }
    let chunk = if !needs_score_buffer {
        // `ceiling` is a floor here (`batch_ceiling` only ever raises it, see
        // its own doc comment), meant to keep many concurrent short prompts
        // from starving against a small tuned constant. When the caller also
        // hands over a model-specific `fp8_ceiling` -- itself measured
        // against real VRAM and kernel-occupancy limits, not a generic
        // one-size floor -- prefer it outright rather than letting the floor
        // silently cap it back down for a model that was never the reason
        // the floor exists (this model needs no score buffer at all).
        fp8_ceiling.unwrap_or(ceiling)
    } else {
        let per_token = n_heads.max(1) * max_seq.max(1) * std::mem::size_of::<f32>();
        (SCORE_BUDGET / per_token.max(1)).clamp(64, ceiling)
    };
    match fp8_ceiling {
        Some(cap) => chunk.min(cap),
        None => chunk,
    }
}

/// Whether this model's attention will ever run the score-materializing
/// three-kernel path (`attn_scores`/`tq_attn_scores`, `attn_softmax`,
/// `attn_output`/`tq_attn_output`) rather than a kernel that tiles the KV
/// range and never writes a `[heads, tokens, kv_len]` matrix to HBM.
///
/// Both fused kernels' gates ([`Kernels::decode_attention`] on CUDA,
/// [`Kernels::tq_decode_attention`]) depend only on the model's GQA shape,
/// not on how many tokens a step carries or how far into the sequence it
/// is -- so this can be decided once at load time, the same way `max_seq`
/// and `kv_quant` are fixed for the process's lifetime.
///
/// [`Kernels::decode_attention`]'s Metal fallback *does* depend on `kv_len`
/// (capped at 8192), which a probe taken once at load time cannot rule out
/// for the rest of the run -- so this stays conservative (`true`) off CUDA
/// rather than risk sizing the score buffer for a kernel that stops being
/// chosen once a sequence grows past that cap.
fn needs_score_buffer(kern: &Kernels, cfg: &Config, max_seq: usize, kv_quant: KvQuant) -> bool {
    if !cfg!(feature = "cuda") {
        return true;
    }
    let dims = AttnDims {
        n_heads: cfg.n_heads,
        n_kv_heads: cfg.n_kv_heads,
        d_head: cfg.d_head,
        n_slots: 0,
        n_tokens: 0,
    };
    let fused = if kv_quant.is_quantized() {
        kern.tq_decode_attention(&dims)
    } else {
        kern.decode_attention(&dims, max_seq)
    };
    !fused
}

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
    /// Vision features to write over this chunk's placeholder tokens.
    ///
    /// Carried on the item rather than held per sequence because a chunked
    /// prefill splits a prompt at token boundaries that know nothing about where
    /// an image or video sits: the caller that cut the chunk is the only one
    /// that can say which feature rows belong to it. This can be the *whole*
    /// clip's features, or -- once a placeholder run outgrows one step's
    /// `batch_tokens` budget -- the same clip across several chunks in a row,
    /// each seeing however many of its placeholder tokens landed in that
    /// chunk; `vision_row_offset` below is what tells `forward_batch_device`
    /// which of the clip's rows this chunk's tokens correspond to. A chunk
    /// with no placeholder tokens at all (before the run starts, or after it
    /// ends) is fine too -- the splice is a no-op for it.
    pub vision: Option<&'a VisionFeatures>,
    /// How many of `vision`'s feature rows earlier chunks of this same
    /// sequence already consumed -- `0` for a clip whose whole placeholder run
    /// lands in one chunk (the common case), and for chunks that come before
    /// or after the run (there `forward_batch_device` finds no placeholder ids
    /// in `tokens` and never reads this). See `Running::vision_at` in the
    /// scheduler, which is what a caller slicing at `from..from+len` computes
    /// this from (`from.saturating_sub(vision_at)`).
    pub vision_row_offset: usize,
    /// This chunk's absolute M-RoPE `[T, H, W]` positions, token-major
    /// (`3 * tokens.len()` entries), for a model with `cfg.mrope_section` set.
    ///
    /// `None` for every model without one, and for a decode step even on a
    /// model that has one — a single generated token is never mid-image, so
    /// its three axes are always `pool.len(seq) + mrope_delta` and there is
    /// nothing here to look up. `Some` only for a prefill chunk that overlaps
    /// a spliced sequence's absolute position array, sliced by the caller at
    /// `from..from+len` — see `Running::mrope` in the scheduler, which owns
    /// the array this borrows from.
    pub mrope: Option<&'a [i32]>,
    /// The constant to add to a plain running length to get this sequence's
    /// M-RoPE position, on every axis, when `mrope` above is `None`.
    ///
    /// Zero for a model or a sequence without M-RoPE, which reproduces
    /// `pool.len(seq) + k` on all three axes — identical to the one-axis
    /// scalar position. Negative for a sequence that has passed through an
    /// image: M-RoPE's image advance rule moves the running position by the
    /// larger spatial extent rather than by token count, so the position
    /// after an image is *behind* where token-counting would put it. See
    /// `qwen35_vision::llm_position_ids`'s doc comment.
    pub mrope_delta: i32,
}

impl<'a> BatchItem<'a> {
    pub fn new(seq: SeqId, tokens: &'a [u32]) -> Self {
        Self {
            seq,
            tokens,
            wants_logits: true,
            vision: None,
            vision_row_offset: 0,
            mrope: None,
            mrope_delta: 0,
        }
    }

    pub fn without_logits(seq: SeqId, tokens: &'a [u32]) -> Self {
        Self {
            seq,
            tokens,
            wants_logits: false,
            vision: None,
            vision_row_offset: 0,
            mrope: None,
            mrope_delta: 0,
        }
    }
}

/// Above this many tokens a projection goes through the library GEMM instead of
/// the mat-vec. The mat-vec re-reads the weights once per token, so on CUDA the
/// crossover is early.
///
/// A backend with no GEMM has a different crossover: infinity. The float
/// mat-vec is correct at any width -- it batches `GEMV_TOKENS` tokens per
/// threadgroup and its grid covers the rest -- so prefill takes it and pays the
/// re-reads rather than failing. That is the whole cost of not having
/// `MPSMatrixMultiplication` wired up yet, and it is a prefill cost only:
/// decode is one token and never reaches here.
#[cfg(feature = "cuda")]
const GEMM_THRESHOLD_DEFAULT: usize = 4;
/// Was twelve times CUDA's 4, because reaching the GEMM means dequantising the
/// weight to f16 first -- 3.56x the bytes of Q4_K -- plus splitting the open
/// command encoder, and a handful of tokens could not pay that back. Measured,
/// prompt tokens through the 27B, one chunk each:
///
/// ```text
///   prompt   GEMM on   GEMM off
///       11    147.2       42.7   ms a token
///       31     53.5       48.9
///       71     27.0       52.0
///      151     15.5       52.7
///      311     14.9       52.2
/// ```
///
/// The crossing was between 31 and 71, and 48 sat in it.
///
/// That table is stale: it dated from before the whole-matrix dequant kernel
/// was one thread a 32-element group instead of one a element (see
/// `dequant_q4_K_f16_vec`), and the fixed cost this threshold exists to weigh
/// against the mat-vec's re-reads was most of what that fix removed.
/// Re-measured after, same method:
///
/// ```text
///   prompt   GEMM on   GEMM off
///       16     38.3       28.1   ms a token
///       20     33.7       41.9
///       30     23.0       31.2
///       70     15.4       32.9
///       90     12.7       33.4
/// ```
///
/// The crossing moved to between 16 and 20. Erring high is still the cheaper
/// mistake -- a short prompt is short, so a few ms a token extra on twenty of
/// them is a rounding error next to what a threshold four times too high did
/// to every longer one -- so 16 rather than 20.
#[cfg(not(feature = "cuda"))]
const GEMM_THRESHOLD_DEFAULT: usize = 16;

/// Tokens above which a matmul takes the GEMM rather than repeating the
/// mat-vec, overridable so the trade can be re-measured rather than argued.
fn gemm_threshold() -> usize {
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        std::env::var("INFERO_GEMM_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| *v > 0)
            .unwrap_or(GEMM_THRESHOLD_DEFAULT)
    })
}
/// Up to this many tokens, a weight type with no tensor-core GEMM repeats the
/// integer mat-vec once per token instead of taking the float path. Measured
/// against Llama-3.1-8B Q4_K_M, whose Q6_K matrices are the ones affected.
const MMVQ_REPEAT_MAX: usize = 12;
/// `Some(n)` routes an FP8 GEMM through `mma_e4m3_cutlass` at `n_tokens >=
/// n`; `None` (unset) never does. New and opt-in rather than a measured
/// default: `cutlass_vs_block` puts the crossover against `mma_e4m3_block`
/// around 128-256 tokens on the 27B's FFN shapes, but that was measured in
/// isolation, not against a real serve loop yet.
#[cfg(feature = "cutlass")]
fn ffn_cutlass_min_tokens() -> Option<usize> {
    static T: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("INFERO_FFN_CUTLASS").ok().and_then(|v| v.parse().ok()))
}
/// Granularity at which decode graphs are captured. A captured graph fixes
/// every launch parameter including `kv_len`, so it is rounded up to a bucket
/// and re-captured only when the bucket changes; attention already masks
/// `j > positions[token]`, so a longer `kv_len` costs a little wasted work and
/// changes no result.
const GRAPH_KV_BUCKET: usize = 64;

/// The bucket, overridable so the trade can be measured: coarser buckets mean
/// fewer captures and more masked KV read per step, finer ones the reverse.
fn graph_kv_bucket() -> usize {
    std::env::var("INFERO_KV_BUCKET")
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
/// It is not the instantiation. `INFERO_GRAPH_MODE` was added to price the two
/// alternatives and they are the same number: `autofree` 8.46 ms a step,
/// `plain` plus an explicit `upload()` 8.60, and `INSTANTIATE_FLAG_UPLOAD` is
/// rejected outright by the driver (`CUDA_ERROR_INVALID_VALUE`) because it
/// needs the `WithParams` form. Most of the 721 us is the node-level tracing
/// that measured it: the same server runs a 7.71 ms step without `nsys` and an
/// 8.72 ms step under it. Dropping the graph entirely costs 0.8 ms a step
/// (`INFERO_NO_GRAPH=1`: 9.30 against 8.49), so the graph is paying — it just
/// is not free.
///
/// The switch stays so the result is re-runnable; the default is what it has
/// always been.
fn graph_instantiate_flags() -> infero_gpu::GraphFlags {
    use infero_gpu::GraphFlags as F;
    static PLAIN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *PLAIN.get_or_init(|| std::env::var("INFERO_GRAPH_MODE").as_deref() == Ok("plain")) {
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
struct SendGraph(infero_gpu::Graph);

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
    x: Buf<f32>,
    /// Normalized copy of `x` feeding the projections.
    xb: Buf<f32>,
    q: Buf<f32>,
    k: Buf<f32>,
    v: Buf<f32>,
    attn: Buf<f32>,
    proj: Buf<f32>,
    gate: Buf<f32>,
    up: Buf<f32>,
    ffn: Buf<f32>,
    /// `[n_heads, chunk, max_seq]`
    scores: Buf<f32>,
    logits: Buf<f32>,
    token_ids: Buf<i32>,
    positions: Buf<i32>,
    /// `[chunk, 3]`, token-major `[T, H, W]` — fed to the rope kernels
    /// instead of `positions` whenever `cfg.mrope_section` is set. Allocated
    /// unconditionally (`3 * chunk` i32, trivial next to the buffers around
    /// it) rather than behind an `Option`, so every model pays the same fixed
    /// shape and only `pos_stride` decides which buffer the rope call reads.
    mrope_positions: Buf<i32>,
    /// Per token: which sequence row it belongs to.
    seq_of: Buf<i32>,
    /// Per token: the pool slot its key/value go to.
    slots: Buf<i32>,
    /// Batch rows whose logits are wanted.
    logit_rows: Buf<i32>,
    attn_partial: Buf<f32>,
    /// Those rows, gathered and normalized.
    head_in: Buf<f32>,
    /// The attention output gate, `[chunk, d_attn]`. Allocated only for models
    /// whose attention blocks carry one.
    attn_gate: Option<Buf<f32>>,
    /// GatedDeltaNet scratch, allocated only for models that have such blocks.
    gdn: Option<GdnActs>,
    /// Mixture-of-experts scratch, allocated only for sparse models.
    moe: Option<MoeActs>,
}

/// Activation buffers a sparse FFN needs.
///
/// Sized by `chunk * n_active` rows rather than `chunk`: every token expands to
/// `n_active` expert rows on the way in and collapses back on the way out. At
/// `MAX_BATCH_TOKENS` of 256 and top-8 that is 2048 rows, which is 6 MiB at the
/// expert width and 16 at the model width — small enough not to be behind a
/// second flag, unlike the GatedDeltaNet buffers next door.
struct MoeActs {
    /// `[chunk, n_experts]` — the router's output, before the top-k.
    router_logits: Buf<f32>,
    /// `[chunk, n_active]`, descending by router logit.
    ids: Buf<i32>,
    weights: Buf<f32>,
    /// `[chunk * n_active, d_ff_expert]` each.
    gate: Buf<f32>,
    up: Buf<f32>,
    hidden: Buf<f32>,
    /// `[chunk * n_active, d_model]` — what the combine reduces.
    down: Buf<f32>,
}

/// Activation buffers the GatedDeltaNet block needs.
///
/// Kept behind an `Option` because on a pure attention model every one of these
/// is dead weight, and on Qwen3.8-27B they are not small: the packed row alone
/// is 40 KiB a token.
struct GdnActs {
    /// `[chunk, conv_channels + value_dim]` — `in_proj_qkv` and `in_proj_z`'s
    /// output in one row, when the loader could stack them; see
    /// [`GdnWeights::in_proj_qz`]. `split2` scatters this into `qkv` and `z`
    /// right after the matmul, the same shape `split_qkv` already handles for
    /// attention.
    qz: Buf<f32>,
    /// `[chunk, conv_channels]` — the input projection's output.
    qkv: Buf<f32>,
    /// The same after the convolution and SiLU. A separate buffer because the
    /// convolution reads three tokens back and would otherwise consume values
    /// it had already overwritten.
    qkv_conv: Buf<f32>,
    /// `[chunk, value_dim]` — the output gate.
    z: Buf<f32>,
    /// `[chunk, value_heads]` each.
    a: Buf<f32>,
    b: Buf<f32>,
    /// `[chunk, 2 * value_heads]` — `a` and `b` from the stacked projection,
    /// interleaved a token at a time. Unused when the loader could not stack.
    ab: Buf<f32>,
    beta: Buf<f32>,
    g: Buf<f32>,
    /// `[chunk, value_dim]` — the recurrence's output, before the gated norm.
    core: Buf<f32>,
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
    stage: [Buf<u8>; 2],
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
    k_rot: Buf<f32>,
    v_rot: Buf<f32>,
    q_rot: Buf<f32>,
    /// `S'·(Π·q)`, the query side of the QJL inner product.
    q_qjl: Buf<f32>,
    /// The attention output before `Πᵀ` maps it back.
    acc_rot: Buf<f32>,
}

/// Staging for the cuBLAS path: a dequantized weight matrix and f16 inputs.
struct Scratch {
    w16: Buf<f16>,
    x16: Buf<f16>,
    /// The activation row in Q8_1, for the integer mat-vec.
    q8_1: Buf<u8>,
    /// The activation, dynamically quantized to e4m3 for `mma_e4m3_block`'s
    /// native W8A8 GEMM — `quantize_act_e4m3`'s `xq`/`xs` outputs.
    xq_e4m3: Buf<u8>,
    xs_e4m3: Buf<f32>,
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
    /// False when `INFERO_NO_MMVQ` is set, forcing decode through the float
    /// mat-vec. Read once at load; the point is to be able to A/B the integer
    /// path's accuracy against a reference that shares everything else.
    use_mmvq: bool,
    /// False when `INFERO_NO_MMQ` is set, sending batches back through
    /// `dequant_to_f16` + cuBLAS. Separate from `use_mmvq` so the tensor-core
    /// GEMM can be A/B'd without also disabling the batch-1 mat-vec.
    use_mmq: bool,
    /// Decode graphs by (tokens, kv bucket). A step issues roughly 700 kernel
    /// launches; replaying one graph removes that cost.
    graphs: std::collections::HashMap<(u64, usize, usize, bool), GraphSlot>,
    /// Cleared by `INFERO_NO_GRAPH`, for measuring what the graphs are worth.
    use_graph: bool,
    max_logit_rows: usize,
    /// Tokens one forward pass carries, resolved once against this session's
    /// context length. See [`batch_tokens_for`].
    batch_tokens: usize,
    max_seq: usize,
    /// Per layer, whether it mixes with a recurrence. Cached because the pool
    /// needs it to size the state and the layer loop consults it every block.
    layer_kinds: Vec<bool>,
    logits_host: Vec<f32>,
    /// Rows the last forward pass left in `act.logits`.
    logit_rows: usize,
    /// Device buffers for [`Model::sample_on_device`], allocated on first use.
    samp: Option<SampleBufs>,
    /// Device-time attribution for a step's three phases; see `PhaseEvents`.
    phase_ev: Option<PhaseEvents>,
    /// Whether the last forward was a decode step — every row wanting logits and
    /// bringing one token. `PhaseEvents` averages only those; a prefill costs
    /// about ten times a decode and inflated `gpu_ms` when it did not.
    last_decode_only: bool,
    /// The multi-token-prediction head, once [`Model::install_mtp_head`] has
    /// loaded it. Absent on a checkpoint with no head, and on one whose head has
    /// not been asked for.
    mtp: Option<mtp::MtpHead>,
    /// `[max_logit_rows, d_model]` — the rows the last pass took logits from,
    /// **after** the final norm.
    ///
    /// Copied out of `act.xb` because that is the one buffer that holds what the
    /// MTP head consumes and the next thing to run overwrites it. Which of the
    /// two hidden states the head wants cannot be settled by the acceptance rate
    /// — `pre_fc_norm_hidden` renormalizes, so feeding the pre-norm one drafts
    /// about as well on real text — so it is settled by reading vLLM's runner,
    /// which passes `Qwen3NextModel.forward`'s return value, whose last statement
    /// is the final norm. `tests/qwen35_mtp.rs` pins it numerically against the
    /// capture, which carries both tensors for exactly this reason.
    mtp_hidden: Option<Buf<f32>>,
    /// The journal that undoes a rejected candidate's effect on the recurrent
    /// state. Only allocated for a model that has linear-attention blocks.
    gdn_rollback: Option<spec::GdnRollback>,
    /// The vision tower, once [`Model::load_vision_tower`] has run.
    ///
    /// Loaded on request rather than with the text weights: it is 921 MiB that
    /// a text-only deployment should not pay for, and the same reasoning the
    /// MTP head is loaded under.
    vision: Option<weights::VisionTower>,
    /// One vision call's activations, sized by the largest image admitted.
    vision_scratch: Option<infero_kernels::vision::VisionScratch>,
}

/// Device-side scratch for sampling. Sized once, at the batch and vocabulary
/// the model was built for.
struct SampleBufs {
    params: infero_gpu::Buf<f32>,
    /// Slice winners for the split greedy argmax; see
    /// `Kernels::sample_rows_greedy`.
    arg_v: infero_gpu::Buf<f32>,
    arg_i: infero_gpu::Buf<i32>,
    pen_tok: infero_gpu::Buf<i32>,
    pen_cnt: infero_gpu::Buf<i32>,
    pen_len: infero_gpu::Buf<i32>,
    rnd: infero_gpu::Buf<f64>,
    out: infero_gpu::Buf<u32>,
    /// Per-slice top-k candidates for `Kernels::sample_rows_split`.
    cand_v: infero_gpu::Buf<f32>,
    cand_i: infero_gpu::Buf<i32>,
    /// The surviving distribution each row drew from, which is what the
    /// speculative acceptance rule composes with.
    surv_id: infero_gpu::Buf<u32>,
    surv_p: infero_gpu::Buf<f32>,
    surv_len: infero_gpu::Buf<i32>,
    stride: usize,
    /// Entries the candidate and survivor buffers hold a row.
    top_k: usize,
}

/// Where a step's *GPU* time goes, under `INFERO_PHASE_EVENTS`.
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
    ev: Vec<infero_gpu::Event>,
    /// Accumulated spans and the step count, so one line covers many steps.
    sums: [f64; 3],
    steps: u64,
}

impl PhaseEvents {
    fn new(dev: &Device) -> Result<Option<Self>> {
        if std::env::var_os("INFERO_PHASE_EVENTS").is_none() {
            return Ok(None);
        }
        let mut ev = Vec::new();
        for _ in 0..4 {
            ev.push(
                dev.context()
                    .new_event(Some(infero_gpu::EVENT_DEFAULT))?,
            );
        }
        Ok(Some(Self { ev, sums: [0.0; 3], steps: 0 }))
    }
}

/// Where a forward pass spent its wall clock, under `INFERO_STEP_TIMING`.
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
        let on = *ON.get_or_init(|| std::env::var_os("INFERO_STEP_TIMING").is_some());
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
        let shards = infero_safetensors::Shards::open_dir(dir)?;
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

    /// Assemble a model from weights that are already on the device.
    ///
    /// The seam a test needs. Qwen3.5's block stack is the only one in this
    /// engine that carries recurrent state, and the only checkpoint of it is 51
    /// GiB — which does not fit on the card this is developed on, and did not
    /// load at all until recently. Speculative decoding's hardest requirement is
    /// about exactly that state, so the alternative to this constructor is
    /// testing the rollback on a model that has nothing to roll back.
    ///
    /// Nothing else uses it: the loaders build their own weights and call the
    /// same private assembly.
    pub fn from_weights(
        dev: Device,
        cfg: Config,
        w: Weights,
        max_seq: usize,
        kv_quant: KvQuant,
        max_logit_rows: usize,
    ) -> Result<Self> {
        let kern = Kernels::new(dev.clone());
        kern.warm_up()?;
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
        let layer_kinds: Vec<bool> = w.layers.iter().map(|l| l.is_linear()).collect();
        let n_linear = layer_kinds.iter().filter(|k| **k).count();
        if n_linear > 0 {
            tracing::info!(
                linear = n_linear,
                attention = cfg.n_layers - n_linear,
                "the block stack is not homogeneous"
            );
        }
        // Resolved here and stored, not recomputed: the buffers below are sized
        // from it and `forward` splits its input by it, and those two disagreeing
        // is the kind of mismatch `add_assign` already cost a day of.
        let needs_scores = needs_score_buffer(&kern, &cfg, max_seq, kv_quant);
        // `MMA_MAX_TOKENS_FP8` bounds `mma_e4m3_block`'s legacy interleaved-
        // layout GEMM, which declines a wider shape outright and falls back
        // to a much slower kernel. The CUTLASS/unified path's GEMM has no
        // such shape decline -- it just tiles a wider `M` into more blocks --
        // so applying that ceiling there only throttles the chunk for a
        // constraint that doesn't exist on the kernel actually running.
        // `CUTLASS_BATCH_TOKENS` is its own, separately measured ceiling on
        // the real 30552-token prefill. Re-measured 2026-08-30 after the
        // CUTLASS f32-direct-epilogue and `gdn_conv` chunking fixes changed
        // what these kernels' own per-call costs look like: 1024 (the prior
        // optimum) measured 7.40s; 2048 measured 7.30s; 4096 measured 7.22s;
        // 8192 measured 7.08s -- and 8192 is not an arbitrarily-chosen bigger
        // number: it's vLLM's own resolved `max_num_batched_tokens` for its
        // OpenAI-compatible server on this exact GPU class (>=70 GiB, not an
        // A100 -- see `vllm/engine/arg_utils.py`'s
        // `_set_default_max_num_seqs_and_batched_tokens_args`, which gives
        // `UsageContext.OPENAI_API_SERVER` 8192 and `UsageContext.LLM_CLASS`
        // 16384 on hardware like this). 16384 measured faster still (6.95s)
        // but OOMs a real `--ctx 65536` server on `attn_prefill`'s partial-
        // reduction scratch, which scales with the chunk; 8192 does not.
        const CUTLASS_BATCH_TOKENS: usize = 8192;
        let fp8_ceiling = if w.dominant_type() != infero_kernels::WeightType::F8E4M3 {
            None
        } else if weights::fp8_unified_layout() {
            Some(CUTLASS_BATCH_TOKENS)
        } else {
            Some(infero_kernels::fp8::MMA_MAX_TOKENS_FP8)
        };
        let batch_tokens =
            batch_tokens_for(cfg.n_heads, max_seq, max_logit_rows, needs_scores, fp8_ceiling);
        tracing::info!(
            batch_tokens,
            score_mib = cfg.n_heads * batch_tokens * (if needs_scores { max_seq } else { 1 }) * 4 >> 20,
            "tokens a pass carries"
        );
        let act = Activations::new(&dev, &cfg, max_seq, max_logit_rows, needs_scores, batch_tokens)?;
        let scratch = Scratch {
            w16: dev
                .stream()
                .alloc_zeros::<f16>(cfg.max_layer_weight_elements())?,
            x16: dev
                .stream()
                // Holds an f16 copy of whichever activation feeds the next
                // matmul: the FFN hidden, the residual, or the attention
                // output — and that last one is `d_attn` wide, which stopped
                // being covered by `d_model` on Qwen3.8.
                .alloc_zeros::<f16>(
                    batch_tokens * cfg.d_ff.max(cfg.d_model).max(cfg.d_attn()),
                )?,
            q8_1: dev.stream().alloc_zeros::<u8>(
                // A sparse FFN quantizes `down`'s activation once per active
                // expert per token, so its row count is `batch_tokens *
                // n_active` rather than `batch_tokens` — narrower rows, more
                // of them, and the product is what has to fit.
                (batch_tokens * Kernels::q8_1_bytes(cfg.d_ff.max(cfg.d_model))).max(
                    cfg.moe.as_ref().map_or(0, |m| {
                        batch_tokens * m.n_active * Kernels::q8_1_bytes(m.d_ff_expert)
                    }),
                ),
            )?,
            // Same width bound as `x16`: whichever activation feeds the next
            // FP8 projection, quantized a byte a value instead of a half.
            xq_e4m3: dev
                .stream()
                .alloc_zeros::<u8>(batch_tokens * cfg.d_ff.max(cfg.d_model).max(cfg.d_attn()))?,
            xs_e4m3: dev.stream().alloc_zeros::<f32>(
                batch_tokens
                    * cfg
                        .d_ff
                        .max(cfg.d_model)
                        .max(cfg.d_attn())
                        .div_ceil(infero_kernels::fp8::ACT_QUANT_GROUP),
            )?,
        };

        let tq = if kv_quant.is_quantized() {
            anyhow::ensure!(
                cfg!(feature = "cuda"),
                "KV cache quantization ({kv_quant:?}) has no kernels on this backend yet; \
                 pass --kv-quant f16 or drop the flag"
            );
            let chunk = batch_tokens;
            let kv_dim = cfg.d_kv();
            // These three hold rotated queries and the attention accumulator,
            // so they are `d_attn` wide, not `d_model`.
            let d_attn = cfg.d_attn();
            Some(TqBuffers {
                tables: TqTables::new(&dev, cfg.d_head, kv_quant)?,
                k_rot: dev.stream().alloc_zeros::<f32>(chunk * kv_dim)?,
                v_rot: dev.stream().alloc_zeros::<f32>(chunk * kv_dim)?,
                q_rot: dev.stream().alloc_zeros::<f32>(chunk * d_attn)?,
                q_qjl: dev.stream().alloc_zeros::<f32>(chunk * d_attn)?,
                acc_rot: dev.stream().alloc_zeros::<f32>(chunk * d_attn)?,
            })
        } else {
            None
        };

        let use_mmvq = std::env::var_os("INFERO_NO_MMVQ").is_none();
        if !use_mmvq {
            tracing::warn!("INFERO_NO_MMVQ set: decode will use the float mat-vec");
        }
        // Per-kernel timing records and synchronises CUDA events, which is
        // illegal on a stream that is capturing. The two tools answer different
        // questions anyway: a graph hides launch cost, and profiling measures
        // it, so asking for one turns the other off.
        // Graph replay needs stream capture, which is CUDA's. Metal's nearest
        // mechanism is an indirect command buffer -- a fixed argument set
        // encoded up front rather than a captured stream -- so the replay path
        // is off here and every step is encoded fresh. That is where a good
        // part of this backend's per-step overhead lives: 880 dispatches at
        // 18.3 us of command-buffer submit each.
        let use_graph = cfg!(feature = "cuda")
            && std::env::var_os("INFERO_NO_GRAPH").is_none()
            && !dev.profile().enabled();
        let use_mmq = std::env::var_os("INFERO_NO_MMQ").is_none();
        if !use_mmq {
            tracing::warn!("INFERO_NO_MMQ set: batches will use dequant + cuBLAS");
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
            batch_tokens,
            max_seq,
            layer_kinds,
            logit_rows: 0,
            samp: None,
            phase_ev: PhaseEvents::new(&dev)?,
            last_decode_only: false,
            logits_host,
            mtp: None,
            mtp_hidden: None,
            gdn_rollback: None,
            vision: None,
            vision_scratch: None,
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// Logit rows one pass may ask for, which bounds how wide a
    /// verification pass can be — and so which tree shapes are possible.
    pub fn max_logit_rows(&self) -> usize {
        self.max_logit_rows
    }

    /// Tokens one forward pass carries. See [`batch_tokens_for`].
    pub fn batch_tokens(&self) -> usize {
        self.batch_tokens
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

    /// Which encoding most of the weights are in, by bytes.
    ///
    /// For a safetensors checkpoint this is the only place the answer exists:
    /// the file states dtypes per tensor and the loader may re-encode them, so
    /// what the model is *running* is a property of the loaded matrices rather
    /// than of the file.
    pub fn dominant_weight_type(&self) -> infero_kernels::WeightType {
        self.w.dominant_type()
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
            // Which blocks carry recurrent state. Read off the loaded weights
            // rather than derived from the config, for the same reason the
            // loader reads it there: the tensors are what the forward pass will
            // actually use.
            &self.layer_kinds,
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

        let chunk_len = self.batch_tokens();
        let n_chunks = tokens.len().div_ceil(chunk_len);
        for (i, chunk) in tokens.chunks(chunk_len).enumerate() {
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
        // One logit row per item that asked for one, which is its last token.
        let tail: Vec<usize> = items
            .iter()
            .map(|i| usize::from(i.wants_logits))
            .collect();
        self.forward_batch_rows(items, pool, &tail)
    }

    /// [`Self::forward_batch_device`] with the logits of more than one token per
    /// sequence.
    ///
    /// `tail[i]` is how many of item `i`'s *trailing* tokens want logits, which
    /// generalizes `wants_logits` (0 or 1) without changing what a caller who
    /// does not need it writes. Speculative verification is the reason it exists:
    /// a pass over `k + 1` candidates needs the target's own prediction at every
    /// one of them, because `logits[j]` is what decides candidate `j`, and taking
    /// only the last row would leave the acceptance rule with nothing to compare.
    pub fn forward_batch_rows(
        &mut self,
        items: &[BatchItem<'_>],
        pool: &mut KvPool,
        tail: &[usize],
    ) -> Result<usize> {
        let phase = crate::StepPhases::start();
        anyhow::ensure!(!items.is_empty(), "empty batch");
        anyhow::ensure!(
            tail.len() == items.len(),
            "{} logit-row counts for {} items",
            tail.len(),
            items.len()
        );
        let n_tokens: usize = items.iter().map(|i| i.tokens.len()).sum();
        anyhow::ensure!(n_tokens > 0, "batch carries no tokens");
        anyhow::ensure!(
            n_tokens <= self.batch_tokens(),
            "batch of {n_tokens} tokens exceeds the {} a pass can carry",
            self.batch_tokens()
        );
        for (i, (item, want)) in items.iter().zip(tail).enumerate() {
            anyhow::ensure!(
                *want <= item.tokens.len(),
                "item {i} brings {} tokens and {want} of them want logits",
                item.tokens.len()
            );
        }
        let n_logit_rows: usize = tail.iter().sum();
        self.last_decode_only = n_tokens == n_logit_rows;
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
        // Token-major `[T, H, W]`, built only for a model with M-RoPE -- see
        // `Kernels::rope_qk_partial`'s doc comment and `BatchItem::mrope`.
        // `positions` above already carries the plain running length every
        // model needs for the slot table and the causal mask; this is a
        // second, parallel array only the rope call reads.
        let has_mrope = self.cfg.mrope_section.is_some();
        let mut mrope = if has_mrope {
            Vec::with_capacity(n_tokens * 3)
        } else {
            Vec::new()
        };
        let mut kv_len = 0usize;

        // Per sequence slot: where its tokens begin in this flat batch, and how
        // long the sequence already was. The recurrence needs the first; the
        // reset decision needs the second.
        let mut starts = vec![(0usize, 0usize); pool.max_seqs()];
        for (item, want) in items.iter().zip(tail) {
            let start = pool.len(item.seq);
            if item.seq.0 < starts.len() {
                starts[item.seq.0] = (token_ids.len(), start);
            }
            let taken = pool.extend(item.seq, item.tokens.len())?;
            if has_mrope {
                match item.mrope {
                    Some(m) => {
                        anyhow::ensure!(
                            m.len() == 3 * item.tokens.len(),
                            "a batch item carries {} mrope entries for {} tokens, \
                             expected {} -- the position array was built for a \
                             different splice than the tokens it is paired with",
                            m.len(),
                            item.tokens.len(),
                            3 * item.tokens.len()
                        );
                        mrope.extend_from_slice(m);
                    }
                    None => {
                        for k in 0..item.tokens.len() {
                            let p = (start + k) as i32 + item.mrope_delta;
                            mrope.extend_from_slice(&[p, p, p]);
                        }
                    }
                }
            }
            for (k, (&tok, &slot)) in item.tokens.iter().zip(&taken).enumerate() {
                token_ids.push(tok as i32);
                seq_of.push(item.seq.0 as i32);
                positions.push((start + k) as i32);
                slots.push(slot);
            }
            kv_len = kv_len.max(start + item.tokens.len());
            // The last `want` of this item's rows, in order, so that a caller
            // reading the logits back finds candidate `j` at row `j`.
            let end = token_ids.len();
            for row in end - want..end {
                logit_rows.push(row as i32);
            }
        }
        // Whether this whole pass is one item — one sequence, contiguous,
        // causal positions increasing by exactly one — which is the only
        // shape `Kernels::attn_prefill` may see (see its doc comment: a tile
        // spanning two sequences would read one's KV slots for the other's
        // rows). A batch of more than one item is not attempted here even
        // when every item happens to be a wide prefill chunk; splitting
        // `attn_prefill`'s tiling across item boundaries mid-kernel is a
        // sharper version of the same hazard and is not implemented.
        let single_seq_run = (items.len() == 1).then(|| items[0].tokens.len());
        // The pool slot this call's one sequence actually occupies, when
        // there is exactly one AND the run is at least `gdn_conv_prefill`'s
        // own `MIN_CHUNK` (32) tokens wide. The kernel is named, and scoped
        // by its wrapper, for the prefill case specifically; gating this on
        // nothing but `items.len() == 1` also took every plain one-sequence
        // *decode* step (one token, sometimes two under speculation) down
        // the same path, and produced visibly degenerate output within a
        // dozen or so decode steps -- caught by testing the real server end
        // to end, not by the kernel-level unit test alone, which only
        // exercised prefill-sized runs. `SeqId` is that slot's index into
        // the pool's per-slot GDN arrays directly (see `KvPool::alloc`), and
        // slot 0 is only the *common* case, not a guarantee, once slots have
        // cycled.
        const MIN_GDN_CONV_PREFILL_RUN: usize = 32;
        let single_seq_slot = (single_seq_run.unwrap_or(0) >= MIN_GDN_CONV_PREFILL_RUN)
            .then(|| items[0].seq.0);

        phase.mark(2);
        let stream = self.dev.stream().clone();
        if let Some(pe) = &self.phase_ev {
            pe.ev[0].record(&stream)?;
        }
        stream.memcpy_htod(&token_ids, &mut self.act.token_ids.slice_mut(..n_tokens))?;
        stream.memcpy_htod(&seq_of, &mut self.act.seq_of.slice_mut(..n_tokens))?;
        stream.memcpy_htod(&positions, &mut self.act.positions.slice_mut(..n_tokens))?;
        if has_mrope {
            stream.memcpy_htod(
                &mrope,
                &mut self.act.mrope_positions.slice_mut(..n_tokens * 3),
            )?;
        }
        stream.memcpy_htod(&slots, &mut self.act.slots.slice_mut(..n_tokens))?;
        if n_logit_rows > 0 {
            stream.memcpy_htod(
                &logit_rows,
                &mut self.act.logit_rows.slice_mut(..n_logit_rows),
            )?;
        }

        // The per-slot layout the recurrence kernels index by, and a reset for
        // any sequence that is starting from nothing.
        //
        // "No tokens seen means no state" is the invariant that keeps a reused
        // slot from carrying a previous conversation's recurrence into the next
        // one. `KvPool::alloc` cannot enforce it — it has no device to memset
        // with — so it is enforced here, where the length is known.
        if pool.has_recurrent_state() {
            let mut spans = vec![(0i32, 0i32); pool.max_seqs()];
            for item in items {
                let slot = item.seq.0;
                anyhow::ensure!(
                    slot < spans.len(),
                    "sequence slot {slot} is past the recurrent pool's {} slots",
                    spans.len()
                );
                anyhow::ensure!(
                    spans[slot].1 == 0,
                    "sequence slot {slot} appears twice in one batch; the \
                     recurrence walks a sequence's tokens in order and cannot \
                     take them in two pieces within a single call"
                );
                spans[slot] = (starts[slot].0 as i32, item.tokens.len() as i32);
                if starts[slot].1 == 0 {
                    pool.reset_recurrent(&self.dev, item.seq)?;
                }
            }
            pool.set_gdn_layout(&self.dev, &spans)?;
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

        // Vision features go in here, over the rows the placeholder ids just
        // gathered a real embedding into.
        //
        // After the gather and not instead of it: the placeholder's own
        // embedding is overwritten, so gathering it is wasted work — but it is
        // one row of one kernel, and skipping it would mean a masked gather,
        // which is a second code path through the hottest launch in the step for
        // no measurable gain.
        //
        // Row indices are per item and then offset by where that item's tokens
        // start in the batch, because a batch interleaves sequences and the
        // placeholder positions in `item.tokens` are relative to the item.
        if items.iter().any(|i| i.vision.is_some()) {
            let mut base = 0usize;
            for item in items {
                if let Some(f) = item.vision {
                    anyhow::ensure!(
                        f.out_hidden == d,
                        "vision features are {} wide and the embedding is {d}",
                        f.out_hidden
                    );
                    let rows = self.vision_targets(item.tokens)?;
                    // A chunk before or after the placeholder run has none --
                    // common once one clip's run spans several chunks, see
                    // `BatchItem::vision_row_offset`'s doc comment.
                    if !rows.is_empty() {
                        let n = rows.len();
                        anyhow::ensure!(
                            item.vision_row_offset + n <= f.tokens,
                            "this chunk wants rows {}..{} of a {}-row clip; a \
                             chunk-slicing bug in the caller, not a request problem",
                            item.vision_row_offset,
                            item.vision_row_offset + n,
                            f.tokens
                        );
                        let shifted: Vec<i32> =
                            rows.iter().map(|r| *r + base as i32).collect();
                        let dst = self.dev.stream().clone_htod(&shifted)?;
                        self.kern.vision_splice(
                            &mut self.act.x.slice_mut(..n_tokens * d),
                            &f.rows_view(item.vision_row_offset, n),
                            &dst.as_view(),
                            d,
                            n,
                        )?;
                    }
                }
                base += item.tokens.len();
            }
        }

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
        // The journal's state is part of the shape. A graph records the copies
        // that stage a layer's recurrent state and journal its inputs, so a
        // captured verification pass and a captured ordinary pass of the same
        // width are *different* graphs — and they collide, because `k + 1` tokens
        // is also a prefill chunk length. Replaying the wrong one is silent both
        // ways round: an ordinary pass would advance a working copy and throw its
        // state update away, and a verification pass would advance the persistent
        // state and then have its journal replayed on top of it.
        // `attn_prefill`'s tiling is chosen from *this call's* item layout
        // (`run_base`/`run_tokens`), not read from device memory the way
        // `attn_decode`'s per-token grid is — so unlike every other kernel a
        // captured graph replays here, its launch would be wrong for any
        // later call that reuses this graph's key (`n_tokens`, bucketed
        // `kv_len`) with a different item layout (say, two items instead of
        // one summing to the same `n_tokens`). Simplest correct fix: a pass
        // that would use it is never captured or replayed, only run eagerly.
        // One block's worth of tiling (four warps of two tokens at this
        // model's group) is the smallest run `attn_prefill` was measured
        // against; below that, `attn_decode`'s one-token-a-block kernel is
        // both simpler and already fast enough.
        const MIN_PREFILL_RUN: usize = 8;
        let prefill_run = single_seq_run.filter(|&t| t >= MIN_PREFILL_RUN);

        let armed = self.gdn_rollback.as_ref().is_some_and(|r| r.is_armed());
        let key = (
            pool.id(),
            n_tokens,
            kv_len.next_multiple_of(graph_kv_bucket()),
            armed,
        );
        let graphable =
            self.use_graph && self.offload.is_none() && key.2 <= self.max_seq && prefill_run.is_none();

        match self.graphs.get(&key) {
            Some(GraphSlot::Ready(g)) if graphable => g.0.launch()?,
            slot => {
                let record = graphable && matches!(slot, Some(GraphSlot::Warm));
                let stream = self.dev.stream().clone();
                if record {
                    stream.begin_capture(
                        infero_gpu::CAPTURE_RELAXED,
                    )?;
                }
                let kv = if graphable { key.2 } else { kv_len };
                let res = (|| -> Result<()> {
                    for layer in 0..n_layers {
                        let s = self.stage_layer(layer)?;
                        // The block stack is not homogeneous on Qwen3.5: 48 of
                        // its 64 blocks mix with a recurrence. Dispatched from
                        // the loaded weights rather than from a stride, so a
                        // model with a different interleaving needs no change
                        // here.
                        // The residual before any block runs -- the embedding
                        // alone. A divergence already present here is a gather
                        // or a token id, not a layer.
                        if layer == 0 {
                            probe(&self.kern, layer, "embedding", &self.act.x.slice(..d));
                        }
                        if self.layer_kinds[layer] {
                            self.linear_attention(layer, n_tokens, pool, s, single_seq_slot)?;
                        } else {
                            self.attention(layer, n_tokens, kv, dims, pool, s, prefill_run)?;
                        }
                        self.feed_forward(layer, n_tokens, s)?;
                        self.release_layer(s)?;
                        // `INFERO_LAYER_RMS=1` reports the residual stream's
                        // magnitude after every block. Nine single-suspect A/Bs
                        // came back negative, so the question stops being
                        // "which component" and becomes "which layer": a stream
                        // that grows smoothly and then jumps names the block to
                        // read, where a component-by-component search does not.
                        // Only meaningful with INFERO_NO_GRAPH=1 — a device copy
                        // cannot happen inside a capture region.
                        if std::env::var_os("INFERO_LAYER_RMS").is_some() {
                            let stream = self.kern.device().stream();
                            let row = stream.clone_dtoh(&self.act.x.slice(..d))?;
                            self.kern.device().synchronize()?;
                            let rms =
                                (row.iter().map(|v| v * v).sum::<f32>() / d as f32).sqrt();
                            let bad = row.iter().filter(|v| !v.is_finite()).count();
                            tracing::info!(layer, rms, non_finite = bad, "layer rms");
                        }
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

        // The MTP head's second input: every token's hidden state, after the
        // final norm.
        //
        // Every token and not just the rows that wanted logits, because the
        // drafter needs a history. Its slot `p` holds `(h_p, emb(t_{p+1}))`, so a
        // prompt of `n` tokens gives it `n` slots to attend over — vLLM hands its
        // drafter the whole `target_hidden_states` array for the same reason. A
        // head primed only on the rows that were sampled from would attend to a
        // cache with holes in it, which is not an error and is not the model.
        //
        // Before the early return below: a mid-prompt chunk takes no logits and
        // still has to reach the drafter.
        if let Some(h) = self.mtp_hidden.as_mut() {
            self.kern.rms_norm(
                &mut h.slice_mut(..n_tokens * d),
                &self.act.x.slice(..n_tokens * d),
                &self.w.output_norm.as_view(),
                n_tokens,
                d,
                rms_eps,
            )?;
        }
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
            && self.kern.device().caps().int_tensor_gemm
            && Self::mmq_shape_ok(head);
        // Asked and answered: this checkpoint's head is Q8_0, 248320 x 5120,
        // and at one row it takes `mmvq` — 885 us for 1.29 GB, which is 1460
        // GB/s and the same rate the FP8 projections get. Nothing to win here.
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

        // `INFERO_LOGIT_PROBE=1` reports the last row's top-5 ids and values.
        // The residual stream is healthy all the way to layer 35 (RMS climbs
        // 0.98 → 62 with no non-finite value), so whatever is wrong sits after
        // the blocks. This splits the two remaining candidates: a sane argmax
        // whose text is wrong means detokenization, and a nonsense argmax means
        // the final norm or the vocab projection.
        if std::env::var_os("INFERO_LOGIT_PROBE").is_some() {
            let row = n_logit_rows - 1;
            let start = row * vocab_size;
            let v = self
                .dev
                .stream()
                .clone_dtoh(&self.act.logits.slice(start..start + vocab_size))?;
            self.dev.synchronize()?;
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_unstable_by(|a, b| v[*b].total_cmp(&v[*a]));
            let top: Vec<(usize, f32)> = idx.iter().take(5).map(|i| (*i, v[*i])).collect();
            let bad = v.iter().filter(|x| !x.is_finite()).count();
            tracing::info!(?top, non_finite = bad, vocab_size, "logit probe");
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
    /// Lay out the device sampler's inputs for these rows: the penalty windows
    /// as sorted unique ids with counts, the four parameters a row, and the
    /// uniform draws. Grows the buffers when a wider `top_k` or window arrives.
    ///
    /// Shared by [`Self::sample_on_device`] and [`Self::survivors_on_device`],
    /// which differ only in what they ask the kernel to return.
    fn prepare_sample_bufs(
        &mut self,
        rows: &[RowSample],
        windows: &[&[u32]],
        max_k: usize,
    ) -> Result<()> {
        let vocab = self.cfg.vocab_size;
        let n = rows.len();
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
        let fits = matches!(
            &self.samp,
            Some(b) if b.stride >= stride && b.pen_len.len() >= n && b.top_k >= max_k
        );
        if !fits {
            self.samp = Some(SampleBufs {
                params: stream.alloc_zeros::<f32>(self.max_logit_rows * 4)?,
                arg_v: stream.alloc_zeros::<f32>(
                    self.max_logit_rows * Kernels::ARGMAX_SPLITS,
                )?,
                arg_i: stream.alloc_zeros::<i32>(
                    self.max_logit_rows * Kernels::ARGMAX_SPLITS,
                )?,
                pen_tok: stream.alloc_zeros::<i32>(self.max_logit_rows * stride)?,
                pen_cnt: stream.alloc_zeros::<i32>(self.max_logit_rows * stride)?,
                pen_len: stream.alloc_zeros::<i32>(self.max_logit_rows)?,
                rnd: stream.alloc_zeros::<f64>(self.max_logit_rows)?,
                out: stream.alloc_zeros::<u32>(self.max_logit_rows)?,
                cand_v: stream.alloc_zeros::<f32>(
                    self.max_logit_rows * Kernels::SAMPLE_SPLITS * max_k,
                )?,
                cand_i: stream.alloc_zeros::<i32>(
                    self.max_logit_rows * Kernels::SAMPLE_SPLITS * max_k,
                )?,
                surv_id: stream.alloc_zeros::<u32>(self.max_logit_rows * max_k)?,
                surv_p: stream.alloc_zeros::<f32>(self.max_logit_rows * max_k)?,
                surv_len: stream.alloc_zeros::<i32>(self.max_logit_rows)?,
                stride,
                top_k: max_k,
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
        Ok(())
    }

    /// The truncated distribution each logit row would be sampled from, on the
    /// device.
    ///
    /// Speculative verification needs `p` as numbers over its support: the
    /// acceptance test is `min(1, p(x)/q(x))` and a rejection draws from
    /// `(p - q)+`. Both are over the nucleus, tens of entries. Reconstructing it
    /// on the host means copying `n * vocab` floats — 2.98 MB at three rows on
    /// the 27B — and walking the whole vocabulary once a row, which measured
    /// about 1.7 ms of a 29.6 ms round.
    ///
    /// Returns `None` when the shape is one the device sampler declines, so the
    /// caller keeps its host path.
    ///
    /// The distributions come back normalized, which is what
    /// `Sampler::pick` and `Self::draw_residual` want with `total = 1.0`. They
    /// take unnormalized weights and a normalizer, and `w / 1.0` is `w`.
    pub fn survivors_on_device(
        &mut self,
        rows: &[RowSample],
        windows: &[&[u32]],
    ) -> Result<Option<Vec<Vec<(u32, f32)>>>> {
        let vocab = self.cfg.vocab_size;
        let n = rows.len();
        anyhow::ensure!(n == windows.len(), "{n} rows against {} windows", windows.len());
        if n == 0 || n != self.logit_rows {
            return Ok(None);
        }
        let max_k = rows.iter().map(|r| r.top_k as usize).max().unwrap_or(1).max(1);
        self.prepare_sample_bufs(rows, windows, max_k)?;
        let b = self.samp.as_mut().unwrap();
        let stride = b.stride;
        let top_k = b.top_k;
        {
            let (pv, tv, cv, lv, rv) = (
                b.params.slice(..n * 4),
                b.pen_tok.slice(..n * stride),
                b.pen_cnt.slice(..n * stride),
                b.pen_len.slice(..n),
                b.rnd.slice(..n),
            );
            let mut out_v = b.out.slice_mut(..n);
            let mut cav = b.cand_v.slice_mut(..n * Kernels::SAMPLE_SPLITS * max_k);
            let mut cai = b.cand_i.slice_mut(..n * Kernels::SAMPLE_SPLITS * max_k);
            let mut id_v = b.surv_id.slice_mut(..n * top_k);
            let mut p_v = b.surv_p.slice_mut(..n * top_k);
            let mut len_v = b.surv_len.slice_mut(..n);
            self.kern.sample_rows_split(
                &mut out_v,
                &mut cav,
                &mut cai,
                &self.act.logits.slice(..n * vocab),
                &pv,
                &tv,
                &cv,
                &lv,
                &rv,
                n,
                vocab,
                stride,
                max_k,
                Some(infero_kernels::Survivors {
                    id: &mut id_v,
                    p: &mut p_v,
                    len: &mut len_v,
                    stride: top_k,
                }),
            )?;
        }
        let stream = self.dev.stream().clone();
        let mut lens = vec![0i32; n];
        let mut ids = vec![0u32; n * top_k];
        let mut ps = vec![0f32; n * top_k];
        stream.memcpy_dtoh(&b.surv_len.slice(..n), &mut lens)?;
        stream.memcpy_dtoh(&b.surv_id.slice(..n * top_k), &mut ids)?;
        stream.memcpy_dtoh(&b.surv_p.slice(..n * top_k), &mut ps)?;
        self.dev.synchronize()?;
        let out = (0..n)
            .map(|r| {
                let keep = (lens[r].max(0) as usize).min(top_k);
                ids[r * top_k..r * top_k + keep]
                    .iter()
                    .copied()
                    .zip(ps[r * top_k..r * top_k + keep].iter().copied())
                    .collect()
            })
            .collect();
        Ok(Some(out))
    }

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

        // Every row greedy means the whole batch can take the split argmax,
        // which fills the device instead of giving each row one block. The test
        // is the kernel's own `is_greedy()`: zero temperature, or a top-k of one.
        let all_greedy = rows.iter().all(|r| r.temperature <= 0.0 || r.top_k == 1);
        self.prepare_sample_bufs(rows, windows, max_k.max(1))?;
        let b = self.samp.as_mut().unwrap();
        let stride = b.stride;
        let stream = self.dev.stream().clone();

        let (params_v, tok_v, cnt_v, len_v, rnd_v) = (
            b.params.slice(..n * 4),
            b.pen_tok.slice(..n * stride),
            b.pen_cnt.slice(..n * stride),
            b.pen_len.slice(..n),
            b.rnd.slice(..n),
        );
        let mut out_v = b.out.slice_mut(..n);
        // Escape hatch for the same reason `INFERO_NO_MMQ` and its neighbours
        // exist: the person measuring a change here is entitled to the old
        // kernel to compare against.
        let no_split = std::env::var_os("INFERO_NO_SAMPLE_SPLIT").is_some();
        if all_greedy {
            let (mut av, mut ai) = (
                b.arg_v.slice_mut(..n * Kernels::ARGMAX_SPLITS),
                b.arg_i.slice_mut(..n * Kernels::ARGMAX_SPLITS),
            );
            self.kern.sample_rows_greedy(
                &mut out_v,
                &mut av,
                &mut ai,
                &self.act.logits.slice(..n * vocab),
                &params_v,
                &tok_v,
                &cnt_v,
                &len_v,
                n,
                vocab,
                stride,
            )?;
        } else if no_split {
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
                // The main decode path needs the token, not the distribution.
                None,
            )?;
        } else {
            // `sample_rows` scans the vocabulary in one block a row — 16 rows
            // is 16 blocks on 188 SMs, and it measured 4.37 ms of a ~6 ms
            // step, worse than the layers that produced the logits it is
            // sampling from. `sample_rows_split` is the same distribution,
            // `SAMPLE_SPLITS` blocks a row instead of one; `survivors_on_device`
            // already takes it for the speculative path, this is the same
            // buffers for the plain-decode one.
            // `top_k = 0` means unlimited, not zero candidates — the same
            // floor `prepare_sample_bufs` already applied when it sized these
            // buffers, and what the kernel's own `cand_k` falls back to.
            let cand_k = max_k.max(1);
            let (mut cav, mut cai) = (
                b.cand_v.slice_mut(..n * Kernels::SAMPLE_SPLITS * cand_k),
                b.cand_i.slice_mut(..n * Kernels::SAMPLE_SPLITS * cand_k),
            );
            self.kern.sample_rows_split(
                &mut out_v,
                &mut cav,
                &mut cai,
                &self.act.logits.slice(..n * vocab),
                &params_v,
                &tok_v,
                &cnt_v,
                &len_v,
                &rnd_v,
                n,
                vocab,
                stride,
                cand_k,
                // The main decode path needs the token, not the distribution.
                None,
            )?;
        }
        if let Some(pe) = &self.phase_ev {
            pe.ev[3].record(&stream)?;
        }
        let mut host = vec![0u32; n];
        stream.memcpy_dtoh(&self.samp.as_ref().unwrap().out.slice(..n), &mut host)?;
        self.dev.synchronize()?;
        // The events are settled now, so the spans can be read without a wait
        // of their own. One line every 64 steps.
        //
        // Decode steps only, the way `Scheduler`'s own window is. A prefill step
        // costs about ten times a decode, so counting them here inflated
        // `gpu_ms` — and comparing that inflated average against a wall step
        // derived from throughput is how "3.2% of a step is not on the GPU"
        // came to be published when the real figure is 1.2%. The test is that
        // every row wants logits and brought exactly one token, which is what a
        // decode step is.
        let decode_only = self.last_decode_only;
        if let Some(pe) = self.phase_ev.as_mut().filter(|_| decode_only) {
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

    /// One GatedDeltaNet block: the linear-attention mixer.
    ///
    /// The shape of the computation, in the order it happens:
    ///
    ///   xb           = rms_norm(x, attn_norm)
    ///   qkv          = xb @ in_proj_qkv        [n, 2*key_dim + value_dim]
    ///   z            = xb @ in_proj_z          [n, value_dim]
    ///   a, b         = xb @ in_proj_a, in_proj_b
    ///   qkv          = silu(depthwise_causal_conv(qkv))
    ///   beta, g      = sigmoid(b), -exp(A_log) * softplus(a + dt_bias)
    ///   q, k         = l2norm in place; q also scaled by 1/sqrt(dk)
    ///   core         = gated delta rule, advancing this sequence's state
    ///   proj         = out_proj(rms_norm(core, norm) * silu(z))
    ///
    /// Two things make this unlike `attention`. The state is advanced in place
    /// and persists between calls, so the tokens of one sequence must arrive
    /// contiguous and in order — which `forward_batch` guarantees, since it lays
    /// a batch out sequence by sequence. And there is no KV cache and no rotary:
    /// position enters only through the order the recurrence sees the tokens in.
    fn linear_attention(
        &mut self,
        layer: usize,
        n: usize,
        pool: &mut KvPool,
        slot: Option<usize>,
        single_seq_slot: Option<usize>,
    ) -> Result<()> {
        let la = self
            .cfg
            .linear_attn
            .context("a linear-attention block in a model whose config has no linear dimensions")?;
        let d = self.cfg.d_model;
        let eps = self.cfg.rms_eps;
        let (key_dim, val_dim) = (la.key_dim(), la.value_dim());
        let width = la.conv_channels();
        let heads = la.value_heads;
        // Asked before the activations are borrowed apart, because it takes
        // `&self`.
        let ffn_absorbs = self.ffn_norm_takes_residual(layer, n);
        let ordinal = pool
            .gdn()
            .and_then(|g| g.ordinal_of(layer))
            .context("no recurrent state slot for a linear-attention layer")?;
        let n_seqs = pool.max_seqs();

        // Normalize the residual stream. No fused f16 variant here: its point is
        // to hand an f16 activation to an MMQ q/k/v group, and this block has
        // none.
        let (shared, shared_f16) = Self::norm_for_group(
            &self.kern,
            &mut self.scratch,
            &mut self.act.xb,
            &self.act.x.slice(..n * d),
            &self.w.layers[layer].attn_norm.as_view(),
            n,
            d,
            eps,
            false,
            false,
        )?;

        // The activations have to be taken apart: the projections read `xb`
        // while writing the GatedDeltaNet buffers, and the output projection
        // writes `proj` while reading them.
        let Activations {
            xb, x, proj, gdn, ..
        } = &mut self.act;
        let acts = gdn
            .as_mut()
            .context("this model has linear-attention blocks but no buffers for them")?;
        let gw = self.w.layers[layer]
            .gdn
            .as_ref()
            .expect("dispatched to the linear path for a layer with no gdn weights");
        let stage = slot.map(|s| &self.offload.as_ref().unwrap().stage[s]);

        // The four input projections share the normalized residual. Not grouped
        // into a fused mat-vec: `in_proj_a` and `in_proj_b` are `value_heads`
        // columns wide — 48 against 10240 — and the fusion helper wants
        // same-shaped matrices anyway.
        //
        // `a` and `b` are the exception the loader stacks: `qkv` and `z` are
        // FP8 and go to tensor cores, but `a` and `b` are F16 and 48 rows wide,
        // so each took a `gemv` whose 14.2 us was all launch against 0.34 us of
        // bytes — 96 of a decode step's 104 `gemv` launches. Stacked they are
        // one launch, and `ab` holds them interleaved a token at a time.
        let stacked = gw.in_proj_ba.as_ref();
        let gate: Vec<(&Matrix, &mut Buf<f32>, usize)> = match stacked {
            Some(ba) => vec![(ba, &mut acts.ab, 2 * heads)],
            None => vec![
                (&gw.in_proj_a, &mut acts.a, heads),
                (&gw.in_proj_b, &mut acts.b, heads),
            ],
        };
        // `in_proj_qkv` and `in_proj_z` run separately, 640 and 384 blocks
        // against 188 SMs, whenever the loader could not stack them: `ncu`
        // put those at 56% and 34% achieved occupancy despite 100% theoretical
        // — not enough blocks to fill the device for the launch's whole
        // duration, the same shape as the gated q projection's own underfill.
        // Stacked, the pair is 1024 blocks. See `GdnWeights::in_proj_qz`.
        let qz: Vec<(&Matrix, &mut Buf<f32>, usize)> = match gw.in_proj_qz.as_ref() {
            Some(fused) => vec![(fused, &mut acts.qz, width + val_dim)],
            None => vec![
                (&gw.in_proj_qkv, &mut acts.qkv, width),
                (&gw.in_proj_z, &mut acts.z, val_dim),
            ],
        };
        let fused_qz = gw.in_proj_qz.is_some();
        for (m, out, cols) in qz.into_iter().chain(gate) {
            Self::matmul_pre(
                &self.kern,
                &mut self.scratch,
                &mut out.slice_mut(..n * cols),
                m,
                stage,
                &xb.slice(..n * d),
                n,
                self.use_mmvq,
                self.use_mmq,
                shared,
                shared_f16,
            )?;
        }
        if fused_qz {
            self.kern.split2(
                &mut acts.qkv.slice_mut(..n * width),
                &mut acts.z.slice_mut(..n * val_dim),
                &acts.qz.slice(..n * (width + val_dim)),
                width,
                val_dim,
                n,
            )?;
        }

        let (first, ntok, mut recurrent, mut conv) = pool.gdn_parts(ordinal);
        let seqs = infero_kernels::gdn::SeqLayout {
            first_token: &first,
            n_tokens: &ntok,
            n_seqs,
            total_tokens: n,
        };

        // A speculative verification pass in flight: keep this layer's
        // convolution taps and take a working copy of its recurrent state, so
        // that the pass can be undone down to the accepted prefix. Both are
        // no-ops on an ordinary step. See `crate::spec`.
        let armed = self.gdn_rollback.as_ref().is_some_and(|r| r.is_armed());
        if armed {
            let r = self.gdn_rollback.as_mut().unwrap();
            r.stage(&self.kern, ordinal, &conv.as_view(), &recurrent.as_view())?;
        }

        // The convolution needs a separate output: it reads three tokens back,
        // so writing in place would consume values it had already overwritten.
        // `gdn_conv_prefill` also splits the token dimension across blocks --
        // real headroom for the one-sequence case (measured 8.28% achieved
        // occupancy without it, at this checkpoint's channel count), and
        // correct but no better for a many-short-sequence decode batch, which
        // is why it's scoped to one sequence *this call* (`single_seq_slot`)
        // rather than replacing `gdn_conv` outright.
        //
        // That is deliberately not `n_seqs == 1` (the pool's configured
        // capacity, `--max-seqs`): a server almost always runs with room for
        // more than one concurrent request, so that check was true only for
        // the single-sequence *benchmark harness* and never in production --
        // this fast path sat unreachable behind it. `gdn_conv_prefill` itself
        // still asserts `n_seqs == 1`, because the one sequence's row in the
        // pool's arrays is not necessarily row 0 once slots have cycled, so
        // it gets a fresh one-element `SeqLayout` sliced at that sequence's
        // own slot rather than the pool-wide arrays sliced from the front.
        if let Some(active_slot) = single_seq_slot {
            let one_first = first.slice(active_slot..active_slot + 1);
            let one_ntok = ntok.slice(active_slot..active_slot + 1);
            let one = infero_kernels::gdn::SeqLayout {
                first_token: &one_first,
                n_tokens: &one_ntok,
                n_seqs: 1,
                total_tokens: n,
            };
            // `conv`, like `first`/`n_tok` above, is laid out one window a
            // slot (`GdnState::conv_span`) -- unlike them it is a `ViewMut`
            // `gdn_conv_prefill` also writes through, so it needs its own
            // slice rather than sharing `conv` unsliced, which would read and
            // write slot 0's window regardless of which slot is active.
            let conv_n = width * (la.conv_kernel - 1);
            let mut one_conv = conv.slice_mut(active_slot * conv_n..(active_slot + 1) * conv_n);
            self.kern.gdn_conv_prefill(
                &mut acts.qkv_conv.slice_mut(..n * width),
                &acts.qkv.slice(..n * width),
                &mut one_conv,
                &gw.conv1d.as_view(),
                &one,
                width,
                la.conv_kernel,
            )?;
        } else {
            self.kern.gdn_conv(
                &mut acts.qkv_conv.slice_mut(..n * width),
                &acts.qkv.slice(..n * width),
                &mut conv,
                &gw.conv1d.as_view(),
                &seqs,
                width,
                la.conv_kernel,
            )?;
        }

        // Stacked, `a` is columns `[0, heads)` of each `2 * heads`-wide row and
        // `b` is the rest, so the same buffer goes in twice with `b` offset and
        // the stride says how to walk it.
        let (a_off, b_off, stride) = if stacked.is_some() {
            (0, heads, 2 * heads)
        } else {
            (0, 0, heads)
        };
        let a_src = if stacked.is_some() { &acts.ab } else { &acts.a };
        let b_src = if stacked.is_some() { &acts.ab } else { &acts.b };
        // The kernel's last read is `(n - 1) * stride + heads - 1` past each
        // pointer, so `b`'s slice is short by `b_off` and no more.
        let b_len = n * stride - b_off;
        self.kern.gdn_gate_decay(
            &mut acts.beta.slice_mut(..n * heads),
            &mut acts.g.slice_mut(..n * heads),
            &a_src.slice(a_off..a_off + n * stride),
            &b_src.slice(b_off..b_off + b_len),
            &gw.a_log.as_view(),
            &gw.dt_bias.as_view(),
            n,
            heads,
            stride,
        )?;

        // q and k are normalized where they lie, inside the packed row.
        self.kern.gdn_qk_l2norm(
            &mut acts.qkv_conv.slice_mut(..n * width),
            n,
            la.key_heads,
            la.key_head_dim,
            width,
            0,
            key_dim,
            1e-6,
        )?;

        // The journal, recorded here and not a line earlier or later. What the
        // recurrence is about to consume is the packed row *after* the
        // convolution and *after* `q` and `k` were l2-normalized, and those are
        // the values a replay has to feed it — journalling `acts.qkv` instead
        // would replay the recurrence over unfiltered, unnormalized inputs, run
        // to completion, and leave a state that is wrong by a few percent.
        if armed {
            let r = self.gdn_rollback.as_mut().unwrap();
            r.record(
                &self.kern,
                ordinal,
                crate::spec::GdnTap {
                    pre_conv: acts.qkv.slice(..n * width),
                    post_conv: acts.qkv_conv.slice(..n * width),
                    g: acts.g.slice(..n * heads),
                    beta: acts.beta.slice(..n * heads),
                },
            )?;
        }
        // The recurrence runs on the working copy while a verification pass is
        // in flight, leaving the persistent state at its pre-step value for the
        // replay to advance. One 3 MiB copy a layer, which
        // `GdnRollback::KERNEL_WANTED` would remove entirely.
        let mut staged = if armed {
            Some(self.gdn_rollback.as_mut().unwrap().state_scratch_mut())
        } else {
            None
        };
        let state = staged.as_mut().unwrap_or(&mut recurrent);
        // Same reasoning and same slot, sliced the same way, as the
        // `gdn_conv`/`gdn_conv_prefill` choice above: `Kernels::
        // gdn_delta_rule` picks its own column-split fast path by
        // `seqs.n_seqs == 1`, and `pool.max_seqs()` would make that always
        // false in any real server (`--max-seqs` > 1), never true outside
        // the single-sequence benchmark harness that first measured the
        // split path's win — exactly the bug `single_seq_slot`'s own note
        // above describes, reproduced here on a different kernel before it
        // ever shipped, because `n_seqs` in `seqs` below is that same
        // `pool.max_seqs()` value. `state`, like `conv`, is one block a
        // slot (`heads * dk * dv` floats), so it needs the same per-slot
        // slice the fast path assumes, not the pool-wide buffer unsliced.
        let one_seq_layout;
        let mut one_state;
        let delta_first;
        let delta_ntok;
        let (seqs_for_delta, state_for_delta): (&infero_kernels::gdn::SeqLayout<'_>, &mut ViewMut<'_, f32>) =
            match single_seq_slot {
                Some(active_slot) => {
                    let state_n = heads * la.key_head_dim * la.value_head_dim;
                    one_state = state.slice_mut(active_slot * state_n..(active_slot + 1) * state_n);
                    delta_first = first.slice(active_slot..active_slot + 1);
                    delta_ntok = ntok.slice(active_slot..active_slot + 1);
                    one_seq_layout = infero_kernels::gdn::SeqLayout {
                        first_token: &delta_first,
                        n_tokens: &delta_ntok,
                        n_seqs: 1,
                        total_tokens: n,
                    };
                    (&one_seq_layout, &mut one_state)
                }
                None => (&seqs, state),
            };
        self.kern.gdn_delta_rule(
            &mut acts.core.slice_mut(..n * val_dim),
            state_for_delta,
            &acts.qkv_conv.slice(..n * width),
            &acts.g.slice(..n * heads),
            &acts.beta.slice(..n * heads),
            seqs_for_delta,
            heads,
            la.key_heads,
            la.key_head_dim,
            la.value_head_dim,
            (width, 0, key_dim, 2 * key_dim),
                    la.v_heads_tiled,
        )?;
        // `staged` borrowed the journal for the launch above; letting it fall out
        // of scope here rather than at the end of the function keeps the journal
        // available to the rest of the block.
        let _ = staged;

        // Normalize each head's output, then gate it with silu(z). This order
        // matters and the other one runs; `gdn_gated_rmsnorm` says why. `qkv` is
        // reused as the destination — its projection has been consumed by the
        // convolution, and this saves a `value_dim`-wide buffer a token.
        self.kern.gdn_gated_rmsnorm(
            &mut acts.qkv.slice_mut(..n * val_dim),
            &acts.core.slice(..n * val_dim),
            &acts.z.slice(..n * val_dim),
            &gw.norm.as_view(),
            n * heads,
            la.value_head_dim,
            eps,
        )?;

        Self::matmul_pre(
            &self.kern,
            &mut self.scratch,
            &mut proj.slice_mut(..n * d),
            &gw.out_proj,
            stage,
            &acts.qkv.slice(..n * val_dim),
            n,
            self.use_mmvq,
            self.use_mmq,
            None,
            false,
        )?;

        // Same residual protocol as `attention`: leave the block's output in
        // `proj` when the FFN's norm will absorb it, add it here otherwise. A
        // disagreement drops the residual or applies it twice.
        if !ffn_absorbs {
            self.kern
                .add_assign(&mut x.slice_mut(..n * d), &proj.slice(..n * d), n * d)?;
        }
        Ok(())
    }

    fn attention(
        &mut self,
        layer: usize,
        n: usize,
        kv_len: usize,
        dims: AttnDims,
        pool: &mut KvPool,
        slot: Option<usize>,
        prefill_run: Option<usize>,
    ) -> Result<()> {
        let stage = slot.map(|s| &self.offload.as_ref().unwrap().stage[s]);
        let cfg = &self.cfg;
        let d = cfg.d_model;
        // The attention interior's width. Equal to `d` on every model before
        // Qwen3.8, which ran 24 heads of 256 against a 5120 residual — so
        // reading `d` for a query row is right by accident there and wrong
        // here. `d` stays the residual stream and the projections' input; `da`
        // is q, the packed row, the attention output, and anything strided by a
        // query row.
        let da = cfg.d_attn();
        let kv_dim = cfg.d_kv();
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
            && [&l.attn().wq, &l.attn().wk, &l.attn().wv]
                .iter()
                .all(|w| Kernels::has_mmvq(w.ty) && w.k == d);
        // Only when this group will actually take the f16 GEMM: at one token
        // the mat-vec runs instead and that path wants Q8_1.
        let want_h = self.use_mmq
            && n > 1
            && n <= infero_kernels::MMQ_MAX_TOKENS
            && Kernels::mmq_f16_variant_for(l.attn().wq.ty).is_some()
            && [&l.attn().wq, &l.attn().wk, &l.attn().wv].iter().all(|w| {
                matches!(
                    w.ty,
                    infero_kernels::WeightType::Q4G128
                        | infero_kernels::WeightType::Q4G128T
                ) && w.k == d
            });
        let eps = cfg.rms_eps;
        let (shared, shared_f16) = if self.attn_norm_takes_residual(layer, n) {
            // The previous layer's FFN left its output in `proj` for this.
            self.kern.add_rms_norm_f16(
                &mut self.act.xb.slice_mut(..n * d),
                Some(&mut self.scratch.x16.slice_mut(..n * d)),
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
        probe(&self.kern, layer, "after_attn_norm", &self.act.xb.slice(..n * d));

        // All three read the same Q8_1 activation, so one launch covers them.
        // Two hundred and twenty-five mat-vecs back to back run at 328 GB/s
        // where one alone runs at 392 — each drains before the next can start
        // — and merging this group and the FFN's removes ninety-six of those.
        if !l.attn().output_gate
            && Self::fusable(&[&l.attn().wq, &l.attn().wk, &l.attn().wv], shared, n, self.use_mmvq)
        {
            // `q8_1_bytes(d)`: the operand is the residual row, which is `d`
            // wide. `q`'s *output* is `da` wide, and the two differ on
            // Qwen3-30B-A3B — the view was `..d`, which the kernel overran
            // harmlessly because `act.q` is allocated `chunk * da`. Naming the
            // real width keeps it that way for a reason rather than by luck.
            //
            // A gated `wq` is excluded above rather than left for `fusable`
            // to catch: `mmvq_fused3` writes each output at its own weight's
            // width, and a gated `wq` is `2 * da` wide with the query and
            // gate interleaved per head — writing that straight into a
            // `da`-wide `q` slice does not recover "the first half", it
            // recovers half of every head, which is what the `else` branch's
            // `split_interleaved` exists to undo. `wk`/`wv` stay ordinary
            // width, which is why `fusable` itself never noticed: it only
            // compares `ty`/`k` across the group, and those still agree.
            let bytes = Kernels::q8_1_bytes(d);
            let (q, k_, v) = (&mut self.act.q, &mut self.act.k, &mut self.act.v);
            self.kern.mmvq_fused3(
                &mut q.slice_mut(..da),
                &mut k_.slice_mut(..kv_dim),
                &mut v.slice_mut(..kv_dim),
                &l.attn().wq.view(stage)?,
                &l.attn().wk.view(stage)?,
                &l.attn().wv.view(stage)?,
                l.attn().wq.ty,
                &self.scratch.q8_1.slice(..bytes),
                d,
                [l.attn().wq.n, l.attn().wk.n, l.attn().wv.n],
            )?;
            for (bias, out, cols) in [
                (&l.attn().bq, &mut self.act.q, da),
                (&l.attn().bk, &mut self.act.k, kv_dim),
                (&l.attn().bv, &mut self.act.v, kv_dim),
            ] {
                if let Some(b) = bias {
                    self.kern
                        .add_bias(&mut out.slice_mut(..n * cols), &b.as_view(), cols, n)?;
                }
            }
        } else if let Some(w) = l.attn().w_qkv.as_ref().filter(|w| {
            // `want_h` is the AWQ mmq case specifically -- capped at
            // `MMQ_MAX_TOKENS` because that is where its kernel stops
            // applying. FP8's own dispatch inside `matmul_pre` already falls
            // back past its own batching limit, so the fused buffer is worth
            // taking at any `n` there; `w_qkv` only exists at all when
            // `stacked3` found it safe to build.
            n > 1 && (want_h || w.ty == infero_kernels::WeightType::F8E4M3)
        }) {
            // One matmul for all three, then a scatter. Separately they cost
            // 14.7 + 8.5 + 8.5 us a layer at a batch of 32 because the two
            // narrow ones cannot fill the device; stacked they cost 16.7. The
            // scatter is what buys that without a row stride in `rope_qk`,
            // `store_kv` and `attn_scores`.
            //
            // `act.gate` is the staging buffer: it is the FFN's, already wide
            // enough, and nothing in this block will read it before the FFN
            // writes it again.
            let fused_w = da + 2 * kv_dim;
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
            packed_qkv = l.attn().bq.is_none()
                && l.attn().bk.is_none()
                && l.attn().bv.is_none()
                && self.tq.is_none();
            if !packed_qkv {
                self.kern.split_qkv(
                    &mut self.act.q.slice_mut(..n * da),
                    &mut self.act.k.slice_mut(..n * kv_dim),
                    &mut self.act.v.slice_mut(..n * kv_dim),
                    &self.act.gate.slice(..n * fused_w),
                    // q's width inside the fused row, which is the attention
                    // interior's — not the residual's. The two are equal on
                    // every model but Qwen3.5.
                    da,
                    kv_dim,
                    n,
                )?;
            }
            for (bias, out, cols) in [
                (&l.attn().bq, &mut self.act.q, da),
                (&l.attn().bk, &mut self.act.k, kv_dim),
                (&l.attn().bv, &mut self.act.v, kv_dim),
            ] {
                if let Some(b) = bias {
                    self.kern
                        .add_bias(&mut out.slice_mut(..n * cols), &b.as_view(), cols, n)?;
                }
            }
        } else {
        // A gated q projection is twice as wide and its two halves interleave
        // per head, so it lands in `gate` and is de-interleaved rather than
        // written straight to `q`. Everything downstream then sees the same
        // shapes it always did.
        if l.attn().output_gate {
            anyhow::ensure!(
                l.attn().bq.is_none(),
                "layer {layer} has both an output gate and a q bias; the bias \
                 would be 2 * d_attn wide and this path does not know which \
                 half it applies to"
            );
            Self::matmul_pre(
                &self.kern,
                &mut self.scratch,
                &mut self.act.gate.slice_mut(..n * 2 * da),
                &l.attn().wq,
                stage,
                &self.act.xb.slice(..n * d),
                n,
                self.use_mmvq,
                self.use_mmq,
                shared,
                shared_f16,
            )?;
            let Activations {
                q, gate, attn_gate, ..
            } = &mut self.act;
            let ag = attn_gate
                .as_mut()
                .context("a gated attention layer with no gate buffer allocated")?;
            self.kern.split_interleaved(
                &mut q.slice_mut(..n * da),
                &mut ag.slice_mut(..n * da),
                &gate.slice(..n * 2 * da),
                n,
                cfg.n_heads,
                cfg.d_head,
            )?;
        }
        // `wk`/`wv` fused into one matmul when the loader found it safe to
        // build (GGUF: always same shape, so `w_kv` is `Some` whenever this
        // is a GGUF checkpoint's non-linear-attention layer). One launch
        // instead of two on a pair that, run separately, are most of a
        // decode step's `gemv` launch count next to their bytes -- the same
        // shape `in_proj_ba`'s fusion already paid off on. `act.gate` is
        // free here: nothing in this branch has written it yet, and the
        // gated-q branch above already finished with it if it ran.
        let fused_kv = l.attn().bk.is_none() && l.attn().bv.is_none();
        if let Some(w_kv) = l.attn().w_kv.as_ref().filter(|_| fused_kv) {
            Self::matmul_pre(
                &self.kern,
                &mut self.scratch,
                &mut self.act.gate.slice_mut(..n * 2 * kv_dim),
                w_kv,
                stage,
                &self.act.xb.slice(..n * d),
                n,
                self.use_mmvq,
                self.use_mmq,
                shared,
                shared_f16,
            )?;
            let Activations { k, v, gate, .. } = &mut self.act;
            self.kern.split2(
                &mut k.slice_mut(..n * kv_dim),
                &mut v.slice_mut(..n * kv_dim),
                &gate.slice(..n * 2 * kv_dim),
                kv_dim,
                kv_dim,
                n,
            )?;
        }
        for (w, bias, out, cols) in [
            (&l.attn().wq, &l.attn().bq, &mut self.act.q, da),
            (&l.attn().wk, &l.attn().bk, &mut self.act.k, kv_dim),
            (&l.attn().wv, &l.attn().bv, &mut self.act.v, kv_dim),
        ] {
            // q is already done when it was gated; k and v are already done
            // when the fused pair above ran.
            if std::ptr::eq(w, &l.attn().wq) && l.attn().output_gate {
                continue;
            }
            if (std::ptr::eq(w, &l.attn().wk) || std::ptr::eq(w, &l.attn().wv))
                && l.attn().w_kv.is_some()
                && fused_kv
            {
                continue;
            }
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

        let fused_w = da + 2 * kv_dim;

        // Qwen3 normalizes every head of q and k before the rotary. The order
        // matters: rotating first and normalizing after is a different
        // function, and a model that wants this and does not get it produces
        // fluent nonsense rather than an error.
        //
        // In the packed path both q and k are still inside the `[q | k | v]`
        // row and are normalized in place there, q at offset 0 and k at
        // `offset = d`.
        //
        // q is the trap. `rope_qk_packed` takes it as `q_dst` — an *output*: it
        // reads q out of the packed row and writes the rotated result into
        // `act.q`. Normalizing `act.q` here instead would touch a buffer that
        // still holds the previous layer's values and is about to be
        // overwritten, so the q normalization becomes a silent no-op while k's
        // works, and the model generates degenerate repetition ("的博客 的博客
        // …") rather than failing. The unit tests cannot see this: they check
        // the kernel against a CPU reference, and the kernel was never wrong.
        // `INFERO_NO_QK_NORM=1` skips both, which is how a bad answer is
        // attributed: a checkpoint that needs QK-norm is degenerate without it,
        // so if the output is *equally* degenerate either way the fault is
        // somewhere else and this path is only taking the blame.
        static NO_QK_NORM: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let skip_qk_norm = *NO_QK_NORM.get_or_init(|| std::env::var_os("INFERO_NO_QK_NORM").is_some());

        if let Some(qn) = l.attn().q_norm.as_ref().filter(|_| !skip_qk_norm) {
            let (buf, stride, len) = if packed_qkv {
                (&mut self.act.gate, fused_w, n * fused_w)
            } else {
                (&mut self.act.q, da, n * da)
            };
            self.kern.qk_norm(
                &mut buf.slice_mut(..len),
                &qn.as_view(),
                n,
                cfg.n_heads,
                cfg.d_head,
                stride,
                0,
                cfg.rms_eps,
            )?;
        }
        // `INFERO_QK_NORM_PROBE=1` checks the invariant at the real call site
        // rather than against a reference implementation. RMS-normalizing head
        // h and scaling by `w` leaves `RMS(out_h / w) == 1`, so if the offsets
        // and stride are right every head reads 1.0 here. The unit tests cannot
        // establish this: they compare the kernel to a CPU reference built from
        // the same assumed layout, so a wrong assumption is wrong in both.
        if std::env::var_os("INFERO_QK_NORM_PROBE").is_some()
            && !skip_qk_norm
            && let Some(qn) = l.attn().q_norm.as_ref()
        {
            let stream = self.kern.device().stream();
            let width = if packed_qkv { fused_w } else { da };
            let row = stream.clone_dtoh(&self.act.gate.slice(..width.max(d)))?;
            let w = stream.clone_dtoh(&qn.as_view())?;
            self.kern.device().synchronize()?;
            let rms: Vec<f32> = (0..cfg.n_heads.min(4))
                .map(|h| {
                    let seg = &row[h * cfg.d_head..(h + 1) * cfg.d_head];
                    let acc: f32 = seg
                        .iter()
                        .zip(&w)
                        .map(|(o, wi)| {
                            let v = if wi.abs() > 1e-6 { o / wi } else { 0.0 };
                            v * v
                        })
                        .sum();
                    (acc / cfg.d_head as f32).sqrt()
                })
                .collect();
            tracing::info!(?rms, packed = packed_qkv, "qk_norm probe: RMS(q/w) per head, want 1.0");
        }

        if let Some(kn) = l.attn().k_norm.as_ref().filter(|_| !skip_qk_norm) {
            let (buf, stride, offset, len) = if packed_qkv {
                (&mut self.act.gate, fused_w, da, n * fused_w)
            } else {
                (&mut self.act.k, kv_dim, 0, n * kv_dim)
            };
            self.kern.qk_norm(
                &mut buf.slice_mut(..len),
                &kn.as_view(),
                n,
                cfg.n_kv_heads,
                cfg.d_head,
                stride,
                offset,
                cfg.rms_eps,
            )?;
        }

        // The same invariant for k. Two things went wrong with the first
        // version of this probe and both produced confident wrong readings:
        // it read from offset 0, so it measured q rather than k; and it sat
        // *above* the normalization it was meant to check, so it reported
        // un-normalized values (0.51, 0.60, 2.67, 0.067) that looked exactly
        // like a real bug. A probe placed before the thing it measures is not
        // a weaker check, it is a check of something else.
        if std::env::var_os("INFERO_QK_NORM_PROBE").is_some()
            && !skip_qk_norm
            && let Some(kn) = l.attn().k_norm.as_ref()
        {
            let stream = self.kern.device().stream();
            let (base, span) = if packed_qkv { (da, fused_w) } else { (0, kv_dim) };
            let row = stream.clone_dtoh(&self.act.gate.slice(..span))?;
            let w = stream.clone_dtoh(&kn.as_view())?;
            self.kern.device().synchronize()?;
            let rms: Vec<f32> = (0..cfg.n_kv_heads.min(4))
                .map(|h| {
                    let off = base + h * cfg.d_head;
                    let seg = &row[off..off + cfg.d_head];
                    let acc: f32 = seg
                        .iter()
                        .zip(&w)
                        .map(|(o, wi)| {
                            let v = if wi.abs() > 1e-6 { o / wi } else { 0.0 };
                            v * v
                        })
                        .sum();
                    (acc / cfg.d_head as f32).sqrt()
                })
                .collect();
            tracing::info!(?rms, "qk_norm probe: RMS(k/w) per kv head, want 1.0");
        }

        // `pos_stride == 1` (every model without M-RoPE) reproduces the
        // original scalar-position rope bit for bit: `self.act.positions`
        // holds one value a token, and `self.w.mrope_axis` is all zeros, so
        // `positions[token * 1 + 0]` is exactly `positions[token]`. A model
        // with `cfg.mrope_section` set reads `self.act.mrope_positions`
        // instead, `3` values a token, `self.w.mrope_axis[i]` choosing which.
        // See `Kernels::rope_qk_partial`'s doc comment.
        let pos_stride = if cfg.mrope_section.is_some() { 3 } else { 1 };
        let rope_positions = if pos_stride == 3 {
            self.act.mrope_positions.slice(..n * 3)
        } else {
            self.act.positions.slice(..n)
        };
        if packed_qkv {
            let (q, packed) = (&mut self.act.q, &mut self.act.gate);
            self.kern.rope_qk_packed_partial(
                &mut q.slice_mut(..n * da),
                &mut packed.slice_mut(..n * fused_w),
                fused_w,
                0,
                da,
                &rope_positions,
                &self.w.rope_freqs.as_view(),
                &self.w.mrope_axis.as_view(),
                pos_stride,
                n,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.d_head,
                // `rotary_dim` equals `d_head` on every model before Qwen3.5,
                // so this call is unchanged for all of them; there it is 64 of
                // 256 and the tail passes through.
                cfg.rotary_dim,
                cfg.rope_theta,
                cfg.rope_freq_scale,
                cfg.interleaved_rope,
            )?;
        } else {
            let (q, k) = (&mut self.act.q, &mut self.act.k);
            self.kern.rope_qk_partial(
                &mut q.slice_mut(..n * da),
                &mut k.slice_mut(..n * kv_dim),
                &rope_positions,
                &self.w.rope_freqs.as_view(),
                &self.w.mrope_axis.as_view(),
                pos_stride,
                n,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.d_head,
                cfg.rotary_dim,
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
                        // Where `k` and `v` start inside `[q | k | v]`, which is
                        // after `q` — and `q` is `da` wide, not `d`. The two are
                        // equal on every model that reached this path before
                        // Qwen3-30B-A3B, whose 32 heads of 128 make `da` twice
                        // `d`; passing `d` reads the *middle of q* as the keys
                        // and the values. The attention output came out 200x
                        // its right magnitude, and only above one token,
                        // because the single-token path does not pack. The
                        // rotary call above already takes `da` for this.
                        da,
                        da + kv_dim,
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
                    probe(&self.kern, layer, "q_after_rope", &self.act.q.slice(..n * da));
                    probe(&self.kern, layer, "k_after_rope", &self.act.k.slice(..n * kv_dim));
                    probe(&self.kern, layer, "v", &self.act.v.slice(..n * kv_dim));
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
                    && n <= infero_kernels::MMQ_MAX_TOKENS
                    && matches!(
                        l.attn().wo.ty,
                        infero_kernels::WeightType::Q4G128
                            | infero_kernels::WeightType::Q4G128T
                    )
                    && Kernels::mmq_f16_variant_for_shape(l.attn().wo.ty, l.attn().wo.n).is_some()
                    && Self::mmq_shape_ok(&l.attn().wo);
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
                // number here was wrong. `INFERO_DECODE_ATTN=0` restores the
                // three.
                // `prefill_run` is `Some(n)` only when this whole pass is one
                // item — one sequence, `n` tokens, contiguous, causal — which
                // `attn_prefill_ws` requires and the caller
                // (`forward_batch_rows`) has already checked; see its own doc
                // comment for why a narrower run is not attempted here.
                if let Some(run_tokens) = prefill_run.filter(|_| self.kern.prefill_attention(&dims)) {
                    self.kern.attn_prefill_ws(
                        &mut attn_out.slice_mut(..n * da),
                        &self.act.q.slice(..n * da),
                        &pool.dense(layer).0.as_view(),
                        &pool.dense(layer).1.as_view(),
                        batch,
                        dims,
                        0,
                        run_tokens,
                        kv_len,
                        attn_scale,
                        &mut partial.as_view_mut(),
                    )?;
                } else if !std::env::var("INFERO_DECODE_ATTN").is_ok_and(|v| v == "0")
                    && self.kern.decode_attention(&dims, kv_len)
                {
                    let mut h16 = self.scratch.x16.slice_mut(..n * da);
                    attn_f16 = self.kern.attn_decode(
                        &mut attn_out.slice_mut(..n * da),
                        wo_f16.then_some(&mut h16),
                        &self.act.q.slice(..n * da),
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
                        &mut attn_out.slice_mut(..n * da),
                        &self.act.q.slice(..n * da),
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
                        &self.act.q.slice(..n * da),
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
                        &mut attn_out.slice_mut(..n * da),
                        &self.act.scores.slice(..score_len),
                        &pool.dense(layer).1.as_view(),
                        batch,
                        dims,
                        kv_len,
                        Some(&mut partial.as_view_mut()),
                    )?;
                    probe(&self.kern, layer, "attn_out", &self.act.attn.slice(..n * da));
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
                    &mut tq.q_rot.slice_mut(..n * da),
                    &self.act.q.slice(..n * da),
                    &tq.tables.rotation.as_view(),
                    d_head,
                    n_q_vecs,
                )?;
                self.kern.tq_matvec(
                    &mut tq.q_qjl.slice_mut(..n * da),
                    &tq.q_rot.slice(..n * da),
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
                // Fused, the same way `attn_decode` fuses the dense path's
                // three kernels and for the same reason: the unfused path
                // below writes the whole score row to HBM and reads it back
                // twice, and at a batch of one that round trip is latency
                // rather than bytes. `INFERO_TQ_DECODE_ATTN=0` restores the
                // three-kernel path this replaces.
                if self.kern.tq_decode_attention(&dims) {
                    let (kcodes, ksigns, kscale, kgamma) = pool.tq_key(layer);
                    let (vcodes, vscale) = pool.tq_value(layer);
                    self.kern.tq_attn_decode(
                        &mut tq.acc_rot.slice_mut(..n * da),
                        &tq.q_rot.slice(..n * da),
                        &tq.q_qjl.slice(..n * da),
                        &kcodes.as_view(),
                        &ksigns.as_view(),
                        &kscale.as_view(),
                        &kgamma.as_view(),
                        &vcodes.as_view(),
                        &vscale.as_view(),
                        batch,
                        &tq.tables.k_levels.as_view(),
                        k_bits,
                        &tq.tables.v_levels.as_view(),
                        v_bits,
                        dims,
                        kv_len,
                        attn_scale,
                        quant.qjl_scale(),
                        &mut self.act.attn_partial.as_view_mut(),
                    )?;
                } else {
                    {
                        let (codes, signs, scale, gamma) = pool.tq_key(layer);
                        self.kern.tq_attn_scores(
                            &mut self.act.scores.slice_mut(..score_len),
                            &tq.q_rot.slice(..n * da),
                            &tq.q_qjl.slice(..n * da),
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
                            &mut tq.acc_rot.slice_mut(..n * da),
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
                }
                // Back out of the rotated basis, once, on the output.
                self.kern.tq_matvec(
                    &mut self.act.attn.slice_mut(..n * da),
                    &tq.acc_rot.slice(..n * da),
                    &tq.tables.rotation_t.as_view(),
                    d_head,
                    n_q_vecs,
                )?;
            }
        }

        // The output gate, applied to the attention output before anything
        // downstream reads it -- both the fast path just below and the
        // generic one after it read `self.act.attn` straight, so gating it
        // here once covers both rather than duplicating the call in each.
        // Sigmoid, not silu: the reference implementation does not read
        // config's `output_gate_type: "swish"`, and the two give different
        // answers. See `the_output_gate_is_sigmoid_not_silu`.
        if l.attn().output_gate {
            let Activations { attn, attn_gate, .. } = &mut self.act;
            let ag = attn_gate
                .as_ref()
                .context("a gated attention layer with no gate buffer allocated")?;
            self.kern.sigmoid_gate(
                &mut attn.slice_mut(..n * da),
                &ag.slice(..n * da),
                n * da,
            )?;
        }

        // Straight into the residual stream: this projection's result is only
        // ever added to it, and the mat-vec can do that itself.
        //
        // Gating happened above rather than being folded in here: a gated
        // layer's `self.act.attn` needed `sigmoid_gate` applied before
        // anything quantized it, and this path used to quantize straight off
        // the raw (un-gated) activation instead -- same width, same shape,
        // wrong values, so it decoded fluent-looking nonsense rather than
        // erroring. Moving the gate above this branch instead of excluding
        // gated layers from it keeps the fused mat-vec-plus-residual-add for
        // them too, rather than paying for a separate `add_assign` a layer.
        if l.attn().bo.is_none() && Self::residual_fusable(&l.attn().wo, n, self.use_mmvq) {
            // `da`, not `d`: this projection contracts over the attention
            // interior and produces the residual width. The two are equal on
            // every model that reached this branch before Qwen3-30B-A3B, whose
            // 32 heads of 128 make `da` twice `d` — and quantizing `d` of a
            // `da`-wide row and then claiming `k = d` does not read half the
            // weights, it reads the wrong *layout*: `nb` comes out 16 where the
            // rows are 32 blocks long, so the scale block lands inside the
            // quants and the mat-vec multiplies by reinterpreted nibbles. Every
            // logit was NaN by layer 0.
            let bytes = Kernels::q8_1_bytes(da);
            debug_assert_eq!(l.attn().wo.k, da, "output projection contracts over d_attn");
            self.kern.quantize_q8_1(
                &mut self.scratch.q8_1.slice_mut(..bytes),
                &self.act.attn.slice(..da),
                da,
            )?;
            self.kern.mmvq_add(
                &mut self.act.x.slice_mut(..d),
                &l.attn().wo.view(stage)?,
                l.attn().wo.ty,
                &self.scratch.q8_1.slice(..bytes),
                da,
                l.attn().wo.n,
            )?;
            return Ok(());
        }

        Self::matmul_pre(
            &self.kern,
            &mut self.scratch,
            &mut self.act.proj.slice_mut(..n * d),
            &l.attn().wo,
            stage,
            &self.act.attn.slice(..n * da),
            n,
            self.use_mmvq,
            self.use_mmq,
            None,
            attn_f16,
        )?;
        if let Some(b) = &l.attn().bo {
            self.kern
                .add_bias(&mut self.act.proj.slice_mut(..n * d), &b.as_view(), d, n)?;
        }
        probe(&self.kern, layer, "o_proj_out", &self.act.proj.slice(..n * d));
        if !self.ffn_norm_takes_residual(layer, n) {
            self.kern.add_assign(
                &mut self.act.x.slice_mut(..n * d),
                &self.act.proj.slice(..n * d),
                n * d,
            )?;
        }
        probe(&self.kern, layer, "x_after_attn", &self.act.x.slice(..n * d));
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
    /// layer's `attention` will do it. `INFERO_FUSE_RESIDUAL=0` turns off this
    /// one and the FFN one together.
    fn attn_norm_takes_residual(&self, layer: usize, n: usize) -> bool {
        // The bounds check comes first: `feed_forward` asks about `layer + 1`,
        // which is one past the end for the last block.
        if layer == 0 || layer >= self.cfg.n_layers {
            return false;
        }
        // A GatedDeltaNet block has no q/k/v group, so there is no f16-reading
        // consumer for the fused add-and-norm to feed. It adds explicitly.
        if self.w.layers[layer].is_linear() {
            return false;
        }
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *OFF.get_or_init(|| std::env::var("INFERO_FUSE_RESIDUAL").as_deref() == Ok("0")) {
            return false;
        }
        let l = &self.w.layers[layer];
        let prev = &self.w.layers[layer - 1];
        let d = self.cfg.d_model;
        // The consumer: this layer's q/k/v group has to be on the f16 path,
        // because the fused add-and-norm is the kernel that writes f16.
        let consumer = self.use_mmq
            && n > 1
            && n <= infero_kernels::MMQ_MAX_TOKENS
            && Kernels::mmq_f16_variant_for(l.attn().wq.ty).is_some()
            && [&l.attn().wq, &l.attn().wk, &l.attn().wv].iter().all(|w| {
                matches!(
                    w.ty,
                    infero_kernels::WeightType::Q4G128 | infero_kernels::WeightType::Q4G128T
                ) && w.k == d
            });
        // The producer: the previous layer's `down` has to have left its output
        // in `proj` rather than adding itself into the stream, which is what the
        // single-token mat-vec path does.
        // A sparse previous layer has no single `down` to ask about, and its
        // combine writes to `proj` unconditionally — so the producer side holds
        // and the question is only whether the consumer wants f16.
        match &prev.dense {
            Some(f) => consumer && !Self::residual_fusable(&f.w_down, n, self.use_mmvq),
            None => consumer,
        }
    }

    fn ffn_norm_takes_residual(&self, layer: usize, n: usize) -> bool {
        let l = &self.w.layers[layer];
        let d = self.cfg.d_model;
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *OFF.get_or_init(|| std::env::var("INFERO_FUSE_RESIDUAL").as_deref() == Ok("0")) {
            return false;
        }
        // The sparse FFN's own matmuls are the integer mat-vec and the
        // per-expert GEMM, neither of which reads an f16 activation, so there is
        // no consumer for the fused add-and-norm to write one for.
        let Some(f) = &l.dense else { return false };
        self.use_mmq
            && n > 1
            && n <= infero_kernels::MMQ_MAX_TOKENS
            && Kernels::mmq_f16_variant_for(f.w_gate.ty).is_some()
            && [&f.w_gate, &f.w_up].iter().all(|w| {
                matches!(
                    w.ty,
                    infero_kernels::WeightType::Q4G128 | infero_kernels::WeightType::Q4G128T
                ) && w.k == d
            })
    }

    fn feed_forward(&mut self, layer: usize, n: usize, slot: Option<usize>) -> Result<()> {
        if self.w.layers[layer].moe.is_some() {
            return self.feed_forward_moe(layer, n, slot);
        }
        let stage = slot.map(|s| &self.offload.as_ref().unwrap().stage[s]);
        let cfg = &self.cfg;
        let (d, d_ff) = (cfg.d_model, cfg.d_ff);
        let l = &self.w.layers[layer];

        // `gate` and `up` share the normalized residual, as `q`/`k`/`v` do.
        let want_q = self.use_mmvq
            && [&l.dense().w_gate, &l.dense().w_up]
                .iter()
                .all(|w| Kernels::has_mmvq(w.ty) && w.k == d);
        let want_h = self.use_mmq
            && n > 1
            && n <= infero_kernels::MMQ_MAX_TOKENS
            && Kernels::mmq_f16_variant_for(l.dense().w_gate.ty).is_some()
            && [&l.dense().w_gate, &l.dense().w_up].iter().all(|w| {
                matches!(
                    w.ty,
                    infero_kernels::WeightType::Q4G128
                        | infero_kernels::WeightType::Q4G128T
                ) && w.k == d
            });
        // When the norm is the f16-writing one it also adds the attention
        // residual, which `attention` left in `proj` for it. See
        // `ffn_norm_takes_residual`.
        let (shared, shared_f16) = if want_h {
            self.kern.add_rms_norm_f16(
                &mut self.act.xb.slice_mut(..n * d),
                Some(&mut self.scratch.x16.slice_mut(..n * d)),
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
        // `want_h` is the AWQ mmq case specifically; see the matching comment
        // on the attention side. FP8 dispatches inside `matmul_pre` on its
        // own and falls back past its own batching limit there, so the fused
        // buffer is worth taking at any `n`.
        let stacked = l.dense().w_gate_up.as_ref().filter(|w| {
            n > 1 && (want_h || w.ty == infero_kernels::WeightType::F8E4M3)
        });
        // Whether `down` will read its activation as f16, which is the only
        // case where writing that copy early is worth anything. Mirrors what
        // `matmul_pre` decides for itself; claiming it when the buffer was not
        // written would hand the GEMM a stale one.
        let ffn_f16 = stacked.is_some()
            && self.use_mmq
            && n > 1
            && n <= infero_kernels::MMQ_MAX_TOKENS
            && matches!(
                l.dense().w_down.ty,
                infero_kernels::WeightType::Q4G128 | infero_kernels::WeightType::Q4G128T
            )
            && Kernels::mmq_f16_variant_for_shape(l.dense().w_down.ty, l.dense().w_down.n).is_some()
            && Self::mmq_shape_ok(&l.dense().w_down);
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
        } else if Self::fusable(&[&l.dense().w_gate, &l.dense().w_up], shared, n, self.use_mmvq) {
            let bytes = Kernels::q8_1_bytes(d);
            let (gate, up) = (&mut self.act.gate, &mut self.act.up);
            self.kern.mmvq_fused2(
                &mut gate.slice_mut(..d_ff),
                &mut up.slice_mut(..d_ff),
                &l.dense().w_gate.view(stage)?,
                &l.dense().w_up.view(stage)?,
                l.dense().w_gate.ty,
                &self.scratch.q8_1.slice(..bytes),
                d,
                l.dense().w_gate.n,
                l.dense().w_up.n,
            )?;
        } else {
            // Gate and up read the same activation, and past `gemm_threshold`
            // each would otherwise run its own `to_f16` over it -- the second
            // one entirely redundant, since `matmul_pre`'s generic GEMM branch
            // writes the whole conversion into the same `scratch.x16` either
            // call would use. Converting it once here and marking both calls
            // `pre_f16` saves that second launch; gated on `WeightType::Q4K`
            // specifically because that is the shape whose GEMM branch this
            // targets, and on `gemm_threshold` because below it neither call
            // reaches `to_f16` at all.
            let gemm_shared = n > gemm_threshold()
                && l.dense().w_gate.ty == infero_kernels::WeightType::Q4K
                && l.dense().w_up.ty == infero_kernels::WeightType::Q4K;
            if gemm_shared {
                self.kern.to_f16(
                    &mut self.scratch.x16.slice_mut(..n * d),
                    &self.act.xb.slice(..n * d),
                    n * d,
                )?;
            }
            Self::matmul_pre(
                &self.kern,
                &mut self.scratch,
                &mut self.act.gate.slice_mut(..n * d_ff),
                &l.dense().w_gate,
                stage,
                &self.act.xb.slice(..n * d),
                n,
                self.use_mmvq,
                self.use_mmq,
                shared,
                shared_f16 || gemm_shared,
            )?;
            Self::matmul_pre(
                &self.kern,
                &mut self.scratch,
                &mut self.act.up.slice_mut(..n * d_ff),
                &l.dense().w_up,
                stage,
                &self.act.xb.slice(..n * d),
                n,
                self.use_mmvq,
                self.use_mmq,
                shared,
                shared_f16 || gemm_shared,
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
        if Self::residual_fusable(&l.dense().w_down, n, self.use_mmvq) {
            let bytes = Kernels::q8_1_bytes(d_ff);
            self.kern.quantize_q8_1(
                &mut self.scratch.q8_1.slice_mut(..bytes),
                &self.act.ffn.slice(..d_ff),
                d_ff,
            )?;
            self.kern.mmvq_add(
                &mut self.act.x.slice_mut(..d),
                &l.dense().w_down.view(stage)?,
                l.dense().w_down.ty,
                &self.scratch.q8_1.slice(..bytes),
                d_ff,
                l.dense().w_down.n,
            )?;
            return Ok(());
        }

        Self::matmul_pre(
            &self.kern,
            &mut self.scratch,
            &mut self.act.proj.slice_mut(..n * d),
            &l.dense().w_down,
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
        probe(&self.kern, layer, "after_ffn", &self.act.x.slice(..self.cfg.d_model));
        Ok(())
    }

    /// The sparse FFN: route, run the selected experts, combine.
    ///
    /// Same contract as [`Self::feed_forward`] — the result lands in `proj` and
    /// the residual add is either the next block's fused norm or the
    /// `add_assign` at the bottom. The dense path's two optimizations are absent
    /// on purpose: there is no stacked `gate_up` because the checkpoint ships
    /// experts separately, and no residual-fusing `down` because the combine,
    /// not a matmul, is what writes the output.
    #[cfg(not(feature = "cuda"))]
    fn feed_forward_moe(&mut self, _layer: usize, _n: usize, _slot: Option<usize>) -> Result<()> {
        anyhow::bail!("this checkpoint has sparse (MoE) layers, which this backend has no kernels for yet")
    }

    #[cfg(feature = "cuda")]
    fn feed_forward_moe(&mut self, layer: usize, n: usize, slot: Option<usize>) -> Result<()> {
        let stage = slot.map(|s| &self.offload.as_ref().unwrap().stage[s]);
        let d = self.cfg.d_model;
        let m = self
            .cfg
            .moe
            .clone()
            .context("a sparse layer on a model whose config is not sparse")?;
        let (k_act, d_ff) = (m.n_active, m.d_ff_expert);

        // The normalized residual, without the f16 copy: nothing downstream of
        // here reads one. `ffn_norm_takes_residual` returns false on a sparse
        // layer for the same reason, so `x` still carries the attention
        // residual and this only normalizes it.
        Self::norm_for_group(
            &self.kern,
            &mut self.scratch,
            &mut self.act.xb,
            &self.act.x.slice(..n * d),
            &self.w.layers[layer].ffn_norm.as_view(),
            n,
            d,
            self.cfg.rms_eps,
            false,
            false,
        )?;

        // The router is f16 and 128 rows wide, so it goes through the same
        // matmul the dense projections use rather than getting a kernel of its
        // own.
        {
            let w = &self.w.layers[layer].moe.as_ref().unwrap().router;
            let logits = &mut self.act.moe.as_mut().unwrap().router_logits;
            Self::matmul_pre(
                &self.kern,
                &mut self.scratch,
                &mut logits.slice_mut(..n * m.n_experts),
                w,
                stage,
                &self.act.xb.slice(..n * d),
                n,
                self.use_mmvq,
                self.use_mmq,
                None,
                false,
            )?;
        }
        {
            let a = self.act.moe.as_mut().unwrap();
            self.kern.moe_topk(
                &mut a.ids.slice_mut(..n * k_act),
                &mut a.weights.slice_mut(..n * k_act),
                &a.router_logits.slice(..n * m.n_experts),
                m.n_experts,
                k_act,
                n,
                m.norm_topk_prob,
            )?;
        }

        // One row per (token, active expert), in that order — which is the
        // layout `moe_combine` reduces, so nothing has to be grouped by expert
        // and scattered back afterwards. The launch count is the same at one
        // token as at two hundred; what grows is the rows each launch covers.
        let slots = n * k_act;

        // Every activation row in one launch: `quantize_q8_1` walks a flat array
        // in blocks of 32 and both widths are multiples of it, so `n` rows of
        // `d` land as `n` rows of `d / 32` blocks — which is what `y_group`
        // indexes into.
        let xb_bytes = Kernels::q8_1_bytes(d) * n;
        self.kern.quantize_q8_1(
            &mut self.scratch.q8_1.slice_mut(..xb_bytes),
            &self.act.xb.slice(..n * d),
            n * d,
        )?;
        {
            let e = self.w.layers[layer].moe.as_ref().unwrap();
            let a = self.act.moe.as_mut().unwrap();
            for (w, out) in [(&e.gate, &mut a.gate), (&e.up, &mut a.up)] {
                self.kern.mmvq_moe(
                    &mut out.slice_mut(..slots * d_ff),
                    &w.view(stage)?,
                    w.ty,
                    &a.ids.slice(..slots),
                    &self.scratch.q8_1.slice(..xb_bytes),
                    d,
                    d_ff,
                    slots,
                    w.stride,
                    // A token's `k` slots all read that token's residual row.
                    k_act,
                )?;
            }
            self.kern.silu_mul(
                &mut a.hidden.slice_mut(..slots * d_ff),
                &a.gate.slice(..slots * d_ff),
                &a.up.slice(..slots * d_ff),
                slots * d_ff,
            )?;
        }

        {
            let hidden_bytes = Kernels::q8_1_bytes(d_ff) * slots;
            self.kern.quantize_q8_1(
                &mut self.scratch.q8_1.slice_mut(..hidden_bytes),
                &self.act.moe.as_ref().unwrap().hidden.slice(..slots * d_ff),
                slots * d_ff,
            )?;
            let e = self.w.layers[layer].moe.as_ref().unwrap();
            let a = self.act.moe.as_mut().unwrap();
            self.kern.mmvq_moe(
                &mut a.down.slice_mut(..slots * d),
                &e.down.view(stage)?,
                e.down.ty,
                &a.ids.slice(..slots),
                &self.scratch.q8_1.slice(..hidden_bytes),
                d_ff,
                d,
                slots,
                e.down.stride,
                // `down` reads each slot's own SwiGLU product.
                1,
            )?;
        }

        {
            let a = self.act.moe.as_ref().unwrap();
            self.kern.moe_combine(
                &mut self.act.proj.slice_mut(..n * d),
                &a.down.slice(..slots * d),
                &a.weights.slice(..slots),
                d,
                k_act,
                n,
            )?;
        }
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
        act_xb: &mut Buf<f32>,
        x: &View<'_, f32>,
        weight: &View<'_, f32>,
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
                Some(&mut scratch.x16.slice_mut(..n_tokens * d)),
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
        // work inside the single block that computed the norm. `INFERO_NO_FUSED_NORM`
        // exists to measure which way that trade actually falls.
        static SEPARATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let separate =
            *SEPARATE.get_or_init(|| std::env::var_os("INFERO_NO_FUSED_NORM").is_some());
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
            infero_kernels::WeightType::Q4K
            | infero_kernels::WeightType::Q6K
            | infero_kernels::WeightType::Q4G128
            | infero_kernels::WeightType::Q4G128T => w.k.is_multiple_of(256),
            _ => w.k.is_multiple_of(32),
        }
    }

    /// `out[t, :] = w · x[t, :]`, picking the integer mat-vec, the float one,
    /// or cuBLAS by batch size.
    #[allow(clippy::too_many_arguments)]
    fn matmul(
        kern: &Kernels,
        scratch: &mut Scratch,
        out: &mut ViewMut<'_, f32>,
        w: &Matrix,
        stage: Option<&Buf<u8>>,
        x: &View<'_, f32>,
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
        out: &mut ViewMut<'_, f32>,
        w: &Matrix,
        stage: Option<&Buf<u8>>,
        x: &View<'_, f32>,
        n_tokens: usize,
        use_mmvq: bool,
        use_mmq: bool,
        pre_quantized: Option<usize>,
        pre_f16: bool,
    ) -> Result<()> {
        let weights = w.view(stage)?;

        // Block-scaled FP8 has its own pair of paths and neither is the integer
        // one below: the activation stays f32, because the weights carry a
        // per-block scale rather than a per-block quantization of the *input*.
        //
        // At one token this is the whole point of storing FP8 at all. With the
        // weights expanded to f16 at load, a batch-1 projection went through
        // cuBLAS's GEMM with m = 1 — the profiler had `gemm_f16` at 75% of a
        // step at 86.8 us a launch — so the mat-vec replaces both the byte count
        // and the wrong kernel shape.
        if w.ty == infero_kernels::WeightType::F8E4M3 {
            // Unified layout (`INFERO_FP8_UNIFIED=1` under the `cutlass`
            // feature): this matrix's device buffer is plain `[n,k]`
            // row-major, not `fp8::ROW_GROUP`-interleaved -- so none of
            // `mma_e4m3_block`/`mma_f8_block`/`mmv_f8_block*` below can read
            // it correctly, only `mmv_f8_plain` and CUTLASS can. Handled
            // completely separately rather than folded into the chain below
            // so that invariant stays checkable by inspection: nothing after
            // this block runs for a matrix loaded this way.
            #[cfg(feature = "cutlass")]
            if crate::weights::fp8_unified_layout() {
                if n_tokens == 1 {
                    kern.mmv_f8_plain(out, &weights, x, w.k, w.n, false)?;
                    return Ok(());
                }
                // `quantize_act_e4m3_cutlass` writes the activation scale
                // straight into the transposed `[scale_cols, n_tokens]`
                // layout `mma_e4m3_cutlass_sfa_f32out`'s CUTLASS kernel
                // wants, folding what used to be a separate
                // `transpose_pad_scale_a_f32` pass (`cutlass_transpose_sfa`
                // in a profile) into this quantizer's existing per-group
                // reduction -- free here because, unlike the AWQ/non-unified
                // path below, nothing else in this branch reads `xs` in its
                // natural `[n_tokens, scale_cols]` layout. No row padding
                // (`m_pad = n_tokens`): CUTLASS's `can_implement`/correctness
                // were verified to accept any M, not just multiples of 128
                // (see project-infero-perf-gap memory's CUTLASS-M-alignment
                // entry). `_f32out` writes `out` directly from CUTLASS's own
                // epilogue -- no bf16 scratch, no separate upconvert kernel
                // after it returns -- measured 1.1-1.3x faster than the
                // bf16-scratch path at this shape/batch (same memory entry).
                let scale_cols = w.k.div_ceil(infero_kernels::fp8::ACT_QUANT_GROUP);
                let xq_len = n_tokens * w.k;
                let sfa_len = scale_cols * n_tokens;
                anyhow::ensure!(
                    scratch.xq_e4m3.len() >= xq_len && scratch.xs_e4m3.len() >= sfa_len,
                    "activation quant scratch too small for {n_tokens} tokens at k={}",
                    w.k
                );
                kern.quantize_act_e4m3_cutlass(
                    &mut scratch.xq_e4m3.slice_mut(..xq_len),
                    &mut scratch.xs_e4m3.slice_mut(..sfa_len),
                    x,
                    w.k,
                    n_tokens,
                    n_tokens,
                )?;
                let cw = w
                    .cutlass_weight(kern)
                    .with_context(|| format!("preparing CUTLASS weight for a {}x{} matrix", w.n, w.k))?;
                let ran = kern.mma_e4m3_cutlass_sfa_f32out(
                    out,
                    &weights,
                    cw,
                    &scratch.xq_e4m3.slice(..xq_len),
                    &scratch.xs_e4m3.slice(..sfa_len),
                    w.k,
                    w.n,
                    n_tokens,
                    false,
                )?;
                anyhow::ensure!(
                    ran,
                    "CUTLASS declined a {}x{} matmul at {n_tokens} tokens under the unified FP8 \
                     layout, which has no other kernel that can read it -- likely k or n not a \
                     multiple of 128",
                    w.n,
                    w.k
                );
                return Ok(());
            }
            // A handful of tokens reads each weight once and spends it on all of
            // them, which is what batching is for. The expansion path below
            // costs five bytes a weight — one read, two written, two read back —
            // against resident f16's two, so taking it at a few tokens made
            // batched decode *slower* than before FP8: the profiler had
            // `dequant_f8_block` at 67% of a batch-32 step and batch scaling
            // down from 36.9x to 8.6x.
            // Tensor cores first, at every token count including one. The
            // scalar mat-vec's inner loop issues sixteen FMA instructions per
            // chunk per token and lands at a seventh of the f32 FMA bound; one
            // `mma.m16n8k16` does the same 2048 MACs in one instruction. On the
            // 27B's widest projection, milliseconds, against a pure weight-load
            // floor of 0.061:
            //
            //   tokens   scalar     mma
            //        1    0.073   0.063
            //        2    0.082   0.064
            //        3    0.100   0.063
            //        4    0.106   0.064
            //        8    0.174   0.067
            //
            // Flat in tokens, because eight is the fragment's own N — and ahead
            // even at one token, where it wastes seven of those eight columns.
            // W8A8 first, at more than one token: two native e4m3 tensor-core
            // operands instead of widening the weight to f16 for `mma_f16`,
            // doubling the throughput `mma_f8_block` gets out of the same
            // instruction issue rate. Declines by itself below `K_TILE = 256`
            // or off this GPU's tensor cores (`caps().fp8`); the extra
            // activation-quantize launch is not worth it at one token, where
            // `mma_f8_block` already sits near the weight-load floor.
            if n_tokens >= 2 {
                let scale_cols = w.k.div_ceil(infero_kernels::fp8::ACT_QUANT_GROUP);
                let xq_len = n_tokens * w.k;
                let xs_len = n_tokens * scale_cols;
                if scratch.xq_e4m3.len() >= xq_len && scratch.xs_e4m3.len() >= xs_len {
                    kern.quantize_act_e4m3(
                        &mut scratch.xq_e4m3.slice_mut(..xq_len),
                        &mut scratch.xs_e4m3.slice_mut(..xs_len),
                        x,
                        w.k,
                        n_tokens,
                    )?;
                    // The AOT CUTLASS SM120 GEMM: ~5-8x `mma_e4m3_block`'s
                    // measured TFLOPS on this model's FFN shapes, but only
                    // past the token count where its fixed per-call
                    // overhead (padding, workspace, activation-scale
                    // transpose) stops dominating -- a prefill-batch lever,
                    // not a decode one, same reasoning as `attn_prefill`'s
                    // `MIN_PREFILL_RUN`. Opt-in behind `INFERO_FFN_CUTLASS`
                    // (a token-count threshold) while this path is new.
                    #[cfg(feature = "cutlass")]
                    if ffn_cutlass_min_tokens().is_some_and(|min| n_tokens >= min)
                        && let Some(cw) = w.cutlass_weight(kern)
                        && kern.mma_e4m3_cutlass(
                            out,
                            &weights,
                            cw,
                            &scratch.xq_e4m3.slice(..xq_len),
                            &scratch.xs_e4m3.slice(..xs_len),
                            w.k,
                            w.n,
                            n_tokens,
                            false,
                        )?
                    {
                        return Ok(());
                    }
                    if kern.mma_e4m3_block(
                        out,
                        &weights,
                        &scratch.xq_e4m3.slice(..xq_len),
                        &scratch.xs_e4m3.slice(..xs_len),
                        w.k,
                        w.n,
                        n_tokens,
                        false,
                    )? {
                        return Ok(());
                    }
                }
            }
            if kern.mma_f8_block(out, &weights, x, w.k, w.n, n_tokens, false)? {
                return Ok(());
            }
            if n_tokens >= 2
                && kern.mmv_f8_block_batch(out, &weights, x, w.k, w.n, n_tokens, false)?
            {
                return Ok(());
            }
            if n_tokens == 1 {
                // `accum` is not offered here. The callers that want the fused
                // residual add pass it through a separate argument on the
                // quantized mat-vecs, and `matmul_pre` has no channel for it —
                // so this path writes and the residual add stays its own
                // launch. Wiring it through is a later change with its own
                // measurement, not something to infer from `pre_quantized`.
                return kern.mmv_f8_block(out, &weights, x, w.k, w.n, false);
            }
            // Above one token the answer is a GEMM. Expand the weights into the
            // f16 staging buffer the float path already uses, and convert the
            // activation the same way that path does.
            // Warn once, because reaching here is a performance bug and the
            // profiler cannot say so: it reports kernels, not shapes, and the
            // last time `dequant_f8_block` showed up at 153 us a launch it took
            // three wrong guesses to find out the caller was prefill.
            {
                use std::sync::OnceLock;
                static SEEN: OnceLock<()> = OnceLock::new();
                if SEEN.set(()).is_ok() {
                    tracing::warn!(
                        k = w.k,
                        n = w.n,
                        n_tokens,
                        k_mod_128 = w.k % 128,
                        "FP8 expansion path taken"
                    );
                }
            }
            let n_x = n_tokens * w.k;
            let elems = w.elements();
            anyhow::ensure!(
                scratch.w16.len() >= elems,
                "the f16 staging buffer holds {} halves, this projection needs {elems}",
                scratch.w16.len()
            );
            if !pre_f16 {
                kern.to_f16(&mut scratch.x16.slice_mut(..n_x), x, n_x)?;
            }
            kern.dequant_f8_block_to_f16(
                &mut scratch.w16.slice_mut(..elems),
                &weights,
                w.k,
                w.n,
            )?;
            return kern.gemm_f16(
                out,
                &scratch.x16.slice(..n_x),
                &scratch.w16.slice(..elems),
                n_tokens,
                w.k,
                w.n,
            );
        }

        let int_x = use_mmvq && Kernels::has_mmvq(w.ty) && w.k.is_multiple_of(32);
        // Whether *this matrix* gets the tensor-core GEMM, not just its type:
        // Q4_K rows that are not a multiple of 256 have the type but not the
        // shape, and they should fall to the mat-vec rather than the float path.
        let mmq_ok = use_mmq
            && Kernels::has_mmq(w.ty)
            && kern.device().caps().int_tensor_gemm
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

        // A handful of tokens is the awkward middle: the tensor-core GEMM pays
        // for a full 16-token tile whatever it is given, while the mat-vec can
        // stream the weights once and spend them on a handful of tokens without
        // staging anything through shared memory. Measured on a 31.5 MiB Q4_K
        // projection: at two tokens 120 us against the GEMM's 182, at four they
        // are level, and by eight the GEMM is ahead 222 to 368 -- so `mmq` takes
        // over there for that type.
        //
        // Q8_0 does not cross over at the same point: a speculative verify pass
        // at `k=4` (5 rows) on a real Q8_0 GGUF measured `mmq` at 155.2 us a
        // launch against `mmvq_batch`'s 47.9 us at two rows on the same
        // projections -- more than 3x, not "level". `mmvqt16_*` is the widest
        // template `Kernels::mmvq_t` picks from, so 16 is the natural ceiling
        // for how far this range can go without a new kernel; Q8_0 gets it and
        // everything else keeps the narrower, separately-measured one until it
        // is measured too.
        let mmvq_batch_max = if w.ty == infero_kernels::WeightType::Q8_0 { 16 } else { 3 };
        if int_x && (2..=mmvq_batch_max).contains(&n_tokens) {
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
        if mmq_ok && n_tokens > 1 && n_tokens <= infero_kernels::MMQ_MAX_TOKENS {
            // The f16-operand kernels take activations unquantized, so they
            // need a different buffer and a different launcher. Off unless
            // `INFERO_MMQ_VARIANT` names one; the default path below is
            // untouched.
            //
            // This re-converts per matmul where the Q8_1 path can hand the
            // same quantized activation to several projections, so the number
            // it measures is if anything pessimistic.
            if matches!(
                w.ty,
                infero_kernels::WeightType::Q4G128 | infero_kernels::WeightType::Q4G128T
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

        // Q4_K gets its own, much higher ceiling on Metal: gemv_mma_shared
        // (see its doc comment) turned the MMA path from a loss against
        // GEMM into a clear win, so raising the crossover point specifically
        // for this weight type is worth it in a way it was not before that
        // kernel existed. Measured end to end (real 27B, real prompts,
        // INFERO_GEMM_THRESHOLD's own methodology): a 20-120 token prompt's
        // queued_ms is 33-50% lower through this path than through GEMM at
        // the same size. No measurement past 120 yet, so 200 is a margin on
        // top of what is actually verified, not a re-measured crossing.
        // INFERO_Q4K_MMA_MAX overrides it for finding the real one.
        const Q4K_MMA_MAX_DEFAULT: usize = 200;
        let q4k_mma_max: usize = std::env::var("INFERO_Q4K_MMA_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| *v > 0)
            .unwrap_or(Q4K_MMA_MAX_DEFAULT);
        // Q8_0 gets the same treatment for the same reason: `gemv_mma_q8_0`
        // beats both the scalar `gemv_q8_0` (which loses to GEMM itself past
        // sixteen tokens, the same measurement `gemm_threshold()` is based
        // on) and MPS's own GEMM from eight tokens up, by 1.2-3.4x depending
        // on token count (`gemv_q8_0_threshold_check.rs`). It crosses back
        // to GEMM winning somewhere between 90 and 128 (0.93-1.17x there),
        // so 100 rather than Q4_K's 200 -- this is every GDN and attention
        // projection in a GGUF checkpoint, not one weight type among several,
        // so erring low costs more of them if wrong. INFERO_Q8_0_MMA_MAX
        // overrides it for re-measuring the real crossing.
        const Q8_0_MMA_MAX_DEFAULT: usize = 100;
        let q8_0_mma_max: usize = std::env::var("INFERO_Q8_0_MMA_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| *v > 0)
            .unwrap_or(Q8_0_MMA_MAX_DEFAULT);
        let use_gemv = if !cfg!(feature = "cuda") && w.ty == infero_kernels::WeightType::Q4K {
            n_tokens <= q4k_mma_max
        } else if !cfg!(feature = "cuda") && w.ty == infero_kernels::WeightType::Q8_0 {
            n_tokens <= q8_0_mma_max
        } else {
            n_tokens <= gemm_threshold()
        };
        if use_gemv {
            return kern.gemv(out, &weights, w.ty, x, w.k, w.n, n_tokens);
        }

        let n_x = n_tokens * w.k;
        // Unlike the FP8 branch above, this one used to convert unconditionally
        // -- so a caller passing `pre_f16` to say "I already wrote `x16`" (gate
        // and up read the same activation) was ignored, and gate's own
        // `to_f16` clobbered what it just wrote before up ever read it.
        if !pre_f16 {
            kern.to_f16(&mut scratch.x16.slice_mut(..n_x), x, n_x)?;
        }

        if w.ty == infero_kernels::WeightType::F16 {
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

/// `INFERO_PROBE=<layer>` reports the RMS of named intermediates inside that
/// block. Enough to bisect a block against a second implementation of the same
/// forward pass -- which is the only way the last two composition bugs were
/// found, and the only tool that would have found them faster.
fn probe(kern: &Kernels, layer: usize, name: &'static str, v: &View<'_, f32>) {
    static WANT: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let want = WANT.get_or_init(|| {
        std::env::var("INFERO_PROBE").ok().and_then(|v| v.parse().ok())
    });
    if *want != Some(layer) {
        return;
    }
    let Ok(row) = kern.device().stream().clone_dtoh(v) else { return };
    let _ = kern.device().synchronize();
    let n = row.len().max(1);
    let rms = (row.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
    tracing::info!(layer, name, rms, len = row.len(), first = row.first().copied(), "probe");
    // `INFERO_PROBE_DUMP=<dir>` also writes the raw f32, so two implementations
    // of the same forward pass can be diffed element by element rather than
    // through summary statistics. RMS and element zero can both agree while the
    // vectors differ, which is how this cost an hour.
    if let Ok(dir) = std::env::var("INFERO_PROBE_DUMP") {
        let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = std::fs::write(format!("{dir}/eng.{name}.f32"), bytes);
    }
}

impl Activations {

    fn new(
        dev: &Device,
        cfg: &Config,
        max_seq: usize,
        max_logit_rows: usize,
        needs_score_buffer: bool,
        chunk: usize,
    ) -> Result<Self> {
        let stream = dev.stream();
        let d = cfg.d_model;
        // The attention interior can be wider than the residual stream — 24
        // heads of 256 against 5120 on Qwen3.8 — so `q` and the attention
        // output are sized by it rather than by `d_model`. They were the same
        // number for every model before that one.
        let d_attn = cfg.d_attn();
        let kv_dim = cfg.d_kv();

        let alloc_f32 = |n: usize, what: &str| -> Result<Buf<f32>> {
            stream
                .alloc_zeros::<f32>(n)
                .with_context(|| format!("allocating {what} ({} MiB)", (n * 4) >> 20))
        };

        Ok(Self {
            x: alloc_f32(chunk * d, "residual")?,
            xb: alloc_f32(chunk * d, "normalized")?,
            q: alloc_f32(chunk * d_attn, "queries")?,
            k: alloc_f32(chunk * kv_dim, "keys")?,
            v: alloc_f32(chunk * kv_dim, "values")?,
            attn: alloc_f32(chunk * d_attn, "attention output")?,
            proj: alloc_f32(chunk * d.max(1), "projection")?,
            // Twice `d_ff`: under `INFERO_FUSE_FFN` one matmul writes `gate` and
            // `up` into a single row of this, and `silu_mul_split` reads the
            // two halves. The unfused path uses the first half only.
            //
            // It doubles as the packed `[q | k | v]` staging row, which is
            // `d_attn + 2·d_kv` wide. That has always been the smaller of the
            // two and still is on every model here, but it stopped being
            // obviously so once `d_attn` came loose from `d_model` — so take
            // the max rather than relying on it.
            gate: alloc_f32(
                chunk * (cfg.d_ff * 2).max(d_attn + 2 * kv_dim),
                "ffn gate / packed qkv",
            )?,
            up: alloc_f32(chunk * cfg.d_ff, "ffn up")?,
            ffn: alloc_f32(chunk * cfg.d_ff, "ffn hidden")?,
            // A fused attention kernel (see `needs_score_buffer`) never reads
            // or writes this, so it costs nothing to leave it at its minimum
            // rather than the `max_seq`-wide buffer the unfused path needs.
            scores: alloc_f32(
                cfg.n_heads * chunk * if needs_score_buffer { max_seq } else { 1 },
                "attention scores",
            )?,
            logits: alloc_f32(max_logit_rows * cfg.vocab_size, "logits")?,
            token_ids: stream.alloc_zeros::<i32>(chunk)?,
            positions: stream.alloc_zeros::<i32>(chunk)?,
            mrope_positions: stream.alloc_zeros::<i32>(chunk * 3)?,
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
            attn_gate: if cfg.attn_output_gate {
                Some(alloc_f32(chunk * d_attn, "attention output gate")?)
            } else {
                None
            },
            gdn: match cfg.linear_attn {
                Some(la) => Some(GdnActs {
                    qz: alloc_f32(
                        chunk * (la.conv_channels() + la.value_dim()),
                        "gdn qkv+z fused",
                    )?,
                    qkv: alloc_f32(chunk * la.conv_channels(), "gdn qkv")?,
                    qkv_conv: alloc_f32(chunk * la.conv_channels(), "gdn qkv post-conv")?,
                    z: alloc_f32(chunk * la.value_dim(), "gdn gate")?,
                    a: alloc_f32(chunk * la.value_heads, "gdn a")?,
                    b: alloc_f32(chunk * la.value_heads, "gdn b")?,
                    ab: alloc_f32(2 * chunk * la.value_heads, "gdn ab")?,
                    beta: alloc_f32(chunk * la.value_heads, "gdn beta")?,
                    g: alloc_f32(chunk * la.value_heads, "gdn g")?,
                    core: alloc_f32(chunk * la.value_dim(), "gdn core")?,
                }),
                None => None,
            },
            moe: match &cfg.moe {
                Some(m) => {
                    let rows = chunk * m.n_active;
                    Some(MoeActs {
                        router_logits: alloc_f32(chunk * m.n_experts, "moe router logits")?,
                        ids: stream.alloc_zeros::<i32>(rows)?,
                        weights: alloc_f32(rows, "moe combine weights")?,
                        gate: alloc_f32(rows * m.d_ff_expert, "moe gate")?,
                        up: alloc_f32(rows * m.d_ff_expert, "moe up")?,
                        hidden: alloc_f32(rows * m.d_ff_expert, "moe hidden")?,
                        down: alloc_f32(rows * d, "moe down")?,
                    })
                }
                None => None,
            },
        })
    }
}

impl Model {
    /// Load `model.visual.*` and size the scratch for `max_patches`.
    ///
    /// Returns false when the checkpoint has no tower, which is not an error.
    ///
    /// `max_patches` bounds one call, not one conversation: the scratch is about
    /// 85 KB a patch, so a 1024x1024 image is 4096 patches and 350 MB. Frames
    /// are independent attention segments, so a caller with more work than this
    /// splits on a frame boundary and changes nothing about the result — which is
    /// why this is a per-call bound rather than a limit on what can be served.
    pub fn load_vision_tower(
        &mut self,
        dir: impl AsRef<std::path::Path>,
        max_patches: usize,
    ) -> Result<bool> {
        let shards = infero_safetensors::Shards::open_dir(dir.as_ref())?;
        let Some(tower) = weights::load_vision(&self.dev, &shards, &self.cfg)? else {
            return Ok(false);
        };
        let scratch = infero_kernels::vision::VisionScratch::new(&self.dev, &tower.shape, max_patches)?;
        tracing::info!(
            max_patches,
            scratch_mib = (max_patches * 85) >> 10,
            "vision scratch allocated"
        );
        self.vision = Some(tower);
        self.vision_scratch = Some(scratch);
        Ok(true)
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    /// The tower's dimensions, for a caller sizing images to it.
    pub fn vision_shape(&self) -> Option<&infero_kernels::vision::VisionShape> {
        self.vision.as_ref().map(|t| &t.shape)
    }

    /// The placeholder ids a prompt uses to reserve room for vision output.
    pub fn vision_tokens(&self) -> Option<(u32, u32)> {
        self.vision.as_ref().map(|t| (t.cfg.image_token, t.cfg.video_token))
    }
}

/// One image's merger output, `[tokens, out_hidden]`, on the device.
///
/// Owned and detached from the scratch it was computed in, so that encoding a
/// second image does not overwrite the first — the scratch is a per-call buffer
/// and features have to outlive the call to reach the forward pass.
pub struct VisionFeatures {
    rows: Buf<f32>,
    /// How many language-model tokens this image occupies, `patches / 4`.
    pub tokens: usize,
    pub out_hidden: usize,
    /// The patch grid, kept for the caller building the prompt: the placeholder
    /// run has to be exactly `tokens` long.
    pub grid_h: usize,
    pub grid_w: usize,
    /// Temporal patch groups: 1 for a still image, `frames / 2` for a clip.
    /// `tokens == grid_t * grid_h * grid_w / merge_unit`.
    pub grid_t: usize,
}

impl VisionFeatures {
    pub fn view(&self) -> View<'_, f32> {
        self.rows.as_view()
    }

    /// Rows `[start, start+count)` -- what a chunked prefill step reads when
    /// only part of this clip's placeholder run lands in that step, rather
    /// than the whole clip (`view()`'s job, and still what `count == self
    /// .tokens` here reduces to).
    pub fn rows_view(&self, start: usize, count: usize) -> View<'_, f32> {
        let w = self.out_hidden;
        self.rows.as_view().slice(start * w..(start + count) * w)
    }
}

impl Model {
    /// `VisionDims` reads off the loaded tower's shape and config, the
    /// common setup `vision_resize`/`vision_resize_video` both start from.
    fn vision_dims(&self) -> Result<qwen35_vision::VisionDims> {
        let t = self.vision.as_ref().context("this model has no vision tower")?;
        Ok(qwen35_vision::VisionDims {
            depth: t.shape.depth,
            hidden: t.shape.hidden,
            heads: t.shape.heads,
            intermediate: t.shape.intermediate,
            out_hidden: t.shape.out_hidden,
            in_channels: t.shape.in_channels,
            patch: t.shape.patch,
            temporal_patch: t.shape.temporal_patch,
            merge: t.shape.merge,
            num_position_embeddings: t.cfg.position_embeddings,
            eps: t.shape.eps,
            rope_theta: t.shape.rope_theta,
        })
    }

    /// The resize target this tower wants for a `src_h x src_w` image.
    ///
    /// Exposed because the caller has to build the prompt's placeholder run
    /// before the image is encoded, and the run's length is decided here.
    pub fn vision_resize(&self, src_h: usize, src_w: usize, max_patches: usize) -> Result<(usize, usize, usize)> {
        let dims = self.vision_dims()?;
        // `min_pixels` is one merge block's worth and `max_pixels` is the
        // caller's patch budget, both in pixels because that is the unit
        // `smart_resize` compares against. `None` means an aspect ratio past
        // 200:1, which the reference also refuses.
        let (h, w) = qwen35_vision::smart_resize(
            src_h,
            src_w,
            dims.resize_factor(),
            dims.patch * dims.patch * dims.merge_unit(),
            max_patches * dims.patch * dims.patch,
        )
        .with_context(|| {
            format!("a {src_h}x{src_w} image is past the 200:1 aspect ratio the \
                     processor accepts")
        })?;
        let tokens = (h / dims.patch) * (w / dims.patch) / dims.merge_unit();
        Ok((h, w, tokens))
    }

    /// [`Self::vision_resize`] for a `frames`-frame clip: the same target
    /// size every frame shares, plus how many temporal-patch groups they
    /// fold into and the total placeholder token count across all of them.
    ///
    /// The one real difference from a single image: `max_patches` bounds the
    /// *whole clip's* patches, not one frame-group's, so the pixel budget
    /// `smart_resize` sizes against is `max_patches` divided by `grid_t` --
    /// matching the reference video processor's own `t_bar * h_bar * w_bar`
    /// comparison, which is this same "the budget is per clip, not per
    /// frame-group" idea expressed with a third axis this codebase's 2-D
    /// `smart_resize` does not carry. Dividing first and resizing once,
    /// rather than resizing at the full budget and refusing if `grid_t`
    /// copies of it overrun, is what keeps every frame-group the same size —
    /// a per-image resize on each frame independently would still balloon
    /// past `max_patches` for any `grid_t > 1`, which is exactly the bug
    /// this function exists to not have.
    pub fn vision_resize_video(
        &self,
        frames: usize,
        src_h: usize,
        src_w: usize,
        max_patches: usize,
    ) -> Result<(usize, usize, usize, usize)> {
        // The vision tower's compute kernels (`vision_patchify`, `vision_attn`,
        // ...) exist only in `crates/kernels/src/cu/vision.cu` -- there is no
        // `vision.metal` yet, still images and all. A still image already
        // shares that gap silently; video is new work added this session, so
        // it gets the explicit refusal a real dispatch failure deep inside
        // `encode_clip` would not: a clear message here, before any device
        // work, rather than a bare "kernel not found" partway through it.
        #[cfg(not(feature = "cuda"))]
        anyhow::bail!(
            "video input needs the vision tower's CUDA kernels; this build has \
             no vision.metal yet, so video requests are refused outright"
        );
        let dims = self.vision_dims()?;
        let (grid_t, per_group_patches) = qwen35_vision::video_resize_budget(
            frames,
            dims.temporal_patch,
            dims.merge_unit(),
            max_patches,
        )
        .with_context(|| {
            format!(
                "a {frames}-frame clip against a {max_patches}-patch budget: either \
                 there are no frames, or the budget split across the resulting \
                 frame-groups leaves less than one merge block a group"
            )
        })?;
        let (h, w) = qwen35_vision::smart_resize(
            src_h,
            src_w,
            dims.resize_factor(),
            dims.patch * dims.patch * dims.merge_unit(),
            per_group_patches * dims.patch * dims.patch,
        )
        .with_context(|| {
            format!("a {src_h}x{src_w} frame is past the 200:1 aspect ratio the \
                     processor accepts")
        })?;
        let tokens_per_group = (h / dims.patch) * (w / dims.patch) / dims.merge_unit();
        Ok((h, w, grid_t, grid_t * tokens_per_group))
    }

    /// Run the tower over one prepared frame.
    ///
    /// A thin `grid_t = 1` wrapper around [`Self::encode_clip`] — see that
    /// for the whole of the vision path. Kept as its own entry point (rather
    /// than making every image caller build a one-frame `PreparedClip`) so
    /// the image path and its tests are untouched by video's generality.
    pub fn encode_image(
        &mut self,
        frame: &qwen35_vision_image::PreparedFrame,
    ) -> Result<VisionFeatures> {
        let tower = self.vision.as_ref().context("this model has no vision tower")?;
        let temporal_patch = tower.shape.temporal_patch;
        // A still image's two temporal taps see the same pixels -- the
        // processor's `expand`, not a real second frame -- so the one-frame
        // planar buffer is repeated `temporal_patch` times, matching what
        // `vision_patchify`'s `n_frames=1` + `min(t, n_frames-1)` read
        // already did here before this became `encode_clip`'s job instead.
        let clip = qwen35_vision_image::PreparedClip {
            planar: frame.planar.repeat(temporal_patch),
            frames: temporal_patch,
            height: frame.height,
            width: frame.width,
            grid_h: frame.grid_h,
            grid_w: frame.grid_w,
        };
        self.encode_clip(&clip)
    }

    /// Run the tower over a multi-frame clip: patchify, the 27 blocks, the
    /// merger. What comes back is what the prompt's placeholder tokens will be
    /// replaced by, and its `tokens` count is what the placeholder run has to be
    /// as long as — a mismatch is refused at splice time rather than silently
    /// truncated, because it means the grid the tower ran on is not the grid the
    /// prompt was built for.
    pub fn encode_clip(&mut self, clip: &qwen35_vision_image::PreparedClip) -> Result<VisionFeatures> {
        let tower = self.vision.as_ref().context("this model has no vision tower")?;
        let scratch = self
            .vision_scratch
            .as_mut()
            .context("the vision tower is loaded but its scratch is not")?;
        let shape = tower.shape;
        anyhow::ensure!(
            clip.frames.is_multiple_of(shape.temporal_patch),
            "a clip of {} frames does not split evenly into temporal patches of {}",
            clip.frames,
            shape.temporal_patch
        );
        let grid_t = clip.frames / shape.temporal_patch;
        let patches_per_group = clip.grid_h * clip.grid_w;
        let patches = grid_t * patches_per_group;
        let merge_unit = shape.merge * shape.merge;
        anyhow::ensure!(
            patches_per_group.is_multiple_of(merge_unit),
            "a {}x{} patch grid does not group into whole {}x{} blocks",
            clip.grid_h,
            clip.grid_w,
            shape.merge,
            shape.merge
        );
        anyhow::ensure!(
            patches <= scratch.max_patches(),
            "this clip needs {patches} patches ({grid_t} frame-groups of \
             {patches_per_group}), the vision scratch was sized for {}",
            scratch.max_patches()
        );

        // Geometry first, on the host, from the same functions the reference
        // capture pinned: one segment per frame, two position axes, and the
        // learned 48x48 grid resampled to this clip's grid. All already
        // `t`-generic — a still image's `grid_t = 1` is the case they were
        // written for, video's `grid_t > 1` costs them nothing new.
        let grid = qwen35_vision::Grid { t: grid_t, h: clip.grid_h, w: clip.grid_w };
        let grids = [grid];
        let cu = qwen35_vision::cu_seqlens(&grids);
        let pos_ids = qwen35_vision::vision_position_ids(&grids, shape.merge);
        let (idx, wts) =
            qwen35_vision::pos_embed_taps(&grids, tower.cfg.grid_per_side(), shape.merge);
        let geo = infero_kernels::vision::VisionGeometry::new(
            &self.kern, &shape, &cu, &pos_ids, &idx, &wts,
        )?;

        // Patchify on the device: one launch for every temporal-patch group
        // in the clip at once, `vision_patchify`'s own `grid_t` picking each
        // group's frames and output slice out of the same contiguous
        // buffers a `grid_t`-loop of launches used to address by hand (see
        // that function's doc comment, and the CUDA source's). `PreparedClip
        // ::frames` being even by construction (`prepare_clip` pads an odd
        // count) is what makes every group exactly `shape.temporal_patch`
        // frames wide -- there is no shorter last group to special-case.
        let planar = self.dev.stream().clone_htod(&clip.planar)?;
        {
            let (mut rows_f32, mut rows_h) = scratch.patchify_views();
            let total = patches_per_group * shape.patch_dim() * grid_t;
            self.kern.vision_patchify(
                &mut rows_f32.slice_mut(..total),
                &mut rows_h.slice_mut(..total),
                &planar.as_view(),
                shape.temporal_patch,
                clip.height,
                clip.width,
                &shape,
                grid_t,
            )?;
        }

        let w = tower.weights();
        infero_kernels::vision::vision_forward(&self.kern, &shape, &w, &geo, scratch)?;

        // Copy the features out of the scratch, which the next call would
        // overwrite.
        let tokens = patches / merge_unit;
        let mut rows = self
            .dev
            .stream()
            .alloc_zeros::<f32>(tokens * shape.out_hidden)?;
        self.dev.stream().memcpy_dtod(
            &scratch.features().slice(..tokens * shape.out_hidden),
            &mut rows.as_view_mut(),
        )?;
        Ok(VisionFeatures {
            rows,
            tokens,
            out_hidden: shape.out_hidden,
            grid_h: clip.grid_h,
            grid_w: clip.grid_w,
            grid_t,
        })
    }

    /// Which rows of `tokens` are vision placeholders, using **this**
    /// checkpoint's ids.
    ///
    /// Not `infero_kernels::vision::splice_targets`, which hardcodes 248056 and
    /// 248057. Those are right for this checkpoint and the config is what says
    /// so; a loader that checks the config and then splices on a constant has
    /// two sources of truth and only tests one.
    /// Positions within `tokens` (this chunk only) that are an image or video
    /// placeholder id -- however many of them there are, from zero (a chunk
    /// entirely before or after a clip's placeholder run) up to `tokens.len()`
    /// (a chunk entirely inside one). The caller matches this count against
    /// `BatchItem.vision_row_offset` before reading `vision`'s rows; this
    /// function itself has no way to know how many rows the *whole* clip has,
    /// only how many placeholder ids are in front of it right now.
    fn vision_targets(&self, tokens: &[u32]) -> Result<Vec<i32>> {
        let (img, vid) = self.vision_tokens().context("no vision tower")?;
        Ok(tokens
            .iter()
            .enumerate()
            .filter(|&(_, &t)| t == img || t == vid)
            .map(|(i, _)| i as i32)
            .collect())
    }
}
