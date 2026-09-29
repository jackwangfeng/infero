//! Native Rust implementations of the dense-decoder kernel subset, dispatched
//! by name from `infero_cpu::Function::launch` (see that crate's `launch.rs`
//! doc comment for why this lives here rather than there: `infero-cpu` is the
//! device layer and cannot depend on this crate without a cycle, since this
//! crate already depends on `infero-gpu`, which re-exports `infero-cpu`).
//!
//! Scope, per the user's own confirmed decision: a plain dense decoder
//! (embedding lookup, RMSNorm, RoPE, causal GQA attention, SwiGLU FFN, KV
//! cache read/write) at F32/F16/Q8_0 weight precision only. Every kernel name
//! reachable from that path is implemented below with the exact same
//! semantics as the corresponding CUDA kernel in `cu/ops.cu`/`cu/quant.cu`
//! (or, for `attn_decode_fused_f32`, the Metal kernel in `msl/ops.metal` --
//! CUDA has no equivalent because it never takes this branch). A kernel name
//! outside this set (MoE, GDN, TurboQuant, tensor-parallel, speculative
//! decoding, AWQ/FP8/FP4, CUDA-graph capture) falls to the `_` arm below and
//! fails loudly at the call site that needed it, rather than silently
//! degrading -- the build still succeeds, since nothing here is a missing
//! symbol, only a missing case.
//!
//! This is not a performance target (see `infero-cpu`'s own top-level doc
//! comment): every loop below is a plain sequential Rust loop. The one
//! exception is [`gemv`], `rayon`-parallelized over its output-row dimension
//! -- every FFN/QKVO projection routes through it, so leaving it
//! single-threaded pins the whole decode step to one of the host's cores
//! regardless of `--max-seqs` or core count (see Keel's latency requirement
//! in `docs/keel-integration.md`, which a single core cannot meet on this
//! model size).

use anyhow::{Result, bail};
use half::f16;
use infero_cpu::{Arg, LaunchConfig};
use rayon::prelude::*;

// ---- byte-level weight decoding ------------------------------------------
//
// Weight matrices arrive as a raw `&[u8]` (the GGUF file's own bytes, mapped
// or copied verbatim) rather than a typed slice, because a Q8_0 block's
// scale sits at a byte offset that is not generally a multiple of 2 --
// `size_of::<block_q8_0>() == 34` -- so reading it through a typed `&[f16]`
// would be misaligned-pointer undefined behavior. Reading every weight type
// through plain byte indexing sidesteps the question entirely, at the cost
// of a bit more arithmetic than a typed read would need.

#[inline]
fn read_f16_le(bytes: &[u8], byte_off: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[byte_off], bytes[byte_off + 1]])).to_f32()
}

#[inline]
fn read_f32_le(bytes: &[u8], byte_off: usize) -> f32 {
    f32::from_le_bytes([
        bytes[byte_off],
        bytes[byte_off + 1],
        bytes[byte_off + 2],
        bytes[byte_off + 3],
    ])
}

const QK8_0: usize = 32;
/// `size_of::<block_q8_0>()`: one `f16` scale plus 32 `i8` quants, packed --
/// see `common.cuh`'s own `block_q8_0` doc comment ("field order and padding
/// must match ggml-common.h exactly").
const BLOCK_Q8_0_BYTES: usize = 2 + QK8_0;

#[inline]
fn deq_f32(w: &[u8], i: usize) -> f32 {
    read_f32_le(w, i * 4)
}

#[inline]
fn deq_f16(w: &[u8], i: usize) -> f32 {
    read_f16_le(w, i * 2)
}

#[inline]
fn deq_q8_0(w: &[u8], i: usize) -> f32 {
    let block = i / QK8_0;
    let j = i % QK8_0;
    let base = block * BLOCK_Q8_0_BYTES;
    let d = read_f16_le(w, base);
    let q = w[base + 2 + j] as i8;
    d * (q as f32)
}

/// Decode element `i` of a raw weight buffer, dispatching on the kernel
/// name's own type suffix. Kept as one indirection point rather than
/// threading a `WeightType` through every call site here -- `dispatch`'s
/// name match already knows the suffix; this just needs the same three
/// arms wherever a decode happens.
#[derive(Clone, Copy)]
enum Ty {
    F32,
    F16,
    Q8_0,
}

impl Ty {
    fn from_suffix(s: &str) -> Option<Self> {
        match s {
            "f32" => Some(Ty::F32),
            "f16" => Some(Ty::F16),
            "q8_0" => Some(Ty::Q8_0),
            _ => None,
        }
    }

    fn from_weight_type(wt: crate::WeightType) -> Option<Self> {
        match wt {
            crate::WeightType::F32 => Some(Ty::F32),
            crate::WeightType::F16 => Some(Ty::F16),
            crate::WeightType::Q8_0 => Some(Ty::Q8_0),
            _ => None,
        }
    }

    #[inline]
    fn deq(self, w: &[u8], i: usize) -> f32 {
        match self {
            Ty::F32 => deq_f32(w, i),
            Ty::F16 => deq_f16(w, i),
            Ty::Q8_0 => deq_q8_0(w, i),
        }
    }
}

// ---- argument helpers -----------------------------------------------------

unsafe fn f32s(a: &Arg) -> &[f32] {
    unsafe { a.as_slice::<f32>() }
}
unsafe fn f32s_mut(a: &Arg) -> &mut [f32] {
    unsafe { a.as_mut_slice::<f32>() }
}
unsafe fn f16s(a: &Arg) -> &[f16] {
    unsafe { a.as_slice::<f16>() }
}
unsafe fn f16s_mut(a: &Arg) -> &mut [f16] {
    unsafe { a.as_mut_slice::<f16>() }
}
unsafe fn i32s(a: &Arg) -> &[i32] {
    unsafe { a.as_slice::<i32>() }
}
unsafe fn i32s_mut(a: &Arg) -> &mut [i32] {
    unsafe { a.as_mut_slice::<i32>() }
}
unsafe fn u32s_mut(a: &Arg) -> &mut [u32] {
    unsafe { a.as_mut_slice::<u32>() }
}
unsafe fn f64s(a: &Arg) -> &[f64] {
    unsafe { a.as_slice::<f64>() }
}
unsafe fn u8s(a: &Arg) -> &[u8] {
    unsafe { a.as_slice::<u8>() }
}
unsafe fn scalar_i32(a: &Arg) -> i32 {
    unsafe { a.as_scalar::<i32>() }
}
unsafe fn scalar_u64(a: &Arg) -> u64 {
    unsafe { a.as_scalar::<u64>() }
}
unsafe fn scalar_f32(a: &Arg) -> f32 {
    unsafe { a.as_scalar::<f32>() }
}

// ---- normalization ---------------------------------------------------------

/// `rms_norm_f32`/`rms_norm_f16_f32`: `out[t,:] = x[t,:] * rsqrt(mean(x[t,:]^2)
/// + eps) * weight`, optionally also writing an f16 copy of `out`. See
/// `ops.cu`'s `rms_norm_f32` and `mmvq.cu`'s `rms_norm_f16_f32`; both compute
/// the same thing, the f16-writing one just also has somewhere to put it.
fn rms_norm(out: &mut [f32], mut hout: Option<&mut [f16]>, x: &[f32], weight: &[f32], n_tokens: usize, d: usize, eps: f32) {
    for t in 0..n_tokens {
        let row = &x[t * d..t * d + d];
        let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let scale = (mean_sq + eps).sqrt().recip();
        let orow = &mut out[t * d..t * d + d];
        for i in 0..d {
            orow[i] = row[i] * scale * weight[i];
        }
        if let Some(h) = hout.as_deref_mut() {
            for i in 0..d {
                h[t * d + i] = f16::from_f32(orow[i]);
            }
        }
    }
}

/// `add_rms_norm_f16_f32`: folds `x += b` (the residual add) into the same
/// pass that then RMS-normalizes the updated `x` into `out`.
fn add_rms_norm(
    out: &mut [f32],
    mut hout: Option<&mut [f16]>,
    x: &mut [f32],
    b: &[f32],
    weight: &[f32],
    n_tokens: usize,
    d: usize,
    eps: f32,
) {
    for t in 0..n_tokens {
        let row = &mut x[t * d..t * d + d];
        let brow = &b[t * d..t * d + d];
        for i in 0..d {
            row[i] += brow[i];
        }
        let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let scale = (mean_sq + eps).sqrt().recip();
        let orow = &mut out[t * d..t * d + d];
        for i in 0..d {
            orow[i] = row[i] * scale * weight[i];
        }
        if let Some(h) = hout.as_deref_mut() {
            for i in 0..d {
                h[t * d + i] = f16::from_f32(orow[i]);
            }
        }
    }
}

/// `qk_norm_f32`: per-(token, head) RMSNorm over one head's `d_head` lane of
/// a packed `[q | k | v]` row, in place. See `ops.cu`'s own doc comment for
/// why `row_stride`/`offset` exist -- k sits inside the fused projection's
/// output row rather than in its own buffer.
#[allow(clippy::too_many_arguments)]
fn qk_norm(buf: &mut [f32], weight: &[f32], n_tokens: usize, n_heads: usize, d_head: usize, row_stride: usize, offset: usize, eps: f32) {
    for t in 0..n_tokens {
        for h in 0..n_heads {
            let base = t * row_stride + offset + h * d_head;
            let row = &mut buf[base..base + d_head];
            let acc: f32 = row.iter().map(|v| v * v).sum();
            let scale = (acc / d_head as f32 + eps).sqrt().recip();
            for i in 0..d_head {
                row[i] *= scale * weight[i];
            }
        }
    }
}

// ---- elementwise -----------------------------------------------------------

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ---- rotary embeddings ------------------------------------------------------

/// `rope_neox_f32`/`rope_norm_f32`: full-width rotary embeddings over
/// `x[n_tokens, n_heads, d_head]`. `interleaved` selects the pairing: `false`
/// pairs `i` with `i + half` (NeoX), `true` pairs `2i` with `2i+1`.
#[allow(clippy::too_many_arguments)]
fn rope_full(x: &mut [f32], positions: &[i32], freq_factors: &[f32], n_tokens: usize, n_heads: usize, d_head: usize, theta_base: f32, freq_scale: f32, interleaved: bool) {
    let half = d_head / 2;
    for t in 0..n_tokens {
        let pos = positions[t] as f32 * freq_scale;
        for h in 0..n_heads {
            let row = &mut x[(t * n_heads + h) * d_head..(t * n_heads + h + 1) * d_head];
            for i in 0..half {
                let inv_freq = theta_base.powf(-2.0 * i as f32 / d_head as f32);
                let angle = pos * inv_freq / freq_factors[i];
                let (sin_a, cos_a) = angle.sin_cos();
                let (ia, ib) = if interleaved { (2 * i, 2 * i + 1) } else { (i, i + half) };
                let a = row[ia];
                let b = row[ib];
                row[ia] = a * cos_a - b * sin_a;
                row[ib] = a * sin_a + b * cos_a;
            }
        }
    }
}

/// `rope_qk_f32`: Q and K in one call, each rotating only the first
/// `rotary_dim` of every head. See `ops.cu`'s own doc comment for the
/// mRoPE (`mrope_axis`/`pos_stride`) generalization this also serves.
#[allow(clippy::too_many_arguments)]
fn rope_qk(
    q: &mut [f32],
    k: &mut [f32],
    positions: &[i32],
    freq_factors: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    d_head: usize,
    rotary_dim: usize,
    theta_base: f32,
    freq_scale: f32,
    interleaved: bool,
    mrope_axis: &[i32],
    pos_stride: usize,
    n_tokens: usize,
) {
    let half = rotary_dim / 2;
    let rotate_row = |row: &mut [f32], t: usize| {
        for i in 0..half {
            let axis = mrope_axis[i] as usize;
            let pos = positions[t * pos_stride + axis] as f32 * freq_scale;
            let inv_freq = theta_base.powf(-2.0 * i as f32 / rotary_dim as f32);
            let angle = pos * inv_freq / freq_factors[i];
            let (sin_a, cos_a) = angle.sin_cos();
            let (ia, ib) = if interleaved { (2 * i, 2 * i + 1) } else { (i, i + half) };
            let a = row[ia];
            let b = row[ib];
            row[ia] = a * cos_a - b * sin_a;
            row[ib] = a * sin_a + b * cos_a;
        }
    };
    for t in 0..n_tokens {
        for head in 0..n_heads {
            rotate_row(&mut q[(t * n_heads + head) * d_head..(t * n_heads + head + 1) * d_head], t);
        }
        for head in 0..n_kv_heads {
            rotate_row(&mut k[(t * n_kv_heads + head) * d_head..(t * n_kv_heads + head + 1) * d_head], t);
        }
    }
}

/// `rope_qk_packed_f32`: [`rope_qk`] reading `k` out of (and rotating it in
/// place inside) a fused `[q | k | v]` row, and copying `q`'s rotated (and,
/// past `rotary_dim`, unrotated-but-still-copied) head out to `q_dst`. See
/// `ops.cu`'s own doc comment for why `q`'s tail needs an explicit copy where
/// `k`'s does not.
#[allow(clippy::too_many_arguments)]
fn rope_qk_packed(
    q_dst: &mut [f32],
    packed: &mut [f32],
    stride: usize,
    q_off: usize,
    k_off: usize,
    positions: &[i32],
    freq_factors: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    d_head: usize,
    rotary_dim: usize,
    theta_base: f32,
    freq_scale: f32,
    interleaved: i32,
    mrope_axis: &[i32],
    pos_stride: usize,
    n_tokens: usize,
) {
    let half = rotary_dim / 2;
    for t in 0..n_tokens {
        for y in 0..(n_heads + n_kv_heads) {
            let is_q = y < n_heads;
            let head = if is_q { y } else { y - n_heads };
            let off = if is_q { q_off } else { k_off };
            let src_base = t * stride + off + head * d_head;

            // The unrotated tail: k is already in place, only q needs a
            // copy across into its own buffer.
            if is_q {
                for d in rotary_dim..d_head {
                    let v = packed[src_base + d];
                    q_dst[(t * n_heads + head) * d_head + d] = v;
                }
            }

            for i in 0..half {
                let axis = mrope_axis[i] as usize;
                let pos = positions[t * pos_stride + axis] as f32 * freq_scale;
                let inv_freq = theta_base.powf(-2.0 * i as f32 / rotary_dim as f32);
                let angle = pos * inv_freq / freq_factors[i];
                let (sin_a, cos_a) = angle.sin_cos();
                let (ia, ib) = if interleaved != 0 { (2 * i, 2 * i + 1) } else { (i, i + half) };
                let a = packed[src_base + ia];
                let b = packed[src_base + ib];
                let (ra, rb) = (a * cos_a - b * sin_a, a * sin_a + b * cos_a);
                if is_q {
                    q_dst[(t * n_heads + head) * d_head + ia] = ra;
                    q_dst[(t * n_heads + head) * d_head + ib] = rb;
                } else {
                    packed[src_base + ia] = ra;
                    packed[src_base + ib] = rb;
                }
            }
        }
    }
}

// ---- KV cache ---------------------------------------------------------------

/// `store_kv_f16`: scatter `src[n_tokens, n_kv_heads, d_head]` into
/// `pool[n_kv_heads, n_slots, d_head]` at each token's physical slot.
fn store_kv(pool: &mut [f16], src: &[f32], slots: &[i32], n_kv_heads: usize, d_head: usize, n_slots: usize, n_tokens: usize) {
    for t in 0..n_tokens {
        let slot = slots[t];
        if slot < 0 || slot as usize >= n_slots {
            continue;
        }
        let slot = slot as usize;
        for h in 0..n_kv_heads {
            for i in 0..d_head {
                let dst = (h * n_slots + slot) * d_head + i;
                let s = (t * n_kv_heads + h) * d_head + i;
                pool[dst] = f16::from_f32(src[s]);
            }
        }
    }
}

/// `store_kv2_f16`: both halves of the cache in one call.
#[allow(clippy::too_many_arguments)]
fn store_kv2(k_pool: &mut [f16], v_pool: &mut [f16], k_src: &[f32], v_src: &[f32], slots: &[i32], n_kv_heads: usize, d_head: usize, n_slots: usize, n_tokens: usize) {
    store_kv(k_pool, k_src, slots, n_kv_heads, d_head, n_slots, n_tokens);
    store_kv(v_pool, v_src, slots, n_kv_heads, d_head, n_slots, n_tokens);
}

/// `store_kv2_packed_f16`: [`store_kv2`] reading k/v out of a fused row.
#[allow(clippy::too_many_arguments)]
fn store_kv2_packed(
    k_pool: &mut [f16],
    v_pool: &mut [f16],
    packed: &[f32],
    stride: usize,
    k_off: usize,
    v_off: usize,
    slots: &[i32],
    n_kv_heads: usize,
    d_head: usize,
    n_slots: usize,
    n_tokens: usize,
) {
    for t in 0..n_tokens {
        let slot = slots[t];
        if slot < 0 || slot as usize >= n_slots {
            continue;
        }
        let slot = slot as usize;
        for h in 0..n_kv_heads {
            for i in 0..d_head {
                let dst = (h * n_slots + slot) * d_head + i;
                k_pool[dst] = f16::from_f32(packed[t * stride + k_off + h * d_head + i]);
                v_pool[dst] = f16::from_f32(packed[t * stride + v_off + h * d_head + i]);
            }
        }
    }
}

/// `write_slot_table`: `table[seq_of[i] * stride + positions[i]] = slots[i]`.
fn write_slot_table(table: &mut [i32], seq_of: &[i32], positions: &[i32], slots: &[i32], stride: usize, n_tokens: usize) {
    for i in 0..n_tokens {
        table[seq_of[i] as usize * stride + positions[i] as usize] = slots[i];
    }
}

// ---- attention ---------------------------------------------------------------

/// `attn_scores_f32`: `scores[h,t,j] = dot(q[t,h,:], k_cache[h/group, slot,:])
/// * scale`, `-inf` where `j` is in `t`'s future. `scores` is
/// `[n_heads, n_tokens, kv_len]`.
#[allow(clippy::too_many_arguments)]
fn attn_scores(
    scores: &mut [f32],
    q: &[f32],
    k_cache: &[f16],
    seq_of: &[i32],
    positions: &[i32],
    slot_table: &[i32],
    table_stride: usize,
    n_heads: usize,
    n_kv_heads: usize,
    d_head: usize,
    n_slots: usize,
    kv_len: usize,
    scale: f32,
    n_tokens: usize,
) {
    let group = n_heads / n_kv_heads;
    for h in 0..n_heads {
        let kv_head = h / group;
        for t in 0..n_tokens {
            let qr = &q[(t * n_heads + h) * d_head..(t * n_heads + h + 1) * d_head];
            let table = &slot_table[seq_of[t] as usize * table_stride..];
            let limit = positions[t] as usize;
            let out_row = &mut scores[(h * n_tokens + t) * kv_len..(h * n_tokens + t + 1) * kv_len];
            for j in 0..kv_len {
                if j > limit {
                    out_row[j] = f32::NEG_INFINITY;
                    continue;
                }
                let slot = table[j] as usize;
                let kr = &k_cache[(kv_head * n_slots + slot) * d_head..(kv_head * n_slots + slot + 1) * d_head];
                let dot: f32 = qr.iter().zip(kr).map(|(a, b)| a * b.to_f32()).sum();
                out_row[j] = dot * scale;
            }
        }
    }
}

/// `attn_softmax_f32`: in-place softmax of `scores[n_heads, n_tokens, kv_len]`
/// over the last axis.
fn attn_softmax(scores: &mut [f32], n_heads: usize, n_tokens: usize, kv_len: usize) {
    for h in 0..n_heads {
        for t in 0..n_tokens {
            let row = &mut scores[(h * n_tokens + t) * kv_len..(h * n_tokens + t + 1) * kv_len];
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - m).exp();
                sum += *v;
            }
            let inv = sum.recip();
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }
}

/// `attn_output_f32`: `out[t,h,:] = sum_j scores[h,t,j] * v_cache[h/group,slot,:]`.
#[allow(clippy::too_many_arguments)]
fn attn_output(
    out: &mut [f32],
    scores: &[f32],
    v_cache: &[f16],
    seq_of: &[i32],
    positions: &[i32],
    slot_table: &[i32],
    table_stride: usize,
    n_heads: usize,
    n_kv_heads: usize,
    d_head: usize,
    n_slots: usize,
    kv_len: usize,
    n_tokens: usize,
) {
    let group = n_heads / n_kv_heads;
    for h in 0..n_heads {
        let kv_head = h / group;
        for t in 0..n_tokens {
            let srow = &scores[(h * n_tokens + t) * kv_len..(h * n_tokens + t + 1) * kv_len];
            let table = &slot_table[seq_of[t] as usize * table_stride..];
            let last = positions[t] as usize;
            let o = &mut out[(t * n_heads + h) * d_head..(t * n_heads + h + 1) * d_head];
            o.fill(0.0);
            for j in 0..kv_len.min(last + 1) {
                let slot = table[j] as usize;
                let vr = &v_cache[(kv_head * n_slots + slot) * d_head..(kv_head * n_slots + slot + 1) * d_head];
                let w = srow[j];
                for i in 0..d_head {
                    o[i] += w * vr[i].to_f32();
                }
            }
        }
    }
}

/// `attn_decode_fused_f32` (Metal-only; see `msl/ops.metal`'s own doc
/// comment -- CUDA never takes this branch): scores, softmax and the
/// V-weighted sum for one (head, token) pair, fused into a single pass
/// instead of three kernels over an intermediate `scores` buffer. Same math
/// as [`attn_scores`] + [`attn_softmax`] + [`attn_output`] in that order.
#[allow(clippy::too_many_arguments)]
fn attn_decode_fused(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f16],
    v_cache: &[f16],
    seq_of: &[i32],
    positions: &[i32],
    slot_table: &[i32],
    table_stride: usize,
    n_heads: usize,
    n_kv_heads: usize,
    d_head: usize,
    n_slots: usize,
    kv_len: usize,
    scale: f32,
    n_tokens: usize,
) {
    // `decode_attention()` (this kernel's own gate, in `infero-kernels/src/lib.rs`)
    // doesn't actually require `n_tokens == 1` -- its name is about the KV
    // cache shape, not the query width -- so this is also the CPU backend's
    // only prefill attention path, not just its decode-step one (there is no
    // CPU `attn_prefill_ws4`/`attn_flash`/tile kernel for a multi-token run
    // to fall to instead). Every `(head, token)` pair is independent (reads
    // its own `q` row, writes its own `out` slice), which is what lets this
    // split across cores; the original serial version's single shared
    // `scores` buffer, reused across every pair in sequence, made that
    // impossible without also giving each task its own copy, which the
    // `vec![0f32; len]` below is. For a real prefill this is a genuine
    // per-core win, not a with_min_len-style scheduling fix: unlike a gemv
    // row, a (head, token) pair's own work (`O(kv_len * d_head)`) isn't tiny,
    // and prefill's causal `kv_len` growing with `t` means this cost is
    // `O(n_tokens^2)` while every other per-request cost here is `O(n_tokens)`
    // -- serial, it increasingly dominates wall time as a query gets longer,
    // which is exactly the effect a longer-than-linear query/latency curve
    // on this backend traced back to.
    let group = n_heads / n_kv_heads;
    let threads = rayon::current_num_threads().max(1);
    let total = n_tokens * n_heads;
    let min_len = (total / threads).max(1);
    // `out` is `[n_tokens, n_heads, d_head]` (t-major, matching the serial
    // version's own indexing), so chunks of `d_head` in order are exactly
    // the `(t, h)` pairs in `t * n_heads + h` order.
    out[..total * d_head].par_chunks_mut(d_head).with_min_len(min_len).enumerate().for_each(|(idx, o)| {
        let (t, h) = (idx / n_heads, idx % n_heads);
        let kv_head = h / group;
        let last = positions[t] as usize;
        let len = (last + 1).min(kv_len);
        let table = &slot_table[seq_of[t] as usize * table_stride..];
        let qr = &q[(t * n_heads + h) * d_head..(t * n_heads + h + 1) * d_head];

        let mut scores = vec![0.0f32; len];
        let mut m = f32::NEG_INFINITY;
        for j in 0..len {
            let slot = table[j] as usize;
            let kr = &k_cache[(kv_head * n_slots + slot) * d_head..(kv_head * n_slots + slot + 1) * d_head];
            let dot: f32 = qr.iter().zip(kr).map(|(a, b)| a * b.to_f32()).sum();
            let s = dot * scale;
            scores[j] = s;
            m = m.max(s);
        }
        let mut sum = 0.0f32;
        for v in scores.iter_mut() {
            *v = (*v - m).exp();
            sum += *v;
        }
        let inv = sum.recip();

        o.fill(0.0);
        for j in 0..len {
            let slot = table[j] as usize;
            let vr = &v_cache[(kv_head * n_slots + slot) * d_head..(kv_head * n_slots + slot + 1) * d_head];
            let w = scores[j] * inv;
            for i in 0..d_head {
                o[i] += w * vr[i].to_f32();
            }
        }
    });
}

// ---- sampling ---------------------------------------------------------------
//
// Faithful ports of `sample.cu`'s device-side sampling -- see that file's own
// doc comment for why the logits never round-trip to the host. The CUDA
// kernels build a per-slice bitset from the penalty window and test
// membership through it purely as a device-parallelism trick; `samp_count`'s
// binary search already tells the same thing (`> 0` iff the token is in the
// window) in one call, so this port skips the bitset and calls it directly at
// each candidate index -- same answer, no shared-memory scratch to build.
//
// Every kernel here mirrors `sample.cu`'s own name; `argmax_*` is the
// `all_greedy` fast path, `sample_rows_f32` is the single-block
// `INFERO_NO_SAMPLE_SPLIT` fallback, and `sample_topk_partial_f32` /
// `sample_rows_topk_f32` are the default two-stage split path real serving
// takes (`Kernels::sample_rows_split`). The two-stage split is mathematically
// the same answer as the single-block scan (a token in the global top-k is in
// its own slice's top-k), so both are ported, faithfully, rather than one
// calling the other.

const SAMPLE_SPLITS: usize = 64;

#[derive(Clone, Copy)]
struct SampleParams {
    temperature: f32,
    top_p: f32,
    top_k: i32,
    rep_penalty: f32,
}

#[inline]
fn read_params(params: &[f32], row: usize) -> SampleParams {
    let p = &params[row * 4..row * 4 + 4];
    SampleParams { temperature: p[0], top_p: p[1], top_k: p[2].to_bits() as i32, rep_penalty: p[3] }
}

/// (value, index) under "higher value first, lower index first" -- the exact
/// tie-break `sample.cu`'s own `samp_better` uses, load-bearing because a
/// reduction that kept an arbitrary tied winner would disagree with the host
/// reference path on equal logits.
#[inline]
fn samp_better(av: f32, ai: i32, bv: f32, bi: i32) -> bool {
    av > bv || (av == bv && ai < bi)
}

/// How many times `tok` appears in this row's penalty window, by binary
/// search over the sorted-unique ids the host uploaded. See `sample.cu`'s
/// `samp_count`.
#[inline]
fn samp_count(toks: &[i32], cnts: &[i32], len: usize, tok: i32) -> i32 {
    let (mut lo, mut hi) = (0i64, len as i64 - 1);
    while lo <= hi {
        let mid = ((lo + hi) / 2) as usize;
        if toks[mid] == tok {
            return cnts[mid];
        } else if toks[mid] < tok {
            lo = mid as i64 + 1;
        } else {
            hi = mid as i64 - 1;
        }
    }
    0
}

/// The penalized logit. `once` picks between infero's two host paths (see
/// `sample.cu`'s own doc comment): greedy penalizes a distinct token once,
/// the top-k/top-p path penalizes it `count` times.
#[inline]
fn samp_penalize(l: f32, count: i32, p: f32, once: bool) -> f32 {
    if count <= 0 || p == 1.0 {
        return l;
    }
    let n = if once { 1 } else { count };
    let mut l = l;
    for _ in 0..n {
        l = if l > 0.0 { l / p } else { l * p };
    }
    l
}

/// One row's penalized value at vocabulary index `i`.
#[inline]
fn penalized(row_logits: &[f32], ptok: &[i32], pcnt: &[i32], plen: usize, rep_penalty: f32, once: bool, i: usize) -> f32 {
    let v = row_logits[i];
    let cnt = samp_count(ptok, pcnt, plen, i as i32);
    if cnt > 0 { samp_penalize(v, cnt, rep_penalty, once) } else { v }
}

#[inline]
fn pen_window<'a>(pen_tok: &'a [i32], pen_cnt: &'a [i32], pen_len: &[i32], row: usize, pen_stride: usize) -> (&'a [i32], &'a [i32], usize) {
    let plen = pen_len[row] as usize;
    (&pen_tok[row * pen_stride..row * pen_stride + plen], &pen_cnt[row * pen_stride..row * pen_stride + plen], plen)
}

/// `argmax_partial_f32`: one (row, slice) winner by penalized value.
#[allow(clippy::too_many_arguments)]
fn argmax_partial(
    pv: &mut [f32],
    pi: &mut [i32],
    logits: &[f32],
    params: &[f32],
    pen_tok: &[i32],
    pen_cnt: &[i32],
    pen_len: &[i32],
    vocab: usize,
    pen_stride: usize,
    splits: usize,
    n_rows: usize,
) {
    let chunk = vocab.div_ceil(splits);
    for row in 0..n_rows {
        let p = read_params(params, row);
        let row_logits = &logits[row * vocab..row * vocab + vocab];
        let (ptok, pcnt, plen) = pen_window(pen_tok, pen_cnt, pen_len, row, pen_stride);
        for s in 0..splits {
            let lo = s * chunk;
            let hi = (lo + chunk).min(vocab);
            let mut best = f32::NEG_INFINITY;
            let mut besti = i32::MAX;
            for i in lo..hi {
                let v = penalized(row_logits, ptok, pcnt, plen, p.rep_penalty, true, i);
                if samp_better(v, i as i32, best, besti) {
                    best = v;
                    besti = i as i32;
                }
            }
            pv[row * splits + s] = best;
            pi[row * splits + s] = besti;
        }
    }
}

/// `argmax_combine_f32`: the winner among a row's slice winners.
fn argmax_combine(out: &mut [u32], pv: &[f32], pi: &[i32], splits: usize, n_rows: usize) {
    for row in 0..n_rows {
        let mut best = f32::NEG_INFINITY;
        let mut besti = i32::MAX;
        for s in 0..splits {
            let v = pv[row * splits + s];
            let idx = pi[row * splits + s];
            if samp_better(v, idx, best, besti) {
                best = v;
                besti = idx;
            }
        }
        out[row] = besti as u32;
    }
}

/// The distribution a `sample_rows_f32`/`sample_rows_topk_f32` draw was made
/// from, for a caller (speculative decoding) that needs it back -- see
/// `Survivors`'s own doc comment. Not exercised by the dense-decoder path
/// this backend targets, but threaded through so the two kernels below match
/// `sample.cu`'s real signature.
struct Surv<'a> {
    id: &'a mut [u32],
    p: &'a mut [f32],
    len: &'a mut [i32],
    stride: usize,
}

/// `sample_rows_f32`: one block (here, one row of plain Rust) scans the whole
/// vocabulary, greedy or top-k/top-p with a host-seeded draw. See
/// `sample.cu`'s own doc comment for the full algorithm this mirrors exactly,
/// including the `once` distinction between the first (possibly-greedy) pass
/// and every later top-k pass.
#[allow(clippy::too_many_arguments)]
fn sample_rows(
    out: &mut [u32],
    logits: &[f32],
    params: &[f32],
    pen_tok: &[i32],
    pen_cnt: &[i32],
    pen_len: &[i32],
    rnd: &[f64],
    vocab: usize,
    pen_stride: usize,
    mut surv: Option<Surv<'_>>,
    n_rows: usize,
) {
    for row in 0..n_rows {
        let p = read_params(params, row);
        let row_logits = &logits[row * vocab..row * vocab + vocab];
        let (ptok, pcnt, plen) = pen_window(pen_tok, pen_cnt, pen_len, row, pen_stride);
        let greedy = p.temperature <= 0.0 || p.top_k == 1;

        let mut best = f32::NEG_INFINITY;
        let mut besti = 0i32;
        for i in 0..vocab {
            let v = penalized(row_logits, ptok, pcnt, plen, p.rep_penalty, greedy, i);
            if samp_better(v, i as i32, best, besti) {
                best = v;
                besti = i as i32;
            }
        }
        if greedy {
            out[row] = besti as u32;
            continue;
        }

        let k = (p.top_k.max(1) as usize).min(vocab);
        let mut kv = vec![0f32; k];
        let mut ki = vec![0i32; k];
        kv[0] = best;
        ki[0] = besti;
        for j in 1..k {
            let (lastv, lasti) = (kv[j - 1], ki[j - 1]);
            let (mut bv, mut bi, mut have) = (f32::NEG_INFINITY, 0i32, false);
            for i in 0..vocab {
                let v = penalized(row_logits, ptok, pcnt, plen, p.rep_penalty, false, i);
                if !samp_better(lastv, lasti, v, i as i32) {
                    continue;
                }
                if !have || samp_better(v, i as i32, bv, bi) {
                    bv = v;
                    bi = i as i32;
                    have = true;
                }
            }
            kv[j] = if have { bv } else { f32::NEG_INFINITY };
            ki[j] = if have { bi } else { i32::MAX };
        }
        finish_sample(out, row, &mut kv, &mut ki, k, p, &rnd[row..=row], &mut surv);
    }
}

/// The softmax-at-temperature, nucleus cut, and weighted draw shared by the
/// tail of `sample_rows_f32` and `sample_rows_topk_f32` -- identical in both,
/// down to the arithmetic order (`f64` accumulation), per `sample.cu`'s own
/// comment that both have to agree bit-for-bit-reproducibly with the host
/// path they replaced.
#[allow(clippy::too_many_arguments)]
fn finish_sample(out: &mut [u32], row: usize, kv: &mut [f32], ki: &mut [i32], k: usize, p: SampleParams, rnd_row: &[f64], surv: &mut Option<Surv<'_>>) {
    let inv_t = 1.0f32 / p.temperature.max(1e-5);
    let mx = kv[0];
    let mut total = 0f64;
    for j in 0..k {
        let q = (((kv[j] - mx) * inv_t) as f64).exp();
        kv[j] = q as f32;
        total += q;
    }
    let mut keep = k;
    if p.top_p < 1.0 {
        let target = total * p.top_p.clamp(1e-4, 1.0) as f64;
        let mut acc = 0f64;
        keep = 0;
        for j in 0..k {
            acc += kv[j] as f64;
            keep += 1;
            if acc >= target {
                break;
            }
        }
        keep = keep.max(1);
        total = 0f64;
        for j in 0..keep {
            total += kv[j] as f64;
        }
    }
    let mut r = rnd_row[0] * total;
    let mut pick = ki[keep - 1] as u32;
    for j in 0..keep {
        r -= kv[j] as f64;
        if r <= 0.0 {
            pick = ki[j] as u32;
            break;
        }
    }
    out[row] = pick;
    if let Some(s) = surv.as_mut() {
        s.len[row] = keep as i32;
        let inv = if total > 0.0 { 1.0 / total } else { 0.0 };
        for j in 0..keep.min(s.stride) {
            s.id[row * s.stride + j] = ki[j] as u32;
            s.p[row * s.stride + j] = (kv[j] as f64 * inv) as f32;
        }
    }
}

/// `sample_topk_partial_f32`: one (row, slice)'s own top-`cand_k`, penalized
/// value, padded with `(-inf, vocab)` once a slice runs out -- see
/// `sample.cu`'s own doc comment for why a slice's top-k always contains the
/// global top-k's share of it.
#[allow(clippy::too_many_arguments)]
fn sample_topk_partial(
    cand_v: &mut [f32],
    cand_i: &mut [i32],
    logits: &[f32],
    params: &[f32],
    pen_tok: &[i32],
    pen_cnt: &[i32],
    pen_len: &[i32],
    vocab: usize,
    pen_stride: usize,
    cand_k: usize,
    n_rows: usize,
) {
    let per = vocab.div_ceil(SAMPLE_SPLITS);
    for row in 0..n_rows {
        let p = read_params(params, row);
        let row_logits = &logits[row * vocab..row * vocab + vocab];
        let (ptok, pcnt, plen) = pen_window(pen_tok, pen_cnt, pen_len, row, pen_stride);
        let greedy = p.temperature <= 0.0 || p.top_k == 1;
        let k = (p.top_k.max(1) as usize).min(vocab);
        for split in 0..SAMPLE_SPLITS {
            let lo = split * per;
            let hi = (lo + per).min(vocab);
            let base = (row * SAMPLE_SPLITS + split) * cand_k;
            let out_v = &mut cand_v[base..base + cand_k];
            let out_i = &mut cand_i[base..base + cand_k];
            let (mut lastv, mut lasti) = (f32::INFINITY, -1i32);
            let mut exhausted = false;
            for j in 0..k {
                if exhausted {
                    out_v[j] = f32::NEG_INFINITY;
                    out_i[j] = vocab as i32;
                    continue;
                }
                let (mut bv, mut bi, mut have) = (f32::NEG_INFINITY, 0i32, false);
                for i in lo..hi {
                    let v = penalized(row_logits, ptok, pcnt, plen, p.rep_penalty, greedy, i);
                    if j > 0 && !samp_better(lastv, lasti, v, i as i32) {
                        continue;
                    }
                    if !have || samp_better(v, i as i32, bv, bi) {
                        bv = v;
                        bi = i as i32;
                        have = true;
                    }
                }
                lastv = if have { bv } else { f32::NEG_INFINITY };
                lasti = if have { bi } else { i32::MAX };
                out_v[j] = lastv;
                out_i[j] = if lasti == i32::MAX { vocab as i32 } else { lasti };
                if lasti == i32::MAX {
                    exhausted = true;
                }
            }
        }
    }
}

/// `sample_rows_topk_f32`: merges `sample_topk_partial_f32`'s per-slice
/// candidates into the row's real top-k (a candidate with `id >= vocab` is a
/// short slice's padding, skipped), then the same softmax/nucleus/draw tail
/// as `sample_rows_f32`.
#[allow(clippy::too_many_arguments)]
fn sample_rows_topk(
    out: &mut [u32],
    cand_v: &[f32],
    cand_i: &[i32],
    params: &[f32],
    rnd: &[f64],
    vocab: usize,
    cand_k: usize,
    mut surv: Option<Surv<'_>>,
    n_rows: usize,
) {
    let total_cand = SAMPLE_SPLITS * cand_k;
    for row in 0..n_rows {
        let p = read_params(params, row);
        let k = (p.top_k.max(1) as usize).min(vocab);
        let cv = &cand_v[row * total_cand..row * total_cand + total_cand];
        let ci = &cand_i[row * total_cand..row * total_cand + total_cand];
        let mut kv = vec![0f32; k];
        let mut ki = vec![0i32; k];
        let (mut lastv, mut lasti) = (f32::INFINITY, -1i32);
        for j in 0..k {
            let (mut bv, mut bi, mut have) = (f32::NEG_INFINITY, 0i32, false);
            for idx in 0..total_cand {
                let id = ci[idx];
                if id as usize >= vocab {
                    continue;
                }
                let v = cv[idx];
                if j > 0 && !samp_better(lastv, lasti, v, id) {
                    continue;
                }
                if !have || samp_better(v, id, bv, bi) {
                    bv = v;
                    bi = id;
                    have = true;
                }
            }
            lastv = if have { bv } else { f32::NEG_INFINITY };
            lasti = if have { bi } else { i32::MAX };
            kv[j] = lastv;
            ki[j] = lasti;
        }
        finish_sample(out, row, &mut kv, &mut ki, k, p, &rnd[row..=row], &mut surv);
    }
}

// ---- weights ------------------------------------------------------------------

/// `gather_rows_{f32,f16,q8_0}`: `out[t,:] = dequant(w[rows[t],:])`.
fn gather_rows(out: &mut [f32], w: &[u8], rows: &[i32], n_tokens: usize, k: usize, ty: Ty) {
    for t in 0..n_tokens {
        let src_row = rows[t] as usize;
        for i in 0..k {
            out[t * k + i] = ty.deq(w, src_row * k + i);
        }
    }
}

/// `dequant_{f32,f16,q8_0}_f16`: decode a whole matrix to f16.
fn dequant_to_f16(out: &mut [f16], w: &[u8], n_elements: usize, ty: Ty) {
    for i in 0..n_elements {
        out[i] = f16::from_f32(ty.deq(w, i));
    }
}

/// Decode a whole weight matrix to f32, once, for [`gemv`]'s `gemm_f32`
/// path. `Ty::deq` re-matches on `ty` every single element, which -- even
/// inlined -- leaves a 3-way branch inside what should be a tight
/// convert-and-store loop; matching once here and running a loop written
/// for exactly one variant is what let this call stop being the dominant
/// per-request cost it was (see `gemv`'s own doc comment on this path).
fn decode_matrix_to_f32(w: &[u8], n_elements: usize, ty: Ty) -> Vec<f32> {
    let mut out = vec![0f32; n_elements];
    let threads = rayon::current_num_threads().max(1);
    let min_len = (n_elements / threads).max(1);
    match ty {
        Ty::F32 => out.par_iter_mut().with_min_len(min_len).enumerate().for_each(|(i, o)| *o = deq_f32(w, i)),
        // Reinterpreting as a proper `&[f16]` and decoding through
        // `half`'s own `to_f32` (`half::f16` is `#[repr(transparent)]` over
        // `u16`) is the same shape `gemm_f16_to_f32`'s operand decode
        // already uses, and it measurably beats going element-by-element
        // through a byte offset even with the match hoisted out: reading
        // `w` as bytes and reassembling a `u16` per element, instead of
        // reading it as `u16`/`f16` in the first place, was still enough to
        // keep this from vectorizing the way the identical-looking loop in
        // `gemm_f16_to_f32` does. Falls back to the byte-offset path only
        // if `w` somehow isn't 2-byte aligned (mmap'd/heap weight buffers
        // always are in practice, but this is UB to get wrong).
        Ty::F16 if w.as_ptr() as usize % 2 == 0 && w.len() >= n_elements * 2 => {
            // SAFETY: alignment and length checked in the guard above; `w`
            // is borrowed for exactly this call's lifetime, matching every
            // other reinterpret this backend does of its own weight bytes.
            let w16: &[f16] = unsafe { std::slice::from_raw_parts(w.as_ptr() as *const f16, n_elements) };
            out.par_iter_mut().zip(w16.par_iter()).with_min_len(min_len).for_each(|(o, h)| *o = h.to_f32());
        }
        Ty::F16 => out.par_iter_mut().with_min_len(min_len).enumerate().for_each(|(i, o)| *o = deq_f16(w, i)),
        Ty::Q8_0 => out.par_iter_mut().with_min_len(min_len).enumerate().for_each(|(i, o)| *o = deq_q8_0(w, i)),
    }
    out
}

/// [`decode_matrix_to_f32`] with a public, `WeightType`-keyed entry point --
/// `crate::Kernels::cpu_decode_weight_f32`'s whole body, kept here since
/// `Ty` (this module's own three-variant subset) is private to it.
pub(crate) fn decode_weight_f32(bytes: &[u8], n_elements: usize, wt: crate::WeightType) -> anyhow::Result<Vec<f32>> {
    let ty = Ty::from_weight_type(wt)
        .ok_or_else(|| anyhow::anyhow!("cpu backend cannot decode weight type {wt}"))?;
    Ok(decode_matrix_to_f32(bytes, n_elements, ty))
}

/// `gemv_{f32,f16,q8_0}` (and its `gemv1_*`/`gemv2_*`/`gemv4_*` token-width
/// aliases, which name the same computation -- see `dispatch`'s own comment
/// on why those extra names exist): `out[t,row] = dot(w[row,:], x[t,:])`.
fn gemv(out: &mut [f32], w: &[u8], x: &[f32], k: usize, n: usize, n_tokens: usize, ty: Ty) {
    if n_tokens == 1 {
        // The true decode-step case: a mat-vec is memory-bandwidth-bound
        // regardless of implementation (one FMA per weight byte read, full
        // stop), so there is nothing for a cache-blocked GEMM microkernel to
        // win here and decoding to a separate f32 buffer first would only
        // add a second pass over that memory. Fused decode+dot, one row per
        // core, stays the right shape. `with_min_len` caps rayon's
        // fork-join tree to about one leaf per core -- an earlier
        // one-leaf-per-row split left ~86% of cycles in `join_context`
        // bookkeeping (`perf` on this exact call shape), rivaling the real
        // work of each (tiny) row.
        let threads = rayon::current_num_threads().max(1);
        let min_rows = (n / threads).max(1);
        out[..n].par_chunks_mut(1).with_min_len(min_rows).enumerate().for_each(|(row, out_row)| {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += ty.deq(w, row * k + i) * x[i];
            }
            out_row[0] = acc;
        });
    } else {
        // `embeddings`/prefill's own case (below `gemm_threshold()` tokens --
        // see `Kernels::gemv`'s doc comment): now enough arithmetic per byte
        // read (`n_tokens` dot products share each decoded weight row) that
        // a cache-blocked GEMM microkernel is worth it. Decode the whole
        // weight matrix to f32 once, then hand the actual product to
        // `gemm_f32` (see its own doc comment for why that's the `gemm`
        // crate rather than a hand-rolled loop). `out`'s layout
        // (`out[t*n+row]`) already matches what `gemm_f32` writes for
        // `dst(m=n_tokens × n)`, so there's no scatter step to write here.
        let w_f32 = decode_matrix_to_f32(w, n * k, ty);
        infero_cpu::gemm_f32(&mut out[..n_tokens * n], &x[..n_tokens * k], &w_f32, n_tokens, k, n);
    }
}

// ---- dispatch -----------------------------------------------------------------

/// Parses a `gemv`/`gemv1`/`gemv2`/`gemv4` name prefix and its `_{suffix}`
/// weight-type tag. These four names are Metal's own token-batching tiling
/// choice (see `Kernels::gemv`'s `per`/`rows` logic) -- purely a grouping
/// detail that changes nothing about the sum each call computes, since
/// `n_tokens` is always passed as a real argument regardless of which name
/// reached this dispatcher. All four resolve to the same [`gemv`].
fn parse_gemv_name(name: &str) -> Option<Ty> {
    let rest = name.strip_prefix("gemv")?;
    let suffix = rest
        .strip_prefix("1_")
        .or_else(|| rest.strip_prefix("2_"))
        .or_else(|| rest.strip_prefix("4_"))
        .or_else(|| rest.strip_prefix("_"))?;
    Ty::from_suffix(suffix)
}

pub fn dispatch(name: &str, args: &[Arg], cfg: LaunchConfig) -> Result<()> {
    unsafe {
        match name {
            "rms_norm_f32" => {
                let n_tokens = cfg.grid_dim.0 as usize;
                let d = scalar_i32(&args[3]) as usize;
                let eps = scalar_f32(&args[4]);
                let out = f32s_mut(&args[0]);
                let x = f32s(&args[1]);
                let weight = f32s(&args[2]);
                rms_norm(out, None, x, weight, n_tokens, d, eps);
            }
            "rms_norm_f16_f32" => {
                let n_tokens = cfg.grid_dim.0 as usize;
                let d = scalar_i32(&args[4]) as usize;
                let eps = scalar_f32(&args[5]);
                let out = f32s_mut(&args[0]);
                let hout = if args[1].is_nil() { None } else { Some(f16s_mut(&args[1])) };
                let x = f32s(&args[2]);
                let weight = f32s(&args[3]);
                rms_norm(out, hout, x, weight, n_tokens, d, eps);
            }
            "add_rms_norm_f16_f32" => {
                let n_tokens = cfg.grid_dim.0 as usize;
                let d = scalar_i32(&args[5]) as usize;
                let eps = scalar_f32(&args[6]);
                let out = f32s_mut(&args[0]);
                let hout = if args[1].is_nil() { None } else { Some(f16s_mut(&args[1])) };
                let x = f32s_mut(&args[2]);
                let b = f32s(&args[3]);
                let weight = f32s(&args[4]);
                add_rms_norm(out, hout, x, b, weight, n_tokens, d, eps);
            }
            "qk_norm_f32" => {
                let n_heads = scalar_i32(&args[2]) as usize;
                let n_tokens = cfg.grid_dim.0 as usize / n_heads.max(1);
                let d_head = scalar_i32(&args[3]) as usize;
                let row_stride = scalar_i32(&args[4]) as usize;
                let offset = scalar_i32(&args[5]) as usize;
                let eps = scalar_f32(&args[6]);
                let buf = f32s_mut(&args[0]);
                let weight = f32s(&args[1]);
                qk_norm(buf, weight, n_tokens, n_heads, d_head, row_stride, offset, eps);
            }
            "add_f32" => {
                let n = scalar_i32(&args[3]) as usize;
                let out = f32s_mut(&args[0]);
                let a = f32s(&args[1]);
                let b = f32s(&args[2]);
                for i in 0..n {
                    out[i] = a[i] + b[i];
                }
            }
            "add_assign_f32" => {
                let n = scalar_i32(&args[2]) as usize;
                let out = f32s_mut(&args[0]);
                let b = f32s(&args[1]);
                for i in 0..n {
                    out[i] += b[i];
                }
            }
            "add_bias_f32" => {
                let n_cols = scalar_i32(&args[2]) as usize;
                let n_rows = scalar_i32(&args[3]) as usize;
                let out = f32s_mut(&args[0]);
                let bias = f32s(&args[1]);
                for t in 0..n_rows {
                    for j in 0..n_cols {
                        out[t * n_cols + j] += bias[j];
                    }
                }
            }
            "silu_mul_f32" => {
                let n = scalar_i32(&args[3]) as usize;
                let out = f32s_mut(&args[0]);
                let gate = f32s(&args[1]);
                let up = f32s(&args[2]);
                for i in 0..n {
                    out[i] = silu(gate[i]) * up[i];
                }
            }
            "silu_mul_split_f32" => {
                let d_ff = scalar_i32(&args[2]) as usize;
                let total = scalar_i32(&args[3]) as usize;
                let out = f32s_mut(&args[0]);
                let xy = f32s(&args[1]);
                for idx in 0..total {
                    let row = idx / d_ff;
                    let col = idx % d_ff;
                    let r = &xy[row * 2 * d_ff..row * 2 * d_ff + 2 * d_ff];
                    out[idx] = silu(r[col]) * r[d_ff + col];
                }
            }
            "silu_mul_split_f16_f32" => {
                let d_ff = scalar_i32(&args[3]) as usize;
                let total = scalar_i32(&args[4]) as usize;
                let out = f32s_mut(&args[0]);
                let hout = f16s_mut(&args[1]);
                let xy = f32s(&args[2]);
                for idx in 0..total {
                    let row = idx / d_ff;
                    let col = idx % d_ff;
                    let r = &xy[row * 2 * d_ff..row * 2 * d_ff + 2 * d_ff];
                    let v = silu(r[col]) * r[d_ff + col];
                    out[idx] = v;
                    hout[idx] = f16::from_f32(v);
                }
            }
            "split_qkv_f32" => {
                let d = scalar_i32(&args[4]) as usize;
                let kv_dim = scalar_i32(&args[5]) as usize;
                let total = scalar_i32(&args[6]) as usize;
                let q = f32s_mut(&args[0]);
                let k = f32s_mut(&args[1]);
                let v = f32s_mut(&args[2]);
                let fused = f32s(&args[3]);
                let row_w = d + 2 * kv_dim;
                for idx in 0..total {
                    let row = idx / row_w;
                    let col = idx % row_w;
                    let x = fused[idx];
                    if col < d {
                        q[row * d + col] = x;
                    } else if col < d + kv_dim {
                        k[row * kv_dim + (col - d)] = x;
                    } else {
                        v[row * kv_dim + (col - d - kv_dim)] = x;
                    }
                }
            }
            "split2_f32" => {
                let width_a = scalar_i32(&args[3]) as usize;
                let width_b = scalar_i32(&args[4]) as usize;
                let total = scalar_i32(&args[5]) as usize;
                let a = f32s_mut(&args[0]);
                let b = f32s_mut(&args[1]);
                let fused = f32s(&args[2]);
                let row_w = width_a + width_b;
                for idx in 0..total {
                    let row = idx / row_w;
                    let col = idx % row_w;
                    let x = fused[idx];
                    if col < width_a {
                        a[row * width_a + col] = x;
                    } else {
                        b[row * width_b + (col - width_a)] = x;
                    }
                }
            }
            "take_rows_f32" => {
                let d = scalar_i32(&args[3]) as usize;
                let n_rows = cfg.grid_dim.1 as usize;
                let out = f32s_mut(&args[0]);
                let x = f32s(&args[1]);
                let rows = i32s(&args[2]);
                for r in 0..n_rows {
                    let src = rows[r] as usize;
                    for i in 0..d {
                        out[r * d + i] = x[src * d + i];
                    }
                }
            }
            "f32_to_f16" => {
                let n = scalar_i32(&args[2]) as usize;
                let out = f16s_mut(&args[0]);
                let x = f32s(&args[1]);
                for i in 0..n {
                    out[i] = f16::from_f32(x[i]);
                }
            }
            "f16_to_f32" => {
                let n = scalar_i32(&args[2]) as usize;
                let out = f32s_mut(&args[0]);
                let x = f16s(&args[1]);
                for i in 0..n {
                    out[i] = x[i].to_f32();
                }
            }
            "rope_neox_f32" | "rope_norm_f32" => {
                let n_heads = scalar_i32(&args[3]) as usize;
                let d_head = scalar_i32(&args[4]) as usize;
                let theta_base = scalar_f32(&args[5]);
                let freq_scale = scalar_f32(&args[6]);
                let n_tokens = cfg.grid_dim.2 as usize;
                let x = f32s_mut(&args[0]);
                let positions = i32s(&args[1]);
                let freq_factors = f32s(&args[2]);
                rope_full(x, positions, freq_factors, n_tokens, n_heads, d_head, theta_base, freq_scale, name == "rope_norm_f32");
            }
            "rope_qk_f32" => {
                let n_heads = scalar_i32(&args[4]) as usize;
                let n_kv_heads = scalar_i32(&args[5]) as usize;
                let d_head = scalar_i32(&args[6]) as usize;
                let rotary_dim = scalar_i32(&args[7]) as usize;
                let theta_base = scalar_f32(&args[8]);
                let freq_scale = scalar_f32(&args[9]);
                let interleaved = scalar_i32(&args[10]);
                let pos_stride = scalar_i32(&args[12]) as usize;
                let n_tokens = cfg.grid_dim.2 as usize;
                let q = f32s_mut(&args[0]);
                let k = f32s_mut(&args[1]);
                let positions = i32s(&args[2]);
                let freq_factors = f32s(&args[3]);
                let mrope_axis = i32s(&args[11]);
                rope_qk(
                    q, k, positions, freq_factors, n_heads, n_kv_heads, d_head, rotary_dim, theta_base, freq_scale,
                    interleaved != 0, mrope_axis, pos_stride, n_tokens,
                );
            }
            "rope_qk_packed_f32" => {
                let stride = scalar_i32(&args[2]) as usize;
                let q_off = scalar_i32(&args[3]) as usize;
                let k_off = scalar_i32(&args[4]) as usize;
                let n_heads = scalar_i32(&args[7]) as usize;
                let n_kv_heads = scalar_i32(&args[8]) as usize;
                let d_head = scalar_i32(&args[9]) as usize;
                let rotary_dim = scalar_i32(&args[10]) as usize;
                let theta_base = scalar_f32(&args[11]);
                let freq_scale = scalar_f32(&args[12]);
                let interleaved = scalar_i32(&args[13]);
                let pos_stride = scalar_i32(&args[15]) as usize;
                let n_tokens = cfg.grid_dim.2 as usize;
                let q_dst = f32s_mut(&args[0]);
                let packed = f32s_mut(&args[1]);
                let positions = i32s(&args[5]);
                let freq_factors = f32s(&args[6]);
                let mrope_axis = i32s(&args[14]);
                rope_qk_packed(
                    q_dst, packed, stride, q_off, k_off, positions, freq_factors, n_heads, n_kv_heads, d_head,
                    rotary_dim, theta_base, freq_scale, interleaved, mrope_axis, pos_stride, n_tokens,
                );
            }
            "store_kv_f16" => {
                let n_kv_heads = scalar_i32(&args[3]) as usize;
                let d_head = scalar_i32(&args[4]) as usize;
                let n_slots = scalar_i32(&args[5]) as usize;
                let n_tokens = scalar_i32(&args[6]) as usize;
                let pool = f16s_mut(&args[0]);
                let src = f32s(&args[1]);
                let slots = i32s(&args[2]);
                store_kv(pool, src, slots, n_kv_heads, d_head, n_slots, n_tokens);
            }
            "store_kv2_f16" => {
                let n_kv_heads = scalar_i32(&args[5]) as usize;
                let d_head = scalar_i32(&args[6]) as usize;
                let n_slots = scalar_i32(&args[7]) as usize;
                let n_tokens = scalar_i32(&args[8]) as usize;
                let k_pool = f16s_mut(&args[0]);
                let v_pool = f16s_mut(&args[1]);
                let k_src = f32s(&args[2]);
                let v_src = f32s(&args[3]);
                let slots = i32s(&args[4]);
                store_kv2(k_pool, v_pool, k_src, v_src, slots, n_kv_heads, d_head, n_slots, n_tokens);
            }
            "store_kv2_packed_f16" => {
                let stride = scalar_i32(&args[3]) as usize;
                let k_off = scalar_i32(&args[4]) as usize;
                let v_off = scalar_i32(&args[5]) as usize;
                let n_kv_heads = scalar_i32(&args[7]) as usize;
                let d_head = scalar_i32(&args[8]) as usize;
                let n_slots = scalar_i32(&args[9]) as usize;
                let n_tokens = scalar_i32(&args[10]) as usize;
                let k_pool = f16s_mut(&args[0]);
                let v_pool = f16s_mut(&args[1]);
                let packed = f32s(&args[2]);
                let slots = i32s(&args[6]);
                store_kv2_packed(k_pool, v_pool, packed, stride, k_off, v_off, slots, n_kv_heads, d_head, n_slots, n_tokens);
            }
            "write_slot_table" => {
                let stride = scalar_i32(&args[4]) as usize;
                let n_tokens = scalar_i32(&args[5]) as usize;
                let table = i32s_mut(&args[0]);
                let seq_of = i32s(&args[1]);
                let positions = i32s(&args[2]);
                let slots = i32s(&args[3]);
                write_slot_table(table, seq_of, positions, slots, stride, n_tokens);
            }
            "attn_scores_f32" => {
                let table_stride = scalar_i32(&args[6]) as usize;
                let n_heads = scalar_i32(&args[7]) as usize;
                let n_kv_heads = scalar_i32(&args[8]) as usize;
                let d_head = scalar_i32(&args[9]) as usize;
                let n_slots = scalar_i32(&args[10]) as usize;
                let kv_len = scalar_i32(&args[11]) as usize;
                let scale = scalar_f32(&args[12]);
                let n_tokens = cfg.grid_dim.2 as usize;
                let scores = f32s_mut(&args[0]);
                let q = f32s(&args[1]);
                let k_cache = f16s(&args[2]);
                let seq_of = i32s(&args[3]);
                let positions = i32s(&args[4]);
                let slot_table = i32s(&args[5]);
                attn_scores(
                    scores, q, k_cache, seq_of, positions, slot_table, table_stride, n_heads, n_kv_heads, d_head,
                    n_slots, kv_len, scale, n_tokens,
                );
            }
            "attn_softmax_f32" => {
                let kv_len = scalar_i32(&args[1]) as usize;
                let n_heads = cfg.grid_dim.0 as usize;
                let n_tokens = cfg.grid_dim.1 as usize;
                let scores = f32s_mut(&args[0]);
                attn_softmax(scores, n_heads, n_tokens, kv_len);
            }
            "attn_output_f32" => {
                let table_stride = scalar_i32(&args[6]) as usize;
                let n_heads = scalar_i32(&args[7]) as usize;
                let n_kv_heads = scalar_i32(&args[8]) as usize;
                let d_head = scalar_i32(&args[9]) as usize;
                let n_slots = scalar_i32(&args[10]) as usize;
                let kv_len = scalar_i32(&args[11]) as usize;
                let n_tokens = cfg.grid_dim.1 as usize;
                let out = f32s_mut(&args[0]);
                let scores = f32s(&args[1]);
                let v_cache = f16s(&args[2]);
                let seq_of = i32s(&args[3]);
                let positions = i32s(&args[4]);
                let slot_table = i32s(&args[5]);
                attn_output(
                    out, scores, v_cache, seq_of, positions, slot_table, table_stride, n_heads, n_kv_heads, d_head,
                    n_slots, kv_len, n_tokens,
                );
            }
            "attn_decode_fused_f32" => {
                let table_stride = scalar_i32(&args[7]) as usize;
                let n_heads = scalar_i32(&args[8]) as usize;
                let n_kv_heads = scalar_i32(&args[9]) as usize;
                let d_head = scalar_i32(&args[10]) as usize;
                let n_slots = scalar_i32(&args[11]) as usize;
                let kv_len = scalar_i32(&args[12]) as usize;
                let scale = scalar_f32(&args[13]);
                let n_tokens = cfg.grid_dim.1 as usize;
                let out = f32s_mut(&args[0]);
                let q = f32s(&args[1]);
                let k_cache = f16s(&args[2]);
                let v_cache = f16s(&args[3]);
                let seq_of = i32s(&args[4]);
                let positions = i32s(&args[5]);
                let slot_table = i32s(&args[6]);
                attn_decode_fused(
                    out, q, k_cache, v_cache, seq_of, positions, slot_table, table_stride, n_heads, n_kv_heads,
                    d_head, n_slots, kv_len, scale, n_tokens,
                );
            }
            "argmax_partial_f32" => {
                let splits = scalar_i32(&args[9]) as usize;
                let n_rows = cfg.grid_dim.1 as usize;
                let vocab = scalar_i32(&args[7]) as usize;
                let pen_stride = scalar_i32(&args[8]) as usize;
                let pv = f32s_mut(&args[0]);
                let pi = i32s_mut(&args[1]);
                let logits = f32s(&args[2]);
                let params = f32s(&args[3]);
                let pen_tok = i32s(&args[4]);
                let pen_cnt = i32s(&args[5]);
                let pen_len = i32s(&args[6]);
                argmax_partial(pv, pi, logits, params, pen_tok, pen_cnt, pen_len, vocab, pen_stride, splits, n_rows);
            }
            "argmax_combine_f32" => {
                let splits = scalar_i32(&args[3]) as usize;
                let n_rows = cfg.grid_dim.0 as usize;
                let out = u32s_mut(&args[0]);
                let pv = f32s(&args[1]);
                let pi = i32s(&args[2]);
                argmax_combine(out, pv, pi, splits, n_rows);
            }
            "sample_rows_f32" => {
                let vocab = scalar_i32(&args[7]) as usize;
                let pen_stride = scalar_i32(&args[8]) as usize;
                let sstride = scalar_i32(&args[12]) as usize;
                let n_rows = cfg.grid_dim.0 as usize;
                let out = u32s_mut(&args[0]);
                let logits = f32s(&args[1]);
                let params = f32s(&args[2]);
                let pen_tok = i32s(&args[3]);
                let pen_cnt = i32s(&args[4]);
                let pen_len = i32s(&args[5]);
                let rnd = f64s(&args[6]);
                let surv = if args[9].is_nil() {
                    None
                } else {
                    Some(Surv { id: u32s_mut(&args[9]), p: f32s_mut(&args[10]), len: i32s_mut(&args[11]), stride: sstride })
                };
                sample_rows(out, logits, params, pen_tok, pen_cnt, pen_len, rnd, vocab, pen_stride, surv, n_rows);
            }
            "sample_topk_partial_f32" => {
                let vocab = scalar_i32(&args[7]) as usize;
                let pen_stride = scalar_i32(&args[8]) as usize;
                let cand_k = scalar_i32(&args[9]) as usize;
                let n_rows = cfg.grid_dim.0 as usize;
                let cand_v = f32s_mut(&args[0]);
                let cand_i = i32s_mut(&args[1]);
                let logits = f32s(&args[2]);
                let params = f32s(&args[3]);
                let pen_tok = i32s(&args[4]);
                let pen_cnt = i32s(&args[5]);
                let pen_len = i32s(&args[6]);
                sample_topk_partial(cand_v, cand_i, logits, params, pen_tok, pen_cnt, pen_len, vocab, pen_stride, cand_k, n_rows);
            }
            "sample_rows_topk_f32" => {
                let vocab = scalar_i32(&args[5]) as usize;
                let cand_k = scalar_i32(&args[6]) as usize;
                let sstride = scalar_i32(&args[10]) as usize;
                let n_rows = cfg.grid_dim.0 as usize;
                let out = u32s_mut(&args[0]);
                let cand_v = f32s(&args[1]);
                let cand_i = i32s(&args[2]);
                let params = f32s(&args[3]);
                let rnd = f64s(&args[4]);
                let surv = if args[7].is_nil() {
                    None
                } else {
                    Some(Surv { id: u32s_mut(&args[7]), p: f32s_mut(&args[8]), len: i32s_mut(&args[9]), stride: sstride })
                };
                sample_rows_topk(out, cand_v, cand_i, params, rnd, vocab, cand_k, surv, n_rows);
            }
            "gather_rows_f32" | "gather_rows_f16" | "gather_rows_q8_0" => {
                let ty = Ty::from_suffix(name.strip_prefix("gather_rows_").unwrap()).unwrap();
                let k = scalar_i32(&args[3]) as usize;
                let n_tokens = cfg.grid_dim.1 as usize;
                let out = f32s_mut(&args[0]);
                let w = u8s(&args[1]);
                let rows = i32s(&args[2]);
                gather_rows(out, w, rows, n_tokens, k, ty);
            }
            "dequant_f32_f16" | "dequant_f16_f16" | "dequant_q8_0_f16" => {
                let ty = Ty::from_suffix(name.strip_prefix("dequant_").unwrap().strip_suffix("_f16").unwrap()).unwrap();
                let n_elements = scalar_u64(&args[2]) as usize;
                let out = f16s_mut(&args[0]);
                let w = u8s(&args[1]);
                dequant_to_f16(out, w, n_elements, ty);
            }
            _ => {
                if let Some(ty) = parse_gemv_name(name) {
                    let k = scalar_i32(&args[3]) as usize;
                    let n = scalar_i32(&args[4]) as usize;
                    let n_tokens = scalar_i32(&args[5]) as usize;
                    let out = f32s_mut(&args[0]);
                    let w = u8s(&args[1]);
                    let x = f32s(&args[2]);
                    gemv(out, w, x, k, n, n_tokens, ty);
                } else {
                    bail!(
                        "kernel '{name}' is not implemented on the CPU backend (dense-decoder-only: \
                         F32/F16/Q8_0 RMSNorm/RoPE/causal-GQA-attention/SwiGLU -- no MoE, vision, GDN, \
                         TurboQuant, tensor parallelism, or speculative decoding)"
                    );
                }
            }
        }
    }
    Ok(())
}
