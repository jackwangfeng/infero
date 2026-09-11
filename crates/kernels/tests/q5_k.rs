//! Correctness for `WeightType::Q5K` -- llama.cpp's `Q5_K` GGUF format
//! (`Q4_K`'s own 256-element super-block and 6-bit scale/min encoding, plus
//! a 5th bit per weight in a separate `qh` array). Added because a real
//! `Qwen3.8-27B-Q4_K_M.gguf` checkpoint mixes `Q5_K` tensors in with `Q4_K`
//! ones (llama.cpp's own real strategy for "more sensitive" tensors like
//! `attn_k`/`attn_v`), and infero's loader loud-failed on it
//! (`weight type Q5_K is not implemented`) before this.
//!
//! No download needed: this builds synthetic-but-real `block_q5_K` bytes by
//! hand and checks every real kernel path (`dequant_to_f16`, the float
//! `gemv`, the integer `mmvq`) against a pure-Rust host reference, the same
//! way `fp4_quantize_act.rs`'s own real-checkpoint-magnitude test does for
//! NVFP4 -- not against a second, independently-fallible reimplementation,
//! but against the exact bit logic verified against real, freshly-fetched
//! ggml source (`ggml-quants.c::dequantize_row_q5_K`,
//! `ggml-cuda/vecdotq.cuh::vec_dot_q5_K_q8_1`, 2026-09-11) before any of
//! this file or the device kernels were written.

mod common;

use anyhow::Result;
use half::f16;
use infero_kernels::{Kernels, WeightType};

use common::*;

const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12;

/// Real on-disk `block_q5_K` layout (`d`, `dmin`, `scales[12]`, `qh[32]`,
/// `qs[128]` -- 176 bytes), matching `common.cuh`'s own struct field for
/// field.
struct BlockQ5K {
    d: f16,
    dmin: f16,
    scales: [u8; K_SCALE_SIZE],
    qh: [u8; QK_K / 8],
    qs: [u8; QK_K / 2],
}

impl BlockQ5K {
    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(176);
        out.extend_from_slice(&self.d.to_le_bytes());
        out.extend_from_slice(&self.dmin.to_le_bytes());
        out.extend_from_slice(&self.scales);
        out.extend_from_slice(&self.qh);
        out.extend_from_slice(&self.qs);
        assert_eq!(out.len(), 176);
        out
    }
}

/// Real `q4k_scale_min` (`common.cuh`), ported to Rust -- byte-for-byte
/// identical to ggml's own real `get_scale_min_k4` (fetched from
/// `ggml-quants.c`, 2026-09-11): `if j<4 { q[j]&63, q[j+4]&63 } else {
/// (q[j+4]&0xF)|((q[j-4]>>6)<<4), (q[j+4]>>4)|((q[j-0]>>6)<<4) }`.
fn scale_min_k4(q: &[u8; K_SCALE_SIZE], j: usize) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Pure-Rust reference dequantization, ported from real, freshly-fetched
/// `ggml-quants.c::dequantize_row_q5_K` (2026-09-11) -- NOT re-derived from
/// `deq_q5_K`'s own CUDA source, so a bug shared between the two would not
/// hide here.
fn dequant_q5_k_row(blocks: &[BlockQ5K]) -> Vec<f32> {
    let mut out = Vec::with_capacity(blocks.len() * QK_K);
    for b in blocks {
        let d = b.d.to_f32();
        let dmin = b.dmin.to_f32();
        let mut is = 0usize;
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for group in 0..4 {
            let (sc1, m1) = scale_min_k4(&b.scales, is);
            let (sc2, m2) = scale_min_k4(&b.scales, is + 1);
            let (d1, mm1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, mm2) = (d * sc2 as f32, dmin * m2 as f32);
            let ql = &b.qs[group * 32..group * 32 + 32];
            for (l, &byte) in ql.iter().enumerate() {
                let lo = byte & 0xF;
                let hi_bit = if b.qh[l] & u1 != 0 { 16 } else { 0 };
                out.push(d1 * (lo as i32 + hi_bit) as f32 - mm1);
            }
            for (l, &byte) in ql.iter().enumerate() {
                let hi = byte >> 4;
                let hi_bit = if b.qh[l] & u2 != 0 { 16 } else { 0 };
                out.push(d2 * (hi as i32 + hi_bit) as f32 - mm2);
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    out
}

/// A real, varied (not all-equal) synthetic `block_q5_K`: every scale/min
/// byte distinct, every `qs`/`qh` byte pseudo-random -- a wrong block index,
/// a wrong bit position, or a swapped low/high nibble would all show up as
/// a wrong value, not a coincidentally-right one.
fn synthetic_block(seed: u64) -> BlockQ5K {
    let mut s = seed | 1;
    let mut next = || -> u8 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 24) as u8
    };
    let mut scales = [0u8; K_SCALE_SIZE];
    for v in &mut scales {
        *v = next();
    }
    let mut qh = [0u8; QK_K / 8];
    for v in &mut qh {
        *v = next();
    }
    let mut qs = [0u8; QK_K / 2];
    for v in &mut qs {
        *v = next();
    }
    BlockQ5K {
        d: f16::from_f32(0.037_f32 + (seed % 7) as f32 * 0.011),
        dmin: f16::from_f32(0.013_f32 + (seed % 5) as f32 * 0.004),
        scales,
        qh,
        qs,
    }
}

#[test]
fn dequant_to_f16_matches_the_real_ggml_reference() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    let n_blocks = 5; // several super-blocks, so a block-index bug shows up
    let blocks: Vec<BlockQ5K> = (0..n_blocks).map(|i| synthetic_block(0xC0FFEE + i as u64)).collect();
    let want = dequant_q5_k_row(&blocks);

    let mut raw = Vec::new();
    for b in &blocks {
        raw.extend_from_slice(&b.to_bytes());
    }
    let d_w = stream.clone_htod(&raw)?;

    let n = blocks.len() * QK_K;
    let mut d_out = stream.alloc_zeros::<f16>(n)?;
    k.dequant_to_f16(&mut d_out.as_view_mut(), &d_w.as_view(), WeightType::Q5K, n)?;
    k.device().synchronize()?;

    let got_f16 = stream.clone_dtoh(&d_out)?;
    let got: Vec<f32> = got_f16.iter().map(|v| v.to_f32()).collect();

    let (abs, at) = max_abs_diff(&got, &want);
    // f16 rounding, not slack for the kernel's own bit logic: this
    // synthetic data's real magnitude reaches ~150-200 (d*sc*31 - dmin*m at
    // this test's own scale range), where f16's mantissa step is
    // 2^(7-10)=0.125 and the max round-to-nearest error is half that,
    // 0.0625 -- a real, measured failure here was 0.0586, just under that
    // bound. 0.1 keeps real margin above the f16 ULP without being loose
    // enough to hide an actual bit-logic bug (which would show up as a
    // multi-unit or sign-flipped difference, not a fraction of one ULP).
    assert!(
        abs < 0.1,
        "element {at}: got {} want {} (abs diff {abs})",
        got[at],
        want[at]
    );
    assert!(want.iter().any(|&v| v != 0.0), "reference is all zeros");
    Ok(())
}

#[test]
fn gemv_matches_the_real_ggml_reference() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    // Two rows, two super-blocks each: exercises both the row stride and
    // more than one block per row.
    let (n, kk) = (2usize, 2 * QK_K);
    let blocks: Vec<BlockQ5K> = (0..n * 2).map(|i| synthetic_block(0xBEEF00 + i as u64)).collect();
    let mut raw = Vec::new();
    for b in &blocks {
        raw.extend_from_slice(&b.to_bytes());
    }
    let d_w = stream.clone_htod(&raw)?;

    let x = pseudo_random(kk, 0xACE1);
    let d_x = stream.clone_htod(&x)?;

    let w_dequant = dequant_q5_k_row(&blocks); // [n * kk], row-major
    let want: Vec<f32> = (0..n)
        .map(|row| {
            (0..kk)
                .map(|j| w_dequant[row * kk + j] as f64 * x[j] as f64)
                .sum::<f64>() as f32
        })
        .collect();

    let mut d_out = stream.alloc_zeros::<f32>(n)?;
    k.gemv(&mut d_out.as_view_mut(), &d_w.as_view(), WeightType::Q5K, &d_x.as_view(), kk, n, 1)?;
    k.device().synchronize()?;
    let got = stream.clone_dtoh(&d_out)?;

    let rel = max_rel_diff(&got, &want);
    assert!(rel < 3e-2, "gemv vs reference: rel diff {rel} (got {got:?}, want {want:?})");
    Ok(())
}

#[test]
fn mmvq_agrees_with_gemv_within_q8_1_quantization_noise() -> Result<()> {
    let k = kernels()?;
    if !k.device().caps().int_tensor_gemm {
        eprintln!("skipping: no int8 tensor-core dp4a support on this device");
        return Ok(());
    }
    let stream = k.device().stream().clone();

    // n=8 rows, not 2: a real dot product over 256+ elements can land near
    // zero from cancellation between ~100-magnitude terms even when every
    // term is exactly correct (verified by hand: `tq_dot_q5_K`'s own 16
    // per-(kbx,iqs) terms matched a quantization-aware host reference to 5
    // decimal places on a seed whose row value was ~3.7 against ~150-300
    // magnitude terms). A single such row makes any single-row error metric
    // meaningless; enough rows makes a whole-vector metric meaningful.
    let (n, kk) = (8usize, 2 * QK_K);
    let blocks: Vec<BlockQ5K> = (0..n * 2).map(|i| synthetic_block(0xFACADE + i as u64)).collect();
    let mut raw = Vec::new();
    for b in &blocks {
        raw.extend_from_slice(&b.to_bytes());
    }
    let d_w = stream.clone_htod(&raw)?;

    let x = pseudo_random(kk, 0x51DE);
    let d_x = stream.clone_htod(&x)?;

    let mut d_gemv = stream.alloc_zeros::<f32>(n)?;
    k.gemv(&mut d_gemv.as_view_mut(), &d_w.as_view(), WeightType::Q5K, &d_x.as_view(), kk, n, 1)?;

    let bytes = Kernels::q8_1_bytes(kk);
    let mut d_q8 = stream.alloc_zeros::<u8>(bytes)?;
    k.quantize_q8_1(&mut d_q8.as_view_mut(), &d_x.as_view(), kk)?;

    let mut d_mmvq = stream.alloc_zeros::<f32>(n)?;
    k.mmvq(&mut d_mmvq.as_view_mut(), &d_w.as_view(), WeightType::Q5K, &d_q8.as_view(), kk, n)?;
    k.device().synchronize()?;

    let gemv_out = stream.clone_dtoh(&d_gemv)?;
    let mmvq_out = stream.clone_dtoh(&d_mmvq)?;

    // Q8_1 quantizes the activation to int8 (real, bounded noise -- not a
    // second bit-logic bug to chase); the weight side is exact in both
    // kernels. Cosine over the whole output vector, the same shape
    // `quant.rs`'s `every_quant_type_tracks_the_f16_build` uses to compare a
    // quantized path against a float reference -- robust to the odd row
    // whose true value is a small residual of much larger cancelling terms,
    // unlike any per-element relative metric.
    let cos = cosine(&mmvq_out, &gemv_out);
    assert!(cos > 0.999, "mmvq vs gemv: cosine {cos} (mmvq {mmvq_out:?}, gemv {gemv_out:?})");
    Ok(())
}

/// `mmvq_batch` picks one of five separately-instantiated `mmvqt{T}_q5_K`
/// kernels by token count (`Kernels::mmvq_t`) -- a real, distinct kernel
/// name per `T`, not a single generic launcher, so exercising only
/// `mmvq`'s own single-row path above would leave these five untested.
/// Sweeps every real `T` the dispatcher can choose.
#[test]
fn mmvq_batch_agrees_with_gemv_at_every_real_tile_width() -> Result<()> {
    let k = kernels()?;
    if !k.device().caps().int_tensor_gemm {
        eprintln!("skipping: no int8 tensor-core dp4a support on this device");
        return Ok(());
    }
    let stream = k.device().stream().clone();

    // n=8 rows (see the comment in
    // `mmvq_agrees_with_gemv_within_q8_1_quantization_noise` above for why).
    let (n, kk) = (8usize, 2 * QK_K);
    let blocks: Vec<BlockQ5K> = (0..n * 2).map(|i| synthetic_block(0x5EED10 + i as u64)).collect();
    let mut raw = Vec::new();
    for b in &blocks {
        raw.extend_from_slice(&b.to_bytes());
    }
    let d_w = stream.clone_htod(&raw)?;

    for &n_tokens in &[1usize, 2, 4, 8, 16] {
        let x = pseudo_random(n_tokens * kk, 0x7A0 + n_tokens as u64);
        let d_x = stream.clone_htod(&x)?;

        let mut d_gemv = stream.alloc_zeros::<f32>(n * n_tokens)?;
        for t in 0..n_tokens {
            k.gemv(
                &mut d_gemv.as_view_mut().slice_mut(t * n..(t + 1) * n),
                &d_w.as_view(),
                WeightType::Q5K,
                &d_x.as_view().slice(t * kk..(t + 1) * kk),
                kk,
                n,
                1,
            )?;
        }

        let bytes = Kernels::q8_1_bytes(kk);
        let mut d_q8 = stream.alloc_zeros::<u8>(n_tokens * bytes)?;
        k.quantize_q8_1(&mut d_q8.as_view_mut(), &d_x.as_view(), n_tokens * kk)?;

        let mut d_batch = stream.alloc_zeros::<f32>(n * n_tokens)?;
        k.mmvq_batch(
            &mut d_batch.as_view_mut(),
            &d_w.as_view(),
            WeightType::Q5K,
            &d_q8.as_view(),
            kk,
            n,
            n_tokens,
        )?;
        k.device().synchronize()?;

        let gemv_out = stream.clone_dtoh(&d_gemv)?;
        let batch_out = stream.clone_dtoh(&d_batch)?;
        // Cosine over the whole output vector (see the comment in
        // `mmvq_agrees_with_gemv_within_q8_1_quantization_noise` above): with
        // n=8 rows this seed's own near-zero-cancellation row (n_tokens=1,
        // row 0) is one of 8 (or more, for wider n_tokens) elements rather
        // than the whole vector, so it can no longer dominate the metric the
        // way a per-element relative diff would.
        let cos = cosine(&batch_out, &gemv_out);
        assert!(
            cos > 0.999,
            "n_tokens={n_tokens}: mmvq_batch vs gemv cosine {cos} (batch {batch_out:?}, gemv {gemv_out:?})"
        );
    }
    Ok(())
}

#[test]
fn mmvq_batch_t1_matches_mmvq_single_row_directly() -> Result<()> {
    let k = kernels()?;
    if !k.device().caps().int_tensor_gemm {
        eprintln!("skipping: no int8 tensor-core dp4a support on this device");
        return Ok(());
    }
    let stream = k.device().stream().clone();

    let (n, kk) = (2usize, 2 * QK_K);
    let blocks: Vec<BlockQ5K> = (0..n * 2).map(|i| synthetic_block(0x1234 + i as u64)).collect();
    let mut raw = Vec::new();
    for b in &blocks {
        raw.extend_from_slice(&b.to_bytes());
    }
    let d_w = stream.clone_htod(&raw)?;

    let x = pseudo_random(kk, 0x9999);
    let d_x = stream.clone_htod(&x)?;
    let bytes = Kernels::q8_1_bytes(kk);
    let mut d_q8 = stream.alloc_zeros::<u8>(bytes)?;
    k.quantize_q8_1(&mut d_q8.as_view_mut(), &d_x.as_view(), kk)?;

    let mut d_single = stream.alloc_zeros::<f32>(n)?;
    k.mmvq(&mut d_single.as_view_mut(), &d_w.as_view(), WeightType::Q5K, &d_q8.as_view(), kk, n)?;

    let mut d_batch = stream.alloc_zeros::<f32>(n)?;
    k.mmvq_batch(&mut d_batch.as_view_mut(), &d_w.as_view(), WeightType::Q5K, &d_q8.as_view(), kk, n, 1)?;
    k.device().synchronize()?;

    let single = stream.clone_dtoh(&d_single)?;
    let batch = stream.clone_dtoh(&d_batch)?;
    // Peak-relative, not per-row `single.abs()`: these two kernels read the
    // same q8_1 bytes and should agree almost to the bit, but a row whose
    // true value is a small residual of much larger cancelling terms would
    // still make an ordinary rounding-order difference look huge under a
    // per-row floor.
    let scale = single.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let (abs, at) = max_abs_diff(&batch, &single);
    assert!(
        abs / scale < 1e-4,
        "mmvq single-row vs mmvq_batch(T=1) should be near-identical: peak-relative diff {} at {at} (single {single:?}, batch {batch:?})",
        abs / scale
    );
    Ok(())
}
