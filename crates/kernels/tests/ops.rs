//! Every non-matmul kernel against a CPU reference.

mod common;

use anyhow::Result;
use half::f16;
use infero_kernels::{AttnDims, BatchLayout, Kernels};

use common::*;

#[test]
fn rms_norm_matches_reference() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    let (n_tokens, d) = (7usize, 896usize);
    let x = pseudo_random(n_tokens * d, 0xA1);
    let w = pseudo_random(d, 0xB2)
        .iter()
        .map(|v| v + 1.5)
        .collect::<Vec<_>>();
    let eps = 1e-6;

    let dx = stream.clone_htod(&x)?;
    let dw = stream.clone_htod(&w)?;
    let mut dout = stream.alloc_zeros::<f32>(n_tokens * d)?;

    k.rms_norm(
        &mut dout.as_view_mut(),
        &dx.as_view(),
        &dw.as_view(),
        n_tokens,
        d,
        eps,
    )?;
    let got = stream.clone_dtoh(&dout)?;
    k.device().synchronize()?;

    let want = rms_norm_ref(&x, &w, n_tokens, d, eps);
    let rel = max_rel_diff(&got, &want);
    assert!(rel < 1e-5, "relative error {rel}");
    Ok(())
}

#[test]
fn rope_matches_rotate_half() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    let (n_tokens, n_heads, d_head) = (5usize, 14usize, 64usize);
    let theta = 1_000_000.0f32;
    let x = pseudo_random(n_tokens * n_heads * d_head, 0xC3);
    let positions: Vec<i32> = (0..n_tokens as i32).map(|i| i + 3).collect();

    let mut dx = stream.clone_htod(&x)?;
    let dpos = stream.clone_htod(&positions)?;

    let ones = stream.clone_htod(&vec![1.0f32; d_head / 2])?;
    k.rope(
        &mut dx.as_view_mut(),
        &dpos.as_view(),
        &ones.as_view(),
        n_tokens,
        n_heads,
        d_head,
        theta,
        1.0,
        false,
    )?;
    let got = stream.clone_dtoh(&dx)?;
    k.device().synchronize()?;

    let want = rope_ref(&x, &positions, n_tokens, n_heads, d_head, theta);
    // __sincosf and __powf are the fast-math intrinsics; a few ulps of slack.
    let (abs, at) = max_abs_diff(&got, &want);
    assert!(abs < 2e-4, "max abs diff {abs} at {at}");
    Ok(())
}

/// The fused Q+K launch against the single-tensor kernel, which is the one
/// pinned to the CPU reference above.
///
/// `rope_qk` splits `blockIdx.y` across two tensors with different head counts,
/// so the row a lane lands on is a function of which half of the grid it is in;
/// that indexing is the whole of what the fused form adds. Both pairings, and a
/// head count for Q that is not a multiple of K's, so a mixed-up head index
/// cannot land on the right row by accident.
#[test]
fn rope_qk_matches_the_single_tensor_form() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    let (n_tokens, n_heads, n_kv_heads, d_head) = (5usize, 14usize, 3usize, 128usize);
    let theta = 500_000.0f32;
    let q = pseudo_random(n_tokens * n_heads * d_head, 0x11);
    let v = pseudo_random(n_tokens * n_kv_heads * d_head, 0x22);
    let positions: Vec<i32> = (0..n_tokens as i32).map(|i| i * 7 + 1).collect();

    let dpos = stream.clone_htod(&positions)?;
    // Not all ones: a dropped `freq_factors` read is invisible at 1.0.
    let ff = pseudo_random(d_head / 2, 0x33)
        .iter()
        .map(|x| x.abs() + 0.5)
        .collect::<Vec<_>>();
    let dff = stream.clone_htod(&ff)?;
    let axis0 = scalar_axis(&stream, d_head / 2)?;

    for interleaved in [false, true] {
        let (mut sq, mut sk) = (stream.clone_htod(&q)?, stream.clone_htod(&v)?);
        k.rope(
            &mut sq.as_view_mut(),
            &dpos.as_view(),
            &dff.as_view(),
            n_tokens,
            n_heads,
            d_head,
            theta,
            1.0,
            interleaved,
        )?;
        k.rope(
            &mut sk.as_view_mut(),
            &dpos.as_view(),
            &dff.as_view(),
            n_tokens,
            n_kv_heads,
            d_head,
            theta,
            1.0,
            interleaved,
        )?;
        let (want_q, want_k) = (stream.clone_dtoh(&sq)?, stream.clone_dtoh(&sk)?);

        let (mut fq, mut fk) = (stream.clone_htod(&q)?, stream.clone_htod(&v)?);
        k.rope_qk(
            &mut fq.as_view_mut(),
            &mut fk.as_view_mut(),
            &dpos.as_view(),
            &dff.as_view(),
            &axis0.as_view(),
            1,
            n_tokens,
            n_heads,
            n_kv_heads,
            d_head,
            theta,
            1.0,
            interleaved,
        )?;
        let (got_q, got_k) = (stream.clone_dtoh(&fq)?, stream.clone_dtoh(&fk)?);
        k.device().synchronize()?;

        // Same arithmetic on the same inputs, so the only slack allowed is the
        // compiler contracting a multiply-add differently between the two.
        let (aq, at) = max_abs_diff(&got_q, &want_q);
        assert!(aq < 1e-6, "pairing {interleaved}: q differs by {aq} at {at}");
        let (ak, at) = max_abs_diff(&got_k, &want_k);
        assert!(ak < 1e-6, "pairing {interleaved}: k differs by {ak} at {at}");
    }
    Ok(())
}

/// The two pairings must differ, and each must be an isometry — a wrong choice
/// is not detectable by norms alone, only by comparing against the right one.
#[test]
fn the_two_rope_pairings_differ_but_both_preserve_norms() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let (n_tokens, n_heads, d_head) = (1usize, 2usize, 64usize);
    let x = pseudo_random(n_heads * d_head, 0xE1);
    let dpos = stream.clone_htod(&vec![7i32])?;
    let ones = stream.clone_htod(&vec![1.0f32; d_head / 2])?;

    let mut out = Vec::new();
    for interleaved in [false, true] {
        let mut dx = stream.clone_htod(&x)?;
        k.rope(
            &mut dx.as_view_mut(),
            &dpos.as_view(),
            &ones.as_view(),
            n_tokens,
            n_heads,
            d_head,
            500_000.0,
            1.0,
            interleaved,
        )?;
        let host = stream.clone_dtoh(&dx)?;
        k.device().synchronize()?;
        // Rotation preserves each head's norm whichever way the pairs are cut.
        for h in 0..n_heads {
            let before: f32 = x[h * d_head..(h + 1) * d_head].iter().map(|v| v * v).sum();
            let after: f32 = host[h * d_head..(h + 1) * d_head]
                .iter()
                .map(|v| v * v)
                .sum();
            assert!(
                (before - after).abs() / before < 1e-5,
                "pairing {interleaved} changed the norm"
            );
        }
        out.push(host);
    }
    assert!(
        max_abs_diff(&out[0], &out[1]).0 > 1e-3,
        "the two pairings produced the same thing, so one of them is not wired up"
    );
    Ok(())
}

/// Llama 3.1 divides each dimension's frequency by a stored factor. Doubling a
/// factor must have exactly the effect of halving that dimension's position.
#[test]
fn rope_frequency_scaling_stretches_a_dimension() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    let (n_tokens, n_heads, d_head) = (1usize, 1usize, 64usize);
    let theta = 500_000.0f32;
    let x = pseudo_random(n_heads * d_head, 0xF1);
    let positions = vec![64i32];
    let dpos = stream.clone_htod(&positions)?;

    // Factor two on every dimension.
    let mut halved = stream.clone_htod(&x)?;
    let twos = stream.clone_htod(&vec![2.0f32; d_head / 2])?;
    k.rope(
        &mut halved.as_view_mut(),
        &dpos.as_view(),
        &twos.as_view(),
        n_tokens,
        n_heads,
        d_head,
        theta,
        1.0,
        false,
    )?;

    // The same thing, expressed as half the position and no scaling.
    let half_pos = stream.clone_htod(&vec![32i32])?;
    let mut direct = stream.clone_htod(&x)?;
    let ones = stream.clone_htod(&vec![1.0f32; d_head / 2])?;
    k.rope(
        &mut direct.as_view_mut(),
        &half_pos.as_view(),
        &ones.as_view(),
        n_tokens,
        n_heads,
        d_head,
        theta,
        1.0,
        false,
    )?;

    let a = stream.clone_dtoh(&halved)?;
    let b = stream.clone_dtoh(&direct)?;
    k.device().synchronize()?;
    let (abs, at) = max_abs_diff(&a, &b);
    assert!(
        abs < 2e-4,
        "scaling by 2 differs from halving the position: {abs} at {at}"
    );

    // And it must actually change something relative to no scaling at all.
    let mut plain = stream.clone_htod(&x)?;
    k.rope(
        &mut plain.as_view_mut(),
        &dpos.as_view(),
        &ones.as_view(),
        n_tokens,
        n_heads,
        d_head,
        theta,
        1.0,
        false,
    )?;
    let c = stream.clone_dtoh(&plain)?;
    k.device().synchronize()?;
    assert!(max_abs_diff(&a, &c).0 > 1e-3, "scaling had no effect");
    Ok(())
}

/// The fused-row form against the two-tensor one it replaces.
///
/// Running `gate` and `up` as one matmul makes a row `2 * d_ff` wide with the
/// two operands `d_ff` apart *inside* a row rather than in separate tensors, so
/// the indexing is the whole of what this kernel adds — and an off-by-a-row
/// version still produces finite, plausible activations.
#[test]
fn silu_mul_split_matches_the_two_tensor_form() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    let (tokens, d_ff) = (5usize, 4864usize);
    let gate = pseudo_random(tokens * d_ff, 0xD4);
    let up = pseudo_random(tokens * d_ff, 0xE5);
    // Interleaved the way one matmul against `concat_t(gate, up)` writes it.
    let mut fused = vec![0f32; tokens * 2 * d_ff];
    for t in 0..tokens {
        fused[t * 2 * d_ff..t * 2 * d_ff + d_ff]
            .copy_from_slice(&gate[t * d_ff..(t + 1) * d_ff]);
        fused[t * 2 * d_ff + d_ff..(t + 1) * 2 * d_ff]
            .copy_from_slice(&up[t * d_ff..(t + 1) * d_ff]);
    }

    let dxy = stream.clone_htod(&fused)?;
    let mut dout = stream.alloc_zeros::<f32>(tokens * d_ff)?;
    k.silu_mul_split(
        &mut dout.as_view_mut(),
        &dxy.as_view(),
        d_ff,
        tokens * d_ff,
    )?;
    let got = stream.clone_dtoh(&dout)?;
    k.device().synchronize()?;

    let want = silu_mul_ref(&gate, &up);
    let (abs, at) = max_abs_diff(&got, &want);
    assert!(abs < 1e-5, "max abs diff {abs} at {at}");
    Ok(())
}

#[test]
fn silu_mul_matches_reference() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    let n = 4864 * 3;
    let gate = pseudo_random(n, 0xD4);
    let up = pseudo_random(n, 0xE5);

    let dg = stream.clone_htod(&gate)?;
    let du = stream.clone_htod(&up)?;
    let mut dout = stream.alloc_zeros::<f32>(n)?;

    k.silu_mul(&mut dout.as_view_mut(), &dg.as_view(), &du.as_view(), n)?;
    let got = stream.clone_dtoh(&dout)?;
    k.device().synchronize()?;

    let want = silu_mul_ref(&gate, &up);
    let (abs, at) = max_abs_diff(&got, &want);
    // __expf is the fast intrinsic; the operands are O(1) so absolute error
    // is the meaningful bound here.
    assert!(abs < 1e-5, "max abs diff {abs} at {at}");
    Ok(())
}

#[test]
fn add_and_bias() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    let (n_rows, n_cols) = (6usize, 512usize);
    let a = pseudo_random(n_rows * n_cols, 0xF6);
    let b = pseudo_random(n_rows * n_cols, 0x17);
    let bias = pseudo_random(n_cols, 0x28);

    let da = stream.clone_htod(&a)?;
    let db = stream.clone_htod(&b)?;
    let dbias = stream.clone_htod(&bias)?;
    let mut dout = stream.alloc_zeros::<f32>(n_rows * n_cols)?;

    k.add(
        &mut dout.as_view_mut(),
        &da.as_view(),
        &db.as_view(),
        n_rows * n_cols,
    )?;
    k.add_bias(&mut dout.as_view_mut(), &dbias.as_view(), n_cols, n_rows)?;
    let got = stream.clone_dtoh(&dout)?;
    k.device().synchronize()?;

    for r in 0..n_rows {
        for (c, &bias_c) in bias.iter().enumerate() {
            let i = r * n_cols + c;
            let want = a[i] + b[i] + bias_c;
            assert!((got[i] - want).abs() < 1e-6, "at ({r},{c})");
        }
    }
    Ok(())
}

#[test]
fn attention_matches_reference() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    // Grouped-query shape: 14 query heads over 2 kv heads, as in Qwen2.5-0.5B.
    let dims = AttnDims {
        n_heads: 14,
        n_kv_heads: 2,
        d_head: 64,
        n_slots: 128,
        n_tokens: 5,
    };
    let kv_len = 9usize;
    let scale = 1.0 / (dims.d_head as f32).sqrt();

    let q = pseudo_random(dims.n_tokens * dims.n_heads * dims.d_head, 0x39);
    let kv_elems = dims.n_kv_heads * dims.n_slots * dims.d_head;
    let k_host: Vec<f16> = pseudo_random(kv_elems, 0x4A)
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let v_host: Vec<f16> = pseudo_random(kv_elems, 0x5B)
        .into_iter()
        .map(f16::from_f32)
        .collect();
    // Tokens 0..4 sit at absolute positions 4..8, so each masks a different
    // slice of the 9-entry history.
    let positions: Vec<i32> = (0..dims.n_tokens as i32).map(|i| i + 4).collect();

    let dq = stream.clone_htod(&q)?;
    let dk = stream.clone_htod(&k_host)?;
    let dv = stream.clone_htod(&v_host)?;
    let dpos = stream.clone_htod(&positions)?;
    // One sequence whose logical positions map straight onto pool slots.
    let seq_of = vec![0i32; dims.n_tokens];
    let table: Vec<i32> = (0..dims.n_slots as i32).collect();
    let dseq = stream.clone_htod(&seq_of)?;
    let dtable = stream.clone_htod(&table)?;
    let (vseq, vpos, vtable) = (dseq.as_view(), dpos.as_view(), dtable.as_view());
    let batch = BatchLayout {
        seq_of: &vseq,
        positions: &vpos,
        slot_table: &vtable,
        table_stride: dims.n_slots,
    };
    let mut dscores = stream.alloc_zeros::<f32>(dims.n_heads * dims.n_tokens * kv_len)?;
    let mut dout = stream.alloc_zeros::<f32>(dims.n_tokens * dims.n_heads * dims.d_head)?;

    k.attn_scores(
        &mut dscores.as_view_mut(),
        &dq.as_view(),
        &dk.as_view(),
        batch,
        dims,
        kv_len,
        scale,
    )?;
    k.attn_softmax(
        &mut dscores.as_view_mut(),
        dims.n_heads,
        dims.n_tokens,
        kv_len,
    )?;
    k.attn_output(
        &mut dout.as_view_mut(),
        &dscores.as_view(),
        &dv.as_view(),
        batch,
        dims,
        kv_len,
        None,
    )?;

    let got_scores = stream.clone_dtoh(&dscores)?;
    let got = stream.clone_dtoh(&dout)?;
    k.device().synchronize()?;

    let gqa = dims.n_heads / dims.n_kv_heads;
    let mut want = vec![0.0f32; got.len()];
    for h in 0..dims.n_heads {
        let kvh = h / gqa;
        for t in 0..dims.n_tokens {
            let qrow = &q[(t * dims.n_heads + h) * dims.d_head..][..dims.d_head];

            let mut row = vec![f32::NEG_INFINITY; kv_len];
            for (j, slot) in row.iter_mut().enumerate() {
                if j as i32 > positions[t] {
                    continue;
                }
                let kbase = (kvh * dims.n_slots + j) * dims.d_head;
                let dot: f32 = (0..dims.d_head)
                    .map(|i| qrow[i] * k_host[kbase + i].to_f32())
                    .sum();
                *slot = dot * scale;
            }
            let probs = softmax_ref(&row);

            let start = (h * dims.n_tokens + t) * kv_len;
            let rel = max_rel_diff(&got_scores[start..start + kv_len], &probs);
            assert!(rel < 1e-4, "softmax head {h} token {t}: rel {rel}");

            for i in 0..dims.d_head {
                let mut acc = 0.0f32;
                for (j, p) in probs.iter().enumerate() {
                    acc += p * v_host[(kvh * dims.n_slots + j) * dims.d_head + i].to_f32();
                }
                want[(t * dims.n_heads + h) * dims.d_head + i] = acc;
            }
        }
    }

    let (abs, at) = max_abs_diff(&got, &want);
    assert!(abs < 1e-4, "attention output diff {abs} at {at}");
    Ok(())
}

/// Bit-level decode of the e4m3 format `attn_e4m3_encode`/`f32_to_e4m3`
/// write -- unambiguous (unlike encode, decode needs no rounding decision),
/// ported from `e4m3_to_f32` in `fp8.cu` so the reference below can use the
/// GPU's own quantized bytes directly instead of re-deriving them.
fn e4m3_to_f32_host(b: u8) -> f32 {
    let sign = (b & 0x80) != 0;
    let exp = ((b >> 3) & 0x0F) as i32;
    let man = (b & 0x07) as i32;
    let v = if exp == 0 {
        man as f32 / 512.0
    } else if exp == 0x0F && man == 0x07 {
        return f32::NAN;
    } else {
        f32::from_bits(((exp + 120) as u32) << 23 | ((man as u32) << 20))
    };
    if sign { -v } else { v }
}

/// `attn_prefill_e4m3k_f32` against a reference that isolates this kernel's
/// own correctness (MMA fragment layout, causal masking, online-softmax
/// bookkeeping, PV) from e4m3 quantization accuracy, which is validated
/// separately (`examples/e4m3_qk_accuracy_probe.rs`, ~3% of a score-std).
/// Both Q and K are quantized to e4m3 on the GPU first (`quantize_k_e4m3`,
/// reused for Q by treating its `n_heads` as the "kv_head" dimension --
/// same `[position, head, d_head]` shape), decoded back to f32 on the host,
/// and the reference computes exact causal attention on *those* dequantized
/// values against full-precision V -- the same values, bit for bit, this
/// kernel's own e4m3 arithmetic is built from. A real bug in this kernel's
/// own logic should show up here even though the quantization noise itself
/// is expected and already characterized elsewhere.
#[test]
fn attn_prefill_e4m3k_matches_a_quantized_reference() -> Result<()> {
    // (n_heads, n_kv_heads, n_tokens): d_head is fixed at 256, the only
    // width this validation kernel's register arrays support. Covers this
    // checkpoint's real GQA ratio (24/4) alongside a smaller one, and
    // token counts landing exactly on `ATTN_E4M3_WK`'s 48-key boundary,
    // just past it, and well past it with a remainder.
    for &(n_heads, n_kv_heads, n_tokens) in &[
        (8usize, 2usize, 48usize),
        (8, 2, 49),
        (8, 2, 130),
        (24, 4, 97),
    ] {
        attn_prefill_e4m3k_case(n_heads, n_kv_heads, n_tokens)?;
    }
    Ok(())
}

fn attn_prefill_e4m3k_case(n_heads: usize, n_kv_heads: usize, n_tokens: usize) -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    let d_head = 256;
    // A fresh single-shot prefill: kv_len == n_tokens.
    let kv_len = n_tokens;
    let scale = 1.0 / (d_head as f32).sqrt();

    let q = pseudo_random(n_tokens * n_heads * d_head, 0x71);
    let k_host = pseudo_random(kv_len * n_kv_heads * d_head, 0x82);
    let v_host: Vec<f16> = pseudo_random(kv_len * n_kv_heads * d_head, 0x93)
        .into_iter()
        .map(f16::from_f32)
        .collect();

    let dq = stream.clone_htod(&q)?;
    let dk = stream.clone_htod(&k_host)?;
    let dv = stream.clone_htod(&v_host)?;

    let mut dkq = stream.alloc_zeros::<u8>(kv_len * n_kv_heads * d_head)?;
    let mut dkscale = stream.alloc_zeros::<f32>(kv_len * n_kv_heads)?;
    k.quantize_k_e4m3(
        &mut dkq.as_view_mut(),
        &mut dkscale.as_view_mut(),
        &dk.as_view(),
        kv_len,
        n_kv_heads,
        d_head,
    )?;

    let mut dqq = stream.alloc_zeros::<u8>(n_tokens * n_heads * d_head)?;
    let mut dqscale = stream.alloc_zeros::<f32>(n_tokens * n_heads)?;
    k.quantize_k_e4m3(
        &mut dqq.as_view_mut(),
        &mut dqscale.as_view_mut(),
        &dq.as_view(),
        n_tokens,
        n_heads,
        d_head,
    )?;

    let kq_bytes = stream.clone_dtoh(&dkq)?;
    let kscale_host = stream.clone_dtoh(&dkscale)?;
    let qq_bytes = stream.clone_dtoh(&dqq)?;
    let qscale_host = stream.clone_dtoh(&dqscale)?;
    k.device().synchronize()?;

    let dequant = |bytes: &[u8], scales: &[f32], n_rows: usize, n_h: usize| -> Vec<f32> {
        let mut out = vec![0f32; n_rows * n_h * d_head];
        for r in 0..n_rows {
            for h in 0..n_h {
                let s = scales[r * n_h + h];
                for i in 0..d_head {
                    let b = bytes[(r * n_h + h) * d_head + i];
                    out[(r * n_h + h) * d_head + i] = s * e4m3_to_f32_host(b);
                }
            }
        }
        out
    };
    let q_deq = dequant(&qq_bytes, &qscale_host, n_tokens, n_heads);
    let k_deq = dequant(&kq_bytes, &kscale_host, kv_len, n_kv_heads);

    let group = n_heads / n_kv_heads;
    let mut want = vec![0f32; n_tokens * n_heads * d_head];
    for t in 0..n_tokens {
        for h in 0..n_heads {
            let kv_head = h / group;
            let last = t.min(kv_len - 1);
            let mut scores = vec![0f32; last + 1];
            let mut m = f32::NEG_INFINITY;
            for (kpos, slot) in scores.iter_mut().enumerate() {
                let mut s = 0f32;
                for d in 0..d_head {
                    s += q_deq[(t * n_heads + h) * d_head + d]
                        * k_deq[(kpos * n_kv_heads + kv_head) * d_head + d];
                }
                s *= scale;
                *slot = s;
                m = m.max(s);
            }
            let probs = softmax_ref(&scores);
            for d in 0..d_head {
                let mut acc = 0f32;
                for (kpos, p) in probs.iter().enumerate() {
                    acc += p * v_host[(kpos * n_kv_heads + kv_head) * d_head + d].to_f32();
                }
                want[(t * n_heads + h) * d_head + d] = acc;
            }
        }
    }

    let dims = AttnDims { n_heads, n_kv_heads, d_head, n_slots: 0, n_tokens };
    let mut dout = stream.alloc_zeros::<f32>(n_tokens * n_heads * d_head)?;
    let mut dpart = stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(n_heads, d_head, n_tokens))?;
    k.attn_prefill_e4m3k(
        &mut dout.as_view_mut(),
        &dq.as_view(),
        &dkq.as_view(),
        &dkscale.as_view(),
        &dv.as_view(),
        dims,
        0,
        n_tokens,
        kv_len,
        scale,
        &mut dpart.as_view_mut(),
    )?;
    let got = stream.clone_dtoh(&dout)?;
    k.device().synchronize()?;

    // Real measured max abs diff on this shape is ~1.9e-4 (fp16-rounding-
    // level noise from the unchanged PV/output path, not from this kernel's
    // own e4m3 QK^T logic) -- the relative bound is looser only because
    // relative error blows up near-zero values, the same artifact the
    // accuracy probe's own "|score| > 1e-3" filter exists to avoid.
    let (abs, at) = max_abs_diff(&got, &want);
    let rel = max_rel_diff(&got, &want);
    assert!(
        abs < 2e-3,
        "n_heads={n_heads} n_kv_heads={n_kv_heads} n_tokens={n_tokens}: max abs diff {abs} at {at}, max rel diff {rel}"
    );
    Ok(())
}

/// `attn_prefill_e4m3k_f32` called twice with growing `run_base`/`kv_len`
/// (the real chunked-prefill usage pattern `attn_prefill_pipe_bench.rs`
/// exercises for `ws4`), against the same single-shot reference the
/// non-chunked test above uses -- causal attention's output for a given
/// token depends only on that token and everything at or before it, so
/// chunking must not change the result. This is the test that would have
/// caught a `run_base` addressing mistake the single-call test above
/// cannot: with `run_base` always 0 there, an off-by-`run_base` bug in Q or
/// output indexing would be invisible.
#[test]
fn attn_prefill_e4m3k_chunked_matches_the_single_shot_case() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    let n_heads = 8;
    let n_kv_heads = 2;
    let d_head = 256;
    let n_tokens = 130usize;
    let kv_len = n_tokens;
    let scale = 1.0 / (d_head as f32).sqrt();

    let q = pseudo_random(n_tokens * n_heads * d_head, 0x71);
    let k_host = pseudo_random(kv_len * n_kv_heads * d_head, 0x82);
    let v_host: Vec<f16> = pseudo_random(kv_len * n_kv_heads * d_head, 0x93)
        .into_iter()
        .map(f16::from_f32)
        .collect();

    let dq = stream.clone_htod(&q)?;
    let dk = stream.clone_htod(&k_host)?;
    let dv = stream.clone_htod(&v_host)?;

    let mut dkq = stream.alloc_zeros::<u8>(kv_len * n_kv_heads * d_head)?;
    let mut dkscale = stream.alloc_zeros::<f32>(kv_len * n_kv_heads)?;
    k.quantize_k_e4m3(
        &mut dkq.as_view_mut(),
        &mut dkscale.as_view_mut(),
        &dk.as_view(),
        kv_len,
        n_kv_heads,
        d_head,
    )?;
    let mut dqq = stream.alloc_zeros::<u8>(n_tokens * n_heads * d_head)?;
    let mut dqscale = stream.alloc_zeros::<f32>(n_tokens * n_heads)?;
    k.quantize_k_e4m3(
        &mut dqq.as_view_mut(),
        &mut dqscale.as_view_mut(),
        &dq.as_view(),
        n_tokens,
        n_heads,
        d_head,
    )?;
    let kq_bytes = stream.clone_dtoh(&dkq)?;
    let kscale_host = stream.clone_dtoh(&dkscale)?;
    let qq_bytes = stream.clone_dtoh(&dqq)?;
    let qscale_host = stream.clone_dtoh(&dqscale)?;
    k.device().synchronize()?;

    let dequant = |bytes: &[u8], scales: &[f32], n_rows: usize, n_h: usize| -> Vec<f32> {
        let mut out = vec![0f32; n_rows * n_h * d_head];
        for r in 0..n_rows {
            for h in 0..n_h {
                let s = scales[r * n_h + h];
                for i in 0..d_head {
                    let b = bytes[(r * n_h + h) * d_head + i];
                    out[(r * n_h + h) * d_head + i] = s * e4m3_to_f32_host(b);
                }
            }
        }
        out
    };
    let q_deq = dequant(&qq_bytes, &qscale_host, n_tokens, n_heads);
    let k_deq = dequant(&kq_bytes, &kscale_host, kv_len, n_kv_heads);

    let group = n_heads / n_kv_heads;
    let mut want = vec![0f32; n_tokens * n_heads * d_head];
    for t in 0..n_tokens {
        for h in 0..n_heads {
            let kv_head = h / group;
            let last = t.min(kv_len - 1);
            let mut scores = vec![0f32; last + 1];
            let mut m = f32::NEG_INFINITY;
            for (kpos, slot) in scores.iter_mut().enumerate() {
                let mut s = 0f32;
                for d in 0..d_head {
                    s += q_deq[(t * n_heads + h) * d_head + d]
                        * k_deq[(kpos * n_kv_heads + kv_head) * d_head + d];
                }
                s *= scale;
                *slot = s;
                m = m.max(s);
            }
            let probs = softmax_ref(&scores);
            for d in 0..d_head {
                let mut acc = 0f32;
                for (kpos, p) in probs.iter().enumerate() {
                    acc += p * v_host[(kpos * n_kv_heads + kv_head) * d_head + d].to_f32();
                }
                want[(t * n_heads + h) * d_head + d] = acc;
            }
        }
    }

    // Two chunks: [0, 55) then [55, 130), `run_base`/`kv_len` growing --
    // deliberately not aligned to `ATTN_E4M3_WK`'s 48 or any tile boundary.
    let dims = AttnDims { n_heads, n_kv_heads, d_head, n_slots: 0, n_tokens };
    let mut dout = stream.alloc_zeros::<f32>(n_tokens * n_heads * d_head)?;
    let mut dpart = stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(n_heads, d_head, n_tokens))?;
    for &(run_base, run_tokens) in &[(0usize, 55usize), (55, 130 - 55)] {
        k.attn_prefill_e4m3k(
            &mut dout.as_view_mut(),
            &dq.as_view(),
            &dkq.as_view(),
            &dkscale.as_view(),
            &dv.as_view(),
            dims,
            run_base,
            run_tokens,
            run_base + run_tokens,
            scale,
            &mut dpart.as_view_mut(),
        )?;
    }
    let got = stream.clone_dtoh(&dout)?;
    k.device().synchronize()?;

    let (abs, at) = max_abs_diff(&got, &want);
    let rel = max_rel_diff(&got, &want);
    assert!(abs < 2e-3, "chunked run: max abs diff {abs} at {at}, max rel diff {rel}");
    Ok(())
}

#[test]
fn masked_positions_get_no_weight() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    let dims = AttnDims {
        n_heads: 2,
        n_kv_heads: 2,
        d_head: 64,
        n_slots: 32,
        n_tokens: 1,
    };
    let kv_len = 8usize;

    let q = pseudo_random(dims.n_heads * dims.d_head, 0x6C);
    let kv: Vec<f16> = pseudo_random(dims.n_kv_heads * dims.n_slots * dims.d_head, 0x7D)
        .into_iter()
        .map(f16::from_f32)
        .collect();
    // The single query sits at position 2: entries 3..7 are the future.
    let positions = vec![2i32];

    let dq = stream.clone_htod(&q)?;
    let dk = stream.clone_htod(&kv)?;
    let dpos = stream.clone_htod(&positions)?;
    let dseq = stream.clone_htod(&vec![0i32; dims.n_tokens])?;
    let dtable = stream.clone_htod(&(0..dims.n_slots as i32).collect::<Vec<_>>())?;
    let (vseq, vpos, vtable) = (dseq.as_view(), dpos.as_view(), dtable.as_view());
    let batch = BatchLayout {
        seq_of: &vseq,
        positions: &vpos,
        slot_table: &vtable,
        table_stride: dims.n_slots,
    };
    let mut dscores = stream.alloc_zeros::<f32>(dims.n_heads * kv_len)?;

    k.attn_scores(
        &mut dscores.as_view_mut(),
        &dq.as_view(),
        &dk.as_view(),
        batch,
        dims,
        kv_len,
        1.0,
    )?;
    k.attn_softmax(&mut dscores.as_view_mut(), dims.n_heads, 1, kv_len)?;
    let got = stream.clone_dtoh(&dscores)?;
    k.device().synchronize()?;

    for h in 0..dims.n_heads {
        let row = &got[h * kv_len..(h + 1) * kv_len];
        for (j, p) in row.iter().enumerate() {
            if j > 2 {
                assert_eq!(*p, 0.0, "head {h} attended to future position {j}");
            } else {
                assert!(*p > 0.0, "head {h} ignored visible position {j}");
            }
        }
        let sum: f32 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "head {h} probabilities sum to {sum}"
        );
    }
    Ok(())
}

#[test]
fn store_kv_lands_at_the_right_positions() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    let (n_kv_heads, d_head, n_slots, n_tokens) = (2usize, 64usize, 16usize, 3usize);
    let src = pseudo_random(n_tokens * n_kv_heads * d_head, 0x8E);
    // Deliberately not consecutive: the pool hands out whatever is free.
    let slots = vec![5i32, 11, 2];

    let dsrc = stream.clone_htod(&src)?;
    let dslots = stream.clone_htod(&slots)?;
    let mut cache = stream.alloc_zeros::<f16>(n_kv_heads * n_slots * d_head)?;

    k.store_kv(
        &mut cache.as_view_mut(),
        &dsrc.as_view(),
        &dslots.as_view(),
        n_kv_heads,
        d_head,
        n_slots,
        n_tokens,
    )?;
    let got = stream.clone_dtoh(&cache)?;
    k.device().synchronize()?;

    for (t, &slot) in slots.iter().enumerate() {
        for h in 0..n_kv_heads {
            for i in 0..d_head {
                let want = src[(t * n_kv_heads + h) * d_head + i];
                let cached = got[(h * n_slots + slot as usize) * d_head + i].to_f32();
                assert!(
                    (cached - want).abs() < 1e-3,
                    "token {t} head {h} dim {i}: {cached} vs {want}"
                );
            }
        }
    }
    // Slots nobody wrote must stay zero.
    assert_eq!(got[0].to_f32(), 0.0);
    Ok(())
}

/// Both halves in one launch against two calls of the kernel above, bit for bit.
///
/// `store_kv2` cuts `blockIdx.y` in half to choose a pool, so K's head 0 and V's
/// head 0 are two different rows reached from the same index — the failure this
/// pins is one half landing in the other's pool, which leaves every value in
/// range and plausible. A `d_head` that is not a multiple of the vector width
/// covers the scalar tail, and a negative slot covers a padded batch row: the
/// scheduler hands those to the kernel and expects them dropped, not written.
#[test]
fn store_kv2_matches_two_store_kv_calls() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    for (n_kv_heads, d_head) in [(8usize, 128usize), (2, 12)] {
        let (n_slots, n_tokens) = (16usize, 5usize);
        let ks = pseudo_random(n_tokens * n_kv_heads * d_head, 0x51);
        let vs = pseudo_random(n_tokens * n_kv_heads * d_head, 0x62);
        // Out of order, and one padded row the kernel must leave alone.
        let slots = vec![9i32, 2, -1, 14, 5];

        let dk = stream.clone_htod(&ks)?;
        let dv = stream.clone_htod(&vs)?;
        let dslots = stream.clone_htod(&slots)?;

        let n = n_kv_heads * n_slots * d_head;
        let (mut wk, mut wv) = (stream.alloc_zeros::<f16>(n)?, stream.alloc_zeros::<f16>(n)?);
        for (cache, src) in [(&mut wk, &dk), (&mut wv, &dv)] {
            k.store_kv(
                &mut cache.as_view_mut(),
                &src.as_view(),
                &dslots.as_view(),
                n_kv_heads,
                d_head,
                n_slots,
                n_tokens,
            )?;
        }
        let (want_k, want_v) = (stream.clone_dtoh(&wk)?, stream.clone_dtoh(&wv)?);

        let (mut gk, mut gv) = (stream.alloc_zeros::<f16>(n)?, stream.alloc_zeros::<f16>(n)?);
        k.store_kv2(
            &mut gk.as_view_mut(),
            &mut gv.as_view_mut(),
            &dk.as_view(),
            &dv.as_view(),
            &dslots.as_view(),
            n_kv_heads,
            d_head,
            n_slots,
            n_tokens,
        )?;
        let (got_k, got_v) = (stream.clone_dtoh(&gk)?, stream.clone_dtoh(&gv)?);
        k.device().synchronize()?;

        assert_eq!(got_k, want_k, "k pool differs at d_head {d_head}");
        assert_eq!(got_v, want_v, "v pool differs at d_head {d_head}");
        // Not vacuous: the two pools must not have ended up identical, and the
        // dropped token's slot must still be untouched.
        assert_ne!(got_k, got_v);
        for h in 0..n_kv_heads {
            for i in 0..d_head {
                assert_eq!(got_k[(h * n_slots + 3) * d_head + i].to_f32(), 0.0);
            }
        }
    }
    Ok(())
}

/// The fused decode kernel against the three it replaces.
///
/// The three-kernel path is itself pinned to a CPU reference above, so this is
/// the whole of what `attn_decode` has to match. The shapes that matter are the
/// awkward ones: a chunk boundary that does not divide the history, a history
/// that does not fill its last 32-key tile, tokens at different positions so
/// the mask cuts each block's chunk differently, and a slot table that
/// interleaves sequences the way a pool under load does.
#[test]
fn attn_decode_matches_the_three_kernels() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    // The last one is not a shape a decode step runs; it is there because
    // `decode_chunks` stops splitting once the grid is wide enough, and that
    // turns on a different exit — the kernel normalizes and writes the answer
    // itself instead of handing partials to the combine pass. 8 KV heads by 128
    // tokens is 1024 blocks, past `sm_count * 4` on anything this runs on.
    // Three shapes of model, not one: Llama-3.1's 4 query heads a KV head over
    // 128 dimensions, Qwen2.5-0.5B's *seven* over 64, and Qwen3.8-27B-FP8's 6
    // over 256 -- the widest `d_head` `attn_decode_mma_f32`'s register arrays
    // are sized for (`ATTN_MMA_MAX_D` in ops.cu). A group that does not divide
    // the MMA's sixteen rows, and a head that does not fill its lanes, are
    // exactly where an off-by-one in a fragment index hides — and where the
    // batching tests caught one that these shapes had not.
    for (n_heads, n_kv_heads, d_head) in [(8usize, 2usize, 128usize), (14, 2, 64), (24, 4, 256)] {
    for (n_tokens, kv_len) in [
        (5usize, 100usize),
        (3, 32),
        (4, 33),
        (2, 7),
        (6, 256),
        (128, 100),
        (64, 64),
        (33, 40),
        // A real chunked prefill's tail: kv_len in the tens of thousands,
        // which none of the shapes above reach anywhere near.
        (256, 30000),
    ] {
        let dims = AttnDims {
            n_heads,
            n_kv_heads,
            d_head,
            n_slots: 512,
            n_tokens,
        };
        let scale = 1.0 / (dims.d_head as f32).sqrt();
        let q = pseudo_random(n_tokens * dims.n_heads * dims.d_head, 0x71);
        let kv_elems = dims.n_kv_heads * dims.n_slots * dims.d_head;
        let kh: Vec<f16> = pseudo_random(kv_elems, 0x82)
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let vh: Vec<f16> = pseudo_random(kv_elems, 0x93)
            .into_iter()
            .map(f16::from_f32)
            .collect();
        // Two masks, because they stress different exits. The decode-ish one
        // puts every token a different distance into its own history; the
        // prefill one is a single sequence at positions 0..n-1, where most
        // tokens' histories end *before* a later chunk begins and a block has
        // to record "nothing here" rather than compute it.
        let prefill = n_tokens > kv_len / 2;
        let positions: Vec<i32> = if prefill {
            (0..n_tokens as i32).map(|t| t.min(kv_len as i32 - 1)).collect()
        } else {
            (0..n_tokens)
                .map(|t| (kv_len as i32 - 1 - (t as i32 * 7) % (kv_len as i32)).max(0))
                .collect()
        };
        // Sequences interleaved through the pool, not laid out in runs.
        let seq_of: Vec<i32> = if prefill {
            vec![0i32; n_tokens]
        } else {
            (0..n_tokens as i32).map(|t| t % 2).collect()
        };
        let table: Vec<i32> = (0..2)
            .flat_map(|s| (0..kv_len).map(move |p| ((p * 2 + s) % 512) as i32))
            .collect();

        let dq = stream.clone_htod(&q)?;
        let dk = stream.clone_htod(&kh)?;
        let dv = stream.clone_htod(&vh)?;
        let dpos = stream.clone_htod(&positions)?;
        let dseq = stream.clone_htod(&seq_of)?;
        let dtable = stream.clone_htod(&table)?;
        let (vseq, vpos, vtable) = (dseq.as_view(), dpos.as_view(), dtable.as_view());
        let batch = BatchLayout {
            seq_of: &vseq,
            positions: &vpos,
            slot_table: &vtable,
            table_stride: kv_len,
        };

        let mut dscores = stream.alloc_zeros::<f32>(dims.n_heads * n_tokens * kv_len)?;
        let out_len = n_tokens * dims.n_heads * dims.d_head;
        let mut want_d = stream.alloc_zeros::<f32>(out_len)?;
        k.attn_scores(
            &mut dscores.as_view_mut(),
            &dq.as_view(),
            &dk.as_view(),
            batch,
            dims,
            kv_len,
            scale,
        )?;
        k.attn_softmax(&mut dscores.as_view_mut(), dims.n_heads, n_tokens, kv_len)?;
        k.attn_output(
            &mut want_d.as_view_mut(),
            &dscores.as_view(),
            &dv.as_view(),
            batch,
            dims,
            kv_len,
            None,
        )?;
        let want = stream.clone_dtoh(&want_d)?;

        let mut got_d = stream.alloc_zeros::<f32>(out_len)?;
        let mut part = stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(
            dims.n_heads,
            dims.d_head,
            n_tokens,
        ))?;
        k.attn_decode(
            &mut got_d.as_view_mut(),
            None,
            &dq.as_view(),
            &dk.as_view(),
            &dv.as_view(),
            batch,
            dims,
            kv_len,
            scale,
            &mut part.as_view_mut(),
        )?;
        let got = stream.clone_dtoh(&got_d)?;
        k.device().synchronize()?;

        // Both sum the same products in different orders, and the weights come
        // from `__expf` either way; the tolerance is for the reassociation.
        //
        // The tensor-core path gets ten times the slack, and it needs it for a
        // reason worth stating: `mma.m16n8k16` takes f16 operands, so the
        // softmax weights are rounded to half before the value product where the
        // scalar kernel keeps them in f32. FlashAttention does the same. It costs
        // about 6e-4 relative here, and it is why that path is not the default —
        // see the note on `attn_decode_mma_f32`.
        //
        // Mirrors `Kernels::attn_decode`'s own selection: the env var opts in,
        // and only up to `d_head` 256 -- past that is disabled outright, see
        // the long comment at that gate for why.
        const ATTN_MMA_MAX_D_HEAD: usize = 256;
        let mma_active =
            std::env::var("INFERO_ATTN_MMA").as_deref() == Ok("1") && d_head <= ATTN_MMA_MAX_D_HEAD;
        let tol = if mma_active { 2e-3 } else { 2e-4 };
        let (abs, at) = max_abs_diff(&got, &want);
        assert!(
            abs < tol,
            "{n_heads}q/{n_kv_heads}kv x {d_head}, tokens {n_tokens} kv {kv_len}: \
             max abs diff {abs} at {at} (got {}, want {})",
            got[at],
            want[at]
        );
        // Not vacuous: an all-zero output would pass any difference test.
        assert!(want.iter().any(|v| v.abs() > 1e-3), "reference is all zeros");
    }
    }
    Ok(())
}

/// [`Kernels::attn_prefill`] against the same three-kernel reference,
/// restricted to the one shape it is allowed to see: a single contiguous,
/// single-sequence run. Requires `INFERO_ATTN_MMA=1` in the environment
/// *before the test binary starts* — the gate reads it through a `OnceLock`
/// shared by every test in this file, so setting it from inside a test can
/// lose a race against whichever test's attention call runs first.
///
/// The run is embedded inside a larger fake batch (`pad` tokens of an
/// unrelated sequence on each side, at unrelated KV slots) rather than run
/// alone, so a bug that reads past `[run_base, run_base + run_tokens)` — the
/// exact multi-tenant hazard the kernel's doc comment warns a cross-sequence
/// tile would create — shows up as a wrong answer here instead of only in
/// production traffic.
#[test]
fn attn_prefill_matches_the_three_kernels() -> Result<()> {
    if std::env::var("INFERO_ATTN_MMA").as_deref() != Ok("1") {
        eprintln!("skipping: needs INFERO_ATTN_MMA=1 set before the test binary starts");
        return Ok(());
    }
    let k = kernels()?;
    let stream = k.device().stream().clone();

    // 256 with d_head 256 is Qwen3.8-27B-FP8's own shape; 4 and 7 are groups
    // that do not divide sixteen evenly, matching the decode test's choices.
    // group=8 (16, 2) is `prefill_attention`'s own upper bound and the one
    // shape where `tpw*group` (16) is exactly the MMA fragment's row count
    // instead of less than it -- the boundary `attn_prefill_ws`'s output
    // staging buffer sizing has to get right, not just the common case.
    for (n_heads, n_kv_heads, d_head) in [(24usize, 4usize, 256usize), (8, 2, 128), (14, 2, 64), (16, 2, 256)] {
        let dims_probe = AttnDims { n_heads, n_kv_heads, d_head, n_slots: 1, n_tokens: 1 };
        if !k.prefill_attention(&dims_probe) {
            continue;
        }
        // Tile-boundary shapes: `tile_tokens` for `group * 2 <= 16` is
        // `4 * (16 / group)`. Exercise under, at, and past one tile, plus a
        // long run that forces `n_chunks > 1`.
        for (run_tokens, kv_len) in [
            (1usize, 40usize),
            (2, 33),
            (7, 33),
            (16, 100),
            (17, 100),
            (63, 4000),
            (256, 30000),
        ] {
            let pad = 5usize;
            let n_tokens = pad + run_tokens + pad;
            let n_slots = 512usize;
            let dims = AttnDims { n_heads, n_kv_heads, d_head, n_slots, n_tokens };
            let scale = 1.0 / (dims.d_head as f32).sqrt();
            let q = pseudo_random(n_tokens * dims.n_heads * dims.d_head, 0x71);
            let kv_elems = dims.n_kv_heads * n_slots * dims.d_head;
            let kh: Vec<f16> = pseudo_random(kv_elems, 0x82).into_iter().map(f16::from_f32).collect();
            let vh: Vec<f16> = pseudo_random(kv_elems, 0x93).into_iter().map(f16::from_f32).collect();

            // Sequence 0 owns the padding on both sides, at positions and
            // slots that share nothing with sequence 1's run in the middle —
            // any read that strays outside the run picks up sequence 0's
            // data and the comparison below catches it.
            let mut seq_of = vec![0i32; n_tokens];
            let mut positions = vec![0i32; n_tokens];
            for t in 0..n_tokens {
                if t >= pad && t < pad + run_tokens {
                    seq_of[t] = 1;
                    positions[t] = (t - pad) as i32;
                } else {
                    positions[t] = (200 + t) as i32 % (n_slots as i32);
                }
            }
            let table: Vec<i32> = (0..2)
                .flat_map(|s| (0..kv_len.max(n_slots)).map(move |p| ((p * 2 + s + 1) % n_slots) as i32))
                .collect();
            let table_stride = kv_len.max(n_slots);

            let dq = stream.clone_htod(&q)?;
            let dk = stream.clone_htod(&kh)?;
            let dv = stream.clone_htod(&vh)?;
            let dpos = stream.clone_htod(&positions)?;
            let dseq = stream.clone_htod(&seq_of)?;
            let dtable = stream.clone_htod(&table)?;
            let (vseq, vpos, vtable) = (dseq.as_view(), dpos.as_view(), dtable.as_view());
            let batch = BatchLayout {
                seq_of: &vseq,
                positions: &vpos,
                slot_table: &vtable,
                table_stride,
            };

            // Reference: the three-kernel path over the *whole* fake batch,
            // then slice out the run's rows.
            let mut dscores = stream.alloc_zeros::<f32>(dims.n_heads * n_tokens * kv_len)?;
            let out_len = n_tokens * dims.n_heads * dims.d_head;
            let mut want_d = stream.alloc_zeros::<f32>(out_len)?;
            k.attn_scores(&mut dscores.as_view_mut(), &dq.as_view(), &dk.as_view(), batch, dims, kv_len, scale)?;
            k.attn_softmax(&mut dscores.as_view_mut(), dims.n_heads, n_tokens, kv_len)?;
            k.attn_output(&mut want_d.as_view_mut(), &dscores.as_view(), &dv.as_view(), batch, dims, kv_len, None)?;
            let want = stream.clone_dtoh(&want_d)?;

            // The kernel under test: only the run's slice of `out` gets
            // written, so seed it with the reference's *own* padding rows
            // and only overwrite the run — a bug that leaves the run
            // untouched would otherwise slip through as a false pass.
            let mut got_d = stream.clone_htod(&want)?;
            let mut part = stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(
                dims.n_heads,
                dims.d_head,
                run_tokens,
            ))?;
            k.attn_prefill(
                &mut got_d.as_view_mut(),
                &dq.as_view(),
                &dk.as_view(),
                &dv.as_view(),
                batch,
                dims,
                pad,
                run_tokens,
                kv_len,
                scale,
                &mut part.as_view_mut(),
            )?;
            let got = stream.clone_dtoh(&got_d)?;
            k.device().synchronize()?;

            // Same tolerance as the decode MMA path: f16 softmax weights on
            // the way into the value product.
            let run_lo = pad * dims.n_heads * dims.d_head;
            let run_hi = (pad + run_tokens) * dims.n_heads * dims.d_head;
            let (abs, at) = max_abs_diff(&got[run_lo..run_hi], &want[run_lo..run_hi]);
            assert!(
                abs < 2e-3,
                "{n_heads}q/{n_kv_heads}kv x {d_head}, run {run_tokens} kv {kv_len}: \
                 max abs diff {abs} at {at} (got {}, want {})",
                got[run_lo + at],
                want[run_lo + at]
            );
            assert!(
                want[run_lo..run_hi].iter().any(|v| v.abs() > 1e-3),
                "reference is all zeros"
            );
            // Padding rows outside the run must come back untouched.
            assert_eq!(
                &got[..run_lo],
                &want[..run_lo],
                "{n_heads}q/{n_kv_heads}kv x {d_head}, run {run_tokens} kv {kv_len}: \
                 wrote before the run"
            );
            assert_eq!(
                &got[run_hi..],
                &want[run_hi..],
                "{n_heads}q/{n_kv_heads}kv x {d_head}, run {run_tokens} kv {kv_len}: \
                 wrote past the run"
            );

            // Same reference, the cp.async-pipelined kernel under test.
            let mut got_pipe_d = stream.clone_htod(&want)?;
            let mut part_pipe = stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(
                dims.n_heads,
                dims.d_head,
                run_tokens,
            ))?;
            k.attn_prefill_pipe(
                &mut got_pipe_d.as_view_mut(),
                &dq.as_view(),
                &dk.as_view(),
                &dv.as_view(),
                batch,
                dims,
                pad,
                run_tokens,
                kv_len,
                scale,
                &mut part_pipe.as_view_mut(),
            )?;
            let got_pipe = stream.clone_dtoh(&got_pipe_d)?;
            k.device().synchronize()?;
            let (abs_p, at_p) = max_abs_diff(&got_pipe[run_lo..run_hi], &want[run_lo..run_hi]);
            assert!(
                abs_p < 2e-3,
                "attn_prefill_pipe {n_heads}q/{n_kv_heads}kv x {d_head}, run {run_tokens} kv {kv_len}: \
                 max abs diff {abs_p} at {at_p} (got {}, want {})",
                got_pipe[run_lo + at_p],
                want[run_lo + at_p]
            );
            assert_eq!(
                &got_pipe[..run_lo],
                &want[..run_lo],
                "attn_prefill_pipe {n_heads}q/{n_kv_heads}kv x {d_head}, run {run_tokens} kv {kv_len}: \
                 wrote before the run"
            );
            assert_eq!(
                &got_pipe[run_hi..],
                &want[run_hi..],
                "attn_prefill_pipe {n_heads}q/{n_kv_heads}kv x {d_head}, run {run_tokens} kv {kv_len}: \
                 wrote past the run"
            );

            // Same reference, the natural-V-layout kernel.
            let mut got_natv_d = stream.clone_htod(&want)?;
            let mut part_natv = stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(
                dims.n_heads,
                dims.d_head,
                run_tokens,
            ))?;
            k.attn_prefill_natv(
                &mut got_natv_d.as_view_mut(),
                &dq.as_view(),
                &dk.as_view(),
                &dv.as_view(),
                batch,
                dims,
                pad,
                run_tokens,
                kv_len,
                scale,
                &mut part_natv.as_view_mut(),
            )?;
            let got_natv = stream.clone_dtoh(&got_natv_d)?;
            k.device().synchronize()?;
            let (abs_nv, at_nv) = max_abs_diff(&got_natv[run_lo..run_hi], &want[run_lo..run_hi]);
            assert!(
                abs_nv < 2e-3,
                "attn_prefill_natv {n_heads}q/{n_kv_heads}kv x {d_head}, run {run_tokens} kv {kv_len}: \
                 max abs diff {abs_nv} at {at_nv} (got {}, want {})",
                got_natv[run_lo + at_nv],
                want[run_lo + at_nv]
            );
            assert_eq!(
                &got_natv[..run_lo],
                &want[..run_lo],
                "attn_prefill_natv {n_heads}q/{n_kv_heads}kv x {d_head}, run {run_tokens} kv {kv_len}: \
                 wrote before the run"
            );
            assert_eq!(
                &got_natv[run_hi..],
                &want[run_hi..],
                "attn_prefill_natv {n_heads}q/{n_kv_heads}kv x {d_head}, run {run_tokens} kv {kv_len}: \
                 wrote past the run"
            );

            // Same reference, K and V both pipelined.
            let mut got_pipev_d = stream.clone_htod(&want)?;
            let mut part_pipev = stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(
                dims.n_heads,
                dims.d_head,
                run_tokens,
            ))?;
            k.attn_prefill_pipev(
                &mut got_pipev_d.as_view_mut(),
                &dq.as_view(),
                &dk.as_view(),
                &dv.as_view(),
                batch,
                dims,
                pad,
                run_tokens,
                kv_len,
                scale,
                &mut part_pipev.as_view_mut(),
            )?;
            let got_pipev = stream.clone_dtoh(&got_pipev_d)?;
            k.device().synchronize()?;
            let (abs_pv, at_pv) = max_abs_diff(&got_pipev[run_lo..run_hi], &want[run_lo..run_hi]);
            assert!(
                abs_pv < 2e-3,
                "attn_prefill_pipev {n_heads}q/{n_kv_heads}kv x {d_head}, run {run_tokens} kv {kv_len}: \
                 max abs diff {abs_pv} at {at_pv} (got {}, want {})",
                got_pipev[run_lo + at_pv],
                want[run_lo + at_pv]
            );
            assert_eq!(
                &got_pipev[..run_lo],
                &want[..run_lo],
                "attn_prefill_pipev {n_heads}q/{n_kv_heads}kv x {d_head}, run {run_tokens} kv {kv_len}: \
                 wrote before the run"
            );
            assert_eq!(
                &got_pipev[run_hi..],
                &want[run_hi..],
                "attn_prefill_pipev {n_heads}q/{n_kv_heads}kv x {d_head}, run {run_tokens} kv {kv_len}: \
                 wrote past the run"
            );

            // Same reference, the warp-specialized producer/consumer kernel.
            let mut got_ws_d = stream.clone_htod(&want)?;
            let mut part_ws = stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(
                dims.n_heads,
                dims.d_head,
                run_tokens,
            ))?;
            k.attn_prefill_ws(
                &mut got_ws_d.as_view_mut(),
                &dq.as_view(),
                &dk.as_view(),
                &dv.as_view(),
                batch,
                dims,
                pad,
                run_tokens,
                kv_len,
                scale,
                &mut part_ws.as_view_mut(),
            )?;
            let got_ws = stream.clone_dtoh(&got_ws_d)?;
            k.device().synchronize()?;
            let (abs_ws, at_ws) = max_abs_diff(&got_ws[run_lo..run_hi], &want[run_lo..run_hi]);
            assert!(
                abs_ws < 2e-3,
                "attn_prefill_ws {n_heads}q/{n_kv_heads}kv x {d_head}, run {run_tokens} kv {kv_len}: \
                 max abs diff {abs_ws} at {at_ws} (got {}, want {})",
                got_ws[run_lo + at_ws],
                want[run_lo + at_ws]
            );
            assert_eq!(
                &got_ws[..run_lo],
                &want[..run_lo],
                "attn_prefill_ws {n_heads}q/{n_kv_heads}kv x {d_head}, run {run_tokens} kv {kv_len}: \
                 wrote before the run"
            );
            assert_eq!(
                &got_ws[run_hi..],
                &want[run_hi..],
                "attn_prefill_ws {n_heads}q/{n_kv_heads}kv x {d_head}, run {run_tokens} kv {kv_len}: \
                 wrote past the run"
            );

            // Same reference, the no-`sq` / 48-key-wide-tile architectural
            // variant.
            let mut got_ws4_d = stream.clone_htod(&want)?;
            let mut part_ws4 = stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(
                dims.n_heads,
                dims.d_head,
                run_tokens,
            ))?;
            k.attn_prefill_ws4(
                &mut got_ws4_d.as_view_mut(),
                &dq.as_view(),
                &dk.as_view(),
                &dv.as_view(),
                batch,
                dims,
                pad,
                run_tokens,
                kv_len,
                scale,
                &mut part_ws4.as_view_mut(),
            )?;
            let got_ws4 = stream.clone_dtoh(&got_ws4_d)?;
            k.device().synchronize()?;
            let (abs_ws4, at_ws4) = max_abs_diff(&got_ws4[run_lo..run_hi], &want[run_lo..run_hi]);
            assert!(
                abs_ws4 < 2e-3,
                "attn_prefill_ws4 {n_heads}q/{n_kv_heads}kv x {d_head}, run {run_tokens} kv {kv_len}: \
                 max abs diff {abs_ws4} at {at_ws4} (got {}, want {})",
                got_ws4[run_lo + at_ws4],
                want[run_lo + at_ws4]
            );
            assert_eq!(
                &got_ws4[..run_lo],
                &want[..run_lo],
                "attn_prefill_ws4 {n_heads}q/{n_kv_heads}kv x {d_head}, run {run_tokens} kv {kv_len}: \
                 wrote before the run"
            );
            assert_eq!(
                &got_ws4[run_hi..],
                &want[run_hi..],
                "attn_prefill_ws4 {n_heads}q/{n_kv_heads}kv x {d_head}, run {run_tokens} kv {kv_len}: \
                 wrote past the run"
            );
        }
    }
    Ok(())
}

/// The two-kernel option-3 split (`attn_prefill_stats_f32` + `attn_prefill_pv_f32`,
/// dispatched together by `Kernels::attn_prefill_split`) against the same
/// three-kernel reference the fused kernels above check against. Only
/// single-chunk shapes: the split path doesn't support the multi-chunk
/// partial-reduce case (see `attn_prefill_split`'s own `ensure!`).
#[test]
fn attn_prefill_split_matches_the_reference() -> Result<()> {
    if std::env::var("INFERO_ATTN_MMA").as_deref() != Ok("1") {
        eprintln!("skipping: needs INFERO_ATTN_MMA=1 set before the test binary starts");
        return Ok(());
    }
    let k = kernels()?;
    let stream = k.device().stream().clone();

    for (n_heads, n_kv_heads, d_head) in [(24usize, 4usize, 256usize), (8, 2, 128), (14, 2, 64), (16, 2, 256)] {
        let dims_probe = AttnDims { n_heads, n_kv_heads, d_head, n_slots: 1, n_tokens: 1 };
        if !k.prefill_attention(&dims_probe) {
            continue;
        }
        // `kv_len <= 32` stays single-chunk (`prefill_chunks` always rounds
        // its chunk size up to a multiple of 32); `4000`/`30000` force
        // multiple chunks, exercising `attn_ms_reduce_f32` and
        // `attn_pv_sum_reduce_f32`, not just the single-chunk direct-write
        // path.
        for (run_tokens, kv_len) in [(1usize, 20usize), (2, 25), (7, 30), (16, 32), (17, 32), (63, 4000), (256, 30000)] {
            let pad = 5usize;
            let n_tokens = pad + run_tokens + pad;
            let n_slots = 512usize;
            let dims = AttnDims { n_heads, n_kv_heads, d_head, n_slots, n_tokens };
            let scale = 1.0 / (dims.d_head as f32).sqrt();
            let q = pseudo_random(n_tokens * dims.n_heads * dims.d_head, 0x71);
            let kv_elems = dims.n_kv_heads * n_slots * dims.d_head;
            let kh: Vec<f16> = pseudo_random(kv_elems, 0x82).into_iter().map(f16::from_f32).collect();
            let vh: Vec<f16> = pseudo_random(kv_elems, 0x93).into_iter().map(f16::from_f32).collect();

            let mut seq_of = vec![0i32; n_tokens];
            let mut positions = vec![0i32; n_tokens];
            for t in 0..n_tokens {
                if t >= pad && t < pad + run_tokens {
                    seq_of[t] = 1;
                    positions[t] = (t - pad) as i32;
                } else {
                    positions[t] = (200 + t) as i32 % (n_slots as i32);
                }
            }
            let table: Vec<i32> = (0..2)
                .flat_map(|s| (0..kv_len.max(n_slots)).map(move |p| ((p * 2 + s + 1) % n_slots) as i32))
                .collect();
            let table_stride = kv_len.max(n_slots);

            let dq = stream.clone_htod(&q)?;
            let dk = stream.clone_htod(&kh)?;
            let dv = stream.clone_htod(&vh)?;
            let dpos = stream.clone_htod(&positions)?;
            let dseq = stream.clone_htod(&seq_of)?;
            let dtable = stream.clone_htod(&table)?;
            let (vseq, vpos, vtable) = (dseq.as_view(), dpos.as_view(), dtable.as_view());
            let batch = BatchLayout {
                seq_of: &vseq,
                positions: &vpos,
                slot_table: &vtable,
                table_stride,
            };

            let mut dscores = stream.alloc_zeros::<f32>(dims.n_heads * n_tokens * kv_len)?;
            let out_len = n_tokens * dims.n_heads * dims.d_head;
            let mut want_d = stream.alloc_zeros::<f32>(out_len)?;
            k.attn_scores(&mut dscores.as_view_mut(), &dq.as_view(), &dk.as_view(), batch, dims, kv_len, scale)?;
            k.attn_softmax(&mut dscores.as_view_mut(), dims.n_heads, n_tokens, kv_len)?;
            k.attn_output(&mut want_d.as_view_mut(), &dscores.as_view(), &dv.as_view(), batch, dims, kv_len, None)?;
            let want = stream.clone_dtoh(&want_d)?;

            let mut got_d = stream.clone_htod(&want)?;
            let mut ms = stream.alloc_zeros::<f32>(Kernels::attn_ms_floats(dims.n_heads, run_tokens))?;
            let mut part_split = stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(
                dims.n_heads,
                dims.d_head,
                run_tokens,
            ))?;
            k.attn_prefill_split(
                &mut got_d.as_view_mut(),
                &dq.as_view(),
                &dk.as_view(),
                &dv.as_view(),
                batch,
                dims,
                pad,
                run_tokens,
                kv_len,
                scale,
                &mut ms.as_view_mut(),
                &mut part_split.as_view_mut(),
            )?;
            let got = stream.clone_dtoh(&got_d)?;
            k.device().synchronize()?;

            let run_lo = pad * dims.n_heads * dims.d_head;
            let run_hi = (pad + run_tokens) * dims.n_heads * dims.d_head;
            let (abs, at) = max_abs_diff(&got[run_lo..run_hi], &want[run_lo..run_hi]);
            assert!(
                abs < 2e-3,
                "attn_prefill_split {n_heads}q/{n_kv_heads}kv x {d_head}, run {run_tokens} kv {kv_len}: \
                 max abs diff {abs} at {at} (got {}, want {})",
                got[run_lo + at],
                want[run_lo + at]
            );
            assert_eq!(
                &got[..run_lo],
                &want[..run_lo],
                "attn_prefill_split {n_heads}q/{n_kv_heads}kv x {d_head}, run {run_tokens} kv {kv_len}: \
                 wrote before the run"
            );
            assert_eq!(
                &got[run_hi..],
                &want[run_hi..],
                "attn_prefill_split {n_heads}q/{n_kv_heads}kv x {d_head}, run {run_tokens} kv {kv_len}: \
                 wrote past the run"
            );
        }
    }
    Ok(())
}


// TEMP, not for the deployed tree: see `Kernels::attn_ws_pair_probe`'s own
// note.
#[test]
fn attn_ws_pair_probe_matches_closed_form() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let iters = 5000i32;
    let mut d_out = stream.alloc_zeros::<f32>(4)?;
    k.attn_ws_pair_probe(&mut d_out.as_view_mut(), iters)?;
    k.device().synchronize()?;
    let got = stream.clone_dtoh(&d_out)?;
    let n = (iters - 1) as f64;
    let want0 = (n * (n + 1.0) / 2.0) as f32;
    let want1 = want0 * 2.0;
    for pair in 0..2 {
        assert_eq!(got[pair * 2], want0, "pair {pair} acc0");
        assert_eq!(got[pair * 2 + 1], want1, "pair {pair} acc1");
    }
    Ok(())
}

// TEMP, not for the deployed tree: does ptxas actually reuse registers
// across two sequential, non-overlapping live ranges (see
// reg_phase_probe_overlap_f32/reg_phase_probe_phased_f32's own comment)?
// This is the load-bearing question for whether splitting
// attn_prefill_mma_ws_f32 into a QK^T/softmax phase and a P@V phase, handed
// off through a small scratch buffer, could actually lower its peak
// register count -- checked before spending real effort building that
// kernel, the same "validate the mechanism in isolation first" discipline
// as attn_ws_pair_probe.
#[test]
fn reg_phase_probe_shows_whether_ptxas_reuses_registers_across_phases() -> Result<()> {
    let k = kernels()?;
    let (overlap_regs, _) = k.kernel_registers("infero_ops", "reg_phase_probe_overlap_f32")?;
    let (phased_regs, _) = k.kernel_registers("infero_ops", "reg_phase_probe_phased_f32")?;
    eprintln!("reg_phase_probe: overlap={overlap_regs} regs, phased={phased_regs} regs");
    Ok(())
}
