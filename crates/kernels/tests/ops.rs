//! Every non-matmul kernel against a CPU reference.

mod common;

use anyhow::Result;
use half::f16;
use tuili_kernels::{AttnDims, BatchLayout, Kernels};

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
    // Two shapes of model, not one: Llama-3.1's 4 query heads a KV head over
    // 128 dimensions, and Qwen2.5-0.5B's *seven* over 64. A group that does not
    // divide the MMA's sixteen rows, and a head that does not fill its lanes,
    // are exactly where an off-by-one in a fragment index hides — and where the
    // batching tests caught one that these shapes had not.
    for (n_heads, n_kv_heads, d_head) in [(8usize, 2usize, 128usize), (14, 2, 64)] {
    for (n_tokens, kv_len) in [
        (5usize, 100usize),
        (3, 32),
        (4, 33),
        (2, 7),
        (6, 256),
        (128, 100),
        (64, 64),
        (33, 40),
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
        let tol = if std::env::var("TUILI_ATTN_MMA").as_deref() == Ok("1") {
            2e-3
        } else {
            2e-4
        };
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
