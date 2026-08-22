//! The GatedDeltaNet kernels, against the host reference in
//! `tuili_model::qwen35`.
//!
//! The reference is not a reimagining of these kernels — it is checked, stage by
//! stage, against a capture of `transformers`' own Qwen3.5 implementation
//! running on the real 27B checkpoint (`crates/model/tests/qwen35_capture.rs`).
//! So agreement here means the kernel matches the checkpoint, not that two
//! readings of the same guess agree with each other. That distinction is the
//! whole reason the reference exists.
//!
//! Three things are checked beyond "the numbers match": that a sequence's state
//! carries correctly when its tokens arrive in separate calls, that two
//! sequences in one batch do not touch each other's state, and that the
//! alternative reading of each layout choice would have failed.

mod common;

use anyhow::Result;
use common::*;
use tuili_model::qwen35;

/// The 27B's linear-attention shape, small enough to run quickly: the real
/// checkpoint has 16 key heads, 48 value heads and dk = dv = 128.
const DK: usize = 128;
const DV: usize = 128;
const KEY_HEADS: usize = 4;
const VAL_HEADS: usize = 12;
const REP: usize = VAL_HEADS / KEY_HEADS;
const CONV_K: usize = 4;

/// Expand q or k from key heads out to value heads the way `repeat_interleave`
/// does: value head `h` reads key head `h / rep`. The modular expansion
/// (`h % key_heads`) also runs and gives a different model.
fn repeat_interleave(x: &[f32], t_len: usize, key_heads: usize, rep: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; t_len * key_heads * rep * d];
    for t in 0..t_len {
        for h in 0..key_heads * rep {
            let src = h / rep;
            let from = (t * key_heads + src) * d;
            let to = (t * key_heads * rep + h) * d;
            out[to..to + d].copy_from_slice(&x[from..from + d]);
        }
    }
    out
}

#[test]
fn the_conv_matches_the_reference_and_carries_its_window() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let channels = 2 * KEY_HEADS * DK + VAL_HEADS * DV;
    let t_len = 9;

    let x = pseudo_random(t_len * channels, 0x51de);
    let w = pseudo_random(channels * CONV_K, 0xc047);

    let mut want = qwen35::depthwise_causal_conv1d(&x, &w, t_len, channels, CONV_K);
    for v in want.iter_mut() {
        *v = qwen35::silu(*v);
    }

    let first = [0i32];
    let ntok = [t_len as i32];
    let d_x = stream.clone_htod(&x)?;
    let d_w = stream.clone_htod(&w)?;
    let d_first = stream.clone_htod(&first)?;
    let d_ntok = stream.clone_htod(&ntok)?;
    let mut d_out = stream.alloc_zeros::<f32>(t_len * channels)?;
    let mut d_state = stream.alloc_zeros::<f32>(channels * (CONV_K - 1))?;

    let seqs = tuili_kernels::gdn::SeqLayout {
        first_token: &d_first.as_view(),
        n_tokens: &d_ntok.as_view(),
        n_seqs: 1,
        total_tokens: t_len,
    };
    k.gdn_conv(
        &mut d_out.as_view_mut(),
        &d_x.as_view(),
        &mut d_state.as_view_mut(),
        &d_w.as_view(),
        &seqs,
        channels,
        CONV_K,
    )?;
    k.device().synchronize()?;
    let got = stream.clone_dtoh(&d_out)?;
    let (worst, at) = max_abs_diff(&got, &want);
    assert!(
        worst < 2e-5,
        "conv disagreed by {worst:.2e} at {at}: got {}, reference {}",
        got[at],
        want[at]
    );

    // Reversed taps must not match, or this says nothing about the direction —
    // and a reversed convolution shifts the whole model one token forward
    // without failing anywhere.
    let mut flipped = w.clone();
    for row in flipped.chunks_mut(CONV_K) {
        row.reverse();
    }
    let rev = qwen35::depthwise_causal_conv1d(&x, &flipped, t_len, channels, CONV_K);
    let differs = rev
        .iter()
        .zip(&want)
        .any(|(a, b)| (qwen35::silu(*a) - b).abs() > 1e-4);
    assert!(differs, "reversing the taps changed nothing");

    // Split the same sequence across two calls; the carried window must make
    // the second half identical to the whole-sequence answer. This is the
    // property that makes a decode step legitimate.
    let split = 5;
    let mut d_state2 = stream.alloc_zeros::<f32>(channels * (CONV_K - 1))?;
    let mut streamed = vec![0.0f32; t_len * channels];
    for (start, count) in [(0usize, split), (split, t_len - split)] {
        let f = stream.clone_htod(&[0i32])?;
        let n = stream.clone_htod(&[count as i32])?;
        let chunk = stream.clone_htod(&x[start * channels..(start + count) * channels])?;
        let mut piece = stream.alloc_zeros::<f32>(count * channels)?;
        let s = tuili_kernels::gdn::SeqLayout {
            first_token: &f.as_view(),
            n_tokens: &n.as_view(),
            n_seqs: 1,
            total_tokens: count,
        };
        k.gdn_conv(
            &mut piece.as_view_mut(),
            &chunk.as_view(),
            &mut d_state2.as_view_mut(),
            &d_w.as_view(),
            &s,
            channels,
            CONV_K,
        )?;
        k.device().synchronize()?;
        streamed[start * channels..(start + count) * channels]
            .copy_from_slice(&stream.clone_dtoh(&piece)?);
    }
    let (worst, at) = max_abs_diff(&streamed, &want);
    assert!(
        worst < 2e-5,
        "splitting the sequence changed the conv output by {worst:.2e} at {at}; \
         the carried window is wrong, which would make decode disagree with \
         prefill"
    );
    Ok(())
}

#[test]
fn the_gate_and_decay_match_the_reference() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let t_len = 7;
    let heads = VAL_HEADS;

    let a = pseudo_random(t_len * heads, 0xa11);
    let b = pseudo_random(t_len * heads, 0xb22);
    // A_log spans the checkpoint's real range, and dt_bias reaches +19 there,
    // which is what the softplus branch exists for.
    let a_log: Vec<f32> = (0..heads).map(|h| -5.5 + h as f32 * 0.4).collect();
    let dt_bias: Vec<f32> = (0..heads)
        .map(|h| if h == 0 { 19.25 } else { -5.7 + h as f32 * 1.3 })
        .collect();

    let d_a = stream.clone_htod(&a)?;
    let d_b = stream.clone_htod(&b)?;
    let d_al = stream.clone_htod(&a_log)?;
    let d_dt = stream.clone_htod(&dt_bias)?;
    let mut d_beta = stream.alloc_zeros::<f32>(t_len * heads)?;
    let mut d_g = stream.alloc_zeros::<f32>(t_len * heads)?;
    k.gdn_gate_decay(
        &mut d_beta.as_view_mut(),
        &mut d_g.as_view_mut(),
        &d_a.as_view(),
        &d_b.as_view(),
        &d_al.as_view(),
        &d_dt.as_view(),
        t_len,
        heads,
    )?;
    k.device().synchronize()?;

    let got_beta = stream.clone_dtoh(&d_beta)?;
    let got_g = stream.clone_dtoh(&d_g)?;
    for t in 0..t_len {
        for h in 0..heads {
            let i = t * heads + h;
            let want_beta = qwen35::sigmoid(b[i]);
            let want_g = -a_log[h].exp() * qwen35::softplus(a[i] + dt_bias[h]);
            assert!(
                (got_beta[i] - want_beta).abs() < 1e-6,
                "beta[{t},{h}]: {} vs {want_beta}",
                got_beta[i]
            );
            assert!(
                (got_g[i] - want_g).abs() < 1e-5 * want_g.abs().max(1e-3),
                "g[{t},{h}]: {} vs {want_g}",
                got_g[i]
            );
        }
    }
    assert!(
        got_g.iter().all(|v| *v <= 0.0),
        "g must be non-positive so that exp(g) decays the state"
    );
    // The head-0 entry exercises the large-z softplus branch: check it did not
    // overflow into infinity, which is what a plain log(1+exp(z)) would do.
    assert!(got_g[0].is_finite(), "the large dt_bias overflowed");
    Ok(())
}

/// Build one packed `[q | k | v]` row per token, with q and k at key-head width
/// and already normalized the way the reference does. Returns the buffer and
/// `(stride, q_off, k_off, v_off)`.
fn packed(q_small: &[f32], k_small: &[f32], v: &[f32], t_len: usize)
    -> (Vec<f32>, (usize, usize, usize, usize))
{
    let key_dim = KEY_HEADS * DK;
    let val_dim = VAL_HEADS * DV;
    let stride = 2 * key_dim + val_dim;
    let mut qn = q_small.to_vec();
    let mut kn = k_small.to_vec();
    qwen35::l2norm_rows(&mut qn, DK, 1e-6);
    qwen35::l2norm_rows(&mut kn, DK, 1e-6);
    for value in qn.iter_mut() {
        *value *= (DK as f32).sqrt().recip();
    }
    let mut row = vec![0.0f32; t_len * stride];
    for t in 0..t_len {
        let base = t * stride;
        row[base..base + key_dim].copy_from_slice(&qn[t * key_dim..(t + 1) * key_dim]);
        row[base + key_dim..base + 2 * key_dim]
            .copy_from_slice(&kn[t * key_dim..(t + 1) * key_dim]);
        row[base + 2 * key_dim..base + stride]
            .copy_from_slice(&v[t * val_dim..(t + 1) * val_dim]);
    }
    (row, (stride, 0, key_dim, 2 * key_dim))
}

/// The host reference's answer for the same inputs, expanding q and k the way
/// `repeat_interleave` does.
fn reference(
    q_small: &[f32],
    k_small: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    t_len: usize,
) -> (Vec<f32>, Vec<f32>) {
    let q = repeat_interleave(q_small, t_len, KEY_HEADS, REP, DK);
    let kk = repeat_interleave(k_small, t_len, KEY_HEADS, REP, DK);
    let mut state = vec![0.0f32; VAL_HEADS * DK * DV];
    let out = qwen35::gated_delta_rule(
        &q, &kk, v, g, beta, &mut state, t_len, VAL_HEADS, DK, DV, 1e-6,
    );
    (out, state)
}

fn decaying(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    pseudo_random(n, seed).iter().map(|v| -(v.abs()) * scale).collect()
}

fn betas(n: usize, seed: u64) -> Vec<f32> {
    pseudo_random(n, seed).iter().map(|v| qwen35::sigmoid(*v)).collect()
}

#[test]
fn the_delta_rule_matches_the_reference() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let t_len = 11;

    let q_small = pseudo_random(t_len * KEY_HEADS * DK, 0x9001);
    let k_small = pseudo_random(t_len * KEY_HEADS * DK, 0x9002);
    let v = pseudo_random(t_len * VAL_HEADS * DV, 0x9003);
    let g = decaying(t_len * VAL_HEADS, 0x9004, 0.7);
    let beta = betas(t_len * VAL_HEADS, 0x9005);

    let (row, off) = packed(&q_small, &k_small, &v, t_len);
    let (want, want_state) = reference(&q_small, &k_small, &v, &g, &beta, t_len);

    let d_row = stream.clone_htod(&row)?;
    let d_g = stream.clone_htod(&g)?;
    let d_beta = stream.clone_htod(&beta)?;
    let d_first = stream.clone_htod(&[0i32])?;
    let d_ntok = stream.clone_htod(&[t_len as i32])?;
    let mut d_out = stream.alloc_zeros::<f32>(t_len * VAL_HEADS * DV)?;
    let mut d_state = stream.alloc_zeros::<f32>(VAL_HEADS * DK * DV)?;

    let seqs = tuili_kernels::gdn::SeqLayout {
        first_token: &d_first.as_view(),
        n_tokens: &d_ntok.as_view(),
        n_seqs: 1,
        total_tokens: t_len,
    };
    k.gdn_delta_rule(
        &mut d_out.as_view_mut(),
        &mut d_state.as_view_mut(),
        &d_row.as_view(),
        &d_g.as_view(),
        &d_beta.as_view(),
        &seqs,
        VAL_HEADS,
        KEY_HEADS,
        DK,
        DV,
        off,
    )?;
    k.device().synchronize()?;

    let got = stream.clone_dtoh(&d_out)?;
    let got_state = stream.clone_dtoh(&d_state)?;
    let rel = max_rel_diff(&got, &want);
    assert!(rel < 2e-3, "the recurrence output diverged by {rel:.2e}");
    let peak = want_state.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let (worst, at) = max_abs_diff(&got_state, &want_state);
    assert!(
        worst < 1e-4 * peak,
        "the final state diverged by {worst:.2e} at {at} against a peak of \
         {peak:.3e}: got {}, reference {}",
        got_state[at],
        want_state[at]
    );

    // The kernel expands 4 key heads to 12 value heads internally. The modular
    // expansion is the other plausible reading; if it gave the same answer this
    // test would pass under either, so require that it does not.
    let mut mod_q = vec![0.0f32; t_len * VAL_HEADS * DK];
    let mut mod_k = vec![0.0f32; t_len * VAL_HEADS * DK];
    for t in 0..t_len {
        for h in 0..VAL_HEADS {
            let src = h % KEY_HEADS;
            let from = (t * KEY_HEADS + src) * DK;
            let to = (t * VAL_HEADS + h) * DK;
            mod_q[to..to + DK].copy_from_slice(&q_small[from..from + DK]);
            mod_k[to..to + DK].copy_from_slice(&k_small[from..from + DK]);
        }
    }
    let mut other_state = vec![0.0f32; VAL_HEADS * DK * DV];
    let other = qwen35::gated_delta_rule(
        &mod_q, &mod_k, &v, &g, &beta, &mut other_state, t_len, VAL_HEADS, DK, DV, 1e-6,
    );
    let spread = max_rel_diff(&other, &want);
    assert!(
        spread > 1e-2,
        "expanding the key heads modularly instead of by repeat_interleave \
         changed the answer by only {spread:.2e}, so this test would pass \
         either way"
    );
    Ok(())
}

#[test]
fn the_recurrence_carries_state_across_calls() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let t_len = 10;

    let q_small = pseudo_random(t_len * KEY_HEADS * DK, 0x7001);
    let k_small = pseudo_random(t_len * KEY_HEADS * DK, 0x7002);
    let v = pseudo_random(t_len * VAL_HEADS * DV, 0x7003);
    let g = decaying(t_len * VAL_HEADS, 0x7004, 0.5);
    let beta = betas(t_len * VAL_HEADS, 0x7005);
    let (row, off) = packed(&q_small, &k_small, &v, t_len);
    let stride = off.0;

    let run = |chunks: &[(usize, usize)]| -> Result<(Vec<f32>, Vec<f32>)> {
        let mut state = stream.alloc_zeros::<f32>(VAL_HEADS * DK * DV)?;
        let mut collected = vec![0.0f32; t_len * VAL_HEADS * DV];
        for &(start, count) in chunks {
            let chunk = stream.clone_htod(&row[start * stride..(start + count) * stride])?;
            let hg = stream.clone_htod(&g[start * VAL_HEADS..(start + count) * VAL_HEADS])?;
            let hb = stream.clone_htod(&beta[start * VAL_HEADS..(start + count) * VAL_HEADS])?;
            let f = stream.clone_htod(&[0i32])?;
            let n = stream.clone_htod(&[count as i32])?;
            let mut out = stream.alloc_zeros::<f32>(count * VAL_HEADS * DV)?;
            let seqs = tuili_kernels::gdn::SeqLayout {
                first_token: &f.as_view(),
                n_tokens: &n.as_view(),
                n_seqs: 1,
                total_tokens: count,
            };
            k.gdn_delta_rule(
                &mut out.as_view_mut(),
                &mut state.as_view_mut(),
                &chunk.as_view(),
                &hg.as_view(),
                &hb.as_view(),
                &seqs,
                VAL_HEADS,
                KEY_HEADS,
                DK,
                DV,
                off,
            )?;
            k.device().synchronize()?;
            let piece = stream.clone_dtoh(&out)?;
            collected[start * VAL_HEADS * DV..(start + count) * VAL_HEADS * DV]
                .copy_from_slice(&piece);
        }
        Ok((collected, stream.clone_dtoh(&state)?))
    };

    let (whole, whole_state) = run(&[(0, t_len)])?;
    let peak = whole.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let speak = whole_state.iter().fold(0.0f32, |m, v| m.max(v.abs()));

    // Prefill four, then continue with the rest — and then one token at a time,
    // which is what decode does.
    for (label, chunks) in [
        ("a 4-token prefill then the rest", vec![(0usize, 4usize), (4, t_len - 4)]),
        ("one token at a time", (0..t_len).map(|t| (t, 1)).collect()),
    ] {
        let (out, state) = run(&chunks)?;
        let (worst, at) = max_abs_diff(&out, &whole);
        assert!(
            worst < 1e-4 * peak.max(1e-6),
            "{label} gave a different answer from one call, by {worst:.2e} at \
             {at}; the carried state is wrong, which makes decode a different \
             model from prefill"
        );
        let (sworst, _) = max_abs_diff(&state, &whole_state);
        assert!(
            sworst < 1e-4 * speak.max(1e-6),
            "{label} left a different state, off by {sworst:.2e}"
        );
    }
    Ok(())
}

#[test]
fn two_sequences_in_one_batch_keep_separate_state() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    // Sequence A gets 3 tokens, B gets 5, and slot 2 is idle — the idle slot is
    // the point: one launch covers the whole pool, and a sequence with no
    // tokens must neither run nor have its state touched.
    let lens = [3usize, 5, 0];
    let total: usize = lens.iter().sum();

    let q_small = pseudo_random(total * KEY_HEADS * DK, 0x3001);
    let k_small = pseudo_random(total * KEY_HEADS * DK, 0x3002);
    let v = pseudo_random(total * VAL_HEADS * DV, 0x3003);
    let g = decaying(total * VAL_HEADS, 0x3004, 0.6);
    let beta = betas(total * VAL_HEADS, 0x3005);
    let (row, off) = packed(&q_small, &k_small, &v, total);
    let stride = off.0;

    let d_row = stream.clone_htod(&row)?;
    let d_g = stream.clone_htod(&g)?;
    let d_beta = stream.clone_htod(&beta)?;
    let first = stream.clone_htod(&[0i32, lens[0] as i32, 0])?;
    let ntok = stream.clone_htod(&[lens[0] as i32, lens[1] as i32, 0])?;
    let mut out = stream.alloc_zeros::<f32>(total * VAL_HEADS * DV)?;
    let mut state = stream.alloc_zeros::<f32>(3 * VAL_HEADS * DK * DV)?;
    // A sentinel in the idle slot, to catch a launch that writes there anyway.
    let sentinel = vec![-12345.0f32; VAL_HEADS * DK * DV];
    stream.memcpy_htod(
        &sentinel,
        &mut state.slice_mut(2 * VAL_HEADS * DK * DV..3 * VAL_HEADS * DK * DV),
    )?;

    let seqs = tuili_kernels::gdn::SeqLayout {
        first_token: &first.as_view(),
        n_tokens: &ntok.as_view(),
        n_seqs: 3,
        total_tokens: total,
    };
    k.gdn_delta_rule(
        &mut out.as_view_mut(),
        &mut state.as_view_mut(),
        &d_row.as_view(),
        &d_g.as_view(),
        &d_beta.as_view(),
        &seqs,
        VAL_HEADS,
        KEY_HEADS,
        DK,
        DV,
        off,
    )?;
    k.device().synchronize()?;
    let batched = stream.clone_dtoh(&out)?;
    let states = stream.clone_dtoh(&state)?;

    assert!(
        states[2 * VAL_HEADS * DK * DV..].iter().all(|v| *v == -12345.0),
        "the idle sequence slot was written; a launch covering the whole pool \
         has to leave slots with no tokens alone"
    );

    // Each sequence alone. Only the second one would show state bleed, which is
    // why both are run rather than just checking the first.
    let mut offset = 0usize;
    for (s, &len) in lens.iter().enumerate() {
        if len == 0 {
            continue;
        }
        let chunk = stream.clone_htod(&row[offset * stride..(offset + len) * stride])?;
        let hg = stream.clone_htod(&g[offset * VAL_HEADS..(offset + len) * VAL_HEADS])?;
        let hb = stream.clone_htod(&beta[offset * VAL_HEADS..(offset + len) * VAL_HEADS])?;
        let f = stream.clone_htod(&[0i32])?;
        let n = stream.clone_htod(&[len as i32])?;
        let mut alone = stream.alloc_zeros::<f32>(len * VAL_HEADS * DV)?;
        let mut alone_state = stream.alloc_zeros::<f32>(VAL_HEADS * DK * DV)?;
        let one = tuili_kernels::gdn::SeqLayout {
            first_token: &f.as_view(),
            n_tokens: &n.as_view(),
            n_seqs: 1,
            total_tokens: len,
        };
        k.gdn_delta_rule(
            &mut alone.as_view_mut(),
            &mut alone_state.as_view_mut(),
            &chunk.as_view(),
            &hg.as_view(),
            &hb.as_view(),
            &one,
            VAL_HEADS,
            KEY_HEADS,
            DK,
            DV,
            off,
        )?;
        k.device().synchronize()?;
        let solo = stream.clone_dtoh(&alone)?;
        let slice = &batched[offset * VAL_HEADS * DV..(offset + len) * VAL_HEADS * DV];
        let (worst, at) = max_abs_diff(slice, &solo);
        let peak = solo.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        assert!(
            worst < 1e-5 * peak.max(1e-6),
            "sequence {s} in a batch of three differs from the same sequence \
             alone by {worst:.2e} at {at}; the two are sharing state"
        );
        offset += len;
    }
    Ok(())
}

#[test]
fn the_l2norm_and_scale_match_the_reference() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let t_len = 6;
    let key_dim = KEY_HEADS * DK;
    let val_dim = VAL_HEADS * DV;
    let stride = 2 * key_dim + val_dim;

    // A full packed row, so the test also establishes that v is left alone.
    let row = pseudo_random(t_len * stride, 0x1a1a);

    let mut want = row.clone();
    for t in 0..t_len {
        let base = t * stride;
        let mut q = want[base..base + key_dim].to_vec();
        let mut kk = want[base + key_dim..base + 2 * key_dim].to_vec();
        qwen35::l2norm_rows(&mut q, DK, 1e-6);
        qwen35::l2norm_rows(&mut kk, DK, 1e-6);
        for v in q.iter_mut() {
            *v *= (DK as f32).sqrt().recip();
        }
        want[base..base + key_dim].copy_from_slice(&q);
        want[base + key_dim..base + 2 * key_dim].copy_from_slice(&kk);
    }

    let mut d_row = stream.clone_htod(&row)?;
    k.gdn_qk_l2norm(
        &mut d_row.as_view_mut(),
        t_len,
        KEY_HEADS,
        DK,
        stride,
        0,
        key_dim,
        1e-6,
    )?;
    k.device().synchronize()?;
    let got = stream.clone_dtoh(&d_row)?;
    let (worst, at) = max_abs_diff(&got, &want);
    assert!(worst < 1e-6, "the packed row disagreed by {worst:.2e} at {at}");

    // v must be untouched, bit for bit: it shares the row and nothing should
    // have written into it.
    for t in 0..t_len {
        let vs = t * stride + 2 * key_dim;
        assert_eq!(
            &got[vs..vs + val_dim],
            &row[vs..vs + val_dim],
            "v was modified at token {t}"
        );
    }

    // The scale must be on q alone: k's rows must still have unit norm.
    for t in 0..t_len {
        for h in 0..KEY_HEADS {
            let base = t * stride + key_dim + h * DK;
            let n: f32 = got[base..base + DK].iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (n - 1.0).abs() < 1e-4,
                "k row (t{t}, h{h}) has norm {n}; the 1/sqrt(dk) scale leaked onto k"
            );
        }
    }
    Ok(())
}

#[test]
fn the_gated_rmsnorm_matches_the_reference_and_normalizes_first() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let rows = 8 * VAL_HEADS;
    // Magnitudes near the real checkpoint's, where the eps does not dominate
    // the denominator. At 1e-3 it would, and then both orders of normalize and
    // gate give the same answer and nothing here would be tested.
    let x: Vec<f32> = pseudo_random(rows * DV, 0x4d4d).iter().map(|v| v * 0.9).collect();
    let z = pseudo_random(rows * DV, 0x5e5e);
    let w: Vec<f32> = (0..DV).map(|i| 0.78 + (i % 7) as f32 * 0.02).collect();
    let eps = 1e-6f32;

    let normed = qwen35::rms_norm_rows(&x, &w, DV, eps);
    let want: Vec<f32> = normed
        .iter()
        .zip(&z)
        .map(|(n, zz)| n * qwen35::silu(*zz))
        .collect();

    let dx = stream.clone_htod(&x)?;
    let dz = stream.clone_htod(&z)?;
    let dw = stream.clone_htod(&w)?;
    let mut out = stream.alloc_zeros::<f32>(rows * DV)?;
    k.gdn_gated_rmsnorm(
        &mut out.as_view_mut(),
        &dx.as_view(),
        &dz.as_view(),
        &dw.as_view(),
        rows,
        DV,
        eps,
    )?;
    k.device().synchronize()?;
    let got = stream.clone_dtoh(&out)?;
    let rel = max_rel_diff(&got, &want);
    assert!(rel < 1e-5, "the gated norm diverged by {rel:.2e}");

    // Gate first, normalize after: must be a different answer here, or the
    // order is not pinned.
    let pre: Vec<f32> = x
        .iter()
        .zip(&z)
        .map(|(v, zz)| v * qwen35::silu(*zz))
        .collect();
    let other = qwen35::rms_norm_rows(&pre, &w, DV, eps);
    let spread = max_rel_diff(&other, &want);
    assert!(
        spread > 1e-2,
        "gating before normalizing changed the answer by only {spread:.2e}; at \
         these magnitudes the test cannot tell the two orders apart"
    );
    Ok(())
}

#[test]
fn the_output_gate_is_sigmoid_not_silu() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let n = 4096;
    let x = pseudo_random(n, 0x6f6f);
    let gate = pseudo_random(n, 0x7070);

    let want: Vec<f32> = x
        .iter()
        .zip(&gate)
        .map(|(v, g)| v * qwen35::sigmoid(*g))
        .collect();
    let mut dx = stream.clone_htod(&x)?;
    let dg = stream.clone_htod(&gate)?;
    k.sigmoid_gate(&mut dx.as_view_mut(), &dg.as_view(), n)?;
    k.device().synchronize()?;
    let got = stream.clone_dtoh(&dx)?;
    let (worst, at) = max_abs_diff(&got, &want);
    assert!(worst < 1e-6, "the gate disagreed by {worst:.2e} at {at}");

    // silu — what config's "swish" would suggest, and what the reference does
    // *not* do — must differ.
    let silu_version: Vec<f32> = x
        .iter()
        .zip(&gate)
        .map(|(v, g)| v * qwen35::silu(*g))
        .collect();
    let spread = max_rel_diff(&silu_version, &want);
    assert!(
        spread > 1e-2,
        "silu and sigmoid agreed to {spread:.2e} on this input, so this test \
         does not establish which the checkpoint wants"
    );
    Ok(())
}
