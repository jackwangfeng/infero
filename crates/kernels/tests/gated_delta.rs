//! The GatedDeltaNet kernels, against the host reference in
//! `infero_model::qwen35`.
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
use infero_model::qwen35;

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

    let seqs = infero_kernels::gdn::SeqLayout {
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
        let s = infero_kernels::gdn::SeqLayout {
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

/// `gdn_conv_prefill` (token dimension split across blocks) against
/// `gdn_conv` (the reference, one thread walks every token of a channel) --
/// same math, different grid, so this expects the tolerance a reassociated
/// float sum gets elsewhere in this file, not bit-for-bit. Shapes chosen to
/// land on both sides of a chunk boundary (`gdn_conv_prefill`'s own internal
/// chunking, not the model's `batch_tokens`): smaller than one chunk, exactly
/// the minimum chunk width, several chunks, and a ragged tail past a whole
/// number of chunks.
#[test]
fn the_chunked_conv_matches_the_unchunked_one() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let channels = 2 * KEY_HEADS * DK + VAL_HEADS * DV;

    for t_len in [1usize, 9, 32, 33, 63, 200, 257, 1024, 1025] {
        let x = pseudo_random(t_len * channels, 0x9a51 + t_len as u64);
        let w = pseudo_random(channels * CONV_K, 0x9a52);

        let d_x = stream.clone_htod(&x)?;
        let d_w = stream.clone_htod(&w)?;
        let d_first = stream.clone_htod(&[0i32])?;
        let d_ntok = stream.clone_htod(&[t_len as i32])?;
        let seqs = infero_kernels::gdn::SeqLayout {
            first_token: &d_first.as_view(),
            n_tokens: &d_ntok.as_view(),
            n_seqs: 1,
            total_tokens: t_len,
        };

        let mut d_want = stream.alloc_zeros::<f32>(t_len * channels)?;
        let mut d_state_want = stream.alloc_zeros::<f32>(channels * (CONV_K - 1))?;
        k.gdn_conv(
            &mut d_want.as_view_mut(),
            &d_x.as_view(),
            &mut d_state_want.as_view_mut(),
            &d_w.as_view(),
            &seqs,
            channels,
            CONV_K,
        )?;

        let mut d_got = stream.alloc_zeros::<f32>(t_len * channels)?;
        let mut d_state_got = stream.alloc_zeros::<f32>(channels * (CONV_K - 1))?;
        k.gdn_conv_prefill(
            &mut d_got.as_view_mut(),
            &d_x.as_view(),
            &mut d_state_got.as_view_mut(),
            &d_w.as_view(),
            &seqs,
            channels,
            CONV_K,
        )?;
        k.device().synchronize()?;

        let want = stream.clone_dtoh(&d_want)?;
        let got = stream.clone_dtoh(&d_got)?;
        let (worst, at) = max_abs_diff(&got, &want);
        assert!(worst < 2e-5, "{t_len} tokens: chunked conv disagreed by {worst:.2e} at {at}");

        let state_want = stream.clone_dtoh(&d_state_want)?;
        let state_got = stream.clone_dtoh(&d_state_got)?;
        let (sworst, sat) = max_abs_diff(&state_got, &state_want);
        assert!(sworst < 2e-5, "{t_len} tokens: chunked conv's carried window disagreed by {sworst:.2e} at {sat}");
    }
    Ok(())
}

/// `Model::linear_attention` picks `gdn_conv_prefill` for a single-sequence
/// call by slicing that one sequence's own row out of the pool-wide
/// `first`/`n_tokens` arrays -- `SeqLayout { n_seqs: 1, .. }` pointed at
/// `first[slot..slot+1]`, not the front of the array. This is the case that
/// slicing gets wrong if it slices from the front instead: the active
/// sequence sits in a slot that is not 0 (the common case once a pool's
/// slots have cycled), with a stale, nonzero `first` value left behind in
/// slot 0 by whichever sequence used to be there. If the fast path ever goes
/// back to reading slot 0 outright, this catches it: the two kernels would
/// read different tokens entirely, not just compute them differently, so a
/// disagreement here is not a rounding question.
#[test]
fn a_sliced_slot_feeds_gdn_conv_prefill_the_right_row_not_slot_zero() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let channels = 2 * KEY_HEADS * DK + VAL_HEADS * DV;
    const MAX_SEQS: usize = 4;
    const ACTIVE_SLOT: usize = 2;
    let t_len = 200usize;

    // Slot 0 looks like a stale previous occupant: a nonzero `first` with
    // `n_tok = 0` -- exactly what a freed-then-reused pool leaves behind,
    // since only `n_tok` is what marks a slot idle.
    let x = pseudo_random(t_len * channels, 0x5c01);
    let w = pseudo_random(channels * CONV_K, 0x5c02);
    let d_x = stream.clone_htod(&x)?;
    let d_w = stream.clone_htod(&w)?;

    let mut first = vec![0i32; MAX_SEQS];
    let mut ntok = vec![0i32; MAX_SEQS];
    first[0] = 777; // stale, must never be read
    first[ACTIVE_SLOT] = 0;
    ntok[ACTIVE_SLOT] = t_len as i32;
    let d_first = stream.clone_htod(&first)?;
    let d_ntok = stream.clone_htod(&ntok)?;

    // Reference: the unchunked kernel, given the whole pool-wide layout --
    // already trusted to route to the right slot, since it walks every slot
    // and skips the idle ones by `n_tok`, not by position.
    let full = infero_kernels::gdn::SeqLayout {
        first_token: &d_first.as_view(),
        n_tokens: &d_ntok.as_view(),
        n_seqs: MAX_SEQS,
        total_tokens: t_len, // only the active slot's tokens are real
    };
    // `gdn_conv`'s state, unlike `gdn_conv_prefill`'s, is pool-wide -- one
    // window a slot (`debug_assert!(state.len() >= seqs.n_seqs * channels *
    // (k - 1))` in `Kernels::gdn_conv`) -- so the reference call needs room
    // for all `MAX_SEQS` windows even though only one slot is live.
    let conv_n = channels * (CONV_K - 1);
    let mut d_want = stream.alloc_zeros::<f32>(t_len * channels)?;
    let mut d_state_want = stream.alloc_zeros::<f32>(MAX_SEQS * conv_n)?;
    k.gdn_conv(
        &mut d_want.as_view_mut(),
        &d_x.as_view(),
        &mut d_state_want.as_view_mut(),
        &d_w.as_view(),
        &full,
        channels,
        CONV_K,
    )?;

    // Subject: exactly what `Model::linear_attention` now does -- slice the
    // active slot's own one-element row out of the same device arrays and
    // hand that to `gdn_conv_prefill`.
    let active_first = d_first.as_view().slice(ACTIVE_SLOT..ACTIVE_SLOT + 1);
    let active_ntok = d_ntok.as_view().slice(ACTIVE_SLOT..ACTIVE_SLOT + 1);
    let one = infero_kernels::gdn::SeqLayout {
        first_token: &active_first,
        n_tokens: &active_ntok,
        n_seqs: 1,
        total_tokens: t_len,
    };
    let mut d_got = stream.alloc_zeros::<f32>(t_len * channels)?;
    let mut d_state_got = stream.alloc_zeros::<f32>(channels * (CONV_K - 1))?;
    k.gdn_conv_prefill(
        &mut d_got.as_view_mut(),
        &d_x.as_view(),
        &mut d_state_got.as_view_mut(),
        &d_w.as_view(),
        &one,
        channels,
        CONV_K,
    )?;
    k.device().synchronize()?;

    let want = stream.clone_dtoh(&d_want)?;
    let got = stream.clone_dtoh(&d_got)?;
    let (worst, at) = max_abs_diff(&got, &want);
    assert!(worst < 2e-5, "sliced active slot {ACTIVE_SLOT} disagreed with the full layout by {worst:.2e} at {at}");

    let state_want = stream.clone_dtoh(&d_state_want)?;
    let state_got = stream.clone_dtoh(&d_state_got)?;
    let active_want = &state_want[ACTIVE_SLOT * conv_n..(ACTIVE_SLOT + 1) * conv_n];
    let (sworst, sat) = max_abs_diff(&state_got, active_want);
    assert!(sworst < 2e-5, "sliced active slot {ACTIVE_SLOT}'s carried window disagreed by {sworst:.2e} at {sat}");
    Ok(())
}

/// `Model::linear_attention` picks `gdn_delta_rule`'s column-split fast path
/// (`gdn_delta_rule_reg128_split4_f32`) the same way it picks
/// `gdn_conv_prefill`'s -- by whether this call has exactly one active
/// sequence, not by `pool.max_seqs()` (a bug caught, and fixed, in the conv
/// path first; reproduced here and fixed the same way before this one ever
/// shipped) -- and slices `state` at that sequence's own slot, not the
/// front of the pool-wide buffer. Same shape of test as
/// `a_sliced_slot_feeds_gdn_conv_prefill_the_right_row_not_slot_zero`, this
/// time against the host reference rather than a second GPU kernel, since
/// there is no separate full-layout path here to also have to trust.
#[test]
fn a_sliced_slot_feeds_the_split_delta_rule_the_right_row_not_slot_zero() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let t_len = 11;
    const MAX_SEQS: usize = 4;
    const ACTIVE_SLOT: usize = 2;

    let q_small = pseudo_random(t_len * KEY_HEADS * DK, 0x7a01);
    let k_small = pseudo_random(t_len * KEY_HEADS * DK, 0x7a02);
    let v = pseudo_random(t_len * VAL_HEADS * DV, 0x7a03);
    let g = decaying(t_len * VAL_HEADS, 0x7a04, 0.7);
    let beta = betas(t_len * VAL_HEADS, 0x7a05);

    let (row, off) = packed(&q_small, &k_small, &v, t_len);
    let (want, want_state) = reference(&q_small, &k_small, &v, &g, &beta, t_len);

    let d_row = stream.clone_htod(&row)?;
    let d_g = stream.clone_htod(&g)?;
    let d_beta = stream.clone_htod(&beta)?;

    // Slot 0 looks like a stale previous occupant, same poisoning as the
    // conv test above.
    let mut first = vec![0i32; MAX_SEQS];
    let mut ntok = vec![0i32; MAX_SEQS];
    first[0] = 777;
    first[ACTIVE_SLOT] = 0;
    ntok[ACTIVE_SLOT] = t_len as i32;
    let d_first = stream.clone_htod(&first)?;
    let d_ntok = stream.clone_htod(&ntok)?;
    let active_first = d_first.as_view().slice(ACTIVE_SLOT..ACTIVE_SLOT + 1);
    let active_ntok = d_ntok.as_view().slice(ACTIVE_SLOT..ACTIVE_SLOT + 1);
    let seqs = infero_kernels::gdn::SeqLayout {
        first_token: &active_first,
        n_tokens: &active_ntok,
        n_seqs: 1, // triggers the split4 fast path, exactly as production now does
        total_tokens: t_len,
    };

    let mut d_out = stream.alloc_zeros::<f32>(t_len * VAL_HEADS * DV)?;
    // Pool-wide state, one block a slot -- sliced the same way
    // `Model::linear_attention` slices `recurrent`/the rollback scratch.
    let mut d_state = stream.alloc_zeros::<f32>(MAX_SEQS * VAL_HEADS * DK * DV)?;
    let state_n = VAL_HEADS * DK * DV;
    let mut d_state_view = d_state.as_view_mut();
    let mut active_state = d_state_view.slice_mut(ACTIVE_SLOT * state_n..(ACTIVE_SLOT + 1) * state_n);

    k.gdn_delta_rule(
        &mut d_out.as_view_mut(),
        &mut active_state,
        &d_row.as_view(),
        &d_g.as_view(),
        &d_beta.as_view(),
        &seqs,
        VAL_HEADS,
        KEY_HEADS,
        DK,
        DV,
        off,
        false,
    )?;
    k.device().synchronize()?;

    let got = stream.clone_dtoh(&d_out)?;
    let got_state = stream.clone_dtoh(&d_state)?;
    let active_state_got = &got_state[ACTIVE_SLOT * state_n..(ACTIVE_SLOT + 1) * state_n];

    let rel = max_rel_diff(&got, &want);
    assert!(rel < 2e-3, "sliced active slot {ACTIVE_SLOT}'s output diverged by {rel:.2e}");
    let peak = want_state.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let (worst, at) = max_abs_diff(active_state_got, &want_state);
    assert!(
        worst < 1e-4 * peak,
        "sliced active slot {ACTIVE_SLOT}'s final state diverged by {worst:.2e} at {at} \
         against a peak of {peak:.3e}"
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
    
    heads,)?;
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

    let seqs = infero_kernels::gdn::SeqLayout {
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
        // V heads grouped by key head, as a Hugging Face checkpoint
        // stores them.
        false,
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
            let seqs = infero_kernels::gdn::SeqLayout {
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
                // V heads grouped by key head, as a Hugging Face checkpoint
                // stores them.
                false,
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

    let seqs = infero_kernels::gdn::SeqLayout {
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
        // V heads grouped by key head, as a Hugging Face checkpoint
        // stores them.
        false,
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
        let one = infero_kernels::gdn::SeqLayout {
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
            // V heads grouped by key head, as a Hugging Face checkpoint
            // stores them.
            false,
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

    let normed = qwen35::rms_norm_rows(&x, &w, DV, eps, 0.0);
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
    let other = qwen35::rms_norm_rows(&pre, &w, DV, eps, 0.0);
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

/// The query and its gate interleave per head, and the split down the middle —
/// the other plausible reading of the same buffer — must give a different
/// answer.
///
/// Head 0 cannot distinguish the two, which is why this checks a head past the
/// first. A test that only looked at head 0 would bless either layout.
#[test]
fn the_split_is_per_head_not_per_half() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let (t_len, heads, hd) = (5usize, 6usize, 16usize);
    let n = t_len * heads * hd;

    let src = pseudo_random(2 * n, 0x8f8f);
    let mut want_q = vec![0.0f32; n];
    let mut want_gate = vec![0.0f32; n];
    for t in 0..t_len {
        for h in 0..heads {
            let row = (t * heads + h) * 2 * hd;
            let dst = (t * heads + h) * hd;
            want_q[dst..dst + hd].copy_from_slice(&src[row..row + hd]);
            want_gate[dst..dst + hd].copy_from_slice(&src[row + hd..row + 2 * hd]);
        }
    }

    let d_src = stream.clone_htod(&src)?;
    let mut d_q = stream.alloc_zeros::<f32>(n)?;
    let mut d_gate = stream.alloc_zeros::<f32>(n)?;
    k.split_interleaved(
        &mut d_q.as_view_mut(),
        &mut d_gate.as_view_mut(),
        &d_src.as_view(),
        t_len,
        heads,
        hd,
    )?;
    k.device().synchronize()?;
    let (worst, at) = max_abs_diff(&stream.clone_dtoh(&d_q)?, &want_q);
    assert_eq!(worst, 0.0, "q disagreed at {at}; this is a copy, not arithmetic");
    let (worst, at) = max_abs_diff(&stream.clone_dtoh(&d_gate)?, &want_gate);
    assert_eq!(worst, 0.0, "gate disagreed at {at}");

    // `[all queries | all gates]`: for head 0 it agrees, past that it does not.
    let half_split_q = &src[..n];
    let mut differing_heads = 0;
    for h in 0..heads {
        let dst = h * hd; // token 0
        if half_split_q[dst..dst + hd] != want_q[dst..dst + hd] {
            differing_heads += 1;
        }
    }
    assert!(
        differing_heads >= heads - 1,
        "only {differing_heads} of {heads} heads distinguish the interleaved \
         layout from the halved one; this test would not catch the mistake"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// The register-blocked delta rule, and what it must not have cost.
//
// `DK` and `DV` above are the checkpoint's 128, so every test up to here now
// runs `DeltaVariant::Reg` — that is the point, the fast path is the tested
// path. What follows covers the two things that change hands as a result: the
// global-memory version those tests used to exercise, which is now only
// reachable by name, and the register residency the fast path depends on and
// which no output check can see.
// ---------------------------------------------------------------------------

use infero_kernels::gdn::DeltaVariant;

/// Run the delta rule over `chunks` of one sequence with a named variant,
/// returning the concatenated output and the final state.
fn run_variant(
    k: &infero_kernels::Kernels,
    row: &[f32],
    g: &[f32],
    beta: &[f32],
    off: (usize, usize, usize, usize),
    t_len: usize,
    chunks: &[(usize, usize)],
    variant: DeltaVariant,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let stream = k.device().stream().clone();
    let stride = off.0;
    let mut state = stream.alloc_zeros::<f32>(VAL_HEADS * DK * DV)?;
    let mut collected = vec![0.0f32; t_len * VAL_HEADS * DV];
    for &(start, count) in chunks {
        let chunk = stream.clone_htod(&row[start * stride..(start + count) * stride])?;
        let hg = stream.clone_htod(&g[start * VAL_HEADS..(start + count) * VAL_HEADS])?;
        let hb = stream.clone_htod(&beta[start * VAL_HEADS..(start + count) * VAL_HEADS])?;
        let f = stream.clone_htod(&[0i32])?;
        let n = stream.clone_htod(&[count as i32])?;
        let mut out = stream.alloc_zeros::<f32>(count * VAL_HEADS * DV)?;
        let seqs = infero_kernels::gdn::SeqLayout {
            first_token: &f.as_view(),
            n_tokens: &n.as_view(),
            n_seqs: 1,
            total_tokens: count,
        };
        k.gdn_delta_rule_variant(
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
            // V heads grouped by key head, as a Hugging Face checkpoint
            // stores them.
            false,
            variant,
        )?;
        k.device().synchronize()?;
        let piece = stream.clone_dtoh(&out)?;
        collected[start * VAL_HEADS * DV..(start + count) * VAL_HEADS * DV]
            .copy_from_slice(&piece);
    }
    Ok((collected, stream.clone_dtoh(&state)?))
}

/// All three kernels compute the same recurrence, whole-sequence and split.
///
/// The register version reassociates both reductions — partial sums a thread
/// and then a partner lane — so this is not a bit-for-bit comparison, but the
/// tolerance is the same 1e-4 of peak the reference comparison uses. A
/// register-blocking mistake does not land at 1e-5: getting the row range, the
/// partner reduction or the barrier wrong moves the answer by whole percent.
#[test]
fn the_three_delta_rule_kernels_agree_with_each_other_and_the_reference() -> Result<()> {
    let k = kernels()?;
    let t_len = 13;

    let q_small = pseudo_random(t_len * KEY_HEADS * DK, 0xb001);
    let k_small = pseudo_random(t_len * KEY_HEADS * DK, 0xb002);
    let v = pseudo_random(t_len * VAL_HEADS * DV, 0xb003);
    let g = decaying(t_len * VAL_HEADS, 0xb004, 0.7);
    let beta = betas(t_len * VAL_HEADS, 0xb005);
    let (row, off) = packed(&q_small, &k_small, &v, t_len);
    let (want, want_state) = reference(&q_small, &k_small, &v, &g, &beta, t_len);

    let peak = want.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let speak = want_state.iter().fold(0.0f32, |m, x| m.max(x.abs()));

    // Whole sequence in one call, and then one token at a time, which is the
    // case where the register version has to have written its state out and
    // read it back correctly rather than merely kept it.
    for (label, chunks) in [
        ("one call", vec![(0usize, t_len)]),
        ("one token at a time", (0..t_len).map(|t| (t, 1)).collect()),
        ("5 then 8", vec![(0usize, 5usize), (5, 8)]),
    ] {
        for variant in [DeltaVariant::Global, DeltaVariant::Reg, DeltaVariant::Shared, DeltaVariant::Chunk] {
            let (out, state) =
                run_variant(&k, &row, &g, &beta, off, t_len, &chunks, variant)?;
            let (worst, at) = max_abs_diff(&out, &want);
            assert!(
                worst < 1e-4 * peak.max(1e-6),
                "{variant:?} on {label} diverged from the reference by {worst:.2e} \
                 at {at}: got {}, reference {}",
                out[at],
                want[at]
            );
            let (sworst, sat) = max_abs_diff(&state, &want_state);
            assert!(
                sworst < 1e-4 * speak.max(1e-6),
                "{variant:?} on {label} left a state {sworst:.2e} from the \
                 reference's at {sat}"
            );
        }
    }
    Ok(())
}

/// The three-kernel split (`gdn_chunk_uw_f32` / `gdn_chunk_state_f32` /
/// `gdn_chunk_output_f32`) against the same host reference every other delta-
/// rule kernel is checked against -- single sequence, single call, the only
/// case this first pass claims to handle (see `gdn_chunk_split3_delta_rule`'s
/// own doc comment in `gdn.rs`). A longer run than the other kernels' tests
/// (`t_len` past one `GDN_CHUNK` of 32) specifically to exercise more than
/// one chunk and the sequential kernel-2 loop across them, not just a single
/// chunk's own parallel kernel-1/kernel-3 math.
#[test]
fn the_three_kernel_split_matches_the_reference() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let t_len = 70;

    let q_small = pseudo_random(t_len * KEY_HEADS * DK, 0xc001);
    let k_small = pseudo_random(t_len * KEY_HEADS * DK, 0xc002);
    let v = pseudo_random(t_len * VAL_HEADS * DV, 0xc003);
    let g = decaying(t_len * VAL_HEADS, 0xc004, 0.7);
    let beta = betas(t_len * VAL_HEADS, 0xc005);
    let (row, off) = packed(&q_small, &k_small, &v, t_len);
    let (want, want_state) = reference(&q_small, &k_small, &v, &g, &beta, t_len);

    let peak = want.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let speak = want_state.iter().fold(0.0f32, |m, x| m.max(x.abs()));

    let chunk = stream.clone_htod(&row)?;
    let hg = stream.clone_htod(&g)?;
    let hb = stream.clone_htod(&beta)?;
    let f = stream.clone_htod(&[0i32])?;
    let n = stream.clone_htod(&[t_len as i32])?;
    let seqs = infero_kernels::gdn::SeqLayout {
        first_token: &f.as_view(),
        n_tokens: &n.as_view(),
        n_seqs: 1,
        total_tokens: t_len,
    };

    use infero_kernels::gdn::GdnChunkStateVariant;
    for (label, k2) in [
        ("plain", GdnChunkStateVariant::Plain),
        ("pipelined", GdnChunkStateVariant::Pipelined),
        ("pipelined_split4", GdnChunkStateVariant::PipelinedSplit4),
    ] {
        let mut out = stream.alloc_zeros::<f32>(t_len * VAL_HEADS * DV)?;
        let mut state = stream.alloc_zeros::<f32>(VAL_HEADS * DK * DV)?;
        k.gdn_chunk_split3_delta_rule(
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
            false,
            k2,
        )?;
        k.device().synchronize()?;
        let got = stream.clone_dtoh(&out)?;
        let got_state = stream.clone_dtoh(&state)?;

        let (worst, at) = max_abs_diff(&got, &want);
        assert!(
            worst < 1e-4 * peak.max(1e-6),
            "three-kernel split ({label}) diverged from the reference by {worst:.2e} at {at}: \
             got {}, reference {}",
            got[at],
            want[at]
        );
        let (sworst, sat) = max_abs_diff(&got_state, &want_state);
        assert!(
            sworst < 1e-4 * speak.max(1e-6),
            "three-kernel split ({label}) left a state {sworst:.2e} from the reference's at {sat}"
        );
    }
    Ok(())
}

/// `gdn_chunk_ab_f32` -- the `(A, B)` affine-recurrence pair,
/// `S(c+1) = A(c)@S(c) + B(c)`, computed from an already-run
/// `gdn_chunk_uw_only` call's own `W`/`U` output. Verified against a
/// HOST-computed `A`/`B` built from that SAME downloaded `W`/`U` plus `K`/`g`
/// -- this isolates exactly the one thing that's actually new here (the
/// `A = decay*I - Kd@W`, `B = Kd@U` formula itself), consistent with
/// `verify_linear_recurrence.py`'s own derivation (see
/// project_infero_perf_gap.md memory for that numerical check). Also checks
/// `gdn_chunk_uw_only`'s own `W`/`U` output directly against a from-scratch
/// host recomputation (forward substitution + WY), since no other test in
/// this file checks `W`/`U` directly rather than through the full pipeline's
/// final output.
///
/// `w`/`u`/`a`/`b` are all sized for the FULL `VAL_HEADS`, not just head 0,
/// even though only head 0 is checked -- both kernels grid over `heads`
/// blocks (`(chunk*heads+head)*...`), so a buffer sized for one head lets
/// every OTHER head's block write out of bounds. Undersizing these exact
/// buffers produced a real, long, confusing debugging session while writing
/// this test: heads 1..11's out-of-bounds writes corrupted whatever GPU
/// memory the allocator placed next (in practice, the LATER-allocated `a`/
/// `b` buffers), which looked exactly like `gdn_chunk_uw_f32`'s own,
/// already-shipped `U` output being wrong -- an early, narrower debug probe
/// (checking `u` right after computing it, before `a`/`b` existed to be
/// corrupted) showed it as correct, which was the real tell once connected
/// to the later failures. It never was wrong; only this test's own buffer
/// sizing was. Single chunk (`t_len < GDN_CHUNK`) -- this test is about the
/// formula, not multi-chunk sequencing, which the 3-kernel-split tests
/// already cover for `W`/`U`.
#[test]
fn the_chunk_ab_affine_recurrence_matches_a_host_reference() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let t_len = 13usize;
    const GDN_CHUNK: usize = 32;

    let q_small = pseudo_random(t_len * KEY_HEADS * DK, 0xd001);
    let k_small = pseudo_random(t_len * KEY_HEADS * DK, 0xd002);
    let v = pseudo_random(t_len * VAL_HEADS * DV, 0xd003);
    let g = decaying(t_len * VAL_HEADS, 0xd004, 0.7);
    let beta = betas(t_len * VAL_HEADS, 0xd005);
    let (row, off) = packed(&q_small, &k_small, &v, t_len);

    let chunk = stream.clone_htod(&row)?;
    let hg = stream.clone_htod(&g)?;
    let hb = stream.clone_htod(&beta)?;
    let f = stream.clone_htod(&[0i32])?;
    let n = stream.clone_htod(&[t_len as i32])?;
    let seqs = infero_kernels::gdn::SeqLayout {
        first_token: &f.as_view(),
        n_tokens: &n.as_view(),
        n_seqs: 1,
        total_tokens: t_len,
    };

    // Sized for ALL `VAL_HEADS` heads, not just head 0 -- the kernel grids
    // over `heads` blocks (`(chunk*heads+head)*GDN_CHUNK*GDN_DV`), so a
    // buffer sized for one head only lets every OTHER head's block write
    // out of bounds. This exact bug produced a real, deeply confusing
    // false alarm while building this test: heads 1..11's out-of-bounds
    // writes corrupted whatever GPU memory the allocator placed next,
    // which looked like `gdn_chunk_uw_f32`'s own `U` output being wrong --
    // it wasn't; head 0's own write was always correct, only this buffer's
    // undersizing was not.
    let mut w = stream.alloc_zeros::<f32>(VAL_HEADS * GDN_CHUNK * DK)?;
    let mut u = stream.alloc_zeros::<f32>(VAL_HEADS * GDN_CHUNK * DV)?;
    k.gdn_chunk_uw_only(
        &mut w.as_view_mut(),
        &mut u.as_view_mut(),
        &chunk.as_view(),
        &hg.as_view(),
        &hb.as_view(),
        &seqs,
        VAL_HEADS,
        KEY_HEADS,
        DK,
        DV,
        off,
        false,
    )?;
    k.device().synchronize()?;
    let got_w = stream.clone_dtoh(&w)?;
    let got_u = stream.clone_dtoh(&u)?;

    let mut a = stream.alloc_zeros::<f32>(VAL_HEADS * DK * DK)?;
    let mut b = stream.alloc_zeros::<f32>(VAL_HEADS * DK * DV)?;
    k.gdn_chunk_ab(
        &mut a.as_view_mut(),
        &mut b.as_view_mut(),
        &w.as_view(),
        &u.as_view(),
        &chunk.as_view(),
        &hg.as_view(),
        &seqs,
        VAL_HEADS,
        KEY_HEADS,
        DK,
        DV,
        off,
        false,
    )?;
    k.device().synchronize()?;
    let got_a = stream.clone_dtoh(&a)?;
    let got_b = stream.clone_dtoh(&b)?;

    // Host reference: only head 0's own K and cumulative g (single-head
    // slice of the packed row/g buffers this kernel itself would read).
    let (stride, _q_off, k_off, _v_off) = off;
    let head = 0usize;
    let khead = head / (VAL_HEADS / KEY_HEADS);
    let mut kmat = vec![0.0f32; t_len * DK];
    for t in 0..t_len {
        for d in 0..DK {
            kmat[t * DK + d] = row[t * stride + k_off + khead * DK + d];
        }
    }
    let mut gc = vec![0.0f32; t_len];
    let mut acc = 0.0f32;
    for t in 0..t_len {
        acc += g[t * VAL_HEADS + head];
        gc[t] = acc;
    }
    let decay_whole = gc[t_len - 1].exp();
    let df: Vec<f32> = gc.iter().map(|&x| (gc[t_len - 1] - x).exp()).collect();

    // Independently recompute W/U via the full forward-substitution
    // algorithm (system matrix -> (I+A)^-1 -> WY) to check `got_w`/`got_u`
    // directly, not just indirectly through A/B.
    let hbeta: Vec<f32> = (0..t_len).map(|t| beta[t * VAL_HEADS + head]).collect();
    let mut amat = vec![0.0f32; t_len * t_len];
    for i in 0..t_len {
        for kk in 0..i {
            let mut dot = 0.0f32;
            for d in 0..DK {
                dot += kmat[i * DK + d] * kmat[kk * DK + d];
            }
            amat[i * t_len + kk] = hbeta[i] * (gc[i] - gc[kk]).exp() * dot;
        }
    }
    let mut ainv = vec![0.0f32; t_len * t_len];
    for i in 0..t_len {
        for kk in 0..=i {
            let mut acc2 = if i == kk { 1.0f32 } else { 0.0f32 };
            for m in kk..i {
                acc2 -= amat[i * t_len + m] * ainv[m * t_len + kk];
            }
            ainv[i * t_len + kk] = acc2;
        }
    }
    let sbg: Vec<f32> = (0..t_len).map(|t| hbeta[t] * gc[t].exp()).collect();
    let vdim = VAL_HEADS * DV;
    let mut want_w = vec![0.0f32; t_len * DK];
    let mut want_u = vec![0.0f32; t_len * DV];
    for i in 0..t_len {
        for d in 0..DK {
            let mut wacc = 0.0f32;
            for kk in 0..=i {
                wacc += ainv[i * t_len + kk] * sbg[kk] * kmat[kk * DK + d];
            }
            want_w[i * DK + d] = wacc;
        }
        for d in 0..DV {
            let mut uacc = 0.0f32;
            for kk in 0..=i {
                uacc += ainv[i * t_len + kk] * hbeta[kk] * v[kk * vdim + head * DV + d];
            }
            want_u[i * DV + d] = uacc;
        }
    }
    let (wworst, wat) = max_abs_diff(&got_w[..t_len * DK], &want_w);
    let wpeak = want_w.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(
        wworst < 1e-3 * wpeak.max(1e-6),
        "gdn_chunk_uw_only's W diverged from the host reference by {wworst:.2e} at {wat}:          got {}, want {}",
        got_w[wat],
        want_w[wat]
    );
    let (uworst, uat) = max_abs_diff(&got_u[..t_len * DV], &want_u);
    let upeak = want_u.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(
        uworst < 1e-3 * upeak.max(1e-6),
        "gdn_chunk_uw_only's U diverged from the host reference by {uworst:.2e} at {uat}: \
         got {}, want {}",
        got_u[uat],
        want_u[uat]
    );

    // Now check A/B, built from `got_w`/`got_u` (already verified above) --
    // isolates the new formula itself from W/U's own correctness.
    let mut want_a = vec![0.0f32; DK * DK];
    let mut want_b = vec![0.0f32; DK * DV];
    for d1 in 0..DK {
        for d2 in 0..DK {
            let mut acc_a = 0.0f32;
            let mut acc_b = 0.0f32;
            for t in 0..t_len {
                let kd = kmat[t * DK + d1] * df[t];
                acc_a += kd * got_w[t * DK + d2];
                acc_b += kd * got_u[t * DV + d2];
            }
            want_a[d1 * DK + d2] = if d1 == d2 { decay_whole } else { 0.0 } - acc_a;
            want_b[d1 * DV + d2] = acc_b;
        }
    }

    // Head 0's own slice only -- `got_a`/`got_b` (like `got_w`/`got_u`) are
    // sized for all `VAL_HEADS` heads, since the kernel grids over all of
    // them; head 0's data sits at offset 0 regardless.
    let (aworst, aat) = max_abs_diff(&got_a[..DK * DK], &want_a);
    let apeak = want_a.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(
        aworst < 1e-3 * apeak.max(1e-6),
        "A diverged from the host reference by {aworst:.2e} at {aat}: got {}, want {}",
        got_a[aat],
        want_a[aat]
    );
    let (bworst, bat) = max_abs_diff(&got_b[..DK * DV], &want_b);
    let bpeak = want_b.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(
        bworst < 1e-3 * bpeak.max(1e-6),
        "B diverged from the host reference by {bworst:.2e} at {bat}: got {}, want {}",
        got_b[bat],
        want_b[bat]
    );
    Ok(())
}

/// `gdn_ab_combine_f32` -- the row-streaming `[128,128]@[128,128]` GEMM the
/// group-scan needs to combine two affine-recurrence pairs, checked against
/// a direct host matmul on plain random matrices (no GDN-specific setup
/// needed -- this is pure matrix algebra once `A`/`B` themselves exist).
/// `A2@A1`/`A2@B1+B2`, not `A1@A2` -- order matters (matrix multiplication
/// does not commute), matching `(a2,b2)` being applied AFTER `(a1,b1)`.
#[test]
fn the_ab_combine_matches_a_host_matmul() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    const DK: usize = 128;
    const DV: usize = 128;

    let a1 = pseudo_random(DK * DK, 0xe001);
    let b1 = pseudo_random(DK * DV, 0xe002);
    let a2 = pseudo_random(DK * DK, 0xe003);
    let b2 = pseudo_random(DK * DV, 0xe004);

    let da1 = stream.clone_htod(&a1)?;
    let db1 = stream.clone_htod(&b1)?;
    let da2 = stream.clone_htod(&a2)?;
    let db2 = stream.clone_htod(&b2)?;
    let mut a_out = stream.alloc_zeros::<f32>(DK * DK)?;
    let mut b_out = stream.alloc_zeros::<f32>(DK * DV)?;
    k.gdn_ab_combine(
        &mut a_out.as_view_mut(),
        &mut b_out.as_view_mut(),
        &da1.as_view(),
        &db1.as_view(),
        &da2.as_view(),
        &db2.as_view(),
    )?;
    k.device().synchronize()?;
    let got_a = stream.clone_dtoh(&a_out)?;
    let got_b = stream.clone_dtoh(&b_out)?;

    // Host reference: A2 @ A1, A2 @ B1 + B2.
    let mut want_a = vec![0.0f32; DK * DK];
    let mut want_b = vec![0.0f32; DK * DV];
    for d1 in 0..DK {
        for d2 in 0..DK {
            let mut acc = 0.0f32;
            for e in 0..DK {
                acc += a2[d1 * DK + e] * a1[e * DK + d2];
            }
            want_a[d1 * DK + d2] = acc;
        }
        for d2 in 0..DV {
            let mut acc = 0.0f32;
            for e in 0..DK {
                acc += a2[d1 * DK + e] * b1[e * DV + d2];
            }
            want_b[d1 * DV + d2] = acc + b2[d1 * DV + d2];
        }
    }

    let (aworst, aat) = max_abs_diff(&got_a, &want_a);
    let apeak = want_a.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(
        aworst < 1e-3 * apeak.max(1e-6),
        "combine's A diverged from the host matmul by {aworst:.2e} at {aat}: got {}, want {}",
        got_a[aat],
        want_a[aat]
    );
    let (bworst, bat) = max_abs_diff(&got_b, &want_b);
    let bpeak = want_b.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(
        bworst < 1e-3 * bpeak.max(1e-6),
        "combine's B diverged from the host matmul by {bworst:.2e} at {bat}: got {}, want {}",
        got_b[bat],
        want_b[bat]
    );
    Ok(())
}

/// `gdn_group_scan_f32` -- one group (`group_size` large enough to cover
/// every chunk in one block) walking `N` chunks of plain random `(A,B)`
/// pairs sequentially, checked against a host-computed sequential scan
/// using the exact same combine rule `the_ab_combine_matches_a_host_matmul`
/// already verified. Checks BOTH outputs: `prefix[c]` (the running
/// transform BEFORE chunk `c`, needed by the not-yet-built correction pass)
/// and the final group total (needed by the not-yet-built cross-group
/// combine). Pure random `(A,B)` inputs, not real GDN data -- this is a
/// test of the scan's own loop/indexing correctness, not of GDN's math
/// (already covered by `the_chunk_ab_affine_recurrence_matches_a_host_reference`).
#[test]
fn the_group_scan_matches_a_host_sequential_scan() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    const DK: usize = 128;
    const DV: usize = 128;
    let n_chunks = 5usize;
    let heads = 1usize;
    // Deliberately smaller than `n_chunks`: 3 groups ([0,2), [2,4), [4,5)),
    // exercising the actual point of this kernel (multiple independent
    // blocks a head), not just a single group covering everything.
    let group_size = 2usize;
    let n_groups = n_chunks.div_ceil(group_size);

    let mut a_all = vec![0.0f32; n_chunks * DK * DK];
    let mut b_all = vec![0.0f32; n_chunks * DK * DV];
    for c in 0..n_chunks {
        let ac = pseudo_random(DK * DK, 0xf001 + c as u64);
        let bc = pseudo_random(DK * DV, 0xf101 + c as u64);
        a_all[c * DK * DK..(c + 1) * DK * DK].copy_from_slice(&ac);
        b_all[c * DK * DV..(c + 1) * DK * DV].copy_from_slice(&bc);
    }

    let da = stream.clone_htod(&a_all)?;
    let db = stream.clone_htod(&b_all)?;
    let mut prefix_a = stream.alloc_zeros::<f32>(n_chunks * DK * DK)?;
    let mut prefix_b = stream.alloc_zeros::<f32>(n_chunks * DK * DV)?;
    let mut group_a = stream.alloc_zeros::<f32>(n_groups * DK * DK)?;
    let mut group_b = stream.alloc_zeros::<f32>(n_groups * DK * DV)?;
    k.gdn_group_scan(
        &mut prefix_a.as_view_mut(),
        &mut prefix_b.as_view_mut(),
        &mut group_a.as_view_mut(),
        &mut group_b.as_view_mut(),
        &da.as_view(),
        &db.as_view(),
        heads,
        n_chunks,
        group_size,
    )?;
    k.device().synchronize()?;
    let got_prefix_a = stream.clone_dtoh(&prefix_a)?;
    let got_prefix_b = stream.clone_dtoh(&prefix_b)?;
    let got_group_a = stream.clone_dtoh(&group_a)?;
    let got_group_b = stream.clone_dtoh(&group_b)?;

    // Host sequential scan: same combine rule as the standalone combine
    // test (A_new = A(c) @ A_running, B_new = A(c) @ B_running + B(c)),
    // starting from (Identity, 0).
    let combine = |a_run: &[f32], b_run: &[f32], ac: &[f32], bc: &[f32]| -> (Vec<f32>, Vec<f32>) {
        let mut a_new = vec![0.0f32; DK * DK];
        let mut b_new = vec![0.0f32; DK * DV];
        for d1 in 0..DK {
            for d2 in 0..DK {
                let mut acc = 0.0f32;
                for e in 0..DK {
                    acc += ac[d1 * DK + e] * a_run[e * DK + d2];
                }
                a_new[d1 * DK + d2] = acc;
            }
            for d2 in 0..DV {
                let mut acc = 0.0f32;
                for e in 0..DK {
                    acc += ac[d1 * DK + e] * b_run[e * DV + d2];
                }
                b_new[d1 * DV + d2] = acc + bc[d1 * DV + d2];
            }
        }
        (a_new, b_new)
    };

    let identity_a = |d1: usize, d2: usize| if d1 == d2 { 1.0f32 } else { 0.0f32 };

    for group in 0..n_groups {
        let c_start = group * group_size;
        let c_end = (c_start + group_size).min(n_chunks);

        // Reset to (Identity, 0) at the START of each group's own range --
        // groups run independently, none inherit a predecessor's state.
        let mut want_a = vec![0.0f32; DK * DK];
        let mut want_b = vec![0.0f32; DK * DV];
        for d1 in 0..DK {
            for d2 in 0..DK {
                want_a[d1 * DK + d2] = identity_a(d1, d2);
            }
        }

        for c in c_start..c_end {
            let want_prefix_slice_a = &got_prefix_a[c * DK * DK..(c + 1) * DK * DK];
            let want_prefix_slice_b = &got_prefix_b[c * DK * DV..(c + 1) * DK * DV];
            let (aworst, aat) = max_abs_diff(want_prefix_slice_a, &want_a);
            let apeak = want_a.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
            assert!(
                aworst < 1e-3 * apeak,
                "group {group} chunk {c}'s prefix A diverged by {aworst:.2e} at {aat}: \
                 got {}, want {}",
                want_prefix_slice_a[aat],
                want_a[aat]
            );
            let (bworst, bat) = max_abs_diff(want_prefix_slice_b, &want_b);
            let bpeak = want_b.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
            assert!(
                bworst < 1e-3 * bpeak,
                "group {group} chunk {c}'s prefix B diverged by {bworst:.2e} at {bat}: \
                 got {}, want {}",
                want_prefix_slice_b[bat],
                want_b[bat]
            );

            let ac = &a_all[c * DK * DK..(c + 1) * DK * DK];
            let bc = &b_all[c * DK * DV..(c + 1) * DK * DV];
            let (na, nb) = combine(&want_a, &want_b, ac, bc);
            want_a = na;
            want_b = nb;
        }

        let got_ga = &got_group_a[group * DK * DK..(group + 1) * DK * DK];
        let got_gb = &got_group_b[group * DK * DV..(group + 1) * DK * DV];
        let (gaworst, gaat) = max_abs_diff(got_ga, &want_a);
        let gapeak = want_a.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
        assert!(
            gaworst < 1e-3 * gapeak,
            "group {group}'s total A diverged by {gaworst:.2e} at {gaat}: got {}, want {}",
            got_ga[gaat],
            want_a[gaat]
        );
        let (gbworst, gbat) = max_abs_diff(got_gb, &want_b);
        let gbpeak = want_b.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
        assert!(
            gbworst < 1e-3 * gbpeak,
            "group {group}'s total B diverged by {gbworst:.2e} at {gbat}: got {}, want {}",
            got_gb[gbat],
            want_b[gbat]
        );
    }
    Ok(())
}

/// The scan-based kernel 2 replacement -- `gdn_group_scan` + `gdn_group_state`
/// + `gdn_scan_finish` -- checked directly against the already-verified,
/// real, production `gdn_chunk_state_f32` (plain kernel 2) on the SAME `w`/
/// `u`/`qkv`/`g` inputs, rather than a hand-rolled host sequential reference.
/// `t_len = 200` gives `n_chunks = 7`; `group_size = 3` gives 3 groups
/// (`[0,3)`, `[3,6)`, `[6,7)`), so the cross-group fold and the per-chunk
/// correction pass both do real work, not a single-group degenerate case
/// `the_group_scan_matches_a_host_sequential_scan` already covers. `s_init`
/// is real random data, not zero -- the whole point of `gdn_group_state` is
/// threading a real initial state through the fold, and zero would not
/// exercise that.
#[test]
fn the_scan_based_kernel2_matches_gdn_chunk_state_f32() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let t_len = 200usize;
    const GDN_CHUNK: usize = 32;
    const DKC: usize = 128;
    const DVC: usize = 128;
    let n_chunks = t_len.div_ceil(GDN_CHUNK);
    let group_size = 3usize;
    let n_groups = n_chunks.div_ceil(group_size);

    let q_small = pseudo_random(t_len * KEY_HEADS * DK, 0x9001);
    let k_small = pseudo_random(t_len * KEY_HEADS * DK, 0x9002);
    let v = pseudo_random(t_len * VAL_HEADS * DV, 0x9003);
    let g = decaying(t_len * VAL_HEADS, 0x9004, 0.7);
    let beta = betas(t_len * VAL_HEADS, 0x9005);
    let (row, off) = packed(&q_small, &k_small, &v, t_len);

    let chunk = stream.clone_htod(&row)?;
    let hg = stream.clone_htod(&g)?;
    let hb = stream.clone_htod(&beta)?;
    let f = stream.clone_htod(&[0i32])?;
    let n = stream.clone_htod(&[t_len as i32])?;
    let seqs = infero_kernels::gdn::SeqLayout {
        first_token: &f.as_view(),
        n_tokens: &n.as_view(),
        n_seqs: 1,
        total_tokens: t_len,
    };

    let mut w = stream.alloc_zeros::<f32>(n_chunks * VAL_HEADS * GDN_CHUNK * DK)?;
    let mut u = stream.alloc_zeros::<f32>(n_chunks * VAL_HEADS * GDN_CHUNK * DV)?;
    k.gdn_chunk_uw_only(
        &mut w.as_view_mut(),
        &mut u.as_view_mut(),
        &chunk.as_view(),
        &hg.as_view(),
        &hb.as_view(),
        &seqs,
        VAL_HEADS,
        KEY_HEADS,
        DK,
        DV,
        off,
        false,
    )?;

    let mut a = stream.alloc_zeros::<f32>(n_chunks * VAL_HEADS * DKC * DKC)?;
    let mut b = stream.alloc_zeros::<f32>(n_chunks * VAL_HEADS * DKC * DVC)?;
    k.gdn_chunk_ab(
        &mut a.as_view_mut(),
        &mut b.as_view_mut(),
        &w.as_view(),
        &u.as_view(),
        &chunk.as_view(),
        &hg.as_view(),
        &seqs,
        VAL_HEADS,
        KEY_HEADS,
        DK,
        DV,
        off,
        false,
    )?;
    k.device().synchronize()?;

    let s_init = pseudo_random(VAL_HEADS * DKC * DVC, 0x9006);

    // Reference: plain sequential kernel 2 on the same w/u/qkv/g, starting
    // from the same s_init.
    let mut delta_ref = stream.alloc_zeros::<f32>(n_chunks * VAL_HEADS * GDN_CHUNK * DV)?;
    let mut s_before_ref = stream.alloc_zeros::<f32>(n_chunks * VAL_HEADS * DKC * DVC)?;
    let mut state_ref = stream.clone_htod(&s_init)?;
    k.gdn_chunk_state_only(
        &mut delta_ref.as_view_mut(),
        &mut s_before_ref.as_view_mut(),
        &mut state_ref.as_view_mut(),
        &w.as_view(),
        &u.as_view(),
        &chunk.as_view(),
        &hg.as_view(),
        &seqs,
        VAL_HEADS,
        KEY_HEADS,
        DK,
        DV,
        off,
        false,
    )?;
    k.device().synchronize()?;
    let got_delta_ref = stream.clone_dtoh(&delta_ref)?;
    let got_sbefore_ref = stream.clone_dtoh(&s_before_ref)?;
    let got_state_ref = stream.clone_dtoh(&state_ref)?;

    // Scan path: group-local scan, cross-group fold, then the parallel
    // per-chunk correction pass.
    let mut prefix_a = stream.alloc_zeros::<f32>(n_chunks * VAL_HEADS * DKC * DKC)?;
    let mut prefix_b = stream.alloc_zeros::<f32>(n_chunks * VAL_HEADS * DKC * DVC)?;
    let mut group_a = stream.alloc_zeros::<f32>(n_groups * VAL_HEADS * DKC * DKC)?;
    let mut group_b = stream.alloc_zeros::<f32>(n_groups * VAL_HEADS * DKC * DVC)?;
    k.gdn_group_scan(
        &mut prefix_a.as_view_mut(),
        &mut prefix_b.as_view_mut(),
        &mut group_a.as_view_mut(),
        &mut group_b.as_view_mut(),
        &a.as_view(),
        &b.as_view(),
        VAL_HEADS,
        n_chunks,
        group_size,
    )?;

    let mut group_start_state = stream.alloc_zeros::<f32>(n_groups * VAL_HEADS * DKC * DVC)?;
    let mut state_scan = stream.clone_htod(&s_init)?;
    k.gdn_group_state(
        &mut group_start_state.as_view_mut(),
        &mut state_scan.as_view_mut(),
        &group_a.as_view(),
        &group_b.as_view(),
        VAL_HEADS,
        n_groups,
    )?;

    let mut delta_scan = stream.alloc_zeros::<f32>(n_chunks * VAL_HEADS * GDN_CHUNK * DV)?;
    let mut s_before_scan = stream.alloc_zeros::<f32>(n_chunks * VAL_HEADS * DKC * DVC)?;
    k.gdn_scan_finish(
        &mut delta_scan.as_view_mut(),
        &mut s_before_scan.as_view_mut(),
        &prefix_a.as_view(),
        &prefix_b.as_view(),
        &group_start_state.as_view(),
        &w.as_view(),
        &u.as_view(),
        &seqs,
        VAL_HEADS,
        n_chunks,
        group_size,
    )?;
    k.device().synchronize()?;
    let got_delta_scan = stream.clone_dtoh(&delta_scan)?;
    let got_sbefore_scan = stream.clone_dtoh(&s_before_scan)?;
    let got_state_scan = stream.clone_dtoh(&state_scan)?;

    let (dworst, dat) = max_abs_diff(&got_delta_scan, &got_delta_ref);
    let dpeak = got_delta_ref.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
    assert!(
        dworst < 1e-3 * dpeak,
        "scan-based delta diverged from gdn_chunk_state_f32 by {dworst:.2e} at {dat}: \
         got {}, want {}",
        got_delta_scan[dat],
        got_delta_ref[dat]
    );

    let (sworst, sat) = max_abs_diff(&got_sbefore_scan, &got_sbefore_ref);
    let speak = got_sbefore_ref.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
    assert!(
        sworst < 1e-3 * speak,
        "scan-based s_before diverged from gdn_chunk_state_f32 by {sworst:.2e} at {sat}: \
         got {}, want {}",
        got_sbefore_scan[sat],
        got_sbefore_ref[sat]
    );

    let (fworst, fat) = max_abs_diff(&got_state_scan, &got_state_ref);
    let fpeak = got_state_ref.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
    assert!(
        fworst < 1e-3 * fpeak,
        "scan-based final state diverged from gdn_chunk_state_f32 by {fworst:.2e} at {fat}: \
         got {}, want {}",
        got_state_scan[fat],
        got_state_ref[fat]
    );
    Ok(())
}

/// The batch properties, held against the fallback kernel by name.
///
/// `two_sequences_in_one_batch_keep_separate_state` above now exercises the
/// register version, because `dk = dv = 128` there. The same two properties —
/// no state bleed between slots, and a slot with no tokens neither read nor
/// written — have to hold for the version every other shape gets, and after
/// this file's shapes moved to the fast path nothing else checks that.
#[test]
fn the_fallback_kernels_keep_sequences_apart_and_idle_slots_untouched() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let lens = [3usize, 5, 0];
    let total: usize = lens.iter().sum();

    let q_small = pseudo_random(total * KEY_HEADS * DK, 0xc001);
    let k_small = pseudo_random(total * KEY_HEADS * DK, 0xc002);
    let v = pseudo_random(total * VAL_HEADS * DV, 0xc003);
    let g = decaying(total * VAL_HEADS, 0xc004, 0.6);
    let beta = betas(total * VAL_HEADS, 0xc005);
    let (row, off) = packed(&q_small, &k_small, &v, total);
    let stride = off.0;
    let per_seq = VAL_HEADS * DK * DV;

    for variant in [DeltaVariant::Global, DeltaVariant::Shared, DeltaVariant::Reg, DeltaVariant::Chunk] {
        let d_row = stream.clone_htod(&row)?;
        let d_g = stream.clone_htod(&g)?;
        let d_beta = stream.clone_htod(&beta)?;
        let first = stream.clone_htod(&[0i32, lens[0] as i32, 0])?;
        let ntok = stream.clone_htod(&[lens[0] as i32, lens[1] as i32, 0])?;
        let mut out = stream.alloc_zeros::<f32>(total * VAL_HEADS * DV)?;
        let mut state = stream.alloc_zeros::<f32>(3 * per_seq)?;
        let sentinel = vec![-12345.0f32; per_seq];
        stream.memcpy_htod(&sentinel, &mut state.slice_mut(2 * per_seq..3 * per_seq))?;

        let seqs = infero_kernels::gdn::SeqLayout {
            first_token: &first.as_view(),
            n_tokens: &ntok.as_view(),
            n_seqs: 3,
            total_tokens: total,
        };
        k.gdn_delta_rule_variant(
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
            // V heads grouped by key head, as a Hugging Face checkpoint
            // stores them.
            false,
            variant,
        )?;
        k.device().synchronize()?;
        let batched = stream.clone_dtoh(&out)?;
        let states = stream.clone_dtoh(&state)?;

        assert!(
            states[2 * per_seq..].iter().all(|v| *v == -12345.0),
            "{variant:?} wrote the idle sequence slot; for the register version \
             even the load-and-store-back round trip counts as a write"
        );

        // Each sequence alone, through the same kernel, which is what makes a
        // difference here state bleed and not a variant disagreement.
        let mut offset = 0usize;
        for (s, &len) in lens.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let (solo, _) = run_variant(
                &k,
                &row[offset * stride..(offset + len) * stride],
                &g[offset * VAL_HEADS..(offset + len) * VAL_HEADS],
                &beta[offset * VAL_HEADS..(offset + len) * VAL_HEADS],
                off,
                len,
                &[(0, len)],
                variant,
            )?;
            let slice = &batched[offset * VAL_HEADS * DV..(offset + len) * VAL_HEADS * DV];
            let (worst, at) = max_abs_diff(slice, &solo);
            let peak = solo.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            assert!(
                worst < 1e-5 * peak.max(1e-6),
                "{variant:?}: sequence {s} in a batch of three differs from the \
                 same sequence alone by {worst:.2e} at {at}; the two are sharing \
                 state"
            );
            offset += len;
        }
    }
    Ok(())
}

/// A non-square head shape, which only the fallback can serve.
///
/// `dk != dv` sends `DeltaVariant::Auto` to the global kernel, and the register
/// one refuses it outright rather than reading `dv` floats of a `dk`-tall
/// column. Both are worth pinning: the refusal is what keeps a future
/// checkpoint with a different `linear_key_head_dim` from silently reading off
/// the end of its state.
#[test]
fn a_non_square_head_shape_falls_back_and_still_matches_the_reference() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let (dk, dv) = (64usize, 128usize);
    let (key_heads, heads) = (2usize, 6usize);
    let rep = heads / key_heads;
    let t_len = 7;

    let key_dim = key_heads * dk;
    let val_dim = heads * dv;
    let stride = 2 * key_dim + val_dim;
    let off = (stride, 0, key_dim, 2 * key_dim);

    let mut qn = pseudo_random(t_len * key_dim, 0xe001);
    let mut kn = pseudo_random(t_len * key_dim, 0xe002);
    let v = pseudo_random(t_len * val_dim, 0xe003);
    let g = decaying(t_len * heads, 0xe004, 0.5);
    let beta = betas(t_len * heads, 0xe005);
    qwen35::l2norm_rows(&mut qn, dk, 1e-6);
    qwen35::l2norm_rows(&mut kn, dk, 1e-6);
    for value in qn.iter_mut() {
        *value *= (dk as f32).sqrt().recip();
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

    let q_wide = repeat_interleave(&qn, t_len, key_heads, rep, dk);
    let k_wide = repeat_interleave(&kn, t_len, key_heads, rep, dk);
    let mut want_state = vec![0.0f32; heads * dk * dv];
    let want = qwen35::gated_delta_rule(
        &q_wide, &k_wide, &v, &g, &beta, &mut want_state, t_len, heads, dk, dv, 1e-6,
    );

    let d_row = stream.clone_htod(&row)?;
    let d_g = stream.clone_htod(&g)?;
    let d_beta = stream.clone_htod(&beta)?;
    let d_first = stream.clone_htod(&[0i32])?;
    let d_ntok = stream.clone_htod(&[t_len as i32])?;
    let mut d_out = stream.alloc_zeros::<f32>(t_len * val_dim)?;
    let mut d_state = stream.alloc_zeros::<f32>(heads * dk * dv)?;
    let seqs = infero_kernels::gdn::SeqLayout {
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
        heads,
        key_heads,
        dk,
        dv,
        off,
        // V heads grouped by key head, as a Hugging Face checkpoint
        // stores them.
        false,
    )?;
    k.device().synchronize()?;
    let got = stream.clone_dtoh(&d_out)?;
    let rel = max_rel_diff(&got, &want);
    assert!(rel < 2e-3, "the 64x128 fallback diverged by {rel:.2e}");
    let peak = want_state.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let (worst, at) = max_abs_diff(&stream.clone_dtoh(&d_state)?, &want_state);
    assert!(
        worst < 1e-4 * peak,
        "the 64x128 fallback's final state diverged by {worst:.2e} at {at}"
    );

    // And the register kernel says no rather than reading past a column.
    let asked = k.gdn_delta_rule_variant(
        &mut d_out.as_view_mut(),
        &mut d_state.as_view_mut(),
        &d_row.as_view(),
        &d_g.as_view(),
        &d_beta.as_view(),
        &seqs,
        heads,
        key_heads,
        dk,
        dv,
        off,
        // V heads grouped by key head, as a Hugging Face checkpoint
        // stores them.
        false,
        DeltaVariant::Reg,
    );
    assert!(
        asked.is_err(),
        "the register kernel accepted {dk}x{dv}, which it is not instantiated for"
    );
    Ok(())
}

/// The register-blocked kernel's state must be in registers, and no output
/// check can see whether it is.
///
/// 64 floats of state a thread live in registers only if every loop over the
/// row index unrolls. Write one of them with a dynamic index and the array
/// moves to local memory, which is the same DRAM the global version streams,
/// with worse coalescing — and the kernel still computes the right answer, so
/// every other test in this file passes while the change it exists for has been
/// undone. `CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES` is the only thing that says so.
#[test]
fn the_register_state_does_not_spill() -> Result<()> {
    let k = kernels()?;
    // 2 * DV threads and 4 * DK floats of shared: what the launcher sends.
    let (regs, stat, spill) = k.gdn_kernel_registers("gdn_delta_rule_reg128_f32")?;
    let blocks = k.gdn_occupancy_blocks("gdn_delta_rule_reg128_f32", 2 * DV as u32, 4 * DK * 4)?;
    eprintln!(
        "  gdn_delta_rule_reg128_f32: {regs} regs, {stat} B static shared, \
         {spill} B spill, {blocks} blocks/SM"
    );
    assert_eq!(
        spill, 0,
        "the register-blocked delta rule spills {spill} bytes a thread; its \
         state is in local memory, not registers, and the kernel is a slower \
         version of the global one that still passes every correctness test"
    );
    // Reported, not asserted, and the reason is worth writing down: this was
    // an `assert!(blocks >= 2)` until it was run on the part the engine
    // actually targets. sm_86's ptxas gives the body 128 registers and two
    // blocks an SM; sm_120's gives it 161 and therefore one. The kernel is
    // 2.2x faster there than here regardless, so two blocks an SM was never
    // the invariant — it was one machine's way of reaching it.
    assert!(
        blocks >= 1,
        "the register-blocked delta rule fits no block an SM at all: {regs} \
         registers over 2 * {DV} threads is past this device's budget, and the \
         launch will fail rather than run slowly"
    );
    Ok(())
}

/// Same check, for `gdn_delta_rule_reg128_split4_f32` -- the column-split
/// variant `Kernels::gdn_delta_rule` picks instead of the kernel above for a
/// solo sequence (`n_seqs == 1`), where the plain kernel's `heads`-blocks-
/// only launch leaves most of the device idle. Its per-thread state (`sc`,
/// `RB` rows) is the identical size and shape to the undivided kernel's --
/// splitting columns across more blocks does not touch how many rows a
/// thread owns -- so it is exactly as vulnerable to the same silent local-
/// memory fallback, and the `qn`/`kn` lookahead arrays this variant adds
/// (`Q_LOADS` wide, to cover `DK_C` in more than one pass once a quarter of
/// `dv`'s threads is fewer than `dk`) are a second, new place the same
/// mistake could happen.
#[test]
fn the_split_column_delta_rule_does_not_spill() -> Result<()> {
    let k = kernels()?;
    // A quarter of `2 * DV` threads a block; shared is unchanged by the
    // split (see the kernel's own comment on `BUF` not depending on `G`).
    let (regs, stat, spill) = k.gdn_kernel_registers("gdn_delta_rule_reg128_split4_f32")?;
    let blocks = k.gdn_occupancy_blocks("gdn_delta_rule_reg128_split4_f32", (2 * DV / 4) as u32, 4 * DK * 4)?;
    eprintln!(
        "  gdn_delta_rule_reg128_split4_f32: {regs} regs, {stat} B static shared, \
         {spill} B spill, {blocks} blocks/SM"
    );
    assert_eq!(
        spill, 0,
        "the column-split delta rule spills {spill} bytes a thread; see \
         `the_register_state_does_not_spill`'s doc comment for why that is a \
         silent, correctness-preserving performance regression rather than a \
         test failure anywhere else"
    );
    assert!(
        blocks >= 1,
        "the column-split delta rule fits no block an SM at all: {regs} \
         registers over {} threads is past this device's budget",
        2 * DV / 4
    );
    Ok(())
}

/// Same check as above, for `gdn_chunk_delta_rule_f32`'s register-resident
/// state -- it uses the identical per-thread layout (`sc[64]`, fully unrolled)
/// so it is exactly as vulnerable to a dynamically-indexed array quietly
/// moving to local memory.
#[test]
fn the_chunked_kernels_register_state_does_not_spill() -> Result<()> {
    let k = kernels()?;
    const GDN_CHUNK: usize = 32;
    const GDN_ROW_PAD: usize = DK + 4;
    const GDN_A_STRIDE: usize = GDN_CHUNK + 1;
    let shared = (3 * GDN_CHUNK * GDN_ROW_PAD * 4)   // sk + sq + sv, f32
        + (3 * GDN_CHUNK * 4)                        // sgc + sbeta + sbg, f32
        + (GDN_CHUNK * GDN_A_STRIDE * 4)              // sA, f32
        + (GDN_CHUNK * GDN_ROW_PAD * 4)                // sW, f32
        + (GDN_CHUNK * GDN_ROW_PAD * 4);               // sD, f32
    let (regs, stat, spill) = k.gdn_kernel_registers("gdn_chunk_delta_rule_f32")?;
    let blocks = k.gdn_occupancy_blocks("gdn_chunk_delta_rule_f32", 2 * DV as u32, shared)?;
    eprintln!(
        "  gdn_chunk_delta_rule_f32: {regs} regs, {stat} B static shared, \
         {spill} B spill, {blocks} blocks/SM ({shared} B dynamic shared)"
    );
    // Unlike `gdn_delta_rule_reg128_f32` above, this one measures a small,
    // fixed 8 B spill (two floats) no matter how the per-token forward-decay
    // factor in the state-update loop is restructured -- moving it out of
    // the `#pragma unroll`'d loop to avoid recomputing it 64 times over
    // (a real, tried fix) made it *worse* (16 B), so this is accepted as
    // ptxas's actual allocation for this body rather than chased further.
    // The failure mode `the_register_state_does_not_spill` (0 B, above)
    // exists for -- the whole `sc[64]` state array falling back to local
    // memory because a loop over it isn't fully unrolled -- would spill on
    // the order of hundreds of bytes, not 8; a small fixed spill here is a
    // minor, bounded cost, not that regression class.
    assert!(
        spill <= 8,
        "the chunked delta rule spills {spill} bytes a thread, more than the \
         8 B measured and accepted when this test was written; its state may \
         have fallen back to local memory rather than staying in registers"
    );
    assert!(
        blocks >= 1,
        "the chunked delta rule fits no block an SM at all: {regs} registers \
         over 2 * {DV} threads plus {shared} B shared is past this device's \
         budget"
    );
    Ok(())
}

/// The gate kernel reads `a` and `b` at a stride, so that they can be the two
/// halves of one stacked projection instead of two buffers.
///
/// The stride is what this checks and nothing else: the same numbers laid out
/// interleaved, a token at a time, must give the same `beta` and `g` as when
/// they are contiguous. Getting the pitch wrong reads a neighbouring head's
/// gate — a plausible number in a plausible range, which is why it wants a
/// test rather than an eyeball.
#[test]
fn the_gate_reads_a_and_b_at_a_stride() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    // Not a multiple of the block, so the tail of the grid is exercised.
    let t_len = 7;
    let heads = VAL_HEADS;

    let a = pseudo_random(t_len * heads, 0xa11);
    let b = pseudo_random(t_len * heads, 0xb22);
    let a_log: Vec<f32> = (0..heads).map(|h| -5.5 + h as f32 * 0.4).collect();
    // dt_bias reaching +19 is what the softplus branch exists for, so keep it.
    let dt_bias: Vec<f32> = (0..heads)
        .map(|h| if h == 0 { 19.25 } else { -5.7 + h as f32 * 1.3 })
        .collect();

    // `a` then `b`, one token at a time — the layout a stacked `[k, 2 * heads]`
    // projection writes.
    let mut ab = Vec::with_capacity(2 * t_len * heads);
    for t in 0..t_len {
        ab.extend_from_slice(&a[t * heads..(t + 1) * heads]);
        ab.extend_from_slice(&b[t * heads..(t + 1) * heads]);
    }

    let d_al = stream.clone_htod(&a_log)?;
    let d_dt = stream.clone_htod(&dt_bias)?;

    let run = |a_src: &[f32], b_src: &[f32], a_off: usize, b_off: usize, stride: usize| -> Result<(Vec<f32>, Vec<f32>)> {
        let d_a = stream.clone_htod(a_src)?;
        let d_b = stream.clone_htod(b_src)?;
        let mut d_beta = stream.alloc_zeros::<f32>(t_len * heads)?;
        let mut d_g = stream.alloc_zeros::<f32>(t_len * heads)?;
        k.gdn_gate_decay(
            &mut d_beta.as_view_mut(),
            &mut d_g.as_view_mut(),
            &d_a.slice(a_off..a_off + t_len * stride),
            &d_b.slice(b_off..b_off + t_len * stride - b_off),
            &d_al.as_view(),
            &d_dt.as_view(),
            t_len,
            heads,
            stride,
        )?;
        k.device().synchronize()?;
        Ok((stream.clone_dtoh(&d_beta)?, stream.clone_dtoh(&d_g)?))
    };

    let (beta_sep, g_sep) = run(&a, &b, 0, 0, heads)?;
    let (beta_int, g_int) = run(&ab, &ab, 0, heads, 2 * heads)?;

    // Same arithmetic on the same values, so this is exact.
    assert_eq!(beta_sep, beta_int, "beta differs between the two layouts");
    assert_eq!(g_sep, g_int, "g differs between the two layouts");
    // And the values have to be non-trivial, or two buffers of zeros would pass.
    assert!(
        beta_sep.iter().any(|v| *v > 0.01 && *v < 0.99) && g_sep.iter().any(|v| *v < -0.01),
        "the gate produced degenerate values, so the comparison proves nothing"
    );
    Ok(())
}
