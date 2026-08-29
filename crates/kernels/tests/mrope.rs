//! `rope_qk_f32`/`rope_qk_packed_f32`'s M-RoPE extension: reading one of
//! `pos_stride` position values per frequency, picked by `mrope_axis`, rather
//! than the one scalar position every model before Qwen3.5 used.
//!
//! `partial_rope.rs` already proves `pos_stride == 1` with an all-zero
//! `mrope_axis` reproduces the pre-mRoPE kernel bit for bit (`k.rope()` versus
//! `rope_qk_partial`/`rope_qk_packed_partial`) — that is the regression half of
//! this change and is not repeated here. What is untested elsewhere is whether
//! the new indexing itself is right: given three genuinely different position
//! values a token, does frequency `i` actually read the axis `mrope_axis[i]`
//! says it should, or something else that happens to produce plausible output.
//!
//! `mrope_axis`'s contents come from `infero_model::qwen35_vision::interleaved_mrope_axis`,
//! which is capture-verified against the reference implementation elsewhere
//! (`crates/model/tests/qwen35_vision.rs`) — reused here rather than
//! re-derived, so this test is answering "does the kernel apply the axis map
//! correctly" and not "is the axis map itself right", which is a different
//! question this file has no fixture to answer.

mod common;

use anyhow::Result;
use infero_model::qwen35_vision::interleaved_mrope_axis;

use common::*;

/// The real checkpoint's section: 11 frequencies read time, 11 read height,
/// 10 read width, interleaved `i % 3` rather than chunked.
const SECTION: [usize; 3] = [11, 11, 10];
const ROTARY_DIM: usize = 64; // section sums to 32 == ROTARY_DIM / 2
const D_HEAD: usize = 64; // == ROTARY_DIM: no partial-rotary tail in play here,
                          // that interaction is `partial_rope.rs`'s subject.
const N_TOKENS: usize = 4;
const N_HEADS: usize = 5;
const N_KV_HEADS: usize = 2;
const THETA: f32 = 10_000_000.0;

/// Three position rows, token-major `[T, H, W]` per token, deliberately in
/// disjoint ranges so no two axes can agree on a value by coincidence -- if
/// the kernel read the wrong axis for any frequency, the angle would be
/// wrong by a margin no amount of f32 rounding explains.
fn positions_thw() -> Vec<i32> {
    let mut out = Vec::with_capacity(N_TOKENS * 3);
    for t in 0..N_TOKENS as i32 {
        out.push(t * 13 + 1); // T: 1, 14, 27, 40
        out.push(t * 7 + 101); // H: 101, 108, 115, 122
        out.push(t * 3 + 1000); // W: 1000, 1003, 1006, 1009
    }
    out
}

/// The axis assignment the kernel is fed, from the same function production
/// code uses to build `Weights::mrope_axis`.
fn axis_map() -> Vec<i32> {
    (0..ROTARY_DIM / 2)
        .map(|i| interleaved_mrope_axis(i, SECTION) as i32)
        .collect()
}

/// Rotate `x[n_tokens, heads, D_HEAD]` with an explicit, arbitrary
/// per-frequency axis map -- the host arithmetic `rope_qk_partial` is
/// supposed to reproduce when given `positions_thw()`/`axis_map()`.
fn reference(x: &[f32], pos: &[i32], axis: &[usize], heads: usize) -> Vec<f32> {
    let half = ROTARY_DIM / 2;
    let mut out = x.to_vec();
    for t in 0..N_TOKENS {
        let triple = &pos[t * 3..t * 3 + 3];
        for h in 0..heads {
            let base = (t * heads + h) * D_HEAD;
            for i in 0..half {
                let p = triple[axis[i]] as f64;
                let inv = (THETA as f64).powf(-((2 * i) as f64 / ROTARY_DIM as f64));
                let (s, c) = (p * inv).sin_cos();
                let (a, b) = (x[base + i] as f64, x[base + i + half] as f64);
                out[base + i] = (a * c - b * s) as f32;
                out[base + i + half] = (a * s + b * c) as f32;
            }
        }
    }
    out
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// The kernel's output matches a host reference that picks each frequency's
/// position from `axis_map()`, for a token whose T/H/W disagree by orders of
/// magnitude. Anything but reading exactly the axis the map names lands far
/// outside the fast-math tolerance -- checked explicitly below rather than
/// assumed.
#[test]
fn rope_qk_reads_the_axis_mrope_axis_names() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    let pos = positions_thw();
    let axis = axis_map();
    let dpos = stream.clone_htod(&pos)?;
    let daxis = stream.clone_htod(&axis)?;
    let ones = stream.clone_htod(&vec![1.0f32; ROTARY_DIM / 2])?;

    let q = pseudo_random(N_TOKENS * N_HEADS * D_HEAD, 0xC1);
    let kk = pseudo_random(N_TOKENS * N_KV_HEADS * D_HEAD, 0xC2);
    let (mut dq, mut dk) = (stream.clone_htod(&q)?, stream.clone_htod(&kk)?);

    k.rope_qk_partial(
        &mut dq.as_view_mut(),
        &mut dk.as_view_mut(),
        &dpos.as_view(),
        &ones.as_view(),
        &daxis.as_view(),
        3,
        N_TOKENS,
        N_HEADS,
        N_KV_HEADS,
        D_HEAD,
        ROTARY_DIM,
        THETA,
        1.0,
        false,
    )?;
    let (got_q, got_k) = (stream.clone_dtoh(&dq)?, stream.clone_dtoh(&dk)?);
    k.device().synchronize()?;

    let axis_usize: Vec<usize> = axis.iter().map(|&a| a as usize).collect();
    for (name, src, got, heads) in [("q", &q, &got_q, N_HEADS), ("k", &kk, &got_k, N_KV_HEADS)] {
        let want = reference(src, &pos, &axis_usize, heads);
        let d = max_abs(got, &want);
        assert!(d < 2e-4, "{name}: kernel vs. reference max abs diff {d:.3e}");
        eprintln!("{name}: max abs diff from reference {d:.2e}");
    }
    Ok(())
}

/// Three wrong axis assignments, each of which runs to completion and
/// produces a plausible-looking rotation. The 2e-4 tolerance above must not
/// be wide enough to accept any of them.
#[test]
fn wrong_axis_readings_are_far_outside_the_tolerance() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    let pos = positions_thw();
    let axis = axis_map();
    let dpos = stream.clone_htod(&pos)?;
    let daxis = stream.clone_htod(&axis)?;
    let ones = stream.clone_htod(&vec![1.0f32; ROTARY_DIM / 2])?;

    let q = pseudo_random(N_TOKENS * N_HEADS * D_HEAD, 0xD1);
    let kk = pseudo_random(N_TOKENS * N_KV_HEADS * D_HEAD, 0xD2);
    let (mut dq, mut dk) = (stream.clone_htod(&q)?, stream.clone_htod(&kk)?);

    k.rope_qk_partial(
        &mut dq.as_view_mut(),
        &mut dk.as_view_mut(),
        &dpos.as_view(),
        &ones.as_view(),
        &daxis.as_view(),
        3,
        N_TOKENS,
        N_HEADS,
        N_KV_HEADS,
        D_HEAD,
        ROTARY_DIM,
        THETA,
        1.0,
        false,
    )?;
    let got_q = stream.clone_dtoh(&dq)?;
    k.device().synchronize()?;

    let half = ROTARY_DIM / 2;

    // (a) Qwen2-VL/2.5-VL's *chunked* layout for the same `SECTION`: the
    // first 11 frequencies are time, the next 11 height, the last 10 width --
    // instead of interleaved by `i % 3`. Same shapes, same section counts,
    // different assignment -- exactly the trap `interleaved_mrope_axis`'s own
    // doc comment names.
    let chunked_axis: Vec<usize> = {
        let mut a = Vec::with_capacity(half);
        for i in 0..half {
            a.push(if i < SECTION[0] {
                0
            } else if i < SECTION[0] + SECTION[1] {
                1
            } else {
                2
            });
        }
        a
    };
    let chunked = reference(&q, &pos, &chunked_axis, N_HEADS);

    // (b) Every frequency reads T -- the "wired pos_stride but forgot to fill
    // mrope_axis, so it stayed zeroed" bug. Numerically identical to what
    // `pos_stride: 1` would have produced on the T row alone.
    let axis_all_t = vec![0usize; half];
    let t_only = reference(&q, &pos, &axis_all_t, N_HEADS);

    // (c) Axis-major reading of the same buffer: frequency `i` reads
    // `positions[axis * n_tokens + token]` instead of
    // `positions[token * pos_stride + axis]`. Built directly against the
    // token-major buffer this test actually uploaded, using the *correct*
    // axis map -- so this isolates the indexing bug from the axis-map bug.
    let axis_major = {
        let mut out = q.clone();
        for t in 0..N_TOKENS {
            for h in 0..N_HEADS {
                let base = (t * N_HEADS + h) * D_HEAD;
                for i in 0..half {
                    let ax = axis[i] as usize;
                    // What the kernel would read if it treated `pos` as
                    // `[axis, token]` rather than `[token, axis]` -- reusing
                    // the same flat buffer, since that is exactly what a
                    // transposed-indexing bug does: same bytes, wrong stride.
                    let wrong_idx = ax * N_TOKENS + t;
                    let p = if wrong_idx < pos.len() { pos[wrong_idx] } else { 0 } as f64;
                    let inv = (THETA as f64).powf(-((2 * i) as f64 / ROTARY_DIM as f64));
                    let (s, c) = (p * inv).sin_cos();
                    let (a, b) = (q[base + i] as f64, q[base + i + half] as f64);
                    out[base + i] = (a * c - b * s) as f32;
                    out[base + i + half] = (a * s + b * c) as f32;
                }
            }
        }
        out
    };

    for (name, wrong) in [
        ("chunked (Qwen2-VL) axis layout", &chunked),
        ("every frequency reads T", &t_only),
        ("axis-major indexing", &axis_major),
    ] {
        let d = max_abs(&got_q, wrong);
        assert!(
            d > 0.1,
            "{name} came within {d:.3e} of the kernel's real output; the 2e-4 \
             tolerance in the reading test would have accepted this"
        );
        eprintln!("{name}: max abs diff from the kernel {d:.3}");
    }
    Ok(())
}

/// `pos_stride: 3` with every token's three axes forced equal degenerates to
/// the scalar case: this is the property that makes it safe to run M-RoPE's
/// code path unconditionally for a model that has it, even on a pure-text
/// request where the reference always sets `T = H = W`.
#[test]
fn equal_axes_reduce_to_the_scalar_case() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    let scalar_pos: Vec<i32> = (0..N_TOKENS as i32).map(|t| t * 37 + 5).collect();
    let triple_pos: Vec<i32> = scalar_pos.iter().flat_map(|&p| [p, p, p]).collect();
    let axis = axis_map();
    let ones = stream.clone_htod(&vec![1.0f32; ROTARY_DIM / 2])?;
    let axis0 = scalar_axis(&stream, ROTARY_DIM / 2)?;

    let q = pseudo_random(N_TOKENS * N_HEADS * D_HEAD, 0xE1);
    let kk = pseudo_random(N_TOKENS * N_KV_HEADS * D_HEAD, 0xE2);

    let (mut sq, mut sk) = (stream.clone_htod(&q)?, stream.clone_htod(&kk)?);
    let dscalar = stream.clone_htod(&scalar_pos)?;
    k.rope_qk_partial(
        &mut sq.as_view_mut(),
        &mut sk.as_view_mut(),
        &dscalar.as_view(),
        &ones.as_view(),
        &axis0.as_view(),
        1,
        N_TOKENS,
        N_HEADS,
        N_KV_HEADS,
        D_HEAD,
        ROTARY_DIM,
        THETA,
        1.0,
        false,
    )?;
    let (want_q, want_k) = (stream.clone_dtoh(&sq)?, stream.clone_dtoh(&sk)?);

    let (mut tq, mut tk) = (stream.clone_htod(&q)?, stream.clone_htod(&kk)?);
    let dtriple = stream.clone_htod(&triple_pos)?;
    let daxis = stream.clone_htod(&axis)?;
    k.rope_qk_partial(
        &mut tq.as_view_mut(),
        &mut tk.as_view_mut(),
        &dtriple.as_view(),
        &ones.as_view(),
        &daxis.as_view(),
        3,
        N_TOKENS,
        N_HEADS,
        N_KV_HEADS,
        D_HEAD,
        ROTARY_DIM,
        THETA,
        1.0,
        false,
    )?;
    let (got_q, got_k) = (stream.clone_dtoh(&tq)?, stream.clone_dtoh(&tk)?);
    k.device().synchronize()?;

    // Bit-identical, not merely close: every frequency reads a different
    // *column* of `triple_pos` than `scalar_pos`, but the same *value*, so
    // the arithmetic is identical bit for bit regardless of which axis map
    // was used to pick the column.
    assert_eq!(got_q, want_q, "q: T=H=W did not reduce to the scalar case");
    assert_eq!(got_k, want_k, "k: T=H=W did not reduce to the scalar case");
    Ok(())
}
