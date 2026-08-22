//! The rotary kernels against the host reference for partial RoPE.
//!
//! The reference is `tuili_model::qwen35::{rope_tables, apply_partial_rope}`,
//! reached through a dev-dependency on the crate above this one. Cargo allows
//! that cycle because it exists only for test targets. Copying the reference in
//! here instead was the alternative and it is worse: two copies of a table whose
//! whole difficulty is which width the exponent is divided by will eventually
//! disagree, and the copy that the kernel is checked against is the one that
//! stops being checked against the capture.
//!
//! What can go wrong here all runs to completion:
//!
//!   * the frequency exponent divided by `d_head` instead of `rotary_dim`,
//!     which makes the table the leading slice of the full-width one rather
//!     than a compression of the same span into fewer dimensions;
//!   * the pair partner taken as `i + d_head/2` (reaching into the tail) or as
//!     `2i+1` (the interleaved convention some models genuinely use);
//!   * `q`'s unrotated tail never copied into `q_dst` in the packed path,
//!     leaving three quarters of every query head holding the previous layer's
//!     values.
//!
//! So every test below that pins one of those also asserts that the other
//! reading disagrees, and by how much. A test that only confirms the choice
//! made is compatible with the choice being wrong.

mod common;

use anyhow::Result;
use tuili_model::qwen35::{apply_partial_rope, rope_tables};

use common::*;

/// The 27B's shape, shrunk in the dimensions that do not matter. `head_dim` and
/// `rotary_dim` are the real 256 and 64, because their ratio is the whole
/// subject; the head counts are small and deliberately not a clean multiple of
/// each other, so a mixed-up head index cannot land on the right row by luck.
const N_TOKENS: usize = 5;
const N_HEADS: usize = 7;
const N_KV_HEADS: usize = 3;
const D_HEAD: usize = 256;
const ROTARY_DIM: usize = 64;
const THETA: f32 = 10_000_000.0;

fn positions() -> Vec<i32> {
    (0..N_TOKENS as i32).map(|i| i * 37 + 11).collect()
}

/// The reference's answer for one tensor.
fn reference(x: &[f32], positions: &[i32], heads: usize, rotary_dim: usize) -> Vec<f32> {
    let pos: Vec<u32> = positions.iter().map(|p| *p as u32).collect();
    let (cos, sin) = rope_tables(THETA, rotary_dim, &pos);
    let mut out = x.to_vec();
    apply_partial_rope(
        &mut out,
        &cos,
        &sin,
        positions.len(),
        heads,
        D_HEAD,
        rotary_dim,
    );
    out
}

/// Rotate with a table whose exponent is normalized by `denom` and whose pairing
/// offset is `pair_stride`, so the tests can build the mistakes explicitly and
/// show they would have been caught.
fn variant(
    x: &[f32],
    positions: &[i32],
    heads: usize,
    denom: usize,
    pair_stride: usize,
    interleaved: bool,
) -> Vec<f32> {
    let half = ROTARY_DIM / 2;
    let mut out = x.to_vec();
    for (t, &p) in positions.iter().enumerate() {
        for h in 0..heads {
            let base = (t * heads + h) * D_HEAD;
            for i in 0..half {
                let inv = (THETA as f64).powf(-((2 * i) as f64 / denom as f64));
                let (s, c) = (p as f64 * inv).sin_cos();
                let (ia, ib) = if interleaved {
                    (2 * i, 2 * i + 1)
                } else {
                    (i, i + pair_stride)
                };
                let (a, b) = (x[base + ia] as f64, x[base + ib] as f64);
                out[base + ia] = (a * c - b * s) as f32;
                out[base + ib] = (a * s + b * c) as f32;
            }
        }
    }
    out
}

/// Relative L2 over the whole tensor. For "did this change at all" questions the
/// worst single element is the wrong measure: near-zero entries move by
/// hundreds of percent while the tensor is otherwise identical.
fn relative_l2(got: &[f32], want: &[f32]) -> f32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&g, &w) in got.iter().zip(want) {
        num += ((g - w) as f64).powi(2);
        den += (w as f64).powi(2);
    }
    (num / den.max(f64::MIN_POSITIVE)).sqrt() as f32
}

/// The rotated prefix agrees with the host reference, and the tail keeps its
/// bits — for the in-place Q+K kernel.
///
/// The tolerance is set by the kernel's fast-math intrinsics: `__powf` and
/// `__sincosf` against the reference's `f64` angle. Measured 1.7e-5 on sm_120
/// at these positions, so 2e-4 leaves an order of magnitude and no more; the
/// two mistakes it must not hide land at 2.2, five orders away, which the next
/// test checks rather than assumes.
#[test]
fn partial_rope_matches_the_host_reference() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let pos = positions();
    let dpos = stream.clone_htod(&pos)?;
    let ones = stream.clone_htod(&vec![1.0f32; D_HEAD / 2])?;

    let q = pseudo_random(N_TOKENS * N_HEADS * D_HEAD, 0x51);
    let kk = pseudo_random(N_TOKENS * N_KV_HEADS * D_HEAD, 0x52);
    let (mut dq, mut dk) = (stream.clone_htod(&q)?, stream.clone_htod(&kk)?);

    k.rope_qk_partial(
        &mut dq.as_view_mut(),
        &mut dk.as_view_mut(),
        &dpos.as_view(),
        &ones.as_view(),
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

    for (name, src, got, heads) in [
        ("q", &q, &got_q, N_HEADS),
        ("k", &kk, &got_k, N_KV_HEADS),
    ] {
        let want = reference(src, &pos, heads, ROTARY_DIM);
        let (abs, at) = max_abs_diff(got, &want);
        assert!(
            abs < 2e-4,
            "{name}: kernel and host reference differ by {abs} at {at} \
             (element {}, reference {})",
            got[at],
            want[at]
        );
        eprintln!("{name}: max abs diff from the host reference {abs:.2e}");

        // The tail past `rotary_dim` must be *bit*-identical, not merely close:
        // this kernel rotates in place and must never address those lanes.
        for t in 0..N_TOKENS {
            for h in 0..heads {
                let base = (t * heads + h) * D_HEAD;
                assert_eq!(
                    &got[base + ROTARY_DIM..base + D_HEAD],
                    &src[base + ROTARY_DIM..base + D_HEAD],
                    "{name}: dims past {ROTARY_DIM} moved at token {t}, head {h}"
                );
            }
        }
    }
    Ok(())
}

/// The two mistakes the tolerance above must not be able to hide.
///
/// `2e-4` is still a wide window when the values are O(1), so this measures how far
/// each wrong reading actually lands. If either came within the tolerance the
/// test above would be blessing it.
#[test]
fn the_wrong_frequency_width_and_the_wrong_pairing_are_far_outside_the_tolerance()
-> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let pos = positions();
    let dpos = stream.clone_htod(&pos)?;
    let ones = stream.clone_htod(&vec![1.0f32; D_HEAD / 2])?;

    let q = pseudo_random(N_TOKENS * N_HEADS * D_HEAD, 0x61);
    let kk = pseudo_random(N_TOKENS * N_KV_HEADS * D_HEAD, 0x62);
    let (mut dq, mut dk) = (stream.clone_htod(&q)?, stream.clone_htod(&kk)?);
    k.rope_qk_partial(
        &mut dq.as_view_mut(),
        &mut dk.as_view_mut(),
        &dpos.as_view(),
        &ones.as_view(),
        N_TOKENS,
        N_HEADS,
        N_KV_HEADS,
        D_HEAD,
        ROTARY_DIM,
        THETA,
        1.0,
        false,
    )?;
    let got = stream.clone_dtoh(&dq)?;
    k.device().synchronize()?;

    let half = ROTARY_DIM / 2;
    // The frequency exponent divided by head_dim: the leading slice of the
    // full-width schedule instead of a table compressed into 64 dimensions.
    let by_head_dim = variant(&q, &pos, N_HEADS, D_HEAD, half, false);
    // The interleaved pairing, (2i, 2i+1).
    let interleaved = variant(&q, &pos, N_HEADS, ROTARY_DIM, half, true);
    // The full-width pairing, (i, i + d_head/2), which reaches into the tail.
    let wide_pairs = variant(&q, &pos, N_HEADS, ROTARY_DIM, D_HEAD / 2, false);

    for (name, wrong) in [
        ("exponent / head_dim", &by_head_dim),
        ("interleaved pairing", &interleaved),
        ("pairing i, i + d_head/2", &wide_pairs),
    ] {
        let (abs, at) = max_abs_diff(&got, wrong);
        assert!(
            abs > 0.1,
            "{name} came within {abs} of the kernel's answer at element {at}; \
             the 2e-4 tolerance in the reference test would have accepted it, so \
             that test is not evidence about this choice"
        );
        eprintln!("{name}: max abs diff from the kernel {abs:.3}");
    }
    Ok(())
}

/// `rotary_dim == d_head` must reproduce what shipped before partial rope
/// existed, bit for bit.
///
/// The comparison is against `rope`, the single-tensor kernel, which this change
/// did not touch and which `ops.rs` pins to a CPU reference. Same arithmetic on
/// the same inputs, so anything but an exact match means the partial path took a
/// different route through the full-width case. Both pairings, and `freq_factors`
/// deliberately not all ones — a dropped read of it is invisible at 1.0.
#[test]
fn the_full_width_case_is_bit_identical_to_the_untouched_kernel() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let (n_tokens, n_heads, n_kv_heads, d_head) = (5usize, 14usize, 3usize, 128usize);
    let theta = 500_000.0f32;
    let q = pseudo_random(n_tokens * n_heads * d_head, 0x71);
    let kk = pseudo_random(n_tokens * n_kv_heads * d_head, 0x72);
    let pos: Vec<i32> = (0..n_tokens as i32).map(|i| i * 7 + 1).collect();
    let dpos = stream.clone_htod(&pos)?;
    let ff = pseudo_random(d_head / 2, 0x73)
        .iter()
        .map(|x| x.abs() + 0.5)
        .collect::<Vec<_>>();
    let dff = stream.clone_htod(&ff)?;

    for interleaved in [false, true] {
        let (mut sq, mut sk) = (stream.clone_htod(&q)?, stream.clone_htod(&kk)?);
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

        // Through the explicit rotary width, and through the wrapper that every
        // existing call site uses — both must land on the same bits.
        for via_wrapper in [false, true] {
            let (mut fq, mut fk) = (stream.clone_htod(&q)?, stream.clone_htod(&kk)?);
            if via_wrapper {
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
            } else {
                k.rope_qk_partial(
                    &mut fq.as_view_mut(),
                    &mut fk.as_view_mut(),
                    &dpos.as_view(),
                    &dff.as_view(),
                    n_tokens,
                    n_heads,
                    n_kv_heads,
                    d_head,
                    d_head,
                    theta,
                    1.0,
                    interleaved,
                )?;
            }
            let (got_q, got_k) = (stream.clone_dtoh(&fq)?, stream.clone_dtoh(&fk)?);
            k.device().synchronize()?;
            assert_eq!(
                got_q, want_q,
                "pairing {interleaved}, wrapper {via_wrapper}: q is not \
                 bit-identical to the full-width kernel"
            );
            assert_eq!(
                got_k, want_k,
                "pairing {interleaved}, wrapper {via_wrapper}: k is not \
                 bit-identical to the full-width kernel"
            );
        }
    }
    Ok(())
}

/// And the same for the packed form, which is the path the forward pass takes.
#[test]
fn the_packed_full_width_case_is_bit_identical_to_the_untouched_kernel() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let (n_tokens, n_heads, n_kv_heads, d_head) = (5usize, 14usize, 3usize, 128usize);
    let theta = 500_000.0f32;
    let da = n_heads * d_head;
    let kv = n_kv_heads * d_head;
    let stride = da + 2 * kv;
    let packed = pseudo_random(n_tokens * stride, 0x81);
    let pos: Vec<i32> = (0..n_tokens as i32).map(|i| i * 7 + 1).collect();
    let dpos = stream.clone_htod(&pos)?;
    let ones = stream.clone_htod(&vec![1.0f32; d_head / 2])?;

    for interleaved in [false, true] {
        // Reference: unpack, then rope each tensor with the untouched kernel.
        let mut q = vec![0.0f32; n_tokens * da];
        let mut kt = vec![0.0f32; n_tokens * kv];
        for t in 0..n_tokens {
            q[t * da..(t + 1) * da].copy_from_slice(&packed[t * stride..t * stride + da]);
            kt[t * kv..(t + 1) * kv]
                .copy_from_slice(&packed[t * stride + da..t * stride + da + kv]);
        }
        let (mut sq, mut sk) = (stream.clone_htod(&q)?, stream.clone_htod(&kt)?);
        k.rope(
            &mut sq.as_view_mut(),
            &dpos.as_view(),
            &ones.as_view(),
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
            &ones.as_view(),
            n_tokens,
            n_kv_heads,
            d_head,
            theta,
            1.0,
            interleaved,
        )?;
        let (want_q, want_k) = (stream.clone_dtoh(&sq)?, stream.clone_dtoh(&sk)?);

        let mut dpacked = stream.clone_htod(&packed)?;
        // Fill q_dst with a value nothing else produces, so an unwritten
        // element is unmistakable rather than plausibly zero.
        let mut dq = stream.clone_htod(&vec![-12345.0f32; n_tokens * da])?;
        k.rope_qk_packed_partial(
            &mut dq.as_view_mut(),
            &mut dpacked.as_view_mut(),
            stride,
            0,
            da,
            &dpos.as_view(),
            &ones.as_view(),
            n_tokens,
            n_heads,
            n_kv_heads,
            d_head,
            d_head,
            theta,
            1.0,
            interleaved,
        )?;
        let got_q = stream.clone_dtoh(&dq)?;
        let got_packed = stream.clone_dtoh(&dpacked)?;
        k.device().synchronize()?;

        assert_eq!(got_q, want_q, "pairing {interleaved}: packed q");
        for t in 0..n_tokens {
            assert_eq!(
                &got_packed[t * stride + da..t * stride + da + kv],
                &want_k[t * kv..(t + 1) * kv],
                "pairing {interleaved}: packed k at token {t}"
            );
            // v sits after k in the same row and must not have been touched.
            assert_eq!(
                &got_packed[t * stride + da + kv..(t + 1) * stride],
                &packed[t * stride + da + kv..(t + 1) * stride],
                "pairing {interleaved}: v moved at token {t}"
            );
        }
    }
    Ok(())
}

/// The packed path at a partial rotary width: the prefix rotates, and q's
/// unrotated tail is *copied* rather than left behind.
///
/// This is the one failure mode partial rope adds that did not exist before.
/// `k` is rotated in place, so its tail is already where it belongs; `q` is
/// read out of the packed row and written into a separate buffer, so a kernel
/// that only writes the first `rotary_dim` leaves 192 of every 256 query
/// dimensions holding whatever was in that buffer before — the previous
/// layer's queries, which are the right shape and the right order of magnitude.
/// The sentinel fill is what makes that visible here.
#[test]
fn the_packed_path_copies_qs_unrotated_tail() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let pos = positions();
    let dpos = stream.clone_htod(&pos)?;
    let ones = stream.clone_htod(&vec![1.0f32; D_HEAD / 2])?;

    let da = N_HEADS * D_HEAD;
    let kv = N_KV_HEADS * D_HEAD;
    let stride = da + 2 * kv;
    let packed = pseudo_random(N_TOKENS * stride, 0x91);

    let mut dpacked = stream.clone_htod(&packed)?;
    const SENTINEL: f32 = -12345.0;
    let mut dq = stream.clone_htod(&vec![SENTINEL; N_TOKENS * da])?;
    k.rope_qk_packed_partial(
        &mut dq.as_view_mut(),
        &mut dpacked.as_view_mut(),
        stride,
        0,
        da,
        &dpos.as_view(),
        &ones.as_view(),
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
    let got_packed = stream.clone_dtoh(&dpacked)?;
    k.device().synchronize()?;

    assert!(
        !got_q.contains(&SENTINEL),
        "q_dst still holds the sentinel, so some of it was never written"
    );

    // q against the reference, computed on the unpacked source.
    let mut q_src = vec![0.0f32; N_TOKENS * da];
    let mut k_src = vec![0.0f32; N_TOKENS * kv];
    for t in 0..N_TOKENS {
        q_src[t * da..(t + 1) * da].copy_from_slice(&packed[t * stride..t * stride + da]);
        k_src[t * kv..(t + 1) * kv]
            .copy_from_slice(&packed[t * stride + da..t * stride + da + kv]);
    }
    let want_q = reference(&q_src, &pos, N_HEADS, ROTARY_DIM);
    let (abs, at) = max_abs_diff(&got_q, &want_q);
    assert!(abs < 2e-4, "packed q differs from the reference by {abs} at {at}");
    eprintln!("packed q: max abs diff from the host reference {abs:.2e}");

    // The tail is a copy, so it must be bit-identical to the packed source.
    for t in 0..N_TOKENS {
        for h in 0..N_HEADS {
            let dst = (t * N_HEADS + h) * D_HEAD;
            let src = t * stride + h * D_HEAD;
            assert_eq!(
                &got_q[dst + ROTARY_DIM..dst + D_HEAD],
                &packed[src + ROTARY_DIM..src + D_HEAD],
                "q's tail at token {t}, head {h} is not a copy of the source"
            );
        }
    }

    // k rotated in place: prefix against the reference, tail untouched, v whole.
    let want_k = reference(&k_src, &pos, N_KV_HEADS, ROTARY_DIM);
    for t in 0..N_TOKENS {
        let row = &got_packed[t * stride + da..t * stride + da + kv];
        let (abs, at) = max_abs_diff(row, &want_k[t * kv..(t + 1) * kv]);
        assert!(abs < 2e-4, "packed k at token {t} differs by {abs} at {at}");
        for h in 0..N_KV_HEADS {
            let off = h * D_HEAD;
            assert_eq!(
                &row[off + ROTARY_DIM..off + D_HEAD],
                &k_src[t * kv + off + ROTARY_DIM..t * kv + off + D_HEAD],
                "k's tail moved at token {t}, head {h}"
            );
        }
        assert_eq!(
            &got_packed[t * stride + da + kv..(t + 1) * stride],
            &packed[t * stride + da + kv..(t + 1) * stride],
            "v moved at token {t}"
        );
    }
    Ok(())
}

/// Shifting every position by a constant leaves the attention score matrix
/// unchanged. This needs no reference implementation: it is what "rotary
/// embeddings encode relative position" means.
///
/// What it does and does not catch is worth being precise about, because it
/// reads stronger than it is. *Any* consistent partition of the rotated
/// dimensions into pairs, rotated by an angle linear in position, is
/// shift-invariant — so this passes under the interleaved pairing and under the
/// `i, i + d_head/2` pairing too, and those are ruled out by the reference
/// comparison instead. What it does catch is a table whose phase is not linear
/// in position, a q and k that disagree about the pairing or the frequencies,
/// and a "rotation" that is not one.
///
/// The control at the end is what makes it a test rather than a tautology:
/// shifting q's positions but not k's must move the scores by orders of
/// magnitude more than the shift-invariant case does.
#[test]
fn shifting_all_positions_leaves_the_score_matrix_unchanged() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let ones = stream.clone_htod(&vec![1.0f32; D_HEAD / 2])?;
    let q = pseudo_random(N_TOKENS * N_HEADS * D_HEAD, 0xA1);
    let kk = pseudo_random(N_TOKENS * N_KV_HEADS * D_HEAD, 0xA2);

    // q and k rotated at `offset + 0..N_TOKENS`, brought back to the host.
    let roped = |q_off: i32, k_off: i32| -> Result<(Vec<f32>, Vec<f32>)> {
        let qp: Vec<i32> = (0..N_TOKENS as i32).map(|t| t + q_off).collect();
        let kp: Vec<i32> = (0..N_TOKENS as i32).map(|t| t + k_off).collect();
        let mut out = Vec::new();
        for (x, heads, p) in [
            (&q, N_HEADS, &qp),
            (&kk, N_KV_HEADS, &kp),
        ] {
            let dpos = stream.clone_htod(p)?;
            let mut dx = stream.clone_htod(x)?;
            // Only the tensor in hand matters, so the other side of the fused
            // launch gets a zero-length view's worth of heads: pass the same
            // buffer twice and take the half that was asked for.
            let mut scratch = stream.clone_htod(x)?;
            k.rope_qk_partial(
                &mut dx.as_view_mut(),
                &mut scratch.as_view_mut(),
                &dpos.as_view(),
                &ones.as_view(),
                N_TOKENS,
                heads,
                0,
                D_HEAD,
                ROTARY_DIM,
                THETA,
                1.0,
                false,
            )?;
            out.push(stream.clone_dtoh(&dx)?);
        }
        k.device().synchronize()?;
        Ok((out.remove(0), out.remove(0)))
    };

    // Query head h reads kv head h * N_KV_HEADS / N_HEADS; the grouping does
    // not matter here as long as it is the same in every run.
    let scores = |q: &[f32], kt: &[f32]| -> Vec<f32> {
        let mut s = vec![0.0f32; N_HEADS * N_TOKENS * N_TOKENS];
        for h in 0..N_HEADS {
            let kvh = h * N_KV_HEADS / N_HEADS;
            for t in 0..N_TOKENS {
                for u in 0..N_TOKENS {
                    let a = &q[(t * N_HEADS + h) * D_HEAD..(t * N_HEADS + h + 1) * D_HEAD];
                    let b =
                        &kt[(u * N_KV_HEADS + kvh) * D_HEAD..(u * N_KV_HEADS + kvh + 1) * D_HEAD];
                    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
                    s[(h * N_TOKENS + t) * N_TOKENS + u] = dot as f32;
                }
            }
        }
        s
    };

    let (q0, k0) = roped(0, 0)?;
    let base = scores(&q0, &k0);

    // A moderate shift first. The angle for the lowest-frequency column is the
    // position itself, so f32's ulp there sets the floor: at 1024 that is
    // 6e-5 rad and the scores should barely move.
    let (q1, k1) = roped(1024, 1024)?;
    let near = relative_l2(&scores(&q1, &k1), &base);
    eprintln!("shift 1024: relative L2 {near:.2e}");
    // Measured 3.8e-6 on sm_120.
    assert!(
        near < 1e-4,
        "shifting every position by 1024 moved the scores by {near:.2e} \
         relative, far more than f32 phase noise explains"
    );

    // And out where the model actually runs. At position 130000 the f32 phase
    // carries about 0.008 rad of quantization, which is the cloud
    // `qwen35_capture.rs` measures against the reference's own table; the
    // scores are allowed to move by that much and no more.
    let (q2, k2) = roped(130_000, 130_000)?;
    let far = relative_l2(&scores(&q2, &k2), &base);
    eprintln!("shift 130000: relative L2 {far:.2e}");
    // Measured 6.3e-4, two orders above the shift-1024 case, which is what
    // 0.008 rad of phase quantization buys you at that magnitude.
    assert!(
        far < 5e-3,
        "shifting every position by 130000 moved the scores by {far:.2e} \
         relative, which is more than the f32 phase quantization at that \
         magnitude accounts for"
    );

    // The control. Shift q only: now the relative positions really did change,
    // and the scores must move by far more than either case above. Without
    // this the two assertions are compatible with a kernel that ignores
    // `positions` altogether.
    let (q3, k3) = roped(1024, 0)?;
    let broken = relative_l2(&scores(&q3, &k3), &base);
    eprintln!("q shifted alone: relative L2 {broken:.2e}");
    assert!(
        broken > 100.0 * near.max(1e-9),
        "shifting q's positions without k's moved the scores by only \
         {broken:.2e}, versus {near:.2e} for a shift of both; this test cannot \
         tell a position-dependent rotation from no rotation at all"
    );
    Ok(())
}

/// A rotary width the kernel cannot pair, or one wider than the head, has to be
/// refused rather than read past the end of a row.
#[test]
fn an_impossible_rotary_width_is_refused() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let dpos = stream.clone_htod(&vec![0i32; 1])?;
    let ones = stream.clone_htod(&vec![1.0f32; D_HEAD / 2])?;
    let mut dq = stream.alloc_zeros::<f32>(D_HEAD)?;
    let mut dk = stream.alloc_zeros::<f32>(D_HEAD)?;

    for bad in [0usize, 63, D_HEAD + 2] {
        let err = k.rope_qk_partial(
            &mut dq.as_view_mut(),
            &mut dk.as_view_mut(),
            &dpos.as_view(),
            &ones.as_view(),
            1,
            1,
            1,
            D_HEAD,
            bad,
            THETA,
            1.0,
            false,
        );
        assert!(err.is_err(), "rotary_dim {bad} should have been refused");
    }

    // And a `freq_factors` buffer shorter than the pairs that rotate, which
    // would otherwise be read out of bounds on the device.
    let short = stream.clone_htod(&vec![1.0f32; ROTARY_DIM / 2 - 1])?;
    assert!(
        k.rope_qk_partial(
            &mut dq.as_view_mut(),
            &mut dk.as_view_mut(),
            &dpos.as_view(),
            &short.as_view(),
            1,
            1,
            1,
            D_HEAD,
            ROTARY_DIM,
            THETA,
            1.0,
            false,
        )
        .is_err(),
        "a freq_factors buffer shorter than rotary_dim/2 should be refused"
    );
    Ok(())
}
