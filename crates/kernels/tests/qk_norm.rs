//! Qwen3's per-head q/k normalization, against a CPU reference.
//!
//! Two layouts matter and the second is the one that can go quietly wrong: the
//! fused QKV path leaves k inside the packed `[q | k | v]` row, so the kernel
//! normalizes it at an offset with the packed row's stride. Getting that
//! arithmetic wrong reads and writes plausible-looking numbers from the
//! neighbouring projection rather than failing.

mod common;

use anyhow::Result;

use common::*;

/// `buf[t, offset + h*d_head .. +d_head] *= rsqrt(mean(sq) + eps) * weight`
fn qk_norm_ref(
    buf: &[f32],
    weight: &[f32],
    n_tokens: usize,
    n_heads: usize,
    d_head: usize,
    row_stride: usize,
    offset: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = buf.to_vec();
    for t in 0..n_tokens {
        for h in 0..n_heads {
            let base = t * row_stride + offset + h * d_head;
            let seg = &buf[base..base + d_head];
            let mean_sq = seg.iter().map(|v| v * v).sum::<f32>() / d_head as f32;
            let scale = 1.0 / (mean_sq + eps).sqrt();
            for i in 0..d_head {
                out[base + i] = seg[i] * scale * weight[i];
            }
        }
    }
    out
}

/// The contiguous case: `q`, laid out `[n_tokens, n_heads * d_head]`.
#[test]
fn the_contiguous_case_matches_the_reference() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    // Qwen3-8B's attention shape.
    let (n_tokens, n_heads, d_head) = (5usize, 32usize, 128usize);
    let row_stride = n_heads * d_head;
    let eps = 1e-6;

    let x = pseudo_random(n_tokens * row_stride, 0x51);
    let w = pseudo_random(d_head, 0x62)
        .iter()
        .map(|v| v + 1.25)
        .collect::<Vec<_>>();

    let mut dx = stream.clone_htod(&x)?;
    let dw = stream.clone_htod(&w)?;

    k.qk_norm(
        &mut dx.as_view_mut(),
        &dw.as_view(),
        n_tokens,
        n_heads,
        d_head,
        row_stride,
        0,
        eps,
    )?;
    let got = stream.clone_dtoh(&dx)?;
    k.device().synchronize()?;

    let want = qk_norm_ref(&x, &w, n_tokens, n_heads, d_head, row_stride, 0, eps);
    let rel = max_rel_diff(&got, &want);
    assert!(rel < 1e-5, "relative error {rel}");
    Ok(())
}

/// The packed case: k inside `[q | k | v]`, at `offset = d` with the row's
/// stride. Also checks that q and v either side of it come back untouched —
/// an off-by-one in the offset would corrupt them without changing k's own
/// numbers enough to notice.
#[test]
fn the_packed_case_touches_only_k() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    // Qwen3-8B: 32 q heads, 8 kv heads, d_head 128.
    let (n_tokens, n_heads, n_kv_heads, d_head) = (5usize, 32usize, 8usize, 128usize);
    let d = n_heads * d_head;
    let kv_dim = n_kv_heads * d_head;
    let fused_w = d + 2 * kv_dim;
    let eps = 1e-6;

    let x = pseudo_random(n_tokens * fused_w, 0x73);
    let w = pseudo_random(d_head, 0x84)
        .iter()
        .map(|v| v + 1.25)
        .collect::<Vec<_>>();

    let mut dx = stream.clone_htod(&x)?;
    let dw = stream.clone_htod(&w)?;

    k.qk_norm(
        &mut dx.as_view_mut(),
        &dw.as_view(),
        n_tokens,
        n_kv_heads,
        d_head,
        fused_w,
        d,
        eps,
    )?;
    let got = stream.clone_dtoh(&dx)?;
    k.device().synchronize()?;

    let want = qk_norm_ref(&x, &w, n_tokens, n_kv_heads, d_head, fused_w, d, eps);
    let rel = max_rel_diff(&got, &want);
    assert!(rel < 1e-5, "relative error {rel}");

    // q and v are the neighbours; nothing should have moved there.
    for t in 0..n_tokens {
        let row = t * fused_w;
        for i in 0..d {
            assert_eq!(got[row + i], x[row + i], "q moved at token {t} lane {i}");
        }
        for i in (d + kv_dim)..fused_w {
            assert_eq!(got[row + i], x[row + i], "v moved at token {t} lane {i}");
        }
    }
    Ok(())
}

/// Each head gets its own scale. A kernel that reduced over the whole row
/// instead of one head would still produce finite, plausible numbers, so the
/// check is that two heads with deliberately different magnitudes come out
/// with the same RMS.
#[test]
fn each_head_is_normalized_on_its_own() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    let (n_tokens, n_heads, d_head) = (1usize, 4usize, 64usize);
    let row_stride = n_heads * d_head;
    let eps = 1e-6;

    // Head h is scaled by 10^h, so a whole-row reduction cannot fake this.
    let mut x = pseudo_random(row_stride, 0x95);
    for h in 0..n_heads {
        let f = 10f32.powi(h as i32);
        for i in 0..d_head {
            x[h * d_head + i] *= f;
        }
    }
    let w = vec![1.0f32; d_head];

    let mut dx = stream.clone_htod(&x)?;
    let dw = stream.clone_htod(&w)?;
    k.qk_norm(
        &mut dx.as_view_mut(),
        &dw.as_view(),
        n_tokens,
        n_heads,
        d_head,
        row_stride,
        0,
        eps,
    )?;
    let got = stream.clone_dtoh(&dx)?;
    k.device().synchronize()?;

    let rms = |h: usize| -> f32 {
        let seg = &got[h * d_head..(h + 1) * d_head];
        (seg.iter().map(|v| v * v).sum::<f32>() / d_head as f32).sqrt()
    };
    let first = rms(0);
    for h in 1..n_heads {
        let r = rms(h);
        assert!(
            (r - first).abs() / first < 1e-4,
            "head {h} rms {r} against head 0 {first}: the reduction is not per-head"
        );
    }
    Ok(())
}
