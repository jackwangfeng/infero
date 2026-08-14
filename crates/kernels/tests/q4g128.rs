//! The repacked-AWQ mat-vec against the same weights dequantized.
//!
//! A quantized dot product can be wrong in ways a dequantization is not — the
//! nibble-to-activation-block mapping, the folded zero point, the `dp4a`
//! packing — so check the integer path against the float one on the same
//! bytes, and both against the CPU's reading of the format.

use anyhow::Result;
use tuili_cuda::Device;
use tuili_kernels::awq::{AwqTensor, unpack_row};
use tuili_kernels::{Kernels, WeightType};

/// A tensor in AWQ's own layout, built from known codes so the expected answer
/// is arithmetic rather than another implementation of the same thing.
fn synthetic(k: usize, n: usize, group: usize) -> (Vec<i32>, Vec<i32>, Vec<half::f16>) {
    let mut qweight = vec![0i32; k * n / 8];
    let mut qzeros = vec![0i32; k / group * n / 8];
    let mut scales = vec![half::f16::ZERO; k / group * n];
    const ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

    for kk in 0..k {
        for col in 0..n / 8 {
            let mut word = 0u32;
            for (i, off) in ORDER.iter().enumerate() {
                let out = col * 8 + off;
                // Something that varies in both axes and spans 0..15.
                let code = ((kk * 7 + out * 3 + kk / group) % 16) as u32;
                word |= code << (4 * i);
            }
            qweight[kk * (n / 8) + col] = word as i32;
        }
    }
    for g in 0..k / group {
        for col in 0..n / 8 {
            let mut word = 0u32;
            for (i, off) in ORDER.iter().enumerate() {
                let out = col * 8 + off;
                word |= (((g + out) % 16) as u32) << (4 * i);
            }
            qzeros[g * (n / 8) + col] = word as i32;
        }
        for out in 0..n {
            scales[g * n + out] =
                half::f16::from_f32(0.002 + ((g * 5 + out) % 23) as f32 * 0.0007);
        }
    }
    (qweight, qzeros, scales)
}

#[test]
fn the_repacked_matvec_matches_the_dequantized_weights() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();

    for (k, n) in [(128usize, 64usize), (4096, 256), (512, 1024)] {
        let (qweight, qzeros, scales) = synthetic(k, n, 128);
        let t = AwqTensor {
            qweight: &qweight,
            qzeros: &qzeros,
            scales: &scales,
            in_features: k,
            out_features: n,
        };
        let packed = t.repack()?;
        let dw = stream.clone_htod(&packed)?;

        // Reference: the CPU's own reading of the packed bytes.
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 2654435761usize) % 401) as f32 / 200.0 - 1.0)
            .collect();
        let want: Vec<f32> = (0..n)
            .map(|r| {
                unpack_row(&packed, k, r)
                    .iter()
                    .zip(&x)
                    .map(|(w, v)| w * v)
                    .sum::<f32>()
            })
            .collect();

        let dx = stream.clone_htod(&x)?;
        let mut q8 = stream.alloc_zeros::<u8>(Kernels::q8_1_bytes(k))?;
        kern.quantize_q8_1(&mut q8.as_view_mut(), &dx.as_view(), k)?;
        let mut out = stream.alloc_zeros::<f32>(n)?;
        kern.mmvq(
            &mut out.as_view_mut(),
            &dw.as_view(),
            WeightType::Q4G128,
            &q8.as_view(),
            k,
            n,
        )?;
        let got = stream.clone_dtoh(&out)?;

        // The float path on the same bytes, as a second opinion.
        let mut fout = stream.alloc_zeros::<f32>(n)?;
        kern.gemv(
            &mut fout.as_view_mut(),
            &dw.as_view(),
            WeightType::Q4G128,
            &dx.as_view(),
            k,
            n,
            1,
        )?;
        let fgot = stream.clone_dtoh(&fout)?;
        dev.synchronize()?;

        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        for r in 0..n {
            // The integer path quantizes the activation to 8 bits, so it is
            // allowed to differ by that; the float path is not.
            assert!(
                (got[r] - want[r]).abs() <= 0.02 * scale,
                "k={k} n={n} row {r}: mmvq {} vs {}",
                got[r],
                want[r]
            );
            assert!(
                (fgot[r] - want[r]).abs() <= 1e-3 * scale,
                "k={k} n={n} row {r}: gemv {} vs {}",
                fgot[r],
                want[r]
            );
        }
    }
    Ok(())
}


/// The tensor-core GEMM against the weights dequantized on the CPU.
///
/// The GEMM reads the same bytes through a completely different path than the
/// mat-vec — staging nibbles into shared memory as int8 and multiplying with
/// `mma.sync` rather than streaming them through `dp4a` — so this is a real
/// check on the tile staging: which of the eight groups a nibble run belongs
/// to, and which of them share a scale.
///
/// Both are checked against the float reference rather than against each other,
/// because the two integer paths do not agree exactly and should not be made
/// to. They form the activation sum that the zero point multiplies differently:
/// a Q8_1 block carries the sum of the original floats, which is what the GEMM
/// uses, while the mat-vec re-derives it from the rounded `int8` with a `dp4a`
/// against all-ones. That is worth about half a percent, and it is activation
/// quantization noise either way.
#[test]
fn the_tensor_core_gemm_tracks_the_dequantized_weights() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();

    let (k, n) = (1024usize, 512usize);
    let (qweight, qzeros, scales) = synthetic(k, n, 128);
    let packed = AwqTensor {
        qweight: &qweight,
        qzeros: &qzeros,
        scales: &scales,
        in_features: k,
        out_features: n,
    }
    .repack()?;
    let dw = stream.clone_htod(&packed)?;
    let rows: Vec<Vec<f32>> = (0..n).map(|r| unpack_row(&packed, k, r)).collect();

    // A ragged count on purpose: the tile is sixteen tokens wide, so 19 leaves
    // a partial second tile whose rows have to stay zero.
    for tokens in [1usize, 2, 8, 16, 19, 32] {
        let x: Vec<f32> = (0..tokens * k)
            .map(|i| ((i * 40503) % 617) as f32 / 300.0 - 1.0)
            .collect();
        let dx = stream.clone_htod(&x)?;
        let bytes = Kernels::q8_1_bytes(k);
        let mut q8 = stream.alloc_zeros::<u8>(tokens * bytes)?;
        kern.quantize_q8_1(&mut q8.as_view_mut(), &dx.as_view(), tokens * k)?;

        let mut gemm = stream.alloc_zeros::<f32>(tokens * n)?;
        kern.mmq(
            &mut gemm.as_view_mut(),
            &dw.as_view(),
            WeightType::Q4G128,
            &q8.as_view(),
            k,
            n,
            tokens,
        )?;
        let got = stream.clone_dtoh(&gemm)?;
        dev.synchronize()?;

        let mut worst = 0.0f32;
        let mut scale = 1e-6f32;
        for t in 0..tokens {
            let xt = &x[t * k..(t + 1) * k];
            for r in 0..n {
                let want: f32 = rows[r].iter().zip(xt).map(|(w, v)| w * v).sum();
                scale = scale.max(want.abs());
                worst = worst.max((got[t * n + r] - want).abs());
            }
        }
        eprintln!("  tokens={tokens:<3} worst error {:.4}% of range", 100.0 * worst / scale);
        assert!(
            worst <= 0.02 * scale,
            "tokens={tokens}: worst error {worst} against a range of {scale}"
        );
    }
    Ok(())
}

/// The `cp.async` ring-buffer variants against the same reference.
///
/// They read the activations through a shared layout nothing else uses — the
/// `block_q8_1` bytes verbatim, 36 to a group, gathered without `ldmatrix` —
/// and fill it from a pipeline whose stage buffers are reused every `stages`
/// k-tiles. Both are ways to be wrong that produce plausible numbers: a stage
/// read one tile early, or a tail tile multiplying against the quants left by
/// the tile `stages` before it. The shapes below are chosen so both happen.
///
/// `k = 384` is three 128-wide blocks, so `kb_total` is 12 and the last of the
/// two 256-wide tiles is half past the end of the row — the reuse case. `n` is
/// not a multiple of any block's row count, so the row masking is live too.
///
/// The reference here is `mmqx` of the same tile shape rather than the
/// dequantized weights, and the tolerance is tight rather than the 2% the test
/// above allows. Both variants accumulate the same groups in the same order and
/// differ only in where the activation scale is converted to float, so they
/// should agree to rounding — and at k=4096 the loose bound is not a check at
/// all: the activation quantization alone puts the *verified* kernel 2.2% off
/// the float answer, so a reference-only test would have to pass anything.
#[test]
fn the_async_gemm_matches_the_wide_tile_it_pipelines() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();

    // `k = 1152` is the shape that puts both failure modes in the same launch:
    // five k-tiles wrap a four-stage ring, and 36 quantized blocks leave the
    // fifth tile half past the end of the row, so the tail lands in a buffer
    // that still holds tile zero.
    for (k, n) in [(1024usize, 512usize), (4096, 296), (1152, 256), (384, 256)] {
        let (qweight, qzeros, scales) = synthetic(k, n, 128);
        let packed = AwqTensor {
            qweight: &qweight,
            qzeros: &qzeros,
            scales: &scales,
            in_features: k,
            out_features: n,
        }
        .repack()?;
        let dw = stream.clone_htod(&packed)?;
        // The float answer, as the anchor at the one k where 2% still means
        // something. Without it this test only proves the two kernels are wrong
        // in the same way.
        let rows: Vec<Vec<f32>> = (0..n).map(|r| unpack_row(&packed, k, r)).collect();

        // Each pipelined shape against the one it was derived from: the async
        // variants against the bare wide tile, and the register-pipelined ones
        // against the async variant of the same shape, which differs from them
        // only by the second pipeline level and accumulates in the same order.
        for (async_v, wide_v) in [
            ("mmqa4w4s4", "mmqx4w4"),
            ("mmqa4w4s2", "mmqx4w4"),
            ("mmqa4w2s4", "mmqx4w2"),
            ("mmqa2w4s4", "mmqx2w4"),
            ("mmqr4w4s4", "mmqa4w4s4"),
            ("mmqr2w4s4", "mmqa2w4s4"),
            ("mmqr2w2s4", "mmqa2w2s4"),
            // The striped partition against the same inner loop walking all of
            // k. These two are not bit-identical — a run that straddles a row
            // group boundary reaches `out` through `atomicAdd`, so the k-slices
            // of that row group are summed in launch order rather than in k
            // order — but they are the same arithmetic to rounding.
            ("mmqsr2w4s4", "mmqr2w4s4"),
            ("mmqsr2w2s4", "mmqr2w2s4"),
            // Depth 2 is the striped register pipeline exactly, so it should
            // agree bit for bit; the deeper rings only change how far ahead the
            // weight loads are issued, not what is multiplied or in what order.
            ("mmqb2w4s4d2", "mmqsr2w4s4"),
            ("mmqb2w4s4d4", "mmqsr2w4s4"),
            ("mmqb2w2s4d4", "mmqsr2w2s4"),
            // The narrow shapes, which the bandwidth probe says are the fast
            // ones. NBLK=1 is a different accumulator shape and a different
            // row-to-warp map, so it is checked against the wide tile it
            // abandons rather than against a sibling.
            ("mmqb1w4s2d2", "mmqsr2w4s4"),
            ("mmqb1w4s4d2", "mmqsr2w4s4"),
            ("mmqb1w2s4d2", "mmqsr2w4s4"),
            ("mmqb1w8s4d2", "mmqsr2w4s4"),
            // `mmql_*` reads the per-token scales at use rather than carrying
            // them; same arithmetic, fewer registers.
            ("mmql1w4s2d2", "mmqb1w4s2d2"),
            ("mmql1w4s4d2", "mmqb1w4s4d2"),
            ("mmql2w4s2d2", "mmqb2w4s4d2"),
        ] {
            for tokens in [1usize, 7, 16, 19, 32] {
                let x: Vec<f32> = (0..tokens * k)
                    .map(|i| ((i * 40503) % 617) as f32 / 300.0 - 1.0)
                    .collect();
                let dx = stream.clone_htod(&x)?;
                let mut q8 = stream.alloc_zeros::<u8>(tokens * Kernels::q8_1_bytes(k))?;
                kern.quantize_q8_1(&mut q8.as_view_mut(), &dx.as_view(), tokens * k)?;

                let run = |variant: &str| -> Result<Vec<f32>> {
                    let mut o = stream.alloc_zeros::<f32>(tokens * n)?;
                    kern.mmq_variant(
                        variant,
                        &mut o.as_view_mut(),
                        &dw.as_view(),
                        WeightType::Q4G128,
                        &q8.as_view(),
                        k,
                        n,
                        tokens,
                    )?;
                    let v = stream.clone_dtoh(&o)?;
                    dev.synchronize()?;
                    Ok(v)
                };
                let got = run(async_v)?;
                let want = run(wide_v)?;

                let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
                let worst = got
                    .iter()
                    .zip(&want)
                    .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
                assert!(
                    worst <= 1e-5 * scale,
                    "{async_v} vs {wide_v}, k={k} n={n} tokens={tokens}: \
                     worst difference {worst} against a range of {scale}"
                );

                if k > 1024 {
                    continue;
                }
                let mut off = 0.0f32;
                let mut range = 1e-6f32;
                for t in 0..tokens {
                    let xt = &x[t * k..(t + 1) * k];
                    for r in 0..n {
                        let f: f32 = rows[r].iter().zip(xt).map(|(w, v)| w * v).sum();
                        range = range.max(f.abs());
                        off = off.max((got[t * n + r] - f).abs());
                    }
                }
                assert!(
                    off <= 0.02 * range,
                    "{async_v} k={k} n={n} tokens={tokens}: {off} off the float \
                     answer, against a range of {range}"
                );
            }
        }
    }
    Ok(())
}

/// The `lop3` dequantization, and the k numbering that goes with it.
///
/// The f16 operand path replaces the whole integer epilogue with four bits
/// placed in the mantissa of f16 1024.0, and it reads the weights in an order
/// that is not the pack's own — the fragment wants k `(lane%4)*2, +1` and `+8,
/// +9` where the pack gives four consecutive weights. Nothing repacks; the
/// numbering bends instead, and the activation gather is what bends with it.
///
/// Both halves of that are wrong in the same quiet way if wrong: a real weight
/// lands at the wrong k and the product stays plausible. So check every one of
/// a block's 128 weights against the CPU's reading of the same bytes, by
/// logical k.
#[test]
fn the_lop3_dequant_puts_every_weight_at_the_k_it_claims() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();

    let (k, n) = (128usize, 64usize);
    let (qweight, qzeros, scales) = synthetic(k, n, 128);
    let packed = AwqTensor {
        qweight: &qweight,
        qzeros: &qzeros,
        scales: &scales,
        in_features: k,
        out_features: n,
    }
    .repack()?;

    // Row 0's single block. `unpack_row` is the CPU's own reading of the pack,
    // which is what the rest of this file checks the kernels against.
    let want = unpack_row(&packed, k, 0);
    let dw = stream.clone_htod(&packed[..tuili_kernels::awq::BLOCK_BYTES])?;
    let mut out = stream.alloc_zeros::<f32>(128)?;
    kern.deq4_f16_probe(&mut out.as_view_mut(), &dw.as_view())?;
    let got = stream.clone_dtoh(&out)?;
    dev.synchronize()?;

    // The scale is an f16 and so is the dequantized weight, so this is a
    // rounding comparison — but a tight one, and a misplaced weight misses it
    // by the spacing between quantization levels rather than by an ulp.
    let range = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    for kk in 0..128 {
        assert!(
            (got[kk] - want[kk]).abs() <= 1e-3 * range,
            "k={kk}: dequantized {} against {}",
            got[kk],
            want[kk]
        );
    }
    Ok(())
}

/// The f16-operand GEMM against the weights dequantized on the CPU.
///
/// This one gets a *tighter* bound than the integer kernels, and that is the
/// point of it. Those quantize the activation to eight bits and are allowed 2%;
/// this path never quantizes the activation at all, so the only error left is
/// f16 rounding on the activation and on the dequantized weight. If it needs
/// 2% to pass, something is wrong that a loose bound would hide.
#[test]
fn the_f16_gemm_tracks_the_dequantized_weights() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();

    for (k, n) in [(1024usize, 512usize), (4096, 296), (1152, 256), (384, 256)] {
        let (qweight, qzeros, scales) = synthetic(k, n, 128);
        let packed = AwqTensor {
            qweight: &qweight,
            qzeros: &qzeros,
            scales: &scales,
            in_features: k,
            out_features: n,
        }
        .repack()?;
        let dw = stream.clone_htod(&packed)?;
        let rows: Vec<Vec<f32>> = (0..n).map(|r| unpack_row(&packed, k, r)).collect();

        for variant in ["mmqf1w8s2", "mmqg1w8s2", "mmqm1w8s2", "mmqm1w4s2"] {
            for tokens in [1usize, 7, 16, 19, 32] {
                let x: Vec<f32> = (0..tokens * k)
                    .map(|i| ((i * 40503) % 617) as f32 / 300.0 - 1.0)
                    .collect();
                let dx = stream.clone_htod(&x)?;
                let mut x16 = stream.alloc_zeros::<half::f16>(tokens * k)?;
                // The `mmqm_*` shapes read A through `ldmatrix`, which means
                // the standard fragment order, which means the activations
                // arrive permuted. Same weights, same answer.
                if variant.starts_with("mmqm") {
                    kern.to_f16_kperm(&mut x16.as_view_mut(), &dx.as_view(), tokens * k)?;
                } else {
                    kern.to_f16(&mut x16.as_view_mut(), &dx.as_view(), tokens * k)?;
                }

                let mut gemm = stream.alloc_zeros::<f32>(tokens * n)?;
                kern.mmq_f16(
                    variant,
                    &mut gemm.as_view_mut(),
                    &dw.as_view(),
                    &x16.as_view(),
                    k,
                    n,
                    tokens,
                )?;
                let got = stream.clone_dtoh(&gemm)?;
                dev.synchronize()?;

                let mut worst = 0.0f32;
                let mut scale = 1e-6f32;
                for t in 0..tokens {
                    let xt = &x[t * k..(t + 1) * k];
                    for r in 0..n {
                        let want: f32 = rows[r].iter().zip(xt).map(|(w, v)| w * v).sum();
                        scale = scale.max(want.abs());
                        worst = worst.max((got[t * n + r] - want).abs());
                    }
                }
                assert!(
                    worst <= 2e-3 * scale,
                    "{variant} k={k} n={n} tokens={tokens}: worst error {worst} \
                     against a range of {scale}"
                );
            }
        }
    }
    Ok(())
}

/// Transpose a packed Q4_G128 tensor into the layout `mmqz_*` reads.
///
/// Quants first, one 64-byte block per (row, 128 weights), with the 4x4 matrix
/// of 4-byte words inside each block transposed so a lane's whole B fragment is
/// one 16-byte chunk. Scales after, one `__half2` per block. Same total bytes
/// as the 68-byte packed blocks, and every block 64-byte aligned where the
/// packed ones are not.
use tuili_kernels::awq::{transpose_words, unpack_row_t};

/// The transposed layout against the same weights dequantized.
///
/// The question this answers is whether a weight repack is worth the loader,
/// `unpack_row`, the mat-vec, the float path and every test that pins them.
/// `mmqfp_*` in `mmq.cu` guessed at it with wrong answers and measured level;
/// this is the layout, computing the right answer, so the timing that follows
/// means something.
#[test]
fn the_transposed_weight_layout_gives_the_same_answer() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();

    // Row stride `nb * 68` has to be 16-byte aligned, which needs k % 512 == 0
    // — `awq::transposable`. Every real projection width is; 1152 is not, and
    // faults rather than reading the wrong thing, which is the right failure.
    for (k, n) in [(1024usize, 512usize), (4096, 296), (2048, 256)] {
        assert!(tuili_kernels::awq::transposable(k));
        let (qweight, qzeros, scales) = synthetic(k, n, 128);
        let packed = AwqTensor {
            qweight: &qweight,
            qzeros: &qzeros,
            scales: &scales,
            in_features: k,
            out_features: n,
        }
        .repack()?;
        // Both references, so the transpose and its reader are checked against
        // each other as well as against the original pack.
        let packed_t = transpose_words(&packed, k, n);
        let rows: Vec<Vec<f32>> = (0..n).map(|r| unpack_row(&packed, k, r)).collect();
        for r in [0usize, 1, n / 2, n - 1] {
            let a = unpack_row_t(&packed_t, k, n, r);
            for (i, (x, y)) in a.iter().zip(&rows[r]).enumerate() {
                assert!((x - y).abs() < 1e-6, "row {r} element {i}: {x} vs {y}");
            }
        }
        let dw = stream.clone_htod(&packed_t)?;

        for tokens in [1usize, 7, 16, 32] {
            let x: Vec<f32> = (0..tokens * k)
                .map(|i| ((i * 40503) % 617) as f32 / 300.0 - 1.0)
                .collect();
            let dx = stream.clone_htod(&x)?;
            let mut x16 = stream.alloc_zeros::<half::f16>(tokens * k)?;
            kern.to_f16(&mut x16.as_view_mut(), &dx.as_view(), tokens * k)?;
            // The mat-vec reads this layout too, and it is the batch-of-one
            // path — a wrong word offset there is a wrong answer nobody's GEMM
            // test would catch.
            if tokens == 1 {
                let bytes = Kernels::q8_1_bytes(k);
                let mut q8 = stream.alloc_zeros::<u8>(bytes)?;
                kern.quantize_q8_1(&mut q8.as_view_mut(), &dx.as_view(), k)?;
                let mut mv = stream.alloc_zeros::<f32>(n)?;
                kern.mmvq(
                    &mut mv.as_view_mut(),
                    &dw.as_view(),
                    WeightType::Q4G128T,
                    &q8.as_view(),
                    k,
                    n,
                )?;
                let got = stream.clone_dtoh(&mv)?;
                dev.synchronize()?;
                let sc = rows
                    .iter()
                    .map(|r| r.iter().zip(&x).map(|(w, v)| w * v).sum::<f32>().abs())
                    .fold(1e-6f32, f32::max);
                for r in 0..n {
                    let want: f32 = rows[r].iter().zip(&x).map(|(w, v)| w * v).sum();
                    assert!(
                        (got[r] - want).abs() <= 0.02 * sc,
                        "mat-vec k={k} row {r}: {} vs {want}",
                        got[r]
                    );
                }
            }

            let mut gemm = stream.alloc_zeros::<f32>(tokens * n)?;
            // `mmqt*` is the TMA family and exists only on sm_90 and newer —
            // the kernel source guards it, so below that the symbol is absent.
            let mut vs = vec!["mmqz1w8s2", "mmqy1w8s2", "mmqy2w8s2", "mmqc1w8s2"];
            if dev.arch() >= 90 {
                vs.push("mmqt1w8s2");
            }
            for v in vs {
            kern.mmq_f16(
                v,
                &mut gemm.as_view_mut(),
                &dw.as_view(),
                &x16.as_view(),
                k,
                n,
                tokens,
            )?;
            let got = stream.clone_dtoh(&gemm)?;
            dev.synchronize()?;

            let mut worst = 0.0f32;
            let mut scale = 1e-6f32;
            for t in 0..tokens {
                let xt = &x[t * k..(t + 1) * k];
                for r in 0..n {
                    let want: f32 = rows[r].iter().zip(xt).map(|(w, v)| w * v).sum();
                    scale = scale.max(want.abs());
                    worst = worst.max((got[t * n + r] - want).abs());
                }
            }
            assert!(
                worst <= 2e-3 * scale,
                "{v} k={k} n={n} tokens={tokens}: worst error {worst} against {scale}"
            );
            }
        }
    }
    Ok(())
}
