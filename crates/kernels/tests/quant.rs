//! Quantized weight kernels, validated against the same model published at a
//! different precision.
//!
//! Checking a Q8_0 decoder against a CPU Q8_0 decoder only proves the two
//! agree. Checking it against the F16 build of the same checkpoint proves it
//! decodes the numbers the quantizer meant.

mod common;

use anyhow::{Context, Result};
use half::f16;
use std::path::PathBuf;
use tuili_gguf::Gguf;
use tuili_kernels::{Kernels, WeightType};

use common::*;

const TENSOR: &str = "blk.0.ffn_gate.weight";

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

fn open(name: &str) -> Option<Gguf> {
    let p = models_dir().join(name);
    if !p.exists() {
        eprintln!("skipping: {} not downloaded", p.display());
        return None;
    }
    Some(Gguf::open(p).expect("opening model"))
}

macro_rules! need {
    ($e:expr) => {
        match $e {
            Some(v) => v,
            None => return Ok(()),
        }
    };
}

/// Upload a tensor's raw block bytes and describe its shape.
struct Loaded {
    bytes: cudarc::driver::CudaSlice<u8>,
    ty: WeightType,
    k: usize,
    n: usize,
}

fn upload(k: &Kernels, f: &Gguf, name: &str) -> Result<Loaded> {
    let info = f.tensor(name)?;
    let ty = WeightType::from_ggml(info.ty)?;
    // ggml dims are [k, n]: k elements per row, n rows.
    let (kk, n) = (info.dims[0] as usize, info.dims[1] as usize);
    let bytes = k.device().stream().clone_htod(f.data(info))?;
    Ok(Loaded {
        bytes,
        ty,
        k: kk,
        n,
    })
}

fn run_gemv(k: &Kernels, w: &Loaded, x: &[f32], n_tokens: usize) -> Result<Vec<f32>> {
    let stream = k.device().stream().clone();
    let dx = stream.clone_htod(x)?;
    let mut out = stream.alloc_zeros::<f32>(n_tokens * w.n)?;
    k.gemv(
        &mut out.as_view_mut(),
        &w.bytes.as_view(),
        w.ty,
        &dx.as_view(),
        w.k,
        w.n,
        n_tokens,
    )?;
    let host = stream.clone_dtoh(&out)?;
    k.device().synchronize()?;
    Ok(host)
}

/// The reference: the exact same tensor from the F16 build of the model.
fn f16_reference(k: &Kernels, x: &[f32], n_tokens: usize) -> Result<Option<(Vec<f32>, Loaded)>> {
    let Some(f) = open("qwen2.5-0.5b-instruct-fp16.gguf") else {
        return Ok(None);
    };
    let w = upload(k, &f, TENSOR)?;
    assert_eq!(w.ty, WeightType::F16, "expected an F16 build");
    let y = run_gemv(k, &w, x, n_tokens)?;
    Ok(Some((y, w)))
}

#[test]
fn q8_0_gemv_tracks_the_f16_build() -> Result<()> {
    let k = kernels()?;
    let quant = need!(open("qwen2.5-0.5b-instruct-q8_0.gguf"));
    let w = upload(&k, &quant, TENSOR)?;
    assert_eq!(w.ty, WeightType::Q8_0);

    let x = pseudo_random(w.k, 0x1234);
    let got = run_gemv(&k, &w, &x, 1)?;
    let (want, _) = need!(f16_reference(&k, &x, 1)?);

    let cos = cosine(&got, &want);
    assert!(cos > 0.9999, "cosine {cos} between Q8_0 and F16 outputs");

    // Q8_0 keeps ~8 bits of mantissa per 32-element block, so a few tenths of
    // a percent on a 896-term dot product is expected; anything larger means
    // the block layout is being read wrong.
    let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let (abs, at) = max_abs_diff(&got, &want);
    assert!(
        abs / scale < 0.01,
        "relative to peak {}: {abs} at {at}",
        scale
    );
    Ok(())
}

/// Every distinct weight encoding in the Q4_K_M build, each checked against the
/// same tensor in the F16 build.
///
/// A "Q4_K_M" file is a mixture. Qwen2.5-0.5B's hidden size of 896 is not a
/// multiple of the 256-element K-quant super-block, so most of its rows fall
/// back to Q5_0 — which makes this the test that actually exercises the
/// decoders this model needs.
#[test]
fn every_quant_type_tracks_the_f16_build() -> Result<()> {
    let k = kernels()?;
    let quant = need!(open("qwen2.5-0.5b-instruct-q4_k_m.gguf"));
    let full = need!(open("qwen2.5-0.5b-instruct-fp16.gguf"));

    // One representative 2-D tensor per encoding.
    let mut by_type: std::collections::BTreeMap<String, &str> = Default::default();
    for t in quant.tensors().values() {
        if t.dims.len() != 2 || !t.ty.is_quantized() {
            continue;
        }
        let ty = WeightType::from_ggml(t.ty)?;
        by_type.entry(ty.to_string()).or_insert(&t.name);
    }
    assert!(
        by_type.len() >= 3,
        "expected a mixture of encodings, found {by_type:?}"
    );

    for (ty_name, name) in &by_type {
        let w = upload(&k, &quant, name)?;
        let reference = upload(&k, &full, name)?;
        assert_eq!(reference.ty, WeightType::F16);
        assert_eq!((reference.k, reference.n), (w.k, w.n));

        let x = pseudo_random(w.k, 0x2345);
        let got = run_gemv(&k, &w, &x, 1)?;
        let want = run_gemv(&k, &reference, &x, 1)?;

        // 4.5 bits per weight is a much coarser grid than F16, but the
        // direction of the output vector still has to survive; a misread block
        // layout destroys the cosine immediately.
        let cos = cosine(&got, &want);
        assert!(cos > 0.99, "{ty_name} ({name}): cosine {cos} against F16");
        eprintln!("  {ty_name:<6} {name:<28} cosine {cos:.6}");
    }
    Ok(())
}

#[test]
fn gemv_matches_a_cpu_dot_product() -> Result<()> {
    let k = kernels()?;
    let f = need!(open("qwen2.5-0.5b-instruct-fp16.gguf"));
    let w = upload(&k, &f, TENSOR)?;

    let x = pseudo_random(w.k, 0x3456);
    let got = run_gemv(&k, &w, &x, 1)?;

    // Decode a few rows on the host straight from the mapped file.
    let info = f.tensor(TENSOR)?;
    let raw = f.data(info);
    for row in [0usize, 1, 17, w.n / 2, w.n - 1] {
        let mut acc = 0.0f64;
        for (i, &xi) in x.iter().enumerate().take(w.k) {
            let off = (row * w.k + i) * 2;
            let h = f16::from_le_bytes([raw[off], raw[off + 1]]);
            acc += h.to_f32() as f64 * xi as f64;
        }
        let want = acc as f32;
        let rel = (got[row] - want).abs() / want.abs().max(1e-3);
        assert!(rel < 1e-4, "row {row}: got {} want {want}", got[row]);
    }
    Ok(())
}

#[test]
fn prefill_gemm_agrees_with_decode_gemv() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let quant = need!(open("qwen2.5-0.5b-instruct-q8_0.gguf"));
    let w = upload(&k, &quant, TENSOR)?;

    let n_tokens = 4usize;
    let x = pseudo_random(n_tokens * w.k, 0x4567);
    let by_gemv = run_gemv(&k, &w, &x, n_tokens)?;

    // The prefill path: decode the whole matrix to f16 once, then one cuBLAS
    // call for every token.
    let mut w16 = stream.alloc_zeros::<f16>(w.k * w.n)?;
    k.dequant_to_f16(&mut w16.as_view_mut(), &w.bytes.as_view(), w.ty, w.k * w.n)?;

    let dx = stream.clone_htod(&x)?;
    let mut x16 = stream.alloc_zeros::<f16>(n_tokens * w.k)?;
    k.to_f16(&mut x16.as_view_mut(), &dx.as_view(), n_tokens * w.k)?;

    let mut out = stream.alloc_zeros::<f32>(n_tokens * w.n)?;
    k.gemm_f16(
        &mut out.as_view_mut(),
        &x16.as_view(),
        &w16.as_view(),
        n_tokens,
        w.k,
        w.n,
    )?;
    let by_gemm = stream.clone_dtoh(&out)?;
    k.device().synchronize()?;

    // The gemm path rounds the activations to f16 first, which the gemv path
    // does not, so agreement is bounded by f16 precision rather than exact.
    let cos = cosine(&by_gemm, &by_gemv);
    assert!(cos > 0.9999, "cosine {cos} between gemm and gemv");
    let scale = by_gemv.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let (abs, at) = max_abs_diff(&by_gemm, &by_gemv);
    assert!(
        abs / scale < 0.01,
        "peak-relative diff {} at {at}",
        abs / scale
    );

    // Each token must get its own row, not a broadcast of token 0.
    assert_ne!(&by_gemm[..w.n], &by_gemm[w.n..2 * w.n]);
    Ok(())
}

#[test]
fn gather_rows_matches_the_f16_build() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    let quant = need!(open("qwen2.5-0.5b-instruct-q8_0.gguf"));
    let full = need!(open("qwen2.5-0.5b-instruct-fp16.gguf"));

    let wq = upload(&k, &quant, "token_embd.weight")?;
    let wf = upload(&k, &full, "token_embd.weight")?;
    let d = wq.k;

    let rows: Vec<i32> = vec![0, 100, 5000, 151_643];
    let drows = stream.clone_htod(&rows)?;

    let mut got = stream.alloc_zeros::<f32>(rows.len() * d)?;
    k.gather_rows(
        &mut got.as_view_mut(),
        &wq.bytes.as_view(),
        wq.ty,
        &drows.as_view(),
        rows.len(),
        d,
    )?;
    let mut want = stream.alloc_zeros::<f32>(rows.len() * d)?;
    k.gather_rows(
        &mut want.as_view_mut(),
        &wf.bytes.as_view(),
        wf.ty,
        &drows.as_view(),
        rows.len(),
        d,
    )?;

    let got = stream.clone_dtoh(&got)?;
    let want = stream.clone_dtoh(&want)?;
    k.device().synchronize()?;

    for (i, row) in rows.iter().enumerate() {
        let g = &got[i * d..(i + 1) * d];
        let w = &want[i * d..(i + 1) * d];
        let cos = cosine(g, w);
        assert!(cos > 0.999, "row {row}: cosine {cos}");
    }

    // Different token ids must produce different embeddings.
    assert!(cosine(&got[..d], &got[d..2 * d]) < 0.99);
    Ok(())
}

#[test]
fn every_quantized_tensor_in_the_model_is_supported() -> Result<()> {
    for name in [
        "qwen2.5-0.5b-instruct-q8_0.gguf",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "qwen2.5-0.5b-instruct-fp16.gguf",
    ] {
        let Some(f) = open(name) else { continue };
        for t in f.tensors().values() {
            WeightType::from_ggml(t.ty).with_context(|| format!("{name}: tensor {}", t.name))?;
        }
    }
    Ok(())
}

/// The integer mat-vec against the float one, which is already known to track
/// the F16 build of the same checkpoint.
///
/// Q8_1 rounds the activation to about eight bits, so this is a check that the
/// ported bit manipulation is right, not that the two agree exactly.
#[test]
fn integer_matvec_tracks_the_float_one() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    for (file, tensor) in [
        ("qwen2.5-0.5b-instruct-q8_0.gguf", "blk.0.ffn_gate.weight"),
        (
            "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "blk.11.ffn_down.weight",
        ),
        ("qwen2.5-0.5b-instruct-q4_k_m.gguf", "blk.0.ffn_down.weight"),
    ] {
        let Some(f) = open(file) else { return Ok(()) };
        let w = upload(&k, &f, tensor)?;
        if !Kernels::has_mmvq(w.ty) {
            eprintln!(
                "  {:<6} {tensor}: no integer path, skipped",
                w.ty.to_string()
            );
            continue;
        }

        let x = pseudo_random(w.k, 0x5150);
        let want = run_gemv(&k, &w, &x, 1)?;

        let dx = stream.clone_htod(&x)?;
        let mut q8_1 = stream.alloc_zeros::<u8>(Kernels::q8_1_bytes(w.k))?;
        k.quantize_q8_1(&mut q8_1.as_view_mut(), &dx.as_view(), w.k)?;

        let mut out = stream.alloc_zeros::<f32>(w.n)?;
        k.mmvq(
            &mut out.as_view_mut(),
            &w.bytes.as_view(),
            w.ty,
            &q8_1.as_view(),
            w.k,
            w.n,
        )?;
        let got = stream.clone_dtoh(&out)?;
        k.device().synchronize()?;

        let cos = cosine(&got, &want);
        let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let (abs, at) = max_abs_diff(&got, &want);
        eprintln!(
            "  {:<6} {tensor:<24} cosine {cos:.6}, peak-relative diff {:.4}",
            w.ty.to_string(),
            abs / scale
        );
        assert!(cos > 0.999, "{} {tensor}: cosine {cos}", w.ty);
        assert!(
            abs / scale < 0.02,
            "{} {tensor}: diff {abs} at {at}, peak {scale}",
            w.ty
        );
    }
    Ok(())
}

/// The tensor-core GEMM against the float mat-vec, for a batch of tokens.
///
/// Both read the same quantized weights; the gap is entirely the 8-bit
/// activation, the same source of error the mat-vec test tolerates. What this
/// pins down is the tile staging: a mis-indexed nibble or a scale applied to
/// the wrong 32-element group would blow the cosine open immediately.
#[test]
fn tensor_core_gemm_tracks_the_float_gemv() -> Result<()> {
    let k = kernels()?;
    if k.device().arch() < 80 {
        eprintln!("  sm_{} predates the int8 mma, skipped", k.device().arch());
        return Ok(());
    }
    let stream = k.device().stream().clone();

    for (file, tensor) in [
        ("qwen2.5-0.5b-instruct-q8_0.gguf", "blk.0.ffn_gate.weight"),
        ("qwen2.5-0.5b-instruct-q8_0.gguf", "output.weight"),
        (
            "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "blk.11.ffn_down.weight",
        ),
        ("llama-3.1-8b-instruct-q4_k_m.gguf", "blk.0.ffn_down.weight"),
        ("llama-3.1-8b-instruct-q4_k_m.gguf", "blk.0.attn_v.weight"),
        ("llama-3.1-8b-instruct-q4_k_m.gguf", "output.weight"),
    ] {
        let Some(f) = open(file) else { continue };
        let w = upload(&k, &f, tensor)?;
        if !Kernels::has_mmq(w.ty) || !w.k.is_multiple_of(32) {
            eprintln!(
                "  {:<6} {tensor}: no tensor-core path, skipped",
                w.ty.to_string()
            );
            continue;
        }
        if w.ty == WeightType::Q4K && !w.k.is_multiple_of(256) {
            continue;
        }

        // Deliberately not a multiple of the 16-token tile: the edge rows are
        // where a bounds slip would hide.
        for n_tokens in [1usize, 5, 16, 19, 33, 64] {
            let x = pseudo_random(n_tokens * w.k, 0x5150 + n_tokens as u64);
            let want = run_gemv(&k, &w, &x, n_tokens)?;

            let dx = stream.clone_htod(&x)?;
            let mut q8_1 = stream.alloc_zeros::<u8>(n_tokens * Kernels::q8_1_bytes(w.k))?;
            for t in 0..n_tokens {
                let qb = Kernels::q8_1_bytes(w.k);
                k.quantize_q8_1(
                    &mut q8_1.slice_mut(t * qb..(t + 1) * qb),
                    &dx.slice(t * w.k..(t + 1) * w.k),
                    w.k,
                )?;
            }

            let mut out = stream.alloc_zeros::<f32>(n_tokens * w.n)?;
            k.mmq(
                &mut out.as_view_mut(),
                &w.bytes.as_view(),
                w.ty,
                &q8_1.as_view(),
                w.k,
                w.n,
                n_tokens,
            )?;
            let got = stream.clone_dtoh(&out)?;
            k.device().synchronize()?;

            let cos = cosine(&got, &want);
            let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let (abs, _) = max_abs_diff(&got, &want);
            eprintln!(
                "  {:<6} {tensor:<24} t={n_tokens:<3} cosine {cos:.6}, peak-relative {:.4}",
                w.ty.to_string(),
                abs / scale
            );
            assert!(cos > 0.999, "{} {tensor} t={n_tokens}: cosine {cos}", w.ty);
            assert!(
                abs / scale < 0.02,
                "{} {tensor} t={n_tokens}: peak-relative diff {}",
                w.ty,
                abs / scale
            );
        }
    }
    Ok(())
}

/// The tensor-core GEMM must give a token the same logits whatever else is in
/// the batch — bit for bit, not just close.
///
/// This is the property that lets the vocab projection use one kernel at every
/// row count. It holds because token `t` always lands at position `t % 16` of a
/// tile and accumulates over `k` in the same order with the same scales; the
/// tile count and the grid shape change around it without touching that order.
/// If it ever stops holding, seeded sampling stops being reproducible under
/// continuous batching, so it is worth an exact assertion.
#[test]
fn tensor_core_gemm_gives_the_same_answer_at_any_batch_size() -> Result<()> {
    let k = kernels()?;
    if k.device().arch() < 80 {
        return Ok(());
    }
    let stream = k.device().stream().clone();

    for (file, tensor) in [
        ("qwen2.5-0.5b-instruct-q8_0.gguf", "output.weight"),
        ("llama-3.1-8b-instruct-q4_k_m.gguf", "blk.0.ffn_gate.weight"),
        ("llama-3.1-8b-instruct-q4_k_m.gguf", "blk.0.ffn_down.weight"),
    ] {
        let Some(f) = open(file) else { continue };
        let w = upload(&k, &f, tensor)?;
        if !Kernels::has_mmq(w.ty) {
            continue;
        }
        if w.ty != WeightType::Q8_0 && !w.k.is_multiple_of(256) {
            continue;
        }

        // Row 0 of a 64-token batch must match row 0 computed alone, and the
        // token-tile count differs between the two (one tile versus two).
        let x = pseudo_random(64 * w.k, 0x1234);
        let dx = stream.clone_htod(&x)?;
        let qb = Kernels::q8_1_bytes(w.k);
        let mut q8_1 = stream.alloc_zeros::<u8>(64 * qb)?;
        k.quantize_q8_1(&mut q8_1.slice_mut(..64 * qb), &dx.as_view(), 64 * w.k)?;

        let mut solo = stream.alloc_zeros::<f32>(w.n)?;
        k.mmq(
            &mut solo.as_view_mut(),
            &w.bytes.as_view(),
            w.ty,
            &q8_1.slice(..qb),
            w.k,
            w.n,
            1,
        )?;
        let solo = stream.clone_dtoh(&solo)?;

        for batch in [5usize, 16, 17, 64] {
            let mut out = stream.alloc_zeros::<f32>(batch * w.n)?;
            k.mmq(
                &mut out.as_view_mut(),
                &w.bytes.as_view(),
                w.ty,
                &q8_1.slice(..batch * qb),
                w.k,
                w.n,
                batch,
            )?;
            let got = stream.clone_dtoh(&out)?;
            k.device().synchronize()?;
            let differing = solo
                .iter()
                .zip(&got[..w.n])
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            eprintln!(
                "  {:<6} {tensor:<24} batch {batch:<3} -> {differing} of {} differ",
                w.ty.to_string(),
                w.n
            );
            assert_eq!(
                differing, 0,
                "{} {tensor}: batch {batch} changed row 0's logits",
                w.ty
            );
        }
    }
    Ok(())
}

/// The multi-token mat-vec against the float one, per token.
#[test]
fn batched_matvec_tracks_the_float_gemv() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    for (file, tensor) in [
        ("qwen2.5-0.5b-instruct-q8_0.gguf", "blk.0.ffn_gate.weight"),
        ("llama-3.1-8b-instruct-q4_k_m.gguf", "blk.0.ffn_gate.weight"),
        ("llama-3.1-8b-instruct-q4_k_m.gguf", "blk.0.ffn_down.weight"),
    ] {
        let Some(f) = open(file) else { continue };
        let w = upload(&k, &f, tensor)?;
        if !Kernels::has_mmvq(w.ty) || !w.k.is_multiple_of(32) {
            continue;
        }
        for n_tokens in [1usize, 3, 8, 17, 32] {
            let x = pseudo_random(n_tokens * w.k, 0x99 + n_tokens as u64);
            let want = run_gemv(&k, &w, &x, n_tokens)?;
            let dx = stream.clone_htod(&x)?;
            let qb = Kernels::q8_1_bytes(w.k);
            let mut q8_1 = stream.alloc_zeros::<u8>(n_tokens * qb)?;
            k.quantize_q8_1(
                &mut q8_1.slice_mut(..n_tokens * qb),
                &dx.as_view(),
                n_tokens * w.k,
            )?;
            let mut out = stream.alloc_zeros::<f32>(n_tokens * w.n)?;
            k.mmvq_batch(
                &mut out.as_view_mut(),
                &w.bytes.as_view(),
                w.ty,
                &q8_1.as_view(),
                w.k,
                w.n,
                n_tokens,
            )?;
            let got = stream.clone_dtoh(&out)?;
            k.device().synchronize()?;
            for t in 0..n_tokens {
                let cos = cosine(&got[t * w.n..(t + 1) * w.n], &want[t * w.n..(t + 1) * w.n]);
                assert!(cos > 0.999, "{} {tensor} t={t}/{n_tokens}: cosine {cos:.6}", w.ty);
            }
        }
        eprintln!("  {:<6} {tensor:<24} all token counts agree", w.ty.to_string());
    }
    Ok(())
}

/// The split Q8_0 layout against the packed one, on the same numbers.
///
/// The two encodings hold the same quants and the same scales — the split form
/// only moves the scales out of the middle of every 32 weights and into a table
/// at the end, so a thread's run of weights is contiguous. Nothing about the
/// arithmetic or its order changes, so the answers must agree *exactly*: a
/// tolerance here would hide precisely the bug this layout can have, which is a
/// row or block landing at the wrong offset. Peak-relative closeness would
/// survive a scale table that is off by one block; equality would not.
#[test]
fn the_split_q8_0_layout_matches_the_packed_one() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let Some(f) = open("qwen2.5-0.5b-instruct-fp16.gguf") else {
        return Ok(());
    };
    let info = f.tensor(TENSOR)?;
    let (kk, n) = (info.dims[0] as usize, info.dims[1] as usize);
    let raw = f.data(info);
    let src: Vec<f16> = raw
        .chunks_exact(2)
        .map(|b| f16::from_le_bytes([b[0], b[1]]))
        .collect();
    assert_eq!(src.len(), kk * n);

    let packed = Loaded {
        bytes: stream.clone_htod(&tuili_kernels::awq::quantize_f16_to_q8_0(&src, kk)?)?,
        ty: WeightType::Q8_0,
        k: kk,
        n,
    };
    let split = Loaded {
        bytes: stream
            .clone_htod(&tuili_kernels::awq::quantize_f16_to_q8_0_split(&src, kk)?)?,
        ty: WeightType::Q8_0S,
        k: kk,
        n,
    };

    // Token counts that are not multiples of the 16-token tile: the edge rows
    // are where a bounds slip would hide.
    for n_tokens in [1usize, 5, 16, 19, 33, 64] {
        let x = pseudo_random(n_tokens * kk, 0x8055 + n_tokens as u64);
        let want = run_gemv(&k, &packed, &x, n_tokens)?;
        let got = run_gemv(&k, &split, &x, n_tokens)?;
        assert_eq!(
            got, want,
            "gemv t={n_tokens}: split and packed disagree at {:?}",
            max_abs_diff(&got, &want)
        );

        let dx = stream.clone_htod(&x)?;
        let qb = Kernels::q8_1_bytes(kk);
        let mut q8_1 = stream.alloc_zeros::<u8>(n_tokens * qb)?;
        for t in 0..n_tokens {
            k.quantize_q8_1(
                &mut q8_1.slice_mut(t * qb..(t + 1) * qb),
                &dx.slice(t * kk..(t + 1) * kk),
                kk,
            )?;
        }
        let mut a = stream.alloc_zeros::<f32>(n_tokens * n)?;
        let mut b = stream.alloc_zeros::<f32>(n_tokens * n)?;
        k.mmq(
            &mut a.as_view_mut(),
            &packed.bytes.as_view(),
            WeightType::Q8_0,
            &q8_1.as_view(),
            kk,
            n,
            n_tokens,
        )?;
        k.mmq(
            &mut b.as_view_mut(),
            &split.bytes.as_view(),
            WeightType::Q8_0S,
            &q8_1.as_view(),
            kk,
            n,
            n_tokens,
        )?;
        let (wa, wb) = (stream.clone_dtoh(&a)?, stream.clone_dtoh(&b)?);
        k.device().synchronize()?;
        let (abs, at) = max_abs_diff(&wb, &wa);
        eprintln!("  t={n_tokens:<3} mmq max diff {abs:e} at {at}");
        assert_eq!(wb, wa, "mmq t={n_tokens}: max diff {abs:e} at row {at}");
    }

    // And the dequantization, which the f16 prefill path would read.
    let total = kk * n;
    let mut da = stream.alloc_zeros::<f16>(total)?;
    let mut db = stream.alloc_zeros::<f16>(total)?;
    k.dequant_to_f16(
        &mut da.as_view_mut(),
        &packed.bytes.as_view(),
        WeightType::Q8_0,
        total,
    )?;
    k.dequant_to_f16(
        &mut db.as_view_mut(),
        &split.bytes.as_view(),
        WeightType::Q8_0S,
        total,
    )?;
    let (ha, hb) = (stream.clone_dtoh(&da)?, stream.clone_dtoh(&db)?);
    k.device().synchronize()?;
    assert!(ha == hb, "dequant: split and packed disagree");
    Ok(())
}
