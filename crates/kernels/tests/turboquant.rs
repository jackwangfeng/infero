//! TurboQuant kernels against the guarantees the paper proves.
//!
//! The interesting claims are not "it roughly works" — they are specific and
//! falsifiable: the MSE quantizer should hit `d·C(f_X,b)` distortion, the
//! MSE-only inner product should be *biased low*, and adding the 1-bit QJL
//! stage should remove that bias. Each gets its own test.

mod common;

use anyhow::Result;
use half::f16;
use infero_kernels::turboquant::{Codebook, DEFAULT_SEED, DeviceTables, KvQuant, Tables};
use infero_kernels::{AttnDims, BatchLayout, Kernels};

use common::*;

const D: usize = 64;

/// A quantized cache holding `n` vectors of one head, for the tests to poke at.
struct Cache {
    codes: cudarc::driver::CudaSlice<u8>,
    signs: cudarc::driver::CudaSlice<u8>,
    scale: cudarc::driver::CudaSlice<f16>,
    gamma: cudarc::driver::CudaSlice<f16>,
}

fn alloc_cache(k: &Kernels, n: usize, bits: u8) -> Result<Cache> {
    let stream = k.device().stream();
    Ok(Cache {
        codes: stream.alloc_zeros::<u8>(n * D * bits as usize / 8)?,
        signs: stream.alloc_zeros::<u8>(n * D / 8)?,
        scale: stream.alloc_zeros::<f16>(n)?,
        gamma: stream.alloc_zeros::<f16>(n)?,
    })
}

/// Rotate `vectors` and quantize them as keys.
fn store_keys(k: &Kernels, tables: &DeviceTables, vectors: &[f32], n: usize) -> Result<Cache> {
    let stream = k.device().stream().clone();
    let bits = tables.quant.k_mse_bits();
    let cache = alloc_cache(k, n, bits)?;

    let src = stream.clone_htod(vectors)?;
    let mut rotated = stream.alloc_zeros::<f32>(n * D)?;
    k.tq_matvec(
        &mut rotated.as_view_mut(),
        &src.as_view(),
        &tables.rotation.as_view(),
        D,
        n,
    )?;

    let positions: Vec<i32> = (0..n as i32).collect();
    let dpos = stream.clone_htod(&positions)?;
    let mut cache = cache;
    k.tq_store_k(
        &mut cache.codes.as_view_mut(),
        &mut cache.signs.as_view_mut(),
        &mut cache.scale.as_view_mut(),
        &mut cache.gamma.as_view_mut(),
        &rotated.as_view(),
        &tables.qjl.as_view(),
        &dpos.as_view(),
        &tables.k_levels.as_view(),
        bits,
        1,
        D,
        n,
        n,
    )?;
    k.device().synchronize()?;
    Ok(cache)
}

/// Decode a cached vector on the host, straight from the documented layout.
///
/// Independent of the kernels on purpose: a pack/unpack pair that is wrong in
/// the same way would round-trip fine on the GPU and still be wrong.
fn host_decode(codes: &[u8], scale: f32, cb: &Codebook, host: &Tables) -> Vec<f32> {
    let bits = cb.bits as usize;
    let per_byte = 8 / bits;
    let mask = (1u8 << bits) - 1;

    let rotated: Vec<f32> = (0..D)
        .map(|i| {
            let byte = codes[i / per_byte];
            let code = (byte >> ((i % per_byte) * bits)) & mask;
            cb.levels[code as usize] * scale
        })
        .collect();

    // Πᵀ · y, with the tables held column-major.
    (0..D)
        .map(|i| {
            (0..D)
                .map(|j| host.rotation_t[j * D + i] * rotated[j])
                .sum()
        })
        .collect()
}

/// Rotate `vectors` and quantize them as values.
fn store_values(k: &Kernels, tables: &DeviceTables, vectors: &[f32], n: usize) -> Result<Cache> {
    let stream = k.device().stream().clone();
    let bits = tables.quant.v_bits();
    let mut cache = alloc_cache(k, n, bits)?;

    let src = stream.clone_htod(vectors)?;
    let mut rotated = stream.alloc_zeros::<f32>(n * D)?;
    k.tq_matvec(
        &mut rotated.as_view_mut(),
        &src.as_view(),
        &tables.rotation.as_view(),
        D,
        n,
    )?;

    let positions: Vec<i32> = (0..n as i32).collect();
    let dpos = stream.clone_htod(&positions)?;
    k.tq_store_v(
        &mut cache.codes.as_view_mut(),
        &mut cache.scale.as_view_mut(),
        &rotated.as_view(),
        &dpos.as_view(),
        &tables.v_levels.as_view(),
        bits,
        1,
        D,
        n,
        n,
    )?;
    k.device().synchronize()?;
    Ok(cache)
}

/// Unpack a cached value vector on the host: uniform dequant, no rotation
/// undone (values are never rotated back either -- see `TqBuffers`' doc
/// comment in `crates/model/src/lib.rs`). Independent of the kernel on
/// purpose, same rationale as `host_decode` above.
fn host_dequant_no_rotation(codes: &[u8], scale: f32, cb: &Codebook) -> Vec<f32> {
    let bits = cb.bits as usize;
    let per_byte = 8 / bits;
    let mask = (1u8 << bits) - 1;
    (0..D)
        .map(|i| {
            let byte = codes[i / per_byte];
            let code = (byte >> ((i % per_byte) * bits)) & mask;
            cb.levels[code as usize] * scale
        })
        .collect()
}

fn unit_vectors(n: usize, seed: u64) -> Vec<f32> {
    let mut v = pseudo_random(n * D, seed);
    for chunk in v.chunks_mut(D) {
        let norm = chunk.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in chunk.iter_mut() {
            *x /= norm;
        }
    }
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn rotation_is_an_isometry_on_the_device() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let tables = DeviceTables::new(k.device(), D, KvQuant::Tq4)?;

    let n = 64;
    let x = pseudo_random(n * D, 0x11);
    let dx = stream.clone_htod(&x)?;
    let mut rotated = stream.alloc_zeros::<f32>(n * D)?;
    let mut back = stream.alloc_zeros::<f32>(n * D)?;

    k.tq_matvec(
        &mut rotated.as_view_mut(),
        &dx.as_view(),
        &tables.rotation.as_view(),
        D,
        n,
    )?;
    k.tq_matvec(
        &mut back.as_view_mut(),
        &rotated.as_view(),
        &tables.rotation_t.as_view(),
        D,
        n,
    )?;
    let rotated_h = stream.clone_dtoh(&rotated)?;
    let back_h = stream.clone_dtoh(&back)?;
    k.device().synchronize()?;

    let (abs, at) = max_abs_diff(&back_h, &x);
    assert!(abs < 1e-4, "ΠᵀΠ is not the identity: {abs} at {at}");

    // Norms and inner products must survive, since that is what lets the
    // estimator work entirely in the rotated basis.
    for i in 0..n {
        let a = &x[i * D..(i + 1) * D];
        let ra = &rotated_h[i * D..(i + 1) * D];
        assert!((dot(a, a).sqrt() - dot(ra, ra).sqrt()).abs() < 1e-4);
        let j = (i + 1) % n;
        let b = &x[j * D..(j + 1) * D];
        let rb = &rotated_h[j * D..(j + 1) * D];
        assert!(
            (dot(a, b) - dot(ra, rb)).abs() < 1e-4,
            "inner product changed under rotation"
        );
    }
    Ok(())
}

/// Theorem 1: quantizing a unit vector at `b` bits costs `d·C(f_X,b)`, which
/// the codebook reports. Measure it on real quantized data.
#[test]
fn measured_distortion_matches_the_codebook() -> Result<()> {
    let k = kernels()?;
    let host = Tables::new(D, DEFAULT_SEED)?;

    for quant in [KvQuant::Tq2, KvQuant::Tq4] {
        let tables = DeviceTables::new(k.device(), D, quant)?;
        let n = 512;
        let x = unit_vectors(n, 0x22);
        let cache = store_keys(&k, &tables, &x, n)?;

        let stream = k.device().stream().clone();
        let codes = stream.clone_dtoh(&cache.codes)?;
        let scales = stream.clone_dtoh(&cache.scale)?;
        k.device().synchronize()?;

        let bytes = D * quant.k_mse_bits() as usize / 8;
        let mut total = 0.0f64;
        for i in 0..n {
            let decoded = host_decode(
                &codes[i * bytes..(i + 1) * bytes],
                scales[i].to_f32(),
                &tables.k_codebook,
                &host,
            );
            let err: f32 = decoded
                .iter()
                .zip(&x[i * D..(i + 1) * D])
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            total += err as f64;
        }
        let measured = total / n as f64;
        let predicted = tables.k_codebook.distortion;
        eprintln!(
            "  {quant} keys: measured D_mse = {measured:.5}, codebook predicts {predicted:.5}"
        );
        // The prediction is an expectation over the rotation; 512 vectors
        // through one fixed rotation land close but not exactly on it.
        assert!(
            (measured - predicted).abs() / predicted < 0.15,
            "{quant}: measured {measured:.5} vs predicted {predicted:.5}"
        );
    }
    Ok(())
}

/// The paper's central claim, and the reason the second stage exists: an
/// MSE-optimal quantizer shrinks inner products, and the 1-bit QJL residual
/// removes that bias.
///
/// The ablation is exact — zeroing `gamma` turns the two-stage estimator back
/// into the MSE-only one, since `gamma` is the only thing scaling the QJL term.
#[test]
fn qjl_stage_removes_the_inner_product_bias() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let tables = DeviceTables::new(k.device(), D, KvQuant::Tq2)?;

    let n_keys = 1024;
    let keys = unit_vectors(n_keys, 0x33);
    let cache = store_keys(&k, &tables, &keys, n_keys)?;

    let n_q = 32;
    let queries = unit_vectors(n_q, 0x44);
    let dq = stream.clone_htod(&queries)?;
    let mut q_rot = stream.alloc_zeros::<f32>(n_q * D)?;
    let mut q_qjl = stream.alloc_zeros::<f32>(n_q * D)?;
    k.tq_matvec(
        &mut q_rot.as_view_mut(),
        &dq.as_view(),
        &tables.rotation.as_view(),
        D,
        n_q,
    )?;
    k.tq_matvec(
        &mut q_qjl.as_view_mut(),
        &q_rot.as_view(),
        &tables.qjl.as_view(),
        D,
        n_q,
    )?;

    let dims = AttnDims {
        n_heads: 1,
        n_kv_heads: 1,
        d_head: D,
        n_slots: n_keys,
        n_tokens: n_q,
    };
    // Nothing masked: every query sees every key.
    let positions = vec![n_keys as i32 - 1; n_q];
    let dpos = stream.clone_htod(&positions)?;
    let dseq = stream.clone_htod(&vec![0i32; n_q])?;
    let dtable = stream.clone_htod(&(0..n_keys as i32).collect::<Vec<_>>())?;
    let (vseq, vpos, vtable) = (dseq.as_view(), dpos.as_view(), dtable.as_view());
    let batch = BatchLayout {
        seq_of: &vseq,
        positions: &vpos,
        slot_table: &vtable,
        table_stride: n_keys,
    };
    let zero_gamma = stream.alloc_zeros::<f16>(n_keys)?;

    let run = |gamma: &cudarc::driver::CudaSlice<f16>| -> Result<Vec<f32>> {
        let mut scores = stream.alloc_zeros::<f32>(n_q * n_keys)?;
        k.tq_attn_scores(
            &mut scores.as_view_mut(),
            &q_rot.as_view(),
            &q_qjl.as_view(),
            &cache.codes.as_view(),
            &cache.signs.as_view(),
            &cache.scale.as_view(),
            &gamma.as_view(),
            batch,
            &tables.k_levels.as_view(),
            tables.quant.k_mse_bits(),
            dims,
            n_keys,
            1.0,
            1.0,
        )?;
        let out = stream.clone_dtoh(&scores)?;
        k.device().synchronize()?;
        Ok(out)
    };

    let two_stage = run(&cache.gamma)?;
    let mse_only = run(&zero_gamma)?;

    let mut truth = Vec::with_capacity(n_q * n_keys);
    for t in 0..n_q {
        for j in 0..n_keys {
            truth.push(dot(&queries[t * D..(t + 1) * D], &keys[j * D..(j + 1) * D]));
        }
    }

    // Bias measured as the mean signed error relative to the RMS true inner
    // product, so it is comparable across scales.
    let rms = (truth.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / truth.len() as f64).sqrt();
    let bias = |est: &[f32]| -> f64 {
        est.iter()
            .zip(&truth)
            .map(|(e, t)| (*e - *t) as f64)
            .sum::<f64>()
            / est.len() as f64
            / rms
    };
    let rmse = |est: &[f32]| -> f64 {
        (est.iter()
            .zip(&truth)
            .map(|(e, t)| ((*e - *t) as f64).powi(2))
            .sum::<f64>()
            / est.len() as f64)
            .sqrt()
            / rms
    };

    // The MSE-only estimator shrinks toward zero, so its error is
    // anti-correlated with the true value. Correlation captures that more
    // robustly than the mean over a symmetric set of inner products.
    let shrinkage = |est: &[f32]| -> f64 {
        let num: f64 = est
            .iter()
            .zip(&truth)
            .map(|(e, t)| *e as f64 * *t as f64)
            .sum();
        let den: f64 = truth.iter().map(|t| (*t as f64).powi(2)).sum();
        num / den
    };

    let s_mse = shrinkage(&mse_only);
    let s_two = shrinkage(&two_stage);
    eprintln!(
        "  mse-only : slope {s_mse:.4}  bias {:+.5}  rmse {:.5}",
        bias(&mse_only),
        rmse(&mse_only)
    );
    eprintln!(
        "  two-stage: slope {s_two:.4}  bias {:+.5}  rmse {:.5}",
        bias(&two_stage),
        rmse(&two_stage)
    );

    // What a softmax actually cares about is the error that *survives* a
    // rescale: a uniform multiplicative bias is a temperature change and
    // barely moves the ranking, while independent per-key noise reshuffles it.
    // Measure each estimator's error after fitting its own optimal slope.
    let residual = |est: &[f32], slope: f64| -> f64 {
        (est.iter()
            .zip(&truth)
            .map(|(e, t)| (*e as f64 / slope - *t as f64).powi(2))
            .sum::<f64>()
            / est.len() as f64)
            .sqrt()
            / rms
    };
    let r_mse = residual(&mse_only, s_mse);
    let r_two = residual(&two_stage, s_two);
    eprintln!(
        "  after removing each estimator's own slope: mse-only {r_mse:.5}, two-stage {r_two:.5}"
    );

    // An unbiased estimator regresses onto the truth with slope 1.
    assert!(
        s_mse < 0.95,
        "the MSE-only estimator should shrink inner products, slope was {s_mse:.4}"
    );
    assert!(
        (s_two - 1.0).abs() < 0.05,
        "the two-stage estimator should be unbiased, slope was {s_two:.4}"
    );
    assert!(
        (s_two - 1.0).abs() < (s_mse - 1.0).abs(),
        "QJL made the bias worse"
    );

    // The trade the paper makes, stated as a test so a future change to the
    // QJL stage cannot quietly flip it: unbiasedness is bought with variance,
    // and the variance is what a softmax feels.
    assert!(
        r_two > r_mse,
        "expected the QJL stage to cost residual accuracy: {r_two:.5} vs {r_mse:.5}"
    );
    Ok(())
}

/// Values are quantized for MSE, so the error of a weighted sum should sit at
/// `sqrt(D_mse)` relative to the sum's own magnitude.
///
/// Random value vectors are the *hard* case: their weighted sum is a near
/// cancellation, so the signal is as small as the noise it is measured
/// against. Real value vectors are nothing like that — see the next test.
#[test]
fn weighted_sum_error_lands_where_the_distortion_predicts() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let tables = DeviceTables::new(k.device(), D, KvQuant::Tq4)?;
    let bits = tables.quant.v_bits();

    let n = 256;
    let values = unit_vectors(n, 0x55);
    let src = stream.clone_htod(&values)?;
    let mut rotated = stream.alloc_zeros::<f32>(n * D)?;
    k.tq_matvec(
        &mut rotated.as_view_mut(),
        &src.as_view(),
        &tables.rotation.as_view(),
        D,
        n,
    )?;

    let positions: Vec<i32> = (0..n as i32).collect();
    let dpos = stream.clone_htod(&positions)?;
    let mut codes = stream.alloc_zeros::<u8>(n * D * bits as usize / 8)?;
    let mut scale = stream.alloc_zeros::<f16>(n)?;
    k.tq_store_v(
        &mut codes.as_view_mut(),
        &mut scale.as_view_mut(),
        &rotated.as_view(),
        &dpos.as_view(),
        &tables.v_levels.as_view(),
        bits,
        1,
        D,
        n,
        n,
    )?;

    // A softmax-like weight vector.
    let mut weights = pseudo_random(n, 0x66);
    for w in weights.iter_mut() {
        *w = w.abs();
    }
    let total: f32 = weights.iter().sum();
    for w in weights.iter_mut() {
        *w /= total;
    }
    let dw = stream.clone_htod(&weights)?;

    let dims = AttnDims {
        n_heads: 1,
        n_kv_heads: 1,
        d_head: D,
        n_slots: n,
        n_tokens: 1,
    };
    let vseq = stream.clone_htod(&vec![0i32])?;
    let vpos = stream.clone_htod(&vec![n as i32 - 1])?;
    let vtable = stream.clone_htod(&(0..n as i32).collect::<Vec<_>>())?;
    let (sq, sp, st) = (vseq.as_view(), vpos.as_view(), vtable.as_view());
    let batch = BatchLayout {
        seq_of: &sq,
        positions: &sp,
        slot_table: &st,
        table_stride: n,
    };
    let mut acc = stream.alloc_zeros::<f32>(D)?;
    k.tq_attn_output(
        &mut acc.as_view_mut(),
        &dw.as_view(),
        &codes.as_view(),
        &scale.as_view(),
        batch,
        &tables.v_levels.as_view(),
        bits,
        dims,
        n,
    )?;
    let mut out = stream.alloc_zeros::<f32>(D)?;
    k.tq_matvec(
        &mut out.as_view_mut(),
        &acc.as_view(),
        &tables.rotation_t.as_view(),
        D,
        1,
    )?;
    let got = stream.clone_dtoh(&out)?;
    k.device().synchronize()?;

    let mut want = vec![0.0f32; D];
    for j in 0..n {
        for i in 0..D {
            want[i] += weights[j] * values[j * D + i];
        }
    }

    // With near-orthogonal values and independent per-vector errors,
    // E||error||^2 / E||sum||^2 is D_mse, so the relative error should be
    // sqrt(D_mse) up to the slack in those two approximations.
    let err_sq: f64 = got
        .iter()
        .zip(&want)
        .map(|(a, b)| ((a - b) as f64).powi(2))
        .sum();
    let sig_sq: f64 = want.iter().map(|v| (*v as f64).powi(2)).sum();
    let relative = (err_sq / sig_sq).sqrt();
    let predicted = tables.v_codebook.distortion.sqrt();
    eprintln!(
        "  incoherent sum: relative error {relative:.4}, sqrt(D_mse) = {predicted:.4}, cosine {:.6}",
        cosine(&got, &want)
    );
    assert!(
        relative < predicted * 1.6,
        "relative error {relative:.4} far above the predicted {predicted:.4}"
    );
    Ok(())
}

/// The case attention actually presents: value vectors within a head share
/// structure, so their weighted sum is coherent while the quantization errors
/// stay independent and average away.
#[test]
fn coherent_values_average_out_quantization_noise() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let tables = DeviceTables::new(k.device(), D, KvQuant::Tq4)?;
    let bits = tables.quant.v_bits();

    let n = 256;
    // A shared direction plus per-vector noise, the shape real value vectors
    // have once they have been through a trained projection.
    let base = unit_vectors(1, 0x81);
    let noise = pseudo_random(n * D, 0x82);
    let mut values = vec![0.0f32; n * D];
    for j in 0..n {
        for i in 0..D {
            values[j * D + i] = base[i] + 0.35 * noise[j * D + i];
        }
    }

    let src = stream.clone_htod(&values)?;
    let mut rotated = stream.alloc_zeros::<f32>(n * D)?;
    k.tq_matvec(
        &mut rotated.as_view_mut(),
        &src.as_view(),
        &tables.rotation.as_view(),
        D,
        n,
    )?;
    let positions: Vec<i32> = (0..n as i32).collect();
    let dpos = stream.clone_htod(&positions)?;
    let mut codes = stream.alloc_zeros::<u8>(n * D * bits as usize / 8)?;
    let mut scale = stream.alloc_zeros::<f16>(n)?;
    k.tq_store_v(
        &mut codes.as_view_mut(),
        &mut scale.as_view_mut(),
        &rotated.as_view(),
        &dpos.as_view(),
        &tables.v_levels.as_view(),
        bits,
        1,
        D,
        n,
        n,
    )?;

    let weights = vec![1.0f32 / n as f32; n];
    let dw = stream.clone_htod(&weights)?;
    let dims = AttnDims {
        n_heads: 1,
        n_kv_heads: 1,
        d_head: D,
        n_slots: n,
        n_tokens: 1,
    };
    let vseq = stream.clone_htod(&vec![0i32])?;
    let vpos = stream.clone_htod(&vec![n as i32 - 1])?;
    let vtable = stream.clone_htod(&(0..n as i32).collect::<Vec<_>>())?;
    let (sq, sp, st) = (vseq.as_view(), vpos.as_view(), vtable.as_view());
    let batch = BatchLayout {
        seq_of: &sq,
        positions: &sp,
        slot_table: &st,
        table_stride: n,
    };
    let mut acc = stream.alloc_zeros::<f32>(D)?;
    k.tq_attn_output(
        &mut acc.as_view_mut(),
        &dw.as_view(),
        &codes.as_view(),
        &scale.as_view(),
        batch,
        &tables.v_levels.as_view(),
        bits,
        dims,
        n,
    )?;
    let mut out = stream.alloc_zeros::<f32>(D)?;
    k.tq_matvec(
        &mut out.as_view_mut(),
        &acc.as_view(),
        &tables.rotation_t.as_view(),
        D,
        1,
    )?;
    let got = stream.clone_dtoh(&out)?;
    k.device().synchronize()?;

    let mut want = vec![0.0f32; D];
    for j in 0..n {
        for i in 0..D {
            want[i] += weights[j] * values[j * D + i];
        }
    }

    let cos = cosine(&got, &want);
    eprintln!("  coherent sum cosine vs exact: {cos:.6}");
    assert!(cos > 0.9995, "cosine {cos}");
    Ok(())
}

/// Averaging many vectors is where MSE quantization is at its best: the
/// per-vector errors are close to independent and largely cancel.
#[test]
fn averaging_suppresses_quantization_noise() -> Result<()> {
    let k = kernels()?;
    let tables = DeviceTables::new(k.device(), D, KvQuant::Tq2)?;
    let host = Tables::new(D, DEFAULT_SEED)?;
    let stream = k.device().stream().clone();

    let n = 1024;
    let x = unit_vectors(n, 0x77);
    let cache = store_keys(&k, &tables, &x, n)?;
    let codes = stream.clone_dtoh(&cache.codes)?;
    let scales = stream.clone_dtoh(&cache.scale)?;
    k.device().synchronize()?;

    let bytes = D * tables.quant.k_mse_bits() as usize / 8;
    let mut mean_exact = vec![0.0f64; D];
    let mut mean_quant = vec![0.0f64; D];
    let mut per_vector_err = 0.0f64;

    for i in 0..n {
        let decoded = host_decode(
            &codes[i * bytes..(i + 1) * bytes],
            scales[i].to_f32(),
            &tables.k_codebook,
            &host,
        );
        for j in 0..D {
            mean_exact[j] += x[i * D + j] as f64 / n as f64;
            mean_quant[j] += decoded[j] as f64 / n as f64;
        }
        per_vector_err += decoded
            .iter()
            .zip(&x[i * D..(i + 1) * D])
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            / n as f64;
    }

    let mean_err: f64 = mean_exact
        .iter()
        .zip(&mean_quant)
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    eprintln!(
        "  per-vector MSE {per_vector_err:.5}, error of the mean {mean_err:.7} ({:.0}x smaller)",
        per_vector_err / mean_err.max(1e-12)
    );
    assert!(
        mean_err < per_vector_err / 50.0,
        "quantization error did not average out: {mean_err} vs {per_vector_err}"
    );
    Ok(())
}

/// The MSE-only ablation, `tq4-mse`: with `qjl_scale` zero there is no second
/// estimator term to fold in, and a dequantized key element is exactly
/// `level * scale`. `k_signs`/`k_gamma` are still passed (the cache carries
/// them either way) and must not touch the output.
///
/// Values are the same at every setting — `tq_attn_output`'s value unpack has
/// no sign/gamma term at all — so this is also the value side's only coverage.
#[test]
fn tq_dequant_kv_matches_store_then_manual_unpack() -> Result<()> {
    let k = kernels()?;
    let tables = DeviceTables::new(k.device(), D, KvQuant::Tq4Mse)?;
    assert!(
        !tables.quant.uses_qjl() && tables.quant.qjl_scale() == 0.0,
        "test setup: tq4-mse is the QJL-free ablation"
    );
    // `DeviceTables` already carries `k_codebook`/`v_codebook: Codebook` as
    // plain host-computed fields (`Codebook::solve` runs on the host,
    // `turboquant.rs:810-811`) -- no separate device->host download needed.
    let n = 5usize;
    let k_bits = tables.quant.k_mse_bits();
    let v_bits = tables.quant.v_bits();

    let keys = unit_vectors(n, 11);
    let values = unit_vectors(n, 12);
    let k_cache = store_keys(&k, &tables, &keys, n)?;
    let v_cache = store_values(&k, &tables, &values, n)?;

    let stream = k.device().stream().clone();
    let slots: Vec<i32> = (0..n as i32).collect();
    let d_slots = stream.clone_htod(&slots)?;
    let mut d_dequant_k = stream.alloc_zeros::<half::f16>(n * D)?;
    let mut d_dequant_v = stream.alloc_zeros::<half::f16>(n * D)?;

    k.tq_dequant_kv(
        &mut d_dequant_k.as_view_mut(),
        &mut d_dequant_v.as_view_mut(),
        &k_cache.codes.as_view(),
        &k_cache.signs.as_view(),
        &k_cache.scale.as_view(),
        &k_cache.gamma.as_view(),
        &v_cache.codes.as_view(),
        &v_cache.scale.as_view(),
        &d_slots.as_view(),
        &tables.k_levels.as_view(),
        k_bits,
        &tables.v_levels.as_view(),
        v_bits,
        &tables.qjl_t.as_view(),
        tables.quant.qjl_scale(),
        1,
        D,
        n,
        n,
    )?;
    k.device().synchronize()?;

    let got_k = stream.clone_dtoh(&d_dequant_k)?;
    let host_k_scale = stream.clone_dtoh(&k_cache.scale)?;
    let host_k_codes = stream.clone_dtoh(&k_cache.codes)?;
    let bytes_per_vec_k = D * k_bits as usize / 8;
    for v in 0..n {
        let expect = host_dequant_no_rotation(
            &host_k_codes[v * bytes_per_vec_k..(v + 1) * bytes_per_vec_k],
            host_k_scale[v].to_f32(),
            &tables.k_codebook,
        );
        let got: Vec<f32> = got_k[v * D..(v + 1) * D]
            .iter()
            .map(|x| x.to_f32())
            .collect();
        for (g, e) in got.iter().zip(&expect) {
            assert!((g - e).abs() < 1e-3, "key vector {v} mismatch: {g} vs {e}");
        }
    }

    let got_v = stream.clone_dtoh(&d_dequant_v)?;
    let host_v_scale = stream.clone_dtoh(&v_cache.scale)?;
    let host_v_codes = stream.clone_dtoh(&v_cache.codes)?;
    let bytes_per_vec_v = D * v_bits as usize / 8;
    for v in 0..n {
        let expect = host_dequant_no_rotation(
            &host_v_codes[v * bytes_per_vec_v..(v + 1) * bytes_per_vec_v],
            host_v_scale[v].to_f32(),
            &tables.v_codebook,
        );
        let got: Vec<f32> = got_v[v * D..(v + 1) * D]
            .iter()
            .map(|x| x.to_f32())
            .collect();
        for (g, e) in got.iter().zip(&expect) {
            assert!(
                (g - e).abs() < 1e-3,
                "value vector {v} mismatch: {g} vs {e}"
            );
        }
    }
    Ok(())
}

/// The full per-key effective vector the QJL-aware `tq_dequant_kv` must
/// produce:
///
/// ```text
/// k_eff = scale·cb[code] + qjl_scale·(sqrt(pi/2)/d)·gamma·(Qᵀ·s)
/// ```
///
/// where `Q` is `tables.qjl`, `Qᵀ` is `tables.qjl_t`, and `s` is this key's
/// sign sketch decoded to ±1. Dotted against a real `q_rot` this reproduces
/// `tq_attn_decode_f32`'s two-term estimator exactly, because the QJL term is
/// linear in the query: `⟨q_qjl, s⟩ = ⟨Q·q_rot, s⟩ = ⟨q_rot, Qᵀ·s⟩`.
///
/// Written straight from the documented layouts, independent of the kernel:
/// the bit unpack is `host_dequant_no_rotation`'s (already independent of
/// `tq_unpack`), and the sign unpack is spelled out here rather than reused
/// from the CUDA side. The sign convention is *not* guessed — `tq_sign_of`
/// (`cu/turboquant.cu`) returns `+1` when the bit is **set**, and `tq_store_k`
/// sets the bit when the projection is positive, so a set bit is `+1`.
fn host_qjl_aware_key_eff(
    codes: &[u8],
    scale: f32,
    signs: &[u8],
    gamma: f32,
    cb: &Codebook,
    qjl_t: &[f32],
    qjl_scale: f32,
) -> Vec<f32> {
    let mse = host_dequant_no_rotation(codes, scale, cb);
    let sign = |i: usize| -> f32 {
        if (signs[i / 8] >> (i % 8)) & 1 == 1 {
            1.0
        } else {
            -1.0
        }
    };
    let c = qjl_scale * (std::f32::consts::PI / 2.0).sqrt() / D as f32 * gamma;
    (0..D)
        .map(|i| {
            // Column-major, exactly as `tq_matvec`/`tq_store_k` read their own
            // matrices: `m[j * D + i]` is row `i`, column `j`, so this is
            // `(Qᵀ·s)[i]`.
            let qt_s: f32 = (0..D).map(|j| qjl_t[j * D + i] * sign(j)).sum();
            mse[i] + c * qt_s
        })
        .collect()
}

/// `tq_dequant_kv` must materialize TurboQuant's **whole** score estimator into
/// the dequantized key, not just its MSE-codebook half. Dropping the QJL term
/// is not a rounding difference: measured end to end on a 0.5B model, a `tq4`
/// cache scored with and without it agrees at cosine 0.289 and picks a
/// different next token.
///
/// Elementwise against a host reference that reconstructs `k_eff` from the
/// downloaded tables; the sibling test below then checks the *consequence* —
/// that dotting this vector with `q_rot` reproduces the deployed two-term
/// estimator `tq_attn_scores` computes.
#[test]
fn tq_dequant_kv_reproduces_the_full_qjl_estimator() -> Result<()> {
    let k = kernels()?;
    let tables = DeviceTables::new(k.device(), D, KvQuant::Tq4)?;
    assert!(
        tables.quant.uses_qjl(),
        "test setup: tq4 must use QJL, or this test proves nothing"
    );
    let n = 5usize;
    let k_bits = tables.quant.k_mse_bits();
    let v_bits = tables.quant.v_bits();

    let keys = unit_vectors(n, 21);
    let values = unit_vectors(n, 22);
    let k_cache = store_keys(&k, &tables, &keys, n)?;
    let v_cache = store_values(&k, &tables, &values, n)?;

    let stream = k.device().stream().clone();
    let slots: Vec<i32> = (0..n as i32).collect();
    let d_slots = stream.clone_htod(&slots)?;
    let mut d_dequant_k = stream.alloc_zeros::<f16>(n * D)?;
    let mut d_dequant_v = stream.alloc_zeros::<f16>(n * D)?;

    k.tq_dequant_kv(
        &mut d_dequant_k.as_view_mut(),
        &mut d_dequant_v.as_view_mut(),
        &k_cache.codes.as_view(),
        &k_cache.signs.as_view(),
        &k_cache.scale.as_view(),
        &k_cache.gamma.as_view(),
        &v_cache.codes.as_view(),
        &v_cache.scale.as_view(),
        &d_slots.as_view(),
        &tables.k_levels.as_view(),
        k_bits,
        &tables.v_levels.as_view(),
        v_bits,
        &tables.qjl_t.as_view(),
        tables.quant.qjl_scale(),
        1,
        D,
        n,
        n,
    )?;
    k.device().synchronize()?;

    let got_k = stream.clone_dtoh(&d_dequant_k)?;
    let host_k_scale = stream.clone_dtoh(&k_cache.scale)?;
    let host_k_codes = stream.clone_dtoh(&k_cache.codes)?;
    let host_k_signs = stream.clone_dtoh(&k_cache.signs)?;
    let host_k_gamma = stream.clone_dtoh(&k_cache.gamma)?;
    let host_qjl_t = stream.clone_dtoh(&tables.qjl_t)?;
    k.device().synchronize()?;

    let bytes_per_vec = D * k_bits as usize / 8;
    let sign_bytes_per_vec = D / 8;
    let mut worst = 0.0f32;
    for v in 0..n {
        let expect = host_qjl_aware_key_eff(
            &host_k_codes[v * bytes_per_vec..(v + 1) * bytes_per_vec],
            host_k_scale[v].to_f32(),
            &host_k_signs[v * sign_bytes_per_vec..(v + 1) * sign_bytes_per_vec],
            host_k_gamma[v].to_f32(),
            &tables.k_codebook,
            &host_qjl_t,
            tables.quant.qjl_scale(),
        );
        // The QJL term has to be a real part of the answer, not a rounding
        // correction: if it were, the MSE-only reference would pass this test
        // too and the assertion below would prove nothing.
        let mse_only = host_dequant_no_rotation(
            &host_k_codes[v * bytes_per_vec..(v + 1) * bytes_per_vec],
            host_k_scale[v].to_f32(),
            &tables.k_codebook,
        );
        let (gap, _) = max_abs_diff(&expect, &mse_only);
        assert!(
            gap > 1e-2,
            "key {v}: the QJL term moves the key by only {gap} -- this test \
             cannot distinguish the two estimators"
        );

        let got: Vec<f32> = got_k[v * D..(v + 1) * D]
            .iter()
            .map(|x| x.to_f32())
            .collect();
        let (d, at) = max_abs_diff(&got, &expect);
        worst = worst.max(d);
        assert!(
            d < 1e-3,
            "key {v} element {at}: kernel {} vs host reference {}",
            got[at],
            expect[at]
        );
    }
    eprintln!("  qjl-aware dequant vs host reference: max abs diff {worst:.2e}");
    Ok(())
}

/// The property the fold actually has to have, checked against the deployed
/// estimator rather than against a restatement of its formula: for every
/// (query, key) pair,
///
/// ```text
/// ⟨q_rot, tq_dequant_kv(key)⟩ == tq_attn_scores(q_rot, q_qjl, key)
/// ```
///
/// `tq_attn_scores` is the kernel the short/decode path runs (and computes the
/// same two-term estimator as `tq_attn_decode_f32`, `cu/turboquant.cu:270` and
/// `:388`), so this is an independent oracle: a transposed `qjl_t`, a flipped
/// sign convention or a missing `gamma` all break it, and none of them can be
/// hidden by the host reference above sharing an error with the kernel.
#[test]
fn a_dequantized_key_dotted_with_the_query_is_the_two_stage_score() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let tables = DeviceTables::new(k.device(), D, KvQuant::Tq4)?;
    let k_bits = tables.quant.k_mse_bits();
    let v_bits = tables.quant.v_bits();

    let n_keys = 64usize;
    let keys = unit_vectors(n_keys, 31);
    let values = unit_vectors(n_keys, 32);
    let k_cache = store_keys(&k, &tables, &keys, n_keys)?;
    let v_cache = store_values(&k, &tables, &values, n_keys)?;

    // The dequantized keys, through the kernel under test.
    let slots: Vec<i32> = (0..n_keys as i32).collect();
    let d_slots = stream.clone_htod(&slots)?;
    let mut d_dequant_k = stream.alloc_zeros::<f16>(n_keys * D)?;
    let mut d_dequant_v = stream.alloc_zeros::<f16>(n_keys * D)?;
    k.tq_dequant_kv(
        &mut d_dequant_k.as_view_mut(),
        &mut d_dequant_v.as_view_mut(),
        &k_cache.codes.as_view(),
        &k_cache.signs.as_view(),
        &k_cache.scale.as_view(),
        &k_cache.gamma.as_view(),
        &v_cache.codes.as_view(),
        &v_cache.scale.as_view(),
        &d_slots.as_view(),
        &tables.k_levels.as_view(),
        k_bits,
        &tables.v_levels.as_view(),
        v_bits,
        &tables.qjl_t.as_view(),
        tables.quant.qjl_scale(),
        1,
        D,
        n_keys,
        n_keys,
    )?;

    // The estimator the short/decode path really runs, over the same cache.
    let n_q = 16usize;
    let queries = unit_vectors(n_q, 33);
    let dq = stream.clone_htod(&queries)?;
    let mut q_rot = stream.alloc_zeros::<f32>(n_q * D)?;
    let mut q_qjl = stream.alloc_zeros::<f32>(n_q * D)?;
    k.tq_matvec(
        &mut q_rot.as_view_mut(),
        &dq.as_view(),
        &tables.rotation.as_view(),
        D,
        n_q,
    )?;
    k.tq_matvec(
        &mut q_qjl.as_view_mut(),
        &q_rot.as_view(),
        &tables.qjl.as_view(),
        D,
        n_q,
    )?;

    let dims = AttnDims {
        n_heads: 1,
        n_kv_heads: 1,
        d_head: D,
        n_slots: n_keys,
        n_tokens: n_q,
    };
    let positions = vec![n_keys as i32 - 1; n_q];
    let dpos = stream.clone_htod(&positions)?;
    let dseq = stream.clone_htod(&vec![0i32; n_q])?;
    let dtable = stream.clone_htod(&(0..n_keys as i32).collect::<Vec<_>>())?;
    let (vseq, vpos, vtable) = (dseq.as_view(), dpos.as_view(), dtable.as_view());
    let batch = BatchLayout {
        seq_of: &vseq,
        positions: &vpos,
        slot_table: &vtable,
        table_stride: n_keys,
    };
    let mut scores = stream.alloc_zeros::<f32>(n_q * n_keys)?;
    k.tq_attn_scores(
        &mut scores.as_view_mut(),
        &q_rot.as_view(),
        &q_qjl.as_view(),
        &k_cache.codes.as_view(),
        &k_cache.signs.as_view(),
        &k_cache.scale.as_view(),
        &k_cache.gamma.as_view(),
        batch,
        &tables.k_levels.as_view(),
        k_bits,
        dims,
        n_keys,
        1.0,
        tables.quant.qjl_scale(),
    )?;

    let dense_k: Vec<f32> = stream
        .clone_dtoh(&d_dequant_k)?
        .iter()
        .map(|x| x.to_f32())
        .collect();
    let q_rot_h = stream.clone_dtoh(&q_rot)?;
    let want = stream.clone_dtoh(&scores)?;
    k.device().synchronize()?;

    let mut got = Vec::with_capacity(n_q * n_keys);
    for t in 0..n_q {
        for j in 0..n_keys {
            got.push(dot(
                &q_rot_h[t * D..(t + 1) * D],
                &dense_k[j * D..(j + 1) * D],
            ));
        }
    }

    // ...and the MSE-only key, i.e. exactly what this kernel produced before
    // this test existed, which must *not* clear the same bar.
    let mut mse_only = Vec::with_capacity(n_q * n_keys);
    let host_k_scale = stream.clone_dtoh(&k_cache.scale)?;
    let host_k_codes = stream.clone_dtoh(&k_cache.codes)?;
    let bytes_per_vec = D * k_bits as usize / 8;
    for t in 0..n_q {
        for j in 0..n_keys {
            let kv = host_dequant_no_rotation(
                &host_k_codes[j * bytes_per_vec..(j + 1) * bytes_per_vec],
                host_k_scale[j].to_f32(),
                &tables.k_codebook,
            );
            mse_only.push(dot(&q_rot_h[t * D..(t + 1) * D], &kv));
        }
    }

    let rel = |a: &[f32]| -> f64 {
        let num: f64 = a
            .iter()
            .zip(&want)
            .map(|(x, y)| ((x - y) as f64).powi(2))
            .sum();
        let den: f64 = want.iter().map(|y| (*y as f64).powi(2)).sum();
        (num / den).sqrt()
    };
    let (folded, dropped) = (rel(&got), rel(&mse_only));
    eprintln!(
        "  vs tq_attn_scores: qjl-aware dequant rel-rmse {folded:.5}, \
         cosine {:.6}; mse-only dequant rel-rmse {dropped:.5}, cosine {:.6}",
        cosine(&got, &want),
        cosine(&mse_only, &want),
    );
    // f16 keys against `tq_attn_scores`' f32 codebook reads is the only gap
    // that should be left.
    assert!(
        folded < 0.01,
        "a dequantized key does not reproduce the two-stage score: rel-rmse {folded:.5}"
    );
    // And the term is worth carrying: the MSE-only key is an order of
    // magnitude further away, which is the gap this task closes.
    assert!(
        dropped > folded * 10.0,
        "the QJL term barely matters here ({dropped:.5} vs {folded:.5}) -- \
         this test would pass without it"
    );
    Ok(())
}

#[test]
fn bit_accounting_is_honest() {
    // Nominal rates plus the per-vector norms, which a 64-wide head amortizes
    // over half as many channels as the paper's 128-wide ones.
    let tq2 = KvQuant::Tq2.bits_per_channel(D);
    let tq4 = KvQuant::Tq4.bits_per_channel(D);
    // Tq2: keys 2+1 bits + 32/64, values 2 + 16/64 -> (3.5 + 2.25) / 2
    assert!((tq2 - 2.875).abs() < 1e-5, "tq2 = {tq2}");
    assert!((tq4 - 4.875).abs() < 1e-5, "tq4 = {tq4}");
    // At d = 128 the same settings get closer to their nominal rates.
    assert!(KvQuant::Tq2.bits_per_channel(128) < tq2);
    assert_eq!(KvQuant::F16.bits_per_channel(D), 16.0);
}
