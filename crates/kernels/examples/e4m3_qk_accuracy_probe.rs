//! What does e4m3 quantization cost QK^T in accuracy, at realistic Q/K
//! magnitudes? The throughput probe (`mma_fp8_vs_f16_probe`) answered whether
//! e4m3 tensor cores are worth it on this card; this is the other half of
//! evaluating an e4m3 attention rewrite before building it — unlike a GEMM
//! output, an attention score feeds a softmax, which exponentially amplifies
//! relative error near the max, so the FP8-GEMM-activation error budget this
//! codebase already accepts elsewhere is not an automatic precedent.
//!
//! Q/K here are per-head RMS-normalized before the rotary (`qk_norm_f32`), so
//! realistic magnitudes are close to unit scale, not raw pre-norm activations
//! — this generates standard-normal vectors to match.
//!
//!     cargo run --release -p infero-kernels --example e4m3_qk_accuracy_probe

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::Kernels;

const D_HEAD: usize = 256;
const GROUP: usize = 128;
const TRIALS: usize = 100_000;

fn pseudo_gaussian(n: usize, seed: u64) -> Vec<f32> {
    // Box-Muller over a simple xorshift stream — good enough for an
    // approximately-standard-normal accuracy probe, not a statistical test.
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f64 / (1u64 << 24) as f64).clamp(1e-9, 1.0 - 1e-9)
    };
    (0..n)
        .map(|_| {
            let u1 = next();
            let u2 = next();
            ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
        })
        .collect()
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let dev = Device::new(0)?;
    let k = Kernels::new(dev.clone());
    let stream = k.device().stream().clone();

    let q = pseudo_gaussian(TRIALS * D_HEAD, 0x9e3779b9);
    let kk = pseudo_gaussian(TRIALS * D_HEAD, 0xc2b2ae3d);

    let d_q = stream.clone_htod(&q)?;
    let d_k = stream.clone_htod(&kk)?;
    let mut d_exact = stream.alloc_zeros::<f32>(TRIALS)?;
    let mut d_quant = stream.alloc_zeros::<f32>(TRIALS)?;

    k.e4m3_qk_dot_accuracy_probe(
        &mut d_exact.as_view_mut(),
        &mut d_quant.as_view_mut(),
        &d_q.as_view(),
        &d_k.as_view(),
        TRIALS,
        D_HEAD,
        GROUP,
    )?;
    k.device().synchronize()?;

    let exact = stream.clone_dtoh(&d_exact)?;
    let quant = stream.clone_dtoh(&d_quant)?;

    // The attention score is `dot * scale`, `scale = 1/sqrt(d_head)` — what
    // actually reaches softmax. Report absolute error in *scaled* score
    // units, next to the scaled scores' own spread, since that comparison is
    // what tells us whether softmax would notice.
    let attn_scale = 1.0f32 / (D_HEAD as f32).sqrt();

    let mut abs_err_sum = 0.0f64;
    let mut abs_err_max = 0.0f32;
    let mut scaled_scores: Vec<f32> = Vec::with_capacity(TRIALS);
    let mut rel_err_sum = 0.0f64;
    let mut rel_err_count = 0u64;
    for i in 0..TRIALS {
        let e = exact[i] * attn_scale;
        let qd = quant[i] * attn_scale;
        let abs_err = (e - qd).abs();
        abs_err_sum += abs_err as f64;
        abs_err_max = abs_err_max.max(abs_err);
        scaled_scores.push(e);
        if e.abs() > 1e-3 {
            rel_err_sum += (abs_err / e.abs()) as f64;
            rel_err_count += 1;
        }
    }
    let score_mean = scaled_scores.iter().map(|v| *v as f64).sum::<f64>() / TRIALS as f64;
    let score_var = scaled_scores
        .iter()
        .map(|v| (*v as f64 - score_mean).powi(2))
        .sum::<f64>()
        / TRIALS as f64;
    let score_std = score_var.sqrt();

    println!("{TRIALS} trials, d_head={D_HEAD}, group={GROUP}, attn_scale={attn_scale:.5}");
    println!("scaled attention score: mean {score_mean:.5}, std {score_std:.5}");
    println!(
        "scaled score abs error : mean {:.6}, max {:.6}",
        abs_err_sum / TRIALS as f64,
        abs_err_max
    );
    println!(
        "mean |rel error| where |score| > 1e-3: {:.4}%  ({rel_err_count} of {TRIALS} trials)",
        (rel_err_sum / rel_err_count.max(1) as f64) * 100.0
    );
    println!(
        "\nabs error / score std: {:.4} -- how many score-standard-deviations the quantization noise adds",
        (abs_err_sum / TRIALS as f64) / score_std
    );
    Ok(())
}
