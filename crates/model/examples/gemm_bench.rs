//! One weight matrix, every kernel that can multiply by it.
//!
//! The full-model benchmark answers "is it faster", which is the wrong question
//! while iterating on a kernel — a step touches 170 matrices and hides where the
//! time went. This isolates a single real GGUF tensor so a change to the tile
//! staging shows up in seconds, and reports effective bandwidth against the
//! quantized weight volume, which is the number that says how much is left.
//!
//!     cargo run --release -p tuili-model --example gemm_bench -- model.gguf [tensor]

use std::time::Instant;

use anyhow::{Context, Result};
use tuili_cuda::Device;
use tuili_gguf::Gguf;
use tuili_kernels::{Kernels, WeightType};

const TOKENS: &[usize] = &[1, 2, 4, 8, 16, 32, 64];

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();

    let path = std::env::args()
        .nth(1)
        .context("usage: gemm_bench <model.gguf> [tensor]")?;
    let want = std::env::args().nth(2);

    let gguf = Gguf::open(&path)?;
    let dev = Device::new(0)?;
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream().clone();

    // Default to the two shapes that dominate a decode step: the widest FFN
    // matrix and the vocab projection.
    let names: Vec<String> = match want {
        Some(n) => vec![n],
        None => [
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_down.weight",
            "output.weight",
        ]
        .iter()
        .filter(|n| gguf.tensor(n).is_ok())
        .map(|n| n.to_string())
        .collect(),
    };

    // The ceiling, measured once.
    {
        let big = stream.alloc_zeros::<u8>(256 << 20)?;
        let mut sink = stream.alloc_zeros::<f32>(1)?;
        for _ in 0..3 {
            kern.stream_read_probe(&mut sink.as_view_mut(), &big.as_view())?;
        }
        dev.synchronize()?;
        let s0 = Instant::now();
        for _ in 0..10 {
            kern.stream_read_probe(&mut sink.as_view_mut(), &big.as_view())?;
        }
        dev.synchronize()?;
        let us = s0.elapsed().as_secs_f64() * 1e6 / 10.0;
        println!(
            "{} — {} SMs, streaming read ceiling {:.0} GB/s",
            dev.name(),
            dev.sm_count(),
            (256 << 20) as f64 / (us * 1e-6) / 1e9
        );
    }

    for (m, kn) in [
        ("tuili_mmq", "mmq_q4_K"),
        ("tuili_mmq", "mmqw8_q4_K"),
        ("tuili_mmq", "mmqw8_2_q4_K"),
        ("tuili_mmvq", "mmvq_q4_K"),
        ("tuili_ops", "attn_scores_gqa_f32"),
    ] {
        if let Ok((mt, bv)) = kern.kernel_limits(m, kn) {
            println!("  {kn:<22} 寄存器允许的最大 block = {mt:>4} 线程  (sm_{bv})");
        }
    }

    for name in &names {
        let info = gguf.tensor(name)?;
        let ty = WeightType::from_ggml(info.ty)?;
        let (k, n) = (info.dims[0] as usize, info.dims[1] as usize);
        let bytes = gguf.data(info);
        let w = stream.clone_htod(bytes)?;

        println!(
            "\n{name}  {ty}  k={k} n={n}  {:.1} MiB",
            bytes.len() as f64 / (1 << 20) as f64
        );
        println!(
            "{:>7} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "tokens", "mmq", "mmvq-T", "读不写", "stage", "deq+gemm", ""
        );

        let max_t = *TOKENS.last().unwrap();
        let x: Vec<f32> = (0..max_t * k)
            .map(|i| ((i * 2654435761usize) % 2003) as f32 / 1000.0 - 1.0)
            .collect();
        let dx = stream.clone_htod(&x)?;
        let mut out = stream.alloc_zeros::<f32>(max_t * n)?;
        let mut q8_1 = stream.alloc_zeros::<u8>(max_t * Kernels::q8_1_bytes(k))?;
        let mut x16 = stream.alloc_zeros::<half::f16>(max_t * k)?;
        let mut w16 = stream.alloc_zeros::<half::f16>(k * n)?;

        for &t in TOKENS {
            let time = |reps: usize, f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
                for _ in 0..3 {
                    f()?;
                }
                dev.synchronize()?;
                let s = Instant::now();
                for _ in 0..reps {
                    f()?;
                }
                dev.synchronize()?;
                Ok(s.elapsed().as_secs_f64() * 1e6 / reps as f64)
            };

            let mmq_us = if Kernels::has_mmq(ty)
                && k.is_multiple_of(32)
                && (ty != WeightType::Q4K || k.is_multiple_of(256))
            {
                let total = t * Kernels::q8_1_bytes(k);
                time(20, &mut || {
                    kern.quantize_q8_1(&mut q8_1.slice_mut(..total), &dx.slice(..t * k), t * k)?;
                    kern.mmq(
                        &mut out.slice_mut(..t * n),
                        &w.as_view(),
                        ty,
                        &q8_1.slice(..total),
                        k,
                        n,
                        t,
                    )
                })?
            } else {
                f64::NAN
            };

            // Attribution: the same tile loop with the tensor cores removed,
            // and with the staging removed. Only defined for Q4_K.
            let mut variant = |name: &str| -> Result<f64> {
                if ty != WeightType::Q4K {
                    return Ok(f64::NAN);
                }
                let total = t * Kernels::q8_1_bytes(k);
                time(20, &mut || {
                    kern.mmq_variant(
                        name,
                        &mut out.slice_mut(..t * n),
                        &w.as_view(),
                        ty,
                        &q8_1.slice(..total),
                        k,
                        n,
                        t,
                    )
                })
            };
            let stage_us = variant("mmq_stage_only")?;
            // Same tile loop, A operands not gathered from shared memory.
            // The multi-token mat-vec: one weight pass, T tokens.
            let mvb_us = if Kernels::has_mmvq(ty) && k.is_multiple_of(32) {
                let total = t * Kernels::q8_1_bytes(k);
                time(20, &mut || {
                    kern.mmvq_batch(
                        &mut out.slice_mut(..t * n),
                        &w.as_view(),
                        ty,
                        &q8_1.slice(..total),
                        k,
                        n,
                        t,
                    )
                })?
            } else {
                f64::NAN
            };
            let ronly_us = if ty == WeightType::Q4K && k.is_multiple_of(256) {
                time(20, &mut || {
                    kern.mmq_variant("mmq_readonly", &mut out.slice_mut(..t*n), &w.as_view(), ty, &q8_1.slice(..Kernels::q8_1_bytes(k)), k, n, t)
                })?
            } else { f64::NAN };

            // The integer mat-vec, repeated per token. This is what the model
            // uses at one token, so it is the number `mmq` has to beat before
            // the two paths can be unified.
            let mmvq_us = if Kernels::has_mmvq(ty) && k.is_multiple_of(32) {
                let qb = Kernels::q8_1_bytes(k);
                time(20, &mut || {
                    kern.quantize_q8_1(&mut q8_1.slice_mut(..t * qb), &dx.slice(..t * k), t * k)?;
                    for i in 0..t {
                        kern.mmvq(
                            &mut out.slice_mut(i * n..(i + 1) * n),
                            &w.as_view(),
                            ty,
                            &q8_1.slice(i * qb..(i + 1) * qb),
                            k,
                            n,
                        )?;
                    }
                    Ok(())
                })?
            } else {
                f64::NAN
            };

            let deq_us = time(20, &mut || {
                kern.to_f16(&mut x16.slice_mut(..t * k), &dx.slice(..t * k), t * k)?;
                kern.dequant_to_f16(&mut w16.slice_mut(..k * n), &w.as_view(), ty, k * n)?;
                kern.gemm_f16(
                    &mut out.slice_mut(..t * n),
                    &x16.slice(..t * k),
                    &w16.slice(..k * n),
                    t,
                    k,
                    n,
                )
            })?;

            let gbs = bytes.len() as f64 / (mvb_us.min(mmq_us) * 1e-6) / 1e9;
            let grid = kern.mmq_grid(n, t);
            println!(
                "{t:>7} {mmq_us:>9.1} {mvb_us:>9.1} {ronly_us:>9.1} {mmvq_us:>9.1} \
                 {stage_us:>9.1} {deq_us:>9.1} {gbs:>8.0} GB/s  \
                 grid {:>5}/{:<5} ({:>3.0}%)",
                grid.0,
                grid.1,
                100.0 * (grid.0 as f64 / grid.1 as f64).min(1.0)
            );
        }
    }
    Ok(())
}
