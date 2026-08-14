//! What would a decode step cost if it were nothing but the weight matrices?
//!
//! A step reads every weight once, and on this class of card that read is the
//! whole job — everything else is arithmetic on a few kilobytes. So the sum of
//! the mat-vecs is the floor, and the distance between it and a real step is
//! the overhead budget. `gemm_bench` measures one matrix at a time and cannot
//! answer this: each of its timings carries its own launch, and a real step
//! issues two hundred and twenty-five of them back to back inside one CUDA
//! graph.
//!
//! This replays exactly the mat-vecs a decode step performs — same tensors,
//! same order, same graph — and nothing else.
//!
//! Takes either a GGUF file or an AWQ checkpoint directory, so the two
//! formats can be compared on the one number that decides a decode step.
//!
//!     cargo run --release -p tuili-model --example decode_floor -- model.gguf
//!     cargo run --release -p tuili-model --example decode_floor -- llama8b-awq/

use std::time::Instant;

use anyhow::{Context, Result};
use tuili_cuda::Device;
use tuili_gguf::Gguf;
use tuili_kernels::{Kernels, WeightType};

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();

    let path = std::env::args()
        .nth(1)
        .context("usage: decode_floor <model.gguf>")?;
    let dev = Device::new(0)?;
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream().clone();

    struct M {
        w: cudarc::driver::CudaSlice<u8>,
        ty: WeightType,
        k: usize,
        n: usize,
    }
    let mut mats: Vec<M> = Vec::new();

    if std::path::Path::new(&path).is_dir() {
        // AWQ: repacked into Q4_G128 the way the loader would, so what this
        // measures is the format as the engine would actually read it.
        let w = tuili_safetensors::Shards::open_dir(&path)?;
        let n_layers = (0..)
            .take_while(|i| w.contains(&format!("model.layers.{i}.self_attn.q_proj.qweight")))
            .count();
        anyhow::ensure!(n_layers > 0, "no layers in {path}");
        let t0 = Instant::now();
        for l in 0..n_layers {
            for name in [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ] {
                let p = format!("model.layers.{l}.{name}");
                let qw = w.tensor(&format!("{p}.qweight"))?;
                let (k, n) = (qw.shape[0], qw.shape[1] * 8);
                let packed = tuili_kernels::awq::AwqTensor {
                    qweight: qw.as_i32()?,
                    qzeros: w.tensor(&format!("{p}.qzeros"))?.as_i32()?,
                    scales: w.tensor(&format!("{p}.scales"))?.as_f16()?,
                    in_features: k,
                    out_features: n,
                }
                .repack()?;
                mats.push(M {
                    w: stream.clone_htod(&packed)?,
                    ty: WeightType::Q4G128,
                    k,
                    n,
                });
            }
        }
        // AWQ leaves the vocab projection in f16 — 1.05 GB, a fifth of the
        // step, at 141 GB/s through the float mat-vec. Quantizing it at load
        // halves the bytes and moves it onto the integer path. Set
        // `TUILI_F16_HEAD=1` to measure it as the checkpoint ships it.
        let head = w.tensor("lm_head.weight")?;
        let raw = std::env::var_os("TUILI_F16_HEAD").is_some();
        let (hw, hty) = if raw {
            (head.data.to_vec(), WeightType::F16)
        } else {
            (
                tuili_kernels::awq::quantize_f16_to_q8_0(head.as_f16()?, head.shape[1])?,
                WeightType::Q8_0,
            )
        };
        mats.push(M {
            w: stream.clone_htod(&hw)?,
            ty: hty,
            k: head.shape[1],
            n: head.shape[0],
        });
        println!("repacked {n_layers} layers in {:.1}s", t0.elapsed().as_secs_f64());
    } else {
        let gguf = Gguf::open(&path)?;
        let per_layer = [
            "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down",
        ];
        let n_layers = (0..)
            .take_while(|i| gguf.tensor(&format!("blk.{i}.attn_q.weight")).is_ok())
            .count();
        anyhow::ensure!(n_layers > 0, "no layers found");
        let mut load = |name: &str| -> Result<()> {
            let t = gguf.tensor(name)?;
            let ty = WeightType::from_ggml(t.ty)
                .with_context(|| format!("{name}: unsupported type {:?}", t.ty))?;
            // GGUF stores [k, n] with k contiguous.
            mats.push(M {
                w: stream.clone_htod(gguf.data(t))?,
                ty,
                k: t.dims[0] as usize,
                n: t.dims[1] as usize,
            });
            Ok(())
        };
        for l in 0..n_layers {
            for t in per_layer {
                load(&format!("blk.{l}.{t}.weight"))?;
            }
        }
        load(if gguf.tensor("output.weight").is_ok() {
            "output.weight"
        } else {
            "token_embd.weight"
        })?;
    }

    let n_layers = (mats.len() - 1) / 7;
    let bytes: usize = mats.iter().map(|m| m.w.len()).sum();

    let max_k = mats.iter().map(|m| m.k).max().unwrap();
    let max_n = mats.iter().map(|m| m.n).max().unwrap();
    let mut x = stream.alloc_zeros::<u8>(Kernels::q8_1_bytes(max_k))?;
    let mut out = stream.alloc_zeros::<f32>(max_n)?;
    // Something non-degenerate in the activation, so no branch short-circuits.
    let seed: Vec<f32> = (0..max_k).map(|i| (i % 19) as f32 / 19.0 - 0.5).collect();
    let dseed = stream.clone_htod(&seed)?;
    kern.quantize_q8_1(&mut x.as_view_mut(), &dseed.as_view(), max_k)?;

    let xf = &dseed;
    let run = |kern: &Kernels,
               x: &cudarc::driver::CudaSlice<u8>,
               out: &mut cudarc::driver::CudaSlice<f32>|
     -> Result<()> {
        for m in &mats {
            // The vocab projection an AWQ checkpoint ships is f16, which the
            // integer path does not take; the float mat-vec reads the same
            // bytes, which is what this is counting.
            if Kernels::has_mmvq(m.ty) {
                kern.mmvq(
                    &mut out.slice_mut(..m.n),
                    &m.w.as_view(),
                    m.ty,
                    &x.slice(..Kernels::q8_1_bytes(m.k)),
                    m.k,
                    m.n,
                )?;
            } else {
                kern.gemv(
                    &mut out.slice_mut(..m.n),
                    &m.w.as_view(),
                    m.ty,
                    &xf.slice(..m.k),
                    m.k,
                    m.n,
                    1,
                )?;
            }
        }
        Ok(())
    };

    // Warm first: capture cannot load a module, so every kernel has to have
    // been launched once before the recording starts.
    for _ in 0..3 {
        run(&kern, &x, &mut out)?;
    }
    dev.synchronize()?;

    let graphed = std::env::var_os("TUILI_NO_GRAPH").is_none();
    let graph = if graphed {
        stream.begin_capture(
            cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
        )?;
        let res = run(&kern, &x, &mut out);
        let g = stream.end_capture(
            cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        res?;
        Some(g?.context("capture produced no graph")?)
    } else {
        None
    };

    // Long enough to reach the same thermal state a real generation does: a
    // twenty-step run finishes before the card's clocks drop, and then reports
    // a floor no server will ever see.
    // The vocab projection is one matrix out of 225 and a fifth of the bytes,
    // and the two formats disagree about it more than about anything else — a
    // Q4_K_M file carries it as 0.43 GB of Q6_K, an AWQ checkpoint as 1.05 GB
    // of f16. Time it separately or the comparison says nothing about the
    // layers.
    let head_bytes = mats.last().map_or(0, |m| m.w.len());
    let layer_bytes = bytes - head_bytes;

    let reps: usize = std::env::var("TUILI_FLOOR_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let t = Instant::now();
    for _ in 0..reps {
        match &graph {
            Some(g) => g.launch()?,
            None => run(&kern, &x, &mut out)?,
        }
    }
    dev.synchronize()?;
    let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let _ = reps;

    // Same again for the layers alone, so the head's cost is separable.
    let layers_only = mats.len() - 1;
    for _ in 0..3 {
        for m in &mats[..layers_only] {
            kern.mmvq(
                &mut out.slice_mut(..m.n),
                &m.w.as_view(),
                m.ty,
                &x.slice(..Kernels::q8_1_bytes(m.k)),
                m.k,
                m.n,
            )?;
        }
    }
    dev.synchronize()?;
    let t = Instant::now();
    for _ in 0..reps {
        for m in &mats[..layers_only] {
            kern.mmvq(
                &mut out.slice_mut(..m.n),
                &m.w.as_view(),
                m.ty,
                &x.slice(..Kernels::q8_1_bytes(m.k)),
                m.k,
                m.n,
            )?;
        }
    }
    dev.synchronize()?;
    let lms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;

    println!(
        "  layers only  {lms:.2} ms   {:.2} GB   {:.0} GB/s\n           head         {:.2} ms   {:.2} GB   {:.0} GB/s   ({})",
        layer_bytes as f64 / 1e9,
        layer_bytes as f64 / 1e9 / (lms / 1e3),
        ms - lms,
        head_bytes as f64 / 1e9,
        head_bytes as f64 / 1e9 / ((ms - lms) / 1e3),
        mats.last().map_or(WeightType::F16, |m| m.ty),
    );

    let gb = bytes as f64 / 1e9;
    println!(
        "\n{n_layers} layers, {} mat-vecs, {gb:.2} GB of weights\n\
         {ms:.2} ms per step   {:.0} GB/s   {:.1} tok/s if nothing else cost anything  ({})",
        mats.len(),
        gb / (ms / 1e3),
        1e3 / ms,
        if graphed { "cuda graph" } else { "plain launches" },
    );
    Ok(())
}
