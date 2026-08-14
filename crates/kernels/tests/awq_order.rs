//! Which nibble of an AWQ `i32` belongs to which output channel — measured,
//! not assumed.
//!
//! AutoAWQ packs output offset `ORDER[i]` at bit `4 * i`, and getting that
//! wrong is invisible from inside the file: every weight still decodes to a
//! plausible value, only attributed to the wrong channel. So recover the
//! permutation from the data, by correlating each nibble position against each
//! output-channel offset of the same model quantized independently as GGUF.
//! The answer comes out as a clean 8x8 assignment — 0.76 to 0.84 on the
//! diagonal against 0.05 elsewhere.
//!
//! Two things this had to learn the hard way. Compare at a *fixed input
//! channel*, across outputs: AWQ multiplies each input channel by a factor
//! chosen to protect the salient ones, and correlating along `k` measures that
//! envelope rather than the weights — it reads 0.89 whether the order is right
//! or not. And do not compare against `attn_q` or `attn_k`: llama.cpp permutes
//! their rows during GGUF conversion, to turn Hugging Face's rotate-half rotary
//! embedding into its own interleaved one, so their output channels do not
//! correspond at all.

use anyhow::Result;
use tuili_safetensors::Shards;

const AWQ: &str = "/mnt/data/vllm-bench/llama8b-awq";
const GGUF: &str = "/mnt/data/tuili-models/llama-3.1-8b-instruct-q4_k_m.gguf";

fn corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut num, mut da, mut db) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        let (u, v) = (x - ma, y - mb);
        num += u * v;
        da += u * u;
        db += v * v;
    }
    num / (da.sqrt() * db.sqrt()).max(1e-20)
}

#[test]
fn the_nibble_permutation_is_the_one_the_packer_assumes() -> Result<()> {
    if !std::path::Path::new(AWQ).exists() || !std::path::Path::new(GGUF).exists() {
        eprintln!("skipping");
        return Ok(());
    }
    let w = Shards::open_dir(AWQ)?;
    let gguf = tuili_gguf::Gguf::open(GGUF)?;
    let name = "model.layers.7.mlp.gate_proj";
    let qw = w.tensor(&format!("{name}.qweight"))?;
    let sc = w.tensor(&format!("{name}.scales"))?;
    let (k, n) = (qw.shape[0], qw.shape[1] * 8);
    let (qweight, scales) = (qw.as_i32()?, sc.as_f16()?);

    const GROUPS: usize = 256; // output channels 0 .. 8*GROUPS
    // The repo's own dequantizer rather than a hand-written one: a mistake in
    // Q4_K's six-bit scale unpacking would look exactly like a mistake in AWQ's
    // nibble order, and this one is covered by its own tests.
    let gg = dequant_on_device(&gguf, gguf.tensor("blk.7.ffn_gate.weight")?, k, 8 * GROUPS)?;
    let kk = 137usize; // any input channel; the AWQ per-channel factor is constant here

    println!("\n        {}", (0..8).map(|m| format!("  out+{m}")).collect::<String>());
    let mut best = [0usize; 8];
    for (i, slot) in best.iter_mut().enumerate() {
        // Nibble `i` of each group, dequantized only up to a constant: the
        // zero point shifts the whole column and a shift does not move a
        // correlation.
        let a: Vec<f32> = (0..GROUPS)
            .map(|g| {
                let word = qweight[kk * (n / 8) + g] as u32;
                let code = ((word >> (4 * i)) & 0xF) as f32;
                code * f32::from(scales[kk / 128 * n + 8 * g])
            })
            .collect();
        let mut row = format!("nib {i}:");
        let (mut top, mut topv) = (0usize, -2.0f32);
        for m in 0..8 {
            let b: Vec<f32> = (0..GROUPS).map(|g| gg[(8 * g + m) * k + kk]).collect();
            let c = corr(&a, &b);
            row.push_str(&format!(" {c:+.3} "));
            if c > topv {
                topv = c;
                top = m;
            }
        }
        *slot = top;
        println!("{row}   -> out+{top} ({topv:+.3})");
    }
    println!("\nnibble i holds output offset: {best:?}\n");
    assert_eq!(
        best,
        [0, 2, 4, 6, 1, 3, 5, 7],
        "the packer's ORDER does not match what the checkpoint actually does"
    );
    Ok(())
}

fn dequant_on_device(
    gguf: &tuili_gguf::Gguf,
    info: &tuili_gguf::TensorInfo,
    k: usize,
    rows: usize,
) -> Result<Vec<f32>> {
    let dev = tuili_cuda::Device::new(0)?;
    let kern = tuili_kernels::Kernels::new(dev.clone());
    let stream = dev.stream().clone();
    let ty = tuili_kernels::WeightType::from_ggml(info.ty)?;
    let bytes = rows * k / ty.block_size() * ty.type_size();
    let w = stream.clone_htod(&gguf.data(info)[..bytes])?;
    let mut out = stream.alloc_zeros::<half::f16>(rows * k)?;
    kern.dequant_to_f16(&mut out.as_view_mut(), &w.as_view(), ty, rows * k)?;
    let host = stream.clone_dtoh(&out)?;
    dev.synchronize()?;
    Ok(host.iter().map(|v| f32::from(*v)).collect())
}
