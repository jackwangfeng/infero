//! AWQ repacking, checked against an independent quantization of the same model.
//!
//! The packing order inside an AWQ `i32` is `[0, 2, 4, 6, 1, 3, 5, 7]`, and
//! getting it wrong is invisible from the inside: every weight still decodes to
//! a plausible value, just attributed to the wrong output channel. Comparing a
//! repacked tensor against itself would pass either way.
//!
//! So compare against the same model's GGUF. A Q4_K_M file and an AWQ file are
//! two independent 4-bit quantizations of one set of f16 weights, so their
//! dequantized values should agree closely — and if the columns are permuted,
//! not at all. That is a test the format can fail.

use anyhow::Result;
use tuili_kernels::awq::{AwqTensor, unpack_row};
use tuili_safetensors::Shards;

const AWQ: &str = "/mnt/data/vllm-bench/llama8b-awq";
const GGUF: &str = "/mnt/data/tuili-models/llama-3.1-8b-instruct-q4_k_m.gguf";

/// Pearson correlation, which is what "the same weights, quantized twice"
/// should show and a column permutation should not.
fn corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let mut num = 0.0;
    let (mut da, mut db) = (0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        let (u, v) = (x - ma, y - mb);
        num += u * v;
        da += u * u;
        db += v * v;
    }
    num / (da.sqrt() * db.sqrt())
}

#[test]
fn awq_repacking_agrees_with_the_same_model_in_gguf() -> Result<()> {
    if !std::path::Path::new(AWQ).exists() || !std::path::Path::new(GGUF).exists() {
        eprintln!("skipping: need both {AWQ} and {GGUF}");
        return Ok(());
    }
    let w = Shards::open_dir(AWQ)?;
    let gguf = tuili_gguf::Gguf::open(GGUF)?;
    // Enough output channels to correlate over, without dequantizing 4096 rows
    // of a reference implementation written for clarity rather than speed.
    const ROWS: usize = 512;

    // Neither of these is permuted on the way into GGUF. `attn_q` and `attn_k`
    // are — llama.cpp reorders their rows for its interleaved rotary
    // convention — so their output channels would not correspond.
    for (awq_name, gguf_name) in [
        ("model.layers.7.mlp.gate_proj", "blk.7.ffn_gate.weight"),
        ("model.layers.3.self_attn.o_proj", "blk.3.attn_output.weight"),
    ] {
        let qw = w.tensor(&format!("{awq_name}.qweight"))?;
        let qz = w.tensor(&format!("{awq_name}.qzeros"))?;
        let sc = w.tensor(&format!("{awq_name}.scales"))?;
        let (k, n) = (qw.shape[0], qw.shape[1] * 8);

        let t = AwqTensor {
            qweight: qw.as_i32()?,
            qzeros: qz.as_i32()?,
            scales: sc.as_f16()?,
            in_features: k,
            out_features: n,
        };
        let packed = t.repack()?;
        assert_eq!(packed.len(), t.packed_bytes());

        // The pack has to round-trip its own reader first.
        for &row in &[0usize, 1, n / 2, n - 1] {
            let got = unpack_row(&packed, k, row);
            for kk in [0usize, 1, 63, 64, 127, 128, k - 1] {
                let want = t.weight(kk, row);
                // In units of the quantization step: the pack folds the zero
                // into the scale and stores the product as f16, which moves
                // every weight in a block by up to an ulp of `scale * zero`.
                let step = t.scale(kk, row);
                assert!(
                    (got[kk] - want).abs() <= 0.02 * step,
                    "{awq_name} row {row} k {kk}: packed {} vs source {want}, step {step}"
                    , got[kk]
                );
            }
        }

        // Then against the other quantization of the same weights — along the
        // output axis, at a fixed input channel.
        //
        // AWQ does not quantize the original weights. It multiplies each input
        // channel by a factor chosen to protect the salient ones and divides
        // that factor back out of whatever feeds the layer, so its stored
        // weights are the originals times a per-input-channel scale. Comparing
        // along `k` would be comparing through that scale, which is what drags
        // the correlation to 0.89. At a fixed `k` the factor is a constant, and
        // a constant does not move a correlation at all.
        let gg = dequantize_gguf_rows(&gguf, gguf.tensor(gguf_name)?, k, ROWS)?;
        let mine: Vec<Vec<f32>> = (0..ROWS).map(|r| unpack_row(&packed, k, r)).collect();

        for kk in [0usize, 1, 129, k / 2, k - 1] {
            let a: Vec<f32> = (0..ROWS).map(|r| mine[r][kk]).collect();
            let b: Vec<f32> = (0..ROWS).map(|r| gg[r * k + kk]).collect();
            let c = corr(&a, &b);
            eprintln!("  {awq_name} k={kk}: correlation {c:.4}");
            // Two independent 4-bit quantizations of one set of weights, with
            // different group sizes, agree to about 0.8. A misattributed
            // channel reads 0.05, so the threshold has a lot of room.
            assert!(
                c > 0.6,
                "{awq_name} k={kk}: correlation {c:.4} — the repacking is wrong"
            );
        }
    }
    Ok(())
}

/// Dequantize the first `rows` output channels of a Q4_K GGUF tensor.
///
/// Only the layout matters here, so this is the plain definition rather than
/// anything fast: a Q4_K super-block is 256 weights as eight 32-weight groups,
/// each with a 6-bit scale and 6-bit minimum packed into twelve bytes and
/// scaled by two f16 factors.
fn dequantize_gguf_rows(
    gguf: &tuili_gguf::Gguf,
    info: &tuili_gguf::TensorInfo,
    k: usize,
    rows: usize,
) -> Result<Vec<f32>> {
    anyhow::ensure!(
        info.ty == tuili_gguf::GgmlType::Q4K,
        "{} is {}, this reference only does Q4_K",
        info.name,
        info.ty
    );
    let data = gguf.data(info);
    let sb = k / 256;
    let n_rows = rows.min(info.dims[1] as usize);
    let mut out = vec![0.0f32; n_rows * k];
    for row in 0..n_rows {
        for b in 0..sb {
            let blk = &data[(row * sb + b) * 144..(row * sb + b) * 144 + 144];
            let d = f32::from(half::f16::from_le_bytes([blk[0], blk[1]]));
            let dmin = f32::from(half::f16::from_le_bytes([blk[2], blk[3]]));
            let sc = &blk[4..16];
            let qs = &blk[16..144];
            for g in 0..8 {
                // ggml packs the 6-bit scale/min pairs six to a byte-triple.
                let (s, m) = if g < 4 {
                    (sc[g] & 63, sc[g + 4] & 63)
                } else {
                    (
                        (sc[g + 4] & 0xF) | ((sc[g - 4] >> 6) << 4),
                        (sc[g + 4] >> 4) | ((sc[g] >> 6) << 4),
                    )
                };
                let (ds, dm) = (d * s as f32, dmin * m as f32);
                for i in 0..32 {
                    let byte = qs[(g / 2) * 32 + i];
                    let q = if g % 2 == 0 { byte & 0xF } else { byte >> 4 };
                    out[row * k + b * 256 + g * 32 + i] = ds * q as f32 - dm;
                }
            }
        }
    }
    Ok(out)
}
