//! Dequantize one FP8 projection out of a real checkpoint and print a sample.
//!
//! The unit tests establish that the decode matches the format and that the
//! block grid is indexed the way the layout says. Neither establishes that a
//! real export means what this code thinks it means — the same gap that let a
//! green QK-norm test sit on top of a model producing nonsense. So: read the
//! actual tensor, print values a reference implementation can be asked about
//! independently, and print the block boundary where a stride error would show.
//!
//!   cargo run --release -p infero-safetensors --example dequant_check -- \
//!     <model-dir> <tensor-name>

use anyhow::{Context, Result, bail};
use infero_safetensors::Shards;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().context("usage: dequant_check <model-dir> <tensor>")?;
    let name = args
        .next()
        .unwrap_or_else(|| "model.language_model.layers.0.linear_attn.in_proj_qkv.weight".into());

    let w = Shards::open_dir(&dir)?;
    let q = w.tensor(&name)?;
    let scales = w.tensor(&format!("{name}_scale_inv"))?;
    println!("{name}");
    println!("  quants {:?} {:?}", q.dtype, q.shape);
    println!("  scales {:?} {:?}", scales.dtype, scales.shape);

    let (rows, cols) = (q.shape[0], q.shape[1]);
    let block = 128;
    if scales.shape[0] != rows.div_ceil(block) {
        bail!(
            "this example assumes block {block}; the grid implies {}",
            rows.div_ceil(scales.shape[0])
        );
    }

    let t = std::time::Instant::now();
    let out = q.dequant_f8_to_f16(&scales, block)?;
    let secs = t.elapsed().as_secs_f64();
    println!(
        "  dequantized {} elements in {secs:.2}s ({:.0} M/s)",
        out.len(),
        out.len() as f64 / secs / 1e6
    );

    let show = |label: &str, at: usize| {
        let v: Vec<f32> = out[at..at + 4].iter().map(|x| f32::from(*x)).collect();
        println!("  {label:<28} {v:?}");
    };
    show("row 0, cols 0..4", 0);
    // A stride error in the scale grid shows at a block boundary and nowhere
    // else, so print both sides of one.
    show("row 0, cols 126..130", 126);
    show("row 127, cols 0..4", 127 * cols);
    show("row 128, cols 0..4", 128 * cols);

    let finite = out.iter().filter(|v| v.is_finite()).count();
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for v in out.iter().map(|x| f32::from(*x)).filter(|v| v.is_finite()) {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    println!("  finite {finite}/{}  range [{lo:.5}, {hi:.5}]", out.len());
    Ok(())
}
