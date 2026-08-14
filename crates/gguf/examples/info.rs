//! Dump a GGUF file's metadata and tensor table.
//!
//!     cargo run -p tuili-gguf --example info -- models/qwen2.5-0.5b-instruct-q8_0.gguf
//!     cargo run -p tuili-gguf --example info -- <file> --tensors

use anyhow::{Context, Result};
use tuili_gguf::Gguf;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .context("usage: info <model.gguf> [--tensors]")?;
    let show_tensors = args.any(|a| a == "--tensors");

    let f = Gguf::open(&path)?;

    println!("file       : {}", f.path().display());
    println!("version    : {}", f.version());
    println!("alignment  : {}", f.alignment());
    println!("tensors    : {}", f.tensors().len());
    println!(
        "data       : {:.2} GiB",
        f.data_len() as f64 / (1u64 << 30) as f64
    );
    if let Some(t) = f.dominant_type() {
        println!("quant      : {t}");
    }

    println!("\nmetadata ({}):", f.metadata().len());
    for (k, v) in f.metadata() {
        println!("  {k:<44} {v}");
    }

    if show_tensors {
        println!("\ntensors:");
        for t in f.tensors().values() {
            println!(
                "  {:<32} {:<7} {:<20} {:>12} B  @{}",
                t.name,
                t.ty.name(),
                format!("{:?}", t.dims),
                t.n_bytes,
                t.offset
            );
        }
    } else {
        // Layer 0 is enough to see the architecture's tensor naming.
        println!("\ntensors (layer 0 and non-layer):");
        for t in f.tensors().values() {
            let is_layer = t.name.starts_with("blk.");
            if is_layer && !t.name.starts_with("blk.0.") {
                continue;
            }
            println!(
                "  {:<32} {:<7} {:<20} {:>12} B",
                t.name,
                t.ty.name(),
                format!("{:?}", t.dims),
                t.n_bytes
            );
        }
        println!("  (pass --tensors for all {})", f.tensors().len());
    }

    Ok(())
}
