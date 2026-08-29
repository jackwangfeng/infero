//! Against a real AWQ checkpoint, when one is present.
//!
//! The header is the whole format, so what is worth checking is that the shapes
//! it claims agree with what AWQ's packing implies: `qweight` holds eight
//! output columns per `i32`, `scales` one `f16` per group per column, and every
//! payload lands inside its shard.

use anyhow::Result;
use infero_safetensors::{Dtype, Shards};

const DIR: &str = "/mnt/data/vllm-bench/llama8b-awq";

#[test]
fn an_awq_checkpoint_reads_back_with_consistent_shapes() -> Result<()> {
    if !std::path::Path::new(DIR).exists() {
        eprintln!("skipping: no checkpoint at {DIR}");
        return Ok(());
    }
    let w = Shards::open_dir(DIR)?;
    assert!(w.len() > 700, "expected a full checkpoint, got {}", w.len());

    let cfg = w.json("config.json")?;
    let hidden = cfg["hidden_size"].as_u64().unwrap() as usize;
    let group = cfg["quantization_config"]["group_size"].as_u64().unwrap() as usize;
    assert_eq!(cfg["quantization_config"]["bits"].as_u64(), Some(4));

    let q = w.tensor("model.layers.0.self_attn.q_proj.qweight")?;
    let s = w.tensor("model.layers.0.self_attn.q_proj.scales")?;
    let z = w.tensor("model.layers.0.self_attn.q_proj.qzeros")?;

    // AWQ transposes: [in_features, out_features / 8] of packed nibbles.
    assert_eq!(q.dtype, Dtype::I32);
    assert_eq!(q.shape, vec![hidden, hidden / 8]);
    assert_eq!(s.dtype, Dtype::F16);
    assert_eq!(s.shape, vec![hidden / group, hidden]);
    assert_eq!(z.dtype, Dtype::I32);
    assert_eq!(z.shape, vec![hidden / group, hidden / 8]);

    assert_eq!(q.as_i32()?.len(), hidden * hidden / 8);
    assert_eq!(s.as_f16()?.len(), hidden / group * hidden);

    // Scales have to be finite and non-zero for the dequantization to mean
    // anything, and a mis-parsed offset shows up here first.
    let scales = s.as_f16()?;
    assert!(scales.iter().all(|v| v.is_finite() && f32::from(*v) != 0.0));

    let head = w.tensor("lm_head.weight")?;
    assert_eq!(head.dtype, Dtype::F16, "AWQ leaves the vocab projection in f16");
    Ok(())
}
