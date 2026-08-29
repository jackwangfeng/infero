//! Parses a real GGUF file if one has been downloaded.
//!
//! Skipped (not failed) when `models/` is empty, so `cargo test` works on a
//! fresh clone. Point `INFERO_TEST_GGUF` at any file to use a different one.

use std::path::PathBuf;

use infero_gguf::{GgmlType, Gguf};

fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("INFERO_TEST_GGUF") {
        return Some(PathBuf::from(p));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let p = root.join("qwen2.5-0.5b-instruct-q8_0.gguf");
    p.exists().then_some(p)
}

macro_rules! model_or_skip {
    () => {
        match model_path() {
            Some(p) => Gguf::open(p).expect("opening model"),
            None => {
                eprintln!("skipping: no model in models/, see README");
                return;
            }
        }
    };
}

#[test]
fn parses_qwen2_header() {
    let f = model_or_skip!();

    assert_eq!(f.version(), 3);
    assert_eq!(f.arch().unwrap(), "qwen2");
    assert_eq!(f.u32(&f.akey("block_count").unwrap()).unwrap(), 24);
    assert_eq!(f.u32(&f.akey("embedding_length").unwrap()).unwrap(), 896);
    assert_eq!(f.u32(&f.akey("attention.head_count").unwrap()).unwrap(), 14);
    assert_eq!(
        f.u32(&f.akey("attention.head_count_kv").unwrap()).unwrap(),
        2
    );
}

#[test]
fn every_layer_has_its_tensors() {
    let f = model_or_skip!();
    let n_layers = f.u32(&f.akey("block_count").unwrap()).unwrap();

    for i in 0..n_layers {
        for suffix in [
            "attn_norm.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ] {
            let name = format!("blk.{i}.{suffix}");
            f.tensor(&name).unwrap_or_else(|e| panic!("{e}"));
        }
    }
    f.tensor("token_embd.weight").unwrap();
    f.tensor("output_norm.weight").unwrap();
}

#[test]
fn tensor_shapes_are_consistent_with_config() {
    let f = model_or_skip!();
    let d_model = f.u64(&f.akey("embedding_length").unwrap()).unwrap();
    let n_head = f.u64(&f.akey("attention.head_count").unwrap()).unwrap();
    let n_kv = f.u64(&f.akey("attention.head_count_kv").unwrap()).unwrap();
    let d_head = d_model / n_head;
    let d_ff = f.u64(&f.akey("feed_forward_length").unwrap()).unwrap();

    // ggml stores [in, out].
    assert_eq!(
        f.tensor("blk.0.attn_q.weight").unwrap().dims,
        vec![d_model, n_head * d_head]
    );
    assert_eq!(
        f.tensor("blk.0.attn_k.weight").unwrap().dims,
        vec![d_model, n_kv * d_head]
    );
    assert_eq!(
        f.tensor("blk.0.ffn_gate.weight").unwrap().dims,
        vec![d_model, d_ff]
    );
    assert_eq!(
        f.tensor("blk.0.ffn_down.weight").unwrap().dims,
        vec![d_ff, d_model]
    );
}

#[test]
fn tensor_data_is_in_bounds_and_sized() {
    let f = model_or_skip!();
    for t in f.tensors().values() {
        let bytes = f.data(t);
        assert_eq!(bytes.len(), t.n_bytes, "{}", t.name);
        assert_eq!(
            t.n_bytes,
            t.ty.size_for(t.n_elements).unwrap(),
            "{}",
            t.name
        );
    }
}

#[test]
fn norm_weights_are_finite_f32() {
    let f = model_or_skip!();
    let t = f.tensor("output_norm.weight").unwrap();
    assert_eq!(t.ty, GgmlType::F32);

    let bytes = f.data(t);
    let vals: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(vals.len(), t.n_elements);
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "norm has non-finite weights"
    );
    // RMSNorm gains sit near 1; a wildly different scale means we misread the
    // offset and are looking at some other tensor's bytes.
    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
    assert!(mean.abs() > 0.01 && mean.abs() < 100.0, "mean gain {mean}");
}

#[test]
fn vocab_matches_token_type_array() {
    let f = model_or_skip!();
    let tokens = f.str_array("tokenizer.ggml.tokens").unwrap();
    let types = f.int_array("tokenizer.ggml.token_type").unwrap();
    assert_eq!(tokens.len(), types.len());
    assert!(tokens.len() > 100_000);

    let eos = f.u32("tokenizer.ggml.eos_token_id").unwrap() as usize;
    assert_eq!(tokens[eos], "<|im_end|>");
}
