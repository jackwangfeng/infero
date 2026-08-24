//! Reading a sparse `config.json`: which layers are MoE and how wide an expert
//! is.
//!
//! Host-only, like `config_nesting.rs` — a config parse needs no device. Shapes
//! taken from Qwen3-30B-A3B, the checkpoint this was written against.

use tuili_model::Config;

/// Qwen3-MoE's config, with room to override the sparsity fields.
fn qwen3_moe_shaped(overrides: serde_json::Value) -> serde_json::Value {
    let mut j = serde_json::json!({
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_hidden_layers": 48,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "norm_topk_prob": true,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": false,
    });
    if let (Some(t), Some(o)) = (j.as_object_mut(), overrides.as_object()) {
        for (k, v) in o {
            t.insert(k.clone(), v.clone());
        }
    }
    j
}

#[test]
fn the_expert_dimensions_come_from_the_moe_fields() {
    let cfg = Config::from_hf(&qwen3_moe_shaped(serde_json::json!({})), "moe").unwrap();
    let moe = cfg.moe.as_ref().expect("this config is sparse");
    assert_eq!(moe.n_experts, 128);
    assert_eq!(moe.n_active, 8);
    // `moe_intermediate_size`, not `intermediate_size`. Taking the dense width
    // here would size every expert 8x too wide and the loader would reject the
    // checkpoint — which is the good case; the bad one is a model where the two
    // are equal and nothing complains.
    assert_eq!(moe.d_ff_expert, 768);
    assert_eq!(cfg.d_ff, 6144, "the dense width is still read, for dense layers");
    assert!(moe.norm_topk_prob);
}

/// Every layer of this checkpoint is sparse, and the two fields that could say
/// otherwise both say so.
#[test]
fn a_sparse_step_of_one_and_no_exceptions_makes_every_layer_moe() {
    let cfg = Config::from_hf(&qwen3_moe_shaped(serde_json::json!({})), "moe").unwrap();
    let moe = cfg.moe.as_ref().unwrap();
    assert!((0..48).all(|i| moe.is_sparse(i)), "every layer should be sparse");
}

/// `mlp_only_layers` names layers that keep a dense FFN, and
/// `decoder_sparse_step` makes every n-th layer sparse. Both are in the config
/// this model ships, so both are read even though this checkpoint exercises
/// neither.
#[test]
fn dense_exceptions_and_a_sparse_stride_are_both_honoured() {
    let cfg = Config::from_hf(
        &qwen3_moe_shaped(serde_json::json!({ "mlp_only_layers": [0, 3] })),
        "moe",
    )
    .unwrap();
    let moe = cfg.moe.as_ref().unwrap();
    assert!(!moe.is_sparse(0));
    assert!(!moe.is_sparse(3));
    assert!(moe.is_sparse(1));

    let cfg = Config::from_hf(
        &qwen3_moe_shaped(serde_json::json!({ "decoder_sparse_step": 2 })),
        "moe",
    )
    .unwrap();
    let moe = cfg.moe.as_ref().unwrap();
    assert!(moe.is_sparse(0), "layer 0 is 0 % 2 == 0");
    assert!(!moe.is_sparse(1));
    assert!(moe.is_sparse(2));
}

/// A dense checkpoint must not grow a router. `qwen3` and `qwen3_moe` differ by
/// these fields and nothing else structural, so reading them optimistically
/// would give every dense model a 128-expert FFN it has no weights for.
#[test]
fn a_dense_config_has_no_moe() {
    let mut j = qwen3_moe_shaped(serde_json::json!({}));
    let o = j.as_object_mut().unwrap();
    o.insert("model_type".into(), serde_json::json!("qwen3"));
    for k in ["num_experts", "num_experts_per_tok", "moe_intermediate_size"] {
        o.remove(k);
    }
    let cfg = Config::from_hf(&j, "dense").unwrap();
    assert!(cfg.moe.is_none());
}

/// Half a sparsity config is a checkpoint nobody can size. Refusing beats
/// picking a default for the missing half: `num_experts` without
/// `num_experts_per_tok` would route to all of them, which runs and is 16x the
/// arithmetic.
#[test]
fn a_partial_moe_config_is_refused() {
    let mut j = qwen3_moe_shaped(serde_json::json!({}));
    j.as_object_mut().unwrap().remove("num_experts_per_tok");
    let err = Config::from_hf(&j, "moe").unwrap_err().to_string();
    assert!(err.contains("num_experts_per_tok"), "{err}");
}

/// Routing to more experts than exist is a config that cannot be satisfied.
#[test]
fn routing_to_more_experts_than_exist_is_refused() {
    let j = qwen3_moe_shaped(serde_json::json!({ "num_experts_per_tok": 200 }));
    let err = Config::from_hf(&j, "moe").unwrap_err().to_string();
    assert!(err.contains("200"), "{err}");
}
