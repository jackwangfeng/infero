//! Reading a multimodal `config.json`, where the language model's dimensions
//! are nested rather than top-level.
//!
//! These run on the host: a config parse needs no device, and the useful checks
//! here are about which object a field is read from. Shapes taken from
//! Qwen3.8-27B, which is the checkpoint that made the nesting matter.

use tuili_model::Config;

/// The shape of a multimodal config: an outer object naming the wrapper and
/// carrying the vision tower, with the text model's dimensions inside.
fn qwen38_shaped(text_overrides: serde_json::Value) -> serde_json::Value {
    let mut text = serde_json::json!({
        "hidden_size": 5120,
        "num_attention_heads": 20,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "num_hidden_layers": 64,
        "intermediate_size": 17408,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-6,
        "rope_theta": 5000000.0,
        "full_attention_interval": 4,
    });
    if let (Some(t), Some(o)) = (text.as_object_mut(), text_overrides.as_object()) {
        for (k, v) in o {
            t.insert(k.clone(), v.clone());
        }
    }
    serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": false,
        "image_token_id": 151655,
        "video_token_id": 151656,
        "text_config": text,
        "vision_config": { "depth": 27, "hidden_size": 1152 },
    })
}

/// Every dimension comes from `text_config`, not the outer object. The outer
/// object here has no `hidden_size` at all, so a parser still reading the top
/// level fails rather than quietly picking up a vision-tower number.
#[test]
fn the_language_model_dimensions_are_read_from_text_config() {
    // 20 heads of 256 is 5120, which keeps the attention width equal to
    // d_model so this exercises the nesting alone.
    let j = qwen38_shaped(serde_json::json!({}));
    let cfg = Config::from_hf(&j, "qwen38").expect("nested dims should parse");
    assert_eq!(cfg.d_model, 5120);
    assert_eq!(cfg.n_heads, 20);
    assert_eq!(cfg.n_kv_heads, 4);
    assert_eq!(cfg.d_head, 256);
    assert_eq!(cfg.n_layers, 64);
    assert_eq!(cfg.d_ff, 17408);
    assert_eq!(cfg.vocab_size, 248320);
    assert_eq!(cfg.context_length, 262144);
    assert_eq!(cfg.rope_theta, 5_000_000.0);
    assert!((cfg.rms_eps - 1e-6).abs() < 1e-12);
}

/// `model_type` stays on the outer object: for a multimodal model the inner
/// type names only the text half, and the architecture decides block layout.
#[test]
fn the_architecture_comes_from_the_outer_object() {
    let mut j = qwen38_shaped(serde_json::json!({}));
    j["text_config"]["model_type"] = serde_json::json!("qwen3");
    let cfg = Config::from_hf(&j, "qwen38").unwrap();
    assert_eq!(cfg.arch, "qwen3_5");
}

/// Tying is a whole-model property stated once on the outer object.
#[test]
fn tied_embeddings_is_read_from_the_outer_object() {
    let mut j = qwen38_shaped(serde_json::json!({}));
    j["tie_word_embeddings"] = serde_json::json!(true);
    assert!(Config::from_hf(&j, "m").unwrap().tied_embeddings);
    j["tie_word_embeddings"] = serde_json::json!(false);
    assert!(!Config::from_hf(&j, "m").unwrap().tied_embeddings);
}

/// A checkpoint that states it only on the inner object is still read. The
/// fallback exists because nothing guarantees which level an exporter uses.
#[test]
fn tied_embeddings_falls_back_to_the_inner_object() {
    let mut j = qwen38_shaped(serde_json::json!({ "tie_word_embeddings": true }));
    j.as_object_mut().unwrap().remove("tie_word_embeddings");
    assert!(Config::from_hf(&j, "m").unwrap().tied_embeddings);
}

/// A flat config — every model tuili loaded before this one — must keep
/// parsing from the top level.
#[test]
fn a_flat_config_still_reads_from_the_top_level() {
    let j = serde_json::json!({
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_hidden_layers": 36,
        "intermediate_size": 12288,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
    });
    let cfg = Config::from_hf(&j, "qwen3-8b").unwrap();
    assert_eq!((cfg.d_model, cfg.n_heads, cfg.d_head), (4096, 32, 128));
    assert_eq!(cfg.arch, "qwen3");
}

/// The real Qwen3.8-27B widths — 24 heads of 256 against a 5120 residual — are
/// accepted, and the attention width is the head product rather than d_model.
/// This test replaces the assertion that used to refuse this shape; it is worth
/// keeping because on every other model the two widths coincide, so a
/// regression to `d_attn == d_model` would be invisible everywhere else.
#[test]
fn the_attention_width_is_the_head_product_not_d_model() {
    let j = qwen38_shaped(serde_json::json!({ "num_attention_heads": 24 }));
    let cfg = Config::from_hf(&j, "qwen38").expect("a wider attention block should load");
    assert_eq!(cfg.d_model, 5120, "the residual keeps its own width");
    assert_eq!(cfg.d_attn(), 6144, "24 heads of 256");
    assert_eq!(cfg.d_kv(), 1024, "4 kv heads of 256");
    assert_ne!(cfg.d_attn(), cfg.d_model, "the point of the test");
}

/// And where the two widths do coincide — every model tuili loaded before this
/// one — `d_attn` must agree with `d_model` rather than drifting.
#[test]
fn the_two_widths_still_agree_on_a_conventional_model() {
    let j = qwen38_shaped(serde_json::json!({}));
    let cfg = Config::from_hf(&j, "qwen38").unwrap();
    assert_eq!(cfg.d_attn(), cfg.d_model, "20 heads of 256 is 5120");
}

/// An odd head dimension cannot be rotated in pairs, and that check has to
/// survive reading from the nested object.
#[test]
fn an_odd_head_dim_is_still_refused_when_nested() {
    let j = qwen38_shaped(serde_json::json!({
        "num_attention_heads": 5120 / 255,
        "head_dim": 255,
    }));
    let err = Config::from_hf(&j, "m").unwrap_err().to_string();
    assert!(!err.is_empty(), "an odd d_head must be refused");
}
