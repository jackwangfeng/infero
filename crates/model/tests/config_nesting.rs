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
    // `out_hidden_size` has to track the text side, because the merger's output
    // is spliced into the embedding rows and `Config::from_hf` refuses a
    // mismatch. Reading it back off `text` rather than repeating 5120 keeps the
    // override-driven tests below working when they change the width.
    let d_model = text["hidden_size"].as_u64().expect("the fixture sets hidden_size");
    serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": false,
        "image_token_id": 151655,
        "video_token_id": 151656,
        "text_config": text,
        // A complete tower, not a two-key stub. `hidden_size` here is still the
        // distractor it always was — 1152 against the text side's 5120, so a
        // parser reading the outer object picks up the wrong number — and the
        // rest is what `vision_config` actually carries, so the vision parse is
        // exercised rather than skipped.
        "vision_config": {
            "depth": 27,
            "hidden_size": 1152,
            "num_heads": 16,
            "intermediate_size": 4304,
            "out_hidden_size": d_model,
            "in_channels": 3,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "num_position_embeddings": 2304,
        },
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

// ------------------------------------------------------- rope_parameters
//
// `rope_theta` and `partial_rotary_factor` sit in
// `text_config.rope_parameters`, one level below the dimensions. Reading them
// off `text_config` does not fail; it finds nothing and substitutes a default.
// So every test here has to show that the *other* reading produces a different
// answer, otherwise it is compatible with the bug it is meant to catch.

/// The real 27B shape: dimensions on `text_config`, rope settings one level
/// further in. `rope_theta` appears only in `rope_parameters`.
fn qwen35_shaped(rope_parameters: serde_json::Value) -> serde_json::Value {
    let mut j = qwen38_shaped(serde_json::json!({
        "num_attention_heads": 24,
        "head_dim": 256,
    }));
    // Drop the flat spelling so nothing can be read from it by accident.
    j["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("rope_theta");
    j["text_config"]["rope_parameters"] = rope_parameters;
    j
}

/// The checkpoint's own `rope_parameters` block.
fn real_rope_parameters() -> serde_json::Value {
    serde_json::json!({
        "rope_type": "default",
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
        "mrope_interleaved": true,
        "mrope_section": [11, 11, 10],
    })
}

/// `rope_theta` comes out of `rope_parameters`, and — the part that makes this
/// a test rather than a restatement — the reading that ignores the nesting
/// produces 10000 instead, which is what shipped before this fix.
#[test]
fn rope_theta_is_read_from_rope_parameters() {
    let j = qwen35_shaped(real_rope_parameters());
    let cfg = Config::from_hf(&j, "qwen35").expect("the real config shape should parse");
    assert_eq!(cfg.rope_theta, 10_000_000.0);

    // The other reading: `text_config.rope_theta`, which this config does not
    // have. Confirm it really is absent, so the assertion above is evidence
    // about where the value was found and not just about its value.
    assert!(
        j["text_config"]["rope_theta"].is_null(),
        "this fixture must not carry the flat spelling, or the test cannot \
         distinguish the two readings"
    );
    assert_ne!(
        cfg.rope_theta, 10_000.0,
        "10000 is the default a parser reaching for the flat spelling lands on; \
         getting it here means the nesting is still being ignored"
    );
}

/// The flat spelling still wins where there is no `rope_parameters` — every
/// checkpoint before this one — and the nested one wins where both exist.
#[test]
fn rope_theta_prefers_the_nested_spelling_but_still_reads_the_flat_one() {
    // Flat only: unchanged behaviour.
    let mut flat = qwen35_shaped(serde_json::json!({ "rope_type": "default" }));
    flat["text_config"]["rope_theta"] = serde_json::json!(1_000_000.0);
    assert_eq!(Config::from_hf(&flat, "m").unwrap().rope_theta, 1_000_000.0);

    // Both, disagreeing: `rope_parameters` is the authoritative spelling, and
    // the two values are different so the assertion picks a side.
    let mut both = qwen35_shaped(real_rope_parameters());
    both["text_config"]["rope_theta"] = serde_json::json!(1_000_000.0);
    let cfg = Config::from_hf(&both, "m").unwrap();
    assert_eq!(cfg.rope_theta, 10_000_000.0, "rope_parameters wins");
    assert_ne!(cfg.rope_theta, 1_000_000.0, "the flat value must lose");
}

/// 24 heads of 256 with a factor of 0.25: 64 dimensions rotate, 192 do not.
#[test]
fn the_27b_rotates_64_of_its_256_dimensions() {
    let j = qwen35_shaped(real_rope_parameters());
    let cfg = Config::from_hf(&j, "qwen35").unwrap();
    assert_eq!(cfg.d_head, 256);
    assert_eq!(cfg.rotary_dim, 64, "int(256 * 0.25)");
    assert_ne!(
        cfg.rotary_dim, cfg.d_head,
        "a rotary width equal to d_head is exactly the bug: it rotates the \
         whole head and normalizes the frequencies by 256"
    );
    // And the per-pair frequency table follows the rotary width, not d_head.
    assert_eq!(cfg.rope_freq_factors(&j).len(), 32, "rotary_dim / 2");
}

/// `partial_rotary_factor` is duplicated on the real checkpoint, so both
/// locations have to be read — and each has to be read *on its own*, which is
/// what these two halves establish separately.
#[test]
fn partial_rotary_factor_is_read_from_either_location() {
    // Nested only.
    let nested = qwen35_shaped(real_rope_parameters());
    assert!(nested["text_config"]["partial_rotary_factor"].is_null());
    assert_eq!(Config::from_hf(&nested, "m").unwrap().rotary_dim, 64);

    // Flat only: `rope_parameters` exists but says nothing about the factor,
    // so a parser that only looks inside it would fall back to the full width.
    let mut flat = qwen35_shaped(serde_json::json!({
        "rope_type": "default",
        "rope_theta": 10000000.0,
    }));
    flat["text_config"]["partial_rotary_factor"] = serde_json::json!(0.25);
    assert!(flat["text_config"]["rope_parameters"]["partial_rotary_factor"].is_null());
    let cfg = Config::from_hf(&flat, "m").unwrap();
    assert_eq!(cfg.rotary_dim, 64);
    assert_ne!(
        cfg.rotary_dim, cfg.d_head,
        "256 is what a nested-only reader would report for this config"
    );

    // Both, agreeing, which is the real file.
    let mut both = qwen35_shaped(real_rope_parameters());
    both["text_config"]["partial_rotary_factor"] = serde_json::json!(0.25);
    assert_eq!(Config::from_hf(&both, "m").unwrap().rotary_dim, 64);
}

/// No factor anywhere means the whole head rotates. This is the regression
/// guard for every model tuili already runs: `rotary_dim` has to default to
/// `d_head` exactly, in both config shapes and in the GGUF path.
#[test]
fn the_rotary_width_defaults_to_the_whole_head() {
    // Nested config with a `rope_parameters` that mentions no factor.
    let j = qwen35_shaped(serde_json::json!({
        "rope_type": "default",
        "rope_theta": 10000000.0,
    }));
    let cfg = Config::from_hf(&j, "m").unwrap();
    assert_eq!(cfg.rotary_dim, cfg.d_head);
    assert_eq!(cfg.rotary_dim, 256);

    // No `rope_parameters` object at all.
    let flat = serde_json::json!({
        "model_type": "qwen3",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_hidden_layers": 36,
        "intermediate_size": 12288,
        "vocab_size": 151936,
        "rope_theta": 1000000.0,
    });
    let cfg = Config::from_hf(&flat, "qwen3-8b").unwrap();
    assert_eq!(cfg.rotary_dim, 128);
    assert_eq!(cfg.rotary_dim, cfg.d_head);
    assert_eq!(cfg.rope_freq_factors(&flat).len(), 64, "d_head / 2");
}

/// A factor of 1.0 is the same thing said explicitly, and must not come out one
/// dimension short through a rounding accident.
#[test]
fn a_factor_of_one_rotates_the_whole_head() {
    let j = qwen35_shaped(serde_json::json!({
        "rope_type": "default",
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 1.0,
    }));
    let cfg = Config::from_hf(&j, "m").unwrap();
    assert_eq!(cfg.rotary_dim, 256);
}

/// A factor that lands on an odd width cannot be paired, and has to be refused
/// rather than silently truncated to something workable.
#[test]
fn a_factor_giving_an_odd_rotary_width_is_refused() {
    // int(256 * 0.1) == 25.
    let j = qwen35_shaped(serde_json::json!({
        "rope_type": "default",
        "partial_rotary_factor": 0.1,
    }));
    let err = Config::from_hf(&j, "m")
        .expect_err("an odd rotary width must not load")
        .to_string();
    assert!(err.contains("25"), "the error should name the width: {err}");

    // And one above 1.0, which would rotate past the end of the head.
    let j = qwen35_shaped(serde_json::json!({
        "rope_type": "default",
        "partial_rotary_factor": 1.5,
    }));
    assert!(
        Config::from_hf(&j, "m").is_err(),
        "a rotary width wider than d_head must not load"
    );
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

/// The vision tower's dimensions come from `vision_config`, and the placeholder
/// ids from the *outer* object.
///
/// Two separate traps in one config. `vision_config.hidden_size` is 1152 where
/// the text side is 5120, so a parser reading the wrong level gets a plausible
/// number rather than an error. And `image_token_id` is not in `vision_config`
/// at all — it is language-model vocabulary, which is why it sits outside — so a
/// parser looking for it beside the tower's dimensions finds nothing and would
/// have to default, and the only available default is another model's ids.
///
/// The fixture deliberately carries Qwen2-VL's 151655 / 151656 rather than this
/// checkpoint's 248056 / 248057, so that a parser substituting a constant for
/// the config reads through as correct here and wrong on the real checkpoint —
/// or, with this test, wrong here and caught.
#[test]
fn the_vision_tower_is_read_from_vision_config_and_its_ids_from_the_outer_object() {
    let j = qwen38_shaped(serde_json::json!({}));
    let cfg = Config::from_hf(&j, "qwen38").expect("the fixture should parse");
    let v = cfg.vision.expect("the fixture has a vision_config");
    assert_eq!(v.depth, 27);
    assert_eq!(v.hidden, 1152, "the tower's own width, not the text model's");
    assert_eq!(v.heads, 16);
    assert_eq!(v.intermediate, 4304);
    assert_eq!(v.grid_per_side(), 48, "2304 position embeddings is 48 on a side");
    assert_eq!(
        v.out_hidden, cfg.d_model,
        "the merger's output is spliced into the embedding rows, so it has to \
         match the text model's width"
    );
    assert_eq!(v.image_token, 151_655, "read, not assumed");
    assert_eq!(v.video_token, 151_656);
    // And the text side is untouched by any of it.
    assert_eq!(cfg.d_model, 5120);
}

/// A `vision_config` that names some dimensions and not others is an error.
///
/// The alternative is to treat a partial object as "no tower", which would load
/// a multimodal checkpoint as text-only and answer questions about images it
/// never looked at.
#[test]
fn a_partial_vision_config_is_refused() {
    let mut j = qwen38_shaped(serde_json::json!({}));
    j["vision_config"]
        .as_object_mut()
        .unwrap()
        .remove("num_heads");
    let err = Config::from_hf(&j, "qwen38").expect_err("a tower with no head count");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("num_heads"),
        "the error should name the missing key, got: {msg}"
    );
}

/// The merger's output width is checked against the text model's, not assumed
/// from it.
///
/// `Qwen3_5VisionModel` defaults `out_hidden_size` to 3584 while this
/// checkpoint's text side is 5120. A loader that trusted the class default would
/// splice 3584 floats into rows of 5120 — no shape error at the config, and
/// features landing in two-thirds of each row.
#[test]
fn a_merger_that_projects_to_the_wrong_width_is_refused() {
    let mut j = qwen38_shaped(serde_json::json!({}));
    j["vision_config"]["out_hidden_size"] = serde_json::json!(3584);
    let err = Config::from_hf(&j, "qwen38").expect_err("3584 into 5120 rows");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("3584") && msg.contains("5120"),
        "the error should name both widths, got: {msg}"
    );
}
