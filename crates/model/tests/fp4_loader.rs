//! Loader classification and dispatch for `RadixArk/Qwen3.8-27B-NVFP4`'s real
//! `hf_quant_config.json` -- see `crates/model/src/weights.rs`'s
//! `classify_fp4_targets` and the `fp4_targets`-gated branch it feeds inside
//! `load_awq`/`projection_bytes`.
//!
//! Both real checkpoints these tests care about (the NVFP4 one, and the
//! existing production FP8 one they must not affect) are large, real, and
//! only downloaded to `bw` -- neither exists in this sandbox. Both tests
//! skip-gate on the real path/env var being present, mirroring this crate's
//! own established pattern (see `tensor_parallel_load.rs`'s
//! `INFERO_TEST_TP_GGUF`) rather than failing or faking data.

use std::path::PathBuf;

use infero_cuda::Device;
use infero_model::config::Config;
use infero_model::weights;

#[test]
fn radixark_nvfp4_targets_classified_correctly() {
    // Real, exhaustive target list, copied from this session's own real
    // verification of hf_quant_config.json's real per-tensor
    // `quantization.quantized_layers` map (193 entries with
    // `quant_algo == "NVFP4"`: lm_head + 64 layers x 3 mlp projections; an
    // earlier pass of this file assumed a `config_groups`/`group_1` shape
    // that turned out not to match the real on-disk file -- corrected).
    // Do not re-derive from a partial read -- this is the ground truth the
    // loader's classification is checked against.
    let quant_config_path = "/home/jeff/models/Qwen3.8-27B-NVFP4/hf_quant_config.json";
    // Skip gracefully if the checkpoint isn't present on this machine
    // (it's real, large, and only downloaded to `bw`) -- mirror how this
    // codebase's own tests already skip when a real GPU/checkpoint isn't
    // available (e.g. `kernels()?` returning early elsewhere in this
    // crate's test suite).
    if !std::path::Path::new(quant_config_path).exists() {
        eprintln!("skipping: real checkpoint not present on this machine");
        return;
    }
    let targets = infero_model::weights::classify_fp4_targets(quant_config_path).unwrap();
    assert!(targets.contains("lm_head"));
    assert!(targets.contains("model.language_model.layers.0.mlp.gate_proj"));
    assert!(targets.contains("model.language_model.layers.63.mlp.down_proj"));
    assert!(!targets.contains("model.language_model.layers.0.self_attn.q_proj"));
    let weights::Fp4Targets::Explicit(set) = &targets else {
        panic!("RadixArk's real checkpoint uses the quantized_layers (Explicit) shape");
    };
    assert_eq!(set.len(), 193);
}

/// A tiny, synthetic `hf_quant_config.json`-shaped fixture -- real-shaped
/// target names, but not the real checkpoint -- so `classify_fp4_targets`'s
/// own parsing logic has fast, deterministic, locally-green test evidence
/// independent of whether the real 21.9 GiB checkpoint is downloaded here.
/// Not a brief requirement; the brief's own two tests (this file's other
/// two) are the ones the task is checked against.
#[test]
fn classify_fp4_targets_parses_a_synthetic_config() {
    let dir = std::env::temp_dir().join(format!(
        "infero-fp4-loader-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("creating temp dir");
    let path = dir.join("hf_quant_config.json");
    // Real schema (verified against the actual checkpoint on `bw`, not the
    // `config_groups`/`group_1` shape an earlier version of this test/loader
    // wrongly assumed): a flat per-tensor map,
    // `quantization.quantized_layers.<name>.quant_algo`, `"NVFP4"` or
    // `"FP8"`.
    std::fs::write(
        &path,
        r#"{
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": "FP8",
                "quantized_layers": {
                    "lm_head": { "quant_algo": "NVFP4" },
                    "model.language_model.layers.0.mlp.gate_proj": { "quant_algo": "NVFP4", "group_size": 16 },
                    "model.language_model.layers.0.mlp.up_proj": { "quant_algo": "NVFP4", "group_size": 16 },
                    "model.language_model.layers.0.mlp.down_proj": { "quant_algo": "NVFP4", "group_size": 16 },
                    "model.language_model.layers.0.self_attn.q_proj": { "quant_algo": "FP8" },
                    "model.language_model.layers.0.linear_attn.out_proj": { "quant_algo": "FP8" }
                },
                "exclude_modules": ["mtp.layers.0.mlp.down_proj"]
            }
        }"#,
    )
    .expect("writing synthetic config");

    let targets = weights::classify_fp4_targets(path.to_str().unwrap()).unwrap();
    let weights::Fp4Targets::Explicit(set) = &targets else {
        panic!("this synthetic fixture uses the quantized_layers (Explicit) shape");
    };
    assert_eq!(set.len(), 4);
    assert!(targets.contains("lm_head"));
    assert!(targets.contains("model.language_model.layers.0.mlp.gate_proj"));
    assert!(targets.contains("model.language_model.layers.0.mlp.up_proj"));
    assert!(targets.contains("model.language_model.layers.0.mlp.down_proj"));
    // FP8 entries in quantized_layers are real, present, non-NVFP4 members
    // of the same map -- they must not show up here.
    assert!(!targets.contains("model.language_model.layers.0.self_attn.q_proj"));
    assert!(!targets.contains("model.language_model.layers.0.linear_attn.out_proj"));

    let _ = std::fs::remove_dir_all(&dir);
}

/// The second real `hf_quant_config.json` shape found in the wild
/// (`AxionML/Qwen3.5-9B-NVFP4`, same `qwen3_5` architecture, attention AND
/// GDN projections quantized to NVFP4 too -- not just the FFN): a uniform
/// top-level `quantization.quant_algo == "NVFP4"` with
/// `quantization.exclude_modules`, a wildcard-supporting denylist. Real
/// shape, copied from the actual file (trimmed to a few representative
/// entries, not all 25 real `conv1d` exclusions).
#[test]
fn classify_fp4_targets_parses_a_synthetic_uniform_denylist_config() {
    let dir = std::env::temp_dir().join(format!(
        "infero-fp4-loader-denylist-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("creating temp dir");
    let path = dir.join("hf_quant_config.json");
    std::fs::write(
        &path,
        r#"{
            "quantization": {
                "quant_algo": "NVFP4",
                "kv_cache_quant_algo": null,
                "group_size": 16,
                "exclude_modules": [
                    "lm_head",
                    "model.language_model.layers.0.linear_attn.conv1d",
                    "model.visual*",
                    "mtp.layers.0*"
                ]
            }
        }"#,
    )
    .expect("writing synthetic config");

    let targets = weights::classify_fp4_targets(path.to_str().unwrap()).unwrap();
    assert!(matches!(targets, weights::Fp4Targets::AllExcept(_)));
    // Exact-name exclusions.
    assert!(!targets.contains("lm_head"));
    assert!(!targets.contains("model.language_model.layers.0.linear_attn.conv1d"));
    // Wildcard exclusions: any tensor under the prefix.
    assert!(!targets.contains("model.visual.blocks.0.attn.qkv"));
    assert!(!targets.contains("mtp.layers.0.mlp.gate_proj"));
    // Real, un-excluded projections -- attention AND GDN, not just FFN,
    // which is the whole point of this checkpoint shape existing.
    assert!(targets.contains("model.language_model.layers.0.mlp.gate_proj"));
    assert!(targets.contains("model.language_model.layers.0.self_attn.q_proj"));
    assert!(targets.contains("model.language_model.layers.0.linear_attn.in_proj_qkv"));
    assert!(targets.contains("model.language_model.layers.0.linear_attn.out_proj"));
    // A different layer's conv1d isn't in this trimmed exclude list, but a
    // real conv1d is never queried through this path anyway (loaded via a
    // separate, non-`fp4_targets`-gated closure) -- not asserted here.

    let _ = std::fs::remove_dir_all(&dir);
}

fn fp8_checkpoint_dir() -> Option<PathBuf> {
    let p = std::env::var("INFERO_TEST_FP8_SAFETENSORS").ok().map(PathBuf::from)?;
    p.exists().then_some(p)
}

#[test]
fn a_non_nvfp4_checkpoint_is_unaffected() {
    // Load one real matrix from the existing production qwen38-27b-fp8
    // checkpoint (no hf_quant_config.json present) and confirm it still
    // classifies as F8E4M3, proving this task's new code path is inert
    // for checkpoints that don't opt into it.
    let Some(dir) = fp8_checkpoint_dir() else {
        eprintln!(
            "skipping: set INFERO_TEST_FP8_SAFETENSORS to the real qwen38-27b-fp8 \
             checkpoint directory (only present on bw)"
        );
        return;
    };
    // The regression this task actually cares about: this checkpoint has no
    // hf_quant_config.json of its own at all.
    assert!(
        !dir.join("hf_quant_config.json").exists(),
        "{}: has a hf_quant_config.json -- this is supposed to be the plain FP8 \
         production checkpoint (no NVFP4 opt-in), not the NVFP4 one; point \
         INFERO_TEST_FP8_SAFETENSORS at the real qwen38-27b-fp8 directory instead",
        dir.display()
    );

    let shards = infero_safetensors::Shards::open_dir(&dir).expect("opening checkpoint");
    let json = shards.json("config.json").expect("reading config.json");
    let name = dir
        .file_name()
        .map_or("unnamed", |s| s.to_str().unwrap_or("unnamed"));
    let cfg = Config::from_hf(&json, name).expect("parsing config");
    let freqs = cfg.rope_freq_factors(&json);

    let dev = Device::new(0).expect("device");
    let w = weights::load_awq(&dev, &shards, &cfg, &freqs, None).expect("loading checkpoint");

    let layer0 = &w.layers[0];
    let dense = layer0.dense.as_ref().expect(
        "layer 0 has no dense FFN -- this test assumes qwen38-27b-fp8's real \
         architecture (dense mlp.gate_proj/up_proj/down_proj per layer)",
    );
    assert_eq!(
        dense.w_gate.ty,
        infero_kernels::WeightType::F8E4M3,
        "layer 0's gate_proj classified as {:?}, not F8E4M3 -- the new NVFP4 \
         routing (gated on fp4_targets, which should be empty here) must be \
         completely inert for a checkpoint with no hf_quant_config.json",
        dense.w_gate.ty
    );
    assert_eq!(dense.w_up.ty, infero_kernels::WeightType::F8E4M3);
    assert_eq!(dense.w_down.ty, infero_kernels::WeightType::F8E4M3);
}
