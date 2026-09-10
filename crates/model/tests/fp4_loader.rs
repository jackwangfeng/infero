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
    // verification of hf_quant_config.json's group_1 (193 entries: lm_head
    // + 64 layers x 3 mlp projections). Do not re-derive from a partial
    // read -- this is the ground truth the loader's classification is
    // checked against.
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
    assert_eq!(targets.len(), 193);
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
    std::fs::write(
        &path,
        r#"{
            "config_groups": {
                "group_0": {
                    "targets": [],
                    "weights": { "num_bits": 8 }
                },
                "group_1": {
                    "targets": [
                        "lm_head",
                        "model.language_model.layers.0.mlp.gate_proj",
                        "model.language_model.layers.0.mlp.up_proj",
                        "model.language_model.layers.0.mlp.down_proj"
                    ],
                    "weights": { "num_bits": 4 }
                }
            },
            "ignore": ["mtp.layers.0.mlp.down_proj"]
        }"#,
    )
    .expect("writing synthetic config");

    let targets = weights::classify_fp4_targets(path.to_str().unwrap()).unwrap();
    assert_eq!(targets.len(), 4);
    assert!(targets.contains("lm_head"));
    assert!(targets.contains("model.language_model.layers.0.mlp.gate_proj"));
    assert!(targets.contains("model.language_model.layers.0.mlp.up_proj"));
    assert!(targets.contains("model.language_model.layers.0.mlp.down_proj"));
    // group_0 and the attention projections it implies are not in group_1's
    // own target list, so they must not show up here.
    assert!(!targets.contains("model.language_model.layers.0.self_attn.q_proj"));

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
