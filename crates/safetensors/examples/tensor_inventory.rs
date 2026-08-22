//! Does this checkpoint have every tensor the loader will ask for?
//!
//! The loader discovers a missing tensor by failing on it, which surfaces one
//! name per run. That was fine when a gap meant one rename; on Qwen3.8-27B it
//! meant four rounds of rebuild-and-retry, each revealing the next name. This
//! reports the whole set at once, before any device memory is touched.
//!
//! It deliberately does not use `tuili_model`'s loader. Calling the loader would
//! make this a second way to run the same code, which tells you nothing new; the
//! point is an independent list of what the loader is *documented* to want,
//! checked against what the file has. When the two disagree, one of them is
//! wrong and both are worth reading.
//!
//!   cargo run --release -p tuili-safetensors --example tensor_inventory -- <dir>

use std::collections::BTreeSet;

use anyhow::{Context, Result};
use tuili_safetensors::Shards;

/// A projection can be stored several ways, and the loader accepts any of them:
/// `.weight` for a plain or FP8 export, and the AWQ triple otherwise. Checking
/// only for `.weight` reports every working AWQ checkpoint as broken — which is
/// what the first version of this file did, and the reason it is worth saying:
/// a checker that fails on the models known to work is telling you about itself.
fn projection_forms(prefix: &str) -> Vec<Vec<String>> {
    vec![
        vec![format!("{prefix}.weight")],
        vec![
            format!("{prefix}.qweight"),
            format!("{prefix}.qzeros"),
            format!("{prefix}.scales"),
        ],
    ]
}

/// Every name the loader constructs, given what the config says.
struct Expected {
    /// Required: a missing one stops the load. Each entry is a set of
    /// alternative spellings, satisfied when *one whole* alternative is present.
    required: Vec<Vec<Vec<String>>>,
    /// Read with a fallback, so absence is a fact rather than a failure.
    optional: Vec<String>,
}

fn expected(cfg: &serde_json::Value, prefix: &str, n_layers: usize, linear: &[bool]) -> Expected {
    let mut required: Vec<Vec<Vec<String>>> = Vec::new();
    let mut optional: Vec<String> = vec!["lm_head.weight".to_string()];
    let stem = if prefix == "model.layers" {
        "model"
    } else {
        "model.language_model"
    };
    // The embedding and the final norm are never quantized in these exports.
    required.push(vec![vec![format!("{stem}.embed_tokens.weight")]]);
    required.push(vec![vec![format!("{stem}.norm.weight")]]);

    for i in 0..n_layers {
        let p = format!("{prefix}.{i}");
        for n in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            required.push(vec![vec![format!("{p}.{n}")]]);
        }
        for m in ["gate_proj", "up_proj", "down_proj"] {
            required.push(projection_forms(&format!("{p}.mlp.{m}")));
        }
        if linear.get(i).copied().unwrap_or(false) {
            let l = format!("{p}.linear_attn");
            for m in ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"] {
                required.push(projection_forms(&format!("{l}.{m}")));
            }
            for t in ["conv1d.weight", "A_log", "dt_bias", "norm.weight"] {
                required.push(vec![vec![format!("{l}.{t}")]]);
            }
        } else {
            for m in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                required.push(projection_forms(&format!("{p}.self_attn.{m}")));
                optional.push(format!("{p}.self_attn.{m}.bias"));
            }
            for m in ["q_norm", "k_norm"] {
                optional.push(format!("{p}.self_attn.{m}.weight"));
            }
        }
    }

    if cfg["text_config"]["mtp_num_hidden_layers"]
        .as_u64()
        .unwrap_or(0)
        > 0
    {
        for t in [
            "fc.weight",
            "norm.weight",
            "pre_fc_norm_embedding.weight",
            "pre_fc_norm_hidden.weight",
        ] {
            optional.push(format!("mtp.{t}"));
        }
    }
    Expected { required, optional }
}

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .context("usage: tensor_inventory <model-dir>")?;
    let w = Shards::open_dir(&dir)?;
    let cfg = w.json("config.json")?;
    let dims = if cfg["text_config"].is_object() {
        &cfg["text_config"]
    } else {
        &cfg
    };
    let n_layers = dims["num_hidden_layers"]
        .as_u64()
        .context("config has no num_hidden_layers")? as usize;

    // Which prefix the layers sit under, probed the way the loader probes.
    let prefix = ["model.layers", "model.language_model.layers"]
        .into_iter()
        .find(|pre| {
            [
                "input_layernorm.weight",
                "self_attn.q_proj.weight",
                "linear_attn.in_proj_qkv.weight",
            ]
            .iter()
            .any(|leaf| w.get(&format!("{pre}.0.{leaf}")).is_some())
        })
        .context("no layer 0 under either prefix")?;

    // Which blocks are linear, read from the tensors rather than from
    // `layer_types` — the same rule the loader uses, so a disagreement between
    // the config and the weights shows up here as a mismatch rather than as a
    // wrong slice much later.
    let linear: Vec<bool> = (0..n_layers)
        .map(|i| {
            w.get(&format!("{prefix}.{i}.linear_attn.in_proj_qkv.weight"))
                .is_some()
        })
        .collect();
    let n_linear = linear.iter().filter(|b| **b).count();

    println!("{dir}");
    println!("  layer prefix        {prefix}");
    println!("  layers              {n_layers}  ({n_linear} linear, {} attention)",
             n_layers - n_linear);
    if let Some(types) = dims["layer_types"].as_array() {
        let from_config: Vec<bool> = types
            .iter()
            .map(|t| t.as_str() == Some("linear_attention"))
            .collect();
        let agree = from_config.len() == linear.len() && from_config == linear;
        println!(
            "  layer_types         {}",
            if agree {
                "agrees with the tensors".to_string()
            } else {
                format!(
                    "DISAGREES with the tensors at {} of {} positions",
                    from_config
                        .iter()
                        .zip(&linear)
                        .filter(|(a, b)| a != b)
                        .count(),
                    linear.len()
                )
            }
        );
    }

    let exp = expected(&cfg, prefix, n_layers, &linear);
    // A requirement is met when one whole alternative spelling is present.
    // Reporting a partial one — two of AWQ's three files, say — is more useful
    // than reporting the whole requirement missing.
    let mut missing: Vec<String> = Vec::new();
    for alts in &exp.required {
        if alts
            .iter()
            .any(|alt| alt.iter().all(|n| w.get(n).is_some()))
        {
            continue;
        }
        let partial = alts
            .iter()
            .filter(|alt| alt.iter().any(|n| w.get(n).is_some()))
            .flat_map(|alt| alt.iter())
            .filter(|n| w.get(n).is_none())
            .cloned()
            .collect::<Vec<_>>();
        if partial.is_empty() {
            missing.push(format!("{} (no spelling present)", alts[0][0]));
        } else {
            for n in partial {
                missing.push(format!("{n} (an incomplete alternative)"));
            }
        }
    }
    let absent_optional: Vec<&String> = exp
        .optional
        .iter()
        .filter(|n| w.get(n).is_none())
        .collect();

    println!(
        "  required            {} named, {} missing",
        exp.required.iter().flatten().flatten().count(),
        missing.len()
    );
    for name in missing.iter().take(40) {
        println!("    MISSING  {name}");
    }
    if missing.len() > 40 {
        println!("    ... and {} more", missing.len() - 40);
    }

    // Optional absences are information, not problems — but the *pattern*
    // matters: q_norm present on every attention layer means Qwen3-style
    // per-head norms, present on none means Qwen2-style biases instead, and
    // present on some means something is wrong.
    let mut by_leaf: std::collections::BTreeMap<&str, (usize, usize)> = Default::default();
    for name in &exp.optional {
        let leaf = name.rsplit('.').take(2).collect::<Vec<_>>().join(".");
        let leaf: &str = Box::leak(leaf.into_boxed_str());
        let e = by_leaf.entry(leaf).or_default();
        e.1 += 1;
        if w.get(name).is_some() {
            e.0 += 1;
        }
    }
    println!("  optional            present / named");
    for (leaf, (have, want)) in &by_leaf {
        let note = if *have == 0 || have == want {
            ""
        } else {
            "   <- present on some layers and not others"
        };
        println!("    {leaf:<28} {have:>4} / {want:<4}{note}");
    }
    let _ = absent_optional;

    // Anything in the file the loader will never ask for. A large count here is
    // not a fault — the vision tower is 333 tensors the text loader ignores —
    // but it is the list to read when a checkpoint loads and behaves oddly.
    let wanted: BTreeSet<&str> = exp
        .required
        .iter()
        .flatten()
        .flatten()
        .chain(exp.optional.iter())
        .map(|s| s.as_str())
        .collect();
    let mut unclaimed: Vec<&str> = w
        .names()
        .filter(|n| !wanted.contains(*n) && !n.ends_with("_scale_inv"))
        .collect();
    unclaimed.sort_unstable();
    let mut groups: std::collections::BTreeMap<String, usize> = Default::default();
    for n in &unclaimed {
        let key = n
            .split('.')
            .map(|p| if p.parse::<u32>().is_ok() { "N" } else { p })
            .take(4)
            .collect::<Vec<_>>()
            .join(".");
        *groups.entry(key).or_default() += 1;
    }
    println!("  unclaimed           {} tensors the text loader ignores", unclaimed.len());
    for (k, n) in groups.iter().take(20) {
        println!("    {k:<44} {n:>4}");
    }

    if missing.is_empty() {
        println!("\nevery required tensor is present");
        Ok(())
    } else {
        anyhow::bail!("{} required tensors are missing", missing.len())
    }
}
