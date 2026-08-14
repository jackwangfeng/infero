//! Model hyper-parameters, read from GGUF metadata or a Hugging Face
//! `config.json`.

use anyhow::{Context, Result, bail};
use tuili_gguf::Gguf;

/// Architectures whose block structure matches the one in `Model::forward`:
/// pre-norm RMSNorm, GQA attention, SwiGLU MLP.
const SUPPORTED: &[&str] = &["qwen2", "qwen3", "llama", "baichuan", "minicpm"];

/// Architectures whose GGUF conversion permutes Q and K so that the
/// interleaved rotary pairing reproduces Hugging Face's rotate-half.
///
/// This is not recorded in the file; it follows from the architecture, exactly
/// as it does in llama.cpp's `llama_model_rope_type`. Guessing wrong produces
/// fluent output that drifts with position rather than an error.
const INTERLEAVED_ROPE: &[&str] = &["llama", "baichuan", "minicpm"];

#[derive(Debug, Clone)]
pub struct Config {
    pub arch: String,
    pub name: String,
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub d_head: usize,
    pub d_ff: usize,
    pub vocab_size: usize,
    pub context_length: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    /// Linear RoPE scaling; 1.0 unless the model was trained with it.
    pub rope_freq_scale: f32,
    /// True when the output projection reuses the embedding matrix.
    pub tied_embeddings: bool,
    /// Rotary pairing: `2i` with `2i+1` rather than `i` with `i + d/2`.
    pub interleaved_rope: bool,
}

impl Config {
    pub fn from_gguf(f: &Gguf) -> Result<Self> {
        let arch = f.arch()?.to_string();
        if !SUPPORTED.contains(&arch.as_str()) {
            // Not fatal: the tensor names are what actually matter, and an
            // unlisted arch with the same block shape will just work.
            tracing::warn!(
                arch = %arch,
                "architecture is untested; expecting a llama-style block layout"
            );
        }

        let key = |s: &str| format!("{arch}.{s}");
        let arch_owned = arch.clone();
        let n_layers = f.usize(&key("block_count"))?;
        let d_model = f.usize(&key("embedding_length"))?;
        let n_heads = f.usize(&key("attention.head_count"))?;
        let n_kv_heads = f.usize(&key("attention.head_count_kv")).unwrap_or(n_heads);
        let d_ff = f.usize(&key("feed_forward_length"))?;

        // Most models leave rope.dimension_count implicit at d_model/n_heads.
        let d_head = f
            .usize(&key("rope.dimension_count"))
            .unwrap_or_else(|_| d_model / n_heads.max(1));

        if n_heads == 0 || n_kv_heads == 0 {
            bail!("head counts must be non-zero");
        }
        if !n_heads.is_multiple_of(n_kv_heads) {
            bail!("{n_heads} query heads do not divide into {n_kv_heads} kv heads");
        }
        if d_head * n_heads != d_model {
            // Some models genuinely have a head dim that isn't d_model/n_heads,
            // but our q/k/v buffers assume the projections are square.
            bail!(
                "d_head {d_head} * n_heads {n_heads} != d_model {d_model}; \
                 this layout is not supported"
            );
        }
        if !d_head.is_multiple_of(2) {
            bail!("d_head {d_head} must be even for rotary embeddings");
        }

        let vocab_size = f
            .get_tensor("token_embd.weight")
            .map(|t| t.dims[1] as usize)
            .context("model has no token_embd.weight")?;

        Ok(Self {
            arch: arch_owned,
            name: f.str("general.name").unwrap_or("unnamed").to_string(),
            n_layers,
            d_model,
            n_heads,
            n_kv_heads,
            d_head,
            d_ff,
            vocab_size,
            context_length: f.usize(&key("context_length")).unwrap_or(4096),
            rms_eps: f
                .f32(&key("attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-5),
            rope_theta: f.f32(&key("rope.freq_base")).unwrap_or(10_000.0),
            rope_freq_scale: f
                .f32(&key("rope.scaling.factor"))
                .map(|s| if s > 0.0 { 1.0 / s } else { 1.0 })
                .unwrap_or(1.0),
            tied_embeddings: f.get_tensor("output.weight").is_none(),
            interleaved_rope: INTERLEAVED_ROPE.contains(&arch.as_str()),
        })
    }

    /// From a Hugging Face `config.json`, as an AWQ checkpoint ships it.
    ///
    /// The one thing that is not simply a rename: Hugging Face weights are not
    /// permuted, so the rotary embedding pairs `i` with `i + d/2` — the NeoX
    /// convention — where the same model out of GGUF pairs `2i` with `2i+1`.
    /// llama.cpp does that permutation during conversion; reading the
    /// checkpoint directly means not needing it, but it does mean the flag is
    /// the opposite of what the architecture name implies.
    pub fn from_hf(j: &serde_json::Value, name: &str) -> Result<Self> {
        let u = |k: &str| -> Result<usize> {
            j[k].as_u64()
                .map(|v| v as usize)
                .with_context(|| format!("config.json has no integer `{k}`"))
        };
        let arch = j["model_type"].as_str().unwrap_or("llama").to_string();
        if !SUPPORTED.contains(&arch.as_str()) {
            tracing::warn!(
                arch = %arch,
                "architecture is untested; expecting a llama-style block layout"
            );
        }
        let (n_heads, d_model) = (u("num_attention_heads")?, u("hidden_size")?);
        let n_kv_heads = j["num_key_value_heads"]
            .as_u64()
            .map_or(n_heads, |v| v as usize);
        anyhow::ensure!(n_heads > 0 && n_kv_heads > 0, "head counts must be non-zero");
        anyhow::ensure!(
            n_heads.is_multiple_of(n_kv_heads),
            "{n_heads} query heads do not divide into {n_kv_heads} kv heads"
        );
        let d_head = j["head_dim"]
            .as_u64()
            .map_or(d_model / n_heads, |v| v as usize);
        anyhow::ensure!(
            d_head * n_heads == d_model,
            "d_head {d_head} * n_heads {n_heads} != d_model {d_model}; \
             this layout is not supported"
        );
        anyhow::ensure!(
            d_head.is_multiple_of(2),
            "d_head {d_head} must be even for rotary embeddings"
        );

        Ok(Self {
            arch,
            name: name.to_string(),
            n_layers: u("num_hidden_layers")?,
            d_model,
            n_heads,
            n_kv_heads,
            d_head,
            d_ff: u("intermediate_size")?,
            vocab_size: u("vocab_size")?,
            context_length: u("max_position_embeddings").unwrap_or(4096),
            rms_eps: j["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            rope_theta: j["rope_theta"].as_f64().unwrap_or(10_000.0) as f32,
            // Llama 3's scaling is per-dimension rather than a single factor;
            // it arrives through `rope_freq_factors` instead. A plain linear
            // `factor` would go here.
            rope_freq_scale: 1.0,
            tied_embeddings: j["tie_word_embeddings"].as_bool().unwrap_or(false),
            interleaved_rope: false,
        })
    }

    /// The per-dimension RoPE frequency divisors a Hugging Face config implies.
    ///
    /// GGUF ships these precomputed as `rope_freqs.weight`; a `config.json`
    /// describes them instead. Llama 3's recipe stretches wavelengths longer
    /// than `original_context / low_freq_factor` by the full factor, leaves
    /// those shorter than `original_context / high_freq_factor` alone, and
    /// ramps between — ignoring it costs nothing at position zero and
    /// progressively more further along, which reads as output that starts
    /// fine and drifts.
    pub fn rope_freq_factors(&self, j: &serde_json::Value) -> Vec<f32> {
        let half = self.d_head / 2;
        let s = &j["rope_scaling"];
        let ty = s["rope_type"].as_str().or_else(|| s["type"].as_str());
        if ty != Some("llama3") {
            return vec![1.0; half];
        }
        let f = |k: &str, d: f64| s[k].as_f64().unwrap_or(d) as f32;
        let (factor, low, high) = (f("factor", 8.0), f("low_freq_factor", 1.0), f("high_freq_factor", 4.0));
        let orig = s["original_max_position_embeddings"]
            .as_u64()
            .unwrap_or(8192) as f32;
        let (low_wl, high_wl) = (orig / low, orig / high);

        tracing::info!(factor, low, high, orig, "using llama3 rope frequency scaling");
        (0..half)
            .map(|i| {
                let inv_freq = self.rope_theta.powf(-2.0 * i as f32 / self.d_head as f32);
                let wavelen = std::f32::consts::TAU / inv_freq;
                if wavelen < high_wl {
                    1.0
                } else if wavelen > low_wl {
                    factor
                } else {
                    // Linear ramp in between, in the smoothing variable the
                    // reference implementation uses.
                    let smooth = (orig / wavelen - low) / (high - low);
                    1.0 / ((1.0 - smooth) / factor + smooth)
                }
            })
            .collect()
    }

    /// Query heads per kv head.
    pub fn gqa_ratio(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    /// 1/sqrt(d_head), the attention logit scale.
    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.d_head as f32).sqrt()
    }

    /// Elements in one layer's largest weight matrix, which sizes the
    /// dequantization scratch buffer used during prefill.
    pub fn max_layer_weight_elements(&self) -> usize {
        (self.d_model * self.d_ff).max(self.d_model * self.d_model)
    }
}

impl std::fmt::Display for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({}) {} layers, d_model {}, {} heads / {} kv, d_head {}, ffn {}, vocab {}, ctx {}, rope {}",
            self.name,
            self.arch,
            self.n_layers,
            self.d_model,
            self.n_heads,
            self.n_kv_heads,
            self.d_head,
            self.d_ff,
            self.vocab_size,
            self.context_length,
            if self.interleaved_rope {
                "interleaved"
            } else {
                "neox"
            },
        )
    }
}
