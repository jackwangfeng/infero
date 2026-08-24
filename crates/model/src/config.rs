//! Model hyper-parameters, read from GGUF metadata or a Hugging Face
//! `config.json`.

use anyhow::{Context, Result, bail};
use tuili_gguf::Gguf;

/// Architectures whose block structure matches the one in `Model::forward`:
/// pre-norm RMSNorm, GQA attention, SwiGLU MLP.
const SUPPORTED: &[&str] = &[
    "qwen2",
    "qwen3",
    // Qwen3.5/3.8: the full-attention blocks are this same layout, but only
    // every fourth block is one — the rest are GatedDeltaNet. Listed so the
    // untested-architecture warning does not fire for a layout that is
    // recognised; the interleaving is enforced where the blocks are built.
    "qwen3_5",
    // The same model as `qwen3_5`, under the name llama.cpp's converter writes
    // into a GGUF. Two spellings because the two loaders read two different
    // files: `config.json`'s `model_type` for a safetensors checkpoint, and
    // `general.architecture` for a GGUF.
    "qwen35",
    // Qwen3-MoE: the same block as `qwen3` with a sparse FFN in place of the
    // dense one. The attention half is unchanged, which is why it is listed
    // here rather than treated as a new layout; what differs is `Config::moe`
    // and the weights the FFN reads.
    "qwen3_moe",
    "llama",
    "baichuan",
    "minicpm",
];

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
    /// How many of each head's dimensions the rotary embedding touches.
    ///
    /// `d_head` for every model tuili loaded before Qwen3.5, which is why this
    /// was not a field. Qwen3.5 rotates `int(head_dim * partial_rotary_factor)`
    /// = 64 of its 256 and passes the remaining 192 through untouched, and the
    /// frequency exponent is normalized by *this* width rather than by `d_head`
    /// — so the table is not the leading slice of the full-width one. Both
    /// mistakes run to completion and cost long-range retrieval only, which is
    /// the hardest kind of wrong to attribute.
    pub rotary_dim: usize,
    /// Linear RoPE scaling; 1.0 unless the model was trained with it.
    pub rope_freq_scale: f32,
    /// True when the output projection reuses the embedding matrix.
    pub tied_embeddings: bool,
    /// Rotary pairing: `2i` with `2i+1` rather than `i` with `i + d/2`.
    pub interleaved_rope: bool,
    /// Set when some of this model's blocks mix with a recurrence rather than
    /// with attention. `None` for every model tuili loaded before Qwen3.5.
    pub linear_attn: Option<LinearAttnConfig>,
    /// Whether the attention blocks carry an output gate, which makes `q_proj`
    /// twice as wide.
    ///
    /// Read from the config only to decide whether to allocate the gate buffer.
    /// What the forward pass *acts* on is `q_proj`'s actual column count, and
    /// the loader refuses a checkpoint where the two disagree — the config
    /// cannot silently change the arithmetic.
    pub attn_output_gate: bool,
    /// How many decoder layers the multi-token-prediction head has; `0` when
    /// the checkpoint has no head.
    ///
    /// One on Qwen3.5, and that does *not* bound the number of speculative
    /// tokens: the drafter loops `spec_step_idx % mtp_num_hidden_layers`, which
    /// is always zero here, so every draft step re-enters the same layer with
    /// the head's own previous output fed back in as the hidden state. See
    /// `notes/qwen3.5-mtp.md`.
    pub mtp_layers: usize,
    /// Whether the head owns an embedding table of its own.
    ///
    /// False on Qwen3.5, which is the config's way of saying what the
    /// checkpoint shows by shipping no `mtp.embed_tokens`: the head reads the
    /// text model's embedding and scores with the text model's `lm_head`. The
    /// loader checks the two agree rather than trusting either alone.
    pub mtp_dedicated_embeddings: bool,
    /// The vision tower, when the checkpoint carries one.
    pub vision: Option<VisionConfig>,
    /// Set when this model's FFN is a mixture of experts. `None` for every
    /// model tuili loaded before Qwen3-30B-A3B.
    pub moe: Option<MoeConfig>,
}

/// How a sparse FFN is shaped, and which layers have one.
///
/// `d_ff` on [`Config`] stays the *dense* width and is still read, because a
/// checkpoint may have both kinds of layer — `mlp_only_layers` names the ones
/// that keep a dense FFN. An expert is `d_ff_expert` wide, which on Qwen3-MoE
/// is an eighth of the dense width; taking `d_ff` for an expert sizes it 8x too
/// large.
#[derive(Debug, Clone)]
pub struct MoeConfig {
    pub n_experts: usize,
    /// How many experts each token is routed to.
    pub n_active: usize,
    pub d_ff_expert: usize,
    /// Whether the top-k router weights are renormalized to sum to one after
    /// the truncation. True on Qwen3-MoE; a model that trained without it and
    /// is served with it has every FFN output scaled by a token-dependent
    /// factor, which is fluent and wrong.
    pub norm_topk_prob: bool,
    /// Every `sparse_step`-th layer is sparse. 1 on Qwen3-MoE, which is every
    /// layer.
    pub sparse_step: usize,
    /// Layers that keep a dense FFN regardless of `sparse_step`.
    pub dense_layers: Vec<usize>,
}

impl MoeConfig {
    /// Whether layer `i` routes through experts.
    pub fn is_sparse(&self, i: usize) -> bool {
        !self.dense_layers.contains(&i) && self.sparse_step > 0 && i.is_multiple_of(self.sparse_step)
    }
}

/// The vision tower's dimensions, and the ids that reserve room for its output.
///
/// Read from `vision_config` rather than taken from
/// [`tuili_kernels::vision::VisionShape::QWEN35_27B`]: that constant is this
/// checkpoint's numbers, and a loader that reached for it would give a different
/// tower the 27B's depth and hidden size and produce a shape error deep inside
/// the tensor loop instead of at the config.
#[derive(Debug, Clone, Copy)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
    /// The width the merger projects to — the *text* model's `d_model`, and not
    /// derivable from it: `Qwen3_5VisionModel` defaults this to 3584 while this
    /// checkpoint's text side is 5120, so it has to be read and then checked.
    pub out_hidden: usize,
    pub in_channels: usize,
    pub patch: usize,
    pub temporal_patch: usize,
    pub merge: usize,
    /// `num_position_embeddings`, 2304 here — the learned grid is its square
    /// root on a side, so it has to be a perfect square.
    pub position_embeddings: usize,
    /// The placeholder ids a prompt uses to reserve one slot per vision token.
    /// 248056 and 248057 on this checkpoint, and emphatically not Qwen2-VL's
    /// 151655/151656.
    pub image_token: u32,
    pub video_token: u32,
}

impl VisionConfig {
    /// LayerNorm epsilon.
    ///
    /// `vision_config` does not carry it, so this cannot be checked against the
    /// checkpoint the way every dimension above is. It is the value the tower's
    /// kernels were validated at against a capture of the reference — see
    /// `crates/kernels/tests/vision.rs`.
    pub const EPS: f32 = 1e-6;

    /// The vision RoPE base, which `vision_config` also does not carry. 1e4,
    /// where the text side's is 1e7.
    pub const ROPE_THETA: f32 = 10_000.0;

    /// The side of the learned position grid, 48 here.
    pub fn grid_per_side(&self) -> usize {
        (self.position_embeddings as f64).sqrt() as usize
    }
}

/// The GatedDeltaNet dimensions, when a model has such blocks.
///
/// Kept as a separate struct rather than five `Option` fields on `Config` so
/// that "this model has linear attention" is one question rather than five that
/// could disagree.
#[derive(Debug, Clone, Copy)]
pub struct LinearAttnConfig {
    /// How many key heads the projection produces — fewer than the value heads.
    pub key_heads: usize,
    /// The width the recurrence actually runs at.
    pub value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    /// Depthwise convolution width; the carried window is one shorter.
    pub conv_kernel: usize,
    /// Whether the checkpoint stores V heads *tiled* rather than grouped by key
    /// head.
    ///
    /// False for Hugging Face, which stores `[G0_v0..v2, G1_v0..v2, ...]` so
    /// value head `h` belongs to key head `h / (heads / key_heads)` --
    /// `repeat_interleave`. True for a GGUF, because llama.cpp reorders to
    /// `[G0_v0, G1_v0, ..., G0_v1, ...]` and then the mapping is
    /// `h % key_heads`. The same permutation is applied to `in_proj_z`,
    /// `in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`, the V channels of `conv1d`
    /// and the columns of `out_proj`, so everything indexed by a value head
    /// agrees and only the q/k lookup has to know.
    ///
    /// Both readings run. Reading a GGUF as grouped produces grammatical,
    /// fluent, content-free text -- the prompt's own words and the commonest
    /// function words, with the answer absent from the top ten.
    pub v_heads_tiled: bool,
}

impl LinearAttnConfig {
    pub fn key_dim(&self) -> usize {
        self.key_heads * self.key_head_dim
    }

    pub fn value_dim(&self) -> usize {
        self.value_heads * self.value_head_dim
    }

    /// The packed `[q | k | v]` row's width, which is also the convolution's
    /// channel count.
    pub fn conv_channels(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    /// How many value heads share one key head.
    pub fn heads_per_key(&self) -> usize {
        self.value_heads / self.key_heads
    }
}

impl Config {
    /// How wide the attention block is inside, `n_heads · d_head`.
    ///
    /// Equal to `d_model` for every model up to Qwen3.8, which is why the two
    /// were the same variable for so long. Qwen3.8-27B runs 24 heads of 256
    /// against a 5120-wide residual, so `q` is 6144 columns and the output
    /// projection narrows back down — the residual stream and the attention
    /// interior are separate widths, and conflating them sizes half the
    /// attention buffers wrong.
    pub fn d_attn(&self) -> usize {
        self.n_heads * self.d_head
    }

    /// How wide `k` and `v` are, `n_kv_heads · d_head`.
    pub fn d_kv(&self) -> usize {
        self.n_kv_heads * self.d_head
    }

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

        // `attention.key_length` when the file states it, because on Qwen3.5 the
        // head is 256 wide and only 64 of it rotates -- so `rope.dimension_count`
        // is the *rotary* width there and reading it as the head width makes
        // every projection the wrong shape. Every other model this path loads
        // states neither and gets `d_model / n_heads`.
        let d_head = f
            .usize(&key("attention.key_length"))
            .or_else(|_| f.usize(&key("rope.dimension_count")))
            .unwrap_or_else(|_| d_model / n_heads.max(1));

        if n_heads == 0 || n_kv_heads == 0 {
            bail!("head counts must be non-zero");
        }
        if !n_heads.is_multiple_of(n_kv_heads) {
            bail!("{n_heads} query heads do not divide into {n_kv_heads} kv heads");
        }
        // The recurrence's shape, when the file states one. Its presence is
        // also what says the attention interior may be wider than the residual
        // and that the attention blocks carry an output gate -- both true of
        // Qwen3.5 and of nothing else this loader reads.
        let linear_attn = match f.usize(&key("ssm.group_count")) {
            Ok(key_heads) => {
                let dim = f.usize(&key("ssm.state_size"))?;
                Some(LinearAttnConfig {
                    key_heads,
                    value_heads: f.usize(&key("ssm.time_step_rank"))?,
                    key_head_dim: dim,
                    value_head_dim: dim,
                    conv_kernel: f.usize(&key("ssm.conv_kernel"))?,
                    v_heads_tiled: true,
                })
            }
            Err(_) => None,
        };

        // `d_attn` is `n_heads * d_head` and equals `d_model` on every model but
        // this one: Qwen3.5 has 24 heads of 256 over a 5120-wide residual, so
        // its `o_proj` is `[6144, 5120]` rather than square. The guard stays for
        // everything else, because there it catches a misread head width.
        if d_head * n_heads != d_model && linear_attn.is_none() {
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
            // `rope.dimension_count` is the rotary width, which coincides with
            // `d_head` on every model but Qwen3.5 -- 64 of 256 there, with the
            // frequency exponent normalized by *this* width rather than by the
            // head's. Reading it as the head width is the mistake the `d_head`
            // expression above avoids; reading the head width as the rotary one
            // is this line's.
            rotary_dim: f
                .usize(&key("rope.dimension_count"))
                .unwrap_or(d_head),
            rope_freq_scale: f
                .f32(&key("rope.scaling.factor"))
                .map(|s| if s > 0.0 { 1.0 / s } else { 1.0 })
                .unwrap_or(1.0),
            tied_embeddings: f.get_tensor("output.weight").is_none(),
            interleaved_rope: INTERLEAVED_ROPE.contains(&arch.as_str()),
            // The output gate follows from the mixer, not from a key: every
            // Qwen3.5 attention block has one, which is why its `attn_q` is
            // twice as wide as its head count implies. The loader still checks
            // the projection's actual width against this rather than trusting
            // it -- a config cannot silently change the arithmetic.
            attn_output_gate: linear_attn.is_some(),
            linear_attn,
            // Likewise for the MTP head: no GGUF conversion carries one, and
            // guessing a depth would build a drafter out of tensors that are
            // not there.
            mtp_layers: 0,
            // No GGUF in the wild carries a Qwen3.5 vision tower, and if one
            // did its dimensions would need names in the GGUF metadata
            // vocabulary rather than a `vision_config` object.
            vision: None,
            mtp_dedicated_embeddings: false,
            // GGUF states sparsity in its own metadata vocabulary, and reading
            // it here means a MoE GGUF fails at the weights with "no
            // ffn_gate_exps" rather than loading as a dense model and answering
            // from a third of its parameters. The expert weights themselves are
            // not read yet — see `docs/superpowers/specs/2026-08-24-moe-design.md`.
            moe: match (
                f.usize(&key("expert_count")).ok(),
                f.usize(&key("expert_used_count")).ok(),
            ) {
                (Some(n_experts), Some(n_active)) if n_experts > 0 => Some(MoeConfig {
                    n_experts,
                    n_active,
                    // GGUF names the expert width separately; falling back to
                    // the dense one is right for the models that omit it,
                    // because there the two are equal.
                    d_ff_expert: f.usize(&key("expert_feed_forward_length")).unwrap_or(d_ff),
                    norm_topk_prob: true,
                    sparse_step: 1,
                    dense_layers: Vec::new(),
                }),
                _ => None,
            },
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
        // A multimodal config describes two models. The outer object names the
        // wrapper and carries the vision tower and the placeholder token ids;
        // the language model's own dimensions sit in `text_config`. Reading the
        // outer object for them finds nothing, so prefer the inner one where it
        // exists and fall back for the single-model configs that have no such
        // nesting.
        let dims = if j["text_config"].is_object() {
            &j["text_config"]
        } else {
            j
        };
        let u = |k: &str| -> Result<usize> {
            dims[k]
                .as_u64()
                .map(|v| v as usize)
                .with_context(|| format!("config.json has no integer `{k}`"))
        };
        // `model_type` stays on the outer object: it identifies the whole model,
        // and for a multimodal one the inner type names only the text half.
        let arch = j["model_type"].as_str().unwrap_or("llama").to_string();
        if !SUPPORTED.contains(&arch.as_str()) {
            tracing::warn!(
                arch = %arch,
                "architecture is untested; expecting a llama-style block layout"
            );
        }
        let (n_heads, d_model) = (u("num_attention_heads")?, u("hidden_size")?);
        let n_kv_heads = dims["num_key_value_heads"]
            .as_u64()
            .map_or(n_heads, |v| v as usize);
        anyhow::ensure!(n_heads > 0 && n_kv_heads > 0, "head counts must be non-zero");
        anyhow::ensure!(
            n_heads.is_multiple_of(n_kv_heads),
            "{n_heads} query heads do not divide into {n_kv_heads} kv heads"
        );
        let d_head = dims["head_dim"]
            .as_u64()
            .map_or(d_model / n_heads, |v| v as usize);
        // `d_head * n_heads` used to have to equal `d_model`, and for every
        // model tuili had loaded before Qwen3.8 it did. Qwen3.8 breaks it: 24
        // heads of 256 is 6144 against a 5120-wide residual, so the attention
        // block widens on the way in and narrows on the way out. The forward
        // pass now carries `d_attn()` separately from `d_model`, so there is
        // nothing left to check here — but note that the two are still equal on
        // every other model, which means the width separation has no regression
        // test of its own beyond "the old models still speak".
        anyhow::ensure!(
            d_head.is_multiple_of(2),
            "d_head {d_head} must be even for rotary embeddings"
        );

        // `rope_theta` and `partial_rotary_factor` live in
        // `text_config.rope_parameters`, one level below the dimensions.
        // Reading them off `dims` does not fail — it finds nothing and falls
        // back to the 10000 default, which is a base 1000x too small on this
        // checkpoint. That does not break anything nearby: the low-frequency
        // dimensions are the ones that carry long distances, so the model keeps
        // answering local questions correctly and loses retrieval across a long
        // context. Nothing points at the rope table.
        //
        // The older flat spelling is still accepted, because that is where
        // every checkpoint before this one put it and there is no announcement
        // of which layout an exporter used.
        let rope = &dims["rope_parameters"];
        let rope_theta = rope["rope_theta"]
            .as_f64()
            .or_else(|| dims["rope_theta"].as_f64())
            .unwrap_or(10_000.0) as f32;

        // `partial_rotary_factor` genuinely appears in both places on this
        // checkpoint, so either spelling has to be read.
        let partial = rope["partial_rotary_factor"]
            .as_f64()
            .or_else(|| dims["partial_rotary_factor"].as_f64());
        // `int(head_dim * partial_rotary_factor)` — truncation, matching the
        // reference's `int()`. Absent means the whole head rotates, which is
        // every model before this one.
        let rotary_dim = match partial {
            Some(f) => (d_head as f64 * f) as usize,
            None => d_head,
        };
        anyhow::ensure!(
            rotary_dim >= 2 && rotary_dim <= d_head && rotary_dim.is_multiple_of(2),
            "partial_rotary_factor {partial:?} gives a rotary width of \
             {rotary_dim} out of d_head {d_head}; it must be even and in 2..=d_head"
        );
        if rotary_dim != d_head {
            tracing::info!(
                rotary_dim,
                d_head,
                "partial rotary embeddings: the tail of each head is not rotated"
            );
        }

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
            rms_eps: dims["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            rope_theta,
            rotary_dim,
            // Llama 3's scaling is per-dimension rather than a single factor;
            // it arrives through `rope_freq_factors` instead. A plain linear
            // `factor` would go here.
            rope_freq_scale: 1.0,
            // Tying is a property of the whole model, so a multimodal config
            // states it once on the outer object; a single-model config has only
            // that level anyway. Fall back to the inner one so a checkpoint that
            // repeats it there is still read correctly.
            tied_embeddings: j["tie_word_embeddings"]
                .as_bool()
                .or_else(|| dims["tie_word_embeddings"].as_bool())
                .unwrap_or(false),
            interleaved_rope: false,
            // The MTP head's depth and whether it has its own embedding.
            // `text_config` on this checkpoint, but read either level: these
            // describe the text model, and an exporter that flattened the
            // config would put them where every other dimension went.
            mtp_layers: dims["mtp_num_hidden_layers"]
                .as_u64()
                .or_else(|| j["mtp_num_hidden_layers"].as_u64())
                .unwrap_or(0) as usize,
            mtp_dedicated_embeddings: dims["mtp_use_dedicated_embeddings"]
                .as_bool()
                .or_else(|| j["mtp_use_dedicated_embeddings"].as_bool())
                .unwrap_or(false),
            // Present only when the checkpoint says some layers are linear. All
            // five dimensions come from one place or none of them do: a
            // partially-specified linear-attention config would produce a
            // plausible width for one of them and a wrong one for another.
            attn_output_gate: dims["attn_output_gate"].as_bool().unwrap_or(false),
            linear_attn: match (
                dims["linear_num_key_heads"].as_u64(),
                dims["linear_num_value_heads"].as_u64(),
                dims["linear_key_head_dim"].as_u64(),
                dims["linear_value_head_dim"].as_u64(),
                dims["linear_conv_kernel_dim"].as_u64(),
            ) {
                (Some(kh), Some(vh), Some(kd), Some(vd), Some(ck)) => {
                    let (kh, vh) = (kh as usize, vh as usize);
                    anyhow::ensure!(
                        kh > 0 && vh > 0 && vh.is_multiple_of(kh),
                        "{vh} linear value heads do not divide into {kh} key \
                         heads, so the repeat_interleave expansion the \
                         recurrence needs is not defined"
                    );
                    anyhow::ensure!(
                        ck >= 2,
                        "a depthwise convolution of width {ck} has no window to \
                         carry between steps"
                    );
                    Some(LinearAttnConfig {
                        key_heads: kh,
                        value_heads: vh,
                        key_head_dim: kd as usize,
                        value_head_dim: vd as usize,
                        conv_kernel: ck as usize,
                        // Hugging Face stores V heads grouped by key head; only a GGUF
                    // is tiled. See the field's own note.
                    v_heads_tiled: false,
                })
                }
                (None, None, None, None, None) => None,
                _ => anyhow::bail!(
                    "this config names some of the linear-attention dimensions \
                     and not others; all five of linear_num_key_heads, \
                     linear_num_value_heads, linear_key_head_dim, \
                     linear_value_head_dim and linear_conv_kernel_dim are needed \
                     to size the recurrence"
                ),
            },
            vision: Self::vision_from_json(j, d_model)?,
            moe: Self::moe_from_json(dims)?,
        })
    }

    /// The sparsity fields, when the checkpoint has them.
    ///
    /// All three sizing fields are required together. Defaulting the missing
    /// half of a pair is the failure mode worth avoiding: `num_experts` with no
    /// `num_experts_per_tok` would route to every expert, which runs, answers,
    /// and does 16x the arithmetic the model was trained for.
    fn moe_from_json(dims: &serde_json::Value) -> Result<Option<MoeConfig>> {
        let u = |k: &str| dims[k].as_u64().map(|v| v as usize);
        let (n_experts, n_active, d_ff_expert) = (
            u("num_experts").or_else(|| u("n_routed_experts")),
            u("num_experts_per_tok"),
            u("moe_intermediate_size"),
        );
        match (n_experts, n_active, d_ff_expert) {
            (None, None, None) => Ok(None),
            (Some(n_experts), Some(n_active), Some(d_ff_expert)) => {
                anyhow::ensure!(
                    n_experts > 0 && n_active > 0,
                    "a mixture of {n_experts} experts routing to {n_active} of them is empty"
                );
                anyhow::ensure!(
                    n_active <= n_experts,
                    "this config routes each token to {n_active} experts out of {n_experts}"
                );
                anyhow::ensure!(
                    d_ff_expert > 0,
                    "moe_intermediate_size must be non-zero"
                );
                let dense_layers = dims["mlp_only_layers"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect())
                    .unwrap_or_default();
                Ok(Some(MoeConfig {
                    n_experts,
                    n_active,
                    d_ff_expert,
                    // Qwen3-MoE ships this true. Absent, the reference's own
                    // default is true as well, so that is what an unstated
                    // config means.
                    norm_topk_prob: dims["norm_topk_prob"].as_bool().unwrap_or(true),
                    sparse_step: u("decoder_sparse_step").unwrap_or(1),
                    dense_layers,
                }))
            }
            _ => anyhow::bail!(
                "this config names some of the mixture-of-experts dimensions and \
                 not others: num_experts={n_experts:?}, \
                 num_experts_per_tok={n_active:?}, \
                 moe_intermediate_size={d_ff_expert:?}. All three are needed to \
                 size a sparse FFN"
            ),
        }
    }

    /// `vision_config`, when there is one.
    ///
    /// Every field is required once the object exists. A vision tower with a
    /// guessed depth or hidden size is not a degraded tower, it is 333 tensors
    /// that will not fit the buffers — and the failure would surface as a shape
    /// mismatch on some block in the middle rather than as the missing config
    /// key it is.
    fn vision_from_json(j: &serde_json::Value, d_model: usize) -> Result<Option<VisionConfig>> {
        let v = &j["vision_config"];
        if !v.is_object() {
            return Ok(None);
        }
        let u = |k: &str| -> Result<usize> {
            v[k].as_u64()
                .map(|n| n as usize)
                .with_context(|| format!("vision_config is missing {k}"))
        };
        // The placeholder ids live on the *outer* config, not in
        // `vision_config`: they are language-model vocabulary, and the tower
        // never sees them. Required, because a prompt cannot reserve room for
        // vision output without them and defaulting to another model's ids would
        // splice features over ordinary text.
        let tok = |k: &str| -> Result<u32> {
            j[k].as_u64()
                .map(|n| n as u32)
                .with_context(|| format!("this config has a vision tower but no {k}"))
        };
        let cfg = VisionConfig {
            depth: u("depth")?,
            hidden: u("hidden_size")?,
            heads: u("num_heads")?,
            intermediate: u("intermediate_size")?,
            out_hidden: u("out_hidden_size")?,
            in_channels: u("in_channels")?,
            patch: u("patch_size")?,
            temporal_patch: u("temporal_patch_size")?,
            merge: u("spatial_merge_size")?,
            position_embeddings: u("num_position_embeddings")?,
            image_token: tok("image_token_id")?,
            video_token: tok("video_token_id")?,
        };
        anyhow::ensure!(
            cfg.hidden.is_multiple_of(cfg.heads) && cfg.heads > 0,
            "{} vision heads do not divide a hidden size of {}",
            cfg.heads,
            cfg.hidden
        );
        let side = cfg.grid_per_side();
        anyhow::ensure!(
            side * side == cfg.position_embeddings,
            "num_position_embeddings {} is not a square, so the learned grid has \
             no side length and the resampling has nothing to interpolate over",
            cfg.position_embeddings
        );
        // The one cross-tower constraint, and the reason `out_hidden_size` is
        // read rather than assumed: the merger's output is spliced into the
        // text model's embedding rows, so a mismatch is not a shape error
        // somewhere later, it is features written into a row of the wrong width.
        anyhow::ensure!(
            cfg.out_hidden == d_model,
            "the vision merger projects to {} and the text model is {d_model} \
             wide; its output is spliced directly into the embedding rows",
            cfg.out_hidden
        );
        anyhow::ensure!(
            cfg.merge >= 1 && cfg.patch >= 1 && cfg.temporal_patch >= 1,
            "vision patch geometry has a zero dimension: patch {}, temporal {}, \
             merge {}",
            cfg.patch,
            cfg.temporal_patch,
            cfg.merge
        );
        anyhow::ensure!(
            cfg.image_token != cfg.video_token,
            "image_token_id and video_token_id are both {}; the splice could not \
             tell a frame from a still",
            cfg.image_token
        );
        tracing::info!(
            depth = cfg.depth,
            hidden = cfg.hidden,
            out_hidden = cfg.out_hidden,
            grid = side,
            image_token = cfg.image_token,
            "vision tower in the config"
        );
        Ok(Some(cfg))
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
    ///
    /// The returned length is `rotary_dim / 2`, one per rotated pair, which is
    /// `d_head / 2` on every model that rotates the whole head. The exponent is
    /// normalized by `rotary_dim` for the same reason the table is: a partial
    /// schedule is a compressed table, not a prefix of the wide one.
    pub fn rope_freq_factors(&self, j: &serde_json::Value) -> Vec<f32> {
        let half = self.rotary_dim / 2;
        // Same nesting as `from_hf`: a multimodal config puts the language
        // model's rope settings in `text_config`.
        let s = if j["text_config"].is_object() {
            &j["text_config"]["rope_scaling"]
        } else {
            &j["rope_scaling"]
        };
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
                let inv_freq = self.rope_theta.powf(-2.0 * i as f32 / self.rotary_dim as f32);
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
            "{} ({}) {} layers, d_model {}, {} heads / {} kv, d_head {}, rotary {}, ffn {}, vocab {}, ctx {}, theta {}, rope {}",
            self.name,
            self.arch,
            self.n_layers,
            self.d_model,
            self.n_heads,
            self.n_kv_heads,
            self.d_head,
            self.rotary_dim,
            self.d_ff,
            self.vocab_size,
            self.context_length,
            self.rope_theta,
            if self.interleaved_rope {
                "interleaved"
            } else {
                "neox"
            },
        )
    }
}
