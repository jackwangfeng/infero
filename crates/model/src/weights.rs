//! Uploading GGUF tensors, to the device or to pinned host memory.
//!
//! A layer is either **resident** — its matrices live in VRAM for the process
//! lifetime — or **offloaded**, in which case its seven big matrices are packed
//! into one page-locked host buffer and DMA'd into a device staging slot just
//! before the layer runs. Compute never leaves the GPU either way; offloading
//! trades PCIe bandwidth for VRAM, not GPU work for CPU work.
//!
//! Norms and biases stay resident regardless. They are a few kilobytes per
//! layer, and streaming them would add descriptors to the transfer without
//! saving anything worth measuring.

use anyhow::{Context, Result};
use tuili_gpu::{Buf, View};
use tuili_gpu::PinnedHostSlice;
use tuili_gpu::Device;
use tuili_gguf::{GgmlType, Gguf, TensorInfo};
use tuili_kernels::WeightType;

use crate::config::Config;

/// Matrices inside a layer blob start on this boundary, which satisfies every
/// ggml block type's alignment and keeps each sub-copy DMA-friendly.
const BLOB_ALIGN: usize = 256;

/// Where a matrix's bytes live.
enum Storage {
    /// In VRAM, for the process lifetime.
    Device(Buf<u8>),
    /// In the owning layer's host blob, at this byte offset. The same offset
    /// addresses it inside the staging buffer once the layer is transferred.
    Streamed { offset: usize },
}

/// A 2-D weight matrix, still in its GGUF block encoding.
pub struct Matrix {
    pub ty: WeightType,
    /// Elements per row (ggml `ne0`), the contraction dimension.
    pub k: usize,
    /// Number of rows (ggml `ne1`), the output dimension.
    pub n: usize,
    pub n_bytes: usize,
    storage: Storage,
}

impl Matrix {
    pub fn elements(&self) -> usize {
        self.k * self.n
    }

    pub fn is_resident(&self) -> bool {
        matches!(self.storage, Storage::Device(_))
    }

    /// A device view of this matrix.
    ///
    /// `stage` must be the staging buffer currently holding this matrix's
    /// layer, and is unused for a resident matrix.
    pub fn view<'a>(&'a self, stage: Option<&'a Buf<u8>>) -> Result<View<'a, u8>> {
        match &self.storage {
            Storage::Device(d) => Ok(d.as_view()),
            Storage::Streamed { offset } => {
                let stage =
                    stage.context("an offloaded matrix was used without its layer being staged")?;
                anyhow::ensure!(
                    offset + self.n_bytes <= stage.len(),
                    "staging buffer holds {} bytes, matrix wants {}..{}",
                    stage.len(),
                    offset,
                    offset + self.n_bytes
                );
                Ok(stage.slice(*offset..offset + self.n_bytes))
            }
        }
    }
}

/// A 1-D parameter — norm gains and biases — always held as f32 on the device.
pub type Vector = Buf<f32>;

impl Matrix {
    /// Upload `[n, k]` f16 values, row-major, as a resident matrix.
    ///
    /// For a caller that has the numbers rather than a checkpoint: the reference
    /// weights a capture carries, chiefly. The loaders build their matrices
    /// through the private paths above and do not go through this.
    pub fn upload_f16(dev: &Device, halves: &[half::f16], k: usize, n: usize) -> Result<Self> {
        anyhow::ensure!(
            halves.len() == k * n,
            "{} halves for a [{n}, {k}] matrix",
            halves.len()
        );
        // Safety: f16 is a transparent u16, so these are already the
        // little-endian halves the device expects, and the view does not outlive
        // `halves`.
        let raw =
            unsafe { std::slice::from_raw_parts(halves.as_ptr() as *const u8, halves.len() * 2) };
        Ok(Self {
            ty: WeightType::F16,
            k,
            n,
            n_bytes: raw.len(),
            storage: Storage::Device(dev.stream().clone_htod(raw)?),
        })
    }
}

/// One offloaded layer's matrices, packed contiguously in page-locked memory.
///
/// One blob per layer means one DMA per layer: the transfer the prefetch has to
/// hide is a single large contiguous copy rather than seven scattered ones.
pub struct LayerBlob {
    host: PinnedHostSlice<u8>,
    pub bytes: usize,
}

impl LayerBlob {
    pub fn host(&self) -> &PinnedHostSlice<u8> {
        &self.host
    }
}

/// The softmax-attention half of a block.
///
/// Its own struct because Qwen3.5 has blocks that do not have one: 48 of its 64
/// layers mix with a recurrence instead. Before that every model tuili loaded
/// had exactly one kind of block, so these fields sat directly on `Layer` and
/// the forward pass could reach them unconditionally.
pub struct AttnWeights {
    pub wq: Matrix,
    pub wk: Matrix,
    pub wv: Matrix,
    pub wo: Matrix,
    pub bq: Option<Vector>,
    pub bk: Option<Vector>,
    pub bv: Option<Vector>,
    pub bo: Option<Vector>,
    /// Qwen3 normalizes each head of `q` and `k` with its own learned
    /// `[d_head]` weight, before the rotary. Absent on llama and qwen2, where
    /// the attention biases play the role these replaced.
    pub q_norm: Option<Vector>,
    pub k_norm: Option<Vector>,
    /// `q`, `k` and `v` stacked along `n`, under `TUILI_FUSE_FFN`. One matmul
    /// and a scatter instead of three; see `stacked` in `load_awq`.
    pub w_qkv: Option<Matrix>,
    /// True when `wq` produces `2 * d_attn` columns: a query and a gate
    /// interleaved per head, which Qwen3.5's attention blocks carry and
    /// nothing before them did.
    pub output_gate: bool,
}

/// The GatedDeltaNet half of a block.
///
/// Field names follow the checkpoint's tensor names rather than being
/// translated, because the mapping is the thing most likely to be got wrong and
/// a reader should be able to check it against `notes/qwen3.5-architecture.md`
/// without a glossary.
pub struct GdnWeights {
    /// `[d_model, 2 * key_dim + value_dim]` — q, k and v in one projection.
    pub in_proj_qkv: Matrix,
    /// `[d_model, value_dim]` — the output gate, value-shaped.
    pub in_proj_z: Matrix,
    /// `[d_model, value_heads]` — the per-head decay input.
    pub in_proj_a: Matrix,
    /// `[d_model, value_heads]` — the per-head write strength.
    pub in_proj_b: Matrix,
    /// `in_proj_a` and `in_proj_b` stacked along the output, when they are the
    /// same dense type. One launch instead of two on matrices whose cost is all
    /// launch. `a` occupies columns `[0, value_heads)` of each row and `b` the
    /// rest, which is why the gate kernel takes a stride.
    pub in_proj_ba: Option<Matrix>,
    /// `[conv_channels, conv_k]`, depthwise, no bias.
    pub conv1d: Vector,
    pub a_log: Vector,
    pub dt_bias: Vector,
    /// `[value_head_dim]`, the gated RMSNorm's gain.
    pub norm: Vector,
    /// `[value_dim, d_model]`.
    pub out_proj: Matrix,
}

pub struct Layer {
    pub attn_norm: Vector,
    /// The mixer. Exactly one of these is `Some`; which one is decided by the
    /// checkpoint's `layer_types`, not by a stride, because Qwen3.5 states the
    /// pattern explicitly and a future model need not repeat every fourth.
    pub attn: Option<AttnWeights>,
    pub gdn: Option<GdnWeights>,
    pub ffn_norm: Vector,
    pub w_gate: Matrix,
    pub w_up: Matrix,
    pub w_down: Matrix,
    /// `gate` and `up` stacked along `n`, under `TUILI_FUSE_FFN`. One matmul
    /// instead of two; see `stacked` in `load_awq`.
    pub w_gate_up: Option<Matrix>,
    /// Present when this layer's matrices are streamed rather than resident.
    pub blob: Option<LayerBlob>,
}

impl Layer {
    pub fn is_offloaded(&self) -> bool {
        self.blob.is_some()
    }

    /// The attention half, for a layer the caller has already established has
    /// one.
    ///
    /// Panics rather than returning an error: reaching here on a
    /// linear-attention layer means the block dispatch is wrong, which is a bug
    /// in this file's caller and not a condition to recover from. The message
    /// names the layer kind so the dispatch is the first place to look.
    pub fn attn(&self) -> &AttnWeights {
        self.attn.as_ref().expect(
            "asked for the attention weights of a linear-attention layer; \
             the block dispatch did not consult Layer::is_linear",
        )
    }

    /// True for a GatedDeltaNet block.
    pub fn is_linear(&self) -> bool {
        self.gdn.is_some()
    }
}

/// The multi-token-prediction head, when a checkpoint carries one.
///
/// Shaped like a [`Layer`] plus four tensors of glue, which is what it is: one
/// full-attention decoder block, the same kind as the text model's layers 3, 7,
/// …, 63, wrapped in `fc` and three norms. See `crates/model/src/qwen35_mtp.rs`
/// for the host reference this is the device counterpart of, and
/// `notes/qwen3.5-mtp.md` for why each of these is the tensor it is.
///
/// Deliberately *not* holding an embedding or a vocabulary projection: the head
/// borrows the text model's. That is what `mtp_use_dedicated_embeddings = false`
/// means, and it is what the checkpoint shows by shipping no `mtp.embed_tokens`
/// and no `mtp.lm_head`. [`load_mtp`] checks both statements and refuses when
/// they disagree, rather than trusting the config over the tensors or the other
/// way round.
pub struct MtpWeights {
    /// `[2 * d_model, d_model]` — `k = 2 * d_model`, so the low half of every
    /// row multiplies the **embedding** and the high half the hidden state.
    pub fc: Matrix,
    /// Applies to the token embedding.
    pub pre_fc_norm_embedding: Vector,
    /// Applies to the text model's final hidden state.
    pub pre_fc_norm_hidden: Vector,
    /// The head's own final norm — a different tensor from the text model's
    /// `model.language_model.norm`.
    pub norm: Vector,
    /// The one decoder layer, in the same shape the text model's blocks use so
    /// that the forward pass can be read against `Model::attention`.
    pub layer: Layer,
    pub device_bytes: usize,
}

pub struct Weights {
    pub token_embd: Matrix,
    pub layers: Vec<Layer>,
    pub output_norm: Vector,
    /// Absent when the model ties the output projection to the embeddings.
    pub output: Option<Matrix>,
    /// The same matrix in [`WeightType::Q8_0S`], for the batched path.
    ///
    /// Only the AWQ loader builds it, because only there does tuili choose the
    /// vocab projection's layout. Held *as well as* `output`: the batch-1
    /// mat-vec reads the packed form, and teaching it the split one is a
    /// separate change from proving the split one is faster. 532 MiB on an 8B
    /// model, so it is gated on there being room.
    pub output_split: Option<Matrix>,
    /// Per-dimension RoPE frequency divisors, `d_head / 2` of them. All ones
    /// unless the file carries `rope_freqs.weight`.
    pub rope_freqs: Vector,
    /// Weight bytes held in VRAM.
    pub device_bytes: usize,
    /// Weight bytes held in page-locked host memory.
    pub host_bytes: usize,
    /// Largest single layer blob, which sizes the staging buffers.
    pub max_blob_bytes: usize,
}

impl Weights {
    /// Load with the first `n_gpu_layers` blocks resident and the rest
    /// offloaded. Embeddings, the output projection and all norms stay
    /// resident: the vocab projection is touched once per token and the norms
    /// are negligible.
    pub fn load(dev: &Device, f: &Gguf, cfg: &Config, n_gpu_layers: usize) -> Result<Self> {
        let started = std::time::Instant::now();
        let n_gpu_layers = n_gpu_layers.min(cfg.n_layers);
        let mut device_bytes = 0usize;
        let mut host_bytes = 0usize;
        let mut max_blob_bytes = 0usize;

        let token_embd = upload_matrix(dev, f, "token_embd.weight", &mut device_bytes)?;
        let output_norm = upload_vector(dev, f, "output_norm.weight", &mut device_bytes)?;
        let output = if cfg.tied_embeddings {
            None
        } else {
            Some(upload_matrix(dev, f, "output.weight", &mut device_bytes)?)
        };

        // Llama 3.1 ships these precomputed; everything else wants no scaling.
        let rope_freqs = match f.get_tensor("rope_freqs.weight") {
            Some(info) if info.n_elements == cfg.d_head / 2 => {
                tracing::info!(dims = info.n_elements, "using rope frequency scaling");
                upload_vector(dev, f, "rope_freqs.weight", &mut device_bytes)?
            }
            Some(info) => {
                anyhow::bail!(
                    "rope_freqs.weight has {} entries, expected d_head/2 = {}",
                    info.n_elements,
                    cfg.d_head / 2
                );
            }
            None => dev.stream().clone_htod(&vec![1.0f32; cfg.d_head / 2])?,
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let t = |s: &str| format!("blk.{i}.{s}");
            // Which mixer this block has is read off the file rather than
            // inferred from an interval: Qwen3.5 states an explicit 64-element
            // `layer_types`, and a block that carries `ssm_a` is a linear one
            // whatever a stride would say.
            let is_linear = f.get_tensor(&t("ssm_a")).is_some();
            let names: Vec<String> = if is_linear {
                vec![
                    t("attn_qkv.weight"),
                    t("attn_gate.weight"),
                    t("ssm_alpha.weight"),
                    t("ssm_beta.weight"),
                    t("ssm_out.weight"),
                    t("ffn_gate.weight"),
                    t("ffn_up.weight"),
                    t("ffn_down.weight"),
                ]
            } else {
                vec![
                    t("attn_q.weight"),
                    t("attn_k.weight"),
                    t("attn_v.weight"),
                    t("attn_output.weight"),
                    t("ffn_gate.weight"),
                    t("ffn_up.weight"),
                    t("ffn_down.weight"),
                ]
            };
            // llama.cpp names the pre-FFN norm `ffn_norm` for a llama-style
            // block and `post_attention_norm` for Qwen3.5. The same norm in the
            // same place; only the spelling moved.
            let ffn_norm_name = if f.get_tensor(&t("ffn_norm.weight")).is_some() {
                t("ffn_norm.weight")
            } else {
                t("post_attention_norm.weight")
            };

            let (matrices, blob) = if i < n_gpu_layers {
                let mut m = Vec::with_capacity(names.len());
                for name in &names {
                    m.push(upload_matrix(dev, f, name, &mut device_bytes)?);
                }
                (m, None)
            } else {
                let (m, blob) = pack_layer(dev, f, &names)
                    .with_context(|| format!("packing layer {i} into host memory"))?;
                host_bytes += blob.bytes;
                max_blob_bytes = max_blob_bytes.max(blob.bytes);
                (m, Some(blob))
            };
            let mut matrices = matrices.into_iter();

            let (attn, gdn) = if is_linear {
                let qkv = matrices.next().unwrap();
                let z = matrices.next().unwrap();
                let a = matrices.next().unwrap();
                let b = matrices.next().unwrap();
                let out = matrices.next().unwrap();
                (
                    None,
                    Some(GdnWeights {
                        in_proj_qkv: qkv,
                        in_proj_z: z,
                        in_proj_a: a,
                        in_proj_b: b,
                        // Stacking `a` and `b` is a launch-count optimisation
                        // the safetensors path takes when they share a dense
                        // type. Not taken here: a GGUF stores them as two
                        // tensors and concatenating quantized blocks is not a
                        // reinterpretation.
                        in_proj_ba: None,
                        conv1d: upload_vector(
                            dev, f, &t("ssm_conv1d.weight"), &mut device_bytes)?,
                        a_log: upload_a_log(dev, f, &t("ssm_a"), &mut device_bytes)?,
                        dt_bias: upload_vector(
                            dev, f, &t("ssm_dt.bias"), &mut device_bytes)?,
                        // The one norm in this architecture that is a plain
                        // gain: `Qwen3_5RMSNormGated`. llama.cpp's converter
                        // adds one to every *other* norm weight and leaves this
                        // one alone, which is why nothing is undone here.
                        norm: upload_vector(
                            dev, f, &t("ssm_norm.weight"), &mut device_bytes)?,
                        out_proj: out,
                    }),
                )
            } else {
                (
                    Some(AttnWeights {
                        wq: matrices.next().unwrap(),
                        wk: matrices.next().unwrap(),
                        wv: matrices.next().unwrap(),
                        wo: matrices.next().unwrap(),
                        // Qwen2 carries QKV biases; Llama does not.
                        bq: upload_optional_vector(dev, f, &t("attn_q.bias"), &mut device_bytes)?,
                        bk: upload_optional_vector(dev, f, &t("attn_k.bias"), &mut device_bytes)?,
                        bv: upload_optional_vector(dev, f, &t("attn_v.bias"), &mut device_bytes)?,
                        bo: upload_optional_vector(
                            dev, f, &t("attn_output.bias"), &mut device_bytes)?,
                        // Qwen3's per-head q/k norms; llama.cpp names them this
                        // way.
                        q_norm: upload_optional_vector(
                            dev, f, &t("attn_q_norm.weight"), &mut device_bytes)?,
                        k_norm: upload_optional_vector(
                            dev, f, &t("attn_k_norm.weight"), &mut device_bytes)?,
                        w_qkv: None,
                        output_gate: cfg.attn_output_gate,
                    }),
                    None,
                )
            };

            layers.push(Layer {
                attn_norm: upload_vector(dev, f, &t("attn_norm.weight"), &mut device_bytes)?,
                attn,
                gdn,
                ffn_norm: upload_vector(dev, f, &ffn_norm_name, &mut device_bytes)?,
                w_gate: matrices.next().unwrap(),
                w_up: matrices.next().unwrap(),
                w_down: matrices.next().unwrap(),
                w_gate_up: None,
                blob,
            });
        }

        dev.synchronize()?;
        tracing::info!(
            gpu_layers = n_gpu_layers,
            offloaded = cfg.n_layers - n_gpu_layers,
            vram_mib = device_bytes / (1 << 20),
            host_mib = host_bytes / (1 << 20),
            ms = started.elapsed().as_millis(),
            "weights loaded"
        );

        let this = Self {
            token_embd,
            layers,
            output_norm,
            output,
            // A GGUF file's vocab projection comes in whatever the file chose;
            // the split layout is only for the one tuili quantizes itself.
            output_split: None,
            rope_freqs,
            device_bytes,
            host_bytes,
            max_blob_bytes,
        };
        this.check_shapes(cfg)?;
        Ok(this)
    }

    pub fn n_offloaded(&self) -> usize {
        self.layers.iter().filter(|l| l.is_offloaded()).count()
    }

    /// Catch a config/tensor mismatch here rather than as silent garbage
    /// several kernels later.
    fn check_shapes(&self, cfg: &Config) -> Result<()> {
        let d = cfg.d_model;
        // The attention block is not square once `n_heads * d_head` stops
        // equalling `d_model`: q widens the residual to `d_attn` and o narrows
        // it back. This check is the only place a width mistake is caught
        // against the actual tensor, so it has to know the difference.
        let da = cfg.d_attn();
        let kv_dim = cfg.d_kv();

        anyhow::ensure!(
            self.token_embd.k == d && self.token_embd.n == cfg.vocab_size,
            "token_embd is [{}, {}], expected [{d}, {}]",
            self.token_embd.k,
            self.token_embd.n,
            cfg.vocab_size
        );

        for (i, l) in self.layers.iter().enumerate() {
            let expect = |m: &Matrix, k: usize, n: usize, what: &str| -> Result<()> {
                anyhow::ensure!(
                    m.k == k && m.n == n,
                    "layer {i} {what} is [{}, {}], expected [{k}, {n}]",
                    m.k,
                    m.n
                );
                Ok(())
            };
            if let Some(g) = &l.gdn {
                // A GatedDeltaNet block. Its widths come from the linear
                // dimensions, not from the attention ones, and checking it
                // against `d_attn` would pass on some of them by coincidence.
                let la = cfg.linear_attn.context(
                    "a block has GatedDeltaNet weights but the config gives no \
                     linear-attention dimensions to check them against",
                )?;
                let (key_dim, val_dim) = (la.key_dim(), la.value_dim());
                expect(&g.in_proj_qkv, d, la.conv_channels(), "in_proj_qkv")?;
                expect(&g.in_proj_z, d, val_dim, "in_proj_z")?;
                expect(&g.in_proj_a, d, la.value_heads, "in_proj_a")?;
                expect(&g.in_proj_b, d, la.value_heads, "in_proj_b")?;
                expect(&g.out_proj, val_dim, d, "out_proj")?;
                // The 1-D parameters, whose lengths encode the head counts.
                for (v, want, what) in [
                    (&g.conv1d, la.conv_channels() * la.conv_kernel, "conv1d"),
                    (&g.a_log, la.value_heads, "A_log"),
                    (&g.dt_bias, la.value_heads, "dt_bias"),
                    (&g.norm, la.value_head_dim, "norm"),
                ] {
                    anyhow::ensure!(
                        v.len() == want,
                        "layer {i} {what} has {} elements, expected {want}",
                        v.len()
                    );
                }
                let _ = key_dim;
            } else {
                let a = l.attn();
                // A gated q projection is twice as wide: a query and its gate
                // interleaved per head.
                let q_cols = if a.output_gate { 2 * da } else { da };
                expect(&a.wq, d, q_cols, "attn_q")?;
                expect(&a.wk, d, kv_dim, "attn_k")?;
                expect(&a.wv, d, kv_dim, "attn_v")?;
                expect(&a.wo, da, d, "attn_output")?;
            }
            expect(&l.w_gate, d, cfg.d_ff, "ffn_gate")?;
            expect(&l.w_up, d, cfg.d_ff, "ffn_up")?;
            expect(&l.w_down, cfg.d_ff, d, "ffn_down")?;
        }
        Ok(())
    }

    /// The encoding most of the model is stored in, for reporting.
    pub fn dominant_type(&self) -> WeightType {
        let mut totals: std::collections::HashMap<WeightType, usize> = Default::default();
        for l in &self.layers {
            // A block's matrices depend on which mixer it has. Reaching for the
            // attention ones unconditionally is what the `attn()` accessor
            // panics about, and this loop runs over every layer.
            let mixer: Vec<&Matrix> = match (&l.attn, &l.gdn) {
                (Some(a), _) => vec![&a.wq, &a.wk, &a.wv, &a.wo],
                (_, Some(g)) => vec![
                    &g.in_proj_qkv,
                    &g.in_proj_z,
                    &g.in_proj_a,
                    &g.in_proj_b,
                    &g.out_proj,
                ],
                _ => vec![],
            };
            for m in mixer
                .into_iter()
                .chain([&l.w_gate, &l.w_up, &l.w_down])
            {
                *totals.entry(m.ty).or_default() += m.n_bytes;
            }
        }
        totals
            .into_iter()
            .max_by_key(|&(_, n)| n)
            .map(|(t, _)| t)
            .unwrap_or(self.token_embd.ty)
    }
}

/// One if this norm weight is stored as a delta from one, zero if it is a gain.
///
/// Qwen3.5 stores most of its norm weights as a *delta from one*:
/// `Qwen3_5RMSNorm` initializes to zeros and computes
/// `normalized * (1 + weight)`, where every other model tuili loads initializes
/// to ones and computes `weight * normalized`. Adding the one here, at load,
/// means every norm kernel stays as it is — the alternative was a variant of
/// `rms_norm` and `qk_norm` each.
///
/// The exception is `linear_attn.norm`, which is `Qwen3_5RMSNormGated` and does
/// use the plain form. Two conventions in one checkpoint.
///
/// Which form a tensor wants follows from the class that consumes it, not from
/// the tensor: the two populations overlap. An `input_layernorm` centred at
/// 0.036 would be annihilated by the plain form, but some trained `q_norm`
/// deltas exceed 0.5 and some gated gains fall below 1.5, so a mean-based guess
/// gets those wrong. Hence a name rule, not a data rule.
///
/// A whitelist, not a blacklist. This same loader path reads the attention
/// biases and the GatedDeltaNet's `A_log`, `dt_bias` and `conv1d.weight`, none
/// of which are norms; a blacklist that forgot one of those would add one to a
/// bias or to a decay exponent. A whitelist that forgets a norm leaves that
/// norm on the old convention, which is bad but confined.
///
/// The MTP head adds three names and no new rule. Its `input_layernorm`,
/// `post_attention_layernorm`, `q_norm` and `k_norm` are already matched by the
/// suffixes above — they are the same classes in the same decoder block — and
/// `notes/qwen3.5-mtp.md` measures all four of the head's own norms as the
/// offset form, so `pre_fc_norm_embedding`, `pre_fc_norm_hidden` and `mtp.norm`
/// are listed here by their exact names. `mtp.norm.weight` cannot be matched by
/// the `norm.weight` suffix for the same reason `model.norm.weight` is not:
/// `linear_attn.norm.weight` ends in it too and is the one gain in the model.
fn norm_offset(arch: &str, name: &str) -> f32 {
    const OFFSET_FORM: &[&str] = &[
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    ];
    const OFFSET_FORM_EXACT: &[&str] = &[
        // The final norm before the vocabulary projection.
        "model.norm.weight",
        "model.language_model.norm.weight",
        // The MTP head's three. Its fourth and fifth — the decoder layer's two
        // — are covered by the suffixes; see the note above.
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight",
    ];
    let is_offset_form = OFFSET_FORM.iter().any(|suffix| name.ends_with(suffix))
        || OFFSET_FORM_EXACT.contains(&name);
    if arch == "qwen3_5" && is_offset_form {
        1.0
    } else {
        0.0
    }
}

/// Load the multi-token-prediction head, if the checkpoint has one.
///
/// Returns `None` when neither the config nor the tensors mention a head, and
/// an error when only one of them does — a checkpoint that names
/// `mtp_num_hidden_layers` and ships no `mtp.*` would otherwise build a drafter
/// out of nothing, and one that ships the tensors under a config that does not
/// mention them means this loader is reading a layout it has not been shown.
///
/// The head's tensors live under **`mtp.`**, not `model.mtp.`; vLLM rewrites the
/// prefix to `model.` at load time, which is where a reader looking for
/// `model.mtp.layers.0` gets the idea. Everything else about the block is the
/// text model's full-attention layout, so it goes through the same code.
///
/// One thing in here is not uniform: `mtp.fc` is BF16 while the rest of the head
/// is FP8. The dispatch is on the tensor's own dtype rather than on a list of
/// exceptions, which agrees with the checkpoint's
/// `quantization_config.modules_to_not_convert` (it names `mtp.fc`) and with
/// vLLM's special case, and keeps working if a future export quantizes it.
pub fn load_mtp(
    dev: &Device,
    w: &tuili_safetensors::Shards,
    cfg: &Config,
) -> Result<Option<MtpWeights>> {
    let present = w.get("mtp.fc.weight").is_some();
    match (cfg.mtp_layers, present) {
        (0, false) => return Ok(None),
        (0, true) => anyhow::bail!(
            "this checkpoint ships `mtp.*` tensors but its config does not say \
             `mtp_num_hidden_layers`; the head's depth decides how the draft \
             loop indexes its layers and guessing it is not safe"
        ),
        (n, false) => anyhow::bail!(
            "config says mtp_num_hidden_layers = {n} but there is no \
             `mtp.fc.weight`; there is nothing to build the drafter out of"
        ),
        (n, true) => anyhow::ensure!(
            n == 1,
            "this loader builds a one-layer MTP head, the config says {n}. The \
             draft loop indexes `spec_step_idx % mtp_num_hidden_layers`, which \
             is only the identity at one layer"
        ),
    }
    anyhow::ensure!(
        !cfg.mtp_dedicated_embeddings,
        "config says the MTP head has dedicated embeddings; this loader has the \
         head share the text model's `embed_tokens` and `lm_head`, which is what \
         `mtp_use_dedicated_embeddings = false` means"
    );
    // The other half of the sharing claim, from the tensors rather than the
    // config. Absence is part of the spec here: a checkpoint that shipped these
    // would want them used, and using the text model's instead would be a
    // silently different drafter.
    for name in ["mtp.embed_tokens.weight", "mtp.lm_head.weight"] {
        anyhow::ensure!(
            w.get(name).is_none(),
            "the checkpoint carries {name}, so the head does not share the text \
             model's — but the config says it does. One of the two is being \
             misread"
        );
    }
    // And the head's layer is a full-attention block, which the checkpoint
    // settles without interpretation: it has q/k/v/o and no recurrence. This is
    // the single most consequential fact for scheduling — a drafter that ran a
    // GatedDeltaNet layer would advance recurrent state on every speculative
    // token — so it is checked here as well as in the tests.
    for absent in [
        "mtp.layers.0.linear_attn.in_proj_qkv.weight",
        "mtp.layers.0.linear_attn.conv1d.weight",
        "mtp.layers.0.linear_attn.A_log",
        "mtp.layers.0.linear_attn.dt_bias",
    ] {
        anyhow::ensure!(
            w.get(absent).is_none(),
            "the MTP head carries {absent}: its layer is not the full-attention \
             block this drafter assumes, and drafting would touch recurrent state"
        );
    }

    let started = std::time::Instant::now();
    let mut bytes = 0usize;
    let vector = |name: &str, total: &mut usize| -> Result<Vector> {
        let t = w.tensor(name)?;
        let mut v = t.to_f32()?;
        let off = norm_offset(&cfg.arch, name);
        if off != 0.0 {
            for x in v.iter_mut() {
                *x += off;
            }
        }
        *total += v.len() * 4;
        Ok(dev.stream().clone_htod(&v)?)
    };
    let projection = |name: &str, total: &mut usize| -> Result<Matrix> {
        let t = w.tensor(&format!("{name}.weight"))?;
        anyhow::ensure!(
            t.shape.len() == 2,
            "{name}.weight has shape {:?}, expected a matrix",
            t.shape
        );
        // Row-major `[out, in]`, as torch writes it, which is the layout the
        // f16 GEMM and the mat-vec both want: `k` is the contraction dimension
        // and one row of the weight is contiguous.
        let (n, k) = (t.shape[0], t.shape[1]);
        if t.dtype == tuili_safetensors::Dtype::F8E4M3 {
            // Kept as FP8, like the text model's projections, and for the same
            // two reasons. This used to dequantize to f16 here — the strategy
            // the text side abandoned — which doubled the head's bytes from
            // 405 MiB to 810 and sent it through `gemv_f16` instead of the FP8
            // mat-vec. A draft step reads the head once, so its bytes are its
            // cost: 5.12 ms measured against a 2.0 ms byte bound, and half of
            // those bytes did not need to exist.
            let scales_t = w
                .tensor(&format!("{name}.weight_scale_inv"))
                .with_context(|| {
                    format!("{name}.weight is FP8, which is meaningless without its block scales")
                })?;
            let scales = scales_t.to_f32()?;
            let want = tuili_kernels::fp8::scale_grid(k, n);
            anyhow::ensure!(
                scales.len() == want,
                "{name}'s scale grid has {} entries; an [{n}, {k}] matrix wants {want}",
                scales.len()
            );
            let mut bytes = Vec::with_capacity(tuili_kernels::fp8::fp8_bytes(k, n));
            bytes.extend_from_slice(&tuili_kernels::fp8::repack_rows(t.data, k, n)?);
            for v in &scales {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            *total += bytes.len();
            return Ok(Matrix {
                ty: WeightType::F8E4M3,
                k,
                n,
                n_bytes: bytes.len(),
                storage: Storage::Device(dev.stream().clone_htod(&bytes)?),
            });
        }
        // `mtp.fc` lands here — BF16 in this checkpoint. Not a special case in
        // the code, only in the checkpoint.
        let halves = t.to_f16()?;
        anyhow::ensure!(halves.len() == k * n, "{name}: {} halves for {k}x{n}", halves.len());
        // Safety: f16 is a transparent u16, so these are already the
        // little-endian halves the device expects, and the view does not outlive
        // `halves`.
        let raw = unsafe {
            std::slice::from_raw_parts(halves.as_ptr() as *const u8, halves.len() * 2)
        };
        *total += raw.len();
        Ok(Matrix {
            ty: WeightType::F16,
            k,
            n,
            n_bytes: raw.len(),
            storage: Storage::Device(dev.stream().clone_htod(raw)?),
        })
    };

    let l = "mtp.layers.0";
    let wq = projection(&format!("{l}.self_attn.q_proj"), &mut bytes)?;
    // Twice the attention width means q_proj carries a gate interleaved with
    // the query, per head. Read off the shape, like the text model's blocks,
    // because the shape is what the rest of the code has to agree with.
    let output_gate = wq.n == 2 * cfg.d_attn();
    anyhow::ensure!(
        wq.n == cfg.d_attn() || output_gate,
        "the MTP head's q_proj has {} columns; expected {} for a plain query or \
         {} for a query and its gate",
        wq.n,
        cfg.d_attn(),
        2 * cfg.d_attn()
    );
    let layer = Layer {
        attn_norm: vector(&format!("{l}.input_layernorm.weight"), &mut bytes)?,
        attn: Some(AttnWeights {
            wq,
            wk: projection(&format!("{l}.self_attn.k_proj"), &mut bytes)?,
            wv: projection(&format!("{l}.self_attn.v_proj"), &mut bytes)?,
            wo: projection(&format!("{l}.self_attn.o_proj"), &mut bytes)?,
            bq: None,
            bk: None,
            bv: None,
            bo: None,
            q_norm: Some(vector(&format!("{l}.self_attn.q_norm.weight"), &mut bytes)?),
            k_norm: Some(vector(&format!("{l}.self_attn.k_norm.weight"), &mut bytes)?),
            w_qkv: None,
            output_gate,
        }),
        gdn: None,
        ffn_norm: vector(&format!("{l}.post_attention_layernorm.weight"), &mut bytes)?,
        w_gate: projection(&format!("{l}.mlp.gate_proj"), &mut bytes)?,
        w_up: projection(&format!("{l}.mlp.up_proj"), &mut bytes)?,
        w_down: projection(&format!("{l}.mlp.down_proj"), &mut bytes)?,
        w_gate_up: None,
        blob: None,
    };

    let fc = projection("mtp.fc", &mut bytes)?;
    anyhow::ensure!(
        fc.k == 2 * cfg.d_model && fc.n == cfg.d_model,
        "mtp.fc is [{}, {}], expected [{}, {}] — it consumes the embedding and \
         the hidden state concatenated and produces one residual row",
        fc.n,
        fc.k,
        cfg.d_model,
        2 * cfg.d_model
    );
    let head = MtpWeights {
        fc,
        pre_fc_norm_embedding: vector("mtp.pre_fc_norm_embedding.weight", &mut bytes)?,
        pre_fc_norm_hidden: vector("mtp.pre_fc_norm_hidden.weight", &mut bytes)?,
        norm: vector("mtp.norm.weight", &mut bytes)?,
        layer,
        device_bytes: bytes,
    };
    tracing::info!(
        vram_mib = bytes >> 20,
        gated = output_gate,
        ms = started.elapsed().as_millis(),
        "mtp head loaded"
    );
    Ok(Some(head))
}

/// The tile an FP8 checkpoint's `weight_scale_inv` covers, from
/// `quantization_config.weight_block_size`.
const FP8_BLOCK: usize = 128;

/// Describe a tensor without moving its bytes anywhere.
fn describe(f: &Gguf, name: &str) -> Result<(WeightType, usize, usize, usize)> {
    let info = f.tensor(name)?;
    anyhow::ensure!(
        info.dims.len() == 2,
        "{name} has {} dimensions, expected 2",
        info.dims.len()
    );
    let ty = WeightType::from_ggml(info.ty).with_context(|| format!("tensor {name}"))?;
    let (k, n) = (info.dims[0] as usize, info.dims[1] as usize);
    anyhow::ensure!(
        k.is_multiple_of(ty.block_size()),
        "{name}: row length {k} is not a multiple of {}'s block size {}",
        ty,
        ty.block_size()
    );
    Ok((ty, k, n, info.n_bytes))
}

/// Load an AWQ checkpoint, repacking every quantized matrix on the way in.
///
/// Everything stays resident: an AWQ file has no offload story yet, and the
/// point of reading one is speed rather than fitting a model that does not.
///
/// Two things differ from the GGUF path beyond the tensor names. The
/// projections arrive in AWQ's transposed, column-packed layout and are
/// repacked to [`WeightType::Q4G128`] here, once, so the mat-vec sees the
/// output-major rows it wants. And the vocabulary projection arrives as `f16` —
/// 1.05 GB on an 8B model, a fifth of a decode step, which the float mat-vec
/// reads at 141 GB/s against the integer path's 366 — so it is quantized to
/// Q8_0 on the way in. Eight bits is not a meaningful loss for a projection
/// whose output is fed to an argmax over 128k logits.
pub fn load_awq(
    dev: &Device,
    w: &tuili_safetensors::Shards,
    cfg: &Config,
    freq_factors: &[f32],
) -> Result<Weights> {
    use tuili_kernels::awq::{AwqTensor, quantize_f16_to_q8_0};

    let started = std::time::Instant::now();
    let mut device_bytes = 0usize;

    let upload = |bytes: &[u8], ty: WeightType, k: usize, n: usize, total: &mut usize| -> Result<Matrix> {
        *total += bytes.len();
        Ok(Matrix {
            ty,
            k,
            n,
            n_bytes: bytes.len(),
            storage: Storage::Device(dev.stream().clone_htod(bytes)?),
        })
    };
    let arch = cfg.arch.clone();
    let norm_offset = move |name: &str| -> f32 { norm_offset(&arch, name) };
    let vector = |name: &str, total: &mut usize| -> Result<Vector> {
        let t = w.tensor(name)?;
        // `to_f32` rather than `as_f16`: Qwen3's AWQ export stores the norm
        // weights as BF16 even though the AWQ scales are F16, and an F16-only
        // read rejects the checkpoint on `model.norm.weight`.
        let mut v = t.to_f32()?;
        let off = norm_offset(name);
        if off != 0.0 {
            for x in v.iter_mut() {
                *x += off;
            }
        }
        *total += v.len() * 4;
        Ok(dev.stream().clone_htod(&v)?)
    };
    // The same, for a tensor a checkpoint may not carry. Qwen2 and Qwen3 put a
    // bias on q, k and v; Llama does not, and asking for one there is not an
    // error. The GGUF loader has always read these — the AWQ path passed `None`
    // and produced fluent-looking nonsense on a Qwen checkpoint, because a
    // missing bias is not a crash, just a wrong answer.
    let optional_vector = |name: &str, total: &mut usize| -> Result<Option<Vector>> {
        match w.tensor(name) {
            Ok(t) => {
                let mut v = t.to_f32()?;
                let off = norm_offset(name);
                if off != 0.0 {
                    for x in v.iter_mut() {
                        *x += off;
                    }
                }
                *total += v.len() * 4;
                Ok(Some(dev.stream().clone_htod(&v)?))
            }
            Err(_) => Ok(None),
        }
    };
    // A quantized projection's bytes, before they reach the device: AWQ's three
    // tensors in, one packed matrix out. Split from the upload so that
    // projections which are stacked into one matrix — see `fuse_ffn` below —
    // can be concatenated in the layout they will be read in.
    let projection_bytes = |prefix: &str| -> Result<(Vec<u8>, WeightType, usize, usize)> {
        // FP8 and plain-float exports name the matrix `{prefix}.weight`; AWQ
        // splits it into qweight/qzeros/scales. Check for the single tensor
        // first, because its absence is the cheap question.
        //
        // Note the transposed convention between the two. AWQ stores
        // `[in_features, out_features / 8]`, so `k` is dimension 0. Everything
        // else stores output-major `[out_features, in_features]`, so `k` is
        // dimension 1. Reading one with the other's convention gives a matrix of
        // plausible size and wrong meaning.
        if let Some(t) = w.get(&format!("{prefix}.weight")) {
            let (n, k) = (t.shape[0], t.shape[1]);
            if t.dtype == tuili_safetensors::Dtype::F8E4M3 {
                // Keep the FP8 bytes and carry the scale grid with them, rather
                // than expanding here. Expanding is correct and was the first
                // version; it doubles what a decode step has to read, and the
                // profiler put the resulting f16 GEMM at 75% of a step.
                //
                // One buffer: `n * k` quants then the grid as f32, which is the
                // layout `WeightType::F8E4M3` documents and both FP8 kernels
                // read. A `Matrix` stays a single allocation, so the offload
                // blob path and `Matrix::view` need no special case.
                let scales_t = w
                    .tensor(&format!("{prefix}.weight_scale_inv"))
                    .with_context(|| format!("{prefix} is FP8 but has no scale grid"))?;
                let scales = scales_t.to_f32()?;
                let want = tuili_kernels::fp8::scale_grid(k, n);
                anyhow::ensure!(
                    scales.len() == want,
                    "{prefix}'s scale grid has {} entries; an [{n}, {k}] matrix \
                     at block {} wants {want}",
                    scales.len(),
                    tuili_kernels::fp8::FP8_BLOCK,
                );
                let mut bytes = Vec::with_capacity(tuili_kernels::fp8::fp8_bytes(k, n));
                // Permuted, not copied: every FP8 kernel reads four interleaved
                // rows as one 16-byte load, which is what took the batched
                // mat-vec off a request-per-row-per-token. The permutation lives
                // in `fp8::repack_rows` so that this and `tests/fp8_matvec.rs`
                // cannot drift apart.
                bytes.extend_from_slice(&tuili_kernels::fp8::repack_rows(t.data, k, n)?);
                for v in &scales {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                debug_assert_eq!(bytes.len(), tuili_kernels::fp8::fp8_bytes(k, n));
                return Ok((bytes, WeightType::F8E4M3, k, n));
            }
            let halves: Vec<half::f16> = if false {
                // Block-scaled FP8. The scale grid is 128x128 and
                // `dequant_f8_to_f16` validates that the grid matches the
                // quants, which is the check that catches a transposed or
                // row-only index — neither of which fails on its own, they just
                // mis-scale every tile past the first.
                let scales = w
                    .tensor(&format!("{prefix}.weight_scale_inv"))
                    .with_context(|| format!("{prefix} is FP8 but has no scale grid"))?;
                t.dequant_f8_to_f16(&scales, 128)
                    .with_context(|| format!("dequantizing {prefix}"))?
            } else {
                t.to_f16()
                    .with_context(|| format!("converting {prefix} to f16"))?
                    .into_owned()
            };
            // Safety: f16 is a transparent u16, so these are already the
            // little-endian halves the device wants.
            let bytes = unsafe {
                std::slice::from_raw_parts(halves.as_ptr() as *const u8, halves.len() * 2)
            }
            .to_vec();
            return Ok((bytes, WeightType::F16, k, n));
        }
        let qw = w.tensor(&format!("{prefix}.qweight"))?;
        let (k, n) = (qw.shape[0], qw.shape[1] * 8);
        // The scales are BF16 in Qwen3's AWQ export, not F16 — bound to a local
        // so the converted halves outlive the borrow `AwqTensor` takes.
        let scales_t = w.tensor(&format!("{prefix}.scales"))?;
        let scales = scales_t.to_f16()?;
        let packed = AwqTensor {
            qweight: qw.as_i32()?,
            qzeros: w.tensor(&format!("{prefix}.qzeros"))?.as_i32()?,
            scales: scales.as_ref(),
            in_features: k,
            out_features: n,
        }
        .repack()
        .with_context(|| format!("repacking {prefix}"))?;
        // The transposed layout, which the f16 tensor-core GEMM reads as one
        // aligned 16-byte fragment per lane rather than four four-byte words.
        // Worth 11% on the GEMM at 32 tokens and 5.8% on the decode step, with
        // the mat-vec level at a batch of one. `TUILI_AWQ_PACKED=1` keeps the
        // old blocks, which is how the two are A/B-ed; `transposable` rejects a
        // row length whose stride would not land the quants on 16 bytes, and
        // every real projection width passes it.
        static PACKED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*PACKED.get_or_init(|| std::env::var_os("TUILI_AWQ_PACKED").is_some())
            && tuili_kernels::awq::transposable(k)
        {
            let t = tuili_kernels::awq::transpose_words(&packed, k, n);
            return Ok((t, WeightType::Q4G128T, k, n));
        }
        Ok((packed, WeightType::Q4G128, k, n))
    };
    let projection = |prefix: &str, total: &mut usize| -> Result<Matrix> {
        let (bytes, ty, k, n) = projection_bytes(prefix)?;
        upload(&bytes, ty, k, n, total)
    };

    // `gate` and `up` as one matrix, and `q`/`k`/`v` as another, which is what
    // vLLM's `MergedColumnParallelLinear` and `QKVParallelLinear` amount to. A
    // matmul's efficiency here rises steeply with its width — 4096x14336
    // reaches 1154 GB/s where 4096x28672 reaches 1368 — because a narrow one
    // cannot fill the device.
    //
    // On by default since it was measured end to end rather than on the GEMM
    // alone: a batch-32 step's matmuls fall from 110.5 ms to 96.8 over twenty
    // steps and its launches from 225 to 129, worth 4.4% of the served
    // throughput on a Blackwell RTX PRO 6000. `TUILI_FUSE_FFN=0` puts the three
    // narrow matmuls back.
    //
    // It costs VRAM: the stacked copies are held *as well as* the originals,
    // 2 GiB on an 8B model, because at a batch of one the integer mat-vec runs
    // instead and reads the originals. Dropping them means teaching the mat-vec
    // to take a column range of a stacked matrix — the scales of a Q4_G128T
    // matrix live past all of its quants, so a sub-matrix is two disjoint byte
    // ranges rather than one. Worth doing; not worth blocking the throughput on.
    //
    // So the default is conditional rather than unconditional: the stacked
    // copies are the whole of the attention and FFN projections again, and a
    // card that cannot spare that would rather have the KV cache. Whatever the
    // decision, it is logged — a throughput number that moved by 4% because the
    // loader quietly declined is the kind of thing that costs a day.
    let fuse_ffn = match std::env::var("TUILI_FUSE_FFN").as_deref() {
        Ok("0") => false,
        Ok(_) => true,
        Err(_) => {
            // What the stacked copies will cost: `q`+`k`+`v` and `gate`+`up`
            // again, which is every projection but `o` and `down`.
            let mut extra = 0usize;
            for i in 0..cfg.n_layers {
                for m in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                          "mlp.gate_proj", "mlp.up_proj"] {
                    // `qweight` is four bits an element; the repacked form adds
                    // a scale and a zero per group of 128, so 68 bytes where
                    // the pack has 64.
                    let n = format!("model.layers.{i}.{m}.qweight");
                    extra += w.tensor(&n).map(|t| t.data.len() * 17 / 16).unwrap_or(0);
                }
            }
            let free = dev.mem_info().map(|(f, _)| f).unwrap_or(0);
            // Leave the KV cache room to be worth having: the pool is what
            // decides how many sequences can run at once, and a batch that
            // narrows costs far more than 4%.
            let room = extra * 3 < free;
            tracing::info!(
                extra_mib = extra >> 20,
                free_mib = free >> 20,
                fused = room,
                "fused projections"
            );
            room
        }
    };
    let stacked = |a: &str, b: &str, total: &mut usize| -> Result<Option<Matrix>> {
        if !fuse_ffn {
            return Ok(None);
        }
        let (ba, ty_a, k, n_a) = projection_bytes(a)?;
        let (bb, ty_b, k_b, n_b) = projection_bytes(b)?;
        // Only the transposed layout stacks: the packed one keeps its scales
        // inside each block, so appending rows is appending bytes and there is
        // nothing to gain by doing it here rather than in the kernel.
        if ty_a != WeightType::Q4G128T || ty_b != ty_a || k_b != k {
            return Ok(None);
        }
        let c = tuili_kernels::awq::concat_t(&ba, n_a, &bb, n_b, k);
        Ok(Some(upload(&c, ty_a, k, n_a + n_b, total)?))
    };
    let stacked3 = |a: &str, b: &str, cc: &str, total: &mut usize| -> Result<Option<Matrix>> {
        if !fuse_ffn {
            return Ok(None);
        }
        let (ba, ty, k, n_a) = projection_bytes(a)?;
        let (bb, ty_b, k_b, n_b) = projection_bytes(b)?;
        let (bc, ty_c, k_c, n_c) = projection_bytes(cc)?;
        if ty != WeightType::Q4G128T || ty_b != ty || ty_c != ty || k_b != k || k_c != k {
            return Ok(None);
        }
        let ab = tuili_kernels::awq::concat_t(&ba, n_a, &bb, n_b, k);
        let abc = tuili_kernels::awq::concat_t(&ab, n_a + n_b, &bc, n_c, k);
        Ok(Some(upload(&abc, ty, k, n_a + n_b + n_c, total)?))
    };

    // Two same-shaped projections of one input, stacked along the output.
    //
    // `in_proj_a` and `in_proj_b` are `value_heads` rows — 48 against a 5120
    // contraction — so their bytes want 0.34 us and each launch measured 14.2,
    // twice a layer and 96 times a decode step for 1.36 ms. Row-major means the
    // stack is byte concatenation and nothing has to be interleaved. Returns
    // `None` when the two are not the same dense type over the same `k`, in
    // which case the caller keeps its two launches.
    let stacked2 = |a: &str, b: &str, total: &mut usize| -> Result<Option<Matrix>> {
        let (mut ba, ty, k, n_a) = projection_bytes(a)?;
        let (bb, ty_b, k_b, n_b) = projection_bytes(b)?;
        if ty != ty_b || k != k_b || !matches!(ty, WeightType::F16 | WeightType::F32) {
            return Ok(None);
        }
        ba.extend_from_slice(&bb);
        Ok(Some(upload(&ba, ty, k, n_a + n_b, total)?))
    };

    // Where the text model sits. A multimodal export nests it under
    // `language_model`, so the same tensor is
    // `model.language_model.embed_tokens.weight` there and
    // `model.embed_tokens.weight` everywhere else. Probed rather than derived
    // from the architecture name: the nesting is a property of how the
    // checkpoint was written.
    //
    // The layer prefix below is derived from this rather than probed separately.
    // Probing them independently is how the first attempt at the 27B got the
    // layers right and the embedding wrong, and failed one tensor into the load.
    let stem = ["model.language_model", "model"]
        .into_iter()
        .find(|s| w.get(&format!("{s}.embed_tokens.weight")).is_some())
        .context(
            "found no embedding under `model.embed_tokens.weight` or \
             `model.language_model.embed_tokens.weight`; the checkpoint's tensor \
             names are not ones this loader recognises",
        )?;
    tracing::info!(stem, "text model tensors");

    let embd = w.tensor(&format!("{stem}.embed_tokens.weight"))?;
    // `to_f16` rather than `embd.data`: this uploaded the mapping's bytes and
    // labelled them `F16`, which reinterprets bf16 bit patterns as halves when
    // the checkpoint stores BF16 — no error, just a wrong number for every
    // token. Qwen3's AWQ export writes every float as BF16; Llama-3.1 and
    // Qwen2.5 write F16, which is why the raw upload had never been wrong
    // before and why both of those models kept working while Qwen3 produced
    // degenerate repetition.
    //
    // The embedding is the first operation in the forward pass, so this is the
    // one place where being wrong is invisible downstream: the block that
    // follows normalizes the magnitude away, leaving a residual stream whose
    // RMS climbs smoothly through all 36 layers while carrying nonsense. A
    // magnitude curve cannot detect it; only comparing values can.
    let embd_halves = embd.to_f16()?;
    // Safety: f16 is a transparent u16, so these are already the little-endian
    // halves the device expects; the view does not outlive `embd_halves`.
    let embd_bytes = unsafe {
        std::slice::from_raw_parts(embd_halves.as_ptr() as *const u8, embd_halves.len() * 2)
    };
    let token_embd = upload(
        embd_bytes,
        WeightType::F16,
        embd.shape[1],
        embd.shape[0],
        &mut device_bytes,
    )?;
    let output_norm = vector(&format!("{stem}.norm.weight"), &mut device_bytes)?;
    let output = if cfg.tied_embeddings {
        None
    } else {
        let h = w.tensor("lm_head.weight")?;
        let (n, k) = (h.shape[0], h.shape[1]);
        // `TUILI_LM_HEAD=f16` keeps the matrix as it came, which prices the Q8_0
        // path against twice the bytes. On a card with a small L2 the quantized
        // path reads 558 MB at 90 GB/s, and the question is whether that is the
        // bytes or the layout: a Q8_0 block is 34 bytes, so its quants are only
        // ever halfword-aligned and `mmq_load_w_q8_0` reads them two at a time.
        // `to_f16` rather than `h.data` / `as_f16`: Qwen3's AWQ export stores
        // lm_head as BF16. Uploading its bytes as `WeightType::F16` would
        // reinterpret bf16 bit patterns as halves — no error, just wrong
        // numbers everywhere the vocab projection is read.
        let halves = h.to_f16()?;
        // The two working control checkpoints both store lm_head as F16, so
        // `to_f16` returns a borrow for them and its BF16 branch was never
        // exercised by a model known to produce sane output. Log the first few
        // converted values so they can be checked against the file directly.
        if std::env::var_os("TUILI_LM_HEAD_PROBE").is_some() {
            let head: Vec<f32> = halves.iter().take(8).map(|x| f32::from(*x)).collect();
            tracing::info!(
                dtype = ?h.dtype,
                n = halves.len(),
                ?head,
                "lm_head probe: first 8 converted values"
            );
        }
        if std::env::var("TUILI_LM_HEAD").as_deref() == Ok("f16") {
            tracing::info!(mib = (halves.len() * 2) >> 20, "vocab projection kept f16");
            // Safety: f16 is a transparent u16, so the halves are already the
            // little-endian byte layout the device wants; the view does not
            // outlive `halves`.
            let bytes = unsafe {
                std::slice::from_raw_parts(halves.as_ptr() as *const u8, halves.len() * 2)
            };
            Some(upload(bytes, WeightType::F16, k, n, &mut device_bytes)?)
        } else {
            let q = quantize_f16_to_q8_0(halves.as_ref(), k).context("quantizing lm_head")?;
            tracing::info!(
                from_mib = h.data.len() >> 20,
                to_mib = q.len() >> 20,
                "vocab projection quantized to Q8_0"
            );
            Some(upload(&q, WeightType::Q8_0, k, n, &mut device_bytes)?)
        }
    };
    // And the split layout for the batched path, when there is room for both.
    // `TUILI_LM_HEAD=packed` keeps only the packed one, which is the A/B.
    let output_split = match (&output, std::env::var("TUILI_LM_HEAD").as_deref()) {
        (Some(o), Ok("packed")) => {
            let _ = o;
            None
        }
        (Some(o), _) if o.ty == WeightType::Q8_0 => {
            let h = w.tensor("lm_head.weight")?;
            let (n, k) = (h.shape[0], h.shape[1]);
            let free = dev.mem_info().map(|(f, _)| f).unwrap_or(0);
            let want = n * k * 17 / 16;
            if want * 3 < free {
                let q = tuili_kernels::awq::quantize_f16_to_q8_0_split(h.to_f16()?.as_ref(), k)
                    .context("quantizing lm_head, split")?;
                tracing::info!(mib = q.len() >> 20, "vocab projection also split");
                Some(upload(&q, WeightType::Q8_0S, k, n, &mut device_bytes)?)
            } else {
                tracing::info!(want_mib = want >> 20, free_mib = free >> 20,
                               "no room for the split vocab projection");
                None
            }
        }
        _ => None,
    };
    device_bytes += freq_factors.len() * 4;
    let rope_freqs = dev.stream().clone_htod(freq_factors)?;

    // Where the decoder layers live. A multimodal checkpoint nests the text
    // model under `language_model`, so the same layer is
    // `model.language_model.layers.0` there and `model.layers.0` everywhere
    // else. Probe rather than branch on the architecture name: the prefix is a
    // property of how the checkpoint was exported, not of the model.
    let layer_prefix = format!("{stem}.layers");
    anyhow::ensure!(
        [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "linear_attn.in_proj_qkv.weight",
        ]
        .iter()
        .any(|leaf| w.get(&format!("{layer_prefix}.0.{leaf}")).is_some()),
        "the embedding is under `{stem}` but there is no layer 0 under \
         `{layer_prefix}`; this checkpoint splits the text model across two \
         prefixes and the loader assumes one"
    );
    tracing::info!(prefix = layer_prefix, "decoder layers");

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let p = format!("{layer_prefix}.{i}");
        // Which mixer this block has, decided by which tensors exist.
        //
        // `text_config.layer_types` says the same thing, and reading it would
        // work. The tensors are the stronger signal: if the config and the
        // weights ever disagree, this way fails with a missing tensor at the
        // layer in question, where trusting the config would slice a projection
        // that is not there — or worse, find one of the right size and mean
        // something else by it. Deriving the pattern from
        // `full_attention_interval` instead would additionally bake in a stride
        // that this checkpoint happens to have and the next need not.
        let is_linear = w
            .get(&format!("{p}.linear_attn.in_proj_qkv.weight"))
            .is_some();

        let attn = if is_linear {
            None
        } else {
            let wq = projection(&format!("{p}.self_attn.q_proj"), &mut device_bytes)?;
            // `q_proj` producing twice the attention width means it carries a
            // gate interleaved with the query, per head. Detected from the
            // shape rather than from the config's `attn_output_gate`, because
            // the shape is what the rest of the code has to agree with.
            let output_gate = wq.n == 2 * cfg.d_attn();
            anyhow::ensure!(
                wq.n == cfg.d_attn() || output_gate,
                "layer {i} q_proj has {} columns; expected {} for a plain query \
                 or {} for a query and its gate",
                wq.n,
                cfg.d_attn(),
                2 * cfg.d_attn(),
            );
            Some(AttnWeights {
                wq,
                wk: projection(&format!("{p}.self_attn.k_proj"), &mut device_bytes)?,
                wv: projection(&format!("{p}.self_attn.v_proj"), &mut device_bytes)?,
                wo: projection(&format!("{p}.self_attn.o_proj"), &mut device_bytes)?,
                bq: optional_vector(&format!("{p}.self_attn.q_proj.bias"), &mut device_bytes)?,
                bk: optional_vector(&format!("{p}.self_attn.k_proj.bias"), &mut device_bytes)?,
                bv: optional_vector(&format!("{p}.self_attn.v_proj.bias"), &mut device_bytes)?,
                bo: optional_vector(&format!("{p}.self_attn.o_proj.bias"), &mut device_bytes)?,
                q_norm: optional_vector(
                    &format!("{p}.self_attn.q_norm.weight"), &mut device_bytes)?,
                k_norm: optional_vector(
                    &format!("{p}.self_attn.k_norm.weight"), &mut device_bytes)?,
                // The fused QKV stack assumes three same-shaped projections of
                // one input; a gated q_proj is twice as wide as the stack
                // expects, so leave it unfused rather than mis-slice it.
                w_qkv: if output_gate {
                    None
                } else {
                    stacked3(
                        &format!("{p}.self_attn.q_proj"),
                        &format!("{p}.self_attn.k_proj"),
                        &format!("{p}.self_attn.v_proj"),
                        &mut device_bytes,
                    )?
                },
                output_gate,
            })
        };

        let gdn = if is_linear {
            let l = format!("{p}.linear_attn");
            Some(GdnWeights {
                in_proj_qkv: projection(&format!("{l}.in_proj_qkv"), &mut device_bytes)?,
                in_proj_z: projection(&format!("{l}.in_proj_z"), &mut device_bytes)?,
                in_proj_a: projection(&format!("{l}.in_proj_a"), &mut device_bytes)?,
                in_proj_b: projection(&format!("{l}.in_proj_b"), &mut device_bytes)?,
                in_proj_ba: stacked2(
                    &format!("{l}.in_proj_a"),
                    &format!("{l}.in_proj_b"),
                    &mut device_bytes,
                )?,
                conv1d: vector(&format!("{l}.conv1d.weight"), &mut device_bytes)?,
                a_log: vector(&format!("{l}.A_log"), &mut device_bytes)?,
                dt_bias: vector(&format!("{l}.dt_bias"), &mut device_bytes)?,
                norm: vector(&format!("{l}.norm.weight"), &mut device_bytes)?,
                out_proj: projection(&format!("{l}.out_proj"), &mut device_bytes)?,
            })
        } else {
            None
        };

        layers.push(Layer {
            attn_norm: vector(&format!("{p}.input_layernorm.weight"), &mut device_bytes)?,
            attn,
            gdn,
            ffn_norm: vector(
                &format!("{p}.post_attention_layernorm.weight"),
                &mut device_bytes,
            )?,
            w_gate: projection(&format!("{p}.mlp.gate_proj"), &mut device_bytes)?,
            w_up: projection(&format!("{p}.mlp.up_proj"), &mut device_bytes)?,
            w_gate_up: stacked(
                &format!("{p}.mlp.gate_proj"),
                &format!("{p}.mlp.up_proj"),
                &mut device_bytes,
            )?,
            w_down: projection(&format!("{p}.mlp.down_proj"), &mut device_bytes)?,
            blob: None,
        });
    }

    tracing::info!(
        layers = cfg.n_layers,
        vram_mib = device_bytes >> 20,
        ms = started.elapsed().as_millis(),
        "awq weights loaded"
    );
    Ok(Weights {
        token_embd,
        layers,
        output_norm,
        output,
        output_split,
        rope_freqs,
        device_bytes,
        host_bytes: 0,
        max_blob_bytes: 0,
    })
}

fn upload_matrix(dev: &Device, f: &Gguf, name: &str, total: &mut usize) -> Result<Matrix> {
    let (ty, k, n, n_bytes) = describe(f, name)?;
    let bytes = f.tensor_data(name)?;
    *total += bytes.len();
    let data = dev
        .stream()
        .clone_htod(bytes)
        .with_context(|| format!("uploading {name} ({} MiB)", bytes.len() >> 20))?;
    Ok(Matrix {
        ty,
        k,
        n,
        n_bytes,
        storage: Storage::Device(data),
    })
}

/// Copy a layer's matrices into one page-locked blob and describe each one's
/// place inside it.
fn pack_layer(dev: &Device, f: &Gguf, names: &[String]) -> Result<(Vec<Matrix>, LayerBlob)> {
    let mut described = Vec::with_capacity(names.len());
    let mut offsets = Vec::with_capacity(names.len());
    let mut total = 0usize;
    for name in names {
        let d = describe(f, name)?;
        offsets.push(total);
        total += d.3.next_multiple_of(BLOB_ALIGN);
        described.push(d);
    }

    // Safety: the allocation is fully written below before any read, and the
    // handle owns the memory for as long as the weights live.
    let mut host = unsafe { dev.context().alloc_pinned::<u8>(total) }
        .with_context(|| format!("allocating {} MiB of pinned host memory", total >> 20))?;
    {
        let dst = host.as_mut_slice()?;
        // Padding is never read, but leaving it uninitialized would make the
        // DMA copy indeterminate bytes into VRAM.
        dst.fill(0);
        for (name, &offset) in names.iter().zip(&offsets) {
            let src = f.tensor_data(name)?;
            dst[offset..offset + src.len()].copy_from_slice(src);
        }
    }

    let matrices = described
        .into_iter()
        .zip(&offsets)
        .map(|((ty, k, n, n_bytes), &offset)| Matrix {
            ty,
            k,
            n,
            n_bytes,
            storage: Storage::Streamed { offset },
        })
        .collect();

    Ok((matrices, LayerBlob { host, bytes: total }))
}

fn upload_optional_vector(
    dev: &Device,
    f: &Gguf,
    name: &str,
    total: &mut usize,
) -> Result<Option<Vector>> {
    match f.get_tensor(name) {
        Some(_) => Ok(Some(upload_vector(dev, f, name, total)?)),
        None => Ok(None),
    }
}

/// Norm gains and biases are tiny, so they are converted on the host and kept
/// in f32 regardless of how the file stores them.
fn upload_vector(dev: &Device, f: &Gguf, name: &str, total: &mut usize) -> Result<Vector> {
    let info = f.tensor(name)?;
    let host =
        to_f32(f.data(info), info).with_context(|| format!("decoding {name} ({})", info.ty))?;
    *total += host.len() * 4;
    Ok(dev.stream().clone_htod(&host)?)
}

/// `ssm_a` back into the `A_log` the recurrence wants.
///
/// llama.cpp's converter stores `-exp(A_log)`, not `A_log`: this checkpoint's
/// `blk.0.ssm_a` spans [-0.3376, -0.0038], which is exactly `-exp` of the
/// reference's [-5.5625, -1.0859]. `gdn_gate_decay` computes
/// `-exp(a_log) * softplus(...)`, so feeding it the file's value directly turns
/// a decay of 0.03 into 0.97 and every linear block remembers everything.
///
/// Inverting here rather than adding a kernel variant keeps one gate kernel for
/// both loaders, which is the same trade the safetensors path makes for the
/// norm gains.
fn upload_a_log(dev: &Device, f: &Gguf, name: &str, total: &mut usize) -> Result<Vector> {
    let info = f.tensor(name)?;
    let raw = to_f32(f.data(info), info).with_context(|| format!("decoding {name}"))?;
    let host: Vec<f32> = raw
        .iter()
        .map(|a| {
            debug_assert!(*a < 0.0, "{name}: {a} is not -exp(A_log)");
            (-a).ln()
        })
        .collect();
    *total += host.len() * 4;
    Ok(dev.stream().clone_htod(&host)?)
}

fn to_f32(bytes: &[u8], info: &TensorInfo) -> Result<Vec<f32>> {
    Ok(match info.ty {
        GgmlType::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        GgmlType::F16 => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect(),
        GgmlType::BF16 => bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect(),
        other => anyhow::bail!("1-D tensors must be F32, F16 or BF16, got {other}"),
    })
}

// ----------------------------------------------------------------- the vision tower

/// The vision tower's 333 tensors, owned.
///
/// [`tuili_kernels::vision::VisionWeights`] is all borrowed views, so something
/// has to hold the allocations; this is it. Everything is f16 or f32 with no
/// quantized path, and that is not an omission: the whole tower sits in the
/// checkpoint's `modules_to_not_convert`, so there are no `weight_scale_inv`
/// tensors to read and a block-dequantizing branch here would be dead code
/// pretending to be generality.
pub struct VisionTower {
    pub shape: tuili_kernels::vision::VisionShape,
    pub cfg: crate::config::VisionConfig,
    patch_embed_w: Buf<half::f16>,
    patch_embed_b: Vector,
    pos_embed: Vector,
    blocks: Vec<VisionBlock>,
    merger_norm_w: Vector,
    merger_norm_b: Vector,
    merger_fc1_w: Buf<half::f16>,
    merger_fc1_b: Vector,
    merger_fc2_w: Buf<half::f16>,
    merger_fc2_b: Vector,
    pub device_bytes: usize,
}

struct VisionBlock {
    norm1_w: Vector,
    norm1_b: Vector,
    norm2_w: Vector,
    norm2_b: Vector,
    qkv_w: Buf<half::f16>,
    qkv_b: Vector,
    proj_w: Buf<half::f16>,
    proj_b: Vector,
    fc1_w: Buf<half::f16>,
    fc1_b: Vector,
    fc2_w: Buf<half::f16>,
    fc2_b: Vector,
}

impl VisionTower {
    /// Borrowed views in the shape the kernels take.
    pub fn weights(&self) -> tuili_kernels::vision::VisionWeights<'_> {
        tuili_kernels::vision::VisionWeights {
            patch_embed_w: self.patch_embed_w.as_view(),
            patch_embed_b: self.patch_embed_b.as_view(),
            pos_embed: self.pos_embed.as_view(),
            blocks: self
                .blocks
                .iter()
                .map(|b| tuili_kernels::vision::VisionBlockWeights {
                    norm1_w: b.norm1_w.as_view(),
                    norm1_b: b.norm1_b.as_view(),
                    norm2_w: b.norm2_w.as_view(),
                    norm2_b: b.norm2_b.as_view(),
                    qkv_w: b.qkv_w.as_view(),
                    qkv_b: b.qkv_b.as_view(),
                    proj_w: b.proj_w.as_view(),
                    proj_b: b.proj_b.as_view(),
                    fc1_w: b.fc1_w.as_view(),
                    fc1_b: b.fc1_b.as_view(),
                    fc2_w: b.fc2_w.as_view(),
                    fc2_b: b.fc2_b.as_view(),
                })
                .collect(),
            merger_norm_w: self.merger_norm_w.as_view(),
            merger_norm_b: self.merger_norm_b.as_view(),
            merger_fc1_w: self.merger_fc1_w.as_view(),
            merger_fc1_b: self.merger_fc1_b.as_view(),
            merger_fc2_w: self.merger_fc2_w.as_view(),
            merger_fc2_b: self.merger_fc2_b.as_view(),
        }
    }
}

/// Load `model.visual.*`, or `None` when the checkpoint has no tower.
///
/// The two claims worth stating, because both were settled by reading the
/// checkpoint rather than the class:
///
/// * **`patch_embed.proj.weight` is `[1152, 3, 2, 16, 16]` and wants to be
///   `[1152, 1536]`.** It is a `Conv3d` whose kernel equals its stride over an
///   input already cut into patches, so the flatten is a free view and the
///   patch embedding is a GEMM. Treating it as a convolution is wasted work,
///   not a different answer.
/// * **Every bias is real.** The text tower has none at all — `attention_bias:
///   false`, bias-free MLPs — so a loader written from that habit drops twelve
///   tensors a block. Dropping only `patch_embed.proj.bias` already moves the
///   patch embedding by 3.05 out of a peak of 3.15. This is the AWQ loader
///   dropping Qwen's QKV bias again, and it reads as fluent nonsense.
///
/// The deepstack mergers are deliberately not loaded: `deepstack_visual_indexes`
/// is empty and the tensors do not exist, whatever `modules_to_not_convert`
/// lists.
pub fn load_vision(
    dev: &Device,
    w: &tuili_safetensors::Shards,
    cfg: &Config,
) -> Result<Option<VisionTower>> {
    let present = w.get("model.visual.patch_embed.proj.weight").is_some();
    let Some(vc) = cfg.vision else {
        anyhow::ensure!(
            !present,
            "this checkpoint ships `model.visual.*` tensors but its config has \
             no `vision_config`; the tower's depth and hidden size decide 333 \
             tensors' shapes and guessing them is not safe"
        );
        return Ok(None);
    };
    anyhow::ensure!(
        present,
        "config describes a vision tower of depth {} but there is no \
         `model.visual.patch_embed.proj.weight`",
        vc.depth
    );
    // `deepstack_visual_indexes` is empty on this checkpoint. If it were not,
    // the tower would emit extra feature streams that the text side has to
    // consume at named layers, and loading only the trunk would quietly drop
    // them.
    anyhow::ensure!(
        w.get("model.visual.deepstack_merger_list.0.norm.weight").is_none(),
        "this checkpoint carries deepstack mergers; the tower here emits one \
         feature stream and splicing only that would drop the rest"
    );

    let started = std::time::Instant::now();
    let mut bytes = 0usize;
    let vector = |name: &str, want: usize, total: &mut usize| -> Result<Vector> {
        let t = w.tensor(name)?;
        let v = t.to_f32()?;
        anyhow::ensure!(
            v.len() == want,
            "{name} holds {} floats, expected {want}",
            v.len()
        );
        *total += v.len() * 4;
        Ok(dev.stream().clone_htod(&v)?)
    };
    // `[rows, cols]` after flattening every trailing dimension, which is what
    // makes `proj.weight`'s five dimensions a two-dimensional GEMM operand
    // without a copy.
    let matrix =
        |name: &str, rows: usize, cols: usize, total: &mut usize| -> Result<Buf<half::f16>> {
            let t = w.tensor(name)?;
            let elems: usize = t.shape.iter().product();
            anyhow::ensure!(
                t.shape[0] == rows && elems == rows * cols,
                "{name} has shape {:?}, which is not [{rows}, {cols}] however it \
                 is flattened",
                t.shape
            );
            let halves = t.to_f16()?;
            anyhow::ensure!(halves.len() == rows * cols, "{name}: {} halves", halves.len());
            *total += halves.len() * 2;
            Ok(dev.stream().clone_htod(halves.as_ref())?)
        };

    let (d, h4) = (vc.hidden, 4 * vc.hidden);
    let patch_dim = vc.in_channels * vc.temporal_patch * vc.patch * vc.patch;
    let mut blocks = Vec::with_capacity(vc.depth);
    for i in 0..vc.depth {
        let p = format!("model.visual.blocks.{i}");
        blocks.push(VisionBlock {
            norm1_w: vector(&format!("{p}.norm1.weight"), d, &mut bytes)?,
            norm1_b: vector(&format!("{p}.norm1.bias"), d, &mut bytes)?,
            norm2_w: vector(&format!("{p}.norm2.weight"), d, &mut bytes)?,
            norm2_b: vector(&format!("{p}.norm2.bias"), d, &mut bytes)?,
            // `[3 * hidden, hidden]`, whole-q-then-whole-k-then-whole-v rows —
            // not the text side's per-head interleaving.
            qkv_w: matrix(&format!("{p}.attn.qkv.weight"), 3 * d, d, &mut bytes)?,
            qkv_b: vector(&format!("{p}.attn.qkv.bias"), 3 * d, &mut bytes)?,
            proj_w: matrix(&format!("{p}.attn.proj.weight"), d, d, &mut bytes)?,
            proj_b: vector(&format!("{p}.attn.proj.bias"), d, &mut bytes)?,
            fc1_w: matrix(&format!("{p}.mlp.linear_fc1.weight"), vc.intermediate, d, &mut bytes)?,
            fc1_b: vector(&format!("{p}.mlp.linear_fc1.bias"), vc.intermediate, &mut bytes)?,
            fc2_w: matrix(&format!("{p}.mlp.linear_fc2.weight"), d, vc.intermediate, &mut bytes)?,
            fc2_b: vector(&format!("{p}.mlp.linear_fc2.bias"), d, &mut bytes)?,
        });
    }

    let tower = VisionTower {
        shape: tuili_kernels::vision::VisionShape {
            depth: vc.depth,
            hidden: vc.hidden,
            heads: vc.heads,
            intermediate: vc.intermediate,
            out_hidden: vc.out_hidden,
            in_channels: vc.in_channels,
            patch: vc.patch,
            temporal_patch: vc.temporal_patch,
            merge: vc.merge,
            eps: crate::config::VisionConfig::EPS,
            rope_theta: crate::config::VisionConfig::ROPE_THETA,
        },
        cfg: vc,
        patch_embed_w: matrix(
            "model.visual.patch_embed.proj.weight",
            vc.hidden,
            patch_dim,
            &mut bytes,
        )?,
        patch_embed_b: vector("model.visual.patch_embed.proj.bias", d, &mut bytes)?,
        pos_embed: vector(
            "model.visual.pos_embed.weight",
            vc.position_embeddings * d,
            &mut bytes,
        )?,
        blocks,
        // `[hidden]`, not `[4 * hidden]`: the merger normalizes each patch
        // before it groups them. A post-shuffle norm would make this 4608 wide
        // and the shape check here is what settles it.
        merger_norm_w: vector("model.visual.merger.norm.weight", d, &mut bytes)?,
        merger_norm_b: vector("model.visual.merger.norm.bias", d, &mut bytes)?,
        merger_fc1_w: matrix("model.visual.merger.linear_fc1.weight", h4, h4, &mut bytes)?,
        merger_fc1_b: vector("model.visual.merger.linear_fc1.bias", h4, &mut bytes)?,
        merger_fc2_w: matrix(
            "model.visual.merger.linear_fc2.weight",
            vc.out_hidden,
            h4,
            &mut bytes,
        )?,
        merger_fc2_b: vector("model.visual.merger.linear_fc2.bias", vc.out_hidden, &mut bytes)?,
        device_bytes: 0,
    };
    let tower = VisionTower { device_bytes: bytes, ..tower };
    tracing::info!(
        tensors = 3 + 6 + 12 * vc.depth,
        vram_mib = bytes >> 20,
        ms = started.elapsed().as_millis(),
        "vision tower loaded"
    );
    Ok(Some(tower))
}

/// The MTP head out of llama.cpp's sidecar GGUF.
///
/// `mtp-Qwen3.8-27B-Q8_0.gguf` is a standalone file rather than a fragment: the
/// head is `blk.64` of a 65-block model, and the file repackages `token_embd`
/// and `output` so llama.cpp can open it alone. Those two are copies of the text
/// model's, not dedicated ones — the head shares, which is what
/// `mtp_use_dedicated_embeddings = false` means — so they are read past rather
/// than loaded.
///
/// That is the one place this differs from [`load_mtp`]. There, the *absence* of
/// `mtp.embed_tokens` is the evidence for sharing, and the loader refuses a
/// checkpoint that ships one. Here the file's self-containment destroys that
/// evidence: a repackaged copy and a dedicated head's own embedding look
/// identical from the tensor list. So the claim has to be carried by the
/// metadata instead, and it is the weaker of the two checks. Worth saying rather
/// than papering over.
///
/// Everything else maps without interpretation. `blk.64` is shaped exactly like
/// one of the text model's sixteen attention blocks — output gate packed into
/// the high half of `attn_q`, q/k norms per head, `post_attention_norm` before
/// the FFN — so the layer loads through the same names at index `n_layers`. The
/// four tensors with no text-model counterpart are the ones under `nextn.`.
pub fn load_mtp_gguf(dev: &Device, f: &Gguf, cfg: &Config) -> Result<Option<MtpWeights>> {
    let t = |s: &str| format!("blk.{}.{s}", cfg.n_layers);
    if f.get_tensor(&t("nextn.eh_proj.weight")).is_none() {
        return Ok(None);
    }
    // The head's depth, from the sidecar rather than from `cfg`: the main
    // checkpoint's metadata does not mention the head at all, so the file
    // carrying the tensors is the only one that can say how many layers they
    // are. `cfg.mtp_layers` stays zero on this path and is not consulted.
    let depth = f
        .usize(&f.akey("nextn_predict_layers")?)
        .context(
            "the sidecar carries `nextn.eh_proj` but no `nextn_predict_layers`; \
             the draft loop indexes layers by it and guessing is not safe",
        )?;
    anyhow::ensure!(
        depth == 1,
        "this loader builds a one-layer MTP head, the sidecar says {depth}. The \
         draft loop indexes `spec_step_idx % nextn_predict_layers`, which is only \
         the identity at one layer"
    );
    // The head's layer is a full-attention block, which the file settles without
    // interpretation. Same check as the safetensors path and for the same
    // reason: a drafter that ran a GatedDeltaNet layer would advance recurrent
    // state on every speculative token.
    anyhow::ensure!(
        f.get_tensor(&t("ssm_a")).is_none(),
        "blk.{} carries `ssm_a`, so it is a linear-attention block rather than \
         the full-attention one this drafter assumes; drafting would touch \
         recurrent state",
        cfg.n_layers
    );

    let started = std::time::Instant::now();
    let mut bytes = 0usize;
    let layer = Layer {
        attn_norm: upload_vector(dev, f, &t("attn_norm.weight"), &mut bytes)?,
        attn: Some(AttnWeights {
            wq: upload_matrix(dev, f, &t("attn_q.weight"), &mut bytes)?,
            wk: upload_matrix(dev, f, &t("attn_k.weight"), &mut bytes)?,
            wv: upload_matrix(dev, f, &t("attn_v.weight"), &mut bytes)?,
            wo: upload_matrix(dev, f, &t("attn_output.weight"), &mut bytes)?,
            bq: upload_optional_vector(dev, f, &t("attn_q.bias"), &mut bytes)?,
            bk: upload_optional_vector(dev, f, &t("attn_k.bias"), &mut bytes)?,
            bv: upload_optional_vector(dev, f, &t("attn_v.bias"), &mut bytes)?,
            bo: upload_optional_vector(dev, f, &t("attn_output.bias"), &mut bytes)?,
            q_norm: upload_optional_vector(dev, f, &t("attn_q_norm.weight"), &mut bytes)?,
            k_norm: upload_optional_vector(dev, f, &t("attn_k_norm.weight"), &mut bytes)?,
            w_qkv: None,
            output_gate: cfg.attn_output_gate,
        }),
        gdn: None,
        ffn_norm: upload_vector(dev, f, &t("post_attention_norm.weight"), &mut bytes)?,
        w_gate: upload_matrix(dev, f, &t("ffn_gate.weight"), &mut bytes)?,
        w_up: upload_matrix(dev, f, &t("ffn_up.weight"), &mut bytes)?,
        w_down: upload_matrix(dev, f, &t("ffn_down.weight"), &mut bytes)?,
        w_gate_up: None,
        blob: None,
    };
    let w = MtpWeights {
        // `[k = 2 * d_model, n = d_model]`, embedding in the low half. The name
        // is llama.cpp's evidence for the order: `eh_proj` projects `[e | h]`,
        // and the conversion writes the HF `fc` through untransposed.
        //
        // If that order were wrong the drafts would simply stop being accepted —
        // speculation verifies every token against the target model, so a broken
        // head costs throughput and never correctness. The acceptance rate is
        // the test, and it is a sharp one: near zero if this is backwards.
        fc: upload_matrix(dev, f, &t("nextn.eh_proj.weight"), &mut bytes)?,
        pre_fc_norm_embedding: upload_vector(dev, f, &t("nextn.enorm.weight"), &mut bytes)?,
        pre_fc_norm_hidden: upload_vector(dev, f, &t("nextn.hnorm.weight"), &mut bytes)?,
        norm: upload_vector(dev, f, &t("nextn.shared_head_norm.weight"), &mut bytes)?,
        layer,
        device_bytes: bytes,
    };
    dev.synchronize()?;
    tracing::info!(
        mib = bytes >> 20,
        ms = started.elapsed().as_millis(),
        "MTP head loaded from sidecar"
    );
    Ok(Some(w))
}
