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
use cudarc::driver::{CudaSlice, CudaView, PinnedHostSlice};
use tuili_cuda::Device;
use tuili_gguf::{GgmlType, Gguf, TensorInfo};
use tuili_kernels::WeightType;

use crate::config::Config;

/// Matrices inside a layer blob start on this boundary, which satisfies every
/// ggml block type's alignment and keeps each sub-copy DMA-friendly.
const BLOB_ALIGN: usize = 256;

/// Where a matrix's bytes live.
enum Storage {
    /// In VRAM, for the process lifetime.
    Device(CudaSlice<u8>),
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
    pub fn view<'a>(&'a self, stage: Option<&'a CudaSlice<u8>>) -> Result<CudaView<'a, u8>> {
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
pub type Vector = CudaSlice<f32>;

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

pub struct Layer {
    pub attn_norm: Vector,
    pub wq: Matrix,
    pub wk: Matrix,
    pub wv: Matrix,
    pub wo: Matrix,
    pub bq: Option<Vector>,
    pub bk: Option<Vector>,
    pub bv: Option<Vector>,
    pub bo: Option<Vector>,
    pub ffn_norm: Vector,
    pub w_gate: Matrix,
    pub w_up: Matrix,
    pub w_down: Matrix,
    /// `gate` and `up` stacked along `n`, under `TUILI_FUSE_FFN`. One matmul
    /// instead of two; see `stacked` in `load_awq`.
    pub w_gate_up: Option<Matrix>,
    /// `q`, `k` and `v` stacked along `n`, under `TUILI_FUSE_FFN`. One matmul
    /// and a scatter instead of three; see `stacked` in `load_awq`.
    pub w_qkv: Option<Matrix>,
    /// Present when this layer's matrices are streamed rather than resident.
    pub blob: Option<LayerBlob>,
}

impl Layer {
    pub fn is_offloaded(&self) -> bool {
        self.blob.is_some()
    }
}

pub struct Weights {
    pub token_embd: Matrix,
    pub layers: Vec<Layer>,
    pub output_norm: Vector,
    /// Absent when the model ties the output projection to the embeddings.
    pub output: Option<Matrix>,
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
            let names = [
                t("attn_q.weight"),
                t("attn_k.weight"),
                t("attn_v.weight"),
                t("attn_output.weight"),
                t("ffn_gate.weight"),
                t("ffn_up.weight"),
                t("ffn_down.weight"),
            ];

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

            layers.push(Layer {
                attn_norm: upload_vector(dev, f, &t("attn_norm.weight"), &mut device_bytes)?,
                wq: matrices.next().unwrap(),
                wk: matrices.next().unwrap(),
                wv: matrices.next().unwrap(),
                wo: matrices.next().unwrap(),
                // Qwen2 carries QKV biases; Llama does not.
                bq: upload_optional_vector(dev, f, &t("attn_q.bias"), &mut device_bytes)?,
                bk: upload_optional_vector(dev, f, &t("attn_k.bias"), &mut device_bytes)?,
                bv: upload_optional_vector(dev, f, &t("attn_v.bias"), &mut device_bytes)?,
                bo: upload_optional_vector(dev, f, &t("attn_output.bias"), &mut device_bytes)?,
                ffn_norm: upload_vector(dev, f, &t("ffn_norm.weight"), &mut device_bytes)?,
                w_gate: matrices.next().unwrap(),
                w_up: matrices.next().unwrap(),
                w_down: matrices.next().unwrap(),
                w_gate_up: None,
                w_qkv: None,
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
        let kv_dim = cfg.n_kv_heads * cfg.d_head;

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
            expect(&l.wq, d, d, "attn_q")?;
            expect(&l.wk, d, kv_dim, "attn_k")?;
            expect(&l.wv, d, kv_dim, "attn_v")?;
            expect(&l.wo, d, d, "attn_output")?;
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
            for m in [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down] {
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
    let vector = |name: &str, total: &mut usize| -> Result<Vector> {
        let t = w.tensor(name)?;
        let v: Vec<f32> = t.as_f16()?.iter().map(|x| f32::from(*x)).collect();
        *total += v.len() * 4;
        Ok(dev.stream().clone_htod(&v)?)
    };
    // A quantized projection's bytes, before they reach the device: AWQ's three
    // tensors in, one packed matrix out. Split from the upload so that
    // projections which are stacked into one matrix — see `fuse_ffn` below —
    // can be concatenated in the layout they will be read in.
    let projection_bytes = |prefix: &str| -> Result<(Vec<u8>, WeightType, usize, usize)> {
        let qw = w.tensor(&format!("{prefix}.qweight"))?;
        let (k, n) = (qw.shape[0], qw.shape[1] * 8);
        let packed = AwqTensor {
            qweight: qw.as_i32()?,
            qzeros: w.tensor(&format!("{prefix}.qzeros"))?.as_i32()?,
            scales: w.tensor(&format!("{prefix}.scales"))?.as_f16()?,
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

    let embd = w.tensor("model.embed_tokens.weight")?;
    let token_embd = upload(
        embd.data,
        WeightType::F16,
        embd.shape[1],
        embd.shape[0],
        &mut device_bytes,
    )?;
    let output_norm = vector("model.norm.weight", &mut device_bytes)?;
    let output = if cfg.tied_embeddings {
        None
    } else {
        let h = w.tensor("lm_head.weight")?;
        let (n, k) = (h.shape[0], h.shape[1]);
        let q = quantize_f16_to_q8_0(h.as_f16()?, k).context("quantizing lm_head")?;
        tracing::info!(
            from_mib = h.data.len() >> 20,
            to_mib = q.len() >> 20,
            "vocab projection quantized to Q8_0"
        );
        Some(upload(&q, WeightType::Q8_0, k, n, &mut device_bytes)?)
    };
    device_bytes += freq_factors.len() * 4;
    let rope_freqs = dev.stream().clone_htod(freq_factors)?;

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let p = format!("model.layers.{i}");
        layers.push(Layer {
            attn_norm: vector(&format!("{p}.input_layernorm.weight"), &mut device_bytes)?,
            wq: projection(&format!("{p}.self_attn.q_proj"), &mut device_bytes)?,
            wk: projection(&format!("{p}.self_attn.k_proj"), &mut device_bytes)?,
            wv: projection(&format!("{p}.self_attn.v_proj"), &mut device_bytes)?,
            wo: projection(&format!("{p}.self_attn.o_proj"), &mut device_bytes)?,
            bq: None,
            bk: None,
            bv: None,
            bo: None,
            ffn_norm: vector(
                &format!("{p}.post_attention_layernorm.weight"),
                &mut device_bytes,
            )?,
            w_gate: projection(&format!("{p}.mlp.gate_proj"), &mut device_bytes)?,
            w_up: projection(&format!("{p}.mlp.up_proj"), &mut device_bytes)?,
            w_qkv: stacked3(
                &format!("{p}.self_attn.q_proj"),
                &format!("{p}.self_attn.k_proj"),
                &format!("{p}.self_attn.v_proj"),
                &mut device_bytes,
            )?,
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
