//! Qwen3.8-27B on an Apple GPU: the hybrid GatedDeltaNet architecture, Q4_K_M.
//!
//! A vertical slice, like `qwen2_f16.rs` and for the same reason -- it proves
//! the kernels and the architecture, not the engine. No scheduler, no paged KV
//! pool, no batching; prefill walks the prompt one token at a time through the
//! decode path so nothing needs a GEMM.
//!
//! ```text
//! cargo run --release -p tuili-metal --example qwen38_27b -- \
//!     models/Qwen3.8-27B-Q4_K_M.gguf --prompt "..."
//! ```
//!
//! 64 blocks, of which 48 are GatedDeltaNet and 16 (3, 7, ... 63) are gated
//! full attention. What each layout choice cost to get right is recorded in
//! `notes/qwen3.5-architecture.md`; every one of them has a second reading that
//! runs to completion and produces fluent nonsense.

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use tuili_gguf::{GgmlType, Gguf};
use tuili_metal::{Buf, Device, Function, LaunchConfig, View, ViewMut};
use tuili_tokenizer::Tokenizer;

const COMMON: &str = include_str!("../../kernels/src/msl/common.metal");
const OPS: &str = include_str!("../../kernels/src/msl/ops.metal");
const QUANT: &str = include_str!("../../kernels/src/msl/quant.metal");
const GDN: &str = include_str!("../../kernels/src/msl/gdn.metal");

const BLOCK: u32 = 256;
const REDUCE_BLOCK: u32 = 256;

fn ops_src() -> String {
    format!("{COMMON}\n{OPS}")
}
fn quant_src() -> String {
    format!("{COMMON}\n{QUANT}")
}
fn gdn_src() -> String {
    format!("{COMMON}\n{GDN}")
}

fn grid1(n: u32, block: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(block).max(1), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn one_row(rows: u32, block: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

// ---- config --------------------------------------------------------------

struct Cfg {
    n_layers: usize,
    d_model: usize,
    n_heads: usize,
    n_kv: usize,
    d_head: usize,
    d_attn: usize,
    d_ff: usize,
    vocab: usize,
    eps: f32,
    theta: f64,
    rotary_dim: usize,
    // GatedDeltaNet
    lin_key_heads: usize,
    lin_value_heads: usize,
    lin_dk: usize,
    conv_k: usize,
    qkv_width: usize,
    z_width: usize,
}

impl Cfg {
    fn read(g: &Gguf) -> Result<Self> {
        let a = g.arch()?.to_string();
        let k = |s: &str| format!("{a}.{s}");
        let d_model = g.usize(&k("embedding_length"))?;
        let n_heads = g.usize(&k("attention.head_count"))?;
        let d_head = g.usize(&k("attention.key_length")).unwrap_or(d_model / n_heads);
        let lin_key_heads = g.usize(&k("ssm.group_count"))?;
        let lin_value_heads = g.usize(&k("ssm.time_step_rank"))?;
        let lin_dk = g.usize(&k("ssm.state_size"))?;
        Ok(Self {
            n_layers: g.usize(&k("block_count"))?,
            d_model,
            n_heads,
            n_kv: g.usize(&k("attention.head_count_kv"))?,
            d_head,
            d_attn: n_heads * d_head,
            d_ff: g.usize(&k("feed_forward_length"))?,
            vocab: g.tensor("token_embd.weight")?.dims[1] as usize,
            eps: g.f32(&k("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6),
            theta: g.f32(&k("rope.freq_base")).unwrap_or(1e7) as f64,
            rotary_dim: g.usize(&k("rope.dimension_count")).unwrap_or(d_head / 4),
            lin_key_heads,
            lin_value_heads,
            lin_dk,
            conv_k: g.usize(&k("ssm.conv_kernel"))?,
            qkv_width: 2 * lin_key_heads * lin_dk + lin_value_heads * lin_dk,
            z_width: g.usize(&k("ssm.inner_size"))?,
        })
    }
}

// ---- weights -------------------------------------------------------------

/// A quantized matrix left in its GGUF block encoding on the device. `k` is the
/// contraction (ggml's `dims[0]`, the fastest axis) and `n` the output width.
struct QMat {
    w: Buf<u8>,
    ty: GgmlType,
    k: usize,
    n: usize,
}

fn qmat(dev: &Device, g: &Gguf, name: &str) -> Result<QMat> {
    let t = g.tensor(name)?;
    let bytes = g.data(t);
    Ok(QMat {
        w: dev.stream().memcpy_stod(bytes)?,
        ty: t.ty,
        k: t.dims[0] as usize,
        n: t.dims[1] as usize,
    })
}

fn f32vec(dev: &Device, g: &Gguf, name: &str) -> Result<Buf<f32>> {
    f32vec_off(dev, g, name, 0.0)
}

/// Qwen3.5 has two RMSNorm classes: `Qwen3_5RMSNorm` initializes to zeros and
/// computes `normalized * (1 + weight)`, while `Qwen3_5RMSNormGated` initializes
/// to ones and computes `weight * normalized`. Loading a Hugging Face
/// checkpoint means adding that one for the first class -- see
/// `weights.rs::norm_gain_offset` on the CUDA side, which does exactly that.
///
/// **A GGUF needs none of it.** llama.cpp's converter folds the one in at
/// conversion time, and the file was checked rather than assumed:
///
/// ```text
///                        capture (HF)          this GGUF
///   attn_norm             mean -0.0334         mean  0.9666
///   attn_q_norm           mean  0.2304         mean  1.2304
/// ```
///
/// Adding it again scales every normed activation by roughly two and inverts
/// nothing, which is a model that runs at plausible magnitudes -- the residual
/// norms grew smoothly from 0.15 to 21.9 across 64 blocks -- and emits noise.
/// That is what it did before this comment existed.
fn norm_gguf(dev: &Device, g: &Gguf, name: &str) -> Result<Buf<f32>> {
    f32vec_off(dev, g, name, 0.0)
}

/// `ssm_a` is not `A_log`.
///
/// The reference computes the decay as `-exp(A_log) * softplus(...)`, and the
/// GGUF stores `-exp(A_log)` already: this file's `blk.0.ssm_a` spans
/// [-0.3376, -0.0038], which is exactly `-exp` of the capture's `A_log` span
/// [-5.5625, -1.0859]. Taking `-exp` again turns a decay of 0.03 into one of
/// 0.97, so the state stops decaying and every linear block remembers
/// everything.
///
/// Inverting it here rather than adding a kernel variant keeps
/// `gdn_gate_decay_f32` identical to its CUDA twin, which is the same trade the
/// norm offset makes on the safetensors path.
fn ssm_a_log(dev: &Device, g: &Gguf, name: &str) -> Result<Buf<f32>> {
    let t = g.tensor(name)?;
    anyhow::ensure!(t.ty == GgmlType::F32, "{name} is {:?}, wanted F32", t.ty);
    let v: Vec<f32> = g
        .data(t)
        .chunks_exact(4)
        .map(|b| {
            let a = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            debug_assert!(a < 0.0, "{name}: {a} is not -exp(A_log)");
            (-a).ln()
        })
        .collect();
    dev.stream().memcpy_stod(&v)
}

fn f32vec_off(dev: &Device, g: &Gguf, name: &str, offset: f32) -> Result<Buf<f32>> {
    let t = g.tensor(name)?;
    anyhow::ensure!(t.ty == GgmlType::F32, "{name} is {:?}, wanted F32", t.ty);
    let raw = g.data(t);
    let v: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) + offset)
        .collect();
    dev.stream().memcpy_stod(&v)
}

struct Ffn {
    norm: Buf<f32>,
    gate: QMat,
    up: QMat,
    down: QMat,
}

struct Gdn {
    norm: Buf<f32>,
    qkv: QMat,
    z: QMat,
    alpha: QMat,
    beta: QMat,
    a_log: Buf<f32>,
    dt_bias: Buf<f32>,
    conv_w: Buf<f32>,
    out_norm: Buf<f32>,
    out: QMat,
}

struct Full {
    norm: Buf<f32>,
    q: QMat,
    k: QMat,
    v: QMat,
    q_norm: Buf<f32>,
    k_norm: Buf<f32>,
    out: QMat,
}

enum Mixer {
    Gdn(Gdn),
    Full(Full),
}

struct Layer {
    mixer: Mixer,
    ffn: Ffn,
}

struct Weights {
    embd: QMat,
    out_norm: Buf<f32>,
    head: QMat,
    layers: Vec<Layer>,
}

impl Weights {
    fn load(dev: &Device, g: &Gguf, cfg: &Cfg) -> Result<Self> {
        let started = std::time::Instant::now();
        let embd = qmat(dev, g, "token_embd.weight")?;
        let out_norm = norm_gguf(dev, g, "output_norm.weight")?;
        let head = if g.get_tensor("output.weight").is_some() {
            qmat(dev, g, "output.weight")?
        } else {
            qmat(dev, g, "token_embd.weight")?
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut n_gdn = 0usize;
        for i in 0..cfg.n_layers {
            let t = |s: &str| format!("blk.{i}.{s}");
            // Which mixer a block has is read off the file rather than inferred
            // from `full_attention_interval`: the reference stores an explicit
            // 64-element `layer_types`, and a block that carries `ssm_a` is a
            // linear one whatever the interval says.
            let mixer = if g.get_tensor(&t("ssm_a")).is_some() {
                n_gdn += 1;
                Mixer::Gdn(Gdn {
                    norm: norm_gguf(dev, g, &t("attn_norm.weight"))?,
                    qkv: qmat(dev, g, &t("attn_qkv.weight"))?,
                    z: qmat(dev, g, &t("attn_gate.weight"))?,
                    alpha: qmat(dev, g, &t("ssm_alpha.weight"))?,
                    beta: qmat(dev, g, &t("ssm_beta.weight"))?,
                    a_log: ssm_a_log(dev, g, &t("ssm_a"))?,
                    dt_bias: f32vec(dev, g, &t("ssm_dt.bias"))?,
                    conv_w: f32vec(dev, g, &t("ssm_conv1d.weight"))?,
                    out_norm: norm_gguf(dev, g, &t("ssm_norm.weight"))?,
                    out: qmat(dev, g, &t("ssm_out.weight"))?,
                })
            } else {
                Mixer::Full(Full {
                    norm: norm_gguf(dev, g, &t("attn_norm.weight"))?,
                    q: qmat(dev, g, &t("attn_q.weight"))?,
                    k: qmat(dev, g, &t("attn_k.weight"))?,
                    v: qmat(dev, g, &t("attn_v.weight"))?,
                    q_norm: norm_gguf(dev, g, &t("attn_q_norm.weight"))?,
                    k_norm: norm_gguf(dev, g, &t("attn_k_norm.weight"))?,
                    out: qmat(dev, g, &t("attn_output.weight"))?,
                })
            };
            layers.push(Layer {
                mixer,
                ffn: Ffn {
                    norm: norm_gguf(dev, g, &t("post_attention_norm.weight"))?,
                    gate: qmat(dev, g, &t("ffn_gate.weight"))?,
                    up: qmat(dev, g, &t("ffn_up.weight"))?,
                    down: qmat(dev, g, &t("ffn_down.weight"))?,
                },
            });
        }
        eprintln!(
            "weights uploaded in {:.1} s  ({n_gdn} linear blocks, {} full attention)",
            started.elapsed().as_secs_f64(),
            cfg.n_layers - n_gdn
        );
        Ok(Self {
            embd,
            out_norm,
            head,
            layers,
        })
    }
}

// ---- session -------------------------------------------------------------

struct Session {
    x: Buf<f32>,
    xb: Buf<f32>,
    // full attention
    qg: Buf<f32>,
    q: Buf<f32>,
    gate: Buf<f32>,
    k: Buf<f32>,
    v: Buf<f32>,
    attn: Buf<f32>,
    kcache: Buf<half::f16>,
    vcache: Buf<half::f16>,
    // GatedDeltaNet
    qkv: Buf<f32>,
    z: Buf<f32>,
    a: Buf<f32>,
    b: Buf<f32>,
    beta: Buf<f32>,
    g: Buf<f32>,
    gdn_out: Buf<f32>,
    gdn_norm: Buf<f32>,
    state: Buf<f32>,
    conv_state: Buf<f32>,
    // shared
    proj: Buf<f32>,
    ff: Buf<f32>,
    ff_out: Buf<f32>,
    logits: Buf<f32>,
    ids: Buf<i32>,
    zero: Buf<i32>,
    one: Buf<i32>,
    cos: Buf<f32>,
    sin: Buf<f32>,
    max_pos: usize,
    n_full: usize,
    n_gdn: usize,
}

impl Session {
    fn new(dev: &Device, cfg: &Cfg, w: &Weights, max_pos: usize) -> Result<Self> {
        let s = dev.stream();
        let n_full = w
            .layers
            .iter()
            .filter(|l| matches!(l.mixer, Mixer::Full(_)))
            .count();
        let n_gdn = cfg.n_layers - n_full;
        let kv = cfg.n_kv * cfg.d_head;
        let dv = cfg.lin_dk;
        Ok(Self {
            x: s.alloc_zeros(cfg.d_model)?,
            xb: s.alloc_zeros(cfg.d_model)?,
            qg: s.alloc_zeros(2 * cfg.d_attn)?,
            q: s.alloc_zeros(cfg.d_attn)?,
            gate: s.alloc_zeros(cfg.d_attn)?,
            k: s.alloc_zeros(kv)?,
            v: s.alloc_zeros(kv)?,
            attn: s.alloc_zeros(cfg.d_attn)?,
            // One plane a full-attention block. Sharing one plane across
            // blocks is the bug the 0.5B slice shipped with for an hour.
            kcache: s.alloc_zeros(n_full * max_pos * kv)?,
            vcache: s.alloc_zeros(n_full * max_pos * kv)?,
            qkv: s.alloc_zeros(cfg.qkv_width)?,
            z: s.alloc_zeros(cfg.z_width)?,
            a: s.alloc_zeros(cfg.lin_value_heads)?,
            b: s.alloc_zeros(cfg.lin_value_heads)?,
            beta: s.alloc_zeros(cfg.lin_value_heads)?,
            g: s.alloc_zeros(cfg.lin_value_heads)?,
            gdn_out: s.alloc_zeros(cfg.lin_value_heads * dv)?,
            gdn_norm: s.alloc_zeros(cfg.lin_value_heads * dv)?,
            // 48 heads x 128 x 128 f32 = 3 MiB a linear block, 147 MiB total.
            // Fixed: it does not grow with the sequence.
            state: s.alloc_zeros(n_gdn * cfg.lin_value_heads * dv * dv)?,
            conv_state: s.alloc_zeros(n_gdn * cfg.qkv_width * (cfg.conv_k - 1))?,
            proj: s.alloc_zeros(cfg.d_model)?,
            ff: s.alloc_zeros(2 * cfg.d_ff)?,
            ff_out: s.alloc_zeros(cfg.d_ff)?,
            logits: s.alloc_zeros(cfg.vocab)?,
            ids: s.alloc_zeros(1)?,
            zero: s.memcpy_stod(&[0i32])?,
            one: s.memcpy_stod(&[1i32])?,
            cos: s.alloc_zeros(cfg.rotary_dim)?,
            sin: s.alloc_zeros(cfg.rotary_dim)?,
            max_pos,
            n_full,
            n_gdn,
        })
    }
}

// ---- engine --------------------------------------------------------------

struct Engine {
    dev: Device,
    cfg: Cfg,
    w: Weights,
    ops: String,
    quant: String,
    gdn: String,
}

impl Engine {
    fn f(&self, module: &'static str, name: &str) -> Result<Function> {
        let src = match module {
            "tuili_ops" => &self.ops,
            "tuili_quant" => &self.quant,
            _ => &self.gdn,
        };
        self.dev.kernels().get(module, src, name)
    }

    /// `out = W x` for a weight still in its GGUF encoding.
    fn gemv(&self, out: &mut ViewMut<'_, f32>, m: &QMat, x: &View<'_, f32>) -> Result<()> {
        let name = match m.ty {
            GgmlType::F32 => "gemv_f32_q",
            GgmlType::F16 => "gemv_f16_q",
            GgmlType::Q8_0 => "gemv_q8_0",
            GgmlType::Q4K => "gemv_q4_K",
            GgmlType::Q6K => "gemv_q6_K",
            other => return Err(anyhow!("no Metal mat-vec for {other:?}")),
        };
        let f = self.f("tuili_quant", name)?;
        let (k, n, t) = (m.k as i32, m.n as i32, 1i32);
        let s = self.dev.stream();
        let mut b = s.launch_builder(&f);
        b.arg(out)
            .arg(&m.w.as_view())
            .arg(x)
            .arg(&k)
            .arg(&n)
            .arg(&t);
        unsafe { b.launch(one_row(m.n as u32, REDUCE_BLOCK)) }
    }

    fn rms_norm(
        &self,
        out: &mut ViewMut<'_, f32>,
        x: &View<'_, f32>,
        w: &View<'_, f32>,
        d: usize,
        rows: u32,
    ) -> Result<()> {
        let f = self.f("tuili_ops", "rms_norm_f32")?;
        let (d_i, eps) = (d as i32, self.cfg.eps);
        let s = self.dev.stream();
        let mut b = s.launch_builder(&f);
        b.arg(out).arg(x).arg(w).arg(&d_i).arg(&eps);
        unsafe { b.launch(one_row(rows, REDUCE_BLOCK)) }
    }

    fn add_assign(&self, out: &mut ViewMut<'_, f32>, v: &View<'_, f32>, n: usize) -> Result<()> {
        let f = self.f("tuili_ops", "add_assign_f32")?;
        let n_i = n as i32;
        let s = self.dev.stream();
        let mut b = s.launch_builder(&f);
        b.arg(out).arg(v).arg(&n_i);
        unsafe { b.launch(grid1(n as u32, BLOCK)) }
    }

    fn ffn(&self, sess: &mut Session, l: &Ffn) -> Result<()> {
        let cfg = &self.cfg;
        self.rms_norm(
            &mut sess.xb.as_view_mut(),
            &sess.x.as_view(),
            &l.norm.as_view(),
            cfg.d_model,
            1,
        )?;
        self.gemv(
            &mut sess.ff.slice_mut(..cfg.d_ff),
            &l.gate,
            &sess.xb.as_view(),
        )?;
        self.gemv(
            &mut sess.ff.slice_mut(cfg.d_ff..2 * cfg.d_ff),
            &l.up,
            &sess.xb.as_view(),
        )?;
        {
            let f = self.f("tuili_ops", "silu_mul_split_f32")?;
            let (dff, total) = (cfg.d_ff as i32, cfg.d_ff as i32);
            let s = self.dev.stream();
            let mut b = s.launch_builder(&f);
            b.arg(&sess.ff_out.as_view_mut())
                .arg(&sess.ff.as_view())
                .arg(&dff)
                .arg(&total);
            unsafe { b.launch(grid1(cfg.d_ff as u32, BLOCK))? };
        }
        self.gemv(
            &mut sess.proj.as_view_mut(),
            &l.down,
            &sess.ff_out.as_view(),
        )?;
        self.add_assign(
            &mut sess.x.as_view_mut(),
            &sess.proj.as_view(),
            cfg.d_model,
        )
    }

    fn full_attention(&self, sess: &mut Session, l: &Full, slot: usize, pos: usize) -> Result<()> {
        let cfg = &self.cfg;
        let s = self.dev.stream();
        self.rms_norm(
            &mut sess.xb.as_view_mut(),
            &sess.x.as_view(),
            &l.norm.as_view(),
            cfg.d_model,
            1,
        )?;

        // q_proj is [d_model, heads * 2 * head_dim]: within one head the query
        // comes first and its gate second. Reading it as [all q | all gates]
        // also runs, and gives a different model.
        self.gemv(&mut sess.qg.as_view_mut(), &l.q, &sess.xb.as_view())?;
        {
            let f = self.f("tuili_gdn", "split_interleaved_f32")?;
            let (heads, dh) = (cfg.n_heads as i32, cfg.d_head as i32);
            let n = cfg.d_attn as i64;
            let mut b = s.launch_builder(&f);
            b.arg(&sess.q.as_view_mut())
                .arg(&sess.gate.as_view_mut())
                .arg(&sess.qg.as_view())
                .arg(&heads)
                .arg(&dh)
                .arg(&n);
            unsafe { b.launch(grid1(cfg.d_attn as u32, BLOCK))? };
        }
        self.gemv(&mut sess.k.as_view_mut(), &l.k, &sess.xb.as_view())?;
        self.gemv(&mut sess.v.as_view_mut(), &l.v, &sess.xb.as_view())?;

        // Per-head RMSNorm over head_dim, before RoPE.
        for (buf, w, heads) in [
            (&mut sess.q, &l.q_norm, cfg.n_heads),
            (&mut sess.k, &l.k_norm, cfg.n_kv),
        ] {
            let f = self.f("tuili_ops", "qk_norm_f32")?;
            let (h, dh) = (heads as i32, cfg.d_head as i32);
            let (stride, off, eps) = ((heads * cfg.d_head) as i32, 0i32, cfg.eps);
            let mut b = s.launch_builder(&f);
            b.arg(&buf.as_view_mut())
                .arg(&w.as_view())
                .arg(&h)
                .arg(&dh)
                .arg(&stride)
                .arg(&off)
                .arg(&eps);
            unsafe { b.launch(one_row(heads as u32, REDUCE_BLOCK))? };
        }

        // Partial rotary: the first 64 of each 256-wide head rotate.
        for (buf, heads) in [(&mut sess.q, cfg.n_heads), (&mut sess.k, cfg.n_kv)] {
            let f = self.f("tuili_ops", "rope_partial_f32")?;
            let (h, dh, rot) = (heads as i32, cfg.d_head as i32, cfg.rotary_dim as i32);
            let mut b = s.launch_builder(&f);
            b.arg(&buf.as_view_mut())
                .arg(&sess.cos.as_view())
                .arg(&sess.sin.as_view())
                .arg(&h)
                .arg(&dh)
                .arg(&rot);
            let half = (cfg.rotary_dim / 2) as u32;
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (1, heads as u32, 1),
                    block_dim: (half, 1, 1),
                    shared_mem_bytes: 0,
                })?
            };
        }

        let kv = cfg.n_kv * cfg.d_head;
        let plane = sess.max_pos * kv;
        let (lo, hi) = (slot * plane, (slot + 1) * plane);
        {
            let f = self.f("tuili_ops", "store_kv_contig_f16")?;
            let (nkv, dh, p) = (cfg.n_kv as i32, cfg.d_head as i32, pos as i32);
            let mut b = s.launch_builder(&f);
            b.arg(&sess.kcache.slice_mut(lo..hi))
                .arg(&sess.vcache.slice_mut(lo..hi))
                .arg(&sess.k.as_view())
                .arg(&sess.v.as_view())
                .arg(&nkv)
                .arg(&dh)
                .arg(&p);
            unsafe { b.launch(grid1(kv as u32, BLOCK))? };
        }
        {
            let f = self.f("tuili_ops", "attn_decode_f32")?;
            let (nh, nkv, dh) = (cfg.n_heads as i32, cfg.n_kv as i32, cfg.d_head as i32);
            let kv_len = (pos + 1) as i32;
            let scale = 1.0f32 / (cfg.d_head as f32).sqrt();
            let mut b = s.launch_builder(&f);
            b.arg(&sess.attn.as_view_mut())
                .arg(&sess.q.as_view())
                .arg(&sess.kcache.slice(lo..hi))
                .arg(&sess.vcache.slice(lo..hi))
                .arg(&nh)
                .arg(&nkv)
                .arg(&dh)
                .arg(&kv_len)
                .arg(&scale);
            unsafe { b.launch(one_row(cfg.n_heads as u32, REDUCE_BLOCK))? };
        }
        // The output gate, applied before o_proj, and sigmoid rather than SiLU.
        {
            let f = self.f("tuili_gdn", "sigmoid_gate_f32")?;
            let n = cfg.d_attn as i64;
            let mut b = s.launch_builder(&f);
            b.arg(&sess.attn.as_view_mut()).arg(&sess.gate.as_view()).arg(&n);
            unsafe { b.launch(grid1(cfg.d_attn as u32, BLOCK))? };
        }
        self.gemv(&mut sess.proj.as_view_mut(), &l.out, &sess.attn.as_view())?;
        self.add_assign(
            &mut sess.x.as_view_mut(),
            &sess.proj.as_view(),
            cfg.d_model,
        )
    }

    fn gdn_block(&self, sess: &mut Session, l: &Gdn, slot: usize) -> Result<()> {
        let cfg = &self.cfg;
        let s = self.dev.stream();
        let dv = cfg.lin_dk;
        self.rms_norm(
            &mut sess.xb.as_view_mut(),
            &sess.x.as_view(),
            &l.norm.as_view(),
            cfg.d_model,
            1,
        )?;
        self.gemv(&mut sess.qkv.as_view_mut(), &l.qkv, &sess.xb.as_view())?;
        self.gemv(&mut sess.z.as_view_mut(), &l.z, &sess.xb.as_view())?;
        // `ssm_alpha` is `in_proj_a`, which feeds the decay; `ssm_beta` is
        // `in_proj_b`, which feeds the update rate. Swapping them also runs.
        self.gemv(&mut sess.a.as_view_mut(), &l.alpha, &sess.xb.as_view())?;
        self.gemv(&mut sess.b.as_view_mut(), &l.beta, &sess.xb.as_view())?;

        // Depthwise causal conv over the whole [q | k | v] row, plus SiLU.
        {
            let cs_plane = cfg.qkv_width * (cfg.conv_k - 1);
            let (lo, hi) = (slot * cs_plane, (slot + 1) * cs_plane);
            let f = self.f("tuili_gdn", "gdn_conv_f32")?;
            let (ch, kk) = (cfg.qkv_width as i32, cfg.conv_k as i32);
            let mut b = s.launch_builder(&f);
            b.arg(&sess.qkv.as_view_mut())
                .arg(&sess.qkv.as_view())
                .arg(&sess.conv_state.slice_mut(lo..hi))
                .arg(&l.conv_w.as_view())
                .arg(&sess.zero.as_view())
                .arg(&sess.one.as_view())
                .arg(&ch)
                .arg(&kk);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: ((cfg.qkv_width as u32).div_ceil(BLOCK), 1, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })?
            };
        }
        {
            let f = self.f("tuili_gdn", "gdn_gate_decay_f32")?;
            let (nt, heads, stride) = (1i32, cfg.lin_value_heads as i32, cfg.lin_value_heads as i32);
            let mut b = s.launch_builder(&f);
            b.arg(&sess.beta.as_view_mut())
                .arg(&sess.g.as_view_mut())
                .arg(&sess.a.as_view())
                .arg(&sess.b.as_view())
                .arg(&l.a_log.as_view())
                .arg(&l.dt_bias.as_view())
                .arg(&nt)
                .arg(&heads)
                .arg(&stride);
            unsafe { b.launch(grid1(cfg.lin_value_heads as u32, BLOCK))? };
        }
        {
            let f = self.f("tuili_gdn", "gdn_qk_l2norm_f32")?;
            let (kh, dk) = (cfg.lin_key_heads as i32, cfg.lin_dk as i32);
            let stride = cfg.qkv_width as i32;
            let (q_off, k_off) = (0i32, (cfg.lin_key_heads * cfg.lin_dk) as i32);
            let (eps, qs) = (1e-6f32, 1.0f32 / (cfg.lin_dk as f32).sqrt());
            let mut b = s.launch_builder(&f);
            b.arg(&sess.qkv.as_view_mut())
                .arg(&kh)
                .arg(&dk)
                .arg(&stride)
                .arg(&q_off)
                .arg(&k_off)
                .arg(&eps)
                .arg(&qs);
            unsafe { b.launch(one_row(cfg.lin_key_heads as u32, REDUCE_BLOCK))? };
        }
        {
            let plane = cfg.lin_value_heads * dv * dv;
            let (lo, hi) = (slot * plane, (slot + 1) * plane);
            let f = self.f("tuili_gdn", "gdn_delta_rule_f32")?;
            let (heads, kh) = (cfg.lin_value_heads as i32, cfg.lin_key_heads as i32);
            let (dk, dvi) = (cfg.lin_dk as i32, dv as i32);
            let stride = cfg.qkv_width as i32;
            let kk = (cfg.lin_key_heads * cfg.lin_dk) as i32;
            let (q_off, k_off, v_off) = (0i32, kk, 2 * kk);
            // A GGUF's V heads are in tiled order; see the kernel's comment.
            let v_tiled = 1i32;
            let mut b = s.launch_builder(&f);
            b.arg(&sess.gdn_out.as_view_mut())
                .arg(&sess.state.slice_mut(lo..hi))
                .arg(&sess.qkv.as_view())
                .arg(&sess.g.as_view())
                .arg(&sess.beta.as_view())
                .arg(&sess.zero.as_view())
                .arg(&sess.one.as_view())
                .arg(&heads)
                .arg(&kh)
                .arg(&dk)
                .arg(&dvi)
                .arg(&stride)
                .arg(&q_off)
                .arg(&k_off)
                .arg(&v_off)
                .arg(&v_tiled);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (cfg.lin_value_heads as u32, 1, 1),
                    block_dim: (dv as u32, 1, 1),
                    // q and k for one token, staged in threadgroup memory.
                    shared_mem_bytes: (2 * cfg.lin_dk * 4) as u32,
                })?
            };
        }
        // RMSNorm then the SiLU'd gate, in that order: `weight * normalized`
        // first, and only then times `silu(z)`.
        {
            let f = self.f("tuili_gdn", "gdn_gated_rmsnorm_f32")?;
            let (dvi, eps) = (dv as i32, cfg.eps);
            let mut b = s.launch_builder(&f);
            b.arg(&sess.gdn_norm.as_view_mut())
                .arg(&sess.gdn_out.as_view())
                .arg(&sess.z.as_view())
                .arg(&l.out_norm.as_view())
                .arg(&dvi)
                .arg(&eps);
            unsafe { b.launch(one_row(cfg.lin_value_heads as u32, REDUCE_BLOCK))? };
        }
        self.gemv(&mut sess.proj.as_view_mut(), &l.out, &sess.gdn_norm.as_view())?;
        self.add_assign(
            &mut sess.x.as_view_mut(),
            &sess.proj.as_view(),
            cfg.d_model,
        )
    }

    /// The rope tables for one position, in f64.
    ///
    /// f32 is not enough: the two obvious f32 formulations of
    /// `theta^(-2i/rot)` differ by an ulp that becomes 2.5e-3 in the cosine at
    /// position 130000, because the angle there is ~7.9e4 and f32's ulp is
    /// 0.0078 rad. The table at this model's 262144-token context is simply not
    /// reproducible across implementations in f32.
    fn rope_tables(&self, pos: usize) -> (Vec<f32>, Vec<f32>) {
        let rot = self.cfg.rotary_dim;
        let half = rot / 2;
        let mut cos = vec![0.0f32; rot];
        let mut sin = vec![0.0f32; rot];
        for i in 0..half {
            let inv = self.cfg.theta.powf(-((2 * i) as f64 / rot as f64));
            let angle = pos as f64 * inv;
            let (s, c) = (angle.sin() as f32, angle.cos() as f32);
            cos[i] = c;
            cos[i + half] = c;
            sin[i] = s;
            sin[i + half] = s;
        }
        (cos, sin)
    }

    fn step(&self, sess: &mut Session, token: u32, pos: usize) -> Result<()> {
        let cfg = &self.cfg;
        let s = self.dev.stream();
        s.copy_into(&mut sess.ids.as_view_mut(), &[token as i32])?;
        let (cos, sin) = self.rope_tables(pos);
        s.copy_into(&mut sess.cos.as_view_mut(), &cos)?;
        s.copy_into(&mut sess.sin.as_view_mut(), &sin)?;

        {
            anyhow::ensure!(
                self.w.embd.ty == GgmlType::Q4K,
                "the embedding table is {:?}; only Q4_K has a row reader",
                self.w.embd.ty
            );
            let f = self.f("tuili_quant", "embed_row_q4_K")?;
            let d = cfg.d_model as i32;
            let mut b = s.launch_builder(&f);
            b.arg(&sess.x.as_view_mut())
                .arg(&self.w.embd.w.as_view())
                .arg(&sess.ids.as_view())
                .arg(&d);
            unsafe { b.launch(grid1(cfg.d_model as u32, BLOCK))? };
        }

        let trace = std::env::var("TUILI_TRACE").is_ok();
        let rms = |v: &[f32]| -> f32 {
            (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
        };

        let mut full_slot = 0usize;
        let mut gdn_slot = 0usize;
        for (li, l) in self.w.layers.iter().enumerate() {
            match &l.mixer {
                Mixer::Full(m) => {
                    self.full_attention(sess, m, full_slot, pos)?;
                    full_slot += 1;
                }
                Mixer::Gdn(m) => {
                    self.gdn_block(sess, m, gdn_slot)?;
                    gdn_slot += 1;
                }
            }
            if trace {
                s.synchronize()?;
                let mix = rms(&sess.x.to_vec());
                self.ffn(sess, &l.ffn)?;
                s.synchronize()?;
                eprintln!(
                    "  layer {li:2} {}  after mixer rms {mix:11.4}  after ffn rms {:11.4}",
                    if matches!(l.mixer, Mixer::Full(_)) { "full" } else { "gdn " },
                    rms(&sess.x.to_vec())
                );
            } else {
                self.ffn(sess, &l.ffn)?;
            }
        }

        self.rms_norm(
            &mut sess.xb.as_view_mut(),
            &sess.x.as_view(),
            &self.w.out_norm.as_view(),
            cfg.d_model,
            1,
        )?;
        self.gemv(&mut sess.logits.as_view_mut(), &self.w.head, &sess.xb.as_view())?;
        s.synchronize()
    }
}

/// Top-k by logit, for the diagnostic line.
fn top_k(v: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut ix: Vec<u32> = (0..v.len() as u32).collect();
    ix.sort_unstable_by(|&a, &b| v[b as usize].total_cmp(&v[a as usize]));
    ix.truncate(k);
    ix.into_iter().map(|i| (i, v[i as usize])).collect()
}

/// Nucleus sampling at a temperature.
///
/// Qwen recommends `temperature 0.7 / top_p 0.95 / top_k 20` for this model and
/// warns against greedy decoding, which degenerates into repetition -- so the
/// default here is the recommendation, not argmax. `--greedy` asks for argmax
/// anyway, which is what a reproducible comparison wants.
fn sample(v: &[f32], temp: f32, top_p: f32, top_k_n: usize, rng: &mut u64) -> u32 {
    if temp <= 0.0 {
        return argmax(v);
    }
    let mut cand = top_k(v, top_k_n);
    let max = cand[0].1;
    let mut probs: Vec<f32> = cand.iter().map(|(_, l)| ((l - max) / temp).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    // Nucleus: keep the shortest prefix whose mass reaches top_p.
    let mut acc = 0.0f32;
    let mut keep = probs.len();
    for (i, p) in probs.iter().enumerate() {
        acc += p;
        if acc >= top_p {
            keep = i + 1;
            break;
        }
    }
    cand.truncate(keep);
    probs.truncate(keep);
    let total: f32 = probs.iter().sum();
    *rng = rng
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut r = ((*rng >> 33) as f32 / (1u64 << 31) as f32) * total;
    for (i, p) in probs.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return cand[i].0;
        }
    }
    cand[keep - 1].0
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as u32
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = args
        .first()
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("usage: qwen38_27b <model.gguf> [--prompt <text>] [-n N]"))?;
    let prompt = args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "The capital of France is".to_string());
    let n_new: usize = args
        .iter()
        .position(|a| a == "-n")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(48);

    let g = Gguf::open(&model).with_context(|| format!("opening {}", model.display()))?;
    let cfg = Cfg::read(&g)?;
    let dev = Device::new(0)?;
    eprintln!(
        "{} | {:.1} GiB working set\n{} blocks, d_model {}, {} heads / {} kv, d_head {}, rotary {}, ffn {}, vocab {}\nGDN: {} value heads / {} key heads, dk {}, conv {}, qkv {}",
        dev.name(),
        dev.working_set_bytes() as f64 / (1u64 << 30) as f64,
        cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_kv, cfg.d_head,
        cfg.rotary_dim, cfg.d_ff, cfg.vocab,
        cfg.lin_value_heads, cfg.lin_key_heads, cfg.lin_dk, cfg.conv_k, cfg.qkv_width,
    );

    let tok = Tokenizer::from_gguf(&g)?;
    if args.iter().any(|a| a == "--cpu") {
        let ids = tok.encode(&prompt, Some(false), true);
        eprintln!("host reference over {} ids (slow) ...", ids.len());
        let t0 = std::time::Instant::now();
        let logits = host::forward(&g, &cfg, &ids)?;
        let top = argmax(&logits);
        let mean = logits.iter().sum::<f32>() / logits.len() as f32;
        let var = logits.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
            / (logits.len() - 1) as f32;
        println!(
            "  host argmax {top} {:?}  std {:.4}  ({:.1} s)",
            tok.id_to_piece(top).unwrap_or("?"),
            var.sqrt(),
            t0.elapsed().as_secs_f64()
        );
        return Ok(());
    }

    let w = Weights::load(&dev, &g, &cfg)?;
    let ids = tok.encode(&prompt, Some(false), true);
    let eng = Engine {
        dev,
        cfg,
        w,
        ops: ops_src(),
        quant: quant_src(),
        gdn: gdn_src(),
    };
    let mut sess = Session::new(&eng.dev, &eng.cfg, &eng.w, ids.len() + n_new + 1)?;
    eprintln!(
        "session: {} full-attention planes, {} linear states\n",
        sess.n_full, sess.n_gdn
    );

    print!("{prompt}");
    use std::io::Write;
    std::io::stdout().flush().ok();

    let t0 = std::time::Instant::now();
    for (p, &t) in ids.iter().enumerate() {
        eng.step(&mut sess, t, p)?;
    }
    let prefill = t0.elapsed();

    let mut det = tok.detokenizer();
    let t1 = std::time::Instant::now();
    let mut produced = 0usize;
    let mut logits = sess.logits.to_vec();

    let mean = logits.iter().sum::<f32>() / logits.len() as f32;
    let var =
        logits.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (logits.len() - 1) as f32;
    eprintln!("\nfirst prediction: std {:.4}", var.sqrt());
    for (i, (t, l)) in top_k(&logits, 10).iter().enumerate() {
        eprintln!(
            "  {i:2}. {t:>7} {:>16}  logit {l:8.3}",
            format!("{:?}", tok.id_to_piece(*t).unwrap_or("?")),
            l = l
        );
    }
    eprintln!();

    let greedy = args.iter().any(|a| a == "--greedy");
    let mut rng: u64 = 0x5eed;
    for p in ids.len()..ids.len() + n_new {
        let next = if greedy {
            argmax(&logits)
        } else {
            sample(&logits, 0.7, 0.95, 20, &mut rng)
        };
        if tok.is_eog(next) {
            break;
        }
        print!("{}", det.push(next));
        std::io::stdout().flush().ok();
        produced += 1;
        eng.step(&mut sess, next, p)?;
        logits = sess.logits.to_vec();
    }
    print!("{}", det.finish());
    println!();
    let d = t1.elapsed();
    eprintln!(
        "\nprefill {:>8.1} ms ({:.2} tok/s over {} tokens)\ndecode  {:>8.1} ms ({:.2} tok/s over {produced} tokens)",
        prefill.as_secs_f64() * 1e3,
        ids.len() as f64 / prefill.as_secs_f64(),
        ids.len(),
        d.as_secs_f64() * 1e3,
        produced as f64 / d.as_secs_f64(),
    );
    Ok(())
}

// ---- the host arbiter ----------------------------------------------------

/// The same forward pass on the CPU, reading the same tensors the same way.
///
/// This exists to answer one question when the GPU disagrees with reality: is
/// the *model* wrong -- a layout, an order, a convention -- or is the *wiring*
/// wrong? On the 0.5B slice this localised a shared-KV-plane bug in a single
/// run, after every per-kernel test had passed. Slow on purpose: it
/// dequantizes each row as it needs it and shares no code with the kernels.
mod host {
    use super::*;

    fn h16(b: &[u8]) -> f32 {
        half::f16::from_le_bytes([b[0], b[1]]).to_f32()
    }

    fn q4k_scale_min(q: &[u8], j: usize) -> (u8, u8) {
        if j < 4 {
            (q[j] & 63, q[j + 4] & 63)
        } else {
            (
                (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
                (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
            )
        }
    }

    /// One row of a tensor, dequantized. From ggml's `dequantize_row_*`.
    fn row(ty: GgmlType, raw: &[u8], k: usize, out: &mut Vec<f32>) {
        out.clear();
        match ty {
            GgmlType::F32 => {
                for c in raw.chunks_exact(4) {
                    out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                }
            }
            GgmlType::Q8_0 => {
                for blk in raw.chunks_exact(34) {
                    let d = h16(&blk[0..2]);
                    for i in 0..32 {
                        out.push(d * (blk[2 + i] as i8) as f32);
                    }
                }
            }
            GgmlType::Q4K => {
                for blk in raw.chunks_exact(144) {
                    let (d, dmin) = (h16(&blk[0..2]), h16(&blk[2..4]));
                    let (scales, qs) = (&blk[4..16], &blk[16..144]);
                    for chunk in 0..4 {
                        let q = &qs[chunk * 32..chunk * 32 + 32];
                        for half in 0..2 {
                            let (sc, m) = q4k_scale_min(scales, chunk * 2 + half);
                            let (d1, m1) = (d * sc as f32, dmin * m as f32);
                            for l in 0..32 {
                                let nib = if half == 1 { q[l] >> 4 } else { q[l] & 0xF };
                                out.push(d1 * nib as f32 - m1);
                            }
                        }
                    }
                }
            }
            GgmlType::Q6K => {
                for blk in raw.chunks_exact(210) {
                    let (ql, qh) = (&blk[0..128], &blk[128..192]);
                    let sc: Vec<i8> = blk[192..208].iter().map(|&b| b as i8).collect();
                    let d = h16(&blk[208..210]);
                    let mut y = [0.0f32; 256];
                    for n in 0..2 {
                        let (ql, qh, sc) = (&ql[n * 64..], &qh[n * 32..], &sc[n * 8..]);
                        for l in 0..32 {
                            let is = l / 16;
                            let hh = qh[l];
                            let q1 = ((ql[l] & 0xF) | (((hh >> 0) & 3) << 4)) as i32 - 32;
                            let q2 = ((ql[l + 32] & 0xF) | (((hh >> 2) & 3) << 4)) as i32 - 32;
                            let q3 = ((ql[l] >> 4) | (((hh >> 4) & 3) << 4)) as i32 - 32;
                            let q4 = ((ql[l + 32] >> 4) | (((hh >> 6) & 3) << 4)) as i32 - 32;
                            y[n * 128 + l] = d * sc[is] as f32 * q1 as f32;
                            y[n * 128 + l + 32] = d * sc[is + 2] as f32 * q2 as f32;
                            y[n * 128 + l + 64] = d * sc[is + 4] as f32 * q3 as f32;
                            y[n * 128 + l + 96] = d * sc[is + 6] as f32 * q4 as f32;
                        }
                    }
                    out.extend_from_slice(&y);
                }
            }
            other => panic!("no host dequant for {other:?}"),
        }
        out.truncate(k);
    }

    struct T<'a> {
        raw: &'a [u8],
        ty: GgmlType,
        k: usize,
        n: usize,
        row_bytes: usize,
    }

    fn t<'a>(g: &'a Gguf, name: &str) -> Result<T<'a>> {
        let i = g.tensor(name)?;
        let n = i.dims[1] as usize;
        Ok(T {
            raw: g.data(i),
            ty: i.ty,
            k: i.dims[0] as usize,
            n,
            row_bytes: i.n_bytes / n,
        })
    }

    fn mv(m: &T<'_>, x: &[f32], out: &mut Vec<f32>) {
        out.clear();
        let mut w = Vec::with_capacity(m.k);
        for r in 0..m.n {
            row(m.ty, &m.raw[r * m.row_bytes..(r + 1) * m.row_bytes], m.k, &mut w);
            out.push((0..m.k).map(|i| w[i] * x[i]).sum());
        }
    }

    fn vecf(g: &Gguf, name: &str, offset: f32) -> Result<Vec<f32>> {
        let i = g.tensor(name)?;
        Ok(g.data(i)
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) + offset)
            .collect())
    }

    fn rms(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let ss: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let s = 1.0 / ((ss / x.len() as f64) + eps as f64).sqrt();
        x.iter().zip(w).map(|(a, b)| (*a as f64 * s) as f32 * b).collect()
    }

    fn rms_rows(x: &[f32], w: &[f32], dv: usize, eps: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(x.len());
        for r in x.chunks(dv) {
            out.extend_from_slice(&rms(r, w, eps));
        }
        out
    }

    pub fn forward(g: &Gguf, cfg: &Cfg, ids: &[u32]) -> Result<Vec<f32>> {
        let embd = t(g, "token_embd.weight")?;
        let onorm = vecf(g, "output_norm.weight", 0.0)?;
        let head = t(g, if g.get_tensor("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        })?;

        let dv = cfg.lin_dk;
        let kvd = cfg.n_kv * cfg.d_head;
        // Per-layer recurrent and KV state.
        let mut gstate: Vec<Vec<f32>> = vec![Vec::new(); cfg.n_layers];
        let mut cstate: Vec<Vec<f32>> = vec![Vec::new(); cfg.n_layers];
        let mut kc: Vec<Vec<f32>> = vec![Vec::new(); cfg.n_layers];
        let mut vc: Vec<Vec<f32>> = vec![Vec::new(); cfg.n_layers];
        let mut logits = Vec::new();
        let mut buf = Vec::new();

        for (pos, &tok) in ids.iter().enumerate() {
            let mut x = Vec::new();
            row(
                embd.ty,
                &embd.raw[tok as usize * embd.row_bytes..(tok as usize + 1) * embd.row_bytes],
                embd.k,
                &mut x,
            );

            for li in 0..cfg.n_layers {
                let tn = |s: &str| format!("blk.{li}.{s}");
                let anorm = vecf(g, &tn("attn_norm.weight"), 0.0)?;
                let xb = rms(&x, &anorm, cfg.eps);

                let is_gdn = g.get_tensor(&tn("ssm_a")).is_some();
                let mixed: Vec<f32> = if is_gdn {
                    let qkvw = t(g, &tn("attn_qkv.weight"))?;
                    let zw = t(g, &tn("attn_gate.weight"))?;
                    let aw = t(g, &tn("ssm_alpha.weight"))?;
                    let bw = t(g, &tn("ssm_beta.weight"))?;
                    // `ssm_a` holds -exp(A_log); the recurrence wants A_log.
                    let a_log: Vec<f32> =
                        vecf(g, &tn("ssm_a"), 0.0)?.iter().map(|a| (-a).ln()).collect();
                    let dtb = vecf(g, &tn("ssm_dt.bias"), 0.0)?;
                    let cw = vecf(g, &tn("ssm_conv1d.weight"), 0.0)?;
                    let nw = vecf(g, &tn("ssm_norm.weight"), 0.0)?;
                    let ow = t(g, &tn("ssm_out.weight"))?;

                    let mut qkv = Vec::new();
                    mv(&qkvw, &xb, &mut qkv);
                    let mut z = Vec::new();
                    mv(&zw, &xb, &mut z);
                    let mut a = Vec::new();
                    mv(&aw, &xb, &mut a);
                    let mut b = Vec::new();
                    mv(&bw, &xb, &mut b);

                    // depthwise causal conv + SiLU, with a carried window
                    let hist = cfg.conv_k - 1;
                    let st = cstate[li].get_or_insert_zeros(cfg.qkv_width * hist);
                    for c in 0..cfg.qkv_width {
                        let mut acc = cw[c * cfg.conv_k + hist] * qkv[c];
                        for j in 0..hist {
                            acc += cw[c * cfg.conv_k + j] * st[c * hist + j];
                        }
                        let cur = qkv[c];
                        for j in 0..hist.saturating_sub(1) {
                            st[c * hist + j] = st[c * hist + j + 1];
                        }
                        if hist > 0 {
                            st[c * hist + hist - 1] = cur;
                        }
                        qkv[c] = acc / (1.0 + (-acc).exp());
                    }

                    let heads = cfg.lin_value_heads;
                    let kh = cfg.lin_key_heads;
                    let beta: Vec<f32> = (0..heads).map(|h| 1.0 / (1.0 + (-b[h]).exp())).collect();
                    let gg: Vec<f32> = (0..heads)
                        .map(|h| {
                            let zz = a[h] + dtb[h];
                            let sp = if zz > 20.0 { zz } else { (zz.exp()).ln_1p() };
                            -(a_log[h].exp()) * sp
                        })
                        .collect();

                    // l2norm q and k in place, scale q
                    let qs = 1.0 / (cfg.lin_dk as f32).sqrt();
                    for hh in 0..kh {
                        for (off, scale) in [(0usize, qs), (kh * cfg.lin_dk, 1.0)] {
                            let base = off + hh * cfg.lin_dk;
                            let ss: f32 = (0..cfg.lin_dk).map(|i| qkv[base + i] * qkv[base + i]).sum();
                            let inv = (ss + 1e-6).sqrt().recip() * scale;
                            for i in 0..cfg.lin_dk {
                                qkv[base + i] *= inv;
                            }
                        }
                    }

                    let s = gstate[li].get_or_insert_zeros(heads * cfg.lin_dk * dv);
                    let mut core = vec![0.0f32; heads * dv];
                    let (q_off, k_off, v_off) =
                        (0usize, kh * cfg.lin_dk, 2 * kh * cfg.lin_dk);
                    for hd in 0..heads {
                        let khead = hd % kh;   // GGUF stores V heads tiled
                        let decay = gg[hd].exp();
                        let bb = beta[hd];
                        let sb = hd * cfg.lin_dk * dv;
                        for j in 0..dv {
                            let vtj = qkv[v_off + hd * dv + j];
                            let mut kvm = 0.0f32;
                            for i in 0..cfg.lin_dk {
                                let sv = s[sb + i * dv + j] * decay;
                                s[sb + i * dv + j] = sv;
                                kvm += sv * qkv[k_off + khead * cfg.lin_dk + i];
                            }
                            let delta = (vtj - kvm) * bb;
                            let mut o = 0.0f32;
                            for i in 0..cfg.lin_dk {
                                let sv = s[sb + i * dv + j]
                                    + qkv[k_off + khead * cfg.lin_dk + i] * delta;
                                s[sb + i * dv + j] = sv;
                                o += sv * qkv[q_off + khead * cfg.lin_dk + i];
                            }
                            core[hd * dv + j] = o;
                        }
                    }
                    // RMSNorm then the SiLU'd gate, in that order.
                    let normed = rms_rows(&core, &nw, dv, cfg.eps);
                    let gated: Vec<f32> = normed
                        .iter()
                        .zip(&z)
                        .map(|(v, zi)| v * (zi / (1.0 + (-zi).exp())))
                        .collect();
                    mv(&ow, &gated, &mut buf);
                    buf.clone()
                } else {
                    let qw = t(g, &tn("attn_q.weight"))?;
                    let kw = t(g, &tn("attn_k.weight"))?;
                    let vw = t(g, &tn("attn_v.weight"))?;
                    let qn = vecf(g, &tn("attn_q_norm.weight"), 0.0)?;
                    let kn = vecf(g, &tn("attn_k_norm.weight"), 0.0)?;
                    let ow = t(g, &tn("attn_output.weight"))?;

                    let mut qg = Vec::new();
                    mv(&qw, &xb, &mut qg);
                    // Per head, the query first and its gate second.
                    let (dh, nh) = (cfg.d_head, cfg.n_heads);
                    let mut q = vec![0.0f32; nh * dh];
                    let mut gate = vec![0.0f32; nh * dh];
                    for hd in 0..nh {
                        for i in 0..dh {
                            q[hd * dh + i] = qg[hd * 2 * dh + i];
                            gate[hd * dh + i] = qg[hd * 2 * dh + dh + i];
                        }
                    }
                    let mut k = Vec::new();
                    mv(&kw, &xb, &mut k);
                    let mut v = Vec::new();
                    mv(&vw, &xb, &mut v);
                    let mut q = rms_rows(&q, &qn, dh, cfg.eps);
                    let mut k = rms_rows(&k, &kn, dh, cfg.eps);

                    // partial rope
                    let rot = cfg.rotary_dim;
                    let half = rot / 2;
                    let tabs: Vec<(f32, f32)> = (0..half)
                        .map(|i| {
                            let inv = cfg.theta.powf(-((2 * i) as f64 / rot as f64));
                            let ang = pos as f64 * inv;
                            (ang.cos() as f32, ang.sin() as f32)
                        })
                        .collect();
                    for (buf2, heads) in [(&mut q, nh), (&mut k, cfg.n_kv)] {
                        for hd in 0..heads {
                            for i in 0..half {
                                let (c0, s0) = tabs[i];
                                let base = hd * dh;
                                let (aa, bb) = (buf2[base + i], buf2[base + i + half]);
                                buf2[base + i] = aa * c0 - bb * s0;
                                buf2[base + i + half] = bb * c0 + aa * s0;
                            }
                        }
                    }
                    kc[li].extend_from_slice(&k);
                    vc[li].extend_from_slice(&v);

                    let kv_len = pos + 1;
                    let group = nh / cfg.n_kv;
                    let scale = 1.0 / (dh as f32).sqrt();
                    let mut attn = vec![0.0f32; nh * dh];
                    for hd in 0..nh {
                        let kvh = hd / group;
                        let mut sc = vec![0.0f32; kv_len];
                        for j in 0..kv_len {
                            let mut d = 0.0f32;
                            for i in 0..dh {
                                d += q[hd * dh + i] * kc[li][j * kvd + kvh * dh + i];
                            }
                            sc[j] = d * scale;
                        }
                        let m = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum = 0.0;
                        for e in sc.iter_mut() {
                            *e = (*e - m).exp();
                            sum += *e;
                        }
                        for i in 0..dh {
                            let mut acc = 0.0f32;
                            for j in 0..kv_len {
                                acc += sc[j] * vc[li][j * kvd + kvh * dh + i];
                            }
                            attn[hd * dh + i] = acc / sum;
                        }
                    }
                    // sigmoid gate, before o_proj
                    for i in 0..attn.len() {
                        attn[i] *= 1.0 / (1.0 + (-gate[i]).exp());
                    }
                    mv(&ow, &attn, &mut buf);
                    buf.clone()
                };
                for i in 0..cfg.d_model {
                    x[i] += mixed[i];
                }

                let fnorm = vecf(g, &tn("post_attention_norm.weight"), 0.0)?;
                let xb = rms(&x, &fnorm, cfg.eps);
                let gw = t(g, &tn("ffn_gate.weight"))?;
                let uw = t(g, &tn("ffn_up.weight"))?;
                let dw = t(g, &tn("ffn_down.weight"))?;
                let mut gate = Vec::new();
                mv(&gw, &xb, &mut gate);
                let mut up = Vec::new();
                mv(&uw, &xb, &mut up);
                let hh: Vec<f32> = (0..cfg.d_ff)
                    .map(|i| (gate[i] / (1.0 + (-gate[i]).exp())) * up[i])
                    .collect();
                mv(&dw, &hh, &mut buf);
                for i in 0..cfg.d_model {
                    x[i] += buf[i];
                }
                if std::env::var("TUILI_TRACE").is_ok() {
                    let r = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
                    eprintln!("  host layer {li:2} rms {r:11.4}");
                }
            }

            let xb = rms(&x, &onorm, cfg.eps);
            mv(&head, &xb, &mut buf);
            logits = buf.clone();
        }
        Ok(logits)
    }

    trait Zeros {
        fn get_or_insert_zeros(&mut self, n: usize) -> &mut Vec<f32>;
    }
    impl Zeros for Vec<f32> {
        fn get_or_insert_zeros(&mut self, n: usize) -> &mut Vec<f32> {
            if self.len() != n {
                *self = vec![0.0; n];
            }
            self
        }
    }
}
