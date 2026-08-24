//! Qwen2 F16, forward pass, on an Apple GPU.
//!
//! A vertical slice: enough of the engine to prove the Metal device layer and
//! the transliterated kernels produce the *right numbers*, checked against the
//! same `transformers` f32 logits that `crates/model/tests/forward.rs` checks
//! the CUDA path against. It is not the engine -- there is no scheduler, no
//! paged KV pool, no batching, and prefill walks the prompt one token at a time
//! through the decode path so that no GEMM is needed at all. Folding this into
//! `tuili-model` is the genericization step, and it is separate work.
//!
//! ```text
//! cargo run --release -p tuili-metal --example qwen2_f16 -- \
//!     models/qwen2.5-0.5b-instruct-fp16.gguf
//! ```
//!
//! With no arguments beyond the model it runs the four fixture prompts and
//! reports argmax against the reference. Given `--prompt`, it generates.

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use half::f16;
use tuili_gguf::{Gguf, GgmlType};
use tuili_tokenizer::Tokenizer;
use tuili_metal::{Buf, Device, LaunchConfig, View, ViewMut};

const COMMON_METAL: &str = include_str!("../../kernels/src/msl/common.metal");
const OPS_METAL: &str = include_str!("../../kernels/src/msl/ops.metal");
const QUANT_METAL: &str = include_str!("../../kernels/src/msl/quant.metal");

const BLOCK: u32 = 256;
/// Threads for the reduction-shaped kernels. Fixed rather than occupancy-
/// derived: the Metal backend has no equivalent of the CUDA side's register
/// probe yet, and a constant that is right for every shape in this model beats
/// a number picked from an API that answers a different question.
const REDUCE_BLOCK: u32 = 256;

fn ops_src() -> String {
    format!("{COMMON_METAL}\n{OPS_METAL}")
}

fn quant_src() -> String {
    format!("{COMMON_METAL}\n{QUANT_METAL}")
}

// ---- config --------------------------------------------------------------

struct Cfg {
    n_layers: usize,
    d_model: usize,
    n_heads: usize,
    n_kv: usize,
    d_head: usize,
    d_ff: usize,
    vocab: usize,
    eps: f32,
    theta: f32,
}

impl Cfg {
    fn read(g: &Gguf) -> Result<Self> {
        let arch = g.arch()?.to_string();
        let key = |s: &str| format!("{arch}.{s}");
        let n_heads = g.usize(&key("attention.head_count"))?;
        let d_model = g.usize(&key("embedding_length"))?;
        let vocab = g
            .tensor("token_embd.weight")
            .map(|t| t.dims[1] as usize)
            .context("token_embd.weight")?;
        Ok(Self {
            n_layers: g.usize(&key("block_count"))?,
            d_model,
            n_heads,
            n_kv: g.usize(&key("attention.head_count_kv")).unwrap_or(n_heads),
            d_head: d_model / n_heads,
            d_ff: g.usize(&key("feed_forward_length"))?,
            vocab,
            eps: g
                .f32(&key("attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-6),
            theta: g.f32(&key("rope.freq_base")).unwrap_or(10_000.0),
        })
    }
}

// ---- weights -------------------------------------------------------------

/// Read a tensor as f32 regardless of how it is stored. Norms are F32 in this
/// checkpoint and matrices are F16, but nothing guarantees that per tensor, so
/// both loaders accept both types rather than trusting a convention.
fn to_f32(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = g.tensor(name)?;
    let raw = g.data(t);
    Ok(match t.ty {
        GgmlType::F32 => raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        GgmlType::F16 => raw
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        other => return Err(anyhow!("{name}: cannot read {other:?} as f32")),
    })
}

fn to_f16(g: &Gguf, name: &str) -> Result<Vec<f16>> {
    let t = g.tensor(name)?;
    let raw = g.data(t);
    Ok(match t.ty {
        GgmlType::F16 => raw
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]]))
            .collect(),
        GgmlType::F32 => raw
            .chunks_exact(4)
            .map(|b| f16::from_f32(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            .collect(),
        other => return Err(anyhow!("{name}: cannot read {other:?} as f16")),
    })
}

/// A linear weight. ggml stores `dims[0]` as the fastest axis, so a torch
/// `[out, in]` matrix is `[in, out]` here -- which lands as exactly the
/// `out`-major, `in`-contiguous layout `gemv_f16` walks.
struct Mat {
    w: Buf<f16>,
    k: usize,
    n: usize,
}

fn mat(dev: &Device, g: &Gguf, name: &str) -> Result<Mat> {
    let t = g.tensor(name)?;
    let (k, n) = (t.dims[0] as usize, t.dims[1] as usize);
    let host = to_f16(g, name)?;
    anyhow::ensure!(host.len() == k * n, "{name}: {} != {k}x{n}", host.len());
    Ok(Mat {
        w: dev.stream().memcpy_stod(&host)?,
        k,
        n,
    })
}

fn vec_f32(dev: &Device, g: &Gguf, name: &str) -> Result<Buf<f32>> {
    let host = to_f32(g, name)?;
    dev.stream().memcpy_stod(&host)
}

fn opt_vec_f32(dev: &Device, g: &Gguf, name: &str) -> Result<Option<Buf<f32>>> {
    if g.get_tensor(name).is_none() {
        return Ok(None);
    }
    Ok(Some(vec_f32(dev, g, name)?))
}

struct Layer {
    attn_norm: Buf<f32>,
    wq: Mat,
    wk: Mat,
    wv: Mat,
    wo: Mat,
    bq: Option<Buf<f32>>,
    bk: Option<Buf<f32>>,
    bv: Option<Buf<f32>>,
    ffn_norm: Buf<f32>,
    w_gate: Mat,
    w_up: Mat,
    w_down: Mat,
}

struct Weights {
    embd: Mat,
    out_norm: Buf<f32>,
    /// Absent when the checkpoint ties the head to the embedding table, which
    /// Qwen2.5-0.5B does.
    head: Option<Mat>,
    layers: Vec<Layer>,
}

impl Weights {
    fn load(dev: &Device, g: &Gguf, cfg: &Cfg) -> Result<Self> {
        let started = std::time::Instant::now();
        let embd = mat(dev, g, "token_embd.weight")?;
        let out_norm = vec_f32(dev, g, "output_norm.weight")?;
        let head = if g.get_tensor("output.weight").is_some() {
            Some(mat(dev, g, "output.weight")?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let t = |s: &str| format!("blk.{i}.{s}");
            layers.push(Layer {
                attn_norm: vec_f32(dev, g, &t("attn_norm.weight"))?,
                wq: mat(dev, g, &t("attn_q.weight"))?,
                wk: mat(dev, g, &t("attn_k.weight"))?,
                wv: mat(dev, g, &t("attn_v.weight"))?,
                wo: mat(dev, g, &t("attn_output.weight"))?,
                bq: opt_vec_f32(dev, g, &t("attn_q.bias"))?,
                bk: opt_vec_f32(dev, g, &t("attn_k.bias"))?,
                bv: opt_vec_f32(dev, g, &t("attn_v.bias"))?,
                ffn_norm: vec_f32(dev, g, &t("ffn_norm.weight"))?,
                w_gate: mat(dev, g, &t("ffn_gate.weight"))?,
                w_up: mat(dev, g, &t("ffn_up.weight"))?,
                w_down: mat(dev, g, &t("ffn_down.weight"))?,
            });
        }
        eprintln!(
            "weights uploaded in {} ms",
            started.elapsed().as_millis()
        );
        Ok(Self {
            embd,
            out_norm,
            head,
            layers,
        })
    }
}

// ---- the forward pass ----------------------------------------------------

struct Session {
    x: Buf<f32>,
    xb: Buf<f32>,
    q: Buf<f32>,
    k: Buf<f32>,
    v: Buf<f32>,
    attn: Buf<f32>,
    proj: Buf<f32>,
    /// `[gate | up]` back to back, which is the layout `silu_mul_split_f32`
    /// reads -- the same reason the CUDA path fuses them.
    ff: Buf<f32>,
    ff_out: Buf<f32>,
    logits: Buf<f32>,
    /// One cache per layer, laid out `[layer][position][kv_head][d_head]`.
    ///
    /// It was briefly a single `[position][...]` plane shared by every layer,
    /// which is wrong in a way that keeps the logits' magnitude plausible: each
    /// layer overwrote the previous one's K and V at the same position, so
    /// layer `n` attended to layer 23's history. std stayed within 0.07 of the
    /// reference and every token was different. The per-kernel tests all passed
    /// -- it was the composition, and the host arbiter is what found it.
    kcache: Buf<f16>,
    vcache: Buf<f16>,
    /// Positions a layer's plane holds, so the layer offset can be computed.
    max_pos: usize,
    kv_stride: usize,
    ids: Buf<i32>,
    pos: Buf<i32>,
    freq: Buf<f32>,
}

impl Session {
    fn new(dev: &Device, cfg: &Cfg, max_pos: usize) -> Result<Self> {
        let s = dev.stream();
        let kv = cfg.n_kv * cfg.d_head;
        Ok(Self {
            x: s.alloc_zeros(cfg.d_model)?,
            xb: s.alloc_zeros(cfg.d_model)?,
            q: s.alloc_zeros(cfg.n_heads * cfg.d_head)?,
            k: s.alloc_zeros(kv)?,
            v: s.alloc_zeros(kv)?,
            attn: s.alloc_zeros(cfg.n_heads * cfg.d_head)?,
            proj: s.alloc_zeros(cfg.d_model)?,
            ff: s.alloc_zeros(2 * cfg.d_ff)?,
            ff_out: s.alloc_zeros(cfg.d_ff)?,
            logits: s.alloc_zeros(cfg.vocab)?,
            kcache: s.alloc_zeros(cfg.n_layers * max_pos * kv)?,
            vcache: s.alloc_zeros(cfg.n_layers * max_pos * kv)?,
            max_pos,
            kv_stride: kv,
            ids: s.alloc_zeros(1)?,
            pos: s.alloc_zeros(1)?,
            // No long-context scaling in this model, so the divisor is one.
            freq: s.memcpy_stod(&vec![1.0f32; cfg.d_head / 2])?,
        })
    }
}

struct Engine {
    dev: Device,
    cfg: Cfg,
    w: Weights,
}

fn grid1(n: u32, block: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(block).max(1), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl Engine {
    fn f(&self, name: &str) -> Result<tuili_metal::Function> {
        self.dev.kernels().get("tuili_ops", &ops_src(), name)
    }

    fn gemv(&self, out: &mut ViewMut<'_, f32>, m: &Mat, x: &View<'_, f32>) -> Result<()> {
        let f = self.dev.kernels().get("tuili_quant", &quant_src(), "gemv_f16")?;
        // The quant module's mat-vecs take `n_tokens`; this slice is batch one.
        let (k, n, t) = (m.k as i32, m.n as i32, 1i32);
        let s = self.dev.stream();
        let mut b = s.launch_builder(&f);
        b.arg(out)
            .arg(&m.w.as_view())
            .arg(x)
            .arg(&k)
            .arg(&n)
            .arg(&t);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (m.n as u32, 1, 1),
                block_dim: (REDUCE_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
    }

    fn rms_norm(
        &self,
        out: &mut ViewMut<'_, f32>,
        x: &View<'_, f32>,
        w: &View<'_, f32>,
    ) -> Result<()> {
        let f = self.f("rms_norm_f32")?;
        let (d, eps) = (self.cfg.d_model as i32, self.cfg.eps);
        let s = self.dev.stream();
        let mut b = s.launch_builder(&f);
        b.arg(out).arg(x).arg(w).arg(&d).arg(&eps);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (REDUCE_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
    }

    fn add_bias(&self, out: &mut ViewMut<'_, f32>, bias: &View<'_, f32>, n: usize) -> Result<()> {
        let f = self.f("add_bias_f32")?;
        let (cols, rows) = (n as i32, 1i32);
        let s = self.dev.stream();
        let mut b = s.launch_builder(&f);
        b.arg(out).arg(bias).arg(&cols).arg(&rows);
        unsafe { b.launch(grid1(n as u32, BLOCK)) }
    }

    fn add_assign(&self, out: &mut ViewMut<'_, f32>, b_in: &View<'_, f32>, n: usize) -> Result<()> {
        let f = self.f("add_assign_f32")?;
        let n_i = n as i32;
        let s = self.dev.stream();
        let mut b = s.launch_builder(&f);
        b.arg(out).arg(b_in).arg(&n_i);
        unsafe { b.launch(grid1(n as u32, BLOCK)) }
    }

    fn rope(&self, x: &mut ViewMut<'_, f32>, pos: &View<'_, i32>, freq: &View<'_, f32>, heads: usize) -> Result<()> {
        let f = self.f("rope_neox_f32")?;
        let (nh, dh) = (heads as i32, self.cfg.d_head as i32);
        let (theta, scale) = (self.cfg.theta, 1.0f32);
        let s = self.dev.stream();
        let mut b = s.launch_builder(&f);
        b.arg(x)
            .arg(pos)
            .arg(freq)
            .arg(&nh)
            .arg(&dh)
            .arg(&theta)
            .arg(&scale);
        let half = (self.cfg.d_head / 2) as u32;
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (half.div_ceil(BLOCK).max(1), heads as u32, 1),
                block_dim: (BLOCK.min(half.next_power_of_two()), 1, 1),
                shared_mem_bytes: 0,
            })
        }
    }

    /// One token through every layer, leaving the logits in `sess.logits`.
    fn step(&self, sess: &mut Session, token: u32, pos: usize) -> Result<()> {
        let s = self.dev.stream();
        let cfg = &self.cfg;
        s.copy_into(&mut sess.ids.as_view_mut(), &[token as i32])?;
        s.copy_into(&mut sess.pos.as_view_mut(), &[pos as i32])?;

        // Embedding.
        {
            let f = self.f("embed_f16")?;
            let d = cfg.d_model as i32;
            let mut b = s.launch_builder(&f);
            b.arg(&sess.x.as_view_mut())
                .arg(&self.w.embd.w.as_view())
                .arg(&sess.ids.as_view())
                .arg(&d);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: ((cfg.d_model as u32).div_ceil(BLOCK).max(1), 1, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
            }?;
        }

        let kv = cfg.n_kv * cfg.d_head;
        let plane = sess.max_pos * sess.kv_stride;
        for (li, l) in self.w.layers.iter().enumerate() {
            let (lo, hi) = (li * plane, (li + 1) * plane);
            // --- attention ---
            self.rms_norm(
                &mut sess.xb.as_view_mut(),
                &sess.x.as_view(),
                &l.attn_norm.as_view(),
            )?;
            self.gemv(&mut sess.q.as_view_mut(), &l.wq, &sess.xb.as_view())?;
            self.gemv(&mut sess.k.as_view_mut(), &l.wk, &sess.xb.as_view())?;
            self.gemv(&mut sess.v.as_view_mut(), &l.wv, &sess.xb.as_view())?;
            if let Some(b) = &l.bq {
                self.add_bias(&mut sess.q.as_view_mut(), &b.as_view(), l.wq.n)?;
            }
            if let Some(b) = &l.bk {
                self.add_bias(&mut sess.k.as_view_mut(), &b.as_view(), l.wk.n)?;
            }
            if let Some(b) = &l.bv {
                self.add_bias(&mut sess.v.as_view_mut(), &b.as_view(), l.wv.n)?;
            }

            let (pos_v, freq_v) = (sess.pos.as_view(), sess.freq.as_view());
            self.rope(&mut sess.q.as_view_mut(), &pos_v, &freq_v, cfg.n_heads)?;
            self.rope(&mut sess.k.as_view_mut(), &pos_v, &freq_v, cfg.n_kv)?;

            {
                let f = self.f("store_kv_contig_f16")?;
                let (nkv, dh, p) = (cfg.n_kv as i32, cfg.d_head as i32, pos as i32);
                let mut b = s.launch_builder(&f);
                b.arg(&sess.kcache.slice_mut(lo..hi))
                    .arg(&sess.vcache.slice_mut(lo..hi))
                    .arg(&sess.k.as_view())
                    .arg(&sess.v.as_view())
                    .arg(&nkv)
                    .arg(&dh)
                    .arg(&p);
                unsafe { b.launch(grid1(kv as u32, BLOCK)) }?;
            }

            {
                let f = self.f("attn_decode_f32")?;
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
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: (cfg.n_heads as u32, 1, 1),
                        block_dim: (REDUCE_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }?;
            }

            self.gemv(&mut sess.proj.as_view_mut(), &l.wo, &sess.attn.as_view())?;
            self.add_assign(
                &mut sess.x.as_view_mut(),
                &sess.proj.as_view(),
                cfg.d_model,
            )?;

            // --- feed forward ---
            self.rms_norm(
                &mut sess.xb.as_view_mut(),
                &sess.x.as_view(),
                &l.ffn_norm.as_view(),
            )?;
            self.gemv(&mut sess.ff.slice_mut(..cfg.d_ff), &l.w_gate, &sess.xb.as_view())?;
            self.gemv(
                &mut sess.ff.slice_mut(cfg.d_ff..2 * cfg.d_ff),
                &l.w_up,
                &sess.xb.as_view(),
            )?;
            {
                let f = self.f("silu_mul_split_f32")?;
                let (dff, total) = (cfg.d_ff as i32, cfg.d_ff as i32);
                let mut b = s.launch_builder(&f);
                b.arg(&sess.ff_out.as_view_mut())
                    .arg(&sess.ff.as_view())
                    .arg(&dff)
                    .arg(&total);
                unsafe { b.launch(grid1(cfg.d_ff as u32, BLOCK)) }?;
            }
            self.gemv(&mut sess.proj.as_view_mut(), &l.w_down, &sess.ff_out.as_view())?;
            self.add_assign(
                &mut sess.x.as_view_mut(),
                &sess.proj.as_view(),
                cfg.d_model,
            )?;
        }

        self.rms_norm(
            &mut sess.xb.as_view_mut(),
            &sess.x.as_view(),
            &self.w.out_norm.as_view(),
        )?;
        let head = self.w.head.as_ref().unwrap_or(&self.w.embd);
        self.gemv(&mut sess.logits.as_view_mut(), head, &sess.xb.as_view())?;
        s.synchronize()?;
        Ok(())
    }

    fn forward(&self, sess: &mut Session, ids: &[u32]) -> Result<Vec<f32>> {
        for (p, &t) in ids.iter().enumerate() {
            self.step(sess, t, p)?;
        }
        Ok(sess.logits.to_vec())
    }
}

// ---- the fixture check ---------------------------------------------------

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as u32
}

fn top_k(v: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..v.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| v[b as usize].total_cmp(&v[a as usize]));
    idx.truncate(k);
    idx
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("usage: qwen2_f16 <model.gguf> [--prompt <text>]"))?;

    let g = Gguf::open(&model).with_context(|| format!("opening {}", model.display()))?;
    let cfg = Cfg::read(&g)?;
    let dev = Device::new(0)?;
    eprintln!(
        "{} | {} layers, d_model {}, {} heads / {} kv, d_head {}, ffn {}, vocab {}, theta {}",
        dev.name(),
        cfg.n_layers,
        cfg.d_model,
        cfg.n_heads,
        cfg.n_kv,
        cfg.d_head,
        cfg.d_ff,
        cfg.vocab,
        cfg.theta
    );

    // `--cpu` runs the host arbiter on the first fixture case instead, which is
    // how a GPU/reference disagreement gets attributed.
    if std::env::args().any(|a| a == "--cpu") {
        let fx: serde_json::Value = serde_json::from_str(include_str!(
            "../../model/tests/fixtures/qwen2.5-0.5b-instruct-logits.json"
        ))?;
        let case = &fx["cases"][0];
        let ids: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        eprintln!("host reference over {} ids ...", ids.len());
        let logits = host::forward(&g, &cfg, &ids)?;
        let mean = logits.iter().sum::<f32>() / logits.len() as f32;
        let var = logits.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
            / (logits.len() - 1) as f32;
        println!(
            "  host argmax {} (want {})  std {:.4} (want {:.4})",
            argmax(&logits),
            case["argmax"].as_u64().unwrap(),
            var.sqrt(),
            case["std"].as_f64().unwrap()
        );
        return Ok(());
    }

    let want_prompt = {
        let mut a = std::env::args().skip_while(|a| a != "--prompt");
        a.next();
        a.next()
    };

    let w = Weights::load(&dev, &g, &cfg)?;
    let eng = Engine { dev, cfg, w };

    if let Some(prompt) = want_prompt {
        let tok = Tokenizer::from_gguf(&g)?;
        let ids = tok.encode(&prompt, Some(false), true);
        let n_new = 96usize;
        let mut sess = Session::new(&eng.dev, &eng.cfg, ids.len() + n_new + 1)?;

        print!("{prompt}");
        use std::io::Write;
        std::io::stdout().flush().ok();

        let started = std::time::Instant::now();
        let mut logits = eng.forward(&mut sess, &ids)?;
        let prefill = started.elapsed();

        let mut det = tok.detokenizer();
        let decode_start = std::time::Instant::now();
        let mut produced = 0usize;
        for p in ids.len()..ids.len() + n_new {
            let next = argmax(&logits);
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

        let d = decode_start.elapsed();
        eprintln!(
            "\nprefill {:>7.1} ms ({:.1} tok/s over {} tokens)\ndecode  {:>7.1} ms ({:.1} tok/s over {produced} tokens)",
            prefill.as_secs_f64() * 1e3,
            ids.len() as f64 / prefill.as_secs_f64(),
            ids.len(),
            d.as_secs_f64() * 1e3,
            produced as f64 / d.as_secs_f64(),
        );
        return Ok(());
    }

    // The four cases from `crates/model/tests/fixtures/`, ids and all, so that
    // this reads the same oracle the CUDA tests read.
    let fx: serde_json::Value = serde_json::from_str(include_str!(
        "../../model/tests/fixtures/qwen2.5-0.5b-instruct-logits.json"
    ))?;
    let cases = fx["cases"].as_array().unwrap();

    let mut failures = 0;
    for case in cases {
        let ids: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let want_argmax = case["argmax"].as_u64().unwrap() as u32;
        let want_std = case["std"].as_f64().unwrap() as f32;
        let theirs: std::collections::HashSet<u32> = case["top_ids"]
            .as_array()
            .unwrap()
            .iter()
            .take(10)
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();

        let mut sess = Session::new(&eng.dev, &eng.cfg, ids.len() + 64)?;
        let started = std::time::Instant::now();
        let logits = eng.forward(&mut sess, &ids)?;
        let ms = started.elapsed().as_secs_f64() * 1e3;

        let mean = logits.iter().sum::<f32>() / logits.len() as f32;
        let var = logits.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
            / (logits.len() - 1) as f32;
        let std = var.sqrt();
        let got = argmax(&logits);
        let overlap = top_k(&logits, 10)
            .iter()
            .filter(|t| theirs.contains(t))
            .count();

        let prompt = case["prompt"].as_str().unwrap();
        let ok = got == want_argmax && overlap >= 8 && (std - want_std).abs() < 0.35;
        if !ok {
            failures += 1;
        }
        println!(
            "  {} {:<44} argmax {:>6} (want {:>6})  std {:>7.4} (want {:>7.4})  top10 {overlap}/10  {:>7.1} ms",
            if ok { "ok  " } else { "FAIL" },
            format!("{:?}", &prompt[..prompt.len().min(38)]),
            got,
            want_argmax,
            std,
            want_std,
            ms
        );
    }

    if failures > 0 {
        return Err(anyhow!("{failures} of {} cases wrong", cases.len()));
    }
    println!("\nall {} fixture cases match the reference", cases.len());
    Ok(())
}

// ---- the host arbiter ----------------------------------------------------

/// The same forward pass on the CPU, reading the same tensors the same way.
///
/// This exists to answer one question when the GPU path disagrees with the
/// fixture: is the *model* wrong (dims, rope convention, layer order) or is the
/// *wiring* wrong (a buffer bound to the wrong argument, a stale residual)? If
/// this agrees with the fixture and the GPU does not, the kernels compose
/// badly; if neither agrees, the misunderstanding is upstream of both.
mod host {
    use super::*;

    struct HMat {
        w: Vec<f32>,
        k: usize,
        n: usize,
    }

    fn hmat(g: &Gguf, name: &str) -> Result<HMat> {
        let t = g.tensor(name)?;
        let (k, n) = (t.dims[0] as usize, t.dims[1] as usize);
        Ok(HMat {
            w: to_f32(g, name)?,
            k,
            n,
        })
    }

    fn mv(m: &HMat, x: &[f32]) -> Vec<f32> {
        (0..m.n)
            .map(|r| (0..m.k).map(|i| m.w[r * m.k + i] * x[i]).sum())
            .collect()
    }

    fn rms(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let ss: f32 = x.iter().map(|v| v * v).sum();
        let s = 1.0 / (ss / x.len() as f32 + eps).sqrt();
        x.iter().zip(w).map(|(a, b)| a * s * b).collect()
    }

    fn rope(x: &mut [f32], heads: usize, d_head: usize, pos: usize, theta: f32) {
        let half = d_head / 2;
        for h in 0..heads {
            for i in 0..half {
                let inv = theta.powf(-2.0 * i as f32 / d_head as f32);
                let ang = pos as f32 * inv;
                let (sa, ca) = (ang.sin(), ang.cos());
                let b = h * d_head;
                let (a, c) = (x[b + i], x[b + i + half]);
                x[b + i] = a * ca - c * sa;
                x[b + i + half] = a * sa + c * ca;
            }
        }
    }

    pub fn forward(g: &Gguf, cfg: &Cfg, ids: &[u32]) -> Result<Vec<f32>> {
        let embd = hmat(g, "token_embd.weight")?;
        let out_norm = to_f32(g, "output_norm.weight")?;
        let head = if g.get_tensor("output.weight").is_some() {
            hmat(g, "output.weight")?
        } else {
            hmat(g, "token_embd.weight")?
        };

        struct L {
            an: Vec<f32>,
            wq: HMat,
            wk: HMat,
            wv: HMat,
            wo: HMat,
            bq: Vec<f32>,
            bk: Vec<f32>,
            bv: Vec<f32>,
            fn_: Vec<f32>,
            wg: HMat,
            wu: HMat,
            wd: HMat,
        }
        let mut ls = Vec::new();
        for i in 0..cfg.n_layers {
            let t = |s: &str| format!("blk.{i}.{s}");
            ls.push(L {
                an: to_f32(g, &t("attn_norm.weight"))?,
                wq: hmat(g, &t("attn_q.weight"))?,
                wk: hmat(g, &t("attn_k.weight"))?,
                wv: hmat(g, &t("attn_v.weight"))?,
                wo: hmat(g, &t("attn_output.weight"))?,
                bq: to_f32(g, &t("attn_q.bias")).unwrap_or_default(),
                bk: to_f32(g, &t("attn_k.bias")).unwrap_or_default(),
                bv: to_f32(g, &t("attn_v.bias")).unwrap_or_default(),
                fn_: to_f32(g, &t("ffn_norm.weight"))?,
                wg: hmat(g, &t("ffn_gate.weight"))?,
                wu: hmat(g, &t("ffn_up.weight"))?,
                wd: hmat(g, &t("ffn_down.weight"))?,
            });
        }

        let kvd = cfg.n_kv * cfg.d_head;
        let mut kc: Vec<Vec<f32>> = vec![Vec::new(); cfg.n_layers];
        let mut vc: Vec<Vec<f32>> = vec![Vec::new(); cfg.n_layers];
        let mut logits = Vec::new();

        for (pos, &tok) in ids.iter().enumerate() {
            let mut x: Vec<f32> = embd.w
                [tok as usize * cfg.d_model..(tok as usize + 1) * cfg.d_model]
                .to_vec();

            for (li, l) in ls.iter().enumerate() {
                let xb = rms(&x, &l.an, cfg.eps);
                let mut q = mv(&l.wq, &xb);
                let mut k = mv(&l.wk, &xb);
                let mut v = mv(&l.wv, &xb);
                for (i, b) in l.bq.iter().enumerate() {
                    q[i] += b;
                }
                for (i, b) in l.bk.iter().enumerate() {
                    k[i] += b;
                }
                for (i, b) in l.bv.iter().enumerate() {
                    v[i] += b;
                }
                rope(&mut q, cfg.n_heads, cfg.d_head, pos, cfg.theta);
                rope(&mut k, cfg.n_kv, cfg.d_head, pos, cfg.theta);
                kc[li].extend_from_slice(&k);
                vc[li].extend_from_slice(&v);

                let kv_len = pos + 1;
                let group = cfg.n_heads / cfg.n_kv;
                let mut attn = vec![0.0f32; cfg.n_heads * cfg.d_head];
                let scale = 1.0 / (cfg.d_head as f32).sqrt();
                for h in 0..cfg.n_heads {
                    let kvh = h / group;
                    let mut sc = vec![0.0f32; kv_len];
                    for j in 0..kv_len {
                        let mut d = 0.0f32;
                        for i in 0..cfg.d_head {
                            d += q[h * cfg.d_head + i]
                                * kc[li][j * kvd + kvh * cfg.d_head + i];
                        }
                        sc[j] = d * scale;
                    }
                    let m = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0;
                    for s in sc.iter_mut() {
                        *s = (*s - m).exp();
                        sum += *s;
                    }
                    for i in 0..cfg.d_head {
                        let mut acc = 0.0f32;
                        for j in 0..kv_len {
                            acc += sc[j] * vc[li][j * kvd + kvh * cfg.d_head + i];
                        }
                        attn[h * cfg.d_head + i] = acc / sum;
                    }
                }

                let proj = mv(&l.wo, &attn);
                for i in 0..cfg.d_model {
                    x[i] += proj[i];
                }

                let xb = rms(&x, &l.fn_, cfg.eps);
                let gate = mv(&l.wg, &xb);
                let up = mv(&l.wu, &xb);
                let h: Vec<f32> = (0..cfg.d_ff)
                    .map(|i| (gate[i] / (1.0 + (-gate[i]).exp())) * up[i])
                    .collect();
                let down = mv(&l.wd, &h);
                for i in 0..cfg.d_model {
                    x[i] += down[i];
                }
            }

            let xb = rms(&x, &out_norm, cfg.eps);
            logits = mv(&head, &xb);
        }
        Ok(logits)
    }
}
