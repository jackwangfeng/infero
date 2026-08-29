//! The Qwen3.5/3.8 kernels against a capture of the reference implementation
//! running on the real 27B checkpoint.
//!
//! Every number here came from outside this repository. That is the point:
//! local self-consistency is what the bf16-as-f16 embedding bug satisfied for a
//! night, across nine component A/Bs, while the model produced nonsense. A test
//! that recomputes a stage the same way the implementation does proves the
//! arithmetic runs, not that it is the right arithmetic.
//!
//! The capture is weight-derived so it is not in git by default. Point the
//! tests at one with `INFERO_QWEN35_CAPTURE=<dir>`, or drop it in
//! `fixtures/qwen35_capture`. Without it these report as skipped rather than
//! passing, because a silent skip is how a suite comes to be green without
//! checking anything.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use infero_metal::{Device, LaunchConfig};

const COMMON: &str = include_str!("../../kernels/src/msl/common.metal");
const OPS: &str = include_str!("../../kernels/src/msl/ops.metal");
const GDN: &str = include_str!("../../kernels/src/msl/gdn.metal");

fn ops() -> String {
    format!("{COMMON}\n{OPS}")
}
fn gdn() -> String {
    format!("{COMMON}\n{GDN}")
}

struct Capture {
    cfg: HashMap<String, f64>,
    arrays: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Capture {
    fn open() -> Option<Self> {
        let dir = std::env::var("INFERO_QWEN35_CAPTURE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/qwen35_capture")
            });
        let manifest = std::fs::read_to_string(dir.join("manifest.json")).ok()?;
        let m: serde_json::Value = serde_json::from_str(&manifest).unwrap();
        assert_eq!(
            m["verified_against_transformers"], true,
            "this capture was written without the cross-check against \
             transformers, so it is one reading of the reference rather than \
             the reference"
        );
        let cfg = m["config"]
            .as_object()
            .unwrap()
            .iter()
            .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
            .collect();
        let mut arrays = HashMap::new();
        for (name, dims) in m["arrays"].as_object().unwrap() {
            let shape: Vec<usize> = dims
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap() as usize)
                .collect();
            let raw = std::fs::read(dir.join(format!("{name}.f32"))).unwrap();
            let v: Vec<f32> = raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            assert_eq!(v.len(), shape.iter().product::<usize>(), "{name}");
            arrays.insert(name.clone(), (shape, v));
        }
        Some(Self { cfg, arrays })
    }

    fn a(&self, k: &str) -> &(Vec<usize>, Vec<f32>) {
        self.arrays.get(k).unwrap_or_else(|| panic!("no array {k}"))
    }
    fn v(&self, k: &str) -> &[f32] {
        &self.a(k).1
    }
    fn c(&self, k: &str) -> usize {
        self.cfg[k] as usize
    }
}

macro_rules! capture {
    () => {
        match Capture::open() {
            Some(c) => c,
            None => {
                eprintln!("skipping: no capture (set INFERO_QWEN35_CAPTURE)");
                return Ok(());
            }
        }
    };
}

/// Relative error against the reference, reported with where it happened.
fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for i in 0..got.len() {
        let d = (got[i] - want[i]).abs() / want[i].abs().max(1.0);
        if d > worst {
            worst = d;
            at = i;
        }
    }
    eprintln!("  {what:34} worst relative error {worst:.3e}");
    assert!(
        worst <= tol,
        "{what}: {worst:.3e} at {at} (got {}, want {})",
        got[at],
        want[at]
    );
}

#[test]
fn the_conv_matches_the_reference() -> Result<()> {
    let c = capture!();
    let dev = Device::new(0)?;
    let s = dev.stream();
    let (ch, k) = (10240usize, c.c("linear_conv_kernel_dim"));
    let nt = c.c("tokens");

    let x = c.v("linear.qkv_pre_conv");
    let w = c.v("linear.w_conv");
    let dx = s.memcpy_stod(x)?;
    let dw = s.memcpy_stod(w)?;
    let mut out = s.alloc_zeros::<f32>(nt * ch)?;
    // Conv state starts empty: the capture is one sequence from position zero.
    let mut state = s.alloc_zeros::<f32>(ch * (k - 1))?;
    let ft = s.memcpy_stod(&[0i32])?;
    let ntb = s.memcpy_stod(&[nt as i32])?;

    let f = dev.kernels().get("gdn", &gdn(), "gdn_conv_f32")?;
    let (ch_i, k_i) = (ch as i32, k as i32);
    let mut b = s.launch_builder(&f);
    b.arg(&out.as_view_mut())
        .arg(&dx.as_view())
        .arg(&state.as_view_mut())
        .arg(&dw.as_view())
        .arg(&ft.as_view())
        .arg(&ntb.as_view())
        .arg(&ch_i)
        .arg(&k_i);
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: ((ch as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })?
    };
    s.synchronize()?;
    close(&out.to_vec(), c.v("linear.qkv_post_conv"), 2e-5, "gdn_conv_f32");
    Ok(())
}

#[test]
fn the_gate_and_decay_match_the_reference() -> Result<()> {
    let c = capture!();
    let dev = Device::new(0)?;
    let s = dev.stream();
    let heads = c.c("linear_num_value_heads");
    let nt = c.c("tokens");

    let da = s.memcpy_stod(c.v("linear.a"))?;
    let db = s.memcpy_stod(c.v("linear.b"))?;
    let dal = s.memcpy_stod(c.v("linear.A_log"))?;
    let ddt = s.memcpy_stod(c.v("linear.dt_bias"))?;
    let mut beta = s.alloc_zeros::<f32>(nt * heads)?;
    let mut g = s.alloc_zeros::<f32>(nt * heads)?;

    let f = dev.kernels().get("gdn", &gdn(), "gdn_gate_decay_f32")?;
    let (nti, hi, st) = (nt as i32, heads as i32, heads as i32);
    let mut b = s.launch_builder(&f);
    b.arg(&beta.as_view_mut())
        .arg(&g.as_view_mut())
        .arg(&da.as_view())
        .arg(&db.as_view())
        .arg(&dal.as_view())
        .arg(&ddt.as_view())
        .arg(&nti)
        .arg(&hi)
        .arg(&st);
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (((nt * heads) as u32).div_ceil(256).max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })?
    };
    s.synchronize()?;
    close(&beta.to_vec(), c.v("linear.beta"), 1e-5, "beta = sigmoid(b)");
    close(&g.to_vec(), c.v("linear.g"), 1e-5, "g = -exp(A_log)*softplus");
    Ok(())
}

#[test]
fn the_delta_rule_matches_the_reference() -> Result<()> {
    let c = capture!();
    let dev = Device::new(0)?;
    let s = dev.stream();
    let heads = c.c("linear_num_value_heads");
    let kh = c.c("linear_num_key_heads");
    let dk = c.c("linear_key_head_dim");
    let dv = c.c("linear_value_head_dim");
    let nt = c.c("tokens");
    let stride = 2 * kh * dk + heads * dv;

    // The l2norm runs in place on the post-conv rows, then the recurrence.
    let mut qkv = s.memcpy_stod(c.v("linear.qkv_post_conv"))?;
    let dg = s.memcpy_stod(c.v("linear.g"))?;
    let dbeta = s.memcpy_stod(c.v("linear.beta"))?;
    let ft = s.memcpy_stod(&[0i32])?;
    let ntb = s.memcpy_stod(&[nt as i32])?;
    let mut out = s.alloc_zeros::<f32>(nt * heads * dv)?;
    let mut state = s.alloc_zeros::<f32>(heads * dk * dv)?;

    {
        let f = dev.kernels().get("gdn", &gdn(), "gdn_qk_l2norm_f32")?;
        let (khi, dki, sti) = (kh as i32, dk as i32, stride as i32);
        let (qo, ko) = (0i32, (kh * dk) as i32);
        let (eps, qs) = (1e-6f32, 1.0f32 / (dk as f32).sqrt());
        let mut b = s.launch_builder(&f);
        b.arg(&qkv.as_view_mut())
            .arg(&khi)
            .arg(&dki)
            .arg(&sti)
            .arg(&qo)
            .arg(&ko)
            .arg(&eps)
            .arg(&qs);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: ((nt * kh) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?
        };
    }
    {
        let f = dev.kernels().get("gdn", &gdn(), "gdn_delta_rule_f32")?;
        let (hi, khi, dki, dvi) = (heads as i32, kh as i32, dk as i32, dv as i32);
        let sti = stride as i32;
        let kk = (kh * dk) as i32;
        let (qo, ko, vo) = (0i32, kk, 2 * kk);
        let mut b = s.launch_builder(&f);
        b.arg(&out.as_view_mut())
            .arg(&state.as_view_mut())
            .arg(&qkv.as_view())
            .arg(&dg.as_view())
            .arg(&dbeta.as_view())
            .arg(&ft.as_view())
            .arg(&ntb.as_view())
            .arg(&hi)
            .arg(&khi)
            .arg(&dki)
            .arg(&dvi)
            .arg(&sti)
            .arg(&qo)
            .arg(&ko)
            .arg(&vo)
            // The capture comes from a Hugging Face checkpoint, whose V heads
            // are grouped by key head rather than tiled.
            .arg(&0i32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (heads as u32, 1, 1),
                block_dim: (dv as u32, 1, 1),
                shared_mem_bytes: (2 * dk * 4) as u32,
            })?
        };
    }
    s.synchronize()?;

    // The output is the interesting one; the final state is the stricter check,
    // because an error that cancels in one token's readout still accumulates
    // there.
    close(&out.to_vec(), c.v("linear.core_attn_out"), 2e-3, "core_attn_out");
    close(&state.to_vec(), c.v("linear.final_state"), 2e-3, "final_state");
    Ok(())
}

#[test]
fn the_gated_norm_matches_the_reference() -> Result<()> {
    let c = capture!();
    let dev = Device::new(0)?;
    let s = dev.stream();
    let dv = c.c("linear_value_head_dim");
    let rows = c.a("linear.after_gated_norm").0[0];

    let dx = s.memcpy_stod(c.v("linear.core_attn_out"))?;
    let dz = s.memcpy_stod(c.v("linear.z"))?;
    let dw = s.memcpy_stod(c.v("linear.norm_w"))?;
    let mut out = s.alloc_zeros::<f32>(rows * dv)?;

    let f = dev.kernels().get("gdn", &gdn(), "gdn_gated_rmsnorm_f32")?;
    let (dvi, eps) = (dv as i32, 1e-6f32);
    let mut b = s.launch_builder(&f);
    b.arg(&out.as_view_mut())
        .arg(&dx.as_view())
        .arg(&dz.as_view())
        .arg(&dw.as_view())
        .arg(&dvi)
        .arg(&eps);
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })?
    };
    s.synchronize()?;
    close(&out.to_vec(), c.v("linear.after_gated_norm"), 1e-4, "after_gated_norm");
    Ok(())
}

#[test]
fn qk_norm_and_partial_rope_match_the_reference() -> Result<()> {
    let c = capture!();
    let dev = Device::new(0)?;
    let s = dev.stream();
    let heads = c.c("num_attention_heads");
    let dh = c.c("head_dim");
    let nt = c.c("tokens");
    let rot = (dh as f64 * c.cfg["partial_rotary_factor"]) as usize;

    // q_norm, then rope, in that order -- the reference normalizes before it
    // rotates and swapping them runs fine.
    let mut q = s.memcpy_stod(c.v("full.q_pre_norm"))?;
    // `q_norm` is a `Qwen3_5RMSNorm`: its weight is a *delta from one*, so the
    // gain is `1 + w`. The engine adds the one at load, which keeps the kernel
    // identical to every other RMSNorm; the capture holds the raw delta, so the
    // test adds it here. Reading it as a gain scales every query by about 0.23
    // instead of 1.23 and inverts the sign wherever the delta is negative --
    // measured at 1.2e0 against the reference, versus 2.1e-7 done right.
    let qw_delta: Vec<f32> = c.v("full.q_norm_w").iter().map(|v| v + 1.0).collect();
    let qw = s.memcpy_stod(&qw_delta)?;
    {
        let f = dev.kernels().get("ops", &ops(), "qk_norm_f32")?;
        let (h, d, st, off, eps) = (heads as i32, dh as i32, (heads * dh) as i32, 0i32, 1e-6f32);
        let mut b = s.launch_builder(&f);
        b.arg(&q.as_view_mut())
            .arg(&qw.as_view())
            .arg(&h)
            .arg(&d)
            .arg(&st)
            .arg(&off)
            .arg(&eps);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: ((nt * heads) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?
        };
    }
    s.synchronize()?;
    close(&q.to_vec(), c.v("full.q_post_norm"), 1e-5, "q_post_norm");

    {
        let cos = s.memcpy_stod(c.v("full.rope_cos"))?;
        let sin = s.memcpy_stod(c.v("full.rope_sin"))?;
        let f = dev.kernels().get("ops", &ops(), "rope_partial_f32")?;
        let (h, d, r) = (heads as i32, dh as i32, rot as i32);
        let mut b = s.launch_builder(&f);
        b.arg(&q.as_view_mut())
            .arg(&cos.as_view())
            .arg(&sin.as_view())
            .arg(&h)
            .arg(&d)
            .arg(&r);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (1, heads as u32, nt as u32),
                block_dim: ((rot / 2) as u32, 1, 1),
                shared_mem_bytes: 0,
            })?
        };
    }
    s.synchronize()?;
    close(&q.to_vec(), c.v("full.q_post_rope"), 1e-5, "q_post_rope");
    Ok(())
}

#[test]
fn the_output_gate_matches_the_reference() -> Result<()> {
    let c = capture!();
    let dev = Device::new(0)?;
    let s = dev.stream();
    let n = c.v("full.attn_out_pre_gate").len();

    let mut x = s.memcpy_stod(c.v("full.attn_out_pre_gate"))?;
    let gate = s.memcpy_stod(c.v("full.gate"))?;
    let f = dev.kernels().get("gdn", &gdn(), "sigmoid_gate_f32")?;
    let ni = n as i64;
    let mut b = s.launch_builder(&f);
    b.arg(&x.as_view_mut()).arg(&gate.as_view()).arg(&ni);
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: ((n as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })?
    };
    s.synchronize()?;
    // sigmoid, not SiLU: `output_gate_type = "swish"` in the config is not read
    // by the reference implementation, which uses sigmoid.
    close(&x.to_vec(), c.v("full.attn_out_post_gate"), 1e-5, "attn_out_post_gate");
    Ok(())
}
