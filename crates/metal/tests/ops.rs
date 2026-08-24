//! Each ported kernel against a host reference.
//!
//! The end-to-end fixture check says whether the whole pass is right; this file
//! says *which kernel* is wrong when it is not. References are written from the
//! CUDA source rather than from memory, so a disagreement here is a porting
//! error and not a difference of opinion about what the operator means.

use anyhow::Result;
use half::f16;
use tuili_metal::{Device, LaunchConfig};

const COMMON: &str = include_str!("../../kernels/src/msl/common.metal");
const OPS: &str = include_str!("../../kernels/src/msl/ops.metal");
const QUANT: &str = include_str!("../../kernels/src/msl/quant.metal");

fn src() -> String {
    format!("{COMMON}\n{OPS}")
}

fn quant_src() -> String {
    format!("{COMMON}\n{QUANT}")
}

const BLOCK: u32 = 256;

fn grid1(n: u32, block: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(block).max(1), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Deterministic pseudo-random floats in [-1, 1); a fixed sequence so a failure
/// reproduces.
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for i in 0..got.len() {
        let d = (got[i] - want[i]).abs();
        let scale = want[i].abs().max(1.0);
        if d / scale > worst {
            worst = d / scale;
            at = i;
        }
    }
    assert!(
        worst <= tol,
        "{what}: relative error {worst:.3e} at {at} (got {}, want {})",
        got[at],
        want[at]
    );
}

#[test]
fn rms_norm_matches_the_host() -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();
    let d = 896usize;
    let eps = 1e-6f32;

    let x = noise(d, 7);
    let w = noise(d, 99);
    let dx = s.memcpy_stod(&x)?;
    let dw = s.memcpy_stod(&w)?;
    let mut out = s.alloc_zeros::<f32>(d)?;

    let f = dev.kernels().get("ops", &src(), "rms_norm_f32")?;
    let (d_i, eps_f) = (d as i32, eps);
    let mut b = s.launch_builder(&f);
    b.arg(&out.as_view_mut())
        .arg(&dx.as_view())
        .arg(&dw.as_view())
        .arg(&d_i)
        .arg(&eps_f);
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })?
    };
    s.synchronize()?;

    let ss: f32 = x.iter().map(|v| v * v).sum();
    let scale = 1.0 / (ss / d as f32 + eps).sqrt();
    let want: Vec<f32> = (0..d).map(|i| x[i] * scale * w[i]).collect();
    close(&out.to_vec(), &want, 1e-5, "rms_norm_f32");
    Ok(())
}

#[test]
fn gemv_f16_matches_the_host() -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();
    // Deliberately not square, and with `k` not a multiple of the block: a
    // transposed weight or a mis-strided row would pass on a square matrix.
    let (k, n) = (896usize, 137usize);

    let wf: Vec<f32> = noise(k * n, 3);
    let w: Vec<f16> = wf.iter().map(|&v| f16::from_f32(v)).collect();
    let x = noise(k, 11);

    let dw = s.memcpy_stod(&w)?;
    let dx = s.memcpy_stod(&x)?;
    let mut out = s.alloc_zeros::<f32>(n)?;

    let f = dev.kernels().get("quant", &quant_src(), "gemv_f16")?;
    // The quant module's mat-vecs take `n_tokens`: one threadgroup decodes a
    // weight once and spends it on every token it holds.
    let (k_i, n_i, t_i) = (k as i32, n as i32, 1i32);
    let mut b = s.launch_builder(&f);
    b.arg(&out.as_view_mut())
        .arg(&dw.as_view())
        .arg(&dx.as_view())
        .arg(&k_i)
        .arg(&n_i)
        .arg(&t_i);
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })?
    };
    s.synchronize()?;

    // `out[r] = dot(w[r, :], x)` -- row-major over `k`, which is ggml's layout.
    let want: Vec<f32> = (0..n)
        .map(|r| {
            (0..k)
                .map(|i| w[r * k + i].to_f32() * x[i])
                .sum::<f32>()
        })
        .collect();
    close(&out.to_vec(), &want, 2e-3, "gemv_f16");
    Ok(())
}

#[test]
fn silu_mul_split_matches_the_host() -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();
    let d_ff = 4864usize;

    let xy = noise(2 * d_ff, 5);
    let dxy = s.memcpy_stod(&xy)?;
    let mut out = s.alloc_zeros::<f32>(d_ff)?;

    let f = dev.kernels().get("ops", &src(), "silu_mul_split_f32")?;
    let (dff_i, total_i) = (d_ff as i32, d_ff as i32);
    let mut b = s.launch_builder(&f);
    b.arg(&out.as_view_mut())
        .arg(&dxy.as_view())
        .arg(&dff_i)
        .arg(&total_i);
    unsafe { b.launch(grid1(d_ff as u32, BLOCK))? };
    s.synchronize()?;

    let want: Vec<f32> = (0..d_ff)
        .map(|i| {
            let g = xy[i];
            (g / (1.0 + (-g).exp())) * xy[d_ff + i]
        })
        .collect();
    close(&out.to_vec(), &want, 1e-5, "silu_mul_split_f32");
    Ok(())
}

#[test]
fn rope_neox_matches_the_host() -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();
    let (heads, d_head) = (14usize, 64usize);
    let theta = 1e6f32;
    let pos = 5i32;

    let x = noise(heads * d_head, 13);
    let mut dx = s.memcpy_stod(&x)?;
    let dpos = s.memcpy_stod(&[pos])?;
    let dfreq = s.memcpy_stod(&vec![1.0f32; d_head / 2])?;

    let f = dev.kernels().get("ops", &src(), "rope_neox_f32")?;
    let (nh, dh) = (heads as i32, d_head as i32);
    let (th, sc) = (theta, 1.0f32);
    let mut b = s.launch_builder(&f);
    b.arg(&dx.as_view_mut())
        .arg(&dpos.as_view())
        .arg(&dfreq.as_view())
        .arg(&nh)
        .arg(&dh)
        .arg(&th)
        .arg(&sc);
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (1, heads as u32, 1),
            block_dim: ((d_head / 2) as u32, 1, 1),
            shared_mem_bytes: 0,
        })?
    };
    s.synchronize()?;

    // Straight from `rope_neox_f32` in ops.cu: the exponent divides by
    // `d_head`, and element `i` pairs with `i + d_head / 2`.
    let half = d_head / 2;
    let mut want = x.clone();
    for h in 0..heads {
        for i in 0..half {
            let inv = theta.powf(-2.0 * i as f32 / d_head as f32);
            let angle = pos as f32 * inv;
            let (sa, ca) = (angle.sin(), angle.cos());
            let base = h * d_head;
            let a = x[base + i];
            let b_ = x[base + i + half];
            want[base + i] = a * ca - b_ * sa;
            want[base + i + half] = a * sa + b_ * ca;
        }
    }
    close(&dx.to_vec(), &want, 1e-5, "rope_neox_f32");
    Ok(())
}

#[test]
fn attention_matches_the_host() -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();
    let (n_heads, n_kv, d_head) = (14usize, 2usize, 64usize);
    let kv_len = 19usize;
    let scale = 1.0f32 / (d_head as f32).sqrt();
    let stride = n_kv * d_head;

    let q = noise(n_heads * d_head, 21);
    let kf = noise(kv_len * stride, 22);
    let vf = noise(kv_len * stride, 23);
    let k: Vec<f16> = kf.iter().map(|&v| f16::from_f32(v)).collect();
    let v: Vec<f16> = vf.iter().map(|&v| f16::from_f32(v)).collect();

    let dq = s.memcpy_stod(&q)?;
    let dk = s.memcpy_stod(&k)?;
    let dv = s.memcpy_stod(&v)?;
    let mut out = s.alloc_zeros::<f32>(n_heads * d_head)?;

    let f = dev.kernels().get("ops", &src(), "attn_decode_f32")?;
    let (nh, nkv, dh, kl) = (n_heads as i32, n_kv as i32, d_head as i32, kv_len as i32);
    let sc = scale;
    let mut b = s.launch_builder(&f);
    b.arg(&out.as_view_mut())
        .arg(&dq.as_view())
        .arg(&dk.as_view())
        .arg(&dv.as_view())
        .arg(&nh)
        .arg(&nkv)
        .arg(&dh)
        .arg(&kl)
        .arg(&sc);
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (n_heads as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })?
    };
    s.synchronize()?;

    let mut want = vec![0.0f32; n_heads * d_head];
    let group = n_heads / n_kv;
    for h in 0..n_heads {
        let kvh = h / group;
        let mut sc_row = vec![0.0f32; kv_len];
        for j in 0..kv_len {
            let mut dot = 0.0f32;
            for i in 0..d_head {
                dot += q[h * d_head + i] * k[j * stride + kvh * d_head + i].to_f32();
            }
            sc_row[j] = dot * scale;
        }
        let m = sc_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for x in sc_row.iter_mut() {
            *x = (*x - m).exp();
            sum += *x;
        }
        for i in 0..d_head {
            let mut acc = 0.0f32;
            for j in 0..kv_len {
                acc += sc_row[j] * v[j * stride + kvh * d_head + i].to_f32();
            }
            want[h * d_head + i] = acc / sum;
        }
    }
    close(&out.to_vec(), &want, 1e-4, "attn_decode_f32")
        ;
    Ok(())
}

#[test]
fn store_kv_then_read_it_back() -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();
    let (n_kv, d_head) = (2usize, 64usize);
    let total = n_kv * d_head;
    let max_pos = 8usize;

    let mut kc = s.alloc_zeros::<f16>(max_pos * total)?;
    let mut vc = s.alloc_zeros::<f16>(max_pos * total)?;
    let f = dev.kernels().get("ops", &src(), "store_kv_f16")?;

    let mut expect_k = vec![f16::ZERO; max_pos * total];
    for pos in 0..3usize {
        let k = noise(total, 100 + pos as u64);
        let v = noise(total, 200 + pos as u64);
        let dk = s.memcpy_stod(&k)?;
        let dv = s.memcpy_stod(&v)?;
        let (nkv, dh, p) = (n_kv as i32, d_head as i32, pos as i32);
        let mut b = s.launch_builder(&f);
        b.arg(&kc.as_view_mut())
            .arg(&vc.as_view_mut())
            .arg(&dk.as_view())
            .arg(&dv.as_view())
            .arg(&nkv)
            .arg(&dh)
            .arg(&p);
        unsafe { b.launch(grid1(total as u32, BLOCK))? };
        for i in 0..total {
            expect_k[pos * total + i] = f16::from_f32(k[i]);
        }
    }
    s.synchronize()?;

    let got = kc.to_vec();
    for i in 0..max_pos * total {
        assert_eq!(
            got[i], expect_k[i],
            "kcache[{i}] (position {}, channel {})",
            i / total,
            i % total
        );
    }
    Ok(())
}

#[test]
fn embed_reads_the_row_it_was_asked_for() -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();
    let (vocab, d) = (997usize, 896usize);

    let tf = noise(vocab * d, 31);
    let table: Vec<f16> = tf.iter().map(|&v| f16::from_f32(v)).collect();
    let dt = s.memcpy_stod(&table)?;
    let row = 613i32;
    let drow = s.memcpy_stod(&[row])?;
    let mut out = s.alloc_zeros::<f32>(d)?;

    let f = dev.kernels().get("ops", &src(), "embed_f16")?;
    let d_i = d as i32;
    let mut b = s.launch_builder(&f);
    b.arg(&out.as_view_mut())
        .arg(&dt.as_view())
        .arg(&drow.as_view())
        .arg(&d_i);
    unsafe { b.launch(grid1(d as u32, BLOCK))? };
    s.synchronize()?;

    let want: Vec<f32> = (0..d)
        .map(|i| table[row as usize * d + i].to_f32())
        .collect();
    close(&out.to_vec(), &want, 0.0, "embed_f16");
    Ok(())
}
