//! The quantized mat-vecs against a host dequantization of real tensors.
//!
//! The block formats are the one part of this port with no reference capture
//! behind them, so the oracle is a host implementation written from ggml's
//! `dequantize_row_*` rather than from the kernel it is checking. A kernel and
//! a reference that share a misreading of the layout agree with each other and
//! with nothing else, which is why this reads the real file: a synthetic block
//! built by the same code that decodes it proves nothing.
//!
//! Needs the 27B checkpoint; skips without it.

use anyhow::Result;
use half::f16;
use tuili_gguf::{GgmlType, Gguf};
use tuili_metal::{Device, LaunchConfig};

const COMMON: &str = include_str!("../../kernels/src/msl/common.metal");
const QUANT: &str = include_str!("../../kernels/src/msl/quant.metal");

fn src() -> String {
    format!("{COMMON}\n{QUANT}")
}

fn model() -> Option<Gguf> {
    let p = std::env::var("TUILI_TEST_GGUF").unwrap_or_else(|_| {
        format!(
            "{}/../../models/Qwen3.8-27B-Q4_K_M.gguf",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    if !std::path::Path::new(&p).exists() {
        eprintln!("skipping: {p} not downloaded");
        return None;
    }
    Gguf::open(&p).ok()
}

fn h(b: &[u8]) -> f32 {
    f16::from_le_bytes([b[0], b[1]]).to_f32()
}

/// ggml's `get_scale_min_k4`.
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

/// One row, dequantized. Written from `dequantize_row_*` in ggml-quants.c.
fn dequant_row(ty: GgmlType, raw: &[u8], k: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(k);
    match ty {
        GgmlType::F32 => {
            for c in raw.chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        GgmlType::F16 => {
            for c in raw.chunks_exact(2) {
                out.push(h(c));
            }
        }
        GgmlType::Q8_0 => {
            for blk in raw.chunks_exact(34) {
                let d = h(&blk[0..2]);
                for i in 0..32 {
                    out.push(d * (blk[2 + i] as i8) as f32);
                }
            }
        }
        GgmlType::Q4K => {
            for blk in raw.chunks_exact(144) {
                let d = h(&blk[0..2]);
                let dmin = h(&blk[2..4]);
                let scales = &blk[4..16];
                let qs = &blk[16..144];
                // Four 64-element chunks; within each, 32 low nibbles then 32
                // high nibbles of the *same* 32 bytes.
                for chunk in 0..4 {
                    let q = &qs[chunk * 32..chunk * 32 + 32];
                    for half in 0..2 {
                        let g = chunk * 2 + half;
                        let (sc, m) = q4k_scale_min(scales, g);
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
                let ql = &blk[0..128];
                let qh = &blk[128..192];
                let sc: Vec<i8> = blk[192..208].iter().map(|&b| b as i8).collect();
                let d = h(&blk[208..210]);
                let mut y = vec![0.0f32; 256];
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
    out
}

fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

/// Check `gemv` on the first `rows` rows of a real tensor.
fn check(g: &Gguf, name: &str, kernel: &str, rows: usize, tol: f32) -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();
    let t = g.tensor(name)?;
    let (k, n) = (t.dims[0] as usize, t.dims[1] as usize);
    let rows = rows.min(n);
    let bytes_per_row = t.n_bytes / n;
    let raw = &g.data(t)[..rows * bytes_per_row];

    let x = noise(k, 4242);
    let dw = s.memcpy_stod(raw)?;
    let dx = s.memcpy_stod(&x)?;
    let mut out = s.alloc_zeros::<f32>(rows)?;

    let f = dev.kernels().get("quant", &src(), kernel)?;
    let (ki, ni, ti) = (k as i32, rows as i32, 1i32);
    let mut b = s.launch_builder(&f);
    b.arg(&out.as_view_mut())
        .arg(&dw.as_view())
        .arg(&dx.as_view())
        .arg(&ki)
        .arg(&ni)
        .arg(&ti);
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })?
    };
    s.synchronize()?;

    let got = out.to_vec();
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for r in 0..rows {
        let w = dequant_row(t.ty, &raw[r * bytes_per_row..(r + 1) * bytes_per_row], k);
        let want: f32 = (0..k).map(|i| w[i] * x[i]).sum();
        let e = (got[r] - want).abs() / want.abs().max(1.0);
        if e > worst {
            worst = e;
            at = r;
        }
    }
    eprintln!("  {name:34} {kernel:12} {rows} rows, worst {worst:.3e}");
    assert!(worst <= tol, "{name} via {kernel}: {worst:.3e} at row {at}");
    Ok(())
}

#[test]
fn the_quantized_matvecs_match_a_host_dequant() -> Result<()> {
    let Some(g) = model() else { return Ok(()) };
    // One tensor of each encoding the checkpoint actually uses.
    check(&g, "blk.0.attn_qkv.weight", "gemv_q8_0", 64, 2e-3)?;
    check(&g, "blk.0.ffn_gate.weight", "gemv_q4_K", 64, 2e-3)?;
    check(&g, "output.weight", "gemv_q6_K", 64, 2e-3)?;
    check(&g, "token_embd.weight", "gemv_q4_K", 64, 2e-3)?;
    Ok(())
}

#[test]
fn the_embedding_row_matches_a_host_dequant() -> Result<()> {
    let Some(g) = model() else { return Ok(()) };
    let dev = Device::new(0)?;
    let s = dev.stream();
    let t = g.tensor("token_embd.weight")?;
    let (k, n) = (t.dims[0] as usize, t.dims[1] as usize);
    assert_eq!(t.ty, GgmlType::Q4K);
    let bytes_per_row = t.n_bytes / n;

    // A few rows spread across the vocabulary rather than row zero, which is a
    // special token and unrepresentative.
    for &row in &[0usize, 785, 12095, 104455, n - 1] {
        let raw = &g.data(t)[row * bytes_per_row..(row + 1) * bytes_per_row];
        let want = dequant_row(GgmlType::Q4K, raw, k);

        let dw = s.memcpy_stod(g.data(t))?;
        let idx = s.memcpy_stod(&[row as i32])?;
        let mut out = s.alloc_zeros::<f32>(k)?;
        let f = dev.kernels().get("quant", &src(), "embed_row_q4_K")?;
        let ki = k as i32;
        let mut b = s.launch_builder(&f);
        b.arg(&out.as_view_mut())
            .arg(&dw.as_view())
            .arg(&idx.as_view())
            .arg(&ki);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: ((k as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?
        };
        s.synchronize()?;
        let got = out.to_vec();
        let mut worst = 0.0f32;
        for i in 0..k {
            worst = worst.max((got[i] - want[i]).abs());
        }
        eprintln!("  embed row {row:>6}: worst absolute {worst:.3e}");
        assert!(worst < 1e-6, "embedding row {row} differs by {worst:.3e}");
    }
    Ok(())
}
