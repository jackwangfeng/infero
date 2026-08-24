//! The kernels the engine takes that the vertical-slice examples do not.
//!
//! The examples use the plain `rms_norm_f32` and a contiguous KV cache; the
//! engine prefers the register-resident norms and addresses a paged pool
//! through a slot table. Those are a separate population and they get their own
//! oracle: where a plain kernel computes the same function, the plain one *is*
//! the reference, because it is the one the fixture tests already vouch for.

use anyhow::Result;
use tuili_metal::{Device, LaunchConfig};

const COMMON: &str = include_str!("../../kernels/src/msl/common.metal");
const OPS: &str = include_str!("../../kernels/src/msl/ops.metal");
const MMVQ: &str = include_str!("../../kernels/src/msl/mmvq.metal");

fn ops() -> String {
    format!("{COMMON}\n{OPS}")
}
fn mmvq() -> String {
    format!("{COMMON}\n{MMVQ}")
}

/// `RMS_REGS` in the MSL; the host sizes the block so `block * REGS >= d`.
const RMS_REGS: usize = 8;

fn rms_block(d: usize) -> u32 {
    ((d as u32).div_ceil(RMS_REGS as u32))
        .next_multiple_of(32)
        .clamp(32, 1024)
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

fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for i in 0..got.len() {
        let e = (got[i] - want[i]).abs() / want[i].abs().max(1.0);
        if e > worst {
            worst = e;
            at = i;
        }
    }
    eprintln!("  {what:32} worst {worst:.3e}");
    assert!(
        worst <= tol,
        "{what}: {worst:.3e} at {at} (got {}, want {})",
        got[at],
        want[at]
    );
}

/// The register-resident norm against the plain one, over the shapes the
/// engine actually launches: `d_model` of a real model, several token counts.
#[test]
fn the_register_resident_norm_matches_the_plain_one() -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();
    for &d in &[896usize, 5120] {
        for &n_tokens in &[1usize, 3, 41] {
            let x = noise(n_tokens * d, 7 + d as u64);
            let w = noise(d, 99);
            let dx = s.memcpy_stod(&x)?;
            let dw = s.memcpy_stod(&w)?;
            let mut plain = s.alloc_zeros::<f32>(n_tokens * d)?;
            let mut fused = s.alloc_zeros::<f32>(n_tokens * d)?;
            let mut h = s.alloc_zeros::<half::f16>(n_tokens * d)?;
            let (d_i, eps) = (d as i32, 1e-6f32);

            let f = dev.kernels().get("ops", &ops(), "rms_norm_f32")?;
            let mut b = s.launch_builder(&f);
            b.arg(&plain.as_view_mut())
                .arg(&dx.as_view())
                .arg(&dw.as_view())
                .arg(&d_i)
                .arg(&eps);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (n_tokens as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?
            };

            let f = dev.kernels().get("mmvq", &mmvq(), "rms_norm_f16_f32")?;
            let mut b = s.launch_builder(&f);
            b.arg(&fused.as_view_mut())
                .arg(&h.as_view_mut())
                .arg(&dx.as_view())
                .arg(&dw.as_view())
                .arg(&d_i)
                .arg(&eps);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (n_tokens as u32, 1, 1),
                    block_dim: (rms_block(d), 1, 1),
                    shared_mem_bytes: 0,
                })?
            };
            s.synchronize()?;

            close(
                &fused.to_vec(),
                &plain.to_vec(),
                2e-5,
                &format!("rms_norm_f16_f32 d={d} n={n_tokens}"),
            );
            // The f16 copy is the same values, narrowed.
            let hv: Vec<f32> = h.to_vec().iter().map(|v| v.to_f32()).collect();
            close(
                &hv,
                &plain.to_vec(),
                2e-3,
                &format!("  its f16 copy   d={d} n={n_tokens}"),
            );
        }
    }
    Ok(())
}

/// The same, with the residual folded in on the way through.
#[test]
fn the_fused_residual_norm_matches_add_then_norm() -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();
    let (d, n_tokens) = (896usize, 3usize);
    let x = noise(n_tokens * d, 11);
    let resid = noise(n_tokens * d, 13);
    let w = noise(d, 17);

    // Reference: add on the host, then the plain norm on the device.
    let summed: Vec<f32> = x.iter().zip(&resid).map(|(a, b)| a + b).collect();
    let dsum = s.memcpy_stod(&summed)?;
    let dw = s.memcpy_stod(&w)?;
    let mut plain = s.alloc_zeros::<f32>(n_tokens * d)?;
    let (d_i, eps) = (d as i32, 1e-6f32);
    let f = dev.kernels().get("ops", &ops(), "rms_norm_f32")?;
    let mut b = s.launch_builder(&f);
    b.arg(&plain.as_view_mut())
        .arg(&dsum.as_view())
        .arg(&dw.as_view())
        .arg(&d_i)
        .arg(&eps);
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (n_tokens as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })?
    };

    let mut dx = s.memcpy_stod(&x)?;
    let dr = s.memcpy_stod(&resid)?;
    let mut fused = s.alloc_zeros::<f32>(n_tokens * d)?;
    let mut h = s.alloc_zeros::<half::f16>(n_tokens * d)?;
    let f = dev.kernels().get("mmvq", &mmvq(), "add_rms_norm_f16_f32")?;
    let mut b = s.launch_builder(&f);
    b.arg(&fused.as_view_mut())
        .arg(&h.as_view_mut())
        .arg(&dx.as_view_mut())
        .arg(&dr.as_view())
        .arg(&dw.as_view())
        .arg(&d_i)
        .arg(&eps);
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (n_tokens as u32, 1, 1),
            block_dim: (rms_block(d), 1, 1),
            shared_mem_bytes: 0,
        })?
    };
    s.synchronize()?;

    close(&fused.to_vec(), &plain.to_vec(), 2e-5, "add_rms_norm_f16_f32");
    // It also writes the sum back through `x`, which the next layer's residual
    // depends on.
    close(&dx.to_vec(), &summed, 1e-6, "  the residual written back");
    Ok(())
}

/// The nil binding: asking for no f16 copy must not write one, and must not
/// crash. The CUDA side passes a `u64` zero; Metal binds nil.
#[test]
fn a_norm_with_no_f16_copy_runs() -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();
    let d = 896usize;
    let x = noise(d, 23);
    let w = noise(d, 29);
    let dx = s.memcpy_stod(&x)?;
    let dw = s.memcpy_stod(&w)?;
    let mut out = s.alloc_zeros::<f32>(d)?;
    let (d_i, eps) = (d as i32, 1e-6f32);

    let f = dev.kernels().get("mmvq", &mmvq(), "rms_norm_f16_f32")?;
    let mut b = s.launch_builder(&f);
    b.arg(&out.as_view_mut())
        .arg(&tuili_metal::NullBuffer)
        .arg(&dx.as_view())
        .arg(&dw.as_view())
        .arg(&d_i)
        .arg(&eps);
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (rms_block(d), 1, 1),
            shared_mem_bytes: 0,
        })?
    };
    s.synchronize()?;

    let ss: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let scale = 1.0 / (ss / d as f64 + 1e-6).sqrt();
    let want: Vec<f32> = (0..d).map(|i| (x[i] as f64 * scale) as f32 * w[i]).collect();
    close(&out.to_vec(), &want, 2e-5, "rms_norm_f16_f32 with nil hout");
    Ok(())
}

/// The paged attention trio against a host reference.
///
/// The slot table is the whole point of this path and the whole risk: a
/// sequence's key `j` lives at `slot_table[seq][j]`, which is an arbitrary
/// index into the pool rather than `j`. Getting that identity-mapped would pass
/// on any test whose table happens to be the identity, so this one deliberately
/// scrambles it.
#[test]
fn the_paged_attention_matches_the_host() -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();
    let (n_heads, n_kv, d_head) = (14usize, 2usize, 64usize);
    let (n_tokens, kv_len, n_slots) = (3usize, 7usize, 64usize);
    let table_stride = 128usize;
    let scale = 1.0f32 / (d_head as f32).sqrt();

    let q = noise(n_tokens * n_heads * d_head, 31);
    // A scrambled table: key j of sequence 0 lives at slot (j * 13 + 5) % 64.
    let mut table = vec![0i32; table_stride];
    for j in 0..kv_len {
        table[j] = ((j * 13 + 5) % n_slots) as i32;
    }
    let kf = noise(n_kv * n_slots * d_head, 32);
    let vf = noise(n_kv * n_slots * d_head, 33);
    let k: Vec<half::f16> = kf.iter().map(|&v| half::f16::from_f32(v)).collect();
    let v: Vec<half::f16> = vf.iter().map(|&v| half::f16::from_f32(v)).collect();
    // Different lengths in one batch, which is the reason the mask is per token.
    let positions: Vec<i32> = vec![2, 4, 6];
    let seq_of: Vec<i32> = vec![0, 0, 0];

    let dq = s.memcpy_stod(&q)?;
    let dk = s.memcpy_stod(&k)?;
    let dv = s.memcpy_stod(&v)?;
    let dtab = s.memcpy_stod(&table)?;
    let dpos = s.memcpy_stod(&positions)?;
    let dseq = s.memcpy_stod(&seq_of)?;
    let mut scores = s.alloc_zeros::<f32>(n_heads * n_tokens * kv_len)?;
    let mut out = s.alloc_zeros::<f32>(n_tokens * n_heads * d_head)?;

    let (ts, nh, nkv, dh, ns, kl) = (
        table_stride as i32,
        n_heads as i32,
        n_kv as i32,
        d_head as i32,
        n_slots as i32,
        kv_len as i32,
    );
    {
        let f = dev.kernels().get("ops", &ops(), "attn_scores_f32")?;
        let sc = scale;
        let mut b = s.launch_builder(&f);
        b.arg(&scores.as_view_mut())
            .arg(&dq.as_view())
            .arg(&dk.as_view())
            .arg(&dseq.as_view())
            .arg(&dpos.as_view())
            .arg(&dtab.as_view())
            .arg(&ts)
            .arg(&nh)
            .arg(&nkv)
            .arg(&dh)
            .arg(&ns)
            .arg(&kl)
            .arg(&sc);
        // Four SIMD groups a threadgroup, one key each.
        let warps = 4u32;
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: ((kv_len as u32).div_ceil(warps), n_heads as u32, n_tokens as u32),
                block_dim: (warps * 32, 1, 1),
                shared_mem_bytes: 0,
            })?
        };
    }
    {
        let f = dev.kernels().get("ops", &ops(), "attn_softmax_f32")?;
        let mut b = s.launch_builder(&f);
        b.arg(&scores.as_view_mut()).arg(&kl);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (n_heads as u32, n_tokens as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })?
        };
    }
    {
        let f = dev.kernels().get("ops", &ops(), "attn_output_f32")?;
        let mut b = s.launch_builder(&f);
        b.arg(&out.as_view_mut())
            .arg(&scores.as_view())
            .arg(&dv.as_view())
            .arg(&dseq.as_view())
            .arg(&dpos.as_view())
            .arg(&dtab.as_view())
            .arg(&ts)
            .arg(&nh)
            .arg(&nkv)
            .arg(&dh)
            .arg(&ns)
            .arg(&kl);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (n_heads as u32, n_tokens as u32, 1),
                block_dim: (d_head as u32, 1, 1),
                shared_mem_bytes: 0,
            })?
        };
    }
    s.synchronize()?;

    let group = n_heads / n_kv;
    let mut want = vec![0.0f32; n_tokens * n_heads * d_head];
    for t in 0..n_tokens {
        for h in 0..n_heads {
            let kvh = h / group;
            let last = positions[t] as usize;
            let mut sc = vec![0.0f32; last + 1];
            for j in 0..=last {
                let slot = table[j] as usize;
                let mut dot = 0.0f32;
                for i in 0..d_head {
                    dot += q[(t * n_heads + h) * d_head + i]
                        * k[(kvh * n_slots + slot) * d_head + i].to_f32();
                }
                sc[j] = dot * scale;
            }
            let m = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for e in sc.iter_mut() {
                *e = (*e - m).exp();
                sum += *e;
            }
            for i in 0..d_head {
                let mut acc = 0.0f32;
                for j in 0..=last {
                    let slot = table[j] as usize;
                    acc += sc[j] / sum * v[(kvh * n_slots + slot) * d_head + i].to_f32();
                }
                want[(t * n_heads + h) * d_head + i] = acc;
            }
        }
    }
    close(&out.to_vec(), &want, 2e-4, "paged attention (scores/softmax/output)");
    Ok(())
}

/// The packed-projection path: rope in place out of a fused `[q|k|v]` row, and
/// the KV store that reads the same row.
///
/// This is what the engine actually launches -- the fused projection writes one
/// row per token and these two read it at offsets -- so the unpacked
/// `rope_qk_f32` and `store_kv2_f16` never run in a real step.
#[test]
fn the_packed_rope_and_kv_store_match_the_host() -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();
    let (n_heads, n_kv, d_head) = (14usize, 2usize, 64usize);
    let (rotary, n_tokens) = (64usize, 2usize);
    let (d, kv_dim) = (n_heads * d_head, n_kv * d_head);
    let stride = d + 2 * kv_dim;
    let (q_off, k_off, v_off) = (0usize, d, d + kv_dim);
    let theta = 1e6f32;

    let packed = noise(n_tokens * stride, 41);
    let positions: Vec<i32> = vec![5, 9];
    let freq = vec![1.0f32; rotary / 2];

    let mut dpacked = s.memcpy_stod(&packed)?;
    let dpos = s.memcpy_stod(&positions)?;
    let dfreq = s.memcpy_stod(&freq)?;
    let mut q_dst = s.alloc_zeros::<f32>(n_tokens * d)?;

    {
        let f = dev.kernels().get("ops", &ops(), "rope_qk_packed_f32")?;
        let (st, qo, ko) = (stride as i32, q_off as i32, k_off as i32);
        let (nh, nkv, dh, rot) = (n_heads as i32, n_kv as i32, d_head as i32, rotary as i32);
        let (th, fs, il) = (theta, 1.0f32, 0i32);
        let mut b = s.launch_builder(&f);
        b.arg(&q_dst.as_view_mut())
            .arg(&dpacked.as_view_mut())
            .arg(&st)
            .arg(&qo)
            .arg(&ko)
            .arg(&dpos.as_view())
            .arg(&dfreq.as_view())
            .arg(&nh)
            .arg(&nkv)
            .arg(&dh)
            .arg(&rot)
            .arg(&th)
            .arg(&fs)
            .arg(&il);
        let lanes = (rotary / 2 + (d_head - rotary)) as u32;
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (1, (n_heads + n_kv) as u32, n_tokens as u32),
                block_dim: (lanes, 1, 1),
                shared_mem_bytes: 0,
            })?
        };
    }
    s.synchronize()?;

    // Host reference.
    let half = rotary / 2;
    let mut want_packed = packed.clone();
    let mut want_q = vec![0.0f32; n_tokens * d];
    for t in 0..n_tokens {
        for y in 0..(n_heads + n_kv) {
            let is_q = y < n_heads;
            let head = if is_q { y } else { y - n_heads };
            let src_base = t * stride + if is_q { q_off } else { k_off } + head * d_head;
            for i in 0..half {
                let inv = theta.powf(-2.0 * i as f32 / rotary as f32);
                let ang = positions[t] as f32 * inv;
                let (sa, ca) = (ang.sin(), ang.cos());
                let (a, b) = (packed[src_base + i], packed[src_base + i + half]);
                let (ra, rb) = (a * ca - b * sa, a * sa + b * ca);
                if is_q {
                    want_q[(t * n_heads + head) * d_head + i] = ra;
                    want_q[(t * n_heads + head) * d_head + i + half] = rb;
                } else {
                    want_packed[src_base + i] = ra;
                    want_packed[src_base + i + half] = rb;
                }
            }
            // q's unrotated tail is copied across; here rotary == d_head so
            // there is none, which is the shape this model launches.
        }
    }
    close(&q_dst.to_vec(), &want_q, 2e-5, "rope_qk_packed_f32 (q out)");
    close(&dpacked.to_vec(), &want_packed, 2e-5, "  k rotated in place");

    // And the KV store off the same row.
    let n_slots = 32usize;
    let slots: Vec<i32> = vec![7, 19];
    let dslots = s.memcpy_stod(&slots)?;
    let mut kpool = s.alloc_zeros::<half::f16>(n_kv * n_slots * d_head)?;
    let mut vpool = s.alloc_zeros::<half::f16>(n_kv * n_slots * d_head)?;
    {
        let f = dev.kernels().get("ops", &ops(), "store_kv2_packed_f16")?;
        let (st, ko, vo) = (stride as i32, k_off as i32, v_off as i32);
        let (nkv, dh, ns, nt) = (n_kv as i32, d_head as i32, n_slots as i32, n_tokens as i32);
        let mut b = s.launch_builder(&f);
        b.arg(&kpool.as_view_mut())
            .arg(&vpool.as_view_mut())
            .arg(&dpacked.as_view())
            .arg(&st)
            .arg(&ko)
            .arg(&vo)
            .arg(&dslots.as_view())
            .arg(&nkv)
            .arg(&dh)
            .arg(&ns)
            .arg(&nt);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (1, (2 * n_kv) as u32, n_tokens as u32),
                block_dim: (d_head as u32, 1, 1),
                shared_mem_bytes: 0,
            })?
        };
    }
    s.synchronize()?;

    let got_k = kpool.to_vec();
    let got_v = vpool.to_vec();
    let rotated = dpacked.to_vec();
    for t in 0..n_tokens {
        for h in 0..n_kv {
            for i in 0..d_head {
                let slot = slots[t] as usize;
                let dst = (h * n_slots + slot) * d_head + i;
                let ks = t * stride + k_off + h * d_head + i;
                let vs = t * stride + v_off + h * d_head + i;
                assert_eq!(
                    got_k[dst],
                    half::f16::from_f32(rotated[ks]),
                    "k pool token {t} head {h} channel {i}"
                );
                assert_eq!(
                    got_v[dst],
                    half::f16::from_f32(rotated[vs]),
                    "v pool token {t} head {h} channel {i}"
                );
            }
        }
    }
    eprintln!("  store_kv2_packed_f16              exact");
    Ok(())
}

/// The embedding gather, which is the first kernel of every forward pass.
///
/// Macro-generated over a per-element decoder, so a mistake in the macro or the
/// decoder is a mistake in every weight type at once -- and it lands on the
/// very first thing the model does, which makes everything downstream noise.
#[test]
fn the_embedding_gather_matches_the_host() -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();
    const QUANT: &str = include_str!("../../kernels/src/msl/quant.metal");
    let quant = format!("{COMMON}\n{QUANT}");

    let (vocab, k) = (997usize, 896usize);
    let tf = noise(vocab * k, 51);
    let table: Vec<half::f16> = tf.iter().map(|&v| half::f16::from_f32(v)).collect();
    let rows: Vec<i32> = vec![0, 613, 42, (vocab - 1) as i32];

    let dt = s.memcpy_stod(&table)?;
    let dr = s.memcpy_stod(&rows)?;
    let mut out = s.alloc_zeros::<f32>(rows.len() * k)?;

    let f = dev.kernels().get("quant", &quant, "gather_rows_f16")?;
    let k_i = k as i32;
    let mut b = s.launch_builder(&f);
    b.arg(&out.as_view_mut())
        .arg(&dt.as_view())
        .arg(&dr.as_view())
        .arg(&k_i);
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: ((k as u32).div_ceil(256), rows.len() as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })?
    };
    s.synchronize()?;

    let mut want = vec![0.0f32; rows.len() * k];
    for (t, &r) in rows.iter().enumerate() {
        for i in 0..k {
            want[t * k + i] = table[r as usize * k + i].to_f32();
        }
    }
    close(&out.to_vec(), &want, 0.0, "gather_rows_f16");
    Ok(())
}

/// The mat-vec at more than one token.
///
/// The examples only ever ask for one, and so does decode -- but prefill asks
/// for the whole prompt at once, and without a library GEMM this backend takes
/// the mat-vec for all of it. One threadgroup holds `GEMV_TOKENS` tokens and
/// the grid's second dimension covers the rest, which is the arrangement no
/// test had exercised.
#[test]
fn the_matvec_matches_the_host_at_many_tokens() -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();
    const QUANT: &str = include_str!("../../kernels/src/msl/quant.metal");
    let quant = format!("{COMMON}\n{QUANT}");

    let (k, n) = (896usize, 137usize);
    let wf = noise(k * n, 61);
    let w: Vec<half::f16> = wf.iter().map(|&v| half::f16::from_f32(v)).collect();
    let dw = s.memcpy_stod(&w)?;

    // 1 is decode, 8 is exactly one threadgroup's worth, 9 and 41 need more
    // than one -- which is where the grid's second dimension starts mattering.
    for &n_tokens in &[1usize, 3, 8, 9, 41] {
        let x = noise(n_tokens * k, 67);
        let dx = s.memcpy_stod(&x)?;
        let mut out = s.alloc_zeros::<f32>(n_tokens * n)?;

        let f = dev.kernels().get("quant", &quant, "gemv_f16")?;
        let (k_i, n_i, t_i) = (k as i32, n as i32, n_tokens as i32);
        let mut b = s.launch_builder(&f);
        b.arg(&out.as_view_mut())
            .arg(&dw.as_view())
            .arg(&dx.as_view())
            .arg(&k_i)
            .arg(&n_i)
            .arg(&t_i);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (n as u32, (n_tokens as u32).div_ceil(8).max(1), 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?
        };
        s.synchronize()?;

        // `out[t * n + r] = dot(w[r, :], x[t, :])`
        let mut want = vec![0.0f32; n_tokens * n];
        for t in 0..n_tokens {
            for r in 0..n {
                want[t * n + r] = (0..k)
                    .map(|i| w[r * k + i].to_f32() * x[t * k + i])
                    .sum();
            }
        }
        close(
            &out.to_vec(),
            &want,
            2e-3,
            &format!("gemv_f16 n_tokens={n_tokens}"),
        );
    }
    Ok(())
}
