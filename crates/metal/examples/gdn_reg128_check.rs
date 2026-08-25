//! Correctness and speed of `gdn_delta_rule_reg128_f32` (register-resident,
//! ported from `cu/gdn.cu`'s `gdn_delta_rule_reg_body<128,128,2,4>`) against
//! the deployed `gdn_delta_rule_f32` (global memory).
//!
//! `DeltaVariant::Auto` resolves to `Reg` on CUDA at this checkpoint's shape
//! (dk = dv = 128) and to `Global` on every other backend, unconditionally --
//! Metal never got a `Reg` kernel, so it has been paying the 2x traffic
//! `cu/gdn.cu`'s own measurements document for that gap every prefill and
//! every decode step since the GDN port. `crates/kernels/tests/gated_delta.rs`
//! is the existing cross-check for this kernel family but pulls in
//! `tuili_model::qwen35`, which drags the workspace's CUDA-default feature
//! unification in with it (a known, separate issue) -- so this reimplements
//! the reference directly, small enough to trust by inspection: `S *=
//! exp(g); kv = kᵀS; delta = (v - kv) * beta; S += k ⊗ delta; o = qᵀS`, with
//! S read *after* its update, matching the note on `gdn_delta_rule_f32`.
//!
//!     cargo run --release -p tuili-metal --example gdn_reg128_check

use anyhow::Result;
use tuili_metal::{Buf, Device, LaunchConfig};

const COMMON: &str = include_str!("../../kernels/src/msl/common.metal");
const GDN: &str = include_str!("../../kernels/src/msl/gdn.metal");

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as i32 as f32) / (1i64 << 24) as f32
    }
}

fn ms(mut f: impl FnMut() -> Result<()>, iters: usize) -> Result<f64> {
    f()?;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        f()?;
    }
    Ok(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

/// The reference, over one (sequence, head): the same six lines the kernel
/// comment gives, computed in f64 so kernel-vs-reference error is legible
/// against the kernel's own f32-vs-f32 noise floor rather than getting lost
/// in reference rounding.
fn reference(
    qkv: &[f32],
    g: &[f32],
    beta: &[f32],
    nt: usize,
    heads: usize,
    head: usize,
    stride: usize,
    q_off: usize,
    k_off: usize,
    v_off: usize,
    khead: usize,
    dk: usize,
    dv: usize,
) -> (Vec<f32>, Vec<f64>) {
    let mut s = vec![0.0f64; dk * dv];
    let mut out = vec![0.0f32; nt * dv];
    for t in 0..nt {
        let row = &qkv[t * stride..(t + 1) * stride];
        let q = &row[q_off + khead * dk..q_off + khead * dk + dk];
        let k = &row[k_off + khead * dk..k_off + khead * dk + dk];
        let v = &row[v_off + head * dv..v_off + head * dv + dv];
        let decay = (g[t * heads + head] as f64).exp();
        let b = beta[t * heads + head] as f64;
        for i in 0..dk {
            for j in 0..dv {
                s[i * dv + j] *= decay;
            }
        }
        let mut kv = vec![0.0f64; dv];
        for i in 0..dk {
            let ki = k[i] as f64;
            for j in 0..dv {
                kv[j] += ki * s[i * dv + j];
            }
        }
        for j in 0..dv {
            let delta = (v[j] as f64 - kv[j]) * b;
            for i in 0..dk {
                s[i * dv + j] += k[i] as f64 * delta;
            }
        }
        for j in 0..dv {
            let mut o = 0.0f64;
            for i in 0..dk {
                o += q[i] as f64 * s[i * dv + j];
            }
            out[t * dv + j] = o as f32;
        }
    }
    (out, s)
}

fn main() -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();
    let gdn_src = format!("{COMMON}\n{GDN}");

    const DK: usize = 128;
    const DV: usize = 128;
    const HEADS: usize = 3;
    const KEY_HEADS: usize = 1;
    const STRIDE: usize = KEY_HEADS * DK * 2 + HEADS * DV;
    const Q_OFF: usize = 0;
    const K_OFF: usize = KEY_HEADS * DK;
    const V_OFF: usize = KEY_HEADS * DK * 2;

    let f_global = dev.kernels().get("gdn", &gdn_src, "gdn_delta_rule_f32")?;
    let f_reg = dev.kernels().get("gdn", &gdn_src, "gdn_delta_rule_reg128_f32")?;

    for &nt in &[1usize, 8, 53] {
        let mut rng = Rng(0x5EED ^ (nt as u64));
        let mut qkv: Vec<f32> = (0..nt * STRIDE).map(|_| rng.next_f32()).collect();
        // q and k reach this kernel already L2-normalized -- see the note on
        // `gdn_delta_rule_f32` -- and unnormalized random vectors are not a
        // shape the recurrence is stable under: it produced a fine match at
        // nt = 1 and 8 but a chaotic one at nt = 53, growing from 1e-7 to
        // 1e16 in a way that tracked sequence length rather than which
        // kernel ran, which is the signature of amplifying float noise
        // through an unstable recursion, not a kernel bug. Normalizing here
        // matches the regime the kernel actually runs in.
        for t in 0..nt {
            for (off, khead_count) in [(Q_OFF, KEY_HEADS), (K_OFF, KEY_HEADS)] {
                for kh in 0..khead_count {
                    let base = t * STRIDE + off + kh * DK;
                    let norm: f32 = qkv[base..base + DK].iter().map(|v| v * v).sum::<f32>().sqrt();
                    for v in &mut qkv[base..base + DK] {
                        *v /= norm.max(1e-6);
                    }
                }
            }
        }
        // Decay near 1 (small negative g) and modest beta, matching what a
        // trained checkpoint's gate actually produces -- g wildly negative
        // decays the state to zero in one step and hides a state bug.
        let g: Vec<f32> = (0..nt * HEADS).map(|_| -rng.next_f32().abs() * 0.1).collect();
        let beta: Vec<f32> = (0..nt * HEADS).map(|_| 0.3 + rng.next_f32().abs() * 0.4).collect();

        let d_qkv = s.clone_htod(&qkv)?;
        let d_g = s.clone_htod(&g)?;
        let d_beta = s.clone_htod(&beta)?;
        let first_token = s.clone_htod(&[0i32])?;
        let n_tok = s.clone_htod(&[nt as i32])?;

        let mut out_global = s.alloc_zeros::<f32>(nt * HEADS * DV)?;
        let mut state_global = s.alloc_zeros::<f32>(HEADS * DK * DV)?;
        let mut out_reg = s.alloc_zeros::<f32>(nt * HEADS * DV)?;
        let mut state_reg = s.alloc_zeros::<f32>(HEADS * DK * DV)?;

        let (h, kh) = (HEADS as i32, KEY_HEADS as i32);
        let (dk_i, dv_i) = (DK as i32, DV as i32);
        let (st, qo, ko, vo, vt) = (STRIDE as i32, Q_OFF as i32, K_OFF as i32, V_OFF as i32, 0i32);

        let launch_global = |out: &mut Buf<f32>, state: &mut Buf<f32>| -> Result<()> {
            let mut b = s.launch_builder(&f_global);
            b.arg(&out.as_view_mut())
                .arg(&state.as_view_mut())
                .arg(&d_qkv.as_view())
                .arg(&d_g.as_view())
                .arg(&d_beta.as_view())
                .arg(&first_token.as_view())
                .arg(&n_tok.as_view())
                .arg(&h).arg(&kh).arg(&dk_i).arg(&dv_i)
                .arg(&st).arg(&qo).arg(&ko).arg(&vo).arg(&vt);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (HEADS as u32, 1, 1),
                    block_dim: (DV.max(32) as u32, 1, 1),
                    shared_mem_bytes: (2 * DK * 4) as u32,
                })?
            };
            s.synchronize()
        };
        let launch_reg = |out: &mut Buf<f32>, state: &mut Buf<f32>| -> Result<()> {
            let mut b = s.launch_builder(&f_reg);
            b.arg(&out.as_view_mut())
                .arg(&state.as_view_mut())
                .arg(&d_qkv.as_view())
                .arg(&d_g.as_view())
                .arg(&d_beta.as_view())
                .arg(&first_token.as_view())
                .arg(&n_tok.as_view())
                .arg(&h).arg(&kh).arg(&dk_i).arg(&dv_i)
                .arg(&st).arg(&qo).arg(&ko).arg(&vo).arg(&vt);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (HEADS as u32, 1, 1),
                    block_dim: (2 * DV as u32, 1, 1),
                    shared_mem_bytes: (4 * DK * 4) as u32,
                })?
            };
            s.synchronize()
        };

        // Correctness: one fresh call each, state starting at zero -- the
        // kernel mutates `state` in place, so `ms`'s repeat-and-time pattern
        // below is wrong for a correctness check (each repeat would continue
        // the recurrence from the previous call's leftover state, not restart
        // it), even though it is exactly right for timing, which does not
        // care what is in `state`, only how long touching it takes.
        launch_global(&mut out_global, &mut state_global)?;
        launch_reg(&mut out_reg, &mut state_reg)?;

        // Timing: fresh zeroed buffers so `ms`'s warm-plus-repeat calls run
        // against comparable (if, after the first, recurrence-contaminated)
        // state either kernel would see in the same circumstance -- the
        // point is wall time, which does not depend on what is in `state`.
        let mut out_global_t = s.alloc_zeros::<f32>(nt * HEADS * DV)?;
        let mut state_global_t = s.alloc_zeros::<f32>(HEADS * DK * DV)?;
        let mut out_reg_t = s.alloc_zeros::<f32>(nt * HEADS * DV)?;
        let mut state_reg_t = s.alloc_zeros::<f32>(HEADS * DK * DV)?;
        let t_global = ms(|| launch_global(&mut out_global_t, &mut state_global_t), 5)?;
        let t_reg = ms(|| launch_reg(&mut out_reg_t, &mut state_reg_t), 5)?;

        let got_global = s.clone_dtoh(&out_global)?;
        let got_reg = s.clone_dtoh(&out_reg)?;
        let got_state_global = s.clone_dtoh(&state_global)?;
        let got_state_reg = s.clone_dtoh(&state_reg)?;

        let mut max_diff_global = 0.0f32;
        let mut max_diff_reg = 0.0f32;
        let mut max_diff_state_global = 0.0f32;
        let mut max_diff_state_reg = 0.0f32;
        for head in 0..HEADS {
            let khead = head / (HEADS / KEY_HEADS).max(1);
            let (want, want_state) = reference(
                &qkv, &g, &beta, nt, HEADS, head, STRIDE, Q_OFF, K_OFF, V_OFF, khead, DK, DV,
            );
            for t in 0..nt {
                for j in 0..DV {
                    let idx = (t * HEADS + head) * DV + j;
                    max_diff_global = max_diff_global.max((got_global[idx] - want[t * DV + j]).abs());
                    max_diff_reg = max_diff_reg.max((got_reg[idx] - want[t * DV + j]).abs());
                }
            }
            for i in 0..DK {
                for j in 0..DV {
                    let idx = head * DK * DV + i * DV + j;
                    let w = want_state[i * DV + j] as f32;
                    max_diff_state_global = max_diff_state_global.max((got_state_global[idx] - w).abs());
                    max_diff_state_reg = max_diff_state_reg.max((got_state_reg[idx] - w).abs());
                }
            }
        }
        println!(
            "nt={nt:3}  global {t_global:7.4}ms (out diff {max_diff_global:.2e}, state diff {max_diff_state_global:.2e})  \
             reg {t_reg:7.4}ms (out diff {max_diff_reg:.2e}, state diff {max_diff_state_reg:.2e})  speedup {:.2}x",
            t_global / t_reg,
        );
    }
    Ok(())
}
