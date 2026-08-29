//! `MtpHead`'s M-RoPE plumbing (`HeadDims::mrope_section` -> the head's own
//! `mrope_axis` buffer, `DraftFeed::mrope` -> `MtpHead::prime`/`step`'s
//! `mrope` argument) actually reaches the rope kernel through the head's real
//! buffer sizing and per-chunk upload, not just the raw kernel call
//! `crates/kernels/tests/mrope.rs` already covers.
//!
//! Both tests measure on **one** head reused across runs (`truncate(0)`
//! between them, the same pattern `tests/qwen35_mtp_device.rs` uses to
//! re-probe a head without rebuilding it), which is what this file caught a
//! real bug with: `run`'s `pos_stride` used to follow `mrope.is_some()` --
//! whether *this call* passed a real array -- rather than
//! `dims.mrope_section.is_some()` -- whether the *head* has one at all. On a
//! head with M-RoPE, `mrope_axis` names axis 1 or 2 for some frequencies
//! regardless of the call, so a `pos_stride: 1` call (`mrope: None`, meant to
//! mean "plain decode-phase step") had those frequencies read
//! `self.positions[token + 1]` / `[token + 2]` -- a different token's
//! position, since `self.positions` is one value a token. `equal_axes...`
//! below is exactly the case that surfaced it: T=H=W should reduce to the
//! scalar path bit for bit, and instead diverged by up to 0.21, identical in
//! shape to `different_axes...`'s genuine signal. Fixed in `MtpHead::run` by
//! making `pos_stride` (and broadcasting `[p,p,p]` for a `None` mrope) a
//! property of the head, matching how `Acts::mrope_positions` already works
//! on the target model's decode path.
//!
//! What the two tests establish now that the fix is in:
//!
//!  * T=H=W degenerates to exactly the scalar-position case, bit for bit --
//!    the same reduction `crates/kernels/tests/mrope.rs`'s
//!    `equal_axes_reduce_to_the_scalar_case` proves at the kernel level, now
//!    proven through the head's own buffer plumbing end to end;
//!  * genuinely different axes produce a genuinely different output, so a
//!    caller that built a real `mrope` array and a head that silently
//!    dropped it are distinguishable rather than merely both plausible.
//!
//! Self-contained, the same reasoning as `tests/mtp_fork.rs`: a mechanism
//! this easy to get silently wrong should not be tested only when a capture
//! fixture happens to be present.

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::Kernels;
use infero_model::mtp::{HeadDims, MtpHead};
use infero_model::weights::{AttnWeights, DenseFfn, Layer, Matrix, MtpWeights};

/// Both tests below build their own `Device`; without this they run on
/// separate threads by default and share the one GPU context. Same pattern
/// `tests/spec.rs`/`tests/qwen35_mtp_device.rs` use for the same reason.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// `mrope_section: Some([2, 3, 3])` sums to 8 == `rotary_dim / 2` -- most of
/// the rotation reads H or W, so a divergent H/W is not diluted down to a
/// couple of the head's 32 dimensions the way an even 3-way split would be.
fn dims() -> HeadDims {
    HeadDims {
        d_model: 128,
        heads: 4,
        kv_heads: 2,
        d_head: 32,
        rotary_dim: 16,
        d_ff: 256,
        eps: 1e-6,
        rope_theta: 10_000.0,
        vocab: 96,
        mrope_section: Some([2, 3, 3]),
    }
}

fn synth(dev: &Device, dims: HeadDims) -> Result<(MtpWeights, Matrix)> {
    let seed = std::cell::Cell::new(0x1234_5678u32);
    let mut next = move || {
        seed.set(seed.get().wrapping_mul(1_664_525).wrapping_add(1_013_904_223));
        (seed.get() >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    };
    let (d, da, dkv) = (dims.d_model, dims.d_attn(), dims.d_kv());
    let m = |k: usize, n: usize| -> Result<Matrix> {
        let v: Vec<half::f16> = (0..k * n).map(|_| half::f16::from_f32(next() * 0.3)).collect();
        Matrix::upload_f16(dev, &v, k, n)
    };
    let vec1 = |n: usize| -> Result<infero_model::weights::Vector> {
        let v: Vec<f32> = (0..n).map(|_| 1.0 + next() * 0.1).collect();
        Ok(dev.stream().clone_htod(&v)?)
    };
    let w = MtpWeights {
        fc: m(2 * d, d)?,
        pre_fc_norm_embedding: vec1(d)?,
        pre_fc_norm_hidden: vec1(d)?,
        norm: vec1(d)?,
        layer: Layer {
            attn_norm: vec1(d)?,
            attn: Some(AttnWeights {
                wq: m(d, 2 * da)?,
                wk: m(d, dkv)?,
                wv: m(d, dkv)?,
                wo: m(da, d)?,
                bq: None,
                bk: None,
                bv: None,
                bo: None,
                q_norm: Some(vec1(dims.d_head)?),
                k_norm: Some(vec1(dims.d_head)?),
                w_qkv: None,
                w_kv: None,
                output_gate: true,
            }),
            gdn: None,
            ffn_norm: vec1(d)?,
            dense: Some(DenseFfn {
                w_gate: m(d, dims.d_ff)?,
                w_up: m(d, dims.d_ff)?,
                w_down: m(dims.d_ff, d)?,
                w_gate_up: None,
            }),
            moe: None,
            blob: None,
        },
        device_bytes: 0,
    };
    let embed = m(d, dims.vocab)?;
    Ok((w, embed))
}

const T: usize = 5;

/// `head`, freshly `truncate(0)`d, stepped once at `mrope` and read back.
fn step_logits(
    head: &mut MtpHead,
    kern: &Kernels,
    embed: &Matrix,
    ids: &[u32],
    positions: &[usize],
    hidden: &infero_gpu::View<'_, f32>,
    mrope: Option<&[i32]>,
) -> Result<Vec<Vec<f32>>> {
    head.truncate(0);
    head.step(kern, embed, ids, positions, hidden, mrope)?;
    (0..T).map(|r| head.logits_row(kern, embed, r).map(|v| v.to_vec())).collect()
}

#[test]
fn equal_axes_are_bit_identical_to_no_mrope() -> Result<()> {
    let _gpu = gpu_lock();
    let Ok(dev) = Device::new(0) else {
        eprintln!("no CUDA device; skipping");
        return Ok(());
    };
    let kern = Kernels::new(dev.clone());
    kern.warm_up()?;
    let dm = dims();
    let d = dm.d_model;
    let (w, embed) = synth(&dev, dm)?;
    let mut head = MtpHead::new(&dev, w, dm, T, 64, 1)?;

    let ids: Vec<u32> = (0..T as u32).map(|i| (i * 7 + 3) % dm.vocab as u32).collect();
    let positions: Vec<usize> = (0..T).collect();
    let hidden_host: Vec<f32> = (0..T * d).map(|i| ((i * 37 % 101) as f32 - 50.0) / 97.0).collect();
    let hidden = dev.stream().clone_htod(&hidden_host)?;

    let scalar = step_logits(&mut head, &kern, &embed, &ids, &positions, &hidden.as_view(), None)?;
    let triples: Vec<i32> = positions.iter().flat_map(|&p| [p as i32; 3]).collect();
    let equal_axes =
        step_logits(&mut head, &kern, &embed, &ids, &positions, &hidden.as_view(), Some(&triples))?;

    for row in 0..T {
        assert_eq!(
            scalar[row], equal_axes[row],
            "row {row}: T=H=W did not reduce to the scalar-position case"
        );
    }
    Ok(())
}

#[test]
fn different_axes_change_the_output() -> Result<()> {
    let _gpu = gpu_lock();
    let Ok(dev) = Device::new(0) else {
        eprintln!("no CUDA device; skipping");
        return Ok(());
    };
    let kern = Kernels::new(dev.clone());
    kern.warm_up()?;
    let dm = dims();
    let d = dm.d_model;
    let (w, embed) = synth(&dev, dm)?;
    let mut head = MtpHead::new(&dev, w, dm, T, 64, 1)?;

    let ids: Vec<u32> = (0..T as u32).map(|i| (i * 7 + 3) % dm.vocab as u32).collect();
    let positions: Vec<usize> = (0..T).collect();
    let hidden_host: Vec<f32> = (0..T * d).map(|i| ((i * 37 % 101) as f32 - 50.0) / 97.0).collect();
    let hidden = dev.stream().clone_htod(&hidden_host)?;

    let scalar = step_logits(&mut head, &kern, &embed, &ids, &positions, &hidden.as_view(), None)?;
    // T = position, H = position + 500_000, W = position + 1_000_000: three
    // genuinely different values a row, in disjoint ranges so none can
    // coincide with the scalar case by chance. Large offsets because the
    // affected frequencies (`rotary_dim: 16`, `theta: 10_000`) have small
    // `inv_freq` at the higher indices `section`'s H/W channels sit at, so a
    // small position delta barely moves their angle -- a real, measured
    // property of this tiny synthetic head, not a threshold picked to make a
    // test pass: offsets of 5_000/10_000 gave only 4.2e-4 of signal, most of
    // it likely from the two lowest-index affected channels alone.
    let triples: Vec<i32> = positions
        .iter()
        .flat_map(|&p| [p as i32, p as i32 + 500_000, p as i32 + 1_000_000])
        .collect();
    let diverged =
        step_logits(&mut head, &kern, &embed, &ids, &positions, &hidden.as_view(), Some(&triples))?;

    let mut max_abs = 0.0f32;
    for row in 0..T {
        for (a, b) in scalar[row].iter().zip(&diverged[row]) {
            max_abs = max_abs.max((a - b).abs());
        }
    }
    // `equal_axes_are_bit_identical_to_no_mrope` establishes the baseline is
    // now genuinely bit-exact (0 divergence, not a noise floor), so any
    // comfortably-above-zero threshold here is real signal. Measured 1.42e-3.
    assert!(
        max_abs > 5e-4,
        "feeding genuinely different T/H/W changed the head's logits by only \
         {max_abs:.3e}; mrope_axis or the mrope buffer is not reaching the \
         rope kernel"
    );
    eprintln!("max abs diff from the scalar case: {max_abs:.6}");
    Ok(())
}
