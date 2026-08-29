//! Do two draft branches at the same position keep their own keys?
//!
//! This is the whole mechanism a tree draft rests on. Siblings sit at the *same*
//! position with *different* tokens, so if they shared a cache slot the second
//! would overwrite the first and both branches would continue from whichever key
//! landed last — a plausible draft, a wrong one, and nothing in the output would
//! say so.
//!
//! Self-contained on purpose. The head's other device tests are gated on a
//! capture from the reference implementation or on a GGUF to supply hidden
//! states, and a mechanism this easy to get silently wrong should not be tested
//! only when a fixture happens to be present.

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::Kernels;
use infero_model::mtp::{HeadDims, MtpHead};
use infero_model::weights::{AttnWeights, DenseFfn, Layer, Matrix, MtpWeights};

/// Small but not degenerate: more than one kv head so the head expansion is
/// exercised, and a rotary dimension short of `d_head` the way the 27B's is.
fn dims() -> HeadDims {
    HeadDims {
        d_model: 64,
        heads: 4,
        kv_heads: 2,
        d_head: 16,
        rotary_dim: 8,
        d_ff: 128,
        eps: 1e-6,
        rope_theta: 10_000.0,
        vocab: 96,
    }
}

/// Deterministic pseudo-random weights. Values around one rather than around
/// zero, so a branch that read the wrong key produces a visibly different answer
/// instead of noise around the same one.
fn synth(dev: &Device, dims: HeadDims) -> Result<(MtpWeights, Matrix)> {
    let seed = std::cell::Cell::new(0x1234_5678u32);
    let mut next = move || {
        seed.set(seed.get().wrapping_mul(1_664_525).wrapping_add(1_013_904_223));
        (seed.get() >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    };
    let (d, da, dkv) = (dims.d_model, dims.d_attn(), dims.d_kv());
    let mut m = |k: usize, n: usize| -> Result<Matrix> {
        let v: Vec<half::f16> = (0..k * n).map(|_| half::f16::from_f32(next() * 0.3)).collect();
        Matrix::upload_f16(dev, &v, k, n)
    };
    let mut vec1 = |n: usize| -> Result<infero_model::weights::Vector> {
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

/// An equivalence, not an inspection: after forking, branch `b`'s continuation
/// must equal what a head that only ever saw branch `b`'s token produces.
///
/// And the two branches must differ from each other, or the test would pass on a
/// head that ignored the fork entirely and gave both the same answer.
#[test]
fn two_branches_at_one_position_keep_their_own_keys() -> Result<()> {
    let Ok(dev) = Device::new(0) else {
        eprintln!("no CUDA device; skipping");
        return Ok(());
    };
    let kern = Kernels::new(dev.clone());
    kern.warm_up()?;
    let dm = dims();
    let d = dm.d_model;
    // A prefix long enough that the branches' keys are a small part of what the
    // attention reads, so an overwrite is a subtle wrong answer rather than a
    // loud one.
    const PREFIX: usize = 6;
    let ids: Vec<u32> = (0..PREFIX as u32).map(|i| (i * 7 + 3) % dm.vocab as u32).collect();
    let positions: Vec<usize> = (0..PREFIX).collect();
    let hidden_host: Vec<f32> = (0..PREFIX * d)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 97.0)
        .collect();
    let hidden = dev.stream().clone_htod(&hidden_host)?;
    let (tok_a, tok_b) = (11u32, 47u32);

    let logits_for = |branches: usize,
                      run: &dyn Fn(&mut MtpHead, &Matrix) -> Result<Vec<usize>>|
     -> Result<Vec<Vec<f32>>> {
        let (w, embed) = synth(&dev, dm)?;
        // Room for the prefix plus one forked slot a branch.
        let mut head = MtpHead::new(&dev, w, dm, PREFIX.max(branches), PREFIX + branches, branches)?;
        head.step(&kern, &embed, &ids, &positions, &hidden.as_view())?;
        let rows = run(&mut head, &embed)?;
        rows.into_iter()
            .map(|r| head.logits_row(&kern, &embed, r).map(|v| v.to_vec()))
            .collect()
    };

    // Both branches in one forked level.
    let got = logits_for(2, &|h, embed| {
        h.fork(PREFIX, 1)?;
        h.step_tree(
            &kern,
            embed,
            &[tok_a, tok_b],
            &[PREFIX, PREFIX],
            &[PREFIX - 1, PREFIX - 1],
            &[0, 1],
            1,
        )?;
        Ok(vec![0, 1])
    })?;

    // Each branch alone, on the path the linear draft already takes.
    let want_a = logits_for(1, &|h, embed| {
        h.step_from_own_output(&kern, embed, tok_a, PREFIX, PREFIX - 1)?;
        Ok(vec![0])
    })?;
    let want_b = logits_for(1, &|h, embed| {
        h.step_from_own_output(&kern, embed, tok_b, PREFIX, PREFIX - 1)?;
        Ok(vec![0])
    })?;

    let rel = |a: &[f32], b: &[f32]| -> f32 {
        let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f32 = b.iter().map(|y| y * y).sum::<f32>().max(1e-12);
        (num / den).sqrt()
    };

    for (i, (g, w)) in got.iter().zip([&want_a[0], &want_b[0]]).enumerate() {
        let err = rel(g, w);
        assert!(
            err < 1e-5,
            "branch {i} diverged from the single-branch head by {err:.2e}; the \
             fork is not isolating its keys"
        );
    }
    let between = rel(&got[0], &got[1]);
    assert!(
        between > 1e-3,
        "the two branches agree to {between:.2e}, so this test cannot tell a fork \
         from a shared slot"
    );
    Ok(())
}
