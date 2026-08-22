//! The MTP head on the device, against the reference implementation's own
//! output on the real 27B weights.
//!
//! `tests/qwen35_mtp.rs` pins the host reference stage by stage against a capture
//! of vLLM's `Qwen3_5MultiTokenPredictor`. That leaves one gap, and it is the gap
//! that matters for shipping: the host reference is not what runs. So this file
//! takes the *same* capture — the head's fifteen weight tensors, the 32 real
//! tokens' embeddings, and the text model's final hidden state, all of it from
//! `/home/jeff/models/qwen38-27b-fp8` — uploads the weights as the loader would,
//! runs [`tuili_model::mtp::MtpHead`] over the real kernels, and compares against
//! the reference's own `output`.
//!
//! Every number here is therefore checked against the real 27B weights. What is
//! *not* is the loader's FP8 path, which the checkpoint's own tensors would
//! exercise but the capture cannot carry (it holds dequantized f32); that has a
//! synthetic test of its own at the bottom, over tensors small enough to state
//! exactly.
//!
//! Regenerate the capture:
//!
//!   python3 tools/capture_qwen35_mtp.py <model-dir> <out> --tokens 32 \
//!       --dump-layer-weights
//!   TUILI_QWEN35_MTP_CAPTURE=<out> cargo test -p tuili-model \
//!       --test qwen35_mtp_device
//!
//! The two draft tests additionally want the text model's `lm_head` and
//! `embed_tokens` as raw f16, beside the capture as `lm_head.f16` and
//! `embed_tokens.f16`. They are 2.5 GB each and the capture does not include
//! them; without them those tests report a skip, because a draft *token* is an
//! argmax over 248320 logits and there is no honest way to fake one.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use half::f16;
use tuili_cuda::Device;
use tuili_kernels::Kernels;
use tuili_model::mtp::{HeadDims, MtpHead};
use tuili_model::weights::{AttnWeights, Layer, Matrix, MtpWeights};

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

struct Capture {
    dir: PathBuf,
    cfg: HashMap<String, f64>,
    arrays: HashMap<String, (Vec<usize>, Vec<f32>)>,
    behaviour: HashMap<String, f64>,
    token_ids: Vec<u32>,
    shifted_ids: Vec<u32>,
}

impl Capture {
    fn open() -> Option<Self> {
        let dir = PathBuf::from(std::env::var("TUILI_QWEN35_MTP_CAPTURE").ok()?);
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(
            manifest["verified_against_reference"], true,
            "this capture was written without the cross-checks against vLLM and \
             transformers"
        );
        assert_eq!(
            manifest["prefix_truncated"], false,
            "the head consumes the text model's *final* hidden state, so a \
             truncated prefix is an oracle for nothing"
        );
        let num = |k: &str| -> HashMap<String, f64> {
            manifest[k]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
                .collect()
        };
        let ids = |k: &str| -> Vec<u32> {
            manifest[k]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect()
        };
        let mut arrays = HashMap::new();
        for (name, shape) in manifest["arrays"].as_object().unwrap() {
            let shape: Vec<usize> = shape
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let bytes = std::fs::read(dir.join(format!("{name}.f32"))).unwrap();
            let vals: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(vals.len(), shape.iter().product::<usize>(), "{name}");
            arrays.insert(name.clone(), (shape, vals));
        }
        Some(Self {
            dir,
            cfg: num("config"),
            behaviour: num("behaviour"),
            token_ids: ids("token_ids"),
            shifted_ids: ids("shifted_token_ids"),
            arrays,
        })
    }

    fn get(&self, name: &str) -> &[f32] {
        &self
            .arrays
            .get(name)
            .unwrap_or_else(|| panic!("the capture has no array {name}"))
            .1
    }

    fn shape(&self, name: &str) -> &[usize] {
        &self.arrays[name].0
    }

    fn u(&self, k: &str) -> usize {
        self.cfg[k] as usize
    }

    fn f(&self, k: &str) -> f32 {
        self.cfg[k] as f32
    }

    fn dims(&self) -> HeadDims {
        let d_head = self.u("head_dim");
        HeadDims {
            d_model: self.u("hidden_size"),
            heads: self.u("num_attention_heads"),
            kv_heads: self.u("num_key_value_heads"),
            d_head,
            rotary_dim: (d_head as f32 * self.f("partial_rotary_factor")) as usize,
            d_ff: self.u("intermediate_size"),
            eps: self.f("rms_norm_eps"),
            rope_theta: self.f("rope_theta"),
            vocab: self.u("vocab_size"),
        }
    }

    /// A raw f16 matrix sitting beside the capture, or `None`.
    fn raw_f16(&self, name: &str, k: usize, n: usize) -> Option<Vec<f16>> {
        let path = self.dir.join(name);
        let bytes = std::fs::read(&path).ok()?;
        assert_eq!(
            bytes.len(),
            k * n * 2,
            "{}: {} bytes for a [{n}, {k}] f16 matrix",
            path.display(),
            bytes.len()
        );
        Some(
            bytes
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]))
                .collect(),
        )
    }
}

fn with_capture(what: &str, body: impl FnOnce(&Capture) -> Result<()>) -> Result<()> {
    match Capture::open() {
        Some(c) => {
            if !c.arrays.contains_key("w.q_proj") {
                eprintln!(
                    "SKIPPED {what}: this capture has no layer weights; \
                     regenerate it with --dump-layer-weights"
                );
                return Ok(());
            }
            body(&c)
        }
        None => {
            eprintln!(
                "SKIPPED {what}: set TUILI_QWEN35_MTP_CAPTURE to a directory \
                 written by tools/capture_qwen35_mtp.py"
            );
            Ok(())
        }
    }
}

fn to_f16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|x| f16::from_f32(*x)).collect()
}

/// Which reading of the head to build, so that a wrong one can be shown to be
/// worse rather than assumed to be.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Reading {
    /// The reference's: embedding in the low half of `fc`, `(1 + w)` norms.
    Reference,
    /// `fc`'s two column halves swapped, which is the same shape and the same
    /// cost and a different model.
    ConcatSwapped,
    /// Norm weights taken as gains rather than as deltas from one — the mistake
    /// the loader's whitelist exists to prevent, and the one that four commits
    /// ago was live in this repository.
    PlainNorms,
}

/// Upload the head's weights out of the capture, the way the loader would.
///
/// The norms get their `+1` here because that is where
/// `weights::norm_offset` puts it: the kernels see a plain gain and the
/// convention lives at load time. `Reading::PlainNorms` skips it.
fn head_from_capture_width(
    dev: &Device,
    c: &Capture,
    reading: Reading,
    width: usize,
) -> Result<MtpHead> {
    let dims = c.dims();
    let d = dims.d_model;
    let offset = if reading == Reading::PlainNorms { 0.0 } else { 1.0 };
    let norm = |name: &str| -> Result<tuili_model::weights::Vector> {
        let v: Vec<f32> = c.get(name).iter().map(|x| x + offset).collect();
        Ok(dev.stream().clone_htod(&v)?)
    };
    let proj = |name: &str| -> Result<Matrix> {
        let shape = c.shape(name);
        let (n, k) = (shape[0], shape[1]);
        Matrix::upload_f16(dev, &to_f16(c.get(name)), k, n)
    };
    // `fc` is `[d_model, 2 * d_model]`: the low half of every row multiplies the
    // embedding. Swapping the halves is the one decision with no runtime symptom
    // at all, so it is built here as a real alternative.
    let fc_src = c.get("w.fc");
    let fc: Vec<f16> = match reading {
        Reading::ConcatSwapped => {
            let mut v = Vec::with_capacity(fc_src.len());
            for row in fc_src.chunks_exact(2 * d) {
                v.extend(to_f16(&row[d..]));
                v.extend(to_f16(&row[..d]));
            }
            v
        }
        _ => to_f16(fc_src),
    };

    let w = MtpWeights {
        fc: Matrix::upload_f16(dev, &fc, 2 * d, d)?,
        pre_fc_norm_embedding: norm("w.pre_fc_norm_embedding")?,
        pre_fc_norm_hidden: norm("w.pre_fc_norm_hidden")?,
        norm: norm("w.norm")?,
        layer: Layer {
            attn_norm: norm("w.input_layernorm")?,
            attn: Some(AttnWeights {
                wq: proj("w.q_proj")?,
                wk: proj("w.k_proj")?,
                wv: proj("w.v_proj")?,
                wo: proj("w.o_proj")?,
                bq: None,
                bk: None,
                bv: None,
                bo: None,
                q_norm: Some(norm("w.q_norm")?),
                k_norm: Some(norm("w.k_norm")?),
                w_qkv: None,
                output_gate: true,
            }),
            gdn: None,
            ffn_norm: norm("w.post_attention_layernorm")?,
            w_gate: proj("w.gate_proj")?,
            w_up: proj("w.up_proj")?,
            w_down: proj("w.down_proj")?,
            w_gate_up: None,
            blob: None,
        },
        device_bytes: 0,
    };
    MtpHead::new(dev, w, dims, width, 128)
}

/// The head sized for the capture's whole token run in one step.
fn head_from_capture(dev: &Device, c: &Capture, reading: Reading) -> Result<MtpHead> {
    head_from_capture_width(dev, c, reading, c.shape("output")[0])
}

/// A stand-in embedding matrix whose row `i` is the capture's `inputs_embeds[i]`.
///
/// The head gathers `emb[shifted_ids[i]]`, and the capture holds those rows
/// already — it does not hold the 2.5 GB table they came out of. Feeding ids
/// `0..t` against this matrix gathers exactly the same numbers, through exactly
/// the same kernel, and keeps the test to the head.
fn stub_embedding(dev: &Device, c: &Capture) -> Result<(Matrix, Vec<u32>)> {
    let d = c.u("hidden_size");
    let t = c.shape("inputs_embeds")[0];
    let m = Matrix::upload_f16(dev, &to_f16(c.get("inputs_embeds")), d, t)?;
    Ok((m, (0..t as u32).collect()))
}

fn relative_l2(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&g, &w) in got.iter().zip(want) {
        num += ((g - w) as f64).powi(2);
        den += (w as f64).powi(2);
    }
    (num / den.max(f64::MIN_POSITIVE)).sqrt() as f32
}

/// The head, on the device, over the real 27B weights, against the reference's
/// own output.
///
/// The tolerance is 1% relative L2 rather than the 1e-6 the host reference gets,
/// and the reason is the whole point of running it here: the engine holds these
/// matrices as f16 and accumulates in f32, where the reference is bf16 in and
/// f32 throughout. What that buys is a check on the plumbing — nine correct
/// stages wired together in the wrong order would still be nine correct stages —
/// so the number to watch is not how small the error is but how much smaller it
/// is than the alternatives, which is why both alternatives are run.
#[test]
fn the_device_head_reproduces_the_reference_output_on_the_real_weights() -> Result<()> {
    let _gpu = gpu_lock();
    with_capture("device head vs reference", |c| {
        let dev = Device::new(0)?;
        let kern = Kernels::new(dev.clone());
        kern.warm_up()?;
        let d = c.u("hidden_size");
        let t = c.shape("output")[0];
        let hidden = dev.stream().clone_htod(c.get("target.final_hidden"))?;
        let positions: Vec<usize> = (0..t).collect();

        let mut errors = Vec::new();
        for reading in [
            Reading::Reference,
            Reading::ConcatSwapped,
            Reading::PlainNorms,
        ] {
            let mut head = head_from_capture(&dev, c, reading)?;
            let (embed, ids) = stub_embedding(&dev, c)?;
            head.step(&kern, &embed, &ids, &positions, &hidden.as_view())?;
            let out = dev.stream().clone_dtoh(&head.output())?;
            dev.synchronize()?;
            assert_eq!(out.len(), t * d);
            let err = relative_l2(&out, c.get("output"));
            eprintln!("{reading:?}: relative L2 against the reference {err:.3e}");
            errors.push((reading, err));
            // Freed before the next reading is uploaded: three copies of the
            // head is 2.5 GB and this test has to run on a 16 GB card.
            drop(head);
        }

        let reference = errors[0].1;
        assert!(
            reference < 1e-2,
            "the device head is {reference:.3e} relative L2 away from the \
             reference implementation on the same weights and the same inputs; \
             f16 weights should cost about 1e-3"
        );
        for (reading, err) in &errors[1..] {
            assert!(
                *err > 20.0 * reference,
                "{reading:?} lands within {err:.3e} against the reference \
                 reading's {reference:.3e}, so this test does not distinguish \
                 them and proves only that the arithmetic runs"
            );
        }
        Ok(())
    })
}

/// The head's own KV cache reaches one position behind the target's, and a draft
/// step past the first one advances it by one.
///
/// The position convention cannot be pinned by a single forward's numbers —
/// shifting every position by a constant leaves a self-consistent attention
/// output unchanged, which the capture measures at 4.5e-07 relative — so it is
/// pinned structurally: what the drafter's cache length is after a step, and that
/// a second step lands in the slot after the first.
#[test]
fn the_drafters_cache_sits_one_position_behind_the_targets() -> Result<()> {
    let _gpu = gpu_lock();
    with_capture("drafter cache positions", |c| {
        let dev = Device::new(0)?;
        let kern = Kernels::new(dev.clone());
        kern.warm_up()?;
        let t = c.shape("output")[0];
        let mut head = head_from_capture(&dev, c, Reading::Reference)?;
        let (embed, ids) = stub_embedding(&dev, c)?;
        let hidden = dev.stream().clone_htod(c.get("target.final_hidden"))?;

        assert_eq!(head.cache_len(), 0, "a fresh drafter has no history");
        let positions: Vec<usize> = (0..t).collect();
        head.step(&kern, &embed, &ids, &positions, &hidden.as_view())?;
        // Slot `p` holds the pair `(h_p, emb(t_{p+1}))`, so a pass over the
        // target's positions `0..t` fills exactly `t` slots — one *behind* the
        // target, which has by then seen `t + 1` tokens counting the one it just
        // sampled.
        assert_eq!(head.cache_len(), t, "the drafter's cache should hold {t}");
        assert_eq!(head.rows(), t);

        // A second draft step: one row, at the next position, fed the head's own
        // output rather than the target's hidden state.
        head.step_from_own_output(&kern, &embed, ids[0], t, t - 1)?;
        assert_eq!(head.cache_len(), t + 1);
        assert_eq!(head.rows(), 1);

        // And rolling back is expressible: the drafter's own coordinate system.
        head.truncate(t);
        assert_eq!(head.cache_len(), t);
        Ok(())
    })
}

/// The tokens the device head drafts are the tokens the reference drafts, and
/// how often those are the target model's own next token.
///
/// This is the acceptance rate for a one-token draft, measured rather than
/// derived: `draft[i]` is compared against `target.argmax[i + 1]`, because slot
/// `i` carries `h_i` and `emb(t_{i+1})` and predicts `t_{i+2}`, which is what the
/// target's row `i + 1` predicts.
#[test]
fn the_device_head_drafts_the_reference_tokens_and_agrees_with_the_target() -> Result<()> {
    let _gpu = gpu_lock();
    with_capture("draft tokens", |c| {
        let d = c.u("hidden_size");
        let vocab = c.u("vocab_size");
        let Some(lm) = c.raw_f16("lm_head.f16", d, vocab) else {
            eprintln!(
                "SKIPPED draft tokens: no lm_head.f16 beside the capture. A \
                 draft token is an argmax over {vocab} logits and there is no \
                 way to check one without the vocabulary projection."
            );
            return Ok(());
        };
        let dev = Device::new(0)?;
        let kern = Kernels::new(dev.clone());
        kern.warm_up()?;
        let t = c.shape("output")[0];
        let lm = Matrix::upload_f16(&dev, &lm, d, vocab)?;
        let mut head = head_from_capture(&dev, c, Reading::Reference)?;
        let (embed, ids) = stub_embedding(&dev, c)?;
        let hidden = dev.stream().clone_htod(c.get("target.final_hidden"))?;
        let positions: Vec<usize> = (0..t).collect();
        head.step(&kern, &embed, &ids, &positions, &hidden.as_view())?;

        let want: Vec<u32> = c.get("draft_argmax").iter().map(|v| *v as u32).collect();
        let target: Vec<u32> = c.get("target.argmax").iter().map(|v| *v as u32).collect();
        let mut drafted = Vec::with_capacity(t);
        for row in 0..t {
            drafted.push(head.draft_row(&kern, &lm, row)?);
        }
        let same = drafted.iter().zip(&want).filter(|(a, b)| a == b).count();
        eprintln!(
            "the device head drafts {same} of {t} tokens identically to the \
             reference"
        );
        assert!(
            same * 10 >= t * 9,
            "the device head and the reference agree on only {same} of {t} draft \
             tokens; f16 weights move a logit by about 1e-3 and should only \
             change a near-tie"
        );

        // The acceptance rate for a one-token draft: how often the head's
        // proposal is the token the target model would itself have chosen.
        let hit = (0..t - 1)
            .filter(|&i| drafted[i] == target[i + 1])
            .count();
        let rate = hit as f64 / (t - 1) as f64;
        eprintln!(
            "k = 1 on real text: {hit} of {} positions accepted, rate {:.1}%, \
             mean acceptance length {:.2} (the capture's own figure for the \
             reference: {:.1}%)",
            t - 1,
            rate * 100.0,
            1.0 + rate,
            c.behaviour["top1_agreement"] * 100.0
        );
        assert!(
            (rate - c.behaviour["top1_agreement"]).abs() < 0.1,
            "the device head accepts at {:.1}% where the reference accepts at \
             {:.1}%; that is too far apart to be f16 rounding",
            rate * 100.0,
            c.behaviour["top1_agreement"] * 100.0
        );
        // Compared against the wrong slot alignment, which is the other reading
        // of "the head predicts the next token": `drafted[i]` against
        // `target.argmax[i]` would be the head predicting `t_{i+1}` rather than
        // `t_{i+2}`.
        let off_by_one = (0..t).filter(|&i| drafted[i] == target[i]).count();
        assert!(
            off_by_one * 2 < hit,
            "the head's drafts agree with the target's prediction at the *same* \
             slot {off_by_one} times against {hit} at the next one, so this \
             measurement does not establish which one the head predicts"
        );
        Ok(())
    })
}

/// Mean acceptance length at `k = 2`, on real text, with the real weights.
///
/// The second draft step is where a one-layer head has to feed itself: its own
/// `mtp.norm` output becomes the hidden state, the token it just drafted becomes
/// the embedded one, and the position advances by one. There is no second layer
/// to make that plausible, so it is worth measuring rather than assuming.
///
/// What can honestly be measured from a teacher-forced capture is stated in the
/// output: the first draft is checked against the target's own argmax, and the
/// second only at the positions where the accepted first draft leaves the
/// sequence equal to the captured one — which is where the capture's next row is
/// the right conditional. Positions where the real text diverges from the model's
/// greedy choice are excluded from the second term and counted, because
/// pretending the capture answers a question it was not asked is how a number
/// becomes wrong.
#[test]
fn the_head_reaches_a_useful_acceptance_length_at_k_2() -> Result<()> {
    let _gpu = gpu_lock();
    with_capture("k = 2 acceptance", |c| {
        let d = c.u("hidden_size");
        let vocab = c.u("vocab_size");
        let (Some(lm), Some(emb)) = (
            c.raw_f16("lm_head.f16", d, vocab),
            c.raw_f16("embed_tokens.f16", d, vocab),
        ) else {
            eprintln!(
                "SKIPPED k = 2 acceptance: needs lm_head.f16 and \
                 embed_tokens.f16 beside the capture. The second draft step \
                 embeds a token the head chose, which no stub embedding can \
                 supply."
            );
            return Ok(());
        };
        let dev = Device::new(0)?;
        let kern = Kernels::new(dev.clone());
        kern.warm_up()?;
        let t = c.shape("output")[0];
        let lm = Matrix::upload_f16(&dev, &lm, d, vocab)?;
        let embed = Matrix::upload_f16(&dev, &emb, d, vocab)?;
        let mut head = head_from_capture(&dev, c, Reading::Reference)?;
        let hidden = dev.stream().clone_htod(c.get("target.final_hidden"))?;
        let target: Vec<u32> = c.get("target.argmax").iter().map(|v| *v as u32).collect();
        let tokens = &c.token_ids;
        // The shift the head is fed: `shifted_ids[i]` is `t_{i+1}`.
        assert_eq!(
            &c.shifted_ids[..t - 1],
            &tokens[1..t],
            "the capture's shifted ids are not the token sequence shifted by one"
        );

        // First step over the whole prefix, from the real embedding table this
        // time, so that the drafter's cache is the one a real step would leave.
        let positions: Vec<usize> = (0..t).collect();
        head.step(
            &kern,
            &embed,
            &c.shifted_ids[..t],
            &positions,
            &hidden.as_view(),
        )?;
        let first: Vec<u32> = (0..t)
            .map(|row| head.draft_row(&kern, &lm, row))
            .collect::<Result<_>>()?;

        // A second draft step from each position, on top of the head's own
        // output. The drafter's cache is rolled back to the first step's extent
        // between probes, which is the same rollback a rejected draft does.
        let mut second = vec![0u32; t];
        for row in 0..t {
            // The first step is re-run before each probe rather than its state
            // being kept: a draft step overwrites the head's activations, and a
            // probe that read `out[row]` after another probe had run would be
            // feeding back the wrong row. Re-running is a few milliseconds and
            // leaves no room for that mistake.
            head.truncate(0);
            head.step(
                &kern,
                &embed,
                &c.shifted_ids[..t],
                &positions,
                &hidden.as_view(),
            )?;
            head.step_from_own_output(&kern, &embed, first[row], row + 1, row)?;
            second[row] = head.draft_row(&kern, &lm, 0)?;
        }

        let mut lengths = Vec::new();
        let mut second_measurable = 0usize;
        let mut second_hit = 0usize;
        let mut diverged = 0usize;
        for i in 0..t - 2 {
            // Draft one is accepted when it is the target's own next token.
            let one = first[i] == target[i + 1];
            if !one {
                lengths.push(1usize);
                continue;
            }
            // With draft one accepted, the sequence is `t_0..t_{i+1}` followed by
            // `target.argmax[i+1]`. The capture's row `i + 2` is the target's
            // prediction after `t_0..t_{i+2}`, so it answers the right question
            // only where the text and the model's greedy choice coincide.
            if target[i + 1] != tokens[i + 2] {
                diverged += 1;
                lengths.push(2);
                continue;
            }
            second_measurable += 1;
            let two = second[i] == target[i + 2];
            if two {
                second_hit += 1;
            }
            lengths.push(if two { 3 } else { 2 });
        }
        let mean = lengths.iter().sum::<usize>() as f64 / lengths.len() as f64;
        let first_rate = (0..t - 1).filter(|&i| first[i] == target[i + 1]).count() as f64
            / (t - 1) as f64;
        // The `diverged` positions are counted as length 2 — the second draft
        // assumed rejected — because the capture cannot say. So this mean is a
        // lower bound, and the bound is loose by at most `diverged` positions'
        // worth of the measured second-step rate.
        let second_rate = second_hit as f64 / second_measurable.max(1) as f64;
        let upper = mean + diverged as f64 * second_rate / lengths.len() as f64;
        eprintln!(
            "k = 2 on the 27B's real weights and 32 real tokens: mean acceptance \
             length {mean:.2} over {} positions, and at most {upper:.2}.\n  first \
             draft accepted {:.1}% of the time; second draft accepted \
             {second_hit} of the {second_measurable} positions where a \
             teacher-forced capture can answer the question ({:.0}%), with \
             {diverged} more counted as rejected because the captured text \
             diverges there from the model's own greedy continuation and the \
             capture holds no logits for the branch the drafter would have \
             taken.\n  independence from the first-step rate would predict \
             {:.2}; vLLM reports 2.0 on this checkpoint; the notes' acceptance \
             line is 1.9.",
            lengths.len(),
            first_rate * 100.0,
            second_rate * 100.0,
            1.0 + first_rate + first_rate * first_rate,
        );
        assert!(
            mean > 1.6,
            "the head's mean acceptance length at k = 2 is {mean:.2}. The notes \
             expect at least 1.9 and say that below 1.6 the thing to suspect is \
             the composition order — the concat, the two pre-fc norms, the norm \
             form — and not the scheduler"
        );
        Ok(())
    })
}

// ------------------------------------------------------- the loader, in the small

/// Write a one-file safetensors checkpoint.
fn write_safetensors(path: &std::path::Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        // Every payload starts on an eight-byte boundary, and the header below is
        // padded to one too. The reader borrows the mapping and checks alignment
        // rather than copying, so an f32 tensor at an odd offset is rejected —
        // which is what real exporters pad for.
        while !payload.len().is_multiple_of(8) {
            payload.push(0);
        }
        let start = payload.len();
        payload.extend_from_slice(bytes);
        header.insert(
            (*name).to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, payload.len()],
            }),
        );
    }
    let mut head = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    while !head.len().is_multiple_of(8) {
        head.push(b' ');
    }
    let mut out = (head.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&head);
    out.extend_from_slice(&payload);
    std::fs::write(path, out).unwrap();
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
        .collect()
}

/// The FP8 code whose value is closest to `x`, found by asking the decoder.
///
/// Encoding E4M3 by hand is a bit-twiddling exercise with an easy off-by-one in
/// the subnormal range; asking the same table the loader will read cannot
/// disagree with it, which is the property a fixture wants.
fn e4m3_table() -> Vec<f32> {
    let dir = std::env::temp_dir().join("tuili_e4m3_probe");
    std::fs::create_dir_all(&dir).unwrap();
    let codes: Vec<u8> = (0..=255u8).collect();
    write_safetensors(
        &dir.join("m.safetensors"),
        &[
            ("t.weight", "F8_E4M3", vec![256, 1], codes),
            (
                "t.weight_scale_inv",
                "F32",
                vec![2, 1],
                [1.0f32, 1.0].iter().flat_map(|x| x.to_le_bytes()).collect(),
            ),
        ],
    );
    let shards = tuili_safetensors::Shards::open_dir(&dir).unwrap();
    let t = shards.tensor("t.weight").unwrap();
    let s = shards.tensor("t.weight_scale_inv").unwrap();
    t.dequant_f8_to_f16(&s, 128)
        .unwrap()
        .iter()
        .map(|h| f32::from(*h))
        .collect()
}

/// The loader reads the head's tensors, adds one to exactly the norms that want
/// it, and takes `mtp.fc` at its own dtype.
///
/// Small enough to state every number. The projections are FP8 with unit block
/// scales and values that E4M3 represents exactly, so the only thing the
/// comparison can be measuring is whether the loader read the right tensor into
/// the right slot; `mtp.fc` is BF16 in the same file, which is the checkpoint's
/// one non-uniform tensor and the one vLLM special-cases.
///
/// The discriminating half is the second load: the same tensors under a config
/// whose `model_type` is not `qwen3_5`, where the offset must *not* be applied.
/// Without that this test would pass whether the whitelist fired or not.
#[test]
fn the_loader_adds_one_to_exactly_the_norms_that_are_stored_as_deltas() -> Result<()> {
    let _gpu = gpu_lock();
    let dev = Device::new(0)?;
    let table = e4m3_table();
    let encode = |x: f32| -> u8 {
        let mut best = 0u8;
        let mut err = f32::INFINITY;
        for (code, v) in table.iter().enumerate() {
            if !v.is_finite() {
                continue;
            }
            let e = (v - x).abs();
            if e < err {
                err = e;
                best = code as u8;
            }
        }
        assert!(err < 1e-6, "{x} is not exactly representable in E4M3");
        best
    };

    // Shapes chosen so that every one is different and none is a multiple of
    // another: 2 heads of 4 against a 6-wide residual, 3 kv... one kv head.
    let (d, heads, kv_heads, d_head, d_ff) = (6usize, 2usize, 1usize, 4usize, 5usize);
    let d_attn = heads * d_head;
    let vals: &[f32] = &[1.0, -0.5, 2.0, 0.25, -1.5, 0.5, -2.0, 1.5];
    let pattern = |n: usize, seed: usize| -> Vec<f32> {
        (0..n).map(|i| vals[(i * 7 + seed) % vals.len()]).collect()
    };
    let fp8 = |out: usize, inn: usize, seed: usize| -> Vec<u8> {
        pattern(out * inn, seed).iter().map(|x| encode(*x)).collect()
    };
    let scale_grid = |out: usize, inn: usize| -> Vec<u8> {
        let n = out.div_ceil(128) * inn.div_ceil(128);
        (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect()
    };

    let dir = std::env::temp_dir().join("tuili_mtp_loader_fixture");
    std::fs::create_dir_all(&dir).unwrap();
    let mut tensors: Vec<(&str, &str, Vec<usize>, Vec<u8>)> = Vec::new();
    // The three head-specific norms and the layer's two, plus the per-head ones.
    let norm_names = [
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
    ];
    for (i, name) in norm_names.iter().enumerate() {
        tensors.push((name, "BF16", vec![d], bf16_bytes(&pattern(d, i))));
    }
    for (i, name) in ["mtp.layers.0.self_attn.q_norm.weight", "mtp.layers.0.self_attn.k_norm.weight"]
        .iter()
        .enumerate()
    {
        tensors.push((name, "BF16", vec![d_head], bf16_bytes(&pattern(d_head, i + 5))));
    }
    // `mtp.fc` is BF16 where everything else is FP8.
    tensors.push((
        "mtp.fc.weight",
        "BF16",
        vec![d, 2 * d],
        bf16_bytes(&pattern(d * 2 * d, 3)),
    ));
    let projections: &[(&str, usize, usize)] = &[
        ("mtp.layers.0.self_attn.q_proj.weight", 2 * d_attn, d),
        ("mtp.layers.0.self_attn.k_proj.weight", kv_heads * d_head, d),
        ("mtp.layers.0.self_attn.v_proj.weight", kv_heads * d_head, d),
        ("mtp.layers.0.self_attn.o_proj.weight", d, d_attn),
        ("mtp.layers.0.mlp.gate_proj.weight", d_ff, d),
        ("mtp.layers.0.mlp.up_proj.weight", d_ff, d),
        ("mtp.layers.0.mlp.down_proj.weight", d, d_ff),
    ];
    for (name, out, inn) in projections {
        tensors.push((name, "F8_E4M3", vec![*out, *inn], fp8(*out, *inn, 2)));
    }
    // The block scales, named the way an FP8 checkpoint names them. One entry
    // per 128x128 tile, so every one of these fixtures has exactly one.
    let scale_names: Vec<String> = projections
        .iter()
        .map(|(p, _, _)| p.replace(".weight", ".weight_scale_inv"))
        .collect();
    for ((_, out, inn), name) in projections.iter().zip(&scale_names) {
        tensors.push((
            name.as_str(),
            "F32",
            vec![out.div_ceil(128), inn.div_ceil(128)],
            scale_grid(*out, *inn),
        ));
    }
    write_safetensors(&dir.join("model.safetensors"), &tensors);

    let config = |model_type: &str| -> serde_json::Value {
        serde_json::json!({
            "model_type": model_type,
            "tie_word_embeddings": false,
            "text_config": {
                "num_hidden_layers": 4,
                "hidden_size": d,
                "num_attention_heads": heads,
                "num_key_value_heads": kv_heads,
                "head_dim": d_head,
                "intermediate_size": d_ff,
                "vocab_size": 32,
                "rms_norm_eps": 1e-6,
                "attn_output_gate": true,
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
                "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.5},
            },
        })
    };
    let shards = tuili_safetensors::Shards::open_dir(&dir).unwrap();

    let read = |v: &tuili_model::weights::Vector| -> Vec<f32> {
        let out = dev.stream().clone_dtoh(v).unwrap();
        dev.synchronize().unwrap();
        out
    };
    let mut loaded = Vec::new();
    for arch in ["qwen3_5", "llama"] {
        let cfg = tuili_model::Config::from_hf(&config(arch), "fixture")?;
        let head = tuili_model::weights::load_mtp(&dev, &shards, &cfg)?
            .expect("the fixture carries mtp.fc.weight");
        loaded.push((
            arch,
            read(&head.pre_fc_norm_embedding),
            read(&head.norm),
            read(head.layer.attn().q_norm.as_ref().unwrap()),
            head,
        ));
    }

    // Under `qwen3_5` every norm is a delta from one and the loader adds it.
    let raw_embedding = pattern(d, 0);
    let raw_q_norm = pattern(d_head, 5);
    for (i, x) in loaded[0].1.iter().enumerate() {
        assert_eq!(
            *x,
            raw_embedding[i] + 1.0,
            "pre_fc_norm_embedding[{i}] should be the stored delta plus one"
        );
    }
    for (i, x) in loaded[0].3.iter().enumerate() {
        assert_eq!(*x, raw_q_norm[i] + 1.0, "q_norm[{i}]");
    }
    // And under an architecture that does not use the offset form it does not,
    // which is what shows the whitelist fired rather than the buffer happening
    // to hold the right numbers.
    assert_eq!(loaded[1].1, raw_embedding, "pre_fc_norm_embedding, llama");
    assert_eq!(loaded[1].3, raw_q_norm, "q_norm, llama");
    assert_ne!(loaded[0].1, loaded[1].1, "the two readings coincide");

    // The projections: FP8 in, f16 out, at the shapes the config implies.
    let head = &loaded[0].4;
    assert_eq!((head.fc.k, head.fc.n), (2 * d, d), "fc is [d, 2d]");
    let l = head.layer.attn();
    assert_eq!((l.wq.k, l.wq.n), (d, 2 * d_attn), "q_proj carries a gate");
    assert!(l.output_gate, "the gate was not detected from the shape");
    assert_eq!((l.wk.k, l.wk.n), (d, kv_heads * d_head));
    assert_eq!((l.wo.k, l.wo.n), (d_attn, d), "o_proj narrows d_attn to d");
    assert_eq!(
        (head.layer.w_down.k, head.layer.w_down.n),
        (d_ff, d),
        "down_proj"
    );
    assert!(head.layer.gdn.is_none(), "the head's layer is full attention");

    // And the values survived the FP8 round trip: the fixture's numbers are
    // exact in E4M3 and the scales are one, so anything but equality means a
    // transposed read.
    let view = l.wk.view(None)?;
    let bytes = dev.stream().clone_dtoh(&view)?;
    dev.synchronize()?;
    let got: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|c| f32::from(f16::from_le_bytes([c[0], c[1]])))
        .collect();
    let want = pattern(kv_heads * d_head * d, 2);
    assert_eq!(got, want, "k_proj came back different from what was written");
    Ok(())
}

/// Priming over a feed wider than the head, in chunks, is the same model.
///
/// This is the test that was missing when speculation first ran on the server.
/// The head is built for one draft step's width, but the first round of a request
/// primes it over the whole prompt — `DraftFeed::after_prefill` insists on that,
/// because the drafter attends over its own cache and a cache with holes is a
/// different model. Nothing checked the two claims against each other, so a
/// 62-token prompt walked 62 rows into a 3-row buffer and cudarc's slice returned
/// `None`. Every existing test here happens to pass a feed that fits.
///
/// So: 32 real tokens through a head built for 32, and the same 32 through a head
/// built for 7. The second takes five chunks. What must hold is not merely that
/// it runs — it is that the rows agree, which is the substance of "a cache with
/// holes is a different model" stated as a check. It also pins the return value,
/// because `rows` describes the *last* chunk and a caller still using the feed's
/// width would read a row that chunk never wrote.
///
/// The control matters as much as the tolerance: comparing against neighbouring
/// rows has to be far worse, or the test would pass on any two runs that happened
/// to produce smooth output.
#[test]
fn chunked_priming_agrees_with_one_wide_step() -> Result<()> {
    let _gpu = gpu_lock();
    with_capture("chunked priming", |c| {
        let dev = Device::new(0)?;
        let kern = Kernels::new(dev.clone());
        kern.warm_up()?;
        let d = c.u("hidden_size");
        let t = c.shape("output")[0];
        const WIDTH: usize = 7;
        assert!(t > WIDTH, "the capture's {t} tokens have to exceed {WIDTH}");
        let hidden = dev.stream().clone_htod(c.get("target.final_hidden"))?;
        let positions: Vec<usize> = (0..t).collect();

        // One step, the way the passing tests do it.
        let whole = {
            let mut head = head_from_capture(&dev, c, Reading::Reference)?;
            let (embed, ids) = stub_embedding(&dev, c)?;
            head.step(&kern, &embed, &ids, &positions, &hidden.as_view())?;
            let out = dev.stream().clone_dtoh(&head.output())?;
            dev.synchronize()?;
            assert_eq!(out.len(), t * d);
            out
        };

        // The same feed, five chunks, into a head that cannot hold it whole.
        let mut head = head_from_capture_width(&dev, c, Reading::Reference, WIDTH)?;
        let (embed, ids) = stub_embedding(&dev, c)?;
        let last = head.prime(&kern, &embed, &ids, &positions, &hidden.as_view())?;
        let chunked = dev.stream().clone_dtoh(&head.output())?;
        dev.synchronize()?;

        // The final chunk holds `t % WIDTH` rows — 4 of the 32 — and `prime`
        // reports the last token's index *within it*.
        let tail = if t % WIDTH == 0 { WIDTH } else { t % WIDTH };
        assert_eq!(chunked.len(), tail * d, "the last chunk's rows");
        assert_eq!(last, tail - 1, "the row `prime` points at");
        assert_eq!(head.cached(), t, "the drafter's cache after priming");

        let start = t - tail;
        let aligned = relative_l2(&chunked, &whole[start * d..]);
        // Shifted by one row: the same numbers, read at the wrong offset.
        let shifted = relative_l2(&chunked, &whole[(start - 1) * d..(t - 1) * d]);
        eprintln!(
            "chunked vs one step: aligned {aligned:.3e}, off by one row {shifted:.3e}"
        );
        assert!(
            aligned < 5e-3,
            "chunked priming diverged from the one-step pass, relative L2 {aligned:.3e}"
        );
        assert!(
            shifted > aligned * 20.0,
            "the off-by-one control is only {shifted:.3e} against {aligned:.3e}, so \
             this comparison would pass on misaligned rows too"
        );
        Ok(())
    })
}
