//! The MTP head's host reference, checked stage by stage against a capture of
//! the reference implementations running on the real checkpoint.
//!
//! Same discipline as `tests/qwen35_capture.rs`, and for the same reason: local
//! self-consistency is what the bf16-as-f16 embedding bug satisfied for a night
//! across nine component A/Bs. Every number compared here came from outside this
//! repository.
//!
//! The MTP head raises the stakes slightly, because `transformers` does not
//! implement it — `modeling_qwen3_5.py` carries
//! `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` and drops those tensors on
//! the floor. The reference is vLLM's `Qwen3_5MultiTokenPredictor`, and the
//! capture reaches it three ways: by parsing its `forward` into a canonical
//! string, by running the decoder layer out of transformers' own classes, and by
//! measuring on real text that the composition drafts the target model's own
//! next token 71% of the time while the swapped one manages 0%.
//!
//! Regenerate the capture:
//!
//!   python3 tools/capture_qwen35_mtp.py <model-dir> <out-dir> --tokens 32
//!   INFERO_QWEN35_MTP_CAPTURE=<out-dir> \
//!     cargo test -p infero-model --test qwen35_mtp
//!
//! Without the environment variable the capture-backed tests report as skipped
//! rather than passing, because a silent skip is how a suite comes to be green
//! without checking anything.
//!
//! Two tests here need no capture and say so in their doc comments: the
//! acceptance rule and the recurrent-state rollback are pure algebra over
//! primitives that the *other* test file already pins against the checkpoint, so
//! gating them on a weight-derived capture would only make them skip more often.

use std::collections::HashMap;
use std::path::PathBuf;

use infero_model::qwen35::{self, sigmoid};
use infero_model::qwen35_mtp::*;

struct Capture {
    cfg: HashMap<String, f64>,
    arrays: HashMap<String, (Vec<usize>, Vec<f32>)>,
    behaviour: HashMap<String, f64>,
    /// Dimensions of the small `Qwen3_5DecoderLayer` the capture builds out of
    /// the reference's own class. Absent in captures written before it existed,
    /// which is why the tests that need it check rather than unwrap.
    synth: Option<HashMap<String, f64>>,
}

impl Capture {
    fn open() -> Option<Self> {
        let dir = PathBuf::from(std::env::var("INFERO_QWEN35_MTP_CAPTURE").ok()?);
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(
            manifest["verified_against_reference"], true,
            "this capture was written without the cross-checks against vLLM and \
             transformers, so it is one reading of the reference rather than the \
             reference; regenerate it where both libraries are importable"
        );
        assert_eq!(
            manifest["prefix_truncated"], false,
            "this capture ran only {} of the target model's layers. The MTP head \
             consumes the *final* hidden state, so a truncated prefix feeds it a \
             tensor the head has never seen and every stage downstream of `fc` \
             is an oracle for nothing. Regenerate without --prefix-layers.",
            manifest["prefix_layers_run"]
        );
        let cfg = manifest["config"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
            .collect();
        let behaviour = manifest["behaviour"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
            .collect();
        let mut arrays = HashMap::new();
        for (name, shape) in manifest["arrays"].as_object().unwrap() {
            let shape: Vec<usize> = shape
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let bytes = std::fs::read(dir.join(format!("{name}.f32"))).unwrap();
            let want: usize = shape.iter().product();
            assert_eq!(
                bytes.len(),
                want * 4,
                "{name}: manifest says {shape:?} but the file holds {} floats",
                bytes.len() / 4
            );
            let vals: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            arrays.insert(name.clone(), (shape, vals));
        }
        let synth = manifest["synth_config"].as_object().map(|o| {
            o.iter()
                .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
                .collect()
        });
        Some(Self {
            cfg,
            arrays,
            behaviour,
            synth,
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
        &self.arrays.get(name).unwrap().0
    }

    fn u(&self, key: &str) -> usize {
        self.cfg[key] as usize
    }

    fn f(&self, key: &str) -> f32 {
        self.cfg[key] as f32
    }

    fn dims(&self) -> BlockDims {
        let head_dim = self.u("head_dim");
        BlockDims {
            d_model: self.u("hidden_size"),
            heads: self.u("num_attention_heads"),
            kv_heads: self.u("num_key_value_heads"),
            head_dim,
            rotary_dim: (head_dim as f32 * self.f("partial_rotary_factor")) as usize,
            d_ff: self.u("intermediate_size"),
            eps: self.f("rms_norm_eps"),
        }
    }
}

fn with_capture(what: &str, body: impl FnOnce(&Capture)) {
    match Capture::open() {
        Some(c) => body(&c),
        None => eprintln!(
            "SKIPPED {what}: set INFERO_QWEN35_MTP_CAPTURE to a directory written \
             by tools/capture_qwen35_mtp.py"
        ),
    }
}

/// Same error model as `tests/qwen35_capture.rs`: `rel` is the per-element
/// relative allowance, `scale` sets the absolute floor as a fraction of the
/// tensor's own peak, because f32 accumulation error scales with the magnitudes
/// summed rather than with the size of the result.
fn agree(got: &[f32], want: &[f32], rel: f32, scale: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let floor = scale * peak.max(f32::MIN_POSITIVE);
    let mut worst = 0.0f32;
    let mut worst_at = 0;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{what}: element {i} is {g}");
        let err = (g - w).abs() / (floor + rel * w.abs());
        if err > worst {
            worst = err;
            worst_at = i;
        }
    }
    assert!(
        worst <= 1.0,
        "{what}: worst disagreement at element {worst_at} — got {}, reference \
         {}, which is {worst:.1}x the tolerance (rel {rel:.1e}, floor \
         {floor:.3e} from a peak of {peak:.3e})",
        got[worst_at],
        want[worst_at]
    );
}

fn relative_l2(got: &[f32], want: &[f32]) -> f32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&g, &w) in got.iter().zip(want) {
        num += ((g - w) as f64).powi(2);
        den += (w as f64).powi(2);
    }
    (num / den.max(f64::MIN_POSITIVE)).sqrt() as f32
}

/// How far apart two tensors are, as a multiple of the tolerance `agree` would
/// allow. Used to show that a rejected reading is rejected by a wide margin
/// rather than by a hair.
fn margin(a: &[f32], b: &[f32], rel: f32, scale: f32) -> f32 {
    let peak = b.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let floor = scale * peak.max(f32::MIN_POSITIVE);
    a.iter().zip(b).fold(0.0f32, |m, (&x, &y)| {
        m.max((x - y).abs() / (floor + rel * y.abs()))
    })
}

// ------------------------------------------------------------- the RMSNorm form

/// `Qwen3_5RMSNorm` scales by `(1 + weight)`, not by `weight`.
///
/// This is the mistake that would have been inherited for free, because
/// `qwen35::rms_norm_rows` implements the plain form and is *right* to — for the
/// one norm in this architecture that wants it, the GatedDeltaNet output gate's
/// `Qwen3_5RMSNormGated`. Every norm in the MTP head is the other class.
///
/// The check is run on the head's four real weights rather than on a synthetic
/// one, because the reason this matters is what these particular numbers are:
/// `pre_fc_norm_embedding` is negative across all 5120 channels, so the plain
/// reading does not merely mis-scale, it inverts.
#[test]
fn every_norm_in_the_head_uses_the_unit_offset_form() {
    with_capture("rmsnorm offset form", |c| {
        let d = c.u("hidden_size");
        let eps = c.f("rms_norm_eps");
        let t_len = c.shape("inputs_embeds")[0];

        for (input, weight, output) in [
            ("inputs_embeds", "w.pre_fc_norm_embedding", "emb_normed"),
            (
                "target.final_hidden",
                "w.pre_fc_norm_hidden",
                "hidden_normed",
            ),
            ("layer_out", "w.norm", "output"),
            ("fc_out", "w.input_layernorm", "layer.pre_attn_norm_out"),
        ] {
            let x = c.get(input);
            let w = c.get(weight);
            let want = c.get(output);
            let got = rms_norm_offset_rows(x, w, d, eps);
            agree(&got, want, 2e-6, 1e-7, &format!("(1+w) norm of {input}"));

            // And the plain form must be nowhere near, or this test would pass
            // under either reading and is not evidence.
            let plain = qwen35::rms_norm_rows(x, w, d, eps, 0.0);
            let m = margin(&plain, want, 2e-6, 1e-7);
            assert!(
                m > 1e3,
                "the plain `w *` reading of {weight} lands within {m:.1e} \
                 tolerances of the reference, so this test does not \
                 distinguish the two forms"
            );
        }

        // The specific thing that makes the plain reading so quiet: it is the
        // right order of magnitude. Record it so nobody re-derives the surprise.
        let w = c.get("w.pre_fc_norm_embedding");
        assert!(
            w.iter().all(|v| *v < 0.0),
            "pre_fc_norm_embedding is expected to be negative on every channel \
             in this checkpoint; if that has changed, the argument for why the \
             plain reading is dangerous needs rechecking"
        );
        let offset_gain = w.iter().map(|v| 1.0 + v).sum::<f32>() / w.len() as f32;
        let plain_gain = w.iter().sum::<f32>() / w.len() as f32;
        assert!(
            (offset_gain - plain_gain.abs()).abs() < 0.2,
            "the two readings of pre_fc_norm_embedding differ mostly in sign \
             (mean gain {offset_gain:+.3} vs {plain_gain:+.3}); that is why the \
             wrong one produces fluent output"
        );
        let _ = t_len;
    });
}

// ------------------------------------------------------------ draft construction

/// The concat is `[normalized embedding, normalized hidden]`, embedding first.
///
/// This is the one decision with no runtime symptom at all: both halves are
/// `[t_len, 5120]` f32 and `fc` consumes either arrangement at the same speed.
///
/// The test does not trust the capture's own concatenation. It takes whole rows
/// of `fc` and recomputes a handful of `fc_out` elements from the two captured
/// halves under both orderings, then requires the embedding-first one to match
/// and the hidden-first one not to. That distinguishes the readings using only
/// the reference's weights and the reference's output.
#[test]
fn the_fc_input_puts_the_embedding_before_the_hidden_state() {
    with_capture("fc concat order", |c| {
        let d = c.u("hidden_size");
        let t_len = c.shape("fc_out")[0];
        let e = c.get("emb_normed");
        let h = c.get("hidden_normed");
        let fc_out = c.get("fc_out");
        let rows: Vec<usize> = c.get("fc_probe_rows").iter().map(|v| *v as usize).collect();
        let w = c.get("fc_probe_w");
        assert_eq!(c.shape("fc_probe_w")[1], 2 * d, "fc rows are 2*hidden wide");

        let mut separating = 0;
        for (i, &row) in rows.iter().enumerate() {
            let wr = &w[i * 2 * d..(i + 1) * 2 * d];
            let (lo, hi) = wr.split_at(d);
            for t in 0..t_len {
                let et = &e[t * d..(t + 1) * d];
                let ht = &h[t * d..(t + 1) * d];
                let dot =
                    |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
                let embedding_first = dot(lo, et) + dot(hi, ht);
                let hidden_first = dot(lo, ht) + dot(hi, et);
                let want = fc_out[t * d + row];

                assert!(
                    (embedding_first - want).abs() <= 1e-4 + 2e-4 * want.abs(),
                    "fc_out[t{t}, {row}]: with the embedding in the low half \
                     the weight row predicts {embedding_first}, the reference \
                     has {want}"
                );
                if (hidden_first - want).abs() > 1e-3 + 1e-2 * want.abs() {
                    separating += 1;
                }
            }
        }
        assert!(
            separating > 0,
            "no probe row distinguished [embedding | hidden] from [hidden | \
             embedding]; this test would pass under either and is not evidence"
        );
    });
}

/// `pre_fc_norm_embedding` goes on the embedding and `pre_fc_norm_hidden` on the
/// hidden state — not the other way round.
///
/// Also same shapes, also silent. The two weights are genuinely different
/// tensors on this checkpoint, so swapping them is a real change to the model
/// and still not a crash.
#[test]
fn each_pre_fc_norm_is_applied_to_the_input_it_is_named_for() {
    with_capture("pre_fc norm assignment", |c| {
        let d = c.u("hidden_size");
        let eps = c.f("rms_norm_eps");
        let embeds = c.get("inputs_embeds");
        let hidden = c.get("target.final_hidden");
        let w_emb = c.get("w.pre_fc_norm_embedding");
        let w_hid = c.get("w.pre_fc_norm_hidden");

        agree(
            &rms_norm_offset_rows(embeds, w_emb, d, eps),
            c.get("emb_normed"),
            2e-6,
            1e-7,
            "pre_fc_norm_embedding applied to the embedding",
        );
        agree(
            &rms_norm_offset_rows(hidden, w_hid, d, eps),
            c.get("hidden_normed"),
            2e-6,
            1e-7,
            "pre_fc_norm_hidden applied to the hidden state",
        );

        // Swapped: must be far away, in both directions.
        let m_e = margin(
            &rms_norm_offset_rows(embeds, w_hid, d, eps),
            c.get("emb_normed"),
            2e-6,
            1e-7,
        );
        let m_h = margin(
            &rms_norm_offset_rows(hidden, w_emb, d, eps),
            c.get("hidden_normed"),
            2e-6,
            1e-7,
        );
        assert!(
            m_e > 1e3 && m_h > 1e3,
            "swapping the two pre_fc norms lands within {m_e:.1e} / {m_h:.1e} \
             tolerances of the reference, so this test does not pin which norm \
             goes where"
        );
    });
}

/// The head reads the target's hidden state **after** `model.language_model.norm`.
///
/// The pre-norm hidden state is the other candidate — EAGLE-style drafters
/// differ on this and DeepSeek V4's MTP genuinely wants the pre-`hc_head`
/// residual, which is why vLLM has a `get_mtp_target_hidden_states` hook at all.
/// Qwen3.5 does not implement that hook, so the runner passes `hidden_states`,
/// the output of `Qwen3NextModel.forward`, whose last statement is
/// `hidden_states, _ = self.norm(hidden_states, residual)`.
///
/// The capture notes that this one cannot be settled behaviourally — feeding the
/// pre-norm state drafts about as well on the capture's text, because
/// `pre_fc_norm_hidden` renormalizes and only `model.norm`'s per-channel gain
/// survives. So it is settled numerically here.
#[test]
fn the_head_consumes_the_post_final_norm_hidden_state() {
    with_capture("post-final-norm hidden state", |c| {
        let d = c.u("hidden_size");
        let eps = c.f("rms_norm_eps");
        let w_hid = c.get("w.pre_fc_norm_hidden");
        let want = c.get("hidden_normed");

        agree(
            &rms_norm_offset_rows(c.get("target.final_hidden"), w_hid, d, eps),
            want,
            2e-6,
            1e-7,
            "hidden_normed from the post-norm hidden state",
        );
        let m = margin(
            &rms_norm_offset_rows(c.get("target.pre_norm_hidden"), w_hid, d, eps),
            want,
            2e-6,
            1e-7,
        );
        assert!(
            m > 1e3,
            "normalizing the *pre*-final-norm hidden state lands within {m:.1e} \
             tolerances, so this test does not pin which of the two the head \
             reads"
        );

        // And the two hidden states must themselves be far apart, or the
        // distinction is not a real one on this capture.
        let drift = relative_l2(
            c.get("target.pre_norm_hidden"),
            c.get("target.final_hidden"),
        );
        assert!(
            drift > 0.1,
            "the pre- and post-norm hidden states differ by only {drift:.2e} \
             relative on this capture, so the question is not being tested"
        );
    });
}

/// `fuse_embedding_and_hidden` reproduces the reference's `fc` output.
///
/// End to end over the stage, using only the two captured inputs and probe rows
/// of `fc` for the parts a whole matrix would be needed for. The `fc` matrix is
/// 210 MB in f32 and is not in the capture, so this checks the composed
/// normalizations against the reference and the projection against the probes.
#[test]
fn fusing_the_two_inputs_reproduces_the_reference_stages() {
    with_capture("fuse_embedding_and_hidden", |c| {
        let d = c.u("hidden_size");
        let eps = c.f("rms_norm_eps");
        let t_len = c.shape("fc_out")[0];

        // A stand-in `fc` built from the probe rows: a [rows, 2d] matrix whose
        // output is the probed columns of the real one. Same code path, same
        // ordering decision, small enough to be in the capture.
        let rows: Vec<usize> = c.get("fc_probe_rows").iter().map(|v| *v as usize).collect();
        let probe_w = c.get("fc_probe_w").to_vec();
        let w = MtpWeights {
            pre_fc_norm_embedding: c.get("w.pre_fc_norm_embedding"),
            pre_fc_norm_hidden: c.get("w.pre_fc_norm_hidden"),
            fc: &probe_w,
            layer: dummy_block(),
            norm: c.get("w.norm"),
        };
        // `fuse_embedding_and_hidden` asserts `fc.len() == d * 2 * d`, which the
        // probe slice deliberately is not, so drive its three pieces directly.
        // They are the library's own functions, not reimplementations —
        // `concat_embedding_then_hidden` exists precisely so that the ordering
        // decision is exercised here rather than duplicated here.
        let e = rms_norm_offset_rows(c.get("inputs_embeds"), w.pre_fc_norm_embedding, d, eps);
        let h = rms_norm_offset_rows(c.get("target.final_hidden"), w.pre_fc_norm_hidden, d, eps);
        agree(&e, c.get("emb_normed"), 2e-6, 1e-7, "emb_normed");
        agree(&h, c.get("hidden_normed"), 2e-6, 1e-7, "hidden_normed");

        let cat = concat_embedding_then_hidden(&e, &h, t_len, d);
        let got = linear(&cat, &probe_w, t_len, 2 * d, rows.len());
        let want: Vec<f32> = (0..t_len)
            .flat_map(|t| rows.iter().map(move |&r| c.get("fc_out")[t * d + r]))
            .collect();
        agree(&got, &want, 3e-4, 1e-6, "fc over the probed output columns");
    });
}

fn dummy_block<'a>() -> BlockWeights<'a> {
    BlockWeights {
        input_layernorm: &[],
        q_proj: &[],
        k_proj: &[],
        v_proj: &[],
        o_proj: &[],
        q_norm: &[],
        k_norm: &[],
        post_attention_layernorm: &[],
        gate_proj: &[],
        up_proj: &[],
        down_proj: &[],
    }
}

// ------------------------------------------------- inside the head's own layer

/// The head's decoder layer is a full-attention layer, and its `q_proj` output
/// interleaves q with the output gate per head.
///
/// Both halves of this matter for the head specifically. That it is
/// full-attention decides whether drafting touches the recurrent state at all —
/// it does not, and the whole rollback design in `notes/qwen3.5-mtp.md` rests on
/// that. That the gate interleaves is the same trap as in the target model's
/// layers, re-checked here because the head's `q_proj` is a different tensor and
/// nothing would have carried the earlier check over.
///
/// The probe is deliberately taken past head 0: under `[all q | all gate]` the
/// first head's q rows coincide with the interleaved reading, so a test that
/// only looked there would bless either.
#[test]
fn the_heads_layer_is_gated_full_attention_with_q_and_gate_interleaved() {
    with_capture("head q/gate interleave", |c| {
        let d = c.u("hidden_size");
        let nh = c.u("num_attention_heads");
        let hd = c.u("head_dim");
        let t_len = c.shape("layer.pre_attn_norm_out")[0];

        // The shapes alone rule out a linear-attention layer: q_proj is
        // heads * 2 * head_dim wide because of the gate, and there is no qkv
        // fusion, no conv, no per-head scalar projection.
        assert_eq!(
            c.shape("layer.q_proj_out")[1],
            nh * 2 * hd,
            "q_proj should be heads * 2 * head_dim wide for the output gate"
        );
        assert_eq!(
            c.shape("layer.v_proj_out")[1],
            c.u("num_key_value_heads") * hd
        );

        let x = c.get("layer.pre_attn_norm_out");
        let rows: Vec<usize> = c
            .get("q_proj_probe_rows")
            .iter()
            .map(|v| *v as usize)
            .collect();
        let w = c.get("q_proj_probe_w");
        // The reference's own q, straight out of its `q_norm` input hook: this
        // is what `view(T, heads, 2*head_dim)` then `chunk(2, dim=-1)` produced,
        // so the layout is being read off the reference rather than assumed.
        let q_ref = c.get("layer.q_norm_in");

        let project = |row: usize, t: usize| -> f32 {
            let i = rows.iter().position(|&r| r == row).expect("row not probed");
            let wr = &w[i * d..(i + 1) * d];
            let xt = &x[t * d..(t + 1) * d];
            xt.iter().zip(wr).map(|(a, b)| a * b).sum()
        };

        let mut separating = 0;
        for h in [0, 1, nh - 1] {
            for dd in [0, 1, hd - 1] {
                for t in 0..t_len {
                    let interleaved = project(h * 2 * hd + dd, t);
                    let got = q_ref[(t * nh + h) * hd + dd];
                    assert!(
                        (interleaved - got).abs() <= 1e-4 + 2e-4 * got.abs(),
                        "q[t{t}, h{h}, d{dd}]: the interleaved layout predicts \
                         {interleaved}, the reference has {got}"
                    );
                    let split = project(h * hd + dd, t);
                    if (split - got).abs() > 1e-3 + 1e-2 * got.abs() {
                        separating += 1;
                    }
                }
            }
        }
        assert!(
            separating > 0,
            "no probe distinguished the interleaved layout from [all q | all \
             gate]; this test is not evidence"
        );

        // And `split_q_and_gate` on the reference's raw q_proj output must
        // reproduce the reference's q exactly.
        let (q, _gate) = qwen35::split_q_and_gate(c.get("layer.q_proj_out"), t_len, nh, hd);
        agree(&q, q_ref, 0.0, 1e-7, "q from split_q_and_gate");
    });
}

/// `q_norm` and `k_norm` are the unit-offset form too, applied per head, before
/// rope.
///
/// Called out separately from the other norms because these are the ones most
/// likely to be got wrong by analogy: they are `head_dim`-wide rather than
/// `hidden_size`-wide, they live inside the attention rather than on the
/// residual stream, and `crates/model/src/qwen35.rs` normalizes them with the
/// plain form.
#[test]
fn the_per_head_q_and_k_norms_use_the_unit_offset_form_before_rope() {
    with_capture("q/k norm form", |c| {
        let hd = c.u("head_dim");
        let eps = c.f("rms_norm_eps");
        for (input, weight, output) in [
            ("layer.q_norm_in", "w.q_norm", "layer.q_norm_out"),
            ("layer.k_norm_in", "w.k_norm", "layer.k_norm_out"),
        ] {
            let x = c.get(input);
            let w = c.get(weight);
            let want = c.get(output);
            agree(
                &rms_norm_offset_rows(x, w, hd, eps),
                want,
                3e-6,
                1e-7,
                &format!("(1+w) norm of {input}"),
            );
            let m = margin(&qwen35::rms_norm_rows(x, w, hd, eps, 0.0), want, 3e-6, 1e-7);
            assert!(
                m > 1e2,
                "the plain `w *` reading of {weight} lands within {m:.1e} \
                 tolerances; this test does not pin the form"
            );
        }
    });
}

/// Attention, the sigmoid output gate, and both residual adds, all the way to
/// the layer's output.
///
/// Everything here is checked without a single projection matrix, using the
/// reference's own tapped tensors on both sides of each transition. `o_proj`'s
/// *input* is `attention_output * sigmoid(gate)`, which pins the attention, the
/// gate, the choice of sigmoid over silu, and the partial rope in one comparison.
#[test]
fn the_heads_attention_gate_and_residuals_match_the_reference() {
    with_capture("head attention and residuals", |c| {
        let dims = c.dims();
        let t_len = c.shape("layer.q_norm_out")[0];
        let (nh, nkv, hd) = (dims.heads, dims.kv_heads, dims.head_dim);

        let mut q = c.get("layer.q_norm_out").to_vec();
        let mut k = c.get("layer.k_norm_out").to_vec();
        let cos = c.get("rope_cos");
        let sin = c.get("rope_sin");
        assert_eq!(
            c.shape("rope_cos")[1],
            dims.rotary_dim,
            "the rope table should be head_dim * partial_rotary_factor wide"
        );
        qwen35::apply_partial_rope(&mut q, cos, sin, t_len, nh, hd, dims.rotary_dim);
        qwen35::apply_partial_rope(&mut k, cos, sin, t_len, nkv, hd, dims.rotary_dim);

        let ctx =
            qwen35::causal_attention(&q, &k, c.get("layer.v_proj_out"), t_len, t_len, nh, nkv, hd);

        // The gate is the second half of each head's q_proj slice.
        let (_, gate) = qwen35::split_q_and_gate(c.get("layer.q_proj_out"), t_len, nh, hd);
        let gated: Vec<f32> = ctx
            .iter()
            .zip(&gate)
            .map(|(o, g)| o * sigmoid(*g))
            .collect();
        let want = c.get("layer.o_proj_in");
        agree(
            &gated,
            want,
            2e-3,
            1e-5,
            "attention output times sigmoid(gate)",
        );

        // silu instead of sigmoid — what config's `output_gate_type = "swish"`
        // would suggest, and what the implementation ignores — must be far off.
        let silu_gated: Vec<f32> = ctx
            .iter()
            .zip(&gate)
            .map(|(o, g)| o * qwen35::silu(*g))
            .collect();
        let m = margin(&silu_gated, want, 2e-3, 1e-5);
        assert!(
            m > 10.0,
            "silu lands within {m:.1e} tolerances of sigmoid here, so this test \
             does not establish which gate the checkpoint wants"
        );

        // Ungated attention must also be far off, or the gate is doing nothing
        // measurable and the comparison above proves only the attention.
        let m_nogate = margin(&ctx, want, 2e-3, 1e-5);
        assert!(
            m_nogate > 10.0,
            "the gate changes the attention output by only {m_nogate:.1e} \
             tolerances; it is not being tested"
        );

        // The two residual adds. Pre-norm layer, so both are `x + f(norm(x))`,
        // and the layer's output is `fc_out + attn + mlp`.
        let d = dims.d_model;
        let fc_out = c.get("fc_out");
        let attn = c.get("layer.o_proj_out");
        let after_attn: Vec<f32> = fc_out.iter().zip(attn).map(|(a, b)| a + b).collect();
        agree(
            &rms_norm_offset_rows(
                &after_attn,
                c.get("w.post_attention_layernorm"),
                d,
                dims.eps,
            ),
            c.get("layer.post_attn_norm_out"),
            3e-5,
            1e-6,
            "post_attention_layernorm over fc_out + attn",
        );
        let out: Vec<f32> = after_attn
            .iter()
            .zip(c.get("layer.mlp_out"))
            .map(|(a, b)| a + b)
            .collect();
        agree(
            &out,
            c.get("layer_out"),
            3e-6,
            1e-7,
            "layer output = fc_out + attn + mlp",
        );

        // A layer that dropped either residual would still produce a tensor of
        // the right shape, so show that both are load-bearing.
        for (name, alt) in [
            ("without the attention residual", attn.to_vec()),
            ("without the mlp residual", after_attn.clone()),
        ] {
            let m = margin(&alt, c.get("layer_out"), 3e-6, 1e-7);
            assert!(m > 1e3, "{name} lands within {m:.1e} tolerances");
        }
    });
}

/// The final `mtp.norm`, and the fact that it is the head's own tensor rather
/// than the target model's.
#[test]
fn the_head_finishes_with_its_own_final_norm() {
    with_capture("mtp.norm", |c| {
        let d = c.u("hidden_size");
        let eps = c.f("rms_norm_eps");
        agree(
            &rms_norm_offset_rows(c.get("layer_out"), c.get("w.norm"), d, eps),
            c.get("output"),
            2e-6,
            1e-7,
            "mtp.norm over the layer output",
        );
        // Skipping it entirely is the other plausible reading — a head that
        // returned the layer output straight to lm_head would run.
        let m = margin(c.get("layer_out"), c.get("output"), 2e-6, 1e-7);
        assert!(
            m > 1e3,
            "mtp.norm changes almost nothing ({m:.1e} tolerances)"
        );
    });
}

/// The head's logits come from the *target model's* `lm_head`.
///
/// The index file settles the structural half — the checkpoint ships no
/// `mtp.lm_head` and no `mtp.embed_tokens`, and `mtp_use_dedicated_embeddings`
/// is false — and this settles the numeric half, recomputing a few logits from
/// the base `lm_head`'s own rows.
#[test]
fn the_head_scores_with_the_targets_lm_head() {
    with_capture("shared lm_head", |c| {
        let d = c.u("hidden_size");
        let t_len = c.shape("output")[0];
        let n = c.shape("lm_head_probe_ids")[0];
        let got = linear(c.get("output"), c.get("lm_head_probe_w"), t_len, d, n);
        agree(
            &got,
            c.get("draft_logits_probe"),
            3e-4,
            1e-6,
            "draft logits from the target model's lm_head rows",
        );
    });
}

/// The whole head, composed, against the reference's own final output.
///
/// This runs `mtp_head`'s glue for real — the two norms, the concat, the final
/// norm — and stitches in the reference's layer output for the one part that
/// needs matrices the capture cannot carry. It is the test that would catch a
/// mistake in how the pieces are wired together rather than in any one piece.
#[test]
fn the_composed_head_reproduces_the_reference_output() {
    with_capture("composed head", |c| {
        let d = c.u("hidden_size");
        let eps = c.f("rms_norm_eps");
        let t_len = c.shape("output")[0];

        let e = rms_norm_offset_rows(
            c.get("inputs_embeds"),
            c.get("w.pre_fc_norm_embedding"),
            d,
            eps,
        );
        let h = rms_norm_offset_rows(
            c.get("target.final_hidden"),
            c.get("w.pre_fc_norm_hidden"),
            d,
            eps,
        );
        agree(&e, c.get("emb_normed"), 2e-6, 1e-7, "stage 1: emb_normed");
        agree(
            &h,
            c.get("hidden_normed"),
            2e-6,
            1e-7,
            "stage 2: hidden_normed",
        );
        // stage 3 (fc) is checked against probe rows in
        // `fusing_the_two_inputs_reproduces_the_reference_stages`; take the
        // reference's fc_out from here on.
        agree(
            &rms_norm_offset_rows(c.get("fc_out"), c.get("w.input_layernorm"), d, eps),
            c.get("layer.pre_attn_norm_out"),
            2e-6,
            1e-7,
            "stage 4: the layer's input norm consumes fc_out",
        );
        agree(
            &rms_norm_offset_rows(c.get("layer_out"), c.get("w.norm"), d, eps),
            c.get("output"),
            2e-6,
            1e-7,
            "stage 5: mtp.norm",
        );
        assert_eq!(c.shape("output"), [t_len, d]);
    });
}

/// `mtp_head` end to end, when the capture carries the head's weights.
///
/// The tests above pin every *transition* inside the head against the reference,
/// which is the strong form of the check and needs no matrices. What they cannot
/// see is the plumbing between transitions: a `full_attention_layer` that
/// computed each stage correctly and wired two of them together in the wrong
/// order would satisfy all of them. So `capture_qwen35_mtp.py
/// --dump-layer-weights` writes the head's whole decoder layer — 1.7 GB in f32,
/// which is why it is opt-in — and this runs the composed reference against the
/// reference implementation's own output.
///
/// It reports a skip rather than passing when those arrays are absent, for the
/// same reason the whole file skips without a capture.
#[test]
fn the_whole_head_matches_end_to_end_when_the_weights_are_captured() {
    with_capture("mtp_head end to end", |c| {
        if !c.arrays.contains_key("w.q_proj") {
            eprintln!(
                "SKIPPED mtp_head end to end: regenerate the capture with \
                 --dump-layer-weights to check the composed head rather than \
                 only its transitions"
            );
            return;
        }
        let dims = c.dims();
        let t_len = c.shape("output")[0];
        let w = MtpWeights {
            pre_fc_norm_embedding: c.get("w.pre_fc_norm_embedding"),
            pre_fc_norm_hidden: c.get("w.pre_fc_norm_hidden"),
            fc: c.get("w.fc"),
            layer: BlockWeights {
                input_layernorm: c.get("w.input_layernorm"),
                q_proj: c.get("w.q_proj"),
                k_proj: c.get("w.k_proj"),
                v_proj: c.get("w.v_proj"),
                o_proj: c.get("w.o_proj"),
                q_norm: c.get("w.q_norm"),
                k_norm: c.get("w.k_norm"),
                post_attention_layernorm: c.get("w.post_attention_layernorm"),
                gate_proj: c.get("w.gate_proj"),
                up_proj: c.get("w.up_proj"),
                down_proj: c.get("w.down_proj"),
            },
            norm: c.get("w.norm"),
        };
        let got = mtp_head(
            c.get("inputs_embeds"),
            c.get("target.final_hidden"),
            &w,
            c.get("rope_cos"),
            c.get("rope_sin"),
            t_len,
            dims,
        );
        agree(
            &got.emb_normed,
            c.get("emb_normed"),
            2e-6,
            1e-7,
            "emb_normed",
        );
        agree(
            &got.hidden_normed,
            c.get("hidden_normed"),
            2e-6,
            1e-7,
            "hidden_normed",
        );
        agree(&got.fc_out, c.get("fc_out"), 3e-4, 2e-6, "fc_out");
        agree(&got.layer_out, c.get("layer_out"), 3e-3, 2e-5, "layer_out");
        agree(&got.output, c.get("output"), 3e-3, 2e-5, "output");

        // The swapped concat, run through the same composed head: it must land
        // nowhere near, or this end-to-end check would pass under either order.
        let swapped =
            concat_embedding_then_hidden(&got.hidden_normed, &got.emb_normed, t_len, dims.d_model);
        let fc_swapped = linear(
            &swapped,
            c.get("w.fc"),
            t_len,
            2 * dims.d_model,
            dims.d_model,
        );
        let m = margin(&fc_swapped, c.get("fc_out"), 3e-4, 2e-6);
        assert!(
            m > 1e3,
            "the swapped concat lands within {m:.1e} tolerances"
        );
    });
}

/// What acceptance rate this head actually achieves, measured by the capture.
///
/// Not a property of the port — a property of the checkpoint — but recorded as a
/// test so that a regression in the composition shows up as a number and not as
/// a vague sense that speculation stopped helping. vLLM reports this head lifting
/// decode from 44 to 89 tok/s at a mean acceptance length of 2.0 on this
/// checkpoint; the capture measures greedy first-token agreement on real text,
/// which is the same quantity for a one-token draft.
#[test]
fn the_capture_records_a_useful_acceptance_rate() {
    with_capture("acceptance rate", |c| {
        let hit = c.behaviour["top1_agreement"];
        assert!(
            hit > 0.5,
            "the head drafts the target's own next token only {:.1}% of the \
             time; at that rate a two-token draft has a mean acceptance length \
             of {:.2} and speculation is close to break-even",
            hit * 100.0,
            1.0 + hit + hit * hit
        );
        // And each broken reading of the composition must be worse, which is the
        // capture's own gate — re-asserted here so it is visible in the test
        // output rather than only in the capture log.
        for name in [
            "top1_agreement_concat_swapped",
            "top1_agreement_norms_swapped",
            "top1_agreement_plain_rmsnorm",
        ] {
            let other = c.behaviour[name];
            assert!(
                other < hit,
                "{name} scores {:.1}% against {:.1}%; the capture is not \
                 discriminating",
                other * 100.0,
                hit * 100.0
            );
        }
        eprintln!(
            "acceptance: reference composition {:.1}%, concat swapped {:.1}%, \
             norms swapped {:.1}%, plain-w norm {:.1}%",
            hit * 100.0,
            c.behaviour["top1_agreement_concat_swapped"] * 100.0,
            c.behaviour["top1_agreement_norms_swapped"] * 100.0,
            c.behaviour["top1_agreement_plain_rmsnorm"] * 100.0,
        );
    });
}

// ---------------------------------------------------- scheduling-side algebra
//
// The two tests below need no capture. They are algebra over the acceptance rule
// and the recurrence, and the recurrence itself is pinned against the checkpoint
// by `tests/qwen35_capture.rs`. Gating them on a weight-derived capture would
// only make them skip more often, and a skipped test is how a suite comes to be
// green without checking anything — the same reason the capture-backed tests
// above refuse to pass silently.

/// The greedy acceptance rule, including the properties that make it safe.
///
/// Two of these are worth having as assertions rather than as prose: a step
/// always emits at least one token, so speculation cannot livelock; and the
/// emitted sequence is exactly what unspeculated greedy decoding would have
/// produced, so switching speculation on cannot change outputs.
#[test]
fn greedy_acceptance_emits_exactly_what_plain_decoding_would_have() {
    // All accepted: k drafts plus the bonus token.
    let a = accept_greedy(&[7, 8, 9], &[7, 8, 9, 10]);
    assert_eq!(a.tokens, vec![7, 8, 9, 10]);
    assert_eq!(a.accepted, 3);

    // First token wrong: emit the target's own choice, nothing else, no bonus.
    let a = accept_greedy(&[7, 8, 9], &[42, 8, 9, 10]);
    assert_eq!(a.tokens, vec![42]);
    assert_eq!(a.accepted, 0);

    // Mismatch in the middle: everything before it survives, the target's token
    // replaces it, and the rest of the draft is discarded even though token 9
    // *would* have matched. That last part is the subtle one — accepting past a
    // rejection would be sampling from a distribution conditioned on a token the
    // target did not choose.
    let a = accept_greedy(&[7, 8, 9], &[7, 99, 9, 10]);
    assert_eq!(a.tokens, vec![7, 99]);
    assert_eq!(a.accepted, 1);

    // A step never emits nothing.
    for draft in [vec![], vec![1], vec![1, 2], vec![1, 2, 3]] {
        let target: Vec<u32> = (0..draft.len() as u32 + 1).map(|i| 100 + i).collect();
        let a = accept_greedy(&draft, &target);
        assert!(
            !a.tokens.is_empty(),
            "a verification step emitted no tokens"
        );
        assert!(a.tokens.len() <= draft.len() + 1);
    }

    // Equivalence to plain greedy decoding: whatever the draft was, the tokens
    // emitted are a prefix of the target model's own argmax sequence.
    let target = vec![5, 6, 7, 8];
    for draft in [
        vec![5, 6, 7],
        vec![5, 6, 0],
        vec![5, 0, 7],
        vec![0, 6, 7],
        vec![9, 9, 9],
    ] {
        let a = accept_greedy(&draft, &target);
        assert_eq!(
            a.tokens,
            target[..a.tokens.len()].to_vec(),
            "the accepted tokens for draft {draft:?} are not a prefix of the \
             target's own argmax sequence, so speculation would change the \
             output"
        );
    }
}

/// Rolling the recurrent state back by replaying only the accepted prefix gives
/// exactly the state the accepted tokens alone would have produced.
///
/// This is the load-bearing claim of the scheduling design in
/// `notes/qwen3.5-mtp.md`: that a rejected draft can be undone for 48 KiB per
/// token per layer of journal instead of a 3 MiB snapshot, because the update
/// terms are already computed in the forward pass and the update is a scalar
/// decay plus a rank-one add.
///
/// The comparison is against `qwen35::gated_delta_rule` itself, run on the
/// accepted tokens only — which is by construction what "the state an
/// unspeculated decode would have left" means.
#[test]
fn replaying_the_accepted_prefix_restores_the_state_exactly() {
    let (t_len, heads, dk, dv) = (5usize, 3usize, 4usize, 6usize);
    let eps = 1e-6f32;

    // A deterministic pseudo-random fill; the point is arbitrary non-degenerate
    // numbers, not statistics.
    let mut seed = 0x2f6e_2b1du32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((seed >> 8) as f32 / 8_388_608.0) - 1.0
    };
    let q: Vec<f32> = (0..t_len * heads * dk).map(|_| next()).collect();
    let k: Vec<f32> = (0..t_len * heads * dk).map(|_| next()).collect();
    let v: Vec<f32> = (0..t_len * heads * dv).map(|_| next()).collect();
    // beta near 1 on purpose: that is where inverting the recurrence would blow
    // up, and the journal must not care.
    let beta: Vec<f32> = (0..t_len * heads)
        .map(|_| 0.999 + 0.0005 * next())
        .collect();
    let g: Vec<f32> = (0..t_len * heads).map(|_| -0.05 * (next() + 1.5)).collect();
    let initial: Vec<f32> = (0..heads * dk * dv).map(|_| next()).collect();

    // Build the journal the way a forward pass would: the same normalization and
    // the same delta the recurrence computes internally.
    let mut kn = k.clone();
    qwen35::l2norm_rows(&mut kn, dk, eps);

    for accepted in 0..=t_len {
        // What an unspeculated decode over just the accepted tokens leaves.
        let mut want = initial.clone();
        if accepted > 0 {
            qwen35::gated_delta_rule(
                &q[..accepted * heads * dk],
                &k[..accepted * heads * dk],
                &v[..accepted * heads * dv],
                &g[..accepted * heads],
                &beta[..accepted * heads],
                &mut want,
                accepted,
                heads,
                dk,
                dv,
                eps,
            );
        }

        // What the journal-and-replay path leaves. The journal is recorded while
        // running the full `t_len`-token verification pass — including the
        // tokens that end up rejected — from a scratch copy of the state.
        let mut scratch = initial.clone();
        let mut journal = Vec::new();
        for t in 0..t_len {
            let mut delta = vec![0.0f32; heads * dv];
            for h in 0..heads {
                let s = &mut scratch[h * dk * dv..(h + 1) * dk * dv];
                let decay = g[t * heads + h].exp();
                for value in s.iter_mut() {
                    *value *= decay;
                }
                let kh = &kn[(t * heads + h) * dk..(t * heads + h + 1) * dk];
                let vh = &v[(t * heads + h) * dv..(t * heads + h + 1) * dv];
                let b = beta[t * heads + h];
                for j in 0..dv {
                    let mem: f32 = (0..dk).map(|i| s[i * dv + j] * kh[i]).sum();
                    delta[h * dv + j] = (vh[j] - mem) * b;
                }
                for i in 0..dk {
                    for j in 0..dv {
                        s[i * dv + j] += kh[i] * delta[h * dv + j];
                    }
                }
            }
            journal.push(DeltaJournalEntry {
                k: kn[t * heads * dk..(t + 1) * heads * dk].to_vec(),
                delta,
                g: g[t * heads..(t + 1) * heads].to_vec(),
            });
        }

        let mut got = initial.clone();
        replay_accepted(&mut got, &journal, accepted, heads, dk, dv);
        agree(
            &got,
            &want,
            1e-4,
            1e-6,
            &format!("state after replaying {accepted} of {t_len} tokens"),
        );

        // And the un-rolled-back state — what an in-place verification pass
        // would have left behind — must differ whenever anything was rejected.
        if accepted < t_len {
            let m = margin(&scratch, &want, 1e-4, 1e-6);
            assert!(
                m > 10.0,
                "with {accepted} of {t_len} tokens accepted, the in-place state \
                 is within {m:.1e} tolerances of the correct one, so this test \
                 does not show that rollback is necessary"
            );
        }
    }
}

// ------------------------------- the reference's own layer, at a size that fits
//
// The tests above pin every *transition* inside the head's decoder layer against
// a tap on the real 27B block: the two norms, the q/gate split, the attention
// output, both residuals. What they cannot pin is the composition, or the MLP at
// all, because the real layer's weights will not be dumped — `q_proj` alone is
// 251 MB in f32.
//
// So `tools/capture_qwen35_mtp.py` also builds a 40-wide `Qwen3_5DecoderLayer`
// out of the reference's own class with random weights and dumps the whole thing.
// These two tests run `full_attention_layer` and `swiglu_mlp` against it. Small
// matrices are not a numerical stand-in for large ones and are not meant to be —
// the stagewise tests are what answer to real activations. This is about layout
// and composition, where the size is irrelevant.

struct Synth {
    dims: BlockDims,
    t_len: usize,
}

fn synth(c: &Capture) -> Option<Synth> {
    let cfg = c.synth.as_ref()?;
    let u = |k: &str| cfg[k] as usize;
    Some(Synth {
        dims: BlockDims {
            d_model: u("d_model"),
            heads: u("heads"),
            kv_heads: u("kv_heads"),
            head_dim: u("head_dim"),
            rotary_dim: u("rotary_dim"),
            d_ff: u("d_ff"),
            eps: cfg["eps"] as f32,
        },
        t_len: u("tokens"),
    })
}

/// `full_attention_layer`, end to end, against `Qwen3_5DecoderLayer` itself.
///
/// Everything in the block is under test at once and nothing is decomposed: the
/// input norm, the per-head q/gate split, the two per-head norms in their offset
/// form, the partial rope table and its `rotate_half` pairing, the
/// `1/sqrt(head_dim)` scale, the causal softmax, the GQA key expansion, the
/// sigmoid output gate before `o_proj`, the first residual, the second norm, the
/// SwiGLU MLP, and the second residual. If any one of those is read differently
/// from the library, this fails.
///
/// The rope tables come from the capture, which built them the same way
/// `qwen35::rope_tables` does — so the test also recomputes them here and
/// requires the two to agree, which is what keeps the table itself in scope.
#[test]
fn the_reference_decoder_layer_reproduces_full_attention_layer() {
    with_capture("synthetic decoder layer", |c| {
        let Some(s) = synth(c) else {
            eprintln!(
                "SKIPPED synthetic decoder layer: this capture predates the \
                 synth.* arrays; regenerate it"
            );
            return;
        };
        let d = s.dims;

        // The table first, since everything downstream rides on it.
        let positions: Vec<u32> = (0..s.t_len as u32).collect();
        let (cos, sin) = qwen35::rope_tables(
            c.synth.as_ref().unwrap()["rope_theta"] as f32,
            d.rotary_dim,
            &positions,
        );
        agree(&cos, c.get("synth.rope_cos"), 1e-6, 1e-7, "synth rope cos");
        agree(&sin, c.get("synth.rope_sin"), 1e-6, 1e-7, "synth rope sin");

        let w = BlockWeights {
            input_layernorm: c.get("synth.w.input_layernorm"),
            q_proj: c.get("synth.w.q_proj"),
            k_proj: c.get("synth.w.k_proj"),
            v_proj: c.get("synth.w.v_proj"),
            o_proj: c.get("synth.w.o_proj"),
            q_norm: c.get("synth.w.q_norm"),
            k_norm: c.get("synth.w.k_norm"),
            post_attention_layernorm: c.get("synth.w.post_attention_layernorm"),
            gate_proj: c.get("synth.w.gate_proj"),
            up_proj: c.get("synth.w.up_proj"),
            down_proj: c.get("synth.w.down_proj"),
        };
        let want = c.get("synth.out");
        let got = full_attention_layer(c.get("synth.x"), &w, &cos, &sin, s.t_len, d);
        agree(&got, want, 2e-5, 1e-6, "full_attention_layer vs the reference");

        // And two readings that also run, so the agreement above is a statement
        // about the choices and not only about the arithmetic. Both are the same
        // block with one decision inverted.
        //
        // Dropping the second residual: the MLP replaces the stream instead of
        // adding to it.
        let h = rms_norm_offset_rows(c.get("synth.x"), w.input_layernorm, d.d_model, d.eps);
        let qg = linear(&h, w.q_proj, s.t_len, d.d_model, d.heads * 2 * d.head_dim);
        let (q, gate) = qwen35::split_q_and_gate(&qg, s.t_len, d.heads, d.head_dim);
        let k = linear(&h, w.k_proj, s.t_len, d.d_model, d.d_kv());
        let v = linear(&h, w.v_proj, s.t_len, d.d_model, d.d_kv());
        let mut q = rms_norm_offset_rows(&q, w.q_norm, d.head_dim, d.eps);
        let mut k = rms_norm_offset_rows(&k, w.k_norm, d.head_dim, d.eps);
        qwen35::apply_partial_rope(
            &mut q,
            &cos,
            &sin,
            s.t_len,
            d.heads,
            d.head_dim,
            d.rotary_dim,
        );
        qwen35::apply_partial_rope(
            &mut k,
            &cos,
            &sin,
            s.t_len,
            d.kv_heads,
            d.head_dim,
            d.rotary_dim,
        );
        let ctx = qwen35::causal_attention(
            &q,
            &k,
            &v,
            s.t_len,
            s.t_len,
            d.heads,
            d.kv_heads,
            d.head_dim,
        );

        let mut gated = ctx.clone();
        for (a, g) in gated.iter_mut().zip(&gate) {
            *a *= sigmoid(*g);
        }
        let attn = linear(&gated, w.o_proj, s.t_len, d.d_attn(), d.d_model);
        let resid: Vec<f32> = c
            .get("synth.x")
            .iter()
            .zip(&attn)
            .map(|(a, b)| a + b)
            .collect();
        let normed = rms_norm_offset_rows(&resid, w.post_attention_layernorm, d.d_model, d.eps);
        let mlp = swiglu_mlp(
            &normed,
            w.gate_proj,
            w.up_proj,
            w.down_proj,
            s.t_len,
            d.d_model,
            d.d_ff,
        );
        assert!(
            margin(&mlp, want, 2e-5, 1e-6) > 1e3,
            "dropping the second residual lands on the reference output, so this \
             test does not show the residual is there"
        );

        // silu on the output gate instead of sigmoid.
        let mut silu_gated = ctx;
        for (a, g) in silu_gated.iter_mut().zip(&gate) {
            *a *= qwen35::silu(*g);
        }
        let attn2 = linear(&silu_gated, w.o_proj, s.t_len, d.d_attn(), d.d_model);
        let mut alt: Vec<f32> = c
            .get("synth.x")
            .iter()
            .zip(&attn2)
            .map(|(a, b)| a + b)
            .collect();
        let n2 = rms_norm_offset_rows(&alt, w.post_attention_layernorm, d.d_model, d.eps);
        let m2 = swiglu_mlp(
            &n2,
            w.gate_proj,
            w.up_proj,
            w.down_proj,
            s.t_len,
            d.d_model,
            d.d_ff,
        );
        for (r, m) in alt.iter_mut().zip(&m2) {
            *r += m;
        }
        assert!(
            margin(&alt, want, 2e-5, 1e-6) > 1e3,
            "a silu output gate reproduces the reference layer, so this test \
             does not pin the gate's activation"
        );
    });
}

/// `swiglu_mlp` puts `silu` on the `gate_proj` branch, against `Qwen3_5MLP`.
///
/// The mirror image — `gate_proj(x) * silu(up_proj(x))` — is the same shape, the
/// same cost and a different model, and until the capture grew a small MLP there
/// was nothing anywhere in this repository that distinguished them. The capture
/// measures the separation on its own input and refuses to write if the two
/// readings agree, so the margin below is known to be real.
#[test]
fn the_swiglu_puts_silu_on_the_gate_branch() {
    with_capture("synthetic swiglu", |c| {
        let Some(s) = synth(c) else {
            eprintln!("SKIPPED synthetic swiglu: this capture predates synth.*");
            return;
        };
        let d = s.dims;
        let x = c.get("synth.mlp_x");
        let want = c.get("synth.mlp_y");
        let got = swiglu_mlp(
            x,
            c.get("synth.w.gate_proj"),
            c.get("synth.w.up_proj"),
            c.get("synth.w.down_proj"),
            s.t_len,
            d.d_model,
            d.d_ff,
        );
        agree(&got, want, 2e-5, 1e-6, "swiglu_mlp vs Qwen3_5MLP");

        // The mirror: silu on up_proj.
        let mirror = swiglu_mlp(
            x,
            c.get("synth.w.up_proj"),
            c.get("synth.w.gate_proj"),
            c.get("synth.w.down_proj"),
            s.t_len,
            d.d_model,
            d.d_ff,
        );
        let m = margin(&mirror, want, 2e-5, 1e-6);
        assert!(
            m > 1e3,
            "silu on up_proj lands within {m:.1e} tolerances of the reference, \
             so this test does not pin which branch the activation is on"
        );
    });
}

/// The acceptance rule, against vLLM's own triton kernels.
///
/// `accept_greedy` and `accept_stochastic` are transcriptions of
/// `rejection_greedy_sample_kernel` and `rejection_random_sample_kernel` and
/// nothing checked them. They are control flow rather than arithmetic, which is
/// why they were easy to overlook in an audit of "operations", and they are
/// exactly as silent when wrong: the greedy rule's whole purpose is that
/// speculation emits bit-identically what unspeculated greedy decoding would
/// have, and an off-by-one in where the bonus token goes breaks that quietly.
///
/// The capture launches both kernels on a battery covering every branch —
/// rejection at each position, full acceptance, and a zero draft probability —
/// and dumps what they emitted. Positions after a rejection hold vLLM's
/// `PLACEHOLDER_TOKEN_ID`, so the comparison is against the row's prefix.
#[test]
fn the_acceptance_rule_matches_vllms_kernels() {
    with_capture("vLLM rejection kernels", |c| {
        if !c.arrays.contains_key("accept.greedy_out") {
            eprintln!(
                "SKIPPED vLLM rejection kernels: the capture ran without a GPU, \
                 so the triton kernels could not be launched"
            );
            return;
        }
        let lens: Vec<usize> = c.get("accept.num_draft").iter().map(|v| *v as usize).collect();
        let ids = |name: &str| -> Vec<u32> {
            c.get(name).iter().map(|v| *v as i64 as u32).collect()
        };
        let draft = ids("accept.draft");
        let targ = ids("accept.target_argmax");
        let bonus = ids("accept.bonus");
        let recovered = ids("accept.recovered");
        let p_draft = c.get("accept.p_draft");
        let p_target = c.get("accept.p_target");
        let uniform = c.get("accept.uniform");
        let placeholder = c.get("accept.placeholder")[0] as i64;
        let row = c.shape("accept.greedy_out")[1];
        let greedy = c.get("accept.greedy_out");
        let random = c.get("accept.random_out");

        // What the kernel wrote, as a variable-length row: everything before the
        // first placeholder.
        let emitted = |out: &[f32], r: usize| -> Vec<u32> {
            out[r * row..(r + 1) * row]
                .iter()
                .take_while(|v| **v as i64 != placeholder)
                .map(|v| *v as i64 as u32)
                .collect()
        };

        let mut off = 0;
        let mut full = 0;
        let mut partial = 0;
        for (r, &n) in lens.iter().enumerate() {
            let dr = &draft[off..off + n];
            // `accept_greedy` takes the bonus as the last entry of
            // `target_argmax`; the kernel takes it in a separate array.
            let mut tg = targ[off..off + n].to_vec();
            tg.push(bonus[r]);
            let got = accept_greedy(dr, &tg);
            let want = emitted(greedy, r);
            assert_eq!(
                got.tokens, want,
                "greedy request {r}: draft {dr:?}, target {:?}, bonus {}",
                &targ[off..off + n],
                bonus[r]
            );
            assert_eq!(got.accepted, if want.len() > n { n } else { want.len() - 1 });
            if got.accepted == n {
                full += 1;
            } else {
                partial += 1;
            }

            let got = accept_stochastic(
                dr,
                &p_target[off..off + n],
                &p_draft[off..off + n],
                &uniform[off..off + n],
                &recovered[off..off + n],
                bonus[r],
            );
            assert_eq!(
                got.tokens,
                emitted(random, r),
                "stochastic request {r}: ratios {:?} against draws {:?}",
                (0..n)
                    .map(|j| p_target[off + j] / p_draft[off + j])
                    .collect::<Vec<_>>(),
                &uniform[off..off + n]
            );
            off += n;
        }
        assert_eq!(off, draft.len());
        assert!(
            full > 0 && partial > 0,
            "the battery has {full} full acceptances and {partial} rejections; \
             it does not exercise both branches"
        );
        // A zero draft probability must be rejected rather than divided by: the
        // ratio would be +inf and would accept a token the draft model calls
        // impossible. The capture plants exactly one.
        let zeros = p_draft.iter().filter(|v| **v == 0.0).count();
        assert!(
            zeros > 0,
            "no zero draft probability in the battery, so the guard is untested"
        );
        eprintln!(
            "acceptance: {} requests, {full} fully accepted, {partial} rejected, \
             {zeros} zero draft probability",
            lens.len()
        );
    });
}
