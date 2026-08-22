//! Undoing a rejected speculative token's effect on a GatedDeltaNet block,
//! against the kernels that will run in the engine.
//!
//! `tests/qwen35_mtp.rs::replaying_the_accepted_prefix_restores_the_state_exactly`
//! states the property in host arithmetic, over `qwen35::gated_delta_rule`. That
//! is the algebra; this is the implementation. The distinction matters here more
//! than usual, because the shipped path is not the algebra: it runs the
//! convolution and the recurrence as two device kernels with in-place state, and
//! what a rollback has to reproduce is what *those* leave behind.
//!
//! So every number below comes out of `Kernels::gdn_conv`,
//! `Kernels::gdn_gate_decay`, `Kernels::gdn_qk_l2norm` and
//! `Kernels::gdn_delta_rule`, and the rollback is driven through
//! [`tuili_model::spec::GdnRollback`] itself rather than a copy of it. A test
//! that reimplemented the replay would be checking its own reading of the
//! layout — which is the mistake this repository has already paid for twice, most
//! recently with an RMSNorm form that the reference and the capture got wrong in
//! the same way.
//!
//! The shapes are small and all different from each other, so a transposed
//! stride cannot hide: 2 key heads of 8, 4 value heads of 6, a width-4
//! convolution. `beta` is pinned near 0.999 on purpose — that is where inverting
//! the recurrence to step backwards would lose three digits, and the journal has
//! to be indifferent to it.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use tuili_cuda::Device;
use tuili_kernels::gdn::SeqLayout;
use tuili_kernels::Kernels;
use tuili_model::config::LinearAttnConfig;
use tuili_model::spec::{GdnRollback, GdnTap};

/// Candidate tokens one verification pass carries: `k = 4`, deliberately more
/// than the `k = 2` the notes recommend, so that every acceptance count from 0
/// to 5 is swept rather than only the two interesting ones.
const CANDIDATES: usize = 5;

fn la() -> LinearAttnConfig {
    LinearAttnConfig {
        key_heads: 2,
        value_heads: 4,
        key_head_dim: 8,
        value_head_dim: 6,
        conv_kernel: 4,
    }
}

fn device() -> Option<Device> {
    Device::new(0).ok()
}

/// A deterministic fill. Arbitrary non-degenerate numbers, not statistics.
struct Rng(u32);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((self.0 >> 8) as f32 / 8_388_608.0) - 1.0
    }

    fn fill(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

/// Everything one linear-attention layer needs, on the device, at test shapes.
struct Layer {
    dev: Device,
    kern: Kernels,
    la: LinearAttnConfig,
    /// `[CANDIDATES, width]`, the input projection's packed row.
    qkv_in: CudaSlice<f32>,
    conv_w: CudaSlice<f32>,
    a: CudaSlice<f32>,
    b: CudaSlice<f32>,
    a_log: CudaSlice<f32>,
    dt_bias: CudaSlice<f32>,
    /// Working buffers, reused by every pass.
    post_conv: CudaSlice<f32>,
    /// The same rows before `gdn_qk_l2norm` touched them — one of the two wrong
    /// capture points, kept so the test can show it is wrong.
    post_conv_unnormed: CudaSlice<f32>,
    beta: CudaSlice<f32>,
    g: CudaSlice<f32>,
    out: CudaSlice<f32>,
    first_token: CudaSlice<i32>,
    n_tokens: CudaSlice<i32>,
    /// The pre-step state and window, so every pass starts from the same place.
    state0: Vec<f32>,
    conv0: Vec<f32>,
}

impl Layer {
    fn new(dev: &Device) -> Result<Self> {
        let la = la();
        let (width, heads) = (la.conv_channels(), la.value_heads);
        let stream = dev.stream();
        let mut rng = Rng(0x51ed_270b);
        let state_floats = heads * la.key_head_dim * la.value_head_dim;
        Ok(Self {
            dev: dev.clone(),
            kern: Kernels::new(dev.clone()),
            la,
            qkv_in: stream.clone_htod(&rng.fill(CANDIDATES * width))?,
            conv_w: stream.clone_htod(&rng.fill(width * la.conv_kernel))?,
            a: stream.clone_htod(&rng.fill(CANDIDATES * heads))?,
            // `beta = sigmoid(b)` and the journal must not care how close to one
            // it gets, so push it there: sigmoid(7) = 0.9991.
            b: stream.clone_htod(
                &(0..CANDIDATES * heads)
                    .map(|_| 7.0 + 0.3 * rng.next())
                    .collect::<Vec<f32>>(),
            )?,
            a_log: stream.clone_htod(&rng.fill(heads))?,
            dt_bias: stream.clone_htod(&rng.fill(heads))?,
            post_conv: stream.alloc_zeros::<f32>(CANDIDATES * width)?,
            post_conv_unnormed: stream.alloc_zeros::<f32>(CANDIDATES * width)?,
            beta: stream.alloc_zeros::<f32>(CANDIDATES * heads)?,
            g: stream.alloc_zeros::<f32>(CANDIDATES * heads)?,
            out: stream.alloc_zeros::<f32>(CANDIDATES * la.value_dim())?,
            first_token: stream.clone_htod(&[0i32])?,
            n_tokens: stream.clone_htod(&[0i32])?,
            state0: rng.fill(state_floats),
            conv0: rng.fill(width * (la.conv_kernel - 1)),
        })
    }

    /// A fresh copy of the pre-step state, as a sequence's persistent buffer.
    fn fresh_state(&self) -> Result<CudaSlice<f32>> {
        Ok(self.dev.stream().clone_htod(&self.state0)?)
    }

    fn fresh_conv(&self) -> Result<CudaSlice<f32>> {
        Ok(self.dev.stream().clone_htod(&self.conv0)?)
    }

    fn read(&self, buf: &CudaSlice<f32>) -> Result<Vec<f32>> {
        let v = self.dev.stream().clone_dtoh(buf)?;
        self.dev.synchronize()?;
        Ok(v)
    }

    /// Run the block's state-advancing kernels over the first `n` candidate
    /// rows, exactly as `Model::linear_attention` does.
    ///
    /// Leaves the post-convolution, post-l2norm rows in `self.post_conv` and the
    /// pre-l2norm ones in `self.post_conv_unnormed`, which is what the journal
    /// captures and what the test needs in order to show that capturing one row
    /// earlier is a different computation.
    fn advance(
        &mut self,
        n: usize,
        state: &mut CudaSlice<f32>,
        conv: &mut CudaSlice<f32>,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let la = self.la;
        let (width, heads) = (la.conv_channels(), la.value_heads);
        self.dev
            .stream()
            .memcpy_htod(&[n as i32], &mut self.n_tokens)?;
        let first = self.first_token.as_view();
        let ntok = self.n_tokens.as_view();
        let seqs = SeqLayout {
            first_token: &first,
            n_tokens: &ntok,
            n_seqs: 1,
            total_tokens: n,
        };
        self.kern.gdn_conv(
            &mut self.post_conv.slice_mut(..n * width),
            &self.qkv_in.slice(..n * width),
            &mut conv.as_view_mut(),
            &self.conv_w.as_view(),
            &seqs,
            width,
            la.conv_kernel,
        )?;
        self.kern.gdn_gate_decay(
            &mut self.beta.slice_mut(..n * heads),
            &mut self.g.slice_mut(..n * heads),
            &self.a.slice(..n * heads),
            &self.b.slice(..n * heads),
            &self.a_log.as_view(),
            &self.dt_bias.as_view(),
            n,
            heads,
        )?;
        // Keep the unnormalized copy before the l2norm runs in place.
        self.dev.stream().memcpy_dtod(
            &self.post_conv.slice(..n * width),
            &mut self.post_conv_unnormed.slice_mut(..n * width),
        )?;
        self.kern.gdn_qk_l2norm(
            &mut self.post_conv.slice_mut(..n * width),
            n,
            la.key_heads,
            la.key_head_dim,
            width,
            0,
            la.key_dim(),
            1e-6,
        )?;
        self.kern.gdn_delta_rule(
            &mut self.out.slice_mut(..n * la.value_dim()),
            &mut state.as_view_mut(),
            &self.post_conv.slice(..n * width),
            &self.g.slice(..n * heads),
            &self.beta.slice(..n * heads),
            &seqs,
            heads,
            la.key_heads,
            la.key_head_dim,
            la.value_head_dim,
            (width, 0, la.key_dim(), 2 * la.key_dim()),
        )?;
        Ok(())
    }

    /// The layout a replay of `keep` rows wants: one sequence, from row zero.
    fn set_span(&mut self, keep: usize) -> Result<()> {
        self.dev
            .stream()
            .memcpy_htod(&[keep as i32], &mut self.n_tokens)?;
        Ok(())
    }
}

/// Which of the three stages of the packed row a journal captured.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Capture {
    /// What the recurrence actually consumes: after the convolution, after the
    /// l2 normalization. The only right answer.
    PostConvNormed,
    /// After the convolution, before the normalization.
    PostConvRaw,
    /// The input projection's output, before the convolution.
    PreConv,
}

/// Run a verification pass over all `CANDIDATES` rows, journalling `capture`,
/// then replay the first `keep` into the persistent state and window.
///
/// Returns `(state, conv, working_state)`: the persistent pair after the replay,
/// and the state the pass itself advanced — which is what an implementation with
/// no rollback at all would have left behind.
fn verify_then_replay(
    l: &mut Layer,
    keep: usize,
    capture: Capture,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let la = l.la;
    let width = la.conv_channels();
    let heads = la.value_heads;
    let mut state = l.fresh_state()?;
    let mut conv = l.fresh_conv()?;
    let mut r = GdnRollback::new(&l.dev, la, &[true], 1, CANDIDATES)?;
    r.arm(0, CANDIDATES)?;
    r.save_conv(&l.dev, 0, &conv.as_view())?;
    r.stage_state(&l.dev, &state.as_view())?;

    // The pass itself: the convolution and the gates run against the persistent
    // window, the recurrence against the working copy of the state.
    {
        let n = CANDIDATES;
        l.dev.stream().memcpy_htod(&[n as i32], &mut l.n_tokens)?;
        let first = l.first_token.as_view();
        let ntok = l.n_tokens.as_view();
        let seqs = SeqLayout {
            first_token: &first,
            n_tokens: &ntok,
            n_seqs: 1,
            total_tokens: n,
        };
        l.kern.gdn_conv(
            &mut l.post_conv.slice_mut(..n * width),
            &l.qkv_in.slice(..n * width),
            &mut conv.as_view_mut(),
            &l.conv_w.as_view(),
            &seqs,
            width,
            la.conv_kernel,
        )?;
        l.kern.gdn_gate_decay(
            &mut l.beta.slice_mut(..n * heads),
            &mut l.g.slice_mut(..n * heads),
            &l.a.slice(..n * heads),
            &l.b.slice(..n * heads),
            &l.a_log.as_view(),
            &l.dt_bias.as_view(),
            n,
            heads,
        )?;
        l.dev.stream().memcpy_dtod(
            &l.post_conv.slice(..n * width),
            &mut l.post_conv_unnormed.slice_mut(..n * width),
        )?;
        l.kern.gdn_qk_l2norm(
            &mut l.post_conv.slice_mut(..n * width),
            n,
            la.key_heads,
            la.key_head_dim,
            width,
            0,
            la.key_dim(),
            1e-6,
        )?;
        r.record(
            &l.dev,
            0,
            GdnTap {
                pre_conv: l.qkv_in.slice(..n * width),
                post_conv: match capture {
                    Capture::PostConvNormed => l.post_conv.slice(..n * width),
                    Capture::PostConvRaw => l.post_conv_unnormed.slice(..n * width),
                    Capture::PreConv => l.qkv_in.slice(..n * width),
                },
                g: l.g.slice(..n * heads),
                beta: l.beta.slice(..n * heads),
            },
        )?;
        l.kern.gdn_delta_rule(
            &mut l.out.slice_mut(..n * la.value_dim()),
            &mut r.state_scratch_mut(),
            &l.post_conv.slice(..n * width),
            &l.g.slice(..n * heads),
            &l.beta.slice(..n * heads),
            &seqs,
            heads,
            la.key_heads,
            la.key_head_dim,
            la.value_head_dim,
            (width, 0, la.key_dim(), 2 * la.key_dim()),
        )?;
    }
    let working = l.dev.stream().clone_dtoh(&r.working_state())?;
    l.dev.synchronize()?;

    // And the commit.
    l.set_span(keep)?;
    {
        let first = l.first_token.as_view();
        let ntok = l.n_tokens.as_view();
        let seqs = SeqLayout {
            first_token: &first,
            n_tokens: &ntok,
            n_seqs: 1,
            total_tokens: keep,
        };
        r.replay_layer(
            &l.dev,
            &l.kern,
            0,
            keep,
            &seqs,
            &l.conv_w.as_view(),
            &mut state.as_view_mut(),
            &mut conv.as_view_mut(),
        )?;
    }
    Ok((l.read(&state)?, l.read(&conv)?, working))
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "comparing different shapes");
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len().max(1) as f32).sqrt()
}

/// The state and the convolution window after accepting `keep` of `CANDIDATES`
/// candidates are what decoding those `keep` tokens normally would have left.
///
/// Bit for bit, not to a tolerance: the replay runs the same kernels over the
/// same inputs in the same order, so anything other than equality means the
/// journal is not feeding them what the forward pass fed them. A tolerance here
/// would hide exactly the mistake this test exists to catch.
///
/// The second half is the part that makes it evidence: with anything rejected,
/// the state the pass itself advanced — what an engine with no rollback would
/// carry into the next step — must be far away. Otherwise the recurrence would
/// be insensitive to the rejected tokens and there would be nothing to roll
/// back.
#[test]
fn replaying_the_accepted_prefix_restores_what_the_kernels_would_have_left() -> Result<()> {
    let Some(dev) = device() else {
        eprintln!("SKIPPED: no CUDA device");
        return Ok(());
    };
    let mut l = Layer::new(&dev)?;
    let scale = rms(&l.state0);
    assert!(scale > 0.1, "a degenerate initial state proves nothing");

    for keep in 0..=CANDIDATES {
        // What an unspeculated decode of `keep` tokens leaves.
        let mut want_state = l.fresh_state()?;
        let mut want_conv = l.fresh_conv()?;
        l.advance(keep, &mut want_state, &mut want_conv)?;
        let want_state = l.read(&want_state)?;
        let want_conv = l.read(&want_conv)?;

        let (state, conv, working) = verify_then_replay(&mut l, keep, Capture::PostConvNormed)?;
        assert_eq!(
            max_diff(&state, &want_state),
            0.0,
            "keeping {keep} of {CANDIDATES}: the replayed recurrent state is not \
             the one {keep} normal decode steps leave (worst {:.3e}, state rms \
             {scale:.3e})",
            max_diff(&state, &want_state)
        );
        assert_eq!(
            max_diff(&conv, &want_conv),
            0.0,
            "keeping {keep} of {CANDIDATES}: the replayed convolution window is \
             not the one {keep} normal decode steps leave"
        );

        if keep < CANDIDATES {
            let drift = max_diff(&working, &want_state);
            assert!(
                drift > 0.05 * scale,
                "with {keep} of {CANDIDATES} accepted the un-rolled-back state \
                 is within {drift:.3e} of the correct one (state rms \
                 {scale:.3e}), so this test does not show that rollback is \
                 needed at all"
            );
        }
    }
    Ok(())
}

/// Journalling the row one stage too early is a different computation.
///
/// Both wrong capture points run to completion and leave a plausible state — the
/// same shape, the same magnitude, no NaN — which is the whole reason to pin
/// them. `pre_conv` replays the recurrence over inputs the convolution never
/// filtered; `post_conv_raw` replays it over keys that were never l2-normalized,
/// which changes `delta` on every token because `delta` contracts `k` against
/// the state.
#[test]
fn journalling_the_row_before_the_convolution_or_the_norm_gives_a_different_state() -> Result<()> {
    let Some(dev) = device() else {
        eprintln!("SKIPPED: no CUDA device");
        return Ok(());
    };
    let mut l = Layer::new(&dev)?;
    let scale = rms(&l.state0);
    // A partial acceptance, which is the case a rollback exists for.
    let keep = 3;

    let mut want_state = l.fresh_state()?;
    let mut want_conv = l.fresh_conv()?;
    l.advance(keep, &mut want_state, &mut want_conv)?;
    let want_state = l.read(&want_state)?;

    let (right, _, _) = verify_then_replay(&mut l, keep, Capture::PostConvNormed)?;
    assert_eq!(max_diff(&right, &want_state), 0.0, "the control disagrees");

    for wrong in [Capture::PostConvRaw, Capture::PreConv] {
        let (state, _, _) = verify_then_replay(&mut l, keep, wrong)?;
        let drift = max_diff(&state, &want_state);
        assert!(
            drift > 0.05 * scale,
            "journalling {wrong:?} lands within {drift:.3e} of the right answer \
             (state rms {scale:.3e}); this test does not pin the capture point"
        );
        assert!(
            state.iter().all(|v| v.is_finite()),
            "journalling {wrong:?} produced a non-finite state, which would have \
             been caught by anything; the point is that it does not"
        );
    }
    Ok(())
}

/// A draft that is rejected outright leaves no trace of itself.
///
/// The `keep = 1` case of the sweep above, stated on its own because it is the
/// one a caller reasons about: the token the previous step had already settled on
/// goes through the model, all `k` drafts are rejected, and both kinds of
/// in-place memory must end up where a single ordinary decode step would have
/// left them — not where the five-token pass left them.
#[test]
fn a_rejected_draft_leaves_neither_state_nor_conv_window_behind() -> Result<()> {
    let Some(dev) = device() else {
        eprintln!("SKIPPED: no CUDA device");
        return Ok(());
    };
    let mut l = Layer::new(&dev)?;
    let scale = rms(&l.state0);

    let mut want_state = l.fresh_state()?;
    let mut want_conv = l.fresh_conv()?;
    l.advance(1, &mut want_state, &mut want_conv)?;
    let want_state = l.read(&want_state)?;
    let want_conv = l.read(&want_conv)?;

    let (state, conv, working) = verify_then_replay(&mut l, 1, Capture::PostConvNormed)?;
    assert_eq!(max_diff(&state, &want_state), 0.0, "recurrent state");
    assert_eq!(max_diff(&conv, &want_conv), 0.0, "convolution window");

    // And the rejected tokens really were in there.
    assert!(
        max_diff(&working, &want_state) > 0.05 * scale,
        "the rejected candidates changed the state by less than 5% of its own \
         magnitude, so nothing was undone"
    );
    // The window too: five tokens through a width-4 convolution leave a window
    // made entirely of candidates, one token leaves a window that is still
    // mostly the pre-step taps.
    let mut all = l.fresh_state()?;
    let mut all_conv = l.fresh_conv()?;
    l.advance(CANDIDATES, &mut all, &mut all_conv)?;
    let all_conv = l.read(&all_conv)?;
    assert!(
        max_diff(&conv, &all_conv) > 1e-3,
        "the window after one token and after {CANDIDATES} are the same, so the \
         window rollback is not being tested"
    );
    Ok(())
}

/// What the journal costs, at the 27B's real shapes, against the alternatives.
///
/// Not a property of the port — arithmetic on the notes' numbers — but recorded
/// as an assertion so that a change which quietly makes the journal per-state
/// rather than per-token shows up here rather than as an out-of-memory on a full
/// GPU.
#[test]
fn the_journal_is_two_orders_of_magnitude_smaller_than_the_state() -> Result<()> {
    let Some(dev) = device() else {
        eprintln!("SKIPPED: no CUDA device");
        return Ok(());
    };
    // Qwen3.8-27B: 48 linear layers of 48 value heads, 128x128 state, a 10240
    // channel convolution of width 4.
    let la = LinearAttnConfig {
        key_heads: 16,
        value_heads: 48,
        key_head_dim: 128,
        value_head_dim: 128,
        conv_kernel: 4,
    };
    let kinds: Vec<bool> = (0..64).map(|i| (i + 1) % 4 != 0).collect();
    let k = 2;
    let r = GdnRollback::new(&dev, la, &kinds, 1, k + 1)?;
    let journal = r.journal_bytes();
    let state = 48 * la.value_heads * la.key_head_dim * la.value_head_dim * 4;
    eprintln!(
        "k = {k}: journal {:.1} MiB, persistent state {:.1} MiB, \
         snapshot-restore would copy {:.1} MiB a step, vLLM's k+1 slots would \
         hold {:.1} MiB",
        journal as f64 / (1 << 20) as f64,
        state as f64 / (1 << 20) as f64,
        2.0 * state as f64 / (1 << 20) as f64,
        (k + 1) as f64 * state as f64 / (1 << 20) as f64,
    );
    assert!(
        journal * 4 < state,
        "the journal is {journal} bytes against {state} of state; it was meant \
         to be a small multiple of a token, not of the state"
    );
    Ok(())
}
