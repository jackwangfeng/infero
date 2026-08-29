//! Continuous batching.
//!
//! The engine runs one step at a time. Each step assembles a batch from every
//! sequence in flight and hands it to the model as a single forward pass; a
//! sequence that finishes leaves at the end of that step and a waiting request
//! takes its place at the start of the next, without anything else pausing.
//!
//! Two rules shape the batch:
//!
//! * **Decodes go first.** They cost one token each and a running sequence
//!   waiting on the token budget is a stall the client sees directly.
//! * **Prefill fills what is left, and may be split.** A long prompt is fed
//!   across several steps rather than blocking decodes for the whole of it,
//!   which is what keeps one 4000-token request from freezing everyone else.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use tokio::sync::mpsc;
use infero_model::qwen35_vision::{Grid, llm_position_ids};
use infero_model::{BatchItem, KvPool, Model, Sampler, SeqId, VisionFeatures};
use infero_tokenizer::Tokenizer;

use crate::engine::{Event, FinishReason, PendingImage, PendingVideo, Request};
use crate::stop::split_at_stop;

/// A sequence the scheduler is currently generating.
struct Running {
    seq: SeqId,
    prompt: Vec<u32>,
    /// Prompt tokens already fed to the model.
    prefilled: usize,
    /// The cached blocks this sequence borrows, and the references it holds on
    /// them. Released when it retires; see `retire`.
    prefix: crate::prefix::Prefix,
    /// The tower's output for this sequence's one image or video, waiting for
    /// however many prefill steps it takes to splice the whole thing in --
    /// one, the common case, or several once a placeholder run outgrows a
    /// single step's `batch_tokens` budget. `admit` fills this in before the
    /// sequence starts running; `step` drops it once every pad token this
    /// clip's rows replace has been consumed (see `vision_pad_tok`'s doc
    /// comment below), the point past which nothing reads it again — holding
    /// the device buffer longer than that is a leak with no reader.
    vision: Option<VisionFeatures>,
    /// The placeholder id `vision`'s rows replace -- `img_tok` or `vid_tok`
    /// (see `Model::vision_tokens`). Meaningless (and never read) once
    /// `vision` is `None`. Set by `admit` alongside `vision`.
    ///
    /// What a prefill chunk `[from, from+len)` uses to compute
    /// `BatchItem::vision_row_offset`: **not** `from` minus some remembered
    /// start index. A video's placeholder tokens are not one contiguous run
    /// the way an image's are -- `encode_pending_video` interleaves each
    /// frame-group's pad run with `<T.T seconds>` text and its own
    /// `vision_start`/`vision_end`, so "how far into the prompt this chunk
    /// starts" and "how many pad tokens came before it" are different
    /// numbers once more than one frame-group exists. The only correct way
    /// to get the second number is to count actual occurrences of
    /// `vision_pad_tok` in `prompt[..from]` -- see the two call sites in
    /// `step`. A real, reproduced bug: an index-arithmetic shortcut here
    /// passed every single-chunk (image, or a video whose whole run fit in
    /// one step) test in this codebase and only broke on a video chunked
    /// across steps with more than one frame-group, which is exactly the
    /// case chunking exists for.
    vision_pad_tok: u32,
    /// Whether this sequence ever carried an image, kept after `vision` above
    /// is taken.
    ///
    /// The reason it outlives `vision`: prefix caching hashes token ids, and an
    /// image's placeholder run is the same `image_token_id` repeated
    /// regardless of which picture produced it — the pixels reach the model
    /// through a splice the hash never sees. Two different images that resize
    /// to the same token count would otherwise look like the same prefix, and
    /// a hit would serve one image's computed KV for another's. `admit` and
    /// `retire` both check this to keep such a sequence out of the cache
    /// entirely, not just out of collisions with itself.
    had_vision: bool,
    /// This sequence's absolute M-RoPE `[T, H, W]` positions, token-major
    /// (`3 * prompt.len()` entries) over the whole *expanded* prompt.
    /// `None` for a text-only sequence, or when the model has no
    /// `cfg.mrope_section`.
    ///
    /// Outlives `vision` the same way `had_vision` does, and for the same
    /// reason: a decode step's position still depends on this sequence ever
    /// having carried an image, long after `vision`'s one prefill has come
    /// and gone. `admit` fills it in from `qwen35_vision::llm_position_ids`
    /// right after the placeholder expansion, while the grid is still in
    /// hand. `step` slices it absolutely at `from..from+len` for a prefill
    /// chunk, which is what makes it correct under a prefix-cache hit too
    /// (`from` starts wherever the cached prefix left off, not at zero).
    mrope: Option<Box<[i32]>>,
    /// The constant M-RoPE adds to a plain running length once the prompt is
    /// in: `0` without M-RoPE, negative for a sequence that passed through an
    /// image (see `BatchItem::mrope_delta`'s doc comment for why negative is
    /// correct, not a bug). Computed alongside `mrope` and constant for the
    /// sequence's lifetime.
    mrope_delta: i32,
    /// The token to feed next, once the prompt is in.
    next: Option<u32>,
    generated: Vec<u32>,
    sampler: Sampler,
    /// Bytes held back because they do not yet form a character.
    partial: Vec<u8>,
    /// Text held back because it might begin a stop sequence.
    pending: String,
    stop: Vec<String>,
    budget: usize,
    events: mpsc::UnboundedSender<Event>,
    admitted: Instant,
    prefill_done: Option<Instant>,
    /// This step's sampled token, filled by the parallel pass below.
    sampled: u32,
    /// What to hand the drafter next round, when speculation is running.
    ///
    /// `None` until the prompt is in — the drafter's first input is the
    /// prefill's own hidden states — and `None` again after any step that did
    /// not go through the speculative path, since a plain decode step leaves the
    /// drafter's cache a token behind.
    spec_feed: Option<infero_model::spec::DraftFeed>,
    /// Set once the drafter's cache has fallen behind this sequence and cannot
    /// be caught up. Latched, so the explanation is logged once rather than per
    /// step, and so the check that follows it stays cheap.
    spec_desynced: bool,
    /// Verification rounds this sequence ran, and tokens they emitted.
    ///
    /// Per request rather than per server: the mean acceptance length is the
    /// number the speedup is proportional to, and averaging it across requests
    /// that speculated and requests that could not says nothing about either.
    spec_rounds: u64,
    spec_emitted: u64,
}

impl Running {
    fn prompt_complete(&self) -> bool {
        self.prefilled >= self.prompt.len()
    }
}

/// What a sequence is doing this step.
enum Work {
    /// Feed this slice of the prompt; `final_chunk` gets logits.
    Prefill { from: usize, len: usize, last: bool },
    /// Feed the token this sequence sampled last step.
    Decode,
}

pub struct Scheduler {
    model: Model,
    pool: KvPool,
    tokenizer: Arc<Tokenizer>,
    waiting: VecDeque<Request>,
    running: Vec<Running>,
    steps: u64,
    profile: bool,
    /// Time to *issue* the forward pass, not to run it. Since the device
    /// sampler took over, nothing between the launches and the sample waits on
    /// the GPU, so this reads near zero and the wait shows up in `t_sample`.
    /// It was `forward_ms` when the step ended in a synchronise, and reading
    /// the new number under the old name is how you conclude the forward pass
    /// got sixty times faster.
    t_issue: f64,
    /// Sampling, which is where the step now blocks on the whole forward pass.
    t_sample: f64,
    /// Per-sequence bookkeeping: detokenizing, stop scanning, delivery.
    t_advance: f64,
    /// Whole `step()`, and the wall clock between one step ending and the next
    /// beginning. A model that gets faster while the served rate does not is
    /// either losing the gain in the bookkeeping around the step or in the gap
    /// between steps, and only measuring both tells you which.
    t_step: f64,
    t_gap: f64,
    /// How many tokens a speculative round drafts. Zero when off, and this is
    /// the round-to-round value under `INFERO_ADAPTIVE_SPEC` -- see
    /// `spec_k_max` for the ceiling `enable_speculation` actually provisioned.
    spec_k: usize,
    /// What `enable_speculation` sized the GDN rollback journal for.
    /// `spec_k` may sit anywhere in `1..=spec_k_max` under adaptive mode
    /// without a reallocation; going higher would need one, which is why
    /// this is a ceiling rather than something adaptive mode also moves.
    spec_k_max: usize,
    /// On when `INFERO_ADAPTIVE_SPEC` asks for it. Off leaves `spec_k` at
    /// whatever `INFERO_SPEC_K` set, which is the behaviour every measurement
    /// in this file's history was taken against.
    adaptive_spec: bool,
    /// Exponential moving average of a round's emitted tokens (`1..=k+1`),
    /// the signal `speculative_step` adjusts `spec_k` from. `None` before the
    /// first round, so the first sample sets it rather than being blended
    /// against an arbitrary starting guess.
    spec_ema_accept: Option<f64>,
    /// Rounds left before `spec_k` may move again. The thresholds it moves
    /// on are relative to the *current* `spec_k`, which is what makes a
    /// cooldown necessary rather than a tuning nicety: dropping a level
    /// lowers the bar for climbing back, in absolute EMA terms, so an
    /// undamped controller measured flipping every few rounds — the ceiling
    /// getting easier to clear is exactly what a level change should not do
    /// to the next decision.
    spec_k_cooldown: u64,
    spec_steps: u64,
    spec_tokens: u64,
    /// A round's three parts, summed over a window; see `speculative_step`.
    spec_draft_ms: f64,
    spec_verify_ms: f64,
    spec_after_ms: f64,
    spec_window: u64,
    last_end: Option<std::time::Instant>,
    window: u64,
    /// Cross-request KV reuse. `None` on a model whose recurrent state a shared
    /// prefix would not reconstruct — see `prefix`'s module note.
    prefix: Option<crate::prefix::PrefixCache>,
    /// The patch budget `Model::load_vision_tower` was given. `admit` has to
    /// pass the same number to `vision_resize`, because the tower's scratch —
    /// allocated once, at that size — is what `encode_image` writes into; a
    /// resize past it would overrun a buffer the loader already fixed.
    vision_max_patches: usize,
    /// Mean absolute pixel difference (0-255 scale) below which a video
    /// frame-group is dropped as a near-duplicate of the last *retained*
    /// group, before it ever reaches the vision tower -- see
    /// `drop_redundant_groups`'s own doc comment and `notes/video-encoding
    /// -optimizations.md`'s item 5. `<= 0.0` (the default) disables this,
    /// matching vLLM's own "pruning off unless a rate is explicitly set"
    /// convention.
    video_dedup_threshold: f64,
}

/// Absolute M-RoPE `[T, H, W]` positions for a prompt already spliced with one
/// image's placeholder run, token-major, plus the running-position delta a
/// decode step should carry forward afterwards.
///
/// `prompt` is the *expanded* prompt -- `encode_pending_image`'s output, `img_tok`
/// repeated `feats.tokens` times where the single placeholder used to be, not
/// the one-token version the template rendered. `qwen35_vision::llm_position_ids`
/// wants `token_types` (0 text, 1 image) at that same width, and a `Grid` in
/// pre-merge patch units, which is exactly what `VisionFeatures::grid_h/grid_w`
/// already are -- see that struct's own doc comment.
///
/// `llm_position_ids` returns `[3, seq]` (axis-major, matching the capture this
/// function is checked against in `crates/model/tests/qwen35_vision.rs`); the
/// transpose to token-major here is so a prefill chunk's slice
/// (`mrope[3*from..3*(from+len)]`) is one contiguous range rather than three.
fn mrope_for_image(prompt: &[u32], img_tok: u32, feats: &VisionFeatures, merge: usize) -> (Box<[i32]>, i32) {
    let grid = Grid { t: 1, h: feats.grid_h, w: feats.grid_w };
    mrope_from_grid(prompt, img_tok, grid, merge)
}

/// Same idea as [`mrope_for_image`], but a video's placeholder run is
/// `feats.grid_t` *separate* contiguous runs of `vid_tok` -- one a
/// frame-group, timestamp text in between -- rather than one run covering the
/// whole grid. `llm_position_ids` already expects exactly this shape (it
/// pulls one `Grid` off `grids` per contiguous visual run it walks into), so
/// the only change from the image path is passing `grid_t` copies of the same
/// per-group `Grid` instead of one.
fn mrope_for_video(prompt: &[u32], vid_tok: u32, feats: &VisionFeatures, merge: usize) -> (Box<[i32]>, i32) {
    let grid = Grid { t: 1, h: feats.grid_h, w: feats.grid_w };
    let grids = vec![grid; feats.grid_t];
    mrope_from_grids(prompt, vid_tok, &grids, merge)
}

/// [`mrope_for_image`]'s arithmetic, independent of `VisionFeatures` so it can
/// be unit-tested without a device.
fn mrope_from_grid(prompt: &[u32], img_tok: u32, grid: Grid, merge: usize) -> (Box<[i32]>, i32) {
    mrope_from_grids(prompt, img_tok, &[grid], merge)
}

/// [`mrope_from_grid`] generalized to `grids.len()` separate visual runs, all
/// marked by the same `marker_tok` -- what [`mrope_for_video`] needs and
/// [`mrope_from_grid`] is now a one-grid instance of.
fn mrope_from_grids(prompt: &[u32], marker_tok: u32, grids: &[Grid], merge: usize) -> (Box<[i32]>, i32) {
    let token_types: Vec<u8> = prompt.iter().map(|&t| u8::from(t == marker_tok)).collect();
    let axis_major = llm_position_ids(&token_types, grids, merge);
    let seq = prompt.len();
    let mut token_major = vec![0i32; 3 * seq];
    for t in 0..seq {
        for axis in 0..3 {
            token_major[3 * t + axis] = axis_major[axis * seq + t] as i32;
        }
    }
    let max = axis_major.iter().copied().max().unwrap_or(0) as i32;
    let delta = max + 1 - seq as i32;
    (token_major.into_boxed_slice(), delta)
}

#[cfg(test)]
mod mrope_tests {
    use super::*;

    /// Text(2) + a 4x4 pre-merge grid at merge 2 (post-merge 2x2, 4 image
    /// tokens) + text(1). Hand-computed against `llm_position_ids`'s own
    /// documented rule (advance by `max(lh, lw)`, not token count), not just
    /// echoed back from the function under test:
    ///
    /// token  0    1    2      3      4      5      6
    /// kind   text text img    img    img    img    text
    /// (T,H,W)(0,0,0)(1,1,1)(2,2,2)(2,2,3)(2,3,2)(2,3,3)(4,4,4)
    ///
    /// The image run's four entries are `(hi,wi)` in `(0,0),(0,1),(1,0),(1,1)`
    /// order, T constant at the text run's length (2), H/W offset by `hi`/`wi`.
    /// The text run after it resumes at `2 + max(2,2) = 4`, not at `2 + 4 = 6`
    /// -- the token-counting mistake `llm_position_ids`'s own doc comment
    /// warns about. Max position is 4, over a 7-token prompt, so delta is
    /// `4 + 1 - 7 = -2`: negative, which is the sign that the advance rule
    /// was applied and not skipped.
    const IMG: u32 = 999;
    fn prompt() -> Vec<u32> {
        vec![10, 11, IMG, IMG, IMG, IMG, 12]
    }
    fn grid() -> Grid {
        Grid { t: 1, h: 4, w: 4 }
    }
    fn expected() -> Vec<[i32; 3]> {
        vec![[0, 0, 0], [1, 1, 1], [2, 2, 2], [2, 2, 3], [2, 3, 2], [2, 3, 3], [4, 4, 4]]
    }

    #[test]
    fn matches_the_hand_computed_positions_and_a_negative_delta() {
        let (mrope, delta) = mrope_from_grid(&prompt(), IMG, grid(), 2);
        let want = expected();
        for (t, w) in want.iter().enumerate() {
            assert_eq!(&mrope[3 * t..3 * t + 3], w, "token {t}");
        }
        assert_eq!(delta, -2, "max position 4 over 7 tokens");
        assert!(delta < 0, "the advance-by-token-count bug gives delta >= 0");
    }

    /// The transpose is a real transpose of `llm_position_ids`'s own output,
    /// not a coincidence of the hand-computed table above: recomputed
    /// independently here via `llm_position_ids` directly plus a manual
    /// transpose, and compared against `mrope_from_grid`'s result.
    #[test]
    fn the_transpose_matches_llm_position_ids_independently_reshaped() {
        let p = prompt();
        let token_types: Vec<u8> = p.iter().map(|&t| u8::from(t == IMG)).collect();
        let axis_major = llm_position_ids(&token_types, &[grid()], 2);
        let seq = p.len();
        let (mrope, _) = mrope_from_grid(&p, IMG, grid(), 2);
        for t in 0..seq {
            for axis in 0..3 {
                assert_eq!(
                    mrope[3 * t + axis],
                    axis_major[axis * seq + t] as i32,
                    "token {t} axis {axis}"
                );
            }
        }
    }

    /// A prefill chunk's slice, `mrope[3*from..3*(from+len)]` -- the shape
    /// `step()` actually reads -- lands on the right tokens. This is the
    /// property a prefix-cache hit depends on: `from` can be nonzero.
    #[test]
    fn a_chunk_slice_lands_on_the_right_tokens() {
        let (mrope, _) = mrope_from_grid(&prompt(), IMG, grid(), 2);
        let (from, len) = (3, 2);
        let chunk = &mrope[3 * from..3 * (from + len)];
        let want = expected();
        assert_eq!(chunk, [want[3], want[4]].concat());
    }

    /// A prompt with no image token at all -- every position is `t` on all
    /// three axes, and the delta is exactly zero: M-RoPE degenerates to the
    /// plain scalar position when there is nothing visual to advance around.
    #[test]
    fn a_text_only_prompt_gives_zero_delta_and_equal_axes() {
        let p: Vec<u32> = vec![1, 2, 3, 4, 5];
        let (mrope, delta) = mrope_from_grid(&p, IMG, grid(), 2);
        assert_eq!(delta, 0);
        for t in 0..p.len() {
            assert_eq!(&mrope[3 * t..3 * t + 3], &[t as i32, t as i32, t as i32]);
        }
    }
}

/// Drop video frame-groups that are near-duplicates of the last *retained*
/// group -- a coarser, much safer relative of vLLM's per-token EVS pruning
/// (`vllm/multimodal/video_prune/evs.py`, read directly from vLLM's real
/// source; see `notes/video-encoding-optimizations.md`'s item 5 for the full
/// writeup and why the token-level version was not attempted). Whole groups
/// are kept or dropped, never split, so every downstream stage of this
/// pipeline -- `vision_resize_video`, `prepare_clip`, `encode_clip`, the
/// M-RoPE position math -- sees an ordinary (if shorter) clip and needs no
/// changes: `grid_t` already varies request to request today.
///
/// Runs on raw decoded pixels, before the tower ever sees a frame -- unlike
/// EVS (which prunes the tower's *output*), this also saves the compute a
/// dropped group would have cost, not just the tokens it would have spent.
///
/// `frames` is `PendingVideo`/`DecodedClip`'s own layout: interleaved RGB8,
/// two entries a temporal-patch group (a group's second frame duplicates the
/// first if the clip's raw frame count was odd, same convention `prepare_clip`
/// documents). `timestamps` has one entry a group. `threshold` is mean
/// absolute per-channel pixel difference (0-255 scale); group 0 is always
/// kept. Group `gi`'s frames are compared against the last *retained* group's,
/// not simply the previous group -- a slow fade across several near-identical
/// groups should not each compare "close enough" to its immediate
/// predecessor and all survive; they should all compare against the one
/// frame actually being kept.
fn drop_redundant_groups(frames: &[Vec<u8>], timestamps: &[f64], threshold: f64) -> (Vec<Vec<u8>>, Vec<f64>) {
    if timestamps.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut kept_frames = Vec::with_capacity(frames.len());
    let mut kept_timestamps = Vec::with_capacity(timestamps.len());
    let mut last_kept_gi: Option<usize> = None;
    for gi in 0..timestamps.len() {
        let a = &frames[2 * gi];
        let b = frames.get(2 * gi + 1).unwrap_or(a);
        let keep = match last_kept_gi {
            None => true,
            Some(lgi) => {
                let la = &frames[2 * lgi];
                let lb = frames.get(2 * lgi + 1).unwrap_or(la);
                mean_abs_diff(a, la) > threshold || mean_abs_diff(b, lb) > threshold
            }
        };
        if keep {
            kept_frames.push(a.clone());
            kept_frames.push(b.clone());
            kept_timestamps.push(timestamps[gi]);
            last_kept_gi = Some(gi);
        }
    }
    (kept_frames, kept_timestamps)
}

/// Mean absolute byte difference between two equal-shaped interleaved RGB8
/// buffers. A shape mismatch (should not happen -- every sampled frame
/// shares one source resolution) returns `f64::INFINITY` so the caller keeps
/// the group rather than silently comparing against nothing.
fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return f64::INFINITY;
    }
    let sum: u64 = a.iter().zip(b).map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64).sum();
    sum as f64 / a.len() as f64
}

#[cfg(test)]
mod dedup_tests {
    use super::*;

    fn solid(v: u8) -> Vec<u8> {
        vec![v; 12] // a tiny "frame": 4 pixels x 3 channels
    }

    #[test]
    fn identical_groups_collapse_to_the_first() {
        let frames = vec![solid(10), solid(10), solid(10), solid(10), solid(10), solid(10)];
        let timestamps = vec![0.0, 1.0, 2.0];
        let (kf, kt) = drop_redundant_groups(&frames, &timestamps, 1.0);
        assert_eq!(kt, vec![0.0], "three identical groups should collapse to one");
        assert_eq!(kf, vec![solid(10), solid(10)]);
    }

    #[test]
    fn a_real_change_survives() {
        let frames = vec![solid(10), solid(10), solid(200), solid(200)];
        let timestamps = vec![0.0, 1.0];
        let (kf, kt) = drop_redundant_groups(&frames, &timestamps, 1.0);
        assert_eq!(kt, vec![0.0, 1.0], "a genuinely different group must survive");
        assert_eq!(kf, frames);
    }

    #[test]
    fn a_threshold_of_zero_still_drops_exact_duplicates() {
        // The "disabled by default" behaviour lives in the *caller*
        // (`encode_pending_video` only calls this at all when
        // `video_dedup_threshold > 0.0`) -- this function has no separate
        // disable guard of its own. At a threshold of exactly zero, two
        // byte-identical groups still diff at 0, which is not `> 0.0`, so
        // the second one is still dropped as redundant.
        let frames = vec![solid(10), solid(10), solid(10), solid(10)];
        let timestamps = vec![0.0, 1.0];
        let (_, kt) = drop_redundant_groups(&frames, &timestamps, 0.0);
        assert_eq!(kt, vec![0.0]);
    }

    #[test]
    fn compares_against_the_last_retained_group_not_the_immediate_predecessor() {
        // A slow fade: 10 -> 12 -> 14 -> ... each step under the threshold
        // versus its immediate neighbour, but a real cumulative change
        // versus whichever group is actually still retained.
        let vals = [10u8, 12, 14, 16, 18, 20];
        let frames: Vec<Vec<u8>> = vals.iter().flat_map(|&v| [solid(v), solid(v)]).collect();
        let timestamps: Vec<f64> = (0..vals.len()).map(|i| i as f64).collect();
        let (_, kt) = drop_redundant_groups(&frames, &timestamps, 3.0);
        // Kept: 0 (10), then the first one that differs from 10 by > 3,
        // which is 16 (index 3, diff 6) -- 12 and 14 diff from the *kept*
        // group (10) by only 2 and 4... 4 > 3, so 14 (index 2) should
        // actually be the second kept group. Recompute by hand: vs kept=10,
        // group1(12) diff=2 <=3 drop; group2(14) diff=4 >3 keep (now kept=14);
        // group3(16) vs 14 diff=2 <=3 drop; group4(18) vs 14 diff=4 >3 keep
        // (kept=18); group5(20) vs 18 diff=2 <=3 drop.
        assert_eq!(kt, vec![0.0, 2.0, 4.0]);
    }
}

impl Scheduler {
    pub fn new(model: Model, pool: KvPool, tokenizer: Arc<Tokenizer>) -> Self {
        let prefix = crate::prefix::PrefixCache::new(&pool);
        Self {
            prefix,
            vision_max_patches: 0,
            video_dedup_threshold: 0.0,
            model,
            pool,
            tokenizer,
            waiting: VecDeque::new(),
            running: Vec::new(),
            steps: 0,
            // A separate switch from `INFERO_PROFILE`: that one turns on
            // per-kernel CUDA events, which cannot coexist with graph capture,
            // so it can only ever report an ungraphed step. This one is pure
            // host-side wall clock and leaves the graphs alone.
            profile: std::env::var_os("INFERO_PROFILE").is_some()
                || std::env::var_os("INFERO_STEP_TIMING").is_some(),
            t_issue: 0.0,
            t_sample: 0.0,
            t_advance: 0.0,
            t_step: 0.0,
            t_gap: 0.0,
            last_end: None,
            window: 0,
            // Speculation is off unless a head is installed and `INFERO_SPEC_K`
            // asks for it. Zero means off, which is also what a checkpoint
            // without an MTP head leaves it at.
            spec_k: 0,
            spec_k_max: 0,
            adaptive_spec: std::env::var_os("INFERO_ADAPTIVE_SPEC").is_some(),
            spec_ema_accept: None,
            spec_k_cooldown: 0,
            spec_steps: 0,
            spec_tokens: 0,
            spec_draft_ms: 0.0,
            spec_verify_ms: 0.0,
            spec_after_ms: 0.0,
            spec_window: 0,
        }
    }

    /// Load the checkpoint's MTP head and turn on speculation.
    ///
    /// Returns false when the checkpoint has no head, which is not an error.
    /// `k` is how many tokens a round drafts; the verification pass is `k + 1`
    /// rows wide, so the model has to have been built with at least that many
    /// logit rows.
    /// Record the patch budget the vision tower was loaded with, so `admit`
    /// resizes images to what the tower's scratch can actually hold.
    pub fn set_vision_max_patches(&mut self, max_patches: usize) {
        self.vision_max_patches = max_patches;
    }

    pub fn set_video_dedup_threshold(&mut self, threshold: f64) {
        self.video_dedup_threshold = threshold;
    }

    pub fn enable_speculation(&mut self, dir: &str, k: usize) -> Result<bool> {
        if k == 0 {
            return Ok(false);
        }
        // Not `k + 1`. That is the verification feed's width, and it was the
        // wrong bound: the first round of a request primes the drafter over the
        // whole prompt, so a 62-token prompt walked 62 rows into a 3-row buffer.
        // The head chunks the feed now, and this is the chunk — wide enough that
        // an ordinary prompt is one pass, narrow enough that `scores` stays at
        // `heads * 64 * max_seq` rather than a prompt's worth of it.
        const PRIME_CHUNK: usize = 64;
        let rows = PRIME_CHUNK.max(k + 1);
        // Where the head lives depends on how the model was packaged. A
        // safetensors checkpoint keeps it among its own shards, so the directory
        // is the whole address. A GGUF keeps it in a second file — llama.cpp's
        // conversion writes `mtp-<model>.gguf`, a standalone 65-block model, and
        // leaves the text model's file ending at `blk.63` — so the path has to
        // be found rather than derived.
        let found = if std::path::Path::new(dir).is_dir() {
            self.model.load_mtp_head(dir, rows)?
        } else if let Some(side) = mtp_sidecar(dir) {
            tracing::info!(path = %side.display(), "MTP sidecar");
            self.model.load_mtp_head_gguf(&side, rows)?
        } else {
            false
        };
        if !found {
            tracing::info!("this checkpoint has no MTP head; speculation stays off");
            return Ok(false);
        }
        self.model.enable_speculation(k, &self.pool)?;
        self.spec_k = k;
        self.spec_k_max = k;
        self.spec_ema_accept = None;
        self.spec_k_cooldown = 0;
        tracing::info!(k, adaptive = self.adaptive_spec, "speculative decoding on");
        Ok(true)
    }

    /// Mean tokens emitted per verification pass, or `None` if none ran.
    ///
    /// The number the speedup is proportional to, and the one to distrust if it
    /// looks too good: a greedy request drafting against degenerate repetition
    /// would report a high acceptance while producing text nobody wants.
    pub fn acceptance_length(&self) -> Option<f64> {
        (self.spec_steps > 0).then(|| self.spec_tokens as f64 / self.spec_steps as f64)
    }

    pub fn enqueue(&mut self, req: Request) {
        self.waiting.push_back(req);
    }

    pub fn is_idle(&self) -> bool {
        self.running.is_empty() && self.waiting.is_empty()
    }

    pub fn in_flight(&self) -> usize {
        self.running.len() + self.waiting.len()
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// (lookups, hits, tokens served from cache). `None` on a model that
    /// carries recurrent state, where prefix caching is off — see
    /// `crate::prefix`'s module note.
    pub fn prefix_cache_stats(&self) -> Option<(u64, u64, u64)> {
        self.prefix.as_ref().map(crate::prefix::PrefixCache::stats)
    }

    pub fn pool(&self) -> &KvPool {
        &self.pool
    }

    /// Where a step's wall time went, under `INFERO_PROFILE` or
    /// `INFERO_STEP_TIMING`.
    ///
    /// The GPU profile answers "which kernel", which stops being the question
    /// once the server is slower than the model. It is not, any more: at a
    /// batch of 32 the host side is 0.18 ms of a 7.36 ms step.
    pub fn timing(&self) -> (f64, f64, f64, u64) {
        (self.t_issue, self.t_sample, self.t_advance, self.steps)
    }

    /// Admit what fits, run one batched forward, and deliver its output.
    /// One speculative round, or `Ok(false)` if this step is not one.
    ///
    /// Deliberately narrow. It runs only when there is exactly one running
    /// sequence, past its prompt, with a sampling (not greedy) request, a
    /// drafter feed from the previous round, and `k + 1` free pool slots.
    /// Everything else falls through to the ordinary step.
    ///
    /// Single-sequence because `verify_draft_sampled` is: batched speculation
    /// needs the recurrent working copy and the journal indexed per slot, which
    /// is a separate piece of work. Not greedy because the acceptance rule here
    /// is a probability ratio — a greedy request has no distribution to take a
    /// ratio of, and `Sampler::distribution` refuses rather than inventing one.
    fn speculative_step(&mut self) -> Result<bool> {
        if self.spec_k == 0 || self.running.len() != 1 {
            return Ok(false);
        }
        let idx = 0usize;
        let k = self.spec_k;
        // Each skip is named, because "speculation is on" and "speculation is
        // running" are different claims and only the second one is worth
        // anything. A round that never fires looks exactly like a round that
        // fires and accepts nothing.
        let skip = |why: &'static str| -> Result<bool> {
            tracing::debug!(why, "speculative step skipped");
            Ok(false)
        };
        {
            let r = &self.running[idx];
            if !r.prompt_complete() {
                return skip("still prefilling");
            }
            if r.next.is_none() {
                return skip("no pending token");
            }
            if r.sampler.params().is_greedy() {
                return skip("greedy request");
            }
            if r.spec_feed.is_none() {
                return skip("no drafter feed from the last pass");
            }
            if r.spec_desynced {
                return skip("drafter desynced earlier in this sequence");
            }
        }
        // The verification pass appends `k + 1` tokens before rolling back to
        // the accepted prefix, so the slots have to be there for the whole pass.
        if self.pool.free_slots() < k + 1 {
            return skip("pool has no free slots");
        }
        if self.pool.headroom(self.running[idx].seq) < k + 1 {
            return skip("sequence is at its context limit");
        }

        let seq = self.running[idx].seq;
        let pending = self.running[idx].next.expect("checked above");
        let feed = self.running[idx].spec_feed.take().expect("checked above");

        // The drafter keeps a cache of its own, and it only advances on the
        // rounds that run. Any step that skips speculation — a second sequence
        // arriving is the ordinary one — leaves it behind the sequence, and it
        // cannot catch up: `mtp_hidden` holds the rows of the *last* pass, so the
        // hidden states for the gap are gone by the time anyone notices.
        //
        // Priming across the gap anyway would be reading slots nobody wrote.
        // That is not a correctness question — verification uses the target
        // model's distribution, so a bad draft only lowers the acceptance rate —
        // but it is a cost with no upside, so the sequence stops speculating
        // instead. Two concurrent requests used to reach this as a 500 from
        // `MtpHead::run`'s own guard, which is where the gap was first seen.
        let cached = self.model.mtp_head().map_or(0, |h| h.cached());
        // An empty feed would slip past the comparison below and then fail
        // inside `prime`, which is a 500 for a condition the scheduler can see.
        let Some(first) = feed.positions.first().copied() else {
            return skip("empty drafter feed");
        };
        if first > cached {
            if !self.running[idx].spec_desynced {
                self.running[idx].spec_desynced = true;
                tracing::info!(
                    seq = seq.0,
                    position = first,
                    cached,
                    "the drafter fell behind this sequence; it will decode \
                     without speculation from here"
                );
            }
            return Ok(false);
        }
        // The window the repetition penalty reads, which both sides have to
        // score against — see `verify_draft_sampled`.
        let history: Vec<u32> = {
            let r = &self.running[idx];
            let mut h = r.prompt.clone();
            h.extend_from_slice(&r.generated);
            r.sampler.window(&h).to_vec()
        };

        tracing::debug!(
            first = feed.positions.first().copied(),
            rows = feed.rows.len(),
            cached = self.model.mtp_head().map(|h| h.cached()),
            "speculative round"
        );
        // A round's three parts, timed separately under `INFERO_STEP_TIMING`.
        // The whole is measurable end to end and the verification pass is
        // measurable on its own, so what this adds is the *difference* — the
        // drafting and the bookkeeping — which at k=3 was 9.7 ms of a 50.8 ms
        // round and had no attribution at all.
        let t0 = self.profile.then(std::time::Instant::now);
        let draft = {
            let r = &mut self.running[idx];
            self.model
                .draft_with_head_sampled(k, &feed, &mut r.sampler, &history)?
        };
        let t1 = self.profile.then(std::time::Instant::now);
        let outcome = {
            let r = &mut self.running[idx];
            self.model.verify_draft_sampled(
                seq,
                &mut self.pool,
                pending,
                &draft,
                &mut r.sampler,
                &history,
                r.mrope_delta,
            )?
        };

        let t2 = self.profile.then(std::time::Instant::now);
        self.steps += 1;
        self.spec_steps += 1;
        self.spec_tokens += outcome.tokens.len() as u64;
        self.running[idx].spec_rounds += 1;
        self.running[idx].spec_emitted += outcome.tokens.len() as u64;

        // `INFERO_ADAPTIVE_SPEC`: fold this round's yield into the running
        // average and let it move `spec_k` toward whatever the average says a
        // draft this deep is worth — up to `spec_k_max`, never past it, since
        // that is what `enable_speculation` sized the GDN rollback journal
        // for and going higher would need a reallocation this is not one.
        // Going lower needs nothing: the journal already holds `k_max + 1`
        // rows and a smaller round just uses fewer of them.
        //
        // The two thresholds are a band, not a line, on purpose — a value
        // that decided every round by comparing the same average to the same
        // single cutoff would flip `spec_k` back and forth across it as the
        // average drifted a few hundredths either side.
        if self.adaptive_spec {
            let yielded = outcome.tokens.len() as f64;
            const ALPHA: f64 = 0.15;
            self.spec_ema_accept = Some(match self.spec_ema_accept {
                Some(prev) => prev * (1.0 - ALPHA) + yielded * ALPHA,
                None => yielded,
            });
            let ema = self.spec_ema_accept.expect("just set above");
            self.spec_k_cooldown = self.spec_k_cooldown.saturating_sub(1);
            // Rounds, not seconds: a round is the unit both the EMA and the
            // thresholds are already in, so this stays the same number of
            // *decisions* apart whatever the round rate a request happens to
            // run at. 20 measured as 0.7 s at this model's round rate — under
            // one exchange's length, so it still flipped inside a single
            // short reply. 80 is closer to a short reply's whole round count,
            // which is the granularity a change should default to.
            const COOLDOWN: u64 = 80;
            if self.spec_k_cooldown == 0 {
                let k = self.spec_k as f64;
                if ema > k * 0.85 + 0.15 && self.spec_k < self.spec_k_max {
                    self.spec_k += 1;
                    self.spec_k_cooldown = COOLDOWN;
                    tracing::debug!(k = self.spec_k, ema, "adaptive spec: deeper draft");
                } else if ema < k * 0.4 && self.spec_k > 1 {
                    self.spec_k -= 1;
                    self.spec_k_cooldown = COOLDOWN;
                    tracing::debug!(k = self.spec_k, ema, "adaptive spec: shallower draft");
                }
            }
        }

        // Every emitted token goes through the same bookkeeping a plain step's
        // single token does, in order, and the first one that ends the sequence
        // stops the rest — a stop sequence found at token two must not be
        // overrun by token three.
        let mut finished = false;
        for &t in &outcome.tokens {
            if self.advance_token(idx, t)? {
                finished = true;
                break;
            }
        }
        if let (Some(t0), Some(t1), Some(t2)) = (t0, t1, t2) {
            let t3 = std::time::Instant::now();
            self.spec_draft_ms += (t1 - t0).as_secs_f64() * 1e3;
            self.spec_verify_ms += (t2 - t1).as_secs_f64() * 1e3;
            self.spec_after_ms += (t3 - t2).as_secs_f64() * 1e3;
            self.spec_window += 1;
            if self.spec_window >= 100 {
                let w = self.spec_window as f64;
                tracing::warn!(
                    draft_ms = format!("{:.2}", self.spec_draft_ms / w),
                    verify_ms = format!("{:.2}", self.spec_verify_ms / w),
                    after_ms = format!("{:.2}", self.spec_after_ms / w),
                    k = self.spec_k,
                    accept = format!("{:.2}", self.spec_tokens as f64 / self.spec_steps as f64),
                    "per-round timing"
                );
                // The per-kernel table, which the device layer has been
                // accumulating all along and nothing was printing. Beside the
                // draft/verify split because the two are one question: that
                // split says *which half* the round is spent in, and this says
                // which kernel inside it. Reset with the window so the next
                // hundred rounds are read on their own.
                let report = self.model.device().profile().report();
                if !report.is_empty() {
                    tracing::warn!("per-kernel, last {} rounds:\n{report}", self.spec_window);
                }
                self.model.device().profile().reset();
                self.spec_draft_ms = 0.0;
                self.spec_verify_ms = 0.0;
                self.spec_after_ms = 0.0;
                self.spec_window = 0;
            }
        }
        if finished {
            let r = self.running.swap_remove(idx);
            self.retire(r);
            return Ok(true);
        }
        self.running[idx].spec_feed = Some(outcome.feed);
        Ok(true)
    }

    pub fn step(&mut self) -> Result<()> {
        let step_start = self.profile.then(std::time::Instant::now);
        self.admit();
        self.drop_disconnected();
        if self.running.is_empty() {
            return Ok(());
        }
        if self.speculative_step()? {
            return Ok(());
        }

        let plan = self.plan();
        if plan.is_empty() {
            // Every running sequence is blocked on pool capacity; releasing a
            // finished one is the only thing that can unblock them, and that
            // happens below.
            anyhow::bail!("no sequence could make progress: kv pool is exhausted");
        }

        let items: Vec<BatchItem<'_>> = plan
            .iter()
            .map(|(idx, work)| {
                let r = &self.running[*idx];
                match work {
                    Work::Prefill { from, len, last } => BatchItem {
                        seq: r.seq,
                        tokens: &r.prompt[*from..*from + *len],
                        wants_logits: *last,
                        vision: r.vision.as_ref(),
                        // How many of `vision`'s rows earlier chunks of this
                        // sequence already consumed -- a *count of actual pad
                        // tokens* in `prompt[..from]`, not an index
                        // subtraction (`Running::vision_pad_tok`'s doc
                        // comment explains why the shortcut is wrong for a
                        // multi-frame-group video). `0` for a text-only
                        // sequence, where no token ever equals
                        // `r.vision_pad_tok` and this is never read anyway.
                        vision_row_offset: if r.vision.is_some() {
                            r.prompt[..*from].iter().filter(|&&t| t == r.vision_pad_tok).count()
                        } else {
                            0
                        },
                        // Sliced absolutely, so a prefix-cache hit (`from >
                        // 0`) still lands on the right rows of `r.mrope` --
                        // see `Running::mrope`'s doc comment. `r.mrope` is
                        // `None` for a text-only sequence or a model without
                        // M-RoPE, in which case `mrope_delta: 0` reproduces
                        // `pool.len(seq) + k` on all three axes.
                        mrope: r.mrope.as_deref().map(|m| &m[3 * from..3 * (from + len)]),
                        mrope_delta: r.mrope_delta,
                    },
                    Work::Decode => BatchItem {
                        seq: r.seq,
                        // Safe: `next` is set before a sequence can be
                        // scheduled for decode.
                        tokens: std::slice::from_ref(r.next.as_ref().unwrap()),
                        wants_logits: true,
                        vision: None,
                        vision_row_offset: 0,
                        // A single generated token is never mid-image, so its
                        // three axes are always `pool.len(seq) + mrope_delta`
                        // -- no per-token array needed here even when `r.mrope`
                        // is `Some` (it was only ever the prompt's array).
                        mrope: None,
                        mrope_delta: r.mrope_delta,
                    },
                }
            })
            .collect();

        let vocab = self.model.config().vocab_size;
        let timed = self.profile;
        let t0 = timed.then(std::time::Instant::now);
        self.model.forward_batch_device(&items, &mut self.pool)?;
        let t1 = timed.then(std::time::Instant::now);

        // Drop a sequence's vision features once every pad token they
        // replace has been consumed by some chunk -- again a *count* of
        // `vision_pad_tok` occurrences in `prompt[..from+len]`, not
        // `from + len` compared against a remembered start index, for the
        // same reason `vision_row_offset` above cannot be either. A
        // placeholder run split across several chunks keeps its features
        // alive until the last of them; the common case (the whole run in one
        // chunk) drops it immediately, same as before. Dropping it here
        // rather than leaving it on `Running` until the sequence retires is
        // the difference between one splice's scratch and a device
        // allocation held for the rest of a long conversation.
        for (idx, work) in &plan {
            if let Work::Prefill { from, len, .. } = work {
                let r = &mut self.running[*idx];
                if let Some(f) = &r.vision {
                    let consumed =
                        r.prompt[..from + len].iter().filter(|&&t| t == r.vision_pad_tok).count();
                    if consumed >= f.tokens {
                        r.vision = None;
                    }
                }
            }
        }

        // Sample where the logits already are.
        //
        // The alternative — and what this did until the device sampler existed
        // — is to copy every row's whole vocabulary back first: 16 MiB at a
        // batch of 32, measured at 2.19 ms of a 12.18 ms step. `t_sample` below
        // covers this call, and since the forward pass no longer ends in a
        // synchronise it is also where the step waits on the GPU.
        let rows: Vec<usize> = plan
            .iter()
            .filter(|(_, w)| match w {
                Work::Prefill { last, .. } => *last,
                Work::Decode => true,
            })
            .map(|(idx, _)| *idx)
            .collect();
        // The draw comes from each sequence's own generator, so seeding and
        // reproducibility survive the move; pulling it needs `running` mutably,
        // which has to finish before the windows borrow it.
        let draws: Vec<f64> = rows
            .iter()
            .map(|i| self.running[*i].sampler.next_draw())
            .collect();
        let specs: Vec<infero_model::RowSample> = rows
            .iter()
            .zip(&draws)
            .map(|(i, d)| {
                let p = self.running[*i].sampler.params();
                infero_model::RowSample {
                    temperature: p.temperature,
                    top_p: p.top_p,
                    top_k: p.top_k as u32,
                    rep_penalty: p.repetition_penalty,
                    rnd: *d,
                }
            })
            .collect();
        let windows: Vec<&[u32]> = rows
            .iter()
            .map(|i| {
                let r = &self.running[*i];
                r.sampler.window(&r.generated)
            })
            .collect();
        let sampled = self.model.sample_on_device(&specs, &windows)?;
        let t2 = timed.then(std::time::Instant::now);

        match sampled {
            Some(ids) => {
                for (i, id) in rows.iter().zip(ids) {
                    self.running[*i].sampled = id;
                }
            }
            None => {
                // Whatever the device sampler will not take — a `top_k` past
                // its bound, or a vocabulary too large for the bitset — still
                // has to produce a token, and it does so exactly as before.
                let logits = self.model.logits_host()?.to_vec();
                self.sample_rows_host(&plan, &logits, vocab, &rows, &draws);
            }
        }
        self.steps += 1;

        // The sampled token already sits on each `Running`; this pass is the
        // bookkeeping around it. It no longer counts logit rows — those are
        // matched to sequences by `rows` above, where the sampling happens.
        let mut finished = Vec::new();
        for (idx, work) in &plan {
            let wants = match work {
                Work::Prefill { last, .. } => *last,
                Work::Decode => true,
            };
            if let Work::Prefill { len, .. } = work {
                self.running[*idx].prefilled += len;
            }
            if !wants {
                continue;
            }
            let ended = self.advance(*idx)?;
            if ended {
                finished.push(*idx);
            } else if self.spec_k > 0 {
                // Hand the drafter what this pass actually covered.
                //
                // Not `DraftFeed::after_prefill`, which assumes the whole prompt
                // went through in one pass: `mtp_hidden` holds only the rows of
                // the *last* pass, so a chunked prefill would point the drafter
                // at hidden states that are not there. The general form is the
                // rows this pass ran, and a decode step is the one-row case.
                let r = &self.running[*idx];
                let pending = r.sampled;
                let (from, len) = match work {
                    Work::Prefill { from, len, .. } => (*from, *len),
                    // `prefilled` and `generated` already include this step's
                    // token, so the row just run sits one before the end.
                    Work::Decode => (r.prompt.len() + r.generated.len() - 2, 1),
                };
                // `shifted[i]` is the embedding that pairs with hidden row `i`:
                // the token *after* the one at that position. For every row but
                // the last that is the next token of the sequence; for the last
                // it is the token this step just sampled.
                let seq_tokens: Vec<u32> = r
                    .prompt
                    .iter()
                    .chain(r.generated.iter())
                    .copied()
                    .collect();
                let mut shifted: Vec<u32> = Vec::with_capacity(len);
                for p in from..from + len {
                    shifted.push(if p + 1 < seq_tokens.len() {
                        seq_tokens[p + 1]
                    } else {
                        pending
                    });
                }
                // Only a prefill's rows can be mid-image; a decode row's
                // `from` indexes past `r.prompt.len()` into `r.generated`,
                // which `r.mrope` (sized to the prompt alone) does not cover
                // and does not need to -- see `MtpHead::run`'s doc comment.
                let mrope = match work {
                    Work::Prefill { .. } => {
                        r.mrope.as_deref().map(|m| m[3 * from..3 * (from + len)].to_vec())
                    }
                    Work::Decode => None,
                };
                self.running[*idx].spec_feed = Some(infero_model::spec::DraftFeed {
                    rows: 0..len,
                    positions: (from..from + len).collect(),
                    shifted,
                    mrope,
                });
            }
        }

        // Decode steps only. A prefill step costs about ten times a decode,
        // and at 32 clients the run's own 32 prefills land inside the first
        // window and drag its mean from 11 ms to 18 — which reads as six
        // milliseconds of forward-pass overhead that does not exist. The
        // window was meant to keep prefills out of a lifetime average; it has
        // to keep them out of the window too.
        let decode_only = plan.iter().all(|(_, w)| matches!(w, Work::Decode));
        if let (Some(t0), Some(t1), Some(t2)) = (t0, t1, t2) {
            let t3 = std::time::Instant::now();
            // `last_end` advances on every step, prefill included, or the next
            // decode's gap would swallow the prefill step whole and report it
            // as time the engine spent doing nothing.
            let prev = self.last_end.replace(t3);
            if decode_only {
                self.t_issue += (t1 - t0).as_secs_f64() * 1e3;
                self.t_sample += (t2 - t1).as_secs_f64() * 1e3;
                self.t_advance += (t3 - t2).as_secs_f64() * 1e3;
                if let (Some(s0), Some(prev)) = (step_start, prev) {
                    self.t_step += (t3 - s0).as_secs_f64() * 1e3;
                    self.t_gap += (s0 - prev).as_secs_f64() * 1e3;
                }
                self.window += 1;
                // A window rather than a lifetime average: the run's own prefill
                // steps cost ten times a decode and would sit in the mean forever.
                if self.window >= 200 {
                    let w = self.window as f64;
                    tracing::warn!(
                        issue_ms = format!("{:.2}", self.t_issue / w),
                        sample_ms = format!("{:.2}", self.t_sample / w),
                        advance_ms = format!("{:.2}", self.t_advance / w),
                        step_ms = format!("{:.2}", self.t_step / w),
                        gap_ms = format!("{:.2}", self.t_gap / w),
                        batch = plan.len(),
                        "per-step timing"
                    );
                    // The same per-kernel table the speculative window prints,
                    // so the two are comparable. That comparison is the whole
                    // point: a verification pass costs 1.71x a one-row step
                    // where a *batched* two-row step costs 1.40x, and 21 ms of
                    // that gap is not accounted for by anything countable --
                    // the extra GDN launches are 0.8 ms of state traffic, the
                    // rollback journal is 13 MiB. Whatever it is, it is a kernel
                    // that behaves differently at two rows of one sequence than
                    // at one row, and only this table can name it.
                    let report = self.model.device().profile().report();
                    if !report.is_empty() {
                        tracing::warn!("per-kernel, last {} steps:\n{report}", self.window);
                    }
                    self.model.device().profile().reset();
                    self.t_issue = 0.0;
                    self.t_sample = 0.0;
                    self.t_advance = 0.0;
                    self.t_step = 0.0;
                    self.t_gap = 0.0;
                    self.window = 0;
                }
            }
        }

        // Retire from the back so the earlier indices stay valid.
        finished.sort_unstable();
        for idx in finished.into_iter().rev() {
            let r = self.running.swap_remove(idx);
            self.retire(r);
        }
        Ok(())
    }

    /// Move waiting requests into the running set while there is room.
    /// Turn one pending image into `VisionFeatures`, expanding the prompt's
    /// single placeholder token into the run the tower's grid actually needs.
    ///
    /// The template renders exactly one `<|image_pad|>` per image — it has no
    /// way to know the real count before the tower has seen the picture — so
    /// this is the point that count becomes real, mirroring what Hugging
    /// Face's own processor does between tokenizing and calling the model.
    /// Runs on this thread because both steps need `&mut self.model`: sizing
    /// the resize target and running the tower itself.
    fn encode_pending_image(&mut self, prompt: &mut Vec<u32>, img: PendingImage) -> Result<(VisionFeatures, u32)> {
        anyhow::ensure!(
            self.model.has_vision(),
            "this request has an image but the loaded model has no vision tower"
        );
        let (img_tok, _) = self.model.vision_tokens().expect("has_vision checked above");
        let mut hits = prompt.iter().enumerate().filter(|&(_, &t)| t == img_tok);
        let Some((at, _)) = hits.next() else {
            anyhow::bail!(
                "this request carries an image but the rendered prompt has no \
                 image placeholder; the chat template may not support vision \
                 the way this server expects"
            );
        };
        anyhow::ensure!(
            hits.next().is_none(),
            "the rendered prompt has more than one image placeholder for a \
             single image; this should not happen and points at a template \
             mismatch"
        );

        let shape = *self.model.vision_shape().expect("has_vision checked above");
        let (th, tw, tokens) = self
            .model
            .vision_resize(img.height, img.width, self.vision_max_patches)
            .context("sizing the image for the vision tower")?;
        let frame = infero_model::qwen35_vision_image::prepare_frame(
            &img.rgb, img.height, img.width, 3, th, tw, shape.patch, shape.merge,
        );
        let feats = self
            .model
            .encode_image(&frame)
            .context("running the vision tower")?;
        anyhow::ensure!(
            feats.tokens == tokens,
            "the tower produced {} tokens where the resize said {tokens}; a \
             loader/config mismatch, not a request problem",
            feats.tokens
        );
        // The splice can now span several prefill steps -- `plan` chunks a
        // vision-carrying sequence's placeholder run exactly the way it
        // already chunks plain text, splitting on `batch_tokens` step
        // boundaries rather than requiring the whole thing in one step (see
        // `BatchItem::vision_row_offset`). What still has to fit is the
        // *sequence's* context, same as any other prompt -- `admit`'s own
        // `need > self.pool.max_seq()` check already covers that generically,
        // so this is a pre-check to avoid running the tower at all on a
        // request that check would refuse anyway, not the real gate.
        let max_seq = self.pool.max_seq();
        let expanded_len = prompt.len() - 1 + tokens;
        anyhow::ensure!(
            expanded_len <= max_seq,
            "this image resizes to {tokens} language-model tokens ({th}x{tw} \
             at patch {}), and the rest of the prompt is {} more — {expanded_len} \
             total, over the {max_seq} tokens this model's context can hold",
            shape.patch,
            prompt.len() - 1,
        );
        prompt.splice(at..at + 1, std::iter::repeat_n(img_tok, tokens));
        Ok((feats, img_tok))
    }

    /// [`Self::encode_pending_image`] for a decoded clip.
    ///
    /// The one real structural difference: the template renders a single
    /// `<|vision_start|><|video_pad|><|vision_end|>` triple (same shape as an
    /// image), but the *reference processor* only ever substitutes the inner
    /// `video_token` -- see `processing_qwen3_vl.py`'s `replace_video_token`,
    /// confirmed by reading the installed transformers on bw -- leaving the
    /// template's outer `vision_start`/`vision_end` in place and expanding the
    /// single pad token into `grid_t` *separate* `<T.T seconds><|vision_start|>
    /// pad×n<|vision_end|>` groups, one a temporal-patch pair. So this only
    /// splices `at..at+1` (the lone pad token), not `at-1..at+2`: the outer
    /// wrap is the template's, not this function's, to touch.
    fn encode_pending_video(&mut self, prompt: &mut Vec<u32>, mut vid: PendingVideo) -> Result<(VisionFeatures, u32)> {
        anyhow::ensure!(
            self.model.has_vision(),
            "this request has a video but the loaded model has no vision tower"
        );
        let (_, vid_tok) = self.model.vision_tokens().expect("has_vision checked above");
        let vision_start = self
            .tokenizer
            .token_to_id("<|vision_start|>")
            .context("this checkpoint's tokenizer has no <|vision_start|> token")?;
        let vision_end = self
            .tokenizer
            .token_to_id("<|vision_end|>")
            .context("this checkpoint's tokenizer has no <|vision_end|> token")?;

        let mut hits = prompt.iter().enumerate().filter(|&(_, &t)| t == vid_tok);
        let Some((at, _)) = hits.next() else {
            anyhow::bail!(
                "this request carries a video but the rendered prompt has no \
                 video placeholder; the chat template may not support video \
                 the way this server expects"
            );
        };
        anyhow::ensure!(
            hits.next().is_none(),
            "the rendered prompt has more than one video placeholder for a \
             single video; this should not happen and points at a template \
             mismatch"
        );
        anyhow::ensure!(
            at > 0 && at + 1 < prompt.len() && prompt[at - 1] == vision_start && prompt[at + 1] == vision_end,
            "the video placeholder is not wrapped in <|vision_start|>/<|vision_end|> \
             the way this checkpoint's template is expected to render it"
        );

        if self.video_dedup_threshold > 0.0 {
            let (frames, timestamps) =
                drop_redundant_groups(&vid.frames, &vid.timestamps, self.video_dedup_threshold);
            tracing::debug!(
                kept = timestamps.len(),
                total = vid.timestamps.len(),
                "video dedup dropped near-duplicate frame-groups"
            );
            vid.frames = frames;
            vid.timestamps = timestamps;
        }

        let shape = *self.model.vision_shape().expect("has_vision checked above");
        let max_seq = self.pool.max_seq();
        let rest = prompt.len() - 1;

        // The splice can now span several prefill steps, so what has to fit
        // is the *sequence's* context (`admit`'s generic `need > max_seq`
        // check), not one step's `batch_tokens` -- see `encode_pending_image`'s
        // matching comment. The shrink-retry loop is still worth keeping: it
        // bounds how much of that context one video request can claim before
        // its own resolution gives, rather than spending the whole budget on
        // one clip at maximum resolution by default. `grid_t` (fixed by the
        // frame count, not the patch budget -- see
        // `qwen35_vision::video_resize_budget`) stays the same across every
        // attempt, so the timestamp-count check only has to run once.
        let mut patch_budget = self.vision_max_patches;
        let mut attempt = None;
        for _ in 0..4 {
            let (th, tw, grid_t, tokens) = self
                .model
                .vision_resize_video(vid.frames.len(), vid.height, vid.width, patch_budget)
                .context("sizing the video for the vision tower")?;
            anyhow::ensure!(
                grid_t == vid.timestamps.len(),
                "the decoder sampled {} frame-groups but produced {} timestamps; a \
                 decode/resize mismatch, not a request problem",
                grid_t,
                vid.timestamps.len()
            );
            let frame_seqlen = tokens / grid_t;
            let mut replacement = Vec::with_capacity(grid_t * (frame_seqlen + 3));
            for i in 0..grid_t {
                let ts = format!("<{:.1} seconds>", vid.timestamps[i]);
                replacement.extend(self.tokenizer.encode(&ts, Some(false), true));
                replacement.push(vision_start);
                replacement.extend(std::iter::repeat_n(vid_tok, frame_seqlen));
                replacement.push(vision_end);
            }
            let expanded_len = rest + replacement.len();
            let fits = expanded_len <= max_seq;
            attempt = Some((th, tw, grid_t, tokens, replacement, expanded_len));
            if fits {
                break;
            }
            patch_budget /= 2;
        }
        let (th, tw, grid_t, tokens, replacement, expanded_len) = attempt.expect("looped at least once");
        anyhow::ensure!(
            expanded_len <= max_seq,
            "this video resizes to {tokens} vision tokens across {grid_t} \
             frame-groups ({th}x{tw} at patch {}) even after repeatedly \
             shrinking the resolution, plus timestamp text, and the rest of \
             the prompt is {rest} more -- {expanded_len} total, over the \
             {max_seq} tokens this model's context can hold. Send fewer frames",
            shape.patch,
        );

        let raw_frames: Vec<&[u8]> = vid.frames.iter().map(Vec::as_slice).collect();
        let clip = infero_model::qwen35_vision_image::prepare_clip(
            &raw_frames, vid.height, vid.width, 3, th, tw, shape.patch, shape.merge,
        );
        let feats = self.model.encode_clip(&clip).context("running the vision tower")?;
        anyhow::ensure!(
            feats.tokens == tokens,
            "the tower produced {} tokens where the resize said {tokens}; a \
             loader/config mismatch, not a request problem",
            feats.tokens
        );
        prompt.splice(at..at + 1, replacement);
        Ok((feats, vid_tok))
    }

    fn admit(&mut self) {
        while !self.waiting.is_empty() {
            if self.pool.active_sequences() >= self.pool.max_seqs() {
                break;
            }
            // Off the queue for everything below, including the one thing
            // that needs `&mut self.model` — encoding an image, which cannot
            // happen while the request is still borrowed from `waiting`. One
            // that turns out not to fit yet goes back on the front, in the
            // same place it held before; see the `push_front` calls below.
            let mut req = self.waiting.pop_front().unwrap();

            // `routes.rs` already refuses a request carrying both an image
            // and a video before it ever reaches the queue; the `(Some,
            // Some)` arm here is a defensive refusal, not a real path.
            let (vision, vision_pad_tok, is_video) = match (req.pending_image.take(), req.pending_video.take()) {
                (Some(img), None) => match self.encode_pending_image(&mut req.prompt, img) {
                    Ok((feats, tok)) => (Some(feats), tok, false),
                    Err(e) => {
                        let _ = req.events.send(Event::Failed(format!("{e:#}")));
                        continue;
                    }
                },
                (None, Some(vid)) => match self.encode_pending_video(&mut req.prompt, vid) {
                    Ok((feats, tok)) => (Some(feats), tok, true),
                    Err(e) => {
                        let _ = req.events.send(Event::Failed(format!("{e:#}")));
                        continue;
                    }
                },
                (None, None) => (None, 0, false),
                (Some(_), Some(_)) => {
                    let _ = req.events.send(Event::Failed(
                        "this request has both an image and a video pending; refused".into(),
                    ));
                    continue;
                }
            };
            let had_vision = vision.is_some();
            // Computed from the *expanded* prompt (the splice above already
            // ran), while the grid is still attached to `vision` -- `Running`
            // does not hold onto `VisionFeatures` past its one prefill, so
            // this is the only point this data is available together.
            let (mrope, mrope_delta) = match (&vision, self.model.config().mrope_section) {
                (Some(feats), Some(_)) => {
                    let merge = self.model.vision_shape().expect("has_vision").merge;
                    let (img_tok, vid_tok) = self.model.vision_tokens().expect("has_vision");
                    let (m, d) = if is_video {
                        mrope_for_video(&req.prompt, vid_tok, feats, merge)
                    } else {
                        mrope_for_image(&req.prompt, img_tok, feats, merge)
                    };
                    (Some(m), d)
                }
                _ => (None, 0),
            };

            // Admit only with room for the whole prompt plus something to
            // generate; a sequence admitted into a pool it cannot finish in
            // would deadlock against itself.
            let need = req.prompt.len() + 1;
            // Cached blocks hold slots a waiting request could use, and a cold
            // block is worth less than an admission. Evict before deciding the
            // prompt does not fit; only unreferenced blocks go.
            if need > self.pool.free_slots()
                && let Some(c) = self.prefix.as_mut()
            {
                c.evict_for(&mut self.pool, need);
            }
            if need > self.pool.free_slots() || need > self.pool.max_seq() {
                if self.running.is_empty() {
                    // Nothing will ever free up: reject rather than spin.
                    let _ = req.events.send(Event::Failed(format!(
                        "prompt of {} tokens does not fit: {} slots free, {} per sequence",
                        req.prompt.len(),
                        self.pool.free_slots(),
                        self.pool.max_seq()
                    )));
                    continue;
                }
                self.waiting.push_front(req);
                break;
            }

            let Some(seq) = self.pool.alloc() else {
                self.waiting.push_front(req);
                break;
            };

            // Whatever of this prompt is already computed. Capped one token
            // short of the whole prompt: a sequence with nothing left to run
            // has no row to take logits from, and `plan` would give it no work
            // forever.
            //
            // Skipped entirely for a vision request: its placeholder run is
            // the same `image_token_id` regardless of which picture produced
            // it, so a hash over token ids alone cannot tell two images apart
            // — see `Running::had_vision`.
            let hit = match self.prefix.as_mut() {
                Some(c) if !had_vision => {
                    let p = c.lookup(&req.prompt, req.prompt.len().saturating_sub(1));
                    if !p.is_empty() {
                        match self.pool.attach_prefix(self.model.device(), seq, &p.slots) {
                            Ok(()) => p,
                            Err(e) => {
                                // Not fatal: recompute the prompt. The slots
                                // stay the cache's, so the references have to go
                                // back or eviction can never reclaim them.
                                tracing::warn!(error = %e, "attaching a cached prefix failed");
                                c.release(&p);
                                crate::prefix::Prefix::default()
                            }
                        }
                    } else {
                        p
                    }
                }
                _ => crate::prefix::Prefix::default(),
            };
            if !hit.is_empty() {
                tracing::debug!(
                    seq = seq.0,
                    reused = hit.len(),
                    prompt = req.prompt.len(),
                    "prefix cache hit"
                );
            }
            let budget = req
                .max_tokens
                .min(self.pool.max_seq().saturating_sub(req.prompt.len()));
            tracing::debug!(
                seq = seq.0,
                prompt = req.prompt.len(),
                running = self.running.len() + 1,
                "admitted"
            );
            self.running.push(Running {
                seq,
                prompt: req.prompt,
                prefilled: hit.len(),
                prefix: hit,
                vision,
                vision_pad_tok,
                had_vision,
                mrope,
                mrope_delta,
                next: None,
                sampled: 0,
                spec_feed: None,
                spec_desynced: false,
                spec_rounds: 0,
                spec_emitted: 0,
                generated: Vec::new(),
                sampler: Sampler::new(req.params),
                partial: Vec::new(),
                pending: String::new(),
                stop: req.stop,
                budget,
                events: req.events,
                admitted: Instant::now(),
                prefill_done: None,
            });
        }
    }

    /// Drop sequences whose client has gone away, freeing their slots at once.
    fn drop_disconnected(&mut self) {
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].events.is_closed() {
                let r = self.running.swap_remove(i);
                tracing::debug!(seq = r.seq.0, "client disconnected");
                self.retire(r);
            } else {
                i += 1;
            }
        }
    }

    /// Decide who gets tokens this step.
    fn plan(&self) -> Vec<(usize, Work)> {
        let mut plan = Vec::with_capacity(self.running.len());
        let mut budget = self.model.batch_tokens();

        // Decodes first: one token each, and a running sequence starved by a
        // prefill is a stall the client feels.
        for (i, r) in self.running.iter().enumerate() {
            if budget == 0 {
                break;
            }
            if r.prompt_complete() && r.next.is_some() {
                plan.push((i, Work::Decode));
                budget -= 1;
            }
        }

        // Whatever is left goes to the other prompts, split across steps if
        // need be -- a vision-carrying sequence included: its placeholder run
        // is no longer required to land in one step (`BatchItem
        // ::vision_row_offset` is what a chunk boundary landing inside it
        // needs), so it is just an ordinary prompt here, same as plain text.
        for (i, r) in self.running.iter().enumerate() {
            if budget == 0 {
                break;
            }
            if r.prompt_complete() {
                continue;
            }
            let remaining = r.prompt.len() - r.prefilled;
            let len = remaining.min(budget);
            plan.push((
                i,
                Work::Prefill {
                    from: r.prefilled,
                    len,
                    last: len == remaining,
                },
            ));
            budget -= len;
        }
        plan
    }

    /// Sample, emit, and report whether the sequence is done.
    fn advance(&mut self, idx: usize) -> Result<bool> {
        let next = self.running[idx].sampled;
        self.advance_token(idx, next)
    }

    /// One token's worth of bookkeeping: end-of-generation, stop sequences,
    /// streaming, budget, pool headroom.
    ///
    /// Split out from `advance` because a speculative step emits between one and
    /// `k + 1` tokens and every one of them has to pass through the same checks.
    /// A speculative path that scanned for stop sequences only on the last token
    /// of a step would run past a stop string by up to `k` tokens, which is a
    /// correctness difference the speedup does not justify.
    fn advance_token(&mut self, idx: usize, next: u32) -> Result<bool> {
        let tokenizer = self.tokenizer.clone();
        let r = &mut self.running[idx];
        if r.prefill_done.is_none() {
            r.prefill_done = Some(Instant::now());
        }
        if tokenizer.is_eog(next) {
            return Self::finish(r, &tokenizer, FinishReason::Stop);
        }

        r.generated.push(next);
        r.next = Some(next);

        let text = tokenizer.stream_push(next, &mut r.partial);
        r.pending.push_str(&text);
        let scan = split_at_stop(&r.pending, &r.stop);
        let emit: String = r.pending.drain(..scan.release_len()).collect();
        if !emit.is_empty() && r.events.send(Event::Text(emit)).is_err() {
            return Ok(true); // client hung up
        }
        if scan.is_hit() {
            r.pending.clear();
            return Self::finish(r, &tokenizer, FinishReason::Stop);
        }

        if r.generated.len() >= r.budget {
            return Self::finish(r, &tokenizer, FinishReason::Length);
        }
        // A sequence that has filled its share of the pool stops here rather
        // than failing the whole batch on the next extend.
        if self.pool.headroom(self.running[idx].seq) == 0 || self.pool.free_slots() == 0 {
            let r = &mut self.running[idx];
            return Self::finish(r, &tokenizer, FinishReason::Length);
        }
        Ok(false)
    }

    /// Fill `Running::sampled` for every row that asked for logits.
    ///
    /// Each sequence owns its sampler and its own history, so the work is
    /// disjoint; the only reason it was serial is that it lived inside the loop
    /// that also does the stop-sequence and streaming bookkeeping, which is
    /// not.
    /// The host fallback, for batches the device sampler declines.
    ///
    /// `draws` were already taken from each sequence's generator before the
    /// device path was tried, so this uses them rather than drawing again —
    /// otherwise falling back would advance the generators twice and a seeded
    /// run would depend on which path served it.
    fn sample_rows_host(
        &mut self,
        _plan: &[(usize, Work)],
        logits: &[f32],
        vocab: usize,
        wanted: &[usize],
        draws: &[f64],
    ) {
        if wanted.is_empty() {
            return;
        }

        // Row number for each running index, so the mutable borrows can be
        // collected in `running` order and still find their logits.
        let mut row_of = vec![usize::MAX; self.running.len()];
        for (row, idx) in wanted.iter().enumerate() {
            row_of[*idx] = row;
        }
        let mut refs: Vec<(usize, &mut Running)> = self
            .running
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| row_of[*i] != usize::MAX)
            .map(|(i, r)| (row_of[i], r))
            .collect();

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(refs.len());
        if threads <= 1 {
            for (row, r) in refs.iter_mut() {
                r.sampled = r.sampler.sample_with_draw(
                    &logits[*row * vocab..(*row + 1) * vocab],
                    &r.generated,
                    draws[*row],
                );
            }
            return;
        }
        let per = refs.len().div_ceil(threads);
        std::thread::scope(|scope| {
            for chunk in refs.chunks_mut(per) {
                scope.spawn(move || {
                    for (row, r) in chunk {
                        let lo = *row * vocab;
                        r.sampled = r.sampler.sample_with_draw(
                            &logits[lo..lo + vocab],
                            &r.generated,
                            draws[*row],
                        );
                    }
                });
            }
        });
    }

    /// Retire a running sequence: give back the prefix blocks it borrowed,
    /// offer the KV it just computed to the cache, and free what nobody kept.
    ///
    /// Not used by [`Self::fail_all`] — a failed forward pass leaves the pool
    /// in an unknown state, and caching out of it would serve a later request
    /// whatever went wrong here.
    fn retire(&mut self, r: Running) {
        if let Some(c) = self.prefix.as_mut() {
            c.release(&r.prefix);
            // Never offer a vision sequence's blocks to the cache: its
            // placeholder run hashes the same regardless of which image
            // produced it, so a later request with a different picture but the
            // same token count would hit this one's KV. `admit` applies the
            // matching half of this — it never looks the cache up for such a
            // request either.
            if !r.had_vision {
                let mut tokens = r.prompt;
                tokens.extend_from_slice(&r.generated);
                if let Err(e) = c.insert(&mut self.pool, r.seq, &tokens) {
                    tracing::warn!(error = %e, seq = r.seq.0, "prefix cache insert failed");
                }
            }
        }
        self.pool.free(r.seq);
    }

    fn finish(r: &mut Running, tokenizer: &Tokenizer, reason: FinishReason) -> Result<bool> {
        let mut tail = std::mem::take(&mut r.pending);
        tail.push_str(&tokenizer.stream_finish(&mut r.partial));
        if !tail.is_empty() {
            let _ = r.events.send(Event::Text(tail));
        }

        let queued = r.prefill_done.unwrap_or(r.admitted) - r.admitted;
        tracing::info!(
            seq = r.seq.0,
            prompt = r.prompt.len(),
            completion = r.generated.len(),
            queued_ms = queued.as_millis(),
            total_ms = r.admitted.elapsed().as_millis(),
            reason = reason.as_str(),
            // The mean acceptance length, or absent when this request never
            // speculated. `1.0` would mean the drafter ran and was rejected
            // every time, which is a different thing entirely and has to look
            // different in the log.
            accept_len = (r.spec_rounds > 0)
                .then(|| r.spec_emitted as f64 / r.spec_rounds as f64),
            spec_rounds = (r.spec_rounds > 0).then_some(r.spec_rounds),
            "request complete"
        );

        let _ = r.events.send(Event::Done {
            reason,
            prompt_tokens: r.prompt.len(),
            completion_tokens: r.generated.len(),
        });
        Ok(true)
    }

    /// Report an error to every sequence and clear the scheduler.
    ///
    /// A failed forward pass leaves the pool in an unknown state, so the whole
    /// batch goes rather than a guess at which sequence caused it.
    pub fn fail_all(&mut self, error: &str) {
        for r in self.running.drain(..) {
            let _ = r.events.send(Event::Failed(error.to_string()));
            self.pool.free(r.seq);
        }
        for req in self.waiting.drain(..) {
            let _ = req.events.send(Event::Failed(error.to_string()));
        }
    }

    pub fn model(&self) -> &Model {
        &self.model
    }
}

/// Build the pool a scheduler needs.
pub fn make_pool(model: &Model, max_seqs: usize, slots: Option<usize>) -> Result<KvPool> {
    let max_seq = model.max_seq();
    if let Some(n) = slots {
        return model
            .new_pool(n.max(max_seq), max_seqs)
            .context("allocating the kv pool");
    }

    // `max_seqs * max_seq` is what a pool would need for every sequence to
    // reach the context limit at once, and at 32 sequences of 4096 tokens on
    // an 8B model that is 17 GB — so a server asked for more concurrency than
    // its VRAM can back used to refuse to start, and the default concurrency
    // was set low enough to avoid it. That default cost a factor of two: the
    // load generator at 32 clients measured 368 tok/s against 725 with
    // `--max-seqs 32`, because eight was the batch the GEMM ever saw.
    //
    // Sequences rarely all run to the context limit, so oversubscribe: take
    // what the card has left after the weights, keep a margin for activations
    // and the CUDA context, and let the scheduler admit against the real slot
    // count. It already refuses a prompt that will not fit.
    let (free, _) = model.device().mem_info().context("querying free vram")?;
    let want = max_seqs.saturating_mul(max_seq).max(max_seq);
    let mut lo = max_seq;
    let mut hi = want;
    // The pool's byte count is not a simple product — TurboQuant carries
    // per-slot tables — so find the largest slot count that fits by bisection
    // on the pool's own accounting rather than by re-deriving it here.
    let budget = free.saturating_sub(free / 8).saturating_sub(512 << 20);
    let fits = |n: usize| -> bool {
        match model.new_pool(n, max_seqs) {
            Ok(p) => p.bytes() <= budget,
            Err(_) => false,
        }
    };
    if fits(hi) {
        return model
            .new_pool(hi, max_seqs)
            .context("allocating the kv pool");
    }
    while lo + max_seq < hi {
        let mid = (lo + hi) / 2;
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    model
        .new_pool(lo, max_seqs)
        .context("allocating the kv pool")
}

/// Where the MTP head's GGUF tensors live, if anywhere findable without
/// being told.
///
/// `INFERO_MTP` names it outright. Otherwise this looks in the model's own
/// directory for a single `mtp*.gguf`, which is what llama.cpp's conversion
/// produces and what the ggml-org repacks ship. Two of them is ambiguous rather
/// than a reason to guess, so it takes neither and speculation stays off — a
/// user with two heads should say which. Failing that, the main file itself
/// is checked for `nextn_predict_layers`: some exports pack the head into the
/// last blocks of the same file instead of shipping a sidecar, and returning
/// the model's own path here is what lets [`Scheduler::enable_speculation`]
/// hand it straight to [`Model::load_mtp_head_gguf`] unchanged.
fn mtp_sidecar(model: &str) -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("INFERO_MTP") {
        let p = std::path::PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
        tracing::warn!(path = %p.display(), "INFERO_MTP does not name a file");
        return None;
    }
    let path = std::path::Path::new(model);
    if path.extension().is_none_or(|e| e != "gguf") {
        return None;
    }
    let dir = path.parent()?;
    let mut hits: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "gguf")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("mtp"))
        })
        .collect();
    hits.sort();
    match hits.len() {
        1 => hits.pop(),
        // No separate file: some GGUFs (llama.cpp/unsloth's Qwen3.8 exports
        // among them) pack the head into the main file instead, as the last
        // `nextn_predict_layers` blocks. `load_mtp_head_gguf` already knows
        // how to read that shape out of a `Gguf` handle -- it just has never
        // been pointed at the main file itself before. Reading the model's
        // own metadata here, rather than assuming, is what tells a plain
        // single-head checkpoint from one that never carried a head at all.
        0 => {
            let f = infero_gguf::Gguf::open(path).ok()?;
            f.usize(&f.akey("nextn_predict_layers").ok()?).ok()?;
            Some(path.to_path_buf())
        }
        n => {
            tracing::warn!(
                count = n,
                "several mtp*.gguf beside the model; name one with INFERO_MTP"
            );
            None
        }
    }
}
