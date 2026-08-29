//! Turning an OpenAI-style `video_url` into sampled RGB frames.
//!
//! Same policy as `vision.rs`, for the same reason: only `data:` URLs are
//! accepted, never a remote fetch on a caller's behalf — see that module's
//! note for why. A video payload is decoded through `ffmpeg`/`ffprobe`
//! subprocesses rather than a Rust codec crate, on the reasoning
//! `docs/superpowers/specs/2026-08-24-infero-metal-port-design.md`-adjacent
//! decisions in this codebase already lean on: the alternative is either
//! `ffmpeg-next` (FFI to the same library, plus a build-time dependency on
//! system headers and a large unsafe surface) or a pure-Rust demuxer/decoder
//! stack, which for a real MP4/H.264 clip is not a small parser. A subprocess
//! adds zero crates for the actual decoding, and the byte-cap/timeout/`-nostdin`
//! discipline below is the same sandboxing story `vision.rs`'s size caps
//! already tell.
//!
//! The container goes to a securely-created temp file rather than a pipe:
//! MP4's index (`moov` atom) is commonly at the *end* of the file, so
//! `ffprobe`/`ffmpeg` need to seek, which a pipe cannot do without buffering
//! the whole payload in the subprocess anyway. `tempfile::Builder` rather
//! than a hand-built path in `/tmp`: this box is shared with other users
//! (see `notes/`), and a predictable temp filename there is a symlink-attack
//! vector, not just untidy.
//!
//! Sampling follows `Qwen3VLVideoProcessor.sample_frames` and
//! `Qwen3VLProcessor._calculate_timestamps` exactly — read off the installed
//! `transformers` on the box with the checkpoint, not recalled from memory —
//! with one deliberate deviation documented on [`pair_timestamps`].

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use base64::Engine;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

/// Bytes after base64 decoding. A short clip easily exceeds a photo's budget;
/// still capped so a request body cannot become an unbounded allocation
/// before decoding starts.
const MAX_ENCODED_BYTES: usize = 64 * 1024 * 1024;

/// Per-frame pixel cap, the same reasoning and the same number as
/// `vision::MAX_PIXELS`.
const MAX_PIXELS_PER_FRAME: usize = 16 * 1024 * 1024;

/// `Qwen3VLVideoProcessor`'s own sampling constant: 2 effective frames a
/// second. This server's own knob (`--video-target-fps`, threaded through as
/// `decode_video_data_url`'s `fps` parameter) can override it per request --
/// this is just what a request that does not care falls back to. Real videos
/// vary a lot in how much temporal density they need (a lecture slide barely
/// needs 1fps; a fast hand grabbing an item off a shelf wants more, see
/// `notes/video-encoding-optimizations.md`'s item 6), so a single
/// compile-time value was never going to be right for every caller.
pub const DEFAULT_TARGET_FPS: f64 = 2.0;
/// At least 4 frames sampled regardless of a very short clip. `max_frames`
/// in the reference is 768; this server takes it as a caller-supplied budget
/// instead (`Scheduler`'s `--video-max-frames`) -- see
/// `notes/mrope-and-video.md`.
const MIN_FRAMES: usize = 4;

/// How long `ffprobe`+`ffmpeg` together may run before this gives up and
/// kills them. Generous for a real clip at the pixel/frame caps below;
/// mainly a backstop against a payload crafted to make the decoder hang
/// rather than fail.
const DECODE_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug)]
pub struct DecodedClip {
    /// Interleaved `[height, width, 3]` per sampled frame, in playback order.
    /// Always non-empty and, past `qwen35_vision_image::prepare_clip`'s own
    /// last-frame duplication, not necessarily even — that padding is
    /// `prepare_clip`'s job, not this module's.
    pub frames: Vec<Vec<u8>>,
    pub height: usize,
    pub width: usize,
    /// One `<{t:.1f} seconds>` timestamp per *temporal-patch group*
    /// (`frames.len().div_ceil(2)` entries), already paired -- see
    /// [`pair_timestamps`].
    pub timestamps: Vec<f64>,
}

/// Decode a `data:video/...;base64,...` URL into sampled RGB8 frames.
///
/// `target_fps` is resolved by the caller before this is reached -- the
/// request's own `video_url.fps` (see `crate::api::VideoUrl`) if it set one,
/// else the server's `--video-target-fps` default (`Engine::video_target_fps`,
/// itself defaulting to `DEFAULT_TARGET_FPS`). Taking a concrete `f64` here
/// rather than an `Option` keeps that precedence in one place (`routes.rs`)
/// instead of duplicating "which default wins" logic into this module too.
pub async fn decode_video_data_url(url: &str, max_frames: usize, target_fps: f64) -> Result<DecodedClip> {
    let rest = url.strip_prefix("data:").with_context(|| {
        format!(
            "video_url must be a data: URL (fetching a remote URL on a \
             caller's behalf is not offered — see the video module note); got \
             a URL starting {:?}",
            &url.chars().take(24).collect::<String>()
        )
    })?;
    let (header, payload) = rest
        .split_once(',')
        .context("data: URL has no comma separating its header from its payload")?;
    anyhow::ensure!(
        header.ends_with(";base64"),
        "data: URL must be base64-encoded (`;base64` in the header), got {header:?}"
    );
    anyhow::ensure!(
        payload.len() <= MAX_ENCODED_BYTES * 4 / 3 + 4,
        "video payload of {} bytes is over the {MAX_ENCODED_BYTES}-byte limit",
        payload.len()
    );
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.as_bytes())
        .context("video payload is not valid base64")?;
    decode_bytes(&bytes, max_frames, target_fps).await
}

async fn decode_bytes(bytes: &[u8], max_frames: usize, target_fps: f64) -> Result<DecodedClip> {
    anyhow::ensure!(!bytes.is_empty(), "empty video payload");
    anyhow::ensure!(max_frames > 0, "max_frames must be positive");
    anyhow::ensure!(target_fps > 0.0, "target_fps must be positive, got {target_fps}");

    let tmp = tempfile::Builder::new()
        .prefix("infero-video-")
        .suffix(".bin")
        .tempfile()
        .context("creating a temp file for the video payload")?;
    tokio::fs::write(tmp.path(), bytes)
        .await
        .context("writing the video payload to a temp file")?;

    let probe = probe_video(tmp.path()).await?;
    anyhow::ensure!(
        probe.width > 0 && probe.height > 0,
        "ffprobe reported a {}x{} stream", probe.width, probe.height
    );
    anyhow::ensure!(
        probe.width.saturating_mul(probe.height) <= MAX_PIXELS_PER_FRAME,
        "a {}x{} frame is {} megapixels, over the {} the server accepts",
        probe.width, probe.height,
        (probe.width * probe.height) / 1_000_000,
        MAX_PIXELS_PER_FRAME / 1_000_000
    );
    anyhow::ensure!(probe.nb_frames > 0, "ffprobe reported no frames");

    let num_frames = sample_frame_count(probe.nb_frames, probe.fps, max_frames, target_fps);
    let indices = sample_indices(probe.nb_frames, num_frames);
    let timestamps = pair_timestamps(&indices, probe.fps);

    let frames = extract_frames(tmp.path(), &indices, probe.fps, probe.width, probe.height).await?;
    Ok(DecodedClip { frames, height: probe.height, width: probe.width, timestamps })
}

/// `Qwen3VLVideoProcessor.sample_frames`'s frame-count rule: `total /
/// video_fps * target_fps`, clamped to `[MIN_FRAMES, max_frames,
/// total_frames]`.
fn sample_frame_count(total_frames: usize, video_fps: f64, max_frames: usize, target_fps: f64) -> usize {
    let n = ((total_frames as f64 / video_fps) * target_fps) as i64;
    (n.max(0) as usize).clamp(MIN_FRAMES.min(total_frames), max_frames.min(total_frames)).max(1)
}

/// `np.linspace(0, total_frames - 1, num_frames).round()`: `num_frames`
/// evenly spaced indices across the whole clip, first and last frame always
/// included when `num_frames > 1`.
fn sample_indices(total_frames: usize, num_frames: usize) -> Vec<usize> {
    if num_frames <= 1 || total_frames <= 1 {
        return vec![0; num_frames.max(1)];
    }
    let last = (total_frames - 1) as f64;
    (0..num_frames)
        .map(|i| {
            let v = last * i as f64 / (num_frames - 1) as f64;
            (v.round() as usize).min(total_frames - 1)
        })
        .collect()
}

/// `Qwen3VLProcessor._calculate_timestamps`: each sampled index becomes
/// `idx / video_fps` seconds, then consecutive pairs are averaged into one
/// timestamp a temporal-patch group (`temporal_patch_size` is always 2 on
/// this checkpoint, so this does not take it as a parameter).
///
/// **Deviation from the reference, found and confirmed by running it**: the
/// reference indexes `timestamps[i + 1]` in the last pair unconditionally,
/// which is an `IndexError` on an odd-length `indices` -- reproduced against
/// the installed `transformers` with `total_frames=18, video_fps=7` (giving
/// `num_frames=5`). Rather than propagate that crash, an odd `indices` here
/// gets its last entry duplicated before pairing, the same story
/// `qwen35_vision::patchify`'s doc comment already tells for a still image's
/// two identical temporal taps and `prepare_clip` tells for an odd frame
/// count: the padding frame is a copy, not a gap, so its timestamp is
/// naturally the same real frame's time, not an average with something that
/// does not exist.
fn pair_timestamps(indices: &[usize], video_fps: f64) -> Vec<f64> {
    let mut padded = indices.to_vec();
    if !padded.len().is_multiple_of(2) && !padded.is_empty() {
        padded.push(*padded.last().unwrap());
    }
    padded
        .chunks(2)
        .map(|pair| {
            let a = pair[0] as f64 / video_fps;
            let b = *pair.last().unwrap() as f64 / video_fps;
            (a + b) / 2.0
        })
        .collect()
}

struct Probe {
    width: usize,
    height: usize,
    nb_frames: usize,
    fps: f64,
}

async fn probe_video(path: &Path) -> Result<Probe> {
    let run = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames,r_frame_rate,avg_frame_rate,duration",
            "-of", "json",
        ])
        .arg(path)
        .stdin(std::process::Stdio::null())
        .output();
    let out = tokio::time::timeout(DECODE_TIMEOUT, run)
        .await
        .context("ffprobe timed out")?
        .context("running ffprobe -- is it installed? this build needs ffmpeg/ffprobe on PATH for video input")?;
    anyhow::ensure!(
        out.status.success(),
        "ffprobe failed: {}",
        String::from_utf8_lossy(&out.stderr).trim()
    );
    let json: serde_json::Value =
        serde_json::from_slice(&out.stdout).context("ffprobe did not print valid JSON")?;
    let stream = json["streams"]
        .get(0)
        .context("ffprobe found no video stream")?;
    let width = stream["width"].as_u64().context("ffprobe reported no width")? as usize;
    let height = stream["height"].as_u64().context("ffprobe reported no height")? as usize;
    let fps = parse_rate(stream["r_frame_rate"].as_str())
        .or_else(|| parse_rate(stream["avg_frame_rate"].as_str()))
        .filter(|f| *f > 0.0)
        .context("ffprobe reported no usable frame rate")?;
    // `nb_frames` is absent for some containers (webm commonly); fall back to
    // `duration * fps`, which is what the reference does when it has no
    // frame count either (`metadata.fps is None` branch aside, the frame
    // *count* path already assumes duration-derived counting is fine).
    let nb_frames = stream["nb_frames"]
        .as_str()
        .and_then(|s| s.parse::<usize>().ok())
        .or_else(|| {
            stream["duration"]
                .as_str()
                .and_then(|s| s.parse::<f64>().ok())
                .map(|d| (d * fps).round().max(1.0) as usize)
        })
        .context("ffprobe reported neither nb_frames nor a duration to derive it from")?;
    Ok(Probe { width, height, nb_frames, fps })
}

/// `"30/1"` or `"30000/1001"` -> `29.97`. `ffprobe` always uses this
/// rational form for `r_frame_rate`/`avg_frame_rate`, never a plain decimal.
fn parse_rate(s: Option<&str>) -> Option<f64> {
    let s = s?;
    let (num, den) = s.split_once('/')?;
    let (num, den): (f64, f64) = (num.parse().ok()?, den.parse().ok()?);
    (den != 0.0).then_some(num / den)
}

/// Run `ffmpeg` once, selecting exactly `indices` by frame number and piping
/// raw `rgb24` frames out. `-vsync 0` is what keeps `select` from also
/// duplicating or dropping frames to match an output frame rate — without it
/// `ffmpeg` still "helpfully" retimes the selected frames against the input's
/// nominal fps, which for a sparse selection silently drops most of them.
///
/// A per-frame `-ss` seek (fast, format-level) was tried here and reverted --
/// see `notes/video-encoding-optimizations.md`, item 2's writeup, for real
/// numbers. Short version: on this real phone-shot `sample.mp4`, seeking to
/// `idx as f64 / fps` and taking the first decoded frame landed one frame
/// early past roughly the 5-second mark (B-frame reordering right after an
/// accurate seek emits a lead-in frame before the true target), and
/// switching to `select='gte(t,ts)'` after a lead-in seek fixed *that* but
/// then gave a *different* checksum for the same target timestamp depending
/// on how far before it the seek landed -- i.e. decode is not reproducibly
/// keyframe-independent on this file, which is a correctness risk this
/// server is not in a position to paper over with more seek heuristics.
/// `fps` stays an unused parameter here (leading underscore) so this
/// function's signature does not have to change again if a real fix (e.g. a
/// proper demuxer binding like vLLM's `pyav`/`torchcodec` backends, not
/// another `-ss` heuristic) lands later.
async fn extract_frames(path: &Path, indices: &[usize], _fps: f64, w: usize, h: usize) -> Result<Vec<Vec<u8>>> {
    let select_expr = indices
        .iter()
        .map(|i| format!("eq(n\\,{i})"))
        .collect::<Vec<_>>()
        .join("+");
    let mut child = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(path)
        .args(["-vf", &format!("select='{select_expr}'"), "-vsync", "0"])
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .context("running ffmpeg -- is it installed? this build needs ffmpeg/ffprobe on PATH for video input")?;

    let frame_bytes = w * h * 3;
    let expected = frame_bytes * indices.len();
    let mut stdout = child.stdout.take().expect("piped stdout");
    let read = async {
        // Read at most `expected` bytes: a well-formed selection produces
        // exactly that many, and capping the read here bounds worst-case
        // memory regardless of what a misbehaving `ffmpeg` does past that.
        let mut buf = vec![0u8; expected];
        stdout.read_exact(&mut buf).await?;
        Ok::<_, std::io::Error>(buf)
    };
    let result = tokio::time::timeout(DECODE_TIMEOUT, read).await;
    // Whatever happened above, the child is done being useful: reading exactly
    // `expected` bytes does not mean it has exited (it may still be flushing
    // or waiting on a pipe), and a timeout or short read both mean it should
    // not be left running.
    let _ = child.kill().await;
    let raw = match result {
        Ok(Ok(buf)) => buf,
        Ok(Err(e)) => {
            let mut stderr = String::new();
            if let Some(mut se) = child.stderr.take() {
                use tokio::io::AsyncReadExt as _;
                let _ = se.read_to_string(&mut stderr).await;
            }
            anyhow::bail!(
                "ffmpeg produced fewer than the expected {expected} bytes for {} frames \
                 ({e}); stderr: {}",
                indices.len(),
                stderr.trim()
            );
        }
        Err(_) => anyhow::bail!("ffmpeg timed out decoding {} frames", indices.len()),
    };
    Ok(raw.chunks_exact(frame_bytes).map(|c| c.to_vec()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_frame_count_matches_the_reference_formula() {
        // (total_frames, video_fps, max_frames) -> expected, hand-computed
        // from `int(total/video_fps*2)` clamped to `[4, max_frames, total]`.
        let cases = [
            (630, 30.0, 768, 42),  // sample.mp4: 21.0s at 30fps -> 42 frames at 2fps
            (48, 24.0, 768, 4),    // 2s clip: int(48/24*2)=4, already at the floor
            (10, 30.0, 768, 4),    // very short: floor still applies
            (100000, 30.0, 16, 16), // a caller-supplied ceiling binds
        ];
        for (total, fps, max_frames, want) in cases {
            let got = sample_frame_count(total, fps, max_frames, DEFAULT_TARGET_FPS);
            assert_eq!(got, want, "total={total} fps={fps} max_frames={max_frames}");
        }
    }

    #[test]
    fn a_higher_target_fps_samples_more_frames() {
        // Same clip as `sample.mp4` above (21.0s at 30fps), but asked for a
        // denser sample -- the request-level override this whole mechanism
        // exists for (see `notes/video-encoding-optimizations.md`, item 6).
        let at_2fps = sample_frame_count(630, 30.0, 768, 2.0);
        let at_6fps = sample_frame_count(630, 30.0, 768, 6.0);
        assert_eq!(at_2fps, 42);
        assert_eq!(at_6fps, 126);
    }

    #[test]
    fn sample_indices_spans_the_whole_clip() {
        // Cross-checked against `np.linspace(0, 99, 5).round()` directly,
        // not hand-computed: 74.25 rounds to 74, not the 75 an eyeballed
        // "evenly spaced" guess lands on.
        let idx = sample_indices(100, 5);
        assert_eq!(idx, vec![0, 25, 50, 74, 99]);
        assert_eq!(sample_indices(10, 1), vec![0]);
        // `total_frames == 1` collapses `linspace(0, 0, n)` to `n` copies of
        // 0 -- also cross-checked, not assumed.
        assert_eq!(sample_indices(1, 4), vec![0, 0, 0, 0]);
    }

    #[test]
    fn pair_timestamps_averages_consecutive_pairs_at_30fps() {
        // 30fps: indices 0,15,30,45 -> seconds 0.0,0.5,1.0,1.5 -> pairs
        // average to 0.25, 1.25.
        let ts = pair_timestamps(&[0, 15, 30, 45], 30.0);
        assert_eq!(ts.len(), 2);
        assert!((ts[0] - 0.25).abs() < 1e-9);
        assert!((ts[1] - 1.25).abs() < 1e-9);
    }

    /// The case that crashes the real reference (`total_frames=18,
    /// video_fps=7` gives an odd `num_frames=5`) — confirmed by running
    /// `transformers`' own `_calculate_timestamps` against it. This module's
    /// version must not propagate that crash; see `pair_timestamps`'s doc
    /// comment for the deviation.
    #[test]
    fn an_odd_index_count_pads_rather_than_panicking() {
        let idx = sample_indices(18, sample_frame_count(18, 7.0, 768, DEFAULT_TARGET_FPS));
        assert_eq!(idx.len() % 2, 1, "this case is only interesting if it is odd");
        let ts = pair_timestamps(&idx, 7.0);
        assert_eq!(ts.len(), idx.len().div_ceil(2));
        // The padded pair's timestamp is the real last frame's own time, not
        // an average with a frame that does not exist.
        let last = *idx.last().unwrap() as f64 / 7.0;
        assert!((*ts.last().unwrap() - last).abs() < 1e-9);
    }

    #[test]
    fn parse_rate_reads_ffprobes_rational_form() {
        assert_eq!(parse_rate(Some("30/1")), Some(30.0));
        assert!((parse_rate(Some("30000/1001")).unwrap() - 29.97).abs() < 1e-2);
        assert_eq!(parse_rate(Some("0/0")), None);
        assert_eq!(parse_rate(None), None);
    }

    #[tokio::test]
    async fn a_remote_url_is_refused_rather_than_fetched() {
        let err = decode_video_data_url("https://example.com/cat.mp4", 16, DEFAULT_TARGET_FPS)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("data:"), "{err}");
    }

    #[tokio::test]
    async fn garbage_after_the_comma_is_refused_not_panicked_on() {
        assert!(
            decode_video_data_url("data:video/mp4;base64,not valid base64!!", 16, DEFAULT_TARGET_FPS)
                .await
                .is_err()
        );
        assert!(decode_video_data_url("data:video/mp4;base64,", 16, DEFAULT_TARGET_FPS).await.is_err());
        assert!(decode_video_data_url("data:text/plain,hello", 16, DEFAULT_TARGET_FPS).await.is_err());
    }

    /// The whole pipeline -- base64, temp file, `ffprobe`, `ffmpeg`, frame
    /// extraction -- against a real video, gated on `INFERO_SAMPLE_VIDEO`
    /// pointing at one. Not run by default: this is the only test in the
    /// file that actually shells out, and CI/dev machines are not
    /// guaranteed to have `ffmpeg` or a sample clip.
    #[tokio::test]
    async fn decodes_a_real_video_end_to_end() {
        let Ok(path) = std::env::var("INFERO_SAMPLE_VIDEO") else {
            eprintln!("SKIPPED: set INFERO_SAMPLE_VIDEO to a real video file to run this");
            return;
        };
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let url = format!("data:video/mp4;base64,{b64}");
        let clip = decode_video_data_url(&url, 64, DEFAULT_TARGET_FPS).await.unwrap();
        eprintln!(
            "{}x{}, {} frames sampled, {} timestamps, first/last ts {:.1}/{:.1}s",
            clip.width, clip.height, clip.frames.len(), clip.timestamps.len(),
            clip.timestamps.first().copied().unwrap_or(0.0),
            clip.timestamps.last().copied().unwrap_or(0.0),
        );
        assert!(clip.width > 0 && clip.height > 0);
        assert!(!clip.frames.is_empty() && clip.frames.len() <= 64);
        for f in &clip.frames {
            assert_eq!(f.len(), clip.width * clip.height * 3, "a short/garbage frame");
        }
        assert_eq!(clip.timestamps.len(), clip.frames.len().div_ceil(2));
        // Real playback order: timestamps must be non-decreasing.
        for w in clip.timestamps.windows(2) {
            assert!(w[1] >= w[0], "timestamps went backwards: {:?}", clip.timestamps);
        }
        // Frames must actually differ -- a decode that returned the same
        // frame `n` times would pass every check above and still be wrong.
        let distinct = clip.frames.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(distinct > 1, "every sampled frame is byte-identical");
    }
}
