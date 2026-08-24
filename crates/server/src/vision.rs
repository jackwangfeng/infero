//! Turning an OpenAI-style `image_url` into pixels the vision tower can read.
//!
//! Only `data:` URLs are accepted. Fetching an `http(s)://` URL the caller
//! supplies would make this server an open SSRF proxy: whatever it can reach
//! — other services on this box, a cloud metadata endpoint, an internal
//! network the operator did not mean to expose — a request could point it at.
//! A `data:` URL carries its own bytes, so accepting only that shape is not a
//! missing feature so much as the one shape that does not need a policy
//! decision about what this server is allowed to fetch on a stranger's behalf.
//! Serving remote URLs later needs an explicit allowlist and is a separate
//! decision, not a default.

use anyhow::{Context, Result};
use base64::Engine;

/// Bytes after base64 decoding. Generous for a real photo, small enough that a
/// request body cannot turn into an unbounded allocation before decoding even
/// starts.
const MAX_ENCODED_BYTES: usize = 24 * 1024 * 1024;

/// Decoded pixels. 16 megapixels is a 4000x4000 photo — well past anything a
/// vision-language prompt needs — capped so a decode cannot itself become the
/// expensive operation an attacker controls the size of.
const MAX_PIXELS: usize = 16 * 1024 * 1024;

#[derive(Debug)]
pub struct DecodedImage {
    /// Interleaved `[height, width, 3]`, row-major — what
    /// `qwen35_vision_image::prepare_frame` reads.
    pub rgb: Vec<u8>,
    pub height: usize,
    pub width: usize,
}

/// Decode a `data:image/...;base64,...` URL into RGB8 pixels.
///
/// Refuses anything else — see the module note — with a message that says why
/// rather than just that the URL was rejected, since "unsupported URL scheme"
/// reads like a bug report waiting to happen otherwise.
pub fn decode_data_url(url: &str) -> Result<DecodedImage> {
    let rest = url.strip_prefix("data:").with_context(|| {
        format!(
            "image_url must be a data: URL (fetching a remote URL on a \
             caller's behalf is not offered — see the vision module note); got \
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
        "image payload of {} bytes is over the {MAX_ENCODED_BYTES}-byte limit",
        payload.len()
    );
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.as_bytes())
        .context("image payload is not valid base64")?;
    decode_bytes(&bytes)
}

/// Decode already-raw (non-data-URL) image bytes, shared with the base64 path
/// and its own unit tests.
fn decode_bytes(bytes: &[u8]) -> Result<DecodedImage> {
    let img = image::load_from_memory(bytes).context("could not decode image data")?;
    let (w, h) = (img.width() as usize, img.height() as usize);
    anyhow::ensure!(
        w > 0 && h > 0,
        "image decoded to zero pixels in one dimension"
    );
    anyhow::ensure!(
        w.saturating_mul(h) <= MAX_PIXELS,
        "a {w}x{h} image is {} megapixels, over the {} the server accepts",
        (w * h) / 1_000_000,
        MAX_PIXELS / 1_000_000
    );
    let rgb = img.into_rgb8().into_raw();
    Ok(DecodedImage { rgb, height: h, width: w })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data_url(rgb: [u8; 3], w: u32, h: u32) -> String {
        let mut buf = std::io::Cursor::new(Vec::new());
        let img = image::RgbImage::from_fn(w, h, |_, _| image::Rgb(rgb));
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut buf, image::ImageFormat::Png)
            .unwrap();
        let b64 = base64::engine::general_purpose::STANDARD.encode(buf.into_inner());
        format!("data:image/png;base64,{b64}")
    }

    #[test]
    fn a_solid_png_round_trips_to_the_same_pixels() {
        let url = data_url([220, 30, 30], 8, 4);
        let got = decode_data_url(&url).unwrap();
        assert_eq!((got.width, got.height), (8, 4));
        assert_eq!(got.rgb.len(), 8 * 4 * 3);
        // Every pixel the same colour, since the source image was solid.
        for px in got.rgb.chunks_exact(3) {
            assert_eq!(px, [220, 30, 30]);
        }
    }

    #[test]
    fn a_remote_url_is_refused_rather_than_fetched() {
        let err = decode_data_url("https://example.com/cat.png").unwrap_err().to_string();
        assert!(err.contains("data:"), "{err}");
    }

    #[test]
    fn garbage_after_the_comma_is_refused_not_panicked_on() {
        assert!(decode_data_url("data:image/png;base64,not valid base64!!").is_err());
        assert!(decode_data_url("data:image/png;base64,").is_err());
        assert!(decode_data_url("data:text/plain,hello").is_err());
    }
}
