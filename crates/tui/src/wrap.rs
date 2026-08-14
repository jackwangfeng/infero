//! Width-aware line wrapping.
//!
//! Not `Paragraph`'s built-in wrapping: the app needs the wrapped line *count*
//! to clamp scrolling, and it has to be right for CJK, where one character
//! occupies two terminal cells. A model that answers in Chinese would otherwise
//! overflow every line it writes.

use unicode_width::UnicodeWidthChar;

/// Break `text` into lines no wider than `width` display cells.
///
/// Existing newlines are kept. Wrapping prefers a space, but falls back to a
/// hard break mid-run — necessary for CJK, which has no spaces, and for long
/// URLs and code.
///
/// The one case that can exceed `width` is a single character wider than the
/// whole box, which has nowhere else to go.
pub fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut out = Vec::new();

    for paragraph in text.split('\n') {
        if paragraph.is_empty() {
            out.push(String::new());
            continue;
        }

        let mut line = String::new();
        let mut line_width = 0usize;
        // Where the last space in `line` is, so a break can back up to it.
        let mut last_space: Option<(usize, usize)> = None;

        for ch in paragraph.chars() {
            let w = ch.width().unwrap_or(0);
            if line_width + w > width && !line.is_empty() {
                match last_space {
                    // Break at the space and carry the rest to the next line.
                    Some((byte, _)) if byte > 0 => {
                        let rest: String = line[byte..].trim_start().to_string();
                        line.truncate(byte);
                        out.push(std::mem::take(&mut line));
                        line_width = rest.chars().map(|c| c.width().unwrap_or(0)).sum();
                        line = rest;
                    }
                    _ => {
                        out.push(std::mem::take(&mut line));
                        line_width = 0;
                    }
                }
                last_space = None;
            }
            if ch == ' ' {
                last_space = Some((line.len(), line_width));
            }
            line.push(ch);
            line_width += w;
        }
        out.push(line);
    }
    out
}

/// Display width of a string in terminal cells.
pub fn width(s: &str) -> usize {
    s.chars().map(|c| c.width().unwrap_or(0)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn breaks_on_spaces_when_it_can() {
        assert_eq!(
            wrap("the quick brown fox", 10),
            vec!["the quick", "brown fox"]
        );
    }

    #[test]
    fn keeps_existing_newlines() {
        assert_eq!(wrap("a\n\nb", 10), vec!["a", "", "b"]);
    }

    #[test]
    fn counts_cjk_as_two_cells() {
        // Six characters, twelve cells: exactly two lines at width six.
        let lines = wrap("你好世界再见", 6);
        assert_eq!(lines.len(), 2);
        for l in &lines {
            assert!(width(l) <= 6, "{l:?} is {} cells wide", width(l));
        }
    }

    #[test]
    fn hard_breaks_a_run_with_no_spaces() {
        let lines = wrap(&"x".repeat(25), 10);
        assert_eq!(lines.len(), 3);
        assert!(lines.iter().all(|l| width(l) <= 10));
    }

    #[test]
    fn never_exceeds_the_width() {
        // Except for a lone character too wide for the box, which cannot be
        // split any further.
        let samples = [
            "hello world",
            "你好，世界！这是一段中文测试文本。",
            "mixed 中文 and English 混合",
            "🦀🦀🦀🦀🦀",
            "",
            "   leading and trailing   ",
        ];
        for text in samples {
            for w in [1usize, 3, 8, 20] {
                for line in wrap(text, w) {
                    let widest_char = line
                        .chars()
                        .map(|c| c.width().unwrap_or(0))
                        .max()
                        .unwrap_or(1);
                    let allowed = w.max(1).max(widest_char);
                    assert!(
                        width(&line) <= allowed,
                        "{text:?} at width {w} produced {line:?} ({} cells)",
                        width(&line)
                    );
                    assert!(
                        line.chars().count() == 1 || width(&line) <= w.max(1),
                        "{line:?} overflows width {w} with more than one character"
                    );
                }
            }
        }
    }

    #[test]
    fn loses_no_characters() {
        let text = "the quick brown fox jumps over the lazy dog";
        let joined: String = wrap(text, 12).join(" ");
        // Spaces move around at break points, but no word may vanish.
        for word in text.split(' ') {
            assert!(joined.contains(word), "lost {word:?}");
        }
    }
}
