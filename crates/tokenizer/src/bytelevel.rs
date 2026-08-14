//! GPT-2's byte-to-unicode alphabet.
//!
//! Byte-level BPE can't put raw control bytes in a vocabulary string, so GPT-2
//! maps all 256 byte values onto printable code points: printable ASCII and
//! Latin-1 map to themselves, everything else moves into the U+0100.. range.
//! That is why a space shows up as `Ġ` and a newline as `Ċ` in GGUF vocabs.

/// byte -> code point
pub struct ByteLevel {
    to_char: [char; 256],
    /// Reverse map, indexed by code point. Sparse but tiny (< 400 entries).
    from_char: std::collections::HashMap<char, u8>,
}

impl ByteLevel {
    pub fn new() -> Self {
        let mut to_char = ['\0'; 256];
        let mut assigned = [false; 256];

        // The three runs GPT-2 keeps as-is.
        for b in b'!'..=b'~' {
            to_char[b as usize] = b as char;
            assigned[b as usize] = true;
        }
        for b in 0xA1u8..=0xAC {
            to_char[b as usize] = b as char;
            assigned[b as usize] = true;
        }
        for b in 0xAEu8..=0xFF {
            to_char[b as usize] = b as char;
            assigned[b as usize] = true;
        }

        // Everything else is pushed above U+0100 in byte order.
        let mut next = 0u32;
        for b in 0..=255usize {
            if !assigned[b] {
                to_char[b] = char::from_u32(256 + next).unwrap();
                next += 1;
            }
        }

        let from_char = to_char
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();

        Self { to_char, from_char }
    }

    /// Raw bytes -> the vocabulary's character alphabet.
    pub fn encode(&self, bytes: &[u8]) -> String {
        bytes.iter().map(|&b| self.to_char[b as usize]).collect()
    }

    /// Inverse of [`encode`]. Characters outside the alphabet are passed
    /// through as UTF-8, which is what lets literal tokens like `<|im_end|>`
    /// survive a decode.
    pub fn decode_into(&self, text: &str, out: &mut Vec<u8>) {
        for c in text.chars() {
            match self.from_char.get(&c) {
                Some(&b) => out.push(b),
                None => {
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
    }
}

impl Default for ByteLevel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_gpt2_alphabet() {
        let bl = ByteLevel::new();
        assert_eq!(bl.encode(b" "), "\u{120}"); // 'Ġ'
        assert_eq!(bl.encode(b"\n"), "\u{10a}"); // 'Ċ'
        assert_eq!(bl.encode(b"hello"), "hello");
        assert_eq!(bl.encode(b" world"), "\u{120}world");
    }

    #[test]
    fn roundtrips_every_byte() {
        let bl = ByteLevel::new();
        let all: Vec<u8> = (0..=255).collect();
        let encoded = bl.encode(&all);
        let mut back = Vec::new();
        bl.decode_into(&encoded, &mut back);
        assert_eq!(back, all);
    }

    #[test]
    fn passes_through_literal_special_tokens() {
        let bl = ByteLevel::new();
        let mut out = Vec::new();
        bl.decode_into("<|im_end|>", &mut out);
        assert_eq!(out, b"<|im_end|>");
    }
}
