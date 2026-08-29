//! The Hugging Face reader against the GGUF one, on the same model.
//!
//! A `tokenizer.json` and a GGUF's embedded vocabulary describe the same
//! byte-level BPE, so the two readers should produce the same token ids for the
//! same text. That is a much sharper check than round-tripping either one
//! against itself: a merge list read in the wrong order, or an `added_tokens`
//! id placed at the wrong index, still decodes back to the input.

use anyhow::Result;
use infero_tokenizer::Tokenizer;

const HF: &str = "/mnt/data/vllm-bench/llama8b-awq";
const GGUF: &str = "/mnt/data/infero-models/llama-3.1-8b-instruct-q4_k_m.gguf";

#[test]
fn the_hf_reader_agrees_with_the_gguf_one() -> Result<()> {
    if !std::path::Path::new(HF).exists() || !std::path::Path::new(GGUF).exists() {
        eprintln!("skipping: need both {HF} and {GGUF}");
        return Ok(());
    }
    let hf = Tokenizer::from_hf_dir(HF)?;
    let gg = Tokenizer::from_gguf(&infero_gguf::Gguf::open(GGUF)?)?;

    assert_eq!(hf.vocab_size(), gg.vocab_size());
    assert_eq!(hf.pretokenizer(), gg.pretokenizer());
    assert_eq!(hf.bos_id(), gg.bos_id());
    assert_eq!(hf.eos_id(), gg.eos_id());

    for text in [
        "Hello, world!",
        "请详细讲解一下 Transformer 的完整结构",
        "  leading and trailing  ",
        "numbers 1234567890 and punctuation —— ok?",
        "<|begin_of_text|>marked<|eot_id|>",
        "a\nb\r\nc\t\td",
        "🙂🙃 emoji and \u{200b}zero width",
    ] {
        for special in [false, true] {
            let a = hf.encode(text, Some(false), special);
            let b = gg.encode(text, Some(false), special);
            assert_eq!(a, b, "text {text:?} (parse_special={special})");
            assert_eq!(hf.decode(&a, true), gg.decode(&b, true));
        }
    }

    // Every id has to name the same piece, not just the ones a prompt reaches.
    for id in (0..hf.vocab_size() as u32).step_by(97) {
        assert_eq!(hf.id_to_piece(id), gg.id_to_piece(id), "id {id}");
    }
    Ok(())
}
