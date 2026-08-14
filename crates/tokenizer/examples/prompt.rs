//! Print the prompt a model's own chat template produces.
//!
//!     cargo run -p tuili-tokenizer --example prompt -- <model.gguf> "hello"

use anyhow::{Context, Result};
use tuili_gguf::Gguf;
use tuili_tokenizer::{ChatMessage, Tokenizer};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .context("usage: prompt <model.gguf> [message]")?;
    let message = args.next().unwrap_or_else(|| "hello".into());

    let gguf = Gguf::open(&path)?;
    let tok = Tokenizer::from_gguf(&gguf)?;
    let template = tok.chat_template().context("model has no chat template")?;
    let rendered = template.render(&[ChatMessage::user(&message)], true)?;

    println!("--- rendered ({} bytes) ---", rendered.len());
    println!("{rendered}");
    let ids = tok.encode(&rendered, None, true);
    println!("--- {} tokens ---", ids.len());
    for id in &ids {
        let piece = tok.id_to_piece(*id).unwrap_or("?");
        let mark = if tok.is_special(*id) { "*" } else { " " };
        print!("{mark}{id}:{piece}  ");
    }
    println!();
    println!("\nbos={:?} eos={:?}", tok.bos_id(), tok.eos_id());
    Ok(())
}
