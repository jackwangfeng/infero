//! Generate text from a GGUF model on the GPU.
//!
//!     cargo run --release -p tuili-model --example generate -- \
//!         models/qwen2.5-0.5b-instruct-q8_0.gguf "Explain RoPE in one sentence."
//!
//! Options: `--raw` skips the chat template, `--greedy` fixes the sampler,
//! `-n <count>` caps the generation.

use std::io::Write;
use std::time::Instant;

use anyhow::{Context, Result};
use tuili_cuda::Device;
use tuili_gguf::Gguf;
use tuili_model::{KvCacheQuant, Model, Sampler, SamplingParams};
use tuili_tokenizer::{ChatMessage, Tokenizer};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_target(false)
        .init();

    let mut args = std::env::args().skip(1);
    let path = args.next().context(
        "usage: generate <model.gguf> [prompt] [--raw] [--greedy] [-n N] \
             [--kv-quant f16|tq2|tq4] [-ngl N]",
    )?;

    let mut prompt = String::new();
    let mut raw = false;
    let mut greedy = false;
    let mut max_new = 128usize;
    let mut kv_quant = KvCacheQuant::F16;
    let mut gpu_layers = usize::MAX;
    let mut rest: Vec<String> = args.collect();
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--raw" => raw = true,
            "--greedy" => greedy = true,
            "--kv-quant" => {
                i += 1;
                kv_quant = KvCacheQuant::parse(rest.get(i).map(String::as_str).unwrap_or("f16"))?;
            }
            "--gpu-layers" | "-ngl" => {
                i += 1;
                gpu_layers = rest
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(usize::MAX);
            }
            "-n" => {
                i += 1;
                max_new = rest.get(i).and_then(|v| v.parse().ok()).unwrap_or(max_new);
            }
            other => {
                if !prompt.is_empty() {
                    prompt.push(' ');
                }
                prompt.push_str(other);
            }
        }
        i += 1;
    }
    rest.clear();
    if prompt.is_empty() {
        prompt = "Give me a short introduction to large language models.".into();
    }

    let gguf = Gguf::open(&path)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let dev = Device::new(0)?;
    let mut model = Model::load_with(dev, &gguf, 4096, kv_quant, gpu_layers)?;

    let text = if raw {
        prompt.clone()
    } else {
        let template = tokenizer
            .chat_template()
            .context("model has no chat template; pass --raw")?;
        template.render(&[ChatMessage::user(&prompt)], true)?
    };
    let tokens = tokenizer.encode(&text, None, true);
    anyhow::ensure!(!tokens.is_empty(), "prompt tokenized to nothing");

    println!("\x1b[2m--- prompt ({} tokens) ---\x1b[0m", tokens.len());
    print!("{prompt}\n\n");
    std::io::stdout().flush()?;

    let (vram, host) = model.weight_bytes();
    println!(
        "\x1b[2mweights  : {:.0} MiB in VRAM, {:.0} MiB offloaded ({} of {} layers streamed)\x1b[0m",
        vram as f64 / (1 << 20) as f64,
        host as f64 / (1 << 20) as f64,
        model.n_offloaded_layers(),
        model.config().n_layers,
    );
    let mut session = model.new_session()?;
    println!(
        "\x1b[2mkv cache : {}, {:.2} bits/channel, {:.1} MiB for {} positions\x1b[0m",
        model.kv_quant(),
        model.kv_quant().bits_per_channel(model.config().d_head),
        session.bytes() as f64 / (1 << 20) as f64,
        session.max_seq(),
    );
    let mut sampler = Sampler::new(if greedy {
        SamplingParams::greedy()
    } else {
        SamplingParams::default()
    });

    // Prefill: the whole prompt in one call, logits for its last position.
    let t0 = Instant::now();
    let logits = model.forward(&tokens, &mut session)?;
    let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut generated: Vec<u32> = Vec::with_capacity(max_new);
    let mut next = sampler.sample(logits, &generated);
    let mut detok = tokenizer.detokenizer();

    let t1 = Instant::now();
    let mut n_decoded = 0usize;
    loop {
        if tokenizer.is_eog(next) || generated.len() >= max_new {
            break;
        }
        generated.push(next);
        print!("{}", detok.push(next));
        std::io::stdout().flush()?;

        if session.remaining() == 0 {
            eprintln!("\n\x1b[33m[context full]\x1b[0m");
            break;
        }
        let logits = model.forward(&[next], &mut session)?;
        n_decoded += 1;
        next = sampler.sample(logits, &generated);
    }
    print!("{}", detok.finish());
    let decode_s = t1.elapsed().as_secs_f64();

    println!("\n");
    println!(
        "\x1b[2mprefill  {:>7.1} ms  ({:.1} tok/s over {} tokens)\x1b[0m",
        prefill_ms,
        tokens.len() as f64 / (prefill_ms / 1000.0),
        tokens.len()
    );
    println!(
        "\x1b[2mdecode   {:>7.1} ms  ({:.1} tok/s over {} tokens)\x1b[0m",
        decode_s * 1000.0,
        n_decoded as f64 / decode_s.max(1e-9),
        n_decoded
    );
    Ok(())
}
