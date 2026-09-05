//! Real, standalone multi-process tensor-parallel generation -- the actual
//! acceptance test for the whole design (implementation plan Task 6), run
//! WITHOUT touching `crates/server`'s scheduler/API (that integration is
//! explicitly deferred -- see the plan's own Task 5 status). Every rank runs
//! the identical fixed prompt through the identical greedy decode loop in
//! lockstep (no request-level branching, so there's nothing for ranks to
//! disagree about), which is exactly what a real NCCL collective needs:
//! every rank participates in the same shape at the same moment. Only rank
//! 0 prints; every rank computes the same logits after its own row-parallel
//! all-reduces, so this isn't a shortcut -- every rank really did the work,
//! rank 0 is just the one this run bothers to print.
//!
//! Run as (2-rank example, adjust `CUDA_VISIBLE_DEVICES` to real free GPUs):
//!   RUN_ID=tp-run-1 TP_WORLD_SIZE=2 TP_RANK=0 CUDA_VISIBLE_DEVICES=0 \
//!     cargo run --release -p infero-model --features nccl --example tp_generate -- \
//!     /path/to/model.gguf "prompt" -n 32 &
//!   RUN_ID=tp-run-1 TP_WORLD_SIZE=2 TP_RANK=1 CUDA_VISIBLE_DEVICES=1 \
//!     cargo run --release -p infero-model --features nccl --example tp_generate -- \
//!     /path/to/model.gguf "prompt" -n 32 &
//!   wait
//!
//! `TP_WORLD_SIZE=1` (or unset) runs today's existing single-GPU path
//! (`Model::load_full_tp` with `tp_size=1` is byte-for-byte `load_full`) --
//! this is also how the single-GPU reference transcript for Task 6's diff
//! gets produced, with the same binary and the same prompt.
#![cfg(feature = "nccl")]

use std::io::Write;

use anyhow::{Context, Result};
use infero_cuda::Device;
use infero_gguf::Gguf;
use infero_model::tp::RankId;
use infero_model::{KvCacheQuant, Model, Sampler, SamplingParams};
use infero_tokenizer::Tokenizer;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().context("usage: tp_generate <model.gguf> [prompt] [-n N]")?;
    let mut prompt = String::new();
    let mut max_new = 32usize;
    let rest: Vec<String> = args.collect();
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
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
    if prompt.is_empty() {
        prompt = "The capital of France is".into();
    }

    let tp_rank: usize = std::env::var("TP_RANK").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let tp_size: usize = std::env::var("TP_WORLD_SIZE").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let run_id = std::env::var("RUN_ID").unwrap_or_else(|_| "tp_generate_default".to_string());
    let rank = RankId { pp_rank: 0, pp_size: 1, tp_rank, tp_size };

    let gguf = Gguf::open(&path)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let dev = Device::new(0)?; // CUDA_VISIBLE_DEVICES remaps this to the right physical GPU per rank
    let mut model = Model::load_full_tp(dev, &gguf, 2048, KvCacheQuant::F16, usize::MAX, 32, &rank, &run_id)?;

    let tokens = tokenizer.encode(&prompt, None, true);
    anyhow::ensure!(!tokens.is_empty(), "prompt tokenized to nothing");

    let mut session = model.new_session()?;
    let mut sampler = Sampler::new(SamplingParams::greedy());

    let logits = model.forward(&tokens, &mut session)?;
    let mut generated: Vec<u32> = Vec::with_capacity(max_new);
    let mut next = sampler.sample(logits, &generated);
    let mut detok = tokenizer.detokenizer();

    if tp_rank == 0 {
        println!("--- rank 0 output (tp_size={tp_size}) ---");
    }
    loop {
        if tokenizer.is_eog(next) || generated.len() >= max_new {
            break;
        }
        generated.push(next);
        if tp_rank == 0 {
            print!("{}", detok.push(next));
            std::io::stdout().flush()?;
        }
        if session.remaining() == 0 {
            break;
        }
        // Every rank calls forward() identically here -- required for the
        // NCCL all-reduces inside it to stay in lockstep across ranks. No
        // rank-specific branching: the whole point of this harness is a
        // single fixed prompt every rank processes the same way.
        let logits = model.forward(&[next], &mut session)?;
        next = sampler.sample(logits, &generated);
    }
    if tp_rank == 0 {
        print!("{}", detok.finish());
        println!();
        println!("TOKEN_IDS: {generated:?}");
    }
    Ok(())
}
