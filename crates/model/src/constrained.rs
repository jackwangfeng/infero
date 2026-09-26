//! Schema-constrained generation: a one-shot decode loop that masks each
//! step's logits down to whatever [`crate::json_grammar::Cursor`] still
//! accepts, for `docs/keel-integration.md`'s "2. Schema-constrained
//! decoding" section.
//!
//! Not wired into the scheduler's continuous batching -- like
//! [`crate::embed::embed_batch`] and [`crate::rerank::rerank_batch`], this
//! runs to completion on its own throwaway session. Unlike those, it is a
//! real multi-step decode loop rather than one forward pass, so it is
//! slower per request and the cost is more front-and-center: every step
//! tests every vocabulary token's bytes against the cursor before sampling,
//! which is the real, measured, not-yet-optimized cost this module's own
//! module-level doc in `crates/server/src/routes.rs`'s handler cites a
//! number for.

use anyhow::{Context, Result, ensure};

use crate::json_grammar::{Cursor, Schema};
use crate::{BatchItemKind, Model, Sampler, SamplingParams};

/// Generate token ids constrained to `schema`, stopping the instant a valid
/// completion exists (not at the model's own EOS, which a schema-constrained
/// model has no natural reason to emit mid-structure).
///
/// `vocab_bytes[i]` must be token id `i`'s raw decoded bytes -- built once
/// per tokenizer by the caller (`crates/server/src/engine.rs`'s
/// `Engine::vocab_bytes`) and shared across requests, since building it is
/// itself a full vocabulary pass.
pub fn generate_json(
    model: &mut Model,
    vocab_bytes: &[Vec<u8>],
    prompt_tokens: &[u32],
    schema: Schema,
    max_tokens: usize,
    params: SamplingParams,
) -> Result<Vec<u32>> {
    ensure!(!prompt_tokens.is_empty(), "generate_json needs at least one prompt token");
    let mut session = model.new_session().context("allocating the constrained-decode session")?;
    ensure!(
        session.remaining() >= prompt_tokens.len(),
        "prompt of {} tokens does not fit this model's context ({} remaining)",
        prompt_tokens.len(),
        session.remaining()
    );

    let mut logits = model.forward(prompt_tokens, BatchItemKind::Prefill, &mut session)?.to_vec();
    let mut cursor = Cursor::new(schema);
    let mut sampler = Sampler::new(params);
    let mut generated: Vec<u32> = Vec::new();

    while !cursor.is_complete() {
        ensure!(
            generated.len() < max_tokens,
            "generated {max_tokens} tokens without the schema's root value closing"
        );
        ensure!(
            session.remaining() > 0,
            "context exhausted before the schema's root value closed"
        );

        // Default-deny: anything past `vocab_bytes.len()` (padding rows a
        // tokenizer's real vocab can leave in a model's configured
        // `vocab_size`) is never legal, rather than left at its real logit
        // value because the masking loop below never reached it.
        let mut masked = vec![f32::NEG_INFINITY; logits.len()];
        for (id, bytes) in vocab_bytes.iter().enumerate() {
            if id < masked.len() && cursor.accepts(bytes) {
                masked[id] = logits[id];
            }
        }

        let next = sampler.sample(&masked, &generated);
        for &b in &vocab_bytes[next as usize] {
            let accepted = cursor.feed(b);
            debug_assert!(accepted, "sampled a token `accepts` had already approved");
        }
        generated.push(next);
        logits = model.forward(&[next], BatchItemKind::Decode, &mut session)?.to_vec();
    }
    Ok(generated)
}
