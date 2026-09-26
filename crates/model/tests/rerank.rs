//! `docs/keel-integration.md`'s `/v1/rerank` requirements, run against a real
//! Qwen3-Reranker-0.6B checkpoint: the model card's own published
//! (query, document) pairs, checked against its own published relative
//! ordering (a relevant document scores far above an irrelevant one), not
//! just against a shape assertion.

use std::path::{Path, PathBuf};

use infero_cuda::Device;
use infero_model::{KvCacheQuant, Model};
use infero_tokenizer::Tokenizer;

fn reranker_checkpoint_dir() -> Option<PathBuf> {
    let p = std::env::var("INFERO_TEST_RERANKER_SAFETENSORS").ok().map(PathBuf::from)?;
    p.exists().then_some(p)
}

fn load(dir: &Path) -> anyhow::Result<(Model, Tokenizer)> {
    let dev = Device::new(0)?;
    let tokenizer = Tokenizer::from_hf_dir(dir.to_str().unwrap())?;
    let model = Model::load_awq(dev, dir.to_str().unwrap(), 2048, KvCacheQuant::F16, 8)?;
    Ok((model, tokenizer))
}

fn score(
    model: &mut Model,
    tokenizer: &Tokenizer,
    query: &str,
    documents: &[&str],
) -> anyhow::Result<Vec<f32>> {
    let prefix_tokens = tokenizer.encode(infero_model::rerank::PREFIX, Some(false), true);
    let suffix_tokens = tokenizer.encode(infero_model::rerank::SUFFIX, Some(false), true);
    let bodies: Vec<Vec<u32>> = documents
        .iter()
        .map(|doc| {
            let body = infero_model::rerank::format_pair(infero_model::rerank::DEFAULT_INSTRUCTION, query, doc);
            tokenizer.encode(&body, Some(false), false)
        })
        .collect();
    let yes = tokenizer.encode("yes", Some(false), false);
    let no = tokenizer.encode("no", Some(false), false);
    assert_eq!(yes.len(), 1, "\"yes\" must be a single token for this recipe");
    assert_eq!(no.len(), 1, "\"no\" must be a single token for this recipe");
    infero_model::rerank::rerank_batch(model, &prefix_tokens, &bodies, &suffix_tokens, yes[0], no[0])
}

/// The model card's own published example
/// (`Qwen/Qwen3-Reranker-0.6B`'s README): a relevant document scores near 1,
/// an irrelevant one near 0.
#[test]
fn a_relevant_document_scores_far_above_an_irrelevant_one() {
    let Some(dir) = reranker_checkpoint_dir() else {
        eprintln!(
            "skipping: set INFERO_TEST_RERANKER_SAFETENSORS to a real Qwen3-Reranker-style \
             checkpoint directory"
        );
        return;
    };
    let (mut model, tokenizer) = load(&dir).expect("loading the reranker checkpoint");

    let scores = score(
        &mut model,
        &tokenizer,
        "What is the capital of China?",
        &[
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other. It gives weight to \
             physical objects and is responsible for the movement of planets around the sun.",
        ],
    )
    .expect("rerank_batch");

    assert!(scores[0] > 0.9, "relevant document scored {}", scores[0]);
    assert!(scores[1] < 0.1, "irrelevant document scored {}", scores[1]);
    assert!(scores[0] > scores[1]);
}

/// The query changes which document is "relevant" -- the score is not just
/// "document 0 always wins", it tracks the actual (query, document) pair.
#[test]
fn relevance_tracks_the_query_not_a_fixed_document() {
    let Some(dir) = reranker_checkpoint_dir() else {
        eprintln!(
            "skipping: set INFERO_TEST_RERANKER_SAFETENSORS to a real Qwen3-Reranker-style \
             checkpoint directory"
        );
        return;
    };
    let (mut model, tokenizer) = load(&dir).expect("loading the reranker checkpoint");

    let documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other.",
    ];
    let capital_scores =
        score(&mut model, &tokenizer, "What is the capital of China?", &documents).expect("rerank_batch");
    let gravity_scores = score(&mut model, &tokenizer, "Explain gravity", &documents).expect("rerank_batch");

    assert!(capital_scores[0] > capital_scores[1], "{capital_scores:?}");
    assert!(gravity_scores[1] > gravity_scores[0], "{gravity_scores:?}");
}

#[test]
fn a_batch_cap_violation_is_a_clean_error_not_a_panic() {
    let Some(dir) = reranker_checkpoint_dir() else {
        eprintln!(
            "skipping: set INFERO_TEST_RERANKER_SAFETENSORS to a real Qwen3-Reranker-style \
             checkpoint directory"
        );
        return;
    };
    let (mut model, tokenizer) = load(&dir).expect("loading the reranker checkpoint");

    let documents: Vec<&str> = (0..infero_model::rerank::MAX_BATCH + 1).map(|_| "x").collect();
    let err = score(&mut model, &tokenizer, "q", &documents).expect_err(
        "a batch over MAX_BATCH must be refused, not silently truncated",
    );
    assert!(err.to_string().contains("batch cap"), "{err}");
}
