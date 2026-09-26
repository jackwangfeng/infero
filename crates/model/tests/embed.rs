//! `docs/keel-integration.md`'s `/v1/embeddings` requirements, run against a
//! real Qwen3-Embedding-0.6B checkpoint: last-token pooling through
//! `Model::hidden_states_host`, server-owned L2 normalization
//! (`infero_model::embed::normalize`), and the semantic margin test the doc
//! itself says is worth stealing for this test suite.

use std::path::{Path, PathBuf};

use infero_cuda::Device;
use infero_model::{KvCacheQuant, Model};
use infero_tokenizer::Tokenizer;

fn embedding_checkpoint_dir() -> Option<PathBuf> {
    let p = std::env::var("INFERO_TEST_EMBEDDING_SAFETENSORS").ok().map(PathBuf::from)?;
    p.exists().then_some(p)
}

fn load(dir: &Path) -> anyhow::Result<(Model, Tokenizer)> {
    let dev = Device::new(0)?;
    let tokenizer = Tokenizer::from_hf_dir(dir.to_str().unwrap())?;
    let model = Model::load_awq(dev, dir.to_str().unwrap(), 2048, KvCacheQuant::F16, 8)?;
    Ok((model, tokenizer))
}

fn tokenize(tokenizer: &Tokenizer, texts: &[&str]) -> Vec<Vec<u32>> {
    texts.iter().map(|t| tokenizer.encode(t, None, true)).collect()
}

#[test]
fn embeddings_are_l2_normalized_and_dimensioned_like_the_checkpoint() {
    let Some(dir) = embedding_checkpoint_dir() else {
        eprintln!(
            "skipping: set INFERO_TEST_EMBEDDING_SAFETENSORS to a real Qwen3-Embedding-style \
             checkpoint directory"
        );
        return;
    };
    let (mut model, tokenizer) = load(&dir).expect("loading the embedding checkpoint");
    let d_model = model.config().d_model;

    let token_lists = tokenize(&tokenizer, &["a red running shoe for men", "a bag of coffee beans"]);
    let embeddings = infero_model::embed::embed_batch(&mut model, &token_lists).expect("embed_batch");

    assert_eq!(embeddings.len(), 2);
    for (i, row) in embeddings.iter().enumerate() {
        assert_eq!(row.len(), d_model, "row {i} width");
        let norm = (row.iter().map(|x| (*x as f64).powi(2)).sum::<f64>()).sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "row {i} norm {norm}, expected ~1.0");
    }
}

/// `docs/keel-integration.md`'s own acceptance test #2: three texts, two
/// related and one not, and the cosine gap between (near, near) and
/// (near, far) has to clear a real margin -- a shape-identical fake (right
/// dim, unit norm, right `model` string) fails only this check, which is why
/// the doc calls it "a much harder bar than 'the server responded'".
#[test]
fn near_texts_score_a_real_margin_above_far_ones() {
    let Some(dir) = embedding_checkpoint_dir() else {
        eprintln!(
            "skipping: set INFERO_TEST_EMBEDDING_SAFETENSORS to a real Qwen3-Embedding-style \
             checkpoint directory"
        );
        return;
    };
    let (mut model, tokenizer) = load(&dir).expect("loading the embedding checkpoint");

    let token_lists = tokenize(
        &tokenizer,
        &[
            "a red running shoe for men",
            "a red running shoe for women",
            "a bag of coffee beans",
        ],
    );
    let e = infero_model::embed::embed_batch(&mut model, &token_lists).expect("embed_batch");

    let cos = |a: &[f32], b: &[f32]| -> f64 {
        a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
    };
    let near = cos(&e[0], &e[1]);
    let far_a = cos(&e[0], &e[2]);
    let far_b = cos(&e[1], &e[2]);
    let margin = near - far_a.max(far_b);

    assert!(
        margin > 0.10,
        "near-near {near:.4} vs near-far {:.4}/{:.4}: margin {margin:.4}, expected > 0.10",
        far_a,
        far_b
    );
}

#[test]
fn a_batch_cap_violation_is_a_clean_error_not_a_panic() {
    let Some(dir) = embedding_checkpoint_dir() else {
        eprintln!(
            "skipping: set INFERO_TEST_EMBEDDING_SAFETENSORS to a real Qwen3-Embedding-style \
             checkpoint directory"
        );
        return;
    };
    let (mut model, tokenizer) = load(&dir).expect("loading the embedding checkpoint");

    let texts: Vec<&str> = (0..infero_model::embed::MAX_BATCH + 1).map(|_| "x").collect();
    let token_lists = tokenize(&tokenizer, &texts);
    let err = infero_model::embed::embed_batch(&mut model, &token_lists)
        .expect_err("a batch over MAX_BATCH must be refused, not silently truncated");
    assert!(err.to_string().contains("batch cap"), "{err}");
}
