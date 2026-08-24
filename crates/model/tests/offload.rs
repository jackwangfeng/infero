//! CPU offload: layer weights held in page-locked host memory and streamed in.
//!
//! The defining property is that offload is *not* an approximation. The same
//! bytes reach the same kernels in the same order; only their route does. So
//! the headline test is exact equality of logits, not a tolerance.

use std::path::PathBuf;

use anyhow::Result;
use tuili_cuda::Device;
use tuili_gguf::Gguf;
use tuili_model::{KvCacheQuant, Model, Sampler, SamplingParams};
use tuili_tokenizer::Tokenizer;

const PROMPT: &str = "The capital of France is";

fn model_path() -> Option<PathBuf> {
    let p = std::env::var("TUILI_TEST_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../models/qwen2.5-0.5b-instruct-q8_0.gguf")
        });
    if !p.exists() {
        eprintln!("skipping: {} not downloaded", p.display());
        return None;
    }
    Some(p)
}

fn load(n_gpu_layers: usize) -> Result<Option<(Model, Tokenizer)>> {
    let Some(path) = model_path() else {
        return Ok(None);
    };
    let gguf = Gguf::open(&path)?;
    let tok = Tokenizer::from_gguf(&gguf)?;
    let model = Model::load_with(
        Device::new(0)?,
        &gguf,
        1024,
        KvCacheQuant::F16,
        n_gpu_layers,
    )?;
    Ok(Some((model, tok)))
}

/// One test in this file at a time; see the note in `batching.rs`. A decode
/// step captures a CUDA graph, and capture dies if another thread allocates on
/// the same context.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

macro_rules! setup {
    ($n:expr) => {
        match load($n)? {
            Some(v) => v,
            None => return Ok(()),
        }
    };
}

/// Offload moves bytes, not arithmetic. Anything short of exact equality here
/// would mean a staging buffer is being read at the wrong offset or before its
/// transfer landed.
#[test]
fn offloaded_logits_are_identical_to_resident_ones() -> Result<()> {
    let _gpu = gpu_lock();
    let (mut resident, tok) = setup!(usize::MAX);
    let ids = tok.encode(PROMPT, Some(false), false);

    let mut session = resident.new_session()?;
    let want: Vec<f32> = resident.forward(&ids, &mut session)?.to_vec();
    drop(resident);

    for n_gpu in [0usize, 1, 12, 23] {
        let (mut model, _) = setup!(n_gpu);
        let mut session = model.new_session()?;
        let got: Vec<f32> = model.forward(&ids, &mut session)?.to_vec();
        assert_eq!(
            got.len(),
            want.len(),
            "-ngl {n_gpu}: logit vector changed length"
        );
        let differing = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(
            differing,
            0,
            "-ngl {n_gpu}: {differing} of {} logits differ from the resident run",
            want.len()
        );
    }
    Ok(())
}

/// The same must hold token by token, which is where a staging slot could be
/// recycled a step too early.
#[test]
fn incremental_decoding_is_identical_too() -> Result<()> {
    let _gpu = gpu_lock();
    let (mut resident, tok) = setup!(usize::MAX);
    let ids = tok.encode(PROMPT, Some(false), false);

    let run = |model: &mut Model| -> Result<Vec<u32>> {
        let mut session = model.new_session()?;
        let mut sampler = Sampler::new(SamplingParams::greedy());
        let mut out = Vec::new();
        let mut logits: Vec<f32> = model.forward(&ids, &mut session)?.to_vec();
        for _ in 0..16 {
            let next = sampler.sample(&logits, &out);
            out.push(next);
            logits = model.forward(&[next], &mut session)?.to_vec();
        }
        Ok(out)
    };

    let want = run(&mut resident)?;
    drop(resident);

    let (mut offloaded, _) = setup!(0);
    let got = run(&mut offloaded)?;
    assert_eq!(
        got,
        want,
        "streamed weights produced a different continuation: {:?} vs {:?}",
        tok.decode(&got, true),
        tok.decode(&want, true)
    );
    Ok(())
}

#[test]
fn vram_falls_as_layers_move_to_the_host() -> Result<()> {
    let _gpu = gpu_lock();
    let Some(path) = model_path() else {
        return Ok(());
    };
    let gguf = Gguf::open(&path)?;
    let dev = Device::new(0)?;

    let mut previous = usize::MAX;
    eprintln!(
        "\n  {:>4}  {:>10}  {:>12}",
        "ngl", "vram MiB", "offload MiB"
    );
    for n_gpu in [24usize, 16, 8, 0] {
        let model = Model::load_with(dev.clone(), &gguf, 512, KvCacheQuant::F16, n_gpu)?;
        let (vram, host) = model.weight_bytes();
        eprintln!(
            "  {n_gpu:>4}  {:>10}  {:>12}",
            vram / (1 << 20),
            host / (1 << 20)
        );
        assert!(
            vram < previous,
            "-ngl {n_gpu} did not reduce VRAM: {vram} vs {previous}"
        );
        assert_eq!(
            model.n_offloaded_layers(),
            model.config().n_layers - n_gpu.min(model.config().n_layers)
        );
        previous = vram;
    }
    Ok(())
}

/// One transfer per offloaded layer per forward pass, no more: a prefetch that
/// re-issued the layer it just fetched would double the PCIe traffic and still
/// produce correct output, so it needs its own check.
#[test]
fn each_offloaded_layer_is_transferred_exactly_once_per_pass() -> Result<()> {
    let _gpu = gpu_lock();
    let (mut model, tok) = setup!(4);
    let offloaded = model.n_offloaded_layers();
    assert!(offloaded > 0);

    let ids = tok.encode(PROMPT, Some(false), false);
    let mut session = model.new_session()?;

    // The prompt is shorter than one prefill chunk, so this is a single pass.
    assert!(ids.len() <= model.batch_tokens());
    model.forward(&ids, &mut session)?;
    assert_eq!(model.weight_transfers(), offloaded as u64);

    // One decode step is another pass over the same layers.
    model.forward(&[ids[0]], &mut session)?;
    assert_eq!(model.weight_transfers(), 2 * offloaded as u64);
    Ok(())
}

#[test]
fn a_fully_offloaded_model_still_generates() -> Result<()> {
    let _gpu = gpu_lock();
    let (mut model, tok) = setup!(0);
    assert_eq!(model.n_offloaded_layers(), model.config().n_layers);

    let prompt = tok.encode("The three primary colors are red,", Some(false), false);
    let mut session = model.new_session()?;
    let mut sampler = Sampler::new(SamplingParams::greedy());
    let mut out = Vec::new();
    let mut logits: Vec<f32> = model.forward(&prompt, &mut session)?.to_vec();
    for _ in 0..24 {
        let next = sampler.sample(&logits, &out);
        if tok.is_eog(next) {
            break;
        }
        out.push(next);
        logits = model.forward(&[next], &mut session)?.to_vec();
    }

    let text = tok.decode(&out, true);
    eprintln!("  fully offloaded: {text:?}");
    assert!(
        text.contains("blue") || text.contains("yellow"),
        "expected the other primary colors, got {text:?}"
    );
    Ok(())
}
