//! `INFERO_SPLIT_MIXED_BATCH`: the mixed-batch attention dispatch split.
//!
//! With the flag on, a batch that mixes a decode item with a prefill item no
//! longer goes through one whole-batch `attn_decode` call. The decode items
//! take one batched call and every prefill item takes its own, each with its
//! own `run_base`/`run_tokens` slice of the flat per-token buffers and its own
//! `kv_len`. Two things can go silently wrong there and neither one crashes:
//!
//!   1. an off-by-one in a per-item buffer offset, which mixes one sequence's
//!      query rows into another's output, and
//!   2. reusing the batch-wide `kv_len` for a per-item call instead of that
//!      item's own extent (see `AttnRun`'s doc comment in `crates/model/src/
//!      lib.rs` for the three distinct ways that goes wrong).
//!
//! These tests hold the split to the only property that rules both out: a
//! sequence must not be able to tell who else was in its batch, and turning
//! the flag off must reproduce the old dispatch's answers.
//!
//! Note on the model these run against: `qwen2.5-0.5b` is `d_head = 64`, so
//! neither `attn_prefill_decoupled6_f16acc` (`d_head == 256` only) nor the
//! `flash_attn2` backend (`d_head == 256` and >= 4096 rows) can serve its
//! prefill runs -- every run here lands on `attn_decode`, which reads the
//! `attn_partial` split-K scratch. `attn_partial` is `max_logit_rows +
//! MIN_PREFILL_RUN` tokens wide once the flag is on, so these tests keep every
//! single run inside that bound on purpose (`MAX_LOGIT_ROWS`/`CHUNK` below); a
//! wider run would be refused by `ensure_partial_fits` rather than corrupt
//! memory, which is itself the correct behavior but not what these tests are
//! for.

use std::path::PathBuf;

use anyhow::Result;
use infero_model::{BatchItem, BatchItemKind, KvCacheQuant, Model};
use infero_tokenizer::Tokenizer;

/// `Model::from_parts` clamps this to at least 1 and uses it for
/// `attn_partial_bound`; 64 leaves `64 + MIN_PREFILL_RUN(8) = 72` tokens of
/// partial scratch, which every run below stays inside.
const MAX_LOGIT_ROWS: usize = 64;
/// The widest run any batch here issues -- comfortably inside the 72 above,
/// and wide enough (`>= MIN_PREFILL_RUN`) to be a real prefill chunk rather
/// than a decode-shaped one.
const CHUNK: usize = 48;

fn model_path() -> Option<PathBuf> {
    let p = std::env::var("INFERO_TEST_GGUF")
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

/// Load with `INFERO_SPLIT_MIXED_BATCH` set to `split` for the duration of the
/// load, which is when `Model::from_parts` resolves it (and sizes
/// `attn_partial` off it). The variable is removed again immediately: nothing
/// reads it after load, and leaving it set would leak into the next test.
///
/// Safety: these tests are serialized by `gpu_lock()` below, and no other test
/// in this binary reads or writes this variable.
fn load(split: &str) -> Result<Option<(Model, Tokenizer)>> {
    let Some(path) = model_path() else {
        return Ok(None);
    };
    let gguf = infero_gguf::Gguf::open(&path)?;
    let tok = Tokenizer::from_gguf(&gguf)?;
    unsafe { std::env::set_var("INFERO_SPLIT_MIXED_BATCH", split) };
    let model = Model::load_full(
        infero_cuda::Device::new(0)?,
        &gguf,
        1024,
        KvCacheQuant::F16,
        usize::MAX,
        MAX_LOGIT_ROWS,
    );
    unsafe { std::env::remove_var("INFERO_SPLIT_MIXED_BATCH") };
    let model = model?;
    assert_eq!(
        model.split_mixed_batch(),
        split == "1",
        "INFERO_SPLIT_MIXED_BATCH={split} did not resolve as expected -- the rest \
         of this test would be measuring nothing"
    );
    Ok(Some((model, tok)))
}

/// One test in this file at a time: each builds its own `Model`, a decode step
/// captures a CUDA graph, and capture is invalidated by an allocation from any
/// other thread on the same context. Same reasoning as `tests/batching.rs`.
/// These tests additionally mutate a process-wide environment variable across
/// a load, which is its own reason to serialize.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold((0u32, f32::NEG_INFINITY), |b, (i, &x)| {
            if x > b.1 { (i as u32, x) } else { b }
        })
        .0
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb + 1e-30)
}

/// A deterministic pseudo-prompt: real token ids, no tokenizer round trip, so
/// the two sides of every comparison are byte-identical inputs.
fn tokens(vocab: usize, seed: u32, n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| ((seed.wrapping_mul(2654435761).wrapping_add(i as u32 * 40503)) % vocab as u32).max(1))
        .collect()
}

/// Drive one sequence `prior` tokens deep in `CHUNK`-sized single-item prefill
/// passes, then leave it ready to decode. Chunked rather than fed in one go so
/// that no single run exceeds the shrunk `attn_partial` (see the module note).
fn prime(model: &mut Model, pool: &mut infero_model::KvPool, seq: infero_model::SeqId, prior: &[u32]) -> Result<()> {
    for chunk in prior.chunks(CHUNK) {
        let item = BatchItem::without_logits(seq, chunk, BatchItemKind::Prefill);
        model.forward_batch_device(std::slice::from_ref(&item), pool)?;
    }
    Ok(())
}

/// The adversarial construction: a sequence 480 tokens deep decodes its next
/// token in the same batch as a *different*, freshly admitted sequence's first
/// 48-token prefill chunk.
///
/// The batch-wide `kv_len` here is 481; the prefill item's own is 48. The
/// prefill item's logits must be exactly what it gets on its own, with no
/// concurrent decode item at all -- it must not be able to tell that the other
/// sequence is 480 tokens into generation, whether through a wrong `kv_len` or
/// a wrong buffer offset.
#[test]
fn a_prefill_item_cannot_tell_a_decode_item_shared_its_batch() -> Result<()> {
    let _gpu = gpu_lock();
    let Some((mut model, _tok)) = load("1")? else {
        return Ok(());
    };
    let vocab = model.config().vocab_size;
    let prompt = tokens(vocab, 7, CHUNK);
    let long = tokens(vocab, 99, 480);

    // Solo: the prefill chunk, alone in its own batch and its own pool.
    let solo = {
        let mut pool = model.new_pool(4096, 4)?;
        let seq = pool.alloc().expect("fresh pool had no rows");
        let item = BatchItem::new(seq, &prompt, BatchItemKind::Prefill);
        model.forward_batch(std::slice::from_ref(&item), &mut pool)?[..vocab].to_vec()
    };

    // Mixed: the same prefill chunk, batched behind a long-running sequence's
    // decode token.
    let mixed = {
        let mut pool = model.new_pool(4096, 4)?;
        let decoder = pool.alloc().expect("fresh pool had no rows");
        let fresh = pool.alloc().expect("fresh pool had no second row");
        prime(&mut model, &mut pool, decoder, &long)?;
        assert_eq!(pool.len(decoder), long.len());
        let next = [long[0]];
        let items = [
            BatchItem::new(decoder, &next, BatchItemKind::Decode),
            BatchItem::new(fresh, &prompt, BatchItemKind::Prefill),
        ];
        let out = model.forward_batch(&items, &mut pool)?;
        // Row 0 is the decode item's logits, row 1 the prefill item's --
        // `logit_rows` is built in item order.
        out[vocab..2 * vocab].to_vec()
    };

    let cos = cosine(&mixed, &solo);
    eprintln!(
        "  prefill argmax in mixed batch {} vs solo {}, cosine {cos:.12}",
        argmax(&mixed),
        argmax(&solo)
    );
    assert_eq!(
        argmax(&mixed),
        argmax(&solo),
        "the prefill item predicted differently with an unrelated 480-token \
         decode sequence in its batch"
    );
    // Tighter than the 0.9995-ish tolerance `tests/batching.rs` uses, and
    // deliberately so: both sides here take the same kernels at the same
    // widths, so the correct answer is not merely close, it is identical to
    // every digit this prints. The threshold is set from two real measured
    // numbers on this model rather than picked:
    //   - correct code:                                   1.000000000000
    //   - `AttnRun::kv_len` replaced by the batch-wide max: 0.999949357...
    // i.e. handing this prefill item the unrelated decode sequence's 481 in
    // place of its own 48 moves its logits by 5e-5 of cosine, which this
    // catches and the looser tolerance would not. What it does NOT prove is
    // that the batch-wide `kv_len` is harmless anywhere else: on this
    // `d_head = 64` model every run lands on `attn_decode`, which masks keys
    // by `positions`, so an over-long `kv_len` is mostly wasted work here. On
    // the production `d_head = 256` shape the same mistake reaches
    // `flash_attn2`, which takes `kv_len` as the real key-sequence length and
    // aligns its causal mask against it — wrong output, not wasted work. That
    // hazard is pinned by `attn_dispatch_tests::
    // prefill_run_carries_its_own_kv_len_not_the_batch_max` in
    // `crates/model/src/lib.rs`, which is deterministic and model-independent.
    assert!(
        cos > 0.999999,
        "prefill logits moved (cosine {cos:.12}) because of an unrelated item in the batch"
    );
    Ok(())
}

/// Two simultaneous prefill items -- `plan()`'s other ordinary mixed shape,
/// and the one that has *no* decode prefix at all, so both items are dispatched
/// as their own runs at their own offsets. Each must match its own solo run.
#[test]
fn two_simultaneous_prefills_each_match_their_solo_run() -> Result<()> {
    let _gpu = gpu_lock();
    let Some((mut model, _tok)) = load("1")? else {
        return Ok(());
    };
    let vocab = model.config().vocab_size;
    // Deliberately different widths, so a run that used its neighbour's
    // `run_tokens` (or its own at the wrong offset) shows up.
    let a = tokens(vocab, 11, CHUNK);
    let b = tokens(vocab, 23, CHUNK - 17);

    let mut solo = Vec::new();
    for p in [&a, &b] {
        let mut pool = model.new_pool(4096, 4)?;
        let seq = pool.alloc().expect("fresh pool had no rows");
        let item = BatchItem::new(seq, p, BatchItemKind::Prefill);
        solo.push(model.forward_batch(std::slice::from_ref(&item), &mut pool)?[..vocab].to_vec());
    }

    let mut pool = model.new_pool(4096, 4)?;
    let sa = pool.alloc().expect("fresh pool had no rows");
    let sb = pool.alloc().expect("fresh pool had no second row");
    let items = [
        BatchItem::new(sa, &a, BatchItemKind::Prefill),
        BatchItem::new(sb, &b, BatchItemKind::Prefill),
    ];
    let both = model.forward_batch(&items, &mut pool)?.to_vec();

    for (i, want) in solo.iter().enumerate() {
        let got = &both[i * vocab..(i + 1) * vocab];
        let cos = cosine(got, want);
        eprintln!("  prefill {i}: argmax {} vs {}, cosine {cos:.12}", argmax(got), argmax(want));
        assert_eq!(argmax(got), argmax(want), "prefill {i} predicted differently in a two-prefill batch");
        assert!(cos > 0.9999, "prefill {i}: cosine {cos:.12}");
    }
    Ok(())
}

/// A run too wide for the shrunk `attn_partial` must be refused, loudly.
///
/// This is the one real limitation of turning the flag on: `attn_partial` is
/// sized for the *decode* subgroup (`max_logit_rows + MIN_PREFILL_RUN`), which
/// is only enough for everything else if the model's wide prefill runs reach a
/// kernel that does not use the split-K scratch at all --
/// `attn_prefill_decoupled6_f16acc` (`d_head == 256`) or the `flash_attn2`
/// backend. This 0.5B model (`d_head == 64`) has neither, so a wide run here
/// falls to `attn_decode`, which does use it.
///
/// The behavior that matters is that this is an error and not a silent write
/// past the end of the buffer -- the failure mode that surfaces several
/// kernels later as plausible-looking corrupted output. `INFERO_SPLIT_MIXED_
/// BATCH=0` (the default) restores the full-width buffer and this shape runs
/// fine, which is what the error message says.
#[test]
fn a_run_wider_than_the_shrunk_partial_is_refused_not_corrupted() -> Result<()> {
    let _gpu = gpu_lock();
    let Some((mut model, _tok)) = load("1")? else {
        return Ok(());
    };
    let vocab = model.config().vocab_size;
    // Well past `MAX_LOGIT_ROWS + MIN_PREFILL_RUN == 72`.
    let wide = tokens(vocab, 5, 200);
    let mut pool = model.new_pool(4096, 4)?;
    let seq = pool.alloc().expect("fresh pool had no rows");
    let item = BatchItem::new(seq, &wide, BatchItemKind::Prefill);
    let err = model
        .forward_batch(std::slice::from_ref(&item), &mut pool)
        .expect_err("a 200-token run must not be allowed into a 72-token partial buffer");
    let msg = format!("{err:#}");
    eprintln!("  refused as expected: {msg}");
    assert!(
        msg.contains("attn_partial") && msg.contains("INFERO_SPLIT_MIXED_BATCH"),
        "the refusal must name the buffer and the escape hatch, got: {msg}"
    );
    Ok(())
}

/// The rollback path: `INFERO_SPLIT_MIXED_BATCH=0` must reproduce the old
/// single-call dispatch's answers, on the very batch shapes the split changes.
///
/// Both sides run the identical token sequence through identical pools, so the
/// only thing that differs between them is how attention was internally
/// dispatched. This is the escape hatch the design doc designates for
/// production, so it is worth a test that would notice if the "off" path had
/// been disturbed by the refactor that introduced the "on" one.
#[test]
fn split_off_reproduces_the_old_dispatch() -> Result<()> {
    let _gpu = gpu_lock();
    if model_path().is_none() {
        return Ok(());
    }

    // The same three shapes, run under each setting.
    fn scenarios(model: &mut Model) -> Result<Vec<Vec<f32>>> {
        let vocab = model.config().vocab_size;
        let prompt = tokens(vocab, 7, CHUNK);
        let long = tokens(vocab, 99, 480);
        let b = tokens(vocab, 23, CHUNK - 17);
        let mut out = Vec::new();

        // (a) a plain single-sequence prefill
        {
            let mut pool = model.new_pool(4096, 4)?;
            let seq = pool.alloc().expect("fresh pool had no rows");
            let item = BatchItem::new(seq, &prompt, BatchItemKind::Prefill);
            out.push(model.forward_batch(std::slice::from_ref(&item), &mut pool)?[..vocab].to_vec());
        }
        // (b) one decode item plus one prefill item
        {
            let mut pool = model.new_pool(4096, 4)?;
            let decoder = pool.alloc().expect("fresh pool had no rows");
            let fresh = pool.alloc().expect("fresh pool had no second row");
            prime(model, &mut pool, decoder, &long)?;
            let next = [long[0]];
            let items = [
                BatchItem::new(decoder, &next, BatchItemKind::Decode),
                BatchItem::new(fresh, &prompt, BatchItemKind::Prefill),
            ];
            out.push(model.forward_batch(&items, &mut pool)?[..2 * vocab].to_vec());
        }
        // (c) two simultaneous prefills
        {
            let mut pool = model.new_pool(4096, 4)?;
            let sa = pool.alloc().expect("fresh pool had no rows");
            let sb = pool.alloc().expect("fresh pool had no second row");
            let items = [
                BatchItem::new(sa, &prompt, BatchItemKind::Prefill),
                BatchItem::new(sb, &b, BatchItemKind::Prefill),
            ];
            out.push(model.forward_batch(&items, &mut pool)?[..2 * vocab].to_vec());
        }
        Ok(out)
    }

    let on = {
        let (mut model, _t) = load("1")?.expect("model path already checked");
        scenarios(&mut model)?
    };
    let off = {
        let (mut model, _t) = load("0")?.expect("model path already checked");
        scenarios(&mut model)?
    };

    assert_eq!(on.len(), off.len());
    for (i, (a, b)) in on.iter().zip(&off).enumerate() {
        assert_eq!(a.len(), b.len(), "scenario {i} produced a different row count");
        let cos = cosine(a, b);
        eprintln!("  scenario {i}: cosine(on, off) {cos:.12}");
        let vocab = a.len() / (if i == 0 { 1 } else { 2 });
        for row in 0..a.len() / vocab {
            let (ra, rb) = (&a[row * vocab..(row + 1) * vocab], &b[row * vocab..(row + 1) * vocab]);
            assert_eq!(
                argmax(ra),
                argmax(rb),
                "scenario {i} row {row}: INFERO_SPLIT_MIXED_BATCH=1 and =0 disagree"
            );
        }
        assert!(cos > 0.9999, "scenario {i}: cosine {cos:.12} between split on and off");
    }
    Ok(())
}
