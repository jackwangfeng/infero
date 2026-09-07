//! `make_pool`'s fast path (`fits(hi)` succeeds on the first try, the common
//! case whenever the card has enough free VRAM for the full requested
//! concurrency) used to construct the same-sized `KvPool` twice: once inside
//! `fits` to check whether it fit, discarded immediately, and once more,
//! identically, to actually return it. Real production startup logs showed
//! exactly this -- two "kv pool allocated" lines with identical
//! `n_slots`/`max_seqs`/`max_seq` back to back. Freed GPU memory is not
//! necessarily handed back to the driver immediately by the allocator, so
//! this was a real, avoidable extra allocation on every real startup, not
//! just wasted time.
//!
//! `KvPool::id()` increments on every real `KvPool::new` call, including
//! ones immediately dropped, so counting the id delta across one `make_pool`
//! call is a direct, real measurement of how many pools it actually
//! constructed -- not an inference from timing or memory totals.
//!
//! Skipped when `models/` is empty, same convention as the other tests in
//! this crate.

use std::path::PathBuf;

use infero_gpu::Device;
use infero_model::{KvCacheQuant, Model};
use infero_server::scheduler::make_pool;

fn model_path() -> Option<PathBuf> {
    let p = std::env::var("INFERO_TEST_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../models/qwen2.5-0.5b-instruct-q8_0.gguf")
        });
    p.exists().then_some(p)
}

#[test]
fn the_fast_path_allocates_the_pool_exactly_once() -> anyhow::Result<()> {
    let Some(path) = model_path() else {
        eprintln!("skipping: no local test GGUF");
        return Ok(());
    };
    let gguf = infero_gguf::Gguf::open(&path)?;
    // Small max_seq/max_seqs so `want` is tiny and comfortably fits this
    // machine's free VRAM on the very first `fits(hi)` check -- the fast
    // path this test targets, matching what real production startup hits
    // whenever the card has room for the full requested concurrency.
    let model = Model::load_quantized(Device::new(0)?, &gguf, 512, KvCacheQuant::F16)?;

    let pool_a = make_pool(&model, 2, None, None)?;
    let id_a = pool_a.id();
    drop(pool_a);

    let pool_b = make_pool(&model, 2, None, None)?;
    let id_b = pool_b.id();

    // Exactly one real KvPool construction per make_pool call in the fast
    // path -- before this fix, this delta was 2 (one discarded trial, one
    // kept). `NEXT_POOL_ID` is a single process-wide counter, so any real
    // pool built in between would inflate this; nothing else runs here.
    assert_eq!(
        id_b - id_a,
        1,
        "make_pool should construct exactly one KvPool in the fast path, \
         not allocate-and-discard a trial before returning the real one"
    );
    Ok(())
}

/// `gpu_memory_fraction` must actually shrink the pool below what the
/// default (whatever free VRAM allows) would pick -- an unenforced cap is
/// worse than no cap at all, since it would look like it works from the CLI
/// help text alone.
#[test]
fn gpu_memory_fraction_caps_the_pool_below_the_unconstrained_size() -> anyhow::Result<()> {
    let Some(path) = model_path() else {
        eprintln!("skipping: no local test GGUF");
        return Ok(());
    };
    let gguf = infero_gguf::Gguf::open(&path)?;
    // A large `max_seqs`/`ctx` so the unconstrained pool genuinely wants to
    // grow large (this checkpoint's real free VRAM should comfortably back
    // a request this size on the fast path), giving the fraction cap real
    // room to bind rather than being a no-op because `want` was already
    // small.
    let model = Model::load_quantized(Device::new(0)?, &gguf, 4096, KvCacheQuant::F16)?;

    let uncapped = make_pool(&model, 32, None, None)?;
    let capped = make_pool(&model, 32, None, Some(0.01))?;

    assert!(
        capped.n_slots() < uncapped.n_slots(),
        "a 1% memory-fraction cap should produce a meaningfully smaller pool \
         than the uncapped default (uncapped n_slots={}, capped n_slots={})",
        uncapped.n_slots(),
        capped.n_slots()
    );
    // The cap must still respect the pool's real floor (one full-context
    // sequence) rather than shrinking below what `make_pool` can actually
    // serve -- `make_pool`'s own `lo = max_seq` invariant, not something
    // this test re-derives.
    assert!(capped.n_slots() >= model.max_seq());
    Ok(())
}
