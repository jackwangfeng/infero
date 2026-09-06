//! Sharded loading correctness: `Config::shard_for_tp` + `Weights::load_sharded`
//! at `tp_size=2` for both ranks against the real small validation
//! checkpoint, on `bw`'s free GPU headroom.
//!
//! Deliberately does NOT go through `Model::load_full_tp` -- that function
//! also performs a real, blocking, collective `ncclCommInitRank`, which
//! needs every rank in the group to call it around the same time from
//! separate OS processes (the real SPMD architecture this whole design
//! targets, exercised for real in the implementation plan's Task 6). Calling
//! it twice, sequentially, in one process/one rank's `world_size=2` view
//! would deadlock waiting for a peer that never joins. This test checks the
//! narrower, real thing Task 3 is actually about -- does each rank's
//! `Weights` come out shaped for its own shard, verified internally by
//! `Weights::load_sharded`'s own `check_shapes` call -- without needing a
//! second process.
#![cfg(feature = "nccl")]

use std::path::PathBuf;

use infero_cuda::Device;
use infero_gguf::Gguf;
use infero_model::config::Config;
use infero_model::tp::RankId;
use infero_model::weights::Weights;
use infero_model::{KvCacheQuant, Model};

fn model_path() -> Option<PathBuf> {
    let p = std::env::var("INFERO_TEST_TP_GGUF").ok().map(PathBuf::from)?;
    p.exists().then_some(p)
}

#[test]
fn loads_both_ranks_of_a_tp2_shard_without_shape_mismatch() {
    let Some(path) = model_path() else {
        eprintln!("skipping: set INFERO_TEST_TP_GGUF to a real GGUF checkpoint");
        return;
    };
    let gguf = Gguf::open(&path).expect("opening checkpoint");
    let full_cfg = Config::from_gguf(&gguf).expect("parsing config");

    for tp_rank in 0..2 {
        let rank = RankId { pp_rank: 0, pp_size: 1, tp_rank, tp_size: 2 };
        let mut cfg = Config::from_gguf(&gguf).expect("parsing config");
        cfg.shard_for_tp(&rank);
        assert_eq!(cfg.n_kv_heads, full_cfg.n_kv_heads / 2, "rank {tp_rank}: n_kv_heads not sharded");
        assert_eq!(cfg.n_heads, full_cfg.n_heads / 2, "rank {tp_rank}: n_heads not sharded");
        assert_eq!(cfg.d_ff, full_cfg.d_ff / 2, "rank {tp_rank}: d_ff not sharded");

        let dev = Device::new(0).expect("device");
        // check_shapes (internal to load_sharded) validates every layer's
        // projection [k, n] against this already-sharded cfg -- reaching
        // Ok(..) here means every Q/K/V/O/gate/up/down shard came out the
        // right shape for this rank.
        let w = Weights::load_sharded(&dev, &gguf, &cfg, usize::MAX, Some((tp_rank, 2)))
            .unwrap_or_else(|e| panic!("rank {tp_rank} failed to load: {e:#}"));
        drop(w);
    }
}

#[test]
fn w_kv_fusion_is_disabled_under_sharding() {
    // Regression test for a real bug: `stacked2_gguf` (which builds the
    // fused K+V matrix `w_kv` as a decode-step launch-count optimization,
    // taken by `attention()` whenever a layer's K and V share a GGUF type)
    // reads `attn_k.weight`/`attn_v.weight` directly out of the file at
    // their full, unsharded width -- it has no `shard` parameter and does
    // not go through `upload_matrix_sharded`. Under sharding this silently
    // built a `w.n` twice the per-rank `kv_dim` every other tensor in the
    // layer uses, and `attention`'s fused-`w_kv` branch wrote/split it
    // against buffers sized for the correctly-sharded `kv_dim` -- garbage
    // K/V from that layer on, with no shape assertion catching it. Real
    // checkpoints that mix quantization types per layer (a Q4_K_M file's
    // Q6_K/Q4_K attn_v mixing, say) mask this whenever K and V happen to
    // differ in type at a given layer, which is why this was found only on
    // one of two validation checkpoints -- so this test does not rely on
    // the fixture actually containing a K/V-type-matched layer; it simply
    // asserts the fused path is never built at all under sharding,
    // regardless of what the checkpoint's own per-layer types happen to be.
    let Some(path) = model_path() else {
        eprintln!("skipping: set INFERO_TEST_TP_GGUF to a real GGUF checkpoint");
        return;
    };
    let gguf = Gguf::open(&path).expect("opening checkpoint");
    let rank = RankId { pp_rank: 0, pp_size: 1, tp_rank: 0, tp_size: 2 };
    let mut cfg = Config::from_gguf(&gguf).expect("parsing config");
    cfg.shard_for_tp(&rank);
    let dev = Device::new(0).expect("device");
    let w = Weights::load_sharded(&dev, &gguf, &cfg, usize::MAX, Some((0, 2)))
        .expect("rank 0 failed to load");
    for (i, l) in w.layers.iter().enumerate() {
        if l.is_linear() {
            continue;
        }
        assert!(
            l.attn().w_kv.is_none(),
            "layer {i}: w_kv must be None under sharding -- built from unsharded \
             GGUF bytes at the wrong width, this is the real bug that produced \
             degenerate garbage output on llama-3.1-8b-instruct-q4_k_m TP=2"
        );
    }
}

/// Task 6: the mixed-batch attention dispatch split must be forced off under
/// real TP, regardless of `INFERO_SPLIT_MIXED_BATCH` -- and by a *real* gate,
/// not merely `attn_dispatch`'s construction-site `debug_assert!` (a no-op in
/// release builds; see `Model::gate_for_tp`'s doc comment and the design
/// doc's Error Handling section, "implementation should gate
/// `split_mixed_batch_enabled` itself on `tp_size <= 1`, not merely assert").
///
/// Deliberately does not call `Model::load_full_tp` -- see this file's module
/// doc for why that would deadlock waiting for a peer rank that never joins
/// in a single test process. Instead this builds a real, TP-sharded `Model`
/// via `Model::from_weights` (`load_full_tp` calls the same private
/// `from_parts` assembly) and then calls `Model::gate_for_tp` directly --
/// exactly the call `load_full_tp`/`load_awq_tp` make right after
/// construction, before their own blocking NCCL bootstrap.
#[test]
fn split_mixed_batch_disabled_under_tp() {
    let Some(path) = model_path() else {
        eprintln!("skipping: set INFERO_TEST_TP_GGUF to a real GGUF checkpoint");
        return;
    };
    let gguf = Gguf::open(&path).expect("opening checkpoint");
    let rank = RankId { pp_rank: 0, pp_size: 1, tp_rank: 0, tp_size: 2 };
    let mut cfg = Config::from_gguf(&gguf).expect("parsing config");
    cfg.shard_for_tp(&rank);
    let dev = Device::new(0).expect("device");
    let w = Weights::load_sharded(&dev, &gguf, &cfg, usize::MAX, Some((0, 2)))
        .expect("rank 0 failed to load");

    // `Model::from_parts` (reached through `from_weights`) resolves
    // `split_mixed_batch` from this env var at construction time -- forced to
    // "1" here so the assertion below actually proves the TP gate overrides
    // an explicit opt-in, not merely that it agrees with an already-off
    // default.
    unsafe { std::env::set_var("INFERO_SPLIT_MIXED_BATCH", "1") };
    let built = Model::from_weights(dev, cfg, w, 512, KvCacheQuant::F16, 8);
    unsafe { std::env::remove_var("INFERO_SPLIT_MIXED_BATCH") };
    let mut model = built.expect("model construction");
    assert!(
        model.split_mixed_batch(),
        "test setup broken: INFERO_SPLIT_MIXED_BATCH=1 must resolve true \
         before any TP gate runs, or the assertion below would prove nothing"
    );
    let batch_tokens = model.batch_tokens();
    assert!(
        model.attn_partial_tokens() < batch_tokens,
        "test setup broken: with the split resolved ON, attn_partial must have \
         been allocated at the shrunk `attn_partial_bound(max_logit_rows)` width \
         ({} tokens) and not the full {batch_tokens} -- otherwise the width \
         assertion after the gate proves nothing",
        model.attn_partial_tokens()
    );

    model.gate_for_tp(&rank).expect("gate_for_tp");
    assert!(
        !model.split_mixed_batch(),
        "split_mixed_batch must be forced off once tp_size > 1, even with \
         INFERO_SPLIT_MIXED_BATCH=1 set -- and this must hold in release \
         builds too, not just via a debug_assert!"
    );
    // The flag and the buffer are one decision, not two. Disabling the split
    // without also restoring `attn_partial` to its full `batch_tokens` width
    // leaves the unsplit dispatch -- which is what runs once the flag is off --
    // pointed at a decode-sized buffer, and `ensure_partial_fits` then hard-
    // fails every wide prefill run. `engine.rs` fails all running *and*
    // waiting requests on any step error, so that is a total outage under TP,
    // not one bad request.
    assert_eq!(
        model.attn_partial_tokens(),
        batch_tokens,
        "gate_for_tp disabled the split but left attn_partial at the shrunk \
         width -- the unsplit dispatch it just reverted to needs the full \
         {batch_tokens}-token buffer, and ensure_partial_fits would hard-fail \
         every wide prefill run under TP"
    );
}

#[test]
fn tp1_is_unaffected() {
    let Some(path) = model_path() else {
        eprintln!("skipping: set INFERO_TEST_TP_GGUF to a real GGUF checkpoint");
        return;
    };
    let gguf = Gguf::open(&path).expect("opening checkpoint");
    let cfg = Config::from_gguf(&gguf).expect("parsing config");
    let dev = Device::new(0).expect("device");
    Weights::load_sharded(&dev, &gguf, &cfg, usize::MAX, None)
        .expect("tp_size=1 (None) must load exactly like the existing single-GPU path");
}
