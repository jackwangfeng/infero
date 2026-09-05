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
