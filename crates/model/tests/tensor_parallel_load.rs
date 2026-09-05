//! Sharded loading correctness: `Model::load_full_tp` at `tp_size=2` for both
//! ranks against the real small validation checkpoint, on `bw`'s free GPU
//! headroom. Real acceptance criterion for the whole tensor-parallel design
//! is the generation comparison in the implementation plan's Task 6 (a real
//! multi-process run) -- this test is the load-time correctness gate that
//! has to pass before that's worth attempting: does each rank's `Weights`
//! actually come out shaped for its own shard (verified internally by
//! `Weights::load_sharded`'s own `check_shapes` call, which uses the
//! already-sharded `Config` as the source of truth), not silently loading
//! the wrong bytes or the wrong shape.
#![cfg(feature = "nccl")]

use std::path::PathBuf;

use infero_cuda::Device;
use infero_gguf::Gguf;
use infero_model::config::Config;
use infero_model::tp::RankId;
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

    for tp_rank in 0..2 {
        let rank = RankId { pp_rank: 0, pp_size: 1, tp_rank, tp_size: 2 };
        let model = Model::load_full_tp(
            Device::new(0).expect("device"),
            &gguf,
            128,
            KvCacheQuant::F16,
            usize::MAX,
            32,
            &rank,
        );
        let model = model.unwrap_or_else(|e| panic!("rank {tp_rank} failed to load: {e:#}"));
        // `check_shapes` (internal to `Weights::load_sharded`) already
        // verified every projection's [k, n] against the sharded `Config`
        // for every layer -- reaching here at all means that passed. Sanity
        // check the config itself actually reflects the halved shape.
        let full_kv_heads = Config::from_gguf(&gguf).unwrap().n_kv_heads;
        assert_eq!(
            model.config().n_kv_heads,
            full_kv_heads / 2,
            "rank {tp_rank}'s config should carry the sharded (halved) kv head count"
        );
        drop(model);
    }
}

#[test]
fn tp1_is_unaffected() {
    let Some(path) = model_path() else {
        eprintln!("skipping: set INFERO_TEST_TP_GGUF to a real GGUF checkpoint");
        return;
    };
    let gguf = Gguf::open(&path).expect("opening checkpoint");
    let rank = RankId { pp_rank: 0, pp_size: 1, tp_rank: 0, tp_size: 1 };
    Model::load_full_tp(Device::new(0).expect("device"), &gguf, 128, KvCacheQuant::F16, usize::MAX, 32, &rank)
        .expect("tp_size=1 must load exactly like the existing single-GPU path");
}
