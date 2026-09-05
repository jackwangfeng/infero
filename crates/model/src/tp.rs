//! Rank identity and cross-process bootstrap for tensor-parallel inference.
//! See `docs/superpowers/specs/2026-09-05-tensor-parallel-design.md`.
//!
//! `RankId` carries both a `pp_rank`/`pp_size` and a `tp_rank`/`tp_size` from
//! day one, even though pipeline parallelism isn't implemented and
//! `pp_size` is always `1` -- every place that would say "the rank" says
//! "the tp_rank within this pp_rank's layer range" instead, so adding real
//! PP later changes what layer range a `pp_rank` owns, not the rank-identity
//! type itself.

use anyhow::{Context, Result};
use infero_kernels::tp::NcclUniqueId;

#[derive(Debug, Clone, Copy)]
pub struct RankId {
    pub pp_rank: usize,
    pub pp_size: usize,
    pub tp_rank: usize,
    pub tp_size: usize,
}

/// How ranks in the same tensor-parallel group agree on an NCCL unique id
/// before any of them can call `ncclCommInitRank`. `tp_rank == 0` (within
/// its `pp_rank`) generates the id; every other rank blocks until it can
/// read what rank 0 published. A future cross-node deployment replaces only
/// the implementation of this trait (e.g. reading the id from a
/// coordination service instead of a local file) -- nothing downstream of
/// `broadcast_unique_id`'s return value needs to change.
pub trait RankBootstrap {
    fn broadcast_unique_id(&self, rank: &RankId) -> Result<NcclUniqueId>;
}

/// Single-node bootstrap: rank 0 writes the id to a local file named after a
/// shared `run_id`; every other rank in the same `pp_rank` polls for it.
pub struct LocalFileBootstrap {
    pub run_id: String,
}

impl LocalFileBootstrap {
    fn path_for(&self, rank: &RankId) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("infero_tp_{}_pp{}.id", self.run_id, rank.pp_rank))
    }
}

impl RankBootstrap for LocalFileBootstrap {
    fn broadcast_unique_id(&self, rank: &RankId) -> Result<NcclUniqueId> {
        let path = self.path_for(rank);
        if rank.tp_rank == 0 {
            let id = NcclUniqueId::generate().context("generating NCCL unique id")?;
            std::fs::write(&path, id.0).context("publishing NCCL unique id")?;
            Ok(id)
        } else {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            loop {
                if let Ok(bytes) = std::fs::read(&path) {
                    if bytes.len() == infero_kernels::tp::NCCL_UNIQUE_ID_BYTES {
                        let mut arr = [0u8; infero_kernels::tp::NCCL_UNIQUE_ID_BYTES];
                        arr.copy_from_slice(&bytes);
                        return Ok(NcclUniqueId(arr));
                    }
                }
                anyhow::ensure!(
                    std::time::Instant::now() < deadline,
                    "timed out waiting for rank 0 to publish the NCCL unique id at {}",
                    path.display()
                );
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_file_bootstrap_round_trips_a_real_unique_id_across_two_calls() {
        let run_id = format!("test-{}", std::process::id());
        let bootstrap = LocalFileBootstrap { run_id: run_id.clone() };
        let rank0 = RankId { pp_rank: 0, pp_size: 1, tp_rank: 0, tp_size: 2 };
        let rank1 = RankId { pp_rank: 0, pp_size: 1, tp_rank: 1, tp_size: 2 };

        let id0 = bootstrap.broadcast_unique_id(&rank0).expect("rank 0 bootstrap");
        let id1 = bootstrap.broadcast_unique_id(&rank1).expect("rank 1 bootstrap");
        assert_eq!(id0.0, id1.0, "both ranks must agree on the same NCCL unique id");
    }
}
