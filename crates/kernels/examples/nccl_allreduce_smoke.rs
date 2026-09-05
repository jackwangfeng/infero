//! Real 2+ rank NCCL all-reduce smoke test -- proves the FFI binding and a
//! crude bootstrap work before any model code depends on them. Run one copy
//! per rank, e.g. for a 2-rank test:
//!
//!   rm -f /tmp/nccl_smoke_id
//!   INFERO_NCCL_RANK=0 INFERO_NCCL_WORLD_SIZE=2 INFERO_NCCL_ID_FILE=/tmp/nccl_smoke_id CUDA_VISIBLE_DEVICES=3 \
//!     cargo run --release --features nccl --example nccl_allreduce_smoke &
//!   INFERO_NCCL_RANK=1 INFERO_NCCL_WORLD_SIZE=2 INFERO_NCCL_ID_FILE=/tmp/nccl_smoke_id CUDA_VISIBLE_DEVICES=1 \
//!     cargo run --release --features nccl --example nccl_allreduce_smoke &
//!   wait

use infero_kernels::tp::{NcclComm, NcclUniqueId, NCCL_UNIQUE_ID_BYTES};

fn main() -> anyhow::Result<()> {
    let rank: i32 = std::env::var("INFERO_NCCL_RANK")?.parse()?;
    let world_size: i32 = std::env::var("INFERO_NCCL_WORLD_SIZE")?.parse()?;
    let id_file = std::env::var("INFERO_NCCL_ID_FILE")?;

    let unique_id = if rank == 0 {
        let id = NcclUniqueId::generate()?;
        std::fs::write(&id_file, id.0)?;
        id
    } else {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            if let Ok(bytes) = std::fs::read(&id_file) {
                if bytes.len() == NCCL_UNIQUE_ID_BYTES {
                    let mut arr = [0u8; NCCL_UNIQUE_ID_BYTES];
                    arr.copy_from_slice(&bytes);
                    break NcclUniqueId(arr);
                }
            }
            anyhow::ensure!(
                std::time::Instant::now() < deadline,
                "timed out waiting for rank 0 to publish the NCCL unique id at {id_file}"
            );
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    };

    let dev = infero_cuda::Device::new(0)?; // CUDA_VISIBLE_DEVICES already remaps this to the right physical GPU
    let comm = NcclComm::init_rank(&unique_id, rank, world_size)?;
    let stream = dev.stream();

    let host_val = (rank + 1) as f32; // rank 0 contributes 1.0, rank 1 contributes 2.0, etc.
    let n = 4usize;
    let mut buf = stream.clone_htod(&vec![host_val; n])?;
    let mut view = buf.as_view_mut();
    comm.all_reduce_sum_f32(&mut view, n, stream)?;
    stream.synchronize()?;
    let out = stream.clone_dtoh(&buf.as_view())?;

    // sum_{r=0}^{world_size-1} (r+1) = world_size*(world_size+1)/2
    let expected = (world_size * (world_size + 1) / 2) as f32;
    println!("rank {rank}: all_reduce result = {out:?} (expected all {expected})");
    for &v in &out {
        assert_eq!(v, expected, "all_reduce produced the wrong sum");
    }
    println!("rank {rank}: OK");
    Ok(())
}
