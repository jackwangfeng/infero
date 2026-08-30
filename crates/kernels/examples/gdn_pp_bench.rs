use anyhow::{bail, Result};
use infero_gpu::Device;
use infero_kernels::Kernels;

const REPEATS: usize = 100;

fn main() -> Result<()> {
    let k = Kernels::new(Device::new(0)?);
    let stream = k.device().stream().clone();

    let mut out_seq = stream.alloc_zeros::<f32>(2)?;
    let mut out_pp = stream.alloc_zeros::<f32>(2)?;
    let mut out_ppb = stream.alloc_zeros::<f32>(2)?;

    k.gdn_pp_sequential_ref(&mut out_seq.as_view_mut())?;
    k.gdn_pp_pipelined_probe(&mut out_pp.as_view_mut())?;
    k.gdn_pp_pipelined_batched_probe(&mut out_ppb.as_view_mut())?;
    k.device().synchronize()?;

    let seq_host = stream.clone_dtoh(&out_seq)?;
    let pp_host = stream.clone_dtoh(&out_pp)?;
    let ppb_host = stream.clone_dtoh(&out_ppb)?;
    println!("sequential      checksum: out_sum={:.6} state_sum={:.6}", seq_host[0], seq_host[1]);
    println!("pipelined       checksum: out_sum={:.6} state_sum={:.6}", pp_host[0], pp_host[1]);
    println!("pipelined-batch checksum: out_sum={:.6} state_sum={:.6}", ppb_host[0], ppb_host[1]);
    let bits_match = |a: f32, b: f32| a.to_bits() == b.to_bits() || (a - b).abs() <= 1e-2;
    if !bits_match(seq_host[0], pp_host[0]) || !bits_match(seq_host[1], pp_host[1]) {
        bail!("checksum mismatch (pipelined) -- handoff protocol bug");
    }
    if !bits_match(seq_host[0], ppb_host[0]) || !bits_match(seq_host[1], ppb_host[1]) {
        bail!("checksum mismatch (pipelined-batched) -- handoff protocol bug");
    }
    println!("checksums match.\n");

    let mut seq_best = f64::INFINITY;
    for _ in 0..REPEATS {
        let t = std::time::Instant::now();
        k.gdn_pp_sequential_ref(&mut out_seq.as_view_mut())?;
        k.device().synchronize()?;
        seq_best = seq_best.min(t.elapsed().as_secs_f64());
    }
    let mut pp_best = f64::INFINITY;
    for _ in 0..REPEATS {
        let t = std::time::Instant::now();
        k.gdn_pp_pipelined_probe(&mut out_pp.as_view_mut())?;
        k.device().synchronize()?;
        pp_best = pp_best.min(t.elapsed().as_secs_f64());
    }
    let mut ppb_best = f64::INFINITY;
    for _ in 0..REPEATS {
        let t = std::time::Instant::now();
        k.gdn_pp_pipelined_batched_probe(&mut out_ppb.as_view_mut())?;
        k.device().synchronize()?;
        ppb_best = ppb_best.min(t.elapsed().as_secs_f64());
    }

    println!("sequential       (1 warp, state+output every iter): {:.3} us", seq_best * 1e6);
    println!("pipelined        (2 warps, handoff every 1 timestep): {:.3} us", pp_best * 1e6);
    println!("pipelined-batch  (2 warps, handoff every 4 timesteps): {:.3} us", ppb_best * 1e6);
    println!("\ncross-warp pipeline speedup: {:.3}x", seq_best / pp_best);
    println!("cross-warp pipeline (batched) speedup: {:.3}x", seq_best / ppb_best);
    Ok(())
}
