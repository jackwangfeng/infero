//! Minimal, single-launch harness for `ncu` profiling of the two deployed
//! prefill attention kernels, at the real production shape (24 heads, 4 kv
//! heads, d_head=256, a representative mid-sequence 1024-token chunk with
//! kv_len=29696, matching a late chunk of the real 30552-token prefill).
//! Deliberately does ONE launch per kernel, not a full 16-layer/30-chunk
//! sweep like `attn_decoupled_vs_ws4_bench` -- that bench also runs nine
//! other kernel variants and multiple warmup/best-of-N repeats, which would
//! make an `ncu --set full` pass (each matched launch replayed many times to
//! collect the full metric set) impractically slow.
//!
//!   cargo run --release --features cuda -p infero-kernels --example attn_ncu_probe -- ws4
//!   cargo run --release --features cuda -p infero-kernels --example attn_ncu_probe -- decoupled6

use anyhow::Result;
use half::f16;
use infero_gpu::Device;
use infero_kernels::{AttnDims, BatchLayout, Kernels};

const N_HEADS: usize = 24;
const N_KV_HEADS: usize = 4;
const D_HEAD: usize = 256;
const RUN_BASE: usize = 0; // an output-buffer offset in this API, not a free "which chunk" param -- the real Model dispatch always passes 0 (it slices Q/out per-chunk instead); kv_len alone already gives the representative large-causal-history shape.

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect()
}

fn main() -> Result<()> {
    let which = std::env::args().nth(1).unwrap_or_else(|| "ws4".to_string());
    let kv_len: usize = env_usize("PROBE_KV_LEN", 29696);
    let run_tokens: usize = env_usize("PROBE_RUN_TOKENS", 1024);
    let k = Kernels::new(Device::new(0)?);
    let stream = k.device().stream().clone();

    let n_slots = kv_len + 128;
    let q = pseudo_random(run_tokens * N_HEADS * D_HEAD, 0x71);
    let kv_elems = N_KV_HEADS * n_slots * D_HEAD;
    let kh: Vec<f16> = pseudo_random(kv_elems, 0x82)
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let vh: Vec<f16> = pseudo_random(kv_elems, 0x93)
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let seq_of = vec![0i32; kv_len];
    let positions: Vec<i32> = (0..kv_len as i32).collect();
    let table: Vec<i32> = (0..n_slots as i32).collect();
    let table_stride = n_slots;

    let dq = stream.clone_htod(&q)?;
    let dk = stream.clone_htod(&kh)?;
    let dv = stream.clone_htod(&vh)?;
    let dpos = stream.clone_htod(&positions)?;
    let dseq = stream.clone_htod(&seq_of)?;
    let dtable = stream.clone_htod(&table)?;
    let (vseq, vpos, vtable) = (dseq.as_view(), dpos.as_view(), dtable.as_view());
    let batch = BatchLayout {
        seq_of: &vseq,
        positions: &vpos,
        slot_table: &vtable,
        table_stride,
    };

    let scale = 1.0 / (D_HEAD as f32).sqrt();
    let dims = AttnDims {
        n_heads: N_HEADS,
        n_kv_heads: N_KV_HEADS,
        d_head: D_HEAD,
        n_slots,
        n_tokens: kv_len,
    };

    let mut out = stream.alloc_zeros::<f32>(run_tokens * N_HEADS * D_HEAD)?;
    let mut part =
        stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(N_HEADS, D_HEAD, run_tokens))?;

    // One warmup launch (JIT compile, cache warmup) not under profiling,
    // then the single launch `ncu` should actually attach to. The second
    // launch is also wrapped in a plain host-side timer (no `ncu`) so a
    // profiled `Duration` can be sanity-checked against unprofiled wall time.
    let reps: usize = env_usize("PROBE_REPS", 1);
    let t0 = std::time::Instant::now();
    match which.as_str() {
        "ws4" => {
            k.attn_prefill_ws4(
                &mut out.as_view_mut(),
                &dq.as_view(),
                &dk.as_view(),
                &dv.as_view(),
                batch,
                dims,
                RUN_BASE,
                run_tokens,
                kv_len,
                scale,
                &mut part.as_view_mut(),
            )?;
            k.device().synchronize()?;
            let t1 = std::time::Instant::now();
            for _ in 0..reps {
                k.attn_prefill_ws4(
                    &mut out.as_view_mut(),
                    &dq.as_view(),
                    &dk.as_view(),
                    &dv.as_view(),
                    batch,
                    dims,
                    RUN_BASE,
                    run_tokens,
                    kv_len,
                    scale,
                    &mut part.as_view_mut(),
                )?;
            }
            k.device().synchronize()?;
            println!(
                "unprofiled per-launch: {:.3} ms",
                t1.elapsed().as_secs_f64() * 1000.0 / reps as f64
            );
        }
        "decoupled6" => {
            k.attn_prefill_decoupled6_f16acc(
                &mut out.as_view_mut(),
                &dq.as_view(),
                &dk.as_view(),
                &dv.as_view(),
                batch,
                dims,
                RUN_BASE,
                run_tokens,
                kv_len,
                scale,
            )?;
            k.device().synchronize()?;
            let t1 = std::time::Instant::now();
            for _ in 0..reps {
                k.attn_prefill_decoupled6_f16acc(
                    &mut out.as_view_mut(),
                    &dq.as_view(),
                    &dk.as_view(),
                    &dv.as_view(),
                    batch,
                    dims,
                    RUN_BASE,
                    run_tokens,
                    kv_len,
                    scale,
                )?;
            }
            k.device().synchronize()?;
            println!(
                "unprofiled per-launch: {:.3} ms",
                t1.elapsed().as_secs_f64() * 1000.0 / reps as f64
            );
        }
        other => anyhow::bail!("unknown kernel {other}, expected ws4 or decoupled6"),
    }
    let _ = t0;
    println!("done: {which}");
    Ok(())
}
