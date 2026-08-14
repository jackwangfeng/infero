//! Decode attention at the one shape the gap to vLLM is measured at.
//!
//! `scripts/flash_attn_bandwidth.py` times `vllm_flash_attn` at batch 32, 512
//! of history, 32 query heads over 8 KV heads of 128: **58.1 us a layer, 1156
//! GB/s of KV**. This runs tuili's path at the same shape so the two are one
//! command apart, and reports the same denominator — K and V once each, which
//! is the floor any correct kernel pays.
//!
//! Prints and passes; `--nocapture`. Correctness lives in `tests/ops.rs`.
//!
//! # Why there are four KV caches
//!
//! One cache and fifty back-to-back launches measures a kernel whose whole
//! working set is in L2: this card has 128 MB of it and a layer's K and V at
//! this shape are 67. That is what both this harness and
//! `flash_attn_bandwidth.py` did, and it is why they agreed at 57.7 us against
//! 58.1 while the engine's own trace showed 67.7 us against 48.8 for the same
//! two kernels. In a real step each layer's cache is cold — 2.1 GB of them per
//! model — so the L2-warm number flatters tuili's path and hides the gap.
//!
//! So the caches are cycled: four of them, 268 MB in all, which does not fit,
//! and each launch reads one that has been evicted since it was last touched.
//! `TUILI_ATTN_POOLS=1` puts the warm measurement back for comparison.

use anyhow::Result;
use std::time::Instant;
use tuili_cuda::Device;
use tuili_kernels::{AttnDims, BatchLayout, Kernels};

const BATCH: usize = 32;
/// The engine's own median, read off the grid dimensions of 31k traced
/// launches, is 384 rather than the 512 this file first used; `TUILI_ATTN_HIST`
/// sweeps it, because a kernel that wins at one history can lose at another.
fn history() -> usize {
    std::env::var("TUILI_ATTN_HIST")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512)
}
const N_HEADS: usize = 32;
const N_KV_HEADS: usize = 8;
const D_HEAD: usize = 128;

#[test]
fn decode_attention_at_the_vllm_shape() -> Result<()> {
    let hist = history();
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();

    // One slot per (sequence, position), laid out the way the pool hands them
    // out under load: sequences interleave, so a sequence's history is a stride
    // through the pool rather than a run. A contiguous table would make the
    // reads look better here than they are in the engine.
    let n_slots = BATCH * hist;
    let dims = AttnDims {
        n_heads: N_HEADS,
        n_kv_heads: N_KV_HEADS,
        d_head: D_HEAD,
        n_slots,
        n_tokens: BATCH,
    };
    // How a sequence's history is laid out in the pool. vLLM pages its cache
    // sixteen tokens at a time, so consecutive keys are a 4 KB run; tuili hands
    // out one slot per token, and what the pool has left when a sequence needs
    // one is whatever the sequences that finished before it gave back. The two
    // ends of that are worth measuring separately, because a 256-byte row is
    // the smallest thing DRAM likes and a run of them is not the same request.
    let page: usize = std::env::var("TUILI_ATTN_PAGE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let table: Vec<i32> = (0..BATCH)
        .flat_map(|s| {
            (0..hist).map(move |p| {
                // `page` consecutive positions land in consecutive slots, then
                // the next sequence's page does.
                ((p / page) * BATCH * page + s * page + p % page) as i32
            })
        })
        .collect();
    let seq_of: Vec<i32> = (0..BATCH as i32).collect();
    let positions: Vec<i32> = vec![hist as i32 - 1; BATCH];

    let pools: usize = std::env::var("TUILI_ATTN_POOLS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let dq = stream.alloc_zeros::<f32>(BATCH * N_HEADS * D_HEAD)?;
    let dk: Vec<_> = (0..pools)
        .map(|_| stream.alloc_zeros::<half::f16>(N_KV_HEADS * n_slots * D_HEAD))
        .collect::<Result<_, _>>()?;
    let dv: Vec<_> = (0..pools)
        .map(|_| stream.alloc_zeros::<half::f16>(N_KV_HEADS * n_slots * D_HEAD))
        .collect::<Result<_, _>>()?;
    let dtable = stream.clone_htod(&table)?;
    let dseq = stream.clone_htod(&seq_of)?;
    let dpos = stream.clone_htod(&positions)?;
    let mut dscores = stream.alloc_zeros::<f32>(N_HEADS * BATCH * hist)?;
    let mut dout = stream.alloc_zeros::<f32>(BATCH * N_HEADS * D_HEAD)?;
    let mut dpart = stream.alloc_zeros::<f32>(Kernels::attn_partial_floats(
        N_HEADS, D_HEAD, BATCH,
    ))?;

    let (vseq, vpos, vtable) = (dseq.as_view(), dpos.as_view(), dtable.as_view());
    let batch = BatchLayout {
        seq_of: &vseq,
        positions: &vpos,
        slot_table: &vtable,
        table_stride: hist,
    };

    let time = |run: &mut dyn FnMut(usize) -> Result<()>| -> Result<f64> {
        for i in 0..5 {
            run(i % pools)?;
        }
        dev.synchronize()?;
        let t0 = Instant::now();
        for i in 0..50 {
            run(i % pools)?;
        }
        dev.synchronize()?;
        Ok(t0.elapsed().as_secs_f64() / 50.0)
    };

    // K and V once each. Everything below is reported against this, so a number
    // above the device's copy ceiling means a kernel is being served by L2.
    let kv_bytes = (BATCH * hist * N_KV_HEADS * D_HEAD * 2 * 2) as f64;

    let scores = time(&mut |p| {
        kern.attn_scores(
            &mut dscores.as_view_mut(),
            &dq.as_view(),
            &dk[p].as_view(),
            batch,
            dims,
            hist,
            0.088_388_35,
        )
    })?;
    let softmax = time(&mut |_| {
        kern.attn_softmax(&mut dscores.as_view_mut(), N_HEADS, BATCH, hist)
    })?;
    let output = time(&mut |p| {
        kern.attn_output(
            &mut dout.as_view_mut(),
            &dscores.as_view(),
            &dv[p].as_view(),
            batch,
            dims,
            hist,
            Some(&mut dpart.as_view_mut()),
        )
    })?;

    let fused = time(&mut |p| {
        kern.attn_decode(
            &mut dout.as_view_mut(),
            None,
            &dq.as_view(),
            &dk[p].as_view(),
            &dv[p].as_view(),
            batch,
            dims,
            hist,
            0.088_388_35,
            &mut dpart.as_view_mut(),
        )
        // Whether the combine wrote an f16 copy is not what this measures.
        .map(|_| ())
    })?;

    let mut sink = stream.alloc_zeros::<f32>(1)?;
    let probe = time(&mut |p| {
        kern.attn_kv_probe(
            &mut sink.as_view_mut(),
            &dk[p].as_view(),
            &dv[p].as_view(),
            batch,
            dims,
            hist,
        )
    })?;

    let total = scores + softmax + output;
    eprintln!(
        "\n  batch {BATCH}, history {hist}, {N_HEADS}q/{N_KV_HEADS}kv x {D_HEAD}, \
         {page}-token pages"
    );
    eprintln!("  attn_scores    {:>7.1} us", scores * 1e6);
    eprintln!("  attn_softmax   {:>7.1} us", softmax * 1e6);
    eprintln!("  attn_output    {:>7.1} us", output * 1e6);
    eprintln!(
        "  ---- total     {:>7.1} us  {:>5.0} GB/s of KV   (vllm 58.1 us, 1156 GB/s)",
        total * 1e6,
        kv_bytes / total / 1e9
    );
    eprintln!(
        "  a layer's attention over 32 layers: {:.2} ms a step",
        total * 32.0 * 1e3
    );
    eprintln!(
        "  kv probe       {:>7.1} us  {:>5.0} GB/s of KV   (the ceiling)",
        probe * 1e6,
        kv_bytes / probe / 1e9
    );
    eprintln!(
        "  attn_decode    {:>7.1} us  {:>5.0} GB/s of KV   ({:.2} ms a step, {:+.1}%)",
        fused * 1e6,
        kv_bytes / fused / 1e9,
        fused * 32.0 * 1e3,
        (fused / total - 1.0) * 100.0
    );
    Ok(())
}
