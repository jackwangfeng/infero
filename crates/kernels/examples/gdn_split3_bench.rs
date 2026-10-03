//! Does the 3-kernel split (`gdn_chunk_uw_f32` / `gdn_chunk_state_f32` /
//! `gdn_chunk_output_f32`) beat `gdn_delta_rule_reg128_f32` at the real
//! shape, now that the state-independent, expensive part (system-matrix
//! inversion + WY reconstruction) runs on a `chunks * heads` grid instead of
//! being serialized inside `reg128`'s (and `gdn_chunk_delta_rule_f32`'s own)
//! `heads`-only 48-block ceiling?
//!
//!     cargo run --release -p infero-kernels --example gdn_split3_bench

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::Kernels;
use infero_kernels::gdn::{DeltaVariant, GdnChunkStateVariant, SeqLayout};

const HEADS: usize = 48;
const KEY_HEADS: usize = 16;
const DK: usize = 128;
const DV: usize = 128;
const LINEAR_LAYERS: usize = 48;
const DEFAULT_TOTAL_TOKENS: usize = 30552;

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
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let dev = Device::new(0)?;
    println!("device: {} (sm_{}, {} SMs)", dev.name(), dev.arch(), dev.sm_count());
    let k = Kernels::new(dev.clone());
    let ctx = dev.context().clone();
    let stream = dev.stream().clone();

    let key_dim = KEY_HEADS * DK;
    let val_dim = HEADS * DV;
    let stride = 2 * key_dim + val_dim;
    let offsets = (stride, 0, key_dim, 2 * key_dim);
    let total = std::env::var("GDN_BENCH_TOTAL_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_TOTAL_TOKENS);

    let row = pseudo_random(total * stride, 0xe317);
    let g: Vec<f32> = pseudo_random(total * HEADS, 0xe318).iter().map(|v| -v.abs() * 0.6).collect();
    let beta: Vec<f32> =
        pseudo_random(total * HEADS, 0xe319).iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
    let first = [0i32];
    let ntok = [total as i32];

    let d_row = stream.clone_htod(&row)?;
    let d_g = stream.clone_htod(&g)?;
    let d_beta = stream.clone_htod(&beta)?;
    let d_first = stream.clone_htod(&first)?;
    let d_ntok = stream.clone_htod(&ntok)?;
    let mut d_out = stream.alloc_zeros::<f32>(total * HEADS * DV)?;

    let per_layer = HEADS * DK * DV;
    let layers = LINEAR_LAYERS.min((1 << 28) / (per_layer * 4)).max(1);
    let mut d_state = stream.alloc_zeros::<f32>(layers * per_layer)?;

    let seqs = SeqLayout {
        first_token: &d_first.as_view(),
        n_tokens: &d_ntok.as_view(),
        n_seqs: 1,
        total_tokens: total,
    };

    println!("\n{total} tokens, 1 seq, {HEADS} heads, {layers} state buffers");

    const GDN_CHUNK: usize = 32;
    let n_chunks = total.div_ceil(GDN_CHUNK).max(1);

    // reg128: today's deployed kernel.
    let mut run_reg = |iters: usize| -> Result<()> {
        for it in 0..iters {
            let layer = it % layers;
            let mut slice = d_state.slice_mut(layer * per_layer..(layer + 1) * per_layer);
            k.gdn_delta_rule_variant(
                &mut d_out.as_view_mut(),
                &mut slice,
                &d_row.as_view(),
                &d_g.as_view(),
                &d_beta.as_view(),
                &seqs,
                HEADS,
                KEY_HEADS,
                DK,
                DV,
                offsets,
                false,
                DeltaVariant::Reg,
            )?;
        }
        Ok(())
    };
    run_reg(2)?;
    dev.synchronize()?;
    let start = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let stop = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    start.record(&stream)?;
    run_reg(LINEAR_LAYERS)?;
    stop.record(&stream)?;
    stop.synchronize()?;
    let reg_ms = start.elapsed_ms(&stop)? as f64;
    println!("  reg128 (deployed)    {reg_ms:>10.2} ms across {LINEAR_LAYERS} layers");

    // Direct, same-process, same-tenant-conditions A/B for the fix this bench
    // exists to measure: the exact old behavior (fresh `stream.alloc_zeros`
    // on all four kernel-1/kernel-2 scratch buffers every call, ~5 GB a call
    // at this shape) against buffers allocated once and reused across calls,
    // both against the *same* kernel-2 variant (`Plain`) back to back so
    // tenant/clock noise on this shared GPU cancels out rather than
    // confounding the comparison. Measured BEFORE the persistent buffers
    // below are allocated, so the "old" run's peak memory matches what the
    // old code actually used (not inflated by the persistent copies sitting
    // resident on top of it, which -- on a GPU this loaded by other tenants --
    // can OOM on its own).
    {
        let mut run_split3_realloc = |iters: usize| -> Result<()> {
            for it in 0..iters {
                let layer = it % layers;
                let mut slice = d_state.slice_mut(layer * per_layer..(layer + 1) * per_layer);
                let mut w_tmp = stream.alloc_zeros::<f32>(n_chunks * HEADS * GDN_CHUNK * DK)?;
                let mut u_tmp = stream.alloc_zeros::<f32>(n_chunks * HEADS * GDN_CHUNK * DV)?;
                let mut delta_tmp = stream.alloc_zeros::<f32>(n_chunks * HEADS * GDN_CHUNK * DV)?;
                let mut s_before_tmp = stream.alloc_zeros::<f32>(n_chunks * HEADS * DK * DV)?;
                k.gdn_chunk_split3_delta_rule(
                    &mut d_out.as_view_mut(),
                    &mut slice,
                    &d_row.as_view(),
                    &d_g.as_view(),
                    &d_beta.as_view(),
                    &seqs,
                    HEADS,
                    KEY_HEADS,
                    DK,
                    DV,
                    offsets,
                    false,
                    GdnChunkStateVariant::Plain,
                    &mut w_tmp.as_view_mut(),
                    &mut u_tmp.as_view_mut(),
                    &mut delta_tmp.as_view_mut(),
                    &mut s_before_tmp.as_view_mut(),
                )?;
            }
            Ok(())
        };
        run_split3_realloc(2)?;
        dev.synchronize()?;
        let start_r = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let stop_r = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        start_r.record(&stream)?;
        run_split3_realloc(LINEAR_LAYERS)?;
        stop_r.record(&stream)?;
        stop_r.synchronize()?;
        let realloc_ms = start_r.elapsed_ms(&stop_r)? as f64;
        drop(run_split3_realloc);

        let mut w_buf2 = stream.alloc_zeros::<f32>(n_chunks * HEADS * GDN_CHUNK * DK)?;
        let mut u_buf2 = stream.alloc_zeros::<f32>(n_chunks * HEADS * GDN_CHUNK * DV)?;
        let mut delta_buf2 = stream.alloc_zeros::<f32>(n_chunks * HEADS * GDN_CHUNK * DV)?;
        let mut s_before_buf2 = stream.alloc_zeros::<f32>(n_chunks * HEADS * DK * DV)?;
        let mut run_split3_persistent = |iters: usize| -> Result<()> {
            for it in 0..iters {
                let layer = it % layers;
                let mut slice = d_state.slice_mut(layer * per_layer..(layer + 1) * per_layer);
                k.gdn_chunk_split3_delta_rule(
                    &mut d_out.as_view_mut(),
                    &mut slice,
                    &d_row.as_view(),
                    &d_g.as_view(),
                    &d_beta.as_view(),
                    &seqs,
                    HEADS,
                    KEY_HEADS,
                    DK,
                    DV,
                    offsets,
                    false,
                    GdnChunkStateVariant::Plain,
                    &mut w_buf2.as_view_mut(),
                    &mut u_buf2.as_view_mut(),
                    &mut delta_buf2.as_view_mut(),
                    &mut s_before_buf2.as_view_mut(),
                )?;
            }
            Ok(())
        };
        run_split3_persistent(2)?;
        dev.synchronize()?;
        let start_p = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let stop_p = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        start_p.record(&stream)?;
        run_split3_persistent(LINEAR_LAYERS)?;
        stop_p.record(&stream)?;
        stop_p.synchronize()?;
        let persistent_ms = start_p.elapsed_ms(&stop_p)? as f64;
        println!(
            "  plain, old per-call alloc_zeros scratch   {realloc_ms:>10.2} ms across {LINEAR_LAYERS} layers"
        );
        println!(
            "  plain, persistent reused scratch          {persistent_ms:>10.2} ms across {LINEAR_LAYERS} layers, {:.3}x speedup from removing the per-call alloc_zeros tax",
            realloc_ms / persistent_ms
        );
    }

    // `gdn_chunk_split3_delta_rule`'s own cross-kernel scratch, allocated
    // once here (this bench's own "load time") and reused for every
    // iteration below -- the fix this bench exists to measure. `total` is
    // fixed for the whole run, so a single `n_chunks`-sized allocation
    // covers every call; a real caller sizes this the same way, once, to
    // `Model::batch_tokens().div_ceil(GDN_CHUNK)` (see the doc comment on
    // `gdn_chunk_split3_delta_rule` in `gdn.rs` for why no re-zeroing between
    // calls is needed).
    let mut w_buf = stream.alloc_zeros::<f32>(n_chunks * HEADS * GDN_CHUNK * DK)?;
    let mut u_buf = stream.alloc_zeros::<f32>(n_chunks * HEADS * GDN_CHUNK * DV)?;
    let mut delta_buf = stream.alloc_zeros::<f32>(n_chunks * HEADS * GDN_CHUNK * DV)?;
    let mut s_before_buf = stream.alloc_zeros::<f32>(n_chunks * HEADS * DK * DV)?;

    // 3-kernel split.
    let mut run_split3 = |iters: usize, k2: GdnChunkStateVariant| -> Result<()> {
        for it in 0..iters {
            let layer = it % layers;
            let mut slice = d_state.slice_mut(layer * per_layer..(layer + 1) * per_layer);
            k.gdn_chunk_split3_delta_rule(
                &mut d_out.as_view_mut(),
                &mut slice,
                &d_row.as_view(),
                &d_g.as_view(),
                &d_beta.as_view(),
                &seqs,
                HEADS,
                KEY_HEADS,
                DK,
                DV,
                offsets,
                false,
                k2,
                &mut w_buf.as_view_mut(),
                &mut u_buf.as_view_mut(),
                &mut delta_buf.as_view_mut(),
                &mut s_before_buf.as_view_mut(),
            )?;
        }
        Ok(())
    };
    for (label, k2) in [
        ("plain", GdnChunkStateVariant::Plain),
        ("pipelined", GdnChunkStateVariant::Pipelined),
        ("pipelined_split4", GdnChunkStateVariant::PipelinedSplit4),
        ("mma", GdnChunkStateVariant::Mma),
        ("mma_vtile", GdnChunkStateVariant::MmaVtile),
        ("mma_pipelined", GdnChunkStateVariant::MmaPipelined),
        ("mma_vtile_pipelined", GdnChunkStateVariant::MmaVtilePipelined),
        ("mma_triton_match", GdnChunkStateVariant::MmaTritonMatch),
    ] {
        run_split3(2, k2)?;
        dev.synchronize()?;
        let start2 = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let stop2 = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        start2.record(&stream)?;
        run_split3(LINEAR_LAYERS, k2)?;
        stop2.record(&stream)?;
        stop2.synchronize()?;
        let split3_ms = start2.elapsed_ms(&stop2)? as f64;
        println!(
            "  3-kernel split ({label:<17}) {split3_ms:>10.2} ms across {LINEAR_LAYERS} layers, {:.3}x vs reg128",
            reg_ms / split3_ms
        );
    }

    // Direct, same-process, same-tenant-conditions A/B for the fix this bench
    // exists to measure: the exact old behavior (fresh `stream.alloc_zeros`
    // on all four scratch buffers every call) against the persistent,
    // reused-across-calls buffers just measured above, both against the
    // *same* kernel-2 variant (`Plain`) back to back so tenant/clock noise on
    // this shared GPU cancels out rather than confounding the comparison.
    {
        let mut run_split3_realloc = |iters: usize| -> Result<()> {
            for it in 0..iters {
                let layer = it % layers;
                let mut slice = d_state.slice_mut(layer * per_layer..(layer + 1) * per_layer);
                let mut w_tmp = stream.alloc_zeros::<f32>(n_chunks * HEADS * GDN_CHUNK * DK)?;
                let mut u_tmp = stream.alloc_zeros::<f32>(n_chunks * HEADS * GDN_CHUNK * DV)?;
                let mut delta_tmp = stream.alloc_zeros::<f32>(n_chunks * HEADS * GDN_CHUNK * DV)?;
                let mut s_before_tmp = stream.alloc_zeros::<f32>(n_chunks * HEADS * DK * DV)?;
                k.gdn_chunk_split3_delta_rule(
                    &mut d_out.as_view_mut(),
                    &mut slice,
                    &d_row.as_view(),
                    &d_g.as_view(),
                    &d_beta.as_view(),
                    &seqs,
                    HEADS,
                    KEY_HEADS,
                    DK,
                    DV,
                    offsets,
                    false,
                    GdnChunkStateVariant::Plain,
                    &mut w_tmp.as_view_mut(),
                    &mut u_tmp.as_view_mut(),
                    &mut delta_tmp.as_view_mut(),
                    &mut s_before_tmp.as_view_mut(),
                )?;
            }
            Ok(())
        };
        run_split3_realloc(2)?;
        dev.synchronize()?;
        let start_r = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let stop_r = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        start_r.record(&stream)?;
        run_split3_realloc(LINEAR_LAYERS)?;
        stop_r.record(&stream)?;
        stop_r.synchronize()?;
        let realloc_ms = start_r.elapsed_ms(&stop_r)? as f64;
        drop(run_split3_realloc);

        // A fresh closure rather than reusing `run_split3` above: that one's
        // borrow of `d_out`/`d_state` is still alive (it gets used again by
        // name), and two closures can't both hold a mutable capture of the
        // same locals at once even sequentially in the same scope.
        let mut run_split3_persistent = |iters: usize| -> Result<()> {
            for it in 0..iters {
                let layer = it % layers;
                let mut slice = d_state.slice_mut(layer * per_layer..(layer + 1) * per_layer);
                k.gdn_chunk_split3_delta_rule(
                    &mut d_out.as_view_mut(),
                    &mut slice,
                    &d_row.as_view(),
                    &d_g.as_view(),
                    &d_beta.as_view(),
                    &seqs,
                    HEADS,
                    KEY_HEADS,
                    DK,
                    DV,
                    offsets,
                    false,
                    GdnChunkStateVariant::Plain,
                    &mut w_buf.as_view_mut(),
                    &mut u_buf.as_view_mut(),
                    &mut delta_buf.as_view_mut(),
                    &mut s_before_buf.as_view_mut(),
                )?;
            }
            Ok(())
        };
        run_split3_persistent(2)?;
        dev.synchronize()?;
        let start_p = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let stop_p = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        start_p.record(&stream)?;
        run_split3_persistent(LINEAR_LAYERS)?;
        stop_p.record(&stream)?;
        stop_p.synchronize()?;
        let persistent_ms = start_p.elapsed_ms(&stop_p)? as f64;
        println!(
            "  plain, old per-call alloc_zeros scratch   {realloc_ms:>10.2} ms across {LINEAR_LAYERS} layers"
        );
        println!(
            "  plain, persistent reused scratch          {persistent_ms:>10.2} ms across {LINEAR_LAYERS} layers, {:.3}x speedup from removing the per-call alloc_zeros tax",
            realloc_ms / persistent_ms
        );
    }
    // Scan-based kernel 2 replacement, wired into the full pipeline.
    // group_size chosen so n_groups ~ sqrt(n_chunks) at this real shape
    // (n_chunks = ceil(30552/32) = 955, sqrt ~ 31).
    let group_size = 31usize;
    let mut run_scan = |iters: usize| -> Result<()> {
        for it in 0..iters {
            let layer = it % layers;
            let mut slice = d_state.slice_mut(layer * per_layer..(layer + 1) * per_layer);
            k.gdn_scan_split_delta_rule(
                &mut d_out.as_view_mut(),
                &mut slice,
                &d_row.as_view(),
                &d_g.as_view(),
                &d_beta.as_view(),
                &seqs,
                HEADS,
                KEY_HEADS,
                DK,
                DV,
                offsets,
                false,
                group_size,
            )?;
        }
        Ok(())
    };
    run_scan(2)?;
    dev.synchronize()?;
    let start3 = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let stop3 = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    start3.record(&stream)?;
    run_scan(LINEAR_LAYERS)?;
    stop3.record(&stream)?;
    stop3.synchronize()?;
    let scan_ms = start3.elapsed_ms(&stop3)? as f64;
    println!(
        "  scan-based split3 (group={group_size:<3})   {scan_ms:>10.2} ms across {LINEAR_LAYERS} layers, {:.3}x vs reg128",
        reg_ms / scan_ms
    );

    // GDN_CHUNK=64 3-kernel split: same shape, same reg128 baseline above,
    // half as many chunks (`n_chunks64 = total.div_ceil(64)`) -- this is the
    // real lever the GDN_CHUNK=64 family exists to test (kernel 2's
    // sequential chunk-loop iteration count halved).
    {
        const GDN_CHUNK64: usize = 64;
        let n_chunks64 = total.div_ceil(GDN_CHUNK64).max(1);
        let mut w_buf64 = stream.alloc_zeros::<f32>(n_chunks64 * HEADS * GDN_CHUNK64 * DK)?;
        let mut u_buf64 = stream.alloc_zeros::<f32>(n_chunks64 * HEADS * GDN_CHUNK64 * DV)?;
        let mut delta_buf64 = stream.alloc_zeros::<f32>(n_chunks64 * HEADS * GDN_CHUNK64 * DV)?;
        let mut s_before_buf64 = stream.alloc_zeros::<f32>(n_chunks64 * HEADS * DK * DV)?;
        let mut run_split3_c64 = |iters: usize| -> Result<()> {
            for it in 0..iters {
                let layer = it % layers;
                let mut slice = d_state.slice_mut(layer * per_layer..(layer + 1) * per_layer);
                k.gdn_chunk_split3_delta_rule_c64(
                    &mut d_out.as_view_mut(),
                    &mut slice,
                    &d_row.as_view(),
                    &d_g.as_view(),
                    &d_beta.as_view(),
                    &seqs,
                    HEADS,
                    KEY_HEADS,
                    DK,
                    DV,
                    offsets,
                    false,
                    &mut w_buf64.as_view_mut(),
                    &mut u_buf64.as_view_mut(),
                    &mut delta_buf64.as_view_mut(),
                    &mut s_before_buf64.as_view_mut(),
                )?;
            }
            Ok(())
        };
        run_split3_c64(2)?;
        dev.synchronize()?;
        let start4 = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let stop4 = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        start4.record(&stream)?;
        run_split3_c64(LINEAR_LAYERS)?;
        stop4.record(&stream)?;
        stop4.synchronize()?;
        let c64_ms = start4.elapsed_ms(&stop4)? as f64;
        println!(
            "  3-kernel split (GDN_CHUNK=64)    {c64_ms:>10.2} ms across {LINEAR_LAYERS} layers, {:.3}x vs reg128",
            reg_ms / c64_ms
        );
    }

    if std::env::var("INFERO_PROFILE").is_ok_and(|v| v == "1") {
        println!("\n{}", k.device().profile().report());
    }
    Ok(())
}
