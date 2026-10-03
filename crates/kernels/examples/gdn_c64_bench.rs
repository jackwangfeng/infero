//! Minimal, low-memory-footprint benchmark: `reg128` (deployed baseline) vs
//! the GDN_CHUNK=64 3-kernel split, at real shapes. Split out from
//! `gdn_split3_bench.rs` because that harness's own realloc-vs-persistent
//! A/B measurement (a separate concern, already answered) transiently
//! double-allocates scratch buffers on top of the persistent ones, which
//! OOMs on this shared, heavily-loaded GPU (only ~8.5 GB free out of ~98 GB
//! at the time of this run, the rest held by another tenant's long-running
//! vLLM worker) at the full 30552-token shape -- this trimmed-down harness
//! only ever holds ONE set of scratch buffers per variant at a time.
//!
//!     GDN_BENCH_TOTAL_TOKENS=30552 cargo run --release -p infero-kernels --example gdn_c64_bench

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
    println!("  reg128 (deployed)             {reg_ms:>10.2} ms across {LINEAR_LAYERS} layers");

    // GDN_CHUNK=32 plain 3-kernel split, for a same-run reference point
    // (this box's absolute ms drifts ~2x run to run from other tenants, so
    // a same-process comparison matters more than any single absolute
    // number).
    {
        const GDN_CHUNK: usize = 32;
        let n_chunks = total.div_ceil(GDN_CHUNK).max(1);
        let mut w_buf = stream.alloc_zeros::<f32>(n_chunks * HEADS * GDN_CHUNK * DK)?;
        let mut u_buf = stream.alloc_zeros::<f32>(n_chunks * HEADS * GDN_CHUNK * DV)?;
        let mut delta_buf = stream.alloc_zeros::<f32>(n_chunks * HEADS * GDN_CHUNK * DV)?;
        let mut s_before_buf = stream.alloc_zeros::<f32>(n_chunks * HEADS * DK * DV)?;
        let mut run_split3 = |iters: usize| -> Result<()> {
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
        run_split3(2)?;
        dev.synchronize()?;
        let s32 = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let e32 = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        s32.record(&stream)?;
        run_split3(LINEAR_LAYERS)?;
        e32.record(&stream)?;
        e32.synchronize()?;
        let ms32 = s32.elapsed_ms(&e32)? as f64;
        println!(
            "  3-kernel split (GDN_CHUNK=32, plain) {ms32:>10.2} ms across {LINEAR_LAYERS} layers, {:.3}x vs reg128",
            reg_ms / ms32
        );
    }

    // GDN_CHUNK=64 3-kernel split -- the new kernel family.
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
        let s64 = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let e64 = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        s64.record(&stream)?;
        run_split3_c64(LINEAR_LAYERS)?;
        e64.record(&stream)?;
        e64.synchronize()?;
        let ms64 = s64.elapsed_ms(&e64)? as f64;
        println!(
            "  3-kernel split (GDN_CHUNK=64)        {ms64:>10.2} ms across {LINEAR_LAYERS} layers, {:.3}x vs reg128",
            reg_ms / ms64
        );
    }

    if std::env::var("INFERO_PROFILE").is_ok_and(|v| v == "1") {
        println!("\n{}", k.device().profile().report());
    }
    Ok(())
}
