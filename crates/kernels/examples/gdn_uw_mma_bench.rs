//! Does the tensor-core version of kernel 1's system-matrix (`A = K@Kᵀ`)
//! computation (`gdn_chunk_uw_mma_f32`) actually beat the plain scalar
//! version (`gdn_chunk_uw_f32`) at the real shape, and where does the full
//! 3-kernel-split pipeline land against `gdn_delta_rule_reg128_f32` once
//! kernel 1 is swapped for the tensor-core version?
//!
//!     cargo run --release -p infero-kernels --example gdn_uw_mma_bench

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::Kernels;
use infero_kernels::gdn::{DeltaVariant, SeqLayout};

const HEADS: usize = 48;
const KEY_HEADS: usize = 16;
const DK: usize = 128;
const DV: usize = 128;
const LINEAR_LAYERS: usize = 48;
const TOTAL_TOKENS: usize = 30552;
const GDN_CHUNK: usize = 32;

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

    println!("\nkernel resources");
    for name in ["gdn_chunk_uw_f32", "gdn_chunk_uw_mma_f32", "gdn_delta_rule_reg128_f32"] {
        let (regs, _static, spill) = k.gdn_kernel_registers(name)?;
        println!("  {name:<28} regs={regs:>4} spill_bytes={spill}");
    }

    let key_dim = KEY_HEADS * DK;
    let val_dim = HEADS * DV;
    let stride = 2 * key_dim + val_dim;
    let offsets = (stride, 0, key_dim, 2 * key_dim);
    let total = TOTAL_TOKENS;
    let n_chunks = total.div_ceil(GDN_CHUNK);

    let row = pseudo_random(total * stride, 0xa317);
    let g: Vec<f32> = pseudo_random(total * HEADS, 0xa318).iter().map(|v| -v.abs() * 0.6).collect();
    let beta: Vec<f32> =
        pseudo_random(total * HEADS, 0xa319).iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
    let first = [0i32];
    let ntok = [total as i32];

    let d_row = stream.clone_htod(&row)?;
    let d_g = stream.clone_htod(&g)?;
    let d_beta = stream.clone_htod(&beta)?;
    let d_first = stream.clone_htod(&first)?;
    let d_ntok = stream.clone_htod(&ntok)?;

    let seqs = SeqLayout {
        first_token: &d_first.as_view(),
        n_tokens: &d_ntok.as_view(),
        n_seqs: 1,
        total_tokens: total,
    };

    println!("\n{total} tokens ({n_chunks} chunks), 1 seq, {HEADS} heads");

    let mut w = stream.alloc_zeros::<f32>(n_chunks * HEADS * GDN_CHUNK * DK)?;
    let mut u = stream.alloc_zeros::<f32>(n_chunks * HEADS * GDN_CHUNK * DV)?;

    let event = || -> Result<_> { Ok(ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?) };

    // Kernel 1 alone: plain scalar vs MMA, direct A/B at the real shape.
    k.gdn_chunk_uw_only(&mut w.as_view_mut(), &mut u.as_view_mut(), &d_row.as_view(), &d_g.as_view(), &d_beta.as_view(), &seqs, HEADS, KEY_HEADS, DK, DV, offsets, false)?;
    dev.synchronize()?;
    let start = event()?;
    let stop = event()?;
    start.record(&stream)?;
    for _ in 0..3 {
        k.gdn_chunk_uw_only(&mut w.as_view_mut(), &mut u.as_view_mut(), &d_row.as_view(), &d_g.as_view(), &d_beta.as_view(), &seqs, HEADS, KEY_HEADS, DK, DV, offsets, false)?;
    }
    stop.record(&stream)?;
    stop.synchronize()?;
    let plain_ms = start.elapsed_ms(&stop)? as f64 / 3.0;
    println!("  gdn_chunk_uw_f32     (scalar) {plain_ms:>10.3} ms/call");

    k.gdn_chunk_uw_mma_only(&mut w.as_view_mut(), &mut u.as_view_mut(), &d_row.as_view(), &d_g.as_view(), &d_beta.as_view(), &seqs, HEADS, KEY_HEADS, DK, DV, offsets, false)?;
    dev.synchronize()?;
    let start2 = event()?;
    let stop2 = event()?;
    start2.record(&stream)?;
    for _ in 0..3 {
        k.gdn_chunk_uw_mma_only(&mut w.as_view_mut(), &mut u.as_view_mut(), &d_row.as_view(), &d_g.as_view(), &d_beta.as_view(), &seqs, HEADS, KEY_HEADS, DK, DV, offsets, false)?;
    }
    stop2.record(&stream)?;
    stop2.synchronize()?;
    let mma_ms = start2.elapsed_ms(&stop2)? as f64 / 3.0;
    println!("  gdn_chunk_uw_mma_f32 (tensor)  {mma_ms:>10.3} ms/call, {:.3}x vs scalar kernel 1", plain_ms / mma_ms);

    // Whole-pipeline context: reg128, for scale.
    let per_layer = HEADS * DK * DV;
    let layers = LINEAR_LAYERS.min((1 << 28) / (per_layer * 4)).max(1);
    let mut d_state = stream.alloc_zeros::<f32>(layers * per_layer)?;
    let mut d_out = stream.alloc_zeros::<f32>(total * HEADS * DV)?;
    let mut run_reg = |iters: usize| -> Result<()> {
        for it in 0..iters {
            let layer = it % layers;
            let mut slice = d_state.slice_mut(layer * per_layer..(layer + 1) * per_layer);
            k.gdn_delta_rule_variant(&mut d_out.as_view_mut(), &mut slice, &d_row.as_view(), &d_g.as_view(), &d_beta.as_view(), &seqs, HEADS, KEY_HEADS, DK, DV, offsets, false, DeltaVariant::Reg)?;
        }
        Ok(())
    };
    run_reg(2)?;
    dev.synchronize()?;
    let start3 = event()?;
    let stop3 = event()?;
    start3.record(&stream)?;
    run_reg(LINEAR_LAYERS)?;
    stop3.record(&stream)?;
    stop3.synchronize()?;
    let reg_ms = start3.elapsed_ms(&stop3)? as f64;
    println!("\n  reg128 (deployed)    {reg_ms:>10.2} ms across {LINEAR_LAYERS} layers ({:.3} ms/layer)", reg_ms / LINEAR_LAYERS as f64);
    println!("  (kernel-1-only numbers above are per single launch, not per-layer-summed -- {n_chunks} chunk-grid launches map to ONE call of gdn_chunk_uw_only/_mma_only each, not {n_chunks} separate launches, since the kernel itself grids over chunks*heads internally)");

    Ok(())
}
