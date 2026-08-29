//! What the gated delta rule costs, and what its state costs it.
//!
//! The recurrent state is `dk * dv` f32 a head — 64 KiB at the 27B's 128 by
//! 128 — and it does not grow with the chunk. So a kernel that leaves it in
//! global memory pays for all of it twice a token, forever: 128 KiB a head a
//! token, which across 48 value heads and the 48 linear layers of one decode
//! step is 288 MiB of traffic for a single token. This measures the places the
//! state can live while a chunk is consumed, at the shapes a served step
//! actually issues.
//!
//!     cargo run --release -p infero-kernels --example gdn_delta_bench
//!
//! One thing this harness has to get right, and got wrong first: a single
//! sequence's state is 3 MiB, which fits this card's 4 MiB L2, so timing the
//! same launch in a loop measures an L2-resident state and flatters every
//! variant. A real step touches 48 different layers' states, so the loop
//! rotates through that many buffers and each launch starts cold.

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::Kernels;
use infero_kernels::gdn::{DeltaVariant, SeqLayout};

/// The 27B's linear-attention shape: `linear_num_value_heads = 48`,
/// `linear_num_key_heads = 16`, `linear_{key,value}_head_dim = 128`.
const HEADS: usize = 48;
const KEY_HEADS: usize = 16;
const DK: usize = 128;
const DV: usize = 128;
/// 48 of the 27B's 64 layers are GatedDeltaNet, so a whole step runs this
/// kernel that many times — and touches that many distinct state buffers.
const LINEAR_LAYERS: usize = 48;

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

/// label, kernel name, threads a block, dynamic shared bytes — matching what
/// `Kernels::gdn_delta_rule_variant` launches for each.
const VARIANTS: [(&str, DeltaVariant, &str, u32, usize); 3] = [
    ("global", DeltaVariant::Global, "gdn_delta_rule_f32", 128, 2 * DK * 4),
    (
        "smem",
        DeltaVariant::Shared,
        "gdn_delta_rule_smem_f32",
        128,
        (2 * DK + DK * DV) * 4,
    ),
    ("reg", DeltaVariant::Reg, "gdn_delta_rule_reg128_f32", 256, 4 * DK * 4),
];

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let dev = Device::new(0)?;
    println!(
        "device: {} (sm_{}, {} SMs)",
        dev.name(),
        dev.arch(),
        dev.sm_count()
    );
    let k = Kernels::new(dev.clone());
    let ctx = dev.context().clone();
    let stream = dev.stream().clone();

    // What the compiler actually did. The register versions' whole premise is
    // that the state lives in registers; a non-zero spill column means it
    // lives in local memory instead, which is the same DRAM the global version
    // streams, and the timings below would be comparing two global kernels.
    println!("\nkernel resources");
    println!(
        "  {:<6} {:<28} {:>5} {:>8} {:>8} {:>7} {:>9}",
        "", "", "regs", "spill B", "threads", "blk/SM", "warps/SM"
    );
    for (label, _v, name, threads, dynamic) in VARIANTS {
        let (regs, _static, spill) = k.gdn_kernel_registers(name)?;
        let blocks = k.gdn_occupancy_blocks(name, threads, dynamic)?;
        println!(
            "  {label:<6} {name:<28} {regs:>5} {spill:>8} {threads:>8} {blocks:>7} {:>9}",
            blocks * threads / 32
        );
    }

    let key_dim = KEY_HEADS * DK;
    let val_dim = HEADS * DV;
    let stride = 2 * key_dim + val_dim;
    let offsets = (stride, 0, key_dim, 2 * key_dim);

    for (label, lens) in [
        ("decode, 1 seq", vec![1usize]),
        ("decode, 32 seqs", vec![1usize; 32]),
        ("prefill 512, 1 seq", vec![512usize]),
    ] {
        let n_seqs = lens.len();
        let total: usize = lens.iter().sum();
        let per_seq = total / n_seqs;

        let row = pseudo_random(total * stride, 0xd317);
        // Non-positive g, as `gdn_gate_decay_f32` guarantees: exp(g) has to be
        // a decay or the state diverges over the repeated launches and this
        // ends up timing arithmetic on infinities.
        let g: Vec<f32> = pseudo_random(total * HEADS, 0xd318)
            .iter()
            .map(|v| -v.abs() * 0.6)
            .collect();
        let beta: Vec<f32> = pseudo_random(total * HEADS, 0xd319)
            .iter()
            .map(|v| 1.0 / (1.0 + (-v).exp()))
            .collect();
        let mut first = Vec::with_capacity(n_seqs);
        let mut at = 0i32;
        for &len in &lens {
            first.push(at);
            at += len as i32;
        }
        let ntok: Vec<i32> = lens.iter().map(|v| *v as i32).collect();

        let d_row = stream.clone_htod(&row)?;
        let d_g = stream.clone_htod(&g)?;
        let d_beta = stream.clone_htod(&beta)?;
        let d_first = stream.clone_htod(&first)?;
        let d_ntok = stream.clone_htod(&ntok)?;
        let mut d_out = stream.alloc_zeros::<f32>(total * HEADS * DV)?;

        // One state buffer a layer, up to what fits: the state of a single
        // sequence is 3 MiB and this card's L2 is 4 MiB, so a loop over one
        // buffer measures a cache that a real step does not have.
        let per_layer = n_seqs * HEADS * DK * DV;
        let layers = (LINEAR_LAYERS).min((1 << 28) / (per_layer * 4)).max(1);
        let mut d_state = stream.alloc_zeros::<f32>(layers * per_layer)?;

        let seqs = SeqLayout {
            first_token: &d_first.as_view(),
            n_tokens: &d_ntok.as_view(),
            n_seqs,
            total_tokens: total,
        };

        // The state traffic each version is structurally obliged to move per
        // launch: the global one streams the whole state twice a token, the
        // blocked ones once each way for the whole chunk. Plus q, k, v and the
        // output, which none of them can avoid.
        let state_bytes = (per_layer * 4) as f64;
        let act_bytes = (total * (2 * KEY_HEADS * DK + 2 * HEADS * DV) * 4) as f64;

        println!(
            "\n{label} ({total} tokens, {} blocks, {} state buffers of {:.1} MiB)",
            HEADS * n_seqs,
            layers,
            state_bytes / (1 << 20) as f64
        );
        println!(
            "  {:<8} {:>10} {:>10} {:>12} {:>13}",
            "variant", "us/launch", "GB/s", "step ms x48", "tok/s ceiling"
        );
        for (vlabel, variant, name, _threads, _shared) in VARIANTS {
            let mut run = |iters: usize| -> Result<()> {
                for it in 0..iters {
                    let layer = it % layers;
                    let mut slice =
                        d_state.slice_mut(layer * per_layer..(layer + 1) * per_layer);
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
                        // V heads grouped by key head, as a Hugging Face
                        // checkpoint stores them.
                        false,
                        variant,
                    )?;
                }
                Ok(())
            };
            run(layers.max(4))?;
            dev.synchronize()?;

            // Events rather than wall clock: at a few microseconds a launch the
            // CPU's submission cost is the same order as the kernel, and wall
            // time cannot tell them apart.
            let iters = if total > 64 { 4 * layers } else { 20 * layers };
            let start = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            let stop = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            start.record(&stream)?;
            run(iters)?;
            stop.record(&stream)?;
            stop.synchronize()?;
            let us = start.elapsed_ms(&stop)? as f64 * 1e3 / iters as f64;

            let moved = if name == "gdn_delta_rule_f32" {
                state_bytes * 4.0 * per_seq as f64
            } else {
                state_bytes * 2.0
            } + act_bytes;
            let step_ms = us * LINEAR_LAYERS as f64 / 1e3;
            println!(
                "  {vlabel:<8} {us:>10.1} {:>10.1} {step_ms:>12.2} {:>13.0}",
                moved / (us * 1e3),
                1e3 / step_ms * n_seqs as f64
            );
        }
    }

    println!(
        "\n`tok/s ceiling` is this kernel alone across {LINEAR_LAYERS} linear layers with \
         every other\nkernel in the step costing zero — an upper bound, not a projection."
    );
    Ok(())
}
