//! Benchmarks the REAL, fully assembled `gdn_full_triton_pipeline` (pack ->
//! cumsum -> kkt -> solve_tril64 -> wy_fast -> kernel-2 (Triton-AOT, already
//! bound in `gdn_triton_aot.rs`) -> chunk_o -> unpack), all real launches
//! included -- not just the 5 new AOT-compiled kernels' own isolated time,
//! and not just kernel-2 alone the way `gdn_triton_aot_probe.rs` does.
//!
//! At the same `T`s and against the same two real, freshly re-measured
//! baselines `gdn_triton_aot_probe.rs` already uses (see that file's own doc
//! comment for how those numbers were obtained and their own caveats).
//!
//!     cargo run --release -p infero-kernels --example gdn_full_triton_pipeline_bench \
//!         --features triton_aot

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::Kernels;
use infero_kernels::gdn::SeqLayout;
use infero_kernels::gdn_fla_stages::GdnFlaScratch;
use infero_kernels::gdn_triton_aot::{H, HG, K, V};

const LINEAR_LAYERS: usize = 48;
const INFERO_REG128_MS_PER_LAYER: f64 = 716.86 / LINEAR_LAYERS as f64;
const INFERO_MMA_TRITON_MATCH_3KERNEL_MS_PER_LAYER: f64 = 1428.28 / LINEAR_LAYERS as f64;

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

fn bench_one(k: &Kernels, total: usize, scratch: &mut GdnFlaScratch) -> Result<f64> {
    let dev = k.device();
    let stream = dev.stream();
    let ctx = dev.context();

    let key_dim = HG * K;
    let val_dim = H * V;
    let stride = 2 * key_dim + val_dim;
    let offsets = (stride, 0usize, key_dim, 2 * key_dim);

    let mut row = pseudo_random(total * stride, 0xf001);
    let g: Vec<f32> = pseudo_random(total * H, 0xf002).iter().map(|v| -v.abs() * 0.6).collect();
    let beta: Vec<f32> = pseudo_random(total * H, 0xf003).iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
    // Scale q/k/v down to a realistic range up front (real callers feed
    // already-normalized q/k; a fresh `pseudo_random` row is not, and an
    // unnormalized huge dot product risks inf/nan feeding into later stages
    // -- harmless for a pure timing run, but avoided here to keep the launch
    // pattern representative rather than pathological).
    for x in row.iter_mut() {
        *x *= 0.02;
    }
    let first = [0i32];
    let ntok = [total as i32];

    let d_row = stream.clone_htod(&row)?;
    let d_g = stream.clone_htod(&g)?;
    let d_beta = stream.clone_htod(&beta)?;
    let d_first = stream.clone_htod(&first)?;
    let d_ntok = stream.clone_htod(&ntok)?;
    let mut d_out = stream.alloc_zeros::<f32>(total * H * V)?;
    let mut d_state = stream.alloc_zeros::<f32>(H * K * V)?;

    let seqs = SeqLayout {
        first_token: &d_first.as_view(),
        n_tokens: &d_ntok.as_view(),
        n_seqs: 1,
        total_tokens: total,
    };

    // Real q/k L2-norm, exactly like every other GDN entry point in this
    // crate expects before the delta rule runs -- once, not part of the
    // timed loop below (every other GDN benchmark in this crate treats it
    // the same way).
    k.gdn_qk_l2norm(&mut d_row.clone().as_view_mut(), total, HG, K, stride, 0, key_dim, 1e-6)?;
    dev.synchronize()?;

    let mut launch = |state: &mut cudarc::driver::CudaSlice<f32>, scratch: &mut GdnFlaScratch| -> Result<()> {
        k.gdn_full_triton_pipeline(
            &mut d_out.as_view_mut(),
            &mut state.as_view_mut(),
            &d_row.as_view(),
            &d_g.as_view(),
            &d_beta.as_view(),
            &seqs,
            H,
            HG,
            K,
            V,
            offsets,
            false,
            scratch,
        )
    };

    for _ in 0..3 {
        launch(&mut d_state, scratch)?;
    }
    dev.synchronize()?;

    const N_REPS: usize = 20;
    let start = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let stop = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    start.record(stream)?;
    for _ in 0..N_REPS {
        launch(&mut d_state, scratch)?;
    }
    stop.record(stream)?;
    stop.synchronize()?;
    let event_ms = start.elapsed_ms(&stop)? as f64 / N_REPS as f64;

    Ok(event_ms)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let dev = Device::new(0)?;
    println!("device: {} (sm_{}, {} SMs)", dev.name(), dev.arch(), dev.sm_count());
    let k = Kernels::new(dev.clone());

    println!(
        "\nbenchmarking the REAL, fully assembled gdn_full_triton_pipeline (pack + cumsum + kkt + \
         solve_tril64 + wy_fast + kernel-2 + chunk_o + unpack), all real launches included:\n"
    );

    // Caller-owned scratch, allocated once for the largest T this run tries
    // and reused across every T below and across every call within a T's own
    // timing loop -- the same convention `gdn_chunk_split3_delta_rule`'s own
    // scratch buffers and `GdnActs` (`crates/model/src/lib.rs`) use. Before
    // this, `gdn_full_triton_pipeline` did a fresh `stream.alloc_zeros` on
    // all 15 intermediate buffers every single call.
    let sizes = [500usize, 2000, 8000, 30552];
    let max_total = *sizes.iter().max().unwrap();
    let mut scratch = GdnFlaScratch::new(&dev, max_total)?;

    for &total in &sizes {
        let ms = bench_one(&k, total, &mut scratch)?;
        let per_layer = ms;
        println!(
            "  T={total:>6}: {ms:>9.4} ms/call = {per_layer:>9.4} ms/layer  ({:.4}x reg128, {:.4}x \
             mma_triton_match-3kernel)",
            per_layer / INFERO_REG128_MS_PER_LAYER,
            per_layer / INFERO_MMA_TRITON_MATCH_3KERNEL_MS_PER_LAYER,
        );
    }

    println!(
        "\n  (both baselines re-measured fresh this session on this same GPU3 at T=30552, per \
         gdn_triton_aot_probe.rs's own doc comment: reg128 {INFERO_REG128_MS_PER_LAYER:.3} ms/layer, \
         mma_triton_match 3-kernel split {INFERO_MMA_TRITON_MATCH_3KERNEL_MS_PER_LAYER:.3} ms/layer)"
    );
    println!(
        "  NOTE: this number includes every real launch this pipeline actually does, including the \
         pack/unpack layout-conversion kernels and per-call fresh scratch allocation -- see \
         gdn_fla_stages.rs's own doc comment for what is and is not amortized."
    );

    Ok(())
}
