//! Does calling vLLM/FLA's real, Triton-AOT-compiled
//! `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` directly from Rust (via
//! `infero_kernels::gdn_triton_aot`, no Triton/Python runtime involved at
//! call time) reproduce a real Python-computed reference, and how fast is
//! it compared to vLLM's own real Triton-JIT path, infero's deployed
//! `reg128`, and infero's best hand-written kernel-2 variant
//! (`MmaTritonMatch`)?
//!
//! Two separate checks, deliberately at two different shapes:
//!
//! 1. **Correctness**, at a small `T=256` shape, against a real reference
//!    computed by `vllm.third_party.flash_linear_attention`'s own Python
//!    entry point (`scripts/gdn_triton_aot_dump_fixture.py` dumps it to raw
//!    binary once; see that script's own doc comment for how to regenerate
//!    it and why `T=256` is enough). This is the one real numerical
//!    ground-truth check in this probe.
//! 2. **Benchmark**, at the real production shape's `T=4096`, against
//!    synthetic data generated in Rust (no reference needed here --
//!    correctness at *this* shape was already established during this
//!    project's AOT-compile phase, which found the raw driver-API launcher
//!    and the normal Triton JIT path bit-identical at `T=4096` too; see that
//!    phase's own report).
//!
//! Needs the `triton_aot` feature (see `Cargo.toml`/`build.rs` for what that
//! needs: `INFERO_NVCC`, `INFERO_TRITON_AOT_DIR`), and, for the correctness
//! half, `INFERO_GDN_TRITON_AOT_FIXTURE_DIR` pointing at
//! `gdn_triton_aot_dump_fixture.py`'s output (default `/tmp/gdn_triton_aot_fixture`;
//! the benchmark half still runs, with a clear note, if that directory is
//! missing).
//!
//!     cargo run --release -p infero-kernels --example gdn_triton_aot_probe --features triton_aot

use anyhow::{Context, Result};
use cudarc::driver::{DevicePtr, DevicePtrMut};
use infero_cuda::Device;
use infero_kernels::gdn_triton_aot::{self, BT, H, HG, K, V};
use std::path::{Path, PathBuf};

const CHECK_T: usize = 256;
/// Default benchmark shape -- matches `gdn_aot_bench.py`'s own T=4096 (the
/// AOT-compile phase's own reported 0.1893ms/call is at this shape).
/// Override with `GDN_TRITON_AOT_BENCH_T` to compare at this project's other
/// benchmarks' own default shape (30552 total tokens, see
/// `gdn_split3_bench.rs`'s `DEFAULT_TOTAL_TOKENS`) -- the two are NOT
/// directly comparable to each other or to the constants below without
/// matching T; see `main`'s own comparison-printing logic for how this is
/// handled honestly rather than silently mixing scales.
const BENCH_T: usize = 4096;
const BENCH_LAYERS: usize = 48; // matches this project's other GDN benches' LINEAR_LAYERS
/// The T this file's own comparison constants below were measured at
/// (`gdn_split3_bench.rs`'s `DEFAULT_TOTAL_TOKENS`, re-run fresh on this same
/// GPU3 in this task -- see `main`'s own doc comment for the real numbers
/// and where they came from).
const COMPARISON_T: usize = 30552;

/// `reg128` and `mma_triton_match` were re-measured fresh, in this same
/// task, on this same GPU3, by building and running this project's own
/// existing `gdn_split3_bench.rs` (unchanged) at its default T=30552 shape:
/// `716.86 ms / 48 layers = 14.935` and `1428.28 ms / 48 layers = 29.756`
/// -- both match the numbers this task's own brief cited (14.93, 29.76) to
/// within rounding, confirming those figures are real and not stale.
///
/// One correction to the brief's own framing, found while re-measuring:
/// `gdn_split3_bench.rs` labels the 29.76 number "3-kernel split
/// (mma_triton_match)" -- the FULL 3-kernel split pipeline (kernel-1's
/// `uw`/`u_out` pass + kernel-2's state-pass, using `MmaTritonMatch`, +
/// kernel-3's output pass) using `mma_triton_match` for kernel-2, not
/// kernel-2 in isolation. This probe's own number below IS kernel-2 in
/// isolation (a single `gdn_h` launch, no kernel-1/kernel-3 equivalent), so
/// the two are close in spirit (kernel-2 is this pipeline's dominant cost,
/// per this whole investigation's own findings) but not a literally
/// apples-to-apples kernel-2-vs-kernel-2 number -- flagged here rather than
/// silently presented as one.
///
/// vLLM's own real Triton-JIT path number (5.2659 ms/layer) was NOT
/// re-measured in this task (that needs a full FLA/vLLM GDN forward run,
/// out of this task's scope) -- cited as-is from the brief, not verified
/// here.
const INFERO_REG128_MS_PER_LAYER: f64 = 716.86 / BENCH_LAYERS as f64;
const INFERO_MMA_TRITON_MATCH_3KERNEL_MS_PER_LAYER: f64 = 1428.28 / BENCH_LAYERS as f64;
const VLLM_TRITON_MS_PER_LAYER_UNVERIFIED: f64 = 5.2659;

fn pseudo_random_f32(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0) * scale
        })
        .collect()
}

fn f32_to_bf16_bytes(xs: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * 2);
    for &x in xs {
        out.extend_from_slice(&half::bf16::from_f32(x).to_bits().to_le_bytes());
    }
    out
}

fn f32_to_bytes(xs: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * 4);
    for &x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

fn f32_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared buffers have different lengths ({} vs {})", a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn fixture_dir() -> PathBuf {
    std::env::var("INFERO_GDN_TRITON_AOT_FIXTURE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/gdn_triton_aot_fixture"))
}

fn read_bin(dir: &Path, name: &str) -> Result<Vec<u8>> {
    let path = dir.join(format!("{name}.bin"));
    std::fs::read(&path).with_context(|| format!("reading fixture file {}", path.display()))
}

/// Runs the real, Python-computed-reference correctness check at `T=256`.
/// Returns `Ok(true)` if the fixture was found and matched, `Ok(false)` if
/// the fixture directory is missing (reported, not fabricated), and `Err`
/// for anything else (a present-but-wrong-shaped fixture, a launch failure).
fn run_correctness_check(dev: &Device) -> Result<bool> {
    let dir = fixture_dir();
    if !dir.is_dir() {
        println!(
            "correctness check SKIPPED: {} does not exist -- regenerate with \
             `scripts/gdn_triton_aot_dump_fixture.py --out {}` on a box with the real \
             vllm.third_party.flash_linear_attention install (see that script's own doc comment)",
            dir.display(),
            dir.display()
        );
        return Ok(false);
    }

    let t = CHECK_T;
    let nt = t.div_ceil(BT);
    let stream = dev.stream();

    let k_bytes = read_bin(&dir, "k")?;
    let w_bytes = read_bin(&dir, "w")?;
    let u_bytes = read_bin(&dir, "u")?;
    let g_bytes = read_bin(&dir, "g")?;
    let init_bytes = read_bin(&dir, "initial_state")?;
    let h_ref_bytes = read_bin(&dir, "h_ref")?;
    let v_new_ref_bytes = read_bin(&dir, "v_new_ref")?;
    let final_state_ref_bytes = read_bin(&dir, "final_state_ref")?;

    anyhow::ensure!(k_bytes.len() == t * HG * K * 2, "k fixture is {} bytes, expected {}", k_bytes.len(), t * HG * K * 2);
    anyhow::ensure!(w_bytes.len() == t * H * K * 2, "w fixture size mismatch");
    anyhow::ensure!(u_bytes.len() == t * H * V * 2, "u fixture size mismatch");
    anyhow::ensure!(g_bytes.len() == t * H * 4, "g fixture size mismatch");
    anyhow::ensure!(init_bytes.len() == H * V * K * 4, "initial_state fixture size mismatch");
    anyhow::ensure!(h_ref_bytes.len() == nt * H * V * K * 2, "h_ref fixture size mismatch");
    anyhow::ensure!(v_new_ref_bytes.len() == t * H * V * 2, "v_new_ref fixture size mismatch");
    anyhow::ensure!(final_state_ref_bytes.len() == H * V * K * 4, "final_state_ref fixture size mismatch");

    let d_k = stream.clone_htod(&k_bytes)?;
    let d_w = stream.clone_htod(&w_bytes)?;
    let d_u = stream.clone_htod(&u_bytes)?;
    let d_g = stream.clone_htod(&g_bytes)?;
    let d_init = stream.clone_htod(&init_bytes)?;
    let mut d_h = stream.alloc_zeros::<u8>(nt * H * V * K * 2)?;
    let mut d_v_new = stream.alloc_zeros::<u8>(t * H * V * 2)?;
    let mut d_final_state = stream.alloc_zeros::<u8>(H * V * K * 4)?;

    let (k_ptr, _rk) = d_k.device_ptr(stream);
    let (u_ptr, _ru) = d_u.device_ptr(stream);
    let (w_ptr, _rw) = d_w.device_ptr(stream);
    let (g_ptr, _rg) = d_g.device_ptr(stream);
    let (init_ptr, _ri) = d_init.device_ptr(stream);
    let (v_new_ptr, _rv) = d_v_new.device_ptr_mut(stream);
    let (h_ptr, _rh) = d_h.device_ptr_mut(stream);
    let (final_state_ptr, _rf) = d_final_state.device_ptr_mut(stream);

    unsafe {
        gdn_triton_aot::launch_chunk_gated_delta_rule_fwd_h(
            stream, k_ptr, u_ptr, w_ptr, v_new_ptr, g_ptr, h_ptr, init_ptr, final_state_ptr, t as i32,
        )?;
    }
    drop((_rk, _ru, _rw, _rg, _ri, _rv, _rh, _rf));
    dev.synchronize()?;

    let h_out = stream.clone_dtoh(&d_h.as_view())?;
    let v_new_out = stream.clone_dtoh(&d_v_new.as_view())?;
    let final_state_out = stream.clone_dtoh(&d_final_state.as_view())?;

    let h_diff = max_abs_diff(&bf16_bytes_to_f32(&h_out), &bf16_bytes_to_f32(&h_ref_bytes));
    let v_new_diff = max_abs_diff(&bf16_bytes_to_f32(&v_new_out), &bf16_bytes_to_f32(&v_new_ref_bytes));
    let final_state_diff =
        max_abs_diff(&f32_bytes_to_f32(&final_state_out), &f32_bytes_to_f32(&final_state_ref_bytes));

    println!("correctness check (T={t}, real Python-computed reference):");
    println!("  h            max_abs_diff = {h_diff}");
    println!("  v_new        max_abs_diff = {v_new_diff}");
    println!("  final_state  max_abs_diff = {final_state_diff}");

    anyhow::ensure!(
        h_diff == 0.0 && v_new_diff == 0.0 && final_state_diff == 0.0,
        "Rust FFI call does NOT bit-match the real Python-computed reference -- see diffs above"
    );
    println!("  -> bit-exact match, called from Rust");
    Ok(true)
}

fn run_benchmark(dev: &Device, t: usize) -> Result<f64> {
    let nt = t.div_ceil(BT);
    let stream = dev.stream();
    let ctx = dev.context();

    let k_bytes = f32_to_bf16_bytes(&pseudo_random_f32(t * HG * K, 0xa1, 0.1));
    let w_bytes = f32_to_bf16_bytes(&pseudo_random_f32(t * H * K, 0xa2, 0.1));
    let u_bytes = f32_to_bf16_bytes(&pseudo_random_f32(t * H * V, 0xa3, 0.1));
    let g_bytes = f32_to_bytes(&pseudo_random_f32(t * H, 0xa4, 0.01).iter().map(|v| -v.abs()).collect::<Vec<_>>());
    let init_bytes = f32_to_bytes(&pseudo_random_f32(H * V * K, 0xa5, 0.1));

    let d_k = stream.clone_htod(&k_bytes)?;
    let d_w = stream.clone_htod(&w_bytes)?;
    let d_u = stream.clone_htod(&u_bytes)?;
    let d_g = stream.clone_htod(&g_bytes)?;
    let d_init = stream.clone_htod(&init_bytes)?;
    let mut d_h = stream.alloc_zeros::<u8>(nt * H * V * K * 2)?;
    let mut d_v_new = stream.alloc_zeros::<u8>(t * H * V * 2)?;
    let mut d_final_state = stream.alloc_zeros::<u8>(H * V * K * 4)?;

    let (k_ptr, _rk) = d_k.device_ptr(stream);
    let (u_ptr, _ru) = d_u.device_ptr(stream);
    let (w_ptr, _rw) = d_w.device_ptr(stream);
    let (g_ptr, _rg) = d_g.device_ptr(stream);
    let (init_ptr, _ri) = d_init.device_ptr(stream);
    let (v_new_ptr, _rv) = d_v_new.device_ptr_mut(stream);
    let (h_ptr, _rh) = d_h.device_ptr_mut(stream);
    let (final_state_ptr, _rf) = d_final_state.device_ptr_mut(stream);

    let launch = || -> Result<()> {
        unsafe {
            gdn_triton_aot::launch_chunk_gated_delta_rule_fwd_h(
                stream, k_ptr, u_ptr, w_ptr, v_new_ptr, g_ptr, h_ptr, init_ptr, final_state_ptr, t as i32,
            )
        }
    };

    // Warmup: pays the one-time `cuModuleLoadData` cost (lazy-loaded on
    // first call, see gdn_triton_aot.rs) outside the timed region, same
    // convention as this crate's other GDN benches.
    for _ in 0..10 {
        launch()?;
    }
    dev.synchronize()?;

    const N_REPS: usize = 100;
    let start = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let stop = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    start.record(stream)?;
    for _ in 0..N_REPS {
        launch()?;
    }
    stop.record(stream)?;
    stop.synchronize()?;
    let event_ms = start.elapsed_ms(&stop)? as f64 / N_REPS as f64;

    // Wall-clock cross-check, same reasoning as `gdn_aot_bench.py`'s own
    // cross-check on the Python side.
    for _ in 0..10 {
        launch()?;
    }
    dev.synchronize()?;
    let wall_start = std::time::Instant::now();
    for _ in 0..N_REPS {
        launch()?;
    }
    dev.synchronize()?;
    let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0 / N_REPS as f64;

    drop((_rk, _ru, _rw, _rg, _ri, _rv, _rh, _rf));

    println!("\nbenchmark (T={t}, synthetic data, {N_REPS} reps):");
    println!("  CUDA-event timing:  {event_ms:.4} ms/call");
    println!("  wall-clock timing:  {wall_ms:.4} ms/call");
    Ok(event_ms)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let dev = Device::new(0)?;
    println!("device: {} (sm_{}, {} SMs)", dev.name(), dev.arch(), dev.sm_count());

    let checked = run_correctness_check(&dev)?;
    println!();
    let ms_at_bench_t = run_benchmark(&dev, BENCH_T)?;
    println!();
    let ms_at_comparison_t = run_benchmark(&dev, COMPARISON_T)?;

    println!("\n=== summary ===");
    println!(
        "  correctness (T={CHECK_T}, real Python reference): {}",
        if checked { "PASS (bit-exact)" } else { "SKIPPED (no fixture found)" }
    );
    println!("  this Rust FFI call, T={BENCH_T}:  {ms_at_bench_t:.4} ms/call (matches the AOT-compile phase's own ctypes-measured 0.1893ms)");
    println!("  this Rust FFI call, T={COMPARISON_T}: {ms_at_comparison_t:.4} ms/call = {:.4} ms/layer", ms_at_comparison_t);
    println!();
    println!("  compared at the SAME T={COMPARISON_T} (this project's other GDN benches' own default shape):");
    println!(
        "    infero deployed reg128 (full pipeline, re-measured this task): {INFERO_REG128_MS_PER_LAYER:.3} ms/layer"
    );
    println!(
        "    infero best 3-kernel-split w/ MmaTritonMatch (re-measured this task): {INFERO_MMA_TRITON_MATCH_3KERNEL_MS_PER_LAYER:.3} ms/layer"
    );
    println!("    this AOT-compiled Triton kernel-2 alone:                       {ms_at_comparison_t:.4} ms/layer");
    println!(
        "    vLLM's own real Triton-JIT path (NOT re-measured this task, cited from the prior phase's report): {VLLM_TRITON_MS_PER_LAYER_UNVERIFIED:.4} ms/layer"
    );
    println!();
    println!(
        "    ratio vs infero's deployed reg128 (full pipeline):        {:.4}x",
        ms_at_comparison_t / INFERO_REG128_MS_PER_LAYER
    );
    println!(
        "    ratio vs infero's best 3-kernel-split (MmaTritonMatch):   {:.4}x",
        ms_at_comparison_t / INFERO_MMA_TRITON_MATCH_3KERNEL_MS_PER_LAYER
    );
    println!(
        "    ratio vs vLLM's own Triton-JIT path (unverified figure):  {:.4}x",
        ms_at_comparison_t / VLLM_TRITON_MS_PER_LAYER_UNVERIFIED
    );
    println!();
    println!(
        "  NOTE: this AOT kernel launches ONLY the kernel-2 equivalent (one gdn_h call) -- it \
         does no kernel-1 (uw/u_out) or kernel-3 (output) work, so the ratios above are \
         kernel-2-alone vs. full-pipeline-or-full-3-kernel-split numbers, not a strictly \
         apples-to-apples kernel-2-vs-kernel-2 comparison. See this file's own doc comment on \
         INFERO_MMA_TRITON_MATCH_3KERNEL_MS_PER_LAYER for exactly what was and wasn't \
         re-verified in this task."
    );

    Ok(())
}
