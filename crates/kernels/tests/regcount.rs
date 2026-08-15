//! How many blocks of each GEMM shape does an SM actually hold?
//!
//! `Kernels::kernel_limits` asks whether registers cap the *block size*, and
//! for every kernel in this file the answer is no — they all sit at their
//! `__launch_bounds__`. That answer is nearly useless, and believing it cost
//! real time here: a kernel can spend 40 more registers a thread, still allow
//! 128 threads, and lose a quarter of its resident blocks. `kernel_registers`
//! reports the number that settles it.
//!
//! It settled two things. The deeper weight prefetch (`mmqb...d4`) measured
//! 5-8% slower than depth 2, and this is why: 168 registers against 128, three
//! resident blocks against four. And the f16 path's loss is *not* occupancy —
//! `mmqf2w4s2` fits more blocks than the integer kernel that beats it.

use anyhow::Result;
use tuili_cuda::Device;
use tuili_kernels::Kernels;

/// sm_86: 64K registers and 100 KiB of shared memory per SM.
fn blocks_per_sm(regs: i32, smem: i32, warps: i32) -> i32 {
    let threads = warps * 32;
    let by_reg = if regs > 0 { 65536 / (regs * threads) } else { 32 };
    let by_smem = if smem > 0 { 102_400 / smem } else { 32 };
    by_reg.min(by_smem)
}

#[test]
fn the_gemm_shapes_still_fit_more_than_one_block_per_sm() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    let kern = Kernels::new(dev.clone());

    // Warps per block is part of each name: `...w<warps>...`.
    let shapes = [
        ("mmqsr2w4s4_q4_g128", 4),
        ("mmqsr2w4s4_2_q4_g128", 4),
        ("mmqb2w4s4d2_q4_g128", 4),
        ("mmqb2w4s4d2_2_q4_g128", 4),
        ("mmqb2w4s4d4_q4_g128", 4),
        ("mmqr2w4s4_q4_g128", 4),
        ("mmqa2w4s4_q4_g128", 4),
        ("mmqb1w4s2d2_q4_g128", 4),
        ("mmqb1w4s2d2_2_q4_g128", 4),
        ("mmqb1w4s4d2_q4_g128", 4),
        ("mmqb1w4s4d2_2_q4_g128", 4),
        ("mmql1w4s2d2_q4_g128", 4),
        ("mmql1w4s2d2_2_q4_g128", 4),
        ("mmql2w4s2d2_2_q4_g128", 4),
        // The default for an AWQ checkpoint, which is what a served step
        // actually runs; the deeper weight prefetch below has to fit inside
        // whatever headroom these have.
        ("mmqy1w8s2_2_q4_g128", 8),
        ("mmqy1w8s2_q4_g128", 8),
        ("mmqy2w8s2_2_q4_g128", 8),
        ("mmqf1w8s2_2_q4_g128", 8),
        ("mmqf1w8s2_4_q4_g128", 8),
        ("mmqf1w8s2_q4_g128", 8),
        ("mmqf1w8s2_2_q4_g128", 8),
        ("mmqfp1w8s2_q4_g128", 8),
        ("mmqfp1w8s2_2_q4_g128", 8),
        ("mmqg1w8s2_q4_g128", 8),
        ("mmqg1w8s2_2_q4_g128", 8),
    ];
    for (name, warps) in shapes {
        let (regs, smem) = kern.kernel_registers("tuili_mmq", name)?;
        let blocks = blocks_per_sm(regs, smem, warps);
        eprintln!("  {name:24} {regs:>4} regs {smem:>6} B smem -> {blocks} blocks/SM");
        // One block per SM is the cliff this kernel keeps falling off: every
        // measured win came from having more concurrent weight loads, and a
        // single block cannot supply them however it is written.
        assert!(
            blocks >= 2,
            "{name} fits only {blocks} block per SM ({regs} regs, {smem} B shared)"
        );
    }
    Ok(())
}

/// What a wider row group costs in registers, which is why Marlin's tile shape
/// does not port: at 256 rows and four warps this body needs 255 of them and
/// spills. See the note above `MMQ_Y_SET` in `mmq.cu`.
#[test]
fn the_wide_row_group_shapes_and_their_registers() -> anyhow::Result<()> {
    let dev = match tuili_cuda::Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    let kern = Kernels::new(dev);
    for v in ["mmqy1w8s2_2", "mmqy2w8s2_2", "mmqy4w8s2_2"] {
        let name = format!("{v}_q4_g128");
        let (regs, smem) = kern.kernel_registers("tuili_mmq", &name)?;
        eprintln!("  {v:<14} {regs:>4} regs  {smem:>6} B static smem");
    }
    Ok(())
}

/// Which resource actually caps this GEMM's blocks an SM — asked of the driver,
/// with the dynamic shared the engine really launches.
///
/// The kernel's own note says the 34.8 KB activation ring at two token tiles is
/// what holds it to two blocks. If registers hold it to two as well, then
/// removing the ring buys nothing, and so does trimming registers — which is
/// the difference between "occupancy is the wall" and "occupancy is *two* walls
/// at the same height".
#[test]
fn what_caps_the_blocks_an_sm() -> anyhow::Result<()> {
    let dev = match tuili_cuda::Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    let sm = dev.sm_count();
    let kern = Kernels::new(dev);
    // `MMQ_XF_STRIDE` is 544 and a stage holds `tiles * 16` rows.
    for (name, threads, tiles, stages) in [
        ("mmqy1w8s2_q4_g128", 256u32, 1usize, 2usize),
        ("mmqy1w8s2_2_q4_g128", 256, 2, 2),
    ] {
        let (regs, _) = kern.kernel_registers("tuili_mmq", name)?;
        let stride: usize = std::env::var("TUILI_XF_STRIDE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(544);
        let dynamic = stages * tiles * 16 * stride;
        let blocks = kern.occupancy_blocks("tuili_mmq", name, threads, dynamic)?;
        eprintln!(
            "  {name:<24} {regs:>3} regs  {:>5} B dynamic  -> {blocks} blocks/SM              ({} SMs, {} resident blocks)",
            dynamic,
            sm,
            blocks * sm
        );
    }
    Ok(())
}
