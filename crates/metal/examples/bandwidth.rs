//! What this machine can actually do, measured rather than quoted.
//!
//! Three numbers decide the decode ceiling of a memory-bound model:
//!
//!   1. streaming read bandwidth -- the wall
//!   2. what the real quantized mat-vec achieves against it
//!   3. per-dispatch overhead, since a 27B decode step issues ~880 of them
//!
//! Reported as GB/s so they can be divided into a weight budget directly.

use anyhow::Result;
use tuili_metal::{Device, LaunchConfig};

const COMMON: &str = include_str!("../../kernels/src/msl/common.metal");
const QUANT: &str = include_str!("../../kernels/src/msl/quant.metal");

const BENCH: &str = r#"
#include <metal_stdlib>
using namespace metal;

/// Vectorised streaming read. One float4 a thread a stride, summed so the
/// compiler cannot elide the loads.
kernel void stream_read(device float* out        [[buffer(0)]],
                        device const float4* x   [[buffer(1)]],
                        constant int& n4         [[buffer(2)]],
                        uint3 tgid  [[threadgroup_position_in_grid]],
                        uint3 tid   [[thread_position_in_threadgroup]],
                        uint3 tgdim [[threads_per_threadgroup]],
                        uint3 ngrid [[threadgroups_per_grid]]) {
    const int stride = int(ngrid.x * tgdim.x);
    float4 acc = 0.0f;
    for (int i = int(tgid.x * tgdim.x + tid.x); i < n4; i += stride) {
        acc += x[i];
    }
    // One store a thread; negligible against the reads.
    if (acc.x == 12345.678f) out[0] = acc.x + acc.y + acc.z + acc.w;
}

kernel void nothing(device float* out [[buffer(0)]]) {
    if (out[0] == 12345.678f) out[0] = 1.0f;
}
"#;

fn ms(mut f: impl FnMut() -> Result<()>, iters: usize) -> Result<f64> {
    f()?; // warm
    let t = std::time::Instant::now();
    for _ in 0..iters {
        f()?;
    }
    Ok(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

fn main() -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();
    println!(
        "{} | {:.1} GiB working set\n",
        dev.name(),
        dev.working_set_bytes() as f64 / (1u64 << 30) as f64
    );

    // ---- 1. streaming read ------------------------------------------------
    let gib = 4usize;
    let n = gib * (1 << 30) / 4;
    let buf = s.alloc_zeros::<f32>(n)?;
    let mut out = s.alloc_zeros::<f32>(1)?;
    let f = dev.kernels().get("bench", &format!("{COMMON}\n{BENCH}"), "stream_read")?;
    let n4 = (n / 4) as i32;

    println!("streaming read of {gib} GiB (float4 loads):");
    let mut best = 0.0f64;
    for groups in [512u32, 2048, 8192, 32768] {
        let t = ms(
            || {
                let mut b = s.launch_builder(&f);
                b.arg(&out.as_view_mut()).arg(&buf.as_view()).arg(&n4);
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: (groups, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })?
                };
                s.synchronize()
            },
            5,
        )?;
        let gbs = (gib as f64 * (1u64 << 30) as f64) / (t / 1e3) / 1e9;
        best = best.max(gbs);
        println!("  {groups:>6} threadgroups   {t:8.2} ms   {gbs:7.1} GB/s");
    }
    println!("  -> ceiling {best:.1} GB/s\n");

    // ---- 2. per-dispatch overhead ----------------------------------------
    let nf = dev.kernels().get("bench", &format!("{COMMON}\n{BENCH}"), "nothing")?;
    let t = ms(
        || {
            for _ in 0..100 {
                let mut b = s.launch_builder(&nf);
                b.arg(&out.as_view_mut());
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    })?
                };
            }
            s.synchronize()
        },
        10,
    )?;
    let per = t * 1e3 / 100.0;
    println!("empty dispatch: {per:.1} us each (batched into one command buffer)");
    println!("  a 27B decode step issues ~880 -> {:.1} ms of pure overhead\n", per * 880.0 / 1e3);

    // ---- 3. the real quantized mat-vec ------------------------------------
    // Shapes from the 27B, with the encoding each actually uses.
    let quant = format!("{COMMON}\n{QUANT}");
    // Each shape at the widths the engine actually launches, with the kernel it
    // picks there. Decode is one token; speculation's verification pass is
    // `k + 1`, so two and three are the widths that decide whether a
    // speculative round pays for itself.
    //
    // The number to read is not GB/s -- it is the **wall time against the
    // one-token row**. A mat-vec reads its weights once whatever the token
    // count, so if two tokens cost what one token costs, the second token is
    // free and a two-row verification pass is as cheap as the decode step it
    // replaces. If two tokens cost twice as much, nothing is amortising and
    // speculation cannot win no matter how good the drafter is.
    for (label, kernel, tokens, k, n_rows, bytes_per_256) in [
        ("output.weight     Q6_K", "gemv1_q6_K", 1usize, 5120usize, 248320usize, 210.0f64),
        ("output.weight     Q6_K", "gemv2_q6_K", 2, 5120, 248320, 210.0),
        ("output.weight     Q6_K", "gemv4_q6_K", 4, 5120, 248320, 210.0),
        ("output.weight     Q6_K", "gemv_q6_K", 8, 5120, 248320, 210.0),
        ("ffn_down          Q4_K", "gemv1_q4_K", 1, 17408, 5120, 144.0),
        ("ffn_down          Q4_K", "gemv2_q4_K", 2, 17408, 5120, 144.0),
        ("ffn_down          Q4_K", "gemv4_q4_K", 4, 17408, 5120, 144.0),
        ("ffn_down          Q4_K", "gemv_q4_K", 8, 17408, 5120, 144.0),
        ("ffn_gate/up       Q4_K", "gemv1_q4_K", 1, 5120, 17408, 144.0),
        ("ffn_gate/up       Q4_K", "gemv2_q4_K", 2, 5120, 17408, 144.0),
        ("attn_qkv          Q8_0", "gemv1_q8_0", 1, 5120, 10240, 34.0 * 8.0),
        ("attn_qkv          Q8_0", "gemv2_q8_0", 2, 5120, 10240, 34.0 * 8.0),
    ] {
        let bytes = (k * n_rows) as f64 * bytes_per_256 / 256.0;
        let w = s.alloc_zeros::<u8>(bytes as usize)?;
        let x = s.alloc_zeros::<f32>(k * tokens)?;
        let mut o = s.alloc_zeros::<f32>(n_rows * tokens)?;
        let f = dev.kernels().get("quant", &quant, kernel)?;
        let (ki, ni, ti) = (k as i32, n_rows as i32, tokens as i32);
        // `token0 = tgid.y * T` in the kernel, so the grid follows the
        // specialisation's width rather than a constant.
        let per_block = match kernel.as_bytes()[4] {
            b'1' => 1u32,
            b'2' => 2,
            b'4' => 4,
            _ => 8,
        };
        // The group width the host would choose for this shape, capped at 128.
        let items = match kernel.as_bytes()[kernel.len() - 1] {
            b'0' => k / 8,            // q8_0
            b'K' if kernel.contains("q4") => k / 32,
            b'K' => k / 4,            // q6_K
            _ => k,
        } as u32;
        let block = items.next_multiple_of(32).clamp(32, 128);
        let t = ms(
            || {
                let mut b = s.launch_builder(&f);
                b.arg(&o.as_view_mut())
                    .arg(&w.as_view())
                    .arg(&x.as_view())
                    .arg(&ki)
                    .arg(&ni)
                    .arg(&ti);
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: (
                            n_rows as u32,
                            (tokens as u32).div_ceil(per_block).max(1),
                            1,
                        ),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    })?
                };
                s.synchronize()
            },
            5,
        )?;
        let gbs = bytes / (t / 1e3) / 1e9;
        println!(
            "  {label}  {tokens} tok  {:>7.0} MiB  {t:8.2} ms  {gbs:7.1} GB/s              ({:.0}% of ceiling)  {:.2} ms/token",
            bytes / (1u64 << 20) as f64,
            gbs / best * 100.0,
            t / tokens as f64
        );
    }
    // ---- 3b. four output rows a threadgroup -------------------------------
    println!("\n  one row a group vs four (Q4_K, one token)");
    for (label, k, n_rows) in [
        ("ffn_gate/up", 5120usize, 17408usize),
        ("ffn_down   ", 17408, 5120),
    ] {
        let bytes = (k * n_rows) as f64 * 144.0 / 256.0;
        let w = s.alloc_zeros::<u8>(bytes as usize)?;
        let x = s.alloc_zeros::<f32>(k * 256)?;
        let mut o = s.alloc_zeros::<f32>(n_rows * 256)?;
        let ki = k as i32;
        let ni = n_rows as i32;
        let mut line = format!("  {label} ");
        for (kernel, rows, tokens) in [
            ("gemv1_q4_K", 1u32, 1usize),
            ("gemv1x4_q4_K", 4, 1),
            ("gemv2_q4_K", 1, 2),
            ("gemv2x4_q4_K", 4, 2),
            ("gemv4_q4_K", 1, 4),
            ("gemv4x4_q4_K", 4, 4),
            // A prefill chunk. `gemv_q4_K` is what the host launched before the
            // multi-row path existed; `gemv4x4` is what it launches now.
            ("gemv_q4_K", 1, 256),
            ("gemv4x4_q4_K", 4, 256),
        ] {
            let f = dev.kernels().get("quant", &quant, kernel)?;
            let ti = tokens as i32;
            let per_tok = if kernel.starts_with("gemv1") {
                1u32
            } else if kernel.starts_with("gemv2") {
                2
            } else if kernel.starts_with("gemv4") {
                4
            } else {
                8
            };
            let mut best_g = 0.0f64;
            let mut best_ms = f64::INFINITY;
            for block in [32u32, 64, 128] {
                let t = ms(
                    || {
                        let mut b = s.launch_builder(&f);
                        b.arg(&o.as_view_mut())
                            .arg(&w.as_view())
                            .arg(&x.as_view())
                            .arg(&ki)
                            .arg(&ni)
                            .arg(&ti);
                        unsafe {
                            b.launch(LaunchConfig {
                                grid_dim: (
                                    (n_rows as u32).div_ceil(rows),
                                    (tokens as u32).div_ceil(per_tok).max(1),
                                    1,
                                ),
                                block_dim: (block, 1, 1),
                                shared_mem_bytes: 0,
                            })?
                        };
                        s.synchronize()
                    },
                    5,
                )?;
                best_g = best_g.max(bytes / (t / 1e3) / 1e9);
                best_ms = best_ms.min(t);
            }
            let _ = best_g;
            line.push_str(&format!("  {tokens}tok x{rows}row: {best_ms:5.2}ms"));
        }
        println!("{line}");
    }

    // ---- 4. how much of a mat-vec is its block reduction ------------------
    //
    // The host sizes the threadgroup from `gemv_work_items(k)`, so a Q4_K
    // mat-vec at k = 5120 gets 160 threads and each one handles exactly one
    // 32-element block: no loop, and the whole kernel is a launch plus a
    // block-wide reduction. That should be visible as a block-size sweep with a
    // minimum well below 160 -- fewer threads, each looping, and the reduction
    // over fewer lanes.
    //
    // The loop is `for (c = tid.x; c < chunks; c += tgdim.x)`, self-bounding at
    // any width, so every row here computes the same answer.
    println!("\n  block-size sweep (same kernel, same answer, different group width)");
    for (label, kernel, k, n_rows, bytes_per_256) in [
        ("ffn_gate/up  Q4_K", "gemv1_q4_K", 5120usize, 17408usize, 144.0f64),
        ("ffn_down     Q4_K", "gemv1_q4_K", 17408, 5120, 144.0),
        ("attn_qkv     Q8_0", "gemv1_q8_0", 5120, 10240, 34.0 * 8.0),
        ("output.w     Q6_K", "gemv1_q6_K", 5120, 248320, 210.0),
    ] {
        let bytes = (k * n_rows) as f64 * bytes_per_256 / 256.0;
        let w = s.alloc_zeros::<u8>(bytes as usize)?;
        let x = s.alloc_zeros::<f32>(k)?;
        let mut o = s.alloc_zeros::<f32>(n_rows)?;
        let f = dev.kernels().get("quant", &quant, kernel)?;
        let (ki, ni, ti) = (k as i32, n_rows as i32, 1i32);
        let mut line = format!("  {label} ");
        for block in [32u32, 64, 128, 160, 256] {
            let t = ms(
                || {
                    let mut b = s.launch_builder(&f);
                    b.arg(&o.as_view_mut())
                        .arg(&w.as_view())
                        .arg(&x.as_view())
                        .arg(&ki)
                        .arg(&ni)
                        .arg(&ti);
                    unsafe {
                        b.launch(LaunchConfig {
                            grid_dim: (n_rows as u32, 1, 1),
                            block_dim: (block, 1, 1),
                            shared_mem_bytes: 0,
                        })?
                    };
                    s.synchronize()
                },
                5,
            )?;
            let gbs = bytes / (t / 1e3) / 1e9;
            line.push_str(&format!(" {block:>4}:{gbs:6.1}"));
        }
        println!("{line}  GB/s");
    }
    Ok(())
}
