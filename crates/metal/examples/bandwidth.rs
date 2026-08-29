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
use infero_metal::{Device, LaunchConfig};

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
    // ---- 3a. every shape the 27B actually launches, one token against two --
    //
    // The two-against-one ratio is what a speculative round lives on: a
    // verification pass is two rows, so if two rows cost twice one row there is
    // nothing to win however good the drafter is. In-situ profiling says the
    // engine's mat-vecs cost 1.79x at two rows while the two big Q4_K shapes
    // measure 1.35x on their own -- so the average is being set by shapes that
    // were never measured. This is all of them.
    println!("\n  every 27B shape, 1 token vs 2 (host's own kernel and group width)");
    println!("  {:<22} {:>8} {:>8} {:>8}  {:>6}", "tensor", "1 tok", "2 tok", "MiB", "ratio");
    for (label, ty, k, n_rows) in [
        ("ffn_down        Q4_K", "q4_K", 17408usize, 5120usize),
        ("ffn_gate/up     Q4_K", "q4_K", 5120, 17408),
        ("attn_qkv        Q8_0", "q8_0", 5120, 10240),
        ("attn_gate       Q8_0", "q8_0", 5120, 6144),
        ("ssm_out         Q8_0", "q8_0", 6144, 5120),
        ("attn_q          Q8_0", "q8_0", 5120, 12288),
        ("attn_k/v        Q8_0", "q8_0", 5120, 1024),
        ("ssm_alpha/beta  Q8_0", "q8_0", 5120, 48),
        ("output          Q6_K", "q6_K", 5120, 248320),
        ("attn_output     Q6_K", "q6_K", 6144, 5120),
    ] {
        let bpb = match ty {
            "q4_K" => 144.0f64,
            "q6_K" => 210.0,
            _ => 34.0 * 8.0,
        };
        let items = match ty {
            "q4_K" => k / 32,
            "q6_K" => k / 4,
            _ => k / 8,
        } as u32;
        let block = items.next_multiple_of(32).clamp(32, 128);
        let bytes = (k * n_rows) as f64 * bpb / 256.0;
        let w = s.alloc_zeros::<u8>(bytes as usize)?;
        let x = s.alloc_zeros::<f32>(k * 2)?;
        let mut o = s.alloc_zeros::<f32>(n_rows * 2)?;
        let ki = k as i32;
        let ni = n_rows as i32;
        let mut t1 = 0.0f64;
        let mut t2 = 0.0f64;
        for tokens in [1usize, 2] {
            // Exactly what the host picks: multi-row for Q4_K at two or more.
            let rows: u32 = if tokens >= 2 && ty == "q4_K" { 4 } else { 1 };
            let name = if rows > 1 {
                format!("gemv{tokens}x{rows}_{ty}")
            } else if tokens == 1 {
                format!("gemv1_{ty}")
            } else {
                format!("gemv{tokens}_{ty}")
            };
            let f = dev.kernels().get("quant", &quant, &name)?;
            let ti = tokens as i32;
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
                            grid_dim: ((n_rows as u32).div_ceil(rows), 1, 1),
                            block_dim: (block, 1, 1),
                            shared_mem_bytes: 0,
                        })?
                    };
                    s.synchronize()
                },
                5,
            )?;
            if tokens == 1 { t1 = t } else { t2 = t }
        }
        println!(
            "  {label:<22} {t1:7.3}ms {t2:7.3}ms {:>8.0}  {:>5.2}x",
            bytes / (1u64 << 20) as f64,
            t2 / t1
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

    // ---- 3c. the matrix-unit mat-vec against the scalar one ----------------
    //
    // The number that decides whether speculation can pay: a two-row pass wants
    // to cost what a one-row pass costs. The scalar kernel spends a FMA per
    // (element, token) and cannot; the MMA kernel fills an 8x8 tile whether
    // there are two tokens in it or eight.
    println!("\n  scalar vs matrix unit (Q4_K), against the 1-token scalar baseline");
    for (label, k, n_rows) in [
        ("ffn_gate/up", 5120usize, 17408usize),
        ("ffn_down   ", 17408, 5120),
    ] {
        let bytes = (k * n_rows) as f64 * 144.0 / 256.0;
        let w = s.alloc_zeros::<u8>(bytes as usize)?;
        let x = s.alloc_zeros::<f32>(k * 8)?;
        let mut o = s.alloc_zeros::<f32>(n_rows * 8)?;
        let ki = k as i32;
        let ni = n_rows as i32;
        let items = (k / 32) as u32;
        let scalar_block = items.next_multiple_of(32).clamp(32, 128);

        let base = {
            let f = dev.kernels().get("quant", &quant, "gemv1_q4_K")?;
            let ti = 1i32;
            ms(
                || {
                    let mut b = s.launch_builder(&f);
                    b.arg(&o.as_view_mut()).arg(&w.as_view()).arg(&x.as_view())
                        .arg(&ki).arg(&ni).arg(&ti);
                    unsafe {
                        b.launch(LaunchConfig {
                            grid_dim: (n_rows as u32, 1, 1),
                            block_dim: (scalar_block, 1, 1),
                            shared_mem_bytes: 0,
                        })?
                    };
                    s.synchronize()
                },
                5,
            )?
        };
        print!("  {label} 1tok scalar {base:6.3}ms  |");
        for tokens in [2usize, 4, 8] {
            for mma in [false, true] {
                let (name, gx, gy, block) = if mma {
                    (
                        "gemv_mma_q4_K".to_string(),
                        (n_rows as u32).div_ceil(8),
                        (tokens as u32).div_ceil(8).max(1),
                        32u32,
                    )
                } else {
                    let per = if tokens == 2 { 2 } else { 4 };
                    (
                        format!("gemv{per}x4_q4_K"),
                        (n_rows as u32).div_ceil(4),
                        (tokens as u32).div_ceil(per as u32).max(1),
                        scalar_block,
                    )
                };
                let f = dev.kernels().get("quant", &quant, &name)?;
                let ti = tokens as i32;
                let t = ms(
                    || {
                        let mut b = s.launch_builder(&f);
                        b.arg(&o.as_view_mut()).arg(&w.as_view()).arg(&x.as_view())
                            .arg(&ki).arg(&ni).arg(&ti);
                        unsafe {
                            b.launch(LaunchConfig {
                                grid_dim: (gx, gy, 1),
                                block_dim: (block, 1, 1),
                                shared_mem_bytes: 0,
                            })?
                        };
                        s.synchronize()
                    },
                    5,
                )?;
                print!("  {tokens}tok {}{:6.3}ms({:.2}x)", if mma { "mma" } else { "scl" }, t, t / base);
            }
        }
        println!();
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

    // ---- 5. staging the activation vector in threadgroup memory -----------
    //
    // `output.weight` is the worst-efficiency tensor in the sweep above (39%
    // against 47-58% for everything else), and `GEMV_SPREAD4`'s own doc says
    // why for Q4_K/Q8_0: the per-token activation load and multiply-add is the
    // majority of the kernel, not the weight stream. Q6_K cannot use
    // `GEMV_SPREAD4` -- its four elements a thread are 32 apart, not 4 -- so it
    // pays that cost as four separate scalar `device` loads instead of one
    // vectorised one. `x` is only 5120 floats (20 KiB) at this shape, comfortably
    // under the 32 KiB threadgroup memory limit, and every thread in the group
    // reads the same 5120 floats, which reads like free money: stage it once,
    // cooperatively, with vectorised float4 loads, and 128 threads' worth of
    // repeated `device` reads become one `threadgroup` read each.
    //
    // Measured on an M4 Max: 1.85x *slower* (6.9ms baseline vs 12.8ms staged),
    // answers identical. The activation was already cache-resident -- every one
    // of 248320 threadgroups reads the same 20 KiB, which does not survive one
    // pass through this GPU's cache hierarchy to fall out of it -- so the
    // `device` reads GEMV_BODY_Q6_K already does were hitting L2, not DRAM.
    // What staging actually pays for is the cooperative copy itself (1280
    // `threadgroup` writes and a barrier, serialised across the group, before
    // any real work starts) and a second barrier before the reduction can
    // trust it. Q6_K's four elements a thread are read exactly once each
    // either way; there was no redundant traffic to remove, only new
    // synchronisation to add. Left in as the record of why nobody should try
    // this again without a different hypothesis.
    println!("\n  output.weight Q6_K: activation staged in threadgroup memory");
    {
        const STAGED: &str = r#"
kernel void gemv1_q6_K_staged(
        device float* out          [[buffer(0)]],
        device const void* w       [[buffer(1)]],
        device const float* x      [[buffer(2)]],
        constant int& k            [[buffer(3)]],
        constant int& n            [[buffer(4)]],
        constant int& n_tokens     [[buffer(5)]],
        threadgroup float* xs      [[threadgroup(0)]],
        uint3 tgid  [[threadgroup_position_in_grid]],
        uint3 tid   [[thread_position_in_threadgroup]],
        uint3 tgdim [[threads_per_threadgroup]]) {
    BLOCK_REDUCE_SCRATCH
    const int row = int(tgid.x);
    if (row >= n) return;

    const int k4 = k / 4;
    device const packed_float4* xp4 = (device const packed_float4*)x;
    threadgroup float4* xs4 = (threadgroup float4*)xs;
    for (int i = int(tid.x); i < k4; i += int(tgdim.x)) xs4[i] = float4(xp4[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    const int nb = k / QK_K;
    device const block_q6_K* wr = (device const block_q6_K*)w + size_t(row) * nb;
    const int chunks = nb * 64;
    for (int c = int(tid.x); c < chunks; c += int(tgdim.x)) {
        const int b = c / 64;
        const int rem = c % 64;
        const int n2 = rem / 32;
        const int l = rem % 32;
        device const block_q6_K* blk = wr + b;
        device const uchar* ql = blk->ql + n2 * 64;
        device const uchar* qh = blk->qh + n2 * 32;
        device const char* sc = blk->scales + n2 * 8;
        const int base = b * QK_K + n2 * 128;
        const float d = float(blk->d);
        const uchar h = qh[l];
        const int is = l / 16;
        const int q0 = int((ql[l] & 0xF) | (((h >> 0) & 3) << 4)) - 32;
        const int q1 = int((ql[l + 32] & 0xF) | (((h >> 2) & 3) << 4)) - 32;
        const int q2 = int((ql[l] >> 4) | (((h >> 4) & 3) << 4)) - 32;
        const int q3 = int((ql[l + 32] >> 4) | (((h >> 6) & 3) << 4)) - 32;
        acc += d * float(sc[is + 0]) * float(q0) * xs[base + l];
        acc += d * float(sc[is + 2]) * float(q1) * xs[base + l + 32];
        acc += d * float(sc[is + 4]) * float(q2) * xs[base + l + 64];
        acc += d * float(sc[is + 6]) * float(q3) * xs[base + l + 96];
    }
    const float total = BLOCK_SUM(acc, tid.x, tgdim.x);
    if (tid.x == 0) out[row] = total;
}
"#;
        let k = 5120usize;
        let n_rows = 248320usize;
        let bytes = (k * n_rows) as f64 * 210.0 / 256.0;
        let w = s.alloc_zeros::<u8>(bytes as usize)?;
        let x = s.alloc_zeros::<f32>(k)?;
        let mut o_base = s.alloc_zeros::<f32>(n_rows)?;
        let mut o_staged = s.alloc_zeros::<f32>(n_rows)?;
        let (ki, ni, ti) = (k as i32, n_rows as i32, 1i32);

        let base_f = dev.kernels().get("quant", &quant, "gemv1_q6_K")?;
        let t_base = ms(
            || {
                let mut b = s.launch_builder(&base_f);
                b.arg(&o_base.as_view_mut()).arg(&w.as_view()).arg(&x.as_view())
                    .arg(&ki).arg(&ni).arg(&ti);
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: (n_rows as u32, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })?
                };
                s.synchronize()
            },
            5,
        )?;
        let gbs_base = bytes / (t_base / 1e3) / 1e9;

        let staged_src = format!("{COMMON}\n{QUANT}\n{STAGED}");
        let staged_f = dev.kernels().get("quant_staged", &staged_src, "gemv1_q6_K_staged")?;
        let smem = (k * 4) as u32;
        let t_staged = ms(
            || {
                let mut b = s.launch_builder(&staged_f);
                b.arg(&o_staged.as_view_mut()).arg(&w.as_view()).arg(&x.as_view())
                    .arg(&ki).arg(&ni).arg(&ti);
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: (n_rows as u32, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: smem,
                    })?
                };
                s.synchronize()
            },
            5,
        )?;
        let gbs_staged = bytes / (t_staged / 1e3) / 1e9;

        let got_base = s.clone_dtoh(&o_base)?;
        let got_staged = s.clone_dtoh(&o_staged)?;
        let max_diff = got_base
            .iter()
            .zip(&got_staged)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!(
            "  baseline {t_base:6.3}ms {gbs_base:6.1} GB/s   staged {t_staged:6.3}ms {gbs_staged:6.1} GB/s   speedup {:.2}x   max|diff| {max_diff:.3e}",
            t_base / t_staged
        );
    }
    Ok(())
}
