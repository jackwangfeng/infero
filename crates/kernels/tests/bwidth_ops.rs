//! Achieved bandwidth of the small per-step kernels, at a decode step's shapes.
//!
//! `bwidth.rs` asks the same question of the GEMM. These are the kernels either
//! side of it: the ones that move a few hundred KiB and are done in single-digit
//! microseconds, where a launch's worth of latency is the whole cost. The bytes
//! counted are the ones the kernel *must* move — inputs read once, outputs
//! written once — so a number well under the device's copy ceiling means loads
//! that are not in flight, not arithmetic.
//!
//! Not an assertion, a measurement. It prints and passes; run it with
//! `--nocapture`. Correctness for everything here lives in `tests/ops.rs`, and a
//! variant that has not passed that file is not worth timing.
//!
//! # These three kernels have no bandwidth left to win
//!
//! RTX A4000, which copies about 380 GB/s against a 448 GB/s spec. Device time
//! from the same event pair the engine's profile uses, at a Llama-3.1-8B decode
//! step's shapes and 32 tokens:
//!
//! | kernel         | device | GB/s at 32t | slope   | fixed   |
//! |----------------|-------:|------------:|--------:|--------:|
//! | null, 1 thread | 3.13us |           — |       — |       — |
//! | `store_kv2`    | 3.85us |         102 | 304GB/s | 2.56us  |
//! | `rope_qk`      | 5.46us |         240 | 369GB/s | 1.91us  |
//! | `attn_softmax` | 11.57us|         363 | 370GB/s | 0.23us  |
//!
//! The middle column is the one the profile reports and it is an artifact. A
//! kernel of one thread and one store measures 3.13 us, so at a decode's shapes
//! `store_kv2` is 0.7 us of work under 3.13 us of floor and its "102 GB/s" is a
//! statement about the event pair, not about the kernel. The honest number is
//! the slope: hold the shape and grow the batch to 512 tokens, and the marginal
//! cost gives 304, 369 and 370 GB/s against a 380 GB/s ceiling.
//!
//! So all three are at or near the memory wall already, and `catching-vllm.md`'s
//! estimate that they were worth 0.4 ms of a step was reading fixed cost as idle
//! bandwidth. Widening the loads in any of them cannot pay: the whole of
//! `store_kv` + `rope_qk` + `attn_softmax` is 1.4% of a batch-32 step in the
//! engine's own profile, and most of that 1.4% is this floor.
//!
//! The floor is worth its own note. It is an event pair plus a launch, so it
//! overstates what these kernels cost in production, where the step is captured
//! into a CUDA graph. Under `INFERO_PROFILE` every small kernel in the table
//! carries about 3 us that the graphed step does not pay — which means the
//! profile's *shares* are right and its microseconds, for anything this small,
//! are not.

use anyhow::Result;
use std::time::Instant;
use infero_cuda::Device;
use infero_kernels::Kernels;

#[test]
fn small_kernels_achieved_bandwidth() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();

    // Llama-3.1-8B: 32 query heads, 8 kv heads, 128 wide.
    let (n_heads, n_kv_heads, d_head) = (32usize, 8usize, 128usize);
    let (kv_len, n_slots) = (512usize, 4096usize);
    let d = n_heads * d_head;
    let kv_dim = n_kv_heads * d_head;

    let max_tokens = 512usize;
    let mut q = stream.alloc_zeros::<f32>(max_tokens * d)?;
    let mut kk = stream.alloc_zeros::<f32>(max_tokens * kv_dim)?;
    let vv = stream.alloc_zeros::<f32>(max_tokens * kv_dim)?;
    let mut k_pool = stream.alloc_zeros::<half::f16>(n_kv_heads * n_slots * d_head)?;
    let mut v_pool = stream.alloc_zeros::<half::f16>(n_kv_heads * n_slots * d_head)?;
    let slots: Vec<i32> = (0..max_tokens as i32).map(|i| i * 3 + 1).collect();
    let dslots = stream.clone_htod(&slots)?;
    let positions: Vec<i32> = vec![kv_len as i32 - 1; max_tokens];
    let dpos = stream.clone_htod(&positions)?;
    let ff = stream.clone_htod(&vec![1.0f32; d_head / 2])?;
    let mut scores = stream.alloc_zeros::<f32>(n_heads * max_tokens * kv_len)?;

    // Host time per launch, and — under `INFERO_PROFILE` — the device time the
    // events actually attribute to the kernel. Back to back these kernels are
    // shorter than a launch costs, so the host number bottoms out around two
    // microseconds whatever the kernel does; the device number is the one that
    // moves. Both are printed because the gap between them is the point.
    let time = |name: &str, run: &mut dyn FnMut() -> Result<()>| -> Result<(f64, f64)> {
        for _ in 0..5 {
            run()?;
        }
        dev.synchronize()?;
        dev.profile().reset();
        let t0 = Instant::now();
        for _ in 0..50 {
            run()?;
        }
        dev.synchronize()?;
        let host = t0.elapsed().as_secs_f64() / 50.0;
        let device = dev
            .profile()
            .snapshot()
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, e)| e.millis / 1e3 / e.launches as f64)
            .unwrap_or(0.0);
        Ok((host, device))
    };

    // The floor. One thread, one store — whatever this costs is what an event
    // pair around a launch costs, and every number below inherits it.
    let mut table = stream.alloc_zeros::<i32>(64)?;
    let zeros = stream.clone_htod(&vec![0i32; 1])?;
    let t = time("write_slot_table", &mut || {
        kern.write_slot_table(
            &mut table.as_view_mut(),
            &zeros.as_view(),
            &zeros.as_view(),
            &zeros.as_view(),
            64,
            1,
        )
    })?;
    eprintln!(
        "  null (1 thread)     host {:>6.2} us | device {:>7.2} us",
        t.0 * 1e6,
        t.1 * 1e6
    );

    let mut slope: Vec<(&str, f64, f64)> = Vec::new();
    for tokens in [1usize, 8, 32, 512] {
        // f32 in, f16 out, both halves of the cache.
        let bytes = (tokens * 2 * kv_dim * (4 + 2)) as f64;
        let t = time("store_kv", &mut || {
            kern.store_kv2(
                &mut k_pool.as_view_mut(),
                &mut v_pool.as_view_mut(),
                &kk.slice(..tokens * kv_dim),
                &vv.slice(..tokens * kv_dim),
                &dslots.slice(..tokens),
                n_kv_heads,
                d_head,
                n_slots,
                tokens,
            )
        })?;
        report("store_kv2   ", tokens, t, bytes);
        slope.push(("store_kv2   ", t.1, bytes));

        // Q and K, read and written in place.
        let bytes = (tokens * (d + kv_dim) * 4 * 2) as f64;
        let t = time("rope_qk", &mut || {
            kern.rope_qk(
                &mut q.slice_mut(..tokens * d),
                &mut kk.slice_mut(..tokens * kv_dim),
                &dpos.slice(..tokens),
                &ff.as_view(),
                tokens,
                n_heads,
                n_kv_heads,
                d_head,
                500_000.0,
                1.0,
                true,
            )
        })?;
        report("rope_qk     ", tokens, t, bytes);
        slope.push(("rope_qk     ", t.1, bytes));

        // One read and one write of the score matrix is the floor; the kernel
        // as written passes over it three times.
        let bytes = (n_heads * tokens * kv_len * 4 * 2) as f64;
        let t = time("attn_softmax", &mut || {
            kern.attn_softmax(
                &mut scores.slice_mut(..n_heads * tokens * kv_len),
                n_heads,
                tokens,
                kv_len,
            )
        })?;
        report("attn_softmax", tokens, t, bytes);
        slope.push(("attn_softmax", t.1, bytes));
    }

    // Fixed cost against marginal cost. Every one of these kernels is under
    // four microseconds at a decode's shapes, and an event pair around a launch
    // cannot see below about that, so the per-shape GB/s above is an
    // underestimate of the kernel and an overestimate of what tuning it can
    // buy. The slope between the 32-token shape and a 512-token one is the
    // kernel's actual bandwidth; the intercept is what no amount of tuning
    // removes.
    for name in ["store_kv2   ", "rope_qk     ", "attn_softmax"] {
        let rows: Vec<_> = slope.iter().filter(|(n, ..)| *n == name).collect();
        let (_, t_small, b_small) = rows[rows.len() - 2];
        let (_, t_big, b_big) = rows[rows.len() - 1];
        if t_big > t_small {
            eprintln!(
                "  {name}       slope {:>5.0} GB/s, fixed {:>5.2} us",
                (b_big - b_small) / (t_big - t_small) / 1e9,
                (t_small - (b_small * (t_big - t_small) / (b_big - b_small))) * 1e6,
            );
        }
    }
    Ok(())
}

fn report(label: &str, tokens: usize, t: (f64, f64), bytes: f64) {
    let (host, device) = t;
    let dev_us = if device > 0.0 {
        format!("{:>7.2} us {:>5.0} GB/s", device * 1e6, bytes / device / 1e9)
    } else {
        "        (no INFERO_PROFILE)".into()
    };
    eprintln!(
        "  {label} @{tokens:>2}t  host {:>6.2} us | device {dev_us}  ({:.0} KiB)",
        host * 1e6,
        bytes / 1024.0
    );
}
