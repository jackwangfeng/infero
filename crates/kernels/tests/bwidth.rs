//! What does the width of a weight load buy, at this kernel's access pattern?
//!
//! # Against Marlin itself
//!
//! Every variant in this file has been measured against tuili's other variants,
//! which cannot say whether the kernel is near its ceiling or far from it. So
//! vLLM's `marlin_gemm` was timed directly, on the same Blackwell RTX PRO 6000,
//! at these shapes and 32 tokens, with the flags the deployed server uses
//! (`should_use_atomic_add_reduce` returns false unless
//! `VLLM_MARLIN_USE_ATOMIC_ADD` is set, and it is not). Microseconds:
//!
//! | shape                    | tuili `mmqy1w8s2` | Marlin |
//! |--------------------------|-------------------|--------|
//! | `attn_k`   4096 x 1024   |  **8.5**          |  27.6  |
//! | `attn_q`   4096 x 4096   | **14.7**          |  27.5  |
//! | qkv fused  4096 x 6144   | **16.7**          |  27.7  |
//! | `ffn_gate` 4096 x 14336  | **27.0**          |  28.1  |
//! | `ffn_down` 14336 x 4096  | **27.0**          |  28.2  |
//! | gate_up    4096 x 28672  |   45.6            | **33.5** |
//!
//! Marlin does not go below about 27.5 us whatever the width, so at everything
//! narrower than `ffn_gate` it is overhead-bound and this kernel is two to
//! three times quicker. It wins only at the widest shape, which is one vLLM
//! manufactures by fusing `gate`/`up` at load time.
//!
//! Two things follow. Porting more of Marlin is not the way forward — it would
//! lose on five of six shapes. And fusing as vLLM does would put a layer's
//! GEMMs at 104.0 us against vLLM's own 116.9, so the remaining gap to vLLM at
//! a batch of 32 is *not* in the matrix multiply: with the faster GEMM it is
//! still about 1.23x, and what is left is attention and the elementwise
//! kernels.
//!
//! # What grows with the batch
//!
//! `mmq` costs 74.85 ms at a batch of 8 and 99.23 at 32 over the same twenty
//! steps — 33% more for weights that are byte-for-byte identical. Two
//! explanations were tested and both are wrong. The tensor cores are still
//! free: `mmqnm_*`, which is `mmqf_*` with the MMAs deleted, matches it to the
//! digit at both widths on Blackwell (20.8 against 20.8 us at 8 tokens, 29.2
//! against 29.4 at 32), so the old finding survives the kernel getting twice as
//! fast. And it is not the token-tile count: forcing `TUILI_MMQ_TILES=1` at a
//! batch of 32 measured 3128 tok/s against the default's 3635.
//!
//! What does grow is the activation traffic. Every block re-reads the
//! activations for its slice of `k`, so the total is `row_tiles * k_tiles *
//! tokens * 512` bytes — for `ffn_gate` that is 14.7 MiB at 8 tokens against
//! 31 MiB of weights, and 58.7 MiB at 32. The obvious lever is a wider row
//! tile, which would cut the re-reads by the same factor, and it does not pay:
//! at 32 tokens `mmqy4w8s2` is worse than `mmqy1w8s2` on five of six shapes
//! and `mmqy2w8s2` on four. Only `gate_up`, the widest, gains — 45.4 us to
//! 41.7. Whatever a wider tile saves in activation reads, it gives back in
//! occupancy.
//!
//! Which blocks are resident together was tried too, and does not help. The
//! striped partition flattens `(row group, k chunk)` k-major, so concurrent
//! blocks sit in one row group at different `k`: they share the weights and
//! each reads its own activation slice, which is the wrong way round when the
//! activations are the larger half. Permuting the slice indices so concurrent
//! blocks span row groups at the same `k` moves the one-block-wide shapes by
//! less than the noise and makes the two-wide ones worse — `gate_up` 41.6 us to
//! 44.2. L2 does not catch the sharing. See the comment on `flat` in
//! `MMQ_Y_BODY`.
//!
//! So the batch-scaling of this kernel is understood but not fixed, and it is
//! half of what separates a 32-token step from vLLM's; the other half is
//! attention.
//!
//! The GEMM reads weights four bytes at a time because a lane's eight bytes are
//! not contiguous in the AWQ pack. Marlin reads sixteen. Closing that means a
//! repack, which touches the loader, the mat-vec, the float path and every test
//! that pins them — so price it first. Both patterns below read the same bytes
//! with the same grid; only the instruction width differs.

use anyhow::Result;
use std::time::Instant;
use tuili_cuda::Device;
use tuili_kernels::Kernels;

#[test]
fn wide_weight_loads_are_worth_measuring_before_repacking() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();

    // Every Q4_G128 shape a Llama-3.1-8B decode step touches. `ffn_gate` is
    // the one the tuning was done on; the others are here because a shape
    // sweep on one matrix is how a kernel gets overfitted. `attn_k` is a
    // fourteenth of `ffn_gate`'s width, which is where a device-sized
    // partition has the least to work with.
    let shapes = [
        ("attn_k    ", 4096usize, 1024usize),
        ("attn_q    ", 4096, 4096),
        ("qkv_fused ", 4096, 6144),
        ("ffn_gate  ", 4096, 14336),
        ("gate_up_f ", 4096, 28672),
        ("ffn_down  ", 14336, 4096),
    ];
    let mut sink = stream.alloc_zeros::<f32>(1)?;
    let blocks = dev.sm_count() * 8;

    for (label, k, n) in shapes {
        let nb = k / 128;
        let w = stream.alloc_zeros::<u8>(n * nb * 68)?;
        let x = stream.alloc_zeros::<u8>(64 * Kernels::q8_1_bytes(k))?;
        let x16 = stream.alloc_zeros::<half::f16>(64 * k)?;
        let mut out = stream.alloc_zeros::<f32>(64 * n)?;
        let weight_bytes = (n * nb * 68) as f64;
        let payload = (n * nb * 64) as f64;

        for (wide, tag) in [(false, "probe  4B"), (true, "probe 16B")] {
            for _ in 0..3 {
                kern.mmq_bw_probe(wide, &mut sink.as_view_mut(), &w.as_view(), nb, n, blocks)?;
            }
            dev.synchronize()?;
            let t0 = Instant::now();
            for _ in 0..20 {
                kern.mmq_bw_probe(wide, &mut sink.as_view_mut(), &w.as_view(), nb, n, blocks)?;
            }
            dev.synchronize()?;
            let secs = t0.elapsed().as_secs_f64() / 20.0;
            eprintln!("  {label} {tag:12}          {:.0} GB/s", payload / secs / 1e9);
        }

        for tokens in [8usize, 32] {
            // `TUILI_MMQ_VARIANTS=a,b` swaps the list, which is how a new
            // shape gets compared against the default without an edit.
            let names = std::env::var("TUILI_MMQ_VARIANTS")
                .unwrap_or_else(|_| "mmqy1w8s2,mmqy2w8s2".into());
            for variant in names.split(',') {
                let f16 = variant.starts_with("mmqf") || variant.starts_with("mmqz") || variant.starts_with("mmqy") || variant.starts_with("mmqk") || variant.starts_with("mmqc") || variant.starts_with("mmqn")
                    || variant.starts_with("mmqk") || variant.starts_with("mmqn")
                    || variant.starts_with("mmqnm")
                    || variant.starts_with("mmqnh")
                    || variant.starts_with("mmqnr");
                let mut run = || -> Result<()> {
                    if f16 {
                        kern.mmq_f16(
                            variant,
                            &mut out.slice_mut(..tokens * n),
                            &w.as_view(),
                            &x16.slice(..tokens * k),
                            k,
                            n,
                            tokens,
                        )
                    } else {
                        kern.mmq_variant(
                            variant,
                            &mut out.slice_mut(..tokens * n),
                            &w.as_view(),
                            tuili_kernels::WeightType::Q4G128,
                            &x.slice(..tokens * Kernels::q8_1_bytes(k)),
                            k,
                            n,
                            tokens,
                        )
                    }
                };
                for _ in 0..3 {
                    run()?;
                }
                dev.synchronize()?;
                let t0 = Instant::now();
                for _ in 0..20 {
                    run()?;
                }
                dev.synchronize()?;
                let secs = t0.elapsed().as_secs_f64() / 20.0;
                eprintln!(
                    "  {label} {variant:11} @{tokens:>2}t  {:>7.1} us  {:>5.0} GB/s",
                    secs * 1e6,
                    weight_bytes / secs / 1e9
                );
            }
        }
    }
    Ok(())
}

/// The vocab projection's shape, in the two encodings that could carry it.
///
/// It is 532 MiB of Q8_0 read in 717 us inside the engine — 780 GB/s where this
/// card gives 1440 — and the suspicion was the layout: a Q8_0 block is 34 bytes,
/// so `mmq_load_w_q8_0` reads its quants two bytes at a time, which is the exact
/// defect that cost `attn_output` 2.4x. `Q4_G128T` at the same shape runs the
/// other pipeline over 16-byte aligned reads.
///
/// **It is not the layout.** On a Blackwell RTX PRO 6000 the two are within 6%
/// of each other per byte — 789 GB/s staged against 837 direct — and both are
/// far under the 1440 GB/s this card reads at. The A4000 says the opposite, 99
/// against 315, which is how this nearly became a rewrite: a three-fold ratio
/// on the card that is not the target.
///
/// What is left to explain the 789 is the activation re-read. Every one of the
/// 2004 row-blocks reads the whole 147 KB of Q8_1 activations, which is 295 MB
/// against 532 of weights; counted in, the kernel moves 827 MB at 1168 GB/s and
/// most of the gap closes. Halving it means 128 rows a block, and the tile for
/// that is 53.8 KB of *static* shared against a 48 KB limit — so it needs
/// `extern __shared__` before it needs anything else.
#[test]
fn the_vocab_projection_in_both_encodings() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();

    let (k, n) = (4096usize, 128256usize);
    let tokens = 32usize;
    let q8_bytes = n * (k / 32) * 34;
    let g128_bytes = n * (k / 128) * 68;
    let w8 = stream.alloc_zeros::<u8>(q8_bytes)?;
    let wg = stream.alloc_zeros::<u8>(g128_bytes)?;
    let ws = stream.alloc_zeros::<u8>(q8_bytes)?;
    let x = stream.alloc_zeros::<u8>(tokens * Kernels::q8_1_bytes(k))?;
    let x16 = stream.alloc_zeros::<half::f16>(tokens * k)?;
    let mut out = stream.alloc_zeros::<f32>(tokens * n)?;

    let time = |run: &mut dyn FnMut() -> Result<()>, bytes: usize| -> Result<()> {
        for _ in 0..3 {
            run()?;
        }
        dev.synchronize()?;
        let t0 = Instant::now();
        for _ in 0..20 {
            run()?;
        }
        dev.synchronize()?;
        let secs = t0.elapsed().as_secs_f64() / 20.0;
        eprintln!(
            "  {:>7.1} us  {:>5.0} GB/s  ({} MiB)",
            secs * 1e6,
            bytes as f64 / secs / 1e9,
            bytes >> 20
        );
        Ok(())
    };

    eprint!("  q8_0    staged   ");
    time(
        &mut || {
            kern.mmq_variant(
                "mmq",
                &mut out.as_view_mut(),
                &w8.as_view(),
                tuili_kernels::WeightType::Q8_0,
                &x.as_view(),
                k,
                n,
                tokens,
            )
        },
        q8_bytes,
    )?;
    // Every shape the integer family instantiates, because the default one
    // (4 warps, 1 tile) was picked on a layer matrix and this row count is
    // thirty times a layer's. The f16 path had the same surprise: sweeping
    // warps turned a 7% loss into a win.
    for v in [
        "mmq", "mmq2", "mmqw1", "mmqw1_2", "mmqw2", "mmqw2_2", "mmqw8", "mmqw8_2",
    ] {
        eprint!("  q8_0s   {v:<9}");
        // Not every shape is launchable at every n; say so and move on.
        let r = time(
            &mut || {
                kern.mmq_variant(
                    v,
                    &mut out.as_view_mut(),
                    &ws.as_view(),
                    tuili_kernels::WeightType::Q8_0S,
                    &x.as_view(),
                    k,
                    n,
                    tokens,
                )
            },
            q8_bytes,
        );
        if let Err(e) = r {
            eprintln!("  unsupported at this shape ({e})");
        }
    }
    eprint!("  q4_g128t direct  ");
    time(
        &mut || {
            kern.mmq_f16(
                "mmqy1w8s2",
                &mut out.as_view_mut(),
                &wg.as_view(),
                &x16.as_view(),
                k,
                n,
                tokens,
            )
        },
        g128_bytes,
    )?;
    Ok(())
}

/// What the eight-transactions-a-warp weight read costs, against the same bytes
/// coalesced.
///
/// Four buffers, cycled, so the working set is 248 MB against this card's 128 MB
/// of L2 — the same trick `bwidth_attn.rs` needs, and for the same reason: one
/// buffer read twenty times measures L2 and says 3479 GB/s on a 1792 GB/s card.
#[test]
fn the_weight_read_against_the_same_bytes_coalesced() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();
    // `gate_up`, the widest shape a layer has.
    let (k, n) = (4096usize, 28672usize);
    let nb = k / 128;
    let bytes = n * nb * 68;
    // Four buffers is 236 MB against 128 MB of L2 — which leaves about half of
    // a launch's bytes possibly resident from four launches ago, and that would
    // make this probe warmer than the kernel it is the ceiling for. The kernel
    // reads 4.24 GB a step and nothing of it is ever resident.
    // `TUILI_MMQ_PROBE_POOLS` says how many.
    let pools: usize = std::env::var("TUILI_MMQ_PROBE_POOLS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let pools: Vec<_> = (0..pools)
        .map(|_| stream.alloc_zeros::<u8>(bytes))
        .collect::<Result<_, _>>()?;
    let mut sink = stream.alloc_zeros::<f32>(1 << 20)?;
    // The probe runs at full occupancy — eight blocks of eight warps an SM — and
    // the GEMM runs at 2.9 blocks, because its 34 KB activation ring says so.
    // `TUILI_MMQ_PROBE_BLOCKS` matches the probe to the kernel, which is the
    // only remaining candidate for the 23% between them.
    let blocks = std::env::var("TUILI_MMQ_PROBE_BLOCKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(dev.sm_count() * 8);
    let payload = (n * nb * 64) as f64;

    eprintln!(
        "\n  gate_up 4096x28672, {} MiB a buffer, four of them, {blocks} blocks",
        bytes >> 20
    );
    for _ in 0..2 {
        for _ in 0..4 {
            kern.mmq_bw_probe(true, &mut sink.as_view_mut(), &pools[0].as_view(), nb, n, blocks)?;
        }
        dev.synchronize()?;
        let t0 = Instant::now();
        for i in 0..20 {
            kern.mmq_bw_probe(
                true,
                &mut sink.as_view_mut(),
                &pools[i % pools.len()].as_view(),
                nb,
                n,
                blocks,
            )?;
        }
        dev.synchronize()?;
        let secs = t0.elapsed().as_secs_f64() / 20.0;
        eprintln!(
            "  {:>28}  {:>7.1} us  {:>5.0} GB/s",
            std::env::var("TUILI_MMQ_PROBE").unwrap_or_else(|_| "as the kernel reads".into()),
            secs * 1e6,
            payload / secs / 1e9
        );
    }
    Ok(())
}

/// Each of a layer's four projections at its own shape, in weight bytes a
/// second.
///
/// The traced engine says the three narrow ones cost 52.5 us a layer together
/// against `gate_up`'s 54.6 — for half the weight bytes. Per row that is 3.66 ns
/// against 1.90, so either the narrow shapes are 1.9x less efficient or the
/// trace was measuring something else. This asks the kernel directly, at the
/// shapes and the batch width the engine actually runs.
///
/// Four buffers cycled, so no launch reads a resident matrix. `TUILI_MMQ_BPS`
/// and `TUILI_MMQ_VARIANT` are read by the kernel launcher, so a sweep is
/// `for b in 1 2 4 8; do TUILI_MMQ_BPS=$b cargo test ...`.
#[test]
fn each_projection_at_its_own_shape() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();
    // `TUILI_MMQ_TOKENS=16` puts the f16 family on one token tile, which halves
    // the activation ring and is the only way the deep weight rings fit at all.
    let tokens: usize = std::env::var("TUILI_MMQ_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let k = 4096usize;
    let x16 = stream.alloc_zeros::<half::f16>(tokens * k)?;
    eprintln!(
        "\n  llama-3.1-8b layer projections, {tokens} tokens, {} SMs",
        dev.sm_count()
    );
    for (name, n) in [
        ("qkv     ", 6144usize),
        ("o       ", 4096),
        ("gate_up ", 28672),
        ("down    ", 4096),
    ] {
        let nb = k / 128;
        let bytes = n * nb * 68;
        let pools: Vec<_> = (0..4)
            .map(|_| stream.alloc_zeros::<u8>(bytes))
            .collect::<Result<Vec<_>, _>>()?;
        let mut out = stream.alloc_zeros::<f32>(tokens * n)?;
        // The shape rule picks `mmqy2w8s2` only above 20k rows. A wider row
        // group is the one knob that cuts the activation re-reads — they scale
        // with the row-group count and the weights do not — so ask every shape
        // at every row-group width the family instantiates.
        let chosen = Kernels::mmq_f16_variant_for_shape(tuili_kernels::WeightType::Q4G128T, n)
            .unwrap_or("mmqy1w8s2");
        // `mmqc*` is the weight ring — weights staged through shared by
        // `cp.async` instead of read into registers — which is the one mechanism
        // the elimination table leaves open for the GEMM's missing 20%: bytes in
        // flight without register cost. It was unreachable from the model until
        // the `mmqc` prefix fix, so it has never been measured on a real shape.
        for v in [
            "mmqy1w8s2", "mmqt1w8s2", "mmqt1w8s3",
        ] {
        let mut run = |i: usize| -> Result<()> {
            kern.mmq_f16(
                v,
                &mut out.as_view_mut(),
                &pools[i % 4].as_view(),
                &x16.as_view(),
                k,
                n,
                tokens,
            )
        };
        let mut refused = None;
        for i in 0..4 {
            if let Err(e) = run(i) {
                refused = Some(e);
                break;
            }
        }
        if let Some(e) = refused {
            // A deep ring asks for more shared memory than a block may have;
            // that is a result, not a broken test.
            eprintln!("  {name} n={n:<6} {v:<11}  will not launch here ({e})");
            continue;
        }
        dev.synchronize()?;
        let t0 = Instant::now();
        for i in 0..20 {
            run(i)?;
        }
        dev.synchronize()?;
        let secs = t0.elapsed().as_secs_f64() / 20.0;
        // What the k-tiling re-reads: every (row group, k chunk) unit stages the
        // token tile's activations for its chunk, so the activation traffic
        // scales with the row-group count and the weights do not.
        let nblk: usize = v[4..5].parse().unwrap_or(1);
        let warps: usize = v[6..].split('s').next().unwrap_or("8").parse().unwrap_or(8);
        let rows_per_block = nblk * warps * 8;
        let units = n.div_ceil(rows_per_block) * k.div_ceil(256);
        let act = units * tokens * 256 * 2;
        eprintln!(
            "  {name} n={n:<6} {v:<11}{} {:>7.1} us  {:>5.0} GB/s of weights  {:>5.0} total ({} MiB of activations, {:.1}x the weights)",
            if v == chosen { "*" } else { " " },
            secs * 1e6,
            bytes as f64 / secs / 1e9,
            (bytes + act) as f64 / secs / 1e9,
            act >> 20,
            act as f64 / bytes as f64
        );
        }
    }
    Ok(())
}
