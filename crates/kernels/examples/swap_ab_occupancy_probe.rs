//! Single-launch harness for `ncu`: does the CURRENTLY SHIPPED `small_m_swap`
//! tile still show real SM-idle at the real decode shape (K=5120,N=17408,
//! n_tokens=16, FFN gate/up), or did swap_ab's own tile-shape change already
//! close the ~28% SM-idle finding from the pre-swap_ab wide tile? One launch
//! only (no warmup loop, no timing loop) so `ncu`'s own output isn't averaged
//! across anything else.
//!
//!   sudo /usr/local/cuda-12.8/bin/ncu --set full \
//!     ./target/release/examples/swap_ab_occupancy_probe

#![cfg(feature = "cutlass")]

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::fp8::{ACT_QUANT_GROUP, FP8_BLOCK};
use infero_kernels::Kernels;

fn quant_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let b = (s >> 24) as u8;
            if b == 0x7F || b == 0xFF { 0x38 } else { b }
        })
        .collect()
}

fn pseudo_random_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0) * 3.0
        })
        .collect()
}

fn packed(quants: &[u8], scales: &[f32], k: usize, n: usize) -> Vec<u8> {
    let mut v = infero_kernels::fp8::repack_rows(quants, k, n).expect("repack");
    for s in scales {
        v.extend_from_slice(&s.to_le_bytes());
    }
    v
}

fn main() -> Result<()> {
    let k = Kernels::new(Device::new(0)?);
    let (kk, n, n_tokens) = (5120usize, 17408usize, 16usize);

    let quants = quant_bytes(n * kk, 0xE4A3);
    let scale_n = n / FP8_BLOCK;
    let scale_k = kk / FP8_BLOCK;
    let scales: Vec<f32> = (0..scale_n * scale_k).map(|i| 0.3 + 0.4 * (i % 5) as f32).collect();
    let w_buf = packed(&quants, &scales, kk, n);
    let stream = k.device().stream().clone();
    let d_w = stream.clone_htod(&w_buf)?;
    let cutlass_w = k.prepare_cutlass_weight(&d_w.as_view(), kk, n, false)?;

    let x: Vec<f32> = pseudo_random_f32(n_tokens * kk, 0xACE0);
    let d_x = stream.clone_htod(&x)?;
    let scale_cols = kk / ACT_QUANT_GROUP;
    let mut d_xq = stream.alloc_zeros::<u8>(n_tokens * kk)?;
    let mut d_sfa_t = stream.alloc_zeros::<f32>(scale_cols * n_tokens)?;
    k.quantize_act_e4m3_cutlass(
        &mut d_xq.as_view_mut(),
        &mut d_sfa_t.as_view_mut(),
        &d_x.as_view(),
        kk,
        n_tokens,
        n_tokens,
    )?;
    let mut d_out = stream.alloc_zeros::<f32>(n_tokens * n)?;

    // Real dispatch (not the bench bypass): at n_tokens=16 this is exactly
    // what production calls, `small_m_swap`.
    k.mma_e4m3_cutlass_sfa_f32out(
        &mut d_out.as_view_mut(),
        &d_w.as_view(),
        &cutlass_w,
        &d_xq.as_view(),
        &d_sfa_t.as_view(),
        kk,
        n,
        n_tokens,
        false,
    )?;
    k.device().synchronize()?;
    println!("one launch done (K={kk} N={n} n_tokens={n_tokens}, real dispatch)");
    Ok(())
}
