//! Device-side NVFP4 (e2m1) dequant, checked against the verified host
//! reference (`infero_kernels::fp4::dequant_f4e2m1_row`, Task 2 of the NVFP4
//! plan) rather than against a second independent reading of the spec. If
//! this test ever fails, the fix is in the kernel (`cu/fp4.cu`), not here.

mod common;

use anyhow::Result;
use common::*;
use infero_kernels::fp4::{F4E2M1_BLOCK, dequant_f4e2m1_row};

/// Deterministic pseudo-random bytes, same xorshift stream this crate's other
/// GPU tests already use for quantized-byte data (see e.g.
/// `examples/swap_ab_vs_small_m_bench.rs`'s own `quant_bytes`) -- reused here
/// rather than reinvented so a failing test's data is reproducible the same
/// way every other kernel test's is.
fn rand_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        })
        .collect()
}

/// A real e4m3 byte, avoiding the two NaN patterns (0x7F/0xFF) -- same dodge
/// `quant_bytes` uses in `fp8_matvec.rs` for the same reason: a NaN scale
/// would make the host and device answers disagree for a reason that has
/// nothing to do with this kernel.
fn rand_scale_bytes(n: usize, seed: u64) -> Vec<u8> {
    rand_bytes(n, seed)
        .into_iter()
        .map(|b| if b == 0x7F || b == 0xFF { 0x38 } else { b })
        .collect()
}

#[test]
fn device_dequant_matches_host_reference() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    // 128 rows x 256 cols: a real multiple of both the 16-element block and
    // CUTLASS's own alignment requirements, wide enough that a wrong
    // block-scale index (row or column) would land on a visibly wrong
    // magnitude rather than a coincidentally-close one.
    let (n, kk) = (128usize, 256usize);
    let bytes_per_row = kk / 2;
    let blocks_per_row = kk / F4E2M1_BLOCK;

    let packed = rand_bytes(n * bytes_per_row, 0xF4E2);
    let scale_bytes = rand_scale_bytes(n * blocks_per_row, 0xB10C);
    let scale2 = 0.7f32;

    // Host reference, row by row -- the ground truth this kernel is checked
    // against.
    let want: Vec<f32> = (0..n)
        .flat_map(|row| {
            let prow = &packed[row * bytes_per_row..(row + 1) * bytes_per_row];
            let srow_bytes = &scale_bytes[row * blocks_per_row..(row + 1) * blocks_per_row];
            let srow_f32: Vec<f32> = srow_bytes
                .iter()
                .map(|&b| infero_safetensors::e4m3_value(b))
                .collect();
            dequant_f4e2m1_row(prow, &srow_f32, scale2, kk)
        })
        .collect();
    assert_eq!(want.len(), n * kk);

    let d_w = stream.clone_htod(&packed)?;
    let d_scale = stream.clone_htod(&scale_bytes)?;
    let mut d_out = stream.alloc_zeros::<f32>(n * kk)?;

    k.dequant_f4e2m1(
        &mut d_out.as_view_mut(),
        &d_w.as_view(),
        &d_scale.as_view(),
        scale2,
        kk,
        n,
    )?;
    k.device().synchronize()?;

    let got = stream.clone_dtoh(&d_out)?;
    let worst = max_rel_diff(&got, &want);
    assert!(
        worst <= 1e-5,
        "device dequant diverged from the host reference by {worst:.3e}"
    );
    Ok(())
}
