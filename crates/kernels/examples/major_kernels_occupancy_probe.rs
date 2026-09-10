//! Single-launch harnesses for `ncu`, one per real production kernel other
//! than the FFN CUTLASS GEMM (already profiled by
//! `swap_ab_occupancy_probe.rs`), at the real batch=16 decode shape. Every
//! kernel here was previously sized only by a wall-clock `INFERO_PROFILE`
//! step-time share (gdn_delta_rule ~6%, quantize_act_e4m3_cutlass ~5% of a
//! decode step) -- `ncu` itself (real occupancy/waves-per-SM data, not
//! wall-clock inference) was assumed blocked for most of this investigation
//! and only confirmed `sudo`-usable very late, so neither kernel has ever
//! actually been profiled this way before.
//!
//! One launch each, no warmup/timing loop, so `ncu`'s own output isn't
//! averaged across anything else:
//!
//!   sudo /usr/local/cuda-12.8/bin/ncu --set full -k regex:gdn_delta_rule_reg128 \
//!     ./target/release/examples/major_kernels_occupancy_probe
//!   sudo /usr/local/cuda-12.8/bin/ncu --set full -k regex:quantize_act_e4m3 \
//!     ./target/release/examples/major_kernels_occupancy_probe

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::gdn::SeqLayout;
use infero_kernels::Kernels;

fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
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

/// The 27B's own GDN shape: `linear_num_value_heads=48`,
/// `linear_num_key_heads=16`, `linear_{key,value}_head_dim=128` -- `dk=dv=128`
/// routes real production to `DeltaVariant::Reg` (`Kernels::gdn_delta_rule`'s
/// own doc comment), i.e. `gdn_delta_rule_reg128_f32`.
fn run_gdn_delta_rule(k: &Kernels) -> Result<()> {
    const HEADS: usize = 48;
    const KEY_HEADS: usize = 16;
    const DK: usize = 128;
    const DV: usize = 128;
    const N_SEQS: usize = 16; // the real batch=16 decode shape

    let stream = k.device().stream().clone();
    let key_dim = KEY_HEADS * DK;
    let val_dim = HEADS * DV;
    let stride = 2 * key_dim + val_dim;
    let offsets = (stride, 0, key_dim, 2 * key_dim);

    let total = N_SEQS; // one token a sequence, real decode shape
    let row = pseudo_random(total * stride, 0xd317);
    let g: Vec<f32> = pseudo_random(total * HEADS, 0xd318).iter().map(|v| -v.abs() * 0.6).collect();
    let beta: Vec<f32> =
        pseudo_random(total * HEADS, 0xd319).iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
    let first: Vec<i32> = (0..N_SEQS as i32).collect();
    let ntok: Vec<i32> = vec![1i32; N_SEQS];

    let d_row = stream.clone_htod(&row)?;
    let d_g = stream.clone_htod(&g)?;
    let d_beta = stream.clone_htod(&beta)?;
    let d_first = stream.clone_htod(&first)?;
    let d_ntok = stream.clone_htod(&ntok)?;
    let mut d_out = stream.alloc_zeros::<f32>(total * HEADS * DV)?;
    let mut d_state = stream.alloc_zeros::<f32>(N_SEQS * HEADS * DK * DV)?;

    let seqs =
        SeqLayout { first_token: &d_first.as_view(), n_tokens: &d_ntok.as_view(), n_seqs: N_SEQS, total_tokens: total };

    k.gdn_delta_rule(
        &mut d_out.as_view_mut(),
        &mut d_state.as_view_mut(),
        &d_row.as_view(),
        &d_g.as_view(),
        &d_beta.as_view(),
        &seqs,
        HEADS,
        KEY_HEADS,
        DK,
        DV,
        offsets,
        false,
    )?;
    k.device().synchronize()?;
    println!("gdn_delta_rule (reg128) done: {HEADS} heads x {N_SEQS} seqs, dk=dv={DK}");
    Ok(())
}

/// The activation quantizer feeding every dense FP8 projection -- shape is
/// `(k, n_tokens)` only (weight-independent), so `K=5120` (this checkpoint's
/// most common hidden size) at the real batch=16 decode `n_tokens` is a
/// representative real call.
fn run_quantize_act_e4m3(k: &Kernels) -> Result<()> {
    const KK: usize = 5120;
    const N_TOKENS: usize = 16;

    let stream = k.device().stream().clone();
    let x = pseudo_random(N_TOKENS * KK, 0xACE0);
    let d_x = stream.clone_htod(&x)?;
    let scale_cols = KK / infero_kernels::fp8::ACT_QUANT_GROUP;
    let mut d_xq = stream.alloc_zeros::<u8>(N_TOKENS * KK)?;
    let mut d_sfa_t = stream.alloc_zeros::<f32>(scale_cols * N_TOKENS)?;

    k.quantize_act_e4m3_cutlass(
        &mut d_xq.as_view_mut(),
        &mut d_sfa_t.as_view_mut(),
        &d_x.as_view(),
        KK,
        N_TOKENS,
        N_TOKENS,
    )?;
    k.device().synchronize()?;
    println!("quantize_act_e4m3_cutlass done: K={KK} n_tokens={N_TOKENS}");
    Ok(())
}

fn main() -> Result<()> {
    let k = Kernels::new(Device::new(0)?);
    run_gdn_delta_rule(&k)?;
    run_quantize_act_e4m3(&k)?;
    Ok(())
}
