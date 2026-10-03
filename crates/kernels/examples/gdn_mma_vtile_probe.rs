//! Does `gdn_chunk_state_mma_vtile_f32` agree with the already-shipped
//! `gdn_chunk_state_mma_f32` (both f16-tensor-core based, so this isolates
//! "did the v-tile retiling introduce a NEW bug" from "does f16 tensor-core
//! precision alone already diverge from the f32 host reference by more than
//! the existing suite's tolerance" -- a separate, pre-existing question this
//! probe does not attempt to answer).
//!
//!     cargo run --release -p infero-kernels --example gdn_mma_vtile_probe

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::Kernels;
use infero_kernels::gdn::{GdnChunkStateVariant, SeqLayout};

const HEADS: usize = 48;
const KEY_HEADS: usize = 16;
const DK: usize = 128;
const DV: usize = 128;

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

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut worst = 0.0f32;
    let mut at = 0;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    (worst, at)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let dev = Device::new(0)?;
    let k = Kernels::new(dev.clone());
    let stream = dev.stream().clone();

    for total in [70usize, 500, 2000] {
        let key_dim = KEY_HEADS * DK;
        let val_dim = HEADS * DV;
        let stride = 2 * key_dim + val_dim;
        let offsets = (stride, 0, key_dim, 2 * key_dim);

        let row = pseudo_random(total * stride, 0xa317);
        let g: Vec<f32> =
            pseudo_random(total * HEADS, 0xa318).iter().map(|v| -v.abs() * 0.6).collect();
        let beta: Vec<f32> =
            pseudo_random(total * HEADS, 0xa319).iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
        let first = [0i32];
        let ntok = [total as i32];

        let d_row = stream.clone_htod(&row)?;
        let d_g = stream.clone_htod(&g)?;
        let d_beta = stream.clone_htod(&beta)?;
        let d_first = stream.clone_htod(&first)?;
        let d_ntok = stream.clone_htod(&ntok)?;

        let seqs = SeqLayout {
            first_token: &d_first.as_view(),
            n_tokens: &d_ntok.as_view(),
            n_seqs: 1,
            total_tokens: total,
        };

        let mut out_plain = stream.alloc_zeros::<f32>(total * HEADS * DV)?;
        let mut state_plain = stream.alloc_zeros::<f32>(HEADS * DK * DV)?;
        k.gdn_chunk_split3_delta_rule(
            &mut out_plain.as_view_mut(),
            &mut state_plain.as_view_mut(),
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
            GdnChunkStateVariant::Plain,
        )?;
        dev.synchronize()?;
        let got_plain = stream.clone_dtoh(&out_plain)?;
        let plain_peak = got_plain.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let plain_inf = got_plain.iter().filter(|x| !x.is_finite()).count();
        println!("total={total:>6}: plain peak={plain_peak:.3e}, non-finite count={plain_inf}");

        let mut out_mma = stream.alloc_zeros::<f32>(total * HEADS * DV)?;
        let mut state_mma = stream.alloc_zeros::<f32>(HEADS * DK * DV)?;
        k.gdn_chunk_split3_delta_rule(
            &mut out_mma.as_view_mut(),
            &mut state_mma.as_view_mut(),
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
            GdnChunkStateVariant::Mma,
        )?;

        let mut out_vt = stream.alloc_zeros::<f32>(total * HEADS * DV)?;
        let mut state_vt = stream.alloc_zeros::<f32>(HEADS * DK * DV)?;
        k.gdn_chunk_split3_delta_rule(
            &mut out_vt.as_view_mut(),
            &mut state_vt.as_view_mut(),
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
            GdnChunkStateVariant::MmaVtile,
        )?;
        dev.synchronize()?;

        let got_mma = stream.clone_dtoh(&out_mma)?;
        let got_vt = stream.clone_dtoh(&out_vt)?;
        let st_mma = stream.clone_dtoh(&state_mma)?;
        let st_vt = stream.clone_dtoh(&state_vt)?;

        let (out_worst, out_at) = max_abs_diff(&got_mma, &got_vt);
        let (st_worst, st_at) = max_abs_diff(&st_mma, &st_vt);
        let peak = got_mma.iter().filter(|x| x.is_finite()).fold(0.0f32, |m, x| m.max(x.abs()));
        let mma_inf = got_mma.iter().filter(|x| !x.is_finite()).count();
        println!(
            "total={total:>6}: out worst-diff(mma vs mma_vtile)={out_worst:.3e} at {out_at} (finite peak {peak:.3e}, non-finite count in mma={mma_inf}/{}), \
             state worst-diff={st_worst:.3e} at {st_at}",
            got_mma.len()
        );
    }
    Ok(())
}
