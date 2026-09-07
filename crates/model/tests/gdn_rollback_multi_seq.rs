//! `GdnRollback::stage` and `replay_layer` narrow their `memcpy_dtod`s to the
//! armed pass's own sequence slot rather than the whole `max_seqs`-wide
//! buffer (see the note on `GdnRollback::stage`). `stage` only ever borrows
//! the persistent conv window and state as `&View` — immutable — so the type
//! system alone rules out it corrupting another sequence's share of either.
//! `replay_layer`'s conv restore is the one call in this pair that writes
//! into the persistent buffer, and it is the one this test is for: with two
//! sequences sharing a layer's conv window, does rewinding the armed slot's
//! window ever touch the other slot's bytes?
//!
//! `gdn_rollback.rs` cannot answer this on its own — its fixture runs
//! `GdnRollback::new` at `max_seqs = 1`, so "the armed slot" and "the whole
//! buffer" are the same range there and a boundary bug has nowhere to show
//! up.

use anyhow::Result;
use infero_cuda::Device;
use infero_model::config::LinearAttnConfig;
use infero_model::spec::{GdnRollback, GdnTap};

fn la() -> LinearAttnConfig {
    LinearAttnConfig {
        key_heads: 2,
        value_heads: 4,
        key_head_dim: 8,
        value_head_dim: 6,
        conv_kernel: 4,
        v_heads_tiled: false,
    }
}

fn device() -> Option<Device> {
    Device::new(0).ok()
}

#[test]
fn replay_never_touches_another_sequences_conv_window() -> Result<()> {
    let Some(dev) = device() else {
        eprintln!("skipping: no cuda device");
        return Ok(());
    };
    let la = la();
    let kern = infero_kernels::Kernels::new(dev.clone());
    let width = la.conv_channels();
    let conv_floats = width * (la.conv_kernel - 1);
    let state_floats = la.value_heads * la.key_head_dim * la.value_head_dim;
    const MAX_SEQS: usize = 3;
    const ARMED_SLOT: usize = 1;

    // Two-tone fill, one value a sequence, so any bleed between slots shows
    // up as the wrong constant rather than a subtly wrong number.
    let tone = |slot: usize| 100.0 + slot as f32;
    let conv: Vec<f32> = (0..MAX_SEQS)
        .flat_map(|s| vec![tone(s); conv_floats])
        .collect();
    let state: Vec<f32> = (0..MAX_SEQS)
        .flat_map(|s| vec![tone(s); state_floats])
        .collect();

    let stream = dev.stream();
    let mut d_conv = stream.clone_htod(&conv)?;
    let mut d_state = stream.clone_htod(&state)?;

    let mut r = GdnRollback::new(&dev, la, &[true], MAX_SEQS, 1)?;
    r.arm(ARMED_SLOT, 1, 0)?;
    r.stage(&kern, 0, ARMED_SLOT, &d_conv.as_view(), &d_state.as_view())?;

    // Stand in for the forward pass having advanced the armed slot's window
    // to something else, which is exactly what a verification pass does and
    // what the replay below has to undo.
    let disturbed = 999.0f32;
    let span = ARMED_SLOT * conv_floats..(ARMED_SLOT + 1) * conv_floats;
    stream.memcpy_htod(&vec![disturbed; conv_floats], &mut d_conv.slice_mut(span))?;

    // Unused when `keep == 0`: `replay_layer` returns before the kernels
    // that would read either array.
    let placeholder = stream.clone_htod(&vec![0i32; MAX_SEQS])?;
    let seqs = infero_kernels::gdn::SeqLayout {
        first_token: &placeholder.as_view(),
        n_tokens: &placeholder.as_view(),
        n_seqs: MAX_SEQS,
        total_tokens: 0,
    };
    // A dummy conv weight — unused too, since `keep == 0` returns before the
    // convolution kernel that would read it.
    let d_conv_w = stream.clone_htod(&vec![0.0f32; width * la.conv_kernel])?;
    r.replay_layer(
        &dev,
        &kern,
        0,
        ARMED_SLOT,
        0,
        &seqs,
        &d_conv_w.as_view(),
        &mut d_state.as_view_mut(),
        &mut d_conv.as_view_mut(),
    )?;

    let got = stream.clone_dtoh(&d_conv)?;
    dev.synchronize()?;
    for s in 0..MAX_SEQS {
        let want = tone(s);
        let slice = &got[s * conv_floats..(s + 1) * conv_floats];
        assert!(
            slice.iter().all(|&v| v == want),
            "slot {s}: expected every value to be {want} (its own pre-step tap), got {:?}",
            slice
        );
    }
    Ok(())
}

/// The scenario `verify_draft_sampled_batch` actually creates: two sequences
/// armed *at once*, each recording its own rows out of one shared, flat
/// `[n_total, ...]` activation buffer (what a fused forward pass over both
/// sequences' candidates leaves behind) rather than each getting a call all
/// to itself. `record`'s `row_start`-based slicing is the only new code this
/// exercises that the single-slot tests above cannot: does slot 0's journal
/// end up holding only slot 0's rows, and slot 1's only slot 1's, or does
/// `row_start` bleed one sequence's numbers into the other's replay?
#[test]
fn two_slots_armed_at_once_record_only_their_own_rows() -> Result<()> {
    let Some(dev) = device() else {
        eprintln!("skipping: no cuda device");
        return Ok(());
    };
    let la = la();
    let kern = infero_kernels::Kernels::new(dev.clone());
    let width = la.conv_channels();
    let heads = la.value_heads;
    const MAX_SEQS: usize = 2;
    // Different candidate counts a slot, on purpose: a bug that only shows
    // up when `row_start` is not simply `slot * cap` (the wrong formula a
    // fixed-width per-slot layout would tempt) needs the two spans to be
    // different widths to expose it.
    const ROWS0: usize = 2;
    const ROWS1: usize = 3;
    let cap = ROWS0.max(ROWS1);

    let stream = dev.stream();
    let mut r = GdnRollback::new(&dev, la, &[true], MAX_SEQS, cap)?;
    // Slot 0's rows start at 0, slot 1's start right after slot 0's --
    // exactly how `verify_draft_sampled_batch` lays a fused call's items out.
    r.arm(0, ROWS0, 0)?;
    r.arm(1, ROWS1, ROWS0)?;

    // One shared flat tap, `[ROWS0 + ROWS1, ...]`, two-tone by slot so any
    // bleed between them shows up as the wrong constant.
    let tone = |slot: usize| 100.0 + slot as f32;
    let n_total = ROWS0 + ROWS1;
    let mk = |per_row: usize| -> Result<cudarc::driver::CudaSlice<f32>> {
        let mut v = vec![tone(0); ROWS0 * per_row];
        v.extend(vec![tone(1); ROWS1 * per_row]);
        Ok(stream.clone_htod(&v)?)
    };
    let pre_conv = mk(width)?;
    let post_conv = mk(width)?;
    let g = mk(heads)?;
    let beta = mk(heads)?;

    for slot in [0usize, 1] {
        r.record(
            &kern,
            0,
            slot,
            GdnTap {
                pre_conv: pre_conv.slice(..n_total * width),
                post_conv: post_conv.slice(..n_total * width),
                g: g.slice(..n_total * heads),
                beta: beta.slice(..n_total * heads),
            },
        )?;
    }
    dev.synchronize()?;

    // The direct check: read each slot's own journal region back and
    // confirm it holds exactly `rows` copies of *that slot's own* tone —
    // not the other slot's, and not a mix from a `row_start` off by the
    // wrong amount.
    for (slot, rows, want) in [(0usize, ROWS0, tone(0)), (1, ROWS1, tone(1))] {
        let got = stream.clone_dtoh(&r.debug_journal_qkv(0, slot))?;
        dev.synchronize()?;
        assert_eq!(got.len(), rows * width, "slot {slot}: wrong row count in its own journal");
        assert!(
            got.iter().all(|&v| v == want),
            "slot {slot}: expected every journalled value to be {want} (its own \
             tone), got {:?} -- record's row_start slicing pulled the wrong span",
            got
        );
    }
    Ok(())
}
