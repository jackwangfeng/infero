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
use tuili_cuda::Device;
use tuili_model::config::LinearAttnConfig;
use tuili_model::spec::GdnRollback;

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
    let kern = tuili_kernels::Kernels::new(dev.clone());
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
    r.arm(ARMED_SLOT, 1)?;
    r.stage(&kern, 0, &d_conv.as_view(), &d_state.as_view())?;

    // Stand in for the forward pass having advanced the armed slot's window
    // to something else, which is exactly what a verification pass does and
    // what the replay below has to undo.
    let disturbed = 999.0f32;
    let span = ARMED_SLOT * conv_floats..(ARMED_SLOT + 1) * conv_floats;
    stream.memcpy_htod(&vec![disturbed; conv_floats], &mut d_conv.slice_mut(span))?;

    // Unused when `keep == 0`: `replay_layer` returns before the kernels
    // that would read either array.
    let placeholder = stream.clone_htod(&vec![0i32; MAX_SEQS])?;
    let seqs = tuili_kernels::gdn::SeqLayout {
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
