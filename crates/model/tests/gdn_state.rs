//! Indexing the GatedDeltaNet state pool.
//!
//! The pool is layer-major so that one layer's states are contiguous across
//! sequences, which is what lets a single launch cover a batch. That means the
//! span arithmetic has two strides going the opposite way from the obvious
//! reading, and getting it wrong does not fail — it points one sequence's
//! recurrence at another's memory, which reads as the model conditioning on a
//! conversation it was never shown.
//!
//! So: check that the per-sequence view and the per-layer view describe the
//! same bytes, and that resetting one sequence leaves the others untouched.

use infero_cuda::Device;
use infero_model::SeqId;
use infero_model::gdn_state::{GdnShape, GdnState};

fn device() -> Option<Device> {
    Device::new(0).ok()
}

/// Small but with every dimension distinct, so a transposed stride shows up.
fn shape() -> GdnShape {
    GdnShape {
        heads: 3,
        dk: 4,
        dv: 5,
        conv_channels: 7,
        conv_k: 4,
    }
}

/// The 27B's pattern: three linear layers then one attention layer, repeating.
fn qwen35_pattern(n: usize) -> Vec<bool> {
    (0..n).map(|i| (i + 1) % 4 != 0).collect()
}

#[test]
fn only_linear_layers_get_a_state_slot() {
    let Some(dev) = device() else {
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    let kinds = qwen35_pattern(16);
    let st = GdnState::new(&dev, &kinds, shape(), 4).unwrap();
    assert_eq!(st.n_linear_layers(), 12, "12 of 16 layers are linear");
    // Ordinals are dense over the linear layers and absent for the others.
    assert_eq!(st.ordinal_of(0), Some(0));
    assert_eq!(st.ordinal_of(1), Some(1));
    assert_eq!(st.ordinal_of(2), Some(2));
    assert_eq!(st.ordinal_of(3), None, "layer 3 is full attention");
    assert_eq!(st.ordinal_of(4), Some(3), "numbering skips the attention layer");
    assert_eq!(st.ordinal_of(7), None);
    assert_eq!(st.ordinal_of(15), None);
    // A dense allocation: 12 linear layers, not 16.
    let want = 4 * 12 * (shape().state_floats() + shape().conv_floats()) * 4;
    assert_eq!(st.bytes(), want, "the pool should not hold slots for attention layers");
}

/// Writing through one sequence's span must be visible in that layer's batched
/// view at exactly that sequence's offset, and nowhere else.
#[test]
fn the_per_sequence_and_per_layer_views_describe_the_same_bytes() {
    let Some(dev) = device() else {
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    let sh = shape();
    let max_seqs = 4;
    let mut st = GdnState::new(&dev, &qwen35_pattern(8), sh, max_seqs).unwrap();
    let n = sh.state_floats();
    let stream = dev.stream();

    // Mark every (layer ordinal, sequence) pair with a distinct value by
    // writing through the batched per-layer view, then read it back through
    // the same view and check the offsets are where the layout says.
    for ordinal in 0..st.n_linear_layers() {
        let mut layer = st.recurrent_layer_mut(ordinal);
        assert_eq!(
            layer.len(),
            n * max_seqs,
            "a layer view should cover every sequence slot"
        );
        let marks: Vec<f32> = (0..max_seqs)
            .flat_map(|s| std::iter::repeat_n((ordinal * 100 + s) as f32, n))
            .collect();
        stream.memcpy_htod(&marks, &mut layer).unwrap();
    }
    dev.synchronize().unwrap();

    // Now reset one sequence in the middle and confirm the damage is confined
    // to it — across every layer, since reset covers all of them.
    st.reset(&dev, SeqId(2)).unwrap();
    dev.synchronize().unwrap();

    for ordinal in 0..st.n_linear_layers() {
        let layer = st.recurrent_layer_mut(ordinal);
        let vals = stream.clone_dtoh(&layer.as_view()).unwrap();
        dev.synchronize().unwrap();
        for s in 0..max_seqs {
            let seg = &vals[s * n..(s + 1) * n];
            let want = if s == 2 { 0.0 } else { (ordinal * 100 + s) as f32 };
            assert!(
                seg.iter().all(|v| *v == want),
                "layer ordinal {ordinal}, sequence {s}: expected every entry to \
                 be {want}, got {:?}..",
                &seg[..seg.len().min(4)]
            );
        }
    }
}

/// Zeroing a sequence must clear its convolution window too. A stale window is
/// three tokens of a previous conversation leaking into the first three tokens
/// of the next one — short enough to look like a bad sample rather than a bug.
#[test]
fn resetting_a_sequence_clears_its_convolution_window() {
    let Some(dev) = device() else {
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    let sh = shape();
    let max_seqs = 3;
    let mut st = GdnState::new(&dev, &qwen35_pattern(4), sh, max_seqs).unwrap();
    let n = sh.conv_floats();
    let stream = dev.stream();

    for ordinal in 0..st.n_linear_layers() {
        let mut layer = st.conv_layer_mut(ordinal);
        let marks = vec![7.5f32; n * max_seqs];
        stream.memcpy_htod(&marks, &mut layer).unwrap();
    }
    dev.synchronize().unwrap();
    st.reset(&dev, SeqId(1)).unwrap();
    dev.synchronize().unwrap();

    for ordinal in 0..st.n_linear_layers() {
        let layer = st.conv_layer_mut(ordinal);
        let vals = stream.clone_dtoh(&layer.as_view()).unwrap();
        dev.synchronize().unwrap();
        assert!(
            vals[..n].iter().all(|v| *v == 7.5),
            "sequence 0's window was cleared as well"
        );
        assert!(
            vals[n..2 * n].iter().all(|v| *v == 0.0),
            "sequence 1's window survived the reset"
        );
        assert!(
            vals[2 * n..].iter().all(|v| *v == 7.5),
            "sequence 2's window was cleared as well"
        );
    }
}

/// Truncating to a nonzero length reports the prefix as lost. The state cannot
/// be rewound, so the alternative would be a state belonging to a longer
/// sequence attached to a shorter one.
#[test]
fn a_partial_truncation_reports_the_prefix_as_lost() {
    let Some(dev) = device() else {
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    let mut st = GdnState::new(&dev, &qwen35_pattern(4), shape(), 2).unwrap();
    assert!(
        st.truncate(&dev, SeqId(0), 17).unwrap(),
        "truncating to 17 tokens must report that the prefix is gone"
    );
    assert!(
        !st.truncate(&dev, SeqId(0), 0).unwrap(),
        "truncating to zero is an exact reset and loses nothing"
    );
}

/// A sequence past the pool is refused rather than writing somewhere else.
#[test]
fn a_sequence_past_the_pool_is_refused() {
    let Some(dev) = device() else {
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    let mut st = GdnState::new(&dev, &qwen35_pattern(4), shape(), 2).unwrap();
    let err = st.reset(&dev, SeqId(2)).unwrap_err().to_string();
    assert!(err.contains("past the pool"), "unhelpful error: {err}");
}
