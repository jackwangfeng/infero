//! Per-sequence state for the GatedDeltaNet layers.
//!
//! This is the third kind of per-sequence memory in the engine, and it does not
//! behave like the other two. A KV cache grows by one slot per token and can be
//! shortened by dropping slots. An activation buffer dies at the end of the
//! step. A GatedDeltaNet state is a fixed-size matrix that is *overwritten* on
//! every token, so:
//!
//! - It costs the same whether the sequence is 10 tokens or 200000, which is
//!   the whole point of linear attention — but it costs that from the first
//!   token, so the fixed cost per concurrent sequence is large.
//! - It cannot be rolled back. `KvPool::truncate(id, len)` returns slots and
//!   the remaining prefix is still exactly right. There is no corresponding
//!   operation here: the state at token `len` was overwritten by tokens
//!   `len+1..`, and nothing retains it. Truncating to a nonzero length has to
//!   invalidate the sequence and force a re-prefill, and `truncate` below says
//!   so rather than quietly leaving a state that belongs to a longer prefix.
//! - Its buffer address has to be stable across a CUDA graph capture and every
//!   replay, because the graph records the pointer. That is why the whole pool
//!   is allocated once at construction and indexed, rather than allocated per
//!   sequence.
//!
//! Sizing, for Qwen3.8-27B: 48 linear layers x 48 value heads x 128 x 128 x
//! f32 is 147 MiB per sequence, plus 5.6 MiB of convolution window. At 8
//! concurrent sequences that is 1.2 GiB; at 64 it is 9.8 GiB. So `max_seqs` is
//! a real memory decision for this model in a way it is not for a pure
//! attention model, and `bytes()` exists to be reported at startup.

use anyhow::Result;
use tuili_gpu::Buf;
use tuili_gpu::Device;

use crate::SeqId;

/// Which layers are linear-attention, and how big their state is.
#[derive(Debug, Clone, Copy)]
pub struct GdnShape {
    /// Value heads — the recurrence runs at this width, not the key-head width.
    pub heads: usize,
    pub dk: usize,
    pub dv: usize,
    /// `2 * key_dim + value_dim`, the convolution's channel count.
    pub conv_channels: usize,
    /// Convolution kernel width; the carried window is one shorter.
    pub conv_k: usize,
}

impl GdnShape {
    pub fn state_floats(&self) -> usize {
        self.heads * self.dk * self.dv
    }

    pub fn conv_floats(&self) -> usize {
        self.conv_channels * (self.conv_k - 1)
    }
}

/// The recurrent and convolution state for every sequence and every
/// linear-attention layer.
///
/// Indexed by `(SeqId, linear layer ordinal)` — note the ordinal, not the model
/// layer index: only 48 of the 27B's 64 layers have state, and numbering them
/// densely keeps the allocation from being 25% empty.
pub struct GdnState {
    /// `[n_linear, max_seqs, heads, dk, dv]` — **layer-major**.
    ///
    /// Layer-major rather than sequence-major so that one layer's states for
    /// every sequence are contiguous and indexed by `SeqId`. That is what lets
    /// a single launch cover the whole batch: the kernel takes
    /// `[n_seqs, heads, dk, dv]` and uses `blockIdx.y` as the slot, so passing
    /// this layer's slice and a per-slot token count is enough. Sequences not
    /// in the batch get a count of zero and their blocks exit immediately.
    ///
    /// The alternative, sequence-major, would need a slot-indirection array in
    /// the kernel — one more thing to get wrong, for no gain.
    recurrent: Buf<f32>,
    /// `[n_linear, max_seqs, conv_channels, conv_k - 1]`, same reasoning.
    conv: Buf<f32>,
    shape: GdnShape,
    n_linear: usize,
    max_seqs: usize,
    /// Model layer index -> linear ordinal, or `None` for a full-attention
    /// layer. Built once so the forward pass does not re-derive it.
    ordinal: Vec<Option<usize>>,
}

impl GdnState {
    /// `layer_is_linear` is indexed by model layer, in order.
    pub fn new(
        dev: &Device,
        layer_is_linear: &[bool],
        shape: GdnShape,
        max_seqs: usize,
    ) -> Result<Self> {
        anyhow::ensure!(shape.conv_k >= 2, "a convolution of width {} has no window", shape.conv_k);
        let mut ordinal = Vec::with_capacity(layer_is_linear.len());
        let mut n = 0;
        for &linear in layer_is_linear {
            ordinal.push(if linear {
                let o = n;
                n += 1;
                Some(o)
            } else {
                None
            });
        }
        let stream = dev.stream();
        Ok(Self {
            recurrent: stream.alloc_zeros::<f32>(max_seqs * n * shape.state_floats())?,
            conv: stream.alloc_zeros::<f32>(max_seqs * n * shape.conv_floats())?,
            shape,
            n_linear: n,
            max_seqs,
            ordinal,
        })
    }

    pub fn shape(&self) -> GdnShape {
        self.shape
    }

    pub fn n_linear_layers(&self) -> usize {
        self.n_linear
    }

    /// Total device memory held, for the startup report. Large enough on this
    /// model to be worth saying out loud.
    pub fn bytes(&self) -> usize {
        (self.recurrent.len() + self.conv.len()) * std::mem::size_of::<f32>()
    }

    /// The linear ordinal of a model layer, or `None` if it is full attention.
    pub fn ordinal_of(&self, layer: usize) -> Option<usize> {
        self.ordinal.get(layer).copied().flatten()
    }

    fn recurrent_span(&self, seq: SeqId, ordinal: usize) -> std::ops::Range<usize> {
        let n = self.shape.state_floats();
        let base = (ordinal * self.max_seqs + seq.0) * n;
        base..base + n
    }

    fn conv_span(&self, seq: SeqId, ordinal: usize) -> std::ops::Range<usize> {
        let n = self.shape.conv_floats();
        let base = (ordinal * self.max_seqs + seq.0) * n;
        base..base + n
    }

    /// One linear layer's recurrent state for *every* sequence slot, which is
    /// what a single batched launch takes.
    pub fn recurrent_layer_mut(&mut self, ordinal: usize) -> tuili_gpu::ViewMut<'_, f32> {
        let n = self.shape.state_floats() * self.max_seqs;
        self.recurrent.slice_mut(ordinal * n..(ordinal + 1) * n)
    }

    /// One linear layer's convolution windows for every sequence slot.
    pub fn conv_layer_mut(&mut self, ordinal: usize) -> tuili_gpu::ViewMut<'_, f32> {
        let n = self.shape.conv_floats() * self.max_seqs;
        self.conv.slice_mut(ordinal * n..(ordinal + 1) * n)
    }

    /// Both of one layer's state buffers at once, since a block advances both.
    pub fn layer_views(
        &mut self,
        ordinal: usize,
    ) -> (
        tuili_gpu::ViewMut<'_, f32>,
        tuili_gpu::ViewMut<'_, f32>,
    ) {
        let rn = self.shape.state_floats() * self.max_seqs;
        let cn = self.shape.conv_floats() * self.max_seqs;
        (
            self.recurrent.slice_mut(ordinal * rn..(ordinal + 1) * rn),
            self.conv.slice_mut(ordinal * cn..(ordinal + 1) * cn),
        )
    }

    /// How many sequence slots a launch covers.
    pub fn max_seqs(&self) -> usize {
        self.max_seqs
    }

    /// Zero every layer's state for one sequence.
    ///
    /// Called when a sequence starts, and when it is reset. Forgetting this on
    /// a reused slot is the failure that looks like the model remembering a
    /// previous conversation it was never shown.
    pub fn reset(&mut self, dev: &Device, seq: SeqId) -> Result<()> {
        anyhow::ensure!(seq.0 < self.max_seqs, "sequence {} is past the pool", seq.0);
        let stream = dev.stream();
        for ordinal in 0..self.n_linear {
            let span = self.recurrent_span(seq, ordinal);
            stream.memset_zeros(&mut self.recurrent.slice_mut(span))?;
            let span = self.conv_span(seq, ordinal);
            stream.memset_zeros(&mut self.conv.slice_mut(span))?;
        }
        Ok(())
    }

    /// Copy one sequence's recurrent and convolution state onto another.
    ///
    /// A tree draft's verification runs every root-to-leaf path as its own
    /// sequence, and each starts from the state the shared prefix left behind.
    /// The recurrence cannot fork — a state is advanced along a sequence, not
    /// branched — so the fork is a copy, and this is it.
    ///
    /// 3 MiB a layer on the 27B, so 151 MiB for a whole state. A four-path tree
    /// copies three of them: 906 MiB moved, about 0.6 ms, which is the price of
    /// admission for a shape the recurrence will not otherwise allow.
    pub fn fork(&mut self, dev: &Device, src: SeqId, dst: SeqId) -> Result<()> {
        anyhow::ensure!(
            src.0 < self.max_seqs && dst.0 < self.max_seqs,
            "sequences {} and {} against a pool of {}",
            src.0,
            dst.0,
            self.max_seqs
        );
        anyhow::ensure!(src.0 != dst.0, "forking sequence {} onto itself", src.0);
        // Both spans live in one buffer, so the borrow checker will not hand out
        // a read and a write at once — correctly, since it cannot know they are
        // disjoint. `split_at_mut` at the later span's start proves it: two
        // halves, the source wholly in one and the destination wholly in the
        // other, whichever way round they fall.
        let copy = |dev: &Device,
                    buf: &mut tuili_gpu::Buf<f32>,
                    from: std::ops::Range<usize>,
                    to: std::ops::Range<usize>|
         -> Result<()> {
            debug_assert_eq!(from.end - from.start, to.end - to.start);
            debug_assert!(from.end <= to.start || to.end <= from.start, "overlapping spans");
            let n = from.end - from.start;
            if from.start < to.start {
                let (lo, mut hi) = buf.split_at_mut(to.start);
                let a = lo.slice(from.start..from.start + n);
                let mut b = hi.slice_mut(..n);
                dev.stream().memcpy_dtod(&a, &mut b)?;
            } else {
                let (mut lo, hi) = buf.split_at_mut(from.start);
                let mut b = lo.slice_mut(to.start..to.start + n);
                let a = hi.slice(..n);
                dev.stream().memcpy_dtod(&a, &mut b)?;
            }
            Ok(())
        };
        for ordinal in 0..self.n_linear {
            let (a, b) = (self.recurrent_span(src, ordinal), self.recurrent_span(dst, ordinal));
            copy(dev, &mut self.recurrent, a, b)?;
            let (a, b) = (self.conv_span(src, ordinal), self.conv_span(dst, ordinal));
            copy(dev, &mut self.conv, a, b)?;
        }
        Ok(())
    }

    /// What a truncation means for a sequence with recurrent state.
    ///
    /// Returns whether the caller must re-prefill from scratch. Truncating to
    /// zero is a reset and is exact. Truncating to anything else is not
    /// expressible: the state that existed at token `len` was overwritten by
    /// the tokens after it, and nothing kept a copy. The honest answer is to
    /// zero the state and tell the caller its prefix is gone, rather than leave
    /// a state belonging to a longer sequence attached to a shorter one — which
    /// would produce fluent output conditioned on text the user had deleted.
    ///
    /// Making partial truncation work means checkpointing the state at chosen
    /// boundaries, at 147 MiB a checkpoint on this model. That is a real design
    /// with a real cost, and it is not this one.
    #[must_use = "the caller has to act on a lost prefix rather than ignore it"]
    pub fn truncate(&mut self, dev: &Device, seq: SeqId, len: usize) -> Result<bool> {
        self.reset(dev, seq)?;
        Ok(len > 0)
    }
}
