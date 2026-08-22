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
use cudarc::driver::CudaSlice;
use tuili_cuda::Device;

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
    /// `[max_seqs, n_linear, heads, dk, dv]`
    recurrent: CudaSlice<f32>,
    /// `[max_seqs, n_linear, conv_channels, conv_k - 1]`
    conv: CudaSlice<f32>,
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
        let base = (seq.0 * self.n_linear + ordinal) * n;
        base..base + n
    }

    fn conv_span(&self, seq: SeqId, ordinal: usize) -> std::ops::Range<usize> {
        let n = self.shape.conv_floats();
        let base = (seq.0 * self.n_linear + ordinal) * n;
        base..base + n
    }

    /// A view of one sequence's recurrent state for one linear layer.
    ///
    /// The kernel takes `[n_seqs, heads, dk, dv]` and indexes sequence 0, so a
    /// batch of several sequences needs their states adjacent. They are not:
    /// the pool is sequence-major, so one layer's states are strided by
    /// `n_linear`. Batched decode therefore issues one launch per sequence for
    /// now — see `batched_is_one_launch_per_sequence` in the tests for why that
    /// is correct rather than merely convenient, and `Self::relayout` for the
    /// shape a single launch would need.
    pub fn recurrent_mut(
        &mut self,
        seq: SeqId,
        ordinal: usize,
    ) -> cudarc::driver::CudaViewMut<'_, f32> {
        let span = self.recurrent_span(seq, ordinal);
        self.recurrent.slice_mut(span)
    }

    pub fn conv_mut(&mut self, seq: SeqId, ordinal: usize) -> cudarc::driver::CudaViewMut<'_, f32> {
        let span = self.conv_span(seq, ordinal);
        self.conv.slice_mut(span)
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
