//! A shared, paged key/value pool.
//!
//! Every sequence in flight draws token slots from one pool and keeps a table
//! mapping its logical positions onto physical slots. That indirection is what
//! makes continuous batching possible: sequences of wildly different lengths
//! coexist, a finished sequence returns its slots immediately, and admitting a
//! new one costs a table write rather than an allocation.
//!
//! The page size is one token. Larger pages would give the attention loop
//! better locality, but at this size there is no internal fragmentation at all
//! and the table costs four bytes per cached token — 256 KB for 16 sequences
//! of 4096, against megabytes for the tokens themselves.
//!
//! Pool layout is `[n_kv_heads, n_slots, d_head]` per layer, so one head's
//! slots stay contiguous and a gather reads a whole head vector coalesced.

use anyhow::{Context, Result};
use cudarc::driver::{CudaSlice, CudaView, CudaViewMut};
use half::f16;
use tuili_cuda::Device;
use tuili_kernels::KvQuant;

use crate::config::Config;

/// Identifies a sequence's row in the slot table. Valid until [`KvPool::free`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SeqId(pub usize);

enum Storage {
    F16 {
        keys: Vec<CudaSlice<f16>>,
        values: Vec<CudaSlice<f16>>,
    },
    TurboQuant {
        quant: KvQuant,
        k_codes: Vec<CudaSlice<u8>>,
        k_signs: Vec<CudaSlice<u8>>,
        k_scale: Vec<CudaSlice<f16>>,
        k_gamma: Vec<CudaSlice<f16>>,
        v_codes: Vec<CudaSlice<u8>>,
        v_scale: Vec<CudaSlice<f16>>,
    },
}

struct SeqState {
len: usize,
    /// Leading slots this sequence borrows from another rather than owning.
    ///
    /// A tree draft's verification runs every root-to-leaf path as its own
    /// sequence over the same prefix, and copying the prefix's keys per path
    /// would cost more than the pass. So a forked sequence points at the
    /// original's slots and must not return them: `free` and `truncate` stop
    /// here. Getting that wrong hands another request's cache to the allocator,
    /// which is why it is a field and not a convention.
    borrowed: usize,
    
    /// Physical slots this sequence holds, in logical order.
    slots: Vec<i32>,
}

static NEXT_POOL_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub struct KvPool {
    storage: Storage,
    /// `[max_seqs, max_seq]` on the device, mirrored on the host.
    slot_table: CudaSlice<i32>,
    max_seqs: usize,
    max_seq: usize,
    n_slots: usize,
    free: Vec<i32>,
    seqs: Vec<Option<SeqState>>,
    bytes: usize,
    /// Distinguishes one pool from another for anything that caches work keyed
    /// to this pool's device pointers — a captured CUDA graph holds them and
    /// must not be replayed against a different pool's memory.
    id: u64,
    /// The GatedDeltaNet state, when the model has linear-attention blocks.
    ///
    /// Here rather than beside the pool because it is per-sequence state with
    /// exactly this lifetime and exactly this `max_seqs`, and because the way to
    /// get it wrong is to reuse a slot without clearing it — the model then
    /// conditions on a conversation it was never shown. Owning it means `alloc`
    /// can clear it, rather than every caller having to remember to.
    gdn: Option<crate::gdn_state::GdnState>,
    /// Per sequence *slot*: where its tokens start in the batch and how many it
    /// contributes. Zero for a slot not in this batch. Sized by `max_seqs` so a
    /// single launch can cover the pool.
    gdn_first: CudaSlice<i32>,
    gdn_ntok: CudaSlice<i32>,
}

impl KvPool {
    /// A value unique to this pool for as long as it lives.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Reserve `n_slots` token slots shared by up to `max_seqs` concurrent
    /// sequences, each at most `max_seq` tokens long.
    pub fn new(
        dev: &Device,
        cfg: &Config,
        n_slots: usize,
        max_seqs: usize,
        max_seq: usize,
        quant: KvQuant,
        layer_is_linear: &[bool],
    ) -> Result<Self> {
        anyhow::ensure!(n_slots > 0 && max_seqs > 0 && max_seq > 0, "empty kv pool");
        let stream = dev.stream();
        let per_head = cfg.n_kv_heads * n_slots;

        let (storage, bytes) = if quant.is_quantized() {
            let k_code_bytes = cfg.d_head * quant.k_mse_bits() as usize / 8;
            let v_code_bytes = cfg.d_head * quant.v_bits() as usize / 8;
            let sign_bytes = cfg.d_head / 8;

            let mut k_codes = Vec::with_capacity(cfg.n_layers);
            let mut k_signs = Vec::with_capacity(cfg.n_layers);
            let mut k_scale = Vec::with_capacity(cfg.n_layers);
            let mut k_gamma = Vec::with_capacity(cfg.n_layers);
            let mut v_codes = Vec::with_capacity(cfg.n_layers);
            let mut v_scale = Vec::with_capacity(cfg.n_layers);
            for i in 0..cfg.n_layers {
                let ctx = |what: &str| format!("allocating {what} for layer {i}");
                k_codes.push(
                    stream
                        .alloc_zeros::<u8>(per_head * k_code_bytes)
                        .with_context(|| ctx("key codes"))?,
                );
                k_signs.push(
                    stream
                        .alloc_zeros::<u8>(per_head * sign_bytes)
                        .with_context(|| ctx("key qjl signs"))?,
                );
                k_scale.push(
                    stream
                        .alloc_zeros::<f16>(per_head)
                        .with_context(|| ctx("key scales"))?,
                );
                k_gamma.push(
                    stream
                        .alloc_zeros::<f16>(per_head)
                        .with_context(|| ctx("key gammas"))?,
                );
                v_codes.push(
                    stream
                        .alloc_zeros::<u8>(per_head * v_code_bytes)
                        .with_context(|| ctx("value codes"))?,
                );
                v_scale.push(
                    stream
                        .alloc_zeros::<f16>(per_head)
                        .with_context(|| ctx("value scales"))?,
                );
            }
            let per_layer = per_head * (k_code_bytes + sign_bytes + v_code_bytes) + per_head * 6;
            (
                Storage::TurboQuant {
                    quant,
                    k_codes,
                    k_signs,
                    k_scale,
                    k_gamma,
                    v_codes,
                    v_scale,
                },
                per_layer * cfg.n_layers,
            )
        } else {
            let per_layer = per_head * cfg.d_head;
            let mut keys = Vec::with_capacity(cfg.n_layers);
            let mut values = Vec::with_capacity(cfg.n_layers);
            for i in 0..cfg.n_layers {
                keys.push(
                    stream
                        .alloc_zeros::<f16>(per_layer)
                        .with_context(|| format!("allocating key pool for layer {i}"))?,
                );
                values.push(
                    stream
                        .alloc_zeros::<f16>(per_layer)
                        .with_context(|| format!("allocating value pool for layer {i}"))?,
                );
            }
            (
                Storage::F16 { keys, values },
                per_layer * cfg.n_layers * 2 * std::mem::size_of::<f16>(),
            )
        };

        let slot_table = stream.alloc_zeros::<i32>(max_seqs * max_seq)?;
        // Highest slot first, so the first sequence gets slot 0 upward and a
        // dump of the table reads naturally.
        let free: Vec<i32> = (0..n_slots as i32).rev().collect();

        tracing::info!(
            quant = %quant,
            mib = (bytes + max_seqs * max_seq * 4) / (1 << 20),
            n_slots,
            max_seqs,
            max_seq,
            "kv pool allocated"
        );

        Ok(Self {
            storage,
            slot_table,
            max_seqs,
            max_seq,
            n_slots,
            free,
            seqs: (0..max_seqs).map(|_| None).collect(),
            bytes: bytes + max_seqs * max_seq * 4,
            id: NEXT_POOL_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            gdn: match cfg.linear_attn {
                Some(la) => {
                    anyhow::ensure!(
                        layer_is_linear.len() == cfg.n_layers,
                        "the layer-kind list has {} entries for {} layers",
                        layer_is_linear.len(),
                        cfg.n_layers
                    );
                    let st = crate::gdn_state::GdnState::new(
                        dev,
                        layer_is_linear,
                        crate::gdn_state::GdnShape {
                            heads: la.value_heads,
                            dk: la.key_head_dim,
                            dv: la.value_head_dim,
                            conv_channels: la.conv_channels(),
                            conv_k: la.conv_kernel,
                        },
                        max_seqs,
                    )?;
                    tracing::info!(
                        linear_layers = st.n_linear_layers(),
                        mib = st.bytes() >> 20,
                        per_seq_mib = (st.bytes() / max_seqs) >> 20,
                        "recurrent state allocated"
                    );
                    Some(st)
                }
                None => None,
            },
            gdn_first: stream.alloc_zeros::<i32>(max_seqs)?,
            gdn_ntok: stream.alloc_zeros::<i32>(max_seqs)?,
        })
    }

    pub fn quant(&self) -> KvQuant {
        match &self.storage {
            Storage::F16 { .. } => KvQuant::F16,
            Storage::TurboQuant { quant, .. } => *quant,
        }
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn n_slots(&self) -> usize {
        self.n_slots
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    pub fn max_seqs(&self) -> usize {
        self.max_seqs
    }

    /// Token slots not currently held by any sequence.
    pub fn free_slots(&self) -> usize {
        self.free.len()
    }

    pub fn active_sequences(&self) -> usize {
        self.seqs.iter().filter(|s| s.is_some()).count()
    }

    pub(crate) fn table_stride(&self) -> usize {
        self.max_seq
    }

    pub(crate) fn slot_table(&self) -> &CudaSlice<i32> {
        &self.slot_table
    }

    pub(crate) fn slot_table_mut(&mut self) -> &mut CudaSlice<i32> {
        &mut self.slot_table
    }

    /// Read a sequence's slot table back off the device.
    ///
    /// Not on any hot path; it exists so a test can assert that two sequences
    /// really were handed disjoint slots.
    pub fn read_slot_table(&self, dev: &Device, id: SeqId) -> Result<Vec<i32>> {
        let base = id.0 * self.max_seq;
        let len = self.len(id);
        Ok(dev
            .stream()
            .clone_dtoh(&self.slot_table.slice(base..base + len.max(1)))?)
    }

    /// Claim a sequence row. Returns `None` when every row is in use.
    pub fn alloc(&mut self) -> Option<SeqId> {
        let idx = self.seqs.iter().position(|s| s.is_none())?;
        self.seqs[idx] = Some(SeqState {
            len: 0,
            slots: Vec::new(),
            borrowed: 0,
        });
        Some(SeqId(idx))
    }

    /// Release a sequence and return its slots to the pool.
    pub fn free(&mut self, id: SeqId) {
        if let Some(state) = self.seqs.get_mut(id.0).and_then(Option::take) {
            // Only what this sequence owns. A forked prefix belongs to the
            // sequence it was forked from, and returning it here would hand the
            // allocator slots another sequence is still reading.
            self.free
                .extend(state.slots.into_iter().skip(state.borrowed).rev());
        }
    }

    /// Point `dst` at `src`'s tokens, sharing the slots rather than copying them.
    ///
    /// A tree draft's verification runs every root-to-leaf path as its own
    /// sequence over the same prefix. Copying the prefix's keys per path would
    /// cost more than the pass it is for — the whole point of the tree is that
    /// the pass is nearly free at these widths — so `dst` borrows them and may
    /// not return them. [`Self::free`] and [`Self::truncate`] both stop at the
    /// fork point.
    ///
    /// The device table has to be copied even though the slots are not: it is
    /// persistent and written a token at a time, so `dst`'s prefix entries have
    /// never been filled in. That is `len` ints, under a kilobyte at these
    /// lengths.
    ///
    /// `dst` must be empty. Forking onto a sequence with tokens of its own would
    /// leave two owners for one slot, which is the failure this method's whole
    /// shape exists to prevent.
    pub fn fork(&mut self, dev: &Device, src: SeqId, dst: SeqId) -> Result<()> {
        anyhow::ensure!(src.0 != dst.0, "forking sequence {} onto itself", src.0);
        anyhow::ensure!(
            self.len(dst) == 0,
            "sequence {} holds {} tokens; a fork wants an empty destination",
            dst.0,
            self.len(dst)
        );
        let (slots, len) = {
            let s = self.seqs[src.0]
                .as_ref()
                .with_context(|| format!("sequence {} is not allocated", src.0))?;
            (s.slots.clone(), s.len)
        };
        {
            let d = self.seqs[dst.0]
                .as_mut()
                .with_context(|| format!("sequence {} is not allocated", dst.0))?;
            d.slots = slots;
            d.len = len;
            d.borrowed = len;
        }
        // The persistent table's prefix, `src`'s row into `dst`'s.
        let stride = self.max_seq;
        let (a, b) = (src.0 * stride, dst.0 * stride);
        let n = len;
        if n > 0 {
            let table = &mut self.slot_table;
            if a < b {
                let (lo, mut hi) = table.split_at_mut(b);
                let from = lo.slice(a..a + n);
                let mut to = hi.slice_mut(..n);
                dev.stream().memcpy_dtod(&from, &mut to)?;
            } else {
                let (mut lo, hi) = table.split_at_mut(a);
                let mut to = lo.slice_mut(b..b + n);
                let from = hi.slice(..n);
                dev.stream().memcpy_dtod(&from, &mut to)?;
            }
        }
        // The recurrence cannot fork, so its state is copied outright.
        if let Some(g) = self.gdn.as_mut() {
            g.fork(dev, src, dst)?;
        }
        Ok(())
    }

    /// Move `keep` of `src`'s own tokens onto `dst`, which forked from it.
    ///
    /// How a tree's winning path becomes the sequence's future. The path forked
    /// `dst`'s prefix and then wrote its own tokens past it; accepting the path
    /// means those tokens are now the sequence's, so their slots change owner
    /// rather than being copied.
    ///
    /// `src` must be a fork of `dst` — its borrowed prefix has to be exactly
    /// `dst`'s current tokens, or the tokens would land after a gap. Whatever
    /// `src` owned past `keep` goes back to the pool, since a rejected suffix is
    /// nobody's.
    ///
    /// The device table needs no copy: `dst`'s new positions are `src`'s old
    /// ones, and the entries there already name the slots being transferred.
    pub fn adopt(&mut self, dst: SeqId, src: SeqId, keep: usize) -> Result<()> {
        anyhow::ensure!(src.0 != dst.0, "sequence {} adopting from itself", src.0);
        let (owned, borrowed) = {
            let s = self.seqs[src.0]
                .as_ref()
                .with_context(|| format!("sequence {} is not allocated", src.0))?;
            (s.slots[s.borrowed..].to_vec(), s.borrowed)
        };
        anyhow::ensure!(
            borrowed == self.len(dst),
            "sequence {} forked at {borrowed} but {} now holds {}; adopting \
             would leave a gap",
            src.0,
            dst.0,
            self.len(dst)
        );
        anyhow::ensure!(
            keep <= owned.len(),
            "adopting {keep} tokens from a fork that wrote {}",
            owned.len()
        );
        {
            let d = self.seqs[dst.0]
                .as_mut()
                .with_context(|| format!("sequence {} is not allocated", dst.0))?;
            d.slots.extend_from_slice(&owned[..keep]);
            d.len += keep;
        }
        {
            let s = self.seqs[src.0].as_mut().unwrap();
            s.slots.truncate(s.borrowed);
            s.len = s.borrowed;
        }
        // The suffix nobody accepted.
        self.free.extend(owned[keep..].iter().rev().copied());
        Ok(())
    }

    /// Give up a forked prefix, so the sequence owns nothing and can be freed or
    /// truncated like any other.
    ///
    /// Separate from `free` because a tree's paths are forked and dropped every
    /// round, and a caller that forgets this leaks nothing — it is the *opposite*
    /// mistake, freeing a borrowed slot, that this design makes impossible.
    pub fn drop_fork(&mut self, id: SeqId) {
        if let Some(s) = self.seqs[id.0].as_mut() {
            // What it owns goes back; what it borrowed is simply forgotten. The
            // first version of this cleared the whole vector, which dropped the
            // owned slots on the floor — a leak that only shows as a pool
            // slowly running out under load.
            let at = s.borrowed.min(s.slots.len());
            let own: Vec<i32> = s.slots.split_off(at);
            s.slots.clear();
            s.len = 0;
            s.borrowed = 0;
            self.free.extend(own.into_iter().rev());
        }
    }

    pub fn len(&self, id: SeqId) -> usize {
        self.seqs[id.0].as_ref().map_or(0, |s| s.len)
    }

    pub fn is_empty(&self, id: SeqId) -> bool {
        self.len(id) == 0
    }

    /// Room left before this sequence hits the per-sequence limit.
    pub fn headroom(&self, id: SeqId) -> usize {
        self.max_seq - self.len(id)
    }

    /// Reserve `n` more slots for `id` and return them in logical order.
    ///
    /// Host-side only. The device table is written by a single scatter kernel
    /// once the whole batch is laid out — a per-sequence copy here would be
    /// pageable, and a pageable host-to-device copy drains the stream.
    pub(crate) fn extend(&mut self, id: SeqId, n: usize) -> Result<Vec<i32>> {
        let state = self.seqs[id.0]
            .as_ref()
            .context("extending a sequence that was never allocated")?;
        let start = state.len;
        anyhow::ensure!(
            start + n <= self.max_seq,
            "sequence {} would reach {} tokens, the limit is {}",
            id.0,
            start + n,
            self.max_seq
        );
        anyhow::ensure!(
            self.free.len() >= n,
            "kv pool exhausted: {} slots free, {n} needed",
            self.free.len()
        );

        let mut taken = Vec::with_capacity(n);
        for _ in 0..n {
            taken.push(self.free.pop().expect("checked above"));
        }

        let state = self.seqs[id.0].as_mut().unwrap();
        state.slots.extend_from_slice(&taken);
        state.len += n;
        Ok(taken)
    }

    /// The GatedDeltaNet state, when this model has any.
    pub(crate) fn gdn(&mut self) -> Option<&mut crate::gdn_state::GdnState> {
        self.gdn.as_mut()
    }

    /// Whether this pool carries recurrent state at all.
    pub fn has_recurrent_state(&self) -> bool {
        self.gdn.is_some()
    }

    /// Zero one sequence's recurrent and convolution state.
    ///
    /// Called for a sequence at length zero: a fresh sequence, or one just
    /// reset. The invariant is "no tokens seen means no state", which is why
    /// this needs no dirty flag — `alloc` cannot do it because it has no device
    /// to memset with, and a flag would be one more thing to forget.
    pub fn reset_recurrent(&mut self, dev: &Device, id: SeqId) -> Result<()> {
        if let Some(g) = self.gdn.as_mut() {
            g.reset(dev, id)?;
        }
        Ok(())
    }

    /// The batch layout the recurrence kernels index by sequence slot, together
    /// with one layer's state.
    ///
    /// Returned as one call because the layout is read while the state is
    /// written, and they are separate fields — asking for them separately would
    /// be a mutable borrow overlapping an immutable one for no reason.
    pub(crate) fn gdn_parts(
        &mut self,
        ordinal: usize,
    ) -> (
        CudaView<'_, i32>,
        CudaView<'_, i32>,
        CudaViewMut<'_, f32>,
        CudaViewMut<'_, f32>,
    ) {
        let g = self.gdn.as_mut().expect("no recurrent state in this pool");
        let (recurrent, conv) = g.layer_views(ordinal);
        (
            self.gdn_first.as_view(),
            self.gdn_ntok.as_view(),
            recurrent,
            conv,
        )
    }

    /// Fill the per-slot batch layout. `spans[slot]` is `(first token, count)`.
    pub(crate) fn set_gdn_layout(&mut self, dev: &Device, spans: &[(i32, i32)]) -> Result<()> {
        anyhow::ensure!(spans.len() == self.max_seqs, "layout covers the wrong slots");
        let first: Vec<i32> = spans.iter().map(|s| s.0).collect();
        let ntok: Vec<i32> = spans.iter().map(|s| s.1).collect();
        let stream = dev.stream();
        stream.memcpy_htod(&first, &mut self.gdn_first)?;
        stream.memcpy_htod(&ntok, &mut self.gdn_ntok)?;
        Ok(())
    }

    /// Drop the tail of a sequence back to `len` tokens, returning its slots.
    pub fn truncate(&mut self, id: SeqId, len: usize) {
        if let Some(state) = self.seqs[id.0].as_mut() {
            while state.len > len && state.len > state.borrowed {
                if let Some(slot) = state.slots.pop() {
                    self.free.push(slot);
                }
                state.len -= 1;
            }
            // Below the fork point is not expressible, and the loop above has
            // already stopped there. Letting `len` fall under `borrowed` would
            // leave the sequence claiming to be shorter than the slots it holds,
            // and the next reservation would append past the borrowed prefix and
            // leave a hole in the middle. So the length clamps, the same stance
            // `GdnState::truncate` takes for a partial rollback: say what
            // happened rather than fake it.
            debug_assert!(
                len >= state.borrowed || state.borrowed == 0,
                "truncating forked sequence {} to {len}, below its borrowed \
                 prefix of {}; drop the fork first",
                id.0,
                state.borrowed
            );
        }
    }

    // ---- dense accessors ------------------------------------------------

    pub(crate) fn dense(&self, layer: usize) -> (&CudaSlice<f16>, &CudaSlice<f16>) {
        match &self.storage {
            Storage::F16 { keys, values } => (&keys[layer], &values[layer]),
            Storage::TurboQuant { .. } => unreachable!("dense accessor on a quantized pool"),
        }
    }

    pub(crate) fn dense_mut(&mut self, layer: usize) -> (&mut CudaSlice<f16>, &mut CudaSlice<f16>) {
        match &mut self.storage {
            Storage::F16 { keys, values } => (&mut keys[layer], &mut values[layer]),
            Storage::TurboQuant { .. } => unreachable!("dense accessor on a quantized pool"),
        }
    }

    // ---- quantized accessors --------------------------------------------

    #[allow(clippy::type_complexity)]
    pub(crate) fn tq_key(
        &self,
        layer: usize,
    ) -> (
        &CudaSlice<u8>,
        &CudaSlice<u8>,
        &CudaSlice<f16>,
        &CudaSlice<f16>,
    ) {
        match &self.storage {
            Storage::TurboQuant {
                k_codes,
                k_signs,
                k_scale,
                k_gamma,
                ..
            } => (
                &k_codes[layer],
                &k_signs[layer],
                &k_scale[layer],
                &k_gamma[layer],
            ),
            Storage::F16 { .. } => unreachable!("quantized accessor on a dense pool"),
        }
    }

    pub(crate) fn tq_value(&self, layer: usize) -> (&CudaSlice<u8>, &CudaSlice<f16>) {
        match &self.storage {
            Storage::TurboQuant {
                v_codes, v_scale, ..
            } => (&v_codes[layer], &v_scale[layer]),
            Storage::F16 { .. } => unreachable!("quantized accessor on a dense pool"),
        }
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn tq_key_mut(
        &mut self,
        layer: usize,
    ) -> (
        &mut CudaSlice<u8>,
        &mut CudaSlice<u8>,
        &mut CudaSlice<f16>,
        &mut CudaSlice<f16>,
    ) {
        match &mut self.storage {
            Storage::TurboQuant {
                k_codes,
                k_signs,
                k_scale,
                k_gamma,
                ..
            } => (
                &mut k_codes[layer],
                &mut k_signs[layer],
                &mut k_scale[layer],
                &mut k_gamma[layer],
            ),
            Storage::F16 { .. } => unreachable!("quantized accessor on a dense pool"),
        }
    }

    pub(crate) fn tq_value_mut(
        &mut self,
        layer: usize,
    ) -> (&mut CudaSlice<u8>, &mut CudaSlice<f16>) {
        match &mut self.storage {
            Storage::TurboQuant {
                v_codes, v_scale, ..
            } => (&mut v_codes[layer], &mut v_scale[layer]),
            Storage::F16 { .. } => unreachable!("quantized accessor on a dense pool"),
        }
    }
}
