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
use cudarc::driver::CudaSlice;
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
        });
        Some(SeqId(idx))
    }

    /// Release a sequence and return its slots to the pool.
    pub fn free(&mut self, id: SeqId) {
        if let Some(state) = self.seqs.get_mut(id.0).and_then(Option::take) {
            self.free.extend(state.slots.into_iter().rev());
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

    /// Drop the tail of a sequence back to `len` tokens, returning its slots.
    pub fn truncate(&mut self, id: SeqId, len: usize) {
        if let Some(state) = self.seqs[id.0].as_mut() {
            while state.len > len {
                if let Some(slot) = state.slots.pop() {
                    self.free.push(slot);
                }
                state.len -= 1;
            }
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
