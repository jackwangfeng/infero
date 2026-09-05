//! Tensor-parallel server integration: rank 0 runs the real `Scheduler`
//! (HTTP, admission, prefix cache, sampling) unchanged and broadcasts what it
//! decided each step; every other rank runs [`run_follower`], a stripped
//! loop with no HTTP and no scheduling state beyond a KV pool and a cache of
//! admitted prompts, driven entirely by those broadcasts. See
//! `docs/superpowers/specs/2026-09-05-tensor-parallel-design.md`'s
//! "Server/scheduler: rank 0 drives, others follow" section.
//!
//! Explicitly out of scope this pass (rank 0 refuses rather than silently
//! mishandling): vision/video requests and M-RoPE (`Scheduler::step` rejects
//! any such request admitted while a real TP group is active -- see its own
//! TP branch) and speculative decoding (`engine::Engine::start` never
//! enables it when TP is active). Both are real, separate follow-up scope,
//! not attempted here.
//!
//! Two broadcasts a step, not one, because of a real ordering constraint:
//! which sequences *finish* this step depends on this step's own sampled
//! tokens (EOG, stop strings, budget), which are only known after the
//! forward pass rank 0 must first tell every other rank how to run --
//! there is no single point before the forward pass where both "what to
//! feed" and "who retires afterward" are simultaneously known.

use std::collections::HashMap;

use anyhow::{Context, Result};
use infero_model::{BatchItem, KvPool, Model, SeqId};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum WorkMsg {
    Prefill { from: usize, len: usize, wants_logits: bool },
    Decode { token: u32 },
}

/// What rank 0 tells every other rank before this step's forward pass.
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct StepPlan {
    /// Sequences admitted since the last step, with their full prompt --
    /// text-only (no vision/mrope; rank 0's own admission path already
    /// refuses those under TP, see `scheduler.rs`).
    pub admitted: Vec<(usize, Vec<u32>)>,
    /// This step's work, keyed by sequence id (not `Running`'s own index,
    /// which a follower has no equivalent of).
    pub plan: Vec<(usize, WorkMsg)>,
}

impl StepPlan {
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("StepPlan always serializes")
    }
    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        serde_json::from_slice(b).context("decoding a broadcast StepPlan")
    }
}

/// What rank 0 tells every other rank after this step's sampling/advance
/// decided who is done -- sequence ids to free.
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct StepRetired(pub Vec<usize>);

impl StepRetired {
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("StepRetired always serializes")
    }
    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        serde_json::from_slice(b).context("decoding a broadcast StepRetired")
    }
}

/// A non-driving rank's whole run loop: no HTTP, no `Scheduler`. Every
/// iteration is one real step, kept in lockstep with rank 0's own
/// `Scheduler::step()` purely by both sides calling `Model::forward_batch_device`
/// with equivalent `BatchItem`s in the same order -- the NCCL collectives
/// inside that call are what actually enforce the lockstep; this loop's job
/// is only to reconstruct the right arguments from what it's told.
pub fn run_follower(mut model: Model, mut pool: KvPool) -> Result<()> {
    let mut prompts: HashMap<usize, Vec<u32>> = HashMap::new();
    loop {
        let mut buf = Vec::new();
        model.tp_broadcast_bytes(&mut buf, false)?;
        let step = StepPlan::from_bytes(&buf)?;

        for (seq_id, tokens) in &step.admitted {
            let allocated = pool
                .alloc()
                .context("follower: kv pool exhausted admitting a sequence rank 0 already admitted")?;
            anyhow::ensure!(
                allocated.0 == *seq_id,
                "follower's KvPool allocated seq {} where rank 0 reported {seq_id} -- the two \
                 pools' allocation order has diverged; every sequence after this one is now on \
                 the wrong physical slot",
                allocated.0
            );
            prompts.insert(*seq_id, tokens.clone());
        }

        if !step.plan.is_empty() {
            let items: Vec<BatchItem<'_>> = step
                .plan
                .iter()
                .map(|(seq_id, work)| {
                    let seq = SeqId(*seq_id);
                    match work {
                        WorkMsg::Prefill { from, len, wants_logits } => {
                            let prompt = prompts
                                .get(seq_id)
                                .unwrap_or_else(|| panic!("seq {seq_id} planned before it was admitted"));
                            BatchItem {
                                seq,
                                tokens: &prompt[*from..*from + *len],
                                wants_logits: *wants_logits,
                                vision: None,
                                vision_row_offset: 0,
                                mrope: None,
                                mrope_delta: 0,
                            }
                        }
                        WorkMsg::Decode { token } => BatchItem {
                            seq,
                            tokens: std::slice::from_ref(token),
                            wants_logits: true,
                            vision: None,
                            vision_row_offset: 0,
                            mrope: None,
                            mrope_delta: 0,
                        },
                    }
                })
                .collect();
            model.forward_batch_device(&items, &mut pool)?;
        }

        let mut rbuf = Vec::new();
        model.tp_broadcast_bytes(&mut rbuf, false)?;
        let retired = StepRetired::from_bytes(&rbuf)?;
        for seq_id in retired.0 {
            pool.free(SeqId(seq_id));
            prompts.remove(&seq_id);
        }
    }
}
