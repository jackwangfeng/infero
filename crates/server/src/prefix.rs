//! Reuse of already-computed KV across requests.
//!
//! A prompt's keys and values depend on the token ids and their positions and
//! nothing else — not on the sampler, not on what shares the batch. So two
//! requests with a common leading run of tokens have, for that run, byte-equal
//! KV, and the second one can point at the first one's slots instead of
//! recomputing them. On an agent workload, where every turn re-sends a long
//! system prompt and the whole conversation, that run is most of the prompt.
//!
//! What is cached is a *block*, `BLOCK` tokens wide, keyed by a hash chain over
//! every block before it. The chain is what makes the key a prefix identity
//! rather than a content identity: the same 32 tokens appearing at a different
//! place in a different conversation hash differently, because their parent
//! does. A content-only key would hand a request the KV of tokens that sat at
//! other positions, which is a wrong answer that reads as a subtle quality
//! regression.
//!
//! Partial trailing blocks are not cached. They would be correct — a prefix is
//! a prefix — but every entry costs a slot that a running sequence could use,
//! and a 5-token tail buys almost nothing.
//!
//! ## What this does not do
//!
//! Models with recurrent layers are refused outright. Sharing KV reconstructs
//! the attention half of a GatedDeltaNet block stack and nothing of the
//! recurrence, so a sequence that skipped the prefill would enter the first
//! linear block with zeroed state and condition on nothing. Making it work
//! means snapshotting the recurrent state at block boundaries, which is its own
//! piece of work; until then [`PrefixCache::new`] returns `None` for such a
//! pool and every lookup misses.

use std::collections::HashMap;

use infero_model::{KvPool, SeqId};

/// Tokens per cached block.
///
/// A block is the unit of both sharing and eviction. Small blocks share more of
/// a divergent prompt and cost more entries; 32 is one `warp`'s worth of tokens
/// and about a percent of a typical agent prompt, which is finer than the point
/// where two conversations usually diverge.
pub const BLOCK: usize = 32;

/// One cached block: the slots holding its KV, and who is reading them.
struct Entry {
    /// Exactly `BLOCK` slots, in logical order.
    slots: Vec<i32>,
    /// How many live sequences currently borrow this block. Evicting a borrowed
    /// block would hand a running request's keys to the allocator, so this is
    /// the one thing eviction has to respect.
    refs: usize,
    /// Bumped on every hit, so eviction can take the coldest. A counter rather
    /// than a clock: monotonic, cheap, and it needs no notion of now.
    last_used: u64,
}

pub struct PrefixCache {
    blocks: HashMap<u64, Entry>,
    tick: u64,
    hits: u64,
    tokens_saved: u64,
    lookups: u64,
}

/// The hash of block `index`, given the hash of everything before it.
///
/// FxHash-style mixing over the parent and the block's tokens. Not
/// cryptographic: a collision hands back the wrong KV, but the inputs are token
/// ids from our own tokenizer rather than anything an attacker chooses, and 64
/// bits over the number of blocks a pool can hold makes it far less likely than
/// the bugs it would be blamed for.
fn chain(parent: u64, tokens: &[u32]) -> u64 {
    const K: u64 = 0x517c_c1b7_2722_0a95;
    let mut h = parent ^ 0x9e37_79b9_7f4a_7c15;
    for &t in tokens {
        h = (h ^ t as u64).wrapping_mul(K);
        h ^= h >> 29;
    }
    h.wrapping_mul(K)
}

impl PrefixCache {
    /// `None` when this pool's model cannot share a prefix — see the module
    /// note. The caller then has no cache at all rather than one that quietly
    /// never hits.
    pub fn new(pool: &KvPool) -> Option<Self> {
        if pool.has_recurrent_state() {
            tracing::info!(
                "prefix caching is off: this model has recurrent layers, whose \
                 state a shared KV prefix does not reconstruct"
            );
            return None;
        }
        Some(Self {
            blocks: HashMap::new(),
            tick: 0,
            hits: 0,
            tokens_saved: 0,
            lookups: 0,
        })
    }

    /// The longest cached prefix of `tokens`, as (slots, hashes per block).
    ///
    /// `limit` caps the answer. The caller's cap is `tokens.len() - 1`: a
    /// sequence whose whole prompt came from the cache has nothing to run a
    /// forward pass over, and the step that would produce its first logits
    /// would have no rows.
    pub fn lookup(&mut self, tokens: &[u32], limit: usize) -> Prefix {
        self.lookups += 1;
        let mut slots = Vec::new();
        let mut hashes = Vec::new();
        let mut parent = 0u64;
        self.tick += 1;
        for block in tokens.chunks_exact(BLOCK) {
            let h = chain(parent, block);
            if slots.len() + BLOCK > limit {
                break;
            }
            match self.blocks.get_mut(&h) {
                Some(e) => {
                    e.last_used = self.tick;
                    slots.extend_from_slice(&e.slots);
                    hashes.push(h);
                    parent = h;
                }
                None => break,
            }
        }
        if !slots.is_empty() {
            self.hits += 1;
            self.tokens_saved += slots.len() as u64;
            for h in &hashes {
                self.blocks.get_mut(h).expect("just matched").refs += 1;
            }
        }
        Prefix { slots, hashes }
    }

    /// Give back the references a [`Prefix`] took, when its sequence retires.
    pub fn release(&mut self, prefix: &Prefix) {
        for h in &prefix.hashes {
            if let Some(e) = self.blocks.get_mut(h) {
                e.refs = e.refs.saturating_sub(1);
            }
        }
    }

    /// Record the blocks of a finished sequence that are not cached yet.
    ///
    /// `tokens` is everything the sequence computed KV for — prompt and
    /// generation both, since the next turn of a conversation sends the
    /// generation back as prompt. `slots` must be that sequence's slot list, so
    /// block `i` is `slots[i * BLOCK..]`.
    ///
    /// Takes ownership of the slots it keeps by way of `KvPool::keep_prefix`,
    /// which is what stops `free` from returning them. Blocks past the first
    /// miss are still inserted: the chain only needs its parent, and the parent
    /// is the block before it whether that one was already present or has just
    /// been added.
    pub fn insert(&mut self, pool: &mut KvPool, id: SeqId, tokens: &[u32]) -> anyhow::Result<()> {
        let full = tokens.len() / BLOCK;
        if full == 0 {
            return Ok(());
        }
        // Take the whole block-aligned prefix out of the sequence's hands in one
        // call, before deciding which blocks are new: `keep_prefix` moves the
        // ownership boundary and its argument has to be monotonic.
        let kept = pool.keep_prefix(id, full * BLOCK)?;
        self.tick += 1;
        let mut parent = 0u64;
        for i in 0..full {
            let h = chain(parent, &tokens[i * BLOCK..(i + 1) * BLOCK]);
            let slots = &kept[i * BLOCK..(i + 1) * BLOCK];
            self.blocks.entry(h).or_insert_with(|| Entry {
                slots: slots.to_vec(),
                refs: 0,
                last_used: 0,
            });
            // Even an entry that already existed is warm now — it is part of a
            // prefix something just finished reading.
            let e = self.blocks.get_mut(&h).expect("inserted above");
            e.last_used = self.tick;
            parent = h;
        }
        Ok(())
    }

    /// Free cached blocks until the pool has `want` slots, coldest first.
    ///
    /// Returns how many slots were recovered. A borrowed block is skipped
    /// rather than waited for: whatever is reading it will release it, and the
    /// caller's alternative is to queue the request, which it was going to do
    /// anyway.
    pub fn evict_for(&mut self, pool: &mut KvPool, want: usize) -> usize {
        let mut freed = 0usize;
        while pool.free_slots() + freed < want {
            let victim = self
                .blocks
                .iter()
                .filter(|(_, e)| e.refs == 0)
                .min_by_key(|(_, e)| e.last_used)
                .map(|(h, _)| *h);
            let Some(h) = victim else { break };
            let e = self.blocks.remove(&h).expect("just found");
            pool.release_slots(&e.slots);
            freed += e.slots.len();
        }
        if freed > 0 {
            tracing::debug!(freed, want, entries = self.blocks.len(), "prefix cache evicted");
        }
        freed
    }

    /// Blocks held, and the slots they cost.
    pub fn held(&self) -> (usize, usize) {
        (self.blocks.len(), self.blocks.len() * BLOCK)
    }

    pub fn stats(&self) -> (u64, u64, u64) {
        (self.lookups, self.hits, self.tokens_saved)
    }
}

/// A cached prefix a sequence is reading, and the references it holds.
///
/// Carried by the running sequence so that retiring it releases exactly what it
/// took. Dropping this without [`PrefixCache::release`] pins the blocks
/// forever, which does not corrupt anything and does stop eviction from ever
/// reclaiming them.
#[derive(Default)]
pub struct Prefix {
    pub slots: Vec<i32>,
    hashes: Vec<u64>,
}

impl Prefix {
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The chain, not the content: the same tokens under a different parent are
    /// a different block. This is the property that keeps a cached block tied to
    /// the positions it was computed at.
    #[test]
    fn the_same_tokens_hash_differently_under_a_different_parent() {
        let block: Vec<u32> = (0..BLOCK as u32).collect();
        assert_ne!(chain(0, &block), chain(1, &block));
        assert_eq!(chain(7, &block), chain(7, &block));
    }

    #[test]
    fn a_longer_prompt_that_shares_a_prefix_hashes_the_shared_blocks_the_same() {
        let a: Vec<u32> = (0..(BLOCK * 3) as u32).collect();
        let mut b = a[..BLOCK * 2].to_vec();
        b.extend((1000..1000 + BLOCK as u32).map(|x| x));

        let hashes = |t: &[u32]| -> Vec<u64> {
            let mut p = 0u64;
            t.chunks_exact(BLOCK)
                .map(|c| {
                    p = chain(p, c);
                    p
                })
                .collect()
        };
        let (ha, hb) = (hashes(&a), hashes(&b));
        assert_eq!(ha[..2], hb[..2], "the shared blocks must match");
        assert_ne!(ha[2], hb[2], "the divergent block must not");
    }
}
