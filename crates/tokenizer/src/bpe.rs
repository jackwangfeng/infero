//! Byte-pair merging over a single pre-token.
//!
//! Merges are resolved in token-id space rather than string space: every half
//! of every merge rule is itself a vocabulary entry, so a pre-token can start
//! as the ids of its individual alphabet characters and stay a `Vec<u32>` for
//! the whole merge loop. No allocation, no string hashing per step.

use rustc_hash::FxHashMap;

#[derive(Clone, Copy)]
struct Merge {
    rank: u32,
    result: u32,
}

pub struct Bpe {
    /// (left id, right id) -> rank and the id of the concatenation.
    rules: FxHashMap<(u32, u32), Merge>,
}

impl Bpe {
    /// Build from GGUF's `tokenizer.ggml.merges` (entries like `"Ġ t"`) plus
    /// the vocabulary needed to resolve each half to an id.
    ///
    /// Rules whose halves or result are missing from the vocab are dropped;
    /// they could never fire anyway.
    pub fn new<'a>(
        merge_lines: impl IntoIterator<Item = &'a str>,
        vocab: &FxHashMap<&str, u32>,
    ) -> Self {
        let mut rules = FxHashMap::default();
        let mut joined = String::new();

        for (rank, line) in merge_lines.into_iter().enumerate() {
            let Some((a, b)) = line.split_once(' ') else {
                continue;
            };
            let (Some(&ia), Some(&ib)) = (vocab.get(a), vocab.get(b)) else {
                continue;
            };
            joined.clear();
            joined.push_str(a);
            joined.push_str(b);
            let Some(&result) = vocab.get(joined.as_str()) else {
                continue;
            };
            // First rule wins: merge files are ordered by priority.
            rules.entry((ia, ib)).or_insert(Merge {
                rank: rank as u32,
                result,
            });
        }

        Self { rules }
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Repeatedly apply the lowest-ranked applicable merge, in place.
    ///
    /// Pre-tokens are short (the regex caps them at a word or a run of
    /// punctuation), so the straightforward rescan beats maintaining a heap.
    pub fn merge(&self, ids: &mut Vec<u32>) {
        while ids.len() >= 2 {
            let mut best: Option<(usize, Merge)> = None;
            for i in 0..ids.len() - 1 {
                if let Some(&m) = self.rules.get(&(ids[i], ids[i + 1]))
                    && best.is_none_or(|(_, b)| m.rank < b.rank)
                {
                    best = Some((i, m));
                }
            }
            let Some((i, m)) = best else { return };
            ids[i] = m.result;
            ids.remove(i + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy vocab: characters a/b/c plus the pieces "ab" and "abc".
    fn toy() -> (FxHashMap<&'static str, u32>, Bpe) {
        let vocab: FxHashMap<&str, u32> = [("a", 0), ("b", 1), ("c", 2), ("ab", 3), ("abc", 4)]
            .into_iter()
            .collect();
        let bpe = Bpe::new(["a b", "ab c"], &vocab);
        (vocab, bpe)
    }

    #[test]
    fn merges_by_rank() {
        let (_, bpe) = toy();
        let mut ids = vec![0, 1, 2]; // a b c
        bpe.merge(&mut ids);
        assert_eq!(ids, vec![4]); // abc
    }

    #[test]
    fn stops_when_no_rule_applies() {
        let (_, bpe) = toy();
        let mut ids = vec![2, 2]; // c c
        bpe.merge(&mut ids);
        assert_eq!(ids, vec![2, 2]);
    }

    #[test]
    fn drops_rules_with_unknown_halves() {
        let vocab: FxHashMap<&str, u32> = [("a", 0)].into_iter().collect();
        let bpe = Bpe::new(["a z", "q q"], &vocab);
        assert!(bpe.is_empty());
    }
}
