//! Stop-sequence scanning for a streaming response.

#[derive(Debug, PartialEq, Eq)]
pub enum StopScan {
    /// A stop sequence was found at this byte offset; everything before it is
    /// safe to emit and generation ends.
    Hit(usize),
    /// No stop sequence yet; this many leading bytes can never be part of one.
    Release(usize),
}

impl StopScan {
    /// Bytes the caller should drain from its buffer and send.
    pub fn release_len(&self) -> usize {
        match *self {
            StopScan::Hit(n) | StopScan::Release(n) => n,
        }
    }

    pub fn is_hit(&self) -> bool {
        matches!(self, StopScan::Hit(_))
    }
}

/// Decide how much of `pending` can be emitted.
///
/// Streaming makes stop sequences awkward: `"</"` might be the start of
/// `"</answer>"`, so it cannot be sent yet, but it also might not be. Text is
/// released only once no stop sequence could still begin inside it.
pub fn split_at_stop(pending: &str, stop: &[String]) -> StopScan {
    if stop.is_empty() {
        return StopScan::Release(pending.len());
    }

    if let Some(at) = stop
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| pending.find(s.as_str()))
        .min()
    {
        return StopScan::Hit(at);
    }

    // The longest suffix of `pending` that prefixes some stop sequence has to
    // stay buffered.
    let mut hold = 0usize;
    for s in stop.iter().filter(|s| !s.is_empty()) {
        let max = s.len().min(pending.len());
        for n in (1..=max).rev() {
            let start = pending.len() - n;
            if !pending.is_char_boundary(start) {
                continue;
            }
            if s.as_bytes().starts_with(&pending.as_bytes()[start..]) {
                hold = hold.max(n);
                break;
            }
        }
    }
    StopScan::Release(pending.len() - hold)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn release<'a>(pending: &'a str, stop: &[&str]) -> &'a str {
        let stop: Vec<String> = stop.iter().map(|s| s.to_string()).collect();
        match split_at_stop(pending, &stop) {
            StopScan::Release(n) => &pending[..n],
            StopScan::Hit(n) => panic!("unexpected stop, prefix {:?}", &pending[..n]),
        }
    }

    #[test]
    fn text_flows_when_there_are_no_stop_sequences() {
        assert_eq!(release("hello", &[]), "hello");
    }

    #[test]
    fn a_partial_stop_sequence_is_held_back() {
        assert_eq!(release("answer: </", &["</answer>"]), "answer: ");
        assert_eq!(release("answer: <", &["</answer>"]), "answer: ");
        assert_eq!(release("answer: x", &["</answer>"]), "answer: x");
    }

    #[test]
    fn a_complete_stop_sequence_truncates() {
        let stop = vec!["</answer>".to_string()];
        let text = "done</answer>tail";
        match split_at_stop(text, &stop) {
            StopScan::Hit(at) => assert_eq!(&text[..at], "done"),
            StopScan::Release(n) => {
                panic!("missed the stop sequence, released {:?}", &text[..n])
            }
        }
    }

    #[test]
    fn the_earliest_of_several_stops_wins() {
        let stop = vec!["END".to_string(), "X".to_string()];
        let text = "abXcdEND";
        match split_at_stop(text, &stop) {
            StopScan::Hit(at) => assert_eq!(&text[..at], "ab"),
            StopScan::Release(n) => panic!("released {:?}", &text[..n]),
        }
    }

    #[test]
    fn holding_back_never_splits_a_character() {
        // The held suffix must stay on a char boundary or the string slice
        // would panic.
        assert_eq!(release("答案：中", &["中文"]), "答案：");
        assert_eq!(release("答案：文", &["中文"]), "答案：文");
    }
}
