//! Conversation state and the editing model behind the input box.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::time::Duration;

use crate::client::{Event, Health};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    System,
    Error,
}

impl Role {
    pub fn label(self) -> &'static str {
        match self {
            Role::User => "you",
            Role::Assistant => "assistant",
            Role::System => "system",
            Role::Error => "error",
        }
    }

    /// What the API calls this role, for the ones it accepts.
    pub fn api(self) -> Option<&'static str> {
        match self {
            Role::User => Some("user"),
            Role::Assistant => Some("assistant"),
            Role::System => Some("system"),
            Role::Error => None,
        }
    }
}

pub struct Message {
    pub role: Role,
    pub text: String,
    /// Filled in once a reply finishes.
    pub stats: Option<String>,
}

/// Whether a reply is in flight, and how to stop it.
pub struct InFlight {
    pub events: Receiver<Event>,
    pub cancel: Arc<AtomicBool>,
}

pub struct App {
    pub addr: String,
    pub health: Health,
    pub messages: Vec<Message>,
    pub input: String,
    /// Caret position in `input`, as a byte offset.
    pub cursor: usize,
    /// Lines scrolled up from the bottom. Zero follows the newest output.
    pub scroll: usize,
    pub in_flight: Option<InFlight>,
    pub status: Option<String>,
    pub temperature: f32,
    pub max_tokens: usize,
    pub should_quit: bool,
    /// Total wrapped height of the transcript at the last render, so paging
    /// can be clamped to something real.
    pub last_transcript_height: usize,
    pub last_viewport_height: usize,
}

impl App {
    pub fn new(addr: String, health: Health, system: Option<String>) -> Self {
        let mut messages = Vec::new();
        if let Some(prompt) = system {
            messages.push(Message {
                role: Role::System,
                text: prompt,
                stats: None,
            });
        }
        Self {
            addr,
            health,
            messages,
            input: String::new(),
            cursor: 0,
            scroll: 0,
            in_flight: None,
            status: None,
            temperature: 0.7,
            max_tokens: 512,
            should_quit: false,
            last_transcript_height: 0,
            last_viewport_height: 0,
        }
    }

    pub fn is_generating(&self) -> bool {
        self.in_flight.is_some()
    }

    /// The conversation in the shape `/v1/chat/completions` wants.
    pub fn api_messages(&self) -> serde_json::Value {
        let items: Vec<serde_json::Value> = self
            .messages
            .iter()
            .filter(|m| !m.text.is_empty())
            .filter_map(|m| {
                m.role
                    .api()
                    .map(|role| serde_json::json!({ "role": role, "content": m.text }))
            })
            .collect();
        serde_json::Value::Array(items)
    }

    pub fn push(&mut self, role: Role, text: impl Into<String>) {
        self.messages.push(Message {
            role,
            text: text.into(),
            stats: None,
        });
        self.scroll = 0;
    }

    /// Drain whatever the worker thread has produced since the last tick.
    ///
    /// Returns true if anything changed, so the caller only redraws when there
    /// is something new to show.
    pub fn poll(&mut self) -> bool {
        let Some(flight) = &self.in_flight else {
            return false;
        };
        let mut changed = false;
        let mut finished = false;

        loop {
            match flight.events.try_recv() {
                Ok(Event::Delta(text)) => {
                    if let Some(last) = self.messages.last_mut() {
                        last.text.push_str(&text);
                    }
                    changed = true;
                }
                Ok(Event::Done {
                    completion_tokens,
                    elapsed,
                }) => {
                    let stats = format_stats(completion_tokens, elapsed);
                    if let Some(last) = self.messages.last_mut() {
                        last.stats = Some(stats);
                        if last.text.is_empty() {
                            last.text.push_str("(empty reply)");
                        }
                    }
                    finished = true;
                    changed = true;
                    break;
                }
                Ok(Event::Failed(message)) => {
                    // Drop the empty assistant turn the error replaces.
                    if self
                        .messages
                        .last()
                        .is_some_and(|m| m.role == Role::Assistant && m.text.is_empty())
                    {
                        self.messages.pop();
                    }
                    self.push(Role::Error, message);
                    finished = true;
                    changed = true;
                    break;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    finished = true;
                    break;
                }
            }
        }

        if finished {
            self.in_flight = None;
            self.status = None;
        }
        changed
    }

    /// Stop the current reply, keeping whatever text already arrived.
    pub fn cancel(&mut self) {
        if let Some(flight) = self.in_flight.take() {
            flight.cancel.store(true, Ordering::Relaxed);
            if let Some(last) = self.messages.last_mut()
                && last.role == Role::Assistant
            {
                if last.text.is_empty() {
                    last.text.push_str("(cancelled)");
                }
                last.stats = Some("cancelled".into());
            }
            self.status = None;
        }
    }

    // ---- input editing --------------------------------------------------

    pub fn insert(&mut self, ch: char) {
        self.input.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
    }

    pub fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let prev = self.input[..self.cursor]
            .chars()
            .next_back()
            .map(char::len_utf8)
            .unwrap_or(1);
        self.cursor -= prev;
        self.input.remove(self.cursor);
    }

    pub fn delete(&mut self) {
        if self.cursor < self.input.len() {
            self.input.remove(self.cursor);
        }
    }

    pub fn move_left(&mut self) {
        if let Some(ch) = self.input[..self.cursor].chars().next_back() {
            self.cursor -= ch.len_utf8();
        }
    }

    pub fn move_right(&mut self) {
        if let Some(ch) = self.input[self.cursor..].chars().next() {
            self.cursor += ch.len_utf8();
        }
    }

    pub fn move_home(&mut self) {
        self.cursor = 0;
    }

    pub fn move_end(&mut self) {
        self.cursor = self.input.len();
    }

    /// Delete the word before the caret.
    pub fn delete_word(&mut self) {
        let head = &self.input[..self.cursor];
        let trimmed = head.trim_end_matches(' ');
        let start = trimmed.rfind(' ').map(|i| i + 1).unwrap_or(0);
        self.input.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    pub fn take_input(&mut self) -> String {
        self.cursor = 0;
        std::mem::take(&mut self.input)
    }

    // ---- scrolling ------------------------------------------------------

    pub fn scroll_up(&mut self, lines: usize) {
        let max = self
            .last_transcript_height
            .saturating_sub(self.last_viewport_height);
        self.scroll = (self.scroll + lines).min(max);
    }

    pub fn scroll_down(&mut self, lines: usize) {
        self.scroll = self.scroll.saturating_sub(lines);
    }

    pub fn scroll_to_bottom(&mut self) {
        self.scroll = 0;
    }
}

fn format_stats(tokens: usize, elapsed: Duration) -> String {
    let secs = elapsed.as_secs_f64();
    if tokens == 0 || secs <= 0.0 {
        return format!("{secs:.1}s");
    }
    format!("{:.0} tok/s · {secs:.1}s", tokens as f64 / secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> App {
        App::new("127.0.0.1:8080".into(), Health::default(), None)
    }

    #[test]
    fn editing_respects_character_boundaries() {
        let mut a = app();
        for ch in "你好ab".chars() {
            a.insert(ch);
        }
        assert_eq!(a.input, "你好ab");
        a.backspace();
        a.backspace();
        assert_eq!(a.input, "你好");
        a.backspace();
        assert_eq!(a.input, "你", "a multi-byte char must delete as one");
        a.move_left();
        assert_eq!(a.cursor, 0);
        a.move_left();
        assert_eq!(a.cursor, 0, "moving past the start must not panic");
    }

    #[test]
    fn delete_word_stops_at_the_previous_space() {
        let mut a = app();
        for ch in "hello brave world".chars() {
            a.insert(ch);
        }
        a.delete_word();
        assert_eq!(a.input, "hello brave ");
        a.delete_word();
        assert_eq!(a.input, "hello ");
    }

    #[test]
    fn system_prompts_reach_the_api_and_errors_do_not() {
        let mut a = App::new("x".into(), Health::default(), Some("be terse".into()));
        a.push(Role::User, "hi");
        a.push(Role::Error, "connection refused");
        let msgs = a.api_messages();
        let arr = msgs.as_array().unwrap();
        assert_eq!(arr.len(), 2, "the error turn must not be sent back");
        assert_eq!(arr[0]["role"], "system");
        assert_eq!(arr[1]["content"], "hi");
    }

    #[test]
    fn scrolling_is_clamped_to_the_transcript() {
        let mut a = app();
        a.last_transcript_height = 50;
        a.last_viewport_height = 20;
        a.scroll_up(1000);
        assert_eq!(a.scroll, 30, "cannot scroll past the top");
        a.scroll_down(1000);
        assert_eq!(a.scroll, 0, "cannot scroll past the bottom");
    }

    #[test]
    fn a_new_message_jumps_back_to_the_bottom() {
        let mut a = app();
        a.last_transcript_height = 50;
        a.last_viewport_height = 10;
        a.scroll_up(5);
        assert_eq!(a.scroll, 5);
        a.push(Role::User, "hi");
        assert_eq!(a.scroll, 0);
    }

    #[test]
    fn stats_read_sensibly() {
        assert_eq!(format_stats(100, Duration::from_secs(2)), "50 tok/s · 2.0s");
        assert_eq!(format_stats(0, Duration::from_millis(300)), "0.3s");
    }
}
