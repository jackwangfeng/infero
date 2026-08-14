//! A small HTTP/1.1 client for the local server.
//!
//! Hand-rolled rather than pulled from a crate for two reasons: the endpoint is
//! an OpenAI-compatible server on localhost, so the surface needed is tiny, and
//! every HTTP crate honours `http_proxy` from the environment by default —
//! which on a machine with a proxy configured turns a loopback request into a
//! confusing failure.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

/// What the UI hears back from a request in flight.
pub enum Event {
    /// A fragment of the assistant's reply.
    Delta(String),
    Done {
        completion_tokens: usize,
        elapsed: Duration,
    },
    Failed(String),
}

/// What `/health` reports, for the header line.
#[derive(Debug, Clone, Default)]
pub struct Health {
    pub model: String,
    pub quantization: String,
    pub kv_quant: String,
    pub max_seq: usize,
    pub max_seqs: usize,
    pub queue_depth: u64,
    pub offloaded_layers: usize,
}

fn connect(addr: &str) -> Result<TcpStream> {
    let target = addr
        .to_socket_addrs()
        .with_context(|| format!("resolving {addr}"))?
        .next()
        .with_context(|| format!("{addr} resolved to nothing"))?;
    let stream = TcpStream::connect_timeout(&target, Duration::from_secs(5))
        .with_context(|| format!("connecting to {addr}"))?;
    // Tokens arrive one at a time; batching them into MSS-sized packets would
    // make the stream visibly stutter.
    stream.set_nodelay(true)?;
    Ok(stream)
}

/// Send a request and return a reader positioned at the start of the body.
fn send(addr: &str, method: &str, path: &str, body: Option<&str>) -> Result<BodyReader> {
    let mut stream = connect(addr)?;
    let mut head =
        format!("{method} {path} HTTP/1.1\r\nHost: {addr}\r\nAccept: text/event-stream\r\n");
    if let Some(b) = body {
        head.push_str("Content-Type: application/json\r\n");
        head.push_str(&format!("Content-Length: {}\r\n", b.len()));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    if let Some(b) = body {
        stream.write_all(b.as_bytes())?;
    }
    stream.flush()?;

    let mut reader = BufReader::new(stream);
    let mut status = String::new();
    reader.read_line(&mut status)?;
    let code: u16 = status
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .with_context(|| format!("unintelligible status line {status:?}"))?;

    let mut chunked = false;
    let mut content_length = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            bail!("connection closed inside the response headers");
        }
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        let lower = line.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("transfer-encoding:") {
            chunked = v.contains("chunked");
        } else if let Some(v) = lower.strip_prefix("content-length:") {
            content_length = v.trim().parse::<usize>().ok();
        }
    }

    Ok(BodyReader {
        inner: reader,
        chunked,
        remaining: content_length.unwrap_or(usize::MAX),
        status: code,
    })
}

/// The response body, with chunked transfer-encoding unwrapped.
struct BodyReader {
    inner: BufReader<TcpStream>,
    chunked: bool,
    /// Bytes left in the current chunk, or in the whole body when not chunked.
    remaining: usize,
    status: u16,
}

impl Read for BodyReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.chunked && self.remaining == 0 {
            // Chunk boundary: an optional trailing CRLF, then the next size.
            let mut line = String::new();
            self.inner.read_line(&mut line)?;
            if line.trim().is_empty() {
                line.clear();
                self.inner.read_line(&mut line)?;
            }
            let size =
                usize::from_str_radix(line.trim().split(';').next().unwrap_or("0").trim(), 16)
                    .unwrap_or(0);
            if size == 0 {
                return Ok(0);
            }
            self.remaining = size;
        }
        if self.remaining == 0 {
            return Ok(0);
        }
        let want = out.len().min(self.remaining);
        let n = self.inner.read(&mut out[..want])?;
        self.remaining -= n;
        Ok(n)
    }
}

/// GET `/health`, so the header can say what is actually loaded.
pub fn health(addr: &str) -> Result<Health> {
    let mut body = send(addr, "GET", "/health", None)?;
    let status = body.status;
    let mut text = String::new();
    body.read_to_string(&mut text)?;
    if status != 200 {
        bail!("health returned {status}");
    }
    let v: serde_json::Value = serde_json::from_str(&text).context("parsing /health")?;
    Ok(Health {
        model: v["model"].as_str().unwrap_or("unknown").to_string(),
        quantization: v["quantization"].as_str().unwrap_or("?").to_string(),
        kv_quant: v["kv_quant"].as_str().unwrap_or("f16").to_string(),
        max_seq: v["max_seq"].as_u64().unwrap_or(0) as usize,
        max_seqs: v["max_seqs"].as_u64().unwrap_or(1) as usize,
        queue_depth: v["queue_depth"].as_u64().unwrap_or(0),
        offloaded_layers: v["offloaded_layers"].as_u64().unwrap_or(0) as usize,
    })
}

/// Stream a chat completion, forwarding deltas to `tx` until done or cancelled.
///
/// Cancelling drops the connection, which the server notices at its next token
/// and treats as a disconnect — the sequence leaves the batch immediately
/// rather than generating into the void.
pub fn stream_chat(
    addr: &str,
    request: serde_json::Value,
    cancel: Arc<AtomicBool>,
    tx: Sender<Event>,
) {
    let started = Instant::now();
    let body = request.to_string();

    let reader = match send(addr, "POST", "/v1/chat/completions", Some(&body)) {
        Ok(r) => r,
        Err(e) => {
            let _ = tx.send(Event::Failed(format!("{e:#}")));
            return;
        }
    };
    let status = reader.status;
    let mut lines = BufReader::new(reader).lines();

    if status != 200 {
        let detail: String = lines.by_ref().flatten().collect::<Vec<_>>().join(" ");
        let message = serde_json::from_str::<serde_json::Value>(&detail)
            .ok()
            .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
            .unwrap_or_else(|| detail.trim().to_string());
        let _ = tx.send(Event::Failed(format!("HTTP {status}: {message}")));
        return;
    }

    let mut tokens = 0usize;
    for line in lines {
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                let _ = tx.send(Event::Failed(format!("stream ended: {e}")));
                return;
            }
        };
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        if payload == "[DONE]" {
            break;
        }
        let Ok(chunk) = serde_json::from_str::<serde_json::Value>(payload) else {
            continue;
        };
        if let Some(message) = chunk["error"]["message"].as_str() {
            let _ = tx.send(Event::Failed(message.to_string()));
            return;
        }
        if let Some(text) = chunk["choices"][0]["delta"]["content"].as_str()
            && !text.is_empty()
        {
            tokens += 1;
            if tx.send(Event::Delta(text.to_string())).is_err() {
                return; // the UI is gone
            }
        }
    }

    let _ = tx.send(Event::Done {
        completion_tokens: tokens,
        elapsed: started.elapsed(),
    });
}
