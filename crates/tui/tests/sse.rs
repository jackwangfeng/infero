//! The hand-rolled HTTP client, against a socket that speaks like the server.
//!
//! Chunked transfer-encoding is the part worth testing: SSE frames do not
//! align with chunk boundaries, so a decoder that assumes they do works right
//! up until a token happens to straddle one.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;

use tuili_tui::client::{Event, stream_chat};

/// Serve one request, writing `chunks` as raw chunked-encoding bodies.
fn serve(chunks: Vec<String>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let (mut sock, _) = listener.accept().unwrap();
        {
            let mut reader = BufReader::new(sock.try_clone().unwrap());
            let mut len = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap() == 0 {
                    return;
                }
                let t = line.trim_end();
                if t.is_empty() {
                    break;
                }
                if let Some(v) = t.to_ascii_lowercase().strip_prefix("content-length:") {
                    len = v.trim().parse().unwrap_or(0);
                }
            }
            let mut body = vec![0u8; len];
            reader.read_exact(&mut body).ok();
        }
        sock.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n",
        )
        .unwrap();
        for c in chunks {
            write!(sock, "{:x}\r\n{}\r\n", c.len(), c).unwrap();
            sock.flush().unwrap();
        }
        sock.write_all(b"0\r\n\r\n").unwrap();
    });
    addr
}

fn collect(addr: &str) -> (String, Option<usize>, Option<String>) {
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let a = addr.to_string();
    let h = std::thread::spawn(move || {
        stream_chat(&a, serde_json::json!({"messages": []}), cancel, tx)
    });

    let mut text = String::new();
    let mut tokens = None;
    let mut error = None;
    while let Ok(ev) = rx.recv() {
        match ev {
            Event::Delta(d) => text.push_str(&d),
            Event::Done {
                completion_tokens, ..
            } => tokens = Some(completion_tokens),
            Event::Failed(e) => error = Some(e),
        }
    }
    h.join().unwrap();
    (text, tokens, error)
}

fn frame(content: &str) -> String {
    format!(
        "data: {}\n\n",
        serde_json::json!({"choices":[{"delta":{"content": content}}]})
    )
}

#[test]
fn reassembles_frames_split_across_chunks() {
    let addr = serve(
        ["Hello", ", ", "world", "!"]
            .iter()
            .map(|c| frame(c))
            .chain(std::iter::once("data: [DONE]\n\n".to_string()))
            .collect(),
    );
    let (text, tokens, error) = collect(&addr);
    assert_eq!(error, None);
    assert_eq!(text, "Hello, world!");
    assert_eq!(tokens, Some(4));
}

#[test]
fn a_frame_straddling_a_chunk_boundary_survives() {
    // One SSE frame cut in half, and two frames sharing a chunk.
    let whole = format!("{}{}", frame("你好"), frame("世界"));
    let (a, b) = whole.split_at(whole.len() / 2);
    let addr = serve(vec![
        a.to_string(),
        b.to_string(),
        format!("{}data: [DONE]\n\n", frame("!")),
    ]);
    let (text, tokens, error) = collect(&addr);
    assert_eq!(error, None);
    assert_eq!(text, "你好世界!");
    assert_eq!(tokens, Some(3));
}

#[test]
fn an_error_frame_is_reported_rather_than_swallowed() {
    let addr = serve(vec![
        frame("partial"),
        "data: {\"error\":{\"message\":\"kv pool is exhausted\"}}\n\n".to_string(),
    ]);
    let (text, _, error) = collect(&addr);
    assert_eq!(text, "partial");
    assert_eq!(error.as_deref(), Some("kv pool is exhausted"));
}

#[test]
fn a_refused_connection_is_an_error_not_a_panic() {
    // Port 1 on loopback: nothing listens there, and binding it needs root.
    let (_, _, error) = collect("127.0.0.1:1");
    assert!(error.is_some(), "expected a connection error");
}
