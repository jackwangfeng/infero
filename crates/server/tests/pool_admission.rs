//! The KV pool must never admit (or schedule) more concurrent work than it
//! can actually back, and one sequence running short of room must never take
//! unrelated sequences down with it.
//!
//! Two real bugs, found under real concurrent long-context traffic against
//! production: (1) `Scheduler::admit()` checked each waiting request's need
//! against the same stale `free_slots()` snapshot, so several large requests
//! admitted in the same pass could each individually fit while collectively
//! overcommitting the pool; (2) `Scheduler::plan()` scheduled prefill/decode
//! work purely against the compute (`batch_tokens`) budget, with no matching
//! check against the KV pool's real remaining slots -- so a step could ask
//! `KvPool::extend()` for more than was left, and that one sequence's
//! "kv pool exhausted" error propagated all the way to `engine.rs`, which
//! reacted to ANY step failure by failing every running and waiting request,
//! not just the one that was short.
//!
//! Skipped when `models/` is empty, same convention as `http.rs`.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::OnceLock;

use tower_http::cors::{Any, CorsLayer};
use infero_server::{Engine, routes};

fn model_path() -> Option<PathBuf> {
    let p = std::env::var("INFERO_TEST_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../models/qwen2.5-0.5b-instruct-q8_0.gguf")
        });
    p.exists().then_some(p)
}

/// Deliberately small: `max_seq` (ctx) 512, `max_seqs` 4, `kv_slots` 700 --
/// two ~280-token prompts already approach the ceiling, so four admitted at
/// once (each individually well under 512, and each below the pool alone)
/// collectively need roughly 4x 265 =~ 1060 slots against a 700-slot pool --
/// comfortably overcommitted, but with enough real margin that eviction (a
/// finished sequence's blocks are reclaimable, not merely "used up") has
/// genuine room to keep pace with real scheduling jitter across a run,
/// rather than hinging on a knife-edge race between "this request's check"
/// and "any other sequence has finished long enough ago to be evictable".
fn server() -> Option<SocketAddr> {
    static ADDR: OnceLock<Option<SocketAddr>> = OnceLock::new();
    *ADDR.get_or_init(|| {
        let path = model_path()?;
        let engine = Engine::start(
            path.to_str().unwrap(),
            512,
            0,
            infero_model::KvCacheQuant::F16,
            usize::MAX,
            4,
            Some(700),
            4096,
            16,
            infero_server::video::DEFAULT_TARGET_FPS,
            0.0,
            None,
        )
        .expect("starting engine");
        let app = routes::router(engine, infero_server::auth::AuthConfig::new(None, None)).layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );

        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
            rt.block_on(async move {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                    .await
                    .expect("binding an ephemeral port");
                tx.send(listener.local_addr().unwrap()).unwrap();
                axum::serve(listener, app).await.expect("serving");
            });
        });
        Some(rx.recv().expect("server failed to start"))
    })
}

macro_rules! addr {
    () => {
        match server() {
            Some(a) => a,
            None => {
                eprintln!("skipping: no model in models/");
                return;
            }
        }
    };
}

fn post(addr: SocketAddr, path: &str, body: serde_json::Value) -> (u16, String) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async move {
        let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let io = hyper_util::rt::TokioIo::new(stream);
        let (mut sender, conn) = hyper::client::conn::http1::handshake(io).await.unwrap();
        tokio::spawn(async move {
            let _ = conn.await;
        });

        let body = body.to_string();
        let req = hyper::Request::builder()
            .method("POST")
            .uri(path)
            .header("host", addr.to_string())
            .header("content-type", "application/json")
            .body(body)
            .unwrap();
        let res = sender.send_request(req).await.unwrap();
        let status = res.status().as_u16();
        let bytes = read_all(res).await;
        (status, bytes)
    })
}

async fn read_all(res: hyper::Response<hyper::body::Incoming>) -> String {
    use http_body_util::BodyExt;
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn json(body: &str) -> serde_json::Value {
    serde_json::from_str(body).unwrap_or_else(|e| panic!("bad json ({e}): {body}"))
}

/// A prompt long enough (~280 tokens on this tokenizer, repeated "banana" --
/// picked because it tokenizes compactly and predictably; less common filler
/// words can blow up to 2-3x the token count under BPE and accidentally
/// exceed `max_seq` instead of exercising pool pacing) that two of them
/// already approach the 600-slot pool, and four admitted at once would need
/// roughly 4x that -- comfortably more than the pool holds.
///
/// The unique tag goes at the *start*, not the end -- four requests that
/// share one long prefix and differ only in a trailing tag would exercise
/// the *prefix cache*'s own sharing/eviction behavior (a real, separate,
/// working-as-intended feature: cached blocks a live sibling request still
/// references cannot be evicted), which is a different concern from the
/// admission/plan pacing this test targets. Diverging from the first token
/// keeps the two concerns apart.
fn long_prompt(i: usize) -> String {
    format!(
        "[request {i}] Repeat the following word many times for testing purposes: {}",
        "banana ".repeat(220)
    )
}

/// Four requests are admitted at once, each individually well under the
/// 512-token ctx and each fitting the 600-slot pool alone, but not four at
/// once. Before the fix: `admit()` checked each against the same stale
/// `free_slots()` snapshot and happily admitted all four; `plan()` then asked
/// `KvPool::extend()` for more than was left, which failed the whole batch
/// step, which made `engine.rs` call `fail_all()` over every running and
/// waiting request -- so unrelated requests (not just the ones that
/// genuinely didn't fit) would come back as errors too.
///
/// After the fix: `admit()` tracks what this pass has already promised, and
/// `plan()` cannot schedule more new tokens than the pool can actually back
/// this step. The requests that do not fit yet are paced across steps (or
/// admission passes) instead of being admitted and then crashing -- every
/// request must eventually complete with a real answer, and none may fail
/// as a side effect of another request's resource shortfall.
#[test]
fn concurrent_requests_exceeding_pool_capacity_all_complete_without_cross_failure() {
    let addr = addr!();

    let handles: Vec<_> = (0..4)
        .map(|i| {
            std::thread::spawn(move || {
                post(
                    addr,
                    "/v1/chat/completions",
                    serde_json::json!({
                        "model": "test",
                        "messages": [{"role": "user", "content": long_prompt(i)}],
                        "max_tokens": 8,
                        "temperature": 0.0,
                    }),
                )
            })
        })
        .collect();

    let results: Vec<(u16, String)> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    for (i, (status, body)) in results.iter().enumerate() {
        assert_eq!(
            *status, 200,
            "request {i} failed (status {status}): {body} -- a request that only ran short \
             because OTHER concurrent requests were also admitted must still be paced and \
             retried, not failed as a side effect of the pool being busy"
        );
        let v = json(body);
        let content = v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        assert!(
            !content.is_empty(),
            "request {i} returned 200 but no real content: {body}"
        );
    }
}

/// A single request whose prompt genuinely cannot fit in the pool even alone
/// (longer than the 512-token ctx itself) must still be refused cleanly --
/// the fix must not turn a real "this can never fit" case into an infinite
/// wait.
#[test]
fn a_prompt_that_can_never_fit_is_still_refused_cleanly() {
    let addr = addr!();
    let huge = format!("{} end", "banana ".repeat(600)); // well over the 512 ctx
    let (status, body) = post(
        addr,
        "/v1/chat/completions",
        serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": huge}],
            "max_tokens": 8,
        }),
    );
    assert_ne!(status, 200, "a prompt over the context limit must be refused, not silently truncated or hung: {body}");
}
