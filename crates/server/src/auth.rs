//! API key authentication and per-key/per-IP rate limiting.
//!
//! Both are opt-in and additive: a server started without `--api-keys` and
//! without `--rate-limit-per-minute` set below its own default-off sentinel
//! behaves exactly as before this module existed. `/health` is never gated by
//! either — an orchestrator's liveness probe shouldn't need a key or count
//! against a client's own budget.

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use axum::extract::{ConnectInfo, Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::api::{ErrorBody, ErrorDetail};

/// `None` disables the corresponding check entirely -- the default,
/// unauthenticated/unlimited behavior this server has always had.
#[derive(Clone)]
pub struct AuthConfig {
    keys: Option<HashSet<String>>,
    rate_limiter: Option<std::sync::Arc<RateLimiter>>,
}

impl AuthConfig {
    pub fn new(api_keys_csv: Option<&str>, rate_limit_per_minute: Option<u32>) -> Self {
        let keys = api_keys_csv.map(|csv| {
            csv.split(',')
                .map(|k| k.trim().to_string())
                .filter(|k| !k.is_empty())
                .collect::<HashSet<_>>()
        });
        let rate_limiter = rate_limit_per_minute.map(|limit| std::sync::Arc::new(RateLimiter::new(limit)));
        Self { keys, rate_limiter }
    }

    pub fn auth_enabled(&self) -> bool {
        self.keys.is_some()
    }

    pub fn rate_limit_enabled(&self) -> bool {
        self.rate_limiter.is_some()
    }
}

fn error_response(status: StatusCode, message: impl Into<String>, code: &'static str) -> Response {
    (
        status,
        Json(ErrorBody {
            error: ErrorDetail {
                message: message.into(),
                kind: code,
                code: Some(code),
            },
        }),
    )
        .into_response()
}

fn bearer_token(headers: &HeaderMap) -> Option<String> {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|s| s.trim().to_string())
}

/// Rejects with 401 unless `Authorization: Bearer <key>` names a configured
/// key. Only ever installed on the router when `AuthConfig::auth_enabled()`
/// is true (see `routes::router`) -- when no keys are configured this
/// middleware is never in the request path at all, not merely a no-op inside
/// it, so there is no per-request cost when auth is off.
pub async fn require_api_key(
    State(cfg): State<AuthConfig>,
    request: Request,
    next: Next,
) -> Response {
    let Some(keys) = &cfg.keys else {
        // Defensive: shouldn't be reachable, this layer isn't installed when
        // `keys` is `None`. Fail open to "unauthenticated" rather than
        // rejecting every request if it somehow is.
        return next.run(request).await;
    };
    match bearer_token(request.headers()) {
        Some(token) if keys.contains(&token) => next.run(request).await,
        Some(_) => error_response(
            StatusCode::UNAUTHORIZED,
            "Incorrect API key provided.",
            "invalid_api_key",
        ),
        None => error_response(
            StatusCode::UNAUTHORIZED,
            "You didn't provide an API key. You need to provide your API key in an \
             Authorization header using Bearer auth (i.e. Authorization: Bearer YOUR_KEY).",
            "missing_api_key",
        ),
    }
}

/// A fixed-window limiter: `limit` requests per key per rolling 60s window.
/// Simple, real, and enough to blunt abuse -- not a token-bucket's smoother
/// burst handling, which isn't worth the extra bookkeeping for a first pass.
struct RateLimiter {
    limit: u32,
    window: Duration,
    // key -> (window start, count so far in this window)
    state: Mutex<HashMap<String, (Instant, u32)>>,
}

impl RateLimiter {
    fn new(limit_per_minute: u32) -> Self {
        Self {
            limit: limit_per_minute,
            window: Duration::from_secs(60),
            state: Mutex::new(HashMap::new()),
        }
    }

    /// `Ok(())` if this call is within budget, `Err(retry_after)` otherwise.
    fn check(&self, key: &str) -> Result<(), Duration> {
        let now = Instant::now();
        let mut state = self.state.lock().expect("rate limiter mutex poisoned");
        let entry = state.entry(key.to_string()).or_insert((now, 0));
        if now.duration_since(entry.0) >= self.window {
            *entry = (now, 0);
        }
        if entry.1 >= self.limit {
            let retry_after = self.window.saturating_sub(now.duration_since(entry.0));
            return Err(retry_after);
        }
        entry.1 += 1;
        Ok(())
    }
}

/// Keyed by the caller's API key when auth is configured and the request
/// carries one (so a key's budget follows it across source IPs); by peer IP
/// otherwise. Only installed on the router when a rate limit is configured
/// (see `routes::router`) -- like `require_api_key`, off means genuinely not
/// in the request path, not a no-op check.
pub async fn rate_limit(
    State(cfg): State<AuthConfig>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    request: Request,
    next: Next,
) -> Response {
    let Some(limiter) = &cfg.rate_limiter else {
        return next.run(request).await;
    };
    let key = bearer_token(request.headers()).unwrap_or_else(|| peer.ip().to_string());
    match limiter.check(&key) {
        Ok(()) => next.run(request).await,
        Err(retry_after) => {
            let mut resp = error_response(
                StatusCode::TOO_MANY_REQUESTS,
                "Rate limit exceeded. Please slow down.",
                "rate_limit_exceeded",
            );
            if let Ok(v) = header::HeaderValue::from_str(&retry_after.as_secs().max(1).to_string()) {
                resp.headers_mut().insert(header::RETRY_AFTER, v);
            }
            resp
        }
    }
}
