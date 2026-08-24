//! tuili — a CUDA inference server for GGUF models, with an OpenAI-compatible API.

use std::net::SocketAddr;

use anyhow::{Context, Result};
use clap::Parser;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tuili_server::{engine, routes};

#[derive(Parser, Debug)]
#[command(name = "tuili", about, version)]
struct Args {
    /// Path to a GGUF model file.
    #[arg(short, long)]
    model: String,

    /// Address to listen on.
    #[arg(long, default_value = "127.0.0.1:8080")]
    host: String,

    /// Maximum sequence length, capped by the model's own context length.
    #[arg(long, default_value_t = 4096)]
    ctx: usize,

    /// CUDA device ordinal.
    #[arg(long, default_value_t = 0)]
    device: usize,

    /// KV cache encoding: `f16`, or TurboQuant at `tq4` / `tq2`.
    #[arg(long, default_value = "f16")]
    kv_quant: String,

    /// Blocks to keep in VRAM. The rest are streamed from pinned host memory.
    /// Omit to keep the whole model resident.
    #[arg(long, short = 'n', value_name = "N")]
    gpu_layers: Option<usize>,

    /// Sequences that may generate concurrently.
    ///
    /// This is the batch the tensor-core GEMM sees, and it is worth a factor
    /// of two: the load generator at 32 clients measures 368 tok/s at eight
    /// and 725 at thirty-two, because throughput stops scaling the moment the
    /// scheduler runs out of sequences to batch.
    #[arg(long, default_value_t = 32)]
    max_seqs: usize,

    /// Token slots in the shared KV pool. Defaults to whatever fits in the
    /// VRAM left after the weights, capped at `max-seqs * ctx`; setting it
    /// lower oversubscribes further, which is fine when requests are shorter
    /// than the context limit.
    #[arg(long)]
    kv_slots: Option<usize>,

    /// Patch budget for one image, when the checkpoint has a vision tower.
    /// 4096 is a 1024x1024 image at patch 16 (~350 MiB of scratch); a single
    /// request is separately refused if it resizes to more language-model
    /// tokens than one prefill step can carry, regardless of this number.
    #[arg(long, default_value_t = 4096)]
    vision_max_patches: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,tower_http=warn".into()),
        )
        .with_target(false)
        .init();

    let kv_quant = tuili_model::KvCacheQuant::parse(&args.kv_quant)?;
    let engine = engine::Engine::start(
        &args.model,
        args.ctx,
        args.device,
        kv_quant,
        args.gpu_layers.unwrap_or(usize::MAX),
        args.max_seqs,
        args.kv_slots,
        args.vision_max_patches,
    )
    .context("starting the inference engine")?;

    let app = routes::router(engine.clone())
        .layer(TraceLayer::new_for_http())
        // Permissive CORS so browser playgrounds can talk to a local server.
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );

    let addr: SocketAddr = args
        .host
        .parse()
        .with_context(|| format!("parsing listen address {:?}", args.host))?;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;

    tracing::info!(
        model = %engine.info.id,
        quant = %engine.info.quant,
        kv_quant = %engine.info.kv_quant,
        kv_mib = engine.info.kv_cache_mib,
        vram_mib = engine.info.vram_mib,
        offloaded_mib = engine.info.offloaded_mib,
        max_seqs = engine.info.max_seqs,
        kv_slots = engine.info.kv_slots,
        max_seq = engine.info.max_seq,
        "listening on http://{addr}"
    );
    println!();
    println!("  curl http://{addr}/v1/chat/completions \\");
    println!("    -H 'Content-Type: application/json' \\");
    println!(
        "    -d '{{\"model\":\"{}\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],\"stream\":true}}'",
        engine.info.id
    );
    println!();

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("serving")?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("installing the ctrl-c handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("installing the SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    tracing::info!("shutting down");
}
