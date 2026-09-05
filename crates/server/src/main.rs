//! infero — a CUDA inference server for GGUF models, with an OpenAI-compatible API.

use std::net::SocketAddr;

use anyhow::{Context, Result};
use clap::Parser;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use infero_server::{engine, routes};

#[derive(Parser, Debug)]
#[command(name = "infero", about, version)]
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

    /// CUDA device ordinal. Ignored under tensor parallelism (`--tensor-
    /// parallel-size` > 1) -- `CUDA_VISIBLE_DEVICES` pins each rank's
    /// process to its own physical GPU there instead, matching
    /// `tp_generate.rs`'s own launch convention (one process a rank, one
    /// GPU a process).
    #[arg(long, default_value_t = 0)]
    device: usize,

    /// How many GPUs to shard this model across. `1` (the default) is
    /// today's exact single-GPU behavior. At `> 1`, launch one process a
    /// rank with `TP_RANK`/`RUN_ID` set and `CUDA_VISIBLE_DEVICES` pinned to
    /// a distinct physical GPU each -- see `docs/superpowers/specs/
    /// 2026-09-05-tensor-parallel-design.md`. Vision/video requests,
    /// M-RoPE, and speculative decoding are not supported yet at `> 1` (the
    /// server refuses them, rather than silently mishandling them).
    #[arg(long, default_value_t = 1)]
    tensor_parallel_size: usize,

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

    /// How many frames a video request may be sampled down to. The real
    /// ceiling is `Model::batch_tokens()` (a runtime value that shrinks with
    /// `--ctx`), which a single request is separately refused against
    /// regardless of this number — this is the earlier, cheaper refusal, and
    /// the knob an operator has for "how much of a step's budget one video
    /// may claim" independent of a specific request's resolution.
    #[arg(long, default_value_t = 16)]
    video_max_frames: usize,

    /// How many frames a second a video request is sampled at, when the
    /// request itself does not set `video_url.fps`. A moving subject wants
    /// more than the default 2fps (see `notes/video-encoding-optimizations
    /// .md`, item 6, for the reasoning); a mostly-static scene can get away
    /// with less, saving `--video-max-frames`' budget for a longer clip.
    #[arg(long, default_value_t = infero_server::video::DEFAULT_TARGET_FPS)]
    video_target_fps: f64,

    /// Drop a video frame-group as a near-duplicate of the last retained one
    /// when their mean per-channel pixel difference (0-255 scale) is at or
    /// below this. `0.0` (the default) disables the check entirely -- a
    /// slow-panning or mostly-static clip otherwise spends its whole token
    /// budget on frames that would tell the model nothing new; see
    /// `notes/video-encoding-optimizations.md`, item 5.
    #[arg(long, default_value_t = 0.0)]
    video_dedup_threshold: f64,
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

    let kv_quant = infero_model::KvCacheQuant::parse(&args.kv_quant)?;

    let tp_rank: usize = std::env::var("TP_RANK").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    if args.tensor_parallel_size > 1 && tp_rank != 0 {
        // A follower rank: no HTTP, no `Scheduler` -- just a `Model` and a
        // `KvPool` driven by rank 0's broadcasts. Returns only when the
        // group shuts down (rank 0 exits) or on a real error; never reaches
        // the HTTP-serving code below.
        #[cfg(feature = "nccl")]
        {
            let run_id = std::env::var("RUN_ID")
                .context("TP_RANK is set but RUN_ID is not -- both are required under --tensor-parallel-size > 1")?;
            let gguf = infero_gguf::Gguf::open(&args.model).with_context(|| format!("opening {}", args.model))?;
            let dev = infero_gpu::Device::new(0) // CUDA_VISIBLE_DEVICES remaps this
                .context("opening this rank's GPU")?;
            let rank = infero_model::tp::RankId {
                pp_rank: 0,
                pp_size: 1,
                tp_rank,
                tp_size: args.tensor_parallel_size,
            };
            let model = infero_model::Model::load_full_tp(
                dev,
                &gguf,
                args.ctx,
                kv_quant,
                args.gpu_layers.unwrap_or(usize::MAX),
                args.max_seqs,
                &rank,
                &run_id,
            )
            .context("loading this rank's shard")?;
            let pool = infero_server::scheduler::make_pool(&model, args.max_seqs, args.kv_slots)
                .context("sizing this rank's kv pool")?;
            tracing::info!(tp_rank, tp_size = args.tensor_parallel_size, "follower rank ready");
            infero_server::tp::run_follower(model, pool).context("follower loop")?;
            return Ok(());
        }
        #[cfg(not(feature = "nccl"))]
        anyhow::bail!("--tensor-parallel-size > 1 needs the `nccl` feature; this build doesn't have it");
    }
    let tp = (args.tensor_parallel_size > 1).then(|| {
        let run_id = std::env::var("RUN_ID").unwrap_or_else(|_| "infero_tp_default".to_string());
        (tp_rank, args.tensor_parallel_size, run_id)
    });

    let engine = engine::Engine::start(
        &args.model,
        args.ctx,
        args.device,
        kv_quant,
        args.gpu_layers.unwrap_or(usize::MAX),
        args.max_seqs,
        args.kv_slots,
        args.vision_max_patches,
        args.video_max_frames,
        args.video_target_fps,
        args.video_dedup_threshold,
        tp,
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
