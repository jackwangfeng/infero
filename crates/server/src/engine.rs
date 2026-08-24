//! The inference worker.
//!
//! One GPU, one model, one thread that owns it. Requests arrive on a channel
//! and stream tokens back on a per-request channel; the HTTP layer never
//! touches CUDA. Dropping the receiver cancels the generation at the next
//! token, which is what makes a client disconnect free the GPU immediately.
//!
//! This is deliberately one-request-at-a-time. Continuous batching belongs
//! here — the queue is already the right shape for it — but a scheduler that
//! interleaves sequences needs a paged KV cache to be worth writing.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use tokio::sync::mpsc;
use tuili_cuda::Device;
use tuili_gguf::Gguf;
use tuili_model::{KvCacheQuant, Model, SamplingParams};

use crate::scheduler::{Scheduler, make_pool};
use tuili_tokenizer::Tokenizer;

/// What a client asked the model to do.
pub struct Request {
    pub prompt: Vec<u32>,
    pub params: SamplingParams,
    pub max_tokens: usize,
    /// Text sequences that end the generation, excluded from the output.
    pub stop: Vec<String>,
    pub events: mpsc::UnboundedSender<Event>,
}

/// Why generation stopped, in OpenAI's vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        }
    }
}

pub enum Event {
    /// A decoded fragment. May be empty when a token completes only part of a
    /// multi-byte character.
    Text(String),
    Done {
        reason: FinishReason,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Failed(String),
}

/// Handle to the worker thread.
pub struct Engine {
    jobs: mpsc::UnboundedSender<Request>,
    pub info: ModelInfo,
    tokenizer: Arc<Tokenizer>,
    in_flight: Arc<AtomicU64>,
    served: Arc<AtomicU64>,
    /// `None` when the model carries recurrent state and prefix caching is off;
    /// otherwise the cumulative (lookups, hits, tokens served from cache) as of
    /// the last completed step.
    prefix_stats: Option<[Arc<AtomicU64>; 3]>,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub path: String,
    pub quant: String,
    pub context_length: usize,
    pub max_seq: usize,
    pub weights_mib: usize,
    pub kv_quant: String,
    /// Device memory one sequence's KV cache occupies at `max_seq`.
    pub kv_cache_mib: usize,
    pub kv_bits_per_channel: f32,
    pub vram_mib: usize,
    pub offloaded_mib: usize,
    pub offloaded_layers: usize,
    /// Sequences that may run concurrently.
    pub max_seqs: usize,
    /// Token slots the pool shares between them.
    pub kv_slots: usize,
}

impl Engine {
    /// Load the model and start the worker.
    pub fn start(
        path: &str,
        max_seq: usize,
        device_ordinal: usize,
        kv_quant: KvCacheQuant,
        n_gpu_layers: usize,
        max_seqs: usize,
        kv_slots: Option<usize>,
    ) -> Result<Arc<Self>> {
        // A directory is a Hugging Face checkpoint, a file is a GGUF.
        let awq = std::path::Path::new(path).is_dir();
        let gguf = if awq {
            None
        } else {
            Some(Gguf::open(path).with_context(|| format!("opening {path}"))?)
        };
        let tokenizer = Arc::new(match &gguf {
            Some(f) => Tokenizer::from_gguf(f)?,
            None => Tokenizer::from_hf_dir(path)?,
        });
        let dev = Device::new(device_ordinal)?;
        // The logits buffer is sized from `max_seqs`, not from a constant:
        // every admitted sequence takes a logit row on the same pass, so a
        // fixed ceiling turns `--max-seqs 64` into a server that starts and
        // then 500s on the 33rd concurrent request.
        let model = match &gguf {
            Some(f) => Model::load_full(dev, f, max_seq, kv_quant, n_gpu_layers, max_seqs)?,
            None => {
                anyhow::ensure!(
                    n_gpu_layers == usize::MAX,
                    "an AWQ checkpoint has no offload path; drop --gpu-layers"
                );
                Model::load_awq(dev, path, max_seq, kv_quant, max_seqs)?
            }
        };
        let pool = make_pool(&model, max_seqs, kv_slots)?;
        let kv_cache_bytes = pool.bytes();
        let pool_slots = pool.n_slots();

        let cfg = model.config().clone();
        let info = ModelInfo {
            id: derive_model_id(path, &cfg.name),
            path: path.to_string(),
            // The GGUF file states its own dominant type; a safetensors
            // checkpoint does not, so ask the loaded weights. The fallback used
            // to be the literal "AWQ-INT4", which was true of every
            // safetensors model tuili could load and is now a lie: an FP8
            // checkpoint reported itself as INT4 while running FP8 kernels.
            // Nothing computed from this — the dispatch reads `WeightType` —
            // but a status line that names the wrong encoding is worse than one
            // that says nothing.
            quant: gguf
                .as_ref()
                .and_then(|f| f.dominant_type().map(|t| t.to_string()))
                .unwrap_or_else(|| model.dominant_weight_type().to_string()),
            context_length: cfg.context_length,
            max_seq: model.max_seq(),
            weights_mib: gguf
                .as_ref()
                .map_or_else(|| model.weight_bytes().0, |f| f.data_len())
                / (1 << 20),
            kv_quant: kv_quant.to_string(),
            kv_cache_mib: kv_cache_bytes / (1 << 20),
            kv_bits_per_channel: kv_quant.bits_per_channel(cfg.d_head),
            vram_mib: model.weight_bytes().0 / (1 << 20),
            offloaded_mib: model.weight_bytes().1 / (1 << 20),
            offloaded_layers: model.n_offloaded_layers(),
            max_seqs,
            kv_slots: pool_slots,
        };

        let (jobs, rx) = mpsc::unbounded_channel();
        let in_flight = Arc::new(AtomicU64::new(0));
        let served = Arc::new(AtomicU64::new(0));

        let mut scheduler = Scheduler::new(model, pool, tokenizer.clone());
        // Speculation, when the checkpoint has a head.
        //
        // On by default at k = 1, which is measured rather than chosen: it is
        // the *fastest* setting, and deeper drafts lose. The 27B, four prompts,
        // 120 tokens each, one request at a time:
        //
        // ```text
        //   k      tok/s   mean acceptance
        //   off     30.2                 —
        //   1       38.5              1.72
        //   2       36.1              2.08
        //   3       34.7              2.46
        // ```
        //
        // Acceptance climbs monotonically while throughput falls, which is the
        // signature of a verification pass that costs more per row than the row
        // buys. Two things charge per drafted token: the pass itself is `k + 1`
        // rows wide and the batched FP8 mat-vec charges about 7.2 ms a row (the
        // sweep is in `fp8.rs`), and each draft step costs 9-10 ms against a
        // memory bound of 3.2 — the head runs a dozen kernels at one row each,
        // so it is launch-bound. A marginal row buys 0.37 tokens; at 28 ms a
        // token it needs to buy 0.6 to pay for itself.
        //
        // Both are fixable and neither is fixed here, so the default is the
        // shallowest draft that still wins. `TUILI_SPEC_K=0` turns it off.
        let spec_k: usize = std::env::var("TUILI_SPEC_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        if spec_k > 0 && std::path::Path::new(path).is_dir() {
            match scheduler.enable_speculation(path, spec_k) {
                Ok(true) => {}
                Ok(false) => {}
                // A checkpoint that has a head but cannot use it is worth
                // saying out loud rather than silently serving without.
                Err(e) => tracing::warn!(error = %e, "speculation stayed off"),
            }
        }

        // Read once, before the scheduler moves into the worker thread: whether
        // this model can cache a prefix at all is fixed at load time, and the
        // handle needs to know that even before the first step reports numbers.
        let prefix_stats = scheduler.prefix_cache_stats().map(|_| {
            [
                Arc::new(AtomicU64::new(0)),
                Arc::new(AtomicU64::new(0)),
                Arc::new(AtomicU64::new(0)),
            ]
        });

        let worker = Worker {
            scheduler,
            in_flight: in_flight.clone(),
            served: served.clone(),
            prefix_stats: prefix_stats.clone(),
        };
        // A dedicated OS thread, not a tokio task: the forward pass blocks on
        // CUDA and would otherwise stall the runtime.
        std::thread::Builder::new()
            .name("tuili-inference".into())
            .spawn(move || worker.run(rx))
            .context("spawning the inference thread")?;

        Ok(Arc::new(Self {
            jobs,
            info,
            tokenizer,
            in_flight,
            served,
            prefix_stats,
        }))
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn queue_depth(&self) -> u64 {
        self.in_flight.load(Ordering::Relaxed)
    }

    pub fn requests_served(&self) -> u64 {
        self.served.load(Ordering::Relaxed)
    }

    /// (lookups, hits, tokens served from cache) as of the last completed
    /// step. `None` on a model whose recurrent state a shared prefix would not
    /// reconstruct — see `crate::prefix`'s module note.
    pub fn prefix_cache_stats(&self) -> Option<(u64, u64, u64)> {
        self.prefix_stats.as_ref().map(|c| {
            (
                c[0].load(Ordering::Relaxed),
                c[1].load(Ordering::Relaxed),
                c[2].load(Ordering::Relaxed),
            )
        })
    }

    /// Submit a request. The returned receiver yields events until `Done` or
    /// `Failed`; dropping it cancels the generation.
    pub fn submit(&self, mut req: Request) -> Result<mpsc::UnboundedReceiver<Event>> {
        let (tx, rx) = mpsc::unbounded_channel();
        req.events = tx;
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        self.jobs
            .send(req)
            .map_err(|_| anyhow::anyhow!("inference worker has stopped"))?;
        Ok(rx)
    }
}

/// Convenience for the non-streaming endpoints: drain a stream into one string.
pub async fn collect(
    mut rx: mpsc::UnboundedReceiver<Event>,
) -> Result<(String, FinishReason, usize, usize)> {
    let mut text = String::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            Event::Text(t) => text.push_str(&t),
            Event::Done {
                reason,
                prompt_tokens,
                completion_tokens,
            } => return Ok((text, reason, prompt_tokens, completion_tokens)),
            Event::Failed(e) => anyhow::bail!(e),
        }
    }
    anyhow::bail!("inference worker dropped the request")
}

/// Drives the scheduler: drains new requests, runs a step, repeats.
struct Worker {
    scheduler: Scheduler,
    in_flight: Arc<AtomicU64>,
    served: Arc<AtomicU64>,
    prefix_stats: Option<[Arc<AtomicU64>; 3]>,
}

impl Worker {
    fn run(mut self, mut jobs: mpsc::UnboundedReceiver<Request>) {
        loop {
            // With nothing in flight there is no reason to spin; block until a
            // request shows up.
            if self.scheduler.is_idle() {
                match jobs.blocking_recv() {
                    Some(req) => self.scheduler.enqueue(req),
                    None => break,
                }
            }
            // Take everything else that is already queued, so a burst of
            // requests joins the same batch rather than trickling in one step
            // at a time.
            while let Ok(req) = jobs.try_recv() {
                self.scheduler.enqueue(req);
            }

            let before = self.scheduler.in_flight();
            if let Err(e) = self.scheduler.step() {
                tracing::error!(error = %format!("{e:#}"), "batch step failed");
                let n = self.scheduler.in_flight() as u64;
                self.scheduler.fail_all(&format!("{e:#}"));
                self.served.fetch_add(n, Ordering::Relaxed);
                self.in_flight.store(0, Ordering::Relaxed);
                continue;
            }
            let after = self.scheduler.in_flight();
            self.in_flight.store(after as u64, Ordering::Relaxed);
            if before > after {
                self.served
                    .fetch_add((before - after) as u64, Ordering::Relaxed);
            }
            // Cumulative counters, so a plain store is right — unlike `served`,
            // which resets are not a concern for the source `stats()` returns.
            if let (Some(cell), Some((lookups, hits, saved))) =
                (&self.prefix_stats, self.scheduler.prefix_cache_stats())
            {
                cell[0].store(lookups, Ordering::Relaxed);
                cell[1].store(hits, Ordering::Relaxed);
                cell[2].store(saved, Ordering::Relaxed);
            }
        }
        tracing::info!(steps = self.scheduler.steps(), "inference worker stopped");
    }
}

fn derive_model_id(path: &str, name: &str) -> String {
    std::path::Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| name.to_string())
}
