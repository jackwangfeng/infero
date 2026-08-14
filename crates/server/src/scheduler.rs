//! Continuous batching.
//!
//! The engine runs one step at a time. Each step assembles a batch from every
//! sequence in flight and hands it to the model as a single forward pass; a
//! sequence that finishes leaves at the end of that step and a waiting request
//! takes its place at the start of the next, without anything else pausing.
//!
//! Two rules shape the batch:
//!
//! * **Decodes go first.** They cost one token each and a running sequence
//!   waiting on the token budget is a stall the client sees directly.
//! * **Prefill fills what is left, and may be split.** A long prompt is fed
//!   across several steps rather than blocking decodes for the whole of it,
//!   which is what keeps one 4000-token request from freezing everyone else.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use tokio::sync::mpsc;
use tuili_model::{BatchItem, KvPool, MAX_BATCH_TOKENS, Model, Sampler, SeqId};
use tuili_tokenizer::Tokenizer;

use crate::engine::{Event, FinishReason, Request};
use crate::stop::split_at_stop;

/// A sequence the scheduler is currently generating.
struct Running {
    seq: SeqId,
    prompt: Vec<u32>,
    /// Prompt tokens already fed to the model.
    prefilled: usize,
    /// The token to feed next, once the prompt is in.
    next: Option<u32>,
    generated: Vec<u32>,
    sampler: Sampler,
    /// Bytes held back because they do not yet form a character.
    partial: Vec<u8>,
    /// Text held back because it might begin a stop sequence.
    pending: String,
    stop: Vec<String>,
    budget: usize,
    events: mpsc::UnboundedSender<Event>,
    admitted: Instant,
    prefill_done: Option<Instant>,
    /// This step's sampled token, filled by the parallel pass below.
    sampled: u32,
}

impl Running {
    fn prompt_complete(&self) -> bool {
        self.prefilled >= self.prompt.len()
    }
}

/// What a sequence is doing this step.
enum Work {
    /// Feed this slice of the prompt; `final_chunk` gets logits.
    Prefill { from: usize, len: usize, last: bool },
    /// Feed the token this sequence sampled last step.
    Decode,
}

pub struct Scheduler {
    model: Model,
    pool: KvPool,
    tokenizer: Arc<Tokenizer>,
    waiting: VecDeque<Request>,
    running: Vec<Running>,
    steps: u64,
    profile: bool,
    /// Time to *issue* the forward pass, not to run it. Since the device
    /// sampler took over, nothing between the launches and the sample waits on
    /// the GPU, so this reads near zero and the wait shows up in `t_sample`.
    /// It was `forward_ms` when the step ended in a synchronise, and reading
    /// the new number under the old name is how you conclude the forward pass
    /// got sixty times faster.
    t_issue: f64,
    /// Sampling, which is where the step now blocks on the whole forward pass.
    t_sample: f64,
    /// Per-sequence bookkeeping: detokenizing, stop scanning, delivery.
    t_advance: f64,
    /// Whole `step()`, and the wall clock between one step ending and the next
    /// beginning. A model that gets faster while the served rate does not is
    /// either losing the gain in the bookkeeping around the step or in the gap
    /// between steps, and only measuring both tells you which.
    t_step: f64,
    t_gap: f64,
    last_end: Option<std::time::Instant>,
    window: u64,
}

impl Scheduler {
    pub fn new(model: Model, pool: KvPool, tokenizer: Arc<Tokenizer>) -> Self {
        Self {
            model,
            pool,
            tokenizer,
            waiting: VecDeque::new(),
            running: Vec::new(),
            steps: 0,
            // A separate switch from `TUILI_PROFILE`: that one turns on
            // per-kernel CUDA events, which cannot coexist with graph capture,
            // so it can only ever report an ungraphed step. This one is pure
            // host-side wall clock and leaves the graphs alone.
            profile: std::env::var_os("TUILI_PROFILE").is_some()
                || std::env::var_os("TUILI_STEP_TIMING").is_some(),
            t_issue: 0.0,
            t_sample: 0.0,
            t_advance: 0.0,
            t_step: 0.0,
            t_gap: 0.0,
            last_end: None,
            window: 0,
        }
    }

    pub fn enqueue(&mut self, req: Request) {
        self.waiting.push_back(req);
    }

    pub fn is_idle(&self) -> bool {
        self.running.is_empty() && self.waiting.is_empty()
    }

    pub fn in_flight(&self) -> usize {
        self.running.len() + self.waiting.len()
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    pub fn pool(&self) -> &KvPool {
        &self.pool
    }

    /// Where a step's wall time went, under `TUILI_PROFILE` or
    /// `TUILI_STEP_TIMING`.
    ///
    /// The GPU profile answers "which kernel", which stops being the question
    /// once the server is slower than the model. It is not, any more: at a
    /// batch of 32 the host side is 0.18 ms of a 7.36 ms step.
    pub fn timing(&self) -> (f64, f64, f64, u64) {
        (self.t_issue, self.t_sample, self.t_advance, self.steps)
    }

    /// Admit what fits, run one batched forward, and deliver its output.
    pub fn step(&mut self) -> Result<()> {
        let step_start = self.profile.then(std::time::Instant::now);
        self.admit();
        self.drop_disconnected();
        if self.running.is_empty() {
            return Ok(());
        }

        let plan = self.plan();
        if plan.is_empty() {
            // Every running sequence is blocked on pool capacity; releasing a
            // finished one is the only thing that can unblock them, and that
            // happens below.
            anyhow::bail!("no sequence could make progress: kv pool is exhausted");
        }

        let items: Vec<BatchItem<'_>> = plan
            .iter()
            .map(|(idx, work)| {
                let r = &self.running[*idx];
                match work {
                    Work::Prefill { from, len, last } => BatchItem {
                        seq: r.seq,
                        tokens: &r.prompt[*from..*from + *len],
                        wants_logits: *last,
                    },
                    Work::Decode => BatchItem {
                        seq: r.seq,
                        // Safe: `next` is set before a sequence can be
                        // scheduled for decode.
                        tokens: std::slice::from_ref(r.next.as_ref().unwrap()),
                        wants_logits: true,
                    },
                }
            })
            .collect();

        let vocab = self.model.config().vocab_size;
        let timed = self.profile;
        let t0 = timed.then(std::time::Instant::now);
        self.model.forward_batch_device(&items, &mut self.pool)?;
        let t1 = timed.then(std::time::Instant::now);

        // Sample where the logits already are.
        //
        // The alternative — and what this did until the device sampler existed
        // — is to copy every row's whole vocabulary back first: 16 MiB at a
        // batch of 32, measured at 2.19 ms of a 12.18 ms step. `t_sample` below
        // covers this call, and since the forward pass no longer ends in a
        // synchronise it is also where the step waits on the GPU.
        let rows: Vec<usize> = plan
            .iter()
            .filter(|(_, w)| match w {
                Work::Prefill { last, .. } => *last,
                Work::Decode => true,
            })
            .map(|(idx, _)| *idx)
            .collect();
        // The draw comes from each sequence's own generator, so seeding and
        // reproducibility survive the move; pulling it needs `running` mutably,
        // which has to finish before the windows borrow it.
        let draws: Vec<f64> = rows
            .iter()
            .map(|i| self.running[*i].sampler.next_draw())
            .collect();
        let specs: Vec<tuili_model::RowSample> = rows
            .iter()
            .zip(&draws)
            .map(|(i, d)| {
                let p = self.running[*i].sampler.params();
                tuili_model::RowSample {
                    temperature: p.temperature,
                    top_p: p.top_p,
                    top_k: p.top_k as u32,
                    rep_penalty: p.repetition_penalty,
                    rnd: *d,
                }
            })
            .collect();
        let windows: Vec<&[u32]> = rows
            .iter()
            .map(|i| {
                let r = &self.running[*i];
                r.sampler.window(&r.generated)
            })
            .collect();
        let sampled = self.model.sample_on_device(&specs, &windows)?;
        let t2 = timed.then(std::time::Instant::now);

        match sampled {
            Some(ids) => {
                for (i, id) in rows.iter().zip(ids) {
                    self.running[*i].sampled = id;
                }
            }
            None => {
                // Whatever the device sampler will not take — a `top_k` past
                // its bound, or a vocabulary too large for the bitset — still
                // has to produce a token, and it does so exactly as before.
                let logits = self.model.logits_host()?.to_vec();
                self.sample_rows_host(&plan, &logits, vocab, &rows, &draws);
            }
        }
        self.steps += 1;

        // The sampled token already sits on each `Running`; this pass is the
        // bookkeeping around it. It no longer counts logit rows — those are
        // matched to sequences by `rows` above, where the sampling happens.
        let mut finished = Vec::new();
        for (idx, work) in &plan {
            let wants = match work {
                Work::Prefill { last, .. } => *last,
                Work::Decode => true,
            };
            if let Work::Prefill { len, .. } = work {
                self.running[*idx].prefilled += len;
            }
            if !wants {
                continue;
            }
            if self.advance(*idx)? {
                finished.push(*idx);
            }
        }

        // Decode steps only. A prefill step costs about ten times a decode,
        // and at 32 clients the run's own 32 prefills land inside the first
        // window and drag its mean from 11 ms to 18 — which reads as six
        // milliseconds of forward-pass overhead that does not exist. The
        // window was meant to keep prefills out of a lifetime average; it has
        // to keep them out of the window too.
        let decode_only = plan.iter().all(|(_, w)| matches!(w, Work::Decode));
        if let (Some(t0), Some(t1), Some(t2)) = (t0, t1, t2) {
            let t3 = std::time::Instant::now();
            // `last_end` advances on every step, prefill included, or the next
            // decode's gap would swallow the prefill step whole and report it
            // as time the engine spent doing nothing.
            let prev = self.last_end.replace(t3);
            if decode_only {
                self.t_issue += (t1 - t0).as_secs_f64() * 1e3;
                self.t_sample += (t2 - t1).as_secs_f64() * 1e3;
                self.t_advance += (t3 - t2).as_secs_f64() * 1e3;
                if let (Some(s0), Some(prev)) = (step_start, prev) {
                    self.t_step += (t3 - s0).as_secs_f64() * 1e3;
                    self.t_gap += (s0 - prev).as_secs_f64() * 1e3;
                }
                self.window += 1;
                // A window rather than a lifetime average: the run's own prefill
                // steps cost ten times a decode and would sit in the mean forever.
                if self.window >= 200 {
                    let w = self.window as f64;
                    tracing::warn!(
                        issue_ms = format!("{:.2}", self.t_issue / w),
                        sample_ms = format!("{:.2}", self.t_sample / w),
                        advance_ms = format!("{:.2}", self.t_advance / w),
                        step_ms = format!("{:.2}", self.t_step / w),
                        gap_ms = format!("{:.2}", self.t_gap / w),
                        batch = plan.len(),
                        "per-step timing"
                    );
                    self.t_issue = 0.0;
                    self.t_sample = 0.0;
                    self.t_advance = 0.0;
                    self.t_step = 0.0;
                    self.t_gap = 0.0;
                    self.window = 0;
                }
            }
        }

        // Retire from the back so the earlier indices stay valid.
        finished.sort_unstable();
        for idx in finished.into_iter().rev() {
            let r = self.running.swap_remove(idx);
            self.pool.free(r.seq);
        }
        Ok(())
    }

    /// Move waiting requests into the running set while there is room.
    fn admit(&mut self) {
        while let Some(req) = self.waiting.front() {
            if self.pool.active_sequences() >= self.pool.max_seqs() {
                break;
            }
            // Admit only with room for the whole prompt plus something to
            // generate; a sequence admitted into a pool it cannot finish in
            // would deadlock against itself.
            let need = req.prompt.len() + 1;
            if need > self.pool.free_slots() || need > self.pool.max_seq() {
                if self.running.is_empty() {
                    // Nothing will ever free up: reject rather than spin.
                    let req = self.waiting.pop_front().unwrap();
                    let _ = req.events.send(Event::Failed(format!(
                        "prompt of {} tokens does not fit: {} slots free, {} per sequence",
                        req.prompt.len(),
                        self.pool.free_slots(),
                        self.pool.max_seq()
                    )));
                    continue;
                }
                break;
            }

            let Some(seq) = self.pool.alloc() else { break };
            let req = self.waiting.pop_front().unwrap();
            let budget = req
                .max_tokens
                .min(self.pool.max_seq().saturating_sub(req.prompt.len()));
            tracing::debug!(
                seq = seq.0,
                prompt = req.prompt.len(),
                running = self.running.len() + 1,
                "admitted"
            );
            self.running.push(Running {
                seq,
                prompt: req.prompt,
                prefilled: 0,
                next: None,
                sampled: 0,
                generated: Vec::new(),
                sampler: Sampler::new(req.params),
                partial: Vec::new(),
                pending: String::new(),
                stop: req.stop,
                budget,
                events: req.events,
                admitted: Instant::now(),
                prefill_done: None,
            });
        }
    }

    /// Drop sequences whose client has gone away, freeing their slots at once.
    fn drop_disconnected(&mut self) {
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].events.is_closed() {
                let r = self.running.swap_remove(i);
                tracing::debug!(seq = r.seq.0, "client disconnected");
                self.pool.free(r.seq);
            } else {
                i += 1;
            }
        }
    }

    /// Decide who gets tokens this step.
    fn plan(&self) -> Vec<(usize, Work)> {
        let mut plan = Vec::with_capacity(self.running.len());
        let mut budget = MAX_BATCH_TOKENS;

        // Decodes first: one token each, and a running sequence starved by a
        // prefill is a stall the client feels.
        for (i, r) in self.running.iter().enumerate() {
            if budget == 0 {
                break;
            }
            if r.prompt_complete() && r.next.is_some() {
                plan.push((i, Work::Decode));
                budget -= 1;
            }
        }

        // Whatever is left goes to prompts, split across steps if need be.
        for (i, r) in self.running.iter().enumerate() {
            if budget == 0 {
                break;
            }
            if r.prompt_complete() {
                continue;
            }
            let remaining = r.prompt.len() - r.prefilled;
            let len = remaining.min(budget);
            plan.push((
                i,
                Work::Prefill {
                    from: r.prefilled,
                    len,
                    last: len == remaining,
                },
            ));
            budget -= len;
        }
        plan
    }

    /// Sample, emit, and report whether the sequence is done.
    fn advance(&mut self, idx: usize) -> Result<bool> {
        let tokenizer = self.tokenizer.clone();
        let r = &mut self.running[idx];
        if r.prefill_done.is_none() {
            r.prefill_done = Some(Instant::now());
        }

        let next = r.sampled;
        if tokenizer.is_eog(next) {
            return Self::finish(r, &tokenizer, FinishReason::Stop);
        }

        r.generated.push(next);
        r.next = Some(next);

        let text = tokenizer.stream_push(next, &mut r.partial);
        r.pending.push_str(&text);
        let scan = split_at_stop(&r.pending, &r.stop);
        let emit: String = r.pending.drain(..scan.release_len()).collect();
        if !emit.is_empty() && r.events.send(Event::Text(emit)).is_err() {
            return Ok(true); // client hung up
        }
        if scan.is_hit() {
            r.pending.clear();
            return Self::finish(r, &tokenizer, FinishReason::Stop);
        }

        if r.generated.len() >= r.budget {
            return Self::finish(r, &tokenizer, FinishReason::Length);
        }
        // A sequence that has filled its share of the pool stops here rather
        // than failing the whole batch on the next extend.
        if self.pool.headroom(self.running[idx].seq) == 0 || self.pool.free_slots() == 0 {
            let r = &mut self.running[idx];
            return Self::finish(r, &tokenizer, FinishReason::Length);
        }
        Ok(false)
    }

    /// Fill `Running::sampled` for every row that asked for logits.
    ///
    /// Each sequence owns its sampler and its own history, so the work is
    /// disjoint; the only reason it was serial is that it lived inside the loop
    /// that also does the stop-sequence and streaming bookkeeping, which is
    /// not.
    /// The host fallback, for batches the device sampler declines.
    ///
    /// `draws` were already taken from each sequence's generator before the
    /// device path was tried, so this uses them rather than drawing again —
    /// otherwise falling back would advance the generators twice and a seeded
    /// run would depend on which path served it.
    fn sample_rows_host(
        &mut self,
        _plan: &[(usize, Work)],
        logits: &[f32],
        vocab: usize,
        wanted: &[usize],
        draws: &[f64],
    ) {
        if wanted.is_empty() {
            return;
        }

        // Row number for each running index, so the mutable borrows can be
        // collected in `running` order and still find their logits.
        let mut row_of = vec![usize::MAX; self.running.len()];
        for (row, idx) in wanted.iter().enumerate() {
            row_of[*idx] = row;
        }
        let mut refs: Vec<(usize, &mut Running)> = self
            .running
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| row_of[*i] != usize::MAX)
            .map(|(i, r)| (row_of[i], r))
            .collect();

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(refs.len());
        if threads <= 1 {
            for (row, r) in refs.iter_mut() {
                r.sampled = r.sampler.sample_with_draw(
                    &logits[*row * vocab..(*row + 1) * vocab],
                    &r.generated,
                    draws[*row],
                );
            }
            return;
        }
        let per = refs.len().div_ceil(threads);
        std::thread::scope(|scope| {
            for chunk in refs.chunks_mut(per) {
                scope.spawn(move || {
                    for (row, r) in chunk {
                        let lo = *row * vocab;
                        r.sampled = r.sampler.sample_with_draw(
                            &logits[lo..lo + vocab],
                            &r.generated,
                            draws[*row],
                        );
                    }
                });
            }
        });
    }

    fn finish(r: &mut Running, tokenizer: &Tokenizer, reason: FinishReason) -> Result<bool> {
        let mut tail = std::mem::take(&mut r.pending);
        tail.push_str(&tokenizer.stream_finish(&mut r.partial));
        if !tail.is_empty() {
            let _ = r.events.send(Event::Text(tail));
        }

        let queued = r.prefill_done.unwrap_or(r.admitted) - r.admitted;
        tracing::info!(
            seq = r.seq.0,
            prompt = r.prompt.len(),
            completion = r.generated.len(),
            queued_ms = queued.as_millis(),
            total_ms = r.admitted.elapsed().as_millis(),
            reason = reason.as_str(),
            "request complete"
        );

        let _ = r.events.send(Event::Done {
            reason,
            prompt_tokens: r.prompt.len(),
            completion_tokens: r.generated.len(),
        });
        Ok(true)
    }

    /// Report an error to every sequence and clear the scheduler.
    ///
    /// A failed forward pass leaves the pool in an unknown state, so the whole
    /// batch goes rather than a guess at which sequence caused it.
    pub fn fail_all(&mut self, error: &str) {
        for r in self.running.drain(..) {
            let _ = r.events.send(Event::Failed(error.to_string()));
            self.pool.free(r.seq);
        }
        for req in self.waiting.drain(..) {
            let _ = req.events.send(Event::Failed(error.to_string()));
        }
    }

    pub fn model(&self) -> &Model {
        &self.model
    }
}

/// Build the pool a scheduler needs.
pub fn make_pool(model: &Model, max_seqs: usize, slots: Option<usize>) -> Result<KvPool> {
    let max_seq = model.max_seq();
    if let Some(n) = slots {
        return model
            .new_pool(n.max(max_seq), max_seqs)
            .context("allocating the kv pool");
    }

    // `max_seqs * max_seq` is what a pool would need for every sequence to
    // reach the context limit at once, and at 32 sequences of 4096 tokens on
    // an 8B model that is 17 GB — so a server asked for more concurrency than
    // its VRAM can back used to refuse to start, and the default concurrency
    // was set low enough to avoid it. That default cost a factor of two: the
    // load generator at 32 clients measured 368 tok/s against 725 with
    // `--max-seqs 32`, because eight was the batch the GEMM ever saw.
    //
    // Sequences rarely all run to the context limit, so oversubscribe: take
    // what the card has left after the weights, keep a margin for activations
    // and the CUDA context, and let the scheduler admit against the real slot
    // count. It already refuses a prompt that will not fit.
    let (free, _) = model.device().mem_info().context("querying free vram")?;
    let want = max_seqs.saturating_mul(max_seq).max(max_seq);
    let mut lo = max_seq;
    let mut hi = want;
    // The pool's byte count is not a simple product — TurboQuant carries
    // per-slot tables — so find the largest slot count that fits by bisection
    // on the pool's own accounting rather than by re-deriving it here.
    let budget = free.saturating_sub(free / 8).saturating_sub(512 << 20);
    let fits = |n: usize| -> bool {
        match model.new_pool(n, max_seqs) {
            Ok(p) => p.bytes() <= budget,
            Err(_) => false,
        }
    };
    if fits(hi) {
        return model
            .new_pool(hi, max_seqs)
            .context("allocating the kv pool");
    }
    while lo + max_seq < hi {
        let mid = (lo + hi) / 2;
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    model
        .new_pool(lo, max_seqs)
        .context("allocating the kv pool")
}
