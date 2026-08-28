//! `batch_tokens_for` is pure, so this needs neither a GPU nor a model: it
//! locks in the fix for the bug where a chunk shrank in lockstep with `--ctx`
//! even on models whose attention kernel never touches the buffer that
//! shrinking was protecting, and the follow-up bug that fix exposed: nothing
//! stopped the resulting chunk from walking past the FP8 tensor-core GEMM's
//! own width limit into its much slower fallback. See `needs_score_buffer`
//! in `src/lib.rs` and `infero_kernels::fp8::MMA_MAX_TOKENS_FP8`.

use infero_model::batch_tokens_for;

/// Qwen3.8-27B-FP8's shape: 24 heads, 4 KV heads, `d_head` 256.
const N_HEADS: usize = 24;

#[test]
fn ctx_still_throttles_the_chunk_when_the_score_buffer_is_needed() {
    let at_8k = batch_tokens_for(N_HEADS, 8192, 4, true, None);
    let at_128k = batch_tokens_for(N_HEADS, 131072, 4, true, None);
    assert!(
        at_128k < at_8k,
        "a 16x larger context should still shrink the chunk on the \
         unfused path: got {at_8k} at 8k, {at_128k} at 128k"
    );
    // Exact regression values: this is the arithmetic a real run logged
    // (`batch_tokens=170` at ctx 65536, `85` at ctx 131072) before the fix.
    assert_eq!(batch_tokens_for(N_HEADS, 65536, 16, true, None), 170);
    assert_eq!(batch_tokens_for(N_HEADS, 131072, 16, true, None), 85);
}

#[test]
fn ctx_does_not_throttle_the_chunk_once_a_fused_kernel_makes_it_unnecessary() {
    let at_8k = batch_tokens_for(N_HEADS, 8192, 4, false, None);
    let at_128k = batch_tokens_for(N_HEADS, 131072, 4, false, None);
    assert_eq!(
        at_8k, at_128k,
        "a model whose attention never materializes the score buffer must \
         get the same chunk size regardless of --ctx: got {at_8k} at 8k, \
         {at_128k} at 128k"
    );
}

#[test]
fn an_fp8_model_never_exceeds_the_gemm_width_even_with_room_to_spare() {
    // The bug this locks in: freeing the score-buffer throttle (above) and a
    // generous `--max-seqs` (raising `batch_ceiling`, see `src/lib.rs`) can
    // each independently push the chunk past 256 -- measured live, this
    // combination reached 1024 and fell to the FP8 GEMM's slow fallback for
    // every prefill step, 278s against a 54s baseline for the same prompt.
    let fp8 = Some(infero_kernels::fp8::MMA_MAX_TOKENS_FP8);
    let no_score_buffer_high_concurrency = batch_tokens_for(N_HEADS, 131072, 16, false, fp8);
    assert_eq!(no_score_buffer_high_concurrency, 256);

    let needs_scores_low_ctx = batch_tokens_for(N_HEADS, 1024, 64, true, fp8);
    assert_eq!(needs_scores_low_ctx, 256);
}

#[test]
fn a_non_fp8_model_is_not_capped_by_the_fp8_gemm_width() {
    let chunk = batch_tokens_for(N_HEADS, 131072, 16, false, None);
    assert_eq!(chunk, 1024, "no fp8_ceiling means no fp8-specific cap");
}

#[test]
fn the_env_override_still_wins_over_every_ceiling() {
    // Safety: this test's own process is the only thing touching this var,
    // and no other test in this binary reads or writes it.
    unsafe { std::env::set_var("INFERO_BATCH_TOKENS", "37") };
    let fp8 = Some(infero_kernels::fp8::MMA_MAX_TOKENS_FP8);
    let forced_needed = batch_tokens_for(N_HEADS, 131072, 4, true, None);
    let forced_unneeded = batch_tokens_for(N_HEADS, 131072, 4, false, None);
    let forced_fp8 = batch_tokens_for(N_HEADS, 131072, 4, false, fp8);
    unsafe { std::env::remove_var("INFERO_BATCH_TOKENS") };
    assert_eq!(forced_needed, 37);
    assert_eq!(forced_unneeded, 37);
    assert_eq!(forced_fp8, 37);
}
