//! `MPSMatrixMultiplication`'s fixed cost a call, isolated from FLOPs -- and
//! whether fusing gate+up into one wider call would help.
//!
//! `gemm.rs`'s own doc says splitting the open compute encoder for an MPS
//! pass "is not free." A 53-token prefill issues ~496 of these -- one a
//! weight tensor -- so if that fixed cost is a meaningful fraction of a call,
//! reducing the call count (fusing gate+up into one tensor, the way `w_qkv`
//! and `w_gate_up` already do for the AWQ/Q4G128 path) is worth doing for
//! Q4_K too. Measured by comparing a real FFN-sized call against a tiny one
//! at the same token count, so the tiny one's cost is close to pure overhead:
//! ~0.15ms against ~1.5-1.6ms for the real shape at 20-53 tokens, so the
//! fixed cost is real but only ~10% of a call -- not where the rest of the
//! time goes.
//!
//! What the token-count sweep actually shows: GFLOPS climbs hard with M --
//! 15% of this GPU's peak at 20 tokens, 76% by 512. MPS is not tuned for the
//! short, wide, small-M shape a prefill chunk actually is. Fusing gate and up
//! by doubling N instead of M was worth checking on that basis -- an FFN
//! layer already has two-thirds of its weight bytes in gate/up, so if a
//! bigger matrix were the fix regardless of which dimension grew, this was
//! the cheap way to find out. It made things *worse* -- roughly 2x the cost
//! of the two separate calls it would replace -- so the bottleneck is
//! specifically small M, and N was never undersized to begin with at 17408.
//! `w_gate_up`-style fusion is not the fix for Q4_K.
//!
//!     cargo run --release -p tuili-metal --example gemm_f16_overhead

use anyhow::Result;
use tuili_metal::Device;

fn ms(mut f: impl FnMut() -> Result<()>, iters: usize) -> Result<f64> {
    f()?;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        f()?;
    }
    Ok(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

fn main() -> Result<()> {
    let dev = Device::new(0)?;
    let s = dev.stream();

    for &(label, k, n, n_tokens) in &[
        ("tiny 32x32, 20tok", 32usize, 32usize, 20usize),
        ("tiny 32x32, 53tok", 32, 32, 53),
        ("ffn_gate/up shape, 20tok", 5120, 17408, 20),
        ("ffn_gate/up shape, 53tok", 5120, 17408, 53),
        ("ffn_gate/up shape, 128tok", 5120, 17408, 128),
        ("ffn_gate/up shape, 256tok", 5120, 17408, 256),
        ("ffn_gate/up shape, 512tok", 5120, 17408, 512),
        ("gate+up STACKED (2n), 20tok", 5120, 17408 * 2, 20),
        ("gate+up STACKED (2n), 53tok", 5120, 17408 * 2, 53),
    ] {
        let a = s.alloc_zeros::<half::f16>(n_tokens * k)?;
        let b = s.alloc_zeros::<half::f16>(n * k)?;
        let mut c = s.alloc_zeros::<f32>(n_tokens * n)?;
        let t = ms(
            || {
                tuili_metal::backend::gemm_f16_to_f32(&dev, &mut c.as_view_mut(), &a.as_view(), &b.as_view(), n_tokens, k, n)?;
                s.synchronize()
            },
            20,
        )?;
        println!("{label:28} k={k:6} n={n:6}  {t:7.3}ms");
    }
    Ok(())
}
