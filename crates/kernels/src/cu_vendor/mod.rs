//! Vendor-sourced native code this crate links against but doesn't itself
//! compile as Rust: `flash_attn2_shim.cu` (AOT-compiled by `build.rs`, its
//! `extern "C"` declarations live directly in `flash_attn2.rs`, not routed
//! through this module) and `nccl_shim`, which IS a real Rust module since
//! NCCL ships a prebuilt `.so` with no compilation step of our own.

#[cfg(feature = "nccl")]
pub mod nccl_shim;
