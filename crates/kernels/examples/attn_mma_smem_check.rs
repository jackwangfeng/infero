//! What this device's actual dynamic-shared-memory ceiling is, checked while
//! chasing a memory-corruption bug that only shows up once
//! `attn_decode_mma_f32`'s shared-memory request crosses the 48 KiB static
//! default (true at d_head=256, 74 KiB; never true at d_head<=128, which is
//! why this path never needed the opt-in before).
use anyhow::{Context, Result};
use infero_cuda::Device;

fn main() -> Result<()> {
    let dev = Device::new(0)?;

    let max_optin = dev
        .context()
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )
        .context("querying CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN")?;
    println!("device max shared mem per block (opt-in): {max_optin} bytes");

    let default_static = dev
        .context()
        .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
        .context("querying CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK")?;
    println!("device max shared mem per block (static default): {default_static} bytes");

    let per_sm = dev
        .context()
        .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
        .context("querying CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR")?;
    println!("device max shared mem per SM: {per_sm} bytes");

    // The exact request attn_decode_mma_f32 issues for our shape.
    let d_head = 256usize;
    const T: usize = 64;
    let requested = 16 * (d_head + 8) * 2 + T * (d_head + 8) * 2 + d_head * (T + 2) * 2;
    println!("attn_decode_mma_f32 requests: {requested} bytes at d_head={d_head}");
    println!(
        "fits under opt-in ceiling: {} (headroom {} bytes)",
        requested <= max_optin as usize,
        max_optin as i64 - requested as i64
    );
    // With 128 threads (4 warps) per block, how many blocks can this SM's
    // shared memory hold if the whole per-SM budget is available to one
    // kernel at a time -- the ceiling occupancy would need regardless of
    // registers.
    println!(
        "blocks/SM shared-memory ceiling at this request size: {}",
        per_sm as i64 / requested as i64
    );

    Ok(())
}
