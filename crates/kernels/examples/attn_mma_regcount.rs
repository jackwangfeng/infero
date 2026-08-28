//! Register/shared-memory footprint of `attn_decode_mma_f32` at Qwen3.8-27B's
//! `d_head` of 256, to check whether the widened accumulator arrays cost
//! occupancy. Compare against a narrower `d_head` for scale.
use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::Kernels;

fn main() -> Result<()> {
    let dev = Device::new(0)?;
    let kern = Kernels::new(dev.clone());
    let (regs, smem) = kern.kernel_registers("infero_ops", "attn_decode_mma_f32")?;
    println!("attn_decode_mma_f32: {regs} registers/thread, {smem} static shared bytes/block");
    Ok(())
}
