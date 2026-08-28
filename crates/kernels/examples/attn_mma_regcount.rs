//! Register/shared-memory footprint of the kernels that dominate a long
//! prefill: `attn_decode_mma_f32` (attention) and `mma_e4m3_block_g8_f32`
//! (the FP8 weight GEMM's widest instantiation), at Qwen3.8-27B's shapes.
use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::Kernels;

fn main() -> Result<()> {
    let dev = Device::new(0)?;
    let kern = Kernels::new(dev.clone());
    for name in [
        "attn_decode_mma_f32",
        "mma_e4m3_block_g8_f32",
        "mma_e4m3_block_g4_f32",
        "mma_e4m3_block_g2_f32",
        "mma_e4m3_block_f32",
        "mma_e4m3_block_w32_f32",
        "mma_e4m3_block_w32_g2_f32",
    ] {
        match kern.kernel_registers("infero_fp8", name) {
            Ok((regs, smem)) => println!("{name}: {regs} registers/thread, {smem} static shared bytes/block"),
            Err(e) => println!("{name}: {e}"),
        }
    }
    Ok(())
}
