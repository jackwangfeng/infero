//! Every MSL kernel compiles and resolves by name.
//!
//! A cheap gate in front of the numeric tests: a typo in a kernel nobody
//! exercises yet still breaks the module for everyone who shares its source
//! string, because a Metal library is compiled whole.

use anyhow::Result;
use infero_metal::Device;

const COMMON: &str = include_str!("../../kernels/src/msl/common.metal");
const OPS: &str = include_str!("../../kernels/src/msl/ops.metal");
const QUANT: &str = include_str!("../../kernels/src/msl/quant.metal");
const GDN: &str = include_str!("../../kernels/src/msl/gdn.metal");
const SAMPLE: &str = include_str!("../../kernels/src/msl/sample.metal");

#[test]
fn every_kernel_compiles_and_resolves() -> Result<()> {
    let dev = Device::new(0)?;
    // The module split mirrors the CUDA side's: a change to the GDN unit must
    // not force the rest to recompile.
    let units: &[(&'static str, String, &[&str])] = &[
        (
            "infero_ops",
            format!("{COMMON}\n{OPS}"),
            &[
                "add_f32",
                "add_assign_f32",
                "add_bias_f32",
                "silu_mul_split_f32",
                "embed_f16",
                "rms_norm_f32",
                "rope_neox_f32",
                "store_kv_contig_f16",
                "attn_decode_f32",
                "qk_norm_f32",
                "rope_partial_f32",
            ],
        ),
        (
            "infero_quant",
            format!("{COMMON}\n{QUANT}"),
            &[
                "gemv_f32",
                "gemv_f16",
                "gemv_q8_0",
                "gemv_q4_K",
                "gemv_q6_K",
                "embed_row_q4_K",
                "gemv_mma_q4_K",
            ],
        ),
        (
            "infero_gdn",
            format!("{COMMON}\n{GDN}"),
            &[
                "gdn_conv_f32",
                "gdn_gate_decay_f32",
                "gdn_qk_l2norm_f32",
                "gdn_delta_rule_f32",
                "gdn_gated_rmsnorm_f32",
                "sigmoid_gate_f32",
                "split_interleaved_f32",
            ],
        ),
        (
            "infero_sample",
            format!("{COMMON}\n{SAMPLE}"),
            &[
                "argmax_partial_f32",
                "argmax_combine_f32",
                "sample_rows_f32",
                "sample_topk_partial_f32",
                "sample_rows_topk_f32",
            ],
        ),
    ];

    let mut total = 0;
    for (module, src, names) in units {
        for name in *names {
            let f = dev.kernels().get(module, src, name)?;
            // A pipeline that cannot hold a full SIMD group would make every
            // ported reduction wrong, so it is worth asserting rather than
            // discovering in the numbers.
            assert!(
                f.max_threads_per_group() >= 32,
                "{module}::{name} allows only {} threads a group",
                f.max_threads_per_group()
            );
            total += 1;
        }
        eprintln!("  {module}: {} kernels", names.len());
    }
    eprintln!("  {total} kernels total");
    Ok(())
}
