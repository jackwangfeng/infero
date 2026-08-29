#!/usr/bin/env python3
"""Apply the text-model stem probe to a pristine checkout's weights.rs.

Exists because the working tree has several agents editing the same files, so a
load attempt has to be made against committed code plus exactly one change. This
applies that change to an exported HEAD and refuses if the target does not look
like what it expects.
"""

import sys

STEM_PROBE = '''    // Where the text model sits. A multimodal export nests it under
    // `language_model`, so the same tensor is
    // `model.language_model.embed_tokens.weight` there and
    // `model.embed_tokens.weight` everywhere else. Probed rather than derived
    // from the architecture name: the nesting is a property of how the
    // checkpoint was written.
    //
    // The layer prefix below is derived from this rather than probed separately.
    // Probing them independently is how the first attempt at the 27B got the
    // layers right and the embedding wrong, and failed one tensor into the load.
    let stem = ["model.language_model", "model"]
        .into_iter()
        .find(|s| w.get(&format!("{s}.embed_tokens.weight")).is_some())
        .context(
            "found no embedding under `model.embed_tokens.weight` or \\
             `model.language_model.embed_tokens.weight`; the checkpoint's tensor \\
             names are not ones this loader recognises",
        )?;
    tracing::info!(stem, "text model tensors");

    let embd = w.tensor(&format!("{stem}.embed_tokens.weight"))?;'''

LAYER_PREFIX = '''    let layer_prefix = format!("{stem}.layers");
    anyhow::ensure!(
        [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "linear_attn.in_proj_qkv.weight",
        ]
        .iter()
        .any(|leaf| w.get(&format!("{layer_prefix}.0.{leaf}")).is_some()),
        "the embedding is under `{stem}` but there is no layer 0 under \\
         `{layer_prefix}`; this checkpoint splits the text model across two \\
         prefixes and the loader assumes one"
    );
'''


FP8_BRANCH = '''    let projection_bytes = |prefix: &str| -> Result<(Vec<u8>, WeightType, usize, usize)> {
        // FP8 and plain-float exports name the matrix `{prefix}.weight`; AWQ
        // splits it into qweight/qzeros/scales. Check for the single tensor
        // first, because its absence is the cheap question.
        //
        // Note the transposed convention between the two. AWQ stores
        // `[in_features, out_features / 8]`, so `k` is dimension 0. Everything
        // else stores output-major `[out_features, in_features]`, so `k` is
        // dimension 1. Reading one with the other's convention gives a matrix of
        // plausible size and wrong meaning.
        if let Some(t) = w.get(&format!("{prefix}.weight")) {
            let (n, k) = (t.shape[0], t.shape[1]);
            let halves: Vec<half::f16> = if t.dtype == infero_safetensors::Dtype::F8E4M3 {
                // Block-scaled FP8. The scale grid is 128x128 and
                // `dequant_f8_to_f16` validates that the grid matches the
                // quants, which is the check that catches a transposed or
                // row-only index — neither of which fails on its own, they just
                // mis-scale every tile past the first.
                let scales = w
                    .tensor(&format!("{prefix}.weight_scale_inv"))
                    .with_context(|| format!("{prefix} is FP8 but has no scale grid"))?;
                t.dequant_f8_to_f16(&scales, 128)
                    .with_context(|| format!("dequantizing {prefix}"))?
            } else {
                t.to_f16()
                    .with_context(|| format!("converting {prefix} to f16"))?
                    .into_owned()
            };
            // Safety: f16 is a transparent u16, so these are already the
            // little-endian halves the device wants.
            let bytes = unsafe {
                std::slice::from_raw_parts(halves.as_ptr() as *const u8, halves.len() * 2)
            }
            .to_vec();
            return Ok((bytes, WeightType::F16, k, n));
        }
        let qw = w.tensor(&format!("{prefix}.qweight"))?;'''


CHECK_SHAPES = '''            if let Some(g) = &l.gdn {
                // A GatedDeltaNet block. Its widths come from the linear
                // dimensions, not from the attention ones, and checking it
                // against `d_attn` would pass on some of them by coincidence.
                let la = cfg.linear_attn.context(
                    "a block has GatedDeltaNet weights but the config gives no \\
                     linear-attention dimensions to check them against",
                )?;
                let (key_dim, val_dim) = (la.key_dim(), la.value_dim());
                expect(&g.in_proj_qkv, d, la.conv_channels(), "in_proj_qkv")?;
                expect(&g.in_proj_z, d, val_dim, "in_proj_z")?;
                expect(&g.in_proj_a, d, la.value_heads, "in_proj_a")?;
                expect(&g.in_proj_b, d, la.value_heads, "in_proj_b")?;
                expect(&g.out_proj, val_dim, d, "out_proj")?;
                // The 1-D parameters, whose lengths encode the head counts.
                for (v, want, what) in [
                    (&g.conv1d, la.conv_channels() * la.conv_kernel, "conv1d"),
                    (&g.a_log, la.value_heads, "A_log"),
                    (&g.dt_bias, la.value_heads, "dt_bias"),
                    (&g.norm, la.value_head_dim, "norm"),
                ] {
                    anyhow::ensure!(
                        v.len() == want,
                        "layer {i} {what} has {} elements, expected {want}",
                        v.len()
                    );
                }
                let _ = key_dim;
            } else {
                let a = l.attn();
                // A gated q projection is twice as wide: a query and its gate
                // interleaved per head.
                let q_cols = if a.output_gate { 2 * da } else { da };
                expect(&a.wq, d, q_cols, "attn_q")?;
                expect(&a.wk, d, kv_dim, "attn_k")?;
                expect(&a.wv, d, kv_dim, "attn_v")?;
                expect(&a.wo, da, d, "attn_output")?;
            }'''


DOMINANT = '''            // A block's matrices depend on which mixer it has. Reaching for the
            // attention ones unconditionally is what the `attn()` accessor
            // panics about, and this loop runs over every layer.
            let mixer: Vec<&Matrix> = match (&l.attn, &l.gdn) {
                (Some(a), _) => vec![&a.wq, &a.wk, &a.wv, &a.wo],
                (_, Some(g)) => vec![
                    &g.in_proj_qkv,
                    &g.in_proj_z,
                    &g.in_proj_a,
                    &g.in_proj_b,
                    &g.out_proj,
                ],
                _ => vec![],
            };
            for m in mixer
                .into_iter()
                .chain([&l.w_gate, &l.w_up, &l.w_down])
            {'''


def main():
    path = sys.argv[1]
    s = open(path).read()

    old = '    let embd = w.tensor("model.embed_tokens.weight")?;'
    assert old in s, "the embedding load site is not where this patch expects it"
    s = s.replace(old, STEM_PROBE, 1)

    old = '    let output_norm = vector("model.norm.weight", &mut device_bytes)?;'
    assert old in s, "the final norm load site moved"
    s = s.replace(
        old, '    let output_norm = vector(&format!("{stem}.norm.weight"), &mut device_bytes)?;', 1
    )

    old = '''    let projection_bytes = |prefix: &str| -> Result<(Vec<u8>, WeightType, usize, usize)> {
        let qw = w.tensor(&format!("{prefix}.qweight"))?;'''
    assert old in s, "projection_bytes moved"
    s = s.replace(old, FP8_BRANCH, 1)

    old = '''            expect(&l.attn().wq, d, da, "attn_q")?;
            expect(&l.attn().wk, d, kv_dim, "attn_k")?;
            expect(&l.attn().wv, d, kv_dim, "attn_v")?;
            expect(&l.attn().wo, da, d, "attn_output")?;'''
    assert old in s, "check_shapes moved"
    s = s.replace(old, CHECK_SHAPES, 1)

    old = '            for m in [&l.attn().wq, &l.attn().wk, &l.attn().wv, &l.attn().wo, &l.w_gate, &l.w_up, &l.w_down] {'
    assert old in s, "dominant_type moved"
    s = s.replace(old, DOMINANT, 1)

    a = s.index('    let layer_prefix = ["model.layers", "model.language_model.layers"]')
    b = s.index('    tracing::info!(prefix = layer_prefix, "decoder layers");')
    s = s[:a] + LAYER_PREFIX + s[b:]
    s = s.replace(
        '    tracing::info!(prefix = layer_prefix, "decoder layers");',
        '    tracing::info!(prefix = %layer_prefix, "decoder layers");',
        1,
    )

    open(path, "w").write(s)
    print("patched")


if __name__ == "__main__":
    main()
