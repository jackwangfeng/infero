// Device-side NVFP4 (e2m1) dequantization.
//
// This is the device counterpart to `fp4.rs`'s pure-Rust host reference
// (`e2m1_value`/`dequant_f4e2m1_row`): the bit table and packing/scale
// conventions here are not re-derived, they mirror that file's doc comments
// exactly, so the two can be cross-checked against each other rather than
// against a second independent reading of the NVFP4 spec. If this kernel and
// the host reference ever disagree, this kernel is the one that is wrong.
//
// Layout (see `crate::WeightType::F4E2M1`'s own doc comment for the on-disk
// version this mirrors): `w` holds `n` rows of `k` packed e2m1 elements each,
// two nibbles per byte -- byte `b`'s LOW nibble (bits 3:0) holds element
// `2*b` and its HIGH nibble (bits 7:4) holds element `2*b + 1`, per NVIDIA
// ModelOpt's own pack/unpack convention. `scale` holds `n` rows of
// `ceil(k / F4E2M1_BLOCK)` f8_e4m3 block scales each, one scale per 16
// elements along k. `scale2` is a single per-tensor f32 scalar (NVFP4's
// `weight_scale_2`) shared by the whole matrix, applied as a third
// multiplicative factor on top of the per-block scale.

// Elements covered by one block scale. Must match `fp4::F4E2M1_BLOCK`.
#define F4E2M1_BLOCK 16

// The 16-entry e2m1 value table, matching `fp4.rs`'s `E2M1_TABLE` exactly:
// codes 0..7 index the magnitude ladder {0, 0.5, 1, 1.5, 2, 3, 4, 6}; bit 3
// (the high bit of the 4-bit code) is the sign, sign-magnitude rather than
// two's complement -- code 0b1000 is negative zero, same as the host table.
__device__ __forceinline__ float e2m1_to_f32(unsigned int nibble) {
    const float mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const unsigned int code = nibble & 0x0Fu;
    const float v = mags[code & 0x07u];
    return (code & 0x08u) ? -v : v;
}

// One thread per output element: `blockIdx.x` is the row, `blockIdx.y` and
// `threadIdx.x` tile the k dimension. This kernel is a correctness oracle for
// the host reference and, later, an input to the CUTLASS NVFP4 GEMM's own
// tests -- not a hot path -- so it reads one packed byte and, once every
// sixteen elements, one scale byte, rather than optimizing for the wide
// vector loads the bandwidth-bound kernels in `fp8.cu`/`mmvq.cu` use.
extern "C" __global__ void dequant_f4e2m1_f32(
        float* __restrict__ out, const unsigned char* __restrict__ w,
        const unsigned char* __restrict__ scale, float scale2, int k, int n) {
    const int row = blockIdx.x;
    const int col = blockIdx.y * (int)blockDim.x + (int)threadIdx.x;
    if (row >= n || col >= k) return;

    const int bytes_per_row = (k + 1) / 2;
    const int blocks_per_row = (k + F4E2M1_BLOCK - 1) / F4E2M1_BLOCK;

    const unsigned char* wrow = w + (size_t)row * bytes_per_row;
    const unsigned char* srow = scale + (size_t)row * blocks_per_row;

    const unsigned char byte = wrow[col / 2];
    const unsigned int nibble = (col % 2 == 0) ? (byte & 0x0Fu) : (byte >> 4);
    const int block = col / F4E2M1_BLOCK;
    const float s = e4m3_to_f32((unsigned int)srow[block]);

    out[(size_t)row * (size_t)k + (size_t)col] = e2m1_to_f32(nibble) * s * scale2;
}

// ---- activation quantizer: f32 -> packed e2m1 + per-block f8_e4m3 scale ----
//
// Ports vLLM's own real NVFP4 reference quantizer verbatim (`cast_to_fp4` and
// `ref_nvfp4_quant`, `vllm/model_executor/layers/quantization/utils/
// nvfp4_emulation_utils.py`, real installed vLLM 0.27.1 on `bw`) rather than
// rederiving the scheme from the abstract "4-bit float" description: unlike
// weight quantization (checkpoint-baked, decoded by `dequant_f4e2m1_f32`
// above), activation quantization computes a REAL per-16-block dynamic scale
// from the actual runtime data, using the checkpoint's static `input_scale`
// only as an input to that computation (vLLM's `global_scale` parameter) --
// not as a single divisor applied to every element.
//
// Per 16-element block:
//   1. vec_max   = max(|x_i|) over the block
//   2. scale_f32 = clamp(global_scale * vec_max / 6.0, -448, 448)
//   3. scale_q   = f8_e4m3(scale_f32) read back as f32 -- the LOSSY
//      requantized value is what both the stored scale byte and the
//      following output_scale computation use, not the pre-quantization
//      scale_f32 (real vLLM computes `output_scale` from `scale.to(fp8).
//      to(f32)`, not from the plain `scale` it clamped a line earlier).
//   4. output_scale = 0 if scale_q == 0, else global_scale / scale_q --
//      mirrors `get_reciprocal(scale_q * get_reciprocal(global_scale))`,
//      which collapses to the same thing once global_scale != 0 (the
//      real, checkpoint-supplied case) but also degrades to 0 the same way
//      vLLM's does if global_scale itself is ever 0.
//   5. each element: scaled = x_i * output_scale, clipped to [-6, 6], then
//      rounded to the e2m1 ladder via `cast_to_fp4`'s own asymmetric
//      <=/< thresholds (copied exactly below, not renormalized to
//      round-half-to-even).
//
// Packing matches `dequant_f4e2m1_f32`/ModelOpt exactly: byte `b`'s LOW
// nibble holds element `2*b`, HIGH nibble holds element `2*b + 1`.
//
// One thread per 16-element block (not per output element): a block this
// narrow has nothing to gain from splitting it across lanes the way
// `quantize_act_e4m3_cutlass_f32`'s 128-wide warp reduction does, and
// keeping the whole block in one thread means the two passes over its 16
// elements (max-abs, then quantize) and the paired-byte packing need no
// shared memory or synchronization at all. Like `dequant_f4e2m1_f32`, this
// is a correctness oracle for the host round-trip test, not a hot path.
extern "C" __global__ void quantize_act_e2m1_f32(
        unsigned char* __restrict__ xq, unsigned char* __restrict__ xq_scale,
        const float* __restrict__ x, float global_scale, int k, int n_tokens) {
    const int tok = blockIdx.x;
    const int blocks_per_row = (k + F4E2M1_BLOCK - 1) / F4E2M1_BLOCK;
    const int block = blockIdx.y * (int)blockDim.x + (int)threadIdx.x;
    if (tok >= n_tokens || block >= blocks_per_row) return;

    const int bytes_per_row = (k + 1) / 2;
    const float* xrow = x + (size_t)tok * k;
    unsigned char* xqrow = xq + (size_t)tok * bytes_per_row;
    unsigned char* xsrow = xq_scale + (size_t)tok * blocks_per_row;

    const int base = block * F4E2M1_BLOCK;
    const int n_in_block = min(F4E2M1_BLOCK, k - base);

    // Pass 1: per-block max-abs.
    float vec_max = 0.0f;
#pragma unroll
    for (int i = 0; i < F4E2M1_BLOCK; ++i) {
        if (i < n_in_block) vec_max = fmaxf(vec_max, fabsf(xrow[base + i]));
    }

    // scale_f32 -> fp8 -> read back the lossy value (step 2-3 above).
    float scale_f32 = global_scale * vec_max * (1.0f / 6.0f);
    scale_f32 = fminf(fmaxf(scale_f32, -448.0f), 448.0f);
    const unsigned char scale_byte = f32_to_e4m3(scale_f32);
    xsrow[block] = scale_byte;
    const float scale_q = e4m3_to_f32((unsigned int)scale_byte);

    // output_scale, with the same zero guard as vLLM's `get_reciprocal`.
    const float output_scale = (scale_q == 0.0f) ? 0.0f : (global_scale / scale_q);

    // Pass 2: quantize each element and pack two nibbles per byte. Elements
    // are processed in pairs so each thread writes each byte exactly once --
    // block boundaries are always byte-aligned (F4E2M1_BLOCK is even and
    // `base` is a multiple of it), so no other thread ever touches these
    // bytes.
#pragma unroll
    for (int i = 0; i < F4E2M1_BLOCK; i += 2) {
        if (i >= n_in_block) break;

        unsigned int codes[2] = {0u, 0u};
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int idx = i + j;
            if (idx >= n_in_block) continue;

            const float scaled = xrow[base + idx] * output_scale;
            const float clipped = fminf(fmaxf(scaled, -6.0f), 6.0f);
            const float mag = fabsf(clipped);

            // `cast_to_fp4`'s own asymmetric threshold table, copied
            // verbatim (not renormalized) -- see this function's doc
            // comment for the real vLLM source these bounds are ported from.
            unsigned int mag_code;
            if (mag <= 0.25f) mag_code = 0u;
            else if (mag < 0.75f) mag_code = 1u;
            else if (mag <= 1.25f) mag_code = 2u;
            else if (mag < 1.75f) mag_code = 3u;
            else if (mag <= 2.5f) mag_code = 4u;
            else if (mag < 3.5f) mag_code = 5u;
            else if (mag <= 5.0f) mag_code = 6u;
            else mag_code = 7u;

            const unsigned int sign_bit = (clipped < 0.0f) ? 0x08u : 0x00u;
            codes[j] = mag_code | sign_bit;
        }

        const unsigned char byte = (unsigned char)((codes[1] << 4) | codes[0]);
        xqrow[(base + i) / 2] = byte;
    }
}

// ---- CUTLASS NVFP4 scale-factor swizzle (`cutlass` feature only, but this
// kernel itself is plain NVRTC-compiled CUDA C like everything else in this
// file -- it feeds the AOT `nvcc`-built `cutlass/fp4_bw_gemm.cu`, but is not
// part of it, the same split `fp8.cu`'s `unrepack_rows_e4m3`/
// `transpose_scale_b_f32` already use for CUTLASS's FP8 GEMM) ----
//
// `dequant_f4e2m1_f32` above and `quantize_act_e2m1_cutlass` (`fp4.rs`) both
// read/write the block-scale grid in plain `[rows, blocks]` row-major --
// convenient for a scalar CUDA kernel, but NOT the physical layout CUTLASS's
// own block-scaled tensor-core MMA requires for its `sfa`/`sfb` operands.
// That real layout (`cutlass::detail::Sm1xxBlockScaledConfig<16>`, the
// default `UMMA::Major::K` variant) is a two-level tile:
//
//   - Rows are grouped into tiles of 128 (`m_tile = row / 128`); within a
//     128-row tile, a row's position is `outer_m = row % 32`,
//     `inner_m = (row / 32) % 4` (SO 32 groups of 4, not 4 groups of 32).
//   - Scale blocks (each covering 16 raw elements) are grouped into tiles of
//     4 (`k_tile = block / 4`, `inner_k = block % 4`).
//   - Byte offset = `(m_tile * num_k_tiles + k_tile) * 512
//                    + outer_m * 16 + inner_m * 4 + inner_k`.
//
// This formula is cross-checked against two independent real sources
// (fetched via `gh api` this session, not derived from the abstract "NVFP4
// scale layout" description alone):
//
//   1. CUTLASS's own `Sm1xxBlockScaledBasicChunk`/`Sm1xxBlockScaledConfig`
//      (`include/cutlass/detail/sm100_blockscaled_layout.hpp:48-114`): the
//      K-major `SfAtom` is a CuTe
//      `Layout<Shape<Shape<32,4>,Shape<SFVecSize,4>>, Stride<Stride<16,4>,Stride<0,1>>>`
//      -- working through its coordinate decomposition by hand gives exactly
//      `outer_m*16 + inner_m*4 + inner_k` for the atom-local offset (max
//      31*16+3*4+3=511, matching the atom's real 128*4=512-entry size), tiled
//      row-major across `(m_tile, k_tile)` pairs by `tile_to_shape`.
//   2. vLLM's own real CUDA repack kernel,
//      `cvt_quant_to_fp4_get_sf_out_offset` (`csrc/libtorch_stable/
//      quantization/fp4/nvfp4_utils.cuh:164-200`, real, currently-installed
//      dispatch that feeds CUTLASS's identical NVFP4 GEMM from the Python
//      side): its own plain integer arithmetic (`mTileIdx = mIdx >> 7;
//      outerMIdx = mIdx & 31; innerMIdx = (mIdx >> 5) & 3; kTileIdx = kIdx >>
//      2; innerKIdx = kIdx & 3; SFOffset = (mTileIdx*numKTiles+kTileIdx)<<9 |
//      outerMIdx<<4 | innerMIdx<<2 | innerKIdx;`) is bit-for-bit the same
//      formula, independently confirming source 1's own CuTe layout algebra.
//
// The padded extent this kernel's caller must allocate is
// `rows.div_ceil(128) * blocks.div_ceil(4) * 512` bytes (matches both
// sources: source 2's own `computeSwizzledSFShape`, `rounded_m =
// round_up(rows,128)`, `rounded_n = round_up(blocks,4)`, total bytes =
// `rounded_m * rounded_n`). This kernel's own launch covers exactly that
// padded `(rows_padded, blocks_padded)` domain -- out-of-range source reads
// (the real tail of a non-128/non-4 multiple) are written as `0`, so the
// caller does not need to pre-zero the output buffer.
extern "C" __global__ void swizzle_sf_e2m1(unsigned char* __restrict__ sf_swizzled,
                                            const unsigned char* __restrict__ sf_flat, int rows, int blocks) {
    const int row = blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int block = blockIdx.y;
    const int num_k_tiles = (blocks + 3) / 4;
    const int rows_padded = ((rows + 127) / 128) * 128;
    if (row >= rows_padded) return;

    const unsigned char v = (row < rows && block < blocks) ? sf_flat[(size_t)row * blocks + block] : 0;

    const int m_tile = row >> 7;
    const int outer_m = row & 31;
    const int inner_m = (row >> 5) & 3;
    const int k_tile = block >> 2;
    const int inner_k = block & 3;
    const long off = ((long)m_tile * num_k_tiles + k_tile) * 512 + outer_m * 16 + inner_m * 4 + inner_k;
    sf_swizzled[off] = v;
}
