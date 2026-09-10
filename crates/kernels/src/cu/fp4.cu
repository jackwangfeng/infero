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
