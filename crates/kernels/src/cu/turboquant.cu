// TurboQuant KV cache (arXiv:2504.19874).
//
// Keys use TurboQuant_prod (Algorithm 2): a (b-1)-bit MSE quantizer plus a
// 1-bit QJL sign on the residual, which is what makes the attention logit an
// *unbiased* estimate of the true one. Values use TurboQuant_mse
// (Algorithm 1), because a value vector is averaged under the softmax weights
// and MSE is the objective that matters there.
//
// Everything below lives in the rotated basis. The paper's estimator is
//
//     <q, x~> = <q, Pi^T y~> + (sqrt(pi/2)/d) * gamma * <S q, qjl>
//
// and since Pi is orthogonal and S is i.i.d. Gaussian, S' = S Pi^T is too.
// Substituting it gives
//
//     <q, x~> = <Pi q, y~> + (sqrt(pi/2)/d) * gamma * <S' (Pi q), qjl>
//
// so the query is rotated once per token and no cached vector is ever rotated
// back. For values the same trick moves the inverse rotation from "once per
// cached vector" to "once per (head, token)" after the weighted sum.

// Largest head dimension the static shared buffers below allow.
#define TQ_MAX_D 256
// sqrt(pi/2), the QJL dequantization constant.
#define TQ_SQRT_HALF_PI 1.2533141373155003f

// ---- bit packing --------------------------------------------------------
// Codes are packed little-end first: coordinate i lives in byte i/(8/bits) at
// shift (i % (8/bits)) * bits.

__device__ __forceinline__ int tq_unpack(const uint8_t* __restrict__ codes,
                                         int i, int bits) {
    const int per_byte = 8 / bits;
    const int shift = (i % per_byte) * bits;
    return (codes[i / per_byte] >> shift) & ((1 << bits) - 1);
}

__device__ __forceinline__ float tq_sign_of(const uint8_t* __restrict__ signs,
                                            int i) {
    return ((signs[i >> 3] >> (i & 7)) & 1) ? 1.0f : -1.0f;
}

// ---- rotation -----------------------------------------------------------

// out[v] = M * in[v] for n_vec vectors of length d.
//
// `mat` is column-major (mat[j*d + i] is row i, column j) so the inner loop
// reads consecutive addresses across threads. The input is staged in shared
// memory, which makes calling this with out == in safe.
extern "C" __global__ void tq_matvec(float* __restrict__ out,
                                     const float* __restrict__ in,
                                     const float* __restrict__ mat, int d,
                                     int n_vec) {
    __shared__ float xs[TQ_MAX_D];

    const int v = blockIdx.x;
    if (v >= n_vec) return;
    const float* src = in + (size_t)v * d;

    for (int i = threadIdx.x; i < d; i += blockDim.x) xs[i] = src[i];
    __syncthreads();

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < d; ++j) acc += mat[(size_t)j * d + i] * xs[j];
        out[(size_t)v * d + i] = acc;
    }
}

// ---- quantized stores ---------------------------------------------------

// Nearest centroid in a sorted codebook. Linear over at most 16 entries beats
// a branchy binary search here.
__device__ __forceinline__ int tq_nearest(const float* __restrict__ cb,
                                          int n_levels, float z) {
    int best = 0;
    float best_err = fabsf(z - cb[0]);
    for (int k = 1; k < n_levels; ++k) {
        const float e = fabsf(z - cb[k]);
        if (e < best_err) {
            best_err = e;
            best = k;
        }
    }
    return best;
}

// TurboQuant_mse over already-rotated value vectors.
//
// src is [n_tokens, n_kv_heads, d]; the cache is [n_kv_heads, max_seq, ...].
extern "C" __global__ void tq_store_v(uint8_t* __restrict__ codes,
                                      __half* __restrict__ scale,
                                      const float* __restrict__ src,
                                      const int* __restrict__ slots,
                                      const float* __restrict__ cb, int bits,
                                      int n_kv_heads, int d, int n_slots,
                                      int n_tokens) {
    __shared__ float xs[TQ_MAX_D];
    __shared__ int idx[TQ_MAX_D];

    const int head = blockIdx.x;
    const int token = blockIdx.y;
    if (token >= n_tokens) return;
    const int slot = slots[token];
    if (slot < 0 || slot >= n_slots) return;

    const int n_levels = 1 << bits;
    const float* row = src + ((size_t)token * n_kv_heads + head) * d;

    float acc = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        xs[i] = row[i];
        acc += xs[i] * xs[i];
    }
    const float norm = sqrtf(block_reduce_sum(acc));
    // A zero vector has no direction to quantize; the codes are irrelevant
    // because the scale is zero.
    const float inv = (norm > 0.0f) ? 1.0f / norm : 0.0f;

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        idx[i] = tq_nearest(cb, n_levels, xs[i] * inv);
    }
    __syncthreads();

    const int per_byte = 8 / bits;
    const int n_bytes = d / per_byte;
    uint8_t* dst = codes + ((size_t)head * n_slots + slot) * n_bytes;
    for (int b = threadIdx.x; b < n_bytes; b += blockDim.x) {
        uint8_t packed = 0;
        for (int k = 0; k < per_byte; ++k) {
            packed |= (uint8_t)(idx[b * per_byte + k] << (k * bits));
        }
        dst[b] = packed;
    }
    if (threadIdx.x == 0) scale[(size_t)head * n_slots + slot] = __float2half(norm);
}

// TurboQuant_prod over already-rotated key vectors: MSE codes, then the QJL
// sign of the residual and its norm.
extern "C" __global__ void tq_store_k(uint8_t* __restrict__ codes,
                                      uint8_t* __restrict__ signs,
                                      __half* __restrict__ scale,
                                      __half* __restrict__ gamma,
                                      const float* __restrict__ src,
                                      const float* __restrict__ qjl,
                                      const int* __restrict__ slots,
                                      const float* __restrict__ cb, int bits,
                                      int n_kv_heads, int d, int n_slots,
                                      int n_tokens) {
    __shared__ float xs[TQ_MAX_D];
    __shared__ float resid[TQ_MAX_D];
    __shared__ float proj[TQ_MAX_D];
    __shared__ int idx[TQ_MAX_D];

    const int head = blockIdx.x;
    const int token = blockIdx.y;
    if (token >= n_tokens) return;
    const int slot = slots[token];
    if (slot < 0 || slot >= n_slots) return;

    const int n_levels = 1 << bits;
    const float* row = src + ((size_t)token * n_kv_heads + head) * d;

    float acc = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        xs[i] = row[i];
        acc += xs[i] * xs[i];
    }
    const float norm = sqrtf(block_reduce_sum(acc));
    const float inv = (norm > 0.0f) ? 1.0f / norm : 0.0f;

    // Stage 1: quantize the unit vector, keep the residual.
    float res_sq = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float z = xs[i] * inv;
        idx[i] = tq_nearest(cb, n_levels, z);
        resid[i] = z - cb[idx[i]];
        res_sq += resid[i] * resid[i];
    }
    // gamma is the residual norm of the *unscaled* vector, per Algorithm 2.
    const float res_norm = sqrtf(block_reduce_sum(res_sq));
    __syncthreads();

    // Stage 2: QJL. sign(S'·r) does not depend on the residual's scale, so it
    // can be taken on the normalized residual directly.
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float p = 0.0f;
        for (int j = 0; j < d; ++j) p += qjl[(size_t)j * d + i] * resid[j];
        proj[i] = p;
    }
    __syncthreads();

    const int per_byte = 8 / bits;
    const int n_bytes = d / per_byte;
    uint8_t* code_dst = codes + ((size_t)head * n_slots + slot) * n_bytes;
    for (int b = threadIdx.x; b < n_bytes; b += blockDim.x) {
        uint8_t packed = 0;
        for (int k = 0; k < per_byte; ++k) {
            packed |= (uint8_t)(idx[b * per_byte + k] << (k * bits));
        }
        code_dst[b] = packed;
    }

    uint8_t* sign_dst = signs + ((size_t)head * n_slots + slot) * (d / 8);
    for (int b = threadIdx.x; b < d / 8; b += blockDim.x) {
        uint8_t packed = 0;
        for (int k = 0; k < 8; ++k) {
            if (proj[b * 8 + k] > 0.0f) packed |= (uint8_t)(1u << k);
        }
        sign_dst[b] = packed;
    }

    if (threadIdx.x == 0) {
        scale[(size_t)head * n_slots + slot] = __float2half(norm);
        gamma[(size_t)head * n_slots + slot] = __float2half(norm * res_norm);
    }
}

// ---- quantized attention ------------------------------------------------

// scores[h, t, j] = ( <q_rot, y~_j> + (sqrt(pi/2)/d) * gamma_j * <q_qjl, qjl_j> )
//                   * attn_scale
//
// One warp per score, as in the dense path.
extern "C" __global__ void tq_attn_scores(
    float* __restrict__ scores, const float* __restrict__ q_rot,
    const float* __restrict__ q_qjl, const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ signs, const __half* __restrict__ scale,
    const __half* __restrict__ gamma, const int* __restrict__ seq_of,
    const int* __restrict__ positions, const int* __restrict__ slot_table,
    int table_stride, const float* __restrict__ cb, int bits, int n_heads,
    int n_kv_heads, int d, int n_slots, int kv_len, float attn_scale,
    float qjl_scale) {
    const int j = blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if (j >= kv_len) return;

    const int head = blockIdx.y;
    const int token = blockIdx.z;
    const int lane = threadIdx.x % WARP_SIZE;

    if (j > positions[token]) {
        if (lane == 0) {
            scores[((size_t)head * gridDim.z + token) * kv_len + j] = -INFINITY;
        }
        return;
    }

    const int kv_head = head / (n_heads / n_kv_heads);
    const float* qr = q_rot + ((size_t)token * n_heads + head) * d;
    const float* qs = q_qjl + ((size_t)token * n_heads + head) * d;

    const int per_byte = 8 / bits;
    const int physical = slot_table[(size_t)seq_of[token] * table_stride + j];
    const size_t slot = (size_t)kv_head * n_slots + physical;
    const uint8_t* code = codes + slot * (d / per_byte);
    const uint8_t* sign = signs + slot * (d / 8);

    float mse_term = 0.0f;
    float qjl_term = 0.0f;
    for (int i = lane; i < d; i += WARP_SIZE) {
        mse_term += qr[i] * cb[tq_unpack(code, i, bits)];
        qjl_term += qs[i] * tq_sign_of(sign, i);
    }
    mse_term = warp_reduce_sum(mse_term);
    qjl_term = warp_reduce_sum(qjl_term);

    if (lane == 0) {
        // qjl_scale is 1 for the two-stage estimator and 0 for the MSE-only
        // ablation; nothing else varies between them.
        const float est = __half2float(scale[slot]) * mse_term
                        + qjl_scale * (TQ_SQRT_HALF_PI / (float)d)
                          * __half2float(gamma[slot]) * qjl_term;
        scores[((size_t)head * gridDim.z + token) * kv_len + j] = est * attn_scale;
    }
}

// out[t, h, :] = sum_j p_j * v~_j, still in the rotated basis. The caller
// applies Pi^T once afterwards.
extern "C" __global__ void tq_attn_output(
    float* __restrict__ out, const float* __restrict__ scores,
    const uint8_t* __restrict__ codes, const __half* __restrict__ scale,
    const int* __restrict__ seq_of, const int* __restrict__ positions,
    const int* __restrict__ slot_table, int table_stride,
    const float* __restrict__ cb, int bits, int n_heads, int n_kv_heads,
    int d, int n_slots, int kv_len) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    const int i = threadIdx.x;
    if (i >= d) return;

    const int kv_head = head / (n_heads / n_kv_heads);
    const int per_byte = 8 / bits;
    const float* srow = scores + ((size_t)head * gridDim.y + token) * kv_len;
    const uint8_t* base = codes + (size_t)kv_head * n_slots * (d / per_byte);
    const __half* scale_base = scale + (size_t)kv_head * n_slots;
    const int* table = slot_table + (size_t)seq_of[token] * table_stride;
    const int last = positions[token];

    float acc = 0.0f;
    for (int j = 0; j <= last && j < kv_len; ++j) {
        const int physical = table[j];
        const uint8_t* code = base + (size_t)physical * (d / per_byte);
        acc += srow[j] * __half2float(scale_base[physical])
             * cb[tq_unpack(code, i, bits)];
    }
    out[((size_t)token * n_heads + head) * d + i] = acc;
}
