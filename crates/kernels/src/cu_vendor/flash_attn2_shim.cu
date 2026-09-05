// A thin, torch-free `extern "C"` wrapper around Dao-AILab/flash-attention's
// real CUDA forward kernel, scoped to exactly one case: causal, fp16,
// d_head<=256, a single contiguous sequence (no varlen, no paged KV, no
// dropout, no rotary, no alibi, no softcap, no split-KV). See
// docs/superpowers/specs/2026-09-05-pluggable-attention-backend-design.md.
//
// `FLASHATTENTION_DISABLE_DROPOUT` (passed via -D on the nvcc command line,
// see build.rs) is what makes this torch-free at all: flash_fwd_kernel.h's
// only ATen reference (`at::cuda::philox::unpack`, for dropout RNG state) is
// `#ifndef`-guarded by that exact macro, and flash.h's own comment confirms
// this is the officially-supported no-ATen build mode, not a hack.
//
// We do NOT include flash_fwd_launch_template.h -- it pulls in
// <c10/cuda/CUDAException.h> for its C10_CUDA_CHECK/C10_CUDA_KERNEL_LAUNCH_CHECK
// macros, a real torch dependency with no disable switch. Everything it does
// for our one fixed case (pick a Kernel_traits, define the __global__
// trampoline, compute a grid, launch) is reproduced directly below instead,
// using plain cudaGetLastError() for error checking.

#define FLASHATTENTION_DISABLE_DROPOUT

#include <cmath>
#include <cuda_runtime.h>

#include "flash.h"
#include "flash_fwd_kernel.h"

using namespace FLASH_NAMESPACE;

// A real (not anonymous) namespace: an anonymous one here collides at the
// mangled-symbol level with CUTLASS's own anonymous namespace inside
// `cute/atom/mma_traits_sm70.hpp` (both end up hashed to the identical
// `_GLOBAL__N__...` internal-linkage name in the same translation unit,
// which nvcc's host-compiler pass then reports as "ambiguous") -- a real,
// reproduced compiler error, not a guess.
namespace infero_flash_attn2_shim {

// H100-style 64x64/4-warp/96KB-smem config for Headdim=256 (matches
// run_mha_fwd_hdim256's own second branch) -- fits this GPU's real ~99KiB
// per-block shared-memory ceiling (established earlier this investigation),
// unlike the 128x64/8-warp/128KB-smem "A100-style" branch, which does not.
using Traits256 = Flash_fwd_kernel_traits<256, 64, 64, 4, false, false, cutlass::half_t>;

// Fixed template booleans for our one supported case: no dropout, causal,
// no local/sliding-window, no alibi, general (not "even") M/N tiling (safe
// for any shape, not just block-size-aligned ones), Is_even_K true (our
// d_head always equals Kernel_traits::kHeadDim exactly), no softcap, no
// softmax-return. Mirrors flash_fwd_launch_template.h's
// DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel, ...) macro instantiation,
// specialized directly instead of going through its runtime *_SWITCH cascade.
__global__ void infero_flash_fwd_kernel(__grid_constant__ const Flash_fwd_params params) {
    FLASH_NAMESPACE::compute_attn<Traits256,
        /*Is_dropout=*/false, /*Is_causal=*/true, /*Is_local=*/false,
        /*Has_alibi=*/false, /*Is_even_MN=*/false, /*Is_even_K=*/true,
        /*Is_softcap=*/false, /*Return_softmax=*/false>(params);
}

} // namespace infero_flash_attn2_shim

extern "C" int infero_flash_attn2_fwd_causal_f16(
    const void* q, const void* k, const void* v, void* out, void* lse_scratch,
    int seqlen_q, int seqlen_k, int n_heads, int n_kv_heads, int d_head,
    int kv_n_slots, float softmax_scale, cudaStream_t stream
) {
    if (d_head != 256) {
        return -1; // this shim only instantiates the d_head=256 Kernel_traits
    }

    Flash_fwd_params params{};
    params.q_ptr = const_cast<void*>(q);
    params.k_ptr = const_cast<void*>(k);
    params.v_ptr = const_cast<void*>(v);
    params.o_ptr = out;

    using index_t = Flash_fwd_params::index_t;
    const index_t hd = d_head;
    // Q/O: infero's real activation layout, confirmed against `ops.cu`'s own
    // indexing (`q[(token * n_heads + head) * d_head + d]`) -- flat
    // [seqlen, n_heads, d_head] row-major, token-major.
    params.q_batch_stride = static_cast<index_t>(seqlen_q) * n_heads * hd;
    params.q_row_stride = static_cast<index_t>(n_heads) * hd;
    params.q_head_stride = hd;
    // K/V: infero's real KV-pool layout is `[n_kv_heads, n_slots, d_head]`
    // per layer (confirmed against `ops.cu`'s own
    // `k_cache + kv_head * n_slots * d_head` base-pointer computation) --
    // head-MAJOR, not token-major like Q/O. `k`/`v` here already point at
    // this sequence's first cached position within kv_head 0 (the Rust
    // caller resolved and verified a physically-contiguous run before this
    // call); `kv_n_slots` (the pool's total per-head slot capacity, NOT
    // `seqlen_k`) is what actually separates one head's region from the
    // next, so head_stride must use it, not seqlen_k.
    params.k_batch_stride = 0; // b=1, never read (bidb is always 0)
    params.v_batch_stride = 0;
    params.k_row_stride = hd;
    params.v_row_stride = hd;
    params.k_head_stride = static_cast<index_t>(kv_n_slots) * hd;
    params.v_head_stride = params.k_head_stride;
    params.h = n_heads;
    params.h_k = n_kv_heads;
    params.h_h_k_ratio = n_heads / n_kv_heads;

    params.o_batch_stride = params.q_batch_stride;
    params.o_row_stride = params.q_row_stride;
    params.o_head_stride = hd;

    params.p_ptr = nullptr;
    params.softmax_lse_ptr = lse_scratch;
    params.softmax_lseaccum_ptr = nullptr;

    params.b = 1;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_knew = 0;
    params.d = d_head;
    params.seqlen_q_rounded = seqlen_q;
    params.seqlen_k_rounded = seqlen_k;
    params.d_rounded = d_head;
    params.rotary_dim = 0;
    params.total_q = seqlen_q;

    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * static_cast<float>(M_LOG2E);

    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.leftpad_k = nullptr;
    params.seqused_k = nullptr;
    params.blockmask = nullptr;
    params.knew_ptr = nullptr;
    params.vnew_ptr = nullptr;
    params.rotary_cos_ptr = nullptr;
    params.rotary_sin_ptr = nullptr;
    params.cache_batch_idx = nullptr;
    params.block_table = nullptr;
    params.page_block_size = 0;

    params.p_dropout = 1.0f; // keep-probability 1.0 == no dropout
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = params.scale_softmax;

    // The real torch-binding entry point (`flash_api.cpp`) special-cases
    // this: "Causal is the special case where window_size_right == 0 and
    // window_size_left < 0" (its own comment, `mask.h` repeats it) --
    // `if (is_causal) { window_size_right = 0; }`, unconditionally, not -1.
    // `mask.h`'s actual per-row bound is
    // `col_idx_limit_right = min(seqlen_k, row_idx + 1 + ... + window_size_right)`;
    // at -1 this excludes the diagonal itself (row R can see cols < R, not
    // <= R), so row 0 gets zero valid columns and later rows lose one
    // increasingly-significant term -- this is the actual bug this shim had.
    params.window_size_left = -1;
    params.window_size_right = 0;
    params.softcap = 0.0f;

    params.rng_state = nullptr;
    params.is_bf16 = false;
    params.is_causal = true;
    params.is_seqlens_k_cumulative = true;
    params.is_rotary_interleaved = false;
    params.num_splits = 1;
    params.alibi_slopes_ptr = nullptr;
    params.alibi_slopes_batch_stride = 0;
    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;

    constexpr size_t smem_size = infero_flash_attn2_shim::Traits256::kSmemSize;
    if (smem_size >= 48 * 1024) {
        cudaError_t attr_status = cudaFuncSetAttribute(
            infero_flash_attn2_shim::infero_flash_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        if (attr_status != cudaSuccess) {
            return static_cast<int>(attr_status);
        }
    }

    const int num_m_block = (seqlen_q + infero_flash_attn2_shim::Traits256::kBlockM - 1) / infero_flash_attn2_shim::Traits256::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    infero_flash_attn2_shim::infero_flash_fwd_kernel<<<grid, infero_flash_attn2_shim::Traits256::kNThreads, smem_size, stream>>>(params);

    cudaError_t err = cudaGetLastError();
    return static_cast<int>(err);
}
