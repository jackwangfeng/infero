/***************************************************************************************************
 * Portions Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Adapted from CUTLASS's examples/87_blackwell_geforce_gemm_blockwise/
 * 87b_blackwell_geforce_fp8_bf16_gemm_groupwise.cu (the per-token/128x128-block
 * scaled FP8->bf16 GEMM for SM120, i.e. GeForce/RTX-PRO consumer Blackwell —
 * `87a` in the same directory block-scales M too and is the wrong one; that
 * mismatch cost a debugging session, see the project memory this came out of).
 * Stripped of the CLI harness, host reference, and CUTLASS `HostTensor`
 * scaffolding down to a bare launchable pair of functions, AOT-compiled with
 * nvcc instead of NVRTC because CUTLASS's template depth is not a realistic
 * JIT-compile target the way every other kernel in this crate is.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// Memory layout this pair of functions expects (verified numerically against
// a from-scratch dequantize-and-matmul reference, not against vLLM's Python
// wrapper -- that wrapper repacks its inputs internally, so bouncing values
// off it is not a reliable check of the raw layout):
//
//   a   [M,K]     e4m3, row-major             -- matches quantize_act_e4m3_f32's xq as-is
//   sfa [K/128,M] f32,  row-major              -- the TRANSPOSE of quantize_act_e4m3_f32's xs
//                                                  ([M,K/128] row-major); caller must transpose
//   b   [N,K]     e4m3, row-major             -- matches WeightType::F8E4M3's quant bytes as-is
//   sfb [K/128,N/128] f32, row-major           -- the TRANSPOSE of WeightType::F8E4M3's scale
//                                                  grid ([N/128,K/128] row-major, see
//                                                  infero_kernels::fp8::scale_grid); caller must
//                                                  transpose once at weight-load time
//   d   [M,N]     bf16, row-major
//
// M must be a multiple of 128 (`can_implement` rejects anything else with a
// non-success status, not a crash) -- callers pad.

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

using ElementA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::float_e4m3_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = cutlass::bfloat16_t;
using LayoutC = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementD = ElementC;
using LayoutD = LayoutC;
constexpr int AlignmentD = AlignmentC;

using ElementAccumulator = float;
using ElementCompute = float;

// Per-token activation scale (M=1), 128x128 weight block scale -- infero's
// existing quantization scheme (quantize_act_e4m3_f32, WeightType::F8E4M3),
// not the DeepSeek-V3-style all-dims-128 scheme example 87a uses.
constexpr int ScaleGranularityM = 1;
constexpr int ScaleGranularityN = 128;
constexpr int ScaleGranularityK = 128;
using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using CooperativeMmaTileShape_MNK = Shape<_128, _128, _128>;
using ClusterShape_MNK = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, CooperativeMmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC, LayoutC,
    AlignmentC, ElementD, LayoutC, AlignmentD, cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB, ElementAccumulator, CooperativeMmaTileShape_MNK,
    ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                                         CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

extern "C" size_t infero_cutlass_fp8_bw_gemm_workspace(int m, int n, int k) {
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
  auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
      {{}, nullptr, stride_D, nullptr, stride_D}};
  return Gemm::get_workspace_size(arguments);
}

extern "C" int32_t infero_cutlass_fp8_bw_gemm(const void* a, const void* b, const float* sfa, const float* sfb,
                                               void* d, void* workspace, int m, int n, int k, cudaStream_t stream) {
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
  auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {static_cast<const ElementA*>(a), stride_A, static_cast<const ElementB*>(b), stride_B, sfa, layout_SFA, sfb,
       layout_SFB},
      {{}, static_cast<const ElementC*>(d), stride_D, static_cast<ElementD*>(d), stride_D}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;

  Gemm gemm;
  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return static_cast<int32_t>(status);
  status = gemm.initialize(arguments, workspace, stream);
  if (status != cutlass::Status::kSuccess) return static_cast<int32_t>(status);
  status = gemm.run(arguments, workspace, stream);
  return static_cast<int32_t>(status);
}

// f32-direct variant: writes straight into the model's own `out` buffer (no
// bf16 scratch, no separate upconvert/discard kernel afterward) -- only
// possible now that `mma_e4m3_cutlass_sfa` no longer pads `M`, so there are no
// padded rows for a separate kernel to discard; the only remaining job of
// that kernel was the bf16->f32 upconvert, which this removes by having
// CUTLASS's own epilogue write f32 (and, when `beta=1`, accumulate into the
// caller's existing `out` directly, matching `bf16_store_or_accum_f32`'s own
// `accum` flag) in one pass instead of two. A prior attempt at f32 output
// (see project memory) kept the second kernel and only changed its element
// width, which is why it regressed (pure once-more-bytes with nothing saved);
// this one is not that experiment.
namespace f32out {
using ElementC = float;
using LayoutC = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementD = ElementC;
using AlignmentD = std::integral_constant<int, AlignmentC>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, CooperativeMmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC, LayoutC,
    AlignmentC, ElementD, LayoutC, AlignmentD::value,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB, ElementAccumulator, CooperativeMmaTileShape_MNK,
    ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                                         CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideD = typename Gemm::GemmKernel::StrideD;
}  // namespace f32out

// `f32out`'s own wide `<128,128,128>` tile with its scheduler swapped from
// the default (`void`, which for `arch::Sm120` resolves to
// `PersistentTileSchedulerSm100` -- see `tile_scheduler.hpp`'s own selector
// table) to `cutlass::gemm::StreamKScheduler` -- CUTLASS's own generic
// scheduler-selector tag, dtype/mainloop-agnostic, that maps `Sm120` onto
// `PersistentTileSchedulerSm100StreamK` (real, verified directly against
// this project's own vendored CUTLASS source, `tile_scheduler.hpp` lines
// ~387-410, not guessed or inferred from a changelog). A prior investigation
// concluded "no sm120 stream-K scheduler exists in this CUTLASS release" --
// that was wrong (or at least incomplete): the selector mapping was already
// present, just never checked because there is no *file* literally named
// `sm120_tile_scheduler_stream_k.hpp` (sm120 reuses sm100's), and no
// blockwise-FP8+stream-K *test* exists upstream (only NVFP4/sparse
// combinations do) -- but `GemmUniversal`'s mainloop/epilogue/scheduler
// template slots are independent, so nothing prevents plugging this
// existing scheduler onto our own existing blockwise FP8 mainloop, which is
// exactly what this instantiation does. Motivation: the wide default tile's
// own ~28% SM-idle at small M (the batch=16 decode shape) is a real,
// previously "doubly closed" finding (see project memory) whose only
// remaining fix was thought to require a newer CUTLASS release or a
// from-scratch hand-rolled split-K kernel -- if this compiles and is
// correct, it is neither.
namespace stream_k {
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, f32out::CollectiveMainloop,
                                                         f32out::CollectiveEpilogue, cutlass::gemm::StreamKScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideD = typename Gemm::GemmKernel::StrideD;
}  // namespace stream_k

// Small-M instantiation of the same f32out kernel, for decode-shaped calls
// (n_tokens in the low tens, not prefill's thousands). `f32out`'s own
// `CooperativeMmaTileShape_MNK` is `<128,128,128>` -- fixed, chosen for
// prefill's large M -- so a batch=16 call still schedules the full 128-wide
// M tile and wastes 112 of it. vLLM's own `cutlass_scaled_mm` binding does
// not special-case small M at the Python call site the way this file's
// `f32out` vs plain split does per-dtype (verified by reading
// `CutlassFp8BlockScaledMMKernel.apply_block_scaled_mm` in the installed
// vLLM source directly: one call, no M threshold) -- so the fix is not "stop
// using CUTLASS below some M", it's "CUTLASS's own tile-shape selection is
// the lever", the same one a well-tuned CUTLASS deployment already pulls.
// `<64,128,128>` still wastes M at n_tokens < 64 but only half as much;
// picked over `<32,...>` because `EpilogueTileAuto` selects a 64-wide
// epilogue tile for this element/layout combination and CUTLASS requires
// EPI_TILE_M to divide CTA_M (`sm90_epilogue_tma_warpspecialized.hpp`'s own
// static_assert) -- 32 failed to compile for exactly this reason, verified
// by trying it first, not assumed. Narrowing to 32 with an explicit
// (non-auto) smaller epilogue tile is a follow-up, not this change.
//
// `KernelTmaWarpSpecializedBlockwisePingpongSm120`, not the plain
// `KernelScheduleSm120Blockwise` tag the other three instantiations in this
// file use (which resolves to the *Cooperative* schedule): CUTLASS's own
// `sm90_gemm_tma_warpspecialized_cooperative.hpp` static_asserts "Cooperative
// kernel requires Tile Size to be greater than or equal to 128 along the
// M-dimension" -- a hard schedule-level floor, not a tunable, so `<64,...>`
// under Cooperative failed to compile for a second, different reason than
// the epilogue-tile one above. Pingpong is CUTLASS's other TMA
// warp-specialized schedule for this same blockwise-scaled family
// (`dispatch_policy.hpp`'s `KernelTmaWarpSpecializedBlockwisePingpongSm120`,
// alongside `...CooperativeSm120` -- both real, sibling schedule tags, not
// one improvised) and has no such floor.
namespace small_m {
using CooperativeMmaTileShape_MNK = Shape<_64, _128, _128>;

using ElementC = float;
using LayoutC = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementD = ElementC;
using AlignmentD = std::integral_constant<int, AlignmentC>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, CooperativeMmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC, LayoutC,
    AlignmentC, ElementD, LayoutC, AlignmentD::value,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB, ElementAccumulator, CooperativeMmaTileShape_MNK,
    ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                                         CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideD = typename Gemm::GemmKernel::StrideD;
}  // namespace small_m

// Small-M, operand-swapped instantiation. `small_m` above (`<64,128,128>`)
// still wastes M at n_tokens=16 (16 real rows in a 64-wide tile is 25%
// utilization on that axis) -- CUTLASS's own schedule-level floors already
// rule out a narrower *non-swapped* tile here (see `small_m`'s own comment:
// Cooperative requires Tile M >= 128, and a 32-wide epilogue tile failed to
// compile under `EpilogueTileAuto`). vLLM does not solve this with a
// narrower tile either -- it swaps which operand plays the GEMM's "A" role,
// so the *weight*'s (thousands-wide) dimension becomes the tiled/parallel
// axis and the real batch width becomes a narrow N=32 tile instead.
// Verified against vLLM's own real, current GitHub source (not recalled from
// training data): `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/
// scaled_mm_blockwise_sm120_fp8_dispatch.cuh`, fetched via `gh api` against
// `vllm-project/vllm`'s main branch 2026-09-08 --
// `sm120_blockwise_fp8_config_swapab` (`TileShape<128,32,128>`,
// `ScaleGranularity(128,1,128)`, `KernelTmaWarpSpecializedBlockwiseCooperativeSm120`),
// selected by `cutlass_gemm_blockwise_sm120_fp8_dispatch`'s own
// `bool swap_ab = (M <= 64);` -- the exact same block-scaled scheme this
// file uses (`ScaleGranularityN=128`), not a different quantization family,
// so this is a real apples-to-apples match, not a guess.
//
// `LayoutTranspose` (real, in this file's own vendored CUTLASS 4.8,
// `cutlass/layout/matrix.h:1332`) is what lets the epilogue write into the
// SAME `[m,n]` row-major `d` buffer every other entry point in this file
// uses -- confirmed by reading `cutlass_gemm_caller_blockwise` in the same
// vLLM source file: `a_stride`/`b_stride` are always computed from each
// buffer's own real physical shape (`a` is always `[m,k]`, `b` is always
// `[n,k]`) regardless of the swap; the swap only changes which buffer/stride
// pair is handed to the mainloop's "A" vs "B" slot, and which of
// `Sm120BlockwiseScaleConfig`'s `majorSFA`/`majorSFB` template args is `K`
// vs `MN` (`cutlass/detail/blockwise_scale_layout.hpp:282`, confirmed to
// take exactly these two `UMMA::Major` params by reading that header
// directly, not assumed).
namespace small_m_swap {
constexpr int ScaleGranularityM = 128;
constexpr int ScaleGranularityN = 1;
// `Major::MN` for BOTH (the same defaults the file-level, non-swapped
// `ScaleConfig` above uses), NOT vLLM's own `Major::K, Major::MN` --
// verified by working out `tile_atom_to_shape_SFA`/`SFB`'s real physical
// layout (`cutlass/detail/blockwise_scale_layout.hpp`) against what this
// file's `sfa`/`sfb` buffers actually contain, not by assuming vLLM's choice
// carries over. vLLM's `Major::K` reflects *vLLM's own* weight-scale buffer
// convention (apparently stored token-block-fast already); this file's `sfb`
// is deliberately pre-transposed at weight-load time to `[K/128,N/128]`
// row-major (N-block fast -- see this file's own header comment), and
// `Major::MN` is what makes `tile_atom_to_shape_SFA` (with the swapped
// problem's "M" = the real weight-N axis) read that exact physical layout:
// working through the stride formula, `Major::MN` gives offset =
// m_block + k_block*ceil_div(M,128), i.e. m_block(=n_block) fast, matching
// `sfb` exactly; `Major::K` would want k_block fast instead, the wrong way
// around for this file's buffers. Found by getting this wrong first (a real
// test failure, `the_f32out_gemm_matches_the_bf16_path`, not assumed
// correct) and re-deriving from the physical layout math rather than
// guessing again.
using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

using ElementC = float;
using LayoutC = cutlass::layout::RowMajor;
using LayoutC_Transpose = typename cutlass::layout::LayoutTranspose<LayoutC>::type;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementD = ElementC;
using LayoutD_Transpose = LayoutC_Transpose;
using AlignmentD = std::integral_constant<int, AlignmentC>;

using MmaTileShape = Shape<_128, _32, _128>;
using SwapClusterShape = Shape<_1, _1, _1>;

// The Collective builders' "A"/"B" here are the *kernel's own* operand
// roles, already swapped: element/layout A is `ElementB`/`LayoutB_Transpose`
// (the weight), element/layout B is `ElementA`/`LayoutA_Transpose` (the
// activation) -- matching vLLM's `cutlass_3x_gemm_fp8_blockwise<..., true>`
// with its `ElementA_=InType` (there, always the activation type, same
// `float_e4m3_t` as the weight in this file so the element type itself does
// not change) and its own conditional `LayoutA/LayoutB` swap.
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, MmaTileShape, SwapClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC,
    LayoutC_Transpose, AlignmentC, ElementD, LayoutC_Transpose, AlignmentD::value,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementB, cute::tuple<LayoutB_Transpose, LayoutSFA>,
    AlignmentB, ElementA, cute::tuple<LayoutA_Transpose, LayoutSFB>, AlignmentA, ElementAccumulator, MmaTileShape,
    SwapClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                                         CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideD = typename Gemm::GemmKernel::StrideD;
}  // namespace small_m_swap

// `small_m_swap`'s own tile/mainloop/epilogue with the scheduler swapped
// from `void` to `cutlass::gemm::StreamKScheduler` -- same reasoning as the
// plain `stream_k` namespace above, but applied to the tile production
// actually dispatches to at n_tokens<=32. Real motivation, not speculative:
// `swap_ab_occupancy_probe`'s own `ncu` run measured this exact tile at the
// real gate/up shape (K=5120,N=17408,n_tokens=16) at `Waves Per SM=0.72` --
// ~28% of this GPU's SMs get zero blocks for the kernel's entire duration,
// the identical structural shape as the pre-swap_ab wide-tile finding this
// file's own history already documented, just never re-checked after
// swap_ab shipped. Swapping A/B changes which axis is tiled but not the
// total block count (`stream_k`'s own bench above found real headroom at
// K=17408,N=5120's wide tile -- only 40 blocks there, an even lower wave
// count than gate/up's 136 -- while gate/up's own wide tile showed no real
// stream-K benefit at all, a genuinely shape-dependent result, not a
// scheduler no-op) -- so this is checked per-shape via a real bench, not
// assumed to help uniformly.
namespace small_m_swap_stream_k {
using GemmKernel =
    cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, small_m_swap::CollectiveMainloop,
                                          small_m_swap::CollectiveEpilogue, cutlass::gemm::StreamKScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideD = typename Gemm::GemmKernel::StrideD;
}  // namespace small_m_swap_stream_k

// Same external contract as `infero_cutlass_fp8_bw_gemm_f32out_small_m` --
// `a`/`sfa` are still the activation, `b`/`sfb` still the weight, `m`/`n`/`k`
// mean what they always do here, `d` is still `[m,n]` row-major. The swap is
// entirely internal: `a_stride`/`b_stride` are computed from each buffer's
// own real physical shape exactly as every other entry point in this file
// computes them (not swapped), only *which* pointer/stride pair lands in the
// mainloop's first vs. second operand slot is swapped, mirroring vLLM's own
// `cutlass_gemm_caller_blockwise` (`w8a8/cutlass/c3x/
// scaled_mm_blockwise_sm120_fp8_dispatch.cuh`) line for line.
extern "C" size_t infero_cutlass_fp8_bw_gemm_f32out_small_m_swap_workspace(int m, int n, int k) {
  // `a_stride`/`b_stride`: exactly like every other entry point in this
  // file, computed from each buffer's own real physical shape (`a` is
  // `[m,k]`, `b` is `[n,k]`) using the kernel's declared `StrideA`/`StrideB`
  // types. The swap happens ONLY below, in which pointer+stride pair is
  // handed to the mainloop's first ("A") vs second ("B") operand slot --
  // mirroring vLLM's own `cutlass_gemm_caller_blockwise` field-for-field
  // (`mainloop_args.dA = b_stride; mainloop_args.dB = a_stride;` in its own
  // `swap_ab` branch): the mainloop's "A" slot pairs the weight *pointer*
  // with the stride computed from `b`'s own shape, not `a`'s.
  auto a_stride = cutlass::make_cute_packed_stride(small_m_swap::StrideA{}, cute::make_shape(m, k, 1));
  auto b_stride = cutlass::make_cute_packed_stride(small_m_swap::StrideB{}, cute::make_shape(n, k, 1));
  auto stride_D = cutlass::make_cute_packed_stride(small_m_swap::StrideD{}, cute::make_shape(n, m, 1));
  auto layout_SFA = small_m_swap::ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(n, m, k, 1));
  auto layout_SFB = small_m_swap::ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(n, m, k, 1));
  typename small_m_swap::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {n, m, k, 1},
      {nullptr, b_stride, nullptr, a_stride, nullptr, layout_SFA, nullptr, layout_SFB},
      {{}, nullptr, stride_D, nullptr, stride_D}};
  return small_m_swap::Gemm::get_workspace_size(arguments);
}

extern "C" int32_t infero_cutlass_fp8_bw_gemm_f32out_small_m_swap(const void* a, const void* b, const float* sfa,
                                                                    const float* sfb, float* d, void* workspace,
                                                                    int m, int n, int k, int accum,
                                                                    cudaStream_t stream) {
  auto a_stride = cutlass::make_cute_packed_stride(small_m_swap::StrideA{}, cute::make_shape(m, k, 1));
  auto b_stride = cutlass::make_cute_packed_stride(small_m_swap::StrideB{}, cute::make_shape(n, k, 1));
  auto stride_D = cutlass::make_cute_packed_stride(small_m_swap::StrideD{}, cute::make_shape(n, m, 1));
  auto layout_SFA = small_m_swap::ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(n, m, k, 1));
  auto layout_SFB = small_m_swap::ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(n, m, k, 1));

  // Swapped: the mainloop's first operand slot ("A") gets the weight
  // pointer (`b`) paired with `b_stride` (computed from `b`'s own `[n,k]`
  // shape), the second ("B") gets the activation pointer (`a`) paired with
  // `a_stride` (from `a`'s own `[m,k]` shape) -- matching vLLM's own
  // `mainloop_args.dA = b_stride; mainloop_args.dB = a_stride;` exactly.
  // `ElementA` and `ElementB` are both `cutlass::float_e4m3_t` in this file,
  // so there is no element-type mismatch to reconcile, only which
  // buffer/stride pair lands in which slot.
  typename small_m_swap::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {n, m, k, 1},
      {static_cast<const ElementB*>(b), b_stride, static_cast<const ElementA*>(a), a_stride, sfb, layout_SFA, sfa,
       layout_SFB},
      {{}, d, stride_D, d, stride_D}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = accum ? 1.0f : 0.0f;

  small_m_swap::Gemm gemm;
  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return static_cast<int32_t>(status);
  status = gemm.initialize(arguments, workspace, stream);
  if (status != cutlass::Status::kSuccess) return static_cast<int32_t>(status);
  status = gemm.run(arguments, workspace, stream);
  return static_cast<int32_t>(status);
}

// `small_m_swap_stream_k`'s own entry points -- identical argument
// construction to `infero_cutlass_fp8_bw_gemm_f32out_small_m_swap` above
// (same swapped m/n roles, same stride/layout math), only the `Gemm` type
// differs (`StreamKScheduler` in place of `void`).
extern "C" size_t infero_cutlass_fp8_bw_gemm_f32out_small_m_swap_stream_k_workspace(int m, int n, int k) {
  auto a_stride = cutlass::make_cute_packed_stride(small_m_swap_stream_k::StrideA{}, cute::make_shape(m, k, 1));
  auto b_stride = cutlass::make_cute_packed_stride(small_m_swap_stream_k::StrideB{}, cute::make_shape(n, k, 1));
  auto stride_D = cutlass::make_cute_packed_stride(small_m_swap_stream_k::StrideD{}, cute::make_shape(n, m, 1));
  auto layout_SFA = small_m_swap::ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(n, m, k, 1));
  auto layout_SFB = small_m_swap::ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(n, m, k, 1));
  typename small_m_swap_stream_k::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {n, m, k, 1},
      {nullptr, b_stride, nullptr, a_stride, nullptr, layout_SFA, nullptr, layout_SFB},
      {{}, nullptr, stride_D, nullptr, stride_D}};
  return small_m_swap_stream_k::Gemm::get_workspace_size(arguments);
}

extern "C" int32_t infero_cutlass_fp8_bw_gemm_f32out_small_m_swap_stream_k(const void* a, const void* b,
                                                                            const float* sfa, const float* sfb,
                                                                            float* d, void* workspace, int m, int n,
                                                                            int k, int accum, cudaStream_t stream) {
  auto a_stride = cutlass::make_cute_packed_stride(small_m_swap_stream_k::StrideA{}, cute::make_shape(m, k, 1));
  auto b_stride = cutlass::make_cute_packed_stride(small_m_swap_stream_k::StrideB{}, cute::make_shape(n, k, 1));
  auto stride_D = cutlass::make_cute_packed_stride(small_m_swap_stream_k::StrideD{}, cute::make_shape(n, m, 1));
  auto layout_SFA = small_m_swap::ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(n, m, k, 1));
  auto layout_SFB = small_m_swap::ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(n, m, k, 1));

  typename small_m_swap_stream_k::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {n, m, k, 1},
      {static_cast<const ElementB*>(b), b_stride, static_cast<const ElementA*>(a), a_stride, sfb, layout_SFA, sfa,
       layout_SFB},
      {{}, d, stride_D, d, stride_D}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = accum ? 1.0f : 0.0f;

  small_m_swap_stream_k::Gemm gemm;
  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return static_cast<int32_t>(status);
  status = gemm.initialize(arguments, workspace, stream);
  if (status != cutlass::Status::kSuccess) return static_cast<int32_t>(status);
  status = gemm.run(arguments, workspace, stream);
  return static_cast<int32_t>(status);
}

extern "C" size_t infero_cutlass_fp8_bw_gemm_f32out_small_m_workspace(int m, int n, int k) {
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  auto stride_D = cutlass::make_cute_packed_stride(small_m::StrideD{}, cute::make_shape(m, n, 1));
  auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  typename small_m::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
      {{}, nullptr, stride_D, nullptr, stride_D}};
  return small_m::Gemm::get_workspace_size(arguments);
}

// Same contract as `infero_cutlass_fp8_bw_gemm_f32out` (`d` is the model's
// own `out` buffer, `accum` selects CUTLASS's own beta=1 accumulate) -- only
// the tile shape differs. Caller (Rust side) picks this over the plain
// entry point below some `n_tokens` threshold; `can_implement` is still
// checked here rather than assumed, so a shape this tile genuinely cannot
// cover fails loudly instead of silently.
extern "C" int32_t infero_cutlass_fp8_bw_gemm_f32out_small_m(const void* a, const void* b, const float* sfa,
                                                              const float* sfb, float* d, void* workspace, int m,
                                                              int n, int k, int accum, cudaStream_t stream) {
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  auto stride_D = cutlass::make_cute_packed_stride(small_m::StrideD{}, cute::make_shape(m, n, 1));
  auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  typename small_m::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {static_cast<const ElementA*>(a), stride_A, static_cast<const ElementB*>(b), stride_B, sfa, layout_SFA, sfb,
       layout_SFB},
      {{}, d, stride_D, d, stride_D}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = accum ? 1.0f : 0.0f;

  small_m::Gemm gemm;
  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return static_cast<int32_t>(status);
  status = gemm.initialize(arguments, workspace, stream);
  if (status != cutlass::Status::kSuccess) return static_cast<int32_t>(status);
  status = gemm.run(arguments, workspace, stream);
  return static_cast<int32_t>(status);
}

// Genuine per-architecture kernel bodies for Hopper (SM90) and Blackwell
// datacenter (SM100) -- NOT the same device code as the SM120 kernel above
// recompiled with a different `-gencode`. CUTLASS's collective builders pick
// materially different mainloop implementations per `cutlass::arch::SmXX`
// tag (Hopper's `wgmma`+TMA warp-specialized path vs. SM120's own MMA atoms),
// so each of these is its own real `GemmKernel` C++ type / device function,
// not a recompilation of the SM120 one. Selecting which to actually call is
// a runtime dispatch on the host side (see `infero_cutlass_fp8_bw_gemm_f32out`
// below), keyed on the detected GPU's real compute capability -- this file
// still produces one fat object with `-gencode` entries for every target
// architecture; nvcc compiles each `SmXX`-tagged kernel's device code only
// for the `-gencode` targets it's actually valid for (CUTLASS's own
// `__CUDA_ARCH__`-gated kernel bodies make the other targets in the same
// translation unit safe no-ops rather than compile failures -- this is the
// same "one source, several `-gencode`s, runtime-select the real one" shape
// CUTLASS's own multi-arch examples use).
//
// Execution-verified: SM120 only (the only hardware this box has). SM90/SM100
// are verified by real compile against the vendored CUTLASS headers plus
// matching CUTLASS's own real per-architecture type names
// (`Sm90BlockwiseScaleConfig`, `MainloopSm90TmaGmmaWarpSpecializedBlockwiseFP8`
// via `KernelTmaWarpSpecializedCooperativeFP8Blockwise`;
// `Sm100BlockwiseScaleConfig`, `KernelScheduleSm100Blockwise`) -- NOT
// execution-tested. Say so explicitly wherever this is reported.
namespace sm90 {
namespace f32out {
using ElementC = float;
using LayoutC = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementD = ElementC;
using AlignmentD = std::integral_constant<int, AlignmentC>;

using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, CooperativeMmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC, LayoutC,
    AlignmentC, ElementD, LayoutC, AlignmentD::value,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB, ElementAccumulator, CooperativeMmaTileShape_MNK,
    ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8Blockwise>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                                         CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideD = typename Gemm::GemmKernel::StrideD;
}  // namespace f32out
}  // namespace sm90

namespace sm100 {
namespace f32out {
using ElementC = float;
using LayoutC = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementD = ElementC;
using AlignmentD = std::integral_constant<int, AlignmentC>;

using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, CooperativeMmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC, LayoutC,
    AlignmentC, ElementD, LayoutC, AlignmentD::value,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB, ElementAccumulator, CooperativeMmaTileShape_MNK,
    ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelScheduleSm100Blockwise>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                                         CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideD = typename Gemm::GemmKernel::StrideD;
}  // namespace f32out
}  // namespace sm100

extern "C" size_t infero_cutlass_fp8_bw_gemm_f32out_workspace(int m, int n, int k) {
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  auto stride_D = cutlass::make_cute_packed_stride(f32out::StrideD{}, cute::make_shape(m, n, 1));
  auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  typename f32out::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
      {{}, nullptr, stride_D, nullptr, stride_D}};
  return f32out::Gemm::get_workspace_size(arguments);
}

// `d` is the model's own `out` buffer (f32), read as C too when `accum`
// (beta=1) -- no separate scratch, no separate store/upconvert kernel after
// this returns.
extern "C" int32_t infero_cutlass_fp8_bw_gemm_f32out(const void* a, const void* b, const float* sfa,
                                                      const float* sfb, float* d, void* workspace, int m, int n,
                                                      int k, int accum, cudaStream_t stream) {
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  auto stride_D = cutlass::make_cute_packed_stride(f32out::StrideD{}, cute::make_shape(m, n, 1));
  auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  typename f32out::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {static_cast<const ElementA*>(a), stride_A, static_cast<const ElementB*>(b), stride_B, sfa, layout_SFA, sfb,
       layout_SFB},
      {{}, d, stride_D, d, stride_D}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = accum ? 1.0f : 0.0f;

  f32out::Gemm gemm;
  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return static_cast<int32_t>(status);
  status = gemm.initialize(arguments, workspace, stream);
  if (status != cutlass::Status::kSuccess) return static_cast<int32_t>(status);
  status = gemm.run(arguments, workspace, stream);
  return static_cast<int32_t>(status);
}

// `stream_k` namespace's own entry points -- same wide `<128,128,128>` tile
// and f32-direct epilogue as `infero_cutlass_fp8_bw_gemm_f32out` above, only
// the scheduler differs (`StreamKScheduler` vs `void`/persistent). Kept as a
// distinct pair of functions (not a runtime flag on the existing one) so the
// existing default entry point is completely unchanged -- same reasoning as
// the sm90/sm100 variants just below.
extern "C" size_t infero_cutlass_fp8_bw_gemm_f32out_stream_k_workspace(int m, int n, int k) {
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  auto stride_D = cutlass::make_cute_packed_stride(stream_k::StrideD{}, cute::make_shape(m, n, 1));
  auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  typename stream_k::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
      {{}, nullptr, stride_D, nullptr, stride_D}};
  return stream_k::Gemm::get_workspace_size(arguments);
}

extern "C" int32_t infero_cutlass_fp8_bw_gemm_f32out_stream_k(const void* a, const void* b, const float* sfa,
                                                                const float* sfb, float* d, void* workspace, int m,
                                                                int n, int k, int accum, cudaStream_t stream) {
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  auto stride_D = cutlass::make_cute_packed_stride(stream_k::StrideD{}, cute::make_shape(m, n, 1));
  auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  typename stream_k::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {static_cast<const ElementA*>(a), stride_A, static_cast<const ElementB*>(b), stride_B, sfa, layout_SFA, sfb,
       layout_SFB},
      {{}, d, stride_D, d, stride_D}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = accum ? 1.0f : 0.0f;

  stream_k::Gemm gemm;
  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return static_cast<int32_t>(status);
  status = gemm.initialize(arguments, workspace, stream);
  if (status != cutlass::Status::kSuccess) return static_cast<int32_t>(status);
  status = gemm.run(arguments, workspace, stream);
  return static_cast<int32_t>(status);
}

// Hopper (SM90) and Blackwell-datacenter (SM100) f32out variants -- real,
// distinctly-typed `GemmKernel`s (see the `namespace sm90`/`sm100` comment
// above), not the SM120 kernel above recompiled. The caller (Rust side)
// picks which of these three `_sm90`/`_sm100`/plain-SM120 entry points to
// call based on the real detected GPU compute capability -- no C-level
// dispatch here, kept as three plain, separately-named functions so the
// existing SM120 entry points above are completely unchanged (zero
// behavior/ABI change on the one architecture this can actually be tested
// on).
extern "C" size_t infero_cutlass_fp8_bw_gemm_f32out_workspace_sm90(int m, int n, int k) {
  auto stride_A = cutlass::make_cute_packed_stride(sm90::f32out::StrideA{}, cute::make_shape(m, k, 1));
  auto stride_B = cutlass::make_cute_packed_stride(sm90::f32out::StrideB{}, cute::make_shape(n, k, 1));
  auto stride_D = cutlass::make_cute_packed_stride(sm90::f32out::StrideD{}, cute::make_shape(m, n, 1));
  auto layout_SFA = sm90::f32out::ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = sm90::f32out::ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  typename sm90::f32out::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
      {{}, nullptr, stride_D, nullptr, stride_D}};
  return sm90::f32out::Gemm::get_workspace_size(arguments);
}

extern "C" int32_t infero_cutlass_fp8_bw_gemm_f32out_sm90(const void* a, const void* b, const float* sfa,
                                                           const float* sfb, float* d, void* workspace, int m, int n,
                                                           int k, int accum, cudaStream_t stream) {
  auto stride_A = cutlass::make_cute_packed_stride(sm90::f32out::StrideA{}, cute::make_shape(m, k, 1));
  auto stride_B = cutlass::make_cute_packed_stride(sm90::f32out::StrideB{}, cute::make_shape(n, k, 1));
  auto stride_D = cutlass::make_cute_packed_stride(sm90::f32out::StrideD{}, cute::make_shape(m, n, 1));
  auto layout_SFA = sm90::f32out::ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = sm90::f32out::ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  typename sm90::f32out::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {static_cast<const ElementA*>(a), stride_A, static_cast<const ElementB*>(b), stride_B, sfa, layout_SFA, sfb,
       layout_SFB},
      {{}, d, stride_D, d, stride_D}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = accum ? 1.0f : 0.0f;

  sm90::f32out::Gemm gemm;
  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return static_cast<int32_t>(status);
  status = gemm.initialize(arguments, workspace, stream);
  if (status != cutlass::Status::kSuccess) return static_cast<int32_t>(status);
  status = gemm.run(arguments, workspace, stream);
  return static_cast<int32_t>(status);
}

extern "C" size_t infero_cutlass_fp8_bw_gemm_f32out_workspace_sm100(int m, int n, int k) {
  auto stride_A = cutlass::make_cute_packed_stride(sm100::f32out::StrideA{}, cute::make_shape(m, k, 1));
  auto stride_B = cutlass::make_cute_packed_stride(sm100::f32out::StrideB{}, cute::make_shape(n, k, 1));
  auto stride_D = cutlass::make_cute_packed_stride(sm100::f32out::StrideD{}, cute::make_shape(m, n, 1));
  auto layout_SFA = sm100::f32out::ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = sm100::f32out::ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  typename sm100::f32out::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
      {{}, nullptr, stride_D, nullptr, stride_D}};
  return sm100::f32out::Gemm::get_workspace_size(arguments);
}

extern "C" int32_t infero_cutlass_fp8_bw_gemm_f32out_sm100(const void* a, const void* b, const float* sfa,
                                                            const float* sfb, float* d, void* workspace, int m, int n,
                                                            int k, int accum, cudaStream_t stream) {
  auto stride_A = cutlass::make_cute_packed_stride(sm100::f32out::StrideA{}, cute::make_shape(m, k, 1));
  auto stride_B = cutlass::make_cute_packed_stride(sm100::f32out::StrideB{}, cute::make_shape(n, k, 1));
  auto stride_D = cutlass::make_cute_packed_stride(sm100::f32out::StrideD{}, cute::make_shape(m, n, 1));
  auto layout_SFA = sm100::f32out::ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = sm100::f32out::ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  typename sm100::f32out::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {static_cast<const ElementA*>(a), stride_A, static_cast<const ElementB*>(b), stride_B, sfa, layout_SFA, sfb,
       layout_SFB},
      {{}, d, stride_D, d, stride_D}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = accum ? 1.0f : 0.0f;

  sm100::f32out::Gemm gemm;
  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return static_cast<int32_t>(status);
  status = gemm.initialize(arguments, workspace, stream);
  if (status != cutlass::Status::kSuccess) return static_cast<int32_t>(status);
  status = gemm.run(arguments, workspace, stream);
  return static_cast<int32_t>(status);
}
