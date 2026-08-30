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
