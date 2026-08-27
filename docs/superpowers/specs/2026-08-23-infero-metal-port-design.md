# 把 tuili 移植到 Apple GPU：设备抽象层与最小 kernel 集

2026-08-23

## 为什么

tuili 是纯 CUDA 引擎。`crates/{kernels,model,server}` 无条件依赖 `tuili-cuda` +
`cudarc`（`driver`/`nvrtc`/`cublas`），feature 只有 `cuda-12`/`cuda-13`，没有可以
关掉的开关。在 macOS 上 `cargo check -p tuili-cuda` 直接 panic 在 `build.rs:12`，
而 macOS 自 CUDA 10.2（2019）起就没有驱动，Apple Silicon 从来没有过。

这份设计交付**一条能在 Apple GPU 上端到端跑出正确 logits 的路**，以及一个后续
kernel 逐个往里填的框架。它不交付 27B——见末尾「不包含什么」。

## 量出来的现状

```
Rust 总计                       52,826 行
  平台无关（原样保留）          32,286 行  gguf · safetensors · tokenizer
                                          model 的配置/调度 · server · tui
CUDA 源码                       18,587 行  .cu/.cuh 14,222 + vendored marlin 4,365
  93 个 __global__ kernel · 17 处内联 PTX · 20 处 cuBLAS 调用

cudarc 类型引用   kernels 454 · model 232 · server 0
  CudaView 233 · CudaSlice 181 · CudaViewMut 145
  LaunchConfig 76 · PushKernelArg 4 · CudaStream 2
launch 站点       160
```

`server` 和 `tui` 已经零 cudarc 引用，不在重构面内。`tuili-tui` 今天就能在 macOS
上编译（实测 12.98s）。

## 一、设备层的形状

缝切在**设备层**，不是算子层：`kernels` 那 19,836 行 host 逻辑（网格计算、分支、
调优表）两边共用，kernel 源码两套。

```
crates/gpu/           trait Backend + Buffer/Stream/Module 抽象
crates/cuda/          impl Backend for CudaBackend（现有 704 行改造）
crates/metal/         impl Backend for MetalBackend
crates/kernels/       泛型于 Backend，host 逻辑共用
    src/cu/*.cu       保留不动
    src/msl/*.metal   新增
```

语义对照，逐项验证过：

| CUDA | Metal | 落到 trait |
|---|---|---|
| NVRTC 源码 → PTX → `CUmodule` | `newLibraryWithSource:` → `MTLLibrary` | 直接对应 |
| `CUfunction` | `MTLFunction` → `MTLComputePipelineState` | Metal 多一步，`get_fn()` 内部缓存 pipeline |
| `cuLaunchKernel(grid, block, smem, stream, void**)` | `dispatchThreadgroups:threadsPerThreadgroup:` | grid/block 直译 |
| 动态 shared memory（launch 参数） | `setThreadgroupMemoryLength:atIndex:` | 可映射，Metal 按 index 绑 |
| `void**` 参数数组 | `setBuffer:offset:atIndex:` / `setBytes:` | 唯一需要改调用侧的地方 |
| `CUstream` | `MTLCommandQueue` → `MTLCommandBuffer` → encoder → `commit` | 默认 `MTLDispatchTypeSerial` 会在 dispatch 间自动插 barrier，**语义与 CUDA stream 一致** |
| `cuFuncGetAttribute`（寄存器/occupancy） | `maxTotalThreadsPerThreadgroup` / `threadExecutionWidth` | 两边回答的不是同一件事 |

**Apple 的 simdgroup 宽度是 32，和 CUDA warp 相同。** 所以 `__shfl_xor_sync` 那套
warp 内归约直接对应 `simd_shuffle_xor` / `simd_sum`——而首批 kernel 里 `rms_norm`、
`gemv`、`attn_softmax` 全是归约形状，这批最接近直译。

调优探针（`kernel_registers` / `occupancy_blocks` / `ldmatrix_probe` /
`mmq_bw_probe`）留在 **CUDA-only 扩展 trait**，不进公共接口。代价：Metal 侧短期不做
occupancy 自动调优，网格配置先写死常量。

`crates/metal` 用 `objc2-metal 0.3`。

### 两个让重构小一个数量级的决定

**静态分派，不用 `dyn Backend`。** `server/src/engine.rs:188` 记着 draft step
"runs a dozen kernels at one row each, so it is launch-bound"——9–10 ms 对着 3.2 的
内存下界。这个代码库明确在乎 launch 开销，没理由在最热路径上加间接层。泛型单态化。

**`model` 不加 `<B>` 类型参数，用 cfg 选类型别名。**

```rust
#[cfg(feature = "cuda")]  pub type Gpu = tuili_cuda::CudaBackend;
#[cfg(feature = "metal")] pub type Gpu = tuili_metal::MetalBackend;
pub type Slice<T> = <Gpu as Backend>::Slice<T>;
```

理由：**没有任何一台机器同时有两个后端**——Mac 没 CUDA，lenserver 没 Metal，一个
二进制只需要一个。于是 `model` 那 232 处引用变成别名重命名（`CudaSlice<f32>` →
`Slice<f32>`），编译器全程守着，而不是把 `<B>` 撒进 24,362 行。

**launch builder API 镜像 cudarc 的 `.arg()` 链**，让 CUDA 实现是薄转发，160 个调用
站点只改「怎么拿到 builder」这一句。

## 二、kernel 源码组织，与 mmq 的绕法

### 绕过 mmq 不需要新增分派逻辑

`model/src/lib.rs:3320` 已经有：

```rust
let mmq_ok = use_mmq && Kernels::has_mmq(w.ty)
    && kern.device().arch() >= 80        // ← 当年为 Ampere 之前的卡写的
    && Self::mmq_shape_ok(w);
```

`arch() >= 80` 全仓库 5 处（`model/lib.rs` 2、`kernels/lib.rs` 3）。绕过 tensor-core
GEMM 的完整路径**已经写好且被测过**。设备层只把它换成能力查询：

```rust
caps().int_tensor_gemm      // CUDA: arch>=80。Metal: 初期 false
```

Q8_0 在 `int_tensor_gemm = false` 下的落点（`GEMM_THRESHOLD = 4`,
`MMVQ_REPEAT_MAX = 12`）：

| n_tokens | 走哪条 | 需要 |
|---|---|---|
| 1 | `mmvq` | `quantize_q8_1` + Q8_0 mat-vec |
| 2–3 | `mmvq_batch` | 批量变体 |
| 4–12 | `mmvq` 逐 token 重复 | 无新 kernel |
| 13+ | `to_f16` + `dequant_to_f16` + `gemm_f16` | ← 库调用 |

`gemm_f16` 对应 MPS 的 `MPSMatrixMultiplication`——**两边都是调库**。手写的
tensor-core 整数 GEMM（`mmq.cu` 7,525 行）不碰，靠能力位关掉。

### 源码组织

现状是 `include_str!` 进来、按模块拼字符串、NVRTC 启动时编译、`OnceLock` 缓存
（`ops_src()` = `common.cuh` + `mma.cuh` + `ops.cu`）。这个结构原样搬到 Metal——
`newLibraryWithSource:` 吃的也是字符串，模块划分一比一保留，「改 gdn 不触发其他
kernel 重编」这个好处也留着。

`msl/common.metal` 先铺地基，剩下的机械：

```
block_reduce_sum()   → simd_sum + threadgroup 内存
block_q8_0 等块结构  → 逐字节照搬（MSL 原生有 half）
GEMV_PROLOGUE/SPREAD/EPILOGUE → 宏三件套照搬，整个 gemv 家族靠它生成
```

逐 kernel 差异只有四类，全机械：`blockIdx.x` → `threadgroup_position_in_grid`、
`threadIdx.x` → `thread_position_in_threadgroup`、指针加 `device`/`threadgroup`
地址空间、参数挂 `[[buffer(n)]]`。

**决定：`.cu` 和 `.metal` 是两份独立源码，文件对文件镜像。** 不用宏把 kernel 体做成
一份两边套壳——MSL 的地址空间和参数属性写在签名和指针上，包不干净，最后是一堆没人
想读的宏；而这个仓库的风格是长注释、明写、可读优先。代价是以后每个新 kernel 写两遍。

## 三、正确性策略

仓库自己的四级阶梯，每级 oracle 都平台无关：

| 级 | 靶子 | 在哪跑 |
|---|---|---|
| 1 tokenizer | 逐 token 对齐 HF | ✅ Mac，已通过（7 测试） |
| 2 F16 前向 | `transformers` f32 logits（`fixtures/qwen2.5-0.5b-instruct-logits.json`，4 case，top_k=20，带 argmax/mean/std） | Mac / Metal |
| 3 量化解码 | 同一 checkpoint 的 F16 构建 | Mac 自比 |
| 4 单 kernel | CPU 参考实现（`kernels/tests/*.rs`，10,678 行） | Mac / Metal |

### F16 先行，Q8_0 紧随

`forward.rs` 的默认就是 fp16，注释写着 "comparing a quantized build against f32
reference logits would measure the quantizer, not the engine"。移植第一次跑，要测的
是引擎而不是量化器。且 F16 路径 kernel 更少：不需要 `quantize_q8_1`/`mmvq`/
`mmvq_batch`/`dequant_to_f16`；F16 权重进 `gemm_f16` 是 `transmute` 重解释。

```
F16 里程碑   ~15 个手写 kernel + 1 个 MPS 调用
             batch=1 且 prompt ≤ 4 token 时连 MPS 都不需要（走 gemv_f16）
Q8_0 叠加    再 4 个，靶子是本机 F16 构建，完全不依赖外部参照
```

### 测试如何同时服务两个后端

测试里已有两种 skip：文件缺失（`forward.rs:44`）和**能力不足**（`quant.rs:382`：
"sm_{} predates the int8 mma, skipped"）。第二种正是所需机制：测试助手泛型于
`Backend`，一份源码两台机器跑，`mma.rs`/`tma.rs`/`fp8_matvec.rs`/`ldmatrix.rs` 在
Metal 下按能力位自动跳过——不是注释掉，不是 `#[ignore]`。

### 一条必须记住的风险

**Metal 的 logits 不会和 CUDA 逐位相同。** `simd_sum` 的归约顺序与 CUDA shuffle 归约
不同，F16 累加不满足结合律。fixtures 的判据是 argmax + top-20 集合 + mean/std，能
容忍；但 README 里的 batch invariance 性质
（`tensor_core_gemm_gives_the_same_answer_at_any_batch_size`）是 CUDA tensor core
特定的，**Metal 侧不承诺**。拿 CUDA 的不变性断言测 Metal 会莫名其妙地红。

## 四、阶段

| | 干什么 | 在哪验 | 验收 |
|---|---|---|---|
| S0 | 基线固定：stash WIP，录全量测试输出 + 两个 0.5B 模型的贪心 token 流 | lenserver | 基线文件进 git |
| S1 | 抽 trait，CUDA 是唯一实现，一行 Metal 都没有 | Mac 编译 + lenserver 跑 | 全绿，**逐位对上 S0** |
| S2 | Metal 骨架：device / buffer / 源码编译 / pipeline 缓存 / encoder，零 kernel | Mac | 一个 `add_f32` 过 `ops.rs` 自己的测试 |
| S3 | F16 的 ~15 个 kernel。`common.metal` 先行 → 归约 → elementwise → rope/kv → attn → `gemv_f16` | Mac | **`forward.rs` 四 case 过 ← 里程碑 1** |
| S4 | MPS `gemm_f16` | Mac | prefill > 4 token 的长提示词 |
| S5 | Q8_0 那 4 个 kernel | Mac | 对齐本机 F16 构建 |

S1 是唯一有「改坏现有东西」风险的阶段，所以它的设计是**不碰 kernel 源码、不碰网格
配置、不碰参数值**——只搬类型和调用形式。CUDA 输出逐位不变是可以要求的，S0 基线是
判据。

验收标准（里程碑 1）：

1. Mac 上 `forward.rs` 四个 case 在 Metal 通过（F16）
2. Mac 上 Q8_0 构建对齐本机 F16 构建
3. Mac 上 `cargo test -p tuili-kernels` 涉及的 kernel 全绿，其余按能力跳过
4. lenserver 上 CUDA 全绿且逐位对上 S0 基线
5. `examples/generate` 在 Metal 上出一段通顺文本

## 环境事实

- **本机**：Mac Studio M4 Max，36 GB 统一内存，32 核 GPU，Metal 4。Metal 默认工作集
  上限约 27 GB（`iogpu.wired_limit_mb` 可调）
- **lenserver**（`ssh jeffwang@192.168.0.110`）：单卡 RTX A4000 16 GB（已用 4.4 GB），
  CUDA 13.1 驱动，**sm_86**。有 `mma.sync.m16n8k16`/`cp.async`/`ldmatrix`，
  **没有 FP8**（需 sm_89+）也**没有 TMA**（需 sm_90+）。仓库在 `~/work/tuili`
- 27B（29–51 GB）在 lenserver 上跑不了；FP8/TMA/27B 路径本来就不在这台的回归范围内
- 模型文件在 lenserver `models/`（→ `/mnt/data/tuili-models`）：
  `qwen2.5-0.5b-instruct-fp16.gguf` 1.27 GB、`qwen2.5-0.5b-instruct-q8_0.gguf` 675 MB
- 在 macOS 上给 CUDA crate 做编译检查：`TUILI_CUDA_DIR=<两个空目录> cargo check
  --workspace --all-targets` — 12.05s 通过。cudarc 开了 `dynamic-loading`，
  `libcuda` 是运行时 dlopen，编译期不链接

## 进度（2026-08-24）

**S0 基线** ✅ lenserver 上 WIP 已 stash，`cargo test --workspace --release` 63 个测试块
全 ok、0 失败；两个 0.5B 模型的贪心输出已录。（坑：非交互 ssh 的 PATH 里没有 cargo，
要显式 `$HOME/.cargo/bin`。）

**S2 Metal 骨架** ✅ `crates/metal`：device / buffer / msl / launch 四个文件。7 个运行时
测试在 M4 Max 上通过（simd 32、working set 28.1 GiB）。其中一个测试抓到真 bug：
pipeline 缓存漏了源码哈希，改了源码会拿到旧 pipeline。

**S3 F16 kernel** ✅ 9 个 MSL kernel（`crates/kernels/src/msl/{common,ops}.metal`），
7 个对 CPU 参考的单测通过。**四个 fixture case 全部匹配参考 logits**——argmax 逐个精确、
top-10 重合 9~10/10、std 差 ≤0.06。`--prompt` 能生成连贯文本，decode 107 tok/s
（A4000 上 CUDA fp16 是 121.9 tok/s）。

那个唯一的 bug 值得记住：7 个 kernel 单测全过而端到端 logits 全错（量级对、token 全不同），
原因在组合层——24 层共用了一个 KV 平面，于是每层都在读第 23 层的历史。
**是 `--cpu` 那个宿主仲裁器一次定位的**：kernel 测试只能发现 kernel 的错，
发现接线错误需要一个独立的整模型参考。

**尚未开始**：S1（把 `kernels`/`model` 泛型化，686 处类型引用 + 160 个 launch 站点，
这是把垂直切片折进真引擎的那一步）、S4（MPS `gemm_f16`，让 prefill 不必逐 token 走）、
S5（Q8_0）。现在的 demo 是 `crates/metal/examples/qwen2_f16.rs`，**它不是引擎**：
没有调度器、没有分页 KV 池、没有批处理。

## 不包含什么

做完 S5，**27B 还跑不了**。还差三样，每样值得独立一份 spec：

1. **INT4 GEMM**（`vendor/marlin` 4,365 行的对等物，从零写，Metal 上无可 vendor 之物）
   ← 27B 装进 36 GB 的唯一途径（权重 29 GB → 16 GB）
2. **GatedDeltaNet**（`gdn.cu` 646 行、8 个 kernel）← 27B 的 48/64 层是这个
3. `mmq` 的 `simdgroup_matrix` 版、FP8 的替代方案（Apple GPU 无 FP8 矩阵指令）、
   vision tower

27B 的 KV 只有 64 KiB/token（混合架构，16/64 层 full attention），所以
32K → 2 GiB、128K → 8 GiB、262144 → 16 GiB。16 GB 权重 + 8 GiB KV ≈ 24 GB，128K 在
36 GB 机器上擦边可行；原生 262144 需要约 32 GB，装不下。
