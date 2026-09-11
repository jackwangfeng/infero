# infero

[English](README.md)

一个用 Rust 编写的推理引擎，支持 GGUF、AWQ、原生 FP8（W8A8）以及原生
NVFP4（W4A4）四种 checkpoint，内核为 NVIDIA（CUDA）和 Apple Silicon（Metal）
两种 GPU 都手写了实现——不依赖 PyTorch、不依赖 `libtorch`、也不依赖 ggml。
CUDA 是主要的、最完整的后端；Metal 目前覆盖了稠密解码、GQA 注意力和
GatedDeltaNet，而 MoE、视觉塔，以及 INT4/FP8/NVFP4 张量核心 GEMM 路径仍然
只有 CUDA 有（见下文 [Metal](#也支持-apple-gpumetal) 一节）。从磁盘上的
一个模型 checkpoint 到一个兼容 OpenAI 的 HTTP 响应，整条链路都在这个仓库里。

<p align="center"><img src="docs/images/demo.png" width="700" alt="infero 通过兼容 OpenAI 的接口提供 GGUF 模型服务"></p>

官方 `openai` Python SDK 不用改动就能直接对接，流式输出也支持。

## 现状

可以运行 Qwen2、Llama 系列、Qwen3-MoE，以及 Qwen3.5 风格的混合
注意力/GatedDeltaNet（线性注意力）模型，格式覆盖 GGUF、AWQ、原生
FP8（W8A8）和原生 NVFP4（W4A4），可以单卡跑，也可以用张量并行分片到多张
GPU 上。正确性不是靠肉眼观察，而是对照参考实现逐项校验的：分词器与
Hugging Face 做逐 token 比对，量化解码器与同一 checkpoint 的 F16 版本比对，
前向过程与 `transformers` 的 logits 比对。

KV 缓存可以用 TurboQuant 压缩；这套压缩在这个模型上实际能换来什么，见下文。

请求通过分页 KV 缓存上的连续批处理（continuous batching）提供服务，CUDA
Graph 把一个解码 step 里几百个内核启动重放成一次；层也可以卸载到主机内存
以便让模型塞进更小的显存里；已完成的 prompt 前缀会跨请求缓存，一段共享的
system prompt 或一场多轮对话只需要为新增的 token 付费（带 GatedDeltaNet
循环层的模型关掉了前缀缓存，因为共享前缀重建不出它的循环状态）。

**不只是纯文本解码：**

- **MoE。** 稀疏 FFN 架构（Qwen3-MoE 及同类）会逐个专家加载——每个专家
  独立用 AWQ 或 FP8 量化——解码时走一个专门的 top-k 路由内核，预填时走
  计数排序后逐专家 GEMM 的路径。
- **混合线性注意力（GatedDeltaNet）。** Qwen3.5 风格的 checkpoint 会把
  普通 GQA 注意力层和 GatedDeltaNet 层交替排列——每个序列一份固定大小的
  循环状态，每一步整体覆写而不是随上下文变长，按 gated delta rule 更新
  （`crates/model/src/qwen35.rs`、`crates/model/src/gdn_state.rs`）。
  分页 KV 池会为每个槽位同步分配一份对应的 GDN 状态数组，跟普通注意力的页
  并排存放。
- **原生 NVFP4（W4A4）。** 一个从头到尾整体量化的 checkpoint——注意力、
  GatedDeltaNet 和 FFN 投影全都量化，不只是 FFN——可以直接读取它自己真实的
  `hf_quant_config.json` 加载运行（两种真实的 schema 都支持：
  `quantized_layers` 白名单形式，和更新的 `quant_algo` + `exclude_modules`
  黑名单形式），走专门的 CUTLASS FP4 GEMM 路径，中间没有反量化再重新量化
  的一步（`crates/model/src/weights.rs` 里的 `Fp4Targets`）。
- **视觉与视频。** Qwen3.5-VL 风格的 checkpoint 可以在同一个
  chat-completions 接口里接收 `image_url` 和 `video_url` 内容分片，用
  M-RoPE 处理由此产生的三轴位置编码，视觉占位符 token 会跨 step 分块
  预填，长视频还有内容感知的 token 剪枝
  （`crates/model/src/qwen35_vision*.rs`，`crates/server/src/video.rs`）。
- **投机解码。** GGUF 内嵌或旁挂（sidecar）的 MTP 头会在主模型前面预先
  起草 `k` 个 token，除了主机端路径之外还有一条设备端常驻的
  Gumbel-max 起草路径可选；`INFERO_SPEC_K` 控制起草深度，设成 `0` 就是关闭
  （`crates/model/src/spec.rs`、`crates/model/src/mtp.rs`）。
- **张量并行。** `--tensor-parallel-size N` 通过 NCCL 把模型切分到 `N` 张
  GPU 上，一个进程对应一个 rank（`crates/model/examples/tp_generate.rs`、
  `docs/superpowers/specs/2026-09-05-tensor-parallel-design.md`）。在
  `N > 1` 时视觉/视频请求、M-RoPE 和投机解码还不支持——服务端会直接拒绝
  这类请求，而不是悄悄处理错。
- **GPU 端采样。** 惩罚项统计、top-k/top-p 以及采样本身，在覆盖到的
  batch 形状下会跑在一个设备端内核里；覆盖不到的形状会退回主机路径——
  是同一套分布，不是退化成另一种采样
  （`Kernels::sample_rows`/`sample_rows_split`/`sample_rows_greedy`）。
- **工具调用。** 支持 OpenAI 风格的 `tools`/`tool_choice`，会从模型自己
  的输出里扫描出 `<tool_call>` 标签，转成结构化的 `tool_calls` 返回，
  流式场景也支持（`crates/server/src/tool_call.rs`）。

**暂不支持：** 分片 GGUF 文件（`*-00001-of-0000N.gguf`）、原生 GPTQ
checkpoint、张量并行下的视觉/视频/投机解码请求。

**精确意义上的 batch 不变性。** 有两条性质是严格成立、并且在测试里断言而非
假设的：

- 一个请求的 logits 不依赖于*同一批次里还有哪些其他请求*
  （`a_batch_does_not_leak_between_its_members`）。
- 张量核心 GEMM 在任意 batch 宽度下都给出逐位相同的结果，所以在每种行数下
  都会用到它的词表投影（vocab projection）是不变的
  （`tensor_core_gemm_gives_the_same_answer_at_any_batch_size`）。

不严格成立的是：单 token 的一步和多 token 的一步之间，各层投影会切换
内核，因为单 token 时整数 mat-vec 明显比张量核心 GEMM 快。把两者统一起来
会在单请求延迟上付出这个代价，而这恰恰是这个引擎存在的场景，所以这个切换
留了下来。两种内核对 `k` 的求和顺序不同，所以贪心解码在极小的数值差距上
最终可能会选到另一边。带温度的固定种子采样在固定 batch 宽度下是可复现的，
跨宽度则不是。

## 快速开始

```bash
./scripts/setup-cuda.sh                     # 把一份 CUDA 用户态链接进 vendor/
mkdir -p models && cd models
curl -LO https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf
cd ..

cargo run --release -p infero-server -- --model models/qwen2.5-0.5b-instruct-q8_0.gguf
```

自带一个终端客户端：

```bash
cargo run --release -p infero-tui -- --host 127.0.0.1:8080
```

token 生成过程中实时流式显示，每条回复都会显示 tok/s，按 `esc` 可以在生成
过程中取消——这会断开连接，调度器会在下一个 step 把这个序列从 batch 里
退役，而不是在无人接收的情况下继续跑完。它说的是纯粹的 OpenAI SSE，所以
对着任何实现了这套接口的服务都能用。

还有一个单次生成的 CLI，排查问题时用它最合适：

```bash
cargo run --release -p infero-model --example generate -- \
    models/qwen2.5-0.5b-instruct-q8_0.gguf "Explain RoPE in one sentence." --greedy
```

以及一个 GGUF 检查工具：

```bash
cargo run -p infero-gguf --example info -- models/qwen2.5-0.5b-instruct-q8_0.gguf --tensors
```

把更大的模型分片到多张 GPU：

```bash
# 一个进程对应一个 rank，CUDA_VISIBLE_DEVICES 把每个 rank 固定到不同的物理 GPU
cargo run --release -p infero-server --features nccl -- \
    --model models/big-model.gguf --tensor-parallel-size 2
```

### 也支持 Apple GPU（Metal）

`infero-gpu` 是一层很薄的设备层 trait，`infero-cuda` 和 `infero-metal` 都
实现了它；编译时选中哪一个，最终就只链接那一个，靠 feature flag 而不是
运行时分支来决定：

```bash
cargo run --release -p infero-server --no-default-features --features metal -- \
    --model models/qwen2.5-0.5b-instruct-q8_0.gguf
```

Metal 后端特意对齐了 `cudarc` 自己的形状——`Buf`、`View`、`ViewMut`、
`LaunchConfig`、方法名、参数顺序——所以 `infero-kernels` 里的内核调用点
完全不用改就能对着它编译；只有下面的设备层不一样。内核是逐文件搬过去的：
`ops.cu` → `ops.metal`、`quant.cu` → `quant.metal`、`gdn.cu` → `gdn.metal`、
`mmvq.cu` → `mmvq.metal`，以此类推，`unimplemented.metal` 占位还没搬过去
的内核，所以缺失的内核会在 pipeline 构建时直接报错，而不是悄悄返回垃圾
数据。

Metal 上目前跑的有：F16 和 Q8_0 解码、整数 mat-vec、一个融合的 GQA
解码注意力内核、GatedDeltaNet、主机端采样，以及 M-RoPE，全部对照 CUDA
路径用的同一套 CPU 参考实现和 logits fixture 校验过。还没搬过去的：
类张量核心的整数 GEMM（`mmq.cu` 和 `vendor/marlin` 在 MSL 里没有对应
实现——Apple GPU 没有对等的矩阵乘法指令形状可以对齐）、MoE、视觉塔、
FP8/NVFP4（Apple GPU 没有 FP8/FP4 矩阵单元），以及 TurboQuant KV
压缩。这些都留在 `#[cfg(feature = "cuda")]` 后面，不会在 Metal 上假装
支持。设计笔记和当时测得的起点在
`docs/superpowers/specs/2026-08-23-infero-metal-port-design.md`。

### 不需要 CUDA 工具链

默认构建下这里没有 `nvcc`，也没有 `/usr/local/cuda`——只有驱动。大部分
内核是运行时用 NVRTC 编译的，`scripts/setup-cuda.sh` 会把
`vendor/cuda` 链接到 pip 的 `nvidia-*` wheel（PyTorch 本来就会拉这些包）
里自带的那份 CUDA 用户态上。想用真正的工具链的话设置 `CUDA_HOME` 即可。
`cutlass` 和 `flash_attn2` 这两个 Cargo feature 是例外——它们在构建时用
`nvcc` 提前编译一个基于 CUTLASS 的 FP8 GEMM 和一份 vendored 的
FlashAttention2 shim，需要完整的工具链（`INFERO_NVCC`、
`INFERO_CUTLASS_DIR`）；除了这两个，其它一切都不需要。

因为那些库不在系统搜索路径上，`infero-cuda` 启动时会用绝对路径加
`RTLD_GLOBAL` 去 `dlopen` 它们；`dlopen` 是按 soname 去重的，所以 cudarc
之后按裸名字查找时能找到。这个技巧就是为什么 `libnvrtc-builtins.so` 不用
设置 `LD_LIBRARY_PATH` 也能解析到。

## 目录结构

| crate | 作用 |
| --- | --- |
| `infero-gguf` | GGUF 容器：header、元数据、张量索引。mmap，零拷贝。 |
| `infero-gpu` | 设备层 trait，`infero-cuda` 和 `infero-metal` 都实现它；实际只会链接其中一个。 |
| `infero-cuda` | NVIDIA 那一半：device、stream、cuBLAS handle、带磁盘 PTX 缓存的 NVRTC 编译、张量并行用的 NCCL。 |
| `infero-metal` | Apple 那一半：device、buffer、MSL 编译、dispatch——对齐 `infero-cuda` 的形状搭建。 |
| `infero-kernels` | `.cu`/`.metal` 源码和它们的启动封装。 |
| `infero-tokenizer` | 从 GGUF 词表构建的 byte-level BPE，加上 chat template。 |
| `infero-model` | 配置、权重上传、前向过程、KV 缓存、采样。 |
| `infero-server` | 连续批处理调度器，以及兼容 OpenAI 的 HTTP API。 |
| `infero-tui` | 终端聊天客户端。HTTP 是手写的，所以没有代理环境变量能重定向本地回环请求。 |

一个普通的解码器 block：

```
x ──► rms_norm ──► q,k,v = W·x + b ──► rope ──► store kv
│                                        │
│                            attention over the cache
│                                        │
└────────────────► + ◄── W_o · attn ─────┘
                   │
                   ├──► rms_norm ──► silu(W_g·x) * (W_u·x) ──► W_d·
                   │                                            │
                   └──────────────────► + ◄──────────────────────┘
```

Qwen3.5 风格的混合模型会把这种普通 block 和 GatedDeltaNet block 交替排列，
在 GatedDeltaNet block 里，上面这张图中"对缓存做 attention"那一步被换成
了一份固定大小的循环状态，按 gated delta rule 更新——每个序列一份，每步
整体覆写，不会像 KV 缓存那样随上下文变长。

### 设计笔记

**解码时权重从不在设备上反量化。** 它们一直待在自己的 GGUF block 编码
（或者自己的 AWQ/FP8/NVFP4 布局）里被直接消费。这正是量化模型比全精度
模型不仅在磁盘上更小、显存占用也更小的全部原因。

**解码走整数路径。** 激活行被量化成 Q8_1，用 `__dp4a` 跟打包好的权重做
点积，一条指令处理四个权重和四个激活，中间从不落地成浮点数。各类型的点积
是从 llama.cpp 的 `vecdotq.cuh`（MIT 协议——见 `vendor/LICENSE.ggml`）
搬过来的；启动器和激活量化器是自己写的。这是那种值得直接借鉴而不是自己
推导的东西：在 Llama-3.1-8B 上量到了 9 倍的差距，而之前三轮猜浮点内核的
优化只换来 1.8 倍。

**batch 走整数张量核心。** 一个批量投影本质上是一个 GEMM，`mmq` 直接在
量化权重上跑它：每 32 个元素的量化组对应一次 `mma.m16n8k32.s8`，block
scale 之后再用浮点数折算回去。K=32 不是调出来的参数——每个 ggml block 本身
就是 32 个元素宽，所以一次 MMA 正好消费一个 block，一个 scale 永远不会
跨在两个累加器之间。结构照搬 llama.cpp 的 `mmq.cu`，vLLM 的 GGUF 路径也是
沿用这套结构。Q6_K 每十六个元素就要换一个 scale，一次 MMA 覆盖不了；
恰好 fragment 布局把寄存器 0/1 放在 `k ∈ [0,16)`、2/3 放在 `[16,32)`，
把 B operand 的一半清零就能单独隔离出一个 scale 组。

Fragment 布局由一个测试钉死（`crates/kernels/tests/mma.rs`），它拿一个
one-hot 的 MMA 去对照整数参考实现，因为那里一个下标错一位，产出的矩阵
乘积在 cosine 测试里看起来仍然过得去，却会毁掉生成结果。

**FP8 和 NVFP4 的 batch 走 CUTLASS。** 一个专门的小 M tile（operand 交换
过，照搬 vLLM 自己的真实技术，搬的过程中真发现并修了两个 bug）覆盖窄
batch 解码的那部分形状；FFN 的 gate/up 投影被融合进一次 GEMM 调用，跟
vLLM 的 `MergedColumnParallelLinear` 做法一样。这两个都是真实、测量过、
已经上线的收益——GGUF 侧对应的等价内容见 `crates/kernels/src/cu/mmq.cu`
的设计笔记和 `vendor/marlin/README.md`。

**什么时候用哪个内核。** 单 token：整数 mat-vec（`mmvq`）。2 到 96
token：张量核心 GEMM（`mmq`/CUTLASS）。超过这个范围，`mmq` 重复读权重的
次数已经多到不如反量化进 f16 scratch 再调 cuBLAS 划算。一个类型有
mat-vec 但没有 GEMM 的矩阵，会把 mat-vec 重复跑到十二个 token 为止——
浮点 `gemv` 每个线程解码一个权重，跑在远低于带宽上限的速度，所以哪怕重复
十几遍也比跑一次浮点路径划算。这些阈值是在具体设备上量出来的，不是推导
出来的；`INFERO_MMQ_TILES` 和 `INFERO_NO_MMQ` 就是留给重新量测用的。

词表投影在*所有*行数下——包括单行——都走张量核心路径。看起来像是在
牺牲吞吐量，其实恰恰相反：这正是让 logits 独立于 batch 宽度的关键，而
且在最早量到这一点的那个模型上，它替换掉的浮点 mat-vec 占了 batch=32
解码 step 的 59%。

**激活是 f32，KV 缓存是 f16。** 让激活保持宽精度会多花一些带宽，这些带宽
如果是 llama.cpp 那种风格的引擎宁可花在别处；但这样一来，每个中间结果都能
直接跟 CPU 参考实现比对——而这正是发现一个 RoPE 约定写错所必须依赖的
手段。

**采样可以跑在主机上，也可以跑在设备上。** 设备路径存在的原因是：为投机
解码的验证阶段在主机上重建采样分布，意味着要拷贝 `n * vocab` 个浮点数、
逐行扫一遍整个词表——在一个较宽的模型上这是真实、可测量的开销。两条路径
都从同一个每序列的 `StdRng` 里取数，所以走了哪条路径在输出上是看不出来的。

### 连续批处理

请求共享同一张 GPU。每个 step 会从所有在途请求里组装出一个 batch，跑一次
前向过程；一个序列完成后会在这个 step 结束时离场，一个等待中的请求会在
下一个 step 开始时补上它的位置，不会有别的请求因此暂停。

```bash
infero --model model.gguf --max-seqs 32 --kv-slots 32768
```

两条规则决定一个 batch 的样子。**解码优先**——它们每个只花一个 token 的
代价，而一个正在跑的序列被别人的 prompt 卡住，是客户端能感觉到的卡顿。
**预填填补剩下的空间，并且可以跨 step 拆分**——这正是防止一个 4000
token 的 prompt 把所有人都冻住的机制。

**KV 缓存是分页的**，页大小是一个 token。序列从共享池里取槽位，维护一张
把逻辑位置映射到物理槽位的表，所以长度可以差得很悬殊，一个完成的序列立刻
归还它的槽位，接纳一个新序列只需要写一次表而不是做一次分配。页大小为一
意味着完全没有内部碎片；这张表每个已缓存 token 花四个字节，而这个模型上
token 本身大约每个要占 24 KB。更大的页大小能让 attention 循环获得更好的
局部性，是很明显的下一步。

**一个 CUDA Graph 把一个解码 step 的所有启动重放成一次。** 一个 step
会发出几百个内核调用；把它们捕获一次、之后靠重放，几乎把这部分启动开销
整个从热路径上拿掉了。`INFERO_NO_GRAPH` 可以关掉捕获，方便调试一个原本会
被 graph 的单次重放藏起来的内核问题。

批处理是一个调度层的决定，不是数值层的决定，测试也是照这个标准来卡的:
四个序列一起解码，产出的 token 跟各自单独解码时逐 token 相同；一个中途
加入正在跑的 batch 的序列，不会受到批里其它成员的影响。

### CPU 卸载

`--gpu-layers N` 让 `N` 个 block 留在显存里，其余的挪到锁页主机内存，按层
流式传回来:

```bash
infero --model model.gguf --gpu-layers 12       # 12 个 block 常驻,其余流式传输
infero --model model.gguf --gpu-layers 0        # 只有词嵌入和词表头留在显存
```

**计算从不离开 GPU。** 这跟 llama.cpp 的 `-ngl` 不一样——那种做法是把
卸载的层放到 CPU 上跑，每种量化格式都需要再写一套内核。这里权重在移动，
算术留在原地不动，所以卸载换的是 PCIe 带宽换显存，不是拿 GPU 吞吐换
CPU 吞吐——每种内核也就只需要一份实现。

一层里七个大矩阵被打包进一个连续的锁页 blob，所以传一层是一次连续的
DMA，而不是七次。两个暂存槽按层的奇偶交替使用：计算流读槽位 `L % 2`
的同时，拷贝流在填槽位 `(L+1) % 2`，两个方向都有事件把关——`ready[s]`
让传输落地之后才允许计算开始，`consumed[s]` 让计算结束之后才允许下一次
传输开始。归一化层和 bias 始终常驻；它们只有几 KB，流式传输只会多加
descriptor，省不下什么。

因为只是路径变了，结果不会变：`cargo test -p infero-model --test offload`
断言在 0、1、12、23 个常驻层的情况下，不管是批量还是逐 token，logits 跟
全量常驻时**逐位相同**。

### KV 缓存：TurboQuant

缓存可以用 [TurboQuant](https://arxiv.org/abs/2504.19874)（Zandieh 等人，
Google Research，ICLR 2026）压缩，这是照着论文实现的:

- **算法 1，`TurboQuant_mse`**——一个随机旋转 `Π` 能把任意单位向量变成球面
  上均匀分布的，于是它的坐标不管输入是什么都服从*已知*的密度
  `f_X(x) ∝ (1-x²)^((d-3)/2)`。这正是能一次性、离线求解出最优标量量化器、
  不需要任何校准数据的原因。`crates/kernels/src/turboquant.rs` 按每个
  head 维度数值求解式 （4）；求出的失真值跟 Max 的 Lloyd-Max 表精确到
  四位有效数字（b = 1..4 时分别是 0.3634 / 0.1175 / 0.03454 /
  0.009497），这正是定理 1 四舍五入引用的数值。
- **算法 2，`TurboQuant_prod`**——一个 MSE 最优的量化器会*收缩*内积，所以
  key 拿 `b-1` 位 MSE 编码加一个 1 位的 QJL 符号位在残差上，这样能让
  attention 的 logit 保持无偏。在内核上量出来的结果：仅用 MSE 的估计器
  回归到真值的斜率是 0.885，两阶段的那个是 1.003。

key 用算法 2，value 用算法 1——key 参与的是内积，value 参与的是加权平均。

**一切都留在旋转后的坐标系里。** `Π` 是正交的，`S` 是独立同分布的高斯，
所以 `S' = S·Πᵀ` 也是，于是估计量变成

```
<q, x~> = <Πq, y~> + (sqrt(pi/2)/d) · gamma · <S'(Πq), qjl>
```

query 每个 token 只旋转一次，**缓存里的向量永远不会被旋转回去**。对 value
来说，同样的替换把逆旋转从每个缓存向量一次，挪到每个 `(head, token)` 一次，
放在加权求和之后做。没有这一步，这套方案就不值得跑了。

没实现的部分：论文里的离群通道拆分（outlier-channel split），这正是它
2.5、3.5 这种非整数比特率的来源（在 `d = 128` 上，32 个通道用 3 位，
96 个用 2 位）。这里的位宽是 2、4、8，方便编码正好按字节打包。

```bash
infero --model model.gguf --kv-quant k8v4     # key 8 位,value 4 位
infero --model model.gguf --kv-quant tq4      # 论文里对称的 4 位方案
```

`tq2` / `tq4` / `tq8` 这几个预设是对称的、带 QJL 的；`tq2-mse` /
`tq4-mse` 去掉了 QJL 阶段；`k<bits>v<bits>[+qjl]` 可以两侧独立设置。

### 支持的权重编码

GGUF:`F32`、`F16`、`Q4_0`、`Q4_1`、`Q5_0`、`Q5_1`、`Q8_0`、`Q4_K`、
`Q5_K`、`Q6_K`。此外还支持原生 AWQ（`Q4_G128`）、原生 FP8 W8A8，以及原生
NVFP4 W4A4 的 Hugging Face checkpoint。

| | 整数 mat-vec | 张量核心 / CUTLASS GEMM |
| --- | --- | --- |
| `Q8_0` | 有 | 有 |
| `Q4_K` | 有 | 有，行数是 256 的倍数 |
| `Q5_K` | 有 | 退回反量化 + cuBLAS（见下文） |
| `Q6_K` | 有 | 有，行数是 256 的倍数 |
| `Q4_G128`（AWQ） | 有 | 有 |
| FP8（W8A8） | 不需要（本身就是原生精度） | 有，CUTLASS |
| NVFP4（W4A4） | 不需要（本身就是原生精度） | 有，CUTLASS |
| 其它 | 无 | 无 |

其余类型退回浮点 mat-vec 或者反量化 + cuBLAS。`Q5_K` 就是 `Q4_K` 加上
每个权重多一位、打包在一个独立的 `qh` 数组里——llama.cpp 自己的
`Q4_K_M`/`Q4_K_L` 策略会把它用在 checkpoint 里"更敏感"的那些张量上
(`attn_k`/`attn_v`），而不是给所有张量用同一个宽度，所以一个 K-quant
路径原本没预期会出现 Q5_K 张量的真实 `Q4_K_M` 文件里出现它并不罕见。
它有 mat-vec（在同一形状下量出来跟 `Q4_K` 差距是个位数百分比，前提是
benchmark 本身正确加了同步——见 `crates/kernels/examples/
q5k_vs_q4k_bench.rs`），但跟 `Q4_G128` 一样，还没有专门的张量核心 GEMM；
宽 batch 下走通用的反量化到 f16 + cuBLAS 路径。给一个新类型加 mat-vec
意味着搬它自己的 `vec_dot_*_q8_1`；加 GEMM 意味着写一个 staging 函数，把
它的 block 展开成一个 int8 tile，外加每 16 或 32 个元素一个 scale。

### 架构

旋转位置编码的配对方式跟着架构走，而且文件里并不会记录这一点:
llama 系列的转换会重排 Q 和 K，让*交错*配对能复现 Hugging Face 的
rotate-half，而 Qwen2/Qwen3 系列想要的是 NeoX 方式。配对方式搞错会得到
看起来很流畅、但随位置漂移的输出，而不是直接报错——这正是当初发现这个
问题的方式。Llama 3.1 还额外带了 `rope_freqs.weight`，这是它 128k
上下文专用的按维度频率除数，而且它的 chat template 自己会输出
`{{ bos_token }}`。

一个"Q4_K_M"文件是混合体。Qwen2.5-0.5B 的隐藏维度 896 不是 256 元素
K-quant 超级块的整数倍，所以它大部分行会退回 `Q5_0`；而一个 `d_model`
*确实是* 256 的倍数的模型，仍然可能在大部分 `Q4_K` 之外掺杂少量真实的
`Q5_K` 张量——这正是那些遗留的 block-32 量化格式，以及现在的 `Q5_K`，
都不是可选项的原因。

## 正确性

`cargo test` 会跑整个工作区几百个测试。需要模型文件或者真实多 GPU
环境的测试，在 fixture 或者第二张卡不存在时会干净地跳过。

| 内容 | 怎么校验 |
| --- | --- |
| 分词器 | 在 25 个用例（中日韩字符、emoji、代码、连续空白）上跟 `AutoTokenizer` 逐 token 比对。chat template 输出逐字节比对。 |
| 量化解码器 | 每种编码的 mat-vec 跟同一张量的 F16 版本比对；对于较新的编码，还会跟一份从头独立写的 Rust 版参考反量化/点积逻辑比对，而不是拿同一份 CUDA 源码的第二个副本来比。 |
| NVFP4 / FP8 | 一个真正整体量化的 checkpoint（注意力 + GatedDeltaNet + FFN，不只是 FFN）能读取它自己真实的 `hf_quant_config.json` 并正常加载生成，两种已知的真实 on-disk schema 都支持。 |
| TurboQuant | 码本失真度跟 Max 的 Lloyd-Max 表精确到四位有效数字比对；量化数据上量出来的失真度跟码本预测值比对；证明仅用 MSE 的估计器会收缩内积，而两阶段的那个不会。 |
| CPU 卸载 | 在 0、1、12、23 个常驻层的情况下，不管批量还是逐 token， logits 跟全量常驻时逐位相同；每次卸载层每轮只传输一次。 |
| 连续批处理 | 四个序列一起预填，产出的 logits 跟各自单独预填时相同；换掉同批次里的其他成员，一个请求的 logits 逐位不变；中途加入正在跑的 batch 不受影响；回收的池槽位不会带着上一个使用者的历史信息。 |
| 张量核心 / CUTLASS GEMM | `mma.m16n8k32.s8` 的 fragment 布局用整数参考实现钉死，包括用 one-hot 输入把一个错位的下标定位到具体某一格。在若干 token 数下对照浮点 mat-vec 算余弦相似度，故意选参差不齐的宽度，因为 token tile 的边界差错正是它们要抓的。多个 batch 宽度下输出逐位相同。 |
| 张量并行 | 分片运行的 logits 跟单卡跑同一模型比对。 |
| TUI | 跨 chunk 边界重组 SSE 帧；折行永不超出一行，中日韩字符按两格算。 |
| 旋转位置编码变体 | 两种配对方式都保持范数不变，且彼此不同；频率翻倍等价于位置减半。 |
| 内核 | RMSNorm、RoPE、SwiGLU、带因果遮罩的 GQA attention、 GatedDeltaNet 的 delta rule，全部对照 CPU 参考实现。 |
| 前向过程 | 在四个 prompt 上对照 `transformers` 的 f32 logits 比 argmax、top-10 集合和 logit 分布。 |
| KV 缓存 | 逐 token 解码必须落在跟批量预填完全相同的状态上。 |
| HTTP | 流式的 chunk 必须能重组成非流式响应；停止序列、种子、用量统计、错误格式都要校验。 |
| `compute-sanitizer` | 新的整数/dp4a 内核在接入调度表之前，都要先在 `--tool memcheck` 和 `--tool racecheck` 下跑过。 |

fixture 由 `scripts/make_tokenizer_fixtures.py` 和
`scripts/make_logits_fixtures.py` 重新生成；`cargo test` 不会跑这两个
脚本。

## 性能

### 内核层面跟 vLLM 打平

一个解码 step 里占比最大的两个内核——FFN GEMM 和 GatedDeltaNet 的
delta-rule 内核——直接跟 vLLM 自己真实、编译出来的内核在同一张 GPU、
同样真实的形状下做过头对头 benchmark，不是靠读 vLLM 源码推断的。在 FFN
GEMM 的真实形状上，infero 自己的小 M CUTLASS tile 跑得比 vLLM 自己的
`cutlass_scaled_mm` 在同一形状下*还快*；GatedDeltaNet 那部分内核，在
两边 benchmark 都改成不讨好缓存的方法（用轮换的多份状态缓冲区，而不是
重复用恰好能塞进 L2 的同一份缓冲区）之后，量出来是统计意义上打平的。
这两个占比最大的内核，都不是端到端剩余差距的所在。

真实、已经上线的杠杆有两个:

- **一个 operand 交换过的小 M CUTLASS FP8 GEMM tile**，vLLM 自己的真实
  技术，搬过来时没有照抄 vLLM 自己的 M≤64 阈值——在这张卡上真实扫过一遍
  发现的是 32 token 的交叉点：在 batch=16、不开投机解码的解码吞吐上，
  量出来大约有 10% 的真实提升。
- **FFN 的 gate/up 投影融合进一次 GEMM**，跟 vLLM 的
  `MergedColumnParallelLinear` 做法一样，修掉这个融合路径里一个真实的
  三倍显存冗余 bug 之后，同一形状下又有大约 3% 的真实提升。

在那之后检查过的其它内核层面候选——窄 N 的 CUTLASS matvec、stream-K
拆分、CTA 光栅化 swizzle、更深的权重预取——不是回归，就是量出来落在
噪声范围内，或者（stream-K）在个别形状上赢了、但没占到一个解码 step
里足够大的比重去撬动端到端数字。跟 vLLM 自己最好数字之间剩下的差距，
目前没能归到任何单一内核上；异步的主机侧调度、GatedDeltaNet 解码的
并行度，以及 CUDA Graph 的启动粒度，都对照过 vLLM 的真实源码，发现设计上
已经跟它一致了。

### 投机解码：真实存在，但不是免费的

支持端到端的 MTP（多 token 预测）起草头，还有一条从头搭建的设备端常驻
起草循环（Gumbel-max 采样，每轮只用一次主机同步而不是好几次），专门用来
去掉起草路径上的主机同步等待。这是量出来的，不是想出来的：在这张卡上，
那条设备端路径相对更简单的主机驱动路径，是一个真实、可复现的**净损失**，
因为 Gumbel-max 本身更低的接受率（没有重复惩罚、没有 top-p 项）抵不过
省下的同步开销——所以最终上线的是主机驱动的起草循环，设备端的内核留在
代码树里作为测试过、但未启用的基础设施，而不是默认路径。投机解码本身
是否真的划算取决于具体 workload：它不是对所有 batch 形状都有帮助，把它
关掉而不是强行打开，是一个刻意、经过测量的选择（`crates/model/src/
spec.rs`)。

### CPU 卸载

Qwen2.5-0.5B-Instruct，Q8_0，41 token 的 prompt，生成 150 个 token:

| `--gpu-layers` | 显存（MiB） | 卸载量（MiB） | 预填 | 解码 |
| --- | --- | --- | --- | --- |
| 24（全部） | 639 | 0 | 745 tok/s | 235 tok/s |
| 18 | 578 | 91 | 712 tok/s | 108 tok/s |
| 12 | 488 | 181 | 645 tok/s | 62 tok/s |
| 6 | 397 | 272 | 596 tok/s | 44 tok/s |
| 0 | 306 | 363 | 557 tok/s | 34 tok/s |

**预填几乎感觉不到，解码要全额承担代价。** 预填会把每次权重读取的成本
摊在一整块 token 上，所以在零常驻层的情况下，速度依然能达到全量常驻时
的 75%。解码每个 token 都要重新读一遍权重，直接砸在 PCIe 总线上:
每 token 363 MiB、34 tok/s，换算下来是 12.2 GB/s，而这台机器锁页
host-to-device 拷贝能到 13.2 GB/s（`cargo run --release -p
infero-kernels --example launch_overhead`）。已经跑到链路上限的 92%，
传输路径上已经没什么可赢的了——预取已经把计算完全藏起来了，剩下的杠杆
是搬更少的字节，而不是搬得更快。

这也是为什么锁页分配很重要：同一个 benchmark 测出可分页内存只有
9.8 GB/s，所以锁页在这里值 35%。

### AWQ、FP8 和 NVFP4 checkpoint

`--model` 既可以指向一个 GGUF 文件，也可以指向一个 Hugging Face
checkpoint 目录。加载器是逐张量检查的，不是看整个 checkpoint 的整体
格式：`.qweight`/`.qzeros`/`.scales` 意味着 AWQ，`.weight`/
`.weight_scale_inv` 意味着原生 FP8（W8A8）checkpoint，一份真实的、带
NVFP4 目标的 `hf_quant_config.json` 意味着原生 NVFP4（W4A4），这几种
还可以混在同一个文件里——一个 MoE checkpoint 里有些专家是 AWQ、有些是
FP8，不需要任何额外的 flag 就能两种都读进来。AWQ 的量化投影在读入时会
被转置并重新打包成 `Q4_G128`——每 128 个权重一个 block，一个 `f16`
scale 和 zero，按输出通道优先排列，这样现有的 mat-vec 和张量核心 GEMM
可以原样读它们，不用改。vLLM 的 `awq_marlin` 出于同样的原因也做重新
打包。FP8 和 NVFP4 张量会被重新打包成它们各自 CUTLASS 路径能直接读的
block 布局，保持原生精度——中间没有反量化再重新量化的一步。

**AWQ 不是字节数更少。** 它的层比 Q4_K_M 文件小 13%——4.25 位对 4.83
位——但它把 `lm_head` 存成 `f16`，如果不单独量化这个头，这个差距会被
完全抵消（见下文）。这个格式赢在*解码成本*，不是体积：一个 Q4_K 点积
每 32 个权重就要从一个打包的十二字节字段里解出一个 6 位 scale 和一个
6 位 minimum；`Q4_G128` 每 128 个权重只读一次 `half2`。

**词表投影值得量化。** 留成 `f16` 的话它占解码 step 的五分之一左右，
浮点 mat-vec 读它的速度明显低于带宽上限；加载时量化成 Q8_0 之后，这部分
成本能降到大约六到七分之一。对于一个输出要拿去在十万乃至更多词表上做
argmax 的投影来说，八位精度不是有意义的损失；而前面"设计笔记"里说的
"任意宽度都走张量核心"这条性质，正是让它能被量化、同时不牺牲 batch
不变性的关键。

### 对比 vLLM 和 llama.cpp

同一档次的 RTX A4000 硬件，同一个压测工具打向每个引擎的 OpenAI 接口，
温度设为 0。这里有两种比较，值得分开说，因为它们回答的是不同的问题:

**对比读同一份字节的引擎。** llama.cpp 读同一个 GGUF 文件，没有格式
差异可以藏在后面：单流吞吐两边差距都在个位数百分比以内，infero 的
连续批处理在中等客户端数下会拉开优势，高并发下差距又会收窄——不管
某一周量出来的具体差距是个位数还是更大，比的都是内核和调度器，不是
文件格式。

**对比读一个更合适格式（AWQ/FP8）的 vLLM。** 这是更难的比较，老实说:
vLLM 自己针对原生格式的 GEMM tiling，比这两个引擎在 GGUF K-quant 路径
上用的都更成熟，补上这个差距是格式和内核 tiling 层面的投入，不是调度层
的投入。已经证明、而不是估计出来的是：infero 自己基于 CUTLASS 的
FP8/NVFP4 内核，不是造成当下这个差距的原因——见上面"内核层面跟 vLLM
打平"一节。想在自己的硬件上得到一个当下、诚实的数字，重新跑一遍
`cargo run --release -p infero-model --example batch_bench` 和服务端
自己的 `INFERO_PROFILE=1` 输出，别去信一个在别人的卡上、某一天测出来的
数字——这个项目自己的历史，以及这一节自己过去的版本，都是"一个过时的
benchmark 比没有 benchmark 更糟"的证据。

### 时间都花在哪儿了

`INFERO_PROFILE=1` 用 CUDA event 给每个内核计时，按占比排序打印出一张
表（它会把 stream 串行化，所以绝对数值是被放大的，只有占比是有意义的）。
`INFERO_STEP_TIMING` 在 CUDA Graph 照常捕获的情况下，给出主机侧的阶段
计时，因为 graph 和逐内核计时这两件事不能同时存在。`cargo run --release
-p infero-model --example decode_floor` 精确重放一个解码 step 要执行的
那些 mat-vec、别的什么都不做，当作一个 step 不可能突破的下限。

这里已经构建并测量过一长串听起来很有道理的优化，结果都是白搭——更窄的
启动网格、把已经被 CUDA Graph 变得几乎免费的内核再融合一次、不用
`ldmatrix` 的 operand 加载、换别的寄存器 tile 形状、把 Marlin 自己那套
负载均衡的 k 拆分搬到一个已经把它要解决的问题解决掉的更粗糙的拆分上。
真正落地的那些有一个共同点：它们打的是一个真实、量出来的瓶颈（一个
占内核 22% 的 scale 查表路径、一个没能给每个 SM 供上足够 block 的
k 拆分、一次本可以只读一遍却读了三遍的按行读取），而不是一个听起来
有道理的猜测。这段历史和它的方法论完整地留在
`crates/kernels/src/cu/mmq.cu` 的设计笔记和 `crates/model/examples/
gemm_bench.rs` 里；放在源码里而不是在这里重复一遍，是因为它又长又带
时间戳，而且是针对这份 README 不该假设你有的具体硬件测的。

## 环境要求

- NVIDIA GPU，计算能力 7.0 及以上（在 sm_86 上测试过），驱动支持
  CUDA 12 或 13，以及来自 pip wheel 或工具链安装的 CUDA 用户态
- 多 GPU 张量并行需要 NCCL（`libnccl.so`），通过 `nccl` 这个 Cargo
  feature 启用
- 或者：Apple Silicon GPU（Metal 3+），macOS——目前那边支持什么见上面
  [Metal](#也支持-apple-gpumetal) 一节
- Rust 1.90+

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=jackwangfeng/infero&type=Date)](https://star-history.com/#jackwangfeng/infero&Date)
