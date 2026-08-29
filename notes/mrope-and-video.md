# M-RoPE 和视频输入：设计与分阶段计划

## 为什么

图片输入已经能跑（`1232aa6`），但主模型的位置编码完全没接 M-RoPE——`llm_position_ids`/
`interleaved_mrope_axis`（`qwen35_vision.rs:889,915`）这两个函数照着 HF 参考实现写好、也有
capture 校验过的单元测试，但**在生产路径上零调用点**：`forward_batch_rows` 现在给每个 token
只算一个标量位置（`start + k`），图片和文本 token 一视同仁地线性递增。这意味着当前图片支持
虽然能跑，但空间位置感知是错的（真实权重是按 `mrope_section=[11,11,10]` 训练的三轴位置）。

视频输入完全没做，被 `scheduler.rs:806-813` 显式拒绝。M-RoPE 是视频的前置条件——视频的时序
信息就是靠这三轴里的 T 轴表达的，不接上三轴，视频就算跑起来时序也是错的。所以两个一起做。

## 关键澄清（跟最初设想不一样的地方）

1. **`batch_tokens()` 不是常量，是运行时算出来的**（`lib.rs:119-130`，随 `--ctx`/`--max-seqs`
   变化），`--ctx 32768` 时只有 341。这是视频能塞多少帧的硬约束，必须按实际启动参数算，不能
   写死。
2. **视频 prompt 不是一整段连续 `<|video_pad|>`**，是"每帧一组"：
   `<0.5 seconds><|vision_start|>pad×N<|vision_end|><1.5 seconds>...`，按 2 帧一个时间 patch
   切分，组间插时间戳文本（`processing_qwen3_vl.py`）。好处：`llm_position_ids` 天然按"每个
   视觉 run 各自 advance"设计，一帧一组直接能用，不用改这个函数本身。
3. **视觉塔目前是纯 CUDA**，`crates/kernels/src/msl/` 没有 vision.metal。视频这部分只能上
   CUDA；Metal 遇到视频请求要显式拒绝，不能让它钻进 `kernels().get()` 才报错。
4. 视觉塔本身对多帧（`t>1`）**已经是通用的且有 capture 测试**（`cu_seqlens`/
   `vision_position_ids`/`pos_embed_taps`/`vision_forward` 全部在 `t=2` 上验证过）。缺口全在
   边缘：`encode_image` 硬编码 `Grid{t:1}`、`vision_resize` 是按单图算的、host 端没有视频解码。

## 位置数据怎么流过去（核心设计）

`act.positions`（标量，每 token 一个 i32）现在身兼三职：喂 RoPE、写 slot table、卡 causal
mask 上界。M-RoPE 的 T 轴在图片/视频段内是**常量**（不随 token 递增），不能顶替这三个角色里
的另外两个。所以：

- **标量 `positions` 保留不动**，继续管 slot table 和 mask。
- **新增 `mrope_positions`**（`3*chunk`，token-major `[T,H,W]` 交错），只喂给 rope kernel。
- kernel 侧只加两个参数：`mrope_axis[i]`（每个频率下标该读哪一轴，来自
  `interleaved_mrope_axis`，加载时算好上传一次）和 `pos_stride`（3 或 1）。`pos_stride=1` 时
  退化成跟今天完全一样的标量寻址、**同一份指令**，不额外分支——这就是"纯文本零回归"的保证，
  也是唯一要做的正确性收口:一个 bit-exact 测试（`pos_stride=1` 全零轴 vs 改动前）。
- `Running`（`scheduler.rs`）新增 `mrope: Option<Box<[i32]>>` + `mrope_delta: i32`，跟
  `had_vision` 一样活得比 `vision` 字段长（`vision` 在 prefill 后就被拿走清空）。prefill 按
  `from..from+len` 绝对切片（配合 prefix cache 命中时非零起点），decode 阶段三轴都等于
  `pool.len(seq) + delta`。
- 投机解码（MTP drafter）复用同一套 `mrope_positions`/`mrope_axis` 分离逻辑，`DraftFeed` 新增
  `mrope` 字段，delta 作为参数传（不锁存在 head 上，因为 head 是单例复用的）。

## 分阶段（每步独立可验证）

**A 组：M-RoPE**
- A0 在 bw 上重新生成 `tools/capture_qwen35_vision.py` 的基准（现在两个 M-RoPE 相关测试
  因为 fixture 缺失一直是 SKIP-as-pass，等于没验证过），并加一个 `INFERO_REQUIRE_CAPTURE=1`
  让缺失 fixture 变成失败而不是静默跳过。
- A1 `config.rs` 解析 `mrope_section`/`mrope_interleaved`，加载时校验 `sum(section)==rotary_dim/2`。
- A2 CUDA + Metal 的 `rope_qk_f32`/`rope_qk_packed_f32` 加 `mrope_axis`/`pos_stride` 两个参数；
  新测试用三个不同轴值的 token，跟 host 参考比对，并对"分块轴而非交错轴""轴全读 T""轴主序
  寻址"三种错误实现做 `assert_discriminates`。
- A3 `BatchItem`/`Acts`/`forward_batch_rows` 打通；bit-identical 测试证明纯文本零回归；graph
  capture 场景跑 50 步比对 `INFERO_GRAPH=0`。
- A4 `scheduler.rs` 用 `llm_position_ids` 给带图请求算 `Running.mrope`/`mrope_delta`；单测
  校验 delta 为负（这是"没有把 advance 算错成 token 数"的判据）、prefix-cache 命中场景切片
  正确。
- A5 MTP drafter 接三轴；用"投机解码接受的 token 跟不投机的逐 token 结果一致"做判据。

**B 组：视频**（依赖 A 组落地）
- B0 capture 脚本加视频 splice case，作为 `llm_position_ids` 处理"每帧一组"结构的判据。
- B1 host 端 `prepare_clip`/多帧 `patchify`。
- B2 device 端 `encode_clip`：`grid_t` 次 `vision_patchify` launch 写入 `pixels_h` 的不同偏移；
  `VisionFeatures` 加 `grid_t`。
- B3 `vision_resize_video`：patch 预算除以 `grid_t`。
- B4 `crates/server/src/video.rs`：ffmpeg 子进程解码 + 采样（严格照抄参考实现的 2fps/
  linspace 采样公式和时间戳格式化，不能自己发明）。
- B5 prompt 拼装（时间戳文本 + 每帧一组 vision_start/pad/vision_end）、`api.rs` 加
  `video_url`、解除 `scheduler.rs` 的拒绝、按 `batch_tokens()` 算能塞多少帧并在超限时先尝试
  缩小分辨率而非直接拒绝。
- B6 端到端：运动方向探针（时间反转答案必须翻转）+ 帧乱序负对照（乱序后答案必须变化，否则
  说明时间轴根本没接上模型）+ 数数探针（逐帧数字，抓采样丢帧/重复）。
- B7 Metal 上视频请求显式拒绝（而不是运行时才炸）。

风险点：Metal 的 buffer index 要跟着新参数重新排号，最容易手滑绑错——靠 A2 的 bit-identical
测试抓。`batch_tokens()` 在长上下文时很小（32k 时约 8-20 帧），视频长度上限是运行时相关的，
要在拒绝信息里把这几个数字都打出来，不能只说"太长了"。
