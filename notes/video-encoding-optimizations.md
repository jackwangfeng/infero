# 视频/图片编码优化方案：vLLM 对照研究

## 背景

`notes/mrope-and-video.md` 落地的视频输入功能验证通过，但过程中暴露了几处已知的性能/能力上限（见该文档"分阶段"末尾风险点，以及 B5-B7 完成后的复盘）。这份文档是照用户要求，先去读 bw 上真实装着的 vLLM 源码（`/home/jeff/vllm312/.../site-packages/vllm/`，版本对应 Qwen3-VL 已支持）搞清楚业界怎么做，再给出针对本仓库的具体方案——不是凭印象设计的。

每一节结构统一：**现状**（我们的代码，带文件行号）→ **vLLM 怎么做**（真实源码，带证据）→ **方案**（具体改法 + 工作量/风险评估）。

---

## 1. 视频/图片的 prefill 不能跨 step 分块（最高优先级）

**状态：已实现并验证，中途发现并修复一个真实 bug。** `BatchItem` 加了 `vision_row_offset`，`Running` 加了跟踪字段，`plan()` 去掉了"整段预留"分支，`vision_targets`/`vision_splice` 改成按 chunk 实际包含的 placeholder 数量切片。`encode_pending_image`/`encode_pending_video` 的硬性预算检查从 `batch_tokens()` 改成 `pool.max_seq()`。验证：
- 模型层：`crates/model/examples/chunked_vision_prefill_check.rs`，真实 27B 上对比"整段一次 splice" vs "从占位符游程中间切开分两次 splice"，用同等切分方式的纯文本请求做对照组（排除深层模型本身的 GEMM/attention 分块噪声），argmax 一致，且视觉场景的误差和纯文本对照组同一量级。
- 端到端：`--max-seqs 4`（`batch_tokens=256`，此前这个配置下任何真实视频请求都会被直接拒绝）下，sample.mp4 默认分辨率（未触发降分辨率重试）成功跑通，`prompt=1092` token 跨约 5 个 step 完成；时长探针答案精确为 21（真实时长 21.03s）。

**中途发现的真实 bug（做第 5 项时暴露）**：最初实现用 `Running.vision_at`（占位符游程*起点在 prompt 里的绝对下标*）+ `from.saturating_sub(vision_at)` 算 `vision_row_offset`——这个公式假设占位符是**一整段连续的** pad token，对图片成立（`<|vision_start|>pad×N<|vision_end|>` 前后都是普通文本，中间没有别的东西），但对视频**不成立**：视频的占位符是 `grid_t` 段不连续的游程，段与段之间夹着 `<T.T seconds>` 时间戳文字——这些文字也会被 `from - vision_at` 计入"已经跨过的距离"，但它们不是 pad token，不该计入"已消费的 feature 行数"。只要视频只有 1 个 frame-group（`grid_t=1`，跟图片同构）或者整个占位符游程碰巧没跨 step，这个公式就是巧合正确的——这正是为什么最初用 sample.mp4（`grid_t=21`）跑通、看起来验证通过了，其实只是没有触发到会暴露这个 bug 的具体切分点。真正暴露它的是给优化 5（dedup）设了一个激进阈值，把 `grid_t` 压到 2、且总 token 数使得切分点恰好落在两段之间的文本区——`scheduler.rs` 自己的边界检查当场报错："this chunk wants rows 722..976 of a 960-row clip"，不是静默算错。

修复：删掉 `Running.vision_at`，改成 `Running.vision_pad_tok`（这个视觉请求用的 pad token id 本身，`img_tok` 或 `vid_tok`），`vision_row_offset` 和"何时可以丢弃 `vision`"的判断都改成**直接数 `prompt[..from]`/`prompt[..from+len]` 里 `vision_pad_tok` 出现了几次**，而不是做下标减法——这个方法对连续（图片）和不连续（视频）两种布局都天然正确，因为它只关心"pad token 本身出现了几次"，不关心中间夹了什么。

新增了一个永久回归测试防止这个 bug 再犯：`crates/server/tests/http.rs` 里的 `a_multi_group_video_survives_chunked_prefill`——用 `ffmpeg testsrc` 现生成一段 320x320/10 秒的合成测试视频（不依赖仓库里的样例文件），逼出多 frame-group + 跨多个 step 的场景。**验证了这个测试真的能抓到这个 bug**：把 bug 临时恢复到代码里重跑这个测试，测试确实失败（报同样的 "chunk wants rows X..Y" 错误），换回修复版本后测试通过——不是只是"写了个测试"，是确认了这个测试有区分度。

### 现状

`scheduler.rs:1371-1407`（`plan()` 里的 `vision_reserved` 分支）：一个带视觉输入的序列，一旦 prefill 开始，必须把**整个** placeholder 展开后的 prompt 长度一次性预留在同一个 `Work::Prefill { from: 0, len: prompt.len(), last: true }` 里：

```rust
// A fresh vision sequence's placeholder splice runs once, over one
// `BatchItem`, so its whole prompt has to land in a single step rather
// than being split the way an ordinary prefill can be
let vision_reserved = self.running.iter()
    .position(|r| r.vision.is_some() && r.prefilled == 0);
if let Some(i) = vision_reserved {
    budget = budget.saturating_sub(self.running[i].prompt.len());
}
```

`encode_pending_video`/`encode_pending_image`（`scheduler.rs`）在 admit 阶段就会拿 `batch_tokens()` 硬卡一次——超了要么降分辨率重试、要么直接拒绝。这也是这次测试时反复撞见"默认 `--max-seqs 4` 时 `batch_tokens=256`，视频请求几乎必拒"的根源。

### vLLM 怎么做

（研究 agent 对 `vllm/v1/core/sched/scheduler.py`、`vllm/v1/worker/gpu/mm/encoder_runner.py`、`vllm/v1/core/encoder_cache_manager.py`、`vllm/multimodal/inputs.py` 的结论）

**核心思路：视觉塔的计算和 decoder 的 chunked prefill 完全解耦。**

1. **视觉塔只跑一次，结果整个缓存下来**（`EncoderRunner.execute_mm_encoder` → `self.model.embed_multimodal(...)`），按 `mm_hash` 存进 `EncoderCacheManager`。这一步和 decoder 的 token 预算无关，运行时机也早于 decoder 调度。
2. **decoder 侧按任意 token 窗口切片已经算好的 embedding**，不需要重新触发计算：
   ```python
   # PlaceholderRange.get_embeds_indices_in_range()
   embeds_start_idx = self.embeds_cumsum[start_idx - 1] if start_idx > 0 else 0
   embeds_end_idx = self.embeds_cumsum[end_idx - 1] if end_idx > 0 else 0
   ```
3. **切片边界完全不对齐帧/patch group**——研究 agent 明确确认：`scheduler.py`、`multimodal/utils.py`、`multimodal/inputs.py` 里都没有按 `grid_thw`/帧/merge_size 对齐的逻辑。之所以敢这么切，是因为 #1 已经保证了被切的是"已经算好的行"，不是"正在算的东西"——mid-item 切一刀不会截断一次真正的矩阵乘法。
4. 默认开启（`SchedulerConfig.disable_chunked_mm_input` 默认 `False`），只有 encoder-decoder 架构强制关闭。

### 方案

**好消息：我们的架构已经完成了第 1 步（视觉塔先算完再进队列），差距只在第 2 步（decoder 侧不能切片消费）。**

证据：`encode_pending_video`/`encode_pending_image` 本来就在 `admit()` 里、进入 `Running` 之前跑完 `encode_clip`，`Running.vision: Option<VisionFeatures>` 已经是"算好缓存着"的语义,不是"在 forward 里现算"。真正卡住的是两处：

1. `plan()` 的 `vision_reserved` 分支：一刀切要求整段放进一步。
2. `vision_splice` kernel（`crates/kernels/src/vision.rs:500`）：其实**已经**是按任意行子集工作的——`features: &View`、`dst_row: &View`、`n_features` 这三个参数天然支持"这次只切 `[offset..offset+len]` 这一段特征行，映射到这次 chunk 里对应的 placeholder 位置"。kernel 本身不用改一行。

具体改法：
- `Running` 里 `vision: Option<VisionFeatures>` 的生命周期从"prefill 那一步用完即弃"改成"活到整个 placeholder run 被消费完为止"（目前它确实已经存在 `Running` 里，只是 `plan()` 假设一步用完；需要确认的是 `retire`/prefill 完成判断要不要跟着改，估计只需把"何时可以丢弃 `vision`"的条件从"这一步跑完"改成"`prefilled` 覆盖了整个 placeholder run"）。
- `BatchItem.vision`（`crates/model/src/lib.rs:203`）新增一个偏移量（比如 `vision_offset: usize`），告诉 `forward_batch_rows` 这次只切 `VisionFeatures` 的 `[vision_offset..vision_offset+n_this_chunk]` 这一段去 splice，而不是假设整段都在这一步。
- `plan()` 去掉"整段预留"逻辑，让带视觉的序列复用已有的"普通 prefill 按 `remaining.min(budget)` 分块"路径——`admit()` 那个基于 `batch_tokens()` 的硬性拒绝检查也可以放宽或去掉（改成软性的"降级警告"，因为不再需要一步吃下整段）。
- M-RoPE 位置（`mrope_for_video`/`Running.mrope`）已经是按 token 索引整段算好存着的（`scheduler.rs` 里 `mrope: Option<Box<[i32]>>`），chunk 切片直接复用现有的 `mrope[3*from..3*(from+len)]` 逻辑（`step()` 里已经有这个模式，图片路径已经在用），不需要额外改动。

**工作量评估：中等**。不用碰视觉塔本身（CUDA kernel、`encode_clip`、patchify 都不用动），改动集中在 `scheduler.rs` 的 `plan()`/`Running` 生命周期管理和 `BatchItem`/`forward_batch_rows` 的 splice 调用点——跟这次 A3/A4 阶段打通 M-RoPE 三轴位置时改的是同一批文件，模式也类似（"标量字段拆成偏移量 + 长度"）。**这是 5 项里投入产出比最高的一项**：直接解除"视频长度被单步预算硬卡死"的限制，不用再靠反复降分辨率去凑预算。

---

## 2. ffmpeg 抽帧是线性扫描，不是 seek

### 现状

`crates/server/src/video.rs:269-278`（`extract_frames`）：

```rust
.args(["-vf", &format!("select='{select_expr}'"), "-vsync", "0"])
```

没有 `-ss` seek，`select` 靠帧号过滤——ffmpeg 得从第 0 帧线性解码到最大目标帧号才能抠出稀疏采样的目标帧。`DECODE_TIMEOUT=30s` 是唯一的兜底，长视频大概率直接超时，而不是优雅地按需跳帧。

### vLLM 怎么做

vLLM 支持五种可插拔后端（`VLLM_VIDEO_LOADER_BACKEND`，默认 `opencv`），关键是**默认后端其实和我们犯的是同一类错误**：

```python
# opencv 后端：_read_frames_no_recovery，同样是线性 grab 到目标帧
for idx in range(max_frame_idx + 1):
    ok = cap.grab()
    if idx in frame_indices:
        ret, frame = cap.retrieve()
```

但**非默认的 `pyav` 后端是真 seek**，而且明确是为了解决这个问题设计的：

```python
# PyAVVideoBackendMixin：按目标帧 seek + 向前解码到 PTS
if decoder is None or last_pts is None or pts <= last_pts:
    container.seek(pts, stream=stream)
    decoder = container.decode(video=0)
```
> docstring: "The seek releases the GIL between frames and scales with the number of sampled frames rather than the video length."

`torchcodec` 后端更进一步，是单次批量随机访问：`decoder.get_frames_at(frame_indices)`。

### 方案

两条路，按投入排序：

**低成本**：把当前一次性 `select` 调用换成"每个目标帧单独一次 `ffmpeg -ss <timestamp> -i video -frames:v 1 ...`"。代价是 N 次进程 spawn（我们最多采 `video_max_frames` 张，默认 16，进程开销可接受），换来的是每次 seek 到最近关键帧再向前解一小段，而不是全量线性解码。这个改法不需要新依赖，`video.rs` 的 `ffprobe`/`extract_frames` 结构基本不用大改，只是把"一次 select 多帧"拆成"循环发起单帧请求"（可以并发发起多个 ffmpeg 子进程，`tokio` 已经在用异步 subprocess）。

**高成本、更彻底**：换成 Rust 直接绑定 libav（`ffmpeg-next` crate），拿到跟 vLLM 的 `pyav` 后端等价的能力（seek + decode-to-pts），同时省掉子进程 spawn 开销和管道 IO。这个改法工作量明显更大（要接一个新的重量级依赖、重写整个 `video.rs` 的解码路径），但长期看是对的方向——vLLM 自己也是绕开裸 ffmpeg 子进程、直接用 libav 绑定。

**工作量评估：低成本方案小，高成本方案中等**。建议先上低成本方案（照抄"per-frame seek"这个思路，不需要新依赖），验证效果后再考虑要不要换 `ffmpeg-next`。

**状态：低成本方案已实现，但发现真实的正确性问题，已回退。** 用 `sample.mp4`（真实手机拍摄视频）实测：

1. 朴素版本（`-ss <ts> -i video -frames:v 1`，直接拿 seek 后的第一帧）：跟 `select` 基准比对，5 个采样点里除第 0 帧外全部对不上——差异幅度是 60-78% 的像素字节不同、最大单字节差到 252，明显是解出了完全不同的画面，不是可以接受的解码噪声。根因用 `showinfo` 滤镜实测确认：accurate seek 到 5.0s 后，`-frames:v 1` 拿到的是 seek 点之后**第一个输出帧**（pts_time≈4.9997s，B 帧重排序导致的"领先帧"），而不是真正对齐 5.0s 的那一帧（那一帧其实是第二个输出帧）。
2. 尝试修复：`-ss <ts-2s>`（提前 2 秒 seek 留缓冲）+ `select='gte(t,{ts})'`（按时间戳而非解码序号选帧）——这个组合确实能精确命中目标时间戳（`showinfo` 验证 `pts_time` 精确等于目标值），但进一步测试发现**同一个目标时间戳，从不同的 seek 起点解码出来，画面内容（checksum）不一样**——说明这份文件的解码不是"从任意关键点重新解码都能复现同一帧"，可能跟这份手机录制素材本身的 GOP 结构或轻微变帧率有关，具体根因没有继续深挖。
3. 鉴于"抽错帧"是会静默污染模型输入的高风险 bug（不会报错，只会让模型看错画面），而线性 `select` 方案本次会话全程验证正确，**决定回退到线性 `select` 方案**，`extract_frames` 恢复原样（`fps` 参数保留在签名里但暂时不用，前置下划线标注，方案文档见上），把这项从"已完成"改成"已调查、已回退，需要更彻底的方案"。
4. 真正靠谱的修法应该是设计文档里说的"高成本"路线——绑定 libav（`ffmpeg-next`）直接控制 PTS 级别的解码，而不是拿 `-ss`/`select` 这类 CLI 级别的启发式去猜；vLLM 自己也是这么做的（`pyav`/`torchcodec` 后端），这次踩的坑正好印证了为什么 vLLM 不用裸 ffmpeg 子进程 seek。

---

## 3. 宿主机侧 resize/normalize 是逐帧串行

### 现状

`qwen35_vision_image.rs::prepare_clip` 里对 `raw_frames[1..]` 是一个 `for` 循环，每帧单独调用 `prepare_frame` 做双三次插值 resize——没有 rayon，没有批量化，纯串行。

### vLLM 怎么做

vLLM 自己不做这部分，直接调 HF transformers 的 `Qwen3VLVideoProcessor._preprocess()`（vLLM 只是 `_call_hf_processor` 委托过去），但那份代码的做法值得参考：

```python
grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
stacked_videos = self.resize(videos=stacked_videos, ...)
```

同尺寸的帧先分组、堆成一个 stacked tensor，resize/rescale/normalize 都是一次批量 tensor 操作（PyTorch 的向量化插值），不是 Python 里逐帧循环。没有找到显式 `.to(device=...)`，默认跑在输入张量所在的设备上（vLLM 管线里通常是 CPU）。

### 方案

我们的场景比 HF 简单一些：一次请求里的所有帧本来就是同一个源分辨率（`prepare_clip` 的文档注释已经写明"All frames share `src_h`/`src_w`"），不需要"按形状分组"这一步。真正对得上的优化点是"批量而非逐帧"：
- **最低成本**：给这个循环加 `rayon` 的 `par_iter`，多帧并行 resize——这是纯 CPU 密集型独立任务，没有共享可变状态，改动量很小（加依赖 + 把 `for` 换成 `.par_iter().map()`）。
- **更彻底**：把双三次插值本身向量化（比如把 N 帧的同一 patch 位置一起算，利用 SIMD），但这个改法工作量大、和 vLLM 用 PyTorch 张量批量算的路子不是一回事（我们是手写 CPU 代码，不是调 BLAS），性价比不如直接上 rayon。

**工作量评估：低**。rayon 并行化是几行改动，建议直接做。

---

## 4. `vision_patchify` 是逐帧组单独 launch，不是一次批量 launch

### 现状

`crates/model/src/lib.rs:4646-4648`（`encode_clip`）：

```rust
for ti in 0..grid_t {
    self.kern.vision_patchify(...)
}
```

21 帧组的视频就是 21 次独立 kernel launch，写进 `pixels_h` 的不同偏移。B2 阶段为了先跑通、正确性优先，故意没做成批量 launch——这是明确留下的技术债。

### vLLM 怎么做

`vllm/model_executor/models/qwen3_vl.py`：

```python
# Qwen3_VisionPatchEmbed.forward —— 一次 Conv3d 覆盖整批 patch
def forward(self, x):
    L, C = x.shape
    x = x.view(L, -1, self.temporal_patch_size, self.patch_size, self.patch_size)
    x = self.proj(x).view(L, self.hidden_size)
    return x
```

调用方 `_process_video_input` 把一批请求里所有图片/视频的 patch **拼成一个张量**，`grid_thw` 记录每一项的形状，**只调一次** `self.visual(pixel_values_videos, grid_thw=grid_thw)`，跑完之后才按 `sizes = grid_thw.prod(-1)...` 切回各自的输出。attention 段边界靠一次性算好的 `cu_seqlens`（block-diagonal 注意力掩码）区分,不是 Python 循环里分别跑。

### 方案

这是我们已经在用 `cu_seqlens` 做 attention 分段的（`qwen35_vision::cu_seqlens`，`Grid{t: grid_t, ...}` 整体传一个 grid），说明 attention 那一步其实已经是"批量"的——**真正没批量的只有 patchify 这一步**，纯粹是 host 侧的 for 循环调度问题，不是算法设计问题。

具体改法：把 `vision_patchify` CUDA kernel 的 launch 配置从"每次处理一个 frame-group"改成"一次 launch 覆盖 `grid_t` 个 frame-group"——kernel 内部按 `blockIdx`（比如加一维 `blockIdx.z` 表示 frame-group 编号，或者直接把 `grid_t` 折进现有的 grid 维度）算出每个线程该读 `pixels_h` 输入里的哪个偏移、写输出的哪个偏移，Rust 侧只需要把 `grid_t` 作为一个新参数传给 kernel，不再需要 host 侧 for 循环发射。这属于纯性能改动,不改变任何数值结果,可以拿 B2 阶段已有的 `video_clip_check.rs` example（对比"一次 4 帧调用" vs "两次 2 帧调用"）反过来验证"一次 `grid_t=21` 批量 launch" 和"现在的 21 次循环 launch"数值一致。

**状态：已实现并验证。** `vision_patchify_f32`（`crates/kernels/src/cu/vision.cu`）加了 `grid_t` 参数，把 `total` 从 `grid_h*grid_w*patch_dim` 扩成 `grid_t` 倍，kernel 内部用 `idx / group_total` 算出 `ti`（属于第几个 frame-group）再复用原来的每 patch 索引逻辑；`grid_t=1` 时数学上精确退化成旧行为。`Kernels::vision_patchify` 加了对应参数，`encode_clip` 从 `for ti in 0..grid_t { launch }` 改成一次 launch。Metal 没有 vision kernel，不用同步改。验证：`crates/kernels/tests/vision.rs` 18 个测试全过（含 `the_patchify_kernel_emits_block_order_and_channel_temporal_row_column`）；`video_clip_check.rs` 重跑，`group 0/1 max abs diff` 分别是 `4.730e-4`/`6.485e-4`——跟这次会话更早、改动前测出的数字**完全一样**，证明批量 launch 和逐组循环在数值上等价。

**工作量评估：中等**。要改 CUDA kernel 本身（`crates/kernels/src/cu/vision.cu` 或对应文件）加一个批量维度,以及 Rust 侧的 launch 配置,但不涉及调度器/M-RoPE 这些更容易出错的地方,而且有现成的数值一致性验证手段。

---

## 5. 没有内容感知的 token 剪枝

### 现状

`vision_resize_video` 单纯 `max_patches / grid_t` 平均分配,不管哪个帧组信息量大、哪个是静止背景,分辨率(进而 token 数)都一样。

### vLLM 怎么做

`vllm/multimodal/video_prune/` 下有两种真实实现,都是**跑完视觉塔之后**、在 token 层面剪枝(不是丢整个帧):

- **EVS**(`evs.py`):对每个 spatial token,算它和"上一帧同一空间位置"的 cosine 相似度,取 `1 - similarity` 作为分数,全局 top-k 保留,第 0 帧强制全保留:
  ```python
  dissimilarity = 1 - F.cosine_similarity(video_embeds[1:], video_embeds[:-1], dim=-1)
  order = torch.argsort(dissimilarity_flat, descending=True)
  topk_indices = order[:retain_num_tokens]
  ```
- **VidCom2**(`vidcom2.py`):对每帧算"和全局/本帧中心的多尺度高斯相似度"当冗余分数,再用 softmax 把 token 预算动态分配给"信息量大的帧"(内容独特的帧分到更多 token,不是均分):
  ```python
  probs = F.softmax((frame_scores - frame_scores.max()) / TEMP, dim=0)
  scales = (base * (1.0 + probs - probs.mean())).clamp(max=1.0)
  ```

关键细节:
- **默认关闭**,靠 `video_pruning_rate: float | None = None` 开关(`vllm/config/multimodal.py:210`),必须显式设一个 `>0` 的值才生效。
- Qwen3-VL 同时支持这两种(`supported_video_pruning_methods = ("evs", "vidcom2")`),Qwen2.5-VL 只支持 EVS。
- 剪枝之后 M-RoPE 位置要重新算(`evs.py::recompute_mrope_positions`)——因为保留下来的 token 不再是"每帧固定数量"这个规整形状了,序列变成"每帧组保留 token 数不等",原来"这一段是第几帧组"的位置推导逻辑要跟着改。

### 方案

**不建议现在做**,原因是这一项和第 1 项(跨 step 分块)、这次刚做完的 M-RoPE 位置计算耦合最深:

- 我们现在 `mrope_for_video`/`llm_position_ids` 全部假设"每个帧组的 token 数在跑视觉塔*之前*就已知、固定"(`vision_resize_video` 返回的 `tokens` 直接决定 placeholder run 长度,`prompt.splice` 在真正跑塔之前就把 token 数定死了)。剪枝会打破这个假设——tower 跑完之后才知道哪些 token 被留下,而我们的 prompt 拼装/M-RoPE 位置计算发生在 tower 跑*之前*(为了让 `encode_clip` 之后能立刻按已知长度 splice)。
- vLLM 能做剪枝,前提是它的 `EncoderCache` 架构本来就把"tower 输出"和"prompt token 序列长度"解耦得很彻底(#1 里那个"chunk 边界不对齐 patch group"的发现,本质上和这里是同一个架构决策的两个体现)。我们要先完成第 1 项(把 vision 消费从"prompt 长度提前定死"改成"按 tower 实际输出动态切片"),剪枝才有地方挂。

等第 1 项落地之后,EVS 是更适合先做的(算法比 VidCom2 简单——单纯是"和前一帧的 cosine 距离 + 全局 top-k",不需要 VidCom2 那套多尺度高斯核),而且 vLLM 两个模型都支持它,证明它在真实场景够用。

**工作量评估:高,且依赖第 1 项**。建议排在最后。

**状态:第 1 项落地后重新评估,选择了一个更保守的替代方案,不是完整 EVS。** 完整 EVS 在 token 粒度剪枝(同一帧组内一部分 token 留、一部分丢),这要求"整个 clip 的最终 token 数"在跑完塔之后才知道,而我们的 prompt 拼装(时间戳文字 + placeholder 游程长度)必须在跑塔*之前*就定下来——vLLM 自己也要专门写一个 `recompute_mrope_positions` 来处理这个错位,是这次调研里公认最容易出錯的一块。加上第 1 项那次真实踩到的教训(视频占位符的"不连续游程"结构比想象中更容易在边界算错),继续往 token 粒度深挖的风险回报比不划算。

实际做的是**帧组粒度**的去重,而不是 token 粒度:整段丢弃跟"上一个被保留的帧组"高度相似的帧组,不拆开某一组内部的 token。好处是完全复用已经验证过的下游流程——`vision_resize_video`/`prepare_clip`/`encode_clip`/M-RoPE 位置计算全都已经是"随便多少个 frame-group 都能处理"的通用逻辑(`grid_t` 本来就是运行时决定的),丢帧组只是让 `grid_t` 变小,不需要新代码路径。而且判重逻辑跑在**解码出的原始像素**上、在视觉塔之前,不只是省 token,连视觉塔的算力也一起省了(vLLM 的 EVS 是塔跑完之后再丢,像素级冗余的算力已经花掉了)。

具体实现(`crates/server/src/scheduler.rs::drop_redundant_groups`):按帧组比较"和上一个*被保留*帧组"（不是紧邻的上一组，防止缓慢渐变的画面每步都"差异不大"从而全部被保留下来）的平均逐通道像素差值，超过 `--video-dedup-threshold`（默认 0.0，即关闭，跟 vLLM `video_pruning_rate` 的"默认关闭"惯例一致）才保留。第 0 组永远保留。

验证:
- 4 个纯函数单测(`dedup_tests` 模块),包括手工验证"和上一个保留帧组比较，不是紧邻上一组"这条关键规则。
- 真实 27B + sample.mp4:阈值 5.0 时 21 个帧组全部保留(kept=21/21——这段真实素材全程运动，没有真正意义上的冗余帧，这本身也是诚实的验证结果，不是没测出来）；阈值 40.0 时激进剪到 2 个帧组(kept=2/21)，请求依然正确完成，模型的推理明显反映了信息丢失（时长判断从 21 秒变成约 16-17 秒——丢了大半信息之后模型如实地"看到更少"，是预期中的、诚实的行为，不是错误）。

## 6. 采样帧率是编译期常量，不是请求级参数

**状态：已实现并验证。** `api.rs` 的 `VideoUrl` 加了 `fps: Option<f64>`；`video.rs::decode_video_data_url`/`decode_bytes`/`sample_frame_count` 都改成接收一个具体的 `target_fps: f64`（不是 `Option`，precedence 在 `routes.rs` 一处解析：`v.fps.unwrap_or(engine.video_target_fps)`）；`--video-target-fps` CLI flag（默认 2.0）。端到端验证：同一段 sample.mp4，默认 fps=2 给出 `prompt_tokens=1190`，显式传 `fps=6` 给出 `prompt_tokens=1548`——采样密度确实按请求变化，且两种密度下模型都正确读出约 20-21 秒的真实时长。用 `--video-max-frames 200`（此前默认 16 会把两种 fps 都截到同一个帧数，掩盖差异）测出来的。

### 现状

`crates/server/src/video.rs:52`：

```rust
const TARGET_FPS: f64 = 2.0;
```

进程启动后，所有请求永远用同一个采样密度，没有任何请求级的入口——想要更密的采样（比如快速运动场景）或更稀的采样（比如长视频只要个大概），唯一办法是改代码重新编译、重启服务。

### vLLM 怎么做

`fps` 是货真价实的**请求级**参数，不是服务端启动时锁死的：

```python
# vllm/entrypoints/openai/chat_completion/protocol.py:364
media_io_kwargs: dict[str, dict[str, Any]] | None = Field(
    ...
    "keyed by modality. Merged with engine-level media_io_kwargs."
)
```

链路是：请求体的 `media_io_kwargs["video"]["fps"]` → `chat_utils.py` 里每个请求一份的 `tracker.media_io_kwargs` → `VideoBackend.compute_frames_index_to_sample(fps=target.fps, ...)`（`vllm/multimodal/video.py:1019`，签名默认 `fps: int = -1`，`-1` 是"用视频原始帧率"的哨兵值）。服务端在 `vllm/config/multimodal.py:125` 有一份引擎级默认（`media_io_kwargs: dict = Field(default_factory=dict)`），请求里传的值会覆盖它——两层default/override 结构，不是非此即彼。

Gemini API 的 `videoMetadata.fps` 是同一个设计思路的另一个实现（前面调研过），说明这不是 vLLM 一家的孤例，是这个问题的通用解法。

### 方案

fps 只影响"decode 阶段采多少帧"，不影响 prompt 模板渲染（模板永远只渲染一个 `<|video_pad|>` 占位符，真正的帧数/token 数是在 `encode_pending_video` 里、拿到解码结果之后才展开的）——所以这个改动不需要碰 `tokenizer/chat.rs`，链路比想象的短：

1. **API 层**（`crates/server/src/api.rs`）：`ContentPart` 现在的 `video_url: Option<ImageUrl>` 是和图片共用的 URL 包装结构。给视频一个自己的包装类型（不要塞进 `ImageUrl` 里，图片不需要这个字段），比如：
   ```rust
   pub struct VideoUrl {
       pub url: String,
       #[serde(default)]
       pub fps: Option<f64>,
   }
   ```
   `Message::video_urls()` 现在只返回 `Vec<&str>`，得跟着改成能带出 `fps` 的形式（比如 `Vec<(&str, Option<f64>)>` 或者返回 `&VideoUrl` 本身）。
2. **路由层**（`routes.rs`）：`chat_completions` 里调用 `crate::video::decode_video_data_url` 的地方，把请求携带的 `fps`（没传则 `None`）一起传下去。
3. **解码层**（`video.rs`）：`decode_video_data_url(url, max_frames, fps: Option<f64>)` 新增一个参数，`sample_frame_count` 内部用 `fps.unwrap_or(server_default_fps)` 代替直接读 `TARGET_FPS` 常量。
4. **服务端默认值**：`TARGET_FPS` 常量退化成"没人传参时的默认值"，可以顺带做成一个新的 CLI flag（`--video-target-fps`，跟现有的 `--video-max-frames` 是同一种模式，`main.rs`/`engine.rs` 已经有现成的线路可以照抄），这样运维也能不改代码就调整默认密度。

这个改动和第 2 项（ffmpeg seek）是互补而不是重复的关系：第 2 项解决"怎么高效抽指定的帧"，这一项解决"谁来决定抽哪些帧对应的密度"。两个可以分开做，但顺序上建议**先做这项**——参数链路打通之后，第 2 项的"per-frame seek"重构正好可以直接用上一个可能变化的 `fps`/目标帧数，不用先写死再返工。

**工作量评估:低**。四层改动都是"加一个可选参数往下传"，不涉及并发/调度/kernel，是这批里最省事的一项，而且不依赖任何其他项。

---

## 优先级建议

| # | 项目 | 工作量 | 是否依赖其他项 | 收益 |
|---|---|---|---|---|
| 1 | 跨 step 分块 prefill | 中 | 无 | 解除"视频长度被单步预算卡死",不用再靠降分辨率凑预算 |
| 6 | fps 变成请求级参数 | 低 | 无 | 不改代码就能按场景调采样密度,是第 2 项的前置准备 |
| 3 | rayon 并行 resize | 低 | 无 | 长视频/多帧场景端到端延迟下降 |
| 2 | ffmpeg per-frame seek | 低(简单版) | 无(建议排在第 6 项之后) | 长视频抽帧不再要求线性解码全片 |
| 4 | 批量 `vision_patchify` launch | 中 | 无 | 多帧组 kernel launch 开销下降 |
| 5 | EVS 内容感知剪枝 | 高 | 依赖 #1 | 同样 token 预算下画质/帧数上限提升,但架构改动最深 |

建议顺序:**1 → 6 → 3/2(并行做)→ 4 → 5**。第 1 项优先级最高不只是因为收益大,也因为它是第 5 项的前置依赖；第 6 项排第二是因为工作量最低、且是第 2 项重构的自然前置（先打通参数链路，再重构抽帧方式，不用返工）；晚做第 1 项不会更省事。

---

## 跟真实 vLLM 的对比测试

bw 上 `192.168.2.8:8080` 正跑着 huicheng 的真实 vLLM 实例(`Qwen3.5-397B-A17B-GPTQ-Int4`,3 卡 pipeline-parallel)——是同事的共享生产环境,不是我起的测试实例,所以这里只做了**一次、只读、低成本**的请求做对比,没有做压测/吞吐对比(那需要独占资源或者自己另起一个实例,超出这次授权范围)。

**同一份 `sample.mp4`,同一个问题("这段视频大概持续多少秒"),两边都给出正确答案(20 秒,真实时长 21.03s),但 token 开销差一个数量级:**

| | 我们 (27B, 默认配置) | vLLM (397B, 默认配置) |
|---|---|---|
| prompt tokens | ~1094-1190 | **12301** |
| 回答 | 正确(20/21) | 正确(20) |
| 端到端延迟 | 数秒 | 7.84s |

两边视觉塔配置是**完全一样的**(这次会话早些时候已经拉过 397B 的 HF config 核对过:`temporal_patch_size=2`、`mrope_section=[11,11,10]`、`patch=16`、`merge=2` 一字不差),所以这个 10 倍的差距不是模型架构差异,大概率是**默认采样密度不同**——vLLM 的 `VideoBackend.compute_frames_index_to_sample`(这次会话早些时候读过的真实源码)`fps` 参数默认是 `-1`,注释写的是"用视频原始帧率";这次请求没有显式传 `fps`,如果这个部署走的是这条默认路径,21 秒 × 30fps ≈ 630 帧全部拿去采样,而我们的默认是 `DEFAULT_TARGET_FPS=2.0`,同样 21 秒只采 ~42 帧、又被 `--video-max-frames`(默认 16)进一步砍到 16 帧——这是一个合理推断,不是我能直接确认对方部署内部配置的断言。

这个数字反过来印证了两件事:
1. **这次做的优化方向是对的**:同样正确回答,我们的 token 成本只有对方的 1/10,主要是因为默认更保守的 fps/max_frames——如果我们把默认值也调到"接近原始帧率",大概率会付出类似的 token 代价,除非配合第 5 项(帧组去重)或更彻底的第 5 项完整版(token 级剪枝)。
2. **优化 6(fps 请求级参数)的价值是双向的**:不仅是"该场景需要更密时能调高",反过来也说明"默认给多密"这个选择本身就是一个需要谨慎权衡的产品决策——vLLM 选择了"默认接近原始帧率、开销大但信息完整",我们选择了"默认保守、开销小但可能丢细节",两种取舍谁对不取决于代码,取决于场景(这也是这次 fps 优化的设计初衷:让调用方自己决定,而不是替所有场景做同一个默认假设)。
