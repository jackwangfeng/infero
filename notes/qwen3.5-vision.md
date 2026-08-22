# Qwen3.5 视觉塔 (Qwen3_5VisionModel) 架构规格

来源：`transformers` 的 `models/qwen3_5/modeling_qwen3_5.py`（`Qwen3_5Vision*` 类）、
`transformers/vision_utils.py`、`models/qwen2_vl/image_processing_qwen2_vl.py`，
配 `/home/jeff/models/qwen38-27b-fp8` 的真实张量形状。**代码是权威，config 字段不是**，
而且视觉侧比文本侧更甚——见下面 `deepstack_visual_indexes`、`out_hidden_size`、
`hidden_act` 三条。

写这份东西的理由和 `qwen3.5-architecture.md` 一样：这些细节每一条弄反都不会崩，
只会产出对着错误图像的流畅描述。视觉塔的陷阱密度比文本侧高，因为它**几乎每一条约定
都和文本侧相反**：

| | 文本塔 | 视觉塔 |
|---|---|---|
| 归一化 | RMSNorm，无 bias | **LayerNorm，有 bias，减均值** |
| MLP | SwiGLU，三个矩阵 | **fc1 / GELU / fc2，两个矩阵** |
| 线性层 bias | 全都没有 | **每一个都有** |
| 注意力 | causal、GQA、带输出门 | **双向、16 个等价头、无门** |
| q/k/v 打包 | q 与其门**按头交错** | **`[全部 q \| 全部 k \| 全部 v]`** |
| RoPE 宽度 | 256 维里转前 64（partial） | **全部 72 维都转** |
| RoPE theta | 1e7 | **1e4** |
| RoPE 轴 | 3 轴，按 `i % 3` 交错 | **2 轴，连续分块** |

把文本侧的习惯带进视觉塔，每一条都能跑完。下面每一条都有 `tools/capture_qwen35_vision.py`
在真实 checkpoint 上跑参考实现拿到的数字，以及"另一种读法偏离多远"的实测值。
**另一种读法偏离多远** 才是这份文档的价值——只说"正确的是这个"，读者无法判断
写错了会不会被现有测试抓住。

## 尺寸（`config.json` 的 `vision_config`）

```
depth                27      hidden_size          1152
num_heads            16   -> head_dim  72          （不是 2 的幂！）
intermediate_size    4304    out_hidden_size      5120
in_channels          3       patch_size           16
temporal_patch_size  2       spatial_merge_size   2
num_position_embeddings 2304 -> 学习到的位置网格是 48 x 48
hidden_act           "gelu_pytorch_tanh"
deepstack_visual_indexes  []
```

派生量：

```
patch_dim   = 3 * 2 * 16 * 16 = 1536     一行 pixel_values 的宽度
merge_unit  = 2 * 2 = 4                  几个 patch 合成一个 LLM token
rope_dim    = head_dim / 2 = 36          频率表宽度（h 占 18，w 占 18）
resize 粒度 = patch_size * merge = 32     不是 16
LayerNorm eps = 1e-6（三处 LayerNorm 全是硬编码 1e-6，config 里没有这个字段）
vision rope theta = 10000.0（`Qwen3_5VisionRotaryEmbedding` 的默认值，
                             构造时没有传，config 里也没有）
```

三个 config 陷阱：

- **`out_hidden_size` 必须从 config.json 读，不能用类默认值。** `Qwen3_5VisionConfig`
  的类默认是 3584（9B 的尺寸），这个 checkpoint 是 5120。这条至少会大声失败——
  merger 输出接不上语言模型——但它说明"vision_config 的字段可以缺省"这个假设不成立。
- **`hidden_act` 只描述了 27 个 block 的 MLP，没有描述 merger。** merger 用的是
  `nn.GELU()`，精确 erf 版本；block 用的是 tanh 近似。一座塔里两个 GELU，config 只提了一个。
- **`deepstack_visual_indexes` 是空数组，checkpoint 里也没有对应权重。** Qwen3-VL 有
  deepstack（把中间层视觉特征注入 LLM 前几层），这份权重把它关了。不要照着
  Qwen3-VL 的实现补上。

`config.json` 顶层（不在 `vision_config` 里）：

```json
"image_token_id": 248056, "video_token_id": 248057,
"vision_start_token_id": 248053, "vision_end_token_id": 248054
```

**不是 151655 / 151656。** 那两个是 Qwen2-VL / Qwen2.5-VL 的 152k 词表里的 id；
在这份 248320 词的词表里它们解码成泰语文本片段（`tokenizer.convert_ids_to_tokens(151655)`
= `'à¸ļà¸²à¸¥'`）。把旧常量抄过来的后果是：占位符变成两个随机词，序列长度对得上，
`get_placeholder_mask` 的计数检查也过不了——但如果你自己实现拼接而不做计数检查，
它就只是安静地把图像特征丢进虚空。对照：

```
248053 '<|vision_start|>'   248054 '<|vision_end|>'
248055 '<|vision_pad|>'     248056 '<|image_pad|>'    248057 '<|video_pad|>'
```

## 权重清单（层 0 / merger 实测，全部来自 `outside.safetensors`）

```
model.visual.patch_embed.proj.weight  BF16 [1152, 3, 2, 16, 16]
model.visual.patch_embed.proj.bias    BF16 [1152]
model.visual.pos_embed.weight         BF16 [2304, 1152]

model.visual.blocks.{0..26}.norm1.weight / .bias        BF16 [1152]
model.visual.blocks.{0..26}.norm2.weight / .bias        BF16 [1152]
model.visual.blocks.{0..26}.attn.qkv.weight             BF16 [3456, 1152]
model.visual.blocks.{0..26}.attn.qkv.bias               BF16 [3456]
model.visual.blocks.{0..26}.attn.proj.weight            BF16 [1152, 1152]
model.visual.blocks.{0..26}.attn.proj.bias              BF16 [1152]
model.visual.blocks.{0..26}.mlp.linear_fc1.weight       BF16 [4304, 1152]
model.visual.blocks.{0..26}.mlp.linear_fc1.bias         BF16 [4304]
model.visual.blocks.{0..26}.mlp.linear_fc2.weight       BF16 [1152, 4304]
model.visual.blocks.{0..26}.mlp.linear_fc2.bias         BF16 [1152]

model.visual.merger.norm.weight / .bias   BF16 [1152]     <- 1152，不是 4608
model.visual.merger.linear_fc1.weight     BF16 [4608, 4608]
model.visual.merger.linear_fc1.bias       BF16 [4608]
model.visual.merger.linear_fc2.weight     BF16 [5120, 4608]
model.visual.merger.linear_fc2.bias       BF16 [5120]
```

共 **333 个张量，460.7M 参数，BF16 879 MiB**。27×12 + 6 + 2 + 1 = 333，没有多余的，
也没有 deepstack merger。

**整座视觉塔是 BF16，没有一个 FP8 张量，没有 `weight_scale_inv`。**
`quantization_config.modules_to_not_convert` 共 882 项，其中 246 项含 `visual`，
覆盖 block 0..26 的 `attn.qkv` / `attn.qkv_proj` / `attn.proj` /
`mlp.linear_fc1` / `mlp.linear_fc2`，加上 `patch_embed.proj`、`pos_embed`、
`merger.*`，且每一项都有 `model.visual.` 和裸 `visual.` 两种前缀拼写——
即整座塔都被排除在量化之外。所以视觉侧的加载器**不需要**块级反量化路径。
但 `tools/capture_qwen35_vision.py` 的 `load_f32` 仍然保留了 FP8 分支：
"我看的那份导出恰好是 BF16"不是可以硬编码的前提，
以后有人量化了视觉塔，静默读垃圾比报错糟得多。

**`modules_to_not_convert` 里有三个不存在的模块。** 它列出了
`visual.deepstack_merger_list.{0,1,2}.{norm, linear_fc1, linear_fc2}`，
但 checkpoint 里**没有任何 deepstack 张量**（333 个视觉张量 = 27×12 + 6 + 2 + 1，
全部占满），`deepstack_visual_indexes` 也是空数组。
这份量化配置显然是从一个带 3 个 deepstack merger 的变体生成的，然后跟着导出被抄了过来。
所以：**不要用 `modules_to_not_convert`（或任何 config 里的名字）来判断结构存在与否。**
权威是张量清单加 `deepstack_visual_indexes`。同理它也列出了
`attn.qkv` 和 `attn.qkv_proj` 两个名字，实际权重只叫 `attn.qkv`。

`merger.norm.weight` 是 `[1152]` 这一条本身就是证据：它排除了 post-shuffle 归一化
（见下面 merger 一节），不需要读代码就能定。

## 预处理：动态分辨率

`preprocessor_config.json`（图像）/ `video_preprocessor_config.json`（视频）：

```
patch_size 16   temporal_patch_size 2   merge_size 2
image_mean [0.5, 0.5, 0.5]   image_std [0.5, 0.5, 0.5]
resample = BICUBIC (PIL 3)   rescale_factor = 1/255
图像: size.shortest_edge 65536      size.longest_edge 16777216
视频: size.shortest_edge 4096       size.longest_edge 25165824
```

**`image_mean` / `image_std` 是 0.5，不是 CLIP 的均值方差。**
`Qwen2VLImageProcessor` 的**类默认**是 `OPENAI_CLIP_MEAN/STD`，config 覆盖了它。
所以像素归一化是 `2 * (x/255) - 1`，落在 `[-1, 1]`。
用 CLIP 的均值会跑完，只是每个 patch 都偏移一点，表现为描述"大致对但细节错"。

`shortest_edge` / `longest_edge` 是**像素总数**，不是边长——65536 = 256×256，
16777216 = 4096×4096。名字骗人。

### `smart_resize`

```python
h_bar = round(h / factor) * factor        # factor = patch_size * merge_size = 32
w_bar = round(w / factor) * factor
if h_bar * w_bar > max_pixels:
    beta = sqrt(h * w / max_pixels)
    h_bar = max(factor, floor(h / beta / factor) * factor); 同理 w_bar
elif h_bar * w_bar < min_pixels:
    beta = sqrt(min_pixels / (h * w))
    h_bar = ceil(h * beta / factor) * factor; 同理 w_bar
```

**`factor` 是 32 = `patch_size * merge_size`，不是 16。** 按 16 取整会得到奇数网格，
然后 `h // merge` 截断：塔照样跑，merger 眼里静默少一行一列 patch，
而位置场是按完整网格算的——两者错位。测试
`smart_resize_reproduces_the_reference_table` 会检查按 16 取整确实能产生奇数网格。

两条边界行为：
- 长宽比 > 200:1 时参考实现**抛异常**而不是缩放。Rust 侧 `smart_resize` 返回
  `Option`，这种情况返回 `None`。
- `round` 是 Python 的 banker's rounding（round-half-to-even）。对 factor=32 来说，
  h/32 恰好是 .5 的输入是 16、48、80 这类缩略图尺寸，不算罕见。

token 预算：

```
图像  最大 16777216 px -> 65536 patch -> 16384 个 LLM token
      最小    65536 px ->   256 patch ->    64 个 LLM token
视频  最大 25165824 px -> 98304 patch -> 24576 个 LLM token
```

### patchify：patch 顺序和 patch 内布局

这是整个视觉塔最容易弄反、且后果最严重的一处。

`patchify` 把 `[B, C, H, W]` reshape 成
`[B, C, gh/m, m, P, gw/m, m, P]`，permute 到
`[B, gh/m, gw/m, m, m, C, P, P]`，在 `C` 和两个 `P` 之间插入时间轴并 expand，
最后 flatten 成 `[B, gh*gw, C*T*P*P]`。两个结论：

**1. patch 按 spatial-merge 块序排列，不是光栅序。**

```
p = ((block_row * blocks_w + block_col) * merge + in_row) * merge + in_col
row = block_row * merge + in_row     col = block_col * merge + in_col
```

即**连续 4 个 patch 构成一个 2x2 空间块**。这正是 merger 的 `view(-1, 4 * hidden)`
能当 2x2 池化用的原因——merger 自己不做任何重排。
用光栅序（`p = row * grid_w + col`）会跑完，
而且会让 merger 去平均横向排开的四个 patch，同时位置嵌入描述的是另一种排列。
在 6x8 网格上，48 个 patch 里有 36 个位置对不上。

**2. patch 内 1536 个数的顺序是 `(c, t, y, x)`，stride 递减。**

```
slot(c, t, y, x) = ((c * temporal_patch + t) * patch + y) * patch + x
```

`(t, c, y, x)`、`(c, t, x, y)`、`(y, x, c, t)` 都填满全部 1536 个槽位、都通过类型检查，
都只是把转置后的 patch 喂给 Conv3d。测试
`the_patch_layout_is_channel_temporal_row_column_in_block_order` 用一张
`arange` 图（每个像素都是不同的数）把这四种读法全部区分开。

**3. 静止图像的两个时间槽装的是同一帧，不是一帧加一份零。**
`patchify` 用 `unsqueeze(6).expand(..., temporal_patch_size, ...)`。
所以 Conv3d 的两个时间 tap 看到的是同样的像素，等效于两组权重求和。
只填第一个 tap、第二个置零会跑完（实测偏离 **0.83 / 峰值 3.15**）。
视频路径下这两个 tap 装的是两个连续帧；帧数为奇数时
`Qwen3VLVideoProcessor` 复制最后一帧补齐。

顺便记一条不属于这一节但属于同一类的：整批丢掉 `patch_embed.proj.bias`
偏离 **3.05 / 峰值 3.15**。视觉塔每个线性层都有 bias，文本塔一个都没有，
所以套用文本塔的加载器会静默丢掉每层 12 个 bias 张量——
和 AWQ 加载器丢掉 Qwen 的 QKV bias 是同一类错误，同样读成流畅的胡话。

所以 **`grid_thw` 的 `t` 数的是时间 patch，不是帧**：4 帧的片段 `t = 2`。

## 前向

```
pixel_values [N, 1536]  ->  patch_embed  ->  [N, 1152]
                         +  interp(pos_embed) ->  [N, 1152]
                        ---------------------------------
hidden [N, 1152] --(27 x VisionBlock, 段内双向注意力)--> [N, 1152]
                        ---------------------------------
                         ->  PatchMerger  ->  [N/4, 5120]
```

`N = sum(t * h * w)`，把一次调用里所有图像/视频的所有 patch 打包在一起。
`forward` 返回 `last_hidden_state`（merger 之前，`[N, 1152]`）和
`pooler_output`（merger 之后，`[N/4, 5120]`）；**语言模型用的是后者**。

### patch_embed：一个 GEMM，不是卷积

```python
kernel_size = [temporal_patch_size, patch_size, patch_size]   # [2, 16, 16]
proj = nn.Conv3d(3, 1152, kernel_size=kernel_size, stride=kernel_size, bias=True)
hidden = proj(x.view(-1, 3, 2, 16, 16)).view(-1, 1152)
```

kernel 等于 stride，输入已经按 patch 切好，所以这是**每个 patch 一次
`[1152, 1536] @ [1536]` 的矩阵乘加 bias**，没有滑窗、没有 padding、没有重叠。
把它当真卷积写是白做功。`proj.weight` 从 `[1152, 3, 2, 16, 16]` flatten 到
`[1152, 1536]` 是零成本的 view，加载器可以直接指过去。

捕获脚本用**单位向量驱动参考 Conv3d**（`patch_embed(eye) - bias`）来证明这个
flatten 顺序，而不是靠我自己 reshape 一遍——这是"证据"和"转写"的区别。

### 位置嵌入：48x48 学习网格双线性重采样

`pos_embed` 是 `nn.Embedding(2304, 1152)`，即一张 48x48 的网格。
每张图按自己的 `(h, w)` 网格重采样：

```python
src = index * (side - 1) / max(size - 1, 1)      # align_corners = True
```

- **`align_corners = True`**，由 `Qwen3_5VisionModel.__init__` 设定。
  而库里 `get_vision_interpolation_indices_and_weights` 的**默认值是 False**
  （那会给 `(index + 0.5) * side / size - 0.5`）。用 False 会跑完，
  位置嵌入偏离 **5.31 / 峰值 6.60**——位置场变成了图像的另一个函数，
  流畅度不受影响，空间定位垮掉。
- 每个 patch 4 个 tap（每轴 2 个的外积），权重和恒为 1。
- gather 的 patch 顺序**也是块序**（`spatial_merge_size > 1` 走块序分支）。
  用光栅序 gather 偏离 **4.83 / 峰值 6.60**。

捕获脚本额外拿 `F.interpolate(mode='bilinear', align_corners=True)` 做了一次
独立对照——tap/weight 的来源和 oracle 不是同一个文件。

### VisionBlock

```python
h = h + attn(norm1(h))      # norm1 = LayerNorm(1152, eps=1e-6)
h = h + mlp(norm2(h))       # norm2 同上
```

pre-norm，残差走**未归一化**的流。

**归一化是 LayerNorm，不是 RMSNorm。** 减均值、有 bias。实测（block 0）：

```
LayerNorm                     偏离 6.0e-7   （峰值 5.09）
RMSNorm + bias                偏离 0.319
RMSNorm 无 bias（只是不减均值）  偏离 2.95
```

**残差流会长四个数量级。** block 0 输入峰值 8.6、逐行方差 ~0.14；
27 个 block 之后 `last_hidden` 峰值 **4184**、逐行方差 21 ~ 1.6e4。
两个后果：

- merger 的 LayerNorm 是承重的，不是装饰。
- f16 上限 65504，塔顶只剩 ~15x 余量；这条残差流应该走 bf16 或 f32 累加。
- 这也是 host reference 里 `layer_norm_rows` 用 f64 做归约的原因：
  1152 个量级 ~4000 的值顺序 f32 求和有 ~7e-4 相对误差，
  在量级 0.02 的输出上表现为 1.2e-5 的绝对偏差，正好压过布局测试想用的容差。
  这是精度问题不是布局问题，但把它消掉之后"和参考不符"才重新等于"有 bug"。

#### 注意力

```python
q, k, v = qkv(h).reshape(seq, 3, heads, -1).permute(1, 0, 2, 3).unbind(0)
q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
# 每个 cu_seqlens 段单独做 softmax(q k^T * head_dim**-0.5) v，非 causal
out = proj(attn_out.reshape(seq, -1))
```

**q/k/v 是 `[全部 q | 全部 k | 全部 v]` 三个连续的 1152 块。**
`3` 在 `heads` 之前，所以 3456 列分成三段。
这和文本侧**正好相反**：文本侧 `q_proj` 的输出是 `view(..., heads, 2 * head_dim)`
然后切最后一维，q 和它的门**按头交错**。两个都是
`reshape(...).permute/unbind`，差别只在 `3` 放在哪一维。
按头交错读这个张量会得到三个形状正确的张量，
block 0 注意力输出偏离 **6.03 / 峰值 1.87**——完全不相干的东西，流畅地。

对于头 0 的第 0 维，两种读法指向同一行权重。所以布局测试必须探到第一个头之后
（`qkv_is_three_contiguous_blocks_not_interleaved_per_head` 探 head 0/1/15）。

**注意力不是 causal。** `Qwen3_5VisionAttention.is_causal = False`。
套用文本塔的注意力 kernel 会让每个 patch 看不到自己右下方的一切，
输出偏离 **2.08 / 峰值 1.87**——即整个替换掉。

**注意力段是按帧切的，不是按 grid 条目切的。**

```python
seqlens = repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
cu_seqlens = pad(seqlens.cumsum(0), (1, 0))
```

`t` 帧的视频是 `t` 个独立的 `h*w` 注意力块。让注意力跨帧会混合模型从没打算
互相看到的帧。也就是说视觉注意力的代价对 `t` 是**线性**的，不是平方——
定 kernel 尺寸之前值得知道。

打包不变量（`packing_images_and_frames_leaves_each_ones_output_alone` 在测）：
把两张不同形状的图，或一段两帧不同内容的视频打进一次调用，
每一份的输出必须逐位不变。实测 block 0 输出偏离 0（qkv/attn 段完全相同），
穿过 27 层后 `last_hidden` 相对漂移 3e-6，`image_embeds` 1.5e-4——f32 累加噪声。
用两帧**相同**内容的视频测不出东西：对重复的 key 集合做 softmax 结果不变。

#### 视觉 RoPE

```python
rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2)   # dim = 36
inv_freq = 1.0 / (10000.0 ** (arange(0, 36, 2).float() / 36))   # 18 个频率
freqs = (position_ids.unsqueeze(-1) * inv_freq).flatten(1)      # [N, 2, 18] -> [N, 36]
emb = cat((freqs, freqs), dim=-1)                               # [N, 72]
cos, sin = emb.cos(), emb.sin()
```

三处陷阱，都是文本侧习惯带过来会犯的：

1. **频率表 36 个槽位，指数除以 36，不是除以 head_dim=72。**
   和文本侧 partial rope 同一种错法："72 维频率表的前 18 维"是**另一张表**。
   按 head_dim 归一化偏离 **0.49 / 峰值 1.87**。
2. **theta 是 1e4。文本侧是 1e7。** 用 1e7 偏离 **0.23**。
   这个值在 `Qwen3_5VisionRotaryEmbedding` 的默认参数里，
   构造时没传，`vision_config` 里也没有——只能从代码读。
3. **h/w 是连续分块的，不是交错的。** `[N, 2, 18].flatten(1)` 给出
   `[h*f0..h*f17, w*f0..w*f17]`。**同一份 checkpoint 的文本侧 mRoPE
   按 `i % 3` 交错三个轴。** 两套方案一个模型。
   在这里交错 h/w 偏离 **0.51**。

`rotate_half` 配对是 `(i, i + 36)`，所以 `cat((emb, emb))` 的复制是必要的：
维 `i` 和维 `i+36` 载同一个角度。整个头都转，没有文本侧那样的未旋转尾部。
用相邻配对 `(2i, 2i+1)` 偏离 **0.26**。

不依赖参考实现的不变量：**把所有 patch 位置整体平移一个常数，注意力输出必须不变**
（`shifting_all_patch_positions_does_not_change_the_attention_output` 在
offset 1/7/40 上测，实测相对漂移 < 1e-4）。RoPE 只编码相对位置，
所以这条成立；不成立就说明频率表或配对写错了。

另外：**视觉 RoPE 只有 `(h, w)` 两个轴，没有时间轴。**
`get_vision_position_ids(include_temporal=False)` 把 `(h, w)` 对**重复 `t` 遍**。
所以视频的两帧携带完全相同的 rotary 相位，区分它们的只有注意力段。

#### MLP

```python
linear_fc2(gelu_pytorch_tanh(linear_fc1(x)))
```

**两个矩阵，不是三个。** 文本塔的 MLP 是 SwiGLU（gate / up / down），
那里的 `intermediate_size` 是其中两个的宽度。这里 4304 就是一个宽度
（4304 = 16 × 269，269 是质数，没有自然的对半分法），
SwiGLU 读法得自己编一个切分点。用 silu 代替 gelu 偏离 **0.19 / 峰值 8.69**；
在 block 里误用精确 GELU 只偏离 **4.7e-4**。

`gelu_pytorch_tanh` 是 `F.gelu(x, approximate='tanh')`：

```
x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

### PatchMerger

```python
hidden_size = config.hidden_size * spatial_merge_size**2       # 4608
norm = nn.LayerNorm(config.hidden_size, eps=1e-6)              # 1152！
linear_fc1 = nn.Linear(4608, 4608)                             # 有 bias
act_fn = nn.GELU()                                             # 精确 erf 版
linear_fc2 = nn.Linear(4608, 5120)                             # 有 bias

x = norm(x).view(-1, 4608)
x = linear_fc2(act_fn(linear_fc1(x)))
```

**归一化在 shuffle 之前，按每个 patch 的 1152 维做。**
`use_postshuffle_norm = False`，而且 checkpoint 自己就把这件事定了：
`merger.norm.weight` 是 `[1152]` 不是 `[4608]`。
对 grouped 的 4608 维归一化（把增益平铺四遍让形状对上）会跑完，
merger 输入偏离 **9.99 / 峰值 6.81**。

**分组是纯 reshape**，把连续 4 个 patch 拼成一行 4608。
它之所以等于 2x2 池化，完全因为预处理按块序输出了 patch（见 patchify 一节）。
merger 里没有任何重排。另一种合理读法是按 stride 分组
（token `t` 取 patch `t, t+T, t+2T, t+3T`，channels-first reshape 会这么给），
实测在 fc1 探针列上偏离 **1.07 / 峰值 0.96**。

**激活是精确 GELU，不是 tanh 近似。** `nn.GELU()` 不带 `approximate` 参数。
27 个 block 用 tanh 版，merger 用 erf 版。两者最大差 **4.7e-4** 绝对值
（|x| < 12 区间），所以搞混是"小而遍布"的数值误差，不是布局灾难——
列在这里是因为"小而遍布"正是会被归因到量化上、然后查一个星期的那一类错误。

输出 `[N/4, 5120]`，5120 == 文本塔的 `hidden_size`。

## 拼接进文本序列

### 占位符

processor 产出的序列形如

```
... <|vision_start|> <|image_pad|> * (t*h*w/4) <|vision_end|> ...
```

`get_placeholder_mask` 硬性要求占位符个数 == merger 输出行数，不等就报错。
拼接就是"第 i 个占位符的 embedding 换成 merger 的第 i 行"，按序，无重排。
`get_image_features` 按 `grid_thw.prod(-1) // merge**2` 把 merger 输出切回每张图。

### 3-D 位置：`get_rope_index`

文本 run 里三个轴都等于同一个递增标量。视觉条目：

```python
T = start_pos                       （对图像 t=1，整块是同一个值）
H = start_pos + row                 row in [0, h/merge)
W = start_pos + col                 col in [0, w/merge)
按 (T, H, W) 的 ij meshgrid 展平
之后 start_pos += max(h, w) // merge
```

**关键是推进规则：视觉条目之后标量位置只前进 `max(llm_h, llm_w)`，
不是 token 数，也不是 Qwen2-VL 用的 `max(t, h, w)`。**
6x8 的 patch 网格是 12 个 image token，但只推进 4。
所以序列的最大位置会**低于**序列长度，参考实现把这个差记为负的 `rope_delta`
（捕获里的 19 token 例子：最大位置 10，长度 19，delta = -8）。
按 token 数推进会跑完，并把图像之后每一个 token 放到错误的相对距离上。

视频在 `get_rope_index` 里先按帧拆开（`repeat_interleave(video_grid_thw[:, 0])`
然后把 `t` 置 1），因为 Qwen3.5 用时间戳文本分隔帧：
`<t1> <vision_start> <frame1> <vision_end> <t2> ...`。

### 交错 mRoPE：`mrope_section = [11, 11, 10]`、`mrope_interleaved: true`

和为 32 = partial rope 的频率个数（`head_dim * 0.25 / 2 = 256 * 0.25 / 2`）。
`apply_interleaved_mrope` 的实现：

```python
freqs_t = freqs[0]                              # 先全填 T
for dim, offset in enumerate((1, 2), start=1):  # H, W
    idx = slice(offset, mrope_section[dim] * 3, 3)
    freqs_t[..., idx] = freqs[dim, ..., idx]
```

给它喂一个"值就是轴编号"的张量，返回的就是轴分配表，实测：

```
[0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1]
 ^ 32 个频率，轴 = i % 3；T 得 11 个，H 得 11 个，W 得 10 个
```

也就是说**交错模式下轴分配就是 `i % 3`**，`mrope_section` 只是把结果的计数写下来。
（一般情况下：H 认领 `i % 3 == 1 且 i < 3*section[1]`，
W 认领 `i % 3 == 2 且 i < 3*section[2]`，其余归 T。在 `[11, 11, 10]` 下正好化简为 `i % 3`。）

**陷阱：Qwen2-VL / Qwen2.5-VL 对同一个 config 字段用的是分块布局**——
`[0..11)` 给 T、`[11..22)` 给 H、`[22..32)` 给 W。
两种读法消费同样的三行位置、产出同样的形状。
更糟的是：**纯文本 prompt 上两者完全一致**，因为 `get_rope_index` 让文本 token 的
T = H = W。分歧只在上下文里进了图像之后才出现——bug 第一次显形的最坏地点。

## 捕获与测试

`tools/capture_qwen35_vision.py` 在 `ssh bw` 上跑参考实现自己的
`Qwen3_5VisionModel`（真实 BF16 权重，f32 算术，CPU），用 forward hook 记下每一级
的端点，写成裸 little-endian f32 + `manifest.json`。
`cross_check_against_transformers` 把这个文件里每一处转写都对着它转写自的库核对，
不一致就拒绝落盘（21 项，全 0 或 f32 精度量级）。q/k/v 切分和 rope 应用在
`Qwen3_5VisionAttention.forward` 内部、hook 不到，所以捕获记下两侧端点
（`attn.qkv` 的输出、`attn.proj` 的输入），并要求转写的中间过程复现它——
Rust 测试用同样的方式重建，答的是参考实现而不是我。

**归一化没有退化。** 文本侧那次教训（随机输入让激活落到 `rms_norm_eps` 以下五个
数量级，归一化退化成常数缩放，捕获对两种 normalize/gate 顺序都点头）在这里被显式检查：
捕获脚本打印每处 LayerNorm 输入的逐行方差，低于 `1000 * eps` 就拒绝落盘。实测

```
hidden_in    逐行方差 min 1.29e-1  median 1.41e-1  max 5.83e-1
b0.resid1    逐行方差 min 2.04e-1  median 2.74e-1  max 3.27e-1
last_hidden  逐行方差 min 2.19e+1  median 2.46e+3  max 1.59e+4
```

对 eps = 1e-6 而言是 5 到 10 个数量级之上，eps 完全不参与。
之所以如此，是因为输入不是白噪声：捕获用的是一张**多尺度正弦加噪声的程序化图像**
过**真实 image processor**。结构是必要的——纯平或纯噪声的输入下，
patch 顺序的任何排列都给同样的答案，位置嵌入也看不出被用到。

测试 `crates/model/tests/qwen35_vision.rs`，26 个，
gate 在 `TUILI_QWEN35_VISION_CAPTURE` 上，未设置时报 skip 而不是通过。
**每一个钉布局的检查都同时断言另一种合理读法不成立**（`assert_discriminates`
要求偏离超过张量自身峰值的 1%），否则通过只说明"某种算术跑了"。
