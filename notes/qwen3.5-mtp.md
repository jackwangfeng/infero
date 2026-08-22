# Qwen3.5 MTP 头规格与投机解码调度设计

来源：**vLLM 0.27.1 的 `vllm/model_executor/models/qwen3_5_mtp.py`**，配
`/home/jeff/models/qwen38-27b-fp8` 的真实张量。

注意权威在哪：`transformers` 5.15 **完全不实现 MTP**——`modeling_qwen3_5.py` 里写着

```python
_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]
```

加载时直接把这些张量丢掉。所以 `transformers` 只能当 decoder layer 的参考（头里那一层就是
它的 `Qwen3_5DecoderLayer`），拼装顺序只有 vLLM 说得上话。
`qwen3_next_mtp.py` 的 `forward` 与它逐字相同，可以当第二个见证。

张量前缀是 **`mtp.`**，不是 `model.mtp.`。vLLM 加载时才把 `mtp.` 改写成 `model.`：

```python
if name.startswith("mtp."):
    name = name.replace("mtp.", "model.")
```

写这份东西的理由和架构那份一样：下面每一条弄反都不会崩，只会让 draft 的接受率悄悄掉到
接近 0，表现为"投机解码开了反而更慢"，看起来像调度问题。实测数字见最后。

## 张量清单（27B，`mtp.safetensors`，共 22 个）

```
mtp.fc.weight                                 BF16     [5120, 10240]
mtp.pre_fc_norm_embedding.weight              BF16     [5120]
mtp.pre_fc_norm_hidden.weight                 BF16     [5120]
mtp.norm.weight                               BF16     [5120]

mtp.layers.0.input_layernorm.weight           BF16     [5120]
mtp.layers.0.post_attention_layernorm.weight  BF16     [5120]
mtp.layers.0.self_attn.q_proj.weight          F8_E4M3  [12288, 5120]   scale [96, 40]
mtp.layers.0.self_attn.k_proj.weight          F8_E4M3  [ 1024, 5120]   scale [ 8, 40]
mtp.layers.0.self_attn.v_proj.weight          F8_E4M3  [ 1024, 5120]   scale [ 8, 40]
mtp.layers.0.self_attn.o_proj.weight          F8_E4M3  [ 5120, 6144]   scale [40, 48]
mtp.layers.0.self_attn.q_norm.weight          BF16     [256]
mtp.layers.0.self_attn.k_norm.weight          BF16     [256]
mtp.layers.0.mlp.gate_proj.weight             F8_E4M3  [17408, 5120]   scale [136, 40]
mtp.layers.0.mlp.up_proj.weight               F8_E4M3  [17408, 5120]   scale [136, 40]
mtp.layers.0.mlp.down_proj.weight             F8_E4M3  [ 5120, 17408]  scale [40, 136]
```

15 个权重 + 7 个 `weight_scale_inv`（128×128 块），合计 22。总参数 **424.7 M**，是 27B
主模型的 1.6%。`mtp.fc` 是 **BF16 而非 FP8**——不要用统一的 FP8 路径去加载它，
vLLM 里也为此专门开了个洞（`fc_quant = None if modelopt_fp4`）。

**不存在的张量**同样是规格的一部分：没有 `mtp.embed_tokens.*`，没有 `mtp.lm_head.*`，
没有 `mtp.layers.0.linear_attn.*`，没有 `conv1d` / `A_log` / `dt_bias` / `in_proj_*`。
头复用的两个：

```
model.language_model.embed_tokens.weight   BF16  [248320, 5120]
lm_head.weight                             BF16  [248320, 5120]
```

`tie_word_embeddings = false`，所以这两个是不同的张量，头用的是 **`lm_head`**（不是
embedding 的转置）。`mtp_use_dedicated_embeddings = False` 就是"没有 `mtp.embed_tokens`"
这件事的 config 表述；checkpoint 和 config 在这里是一致的，capture 会同时检查两边。
注意头有**自己的** `mtp.norm`，它和主模型的 `model.language_model.norm` 是两个张量。

## `Qwen3_5RMSNorm` 是 `(1 + weight)`，不是 `weight *`

这是本文最容易被继承下来的错误，独立成一节。

```python
class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        self.weight = nn.Parameter(torch.zeros(dim))     # 零初始化
    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())     # <-- 这里
```

vLLM 干脆把它 alias 成 Gemma 的那个：
`from vllm.model_executor.layers.layernorm import GemmaRMSNorm as Qwen3_5RMSNorm`，
其 docstring 第一行就是 *"x * (1 + w) instead of x * w"*。两边逐位一致。

为什么危险：Qwen3.5 里**只有一个** norm 用朴素的 `weight *` 形式，就是 GatedDeltaNet
的输出门 `Qwen3_5RMSNormGated`（`nn.Parameter(torch.ones(...))`，一初始化，实测权重
0.87±0.07）。`crates/model/src/qwen35.rs::rms_norm_rows` 实现的是那一个，而且是对的。
但其余全部——`input_layernorm`、`post_attention_layernorm`、`q_norm`、`k_norm`、
`model.norm`、以及 MTP 头的四个——都是 `Qwen3_5RMSNorm`，权重存的是**相对 1 的偏移**。

把偏移当增益用，在这个 checkpoint 上的后果：

```
mtp.pre_fc_norm_embedding   mean -0.4606  范围 [-0.7500, -0.1855]   全负
mtp.pre_fc_norm_hidden      mean -0.1572  范围 [-0.3750, +0.4551]
mtp.layers.0.input_layernorm mean +0.0361 范围 [-0.2256, +0.9531]
mtp.layers.0.self_attn.q_norm mean +0.7906 范围 [-0.5547, +1.9688]
mtp.norm                    mean +1.2520  范围 [-0.2246, +1.9297]
--- 对比 ---
layers.0.linear_attn.norm   mean +0.8686  范围 [+0.7852, +0.9297]   一初始化那个
```

`pre_fc_norm_embedding` 5120 个通道**全是负数**。朴素读法把每一维的符号翻过来，幅度
（|−0.46| vs 1−0.46=0.54）却差不多——这正是它不崩、只产生流畅胡话的原因。
从权重的分布就能判断类别：`Qwen3_5RMSNorm` 的权重在 0 附近，
`Qwen3_5RMSNormGated` 的在 1 附近。

> **与现有代码的分歧，需要另开一处修**：`tools/capture_qwen35_layers.py::rms_norm` 和
> `crates/model/src/qwen35.rs::rms_norm_rows` 用朴素形式处理了 **layer 3 的
> `q_norm` / `k_norm`**，而那两个是 `Qwen3_5RMSNorm`。capture 和 Rust 用同一个错误公式，
> 所以互相同意、测试全绿——就是那份 capture 的注释里警告的那种局部自洽。
> `layers.3.self_attn.q_norm.weight` 的 mean 是 +0.2304，朴素读法给的增益约 0.23，
> 正确读法约 1.23，差 5 倍多。本文件的 `qwen35_mtp.rs` 用了正确形式，
> `crates/model/tests/qwen35_mtp.rs::the_per_head_q_and_k_norms_use_the_unit_offset_form_before_rope`
> 会在头的那一层上把它钉住。主模型那侧不在本次可改文件范围内。

`eps` 加在均方上、开方之内，不是加在 rms 上。

## draft token 是怎么产生的

vLLM 的 `Qwen3_5MultiTokenPredictor.forward`，一字不改：

```python
inputs_embeds  = self.pre_fc_norm_embedding(inputs_embeds)
hidden_states  = self.pre_fc_norm_hidden(hidden_states)
hidden_states  = torch.cat([inputs_embeds, hidden_states], dim=-1)
hidden_states  = self.fc(hidden_states)
...
hidden_states, residual = mtp_layer(positions=positions,
                                    hidden_states=hidden_states, residual=None)
hidden_states, _ = self.norm(hidden_states, residual)
```

拆成规格：

| 项 | 结论 |
|---|---|
| concat 顺序 | **embedding 在前，hidden 在后**。`fc` 的列 `0..5120` 乘 embedding，列 `5120..10240` 乘 hidden state |
| `pre_fc_norm_embedding` | 作用在 **token embedding** 上 |
| `pre_fc_norm_hidden` | 作用在 **主模型的 hidden state** 上 |
| 哪个 hidden state | 主模型 `forward` 的返回值，即 **过完 `model.language_model.norm` 之后**的 |
| 哪个 embedding | 主模型的 `model.language_model.embed_tokens` |
| 哪个 lm_head | 主模型的 `lm_head`（顶层的那个，不带 `model.` 前缀） |
| 头里那层 | **full attention**（带输出门 + partial RoPE），不是 linear attention |
| 残差 | 进 `mtp_layer` 时 `residual=None`，所以层内是标准 pre-norm 双残差；最后 `self.norm(h, residual)` 是 `RMSNorm(h + residual)` |

**槽位对齐**：槽 `i` 拿到的是 *(token `t_i` 的 hidden state, token `t_{i+1}` 的 embedding)*，
预测 `t_{i+2}`。vLLM 就是这么拼输入的：

```python
# Shift the input ids by one token.
# E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
self.input_ids[: num_tokens - 1] = target_token_ids[1:]
self.input_ids[token_indices_to_sample] = next_token_ids   # 刚采出来的那个填最后一格
self._set_positions(num_tokens, target_positions)          # positions 原样不动
self.hidden_states[:num_tokens] = target_hidden_states
```

**位置约定**：`positions` 用的是 **hidden state 那个 token 的位置 `i`**，不是被 embed 的
`t_{i+1}` 的位置 `i+1`；后续每步 draft `+1`（`eagle_step_update_slot_mapping_and_metadata`
里 `new_position = position + 1`）。所以 drafter 的 KV cache 在位置 `p` 上存的是
`(h_p, emb(t_{p+1}))` 这一对，比 target 的序列整体落后一格。

这一条**没法用单次前向的数值钉住**，而且不该假装能：把所有位置整体平移一个常数，
自洽的一次 attention 输出必须逐位不变（这正是 `notes/qwen3.5-architecture.md` 里那条
不变量）。capture 实测整体 +1 的输出相对 L2 是 **4.51e-07**，即不变量成立。
约定只在**跨 draft 步**时才可观测——它决定新 token 与 drafter 缓存历史之间的相对偏移——
所以这条靠读 vLLM 源码定，不靠数值。

**多步 draft**：`mtp_num_hidden_layers = 1` 不限制 `num_speculative_tokens`。
循环用 `spec_step_idx % num_mtp_layers`，永远是同一层，把自己**过完 `mtp.norm` 的输出**
再喂回 `pre_fc_norm_hidden`，配上刚 draft 出来的 token 的 embedding。也就是说
`hidden_states` 这个入口在第 1 步是 target 的最终 hidden，第 2 步及以后是头自己的输出。

### 一句话伪码

```
# 输入：h[0..T)  主模型最终 hidden（过 model.norm 之后）
#      ids[0..T) 主模型这一批的 token
#      next      刚采样出的 token
shifted = concat(ids[1:], [next])
e = RMSNorm_offset(embed_tokens[shifted], pre_fc_norm_embedding)   # (1+w)
n = RMSNorm_offset(h,                     pre_fc_norm_hidden)      # (1+w)
x = fc @ concat([e, n], dim=-1)                                    # [T, 5120]
x = full_attention_layer(x, positions=0..T)                        # mtp.layers.0
o = RMSNorm_offset(x, mtp.norm)
draft = argmax(lm_head @ o)
```

## 头里那一层是 full attention

三份互不相干的证据：

1. checkpoint：`mtp.layers.0.` 下有 `self_attn.{q,k,v,o}_proj` 和
   `self_attn.{q,k}_norm`，**没有** `linear_attn` / `conv1d` / `A_log` / `dt_bias` /
   `in_proj_*`。
2. 形状：`q_proj` 是 `[12288, 5120]`，12288 = 24 × 512 = 24 头 ×（256 q + 256 门），
   正是带输出门的那种；`o_proj` 是 `[5120, 6144]`，输入 d_attn 输出 d_model。
3. vLLM 构造时写死了：`Qwen3_5DecoderLayer(vllm_config, layer_type="full_attention", ...)`。

所以架构那份文档里 full attention 层的两条额外机制**都适用于这一层**：

- **输出门与 q 在 `q_proj` 输出里逐头交错**：每头 512 里前 256 是 q、后 256 是门；
  门在 `o_proj` **之前**乘上，用 **`sigmoid`**（不是 silu，`output_gate_type = "swish"`
  没有被任何实现读取）。
- **partial RoPE 只旋转每头前 64 维**，频率表按 64 归一化（不是 256 维表的前 64 维），
  `rotate_half` 配对 `(i, i+32)`，`[64, 256)` 原样透传。

MLP 是普通 SwiGLU，`intermediate_size = 17408`，`hidden_act = silu`，silu 落在
`gate_proj` 那一支。

**这一条对调度最重要的推论**：draft 只跑 1 个 full attention 层，
**完全不碰 GatedDeltaNet 的递推状态**。draft 阶段没有状态回滚问题；状态回滚只发生在
verify 阶段。

## capture 实测的数值证据

`tools/capture_qwen35_mtp.py` 在真实 checkpoint 上跑完全部 64 层 + 头，32 个真实
token。cross-check：

```
(1+w) 形式 vs transformers Qwen3_5RMSNorm       Δ = 0.00e+00
(1+w) 形式 vs vLLM GemmaRMSNorm                Δ = 0.00e+00
朴素 `w *` 形式与之相差                          142.01% of peak
fused add-norm == norm(x + residual)           Δ = 0.00e+00
vLLM 的 fc 输入 AST 归约（两个见证都是）
  cat[pre_fc_norm_embedding(EMBEDDING)|pre_fc_norm_hidden(TARGET_HIDDEN)]@-1
```

行为检查——头的 argmax 与主模型自己在下一位置的 argmax 的一致率，
即单 token draft 的贪心接受率：

```
变体                          接受率      输出相对 L2
参考实现的拼法                 71.0%       —
concat 两半互换                 0.0%       1.47e+00
两个 pre_fc norm 互换          45.2%       6.63e-01
朴素 `w *` 而非 (1+w)           6.5%       1.20e+00
用 model.norm 之前的 hidden    71.0%       2.05e-01
positions 整体 +1              71.0%       4.51e-07   <- 不变量，不是变体
```

三个真错误都明显更差，capture 以此为硬门槛，不达标就拒绝写盘。
"model.norm 之前的 hidden"这一条**行为上分不开**（`pre_fc_norm_hidden` 反正要重新
归一化，只有 `model.norm` 的逐通道增益能活下来），但它确实是不同的计算（ΔL2 0.205），
所以改用数值钉：capture 把两个 hidden state 都落盘，Rust 测试断言
`hidden_normed` 匹配过完 norm 的那个、且不匹配另一个。

eps 占 RMS 分母的比例（逐行取最坏）：

```
进 model.norm 的 hidden               0.001%
进 pre_fc_norm_embedding 的 embedding 0.771%
进 pre_fc_norm_hidden 的 hidden       0.000%
```

embedding 那一行天然小——这个 checkpoint 的 embedding 行 RMS 约 0.014，比最终 hidden
小 60 倍，`mean(x²)` 就是 2e-4，跟文本真不真实无关。0.77% 不算被 eps 支配。

另一条更要紧的：同样的真实文本，只跑前 4 层前缀时一致率是 **14.3%**，跑满 64 层是
**71.0%**。**截断前缀不是 oracle**——头吃的是最终 hidden，没有捷径。`--prefix-layers`
只用于冒烟，manifest 会记 `prefix_truncated`，Rust 测试会直接拒绝这样的 capture。

## tuili 的调度设计

### 尺寸与代价

一次 draft 步 = 425 M 参数的 MTP 层 + 1.271 G 参数的 `lm_head`（BF16，2.54 GB）。
`lm_head` 是大头：词表 248320 使它比整个 MTP 层贵 6 倍。按权重字节算，
27B FP8 主模型一次前向约 27 GB，所以

```
一次 draft 步 ≈ (0.37 GB FP8 层 + 0.10 GB BF16 fc + 2.54 GB lm_head) / 27 GB ≈ 11%
k = 2 → 两次 draft ≈ 22% 的一次 target 步
verify 一次 = 一次 target 步（k+1 个 token 仍然只读一遍权重，decode 是带宽瓶颈）
```

`k = 2`、平均接受长度 2.0：`2.0 / 1.22 = 1.64×`。vLLM 实测 44 → 89 tok/s 是 2.02×，
比这个估算好，说明他们的 draft 步比按权重字节算的更便宜（`lm_head` 那一步大概做了
top-k 截断或者复用了 target 步已经在片上的东西）。**结论：先按 `k = 2` 做，
把 `lm_head` 在 draft 路径上的代价当作首要优化目标，而不是去优化那 425 M 的层。**

接受率的期望值：单 token 贪心一致率实测 71%。若各步独立，`k=2` 的平均接受长度是
`1 + 0.71 + 0.71² = 2.21`；vLLM 报的是 2.0，略低是应当的——后面的步建立在 draft 出来的
上下文和头自己回灌的 hidden 上，条件分布更差。**验收线：平均接受长度 ≥ 1.9。低于 1.6
先怀疑拼装顺序，不要先怀疑调度。**

### draft 生成

drafter 需要自己的一份 KV cache：**1 个 full attention 层**，
`4 kv heads × 256 × 2(K,V)` = 每 token 每序列 2048 个元素。相对主模型的 16 个
full attention 层，是 1/16 的增量，可以忽略。drafter 的 KV 与 target 的 KV
在同一个 pool 里另开一组即可。

每步：

1. 取 target 这一步的最终 hidden `h`（**过完 `model.language_model.norm`**）和刚采出的
   token；
2. `shifted_ids` = target 这批 id 左移一格、最后一格填新采的 token；
3. 跑头，位置 = target 的位置；取 `argmax`（贪心）或采样（随机）；
4. 第 2..k 步：把头自己过完 `mtp.norm` 的输出当 `hidden_states` 回灌，
   `input_ids` 用上一步 draft 出来的 token，位置 `+1`。

`spec_step_idx % mtp_num_hidden_layers` 永远是 0，只有一层，所以 k 步共享同一份权重，
权重可以常驻不动。

### 并行验证

一次 target 前向，喂 `k+1` 个 token：上一步已确定的那个 + k 个 draft。因果 mask 是
普通下三角（**线性链，不是树**——vLLM 这条路径不做 tree attention，别一开始就上树）。
出来 `k+1` 组 logits：位置 `j` 的 logits 是"给定前 `j` 个 token"对第 `j+1` 个的预测，
所以 `logits[j]` 用来判 `draft[j]`，`logits[k]` 是 bonus。

### 接受规则

贪心（`rejection_greedy_sample_kernel`）：

```
从左到右扫；draft[j] == argmax(target_logits[j]) 就收下
第一次不等：吐出 argmax(target_logits[j])，停
全对：再吐出 bonus token argmax(target_logits[k])
```

一步吐出 `1 .. k+1` 个 token，**永远至少 1 个**（所以不会活锁），且吐出的序列
**逐 token 等于不投机的贪心解码**。这条精确性值得守住：它意味着投机可以随时开关而
不改变输出，于是 draft 头的退化只表现为吞吐变化、永远不表现为质量变化。

注意"第一次不等之后即使 `draft[j+1]` 恰好等于 `argmax(logits[j+1])` 也要丢掉"——
`logits[j+1]` 是在一个 target 没有选的 token 的条件下算出来的，收下它等于从错误的
条件分布里采样。

随机（`rejection_random_sample_kernel`）：`u ~ U(0,1)`，
`accept ⟺ p_target(draft_j) / p_draft(draft_j) ≥ u`；拒绝时从残差分布
`normalize(max(0, p_target − p_draft))` 采一个"recovered" token 吐出去；全接受再加
bonus。`p_draft == 0` 直接拒（否则比值是 +inf，会收下 draft 模型认为不可能的 token）。

主机侧参考：`qwen35_mtp::accept_greedy` / `accept_stochastic`，
`accepted` 字段给的是**被接受的 draft 个数**；要保留的 KV / 状态条目数是 `accepted + 1`
（接受的 draft 加上 target 自己产出的那个）。

### KV cache 回滚

append-only pool 上是最简单的一种：verify 写了 `k+1` 个 token 的 K/V，把序列长度退回
`accepted + 1`，多写的那几格下一步直接覆盖。只有两件事要做对：

- **drafter 自己的 KV 也要退**，而且退的是它自己的坐标系（比 target 落后一格）。
  drafter 在 draft 步里已经把 `k` 个投机 token 的 K/V 写进去了。
- 分配 block 时给每个序列多留 `ceil((k+1)/block_size)` 个 block，
  否则 verify 会跨块越界。vLLM 走的是同一条路（`num_speculative_blocks`）。

### GatedDeltaNet 递推状态的回滚

这是本设计唯一的难点。48 个线性层的状态 `S[h]` 是 `[128, 128]` f32、48 头，
**3 MiB / 序列 / 层，147 MiB / 序列**，而且是**就地更新**。verify 那一遍在 logits 回来
之前，已经把 `k+1` 个候选 token 全部揉进了 `S`——和 KV cache 不一样，这里没有"多余的格子"
可以截断，被拒绝的 token 和被接受的混在同一批数字里。

三条路，按代价排：

**(a) 快照-恢复（不要）。** 步前把 `S` 拷一份，拒绝时拷回来。147 MiB / 序列，
每步两次拷贝共 294 MiB / 序列。batch 32 时是 9.4 GiB / 步，而这一步本来只需要读约 26 GB
权重——凭空加三成以上的带宽。**否掉。**

**(b) vLLM 的做法：`k+1` 份状态槽。** `MambaSpec.num_speculative_blocks =
num_speculative_tokens`，每序列常驻 `1 + k` 份状态块（`mamba_cache_mode=align` 是
`2 + k`）。kernel 每处理完一个候选 token 就把当前状态写进 `ssm_state_indices[n, i_t]`
指的那一槽：

```python
# fused_sigmoid_gating.py，融合 GDN update kernel
if INPLACE_FINAL_STATE:
    final_state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t)
    if final_state_idx > 0:
        tl.store(p_ht, b_h, mask=mask_h)          # 每个 t 落一份
...
# 下一步读的时候按接受数选槽
if IS_SPEC_DECODING:
    i_t = tl.load(num_accepted_tokens + i_n) - 1
state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t)
```

**没有回滚，也没有拷贝**——扇出写，下一步靠索引挑。代价是显存：`k=2` 时
`3 × 147 MiB = 441 MiB / 序列`的递推状态。batch 32 就是 13.8 GiB，只为了投机。
GPU 本来就快满了，这个不能接受。

**(c) 推荐：journal + replay。** 递推的每 token 更新恰好是

```
S ← S * exp(g_t)            # 逐头标量
S ← S + k_t ⊗ delta_t       # 逐头 rank-1
```

而 `k_t`（l2 归一化后的）、`delta_t = (v_t − k_tᵀS) * beta_t`、`g_t`
**前向过程本来就都算出来了**。所以：

1. verify 时递推从**持久的 `S`** 出发，但结果写在片上/scratch 里，**不回写 `S`**；
2. 逐 token 把 `(k_t, delta_t, g_t)` 记进 journal；
3. logits 回来知道 `n = accepted + 1` 之后，用一个极小的 commit kernel 把 journal 的
   前 `n` 项按顺序作用到持久的 `S` 上。

代价：持久状态 **147 MiB / 序列不变**，journal 是
`48 heads × (128 + 128) × 4 B + 48 × 4 B ≈ 48.2 KiB / token / 层`，
48 层 × (k+1) = **`k=2` 时 6.9 MiB / 序列**。比 (b) 省 64 倍，比 (a) 完全没有拷贝带宽。
replay 的算术量是 `n` 次逐头 rank-1 更新，和前向已经做过的那一份一样多，可忽略。

而且是**精确的**，不是近似——journal 存的是更新项本身，不需要重建任何东西。
主机侧参考：`qwen35_mtp::DeltaJournalEntry` / `replay_accepted`，
`crates/model/tests/qwen35_mtp.rs::replaying_the_accepted_prefix_restores_the_state_exactly`
对 `accepted = 0..=T` 每个值都比对 `qwen35::gated_delta_rule` 只跑接受前缀的结果，
并且断言**不回滚的那个状态必须不同**（否则这条测试什么都没证明）。

**为什么不能求逆倒着走**：给定 `S_new`、`k`、`v`、`beta`、`g`，
`m = k^T S_decayed` 满足 `m (1 − ‖k‖²β) = k^T S_new − ‖k‖²βv`。`k` 是 l2 归一化的，
`‖k‖² ≈ 1`，于是要除以 `1 − β`。而 `β = sigmoid(b)` 可以任意接近 1，
一个 `β = 0.999` 的 token 就吃掉三位有效数字。**数值上不可行，别试。**
（测试里的 `beta` 特意取在 0.999 附近，就是为了证明 journal 路径不在乎这一点。）

### conv 状态的回滚

`conv1d` 的窗口是 `[10240, 3]` / 序列 / 层。这个便宜，照 vLLM 的办法把窗口加宽即可：

```python
conv_state_shape = (conv_dim, conv_kernel_size - 1 + num_spec)
```

`k=2` 时 `[10240, 5]` f32 = 200 KiB / 层，48 层 9.6 MiB / 序列。回滚就是把读指针
往回挪——窗口里保留的就是候选 token 的 `qkv`，接受几个就从第几个往前数 3 个。
不需要 journal。

### CUDA graph 与地址稳定

递推状态就地更新，所以 graph 捕获/重放时状态缓冲区**地址必须稳定**
（`notes/qwen3.5-architecture.md` 已记）。方案 (c) 在这一点上比 (b) 好：持久 `S` 的
地址永远不变，journal 是固定大小的 scratch，commit kernel 的形状只依赖 `k` 和
batch，`n` 走的是设备端的 `num_accepted_tokens`。也就是说
**verify 步和 commit 步都可以被 graph 捕获，不需要按接受数换图**。
方案 (b) 需要每步换 `ssm_state_indices`，虽然也能做（索引在设备内存里），
但多一份 `k+1` 倍的状态池就得跟着换分配器。

### 落地顺序

1. 主机侧参考 + capture 测试（本次做完了）。
2. drafter 的 KV 组、`shifted_ids` 构造、位置约定，`k=1` 先跑通；`k=1` 不需要任何
   状态回滚（verify 只有 2 个 token，其中至少 1 个必然被接受，但第 2 个可能被拒，
   所以其实需要 journal——**`k=1` 就要把 journal 做上**，别指望有免费阶段）。
3. 接受规则 + KV 回退，先只做贪心。
4. journal + commit kernel。
5. 再上 `k=2`，量平均接受长度，对着 1.9 的验收线看。
6. `lm_head` 在 draft 路径上的裁剪（2.54 GB / draft 步是最大的单项开销）。

不要做的：tree attention（vLLM 这条路径也不做，线性链已经拿到 2.0 的接受长度）；
`mamba_cache_mode="all"` 的前缀缓存（vLLM 自己在 `Qwen3_5MTP.__init__` 里直接
`raise NotImplementedError`，要求 `--mamba-cache-mode=align`）。

## 复现

```
python3 tools/capture_qwen35_mtp.py /home/jeff/models/qwen38-27b-fp8 <out> --tokens 32
#   全部 64 层，流式一层一层加载，峰值内存是一层而不是六十四层
#   加 --dump-layer-weights 会多写 1.7 GB，用来把 mtp_head 端到端跑起来
TUILI_QWEN35_MTP_CAPTURE=<out> cargo test -p tuili-model --test qwen35_mtp
```
