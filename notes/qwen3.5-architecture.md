# Qwen3.5 (qwen3_5) 架构规格

来源：`transformers` 5.15 的 `models/qwen3_5/modeling_qwen3_5.py`，配 `/home/jeff/models/qwen38-27b-fp8`
的真实张量形状。**代码是权威，config 字段不是**——见下面 `output_gate_type` 那条。

写这份东西的理由：这些细节每一条弄反都不会崩，只会产出流畅的胡话，和 bf16-当-f16 那次
一模一样。重新从源码推导一遍要半天。

## 27B 的尺寸

```
hidden_size          5120        num_hidden_layers    64
num_attention_heads  24          head_dim             256   -> d_attn = 6144 ≠ d_model
num_key_value_heads  4                                      -> d_kv   = 1024
intermediate_size    17408       vocab_size           248320
rope_theta           1e7*        max_position         262144
rms_norm_eps         1e-6        partial_rotary_factor 0.25
linear_num_key_heads   16   linear_key_head_dim   128  -> key_dim   = 2048
linear_num_value_heads 48   linear_value_head_dim 128  -> value_dim = 6144
linear_conv_kernel_dim 4    mamba_ssm_dtype       float32
attn_output_gate     True   mtp_num_hidden_layers 1
```

\* `rope_theta` 和 `partial_rotary_factor` 都在 `text_config.rope_parameters`
里，**不在 `text_config` 顶层**：

```json
"rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                    "partial_rotary_factor": 0.25,
                    "mrope_interleaved": true, "mrope_section": [11, 11, 10]}
```

从顶层读不会报错，只会静默换成默认频率基底——长上下文下表现为远距离检索变差，
而近距离全对，很难归因。`partial_rotary_factor` 在两处都有，`rope_theta` 只在里面。

`layer_types` 是显式的 64 元数组，不要用 `full_attention_interval` 反推：

```
[linear, linear, linear, full, linear, linear, linear, full, ...]
```

即 **层 3,7,11,…,63 是 full attention（16 层），其余 48 层是 GatedDeltaNet**。层不同质，
层循环不能再假设结构相同。

## full attention 层（3,7,…）——不是普通的 Qwen3 注意力

两处额外机制：

**1. 输出门，而且和 q 在 q_proj 输出里逐头交错。** `q_proj` 是 `[12288, 5120]`，
12288 = 24 × 512。源码：

```python
query_states, gate = torch.chunk(
    self.q_proj(h).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
```

先 view 成 `[..., 24, 512]` 再在最后一维 chunk。所以布局是 **每头 512 里前 256 是 q、
后 256 是门**，不是 `[全部 q | 全部门]`。这是最容易弄反的一处：两种解释都能跑完。

门的用法在 o_proj **之前**，且是 `sigmoid` 不是 silu：

```python
attn_output = attn_output * torch.sigmoid(gate)
attn_output = self.o_proj(attn_output)
```

config 里的 `output_gate_type = "swish"` 在这份实现里没被读取。代码用 sigmoid。

**2. partial RoPE：只旋转每头前 64 维。** `dim = int(head_dim * 0.25) = 64`，且频率表
按 `dim` 归一化：

```python
inv_freq = 1.0 / (base ** (torch.arange(0, 64, 2).float() / 64))   # 32 个频率
```

注意除的是 64 不是 256——这**不是**「256 维频率表的前 64 维」，是压缩到 64 维的另一张表。
旋转采用 `rotate_half` 的非交错布局（配对是 `(i, i+32)`），维度 `[64, 256)` 原样透传。

q_norm/k_norm 是 `[256]` 的 per-head RMSNorm，在 RoPE 之前——这条 infero 已经有了。

捕获时顺手落下一条不依赖实现的不变量：**把所有位置整体平移一个常数，注意力输出必须逐位不变**
（`full` 与 `full_far` 的 `attn_out_pre_gate` 完全相同，位置 0..11 对 130000..130011）。
RoPE 只编码相对位置，所以这条成立；不成立就说明频率表或配对写错了。它比对着数值比大小
好用，因为不需要参考实现在手。

`o_proj` 是 `[5120, 6144]`：输入 d_attn，输出 d_model。

## GatedDeltaNet 层（0,1,2,4,…）

张量（层 0 实测）：

```
in_proj_qkv  F8  [10240, 5120]   10240 = 2048(q) + 2048(k) + 6144(v)
in_proj_z    F8  [ 6144, 5120]
in_proj_a   BF16 [   48, 5120]   每值头一个标量
in_proj_b   BF16 [   48, 5120]
conv1d.weight BF16 [10240, 1, 4] depthwise，groups=10240，kernel 4，**无 bias**
A_log       BF16 [48]     dt_bias  BF16 [48]
norm.weight BF16 [128]    gated RMSNorm，宽度是 value_head_dim
out_proj     F8  [5120, 6144]
```

前向：

```
qkv  = in_proj_qkv(x)                     [T, 10240]
z    = in_proj_z(x)                       [T, 6144]  -> [T, 48, 128]
a    = in_proj_a(x)   b = in_proj_b(x)    [T, 48]

qkv  = silu(depthwise_causal_conv1d(qkv, w, kernel=4))    # 无 bias
q, k, v = split(qkv, [2048, 2048, 6144])
q -> [T, 16, 128]   k -> [T, 16, 128]   v -> [T, 48, 128]

beta = sigmoid(b)                                  [T, 48]
g    = -exp(A_log) * softplus(a + dt_bias)         [T, 48]，f32，恒为负
q, k = repeat_interleave(q, 3), repeat_interleave(k, 3)   # 16 -> 48 头
```

递推（`torch_recurrent_gated_delta_rule`，状态 `S[h]` 形状 `[k_dim=128, v_dim=128]`，f32）：

```
q = l2norm(q, eps=1e-6);  k = l2norm(k, eps=1e-6)     # 在 kernel 内做
q = q * (1/sqrt(128))                                  # 只缩放 q

每个 t：
  S      *= exp(g_t)                       # 标量衰减，逐头
  kv_mem  = kᵀ S                           [v_dim]
  delta   = (v_t - kv_mem) * beta_t        [v_dim]
  S      += k_t ⊗ delta                    # 外积
  o_t     = qᵀ S                           [v_dim]
```

输出：

```
o = RMSNorm(o, norm.weight, eps=1e-6) * silu(z)    # 先 norm 后门；norm 在 f32
out = out_proj(o.reshape(T, 6144))
```

`RMSNormGated` 的顺序要注意：`weight * normalized` 之后才乘 `silu(gate)`，且 gate 转 f32。

### 状态与缓存

- 递推状态：48 头 × 128 × 128 × f32 = 3 MiB / 序列 / 层，48 个线性层 → **147 MiB / 序列**。
  比 full attention 的 KV 便宜得多，但它不随 token 增长，是固定开销。
- conv 状态：`[10240, 3]` / 序列 / 层。
- `mamba_ssm_dtype = float32`：状态必须 f32，不能省成 f16。
- 状态是就地更新的，这和 infero 现在的 KV pool（append-only）语义不同；CUDA graph
  捕获/重放时状态缓冲区必须地址稳定。

## MTP

`mtp_num_hidden_layers = 1`，`mtp_use_dedicated_embeddings = False`（复用主 embedding）。

## 视觉

`mrope_section = [11, 11, 10]`，和为 32 = partial rope 的频率个数。纯文本走 1-D position
的简单路径；视觉要 3-D mRoPE。
