# 给 huicheng 的两件事（2026-08-18）

两件事互不相关，可以分开决定。第一件只需要加两个参数，第二件需要他自己拿真实
数据验证后再定。

---

## 一、397B 打开工具调用（低风险，加两个参数）

**现状**：`Qwen3.5-397B` 服务（:8080，已连续跑 17 天）没开工具调用。任何带
`tools` 字段的请求直接返回 400：

```
"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set
```

是 400 而不是静默降级 —— vLLM 看到 `tools` 就默认 `auto`，直接拒。

**为什么想要它**：Claude Code 之类的 agent 每个请求都带整套工具定义，所以现在
完全不可用。397B 的单轮编程能力实测很强（三道题 38/38，其中一道 30B-A3B 要花
10204 token 的，它 710 token / 16s 解掉），开了工具调用就能当 agent 后端。

**要加的参数**：

```
--enable-auto-tool-choice \
--tool-call-parser qwen3_xml
```

**风险评估**：

- 只在请求带 `tools` 时生效。而现在带 tools 一律 400 —— 这反过来证明现存调用方
  没有一个在用，打开后它们走的还是原路径，行为逐字节不变。
- 纯 CPU 侧输出解析，不碰 engine、不碰 KV cache、不占显存。
- 失败模式是快失败：parser 名字不对，vLLM 启动几秒内退出并打印合法列表。

**建议不动 `--max-num-seqs 1`**。它跟工具调用完全无关，改它才是有风险的改动
（并发从"排队"变成"一起挤"，262144 上下文下 KV 压力会触发 preemption 重算，
表现是偶发长尾延迟）。低并发业务改了只有坏处。

**上线前先做个零风险预检**（只读，不碰运行中的服务）：

```bash
/home/huicheng/huicheng/vllm397b/venv/bin/python3.12 -c \
  "from vllm.entrypoints.openai.tool_parsers import ToolParserManager as M; print(sorted(M.tool_parsers.keys()))"
```

- 有 `qwen3_xml` → 直接用
- 没有但有 `qwen3_coder` → 用它，是同一个 XML 解析器的旧名字
- 两个都没有 → **别用 `hermes` 顶替**。hermes 解的是
  `<tool_call>{"name":...,"arguments":{...}}</tool_call>` 这种 JSON，而 397B
  实际吐的是 `<function=...><parameter=...>` 嵌套 XML（我实测过）。名字不对会
  解不出来，或者解出错误的参数。

他那台是 vLLM 0.26.0，我这边 0.27.1（0.27.1 里确认注册了 `qwen3_xml`，映射到
`qwen3_engine_tool_parser`），0.26.0 的目录我没权限读，所以要他自己确认。

**完整命令**（在现有基础上加最后两行）：

```bash
/home/huicheng/huicheng/vllm397b/venv/bin/vllm serve \
  /home/huicheng/huicheng/models/Qwen3.5-397B-A17B-GPTQ-Int4 \
  --served-model-name Qwen3.5-397B \
  --host 192.168.2.8 --port 8080 \
  --tensor-parallel-size 1 --pipeline-parallel-size 3 \
  --quantization moe_wna16 \
  --max-model-len 262144 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

⚠️ **一个坑**：如果从 `ps` 输出里复制现有命令，`--default-chat-template-kwargs`
后面那段 JSON 的**单引号会丢**（`ps` 显示的是 shell 处理完的结果）。少了引号
shell 会把它拆成两个词，启动直接失败。上面这份是补好引号的。

**回滚命令**（起不来就原样起回去，去掉最后两行即可）—— 真正的风险不在参数，在
重启本身：~200 GB 跨三卡重载，7月31 那次能起来说明环境没问题，但值得挑个安静
窗口做。

---

## 二、GPU 3 的显存，和 27B 能不能顶掉 VL-7B（需要他验证）

### GPU 3 现在的账（总 97.9 GB，已用 94.6 GB）

| 服务 | 占用 | 实际是什么 |
|---|---|---|
| llama-server VL-7B (:16007) | 20.7 GB | `Qwen2.5-VL-7B-Instruct-BF16.gguf` —— **未量化**，BF16 权重 ~15.2 GB；投影器还是 **F32**（`mmproj-F32.gguf`）；`-c 32768`、4 slot 的 KV |
| mineru 内部 vLLM (:8000) | 30.1 GB | `--gpu-memory-utilization 0.30` × 97887 MiB = 29.4 GB。**是占总显存的固定比例，不是按需** |
| mineru 主进程 | 2.1 GB | |
| 我的 Qwen3.8-27B (:8300) | 41.7 GB | 0.42，当前 65536 上下文 |

四张卡此刻都是 0% 利用率 —— 这台机器是被"预留"占满的，不是被计算占满的。

### 观察：27B 也是原生视觉模型，读得比 VL-7B 准

Qwen3.8-27B 是原生 VL 模型（能处理图像和视频）。做了四张探针图，每张考一个维度，
判分点是「真值字符串必须逐字出现」，采样用 Qwen 推荐的
`temperature 0.7 / top_p 0.95 / top_k 20`（**不用 0** —— 这个模型贪心解码会退化
重复）。三个模型都关思考，对齐 397B 服务端的 `enable_thinking: false`：

| | A 正立 24px | B 倾斜 8° | C 小字 11px | D 柱状图读数 | 合计 | 平均耗时 |
|---|---|---|---|---|---|---|
| **Qwen3.8-27B** | 6/6 | 6/6 | 6/6 | 6/6 | **24/24** | **0.7s** |
| **Qwen3.5-397B** | 6/6 | 6/6 | 6/6 | 6/6 | **24/24** | 1.8s |
| Qwen2.5-VL-7B | 5/6 | **3/6** | 6/6 | 6/6 | 20/24 | 0.9s |

判分点包括千分号 `1.4 ‰` 和微符号 `3.1 µg/L`（U+00B5）—— 之前测 OvisOCR2、
PaddleOCR-VL、Qwen VL 时**三个模型都把 ‰ 读成 %，只有 mineru 没错**；27B 和 397B
都读对了。VL-7B 在 A 上把 `µg` 写成 `μg`（希腊 mu，码位不同），在 B 上漏了三项。

**VL-7B 的短板集中在倾斜图**（3/6），11px 小字它反而满分 —— 也就是说它的问题是
几何鲁棒性，不是纯 OCR 精度。

⚠️ **这套题太简单，区分不出 27B 和 397B 的上限** —— 两个都满分，只能说「都吃满了」，
不能说 27B 的视觉能力 ≥ 397B。而且四张图都是 PIL 画的干净合成图、DejaVu 字体、
每张单次采样。真实扫描件的噪点、JPEG 压缩、印章遮挡、手写批注、表格线断裂，
这套题一个都没考到。**所以下面那条「用你自己的 PDF 验」不是客套，是必要条件。**

（脚本在服务器上：`/tmp/mkvl.py` 生成图，`/tmp/vlrun.py` 跑对比。换成自己的图
只要改 `/tmp/vl_tests.json` 里的图片路径和 `facts` 判分点。**注意 /tmp 会被清理**，
要留就挪到 home 下。）

### 如果 27B 顶掉 VL-7B

停掉 VL-7B → GPU 3 只剩 mineru 的 32.2 GB，空出 **65.7 GB**：

- `gpu-memory-utilization` 可开到 ~0.63（61.7 GB），留 4 GB 余量
- KV 预算 ≈ 61.7 − 29(权重) − 3.5(开销+mamba) ≈ 29 GB
- 这个模型是混合架构（`full_attention_interval=4`，64 层里只有 16 层 full
  attention，其余走 GatedDeltaNet 线性注意力），KV 只有 **64 KiB/token**
  → 池子约 **47 万 token**
- **原生 262144 只要 16 GiB**，绰绰有余，还留得下并发

也就是一步同时解决上下文和 VL 需求。

### 三个真障碍

1. **绑定地址**：27B 在 `127.0.0.1:8300`，他的应用调 `192.168.2.8:16007`。要顶替
   得改绑，可以和提 max-model-len 一次重启做掉。
2. **模型名校验**：llama-server 不校验模型名（我传 `gpt-3.5-turbo` 它也认），
   **vLLM 严格校验**。他应用里写死的名字要改，或者加网关别名。
3. **故障域从隔离变共享** —— 这条最重要。现在他的抽取流水线和我的 Claude Code
   是两个进程，我随便重启不影响他；合并后我为 infero 实验重启一次，他的服务就断。
   而且 KV 池并发有限，我跑长提示词 agent 时他的图片请求要排队。他今天有 380 次
   prompt cache 驱逐，是真流量。

这个耦合代价可能比省 20 GB 更值得他在意。

### 建议的做法

**不要单方面换。** 拿这份对比让他用**自己真实的 PDF**（扫描件、印章、表格线）
跑一批 —— 合成图能证明 ‰ 这类字符 27B 更强，证明不了他的实际场景。

如果他认可，形态应该是**先并存验证**：应用加个开关，一部分请求打到 27B，比一段
时间的抽取质量，确认后再停 VL-7B。但现在 GPU 3 已经 94.6/97.9，**并存跑不动**
—— 真要试得先决定临时牺牲哪个（我的 27B 可以随时停，这个代价最小）。

### 备选：不动他任何东西也能拿到更多上下文

如果他不想动，我这边有两条路：

- **现有 KV 池就能提到 98304**（池子 111,550 token，我保守设了 65536）。只需重启
  我的 27B，5 分钟，谁都不用碰。代价是并发从 1.70x 降到 1.13x。
- **换 INT4 权重**（29 → 16 GB）能摸到 262144。代价是质量下降、可能丢 MTP 头
  （那个头让解码从 44 提到 89 tok/s），而且社区量化版的格式要先验。
