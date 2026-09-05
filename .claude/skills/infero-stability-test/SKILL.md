---
name: infero-stability-test
description: Use after any change to infero's server, attention backend, or scheduler code, before calling a running infero deployment "stable". Drives a real, already-running infero server over its real HTTP API with varied traffic (multi-turn conversations, concurrency, long/chunked prompts, tool calls, sequence retirement) across several repeated passes, to catch bugs that only surface under realistic traffic -- not single-shot benchmarks. Do not use this to debug a specific already-known crash; use it to get real confidence a deployment is stable, or to find what single-shot testing missed.
---

# infero 真实流量稳定性测试

## 背景

2026-09-05 那次session里，`FlashAttn2Ffi`这个attention后端被切成默认后端，全天用的都是`prefill_profile`这种**单次、全新prefill**的benchmark验证过，从未测过真实多轮对话/并发场景。结果用户拿`claude-qwen.sh`一测，第一次真实多轮流量就撞上了`flash_attn2 backend: KV slots ... not physically contiguous`——这个限制其实早就写在代码注释里了，只是从来没有真实流量踩到过。

这个skill就是为了不再犯"只测过benchmark shape、没测过真实使用模式"这个错——任何改了`crates/server`/attention后端/调度器代码之后，跑一遍这个，再说"稳定了"。

## 何时用
- 改了`crates/server/`（scheduler、engine、routes）之后
- 改了attention/GDN后端选择逻辑之后（`AttentionBackend`实现、优先级）之后
- 张量并行代码改动之后
- 任何"看起来能跑通一次curl"但还没有真正扛过多轮/并发流量验证的改动之后
- 不要用来调试一个已经定位到的具体崩溃——那种情况直接改代码、跑一次针对性repro就够了，这个skill是用来查"还有什么没测到"的

## 怎么用

1. **确认目标server在跑，而且能回应一次简单请求**（先别急着跑压测）：
   ```bash
   curl -s http://<host>:<port>/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model":"<model-name>","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
   ```
   如果这一步都不通，先别跑测试脚本——那测出来的只是"server没起来"，不是这个skill要查的东西。

2. **跑测试脚本**，指向真实server地址：
   ```bash
   python3 scripts/server_stress_test.py --base-url http://<host>:<port> --passes 5
   ```
   - `--passes`默认5，建议至少3——**只跑一次看不出flaky**，这个脚本本身的设计就是重复跑几次对比，一次PASS不代表稳定。
   - `--only`可以指定只跑某几类（逗号分隔），比如先复现一个具体场景：`--only multiturn_5,retire_and_reuse`
   - 脚本里`MODEL`常量硬编码成了`qwen38-27b-fp8`——测别的模型要改这一行，或者加个`--model`参数（当前版本没加，先手动改）。

3. **看结果**：脚本自己会打印每个类别每一轮的PASS/FAIL，最后给一个汇总表，区分`STABLE-PASS`/`FLAKY`/`STABLE-FAIL`三种状态。**FLAKY比STABLE-FAIL更值得警惕**——间歇性失败往往意味着一个跟并发/状态相关的真实bug（正是2026-09-05那次的情况：第一次多轮请求命中非连续slot之后，后续所有请求，包括原本简单的单轮请求，都会持续失败，直到重启server）。

## 测的是什么（`scripts/server_stress_test.py`里的类别）

- `single_turn_short`/`medium`/`long_chunked` —— 不同长度的单轮请求，`long_chunked`故意造一个够长的prompt去触发prefill的多chunk路径（真实生产是`CUTLASS_BATCH_TOKENS=8192`一chunk）
- `multiturn_2`/`5`/`10` —— 多轮对话，且最后一轮要求模型回忆第一轮提到的一个数字，不只是"没崩"，还真的检查上下文语义连贯
- `concurrent_at_capacity_2`/`over_capacity_4` —— 并发请求，一个卡在`--max-seqs`配置值上，一个故意超过，测调度器排队/backpressure行为
- `retire_and_reuse` —— 跑完一个对话、马上开一个新对话，测KV pool的slot释放/复用路径（正是"非连续slot"这类bug最容易藏身的地方）
- `tool_calling` —— 真实工具调用请求，测一条跟纯文本对话不一样的代码路径

## 已知的重要发现（2026-09-05）

- 第一次任何请求触发"非连续slot"错误之后，**server会进入一种卡死状态：之后哪怕是最简单的单轮全新请求也会持续用同一个错误报500**，直到重启server才恢复。这比"偶尔某个多轮请求失败"严重得多——这不是一个孤立请求的问题，是一次真实错误会让整个server对所有后续流量失效。跑这个skill如果测出`STABLE-FAIL`且错误信息在所有类别间雷同，先怀疑是不是server已经被前面某一轮测试自己弄卡死了，重启server再跑一遍单个简单请求确认。

## 扩展这个skill

新增测试场景直接改`scripts/server_stress_test.py`：在`CATEGORIES`列表里加一个`(name, fn)`元组，`fn`签名是`(base_url: str) -> tuple[bool, str]`（bool是通过与否，str是详情，会被截断打印到200字符）。别把新场景写成只能跑一次的——这个脚本的价值就在于能反复跑，加场景时也保持这个设计。
