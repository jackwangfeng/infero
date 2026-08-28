---
name: cuda-memory-corruption
description: Use when a CUDA kernel in this repo causes CUDA_ERROR_ILLEGAL_ADDRESS, or intermittent/hard-to-repro GPU corruption that surfaces in an unrelated *later* kernel (a different kernel name in the error, sometimes a different layer/step). Applies to infero's hand-written .cu kernels under crates/kernels/src/cu/. Do not use for host-side Rust panics or plain OOM errors.
---

# CUDA 显存腐蚀排查 (illegal address / 下游kernel报错)

## 何时用
- 报错里的 kernel 名字和你刚改的 kernel 不是同一个(比如你在调 `attn_decode_mma_f32`,但崩溃信息说 `gather_rows_f16`/`rms_norm`/`add_assign` 出错)。
- 单次孤立的单元测试(如 `tests/ops.rs` 里单独调一次 kernel)测不出问题,但完整 server 跑真实请求会崩。
- 崩溃不是每次都发生在同一行/同一个 kernel,报错信息在不同次运行间跳来跳去。

这些特征合起来基本就是**堆/共享内存腐蚀**:某个 kernel 写坏了内存,但当场没有触发硬件异常,等**另一个后来的 kernel**读到这块脏数据时才炸。真正的坏人几乎从不是报错信息里那个 kernel。

## 核心教训:先怀疑自己的调试代码,再怀疑原始 bug

用 `#if DEBUG_SKIP_X 1` 这类开关做二分法时,**开关本身极容易引入新 bug,把你带偏**。本仓库真实踩过两个坑:

1. 强制把 `shared_mem_bytes` 砍到一个很小的值(比如 1024)来测试"是不是 shared memory 大小的问题"——如果这时候 kernel 体内其实还在写正常大小的 shared buffer,这个调试开关自己就造成了越界写。
2. 强制改 `block_dim`(比如 128→192)来测试线程数——这可能打破 kernel 里 `nwarps` 相关的 shared memory 别名复用假设(比如 combine 阶段把 `sacc`/`sml` 复用 K/V 的 shared buffer,这个复用的大小是按*原本*的 warp 数算的)。

**每加一个调试开关,先问:这个开关会不会让 kernel body 里的下标/大小计算和"这个开关暗中改变的东西"（shared mem 大小、block_dim、grid_dim...)变得不一致?** 如果会,这个开关本身就是新 bug 的来源,得出的"仍然崩溃"结论是假的,必须先修掉调试开关自身引入的问题,再继续二分。

## 排查步骤

1. **造一个尽量小、能稳定复现、不需要"热身"的 repro。**
   一开始怀疑"第二个请求才崩"之类的条件,往往只是因为大 repro 跑得太慢/太大,盖住了其实第一次请求就能崩的事实。优先用能几秒内跑完的最小输入(小 prompt、小 ctx)反复验证,别死磕最初发现问题时的那个大 repro。

2. **用 `#if` 开关做增量二分,一次只加回一小块真实逻辑**,每步都跑一遍最小 repro(能跑多次更好,尤其是要验证"×次请求后崩"这种条件是否还成立):
   - 整个 kernel body 换成一行 trivial 写(比如全写 0),确认 launch 本身(grid/block/shared mem 大小/参数列表)没问题。
   - 只加回"读输入、写 shared memory"这段,combine/reduce 阶段仍是 trivial 写。
   - 只加回 combine/reduce 阶段,主循环仍是 trivial(全零累加器)。
   - 两段都恢复真实逻辑。
   哪一步开始复现崩溃,bug 就在那一步新加回的代码里。**每次改开关都要重新 `rm -rf ~/.cache/infero/ptx`**(见下面 PTX 缓存的坑)。

3. **`compute-sanitizer --tool memcheck` 排查越界地址。**
   干净、确定性,但很慢。用第 1 步的最小 repro 而不是原始大 repro,否则可能几十分钟都跑不完。如果 memcheck 在你怀疑的 kernel 内部**什么都没报**,但下游 kernel 确实读到了坏数据——说明**不是地址越界**,写入落在了"合法但逻辑上错误"的位置。这时候不要继续死磕 memcheck,直接跳到第 5 步。

4. **`compute-sanitizer --tool racecheck` 排查 shared memory 竞态。**
   比 memcheck 慢得多(实测一个大请求几十分钟 CPU 时间都跑不完,一定要用最小 repro)。如果 racecheck 也是 0 hazard,说明也不是 block 内的读写序问题。

5. **memcheck 和 racecheck 都测不出来时,怀疑数值传播(NaN/Inf),不是地址。**
   这是最容易被忽略的一类 bug:某段代码通过"乘以 softmax 权重 0"来 mask 掉不该用的数据,但如果*另一个操作数*(比如从未初始化/未清零的 shared memory 读出来的 V 值)恰好是 NaN 或 Inf,那么 `0 * NaN = NaN`、`0 * Inf = NaN`——mask 失败,NaN 悄悄污染了整个累加器。这类 bug 的指纹:
   - memcheck/racecheck 都干净;
   - 崩溃发生在下游好几个 kernel/好几层之后(NaN 需要传播到某个被当成"索引"用的地方,比如 token id、argmax,才会变成真正的非法地址);
   - 不是每次都崩,和输入的具体形状(尤其是"边界不是 tile 大小整数倍"的情况,比如 kv_len 不是 64 的整数倍导致最后一个 key tile 不满)相关。

   最快的验证方法不是继续套 sanitizer,而是**直接在 kernel 里加 `printf`**,在可疑的输出值上做 `isnan()/isinf()` 检查,打印相关的中间变量(比如 `m`/`den`/`acc` 这几个如果只有一个是 NaN,能立刻告诉你问题出在哪个来源缓冲区)。这个方法几秒钟出结果,比等 sanitizer 快几十倍。

   修法通常是:找到那块被"乘零"忽略、但没有被真正初始化的缓冲区(常见于 tile/pad 越界那部分,比如最后一个不满的 key tile 对应的 shared memory 尾部),显式清零它,而不是依赖"权重是 0 所以值无所谓"的假设。

## 环境相关的坑(本仓库特有,但类似项目也常见)

- **PTX 缓存不认编译选项,只认 `.cu` 源码内容。** `crates/cuda/src/nvrtc.rs` 的 `source_key()` 只 hash 了源码字符串 + arch + crate 版本,不包括 `CompileOptions`(比如临时加 `--generate-line-info`)。改了编译选项但没改 `.cu`/`.rs` 源码本身时,必须手动 `rm -rf ~/.cache/infero/ptx` 才会真正重新编译,否则会一直加载旧的 PTX,调试结果全是错的。
- **`pkill -f <pattern>` 可能杀死自己。** 如果 `<pattern>` 恰好是你正在执行的这条 ssh/bash 命令行本身包含的字符串(比如 `pkill -f "target/release/infero"` 这条命令自己的参数里就有这段文本),会把当前这个 shell 也匹配上杀掉,导致 ssh 连接意外断开。改用先 `pgrep -x <exact-process-name>` 或 `ps` 拿到精确 PID,再 `kill -9 <pid>`。
- **SSH ControlMaster 连接可能已经失效但 socket 文件还在**,导致新命令诡异地在中途返回 exit 255(尤其是上一次会话被打断、有残留的后台等待进程时)。表现为:简单命令(`echo ok`)也在执行到一半时断连。用 `ssh -O check <host>` 确认,坏了就 `ssh -O exit <host>` 强制关闭旧连接,让下一条命令重新握手。
- **崩溃一次的 CUDA context 通常整个报废**,同一进程后续所有请求都会失败(illegal address 是不可恢复的)。确认修复效果时必须重启 server 进程,不能用同一个已经崩过的进程接着测。
