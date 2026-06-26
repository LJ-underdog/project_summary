# M6 deterministic 图文 HTML 报告 + 顺带文档级 review 派单 (pane-3)

> 你写过 M5、M5b 报告。这单做 M6(deterministic dQ):① 图文 HTML 讲义;② 写时顺带第三方文档级 review(coder=pane-1、reviewer=pane-2 已过,你查漏)。
> M6 已 promoted、git `c79d3296`。范围 = no_group(batched+jagged)× SiLU+softmax。

## 输入(全读)
- 规格 `/tmp/hstu-bwd-design/M6-dispatch.md`;coder 自述 `M6-done.md`;reviewer 结论 `M6-review-findings.md`(很详尽,含 A.5 atomic-vs-determ 逐位 diff=0、B4 split_stride 一致性证明、**O1 group+determ 静默走 atomic**)。
- 活状态 `/root/workspace/hstu-bwd-impl/docs/HANDOFF.md` M6 节(权威,含 O1 修正)。
- 代码(git `c79d3296`,相对 M5b `dc8c6b21` 的改动):
  - `hstu_attention_bwd_kernel.hpp`(dq_acc 窗口 set/atomic 分叉 + split 偏移;新 POST `hstu_bwd_reduce_convert_dq_kernel`)
  - `hstu_attention_batched_backward_dispatch.hpp`(`launch_main_and_post` + kIsDeterministic 轴)
  - `hstu_attention_no_group_backward_bf16.cpp`(BOOL_SWITCH_3)
  - `generate_instances.py`(determ 轴)+ 4 新 `*deterministic*` instance
  - `example_hstu_attention_bwd.cpp`(determ workspace)
  - **复用未改**:两 pipeline 的 `if constexpr(kIsDeterministic) store_tile` 分支(早就在)
- 对拍/证据:`runs/run-M6-correctness.log`、`run-M6-repro.log`、`test-20260609-093735.log`。

## 报告(html-report skill,风格对齐你写的 M5/M5b 报告)
输出:`/root/workspace/hstu-b1052-report/hstu-bwd-M6-deterministic-20260609.html`。结构建议:
1. **为什么需要 determinism**:atomic_add 并行归约顺序不定 → dQ 非逐位可复现;训练复现/调试需要 bit-reproducible。
2. **机制图**:atomic 路(N 个 KV-block atomic_add 进 1 份 dq_acc)vs determ 路(N 个 block plain-store 进各自 split 副本 → POST 固定升序 reduce)。画 split workspace 布局 + reduce。
3. **关键实现**:kernel `mop = determ ? set : atomic_add` + `base += i_tile_n*split_stride`;pipeline 已有的 store/update 分支;POST 固定序 `Σ_s dq_acc[s*stride+i]`;num_splits=ceil(seqlen_kv/kN0)=grid.x;dispatch `launch_main_and_post` 共享 + determ 轴;instance determ 轴翻倍(extern template);harness workspace×num_splits。
4. **为什么 bit-reproducible**:无 atomic + 固定求和序 ⇒ 与 block 调度无关。对比 atomic 仅"本机偶然稳定"。
5. **验证**:四方闭合;正确性对拍表;**可复现性 byte-identical**(multi-split);**A.5 atomic-vs-determ diff=0**(determ=正确梯度,非自洽错值);套件 77/77/0/0;atomic 零回归。
6. **范围与 O1**:no_group only;**诚实写出 O1——group+determ 现静默走 atomic(非可复现)、throw 不可达,M6b 待修**(别美化)。M7/M8 后续。

## 顺带 review(写时同步,单独报 lead)
全新视角核:memory_op set↔store / atomic↔update 匹配、split 偏移无重叠/越界、POST 固定序、num_splits==grid.x、写读 split_stride 一致(silent-corrupt 风险)、atomic 路重构后零回归、O1 描述准确。**产出** `/tmp/hstu-bwd-design/M6-doc-review.md`(GREEN 确认 / 或疑点+文件:行号)。发现真问题立刻停下报 lead。

## 注意
只读不改源码;代码片段引真实行号。不动 fwd / 不碰 M6b/M7。
