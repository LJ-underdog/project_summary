# 认真 review:HSTU bwd 总览讲义(pane 0.2)

pane-3 已优化 `/root/workspace/hstu-b1052-report/hstu-bwd-overview-20260615.html`(加了 5 张 SVG + 逻辑梳理)。你做**认真独立 review**,要求**逻辑清晰 + 图文明确正确**。只读不改文档,问题列给 lead。

## 审查清单(逐条 GREEN/RED + 证据)
1. **逻辑清晰**:叙事流(概述→能力边界→里程碑时间线→M8 性能→方法论→教训→git 链→后续)是否顺、无跳跃/重复/矛盾;接手者能否据此快速建立全局。
2. **★ 图文明确(重点,新加的 5 张 SVG)**:逐张看——① 里程碑能力累积时间线 ② bwd kernel 结构(PRE/MAIN 5-GEMM/POST)③ 能力边界矩阵 ④ 四方闭合+证据栈 ⑤ M8 perf 条形图。每张:viewBox 不溢出/文字不重叠/标注正确/与正文呼应(有 caption + 正文引用)、配色用 :root 变量、**无 dingbat/emoji**。图里的数字/结构与正文及源一致(如 perf 条形图的 1.30/1.60/4.71… 与 benchmark.csv)。
3. **数字/事实核对**(对照 `docs/HANDOFF.md` + `docs/*-done.md` + `candidates.jsonl` + `benchmark.csv` + git):能力边界、253/253、causal 1.25–1.60×/window 4.7–9.8×、12 节点 git 链、各 commit hash、VALUBusy 41%/SiLU 26% 等——无臆造/无优化引入的错。
4. **20 个超链接全有效**(指向同目录存在文件)+ 那篇作废的 fwd-group-maxseqlen-bug 已正确排除/标注。
5. **范围诚实**:真 reject(hdim>256)vs out-of-scope(target_in_kv/非方形/dO layout)区分准确,无夸大。
6. **HTML 质量**:浏览器可渲染、无占位符、无外链(锚点+sibling html 除外)、无 dingbat/emoji。

## 产出
写 `/tmp/hstu-bwd-design/overview-review.md`,逐条 GREEN/RED + 证据,RED/图问题给具体位置 + 应改。结论:可发布/需改。完成 pane 报。
