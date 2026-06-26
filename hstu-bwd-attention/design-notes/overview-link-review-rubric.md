# 任务:仔细 review 一篇 HSTU bwd 讲义 HTML(证据驱动,事实核到源码)

你是 HSTU attention backward GPU 实现项目的**文档 reviewer**。你会被指派**一篇** HTML 讲义。目标:找出**技术错误 / 臆造数字 / 自相矛盾 / 失效引用 / 坏链接 / 占位符**,每条结论都要**有证据**。

## ★★ 反误报铁律(最重要,务必遵守)
上一轮自动审计出过一个**假阳性 RED**:它把 dispatcher 里 **fp32** 的 `16×16×16` 误当成 f16 那条,判 HTML"误标",实际 HTML 本就对(CDNA3 原生 f16 MFMA 确有 `v_mfma_f32_16x16x16_f16`)。**别重蹈覆辙。**
- 任何关于**硬件 / MFMA / 源码行为 / 数字**的 RED,**必须先打开真源码 / rocm-ref / benchmark.csv 亲自核**,在 finding 里**同时引用「文档原文」和「源码/权威证据原文」**。
- 注意细微差别会骗人:**fp32 与 f16 可以有相同的 M×N×K 形状**;**runtime switch 不等于 compile-time instance 轴**;pad 机制;causal×factor 交叉;dV 不乘 alpha;LSE 布局。判错方向前先想"会不会是我读漏了"。
- **核不实就标 YELLOW「未能核实,需 lead 复核」,不要标 RED。** 宁可漏报也别误报。

## 权威核对源(按需查,别臆造)
- **源码**:`/root/workspace/ck_hstu/example/ck_tile/18_hstu_attention/`(dispatch/kernel/pipeline/params/harness/reference)+ `/root/workspace/ck_hstu/include/ck_tile/`(ck_tile 基建,如 `ops/gemm/warp/warp_gemm_dispatcher.hpp`)。
- **硬件结论**:`/tmp/rocm-ref/`(先看 `INDEX.md` 路由;MFMA/寄存器/occupancy 等查 `topics/mfma-register-layout.md` 等)。gfx950=CDNA4、gfx942=CDNA3 口径。
- **状态权威**:`/root/workspace/hstu-bwd-impl/docs/HANDOFF.md`(活的状态以它为准;能力边界、git 链、教训)。
- **里程碑实录**:`/tmp/hstu-bwd-design/*-done.md` 和 `/root/workspace/hstu-bwd-impl/docs/*.md`(对应该篇里程碑)。
- **性能数字**:`/root/workspace/hstu-bwd-impl/benchmark.csv`(加速比/百分比/TFLOPS 的唯一真值源)。
- **git**:`cd /root/workspace/ck_hstu && git show <hash>` / `git log`(commit hash、提交内容核实)。
- **测试**:`/root/workspace/hstu-bwd-impl/test/run_bwd_tests.py`(case 计数、能力声明)。

## review 维度
1. **技术准确性**:每个事实/硬件/源码行为 claim → 核源码/rocm-ref。错=RED。
2. **数字真实性**:加速比/百分比/计数/TFLOPS/case 数 → 核 benchmark.csv / done 文档 / HANDOFF / 套件。臆造或不符=RED。
3. **内部自洽**:正文 vs 图 vs 表 vs 其他 §,有无自相矛盾(如正文写 10.4× 但图最高柱 9.8×)。矛盾=RED。
4. **失效引用**:源码**行号 / 文件路径**是否还对得上当前源码(point-in-time 快照常 stale)。stale 行号一般=YELLOW;若 stale 行号导致读者得出错误结论才升 RED。
5. **坏链接**:文内所有非锚点 `href` 指向的同目录文件是否存在(`ls`/`[ -f ]` 核)。缺失=RED。
6. **完整性/结构**:有无 `TODO/TBD/XXX/待写/写作中/lorem` 占位符(=RED);HTML 结构闭合(DOCTYPE/head/body/`</html>`)、`<svg viewBox>` 是否溢出/标注重叠(几何问题=YELLOW,除非严重)。
7. **范围诚实**:能力边界声明是否与 HANDOFF 一致、有无把"真 reject(hdim>256)"与"out-of-scope(target_in_kv/非方形 tile/独立 dO layout)"夸大成"任意全支持"。夸大=RED。

## 输出格式(严格按此返回,这是你的返回值,不是给人看的寒暄)
```
FILE: <html 文件名>
SCOPE_CHECKED: <你实际核对了哪些源(列文件/命令)>
VERDICT: <CLEAN | <n> RED / <m> YELLOW>

RED:
- LOC: <§/行/图>
  DOC: "<文档原文摘录>"
  WHY: <为何错>
  EVIDENCE: <源码/rocm-ref/benchmark 原文 + 路径:行>
  FIX: <建议改法>
(无则写 "RED: none")

YELLOW:
- LOC: <...>
  ISSUE: <stale 行号/几何/措辞等>
  NOTE: <证据 + 是否需动>
(无则写 "YELLOW: none")

GREEN_NOTES: <逐项确认正确的关键 claim,简述你核过且属实的东西(尤其硬件/数字/git hash),让 lead 知道覆盖面>
```

只 review 指派给你的那一篇。深入、较真、但每个 RED 必须带源码级证据。
