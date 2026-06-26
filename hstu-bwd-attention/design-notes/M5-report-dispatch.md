# M5 图文 HTML 报告 + 顺带文档级 review 派单 (pane-3)

> 双目标:① 写一篇图文并茂的 M5 softmax bwd HTML 讲义;② **写的过程中做一遍全新的文档级 review**(你是第三方独立视角,coder=pane-1、code-reviewer=pane-2 已过,你再过一遍能抓漏)。
> M5 已 promoted、对拍全绿;你的报告面向"想搞懂 HSTU softmax bwd 怎么实现的人"。

## 输入材料(全读)
- 规格:`/tmp/hstu-bwd-design/M5-dispatch.md`
- coder 自述:`/tmp/hstu-bwd-design/M5-done.md`
- reviewer 结论:`/tmp/hstu-bwd-design/M5-review-findings.md`(含 LSE 数值盲区分析)
- 活状态:`/root/workspace/hstu-bwd-impl/docs/HANDOFF.md` §6
- 代码(git `aced5784`):
  - 新 `example/ck_tile/18_hstu_attention/hstu_attention_with_softmax_bwd_pipeline.hpp`(MAIN,STAGE2/5)
  - `hstu_attention_bwd_kernel.hpp`(softmax kernel + PRE `hstu_bwd_dot_do_o_kernel`)
  - `hstu_attention_batched_backward_dispatch.hpp`(`RunSoftmax`)
  - `example_hstu_attention_bwd.cpp`(harness:fwd 开 is_training 产 LSE + 转置喂 reference)
  - oracle:`reference_hstu_attention_bwd.hpp` kUseSoftmax 分支
- 蓝本对照:`include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp`
- 对拍数据:`/root/workspace/hstu-bwd-impl/runs/run-M5-sweep.log`、`runs/test-20260608-065351.log`

## 报告要求(用 html-report skill)
- skill:`/root/.claude/skills/html-report/`(SKILL.md);**风格对齐**已有的 `/root/workspace/hstu-b1052-report/hstu-bwd-M4-changes-20260608.html`(同目录同风格)。
- 输出:`/root/workspace/hstu-b1052-report/hstu-bwd-M5-softmax-20260608.html`。
- 内容建议结构(图文并茂、有公式有代码片段有数据表):
  1. **M5 是什么**:在 M0–M4b(SiLU 全模式)之上补 softmax bwd(no_group=batched+jagged);消费上游 fwd kStoreLSE。
  2. **数学**:softmax bwd 链路 `P=exp(αS−LSE)` / `D=ΣO⊙dO` / `dS=P(dP−D)` / dV=PᵀdO / dQ=dSK / dK=dSᵀQ;与 SiLU 路对比(silu/dsilu vs exp/LSE,掩码 0 vs −inf,scale_p vs 无)。画个 3-kernel(PRE→MAIN→POST)+ STAGE 流程图(ASCII/SVG 皆可)。
  3. **关键实现**:STAGE2(−inf 掩 + log2 域 exp2 + get_validated_lse)、STAGE5、scale 接线(dV 不乘)、LSE/D 布局([batch,head,seq] 连续 + reference 转置 + 四方溯源图)、PRE kernel、smem 复用 SiLU 预留 region。
  4. **怎么复用 FMHA**:SiLU pipeline 是 FMHA kr_ktr_vr 的 fork,softmax 又把 FMHA 的 LSE/D 加载搬回——讲清复用边界。
  5. **验证**:三方闭合(coder/reviewer/lead)、对拍数据表(逐档误差)、套件 60/59/0/1、零回归;**诚实写出 LSE 数值盲区**及其闭合方式。
  6. **范围与后续**:M5b group / cross softmax 未做;M6/M7。

## 顺带 review(写时同步做,产出附在报告末 + 单独报 lead)
你边读边以全新视角核这些(pane-2 已过,你查漏补缺):
- 数学公式 ↔ 代码是否真一致(尤其 log2e 因子、−inf 方向、D 的 per-row 广播)。
- LSE 四方布局(fwd 写 / bwd 读 / PRE 写 / reference 转置)地址是否真对齐——自己推一遍偏移。
- 有没有 pane-2 没提的边界/可读性/潜在 silent-wrong。
- 报告里任何你"讲不圆"的地方往往就是代码可疑点——记下来。
**产出**:在 `/tmp/hstu-bwd-design/M5-report-dispatch.md` 同目录写 `/tmp/hstu-bwd-design/M5-doc-review.md`(几行即可:GREEN 确认 / 或发现的疑点+文件:行号),报 lead。发现真问题立刻停下报 lead,别只写进报告。

## 注意
- 不改任何源码(你是文档+review,只读)。报告里代码片段要与磁盘一致(引用真实行号)。
- 不动 fwd、不碰 M5b。
