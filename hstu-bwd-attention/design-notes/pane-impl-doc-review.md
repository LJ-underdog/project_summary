# 派给 pane-2(角色:reviewer + optimizer)— 审 + 优化 HSTU bwd 实现现状文档

调度模式:tmux pane-2。独立 review(你没参与写这份文档/代码)。对照**真实代码**核验准确性,再优化。P0(与代码不符的事实错误)直接最小修复;P1/P2 视情况优化或列出。不要派 sub-teammate。

## 对象
`/root/workspace/hstu-b1052-report/hstu-bwd-impl-status-20260604.html`(HSTU bwd 实现现状,pane-3 写)

## 基准(以真实代码/产物为唯一判据,逐个 Read/核)
代码 `/root/workspace/ck_hstu/example/ck_tile/18_hstu_attention/`:
- `hstu_attention_bwd_params.hpp`、`hstu_attention_no_softmax_bwd_pipeline.hpp`、`hstu_attention_bwd_kernel.hpp`、`hstu_attention_batched_backward_dispatch.hpp`、`hstu_attention_no_group_backward_bf16.cpp`、`example_hstu_attention_bwd.cpp`、`hstu_attention_api.hpp`、`instances/hstu_attention_batched_backward_*`、`CMakeLists.txt`、`generate_instances.py`
- 进度报告:`/tmp/hstu-bwd-design/M0-done.md`、`M1-done.md`
- 可复跑验证:`cd /root/workspace/ck_hstu && ./build/bin/tile_example_hstu_attention_bwd -prec=bf16 -b=2 -nhead=2 -hdim_qk=64 -hdim_v=64 -seqlens=128 -softmax=0 -causal=0 -attn_scale=1.0 -v=1`(应三梯度 PASS,exit 0)

## 第一部分:REVIEW(对照代码,逐条 ✅/⚠️/❌)
1. **文件清单准确性**:每个文件的行数(`wc -l` 核)、角色、"新写 vs 改动"、"复用了哪些 FMHA 组件"是否与代码一致。
2. **include/依赖图(图1)**:文档画的 include 关系是否与文件实际 `#include` 一致(尤其 dispatch→`no_softmax_bwd_pipeline`+`bwd_kernel`、pipeline→`block_fmha_bwd_pipeline_default_policy`、复用的 `GenericAttentionMask`/`Default2DEpilogue`/`BlockFmhaBwdPipelineProblem`)。
3. **调用链(图2)**:harness→fwd(产O)→`no_group_backward_bf16`→BOOL_SWITCH→`run_batched_backward_dispatch::Run`→`RunSilu`→memset dq_acc→MAIN(atomic 写 float dq_acc + dk/dv)→POST(convert dq_acc→dq)→CPU reference→check_err —— 是否与 dispatch/kernel 实际代码一致(GPU/CPU 边界、dq_acc 作用)。
4. **七阶段(图3)+ 代码佐证**:STAGE2(alpha+silu/dsilu)、STAGE5(ds=dp·g)、收尾(dQ/dK ×alpha、dV 不乘)的描述是否与 `no_softmax_bwd_pipeline.hpp` 实际一致;silu/dsilu 公式与代码 device 函子一致。
5. **复用 vs 新写**:文档分类是否准确(直接 include 复用 vs 新写)。
6. **关键决策**:4 条(ck_tile 版本差异、自写 kernel 因双 scale、float dq_acc+atomic+POST、NO_BIAS dummy)是否与 M1 报告/代码相符。
7. **覆盖面/TODO 表**:✅6(batched/SiLU/no-mask/bf16/hd64/atomic)/⏳7(M2-M8)是否与 dispatch 的 throw 门控实际一致(causal/softmax/group/deterministic 是否真 throw)。
8. **数值/资源数**:M1 的 err、R1 结论、R2(VGPR248/AGPR0/Scratch0/occ2)是否与 M1 报告一致;**建议你亲自复跑一次上面的验证命令**确认 PASS+exit0,把结果作为佐证。

## 第二部分:OPTIMIZE
- 可读性、过渡、零基础友好(术语首现解释);图是否清晰、有图例、不溢出。
- 在不改事实的前提下让"文件关联 + 调用逻辑"更一目了然(如补一句总览导读、强化图注)。
- 与 design HTML / 既有报告风格一致;配色 clay(新写)/olive(复用)统一。

## 铁则
- ❌(与代码不符)直接最小修复并记录;P1/P2 优化项可改可列。**不臆造**——拿不准就 Read 代码或复跑。
- 改完独立复核 HTML 标签平衡(div/section/svg/h2/h3/table)+ TOC 锚点 + SVG 不溢出。
- 报告写 `/tmp/hstu-bwd-design/impl-doc-review-done.md`:逐条判定(✅/⚠️/❌ + 代码证据 + 改法)、改了哪些、复跑验证结果、标签平衡。正文改进写进 HTML,不在终端长输出。
