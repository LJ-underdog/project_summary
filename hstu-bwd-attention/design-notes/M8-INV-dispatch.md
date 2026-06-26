# M8 investigation —— 解 VGPR 124-vs-248 矛盾 + 诊断 SiLU 26% 异常(coder pane 0.1)

M8(MI+B2+B3)已 commit `048f0a9a` 闭合。本单是**低风险 investigation(纯 profiling + 分析,不改库码、不收紧)**,为占用率类(B7,最大剩余 perf lever:MAIN occupancy 仅 10.6%、矩阵核闲 90%)解开两个前置开放问题。基线 HEAD=`048f0a9a`。

## 任务 1:解 VGPR 124-vs-248 矛盾(阻塞 occupancy 决策)
- scoping 的 rocprofv3 kernel-trace 报 MAIN VGPR=124;但编译器报告 `profile/M1-resource.md` 报 248。两者差 2×,决定占用率是否被 VGPR 卡(若 248→VGPR 限 ~8 waves/CU,与实测 6.96 plateau 吻合→B7 提占用率必须先砍 VGPR)。
- 用 **rocprofv3 + 编译期资源**双向核实:`rocprofv3 --kernel-trace` 抽当前 MAIN(canonical config)的 VGPR/AGPR/SGPR/Scratch/LDS;再从 amdclang++ `-Rpass-analysis=kernel-resource-usage`(或 build log / .s)取编译期 VGPR。**判定:真实 archVGPR 是多少?occupancy 限制器到底是 VGPR 还是 LDS(32KB/block ÷ 64KB)还是 dependency chain?** CDNA4 加法模型(archVGPR+AGPR)。
- 给结论:MAIN occupancy(10.6%/3.39 waves)的**真限制器**,及 B7(occupancy 1→2 / 提占用率)是否可行、要动什么(砍 VGPR? 砍 LDS? launch_bounds kBlockPerCu)。

## 任务 2:诊断 SiLU 26% 异常
- MI 复现:同 shape 下 SiLU MAIN(0.333ms)比 softmax(0.263ms)慢 1.27×(profile 335/266us)。直觉上 SiLU 该更便宜(无 LSE/exp)。
- 用 -perf + rocprofv3 对比 SiLU vs softmax MAIN 的:VALUBusy / MfmaUtil / VGPR / LDS / 指令数 / occupancy。**找根因**(SiLU 重算 S+dsilu 的 VALU?额外寄存器?occupancy 差异?)。判断是否有**便宜的 SiLU MAIN 快速 win**。

## 产出(写 `/tmp/hstu-bwd-design/M8-INV-findings.md`)
- VGPR 124-vs-248 定论 + MAIN occupancy 真限制器 + B7 可行性/代价评估。
- SiLU 26% 根因 + 是否有 quick win。
- **纯 investigation,不改库码、不 commit**。证据(rocprofv3 输出/资源数)落 `profile/M8-INV-*`。完成 pane 报,等 lead 据此决定是否上 B7/occupancy。
