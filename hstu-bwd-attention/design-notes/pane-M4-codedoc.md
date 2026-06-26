# 派给 pane-2(角色:drafter)— M4 代码改动 review 文档(图文并茂 HTML,同 M2/M3 风格)

调度模式:tmux pane-2。把 **M4(group 模式)的代码改动**写成图文并茂 HTML,**结构/风格对齐你写过的 M2/M3 文档**。忠实真实代码,落笔前 Read 核实,不臆造。不要派 sub-teammate。

## 输出
`/root/workspace/hstu-b1052-report/hstu-bwd-M4-changes-20260608.html`

## 样式
逐字复用 M3 文档 `/root/workspace/hstu-b1052-report/hstu-bwd-M3-changes-20260608.html` 的 `<head>`/CSS/布局/配色(clay=新增改动、olive=复用、slate),左正文右 sticky TOC,`.code-block` 高亮,10 节骨架。三篇(M2/M3/M4)观感一致。

## 取材(以真实代码为准)
- 权威清单:`/tmp/hstu-bwd-design/M4-done.md`。
- bwd 文件均未跟踪(git `??`,无 M4-vs-M3 基线)→ M4 改动用 M4-done.md + 直接 Read 当前代码区段呈现,标注"M4 新增/改动"(同 M2/M3 处理)。要读的:
  - `hstu_attention_bwd_params.hpp`:`HstuAttentionGroupBwdParams`(填充后;per-group `group_*_ptr` + num_group/num_batch + alpha)。
  - `hstu_attention_bwd_kernel.hpp`:新增 `HstuAttentionBwdDQDKDVGroupKernel`(模板 `<PipelineLocal,PipelineNoLocal,DKEpi,DVEpi>`;operator() 里 jagged offset 复用 + `i_group=i_batch/num_batch_per_group` 取 per-group scale_p/mask + **运行时 `if(window>0){with_local+PipelineLocal}else{without_local+PipelineNoLocal}`** + 共享 write_dkdv lambda)。**强调 no_group kernel 未改**。
  - `hstu_attention_group_backward_dispatch.hpp`(新增):两 Problem/Pipeline、RunSilu、grid、POST、Run 门控(hdim/determ/softmax throw)。
  - `hstu_attention_group_backward_bf16.cpp`(新增 entry)、`hstu_attention_api.hpp`、`CMakeLists.txt`(BWD_INTERFACES_SRCS += group entry)。
  - `example_hstu_attention_bwd.cpp`:`run_group_hstu_bwd` + `-g/-g_max_seqlens/-g_local_lens/-g_context_lens/-g_minfull_lens/-g_attn_scales`;喂 `reference_group_hstu_attention_bwd`。
  - `test/run_bwd_tests.py`:`reject-group-g2`→8 个 M4 pass。
- 结果日志:`runs/run-bwd-M4-sweep.log`(8/8)、`runs/test-20260608-032623.log`(34/33PASS/1skip/exit0)、`runs/build-bwd-M4.log`。

## 文档要讲清(对齐 M3 文档 10 节,内容换 M4)
1. **总览+TL;DR**:M4 = group(jagged 超集)+ per-group device-ptr 超参(i_group 取)+ **运行时双 pipeline 选(per-group window 无法编译期定)**;alpha 全局、num_target per-batch。徽章:8/8 group 档 PASS(含全异构 + g4)/ 套件 34 案 exit0 / no_group 零回归。
2. **改动文件清单表**:params/kernel(新 GroupKernel)/dispatch(新)/entry(新)/api/cmake/harness/test;注 bwd 文件未跟踪;**本次无 instance 噪音**(group entry 直接实例化,无 instance 文件;核实 generate_instances 未动)。
3. **【图1】group 取数模型**:dim0=1 packed + cu_seqlens(复用 jagged)+ per-group 超参按 `i_group=i_batch/num_batch_per_group` 从 device 指针读;标 alpha 全局 / scale_p+mask per-group / num_target per-batch。
4. **【图2】运行时双 pipeline 选择**(M4 核心难点):per-group window 在同一 launch 可 0 可 >0 → 不能编译期定 kUseLocal → 同时实例化 with-local/without-local 两条 pipeline,kernel 内 `if(window>0)` 运行时选;causal 仍编译期轴。贴 kernel 关键片段。
5. **per-group scale_p / mask 取数**:`scale_p=group_attn_scale[i_group]?:1/group_max_seqlen_q[i_group]`;mask 4 参 per-group;贴片段。
6. **params 填充 + dispatch/entry/cmake**:GroupBwdParams 字段、group dispatch、新 entry、CMake 接入。
7. **harness `-g` 路**:per-group 数组 supplement、cu_seqlens、喂 reference_group;SiLU 跳过 GPU fwd。
8. **测试套件**:8 个 group pass case(g2-nomask/causal/pergroup-window/pergroup-attnscale/fallback/**heterogeneous**/g3/g4-singleton);no_group 零回归。
9. **对拍结果**:8/8 sweep 表(各档 + 误差 bf16 级,强调全异构/g4 验 per-group 取数真实)+ 套件 34/33/1skip/exit0。
10. **遗留**:M5 softmax(group O 现跳过,M5 要接 group fwd 产 O+LSE)/ cross 仅 self / 双 pipeline 代码体积(perf M8)/ Y-range 保守。

## 铁则
- 忠实代码;M4 改动 vs 复用分清;**核实本次有无 instance 噪音**(像 M3 那样查 mtime,如实写,别照搬)。
- 中文为主、字段/文件名英文;零基础友好。
- 自检 HTML 标签平衡 + TOC 锚点 + SVG(≥2)不溢出。
- 完成写 `/tmp/hstu-bwd-design/M4-codedoc-done.md`:小节/SVG/字节/标签平衡/覆盖文件 + 与 M4-done.md 核出的任何差异(诚实标注)。正文写进 HTML,不在终端长输出。
