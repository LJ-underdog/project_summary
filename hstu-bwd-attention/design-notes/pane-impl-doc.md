# 派给 pane-3(角色:drafter)— HSTU bwd 实现现状文档(图文并茂 HTML,供 review)

调度模式:tmux pane-3。把**当前 HSTU bwd 实现成果**写成一份图文并茂、零基础友好的 HTML(供用户 review):写了哪些文件、文件间关联(include/依赖)、调用逻辑(调用链/数据流)、复用 vs 新写、当前覆盖面与 TODO。不要派 sub-teammate。**忠实当前真实代码**,落笔前 Read 实际文件,不臆造。

## 输出
`/root/workspace/hstu-b1052-report/hstu-bwd-impl-status-20260604.html`(与设计 HTML 同目录)

## 样式
复用 `/root/workspace/hstu-b1052-report/hstu-bwd-design-20260604.html`(或 `ck-vs-hstu-bwd-20260529.html`)的整套 `<head>`/CSS + 左正文右 sticky TOC。配色:clay=#D97757 标新写 HSTU 代码,olive=#788C5D 标复用 FMHA,slate 文字。

## 必读真实源(以代码为准,逐个 Read)
代码目录 `/root/workspace/ck_hstu/example/ck_tile/18_hstu_attention/`:
- `hstu_attention_bwd_params.hpp`(119 行)— bwd 参数结构
- `hstu_attention_no_softmax_bwd_pipeline.hpp`(518 行)— SiLU MAIN pipeline(核心)
- `hstu_attention_bwd_kernel.hpp`(376 行)— MAIN kernel 包装 + POST convert kernel
- `hstu_attention_batched_backward_dispatch.hpp`(233 行)— dispatch(RunSilu / Run 门控)
- `hstu_attention_no_group_backward_bf16.cpp`(33 行)— bf16 入口
- `example_hstu_attention_bwd.cpp`(483 行)— 对拍 harness
- `hstu_attention_api.hpp`(改动:bwd 声明)、`reference_hstu_attention_bwd.hpp`(oracle)、`instances/hstu_attention_batched_backward_*`、`CMakeLists.txt`(bwd target)、`generate_instances.py`(bwd 分支)
进度报告(背景):`/tmp/hstu-bwd-design/M0-done.md`、`M1-done.md`;设计:`hstu-bwd-design-20260604.html`。

## 已知 include/依赖关系(供画图,落笔前 Read 复核)
```
example_hstu_attention_bwd.cpp ──includes──> bwd_params.hpp, reference_hstu_attention_bwd.hpp, api.hpp, (fwd type_config/util/params)
no_group_backward_bf16.cpp ─> batched_backward_dispatch.hpp, instances/..._ref.hpp, bwd_params.hpp
batched_backward_dispatch.hpp ─> ck_tile/ops/fmha.hpp, epilogue.hpp, bwd_params.hpp, no_softmax_bwd_pipeline.hpp, bwd_kernel.hpp
no_softmax_bwd_pipeline.hpp ─> block_fmha_bwd_pipeline_default_policy.hpp (复用 FMHA policy), block_attention_bias_enum.hpp
api.hpp ─> bwd_params.hpp
```

## 文档要讲清(核心)
1. **总览**:HSTU bwd 现状 = **M0 脚手架 + M1 闸门已通过**(batched+SiLU+no-mask+bf16+hd64+atomic 端到端对拍 PASS);gfx950/CDNA4 主目标。TL;DR 含三枚徽章(M1 PASS / R1 policy 零覆写复用 / R2 VGPR248·AGPR0·Scratch0·occ2)。
2. **文件清单表**:每个 bwd 文件 — 路径 / 行数 / 角色 / 新写还是改动 / 复用了哪些 FMHA 组件。
3. **【图1 依赖图】include/依赖关系 SVG**:上面那张 include 图,clay 标新写文件、olive 标复用的 FMHA(`block_fmha_bwd_pipeline_default_policy`、`GenericAttentionMask`、`Default2DEpilogue`、`BlockFmhaBwdPipelineProblem`)。
4. **【图2 调用链/数据流 SVG】运行时调用逻辑**(从 harness 到 kernel):
   `example main → (gen Q/K/V/dO) → GPU fwd(产 O) → hstu_attention_no_group_backward_bf16 → BOOL_SWITCH → run_batched_backward_dispatch::Run → RunSilu → [hipMemset dq_acc=0] → launch MAIN kernel(HstuAttentionBwdDQDKDVKernel → HstuAttentionBwdDQDKDVPipelineKRKTRVR:5 GEMM+silu/dsilu,atomic 写 float dq_acc + 写 dk/dv)→ launch POST(hstu_bwd_convert_dq_kernel:dq_acc float→dq bf16)→ CPU reference_hstu_attention_bwd → ck_tile::check_err`。标注哪步在 GPU、哪步在 CPU、dq_acc workspace 的作用。
5. **【图3 MAIN 七阶段 SVG】**(可复用设计 HTML 的思路,但标注"当前已实现 SiLU 路、softmax 为 M5 TODO"):STAGE1-7,高亮 STAGE2(alpha+silu/dsilu)/STAGE5(ds=dp·g)/收尾(dQ,dK ×alpha,dV 不乘)是 HSTU 改写点,其余 5 GEMM 复用。
6. **复用 vs 新写**:明确直接 include 复用的 FMHA 组件(policy/mask/epilogue/problem)vs 新写的 HSTU(pipeline/kernel/dispatch/params/harness)。
7. **关键工程决策**(从 M1 报告):①目标 ck_tile 是 ck_hstu 自带较新版(接口与 /root/ck 不同,已对齐);②自写 kernel 因 FMHA kernel 只传单 scale、HSTU 需 alpha+scale_p 两个;③float dq_acc + atomic + POST convert(dq 是 bf16);④保留 BiasEnum=NO_BIAS dummy 以复用 default policy。
8. **当前覆盖面 vs TODO 表**:✅ batched/SiLU/no-mask/bf16/hd64/atomic;⏳ M2 mask(causal throw)/M3 jagged/M4 group/M5 softmax/M6 deterministic/M7 fp16+多hdim/M8 perf(trload)。
9. **验证现状**:对拍 oracle = CPU reference;M1 数值(attn_scale=1.0:dQ/dK/dV err bf16 舍入级、dK 逐位0);6/6 稳定性 case;R2 资源数。
10. **如何构建/运行**(给 review 者复现):`cmake -B build -DBUILD_DEV=OFF -DGPU_TARGETS=gfx950` + `cmake --build build --target tile_example_hstu_attention_bwd -j` + 运行命令。

## 铁则
- 忠实当前代码;文件行数/include/调用以实际 Read 为准,不臆造。区分"已实现"与"TODO"。
- 中文为主、字段/文件名英文;零基础友好(关键术语一句白话)。
- 自检 HTML 标签平衡(div/section/svg/h2/h3/table)+ TOC 锚点 + SVG(≥3)文字不溢出。
- 完成写 `/tmp/hstu-bwd-design/impl-doc-done.md`:小节/SVG 数/字节/标签平衡/覆盖了哪些文件。正文写进 HTML,不在终端长输出。
