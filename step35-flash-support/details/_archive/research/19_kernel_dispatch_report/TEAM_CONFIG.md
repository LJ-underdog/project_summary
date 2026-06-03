# TEAM_CONFIG — kernel_dispatch_report

> 创建日期：2026-04-29
> 目标：产出一份完整报告，说明 Step-3.5-Flash-FP8 在 tp=2/tp=4 下每类操作究竟调用的是 torch op、CK kernel 还是 ASM kernel，并通过实验验证

## WORK_DIR / DOC_DIR / LOG_DIR
- WORK_DIR = `/home/hanchang/project_summary/step35-flash-support/19_kernel_dispatch_report`
- DOC_DIR  = `/home/hanchang/project_summary/step35-flash-support/19_kernel_dispatch_report`
- LOG_DIR  = `/home/hanchang/project_summary/step35-flash-support/19_kernel_dispatch_report/logs`

## GOAL
产出 `REPORT.md`，精确说明 Step-3.5-Flash-FP8（FP8 tp=2 + FP8 tp=4）推理时每类操作的 kernel 类型：
- **torch op**（F.linear、torch.mm、F.scaled_dot_product_attention 等）
- **CK kernel**（aiter fused_moe 2-stage、ck_moe_stage1/stage2 等）
- **ASM kernel**（fmoe_fp8_blockscale_g1u1、bf16gemm_bf16_tn_256x256 等）

报告须覆盖：MoE routed experts、Attention（prefill/decode）、Linear proj（BF16）、lm_head，区分 prefill 和 decode 场景。

## CONSTRAINTS
- ❌ 不修改 ATOM/aiter/CK 任何源码
- ❌ tp=2 和 tp=4 串行执行（不同时占用 GPU）
- ❌ GPU5 禁用
- ✅ 单 kernel 验证用 GPU 4,6（tp=2 推理完成后）
- ✅ 所有结论须有代码行号或日志行号支撑

## ENVIRONMENT
```bash
cd /tmp
CUDA_VISIBLE_DEVICES=0,1   # tp=2（或 0,1,2,3 for tp=4）
HF_HOME=/root/.cache/huggingface
AITER_LOG_LEVEL=INFO
AITER_LOG_TUNED_CONFIG=1   # 关键：打印 kernel dispatch 信息
/opt/venv/bin/python ...

MODEL=/root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/6eebda59dd87ca5729648ec7cfed0becfceb273e
```

## KNOWN_FACTS
- MoE routed experts：prefill 走 ASM（fmoe_fp8_blockscale_g1u1，run_1stage=True when token>32 and inter%256==0），decode 走 CK 2-stage（run_1stage=False when token=1）
- BF16 linear（attn/dense/shared/lm_head）：aiter/tuned_gemm.py tgemm.mm → bf16_tuned_gemm.csv 全 miss → torch.mm
- Attention：aiter flash/paged attention kernel（非 torch SDPA）
- tp=2 inter_dim=640（640%256==0，满足 ASM 条件）；tp=4 inter_dim=384（384%256≠0，需查是否走 ASM）
- commit：ATOM acff926d / aiter 0f8164017

## 阶段结构
- Phase 0 #000：预检
- Phase 1（并行）：
  - #101 代码追踪：逐类操作读 dispatch 路径（无需 GPU）
  - #102 FP8 tp=2 推理日志：带 AITER_LOG_TUNED_CONFIG=1 跑，提取 kernel dispatch
  - #103 单 kernel 验证：用最小脚本直接测 fused_moe FP8 blockscale dispatch
- Phase 2 #201：写 REPORT.md
