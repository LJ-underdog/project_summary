# TEAM_CONFIG — perf_gfx950_verified

> 子类实例（继承 `/home/hanchang/agent_skill/.claude/skills/agent-team/SKILL.md`）
> 创建日期：2026-04-29
> 目标：在 gfx950 上用标准化脚本测出 FP8 tp=2 / tp=4 准确性能，结果写入 project_summary，供 gfx942 对比

---

## PROJECT
`perf_gfx950_verified`

## WORK_DIR / DOC_DIR / LOG_DIR
- WORK_DIR = `/home/hanchang/project_summary/step35-flash-support/16_perf_gfx950_verified`
- DOC_DIR  = `/home/hanchang/project_summary/step35-flash-support/16_perf_gfx950_verified`
- LOG_DIR  = `/home/hanchang/project_summary/step35-flash-support/16_perf_gfx950_verified/logs`

## CODE_ROOTS
- ATOM   = `/home/hanchang/junlin12_repos/atom`（commit `acff926d`）
- aiter  = `/home/hanchang/junlin12_repos/aiter`（commit `0f8164017`）
- CK     = aiter 子模块（commit `defd7ad29`）
- 模型   = `/root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/6eebda59dd87ca5729648ec7cfed0becfceb273e`

## GOAL（一句话，可量化）

在 gfx950（8x MI350X）上，用统一的 `perf_correctness_bench.py` 脚本，分别测出：
- **FP8 tp=2**（GPU 0,1）：TTFT / TPOT / total_lat / output_tokens，正确性 PASS
- **FP8 tp=4**（GPU 0,1,2,3）：同上

两组结果写入 DOC_DIR/RESULTS.md，供 gfx942 对比使用。
不设数值阈值（以实测为准），但正确性（non-BOS + 输出不为空）必须 PASS。

## CONSTRAINTS（红线）
- ❌ 不修改 ATOM / aiter / CK 任何源码
- ❌ 不修改 `perf_correctness_bench.py`（已调试完毕，用 `--model` 显式传路径）
- ❌ tp=2 和 tp=4 **串行**执行，不能同时占用 GPU（用户明确要求）
- ❌ 不使用 GPU5（硬件异常，~700ms/tensor）
- ✅ 每次推理结束后必须确认 VRAM 归零再进行下一个
- ✅ 所有结果和日志写入 DOC_DIR（持久路径，非 /tmp）

## ENVIRONMENT
```bash
# Python 环境
/opt/venv/bin/python

# 运行前置（必须 cd /tmp，避免 aiter namespace 问题）
cd /tmp

# tp=2 GPU 配置
CUDA_VISIBLE_DEVICES=0,1

# tp=4 GPU 配置（排除 GPU5）
CUDA_VISIBLE_DEVICES=0,1,2,3

# 模型路径（必须显式传，不能依赖 EngineArgs 默认值）
MODEL=/root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/6eebda59dd87ca5729648ec7cfed0becfceb273e

# HF 缓存
HF_HOME=/root/.cache/huggingface

# 降低日志噪声
AITER_LOG_LEVEL=WARNING

# 清 ATOM 缓存（改代码后必须清，本任务不改代码可跳过）
rm -rf /root/.cache/atom/*

# 推理结束后检查显存
rocm-smi --showmemuse 2>/dev/null | grep "GPU\["
```

## KNOWN_FACTS（已验证，无需重证）

| ID | 事实 | 来源 |
|----|------|------|
| F1 | gfx950 commit：ATOM `acff926d` / aiter `0f8164017` / CK `defd7ad29` | MEMORY.md |
| F2 | GPU5 硬件异常（~700ms/tensor），tp=2 用 GPU 0,1，tp=4 用 GPU 0,1,2,3（串行，不冲突）| MEMORY.md + 用户确认 |
| F3 | FP8 tp=2 短 prompt 历史值：TTFT=87ms, TPOT=14ms（V05 Exp2，input≈20 tokens） | verification_pipeline/results/v05_fp8_inference.md |
| F4 | FP8 tp=4 短 prompt 历史值：TTFT=86ms, TPOT=13ms（V06 Exp2，input≈20 tokens） | verification_pipeline/results/v06_fp8_tp4.md |
| F5 | 15_perf（gfx942 MI308X，10k input）：tp=2 TTFT=186ms/TPOT=5.2ms；tp=4 TTFT=110ms/TPOT=5.5ms | 15_perf_tp2_tp4_tp8_eval/progress/perf-t1.md, perf-t2.md |
| F6 | `perf_correctness_bench.py` 已调试完毕：必须用 `--model` 显式传路径，否则 EngineArgs 默认加载 Qwen | 本任务 Phase 0 排查结论 |
| F7 | 模型路径：`/root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/6eebda59dd87ca5729648ec7cfed0becfceb273e` | bash ls 验证 |
| F8 | `cudagraph_capture_sizes` 必须传字符串 `"[1]"` 给 EngineArgs（已在脚本中修复） | bench-tp2/tp4 agent 报告 |

## BASELINE（基准命令）

```bash
# FP8 tp=2
cd /tmp && \
CUDA_VISIBLE_DEVICES=0,1 \
HF_HOME=/root/.cache/huggingface \
AITER_LOG_LEVEL=WARNING \
/opt/venv/bin/python \
  /home/hanchang/project_summary/step35-flash-support/perf_correctness_bench.py \
  --tp 2 \
  --model /root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/6eebda59dd87ca5729648ec7cfed0becfceb273e \
  --input-tokens 10240 \
  --output-tokens 1024 \
  --runs 2 \
  --log-file /home/hanchang/project_summary/step35-flash-support/16_perf_gfx950_verified/logs/fp8_tp2.log \
  2>&1 | tee /home/hanchang/project_summary/step35-flash-support/16_perf_gfx950_verified/logs/fp8_tp2_full.log

# FP8 tp=4
cd /tmp && \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
HF_HOME=/root/.cache/huggingface \
AITER_LOG_LEVEL=WARNING \
/opt/venv/bin/python \
  /home/hanchang/project_summary/step35-flash-support/perf_correctness_bench.py \
  --tp 4 \
  --model /root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/6eebda59dd87ca5729648ec7cfed0becfceb273e \
  --input-tokens 10240 \
  --output-tokens 1024 \
  --runs 2 \
  --log-file /home/hanchang/project_summary/step35-flash-support/16_perf_gfx950_verified/logs/fp8_tp4.log \
  2>&1 | tee /home/hanchang/project_summary/step35-flash-support/16_perf_gfx950_verified/logs/fp8_tp4_full.log
```

**预期**：CORRECTNESS=PASS，TTFT/TPOT 有明确数值（用于与 gfx942 对比，不设阈值）

## TASK_SPECIFIC_VERIFICATION

- 每次推理后检查 `CORRECTNESS: PASS`（non-BOS，输出 ≥50 chars）
- 记录 Run1 和 Run2（取 Run2 稳态）
- 推理结束后用 `rocm-smi --showmemuse` 确认 VRAM% 归零
- 输出第一句话摘录（first 80 chars），确认是 Step-3.5-Flash 而非 Qwen（Qwen 输出会有 `<think>` 标签）

## 阶段结构

### Phase 0（串行，必须先跑）
- [ ] #000 [验证] 环境预检：确认脚本存在、GPU 显存干净、模型路径有效

### Phase 1（串行，tp=2 先跑完再跑 tp=4）
- [ ] #101 [验证] FP8 tp=2 benchmark（GPU 0,1）[depends: #000]
- [ ] #102 [验证] FP8 tp=4 benchmark（GPU 0,1,2,3）[depends: #101 VRAM 归零]

### Phase 2（汇总写报告）
- [ ] #201 [执行] 写 DOC_DIR/RESULTS.md（两组数据 + 与 gfx942 对比分析）[depends: #101 #102]

## Promotion Candidates
（待任务结束后追加）
