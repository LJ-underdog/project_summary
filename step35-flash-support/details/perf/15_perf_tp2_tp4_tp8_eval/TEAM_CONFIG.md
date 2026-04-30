# TEAM_CONFIG — perf_tp_eval

> 子类实例（继承 `~/.claude/skills/agent-team/SKILL.md`）
> 父任务：`fp8-tp4-repro`（已 CLOSED）
> 创建日期：2026-04-29

## PROJECT
`fp8-tp4-repro / perf_tp_eval`

## WORK_DIR / DOC_DIR / LOG_DIR
- WORK_DIR = `/home/junlin12/project_fp8_tp4_repro/perf_tp_eval`
- DOC_DIR  = `/home/junlin12/project_fp8_tp4_repro/perf_tp_eval`
- LOG_DIR  = `/home/junlin12/project_fp8_tp4_repro/perf_tp_eval/logs`

## CODE_ROOTS
- ATOM   = `/home/junlin12/ATOM`（commit `acff926`）
- aiter  = `/workspace/aiter`（commit `0f8164017`，含 fused_moe.py:881-886 dirty patch）
- CK     = `/workspace/aiter/3rdparty/composable_kernel`（commit `defd7ad29`）
- 模型   = `stepfun-ai/Step-3.5-Flash-FP8`（HF_HOME=`/workspace/hf_cache`）

## GOAL（一句话，可量化）
1. **测出 tp=2 / tp=4 单请求 TTFT 与 TPOT**（input=10k tokens，output=1024 tokens，concurrency=1，温度=0）
2. **静态评估 tp=8 可行性**（读 ATOM / aiter / CK 代码 + 项目内 KNOWN_FACTS，列 tp=8 还需补做的工作清单，区分 block / 已自动满足 / 待实测）
3. **实测 tp=8 baseline**（仅"能起服 + 单 prompt 1 轮 generate 输出语义合理"，不验 byte-identical / 不验 PASS V 则）

## CONSTRAINTS（红线）
- ❌ 不修改 ATOM / aiter / CK 任何源码（包括不动 fused_moe.py:881-886 现有 dirty patch）
- ❌ 不动项目内已存的 doc（不改 SESSION_HANDOFF / FINAL_REPORT / PROJECT_SUMMARY / MIGRATION_REPORT / TEAM_CONFIG / docs/baseline_*.md / progress/teammate-*.md）
- ❌ 不动 `~/ATOM` 之外的源仓库 git state（不 commit / 不 push / 不 reset）
- ✅ 允许在 `WORK_DIR/` 下新建任何文件（脚本、log、progress、报告）
- ✅ 允许 GPU 命令（rocm-smi / 启动 ATOM 推理 / kill 进程），但每次 GPU 占用结束必须 kill 干净
- ✅ 必须中文 + file:line 引用

## ENVIRONMENT
```bash
# 启动 ATOM 推理的标准环境（来自 SESSION_HANDOFF.md:186-198）
cd /tmp && \
HF_HOME=/workspace/hf_cache \
CUDA_VISIBLE_DEVICES=<视 tp 设置> \
AITER_LOG_LEVEL=INFO \
AITER_LOG_TUNED_CONFIG=1 \
/opt/venv/bin/python <脚本路径> \
  --model stepfun-ai/Step-3.5-Flash-FP8 \
  --trust-remote-code \
  --tensor-parallel-size <2|4|8> \
  --level 0 --temperature 0 --max-tokens <视 output> \
  --max-num-batched-tokens 16384 --max-num-seqs 1 \
  2>&1 | tee LOG_DIR/<run_id>.log
```

- **GPU 数量**：本机 8 张 MI308X（`rocm-smi --showid` 显示 GPU[0]-GPU[7]）
- **HF 缓存**：`/workspace/hf_cache`（已下载 `stepfun-ai/Step-3.5-Flash-FP8` 全 44 shards）
- **JIT cache**：`~/.cache/aiter/build/`（不要清，复用编译产物）
- **简单 prompts 例**：`atom/examples/simple_inference.py` 用了 4 个短 prompt + max-tokens=128；本任务需扩到 10k input + 1024 output，**不能改 simple_inference**，必须在 `WORK_DIR/` 写新脚本（template：直接调 `from atom import LLM, SamplingParams; LLM(...).generate([long_prompt], SamplingParams(max_tokens=1024, temperature=0))`，对应 `simple_inference.py:6-7,49-74`）
- **TTFT/TPOT 测量**：脚本必须显式记录 `t0 = time.perf_counter()` → 第一个 token → TTFT；之后每个 token 间隔 → TPOT 平均。ATOM `llm.generate()` 同步返回所有 token，得用 `RequestOutput.metrics`（参考 atom 源码搜 `arrival_time / first_token_time / last_token_time`）；若 metrics 缺失，**fallback** 用同步计时 + sampling_params(max_tokens=1) 测 TTFT，再用同步计时 + max_tokens=1024 测 total_latency，TPOT = (total - TTFT) / 1023

## KNOWN_FACTS（已验证，无需重证；继承自 fp8-tp4-repro 主任务）

| ID | 事实 | 来源 |
|---|---|---|
| F1 | 三仓 commit：ATOM `acff926` / aiter `0f8164017` / CK `defd7ad29` | `MIGRATION_REPORT.md:46-50,575-590` |
| F2 | NEW-RC-1：ATOM 自动 normalize e4m3fn → e4m3fnuz；fused_moe q_dtype=`torch.float8_e4m3fnuz`（M2 log 40 处） | `MIGRATION_REPORT.md §4` |
| F3 | NEW-RC-2：weight_scale * 2.0 方向正确（forward），无需修 | `MIGRATION_REPORT.md §5` |
| F4 | NEW-RC-3：唯一源码改动 `aiter/fused_moe.py:881-886` `run_1stage = False`，per_1x128 prefill 走 CK 2-stage 而非 ASM `fmoe_g1u1` | `MIGRATION_REPORT.md §6` |
| F5 | M1（tp=2）+ M2（tp=4）已 PASS，byte-identical 143/143 | `MIGRATION_REPORT.md §10.3` |
| F6 | M2 tp=4 触发 ATOM padding：inter_dim 320 → 384（ATOM `_process_block_quant` 自动 align=KPack=32），dispatch 用 384 | `MIGRATION_REPORT.md §7` |
| F7 | hardware = 8 张 MI308X（gfx942 CDNA3），不是 40 张 | `rocm-smi --showid` + `MEMORY.md` 记录 |
| F8 | 模型 routed_experts=288, top_k=8, hidden=4096, moe_inter=1280, weight_block=[128,128] | `MIGRATION_REPORT.md §1.2` |

## TASK_SPECIFIC_VERIFICATION

- **TTFT/TPOT 必须报数值 + 来源行号**（log 行号 / 自定义脚本输出文件行号）
- **静态评估 tp=8 必须列出**：(a) inter_dim 在 tp=8 下是 1280/8=160（不是整数？需算）→ 实际 = 160；padding 到下一个 KPack=32 倍数 = 160；如果 160 不被 32 整除则触发 padding；事实上 160 % 32 = 0，**理论上不需要 padding**（与 M2 tp=4 的 320→384 不同）；这点必须由 teammate 重新核对 `atom/model_ops/moe.py:1715-1727` 的 padding 条件式
- **tp=8 实测必须确认 dispatch 路径**：log 中查 `module_moe_ck2stages_..._per_1x128_*` 是否命中（V1）+ `q_dtype=torch.float8_e4m3fnuz`（V2）+ 0 处 `no instance found`（V3）
- **GPU 资源回收**：每次 ATOM 进程结束必须 `pkill -f atom.examples` 或等待自然退出 + `rocm-smi --showmemuse` 确认 VRAM% 归零

## 阶段结构

### Phase 0（串行，约 5 min）
- [ ] #000 [验证] perf-T0 写 ttft_tpot 测量脚本骨架（仅 import atom + 最小 LLM 启动 + 测 TTFT/TPOT 的 helper），dry-run 测试小 input/output 跑通

### Phase 1（并行，最多 2 个 teammate 同时；不同 tp 共用 GPU 必须串行调度）
- [ ] #P1-A [验证] perf-T1 跑 tp=2 baseline（CUDA_VISIBLE_DEVICES=0,1）→ TTFT_tp2 / TPOT_tp2 数值
- [ ] #P1-B [验证] perf-T2 跑 tp=4 baseline（CUDA_VISIBLE_DEVICES=0,1,2,3）→ TTFT_tp4 / TPOT_tp4 数值
- [ ] #P1-C [调查] perf-T3 静态评估 tp=8 可行性（读 ATOM padding / aiter dispatch / CK 2-stage 入口；输出 tp=8 风险表 + 工作清单）

### Phase 2（串行，依赖 Phase 1 完成）
- [ ] #P2-D [验证] perf-T4 跑 tp=8 实测 baseline（CUDA_VISIBLE_DEVICES=0-7，仅验"起服 + 1 次 generate 输出语义合理"，不跑 PASS 多 V 验证）

### Phase 3（验证 + 收尾）
- [ ] #P3-W [执行] perf-T5 写 PERF_REPORT.md（数据表 + tp=8 评估清单 + mermaid 图 + reviewer 抽查指引）
- [ ] #P3-R [验证] perf-T6 critical review（数值真实度、来源行号抽查、tp=8 评估有无遗漏 ATOM padding / dispatch 风险点）

## Promotion Candidates
（待 teammate progress 中追加）

---

## 启动检查清单

- [x] WORK_DIR / DOC_DIR / LOG_DIR 已创建
- [x] CODE_ROOTS 路径列全
- [x] GOAL 一句话且可量化（数值 + 工作清单）
- [x] CONSTRAINTS 已写明红线
- [x] KNOWN_FACTS F1-F8 全部从主任务继承
- [x] BASELINE 命令模板已给（含 GPU 设置、tp、log 落点）
- [x] ENVIRONMENT 已说明 TTFT/TPOT 测量难点
- [x] 初始 todo 已写（Phase 0/1/2/3，含依赖）
- [ ] 等 lead 派 teammate
