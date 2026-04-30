# TEAM_CONFIG — fp8-tp4-repro 文档整合任务

> 子任务名：`doc_consolidation`
> 父项目：`/home/junlin12/project_fp8_tp4_repro`（PROJECT CLOSED，详见 `SESSION_HANDOFF.md:329`）
> 任务性质：纯文档整合 + 可视化重写，不涉及任何源码改动 / GPU 实验
> 起始时间：2026-04-29

---

## PROJECT

将散落在 `project_fp8_tp4_repro/` 下的多份文档（4 份根级主文档 + 18 份 teammate progress + docs/ 已有素材）整合为**一份完善的最终迁移报告**，主题：

> **「Step-3.5-Flash-FP8 模型从 gfx950 迁移到 gfx942(MI308X) 的问题排查与解决」**

最终交付：`/home/junlin12/project_fp8_tp4_repro/MIGRATION_REPORT.md`（与 SESSION_HANDOFF.md 平级）

---

## WORK_DIR / DOC_DIR / LOG_DIR

| 名 | 路径 |
|---|---|
| WORK_DIR | `/home/junlin12/project_fp8_tp4_repro/doc_consolidation/` |
| DOC_DIR | `/home/junlin12/project_fp8_tp4_repro/`（最终报告写在项目根） |
| LOG_DIR | （本任务无 GPU 实验，无 log）|

---

## CODE_ROOTS（仅供 reviewer 验证 file:line 时查阅）

| 仓 | 路径 | HEAD | 备注 |
|---|---|---|---|
| ATOM | `/home/junlin12/ATOM` | `acff926` | 唯一 dirty 已闭环 |
| aiter | `/workspace/aiter` | `0f8164017` | 含 NEW-RC-3 patch |
| CK | `/workspace/aiter/3rdparty/composable_kernel` | `defd7ad29` | swiglustep_and_mul 分支 |

来源：`progress/teammate-20.md:38-43`

---

## GOAL（一句话）

产出一份**面向新读者的、step-by-step、图表丰富的**迁移报告，让读者无需翻 18 份 progress 也能完整理解：(a) 迁移过程中遇到了哪些根因级问题；(b) 每个问题如何被发现、定位、修复；(c) 当前 PASS 状态与证据链。

---

## CONSTRAINTS（红线）

| 约束 | 来源 |
|---|---|
| 不修改任何已有文档（仅 Read，仅 Write 新文件） | 用户原始 prompt + `progress/teammate-20.md:7` |
| 不修改源码 | 同上 |
| 不跑任何 GPU 命令 | 同上 |
| 不动 PASS 判定（M1 / M2 / NEW-RC-1/2/3 闭环） | `progress/teammate-20.md:206-215` §5.2 |
| 错误的尝试可省略，只保留有效证据链 | 用户原始 prompt |
| 必须有独立 reviewer teammate 验证 file:line + 技术结论 | 用户原始 prompt |
| 大量使用 mermaid 图表/流程图/表格 | 用户原始 prompt |
| 中文输出 | `~/.claude/CLAUDE.md` 第 1 条 |
| 所有引用用 `file:line` 格式 | 项目惯例 |

---

## ENVIRONMENT（teammate 工作环境）

本任务仅涉及文件读写，无 GPU/Python/编译/网络依赖。所有 teammate 启动时需知：

```
- 工作目录：/home/junlin12/project_fp8_tp4_repro/
- 仅用 Read / Grep / Glob / Write，不用 Bash 跑业务命令
- 所有源文档不可修改（包括 typo 修复都不动）
- 输出统一中文，引用统一 file:line 格式
```

---

## KNOWN_FACTS（已验证事实，无需重验）

> 所有事实直接引自 `progress/teammate-20.md` 的 §1（环境快照）+ §5（禁区清单）。teammate **不需要重新验证**这些事实，只需在引用时附 file:line。

### F1 硬件 + 模型
- 架构：gfx942（CDNA3）/ MI308X / 40 张卡 → `progress/teammate-20.md:17-19`
- FP8 numeric format：e4m3fnuz（NaN=0x80）→ `progress/teammate-20.md:22`
- 模型：`stepfun-ai/Step-3.5-Flash-FP8`，hidden=4096, moe_inter=1280, n_experts=288, top_k=8, weight_block=[128,128] → `progress/teammate-20.md:49,54`

### F2 三仓 commit
- ATOM `acff926` / aiter `0f8164017` / CK `defd7ad29` → `progress/teammate-20.md:38-43`

### F3 PASS 判定（不可重新打开）
- M1 PASS（tp=2）→ `progress/teammate-5.md:189-191` + `FINAL_REPORT.md:15-31`
- M2 PASS（tp=4）→ `progress/teammate-12.md:217-219` + `FINAL_REPORT.md:33-50`

### F4 三大 root cause（NEW-RC-1/2/3）
| RC | 一句话 | 闭环证据 | 引用 |
|---|---|---|---|
| **NEW-RC-1** | gfx942 必须用 e4m3fnuz（NaN=0x80），原 gfx950 路径用 e4m3fn | aiter `dtypes.py` 静态映射 + M2 log 40 处 q_dtype=float8_e4m3fnuz | `progress/teammate-20.md:203,211` |
| **NEW-RC-2** | weight_scale 需要 `* 2.0` 而不是 `/ 2.0` | T-6 静态 fp32 dequant absmax forward=0.093 vs inverse=2.15e6（7 数量级差距）+ M1+M2 PASS 反证 | `progress/teammate-20.md:202,212` + `progress/teammate-6.md:138-146` |
| **NEW-RC-3** | per_1x128 prefill 需 bypass ASM `fmoe_g1u1` 路径，强制走 CK 2-stage | M1+M2 0 处 `fmoe_g1u1`，全部 `module_moe_ck2stages_..._per_1x128_*Stage2` | `progress/teammate-20.md:199,213` + `progress/teammate-5.md:67-84` |

### F5 关键修复点（不可改动）
- `aiter/fused_moe.py:881-886`（NEW-RC-3 唯一允许的 dirty）
- `atom/model_ops/moe.py:1709-1746`（M2 padding inter_dim 320→384）
- `atom/model_ops/utils.py:79`（NEW-RC-2 `weight_scale * 2.0`）
- `aiter/utility/dtypes.py:10-25`（NEW-RC-1 gfx942 → e4m3fnuz 映射）

来源：`progress/teammate-20.md:199-204`

### F6 已 byte-identical（不要重跑试图证伪）
- M1↔M2 byte-for-byte 完全一致 143/143 chars（P3 prompt "1+2+3=?"）→ `progress/teammate-16.md:44-49`

---

## TASK_SPECIFIC_VERIFICATION（teammate 验证注意）

### Reviewer 必须做的引用真实性检查

每条 file:line 引用需满足：
1. **文件存在**（用 Read 验证）
2. **行号附近内容与报告中陈述匹配**（不能 cite 一个无关行）
3. **若引用 progress 文件，需追溯到底层证据**（progress 也是二手；底层是 source code 行号 / GPU log / 实验数值）

### 必须使用的 mermaid 图表种类（强制）

报告至少包含以下可视化：
1. **总体迁移流程图**（gfx950 → gfx942 的从 0 到 PASS 的流程）— `flowchart TD`
2. **三大 root cause 关系图**（NEW-RC-1/2/3 之间是否独立、触发顺序）— `graph LR` 或 `flowchart`
3. **每个 RC 的「症状 → 调查 → 定位 → 修复」时序图**（至少 1 个用 `sequenceDiagram`，其他可用 `flowchart`）
4. **dispatch 路径对比表**（gfx950 vs gfx942 各 op 的路由差异）— markdown 表格 + 可选 `graph`
5. **三仓改动 summary 表**（ATOM / aiter / CK 各改了什么、为什么）— markdown 表格

---

## BASELINE（本任务无 GPU baseline，仅文档基线）

- 起始素材：`SESSION_HANDOFF.md` / `FINAL_REPORT.md` / `PROJECT_SUMMARY.md` / `TEAM_CONFIG.md` / `M1_BASELINE_DISPATCH_PLAN.md` + `progress/teammate-{1..20}.md` + `docs/gfx950_to_gfx942_migration.md`
- 起始问题：散落在 ≥22 份文档里，新读者无法快速理解迁移过程
- 验收标准：reviewer 在最终报告中找到所有 NEW-RC-1/2/3 的 root cause + step-by-step 修复路径，且至少抽查 5 处 file:line 引用全部真实

---

## 团队分工（Phase 1 并行 + Phase 2 串行 + Phase 3 验证）

| Teammate | Phase | 职责 |
|---|---|---|
| **DC-T1（doc-survey）** | Phase 1 | 通读 4 份根级主文档（SESSION_HANDOFF / FINAL_REPORT / PROJECT_SUMMARY / TEAM_CONFIG），输出「权威结论 + 证据 file:line」清单 |
| **DC-T2（progress-mining）** | Phase 1 | 通读 18 份 teammate progress，按时间线提取每个 RC 的「发现 → 调查 → 定位 → 修复」事件流（错误尝试省略） |
| **DC-T3（writer）** | Phase 2 | 基于 DC-T1 + DC-T2 产出，撰写 `MIGRATION_REPORT.md`（含 mermaid 图表，按 §TASK_SPECIFIC_VERIFICATION 强制清单） |
| **DC-T4（reviewer）** | Phase 3 | 独立 critical review：file:line 抽查 ≥10 条 + 技术结论与 KNOWN_FACTS 一致性 + mermaid 渲染语法 + 大纲完整性 |

依赖关系：`DC-T1 ‖ DC-T2 → DC-T3 → DC-T4`

---

## Promotion Candidates

（任务结束后填写）

---

## 引用文档清单（teammate 工作时按需 Read）

### 4 份根级主文档
- `/home/junlin12/project_fp8_tp4_repro/SESSION_HANDOFF.md`
- `/home/junlin12/project_fp8_tp4_repro/FINAL_REPORT.md`
- `/home/junlin12/project_fp8_tp4_repro/PROJECT_SUMMARY.md`
- `/home/junlin12/project_fp8_tp4_repro/TEAM_CONFIG.md`
- `/home/junlin12/project_fp8_tp4_repro/M1_BASELINE_DISPATCH_PLAN.md`

### 18 份 teammate progress
- `/home/junlin12/project_fp8_tp4_repro/progress/teammate-{1..20}.md`（缺 8、15）

### docs/ 已有素材
- `/home/junlin12/project_fp8_tp4_repro/docs/gfx950_to_gfx942_migration.md`（早期草稿，可参考结构）
- `/home/junlin12/project_fp8_tp4_repro/docs/lead_progress.md`
- `/home/junlin12/project_fp8_tp4_repro/docs/baseline_tp{2,4}_result{,_mt512}.md`（实测 log，仅按需引用片段）
