# TEAM_CONFIG — issue_wave

> Wave 编号：fp8-tp4-repro / wave 6 / issue_wave
> 起始日期：2026-04-29
> Lead：Claude session（本会话）
> 目标：把 corr wave F-OPEN-1（tp=8 `_load_w2` narrow size<0 block bug）打包成 ATOM 上游 issue 草稿，使下游可直接 copy-paste 到 GitHub issue
> 产物：`issue_wave/ATOM_ISSUE_DRAFT.md`

---

## 1. 任务定义

| 项 | 值 |
|---|---|
| Task | 写 ATOM 上游 issue 草稿（不实施 fix；不开 PR；不修源码） |
| 来源 | `handoff_wave/HANDOFF_PACKET.md` §6 P0 |
| 主证据 | `correctness_eval/CORRECTNESS_REPORT.md` §4 + `correctness_eval/progress/corr-t1.md:50-80` + `~/ATOM/atom/model_ops/moe.py:2335-2364` |
| 三仓 commit（不变）| ATOM `acff926` / aiter `0f8164017` / CK `defd7ad29` |
| 模型 | `stepfun-ai/Step-3.5-Flash-FP8` |
| 触发场景 | `python -m atom.entrypoints.openai_server -tp 8 ...` weight load 阶段 |

---

## 2. 红线（写 prompt 时复制给 teammate）

| # | 红线 |
|---|---|
| R1 | 不修任何源码（ATOM / aiter / CK 任何文件都不动）|
| R2 | 不修任何已有 .md（顶层 4 doc + 各 wave 主报告 + 各 progress/ 全 frozen；本 wave 新产出只 Write 在 `issue_wave/` 内）|
| R3 | 不动任何 PASS 判定（M1/M2 PASS、perf wave PASS、corr wave finding 严重度判定）|
| R4 | issue 草稿中"推测 fix"段必须明确标注 advisory（不开 PR / 等 ATOM 上游决策）|
| R5 | issue 草稿引用代码必须 file_path:line_number 形式 + 引用本 wave 之前的 .md 必须 path 正确 |

---

## 3. Wave 子 task 顺序

| # | task | owner | status |
|---|---|---|---|
| T0 | Lead 收齐证据（read CORRECTNESS_REPORT §4 / corr-t1.md / moe.py:2335-2364）| Lead | ✅ done（本 session 已完成）|
| T1 | Lead 写 `TEAM_CONFIG.md`（本文件）| Lead | ✅ done |
| T2 | Lead 写 `ATOM_ISSUE_DRAFT.md` | Lead | 🟡 in_progress |
| T3 | 派 reviewer（critical review，不修复 finding，只 raise）| Teammate (issue-reviewer) | pending |
| T4 | Lead 综合 reviewer finding，决策 (a) 接受标注 / (b) 修订 draft / (c) 派补证据 task | Lead | pending |
| T5 | Lead 写 `WAVE_CLOSE.md`（本 wave 的 mini-handoff，引到 HANDOFF_PACKET 下一版可选）| Lead | pending |

---

## 4. ATOM_ISSUE_DRAFT.md 的目标 ToC（issue 草稿模板）

1. Title（短，标识 model + tp + 崩溃点）
2. Summary（3-5 句，含 deterministic 标记）
3. Environment（commit / model / hardware / launch cmd）
4. Reproduction（4 prompts + tp=8 + cudagraph_capture_sizes=[1,2,4]）
5. Traceback（完整 stack，含 rank5 与 rank6/7 双 symptom）
6. Root cause analysis（数学推断：D mod tp_size 边界条件）
7. Affected configurations（哪些 D / tp_size 组合会触发；back-of-envelope D≈10 估算）
8. Proposed fix - advisory only（option A early-return / option B even-split + remainder to rank0；指明需 ATOM 上游决策；附与 `_load_w13` 对齐的考量）
9. References（本 wave 内的 path + 三仓 commit + 上游可读的 corr-t1.md / CORRECTNESS_REPORT.md）

---

## 5. 验收标准（Lead T4 决策时用）

| # | 标准 | 来源 |
|---|---|---|
| A1 | 任何打开 issue 草稿的 ATOM upstream maintainer 能在 5 min 内：定位崩溃文件:行号 + 复述触发条件 + 看到 advisory fix（不需要他读 corr wave 全部 .md） | wave 目标 |
| A2 | 数学条件用纯文字 + 单一公式表达（不要求读者跑代码）| 上游可读性 |
| A3 | 推测 fix 至少 2 个 option 并列；明确标注未实施 + 等上游决策 | R4 |
| A4 | reviewer 不能找出任何 (a) miscite (b) 数学错误 (c) traceback 与代码不对齐 的 high finding；warn 可接受 | wave 闭环 |
| A5 | draft 长度上限 ~250 行（issue 体裁要求紧凑；超 300 视为冗长，需 Lead 砍）| 体裁 |

---

**TEAM_CONFIG 至此 frozen。下一步 T2 = Lead 写 `ATOM_ISSUE_DRAFT.md`。**
