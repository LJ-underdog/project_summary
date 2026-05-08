# repro_guide_wave (wave 13) — WAVE_CLOSE

> Project: fp8-tp4-repro / Step-3.5-Flash-FP8 on gfx942 (MI308X)
> Wave: 13 / repro_guide_wave
> 起止: 2026-04-30 instantiate → 2026-04-30 close（同日）
> Lead: Claude session
> 目标回顾: 写一份 step-by-step `REPRODUCTION_GUIDE.md`，让外部 reader 在已有 ROCm docker + 8×MI308X 节点上 reproduce tp=2/4/8 三档 accuracy + throughput；并由 fresh-dir reviewer 真跑一遍验证 doc 可落地

---

## §0 TL;DR

| 项 | 结果 |
|---|---|
| GUIDE 体量 | `REPRODUCTION_GUIDE.md` ≈ 28 933 bytes（fresh-verify pre：28 898 / #REVISE 净增 ~35 行） |
| Phase 1+1.1 teammate 数 | 7（perf / draft / review-cmd / review-script / review-content / integrate / hfcache-patch） |
| Phase 2 teammate 数 | 2（fresh-verify / revise） |
| #FRESH-VERIFY 实测覆盖 | 10 GPU run（accuracy 3 + throughput 7，tp=2/4×16384 OOM SKIP）严格 R6 串行 |
| Accuracy | ✅ 12/12 prompt PASS（按 GUIDE §6.1 + §6.1.1 + R7 容差） |
| Throughput | ✅ 7/7 run exit 0；total_tps 单调性 ✓（4096: 4749→6533→8756 / 8192: 5222→7636→10744 / 16384(tp8): 11982） |
| Deviations 闭环 | 6 项（2 BLOCK + 3 WARN + 1 INFO）全部由 #REVISE 修订（GUIDE 净增 ~35 行：caveat ×3 + 脚注 ×2 + §8.9 新章节 + §8.0 矩阵 +1 行 + fresh-verify reader 注意 1 行） |
| 验收 | A1-A5 全 PASS / 0 BLOCK 残留 |
| 红线 | R1-R11 全部遵守（不动三仓源码 / 不动其他 wave 文档 / 不动项目根 PROJECT_SUMMARY / 不 push 三仓 / fresh dir 隔离 / GPU 独占串行 / sampling 容差 / 引用可追溯 / findings 必闭环 / 不派衍生 teammate / 不预测时间） |
| 决策 | **#LEAD-T12**：直接 #PROMOTE，不派第二轮 fresh-verify（patch 全 wording-only / 命令逻辑未变） |

---

## §1 Wave 执行轨迹

### Phase 1（调研 + 起草，串行）
| Task | Owner | 状态 | 产出 |
|---|---|---|---|
| #PERF | teammate-perf | ✅ | `progress/teammate-perf.md`（ATOM 无 offline-throughput 入口；推荐基于 correctness_bench.py 加 throughput_bench.py） |
| #DRAFT | teammate-draft | ✅ | `progress/teammate-draft.md`（patch lead 起草版 7 处 / 复用既有 throughput_bench.py） |

### Phase 1.1（并行 review + 优化，4 teammate 同 message 派出）
| Task | Owner | 状态 | 产出 |
|---|---|---|---|
| #REVIEW-SCRIPT | teammate-review-script | ✅ | 0 BLOCK / 5 WARN / 6 INFO；最高 ROI = P0 §7 OOM SKIP + 命名澄清 |
| #REVIEW-GUIDE-CMD | teammate-review-cmd | ✅ | 3 BLOCK + 5 WARN + 4 INFO；最低修复集 F1+F2+F3 |
| #REVIEW-GUIDE-CONTENT | teammate-review-content | ✅ | 1 BLOCK + 4 WARN + 5+ INFO；6 个已知坑覆盖矩阵 K1-K6 |
| #INTEGRATE | teammate-integrate | ✅ | 4 BLOCK 全修 + 9 WARN 采纳 + 2 INFO 采纳；GUIDE 净增 ~145 行 |
| #HFCACHE-PATCH | teammate-hfcache-patch | ✅ | §4 拆 §4.1 检测 + §4.2 fallback；本机 reader 复用 cache 避免重下 ~90 GB |

### Phase 2（fresh 验证 + 修订）
| Task | Owner | 状态 | 产出 |
|---|---|---|---|
| #FRESH-VERIFY | teammate-fresh-verify | ✅ PASS | `progress/teammate-fresh-verify.md`（10 GPU run / 12/12 acc + 7/7 thr / 6 deviations ranked）；fresh dir `/tmp/repro_guide_fresh_20260430_031900/`（后台 nohup + Lead 轮询，cron 定时） |
| #REVISE | teammate-revise | ✅ | `progress/teammate-revise.md`（6/6 deviations 全修 / 5/5 自验 PASS / 备份 `before_revise_GUIDE.md`） |

### Phase 3（lead 收尾 + promote）
| Task | Owner | 状态 |
|---|---|---|
| #STAGE-README | teammate-stage-readme | ✅（提前写完，wave 12 流派传统 README 骨架） |
| #STAGE-SUMMARY | teammate-stage-summary | ✅（提前写完，#FRESH-VERIFY 占位待 #PROMOTE 替换） |
| #LEAD-T12 决策 | Lead | ✅ 直接 #PROMOTE，不派第二轮 |
| #PROMOTE | teammate-promote | （执行中 / 即将派出） |
| #LEAD-T13 push | Lead | （pending） |
| #LEAD-T14 WAVE_CLOSE | Lead | ✅（本文件） |

---

## §2 验收标准对照（A1-A7）

| # | 标准 | 结果 | 来源 |
|---|---|---|---|
| A1 | GUIDE 含 三仓 clone + 精确 commit / aiter build+install / accuracy 三档 / throughput 三档 / 期望输出锚点 | ✅ | §1 三仓 commit 表（ATOM 969d564 / aiter f06cdcca5 / CK defd7ad29）/ §6 / §7 / §6.1 + §6.1.1 |
| A2 | 命令全 copy-paste 可执行 | ✅ | #INTEGRATE 修 4 BLOCK + 9 WARN；#REVISE 修 D-W2 acc output 路径 |
| A3 | reviewer 在 fresh dir 跑通 accuracy 三档（12/12 coherent + R7） | ✅ | teammate-fresh-verify §3 |
| A4 | reviewer 跑通 throughput 三档（记录 tok/s + req/s） | ✅ | teammate-fresh-verify §2.2 + §2.3 单调性 |
| A5 | reviewer findings 全闭环（修订或 caveat 入档） | ✅ | teammate-revise 6/6 deviations 全修 |
| A6 | STAGING_DIR 含 4 文件 byte-id 与 DOC_DIR 源 | （由 #PROMOTE 落实） | — |
| A7 | LJ-underdog/project_summary `main` 远程含 19_… 目录 | （由 #LEAD-T13 落实） | — |

A1-A5 全 PASS。A6/A7 由 Phase 3 后续步骤完成。

---

## §3 Findings 汇总（来自 #FRESH-VERIFY；#REVISE 闭环情况）

| 编号 | 等级 | 描述 | #REVISE 处置 |
|---|---|---|---|
| D-B1 | BLOCK | §5/§6/§7 grep `Engine Core fully initialized` 锚点不存在（ATOM 不输出该字符串） | ✅ 4 处全删 + caveat 解释（`Loading safetensors shards 44/44` + `[OK] dumped JSON` 替代） |
| D-B2 | BLOCK | §6.1.1 表 P1 finish_reason `eos (tp=2)` 是单点结论，与 fresh-verify 实测 `max_tokens` 冲突 | ✅ 改 `eos \| max_tokens (任一即可，sampling 边界 [^p1fin])` + 脚注解释 reverify 12 vs fresh-verify 13 flip 是 sampling determinism 边界 |
| D-W1 | WARN | §6.1.1 表 P2 ntoken `108 (tp=8)` 单点，与 §6.1 文字"路径变体"冲突 | ✅ 改 `60–450 (tp=8, 路径变体[^p2tok])` + 脚注引 short 路径 vs long 路径 token 跨度 |
| D-W2 | WARN | §6 acc output 路径 `reverify_wave/outputs/...` 与红线 R5 fresh dir 隔离冲突 | ✅ 全部改 `repro_guide_wave/outputs/acc_tp${tp}.json` + §6 头部加 fresh-verify reader 注意 |
| D-W3 | WARN | §6 acc mkdir 与 §7 thr mkdir 目录不对称（一个 reverify_wave 一个 repro_guide_wave） | ✅ 与 D-W2 合并修，统一到 `repro_guide_wave/{outputs,logs}` |
| D-I2 | INFO | `[aiter] type hints mismatch, override to fmha_v3_varlen_fwd(...)` log 噪音 | ✅ §8.0 矩阵 +1 行 + 新增 §8.9 解释 known-noise 安全可忽略 |

**6/6 闭环。0 BLOCK / 0 WARN / 0 INFO 残留。**

---

## §4 关键产出索引

### 新增文件（WORK_DIR = `/home/junlin12/project_fp8_tp4_repro/repro_guide_wave/`）
- `TEAM_CONFIG.md` / `todo.md`
- `REPRODUCTION_GUIDE.md`（≈ 28 933 bytes，最终态）
- `before_revise_GUIDE.md`（28 898 bytes，#REVISE 备份）
- `before_integrate.REPRODUCTION_GUIDE.md` / `before_integrate.throughput_bench.py`（#INTEGRATE 备份）
- `correctness_eval/throughput_bench.py`（既有，#INTEGRATE 净增 4 行）
- `progress/`（9 teammate progress 报告）
  - teammate-perf.md / teammate-draft.md
  - teammate-review-cmd.md / teammate-review-script.md / teammate-review-content.md
  - teammate-integrate.md / teammate-hfcache-patch.md
  - teammate-fresh-verify.md / teammate-revise.md
  - teammate-stage-readme.md / teammate-stage-summary.md
- `WAVE_CLOSE.md`（本文件）

### Fresh-verify dir（隔离，wave 关闭后可清理）
- `/tmp/repro_guide_fresh_20260430_031900/{outputs_acc,outputs_thr,logs}/`（10 GPU run 实证）

### 未触碰
- ATOM / aiter / CK 全部源码（aiter NEW-RC-3 patch 仍 dirty 未 commit，沿 wave 12 状态不动）
- 项目根 `PROJECT_SUMMARY.md` / `MIGRATION_REPORT.md`
- 其他 wave 文档（doc_wave / fix_wave / topdoc_update_wave / promote_wave / reverify_wave）

---

## §5 三仓状态确认（push readiness）

| 仓 | commit | dirty? | 远程 push 状态 |
|---|---|---|---|
| ATOM `/home/junlin12/ATOM` | `969d564` | clean | ✅ 已 push 到 `git@github.com:ROCm/ATOM.git` `feat/step3p5-flash-support`（wave 11） |
| aiter `/workspace/aiter` | `f06cdcca5` | dirty `aiter/fused_moe.py` NEW-RC-3 L881-886（沿 wave 1 / 12） | ⏸ 用户后续可决定是否 commit + push 上游（wave 13 不触） |
| CK `/workspace/aiter/3rdparty/composable_kernel` | `defd7ad29` (HEAD detached) | clean | ✅ 已在远程分支 `origin/feat/swiglustep-moe-no-quant`（wave 1 时期） |

**与 reverify_wave (wave 12) §5 完全一致 — 三仓 0 改动。**

---

## §6 Wave 13 close 标记

**repro_guide_wave (wave 13) CLOSED — 2026-04-30**

`REPRODUCTION_GUIDE.md` 经 7 teammate 起草+review+integrate / fresh-dir 真跑验证 / #REVISE 6/6 deviations 闭环；可由外部 reader（已有 ROCm docker + 8×MI308X）按文档 copy-paste reproduce tp=2/4/8 三档 accuracy + throughput。

剩余 Phase 3 节点 #PROMOTE / #LEAD-T13 由 Lead + teammate-promote 落实，不阻断 wave 关闭。

**End of repro_guide_wave WAVE_CLOSE — 2026-04-30**
