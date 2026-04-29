# WAVE_CLOSE — wave 7-10 合并摘要（fp8 tp=8 fix promote）

> Project: fp8-tp4-repro / Step-3.5-Flash-FP8 on gfx942 (MI308X)
> 涉及 wave: fix_wave (wave 7-8) + doc_wave (wave 9) + topdoc_update_wave (wave 10)
> 全部 close 日期：2026-04-29
> 本文件位置：promote 学习存档（`step35-flash-support/18_fp8_tp8_root_cause_and_fix/WAVE_CLOSE.md`）；项目内原始 close packet 见各 wave 自身的 `WAVE_CLOSE.md`
> 红线遵守：本文件仅整理 promote 视角的摘要，不改任何源 wave 文档

---

## §0 三 wave 一图览

| wave | name | 时间 | 核心动作 | 主交付 | 关闭日期 |
|---|---|---|---|---|---|
| 7-8 | fix_wave | 2026-04-29 | 实施双层 fix + tp=2/4/8 三档 verify + reviewer 闭环 + commit `969d564` | ATOM commit `969d564`（单 commit / 31 行 / 可 revert）push 到 `feat/step3p5-flash-support` | 2026-04-29 |
| 9 | doc_wave | 2026-04-29 | 写 user-facing `TP8_ROOT_CAUSE_AND_FIX.md`（项目根，5 分钟读完）+ 顶层文档追加引用 | `TP8_ROOT_CAUSE_AND_FIX.md`（175 行 / 8 章节）+ `SESSION_HANDOFF.md` +2 / `FINAL_REPORT.md` +14 | 2026-04-29 |
| 10 | topdoc_update_wave | 2026-04-29 | `MIGRATION_REPORT.md` 扩 M3 阶段（M1/M2 0 删除）+ `PROJECT_SUMMARY.md` 顶部 deprecation header | MIGRATION_REPORT 625→781 / PROJECT_SUMMARY 306→314 | 2026-04-29 |

---

## §1 fix_wave (wave 7-8) — 真正动 ATOM 源码 + verify

### 1.1 起因

`correctness_eval/CORRECTNESS_REPORT.md` §1 §4 报 F-OPEN-1：tp=8 起服时 `atom/model_ops/moe.py:2357 _load_w2 narrow()` raise（`size < 0`），进程 crash。

### 1.2 双层根因（teammate-5 layer-by-layer dump 实验定位）

1. **K1（第一层 / crash）**：ceil-split 在 inter=1280 + tp=8 + per_1x128 量化下，D=10 / `ceil(10/8)=2` / starts=[0,2,4,6,8,10,12,14] → rank 5 命中 `start==D`（symptom B：`size=0` 后 `copy_` shape mismatch）；rank 6/7 命中 `start>D`（symptom A：`narrow size<0` raise）
2. **K2（第二层 / silent corruption）**：lead 单层 early-return 让 weight load 通过但 4/4 prompt 输出乱码（teammate-3 §2.5）；teammate-5 layer dump 证立：trailing rank 的 fp32 scale tensor `torch.ones()` 初值未被覆盖，下游 fp8 dequant `bf16 = fp8_w * 1.0` 错算，RowParallel reduction 把噪声汇进结果

### 1.3 双层 fix

```python
# _load_w13 / _load_w2 (commit 969d564, 31 行 / 单 commit / 可 revert)
if start >= loaded_weight.shape[shard_dim]:
    # zero scale slot so dequant=0 instead of multiplying by stale init=1.0
    if expert_data.dtype == torch.float32:
        # ... narrow().zero_() or full .zero_()
        ...
    return
```

详见 `TP8_ROOT_CAUSE_AND_FIX.md` §4.1（patch 全文）。

### 1.4 三档 verify A1-A4 全 PASS

| 档 | A1 weight load | A2 4 prompt 无 NaN/Inf | A3 vs corr baseline | A4 输出语义合理 |
|---|---|---|---|---|
| tp=2 | ✅ | ✅ | byte-diff = sampling noise (P2 byte-id 锚点保持) | ✅ coherent |
| tp=4 | ✅ | ✅ | 同（zero-fix 分支不触发，starts=[0,3,6,9]<10）| ✅ coherent |
| tp=8 | ✅ | ✅ | n/a | ✅ coherent，与 tp=2/4 同质量 |

### 1.5 Reviewer (teammate-6) 闭环

- 评级：2 BLOCK / 4 WARN / 10 INFO
- 2 BLOCK 全部闭环：F7（tp=2/4 用最终 patch 重 verify → teammate-8 PASS）/ F8（去 env-gate + import 提模块顶 → commit `969d564`）
- 4 WARN 接受+标注（dtype 检查、tp>D 容量损失、K2 间接证立、A4 量化补证 ROI 低）
- 10 INFO 接受入档（zero_() 语义安全 / 覆盖范围唯一 / K3 ceil 注释相容 / verify 强度）

### 1.6 关闭

**fix_wave (wave 7-8) CLOSED — 2026-04-29**

---

## §2 doc_wave (wave 9) — user-facing 文档

### 2.1 主产出

`TP8_ROOT_CAUSE_AND_FIX.md`（项目根，175 行 / 8 章节）— 让项目 owner 5 分钟读完掌握 root cause + fix + verify 证据链。

### 2.2 内部一致性 verify (teammate-verify)

A3/A4/A5 全 PASS：
- 所有数字 / 路径 / commit hash / 输出文本 byte-id 命中源文档
- 与 `fix_wave/WAVE_CLOSE.md` 内部一致（根因 / 修复 / 验证 / commit hash 全一致）
- patch 31 行 byte-id 与 `git show 969d564` 100% 一致

Findings：0 BLOCK / 2 WARN（行数 17L/24L/31L 演化备注、措辞精化）/ 3 INFO；2 WARN 接受不修。

### 2.3 顶层文档同步 (teammate-sync)

- `SESSION_HANDOFF.md` +2 行（wave 9 close 单行引用，既有 §0-§12 / wave 5/6/7 close 行 0 修改）
- `FINAL_REPORT.md` +14 行（新 §5 章节，既有 §1-§4 + References 0 修改）

### 2.4 关闭

**doc_wave (wave 9) CLOSED — 2026-04-29**

---

## §3 topdoc_update_wave (wave 10) — 顶层文档增量同步

### 3.1 主产出

- `MIGRATION_REPORT.md` 625 → 781 行（+156，全部为 §M3 章节扩入；M1/M2 既有 10 个章节 0 删除）
- `PROJECT_SUMMARY.md` 306 → 314 行（顶部 +8 行 deprecation header 指向 MIGRATION_REPORT + TP8_ROOT_CAUSE_AND_FIX；正文 0 修改）

### 3.2 验证 (teammate-verify)

A1-A7 全 PASS / 0 BLOCK / 0 WARN / 4 INFO：
- MIGRATION_REPORT grep ⚠️ = 0
- 与 `TP8_ROOT_CAUSE_AND_FIX.md` `doc_wave/WAVE_CLOSE.md` 0 冲突（9 维度对照）
- M1/M2 关键词 grep count = 61，10 个既有章节完整保留

INFO 4 条全部接受不修（含 P3 引文 1 字笔误、wave 编号重叠维度不同等 trivial 项）。

### 3.3 关闭

**topdoc_update_wave (wave 10) CLOSED — 2026-04-29**

---

## §4 文档体例传承

三 wave 共同遵守的体例约定（已固化进 `agent-team` skill 父类）：
- TEAM_CONFIG 子类实例化（PROJECT / WORK_DIR / DOC_DIR / KNOWN_FACTS / 红线 / 验收 / 子任务表）
- Phase 0 (baseline) 跳过（文档 wave 无 GPU）→ Phase 1 (实施，可并行) → Phase 2 (verify) → Phase 3 (lead 收尾决策 + 写 WAVE_CLOSE)
- 派单 prompt 极简（≤25 行 prompt 体）+ 必读上下文路径列出
- 每 teammate 在 `progress/teammate-{N}.md` 收尾 + tool calls 自计

---

## §5 promote 范围

本目录 (`18_fp8_tp8_root_cause_and_fix/`) 由 wave 11 promote_wave 的 `teammate-promote` 准备到 staging（`/tmp/project_summary_check/step35-flash-support/`），再由 lead 在 verify 后 push 到 `LJ-underdog/project_summary` 仓。

**未拷贝**（保留在源项目）：
- 各 wave 的 `TEAM_CONFIG.md` / `todo.md` / `progress/teammate-*.md`（仅 wave 内调度用，promote 后读者无需）
- `fix_wave/{logs,outputs}/` 的全部 verify 数据（GB 级 GPU log；若需要可按 `progress/SUMMARY.md` 的索引回溯到源项目）
- 其他相关 wave（correctness / perf / handoff / issue / doc_consolidation / fix_wave 的 SESSION_STATE 等）

---

**End of WAVE_CLOSE — wave 7-10 合并摘要 / 2026-04-29 / promote_wave (wave 11)**
