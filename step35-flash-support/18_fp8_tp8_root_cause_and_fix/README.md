# 18_fp8_tp8_root_cause_and_fix — Step-3.5-Flash-FP8 在 tp=8 起服的 root cause + 双层 fix

> 日期：2026-04-29
> 来源 wave：fp8-tp4-repro / fix_wave (wave 7-8) + doc_wave (wave 9) + topdoc_update_wave (wave 10) + promote_wave (wave 11)
> 状态：✅ 上游 fix 已 push（ATOM commit `969d564` on `feat/step3p5-flash-support`）；本目录是 promote 学习存档
> 红线遵守：本目录仅整理文档（无源码改写、无 GPU 跑、无其他 wave 文档改）

---

## 1. 这是什么

`stepfun-ai/Step-3.5-Flash-FP8` 在 MI308X (gfx942) 上 `--tp 8` 起服时 weight load 阶段确定性崩溃，绕开 crash 后 4/4 prompt 输出乱码 — 本 topic 是该问题的 **root cause + 双层修复方案 + 三档（tp=2/4/8）验证证据** 的 user-facing 完整说明。

**双层根因**：
1. **第一层（crash）**：`atom/model_ops/moe.py` `_load_w13` / `_load_w2` 用 `ceil(D/tp_size)` 切 inter 维度，trailing rank 上 `start >= D` 让 `narrow(start, size)` 拿到 `size <= 0` → `_load_w2 narrow()` raise，进程 crash
2. **第二层（silent corruption）**：仅 early-return 跳过 narrow 后 weight load 通过，但 trailing rank 的 fp32 scale tensor 保持 `torch.ones()` 初值 1.0 → fp8 dequant `bf16 = fp8_w * 1.0` 把未拷贝的 fp8 raw bits 当 bf16 用 → RowParallel reduction 累积错算 → 4/4 prompt 完全乱码

**双层 fix**（ATOM commit `969d564`，单 commit / 31 行 / 可 revert）：
- (a) `start >= D` 时 early-return 跳过 `narrow + copy_`（避开 crash）
- (b) 同分支内对 `dtype == torch.float32` 的 scale tensor 调用 `.zero_()` 让 trailing rank 在 RowParallel reduction 中贡献严格 0（修 silent corruption）

**验证**：tp=2/4/8 三档 A1-A4 全 PASS，tp=8 4/4 prompt 与 tp=2/4 baseline 同质量 coherent；tp=2/4 zero-fix 分支不触发（D=10，starts 全 < D），无回归。

---

## 2. 文件清单

| 文件 | 用途 | 行数 |
|---|---|---|
| `TP8_ROOT_CAUSE_AND_FIX.md` | **主交付**（user-facing，5 分钟读完掌握 root cause + fix + verify 证据链）| 216 |
| `WAVE_CLOSE.md` | wave 7-10 四 wave 关键交付合并摘要 |  -  |
| `progress/SUMMARY.md` | 各 wave key teammate 一行摘要（不拷全部 progress 文件，仅索引）|  -  |
| `README.md` | 本文件 |  -  |

---

## 3. 与 17_atom_moe_tp8_load_crash 的时序关系

| topic | wave | 阶段 | 状态 |
|---|---|---|---|
| `17_atom_moe_tp8_load_crash/` | issue_wave (wave 6) | issue draft（仅描述 crash 现象 + 推测修复 A/B）| ⏸ 未 file 到 ATOM upstream / 未实施 fix |
| `18_fp8_tp8_root_cause_and_fix/` (本 topic) | fix_wave (wave 7-8) + doc_wave (wave 9) + topdoc_update_wave (wave 10) | 落实 fix（实测双层根因 + 双层 fix + 三档 verify + 顶层文档同步）| ✅ 上游 commit `969d564` 已 push |

17 是问题描述 / 草稿；18 是真正修复 + 验证 + 文档化。读者推荐顺序：先读 18（已有完整解法）；如需原始 issue draft 措辞或方案 A/B/C 对照，再回读 17。

17 的 issue draft 推测了方案 A（early-return）与方案 B（余数挂 rank0）；18 实测发现仅 A 不够（A1/A2 PASS 但 A4 4/4 prompt 乱码 — 第二层 silent corruption），最终采纳 **A + scale.zero_() 双层 fix**。

---

## 4. 上游 commit

- **ATOM commit `969d564`** on branch `feat/step3p5-flash-support`（已 push）
- 单 commit / 31 行 / 可 revert
- 文件：`atom/model_ops/moe.py`（`_load_w13` + `_load_w2` 各一处 hunk）
- base commit：`acff926` "fix(moe): correct FP8 blockscale inter_dim padding align for all tp configs"
- patch 全文：见 `TP8_ROOT_CAUSE_AND_FIX.md` §4.1

---

## 5. 引用源（项目内文档原始路径，仅 promote 学习用）

| 源文档 | 来源 wave | 用途 |
|---|---|---|
| `TP8_ROOT_CAUSE_AND_FIX.md` | doc_wave (wave 9) | user-facing 主交付（本 topic 同名文件即从此处 byte-id 拷贝） |
| `fix_wave/WAVE_CLOSE.md` | fix_wave (wave 7-8) | 实际 fix 工作 + reviewer F7/F8 BLOCK 闭环 + commit `969d564` |
| `fix_wave/progress/teammate-5.md` | fix_wave (wave 7) | layer-level dump 实验定位 K1（trailing rank scale init=1.0）|
| `fix_wave/progress/teammate-8.md` | fix_wave (wave 8) | tp=2/4/8 三档 final verify A1-A4 全 PASS |
| `doc_wave/WAVE_CLOSE.md` | doc_wave (wave 9) | user-facing 文档撰写（A3/A4/A5 PASS）|
| `topdoc_update_wave/WAVE_CLOSE.md` | topdoc_update_wave (wave 10) | 顶层 MIGRATION_REPORT 扩 M3 + PROJECT_SUMMARY deprecation header |
| `correctness_eval/CORRECTNESS_REPORT.md` §1 §4 | correctness wave | F-OPEN-1 原始证据 |
| `issue_wave/ATOM_ISSUE_DRAFT.md` | issue_wave (wave 6) | 方案 A/B/C 原始定义（即 17 topic）|

---

**End of README.**
