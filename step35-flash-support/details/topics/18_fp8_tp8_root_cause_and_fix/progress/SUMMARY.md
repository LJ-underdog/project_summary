# progress/SUMMARY.md — wave 7-10 各 key teammate 一行摘要

> 本文件仅是 **索引**，不复制 progress 全文。
> 若需具体 teammate 报告，请回查源项目对应 wave 的 `progress/teammate-{N}.md`。
> 源项目根：`/home/junlin12/project_fp8_tp4_repro/`（promote 时点；后续可能迁移）

---

## fix_wave (wave 7-8) — `fix_wave/progress/`

| teammate | 角色 | 一行摘要 |
|---|---|---|
| teammate-3 | 实验主线 | env 诊断 + tp=2/4/8 attempt2 三档 verify + A1-A5 判定（lead 17L 单层 patch；A1/A2 PASS / A4 FAIL = tp=8 4/4 prompt 乱码） |
| teammate-4 | 独立交叉验证 | 验证 teammate-3 的 tp=4 A3 fail 归因（V1 代码层硬证 zero-trigger / V2 fd 量级一致 / V3 tp=4 unpatched 直接对照缺失，标 indirect gap） |
| teammate-5 | 根因实验 | layer-level dump 定位 K1（trailing rank fp32 scale `torch.ones()` 初值未被覆盖）+ zero-fix 实测：4/4 prompt 从乱码恢复 coherent |
| teammate-6 | reviewer | 2 BLOCK (F7/F8) / 4 WARN / 10 INFO；主体结论"zero-fix 接受"；F7/F8 双 BLOCK 闭环路径 reviewer 已设计 |
| teammate-7 | 文档员 | 写 `fix_wave/WAVE_CLOSE.md`（wave 7 close packet，350 行；A1-A5 验收对照 + reviewer findings 处理表）|
| teammate-8 | final verify | 三档（tp=2/4/8）跑最终 24L patch（去 env-gate）= commit `969d564` 的 final_tp{2,4,8}.{log,json}，A1-A4 全 PASS，闭环 F7 BLOCK |
| teammate-9 | wave 8 close 收尾 | 写 wave 8 close 增量到 `fix_wave/WAVE_CLOSE.md` §0 末尾（CLOSED PENDING REVIEWER → CLOSED）|
| teammate-analyze-tp8 | attempt1 根因 | tp=8 attempt1 disk-full 根因：HF_HOME 未透传到 ModelRunner 子进程，8 rank 并发重下 209GB 到 13GB 自由的 sda3 → exit 120 |

---

## doc_wave (wave 9) — `doc_wave/progress/`

| teammate | 角色 | 一行摘要 |
|---|---|---|
| teammate-doc | 主交付撰写 | 写 user-facing `TP8_ROOT_CAUSE_AND_FIX.md`（项目根，175 行 / 8 章节，5 分钟读完掌握 root cause + fix + verify 证据链；0 ⚠️ 引用待补）|
| teammate-verify | 内部一致性 verify | A3/A4/A5 全 PASS（0 BLOCK / 2 WARN / 3 INFO）；patch 31 行 byte-id 与 `git show 969d564` 100% 一致；行号 sed 实测命中 |
| teammate-sync | 顶层文档 sync | `SESSION_HANDOFF.md` +2 行（wave 9 close 引用）+ `FINAL_REPORT.md` +14 行（新 §5），既有内容 0 修改 |

---

## topdoc_update_wave (wave 10) — `topdoc_update_wave/progress/`

| teammate | 角色 | 一行摘要 |
|---|---|---|
| teammate-mig | MIGRATION_REPORT 扩展 | 把 `MIGRATION_REPORT.md` 从 M1/M2 双阶段扩为 M1/M2/M3 三阶段（625→781，+156 全是 M3 章节，M1/M2 既有 10 章节 0 删除）|
| teammate-ps | PROJECT_SUMMARY 加 header | `PROJECT_SUMMARY.md` 顶部 +8 行 deprecation header 指向 MIGRATION_REPORT + TP8_ROOT_CAUSE_AND_FIX；正文 0 修改（306→314）|
| teammate-verify | 顶层 verify | A1-A7 全 PASS / 0 BLOCK / 0 WARN / 4 INFO；MIGRATION_REPORT grep ⚠️ = 0；与 TP8_ROOT_CAUSE_AND_FIX 0 冲突（9 维度对照）|

---

## promote_wave (wave 11) — `promote_wave/progress/`

| teammate | 角色 | 一行摘要 |
|---|---|---|
| teammate-promote | promote 学习 | 在 staging (`/tmp/project_summary_check/step35-flash-support/18_fp8_tp8_root_cause_and_fix/`) 准备 README + TP8_ROOT_CAUSE_AND_FIX (byte-id 拷贝) + WAVE_CLOSE + progress/SUMMARY，git commit 但不 push（红线 R5 / lead 后续亲自 push）|

---

**End of SUMMARY — 2026-04-29 / wave 11 / teammate-promote**
