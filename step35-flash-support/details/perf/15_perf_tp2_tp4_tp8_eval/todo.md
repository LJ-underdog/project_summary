# TODO — perf_tp_eval

> **🔴 WAVE CLOSED 状态已被 audit 翻转（2026-05-09 by `tp2_verify_post_merge_wave` / L17c+L19b+L19d+L24+L28）**
>
> 本 wave 顶部宣称"WAVE CLOSED"+ 下方 Phase 1/4 ✓ closed bullet + 末尾"关键数字"表展示的 tp=2 / tp=4 / tp=8 long / tp=8 起服 baseline 数值（TTFT/TPOT/total/decode_thru/engine_init 等）**全部为 `Qwen/Qwen3-0.6B`（dense, non-MoE）数据，非任务目标 `stepfun-ai/Step-3.5-Flash-FP8`**。raw log 实测：
> - `logs/tp2_run2_full.log:47,50` → `Model load done: Qwen/Qwen3-0.6B`
> - `logs/tp4_run2_full.log:79,81,84,86` → `Model load done: Qwen/Qwen3-0.6B`（8/8 行一致）
> - `logs/tp8_long_run2_full.log:144,146,148,150,152,154` → `Model load done: Qwen/Qwen3-0.6B`（12/12 行一致）
> - `logs/tp8_launch_full.log:144,146,148,151,153,155` → `Model load done: Qwen/Qwen3-0.6B`（6/6 行一致）
>
> 根因：本 wave 启动命令模板**全部漏写 `--model` 参数**（实际命令显式传 `--model Qwen/Qwen3-0.6B`，但脚本 docstring + 文档命令均未记录），ATOM EngineArgs `--model` default 抢先生效（详见 `step35-flash-support/REPRODUCE.md §7.13` KNOWN_FACT）。
>
> 影响：本 wave **未产生**任何 stepfun-Flash-FP8 gfx942 perf baseline；**首次** stepfun gfx942 实测见 `tp2_verify_post_merge_wave/progress/teammate-L18-perf-rerun.md`（tp=2，TTFT≈1665.1ms / TPOT≈15.5ms）+ `teammate-L20-perf-tp4-tp8.md`（tp=4：TTFT≈980.4ms / TPOT≈14.5ms；tp=8 long-prompt 同样产出，详见 progress 表）。
>
> 下方 Phase 1/4 ✓ closed bullet + 关键数字表保留作 historical reference（不删除原结论文字），已用 strikethrough 标记 + ⚠️ 备注 model 误归属。

> ~~**WAVE CLOSED** — 2026-04-29~~（已被 audit 翻转，见上方 🔴 banner）
> 最终交付：`/home/junlin12/project_fp8_tp4_repro/perf_tp_eval/PERF_REPORT.md`（587 行 / 17 表格 / 2 mermaid，A 评级）— ⚠️ 数值已被 L17c/L19b/L19d 翻转 model 归属

## 规范
- [~] 进行中
- [x] 完成（附结论一行）
- [!] 卡住
- [ ] Pending

## Phase 0
- [x] #000 @perf-T0 写 ttft_tpot 测量脚本骨架 + dry-run → `perf_bench.py` + `progress/perf-t0.md`，方案 A，dry-run TTFT=0.038s 通过

## Phase 1（并行）
- [x] ⚠️ #P1-A @perf-T1 tp=2 baseline → ~~TTFT=0.186s / TPOT=5.245ms·tok / 10265→317(eos)，V3=0/W=0 + JIT cache 间接证据 ✓~~ — 🔴 实为 Qwen/Qwen3-0.6B 数据（非 stepfun），详见 PERF_REPORT.md 顶部 🔴 块 + perf-t1.md 附录 X；stepfun 真 baseline 见 `tp2_verify_post_merge_wave/progress/teammate-L18-perf-rerun.md`
- [x] ⚠️ #P1-B @perf-T2 tp=4 baseline → ~~TTFT=0.110s / TPOT=5.451ms·tok / 10265→416(eos)，V3=0/W=0 + JIT cache ✓~~ — 🔴 实为 Qwen/Qwen3-0.6B 数据（L19d audit 100% 命中：tp4_run2_full.log:79 等 8/8 行 `Model load done: Qwen/Qwen3-0.6B`）；详见 perf-t2.md 顶部 🔴 块；stepfun 真 baseline 见 `tp2_verify_post_merge_wave/progress/teammate-L20-perf-tp4-tp8.md`
- [x] #P1-C @perf-T3 tp=8 静态评估 → 预测 inter_per_rank=160→padding 256；probably PASS（perf-T3 是纯静态分析，不引 raw log，无 model 归属问题；详见 perf-t3.md §13 ✅ audit 通过节）

## Phase 2
- [x] ⚠️ #P2-D @perf-T4 tp=8 实测起服 → ~~PASS（engine_init=45.82s，1 次 generate=64 tok 完整退出，path 与预测一致）~~ — 🔴 实为 Qwen/Qwen3-0.6B 起服（L19d audit：tp8_launch_full.log:144 等 6/6 行 `Model load done: Qwen/Qwen3-0.6B`）；详见 perf-t4.md 顶部 🔴 块

## Phase 3
- [x] #P3-W @perf-T5 写 PERF_REPORT.md → 587 行 / 17 表格 / 2 mermaid / 11 章 + 附录 A/B（⚠️ 主表 tp=2/4/8 行数值已被 model 归属翻转，详见 PERF_REPORT.md 顶部 🔴 块；perf-t5.md 顶部已加 historical disclaimer note）
- [x] #P3-R @perf-T6 critical review → ~~A 评级，0 block / 1 warn / 7 info；18/18 数值真实可追溯~~ — ⚠️ reviewer ✓ 验证仅 audit "数值与 raw log 一致"，**未审 model 归属**，L17c/L19d 翻转后该 ✓ 标签需作 partial PASS 解读（详见 perf-t6-review.md 顶部 historical disclaimer note）
- [x] #P3-F lead 应用 F-1 / F-2 行号修订（2 处 Edit）→ inline 修订完成

## Phase 4（P1 补完，post-close 增量 wave）
- [x] ⚠️ #P4-7 @perf-T7 tp=8 long-prompt perf → ~~TTFT=0.071s / TPOT=5.542ms·tok / total=1.629s / decode_thru=180.43tok/s / 10265→282(eos)，V3=0/W=0(实质) + JIT cache ck2stages-only ✓~~ — 🔴 实为 Qwen/Qwen3-0.6B 数据（L19d audit 100% 命中：tp8_long_run2_full.log:144 等 12/12 行 `Model load done: Qwen/Qwen3-0.6B`）；详见 perf-t7.md 顶部 🔴 块；stepfun 真 baseline 见 `tp2_verify_post_merge_wave/progress/teammate-L20-perf-tp4-tp8.md`
- [x] #P4-U lead 应用 perf-T7 数据到 PERF_REPORT.md（TL;DR 数字表+§5.3 对比表+§7 P1 closed+§8 引用增）→ 6 处 Edit（⚠️ 同 #P4-7：写入数据 model 归属误标）

## In Progress
（无）

## Done
- #000 / #P1-A / #P1-B / #P1-C / #P2-D / #P3-W / #P3-R / #P3-F / #P4-7 / #P4-U

## Blocked
（无）

---

## 闭环证据链

| 阶段 | 证据 |
|---|---|
| Phase 1 完整性 | tp=2 / tp=4 stable 数据 + tp=8 静态预测三仓共闭环 |
| 报告大纲 | TL;DR + §1-§7 + §8 引用 + 附录 A/B + 红线自查（perf-T6 R5 ✓）|
| Mermaid 完整性 | 2/2 强制图齐全且语法合法（perf-T6 R3 ✓）|
| 数值真实度 | 18/18 核心数值抽查（perf-T6 R1 ✓）|
| 引用真实度 | 8/8 file:line 抽查（6 精确 + 2 行号偏差已修，perf-T6 R4）|
| 逻辑一致性 | tp=8 不可比性 4 处独立警告（perf-T6 R2）|

---

## 关键交付

| 产物 | 路径 | 行数 |
|---|---|---|
| 性能报告 | `PERF_REPORT.md` | 587 |
| 测量脚本 | `perf_bench.py` | ~200 |
| 数据 logs | `logs/{tp2,tp4,tp8}_*.log` | 7 文件 |
| Progress | `progress/perf-t{0,1,2,3,4,5,6-review}.md` | 7 文件 |

---

## 关键数字

> **🔴 BASELINE 误归属修正（2026-05-09）**：下表 tp=2 / tp=4 / tp=8 long / tp=8 起服 4 行**全部为 `Qwen/Qwen3-0.6B`（dense, non-MoE）数据，非 stepfun-Flash-FP8**（已由 L17c/L19d audit raw log `Model load done:` 字段 100% 实证）。原行保留作 historical reference + strikethrough；实际 stepfun gfx942 perf anchor 见 `step35-flash-support/REPRODUCE.md §6.2`（数据来自 `tp2_verify_post_merge_wave` L18+L20 实测）。

| tp | TTFT | TPOT | total | decode_thru | input/output | engine_init | model 实测 |
|---|---|---|---|---|---|---|---|
| **tp=2** | ~~0.186 s~~ | ~~5.245 ms/tok~~ | ~~1.843 s~~ | ~~190.66 tok/s~~ | ~~10265 / 317 (eos)~~ | ~~25.38 s~~ | 🔴 Qwen/Qwen3-0.6B（非 stepfun）|
| **tp=4** | ~~0.110 s~~ | ~~5.451 ms/tok~~ | ~~2.373 s~~ | ~~183.44 tok/s~~ | ~~10265 / 416 (eos)~~ | ~~30.25 s~~ | 🔴 Qwen/Qwen3-0.6B（非 stepfun）|
| **tp=8 (long)** | ~~0.071 s~~ | ~~5.542 ms/tok~~ | ~~1.629 s~~ | ~~180.43 tok/s~~ | ~~10265 / 282 (eos)~~ | ~~44.98 s~~ | 🔴 Qwen/Qwen3-0.6B（非 stepfun）|
| tp=8 (起服) | ~~0.037 s~~ | ~~3.562 ms/tok~~ | ~~0.262 s~~ | ~~280.72 tok/s~~ | ~~269 / 64 (短)~~ | ~~45.82 s~~ | 🔴 Qwen/Qwen3-0.6B（非 stepfun）|

> ~~✅ tp=8 long-prompt 已由 perf-T7 补完（与 tp=2/4 同口径），§7 P1 闭环。tp=8 起服行仅作冒烟参考。~~ — ⚠️ "同口径"前提失效（4 行同样误归属 Qwen3-0.6B）。

> ~~**perf_tp_eval wave 正式 CLOSED（含 P1 补完）。**~~ — 🔴 CLOSED 状态被翻转（L17c/L19d audit 实证 baseline 误归属），见顶部 🔴 banner。

---

## Cross-link（audit & 修正 trail）

| 链路节点 | 路径 |
|---|---|
| L17c baseline 实证翻转（首发现 tp=2 raw log = Qwen3-0.6B）| `tp2_verify_post_merge_wave/progress/teammate-L17c-baseline-audit.md` |
| L18 stepfun gfx942 tp=2 真实 perf 实测（首次）| `tp2_verify_post_merge_wave/progress/teammate-L18-perf-rerun.md` |
| L19b project_summary repo 同步翻转结论（perf-t1.md + PERF_REPORT.md 加 🔴 块）| `tp2_verify_post_merge_wave/progress/teammate-L19b-summary-repo-fix.md` |
| L19d tp=4/tp=8 系列 audit + 文档修正（perf-t2/t3/t4/t7）| `tp2_verify_post_merge_wave/progress/teammate-L19d-tp-extension-audit.md` |
| L19e KNOWN_FACT 全局化（REPRODUCE.md §7.13 加 ATOM `--model` trap 警示）| `tp2_verify_post_merge_wave/progress/teammate-L19e-global-known-fact.md` |
| L20 stepfun gfx942 tp=4/tp=8 真实 perf 实测 | `tp2_verify_post_merge_wave/progress/teammate-L20-perf-tp4-tp8.md` |
| L22 REPRODUCE.md §6.2 写入实测三档 anchor | `tp2_verify_post_merge_wave/progress/teammate-L22-reproduce-doc-finalize.md` |
| L24 全 repo audit 找出 24 处残留误归属 | `tp2_verify_post_merge_wave/progress/teammate-L24-audit-data-residue.md` |
| L25 commit / dispatch 描述 audit | `tp2_verify_post_merge_wave/progress/teammate-L25-audit-commit-currency.md` |
| L26 perf coverage audit | `tp2_verify_post_merge_wave/progress/teammate-L26-audit-perf-coverage.md` |
| L28（本次）todo.md / TEAM_CONFIG.md / PERF_REPORT.md ⚠️→🔴 升级 + perf-t5/t6 disclaimer | `tp2_verify_post_merge_wave/progress/teammate-L28-fix-15-todo-and-perf-report.md` |
| 项目根 KNOWN_FACT（ATOM `--model` trap） | `step35-flash-support/REPRODUCE.md §7.13` |
| 当前 stepfun gfx942 三档 perf anchor | `step35-flash-support/REPRODUCE.md §6.2` |
