# TODO — perf_tp_eval

> **WAVE CLOSED** — 2026-04-29
> 最终交付：`/home/junlin12/project_fp8_tp4_repro/perf_tp_eval/PERF_REPORT.md`（587 行 / 17 表格 / 2 mermaid，A 评级）

## 规范
- [~] 进行中
- [x] 完成（附结论一行）
- [!] 卡住
- [ ] Pending

## Phase 0
- [x] #000 @perf-T0 写 ttft_tpot 测量脚本骨架 + dry-run → `perf_bench.py` + `progress/perf-t0.md`，方案 A，dry-run TTFT=0.038s 通过

## Phase 1（并行）
- [x] #P1-A @perf-T1 tp=2 baseline → TTFT=0.186s / TPOT=5.245ms·tok / 10265→317(eos)，V3=0/W=0 + JIT cache 间接证据 ✓
- [x] #P1-B @perf-T2 tp=4 baseline → TTFT=0.110s / TPOT=5.451ms·tok / 10265→416(eos)，V3=0/W=0 + JIT cache ✓
- [x] #P1-C @perf-T3 tp=8 静态评估 → 预测 inter_per_rank=160→padding 256；probably PASS

## Phase 2
- [x] #P2-D @perf-T4 tp=8 实测起服 → PASS（engine_init=45.82s，1 次 generate=64 tok 完整退出，path 与预测一致）

## Phase 3
- [x] #P3-W @perf-T5 写 PERF_REPORT.md → 587 行 / 17 表格 / 2 mermaid / 11 章 + 附录 A/B
- [x] #P3-R @perf-T6 critical review → A 评级，0 block / 1 warn / 7 info；18/18 数值真实可追溯
- [x] #P3-F lead 应用 F-1 / F-2 行号修订（2 处 Edit）→ inline 修订完成

## In Progress
（无）

## Done
- #000 / #P1-A / #P1-B / #P1-C / #P2-D / #P3-W / #P3-R / #P3-F

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

| tp | TTFT | TPOT | total | decode_thru | input/output | engine_init |
|---|---|---|---|---|---|---|
| **tp=2** | 0.186 s | 5.245 ms/tok | 1.843 s | 190.66 tok/s | 10265 / 317 (eos) | 25.38 s |
| **tp=4** | 0.110 s | 5.451 ms/tok | 2.373 s | 183.44 tok/s | 10265 / 416 (eos) | 30.25 s |
| **tp=8 (起服)** | 0.037 s | 3.562 ms/tok | 0.262 s | 280.72 tok/s | 269 / 64 (短) | 45.82 s |

> ⚠ tp=8 数据是起服测试（短 prompt），**禁止与 tp=2/tp=4 横向 perf 比较**。完整 tp=8 long-prompt perf 测试见 PERF_REPORT.md §7 P1。

**perf_tp_eval wave 正式 CLOSED。**
