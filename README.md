# Project Summary

记录开发工作中每个任务的背景、调查过程、根因与解决路径。

---

## 任务索引

### Step-3.5-Flash 全栈推理支持（2026-04-23 ~ 2026-04-29）

**背景**：在 AMD MI350X (gfx950) 上为 StepFun Step-3.5-Flash 模型建立完整推理能力，包括 BF16 多种 TP 配置及 FP8 量化权重版本；后续扩展支持 MI308X (gfx942) 平台。从零跑通到 tp=2/4 BF16 + tp=2/4/8 FP8 全部通过。

**入口**：[`step35-flash-support/README.md`](./step35-flash-support/README.md) — 项目 TL;DR + 时间线 + 4 一级条目导航 + details/ 完整索引

| 顶层文件 | 用途 |
|---|---|
| [`step35-flash-support/README.md`](./step35-flash-support/README.md) | 项目入口 + 概览 + 时间线 + details/ 索引 |
| [`step35-flash-support/REPRODUCE.md`](./step35-flash-support/REPRODUCE.md) | 端到端复现指南（gfx942 / MI308X / FP8 单路径；gfx950 历史路径见 details/） |
| [`step35-flash-support/CODE_CHANGES.md`](./step35-flash-support/CODE_CHANGES.md) | 三仓所有 code 改动总账（per-repo + per-feature 视图） |
| [`step35-flash-support/details/`](./step35-flash-support/details/) | 所有详细内容下沉（topics/research/perf/issues/projects/meta/scripts/verification_pipeline 8 类） |

**子任务清单**（详见 `step35-flash-support/README.md` 的 details/ 完整索引；下表给出 details/ 直链）：

| # | 子任务 | 路径 |
|---|---|---|
| 01 | MoE Pipeline 修复 | [`details/topics/01_moe_pipeline.md`](./step35-flash-support/details/topics/01_moe_pipeline.md) |
| 02 | SwigluStep Wiring | [`details/topics/02_swiglu_step.md`](./step35-flash-support/details/topics/02_swiglu_step.md) |
| 03 | Sliding Window 修复 | [`details/topics/03_sliding_window.md`](./step35-flash-support/details/topics/03_sliding_window.md) |
| 04 | TP=4/8 支持 | [`details/topics/04_tp_support.md`](./step35-flash-support/details/topics/04_tp_support.md) |
| 05 | FP8 tp=2 推理 | [`details/topics/05_fp8_inference.md`](./step35-flash-support/details/topics/05_fp8_inference.md) |
| 06 | FP8 tp=4 推理（三层 bug） | [`details/topics/06_fp8_tp4.md`](./step35-flash-support/details/topics/06_fp8_tp4.md) |
| 07 | tp=4 长序列 BOS 修复 | [`details/topics/07_tp4_longseq_bos_fix.md`](./step35-flash-support/details/topics/07_tp4_longseq_bos_fix.md) |
| 08 | MoE no-padding 调研 | [`details/research/08_moe_no_padding_research.md`](./step35-flash-support/details/research/08_moe_no_padding_research.md) |
| 09 | MoE no-padding 深挖 | [`details/research/09_moe_no_padding_deep_dive.md`](./step35-flash-support/details/research/09_moe_no_padding_deep_dive.md) |
| 10 | gfx950 FP8 mfma KPack=32 约束 | [`details/research/10_fp8_mfma_kpack32_constraint.md`](./step35-flash-support/details/research/10_fp8_mfma_kpack32_constraint.md) |
| 11 | 张量并行策略 | [`details/research/11_tensor_parallelism_strategy.md`](./step35-flash-support/details/research/11_tensor_parallelism_strategy.md) |
| 12 | FP8 tp=4 复现指南（历史） | [`details/topics/12_reproduction_guide_fp8_tp4.md`](./step35-flash-support/details/topics/12_reproduction_guide_fp8_tp4.md) |
| 13 | Recall 系统分析 | [`details/meta/13_recall_system_analysis.md`](./step35-flash-support/details/meta/13_recall_system_analysis.md) |
| 14 | gfx950 → gfx942 (MI308X) 迁移 | [`details/projects/14_migration_gfx942/MIGRATION_REPORT.md`](./step35-flash-support/details/projects/14_migration_gfx942/MIGRATION_REPORT.md) |
| 15 | TP=2/4/8 性能评估（gfx942） | [`details/perf/15_perf_tp2_tp4_tp8_eval/PERF_REPORT.md`](./step35-flash-support/details/perf/15_perf_tp2_tp4_tp8_eval/PERF_REPORT.md) |
| 16 | gfx950 性能基线 | [`details/perf/16_perf_gfx950_verified/RESULTS.md`](./step35-flash-support/details/perf/16_perf_gfx950_verified/RESULTS.md) |
| 17 | ATOM MoE tp=8 load crash（issue draft） | [`details/issues/17_atom_moe_tp8_load_crash/README.md`](./step35-flash-support/details/issues/17_atom_moe_tp8_load_crash/README.md) |
| 18 | FP8 tp=8 root cause + 双层 fix | [`details/topics/18_fp8_tp8_root_cause_and_fix/README.md`](./step35-flash-support/details/topics/18_fp8_tp8_root_cause_and_fix/README.md) |
| 19 | Kernel Dispatch 报告（gfx950） | [`details/research/19_kernel_dispatch_report/REPORT.md`](./step35-flash-support/details/research/19_kernel_dispatch_report/REPORT.md) |

**跨 topic 资产**：[`details/verification_pipeline/`](./step35-flash-support/details/verification_pipeline/) — V01-V07 验证 pipeline（覆盖 01-07），含 `MASTER_PIPELINE.md` / `PIPELINE_REVIEW_FINAL.md` / `results/SUMMARY.md`。

---

## 调试方法论

[DEBUGGING_METHODOLOGY.md](./DEBUGGING_METHODOLOGY.md) 汇总了所有任务中使用的调试手段，包含：

| 手段 | 用途 |
|------|------|
| 最小复现 + 组件隔离 | 将问题定位到单个 kernel/函数 |
| Canary 实验 | 验证"内存是否被越界写入"等假说 |
| 参数 Sweep | 找到触发 bug 的边界值（inter_dim/ctx_len 等） |
| Cos-sim 层级验证 | 验证 kernel 正确性，与端到端解耦 |
| Bug 掩盖检测 | 识别多 bug 场景下的互相掩盖 |
| Monkey-patch 隔离 | 快速替换单个组件定位问题 |
| 对比法 | 正确/错误分支的代码差异定位 |
| Bisection | 逐步开关组件找最小复现集合 |
| Stale JIT Cache | 排除旧编译缓存干扰 |
| op_test vs 生产路径 | 避免测试路径与生产路径不一致 |

---

## 文档规范

每个任务文档包含以下固定结构：
1. **背景** — 问题从何而来
2. **调查过程** — 实际走过的路，包含走过的弯路和纠正
3. **根因** — 有实验数据支撑的精确结论
4. **解决方案** — 代码改动精确描述（文件+行号）
5. **验证结果** — 数值指标 + 端到端测试
6. **教训** — 可迁移的方法论经验
