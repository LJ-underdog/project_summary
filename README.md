# Project Summary

记录开发工作中每个任务的背景、调查过程、根因与解决路径。

---

## 任务索引

### Step-3.5-Flash 全栈推理支持（2026-04-23 ~ 2026-04-25）

**背景**：在 AMD MI350X（gfx950）上为 StepFun Step-3.5-Flash 模型建立完整推理能力，
包括 BF16 多种 TP 配置及 FP8 量化权重版本。从零跑通到 tp=2/4 BF16 + tp=2/4 FP8 全部通过。

| 子任务 | 状态 | 关键 commit |
|--------|------|-------------|
| [MoE Pipeline 修复](./step35-flash-support/01_moe_pipeline.md) | ✅ 完成 | ATOM `ec8cbe8`，aiter `68fc7d48b`+`3771835ac` |
| [SwigluStep Wiring](./step35-flash-support/02_swiglu_step.md) | ✅ 完成 | ATOM `4a8495e`，aiter `6d70f7b54` |
| [Sliding Window 修复](./step35-flash-support/03_sliding_window.md) | ✅ 完成 | aiter `7ebae9afb` |
| [TP=4/8 支持](./step35-flash-support/04_tp_support.md) | ✅ tp=4 完成；tp=8 硬件阻塞 | ATOM `635e59e`，aiter `7312ea166` |
| [FP8 tp=2 推理](./step35-flash-support/05_fp8_inference.md) | ✅ 完成 | aiter `c38d0c9e6`，ATOM `9a67e49` |
| [FP8 tp=4 推理](./step35-flash-support/06_fp8_tp4.md) | ✅ 完成（三层 bug，scale sharding 为根因） | ATOM `ccb64621` |

**详情**：[step35-flash-support/README.md](./step35-flash-support/README.md)

### Step-3.5-Flash 验证状态（2026-04-26）

V01-V07 验证 pipeline **全部 PASS**。详见：
- 验证结果汇总：`step35-flash-support/verification_pipeline/results/SUMMARY.md`
- 下一步任务：`step35-flash-support/verification_pipeline/NEXT_TASK_BRIEF.md`
  （目标：FP8 tp=4 无 padding CK kernel，消除 inter_dim=320→384 的 20% 显存浪费）

### Step-3.5-Flash 后续 wave（07-19，2026-04-25 ~ 2026-04-29）

| 子任务 | 内容 | 状态 |
|--------|------|------|
| [07 tp=4 长序列 BOS 修复](./step35-flash-support/07_tp4_longseq_bos_fix.md) | 10k token prefill 全 BOS 根因与修复 | ✅ |
| [08 MoE no-padding 调研](./step35-flash-support/08_moe_no_padding_research.md) | inter_dim=320→384 padding 是否可消除 | ✅ |
| [09 MoE no-padding 深挖](./step35-flash-support/09_moe_no_padding_deep_dive.md) | 为什么 FP8 MoE kernel 需要 padding | ✅ |
| [10 gfx950 FP8 mfma KPack=32 约束](./step35-flash-support/10_fp8_mfma_kpack32_constraint.md) | blockscale MoE 不能去 padding 的 ISA 级原因 | ✅ |
| [11 张量并行策略](./step35-flash-support/11_tensor_parallelism_strategy.md) | TP 原理 + 每个算子 TP 行为分析 | ✅ |
| [12 FP8 tp=4 复现指南](./step35-flash-support/12_reproduction_guide_fp8_tp4.md) | 新机器复现 TTFT≈86ms / TPOT≈13ms | ✅ |
| [13 Recall 系统分析](./step35-flash-support/13_recall_system_analysis.md) | Recall 工具实战指南 | ✅ |
| [14 gfx950 → gfx942(MI308X) 迁移](./step35-flash-support/14_migration_gfx942/MIGRATION_REPORT.md) | M1 tp=2 + M2 tp=4 PASS；NEW-RC-1/2/3 三 RC | ✅ CLOSED |
| [15 TP=2/4/8 性能评估](./step35-flash-support/15_perf_tp2_tp4_tp8_eval/PERF_REPORT.md) | gfx942 上 tp=2/4 perf + tp=8 起服评估 | ✅ |
| [16 gfx950 性能基线](./step35-flash-support/16_perf_gfx950_verified/RESULTS.md) | 统一脚本测得的 gfx950 perf 基线 | ✅ |
| [17 ATOM MoE tp=8 load crash](./step35-flash-support/17_atom_moe_tp8_load_crash/README.md) | tp=8 load_w2 / load_w13 narrow size<0 issue draft | ✅ 内部 CLOSED；未 file upstream |
| [18 FP8 tp=8 root cause + fix](./step35-flash-support/18_fp8_tp8_root_cause_and_fix/README.md) | tp=8 起服双层 root cause；ATOM `969d564` | ✅ |
| [19 Kernel Dispatch 报告](./step35-flash-support/19_kernel_dispatch_report/REPORT.md) | FP8 tp=2/4 每类 op 的 torch / CK / ASM kernel 归属（gfx950） | ✅ |

**跨 topic 资产**：[verification_pipeline/](./step35-flash-support/verification_pipeline/) — V01-V07 验证 pipeline（覆盖 01-07），含 `MASTER_PIPELINE.md` / `PIPELINE_REVIEW_FINAL.md` / `results/SUMMARY.md`。

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
