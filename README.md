# Project Summary

记录开发工作中每个任务的背景、调查过程、根因与解决路径。

---

## 任务索引

### Step-3.5-Flash 全栈推理支持（2026-04-23 ~ 2026-04-24）

**背景**：在 AMD MI350X（gfx950）上为 StepFun Step-3.5-Flash 模型建立完整推理能力，
包括 BF16 多种 TP 配置及 FP8 量化权重版本。从零开始跑通到 tp=2/4 BF16 + tp=2 FP8 全部通过。

| 子任务 | 状态 | 关键 commit |
|--------|------|-------------|
| [MoE Pipeline 修复](./step35-flash-support/01_moe_pipeline.md) | ✅ 完成 | ATOM `ec8cbe8`，aiter `68fc7d48b`+`3771835ac` |
| [SwigluStep Wiring](./step35-flash-support/02_swiglu_step.md) | ✅ 完成 | ATOM `4a8495e`，aiter `6d70f7b54` |
| [Sliding Window 修复](./step35-flash-support/03_sliding_window.md) | ✅ 完成 | aiter `7ebae9afb` |
| [TP=4/8 支持](./step35-flash-support/04_tp_support.md) | ✅ tp=4 完成；tp=8 硬件阻塞 | ATOM `635e59e`，aiter `7312ea166` |
| [FP8 推理支持](./step35-flash-support/05_fp8_inference.md) | ✅ tp=2 完成 | aiter `c38d0c9e6`，ATOM `9a67e49` |

**详情**：[step35-flash-support/README.md](./step35-flash-support/README.md)

---

## 文档规范

每个任务文档包含以下固定结构：
1. **背景** — 问题从何而来
2. **调查过程** — 实际走过的路，包含走过的弯路和纠正
3. **根因** — 有实验数据支撑的精确结论
4. **解决方案** — 代码改动精确描述（文件+行号）
5. **验证结果** — 数值指标 + 端到端测试
6. **教训** — 可迁移的方法论经验
