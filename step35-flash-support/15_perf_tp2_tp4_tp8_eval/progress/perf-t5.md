# perf-T5 progress

> 任务：`#P3-W` — 撰写 PERF_REPORT.md
> WORK_DIR = `/home/junlin12/project_fp8_tp4_repro/perf_tp_eval`
> 日期：2026-04-29
> 角色：writer（lead 文档员）
> 红线遵守：未改 ATOM / aiter / CK / perf_bench.py 任何源码；未动 perf-t0/t1/t2/t3/t4 任何 progress；仅新建 `PERF_REPORT.md` + 本文件

---

## 1. 章节完成度对照表

任务 prompt 强制大纲 11 项全部覆盖：

| # | 强制章节 | PERF_REPORT.md 位置 | 状态 |
|---|---|---|---|
| 1 | TL;DR（5-10 行 + 核心数字表） | 顶部 TL;DR 节（5 条要点 + 核心数字表 1） | ✓ |
| 2.1 | §1 测试目标 | §1.1 表 | ✓ |
| 2.2 | perf_bench.py 方案 A 设计（从 perf-T0） | §1.2 含核心代码 + 字段引用 | ✓ |
| 2.3 | 测量协议（2 次取 Run 2 / warmup CUDAGraph） | §1.3 表 | ✓ |
| 2.4 | 验证 path 协议（V1/V2/V3/W + multi-process 限制说明） | §1.4 表 + 解释 | ✓ |
| 3.1 | tp=2/tp=4 完整数据表 | §2.1 表 | ✓ |
| 3.2 | tp=2 vs tp=4 trade-off 分析 | §2.2 表 + 结论 | ✓ |
| 3.3 | 与 docs/baseline_tp{2,4}_result.md M1/M2 dispatch 一致性 | §2.3 表 | ✓ |
| 4.1 | tp=8 验收 4 项 | §3.1 表 | ✓ |
| 4.2 | inter_dim 实际 vs perf-T3 预测 | §3.2 表 | ✓ |
| 5.1 | tp=8 已自动满足项 | §4.1 表 | ✓ |
| 5.2 | 待实测验证项（含 ✓/⏸/✗） | §4.2 表 | ✓ |
| 5.3 | 长 prompt + 高 throughput 未测项 | §4.3 表 | ✓ |
| 6.1 | mermaid: tp=2/4/8 padding 分支 | §5.1（迁移自 perf-T3 §10） | ✓ |
| 6.2 | mermaid: 数据流 + TTFT/TPOT 对比 | §5.2 数据流验收图 + §5.3 TTFT/TPOT 对比表 | ✓ |
| 7 | 风险表 + 工作清单（block/warn/info + 闭环状态） | §6.1 表（25 条 risk）+ §6.2 分类汇总 | ✓ |
| 8 | 后续 P1-P5 + 长 context | §7（P1-P6） | ✓ |
| 9 | 引用清单 | §8 按源码 / progress / doc / log / 外部 doc 分组 | ✓ |
| 10 | 附录 A：完整 stable 数值 raw log | A.1 (tp=2) / A.2 (tp=4) / A.3 (tp=8) | ✓ |
| 11 | 附录 B：reviewer 抽查指引 | 10 条抽查项 + 复核命令 | ✓ |

强制元素：

| 项 | 要求 | 实际 | 状态 |
|---|---|---|---|
| mermaid 图 | ≥ 2 | 2（§5.1 + §5.2） | ✓ |
| markdown 表格 | ≥ 5 | 17 | ✓ |
| 节末"引用："聚合 | 每节末尾 | §1-§7 全有 | ✓ |
| 中文 + file:line 引用 | 全程 | 全程 | ✓ |
| TTFT/TPOT 数值有 log 行号 | 全部 | 三档全有 `logs/tp{2,4,8}_*.log:N` | ✓ |
| tp=8 不可比标注 | 显式 | TL;DR + §3 + §5.3 + 附录 A.3 + §6 多处 | ✓ |

---

## 2. mermaid 图统计

| # | 位置 | 类型 | 内容 |
|---|---|---|---|
| 1 | §5.1 | flowchart TD | tp=2/4/8 inter_per_rank padding 分支（迁移自 `perf-t3.md:312-342`，验证 perf-T4 已实测部分加注） |
| 2 | §5.2 | flowchart LR | 数据流：perf_bench → engine_init → warmup → measure → ATOM postprocess → V1/V3/W 验收 |

合计 2 张（满足 ≥ 2 要求）。

---

## 3. 表格 / 引用统计

### 3.1 表格分布（17 张）

| 章节 | 表数 | 内容摘要 |
|---|---|---|
| TL;DR | 1 | 核心数字表 |
| §1 | 4 | 测试目标 / 测量协议 / 验证 path |
| §2 | 2 | tp=2/4 完整数据 / dispatch path 对照 |
| §3 | 2 | 验收 4 项 / inter_dim 间接证据 |
| §4 | 3 | 已自动满足 / 待验 / 未测 |
| §5 | 1 | TTFT/TPOT 对比含可比性提示 |
| §6 | 2 | 风险汇总（25 条）+ 分类计数 |
| §7 | 1 | P1-P6 后续优化 |
| 附录 B | 1 | reviewer 抽查 10 条 |

### 3.2 file:line 引用密度

| 类别 | 引用数（含重复） |
|---|---|
| ATOM 源码 | 8 处（model_engine / model_ops / models / arg_utils） |
| aiter 源码 | 4 处（fused_moe.py:867-926 / 881-886 / configs/tuned_fmoe.csv / jit/） |
| HF 模型 config.json | 1 处 |
| 项目 progress | perf-t0/t1/t2/t3/t4 全部引用，每个 ≥ 3 处 |
| 项目 doc | MIGRATION_REPORT / docs/baseline_tp{2,4}_result.md / TEAM_CONFIG | 多处 |
| log 行号 | tp{2,4}_run{1,2}.log + tp8_launch.log + dry_run | 三档 stable + raw 全部带行号 |
| 外部（rocm-ref） | 2 处（hardware-specs-table / glossary） |

---

## 4. 给 perf-T6 reviewer 的抽查建议

按重要度排序（详见 PERF_REPORT.md 附录 B 完整 10 条）：

1. **数值真实度抽查**（基础）：
   - tp=2: `logs/tp2_run2.log:1-12`，TTFT=0.186 / TPOT=5.245
   - tp=4: `logs/tp4_run2.log:1-12`，TTFT=0.110 / TPOT=5.451
   - tp=8: `logs/tp8_launch.log:1-13`，TTFT=0.037 / TPOT=3.562

2. **V1/V2 grep=0 间接证据是否被接受**（核心争议点）：
   - 论证模式：`progress/perf-t1.md:107-138` + `perf-t4.md:96-125`
   - 接受路径：multi-process stderr 限制 + JIT cache 间接证据 + 与 M1/M2 PASS dispatch 一致性
   - 不接受 → 需 lead 决定是否新派 task 改脚本（红线本任务禁改）

3. **JIT cache 现场可独立验证**：
   ```
   ls /workspace/aiter/aiter/jit/ | grep -E "module_moe|fmoe"
   # 应见：
   #   module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2.so
   #   module_moe_ck2stages_f8_f8_preshuffle_on_b16_swiglustep_per_1x128_mulWeightStage2.so
   #   module_moe_sorting.so
   ```

4. **tp=8 inter_dim=256 三重间接证据**：
   - `atom/model_ops/moe.py:1719-1727` padding 公式
   - `atom/model_ops/moe.py:1725` 注释明文 "tp=8 inter=160 → 256"
   - JIT cache 无新 module
   - 0 处 `no instance found` / `IsSupportedArgument false`

5. **tp=8 异常 log 独立 grep**：
   ```
   grep -nE "RuntimeError|Traceback|ERROR|HSAMemoryAllocFailed" \
     /home/junlin12/project_fp8_tp4_repro/perf_tp_eval/logs/tp8_launch_full.log
   # 预期 0 命中（仅 1 行 [W...] Guessing device ID 是 NCCL warning）
   ```

6. **tp=8 不可与 tp=2/4 横向比的标注是否充分**（reviewer 容易遗漏）：
   - TL;DR 表格、§3 验收、§5.3 表头、附录 A.3 全部带显式警告
   - 若觉得不充分可建议 lead 加更多 disclaimer

7. **风险闭环状态**（§6.1）：
   - 25 条 risk 全部归类（block=0 / warn 已闭环=4 / warn 已接受=2 / info=19）
   - 抽查 R-T3-R1（inter padding 未实测）→ 看 perf-T4 §5.2 是否真闭环

8. **附录 B 10 条抽查项**：可以一一独立复核

---

## 5. 已知 limitation（perf-T5 自查）

| # | 项 | 说明 |
|---|---|---|
| L1 | tp=8 数值未跑 long prompt（10k input） | 任务范围明确不要求；写入 §7 P1 |
| L2 | output ≠ 1024（eos / max_tokens 提前停） | 写入 §7 P3，建议下一轮加 ignore_eos |
| L3 | V1/V2 直接 grep 不可得 | multi-process stderr 限制；红线禁改脚本；用三重间接证据替代 |
| L4 | MFU 未计算 | 写入 §7 P5 |
| L5 | 32k+ context 未评估 | 写入 §7 P6 |

所有 limitation 全部在 PERF_REPORT.md §7 落地为后续 task 候选；无遗漏。

---

## 6. 文件清单（本任务产出）

- `WORK_DIR/PERF_REPORT.md`（587 行；17 张表格；2 张 mermaid；满足全部强制大纲）
- `WORK_DIR/progress/perf-t5.md`（本文件）

---

## 7. 红线自查

- [x] 未改 ATOM / aiter / CK 任何源码
- [x] 未改 perf_bench.py
- [x] 未动 perf-t0/t1/t2/t3/t4 任何 progress 文件
- [x] 仅新建 PERF_REPORT.md + 本文件
- [x] 中文 + file:line 引用全程
- [x] PERF_REPORT.md 大纲 11 项全部覆盖
- [x] mermaid ≥ 2（实际 2）
- [x] 表格 ≥ 5（实际 17）
- [x] TTFT/TPOT 数值有 log 行号引用
- [x] tp=8 短 prompt 不可比警告显式标注（多处）
