# V01 MoE Pipeline 验证计划

适用范围：ATOM commit `ec8cbe8`（shuffle_weights gfx950 skip 移除）+ aiter commit `68fc7d48b`（block_m=128 V1->V3 强制路径）。
覆盖 Bug 0 / Bug 1，及 Bug 2-4（buffer padding revert）的 canary 复核。

---

## A. Code Review 结论

### Fix 1 分析（shuffle_weights，ATOM `ec8cbe8`）

参考代码：`/home/hanchang/ATOM/atom/model_ops/moe.py` L483-L525（`UnquantizedFusedMoEMethod.process_weights_after_loading`）。
当前线上代码（已修复）：

```python
520  # Shuffle weights for CK/ASM kernels.
521  # Previously skipped for gfx950 bf16 g1u1 on the assumption that the CK
522  # 2-stage preshuffle_off (NSwizzle=0) kernel expected un-shuffled weights.
523  # Verified 2026-04-23: preshuffle_off GEMM is wrong on gfx950; preshuffle_on
524  # (NSwizzle=1) is correct. Always shuffle so the right kernel path is used.
525  shuffle_weights(layer.w13_weight, layer.w2_weight)
```

- 是否必要：是。修复前的形态是 `if get_gfx() == "gfx950": pass`，相当于在 gfx950 上把权重以"未 shuffle"的物理布局喂给 CK 2-stage GEMM。CK 2-stage 的 preshuffle_off（NSwizzle=0）路径在 gfx950 上 GEMM 计算错误，cos_sim ≈ -0.006（即输出与参考完全无关），表现为推理乱码 / 无效输出。
- 是否是根因：是。问题的最终触发点在 CK kernel（preshuffle_off NSwizzle=0 在 gfx950 NSwizzle metadata 处理不正确），但在 ATOM 侧通过强制走 preshuffle_on（NSwizzle=1）路径绕开。属于"在能改的最上层修复"，不是临时 hack。
- 需核查的代码：
  - `/home/hanchang/ATOM/atom/model_ops/moe.py` L520-L525：BF16 路径的 shuffle 调用。
  - `/home/hanchang/ATOM/atom/model_ops/moe.py` L922 / L934 / L1749 / L1770 / L1800：FP4/quark/blockscale/channel/tensor 各 quant 路径的 shuffle，用以确认 fix-then-sweep 已覆盖到，且没有在其它 quant 类型上残留 gfx950 skip。
  - `/home/hanchang/aiter/aiter/fused_moe.py` L880-L920：CK kernel dispatch 表，确认 `is_shuffled` flag 与 `kernelName2` 选择如何耦合 NSwizzle；与 `block_m=128` workaround 的 `kernelName2 = "...Nswizzle0..."` 行（L909-L910）逻辑一致。
- 风险评估：
  - 对其他 GPU 架构（gfx942 / gfx90a）：原本 gfx942 路径就在调用 `shuffle_weights`，本 fix 只是把 gfx950 的 skip 去掉，不影响其它架构。
  - 对内存：`shuffle_weights` 是原地变换（同 shape），不引入额外占用。
  - 对其它 quant：shuffle_weights 在 `_process_block_quant` / `_process_channel_quant` / `_process_tensor_quant` / FP4 路径上一直存在；本 fix 没有改这些路径，但应核查这些路径内是否曾经也有 gfx950 skip（grep 结果显示无）。
  - 对 preshuffle_on/off 选择的稳定性：依赖 `is_shuffled` 在 dispatch 阶段被正确传递。需确认 ATOM 调用 `fused_moe(...)` 时 `is_shuffled=True` 与 weight 的物理布局一致，否则会反向走错。

### Fix 2 分析（block_m=128 强制，aiter `68fc7d48b`）

参考代码：`/home/hanchang/aiter/aiter/fused_moe.py` L900-L910：

```python
900  # gfx950 workaround: V1 CK kernel produces wrong results for inter_dim>192
901  # (memory corruption / incorrect computation for both preshuffle_on and
902  # preshuffle_off paths). Force block_m=128 to select the correct V3 stage1
903  # kernel. For preshuffle_off, also force the V3 stage2 kernel by name.
904  # Note: blockscale (per_1x128/per_1x32) dispatch only supports block_m<=64
905  # and is not affected by the V1 bug, so exclude it from this override.
906  if not run_1stage and inter_dim > 192 and get_gfx() == "gfx950" \
907          and q_type not in (QuantType.per_1x128, QuantType.per_1x32):
908      block_m = 128
909      if not is_shuffled and not kernelName2:
910          kernelName2 = "moe_ck2stages_gemm2_256x128x128x64_1x4_TypeCast_v3_Nswizzle0_Quant0_MulRoutedWeight1_B16_B16_B16"
```

- 是否必要：是。CK 2-stage V1 kernel（block_m < 128 → V1，常见 block_m=16/64）在 gfx950 + `inter_dim > 192` 时输出错误（cos_sim ≈ 0.004，与参考无相关性）。Step-3.5-Flash 在 tp=2 时 `inter_dim=640`，tp=4 时 `inter_dim=320`，tp=8 时 `inter_dim=160`，因此 tp=2/tp=4 触发 bug，tp=8 不触发该 V1 bug（但 tp=8 仍受 GPU5 等其它问题影响）。
- 是否是根因：从现象看是 V1 在 inter_dim>192 时 N-tile pass 数量大于某阈值（实测 cos_sim 在 inter_dim=192 PASS、inter_dim=256 FAIL），属于 V1 kernel 自身缺陷。修复策略是强制走 V3（256x128x128x64 块大小）。这是绕开 V1 缺陷的可靠路径，但根因在 CK V1 kernel 实现，未被修复，只是被避开。
- 需核查的代码：
  - `/home/hanchang/aiter/aiter/fused_moe.py` L883-L899：`run_1stage`、`block_m` 默认计算逻辑（`get_block_size_M(token, topk, expert, inter_dim)`），确认仅在"未触发 1stage 且 q_type 非 blockscale"时才进入 V1 分支。
  - `/home/hanchang/aiter/aiter/fused_moe.py` L517-L555：`get_block_size_M` 与 `get_ksplit`，确认默认值确实可能落到 16/64（即 V1）。
  - kernelName2 的可用列表（CK 编译产物）：需确认 `moe_ck2stages_gemm2_256x128x128x64_1x4_TypeCast_v3_Nswizzle0_Quant0_MulRoutedWeight1_B16_B16_B16` 已注册到 dispatch 表（否则 fallback 仍是 V1）。
- 风险评估：
  - 性能影响：block_m 从 16/64 升到 128 在 token 数较小时会浪费 padding（tile 利用率下降）。但 Step-3.5-Flash decode TPOT 实测仍优于 BF16 同 tp（FP8 tp=4 比 BF16 tp=4 快 19%），说明性能没有明显回退。需确认在 token=1（pure decode）也未触发性能悬崖。
  - 其它 quant 影响：condition 已排除 `per_1x128`、`per_1x32`（blockscale 只支持 block_m<=64）。但 `per_Token`（FP8/INT8）会进入此 override；需确认 FP8 per_Token 路径的 V3 kernel 也已实现。
  - 对非 gfx950：`get_gfx() == "gfx950"` 守卫，无影响。

### Buffer padding revert 分析（Bug 2-4，commit 3771835ac revert）

- canary 实验证明了什么？在 Fix 1 + Fix 2 已落地后，把 buffer padding（推测包括 stage1/stage2 中间 buffer 的 N/K 对齐 padding）单独 revert，端到端推理仍然正确（cos_sim、输出文本均通过）。说明这些 padding 在 V3 kernel 路径下并不必要——V3 内部 tile（256x128x128x64）已经把对齐需求内吞，未对齐的 inter_dim 会由 CK 自身处理 partial tile。
- 为何确认 padding 不必要？因为问题的实际根因是 V1 kernel（block_m<128）+ NSwizzle 路径错误，buffer padding 当时是误以为"对齐不足导致越界"才加的。当强制走 V3 后，V1 不再被命中，padding 的 hypothesis 就不成立。Reviewer 提示：仍需在 ATOM `moe.py` L489-L518 处确认 inter_dim padding 的代码是否也被一并 revert——若 ATOM 代码里仍保留 inter=160→192、320→384 padding，则它属于"额外保险"而非纯粹无害，应在性能审计中再评估其开销。

---

## B. 验证实验设计

### 实验 1：Fix 1 正确性验证（preshuffle_off vs preshuffle_on，最小复现）

- 目标：在 gfx950 上独立复现 preshuffle_off cos_sim<0.01、preshuffle_on cos_sim>0.9999。
- 测试脚本：`/home/hanchang/aiter/op_tests/test_moe_2stage.py`（`test_fmoe` 接受 `preshuffle` 参数，L54）。
- 命令（gfx950，单卡）：

```bash
cd /tmp && /opt/venv/bin/python /home/hanchang/aiter/op_tests/test_moe_2stage.py \
  --dtype bf16 --token 32 --model_dim 7168 --inter_dim 320 \
  --E 256 --topk 8 --use_g1u1 --preshuffle off 2>&1 | tee /tmp/v01_exp1_off.log

cd /tmp && /opt/venv/bin/python /home/hanchang/aiter/op_tests/test_moe_2stage.py \
  --dtype bf16 --token 32 --model_dim 7168 --inter_dim 320 \
  --E 256 --topk 8 --use_g1u1 --preshuffle on  2>&1 | tee /tmp/v01_exp1_on.log
```

注：若 `test_moe_2stage.py` 不接受 `--preshuffle` 命令行开关（实际由内部 itertools 笛卡尔生成），则改为在脚本入口固定 `preshuffle=True/False` 各跑一次（不修改原逻辑，新建一份临时 driver 在 `/tmp`）。

- 判断标准：
  - preshuffle_off 在 gfx950 上 `cos_sim < 0.01`（与 PROJECT_SUMMARY 报告的 -0.006 同号同量级）→ 复现成功。
  - preshuffle_on 在 gfx950 上 `cos_sim ≥ 0.9999`（目标 0.999989）→ 修复有效。
  - 若两者都 PASS，说明 fix 不再"必要"——需重新评估根因是否已在 CK 侧悄悄修复。
- 所需文件：`/home/hanchang/aiter/op_tests/test_moe_2stage.py`。

### 实验 2：Fix 2 正确性验证（V1 vs V3 kernel @ inter_dim 边界）

- 目标：复现"inter_dim=192 PASS、inter_dim=256 FAIL（V1 路径）；inter_dim=256 + V3 PASS"。
- 强制 kernel 选择方式：通过 `block_m` 控制——V1 路径要求 `block_m < 128`（用 `block_m=64`），V3 路径用 `block_m=128`。在 `aiter/aiter/fused_moe.py` L891-L910 的 dispatch 内，`block_m` 是关键判别量；`test_moe_2stage.py` 的入口可显式传 `block_size_M` 给 `fused_moe_2stages` 内部的 `torch_moe_stage1`/`torch_moe_stage2` 比对。
- 条件矩阵：

| 实验编号 | inter_dim | block_m（强制） | 期望 cos_sim |
|----------|-----------|-----------------|-------------|
| 2.a | 192 | 64 (V1)  | ≥ 0.9999（边界 PASS） |
| 2.b | 256 | 64 (V1)  | < 0.01（V1 bug 触发） |
| 2.c | 256 | 128 (V3) | ≥ 0.9999（fix 路径） |
| 2.d | 320 | 128 (V3) | ≥ 0.9999（生产配置 tp=4） |
| 2.e | 640 | 128 (V3) | ≥ 0.9999（生产配置 tp=2） |

- 命令模板（基于 test_moe_2stage.py，需在脚本中或临时 driver 中固定 `block_size_M` 与 `inter_dim`）：

```bash
cd /tmp && AITER_LOG_LEVEL=INFO /opt/venv/bin/python -c "
import sys; sys.path.insert(0, '/home/hanchang/aiter/op_tests')
from test_moe_2stage import test_fmoe
import aiter
from aiter import dtypes
# 调用形如：
# test_fmoe(dtypes.bf16, token=32, model_dim=7168, inter_dim=256, E=256, topk=8,
#           actType=..., qType=aiter.QuantType.No, AQDType=..., WQDType=...,
#           use_g1u1=True, doweight_stage1=False, preshuffle=True)
" 2>&1 | tee /tmp/v01_exp2_<case>.log
```

实操中，建议直接在 `/tmp/v01_exp2_driver.py` 写 driver 文件（不修改 aiter 源码），通过 `aiter.fused_moe(..., block_size_M=64)` / `block_size_M=128` 显式传参；如果 fused_moe 对外 API 不暴露 block_size_M，可临时 monkey-patch `get_block_size_M` 返回固定值。

- 判断标准：每个 case 的 cos_sim 满足上表"期望"列即 PASS。
- 失败处理：
  - 若 2.a FAIL（inter=192+V1 也错），说明 V1 边界比 192 更小，需重新探边界，proposed_fix_A01.md 的"边界=192"结论需修订。
  - 若 2.b PASS（V1 在 inter=256 也正确），说明 CK V1 已被修复，Fix 2 可降级为"性能/稳健性"而非"正确性"修复。
  - 若 2.c FAIL（V3 也错），说明 fix 不充分，需进一步排查 NSwizzle 与 kernelName2 dispatch。

### 实验 3：端到端集成验证

- 目标：在已修复的 ATOM/aiter 之上，Step-3.5-Flash bf16 tp=2 完整推理无乱码无 crash。
- 命令：

```bash
cd /tmp && rm -rf /root/.cache/atom/* && CUDA_VISIBLE_DEVICES=0,1 \
  AITER_LOG_LEVEL=WARNING \
  /opt/venv/bin/python -m atom.examples.simple_inference \
  --model stepfun-ai/Step-3.5-Flash --kv_cache_dtype bf16 --trust-remote-code \
  --tensor-parallel-size 2 --level 0 --temperature 0 --max-tokens 128 \
  --max-num-batched-tokens 4096 --max-num-seqs 2048 \
  2>&1 | tee /tmp/v01_exp3_tp2_bf16.log
```

补充推荐 tp=4 同步验证（避开 GPU5）：

```bash
cd /tmp && rm -rf /root/.cache/atom/* && CUDA_VISIBLE_DEVICES=0,1,2,3 \
  AITER_LOG_LEVEL=WARNING \
  /opt/venv/bin/python -m atom.examples.simple_inference \
  --model stepfun-ai/Step-3.5-Flash --kv_cache_dtype bf16 --trust-remote-code \
  --tensor-parallel-size 4 --level 0 --temperature 0 --max-tokens 128 \
  --max-num-batched-tokens 4096 --max-num-seqs 2048 \
  2>&1 | tee /tmp/v01_exp3_tp4_bf16.log
```

- 判断标准（全部成立才 PASS）：
  - 进程 exit code = 0，无 Python traceback。
  - 4 个 sample prompt 全部产出连贯英文 / 中文，无重复 BOS、无乱码（人工目检 + 简单 grep 排除 `<s><s><s>`）。
  - 日志中无 `cos_sim` 警告、无 `NaN`、无 `inf`。
- 注：tp=4 长 prompt（≥10k tokens）已知存在"输出全 BOS"open bug（见 MEMORY tp48-fixes.md），本实验只覆盖短 prompt，长序列归属 V03/V05 验证组。

### 实验 4：回归验证（gfx942 / 其它 quant 路径）

- 目标：确认 Fix 1（去除 gfx950 skip）不影响其它 GPU 架构与其它 quant 路径。
- 实操限制：本机仅 gfx950，gfx942 不可用，按要求降级为代码逻辑验证。
- 验证方式：
  1. 在 `/home/hanchang/ATOM/atom/model_ops/moe.py` 全文 `grep -n "gfx950"`，确认 BF16 `process_weights_after_loading` 路径中无残留 skip；其它 quant 路径（FP4 L922、blockscale L934、FP8 channel L1770、tensor L1800）shuffle_weights 调用对所有架构都执行，不依赖 `get_gfx()`。
  2. 在 `/home/hanchang/aiter/aiter/fused_moe.py` 确认 L906 `if ... and get_gfx() == "gfx950"` 守卫——仅 gfx950 强制 block_m=128，其它架构维持原行为。
  3. 阅读 git log/diff `ec8cbe8` 与 `68fc7d48b`，确认 diff 仅触及 gfx950 守卫与 shuffle 行，无副作用 hunk。
- 判断标准：上述三项 grep / read 全部成立 → "代码层面无回归风险"。如发现任何 quant 路径仍有 gfx950 skip 或 fix 改动外溢到其它架构，记录到 E 节"待确认问题"。

### 实验 5（canary 复核）：Buffer padding 是否仍可 revert

- 目标：复核 PROJECT_SUMMARY 报告"3771835ac revert padding 后端到端仍正确"的结论在当前 HEAD 仍然成立。
- 命令：先在 ATOM 工作树上临时屏蔽 L504-L518 inter_dim padding 块（用 `if False:` 注释包裹，不 commit），重跑实验 3：

```bash
cd /tmp && rm -rf /root/.cache/atom/* && CUDA_VISIBLE_DEVICES=0,1 \
  AITER_LOG_LEVEL=WARNING \
  /opt/venv/bin/python -m atom.examples.simple_inference \
  --model stepfun-ai/Step-3.5-Flash --kv_cache_dtype bf16 --trust-remote-code \
  --tensor-parallel-size 2 --level 0 --temperature 0 --max-tokens 128 \
  2>&1 | tee /tmp/v01_exp5_tp2_no_pad.log
```

- 判断标准：与实验 3 相同的"无 crash + 输出连贯"。
- 失败处理：若 revert 后失败，说明 padding 仍是必要的（即使 V3 kernel），PROJECT_SUMMARY 的 Bug 2-4 revert 结论需要打补丁。
- 风险提示：本实验需要修改工作树文件，必须在执行后 `git checkout` 还原。

---

## C. 验证依赖与顺序

| 步骤 | 依赖 | 优先级 | 失败时是否阻断后续 |
|------|------|--------|--------------------|
| 实验 4（代码静态核查） | 无 | P0 | 否（仅记录） |
| 实验 1（Fix 1 单元复现） | 无 | P0 | 是（若 fix 1 失效，端到端不可信） |
| 实验 2（Fix 2 边界矩阵） | 无 | P0 | 是 |
| 实验 3（端到端 tp=2 / tp=4） | 实验 1、2 PASS | P0 | 是 |
| 实验 5（canary 复核） | 实验 3 PASS | P1 | 否（信息性） |

建议执行顺序：实验 4 → 实验 1 + 实验 2 并行 → 实验 3 → 实验 5。

---

## D. 通过标准汇总

| 验证项 | 通过标准 | 数据来源 | 数值阈值 |
|--------|---------|---------|---------|
| Fix 1 复现-错误态 | preshuffle_off cos_sim < 0.01 | 实验 1 off log | abs(cos_sim) ≤ 0.01 |
| Fix 1 复现-修复态 | preshuffle_on cos_sim ≥ 0.9999 | 实验 1 on log | cos_sim ≥ 0.9999 |
| Fix 2 边界 inter=192 V1 | cos_sim ≥ 0.9999 | 实验 2.a | ≥ 0.9999 |
| Fix 2 边界 inter=256 V1 | cos_sim < 0.01 | 实验 2.b | < 0.01 |
| Fix 2 修复 inter=256 V3 | cos_sim ≥ 0.9999 | 实验 2.c | ≥ 0.9999 |
| Fix 2 生产 inter=320 V3 | cos_sim ≥ 0.9999 | 实验 2.d | ≥ 0.9999 |
| Fix 2 生产 inter=640 V3 | cos_sim ≥ 0.9999 | 实验 2.e | ≥ 0.9999 |
| 端到端 tp=2 bf16 | exit 0 + 输出无乱码 | 实验 3 log | 人工目检 + grep |
| 端到端 tp=4 bf16 | exit 0 + 输出无乱码 | 实验 3 tp4 log | 人工目检 + grep |
| 回归 gfx942 路径 | 代码 grep 无 gfx950 残留 skip | 实验 4 | 布尔 |
| Padding canary | revert padding 后端到端仍 PASS | 实验 5 | 同实验 3 |

---

## E. 待确认问题

1. **`test_moe_2stage.py` 是否对外暴露 `--preshuffle` 与 `--block_size_M` CLI 参数**？读 L38-L80 显示 `test_fmoe` 接受 `preshuffle` 参数，但 `__main__` 入口的 argparse 是否已注册需进一步确认。如未注册，需要写临时 driver（不改 aiter 源码）。
2. **kernelName2 `moe_ck2stages_gemm2_256x128x128x64_1x4_TypeCast_v3_Nswizzle0_Quant0_MulRoutedWeight1_B16_B16_B16` 是否已编译进 CK JIT cache**？若 JIT 缺失，会触发 fallback，实验 2.c 可能假性 PASS 或假性 FAIL。建议实验 2 跑前 `ls /root/.cache/aiter/...` 或在 INFO 日志中确认实际命中 kernel 名。
3. **ATOM `moe.py` L489-L518 的 inter_dim padding 是否在 PROJECT_SUMMARY 所述 commit 3771835ac 中被 revert**？读到的代码（L489 注释 "Verified 2026-04-24"）显示 padding 仍存在，与 summary 中"Bug 2-4 全部 revert"的描述存在张力——可能 summary 指 buffer padding（aiter 侧 stage 中间 buffer），而 ATOM 侧 weight padding 仍保留。需 reviewer 与 fix author 对齐。
4. **FP8 per_Token 路径是否走相同的 V3 fix**？L906 `q_type not in (per_1x128, per_1x32)` 没有排除 per_Token，但 FP8 per_Token 是否真的有对应 V3 kernel name 注册需查 dispatch 表。FP8 tp=4 当前 PASS（MEMORY），间接说明走通了，但缺直接 kernel-level 验证。
5. **Fix 2 性能影响**：block_m 16/64→128 的 padding 浪费在 token 数极小（decode token=1）时是否仍可接受？建议在性能验证组（V0X-perf）补 micro-benchmark：固定 inter_dim=320，对比 token=1/8/32/128 在 V1（假设 V1 修好）vs V3 的 latency。本验证组不负责，但需登记。
6. **gfx942 回归**：本环境无 gfx942，纯代码核查无法完全证伪"shuffle_weights 在 gfx942 是否有副作用"。建议在 gfx942 机器上至少跑一次实验 1 等价测试（`get_gfx()=="gfx942"`），由其它团队执行或排入 CI matrix。

---

V01 MoE 验证计划完成
