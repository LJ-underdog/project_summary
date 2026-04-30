# V05 — FP8 Inference 验证计划

专题组角色：Code Reviewer + Experiment Designer
范围：Step-3.5-Flash-FP8 在 gfx950 上 tp=2 的 FP8 端到端推理修复验证
依据：05_fp8_inference.md（Bug 1: aiter c38d0c9e6；Bug 2: ATOM 9a67e49）

---

## 0. 修复要点速览

### Fix 1（aiter `aiter/fused_moe.py` L906-908）
关键代码段（已落地）：
```python
# Note: blockscale (per_1x128/per_1x32) dispatch only supports block_m<=64
# and is not affected by the V1 bug, so exclude it from this override.
if not run_1stage and inter_dim > 192 and get_gfx() == "gfx950" \
        and q_type not in (QuantType.per_1x128, QuantType.per_1x32):
    block_m = 128
    if not is_shuffled and not kernelName2:
        kernelName2 = "moe_ck2stages_gemm2_256x128x128x64_1x4_TypeCast_v3_..."
```
即在 V1-CK-bug 的 workaround（强制 block_m=128）前增加 q_type guard，避免把 blockscale FP8 路径推入不支持的 dispatch。

### Fix 2（ATOM `atom/model_ops/moe.py` L1815-1820，Fp8MoEMethod.get_fused_moe_quant_config）
```python
elif self.quant_type == QuantType.per_1x128:
    block_shape = [128, 128]
elif self.quant_type == QuantType.per_1x32:
    block_shape = [1, 32]
else:
    block_shape = None
return fp8_w8a8_moe_quant_config(..., block_shape=block_shape)
```
即把原先无差别 `block_shape=None` 的 else 分支拆开，per_1x128 显式传 [128,128]，per_1x32 传 [1,32]。

---

## A. Code Review

### A.1 Fix 1 — q_type guard 是否覆盖所有 blockscale 路径？

**问题**：`q_type not in (per_1x128, per_1x32)` 是否枚举完整？

**分析**：
- aiter `QuantType` 中与 blockscale 相关的枚举值常见有：`per_1x128`（fp8 blockscale 主路径）、`per_1x32`（int4/MXFP 类）。
- 同文件 L881-888 dispatch 分支也以这两者作为 blockscale 判定（`q_type == per_1x128` 与 `elif q_type != per_1x32`）。
- 结论：在当前 aiter 代码中 guard 与 dispatch 的 blockscale 集合一致，无遗漏。

**Review 结论**：✅ guard 集合完整。
**潜在隐患**：若未来 aiter 引入新 blockscale q_type（如 `per_1x64` / `mx_*`），需同步扩充该 tuple，否则会再次踩 V3 强制 block_m=128 的坑。建议在 PR description 中标注此「同步点」。

### A.2 BF16 路径回归安全（q_type=a16w16）

**论证**：
- BF16 路径：`q_type == QuantType.No`（即 a16w16）。该值不在 `(per_1x128, per_1x32)` 中。
- 对 BF16，`q_type not in (per_1x128, per_1x32)` 为 True，与原 guard 前的行为完全等价 → 仍走 `block_m = 128` 的 V3 workaround。
- 因此 BF16 tp=2/4 的 prefill/decode 路径与 FP8 修复前完全相同。

**Review 结论**：✅ Fix 1 对 BF16 零影响（行为不变）。
**回归方式**：以实验 3 复跑 BF16 tp=2 比对 TPOT/TTFT（应与 ec8cbe8 / 4a8495e 之后基线一致：TPOT≈17ms, TTFT≈92ms）。

### A.3 Fix 2 — block_shape=None vs [128,128] 影响

**TP 路径分析**：
- TP 路径中 `fused_moe_quant_config.block_shape` 主要用于：
  1. 选择 block-wise per_1x128 dispatch（部分 wrapper 据此走 a8w8blkscale 而非 per-tensor）。
  2. 校验 weight_scale shape（block_n × block_k 与 W shape 整除关系）。
  3. 触发 weight scale padding / reshape（per-block 对齐）。
- 当 `quant_type == per_1x128` 但 `block_shape=None` 时：
  - 下游 wrapper 可能误判为 per-tensor / channel-quant，调用错误的 fused_moe API；或在校验阶段抛 shape mismatch；或静默走错 kernel 产生 gibberish。
- 显式传 `[128,128]` 后，dispatch 与 scale 校验路径都拿到正确元数据。

**EP 路径分析**：
- EP（Expert Parallel）下 token 分发后仍调用 fused_moe，每个 EP rank 上的 expert 子集 inter_dim 不变（仅 expert 数量切分）。
- block_shape 校验同样作用于 W shape，故 EP 路径同 TP 路径，[128,128] 必要。
- 若 EP + DP 共用，token 量更易跨过 1stage 阈值进入 2stage CK，对 block_shape 元数据依赖更敏感。

**Review 结论**：✅ Fix 2 在 TP/EP 两条路径都必要；区分价值在于实验 5 用单层注入观察 dispatch / shape check 行为。

### A.4 FP8 不需要 inter_dim padding（640 % 128 == 0）

**推导**：
- Step-3.5-Flash MoE expert intermediate dim = 640（每专家 inter_dim/expert=640）。
- aiter 2stage CK blockscale dispatch 要求 `inter_dim % 128 == 0`（与 block_n=128 对齐），见 L883 `inter_dim % 256 == 0` 判 1stage、L887 `inter_dim % 128 != 0` 退回 1stage。
- 640 = 5 × 128 → 满足 128 对齐，可直接走 2stage CK；同时 640 不满足 256 对齐 → 不进入 1stage 优化路径，与代码 dispatch 一致。
- 对比 BF16 tp=4：inter_dim 切片后 = 320，仍满足 128 对齐；但若涉及 V3 stage1 kernel 的 16/32/128 选择，BF16 在 tp=4 大 token 下需要 padding（见 635e59e）。FP8 因强制走 blockscale 2stage CK，不进入该 padding 路径。

**Review 结论**：✅ 推导正确。FP8 tp=2 / tp=4（inter_dim=640 或 320）均 128 对齐，无需 inter_dim padding。
**风险点**：若未来 tp=8（inter_dim=160）—— 仍 128 对齐 ✅；tp=16 才会跌破 128 对齐（80 % 128 ≠ 0），此时需重新评估。

---

## B. Experiment Design

### 实验 1 — Crash 复现验证（Fix 1 必要性）

**目的**：证明 q_type guard 缺失会导致 blockscale 路径在 V3 强制 block_m=128 上 TORCH_CHECK fail。

**操作**：
1. 临时回退 guard：编辑 `/home/hanchang/aiter/aiter/fused_moe.py` L906-907，将
   ```python
   if not run_1stage and inter_dim > 192 and get_gfx() == "gfx950" \
           and q_type not in (QuantType.per_1x128, QuantType.per_1x32):
   ```
   改为
   ```python
   if not run_1stage and inter_dim > 192 and get_gfx() == "gfx950":
   ```
2. `cd /tmp && rm -rf /root/.cache/atom/*`
3. 运行实验 2 的命令。

**预期**：FP8 forward 在 fused_moe 2stage dispatch 抛 TORCH_CHECK（block_m 不被 blockscale dispatch 支持）或产生 gibberish 输出。

**复原**：实验完成立刻恢复 guard 行（git checkout 该文件）。

**通过标准**：观察到 crash 或明确的 dispatch error；恢复 guard 后实验 2 通过。

---

### 实验 2 — FP8 tp=2 端到端（核心验证）

**命令**：
```bash
cd /tmp && rm -rf /root/.cache/atom/* && CUDA_VISIBLE_DEVICES=0,1 \
  AITER_LOG_LEVEL=WARNING \
  /opt/venv/bin/python -m atom.examples.simple_inference \
  --model stepfun-ai/Step-3.5-Flash-FP8 --trust-remote-code \
  --tensor-parallel-size 2 --level 0 --temperature 0 --max-tokens 1000 \
  --max-num-batched-tokens 4096 --max-num-seqs 2048
```

**通过标准**：
- 4 prompts 全部 EOS 正常完成
- TTFT < 150ms（参考值 ~91ms）
- TPOT < 25ms（参考值 ~16ms）
- 无 crash、无 NaN、无 gibberish；包含可验证内容（如质数列表正确）
- VRAM 实际占用 > 0（rocm-smi 复核）

**采集**：保存 stdout 到 `logs/V05_exp2_fp8_tp2.log`。

---

### 实验 3 — BF16 tp=2 回归（Fix 1 不影响 BF16）

**命令**：与实验 2 相同，模型替换为 `stepfun-ai/Step-3.5-Flash`（BF16）。

**通过标准**：
- TPOT ≈ 17ms（容差 ±2ms），TTFT ≈ 92ms（容差 ±15ms）—— 与 MEMORY.md 速查一致。
- 4 prompts 输出与历史 BF16 baseline 文本基本一致（temperature=0 决定确定性）。
- 关键：与 Fix 1 加入前的 BF16 输出做 byte-level diff，预期完全相同。

**采集**：`logs/V05_exp3_bf16_tp2.log` + diff 报告。

---

### 实验 4 — FP8 量化精度验证（可选，单层 cos_sim 对比）

**目的**：验证 FP8 fused_moe 输出与 BF16 reference 在单 MoE 层上的余弦相似度。

**步骤**：
1. 注入 hook 到 `Fp8MoEMethod.apply`（或在 `ModelRunner.run_model` 处包裹），dump 第 N 层 MoE 的输入/输出 tensor。
2. 同 prompt 同 seed 跑 BF16 一次，dump 同层输出。
3. 计算 `cos_sim(out_fp8, out_bf16)` 与 `max_abs_err`。

**通过标准**：cos_sim > 0.995；max_abs_err 在 FP8 量化噪声合理范围（依层 norm 而定，<0.1 量级）。

**注意**：需保证两次运行 router topk 一致（temperature=0 + 同 input embedding）。若 topk 不一致需做 expert-mask 后再比较。

---

### 实验 5 — block_shape 验证（Fix 2 影响路径）

**目的**：区分 `block_shape=None` 与 `[128,128]` 在 TP 路径下的实际差异。

**步骤**：
1. 临时把 ATOM moe.py L1816 改回 `block_shape = None`（保留 quant_type 判断）。
2. `rm -rf /root/.cache/atom/*` 后跑实验 2 命令。
3. 观察：
   - 是否在 fused_moe 入口抛 shape assertion / scale 校验失败；
   - 或 dispatch 走入错误 kernel 路径（用 AITER_LOG_LEVEL=INFO 看 `run_1stage = ... q_type = ... block_m = ...` 行）；
   - 或输出 gibberish（temperature=0 下与实验 2 byte-diff）。
4. 复原 L1816。

**通过标准**：能观测到三类异常之一（最优为明确 assertion，最弱为 cos_sim 显著下降）；若现象与实验 2 完全一致，说明此分支在 TP-only / 当前 wrapper 下被下游 inferred，需进一步追溯 fp8_w8a8_moe_quant_config 内部，并把结论补回此节。

---

## C. 关键问题（待回答）

### C.1 Step-3.5-Flash-FP8 模型在当前系统上是否可加载？路径是否正确？

- HuggingFace repo id：`stepfun-ai/Step-3.5-Flash-FP8`（见命令）。
- 本地缓存对照点：MEMORY 已存在 `stepfun-ai/Step-3.5-Flash` BF16 权重 (`snapshots/ab446a3.../`)。FP8 版本需检查 `~/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash-FP8/` 是否已下载；若无，首次运行会触发下载（需联网与 HF token 状态）。
- 预检命令（仅用 ls，不读取）：`ls ~/.cache/huggingface/hub/ | grep -i flash-fp8`。
- 若未下载：先 `huggingface-cli download stepfun-ai/Step-3.5-Flash-FP8` 再跑实验 2。

### C.2 block_shape=None vs [128,128] 在 TP 路径下是否实质区分？

- 仅靠 fused_moe_quant_config 看不出绝对差异；必须以实验 5 跑实测来区分。
- 如果实测「无差异」也是有价值的结论 —— 说明 Fix 2 是 EP/正确性兜底，不是 TP 当前的 must-have，但仍是 name-matches-function 原则下的正确改法（meta 信息显式化）。

### C.3 如何验证 kernel 实际走 a8w8blkscale 而非 a16w16？

- 在 `aiter/fused_moe.py` L921-923 已有 `aiter.logger.info(f"run_1stage = ... q_type = ... block_m = ...")`。
- 把环境变量改为 `AITER_LOG_LEVEL=INFO` 跑实验 2，应观察到：
  - `q_type = QuantType.per_1x128`
  - `run_1stage = False`（token > 32 时）
  - `block_m = 64`（blockscale 分支，L894-896）
  - 不出现 `block_m = 128`（说明 guard 生效，未被 V3 workaround 覆盖）。
- 进一步可在 fused_moe 入口加一行 print（仅本地诊断）打印实际调用的 kernel 名 / dtype，但代码侵入性较高，仅 PR-debug 阶段使用。

---

## D. 通过 / 失败矩阵

| 实验 | 通过条件 | 失败处理 |
|------|----------|----------|
| 1 | 观察到 crash / dispatch error | 若不 crash：说明对 inter_dim=640 触发条件理解有误，回查 `inter_dim > 192` 与 token 阈值 |
| 2 | TTFT<150 / TPOT<25 / 4-EOS / 无 gibberish | 检查 q_type log；回查 Fix 2 是否生效；检查 weight scale shape |
| 3 | 与历史 BF16 baseline byte-diff 为空 | Fix 1 引入回归，需重审 guard 是否误改其他分支 |
| 4 | cos_sim>0.995 | FP8 量化精度问题，可能与 weight_scale 加载或 router 不一致相关 |
| 5 | 观察到 shape assertion / kernel mismatch / gibberish 之一 | 若全无差异：将 C.2 结论补充回此文件 |

---

## E. 辅助文件路径

- aiter Fix 1：`/home/hanchang/aiter/aiter/fused_moe.py` L906-910
- ATOM Fix 2：`/home/hanchang/ATOM/atom/model_ops/moe.py` L1804-1827（Fp8MoEMethod.get_fused_moe_quant_config）
- 入口：`/home/hanchang/ATOM/atom/examples/simple_inference.py`
- 已有性能基线：MEMORY.md「Step-3.5-Flash 当前状态」表 + `memory/fp8-work.md`

## F. 执行顺序建议

1. C.1 预检模型缓存（避免实验时阻塞下载）
2. 实验 2（核心通过）→ 实验 3（BF16 回归）
3. 实验 1（必要性反向证明，会临时改代码并复原）
4. 实验 5（Fix 2 区分性验证）
5. 实验 4（精度选做）

每步保留 log 到 `/home/hanchang/project_fp8_tp4/verification_pipeline/logs/V05_*.log`，最终把通过/失败结果回填至本文件 D 节。
