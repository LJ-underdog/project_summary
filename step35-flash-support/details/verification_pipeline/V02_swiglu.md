# V02 — SwigluStep Wiring 验证计划

**专题**：02_swiglu_step.md（ATOM commit `4a8495e`，aiter `6d70f7b54`）
**前置依赖**：V01（MoE Pipeline 修复）必须先验证通过
**编制日期**：2026-04-25
**编制方**：V02-SwigluStep 专题组（Code Reviewer + Experiment Designer）

---

## 0. 修复一句话总结

ATOM `Step3p5MoE.__init__` 之前只把 `self.clamp_limit` 存为字段，但**没有传给 FusedMoE / CK kernel**，导致 layer 43-44 实际走 plain silu，clamp 被丢弃。修复在构造 `FusedMoE` 时根据 `clamp_limit` 决定 `activation=ActivationType.SwigluStep | ActivationType.Silu`，并通过 `_fuse_shared_at_layer` 强制 SwigluStep 层的 shared expert 走 dense path。

---

## A. Code Review

### A.1 `clamp_limit` → `activation_type` 传递链

**审查范围**：`/home/hanchang/ATOM/atom/models/step3p5.py`

| 步骤 | 代码位置 | 检查点 |
|------|----------|--------|
| 1. 读取 per-layer 限制 | L184-189 `swiglu_limits[layer_idx]` | swiglu_limits 必须按 `layer_idx` 索引；越界或 `<=0` 时回落为 `None` |
| 2. 决定 activation 枚举 | L190-193 `self._activation = SwigluStep if uses_swiglustep else Silu` | 二选一；不允许第三种值 |
| 3. 传给 FusedMoE | L223-235 `FusedMoE(..., activation=self._activation)` | 关键修复点：必须出现 `activation=` kwarg |
| 4. FusedMoE 透传到 CK | aiter `fused_moe(activation=...)` 链路 | 见 A.3 实证验证 |

**审查结论（待实验确认）**：
- ✅ wiring 看起来完整：`clamp_limit` → `_activation` → `FusedMoE(activation=...)`
- ⚠️ 但 ATOM 端只到 `FusedMoE` 入口，是否真传到 CK kernel 需 A.3 + 实验 5 验证
- ✅ swiglu_limits 默认值处理（`getattr(config, "swiglu_limits", None)`）安全

**风险点**：
- **R-A1**：`extract_layer_index(prefix)` 失败时 `layer_idx=None`，会绕过 SwigluStep 路径（实际走 Silu）。layer 43-44 的 prefix 必须能正确解析出 layer index。验证：实验 5 的 weight check。
- **R-A2**：`swiglu_limits[layer_idx] > 0` 这一条件依赖配置文件正确性。若配置文件错把 layer 43 的 clamp 写成 0，则静默退回 Silu，无报错。
- **R-A3**：L177-181 注释明确指出 "CK kernel 实现为 SwigluStep with hard-coded ±7 clamp"。**`clamp_limit` 数值本身（7）并未传给 kernel，仅作为开关使用**。这意味着如果未来某层使用 ≠7 的 clamp，会静默错误。需要 defensive assertion（见 E.1）。

### A.2 Shared Expert 不接 SwigluStep 的逻辑

**审查范围**：`step3p5.py` L75-90 (`_fuse_shared_at_layer`)，L215-235，L549-576。

**核心约束**：
- routed expert clamp = 7（CK kernel 硬编码）
- shared expert clamp 不同（layer 43: 0=禁用；layer 44: 16）
- 若 fuse 进 FusedMoE，会被强制使用 ±7，与 HF 行为不符

**代码实现**：
```
_fuse_shared_at_layer(config, layer_idx):
    SwigluStep 层 → False（禁止 fuse，shared 走 dense path）
    其他层 → 默认行为（受 ATOM_FORCE_FUSE_SHARED 等环境变量影响）
```

L559-566 在 SwigluStep 层走 `Step3p5MLP(clamp_limit=clamp_limit_shared)`，把 shared expert 的不同 clamp 限制走纯 PyTorch dense 路径（L130-141 `forward` 用 `torch.nn.functional.silu(...).clamp(...)`）。

**审查结论**：
- ✅ 设计正确：SwigluStep 层的 shared expert 强制走 dense path（L220 `self._fuse_shared = _fuse_shared_at_layer(config, layer_idx)`），并在 weight load（L862-866）也阻断 share_expert 进 FusedMoE。
- ⚠️ L83-84 提到 `ATOM_FORCE_FUSE_SHARED=1` 是 verification helper，但若实验中误开会破坏修复。验证脚本必须显式 unset 该变量或在日志中确认。
- ⚠️ shared expert 的 clamp_limit 数值（7/16/0）来自 `swiglu_limits_shared`，必须与 HF config 完全一致。

### A.3 BOS-spam "bf16 噪声累积" 结论的合理性

**summary 给的 bisection 表**：

| 变体 | SwigluStep 层 | BOS-spam 命中率 |
|------|--------------|-----------------|
| baseline | 无 | 0/4 |
| layer-44 only | {44} | 1/4 |
| layer-43 only | {43} | 2/4 |
| 两层均开 | {43, 44} | 3/4 |

**审查结论 — 当前结论"噪声累积"证据强度**：
- ⚪ **效应叠加**（layer 数越多越严重）确实符合"噪声累积"假说，但同样符合"layer 43-44 kernel 在某些 outlier 输入下数值偏差"假说，**不能区分**。
- ⚪ Phase G 的 cos_sim=0.999989 是在合成输入（scale=0.5~5.0）和真实权重组合下测的，**不等于真实 prompt decode 路径上的输入分布**。outlier path 可能未被覆盖。
- ⚠️ "bf16 精度上限"作为最终解释，需要量化证据：相同 RNG 下 plain silu 与 SwigluStep 在 200+ decode steps 后的 KL/perplexity 差异是否同量级。

**结论**：当前判断**方向正确但证据不充分**，需要实验 4 + 实验 6 加强。

---

## B. Experiment Design

> 所有实验必须在 V01 验证通过后再启动。环境：`/opt/venv` Python，必须 `cd /tmp &&` 启动；避开 GPU 5。
> 每个实验运行前：`rm -rf /root/.cache/atom/*` 并清理 stale `.so`：`rm -f aiter/jit/module_moe_ck2stages_*swiglustep*.so && rm -rf aiter/jit/build/module_moe_ck2stages_*swiglustep*`。

---

### 实验 1：op_test 验证 SwigluStep kernel 正确性

**目的**：在 kernel 层确认 SwigluStep CK 路径与 HF reference 数学等价。

**命令**：
```bash
cd /tmp && python -m aiter.test_moe_2stage \
    --activation swiglustep --preshuffle 1 \
    --M 32 --N 1024 --K 7168 --E 288 --topk 8 \
    --seed 0 --dtype bf16
```

**变量**：
- `M ∈ {1, 8, 16, 32, 64, 256, 1024}`（覆盖 V1/V3 kernel 切换边界 block_m=128）
- `preshuffle ∈ {0, 1}`（preshuffle_off 在 bf16 下 codegen 实际等价，但仍要验证）
- `seed ∈ {0, 1, 42}`

**通过标准**：
- 所有组合 `cos_sim ≥ 0.99998`（与 summary 中 `0.999989` 一致）
- `max_abs_err ≤ 0.05`（bf16 量级合理）

**失败处理**：
- 若 cos_sim=0.0 → stale .so，按上面命令清理后重跑
- 若 cos_sim≈0.99 但低于阈值 → 检查 CK kernel 内 clamp 常量（grep `7.0f` in `gridwise_moe_gemm.hpp`）

---

### 实验 2：层级验证（layer 43-44 真实权重 + 多 scale）

**目的**：验证 SwigluStep 在真实权重 + 不同输入幅度下的精度。

**脚本**：基于 summary 提到的 Phase G 方法（`D_check_weights.py` + 单层 forward 对比）。

**矩阵**：
- layer ∈ {43, 44}
- `M ∈ {16, 64, 256, 1024}`
- `scale ∈ {0.5, 2.0, 5.0, 8.0}`（**新增 scale=8.0** 以测试超出 ±7 clamp 范围的 outlier 行为）
- 对比对象：HF reference `silu(gate).clamp(max=7) * up.clamp(±7) → down_proj`

**通过标准**：
- scale ≤ 5.0 各组合 `cos_sim ≥ 0.99998`
- scale = 8.0（深度 clamp）`cos_sim ≥ 0.9999`（允许略低，因 clamp 引入截断不可微）

**额外采集**：
- 每组合记录 `max_abs_err`、`rel_l2`，用于实验 6 的累积误差建模

---

### 实验 3：端到端推理 max_tokens=128（regression baseline）

**目的**：复现 summary 的"tp=2 4 prompts max_tokens=128 全部正常"结论。

**命令**：
```bash
cd /tmp && python -m atom.examples.simple_inference \
    --model /root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash/snapshots/ab446a3.../ \
    -tp 2 --max_tokens 128 \
    --prompts "1+2+3=?" "What is the capital of France?" \
              "Write a haiku about GPUs." "解释一下注意力机制。"
```

**通过标准**：
- 所有 4 prompt 输出语义合理（人工 review）
- "1+2+3=?" 给出 6 的正确答案
- TTFT/TPOT 在 baseline ±10% 内（baseline = MEMORY.md 中 tp=2 bf16: TTFT=92ms, TPOT=17ms）
- 无 BOS token 重复 spam

**对照组**：
- A：禁用 SwigluStep（环境变量 hack 把 layer 43-44 退回 Silu），看是否仍通过 → 若通过说明短输出下两种 activation 都能给出合理结果（已知现象）
- B：启用 SwigluStep（main 修复路径）

---

### 实验 4：BOS-spam 可控复现（max_tokens=512+）

**目的**：复现 summary 的 bisection 表，确认效应可重复且与 SwigluStep 层数正相关。

**矩阵**（每行 4 prompts × 3 seeds = 12 runs）：

| 配置 | SwigluStep 启用层 | 预期 BOS-spam 命中 |
|------|-------------------|-------------------|
| C0 | 无（all Silu） | 0/12 |
| C1 | {44} only | ~3/12 |
| C2 | {43} only | ~6/12 |
| C3 | {43, 44}（main） | ~9/12 |

**实施方式**：通过环境变量 `ATOM_DISABLE_SWIGLUSTEP_LAYERS="43,44"` / `"43"` / `"44"` / `""` 控制（**若该 env 不存在，需在 step3p5.py 加临时实验开关**）。

**`max_tokens=1024`，温度 0.7（保留随机性）**

**通过标准**：
- 趋势单调：C0 < C1 ≤ C2 < C3 的 spam 率
- C3 的 spam 率与 summary 报告（3/4 ≈ 75%）一致 ±20%
- 若 C0 也出现 spam → 推翻"SwigluStep 引起"，需重新调查

**失败处理**：
- 若 C1/C2 出现 0 spam，可能是 seed 选择问题，扩大到 12 seeds 重测
- 若 C3 ≫ C0 但 cos_sim 极高（已由实验 2 确认） → 强化"噪声累积"假说

---

### 实验 5：CK kernel `_activation` 透传验证

**目的**：回答关键问题 3 — 确认 ATOM 传入的 `activation` 不被链路上某处 override。

**方法 A：日志注入**
- 在 `aiter/fused_moe.py` 入口打印 `activation` 参数
- 在 CK kernel JIT codegen（`module_moe_ck2stages_*.py`）打印 `activation` 字符串
- 跑一次推理，grep 日志确认 layer 43/44 出现 `SwigluStep`，其他层出现 `Silu`

**方法 B：codegen artifact 检查**
- 推理后检查 `aiter/jit/build/` 下生成的 `.so` 文件名，应同时包含 `*silu*.so` 和 `*swiglustep*.so`
- 若只有 silu 文件 → activation 未透传，wiring 失败

**方法 C：weight check 脚本**
- 复用 summary 提到的 `D_check_weights.py`：每层 print `_activation` 字段
- 期望输出：layer 0-42 + layer 45+ → Silu；layer 43-44 → SwigluStep

**通过标准**：三种方法均确认 layer 43-44 走 SwigluStep 分支。

---

### 实验 6（可选 / 推荐）：BOS-spam "bf16 噪声累积" 量化验证

**目的**：把"噪声累积"从定性结论升级为定量论证，回应关键问题 2。

**方法**：在 `model_runner.py` 调用点 hook 每层 hidden_state，逐 token decode 时记录：
1. SwigluStep ON vs OFF 两次推理（同 prompt，同 seed）
2. 每个 decode step `t`，每层 `l`，记录 `cos_sim(h_on[t,l], h_off[t,l])`
3. 绘制 cos_sim 随 t 的衰减曲线

**预期**：
- 若是噪声累积：cos_sim 在 layer 43-44 处一次性下降，但每层下降幅度恒定，整体 cos_sim 随 t 单调衰减（线性 / 慢指数）
- 若是 kernel bug：cos_sim 可能在某些 t（特定 outlier 输入）出现陡降，非平滑衰减

**通过标准**：cos_sim(t) 单调下降且对 t 的导数有上界 → 支持"噪声累积"。

**说明**：此实验工程量较大，列为可选。但若决策需要"高置信度"，应该执行。

---

### 实验 7：shared expert 不接 SwigluStep 的反向验证

**目的**：回答关键问题 1 — 验证 shared expert 不进 FusedMoE 这个决策的必要性。

**正向**：当前 main 路径（shared expert 走 dense）已由实验 3、4 覆盖。

**反向（破坏测试）**：
- 强制开启 `ATOM_FORCE_FUSE_SHARED=1`，让 layer 44 的 shared expert 也进入 FusedMoE（被迫使用 ±7 clamp 而非 ±16）
- 跑实验 3 的 prompt 集合
- 期望：输出质量明显下降（cos_sim 降至 ~0.99 或人工可感知错误）

**通过标准**：
- 反向实验确实劣化 → 证明 `_fuse_shared_at_layer` 的存在必要
- 若反向实验输出仍正常 → 决策可能过度防御，但**不必移除**（保留是安全的）

---

## C. 依赖和顺序

```
V01 (MoE Pipeline) ✅
   │
   ├─→ 实验 1 (op_test) ─┐
   ├─→ 实验 2 (层级验证) ─┤
   ├─→ 实验 5 (透传验证) ─┴─→ 实验 3 (e2e 短输出)
   │                              │
   │                              └─→ 实验 4 (BOS-spam 复现)
   │                                         │
   │                                         └─→ 实验 6 (噪声累积量化, 可选)
   │
   └─→ 实验 7 (shared expert 反向, 独立可并行)
```

**关键串行点**：
- 实验 1 必须先于 3/4（kernel 不正确则 e2e 无意义）
- 实验 5 必须先于 3/4（确认 wiring 真到 kernel）
- 实验 4 依赖实验 3 通过（短输出 OK 才有意义讨论长输出问题）

**并行机会**：实验 1、2、5、7 可并行启动。

---

## D. 通过标准（Exit Criteria）

V02 整体通过当且仅当：

1. **Kernel 正确性**：实验 1 全部 cos_sim ≥ 0.99998
2. **真实权重精度**：实验 2 中 scale ≤ 5.0 的全部组合 cos_sim ≥ 0.99998
3. **Wiring 正确**：实验 5 三种方法均确认 layer 43-44 路径为 SwigluStep
4. **短输出 e2e**：实验 3 全部 prompt 输出合理，性能 ±10%
5. **长输出可解释**：实验 4 的 spam 率与 SwigluStep 层数单调相关，且 baseline (C0) spam=0
6. **设计决策可辩护**：实验 7 反向实验确认 shared expert 不能 fuse 进 SwigluStep

可选加分：实验 6 给出 cos_sim(t) 衰减曲线，定量支持"噪声累积"结论。

---

## E. 待确认问题

### E.1 `clamp_limit` 数值未传给 kernel —— 是否要加防御性 assert？

CK kernel 硬编码 ±7，ATOM 只用 `clamp_limit > 0` 作为开关。若未来某层 `swiglu_limits[l] = 6`，会被静默当作 7 处理。

**建议**：在 `Step3p5MoE.__init__` 加 `assert self.clamp_limit in (None, 7.0), "CK kernel hard-codes ±7"`。需要在验证完成后向 owner 提议。

### E.2 BOS-spam 是否应当直接修复？

当前结论是"已知问题，bf16 精度上限不可避免"。但：
- 是否能用 fp32 accumulation 在 down_proj 后缓解？
- 是否能用 logit processor 在 inference 端 mask 异常 BOS？
- 这是否在 V02 范围内？

**建议**：V02 范围内**仅验证可解释性**，修复方案放到独立专题（潜在 V08）。

### E.3 实验 4 的 `ATOM_DISABLE_SWIGLUSTEP_LAYERS` 是否已存在？

如果不存在，需要先加临时实验开关（不进 main，仅验证用）。需向 owner 确认。

### E.4 BOS-spam bisection 的 prompt 集是否固定可复现？

summary 中"3/4 prompt"未列出具体 4 个 prompt。需要拿到原始 prompt 集才能严格复现。

### E.5 cos_sim=0.999989 是否真是 bf16 精度上限？

需要做对照：跑两次 `silu(gate.bf16) * up.bf16` 但用不同 fp32 reference path 的 cos_sim 上限值，确认 0.999989 不是被某个常数因子抑制。

---

## 附录：关键文件路径

| 文件 | 用途 |
|------|------|
| `/home/hanchang/ATOM/atom/models/step3p5.py` | wiring 主体；L177-235 是修复区，L75-90 是 fuse helper |
| `/home/hanchang/project_summary/step35-flash-support/02_swiglu_step.md` | 修复 summary（本计划的输入） |
| `/home/hanchang/aiter/op_tests/test_moe_2stage.py`（待确认路径） | 实验 1 的 op_test 入口 |
| `/home/hanchang/aiter/csrc/include/ck_tile/.../gridwise_moe_gemm.hpp`（待确认路径） | CK kernel 中 SwigluStep 实现，含硬编码 7.0 |
| `aiter/jit/module_moe_ck2stages_*swiglustep*.so` | 实验前需清理的 stale codegen artifact |
