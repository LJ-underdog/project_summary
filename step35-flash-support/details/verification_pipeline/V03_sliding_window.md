# V03 Sliding Window 验证计划

> 专题：`pa_decode_gluon.py` sliding-window decode mask 下界 off-by-one bug 修复验证
> 修复 commit：aiter `7ebae9afb`
> 关联文档：`03_sliding_window.md`（修复说明 & 初步实验结果）

---

## 0. Bug 摘要（Recap）

**位置**：`/home/hanchang/aiter/aiter/ops/triton/gluon/pa_decode_gluon.py` L1497–L1507（IS_CAUSAL=False 分支）

**修复前（buggy）**：
```python
else:                          # IS_CAUSAL == False
    causal_mask = qk_column_offsets[None, :] < sequence_end_idx
    if SLIDING_WINDOW > 0:
        causal_mask = causal_mask & (
            qk_column_offsets[None, :]
            >= sequence_start_idx + query_token_idx[:, None] + 1   # <-- 多出 +query_token_idx + 1
        )
```

**修复后（current HEAD, L1499-1507）**：
```python
else:
    causal_mask = qk_column_offsets[None, :] < sequence_end_idx
    if SLIDING_WINDOW > 0:
        causal_mask = causal_mask & (
            qk_column_offsets[None, :]
            >= sequence_start_idx + query_token_idx[:, None]       # 仅删除了 "+1"，保留 query_token_idx
        )
    else:
        causal_mask = causal_mask & (
            qk_column_offsets[None, :] >= sequence_start_idx
        )
```

> 注意：当前 HEAD 中下界为 `sequence_start_idx + query_token_idx[:, None]`（无 `+1`）。
> 对 decode 而言（`query_seq_len == 1`，`query_token_idx == 0`），等价于 `>= sequence_start_idx`，正好把 `[ctx-W, ctx)` 共 W 个 token 全部纳入窗口。
> bug 形式（`+ query_token_idx + 1`）让下界变成 `sequence_start_idx + 1`，第一个窗口 token 被切掉。

**Sequence index 来源（L1271-1273）**：
```python
if SLIDING_WINDOW > 0:
    sequence_start_idx = context_length - SLIDING_WINDOW
    sequence_end_idx   = context_length
```

---

## A. Code Review

### A.1 精确定位

| 项 | 值 |
|---|---|
| 文件 | `/home/hanchang/aiter/aiter/ops/triton/gluon/pa_decode_gluon.py` |
| Bug 行 | L1502（`>= sequence_start_idx + query_token_idx[:, None]`） |
| 所在分支 | `else:`（IS_CAUSAL=False） → `if SLIDING_WINDOW > 0:` |
| Helper：`sequence_start_idx` 定义 | L1272：`= context_length - SLIDING_WINDOW`（**可能为负**） |
| `query_token_idx` 定义 | L1479：`= qk_row_offsets // ONE_QUERY_GROUP_SIZE_POW2`，decode 阶段 == 0 |

### A.2 验证修复逻辑

**非 sliding 分支（L1505-1507）**：
```python
causal_mask = causal_mask & (qk_column_offsets[None, :] >= sequence_start_idx)
```
此时 `sequence_start_idx = page_size * sequence_split_idx`（L1280，非 sliding），是非负的 partition 起点，含义是「当前 partition 处理 `[start, end)` 的 KV」。语义正确：partition 内只看 partition 范围 token。

**sliding 分支（修复后 L1499-1503）**：
```python
causal_mask = qk_column_offsets[None, :] < sequence_end_idx
if SLIDING_WINDOW > 0:
    causal_mask &= (qk_column_offsets[None, :] >= sequence_start_idx + query_token_idx[:, None])
```
- decode（query_seq_len=1）：`query_token_idx[:, None] == 0` → 下界为 `context_length - SLIDING_WINDOW`，窗口长度 = W。**正确**。
- prefill / extend（query_seq_len > 1）：第 i 个 query 的有效 KV 窗口下界向右平移 i，是符合 sliding 定义的（每个 token 看自己向前 W 个）。**正确**。

**结论**：修复后的下界对 decode 与 prefill 都自洽。

### A.3 ctx < window_size 不受影响的原因

当 `context_length < SLIDING_WINDOW`（例 ctx=256, W=512）：
- `sequence_start_idx = 256 - 512 = -256 < 0`
- `qk_column_offsets >= 0`（block offsets 总是非负）
- `qk_column_offsets >= -256` 永远为 True
- 上界 `< sequence_end_idx == context_length` 单独决定有效区域 → 全部 KV 被纳入
- 与 bug 形式（`>= -255`）结果相同

**所以 V03 的 cos_sim 退化只在 ctx ≥ SLIDING_WINDOW 时显现**；对应实验 1 的预期：ctx ∈ {256, 511} 修复前后都 PASS，分界点在 ctx == W == 512。

> 边界细节：ctx == W（如 512 == 512）时 `sequence_start_idx == 0`。bug 形式下界为 `1`，正好把第 0 个 token 排除——这就是 summary 中 ctx=512 cos_sim=0.998982 的根因。

### A.4 prefill 路径不走此函数

- `pa_decode_gluon` 仅服务 paged attention **decode**（query_seq_len 通常 = 1，最多为 chunked prefill 的小 chunk）。
- prefill 走 `fmha_v3_varlen_fwd`（CK / asm 内核），sliding window 由该路径独立处理，不复用 L1502 的 mask 表达式。
- 因此本 bug 仅影响 decode；prefill 行为需独立验证（见 B.3 实验 3 端到端）。

---

## B. Experiment Design

所有实验在 8x MI350X (gfx950) + ROCm + `/opt/venv` 下执行。运行 python 必须 `cd /tmp &&`。

### 实验 1：ctx sweep cos_sim 验证（核心回归测试）

**目的**：确认修复在所有 ctx ≥ W 下恢复 cos_sim ≈ 1，且 ctx < W 不退化。

**配置**：
- 模型：Step-3.5-Flash bf16（含 sliding window attention 层）
- SLIDING_WINDOW = 512（Step-3.5-Flash 默认值，需确认）
- ctx ∈ {256, 511, 512, 513, 514, 1024, 4096}
- 对照：参考实现（HF / 非 paged decode 或 SDPA）vs aiter `pa_decode_gluon`
- 度量：每层 attention output 的 cosine similarity

**通过标准**：

| ctx | 修复前 cos_sim | 修复后 cos_sim | 期望 |
|-----|----------------|---------------|-----|
| 256 | ~1.0 | ~1.0 | 不受影响 |
| 511 | ~1.0 | ~1.0 | 不受影响（W-1） |
| 512 | 0.998982 (FAIL) | 0.999998 (PASS) | bug 边界，必须复现 |
| 513 | < 0.999 (FAIL) | 0.999998 (PASS) | |
| 514 | < 0.999 (FAIL) | 0.999998 (PASS) | |
| 1024 | < 0.999 (FAIL) | 0.999998 (PASS) | |
| 4096 | < 0.999 (FAIL) | 0.999998 (PASS) | 长 ctx 验证 |

**复现步骤草案**：
1. checkout aiter 至 `7ebae9afb^`（修复前父 commit），运行脚本，记录 cos_sim
2. checkout aiter 至 `7ebae9afb`（修复后），重新运行同脚本
3. diff 两轮结果，确认与上表一致

### 实验 2：decode 阶段专项（T=1, long context）

**目的**：在 query_seq_len=1（纯 decode）场景，验证最早窗口 token 被正确包含。

**配置**：
- 直接调用 `pa_decode_gluon` kernel（绕过模型）
- query_seq_len = 1
- context_length ∈ {512, 2048, 8192}
- SLIDING_WINDOW = 512
- 构造 KV cache 的第 `context_length - SLIDING_WINDOW` 位置（窗口最早 token）使用「有特征」value（其余位置为 0），观察 attention output 是否包含该 value 的贡献。

**通过标准**：
- 修复后：output 的相应 head 上能检出该特征值的非零贡献（softmax weight > 0）
- 修复前：该位置贡献为 0（被 mask）

### 实验 3：端到端推理（去掉 workaround）

**目的**：确认无需 `ATOM_STEP3P5_NO_SLIDING=1` 即可正确生成。

**配置**：
- ATOM 启动 Step-3.5-Flash bf16，tp=4
- **不**设置 `ATOM_STEP3P5_NO_SLIDING=1`
- prompt 长度跨越 sliding window：分别测试 ctx ≈ {600, 2048, 10000} 的输入
- 多轮 sample（temperature=0 + temperature>0），对比修复前后输出

**通过标准**：
- 修复后：输出连贯、无 "ungi" / "ungi ungi ..." 等乱码 pattern
- 修复后：与 BF16 reference（无 sliding workaround 的标准实现）输出语义一致
- 修复前：能稳定复现 "ungi" 乱码（作为 bug 存在性证据）

> 与 V01 / V02 的 tp=4 长序列 BOS bug（`memory/tp48-fixes.md` Open bug）解耦：本实验若必要可在 tp=2 跑，避免噪声。

### 实验 4：Regression（ctx < W 不退化）

**目的**：确保修复不影响短序列。

**配置**：
- 同实验 1 的 ctx ∈ {64, 128, 256, 511} 子集
- 对比修复前后 cos_sim 完全一致（diff < 1e-6）

**通过标准**：cos_sim 修复前后字节级别一致 / 数值差异 ≤ FP 噪声。

---

## C. 关键问题

### C.1 IS_CAUSAL=True vs False 路径差异

| | IS_CAUSAL=True (L1486-1496) | IS_CAUSAL=False (L1497-1507) |
|---|---|---|
| 上界 | `sequence_position_extension + qk_col < sequence_end_idx` | `qk_col < sequence_end_idx` |
| 下界（sliding） | `sequence_position_extension + qk_col >= sequence_start_idx` | `qk_col >= sequence_start_idx + query_token_idx`（修复后） |
| 用途 | 训练 / prefill-style decode（query 在 KV 之后，需要因果约束） | 纯 decode（query 视为 1 个 token，不需要 query 间因果） |
| 此 bug 是否影响 | **不影响**（该路径的下界没有 `+1`，且使用 sequence_position_extension 而非 query_token_idx） | **影响**：bug 在此分支 |

Step-3.5-Flash decode 走的是 IS_CAUSAL=False 路径（paged attention 的标准 decode 用法），故此 bug 命中。

### C.2 monkey-patch 实验（问题在 pa_decode_gluon）是否还需要额外验证？

monkey-patch 已在 03_sliding_window.md 中证明：把 `pa_decode_gluon` 替换为 reference 实现后乱码消失，定位充分。**不需要再独立验证**。但建议：
- 在实验 3（端到端）中保留一个 monkey-patch 对照组（reference impl）作为 ground truth，三方对比（buggy / fixed / reference）一次性闭环。

### C.3 prefill 路径是否需要独立测试？

**需要，但优先级 P2**：
- prefill 走 `fmha_v3_varlen_fwd`，与本修复无代码重叠。
- 但 sliding-window 的端到端正确性（实验 3）会同时覆盖 prefill+decode；如果实验 3 PASS，说明 prefill sliding 路径未破坏。
- 若实验 3 在 prompt 长度 > W 时仍有异常，需独立排查 `fmha_v3_varlen_fwd` 的 sliding window 参数处理（建议构造 prefill-only 单元测试：直接喂 ctx ∈ {600, 1024} 的 prompt，对比 HF reference logits）。

---

## D. 执行顺序建议

1. 实验 1（ctx sweep）→ 最基础回归，30 分钟内可完成
2. 实验 4（短 ctx regression）→ 与实验 1 同脚本扩展 ctx 列表
3. 实验 2（decode 专项）→ 单元级别证据
4. 实验 3（端到端）→ 用户可见效果验证
5. 视情况补 prefill 独立测试

---

## E. 已知风险 / 注意事项

- **GPU5 硬件异常**：tp=8 不要用；tp=4 可能受 BOS bug 干扰，建议 sliding 验证主跑 tp=2 + bf16，避免与 tp=4/FP8 的其他 open bug 混淆。
- **缓存清理**：修改 aiter / ATOM 后需 `rm -rf /root/.cache/atom/*`。
- **commit 切换**：在 `/home/hanchang/aiter` 下做 checkout，但任何 commit/push 必须从 `/home/hanchang/junlin12_repos/` 仓库执行。
- **SLIDING_WINDOW 取值**：实验前必须从 Step-3.5-Flash config 确认实际 W；本计划假设 W=512，若实际不同需按比例调整 ctx sweep 边界。

---

## F. 交付物

完成验证后输出：
- ctx sweep cos_sim 表格（实验 1+4）
- decode 专项的 attention weight 可视化或数值证据（实验 2）
- 端到端生成样本对比（buggy vs fixed vs reference，实验 3）
- 结论：是否可以从 ATOM 中删除 `ATOM_STEP3P5_NO_SLIDING=1` workaround 代码
