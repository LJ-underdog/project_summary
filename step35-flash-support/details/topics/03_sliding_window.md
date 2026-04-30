# 子任务 3：Sliding Window Attention 修复

**日期**：2026-04-23
**状态**：✅ 完成
**commit**：aiter `7ebae9afb`

---

## 1. 背景

Step-3.5-Flash 约 3/4 层使用 sliding window attention（window=512），其余层用 full attention。
在 MoE pipeline 修复完成后（`ATOM_STEP3P5_NO_SLIDING=1` workaround 下），tp=2 推理输出正常。

去掉 workaround 后（启用 sliding window），输出含 "ungi" 等乱码字符，多种 prompt 均触发。

---

## 2. 调查过程

### 2.1 隔离 kernel 问题

**monkey-patch 实验**：修改 `dispatch_backend`，decode 阶段强制走 `paged_attention_asm`
（保留 sliding_window 配置不变），"ungi" 消失。

→ 问题在 `pa_decode_gluon` Triton kernel 本身，不在 KV cache 管理或 sliding_window 配置。

### 2.2 排除 CDNA 版本差异

gfx950 属于 CDNA4，检查 `pa_decode_gluon.py` 中 CDNA_VERSION==4 的分支：

- 读 rocm-ref 文档（`/home/hanchang/agent_skill/rocm-ref.2026.03.25.gz`）
- 确认 CDNA3/CDNA4 的 `buffer_load` / `ds_bpermute` 语义相同
- 实验：CDNA3/CDNA4 patch 结果数值完全一致

→ 排除 CDNA4 路径差异。

### 2.3 定位精确 bug

仔细读 `pa_decode_gluon.py` L1499-1507（`IS_CAUSAL=False` 分支，decode 路径）：

```python
if SLIDING_WINDOW > 0:
    causal_mask = causal_mask & (
        qk_column_offsets[None, :]
        >= sequence_start_idx + query_token_idx[:, None] + 1  # ← 多了 +1
    )
```

非 sliding 分支：`>= sequence_start_idx`（无 +1）。

decode 阶段 `query_token_idx=0`，sliding 分支下界变为 `sequence_start_idx + 1`，
实际把 `sequence_start_idx` 处的 token 排除在窗口之外，窗口有效长度变成 `SLIDING_WINDOW - 1`。

### 2.4 实验验证

用不同 context_length 跑精度测试（cos_sim 对比 full attention reference）：

| ctx_len | 修复前 cos_sim | 修复后 cos_sim |
|---------|----------------|----------------|
| 508-511 | 0.999998 PASS | 0.999998 PASS |
| **512** | **0.998982 FAIL** | **0.999998 PASS** |
| **513-516** | **0.998942~0.999016 FAIL** | **0.999998 PASS** |
| 1024 | 0.999092 | **0.999998 PASS** |

ctx≥512（sliding window 开始生效）时 FAIL，修复后全部 PASS。"ungi" 乱码消失。

---

## 3. 根因

**文件**：`aiter/aiter/ops/triton/gluon/pa_decode_gluon.py`
**位置**：L1502，sliding window decode path

**问题**：mask 下界应为 `>= sequence_start_idx`，
代码多加了 `+ query_token_idx[:, None] + 1`（decode 时 query_token_idx=0，
等效多加了 `+ 1`），导致 sliding window 的最早一个 KV token 被错误 mask 掉。

**为何 ctx < window_size 不受影响**：
此时 `sequence_start_idx = context_length - SLIDING_WINDOW < 0`，
条件 `>= negative_value + 1` 对 `qk_column_offsets >= 0` 始终成立，无 token 被 mask。

---

## 4. 解决方案

**文件**：`aiter/aiter/ops/triton/gluon/pa_decode_gluon.py` L1502

```python
# 修复前
>= sequence_start_idx + query_token_idx[:, None] + 1

# 修复后
>= sequence_start_idx
```

删除多余的 `+ query_token_idx[:, None] + 1`。

---

## 5. 验证结果

| 验证项 | 结果 |
|--------|------|
| ctx=512~1024 cos_sim | 0.999998 PASS（全部） |
| "ungi" 乱码 | 消失 |
| baseline（ctx<512）不退化 | cos_sim=0.999998 PASS |
| tp=2 端到端推理（不加 workaround） | 正常输出 |

workaround `ATOM_STEP3P5_NO_SLIDING=1` 不再需要。

---

## 6. 教训

| 教训 | 说明 |
|------|------|
| monkey-patch 隔离法 | 快速定位问题在哪个 kernel，无需读全部代码 |
| ctx sweep 精确定位 | 不同 ctx_len 的 cos_sim 变化准确指向 sliding window 生效点（=512） |
| 非 sliding 路径对比 | 对比 sliding 分支和非 sliding 分支的代码差异，直接找到多余的 +1 |
