# TP=8 Root Cause and Fix — Step-3.5-Flash-FP8 on MI308X

> 项目：fp8-tp4-repro
> 模型：`stepfun-ai/Step-3.5-Flash-FP8`
> 平台：gfx942 / MI308X (8 GPU/节点)
> 修复 commit：ATOM `969d564` (branch `feat/step3p5-flash-support`，单 commit / 31 行 / 可 revert)
> 日期：2026-04-29
> 受众：项目 owner（5 分钟读完即可掌握 root cause + fix + verify 证据链）

---

## 1. TL;DR

- **Bug**：tp=8 时 fp8 MoE 权重加载在 `_load_w13` / `_load_w2` 的 `narrow()` 上崩溃；早返回绕开后，trailing rank 的 fp32 scale tensor 保持 `torch.ones()` 初值 1.0，导致下游 fp8 dequant `bf16 = fp8 * 1.0` 错算 → 4/4 prompt 输出乱码。
- **Fix**：双层 patch — (a) `start >= D` 时 early-return 跳过 `narrow + copy_`；(b) 同分支内对 `dtype == float32` 的 scale tensor 调用 `.zero_()`，让 trailing rank 在 RowParallel reduction 中贡献严格 0。tp=2/4/8 三档 A1-A4 全 PASS。

---

## 2. 症状演化矩阵

3 种 patch 形态 × 3 个 tp 档位（参考 `fix_wave/WAVE_CLOSE.md` §1.3 + §6.1 + `fix_wave/progress/teammate-8.md`）：

| | **tp=2** | **tp=4** | **tp=8** |
|---|---|---|---|
| **修复前**（无 patch）| ✅ load OK / 输出 coherent | ✅ load OK / 输出 coherent | ❌ `_load_w2 narrow()` size<0 raise → 进程 crash |
| **early-return only**（lead 17L）| ✅ load OK / 输出 coherent (`tp2_attempt2.log`) | ✅ load OK / 输出 coherent (`tp4_attempt2.log`) | ✅ load OK + 推理跑通；❌ 4/4 prompt 完全乱码（`tp8_attempt2.log` / `outputs/tp8.json`，例：`"小弟sets邪倾倒uiropa盐尚书arms cryptic..."`）|
| **双层 fix**（commit `969d564` / 24L 净增）| ✅ load OK / 输出 coherent / A3 byte-id 锚点保持（`final_tp2.{log,json}`）| ✅ 同上（`final_tp4.{log,json}`，starts=[0,3,6,9] 全 < D=10 → zero-fix 不 trigger）| ✅ load OK + 4/4 prompt **coherent**，与 tp=2/4 同质量（`final_tp8.{log,json}`，例：`"Hmm, the user asked me to introduce myself..."`）|

> tp=2/4 在双层 fix 下 zero-fix 分支不触发（D=10，starts 全 < D），等价于 early-return-only 行为，无回归（`fix_wave/progress/teammate-4.md` §V1 + `fix_wave/progress/teammate-8.md` §3）。

---

## 3. 根因

### 3.1 第一层：weight-load crash（`narrow` size<0 / shape mismatch）

`atom/model_ops/moe.py` `_load_w13` 与 `_load_w2` 用 ceil 公式切 inter 维度（来源：当前 commit `969d564` 上下文 L2310-L2313 / L2372-L2375）：

```python
load_shard_size = (D + tp_size - 1) // tp_size       # ceil(D / tp_size)
start           = load_shard_size * tp_rank
```

ceil 是为了不丢 partial scale block（注释见源码 L2306-L2309，原意：避免 trailing partial block 留 init 1.0）。但 `start` 可能 ≥ D，于是 `size = D - start < 0` 让 `narrow(start, size)` 直接 raise。

**触发参数实测**（来源：`fix_wave/WAVE_CLOSE.md` §1.1 + `correctness_eval/CORRECTNESS_REPORT.md`）：

- step3p5-Flash-FP8 是 `per_1x128` block-quantized FP8，每 128 通道一个 fp32 scale；inter=1280 → **D = 10 个 scale block**
- tp=8 时 `ceil(10/8) = 2`，`starts = [0, 2, 4, 6, 8, 10, 12, 14]`
- rank 5 命中 `start = 10 == D`（symptom B：`size=0` 后下游 `copy_` shape mismatch）
- rank 6/7 命中 `start ∈ {12, 14} > 10`（symptom A：`narrow` size<0 raise）
- tp=2 starts=[0,5]、tp=4 starts=[0,3,6,9]：全部 < D → 安全（实测 zero-trigger，见 `fix_wave/progress/teammate-4.md` §V1）

崩溃点：`_load_w2 narrow()`（commit `969d564` 之前对应 `atom/model_ops/moe.py:2357`），同形态 `_load_w13`（同 commit 之前 L2315）。

### 3.2 第二层：silent corruption（fp8 scale `torch.ones()` init 未被覆盖）

仅做 early-return 让 weight load 通过，但 4/4 prompt 输出乱码（见 §2 矩阵）。teammate-5 layer-by-layer dump 实验定位（来源：`fix_wave/progress/teammate-5.md` §1.1, §2.1, §4.1）：

1. `Fp8MoEMethod._create_weights` 用 `torch.ones(...)` 初始化所有 fp32 scale tensor（实测在 `~/ATOM/atom/model_ops/moe.py` L1621/1628/1634/1642/1655/1658）
2. early-return 让 trailing rank 的 scale slice **保持 1.0**
3. forward 期 fp8 dequant 是乘法：`bf16 = fp8_w * scale`。trailing rank 用 scale=1.0 把「未拷贝的 fp8 raw bits」直接当 bf16 使用，数值错乱 4-5 个数量级
4. RowParallel reduction `output = Σ rank_output * router_weight` 把错算的 trailing rank 贡献汇进结果 → 输出分布跑偏 → argmax 选到噪声 token → 乱码
5. Step-3.5-Flash 的 dense layer 0-2 + MoE layer 3-44（来源：`~/ATOM/atom/models/step3p5.py:14-16`），**第一发散 layer = 3**（首个 MoE 层）

**trailing rank early-return 触发覆盖**（来自 teammate-5 dump，单次 `--max-tokens 1` run）：

| rank | `_load_w13` 早返回次数 | `_load_w2` 早返回次数 |
|---:|---:|---:|
| 5 | 1728 | 864 |
| 6 | 5184 | 2592 |
| 7 | 2304 | 1152 |

→ 整模型 42 个 MoE layer 累积错误（每个 trailing rank 的 scale tensor 都有未覆盖区）。

dump 也直接证立："所有 early-return 的 `expert.dtype` 都是 `torch.float32`" — 早返回 **100% 落在 scale tensor**，从未落在 fp8/bf16 weight tensor（来源：`fix_wave/progress/teammate-5.md` §2.2）。

---

## 4. 修复方案

### 4.1 双层 fix（commit `969d564` diff 全文）

文件：`atom/model_ops/moe.py`，patch 共 31 行新增 / 单 hunk × 2 函数：

```python
# _load_w13 (commit 969d564 patch hunk @@ -2311,6 +2311,26 @@)
load_shard_size = (
    loaded_weight.shape[shard_dim] + self.tp_size - 1
) // self.tp_size
start = load_shard_size * tp_rank
# When D < tp_size (e.g. per_1x128 scale block count smaller than
# tp_size, observed at tp=8 with inter=1280 → D=10), the ceil split
# gives some trailing ranks start >= D so they hold no slice of the
# loaded tensor. Skip narrow + copy_ for those ranks; the rank's
# slice of expert_data stays at its initialised value (0 for weight,
# 1.0 for scale) and the rank contributes a no-op to the column
# gather / row reduction.
if start >= loaded_weight.shape[shard_dim]:
    # FP8 scale tensors are torch.ones() initialised. If we leave the
    # trailing rank's slice at 1.0, the downstream FP8 dequant multiplies
    # the (uninitialised) fp8 weight by 1.0 instead of the correct
    # quantization scale, contaminating the column gather / row reduction.
    # Zero the slot so dequant produces 0 and the rank contributes a
    # true no-op (matches MXFP4 scale init at moe.py:776,813).
    if expert_data.dtype == torch.float32:
        if shard_id == "w1":
            expert_data.narrow(shard_dim, 0, expert_shard_size).zero_()
        else:
            expert_data.narrow(shard_dim, expert_shard_size, expert_shard_size).zero_()
    return

# _load_w2 (commit 969d564 patch hunk @@ -2353,6 +2373,17 @@)
load_shard_size = (
    loaded_weight.shape[shard_dim] + self.tp_size - 1
) // self.tp_size
start = load_shard_size * tp_rank
# See _load_w13 comment above: when D < tp_size the ceil split
# leaves trailing ranks with no slice; skip narrow + copy_.
if start >= loaded_weight.shape[shard_dim]:
    # Zero the scale slice so dequant=0 instead of multiplying by
    # stale init=1.0; see _load_w13 comment for full rationale.
    if expert_data.dtype == torch.float32:
        if load_shard_size != shard_size:
            expert_data.narrow(shard_dim, 0, load_shard_size).zero_()
        else:
            expert_data.zero_()
    return
```

**语义闭环**（参考 `fix_wave/progress/teammate-5.md` §4.2）：

- weight tensor early-return 时 init=0，dequant `0 * scale = 0`，无害
- scale tensor early-return 时若保持 init=1.0，dequant `random_fp8_bits * 1.0` ≠ 0 → 污染
- 把 scale 也置 0 后 `0 * 0 = 0`，trailing rank 在 RowParallel reduction 中贡献严格 0（等价于"该 rank 该 inter 区段不存在"）

### 4.2 为什么不是方案 B / C？

来源：`issue_wave/ATOM_ISSUE_DRAFT.md` §"Proposed fix" + `fix_wave/WAVE_CLOSE.md` §4.3。

| 方案 | 描述 | 是否采纳 | 理由 |
|---|---|---|---|
| **A** early-return + zero-fix（本 commit）| 跳过 narrow + zero scale slot | ✅ | 最小改动；A1-A4 实测全 PASS；行为局限在 trailing rank 2 函数 |
| **B** 余数挂 rank0（修改 split 分布）| 总保证 `start + size <= D`，partial block 落到 rank0 | ❌ | 改动 split 分布，issue draft 自标 "kernel 是否容忍未审计"；A 双层已 PASS 无须 |
| **C** D<tp_size 时降级 effective_tp_size = D | 把 trailing rank 整体不参与 | ❌ | 需改 dispatch 逻辑 + cudagraph capture + TP all-reduce 全套；改动量太大 |

**容量损失评估**：tp=8 trailing 失去 inter slice ≈ `(8*2 - 10)/8 = 0.75 block ≈ 9.4%` 有效 inter 容量（仅 rank 5/6/7）。teammate-5 §3.2 实测 4/4 prompt coherent，无可观察精度影响。

### 4.3 与 MXFP4 的设计一致性

ATOM 自家 MXFP4 path 的 scale tensor 用 `torch.zeros(...)` 初始化（来源：`atom/model_ops/moe.py:776` w13 scale + L813 w2 scale，已实测）。fp8 path 用 `torch.ones(...)` 是 method 自身 inconsistency — 本 patch 在 trailing rank 上把 scale 置 0，等价于让 fp8 path 在 "no-slice" 区段对齐 MXFP4 的 zero-init 语义。

---

## 5. 验证矩阵

来源：`fix_wave/progress/teammate-8.md` final verify + `fix_wave/{logs,outputs}/final_tp{2,4,8}.*`。

| 档 | A1 weight load | A2 4 prompt 无 NaN/Inf | A3 vs corr baseline | A4 输出语义合理 | log / output |
|---|---|---|---|---|---|
| **tp=2** | ✅ | ✅ | byte-diff（**P2 锚点 byte-id 保持**；P0/P1/P3 first_diff 5-119，与 tp2_unpatched sampling noise 同量级）| ✅ coherent | `fix_wave/logs/final_tp2.log` / `fix_wave/outputs/final_tp2.json` |
| **tp=4** | ✅ | ✅ | byte-diff 同上归因（teammate-4 §V1 代码层证 zero-trigger，starts=[0,3,6,9]<10）| ✅ coherent | `fix_wave/logs/final_tp4.log` / `fix_wave/outputs/final_tp4.json` |
| **tp=8** | ✅ | ✅ | n/a（baseline 无 tp=8）| ✅ coherent，与 tp=2/4 同质量 | `fix_wave/logs/final_tp8.log` / `fix_wave/outputs/final_tp8.json` |

**关键输出对照**（teammate-5 §3.2，三态对比）：

| Prompt | tp=8 early-return-only（gibberish）| **tp=8 双层 fix（coherent）** | tp=2 baseline |
|---|---|---|---|
| P0 introduce | `小弟sets邪倾倒uiropa盐尚书arms cryptic...` | `Hmm, the user asked me to introduce myself. That's a straightforward request...` | `Hmm, the user asked me to introduce myself. This is a straightforward request...` |
| P2 1+2+3 | `ureau版沥bla趋zyn Tanner清真...` | `... 1+2=3, then 3+3=6 ... = 6` (eos 183t) | `... 1+2=3, then 3+3=6. So the answer is 6.` (eos 60t) |
| P3 增肌 | `既是olerCharge密 effect pinsilet捷符合...` | `好的，用户问的是"如何在一个月内增肌10公斤"...` | `好的，用户问怎么在一个月内增肌10公斤...` |

A3 byte-diff 归因：sampling 非确定性（multi-GPU + cudagraph + greedy temperature=0），同 patch 同 commit 重跑也不 byte-id（teammate-3 §2.4 + teammate-4 §V1 + tp2_unpatched 反证）。

---

## 6. 调查方法论（3 条）

1. **Layer-by-layer dump**：当 weight-load 通过但输出乱码时，加 instrumentation log 每 layer 的 trailing-rank early-return 触发情形 + dtype + shape，定位首个发散 layer（teammate-5 §2 — 一次 dump 即定位 K1）
2. **假设必须实验证立**：reviewer 对 "scale init=1.0 是 root cause" 不接受推断，要求 zero-fix 实验 → 实测 4/4 prompt 从乱码恢复 coherent，证立完成（fix_wave §0 TL;DR）
3. **跨 tp 不回归实测**：每次 patch 形态变化（17L → 24L → commit）都重跑 tp=2/4 三档，确认 zero-trigger（D=10，tp≤4 starts 全 < D）+ A3 byte-id 锚点保持（teammate-8 §3）

---

## 7. 代码索引

| 项 | 位置 |
|---|---|
| Patch commit | ATOM `969d564` on branch `feat/step3p5-flash-support`（单 commit / 31 行 / 可 revert）|
| Patch 内容 `_load_w13` 早返回块 | `atom/model_ops/moe.py` L2314-L2333（commit 上下文行号）|
| Patch 内容 `_load_w2` 早返回块 | `atom/model_ops/moe.py` L2376-L2386 |
| ceil split 公式（保留行）| `atom/model_ops/moe.py` L2310-L2313 + L2372-L2375 |
| ceil 注释（原意：不丢 partial scale block）| `atom/model_ops/moe.py` L2306-L2309 |
| fp8 scale `torch.ones()` init（init 反差源）| `atom/model_ops/moe.py` L1621, L1628, L1634, L1642, L1655, L1658 (`Fp8MoEMethod._create_weights`) |
| MXFP4 scale `torch.zeros()` init（设计一致性参考）| `atom/model_ops/moe.py` L776 (w13 scale), L813 (w2 scale) |
| Step-3.5 layer dispatch（dense 0-2 + MoE 3-44）| `~/ATOM/atom/models/step3p5.py:14-16` |
| fp8 dispatch path（`bf16 = fp8 * scale` 乘法路径，无 div-by-zero 风险）| reviewer F1/F15/F16 in `fix_wave/progress/teammate-6.md` |
| Base commit（patch 之前）| ATOM `acff926` "fix(moe): correct FP8 blockscale inter_dim padding align for all tp configs" |

---

## 8. 相关文档

| 内容 | 路径 |
|---|---|
| Wave 7-8 close 报告（fix 实验线 + reviewer + verify 全文）| `fix_wave/WAVE_CLOSE.md` |
| Layer-level 根因实验 + zero-fix 实测 | `fix_wave/progress/teammate-5.md` |
| Reviewer pass（2 BLOCK 闭环 + 4 WARN 标注 + 10 INFO）| `fix_wave/progress/teammate-6.md` |
| Final verify tp=2/4/8 三档 A1-A4 全 PASS | `fix_wave/progress/teammate-8.md` |
| F-OPEN-1 原始 issue + ceil split 行为分析 | `correctness_eval/CORRECTNESS_REPORT.md` §1 §4 + `issue_wave/ATOM_ISSUE_DRAFT.md` |
| 项目顶层状态 | `SESSION_HANDOFF.md` / `PROJECT_SUMMARY.md` / `FINAL_REPORT.md` |
| Wave 9 (本 wave) config | `doc_wave/TEAM_CONFIG.md` |

---

**End of TP8_ROOT_CAUSE_AND_FIX — 2026-04-29 / wave 9 / teammate-doc**
