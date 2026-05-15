# Patch B — ATOM SWA Path-1 per-layer KV head 修复

> 此文档为 **step35-flash-support 系列 patch 文档之一**, 完整索引见 [00_overview.md](./00_overview.md)。兄弟 doc: [01_patch_swiglustep.md](./01_patch_swiglustep.md) (vllm SwigluStep enum 加白) | 共同教训: [03_lessons.md](./03_lessons.md)。

**文件**: `/home/junlin12/ATOM/atom/plugin/attention.py`
**当前 md5** (patched): `5289f0d6c7f6a2364f5f6b8025c352c9`
**Pre-patch md5** (git HEAD): `d49ef0b4596ebbaca240f9d83b224820`
**Wave**: G-3(i) Phase 1-7 (2026-05-14 → 2026-05-15)

---

## 概述

ATOM `attention.py` 的 SWA (sliding-window attention) workspace shape 在异质 per-layer KV head 模型 (例: stepfun-Step-3.5-Flash-FP8) + tp=2 路径上触发 aiter GQA 整除 assert, EngineCore crash。Patch B 改 swa_workspace 维度从 `self.num_heads_kv` (model-wide 错值, 走 vllm `ModelConfig.get_num_kv_heads()` fallback 推断) 切到"全 model 所有 vllm Attention layer 中 max(per-layer num_kv_heads)" 实测真值, 解决 GQA shape assert。

> **CAVEAT (强标注, 不可 strip)**: 本 patch 解决 SWA workspace shape assert (历史 EngineCore crash on stepfun-Flash-FP8 + tp=2), **不解决** wave G-3(i) NaN 问题。T65 实证: revert SWA patch + tp=8 仍 NaN, NaN P0 在 vllm v1 inductor / cudagraph (T67 + T71 双证, workaround = `--enforce-eager`, 见 `USER_REPORT.md` §"实证有效 workaround")。Patch B 在 tp=8 路径下**非必需** (T65 cold start 无 crash), 保留是因为 SWA assert 修复本身在 tp=2 路径下有效; 可考虑 wave 后回滚以减少代码维护面 (USER_REPORT.md L192 跨 session 指针)。

---

## 业务背景

### Mechanism: SWA workspace shape assert 在 step3p5 异质 GQA 上 crash

stepfun-Step-3.5-Flash-FP8 是**异质 per-layer attention head** 模型:

| 层类型 | num_attention_heads (q) | num_attention_groups (kv) | config 字段位置 |
|---|---|---|---|
| Full 层 | 64 | 8 | 顶层 (vllm 标准字段) |
| Sliding 层 | 96 | 8 | 私有 `attention_other_setting` (stepfun-only) |

**根因链**:

1. ATOM `attention.py:225`:
   ```python
   self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
   ```
   走 vllm `ModelConfig.get_num_kv_heads()`。该方法**不识别 stepfun 私有 `attention_other_setting`**, 在缺顶层 `num_key_value_heads` 字段时 fallback 到 `num_attention_heads = 64` (按 MHA 处理) → 接到错值 64。

2. tp=2 后 `self.num_heads_kv = 64/2 = 32` (T51 实测 runtime k=32)。

3. ATOM `attention.py:429-433` (pre-patch) swa_workspace 按 `self.num_heads_kv` 分配:
   ```python
   swa_workspace = torch.empty(
       (2, fetched_shape, self.num_heads_kv, self.head_dim),  # k_dim = 32
       dtype=...,
       device=...,
   )
   ```

4. 下游 `attention_mha.py:396` 把 `swa_workspace[0/1]` 当 (key_fetched, value_fetched) 喂给 `aiter.flash_attn_varlen_func`。sliding 层 q = 96/2 = 48, k = 32, ratio q/k = 1.5 → **不整除** → aiter `mha_varlen_fwd_kernels.cu:478` GQA assert raise:
   ```
   GQA requires num_heads_q % num_heads_k == 0
   ```
   EngineCore crash, vllm serve cold start fail。

**期望** (per-layer 真值, kv = 8): tp=2 → k = 4, q = 48, ratio = 12 ✅ (整除)。

### 为什么 ATOM `step3p5.py` 那边读对了, builder 这边读错了

ATOM `step3p5.py:325-337` (`Step3p5Attention.__init__`) **正确** 按 `layer_types[layer_idx]` 分支取 `attention_other_setting["num_attention_groups"]`, 把正确 per-layer 值传给 vllm `Attention(num_kv_heads=self.num_kv_heads, ...)` (`step3p5.py:441`)。

问题**只在** ATOM attention metadata builder (`plugin/attention.py`):
- builder 是**单例 / 跨层共享** (一个 model_runner 全局只有一份 builder)
- builder 绕过 model 层、直接问 vllm `model_config` (model-wide 单值)
- vllm `Attention` layer 实例上有 per-layer 真值 — 只是 builder 没去读它

---

## 改动详情

### Hunk 1: `attention.py` init 段 (L233 → L243-259)

**改前** (`bak.t50`, pre-patch):
```python
self.block_ratio = 1

sliding_window_sizes: set[tuple[int, int] | None] = set()
layers = get_layers_from_vllm_config(config, Attention)
for layer in layers.values():
    assert isinstance(layer.impl, PagedAttentionImpl)
    sliding_window = layer.impl.sliding_window
    if sliding_window is None or sliding_window == -1:
        sliding_window_sizes.add(None)
    elif isinstance(sliding_window, tuple):
        sliding_window_sizes.add(sliding_window)
    else:
        sliding_window_sizes.add((sliding_window - 1, 0))
```

**改后** (现态, file:line `attention.py:233-259`):
```python
self.block_ratio = 1

sliding_window_sizes: set[tuple[int, int] | None] = set()
# SWA per-layer KV head workaround:
# Initialize to 0 (not self.num_heads_kv): self.num_heads_kv comes from
# ModelConfig.get_num_kv_heads() which may not match per-layer vllm
# Attention.num_kv_heads (e.g. stepfun-Flash-FP8 returns 32 from ModelConfig
# but per-layer = 4). Using self.num_heads_kv as a floor would mask the
# real per-layer values. We rely on the loop below to populate from each
# vllm Attention layer; defensive getattr fallback keeps it 0 if field missing.
max_per_layer_num_kv_heads = 0                                       # L243
layers = get_layers_from_vllm_config(config, Attention)
for layer in layers.values():
    assert isinstance(layer.impl, PagedAttentionImpl)
    sliding_window = layer.impl.sliding_window
    if sliding_window is None or sliding_window == -1:
        sliding_window_sizes.add(None)
    elif isinstance(sliding_window, tuple):
        sliding_window_sizes.add(sliding_window)
    else:
        sliding_window_sizes.add((sliding_window - 1, 0))
    per_layer_kv = getattr(layer, "num_kv_heads", None)              # L254
    if per_layer_kv is None:
        per_layer_kv = getattr(layer.impl, "num_kv_heads", 0)        # L256
    if per_layer_kv and per_layer_kv > max_per_layer_num_kv_heads:
        max_per_layer_num_kv_heads = per_layer_kv
self.swa_max_num_heads_kv = max_per_layer_num_kv_heads               # L259
```

**关键点**:
- L243 `init = 0` (**T52 修正**, 不是 proposed §3.3 原文的 `= self.num_heads_kv`) — 否则 floor=32 盖住真值 4, max 永远 32, patch noop (T51 实证)
- L254 一级 fallback `getattr(layer, "num_kv_heads", None)` (vllm `Attention` public 字段, T50 已实测 `vllm/.../attention.py:262` 字段存在)
- L256 二级 fallback `layer.impl.num_kv_heads` 防御性
- L259 写入新字段 `self.swa_max_num_heads_kv` (新增字段, init 阶段一次算完, builder 单例假设下后续 build() 复用)

### Hunk 2: `attention.py` build 段 (L444)

**改前** (`bak.t50` L429-433):
```python
swa_workspace = torch.empty(
    (2, fetched_shape, self.num_heads_kv, self.head_dim),
    dtype=self.vllm_config.model_config.dtype,
    device=self.device,
)
```

**改后** (现态 `attention.py:441-446`):
```python
# Use per-layer max KV heads (computed in __init__) so
# heterogeneous-head models (e.g. stepfun-Flash-FP8) get a
# workspace large enough for the layer with most KV heads.
swa_workspace = torch.empty(
    (2, fetched_shape, self.swa_max_num_heads_kv, self.head_dim),  # L444
    dtype=self.vllm_config.model_config.dtype,
    device=self.device,
)
```

唯一变化: `self.num_heads_kv` → `self.swa_max_num_heads_kv`。

### 不变更

- `extend_workspace` (L273-277) 仍用 `self.num_heads_kv` — extend path 不在 fail 路径上 (teammate-42 baseline 4-prompt PASS), 最小改动原则
- `self.num_heads_kv` 字段保留 — 其他 5 处 grep 命中 (含 extend_workspace shape + log) 仍依赖

---

## Caveat (强标注, 防 strip)

| 维度 | 状态 | Evidence |
|---|---|---|
| SWA assert 修复 (tp=2 历史 EngineCore crash) | ✅ 解决 | T52 修正版 vllm tp=2 启动 PASS (post-fix log + resp.json) |
| **NaN 问题 (wave G-3(i))** | ❌ **不解决** | **T65 实证: revert SWA patch + tp=8 cold start, 3 探针 R3 = HTTP 500 `nan`, 与 T63 patched 状态完全相同的 NaN ValueError 错误链** |
| Patch 必要性 (tp=8 路径) | ❌ **非必需** | T65 实证: HEAD clean (无 patch) + tp=8 cold start PASS at 145s, **无 GQA assert** |
| Patch 必要性 (tp=2 路径) | ✅ 必要 | T50/T51/T52 实证 pre-patch tp=2 启动 GQA assert raise |

**强声明 (USER_REPORT.md §"3-axis 隔离结论" L93-97 锁定)**:
- **Axis 1 — MTP**: 排除 (T63)
- **Axis 2 — SWA Path-1 patch**: **排除** (T65)
- **Axis 3 — vllm v1 inductor compile / cudagraph**: **必要条件** (T67 Python API + T71 HTTP serve 双证)

NaN workaround = **`--enforce-eager` (HTTP serve) 或 `enforce_eager=True` (Python LLM API)**, 严格实证消除 NaN, 性能 2-5x 慢 (关闭 cudagraph + inductor fusion)。详见 USER_REPORT.md "用户立即可用的 fix workaround" 节。

---

## Timeline

| Teammate | 日期 | 动作 | 终态 | bak |
|---|---|---|---|---|
| **T48** | 05-14 | 设计 (proposed_fix_atom_swa_perlayer.md 作者), 选定 Candidate B | unchanged | — |
| **T50** | 05-14 ~07:59 | 首次 apply (照 proposed §3.3 字面: init = self.num_heads_kv); vllm tp=2 仍 FAIL (patch noop, 当时未发现) | patched (有 init=floor bug) | `bak.t50` (pre-patch) |
| **T51** | 05-14 ~08:10 | 加 dump prints, 实证 patch noop 因 init=floor 盖住真值; 反证应改 init=0 | patched + prints | `bak.t51` (pre-patch) |
| **T52** | 05-14 ~08:22 | 重 apply Candidate B + **init=0 修正**; vllm tp=2 SWA assert FIXED, 启动 PASS, 但下游出现 NaN; **self-report `patch_state: ROLLED_BACK`** | patched (init=0 修正版) — **但 self-report 说"已 cp 回 bak.t52"** | `bak.t52` (pre-patch) |
| **T54** | 05-14 ~08:59 | implementer (no-MTP 实验); 启动前 cp `bak.t54` = patched md5 → 实际跑的是 patched 状态 | patched | `bak.t54` (**patched** 5289f0...) |
| **T55** | 05-14 | parallel design auditor (与 T54 同步); **挑出 prompt 与文件系统 fact mismatch** ("T52 prompt 说回滚, 实地仍 patched") | unchanged | — |
| **T63** | 05-15 | NaN 调查; vllm tp=8 patched + no-MTP, 仍 NaN → MTP 排除 (Axis 1) | patched | — |
| **T65** | 05-15 ~00:29 | **revert for control test**; cp `bak.t65` → `git checkout HEAD` → tp=8 cold start (无 crash) + 3 探针 (R3 HTTP 500 nan) → SWA patch 与 NaN 解耦 + tp=8 路径 patch 非必需 → cp `bak.t65` restore | patched (与 wave 启动时一致) | `bak.t65` (**patched** 5289f0...) |
| **T71** | 05-15 ~03:32 | final state verify; 清磁盘 ~2.2 GB + vllm serve `--enforce-eager` HTTP 严格证 PASS; 收尾确认 attention.py md5 = `5289f0d6...` (与 wave 启动一致) | patched (md5 终态) | — |

### T52 反模式 #20 (self-report-as-ground-truth) 诚实标注

**矛盾 fact pair**:
- T52 progress L11 `patch_state: ROLLED_BACK` + L178-179 "attention.py 已 cp 回 .bak.t52"
- 后续 `bak.t54` md5 = `5289f0d6...` (patched, 含 T52 init=0 修正) → T54 启动前 attention.py 已是 patched 状态
- T55 audit 实地 grep 看到 patched 字段在文件中

**两种诚实可能**:
1. T52 self-report 错误: 实际未 cp 回, 但 progress 写了"已回滚" (反模式 #20 实例)
2. T52 真回滚, T52→T54 之间另派未编号动作重 apply (但 progress 链无此 evidence)

**保留此矛盾的诚实记录**, 不强行解释, 供后续 promotion 参考。本反模式实例的 lessons-learned 解读详见 [03_lessons.md](./03_lessons.md) 教训 1 (self-report ≠ ground truth)。

> **另见**: 本 patch 在 fp8 / AITER 集成路径上与 [01_patch_swiglustep.md](./01_patch_swiglustep.md) (Patch A) 是**互补**关系 — Patch A 解 vllm AITER MoE dispatch 阻塞, Patch B 解 ATOM SWA workspace shape 阻塞; 两者命中**不同**层 (Patch A 在 fused_moe layer 43-44, Patch B 在 SWA attention builder 全局), 互不依赖, 但同 wave 期间一起暴露 + 一起验证。

---

## References

| Type | Path |
|---|---|
| Proposal MD (T48 设计) | `/home/junlin12/project_step35_vllm_repro/proposed_fix_atom_swa_perlayer.md` |
| Source (现态) | `/home/junlin12/ATOM/atom/plugin/attention.py` (md5 `5289f0d6c7f6a2364f5f6b8025c352c9`) |
| Pre-patch backups | `attention.py.bak.t50` / `bak.t51` / `bak.t52` (md5 `d49ef0b4596ebbaca240f9d83b224820`) |
| Post-patch backups | `attention.py.bak.t54` / `bak.t65` (md5 `5289f0d6c7f6a2364f5f6b8025c352c9`) |
| Wave G-3(i) USER_REPORT | `/home/junlin12/project_step35_vllm_repro/USER_REPORT.md` (3-axis 隔离 + workaround) |
| Key progress (timeline 实证源) | `progress/teammate-{48,50,51,52,55,58,63,65,71}.md` |
| Down-stream consumer (GQA assert raise 点) | `/home/junlin12/ATOM/atom/plugin/attention_mha.py:396` (`aiter.flash_attn_varlen_func`) — caveat: L396 是 SWA path 调用 aiter kernel (kernel 内 q/kv head 不齐时 assert);  Python-level GQA 整除 assert 在同文件 L240 (`assert num_q_heads_total % num_kv_heads == 0`); line ref 来自 wave 期间, 后续 file 可能变化, 以实际为准 |
| vllm Attention.num_kv_heads 字段 (L1 fallback 来源) | `/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/attention/attention.py:262` |
| Model 端正确读 per-layer kv 的对照 | `/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/step3p5.py:325-337,441` (or ATOM `step3p5.py` 同结构) |
