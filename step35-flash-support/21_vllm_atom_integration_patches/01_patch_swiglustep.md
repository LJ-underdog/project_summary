# Patch A — vllm AITER MoE SwigluStep activation enum 加白

> 此文档为 **step35-flash-support 系列 patch 文档之一**, 完整索引见 [00_overview.md](./00_overview.md)。兄弟 doc: [02_patch_swa_perlayer.md](./02_patch_swa_perlayer.md) (ATOM SWA per-layer KV) | 共同教训: [03_lessons.md](./03_lessons.md)。
>
> **来源**: project_step35_vllm_repro Wave 15-B/C/D (T31 propose / T32 apply / T33 verify, 2026-05-14)
> **目标文件 (单文件 in-place patch)**: `/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py`
> **vllm commit**: `b31e9326a` ; **aiter commit**: `f06cdcca5` ; **平台**: ROCm gfx942 (MI308X 8 GPU/节点)
> **Status**: applied (md5 current = `78ddbcd3a805e39740bc731cecd46a8d`); 自 T32 apply 后**无回滚**

---

## 概述

vllm 默认 AITER MoE backend 仅白名单 `SILU` / `GELU` 两种 activation。ATOM step3p5 model 在 layer 43-44 上需要使用 `ActivationType.SwigluStep` (CK kernel 硬编码 `silu(g).clamp(max=7) * up.clamp(±7)`), 与 model 数学语义对齐。原 vllm 代码不识别 `SWIGLUSTEP`, 导致 oracle 阶段被剔除 fallback 到 torch/triton (性能损失 + 数学语义偏差), 或在 dispatch 阶段直接 `raise ValueError`.

本 patch 在单文件内 +3 行净变更, 让 vllm 端识别并 dispatch SwigluStep 到 aiter CK kernel, 与 ATOM model 所选 activation 完全闭环。

---

## 业务背景 (为什么需要这个 patch)

### Step-3.5-Flash model 的 SwigluStep 触发条件

ATOM `atom/models/step3p5.py:55-72` (`_uses_swiglustep_at_layer`) 注释明示:

> "The CK kernel hard-codes the clamp at 7.0; **Step-3.5-Flash uses 7.0 at layers 43 and 44**, which is why the kernel is only valid at those layers. Other layers must keep the plain Silu path."

机制 (`step3p5.py:185-193` Step3p5MoE `__init__`):
- 读 model config 字段 `swiglu_limits` (per-layer list)
- 仅当 `swiglu_limits[layer_idx] > 0` 时 `self._activation = ActivationType.SwigluStep`, 否则 `Silu`
- 该 `self._activation` 在 L223-234 透传给 `FusedMoE(... activation=self._activation, ...)`, 最终经 vllm fused_moe oracle 流到 `rocm_aiter_fused_experts`

forward 注释 (`step3p5.py:290-293`):
> "Routed experts. At SwigluStep layers (43-44) the FusedMoE was constructed with `activation=ActivationType.SwigluStep` so the CK kernel applies `silu(g).clamp(max=7) * up.clamp(±7)` per expert."

### 不打 patch 的 fail mode

| 路径 | 不打 patch 的行为 |
|---|---|
| Oracle 阶段 | `AiterExperts._supports_activation(MoEActivation.SWIGLUSTEP) → False` (白名单只含 SILU/GELU) → vllm Oracle 剔除 AITER backend, fallback 到 torch/triton — layer 43/44 不走 CK kernel, 性能损失 + 无 hard ±7 clamp |
| Dispatch 阶段 (oracle 绕过时) | `else: raise ValueError(f"Unsupported activation: {activation}")` → EngineCore crash |

---

## 改动详情 (diff 引用 + file:line)

**目标文件**: `/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py`

**完整 unified diff archive**: `/home/junlin12/project_step35_vllm_repro/after_fix_swiglustep.patch.t31` (29 行, 与实测 `diff bak.t31 current` 逐字一致)

### Hunk 1 — `ActivationMethod` IntEnum 加 SWIGLUSTEP (line 46-51)

改前 (`.bak.t31` 行 46-50):
```python
class ActivationMethod(IntEnum):
    # This allows interfacing with AITER ActivationType enum
    # without importing the ActivationType enum from AITER globally.
    SILU = 0
    GELU = 1
```

改后 (current 行 46-51, 净 +1):
```python
class ActivationMethod(IntEnum):
    # This allows interfacing with AITER ActivationType enum
    # without importing the ActivationType enum from AITER globally.
    SILU = 0
    GELU = 1
    SWIGLUSTEP = 3
```

**为什么是 3 而非 2**: aiter `ActivationType` Python enum value 表为 `Silu=0 / Gelu=1 / Swiglu=2 / SwigluStep=3` (T29 §Q1.C+Q4.A live introspect 实证: `aiter.ActivationType.SwigluStep.value == 3`)。vllm `_aiter_ops.py:98` 内 `_rocm_aiter_fused_moe_impl` 直接用 `ActivationType(activation_method)` 构造 aiter enum, 无中间映射, 故传入 int 必须 = aiter Python enum value 而**不是** ck 内部 method int (2, 那个由 aiter C++ 内部映射, 上层不感知)。

### Hunk 2 — dispatch elif (line 201-208)

改前 (`.bak.t31`):
```python
    if activation == MoEActivation.SILU:
        activation_method = ActivationMethod.SILU
    elif activation == MoEActivation.GELU:
        activation_method = ActivationMethod.GELU
    else:
        raise ValueError(f"Unsupported activation: {activation}")
```

改后 (current 行 201-208, 净 +2):
```python
    if activation == MoEActivation.SILU:
        activation_method = ActivationMethod.SILU
    elif activation == MoEActivation.GELU:
        activation_method = ActivationMethod.GELU
    elif activation == MoEActivation.SWIGLUSTEP:
        activation_method = ActivationMethod.SWIGLUSTEP
    else:
        raise ValueError(f"Unsupported activation: {activation}")
```

### Hunk 3 — `_supports_activation` 白名单 (line 332-334)

改前 (`.bak.t31` 行 329-331):
```python
    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [MoEActivation.SILU, MoEActivation.GELU]
```

改后 (current 行 332-334, 净 +0 — list 内追加 1 个成员):
```python
    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [MoEActivation.SILU, MoEActivation.GELU, MoEActivation.SWIGLUSTEP]
```

### 总变更量

3 hunk, 单文件, 净 +3 行 (改前 404 行 → 改后 407 行; 注: T31 proposal 估 411 行, T32 实测 404 行, 不影响 patch — 行号锚点对齐, hunk 全文一字不差)。

---

## 验证证据 (apply 后 dispatch path work 的 evidence)

### V1: T33 Wave D verify SUCCESS (5/5 PASS)

**log**: `/home/junlin12/project_step35_vllm_repro/logs/vllm_serve_v6_swiglustep_aiter.log`

| 维度 | 信号 | 实测 |
|---|---|---|
| DIM 1 | AITER backend selected | line 469: `Using AITER Fp8 MoE backend` PASS |
| **DIM 2** | **SWIGLUSTEP dispatched** | **96 次** `ActivationType.SwigluStep` 命中 (= 12 MoE layer × 8 worker, line 664-686 全 8 worker [TP0-TP7] 都命中); JIT build 成功 line 689: `finish build module_moe_ck2stages_f8_f8_preshuffle_off_b16_swiglustep_per_1x128_mulWeightStage2, cost 84.6s` |
| DIM 3 | T19 fall-through NOT triggered | grep "falling back / fall back to torch / fallback path" = 0 hit |
| DIM 4 | dtype mismatch NOT triggered | 无 dtype 错 |
| DIM 5 | 8 worker 全 ready | "Application startup complete" |

cold start wall-time = ~6m 20s (与 USER_REPORT.md T27 4-7min 历史区间一致)。

→ vllm `MoEActivation.SWIGLUSTEP → ActivationMethod.SWIGLUSTEP=3 → ActivationType.SwigluStep` 全链路 dispatch **实证 work**, 命中真正的 ck swiglustep CK kernel。

### V2: T67 enforce_eager 模式语义正确

T67 `enforce_eager=True` 跑 step3p5 fp8 + tp=8 + EP, `"List primes within 50:" → " 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31"` — **输出语义完全正确**, 间接证实 SwigluStep MoE forward 数学路径无误。

### V3: patch 状态自 T32 apply 起持续 active

| 文件 | md5 | 说明 |
|---|---|---|
| `rocm_aiter_fused_moe.py` (current) | `78ddbcd3a805e39740bc731cecd46a8d` | T32 apply 后, 至今未触碰 |
| `.bak.t31` (apply 前快照) | `258a7b041afee6c6d4dbaa19c0b736f3` | mtime 03:24:32 |
| `.bak.t19` (T19 时点更早快照) | `258a7b041afee6c6d4dbaa19c0b736f3` | **与 .bak.t31 md5 一致**, 实证 T19→T31 间无第三方改动 |

T63 (G-3(i+) tp=8 cold start) 仍正常 dispatch SwigluStep, 无 oracle gate 拒绝, 间接证实 patch 仍 active。

---

## Caveat / 已知局限

### 🔴 与 wave G-3(i) NaN 问题解耦 (§22 防 strip)

**本 patch 仅修复 vllm 端 SwigluStep dispatch enum gate, 不影响也不解决 wave G-3(i) NaN**。

实证依据 (T67 progress):
- `enforce_eager=True` (绕开 vllm v1 inductor compile / cudagraph) 跑 step3p5 fp8 + tp=8 + EP **完全无 NaN**, 输出语义正确
- NaN root cause 实证 = **vllm v1 inductor compile / cudagraph 优化路径**, 与 step3p5 fp8 相互作用; **与 SWIGLUSTEP / MoE 路径正交**

→ 详见兄弟 doc [02_patch_swa_perlayer.md](./02_patch_swa_perlayer.md) (Patch B SWA 也已实证与 NaN 解耦, T65) + `USER_REPORT.md` (整 task 串联) + [03_lessons.md](./03_lessons.md) 教训 2 / 教训 3 (dispatch 加白 ≠ 数值正确性 + caveat-stripping 防御)。

### 其它已知未实证假设 (沿用 T31 unverified)

- **U3**: w13_weight.dtype = e4m3fnuz 基于 process_weights_after_loading 间接证据 + T33 startup PASS + JIT build swiglustep CK kernel 84.6s 成功间接证实, **未直接 print 实测**
- **U2**: vllm 端 `swiglustep_and_mul_triton` 默认 `limit=7.0` 与 aiter `swiglustep(limit=7.0)` 一致, 整链路无 limit 参数; **若 model card 指定其他 limit 值, 双方会一致地用错**
- **T19 patch**: oracle/fp8.py:364-388 patch 与本 patch 互补 (oracle 安全网 + SWIGLUSTEP 实际支持), **保留不动** (T31 §决策 2 + T30 §Q3.C 二次确认)
- **vllm 上游 PR #39436**: 与本 patch 范围不同, 本 patch 是适配本机 b31e9326a 的最小变更; PR #39436 完整 diff 范围未读 (lead 红线: 不 WebFetch)

---

## Timeline (teammate 表)

| Teammate | Wave | 日期/mtime | 角色 / 动作 |
|---|---|---|---|
| T7 / T14-T22 | early discovery | 2026-05 早 | swiglustep ck kernel 定位 / 初步 propose / aiter+vllm 侧 codepath 调研 |
| T19 | 早期 patch | 2026-05-14 01:04 | oracle/fp8.py:364-388 T19 patch (与本 patch 互补; `.bak.t19` 此时建立) |
| T29 | 15-A1 | 2026-05-14 | aiter 侧实证 (`ActivationType.SwigluStep.value == 3`) |
| T30 | 15-A2 | 2026-05-14 | vllm 侧实证 (3 hunk 行号 + Oracle gate 论证) |
| **T31** | 15-B implementer | 2026-05-14 | 写 `proposed_fix_swiglustep_aiter.md` (3 hunks + U1/U2/U3 实证补完 + 7-step apply checklist) |
| **T32** | C apply | 2026-05-14 03:25 | **Edit 应用 3 hunk patch + 创建 `.bak.t31` + 生成 `after_fix_swiglustep.patch.t31`** |
| **T33** | D verify | 2026-05-14 03:36-03:42 | **vllm serve startup 5/5 PASS, 96 次 SwigluStep dispatch + CK kernel build 84.6s 成功** |
| T34 | E inference | 2026-05-14 04:05 | inference fail, 但根因 = 独立的 reshape_and_cache 类型 mismatch (与 SWIGLUSTEP 解耦) |
| T35 / T44 / T45 / T58 / T63 / T67 / T71 | F-pre / F-A / G-1 / G-3(i)+ | 后续 | 引用 SWIGLUSTEP 上下文; **无人改动本 patch**; T67 实证 NaN 与本 patch 解耦 |

**关键**: 自 T32 apply (mtime 03:25:04) 后, 文件 md5 一直保持 `78ddbcd3a805e39740bc731cecd46a8d` — 本 patch 是**单次 forward apply, 无回滚 / re-apply** (区别于 Patch B SWA 的 5 次 .bak)。

---

## References

- **Proposal**: `/home/junlin12/project_step35_vllm_repro/proposed_fix_swiglustep_aiter.md` (T31 主交付, 含 6 节: 3 hunk patch / U1/U2/U3 实证 / 回归测试 / 备份策略 / 风险节 / apply checklist)
- **Patch archive**: `/home/junlin12/project_step35_vllm_repro/after_fix_swiglustep.patch.t31` (29 行 unified diff, 可 `patch -R` rollback)
- **Backup files**:
  - `/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py.bak.t31` (主备份, T32 apply 前快照)
  - `/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py.bak.t19` (T19 时点更早快照, md5 与 .bak.t31 一致)
- **Verification log**: `/home/junlin12/project_step35_vllm_repro/logs/vllm_serve_v6_swiglustep_aiter.log` (T33 Wave D verify, 5/5 PASS)
- **Key progress**:
  - `progress/teammate-29.md` (aiter 侧 A1)
  - `progress/teammate-30.md` (vllm 侧 A2)
  - `progress/teammate-31.md` (proposed_fix implementer)
  - `progress/teammate-32.md` (apply)
  - `progress/teammate-33.md` (Wave D verify SUCCESS)
  - `progress/teammate-67.md` (NaN root cause 实证 = inductor/cudagraph, 与本 patch 解耦)
  - `progress/teammate-DOC1-A.md` (本 doc 调研报告)
- **ATOM model 源** (业务动机方):
  - `/home/junlin12/ATOM/atom/models/step3p5.py:55-72` (`_uses_swiglustep_at_layer`)
  - `/home/junlin12/ATOM/atom/models/step3p5.py:175-234` (Step3p5MoE `__init__`, `self._activation = ActivationType.SwigluStep`)
  - `/home/junlin12/ATOM/atom/models/step3p5.py:290-293` (forward 注释 layer 43-44 SwigluStep dispatch)
- **Integration context**: `USER_REPORT.md` §4 (本 task 整体串联)
