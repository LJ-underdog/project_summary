# Step-3.5-Flash-FP8 集成 patch 集 (vllm + ATOM)

> 此文档为 **step35-flash-support 系列 patch 文档之首**, 完整索引见下方"文档地图"节。
> **来源**: project_step35_vllm_repro Wave 15-B/C/D + Wave G-3(i) Phases 4-14 (2026-05-14 → 2026-05-15)
> **平台**: ROCm gfx942 (MI308X 8 GPU/节点)
> **vllm**: `b31e9326a` (v0.17.1.dev0+gb31e9326a) | **aiter**: `f06cdcca5` | **target model**: `stepfun-ai/Step-3.5-Flash-FP8` sha=6eebda5...

---

## 任务背景

把 **stepfun-ai/Step-3.5-Flash-FP8** model 在 ROCm + MI308X + vllm v1 + AITER 上集成跑通, 包括:

- vllm 端 fp8 quant + AITER MoE backend dispatch (含 stepfun 私有 SwigluStep activation)
- ATOM (本机 vllm plugin layer) 端 sliding-window attention (SWA) workspace 在 stepfun **异质 per-layer attention head** (full 层 q=64/kv=8, sliding 层 q=96/kv=8) 配置下的 GQA shape 兼容
- vllm serve HTTP / Python LLM API 双路径输出可用

期间发现 vllm site-packages + ATOM 源各需 1 个 source-level patch 才能突破集成路径上的 hard fail (dispatch fail / GQA assert crash)。本 doc 集合化两个 patch 的完整证据链 + 共同教训, 供后续接手人参考。

> **注**: 本 doc 集**仅文档化 source patch**, 不文档化 wave G-3(i) 收尾发现的 NaN 问题 root cause (= vllm v1 inductor compile / cudagraph; workaround = `--enforce-eager`); 该 NaN root cause 与本 doc 集的 2 个 patch **解耦**, 见各 doc Caveat 节 + `USER_REPORT.md` "3-axis 隔离结论" 节。

---

## 2 个 patch 一览

| Patch | 文件 | 状态 (md5) | 解决的问题 | 与 NaN 关系 |
|---|---|---|---|---|
| **A — vllm SwigluStep enum 加白** | `/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py` | applied (`78ddbcd3a805e39740bc731cecd46a8d`); 自 T32 单次 forward apply 后**无回滚** | layer 43-44 `ActivationType.SwigluStep` 被 vllm AITER backend `_supports_activation` 白名单剔除 (oracle gate) 或 dispatch 阶段 `raise ValueError`; 需让 vllm 端识别 + 路由到 aiter CK swiglustep kernel | 不影响 NaN; T67 `enforce_eager` 模式输出语义正确, 间接证实 SwigluStep MoE forward 数学路径无误 |
| **B — ATOM SWA per-layer KV** | `/home/junlin12/ATOM/atom/plugin/attention.py` | applied (`5289f0d6c7f6a2364f5f6b8025c352c9`); 经过 5 个 .bak (T50/T51/T52/T54/T65) 多次 apply / dump / partial-revert / restore | tp=2 路径下 `swa_workspace` shape 用 `self.num_heads_kv` (走 vllm `ModelConfig.get_num_kv_heads()` fallback 推断的 model-wide 错值 64) 算 kv dim → sliding 层 q=48 / k=32 不整除, aiter `mha_varlen_fwd_kernels.cu:478` GQA assert raise → EngineCore crash | **不解 NaN** (T65 实证: revert + tp=8 仍 NaN); 在 tp=8 路径下**非必需** (T65 cold start 无 GQA assert) |

### 净改动量对照

| Patch | hunk 数 | 净行变化 | 影响范围 | 回滚方式 |
|---|---|---|---|---|
| A | 3 hunk (单文件) | +3 行 (含 enum / dispatch elif / `_supports_activation` 白名单) | vllm AITER MoE oracle + dispatch | `patch -R < /home/junlin12/project_step35_vllm_repro/after_fix_swiglustep.patch.t31` |
| B | 2 hunk (单文件) + 7 行注释 | +14 行 (init 段 collection loop + workspace shape 字段切换) | ATOM SWA workspace shape 计算 | `cp /home/junlin12/ATOM/atom/plugin/attention.py.bak.t50 /home/junlin12/ATOM/atom/plugin/attention.py` |

---

## 共同 caveat (链接到 03_lessons.md)

两个 patch 均**不解决** wave G-3(i) NaN 问题 (USER_REPORT.md §"3-axis 隔离结论" L93-97 严格锁定):

- **Axis 1 — MTP**: 排除 (T63 实证)
- **Axis 2 — SWA Path-1 patch (本 Patch B)**: **排除** (T65 实证, revert + tp=8 仍 NaN)
- **Axis 3 — vllm v1 inductor compile / cudagraph**: **必要条件** (T67 Python API + T71 HTTP serve 双证)

NaN workaround: `--enforce-eager` (HTTP serve) 或 `enforce_eager=True` (Python LLM API), 严格实证消除 NaN, 性能 2-5x 慢 (关闭 cudagraph + inductor fusion)。详见 `USER_REPORT.md` §"用户立即可用的 fix workaround"。

> **§22 防 caveat-stripping**: 本节及各 doc Caveat 节的"patch 不解决 NaN"标注**不可在任何下游引用 / 综合 / review 阶段被 strip 掉**。详细反模式分析见 `03_lessons.md` 教训 3。

---

## 文档地图

| Doc | 内容 | 主要受众 |
|---|---|---|
| **[00_overview.md](./00_overview.md)** (本文档) | 任务背景 / 2 patch 一览 / 共同 caveat / 文档地图 | 接手人快速 entry |
| **[01_patch_swiglustep.md](./01_patch_swiglustep.md)** | Patch A (vllm SwigluStep) 完整证据链: 业务背景 / 3 hunk diff / 验证证据 (T33 96 次 dispatch + 84.6s JIT build) / Caveat (与 NaN 解耦) / Timeline / References | 想理解 / rollback / port 此 patch 的人 |
| **[02_patch_swa_perlayer.md](./02_patch_swa_perlayer.md)** | Patch B (ATOM SWA per-layer KV) 完整证据链: 业务背景 (异质 GQA 模型 + builder 单例反模式) / 2 hunk diff / 5 .bak 文件表 / Caveat (与 NaN 解耦 + tp=8 非必需) / Timeline (含 T52 反模式 #20 诚实标注) / References | 想理解 / rollback / port 此 patch 的人 |
| **[03_lessons.md](./03_lessons.md)** | 跨 patch 教训 (5 条): self-report ≠ ground truth (#20) / dispatch 加白 ≠ 数值正确性 / caveat-stripping 防御 (#22) / 多次 patch-revert-restore 的 .bak audit 价值 / vllm v1 inductor + ROCm fp8 的 P0 真因 | 后续 wave 派单 / promotion 候选 |

---

## References

- **本机源 wave**: `/home/junlin12/project_step35_vllm_repro/`
  - `WAVE_DOC1_CONFIG.md` — DOC-1 wave 配置 (本 doc 集生成依据)
  - `USER_REPORT.md` — Wave G-3(i) Phases 4-14 用户最终报告 (3-axis 隔离 + workaround + 5 维 GPA)
  - `WAVE_CLOSE.md` — 早期 wave (15-B/C/D) 收尾
  - `proposed_fix_swiglustep_aiter.md` — Patch A T31 proposal
  - `proposed_fix_atom_swa_perlayer.md` — Patch B T48 proposal
  - `after_fix_swiglustep.patch.t31` — Patch A unified diff archive (29 行)
- **本 doc 集 Phase 1 调研**: `progress/teammate-DOC1-A.md` (Patch A) + `progress/teammate-DOC1-B.md` (Patch B)
- **跨 wave 关联项目**: `stepfun_fp8_fmoe_wave2.md` (含 q_dtype e4m3fn→fnuz 修正 + aiter `apply_act_and_mul` 上游 bug 教训, 与本 task 同硬件 + 同 model 系列)
