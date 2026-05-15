# vllm/ATOM 集成 patch 集 (Step-3.5-Flash-FP8)

> 此文档为 **step35-flash-support 项目 vllm 集成路径 patch 顶层 entry**, 完整证据链下沉 [`details/integration/`](./details/integration/README.md)。
> **来源**: project_step35_vllm_repro Wave 15-B/C/D + Wave G-3(i) Phases 4-14 + Wave DOC-1 (2026-05-14 → 2026-05-15)
> **平台**: ROCm gfx942 (MI308X 8 GPU/节点)
> **vllm**: `b31e9326a` (v0.17.1.dev0+gb31e9326a) | **aiter**: `f06cdcca5` | **target model**: `stepfun-ai/Step-3.5-Flash-FP8`

---

## TL;DR

把 **stepfun-ai/Step-3.5-Flash-FP8** 在 ROCm + MI308X + vllm v1 + AITER 上集成跑通时, 发现 vllm site-packages 与 ATOM plugin 各需 1 个 source-level patch 才能突破集成路径上的 hard fail (dispatch fail / GQA assert crash)。本 entry 集合化两个 patch 的状态摘要 + 共同教训 + 与 NaN 解耦的关键 caveat, 详细叙述指向 `details/integration/`。

> **Caveat (§22 防 caveat-stripping, 顶层 entry 显式声明)**: 本 doc 集 2 个 patch 均 **不解决** wave G-3(i) 的 NaN 问题。
> - Patch A (vllm SwigluStep enum 加白) 是 **dispatch 层**修复, 不影响数值正确性
> - Patch B (ATOM SWA per-layer KV) 是 **shape/assert 层**修复; T65 实证 revert + tp=8 仍 NaN, **与 NaN 解耦**, 在 tp=8 路径下**非必需**
> - NaN root cause = vllm v1 inductor compile / cudagraph; workaround = `--enforce-eager`
> - 详见 [`details/integration/03_lessons.md`](./details/integration/03_lessons.md) 教训 3 (caveat-stripping 防御) 与 教训 5 (NaN P0 真因)

---

## 2 patch 一览

| Patch | 文件 | 状态 (md5) | 解决的问题 | 与 NaN 关系 | 详细 |
|---|---|---|---|---|---|
| **A — vllm SwigluStep enum 加白** | `/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py` | applied (`78ddbcd3a805e39740bc731cecd46a8d`); 自 T32 单次 forward apply 后**无回滚** | layer 43-44 `ActivationType.SwigluStep` 被 vllm AITER backend `_supports_activation` 白名单剔除 / dispatch raise; 需让 vllm 端识别 + 路由到 aiter CK swiglustep kernel | 不影响 NaN; T67 `enforce_eager` 模式输出语义正确, 间接证实 SwigluStep MoE forward 数学路径无误 | [`details/integration/01_patch_swiglustep.md`](./details/integration/01_patch_swiglustep.md) |
| **B — ATOM SWA per-layer KV** | `/home/junlin12/ATOM/atom/plugin/attention.py` | applied (`5289f0d6c7f6a2364f5f6b8025c352c9`); 经过 5 个 .bak (T50/T51/T52/T54/T65) 多次 apply / dump / partial-revert / restore | tp=2 路径下 `swa_workspace` shape 用 `self.num_heads_kv` (model-wide 错值) 算 kv dim → sliding 层 q=48/k=32 不整除, aiter `mha_varlen_fwd_kernels.cu:478` GQA assert raise → EngineCore crash | **不解 NaN** (T65 实证: revert + tp=8 仍 NaN); 在 tp=8 路径下**非必需** (T65 cold start 无 GQA assert) | [`details/integration/02_patch_swa_perlayer.md`](./details/integration/02_patch_swa_perlayer.md) |

### 净改动量对照

| Patch | hunk 数 | 净行变化 | 影响范围 | 回滚方式 |
|---|---|---|---|---|
| A | 3 hunk (单文件) | +3 行 (含 enum / dispatch elif / `_supports_activation` 白名单) | vllm AITER MoE oracle + dispatch | `patch -R < /home/junlin12/project_step35_vllm_repro/after_fix_swiglustep.patch.t31` |
| B | 2 hunk (单文件) + 7 行注释 | +14 行 (init 段 collection loop + workspace shape 字段切换) | ATOM SWA workspace shape 计算 | `cp /home/junlin12/ATOM/atom/plugin/attention.py.bak.t50 /home/junlin12/ATOM/atom/plugin/attention.py` |

---

## 共同教训速览

跨 patch 5 条核心教训 (完整叙述见 [`details/integration/03_lessons.md`](./details/integration/03_lessons.md)):

- **教训 1 — self-report ≠ ground truth (反模式 #20)**: T52 PARTIAL apply + 自报 "已 apply" → T54 grep 实证缺 1 处, .bak audit 才挑穿
- **教训 2 — dispatch 加白 ≠ 数值正确性**: vllm SwigluStep 加白让 dispatch 通过, 但**不证明**数值正确; 需独立 forward eval 证实
- **教训 3 — caveat-stripping 防御 (反模式 #22)**: "Patch B 与 NaN 解耦" 这条 caveat 在 4 doc + 顶层 entry 任何位置都不能 strip; 顶层 entry 是最易被下游引用 strip 的位置
- **教训 4 — .bak audit 价值**: Patch B 经 5 次 apply / partial-revert / restore, .bak 文件是定位"哪一版被 apply"的 ground truth, 自报 commit log 不可信
- **教训 5 — NaN P0 真因 = vllm v1 inductor + cudagraph**: 不在 SwigluStep 也不在 SWA, 在更底层的 inductor compile / cudagraph + ROCm fp8 互动; workaround = `--enforce-eager`

---

## References

- **本 doc 集详细索引**: [`details/integration/README.md`](./details/integration/README.md) (=改造自原 `00_overview.md`)
- **CODE_CHANGES.md 集成 patch 子节**: [`CODE_CHANGES.md §3.8`](./CODE_CHANGES.md) (vllm 集成 patch 索引, 与本 entry 互补 — 本 entry 偏 patch 证据链, §3.8 偏 commit/feature 索引)
- **CODE_CHANGES.md 已有相关节**: [`§3.3 SwigluStep`](./CODE_CHANGES.md) (aiter+CK+ATOM 三仓 wiring) + [`§3.7 Sliding Window Attention`](./CODE_CHANGES.md) (aiter mask 修复) — 本 entry 是 vllm/ATOM **集成路径**新发现, 与三仓内部 commit 互补不冲突
- **REPRODUCE.md**: [`REPRODUCE.md`](./REPRODUCE.md) (gfx942 / FP8 单路径复现)
- **本机源 wave** (本 doc 集生成依据, 不在本 repo 内): `/home/junlin12/project_step35_vllm_repro/{WAVE_DOC1_CONFIG.md, USER_REPORT.md, WAVE_CLOSE.md, proposed_fix_*.md, after_fix_swiglustep.patch.t31}`

---

_Promoted from project_step35_vllm_repro/ (wave DOC-1) at 2026-05-15._
