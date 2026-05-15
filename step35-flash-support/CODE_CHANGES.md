# Step-3.5-Flash 全栈推理 — 三仓代码改动总账（聚合视图）

> **范围**：ATOM / aiter / composable_kernel 三仓所有 step35-flash-support 项目相关 commit + working-tree patch 的聚合摘要。
> **来源声明**：
> - 主源：`details/topics/code_changes_all_repos.md`（719 行原文，三仓 commit 全记录）
> - tp=8 双层 fix：`details/topics/18_fp8_tp8_root_cause_and_fix/TP8_ROOT_CAUSE_AND_FIX.md`（ATOM `969d564`）
> - gfx942 patch：`details/projects/14_migration_gfx942/MIGRATION_REPORT.md`（NEW-RC-1/2/3 + M2 padding）
> - kernel dispatch 校验：`details/research/19_kernel_dispatch_report/REPORT.md`（gfx950 tp=2/4 路径实测）
> - 子任务详细叙述：`details/topics/01-07_*.md`
>
> **本文档定位**：聚合摘要 + commit/feature 双维度索引；不复制 719 行原文；详细叙述指向 details/。

---

## §1 概览表

| Repo | 分支 | 终点 commit | 主要改动 commit 数 | 备注 |
|---|---|---|---|---|
| ATOM | `feat/step3p5-flash-support` | **`969d564`** (含 tp=8 双层 fix) / `ccb64621` (路径 A 最低) / `acff926d` (中间态) | 7 | tp=8 双层 fix 在 `969d564` |
| aiter | `feat/step3p5-moe-swiglustep` | **`f06cdcca5`** (含 NEW-RC-3 commit 化) ／历史中间态 `0f8164017` | 10 (含 1 revert + NEW-RC-3 commit 化) | NEW-RC-3 patch 已上 commit (`f06cdcca5`)；X2+X2.5 verify-tool patch 仍 working-tree dirty（aiter/jit/core.py，详见 §2.2 末节） |
| composable_kernel | `feat/swiglustep-moe-no-quant` | `defd7ad29` | 1 | 领先 `origin/develop` 1 commit |

> **UPDATE 2026-04-30 / 2026-05-09 — aiter 行修订**：原表（`0f8164017` + "NEW-RC-3 patch 仅 working-tree dirty，未上 commit"）反映 wave 14 进行时（2026-04-26）状态。2026-04-30 NEW-RC-3 patch 已 commit 化为 aiter `f06cdcca5`（commit message: `fix(moe): force per_1x128 fp8 blockscale to CK 2-stage on gfx942`，diff 与原 working-tree patch byte-id 一致；仍在 `feat/step3p5-moe-swiglustep` 分支，未 push 到 `origin/main`）。当前 reproduce 锁定 commit 见 `REPRODUCE.md §3.1`。证据：`tp2_verify_post_merge_wave/progress/teammate-L25-audit-commit-currency.md` §1.2（`git show f06cdcca5` 实测 + ancestor 验证）。

**git push 操作路径**：`/home/hanchang/junlin12_repos/{atom,aiter,composable_kernel}`（author: `Jun Lin <junlin12@amd.com>`）

---

## §2 Per-repo 视图

### §2.1 ATOM repo

**主要改动文件**：`atom/model_ops/moe.py`、`atom/models/step3p5.py`、`atom/model_engine/model_runner.py`、`atom/model_loader/loader.py`、`atom/model_ops/attentions/aiter_attention.py`

| Commit | 日期 | 改动 / 目的 | 详细叙述 |
|---|---|---|---|
| `ec8cbe87` | 2026-04-23 | **基础支持 + gfx950 preshuffle 修复**：新增 `Step3p5MLP` / `Step3p5MoE` / `Step3p5DecoderLayer` / `Step3p5ForCausalLM`（860 行）；删除 gfx950 skip-shuffle 分支强制 `shuffle_weights()`；注册 architecture；KV head 兼容 `num_attention_groups`；fused expert detection 前置防 `gate_proj` 误匹配 | `details/topics/01_moe_pipeline.md`，原文 `details/topics/code_changes_all_repos.md:29-77` |
| `4a8495ec` | 2026-04-24 | **SwigluStep per-layer wiring**：layer 43-44 routed expert 走 SwigluStep（silu + clamp ±7）；shared expert 在该层走 dense MLP（limit=16 与 CK 硬编码 7.0f 不兼容） | `details/topics/02_swiglu_step.md`，原文 `code_changes_all_repos.md:79-128` |
| `635e59e9` | 2026-04-24 | **BF16 inter_dim padding（tp=4/8）**：`_process_block_quant` 把 inter_dim 320 padding 到 384 满足 MoE kernel 约束 | `details/topics/04_tp_support.md`，原文 `code_changes_all_repos.md:130-172` |
| `9a67e493` | 2026-04-24 | **FP8 `get_fused_moe_quant_config` block_shape fix** | `details/topics/05_fp8_inference.md`，原文 `code_changes_all_repos.md:174-201` |
| `ccb64621` | 2026-04-25 | **FP8 tp=4 三处协调修复（核心）**：(a) `block_n` divisible 检查放宽；(b) padding 与 kernel 约束协调；(c) `_load_w13`/`_load_w2` 用 ceil 整除（含 `+ self.tp_size - 1`）防 scale block [8,9] 残留 `torch.ones()` | `details/topics/06_fp8_tp4.md`，原文 `code_changes_all_repos.md:203-265` |
| `acff926d` | 2026-04-25 | **FP8 blockscale align bug fix（tp=8 base）**：`align = block_n` 无条件分支（替代旧 `align = 64 if inter_dim <= 192 else 128`）；inter_dim=320 padding 到 384 而非 192 | 原文 `code_changes_all_repos.md:267-293`；中间尝试 `270fee71` (align=64) + `3696345e` (revert) |
| **`969d564`** | 2026-04-28 | **tp=8 双层 fix（gfx942 必需）**：(a) trailing rank early-return（rank 命中越界 starts 时跳过 narrow + copy_）；(b) `dtype == float32` 的 fp32 scale tensor `.zero_()` 初始化（防 fp8 dequant 把 raw bits 当 bf16 用 → gibberish）。31 行净增可 revert | `details/topics/18_fp8_tp8_root_cause_and_fix/TP8_ROOT_CAUSE_AND_FIX.md` §3-4 |

### §2.2 aiter repo

**主要改动文件**：`aiter/fused_moe.py`、`aiter/jit/*`、`aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv`、`aiter/dist/*`

| Commit | 日期 | 改动 / 目的 | 详细叙述 |
|---|---|---|---|
| `6d70f7b54` | 2026-04-23 | **SwigluStep enum + C++ codegen + CK submodule bump**：添加 `ActivationType.SwigluStep`；CK submodule 升 `defd7ad29` | `details/topics/02_swiglu_step.md`，原文 `code_changes_all_repos.md:308-366` |
| `68fc7d48b` | 2026-04-23 | **gfx950 MoE pipeline 修复（V1→V3 强制 + Python 端 SwigluStep）**：fused_moe Python dispatch 强制走 V3；wire SwigluStep activation | `details/topics/01_moe_pipeline.md`，原文 `code_changes_all_repos.md:368-418` |
| `3771835ac` | 2026-04-23 | revert 不必要的 +2 row padding | 原文 `code_changes_all_repos.md:420-443` |
| `7ebae9afb` | 2026-04-23 | **sliding window mask off-by-one 修复** | `details/topics/03_sliding_window.md`，原文 `code_changes_all_repos.md:445-477` |
| `7312ea166` | 2026-04-24 | **gfx950 分布式 allreduce/allgather 兼容性修复** | 原文 `code_changes_all_repos.md:479-516` |
| `c38d0c9e6` | 2026-04-24 | **FP8 blockscale 排除 V1→V3 强制的 guard**（FP8 tp=2 dispatch 修复）| `details/topics/05_fp8_inference.md`，原文 `code_changes_all_repos.md:518-543` |
| `a2883ab37` | 2026-04-26 | **删除 buggy ASM kernel tuning 条目**：`glm5_bf16_tuned_gemm.csv` 移除触发 `bf16gemm_bf16_tn_256x256` (N=4096,K=2048) 的 entry，修复 tp=4 长序列 BOS bug（gfx950）；保留 `splitk_clean` variant | `details/topics/07_tp4_longseq_bos_fix.md`，原文 `code_changes_all_repos.md:545-586` |
| `a2247989d` + `dd4257d8f` + `0f8164017` | 2026-04-26 | **stage1 NPerBlock=64 blockscale kernels** | 原文 `code_changes_all_repos.md:588-630` |

**aiter 仓 patch / commit 状态（按时间线）**：

> **UPDATE 2026-04-30 / 2026-05-09**：原"未上 commit 的 working-tree patch"节描述对应 wave 14（2026-04-26）状态。NEW-RC-3 dispatch patch 已 commit 化为 aiter `f06cdcca5`，现以下表区分"已 commit"vs"仍 working-tree dirty"。

| Patch | 状态 | 文件 / 行号 | 用途 | 详细叙述 |
|---|---|---|---|---|
| **NEW-RC-3 dispatch patch** | **✅ 已 commit 为 `f06cdcca5`**（2026-04-30；`feat/step3p5-moe-swiglustep` 分支，未 push origin/main）| `aiter/fused_moe.py:881-886`（`run_1stage = False` 强制 CK 2-stage） | gfx942 tp=8 prefill 需 bypass `fmoe_fp8_blockscale_g1u1` ASM（gfx942 签名不带 block shape，会 dispatch miss → gibberish）。`git checkout f06cdcca5` 后已自动包含，无需手工 apply | `details/projects/14_migration_gfx942/MIGRATION_REPORT.md` §6（NEW-RC-3 详解；描述对应 working-tree dirty 时点）+ `REPRODUCE.md §3.4` 顶部 NOTE（当前 commit 化状态） |
| **X2 + X2.5 verify-tool patch** | ⚠️ 仍 working-tree dirty（未 commit）| `aiter/jit/core.py`（+32 −7 行；`check_args` `Optional[aiter_tensor_t]` 分支 TypeError 修复）| aiter 自 `315123ace` HEAD 起，stepfun engine_init warmup 通过 `Optional[aiter_tensor_t]` annotation 检查 `moe_sorting_opus_fwd` 时撞 TypeError（pybind doc_str 暴露 `aiter_tensor_t`，caller 传 `torch.Tensor`，develop=True 转换在 check_args 之后）；本 patch 修复 wrapper。**非 reproduce 阻塞**（NEW-RC-3 + ATOM `969d564` 核心路径不触达），仅在用 `f06cdcca5` 之后更新 commit 跑内部 verify 工具时撞到 | `tp2_verify_post_merge_wave/progress/teammate-L14-x2.5-patch.md` + `imp_l14_x25.diff`（72 行 unified diff） |

### §2.3 composable_kernel repo

**主要改动文件**：CK MoE GEMM gridwise codegen

| Commit | 日期 | 改动 / 目的 | 详细叙述 |
|---|---|---|---|
| `defd7ad29` | 2026-04-23 | **添加 swiglustep_and_mul epilogue 分支**：在 `gridwise_moe_gemm` 中支持 SwigluStep activation 作为 epilogue | `details/topics/02_swiglu_step.md`，原文 `code_changes_all_repos.md:640-667` |

---

## §3 Per-feature 视图

### §3.1 FP8 推理（tp=2/4/8 全档）

| 子问题 | Repo / Commit | 改动定位 | 详细叙述 |
|---|---|---|---|
| FP8 blockscale dispatch（tp=2）| aiter `c38d0c9e6` + ATOM `9a67e493` | 排除 V1→V3 强制的 guard + `block_shape` fix | `details/topics/05_fp8_inference.md` |
| FP8 tp=4 三处协调修复 | ATOM `ccb64621` | (a) divisible 检查；(b) padding/kernel 协调；(c) ceil 整除 | `details/topics/06_fp8_tp4.md` |
| FP8 align bug（tp=8 base）| ATOM `acff926d` | `align = block_n` 无条件 | `code_changes_all_repos.md:267-293` |
| FP8 tp=8 双层 fix（gfx942 必需）| ATOM `969d564` | early-return + `.zero_()` 初始化 | `details/topics/18_fp8_tp8_root_cause_and_fix/TP8_ROOT_CAUSE_AND_FIX.md` |
| FP8 stage1 NPerBlock=64 kernels | aiter `a2247989d/dd4257d8f/0f8164017` | 新增 blockscale kernel variant | `code_changes_all_repos.md:588-630` |

### §3.2 MoE Pipeline

| 子问题 | Repo / Commit | 改动定位 | 详细叙述 |
|---|---|---|---|
| gfx950 preshuffle bug | ATOM `ec8cbe87` | 删除 skip-shuffle 分支强制 `shuffle_weights()` | `details/topics/01_moe_pipeline.md` |
| Python 端 V1→V3 强制 | aiter `68fc7d48b` | `fused_moe` dispatch 强制走 V3 | `details/topics/01_moe_pipeline.md` |
| Step3p5 model 注册 + fused expert 检测 | ATOM `ec8cbe87` | `Step3p5MoE` + `detect_fused_expert_format()` + `get_fused_expert_mapping()` | `details/topics/01_moe_pipeline.md` |
| TP=4/8 inter_dim padding（320→384）| ATOM `635e59e9` (BF16) + `acff926d` (FP8) + 自动机制（gfx942 借用） | `_process_block_quant` 把 inter_dim padding 满足 kernel 约束 | `details/topics/04_tp_support.md` + `details/projects/14_migration_gfx942/MIGRATION_REPORT.md` §7 |

### §3.3 SwigluStep 激活

| 子问题 | Repo / Commit | 改动定位 |
|---|---|---|
| Activation enum + C++ codegen | aiter `6d70f7b54` | `ActivationType.SwigluStep` + codegen wire |
| CK epilogue 分支 | CK `defd7ad29` | `gridwise_moe_gemm` swiglustep_and_mul |
| ATOM per-layer wiring（layer 43-44） | ATOM `4a8495ec` | `_uses_swiglustep_at_layer()` + `_fuse_shared_at_layer()` |

详细叙述：`details/topics/02_swiglu_step.md`

### §3.4 Kernel Dispatch（gfx950 tp=2/4 实测验证）

| Op | Path（gfx950）| 来源 |
|---|---|---|
| MoE prefill | CK 2-stage（不走 ASM `fmoe_fp8_blockscale_g1u1`，因 N/K dim 不满足 NPerBlock 约束）| `details/research/19_kernel_dispatch_report/REPORT.md` §3 |
| MoE decode | CK 2-stage | 同上 §3.3 |
| Attention prefill | aiter Flash Attention v3 | 同上 §4.1 |
| Attention decode | full → FA v3 / sliding window → 专用 kernel | 同上 §4.2 |
| BF16 Linear | hipblaslt fallback（CSV 全 miss） | 同上 §5 |

### §3.5 gfx942 迁移（路径 B）

| Root cause | 类型 | 改动 | 详细叙述 |
|---|---|---|---|
| **NEW-RC-1**：FP8 dtype `e4m3fn` → `e4m3fnuz` | 自动机制 | 利用 ATOM/aiter 既有 `normalize_e4m3fn_to_e4m3fnuz` 链，**无须新增改动** | `MIGRATION_REPORT.md` §4 |
| **NEW-RC-2**：weight_scale 方向 `×2.0` | 自动机制 | 利用 forward 语义，**无须新增改动** | `MIGRATION_REPORT.md` §5 |
| **NEW-RC-3**：per_1x128 prefill ASM bypass | working-tree patch | `aiter/fused_moe.py:881-886` 强制 `run_1stage=False` 走 CK 2-stage | `MIGRATION_REPORT.md` §6 |
| **M2 padding**：tp=4 inter_dim 320→384 | 自动机制 | `_process_block_quant` padding，**无须新增改动** | `MIGRATION_REPORT.md` §7 |

**净改动**：1 hunk × 3 行（NEW-RC-3 patch）。**状态更新（2026-04-30）**：该 patch 已 commit 化为 aiter `f06cdcca5`，不再是 working-tree dirty；`feat/step3p5-moe-swiglustep` 分支保留，未 push 到 `origin/main`。详见 §2.2 表 "aiter 仓 patch / commit 状态" + `REPRODUCE.md §3.4` 顶部 NOTE。

### §3.6 长序列 BOS bug（gfx950 tp=4 only）

- aiter `a2883ab37`：删除 `glm5_bf16_tuned_gemm.csv` 中触发 `bf16gemm_bf16_tn_256x256` (N=4096,K=2048) 的 entry
- 详细叙述：`details/topics/07_tp4_longseq_bos_fix.md`

### §3.7 Sliding Window Attention

- aiter `7ebae9afb`：sliding window mask off-by-one 修复
- 详细叙述：`details/topics/03_sliding_window.md`

### §3.8 vllm 集成 patch (Step-3.5-Flash-FP8 wave DOC-1)

vllm v1 + AITER + ATOM plugin 集成路径 source-level patch, 与 §3.3 (SwigluStep aiter+CK+ATOM 三仓 wiring) 和 §3.7 (aiter SWA mask) 互补 — 这两条是 **vllm 端 + ATOM plugin 端** 集成路径上的新发现, 不在三仓内部 commit 范围。

- **Patch A — vllm SwigluStep enum 加白**: `/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py` 加 `MoEActivation.SWIGLUSTEP=3` enum + `_supports_activation` 白名单 + dispatch elif 共 3 hunk +3 行; 让 vllm AITER backend 识别 layer 43-44 的 `ActivationType.SwigluStep` 并路由到 aiter CK swiglustep kernel (否则 dispatch 阶段 raise ValueError 或 oracle gate 剔除)。
- **Patch B — ATOM SWA per-layer KV workspace**: `/home/junlin12/ATOM/atom/plugin/attention.py` 加 `swa_max_num_heads_kv` 字段 + per-layer GQA 最大 KV head collection loop, 2 hunk +14 行; 修复 stepfun 异质 q=48/k=4 (tp=2) 配置下 `swa_workspace` 用 model-wide `num_heads_kv` 算 shape → aiter `mha_varlen_fwd_kernels.cu:478` GQA assert raise → EngineCore crash。

> **Caveat (§22 防 caveat-stripping)**: 本 §3.8 两 patch 解决的是**集成路径上的 dispatch fail / shape assert**, **不解决** wave G-3(i) NaN 问题。NaN root cause = vllm v1 inductor compile / cudagraph + ROCm fp8 互动 (T67/T71 双证), workaround = `--enforce-eager`; Patch B 在 tp=8 路径下**非必需** (T65 实证 revert + tp=8 不触发 GQA assert 也不解 NaN)。NaN root cause 详见 `details/integration/03_lessons.md` 教训 5。

详细叙述: [`details/integration/README.md`](./details/integration/README.md) (含 4 doc 子集) + 顶层 entry [`INTEGRATION_PATCHES.md`](./INTEGRATION_PATCHES.md)。

---

## §4 跨 repo 修改关系图

```
ATOM:  ec8cbe87 (Step3p5 模型 + preshuffle fix)            ← 基础，其余 commits 依赖此
       ├── 4a8495ec (SwigluStep wiring; 依赖 aiter+CK SwigluStep)
       ├── 635e59e9 (BF16 inter_dim padding; 依赖 ec8cbe87)
       ├── 9a67e493 (FP8 block_shape fix; 依赖 635e59e9)
       ├── ccb64621 (FP8 tp=4 三处协调; 依赖 9a67e493 + 635e59e9)
       ├── acff926d (FP8 align bug fix; 依赖 ccb64621)
       └── 969d564  (tp=8 双层 fix; 依赖 acff926d)
aiter: 6d70f7b54 (SwigluStep + CK bump) → 68fc7d48b (V1→V3) → 7ebae9afb (sliding) → 7312ea166 (allreduce)
       → c38d0c9e6 (FP8 guard) → a2883ab37 (delete buggy entry) → a2247989d/dd4257d8f/0f8164017 (NPerBlock=64)
       → f06cdcca5 (NEW-RC-3 commit 化, 2026-04-30 — fix: force per_1x128 fp8 blockscale to CK 2-stage on gfx942)
       + working-tree X2/X2.5 verify-tool patch (aiter/jit/core.py, 非 reproduce 阻塞)
CK:    defd7ad29 (swiglustep epilogue; aiter 6d70f7b54 引用)
```

---

## §5 Upstream pending

| Issue | 状态 | 文档 |
|---|---|---|
| ATOM tp=8 weight load crash (`_load_w2 narrow() size<0`) | **未 file upstream**（draft 已写）；本地解：commit `969d564` 双层 fix | `details/issues/17_atom_moe_tp8_load_crash/ATOM_ISSUE_DRAFT.md` + `_zh.md` |

---

## §6 验证状态

V01-V07 全 PASS（2026-04-26）。详细矩阵见 `details/verification_pipeline/results/SUMMARY.md` + `details/topics/code_changes_all_repos.md` §五。

byte-identical 闭环：M1 tp=2 ↔ M2 tp=4 P3 prompt 143/143 chars 完全一致（`details/projects/14_migration_gfx942/MIGRATION_REPORT.md` §摘要）。

---

## §7 改动文件全索引（grep-friendly）

| Repo | 文件 | 关联 commit |
|---|---|---|
| ATOM | `atom/model_ops/moe.py` | `ec8cbe87` / `4a8495ec` / `635e59e9` / `9a67e493` / `ccb64621` / `acff926d` / `969d564` |
| ATOM | `atom/models/step3p5.py` | `ec8cbe87` / `4a8495ec` |
| ATOM | `atom/model_engine/model_runner.py` | `ec8cbe87` |
| ATOM | `atom/model_loader/loader.py` | `ec8cbe87` |
| ATOM | `atom/model_ops/attentions/aiter_attention.py` | `7ebae9afb` (依赖) |
| aiter | `aiter/fused_moe.py` | `68fc7d48b` / `c38d0c9e6` / **`f06cdcca5`** (NEW-RC-3 commit 化) |
| aiter | `aiter/jit/core.py` | working-tree X2 / X2.5 verify-tool patch（非 reproduce 阻塞，仅在 `f06cdcca5` 之后更新 commit 跑 stepfun engine_init warmup 时撞 TypeError）|
| aiter | `aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv` | `a2883ab37` |
| aiter | `aiter/dist/*` | `7312ea166` |
| aiter | `aiter/jit/*` | `6d70f7b54` (codegen) |
| CK | `gridwise_moe_gemm/*` | `defd7ad29` |

---

> **完整 commit 全文 + diff hunks**：见 `details/topics/code_changes_all_repos.md`（719 行原版，未删）。
> **本文件（CODE_CHANGES.md）的角色**：聚合视图 + per-repo / per-feature 双维度索引；**禁止替代** code_changes_all_repos.md 作为代码改动 source of truth。

---

**End of CODE_CHANGES.md**
