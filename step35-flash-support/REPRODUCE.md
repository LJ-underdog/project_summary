# Step-3.5-Flash 全栈推理 — 复现指南（gfx942 / MI308X / FP8）

> **范围**：本指南覆盖 step35-flash-support 项目在 **AMD MI308X (gfx942)** 上以 **FP8 blockscale** 量化权重端到端复现 `stepfun-ai/Step-3.5-Flash-FP8` 模型推理的完整步骤，含 tp=2/4/8 三档 tensor parallel。
> **来源整合**：以 `details/projects/14_migration_gfx942/MIGRATION_REPORT.md`（gfx942 迁移报告）为主路径参考；`details/topics/18_fp8_tp8_root_cause_and_fix/`（tp=8 双层 fix）为 tp=8 路径来源；**§6.2 性能数据来源 = wave `tp2_verify_post_merge_wave`（首次正确实测 stepfun MoE）**。⚠️ `details/perf/15_perf_tp2_tp4_tp8_eval/PERF_REPORT.md` 的 TTFT≈186ms 等数值是 **Qwen3-0.6B（dense, non-MoE）误归属**（§6.2 注 + §7.13），**不是** stepfun perf 源。
> **gfx950 路径**：本指南只保留 gfx942 复现路径。如需 gfx950 (MI350X) / BF16 路径，参见 §8 延伸索引指向的 `details/topics/12_reproduction_guide_fp8_tp4.md` 与 `details/perf/16_perf_gfx950_verified/`（保留作历史参考）。

---

## §1 TL;DR

`stepfun-ai/Step-3.5-Flash-FP8`（FP8 blockscale 量化权重）模型基于 ATOM 推理框架 + AITER kernel 库 + Composable Kernel 在 AMD MI308X (gfx942) 上端到端跑通。复现完成后预期：

- **gfx942 (MI308X) FP8 tp=2/4/8 三档全部 PASS**（functional A1-A4 anchors，详见 §6.1 / §6.3）
- **首次实测 stepfun-Flash-FP8 MoE perf 三档**（详见 §6.2；wave `tp2_verify_post_merge_wave` 产出，无历史 baseline 可对比 —— 旧 perf 数据归属勘误见 §6.2 注）

复现核心依赖：
1. 三仓 pinned commit（ATOM `969d564` / aiter `f06cdcca5` / CK `defd7ad29`）
2. HuggingFace 模型 snapshot（`stepfun-ai/Step-3.5-Flash-FP8` ~90 GB）
3. **NEW-RC-3 dispatch patch**（aiter `fused_moe.py:881-886`，per_1x128 prefill ASM bypass）— 已固化为 aiter commit `f06cdcca5`，§3.1 checkout 后自动包含，**无需**手工 apply；patch 来源 + 失效场景 + 历史 working-tree dirty 模式见 §3.4 + §7.1

---

## §2 环境要求

### 2.1 硬件

| GPU | 数量 | 说明 |
|---|---|---|
| AMD MI308X (gfx942 / CDNA3) | 8（UBB 平台标准 8 GPU/节点；`rocm-smi --showid` 实测 GPU[0]–GPU[7]） | 14_migration_gfx942 验证硬件 |

**关键硬件验证命令**：

```bash
rocm-smi --showid          # 应列出 8 张 GPU
rocm-smi --showmemuse      # 显存可用 ≥ 192 GB/卡（MI308X）
```

### 2.2 软件栈

- **OS / 内核**：Linux + ROCm 7.x 内核驱动（`/dev/kfd`、`/dev/dri/*` 可用）
- **容器（推荐）**：`rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0` 或 `rocm/atom-dev:latest`
- **Python**：3.12.x
- **Shell**：`bash 4+`（用到 brace expansion + `case`）
- **磁盘**：`HF_HOME` ≥ 100 GB（FP8 模型 snapshot ~90 GB）
- **网络**：可访问 `huggingface.co` + `github.com`
- **HuggingFace 账户**：已 `hf auth login` 并接受 `stepfun-ai/Step-3.5-Flash-FP8` 模型条款

### 2.3 不在本指南范围

- ROCm 内核驱动 / dkms 安装
- HuggingFace 账户申请 + license 接受流程
- Docker / 容器运行时安装
- gfx950 (MI350X) / BF16 路径（参见 §8 延伸索引）

---

## §3 依赖准备

### 3.1 三仓 pinned commit

> ⚠️ 本节 = **pad/通用 tp8 历史复现**（pad-256 anchor，2026-05-09，pre-nopad）；**nopad inter=160 / EP（W8）复现** 用不同 pin，见 `details/perf/22_ep_cudagraph_perf_accuracy_2026-06-03.md §6.1`（ATOM feat-tip `0526446` = e18b467/PR #641 + stepfun SWA per-layer kv-head 补丁 + aiter `360ebdb66` + CK `e90ecddea`）。两套各服务各路径、均保留。

| 仓库 | Commit | Branch on `origin` | 备注 |
|---|---|---|---|
| ATOM | **`969d564`** | `feat/step3p5-flash-support` | 含 tp=8 双层 fix（详见 `details/topics/18_fp8_tp8_root_cause_and_fix/`）|
| AITER | **`f06cdcca5`** | `feat/step3p5-moe-swiglustep` | **已含** NEW-RC-3 dispatch patch（commit message: `fix(moe): force per_1x128 fp8 blockscale to CK 2-stage on gfx942`，内容与 §3.4 内嵌 patch byte-id 一致）。如严格 checkout 此 commit，§3.4 无需手工 git apply；详见 §3.4 顶部的 NOTE。|
| CK | `defd7ad29` | `feat/swiglustep-moe-no-quant`（aiter 子模块自带）| `swiglustep_and_mul` branches |

```bash
cd $HOME

# 1) ATOM
git clone https://github.com/ROCm/ATOM.git
cd ATOM
git fetch origin feat/step3p5-flash-support
git checkout 969d564
cd ..

# 2) AITER（含 CK 子模块）
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
git fetch origin feat/step3p5-moe-swiglustep
git checkout f06cdcca5
git submodule sync && git submodule update --init --recursive
( cd 3rdparty/composable_kernel && git log -1 --oneline )
# 期望包含：defd7ad29 Add swiglustep_and_mul branches to gridwise_moe_gemm
cd ..
```

> **三仓 commit reachability caveat**：以上 pinned commit 为 step35 + fp8-tp4-repro wave 实测验证的快照；若上游做 force-push / history rewrite 抹掉 commit object，`git checkout <hash>` 会报 `unknown revision`。届时参考 `details/projects/14_migration_gfx942/` 与 `details/topics/code_changes_all_repos.md` 自行复刻。

### 3.2 安装 AITER（先打 NEW-RC-3 patch，再 develop）

```bash
cd $HOME/aiter
# 注意：如果 §3.1 严格 checkout 了 aiter `f06cdcca5`（或更新的 ancestor 含此 commit 的 HEAD），
# NEW-RC-3 dispatch patch 已在 commit 中固化，此处直接 develop 即可，无需手工 git apply。
# 仅当 aiter HEAD 早于 `f06cdcca5`（即旧 working-tree dirty 模式）时，才需先按 §3.4 / §7.1 手工应用 patch。
python3 setup.py develop
```

观察锚点：
- 编译输出含 `Building extension ...` + `g++ -shared` 链接行
- 完成后无 `error:` / `Traceback`
- `python -c "import aiter; print(aiter.__file__)"` 应能 import

> **首次 CK 编译耗时**：约 10-30 分钟（默认 `ENABLE_CK=1`）。

### 3.3 安装 ATOM

```bash
cd $HOME/ATOM
pip install -e .
pip install ninja
pip install -U "huggingface_hub" "transformers>=4.45" "tokenizers"
hf auth login    # 或 export HF_TOKEN=hf_xxxxx
```

观察锚点：
- 末尾 `Successfully installed atom-...`
- `python -c "from atom import LLMEngine, SamplingParams; print('ok')"` 不报 ImportError
- `python -c "from atom.model_engine.arg_utils import EngineArgs; print('ok')"` 不报 ImportError

> **cwd 必须不在 aiter 仓内**：运行 python 前 `cd /tmp`（或任意非 aiter repo 目录），否则 aiter 被识别为 namespace package 导致 import 失败。

### 3.4 NEW-RC-3 dispatch patch（aiter — 已 commit 化）

> **🔴 NOTE（2026-05-09 更新 — 必读，与 §3.1 cross-link）**：
>
> aiter commit **`f06cdcca5`** 本身就是 NEW-RC-3 patch 的 commit 化（commit message: `fix(moe): force per_1x128 fp8 blockscale to CK 2-stage on gfx942`，diff 与下方 patch 内嵌 hunk byte-id 一致 —— 见 `git show f06cdcca5 -- aiter/fused_moe.py`）。
>
> - 如果 §3.1 严格 `git checkout f06cdcca5`（或后续 ancestor 链含此 commit 的 HEAD），本节 patch **已自动包含**：`git status` 不会显示 dirty，**无需** `git apply`，**无需** 因 patch 重新 `setup.py develop`（首次 develop 已含此 patch 编译产物）。
> - 本节余下文字保留作 **历史参考 + 路径 explainer**（描述 patch 内容、root cause、当年为何 working-tree 而非 commit）。仅以下场景仍需手工 apply：
>   - 你 checkout 了**早于** `f06cdcca5` 的 aiter HEAD（如 `0f8164017` / `c38d0c9e6` 等历史 baseline）
>   - 你想在 fork 里 cherry-pick 这条修复到不同分支
> - 闭环证据：tp2_verify_post_merge_wave/progress/teammate-L25-audit-commit-currency.md §1.2（实测 commit message + diff byte-id）+ teammate-L29-fix-REPRODUCE-CODE_CHANGES-toplevel.md。

**作用**：aiter `fused_moe.py:881-883` 历史启发式 `run_1stage = token > 32 and (inter_dim % 256 == 0)` 会把 per_1x128 prefill 路由到 ASM kernel `aiter.fmoe_g1u1`（签名**不带 block shape 参数**，gfx942 上数值会错 / gibberish）。本 patch 强制 `run_1stage = False`，使 dispatch 走 CK 2-stage blockscale 路径。**已固化为 aiter commit `f06cdcca5`，§3.1 严格 checkout 后自动生效，无需手工 apply / 无需重 develop。**

**Patch（单 hunk，3 行实质改动 — 仅当 aiter HEAD 早于 `f06cdcca5` 时手工应用）**：

```diff
--- a/aiter/fused_moe.py
+++ b/aiter/fused_moe.py
@@ -880,7 +880,10 @@
             if q_type == QuantType.per_1x128:
                 # for fp8 blockscale, ck has better performance so disable assembly kernel
-                run_1stage = token > 32 and (inter_dim % 256 == 0)
+                # NEW-RC-3 patch (2026-04-28): force CK blockscale path on gfx942 to avoid
+                # routing per_1x128 prefill to ASM fmoe_g1u1 which lacks block shape param
+                # original: run_1stage = token > 32 and (inter_dim % 256 == 0)
+                run_1stage = False
```

（仅当 aiter HEAD 早于 `f06cdcca5`（如 `0f8164017`）才需手工 `git apply` 上方 diff + 重 `python3 setup.py develop`。）

> **历史背景（可跳过）**：此 patch 2026-04-28 引入时是 working-tree dirty，2026-04-30 起 commit 化为 `f06cdcca5`（`feat/step3p5-moe-swiglustep` 分支，未 push `origin/main`）。它是 gfx942 + per_1x128 的 workaround（无条件禁 1-stage ASM），非 production-ready upstream fix；真正上游修法 = dispatch 表给 `(per_1x128,gfx942,prefill)` 单列带 block-shape 的 ASM 入口（不在本指南范围）。引用：`details/projects/14_migration_gfx942/MIGRATION_REPORT.md` §6.4/§9.2（描述对应 wave 14 时点的 working-tree dirty 状态）。

---

## §4 数据 / 模型准备

### 4.1 模型选择

| 模型 ID | 用途 | 大小（snapshot） |
|---|---|---|
| `stepfun-ai/Step-3.5-Flash-FP8` | FP8 blockscale 推理（gfx942 主路径） | ~90 GB |

### 4.2 设置 HF_HOME 并下载

```bash
export HF_HOME=/workspace/hf_cache    # 或 ≥ 100 GB 的任意路径
export HF_HUB_ENABLE_HF_TRANSFER=0    # ROCm container 中实测更稳定
mkdir -p "$HF_HOME"
```

**先检测既有 cache**（避免重下 90 GB）：

```bash
MODEL_DIR="$HF_HOME/hub/models--stepfun-ai--Step-3.5-Flash-FP8"
SNAP_DIR="$MODEL_DIR/snapshots"
if [ -d "$SNAP_DIR" ] && [ -n "$(ls -A "$SNAP_DIR" 2>/dev/null)" ]; then
  SNAP_REV="$(ls "$SNAP_DIR" | head -n 1)"
  SHARD_CNT="$(ls "$SNAP_DIR/$SNAP_REV"/model-*-of-*.safetensors 2>/dev/null | wc -l)"
  echo "snapshot=$SNAP_REV shards=$SHARD_CNT"
  [ "$SHARD_CNT" -ge 44 ] && echo "[hf_cache] SKIP — existing cache complete."
fi
```

**fallback：完整下载**（仅当上述未命中时）：

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='stepfun-ai/Step-3.5-Flash-FP8',
    cache_dir='$HF_HOME/hub',
)
"
```

观察锚点：
- 无 `HTTPError` / `EntryNotFoundError` / `401` / `403`
- `du -sh $HF_HOME/hub/models--stepfun-ai--Step-3.5-Flash-FP8` ≈ 90 GB
- `ls $HF_HOME/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/*/` 应含 44 个 `model-*-of-*.safetensors`

---

## §5 运行步骤

### 5.1 Sanity check（gfx942 / FP8 tp=2）

最小可用配置；首次跑通 stack。

```bash
cd /tmp    # 必须不在 aiter 仓内
mkdir -p /tmp/sanity

HF_HOME=/workspace/hf_cache HF_HUB_ENABLE_HF_TRANSFER=0 \
TRUST_REMOTE_CODE=1 AITER_LOG_LEVEL=WARNING \
CUDA_VISIBLE_DEVICES=0,1 \
python -m atom.examples.simple_inference \
  --model stepfun-ai/Step-3.5-Flash-FP8 \
  --tensor-parallel-size 2 \
  --kv_cache_dtype fp8 \
  --trust-remote-code \
  --max-tokens 64 \
  > /tmp/sanity/tp2_simple.log 2>&1

echo "exit=$?"
grep -E "Engine Core fully initialized|Loading safetensors shards|Generated text" /tmp/sanity/tp2_simple.log
```

通过判定（按出现顺序）：
- `Loading safetensors shards 44/44`
- `Engine Core fully initialized`
- 至少 4 段 `Generated text:`（simple_inference 自带 4 prompt）
- `exit=0`

未通过 → §7 Troubleshooting。

### 5.2 完整 accuracy 验证（tp=2 / tp=4 / tp=8 串行）

> **注意**：此节命令采用 fp8-tp4-repro 项目 `correctness_eval/correctness_bench.py` 的运行模板。如复现者无该脚本，可改用 `atom.examples.simple_inference` 模块直接跑（§5.1 模板，调高 `--max-tokens` 至 512），或使用本仓 `details/scripts/perf_correctness_bench.py`（perf + correctness 联跑等价脚本）。脚本来源：`/home/junlin12/project_fp8_tp4_repro/correctness_eval/correctness_bench.py`。

```bash
# 三档串行（GPU 独占）；每档前 cleanup 防止 port/显存残留
for TP in 2 4 8; do
  case $TP in 2) PORT=8018; DEVS=0,1 ;; 4) PORT=8017; DEVS=0,1,2,3 ;; 8) PORT=8016; DEVS=0,1,2,3,4,5,6,7 ;; esac

  pkill -9 -f 'correctness_bench.py|simple_inference' 2>/dev/null; sleep 3

  HF_HOME=/workspace/hf_cache HF_HUB_ENABLE_HF_TRANSFER=0 \
  TRUST_REMOTE_CODE=1 AITER_LOG_LEVEL=WARNING \
  CUDA_VISIBLE_DEVICES=$DEVS \
  python correctness_eval/correctness_bench.py \
    --model stepfun-ai/Step-3.5-Flash-FP8 \
    --tensor-parallel-size $TP \
    --kv_cache_dtype fp8 --port $PORT --trust-remote-code --max-tokens 512 \
    --output-json outputs/tp${TP}.json \
    > logs/tp${TP}.log 2>&1
  echo "tp=$TP exit=$?"
done
```

> **tp=8 关键依赖**：必须使用 ATOM `969d564`（含双层 fix）+ aiter NEW-RC-3 working-tree patch（§3.4），否则 weight load 阶段 crash 或 4/4 prompt 全乱码。详见 §7.7 + `details/topics/18_fp8_tp8_root_cause_and_fix/`。

### 5.3 Throughput 测试（可选）

step35-flash-support 仓内未提供与 fp8-tp4-repro 等价的 throughput_bench.py 通用脚本。perf 数据见 `details/perf/15_perf_tp2_tp4_tp8_eval/`，但其使用的是 `details/perf/15_perf_tp2_tp4_tp8_eval/perf_bench.py`（perf-only，复用 ATOM 内置 ttft/tpot 字段），参数与 throughput 矩阵不同。

如需复现 throughput 矩阵：
- 借用 fp8-tp4-repro 的 `throughput_bench.py`（路径：`/home/junlin12/project_fp8_tp4_repro/correctness_eval/throughput_bench.py`），作为外部依赖
- 或参考 `details/perf/15_perf_tp2_tp4_tp8_eval/PERF_REPORT.md` 的命令模板手写

详见 Appendix B。

---

## §6 预期结果

### 6.1 Accuracy anchors（4/4 prompt 内容期望）

| Prompt idx | 输入 | 期望特征 |
|---|---|---|
| P0 | `introduce yourself` | 英文 introduce-myself reasoning，开头近似 `Hmm, the user simply asked me to introduce myself...` |
| P1 | `list all prime numbers within 100` | 英文 prime numbers reasoning，开头含 `We are asked to list all prime numbers within 100...` |
| P2 | `1+2+3=?` | **必须最终给出 6**；推理路径 `1+2=3, then 3+3=6`；finish_reason=eos |
| P3 | `如何在一个月内增肌10公斤` | 中文回答，至少命中 1 个近义词：`不现实` / `几乎不可能` / `不健康` / `不科学` / `超出生理极限` / `高风险` / `激进` / `健康风险`；无乱码 |

来源：`/home/junlin12/project_fp8_tp4_repro/reverify_wave/progress/teammate-reverify.md` §2.2 + `details/topics/12_reproduction_guide_fp8_tp4.md` §6.1。

### 6.2 性能 anchors（gfx942 / MI308X / FP8 / stepfun-Flash-FP8 MoE）

> **勘误背景**：本节原引用 `details/perf/15_perf_tp2_tp4_tp8_eval/PERF_REPORT.md` 数值（TTFT 0.186/0.110/0.071s 等）实际是 Qwen3-0.6B（dense, non-MoE）path —— ATOM `EngineArgs --model` default 陷阱（§7.13）导致 raw log 实跑 Qwen 而非 stepfun MoE。本表已替换为 wave `tp2_verify_post_merge_wave` 首次实测 stepfun-Flash-FP8 MoE 数据；原 Qwen 数值归属勘误参见 `details/perf/15_perf_tp2_tp4_tp8_eval/PERF_REPORT.md`（已由 wave L19b 标注归属）。

> 🔴 **配置标注（2026-06-03 补注，teammate-30 考古坐实）**：**下表 anchor 全部是纯 TP**（`--enable-expert-parallel` **未启用**，EP 引入前 2026-05-09 实测）。inter 沿 TP 分片：tp=2→inter_dim=640 / tp=4→320 / tp=8→**160**（见 §7.1 "tp=8 inter_pad=256"）。四重证据：perf 脚本无 EP arg、命令无 `--enable-expert-parallel`、ATOM `EngineArgs` 默认 `enable_expert_parallel=False`、§7.1 inter 分片 640/256；来源 wave 0 EP 命中。**EP（inter=1280）配置的 perf 是另一组数字，见下方「6.2-EP」小节，二者不同 parallelism 路径不可直接比。**

测试条件（**纯 TP**）：`stepfun-ai/Step-3.5-Flash-FP8`（**显式** `--model $STEP35_PATH` + `--kv_cache_dtype fp8`，避免 §7.13 陷阱）；input target 10240 → actual 10213 tokens；output 由 eos 提前停（max_tokens=1024，三档 actual output 不同：240 / 266 / 937）；concurrency=1；temperature=0；method=A（复用 ATOM 内置 ttft/tpot 字段）；脚本 `details/scripts/perf_correctness_bench.py`；runs=2 取 last as stable。

| 配置 | TTFT | TPOT | total_latency | decode throughput | actual input/output | engine_init |
|---|---|---|---|---|---|---|
| FP8 tp=2 | **1665.1 ms** | **15.5 ms/tok** | 5.380 s | 64.3 tok/s | 10213 / 240 (eos) | 82.32 s |
| FP8 tp=4 | **980.4 ms** | **14.5 ms/tok** | 4.816 s | 69.1 tok/s | 10213 / 266 (eos) | 125.72 s |
| FP8 tp=8 | **747.1 ms** | **13.7 ms/tok** | 13.550 s | 73.1 tok/s | 10213 / 937 (eos) | 223.31 s |

观察（首次实测 stepfun MoE，无历史 baseline 对比）：
- **TTFT** 随 tp 单调下降（1665.1 → 980.4 → 747.1 ms），tp=2 → tp=8 提速 2.23×（次线性，prefill 阶段 weight × hidden 计算被 TP 切分，受 collective 通信开销影响）。
- **TPOT** 随 tp 单调微降（15.5 → 14.5 → 13.7 ms/tok），tp=2 → tp=8 仅 11.6% 加速 —— decode 阶段 batch=1 + all-reduce 通信 overhead 主导，TP 扩展回报递减（典型 MoE expert routing + all-to-all 瓶颈）。
- **decode throughput** 随 tp 单调上升（64.3 → 69.1 → 73.1 tok/s），tp=8 vs tp=2 仅 +13.7% 提升，与 TPOT 趋势一致。
- **total_latency** 三档不可直接对比：tp=8 因 eos 时机不同导致 output_tokens=937（≈4× tp=4 的 266），故总时间反而增长；**单 token 延迟仍单调下降**才是 TP 扩展的正确读法。
- **engine_init** 随 worker 数非线性增长（82.32 → 125.72 → 223.31 s），weight 加载并行化非线性 + per-worker IPC/CUDA context 初始化叠加。

> **数值 vs 旧 Qwen3-0.6B 误归属表的量级差异**：stepfun-Flash-FP8 是 MoE（含 expert routing + per_1x128 fp8 blockscale dispatch），TTFT/TPOT 量级显著大于 Qwen3-0.6B dense path（~1665 ms vs ~186 ms TTFT 在 tp=2 档），这是模型路径本质差异（非 perf regression）。

数据来源：
- tp=2：`tp2_verify_post_merge_wave/progress/teammate-L18-perf-rerun.md` §2（Run B stable，stepfun_fp8_tp2_v2_full.log raw 实测，4 个 worker `Model load done:` 全部 stepfun snapshot）
- tp=4 / tp=8：`tp2_verify_post_merge_wave/progress/teammate-L20-perf-tp4-tp8.md`（同脚本 + 显式 `--model $STEP35_PATH`，4/8 个 worker raw log 强制核对均 stepfun snapshot 路径）
- baseline 误归属勘误：`tp2_verify_post_merge_wave/progress/teammate-L17c-baseline-audit.md` §1（raw log `tp2_run2_full.log:47,50` 实证 = Qwen3-0.6B）

#### EP vs 纯 TP 速查（读 §6.2 / §6.2-EP 前必看）

> 两节 anchor 是**不同 parallelism 路径**，下表帮你定位「哪组数字属于哪条路径、哪条触发 nopad」。详细辨析见 [`details/TP8_THREE_BUGS.md`](./details/TP8_THREE_BUGS.md)。

| 维度 | 纯 TP（§6.2） | EP / expert-parallel（§6.2-EP） |
|---|---|---|
| 启用方式 | 无 `--enable-expert-parallel`（默认） | `--enable-expert-parallel` |
| `inter_dim`（tp=8） | 1280/8 = **160**（沿 TP 分片，`<256`） | **1280**（不分片，`≥256`） |
| nopad smalltile | **触发**（NPerBlock=32） | **不触发** |
| `ATOM_FP8_MOE_DISABLE_PAD` | 有效（nopad/pad 不同路径） | **no-op**（nopad≡pad，微差=噪声） |
| B2 ÷8 b_scale bug | **活在此路径**（fix=`360ebdb66`；op-isolate + ✅ **e2e-TP 已验**（2026-06-03 T56，见 §6.2-纯TP-nopad）） | 不受 B2 影响（inter=1280 对齐） |
| perf anchor | §6.2（2026-05-09，纯 TP） | §6.2-EP + `perf/22`（2026-06-03，EP） |
| 精度验证 | op-isolate（inter=160）+ ✅ **e2e-TP 4/4 连贯**（2026-06-03 T56，§6.2-纯TP-nopad） | e2e「看着连贯」，非严格数值验证 |

🔴 **两组 perf 数字不可直接比**（不同路径 + 不同口径 gpu-mem/max-len/input 档）。

### 6.2-EP 性能 anchors（EP / expert-parallel，2026-06-03 实测）

> 🔴 **铁律**：本小节全部是 **EP（`--enable-expert-parallel`，inter=1280 未沿 TP 分片）**，**不可**与上方 §6.2 纯 TP anchor 混读 —— 不同 parallelism 路径。来源：W8_resume wave teammate-27/28；完整记录 `details/perf/22_ep_cudagraph_perf_accuracy_2026-06-03.md`。
>
> ▶ **EP 完整复现脚本 + commit pin 见 `perf/22 §6` + `details/scripts/ep_e2e_cudagraph.sh`（e2e）/ `ep_perf_bench.sh`（perf）。**

测试条件（**EP**）：`stepfun-ai/Step-3.5-Flash-FP8` + 显式 `--model $STEP35_PATH` + `--kv_cache_dtype fp8` + `--enable-expert-parallel`；TP8；**cudagraph ON（无 `--enforce-eager`）**，`cudagraph_capture_sizes=[1]`；单序列 batch=1；temperature=0；`ignore_eos`（口径干净）。

| 口径（**全 EP, cudagraph, TP8, batch=1**） | 值 | 来源 |
|---|---|---|
| decode TPOT（短输出 256 步） | **~12.62 ms/tok** | teammate-27 |
| decode throughput（同上） | **~77 tok/s** | teammate-27 |
| cudagraph vs eager 加速 | **~7.8–8.2×**（eager TPOT ~99–104ms → ~12.62ms） | teammate-25/27 |
| prefill TTFT（input 10213 tok） | **~560–571 ms** | teammate-28 |
| decode TPOT（长输入档） | **~13.5 ms/tok** | teammate-28 |
| decode throughput（长输入档） | **~74 tok/s** | teammate-28 |

观察 / caveat：
- **EP TPOT 13.5ms 与 §6.2 TP anchor 13.7ms 巧合性接近，但二者是不同路径（EP vs TP），不构成可比对照**；TTFT 560–571ms < TP anchor 747ms 是口径差异（gpu-mem-util / max-model-len / input 不同档），非可比 regression。
- 🔴 **EP 下 "nopad vs pad" = 同一路径**：EP inter=1280 ≥ 256，`ATOM_FP8_MOE_DISABLE_PAD` 是 **no-op**（teammate-29 坐实）。T27/T28 progress 标注的 "nopad/pad" 微差（TTFT ~1.9%、TPOT 0.06%）**是 run-to-run 噪声，不可解读为 nopad/pad perf 对比**。上表是 **EP 整体 perf**。
- **EP 精度（teammate-24）**：ATOM 原生 e2e（EP+cudagraph）4/4 连贯、非 Qwen、有自然 eos。**⚠️ 但未对 ground-truth/参考做严格数值验证**（"pad_parity" 是 EP-vs-EP 自比 = 无意义；EP inter=1280 对齐本就不受 ÷8 bug 影响）→ **EP 精度 = "看着连贯"，非严格验证**。

#### cudagraph 现已可用（2026-06-03，teammate-26）
- 早前 "TP8 cudagraph 崩" 的根因 = **custom-allreduce IPC 不兼容**（`hipIpcGetMemHandle invalid argument` @ `allocate_kv_cache` barrier），**非 nopad 特异（pad 也崩）**（详 teammate-25）。
- **ATOM 原生 `simple_inference` / `EngineArgs.create_engine()` 默认全关 IPC-allreduce**（custom-allreduce OFF + quick-reduce NONE，走 RCCL）→ **去掉 `--enforce-eager` 即可用 cudagraph，无需改代码**。仅 custom-allreduce ON 的路径（vllm-direct/plugin）才需显式关。
- 故 §5.x 若要跑 cudagraph perf：用 ATOM 原生栈、**不要**加 `--enforce-eager`（旧 workaround，已不必要）。

#### EP 引入时间线（teammate-30）
- 框架能力：ATOM `aaf83fa`（2025-08-12，default `enable_expert_parallel=False`）。
- 本项目 serving：`start_vllm_v5.sh`（2026-05-14，首次 `--enable-expert-parallel` tp=8）。
- 本项目 e2e / nopad wave：`dryrun_e2e_tp8.py` / V65B（2026-05-28，为修 `output_size=160 not divisible by block_n=128` 抄来 EP）。
- → 自 2026-05-28 起 e2e 用 EP（inter=1280）→ **nopad（inter=160 TP）路径不再被 e2e 覆盖**；§6.2 纯 TP anchor（2026-05-09）是唯一覆盖过 inter 分片路径的数据。

#### nopad（TP inter=160）bug/fix 现状 → 见 NOPAD_TP_HANDOFF
- nopad smalltile（inter=160, NPerBlock=32）的 **e2e 正确性 + cudagraph perf 实测见下方 §6.2-纯TP-nopad**（2026-06-03 首次，已闭合 NOPAD_TP_HANDOFF §4.1「TP e2e 验证缺失」缺口）；÷8 b_scale bug 根因、stage1 镜像 bug（仍未修，须 quant 层）、host 广播 fix 等细节，**见 `NOPAD_TP_HANDOFF.md`**（当前散见 W8_resume/progress/teammate-19b stage2-fix、-22/-23 stage1、-24 e2e、-25 cudagraph、-30 EP/TP 考古、-56/-57 纯 TP e2e+perf）。

### 6.2-纯TP-nopad 性能 + e2e anchor（纯 TP / inter=160 / nopad smalltile，2026-06-03 实测，**首次**）

> 🟢 **闭合 `NOPAD_TP_HANDOFF.md` §4.1「TP e2e 验证缺失」缺口**：以往 inter=160 nopad fix **仅 op-isolate 验证**，e2e 层从未触发（2026-05-28 起 e2e 误用 EP=inter1280，nopad 路径失覆盖）。本节是**史上首次**在真实纯 TP nopad e2e 上验证 fix 真执行 + 实测 perf。来源：W8_resume teammate-56（e2e）/ teammate-57（cudagraph perf），GPU 实测。
>
> **栈**：ATOM `0526446`（e18b467 + SWA 补丁）+ aiter `360ebdb66`（stage2 ÷8 fix 禁广播）+ ck `e90ecddea`；stepfun Step-3.5-Flash-FP8 @ `6eebda59`；gfx942 TP8。

**① e2e 正确性（T56，enforce-eager）**：`simple_inference` **无 `--enable-expert-parallel`** + `ATOM_FP8_MOE_DISABLE_PAD=1` → **4/4 prompt 连贯**。运行时插桩 definitively 坐实 **`inter_dim=160` + `_w4_nopad=True` + stage2-fix broadcast DISABLED 三者真执行**（fix 确实跑在 inter=160 nopad 路径上，非 op-isolate 层）。

**② cudagraph perf（T57，非 eager）**：`perf_correctness_bench.py` 无 EP + `DISABLE_PAD=1` + `--enable-cudagraph`（**无 enforce-eager**），input 10213 / output 256 / gpu-util 0.5 / runs 2：

| 口径（**纯 TP, cudagraph, nopad inter=160, TP8, batch=1**） | 值 |
|---|---|
| prefill TTFT | **599.3 ms** |
| decode TPOT | **14.2 ms/tok** |
| decode throughput | **70.4 tok/s** |
| 稳定性 | cv 0.0%（runs=2）、正确性 PASS、cudagraph 真捕获（0.41s）、两轮无显存泄漏 |

🔴 **口径区分（以下三组数字均不可直接比）**：
- **vs §6.2 纯 TP anchor**（eager + **pad-256**，tp8 ~747 ms / 13.7 ms）：同为纯 TP，但本节 = **cudagraph + nopad inter=160** → **eager vs cudagraph + pad vs nopad 双重差异，不可直接比**。
- **vs §6.2-EP / `details/perf/22_*.md`**（**EP** inter=1280 + cudagraph，TTFT ~560–571 / TPOT ~13.5）：**不同 parallelism（EP vs 纯 TP）**，不可比。
- ⏳ **纯 TP pad-256 cudagraph perf 正在补测（Phase B）**：同口径（cudagraph + 纯 TP）下 **pad-256 vs nopad-160** 的对照（真正衡量 nopad 省掉 padding 的开销）待该数据齐后定稿。

| 同口径对照（纯 TP + cudagraph） | TTFT | TPOT | decode tput |
|---|---|---|---|
| nopad inter=160（本节，已测） | **599.3 ms** | **14.2 ms/tok** | **70.4 tok/s** |
| pad-256（Phase B，**待测**） | _待测_ | _待测_ | _待测_ |

### 6.3 PASS 判定（端到端 A1-A4）

- **A1**：exit 0 + log 含 `Engine Core fully initialized` + `Loading safetensors shards 44/44`
- **A2**：log `grep -cE "Traceback|OOM|dispatch.*miss|no.instance|division by zero|NaN|Inf"` = 0
- **A3**：4/4 prompt 全部 coherent（按 §6.1 表）
- **A4**：tp=2 P2 与 tp=4 P2 应 byte-identical（`1+2+3` logit margin 极宽，sampling 不翻转）；tp=8 允许非 byte-identical 但 4/4 必须 coherent

---

## §7 常见问题 / 已知坑

### §7.0 Known-issue 快查矩阵

| Symptom 关键字 | 跳到 § |
|---|---|
| `dispatch miss` / `no instance found` / 乱码 (gibberish) on tp=8 | §7.1 |
| `ValueError: ... block_n ... not divisible` | §7.3 |
| 推理结果乱码但无 crash | §7.4 |
| `import aiter` namespace 错误 | §7.5 |
| `Loading checkpoint shards: 0%` 卡住 | §7.6 |
| tp=8 `_load_w2 narrow() size<0` crash（老 ATOM commit） | §7.7 |
| `HIP out of memory` / `BadAlloc` | §7.8 |
| `Address already in use` (port 8016/7/8) / GPU 显存残留 | §7.9 |
| `snapshot_download` 401 / 403 | §7.11 |
| perf / correctness bench 跑出 Qwen3 风格 `<think>` 输出（应为 stepfun） | §7.13 |
| aiter `moe_sorting` dispatch 默认 OPUS（旧 CK 路径需 env var 显式回退） | §7.14 |

### §7.1 AITER NEW-RC-3 patch（tp=8 dispatch miss / per_1x128 prefill 乱码）

**症状**：tp=2/4/8 accuracy 测试 log 出现 `dispatch miss` / `no instance found` / `RuntimeError: ck::*`，或生成乱码（如 `小弟sets邪倾倒` 大段非中文非英文 gibberish）。

**原因**：aiter `fused_moe.py:881-886` 的历史启发式 `run_1stage = token > 32 and (inter_dim % 256 == 0)` 把 per_1x128 prefill 路由到 ASM `aiter.fmoe_g1u1`；该 ASM 签名不带 block shape 参数，gfx942 上数值会错。tp=2 时 inter_dim=640、tp=8 时 inter_pad=256（`160` ceil 到 `256`）均满足 `% 256 == 0` 触发该 bug；tp=4 时 inter_pad=384（`% 256 != 0`）幸运绕过。该启发式已由 NEW-RC-3 patch 替换为 `run_1stage = False`；patch 已固化为 aiter commit **`f06cdcca5`**（详见 §3.4 顶部 NOTE）。

**解决**：

- 如果 §3.1 严格 `git checkout f06cdcca5`（或 ancestor 链含此 commit 的 HEAD）→ patch 已自动包含，无需手工动作；`Engine Core fully initialized` 后 4/4 prompt 应正常。
- 仅当 aiter HEAD 早于 `f06cdcca5`（如 `0f8164017` / `c38d0c9e6`）时 → 按 §3.4 手工 apply patch + 重新 `python3 setup.py develop` 让改动编译进 `.so`。
- 详见 §3.4 顶部 🔴 NOTE + `details/projects/14_migration_gfx942/MIGRATION_REPORT.md` §6（注：MIGRATION_REPORT.md 描述对应 wave 14 进行时 working-tree dirty 状态）。

### §7.3 `ValueError: ... block_n ... not divisible`

**症状**：

```
ValueError: The output_size of gate's and up's weight = 320 is not divisible by block_n = 128
```

**原因**：FP8 align bug —— ATOM `moe.py` `_process_block_quant` 用 `align = 64 if inter_dim <= 192 else 128`（旧逻辑），inter_dim=320 padding 到 192，但 192 % 128 ≠ 0。

**解决**：确认 ATOM 在 commit `969d564`（含 `_process_block_quant` 修复）。`moe.py` ~L1726 应为 `align = block_n`（无条件分支）。详见 `details/topics/06_fp8_tp4.md` + `details/projects/14_migration_gfx942/MIGRATION_REPORT.md` §7。

### §7.4 推理结果乱码（无 crash）

**症状**：生成 token 序列语义完全错误，但无 ValueError / 无 traceback。

**原因（FP8 tp=2/4 通用）**：FP8 scale ceil 整除未修复 —— `_load_w13` / `_load_w2` 用 floor 整除，`load_shard_size = 10 // 4 = 2`，scale block [8,9] 永远未被加载，残留 `torch.ones()` 默认值 → fp8 dequant 严重偏离。

**解决**：确认 ATOM `moe.py` `_load_w13` (~L2310-2312) + `_load_w2` (~L2352-2354) 的 `load_shard_size` 使用 ceil 整除（含 `+ self.tp_size - 1`）。修复在 commit `969d564` 中。

**原因（tp=8 第二层 silent corruption）**：仅 ceil 整除 + early-return 不够，trailing rank fp32 scale 残留 `torch.ones()` 让 fp8 raw bits 当 bf16 用。修复见 §7.7。

### §7.5 `import aiter` namespace package 错误

**症状**：

```
ImportError: cannot import name 'ActivationType' from 'aiter' (unknown location)
# 或
AttributeError: module 'aiter' has no attribute 'fused_moe'
```

**原因**：在 `$HOME/aiter/` 目录下运行了 python，当前目录的 `aiter/` 文件夹被识别为 namespace package。

**解决**：`cd /tmp` 后再运行 python。

### §7.6 HF cache miss（首跑卡在 model loading）

**症状**：log 长时间停在 `Loading checkpoint shards: 0%` 或反复 `Downloading model.safetensors`。

**解决**：
1. `du -sh $HF_HOME/hub/models--stepfun-ai--Step-3.5-Flash-FP8` 检查 ≥ 80 GB
2. `ls $HF_HOME/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/*/` 应有 44 个 safetensors
3. 不全则重跑 §4 `snapshot_download`

### §7.7 tp=8 `_load_w2 narrow() size<0` crash（老 ATOM commit）

**症状**：log 中出现 `RuntimeError: ... narrow(): start (X) ...` 或 `_load_w2` / `_load_w13` traceback；进程在 weight load 阶段 crash（未到 `Engine Core fully initialized`）。

**原因**：你 checkout 了 ATOM 老 commit（如 `acff926` 或更早）。Step-3.5-Flash-FP8 `moe_inter=1280` + `per_1x128` → D=10 个 fp32 scale block。tp=8 时 `ceil(10/8)=2`，starts=`[0,2,4,6,8,10,12,14]`，rank 5/6/7 命中 `start ≥ D=10` 触发 narrow size≤0。

**解决**：确认 ATOM 在 commit `969d564`（含双层 fix）：
1. trailing rank early-return（rank 命中越界 starts 时跳过 load）
2. fp32 scale tensor `.zero_()` 初始化（替换 `torch.ones()` 残留 —— 仅 early-return 不够，残留 `1.0` 会让 dequant 把 fp8 raw bits 当 bf16 用，生成乱码）

详见 `details/topics/18_fp8_tp8_root_cause_and_fix/TP8_ROOT_CAUSE_AND_FIX.md` + `details/projects/14_migration_gfx942/MIGRATION_REPORT.md` §M3。

### §7.8 OOM at long context

**症状**：`HIP out of memory` / `BadAlloc` / `KV cache cannot fit`。

**典型场景**：tp=2 + input_len=16384 单卡 KV cache 超过 192 GB HBM。

**解决**：
1. 跳过该 OOM 组合（如 tp=2 长 context 不是合理部署配置）
2. 降低 `--num-prompts` 或 `--max-num-batched-tokens`
3. 不要降 `--gpu-memory-utilization` 0.9 上限以下

### §7.9 Port already in use / GPU 显存残留

**症状**：`RuntimeError: ... bind ... Address already in use`，或起 engine 时 `HIP error: out of memory` 但 `rocm-smi` 显示 GPU 应空闲。

**解决**：
```bash
pkill -9 -f 'correctness_bench.py|simple_inference|vllm'
sleep 5
rocm-smi --showpids       # 应显示 0 进程
rocm-smi --showmemuse     # 显存应回到接近 0%
```

### §7.11 snapshot_download 401 / 403

**症状**：`HTTPError 401/403` 或 `GatedRepoError`。

**解决**：
```bash
hf auth login
huggingface-cli whoami
# 浏览器访问 https://huggingface.co/stepfun-ai/Step-3.5-Flash-FP8 接受条款
```

### §7.12 缓存清理（修改 ATOM/aiter 代码后必须）

```bash
# ATOM JIT 缓存
rm -rf /root/.cache/atom/*

# aiter JIT 缓存（修改 CK codegen 代码后必须；只删 .so 不够，必须同时删 build/）
rm -f $HOME/aiter/aiter/jit/module_moe_ck2stages_*.so
rm -rf $HOME/aiter/aiter/jit/build/module_moe_ck2stages_*
```

### §7.13 ATOM `EngineArgs --model` default = `Qwen/Qwen3-0.6B` 陷阱（perf / correctness bench 必读）

**症状**：用 `details/perf/15_perf_tp2_tp4_tp8_eval/perf_bench.py` / `details/scripts/perf_correctness_bench.py` / 任何基于 ATOM EngineArgs 的脚本跑 perf 或 correctness，未显式传 `--model`，结果生成的输出是 Qwen3 思考模板（`<think>\nOkay, let's see ...`），perf 数值看上去"成功"但与 stepfun-Flash MoE 路径完全无关。

**原因**：ATOM `EngineArgs.add_cli_args` 给 `--model` 注册了 default = `Qwen/Qwen3-0.6B`。脚本里的 `if not getattr(args, "model", None): args.model = _find_model_path()` 永远进不去（`args.model` 总是 truthy），STEP35_MODEL_PATH / `_find_model_path()` 推断逻辑被 silent 覆盖，实际加载的就是 Qwen3-0.6B dense 模型，跟目标 `stepfun-ai/Step-3.5-Flash-FP8` MoE 路径毫无关系。

**解决**：调用任何 ATOM-based perf / correctness bench 脚本时，**必须显式**在命令行传 `--model $STEP35_PATH`：

```bash
STEP35_PATH=$HF_HOME/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/<sha>
# perf bench
/opt/venv/bin/python details/perf/15_perf_tp2_tp4_tp8_eval/perf_bench.py \
  --model $STEP35_PATH --tp 2 --kv_cache_dtype fp8 ...
# correctness bench
/opt/venv/bin/python details/scripts/perf_correctness_bench.py \
  --model $STEP35_PATH --tp 2 --kv_cache_dtype fp8 ...
```

**强制验证**：跑完后 grep raw log 第一行 `Model load done:`，必须显式核对：

```bash
grep -m2 'Model load done' logs/<your_run>_full.log
# 期望：[atom HH:MM:SS] Model load done: /workspace/hf_cache/.../models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/...
# 反例：[atom HH:MM:SS] Model load done: Qwen/Qwen3-0.6B   ← 触发本陷阱，数据无效
```

如果 raw log 上 `Model load done:` 字段是 `Qwen/Qwen3-0.6B`，本次 run 数据**全部作废**，必须加 `--model $STEP35_PATH` 重跑。

**来源**：`fp8-tp4-repro / tp2_verify_post_merge_wave / progress/teammate-L17c-baseline-audit.md` §1.1（raw log `tp2_run2_full.log:47,50` 实证；perf-t1.md baseline 误归属为 stepfun，实际是 Qwen3-0.6B）+ `progress/teammate-L18-perf-rerun.md` §4.1（Run A 二次踩坑：未传 `--model` 即跑出 Qwen3 `<think>` 输出，Run B 加 `--model $STEP35_PATH` 后才跑对）。

### §7.14 aiter `moe_sorting` dispatch 默认翻转（OPUS 现为默认；2026-05 主线行为）

**KNOWN_FACT（不阻塞复现，但与 perf / dispatch 文档基线行为不同）**：

aiter 自上游主线 commit **`acf1dbd3f use opus moe as default (#3011)`** 起（祖先链中位于 `f06cdcca5..315123ace` 之间），`moe_sorting` 子 op dispatch 默认从历史的 "OPUS opt-in"（旧 `_USE_OPUS_MOE_SORTING`）翻转为 **"CK opt-in"**（新 `_USE_CK_MOE_SORTING`，默认 `0` → `use_opus=True`）。当前 `aiter/fused_moe.py` 实测：

```python
# aiter/fused_moe.py:27 (实测 in-tree)
_USE_CK_MOE_SORTING = os.environ.get("AITER_USE_CK_MOE_SORTING", "0") == "1"
# aiter/fused_moe.py:123
use_opus=not _USE_CK_MOE_SORTING,
```

**对复现的影响**：

- **不影响**正确性：MoE 主体 fp8 blockscale GEMM dispatch（`module_moe_ck2stages_f8_f8_preshuffle_on_b16_*_per_1x128_mulWeightStage2`）路径不变；§6.1 PASS 判定 + §6.3 A1-A4 anchors 不变。
- **影响 dispatch trace**：现在 `moe_sorting` 子 op 走 `moe_sorting_opus_*` kernel（而非历史的 `moe_sorting_ck_*`）。`details/perf/15_perf_tp2_tp4_tp8_eval/PERF_REPORT.md` + `details/research/19_kernel_dispatch_report/REPORT.md` + `details/perf/16_perf_gfx950_verified/RESULTS.md` 提到的 dispatch path 描述对应 OPUS opt-in 时期，与当前 in-tree 默认不同。
- **如需复现历史 CK sorting 行为**：`export AITER_USE_CK_MOE_SORTING=1` 即可强制走旧路径。

**来源**：`tp2_verify_post_merge_wave/progress/teammate-L25-audit-commit-currency.md` §2.3（grep 全 repo 0 处提及该翻转 + `git log f06cdcca5..315123ace` 确认含 `acf1dbd3f` + 在线实测 `aiter/fused_moe.py:27,123`）。

---

> **本节 last-verified**：§7.1–§7.14 + §6.2 纯 TP anchor = **2026-05-09**（wave `tp2_verify_post_merge_wave` L29 收尾；commit currency 见 L25；perf coverage 审计见 L26）。**§6.2-EP / cudagraph / EP 引入时间线 / nopad 指针 = 2026-06-03**（W8_resume wave teammate-24/25/26/27/28/29/30）。

---

## §8 延伸（指向 details/ 子目录的指针）

> 所有路径均与 `details/` 重构后的实际目录对齐（与 README.md 终态一致），无 dead link。
> **注意**：本指南只覆盖 gfx942 复现路径；`details/` 下保留了完整的 gfx950 / BF16 / 多硬件深度文档（作历史参考）。

### 8.1 单 topic 深度（按 root cause 类）

| 想了解什么 | 去读 |
|---|---|
| MoE GEMM 数值正确性根因 | `details/topics/01_moe_pipeline.md` |
| SwigluStep 激活函数 wiring | `details/topics/02_swiglu_step.md` |
| Sliding window mask off-by-one | `details/topics/03_sliding_window.md` |
| TP=4/8 MoE kernel alignment | `details/topics/04_tp_support.md` |
| FP8 block-quantized 推理（tp=2 入门）| `details/topics/05_fp8_inference.md` |
| FP8 tp=4 三层 bug（含 scale sharding ceil）| `details/topics/06_fp8_tp4.md` |
| tp=4 长序列 BOS 修复（gfx950 ASM kernel；gfx942 不触发） | `details/topics/07_tp4_longseq_bos_fix.md` |
| MoE no-padding 调研（为什么 inter_dim=320 必须 padding 到 384）| `details/research/08_moe_no_padding_research.md` + `details/research/09_moe_no_padding_deep_dive.md` |
| gfx950 FP8 mfma KPack=32 ISA 级约束 | `details/research/10_fp8_mfma_kpack32_constraint.md` |
| 张量并行原理 + 每个算子 TP 行为 | `details/research/11_tensor_parallelism_strategy.md` |
| FP8 tp=4 详细复现指南（gfx950 路径完整版，历史参考）| `details/topics/12_reproduction_guide_fp8_tp4.md` |
| Recall 工具实战 | `details/meta/13_recall_system_analysis.md` |

### 8.2 跨 topic / 集成类

| 想了解什么 | 去读 |
|---|---|
| **gfx950 → gfx942 (MI308X) 迁移完整报告（本指南主路径源）** | `details/projects/14_migration_gfx942/MIGRATION_REPORT.md` |
| **gfx942 纯 TP perf 数据（§6.2 来源 = `tp2_verify_post_merge_wave`）** | §6.2 本文内（数值权威源）；wave progress `teammate-L18/L20` |
| EP perf + cudagraph（§6.2-EP 来源）| `details/perf/22_ep_cudagraph_perf_accuracy_2026-06-03.md` |
| ⚠️ `perf/15_*/PERF_REPORT.md`（**TTFT≈186ms 是 Qwen3-0.6B 误归属，非 stepfun**；勿当 §6.2 源）| `details/perf/15_perf_tp2_tp4_tp8_eval/PERF_REPORT.md` |
| gfx950 perf 基线（历史参考） | `details/perf/16_perf_gfx950_verified/RESULTS.md` |
| ATOM tp=8 load crash issue draft | `details/issues/17_atom_moe_tp8_load_crash/README.md` |
| **FP8 tp=8 双层 root cause + fix（ATOM `969d564`）** | `details/topics/18_fp8_tp8_root_cause_and_fix/TP8_ROOT_CAUSE_AND_FIX.md` |
| FP8 tp=2/4 每类 op 的 torch / CK / ASM kernel 归属 | `details/research/19_kernel_dispatch_report/REPORT.md` |
| 三仓全部代码改动 commit 索引（聚合视图） | `CODE_CHANGES.md`（顶层）|
| 三仓全部代码改动 commit 索引（原版 719 行）| `details/topics/code_changes_all_repos.md` |
| FP8 tp=4 复现环境信息（snapshot at 2026-04-28）| `details/topics/repro_info.md` |
| V01-V07 端到端验证 pipeline | `details/verification_pipeline/MASTER_PIPELINE.md` |

### 8.3 复现脚本

| 想了解什么 | 去读 |
|---|---|
| 标准化 perf + correctness 联跑脚本（gfx950/gfx942 通用） | `details/scripts/perf_correctness_bench.py` |
| V01-V07 phase0 环境预检脚本 | `details/verification_pipeline/phase0_preflight.sh` |

---

## Appendix A — 已知 TODO

- 无未决 TODO（性能数据已从 `details/perf/15_perf_tp2_tp4_tp8_eval/PERF_REPORT.md` 抽取并填入 §6.2；NEW-RC-3 patch "为何不 commit" 已在 §3.4 Note 解释；gfx950 双路径已按 user review 移除）。

---

## Appendix B — Throughput 测试外部脚本说明

step35-flash-support 仓内 perf 数据（`details/perf/15_perf_tp2_tp4_tp8_eval/`）使用其内置 `perf_bench.py` 联跑生成；该脚本以 ttft/tpot 单 prompt 评估为主，未导出 throughput 矩阵格式（QPS / token/s 维度）。

如需复现与 fp8-tp4-repro 一致的 throughput 矩阵，使用其仓内 `correctness_eval/throughput_bench.py`：

```bash
# 假设 fp8-tp4-repro 仓在 /home/junlin12/project_fp8_tp4_repro
python /home/junlin12/project_fp8_tp4_repro/correctness_eval/throughput_bench.py \
  --model stepfun-ai/Step-3.5-Flash-FP8 \
  --tensor-parallel-size 4 \
  --kv_cache_dtype fp8 --port 8017 \
  --num-prompts 200 --output-len 256
```

**外部依赖声明**：本仓不携带 throughput_bench.py。

---

**End of REPRODUCE.md**
