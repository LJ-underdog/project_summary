# Step-3.5-Flash 全栈推理 — 复现指南

> **范围**：本指南覆盖 step35-flash-support 项目的端到端复现路径，含 BF16 + FP8 两种权重格式、tp=2/4/8 三档 tensor parallel、gfx950（MI350X）与 gfx942（MI308X）两种硬件目标。
> **来源整合**：以 `details/topics/12_reproduction_guide_fp8_tp4.md`（gfx950 路径完整版）为路径 A 参考；`details/projects/14_migration_gfx942/MIGRATION_REPORT.md`（gfx942 迁移）为路径 B 参考；`details/topics/repro_info.md`（snapshot at 2026-04-28）为代码版本信息来源；`details/topics/18_fp8_tp8_root_cause_and_fix/`（tp=8 双层 fix）为 tp=8 路径来源。

---

## §1 TL;DR

`stepfun-ai/Step-3.5-Flash`（BF16）与 `stepfun-ai/Step-3.5-Flash-FP8`（FP8 blockscale 量化权重）模型基于 ATOM 推理框架 + AITER kernel 库 + Composable Kernel 在 AMD GPU 上端到端跑通的复现指南。复现完成后预期：

- **gfx950 (MI350X)**：BF16 tp=2/4 与 FP8 tp=2/4 全部 PASS（4/4 prompt 输出连贯，TTFT≈86ms / TPOT≈13ms 量级；tp=8 因 GPU5 硬件异常阻塞）
- **gfx942 (MI308X)**：FP8 tp=2/4/8 三档全部 PASS（A1-A4 anchors，详见 §6）

复现核心依赖：(1) 三仓 pinned commit（ATOM / aiter / CK）；(2) HuggingFace 模型 snapshot；(3) 关键 workaround patch（视目标硬件而定）。

> **路径 A vs 路径 B 选择决策**：
> - 想 gfx950 / FP8 / `simple_inference` 入门：走路径 A（§5.1 + §5.2 / 路径 A 注意），完整长版见 `details/topics/12_reproduction_guide_fp8_tp4.md`
> - 想 gfx942 / FP8 / 全 tp 档（含 tp=8）：走路径 B（§5.1 + §5.2 / 路径 B 注意），完整长版见 `details/projects/14_migration_gfx942/MIGRATION_REPORT.md`
> - 双路径并存 by design：用户 review 后再决定是否压缩。

---

## §2 环境要求

### 2.1 硬件

| 路径 | GPU | 数量 | 说明 |
|---|---|---|---|
| 路径 A（gfx950 参考） | AMD MI350X (gfx950 / CDNA3) | 8（tp=2/4 实际仅用 2/4 卡，避开 GPU5 硬件异常） | 项目原始开发硬件 |
| 路径 B（gfx942 迁移） | AMD MI308X (gfx942 / CDNA3) | 8（UBB 平台标准 8 GPU/节点；`rocm-smi --showid` 实测 GPU[0]–GPU[7]） | 14_migration_gfx942 验证硬件 |

**关键硬件验证命令**：

```bash
rocm-smi --showid          # 应列出 8 张 GPU
rocm-smi --showmemuse      # 显存可用 ≥ 192 GB/卡（gfx942）或 252 GB/卡（gfx950）
```

> **gfx950 GPU5 异常**：单 tensor 操作 ~700ms（正常 < 1ms）。所有 `CUDA_VISIBLE_DEVICES` 必须避开 GPU5。验证脚本见 `details/topics/12_reproduction_guide_fp8_tp4.md` §"前置条件检查 §4"。

### 2.2 软件栈

- **OS / 内核**：Linux + ROCm 7.x 内核驱动（`/dev/kfd`、`/dev/dri/*` 可用）
- **容器（推荐）**：`rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0` 或 `rocm/atom-dev:latest`
- **Python**：3.12.x（路径 A 实测用 `/opt/venv/bin/python`）
- **Shell**：`bash 4+`（用到 brace expansion + `case`）
- **磁盘**：`HF_HOME` ≥ 200 GB（FP8 模型 snapshot ~90 GB；BF16 模型另 ~140 GB）
- **网络**：可访问 `huggingface.co` + `github.com`
- **HuggingFace 账户**：已 `hf auth login` 并接受 `stepfun-ai/Step-3.5-Flash-FP8` 模型条款

### 2.3 不在本指南范围

- ROCm 内核驱动 / dkms 安装
- HuggingFace 账户申请 + license 接受流程
- Docker / 容器运行时安装

---

## §3 依赖准备

### 3.1 三仓 pinned commit

| 仓库 | Commit | Branch on `origin` | 备注 |
|---|---|---|---|
| ATOM | **`969d564`**（路径 B / tp=8 必需） 或 `ccb64621`（路径 A / FP8 tp=4 最低） | `feat/step3p5-flash-support` | `969d564` 含 tp=8 双层 fix（详见 `details/topics/18_fp8_tp8_root_cause_and_fix/`）|
| AITER | `f06cdcca5` | `feat/step3p5-moe-swiglustep` | **不含** NEW-RC-3 dispatch patch，路径 B 需手工 working-tree dirty（§7.1）|
| CK | `defd7ad29` | `feat/swiglustep-moe-no-quant`（aiter 子模块自带）| `swiglustep_and_mul` branches |

```bash
cd $HOME

# 1) ATOM
git clone https://github.com/ROCm/ATOM.git
cd ATOM
git fetch origin feat/step3p5-flash-support
git checkout 969d564     # 或路径 A 的 ccb64621
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

### 3.2 安装 AITER

```bash
cd $HOME/aiter
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

> **cwd 必须不在 aiter 仓内**：运行 python 前 `cd /tmp`（或任意非 aiter repo 目录），否则 aiter 被识别为 namespace package 导致 import 失败。来源：`details/topics/12_reproduction_guide_fp8_tp4.md` §"错误 4"。

---

## §4 数据 / 模型准备

### 4.1 模型选择

| 模型 ID | 用途 | 大小（snapshot） |
|---|---|---|
| `stepfun-ai/Step-3.5-Flash` | BF16 推理（路径 A 主路径之一） | ~140 GB |
| `stepfun-ai/Step-3.5-Flash-FP8` | FP8 blockscale 推理（两路径主路径） | ~90 GB |

### 4.2 设置 HF_HOME 并下载

```bash
export HF_HOME=/workspace/hf_cache    # 或 ≥ 200 GB 的任意路径
export HF_HUB_ENABLE_HF_TRANSFER=0    # ROCm container 中实测更稳定（详见 §7.x caveat）
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

### 5.1 Sanity check（路径 A：gfx950 / FP8 tp=2，或路径 B：gfx942 / FP8 tp=2）

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

> **路径 A（gfx950）注意**：`CUDA_VISIBLE_DEVICES` 必须排除 GPU5。tp=2 推荐 `0,1`；tp=4 推荐 `0,1,2,3`；tp=8 因 GPU5 阻塞**默认不跑**（如必须跑参考 `details/perf/15_perf_tp2_tp4_tp8_eval/PERF_REPORT.md`）。

> **路径 B（gfx942）注意**：tp=8 必须使用 ATOM `969d564`（含双层 fix），否则 weight load 阶段 crash 或 4/4 prompt 全乱码。详见 §7.7 + `details/topics/18_fp8_tp8_root_cause_and_fix/`。

### 5.3 Throughput 测试（可选）

step35-flash-support 仓内未提供与 fp8-tp4-repro 等价的 throughput_bench.py 通用脚本。perf 数据见 `details/perf/15_perf_tp2_tp4_tp8_eval/` 与 `details/perf/16_perf_gfx950_verified/`，但其使用的是 `details/scripts/perf_correctness_bench.py`（perf + correctness 联跑），参数与 throughput 矩阵不同。

如需复现 throughput 矩阵：
- 借用 fp8-tp4-repro 的 `throughput_bench.py`（路径：`/home/junlin12/project_fp8_tp4_repro/correctness_eval/throughput_bench.py`），作为外部依赖
- 或参考 `details/perf/16_perf_gfx950_verified/RESULTS.md` 的命令模板手写

详见 Appendix C。

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

### 6.2 性能 anchors

| 配置 | 硬件 | TTFT | TPOT | 来源 |
|---|---|---|---|---|
| BF16 tp=2 | gfx950 (MI350X) | ~91 ms | ~16 ms | step35 README |
| BF16 tp=4 | gfx950 | ~76 ms | ~15 ms | step35 README |
| FP8 tp=2 | gfx950 | ~91 ms | ~16 ms | step35 README + repro_info.md |
| FP8 tp=4 | gfx950 | ~86–93 ms | ~13 ms | `details/topics/12_reproduction_guide_fp8_tp4.md` |
| FP8 tp=2/4/8 | gfx942 (MI308X) | `[TODO: 待补具体数值 — 14_migration_gfx942 仅给 PASS/FAIL anchor，未给 TTFT/TPOT 量化；需读 details/perf/15_perf_tp2_tp4_tp8_eval/PERF_REPORT.md 抽取]` | 同左 | `details/perf/15_perf_tp2_tp4_tp8_eval/` |

### 6.3 PASS 判定（端到端 A1-A4，源自 reverify TEAM_CONFIG）

- **A1**：exit 0 + log 含 `Engine Core fully initialized` + `Loading safetensors shards 44/44`
- **A2**：log `grep -cE "Traceback|OOM|dispatch.*miss|no.instance|division by zero|NaN|Inf"` = 0
- **A3**：4/4 prompt 全部 coherent（按 §6.1 表）
- **A4**：tp=2 P2 与 tp=4 P2 应 byte-identical（`1+2+3` logit margin 极宽，sampling 不翻转）；tp=8 允许非 byte-identical 但 4/4 必须 coherent

---

## §7 常见问题 / 已知坑

### §7.0 Known-issue 快查矩阵

| Symptom 关键字 | 跳到 § |
|---|---|
| `dispatch miss` / `no instance found` / 乱码 (gibberish) on tp=8（gfx942） | §7.1 |
| 全 BOS 输出（`<s><s><s><s>...`，tp=4 长序列）（gfx950） | §7.2 |
| `ValueError: ... block_n ... not divisible` | §7.3 |
| 推理结果乱码但无 crash | §7.4 |
| `import aiter` namespace 错误 | §7.5 |
| `Loading checkpoint shards: 0%` 卡住 | §7.6 |
| tp=8 `_load_w2 narrow() size<0` crash（gfx942 / 老 ATOM commit） | §7.7 |
| `HIP out of memory` / `BadAlloc` | §7.8 |
| `Address already in use` (port 8016/7/8) / GPU 显存残留 | §7.9 |
| 单 GPU 速度极慢（TPOT > 500ms）（gfx950） | §7.10 |
| `snapshot_download` 401 / 403 | §7.11 |

### §7.1 AITER NEW-RC-3 patch（gfx942 tp=8 dispatch miss）

**症状**：路径 B（gfx942）tp=8 accuracy 测试 log 出现 `dispatch miss` / `no instance found` / `RuntimeError: ck::*`，或生成乱码（如 `小弟sets邪倾倒` 大段非中文非英文 gibberish）。

**原因**：commit `f06cdcca5` 不含 `aiter/fused_moe.py` L881-886 的 NEW-RC-3 dispatch patch。该 patch 在 reverify_wave 实测中以 working-tree dirty 形式存在，未上 commit。

**解决**：在 `$HOME/aiter/aiter/fused_moe.py` 中定位 `_fused_moe_kernel_per_token_blockscale` dispatch 段（约 L880 附近），把 `gfx942` + `per_1x128` fp8 blockscale 路径强制走 CK 2-stage。完整 patch 见 `details/projects/14_migration_gfx942/MIGRATION_REPORT.md` §M2.NEW-RC-3 或 fp8-tp4-repro `fix_wave/progress/teammate-8.md`。修改后 working-tree dirty 即可，**不要 commit 到 aiter 仓**；在 `$HOME/aiter/` 下重新 `python3 setup.py develop` 让 patch 编译进 .so 生效。

### §7.2 BOS 全 token 输出（gfx950 / tp=4 长序列）

**症状**：tp=4 长序列（>8209 tokens）prefill 后 decode 全 `<s>`，连续 512 token 均 BOS。

**原因**：aiter `glm5_bf16_tuned_gemm.csv` 中含 `bf16gemm_bf16_tn_256x256` 用于 (N=4096,K=2048) 的 entry。该 ASM kernel 对非 256 对齐 M（如 8209-8223）输出错误，影响 Step-3.5-Flash o_proj 在 tp=4 长序列 prefill。

**解决**：使用 aiter commit `a2883ab37` 或更新（已删除该 entry）；或手动删除 `aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv` 中触发 `_ZN5aiter24bf16gemm_bf16_tn_256x256E` 的 N=4096,K=2048 行（保留 `splitk_clean` variant）。详见 `details/topics/07_tp4_longseq_bos_fix.md`。

> **gfx942 不触发**：dispatcher 在 gfx942 上 fallback `hipblaslt`，BOS bug 不会发生。

### §7.3 `ValueError: ... block_n ... not divisible`

**症状**：

```
ValueError: The output_size of gate's and up's weight = 320 is not divisible by block_n = 128
```

**原因**：FP8 align bug —— ATOM `moe.py` `_process_block_quant` 用 `align = 64 if inter_dim <= 192 else 128`（旧逻辑），inter_dim=320 padding 到 192，但 192 % 128 ≠ 0。

**解决**：确认 ATOM 在 commit `acff926d` 或更新（路径 A 为 `ccb64621`，路径 B 为 `969d564` 的 base）。`moe.py` ~L1726 应为 `align = block_n`（无条件分支）。详见 `details/topics/06_fp8_tp4.md` + `details/topics/12_reproduction_guide_fp8_tp4.md` §3.3。

### §7.4 推理结果乱码（无 crash）

**症状**：生成 token 序列语义完全错误，但无 ValueError / 无 traceback。

**原因（路径 A / gfx950）**：FP8 scale ceil 整除未修复 —— `_load_w13` / `_load_w2` 用 floor 整除，`load_shard_size = 10 // 4 = 2`，scale block [8,9] 永远未被加载，残留 `torch.ones()` 默认值 → fp8 dequant 严重偏离。

**解决**：确认 ATOM `moe.py` `_load_w13` (~L2310-2312) + `_load_w2` (~L2352-2354) 的 `load_shard_size` 使用 ceil 整除（含 `+ self.tp_size - 1`）。修复在 commit `ccb64621` 中。

**原因（路径 B / gfx942 tp=8）**：第二层 silent corruption —— 仅 ceil 整除 + early-return 不够，trailing rank fp32 scale 残留 `torch.ones()` 让 fp8 raw bits 当 bf16 用。修复见 §7.7。

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

### §7.7 tp=8 `_load_w2 narrow() size<0` crash（gfx942 / 老 ATOM commit）

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

### §7.10 单 GPU 速度极慢（gfx950）

**症状**：TPOT > 500ms。

**原因**：`CUDA_VISIBLE_DEVICES` 包含 GPU5（硬件异常 ~700ms/tensor）。

**解决**：检查 `CUDA_VISIBLE_DEVICES` 设置，排除 GPU5。

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

---

## §8 延伸（指向 details/ 子目录的指针）

> 所有路径均与 `details/` 重构后的实际目录对齐（与 README.md / REORG_PLAN.md 终态一致），无 dead link。

### 8.1 单 topic 深度（按 root cause 类）

| 想了解什么 | 去读 |
|---|---|
| MoE GEMM 数值正确性根因 | `details/topics/01_moe_pipeline.md` |
| SwigluStep 激活函数 wiring | `details/topics/02_swiglu_step.md` |
| Sliding window mask off-by-one | `details/topics/03_sliding_window.md` |
| TP=4/8 MoE kernel alignment | `details/topics/04_tp_support.md` |
| FP8 block-quantized 推理（tp=2 入门）| `details/topics/05_fp8_inference.md` |
| FP8 tp=4 三层 bug（含 scale sharding ceil）| `details/topics/06_fp8_tp4.md` |
| tp=4 长序列 BOS 修复（gfx950 ASM kernel） | `details/topics/07_tp4_longseq_bos_fix.md` |
| MoE no-padding 调研（为什么 inter_dim=320 必须 padding 到 384）| `details/research/08_moe_no_padding_research.md` + `details/research/09_moe_no_padding_deep_dive.md` |
| gfx950 FP8 mfma KPack=32 ISA 级约束 | `details/research/10_fp8_mfma_kpack32_constraint.md` |
| 张量并行原理 + 每个算子 TP 行为 | `details/research/11_tensor_parallelism_strategy.md` |
| FP8 tp=4 详细复现指南（gfx950 路径 A 完整版）| `details/topics/12_reproduction_guide_fp8_tp4.md` |
| Recall 工具实战 | `details/meta/13_recall_system_analysis.md` |

### 8.2 跨 topic / 集成类

| 想了解什么 | 去读 |
|---|---|
| gfx950 → gfx942 (MI308X) 迁移完整报告 | `details/projects/14_migration_gfx942/MIGRATION_REPORT.md` |
| gfx942 上 TP=2/4/8 perf 数据 | `details/perf/15_perf_tp2_tp4_tp8_eval/PERF_REPORT.md` |
| gfx950 perf 基线 | `details/perf/16_perf_gfx950_verified/RESULTS.md` |
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
| V01-V07 phase0 环境预检脚本 | `details/verification_pipeline/phase0_preflight.sh`（保留原位 by R7）|

---

## Appendix A — 与 fp8-tp4-repro REPRO_REF 的关系

本指南与 `/home/junlin12/project_fp8_tp4_repro/repro_guide_wave/REPRODUCTION_GUIDE.md`（fp8-tp4-repro 项目）的关系：

- **fp8-tp4-repro REPRO_REF**：聚焦 gfx942 / MI308X / FP8 / `correctness_bench.py` + `throughput_bench.py` 工具链，提供 §6 quantitative tolerance + §7 throughput 矩阵。
- **本 step35 REPRODUCE**：覆盖 step35-flash-support 的更大范围（BF16 + FP8 / gfx950 + gfx942 / 12 + 14 + 18 三个复现相关 topic 的合并）；引用 `details/topics/12_reproduction_guide_fp8_tp4.md` 与 `details/topics/18_fp8_tp8_root_cause_and_fix/` 作为子路径深度。

如果你**只关心 gfx942 / FP8 / `correctness_bench.py`** 复现路径，且你能拿到 fp8-tp4-repro 仓库（含 `correctness_eval/` 与 `reverify_wave/outputs/` 锚点），优先读 REPRO_REF 全文，再回 §7 Troubleshooting 即可。

如果你**没有 fp8-tp4-repro 仓库**，本指南 §5.1 + `details/topics/12_reproduction_guide_fp8_tp4.md` 提供 `atom.examples.simple_inference` 模块的等价路径（不依赖外部 bench 脚本）。

---

## Appendix B — 已知 TODO（保留待 user review 时补全）

- `[TODO: §6.2]` gfx942 (MI308X) 上 tp=2/4/8 的 TTFT/TPOT 量化数值需从 `details/perf/15_perf_tp2_tp4_tp8_eval/PERF_REPORT.md` 抽取
- `[TODO: §3.1]` ATOM commit `969d564` vs `ccb64621` 的关系图（路径 A vs 路径 B 选择决策树）需要更显式说明（建议加流程图）
- `[TODO: §5.2]` 双路径 A/B 是否最终压缩为单路径，待 user review 后决定

> §5.3 throughput 测试 (REP-3) + §8 details 路径表 (REP-4) 已在本版完成对齐，不再列 TODO。

---

## Appendix C — Throughput 测试外部脚本说明（REP-3）

step35-flash-support 仓内 perf 数据（`details/perf/15_perf_tp2_tp4_tp8_eval/` + `details/perf/16_perf_gfx950_verified/`）使用 `details/scripts/perf_correctness_bench.py` 联跑生成；该脚本以 correctness 验证为主，未导出 throughput 矩阵格式（QPS / token/s 维度）。

如需复现与 fp8-tp4-repro 一致的 throughput 矩阵，使用其仓内 `correctness_eval/throughput_bench.py`：

```bash
# 假设 fp8-tp4-repro 仓在 /home/junlin12/project_fp8_tp4_repro
python /home/junlin12/project_fp8_tp4_repro/correctness_eval/throughput_bench.py \
  --model stepfun-ai/Step-3.5-Flash-FP8 \
  --tensor-parallel-size 4 \
  --kv_cache_dtype fp8 --port 8017 \
  --num-prompts 200 --output-len 256
```

**外部依赖声明**：本仓不携带 throughput_bench.py；若需独立可 fork（建议项 → 见 Appendix B TODO）。

---

**End of REPRODUCE.md**
