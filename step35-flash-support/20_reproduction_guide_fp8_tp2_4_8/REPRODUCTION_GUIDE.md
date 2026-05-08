# Step-3.5-Flash-FP8 on ATOM + AITER + CK — Reproduction Guide

> Wave 13 / repro_guide_wave 起草版（PRE fresh-verify）
> 本 guide 重现：`stepfun-ai/Step-3.5-Flash-FP8` 在 8×MI308X 上以 tp=2/4/8 运行 ATOM offline inference，含 accuracy 与 throughput。

---

## §0 Prerequisites（不在本 guide 范围）

假设你已有：

- **硬件**：8×AMD Instinct MI308X（gfx942 / CDNA3，单节点 UBB 平台，Infinity Fabric 全互连）
  - 验证：`rocm-smi --showid` 应列出 GPU[0]–GPU[7]（8 张）
- **OS / 内核**：Linux + ROCm 7.x 内核驱动已加载（`/dev/kfd`、`/dev/dri/*` 可用）
- **容器**：已进入 `rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0` 或 `rocm/atom-dev:latest` 容器（含 PyTorch + ROCm 7.x stack）
- **Shell**：`bash 4+`（本 guide 命令使用 brace expansion 与 `case` 控制结构；macOS 默认 `bash 3.2` / POSIX `dash` 不支持，请显式 `bash` 入口执行）
- **Python venv**：可装 pip 包（如 `/opt/venv/bin/python`）；本 guide 命令默认用 `python` 即虚拟环境的解释器
- **磁盘**：≥ 200 GB 给 `HF_HOME`（Step-3.5-Flash-FP8 全量 safetensors 约 90 GB；剩余空间留给 tokenizer / 临时下载缓冲）
- **网络**：可访问 `huggingface.co`、`github.com`
- **HuggingFace 账号**：已 `hf auth login` 或 `export HF_TOKEN=...`（详见 §3 末尾）

不在范围：docker 安装、rocm-dkms 安装、HuggingFace 账号申请。

---

## §0.5 Licensing & Model Access

复现前请确认你有合规权利使用以下组件：

- **三仓 license**：ATOM / aiter / CK 均托管于 ROCm GitHub 组织，请按各仓 LICENSE 条款使用（本 guide 仅引用其公开 commit，不再分发源码）。
- **HuggingFace model**：`stepfun-ai/Step-3.5-Flash-FP8` 在 HF 上需登录账号并接受模型条款；首次 `snapshot_download` 前请在浏览器访问模型页面 accept license。
- **本 guide 与复现数据**（`reverify_wave/outputs/tp{2,4,8}.json` 锚点 / `repro_guide_wave/outputs/*` throughput）仅用于 ROCm/AMD MI308X 上的功能验证，不构成性能官方数据，不得引用为对外 SOTA 基线。

---

## §1 Clone Three Repos + Project at Pinned Commits

工作根目录建议 `/home/$USER/`。三仓 commit 已在 `feat/step3p5-flash-support` 远程分支固定。

```bash
cd $HOME

# 1) ATOM
git clone https://github.com/ROCm/ATOM.git
cd ATOM
git fetch origin feat/step3p5-flash-support
git checkout 969d564
cd ..

# 2) AITER
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
git fetch origin feat/step3p5-moe-swiglustep
git checkout f06cdcca5
git submodule sync && git submodule update --init --recursive
cd ..

# 3) CK (Composable Kernel) — AITER 子模块自带，仅当需要 override 时单 clone
# 验证 aiter 自带 CK 已在正确 commit：
( cd $HOME/aiter/3rdparty/composable_kernel && git log -1 --oneline )
# 期望输出包含：defd7ad29 Add swiglustep_and_mul branches to gridwise_moe_gemm
# 若不匹配，单独 clone：
# git clone https://github.com/ROCm/composable_kernel.git ck
# ( cd ck && git checkout defd7ad29 )

# 4) project_fp8_tp4_repro（含 correctness_bench.py / throughput_bench.py / reverify_wave 锚点）
#
# === 路径分叉 ===
# (A) 本 wave 内部 fresh-verify teammate（#FRESH-VERIFY）：直接复用本机 lead session 的
#     project_fp8_tp4_repro 目录，跳过 git clone：
#         export REPRO_ROOT=/home/junlin12/project_fp8_tp4_repro
#     wave 13 promote（Phase 3 #LEAD-T13）完成前，下面 (B) 路径的远程目录还不存在。
#
# (B) 外部接手者（promote 完成后）：通过 LJ-underdog/project_summary
#     的 step35-flash-support/20_reproduction_guide_fp8_tp2_4_8/ 分发；
#     promote 包含以下最低文件集（#PROMOTE owner 须保证 staging 一并打包）：
#         README.md / REPRODUCTION_GUIDE.md / WAVE_CLOSE.md / progress/SUMMARY.md
#         correctness_eval/correctness_bench.py
#         correctness_eval/throughput_bench.py
#         reverify_wave/outputs/tp{2,4,8}.json                  # accuracy 锚点
#         fix_wave/progress/teammate-8.md                       # NEW-RC-3 patch 来源（§8.1 引用）
git clone https://github.com/LJ-underdog/project_summary.git
export REPRO_ROOT=$HOME/project_summary/step35-flash-support/20_reproduction_guide_fp8_tp2_4_8
ls "$REPRO_ROOT"
# 应含：README.md / REPRODUCTION_GUIDE.md / WAVE_CLOSE.md / progress/SUMMARY.md
#       correctness_eval/{correctness_bench.py, throughput_bench.py}
#       reverify_wave/outputs/tp{2,4,8}.json
#       fix_wave/progress/teammate-8.md
```

> **REPRO_ROOT 约定**：本 guide 后续所有 `python correctness_eval/...` 命令都基于 `cd $REPRO_ROOT` 后执行。本机原始 lead session 的 `REPRO_ROOT=/home/junlin12/project_fp8_tp4_repro`，其下含完整的 `correctness_eval/` 与 `reverify_wave/`；外部 reader 通过 promote 仓的 19_… 目录获得相同目录结构（按上方期望文件清单核验）。
>
> **三仓 commit reachability 注意**：本 guide 的三个 pinned commit 是 wave 11–13 实测验证的快照；后续若三仓做 force-push / history rewrite 抹掉这些 commit object，`git checkout <hash>` 会报 `unknown revision` —— 此时请联系 lead 复活历史 ref 或参考 `fix_wave` 文档自行复刻。

**Pinned commits（务必精确匹配）**：

| 仓库 | Commit | Branch on `origin` | 备注 |
|---|---|---|---|
| ATOM | `969d564` | `feat/step3p5-flash-support` | **含 TP8 D<tp_size 双层 fix**（详见 §8.7 / `MIGRATION_REPORT.md` §M3 / `TP8_ROOT_CAUSE_AND_FIX.md`）|
| AITER | `f06cdcca5` | `feat/step3p5-moe-swiglustep` | **不含** NEW-RC-3 dispatch patch，需手工 working-tree dirty（§8.1）|
| CK | `defd7ad29` | `feat/swiglustep-moe-no-quant` | swiglustep_and_mul branches |

---

## §2 Build & Install AITER

参考 `~/aiter/README.md` §Installation。

```bash
cd $HOME/aiter

# 编译并以开发模式安装（C++ 扩展含 ASM / CK / Triton 三类后端）
python3 setup.py develop
```

观察锚点：
- 编译过程会输出 `Building extension ...` 及 `g++ -shared` 链接行
- 完成后无 `error:` / `Traceback`
- `python -c "import aiter; print(aiter.__file__)"` 应能 import

> **注意（NEW-RC-3 patch）**：本 wave 的 commit `f06cdcca5` 是上游 base，**不**含 `aiter/fused_moe.py` L881-886 的 NEW-RC-3 dispatch patch。如果 §6 accuracy 测试时 tp=8 出现 `dispatch miss` / `no instance found` 类错误，参见 §8.1。

---

## §3 Install ATOM

参考 `~/ATOM/README.md` §Installation Option B step 2 + `~/ATOM/CLAUDE.md`。

```bash
cd $HOME/ATOM

# editable install（沿 ATOM/CLAUDE.md 推荐 dev 模式）
pip install -e .

# HuggingFace + ninja（参考 ATOM README §Basic Example）
pip install ninja
pip install -U "huggingface_hub"

# HF 登录（首次环境必须）—— Step-3.5-Flash-FP8 在 HF 需登录账号 + 接受模型条款
hf auth login          # 或 export HF_TOKEN=hf_xxxxx
# 若已登录则 huggingface-cli whoami 应回显你的账户名
```

观察锚点：
- `pip install -e .` 末尾应输出 `Successfully installed atom-...`
- `python -c "from atom import LLMEngine, SamplingParams; print('ok')"` 不报 ImportError
- `python -c "from atom.model_engine.arg_utils import EngineArgs; print('ok')"` 不报 ImportError
- `python -c "import atom.examples.simple_inference; print('ok')"` 不报 ImportError（§5 sanity 依赖）

---

## §4 Prepare HuggingFace Cache

把 `stepfun-ai/Step-3.5-Flash-FP8` 全量 snapshot 下到 `HF_HOME`（约 90 GB）。

```bash
export HF_HOME=/workspace/hf_cache    # 或任何 ≥ 200 GB 的路径
export HF_HUB_ENABLE_HF_TRANSFER=0    # 与本项目 verify 命令一致；解释见下方 caveat
mkdir -p "$HF_HOME"
```

### §4.1 先检测：是否已有可复用 snapshot（避免重下 ~90 GB）

如果本机/容器已挂载 `$HF_HOME`（典型如本项目 `/workspace/hf_cache`），先做一次本地存在性检测，命中则直接跳过下载。

```bash
MODEL_DIR="$HF_HOME/hub/models--stepfun-ai--Step-3.5-Flash-FP8"
SNAP_DIR="$MODEL_DIR/snapshots"

if [ -d "$SNAP_DIR" ] && [ -n "$(ls -A "$SNAP_DIR" 2>/dev/null)" ]; then
  # 命中：至少有一个 snapshot revision 目录
  SNAP_REV="$(ls "$SNAP_DIR" | head -n 1)"
  SHARD_CNT="$(ls "$SNAP_DIR/$SNAP_REV"/model-*-of-*.safetensors 2>/dev/null | wc -l)"
  TOTAL_SIZE="$(du -sh "$MODEL_DIR" | awk '{print $1}')"
  echo "[hf_cache] reuse detected: $MODEL_DIR"
  echo "[hf_cache] snapshot=$SNAP_REV  shards=$SHARD_CNT  size=$TOTAL_SIZE"
  # 判定阈值：至少 44 个 safetensors 分片 + 总大小 ≥ 80G（90G 留 ~10% 余量）
  if [ "$SHARD_CNT" -ge 44 ]; then
    echo "[hf_cache] SKIP snapshot_download — existing cache is complete."
  else
    echo "[hf_cache] WARN shard count $SHARD_CNT < 44, fall through to snapshot_download to repair."
  fi
else
  echo "[hf_cache] no local snapshot, fall through to snapshot_download."
fi
```

复用判定通过（`SHARD_CNT ≥ 44`）时本节即结束，直接进入 §5；后续步骤的 `HF_HOME=/workspace/hf_cache` env 已经能让 vLLM / huggingface_hub 直接命中。

### §4.2 Fallback：snapshot_download 完整下载

仅当 §4.1 未命中（或 shard 数不足）时执行：

```bash
# 触发完整下载（trust_remote_code=True 用于 Step-3.5 的自定义模型类）
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='stepfun-ai/Step-3.5-Flash-FP8',
    cache_dir='$HF_HOME/hub',   # 与 hf cache 标准布局一致
)
"
```

观察锚点（无论走 §4.1 复用还是 §4.2 下载，最终都要满足）：
- 输出最后一行不报 `HTTPError` / `EntryNotFoundError` / `401` / `403`
- `du -sh $HF_HOME/hub/models--stepfun-ai--Step-3.5-Flash-FP8` ≈ 90 GB（本机实测 195 GB 是包含历史 blob 的全量 cache，正常）
- 应看到 44 个 `model-*-of-*.safetensors` 分片（在 `snapshots/<rev>/` 下）

> **`HF_HUB_ENABLE_HF_TRANSFER=0` 反直觉 caveat**：该值是 wave 12 reverify PASS 配置的一部分。`hf_transfer` 在某些 ROCm container 中触发 thread-pool stall，本项目实测**关闭后** snapshot_download 与 runtime mmap 都更稳定。如想为加速下载开启 `=1`，请自行验证 44/44 shards 完整且 runtime 不挂起。

---

## §5 Sanity Check（tp=2，max-tokens=64，~3 分钟内首次出文）

确认整套 stack 能跑通最小 inference。**tp=2 是最小可用 parallel；不要起 tp=1**（项目未验证）。

```bash
cd $HOME    # 任意目录皆可，本步不依赖 project_fp8_tp4_repro
mkdir -p /tmp/sanity

HF_HOME=/workspace/hf_cache HF_HUB_ENABLE_HF_TRANSFER=0 \
TRUST_REMOTE_CODE=1 AITER_LOG_LEVEL=WARNING \
python -m atom.examples.simple_inference \
  --model stepfun-ai/Step-3.5-Flash-FP8 \
  --tensor-parallel-size 2 \
  --kv_cache_dtype fp8 \
  --trust-remote-code \
  > /tmp/sanity/tp2_simple.log 2>&1

echo "exit=$?"
grep -E "Loading safetensors shards|Profiling completed|Generated text" /tmp/sanity/tp2_simple.log
```

通过判定（按出现顺序）：
- `Loading safetensors shards 44/44`
- 至少 4 段 `Generated text:` 输出（simple_inference 自带 4 prompt）
- `exit=0`

> **锚点更新（fresh-verify 13 校正）**：早期版本曾用 `Engine Core fully initialized` 作为 startup 锚点，实测 ATOM log **不输出**该字符串（fresh-verify 13 实测 `grep -c` = 0/0/0）。改用 `Loading safetensors shards 44/44`（model load 完成）+ correctness/sanity 自身打印作为锚点。

未通过：去 §8 Troubleshooting。

---

## §6 Accuracy Test（tp=2 / tp=4 / tp=8 串行）

本步对应 reverify_wave 的 verify 流程（`reverify_wave/progress/teammate-reverify.md` §1）。三档串行（GPU 独占）。

> **fresh-verify reader 注意**：`reverify_wave/outputs/tp{2,4,8}.json` 是 wave 12 的 accuracy 锚点（read-only baseline）；本 §6 的 output 路径写到独立 `repro_guide_wave/outputs/`，**不要覆盖** `reverify_wave/outputs/` 锚点。如果你跑在 fresh dir（如 `/tmp/repro_guide_fresh_*/`），把 outputs / logs 全部落到 fresh dir 内即可。

```bash
# 进入项目根（见 §1 第 4 步设置的 REPRO_ROOT）
cd "$REPRO_ROOT"
mkdir -p repro_guide_wave/outputs repro_guide_wave/logs

# Per-run precondition：每档 GPU run 之间确保前一进程完全退出（防 port/显存残留，见 §8.6）
pkill -9 -f 'correctness_bench.py|throughput_bench.py' 2>/dev/null; sleep 3

# tp=2
HF_HOME=/workspace/hf_cache HF_HUB_ENABLE_HF_TRANSFER=0 \
TRUST_REMOTE_CODE=1 AITER_LOG_LEVEL=WARNING \
python correctness_eval/correctness_bench.py \
  --model stepfun-ai/Step-3.5-Flash-FP8 --tensor-parallel-size 2 \
  --kv_cache_dtype fp8 --port 8018 --trust-remote-code --max-tokens 512 \
  --output-json repro_guide_wave/outputs/acc_tp2.json \
  > repro_guide_wave/logs/acc_tp2.log 2>&1

# tp=4（同模板，--tensor-parallel-size 4 --port 8017，输出 acc_tp4.{json,log}）
HF_HOME=/workspace/hf_cache HF_HUB_ENABLE_HF_TRANSFER=0 \
TRUST_REMOTE_CODE=1 AITER_LOG_LEVEL=WARNING \
python correctness_eval/correctness_bench.py \
  --model stepfun-ai/Step-3.5-Flash-FP8 --tensor-parallel-size 4 \
  --kv_cache_dtype fp8 --port 8017 --trust-remote-code --max-tokens 512 \
  --output-json repro_guide_wave/outputs/acc_tp4.json \
  > repro_guide_wave/logs/acc_tp4.log 2>&1

# tp=8（同模板，--tensor-parallel-size 8 --port 8016，输出 acc_tp8.{json,log}）
HF_HOME=/workspace/hf_cache HF_HUB_ENABLE_HF_TRANSFER=0 \
TRUST_REMOTE_CODE=1 AITER_LOG_LEVEL=WARNING \
python correctness_eval/correctness_bench.py \
  --model stepfun-ai/Step-3.5-Flash-FP8 --tensor-parallel-size 8 \
  --kv_cache_dtype fp8 --port 8016 --trust-remote-code --max-tokens 512 \
  --output-json repro_guide_wave/outputs/acc_tp8.json \
  > repro_guide_wave/logs/acc_tp8.log 2>&1
```

每档完成后等待出现以下锚点（不写"几分钟"，只看日志）：

```bash
for tp in 2 4 8; do
  echo "=== tp=$tp ==="
  grep -E "Loading safetensors shards|\[OK\] dumped JSON" \
    repro_guide_wave/logs/acc_tp${tp}.log
done
```

通过判定（A1–A4，源自 reverify TEAM_CONFIG，fresh-verify 13 校正后）：
- exit 0
- log 含 `Loading safetensors shards 44/44`（model load 完成）+ `[OK] dumped JSON`（correctness_bench.py 收尾自身打印；必然存在）
- log `grep -cE "Traceback|OOM|dispatch.*miss|no.instance|division by zero|NaN|Inf"` = 0
- 4/4 prompt 全部 coherent（见 §6.1）

> **锚点更新（fresh-verify 13 校正）**：早期版本含 `Engine Core fully initialized`，ATOM log 实测**不输出**该字符串，已删除。`[OK] dumped JSON` 是 `correctness_bench.py` 自身在所有 prompt 完成、写出 output JSON 后打印的收尾行，比 vLLM/ATOM engine 内部锚点更可靠。

### §6.1 Expected Output Anchors

参照 `reverify_wave/progress/teammate-reverify.md` §2.2，三档 4/4 prompt 的内容期望（**非 byte-identical 要求，仅检查语义合理性**）：

| Prompt idx | 输入 | 期望特征 |
|---|---|---|
| P0 | `introduce yourself` | 英文 introduce-myself reasoning，开头近似 `Hmm, the user simply asked me to introduce myself...` 或 `Hmm, the user asked me to introduce myself...` |
| P1 | `list all prime numbers within 100` | 英文 prime numbers reasoning，开头含 `We are asked to list all prime numbers within 100...` |
| P2 | `1+2+3=?` | **必须最终给出 6**；推理路径 `1+2=3, then 3+3=6`；finish_reason=eos |
| P3 | `如何在一个月内增肌10公斤` | 中文回答，必须**至少命中 1 个**近义词：`不现实` / `几乎不可能` / `不健康` / `不科学` / `超出生理极限` / `高风险` / `激进` / `健康风险`；无乱码（无 `小弟sets邪倾倒` 类 gibberish） |

跨 tp 的 byte-identical 锚点：tp=2 P2 + tp=4 P2 与 baseline 应 byte-identical（`1+2+3` logit margin 极宽，sampling 不翻转）。tp=8 全档允许非 byte-identical 但 4/4 必须 coherent。tp=8 上 P1 开头变体可能含引号（`We are asked: "list all prime numbers within 100"...`），P2 reasoning 路径可能不走 `1+2=3, then 3+3=6` 而直接 `answer is 6` —— 两种皆视为 PASS。

### §6.1.1 Quantitative Tolerance（参考 reverify_wave/outputs + fix_wave/teammate-8 §3）

| Prompt | num_completion_tokens 范围 | finish_reason |
|---|---|---|
| P0 | 369–490 | eos |
| P1 | 478–512 | eos \| max_tokens（任一即可，sampling 边界 [^p1fin]） |
| P2 | 60 (tp=2/4 byte-id) / 60–450 (tp=8, 路径变体[^p2tok]) | eos |
| P3 | 通常 = 512 | max_tokens |

- `first_diff` vs reverify baseline：5–203 范围属 sampling noise；P2 应保持 byte-id（first_diff = N/A）
- 任何 `first_diff = 0` 全文不一致，或输出含非中英文 token cluster = 视为 fail

[^p1fin]: tp=2 P1 在 reverify wave 12 实测 ntoken=478 / finish=eos，但 fresh-verify wave 13 实测同档 ntoken=512 / finish=max_tokens；这是 sampling determinism 边界附近的常见 flip（temperature=0 仍受 mantissa 路径影响）。两种 finish_reason 皆视为 PASS。
[^p2tok]: tp=8 P2 ntoken 因 §6.1 已许可的 reasoning 路径变体（"answer is 6 directly" 短路径 vs `1+2=3, then 3+3=6` 长路径）跨 60–450 范围（reverify 实测 108 / fresh-verify 实测 404，均 PASS）。任一路径只要最终给出 6、finish=eos 即视为 PASS。

---

## §7 Throughput Test（tp=2/4/8 × input_len 4096/8192/16384，串行 9 run）

使用本 wave 提供的 `correctness_eval/throughput_bench.py`。三档 input_len，每档 output_len=512、num_prompts=32。
**`--ignore-eos` 是脚本默认 True，命令模板无需重复**（写也可，效果相同；如需关闭 EOS 提前停用 `--no-ignore-eos`）。

```bash
cd "$REPRO_ROOT"
mkdir -p repro_guide_wave/outputs repro_guide_wave/logs

# 7-run 串行（GPU 独占；跳过 (tp=2,16384) 与 (tp=4,16384) 两个 OOM near-certain 组合，详见 §8.3）。
# 每次 run 必须等上一次 [OK] dumped JSON 后再启。
# PORT 跟随 TP（避免与 §6 残留进程冲突）：tp=2→8018 / tp=4→8017 / tp=8→8016
for TP in 2 4 8; do
  case $TP in 2) PORT=8018;; 4) PORT=8017;; 8) PORT=8016;; esac
  for ILEN in 4096 8192 16384; do
    # 跳过 (tp=2,16384) / (tp=4,16384) —— 单卡 KV cache 估算超 192 GB HBM
    if [ "$ILEN" = "16384" ] && [ "$TP" != "8" ]; then
      echo "=== SKIP (tp=$TP, input_len=$ILEN): expected OOM, see §8.3 ==="
      continue
    fi
    # 上一 run cleanup（防 port/显存残留，见 §8.6）
    pkill -9 -f 'correctness_bench.py|throughput_bench.py' 2>/dev/null; sleep 3
    echo "=== throughput run: tp=${TP} input_len=${ILEN} port=${PORT} ==="
    HF_HOME=/workspace/hf_cache HF_HUB_ENABLE_HF_TRANSFER=0 \
    TRUST_REMOTE_CODE=1 AITER_LOG_LEVEL=WARNING \
    python correctness_eval/throughput_bench.py \
      --model stepfun-ai/Step-3.5-Flash-FP8 --tensor-parallel-size ${TP} \
      --kv_cache_dtype fp8 --port ${PORT} --trust-remote-code \
      --max-num-batched-tokens 16384 --max-num-seqs 512 \
      --gpu-memory-utilization 0.9 \
      --num-prompts 32 --input-len ${ILEN} --output-len 512 \
      --output-json repro_guide_wave/outputs/tp${TP}_in${ILEN}.json \
      > repro_guide_wave/logs/tp${TP}_in${ILEN}.log 2>&1
    echo "    exit=$? log=repro_guide_wave/logs/tp${TP}_in${ILEN}.log"
    grep -E "\[OK\] dumped JSON|throughput summary|Loading safetensors shards" \
      repro_guide_wave/logs/tp${TP}_in${ILEN}.log | tail -5
  done
done
```

> **若不想用 for-loop**：上面循环展开等价于 7 条命令（按下表"是否运行"列）；按矩阵表逐条手动执行也可（务必串行 + 每条之间手动 `pkill`）。

7+2 次 run 矩阵（**串行 GPU 独占**；OOM 组合默认 SKIP）：

| TP | input_len | port | 是否运行 | 输出 JSON | 输出 log |
|---|---:|---:|:---:|---|---|
| 2 | 4096 | 8018 | ✓ | `repro_guide_wave/outputs/tp2_in4096.json` | `tp2_in4096.log` |
| 2 | 8192 | 8018 | ✓ (tight, may OOM) | `repro_guide_wave/outputs/tp2_in8192.json` | `tp2_in8192.log` |
| 2 | 16384 | 8018 | **SKIP (OOM)** | — | — |
| 4 | 4096 | 8017 | ✓ | `repro_guide_wave/outputs/tp4_in4096.json` | `tp4_in4096.log` |
| 4 | 8192 | 8017 | ✓ | `repro_guide_wave/outputs/tp4_in8192.json` | `tp4_in8192.log` |
| 4 | 16384 | 8017 | **SKIP (OOM)** | — | — |
| 8 | 4096 | 8016 | ✓ | `repro_guide_wave/outputs/tp8_in4096.json` | `tp8_in4096.log` |
| 8 | 8192 | 8016 | ✓ | `repro_guide_wave/outputs/tp8_in8192.json` | `tp8_in8192.log` |
| 8 | 16384 | 8016 | ✓ (tight) | `repro_guide_wave/outputs/tp8_in16384.json` | `tp8_in16384.log` |

通过判定：
- 7 个 run（默认 SKIP 后剩余）全部 exit 0
- 每个 log 末尾出现 `[OK] dumped JSON -> ...` 与 `throughput summary` 表
- 每个 JSON 的 `totals.total_output_tokens` ≈ `num_prompts × output_len = 32 × 512 = 16384`（`--ignore-eos` 默认 True 强制跑满；如显著低于 16384 说明 ignore_eos 未生效，去 §8.4）
- `throughput.total_tokens_per_s` 单调性参考：相同 input_len 下 tp 升 → tps 升；相同 tp 下 input_len 升 → prefill 路径占比升

> **`prefill_tokens_per_s` / `decode_tokens_per_s` 命名 caveat**：脚本 JSON 里这两个字段的物理定义都是 `<token 数> / wall_time_s`（**整段 wall 时间**，不是真正的 prefill / decode 分离阶段时间）。三者满足 `prefill + decode = total` 这条 algebraic identity，但不要解释为"prefill phase 速率"或"decode phase 速率"。如需阶段分离速率，请用 per-request `mean_ttft_s`（≈ prefill 时间）与 `mean_tpot_s`（≈ 单 token decode 时间）：
> - `prefill_actual_tps ≈ input_len / mean_ttft_s`
> - `decode_actual_tps ≈ 1 / mean_tpot_s`
>
> **Wall-time variance caveat**：throughput 数字是 wall-time 测量，受调度器 / cudagraph capture 顺序 / KV block 分配影响，跨 run **预期 ±5–15% 抖动**（ATOM 当前不支持 `--seed`，determinism 仅靠 `temperature=0` 维持 token 序列稳定，wall 不稳定）。**单次 run 数字不应作为外部权威基线**。

汇总命令（人工读）：

```bash
for tp in 2 4 8; do
  for ilen in 4096 8192 16384; do
    f=repro_guide_wave/outputs/tp${tp}_in${ilen}.json
    [ -f "$f" ] && python -c "
import json
d = json.load(open('$f'))
t = d['throughput']
tot = d['totals']
print(f'tp=$tp ilen=$ilen wall={d[\"wall_time_s\"]:.1f}s tot_in={tot[\"total_input_tokens\"]} tot_out={tot[\"total_output_tokens\"]} pre_tps={t[\"prefill_tokens_per_s\"]:.0f} dec_tps={t[\"decode_tokens_per_s\"]:.0f} total_tps={t[\"total_tokens_per_s\"]:.0f}')
"
  done
done
```

---

## §8 Troubleshooting

### §8.0 Known-issue 快查矩阵

| Symptom 关键字 | 跳到 § |
|---|---|
| `dispatch miss` / `no instance found` / 乱码 (gibberish) on tp=8 | §8.1 |
| `Loading checkpoint shards: 0%` 卡住 / `Downloading model.safetensors` | §8.2 |
| `HIP out of memory` / `BadAlloc` / `KV cache cannot fit` (input_len=16384) | §8.3 |
| `total_output_tokens << num_prompts × output_len` | §8.4 |
| `ModuleNotFoundError: huggingface_hub / transformers` | §8.5 |
| `Address already in use` (port 8016/7/8) / GPU 显存仍被占 | §8.6 |
| `narrow size<0` / `_load_w2` traceback on tp=8（仅当 ATOM checkout 老 commit）| §8.7 |
| `snapshot_download` 401 / 403 / 未登录 | §8.8 |
| `[aiter] type hints mismatch, override to --> fmha_v3_varlen_fwd(...)` log 噪音 | §8.9 |

### §8.1 AITER NEW-RC-3 patch（tp=8 dispatch miss）

**症状**：tp=8 accuracy 测试 log 出现 `dispatch miss` / `no instance found` / `RuntimeError: ck::*` 类错误，或生成乱码（如 `小弟sets邪倾倒` / 大段非中文非英文 gibberish）。

**原因**：commit `f06cdcca5` 不含 `aiter/fused_moe.py` L881-886 的 NEW-RC-3 dispatch patch。该 patch 在 reverify_wave 实测中是以 working-tree dirty 形式存在，未上 commit（参见 `reverify_wave/progress/teammate-reverify.md` §4 finding 8）。

**解决**：在 `$HOME/aiter/aiter/fused_moe.py` 中定位 `_fused_moe_kernel_per_token_blockscale` dispatch 段（约 L880 附近），把 `gfx942` + `per_1x128` fp8 blockscale 路径强制走 CK 2-stage。完整 patch 来源：`$REPRO_ROOT/fix_wave/progress/teammate-8.md` § "NEW-RC-3 dispatch patch"（已纳入 promote 包，见 §1 第 4 步）；wave 12 reverify 的 `reverify_wave/progress/teammate-reverify.md` §4 finding 8 复述。修改完**仅保留 working-tree dirty，不要 commit 到 aiter 仓**，然后在 `$HOME/aiter/` 下**重新 `python3 setup.py develop`** 让 patch 编译进 .so 生效。

> 该 patch 不在三仓 pinned commit 里是历史包袱，wave 11/12 已验证 working-tree 形式 PASS；本 wave 不动 aiter 源码（红线 R1），故 reproduction 路径仍要求 reader 手工 apply。如 promote 包中的 `fix_wave/progress/teammate-8.md` 缺失，请联系 lead 索取 24 行 diff 文本。

### §8.2 HF cache miss（首跑卡在 model loading）

**症状**：log 长时间停在 `Loading checkpoint shards: 0%` 或反复 `Downloading model.safetensors`。

**原因**：`HF_HOME` 指向的目录未完成 §4 的 snapshot_download，runtime 走兜底网络下载。

**解决**：
1. `du -sh $HF_HOME/hub/models--stepfun-ai--Step-3.5-Flash-FP8` 检查 ≥ 80 GB
2. `ls $HF_HOME/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/*/` 应有 44 个 safetensors 文件
3. 不全则重跑 §4 `snapshot_download`

### §8.3 OOM at input_len=16384

**症状**：tp=2 的 `tp2_in16384.log` 出现 `HIP out of memory` / `BadAlloc` / `KV cache cannot fit`。

**原因**：tp=2 时单卡承担 4×（相对 tp=8）的 KV cache + activation。input_len=16384 × num_prompts=32 = 524288 token 一次性 prefill，KV blocks 不足。

**解决**（按优先级）：
1. **首选**：把 tp=2 这档跳过，仅记录 tp=4/tp=8 的 16384 数据（tp=2 在 long-context 下不是合理的部署配置）
2. 若必须跑：降低 `--num-prompts 32` → `--num-prompts 16` 或 `8`，单独这一档说明降批
3. 不要降 `--gpu-memory-utilization`（已是 0.9 上限）

### §8.4 ignore_eos 未生效（output_tokens << 32×512）

**症状**：throughput JSON 的 `totals.total_output_tokens` 远小于 16384（如几千）。

**原因**：`--ignore-eos` 未透传到 SamplingParams，模型遇 EOS 早停。

**解决**：检查命令行确实写了 `--ignore-eos`（不是 `--no-ignore-eos`）。脚本默认 `True`，不传也应生效；如改写过脚本 default，恢复为 `True`。

### §8.5 fresh container 缺包（ImportError on `huggingface_hub` / `transformers`）

**症状**：§4 snapshot_download 报 `ModuleNotFoundError`。

**原因**：base ROCm 镜像不含 huggingface_hub / transformers 完整依赖。

**解决**：
```bash
pip install -U "huggingface_hub" "transformers>=4.45" "tokenizers"
```
（ATOM 通过 `pip install -e .` 也会拉，但 §4 在 ATOM 安装前跑则会缺。）

### §8.6 Port already in use / GPU 残留显存

**症状**：log 头部 `RuntimeError: ... bind ... Address already in use` 或起 engine 时 `HIP error: out of memory` 但 `rocm-smi` 显示 GPU 应空闲。

**原因**：上一个 GPU run（vLLM/ATOM engine）未干净退出（OOM 半路被 kill / 用户 Ctrl-C 一次未杀干净），残留进程占着 port + 显存。

**解决**：
```bash
pkill -9 -f 'correctness_bench.py|throughput_bench.py|vllm'
sleep 5
rocm-smi --showpids       # 应显示 0 进程
rocm-smi --showmemuse     # 显存应回到接近 0%
# 然后重跑该档
```

### §8.7 tp=8 _load_w2 narrow() size<0 crash（仅当 ATOM checkout 老 commit）

**症状**：log 中出现 `RuntimeError: ... narrow(): start (X) ...` 或 `_load_w2` / `_load_w13` traceback；进程在 weight load 阶段 crash（未到 `Loading safetensors shards 44/44` 完成）。

**原因**：你 checkout 了 ATOM 老 commit（如 `acff926` 或更早，M1+M2 时代）。Step-3.5-Flash-FP8 `moe_inter=1280` + `per_1x128` → D=10 个 fp32 scale block。tp=8 时 `ceil(10/8)=2`，starts=`[0,2,4,6,8,10,12,14]`，rank 5/6/7 命中 `start ≥ D=10` 触发 narrow size≤0。

**解决**：确认 ATOM 在 commit `969d564`（本 guide pinned；§1 表格已加备注）。该 commit 含**双层 fix**：
1. trailing rank early-return（rank 命中越界 starts 时跳过 load）
2. fp32 scale tensor `.zero_()` 初始化（替换 `torch.ones()` 残留 —— 仅 early-return 不够，残留 `1.0` 会让 dequant 把 fp8 raw bits 当 bf16 用，生成乱码）

详见 `MIGRATION_REPORT.md §M3` / `TP8_ROOT_CAUSE_AND_FIX.md`（promote 包附带）。如要 fork ATOM `_load_w13/_load_w2`，**必须保留 `.zero_()` 语义**，否则 4/4 prompt 会全 gibberish 即使 weight load 不 crash。

### §8.8 snapshot_download 401 / 403 / 未登录

**症状**：§4 `snapshot_download` 报 `HTTPError 401/403` 或 `GatedRepoError`。

**原因**：HF_TOKEN 未设置 / 未在浏览器接受 `stepfun-ai/Step-3.5-Flash-FP8` 模型条款。

**解决**：
```bash
hf auth login                        # 或 export HF_TOKEN=hf_xxxxx
huggingface-cli whoami               # 应回显你的账户名
# 浏览器访问 https://huggingface.co/stepfun-ai/Step-3.5-Flash-FP8 接受条款
# 然后重跑 §4
```

### §8.9 `[aiter] type hints mismatch, override to ...` log 噪音（known-noise，可忽略）

**症状**：accuracy / throughput log 中频繁出现：
```
[aiter] type hints mismatch, override to --> fmha_v3_varlen_fwd(...)
```
（典型每次 prefill 都打印一次，tp=2 acc log 可累积数十行。）

**原因**：aiter 内部 attention dispatch 在 dtype 推断时，发现传入 tensor 的 hint type 与 fmha_v3 内核签名差一档，自动 override 到正确的 fp8 varlen 路径。这是**兼容声明**，不是错误。

**解决**：**无需处理**，安全可忽略。fresh-verify wave 13 实测 10 GPU run（acc 3 + thr 7）log 中均出现此 warn，全部 PASS。如想消除噪音，可在命令前加 `AITER_LOG_LEVEL=ERROR`（注意会同时屏蔽真实错误，**不推荐**）。

---

## §9 Pinned Commits Reference

| 仓库 | Commit | Branch on `origin` | URL |
|---|---|---|---|
| ATOM | `969d564` | `feat/step3p5-flash-support` | https://github.com/ROCm/ATOM |
| AITER | `f06cdcca5` | `feat/step3p5-moe-swiglustep` | https://github.com/ROCm/aiter |
| CK | `defd7ad29` | `feat/swiglustep-moe-no-quant` | https://github.com/ROCm/composable_kernel |

> 三仓 `feat/step3p5-flash-support` 远程分支已由 lead push（参考 `promote_wave` 文档）。本 guide 命令的 commit pin 与 reverify_wave PASS 状态严格对齐（差别仅在 §8.1 标注的 aiter NEW-RC-3 working-tree patch）。

---

## Appendix A — 文件清单

本 guide 引用 / 产生的关键文件：

| 用途 | 路径 |
|---|---|
| 本 guide | `/home/junlin12/project_fp8_tp4_repro/repro_guide_wave/REPRODUCTION_GUIDE.md` |
| Throughput 入口（本 wave 新写） | `/home/junlin12/project_fp8_tp4_repro/correctness_eval/throughput_bench.py` |
| Accuracy 入口（已有） | `/home/junlin12/project_fp8_tp4_repro/correctness_eval/correctness_bench.py` |
| Accuracy 锚点参考 | `/home/junlin12/project_fp8_tp4_repro/reverify_wave/progress/teammate-reverify.md` §2.2 |
| NEW-RC-3 patch 来源 | `/home/junlin12/project_fp8_tp4_repro/fix_wave/progress/teammate-8.md` |
| ATOM serving / generate dict 字段定义 | `~/ATOM/docs/serving_benchmarking_guide.md` §2.4（line ~198） |
| ATOM install 参考 | `~/ATOM/README.md` §Installation + `~/ATOM/CLAUDE.md` |
| AITER install 参考 | `~/aiter/README.md` §Installation |
