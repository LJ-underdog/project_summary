# Step-3.5-Flash FP8 tp=4 推理复现指南

> **目标**：在新机器上完整复现 Step-3.5-Flash-FP8 tp=4 推理，达到 TTFT≈86ms、TPOT≈13ms、4/4 输出连贯。
> **验证基准**：V06 Exp2（2026-04-22）、V07（2026-04-25）全 PASS。
> **适用平台**：8× AMD MI350X (gfx950)，ROCm，Python 3.12。

---

## 前置条件检查

在开始前，确认以下硬件和软件就绪：

```bash
# 1. 确认 ROCm 可用
rocm-smi --showmemuse | head -20

# 2. 确认 GPU 数量（应显示 8 张）
rocm-smi --showid | grep "GPU\["

# 3. 确认 Python 环境
/opt/venv/bin/python --version   # 期望：Python 3.12.x

# 4. 验证 GPU5 是否异常（重要！）
# 注意：必须用子进程隔离，在循环内修改 CUDA_VISIBLE_DEVICES 对已初始化的 torch 无效
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i /opt/venv/bin/python -c "
import torch, time
torch.zeros(128, device='cuda'); torch.cuda.synchronize()  # warmup
s = time.time()
for _ in range(10): torch.zeros(128, device='cuda')
torch.cuda.synchronize()
ms = (time.time()-s)*100
flag = '*** SLOW - hardware fault!' if ms > 100 else 'OK'
print(f'GPU$i: {ms:.1f}ms/op  {flag}')
"
done
```

**期望结果**：GPU0-4、GPU6-7 均 `< 10ms`；GPU5 通常 `~700ms`（硬件异常，全程避开）。

---

## Step 1：获取代码

### 1.1 克隆 ATOM

```bash
git clone git@github.com:ROCm/ATOM.git /path/to/ATOM
cd /path/to/ATOM
git checkout feat/step3p5-flash-support
```

验证当前 HEAD 包含关键 commit：

```bash
git log --oneline | head -5
# 期望第一行包含 acff926d：
# acff926d fix(moe): correct FP8 blockscale inter_dim padding align for all tp configs
```

> **关键**：必须在 `acff926d` 或更新的 commit 上。该 commit 修复了 FP8 tp=4 的 align bug（tp=8 inter=160 会错误 padding 到 192 而非 256）。

### 1.2 克隆 aiter

```bash
git clone git@github.com:ROCm/aiter.git /path/to/aiter
cd /path/to/aiter
git checkout feat/step3p5-moe-swiglustep
```

> **重要**：上游 `ROCm/aiter` 的 `feat/step3p5-moe-swiglustep` 分支可能不包含 BOS workaround。**权威来源是 `junlin12_repos/aiter`**（author: Jun Lin <junlin12@amd.com>）。如果你有此仓库的访问权限，请从那里 clone 或 cherry-pick `a2883ab37`。

验证包含 BOS workaround commit（**必须**）：

```bash
git log --oneline | grep a2883ab37
# 必须看到：
# a2883ab37 fix: remove buggy ASM kernel entry for (N=4096,K=2048) bf16 GEMM on gfx950
```

如果没有此 commit，cherry-pick 它：

```bash
git remote add junlin12 git@github.com:LJ-underdog/aiter.git  # 或联系维护者获取 remote
git fetch junlin12 feat/step3p5-moe-swiglustep
git cherry-pick a2883ab37
```

> **最常见的坑**：如果 `a2883ab37` 缺失，tp=4 长序列（>8209 tokens）prefill 后第一个 decode token 就是 BOS，且全 512 个 token 输出均为 BOS。

同步 CK submodule：

```bash
cd /path/to/aiter
git submodule update --init 3rdparty/composable_kernel
# 验证 CK commit
git -C 3rdparty/composable_kernel log --oneline -1
# 期望：defd7ad29 Add swiglustep_and_mul branches to gridwise_moe_gemm
```

---

## Step 2：安装依赖

### 2.1 安装 aiter

```bash
cd /path/to/aiter
/opt/venv/bin/pip install -r requirements.txt
/opt/venv/bin/pip install -e .
```

> **注意**：安装会触发 CK 编译（`ENABLE_CK=1` 默认开启）。首次编译耗时约 10-30 分钟。

验证安装：

```bash
cd /tmp && /opt/venv/bin/python -c "import aiter; print(aiter.__file__)"
# 必须指向 /path/to/aiter/aiter/__init__.py（非系统路径）
```

### 2.2 安装 ATOM

```bash
cd /path/to/ATOM
/opt/venv/bin/pip install -e .
```

验证安装：

```bash
cd /tmp && /opt/venv/bin/python -c "import atom; print(atom.__file__)"
```

---

## Step 3：关键 Workaround 验证

### 3.1 确认 BOS workaround CSV 行已删除

触发 BOS bug 的是 dispatcher 在 M∈[8193,16384] 时选中 `bf16gemm_bf16_tn_256x256` kernel 处理 (N=4096,K=2048) 的 GEMM。该 kernel 对非 256 对齐的 M 输出错误。`a2883ab37` 删除了 CSV 中触发此路由的 entry。

```bash
# 精确检查：看是否存在触发 bf16gemm_bf16_tn_256x256 用于 N=4096,K=2048 的 entry
grep "bf16gemm_bf16_tn_256x256" /path/to/aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv | grep "4096.*2048\|2048.*4096"
# 期望：无输出（已删除）
# 如果有输出：必须删除该行
```

> **注意**：CSV 中还有其他含 4096/2048 的行（`splitk_clean` variant），那些是正常的，**不要删除**。只删除包含 `bf16gemm_bf16_tn_256x256` 且对应 N=4096,K=2048 的那一行。

**推荐方式**（最安全）：直接使用 `junlin12_repos/aiter` 的 `a2883ab37` commit，workaround 已自动包含，无需手动操作 CSV。

### 3.2 确认 FP8 tp=4 ceil 整除修复

```bash
grep -n "load_shard_size\|tp_size - 1" /path/to/ATOM/atom/model_ops/moe.py | head -10
# 期望（约 _load_w13 L2310-2312、_load_w2 L2352-2354）找到：
# load_shard_size = (loaded_weight.shape[shard_dim] + self.tp_size - 1) // self.tp_size
```

如果看到的是 `// self.tp_size`（无 `+ self.tp_size - 1`），说明是旧代码，会导致 FP8 scale 未全部加载（gibberish 输出）。

### 3.3 确认 align 修复（FP8 blockscale）

```bash
grep -n "align = block_n\|align = 64 if" /path/to/ATOM/atom/model_ops/moe.py | head -5
# 期望（约 L1726）：align = block_n（不含条件分支）
# 如果看到 align = 64 if inter_dim <= 192 else block_n 则是旧 bug（tp=8 会 padding 到 192）
```

---

## Step 4：下载模型权重

### 4.1 FP8 模型

```bash
# 使用 huggingface-cli 下载
/opt/venv/bin/huggingface-cli download \
  stepfun-ai/Step-3.5-Flash-FP8 \
  --local-dir /path/to/model/Step-3.5-Flash-FP8
```

或通过 HF 自动缓存（推理时传 model id 自动下载）：

```bash
# 确认权重已在缓存中（本机已有）
ls ~/.cache/huggingface/hub/ | grep Step-3.5-Flash-FP8
```

### 4.2 确认权重文件完整

```bash
ls ~/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/*/
# 期望：config.json、model*.safetensors、tokenizer*、generation_config.json
python3 -c "
from safetensors import safe_open
import glob
base = '~/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/'
import os; base = os.path.expanduser(base)
snaps = sorted(glob.glob(base + '*/config.json'))
if snaps: print('Config found:', snaps[0])
else: print('ERROR: config not found')
"
```

---

## Step 5：运行推理

### 5.1 清理 ATOM 编译缓存

```bash
rm -rf /root/.cache/atom/*
# 每次修改 ATOM 代码后必须执行
```

### 5.2 FP8 tp=4 推理命令（标准验证）

```bash
cd /tmp && \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
ATOM_LOG_LEVEL=WARNING \
AITER_LOG_LEVEL=WARNING \
/opt/venv/bin/python -m atom.examples.simple_inference \
  --model stepfun-ai/Step-3.5-Flash-FP8 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --level 0 \
  --temperature 0 \
  --max-tokens 128 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 2048
```

参数说明：
- `--level 0`：eager mode（无 CUDAGraph），最稳定，适合初次验证
- `--temperature 0`：确定性输出，便于对比
- `CUDA_VISIBLE_DEVICES=0,1,2,3`：**避开 GPU5**（GPU5 有硬件异常）
- `cd /tmp`：**必须**，否则 aiter 被识别为 namespace package 导致 import 失败

### 5.3 等待启动

首次启动会触发 JIT kernel 编译，耗时约 2-5 分钟。看到以下日志说明正在编译（正常）：

```
[aiter] JIT compiling module_moe_ck2stages_...
```

编译完成后进入推理：

```
Processed prompts: 100% ...
```

---

## Step 6：验证输出

### 6.1 期望输出格式

```
Prompt: "introduce yourself"
Generated: "Hmm, the user simply asked me to introduce myself..."
[TTFT = 86ms, TPOT = 13ms]

Prompt: "list all prime numbers within 100"
Generated: "The user wants a list of prime numbers within 100..."

Prompt: "1+2+3=?"
Generated: "The user is asking 1+2+3=?"

Prompt: "如何在一个月内增肌10公斤"
Generated: "这个问题..."
```

### 6.2 验证标准

| 指标 | 期望值 | 失败症状 |
|------|--------|---------|
| TTFT | < 200ms（典型 86ms） | 如果 > 500ms：可能在 CPU 上跑，检查 CUDA_VISIBLE_DEVICES |
| TPOT | < 20ms（典型 13ms） | 如果 > 50ms：可能 kernel 未编译，等 JIT 完成 |
| 输出连贯性 | 4/4 正常中英文 | 全是 `<s>`：BOS workaround 未生效（见 §3.1） |
| 无 crash | 无 ValueError/shape error | ValueError: 检查 align 修复（§3.3）；shape mismatch：检查 ceil 修复（§3.2） |

### 6.3 常见错误排查

**错误 1：输出全为 `<s>` (BOS token)**

```
Generated: "<s><s><s><s><s>..."
```

原因：`a2883ab37` workaround 未生效，tp=4 o_proj 在长序列下输出错误。
修复：确认 `glm5_bf16_tuned_gemm.csv` 中无 `N=4096,K=2048` 对应的 ASM kernel 行。

**错误 2：`ValueError: ... block_n ... not divisible`**

```
ValueError: The output_size of gate's and up's weight = 320 is not divisible by block_n = 128
```

原因：FP8 ceil 整除修复未生效（commit `ccb64621` 未包含）。
修复：确认 ATOM `moe.py` 中 `load_shard_size` 使用 ceil 整除。

**错误 3：推理结果乱码（gibberish）**

原因：FP8 scale 未全部加载（floor 整除导致最后几个 scale block 用 `torch.ones()` 默认值）。
修复：同错误 2。

**错误 4：`import aiter` 报错 namespace package**

```
AttributeError: module 'aiter' has no attribute 'fused_moe'
```

原因：在 `/path/to/aiter/` 目录下运行了 python，导致当前目录的 `aiter/` 文件夹被识别为 namespace package。
修复：`cd /tmp` 后再运行 python。

**错误 5：单个 GPU 速度极慢（TPOT > 500ms）**

原因：CUDA_VISIBLE_DEVICES 包含了 GPU5（硬件异常，~700ms/tensor）。
修复：检查 `CUDA_VISIBLE_DEVICES` 设置，排除 GPU5。

---

## Step 7：回归验证（可选）

验证不同配置不退化：

### 7.1 BF16 tp=4 回归

```bash
cd /tmp && \
CUDA_VISIBLE_DEVICES=0,1,2,3 AITER_LOG_LEVEL=WARNING \
/opt/venv/bin/python -m atom.examples.simple_inference \
  --model stepfun-ai/Step-3.5-Flash \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --level 0 --temperature 0 --max-tokens 128 \
  --gpu-memory-utilization 0.7 \
  --max-num-batched-tokens 2048 --max-num-seqs 256
# 期望：TTFT≈82ms，TPOT≈17ms，4/4 连贯
```

### 7.2 FP8 tp=2 回归

```bash
cd /tmp && \
CUDA_VISIBLE_DEVICES=0,1 AITER_LOG_LEVEL=WARNING \
/opt/venv/bin/python -m atom.examples.simple_inference \
  --model stepfun-ai/Step-3.5-Flash-FP8 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --level 0 --temperature 0 --max-tokens 128 \
  --max-num-batched-tokens 4096 --max-num-seqs 2048
# 期望：TTFT≈85ms，TPOT≈13.5ms，4/4 连贯
```

---

## 快速检查清单

复现前逐项确认：

- [ ] GPU5 已确认异常，`CUDA_VISIBLE_DEVICES` 不含 5
- [ ] aiter 分支包含 commit `a2883ab37`（BOS workaround）
- [ ] `glm5_bf16_tuned_gemm.csv` 中无 `N=4096,K=2048` ASM kernel 行
- [ ] ATOM 分支包含 commit `acff926d`（align bug fix）
- [ ] ATOM `moe.py` 中 `load_shard_size` 使用 ceil 整除（含 `+ self.tp_size - 1`）
- [ ] ATOM `moe.py` 中 `align = block_n`（不含 `64 if inter_dim <= 192` 条件）
- [ ] CK submodule 已 update（commit `defd7ad29` 含 SwigluStep）
- [ ] aiter 和 ATOM 均已 `pip install -e .`
- [ ] `/root/.cache/atom/*` 已清理
- [ ] 在 `/tmp` 目录下运行 python（不在 aiter 仓库目录内）
- [ ] FP8 模型权重已下载（`stepfun-ai/Step-3.5-Flash-FP8`）

---

## 附：关键代码路径索引

| 修复项 | 文件 | 关键行 |
|--------|------|--------|
| FP8 scale ceil 整除 | `atom/model_ops/moe.py` | `_load_w13` ~L2310-2312、`_load_w2` ~L2352-2354 |
| FP8 align bug fix | `atom/model_ops/moe.py` | ~L1726 `align = block_n` |
| BOS workaround CSV | `aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv` | 触发 `bf16gemm_bf16_tn_256x256` 的 N=4096,K=2048 entry 已删（commit `a2883ab37`） |
| SwigluStep CK kernel | `aiter/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm.hpp` | swiglustep_and_mul 分支 |
| preshuffle_on 强制 | `atom/model_ops/moe.py` | `process_weights_after_loading` → `shuffle_weights()` |
| tp=4 inter_dim padding | `atom/model_ops/moe.py` | `_process_block_quant` → `align = block_n` |

---

## 附：已知限制

- **tp=8**：GPU5 硬件异常导致推理阻塞，暂不可用
- **长序列 tp=4**（>8209 tokens）：BOS workaround 必须生效，否则 prefill 后全 BOS
- **CUDA Graph（--level 3）**：初次复现建议 `--level 0`（eager），稳定后再尝试更高 level
- **FP8 blockscale tp=8 inter=160**：需 padding 到 256（align=128），尚无实测数据（GPU5 阻塞）
