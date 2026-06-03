#!/bin/bash
# EP (expert-parallel) e2e correctness with cudagraph — stepfun-Flash-FP8, TP8, gfx942.
# 来源: W8_resume/artifacts_T24/run_atom_cudagraph.sh (teammate-26 cudagraph 实验) 清理路径后版本。
#
# 用途: 验证 ATOM 原生栈在 EP + cudagraph 下 e2e 跑通且 4/4 prompt 连贯。
#   cudagraph = 去掉 --enforce-eager (ATOM 原生默认全关 IPC-allreduce → cudagraph 可用)。
#   ⚠️ EP 精度 = "看着连贯", 未对 ground-truth 做严格数值验证 (见 perf/22 §3)。
#
# 🔴 路径为开发机示例, 按你的环境改 (MODEL / ATOM_DIR)。

set -euo pipefail

# ---- 可改路径 (开发机示例) ----
MODEL=${MODEL:-/workspace/hf_cache/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/6eebda59dd87ca5729648ec7cfed0becfceb273e/}
ATOM_DIR=${ATOM_DIR:-/home/junlin12/ATOM}

# ---- env (缓存 + 可见 GPU) ----
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache
export TORCHINDUCTOR_CACHE_DIR=/workspace/cache/inductor
export TRITON_CACHE_DIR=/workspace/cache/triton
export TORCH_EXTENSIONS_DIR=/workspace/cache/torch
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# EP 下 inter=1280 (≥256), 此 env 是 no-op — 设不设都同一路径 (B2 nopad 不触发)。保留仅为显式。
export ATOM_FP8_MOE_DISABLE_PAD=1

# IPC-allreduce 全关 (ATOM 原生默认态): custom-allreduce OFF + quick-reduce NONE/disabled → cudagraph 可用。
unset AITER_QUICK_REDUCE_QUANTIZATION

echo "[ep_e2e_cudagraph] EP=on cudagraph=ON(no enforce-eager) custom_allreduce=default-off quick_reduce=NONE TP8"
echo "[ep_e2e_cudagraph] MODEL=$MODEL"

cd "$ATOM_DIR"
python -m atom.examples.simple_inference \
  --model "$MODEL" \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --enable-expert-parallel \
  --kv_cache_dtype fp8 \
  --gpu-memory-utilization 0.4 \
  --max-model-len 1024 \
  --max-num-seqs 8 \
  --temperature 0.0 \
  --max-tokens 64

# 预期: Engine Core fully initialized + 44/44 shards + 4 段 Generated text (4/4 连贯, 非 Qwen, 自然 eos)。
# 🔴 R5: TP8 cudagraph 有显存泄漏史 — 判泄漏看内存趋势 (rocm-smi --showmeminfo vram) 非 ps -p;
#         teardown ~3min settle 回基线属正常; 详 details/TP8_THREE_BUGS.md B3。
