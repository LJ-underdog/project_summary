#!/bin/bash
# EP (expert-parallel) perf bench — prefill TTFT + decode TPOT, cudagraph 态, TP8, gfx942.
# 来源: W8_resume/artifacts_T24/run_prefill_t28.sh (teammate-28 长输入 perf 实验) 清理路径后版本。
#
# 用途: 在 EP + cudagraph 下同时测 prefill TTFT 与 decode TPOT (单序列 batch=1, 长输入)。
#   走仓内 details/scripts/perf_correctness_bench.py (decode-only 的 perf_tpot_t27 临时模块已删)。
#   cudagraph = --enable-cudagraph + --cudagraph-capture-sizes "[1]"; IPC-allreduce 默认全关。
#   ⚠️ EP 下 inter=1280 ≥ 256 → nopad/pad 同一路径, perf 与 pad/nopad 无关 (见 perf/22 §1c)。
#
# 🔴 路径为开发机示例, 按你的环境改 (MODEL / BENCH)。

set -euo pipefail

# ---- 可改路径 (开发机示例) ----
MODEL=${MODEL:-/workspace/hf_cache/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/6eebda59dd87ca5729648ec7cfed0becfceb273e/}
BENCH=${BENCH:-/home/junlin12/project_summary/step35-flash-support/details/scripts/perf_correctness_bench.py}

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

echo "[ep_perf_bench] EP=on cudagraph=ON input=10240 output=256 TP8 measure=A runs=2"
echo "[ep_perf_bench] MODEL=$MODEL"

python "$BENCH" \
  --tp 8 \
  --model "$MODEL" \
  --trust-remote-code \
  --enable-expert-parallel \
  --kv_cache_dtype fp8 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 11520 \
  --input-tokens 10240 \
  --output-tokens 256 \
  --measure-method A \
  --runs 2 \
  --enable-cudagraph \
  --cudagraph-capture-sizes "[1]" \
  --ignore-eos \
  --num-prompts 1

# 预期 (EP, cudagraph, input 实际 10213): prefill TTFT ~560–571ms; decode TPOT ~13.5ms/tok;
#   decode throughput ~74 tok/s (见 perf/22 §1b)。
# 🔴 R5: TP8 cudagraph 有显存泄漏史 — 判泄漏看内存趋势 (rocm-smi --showmeminfo vram) 非 ps -p;
#         teardown ~3min settle 回基线属正常; 详 details/TP8_THREE_BUGS.md B3。
