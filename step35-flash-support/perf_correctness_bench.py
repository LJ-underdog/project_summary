#!/usr/bin/env python3
"""
perf_correctness_bench.py — Step-3.5-Flash FP8 标准化性能 + 正确性测试
适用平台：gfx950 (MI350X) / gfx942 (MI300X)

【运行方法】
  # gfx950, FP8 tp=2（GPU 4,6，排除硬件异常的 GPU5）
  cd /tmp && CUDA_VISIBLE_DEVICES=4,6 \\
    HF_HOME=/root/.cache/huggingface AITER_LOG_LEVEL=WARNING \\
    /opt/venv/bin/python /path/to/perf_correctness_bench.py --tp 2

  # gfx950, FP8 tp=4（GPU 0,1,2,3）
  cd /tmp && CUDA_VISIBLE_DEVICES=0,1,2,3 \\
    HF_HOME=/root/.cache/huggingface AITER_LOG_LEVEL=WARNING \\
    /opt/venv/bin/python /path/to/perf_correctness_bench.py --tp 4

  # gfx942（按实际 GPU 编号）
  cd /tmp && CUDA_VISIBLE_DEVICES=0,1 STEP35_MODEL_PATH=/path/to/model \\
    /opt/venv/bin/python /path/to/perf_correctness_bench.py --tp 2

【固定测试参数（可 CLI 覆盖）】
  --input-tokens  10240   目标 prompt token 数（±32 容差）
  --output-tokens 1024    最大输出 token 数
  --runs          2       测量轮数（取最后一轮稳态）
  --temperature   0.0     greedy decoding

【注意事项】
  - 必须 cd /tmp 再运行（避免 aiter namespace package 被错误识别）
  - gfx950 必须从 CUDA_VISIBLE_DEVICES 中排除 GPU5（硬件异常，~700ms/tensor）
  - gfx942 无此限制
"""

import argparse
import os
import subprocess
import sys
import time

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from transformers import AutoTokenizer

# ─── 模型路径候选（按优先级）────────────────────────────────────────
_MODEL_CANDIDATES = [
    "/root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots",
    "/workspace/hf_cache/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots",
    "/data/hf_cache/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots",
]
_DEFAULT_MODEL_NAME = "stepfun-ai/Step-3.5-Flash-FP8"

# ─── 正确性阈值 ──────────────────────────────────────────────────────
_CORR_MIN_CHARS    = 50     # 输出文本最少字符数
_CORR_MIN_WORDS    = 10     # 输出文本最少词数（空格分割）
_CORR_BOS_PATTERNS = [      # BOS spam 的典型文本特征
    "\x00", "\x01",         # null / SOH
    "<|begin_of_text|>",    # llama-style BOS
    "<s>" * 5,              # 连续 BOS token
]


def _find_model_path():
    env_path = os.environ.get("STEP35_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    for base in _MODEL_CANDIDATES:
        from pathlib import Path
        p = Path(base)
        if p.exists():
            snapshots = sorted(p.iterdir())
            if snapshots:
                return str(snapshots[-1])
    return _DEFAULT_MODEL_NAME


def _detect_gpu_arch():
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showproductname"], stderr=subprocess.DEVNULL, text=True
        )
        if "MI350" in out:
            return "gfx950"
        if "MI308" in out or "MI300" in out:
            return "gfx942"
    except Exception:
        pass
    try:
        from pathlib import Path
        for uevent in Path("/sys/class/drm").glob("card*/device/uevent"):
            t = uevent.read_text()
            if "75a0" in t:
                return "gfx950"
            if "74a1" in t or "74a0" in t:
                return "gfx942"
    except Exception:
        pass
    return "unknown"


def _get_git_hash(paths):
    for p in paths:
        try:
            h = subprocess.check_output(
                ["git", "-C", p, "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL, text=True
            ).strip()
            if h:
                return h
        except Exception:
            pass
    return "unknown"


def build_long_prompt(tokenizer, target_tokens: int, tolerance: int = 32):
    """构造约 target_tokens 长度的 chat prompt（与 perf_bench.py 相同策略）。"""
    seed = (
        "The quick brown fox jumps over the lazy dog. "
        "敏捷的棕色狐狸跳过了懒惰的狗。"
        "In a distant galaxy, AI engineers benchmarked TTFT and TPOT "
        "to compare tensor-parallel sizes 2, 4, and 8. "
        "在遥远的星系，工程师们正在比较张量并行规模 2、4、8 的 TTFT 与 TPOT。"
    )
    seed_tokens = len(tokenizer.encode(seed, add_special_tokens=False))
    repeats = max(1, (target_tokens - 30) // max(1, seed_tokens))

    def _make(n):
        body = (seed + "\n") * n + "\nPlease summarize the above content in one sentence."
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False,
            add_generation_prompt=True,
        )
        n_tok = len(tokenizer.encode(chat, add_special_tokens=False))
        return chat, n_tok

    chat_prompt, n_actual = _make(repeats)
    tries = 0
    while abs(n_actual - target_tokens) > tolerance and tries < 60:
        repeats = repeats + 1 if n_actual < target_tokens else max(1, repeats - 1)
        chat_prompt, n_actual = _make(repeats)
        tries += 1
    return chat_prompt, n_actual


def _check_correctness(output_text: str, output_token_ids=None) -> dict:
    """正确性检查，返回结果字典。"""
    text = output_text or ""
    word_count = len(text.split())
    char_count = len(text)

    # BOS spam 检测
    bos_spam = any(p in text for p in _CORR_BOS_PATTERNS)

    # 如果有 token_ids，做更精确的检查
    bos_token_ratio = None
    first_token_id = None
    unique_token_count = None
    if output_token_ids:
        n = len(output_token_ids)
        bos_count = sum(1 for t in output_token_ids if t in {0, 1})
        bos_token_ratio = bos_count / n if n > 0 else 1.0
        bos_spam = bos_spam or (bos_token_ratio > 0.05)
        first_token_id = output_token_ids[0] if output_token_ids else -1
        unique_token_count = len(set(output_token_ids))

    checks = {
        "char_count": char_count,
        "word_count": word_count,
        "bos_spam": bos_spam,
        "bos_token_ratio": bos_token_ratio,
        "first_token_id": first_token_id,
        "unique_token_count": unique_token_count,
        "len_ok": char_count >= _CORR_MIN_CHARS and word_count >= _CORR_MIN_WORDS,
        "no_bos_spam": not bos_spam,
    }
    checks["all_pass"] = checks["len_ok"] and checks["no_bos_spam"]
    return checks


def _emit(line, log_fh=None):
    print(line, flush=True)
    if log_fh:
        log_fh.write(line + "\n")
        log_fh.flush()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Step-3.5-Flash FP8 性能+正确性标准化基准",
    )
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--tp", type=int, required=True, choices=[1, 2, 4, 8])
    parser.add_argument("--input-tokens",  type=int, default=10240)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--runs",          type=int, default=2,
                        help="测量轮数，取最后一轮稳态")
    parser.add_argument("--log-file",      type=str, default=None)
    parser.add_argument("--measure-method", type=str, default="A", choices=["A", "B"])
    parser.add_argument("--temperature",   type=float, default=0.0)
    args = parser.parse_args()

    # ─── 设置 ATOM 必要参数 ────────────────────────────────────────
    args.tensor_parallel_size = args.tp
    if not getattr(args, "model", None):
        args.model = _find_model_path()
    args.trust_remote_code = True
    args.cudagraph_capture_sizes = str([1])   # 字符串形式，与 EngineArgs 契约一致
    args.max_num_batched_tokens = 16384
    args.max_num_seqs = 1
    args.level = 0

    # ─── 日志文件 ────────────────────────────────────────────────────
    log_fh = None
    if args.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
        log_fh = open(args.log_file, "w")

    # ─── 环境信息 ────────────────────────────────────────────────────
    gpu_arch      = _detect_gpu_arch()
    cuda_visible  = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    atom_hash     = _get_git_hash(["/home/hanchang/junlin12_repos/atom",
                                   "/home/hanchang/ATOM", "/workspace/atom"])
    aiter_hash    = _get_git_hash(["/home/hanchang/junlin12_repos/aiter",
                                   "/home/hanchang/aiter", "/workspace/aiter"])

    _emit("=" * 64, log_fh)
    _emit("=== PERF CORRECTNESS BENCH — Step-3.5-Flash FP8 ===", log_fh)
    _emit("=" * 64, log_fh)
    _emit(f"GPU arch:          {gpu_arch}", log_fh)
    _emit(f"CUDA_VISIBLE_DEVS: {cuda_visible}", log_fh)
    _emit(f"TP:                {args.tp}", log_fh)
    _emit(f"Model:             {args.model}", log_fh)
    _emit(f"ATOM commit:       {atom_hash}", log_fh)
    _emit(f"aiter commit:      {aiter_hash}", log_fh)
    _emit(f"Target input tok:  {args.input_tokens} (±32)", log_fh)
    _emit(f"Max output tok:    {args.output_tokens}", log_fh)
    _emit(f"Runs:              {args.runs} (take last as stable)", log_fh)
    _emit("-" * 64, log_fh)

    # ─── Tokenizer + Prompt ──────────────────────────────────────────
    _emit("[1/4] Loading tokenizer ...", log_fh)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    _emit("[1/4] Building prompt ...", log_fh)
    chat_prompt, n_input_actual = build_long_prompt(tokenizer, args.input_tokens)
    _emit(f"[1/4] Actual input tokens: {n_input_actual}", log_fh)

    # ─── Engine ─────────────────────────────────────────────────────
    _emit("[2/4] Initializing ATOM engine ...", log_fh)
    t0_engine = time.perf_counter()
    engine_args = EngineArgs.from_cli_args(args)
    llm = engine_args.create_engine(tokenizer=tokenizer)
    engine_init_s = time.perf_counter() - t0_engine
    _emit(f"[2/4] Engine init: {engine_init_s:.2f}s", log_fh)

    sp_warm = SamplingParams(temperature=args.temperature, max_tokens=4)
    sp_meas = SamplingParams(temperature=args.temperature, max_tokens=args.output_tokens)

    def _run_one(label):
        if args.measure_method == "A":
            _ = llm.generate([chat_prompt], sp_warm)  # warmup 在首轮前已跑，此处保留供后续轮次
            t0 = time.perf_counter()
            outputs = llm.generate([chat_prompt], sp_meas)
            wall = time.perf_counter() - t0
            out = outputs[0]
            ttft     = float(out.get("ttft",   0.0))
            tpot_s   = float(out.get("tpot",   0.0))
            n_out    = int(out.get("num_tokens_output", 0))
            n_in     = int(out.get("num_tokens_input", n_input_actual))
            total_s  = float(out.get("latency", wall))
            text_out = out.get("text", "") or ""
        else:  # method B
            sp1 = SamplingParams(temperature=args.temperature, max_tokens=1)
            _ = llm.generate([chat_prompt], sp_warm)
            t0 = time.perf_counter()
            out1 = llm.generate([chat_prompt], sp1)
            ttft = time.perf_counter() - t0
            t0 = time.perf_counter()
            outputs = llm.generate([chat_prompt], sp_meas)
            total_s = time.perf_counter() - t0
            out = outputs[0]
            n_out   = int(out.get("num_tokens_output", 0))
            n_in    = int(out.get("num_tokens_input", n_input_actual))
            tpot_s  = (total_s - ttft) / max(1, n_out - 1)
            text_out = out.get("text", "") or ""

        tpot_ms  = tpot_s * 1000.0
        decode_th = (n_out - 1) / max(1e-9, total_s - ttft) if n_out > 1 else 0.0
        _emit(f"  {label}: TTFT={ttft*1000:.1f}ms  TPOT={tpot_ms:.1f}ms  "
              f"total={total_s:.2f}s  out_tokens={n_out}  "
              f"decode_throughput={decode_th:.1f}tok/s", log_fh)
        return ttft, tpot_s, total_s, n_out, n_in, text_out

    # ─── Warmup ──────────────────────────────────────────────────────
    _emit("[3/4] Warmup ...", log_fh)
    _ = llm.generate([chat_prompt], sp_warm)
    _emit("[3/4] Warmup done.", log_fh)

    # ─── Measurement ─────────────────────────────────────────────────
    _emit(f"[4/4] Measurement ({args.runs} run(s)) ...", log_fh)
    results = []
    try:
        for i in range(1, args.runs + 1):
            r = _run_one(f"Run{i}")
            results.append(r)
            if i < args.runs:
                time.sleep(3)
    finally:
        try:
            llm.close()
        except Exception as e:
            _emit(f"llm.close() error: {e!r}", log_fh)

    # 取最后一轮（稳态）
    ttft_s, tpot_s, total_s, n_out, n_in, text_out = results[-1]

    # ─── 正确性检查 ───────────────────────────────────────────────────
    corr = _check_correctness(text_out)
    first_chars = repr(text_out[:80]) if text_out else "(empty)"

    # ─── 最终汇总 ─────────────────────────────────────────────────────
    _emit("", log_fh)
    _emit("=" * 64, log_fh)
    _emit("=== RESULTS (stable = last run) ===", log_fh)
    _emit("=" * 64, log_fh)
    _emit(f"GPU arch:          {gpu_arch}", log_fh)
    _emit(f"CUDA_VISIBLE_DEVS: {cuda_visible}", log_fh)
    _emit(f"TP:                {args.tp}", log_fh)
    _emit(f"ATOM commit:       {atom_hash}", log_fh)
    _emit(f"aiter commit:      {aiter_hash}", log_fh)
    _emit(f"Input tokens:      {n_in}", log_fh)
    _emit(f"Output tokens:     {n_out}", log_fh)
    _emit(f"TTFT (stable):     {ttft_s*1000:.1f} ms", log_fh)
    _emit(f"TPOT (stable):     {tpot_s*1000:.1f} ms/token", log_fh)
    _emit(f"Total lat (stable):{total_s:.3f} s", log_fh)
    _emit(f"Decode throughput: {(n_out-1)/max(1e-9,total_s-ttft_s):.1f} tok/s", log_fh)
    _emit(f"Engine init:       {engine_init_s:.2f} s", log_fh)
    _emit("-" * 64, log_fh)
    _emit(f"CORRECTNESS:       {'PASS' if corr['all_pass'] else 'FAIL'}", log_fh)
    _emit(f"  output chars:    {corr['char_count']}", log_fh)
    _emit(f"  output words:    {corr['word_count']}", log_fh)
    _emit(f"  bos_spam:        {corr['bos_spam']}", log_fh)
    _emit(f"  first 80 chars:  {first_chars}", log_fh)
    if not corr["all_pass"]:
        fails = []
        if not corr["len_ok"]:
            fails.append(f"output too short (chars={corr['char_count']}, words={corr['word_count']})")
        if not corr["no_bos_spam"]:
            fails.append("BOS spam detected")
        _emit(f"  FAIL reasons:    {'; '.join(fails)}", log_fh)
    _emit("=" * 64, log_fh)

    if log_fh:
        log_fh.close()
        print(f"\nLog saved: {args.log_file}", flush=True)

    return 0 if corr["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
