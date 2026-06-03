#!/usr/bin/env python3
"""
perf_correctness_bench.py — Step-3.5-Flash FP8 标准化性能 + 正确性测试
适用平台：gfx950 (MI350X) / gfx942 (MI300X)

【运行方法】
  # ⚠️ 必须显式传 --model $STEP35_PATH（否则 ATOM EngineArgs 默认加载 Qwen/Qwen3-0.6B
  #    dense 模型，silent 覆盖 _find_model_path()，跑出来 perf 数值与 stepfun-Flash MoE
  #    路径完全无关 —— 详见 step35-flash-support/REPRODUCE.md §7.13 KNOWN_FACT）
  # 验证方法：跑完 grep -m2 'Model load done' <full.log>，必须显示 stepfun snapshot 路径

  # 推荐：先 export STEP35_PATH=stepfun-ai/Step-3.5-Flash-FP8
  #       （或本地 snapshot 绝对路径，如 /root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/<sha>）

  # gfx950, FP8 tp=2（GPU 4,6，排除硬件异常的 GPU5）
  cd /tmp && CUDA_VISIBLE_DEVICES=4,6 \\
    HF_HOME=/root/.cache/huggingface AITER_LOG_LEVEL=WARNING \\
    /opt/venv/bin/python /path/to/perf_correctness_bench.py \\
    --model $STEP35_PATH --tp 2     # ← --model 不可省，详 REPRODUCE.md §7.13

  # gfx950, FP8 tp=4（GPU 0,1,2,3）
  cd /tmp && CUDA_VISIBLE_DEVICES=0,1,2,3 \\
    HF_HOME=/root/.cache/huggingface AITER_LOG_LEVEL=WARNING \\
    /opt/venv/bin/python /path/to/perf_correctness_bench.py \\
    --model $STEP35_PATH --tp 4     # ← --model 不可省，详 REPRODUCE.md §7.13

  # gfx942（按实际 GPU 编号）
  # 注意：仅设 STEP35_PATH 环境变量【不够】—— EngineArgs --model default=Qwen
  # 会 silent 覆盖脚本内 _find_model_path() 的推断逻辑（参见本文件 line ~140）
  cd /tmp && CUDA_VISIBLE_DEVICES=0,1 STEP35_PATH=/path/to/model \\
    /opt/venv/bin/python /path/to/perf_correctness_bench.py \\
    --model $STEP35_PATH --tp 2     # ← 仅 export STEP35_PATH 不够，必须显式 --model，详 REPRODUCE.md §7.13

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
    """正确性检查，返回结果字典。

    [HOOK for OPT-6 / wave2 P4 — stub only]
    本函数保持启发式 char/word/BOS 检测；wave2 P0-2=β 决策下，**不**在本脚本实施
    bf16 reference cos-sim（cos-sim 路径在 perf_correctness_bench.py 不存在，详见
    cos_sim_path_survey.md）。GOAL 已降级为 fp8-vs-fp8 byte-equal + 启发式 PASS。
    若未来需接入 cos-sim，应在主流程 `_check_correctness(text_out)` 调用之后追加：
        # cos_sim = _check_cos_sim_vs_bf16_ref(text_out, output_token_ids,
        #                                       args.bf16_ref_path,
        #                                       layer_indices=[N1,N2,N3,N4])
        # corr.update(cos_sim)
    本函数本身保持启发式 char/word/BOS 检测不变；cos-sim 是 *additive*，不替换
    _check_correctness 的现有契约。详 wave2 proposed_fix_B01.md §4。
    """
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
    parser.add_argument("--ignore-eos",    action="store_true", default=False,
                        help="忽略 eos token，让 output 跑满 max_tokens（用于 TPOT 样本量提升）。"
                             "默认 False = 保持原 eos 提前停行为。")
    parser.add_argument("--num-prompts",   type=int, default=1,
                        help="每轮测量 batch 大小（同 prompt 复制 N 份）。"
                             "N>1 用于多 sample TPOT 噪声估算。默认 1 = 保持现状。")
    parser.add_argument("--enable-cudagraph", action="store_true", default=False,
                        help="开启 ATOM CUDAGraph capture（覆盖 --level 默认 0 → 3，"
                             "并尊重 --cudagraph-capture-sizes ATOM CLI flag）。"
                             "默认 False = 保持现 eager-mode 行为。")
    args = parser.parse_args()

    # ─── 设置 ATOM 必要参数 ────────────────────────────────────────
    args.tensor_parallel_size = args.tp
    if not getattr(args, "model", None):
        args.model = _find_model_path()
    args.trust_remote_code = True
    # P2 (wave2 OPT-6): max_num_seqs 必须 ≥ batch 大小（args.num_prompts）
    args.max_num_batched_tokens = 16384
    args.max_num_seqs = max(1, args.num_prompts)

    # P1-3 sanity (wave2 §11.3 candidate A auto-bump): num_prompts × input_tokens 若超过
    # max_num_batched_tokens，ATOM scheduler 会拆 chunk / 拒绝调度 → decode batch 不再
    # 恒定 = N，CUDAGraph capture sizes 全 miss，TPOT/TTFT 测量失真。auto-bump 上限以
    # 容纳全 batch prefill 一次入队，并保留 1024 token 余量。
    _required_batched = args.num_prompts * args.input_tokens
    if _required_batched > args.max_num_batched_tokens:
        _bumped = _required_batched + 1024
        print(f"[bench][P1-3] auto-bump max_num_batched_tokens "
              f"{args.max_num_batched_tokens} → {_bumped} "
              f"(num_prompts={args.num_prompts} × input_tokens={args.input_tokens} "
              f"= {_required_batched})")
        args.max_num_batched_tokens = _bumped
        # 注意：max_model_len 也需相应放大，否则单 prompt 长度仍受 16384 限制
        if getattr(args, "max_model_len", 0) < args.input_tokens + 1024:
            args.max_model_len = args.input_tokens + 1024

    # P3 (wave2 OPT-6): cudagraph 路径默认 eager (level=0)，与 baseline anchor 兼容；
    #     用户显式 --enable-cudagraph 才开启 cudagraph capture，并尊重 ATOM EngineArgs
    #     `--cudagraph-capture-sizes` CLI flag (default `[1,2,4,8,16,32,48,64,128,256]`,
    #     arg_utils.py:43+115)。删除原 hardcoded `args.cudagraph_capture_sizes = str([1])`。
    if args.enable_cudagraph:
        args.level = 3
        # 不显式覆盖 args.cudagraph_capture_sizes — 由 ATOM CLI default 或用户
        # --cudagraph-capture-sizes 决定
    else:
        args.level = 0
        # eager-mode 下 cudagraph_capture_sizes 不生效，但保留 CLI 值（不强制覆盖）

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
    chat_prompts = [chat_prompt] * args.num_prompts          # P2 (wave2 OPT-6): 复制 N 份
    _emit(f"[1/4] Actual input tokens: {n_input_actual}", log_fh)
    _emit(f"[1/4] Num prompts (batch): {args.num_prompts}", log_fh)

    # ─── Engine ─────────────────────────────────────────────────────
    _emit("[2/4] Initializing ATOM engine ...", log_fh)
    t0_engine = time.perf_counter()
    engine_args = EngineArgs.from_cli_args(args)
    llm = engine_args.create_engine(tokenizer=tokenizer)
    engine_init_s = time.perf_counter() - t0_engine
    _emit(f"[2/4] Engine init: {engine_init_s:.2f}s", log_fh)

    sp_warm = SamplingParams(temperature=args.temperature, max_tokens=4)
    sp_meas = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.output_tokens,
        ignore_eos=args.ignore_eos,   # P1 (wave2 OPT-6): 用户开启时 output 强制跑到 max_tokens
    )

    def _run_one(label):
        # P2 (wave2 OPT-6): 用 chat_prompts (length = args.num_prompts)
        if args.measure_method == "A":
            _ = llm.generate(chat_prompts, sp_warm)  # warmup 在首轮前已跑，此处保留供后续轮次
            t0 = time.perf_counter()
            outputs = llm.generate(chat_prompts, sp_meas)
            wall = time.perf_counter() - t0

            # P2: 聚合 N 个 prompt 的 ttft/tpot
            ttfts   = [float(o.get("ttft", 0.0)) for o in outputs]
            tpots_s = [float(o.get("tpot", 0.0)) for o in outputs]
            n_outs  = [int(o.get("num_tokens_output", 0)) for o in outputs]
            n_in    = int(outputs[0].get("num_tokens_input", n_input_actual))
            total_s = float(outputs[0].get("latency", wall))
            text_out = outputs[0].get("text", "") or ""

            ttft   = sum(ttfts) / len(ttfts)            # mean
            tpot_s = sum(tpots_s) / len(tpots_s)
            n_out  = sum(n_outs) // len(n_outs)         # mean rounded
        else:  # method B
            sp1 = SamplingParams(temperature=args.temperature, max_tokens=1)
            _ = llm.generate(chat_prompts, sp_warm)
            t0 = time.perf_counter()
            out1 = llm.generate(chat_prompts, sp1)
            ttft = time.perf_counter() - t0
            t0 = time.perf_counter()
            outputs = llm.generate(chat_prompts, sp_meas)
            total_s = time.perf_counter() - t0

            n_outs = [int(o.get("num_tokens_output", 0)) for o in outputs]
            n_out  = sum(n_outs) // len(n_outs)
            n_in   = int(outputs[0].get("num_tokens_input", n_input_actual))
            tpot_s = (total_s - ttft) / max(1, n_out - 1)
            text_out = outputs[0].get("text", "") or ""
            ttfts   = [ttft] * args.num_prompts          # method B 没有 per-prompt ttft，复制（R3 标记 cv 假象）
            tpots_s = [tpot_s] * args.num_prompts

        # P2: per-prompt cv (std/mean) — 只在 N>=2 时有意义
        if args.num_prompts >= 2:
            import statistics
            ttft_cv = statistics.stdev(ttfts) / max(1e-9, statistics.mean(ttfts))
            tpot_cv = statistics.stdev(tpots_s) / max(1e-9, statistics.mean(tpots_s))
        else:
            ttft_cv = 0.0
            tpot_cv = 0.0

        tpot_ms  = tpot_s * 1000.0
        decode_th = (n_out - 1) / max(1e-9, total_s - ttft) if n_out > 1 else 0.0
        _emit(f"  {label}: N={args.num_prompts}  "
              f"TTFT={ttft*1000:.1f}ms (cv={ttft_cv:.1%})  "
              f"TPOT={tpot_ms:.1f}ms (cv={tpot_cv:.1%})  "
              f"total={total_s:.2f}s  out_tokens={n_out}  "
              f"decode_throughput={decode_th:.1f}tok/s", log_fh)
        return ttft, tpot_s, total_s, n_out, n_in, text_out, ttft_cv, tpot_cv

    # ─── Warmup ──────────────────────────────────────────────────────
    _emit("[3/4] Warmup ...", log_fh)
    _ = llm.generate(chat_prompts, sp_warm)  # P2 (wave2 OPT-6): 用 chat_prompts (N copies)
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

    # 取最后一轮（稳态）— P2 (wave2 OPT-6): 增加 ttft_cv / tpot_cv 字段
    ttft_s, tpot_s, total_s, n_out, n_in, text_out, ttft_cv, tpot_cv = results[-1]

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
    _emit(f"TTFT (stable):     {ttft_s*1000:.1f} ms (per-prompt cv={ttft_cv:.1%}, N={args.num_prompts})", log_fh)
    _emit(f"TPOT (stable):     {tpot_s*1000:.1f} ms/token (per-prompt cv={tpot_cv:.1%}, N={args.num_prompts})", log_fh)
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
