# SPDX-License-Identifier: MIT
# perf_bench.py — TTFT/TPOT measurement script for fp8-tp4-repro / perf_tp_eval
#
# 红线：不修改 ATOM/aiter/CK 任何源码。本脚本只调用 ATOM 公共 API。
#
# 用法（dry-run 示例）:
#   CUDA_VISIBLE_DEVICES=0,1 HF_HOME=/workspace/hf_cache \
#   AITER_LOG_LEVEL=INFO AITER_LOG_TUNED_CONFIG=1 \
#   /opt/venv/bin/python perf_bench.py --tp 2 --input-tokens 256 --output-tokens 32 \
#       --log-file logs/dry_run_tp2.log
#
# 测量方案:
#   方案 A (首选): ATOM postprocess 直接给出 ttft / tpot
#                 (atom/model_engine/llm_engine.py:260-261)
#   方案 B (fallback): 跑两次 generate，max_tokens=1 测 TTFT，max_tokens=N 测 total。
#
# 当前 ATOM commit acff926 已经 export 方案 A 字段，所以默认走 A。

import argparse
import os
import sys
import time

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from transformers import AutoTokenizer


def build_long_prompt(tokenizer, target_tokens: int, tolerance: int = 32) -> tuple[str, int]:
    """构造一段 token 数 ≈ target_tokens 的 prompt（使用 chat template）。

    策略：用一段中英混合的种子文本反复重复 + 二分调整长度，
    再 apply_chat_template，最后核对 token 数在 [target-tolerance, target+tolerance] 范围。
    """
    seed = (
        "The quick brown fox jumps over the lazy dog. "
        "敏捷的棕色狐狸跳过了懒惰的狗。"
        "In a distant galaxy, the AI engineers benchmarked TTFT and TPOT "
        "to compare tensor-parallel sizes 2, 4, and 8. "
        "在一个遥远的星系，工程师们正在比较张量并行规模 2、4、8 的 TTFT 与 TPOT。"
    )
    # 粗估一份种子约多少 token
    seed_tokens = len(tokenizer.encode(seed, add_special_tokens=False))
    # 估计需要重复多少次（留一点 chat template 开销 ~ 30 token）
    repeats = max(1, (target_tokens - 30) // max(1, seed_tokens))

    def _make_prompt(n_repeats: int) -> tuple[str, int]:
        body = (seed + "\n") * n_repeats + "\nPlease summarize the above content in one sentence."
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False,
            add_generation_prompt=True,
        )
        n = len(tokenizer.encode(chat_prompt, add_special_tokens=False))
        return chat_prompt, n

    chat_prompt, n_actual = _make_prompt(repeats)
    # 简单调整：若不在容差内则 +/- 1 重试若干次
    tries = 0
    while abs(n_actual - target_tokens) > tolerance and tries < 50:
        if n_actual < target_tokens:
            repeats += 1
        else:
            repeats = max(1, repeats - 1)
        chat_prompt, n_actual = _make_prompt(repeats)
        tries += 1

    return chat_prompt, n_actual


def _emit(line: str, fh):
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def _generate_cuda_graph_sizes(max_size: int) -> list[int]:
    sizes, p = [], 1
    while p <= max_size:
        sizes.append(p)
        p *= 2
    return sizes


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="TTFT/TPOT bench for ATOM (fp8-tp4-repro / perf_tp_eval)",
    )
    # ATOM engine 的全部 CLI 参数
    EngineArgs.add_cli_args(parser)

    # 我们自己的参数（注意：不用 --tensor-parallel-size 因为 EngineArgs 已加；用 --tp 转发）
    parser.add_argument("--tp", type=int, required=True,
                        help="tensor parallel size (will override --tensor-parallel-size)")
    parser.add_argument("--input-tokens", type=int, default=10240,
                        help="target input prompt token count")
    parser.add_argument("--output-tokens", type=int, default=1024,
                        help="max_new_tokens to generate")
    parser.add_argument("--log-file", type=str, default=None,
                        help="optional path to dump [PERF] lines")
    parser.add_argument("--measure-method", type=str, default="A", choices=["A", "B"],
                        help="A=use ATOM ttft/tpot fields; B=fallback two-pass timing")
    parser.add_argument("--temperature", type=float, default=0.0)

    args = parser.parse_args()

    # 强制把 --tp 透传给 ATOM 的 tensor_parallel_size
    args.tensor_parallel_size = args.tp
    # 设置必要默认值（参考 simple_inference.py 与 SESSION_HANDOFF.md:186-198）
    if not getattr(args, "model", None):
        args.model = "stepfun-ai/Step-3.5-Flash-FP8"
    args.trust_remote_code = True
    # cudagraph: 单 prompt，max_size=1 即可（避免无效 capture 浪费时间）
    args.cudagraph_capture_sizes = str(_generate_cuda_graph_sizes(1))

    log_fh = None
    if args.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
        log_fh = open(args.log_file, "w")

    _emit(f"[PERF] script_start tp={args.tp} target_input={args.input_tokens} "
          f"target_output={args.output_tokens} measure_method={args.measure_method}", log_fh)

    # ---- step 1: tokenizer + long prompt ----
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_prompt, n_input_actual = build_long_prompt(tokenizer, args.input_tokens)
    _emit(f"[PERF] actual_input_tokens={n_input_actual} (target={args.input_tokens})", log_fh)

    # ---- step 2: build engine ----
    engine_args = EngineArgs.from_cli_args(args)
    _emit("[PERF] creating LLMEngine ...", log_fh)
    t_engine_start = time.perf_counter()
    llm = engine_args.create_engine(tokenizer=tokenizer)
    t_engine_ready = time.perf_counter()
    _emit(f"[PERF] engine_init_secs={t_engine_ready - t_engine_start:.2f}", log_fh)

    try:
        if args.measure_method == "A":
            sp = SamplingParams(temperature=args.temperature, max_tokens=args.output_tokens)
            # warmup 1 次（参考 KNOWN_FACTS：CUDAGraph capture / JIT 在第一个 step 触发）
            _emit("[PERF] warmup generate (max_tokens=4) ...", log_fh)
            warm_sp = SamplingParams(temperature=args.temperature, max_tokens=4)
            _ = llm.generate([chat_prompt], warm_sp)

            # 正式测量
            _emit("[PERF] measure generate (method A) ...", log_fh)
            t0 = time.perf_counter()
            outputs = llm.generate([chat_prompt], sp)
            t1 = time.perf_counter()
            wall = t1 - t0
            assert len(outputs) == 1, f"expected 1 output, got {len(outputs)}"
            out = outputs[0]
            ttft = float(out.get("ttft", 0.0))
            tpot_s = float(out.get("tpot", 0.0))   # ATOM 这里 tpot 单位是秒/token
            n_out = int(out.get("num_tokens_output", 0))
            n_in = int(out.get("num_tokens_input", n_input_actual))
            total_lat = float(out.get("latency", wall))

            tpot_ms = tpot_s * 1000.0
            decode_throughput = (n_out - 1) / max(1e-9, (total_lat - ttft)) if n_out > 1 else 0.0

            _emit(f"[PERF] tp={args.tp} input={n_in} output={n_out}", log_fh)
            _emit(f"[PERF] method=A", log_fh)
            _emit(f"[PERF] TTFT = {ttft:.3f} s", log_fh)
            _emit(f"[PERF] TPOT = {tpot_ms:.3f} ms/token", log_fh)
            _emit(f"[PERF] total_latency = {total_lat:.3f} s", log_fh)
            _emit(f"[PERF] throughput_decode = {decode_throughput:.2f} tokens/s", log_fh)
            _emit(f"[PERF] wall_clock = {wall:.3f} s (sanity)", log_fh)

        else:
            # 方案 B: 两次 generate
            sp_first = SamplingParams(temperature=args.temperature, max_tokens=1)
            sp_full = SamplingParams(temperature=args.temperature, max_tokens=args.output_tokens)

            _emit("[PERF] warmup generate (max_tokens=4) ...", log_fh)
            warm_sp = SamplingParams(temperature=args.temperature, max_tokens=4)
            _ = llm.generate([chat_prompt], warm_sp)

            _emit("[PERF] pass1 generate (max_tokens=1) ...", log_fh)
            t0 = time.perf_counter()
            out1 = llm.generate([chat_prompt], sp_first)
            ttft = time.perf_counter() - t0
            n_in = int(out1[0].get("num_tokens_input", n_input_actual))

            _emit("[PERF] pass2 generate (max_tokens=N) ...", log_fh)
            t0 = time.perf_counter()
            out2 = llm.generate([chat_prompt], sp_full)
            total_lat = time.perf_counter() - t0
            n_out = int(out2[0].get("num_tokens_output", 0))

            tpot_s = (total_lat - ttft) / max(1, n_out - 1) if n_out > 1 else 0.0
            tpot_ms = tpot_s * 1000.0
            decode_throughput = (n_out - 1) / max(1e-9, (total_lat - ttft)) if n_out > 1 else 0.0

            _emit(f"[PERF] tp={args.tp} input={n_in} output={n_out}", log_fh)
            _emit(f"[PERF] method=B", log_fh)
            _emit(f"[PERF] TTFT = {ttft:.3f} s", log_fh)
            _emit(f"[PERF] TPOT = {tpot_ms:.3f} ms/token", log_fh)
            _emit(f"[PERF] total_latency = {total_lat:.3f} s", log_fh)
            _emit(f"[PERF] throughput_decode = {decode_throughput:.2f} tokens/s", log_fh)
    finally:
        try:
            llm.close()
        except Exception as e:
            _emit(f"[PERF] llm.close() failed: {e!r}", log_fh)
        if log_fh is not None:
            log_fh.close()


if __name__ == "__main__":
    sys.exit(main())
