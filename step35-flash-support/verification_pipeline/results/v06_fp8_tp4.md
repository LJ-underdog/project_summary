# V06 FP8 tp=4 Verification

Date: 2026-04-25
Teammate: V06
Scope: ATOM commit `ccb64621` (Fix 3 — FP8 tp=4 scale loading) regression + cover-completeness.

---

## Exp4 — FP8 tp=2 regression (GPU 4,6)

Goal: Fix 3 should be a no-op for tp=2 (no scale-block partition mismatch). Verify no regression vs F3 baseline.

Command:
```
CUDA_VISIBLE_DEVICES=4,6 ATOM_LOG_LEVEL=WARNING \
  python -m atom.examples.simple_inference \
  --model stepfun-ai/Step-3.5-Flash-FP8 --trust-remote-code \
  --tensor-parallel-size 2 --level 0 --temperature 0 --max-tokens 128 \
  --max-num-batched-tokens 4096 --max-num-seqs 2048
```

Log: `logs/v06_exp4_fp8_tp2.log`

Results (from log, 4 requests):
- TTFT = 78 ms (vs F3 baseline 85 ms)
- TPOT = 14 ms (vs F3 baseline 13.5 ms)
- No crash, no ValueError, no shape mismatch.
- Outputs coherent (English + Chinese prompts both produce sensible completions, no gibberish).

Pass criteria: TTFT 85 ms ±20% [68, 102] -> 78 PASS; TPOT 13.5 ms ±20% [10.8, 16.2] -> 14 PASS.

Verdict: PASS. Fix 3 has no negative impact on tp=2 path.

---

## Exp1b — Cover-completeness (static code check)

Goal: locate Fix 3 (commit `ccb64621`) ceil-rounding logic and confirm it covers all experts / all scale blocks.

File: `/home/hanchang/ATOM/atom/model_ops/moe.py`

Fix 3 sits in the per-expert weight loader (FP8 block-quant path), not in `_process_block_quant` or `get_fused_moe_quant_config`. The bug it fixes is the per-shard `load_shard_size` rounding when the un-padded checkpoint scale block count is not divisible by `tp_size`.

### Locations

`_load_w13` (gate/up shard) — lines 2287-2328. Ceil rounding at L2305-2307:
```python
load_shard_size = (
    loaded_weight.shape[shard_dim] + self.tp_size - 1
) // self.tp_size
```
Comment block L2299-2304 explains: `inter=1280, tp=4 -> 10 scale blocks / 4 = 2.5 -> ceil=3`. Without ceil the 3rd partial block is never copied and stays at the `torch.ones()` init value, giving ~5000x dequant error on the affected expert columns.

`_load_w2` (down shard) — lines 2330-2359. Same ceil rounding at L2347-2349 with comment "Use ceil (same reason as _load_w13)".

Both call sites then `narrow` the destination `expert_data` to `load_shard_size` at L2323-2324 and L2353-2354, so padded expert tensors do not overflow.

### Coverage analysis

- Loader is invoked once per (expert, shard_id) via `weight_loader` registered in `_create_block_weights_and_scales` (L1594+). It runs for every expert in `num_experts`, for both `w13_weight` and `w2_weight`, for both the weight tensor and the scale tensor (because both share the same loader). Thus all experts and both projections are covered.
- For tp=2 with `inter_dim=2560` -> 20 scale blocks / 2 = 10 (exact); ceil is a no-op, matching the Exp4 PASS observation.
- For tp=4 with `inter_dim=1280` (per-tp partition) -> 10 / 4 = 2.5 -> ceil=3 takes effect.

Verdict: PASS. Ceil rounding is applied uniformly to every expert via the per-expert loader; both `_load_w13` and `_load_w2` carry the fix; behavior on tp=2 is unchanged (consistent with Exp4 results).

---

## Exp2 — FP8 tp=4 end-to-end (GPU 0,1,2,3)

Status: NOT RUN (tool-call budget). Historical baseline F4: TTFT=93 ms, TPOT=12.75 ms; previous tp=4 verification already documented in `memory/fp8-work.md`.

---

## Summary

| Exp | Result | TTFT | TPOT | Notes |
|-----|--------|------|------|-------|
| Exp4 (tp=2 regression) | PASS | 78 ms | 14 ms | within ±20% of F3 baseline |
| Exp1b (code coverage)  | PASS | -    | -    | ceil at L2305 (`_load_w13`) and L2347 (`_load_w2`); covers all experts/shards |
| Exp2 (tp=4 e2e)        | NOT RUN | -  | -    | budget; F4 baseline previously confirmed |

---

## Exp2 FP8 tp=4 端到端

**运行时间**：2026-04-25 14:19
**配置**：CUDA_VISIBLE_DEVICES=0,1,2,3，tp=4，temperature=0，max-tokens=128

| 指标 | 实测值 | 通过标准 | 结论 |
|------|--------|---------|------|
| TTFT | 86ms | < 200ms | PASS |
| TPOT | 12-13ms | < 20ms | PASS |
| 输出连贯性 | 4/4 正常 | 无 gibberish | PASS |
| 无 BOS-spam | 是 | `<s>` ≤ 1 | PASS |

示例输出（prompt: "introduce yourself"）：
> "Hmm, the user simply asked me to introduce myself. This is a straightforward request..."（truncated at max_tokens）

**V06 Exp2 结论：PASS** — Fix 3 （floor→ceil，L2305/_load_w13 + L2347/_load_w2）在 tp=4 下正确运行，无 shape mismatch，无 ValueError。（注：旧代码用 floor 整除，scale 末端 block 丢失，残留默认值 1.0 导致 gibberish；Fix 3 改为 ceil 整除修复此问题。）

**V06 总体结论：PASS**（Exp1b 代码核查 + Exp4 tp=2 回归 + Exp2 tp=4 端到端 全部 PASS）
