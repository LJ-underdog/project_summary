# V07 LongSeq BOS Verification

## Background

aiter commit `a2883ab37` deletes `glm5_bf16_tuned_gemm.csv` L45 (entry
`gfx950,X,4096,2048,...,asm,...,bf16gemm_bf16_tn_256x256`) which dispatches
the buggy ASM kernel `_ZN5aiter24bf16gemm_bf16_tn_256x256E` for
M >= 8209 producing wrong outputs (cos_sim severely degraded).

## Exp5.a CSV Scan (CPU only)

Scan command (Grep tool):
```
pattern: "bf16gemm_bf16_tn_256x256"
path:    /home/hanchang/aiter/aiter/configs/model_configs/
```

Per-file occurrences of `bf16gemm_bf16_tn_256x256`:

| CSV file | count |
|---|---|
| llama405B_bf16_tuned_gemm.csv | 80 |
| qwen32B_bf16_tuned_gemm.csv | 51 |
| llama70B_bf16_tuned_gemm.csv | 69 |
| glm5_bf16_tuned_gemm.csv | 2 |

Refined scan for the **exact buggy shape** N=4096, K=2048 with this ASM kernel:
```
pattern: "^gfx950,\d+,4096,2048,.*,asm,.*bf16gemm_bf16_tn_256x256"
```
Result: **No matches found** (in any file, including glm5 — confirming
the buggy entry has been deleted).

Broader scan for any row with `,4096,2048,` (any kernel type):
- llama70B_bf16_tuned_gemm.csv: no matches
- llama405B_bf16_tuned_gemm.csv: no matches
- qwen32B_bf16_tuned_gemm.csv: no matches

Conclusion:
- **Only glm5_bf16_tuned_gemm.csv** had the exact buggy (N=4096, K=2048)
  ASM-256x256 entry, and it has been removed (preflight 0.11 confirmed 72
  rows remaining).
- llama70B / llama405B / qwen32B still use the same ASM kernel
  `bf16gemm_bf16_tn_256x256`, but for **other N,K shapes** (none with
  N=4096, K=2048). Whether those other shapes also misbehave at large M
  is **out of scope here** and should be tracked as a separate open bug
  if observed.

## Exp1 tgemm Direct Call (GPU 7)

Script: `/tmp/v07_exp1_tgemm.py`
Log:    `/home/hanchang/project_fp8_tp4/verification_pipeline/results/logs/v07_exp1_tgemm.log`

Setup: `tgemm.mm(a, b)` with a:[M, K=2048] bf16, b:[N=4096, K=2048] bf16,
dispatched via `aiter.tuned_gemm.tgemm` (alias for `gemm_a16w16`).

Dispatcher log shows for every M tested:
```
shape M:{M}, N:4096, K:2048 ... not found tuned config in
/tmp/aiter_configs/bf16_tuned_gemm.csv, will use default config!
using torch solution:0
```
Confirming the buggy ASM kernel is **no longer reachable** at this shape
post-fix (no tuned entry → torch fallback).

Results:

| M    | max_diff | Status |
|------|----------|--------|
| 8192 | 0.00     | PASS   |
| 8208 | 0.00     | PASS   |
| 8209 | 0.00     | PASS   |
| 8216 | 0.00     | PASS   |
| 10021| 0.00     | PASS   |

All M >= 8209 pass with max_diff = 0.00 (<< 50 threshold).

## Overall Conclusion

- Exp5.a: **PASS** — only glm5 was affected; fix already applied.
  Other CSVs use the same ASM kernel but for different N,K — not the
  shape known to be buggy. No new entries to clean up for this specific
  bug.
- Exp1: **PASS** — at the buggy shape (N=4096, K=2048), tgemm now falls
  back to torch for all M; numerical output correct for M up to 10021.
- Workaround in aiter `a2883ab37` is verified effective.
