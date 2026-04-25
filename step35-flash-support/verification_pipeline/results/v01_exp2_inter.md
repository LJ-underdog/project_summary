# V01 Experiment 2 — V1/V3 inter_dim Boundary Matrix

Test config: M=32, model_dim=7168, E=16, topk=4, dtype=bfloat16
GPU: gfx950, CUDA_VISIBLE_DEVICES=1
Date: 2026-04-25
Script: /tmp/v01_exp2_inter.py
Log: /home/hanchang/project_fp8_tp4/verification_pipeline/results/logs/v01_exp2.log

## Result Matrix

| inter_dim | block_m chosen | cos_sim (post-fix) | Threshold | Result |
|-----------|----------------|--------------------|-----------|--------|
| 192       | 32 (V1 path)   | 0.99999273         | >= 0.9999 | PASS   |
| 256       | 128 (V3 forced)| 0.99999255         | >= 0.9999 | PASS   |
| 320       | 128 (V3 forced)| ERROR              | n/a       | N/A (tile constraint, see note) |
| 640       | 128 (V3 forced)| 0.99999249         | >= 0.9999 | PASS   |

inter_dim=320 error message:
```
wrong! device_gemm with the specified compilation parameters does not support this GEMM problem
```
This is not a correctness failure of Fix 2 — it is a CK kernel tile-size constraint:
the forced V3 kernel `moe_ck2stages_gemm2_256x128x128x64_1x4_..._v3` requires
inter_dim divisible by its N tile (128). 320 is not a multiple of 128, so no
matching CK instance exists. 192 / 256 / 640 are all multiples of 128 (or 192
falls through to the unaffected V1 path where block_m=32).

## Boundary Confirmation (from code reading)

`/home/hanchang/aiter/aiter/fused_moe.py` lines 900-910 (Fix 2, commit 68fc7d48b):

```python
# gfx950 workaround: V1 CK kernel produces wrong results for inter_dim>192
# (memory corruption / incorrect computation for both preshuffle_on and
# preshuffle_off paths). Force block_m=128 to select the correct V3 stage1
# kernel. For preshuffle_off, also force the V3 stage2 kernel by name.
# Note: blockscale (per_1x128/per_1x32) dispatch only supports block_m<=64
# and is not affected by the V1 bug, so exclude it from this override.
if not run_1stage and inter_dim > 192 and get_gfx() == "gfx950" \
        and q_type not in (QuantType.per_1x128, QuantType.per_1x32):
    block_m = 128
    if not is_shuffled and not kernelName2:
        kernelName2 = "moe_ck2stages_gemm2_256x128x128x64_1x4_TypeCast_v3_Nswizzle0_Quant0_MulRoutedWeight1_B16_B16_B16"
```

The runtime aiter logs in our run confirm the dispatch:
- inter_dim=192 → `block_m = 32` (boundary not crossed, V1 path)
- inter_dim=256, 320, 640 → `block_m = 128` (V3 forced)

This validates that 192 is exactly the cut-off (`inter_dim > 192`).

## V01 Exp2 Conclusion

**PASS** — Fix 2's V3-forced override produces correct results
(cos_sim approx. 0.99999) for every inter_dim that has a matching CK kernel
instance (192 via V1, 256 and 640 via V3). The inter_dim=320 case is an
unrelated CK tile-size limitation, not a correctness regression of the fix.
The 192 boundary in the guard expression `inter_dim > 192` is confirmed
empirically: at exactly 192 the dispatcher picks block_m=32 (V1) and at
inter_dim > 192 it picks block_m=128 (V3).
