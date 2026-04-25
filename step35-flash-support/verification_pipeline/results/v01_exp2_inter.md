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

## 补充：inter_dim=384（ATOM tp=4 padding 后实际值）

ATOM padding 逻辑：`/home/hanchang/ATOM/atom/model_ops/moe.py` L489-516
（BF16 path, `_maybe_pad_weight`）。核心判定（L502-503）：

```python
align = 64 if inter_dim <= 192 else 128
inter_pad = (inter_dim + align - 1) // align * align
```

对 tp=4（inter_dim=320）：align=128，inter_pad=384，触发 zero-pad；
gate/up 各扩到 384 行，w2 沿最后一维扩到 384。FP8 blockscale 路径
有等价处理，见 L1713-1740（block_n=128 对齐，padding 等价）。

补测脚本 `/tmp/v01_exp2b.py` 在 GPU 2 上对 inter_dim=320 与 384 跑 fused_moe
correctness。日志：
`/home/hanchang/project_fp8_tp4/verification_pipeline/results/logs/v01_exp2b.log`

| inter_dim | cos_sim | 结论 |
|-----------|---------|------|
| 320 (无 padding)   | ERROR (脚本 API 调用错误：fused_moe got multiple values for 'topk_ids') | 未能复现内核错误，但 V01 Exp2 主表已记录 CK tile 限制 |
| 384 (padding 后)   | ERROR (同上 API 错误) | 测量未完成 |

说明：本次 /tmp/v01_exp2b.py 因调用签名问题（fused_moe 同时收到位置/关键字
形式的 topk_ids）未能取得 cos_sim 数值。代码侧已确认 ATOM padding 320→384
逻辑（moe.py L489-516），与 aiter Fix 2 的 V3 内核 N-tile=128 约束一致：
inter_pad=384 是 128 的 3 倍，满足 stage1/stage2 对齐要求，理论上可正常
工作。运行时验证留待修复测试脚本后补做。

## 补充：inter_dim=384（ATOM tp=4 padding 后实际值）

ATOM padding 逻辑：moe.py L502-503（padding 320→384）

补测脚本 `/tmp/v01_exp2b_fix.py`（修复 API 调用：使用位置参数 tw, ti）。
GPU: gfx950, CUDA_VISIBLE_DEVICES=2。日志：
`/home/hanchang/project_fp8_tp4/verification_pipeline/results/logs/v01_exp2b_fix.log`

aiter dispatch 日志确认 V3 forced 路径：
`[aiter] [fused_moe] using 2stage default for (256, 32, 7168, 384, 16, 4, ...)`
`module_moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2`

| inter_dim | cos_sim | 结论 |
|-----------|---------|------|
| 320 (无padding) | CK tile 约束 ERROR | 必须 pad |
| 384 (padding后) | 0.99999249 | PASS |

结论：ATOM padding 320→384 是必要的，实际模型路径正确 ✓
