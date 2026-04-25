# V02 Exp1: SwigluStep / fused_moe Correctness (Fixed API)

Date: 2026-04-25
GPU: CUDA_VISIBLE_DEVICES=7 (MI350X, gfx950)
Script: `/tmp/v02_exp1_fix.py`
Log: `/home/hanchang/project_fp8_tp4/verification_pipeline/results/logs/v02_exp1_fix.log`

This run replaces the prior V02 Exp1 attempt, which failed with
`fused_moe() got an unexpected keyword argument 'inplace'`. The fixed harness
drops `inplace` and calls `fused_moe` positionally with the first 5 params.

## 1. fused_moe Actual Signature (first 10 parameters)

Source: `/home/hanchang/aiter/aiter/fused_moe.py:120`

```python
def fused_moe(
    hidden_states,
    w1,                                              # [E, inter_dim*2, dim]  N,K
    w2,                                              # [E, dim, inter_dim]
    topk_weight,
    topk_ids,
    expert_mask: Optional[torch.tensor] = None,      # EP
    activation = ActivationType.Silu,
    quant_type = QuantType.No,
    doweight_stage1 = False,
    w1_scale: Optional[torch.tensor] = None,         # [E, inter_dim, 1]
    ...
)
```

Note: there is NO `inplace` argument. The fixed test uses positional
invocation `fused_moe(x, w1, w2, tw, ti)` matching the first 5 params.
Defaults => `activation=Silu`, `quant_type=No`, `doweight_stage1=False`.

## 2. Exp1 Matrix (BF16, model_dim=7168, E=8, topk=4, inter_dim=384)

| M   | inter_dim | seed | cos_sim | Status |
|-----|-----------|------|---------|--------|
| 1   | 384       | 0    | -       | ERROR  |
| 1   | 384       | 42   | -       | ERROR  |
| 32  | 384       | 0    | -       | ERROR  |
| 32  | 384       | 42   | -       | ERROR  |
| 256 | 384       | 0    | -       | ERROR  |
| 256 | 384       | 42   | -       | ERROR  |

All 6 cases failed with the same runtime error:

```
CKPyInterface: Unsupported data type 4
[aiter] Error in moe_sorting: CKPyInterface: Unsupported data type 4
[aiter] Moe_sorting info: max_num_tokens_padded=1024 block_size=128 num_experts=8 topk=4
```

The error originates in `moe_sorting` (CK Python interface) BEFORE the GEMM
kernels execute. This is a pre-kernel dispatcher failure, not a numerical
divergence. The aiter dispatcher logs confirm it picks `2stage default` for
`(N=256, M, K=7168, inter=384, E=8, topk=4, Silu, bf16, QuantType.No,
use_nt=True/False, doweight_stage1=False)`, then crashes inside CK
`moe_sorting`.

Overall: **FAIL (0/6)** — failure mode is API/dispatch (CK `moe_sorting`
unsupported data type 4), NOT silent numerical mismatch. The Silu/no-quant
BF16 path with these shapes is unusable in the current aiter build on gfx950.

## 3. SwigluStep Code Existence

### aiter (`/home/hanchang/aiter/aiter/`)

| File | Line | Content |
|------|------|---------|
| `aiter/fused_moe.py` | 974 | `and activation != ActivationType.SwigluStep` |
| `aiter/fused_moe.py` | 1389 | `if activation == ActivationType.SwigluStep:` |
| `aiter/fused_moe.py` | 1390 | `return swiglustep(gate, up)` |
| `aiter/fused_moe.py` | 1541 | `def swiglustep(x_glu, x_linear, limit: float = 7.0):` |
| `aiter/fused_moe.py` | 1641 | `use_swiglustep = activation == aiter.ActivationType.SwigluStep` |
| `aiter/fused_moe.py` | 1647-1648 | fallback: `out = swiglustep(gate, up)` |
| `aiter/utility/dtypes.py` | 151 | `"swiglustep": ActivationType.SwigluStep` |
| `aiter/jit/utils/moe_recipes.py` | 88, 94 | gfx950 swiglustep+no-quant -> preshuffle_off |
| `aiter/ops/quant.py` | 463-472 | reference `_swiglustep_single` (clamp limit) |

### ATOM (`/home/hanchang/ATOM/atom/`)

| File | Line | Content |
|------|------|---------|
| `models/step3p5.py` | 55 | `def _uses_swiglustep_at_layer(config, layer_idx) -> bool:` |
| `models/step3p5.py` | 78-81 | R5 mitigation comment: kernel clamps experts at 7.0 |
| `models/step3p5.py` | 89 | gating: skip fusion when layer is swiglustep |
| `models/step3p5.py` | 180, 190-192 | selects `ActivationType.SwigluStep` if `clamp_limit` set |
| `models/step3p5.py` | 216 | R5 mitigation: must NOT fuse shared expert at SwigluStep layers |
| `models/step3p5.py` | 290 | Routed experts at SwigluStep layers 43-44 |

**Conclusion: SwigluStep code EXISTS** end-to-end — aiter kernel + dispatcher
+ ATOM Step-3.5 model wiring. Hard-coded clamp limit = 7.0.

## 4. Overall Conclusion

- **fused_moe API verified**: signature confirmed at `fused_moe.py:120`. No
  `inplace` kwarg; the 5-positional invocation works in the dispatcher.
- **SwigluStep integration verified**: present in aiter
  (`fused_moe.py:1541`) and consumed by ATOM Step-3.5
  (`step3p5.py:192`) with the R5 clamp-at-7.0 mitigation at routed-MoE
  layers 43-44.
- **Exp1 matrix FAILED 0/6** due to a `moe_sorting` CK Python interface error
  (`Unsupported data type 4`). Pre-kernel dispatch failure, not a numerical
  bug. The Silu BF16 / QuantType.No path is currently broken on this gfx950
  build for these shapes; numerical SwigluStep cosine cannot be measured by
  this harness as written.
- Suggested follow-ups (not executed in this run):
  1. Force the SwigluStep path explicitly (`activation=ActivationType.SwigluStep`).
  2. Use the `preshuffle_off` recipe (see `moe_recipes.py:88-94`).
  3. Rebuild aiter so CK `moe_sorting` accepts data type 4.

V02 Exp1 status: **FAIL (0/6 cases, dispatch error)**, SwigluStep: **EXISTS**.
