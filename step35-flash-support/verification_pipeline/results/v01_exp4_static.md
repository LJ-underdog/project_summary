# V01 实验 4 — 静态代码核查

实验类型：静态代码审查（只读 grep + Read），无 GPU 依赖。
来源：MASTER_PIPELINE §V01.4 + V01 §A Fix 1 风险评估扩展。

---

## 检查 1：ATOM moe.py gfx950 残留

grep 模式：`gfx950|gfx942|get_gfx|preshuffle_off|preshuffle_on`
文件：`/home/hanchang/ATOM/atom/model_ops/moe.py`

```
13:from aiter.jit.utils.chip_info import get_gfx
489:        # gfx950 CK a16w16 stage2 requires inter_dim % 64 == 0.
521:        # Previously skipped for gfx950 bf16 g1u1 on the assumption that the CK
522:        # 2-stage preshuffle_off (NSwizzle=0) kernel expected un-shuffled weights.
523:        # Verified 2026-04-23: preshuffle_off GEMM is wrong on gfx950; preshuffle_on
723:        gfx = get_gfx()
```

分析：
- L13: `get_gfx` import（必要）
- L489/521-523: 都是注释，无可执行 skip 分支
- L723: 在 `Mxfp4MoEMethod.__init__` 内 `gfx = get_gfx()`，配合 L724 `self.use_triton = gfx.startswith("gfx94") or (gfx.startswith("gfx95") and ATOM_USE_TRITON_GEMM)`，是 triton 路径选择，不是 shuffle skip

结论：**无残留 `if get_gfx() == "gfx950": pass` skip ✓**

---

## 检查 2：ATOM 各 quant 路径 shuffle_weights 覆盖

grep `shuffle_weights` 在 `/home/hanchang/ATOM/atom/model_ops/moe.py`：

```
49:    shuffle_weights,                                      # import
525:        shuffle_weights(layer.w13_weight, layer.w2_weight)   # Unquantized BF16 path
922:            shuffle_weights(layer.w13_weight, layer.w2_weight)   # Mxfp4MoEMethod 路径
934:            shuffle_weights(layer.w13_weight, layer.w2_weight)   # Mxfp4MoEMethod 路径
1592:        # later in _process_block_quant (after normalize, before shuffle_weights),
1749:        shuffle_weights(layer.w13_weight, layer.w2_weight)   # quark/blockscale 路径
1770:        shuffle_weights(layer.w13_weight, layer.w2_weight)   # channel quant 路径
1800:        shuffle_weights(layer.w13_weight, layer.w2_weight)   # tensor quant 路径
```

L520-525（BF16 关键修复）：
```
520        # Shuffle weights for CK/ASM kernels.
521        # Previously skipped for gfx950 bf16 g1u1 on the assumption that the CK
522        # 2-stage preshuffle_off (NSwizzle=0) kernel expected un-shuffled weights.
523        # Verified 2026-04-23: preshuffle_off GEMM is wrong on gfx950; preshuffle_on
524        # (NSwizzle=1) is correct. Always shuffle so the right kernel path is used.
525        shuffle_weights(layer.w13_weight, layer.w2_weight)
```

结论：**所有 6 个 shuffle 调用点（L525/922/934/1749/1770/1800）均为无条件调用，无任何 gfx950 / gfx942 守卫包裹 ✓**
（FP4/quark/blockscale/channel/tensor 各 quant 路径均覆盖到 shuffle_weights，无残留 skip）

---

## 检查 3：aiter fused_moe.py L906 guard

grep `gfx950|get_gfx|per_1x128|per_1x32` 在 `/home/hanchang/aiter/aiter/fused_moe.py`（关键行）：

```
18:from aiter.jit.utils.chip_info import get_cu_num, get_gfx
271:            if get_gfx() != "gfx950" or M < bf16_fp8_bound:
586:    "gfx950":          # fused_moe_1stage_dict 表
880:        ) in fused_moe_1stage_dict[get_gfx()]:
906:        if not run_1stage and inter_dim > 192 and get_gfx() == "gfx950" \
907:                and q_type not in (QuantType.per_1x128, QuantType.per_1x32):
```

L900-910 原文：
```
900        # gfx950 workaround: V1 CK kernel produces wrong results for inter_dim>192
901        # (memory corruption / incorrect computation for both preshuffle_on and
902        # preshuffle_off paths). Force block_m=128 to select the correct V3 stage1
903        # kernel. For preshuffle_off, also force the V3 stage2 kernel by name.
904        # Note: blockscale (per_1x128/per_1x32) dispatch only supports block_m<=64
905        # and is not affected by the V1 bug, so exclude it from this override.
906        if not run_1stage and inter_dim > 192 and get_gfx() == "gfx950" \
907                and q_type not in (QuantType.per_1x128, QuantType.per_1x32):
908            block_m = 128
909            if not is_shuffled and not kernelName2:
910            kernelName2 = "moe_ck2stages_gemm2_256x128x128x64_1x4_TypeCast_v3_Nswizzle0_Quant0_MulRoutedWeight1_B16_B16_B16"
```

guard 条件四件套：`not run_1stage` AND `inter_dim > 192` AND `gfx == gfx950` AND `q_type ∉ {per_1x128, per_1x32}`。

结论：**L906 guard 正确，仅排除 per_1x128/per_1x32 blockscale 路径 ✓**
- 条件覆盖 BF16 no-quant 与其它非 blockscale quant；
- blockscale 由于 dispatch 仅支持 block_m<=64，不能强制 128，已正确豁免。

---

## 检查 4：git show 触及范围

### ec8cbe8 (ATOM)
```
 atom/examples/simple_inference.py            |  12 +-
 atom/model_engine/model_runner.py            |  45 +-
 atom/model_loader/loader.py                  |  11 +
 atom/model_ops/attentions/aiter_attention.py |   4 +-
 atom/model_ops/moe.py                        |  22 +-
 atom/models/step3p5.py                       | 860 +++++++++++++++++++++++++++
 6 files changed, 938 insertions(+), 16 deletions(-)
```
分析：除 moe.py 的核心修复外，其余均为 Step-3.5-Flash 模型新增（step3p5.py）+ 配套的 num_attention_groups 处理 + fused expert 加载顺序修复。范围合理。

### 68fc7d48b (aiter)
```
 aiter/dist/parallel_state.py   |  2 +-
 aiter/fused_moe.py             | 90 ++++++++++++++++++++++++++++++++----------
 aiter/jit/utils/moe_recipes.py |  9 +++--
 3 files changed, 77 insertions(+), 24 deletions(-)
```
分析：fused_moe.py 90 行（4 个 bug 修复）+ parallel_state.py CustomAllreduce 禁用 + moe_recipes.py preshuffle 模式推断修复。三处均在 commit message 中已声明。

结论：**两个 commit 均仅触及目标文件范围 ✓**

---

## 检查 5：gfx942 回归

`gfx942` 在 ATOM moe.py：**无独立匹配**（仅 L723-724 通过 `gfx.startswith("gfx94")` 间接覆盖 gfx942 走 triton 路径，是既有逻辑非新增）

`gfx942` 在 aiter fused_moe.py：
```
571:    "gfx942":      # fused_moe_1stage_dict 表 key（已有）
1933:        get_gfx() in ["gfx942", "gfx950"]
```

分析：两处均为既有 dispatch 表 key 与 既有判断；ec8cbe8 / 68fc7d48b 修复未引入任何新的针对 gfx942 的条件分支。L906 的 guard `get_gfx() == "gfx950"` 严格限定 gfx950，不会影响 gfx942 路径。

结论：**无 gfx942 回归风险 ✓**

---

## 附加：3771835ac 归属

`git -C /home/hanchang/aiter log --oneline 3771835ac -1` 输出：
```
3771835ac revert: remove unnecessary +2 row padding in fused_moe.py (gfx950)
```
ATOM 仓库报 unknown revision。**3771835ac 为 aiter commit，不触及 ATOM moe.py**，与 V01 §A 评估一致。

---

## V01 实验 4 总体结论

**PASS**

具体证据：
1. ATOM moe.py 所有 6 处 shuffle_weights 调用点均无 gfx950 skip 守卫
2. aiter fused_moe.py L906 guard 严谨：四条件 AND，仅作用于 gfx950 非 blockscale 路径
3. ec8cbe8 / 68fc7d48b diff 范围与 commit message 声明一致，无意外修改
4. 无 gfx942 回归（既有 dispatch 表 key 不变，无新增 gfx942 分支）
5. 3771835ac revert 来源已确认（aiter，不触及 ATOM）
