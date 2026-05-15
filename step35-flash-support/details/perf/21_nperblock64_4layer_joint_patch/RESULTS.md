# NPerBlock=64 path 4 层 Joint Patch — RESULTS

> **atomic story**: stepfun-Flash-FP8 tp=4（inter_dim=320）走 CK MoE block-scale GEMM 的 `NPerBlock=64` dispatch path 触发 4 层耦合 scale layout mismatch；4 层 joint patch（CK 3 层 + caller 1 层）落地后 → bit-exact correctness（max_err=0.0）+ kernel-level 快 15.80% + model-level PERF_NEUTRAL（HIGH confidence）
> **wave triple (atomic story)**: `m1quad_joint_fix_wave` (correctness) + `m1quad_perf_compare_wave` (kernel-level perf) + `m1quad_model_perf_wave` (model-level e2e perf)
> **WORK_DIR (project summary 整合 wave)**: `/home/junlin12/m1quad_project_summary_wave/`
> **结案日期**: 2026-05-15
> **硬件 axis**: ⚠️ **MI308X (gfx942) 实测**；本 wave 不引用 gfx950 entry 数据（参 §10 hardware-axis 红线）
> **承接**: 上游 partial root cause `m1ppp_padding_explanation_wave/PADDING_EXPLAINED.md §3`（H_alt_1 partial mechanism）
> **措辞 prefix 4 级**：【实证】 / 【实证修正】 / 【推测未实证】 / 【通用教训】

---

## TL;DR

> # ✅ Wave 结论 — PASS（dual-superior on this fixture，多 caveat 限定）

- 【实证】**NPerBlock=64 path 4 层 joint patch（CK `gridwise_moe_gemm_blockscale.hpp` 105 加 / 15 减 + caller `w1_quant_blk_n` dispatch-aware 改）** 在 stepfun-Flash-FP8 tp=4 + MI308X (gfx942) 上达成：
  - **Correctness**：inter_dim ∈ {192, 320, 448} 三点 NPerBlock=64 path **全部 max_err = 0.0 bit-exact**（vs 上游 wave 实证 baseline max_err=0.6956 / 69.6% 元素超 atol=0.05）；inter_dim=384 NPerBlock=128 control 不 regress
  - **Kernel-level perf**：no-padding (NPerBlock=64 + 4 层 patch) 比 padding (NPerBlock=128) **快 12.186 us / 15.80%**（3 run avg median，信号 / 噪声 ≈ 5×）；反向 padding kernel-level cost = +18.77%
  - **Model-level e2e perf**：F-A 同 HEAD `f06cdcca5` 唯一变量 = CK patch on/off，3 runs median delta 全部 ≤ 0.81%（TTFT -1.5% / TPOT +0.7% / Decode -0.7% 均落 noise floor）→ **PERF_NEUTRAL (HIGH confidence)**
- 【实证】**Cross-wave 自洽**：kernel-level 15.80% 收益在 model e2e 被 attention/MLA/sampler/dispatch 等 overhead 稀释 ~19.5–21.6×（与 KNOWN_FACT F3「e2e overhead amortize kernel cost」一致）
- 【通用教训】**Joint patch 必要性**：4 层任一缺失都 FAIL — candidate A 仅 L1 → max_abs 588 反劣 -0.7%；candidate B 仅 caller reshape → max_err +17%。**joint = atomic apply 全 4 层才能 PASS**（§2 + §6 详）
- 【推测未实证】**production 边界**：Verify path = 自定义 `ck_moe_stage1` 直调 + verify-script L4 caller patch；vLLM serve 走 `aiter.fused_moe()` 高层是否 route 到同一 CK kernel = 部分路径未实证；aiter `fused_moe.py` `fc_scale_blkn` production dispatch path 改造 = A2 deferred（§7 + §10 caveat C3/C6）
- 【实证】**3 个 incident 透明度记录**（model-level F-A 期间）：stash 顺序混淆 / cache.py develop=True 误 stash / `/tmp/comgr-*` disk full — 全部修复 + state restored ✅，无环境破坏（§8）

---

## 1. Outcome（PASS 显式声明 + 章节 PASS/FAIL 标注）

### 1.1 一句话结论

【实证】**Wave PASS**：在 fixture (TOKEN=32 / MODEL_DIM=7168 / E=8 / TOPK=2 / dtype=bf16 / 单 SEED=0xc0de 等) 实证 4 层 joint patch 达成 dual-superior（correctness bit-exact + kernel-level 快 15.80%），model-level e2e PERF_NEUTRAL HIGH confidence；适用边界严格限定为 stepfun-Flash-FP8 tp=4 + MI308X (gfx942) + NPerBlock=64 path（详 §10 caveat）。

来源：`m1quad_joint_fix_wave/WAVE_CLOSE.md` + `m1quad_perf_compare_wave/PERF_COMPARE.md` + `m1quad_model_perf_wave/WAVE_CLOSE.md`。

### 1.2 章节级 PASS/FAIL 标注

| Phase / Wave | 内容 | verdict |
|---|---|---|
| 上游 (partial root cause) | `m1ppp_padding_explanation_wave` H_alt_1 mechanism partial 提出 | ⚠️ partial（C1 caveat：sanity coverage 仅 83%）|
| Wave 1 — Joint Fix | 4 层 patch 落地 + 三点 NPerBlock=64 bit-exact | ✅ PASS |
| Wave 2 — Kernel-level Perf | pad 77.118 us vs nopad 64.932 us → 15.80% delta | ✅ PASS（信号 5× 噪声）|
| Wave 3 — Model-level Perf (init) | nopad 3 runs + L20 historical baseline 对比 | ✅ PASS（PERF_NEUTRAL，但 caveat C1/C4/C7 open）|
| Wave 3 — Model-level Perf (F-A 终极) | 同 HEAD CK patch on/off 3 runs 对比 | ✅ PASS（PERF_NEUTRAL HIGH confidence；C1/C4/C7 全破）|
| Production e2e (vLLM serve) | A2 production patch + 实测 | 🟡 DEFERRED（C3/C6 未验证，需 follow-up wave）|

---

## 2. 工程动作完成度（4 层 patch + 3 wave 实测）

### 2.1 4 层 patch 落地表

> 行号字段 = **post-patch 实地行号**（per `git -C /workspace/aiter/3rdparty/composable_kernel diff include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm_blockscale.hpp` + `grep -n "m1quad_joint_fix_wave / 2026-05-14"` 实地核对，不二次引用 IMPL_VERIFY_REPORT）。Run 段 marker 实地：L1=L1237 / L2=L1164 / L3-alt=L1483；Run_2Lds 段 marker 实地：L1=L1830 / L3-alt=L2079。L3 (gate 段) hpp 内**无独立 marker 注释**，定位 = `git diff` hunk `@@ -1415,7 +1440,16 @@`（Run 段）+ `@@ -1971,7 +2036,16 @@`（Run_2Lds 段）。

| # | 层 | 文件 + 实地行号 | 改动量 | 修复 mechanism |
|---|---|---|---|---|
| L1 | `expert_scale_stride` 公式 | `gridwise_moe_gemm_blockscale.hpp` marker L1237 + 改动 L1242-1254 (Run) / marker L1830 + 改动 L1834-1846 (Run_2Lds) | constexpr if 分支（`[&]()` capture-default 实地）| per-expert scale tensor N 方向 stride：`NPerBlock<ScaleBlockN` 时改用 `ceil(2N, NPerBlock)` (=10) 而非 `ceil(N, ScaleBlockN)*2` (=6) |
| L2 | `b_scale_grid_desc_bn_ak` tensor descriptor shape | marker L1164 + 改动 L1165-1178 (Run) / 改动 L1718 区段 (Run_2Lds，无独立 marker，与 Run 段双 codegen 复制)| constexpr if 分支（`[&]()` capture-default 实地）| scale buffer 逻辑 N 维：`NPerBlock=64` path desc N=10 而非 3 |
| L3 | gate 段 `b_scale_thread_copy` multi_index 起点 | hpp 改动 L1444-1452 (Run, hunk `@@ -1415,+1440 @@`) + L2036-2045 (Run_2Lds, hunk `@@ -1971,+2036 @@`)（hpp 内无独立 marker 注释）| lambda 分支（`[&]()` capture-default 实地）| thread tile 在 scale desc 内 N 起点：`NPerBlock=64` path 改 `block_n_id` 直接用（避免 `block_n_id * 64/128 = 0` 截断 collapse）|
| L3-alt | up 段 dual-pointer 简化 + multi_index | marker L1483 + dual-pointer 改 L1487-1490 + multi_index 改 L1507-1517 (Run) / marker L2079 + 改动 L2083-2086 + L2104-2114 (Run_2Lds) | ternary + lambda（`[&]()` capture-default 实地）| NPerBlock=64 path：caller 已把 gate (row 0..4) + up (row 5..9) 在同张量 stack；CK 不再 dual-pointer 偏移；up 起 row = `block_n_id + ceil(2N, NPerBlock)/2` |
| L4 | caller `w1_quant_blk_n` dispatch-aware | verify fixture: `m1quad_joint_fix_wave/scripts/correctness_compare_v2.py:91-101`; **production 等价**: `/workspace/aiter/op_tests/test_moe_blockscale.py` L190-215（PATCH_SPEC 描述）| Python 5 行 | caller 端 fp8 量化块 N 方向必须与 dispatch NPerBlock 一致：`64 if (inter_dim%128!=0 and inter_dim%64==0) else scale_blk_n` |

【实证】**统计**：CK hpp 文件 105 insertions / 15 deletions（实地 `git -C /workspace/aiter/3rdparty/composable_kernel diff --stat include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm_blockscale.hpp` 实证；本 wave synth 阶段亦实地核对：`120 ++++++++++++++++++--- | 1 file changed, 105 insertions(+), 15 deletions(-)`）。

【实证】**hpp 文件 Run + Run_2Lds 双 codegen 段**：4 形态 patch 全在两段都 apply（共 8 hunk，patch 文件含 4 唯一形态 × 2 段重复）。来源：`m1quad_joint_fix_wave/IMPL_VERIFY_REPORT_v2.md:60` 注 1 + grep marker 出现在 L1164+L1718 / L1227+L1830 / L1418+L1974 / L1450+L2007 等。

### 2.2 4 层 patch verbatim diff（关键 snippet）

> 以下 snippet 为 `git -C /workspace/aiter/3rdparty/composable_kernel diff include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm_blockscale.hpp` **直接 copy-paste** 输出（含 `+` / `-` 行 + hunk header），未改写 / 未简化；lambda 实际形态 = `[&]()` capture-default（C++20 `constexpr if` 内访问 `problem` 必须显式 capture，否则编译 fail — IMPL_VERIFY_REPORT_v2.md §3 注 2 已实证 86s rebuild 修复）。

**L2 — `b_scale_grid_desc_bn_ak`**（marker L1164 / Run 段，hunk `@@ -1161,+1161 @@`）：
```diff
@@ -1161,11 +1161,21 @@ struct GridwiseMoeGemmBlockScale
                                                   ScaleBlockM),
                        math::integer_divide_ceil(problem.K, ScaleBlockK)),
             make_tuple(math::integer_divide_ceil(problem.K, ScaleBlockK), 1));
-        const auto b_scale_grid_desc_bn_ak = make_naive_tensor_descriptor(
-            make_tuple(math::integer_divide_ceil(problem.N * (IsInputGemm && IsSplitK ? 2 : 1),
-                                                 ScaleBlockN),
-                       math::integer_divide_ceil(problem.K, ScaleBlockK)),
-            make_tuple(math::integer_divide_ceil(problem.K, ScaleBlockK), 1));
+        // [m1quad_joint_fix_wave / 2026-05-14 patch L2]
+        const auto b_scale_grid_desc_bn_ak = [&]() {
+            if constexpr (NPerBlock < ScaleBlockN) {
+                return make_naive_tensor_descriptor(
+                    make_tuple(math::integer_divide_ceil(problem.N * 2, NPerBlock),
+                               math::integer_divide_ceil(problem.K, ScaleBlockK)),
+                    make_tuple(math::integer_divide_ceil(problem.K, ScaleBlockK), 1));
+            } else {
+                return make_naive_tensor_descriptor(
+                    make_tuple(math::integer_divide_ceil(problem.N * (IsInputGemm && IsSplitK ? 2 : 1),
+                                                         ScaleBlockN),
+                               math::integer_divide_ceil(problem.K, ScaleBlockK)),
+                    make_tuple(math::integer_divide_ceil(problem.K, ScaleBlockK), 1));
+            }
+        }();
```

**L1 — `expert_scale_stride` 公式**（marker L1237 / Run 段，hunk `@@ -1224,+1234 @@`；Run_2Lds 段 marker L1830 hunk `@@ -1777,+1827 @@` 内容形式同）：
```diff
@@ -1224,9 +1234,24 @@ struct GridwiseMoeGemmBlockScale
         });
         const long_index_t expert_stride = __builtin_amdgcn_readfirstlane(
             static_cast<long_index_t>(problem.N) * problem.K * (IsInputGemm ? 2 : 1));
+        // [m1quad_joint_fix_wave / 2026-05-14 patch L1]
+        // NPerBlock=64 path: caller passes per-N-tile quant layout
+        //   shape (E, ceil(2*N, NPerBlock), ceil(K, SBK)) = (E, 10, 56) for N=320
+        //   per-expert stride = 10 * 56 = 560 fp32
+        // NPerBlock>=128 path: unchanged
         const long_index_t expert_scale_stride = __builtin_amdgcn_readfirstlane(
-            static_cast<long_index_t>(math::integer_divide_ceil(problem.N, ScaleBlockN)) *
-            (IsInputGemm ? 2 : 1) * math::integer_divide_ceil(problem.K, ScaleBlockK));
+            [&]() {
+                if constexpr (NPerBlock < ScaleBlockN) {
+                    return static_cast<long_index_t>(
+                        math::integer_divide_ceil(problem.N * 2, NPerBlock)) *
+                        math::integer_divide_ceil(problem.K, ScaleBlockK);
+                } else {
+                    return static_cast<long_index_t>(
+                        math::integer_divide_ceil(problem.N, ScaleBlockN)) *
+                        (IsInputGemm ? 2 : 1) *
+                        math::integer_divide_ceil(problem.K, ScaleBlockK);
+                }
+            }());
```

**L3 — gate 段 `b_scale_thread_copy` multi_index 起点**（Run 段 hunk `@@ -1415,+1440 @@`，L1444-1452；Run_2Lds 段 hunk `@@ -1971,+2036 @@` 内容形式同）— hpp 内**无独立 marker 注释**：
```diff
@@ -1415,7 +1440,16 @@ struct GridwiseMoeGemmBlockScale
                                              ScaleSliceSizeK,
                                              1,
                                              false>(
-                b_scale_grid_desc_bn_ak, make_multi_index(block_n_id * NPerBlock / ScaleBlockN, 0));
+                b_scale_grid_desc_bn_ak,
+                make_multi_index(
+                    [&]() {
+                        if constexpr (NPerBlock < ScaleBlockN) {
+                            return block_n_id;
+                        } else {
+                            return block_n_id * NPerBlock / ScaleBlockN;
+                        }
+                    }(),
+                    0));
```

**L3-alt — up 段 dual-pointer 简化 + multi_index**（marker L1483 / Run 段，hunk `@@ -1446,+1480 @@` + `@@ -1464,+1504 @@`；Run_2Lds 段 marker L2079 hunk `@@ -2002,+2076 @@` + `@@ -2021,+2101 @@` 内容形式同）：
```diff
@@ -1446,8 +1480,14 @@ struct GridwiseMoeGemmBlockScale
                                        get_warp_local_1d_id() % NWave,
                                        0,
                                        KPack / KGroup * (get_thread_local_1d_id() % WarpSize)));
+            // [m1quad_joint_fix_wave / 2026-05-14 patch L3-alt]
+            // NPerBlock=64 path: scale buf_up shares same per-expert offset as buf;
+            //   up rows start at block_n_id = ceil(2N, NPerBlock)/2 inside full desc
+            // NPerBlock>=128 path: unchanged dual-pointer offset
             const BScaleType* p_b_scale_grid_up =
-                p_b_scale_grid + expert_scale_stride / 2 / BPackedSize;
+                (NPerBlock < ScaleBlockN)
+                    ? p_b_scale_grid
+                    : p_b_scale_grid + expert_scale_stride / 2 / BPackedSize;
@@ -1464,7 +1504,17 @@ struct GridwiseMoeGemmBlockScale
                                                  1,
                                                  false>(
                     b_scale_grid_desc_bn_ak,
-                    make_multi_index(block_n_id * NPerBlock / ScaleBlockN, 0));
+                    make_multi_index(
+                        [&]() {
+                            if constexpr (NPerBlock < ScaleBlockN) {
+                                // NPerBlock=64 path: up rows start at ceil(2N,NPerBlock)/2
+                                return block_n_id +
+                                    math::integer_divide_ceil(problem.N * 2, NPerBlock) / 2;
+                            } else {
+                                return block_n_id * NPerBlock / ScaleBlockN;
+                            }
+                        }(),
+                        0));
```

**L4 — caller `w1_quant_blk_n` dispatch-aware**（verify script 91-101）：
```python
w1_quant_blk_n = (
    64 if (inter_dim % 128 != 0 and inter_dim % 64 == 0) else scale_blk_n
)
# 同时改 w1_q rearrange 与 w1_scale view 都用 w1_quant_blk_n
```

来源：以上 4 层 verbatim diff snippet 为本 wave synth-v2 阶段实地 `git diff` 直接 copy-paste（不二次引用 IMPL_VERIFY_REPORT_v2.md / reader-joint-fix progress 的 pre-fix 旧版本）；本 file 在 CK working tree 共 105 insertions / 15 deletions（实地 `git diff --stat` 实证）。

### 2.3 三 wave 通用教训：correctness + kernel-level + model-level 三层缺一不可

【通用教训】4 层 patch 落地后**仅 verify correctness bit-exact**不构成 wave 完整 PASS：
- correctness PASS = 改动数学正确 / 无 OOB / 无 silu(0) collapse ✅
- kernel-level perf 测 = 验证改动是否引入新 perf 退化（NPerBlock=64 path kernel 是否反而慢于 NPerBlock=128 path） ✅
- model-level e2e perf 测 = 验证 kernel-level 收益是否 amortize 后仍可见 / 是否引入新 e2e 退化 ✅

【通用教训】**反模式**：correctness PASS 后跳过 kernel + model perf 实测直接 promote 到 production — 本 atomic story 三 wave 实测明确显示 kernel-level 15.80% 收益被 e2e overhead 稀释 ~20×，model-level 落 noise floor (PERF_NEUTRAL HIGH)，**production 决策依据是「不退化 + correctness 修复」**而非「kernel-level 显著加速」。如跳过 model perf 验证，无法确认 e2e 是否引入新瓶颈。

---

## 3. 数据矩阵

### 3.1 Correctness 表（来源：`m1quad_joint_fix_wave`）

| Path | inter_dim | NPerBlock | max_err pre-patch | max_err post-patch | Verdict |
|---|---|---|---|---|---|
| A | 320 | 64 | 0.6956（max_abs 588 / 69.6% 元素 > atol=0.05）| **0.0 bit-exact** | ✅ PASS |
| B (control) | 384 | 128 | 0.0 | **0.0** | ✅ no regression |
| Cross | 192 | 64 | (未跑) | **0.0 bit-exact** | ✅ PASS（P1 closure）|
| Cross | 448 | 64 | (未跑) | **0.0 bit-exact** | ✅ PASS（P1 closure）|

【实证】Test fixture：SEED = `0xc0de`（手动 `torch.manual_seed`），TOKEN=32, MODEL_DIM=7168, E=8, TOPK=2, dtype=bf16, atol=rtol=0.05, `use_g1u1=True` 强制 stacked-N path。

【实证】Reference 基线：`torch_ref_stage1`（本 wave `scripts/correctness_compare_v2.py:152-191` 自定义 inline dequant 实现，因 aiter built-in `torch_moe_stage1` 硬编码 128-row dequant 不支持 dispatch-aware blk_n）。被测：`ck_moe_stage1` 直调 CK 2-stage 入口 (CK module `module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2.so`)。

来源：`m1quad_joint_fix_wave/IMPL_VERIFY_REPORT_v2.md:73-75` + `CROSS_CONFIG_VERIFY.md:51-52` + 4 个 log。

### 3.2 Kernel-level Perf 表（来源：`m1quad_perf_compare_wave/PERF_COMPARE.md`）

#### Padding case (inter_dim=384, NPerBlock=128 path)

| seed | median (us) | p50 | p95 | min | max |
|---|---|---|---|---|---|
| 0x0000c0de | 78.198 | 78.318 | 84.718 | 74.839 | 98.517 |
| 0x0000beef | 76.658 | 76.678 | 81.078 | 75.159 | 94.798 |
| 0x00001234 | 76.498 | 76.519 | 81.239 | 75.039 | 93.158 |
| **avg(median)** | **77.118** | — | — | — | — |

【实证】run-to-run median spread = 1.70 us (~2.2%)。

#### No-padding case (inter_dim=320, NPerBlock=64 path + 4 层 joint patch)

| run | seed | iters | median (us) | p50 | p95 | min | max | mean | stdev |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0xc0de | 50 | 65.599 | 65.599 | 98.637 | 59.919 | 196.556 | 71.542 | 25.803 |
| 2 | 0xbeef | 50 | 64.479 | 64.479 | 72.358 | 62.039 | 96.957 | 65.439 | 5.133 |
| 3 | 0xdead | 50 | 64.718 | 64.758 | 74.479 | 63.119 | 90.797 | 66.200 | 5.020 |
| **avg(median)** | — | — | **64.932** | — | — | — | — | — | — |

【实证】run-to-run median spread = 1.12 us (~1.7% of median)；3 run min-of-min = 59.919 us (pure-GPU lower bound 估计)。

#### Delta 表

| 指标 | 公式 | 数值 |
|---|---|---|
| 绝对 delta (us) | nopad − pad | **−12.186 us**（负 = no_pad 更快）|
| 相对 (no_pad vs pad) | (nopad − pad) / pad | **−15.80%** |
| 反向 padding cost | (pad − nopad) / nopad | **+18.77%** |
| 合并噪声估计 | pad_spread + nopad_spread ≈ 2.2% + 1% | **~3%** |
| 信号/噪声比 | 15.80% / 3% | **≈ 5×**（信号显著）|

【实证】Bench 方法：`torch.cuda.Event(enable_timing=True)` + record start / op / record end + `cuda.synchronize()` per iter；WARMUP = 10 / MEASURE = 50；含 host-side launch overhead（pad / nopad 同方法 → delta 中 overhead 抵消，caveat C11）。

### 3.3 Model-level e2e Perf 表（来源：`m1quad_model_perf_wave/PERF_PAD_CURRENT_MODEL_RESULT.md` F-A 终极实证）

#### F-A 设计核心
**唯一变量 = CK 4 层 patch on/off**；同 aiter HEAD `f06cdcca5` / 同 cmd / 同 prompt / 同 tp=4 / 同 model / 同 GPU 配置（4×gfx942 / MI308X）。stash 操作：`git stash push -m "f-a-stash-ck-patch" include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm_blockscale.hpp` + 删除 prebuild .so 强制 rebuild against clean CK。

#### F-A Padding case (CK pre-patch, NPerBlock=128 主路径) 3 runs

| Run | TTFT (ms) | TPOT (ms/tok) | Decode TPS | Total lat (s) | Engine init (s) | n_out | CORRECTNESS |
|---|---|---|---|---|---|---|---|
| 1 | 968.6 | 14.8 | 67.8 | 5.379 | 136.56 | 300 | PASS |
| 2 | 973.7 | 14.7 | 68.0 | 4.722 | 127.25 | 256 | PASS |
| 3 | 973.2 | 14.7 | 68.2 | 4.142 | 136.00 | 217 | PASS |
| **Median** | **973.2** | **14.7** | **68.0** | — | 136.00 | — | 3/3 PASS |
| spread | 5.1 (0.5%) | 0.1 (0.7%) | 0.4 (0.6%) | — | 9.31 (6.8%) | — | — |

#### No-padding case (CK post-patch, NPerBlock=64 path + 4 层 patch) 3 runs

| Run | n_in | n_out | TTFT (ms) | TPOT (ms/tok) | Decode tok/s | Total lat (s) | Engine init (s) | CORRECTNESS |
|---|---|---|---|---|---|---|---|---|
| 1 | 10213 | 264 | 965.0 | 14.6 | 68.5 | 4.803 | 135.13 | PASS |
| 2 | 10213 | 215 | 970.1 | 14.6 | 68.5 | (~4.10) | 121.22 | PASS |
| 3 | 10213 | 245 | 965.4 | 14.7 | 68.2 | (~4.55) | 135.89 | PASS |
| **Median** | — | — | **965.4** | **14.6** | **68.5** | — | — | 3/3 PASS |
| spread | — | — | 5.1 (0.53%) | 0.1 (0.7%) | 0.3 (0.4%) | — | — | — |

#### F-A 直接对比

| Metric | **Padding (CK pre-patch, F-A)** | No-padding (CK post-patch) | Delta (pad vs nopad) |
|---|---|---|---|
| TTFT (ms) | **973.2** | 965.4 | +7.8 ms (**+0.81%**) — pad 慢一丝 |
| TPOT (ms/tok) | **14.7** | 14.6 | +0.1 ms (**+0.68%**) — pad 慢一丝 |
| Decode TPS | **68.0** | 68.5 | -0.5 (**-0.73%**) — pad 慢一丝 |
| CORRECTNESS | 3/3 PASS | 3/3 PASS | 等价 |

【实证】**全部 ≤ 1%，落在 noise floor 内** → **PERF_NEUTRAL (HIGH confidence)**。

【实证】Bench cmd verbatim：`--input-tokens 10240 --output-tokens 1024 --runs 2 --measure-method A`；env：`CUDA_VISIBLE_DEVICES=0,1,2,3`、`AITER_LOG_LEVEL=WARNING`、`HF_HOME=/mnt/nvme0/arliu/hf_cache`；method A：TTFT + TPOT 来自 ATOM engine 内部 metrics `out["ttft"]` / `out["tpot"]`；显式 `--model $STEP35_PATH` 双保险（env + CLI），规避 ATOM `EngineArgs --model` default = `Qwen/Qwen3-0.6B` 陷阱（关联 KNOWN_FACT 「Qwen3-0.6B 抢占陷阱」）。

【实证】Model checkpoint：`stepfun-ai/Step-3.5-Flash-FP8`，snapshot SHA `6eebda59dd87ca5729648ec7cfed0becfceb273e`；实际加载 path `/mnt/nvme0/arliu/hf_cache/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/6eebda59dd87ca5729648ec7cfed0becfceb273e`（TEAM_CONFIG 默认 `/workspace/hf_cache/hub/...` snapshot 目录 EMPTY，0 blobs，PID 20785 download 进程已死 → 改用 `/mnt/nvme0/...` 备用副本，同一 snapshot SHA）。

### 3.4 Dispatch path 实证（model-level F-A 期间）

```
$ ls -la /workspace/aiter/aiter/jit/module_moe_ck2stages_*per_1x128*.so
-rwxr-xr-x. 4887384 May 15 01:36 module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2.so
-rwxr-xr-x. 4916176 May 15 01:38 module_moe_ck2stages_f8_f8_preshuffle_on_b16_swiglustep_per_1x128_mulWeightStage2.so
```

【实证】.so mtime = May 15 01:36/01:38（本次 3 runs 期间）= **clean CK state（patch stashed）** 重新编译产物 ✅；文件名含 `per_1x128` → **NPerBlock=128 padding 主路径** ✅；没有 `NPerBlock=64` (nopad 路径) `.so` 出现 ✅；`swiglustep` + `silu` 两条都触发了。来源：`PERF_PAD_CURRENT_MODEL_RESULT.md:53-66`。

---

## 4. Root Cause（4 层耦合 scale layout mismatch）

【实证】**一句话 root cause**：stepfun-Flash-FP8 tp=4 (inter_dim=320) 走 CK MoE block-scale GEMM 的 `NPerBlock=64` dispatch path 时，CK kernel 与 caller 之间 **scale tensor layout 契约不一致** —— caller 提供 `(E, ceil(2N, NPerBlock=64), ceil(K, SBK)) = (8, 10, 56)` 但 CK kernel 在 4 个独立位置都按 `ScaleBlockN=128` 计算 stride / descriptor / multi_index 起点 / dual-pointer 偏移，形成 **4 层耦合 mismatch**。

### 4.1 触发条件（dispatch entry）

【实证】`if (inter_dim%128 != 0 && inter_dim%64 == 0)` → `ck_moe_stage1_gemm<...,V1,256,X,64,...>` lookup entry 6 = `moe_ck2stages_gemm1_256x64x64x128_1x4_..._silu_F8_F8_B16`（KNOWN_FACT F7）。stepfun-Flash-FP8 tp=4 时 `inter_dim = moe_intermediate_size / TP = 1280 / 4 = 320`，正好命中。

### 4.2 4 层 mechanism 链

| 层 | mechanism | pre-patch 行为 |
|---|---|---|
| L1 | `expert_scale_stride` 公式按 `ceil(N=320, ScaleBlockN=128)*2 = 6` 算 | per-expert scale tensor stride 错（caller 实际 layout 是 `ceil(2N=640, NPerBlock=64) = 10`）|
| L2 | `b_scale_grid_desc_bn_ak` desc N 维 = `ceil(N, ScaleBlockN) = 3` | descriptor 仅 3 row，访问 row 3..9 OOB |
| L3 | gate 段 `block_n_id * NPerBlock(64) / ScaleBlockN(128) = block_n_id / 2` 截断 | block_n_id=1 与 0 都映射 row 0 → N tile 不可区分 |
| L3-alt | up 段 `expert_scale_stride/2` dual-pointer 偏移 | up 跨 expert 边界读到下个 expert 的 gate scale |

【实证】**结果链**：scale buffer 读取的 stride / index 与实际 layout 错位 → 触发 buffer_resource OOB（候选 H_alt_1 mechanism）→ silu(0)=0 塌缩 → 数值错误（pre-patch max_err=0.6956 / max_abs 588 / 69.6% 元素超 atol=0.05）。

### 4.3 影响范围

【实证】仅 stage1 IsInputGemm=true 的 MoE block-scale 路径；NPerBlock=128 path（如 inter_dim=384）不受影响（pre-patch 即 bit-exact，KNOWN_FACT F2 / 本 wave §3.1 control row 实证）。

### 4.4 Joint 必要性实证

【实证】4 层修任一缺失都 FAIL（KNOWN_FACT F4/F5 实证）：
- **candidate A 仅 L1**：max_abs 588 vs baseline 584 反向 -0.7%
- **candidate B 仅 L4-style reshape**（caller 不 dispatch-aware，CK 不动）：max_err 反升 +17%（0.6956 → 0.815）
- **joint = atomic apply 全 4 层**：max_err = 0.0 bit-exact

来源：`m1ppp_ck_bscale_fix_verify_wave/M1PPP_FIX_VERIFY_RESULTS.md` + `m1quad_joint_fix_wave/PATCH_SPEC.md §6 R6 / Step 严格顺序红线`。

---

## 5. Cross-wave 自洽 — kernel-level vs model-level

| 层级 | 数据 | wave 来源 |
|---|---|---|
| Kernel level | no-padding **快 12.186 us / 15.80%** vs padding | `m1quad_perf_compare_wave/PERF_COMPARE.md` |
| Model level | no-padding **快 0.7-0.8%**（统计上不显著）| `m1quad_model_perf_wave/WAVE_CLOSE.md` |
| Amortize 比 | 15.80% / 0.81% ≈ **19.5×**；15.80% / 0.73% ≈ **21.6×** | math 校验 |

【实证】**Kernel level 15.80% 收益在 model e2e 被 attention / MLA / sampler / dispatch / KV cache 等 overhead 稀释 ~19.5–21.6×**，与 KNOWN_FACT F3（"e2e overhead amortize kernel cost"）完全一致。

【通用教训】**kernel-level perf 收益不能直接外推到 e2e**：本 atomic story 实证 9.4–21.6× 量级差区间（`m1quad_perf_compare_wave/PERF_COMPARE.md §3` 报告 9.4× 来自 kernel +18.77% vs F6 端到端 +1.99%；本 wave model-level F-A 实测 19.5–21.6×）— 量级差具体值取决于对照参考点（端到端历史 baseline vs 同 HEAD F-A）；任何 kernel-level perf 改善的 production 决策必须配合 e2e 实测。

---

## 6. 9× discrepancy 解析（model wave 内部，关于 baseline 1 EXCLUDED 决策）

### 6.1 两条候选 padding baseline 数值差异

| Source | TTFT | TPOT | decode tps |
|---|---|---|---|
| **Padding baseline 1** (`perf_tp_eval/PERF_REPORT.md`, perf_bench.py)| **0.110 s (110 ms)** | 5.451 ms/tok | 183.44 tok/s |
| **Padding baseline 2** (L20, `/workspace/perf_logs/L20/stepfun_fp8_tp4.log`) | **0.980 s (980 ms)** | 14.5 ms/tok | 69.1 tok/s |

【实证】**差距：TTFT 9× / TPOT 2.66× / decode tps 2.66×** —— 同硬件 / 同 model / 同 tp / 同 input 量级 (~10240) **不可能由 aiter HEAD 漂移 (`0f8164017` vs `315123ace`) 解释**，必由 script 方法学差异主导。

### 6.2 关键归因 — Bench script 不同

【实证】**baseline 1** 来源：`/home/junlin12/project_fp8_tp4_repro/perf_tp_eval/perf_bench.py`（perf_tp_eval wave 2026-04-29）；**baseline 2 (L20) + nopad** 来源：`/tmp/project_summary/step35-flash-support/perf_correctness_bench.py`。

【实证】Raw log 字段 / 单位一致：两边都报 TTFT (s/ms) + TPOT (ms/tok) + decode_throughput (tok/s) + total_latency (s)；内部自洽性双向通过：
- baseline 1: TTFT 0.110 + (416-1)·5.451 ms / 1000 = 2.372 s ≈ reported total 2.373 s ✓
- nopad/baseline 2: TTFT 0.970 + (215-1)·14.6 ms / 1000 = 4.094 s ≈ reported total 4.096 s ✓

→ TTFT 在两边都是真 prefill 指标，**不是 ITL/TPOT 的 mislabel**。

### 6.3 Baseline 1 EXCLUDED 决策

| 对比对 | apples-to-apples? | 处理 |
|---|---|---|
| Padding baseline 1 (PERF_REPORT.md) ↔ nopad | **NO**（不同 script + 不同 prompt + 9× TTFT 差距）| **EXCLUDED 出主对比**，仅留作参考 |
| Padding baseline 2 (L20) ↔ nopad | **YES**（同 script + 同 input ≈10213 + 同 model + 同 tp）| 主对比基础（C1 caveat 适用，详 §10）|

### 6.4 9× 差距的候选机制（推断未实证）

【推测未实证】候选机制（`MODEL_PERF_COMPARE.md:70-76`，**未派 sub-teammate 实证**）：
- (a) `perf_bench.py` 与 `perf_correctness_bench.py` warmup 实现不同（前者 `generate(max_tokens=4)`，后者实现可能不同）
- (b) 两 script 构造 prompt 内容不同（同样 ≈10240 tokens 但语义/分布不同 → ATOM scheduler / cudagraph capture overhead 不同）
- (c) ATOM `out["ttft"]` 在两个 script 调用 path 中是否含 RPC + serialization cost 不同
- (d) Run between OS / page-cache / VRAM warm 状态不同

→ 这些是推断，**未实证**。F-A 终极实证（§3.3）不依赖 baseline 1，绕过此 discrepancy 直接同 HEAD CK patch on/off 对比。

【实证】**未来重跑必要性 + 本 wave 不重跑成本判断**（verbatim 摘 `MODEL_PERF_COMPARE.md:76`）：

> 这些是推断，**未派 sub-teammate 实证**。如果未来要把 baseline 1 也纳入对比，**必须**用同一 bench script 在 post-patch state 重跑 padding case。本 wave 推荐**不重跑**（成本不划算 + baseline 2 已 sufficient）。

---

## 7. F-A 终极实证如何破 model-level wave init verdict 的 caveat

### 7.1 Wave init verdict 三大 caveat（来源 `MODEL_PERF_COMPARE.md:170-176`）

| Caveat | 原内容 |
|---|---|
| **C1** | L20 历史 baseline (aiter `315123ace`) 命名为 "padding hotfix" 未独立核实 |
| **C4** | nopad 是 3 runs median，L20 baseline 是 single run，统计不对称 |
| **C7** | aiter `315123ace` vs `f06cdcca5` 之间可能有 4 层 patch 之外的其它改动（dispatch / kernel tune / quant 等）|

### 7.2 F-A 全破依据

| Caveat | 本次实证结果 |
|---|---|
| **C1** | ✅ **破** — 本次直接在 `f06cdcca5` 上跑 padding (CK pre-patch)，无需引用 L20 历史 baseline |
| **C7** | ✅ **破** — 本次同 HEAD `f06cdcca5`，唯一变量 = CK patch on/off，aiter python 层完全一致 |
| **C4** | ✅ **破** — 本次 padding 也是 3 runs median，与 nopad 完全对称 (run-to-run spread ≤1%) |

### 7.3 Verdict 升级

【实证】**PERF_NEUTRAL（HIGH confidence）** — 4 层 joint patch 在 stepfun-Flash-FP8 tp=4 model-level perf 上**没有可测量的 perf 影响**。

Confidence 拉满依据（`WAVE_CLOSE.md:32-36`）：
- 同 aiter HEAD ✅ / 同 cmd / prompt / tp / model ✅
- 同 measurement (3 runs median) ✅ / 同 GPU 配置 (4×gfx942) ✅
- 唯一变量 = CK 4 层 patch 的 working-tree 状态 ✅
- Dispatch path 实证（per_1x128 .so 重建）✅
- CORRECTNESS 3/3 PASS ✅

---

## 8. 3 个 incident 透明度记录（model-level F-A 期间）

> 【实证】verbatim 不删 — `PERF_PAD_CURRENT_MODEL_RESULT.md:184-187`

### Incident #1 — stash 顺序混淆误 pop teammate-1 历史 WIP

> **stash 顺序混淆**：误 pop `stash@{1}` (teammate-1 历史 WIP)，错误带入 `aiter/jit/core.py` + `aiter/configs/profile_fmoe.csv` → 立即 `git checkout aiter/jit/core.py` + `rm profile_fmoe.csv` 清理 → pop 真正的 `f-a-stash-aiter-cache-py`。**最终未污染 wave state**。

### Incident #2 — Run 1 first fail (`reshape_and_cache` type guard) → cache.py develop=True 误 stash

> **Run 1 第一次 fail (`reshape_and_cache` type guard)**：根因 = stash 掉了 `cache.py` 的 `develop=True` 标记 → ATOM tensor 类型 guard 抛错。**修复 = restore cache.py 但保持 CK patch stashed**（cache.py 不属于 4 层 patch）。

错误信息：`TypeError: reshape_and_cache: key needs to be aiter_tensor_t but got torch.Tensor`。

### Incident #3 — Run 1 second fail (LLVM disk full) → /tmp/comgr-* 累积

> **Run 1 第二次 fail (`LLVM ERROR: No space left on device`)**：根因 = `/tmp/comgr-*` 累积 3.3 GB JIT 中间文件 + `/tmp/torchinductor_root` 2.4 GB → 清理后立即 success。Disk 实际使用 81%（1.3 TB free），inode 健康，问题是个别 build 临时目录碎片。

修复：`rm -rf /tmp/comgr-* /tmp/aiter_* /tmp/torchinductor_root`。

### 另一个观察 — `.so` mtime 混合（非 fail，incident-class）

> **`.so` mtime 混合 pre-/post-patch**：发现现存 `.so` 是不同时间编译的（Apr 28 swiglustep / May 14 silu）→ 备份后全删 → 跑后用备份 restore。Md5sum 完全匹配。

### Patch state restore 验证（`PERF_PAD_CURRENT_MODEL_RESULT.md:114-125`）

```
aiter HEAD       = f06cdcca5 (post-patch) ✅
aiter cache.py   = 5x develop=True (ATOM 兼容) ✅
CK HEAD          = defd7ad29 ✅
CK working tree  = 105+/15- in gridwise_moe_gemm_blockscale.hpp ✅
.so files        = 全部 md5sum 匹配 stash 前备份 ✅
残留 CK stash@{0} = teammate-1 历史 WIP（无关，未触碰）
```

→ **Patch state = restored**，后续 wave 环境无破坏。

---

## 9. Lessons & Promotes（本 atomic story 衍生）

| # | Candidate | 措辞 | 主仓位 |
|---|---|---|---|
| 9.1 | Joint patch 必要性 — 跨层耦合 mismatch 无单点修 | 【实证】 | §2 + §4.4 |
| 9.2 | Kernel-level perf 收益不能直接外推 e2e（本 atomic story 实测 ~20× amortize）| 【实证】 + 【通用教训】 | §5 |
| 9.3 | F-A 同 HEAD 唯一变量 stash/pop 是破 historical baseline caveat 的高 ROI 设计 | 【通用教训】 | §7 |
| 9.4 | correctness PASS 不构成 wave 完整 PASS，必须 + kernel + model 三层 perf 实证 | 【通用教训】 | §2.3 |
| 9.5 | A2 production patch (aiter `fused_moe.py` `fc_scale_blkn` dispatch-aware) deferred 是 atomic story 已知边界 | 【实证】 | §10 caveat C3 |
| 9.6 | hpp Run + Run_2Lds 双 codegen 段必须双 apply（4 形态 × 2 = 8 hunk）— 单段 apply 触发部分 path 漏修 | 【实证】 | §2.1 + §2.2 |

---

## 10. Caveat 节（verbatim 保留 — §22 反模式防御）

> **【红线】本 atomic story production verdict 「可放心 merge」仅适用于：stepfun-Flash-FP8 tp=4 + MI308X (gfx942) + NPerBlock=64 path + verify-script L4 caller patch 形态**；任何更宽 deployment context（其它 model / 其它 NPerBlock / 其它 hardware / production aiter `fused_moe.py` 调用栈）必须明示 **未验证**。

### 10.1 Joint-fix wave 7 条 critical caveat（verbatim 摘 `m1quad_joint_fix_wave/WAVE_CLOSE.md §3`）

**C1. H_alt_1 仍是 partial root cause（不升格为 complete）— 重点 §22 自检**
- 来源：`m1ppp_padding_explanation_wave/PADDING_EXPLAINED.md` §3 节末
- 原措辞：**"sanity coverage 仅 83%（sanity 486 / 实测 584），剩余 17% 残差源未追查"**
- 本 atomic story 实证：**单点 fixture (SEED=0xc0de TOKEN=32 inter_dim=320) bit-exact，不等于全谱证伪 H_alt_1 partial 性质**
- IMPL_VERIFY_REPORT_v2.md §6 中 "17% 残差完全消除 → (a)" 措辞 **过于强**；正确表述应是 "在本 fixture 单点上 17% 残差不可见，是否全谱消除需更多 SEED/TOKEN/inter_dim 实证"
- ✅ Action: 不 promote 任何 "H_alt_1 = complete root cause" 表述到上游 doc
- **本 atomic story NOTE**: 即便 P1 closure 把覆盖扩到 192/320/448 三点 NPerBlock=64，**仍未覆盖 NPerBlock=32/96 + 多 SEED + 多 TOKEN**；synth 写 project summary 时不可由"3 点 PASS"升格为"complete root cause 已证伪"

**C2. NPerBlock=64 三点覆盖 ≠ NPerBlock=32/96 等其他路径**
- 本 atomic story 仅覆盖 (192/320/448) 都 NPerBlock=64
- NPerBlock=32 / NPerBlock=96 / NPerBlock 其他切片 **未测**

**C3. A2 production patch 未实证**
- 本 atomic story 仅修了 verify fixture (`scripts/correctness_compare_v2.py`)
- aiter `fused_moe.py` 内 `fc_scale_blkn` 的 production dispatch path **未改 / 未验证**
- 真实 vLLM serve / production fmoe 调用栈是否走 `ck_moe_stage1` 直接路径 = 未验证

**C4. aiter HEAD 漂移**
- TEAM_CONFIG.md 标 `315123ace`，实际 HEAD `f06cdcca5`（**ancestor**, 109 commits 老）
- patches 是对当前 HEAD source 的实测，对 `315123ace` 的可移植性 **未验证**

**C5. CK HEAD + JIT cache 行为**
- CK module rebuild 仅做 `rm -rf module_moe_ck2stages_*`，未做完整 `aiter setup.py develop`
- 是否所有 dependent module 都 stale-rebuild = 未严格验证

**C6. Verify path = 自定义 ck_moe_stage1 调用，非 production fmoe 调用栈**
- `correctness_compare_v2.py` 直调 `ck_moe_stage1`（CK 2-stage 入口）
- vLLM serve 走 `aiter.fused_moe()` 高层 → 是否 route 到同一 CK kernel = 部分路径未实证

**C7. Reviewer raise 的 #22 升格风险（C1 加强版）**
- IMPL_VERIFY_REPORT_v2.md §6 "17% 残差完全消除 → (a)" 已触发 #22 反模式（reasoned-but-unverified inference 升格为事实）
- 必须在 WAVE_CLOSE 显式标记 = "**单点 fixture 实证**"，不能升格为 "**完全证伪 H_alt_1 partial**"

### 10.2 Kernel-level perf wave 14 条 caveat（verbatim 摘 `m1quad_perf_compare_wave/PERF_COMPARE.md §5`）

**继承 caveat（来自上 wave m1quad_joint_fix WAVE_CLOSE）**：
- **C1.** 单点 fixture (TOKEN=32) — 大 batch 行为可能不同（padding 在大 batch 可能更好 amortize 单 kernel cost）
- **C2.** 仅测 inter_dim=320 vs 384 一对 — 其他 inter_dim (192/448/...) kernel 性能特征未测；cross-config 不可直接外推
- **C3.** NPerBlock=128 instance 命中按 dispatch 规则 `inter_dim % 128 == 0` 推断（t-perf-pad caveat verbatim）— 未 ROCm tracing 实证 kernel binary 选择
- **C4.** 4 层 patch 仅修 NPerBlock=64 path code，对 NPerBlock=128 path 无影响 — 比较是公平的
- **C5.** kernel-level perf ≠ end-to-end perf。F6（端到端 +1.99% padding cost）与本 wave kernel-level 数值不可直接比较
- **C6.** 计算量差异：320 vs 384 = inter_dim 实际 GEMM work load padding 多 20%（384/320=1.20）；padding 增加无效 compute 但用更高效 kernel instance（NPerBlock=128 通常在 large-N 上 hardware utilization 更优）
- **C7.** 真实 production load (vLLM serve) 调用栈 / batch size / TOKEN 分布与本 wave fixture 不同 — 推外推到 production 需谨慎

**从 t-perf-pad progress verbatim 提取**：
- **C8.** **(t-perf-pad verbatim)** "未直接实证 NPerBlock=128 instance 命中（仅按 inter_dim % 128 == 0 dispatch 规则推断）；如需强证据建议 t-synth 阶段加 `AITER_LOG_MORE=1` 或 `HIPCC_VERBOSE` env 重跑一遍。" — t-synth 未做此追加实证（保持原 caveat）
- **C9.** **(t-perf-pad verbatim)** "每 run 第 1 个 measurement iter 都偏高（98 / 95 / 93 us），疑似 launch overhead 残留，但 median/p50 不受影响。" — median 已过滤，但 p95 含残留
- **C10.** **(t-perf-pad verbatim)** "本测量仅 stage1 GEMM kernel；不含 stage2、moe_sorting overhead。" — 同样适用 nopad

**从 t-perf-nopad progress verbatim 提取**：
- **C11.** **(t-perf-nopad verbatim)** "timing 含 host-side launch overhead（cuda.Event 测的是 launch→complete，而非 pure GPU kernel time）；如 t-perf-pad 用同方法测，delta 数值仍可信（同 overhead 抵消）" — pad nopad 同方法 → delta 中 overhead 抵消 ✓
- **C12.** **(t-perf-nopad verbatim)** "dispatch-aware quant 在 CK lookup table 内的精确 instance（256x64x64x128 vs 其他）未 grep 确认；不影响 wall-time 数值"
- **C13.** **(t-perf-nopad verbatim)** "seed=0xc0de run 1 出现 2 个 outlier（>190us），但 median 不受影响，且 run 2/3 重现稳定" — outlier 来自 GPU 之外干扰
- **C14.** **(t-perf-nopad verbatim)** "未跑 ncu / rocprof tracing 拿 GPU-side 单 kernel time（torch.cuda.Event 含 launch overhead，理论上比纯 GPU kernel time 略大；但 launch overhead 通常 <5us，对 65us median 影响 <8%）"

### 10.3 Model-level perf wave caveat（verbatim 沿用上游 `MODEL_PERF_COMPARE.md §5.3` C1-C7 全 7 条编号 + 措辞 + 标 status）

> §7.1 列「F-A 全破 C1 / C4 / C7」对应上游同一编号体系；本节 verbatim 收全 7 条防止编号 collision。

- **C1** [**已破 by F-A**] **L20 命名未独立实证 / 未实证假设**：L20 (`/workspace/perf_logs/L20/stepfun_fp8_tp4.log`, aiter `315123ace`) 是否严格对应"padding hotfix" 标签 — 由 t-perf-nopad 在 caveat #1 提出未确认 + t-baseline-recon 找到的是另一份 baseline (perf_tp_eval/PERF_REPORT.md)。本 t-synth **未** Read L20 raw log 也未在 project_summary 索引中独立核实命名映射。verdict PERF_NEUTRAL **建立在 "L20 = padding hotfix" 这条未实证假设上**。如假设破，主对比基础需重新评估。
  - **F-A 破的依据**（§7.2）：本次直接在 `f06cdcca5` 上跑 padding (CK pre-patch)，无需引用 L20 历史 baseline → C1 不再是 verdict 的 critical assumption

- **C2** [**仍 open**] **baseline 1 EXCLUDED 是基于 cross-check 推断，未实证**：§6 论证 baseline 1 vs nopad 9× TTFT 差距由 script 方法学主导（不同 perf_bench.py vs perf_correctness_bench.py + 不同 prompt + warmup 差异），但**未派 sub-teammate** 用 perf_correctness_bench.py 在同 aiter `0f8164017` HEAD 重跑 padding case 来独立验证这条推断。如未来要纳入 baseline 1，必须实证。

- **C3** [**仍 open**] **script 方法学差异候选机制 (a)/(b)/(c)/(d) 全是推断**：§6.4 列的 4 条机制 (warmup / prompt 内容 / RPC overhead / OS cache) 都没 Read 两 script 源码做实证对比，仅凭 raw log surface 推测。

- **C4** [**已破 by F-A**] **verdict 仅 1 个 apples-to-apples baseline data point**：L20 是单一 stable run（n_in 10213, "stable" 标准未在本 teammate 处独立核实是否同 nopad 取 median 协议）；nopad 有 3 runs median。统计 confidence 不对称。如要严格统计 verdict，padding 也应 3 runs。
  - **F-A 破的依据**（§7.2）：本次 padding 也是 3 runs median，与 nopad 完全对称（run-to-run spread ≤1%）→ C4 统计不对称问题已消除

- **C5** [**仍 open**] **single prompt / concurrency=1 / single ISL=10213**：verdict PERF NEUTRAL **不**外推到：(a) 多 prompt 并发 / batched serving；(b) 不同 ISL (短 input 1024 / 长 input 32768) 下 prefill TTFT 比例可能改变 kernel relative weight；(c) decode-heavy (大 OSL ≥ 4096) 场景；(d) 不同 sampling (top-k/top-p/non-greedy)

- **C6** [**仍 open**] **kernel-level F2 vs model-level NEUTRAL 反差未深 attribute**：F2 实证 nopad kernel 快 15.80%，但 model-level 这 15.80% 完全消失。是 F3 "e2e overhead amortize" 的另一证据点，但**未在本 wave attribute 到具体 op**（比如 attention dominate / kv copy dominate / sched dominate？）— 如要进一步优化 nopad path 需要 nsys trace 拆解。

- **C7** [**已破 by F-A**] **aiter HEAD 漂移 nopad vs L20**：L20 = `315123ace`，nopad = `f06cdcca5` + 4-layer joint patch。两者 aiter HEAD 不同。L20 那个版本的 aiter 是否含其它非 joint-patch 改动 (如 dispatch / kernel tune / quant) 影响 padding path 的数值，**未实证排除**。如有非 joint-patch 改动也影响 padding path，主对比的 1.53% delta 归因可能不纯。
  - **F-A 破的依据**（§7.2）：本次同 HEAD `f06cdcca5`，唯一变量 = CK patch on/off，aiter python 层完全一致 → 跨 HEAD 漂移问题已消除

### 10.4 §22 防升格红线（lead 显式声明）

【实证】**禁措辞**（§22 反模式防御）：
- ❌ 禁说 "4 层联合修证伪 H_alt_1 partial 假设" — 应说 "在 NPerBlock=64 / inter_dim ∈ {192, 320, 448} 三点单 SEED fixture 下 bit-exact，未在更广谱（NPerBlock=32/96 / 多 SEED / 多 TOKEN / production e2e）证伪"
- ❌ 禁说 "production stepfun 推理已修复" — 应说 "verify fixture path 已修复；production aiter `fused_moe.py` `fc_scale_blkn` dispatch-aware 改造未做（A2 deferred）；vLLM serve 是否走同一 CK kernel 未验证"
- ❌ 禁 strip "未实证" / "未验证" / "partial" / "推断" 等 caveat 措辞
- ✅ 必须保留 "单点/三点 fixture 实证" vs "全谱证伪" 区分
- ✅ 必须明示 105+/15- patch 范围与 hpp 两段 codegen 双 apply 事实

【实证】**Hardware-axis 红线**：本 atomic story 实测 hardware = **MI308X (gfx942)**；**不引用 gfx950 entry 数据 / 不与 gfx950 entry (entry 16) 做 cross-link 推断**（防 wave2 跨硬件代际 cross-link 反模式 — 本 wave TEAM_CONFIG 已显式声明）。

### 10.5 GPA 5 维 reviewer 评分（来源：`m1quad_joint_fix_wave/WAVE_CLOSE.md:75-83`）

| 维度 | 分数 |
|---|---|
| Goal | 5/5 |
| Logical | 4/5 |
| Execution | 4/5 |
| Plan Quality | 3/5 |
| Plan Adherence | 4/5 |

---

## 11. Production Verdict + 适用边界

### 11.1 Production verdict

【实证】**4 层 joint patch 可放心 merge**，限定 deployment context：
- **Model**: stepfun-ai/Step-3.5-Flash-FP8（snapshot SHA `6eebda59dd87ca5729648ec7cfed0becfceb273e`）
- **TP**: 4
- **Hardware**: MI308X (gfx942) / 4 卡组合
- **Path**: NPerBlock=64 dispatch path (`inter_dim % 128 != 0 && inter_dim % 64 == 0`)
- **Caller patch 形态**: verify-script style L4（caller `w1_quant_blk_n` dispatch-aware）

合理性依据：
- correctness 修复显著（pre-patch max_err 0.6956 / 69.6% 元素错 → post-patch 0.0 bit-exact）
- model-level e2e 不引入退化（PERF_NEUTRAL HIGH confidence，3 metric delta 全 ≤0.81%）
- kernel-level 不引入新 perf 退化反而更快（15.80%）
- 3 incident 全部修复 + state 完整 restore，无环境破坏
- patch 范围明确（CK 1 文件 105+/15- + caller 5 行）易 review

### 11.2 适用边界（未验证 / Out-of-scope）

【推测未实证】**以下 context 必须 follow-up wave 验证后才能扩展 production**：
- 其它 model（不只 stepfun-Flash-FP8）
- 其它 TP（tp=2 / tp=8）
- 其它 NPerBlock path（NPerBlock=32 / 96 / 其它切片）
- 其它 hardware（gfx950 / 其它 ROCm 平台）
- 大 batch（TOKEN > 32 的工况）
- 多 SEED 全谱（本 wave 单 SEED 0xc0de / 0xbeef / 0xdead 等少数 seed）
- vLLM serve 真实 production 调用栈（C6）
- aiter `fused_moe.py` `fc_scale_blkn` production dispatch path 改造（A2 deferred，C3）

### 11.3 Follow-up wave 建议

【推测未实证】未来 wave 推荐覆盖（按 ROI 排序）：
1. **A2 production patch**：把 verify-script L4 caller patch 等价化到 `aiter/fused_moe.py` `fc_scale_blkn` 内 + 验证 vLLM serve 是否走同一 CK kernel
2. **多 SEED + 多 TOKEN 全谱实证**：在 NPerBlock=64 三点上扩 SEED 至 ≥10 个 / TOKEN 至 ≥{16, 32, 64, 128}，证伪/证实 H_alt_1 全谱
3. **NPerBlock=32 / 96 path 测试**：是否同 mechanism / 同 4 层 patch 形态可复用 / 是否需新 patch
4. **kernel-level perf 在 NPerBlock=64 nopad path .so 命名实地 grep + ROCm tracing**：补 C5/C8/C12 实证
5. **multi-prompt + tp≥4 工况实测**：（关联：上游 aiter `apply_act_and_mul` shape bug，详见 entry 20 §4 救命级 #2 — 跑前必须先 sanity check bug 是否已 fix）

---

## 12. Appendix

### 12.1 物料 path 索引

| 类型 | 路径 |
|---|---|
| 本 entry 整合 wave artifacts | `/home/junlin12/m1quad_project_summary_wave/` |
| Joint-fix correctness wave | `/home/junlin12/m1quad_joint_fix_wave/` |
| Kernel-level perf wave | `/home/junlin12/m1quad_perf_compare_wave/` |
| Model-level perf wave | `/home/junlin12/m1quad_model_perf_wave/` |
| 上游 partial root cause wave | `/home/junlin12/m1ppp_padding_explanation_wave/` |
| Joint-fix WAVE_CLOSE | `/home/junlin12/m1quad_joint_fix_wave/WAVE_CLOSE.md` |
| Joint-fix IMPL_VERIFY_REPORT_v2 | `/home/junlin12/m1quad_joint_fix_wave/IMPL_VERIFY_REPORT_v2.md` |
| Joint-fix CROSS_CONFIG_VERIFY | `/home/junlin12/m1quad_joint_fix_wave/CROSS_CONFIG_VERIFY.md` |
| Joint-fix patches | `/home/junlin12/m1quad_joint_fix_wave/patches/applied_ck.patch` |
| Kernel-level PERF_COMPARE | `/home/junlin12/m1quad_perf_compare_wave/PERF_COMPARE.md` |
| Kernel-level PERF_PAD/NOPAD_RESULT | `/home/junlin12/m1quad_perf_compare_wave/PERF_{PAD,NOPAD}_RESULT.md` |
| Kernel-level scripts | `/home/junlin12/m1quad_perf_compare_wave/scripts/perf_kernel_{pad,nopad}.py` |
| Model-level WAVE_CLOSE | `/home/junlin12/m1quad_model_perf_wave/WAVE_CLOSE.md` |
| Model-level F-A 终极实证 | `/home/junlin12/m1quad_model_perf_wave/PERF_PAD_CURRENT_MODEL_RESULT.md` |
| Model-level NOPAD result | `/home/junlin12/m1quad_model_perf_wave/PERF_NOPAD_MODEL_RESULT.md` |
| Model-level synth + caveat 全集 | `/home/junlin12/m1quad_model_perf_wave/MODEL_PERF_COMPARE.md` |
| Model-level baseline 9× discrepancy | `/home/junlin12/m1quad_model_perf_wave/BASELINE_RECON.md` |
| CK 实地 patch 文件 | `/workspace/aiter/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm_blockscale.hpp` |
| Verify fixture script | `/home/junlin12/m1quad_joint_fix_wave/scripts/correctness_compare_v2.py` |
| Bench script (e2e) | `/tmp/project_summary/step35-flash-support/perf_correctness_bench.py` (与本 project_summary 仓 `details/scripts/perf_correctness_bench.py` 同款) |

### 12.2 三仓 SHA 终态

- **aiter HEAD**: `f06cdcca5`（post-patch state, detached from `main`；TEAM_CONFIG 标 `315123ace` 是 ancestor 109 commits 老，详 caveat C4）
- **CK submodule HEAD**: `defd7ad29`（+ 4 层 working-tree patch: 105 insertions / 15 deletions in `gridwise_moe_gemm_blockscale.hpp`）
- **ATOM commit**: unknown（历史 wave 也未实证）

### 12.3 Cross-link

#### 入向 (上游)
- Partial root cause 起点：`/home/junlin12/m1ppp_padding_explanation_wave/PADDING_EXPLAINED.md §3`
- Candidate 方案 verify wave：`/home/junlin12/m1ppp_ck_bscale_fix_verify_wave/M1PPP_FIX_VERIFY_RESULTS.md`（candidate A/B FAIL → joint 必要性实证）
- 4 个 wave artifacts（本 entry 三 wave 整合源）：详 §12.1

#### 反向 (本 entry → 仓内已有 entry)
- ⚠️ **不与 entry 16 (`16_perf_gfx950_verified`) 做反向 cross-link**：entry 16 = gfx950，本 entry = gfx942 (MI308X)，跨硬件代际不应 cross-link（防 wave2 「hardware axis 错配」反模式 — entry 20 README 已 commit `126a3b4` clarify）。
- **entry 20 (`20_fp8_fmoe_tuning_wave2`) 关联但不 cross-link**：entry 20 探索 OPT-1 (fmoe csv tuning) axis 在 stepfun-Flash-FP8 上证伪；本 entry 探索 NPerBlock=64 path 4 层 joint patch axis；两者 axis 正交，**仅在 §11.3 follow-up §5「multi-prompt + tp≥4 工况」处提及关联 caveat**（上游 aiter `apply_act_and_mul` shape bug，entry 20 §4 救命级 #2 已记录）。
- **entry 18 (`18_fp8_tp8_root_cause_and_fix`)**：FP8 tp=8 双层 fix，与本 entry tp=4 路径不同；不直接 cross-link。

### 12.4 命令速查

```bash
# Verify CK patch state
cd /workspace/aiter/3rdparty/composable_kernel
git rev-parse HEAD
# defd7ad29...
git diff --stat include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm_blockscale.hpp
# 1 file changed, 105 insertions(+), 15 deletions(-)

# Verify aiter HEAD
cd /workspace/aiter
git rev-parse HEAD
# f06cdcca5...

# Reproduce model-level F-A (after applying CK patch + cache.py develop=True)
cd /workspace/aiter
CUDA_VISIBLE_DEVICES=0,1,2,3 AITER_LOG_LEVEL=WARNING HF_HOME=/mnt/nvme0/arliu/hf_cache \
  python /tmp/project_summary/step35-flash-support/perf_correctness_bench.py \
  --model /mnt/nvme0/arliu/hf_cache/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/6eebda59dd87ca5729648ec7cfed0becfceb273e \
  --tensor-parallel-size 4 --input-tokens 10240 --output-tokens 1024 \
  --runs 2 --measure-method A --max-model-len 11264

# Reproduce kernel-level perf
cd /home/junlin12/m1quad_perf_compare_wave
python3 scripts/perf_kernel_pad.py        # 3 seeds, padding
python3 scripts/perf_kernel_nopad.py 0xc0de
python3 scripts/perf_kernel_nopad.py 0xbeef
python3 scripts/perf_kernel_nopad.py 0xdead

# Reproduce correctness
cd /home/junlin12/m1quad_joint_fix_wave
python3 scripts/correctness_compare_v2.py --inter-dim 320  # core PASS
python3 scripts/correctness_compare_v2.py --inter-dim 384  # control no regression
python3 scripts/correctness_compare_v2.py --inter-dim 192  # cross-config
python3 scripts/correctness_compare_v2.py --inter-dim 448  # cross-config
```

---

> **wave 关闭。** 本 doc 是 m1quad atomic story（joint_fix + perf_compare + model_perf 三 wave）的项目总结主报告（PASS 显式声明 + 4 层 patch verbatim + correctness/kernel/model 三层数据 + 14 + 7 caveat verbatim + 3 incident 透明 + §22 反模式防御 + hardware axis (gfx942) 显式红线）。Production rollout 严格限定 §11.1 deployment context；扩展边界请走 §11.3 follow-up wave。
