# 修复方案 #401 — H1 验证：临时强制 run_1stage=False

## 背景
- gfx942 dirty patch 强制 `run_1stage = False`，使 per_1x128 FP8 走 CK 2-stage（而不是 ASM `fmoe_g1u1`）。
- 本任务调查 gfx950 aiter（commit 0f8164017）当前对应位置的实际代码与控制流，为 H1 假设（"gfx950 默认走 1-stage ASM `fmoe_fp8_blockscale_g1u1` 是 prefill 较慢的根因"）准备一个临时验证 patch。
- **此为 verification-only patch，验证后必须还原 (#405)，不 commit**。

---

## 文件
`/home/hanchang/junlin12_repos/aiter/aiter/fused_moe.py`

## 行号
L881–L889（`run_1stage` 在 q_type 分支上的赋值；处于 `cfg is None or AITER_BYPASS_TUNE_CONFIG` 分支下，外层条件 L880）

## 改动前（原文，L881–L889，不省略）

```python
            if q_type == QuantType.per_1x128:
                # for fp8 blockscale, ck has better performance so disable assembly kernel
                run_1stage = token > 32 and (inter_dim % 256 == 0)
            elif q_type == QuantType.per_Token and q_dtype_w == dtypes.i8:
                run_1stage = token > 32
            elif q_type == QuantType.per_Token and q_dtype_w == dtypes.fp8:
                run_1stage = token > 16 or inter_dim % 128 != 0
            elif q_type != QuantType.per_1x32:
                run_1stage = token < 256
```

## 改动后（强制 per_1x128 走 2-stage）

```python
            if q_type == QuantType.per_1x128:
                # [TEMP H1 VERIFICATION PATCH — DO NOT COMMIT]
                # Force CK 2-stage for per_1x128 FP8 blockscale on gfx950 to
                # match gfx942 dirty patch behavior. Original heuristic:
                #   run_1stage = token > 32 and (inter_dim % 256 == 0)
                run_1stage = False
            elif q_type == QuantType.per_Token and q_dtype_w == dtypes.i8:
                run_1stage = token > 32
            elif q_type == QuantType.per_Token and q_dtype_w == dtypes.fp8:
                run_1stage = token > 16 or inter_dim % 128 != 0
            elif q_type != QuantType.per_1x32:
                run_1stage = token < 256
```

---

## 理由（来自 fused_moe.py 实际代码）

### 1. `run_1stage` 控制流（L867–L932）
- L867 进入 `cfg is None or AITER_BYPASS_TUNE_CONFIG=1` 分支（即 tune cache 未命中走启发式）。
- L871 默认 `run_1stage = False`。
- L872–L880 仅当 `(activation, q_type, dtype, q_dtype_a, q_dtype_w, use_g1u1, doweight_stage1)` 元组在 `fused_moe_1stage_dict[get_gfx()]` 中存在，才会按 q_type 进入 L881–L889 的启发式赋值；否则保持 False。
- L924–L932（else 分支）：tune cache 命中时 `run_1stage = cfg.get("run_1stage", False)`。

### 2. True/False 分别走的 kernel 路径
- **`run_1stage = True`** → L951 `if run_1stage:` → 返回 `MOEMetadata(stage1=functools.partial(fused_moe_1stage, kernelName=kernelName1, ...))`。fused_moe.py L312 `if metadata.run_1stage: return metadata.stage1(...)` 直接调用 ASM 1-stage kernel。具体 kernel 由 `fused_moe_1stage_dict[get_gfx()]` 表项决定。
- **`run_1stage = False`** → L339 `return fused_moe_2stages(...)` → CK 2-stage 双 GEMM 路径（L1142 `def fused_moe_2stages`）。

### 3. **gfx950 vs gfx942 关键差异**（L570–L597）
- L582（gfx942）：`per_1x128, fp8, fp8 → aiter.fmoe_g1u1`（通用 FP8 ASM kernel）
- L590（gfx950）：`per_1x128, fp8, fp8 → aiter.fmoe_fp8_blockscale_g1u1`（**专用** FP8 blockscale ASM kernel）
- L591（gfx950 only）：`Gelu, per_1x128 → aiter.fmoe_fp8_blockscale_g1u1`

→ gfx950 上当 `run_1stage=True` 时调用的是 **`fmoe_fp8_blockscale_g1u1`**，与 gfx942 的 `fmoe_g1u1` 不同。这正是怀疑的性能瓶颈点：H1 验证此 ASM kernel 在 prefill 场景下是否慢于 CK 2-stage。

### 4. 当前 gfx950 默认行为（per_1x128 FP8）
- 元组 `(Silu, per_1x128, bf16, fp8, fp8, True, False)` 在 gfx950 dict 中存在（L590）。
- 因此 L883 启发式触发：`run_1stage = (token > 32) and (inter_dim % 256 == 0)`。
- Step-3.5-Flash 的 `inter_dim`（MoE intermediate per partition）已知 256/512 对齐（FP8 已满足 % 256 == 0），prefill 阶段 token >> 32。
- → **prefill 阶段 gfx950 默认 `run_1stage=True`，调用 `fmoe_fp8_blockscale_g1u1` ASM kernel**。
- decode 阶段单 token，`token > 32` 为 False，`run_1stage=False` 走 CK 2-stage（解释 decode TPOT 与 gfx942 接近）。

### 5. gfx942 dirty patch 上下文
- gfx942 注释（L882）："for fp8 blockscale, ck has better performance so disable assembly kernel"。该注释提示历史上即便在 gfx942 上，per_1x128 的 ASM kernel 性能也不如 CK 2-stage。dirty patch 进一步强制完全禁用 1-stage（无视 `inter_dim % 256` 条件）。
- gfx950 的 `fmoe_fp8_blockscale_g1u1` 是新增的、可能未充分调优的 kernel，更有理由怀疑 prefill 性能。

---

## 回归测试计划（执行人为 #402–#406）

1. **加 patch**（#402，临时直接编辑 `/home/hanchang/junlin12_repos/aiter/aiter/fused_moe.py` L881–L889，注意不 commit、不 push）。
2. **重跑 perf_bench.py tp=2**（#403）：GPU 0,1，FP8。
3. **重跑 perf_bench.py tp=4**（#404）：GPU 0,1,2,3，FP8（VRAM 归零后）。
4. **对比指标**（vs 当前基线 perf_bench.py，#301/#302 实测）：
   - tp=2：基线 TTFT=388ms / TPOT=12.28ms
   - tp=4：基线 TTFT=241ms / TPOT=12.504ms
   - 期望（H1 成立）：TTFT 下降；TPOT 大致不变（decode 本来就走 2-stage，patch 不影响 decode）。
   - 若 H1 成立：H1 升级为根因，进入 long-term fix（修改 gfx950 dict 移除 per_1x128 表项，或改启发式条件）。
   - 若 H1 不成立（TTFT 无变化或更慢）：H1 排除，转 H2/H3/H4 验证。
5. **还原 patch**（#405，无论结果如何必须执行）→ `git diff` 应为空。
6. **更新 RESULTS.md**（#406）记录 H1 结论。

## 注意事项
- 此 patch 仅修改 L881–L889 中 `if q_type == QuantType.per_1x128:` 分支体两行（注释 + 一行赋值）。
- aiter 是 Python 库，无需重编译；但 ATOM 会缓存 MoE 配置，验证前需 `rm -rf /root/.cache/atom/*`（参考 MEMORY.md ATOM 开发规范）。
- 验证完成后必须 `git -C /home/hanchang/junlin12_repos/aiter status` 确认 working tree 干净。
- patch 临时性强，**严禁 commit / push**。
