# teammate-5 progress — #401 调查

## 任务
读 `/home/hanchang/junlin12_repos/aiter/aiter/fused_moe.py`，确认 `run_1stage` 控制流，写 `proposed_fix_401.md`。

## 关键发现（基于代码）

### 1. `run_1stage` 实际控制逻辑
源文件：`/home/hanchang/junlin12_repos/aiter/aiter/fused_moe.py`

- **L867** 进入条件：`cfg is None or int(os.environ.get("AITER_BYPASS_TUNE_CONFIG", "0"))`（tune cache miss 或显式 bypass）。
- **L871** 默认 `run_1stage = False`。
- **L872–L880** 门控：仅当 `(activation, q_type, dtype, q_dtype_a, q_dtype_w, use_g1u1, doweight_stage1)` 元组存在于 `fused_moe_1stage_dict[get_gfx()]` 时，才进入 L881–L889 的 q_type 启发式。
- **L881–L889** 启发式赋值：
  - `per_1x128`：`run_1stage = token > 32 and (inter_dim % 256 == 0)`
  - `per_Token + i8`：`token > 32`
  - `per_Token + fp8`：`token > 16 or inter_dim % 128 != 0`
  - 其他（非 per_1x32）：`token < 256`
- **L924–L932**（cfg 命中分支）：`run_1stage = cfg.get("run_1stage", False)`。

### 2. True/False 分别走的 kernel 路径
- **True**：L951 → `MOEMetadata.stage1 = functools.partial(fused_moe_1stage, kernelName=kernelName1, ...)`；L312 实际调用。kernel 由 `fused_moe_1stage_dict[get_gfx()]` 表项决定。
- **False**：L339 `fused_moe_2stages(...)` → CK 2-stage 双 GEMM（L1142 定义）。

### 3. **gfx950 vs gfx942 ASM kernel 差异（最关键发现）**
`fused_moe_1stage_dict` 在 L570–L597：

| 配置 | gfx942（L582） | gfx950（L590） |
|------|----------------|----------------|
| `(Silu, per_1x128, bf16, fp8, fp8, True, False)` | `aiter.fmoe_g1u1` | `aiter.fmoe_fp8_blockscale_g1u1` |

gfx950 还独有 `(Gelu, per_1x128, ...)` → `aiter.fmoe_fp8_blockscale_g1u1`（L591）。

**gfx950 的 1-stage 路径走的是专用 `fmoe_fp8_blockscale_g1u1` ASM kernel**，与 gfx942 的通用 `fmoe_g1u1` 不同。这是 H1 假设的具体技术对应物。

### 4. gfx950 当前 per_1x128 FP8 默认路径
- 元组在 gfx950 dict 中存在 → 触发启发式 L883。
- prefill 时 `token > 32` 满足、`inter_dim % 256 == 0` 在 Step-3.5-Flash FP8 配置下满足 → **`run_1stage=True`，走 ASM `fmoe_fp8_blockscale_g1u1`**。
- decode 时单 token，`token > 32` 为 False → `run_1stage=False`，走 CK 2-stage（解释了为何 TPOT 与 gfx942 接近）。

### 5. gfx942 注释提示
L882 注释："for fp8 blockscale, ck has better performance so disable assembly kernel" — 即便在 gfx942 上，per_1x128 的 ASM kernel 历史性能也不如 CK 2-stage。dirty patch 进一步强制禁用 1-stage。

## proposed_fix_401.md 核心改动摘要
- 将 L883 `run_1stage = token > 32 and (inter_dim % 256 == 0)` 替换为 `run_1stage = False`，同时更新注释标明为 H1 临时验证 patch。
- 仅影响 per_1x128 分支；其他 q_type 启发式不变。
- 无需重编译 aiter；但需 `rm -rf /root/.cache/atom/*` 清 ATOM 缓存。

## 给 lead 的审批建议
**建议批准**，理由：
1. 改动最小化（一行赋值 + 注释），影响范围明确（仅 per_1x128 + cfg miss 分支）。
2. 与 gfx942 dirty patch 完全等价的实验对照，可干净检验 H1。
3. 风险低：CK 2-stage 路径是 gfx950 decode 已经在用的代码路径，正确性已被 #101/#102 PASS 验证。
4. 临时性强，#405 强制还原步骤已写明。

**注意事项**：
- 务必从 `/home/hanchang/junlin12_repos/aiter` 操作（即便不 push 也保持工作目录一致），按 MEMORY.md 铁则。
- 验证前后均执行 `git status` 确认。
- decode TPOT 预期不变（因 decode 本来就 `run_1stage=False`），主要观察 prefill TTFT 变化。

## 关联文件
- 代码：`/home/hanchang/junlin12_repos/aiter/aiter/fused_moe.py` L570–L597（dict）、L867–L932（控制流）、L312/L339/L951（dispatch）
- 输出：`/home/hanchang/project_summary/step35-flash-support/16_perf_gfx950_verified/proposed_fix_401.md`
