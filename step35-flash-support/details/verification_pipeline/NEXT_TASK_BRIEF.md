# 新任务：FP8 tp=4 无 padding CK kernel 开发

## 背景与动机

### 为何当前需要 padding

Step-3.5-Flash 在 tp=4 切分下，每张卡分到的 inter_dim = 1280 / 4 = 320（FP8 模型）或 BF16 模型上表现为 320。CK 2-stage GEMM 在 gfx950 上可用的 tile 配置存在严格的 N 对齐约束：

- inter_dim ≤ 192：选 V1 路径，NPerBlock=64，要求 inter_dim % 64 == 0
- inter_dim  > 192：被强制选 V3 kernel（`moe_ck2stages_gemm2_256x128x128x64_1x4_..._v3`），NPerBlock=128，要求 inter_dim % 128 == 0

inter_dim=320 既 >192（落入 V3 强制分支），又 320 % 128 = 64 ≠ 0，因此不存在匹配的 CK 实例，运行时报：
```
wrong! device_gemm with the specified compilation parameters does not support this GEMM problem
```

实证来源：`/home/hanchang/project_fp8_tp4/verification_pipeline/results/v01_exp2_inter.md`（V01 Exp2 矩阵：inter_dim=320 直接 ERROR；192 / 256 / 384 / 640 PASS）。

### 当前 workaround 的代价

ATOM 在 `process_weights_after_loading` 阶段对权重做 zero-pad：320 → 384（tp=4），160 → 192（tp=8）。代价：

- **权重显存增加**：tp=4 时 w13 与 w2 的 inter 维多 64/320 = 20% 容量（zero rows）。
- **权重加载时间增加**：复制 + zero-tensor 分配，发生在每次 server 启动的 `process_weights_after_loading`。
- **dequant / 计算 FLOPs 也按 padded 形状执行**：尽管 zero rows 在数值上是 no-op，硬件仍按 384 列做 GEMM（约 384/320 = 20% 算力浪费）。
- **scale tensor 切分复杂度**：FP8 路径下 Fix 3（commit `ccb64621`）使用 ceil 整除处理 N=1280 / tp=4 = 10 / 4 = 2.5 → 3 blocks，逻辑额外引入 narrow 截断；若 inter_dim 原生支持 tile 对齐，可消除这一不对称。

详见 `/home/hanchang/project_fp8_tp4/verification_pipeline/results/v04_tp_support.md`（padding 可视化矩阵）和 `/home/hanchang/project_fp8_tp4/verification_pipeline/results/v06_fp8_tp4.md`（FP8 tp=4 e2e baseline TTFT=86ms）。

---

## 当前实现（Baseline）

### ATOM 侧（padding 逻辑）

**文件**：`/home/hanchang/ATOM/atom/model_ops/moe.py`

- padding 位置：**L489-518**（在 `process_weights_after_loading` 中，紧随 `_maybe_pad_weight` 之后）
- align 计算：**L502** `align = 64 if inter_dim <= 192 else 128`
- inter_pad 计算：**L503** `inter_pad = (inter_dim + align - 1) // align * align`
- 权重 zero-pad（w13 / w2）：L504-518
- padding 规则实测：
  - tp=4：inter_dim=320 → 384（+64，align=128）
  - tp=8：inter_dim=160 → 192（+32，align=64）
  - tp=2：inter_dim=640 → 不需 padding（已 % 128 == 0）

FP8 路径相关 scale loader：`_load_w13` L2287-2328（ceil 在 L2305）、`_load_w2` L2330-2359（ceil 在 L2347），属 Fix 3，间接受 padding 形状影响。

### aiter 侧（V3 kernel dispatch）

**文件**：`/home/hanchang/aiter/aiter/fused_moe.py`

- block_m 选择：L891-898（默认 `get_block_size_M(token, topk, expert, inter_dim)`）
- V3 kernel 强制：**L900-910**
  ```python
  if not run_1stage and inter_dim > 192 and get_gfx() == "gfx950" \
          and q_type not in (QuantType.per_1x128, QuantType.per_1x32):
      block_m = 128
      if not is_shuffled and not kernelName2:
          kernelName2 = "moe_ck2stages_gemm2_256x128x128x64_1x4_TypeCast_v3_Nswizzle0_Quant0_MulRoutedWeight1_B16_B16_B16"
  ```
- 当前 V3 kernel：`moe_ck2stages_gemm2_256x128x128x64_1x4_..._v3`
  - tile 解读：BlockSize=256, MPerBlock=128, NPerBlock=128, KPerBlock=64, MWaves=1, NWaves=4
  - 之所以 N=128 要求 inter_dim 为 128 的倍数：CK tile loop 在 N 方向以 NPerBlock=128 步长展开，无 partial-tile fallback 路径。

注：blockscale 量化（`per_1x128` / `per_1x32`）走另一组 kernel（dispatch L881-889 的 `run_1stage` 分支），与 V3 强制无关；FP8 blockscale 在 tp=4 仍由 ATOM 侧 padding 保护到 inter=384。

### CK 侧（kernel 编译 / codegen）

- **CK codegen 路径**：`/home/hanchang/aiter/csrc/ck_gemm_moe_2stages_codegen/`
- 目录内容：
  - `gemm_moe_ck2stages.cu/.h` — 入口实例
  - `gemm_moe_ck2stages_common.cuh` — 通用模板
  - `gemm_moe_ck2stages_common.py` — kernel instance 表（含 tile 配置）
  - `gemm_moe_ck2stages_common_blockscale.cuh` — FP8 blockscale 路径
  - `gemm_moe_ck2stages_common_mxfp4.cuh`, `..._mxfp4_bns.cuh` — MXFP4 路径
  - `gemm_moe_tune.py`, `gen_instances.py` — codegen + 调优脚本
- 当前 GEMM1 stage instance 表（节选自 `gemm_moe_ck2stages_common.py`）：
  - 含 tile (BlockSize, MPerBlock, NPerBlock, KPerBlock, MWaves, NWaves, Stages)
  - 例：`(256, 128, 128, 64, 1, 4, 3)`、`(256, 256, 128, 64, 1, 4, 3)`、`(256, 64, 128, 128, 1, 4, 3)` 等
  - **NPerBlock 当前可见值集合：64, 128, 256**；无 320 / 40 / 80 等
- CK submodule：`/home/hanchang/aiter/3rdparty/composable_kernel/`（同时存在 `ck_helper/`）

---

## 新 kernel 目标

**目标**：新增一组支持 inter_dim=320（N=320）的 CK MoE 2-stage kernel 实例，使 FP8 tp=4 / BF16 tp=4 在 ATOM 侧无需 weight padding 即可正确运行，并保持或优于现有 e2e 性能。

**候选方案**（供新 session 探索；按实现成本由低到高排序）：

### 方案 A：新增 N=320 / 小 N tile 的 CK kernel 实例

- 在 `gemm_moe_ck2stages_common.py` 中添加 NPerBlock 取值，使 320 可整除：
  - 候选 NPerBlock：320（一次性覆盖整个 N 方向）、160（=320/2）、80、64（=320/5）、40
- 优点：无需修改 ATOM padding 逻辑；aiter dispatch 仅需新增分支选择新 kernel name
- 挑战：
  - CK tile 大小受 GPU 资源（VGPR/SGPR/LDS）和 wave 配置约束，并非任意 NPerBlock 都能编译
  - 需要在 `gen_instances.py` 中跑通编译并加入 build manifest
  - 性能未知：小 NPerBlock 可能导致 K-loop 占比上升，吞吐下降

### 方案 B：CK kernel 内部 padding（kernel-side masking）

- 修改 `gemm_moe_ck2stages_common.cuh` 让 V3 kernel 接受非对齐 N，在 tile loop 内部对 N >= inter_dim 的行做 masked-load（读 0）+ masked-store（不写）
- 优点：ATOM 与 aiter 上层接口完全不变；一次修改覆盖所有未对齐 inter_dim
- 挑战：
  - 需要修改 CK 模板（涉及 3rdparty/composable_kernel），与上游同步成本高
  - kernel 内部 masking 引入 predication 开销（约 5-10% perf hit 估计）
  - FP8 / BF16 / blockscale 三条路径都要分别验证

### 方案 C：调整 V3 tile 参数（NPerBlock=64，使 320=64×5）

- 在 aiter `fused_moe.py` L900-910 的 V3 强制分支中，针对 inter_dim % 128 != 0 但 inter_dim % 64 == 0 的情况，选用现有的 `(256, *, 64, *, ...)` tile（NPerBlock=64 已有 instance，例如 L130 `(256, 128, 64, 64, 1, 4, 1)`）
- 优点：实现最简单，零新增 CK 编译；可能仅修改 dispatch 表
- 挑战：
  - **关键前提**：必须先确认现有 NPerBlock=64 的 V3 instance 在 gfx950 上不重现 V01 Exp2 中描述的 V1 bug（"V1 kernel produces wrong results for inter_dim>192"）。若 NPerBlock=64 的 V3 路径同样出错，方案 C 不可行
  - inter_dim 较大时（如 tp=2 的 640），小 N tile 可能拖慢性能（需对比 tp=2 baseline TTFT=92ms）
  - 该方案不彻底解决 320 之外的其它非对齐 inter_dim

### 方案优先级建议

1. **先做方案 C 的可行性实验**（改 dispatch 1 行 + 跑 cos_sim 对比，<5 tool calls 可结论）
2. 若 C 不可行 → 走方案 A（新增专用 NPerBlock）
3. 方案 B 留作长期上游贡献方向

---

## 关键代码路径

| 层次 | 文件 | 关键位置 |
|------|------|---------|
| ATOM padding | `/home/hanchang/ATOM/atom/model_ops/moe.py` | L489-518（align 在 L502，inter_pad 在 L503，权重 zero-pad 在 L504-518） |
| ATOM FP8 scale loader（受 padding 影响） | `/home/hanchang/ATOM/atom/model_ops/moe.py` | `_load_w13` L2287-2328（ceil L2305）；`_load_w2` L2330-2359（ceil L2347） |
| fused_moe dispatch | `/home/hanchang/aiter/aiter/fused_moe.py` | L880-919（block_m 选择 + V3 强制）；核心 V3 强制 L900-910 |
| CK kernel instance 表 | `/home/hanchang/aiter/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py` | L33（NPerBlock 字段定义）、L126-187（GEMM1 stage instances） |
| CK 模板 | `/home/hanchang/aiter/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.cuh`，`..._common_blockscale.cuh` | tile loop / masking 修改入口（方案 B） |
| CK codegen 入口 | `/home/hanchang/aiter/csrc/ck_gemm_moe_2stages_codegen/gen_instances.py`，`gemm_moe_tune.py` | 新增实例后需 regenerate |
| CK submodule | `/home/hanchang/aiter/3rdparty/composable_kernel/`（另存在 `ck_helper/`） | 上游模板源 |

---

## 验证标准

新 kernel 开发后需通过：

1. **单元正确性**（V01 Exp2 扩展）：
   - inter_dim=320，M ∈ {32, 256, 1024}, model_dim=7168, E=16, topk=4, dtype ∈ {bf16, fp8 blockscale}
   - cos_sim ≥ 0.9999 vs torch reference
   - 复用脚本：参考 `/tmp/v01_exp2_inter.py` 与 `/tmp/v01_exp2b_fix.py`

2. **e2e 性能**（V06 Exp2 baseline 对比）：
   - FP8 tp=4 TTFT ≤ 86ms（不差于当前 padding 方案）
   - FP8 tp=4 TPOT ≤ 13ms
   - BF16 tp=4 TTFT ≤ 84ms（V04 Exp2 baseline 81.25ms ±20%）
   - 4/4 prompt 输出连贯，无 BOS-spam，无 gibberish

3. **权重未被 padding**：
   - 在 `process_weights_after_loading` 完成后，断言 `layer.w2_weight.shape[2] == 320`（tp=4 FP8）/ `== 160`（tp=8）
   - rocm-smi 显存占用应较 baseline 下降（FP8 tp=4 估算节省 ~20% MoE 权重显存）

4. **回归**：
   - tp=2（inter_dim=640）TTFT=92ms ±20%
   - tp=8（inter_dim=160）静态验证通过（GPU5 不可用，e2e 跳过）

---

## 环境约束

- 平台：8× MI350X（gfx950），ROCm，Python `/opt/venv`
- 运行 python 必须先 `cd /tmp &&`（否则 aiter 被识别为 namespace package）
- **GPU 5 禁用**（硬件异常 ~700ms/tensor），tp=4 用 GPU 0,1,2,3；tp=2 用 GPU 4,6
- 修改 ATOM/aiter 代码后清缓存：`rm -rf /root/.cache/atom/* /root/.cache/aiter/*`
- 修改 CK codegen / CK 模板后需 rebuild aiter（参考 `csrc/ck_gemm_moe_2stages_codegen/README.md`）
- git commit/push 必须从 `/home/hanchang/junlin12_repos/` 执行（author: Jun Lin <junlin12@amd.com>，SSH key: id_ed25519_junlin12）；**不得**从 `/home/hanchang/aiter` 或 `/home/hanchang/ATOM` 直接 push
- 运行模型：`stepfun-ai/Step-3.5-Flash`（BF16）/ `stepfun-ai/Step-3.5-Flash-FP8`，权重缓存于 `/root/.cache/huggingface/hub/`
- 启动前设置 `AITER_LOG_LEVEL=WARNING` 防 kernel log flooding；ATOM 侧 `ATOM_LOG_LEVEL=WARNING`

---

## 参考文档

- V01 inter_dim 边界矩阵：`/home/hanchang/project_fp8_tp4/verification_pipeline/results/v01_exp2_inter.md`
- V04 TP 支持验证（含 padding 可视化）：`/home/hanchang/project_fp8_tp4/verification_pipeline/results/v04_tp_support.md`
- V06 FP8 tp=4 e2e（含 baseline 性能）：`/home/hanchang/project_fp8_tp4/verification_pipeline/results/v06_fp8_tp4.md`
- 项目记忆：`/root/.claude/projects/-home-hanchang/memory/MEMORY.md`，`memory/fp8-work.md`，`memory/moe-kernels.md`，`memory/tp48-fixes.md`
