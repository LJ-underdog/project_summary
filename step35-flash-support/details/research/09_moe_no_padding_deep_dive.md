# 为什么 FP8 MoE Kernel 需要 Padding？如何去掉它？

> 本文档由 agent team 调研结论写成，所有结论均有代码行号或实验数据支撑。
> 数据来源：teammate-1 ~ teammate-9 的 progress 文件。
> **重要说明**：本文包含一处对之前结论的重大修正——"必须 per_1x64"在代码层面不成立。

---

## 第一层：最直观的问题

Step-3.5-Flash-FP8 在 **tp=4** 时，每张 GPU 负责的 MoE 中间维度：

```
moe_intermediate_size = 1280  （模型全局）
tp=4 → 每张 GPU 的 inter_dim = 1280 / 4 = 320
```

当前 CK MoE blockscale kernel 每次处理 N 方向时，以 128 列为一个 "tile"（`NPerBlock=128`）。

**320 ÷ 128 = 2.5**，不是整数。

Kernel 无法处理"半个 tile"，所以 ATOM 把 weight 从 320 列 padding 到 384 列（= 3 × 128），浪费约 **17%** 的计算量。

---

## 第二层：Kernel 为什么不能接受 N=320？

### CK kernel 的 N 对齐检查

**文件**：`composable_kernel/include/ck/tensor_operation/gpu/device/impl/device_moe_gemm_blockscale.hpp:448`

```cpp
if(arg.N % NPerBlock != 0 || arg.K % KPerBlock != 0)
{
    return false;   // IsSupportedArgument 返回 false，kernel 不执行
}
```

N=320，NPerBlock=128：`320 % 128 = 64 ≠ 0` → **kernel 直接拒绝**。

这是唯一的硬性入口卡点（`gridwise` 层的 CheckValidity 因为 B 是 Col layout 而跳过了 N 检查）。

### ATOM 的 padding 方案

**文件**：`ATOM/atom/model_ops/moe.py:1709-1749`（`_process_block_quant`）

```python
align = 64 if inter_dim <= 192 else 128   # L1721
inter_pad = (inter_dim + align - 1) // align * align
# inter_dim=320 → align=128 → inter_pad=384

# 把 weight 从 320 列 zero-pad 到 384 列
w13_new = torch.zeros(E, 2 * inter_pad, hidden, ...)  # L1727
w13_new[:, :inter_dim, :] = w13[:, :inter_dim, :]      # 复制真实数据
w13_new[:, inter_pad:inter_pad+inter_dim, :] = w13[:, inter_dim:, :]
```

Zero-padding 是数值安全的：FP8 的 0 × 任意 scale = 0，padding 区域对输出无影响。

---

## 第三层：Blockscale 量化的本质

### 什么是 per_1x128 blockscale？

Step-3.5-Flash-FP8 的量化方式（来自 checkpoint `config.json:313`）：

```json
"weight_block_size": [128, 128]
```

含义：weight 矩阵中每个 **[1行 × 128列]** 的小块共享一个 `scale_inv`：

```
weight[行 r, 列 c] 所在的 block：
  行 block = r // 1 = r
  列 block = c // 128

scale_inv 的 shape = (num_experts, ceil(N/128), ceil(K/128))
```

**实验验证**（teammate-9，layer 10 expert 0）：
```
weight shape: (288, 1280, 4096)  dtype=float8_e4m3fn
scale shape:  (288,   10,   32)  dtype=float32

scale_inv[0, 2, 0] = 1.907e-4
FP8_MAX × scale_inv = 448 × 1.907e-4 = 0.0854
→ 该 block 的 BF16 weight max ≈ 0.0854（合理的权重数值）
```

还原公式：`bf16_approx = fp8_value × scale_inv`

### Checkpoint 存储的是全局权重

**实验验证**（teammate-7，直接读 safetensors）：
```
model.layers.3.moe.gate_proj.weight           shape=(288, 1280, 4096)
model.layers.3.moe.gate_proj.weight_scale_inv shape=(288,   10,   32)
```

- Weight 是全局 1280 列（**未做 TP partition**）
- Scale 是全局 10 个 N-block（= ceil(1280/128)）
- TP partition 在 ATOM 加载时做：`_load_w13`（`ATOM/atom/model_ops/moe.py:2287-2310`）

**TP=4 rank 0 的 narrow 逻辑**（代码 L2305-2310）：
```python
# weight：1280列 → 每 rank 320列
load_shard_size = ceil(1280 / 4) = 320
# → rank 0 取 weight[:, 0:320, :]

# scale：10 blocks → 每 rank 3 blocks（ceil(10/4)=3）
load_shard_size = ceil(10 / 4) = 3
# → rank 0 取 scale[:, 0:3, :]
```

这意味着：
- `scale[2]` 对应全局列 **[256:384)**，是对 128 列数据算出的
- 但 rank 0 的 weight 只有列 **[256:320)**，即 64 列真实数据

**scale[2] 覆盖的范围比 rank 0 实际持有的 weight 多出 [320:384) 这 64 列。**

---

## 第四层：NPerBlock=128 + N=320 为什么必须 padding？

回到 kernel 的视角。当 ATOM 把 weight padding 到 384 列后：

```
N=384，NPerBlock=128 → ceil(384/128) = 3 个 N-tile
tile 0: 列 [0,128)     → 使用 scale[0]
tile 1: 列 [128,256)   → 使用 scale[1]
tile 2: 列 [256,384)   → 使用 scale[2]
```

Kernel 的 scale 索引公式（`gridwise_moe_gemm_blockscale.hpp:1418`）：

```cpp
make_multi_index(block_n_id * NPerBlock / ScaleBlockN, 0)
//              = block_n_id * 128 / 128
//              = block_n_id（整数除，无损）
```

Tile 2（列 [256:384]）使用 scale[2]，该 scale 基于全局 [256:384] 算出。

**Weight padding 的意义**：让 tile 2 的 weight 列 [320:384] 填零，使得 scale[2] 的量化误差不影响输出（FP8(0) × scale = 0）。

---

## 第五层：直觉上的"解法"——缩小 NPerBlock

如果把 NPerBlock 从 128 缩小到 **64**：

```
N=320，NPerBlock=64 → ceil(320/64) = 5 个 N-tile，320 % 64 = 0
```

N=320 可以被 64 整除！不需要 weight padding！

**但问题来了：ScaleBlockN 还是 128，scale 的粒度没变。**

如何把 5 个 64 列的 tile 映射到 3 个 128 列的 scale？

---

## 第六层：NPerBlock=64 + ScaleBlockN=128 的 scale 索引分析

### scale 索引公式（整数除法）

公式仍是（`gridwise_moe_gemm_blockscale.hpp:1418`）：

```cpp
block_n_id * NPerBlock / ScaleBlockN
= block_n_id * 64 / 128   // 整数截断
```

**逐 tile 推导**（N=320，5 个 tile）：

| tile id | 覆盖列 | `block_n_id × 64 ÷ 128`（整数除） | 使用 scale | scale 对应全局列 |
|---------|--------|-----------------------------------|-----------|----------------|
| 0 | [0, 64)   | 0 × 64 / 128 = **0** | scale[0] | [0, 128) |
| 1 | [64, 128) | 1 × 64 / 128 = **0** | scale[0] | [0, 128) |
| 2 | [128, 192) | 2 × 64 / 128 = **1** | scale[1] | [128, 256) |
| 3 | [192, 256) | 3 × 64 / 128 = **1** | scale[1] | [128, 256) |
| 4 | [256, 320) | 4 × 64 / 128 = **2** | scale[2] | [256, 384) |

**关键发现**：
- Tile 0 + Tile 1 = [0, 128)，合在一起正好是 scale[0] 量化的范围 → **完全正确**
- Tile 2 + Tile 3 = [128, 256)，合在一起正好是 scale[1] 量化的范围 → **完全正确**
- Tile 4 = [256, 320)，使用 scale[2]（对应 [256, 384)）→ **scale 基于 128 列，用于 64 列**

### ScaleSliceSizeN 的值

`gridwise_moe_gemm_blockscale.hpp:1354`：
```cpp
constexpr index_t ScaleSliceSizeN = math::integer_divide_ceil(NPerBlock, ScaleBlockN);
// = ceil_div(64, 128) = 1
```

每个 tile 加载 1 个 scale，张量描述符不越界，与 NPerBlock=128 时完全一致。

### 代码层面的结论

**device_moe_gemm_blockscale.hpp:448 的检查**：
```cpp
if(arg.N % NPerBlock != 0 || arg.K % KPerBlock != 0)
    return false;
```

N=320，NPerBlock=64：`320 % 64 = 0` → **通过**。

**代码中没有任何要求 `NPerBlock == ScaleBlockN` 或 `NPerBlock % ScaleBlockN == 0` 的 static_assert。**（teammate-8 V3 验证，全文件 grep 0 命中）

**结论：NPerBlock=64 + ScaleBlockN=128 + N=320 在 CK 代码层面完全合法，不需要 weight padding。**

---

## 第七层：scale[2] 的精度问题

Tile 4 使用的 scale[2] 是基于全局列 [256:384]（128 列）计算的，但 rank 0 只用 [256:320]（64 列）。

### 实测数据（teammate-9，layer 10 expert 0）

对 expert 0 的 320 个 (N-block, K-block) pair，分别计算"前 64 列"和"后 64 列"与"整 128 列"的最大值之比：

```
误差比 = (整 block max − 子段 max) / 整 block max
       = scale 偏大的比例（越大 → 精度损失越多）
```

| 统计量 | 前 64 列（rank 0 持有）| 后 64 列（rank 1 持有）|
|--------|----------------------|----------------------|
| mean   | 4.11%                | 6.37%                |
| median | 0%                   | 0%                   |
| p90    | 14.29%               | —                    |
| max    | 28.57%               | **46.43%**           |

### 如何理解这个数据？

- **中位数 0%**：超过一半的 block，前后两半的最大值是相同的（同一个 FP8 最大值恰好在两半都出现，或整个 block 的最大值就在其中一半）
- **平均 4-6%**：多数 block 误差很小
- **最坏 46.43%**：极端情况下，某 block 的后 64 列比前 64 列的权重幅值大 46%，用整体 scale 量化前 64 列会损失约 0.7 个 bit 的有效精度

### 关键认识

**这个精度损失并不是 NPerBlock=64 带来的新问题。**

当前方案（NPerBlock=128 + weight padding 到 384）：
- Tile 2 处理列 [256, 384)，其中 [256:320] 是真实权重，[320:384] 是 zero-padding
- Scale[2] 仍然是基于全局 [256:384]（128 列）计算的
- rank 0 在 tile 2 中，有效权重仍然只有 [256:320] 64 列

**换成 NPerBlock=64 + 不 padding：**
- Tile 4 处理列 [256, 320)，scale[2] 基于全局 [256:384] 计算
- 完全一样！精度不更差，也不更好。

---

## 第八层：那为什么之前说"需要 per_1x64"？

这个结论来自对 kernel 约束的错误推断，**代码层面不成立**。

正确的关系链是：

```
想要 NPerBlock=64
  → N=320 / NPerBlock=64 → 5 个 tile，320%64=0 ✓
  → scale 索引 block_n_id*64/128（整数除）→ 0,0,1,1,2 映射
  → 相邻两个 tile 共用一个 per_1x128 scale，数学上正确
  → device 检查 320%64=0 通过
  → 不需要 per_1x64！
```

**per_1x64 本来的含义**：每 64 列一个 scale，scale tensor 变成 `(288, 20, 32)` 而非 `(288, 10, 32)`，需要重新对 BF16 权重做量化。这在代码层面、精度层面都不是必须的。

---

## 第九层：真正需要做什么？

要实现"去掉 weight padding"，实际需要：

### 必须做的（3 处代码改动）

**1. aiter：新增 NPerBlock=64 的 blockscale kernel 实例**

文件：`aiter/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_blockscale.cuh`

- Stage1 L107：`Scale_Block_N = 128`（保持不变）
- 新增一套 `ck_moe_stage1_gemm_blockscale` 模板实例，NPerBlock=64，Scale_Block_N=128

文件：`aiter/csrc/ck_gemm_moe_2stages_codegen/gen_instances.py`

- L340-368（stage1 dispatch）：当 inter_dim=320 时，选择 NPerBlock=64 的 kernel 实例
- L739-770（stage2 dispatch）：同理（stage2 K=inter_dim=320，需要 KPerBlock=64）

**2. ATOM：去掉 inter_dim padding**

文件：`ATOM/atom/model_ops/moe.py:1709-1749`（`_process_block_quant`）

去掉或修改 L1721-1741 的 weight zero-pad 逻辑，直接用原始 320 列 weight。

Scale 不需要改动（仍是 `ceil(320/128)=3` 个 N-block）。

**3. aiter：放宽 device 层 N 检查（如果新增 NPerBlock=64 实例后仍触发）**

文件：`device_moe_gemm_blockscale.hpp:448`

由 `arg.N % NPerBlock` 触发，NPerBlock=64 时 `320 % 64 = 0`，这个检查会自动通过。

### 不需要做的

- **不需要重新量化模型**：checkpoint 的 scale tensor 直接复用
- **不需要改 ScaleBlockN**（保持 128）
- **不需要 per_1x64**
- **不需要改 CK gridwise kernel 源码**（模板参数已支持）

### 还需要验证的

- NXdlPerWave=1（NPerBlock=64 时）在 gfx950 FP8 上的实际 MFMA 利用率（需 microbench）
- `gridwise_moe_gemm_blockscale.hpp:1556` 的 `N0*N1*N2*N3*N4==NPerBlock` 在 NPerBlock=64 时能否编译（需试编译）
- Stage2 中 K=inter_dim=320，KPerBlock 是否也需要改（`320 % 128 = 64 ≠ 0`，stage2 需要 KPerBlock=64 或同样的 padding 处理）

---

## 总结

| 层次 | 核心问题 | 答案 |
|------|----------|------|
| 1 | 为什么要 padding？ | 320 % 128 ≠ 0，kernel 不接受非整除的 N |
| 2 | Kernel 在哪里拒绝 N=320？ | `device_moe_gemm_blockscale.hpp:448`，`arg.N % NPerBlock != 0` |
| 3 | Blockscale 的 scale 是什么？ | 每 128 列共享一个 scale_inv，checkpoint 全局 1280 列 |
| 4 | NPerBlock=128 为什么必须 padding？ | 必须让 N 是 NPerBlock 的整数倍才能通过 device 检查 |
| 5 | NPerBlock=64 的 scale 索引如何？ | `block_n_id*64/128` 整数截断 → `0,0,1,1,2`，两个 64 列 tile 共用一个 per_1x128 scale，数学正确 |
| 6 | NPerBlock=64 + ScaleBlockN=128 在代码上合法吗？ | 合法。device 检查 `320%64=0` 通过，无 static_assert 要求两者相等 |
| 7 | Scale 精度损失多大？ | 中位 0%，平均 4-6%，max 46.43%；与当前方案等价，不更差 |
| 8 | 需要 per_1x64 吗？ | **不需要**，之前结论有误 |
| 9 | 真正要做什么？ | 新增 NPerBlock=64 的 CK kernel 实例 + 去掉 ATOM weight padding |

---

## 附：已验证事实来源

| 事实 | 来源 |
|------|------|
| device_moe_gemm_blockscale.hpp:448 是硬性入口检查 | teammate-2 Q4，teammate-8 V5 |
| scale 索引公式 `block_n_id * NPerBlock / ScaleBlockN`（整数除） | teammate-8 V1，gridwise L1418 |
| ScaleSliceSizeN = ceil_div(NPerBlock, ScaleBlockN) | teammate-8 V2，gridwise L1354 |
| NPerBlock=64 + ScaleBlockN=128 + N=320 无 static_assert 阻拦 | teammate-8 V3，全文件 grep 0 命中 |
| device 检查 320%64=0 通过 | teammate-8 V5，DEV L448 |
| Checkpoint weight shape=(288,1280,4096)，scale shape=(288,10,32) | teammate-7 审查 1，safetensors 直读 |
| rank 0 的 scale[2] 来自 checkpoint narrow（基于全局 [256:384]） | teammate-9 Q3，ATOM moe.py:2287-2310 |
| 精度误差：median=0%，mean=4-6%，max=46.43% | teammate-9 Q4，layer 10 expert 0 实测 |
| NPerBlock=64 精度与 NPerBlock=128 等价 | teammate-9 核心结论 3 |
