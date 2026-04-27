# gfx950 FP8 mfma KPack=32 约束：为什么 blockscale MoE 无法去掉 padding

> **状态**：终稿（writer-A/B/C 起草，reviewer-A/B 交叉审核，Lead 整合）
> **审核修正**：2026-04-27
> 所有结论均附代码文件路径 + 精确行号，或实验数据，或 ISA 文档引用。

---

## 1. 背景：为什么要去掉 padding？

### 1.1 模型配置

来源：`/root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/6eebda59.../config.json`

| 字段 | 值 |
|------|----|
| `moe_intermediate_size` | 1280 |
| `moe_num_experts` | **288** |
| `quantization_config.quant_method` | `fp8` |
| `quantization_config.fmt` | `e4m3` |
| `quantization_config.weight_block_size` | `[128, 128]`（per_1x128 blockscale） |

### 1.2 Padding 的计算浪费

按 ATOM 实际 align 规则（`align = 64 if inter_dim <= 192 else 128`）：

| TP | `inter_dim = 1280/tp` | align | padding 目标 | 浪费比例 |
|----|----------------------|-------|-------------|---------|
| tp=2 | 640 | 128 | 640 | 0.0%（已对齐）|
| tp=4 | 320 | 128 | **384** | **16.7%** |
| tp=8 | 160 | 64  | **192** | **16.7%** |

实测输出（2026-04-27，ATOM `moe.py` L1724 align 规则）：
```
tp=2: inter=640, align=128, inter_pad=640, waste=0.0%
tp=4: inter=320, align=128, inter_pad=384, waste=16.7%
tp=8: inter=160, align=64,  inter_pad=192, waste=16.7%
```

**结论**：tp=4 与 tp=8 的 MoE GEMM 各自浪费约 16.7% 的计算量（stage1 N 维和 stage2 K 维各多处理 64 列无效数据）。去掉 padding 可直接节省这部分 compute。

---

## 2. ATOM 层：当前 padding 逻辑

### 2.1 `_process_block_quant` 的 align 规则

来源：`/home/hanchang/ATOM/atom/model_ops/moe.py`

```python
# L1709  def _process_block_quant(self, layer: nn.Module) -> None:
# L1719  inter_dim = layer.w2_weight.shape[-1]   # stage2 K = stage1 N = inter per rank
# L1720  block_n = 128 if self.quant_type == QuantType.per_1x128 else 32
# L1721-L1723  # NOTE: stage2 KPerBlock=64 不支持（gfx950 FP8 mfma KPack=32 约束）
# L1724  align = 64 if inter_dim <= 192 else block_n
# L1725  inter_pad = (inter_dim + align - 1) // align * align
```

- **`align=128`（inter>192，如 tp=4 inter=320）**：stage2 KPerBlock 在 blockscale FP8 路径只有 128，inter_dim 必须 128 对齐。
- **`align=64`（inter≤192，如 tp=8 inter=160）**：小 inter 时 stage1 存在 NPerBlock=64 的实例（见 §3.2），只需 64 对齐，pad 到 192。

### 2.2 为什么 w13 和 w2 必须同步 padding

代码：`moe.py` L1727-L1744

```python
# L1727-L1736  w13_new = zeros(E, 2*inter_pad, hidden)
#              w13_new[:, :inter_dim, :] = w13[:, :inter_dim, :]      # gate
#              w13_new[:, inter_pad:inter_pad+inter_dim, :] = ...     # up
# L1738-L1744  w2_new = zeros(E, hidden, inter_pad)
#              w2_new[:, :, :inter_dim] = w2
```

原因：fused MoE 数据流为 `x @ w13ᵀ → silu_and_mul → @ w2ᵀ`。**stage1 输出的 N 维 = stage2 输入的 K 维**，两者必须相同。若 w13 用 N=320 但 w2 用 K=384，shape 不匹配。

零填充对数值安全：`dequant(fp8(0), scale) = 0`（`moe.py` L1717 注释），padding 区对输出无影响。

### 2.3 scale tensor 不需要 padding

来源：`moe.py` L1718 注释："Scale tensors use ceil(inter/block_n) and are already shape-compatible."

```
ceil(320/128) = 3  =  ceil(384/128) = 3   (tp=4：padding 前后 scale N-blocks 数相同)
ceil(160/128) = 2  =  ceil(192/128) = 2   (tp=8：同上)
```

因此 `_process_block_quant` 只重建 weight，不触碰 scale tensor。

---

## 3. aiter dispatch 层：KPerBlock 的选择

### 3.1 blockscale stage2 dispatch（全为 KPerBlock=128）

来源：`/home/hanchang/aiter/csrc/ck_gemm_moe_2stages_codegen/gen_instances.py`
字符串模板 `A8W8_blockscale_gemm2_heuristic_dispatch`，L768-L799：

| block_m 分支 | 行号 | KPerBlock | NPerBlock | BLOCKSIZE | PipelineVer |
|--------------|------|-----------|-----------|-----------|-------------|
| `16 && inter_dim%256==0` | L779 | **128** | 16 | 256 | V1 |
| `16` else | L781 | **128** | 16 | 128 | V1 |
| `32` | L785 | **128** | 32 | 256 | V1 |
| `64` | L789 | **128** | 64 | 256 | **V3** |
| else | L793 | — | — | — | TORCH_CHECK |

所有分支 KPerBlock 均为 128，无任何 KPerBlock=64 选项。

### 3.2 `a8w8_gemm2_blockscale_kernels_list` 全部实例

来源：`/home/hanchang/aiter/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py` L314-L329：

| id | BLOCKSIZE | MPerBLOCK | NPerBLOCK | **KPerBLOCK** | MWaves | NWaves | Pipeline |
|----|-----------|-----------|-----------|---------------|--------|--------|----------|
| 0 | 256 | 16 | 128 | **128** | 1 | 4 | V1 |
| 1 | 128 | 16 | 128 | **128** | 1 | 2 | V1 |
| 2 | 256 | 32 | 128 | **128** | 1 | 4 | V1 |
| 3 | 256 | 64 | 128 | **128** | 1 | 4 | V3 |

L326-L328 注释明确写明 KPerBlock=64 不可用：
```python
# NOTE: KPerBlock=64 for FP8 blockscale is NOT supported on gfx950:
# static_assert(KPerThread % KPack == 0) fails (KPack=32 for FP8 mfma).
# Stage2 K=inter_dim requires padding to next multiple of 128.
```

### 3.3 对比：bf16 路径为什么有 KPerBlock=64

`a16w16_gemm2_kernels_list_gfx950`（L238-L253）中大量存在 KPerBlock=64 实例（如 id 2/3/9/10/11/12）。
`a8w8_gemm2_kernels_list_gfx950`（L280-L293）中 KPerBlock=64 实例被**注释禁用**，注释 L289：
> "disabled currently due to gfx950 fp8 mfma instruction used in CK not supporting these cases"

关键差异：bf16 mfma 的 KPack=8（BF16 路径），KPerBlock=64 → KPerThread=64/2=32，32%8=0 ✓；FP8 mfma 的 KPack=32，KPerBlock=64 → KPerThread=16，16%32≠0 ✗。

---

## 4. CK Kernel 模板层：KPerBlock 的传递路径

### 4.1 从 dispatch 到 DeviceMoeGemmBlockScale

来源：`/home/hanchang/aiter/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_blockscale.cuh`

stage2 模板参数声明（**L318-L334**）：
```cpp
template <typename A0DataType,
          ...
          int KPerBlock,      // L327 — 调用方控制的 K 维 tile
          int MWaves,
          int NWaves, ...>
```

KPerBlock 经 L416 直接传给 `DeviceMoeGemmBlockScale`：
```cpp
// L410-L423
using DeviceOpInstanceNormal = ck::tensor_operation::device::DeviceMoeGemmBlockScale
    < Row, Col, DsLayout, ELayout,
      A0DataType, ..., GemmSpec,
      BLOCKSIZE, Scale_Block_M, Scale_Block_N, Scale_Block_K,
      MPerBlock, NPerBlock, KPerBlock,  // L416 — KPerBlock 在此传入
      AK1, BK1, ...>;
```

### 4.2 KPerBlock → AK1/BK1 → K0 的派生（**L393-L405**）

```cpp
static constexpr ck::index_t AK1 = 16 / sizeof(A0DataType);  // L393 → FP8: AK1=16
static constexpr ck::index_t BK1 = 16 / sizeof(B0DataType);  // L394 → FP8: BK1=16
static constexpr ck::index_t K0_A = KPerBlock / AK1;         // L402
static constexpr ck::index_t K0_B = KPerBlock / BK1;         // L403
```

| KPerBlock | AK1 | BK1 | K0_A | K0_B |
|-----------|-----|-----|------|------|
| 64 | 16 | 16 | 4 | 4 |
| **128** | 16 | 16 | **8** | **8** |
| 256 | 16 | 16 | 16 | 16 |

---

## 5. 静态断言：`KPerThread % KPack == 0`

### 5.1 断言原文

来源：`/home/hanchang/aiter/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_base.hpp`

KPack 是模板参数（**L35**）：
```cpp
index_t KPack,
```

KPerThread 的计算公式（**L68-L70**）：
```cpp
static constexpr index_t KPerThread    = KPerBlock / xdlops_gemm.K0PerXdlops;  // L68
static constexpr index_t KRepeat       = KPerThread / KPack;                   // L69
static constexpr index_t KPerInnerLoop = KPack;                                 // L70
```

静态断言（**L105-L106**）：
```cpp
static_assert(KPerThread % KPack == 0,
              "Wrong KPack setting; try increasing KPerThread or decreasing KPack");
```

### 5.2 KPack=32 的来源（FP8 路径）

**5.2.1 gridwise 层的 KPack 计算**

来源：`gridwise_moe_gemm_blockscale.hpp` **L287-L289**：
```cpp
using mfma_selector = MfmaSelector<ComputeTypeA, MPerXdl, NPerXdl, ComputeTypeB>;
static constexpr index_t KPack =
    math::max(math::lcm(AK1Number, BK1Number), mfma_selector::selected_mfma.k_per_blk);
```

FP8 路径代入：
- `lcm(AK1=16, BK1=16) = 16`
- `mfma_selector::selected_mfma.k_per_blk = 32`（见 §5.2.2）
- **`KPack = max(16, 32) = 32`**

**5.2.2 gfx950 FP8 mfma 的 k_per_blk**

来源：`xdlops_gemm.hpp`，gfx950 FP8 `mfma_scale_f32_16x16x128_f8f6f4` 结构（**L919-L932**）：

```cpp
k_per_blk      = 32        // L929
num_input_blks = 4         // L924
is_k_reduction = true      // L931
```

由 `GetKPerXdlops`（**L1855-L1859**）推导：
```cpp
KPerXdlops  = num_input_blks * k_per_blk = 4 × 32 = 128
K1PerXdlops = k_per_blk = 32
K0PerXdlops = KPerXdlops / K1PerXdlops = 128 / 32 = 4
```

**5.2.3 代码注释明确 gfx950 FP8 需要 32 个元素**

`blockwise_gemm_pipeline_xdlops_base.hpp` **L74-L78**：
```cpp
// On gfx950, we have mfma that required 32 f8 elements as input,
// splited into 2 groups of 16 f8 elements.
// the 2 groups is not contiguous in the B preshuffed layout.
// and we do not want it to be contiguous in the B preshuffled layout
// because a memory instruction can only read 16 f8 elements at a time.
```

### 5.3 调用链：从 dispatch 到 static_assert

| 层 | 文件 | 关键行号 |
|----|------|---------|
| 1. dispatch | `gemm_moe_ck2stages_common_blockscale.cuh` | L327（KPerBlock 参数），L416（传给 DeviceOp） |
| 2. DeviceMoeGemmBlockScale | `device_moe_gemm_blockscale.hpp` | 实例化 GridwiseMoeGemmBlockScale |
| 3. GridwiseMoeGemmBlockScale | `gridwise_moe_gemm_blockscale.hpp` | L287-L289（KPack=32 固化），L876-L903（Selector） |
| 4. BlockGemmPipeline Selector | `blockwise_gemm_pipeline_xdlops_moe_blockscale_b_preshuffle_selector.hpp` | L37-L94（v1），L96-L153（v3） |
| 5. Pipeline v1/v3 | `blockwise_gemm_pipeline_xdlops_moe_blockscale_b_preshuffle_v1.hpp` | L90/L130 继承 base |
| 6. **base（触发点）** | `blockwise_gemm_pipeline_xdlops_base.hpp` | **L68**（KPerThread 公式），**L105-L106**（static_assert） |

---

## 6. gfx950 硬件层：FP8 mfma 指令的 K 维度

### 6.1 指令规格（rocm-ref + 代码）

来源：`/tmp/rocm-ref/rocm-ref/topics/mfma-register-layout.md`

**CDNA3（gfx940/942）原 FP8 mfma**（L35-L36）：
```
v_mfma_f32_16x16x32_f8f8   → M=16, N=16, K=32（每次 32 个 FP8 元素）
v_mfma_f32_32x32x16_f8f8   → M=32, N=32, K=16
```

**CDNA4（gfx950/MI350X）新增 mfma**（L67-L68）：
```
v_mfma_scale_f32_16x16x128_f8f6f4  → M=16, N=16, K=128（4 组×32）
v_mfma_scale_f32_32x32x64_f8f6f4   → M=32, N=32, K=64（2 组×32）
```

两代指令均以 **32 个 FP8 元素**为最小 K 处理单元（wave lane level），这是 KPack=32 的硬件根源。

### 6.2 KPack=32 与 mfma K=32 的对应

| 硬件层 | 值 | 来源 |
|--------|------|------|
| gfx950 mfma 单 wave lane FP8 输入（每 instruction 组） | 32 个 FP8 | rocm-ref mfma-register-layout.md L67；base.hpp L74-L78 |
| `mfma_type::k_per_blk`（gfx950 FP8） | 32 | `xdlops_gemm.hpp` L929 |
| `KPack = max(lcm(16,16), 32)` | **32** | `gridwise_moe_gemm_blockscale.hpp` L288-L289 |
| `K0PerXdlops = 128 / 32` | **4** | `xdlops_gemm.hpp` L2254-L2256 |

---

## 7. 数学推导：KPerBlock=64 为什么触发 static_assert

### 7.1 KPerThread 公式

来源：`blockwise_gemm_pipeline_xdlops_base.hpp` L68，以及 `xdlops_gemm.hpp` L2254-L2256：

```
KPerThread = KPerBlock / K0PerXdlops

其中：
  KPerXdlops  = num_input_blks × k_per_blk = 4 × 32 = 128
  K1PerXdlops = k_per_blk = 32
  K0PerXdlops = KPerXdlops / K1PerXdlops = 128 / 32 = 4

化简：KPerThread = KPerBlock / 4
约束：KPerThread % KPack == 0  →  (KPerBlock / 4) % 32 == 0  →  KPerBlock % 128 == 0
```

### 7.2 KPerBlock=64（失败案例）

```
参数：BLOCKSIZE=256, MPerBlock=32, NPerBlock=128, KPerBlock=64
      MWaves=1, NWaves=4, KPack=32（FP8）

计算：
  K0PerXdlops = 4
  KPerThread  = 64 / 4 = 16
  KRepeat     = 16 / 32 = 0  （整数除法，KRepeat=0 还会引发下游描述符崩溃）

验证约束：
  KPerThread % KPack = 16 % 32 = 16 ≠ 0
  → static_assert 触发 → 编译失败
```

报错原文（实际编译错误）：
```
error: static assertion failed due to requirement 'KPerThread % 32 == 0':
       Wrong KPack setting; try increasing KPerThread or decreasing KPack
```

### 7.3 KPerBlock=128（合法案例）

```
KPerBlock = 128（其余参数相同）

计算：
  KPerThread = 128 / 4 = 32
  KRepeat    = 32 / 32 = 1 ✓

验证约束：
  KPerThread % KPack = 32 % 32 = 0 ✓ → 编译通过
```

### 7.4 最小合法 KPerBlock 的推导

```
约束化简：KPerBlock % 128 == 0 且 KPerBlock ≥ 128
合法集合：{128, 256, 384, 512, ...}

对 stage2 K = inter_dim = 320（tp=4）：
  320 不在上述集合中（320 / 128 = 2.5，不整除）
  最近的合法值 = ⌈320/128⌉ × 128 = 3 × 128 = 384
  → 必须 padding 320 → 384
```

---

## 8. Stage1 与 Stage2 的不对称

核心原因：**KPack 约束只作用于 GEMM 的 K 维**，而 stage1 和 stage2 的 K 维分别对应不同的模型维度。

| | Stage1 | Stage2 |
|-|--------|--------|
| GEMM 含义 | `[M, K_hidden] × [K_hidden, N_inter]ᵀ` | `[M, K_inter] × [K_inter, N_hidden]ᵀ` |
| **K 维度** | hidden_dim/tp（如 4096/4=1024） | **inter_dim**（如 320） |
| N 维度 | **inter_dim**（如 320） | hidden_dim/tp（如 1024） |
| KPack 约束 | 对 K=1024：KPerBlock=128 → KPerThread=1024/4÷(128/4)=256，256%32=0 ✓ | 对 **K=320**：KPerBlock=64 → KPerThread=16，16%32≠0 ✗ |

**Stage1 的 N 维是 inter_dim=320，使用 NPerBlock=64（N 方向）不受 KPack 约束——因为 KPack 约束的是 K 方向**。这就是为什么 stage1 NPerBlock=64 可行，而 stage2 KPerBlock=64 不可行。

### 实验验证：stage1 NPerBlock=64 的 3 个 kernel 已成功编译

来源：`/home/hanchang/aiter/aiter/jit/build/.../build/` 目录中存在的 `.cuda.o` 文件（2026-04-27 实测）：

```
✓ moe_ck2stages_gemm1_256x16x64x256_1x4_...v1...F8_F8_B16.cuda.o  (M=16, NPerBlock=64, K=256)
✓ moe_ck2stages_gemm1_256x32x64x128_1x4_...v1...F8_F8_B16.cuda.o  (M=32, NPerBlock=64, K=128)
✓ moe_ck2stages_gemm1_256x64x64x128_1x4_...v1...F8_F8_B16.cuda.o  (M=64, NPerBlock=64, K=128)
```

KPack 约束验证：
- M=16: KPerThread = 256/4 = 64; 64 % 32 = 0 ✓
- M=32: KPerThread = 128/4 = 32; 32 % 32 = 0 ✓
- M=64: KPerThread = 128/4 = 32; 32 % 32 = 0 ✓

---

## 9. 完整因果链

```
[硬件层] gfx950 FP8 mfma 指令族
  v_mfma_scale_f32_16x16x128_f8f6f4（M=16, N=16, K=128; k_per_blk=32, num_input_blks=4）
          ↓ 每次指令最小 K 处理单元 = 32 个 FP8 元素
          ↓ 来源：rocm-ref mfma-register-layout.md L67；xdlops_gemm.hpp L929

[CK 模板层] KPack = max(lcm(AK1=16, BK1=16), mfma.k_per_blk=32) = 32
          ↓ 计算：gridwise_moe_gemm_blockscale.hpp L287-L289
          ↓ K0PerXdlops = 128/32 = 4；xdlops_gemm.hpp L2254-L2256

[编译期约束] static_assert(KPerThread % KPack == 0)
          KPerThread = KPerBlock / K0PerXdlops = KPerBlock / 4
          → KPerBlock % 128 == 0（化简约束）
          ↓ 来源：blockwise_gemm_pipeline_xdlops_base.hpp L68, L105-L106

[模型层] stage2 K = inter_dim（per TP rank）
  tp=2: inter_dim = 640 → 640 % 128 = 0 ✓ → KPerBlock=128 合法，无需 padding
  tp=4: inter_dim = 320 → 320 % 128 = 64 ≠ 0 → 无合法 KPerBlock，必须 padding 到 384
  tp=8: inter_dim = 160 → 160 % 128 = 32 ≠ 0 → 必须 padding 到 192（align=64）

[结论]
gfx950 + FP8 blockscale + tp=4（inter=320）的 stage2 MoE GEMM
必须保持 padding 320→384。
此 padding 是：
  硬件 mfma K=32（物理约束）
→ CK KPack=32（模板常量）
→ KPerBlock % 128 == 0（编译期断言）
→ inter_dim=320 不整除 128（模型维度）
四层叠加的必然结果，无法仅通过 CK kernel 调参消除。
```

精简因果表：

| 层次 | 关键值 | 代码来源 |
|------|--------|---------|
| gfx950 FP8 mfma 指令 | k_per_blk=32, num_input_blks=4 | `xdlops_gemm.hpp` L929, L924 |
| KPack 计算 | max(16, 32) = **32** | `gridwise_moe_gemm_blockscale.hpp` L287-L289 |
| K0PerXdlops | 128 / 32 = **4** | `xdlops_gemm.hpp` L2254-L2256 |
| KPerThread（KPerBlock=64） | 64 / 4 = **16** | `blockwise_gemm_pipeline_xdlops_base.hpp` L68 |
| static_assert 触发 | 16 % 32 = 16 ≠ 0 | 同上 L105-L106 |
| 最小合法 KPerBlock | **128** | 数学推导 §7.4 |
| 模型 inter_dim（tp=4） | **320** | Step-3.5-Flash config.json |
| 推导 padding 后值 | ⌈320/128⌉×128 = **384** | `moe.py` L1724-L1725 |

---

## 10. 展望：可能的解决路径

> 以下全部为理论分析，未经实验验证。

### 路径 A：等待 AMD 提供更小 K-tile 的 FP8 mfma 指令

若未来 mfma 支持 `v_mfma_f32_16x16x64_fp8`（k_per_blk=16），则 KPack 可降为 max(16,16)=16，约束变为 KPerBlock%64==0，inter=320=5×64 完全合法，无需任何 padding。
**代价**：依赖下一代硬件，gfx950 无法绕过。

### 路径 B：用 Triton kernel 替代 CK stage2

Triton 可在 host 端生成任意 BLOCK_K 的 GEMM kernel，不受 CK 编译期 static_assert 约束。
**代价**：需重写 blockscale 的 scale 加载逻辑；Triton 在 AMD 上通常略慢于精调 CK。

### 路径 C：Stage2 改用 BF16（先 dequant 再 GEMM）

Stage1 输出 dequant → BF16 → stage2 BF16 GEMM，BF16 KPack=8，KPerBlock=64 完全合法，无 padding。
**代价**：额外 dequant 访存；stage2 BF16 算力约为 FP8 的 1/2，但消除 16.7% padding 有部分抵消。

### 路径 D：在 tp=2 部署（当前已采用的折中）

tp=2 的 inter=640=5×128，已满足 KPerBlock=128，无 padding，且无需改任何代码。
**代价**：tp 选择受模型维度与硬件约束双重绑定；decode TPOT 上 tp=2 已优于 tp=4（MEMORY.md 实测）。

| 路径 | 是否改硬件 | 是否改算子 | 工程量 | 当前可执行 |
|------|-----------|-----------|--------|-----------|
| A | 是 | 否 | 极高 | 不可控 |
| B | 否 | 是（Triton kernel） | 中 | 可立项 |
| C | 否 | 是（混合精度 stage2） | 中低 | 可试验 |
| D | 否 | 否 | 极低 | **已采用** |

---

## 附录：编译错误原文与验证数据

### A.1 静态断言原文

`blockwise_gemm_pipeline_xdlops_base.hpp` **L104-L107**：
```cpp
#if defined(__HIP_DEVICE_COMPILE__)
static_assert(KPerThread % KPack == 0,
              "Wrong KPack setting; try increasing KPerThread or decreasing KPack");
#endif
```

实际编译输出（stage2 KPerBlock=64 尝试编译时，2026-04-27 实测）：
```
error: static assertion failed due to requirement 'KPerThread % 32 == 0':
       Wrong KPack setting; try increasing KPerThread or decreasing KPack
```

### A.2 失败的 stage2 KPerBlock=64 kernel（全部 4 个）

```
✗ gemm2_256x16x128x64_1x4_...F8_F8_B16  → KPerThread=16, 16%32≠0
✗ gemm2_256x32x128x64_1x4_...F8_F8_B16  → KPerThread=16, 16%32≠0
✗ gemm2_256x64x128x64_1x4_...F8_F8_B16  → KPerThread=16, 16%32≠0（同时：V3 pipeline 另有 MRepeat 约束）
```

.cu 文件存在于 blob/instances/，但 .cuda.o 均缺失（编译失败）。

### A.3 成功编译的 stage1 NPerBlock=64 kernel（3 个，实测 .cuda.o 存在）

```
✓ gemm1_256x16x64x256_1x4_...v1...F8_F8_B16.cuda.o  (KPerThread=64, 64%32=0)
✓ gemm1_256x32x64x128_1x4_...v1...F8_F8_B16.cuda.o  (KPerThread=32, 32%32=0)
✓ gemm1_256x64x64x128_1x4_...v1...F8_F8_B16.cuda.o  (KPerThread=32, 32%32=0)
```

---

## Review 记录

| 审核方 | 范围 | 关键修正 |
|--------|------|---------|
| reviewer-A | §1-5 | moe_num_experts 288（非 48）；tp=8 padding 到 192（非 256）；§3.1 补 PipelineVer 列；§4.1 L318 起（非 L319） |
| reviewer-B | §6-10 | tp=8 padding 到 192（非 256）确认；KPack 公式链 ✓；rocm-ref 引用 ✓；数学推导 ✓ |
| verifier-A（Lead 代执行） | 编译产物 | stage1 NPerBlock=64 的 3 个 .cuda.o 实测确认存在 |
