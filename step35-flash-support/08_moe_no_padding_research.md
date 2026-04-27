# PROJECT SUMMARY — MoE No-Padding 调研

> 状态：**已审查定稿**（teammate-6 撰写，teammate-7 严格审查，Lead 更新）
> 数据来源：`progress/teammate-1.md` ~ `progress/teammate-7.md`
> 关键结论已修正：见第 5 节

---

## 1. 任务背景与目标

Step-3.5-Flash 在 tp=4 时，每个 rank 的 MoE `intermediate_size_per_partition = 1280 / 4 = 320`，但 CK FP8 blockscale MoE kernel 当前实例化的 `NPerBlock = 128`、`ScaleBlockN = 128`，N=320 不是 128 的倍数。ATOM 当前通过把 weight padding 到 `ceil(320/128)*128 = 384` 绕开此约束，N 方向算力浪费约 16.7%。本次调研评估两条"去 padding"路径的可行性、改造成本、对模型 checkpoint 的影响，给出后续推进建议。

---

## 2. 当前 Padding 方案（已验证基线）

- 配置：`inter_dim=320`（tp=4），padding 到 `384 = 3 × 128`，CK `NPerBlock=128`，`ScaleBlockN=ScaleBlockK=128`。
- ATOM padding 入口：`/home/hanchang/ATOM/atom/model_ops/moe.py`
  - `Fp8MoEMethod._process_block_quant`（L1709-1749）只 pad **weight**（`torch.zeros` 填充，L1727 / L1736），**不 pad scale**。
  - `align = 64 if inter ≤ 192 else block_n`（L1721）。
  - `create_weights` 中 `padded_inter = ceil(intermediate_size_per_partition / block_n) * block_n`（L1554-1571），ValueError 校验在 L1576-1581（`padded_inter % block_n != 0`，因 ceil 对齐永远不触发）；tp_size>1 额外校验 `padded_inter % block_k != 0`（L1582-1588）。
  - scale 张量按 `ceil(intermediate_size_per_partition / 128)` 分配（L1636），与 padding 后 weight 的 `ceil(384/128)=3` 一致。
- 数学约束：per_1x128 weight quant 要求每 N 维 block 共享一个 scale，`ScaleBlockN=128`；CK kernel grid 按 `ceil(N / NPerBlock)` 调度，host 必须保证 weight storage 至少覆盖 `ceil(N / NPerBlock) * NPerBlock` 才不越界。
- 数值正确性依赖：FP8 e4m3fnuz 的 zero × scale = 0，padding 区域计算结果不影响输出；并且 `ceil(320/128) == ceil(384/128) == 3`，scale tensor 与 padded weight 自然 shape-compatible（来源：teammate-1 § Q6.1，关键发现 #3）。
- 已验证里程碑：FP8 tp=4 V06 Exp2 实测 TTFT=86ms / TPOT=13ms / gmu=0.7（MEMORY.md）。

---

## 3. 关键技术发现

### 3.1 CK kernel 层分析（来自 teammate-1, teammate-2）

文件：
- gridwise: `/home/hanchang/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm_blockscale.hpp`
- device:   `/home/hanchang/composable_kernel/include/ck/tensor_operation/gpu/device/impl/device_moe_gemm_blockscale.hpp`
- aiter wrapper: `/home/hanchang/aiter/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_blockscale.cuh`

#### CheckValidity 对 N 的实际行为
- gridwise L932-948：`if(!(karg.N % NPerBlock == 0)) return false;` 被 `is_same<RowMajor, BLayout>` 短路（B 是 Col layout），**对 blockscale MoE 不触发**（teammate-1 § Q1.3，关键发现 #1）。
- gridwise L1010：`karg.N % BBlockTransferSrcScalarPerVector` 的 N 维校验同样在 BLayout=Col 时跳过。
- gridwise L1039：`if(CLayout==RowMajor && karg.N % CShuffleBlockTransferScalarPerVector_NPerBlock != 0) return false;` 对 CLayout=Row 仍生效，但 N=320 通常能整除 ScalarPerVector=8（teammate-2 § Q4）。

#### device_moe_gemm_blockscale.hpp L448 的硬性检查（**真正的卡点**）
```cpp
// L448
if(arg.N % NPerBlock != 0 || arg.K % KPerBlock != 0)
    return false;
```
该检查在调用 gridwise CheckValidity 之前（L462/L469），**无 GemmSpec 守护**——即使把 GemmSpec 改成 NPadding，N=320 仍在 device 层被拒（teammate-2 § Q4）。

#### C-store / B-load / B-scale 各自的 padding 需求
- **C-store（teammate-2 § Q1）**：gridwise L634-639 `MakeCGridDescriptor_M_N` 无条件应用 `make_right_pad_transform(M, MPad - M)` / `(N, NPad - N)`；C-store 经 `RunMoeEpilogue`（gridwise_gemm_xdl_cshuffle_common.hpp L1642）+ `ThreadGroupTensorSliceTransfer_v7r3_scatter`（L1728），CK buffer-store 配合 right_pad 会产生 OOB mask。**天然 OK，无需改动**。
  - 【未验证假设】v7r3_scatter 内部 mask 行为仅基于 CK 通用约定推断，未直接读源码确认。
- **B-load（teammate-2 § Q2）**：B 走 preshuffled 路径 `MakeBGridDescriptor_Preshuffled(BN0Shuffled, BK0Shuffled)`（L1149-1150），**完全不调用** `MakeBGridDescriptor_BK0_N_BK1` 中的 right_pad（L557-568），即使开启 `GemmSpec=NPadding` 也对 blockscale **无效**。`BN0Shuffled = ceil(N, NLane)`（L357-360）只是 N0 维 wave-shuffle 后的逻辑维，无 element-level mask。`b_blockwise_copy` 模板（L1318-1328、L1434-1448）无任何 OOB predicate。**N=320 时第 3 个 N-block 会读越界**——是否 fault 取决于 host 侧 weight 实际分配大小与 buffer-resource 行为。
- **B-scale（teammate-1 § Q1.2、teammate-2 § Q3）**：scale grid descriptor 用 `integer_divide_ceil(N, ScaleBlockN)`（L1164-1168 / L1700-1702）；起始 `make_multi_index(block_n_id * NPerBlock / ScaleBlockN, 0)`（L1418、L1467）；`ScaleSliceSizeN = ceil(NPerBlock, ScaleBlockN) = 1`。N=320 时 ceil(320/128)=3 行 scale，第 3 个 N-block (block_n_id=2) 索引 2，合法。**天然 OK**。
- **expert scale stride**（gridwise L1227-1229）：`ceil(N/ScaleBlockN) * (IsInputGemm?2:1) * ceil(K/ScaleBlockK)`，host 必须按 ceil(N/ScaleBlockN) 分配 scale buffer。N=320 时 ceil=3，与 ATOM 当前 scale 分配（L1636）一致。

#### GemmSpec=NPadding 的作用范围
- aiter wrapper 当前是 `GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default`（cuh L83 stage1, L380 stage2）。
- 对 blockscale + Col B：把 GemmSpec 改为 NPadding **几乎无效**——right_pad 只在 `MakeBGridDescriptor_BK0_N_BK1` 中应用，但 blockscale 走的是 `MakeBGridDescriptor_Preshuffled`（teammate-2 § Q2）。
- C-store 已经无条件 right_pad；B-scale 已经 ceil；只有 B-load 缺 mask，且无法仅靠 GemmSpec 解决。

### 3.2 N=320 实验结果（来自 teammate-3）

实验脚本：`/home/hanchang/project_moe_no_padding/microbench_n320.py`；日志：`/home/hanchang/project_moe_no_padding/logs/microbench_n320.log`。

| 变体 | inter_dim_phys | intermediate_pad | 结果 |
|------|---------------:|-----------------:|------|
| A. N=384 padded baseline | 384 | 64 | OK |
| B. N=384 alloc, intermediate_pad=0 | 384 | 0 | OK |
| C. N=320 surrogate (zero-tail, intermediate_pad=0) | 384 (tail [320:384] 置零) | 0 | OK |

结论：
- 三个变体均跑通，无 SIGBUS / illegal address / TORCH_CHECK 失败 / NaN / Inf。
- **wrapper 层无 N 对齐强校验**：aiter `fused_moe` 的 `intermediate_pad` 参数只影响 wrapper 里 fused-bias / zero-mask 逻辑，**不会改变 CK kernel 的 N-grid（NPerBlock=128 仍 hardcoded）**。
- **真正的 bug 屏障在 CK 模板层，不在 wrapper 层**。物理 N=320 直接喂 kernel 无法从 Python 触发（per_128x128 weight quant 要求 N % 128 == 0；CK NPerBlock=128 hardcoded）。
- 真正物理 N=320 是否 crash 仍是**未验证假设**——需要改 CK 源码后再跑。

### 3.3 per_1x128 下 N=320 的数学证明（来自 teammate-5 § Q5）

约束：
- N=320，per_1x128 → `ScaleBlockN=128`。
- CK MoE blockscale kernel 要求 `NPerBlock` 是 `ScaleBlockN` 的倍数（基于 scale 索引 `block_n_id * NPerBlock / ScaleBlockN` 必须落在合法行；【未验证假设】未读 CK template 直接确认 static_assert）。
- 同时要求 `NPerBlock | N`（否则越界）。

推导：
- 设 `NPerBlock = 128k`（`k ∈ ℤ⁺`），需 `128k | 320`。
- `320 = 2⁶ × 5`，`128 = 2⁷`，`gcd(320, 128) = 64`。320 不是 128 的倍数，更不是 256/384/... 的倍数。
- ∴ **不存在** `k ∈ ℤ⁺` 使得 `128k | 320`。

**结论**：在 per_1x128 约束下，N=320 不存在无 padding 的 NPerBlock 选择。padding 到 384 是数学上**唯一**的方案。

### 3.4 per_1x64（方案 B）CK kernel 层可行性（来自 teammate-4）

- **ScaleBlockN 是模板参数**：`gridwise_moe_gemm_blockscale.hpp:128` `index_t ScaleBlockN,`；但 aiter wrapper `gemm_moe_ck2stages_common_blockscale.cuh:107`（stage1）和 L407（stage2）`static constexpr ck::index_t Scale_Block_N = 128;` 硬编码。CK 全部 7 个 fp8 blockscale example 也都 hardcode 128。
- **NPerBlock=64 静态约束**（gridwise）：
  - L910-912：`(NPerBlock % (NXdlPerWave * NPerXdl)) == 0`，64 % 16 == 0 ✓。
  - L305：`NWave = NPerBlock / NPerXdl / NXdlPerWave`，64/16/1=4（与 BLOCKSIZE=256 → 4 waves 一致）。
  - L1556-1558：`N0*N1*N2*N3*N4 == NPerBlock` 在 NPerBlock=64 下能否找到合法分解，**待编译验证**。
  - L1354：`ScaleSliceSizeN = ceil_div(NPerBlock, ScaleBlockN)`，使用 ceil_div，理论允许 NPerBlock < ScaleBlockN。
  - **未发现** `static_assert(NPerBlock % ScaleBlockN == 0)`。
- **gfx950 FP8 mfma 的限制**：`gen_instances.py:431` 的注释 "temporarily not using KPerBlock=64 for inter_dim=192 cases due to gfx950 fp8 mfma instruction limitation" 说的是 **KPerBlock=64**，**不是 NPerBlock=64**。FP8 mfma 16x16x32 的 N tile 单位是 16，NPerBlock=64 = 4×16 是合法的。
- **改造工作量评估（中）**：
  1. `gemm_moe_ck2stages_common_blockscale.cuh` L107/L407 的 `Scale_Block_N` 参数化或新增 `_n64` 镜像 wrapper。
  2. `gen_instances.py` L340-368（stage1）、L739-770（stage2）的 dispatch 字符串模板新增 NPerBlock=64 分支或新 quanttype（`_blockscale_n64`）。
  3. `device_moe_gemm_blockscale.hpp` 新增 ScaleBlockN=64 的显式实例化（避免链接缺符号）。
  4. 新增 tuning 表 + 单元测试。
- 风险：
  - NXdlPerWave=1 (vs 当前 2) 的 mfma ILP 实际效率，需 microbench 验证。
  - L1556 N0..N4 分解能否成立，需编译验证。
  - 全代码库无任何 `Scale_Block_N != 128` 实例，"first of its kind"。

### 3.5 per_1x64 量化侧可行性（来自 teammate-5 § Q1-Q4）

#### Step-3.5-Flash-FP8 checkpoint 的量化配置
- 配置：`/root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash-FP8/.../config.json:313-321`：
  - `weight_block_size: [128, 128]`、`activation_scheme: dynamic`、`fmt: e4m3`。
- safetensors 已经按 [128,128] block_size 离线量化，weight_scale_inv 张量随 checkpoint 发布。
- StepFun **未发布** per_1x64 版本。
- `modules_to_not_convert` 排除 layer 0-2 mlp、attention 投影、share_expert、mtp_block、moe.gate；只有 layer 3-44 的 routed experts 走 FP8 + per_1x128。

#### ATOM/aiter 对 per_1x64 的支持情况
- ATOM `quant_spec.py:114-119`：`_QSCHEME_TO_QUANT_TYPE` 只有 `per_block → per_1x128`、`per_group → per_1x32`，**无 64**。
- ATOM `moe.py:1142-1161`：`Fp8MoEMethod` 二分支硬编码 `block_n/block_k`（per_1x128 → 128/128，per_1x32 → 1/32）。
- aiter `csrc/include/aiter_enum.h:15-25`：QuantType 枚举仅 `No / per_Tensor / per_Token / per_1x32 / per_1x128 / per_128x128 / per_256x128 / per_1024x128`，**无 per_1x64**。
- aiter `aiter/fused_moe.py:468-474`：`fc_scale_blkn=128, fc_scale_blkk=128` 写死在 `functools.partial`。
- aiter `aiter/ops/quant.py:243-282`：dispatch table 无 per_1x64。
- 同样硬编码出现在 `fused_moe_dp_shared_expert.py:282`。
- `grep per_1x64` 在 aiter / ATOM 两个仓库**均无任何匹配**。

#### 模型重新量化的代价
- 必须基于 BF16 原始权重 `/root/.cache/.../models--stepfun-ai--Step-3.5-Flash` 自建量化 pipeline，重新生成 45 layers × ~288 routed experts 的 w13/w2 与 weight_scale_inv。
- 精度需 lm_eval / PPL 重测（per_1x64 理论精度略优于 per_1x128，但实际收益依赖权重分布）。
- **是否真的需要重新量化** —— 见第 5 节，待 teammate-7 审查结论。

---

## 4. 两种方案对比

| 维度 | 方案 A（语义去 padding，CK kernel 接受非对齐 N） | 方案 B（per_1x64 真正去 padding） |
|------|------------------------|----------------------------------|
| 计算量 | 同当前（grid 仍 3 blocks × 128，最后 1/3 浪费） | 更少（5 blocks × 64，但 NXdlPerWave 2→1，MFMA ILP 可能下降） |
| 内存 | 仍需 zero-pad weight（host 侧）；scale 已是 ceil 形态 | 真正 320 列 weight（节省 16.7%）；scale 在 N 方向多 67% |
| CK 改动 | 放宽 `device_moe_gemm_blockscale.hpp:L448` 检查；可能给 `b_blockwise_copy` 加 N tail predicate（侵入性高） | 全新 `ScaleBlockN=64` kernel 实例 + dispatch 分支（中） |
| ATOM/aiter 改动 | 小：去掉 `_process_block_quant` 中 weight padding；inter_dim 直接传 320 | 中大：新 `QuantType.per_1x64` 枚举 + 量化 pipeline + dispatch 镜像 |
| 模型重新量化 | **不需要**（仍是 per_1x128 checkpoint） | **重新生成 scale 必须；回 BF16 重量化推荐但非必须（可 FP8 rebinning）** |
| tp=2 场景（inter_dim=640，本身 5×128 已无 padding） | 同当前，无影响 | scale tensor 翻倍，**净亏损** |
| 现存参考实现 | 无（CK blockscale 当前无 partial-N-block 支持，gridwise grep 0 命中 tail 处理，teammate-1 § Q5） | 无（CK 全部 blockscale example 均 hardcode 128，teammate-4 § Q1） |

---

## 5. 审查结论："需要重新量化"是否正确？

**审查来源**：teammate-7，直接读取 safetensors checkpoint。

### 实测数据（决定性）

```
文件：model-00018.safetensors，layer 3 routed experts
model.layers.3.moe.gate_proj.weight           shape=(288, 1280, 4096)  dtype=float8_e4m3fn
model.layers.3.moe.gate_proj.weight_scale_inv shape=(288, 10, 32)      dtype=float32
model.layers.3.moe.down_proj.weight           shape=(288, 4096, 1280)  dtype=float8_e4m3fn
model.layers.3.moe.down_proj.weight_scale_inv shape=(288, 32, 10)      dtype=float32
```

- weight 存储**全局 1280 列**（未做 TP partition）
- scale N_scale = **10 = ceil(1280/128)**（全局 inter_dim 的 N-block 数）
- TP partition 发生在推理加载时：`ATOM/atom/model_ops/moe.py` L2298-2310 `_load_w13` 做 narrow，tp=4 rank 0 拿 weight cols [0:320]、scale blocks [0,1,2]（load_shard_size = ceil(10/4) = 3）

### 关键不对称（根因）

rank 0 拿到的 scale[2] 是基于**全局列 [256:383]** 量化的（checkpoint 里原本 128 列真实数据），而 rank 0 实际 weight 只有 [0:320] 即 weight[256:319]（64 列）。这就是为什么 ATOM 必须把 weight zero-pad 到 384——让 kernel 的第 3 个 N-block 能与 scale[2] 对齐（scale[2] 覆盖 [256:383]，含 64 列真实数据 + 64 列零）。

### 修正后的结论

**原结论（过强）**：
> "必须从 BF16 原始权重重新量化"

**修正后的准确表述**：

| 需要做的事 | 是否必须 | 说明 |
|-----------|---------|------|
| 重新生成 per_1x64 的 scale tensor | **必须** | 将 10 个 per_1x128 N-block scale 拆为 20 个 per_1x64 N-block scale |
| 回到 BF16 重新量化权重 | **推荐但非必须** | 可直接从现有 FP8 权重做 scale rebinning（近似，有少量精度损失）；若要"无损"精度则必须 BF16 重算 |
| 修改 ATOM block_n=64 + 去掉 384 padding 路径 | **必须** | 否则 ATOM 仍会 narrow 到 ceil(320/64)=5 个 scale 但按 128 对齐 |

**Scale rebinning 的可行性**（数学）：
```
new_scale[2k]   = max(|w_fp8[128k : 128k+64]|) × old_scale[k] / FP8_MAX
new_scale[2k+1] = max(|w_fp8[128k+64 : 128k+128]|) × old_scale[k] / FP8_MAX
```
在已有 FP8 权重上重算，无需 BF16 原权重，但精度略低于 BF16→FP8 重量化。

### 额外发现：per_1x64 对 tp=8 同样无效

tp=8 → inter=160，160/64=2.5，**不整除**，per_1x64 仍需 padding 到 192（= 3×64）。
per_1x64 只对 inter_dim 是 64 整数倍的 TP 配置有效：
- tp=4 (inter=320=5×64) ✅ 无 padding
- tp=2 (inter=640=10×64) ✅ 无 padding，但本来 640/128=5 已无 padding，**净亏损**（scale 翻倍）
- tp=8 (inter=160，160/64=2.5) ❌ 仍需 padding

---

## 6. 性能影响估算

- **当前 padding 方案**：N 方向浪费 16.7%（320 vs 384）。teammate-4 § Q6 估算表：

  | 方案 | NPerBlock | NXdlPerWave | N blocks for N=320 | 浪费率 |
  |------|-----------|-------------|---------------------|--------|
  | 当前 (padding) | 128 | 2 | 3 (=384) | 16.7% |
  | per_1x64 | 64 | 1 | 5 (=320) | 0% |

- **方案 B 理论收益**：消除 16.7% N 维 GEMM；但 NXdlPerWave 2→1 可能降低 mfma ILP（K 方向流水线深度减半），具体退化幅度需 microbench。若退化 ≤16.7%，方案 B 净胜。
- **scale 开销**：tp=4 场景 N 方向 scale 数量 5/3 ≈ 1.67×；若同时改 K 方向 block 至 64，K 方向 scale 数量 2×，组合后 scale 总开销 ~3.3×（来源：teammate-5 § Q4）。
- **tp=4 TPOT 当前基线**：13ms（FP8 tp=4 V06 Exp2 实测，MEMORY.md）。
- 端到端收益估算：MoE GEMM 仅是 decode TPOT 的一部分，16.7% N 维节省的端到端 TPOT 改善应远小于 16.7%（具体比例**未实测**）。

---

## 7. 建议与后续

基于现有调研：

1. **短期建议：维持当前 padding 方案**。已 V06/V07 验证 PASS（FP8 tp=4 TPOT=13ms），且 per_1x128 + N=320 数学上不可能去 padding（第 3.3 节）。
2. **方案 A（语义去 padding）不推荐优先推进**：
   - C-store 天然 OK，B-scale 天然 OK；但 B-load 走 preshuffled 路径无 right_pad，且 GemmSpec=NPadding 对 blockscale 路径无效（teammate-2 § Q2）。
   - 即使放宽 `device.L448` 检查 + host zero-pad weight，相比当前方案净改善仅是"去掉 ATOM `_process_block_quant` 的 weight zero-fill"，**计算量不变**（kernel 仍跑 3×128 blocks）。
   - 收益微小：仅省去 host 侧 padding 的一次 zero-fill 内存开销，对运行时 TPOT 无影响。
3. **方案 B（per_1x64）评估为"中等可行但风险高"**：
   - CK gridwise 层无硬性阻拦（ScaleBlockN 是模板参数，无 NPerBlock=64 相关 static_assert）。
   - 主要工作量在 aiter wrapper（`gemm_moe_ck2stages_common_blockscale.cuh` 解硬编码 + `gen_instances.py` 新增 dispatch 分支）+ 模型重新量化【待 teammate-7 审查】。
   - 风险：NXdlPerWave=1 的 mfma 效率不确定；tp=2 场景净亏损（scale 翻倍且本无 padding）；checkpoint 重新量化需 lm_eval 重验精度。
4. **决策门槛**：除非 padding 浪费对端到端 TPOT 影响 >5%（当前未实测，建议先量化端到端代价），否则不投资方案 B。
5. **下一步实验建议**（如需推进）：
   - 在 CK 侧基于 `composable_kernel/example/65_gemm_multiply_multiply/moe_gemm1_xdl_fp8_blockscale.cpp` 改 `Scale_Block_N=64`，跑 N=320 microbench 验证编译可过 + 实测 vs NPerBlock=128+padding 的性能差。
   - 若 PoC 通过且性能正向，再启动 aiter dispatch 改造与量化 pipeline。

---

## 8. 附：已验证事实索引

| # | 事实 | 来源 |
|---|------|------|
| F1 | aiter blockscale `Scale_Block_N=128` 硬编码（stage1） | `aiter/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_blockscale.cuh:107`（teammate-1 Q2.3, teammate-4 Q1） |
| F2 | aiter blockscale `Scale_Block_N=128` 硬编码（stage2） | 同上文件 L407 |
| F3 | aiter blockscale `GemmSpec=Default`（stage1/stage2） | 同上文件 L83 / L380（teammate-1 Q2.1） |
| F4 | aiter dispatch stage1 `NPerBlock=128` 硬编码所有 block_m 分支 | `aiter/csrc/ck_gemm_moe_2stages_codegen/gen_instances.py:340-368`（teammate-1 Q3.1） |
| F5 | aiter dispatch stage2 `NPerBlock=128` 硬编码 | 同上 L739-770（teammate-1 Q3.2） |
| F6 | C++ 侧 N 来源：stage1 `w1.size(1)/2`、stage2 `w2.size(1)` | `aiter/csrc/.../moe_ck_2stages_kernel.cu:193 / 469`（teammate-1 Q3.4） |
| F7 | gridwise CheckValidity `N % NPerBlock` 在 BLayout=Col 时短路跳过 | `composable_kernel/.../gridwise_moe_gemm_blockscale.hpp:932-948`（teammate-1 Q1.3） |
| F8 | device 层 `arg.N % NPerBlock != 0 → return false` 无 GemmSpec 守护（**真正硬卡点**） | `composable_kernel/.../device_moe_gemm_blockscale.hpp:448`（teammate-2 Q4） |
| F9 | gridwise C-store 无条件 right_pad（M, N 维都 pad） | gridwise_moe_gemm_blockscale.hpp:634-639（teammate-2 Q1） |
| F10 | gridwise B 走 preshuffled `MakeBGridDescriptor_Preshuffled`，**不走** right_pad 路径 | 同文件 L1149-1150 vs L557-568（teammate-2 Q2） |
| F11 | gridwise B-scale grid descriptor = `[ceil(N/ScaleBlockN), ceil(K/ScaleBlockK)]` | 同文件 L1164-1168 / L1700-1702（teammate-1 Q1.1） |
| F12 | gridwise expert scale stride = `ceil(N/ScaleBlockN) * (IsInputGemm?2:1) * ceil(K/ScaleBlockK)` | 同文件 L1227-1229 / L1761-1763（teammate-1 Q1.2） |
| F13 | gridwise B-scale thread copy 起点 `block_n_id * NPerBlock / ScaleBlockN` | 同文件 L1418 / L1467（teammate-1 Q1.2） |
| F14 | ATOM `_process_block_quant` 只 pad weight、不 pad scale | `ATOM/atom/model_ops/moe.py:1709-1749`（teammate-1 Q6.1） |
| F15 | ATOM scale 分配 `(intermediate_size_per_partition + block_n - 1) // block_n` | 同文件 L1636 |
| F16 | ATOM 量化映射表无 64 | `ATOM/atom/quant_spec.py:114-119`（teammate-5 Q2） |
| F17 | aiter QuantType 枚举无 per_1x64 | `aiter/csrc/include/aiter_enum.h:15-25`（teammate-5 Q3） |
| F18 | aiter `fc_scale_blkn=128, fc_scale_blkk=128` 写死 partial | `aiter/aiter/fused_moe.py:468-474`（teammate-5 Q3） |
| F19 | Step-3.5-Flash-FP8 checkpoint `weight_block_size=[128,128]` | `/root/.cache/.../Step-3.5-Flash-FP8/.../config.json:313-321`（teammate-5 Q1） |
| F20 | gridwise 无 `static_assert(NPerBlock % ScaleBlockN == 0)`，无 N partial/tail 处理 | 全文件 grep 0 命中（teammate-1 Q5、teammate-4 Q4） |
| F21 | gen_instances.py L431 注释 "KPerBlock=64 ... fp8 mfma limitation" 仅针对 K 维，**不限制 NPerBlock=64** | `aiter/csrc/.../gen_instances.py:431`（teammate-4 Q3） |
| F22 | per_1x128 + N=320 数学不可去 padding（gcd(320,128)=64） | 数学证明（teammate-5 Q5） |
| F23 | N=320 surrogate microbench（zero-tail + intermediate_pad=0）跑通无 crash | `/home/hanchang/project_moe_no_padding/microbench_n320.py` + logs（teammate-3） |
| F24 | wrapper 层（aiter `fused_moe`）无 N % 128 强校验，bug 屏障在 CK 模板层 | teammate-3 关键结论 |
| F25 | FP8 tp=4 V06 Exp2 实测 TTFT=86ms / TPOT=13ms / gmu=0.7 | MEMORY.md |
| F26 | checkpoint expert weight 全局 shape=(288, 1280, 4096)，dtype=float8_e4m3fn（未 TP partition） | safetensors 直读 model-00018.safetensors（teammate-7 审查 1/2） |
| F27 | checkpoint scale shape=(288, 10, 32)，N_scale=10=ceil(1280/128)（全局） | 同上（teammate-7 审查 1） |
| F28 | ATOM `_load_w13` L2298-2310：tp=4 rank 0 拿 scale blocks [0,1,2]（load_shard_size=ceil(10/4)=3），scale[2] 基于全局列 [256:383] 量化 | `ATOM/atom/model_ops/moe.py:2298-2310`（teammate-7 审查 3） |
| F29 | per_1x64 对 tp=8（inter=160，160/64=2.5 不整除）同样无效，仍需 padding 到 192 | 数学推导（teammate-7 审查 4，额外发现） |

---

## 已验证 vs 未验证 标记说明

- 上文 F1-F25 全部为"已验证事实"（来自代码行号或实测）。
- 未验证假设（汇总）：
  1. C-store v7r3_scatter + right_pad 的 mask 行为是 CK 通用约定，未直接读 v7r3_scatter 内部源码（teammate-2 Q1）。
  2. host 侧 expert weight 实际分配大小是按 N 还是按 NPadded（teammate-2 Q2）—— 影响方案 A 改造路径。
  3. CK buffer-load 在越界 N 维（第 3 个 N-block 多读 64 列）的硬件行为（buffer-resource mask）（teammate-2 Q2）。
  4. CShuffleBlockTransferScalarPerVector_NPerBlock 实际 tuning 值（teammate-2 Q4）。
  5. 真正物理 N=320 直接喂 CK kernel 是否 crash（teammate-3）—— 无法从 Python 触发，需改 CK。
  6. NPerBlock=64, NXdlPerWave=1 的 fp8 mfma 实际吞吐退化幅度（teammate-4 Q6）。
  7. gridwise L1556-1558 N0×...×N4=NPerBlock 在 NPerBlock=64 下能否找到合法分解（teammate-4 Q4）。
  8. per_1x64 vs per_1x128 在 Step-3.5-Flash 实际权重上的精度差异（teammate-5 Q4）。
  9. ATOM 第二份 `Fp8MoEMethodCutlass`（moe.py:1517+）是否在 Step-3.5-Flash-FP8 实际路径上（teammate-5 Q4）。
  10. gridwise blockscale L1500-2200 段（IsInputGemm=false）scale 索引行为是否与 IsInputGemm=true 对称（teammate-1 未验证假设）。
  11. FP8-to-FP8 scale rebinning（从 per_1x128 拆为 per_1x64）的实际精度损失量级（teammate-7 未验证假设 2）。
  12. per_1x64 + tp=4 下 ATOM `_load_w13` 是否确实走非 padding 路径（需代码验证，teammate-7 未验证假设 1）。
