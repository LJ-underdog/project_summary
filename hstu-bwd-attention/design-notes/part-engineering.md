# HSTU bwd —— 第 3 部分「工程落地 & 验证」设计（pane-3 / architect）

> 范围:文件/目录结构、problem/traits/tile、dispatch、dQ 写回两路、instances/codegen、CMake、测试验证、分阶段里程碑、风险。
> 算法细节(7-stage / 5-GEMM / dsilu)归 pane-1;mask 几何与参数语义归 pane-2。本文**消费**二者接口,文末列依赖假设。
> 所有行号/结构均按当前源码核实(2026-06-04)。

---

## 0. 源码现状基线(已核实)

| 事项 | 现状 |
|---|---|
| HSTU fwd 文件 | 全在 `example/ck_tile/18_hstu_attention/`(平铺,**无** `codegen/` 子目录) |
| HSTU 实例生成 | `generate_instances.py`(手写模板字符串,**非** FMHA 那套 `codegen/ops/*.py` dataclass 体系) |
| HSTU params | `hstu_attention_params.hpp`:`HstuAttentionNoGroupFwdParams` + `HstuAttentionGroupFwdParams`,**无任何 bwd 字段** |
| fwd dispatch | `hstu_attention_{batched,jagged,group}_forward_dispatch.hpp`,入口 `run_<mode>_forward_causal_softmax_bias_dropout_dispatch<dtype,kUseCausal,kUseSoftmax,kHasBias,kHasDropout,MaxK>` |
| fwd API 接缝 | `hstu_attention_{no_group,group}_forward_{bf16,fp16}.cpp` → `BOOL_SWITCH` + `HDIM_SWITCH` → run dispatch;声明在 `hstu_attention_api.hpp` |
| fwd tile | `hstu_attention_tile_setting_define.hpp`(`HstuAttentionFwdTileSettingClass`,BlockTile=`sequence<kM0,kN0,kN0Sub,kN1,kK1,kQKHeaddim>`)+ `hstu_attention_fwd_setting.hpp`(per-MaxK 预设) |
| FMHA bwd 基建 | `/root/ck/.../ops/fmha/`:`pipeline/block_fmha_bwd_{dot_do_o,convert_dq,dq_dk_dv_pipeline_kr_ktr_vr,pipeline_problem,pipeline_default_policy,pipeline_enum}.hpp` + `kernel/fmha_bwd_kernel.hpp`(3 个 kernel) |
| oracle | `reference_hstu_attention_bwd.hpp`(855 行):`reference_no_group_hstu_attention_bwd<...>` + `reference_group_hstu_attention_bwd<...>` |

> 重要:HSTU 走的是**自写 dispatch + 简单 `generate_instances.py`** 路线,**不**搬 FMHA 的 `codegen/ops/fmha_bwd.py`(那套依赖 `cpp_symbol_map`/`cmake_config`,与 HSTU 例子隔离)。bwd 一律沿 HSTU 风格扩展,**只复用 FMHA 的 device 端 pipeline/kernel 模板**。

---

## 1. 文件/目录结构 —— 新建 vs 复用 FMHA 映射表

新增文件全部落在 `example/ck_tile/18_hstu_attention/`,**镜像 fwd 命名**。

### 1.1 新建文件清单

| 新建文件 | 作用 | 蓝本(镜像/复用) |
|---|---|---|
| `hstu_attention_bwd_type_config.hpp` | bwd dtype 配置(Q/K/V/dO/dQ/dK/dV/D/LSE/Acc/Gemm)| 镜像 `hstu_attention_fwd_type_config.hpp` + FMHA `FmhaBwdTypeConfig` |
| `hstu_attention_bwd_tile_setting_define.hpp` | `HstuAttentionBwdTileSettingClass`(9 元 BlockTile)| 借 FMHA `TileFmhaBwdShape` 字段语义 |
| `hstu_attention_bwd_setting.hpp` | per-MaxK × {SiLU,softmax} bwd tile 预设 + mtile chooser | 镜像 `hstu_attention_fwd_setting.hpp` |
| `hstu_attention_bwd_pipeline_problem.hpp` | `HstuAttentionBwdPipelineProblem` + PRE/POST sub-problem | 镜像 `hstu_attention_pipeline_problem.hpp`;字段抄 FMHA `BlockFmhaBwdPipelineProblem` |
| `hstu_attention_bwd_traits.hpp` | `HstuAttentionBwdTraits`(kPad* + kBlockPerCu)| 镜像 `hstu_attention_traits.hpp` |
| `hstu_attention_no_softmax_bwd_pipeline.hpp` | **SiLU 路** MAIN pipeline(HSTU 特化:重算 S→dsilu,masked-out 显式 0)| 改写自 FMHA `block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp` |
| `hstu_attention_with_softmax_bwd_pipeline.hpp` | **softmax 路** MAIN pipeline(读 LSE/D,`dS=P*(dP-D)`)| 同上,更贴近 FMHA 原版 |
| `hstu_attention_bwd_dot_do_o_pipeline.hpp` | PRE:D=rowsum(O⊙dO),**仅 softmax 路启用** | thin wrapper / 直接复用 FMHA `BlockFmhaBwdOGradDotO` |
| `hstu_attention_bwd_convert_dq_pipeline.hpp` | POST:deterministic 时归约 dq_acc→dQ | thin wrapper / 直接复用 FMHA `BlockFmhaBwdConvertQGrad` |
| `hstu_attention_bwd_kernel.hpp` | 3 个 kernel 包装(PRE/MAIN/POST)+ HSTU MakeKargs/GridSize | 改写自 FMHA `fmha_bwd_kernel.hpp` |
| `hstu_attention_{batched,jagged,group}_backward_dispatch.hpp` | 三套 bwd dispatch | 镜像 `*_forward_dispatch.hpp` |
| `hstu_attention_{no_group,group}_backward_{bf16,fp16}.cpp` | bwd API 接缝(BOOL/HDIM switch)| 镜像 `*_forward_{bf16,fp16}.cpp` |
| `instances/hstu_attention_*_backward_*.cpp`(生成) | bwd 显式实例化 | `generate_instances.py` 扩展生成 |
| `example_hstu_attention_bwd.cpp`(或合并进 fwd 主程序) | bwd 端到端 + 对拍 | 镜像 `example_hstu_attention_fwd.cpp` |

### 1.2 直接复用 FMHA(只 `#include`,不复制)

| 复用 FMHA 文件 | 复用方式 |
|---|---|
| `pipeline/block_fmha_bwd_pipeline_default_policy.hpp` | **整体继承**(见 §2.4);差异点用 HSTU 派生 policy 覆写少量 `Make*Distribution` |
| `pipeline/block_fmha_bwd_pipeline_enum.hpp` | 直接用 `BlockFmhaBwdPipelineEnum::KRKTRVR` |
| `pipeline/block_fmha_bwd_dot_do_o.hpp` | softmax 路 PRE 直接实例化(D 公式与 HSTU 一致) |
| `pipeline/block_fmha_bwd_convert_dq.hpp` | POST 直接实例化(纯归约+cast,与激活无关) |
| `pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp` | 作为 MAIN **结构蓝本**;HSTU 改写激活/mask/scale/索引段(不直接 include) |
| `pipeline/tile_fmha_shape.hpp::TileFmhaBwdShape` | bwd shape 类直接复用(9 元 BlockTile 语义对齐) |

> 结论:**PRE/POST/policy/shape/enum 几乎零成本复用,MAIN 必须 HSTU 特化**(激活双路 + 5 因子 mask + scale_p + jagged/group 索引)。这与 BRIEF §复用策略主线一致。

---

## 2. problem / traits / tile_setting

### 2.1 `HstuAttentionBwdPipelineProblem`

抄 FMHA `BlockFmhaBwdPipelineProblem`(`block_fmha_bwd_pipeline_problem.hpp:10-65`)的字段骨架,叠加 HSTU 特化布尔与命名约定(对齐 fwd 的 `HstuAttentionFwdPipelineProblem`):

```cpp
template <typename InOutDataType_,     // Q/K/V/O/dO/dQ/dK/dV (bf16/fp16)
          typename GemmAccDataType_,   // float
          typename CompDataType_,      // float, dsilu / D / LSE 计算
          typename DDataType_,         // float, PRE 产物(仅 softmax 路)
          typename LSEDataType_,       // float, fwd 产物(仅 softmax 路)
          bool kIsCrossAttention_,
          bool kUseGroup_,             // group 模式(蕴含 jagged)
          bool kIsJagged_,
          bool kHasBias_,              // 与 fwd 对称(MVP 可恒 false)
          bool kHasCausal_,
          bool kUseSoftmax_,           // 切 SiLU / softmax 两路
          bool kIsDeterministic_,      // 切 dQ 写回两路(§4)
          typename AttentionBwdTileSetting_>
struct HstuAttentionBwdPipelineProblem {
    // dtypes: QKV=dO=dQ=dK=dV=InOutDataType; D/LSE/Acc=float; GemmDataType=InOutDataType
    static constexpr bool kUseSoftmax       = kUseSoftmax_;
    static constexpr bool kIsDeterministic  = kIsDeterministic_;
    // ... kIsJagged/kUseGroup/kHasCausal/kIsCrossAttention 同 fwd
    static_assert(!kUseGroup || kIsJagged, "Group HSTU 仅用于 jagged");
    using HstuAttentionBwdTileSetting = remove_cvref_t<AttentionBwdTileSetting_>;
    static constexpr index_t kBlockSize = AttentionBwdTileSetting_::NumWarps * get_warp_size();
    // GetQ/K/V/dO DramTileAccessMaxVectorSize() —— 复用 fwd problem 里 detail::GetDramTileAccessMaxVectorSize 范式
};
```

**与 FMHA problem 的取舍**:
- 砍掉 FMHA 的 `RandValOutputDataType / FmhaDropout / BiasGradDataType / kHasBiasGrad`(HSTU bwd MVP 无 dropout、无 dbias);保留接缝但默认关。
- 新增 `kUseSoftmax`(FMHA 无此轴,固定 softmax)、`CompDataType`(dsilu)、`kUseGroup`/`kIsJagged`/`kIsCrossAttention`(HSTU 的多模式)。
- `mask` 不进 problem 模板(fwd 也没有);mask 在 dispatch 阶段由 `HstuBlockMasking` 选型后传给 MAIN pipeline(沿用 fwd 做法,**消费 pane-2 接口**)。

PRE/POST sub-problem 直接复用 FMHA 的 `BlockFmhaBwdOGradDotOPipelineProblem` / `BlockFmhaBwdConvertQGradPipelineProblem`(`block_fmha_bwd_pipeline_problem.hpp:67-123`),仅把 dtype/`kIsGroupMode`/`kPad*` 接到 HSTU 的 trait。

### 2.2 `HstuAttentionBwdTraits`

镜像 `hstu_attention_traits.hpp`,bwd 多一个 hdim 维度(qk≠v):

```cpp
template <bool kPadSeqLenQ_, bool kPadSeqLenK_,
          bool kPadHeadDimQK_, bool kPadHeadDimV_, index_t kBlockPerCu_>
struct HstuAttentionBwdTraits { /* 同 fwd 字段 */ };
```
PRE 只需 `kPadSeqLenQ/kPadHeadDimV`;POST 只需 `kPadSeqLenQ/kPadHeadDimQK`(对齐 FMHA sub-problem 的 trait 取用)。

### 2.3 tile setting(`HstuAttentionBwdTileSettingClass`)

bwd 的 GEMM 拓扑与 fwd 不同(5 GEMM),**采用 FMHA 的 9 元 BlockTile 语义**(`tile_fmha_shape.hpp:98-112`):

```
BlockTile = sequence<kM0, kN0, kK0, kK1, kK2, kK3, kK4, kQKHeaddim, kVHeaddim>
  kM0 : q seqlen tile        kN0 : k seqlen tile
  kK0 : gemm0 (Q@Kᵀ=S)  unroll      kK1 : gemm1 (Pᵀ@dO=dV) unroll
  kK2 : gemm2 (dO@Vᵀ=dP) unroll     kK3 : gemm3 (dSᵀ@Q=dK) unroll
  kK4 : gemm4 (dS@K=dQ) unroll
  kQKHeaddim, kVHeaddim  (hdim_qk ≠ hdim_v 在此分离)
```
配 5 组 warp 划分(GSdP/GdKV/GdQ),直接借 FMHA `TileFmhaBwdShape` 的 11-参构造(block_warps0/1/2 + warp_tile0/1)。

**预设来源(`hstu_attention_bwd_setting.hpp`)**:
- 第一阶段**直接照搬 FMHA bwd 的 gfx942/gfx950 预设**(per-hdim 64/96/128/256),按 `BUILD_HSTU_FOR_GFX95_ONLY` 宏分叉(与 fwd 的 trload 分叉同构)。FMHA 已为这些 hdim 调优过 bwd tile,先验最稳。
- mtile chooser(`get_hstu_attention_bwd_mtile`)第一阶段可恒定(bwd 网格按 kN0 切 seqlen_k,不像 fwd 需要 splitkv;见 §6 GridSize),后续再按 (batch,nhead,seqlen) 调。
- HSTU 特有约束:`kSubQKHeaddim % kN1 == 0` 之类沿用 fwd 的 `ceil_to_qualified_tile_length`;`kQKHeaddim % kK0 == 0`(FMHA static_assert)。

### 2.4 policy:直接继承 FMHA bwd default policy

`BlockFmhaBwdPipelineDefaultPolicy`(86KB)封装了全部 GEMM 的 warp/LDS 分布、向量化、转置(`kr_ktr_vr` = K-resident / Kᵀ-resident / V-resident)。**HSTU MAIN pipeline 默认 `using Policy = ck_tile::BlockFmhaBwdPipelineDefaultPolicy;`**,只在必须处用 `struct HstuBwdPolicy : BlockFmhaBwdPipelineDefaultPolicy { ... 覆写 ... }`。预期**几乎无需覆写**——HSTU 与 FMHA 的 5 GEMM 形状一致,差异在 element-wise(激活/mask/scale)而非 GEMM 分布。可行性见 §9 风险条目。

---

## 3. dispatch —— 三套 backward,fwd 对称

镜像 `hstu_attention_batched_forward_dispatch.hpp` 结构(`struct ..._dispatch` + `Run` + `RunWithKernel` + 自由函数 `run_*`)。

### 3.1 入口签名(与 fwd 对称,去掉 dropout 轴 MVP)

```cpp
template <typename InOutDataType, bool kUseCausal, bool kUseSoftmax,
          bool kHasBias, bool kIsDeterministic, ck_tile::index_t MaxK>
void run_batched_backward_dispatch(HstuAttentionNoGroupBwdParams& param, hipStream_t stream);
// jagged / group 同构;group 用 HstuAttentionGroupBwdParams
```

### 3.2 `Run` 内部流程(三段式 launch)

```
Run(param, stream):
  1. 选 tile setting(kUseSoftmax ? WithSoftmaxBwd : NoSoftmaxBwd)
  2. 算 pad: pad_seqlen_q / pad_seqlen_k / pad_hdim_qk / pad_hdim_v
  3. BOOL_SWITCH_4(pad...) → HstuAttentionBwdTraits
  4. 用 HstuBlockMasking<...>::type 选 mask(消费 pane-2)
  5. 组 problem → 选 pipeline:
        kUseSoftmax ? HstuAttentionWithSoftmaxBwdPipeline : HstuAttentionNoSoftmaxBwdPipeline
  6. 组 3 个 kernel:
        PRE  = HstuAttentionBwdOGradDotOKernel   (仅 kUseSoftmax 时 launch)
        MAIN = HstuAttentionBwdDQDKDVKernel<Pipeline, dKEpilogue, dVEpilogue>
        POST = HstuAttentionBwdConvertQGradKernel (仅 kIsDeterministic 时 launch)
  7. 顺序 launch:[PRE?] → MAIN → [POST?]
```

### 3.3 实例化矩阵收敛(关键:控制爆炸)

bwd 轴:`mode(3) × dtype(2) × causal(2) × softmax(2) × bias(2) × deterministic(2) × maxk(4)` = **384**(fwd 是 96)。收敛手段:

| 轴 | MVP | 收敛策略 |
|---|---|---|
| bias | 恒 `false` | HSTU bwd 暂不支持 dbias;砍一半 → 192 |
| deterministic | 恒 `false`(atomicAdd)| 仅在用户显式开 deterministic 时编 → 默认砍一半 → 96 |
| dropout | 去除该轴 | bwd 不支持 dropout(已不在模板) |
| maxk | 先 {64,128} | 覆盖主流 hdim,后补 {96,256} |
| softmax | 保留 | 业务双路都要 |

→ **MVP 实例数 ≈ mode(3)×dtype(2)×causal(2)×softmax(2)×maxk(2) = 48**;完整态 192~384(bias/deterministic 按需)。与 fwd 同量级,编译可控。

---

## 4. dQ 写回两条路

dQ 的难点:MAIN kernel 按 **seqlen_k tile(kN0)** 切网格(每个 block 负责一段 K,遍历全部 Q),因此**多个 block 都会向同一段 dQ 累加** → 写冲突。两条路(对齐 FMHA,`fmha_bwd_kernel.hpp` 的 `kIsDeterministic` 分叉):

### 4.1 路 A:atomicAdd(默认,`kIsDeterministic=false`)

- MAIN 内 dQ 直接 `atomic_add` 写回 `param.dq_ptr`(float 累加或经 epilogue)。
- **无** dq_acc workspace,**无** POST kernel。
- 缺点:atomic 在 bf16/fp16 不可直接累加 → 通常 dQ 累加用 float 中转或 atomicAdd on float dq buffer 再 cast。FMHA 的做法:dq_acc(float)在 device 上 atomic;HSTU MVP 可令 `dq_ptr` 指向 float 临时 buffer + 末尾 cast,或沿用 FMHA 的 dq_acc(float, nsplits=1)+ POST convert-only(见 4.3)。**推荐 MVP 用「float dq_acc + atomicAdd + POST convert-only」**,实现最省心且与 deterministic 路共用 workspace 布局。

### 4.2 路 B:deterministic split-workspace + convert_dq(`kIsDeterministic=true`)

- 每个 K-tile(split)写**独立**的 dq_acc slice(无 atomic,逐位可复现)。
- POST kernel `BlockFmhaBwdConvertQGrad` 的 **Reduce+Convert** 重载(`block_fmha_bwd_convert_dq.hpp:64-138`)沿 split 维归约 `nsplits` 份 → cast → `dq_ptr`。
- 逐位可复现:归约顺序固定(do-while 累加),无浮点 atomic 不确定性。

### 4.3 `dq_acc` workspace 分配与 stride

dq_acc 形状(对齐 FMHA `dq_acc` 语义,`fmha_bwd_kernel.hpp:129/1105/1121/1168`):

```
deterministic: [nsplits, total_seqlen_q, nhead, hdim_qk]  (float)
  nsplits = integer_divide_ceil(max_seqlen_k, kN0)   // K-tile 数 = MAIN grid.x
atomicAdd : [total_seqlen_q, nhead, hdim_qk]          (float, nsplits 维退化为 1)
```
stride 字段(进 params,见 §5):`stride_dq_acc`(seq)、`nhead_stride_dq_acc`、`batch_stride_dq_acc`(batched)、`split_stride_dq_acc`(deterministic,= 单 split 的总元素数)。

分配位置:host 端 dispatch 前 `hipMalloc`(对齐 fwd splitkv 的 `o_acc_ptr/lse_acc_ptr` 临时 buffer 模式,见 fwd params 末尾),调用后释放。host 计算 `nsplits` 并填 `param.num_splits`(复用 fwd 已有字段名习惯)。

### 4.4 与 params 联动

```
host: if kIsDeterministic: nsplits = ceil(max_seqlen_k/kN0); alloc dq_acc[nsplits,...]
      else                : nsplits = 1; alloc dq_acc[1,...] (float 中转)
MAIN: 按 i_n0(=blockIdx.x*kN0) 决定写哪个 split / 或 atomic 到 split-0
POST: deterministic→Reduce+Convert(nsplits); atomic 路→Convert-only(单份 cast)
```

---

## 5. params 扩展(消费 pane-2,但工程字段在此列全)

在 `hstu_attention_params.hpp` 新增 `HstuAttentionNoGroupBwdParams` / `HstuAttentionGroupBwdParams`(**继承或并列** fwd params,复用全部 Q/K/V 指针、stride、mask 超参、scale_s/attn_scale、num_targets_ptr、jagged offsets)。bwd 专属新增:

```cpp
struct HstuAttentionNoGroupBwdParams /* : 复用 fwd 的全部输入字段 */ {
  // ---- bwd 输入 ----
  const void* o_ptr;     // fwd 产物
  const void* do_ptr;    // 上游梯度
  const void* lse_ptr;   // 仅 kUseSoftmax;SiLU 路为 nullptr
  // ---- bwd 输出 ----
  void* dq_ptr; void* dk_ptr; void* dv_ptr;
  // ---- PRE 产物(仅 softmax 路)----
  void* d_ptr;                          // D=rowsum(O⊙dO)
  ck_tile::index_t seq_stride_d, nhead_stride_d, batch_stride_d;
  // ---- dQ 归约 workspace(§4)----
  void* dq_acc_ptr;
  ck_tile::index_t stride_dq_acc, nhead_stride_dq_acc, batch_stride_dq_acc, split_stride_dq_acc;
  int  num_splits;                      // = nsplits(deterministic)/1(atomic)
  bool is_deterministic;
  // dO/dQ/dK/dV 各自的 seq/nhead/batch stride(与 q/k/v 对称,可复用 fwd 同名或新增)
  ck_tile::index_t seq_stride_do, nhead_stride_do, batch_stride_do, /* dq/dk/dv 同理 */ ;
};
```
> 字段语义与 alpha/scale_p/contextual/window/min_full/num_target 的精确含义**以 pane-2 为准**;本文只保证工程上「指针 + stride + workspace + nsplits」齐备且与 kernel MakeKargs 对齐。

---

## 6. kernel 包装与 GridSize(`hstu_attention_bwd_kernel.hpp`)

镜像 FMHA `fmha_bwd_kernel.hpp` 的三 kernel,改 MakeKargs/索引以吃 HSTU params:

| kernel | GridSize | 说明 |
|---|---|---|
| PRE `BwdOGradDotO` | `(ceil(seqlen_q, kBlockSize), nhead, batch)`(FMHA `:1732`)| 仅 softmax 路;算 D |
| MAIN `BwdDQDKDV` | `(ceil(seqlen_k, kN0), nhead, batch)`(FMHA `:1064-1068`)| 每 block 一段 K,遍历全 Q |
| POST `BwdConvertQGrad` | `(ceil(seqlen_q, kM0), nhead, batch)`(FMHA `:2009`)| 仅 deterministic;归约 dq_acc |

- group 模式:grid.z = num_batch,kernel 内用 `seqstart_q/k_ptr[i_batch]` 求 per-batch offset 与真实 seqlen(FMHA `:1110-1159` 的 group 分支可几乎照搬),并按 `seqlen_k <= i_n0` early-return。
- jagged 非 group:同 group 的 offset 机制但**单组超参**(window/contextual/... 来自标量 param 而非 per-group 数组)。
- `HSTU_SCHED_BATCH_AS_FIRST_GRID_DIM` 之类的 grid 维序优化沿用 fwd CMake 宏(可选)。

`MakeKargs` 把 §5 的指针/stride/scale/mask 超参/dq_acc 全量透传(对齐 batched_forward_dispatch 的 `MakeKargs` 长参列表风格)。

---

## 7. instances + codegen(扩展 `generate_instances.py`)

沿用 HSTU 的字符串模板路线,新增 `create_backward_instances` / `create_backward_instances_ref`,与 fwd 对称:

```python
HSTU_BACKWARD_INSTANCE_TEMPLATE = """
{extern}template void run_{mode}_backward_dispatch<
    {dtype},{has_causal},{use_softmax},{has_bias},{is_deterministic},{max_k}>(
    HstuAttention{group_or_not}BwdParams& param, hipStream_t stream);
"""
HSTU_BACKWARD_INSTANCE_FNAME = ("hstu_attention_{mode}_backward_{dtype_str}_"
    "{causal}_{softmax}_{bias}_{determ}_{maxk}.cpp")

def create_backward_instances(out, headdims):
  for mode in ["batched","jagged","group"]:
    for dtype in ["fp16","bf16"]:
      for has_causal in [True,False]:
        for use_softmax in [True,False]:
          for has_bias in [False]:          # MVP 砍 bias
            for is_determ in [False]:        # MVP 砍 deterministic(默认)
              for max_k in headdims:         # MVP: [64,128]
                ...write one .cpp...
```

- 命名规范:`hstu_attention_<mode>_backward_<dtype>_<has|no_causal>_<softmax_true|false>_<has|no_bias>_<deterministic|ndeterministic>_maxk_<hd>.cpp`(直接套 fwd 的 `BOOL_MAP_*`,新增 `BOOL_MAP_DETERM`)。
- ref 头文件(`*_backward_<dtype>_instances_ref.hpp`)被 API 接缝 `.cpp` include,提供 `extern template` 声明,避免重复实例化。
- **编译规模/时间**:MVP 48 instance(§3.3),bwd 单 instance 比 fwd 重(5 GEMM,~37KB pipeline)。预估单文件编译 1.5~3× fwd instance。建议:① `generate_instances.py` 每次先清空 `instances/`(fwd 已这么做,`:175-179`);② CMake `file(GLOB)` 自动纳入(§8);③ 完整态(192+)用 ninja `-j` 并行 + 按需开 bias/deterministic 子集,避免一次全编。

---

## 8. CMake

最小改动(`CMakeLists.txt` 已 `file(GLOB INSTANCE_SRCS instances/*.cpp)`,bwd instance 自动纳入)。新增:

```cmake
# 新 bwd example target(与 fwd 并列,EXCLUDE_FROM_ALL)
set(EXAMPLE_HSTU_ATTENTION_BWD "tile_example_hstu_attention_bwd")
set(BWD_INTERFACES_SRCS
    hstu_attention_no_group_backward_bf16.cpp hstu_attention_no_group_backward_fp16.cpp
    hstu_attention_group_backward_bf16.cpp    hstu_attention_group_backward_fp16.cpp)
# 复用同一 instances/ glob(fwd+bwd 混在 INSTANCE_SRCS)
add_executable(${EXAMPLE_HSTU_ATTENTION_BWD} EXCLUDE_FROM_ALL example_hstu_attention_bwd.cpp)
target_sources(${EXAMPLE_HSTU_ATTENTION_BWD} PRIVATE ${BWD_INTERFACES_SRCS} ${INSTANCE_SRCS})
target_include_directories(${EXAMPLE_HSTU_ATTENTION_BWD} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(${EXAMPLE_HSTU_ATTENTION_BWD} PRIVATE ${EXAMPLE_HSTU_ATTENTION_COMPILE_OPTIONS})
```
- gfx95 / gfx94 分叉、`-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3`、`RULE_MESSAGES OFF` 全沿用 fwd 现有逻辑。
- 可选:也可把 bwd 合进现有 `example_hstu_attention_fwd.cpp` 加 `--bwd` CLI(省一个 target),但**独立 target 编译更快、隔离更清**,推荐独立。

### 8.1 example 主程序 CLI(bwd 增量参数)

复用 fwd 的全部 CLI(`example_hstu_attention_fwd.cpp:90-125`),新增:
```
--deterministic 0   # dQ 写回:0=atomicAdd, 1=split-workspace 逐位可复现
--bwd_v 1           # 是否做 bwd CPU 对拍
--dump_grad 0       # dump dQ/dK/dV(device & ref)
```
dO 在 host 随机初始化(同 q/k/v 的 `seed`/`norm_dist`);O/LSE 由 GPU fwd 现算(端到端),或 reference fwd 现算(纯 bwd 单测)。

---

## 9. 测试与验证(重点)—— 以 reference_*_hstu_attention_bwd 为 oracle

### 9.1 对拍主流程

```
随机 seed → 生成 Q/K/V/dO (+ jagged offsets / num_targets / 各 mask 超参)
  ├─ GPU 路:fwd kernel 产 O(+LSE if softmax) → bwd kernel(PRE→MAIN→POST)产 dQ/dK/dV
  └─ CPU 路:reference fwd 产 O/LSE → reference_{no_group|group}_hstu_attention_bwd 产 dQ*/dK*/dV*
对比 (dQ,dK,dV) vs (dQ*,dK*,dV*)
```
> oracle 签名已核实:`reference_no_group_hstu_attention_bwd<InOutDataType,GemmAccDataType,CompDataType,kIsJagged,kUseSoftmax,kUseCausal>::Run(is_cross, q,k,v,lse,o,do, dq,dk,dv, num_batch, alpha, attn_scale, max_seqlen_q, max_seqlen_kv, seq_q_offsets, seq_kv_offsets, num_targets, contextual_seqlen, window_size, min_full_attn_seqlen)`;group 版多 `num_batch_per_group` 与 5 个 `group_*` 数组(`reference_hstu_attention_bwd.hpp:65,503`)。**SiLU 路 lse 不参与**(可传空 tensor)。

### 9.2 容差(bf16/fp16)

bwd 误差比 fwd 大(5 GEMM 串联 + dsilu/exp 重算 + dQ 跨 block 累加)。建议:

| dtype | 量 | 指标 | 阈值(初值,按实测收紧)|
|---|---|---|---|
| bf16 | dQ/dK/dV | rel-err(‖x−x*‖/‖x*‖)| ≤ 2e-2 |
| bf16 | dQ/dK/dV | max abs-err(归一化后)| ≤ 5e-2 |
| fp16 | dQ/dK/dV | rel-err | ≤ 5e-3 |
| fp16 | dQ/dK/dV | max abs-err | ≤ 1e-2 |

- 用 `ck_tile::check_err`(host_tensor 工具),分 dQ/dK/dV 各报 max-err / mean-err。
- dQ 通常误差最大(atomic/归约累加);dV 最小。阈值**分张量**设。
- masked-out 位置:SiLU 路必须**显式 0**(`silu(0)=0` 非自然零);对拍要专门校验 masked-out 区 dS=0 → 不污染 dK/dV(对应 BRIEF/pane-2 的关键差异 1)。

### 9.3 测试矩阵

```
激活:    {SiLU, softmax}
模式:    {batched, jagged, group}
mask 因子: {no-mask, causal, +window(local_len), +contextual(context_len),
            +min_full(minfull_len), +num_target(targets), 组合}
dtype:    {bf16, fp16}
hdim:     {(64,64),(128,128),(128,64 即 qk≠v),(256,256)}
dQ 路:    {atomicAdd, deterministic}
```
笛卡尔积很大 → 用**分层抽样**:核心组合(SiLU×batched×causal×bf16×64)全跑;其余轴各与核心做「单轴变更」覆盖,再加少量全组合 smoke。

### 9.4 deterministic 逐位可复现

`--deterministic 1` 同输入跑两遍 → dQ **逐位 bitwise 相等**(`memcmp`);atomic 路不保证(仅数值容差)。这是 deterministic 路独有验收。

### 9.5 边界用例

seqlen 非 tile 整除(测 pad)、seqlen_q≠seqlen_kv(cross)、单 batch、空 target、window=0、contextual=0、hdim_qk≠hdim_v、jagged 各组长度差异大(测 group early-exit)。

---

## 10. 分阶段里程碑(MVP → 完整)

| 阶段 | 范围 | 验收标准 |
|---|---|---|
| **M0 脚手架** | params bwd 字段、3 kernel 空壳、dispatch、CMake bwd target、`generate_instances.py` bwd 分支、CLI | 编译过;dQ/dK/dV 全 0 输出,launch 不崩;PRE/POST 占位 |
| **M1 端到端打通** | **batched + SiLU + no-mask + bf16 + atomicAdd + hdim64**;MAIN 跑通 5 GEMM + dsilu + dQ(float dq_acc + POST convert-only)| 对拍 dQ/dK/dV 过 §9.2 bf16 阈值;单组合 PASS |
| **M2 mask 因子** | 逐个加 causal → window → contextual → min_full → num_target → 组合(消费 pane-2 mask)| 每加一因子,对拍 PASS;masked-out 区 dS=0 校验 PASS |
| **M3 jagged** | jagged 单组超参 + cu_seqlens 索引 | jagged 对拍 PASS;非整除 seqlen pad PASS |
| **M4 group** | group 模式 per-group 超参数组、grid.z=batch、early-exit | group 对拍 PASS(用 `reference_group_*`)|
| **M5 softmax 路** | PRE(D)启用 + LSE 读取 + `dS=P*(dP-D)`;`with_softmax_bwd_pipeline` | softmax 对拍 PASS;消费 fwd 的 kStoreLSE 产物 |
| **M6 deterministic** | split-workspace + POST Reduce+Convert | 逐位可复现 PASS;数值同 atomic 路 |
| **M7 多 dtype/maxk** | fp16 + hdim{96,128,256} + qk≠v | 全 §9.3 矩阵抽样 PASS |
| **M8 性能/收尾** | tile 调优、occupancy、perf 计时、(可选)bias/dbias | perf 数 vs FMHA bwd 同 hdim 不显著落后;实例规模可控 |

> 关键路径:**M1 是风险闸门**——它一次性验证「FMHA MAIN pipeline 能否被 HSTU 特化(SiLU 重算 S + masked-out 显式 0 + scale_p)」且「FMHA policy 直接复用」。M1 通过则后续多为组合扩展。

---

## 11. 风险 / 未决

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| 1 | **FMHA bwd policy 直接复用可行性**:`BlockFmhaBwdPipelineDefaultPolicy`(86KB)的 tile 分布是否兼容 HSTU 在 GEMM 间插入 dsilu/mask 的 element-wise(需要中间 S/dP 以特定 distribution 驻留)| 高 —— 决定 MAIN 改写量 | M1 先验:若 distribution 不匹配,派生 `HstuBwdPolicy` 覆写 `MakeShuffled*`/中间 reg 分布;最坏情况复制 MAIN pipeline 全文再改 |
| 2 | **模板组合爆炸**(384 全态)| 中 —— 编译时间/产物体积 | §3.3 收敛:bias/deterministic/dropout 默认关;maxk 分批;ninja 并行;ref 头 `extern template` |
| 3 | **编译时间**:bwd instance ~3× fwd | 中 | 每次清 `instances/`;按需子集编;CI 只编核心抽样 |
| 4 | **dq_acc workspace 显存**:deterministic 下 `[nsplits, Σseqlen_q, nhead, hdim_qk]` float,长序列爆显存 | 中 | atomic 路为默认(nsplits=1);deterministic 仅显式开;可加 nsplits 上限/分块 |
| 5 | **SiLU masked-out 显式 0**:STAGE5 必须对 masked-out 写 0 而非依赖 -inf | 高(正确性)| M2 专项校验;pane-1 提供 dS 写 0 的 stage 伪码,pane-2 提供 mask predicate |
| 6 | **hdim_qk ≠ hdim_v**:bwd 5 GEMM 中 dV/dQ 走不同 headdim,tile 的 kQKHeaddim/kVHeaddim 分离需校验 | 中 | tile setting 已分离(§2.3);M7 专测 (128,64) |
| 7 | **PRE 是否必要(SiLU 路)**:SiLU 路不需 D,但 atomic-only 也可不要 POST | 低 | dispatch 按 `kUseSoftmax` 跳 PRE、按 `kIsDeterministic` 跳 POST(§3.2) |
| 8 | **HSTU 无 FMHA codegen 体系**:不能直接搬 `fmha_bwd.py` | 低 | 用 `generate_instances.py` 字符串模板(§7),已确认风格一致 |

---

## 12. 对 pane-1(算法)/ pane-2(mask & 参数)的依赖假设

**依赖 pane-1(算法)提供:**
- MAIN pipeline 的 7-stage / 5-GEMM 调度骨架与寄存器驻留方案(决定 §2.4 policy 是否需覆写)。
- SiLU 路 STAGE2「重算 S」与 STAGE5「`dS=dP*scale_p*dsilu(S)`、masked-out 显式 0」的 stage 级伪码。
- softmax 路 STAGE5「`dS=P*(dP-D)`」、P=exp(S−LSE) 的 stage 级伪码、PRE 的 D 公式确认(应与 FMHA `BlockFmhaBwdOGradDotO` 一致)。
- dQ 在 MAIN 内以 float 累加 / 写 split 的具体 stage(决定 §4 与 dq_acc 的对接点)。

**依赖 pane-2(mask & 参数)提供:**
- `HstuBlockMasking<...>` 选出的 mask 类型(`HstuCrossAttentionBlockMaskWithLocal<kUseCausal>` 等,`hstu_block_masking.hpp:12/268/503/656`)及其 device 端 tile-range / predicate 接口——MAIN/PRE 直接消费,本文不重设计。
- `alpha`(=params `scale_s`,QK 缩放)、`scale_p`(=params `attn_scale`,SiLU 输出缩放)、`contextual_seqlen`/`window_size`/`min_full_attn_seqlen`/`num_target` 的精确语义与默认值(`scale_p` 默认 1/max_seqlen_q)。
- group 模式 per-group 超参数组的内存布局(`group_*_ptr`,对齐 fwd `HstuAttentionGroupFwdParams:125-129`),供 §5 bwd group params 与 §6 group kernel 索引复用。
- bwd kargs 中 mask 相关字段清单(本文 §5/§6 预留透传位,字段名以 pane-2 为准)。

**本文对外保证(供 pane-1/2 消费):**
- 文件/target/instance/CMake/CLI 的落地骨架与命名;params 的指针/stride/workspace/nsplits 完整字段;3-kernel GridSize 与 launch 顺序;dQ 两路写回机制与 dq_acc 布局;对拍流程、容差、测试矩阵、分阶段验收。
