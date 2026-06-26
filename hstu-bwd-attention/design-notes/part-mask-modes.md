# Part 2 — HSTU bwd: mask / 模式 / 索引 / 参数设计

> 范围:HSTU 特有的 5 因子 mask 在 bwd 的接入、jagged/group/batch 三模式与索引、bwd 参数结构、scale 语义。
> 算法阶段(STAGE 1-7 GEMM/SiLU/softmax)归 pane-1;文件/codegen/instances/CMake/测试归 pane-3。本文末给出对二者的接口假设。
> 事实以下列源码为准:`hstu_block_masking.hpp`、`reference_hstu_attention_bwd.hpp`、`hstu_attention_params.hpp`、`/root/ck/include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp`、`.../block/block_masking.hpp`、`.../pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp`。

---

## 0. 关键架构判定(贯穿全文的前提)

**FMHA bwd MAIN kernel 是 KV 外循环 / Q 内循环。** 实证:
- `FmhaBwdDQDKDVKernel::GridSize = dim3(ceil(seqlen_k/kN0), nhead, batch)`,grid.x 沿 **seqlen_k** 分块(`fmha_bwd_kernel.hpp:1063-1068`)。每个 block 固定一个 KV 块 `i_n0 = i_tile_n*kN0`(L1096),把 `dk_acc/dv_acc` 留在寄存器里、对 sq 累加,最后一次性 epilogue 写回(L1523-1581)。
- MAIN pipeline 内 `mask.GetTileRangeAlongY(k_origin_y, kM0, kN0)` 求出该 KV 块要覆盖的 **seqlen_q tile 范围** `[seqlen_q_start, seqlen_q_end)`,`num_total_loop = ceil((end-start)/kM0)`,`if(num_total_loop<=0) 整块 early-exit`,然后 `while` 沿 Q 走 tile;tile 内用 `mask.IsEdgeTile(...)` 决定是否逐像素 check(`block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp:160-168,481-559`)。

这与 CPU reference 完全同构:reference 也是固定 `sk` 维上 `dk_acc[sk]/dv_acc[sk]` 累加器、外层 `for sq`、内层 `for sk`(`reference_hstu_attention_bwd.hpp:227-465`),只是 reference 把 sq/sk 两层都展开成标量循环。**HSTU bwd 采用同样的 KV 外 / Q 内布局。**

**由此产生的头号缺口(本文 §1 的核心):**
HSTU 的 `HstuCrossAttentionBlockMaskWithLocal` 等四个 mask 类**只实现了 `GetTileRangeAlongX(i_y, …)`**(给定一个 **seqlen_q tile 起点 i_y**,返回该行 tile 要算的 **seqlen_k 区间** `[x_start,x_end)`)。这是 **fwd 方向**(Q 外 / KV 内,fwd dispatch 正是这么用的)。
HSTU mask **没有 `GetTileRangeAlongY`**(给定 seqlen_k tile 求 seqlen_q 区间),**也没有 `IsEdgeTile`**(只有 `IsFullTileInsideMask` + 标量 `IsTokenPairInsideMask`)。

bwd 走 KV 外循环就需要"反方向"的 tile 范围。三条路:
- **方案 A(推荐):给 HSTU mask 新增 `GetTileRangeAlongY` + `IsEdgeTile` 两个成员**(纯几何推导,见 §1.1/§1.2),bwd pipeline 直接调用,拿到与 FMHA 同构的接口。
- 方案 B:bwd 内层 Q 从 `[0, seqlen_q)` 全扫,靠 `IsFullTileInsideMask`(已存在)做"整块在 mask 内 → 免逐像素",靠 `IsTokenPairInsideMask` 兜底逐像素。**正确但不省 tile**(window/contextual 稀疏时白算大量空 tile),仅作为 A 落地前的 fallback。
- 方案 C:复用 FMHA 的 `GenericAttentionMask` 而非 HSTU mask。**否决**——FMHA generic mask 只有 causal+左右 window 两参,无法表达 contextual_seqlen / min_full_attn_seqlen / num_target 这三个 HSTU 业务因子(见 §1.4)。

> 决策:**方案 A**。新增两个 device 成员函数到现有四个 HSTU mask struct,签名见 §1.1/§1.2。这是纯新增、不改 fwd 行为,风险低。pane-1 的 MAIN pipeline 据此写。

---

## 1. 5 因子 HSTU mask 在 bwd 的接入

### 1.0 五因子回顾(来自 `hstu_block_masking.hpp` 构造体)
以 `HstuCrossAttentionBlockMaskWithLocal`(L11-265)为代表,构造入参 `(is_tile_in_first_split, seqlen_q, seqlen_k, contextual_seqlen, max_attn_len(=window_size), min_full_attn_seqlen, num_target)`,内部派生:
- `max_q_uih_len = seqlen_q - num_target`、`max_k_uih_len = seqlen_k`(假定 target 不在 KV,L49-50)。
- `max_attn_len = min(max_k_uih_len, min(max_q_uih_len, window_size))`(window 上限收缩,L53)。
- `contextual_seqlen = min(contextual_seqlen, max_q_uih_len - min_full_attn_seqlen)`(contextual 与 min_full 不重叠,L57)。
- `diff_q_kv_len = max_k_uih_len - max_q_uih_len`、`max_row_id += diff_q_kv_len`(cross-attn 行列错位补偿,L70-71)。

`IsMasking` 谓词维度(causal 与否分支,L216-232):
1. **causal**:`((row_id>col_id)||(row==col)) && ((row_id-col_id<=max_attn_len)||in_min_full_scope)`。
2. **non-causal(window)**:`(abs(row_id-col_id)<=max_attn_len)||in_min_full_scope`。
3. `in_min_full_scope = min_full_attn_seqlen>0 && row_id >= max_row_id - min_full_attn_seqlen`(L218-219)。
4. **contextual**:首行/首列特例 `if(row_id==diff_q_kv_len && col_id<max_col_id) return true`(L203-204),即 contextual token 对全行可见。
5. **num_target**:通过 `max_q_uih_len = seqlen_q - num_target` 进入,target 行(`sq >= max_q_uih_len`)走"完整可见"分支(`GetTileRangeAlongX` 的 `i_y >= max_q_uih_len` 段,L97-101/129-135/164-167)。

bwd **不需要重新发明这些谓词**——`IsTokenPairInsideMask(sq,sk)` 已经把 5 因子全编码进去了,且 reference bwd 的 STAGE1/STAGE4 正是直接调用它(`reference_...bwd.hpp:246,380`)。bwd 只需补"反方向 tile 范围"和"边缘 tile 判定"。

### 1.1 新增 `GetTileRangeAlongY`(KV-block → seqlen_q tile 区间)

语义:给定当前 KV tile 左上角列 `i_x`(= `i_n0`),返回需要遍历的 seqlen_q 行 tile 区间 `[y_start, y_end)`,tile 对齐;`y_end<=y_start` 时调用方整块 early-exit。

推导是 `GetTileRangeAlongX` 的转置(把"行 tile 求列区间"翻成"列 tile 求行区间")。要点:
- mask 的有效行列关系由 `row_id - col_id`(causal/window 是 `<=max_attn_len`,window 还有 `>= -max_attn_len`)+ contextual 首行特例 + min_full 全可见区 + target 区共同决定。
- cross-attn 的 `diff_q_kv_len` 偏移、`is_tile_in_first_split=false`(min_full 区)分支、contextual 列 `< contextual_seqlen` 的"对所有行可见"都要镜像处理。

骨架(以 cross + local 为例,与 L107-183 对称;`physical row = row_id - diff_q_kv_len`):
```cpp
template <index_t YTile, index_t XTile>
CK_TILE_DEVICE constexpr auto
GetTileRangeAlongY(index_t i_x, number<YTile>, number<XTile>) const
{
    // i_x: 当前 KV tile 起始列(seqlen_k 维)。返回 [y_start, y_end) (seqlen_q 维)
    if (!is_tile_in_first_split) {
        // min_full 区:该 KV 列对 [某下界, seqlen_q) 行可见 → y_end = seqlen_q
        // causal: 行需 row+diff_q_kv_len >= col → y_start = align_down(max(i_x - diff_q_kv_len,0))
        index_t y_start = kUseCausal ? align_down(max(i_x - diff_q_kv_len, 0), YTile) : 0;
        return make_tuple(y_start, seqlen_q);
    }
    if constexpr (kUseCausal) {
        // 窗口下界: col - max_attn_len <= row+diff  → row >= i_x - diff - (XTile?) ...
        index_t y_start = align_down(max(i_x - diff_q_kv_len - 0, 0), YTile);
        // 上界: causal row+diff >= col_end-1, 且 target 行(>=max_q_uih_len)恒可见 → 收到 seqlen_q
        index_t y_end   = seqlen_q;               // target 段保守取满;可再按 col 收紧
        // contextual: 若该 KV 列落在 contextual 列区, y_start 收到 0
        if (i_x < contextual_seqlen) y_start = 0;
        return make_tuple(y_start, y_end);
    } else { // window (non-causal)
        index_t y_lo = align_down(max(i_x - diff_q_kv_len - max_attn_len, 0), YTile);
        index_t y_hi = min(i_x - diff_q_kv_len + XTile + max_attn_len, seqlen_q);
        y_hi = align_up(y_hi, YTile);
        if (i_x < contextual_seqlen) y_lo = 0;       // contextual 列对全列行可见
        if (min_full_attn_seqlen > 0) y_hi = seqlen_q; // min_full 全可见
        return make_tuple(y_lo, y_hi);
    }
}
```
> 注:上面是**结构骨架**,精确边界(对齐方向、target 段是否能收紧、self vs cross 的 `diff_q_kv_len=0`)需对照 `GetTileRangeAlongX` 各分支逐一转置后用 CPU reference 的 `IsTokenPairInsideMask` 做"范围内必含所有 true 像素"的离线校验(见 §5 风险)。**正确性铁律:`GetTileRangeAlongY` 返回的区间必须是 `IsTokenPairInsideMask` 真值集在 Y 方向的超集**;宁可放宽(多算 tile),不可漏(漏算 = 梯度错)。self-attn 版只是 `diff_q_kv_len=0` 的特化。

### 1.2 新增 `IsEdgeTile`(整块判定,决定是否逐像素)

复用现成 `IsFullTileInsideMask`(L236-264,causal/non-causal 两分支已给)取反即可,无需新写复杂逻辑:
```cpp
template <index_t TileHeight, index_t TileWidth>
CK_TILE_DEVICE constexpr bool
IsEdgeTile(index_t i_tile_top /*sq*/, index_t i_tile_left /*sk*/,
           number<TileHeight>, number<TileWidth>) const
{
    // 整块都在 mask 内 → 非 edge,免逐像素;否则需要逐像素 IsTokenPairInsideMask
    return !IsFullTileInsideMask(i_tile_top, i_tile_left,
                                 number<TileWidth>{}, number<TileHeight>{});
}
```
> 注意 `IsFullTileInsideMask` 形参顺序是 `(i_tile_top, i_tile_left, TileWidth, TileHeight)`(L237-240),与 FMHA `IsEdgeTile(i_y,i_x,TileHeight,TileWidth)` 形参顺序不同,封装时统一成 FMHA 习惯。`IsFullTileInsideMask` 当前只在 `!is_tile_in_first_split` 时给 true(L246/258),即"非 min_full 区一律按 edge 逐像素"——保守正确但偏慢;若 profiling 需要可再补"窗口内整块"判定,但**首版按现状即可,正确优先**。

### 1.3 masked-out 显式置 0(SiLU 路径必须,本部分最易错点)

数学根因(BRIEF L31、reference 注释 L34-37):SiLU 路径 `P = silu(S)*scale_p`,而 `silu(0) = 0/(1+e^0) = 0` → 看似零,但 **masked-out 的 S 在 SiLU 路径被显式设成 `0`(reference L276),`silu(0)*scale_p = 0`**——这一项 P 恰好是 0,所以 **STAGE2 的 dV 累加天然不被污染**。真正的坑在 **dS**:`dsilu(0) = sigmoid(0)*(1+0) = 0.5 ≠ 0`。若不显式屏蔽,masked-out 位置会得到 `dS = dP*scale_p*0.5 ≠ 0`,污染 dQ/dK。reference 用 `if(mask.IsTokenPairInsideMask) … else locals_dS=0`(L380-385)。

GPU 落地(与 mask 谓词协同的具体做法):
- **STAGE2 P 置 0**:S-recompute tile(STAGE1)算出 `S` 后,对 edge tile 调 `IsTokenPairInsideMask` 生成布尔 tile `m`;masked-out 位 `S := 0`(SiLU 路径)或 `S := -inf`(softmax 路径,使 `exp(S-LSE)=0`)。这步与 fwd 完全同构,可直接搬 fwd 的 mask-apply 接口(pane-1 接 fwd 的 `set_tile_if`/predicate)。
- **STAGE5 dS 置 0**:`dS = dP*scale_p*dsilu(S)` 之后(或之中),对同一布尔 tile `m` 做 `dS = m ? dS : 0`。**必须用与 STAGE1 同一份谓词**(同 sq,sk,同 mask 对象),否则两处不一致会漏屏蔽。
- **整块 vs 边缘**:`IsEdgeTile==false`(整块在 mask 内)时跳过置 0,直接全算;只有 edge tile 才生成 `m` 并屏蔽。这正是 §1.2 `IsEdgeTile` 的用途。
- **softmax 路径**:masked-out 靠 `S=-inf → P=exp(-inf-LSE)=0`(reference L294 注释)自然为 0,dS=P*(dP-D) 也自然为 0,**无需显式置 0**;但 STAGE1 仍需把 masked-out 的 S 设成 -inf(置 0 与置 -inf 的二选一由 `kUseSoftmax` 决定,reference L275-279)。

> 给 pane-1 的接口假设:mask-apply 谓词以 `mask.IsTokenPairInsideMask(sq, sk)` 为唯一真值源,STAGE1 与 STAGE5 共用同一布尔 tile;置零常量在 SiLU 路径为 `0.0f`、softmax 路径 STAGE1 为 `-inf`。

### 1.4 contextual / min_full / num_target 在 bwd 的语义对称性

bwd 与 fwd **完全对称**,无需为 bwd 重定义这三个因子:
- **同一个 mask 对象、同一套构造参数**:reference bwd 的 mask 构造块(L167-225)与 fwd 逐字相同(`make_hstu_*_block_mask_*`,L800-856),包括 `is_tile_in_first_split` 的判定 `seqlen_q - num_target > min_full_attn_seqlen ? min_full : seqlen_q-num_target`(L177-194)。GPU bwd dispatch 必须复刻这段构造逻辑。
- **contextual**:首行/首列全可见特例(L203-204)在 bwd 中通过同一个 `IsTokenPairInsideMask` 体现于 dS 屏蔽;dV/dK 对 contextual 列的累加随之正确。语义对称。
- **min_full_attn_seqlen**:决定 `is_tile_in_first_split`,bwd 的 `GetTileRangeAlongY` 在 `!is_tile_in_first_split` 分支要把 y_end 放到 `seqlen_q`(min_full 区行全可见,§1.1)。语义对称。
- **num_target**:`max_q_uih_len = seqlen_q - num_target`,target 行(`sq>=max_q_uih_len`)完整可见。bwd 仍用 `num_targets_ptr` 取 per-batch num_target(同 fwd)。语义对称。
- 唯一 bwd 特有动作:把 fwd 的"Q 外 / KV 内"翻成"KV 外 / Q 内"(§0),这是 **tile 调度**的差异,**不是 mask 语义**的差异。

---

## 2. 三模式 + 索引

三模式与 fwd 一致(`hstu_attention_{jagged,group,batched}_forward_dispatch.hpp` 的对应物)。bwd 三个 kernel(PRE/MAIN/POST)各自 grid 沿 batch×nhead×seq-tile 排布(对照 FMHA 三个 GridSize:MAIN 沿 seqlen_k,PRE/POST 沿 seqlen_q,`fmha_bwd_kernel.hpp:1063,1731,2008`)。

### 2.1 batch 模式(dim0 = num_batch)
- 张量布局 `[num_batch, seqlen, nhead, hdim]`(bshd)。seqlen_q/seqlen_kv 是标量(`HstuAttentionNoGroupFwdParams::seqlen_q/seqlen_kv`)。
- 索引:`batch_offset_* = i_batch * batch_stride_*`(FMHA batch 分支 `fmha_bwd_kernel.hpp:1163-1170`)。每个 (i_batch,i_nhead) 取自己的 `num_target = num_targets_ptr[i_batch]`。
- grid:MAIN `dim3(ceil(seqlen_kv/kN0), nhead, num_batch)`;PRE/POST `dim3(ceil(seqlen_q/kM0), nhead, num_batch)`。

### 2.2 jagged 模式(dim0 = 1 + cu_seqlens)
- 张量 `[1, total_seq, nhead, hdim]`,每 batch 段长由 `seq_q_offsets[b+1]-seq_q_offsets[b]` 给出(reference L159-162)。
- 索引:`query_start = seq_q_offsets[i_batch]`、`key_start = seq_kv_offsets[i_batch]`;
  `batch_offset_q = query_start*stride_q`、`batch_offset_k = key_start*stride_k` …(FMHA group 分支 `fmha_bwd_kernel.hpp:1110-1152` 即此结构)。
  访问元素:`tensor(0, seq_off[b]+s, h, k)`(reference L253-256/817)→ GPU 即 `ptr + (off+s)*seq_stride + h*nhead_stride + k`。
- 段内 seqlen 在 device 上现取:`seqlen_q = seqstart_q[i_batch+1]-seqstart_q[i_batch]`(FMHA L1142-1152);
  **per-block early-exit**:`if(seqlen_kv <= i_n0) return;`(MAIN,FMHA L1156-1159),PRE/POST 用 `seqlen_q<=i_m0`。
- HSTU jagged 与 FMHA group 的索引完全同构,差别仅在多了 mask 5 因子(标量,来自 params 全局字段)。

### 2.3 group 模式(每段独立超参)
- 同样 jagged-packed(dim0=1,`reference_group_...bwd` L528),但**每个 group 一套独立超参**:`group_attn_scales / group_contextual_seqlens / group_window_sizes / group_min_full_attn_seqlens / group_max_seqlens_q`(reference L519-523)。
- group 解析:`i_group = i_batch / num_batch_per_group`(reference L579),再 `group_xxx[i_group]` 查表(L585-591)。GPU 上 `num_batch_per_group = num_batch / num_group`。
- 这些 per-group 超参在 GPU 上的取法是 group 模式的**主要新增复杂度**(见 §5 风险):它们是 device 指针(fwd 已是 `group_*_ptr`,`HstuAttentionGroupFwdParams` L125-129),kernel 内按 `i_group` 索引读出标量再构造 mask + 算 scale_p。

### 2.4 bwd 的 (batch,head) → seqlen/基址 计算(汇总)
| 模式 | seqlen_q/kv 来源 | base offset | num_target | mask 超参 |
|---|---|---|---|---|
| batch | params.seqlen_q/kv(标量) | `i_batch*batch_stride` | `num_targets_ptr[i_batch]` | 全局标量 `window/contextual/min_full` |
| jagged | `seqstart[b+1]-seqstart[b]` | `seqstart[b]*seq_stride` | `num_targets_ptr[i_batch]` | 全局标量 |
| group | `seqstart[b+1]-seqstart[b]` | `seqstart[b]*seq_stride` | `num_targets_ptr[i_batch]` | `group_*_ptr[i_group]` |

grid.z = num_batch,grid.y = nhead,grid.x = seq-tile(MAIN 沿 kv,PRE/POST 沿 q)。`GetTileIndex` 直接 `(blockIdx.x, blockIdx.y, blockIdx.z)`(FMHA L1070-1077),沿用。

### 2.5 hdim_qk ≠ hdim_v 的影响
布局上 Q/K/dQ/dK 用 `hdim_qk`,V/O/dO/dV 用 `hdim_v`(reference assert L134-141)。bwd 各张量维度:
- **dV[sk, hdim_v]** ← `P^T @ dO`,dO 是 `hdim_v` 列(STAGE2,reference L320 `for k<hdim_v`)。dV DRAM view 用 `hdim_v`(FMHA L1556-1568 `kVHeaddim`)。
- **dP[sq,sk]** ← `dO @ V^T`,沿 `hdim_v` 收缩(STAGE3,L342)。
- **dQ[sq, hdim_qk]** ← `alpha * dS @ K`,K 是 `hdim_qk` 列(STAGE5,L420 `for k<hdim_qk`)。
- **dK[sk, hdim_qk]** ← `alpha * dS^T @ Q`(STAGE6,L452)。
- **D[sq]**(softmax 路径)= `dO·O`,沿 `hdim_v`(L392)。
影响到 kargs 与 tile 设定:GEMM0/1(QK^T、dO@V^T)的 K 维分别是 `hdim_qk`、`hdim_v`;dQ/dK epilogue 写 `hdim_qk` 列、dV 写 `hdim_v` 列。pane-3 的 traits/tile-setting 需为 `kQKHeaddim` 与 `kVHeaddim` 各开一档(FMHA `BlockFmhaShape::kQKHeaddim/kVHeaddim` 已分离,L94-97)。pane-1 的 5-GEMM 收缩维按上表接线。

---

## 3. bwd 参数结构设计

现状:`hstu_attention_params.hpp` **只有 fwd 两个 struct,无任何 bwd 字段**。新增两个 bwd struct,镜像 fwd 命名 + 复用 fwd 已有字段,新增梯度指针/stride 与 bwd 专用标量。

### 3.1 `HstuAttentionNoGroupBwdParams`(batch + jagged 共用,`is_jagged` 区分)
```cpp
struct HstuAttentionNoGroupBwdParams
{
    // ---- 复用 fwd 已有(语义不变) ----
    bool is_cross_attention;
    bool is_jagged;
    ck_tile::index_t num_batch;
    ck_tile::index_t seqlen_q;        // batched only
    ck_tile::index_t seqlen_kv;       // batched only
    const void* seq_q_offsets_ptr;    // jagged only
    const void* seq_kv_offsets_ptr;   // jagged only
    ck_tile::index_t max_seqlen_q;    // scale_p 默认值 & dq_acc 切分都要用

    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* o_ptr;                // bwd 需要 O(softmax 路径算 D=dO·O)
    ck_tile::index_t hdim_qk;
    ck_tile::index_t hdim_v;
    ck_tile::index_t num_head;

    const void* num_targets_ptr;
    bool use_causal;
    ck_tile::index_t window_size;
    ck_tile::index_t contextual_seqlen;
    ck_tile::index_t min_full_attn_seqlen;
    bool use_softmax;

    // fwd 的 seq/nhead/batch stride 复用(q/k/v/o)
    ck_tile::index_t seq_stride_q, seq_stride_k, seq_stride_v, seq_stride_o;
    ck_tile::index_t nhead_stride_q, nhead_stride_k, nhead_stride_v, nhead_stride_o;
    ck_tile::index_t batch_stride_q, batch_stride_k, batch_stride_v, batch_stride_o; // batched only

    // ---- bwd 新增:梯度 I/O 指针 ----
    const void* do_ptr;               // dO (grad of O)
    void* dq_ptr;                     // 最终 dQ (hdim_qk)
    void* dk_ptr;                     // dK     (hdim_qk)
    void* dv_ptr;                     // dV     (hdim_v)
    const void* lse_ptr;              // softmax 路径才用 (kUseSoftmax)
    void* d_ptr;                      // PRE pass 产物 D=rowsum(O⊙dO); softmax 路径用
    void* dq_acc_ptr;                 // MAIN 累加 dQ(fp32);POST 归约成 dq_ptr

    // ---- bwd 新增:各梯度的 stride ----
    ck_tile::index_t seq_stride_do, seq_stride_dq, seq_stride_dk, seq_stride_dv, seq_stride_dq_acc;
    ck_tile::index_t nhead_stride_do, nhead_stride_dq, nhead_stride_dk, nhead_stride_dv;
    ck_tile::index_t nhead_stride_lsed;     // LSE 与 D 同布局(每 (b,h,sq) 一个标量)
    ck_tile::index_t nhead_stride_dq_acc;
    ck_tile::index_t batch_stride_do, batch_stride_dq, batch_stride_dk, batch_stride_dv; // batched
    ck_tile::index_t batch_stride_lsed, batch_stride_dq_acc;                             // batched

    // ---- bwd 新增:标量/开关 ----
    float alpha;        // QK 缩放 (= fwd scale_s);STAGE1/STAGE5/6
    float attn_scale;   // SiLU 输出缩放原值;scale_p = attn_scale ? attn_scale : 1/max_seqlen_q
    // 运行期 scale_p 由 kernel 算(见 §4),不必单独存
    int num_splits;     // deterministic 时 dq_acc 的 KV-split 数
    // 模板侧 kUseSoftmax / kIsDeterministic / kIsJagged 通过 instance 模板参数表达,
    // 不必进 struct(与 FMHA 一致:这些是编译期);此处可留 bool 便于 host 选 instance。
    bool kIsDeterministic;
};
```

### 3.2 `HstuAttentionGroupBwdParams`(group 模式)
在 §3.1 基础上去掉单标量 mask 超参,改为 per-group 指针(镜像 `HstuAttentionGroupFwdParams` L124-129):
```cpp
struct HstuAttentionGroupBwdParams
{
    bool is_cross_attention;
    ck_tile::index_t num_group;
    ck_tile::index_t num_batch;
    const void* seq_q_offsets_ptr;
    const void* seq_kv_offsets_ptr;
    ck_tile::index_t max_seqlen_q;          // 所有 group max_seqlen_q 的最大值(dq_acc 分配/切分)

    const void* q_ptr,*k_ptr,*v_ptr,*o_ptr;
    const void* do_ptr;
    void *dq_ptr,*dk_ptr,*dv_ptr,*d_ptr,*dq_acc_ptr;
    const void* lse_ptr;
    ck_tile::index_t hdim_qk, hdim_v, num_head;
    const void* num_targets_ptr;
    bool use_causal, use_softmax, kIsDeterministic;

    // per-group 超参(device 指针,kernel 内按 i_group 取)
    const void* group_attn_scale_ptr;            // → scale_p
    const void* group_max_seqlen_q_ptr;
    const void* group_window_size_ptr;
    const void* group_contextual_seqlen_ptr;
    const void* group_min_full_attn_seqlen_ptr;

    // strides:q/k/v/o/do/dq/dk/dv/dq_acc 的 seq+nhead;lse/d 的 nhead;(group 无 batch_stride)
    ck_tile::index_t seq_stride_q,seq_stride_k,seq_stride_v,seq_stride_o,
                     seq_stride_do,seq_stride_dq,seq_stride_dk,seq_stride_dv,seq_stride_dq_acc;
    ck_tile::index_t nhead_stride_q,nhead_stride_k,nhead_stride_v,nhead_stride_o,
                     nhead_stride_do,nhead_stride_dq,nhead_stride_dk,nhead_stride_dv,
                     nhead_stride_lsed,nhead_stride_dq_acc;
    float alpha;
    int num_splits;
};
```

### 3.3 kargs 设计
host-side `params` → device-side `kargs` 的转换沿用 FMHA `MakeKargs`(`fmha_bwd_kernel.hpp:305-1061`)。HSTU 三个 kernel 各一套 kargs:
- **PRE(dot_do_o)kargs**:`{o_ptr, do_ptr, d_ptr, seqlen_q, hdim_v, stride_o/do, nhead_stride_*}` + batch/group 分支(FMHA `FmhaBwdOGradDotOKernel::Kargs` L1632-1664)。HSTU 删掉 `p_undrop`(无 dropout,固定 1.0)或保留传 1.0。
- **MAIN(dq_dk_dv)kargs**:见下表与 FMHA 对应。
- **POST(convert_dq)kargs**:`{dq_acc_ptr, dq_ptr, seqlen_q, seqlen_k, hdim_qk, stride_dq/dq_acc, nhead_stride_*, split_stride_dq_acc(deterministic)}`(FMHA `FmhaBwdConvertQGradKernel::Kargs` L1894-1936)。

### 3.4 与 FMHA `fmha_bwd_args` 的对应 / 差异表(MAIN kernel)
| FMHA `FmhaBwdCommonKargs` 字段 | HSTU bwd 对应 | 差异 |
|---|---|---|
| `q/k/v_ptr, do_ptr, dk/dv_ptr, dq_acc_ptr` | 同名 | 一致 |
| `lse_ptr, d_ptr` | 同名 | HSTU 仅 `kUseSoftmax` 用;SiLU 路径 lse/d 可空 |
| `seqlen_q, seqlen_k, hdim_q, hdim_v` | `seqlen_q, seqlen_kv, hdim_qk, hdim_v` | 改名;hdim_qk≠hdim_v 已是 FMHA 支持的 |
| `num_head_q, nhead_ratio_qk` | `num_head`, (HSTU 无 MQA/GQA → ratio=1) | HSTU 暂无 GQA,`nhead_ratio_qk=1` |
| `raw_scale, scale` | `alpha`, `alpha*log2e`(softmax)/ 不需要(SiLU) | **HSTU 多一个 `scale_p`**(FMHA 无),见 §4 |
| `stride_*` / `nhead_stride_*` | 同名(seq_stride_* 对应) | 一致 |
| `FmhaBwdMaskKargs{window_left,window_right,mask_type}` | **替换为 HSTU 5 因子** `{window_size, contextual_seqlen, min_full_attn_seqlen, num_targets_ptr, use_causal}` | **核心差异**:FMHA 只有左右 window;HSTU 用自有 mask struct |
| `FmhaBwdDropoutKargs` | **删除** | HSTU bwd 无 dropout |
| `FmhaBwdCommonBiasKargs / BiasGrad` | **删除** | HSTU 无 bias/dbias(fwd `bias_ptr` 实际未用) |
| `FmhaBwdDeterministicKargs{split_stride_dq_acc}` | 保留 | 一致 |
| group: `seqstart_q/k_ptr, seqlen_k_ptr` | `seq_q/kv_offsets_ptr` | 改名;HSTU group 另加 `group_*_ptr` |
| (无) | **新增 `scale_p` 或 `attn_scale+max_seqlen_q`** | HSTU 特有 |
| (无) | **新增 group `group_attn_scale_ptr` 等 5 个** | HSTU group 特有 |

**净结论:删 dropout+bias+dbias 三组 kargs,mask kargs 整体换成 HSTU 5 因子(batch 标量 / group 指针),新增 `scale_p` 通路;其余字段与 FMHA 一一对应。**

---

## 4. scale 语义(`alpha` vs `scale_p`)

两个独立缩放,**不可混淆**(BRIEF L28、reference L165/270-271/286/439):

| 名称 | 含义 | 默认/来源 | fwd 用处 | bwd 用处 |
|---|---|---|---|---|
| `alpha` | QK^T 缩放(占 FMHA 的 `1/√d` 位) | = fwd `scale_s`(`HstuAttention*FwdParams::scale_s`) | `S = alpha*Q·K^T` | STAGE1 重算 S 乘 `alpha`(L270-271);**STAGE5 dQ 末乘 `alpha`**(L439);**STAGE6 dK 末乘 `alpha`**(L474/843) |
| `scale_p` | SiLU 输出缩放 | `attn_scale ? attn_scale : 1.0f/max_seqlen_q`(L165;group:L587) | `P = silu(S)*scale_p` | STAGE2 P 含 `scale_p`(经 locals_P);**STAGE4 dS 乘 `scale_p`**(SiLU 路径,L382);softmax 路径 **不用 scale_p** |

bwd 传递方式:
- `alpha`:host 直接存进 kargs(标量,batch/jagged 全局;group 仍是全局 `alpha`,reference group 的 `alpha` 是单标量入参 L515,**不随 group 变**)。
- `scale_p`:
  - batch/jagged:kernel 内 `scale_p = (attn_scale!=0) ? attn_scale : 1.0f/max_seqlen_q`(用 kargs 的 `attn_scale` + `max_seqlen_q`),或 host 预先算好直接传 `scale_p`。**推荐 host 预算传标量**,避免 kernel 内分支。
  - group:**per-group**,`scale_p = group_attn_scale[i_group] ? group_attn_scale[i_group] : 1/group_max_seqlen_q[i_group]`(reference L585-587)。必须 kernel 内按 `i_group` 取(device 指针),不能 host 预算单值。
- softmax 路径:`scale_p` 完全不参与(P=exp(S-LSE),dS=P*(dP-D));但 `alpha` 仍用于 S 重算与 dQ/dK 末缩放。FMHA 把 `scale` 在 MAIN 内折进 log2e 做 exp2,HSTU softmax 路径可照搬;SiLU 路径不需要 log2e。

---

## 5. 风险 / 未决问题

1. **`GetTileRangeAlongY` 的精确边界(最高风险)。** §1.1 只给了骨架。四个 mask 类 × causal/window × is_tile_in_first_split × contextual/target 多分支,转置极易错。**缓解**:(a) 先按"放宽超集"实现(y_start 取下界、y_end 取 seqlen_q),保证不漏;(b) 写离线校验程序:对随机 (seqlen,5 因子) 枚举每个 KV tile,断言 `[y_start,y_end)` ⊇ `{sq : ∃sk∈tile, IsTokenPairInsideMask(sq,sk)}`;(c) 通过后再逐分支收紧。**fallback**:方案 B(Q 全扫 + IsFullTileInsideMask),先正确后优化。
2. **group per-segment 超参在 GPU 上怎么取。** fwd 已用 device 指针 `group_*_ptr`(L125-129),bwd 沿用:kernel 内 `i_group=i_batch/num_batch_per_group` 后读 5 个标量构造 mask。开销:每 block 5 次 global scalar load(可 `s_load`/readfirstlane 广播)。**未决**:是否值得做 constant buffer / __grid_constant__;首版直接 global load,profiling 再说。
3. **`max_attn_len` = `window_size` 还是别名?** mask 构造把 `max_attn_len_` 收缩为 `min(max_k_uih_len,min(max_q_uih_len,window_size))`(L53)。host 传的是原始 `window_size`,收缩在 mask 构造内完成 —— bwd dispatch 必须传**原始 window_size**给 mask 构造,不能传收缩后的值(否则二次收缩)。已在 §1.0 标注。
4. **target / contextual 边界 case**:`num_target=0`(无 target)、`contextual_seqlen=0`、`min_full_attn_seqlen >= max_q_uih_len`(整段都"first split"或都不是)。reference L177-194 的 `is_tile_in_first_split` 三元判定必须在 GPU dispatch 逐字复刻;`window_size==0`(`kHasLocal=false`)走 NoLocal mask 类(`BOOL_SWITCH_2(window_size>0,kHasLocal,…)`,L167)。
5. **`is_tile_in_first_split` 在 bwd 是否仍是构造期常量?** reference 对整个 (batch,head) 用单一 `is_tile_in_first_split=true` 构造(L179),真正的 per-tile split 判定下沉到 `GetTileRangeAlongX/IsFullTileInsideMask` 内的 `!is_tile_in_first_split` 分支——但构造时恒传 `true`。**疑点**:既然恒传 true,`!is_tile_in_first_split` 分支(L82-103/246)在 reference 路径下永不触发?需 pane-1 与我确认 fwd GPU 是否在别处按 tile 重建 mask 切换该 flag。**保守做法**:bwd 复刻 fwd 的构造方式(恒 true),min_full 区的行覆盖靠 `GetTileRangeAlongY` 显式放到 seqlen_q。
6. **`hdim_qk≠hdim_v` 的 dq_acc 宽度**:`dq_acc` 是 `hdim_qk` 宽(dQ 维度),`d_ptr`(D)是每 sq 一个标量(`hdim` 无关)。pane-3 分配 dq_acc 时用 `hdim_qk`,deterministic 时再 ×`num_splits`(=`ceil(seqlen_k/kN0)`,FMHA L2078/2141)。

---

## 附:对 pane-1 / pane-3 的接口假设

**给 pane-1(算法/pipeline)**
- mask 谓词唯一真值源:`mask.IsTokenPairInsideMask(sq, sk)`;STAGE1(P 置 0 / -inf)与 STAGE5(dS 置 0)**共用同一布尔 tile**。
- bwd 用 KV-外 / Q-内,内层 Q 范围由 `mask.GetTileRangeAlongY(i_n0, kM0, kN0) → [seqlen_q_start, seqlen_q_end)`(我将新增此成员);`num_total_loop<=0` 整块 early-exit;tile 内 `mask.IsEdgeTile(sq_tile, sk_tile, kM0, kN0)` 决定逐像素。
- scale:STAGE1 乘 `alpha`;SiLU 路径 STAGE2 P 与 STAGE4 dS 乘 `scale_p`,dsilu 用**重算的 S**;STAGE5 dQ、STAGE6 dK 末乘 `alpha`。softmax 路径用 LSE+D,不用 scale_p。
- 收缩维:GEMM(QK^T、dS@K、dS^T@Q)用 `hdim_qk`;GEMM(dO@V^T、P^T@dO、dO·O)用 `hdim_v`。

**给 pane-3(文件/codegen/测试)**
- 新增 mask 成员函数:`GetTileRangeAlongY`、`IsEdgeTile`(加到 `hstu_block_masking.hpp` 四个 struct;纯新增,不动 fwd)。需配套离线校验(§5.1)。
- kargs 字段名以 §3 为准:`do_ptr, dq_ptr, dk_ptr, dv_ptr, lse_ptr, d_ptr, dq_acc_ptr` + 对应 `seq_/nhead_/batch_stride_*` + `alpha, scale_p(或 attn_scale+max_seqlen_q), num_splits, kIsDeterministic`;group 另加 `group_attn_scale_ptr / group_window_size_ptr / group_contextual_seqlen_ptr / group_min_full_attn_seqlen_ptr / group_max_seqlen_q_ptr`。
- 两个新 params struct:`HstuAttentionNoGroupBwdParams`(batch+jagged,`is_jagged` 区分)、`HstuAttentionGroupBwdParams`(加进 `hstu_attention_params.hpp`)。
- instances 维度:`kIsJagged × kUseSoftmax × kUseCausal × kHasLocal(window>0) × kIsGroupMode × kIsDeterministic × (hdim_qk,hdim_v) 档`;mask 类经 `HstuBlockMasking<kIsCrossAttention,kUseCausal,kUseLocal>::Type` 选取(L787-798),`BOOL_SWITCH_2(window_size>0, kHasLocal, is_cross_attention, kIsCrossAttention,…)` 复刻 reference L167。
- 三 kernel grid:PRE/POST `dim3(ceil(seqlen_q/kM0), nhead, num_batch)`;MAIN `dim3(ceil(seqlen_kv/kN0), nhead, num_batch)`。group/jagged 的 batch_offset 走 `seq_*_offsets_ptr[i_batch]`,per-block `seqlen<=i_*0` early-return。
