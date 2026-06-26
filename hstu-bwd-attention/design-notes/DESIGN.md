# HSTU Attention Backward — GPU 实现方案 (DESIGN)

> 整合自三份设计:`part-algo.md`(算法&pipeline)、`part-mask-modes.md`(mask/模式/参数)、`part-engineering.md`(工程&验证)。
> 本文是**可交付用户最终 review 的单一方案**,已逐条裁决跨组接口、消解冲突(见 §8)。
> **已过双 review(正确性 pane-2 + 工程可行性 pane-3):P0 = 0;P1 全部落实(见 §8.5 review 落实记录);关键风险闸门 = M1。**
> 事实/行号以源码为准:`reference_hstu_attention_bwd.hpp`(oracle, 855 行)、`/root/ck/include/ck_tile/ops/fmha/`(FMHA bwd 基建)、HSTU fwd 同目录文件。日期 2026-06-04。

---

## §0 摘要

**目标**:为 HSTU attention 反向实现一套 GPU kernel(现状仅有 855 行 CPU reference,无 GPU bwd)。
**目标芯片**:**主目标 gfx950(MI350 / CDNA4),gfx942(CDNA3)次之**(沿用 fwd 的 `BUILD_HSTU_FOR_GFX95_ONLY` 宏分叉;CDNA4 专项见 §4.8)。
**总策略一句话**:**复用 FMHA bwd 的 3-kernel(PRE/MAIN/POST)+ `kr_ktr_vr` 7-stage / 5-GEMM 体系与 default policy;PRE/POST/policy/shape/enum 近零成本复用,差异全部收敛到 MAIN pipeline 的两个 elementwise 段(激活双路 SiLU/softmax + HSTU 5 因子 mask)与两个 scale(alpha / scale_p),外加 HSTU 三模式(batch/jagged/group)索引与新增的反方向 mask 几何成员。**

**与 FMHA bwd 的 6 个本质差异**(决定改动面):
1. 激活默认 **SiLU**(非 softmax):STAGE2 重算 S 取 dsilu、不读 LSE;masked-out 必须**显式置 0**(`dsilu(0)=0.5≠0`,禁用 -inf)。
2. 两套 scale:`alpha`(QK)+ `scale_p`(SiLU 输出);FMHA 只有 `1/√d`。
3. HSTU 5 因子 mask(causal/window/contextual/min_full/num_target);bwd 走 KV 外循环需**反方向** tile 范围 → 新增 mask 成员。
4. jagged/group/batch 三模式 + bshd + cu_seqlens;group 每段独立超参。
5. `hdim_qk ≠ hdim_v`。
6. softmax 路才需 LSE(fwd `kStoreLSE` 接线柱已就绪)+ O(PRE 算 D);SiLU 路都不需要。

---

## §1 总体架构

### 1.1 3-kernel 结构(沿用 FMHA,条件发射)
```
[PRE]  HstuAttentionBwdOGradDotO   —— D[sq]=rowsum(O⊙dO)        仅 kUseSoftmax 发射
[MAIN] HstuAttentionBwdDQDKDV      —— 5 GEMM / 7 stage,产 dV/dK + 累加 float dQ_acc   恒发射
[POST] HstuAttentionBwdConvertQGrad—— dq_acc(float)→dQ(bf16/fp16)   恒发射
```
- **PRE**:`block_fmha_bwd_dot_do_o.hpp` 零改造复用(`p_undrop=1.0`)。SiLU 路编译期 `if constexpr(kUseSoftmax)` 整体跳过,连 D buffer 都不分配。
- **MAIN**:`block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp` 作**结构蓝本**改写(不直接 include),见 §2。
- **POST**:`block_fmha_bwd_convert_dq.hpp` 复用,**两路恒发射**(P1-E 修正):dQ 最终是 bf16/fp16,而 MAIN 内只能向 **float `dq_acc`** atomic 累加 → 必须 POST 做 cast→`dq_ptr`。**atomic 路 POST = convert-only**(单份 cast)、**deterministic 路 POST = reduce+convert**(沿 split 归约后 cast)。故每次 bwd kernel 计数恒为 **2(SiLU)/ 3(softmax)**,atomic 路**也需分配 float `dq_acc`**(nsplits=1),并非"零额外 workspace"。

### 1.2 tile 布局:KV 外 / Q 内(与 CPU reference 同构)
- MAIN grid 沿 **seqlen_k** 分块:每个 block 固定一个 KV 块(`i_n0 = blockIdx.x*kN0`),把 `dk_acc/dv_acc` 留寄存器、沿 Q 累加,末尾 epilogue 一次写回。实证 FMHA `fmha_bwd_kernel.hpp:1063-1068, 1523-1581`;reference 同构(固定 `sk` 累加器、外 `for sq`、`reference_...bwd.hpp:227-465`)。
- 内层 Q tile 范围由 `mask.GetTileRangeAlongY(i_n0, kM0, kN0)→[q_start,q_end)` 求得;`num_total_loop≤0` 整块 early-exit;tile 内 `mask.IsEdgeTile(...)` 决定是否逐像素。

### 1.3 GridSize / launch 顺序
| kernel | GridSize | 备注 |
|---|---|---|
| PRE | `(ceil(seqlen_q,kM0), nhead, num_batch)` | FMHA :1733;dot_do_o 块长断言 `kM0==kBlockSize`(`block_fmha_bwd_dot_do_o.hpp:48`)|
| MAIN | `(ceil(seqlen_kv,kN0), nhead, num_batch)` | FMHA :1064 |
| POST | `(ceil(seqlen_q,kM0), nhead, num_batch)` | FMHA :2011 |

launch:`[PRE if softmax] → MAIN → POST`(**POST 两路恒发射**,P1-E)。group/jagged:grid.z=num_batch,kernel 内 `seq_*_offsets_ptr[i_batch]` 求 offset、`seqlen≤i_*0` per-block early-return。

### 1.4 数据流
```
SiLU :   Q,K,V,dO ─► MAIN ─► dV,dK, float dq_acc ─► POST ─► dQ
Softmax: O,dO ─► PRE ─► D ┐
         Q,K,V,dO,LSE,D ──┴► MAIN ─► dV,dK, float dq_acc ─► POST ─► dQ
```
(POST 两路恒发射:atomic→convert-only / deterministic→reduce+convert,P1-E)

---

## §2 算法与七阶段(SiLU / softmax 双路)

记号:`s_acc`=gemm_0 的 C tile(`SPBlockTileType`,FMHA:477);`p`/`g`=同分布派生 tile;`dp_acc`=gemm_2 的 C tile。FMHA 中 `p` 与 `dp_acc` 同 `i_j_idx` 索引(:665)→ `s_acc/p/g` 与 `dp_acc` 可在 STAGE5 逐元素混算,这是 SiLU 留 S/g **无需额外 LDS** 的关键。

### 2.1 reference 6 步 ↔ FMHA 7 stage ↔ 5 GEMM
| ref step | 数学 | FMHA stage / GEMM | HSTU 改造 |
|---|---|---|---|
| 1 重算 S,P | `S=alpha·Q·Kᵀ`;`P=silu(S)·scale_p` 或 `exp(S−LSE)` | STAGE1 `gemm_0` + STAGE2 | 双路;alpha 在 STAGE2 头乘入;SiLU 留 dsilu 因子 `g` |
| 2 dV | `dV += Pᵀ@dO` | STAGE3 `gemm_1` | 同 FMHA(P 已含 scale_p) |
| 3 dP | `dP = dO@Vᵀ` | STAGE4 `gemm_2` | 同 FMHA(沿 hdim_v) |
| 4 dS | SiLU `dP·scale_p·dsilu(S)` / softmax `P·(dP−D)` | STAGE5 | 双路;SiLU masked-out 显式 0 |
| 5 dQ | `alpha·dS@K` | STAGE7 `gemm_4` + 收尾 scale | alpha 用 FMHA `raw_scale` 槽 |
| 6 dK | `alpha·dSᵀ@Q` | STAGE6 `gemm_3` + 末尾 scale | alpha 用 `raw_scale` 槽 |

**5 个 GEMM 形状全部不变**(FMHA:141-145),差异只在 STAGE2/STAGE5 的 elementwise 段 + 两个 scale + mask。

### 2.2 逐阶段改造
- **STAGE1**(:518)`s_acc = gemm_0(q,k)`,产**未缩放** Q·Kᵀ,原样复用。
- **STAGE2**(替换 :520-622):
  - 公共:`s_acc *= alpha`(物化真正的 S;两路统一)。**SiLU 必须拿到已缩放 S 喂 dsilu,无法折进 exp**。
  - SiLU:`p = silu(s_acc)*scale_p`(→dV);`g = scale_p*dsilu(s_acc)`(→STAGE5);**edge tile 用 `set_tile_if` 把 `p`、`g` 的 masked-out 元素清 0**。LSE 不读。
  - Softmax:`p = exp2(log2e*s_acc − log2e*get_validated_lse(lse))`(= `exp(S−LSE)`;lse 自然对数域)。**P1-1 NaN 守卫**:全 masked 行 fwd 存 `LSE=−inf`,直接 `(−inf)−(−inf)=NaN`;复用 FMHA `get_validated_lse`(`raw_lse==−inf ? 0 : raw_lse`,:571-583)→ masked 位 `s=−inf` 经 `exp2(−inf)=0`,整行不产 NaN。**仅 softmax 路需要**(SiLU 路 dsilu 已对极值饱和为 0,无 NaN)。masked-out 沿用 FMHA「set s=−inf → p=0」(:557-568),无需显式补零。
- **STAGE3**(:624-646)`gemm_1(dv_acc, pt(p), dot)`,原样;dV 不乘任何 scale。
- **STAGE4**(:648-655)`dp_acc = gemm_2(do, v)`,原样,沿 hdim_v。
- **STAGE5**(替换 :657-669):
  - SiLU:`ds = dp_acc * g`(g 已含 scale_p;masked-out 因 g=0 自动为 0,无需再判 mask)。
  - Softmax:`ds = p * (dp_acc − d)`(去掉 FMHA dropout 三元;d 来自 PRE)。
- **STAGE6**(:698-708)`gemm_3(dk_acc, dst(ds), qt)`,原样;alpha 留收尾。
- **STAGE7**(:718-757)`gemm_4(dq_acc, ds, kt_slice)`;**收尾 `dq_acc *= alpha`**(:747 raw_scale→alpha);`update_tile`(atomic)/ deterministic 时 `store_tile` 到 split。
- **收尾**(:763-773)`dk_acc *= alpha`;`dv_acc` 不乘。返回 `(dk_acc, dv_acc)`。

> alpha 出现两处(STAGE2 头 + dQ/dK 收尾,**dV 不吃**);scale_p 折进 `p` 与 `g`(softmax 路不用 scale_p)。结构与 FMHA `raw_scale` 同构 → 「alpha == FMHA raw_scale 槽」。

### 2.3 MAIN `operator()` 双路骨架(只标改造点)
```cpp
// [同 FMHA] KV 预载寄存器、LDS 编排、early-exit (:151-491)
//   ★ GetTileRangeAlongY 在 :160-161 被【无条件】调用(IsMasking 守卫之外)→ M1 no-mask 也需提供平凡版返回 (0,seqlen_q)
//   ★ SiLU 路 lse/d 的 load 必须 if constexpr(kUseSoftmax) 跳过(FMHA 是无条件 load_tile,会读空指针;传 null window 不够)
while (i_total_loops < num_total_loop) {
  auto q_reg = load_tile(q_lds_read_window);
  decltype(...) lse;
  if constexpr(kUseSoftmax) lse = load_tile(lse_lds_read_window);  // SiLU 路不 load

  auto s_acc = gemm_0(q_reg, k_reg_tensor);                       // STAGE1
  tile_elementwise_inout([&](auto& x){ x = alpha * x; }, s_acc);  // ★ alpha 入 S
  SPBlockTileType p, g;
  if constexpr(!kUseSoftmax) {                 // SiLU
    p = silu(s_acc) * scale_p;                 // → dV
    g = scale_p * dsilu(s_acc);                // → dS(留到 STAGE5)
    if (mask.IsEdgeTile(seqlen_q_step, k_origin_x, kM0, kN0))     // ★ 显式置 0
      set_tile_if(p & g <- 0, [&](idx){ return !mask.IsTokenPairInsideMask(row,col); });
  } else {                                     // Softmax
    if (mask.IsEdgeTile(...)) set_tile_if(s_acc, -inf, !IsTokenPairInsideMask);   // [同 FMHA]
    // ★ P1-1 NaN 守卫:全 masked 行 LSE=-inf,(-inf)-(-inf)=NaN → 用 FMHA get_validated_lse(raw==-inf?0:raw)
    p = exp2(log2e*s_acc - log2e*get_validated_lse(lse));   // = exp(S-LSE);仅 softmax 路需此守卫
  }
  auto p_gemm = cast_tile<GemmDataType>(p);
  gemm_1(dv_acc, pt(p_gemm), dot_reg);                            // STAGE3 dV
  // ★ SiLU 路 d 的 load 同样 if constexpr(kUseSoftmax) 跳过(FMHA :628/:650 无条件 load_tile(d))
  auto dp_acc = gemm_2(do_reg, v_reg_tensor);                     // STAGE4 dP
  SPGradBlockTileType ds;                                         // STAGE5
  if constexpr(!kUseSoftmax) sweep: ds[i_j] = dp_acc[i_j] * g[i_j];
  else                       sweep: ds[i_j] = p[i_j] * (dp_acc[i_j] - d[i]);  // d=load_tile(d_lds) if softmax
  auto ds_gemm = cast_tile<GemmDataType>(ds);
  gemm_3(dk_acc, dst(ds_gemm), qt_reg);                           // STAGE6 dK
  gemm_4(dq_acc, ds(ds_gemm), kt_slice);                          // STAGE7 dQ
  tile_elementwise_inout([&](auto& x){ x = x * alpha; }, dq_acc); // ★ raw_scale→alpha
  kIsDeterministic ? store_tile(...) : update_tile(...);          // atomic / split
}
tile_elementwise_inout([&](auto& x){ x = x * alpha; }, dk_acc);   // ★ dK ×alpha;dV 不乘
return make_tuple(dk_acc, dv_acc);
```

### 2.4 SiLU 留 `g` vs 留整张 `S`(裁决见 §8-R2)
设计选**留 `g`(=scale_p·dsilu(S))**:与 FMHA 的 `p`(活到 STAGE5)生命周期同形,distribution 兼容性已被 FMHA STAGE5 跨 tile-type 索引(`p`[gemm_0 C] over `ds_spans`[gemm_2 C],:660-668)证明可行。
**资源代价(P1-B)**:SiLU 路 STAGE2-3 期间需**同时**留 `p`(→dV)与 `g`(→dS),峰值比 FMHA(只留 `p`)**+1 个 SPBlockTileType**;以 kM0×kN0=64×128、256 线程估 ≈ **+32 VGPR/线程**。**占用率模型按芯片分叉(关键,§4.8)**:gfx942(CDNA3)= ArchVGPR 与 AGPR 两个独立 512 文件,`occupancy=max(ArchVGPR,AGPR)`,+32 ArchVGPR 多半被 AGPR 侧吸收;**gfx950(CDNA4)= 统一 512 寄存器池,`occupancy=ArchVGPR+AGPR`(加法)**,留 g 的 +32 ArchVGPR 直接**叠加**到 MFMA 累加器(dk_acc/dv_acc/dq_acc 占 AGPR)之上 → **主目标 gfx950 上掉 wave 风险高于 gfx942**。**定性:低实现风险 / 中资源风险(gfx950 偏中高)。M1 必须按 CDNA4 加法模型验 `ScratchSize=0` 且 VGPR 不致掉 wave**;溢出则按 §8-R2 把 `g` 暂存 LDS(复用删掉的 bias LDS 区段;CDNA4 LDS 更大 + 64 banks + `ds_read_tr`,fallback 代价更低)。备选「留 S、STAGE5 再算 dsilu」等价但更费。

### 2.5 fwd 副产物契约
| 路径 | bwd 输入集合 | fwd 需存 |
|---|---|---|
| **SiLU**(默认) | `{Q,K,V,dO}` | **无**(S 重算;无 LSE/O/D/PRE) |
| **Softmax** | `{Q,K,V,dO,O,LSE}` | **LSE**(自然对数域,fwd `with_softmax` pipeline `:604-613` 已存 `m+log(l)`)+ **O**(PRE 算 D) |

D 由 PRE 现算(非 fwd 存)。bwd softmax 必须用 `exp(S−LSE)`,**alpha 已在 STAGE2 物化,切勿在 exp2 再乘 scale**(双 scale 高危 bug)。**LSE=−inf 守卫(P1-1)**:softmax 路必须对 LSE 套 `get_validated_lse`(−inf→0)避免全 masked 行产生 NaN;SiLU 路无此问题(无 LSE、dsilu 饱和)。

---

## §3 mask / 三模式 / 索引 / 参数

### 3.1 反方向 mask 几何缺口 → 方案 A(新增成员,纯加不改 fwd)
HSTU 四个 mask struct 只有 `GetTileRangeAlongX`(fwd 方向)、`IsFullTileInsideMask`、标量 `IsTokenPairInsideMask`;**缺 `GetTileRangeAlongY`(KV-block→Q tile 区间)与 `IsEdgeTile`**(bwd KV 外循环必需)。

**裁决:采纳方案 A** —— 实测净新增 **1 个非平凡成员(`GetTileRangeAlongY`)+ 1 个 wrapper(`IsEdgeTile`)**;越界谓词直接内联 `!IsTokenPairInsideMask`,**无需第三个成员**。否决方案 C(FMHA generic mask 表达不了 contextual/min_full/num_target);方案 B 全扫仅作 fallback。
- `GetTileRangeAlongY(i_x, YTile, XTile) → [y_start,y_end)`:`GetTileRangeAlongX` 的转置。**它在 MAIN 中被【无条件】调用(:160-161,在 `IsMasking` early-exit 守卫之外)**——故**连 M1 的 no-mask 路也必须提供**(此时平凡返回 `(0, seqlen_q)`),里程碑估时已纳入(§6 M1)。
  - **正确性铁律:返回区间必须是 `IsTokenPairInsideMask` 真值集在 Y 方向的超集**(宁放宽多算 tile,不可漏)。
  - **非连续陷阱(P1-C)**:contextual/min_full/num_target 叠加时,attend 某 KV-tile 的 Q 行集可能**非连续**(contextual 顶块 + 对角带 + target 尾块),而单段 `[y_start,y_end)` 无法表达 → **首版返回连续保守超集**(覆盖最小/最大行,中间空 tile 由逐像素清零兜底,perf 退化非正确性问题)。min_full 叠加下可能退化近全扫,profiling 后再分段优化。
  - 离线校验(§5.4)断言 `[y_start,y_end) ⊇ 真值集` 为 **M2 硬性前置**。骨架见 part-mask-modes §1.1。
- `IsEdgeTile(i_y,i_x,TH,TW) := !IsFullTileInsideMask(i_tile_top,i_tile_left,number<TileWidth>,number<TileHeight>)`(注意 4 参顺序,mask:237;封装统一成 FMHA 习惯)。首版 `IsFullTileInsideMask` 仅在 `!is_tile_in_first_split` 给 true → 偏保守(多逐像素),正确优先,后续 profiling 再收紧。

### 3.2 masked-out 显式置 0(SiLU 最易错点)—— 统一口径
- 真值唯一源:`mask.IsTokenPairInsideMask(sq,sk)`(已编码 5 因子;reference bwd STAGE1/4 直接用,:246/380)。
- **GPU 落地(统一):STAGE2 在 edge tile 上对 `p` 与 `g` 用 `set_tile_if` 清 0**(谓词 `IsOutOfBound`/`!IsTokenPairInsideMask`)。这等价 reference「STAGE5 dS else 0」(:380-385),前移到 STAGE2 只做一次更省;STAGE5 因 g=0 自动得 ds=0。整块 in-mask(`IsEdgeTile==false`)跳过。
- 注意:GPU 上 gemm 会算出 masked-out 位的真实 Q·K(非 0)→ 必须**清输出 `p`、`g`**(而非依赖 reference 的「S 置 0」CPU 写法);两者结果一致。
- softmax 路:masked-out 走 `s=-inf → p=0 → ds=0` 自然零,无需显式清。**禁止** SiLU 走 -inf(`dsilu(-inf)`→NaN)。

### 3.3 5 因子语义 bwd↔fwd 完全对称
同一 mask 对象、同一构造参数(bwd dispatch 复刻 fwd 的 `make_hstu_*_block_mask_*` 构造,含 `is_tile_in_first_split` 与最后一参的三元判定)。contextual(首行/列全可见)、min_full、num_target(`max_q_uih_len=seqlen_q−num_target`,target 行完整可见)在 bwd 经同一 `IsTokenPairInsideMask` 体现。唯一 bwd 特有动作是 tile 调度方向(Q 外→KV 外),**非 mask 语义**差异。
> **`is_tile_in_first_split` 已查清**(见 §8 已决):fwd **kernel** 按 **Q-tile** 重算该 flag(`hstu_attention_fwd_kernel.hpp:691-716`),非 (batch,head) 常量;而元素级 `IsTokenPairInsideMask` 自洽、不依赖该 flag。故 bwd 元素级置零不受影响;bwd 的 `GetTileRangeAlongY` 首版用保守超集(min_full 区把 y_end 放到 seqlen_q),first_split 的 tile 级优化作为后续 perf 项。

### 3.4 三模式 + 索引
| 模式 | seqlen_q/kv | base offset | num_target | mask 超参 |
|---|---|---|---|---|
| batch | params 标量 | `i_batch*batch_stride` | `num_targets_ptr[i_batch]` | 全局标量 window/contextual/min_full |
| jagged | `seqstart[b+1]−seqstart[b]` | `seqstart[b]*seq_stride` | `num_targets_ptr[i_batch]` | 全局标量 |
| group | `seqstart[b+1]−seqstart[b]` | `seqstart[b]*seq_stride` | `num_targets_ptr[i_batch]` | `group_*_ptr[i_group]`,`i_group=i_batch/num_batch_per_group` |

`GetTileIndex=(blockIdx.x,blockIdx.y,blockIdx.z)`(FMHA:1070);grid.z=num_batch、grid.y=nhead、grid.x=seq-tile。HSTU jagged/group 索引与 FMHA group 分支同构(:1110-1159),差别仅多 mask 5 因子标量。

### 3.5 hdim_qk ≠ hdim_v 接线
Q/K/dQ/dK 用 `hdim_qk`;V/O/dO/dV 用 `hdim_v`。收缩维:gemm_0(QKᵀ)/gemm_3(dSᵀ@Q)/gemm_4(dS@K) 沿 `hdim_qk`;gemm_1(Pᵀ@dO)/gemm_2(dO@Vᵀ)/PRE(dO·O) 沿 `hdim_v`。tile setting 的 `kQKHeaddim`/`kVHeaddim` 各开一档(FMHA `BlockFmhaShape` 已分离)。

---

## §4 工程落地

### 4.1 文件结构(新建 vs 复用 FMHA)
路线:**HSTU 自有风格(自写 dispatch + `generate_instances.py` 字符串模板),不搬 FMHA `codegen/ops/fmha_bwd.py`;只复用 FMHA device 端 pipeline/kernel 模板。**

新建(全落 `example/ck_tile/18_hstu_attention/`,镜像 fwd 命名):
`hstu_attention_bwd_{type_config,tile_setting_define,setting,pipeline_problem,traits,kernel}.hpp`、`hstu_attention_{no,with}_softmax_bwd_pipeline.hpp`(MAIN 双路)、`hstu_attention_bwd_{dot_do_o,convert_dq}_pipeline.hpp`(PRE/POST thin wrapper)、`hstu_attention_{batched,jagged,group}_backward_dispatch.hpp`、`hstu_attention_{no_group,group}_backward_{bf16,fp16}.cpp`、`instances/...`(生成)、`example_hstu_attention_bwd.cpp`。
> `hstu_attention_bwd_setting.hpp` 必须含 **gfx95 专用 tile setting 分支**(对齐 `hstu_attention_fwd_setting.hpp` 的 gfx95 路径);dispatch 文件镜像 fwd 的 `BUILD_HSTU_FOR_GFX95_ONLY` 宏分叉。详见 §4.8。

直接 `#include` 复用 FMHA:`block_fmha_bwd_pipeline_default_policy.hpp`(整体继承)、`..._enum.hpp`、`..._dot_do_o.hpp`、`..._convert_dq.hpp`、`tile_fmha_shape.hpp::TileFmhaBwdShape`。

### 4.2 problem / traits / tile / policy
- `HstuAttentionBwdPipelineProblem`:抄 FMHA `BlockFmhaBwdPipelineProblem` 骨架,**砍** `RandValOutputDataType/FmhaDropout/BiasGradDataType/kHasBiasGrad`(已核 default policy **不引用**这四个,grep 0 命中,砍掉安全);**加** `kUseSoftmax`(新轴)、`CompDataType`(dsilu)、`kUseGroup/kIsJagged/kIsCrossAttention`。`static_assert(!kUseGroup||kIsJagged)`。mask 不进模板(dispatch 阶段由 `HstuBlockMasking<...>` 选型后传入,同 fwd)。PRE/POST sub-problem 复用 FMHA。
  - **★ 复用 default policy 的硬前提(P1-A):Problem 必须保留** `static constexpr auto BiasEnum = BlockAttentionBiasEnum::NO_BIAS;` + `using BiasDataType = InOutDataType;`(dummy)。因 `BlockFmhaBwdPipelineDefaultPolicy::GetSmemSizeBias()`(:1627-1628)内 `if constexpr(Problem::BiasEnum==ELEMENTWISE_BIAS) sizeof(Problem::BiasDataType)*...`,被 `GetSmemSize<Problem>()`(:1641-1647)无条件聚合、再被 pipeline `GetSmemSize()`(:80-83)必然实例化 → 删掉这 2 个成员 default policy 直接编译失败、policy 复用破产。**kargs 不传 bias**,只保留这 2 个 dead typedef(免费)。这把 R1 从"政策不确定"收敛为"保留 2 个占位 typedef",并顺带回答 U1。
- `HstuAttentionBwdTraits`:`kPadSeqLenQ/K`、`kPadHeadDimQK/V`、`kBlockPerCu`。
- tile:9 元 `BlockTile = sequence<kM0,kN0,kK0,kK1,kK2,kK3,kK4,kQKHeaddim,kVHeaddim>`(FMHA `TileFmhaBwdShape` 语义)。**按芯片分叉**:gfx942(CDNA3)首版照搬 FMHA bwd per-hdim 预设(64/96/128/256);**gfx950(CDNA4,主目标)用 CDNA4 K-doubled MFMA tile(`32×32×16` / `16×16×32`,2× 吞吐),镜像 `hstu_attention_fwd_setting.hpp` 的 gfx95 分支**,勿照搬 gfx942 的 `32×32×8`/`16×16×16`。按 gfx95 宏分叉(§4.8)。
- policy:`using Policy = BlockFmhaBwdPipelineDefaultPolicy;`,**预期几乎无需覆写**(HSTU 与 FMHA 5 GEMM 形状一致,差异在 elementwise),**前提是保留上述 BiasEnum/BiasDataType**(P1-A)。可行性 M1 验证(§8-R1)。

### 4.3 dispatch
入口(去 dropout 轴):`run_<mode>_backward_dispatch<InOutDataType,kUseCausal,kUseSoftmax,kHasBias,kIsDeterministic,MaxK>(param, stream)`。`Run` 流程:选 tile setting → 算 pad → `BOOL_SWITCH` traits → `HstuBlockMasking` 选 mask → 组 problem/选 pipeline(`kUseSoftmax?WithSoftmax:NoSoftmax`)→ 组 3 kernel → 顺序 launch `[PRE?]→MAIN→[POST?]`。

### 4.4 dQ 写回两路(**两路都过 POST**,P1-E)
- **路 A(默认,`kIsDeterministic=false`)**:float `dq_acc`(nsplits=1)+ MAIN `atomic_add` + **POST convert-only**(cast→`dq_ptr`)。bf16/fp16 不能直接 atomic → 用 float dq_acc 中转;**故 atomic 路也必须分配 float dq_acc(nsplits=1)并发 POST,非"零额外 workspace"**。
- **路 B(`kIsDeterministic=true`)**:每 K-tile 写独立 dq_acc slice(无 atomic)+ POST `BlockFmhaBwdConvertQGrad` 的 **Reduce+Convert**(`:64-138`)沿 split 归约 → 逐位可复现。
- `dq_acc` 形状:deterministic `[nsplits, Σseqlen_q, nhead, hdim_qk]` float,`nsplits=ceil(max_seqlen_k/kN0)`(=MAIN grid.x);atomic 退化 nsplits=1。host 端 `hipMalloc`(对齐 fwd `o_acc_ptr/lse_acc_ptr` 临时 buffer 模式),填 `param.num_splits`。

### 4.5 instances / CMake
- 轴:`mode(3)×dtype(2)×causal(2)×softmax(2)×bias(2)×deterministic(2)×maxk(4)=384`。**MVP 收敛**:bias 恒 false、deterministic 恒 false、maxk 先 {64,128} → **≈48 instance**(与 fwd 同量级)。
- `generate_instances.py` 加 `create_backward_instances`(字符串模板),每次清空 `instances/`;CMake `file(GLOB instances/*.cpp)` 自动纳入。新增独立 bwd target `tile_example_hstu_attention_bwd`(EXCLUDE_FROM_ALL)。gfx95/94 分叉、`-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3` 沿用 fwd;**gfx95 target 追加 `-fno-slp-vectorize`(改善 gfx950 pipelining,镜像 fwd CMake)+ `BUILD_HSTU_FOR_GFX95_ONLY` 宏**(§4.8)。
- CLI 增量:`--deterministic`、`--bwd_v`(对拍)、`--dump_grad`。

### 4.6 canonical 字段表(params / kargs — 三组统一命名)
两个新 struct 进 `hstu_attention_params.hpp`(复用 fwd 全部输入字段):
- `HstuAttentionNoGroupBwdParams`(batch+jagged,`is_jagged` 区分)
- `HstuAttentionGroupBwdParams`(group)

| 类别 | 字段(canonical) | 说明 |
|---|---|---|
| 复用 fwd 输入 | `q_ptr,k_ptr,v_ptr,o_ptr`,seq/nhead/batch stride,`num_targets_ptr`,`seq_q/kv_offsets_ptr`,`max_seqlen_q`,`is_cross_attention,is_jagged,use_causal,use_softmax` | o_ptr softmax 路用 |
| mask 超参 | batch/jagged:`window_size,contextual_seqlen,min_full_attn_seqlen`(标量);group:`group_{window_size,contextual_seqlen,min_full_attn_seqlen,max_seqlen_q,attn_scale}_ptr` + **`num_group,num_batch_per_group`**(P2-3,reference:514;`i_group=i_batch/num_batch_per_group`)| group 按 i_group |
| bwd 输入 | `do_ptr`;`lse_ptr`(仅 softmax,SiLU=nullptr) | |
| bwd 输出 | `dq_ptr,dk_ptr,dv_ptr` + 各自 seq/nhead/batch stride | dq/dk 宽 hdim_qk,dv 宽 hdim_v |
| PRE 产物 | `d_ptr` + `nhead_stride_lsed` + **`batch_stride_lsed`**(P2-2,batched softmax 需要;LSE 与 D 同布局:每 (b,h,sq) 一标量,共用 stride 命名)| 仅 softmax |
| dQ workspace | `dq_acc_ptr,stride_dq_acc,nhead_stride_dq_acc,batch_stride_dq_acc,split_stride_dq_acc,num_splits` | float |
| scale | `alpha`(=fwd `scale_s`),`attn_scale`(scale_p 源) | 见 §4.7 |
| 开关 | `kIsDeterministic` | 模板轴,struct 留 bool 便于 host 选 instance |

与 FMHA `fmha_bwd_args` 净差异:**kargs 删** dropout+bias+dbias 三组(**注:Problem 仍保留 `BiasEnum=NO_BIAS`+`BiasDataType` dummy typedef,P1-A,仅 kargs 不传 bias**);mask kargs **换** HSTU 5 因子(batch 标量 / group 指针);**加** scale_p 通路;`seqlen_k→seqlen_kv`、`hdim_q→hdim_qk` 改名;`nhead_ratio_qk=1`(HSTU 暂无 GQA,留常量占位,U4)。

### 4.7 scale 语义与接线表(固化,防双 scale bug)
| 名称 | 含义 | 来源/默认 | bwd 用处 | 是否 per-group |
|---|---|---|---|---|
| `alpha` | QK 缩放(FMHA `1/√d` 槽) | = fwd `scale_s` | STAGE2 头 `s_acc*=alpha`;dQ(STAGE7)、dK(收尾)末乘 alpha;**dV 不乘** | **否**(group 也是单标量,reference:515) |
| `scale_p` | SiLU 输出缩放 | `attn_scale ? attn_scale : 1/max_seqlen_q` | 折进 `p`(→dV)与 `g`(→dS);**softmax 路不用** | **是**(group:`group_attn_scale[i_group]?...:1/group_max_seqlen_q[i_group]`) |

- `alpha`:host 存 kargs 标量(三模式均单值)。
- `scale_p`:batch/jagged 推荐 **host 预算单值**传入(免 kernel 分支);group **必须 kernel 内按 i_group 取**(device 指针)。
- softmax 路:仅用 alpha + LSE + D,`exp(S−LSE)`(自然对数域),**不在 exp2 再乘 scale**。

### 4.8 gfx950 / CDNA4 注意事项(主目标芯片)
当前机器为 **gfx950(MI350 / CDNA4)**,设为主目标;gfx942(CDNA3)次之。**算法/七阶段双路、scale 接线、复用边界、wave64、MFMA 适用性在两代一致**,差异集中在 **占用率模型 / tile-MFMA preset / build**。fwd 已有成套 gfx950 路径,bwd **必须镜像**。

**① 占用率模型变了(最关键,升级 R2/P1-B)**
- **CDNA3(gfx942)**:ArchVGPR(512)与 AGPR(512)是**两个独立寄存器文件**,`occupancy = max(ArchVGPR, AGPR)`。
- **CDNA4(gfx950)**:ArchVGPR + AGPR 共享**统一 512 寄存器池**,`occupancy = ArchVGPR + AGPR`(**加法**)。
- 影响:bwd 的 MFMA 累加器(`dk_acc/dv_acc/dq_acc`)占 AGPR,而留 `g` 的 **+~32 ArchVGPR** 在 CDNA4 上**直接叠加**其上 → **gfx950 掉 wave 风险高于 gfx942**。
- 处置:**M1 的 VGPR 验收必须按 CDNA4 加法模型算 occupancy**;若掉 wave,优先 §8-R2 的 `g` 暂存 LDS fallback —— CDNA4 LDS 更大 + 64 banks + `ds_read_tr` 转置读,fallback 代价比 CDNA3 更低。

**② tile / MFMA preset**
- gfx950 tile_setting 用 **CDNA4 K-doubled MFMA 形状 `32×32×16` / `16×16×32`(f16/bf16,2× 吞吐 vs CDNA3 的 `32×32×8` / `16×16×16`)**;勿照搬 gfx942 预设。走 gfx95 分支,镜像 `hstu_attention_fwd_setting.hpp` 的 gfx95 路径。

**③ build / 结构镜像 fwd**
- bwd dispatch/setting 加 `BUILD_HSTU_FOR_GFX95_ONLY` 分支;CMake 对 gfx95 target 追加 `-fno-slp-vectorize`(改善 gfx950 pipelining);policy/pipeline 需要处加 `#ifdef __gfx950__` 设备分支(对齐 fwd `hstu_attention_fwd_pipeline_policy.hpp` 的 3 处)。

**④ LDS 更宽裕(利好 fallback)**:CDNA4 LDS 比 CDNA3(64KB)更大、**64 banks**(CDNA3 32)+ 新 `ds_read_tr` 转置读变体(`B64_TR_B4/B8/B16`、`B96_TR_B6`)。(rocm-ref 对 CDNA4 LDS 具体容量记述不一致,此处只取定性"更大",不固化数值。)

**不变项**:wave64、MFMA 适用、算法/七阶段双路、两个 scale 接线、复用 vs 新写边界 —— gfx942/gfx950 完全一致。

---

## §5 正确性验证

### 5.1 对拍主流程(oracle = `reference_*_hstu_attention_bwd`)
```
随机 seed → Q/K/V/dO (+ jagged offsets/num_targets/mask 超参)
 ├ GPU:fwd 产 O(+LSE if softmax) → bwd(PRE→MAIN→POST)产 dQ/dK/dV
 └ CPU:reference fwd 产 O/LSE → reference bwd 产 dQ*/dK*/dV*
对比 (dQ,dK,dV) vs (dQ*,dK*,dV*),用 ck_tile::check_err 分张量报 max/mean-err
```
oracle 签名(已核实):`reference_no_group_hstu_attention_bwd<InOut,GemmAcc,Comp,kIsJagged,kUseSoftmax,kUseCausal>::Run(is_cross, q,k,v,lse,o,do, dq,dk,dv, num_batch, alpha, attn_scale, max_seqlen_q, max_seqlen_kv, seq_q_offsets, seq_kv_offsets, num_targets, contextual_seqlen, window_size, min_full_attn_seqlen)`;group 版多 `num_batch_per_group` + 5 个 `group_*` 数组。SiLU 路 lse 传空。

### 5.2 容差(初值,按实测收紧;分张量)
| dtype | rel-err | max abs-err(归一) |
|---|---|---|
| bf16 dQ/dK/dV | ≤ 2e-2 | ≤ 5e-2 |
| fp16 dQ/dK/dV | ≤ 5e-3 | ≤ 1e-2 |
dQ 误差通常最大(跨 block 累加),dV 最小。**专项**:SiLU masked-out 区校验 dS=0(不污染 dK/dV)。

### 5.3 测试矩阵(分层抽样)
激活{SiLU,softmax} × 模式{batched,jagged,group} × mask{no-mask,causal,+window,+contextual,+min_full,+num_target,组合} × dtype{bf16,fp16} × hdim{(64,64),(128,128),(128,64),(256,256)} × dQ 路{atomic,deterministic}。核心组合(SiLU×batched×causal×bf16×64)全跑,其余轴单轴变更 + 少量全组合 smoke。

### 5.4 deterministic 逐位 + 边界
- `--deterministic 1` 同输入两遍 → dQ bitwise 相等(memcmp);atomic 路仅数值容差。
- 边界:seqlen 非 tile 整除(pad)、cross(seqlen_q≠kv)、单 batch、空 target、window=0、contextual=0、hdim_qk≠hdim_v、jagged 各段长度差异大。
- **`GetTileRangeAlongY` 离线校验**(高优):随机 (seqlen,5 因子) 枚举每 KV tile,断言 `[y_start,y_end) ⊇ {sq:∃sk∈tile, IsTokenPairInsideMask(sq,sk)}`。

---

## §6 分阶段里程碑(M1 为风险闸门)
| 阶段 | 范围 | 验收 |
|---|---|---|
| M0 脚手架 | params bwd 字段、3 kernel 空壳、dispatch、CMake target、instances bwd 分支、CLI | 编译过;launch 不崩;全 0 输出 |
| **M1 端到端**(闸门) | **batched+SiLU+no-mask+bf16+atomic+hdim64**(主目标 **gfx950 / CDNA4**,gfx95 tile preset):MAIN 5 GEMM+dsilu+dQ(float dq_acc+POST convert-only);**含平凡 `GetTileRangeAlongY`→(0,seqlen_q)**(P1-D:该调用在 IsMasking 守卫外,no-mask 也需)+ 保留 BiasEnum/BiasDataType(P1-A)| 对拍过 bf16 阈值;**`ScratchSize=0` 且 VGPR 不掉 wave —— 按 CDNA4 加法模型 `occupancy=ArchVGPR+AGPR` 判据(P1-B / §4.8)**。**一次性验证:FMHA MAIN 能否被 HSTU 特化(SiLU 重算 S + masked-out 0 + scale_p)+ FMHA policy 直接复用 + 留 g 的 VGPR(CDNA4 加法模型下)** |
| M2 mask 因子 | 逐加 causal→window→contextual→min_full→num_target→组合(非平凡 `GetTileRangeAlongY`/`IsEdgeTile`)| 每因子对拍过;masked-out dS=0 校验过;**`GetTileRangeAlongY` 离线超集校验(§5.4)硬性前置、必须先过(P1-C)** |
| M3 jagged | 单组超参 + cu_seqlens | jagged 对拍过;非整除 pad 过 |
| M4 group | per-group 超参数组、grid.z=batch、early-exit | group 对拍过(`reference_group_*`)|
| M5 softmax 路 | PRE(D)+ LSE 读取 + `dS=P*(dP−D)` | softmax 对拍过;消费 fwd kStoreLSE 产物 |
| M6 deterministic | split-workspace + POST Reduce+Convert | 逐位可复现 |
| M7 多 dtype/maxk | fp16 + hdim{96,128,256}+qk≠v | 抽样矩阵过 |
| M8 性能/收尾 | tile 调优、occupancy、perf;(可选)bias/dbias | perf vs FMHA bwd 同 hdim 不显著落后 |

---

## §7 复用 vs 新写总清单
**零/极小改动复用 FMHA**:PRE(`BlockFmhaBwdOGradDotO`)、POST(`BlockFmhaBwdConvertQGrad`)、default policy、`TileFmhaBwdShape`、pipeline enum(KRKTRVR)、MAIN 全部 5 GEMM 与 KV-resident/LDS 编排/shuffle/early-exit/STAGE1,3,4,6,7 的 GEMM 调用与写回。
**HSTU 新写/特化**:① MAIN STAGE2(alpha 物化 + SiLU/softmax 双路 + SiLU 显式置 0 + **softmax 路 `get_validated_lse` 的 LSE=−inf NaN 守卫**,P1-1);② MAIN STAGE5(dS 双路);③ 收尾 scale raw_scale→alpha;④ 删 bias/dropout/position_encoding 死分支但**保留 `BiasEnum=NO_BIAS`+`BiasDataType` dummy**(P1-A,policy 复用硬前提)+ **SiLU 路 lse/d load 用 `if constexpr(kUseSoftmax)` 跳过**(P2-2);⑤ `silu`/`dsilu` device 函子;⑥ 新模板轴 `kUseSoftmax`;⑦ mask 新增 `GetTileRangeAlongY`(非平凡)+ `IsEdgeTile`(wrapper);**越界谓词直接内联 `!IsTokenPairInsideMask(row,col)` 做 `set_tile_if`,无需新增第三个 mask 成员**(P2-1);⑧ 两个 bwd params struct + kargs;⑨ 三套 dispatch + API 接缝 + instances + CMake target + example/对拍。

---

## §8 风险与未决问题

### 8.1 已决(跨组裁决,单一结论)
- **D1 反方向 mask 缺口 → 方案 A**:新增 `GetTileRangeAlongY` + `IsEdgeTile`(纯加不改 fwd)。与 STAGE2/5 置零统一为**一套谓词接口**:真值源 `IsTokenPairInsideMask`(标量)+ tile 版 `IsOutOfBound`/`IsEdgeTile`/`GetTileRangeAlongY`。STAGE2 与 STAGE5 共用同一布尔判定。
- **D2 `is_tile_in_first_split`**:**已查清**——fwd kernel 按 **Q-tile** 重算此 flag(`hstu_attention_fwd_kernel.hpp:691-716`),非 (batch,head) 常量;元素级 `IsTokenPairInsideMask` 自洽、不依赖该 flag。**结论**:bwd 元素级置零不受影响;`GetTileRangeAlongY` 首版用保守超集(min_full 区 y_end→seqlen_q),first_split 的 tile 级 Y 方向优化列为后续 perf 项(非正确性阻塞)。**pane-2 §5.5 未决 → 关闭。**
- **D3 留 g vs 留 S**:设计选**留 g**(与 FMHA 寄存器同形),M1 实测验证 VGPR(见 R1)。
- **D4 masked-out 置零**:SiLU 必清 `g`(及 `p`),`dsilu(0)=0.5`;**禁用 -inf**(NaN)。在 **STAGE2** 用 `set_tile_if` 对 edge tile 清零(等价 reference STAGE5 else-0,前移更省)。softmax 走 -inf 自然零。
- **D5 scale 接线**:见 §4.7 表。alpha 两处(STAGE2 头 + dQ/dK 收尾,dV 不吃,三模式单标量);scale_p 折进 p/g(softmax 不用,group per-group);softmax `exp(S−LSE)` 自然对数域,**不重复乘 scale**。
- **D6 group 超参取数**:`alpha` 全局单标量;`scale_p` 与 mask 超参(window/contextual/min_full/max_seqlen_q)per-group device 指针,kernel 内 `i_group=i_batch/num_batch_per_group` 索引;`num_target` per-batch(`num_targets_ptr[i_batch]`)。
- **D7 命名统一**:params/kargs 字段以 §4.6 canonical 表为准(消解 pane-2 `nhead_stride_lsed` 与 pane-3 `nhead_stride_d` 的出入:LSE/D 同布局,统一用 `*_lsed` 命名;`is_deterministic`→`kIsDeterministic`)。

### 8.2 建议(已定方向,需 M1/实测确认)
- **R1 FMHA default policy 直接复用可行性**(最高工程风险):policy 的 tile 分布是否兼容 HSTU 在 GEMM 间插入 dsilu/mask 的中间 reg 分布。M1 先验;不匹配则派生 `HstuBwdPolicy` 覆写少量 `Make*Distribution`,最坏复制 MAIN 全文再改。
- **R2 留 g 的 VGPR/occupancy**(P1-B,低实现 / 中资源风险,**gfx950 偏中高**):峰值 +1 SPBlockTileType ≈ +32 VGPR。**主目标 gfx950(CDNA4)为加法占用模型 `occupancy=ArchVGPR+AGPR`,+32 ArchVGPR 叠加在 AGPR 累加器上,风险高于 gfx942 的 `max()` 模型**(§4.8)。M1 在 hdim64 **按 CDNA4 加法模型**实测 `ScratchSize=0` 且不掉 wave;溢出则 g 暂存 LDS(复用删掉的 bias LDS 区段;CDNA4 LDS 更大 + 64 banks + `ds_read_tr`,fallback 代价更低)。
- **R3 `GetTileRangeAlongY` 精确边界**(P1-C):多分支转置易错,且 5 因子叠加下 attend 行集可能非连续。首版返回**连续保守超集**(不漏)→ 离线校验(§5.4,**M2 硬性前置**)→ 逐分支/分段收紧。fallback 方案 B(Q 全扫 + `IsFullTileInsideMask`)。注:该成员被 MAIN 无条件调用,M1 no-mask 需平凡版(P1-D)。
- **R4 deterministic dq_acc 显存**:长序列 `[nsplits,Σseqlen_q,nhead,hdim_qk]` float 可能爆;atomic 为默认(nsplits=1),deterministic 显式开,可加 nsplits 上限/分块。
- **R5 编译规模**:bwd instance ~3× fwd;MVP 48、按需开 bias/deterministic、ninja 并行、`extern template`。
- **R6 SiLU 路 LDS 浪费(perf,M1 后)**:若 SiLU MAIN 直接复用 `Policy::GetSmemSize`,会白算 `GetSmemSizeLSE/D`(:1570/1579)段。M1 通过后让 HSTU MAIN 自算 smem(SiLU 减去 LSE/D 段)省 LDS、提 occupancy。非阻塞。

### 8.3 建议默认值(✅ 用户已确认 2026-06-04:全用默认)
> **用户已于 2026-06-04 批准方案,U1–U4 全部采用以下默认值**,并指明当前机器为 gfx950(CDNA4)、需重点关注 CDNA4 差异(见 §4.8)。下列默认即最终决策,无待确认项。
- **U1 bias/dbias** → **✅ 已确认:不支持 dbias**。fwd `bias_ptr` 实际未用,下游大概率不需要 bias 梯度。Problem 仍**保留 `BiasEnum=NO_BIAS`+`BiasDataType` dummy**(P1-A 强制、免费);真 dbias 路(STAGE5 后存,FMHA :671-696 有蓝本)留 post-MVP。
- **U2 deterministic** → **✅ 已确认:默认 atomic**(省 split-workspace 与归约开销;**但两路都过 POST**,见 P1-E,atomic 不省 POST/不省 float dq_acc)。deterministic 作显式可选,需逐位可复现再开(M6,接受 R4 显存)。
- **U3 MVP 覆盖面** → **✅ 已确认**:SiLU 全覆盖优先(默认路 + M1 闸门),softmax 次之;bf16 优先(`CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3` 已设)、fp16 跟进;hdim 先 {64,128};三模式按 batched→jagged→group(M1→M3→M4)。
- **U4 GQA/MQA** → **✅ 已确认:`nhead_ratio_qk=1`(MHA)**,留 1 个常量字段占位。若后续模型共享 KV(KV head = q head/ratio),在三个 kernel 的 `i_nhead` 索引引入 ratio(FMHA 已支持),改动可控。

### 8.4 三份间冲突点(显式标注,均已消解)
- **C1**(措辞→已统一):置零位置 pane-1 选 STAGE2 清 `p,g` / pane-2 述 STAGE5 清 `dS`。**裁决**:STAGE2 清 `p,g`(D4),与 STAGE5-else-0 语义等价,前移更省;非事实冲突。
- **C2**(表述→已统一):pane-2「STAGE1 set masked-out S=0/-inf」vs GPU 实情(gemm 产全 tile,需清输出)。**裁决**:SiLU 清输出 `p,g`;softmax set `s=-inf`。结果一致。
- **C3**(命名→已统一):LSE/D stride 命名(`*_lsed` vs `*_d`)、`kIsDeterministic` vs `is_deterministic` → 以 §4.6 为准(D7)。
- **无事实级冲突**:三份对架构(KV 外/Q 内)、scale 接线、3-kernel 条件发射、复用边界一致。alpha 在 group 为单标量(pane-2 §4 与 pane-3 group params 一致),无分歧。

### 8.5 双 review 落实记录(P0=0)
正确性(pane-2)+ 工程可行性(pane-3)两份 review 均**无 P0**。逐条落实:

| 项 | 来源 | 落实位置 |
|---|---|---|
| **P1-1** softmax LSE=−inf NaN 守卫(`get_validated_lse`)| 正确性 | §2.2 / §2.3 骨架 / §2.5 / §7-① |
| **P1-A** 保留 `BiasEnum=NO_BIAS`+`BiasDataType`(policy 复用硬前提)| 可行性 | §4.2 / §4.6 / §7-④ / §8.3-U1 |
| **P1-E** atomic 路也恒发 POST(修 §1.1↔§4.4 矛盾)| 可行性 | §1.1 / §1.3 / §1.4 / §4.4 / §8.3-U2 |
| **P1-B** 留 g 的 VGPR ≈+32(中资源/低实现风险;**gfx950 CDNA4 加法占用模型下风险更高,M1 按加法模型验 ScratchSize=0**)| 可行性 | §2.4 / §4.8 / §6-M1 / §8.2-R2 |
| **P1-C** `GetTileRangeAlongY` 非连续→连续保守超集 + 离线校验 M2 前置 | 可行性 | §3.1 / §5.4 / §6-M2 / §8.2-R3 |
| **P1-D** `GetTileRangeAlongY` 无条件调用,M1 no-mask 需平凡版 | 可行性 | §3.1 / §6-M1 / §8.2-R3 |
| P2-1 越界谓词内联 `!IsTokenPairInsideMask`(无第三成员)| 正确性 | §3.1 / §7-⑦ |
| P2-2 SiLU 路 lse/d load `if constexpr` 跳过 | 正确性 | §2.3 骨架 / §7-④ |
| P2-3 group params 补 `num_group/num_batch_per_group` | 正确性 | §4.6 |
| P2(feas) PRE/POST GridSize 除数 `kBlockSize`→`kM0` | 可行性 | §1.3 |
| P2(feas) 补 `batch_stride_lsed` | 可行性 | §4.6 |
| P2(feas) SiLU 路自算 smem 省 LSE/D 段(perf)| 可行性 | §8.2-R6 |
| U1–U4 收为"建议默认 + 待用户确认" | 两份 | §8.3 |

**用户拍板状态**:✅ **U1–U4 已全部确认(2026-06-04,全用默认),无待确认项**;并指定主目标芯片 gfx950(CDNA4),CDNA4 专项见 §4.8。技术风险(R1 policy 复用、R2 VGPR——**按 CDNA4 加法占用模型**、R3 Y-range 边界)在 **M1/M2 实测闭环**,非设计期未决。
