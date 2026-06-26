# HSTU bwd · Part 1 —— 核心算法 & pipeline 设计 (pane-1 / architect)

> 范围:把 `reference_hstu_attention_bwd.hpp`(oracle)的反向数学映射到 GPU pipeline,
> 复用 FMHA bwd 的 3-kernel + `kr_ktr_vr` 7-stage 体系。只设计**算法与阶段**;mask/索引(pane-2)、
> 工程落地(pane-3)见文末接口假设。行号:FMHA MAIN =
> `block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp`;PRE = `block_fmha_bwd_dot_do_o.hpp`;
> oracle = `reference_hstu_attention_bwd.hpp`。

---

## 0. 速查:reference 6 步 ↔ FMHA 7 stage ↔ 5 GEMM

| reference step | 数学 | FMHA stage / GEMM | HSTU 改造 |
|---|---|---|---|
| Step1 重算 S,P | `S=alpha·Q·Kᵀ`;`P=silu(S)·scale_p` 或 `exp(S−LSE)` | STAGE1 `gemm_0`(S=Q@K)+ STAGE2(激活) | **双路**;alpha 在 STAGE2 头部乘进 s_acc;SiLU 需**留 S/留 dsilu 因子** |
| Step2 dV | `dV += Pᵀ@dO` | STAGE3 `gemm_1`(PT@dOT) | 同 FMHA(P 已含 scale_p) |
| Step3 dP | `dP = dO@Vᵀ` | STAGE4 `gemm_2`(dO@V) | 同 FMHA(hdim_v) |
| Step4 dS | SiLU `dS=dP·scale_p·dsilu(S)`;Softmax `dS=P·(dP−D)` | STAGE5(p·(dp−d)) | **双路**;SiLU masked-out **显式置 0** |
| Step5 dQ | `dQ = alpha·dS@K` | STAGE7 `gemm_4`(dS@KT)+ 收尾 scale | alpha 用 FMHA `raw_scale` 槽 |
| Step6 dK | `dK += alpha·dSᵀ@Q` | STAGE6 `gemm_3`(dST@QT)+ 末尾 scale | alpha 用 `raw_scale` 槽 |

5 GEMM 名称(MAIN 行 141-145)完全沿用:`gemm_0..gemm_4`。**没有任何 GEMM 需要换形状/换 policy**——HSTU 的差异全部落在 STAGE2 / STAGE5 的逐元素段(elementwise sweep)和两个 scale 上。这是能大量复用的根本原因。

---

## 1. 3-kernel 总体结构决策

沿用 FMHA 三段式:**PRE(dot_do_o)→ MAIN(dq_dk_dv kr_ktr_vr)→ POST(convert_dq)**。

### PRE(`BlockFmhaBwdOGradDotO`,计算 `D[sq]=rowsum(O⊙dO)`)
- **决策:做成编译期分支 `kUseSoftmax`,SiLU 路径完全不发射 PRE kernel。**
- 依据:oracle 中 D 只在 softmax 分支出现(line 391-411);SiLU 分支的 dS 公式(line 376-386)不含 D。SiLU 也没有 O 这个输入(SiLU fwd 不产出可复用的 D,且不需要 LSE)。
- 落地形态:host dispatch 里 `if constexpr(kUseSoftmax) { launch PRE; }`,SiLU 路径连 `D` 的 device buffer 都不分配。MAIN 的 `D` DRAM window 在 SiLU 下传 null tile window(见 §4 伪代码),STAGE5 走 SiLU 分支不读 D。
- PRE 本身**零改造**直接复用(它只读 O/dO 算点积行和,与激活无关)。注意 PRE 的 `p_undrop` 入参 HSTU 恒为 1.0(无 dropout,见 §5)。

### MAIN(`BlockFmhaBwdDQDKDVPipelineKRKTRVR`)
- 复用整套 7-stage 骨架、5 GEMM、KV-resident + Q/dO 流式的 LDS 编排、early-exit。
- HSTU 特化集中在三处:(a) STAGE2 激活双路 + alpha;(b) STAGE5 dS 双路 + SiLU 显式置零;(c) mask 换成 HSTU 5 因子(pane-2)。详见 §2。
- **删掉** FMHA 的 bias / dbias / dropout / position_encoding 分支(HSTU 无这些;BiasEnum 恒 NONE、`FmhaDropout::IsDropout=false`)。这些都是 `if constexpr` 死分支,模板实例化时自动消失,但建议出一个 HSTU 精简版 pipeline 以降低模板复杂度(见 §5 复用清单)。

### POST(`BlockFmhaBwdConvertDQ`,deterministic dQ 归约)
- **决策:与 FMHA 完全一致,仅在 `kIsDeterministic=true` 时发射。**
- 依据:dQ 在 MAIN 里是「对每个 kN0 块 atomic 累加到同一行」(MAIN 行 749-756:deterministic 时 `store` 到 `dq_acc` split buffer,否则 `update_tile` 直接原子加)。HSTU 的 dQ 语义同 FMHA(每个 sq 行被多个 sk 块贡献),归约逻辑与激活无关 → 直接复用。
- 非 deterministic(默认)不需要 POST,MAIN 内 `update_tile` 原子累加即可。

> 结论:**PRE 编译期可跳过(SiLU)/POST 条件发射(deterministic)/MAIN 必改**。三段都基于 FMHA,改动面 = MAIN 的两个 elementwise 段。

---

## 2. MAIN 七阶段的 HSTU 改造(逐阶段对照 FMHA)

记号:`s_acc` = gemm_0 的 C tile(`SPBlockTileType`,FMHA 行 477);`p` / `g` = 同分布的派生 tile;`dp_acc` = gemm_2 的 C tile(`SPGradBlockTileType`)。FMHA 中 `p` 与 `dp_acc` 同 `i_j_idx` 索引(行 665),故 `s_acc`/`p`/`g` 与 `dp_acc` 在 STAGE5 可逐元素混算——这是 HSTU 留 S 不需要额外 LDS 的关键。

### STAGE1 — Q@K → S(行 515-518)
- `s_acc = gemm_0(q_reg_tensor, k_reg_tensor)`,**原样复用**,产出**未缩放**的 `Q·Kᵀ`。
- alpha 不在这里乘(GEMM 不带 scale),留到 STAGE2 头部,理由见下。

### STAGE2 — 重算激活(双路;替换 FMHA 行 520-622)
FMHA 这段是 scale+bias+mask+softmax+dropout。HSTU 替换为:

**公共第 0 步(两路都做):把 alpha 乘进 s_acc**
```
tile_elementwise_inout([&](auto& x){ x = alpha * x; }, s_acc);   // s_acc 现在是真正的 S=alpha·Q·Kᵀ
```
为什么放这里而不是折进 exp(像 FMHA 把 raw_scale 折进 exp2):因为 **SiLU 必须拿到已缩放的 S 喂 dsilu**,无法靠 exp 折叠规避;为两路统一,softmax 路也先物化 `s_acc*=alpha`。代价仅一次逐元素乘,可忽略。

**SiLU 路(`kUseSoftmax=false`,默认)**
```
// 1) 计算 P = silu(S)*scale_p  —— 供 STAGE3 dV
p = silu(s_acc) * scale_p;                  // silu(x)=x*sigmoid(x)
// 2) 计算并保留 dsilu 因子 g = scale_p * dsilu(s_acc) —— 供 STAGE5 dS
g = scale_p * dsilu(s_acc);                 // dsilu(x)=sig*(1+x*(1-sig))
// 3) masked-out 显式置零(见下「置零」小节):对 edge tile 把 p、g 越界元素清 0
```
- **关键:留 `g`(dsilu 因子)而非留整张 `S`。** `g` 与 `p` 同分布、同生命周期模式(`p` 活到 STAGE3、`g` 活到 STAGE5),峰值寄存器与 FMHA(`p` 活到 STAGE5)同量级,且 STAGE5 省掉重算 dsilu。若实现上更省事,也可改为「保留 `s_acc`(=S)到 STAGE5、在 STAGE5 再算 dsilu」——二选一,推荐留 `g`。
- LSE **不读**(SiLU 路 lse_dram_window 传 null,见 §3/§4)。

**Softmax 路(`kUseSoftmax=true`)**
```
P[i_j] = exp(s_acc[i_j] - lse[i])           // lse 来自 fwd(自然对数域,见 §3)
```
- 可直接照搬 FMHA 的 log2e/exp2 形式:`p = exp2(log2e*s_acc - log2e*lse)`(数学等价 `exp(S−LSE)`,exp2 走硬件指令)。因 alpha 已在第 0 步乘进 s_acc,**不要**再像 FMHA 那样在 exp2 里乘 scale。
- masked-out:softmax 路可沿用 FMHA 的「越界 set S=-inf → P=exp(-inf)=0」(行 557-568),自然零,**无需**像 SiLU 那样显式补零。

**置零小节(SiLU 专属,最易错点)**
- 现象:reference 对 SiLU 把 masked-out 的 S 置 **0**(line 276,不是 -inf),于是 `P=silu(0)*scale_p=0` 自然为零(dV 正确);**但** `dsilu(0)=0.5≠0`,若不处理则 `dS=dP*scale_p*0.5≠0`(line 380-385 用 `if(IsTokenPairInsideMask) … else 0` 强制清零)。
- 不能用 -inf 规避:`dsilu(-inf)` 数值上是 `0*(-inf)`→NaN(`exp(+inf)` 溢出),所以 SiLU **禁止**走 -inf 那条 FMHA 路径。
- GPU 落地:沿用 FMHA 的 edge-tile 检测(行 557-568 的 `mask.IsEdgeTile` + `set_tile_if`),但谓词动作改为——对越界 `(row,col)`:`p(idx)=0` 且 `g(idx)=0`(而非 set s=-inf)。内部全 masked-in 的 tile 跳过逐元素检查。mask 谓词由 pane-2 提供(`IsOutOfBound`/`IsTokenPairInsideMask` 的 tile 版)。
- 这一步同时保证 dV 不被 masked-out 污染(p=0)、dS/dK/dQ 不被污染(g=0 → ds=0)。

### STAGE3 — dV += Pᵀ@dO(行 624-646)
- `gemm_1(dv_acc, pt_reg_tensor, dot_reg_tensor)`,**原样复用**。`p` 在 SiLU 已含 scale_p,softmax 已是归一化 P → dV 数值正确。
- 维度:gemm_1 沿 `kVHeaddim`(hdim_v),与 hdim_qk 解耦,FMHA 本就支持 hdim_qk≠hdim_v,无需改。
- dV **不乘任何 scale**(收尾 §STAGE-final 也不碰 dv_acc),与 reference(line 479-487 dV 直接写、无 alpha)一致。

### STAGE4 — dP = dO@Vᵀ(行 648-655)
- `dp_acc = gemm_2(do_reg_tensor, v_reg_tensor)`,**原样复用**。沿 hdim_v 归约。

### STAGE5 — dS(双路;替换 FMHA 行 657-669)
FMHA 原式 `ds = p*(dp - d)`。HSTU:
```
if constexpr(!kUseSoftmax)   // SiLU
    ds[i_j] = dp_acc[i_j] * g[i_j];          // g = scale_p*dsilu(S);masked-out 已经 g=0 → ds=0
else                          // Softmax —— 与 FMHA 同式
    ds[i_j] = p[i_j] * (dp_acc[i_j] - d[i]); // d 来自 PRE 的 D[sq]
```
- SiLU 分支:不读 `d`(D window 为 null);依赖 STAGE2 已把 `g` 越界清零,这里无需再判 mask(等价 reference line 380-385,把判断前移到 STAGE2 更省)。
- Softmax 分支:几乎逐字复用 FMHA(去掉 dropout 的 `undrop_flag` 三元,HSTU 无 dropout → `ds=p*(dp-d)`)。
- 之后的 `cast_tile<GemmDataType>(ds)`(行 702)两路共用,不变。

### STAGE6 — dK += dSᵀ@Q(行 698-708)
- `gemm_3(dk_acc, dst_reg_tensor, qt_reg_tensor)`,**原样复用**。alpha 不在此乘,留收尾。

### STAGE7 — dQ = dS@K + 写回(行 718-757)
- `gemm_4(dq_acc, ds_reg_tensor, kt_reg_tensor_slice)` 循环复用。
- **收尾 scale 用 alpha 替换 raw_scale**(行 747):`x = x * alpha`(HSTU 无 dropout,走 `else` 分支)。
- 写回:`update_tile`(默认原子累加)/ deterministic 时 `store_tile` 到 split buffer(行 749-756),原样复用。

### STAGE-final(循环后,行 763-775)
- `dk_acc * alpha`(行 772 把 raw_scale 换 alpha);`dv_acc` **不乘**。与 reference(dK×alpha line 474-477、dV 无 scale)一致。
- 返回 `make_tuple(dk_acc, dv_acc)` 不变。

> 一句话:**alpha 出现在两处(STAGE2 头 + dQ/dK 收尾),scale_p 折进 `p` 与 `g`,dV 两个 scale 里只吃 scale_p、不吃 alpha**。与 FMHA `raw_scale` 的两处出现(STAGE2 折 exp + dQ/dK 收尾、dV 不碰)结构同构——这是「alpha == FMHA raw_scale 槽」论断的依据。

---

## 3. fwd 副产物契约(每路 bwd 的输入集合)

| 路径 | bwd 输入集合 | fwd 需存什么 | 备注 |
|---|---|---|---|
| **SiLU**(`kUseSoftmax=false`,默认) | `{Q, K, V, dO}` | **无额外副产物** | S 在 bwd 重算;无 LSE、无 O、无 D、无 PRE |
| **Softmax**(`kUseSoftmax=true`) | `{Q, K, V, dO, O, LSE}` | **LSE**(fwd 存)+ **O**(给 PRE 算 D) | D 由 PRE 现算,不是 fwd 存的 |

- **LSE 接线柱已就绪**:fwd `with_softmax` pipeline 在 `LSEaccDramBlockWindowTmp` 非空时存 `lse = m + log(l)`(`hstu_attention_with_softmax_fwd_pipeline.hpp:604-613`),**自然对数域**。故 bwd softmax 必须用 `P=exp(S−LSE)`(自然 exp;若用 exp2 须自带 log2e 转换,如 §2)。`no_softmax`(SiLU)pipeline 不产 LSE — 与「SiLU 不需要 LSE」自洽。
- **O 的来源**:softmax bwd 的 PRE 需要 fwd 的输出 O(`D=rowsum(O⊙dO)`)。O 本就是 fwd 的主输出(`o_ptr`),bwd 直接读,无需额外存储。
- **D 的生命周期**:PRE 产 → DRAM(`d_ptr`)→ MAIN STAGE5 读。SiLU 路这条链整体不存在。

---

## 4. 关键骨架伪代码

### 4a. MAIN `operator()` 双路骨架(只标改造点,其余「同 FMHA」)
```cpp
// ... [同 FMHA] KV 预载入寄存器、LDS 编排、early-exit (行 151-491) ...
//     差异:lse/d 的 DRAM window 在 SiLU 路传 null tile window(host 侧不挂 buffer)

while (i_total_loops < num_total_loop) {
    // ── 载 Q (+ SiLU 路不载 lse) ───────────────── [同 FMHA 行 496-513]
    auto q_reg = load_tile(q_lds_read_window);
    auto lse   = kUseSoftmax ? load_tile(lse_lds_read_window) : /*未使用*/;

    // ── STAGE1: S = Q@K ───────────────────────── [同 FMHA 行 518]
    auto s_acc = gemm_0(q_reg, k_reg_tensor);

    // ── STAGE2: alpha + 激活双路 (替换 FMHA 行 520-622) ──
    tile_elementwise_inout([&](auto& x){ x = alpha * x; }, s_acc);   // ★ alpha 入 S
    SPBlockTileType p, g;
    if constexpr (!kUseSoftmax) {                 // SiLU
        p = silu(s_acc) * scale_p;                // → dV
        g = scale_p * dsilu(s_acc);               // → dS(留到 STAGE5)
        if (mask.IsEdgeTile(...))                 // ★ 显式置零(pane-2 谓词)
            set_tile_if(p,g <- 0, [&](idx){ return mask.IsOutOfBound(row,col); });
    } else {                                      // Softmax
        if (mask.IsEdgeTile(...)) set_tile_if(s_acc, -inf, IsOutOfBound);  // [同 FMHA]
        p = exp2(log2e*s_acc - log2e*lse);        // = exp(S-LSE)
    }
    auto p_gemm = cast_tile<GemmDataType>(p);

    // ── STAGE3: dV += Pᵀ@dO ───────────────────── [同 FMHA 行 624-646]
    gemm_1(dv_acc, pt_reg(p_gemm), dot_reg);
    // ── STAGE4: dP = dO@V ─────────────────────── [同 FMHA 行 648-655]
    auto dp_acc = gemm_2(do_reg, v_reg_tensor);

    // ── STAGE5: dS 双路 (替换 FMHA 行 657-669) ──
    SPGradBlockTileType ds;
    if constexpr (!kUseSoftmax)                   // SiLU:masked-out 已由 g=0 保证 ds=0
        sweep: ds[i_j] = dp_acc[i_j] * g[i_j];
    else                                          // Softmax:同 FMHA(去 dropout)
        sweep: ds[i_j] = p[i_j] * (dp_acc[i_j] - d[i]);   // d 来自 PRE
    auto ds_gemm = cast_tile<GemmDataType>(ds);

    // ── STAGE6: dK += dSᵀ@Q ───────────────────── [同 FMHA 行 698-708]
    gemm_3(dk_acc, dst_reg(ds_gemm), qt_reg);
    // ── STAGE7: dQ = dS@K, ×alpha, 写回 ──────────
    gemm_4(dq_acc, ds_reg(ds_gemm), kt_reg_slice);          // [同 FMHA 行 722-737]
    tile_elementwise_inout([&](auto& x){ x = x * alpha; }, dq_acc);   // ★ raw_scale→alpha
    kIsDeterministic ? store_tile(...) : update_tile(...);  // [同 FMHA 行 749-756]
}
// ── 收尾:dK ×alpha;dV 不乘 (替换 FMHA 行 763-773) ──
tile_elementwise_inout([&](auto& x){ x = x * alpha; }, dk_acc);   // ★
return make_tuple(dk_acc, dv_acc);
```

### 4b. PRE 是否跳过的编译期逻辑(host dispatch 层)
```cpp
// 仅 softmax 路才算 D;SiLU 路不发射 PRE、不分配 D buffer
if constexpr (kUseSoftmax) {
    bwd_dot_do_o_kernel(o_ptr, do_ptr, /*->*/ d_ptr, /*p_undrop=*/1.0f);  // PRE 零改造复用
}
// MAIN:
bwd_dq_dk_dv_kernel(Q,K,V,dO,
                    /*lse=*/ kUseSoftmax ? lse_ptr : nullptr,   // SiLU 传 null window
                    /*d  =*/ kUseSoftmax ? d_ptr   : nullptr,
                    dQ,dK,dV, alpha, scale_p, mask...);
// POST:仅 deterministic
if constexpr (kIsDeterministic) bwd_convert_dq_kernel(dq_acc_ptr, /*->*/ dq_ptr);  // 零改造
```

---

## 5. 复用 vs 新写清单(算法层)+ 风险

### 直接复用(零/极小改动)
- **PRE** `BlockFmhaBwdOGradDotO`:零改造(softmax 路;`p_undrop=1`)。
- **POST** `BlockFmhaBwdConvertDQ`:零改造(deterministic 路)。
- **MAIN 全部 5 GEMM**(gemm_0..4)、KV-resident/Q-dO 流式 LDS 编排、shuffle、early-exit(行 159-174)、STAGE1/3/4/6/7 的 GEMM 调用与写回。
- bwd `pipeline_problem` / `default_policy` / `pipeline_enum`(KRKTRVR)尽量继承 FMHA bwd。

### HSTU 新写 / 特化(算法层)
1. **STAGE2 重写**:alpha 物化 + SiLU/softmax 双路激活 + SiLU 显式置零(替换 FMHA 行 520-622 的 bias/softmax/dropout 整段)。
2. **STAGE5 重写**:dS 双路(替换行 657-669)。
3. **收尾 scale**:raw_scale → alpha(STAGE7 行 747、final 行 772)。
4. **删死分支**:bias/dbias/dropout/position_encoding/`get_validated_lse` 的 bias 分支 → 出一份 HSTU 精简 MAIN pipeline(`block_hstu_bwd_dq_dk_dv_pipeline.hpp`)。
5. **新增 device 标量**:`silu` / `dsilu`(逐元素,见 oracle line 145-156),作 tile_elementwise 函子。
6. **新模板形参**:`kUseSoftmax`(决定 STAGE2/5 分支 + PRE 发射 + lse/d window 是否 null);沿用 `kIsGroupMode`/`kIsDeterministic`/pad 系列。

### 风险 / 未决问题
- **R1 SiLU 留 S/g 的寄存器压力**:SiLU 路比 FMHA 多一张 `SPBlockTile`(`g` 或 `S`)活到 STAGE5。需确认在目标 hdim(64/128/256)下 VGPR 不溢出、不掉 occupancy。缓解:留 `g`(而非 S)使生命周期与 FMHA 的 `p` 同形;必要时把 `g` 暂存 LDS(复用 bias 腾出的 LDS 空间)。**待 pane-3 在真实 tile size 上验证寄存器分配。**
- **R2 双路模板膨胀**:`kUseSoftmax × kIsGroupMode × kIsDeterministic × hdim × dtype` 实例数翻倍。SiLU 是默认且更轻(无 PRE/LSE/D),建议 codegen 优先覆盖 SiLU,softmax 按需生成(pane-3 的 `generate_instances.py`)。
- **R3 SiLU masked-out 置零的正确性**:`dsilu(0)=0.5`,必须显式清 `g`;且**禁用** -inf 路径(NaN 风险)。置零必须发生在 dS 形成之前(本设计放 STAGE2 清 `p`/`g`)。**与 pane-2 的 mask tile 谓词强耦合**——需 pane-2 提供 `IsEdgeTile`/`IsOutOfBound`/`IsTokenPairInsideMask` 的 block-tile 版本。
- **R4 LSE 数值域**:fwd 存自然对数 LSE;bwd 若用 exp2 须自带 log2e 转换。alpha 已在 STAGE2 物化,**切勿**在 exp2 里再乘一次 scale(FMHA 会乘,HSTU 不可)——双重 scale 是高危 bug 点。
- **R5 group 模式逐段超参**:`alpha`/`scale_p`/`window`/`contextual`/`min_full_attn`/`num_target` 在 group 路按 group 取(oracle line 585-591)。算法上 STAGE2/5 用到 `alpha`、`scale_p`;这两个标量须按 i_group 喂入。**索引/取数留给 pane-2/pane-3**,算法只假设「进入 pipeline 时 alpha/scale_p 已是本 batch 的正确值」。
- **R6 scale_p 默认值**:`scale_p = attn_scale ? attn_scale : 1/max_seqlen_q`(oracle line 165)。该 fallback 计算放 host(params 准备),pipeline 只收最终 `scale_p` float。

---

## 6. 对 pane-2 / pane-3 的接口假设

### 给 pane-2(mask / 索引 / 模式)
- STAGE2/5 需要 **block-tile 粒度**的 mask 谓词:`mask.IsEdgeTile(q_step,k_origin,kM0,kN0)`、`mask.IsOutOfBound(row,col)`(沿用 FMHA 签名,行 559-566),底层换成 HSTU 5 因子 `HstuCrossAttentionBlockMaskWithLocal`。
- early-exit 复用 `mask.GetTileRangeAlongY(...)`(行 160-161),需 HSTU mask 实现该接口。
- alpha/scale_p 等标量,假设按当前 batch/group **已解析为单一 float** 传入 pipeline(group 逐段取数由 pane-2/3 负责)。
- SiLU 置零谓词:masked-out 判定须与 fwd **逐元素一致**(同一个 mask 对象),否则重算 P/S 与 fwd 不符。

### 给 pane-3(工程 / params / codegen)
- **params 需补 bwd 字段**:`dq_ptr/dk_ptr/dv_ptr/do_ptr`、softmax 路 `o_ptr(已存在)/lse_ptr/d_ptr`、deterministic 路 `dq_acc_ptr`,及各自 stride;复用 fwd 的 `scale_s`(=alpha)、`attn_scale`(→ host 算 scale_p)。
- **三 kernel 发射**:`if constexpr(kUseSoftmax) PRE` → MAIN → `if constexpr(kIsDeterministic) POST`(见 §4b)。SiLU 路不分配 `lse/d/o`(o 实际仍是 fwd 输出,bwd SiLU 不读)。
- **null window 协议**:SiLU 路 lse/d 的 DRAM window 传 null tile window,pipeline 内 `if constexpr(kUseSoftmax)` 守卫读取(本设计已假设此协议)。
- **模板轴**:`kUseSoftmax / kIsGroupMode / kIsDeterministic / hdim_qk / hdim_v / dtype`;建议 SiLU 优先全覆盖,softmax 按需。
- `silu/dsilu` 作为 device 函子放 HSTU 侧 header,供精简 MAIN pipeline include。
