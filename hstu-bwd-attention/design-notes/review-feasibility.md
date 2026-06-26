# Review · 工程可行性(pane-3 / reviewer)— HSTU bwd DESIGN.md

> 对抗式 review,以源码为唯一判据,不改正文。结论:**方案整体可行,无 P0 致命问题**;DESIGN 的逐 stage 行号引用经核验**全部准确**(罕见)。但有 **5 个 P1**(其中 1 个是 §1.1 与 §4.4 的真实内部矛盾,1 个会让"复用 default policy"在字面执行下编译失败)。
> 核验基准:`block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp`(已通读 779 行)、`block_fmha_bwd_pipeline_default_policy.hpp`、`fmha_bwd_kernel.hpp`、`hstu_block_masking.hpp`、`hstu_attention_with_softmax_fwd_pipeline.hpp`、rocm-ref。

---

## 总判定

| 维度 | 判定 | 一句话 |
|---|---|---|
| 总体架构(3-kernel / KV 外 Q 内 / 复用边界)| ✅ | 与 FMHA 源码同构,可落地 |
| 行号/事实引用准确性 | ✅ | 抽查 STAGE1-7、raw_scale、GridSize、LSE 域、mask hook 全部命中 |
| "PRE/POST/policy/shape/enum 近零成本复用" | ⚠️ | PRE/POST/shape/enum 成立;**policy 复用有前提**(P1-A) |
| "MAIN 仅特化 STAGE2/5" | ✅ | 改造面判断准确;留 g 与 FMHA 留 p 同构(低风险) |
| 内部自洽 | ⚠️ | §1.1 vs §4.4 关于 atomic 路是否发射 POST **矛盾**(P1-E) |

---

## P0(不可行 / 会算错)—— **无**

通读 MAIN pipeline 后确认:DESIGN 主张的"FMHA MAIN 作结构蓝本、mask 经 `GetTileRangeAlongY`/`IsEdgeTile`/`IsOutOfBound` 接入、alpha 走 `raw_scale` 槽、SiLU 留中间 tile 喂 STAGE5"——**全部有源码对应、无逻辑漏洞**。M1 闸门设计正确,确实一次性压到 R1/R2 两个真风险。故无 P0。

---

## P1(风险大,需缓解)

### P1-A ⚠️ 复用 default policy 必须**保留** `BiasEnum` + `BiasDataType`(§4.2 字面会编译失败)
- **证据**:`block_fmha_bwd_pipeline_default_policy.hpp:1627-1628` 在 `GetSmemSizeBias()` 内 `if constexpr(Problem::BiasEnum==ELEMENTWISE_BIAS) return sizeof(Problem::BiasDataType)*...`;该函数被 `GetSmemSize<Problem>()`(:1641-1647)无条件聚合,而 pipeline `GetSmemSize()`(:80-83)→ `Policy::GetSmemSize<Problem>()` **必然实例化**。即只要复用 default policy,`Problem` 就**必须**有 `BiasEnum` 与 `BiasDataType` 两个成员。
- **冲突点**:DESIGN §4.2 "`HstuAttentionBwdPipelineProblem` **砍** RandValOutput/FmhaDropout/BiasGrad/kHasBiasGrad" + §4.6 "删 dropout+bias+dbias 三组 kargs"。若连 `BiasEnum/BiasDataType` 一并从 Problem 删掉 → default policy 不编译 → "policy 直接复用"破产、被迫 fork。
- **核实**:policy **不**引用 `FmhaDropout/RandValOutputDataType/kHasBiasGrad/BiasGradDataType`(grep 0 命中),故这四个砍掉**安全**;**只有** `BiasEnum/BiasDataType` 必须留。
- **修法(低成本)**:HSTU `Problem` 保留 `static constexpr auto BiasEnum = BlockAttentionBiasEnum::NO_BIAS;` + `using BiasDataType = InOutDataType;`(dummy)。kargs 不传 bias 即可。这把 R1 从"政策不确定"收敛成"保留 2 个 dead typedef"。**这也顺带回答 U1**(见下)。

### P1-E ⚠️ §1.1 与 §4.4 矛盾:atomic(默认)路**也要发射 POST**
- **证据**:HSTU `dq_ptr` 是 bf16/fp16(`InOutDataType`),而 atomicAdd 只能落 float → §4.4 路 A 自己写明"float `dq_acc` + atomic + **POST convert-only**(cast→dq_ptr)"。FMHA MAIN 的 `update_tile(dq_dram_window, dq_acc)`(:755)写的是 float dq_acc,**必须**再有一个 cast 通道。
- **矛盾**:§1.1 却写"非 deterministic(默认)由 MAIN 内 atomic 累加,**不发射 POST**"。二者不能同真。
- **结论**:§1.1 表述错误。**正解:POST 在两路都发射**——atomic 路 convert-only、deterministic 路 reduce+convert。(M1 §6 写的"atomic+POST convert-only"是对的,只有 §1.1/§1.4 的"不发射 POST"措辞错。)
- **影响**:kernel 计数与 host 编排(每次 bwd 恒 2~3 个 kernel,而非"默认 1 个 MAIN")。dq_acc float workspace 在 atomic 路**也必须分配**(nsplits=1),DESIGN §4.4 已隐含,但 §1.1 给人"默认零额外 workspace"的错觉。**修法**:统一为"MAIN→POST 恒发射;PRE 仅 softmax"。

### P1-B ⚠️ 留 `g` 的 VGPR 压力:在已近满载的 pipeline 上 +1 个 SP tile
- **证据**:MAIN 是 MFMA 重 + KV/Kᵀ/Qᵀ/dOᵀ 常驻寄存器 + `dk_acc/dv_acc/dq_acc` 三累加器(:148-149,719);FMHA `p`(SPBlockTileType)活到 STAGE5(:664)。HSTU SiLU 需 **同时**留 `p`(silu→dV@STAGE3)与 `g`(dsilu→dS@STAGE5)→ STAGE2-3 期间峰值 **+1 个 SPBlockTileType**。
- **量级**:kM0×kN0=64×128、256 线程 → 每线程 32 个 f32 ≈ **+32 VGPR**。
- **硬件**(rocm-ref `occupancy-register-pressure.md:26`、`vgpr-sgpr-agpr.md:291`):gfx942 CDNA3 = 512 ArchVGPR/SIMD,占用=max(ArchVGPR,AGPR),≤8 waves。FMHA bwd kr_ktr_vr 经验在 1~2 wave 区间。+32 VGPR 在 1~2 wave 区**多半可吸收**,但若已逼近 512 → 掉到 1 wave 或触发 scratch spill(每次 spill ≈400 cycle,`vgpr-sgpr-agpr.md:259`)。
- **修法**:M1 在 hdim64 看 `ScratchSize`(必须=0)与 VGPR 数;溢出则按 R2 把 `g` 暂存 LDS(复用删掉的 bias LDS,`GetSmemSizeBias` 区段)。**注**:留 g 与 FMHA 留 p 的生命周期同形,distribution 兼容性已被 FMHA STAGE5 跨 tile-type 索引(`p[i_j_idx]` over gemm_2 spans,:664)证明可行 → 这是**低实现风险、中资源风险**。

### P1-C ⚠️ `GetTileRangeAlongY` 对 5 因子 mask 的转置正确性(唯一非平凡新成员)
- **证据**:四个 HSTU mask struct(`hstu_block_masking.hpp:12/268/503/656`)均有 `GetTileRangeAlongX`(:79/319/549/685)、`IsTokenPairInsideMask`、`IsFullTileInsideMask`,**全部缺** `GetTileRangeAlongY`。而 MAIN **无条件**调用 `mask.GetTileRangeAlongY(k_origin, kM0, kN0)`(:161,不受 `IsMasking` 守卫)。
- **风险**:contextual/min_full/num_target 叠加时,attend 某 KV-tile 的 Q-行集合可能**非连续**(如 contextual 顶块 + 对角带 + target 尾块),而 `GetTileRangeAlongY` 只能返回单段 `[y_start,y_end)`。DESIGN 的"保守超集"(放宽到连续超集,宁多算不漏)是**正确的安全策略**,但 min_full 叠加下可能退化成近全扫(perf,非正确性)。
- **判定**:✅ 可行且安全,但**离线校验是硬性前置**。DESIGN §5.4 的断言(`[y_start,y_end)⊇{sq:∃sk∈tile,IsTokenPairInsideMask}`)必须在 M2 落地。`IsEdgeTile := !IsFullTileInsideMask` 是平凡 wrapper(注意 `IsFullTileInsideMask(i_tile_top,i_tile_left,number<TileWidth>,number<TileHeight>)` 的 4 参顺序,:237);`IsOutOfBound` 可直接用 `!IsTokenPairInsideMask(row,col)` 内联(HSTU 自写 MAIN,无需新增第三个成员)。→ **实测净新增:1 个非平凡成员(GetTileRangeAlongY)+ 1 个 wrapper**,DESIGN "方案 A 加 2 成员"基本准确。

### P1-D ⚠️ M1 "no-mask" 仍需先实现(平凡)`GetTileRangeAlongY`
- **证据**:`:160-161` 的 `GetTileRangeAlongY` 调用在 `IsMasking` 守卫**之外**(:166 的 early-exit 才受守卫)。故即便 M1 的 no-causal mask(`IsMasking=false`),也必须提供返回 `(0,seqlen_q)` 的 `GetTileRangeAlongY`。
- **影响**:M1 工作量被低估了一点——M1 不是"零 mask 成员",而是"至少 1 个平凡 GetTileRangeAlongY"。不阻塞(平凡),但里程碑估时应纳入。

---

## P2(可选 / 引用纠偏)

- **P2-1 PRE GridSize 除数**:DESIGN §1.3 写 `ceil(seqlen_q, kBlockSize)`,源码 `fmha_bwd_kernel.hpp:1733` 实为 `ceil(seqlen_q, kM0)`(POST :2011 同为 kM0)。需确认 OGradDotO kernel 的 `kM0==kBlockSize`(dot_do_o 断言块长=kBlockSize,`block_fmha_bwd_dot_do_o.hpp:48`)。结论方向(PRE/POST 沿 seqlen_q、MAIN 沿 seqlen_kv)✅,仅除数符号需对齐。
- **P2-2 §4.6 字段**:LSE/D 的 `batch_stride_lsed` 未在表中显式列出(只列 `nhead_stride_lsed`);batched softmax 路需要。补全即可。
- **P2-3 SiLU 路 LDS**:若 HSTU MAIN 复用 `Policy::GetSmemSize`,SiLU 路仍会计入 `GetSmemSizeLSE/D`(:1570/1579)→ 浪费 LDS。建议 HSTU MAIN 自算 smem(SiLU 减去 LSE/D 段),作为 M1 后的 perf 项。
- **P2-4 §2.3 骨架简化**:`gemm_4(dq_acc, ds, kt_slice)` 实为 `k4_loops` 切片循环(:722-737);骨架"只标改造点"可接受,实现时注意。

---

## 逐条审查(任务 1-8)

**1. M1 风险闸门 / FMHA MAIN 可复用性 — ✅(附 P1-A/B)**
通读 MAIN 后给出真实判定:
- **可直接复用**:5 个 GEMM(`gemm_0..4`,:141-145)、KV-resident + Kᵀ/Qᵀ/dOᵀ shuffle/LDS 编排(:151-370)、early-exit(:159-174)、STAGE1/3/4/6/7 的 GEMM 调用与写回、`raw_scale` 槽(:109,747,772)、`update_tile`/`store_tile` 双写回(:749-756)。
- **需小改(在自写 MAIN 内)**:STAGE2(:520-622,删 bias/dropout/exp,换 alpha 物化 + silu/dsilu 或 softmax)、STAGE5(:657-669,删 dropout 三元,换双路 dS)。
- **必须 fork/特化、不能直接 include**:整支 MAIN(因 STAGE2/5 改写 + 删 LSE/D 加载的 if constexpr)。DESIGN 本就说 MAIN 是"结构蓝本不直接 include",✅。
- **硬编 softmax 假设点**(DESIGN 已覆盖):STAGE2 恒 `exp2`(无 kUseSoftmax 轴,:585-604)、恒 `load_tile(lse)`(:499,511)与 `load_tile(d)`(:628,650)。SiLU 路必须 `if constexpr` 去除这些加载(传 null window 不够,会读空指针)。DESIGN §2.3 骨架用 `if constexpr` 正确处理。
- **policy 复用的真前提** = P1-A。

**2. mask Y 成员新增 — ✅(= P1-C)**
工作量现实:1 非平凡(GetTileRangeAlongY,GetTileRangeAlongX 的转置)+ 1 wrapper(IsEdgeTile)。正确性靠离线校验(§5.4)兜底。否决方案 C(FMHA generic mask 表达不了 contextual/min_full/num_target)判断正确。

**3. params/kargs 字段表(§4.6)— ✅(附 P2-2)**
对照 `fmha_bwd_kernel.hpp` kargs:`dq_acc_ptr`(:129)、`stride/nhead/batch/split_stride_dq_acc`(:149/158/281/261)全部有对应;DESIGN 表齐全。group 指针、lse_ptr/d_ptr、dq/dk/dv stride 完整。仅缺 `batch_stride_lsed`(P2-2)。命名 `seqlen_kv/hdim_qk` 改名合理。

**4. dQ 两路 + convert_dq 复用 — ✅(矛盾见 P1-E)**
`BlockFmhaBwdConvertQGrad` 的 Convert-only(:37-61)与 Reduce+Convert(:64-138,沿 split 维 do-while 归约)可直接复用。deterministic split offset 机制(`split_stride_dq_acc`,kernel :261/467/855)与 DESIGN §4.4 一致。dq_acc 形状/stride 估算合理。**唯一问题是 §1.1 误称 atomic 路不发 POST(P1-E)**。

**5. instances/codegen + CMake — ✅**
不搬 FMHA `codegen/ops/fmha_bwd.py`(那套依赖 `cpp_symbol_map/cmake_config`,与 HSTU 例隔离)、扩 HSTU `generate_instances.py` 字符串模板——可行且与现状一致。384→MVP 48 收敛合理(砍 bias 靠 P1-A 的 NO_BIAS 默认即可,无需删 Problem 成员)。`file(GLOB instances/*.cpp)` 自动纳入。编译规模风险中等(bwd instance ~3× fwd,5 GEMM),建议如 R5。

**6. 3-kernel GridSize / launch — ✅(除数 P2-1)**
MAIN=`(ceil(seqlen_kv,kN0),nhead,batch)`(:1064)✅;PRE/POST 沿 seqlen_q(:1733/2011,除数 kM0 见 P2-1)✅;launch `[PRE if softmax]→MAIN→[POST]`(P1-E 修正后两路恒含 POST)。group/jagged 经 `seqstart_*_ptr[i_batch]` 求 offset,与 FMHA group 分支(:1110-1159)同构 ✅。

**7. 里程碑 M0-M8 + 测试矩阵 — ✅(M1 估时见 P1-D)**
阶梯合理,M1 精准命中 R1(policy)+R2(VGPR)。对拍流程可操作:oracle 签名已核实(`reference_hstu_attention_bwd.hpp:63/501`),fwd LSE 存 `m+log(l)` 自然对数域(`with_softmax_fwd_pipeline:609`)与 bwd `exp(S−LSE)` 用法自洽 ✅。容差分张量、deterministic 逐位 memcmp、masked-out dS=0 专项均可测。唯一:M1 "no-mask" 仍需平凡 GetTileRangeAlongY(P1-D)。

**8. 占用率 / 资源 — ⚠️✅(= P1-B)**
gfx942:512 VGPR/SIMD、occupancy=max(ArchVGPR,AGPR)、≤8 wave。kr_ktr_vr 本就 1~2 wave;+g(~32 VGPR@64×128)多半可吸收但需 M1 验 `ScratchSize=0`。rocm-ref 印证"MFMA kernel 低占用仍可高吞吐"(`wavefront-scheduling.md:41`)→ 1~2 wave 可接受。gfx950(CDNA4)512 unified pool、加法式占用,压力略升,按宏分叉的 tile 预设需各自验。

---

## 对 U1-U4 真未决的工程视角建议

- **U1 bias/dbias — 建议:MVP 不支持 dbias,但 Problem **保留** `BiasEnum=NO_BIAS`+`BiasDataType` dummy(P1-A 强制)**。fwd `bias_ptr` 实际未用 → 下游大概率不需要 bias 梯度。保留 typedef 是 policy 复用的免费副产物,真 dbias 路(STAGE5 后存,FMHA :671-696 有蓝本)留作 post-MVP。**默认按不需要推进。**
- **U2 deterministic 默认 — 建议:atomic 为默认(✅),但须明确两路都过 POST(P1-E)**。atomic 路省的是 split-workspace(nsplits=1 vs nsplits 份)与归约开销,**不省 POST、不省 float dq_acc**。若业务要逐位可复现为默认,M6 需前置且接受 `[nsplits,...]` 显存(R4)。
- **U3 MVP 覆盖面 — 建议:SiLU 全覆盖优先(它是默认路 + M1 闸门),softmax 次之;bf16 优先(`CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3` 已设),fp16 跟进;hdim 先 {64,128}**。三模式建议 batched→jagged→group 顺序(M3/M4),不必首版全要。
- **U4 GQA/MQA — 建议:首版 `nhead_ratio_qk=1`(✅)**。HSTU 生成式推荐场景通常 MHA。FMHA 已有 ratio 支持;若后续模型共享 KV,需在三个 kernel 的 `i_nhead` 索引引入 ratio(KV head = q head/ratio)。留 1 个常量字段占位,改动可控。**默认按不需要推进。**

---

## 附:DESIGN 已正确处理、值得肯定的点(对抗式复核未推翻)
- alpha 落点(STAGE2 头 + dQ/dK 收尾、**dV 不吃**)与 FMHA `raw_scale`(:747 dq、:772 dk、dv 不乘)**逐字吻合**。
- softmax `exp2(log2e·s − log2e·lse)` 在 s_acc 已 alpha 物化后正确(等价 FMHA :601 的 scale 折叠);"切勿在 exp2 再乘 scale"的警告是真高危点,处理对。
- SiLU 留 g 的 distribution 兼容性,被 FMHA STAGE5 既有的 `p`(gemm_0 C)over `ds_spans`(gemm_2 C)跨 tile-type 索引(:660-668)证明可行 → 留 g 是低实现风险选择。
- `is_tile_in_first_split` 的 D2 裁决(元素级 `IsTokenPairInsideMask` 自洽、不依赖该 flag)与源码一致(mask 元素级判定不读该 flag)。
