# docs/draft-cross-attn.md — HSTU bwd Milestone: cross-attention (seqlen_q != seqlen_kv)

> 状态:**draft 闸门稿,等 lead/用户审批,未写实现码 / 未改库 / 未 build。** 基线 = M7c-done (HEAD per docs/M7c-done.md)。
> 来源:cross-attn 设计 workflow(6 路并行分析 + 综合 + 完整性 critique)→ **lead 闸门裁决 + 据 critique must-fix 修正**(见本节下)。对拍铁律 `-attn_scale=1.0`。

> ## ★ 闸门裁决(lead/用户,2026-06-15)+ critique 解决记录 —— 实现以此为准,与下文冲突处以此覆盖
> - **范围决策 = 做全(Option B)**:支持 `seqlen_kv` 任意方向(含 **jagged/group 的 seqlen_kv > seqlen_q**)。
> - **must-fix #2(范围矛盾)解决**:**新增 `max_seqlen_kv` 字段**进两个 bwd params 结构(`HstuAttentionNoGroupBwdParams` + `HstuAttentionGroupBwdParams`,现都只有 `max_seqlen_q`),并接进 jagged/group dispatch 的 grid + num_splits sizing。**∴ 下文 §3B「无新 params 字段」作废 —— 本里程碑确加 1 个字段 `max_seqlen_kv`(其余无新字段)。** batched 用 `seqlen_kv` 标量、两方向本就安全、不需新字段。
> - **must-fix #1(dispatch grid 漏进 change-surface)解决**:**kernel 启动 grid 在 dispatch 文件按 `max_seqlen_q` 开**(batched `hstu_attention_batched_backward_dispatch.hpp:66-70,:79` 的 `grid_seqlen_kv = is_jagged ? max_seqlen_q : seqlen_kv`;group `:87-99` grid+num_splits 用 `max_seqlen_q`)→ cross 须改用 `max_seqlen_kv`。**这是 dispatch 改面(§3A),不只是 harness 的 dq_acc workspace(§3C)** —— 两处都要改,且须区分:dispatch=kernel 启动 grid,harness=workspace 分配,两者都按 kv 块数。
> - **must-fix #3(测试 kN0)解决**:§6 的"非整除"和"determ multi-block"按**选定 tile 的 kN0**(`bwd_kN0_for`:hd256=64、否则 128,按 MaxK 分桶非 raw hdim,harness `:362` 注),不写死 128;**至少一个 determ 用例 seqlen_kv > seqlen_q 且跨多个 KV 块**(用选定 kN0 算),真正命中 R4。
> - **overclaim 解决**:§3D 把 pipeline(with/no_softmax)从"NOT touching 硬边界"降级为"**pending R5 确认**"(它本是未验开放风险,不能既是硬边界又是开放风险)。
> - **open questions 拍板**:#1 batched cross 纳入主套件(uniform seqlen,两方向);#2 group kv 长入口 = per-group(与 max_seqlen_q 对称,新增 `max_seqlen_kv` group 字段/入口);#3 `-seqlens_kv` 给出且≠seqlens 即自动 cross(默认空=self 向后兼容);#4 = 是,`max_seqlen_kv` 进 params(见 #2)。
> 命名警告:**本里程碑的 "cross" = cross-attention(Q/KV 不同长)。** 测试驱动里既有的 "P1-1 cross" / "cross matrix" 指的是**配置笛卡尔积**(`run_bwd_tests.py:193 "CROSS MATRIX"`, `:401 "(P1-1 cross)"`),与本里程碑同名异义。新增用例命名一律用 `xattn-` 前缀,避免混淆。

---

## (0) 范围 — 一句话 + 边界

**范围:** 让 HSTU **bwd** 正确处理 **seqlen_q != seqlen_kv**(cross-attention),办法 = **完整镜像 fwd 的 cross 范式** —— 把当前钉死的 `HstuBlockMasking<false /*cross*/,...>` 改成**运行时 BOOL_SWITCH(`param.is_cross_attention`)** 选择的 `kIsCrossAttention` 轴,并在 4 个 bwd kernel 的 mask 构造处用 `if constexpr(FmhaMask::kIsCrossAttention)` 改调 `make_hstu_cross_attention_block_mask_*`(把 `kargs.seqlen_kv` 作为 `seqlen_k` 传入)。**cross 是运行时 switch,不是 instance 轴**(fwd 已证)。

**In scope**
- dispatch:把 5 处硬编码 `HstuBlockMasking<false /*cross*/,...>` 改为 `HstuBlockMasking<kIsCrossAttention,...>`,外包一层 `BOOL_SWITCH(param.is_cross_attention,...)`(`hstu_attention_batched_backward_dispatch.hpp:387-388`;`hstu_attention_group_backward_dispatch.hpp:144,146,211,213`)。
- kernel:4 个 mask 构造 lambda 用 `if constexpr` 分叉到 cross builder,把 **`kargs.seqlen_kv` 作为 `seqlen_k`** 传入(`hstu_attention_bwd_kernel.hpp:404/414` SiLU、`:828/834` softmax、`:1221/1231` group SiLU、`:1591/1602` group softmax)。
- harness(`example_hstu_attention_bwd.cpp`):新增独立 kv 长度入口、解除 `is_cross_attention=false` 钉死(`:205,:749`)、解除 kv 别名(`:227,:244,:823`)、按 `phy_seqlen_kv` 分配 K/V/dK/dV、**修 determ grid 用 max_seqlen_kv**(`:365`)、调 cross reference(reference 已就绪,只需翻第一参 + 喂独立 kv offsets)。
- 测试矩阵扩 `seqlen_q != seqlen_kv` 双向 + causal 对齐 + P1-1(num_target/contextual/window)逐配置,经 **`/root/workspace/hstu-bwd-impl/test/run_bwd_tests.py`**(220/220 的真实驱动,**不是** `scripts/`)。

**Out of scope / 不碰(诚实边界)**
- `reference_hstu_attention_bwd.hpp`:**已是 cross-ready oracle,一行不改**(见 §1)。
- promoted self byte-identical 路:`is_cross_attention=false` 腿必须 byte-identical 于 M7c(见 §4)。
- **target_in_kv**:cross mask 硬编码 `max_k_uih_len = seqlen_k`,注释 `// assuming target_in_kv == false`(`hstu_block_masking.hpp:53,:566`)。本里程碑**只支持 targets 留在 Q 侧**。targets-in-KV 是结构性新路,不做。
- mask 数学 / `diff_q_kv_len` / `GetTileRangeAlongY`:cross mask 家族已含 bwd 钩子(`hstu_block_masking.hpp:273-278 GetTileRangeAlongY`, `:281-287 IsEdgeTile`),**不改 mask 内部**。
- batched(非 jagged)cross **是否做**:见 §6 决策 —— 主测面是 jagged/group;batched cross 仅 uniform seqlen,作可选附加。

---

## (1) 核心洞察 & 复用 fwd + 就绪的 reference oracle

**结论:bwd 在内存/grid/loop/PRE 层面已经结构性 cross-ready。唯一真正的破绽是 MASK 被钉死成 self。** cross kv offset 三元组**已经接好**(`is_cross_attention ? seq_kv_offsets_ptr : seq_q_offsets_ptr`,`batched_backward_dispatch.hpp:189-190,:323-324`;`group:172-173,:269-270`),所以今天若强行 `is_cross_attention=true`,会**加载正确的 K/V 但用 self mask 几何去 mask 它们** → dK/dV/dQ 数值错、不崩溃。这是本里程碑的头号 silent-wrong 陷阱(§2)。

**fwd 是精确蓝图,三层协同(`fwd-cross-blueprint`):**
1. **DISPATCH** `BOOL_SWITCH(param.is_cross_attention, kIsCrossAttention, ...)` 选 pipeline-problem 模板参,**同时**喂不同 kv 长/offset 源(`hstu_attention_batched_forward_dispatch.hpp:97`;jagged/group `:90`)。
2. **PIPELINE PROBLEM** 把 `kIsCrossAttention` 作为纯 constexpr bool 传播,**不**改 layout/compute(`hstu_attention_pipeline_problem.hpp:67,:90`)。
3. **KERNEL** 是唯一行为分叉点:K/V DRAM extent = `seqlen_kv`,mask 构造 constexpr 分叉 cross vs self(`hstu_attention_fwd_kernel.hpp:983-1004 with-local`,`:1046-1057 no-local`,选择器 `:980/:1043`)。mask 建好后 pipeline 调用 cross/self **完全相同**(`:1006-1039`)—— pipeline 只吃 `[seqlen_k_start, seqlen_k_end]`。

**bwd 已经做对的 cross 部分(`bwd-self-assumptions` / `per-gemm-kv-seq`):**
- kv 侧 GEMM 算子(GEMM0 K、GEMM2 V、GEMM1 dV、GEMM3 dK)的 DRAM view **已用 `kargs.seqlen_kv`** 不是 seqlen_q:`hstu_attention_bwd_kernel.hpp:320/330(k/v)、:432/442(dk/dv)`;softmax `:753/759/852/858`;group `:1144/1150/1196/1202`。
- jagged 每 batch base **独立** split q/kv:`query_start=q_offsets[i_batch]; key_start=kv_offsets[i_batch]`,`seqlen_q/seqlen_kv` 各自从自己的 delta 求(`:252-267`,softmax `:686-701`,group `:1074-1078`)。
- grid **kv-tile-major**:`grid.x = ceil(seqlen_kv/kN0)`,early-exit `i_n0 >= seqlen_kv`(`:210-212/:282`,`:647-649/:715`,`:1042-1046/:1081`)。q 是内层 pipeline loop(`GetTileRangeAlongY`)。
- batched 模式也独立携带 `seqlen_kv` 标量(dispatch 传 `param.seqlen_kv`,`batched_backward_dispatch.hpp:192,:326`),K/V/dK/dV view 用之。
- PRE `dot_do_o`(D=rowsum(O*dO))纯 q-seq,POST convert/reduce 对 q-sized dQ buffer elementwise —— **已 cross-safe**(`:1627-1681`,`:1689-1720`)。

**READY reference oracle —— 一行不改即可对拍(`reference-oracle`):**
`reference_hstu_attention_bwd.hpp` 两个 struct 都**已**接受 `is_cross_attention` 为 Run 首参,独立从 `seq_q_offsets/seq_kv_offsets` 求两个 seqlen,`BOOL_SWITCH_2` 选 `HstuBlockMasking<kIsCrossAttention,...>`,调 `make_hstu_cross_attention_block_mask_with/without_local`:
- no-group:`reference_hstu_attention_bwd.hpp:65 Run(bool is_cross_attention,...)`、`:159-162` 两 seqlen 独立、`:167 BOOL_SWITCH_2`、`:178-194 with_local(seqlen_q,seqlen_kv,...)`、`:219-220 without_local`。
- group:镜像于 `:580-581,:593,:604-620,:645-646`。
- dQ/dK/dV 维度对 seqlen_q!=seqlen_kv **已正确**:dk_acc/dv_acc 为 `seqlen_kv x hdim`(`:229-232`,group `:654-657`),内层 `sk in [0,seqlen_kv)`、外层 `sq in [0,seqlen_q)`,K/V/dK/dV 用 `seq_kv_offsets`、Q/dO/O/LSE/dQ 用 `seq_q_offsets`。
- **scale_p 用 `max_seqlen_q`**(`:165`),`max_seqlen_kv` 是死的签名占位符(`:79 注释 "only used to match the forward signature"`)。

**对拍铁律:kernel 的 cross mask ctor 参数必须与 reference 的 cross 调用点逐字对齐**(`diff_q_kv_len`/contextual 处理),否则 对拍 fail。

---

## (2) 逐 GEMM / kv-sequence 计划(seqlen_q != seqlen_kv)+ silent-wrong 陷阱

**几何不变式:** cross mask 跟踪 `diff_q_kv_len = max_k_uih_len - max_q_uih_len = (seqlen_kv - num_target) - (seqlen_q - num_target)`,在距离比较前对 row 加偏(`hstu_block_masking.hpp:73,:194 row += diff_q_kv_len`;no-local `:579,:621`)。这把 causal 对角线对齐到右下角。self mask 只有单个 `seqlen`(`:301,:322`),**对 seqlen_q!=seqlen_kv 数学上错**。

| GEMM | 角色 | 行轴 | 列轴 | cross 关键点 |
|---|---|---|---|---|
| GEMM0 | S = Q·Kᵀ | seqlen_q(Q tile) | seqlen_kv(KV tile) | mask 在 (row=q, col=kv) 上判 validity,需 `diff_q_kv_len` |
| GEMM1 | dV += Pᵀ·dO | seqlen_kv(输出行) | hdim_v | dV 写 `seq_kv_offsets`,行数 seqlen_kv |
| GEMM2 | dP = dO·Vᵀ | seqlen_q | seqlen_kv | 同 GEMM0 的 mask 几何 |
| GEMM3 | dK += dSᵀ·Q | seqlen_kv(输出行) | hdim_qk | dK 写 `seq_kv_offsets`,行数 seqlen_kv |
| GEMM4 | dQ += dS·K | seqlen_q(输出行) | hdim_qk | dQ 写 `seq_q_offsets`,行数 seqlen_q;determ split 沿 KV 块 |

**Silent-wrong 陷阱清单(全部 numerically-wrong, 不崩溃):**

1. **MASK 钉死 self(头号):** dispatch `HstuBlockMasking<false>` + kernel `make_hstu_self_*` 即使 `is_cross_attention=true` 也用 self 几何。self-attn 测试(等长)**永远抓不到**。必须 `seqlen_q != seqlen_kv` 显式测。证据 `batched_backward_dispatch.hpp:387-388`;`group:144,146,211,213`;kernel `bwd_kernel.hpp:404/414/828/834/1221/1231/1591/1602`。

2. **cross ctor 参数顺序/取值错:** self builder 收单 `seqlen`;cross builder 收 `(seqlen_q, seqlen_k)`,ctor 顺序 `(is_tile_in_first_split_, seqlen_q_, seqlen_k_, ...)`(`hstu_block_masking.hpp:38-44`)。把 `seqlen_q` 喂进 `seqlen_k` 槽 → `diff_q_kv_len=0` → self 几何静默复活,**编译通过**(全 int),static_assert 只检 `kIsCrossAttention` 不检参值(`:881/:915`)。**必须走 `make_hstu_cross_attention_block_mask_*` 并保持精确顺序。**

3. **with-local 包装器 REORDER 陷阱:** `make_hstu_cross_attention_block_mask_with_local` 签名是 `(is_first_split, seqlen_q, seqlen_k, contextual, num_target, max_attn_len, min_full)`,但 ctor 取 `num_target` **最后**,包装器内部重排(`hstu_block_masking.hpp:873-889`)。**绝不直接调 ctor、绝不改包装器参序** —— 否则 `max_attn_len <-> num_target` 静默互换,小 shape 仍可能过。

4. **determ grid 用 max_seqlen_q 而非 max_seqlen_kv:** bwd 是 KV-block-parallel。jagged/group 的 `grid_seqlen_kv_h = is_jagged ? max_seqlen_q : phy_seqlen_kv`(harness `example_hstu_attention_bwd.cpp:365`;group dispatch keys off `max_seqlen_q` `:90/:99`)。cross 下 `seqlen_kv > seqlen_q` 时 grid.x 短 → **尾部 KV 块的 dK/dV 静默归零、dQ 漏其贡献**;num_splits 同理须用 kv 块数 `ceil(max_seqlen_kv/kN0)`。修点见 §3C。

5. **self 用 self mask 时 seqlen_kv 也写错:** 即使 cross builder 正确,若把 `seqlen_q` 误传 K/V/dK/dV view 的行 extent(self 捷径),`seqlen_q < seqlen_kv` 时 dK/dV 截断、`>` 时 OOB。bwd view **已用 `seqlen_kv`**(§1),保持即可,**勿回退**。

6. **scale_p 的分母:** reference 权威 = `1/max_seqlen_q`(`reference:165`)。kernel cross 必须用同一 **Q-side** 分母,**勿**用 seqlen_kv(`fwd batched MakeKargs:360` 也是 seqlen_q)。

7. **PRE dO stride 别名:** softmax PRE 用 **O 的 stride** 读 dO(`batched_backward_dispatch.hpp:301-304` 注释明确标 cross hazard)。本里程碑测试**给 dO 与 O 同 layout**即可规避;若未来 cross harness 给 dO 独立 layout,须传 `param.{seq,nhead,batch}_stride_do`。记入 §8 风险,本期不实现独立 dO layout。

8. **contextual_seqlen 越界:** cross no-local mask 用 `max_col_id = seqlen_k - (contextual_seqlen-1)`(`hstu_block_masking.hpp:571`)。测试须保 `contextual_seqlen <= min(seqlen_q, seqlen_kv)`。

---

## (3) 完整 change-surface 清单

### (3A) dispatch —— `kIsCrossAttention` 轴 via BOOL_SWITCH
- **batched** `hstu_attention_batched_backward_dispatch.hpp:387-388`:`HstuBlockMasking<false /*cross*/, kUseCausal, kUseLocal>` → 包一层 `BOOL_SWITCH(param.is_cross_attention, kIsCrossAttention, [&]{ ... HstuBlockMasking<kIsCrossAttention, ...> ... })`。**switch 只包 mask typedef + 下游 Pipeline/Kernel**(镜像 fwd `:97`),勿提到 pad/local switch 之上(否则 false 腿类型构造顺序变 → 符号 churn,违反 §4 byte-identity)。RunSilu 与 RunSoftmax 共享此点。
- **group** `hstu_attention_group_backward_dispatch.hpp:144,146`(RunSilu Local/NoLocal)与 `:211,213`(RunSoftmax Local/NoLocal)4 处同改。group 双 pipeline(local+nolocal)各自加 cross 子腿。
- kv-offset 三元组**已接好**(`:189-190,:323-324`;group `:172-173,:269-270`),false 腿选 `seq_q_offsets_ptr` = pre-cross 同值,**无需改**。
- **★ dispatch grid + num_splits sizing(闸门 must-fix #1,本里程碑必改)**:kernel 启动 grid 在 dispatch 文件按 q 长开,cross(尤其 seqlen_kv>seqlen_q)须改按 **max_seqlen_kv**:
  - **batched** `hstu_attention_batched_backward_dispatch.hpp:66-70`:`grid_seqlen_kv = is_jagged ? param.max_seqlen_q : param.seqlen_kv` → jagged 分支改 `param.max_seqlen_kv`(batched 非 jagged 已用 seqlen_kv 标量、安全);`num_splits = ceil(grid_seqlen_kv/kN0)`(`:70`)与 `GridSize(...,grid_seqlen_kv)`(`:79`)随之。
  - **group** `hstu_attention_group_backward_dispatch.hpp:87-99`:`num_splits = ceil(max_seqlen_q/kN0)`(`:90`)+ `GridSize(...,max_seqlen_q)`(`:99`)→ cross 改用 `param.max_seqlen_kv`。
  - 这要求 params 新增 `max_seqlen_kv`(见 §3B 修正);self(`is_cross_attention=false`)路须取 `max_seqlen_kv==max_seqlen_q`(harness 别名)→ byte-identical 不变(§4)。
  - **区分**:此处 = kernel 启动 grid(dispatch);§3C `:365` = dq_acc workspace 分配(harness)。**两处都按 kv 块数,缺一即 R4 silent-wrong。**

### (3B) kernel —— mask builder constexpr 分叉
4 个 mask 构造 lambda,改成:
```cpp
if constexpr (FmhaMask::kIsCrossAttention) {
    mask = make_hstu_cross_attention_block_mask_with_local(
        is_tile_in_first_split, kargs.seqlen_q, kargs.seqlen_kv,   // seqlen_k = seqlen_kv
        contextual_seqlen, num_target, window_size, min_full_attn_seqlen);
} else {
    /* 现有 self builder 逐字不动 */
}
```
- `FmhaMask::kIsCrossAttention` 是每个 mask 上已有的 static constexpr(`hstu_block_masking.hpp:19,:295,:546,:715`)。
- 改点:SiLU `bwd_kernel.hpp:404/414`、softmax `:828/834`、group SiLU `:1221/1231`、group softmax `:1591/1602`。
- **`kargs.seqlen_kv` 已在 kargs 里且已正确派生**(§1)—— 仅需把它喂进 cross builder 的 `seqlen_k` 槽,mask 构造**无新 Kargs 字段**。
- **⚠ 修正(闸门 must-fix #2)**:grid sizing 路**确需新增 `max_seqlen_kv` params 字段**(见 §3A grid 子项 + §3C)。即:mask 路无新字段,但 jagged/group 的 grid/num_splits 路要加 `max_seqlen_kv`。
- **必须 `if constexpr`**:false 腿须 dead-code-eliminate 到与 M7c 逐指令相同(§4)。

### (3C) harness `example_hstu_attention_bwd.cpp` —— 独立 kv 序列
- **CLI:** 新增 `-seqlens_kv`(jagged: 逗号列;batched: 单值;空 = 别名 seqlens,即 self)。注意当前 CLI **无** kv 入口(`:118 seqlens` 只描述 query)。group 模式新增 `-g_max_seqlens_kv` 或复用 per-group kv 列(见 §6 决策)。
- **解钉:** `is_cross_attention = false; // M3`(`:205`)、`// M4: self-attention only`(`:749`)→ 由 `-seqlens_kv` 是否给出派生(给出且不等于 seqlens → true)。
- **解别名:** `max_seqlen_kv = max_seqlen_q`(`:227`)、`phy_seqlen_kv = phy_seqlen_q`(`:244,:823`)、host reference `seq_offsets_kv = seq_offsets_q`(`reference-oracle` 指出的 `:251,:824` 别名)→ 改为独立。
- **独立设备 buffer:** 当前 `fp/bp.seq_kv_offsets_ptr = seq_offsets_q_dev`(`:414,:492,:981,:1041`)—— cross 须分配**独立** `seq_offsets_kv_dev` 并喂之,否则 seqlen_kv 静默塌回 seqlen_q(reference-oracle silent trap)。
- **K/V/dK/dV 分配按 `phy_seqlen_kv`:** view 分配已用 `phy_seqlen_kv`(`:273-296,:840-861`)—— 只要 `phy_seqlen_kv` 真正 != `phy_seqlen_q` 即可;勿保留别名。
- **determ grid / num_splits 修(§2 陷阱4):** `grid_seqlen_kv_h = is_jagged ? max_seqlen_q : phy_seqlen_kv`(`:365`)→ jagged 也须用 **max_seqlen_kv**;`num_splits = ceil(grid_seqlen_kv / kN0)`(`:367`)随之。POST reduce loop 与 dq_acc workspace sizing 同步按 kv 块数。
- **reference 调用:** `Run(is_cross_attention, ...)`(`:607,:1140`)第一参翻为运行时 `is_cross_attention`,kv offsets/max_seqlen_kv 传独立值(`:621 max_seqlen_kv`)。**reference 签名已就绪,无改。**

### (3D) NOT touching(诚实边界)
- `reference_hstu_attention_bwd.hpp`:**零改**(§1)。
- mask 内部数学 / `diff_q_kv_len` / `GetTileRangeAlongY` / `IsEdgeTile`:**零改**(已含 bwd 钩子)。
- promoted self byte-identical 路:见 §4。
- pipeline 数学(`with_softmax_bwd_pipeline.hpp` / `no_softmax_bwd_pipeline.hpp`):loop bound 来自 cross-aware 的 mask `GetTileRangeAlongX/Y`,**预期零改**(但 §8 R5 须确认无其它 seqlen_q==seqlen_kv 隐含假设)。

### (3E) cross 是 **runtime switch**,不是 instance 轴
镜像 fwd:`generate_instances.py` **不加 cross 轴**(`zero-regression`)。理由见 §5。

---

## (4) 零回归策略(cross=false byte-identical to M7c)

**机制保证:** `HstuBlockMasking::Type = std::conditional_t<kIsCrossAttention, Cross..., Self...>`(`hstu_block_masking.hpp:861-868`)。`kIsCrossAttention=false` 腿**逐字**解析到今天 `<false>` 同一 Type。dispatch 的 kv-offset 三元组 false 腿选 `seq_q_offsets_ptr` = pre-cross 同 host 值 → MakeKargs 参数 byte-wise 不变。kernel 的 `if constexpr(FmhaMask::kIsCrossAttention)` false 腿是现有 self builder 逐字 → dead-code-elim 到现有指令流。

**硬性要求:**
- kernel mask 分叉**必须** `if constexpr`(不是运行时 if),否则 cross 分支可能扰动 false 腿 codegen。
- dispatch BOOL_SWITCH **只包** mask typedef + 下游,**保持 false 腿类型构造顺序与今天一致**;勿提到 pad/local switch 之上。

**Gate = `test/co_symbols.py` verify(`zero-regression`):**
1. 从 M7c HEAD dump baseline 符号(`co_symbols.py:74-95` verify 模式:只对 baseline 符号报 MISSING/DIFF,新符号 `(new ... allowed)`)。
2. cross 改动后 verify:**每个 self bwd 符号必须 byte-identical**;`mask<true>` 是全新 mangled 符号(模板参不同),不与 self 符号冲突。
3. 任一 self 符号 DIFF = `if constexpr` 守卫泄漏 = 阻断合并。
- self 回归套件 220/220 须仍绿(`run_bwd_tests.py`)。

---

## (5) instance / 编译期预算(cross 会 2x instances 吗?)

**不会。** cross 是 dispatch `Run()` 内的运行时 BOOL_SWITCH(fwd 已证此范式,`forward_dispatch.hpp:97`)—— 两腿在**同一** `.cpp` 内编译。`generate_instances.py` 无 cross 轴,现有 64 个 batched_backward `.cpp`(2 dtype × 2 causal × 2 softmax × 2 determ × 4 hdim)+ no_group 入口的 `BOOL_SWITCH_3` **文件名/内容不变**(`zero-regression`)。

**编译期/寄存器成本(诚实):** cross BOOL_SWITCH 在每个 dispatch body 内多实例化一份 `mask<true>` pipeline+kernel。group 已持 local+nolocal 双 pipeline,加 cross 子腿 → 每 group 入口 `{local,nolocal} × {cross,self}` = 4 份(编译期 + 寄存器压力)。这是**编译期**成本不是 instance 数爆炸。须在改 group dispatch 前确认编译时长/寄存器不超预算(§7 checkpoint 实测)。

---

## (6) 测试矩阵(真实驱动 `run_bwd_tests.py`)

**铁律:** `seqlen_q != seqlen_kv` **双向**(q<kv 与 q>kv)是抓 §2 陷阱1/4/5 的唯一手段;等长测试一律抓不到。`-attn_scale=1.0`。用例前缀 `xattn-` 避免与既有 "P1-1 cross" 混淆。

| 维度 | 取值 |
|---|---|
| 方向 | seqlen_q < seqlen_kv;seqlen_q > seqlen_kv(**两向都要**) |
| 模式 | no_group jagged(主);group(主);batched(可选,仅 uniform) |
| 激活 | SiLU;softmax |
| mask | causal=0;**causal=1(对齐验证,抓 diff_q_kv_len)** |
| P1-1 | num_target(Q 侧)、contextual(`<= min(q,kv)`)、local window>0、minfull —— 逐配置 |
| 非整除 | kv seqlen 非 kN0=128 整除(如 200);q 非 kM0=32 整除(如 130) |
| determ | deterministic=1 + multi-KV-block(`seqlen_kv=512`),抓 §2 陷阱4 grid/num_splits |
| dtype | bf16(主);fp16 一两例 |

**代表性用例(伪 CLI,待 `-seqlens_kv` 落地):**
- `xattn-jagged-qlt-kv-silu-causal1`:`-seqlens=128 -seqlens_kv=256 -softmax=0 -causal=1`
- `xattn-jagged-qgt-kv-silu-causal1`:`-seqlens=256 -seqlens_kv=128 -softmax=0 -causal=1`
- `xattn-jagged-qlt-kv-softmax-causal1`:`-seqlens=128 -seqlens_kv=256 -softmax=1 -causal=1`
- `xattn-jagged-causal0-target`:`-seqlens=128 -seqlens_kv=200 -causal=0 -targets=8`(target 留 Q 侧)
- `xattn-jagged-causal0-context`:`-seqlens=128 -seqlens_kv=200 -context_len=8`
- `xattn-jagged-local-qlt-kv`:`-seqlens=128 -seqlens_kv=256 -local_len=16 -causal=1`(local + q<kv)
- `xattn-determ-qlt-kv-multiblk`:`-seqlens=128 -seqlens_kv=512 -deterministic=1`(grid/split 陷阱)
- `xattn-group-qlt-kv-softmax`:`-g=2 -seqlens=128 -seqlens_kv=256 -softmax=1 -causal=1`
- `xattn-non-divisible`:`-seqlens=130 -seqlens_kv=200`
- (可选) `xattn-batched-uniform-qlt-kv`:`-jagged=0 -seqlens=128 -seqlens_kv=256`

**门控翻转:** 这些用例当前应是 REJECT(harness 钉死 + CLI 不识别 `-seqlens_kv`);harness+dispatch+kernel 落地后逐个翻 PASS(`run_bwd_tests.py` 既定流程,见 test/README)。

**待决(§8 open):** group kv 长入口形态(per-group vs 单列);batched cross 是否纳入主套件还是仅 smoke;`-seqlens_kv` 默认空 = self(向后兼容,确保既有 101 用例不受影响)。

---

## (7) 分阶段实施顺序(零回归证明后设硬 checkpoint)

**Stage A — 零回归重构(纯 false 腿等价):**
1. dispatch 5 处 `<false>` → `BOOL_SWITCH(param.is_cross_attention, kIsCrossAttention)` 包 `HstuBlockMasking<kIsCrossAttention,...>`(只包 mask typedef + 下游)。
2. kernel 4 处 mask lambda 加 `if constexpr(FmhaMask::kIsCrossAttention){cross}else{现有 self 逐字}`。
3. 此刻 `is_cross_attention` 仍全 false(harness 未解钉)。

**==== 硬 checkpoint(阻断式)====**
- build(`BUILD_DEV=OFF`,见 HANDOFF)。
- `test/co_symbols.py` dump M7c baseline + verify:**所有 self 符号 byte-identical**,否则停、查 `if constexpr` 泄漏。
- `run_bwd_tests.py` self 套件 **220/220 绿**。
- 量编译时长 / 寄存器(§5 group 4-腿预算)。
- **未过此 checkpoint 不得进 Stage B。**

**Stage B — harness 解钉 + 独立 kv:**
4. CLI `-seqlens_kv`(+ group kv 入口);解钉 `is_cross_attention`、解别名 max/phy_seqlen_kv、独立 `seq_offsets_kv_dev`。
5. 修 determ grid/num_splits 用 max_seqlen_kv(`:365,:367`)+ POST reduce/workspace。
6. reference 调用第一参翻运行时 + 喂独立 kv。

**Stage C — cross 对拍 + 矩阵翻转:**
7. 先跑 `xattn-jagged-qlt-kv-silu-causal1` 单点对拍(最简 cross),确认 mask ctor 参与 reference 逐字对齐(§2 陷阱2/3)。
8. 逐个翻 §6 用例 REJECT→PASS;补 causal=1 对齐 + determ multi-block。
9. 全套件复跑(self 220 + 新 xattn)。

---

## (8) Red-flag 风险摘要

- **R1 (silent-wrong, 头号):** mask 钉死 self + cross kv offset 已接好 → `is_cross_attention=true` 今天即加载正确 K/V 但用 self 几何。必须 `seqlen_q != seqlen_kv` 双向显式测才能抓(`batched_backward_dispatch.hpp:388`;kernel `:404/414/...`)。
- **R2 (silent-wrong):** cross ctor 参数顺序 —— `seqlen_q` 误入 `seqlen_k` 槽 → `diff_q_kv_len=0` → self 几何静默复活,编译通过(`hstu_block_masking.hpp:38-44`)。**只走 `make_hstu_cross_attention_block_mask_*`。**
- **R3 (silent-wrong):** with-local 包装器把 `num_target` 重排到 ctor 末位(`hstu_block_masking.hpp:873-889`);直接调 ctor 或改包装器参序 → `max_attn_len <-> num_target` 互换,小 shape 仍过。
- **R4 (silent-wrong):** determ grid `grid_seqlen_kv_h = is_jagged ? max_seqlen_q : ...`(`example_hstu_attention_bwd.cpp:365`)+ num_splits 用 max_seqlen_q;cross `seqlen_kv > seqlen_q` 时尾 KV 块 dK/dV 静默归零。**必修。**
- **R5 (未确认):** with-softmax / no-softmax pipeline 是否在 mask/addressing 之外烘了 `seqlen_q==seqlen_kv`?loop bound 来自 cross-aware mask,但 dS LDS round-trip(kM0×kN0)等需一次非对称 seqlen 通读确认(`with_softmax_bwd_pipeline.hpp` 未读全)。
- **R6 (scope, silent-wrong if violated):** target_in_kv == false 硬假设(`hstu_block_masking.hpp:53,:566`)。本里程碑 targets **只在 Q 侧**;若测试把 target 放 KV → reference 与 GPU 静默不一致。
- **R7 (silent-wrong, deferred):** PRE 用 O stride 读 dO(`batched_backward_dispatch.hpp:301-304`)。本期 dO 与 O 同 layout 规避;独立 dO layout 留后续。
- **R8 (silent-wrong):** reference 权威 scale_p = `1/max_seqlen_q`(`reference:165`),`max_seqlen_kv` 是死占位符(`:79`)。kernel cross 勿用 seqlen_kv 当分母。
- **R9 (regression):** 任何人把 cross 加进 `generate_instances.py`(64 → 128 .cpp)= 不必要且改 instance 文件 → 强制重编 + 符号布局变。**保持 cross 离 instance 轴。**
- **R10 (correctness gate):** kernel cross mask ctor 参须与 reference cross 调用点逐字对齐,否则 对拍 fail —— Stage C 单点先验(§7.7)。
- **R11 (build budget):** group dispatch 加 cross → `{local,nolocal}×{cross,self}` 4 腿/入口,编译时长 + 寄存器须在 checkpoint 实测(§5)。

---

## 待决 open questions(交 lead 拍板)
1. **batched(非 jagged)cross** 纳入主套件还是仅 smoke?reference no-group batched 支持 cross(空 offset + 不同 shape k/v,`reference:104-105,:162`),但仅 uniform seqlen。
2. **group kv 长入口**:per-group 列(`-g_max_seqlens_kv`)还是单列?params 当前只有 `max_seqlen_q`(无 `max_seqlen_kv` 字段)—— group determ grid 是否需新 params 字段。
3. **cross 启用方式**:`-seqlens_kv` 给出即自动 cross,还是显式 `-cross=1` 标志?(默认空=self 保向后兼容。)
4. **`max_seqlen_kv` 是否需进 bwd params 结构**(grid sizing + determ num_splits)?今天两个 bwd params 结构都只有 `max_seqlen_q`。
