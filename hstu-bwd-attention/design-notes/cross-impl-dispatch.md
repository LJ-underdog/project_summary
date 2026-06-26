# cross-attention 实现单 —— draft 已批准(coder pane 0.1,已 /clear)

新里程碑 **cross-attention(seqlen_q != seqlen_kv)**。设计稿 **`/root/workspace/hstu-bwd-impl/docs/draft-cross-attn.md` 已过 lead 闸门**——**先完整读它**(尤其顶部"★ 闸门裁决 + critique 解决记录",实现以此为准、与下文冲突处以此覆盖)。基线 HEAD=`17515fcc`(M7c)。全程 `-attn_scale=1.0`,**不动 reference(已 cross-ready)/promoted self 逻辑**。

## 闸门关键裁决(必须遵守)
- **范围 = 做全**:支持 seqlen_kv 任意方向(含 jagged/group 的 kv>q)。
- **新增 1 个字段 `max_seqlen_kv`** 进两个 bwd params 结构 + 接进 jagged/group dispatch 的 grid + num_splits(must-fix #1/#2)。batched 用 seqlen_kv 标量、两方向本就安全、不需新字段。
- 测试 kN0 按选定 tile(hd256=64 否则 128);≥1 个 determ 用例 kv>q 且跨多 KV 块(must-fix #3)。

## ⛔ 分阶段 + Stage A 后硬检查点(必停报 lead)

**Stage A — 零回归重构(纯 false 腿等价,is_cross_attention 仍全 false)**:
- dispatch 5 处 `HstuBlockMasking<false /*cross*/,...>` → `BOOL_SWITCH(param.is_cross_attention, kIsCrossAttention)` 包 `HstuBlockMasking<kIsCrossAttention,...>`(只包 mask typedef + 下游,勿提到 pad/local switch 之上)。
- kernel 4 处 mask lambda 加 `if constexpr(FmhaMask::kIsCrossAttention){ make_hstu_cross_attention_block_mask_with/without_local(...,kargs.seqlen_q, kargs.seqlen_kv,...) } else { 现有 self builder 逐字 }`。**必须 if constexpr**。
- **先不加 max_seqlen_kv 字段、不解钉 harness、不跑任何 cross case。**

**★ HARD CHECKPOINT(完成 Stage A 停,报 lead 亲验):**
1. `test/co_symbols.py` dump M7c(HEAD 17515fcc)baseline + verify:**所有 self 符号 byte-identical**(`mask<true>` 是全新 mangled 符号、允许);任一 self 符号 DIFF = if constexpr 泄漏 = 停。
2. self 套件 **220/220 exit 0**(`python3 /root/workspace/hstu-bwd-impl/test/run_bwd_tests.py`)。
3. 量 group dispatch 加 cross 子腿后的**编译时长 + 寄存器**(§5:group 现 local+nolocal,加 cross = {local,nolocal}×{cross,self} 4 腿/入口)。
报 lead:"Stage A 零回归证毕,co_symbols + 220/220 + 编译预算见 X",**停**。

## Stage B+(检查点放行后)
- Stage B:harness 解钉(CLI `-seqlens_kv` 给出且≠seqlens 即 cross;解别名 max/phy_seqlen_kv + 独立 seq_offsets_kv_dev)+ **加 max_seqlen_kv 字段进 params + 接 jagged/group dispatch grid/num_splits(`batched:66-70/79` jagged 分支、`group:90/99`)+ determ grid/workspace(harness `:365/:367`)** + reference 调用第一参翻运行时 + 喂独立 kv offsets。self 路 max_seqlen_kv==max_seqlen_q 保 byte-identical。
- Stage C:cross 对拍——先 `xattn-jagged-qlt-kv-silu-causal1` 单点验 mask ctor 与 reference 逐字对齐(R2/R3 陷阱)→ 逐个翻 §6 用例 REJECT→PASS(双向 kv、causal 对齐、P1-1 逐配置、determ kv>q multi-block)→ 全套件复跑(self 220 + 新 xattn)。

## 纪律(draft §2/§8 红旗)
- R1-R11 silent-wrong 全是数值错不崩溃:**seqlen_q != seqlen_kv 双向显式测**才抓得到;cross mask **只走 make_hstu_cross_attention_block_mask_***(别直调 ctor、别改包装器参序);scale_p 分母用 max_seqlen_q;target 只在 Q 侧(target_in_kv=false out-of-scope)。
- 容差禁松;带 FAIL/裸 PASS 不充数;**不 commit**(lead 闭合后统一)。
- Stage A 完停报;Stage B/C 各完也停报 lead 亲核。
