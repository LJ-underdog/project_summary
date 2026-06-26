# revise DONE — DESIGN.md 按双 review 修订完成

两份 review(正确性 pane-2 / 工程可行性 pane-3)均无 P0。已落实全部 P1×5 + 值得的 P2×6 + U1–4 收口。DESIGN.md:21.5K→**27.1K 字符 / 344 行**。新增 §8.5「双 review 落实记录」表 + §0 标注「已过双 review,P0=0,闸门=M1」。

## P1(5,全落实)
- **P1-1** softmax LSE=−inf NaN 守卫:§2.2/§2.3 骨架(`get_validated_lse(lse)`)/§2.5/§7-① 均加;注明仅 softmax 路、SiLU 已饱和无需。
- **P1-A** 保留 BiasEnum=NO_BIAS+BiasDataType dummy(default policy `GetSmemSizeBias` 硬前提):§4.2 写清"复用 policy 硬前提"+证据行号(:1627-1647/:80-83);§4.6 kargs 不传 bias 但 Problem 留 typedef;§7-④。
- **P1-E** atomic 路也恒发 POST(修 §1.1↔§4.4 内部矛盾):§1.1/§1.3/§1.4/§4.4 统一为"MAIN→POST 恒发射;atomic=convert-only、determ=reduce+convert;PRE 仅 softmax;kernel 计数恒 2/3;atomic 也需 float dq_acc"。
- **P1-B** 留 g 的 VGPR≈+32(中资源/低实现风险):§2.4 量化+gfx942 512VGPR 背景;§6-M1 验收加"ScratchSize=0 且不掉 wave";§8.2-R2。
- **P1-C/D** GetTileRangeAlongY:§3.1 强化——(D)被 MAIN 无条件调用(:160-161,守卫外),M1 no-mask 也需平凡 (0,seqlen_q),里程碑纳入;(C)5 因子叠加 attend 行集非连续→首版连续保守超集,离线校验为 M2 硬性前置;§5.4/§6-M2/§8.2-R3 同步。

## P2(6,全落实)
- [正P2-1] §3.1/§7-⑦:越界谓词内联 `!IsTokenPairInsideMask`,无需第三个 mask 成员。
- [正P2-2] §2.3 骨架:SiLU 路 lse/d load 用 `if constexpr(kUseSoftmax)` 跳过(非 null window)。
- [正P2-3] §4.6:group params 补 `num_group/num_batch_per_group`(reference:514)。
- [可P2-1] §1.3:PRE/POST GridSize 除数 kBlockSize→kM0(:1733/:2011)+ dot_do_o kM0==kBlockSize 断言注。
- [可P2-2] §4.6:补 `batch_stride_lsed`(batched softmax)。
- [可P2-3] §8.2-R6:SiLU 路自算 smem 省 LSE/D 段(M1 后 perf 项)。

## U1–U4 收口(§8.3 改为"建议默认 + 待用户确认")
U1 不支持 dbias(留 NO_BIAS dummy)· U2 默认 atomic(两路都过 POST)· U3 SiLU/bf16/hdim{64,128}/batched→jagged→group 优先 · U4 nhead_ratio=1 占位。

## 还剩什么真需用户拍板
**无硬阻塞**。U1–U4 已给双 review 认可的默认值,按默认即可推进;仅用户改变需求(要 dbias / deterministic 默认 / 扩 dtype·hdim·模式 / 要 GQA)才需调整。技术风险 R1(policy 复用)/R2(VGPR)/R3(Y-range 边界)在 M1/M2 实测闭环,非设计期未决。交用户最终 review。
