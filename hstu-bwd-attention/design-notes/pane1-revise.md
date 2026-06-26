# 派给 pane-1(integrator)— 按两份 review 修订 DESIGN.md

调度模式:tmux pane-1。按下列 review 结论修订你写的 `/tmp/hstu-bwd-design/DESIGN.md`。两份 review 全文在:
- `/tmp/hstu-bwd-design/review-correctness.md`(pane-2)
- `/tmp/hstu-bwd-design/review-feasibility.md`(pane-3)
先把两份读一遍,再逐条改。两份均**无 P0**;落实下面 P1 + 值得的 P2,并把 U1–U4 收成"建议默认值(待用户确认)"。

## 必改 P1

1. **[正确性 P1-1] softmax 路 LSE=−inf NaN 守卫**:§2.3 骨架 + §2.5 + §7 新写清单,显式加"softmax 路用 FMHA `get_validated_lse`(raw_lse==−inf?0:raw_lse),骨架改 `p=exp2(log2e·s_acc − log2e·get_validated_lse(lse))`"。注明仅 softmax 路需要,SiLU 路已饱和为 0 无需。

2. **[可行性 P1-A] 保留 BiasEnum/BiasDataType**:§4.2 + §4.6 改:`HstuAttentionBwdPipelineProblem` 砍 RandValOutput/FmhaDropout/BiasGrad/kHasBiasGrad(已验证 policy 不引用,安全),**但必须保留** `static constexpr auto BiasEnum = NO_BIAS;` + `using BiasDataType = InOutDataType;`(dummy)——否则 default policy 的 `GetSmemSizeBias`(:1627-1647)实例化失败、policy 复用破产。kargs 不传 bias。在 §4 写清这个"复用 default policy 的硬前提"。

3. **[可行性 P1-E] atomic 路也发 POST(修内部矛盾)**:§1.1 / §1.4 现写"非 deterministic 默认不发 POST"是**错的**——dq_ptr 是 bf16/fp16,atomicAdd 只能落 float dq_acc,**必须** POST 做 cast→dq_ptr。统一为:**"MAIN→POST 恒发射;atomic 路 POST=convert-only,deterministic 路 POST=reduce+convert;PRE 仅 softmax 路发射"**。同步改"默认零额外 workspace"的错觉:atomic 路也需 float dq_acc(nsplits=1)。kernel 计数恒 2(SiLU)/3(softmax)。

4. **[可行性 P1-B] 留 g 的 VGPR 压力**:§2/§8 标注"留 g 峰值约 +32 VGPR(64×128/256线程);M1 必须验 `ScratchSize=0` 且 VGPR 不致掉 wave;溢出则按 R2 把 g 暂存 LDS(复用 bias LDS 区段)"。明确这是**中资源风险/低实现风险**。

5. **[可行性 P1-C/P1-D] GetTileRangeAlongY**:§3.1/§5.4/§8 强化——(a)它被 MAIN **无条件**调用(:160-161,在 IsMasking 守卫外),故**连 M1 no-mask 也要**提供(平凡返回 (0,seqlen_q));里程碑估时纳入。(b)5 因子叠加下 attend 行集可能非连续,首版返回**连续保守超集**(宁多算不漏),离线校验断言 `[y_start,y_end)⊇真值集` 为 M2 硬性前置。

## 值得改的 P2
- [正确性 P2-1] §7 新写清单补"mask 的越界谓词(直接用 `!IsTokenPairInsideMask(row,col)` 内联做 set_tile_if,HSTU 自写 MAIN 无需新增第三个 mask 成员)"。
- [正确性 P2-2] §2.3 骨架补一句:SiLU 路 `d`/`lse` 的 load 均 `if constexpr(kUseSoftmax)` 跳过(不能只传 null window,FMHA 是无条件 load,会读空)。
- [正确性 P2-3] §4.6 group params 补 `num_batch_per_group`(reference :514)。
- [可行性 P2-1] §1.3 PRE/POST GridSize 除数 `kBlockSize`→`kM0`(源码 :1733/:2011;注 dot_do_o 块长=kBlockSize 断言)。
- [可行性 P2-2] §4.6 补 `batch_stride_lsed`(batched softmax 需要)。
- [可行性 P2-3] 记一条 perf 项:SiLU 路若复用 `Policy::GetSmemSize` 会白算 LSE/D 段,M1 后自算 smem 省 LDS。

## U1–U4 收口(改 §8.3 为"建议默认 + 待用户确认")
- **U1 bias/dbias**:MVP 不支持 dbias,但 Problem 保留 NO_BIAS+BiasDataType dummy(P1-A 强制,免费)。默认不做。
- **U2 deterministic**:默认 atomic(但两路都过 POST,见 P1-E);deterministic 作显式可选,需要逐位可复现再开。
- **U3 MVP 覆盖**:SiLU 优先(默认路+M1 闸门),softmax 次之;bf16 优先 fp16 跟进;hdim {64,128};模式 batched→jagged→group。
- **U4 GQA/MQA**:首版 nhead_ratio=1(MHA),留常量占位。

## 铁则
- 只按 review 改,不引入新设计;保持设计粒度(不灌满 kernel)。
- 改完写 `/tmp/hstu-bwd-design/revise-done.md`:逐条落实情况(P1×5 + P2×6 + U1–4)、DESIGN 新字数、还剩什么真需用户拍板。
- 这是要交给**用户最终 review** 的版本,§0 摘要里点明"已过双 review,P0=0,关键风险闸门=M1"。progress 简洁。
