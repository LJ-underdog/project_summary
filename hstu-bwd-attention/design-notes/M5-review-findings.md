# M5 softmax bwd — 独立验证 + 对抗式 review 结论 (pane-2 / reviewer)

> 验证者:pane-2。基线 `b0c08cba`(M5 改动 = working-tree 未提交 + 1 新文件)。
> 结论先行:**可 promote。逐条核过 GREEN,无 blocker。** 独立机器验证全绿(干净重建 exit 0、
> 套件 60/59/0/1 exit 0、5 档自跑对拍全 PASS),代码对抗式 review 未发现 silent-wrong。

---

## 任务 A:独立机器验证(权威闸门)

### A1. 干净重建 — ✅ exit 0
`run_bwd_tests.py` 跑前我发现 coder 的 binary 已存在(06:37),`ninja` 报 "no work to do"——
**这不是独立验证**。故 `touch` 全部 4 个改动源(cpp/dispatch/kernel/新 pipeline)强制重编,
从磁盘源真实重建:
```
cmake --build build --target tile_example_hstu_attention_bwd -j128
→ [8/9] ... [9/9] Linking CXX executable bin/tile_example_hstu_attention_bwd
→ BUILD_EXIT=0  (仅 warnings,0 error)
```
日志:`runs/build-M5-review.log`。

### A2. 独立复跑套件 — ✅ TOTAL 60 / PASS 59 / FAIL 0 / SKIP 1, exit 0
```
python3 /root/workspace/hstu-bwd-impl/test/run_bwd_tests.py
→ TOTAL 60   PASSED 59   FAILED 0   SKIPPED 1
→ RESULT: OK (all expectations met)   SUITE_EXIT=0
```
- 与自述 60/59/1 **完全一致**。M1–M4b(44 案 SiLU/group)全 PASS = SiLU 零回归证实。
- `reject-softmax` 已删,替换为 16 个 M5 softmax pass 案(batched/jagged × causal{0,1} × 因子)。
- `reject-fp16`(exit 253)/`reject-hdim128`(throw "hdim_qk=hdim_v=64 only")仍正确拒绝;
  `skip-deterministic` 仍 N/A。日志:`runs/test-20260608-065351.log`。

### A3. 自抽 5 档独立对拍(故意用 ≠ 套件的 b/nhead/seqlens,全 -attn_scale=1.0)— ✅ 全 PASS
| # | 配置 | dQ max_abs | dK max_abs | dV max_abs | 判定 |
|---|---|---|---|---|---|
| 1 | batched causal=0 **+num_target** b4 h8 seq192 tgt24 | 2.4e-4 | 2.4e-4 | 4.9e-4 | PASS |
| 2 | **jagged** causal=0 **+per-batch num_target** b3 h4 seq160,77,224 tgt16,8,32 | 1.2e-4 | 1.2e-4 | 4.9e-4 | PASS |
| 3 | batched **全因子 combo** causal=1 b3 h4 **seq201(非整除)** | 2.4e-4 | 2.4e-4 | 2.0e-3 | PASS |
| 4 | **jagged 全因子 combo** b4 h2 seq300,**5**,128,177(tiny+full 混) | 6.1e-5 | 1.2e-4 | 9.8e-4 | PASS |
| 5 | batched causal=1 b1 h3 **seq513(非整除多tile)** | 1.2e-4 | 2.4e-4 | 4.9e-4 | PASS |

误差均为 bf16 舍入级(mean_abs ~1e-8,max|ref| 0.07–2.0),阈值 rel≤2e-2/abs≤5e-2 远未触顶。

---

## 任务 B:对抗式代码 review(逐条核 silent-wrong)

**1. LSE 域 — GREEN.** fwd 存自然对数 `m+log(l)`(= log Σexp(αS))。pipeline
`p=exp2(scale·s_acc − log2e·lse)`,`scale=alpha·log2e_v`,`row_lse=log2e_v·get_validated_lse(lse)`
(pipeline:410/464/467)= exp(αS−LSE),逐字对齐 FMHA 蓝本 587/599。`get_validated_lse`(-inf→0)
存在(413-416);FMHA 是有条件应用(仅 IsMasking/bias),此处**无条件**应用 → 严格更安全,无害。

**2. 掩码方向 — GREEN.** STAGE2 对 `!IsTokenPairInsideMask` 置 **-inf**(pipeline:457
`set_tile_if(s_acc, -inf, is_masked_out)`),exp2→0,与 SiLU 的置 0 相反——正确。
边界门用**运行时** `mask.IsEdgeTile(...)`(449,非编译期 IsMasking),沿用 P1-1。
causal=0+num_target(IsMasking=false)走全 tile + 逐 tile IsEdgeTile 掩码——
**A3 #1(batched)+ #2(jagged)独立对拍佐证此路正确**。row/col 计算与 SiLU/reference 一致。

**3. STAGE5 — GREEN.** `ds = p·(dp_acc − d[i_idx])`(pipeline:512),`d` per-row(i_idx=tuple(idx0)),
广播正确,无 dropout 分支。= FMHA 663-665 去 undrop 后的形态。符号/广播对。

**4. scale 接线 — GREEN.** dQ:`x*alpha`(559);dK:`x*alpha`(576);**dV:`return ...dv_acc` 未乘**
(578)。softmax 路 MakeKargs **不传 scale_p**(dispatch:309 只传 alpha),kernel Kargs 无 scale_p 字段。

**5. LSE/D 双布局对齐(最大风险)— GREEN(代码级闭合,非仅靠对拍).**
- GPU 侧统一 `[head, global_token]`:nhead_stride_lsed=phy_seqlen_q,seq stride=1。
  - jagged:`phy_seqlen_q=ΣL, batches_for_alloc=1`(harness:213/235),buffer=[num_head, ΣL]。
  - **fwd 写**(fwd_kernel:724 `batch_offset_lse=query_start*seq_stride_lse=query_start`;
    934-936 `+i_nhead*nhead_stride_lse`)→ 写在 `i_nhead*ΣL + query_start + s`。
  - **bwd 读**(bwd_kernel:`batch_offset_lsed=query_start`,`+i_nhead*nhead_stride_lsed`)→ 同址。
  - **PRE 写 D**(bwd_kernel jagged `d_base=i_nhead*ΣL + (q_start+sq)`)→ 同址。
  - **reference 转置**(harness:`lse_host(b,s,h)=lse_flat[(b*H+h)*phy_seqlen_q+s]`)读 `h*ΣL+token`→ 同址。
  四方一致。**关键:GPU-bwd 与 reference 同吃一份 GPU-产 LSE**(harness 取回后转置),P 两侧相同。
- **额外审计点(对拍盲区)**:GPU-vs-ref 对拍**结构上无法**独立验证 LSE *数值* 正确性
  (两侧共用同一份 GPU LSE,即便 LSE 全错也会"双错相等"而 PASS)。此盲区靠**代码审计闭合**:
  fwd 写的 `m+log(l)` 正是产出 O 所用的归一化子本身,且写入布局与读取一致 → P_bwd==P_fwd==P_ref,
  既自洽又正确。残余信任锚 = fwd 里程碑对 LSE 数值的验证(本单未改 fwd 逻辑)。**判定可接受。**

**6. PRE kernel — GREEN.** `float acc` 累加(bwd_kernel)。O/dO 用 [b,s,h,hd] strides 定位,
jagged 用 `seq_q_offsets[i_batch]` 定 token 基址;D 写 [head,seq] 连续(==MAIN 读布局,见 5)。
`if(sq>=seqlen_q) return` 防 jagged 跨批越界,各 D 位精确写一次(免 memset 成立)。
*次要(非 blocker)*:PRE 对 O 与 dO **共用 o_seq/nhead/batch_stride**(dispatch:284-286 只传 O strides),
隐含 O、dO 同布局——本 harness 二者 shape 逐字相同(harness:247/249)恒成立;若未来 dO 异布局会失效,建议注释标注。

**7. smem — GREEN.** softmax 的 ds_lds_ptr 偏移 = QT+OGrad+OGradT+Q+LSE+D(pipeline:367-372),
**与 SiLU pipeline 逐字相同**(no_softmax:323-328 也预留 LSE+D 空洞)→ 同一 `Policy::GetSmemSize`
覆盖,softmax 只是填了 SiLU 留空的 LSE/D region。无 LDS 爆量/越界。

**8. harness 边界数组 — GREEN.** num_targets `supplement_array_by_last_element(...,num_batch)`(196),
device 缓冲 `max(size,1)`(292),kernel 仅 ptr!=null 时读 [i_batch]。无 M2 式越界。

**9. SiLU 零回归 — GREEN.** diff 仅碰 3 文件 + 1 新文件;`no_softmax_bwd_pipeline.hpp`、SiLU kernel
(`HstuAttentionBwdDQDKDVKernel`)、group kernel **一字未改**(softmax 为独立新增 struct/kernel/函数)。
fwd 逻辑零改(仅 harness 切 `is_training=use_softmax`)。套件 M1–M4b 44 案全 PASS。

---

## 总评

**可 promote。** 任务 A(干净重建/独立套件/5 档自跑对拍)全绿;任务 B 9 条逐条 GREEN,
含最高风险的 LSE 双布局——已用 fwd/bwd/PRE/reference 四方代码级溯源闭合,并指出并接受了
"对拍无法验 LSE 数值"的盲区(由 fwd 里程碑兜底 + 写读布局自洽)。无 blocker。

非阻塞建议(可留作技术债):
- PRE kernel 注释标注"假定 O、dO 同 strides";若后续 cross/异布局需显式传 dO strides。
- (范围外提醒)M5b group softmax / cross-attention softmax 仍未做,与本单结论无关。
