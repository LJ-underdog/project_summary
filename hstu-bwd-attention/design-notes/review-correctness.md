# Review — DESIGN.md 正确性(算法/数学/mask/scale)· reviewer=pane-2

对抗式逐条核验,基准=源码:`reference_hstu_attention_bwd.hpp`(oracle)、`block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp`(FMHA MAIN)、`hstu_block_masking.hpp`、`hstu_attention_with_softmax_fwd_pipeline.hpp`、`hstu_attention_fwd_kernel.hpp`。结论:**无 P0;1 个 P1(softmax 路 NaN 守卫缺失);其余 P2/确认项。** 核心七阶段双路数学、scale 接线、masked-out 置零口径与 oracle 等价,可放行进入实现,但 P1 必须在 MAIN softmax 实现时补回。

---

## 总评:逐条 ✅/⚠️/❌

**① 七阶段双路 ↔ reference 数学等价 — ✅(全部核对通过)**
- STAGE1 `s_acc=gemm_0(q,k)` 未缩放 = FMHA:518;DESIGN §2.2 准确。
- STAGE2 `s_acc*=alpha` 物化 S,SiLU `p=silu(S)·scale_p`、`g=scale_p·dsilu(S)`:与 reference `P=silu(alpha·QK)·scale_p`(:286)、`dS=dP·scale_p·dsilu(S)`(:381-383)逐项一致。
- STAGE3 dV=Pᵀ@dO **不乘 alpha**(scale_p 已在 P 内)= reference 写回无 alpha(:483)。✅
- STAGE4 dP=dO@Vᵀ 沿 hdim_v = reference(:342)。✅
- STAGE5 softmax `ds=p·(dp−d)` = reference `dS=P·(dP−D)`(:413)= FMHA(:665);SiLU `ds=dp·g` = reference(:381)。✅
- STAGE6 dK、STAGE7 dQ 末乘 alpha = reference dQ `acc*alpha`(:439)、dK `*alpha`(:474)= FMHA dq/dk `*raw_scale`(:747/:772)。✅
- **符号/scale 落点/漏乘**:逐一比对无误。alpha 恰两处(STAGE2 头 + dQ/dK 收尾),dV 不吃,与 oracle 完全吻合。

**② scale 接线高危点 — ✅**
- softmax `exp(S−LSE)`:DESIGN §2.5/§4.7 明确「alpha 已在 STAGE2 物化 → exp2 处只用 log2e、**不重复乘 scale**」。骨架 §2.3:101 行 `s_acc*=alpha` 后,110 行 `p=exp2(log2e·s_acc − log2e·lse)=exp(S−LSE)`,**无双 scale**。已核 FMHA 的等价写法(scale=alpha·log2e 折进 exp2,:597-602)——两种因子化等价。
- LSE 自然对数域:已验证 fwd 存 `m+log(l)`(`with_softmax_fwd_pipeline.hpp:607-608`),与 bwd `exp(S−LSE)` 匹配。DESIGN §2.5「自然对数域」准确。
- alpha=fwd `scale_s`、scale_p=`attn_scale?attn_scale:1/max_seqlen_q`(reference:165)、group alpha 单标量(reference:515)、group scale_p per-group(reference:587):§4.7 表全部正确。

**③ masked-out 置零 — ✅(且识别到 GPU 特有陷阱)**
- DESIGN §3.2:line 153 正确指出「GPU gemm 会算出 masked-out 位真实 Q·K(非 0),必须清输出 `p`、`g`,而非套用 reference 的『S 置 0』CPU 写法」。这是关键且常被忽略的点——reference 靠 `silu(0)=0` 得 P=0(:276),GPU 上 `silu(alpha·QK)≠0`,**不清 p 则 dV 被污染**。DESIGN 抓住了。
- STAGE2 清 `p`+`g`(edge tile,谓词 `!IsTokenPairInsideMask`):等价 reference「STAGE5 dS else 0」(:380-385)且覆盖 dV(p=0)与 dK/dQ(g=0→ds=0)。`dsilu(0)=0.5≠0` 的禁 -inf 也正确(§3.2 line 154)。
- 超集 + 逐像素的兜底:全 masked tile 走 `IsEdgeTile=true→逐像素全清` 得 0 贡献,正确(仅浪费算力)。✅

**④ 反方向 mask 成员(GetTileRangeAlongY/IsEdgeTile)— ✅ 铁律正确,实现风险已隔离**
- 「返回区间须为真值集**超集**」铁律正确:逐像素 `IsTokenPairInsideMask` 精确清零,只要 Y-range 不漏 tile 即正确。DESIGN §3.1/§8-R3 + §5.4 离线校验,方向稳。
- `IsEdgeTile:=!IsFullTileInsideMask` 保守(is_tile_in_first_split=true 时恒 edge→恒逐像素)→**正确优先**,perf 后置(§3.1 line 148)。已验证 `IsFullTileInsideMask` 仅 `!is_tile_in_first_split` 返 true(mask:246/258/482/493),不会把含 masked 位的 tile 误判为 full,安全。
- 5 因子 bwd↔fwd 对称:同 mask 对象 + 同构造参数,正确。`IsTokenPairInsideMask(row=sq,col=sk)` 参数序与 reference(:246)一致;jagged/group 段内 0-based 索引与 reference sq/sk 对齐。✅

**⑤ D 来源 / SiLU 跳过 — ✅**
- softmax `D=rowsum(O⊙dO)` 由 PRE 算(= reference:408),SiLU 路确实不读 D/O/LSE(reference SiLU 分支无此三者)。DESIGN §1.1/§1.4/§2.5「PRE 仅 softmax 发射、SiLU 连 D buffer 都不分配」正确。

**⑥ group/jagged 索引 — ✅**
- per-group alpha=全局单标量(reference group Run 形参 alpha 为标量,:515);scale_p/window/contextual/min_full/max_seqlen_q per-group device 指针,`i_group=i_batch/num_batch_per_group`(reference:579);num_target per-batch(`num_targets_ptr[i_batch]`)。§3.4/§4.7/§8-D6 与 oracle 一致。cu_seqlens 索引与 FMHA group 分支(kernel:1110-1159)同构。✅

**⑦ 数值陷阱 — ⚠️(见 P1-1;其余已覆盖/确认安全)**
- bf16 累加:dq_acc/dk_acc/dv_acc 用 fp32 acc(§4.4 float dq_acc),容差 §5.2 合理。✅
- dsilu/silu 大/小 S:fp32(CompDataType,§4.2)下 `exp(-x)` 对极负 S 溢出→`x/inf=0`、`sig=0`,**饱和为 0 无 NaN**;极正 S→sig≈1。确认安全,无需额外守卫。
- **LSE=−inf 整行:未覆盖 → P1-1。**

---

## P1(不准确 / 正确性风险,实现前须落实)

### P1-1 · softmax 路缺 `LSE=−inf` 的 NaN 守卫(§2.3 骨架 line 110、§2.5、§5.2)
**问题**:softmax 路若某 query 行全部 key 被 mask(window/contextual 激进时可发生),fwd 存的 LSE 该行 = −inf(`m+log(l)`,m=−inf/l=0)。bwd 骨架 §2.3:110 `p = exp2(log2e·s_acc − log2e·lse)`,masked 位 `s_acc=−inf`、`lse=−inf` → `−inf − (−inf) = NaN` → `exp2(NaN)=NaN` → p/ds/dQ/dK/dV 全 NaN。
**源码证据**:
- oracle 显式守卫:`reference_hstu_attention_bwd.hpp:301-305`(no_group)、`:711-715`(group)——`if(lse_sq==−inf){ 整行 P=0 }`。
- FMHA 守卫:`block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp:571-583` `get_validated_lse`(masking 时把 −inf→0),配合 `exp2(scale·s − log2e·validated_lse)`(:597-602),masked `s=−inf`→`exp2(−inf)=0`,避免 NaN。
**修法**:HSTU MAIN softmax 路**保留 FMHA 的 `get_validated_lse`**(`raw_lse==−inf ? 0 : raw_lse`),即骨架改为 `p=exp2(log2e·s_acc − log2e·get_validated_lse(lse))`。DESIGN 文中应在 §2.3/§2.5/§7(新写清单)显式记一笔「softmax 路 LSE −inf 守卫」。**仅影响 softmax 路;SiLU 路无此问题(已确认饱和为 0)。**

---

## P2(可选 / 完备性,不阻塞)

### P2-1 · `IsOutOfBound` 谓词未进「新写清单」(§3.2、§7-⑦、§8-D1)
§3.2/§8-D1 用到 tile 级谓词 `IsOutOfBound`(= `!IsTokenPairInsideMask(row,col)`)做 `set_tile_if` 清零,但 §7 新写清单 ⑦ 只列 `GetTileRangeAlongY`/`IsEdgeTile`。FMHA mask 有 `IsOutOfBound`(被 :566 调用),HSTU mask 无。虽是一行 wrapper,建议在 §7 显式补「mask 新增 `IsOutOfBound`(或直接以 `!IsTokenPairInsideMask` 实现 set_tile_if 谓词)」,避免实现时遗漏 mask 改动面。

### P2-2 · 骨架未示 SiLU 路 `d`/`lse` load 的 `if constexpr` 守卫(§2.3)
骨架 line 98 守卫了 `lse` load,但 STAGE4/5 的 `d` load(FMHA 无条件 :628/:650)未在骨架体现需 `if constexpr(kUseSoftmax)` 包裹。SiLU 路若误发 d/lse DRAM 读会越界(无 buffer)。§2.3 line 95 文字提到「SiLU 路传 null tile window」,但 FMHA 是无条件 `load_tile(d_dram_window)`——需确认改为编译期跳过或 null-window 安全。建议在 §2.3 补一句「SiLU 路 d/lse load 均 `if constexpr(kUseSoftmax)` 跳过」。**非正确性错误**(意图正确),仅骨架完备性。

### P2-3 · group params 须显式含 `num_group`/`num_batch_per_group`(§4.6)
§3.4 用 `i_group=i_batch/num_batch_per_group`(reference:579),但 §4.6 canonical 表未把 `num_group`(或 `num_batch_per_group`)列入 group params 字段。建议补入 `HstuAttentionGroupBwdParams`(reference 形参 `num_batch_per_group`,:514)。完备性。

### P2-4 · softmax masked-out 与 SiLU 置零路径不对称的说明(§3.2)
§3.2 已说明 softmax 走「s=−inf 自然零」、SiLU「清 p,g」。补充确认:softmax 路 masked 的 `s=−inf` 在骨架 line 109 是在 `s_acc*=alpha`(line 101)**之后** set,`alpha·(任意) → 覆盖为 −inf` 安全无误(已核)。可在文中点明置零与 alpha 缩放的先后无关,免实现者误排顺序。仅文档清晰度。

---

## 确认无误的高危点(供 lead 放心)
- **无双 scale bug**:alpha 仅 STAGE2 头 + dQ/dK 收尾;scale_p 仅折进 p/g;softmax exp 用 log2e 不重复乘——三处独立核对通过。
- **dV 不乘 alpha、dQ/dK 各乘一次 alpha**:与 oracle 逐行一致。
- **masked-out**:SiLU 清输出 p,g(非清 S);softmax −inf;`dsilu(0)=0.5` 禁 -inf——口径与 oracle 等价。
- **D2 已查清准确**:fwd kernel 确按 Q-tile 重算 `is_tile_in_first_split`(:691-716,已验证),元素级 `IsTokenPairInsideMask` 自洽;bwd 保守超集不受影响。
- **LSE 自然对数域**:fwd 存 `m+log(l)`(:607-608,已验证),bwd `exp(S−LSE)` 匹配。

— 不改正文,交 lead 汇总。
