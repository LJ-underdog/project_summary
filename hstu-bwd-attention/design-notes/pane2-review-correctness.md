# 派给 pane-2(角色:reviewer · 正确性)— 审 HSTU bwd DESIGN.md 的算法/数学/mask/scale 正确性

调度模式:tmux pane-2。独立、对抗式 review。不要派 sub-teammate,不重写正文。

## 对象
`/tmp/hstu-bwd-design/DESIGN.md`(HSTU bwd GPU 实现方案)

## 基准(以源码为准)
- oracle:`/root/workspace/ck_hstu/.../reference_hstu_attention_bwd.hpp`(顶部公式 + no_group/group 两套)
- mask:`hstu_block_masking.hpp`;params:`hstu_attention_params.hpp`;fwd:`hstu_attention_fwd_kernel.hpp`
- FMHA bwd:`/root/ck/.../block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp`、`fmha_bwd_kernel.hpp`

## 重点审这些(逐条给 ✅/⚠️/❌ + 源码证据 + 修法)
1. **七阶段双路与 reference 是否数学等价**:STAGE1 `S=alpha·QKᵀ`;STAGE2 SiLU `p=silu(S)·scale_p` / softmax `p=exp(S−LSE)`;STAGE3/4 dV/dP;STAGE5 SiLU `dS=dP·scale_p·dsilu(S)` / softmax `dS=p·(dP−D)`;STAGE6/7 dK/dQ `×alpha`。逐条比对 reference 公式,尤其**符号、scale 落点、是否漏乘**。
2. **scale 接线高危点**:确认 softmax 路 `exp(S−LSE)` 用 fwd 的自然对数 LSE,**不在 exp2 处重复乘 scale**(DESIGN §其中一节有"接线表"——核它对不对)。alpha 两处+dV 不吃,是否与 reference `dV=Σ P·dO`(无 alpha)、`dQ/dK=alpha·…` 一致。
3. **masked-out 置零**:SiLU `dsilu(0)=0.5≠0` 必须显式清 p,g;确认 DESIGN 的置零位置/谓词能真正让 dV/dP/dS 在被遮位置为 0(对照 reference masked-out 行为)。softmax 靠 S=−inf。
4. **mask Y 方向新增成员**:`GetTileRangeAlongY`/`IsEdgeTile` 的正确性铁律(返回区间须为真值集**超集**,否则漏算梯度);DESIGN 的"保守超集 + 离线校验"是否够稳。5 因子(causal/window/contextual/min_full/num_target)bwd 与 fwd 对称性是否成立。
5. **D 的来源**:softmax 路 `D=rowsum(O⊙dO)` 由 PRE(dot_do_o)算;确认 SiLU 路确实不需要 D/O/LSE,PRE 跳过逻辑正确。
6. **group/jagged 索引**:per-group alpha(全局标量?)/scale_p/mask 超参取数、cu_seqlens 索引是否与 reference group 版一致。
7. **数值陷阱**:LSE=−inf 整行、bf16 累加精度、dsilu 在大/小 S 的稳定性 —— DESIGN 有没有遗漏。

## 收尾
- 报告写 `/tmp/hstu-bwd-design/review-correctness.md`:P0(数学/正确性错误)/P1(不准确或风险)/P2(可选);每条 文件§位置 + 源码证据 + 修法。**不改正文**,交 lead 汇总。
- progress 简洁。
