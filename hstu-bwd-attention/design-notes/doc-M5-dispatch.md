# 派单:写 M5 讲义(softmax 路径)

**先读硬规格:`/tmp/hstu-bwd-design/doc-series-spec.md`(全文)。本卡给 M5 专属输入。**

- **里程碑**:M5 = softmax 反向路径(no_group = batched + jagged;group softmax 是 M5b)。需要 PRE kernel 算 D、读 fwd 存的 LSE。
- **commit(行号锚定)**:`476bc16a`。`cd /root/workspace/ck_hstu && git show --stat 476bc16a`。(注意:M5 baseline 涉及上游 merge `2eebacd0` 适配 kStoreLSE,若引用到 LSE 存储相关 fwd 代码可参 `2eebacd0`,但 M5 本体改动锚 `476bc16a`。)
- **旧 HTML(参考叙事,行号/数值必重核)**:`/root/workspace/hstu-b1052-report/hstu-bwd-M5-softmax-20260608.html`。
- **事实来源**:`/tmp/hstu-bwd-design/M5-done.md`(+ `M5-review-findings.md` 若在)+ `candidates.jsonl` 里 `"id":"M5-softmax"` 那条。
- **输出**:`/root/workspace/hstu-b1052-report/hstu-bwd-M5-softmax-20260625.html`
- **M5 讲解重点**:
  1. 新 `hstu_attention_with_softmax_bwd_pipeline.hpp`(SiLU pipeline 的 fork):载 LSE+D(HBM→LDS→Reg,用 FMHA 预留的 LDS offset,smem 不变);**STAGE2** edge-tile `set_tile_if(s_acc, -inf, !IsTokenPairInsideMask)` 然后 `P=exp2(scale·s_acc − log2e·LSE)`,scale=alpha·log2e(log2 域 == reference 的 `exp(αS−LSE)`);**STAGE5** `dS=P·(dP−D)`;无 scale_p/g。
  2. 新 `HstuAttentionBwdDQDKDVSoftmaxKernel`(LSE/D dram window,1-D packed seq,batched+jagged base offset)+ **PRE** `hstu_bwd_dot_do_o_kernel`(`D=rowsum(O⊙dO)`,`[batch,head,seq]` 布局)。
  3. dispatch `RunSoftmax`:PRE→memset dq_acc→MAIN→POST。
  4. harness:fwd `is_training`+`lse_ptr`(seq-连续 `[batch,head,seq]`),取 GPU LSE 转置成 reference `[batch,seq,head]`,alloc d_dev,接线 lse/d_ptr。fwd 仅切 is_training,SiLU 路 byte-identical。
- **易错提示(lead 在原派单标注的风险,可作 note-block)**:① LSE 布局 fwd 产出 vs reference 期望必须对齐(否则 silent-wrong);② softmax 用 `-inf` 掩(**SiLU 是置 0,别抄错**);③ `get_validated_lse` 防 LSE=-inf→NaN;④ dV 不乘 alpha。
- **已知盲区(诚实写)**:对拍无法独立验 LSE *数值*(两侧共用同一份 GPU LSE),靠代码审计 + fwd 里程碑 LSE 验证兜底。
- 写完按规格 §6 回报。
