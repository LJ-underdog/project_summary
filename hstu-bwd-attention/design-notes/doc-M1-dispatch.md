# 派单:写 M1 讲义(SiLU MAIN 闸门)

**先读硬规格:`/tmp/hstu-bwd-design/doc-series-spec.md`(全文,含体例/铁律/反幻觉)。本卡只给 M1 专属输入。**

- **里程碑**:M1 = SiLU MAIN 闸门(第一段真正的反向数学:5 GEMM + dsilu,batched + no-mask + bf16 + atomic + hd64)。
- **commit(行号锚定它)**:`1b3c90b4`。先 `cd /root/workspace/ck_hstu && git show --stat 1b3c90b4` 看改了哪些文件。
- **旧 HTML**:**无**(M1 此前没写过讲义,你是从零写,但体例严格照 M0)。
- **事实来源**:`/tmp/hstu-bwd-design/M1-done.md` + `candidates.jsonl` 里 `"id":"M1-silu-gate"` 那条(含 R1/R2 闸门结论、对拍数值 dQ1.2e-4/dK0/dV3.9e-3、VGPR248/occ2)。R2 资源数据可参 `profile/M1-resource.md`(在 `/root/workspace/hstu-bwd-impl/`)。
- **输出**:`/root/workspace/hstu-b1052-report/hstu-bwd-M1-silu-gate-20260625.html`
- **M1 讲解重点(务必覆盖)**:
  1. 新 `hstu_attention_no_softmax_bwd_pipeline.hpp`(FMHA `kr_ktr_vr` 蓝本):7-stage / 5-GEMM 流水,**STAGE2** alpha + SiLU `p` & dsilu `g`,**STAGE5** `ds=dp*g`,dq/dk `*=alpha`(**dV 不乘**——这是易错点,要强调)。
  2. 自定义 kernel wrapper(携带 alpha + scale_p)+ float `dq_acc` atomic 累加 + POST convert。
  3. **R1 闸门**:FMHA `BlockFmhaBwdPipelineDefaultPolicy` 零覆写直接复用(为什么能复用)。
  4. **R2 闸门**:VGPR=248/AGPR=0/Scratch=0(无 spill)/occupancy=2,CDNA4 加法模型——闸门 MET 的含义。
  5. dispatch 从 M0 的 memset 桩 → 真 `RunSilu`(PRE 跳过、MAIN→POST)。
- **易错提示**:masked-out 时 SiLU 必须**显式置 0**(`dsilu(0)=0.5≠0`,禁 -inf);scale 双轴(alpha vs scale_p)别混。这些可作为「设计动机/易错点」note-block。
- 写完按规格 §6 回报。
