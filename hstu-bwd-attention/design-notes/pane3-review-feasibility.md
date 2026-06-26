# 派给 pane-3(角色:reviewer · 工程可行性)— 审 HSTU bwd DESIGN.md 的落地与风险

调度模式:tmux pane-3。独立、对抗式 review。不要派 sub-teammate,不重写正文。

## 对象
`/tmp/hstu-bwd-design/DESIGN.md`

## 基准(以源码为准)
- FMHA bwd 全套:`/root/ck/.../pipeline/block_fmha_bwd_*.hpp`、`kernel/fmha_bwd_kernel.hpp`、`pipeline/block_fmha_bwd_pipeline_default_policy.hpp`
- HSTU fwd 工程:`hstu_attention_*_forward_dispatch.hpp`、`hstu_attention_pipeline_problem.hpp`、`hstu_attention_traits.hpp`、`tile_setting_define.hpp`、`generate_instances.py`、`CMakeLists.txt`
- 硬件:rocm-ref `/tmp/rocm-ref`(occupancy/VGPR/LDS)

## 重点审这些(逐条 ✅/⚠️/❌ + 证据 + 修法)
1. **M1 风险闸门可行性(最关键)**:DESIGN 说"仅 MAIN 特化、PRE/POST/policy/shape/enum 零成本复用 FMHA"。**真能吗?** 核查:FMHA `block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr` 的 Problem/Policy 模板形参是否容得下 HSTU 的(SiLU 双路、5 因子 mask 类型、scale_p、hdim_qk≠hdim_v、jagged/group 索引)?哪些是硬编 softmax/dropout/bias 假设、需要 fork 而非复用?给出"可直接复用 / 需小改 / 必须 fork"的真实判定。
2. **mask Y 成员新增**:给 4 个 mask struct 加 `GetTileRangeAlongY`/`IsEdgeTile`,工作量与正确性验证(离线校验方案)是否现实。
3. **params/kargs 字段表(§4.6)**:对照 FMHA `fmha_bwd_args` + HSTU fwd params,字段是否齐(stride 全、lse/d/dq_acc、group 指针),命名一致无遗漏。
4. **dQ 写回两路 + convert_dq 复用**:atomicAdd(float dq_acc)默认 + deterministic split-workspace;`BlockFmhaBwdConvertQGrad` 能否直接复用;dq_acc workspace 显存与 stride 估算是否合理。
5. **instances/codegen + CMake**:不搬 FMHA codegen、扩 HSTU `generate_instances.py` 是否可行;实例化矩阵 384→MVP 48 收敛是否合理;编译规模/时间风险。
6. **3-kernel GridSize/launch 顺序**:PRE(seqlen_q)→MAIN(seqlen_k,nhead,batch)→POST(seqlen_q),与 FMHA 一致性核实。
7. **里程碑 M0–M8 + 测试矩阵**:阶梯是否合理、验收标准可测、对拍流程(fwd 产 O/LSE → bwd → diff CPU ref)可操作。
8. **占用率/资源**:SiLU 留 g + 常驻 K/Kᵀ/V + dk/dv_acc 的 VGPR 压力,对照 rocm-ref(512 VGPR/SIMD,占用阶梯)粗估是否会掉 occupancy / 触发 spill。

## 收尾
- 报告写 `/tmp/hstu-bwd-design/review-feasibility.md`:P0(不可行/会错)/P1(风险大需缓解)/P2(可选);每条 §位置 + 源码证据 + 缓解或修法。**不改正文**。
- 特别给出对 **U1–U4 四个真未决** 的工程视角建议(bias、deterministic 默认、MVP 覆盖面、GQA/MQA)。
- progress 简洁。
