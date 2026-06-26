# 派给 pane-1(角色:coder)— HSTU bwd 实现 M1(风险闸门)

调度模式:tmux pane-1。接续你刚完成的 M0。这是**全项目最关键一步**:让 SiLU 路 MAIN 真实计算、数值对拍 PASS。严守 kernel-design-rocm skill:每步编译+对拍验证,证据进 `/root/workspace/hstu-bwd-impl/`。不要派 sub-teammate。

## 依据
- 设计骨架:`/tmp/hstu-bwd-design/DESIGN.md` §2.2/§2.3(MAIN 双路骨架,逐 STAGE 改造)、§4.2(problem/policy + P1-A)、§4.4(dQ 两路)、§6(M1 验收)、§4.8(gfx950/CDNA4)。
- M0 接力点:见 `/tmp/hstu-bwd-design/M0-done.md` §6。
- FMHA MAIN 蓝本:`/root/ck/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp`(STAGE1:518 / STAGE2:520-622 / STAGE3:624-646 / STAGE4:648-655 / STAGE5:657-669 / STAGE6:698-708 / STAGE7:718-757 / 收尾:763-773);POST:`block_fmha_bwd_convert_dq.hpp`;policy:`block_fmha_bwd_pipeline_default_policy.hpp`;kernel 包装:`fmha_bwd_kernel.hpp`。

## M1 范围(只此组合)
**batched + SiLU(kUseSoftmax=false)+ no-mask(causal=0)+ bf16 + atomic + hdim_qk=hdim_v=64**。其余(mask/jagged/group/softmax/fp16/deterministic)M2+ 再做,保持 TODO。

## 要做的
1. **HSTU MAIN pipeline**(新建 `hstu_attention_no_softmax_bwd_pipeline.hpp`,以 FMHA kr_ktr_vr 为结构蓝本改写,**不直接 include**):
   - STAGE1 复用 `gemm_0` 出未缩放 s_acc。
   - **STAGE2**:`s_acc *= alpha`;`p = silu(s_acc) * scale_p`(→dV);`g = scale_p * dsilu(s_acc)`(→STAGE5)。**SiLU 路不 load LSE**(`if constexpr(kUseSoftmax)` 跳过 lse/d 的 load,见 M0/§2.3)。no-mask 阶段无需置零(M2 再加 set_tile_if)。
   - STAGE3/4 复用(`gemm_1` dV、`gemm_2` dP,沿 hdim_v)。
   - **STAGE5**:`ds = dp_acc * g`(逐元素;g 已含 scale_p)。
   - STAGE6 `gemm_3` dK、STAGE7 `gemm_4` dQ;**收尾 `dq_acc *= alpha`、`dk_acc *= alpha`,dv 不乘**(raw_scale 槽→alpha)。
   - 提供 device 函子 `silu(x)=x*sigmoid(x)`、`dsilu(x)=sigmoid(x)*(1+x*(1-sigmoid(x)))`(fp32 compute)。
2. **problem/traits**:`HstuAttentionBwdPipelineProblem`,**保留 `BiasEnum=NO_BIAS`+`BiasDataType` dummy**(P1-A,否则 default policy 不编译);砍 dropout/randval/biasgrad;加 `kUseSoftmax`/`CompDataType`。`using Policy = BlockFmhaBwdPipelineDefaultPolicy;`(先直接复用,验 R1)。
3. **mask 平凡版**:M1 no-mask 也需 `GetTileRangeAlongY→(0,seqlen_q)` 与 `IsEdgeTile→false`(MAIN 无条件调用,见 P1-D)。给 HSTU mask 加这两个平凡成员,或在 no-mask 路用一个 trivial mask 适配。
4. **3-kernel 接线**(改 `hstu_attention_batched_backward_dispatch.hpp`):memset 换成 **MAIN(写 float dq_acc + dk/dv)→ POST(convert dq_acc→dq_ptr)**;SiLU 路**不发 PRE**。用 `fmha_bwd_kernel.hpp` 的 kernel 包装或自写最小包装。
5. **harness**(改 `example_hstu_attention_bwd.cpp`):分配 float `dq_acc` workspace(nsplits=1,hipMalloc);exit code 改 `numeric_pass?0:-2`(M0 的接力点)。

## 验收(M1 闸门 —— 全过才算通过)
1. **编译 0 error**(`cmake --build build --target tile_example_hstu_attention_bwd -j`,log 进 runs/build-bwd-M1.log)。
2. **数值对拍 PASS**,且必须在**梯度量级有意义**下验(用 `-attn_scale=1.0` 放大,避免 M0 那种"ref 太小巧合 PASS"):
   ```
   ./build/bin/tile_example_hstu_attention_bwd -prec=bf16 -b=2 -nhead=2 -hdim_qk=64 -hdim_v=64 -seqlens=128 -softmax=0 -causal=0 -attn_scale=1.0 -v=1
   ```
   要求 dQ/dK/dV 三者 [PASS](bf16 rel≤2e-2/abs≤5e-2),exit 0。再跑默认 attn_scale 与几个 seqlen/b/nhead 变体确认稳。
3. **R1**:记录 FMHA default policy 是否直接复用成功(若编译/分布不兼容,记录如何 override)。
4. **R2(CDNA4 占用)**:`--save-temps` 或 `-Rpass-analysis=kernel-resource-usage` 取 MAIN 的 **VGPR/AGPR/ScratchSize**;按 **CDNA4 加法模型 occupancy=ArchVGPR+AGPR**(512/SIMD)评估 wave 数;**要求 ScratchSize=0**(无 spill)。结果记 `profile/M1-resource.md`。
5. candidates.jsonl 加 `M1-silu-gate`(pass/fail + 数值 err + VGPR/occupancy + R1 结论);benchmark.csv 记一行(可选 TFLOPS)。

## 铁则
- 不改 fwd 行为;数值不对就调到对(这是 M1 的核心攻坚)。**严禁**为了 PASS 放宽容差或挑巧合 case——必须 attn_scale=1.0 下真 PASS。
- 卡住超过合理尝试,在报告里如实写阻塞点 + 已试方案 + 怀疑方向(尤其 R1 policy 分布不兼容、dsilu 数值、scale 落点),别假装通过。
- 完成写 `/tmp/hstu-bwd-design/M1-done.md`:文件改动、build、对拍数值(attn_scale=1.0 必列)、R1 结论、R2 资源数(VGPR/AGPR/Scratch/occupancy CDNA4)、candidates/benchmark 更新、遗留。
- progress 简洁;长 log 进文件。
