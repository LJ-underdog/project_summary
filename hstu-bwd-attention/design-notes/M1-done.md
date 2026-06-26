# M1 端到端闸门 — 完成报告 (pane-1 / coder)

状态:**✅ 闸门通过**。SiLU 路 MAIN 真实计算,数值对拍在 **attn_scale=1.0(梯度量级 ~5)下真 PASS**,exit 0。R1/R2 均闭环。
范围:batched + SiLU(kUseSoftmax=false)+ no-mask(causal=0)+ bf16 + atomic + hdim_qk=hdim_v=64。日期 2026-06-04。

## 0. 关键发现:目标 ck_tile 是 ck_hstu 自带的较新版本
`/root/workspace/ck_hstu/include/ck_tile` 与 `/root/ck` 的 FMHA bwd **接口不同**(ck_hstu 更新):
- `TileFmhaBwdTraits<kPadHeadDimQ, kPadHeadDimV, BiasEnum, kHasBiasGrad, kBlockPerCu>`(headdim-pad 为 index_t 0/8/1;**seqlen 从不 padding**——OOB 由 buffer_load 归零)。
- `TileFmhaBwdShape<...10 warps/tiles..., kMaxSeqLenQ=0>`(多 maxq;Gemm4WarpTile=`<16,16,min(wk0,bk4)>`)。
- `BlockFmhaBwdPipelineProblem<...,Mask,Dropout, kUseTrLoad, Traits>`(多 `kUseTrLoad`)。
- kr_ktr_vr `operator()(void* smem_ptr, ...)`(smem 在首位);policy 方法名一致。
- bwd kernel 用 `sequence<false,(kPadHeadDim>0)>` 做 pad,确认 **seqlen 无需 padding**。

→ 据此把 problem/traits/shape/dispatch 全部对齐到 ck_hstu 版本(初版误用 /root/ck 接口,已修正)。

## 1. 文件改动(`example/ck_tile/18_hstu_attention/`)
- **新建 `hstu_attention_no_softmax_bwd_pipeline.hpp`** — `HstuAttentionBwdDQDKDVPipelineKRKTRVR<Problem,Policy>`,以 ck_hstu kr_ktr_vr 为结构蓝本改写(不直接 include):
  - STAGE1 `gemm_0` 出未缩放 s_acc;
  - **STAGE2**:`s=alpha*s_acc`;`p=silu(s)*scale_p`(→dV);`g=scale_p*dsilu(s)`(→STAGE5)。device 函子 `sigmoid=rcpf(1+expf(-x))`、`silu=s*sigmoid`、`dsilu=sigmoid*(1+s*(1-sigmoid))`(fp32,镜像 fwd `f_silu`)。**不 load LSE/D**;
  - STAGE3/4 复用(`gemm_1` dV、`gemm_2` dP);
  - **STAGE5**:`ds=dp_acc*g`;
  - STAGE6/7 `gemm_3` dK、`gemm_4` dQ;**收尾 `dq_acc*=alpha`、`dk_acc*=alpha`,dV 不乘**;
  - LDS 指针偏移与 FMHA 字节一致(保留未用的 LSE/D/bias 段 → GetSmemSize 不变,R6 可省)。
- **改写 `hstu_attention_bwd_kernel.hpp`** — `HstuAttentionBwdDQDKDVKernel<Pipeline,DKEpi,DVEpi>`(batched,无 bias/dropout/group),plain kargs **携带 alpha+scale_p**(FMHA kernel 仅传 raw_scale/scale 且 scale=raw_scale*log2e,无法塞第二标量→必须自写 kernel);建 q/k/v/do 窗口 + **float dq_acc 的 atomic_add 窗口** + dk/dv 窗口;调 pipeline 后 `Default2DEpilogue{}(win,acc,nullptr)` 写 dk/dv。另含 **POST** `hstu_bwd_convert_dq_kernel`(模板化 __global__,vague linkage):dq_acc(float)→dq(bf16) elementwise cast(atomic 路 nsplits=1、dq_acc 与 dq 同布局)。
- **改写 `hstu_attention_batched_backward_dispatch.hpp`** — `RunSilu` 组 shape/traits/problem/pipeline/epilogue/kernel;`hipMemsetAsync(dq_acc,0)` → launch MAIN(atomic)→ `hipLaunchKernelGGL` POST。`Run` 用 `if constexpr` 门控:deterministic→M6 throw、softmax→M5 throw、causal→M2 throw,仅 (SiLU,no-mask,hd64) 实例化真实 kernel。
- **改 `example_hstu_attention_bwd.cpp`** — 分配 float dq_acc workspace(nsplits=1,与 dQ 同 [b,sq,h,hdim] 布局)、设 `max_seqlen_q`(scale_p 来源)、exit code 改 `numeric_pass?0:-2`。
- 复用 FMHA(直接 include):`BlockFmhaBwdPipelineDefaultPolicy`、`GenericAttentionMask<false>`(平凡 `GetTileRangeAlongY→(0,seqlen_q)`,P1-D)、`BlockDropoutBwd<false,true,false>`、`Default2DEpilogue`、`BlockFmhaBwdPipelineProblem`(保留 `BiasEnum=NO_BIAS`+`BiasDataType` dummy,P1-A)。

**未改 fwd 行为**;mask 未碰(no-mask)。

## 2. 编译 — 0 error
`cmake --build build --target tile_example_hstu_attention_bwd -j` → **0 error**,binary 链接成功。证据 `runs/build-bwd-M1.log`。

## 3. 数值对拍(attn_scale=1.0 必列,梯度量级 ~5)
命令:
```
./build/bin/tile_example_hstu_attention_bwd -prec=bf16 -b=2 -nhead=2 -hdim_qk=64 -hdim_v=64 -seqlens=128 -softmax=0 -causal=0 -attn_scale=1.0 -v=1
```
结果(`runs/run-bwd-M1-attnscale1.log`,exit 0):
```
  dQ: max_abs_err=0.00012207 mean_abs_err=3.7e-09 (max|ref|=5.125)
  dK: max_abs_err=0          mean_abs_err=0       (max|ref|=5.21875)
  dV: max_abs_err=0.00390625 mean_abs_err=2.9e-07 (max|ref|=4.625)
[PASS] dQ   [PASS] dK   [PASS] dV
```
误差为 bf16 舍入级,远内于阈值(rel≤2e-2/abs≤5e-2)。**非巧合**:量级 ~5 的真实梯度下三者皆 PASS,dK 逐位为 0。

稳定性(6/6 PASS,exit 0):默认 attn_scale(scale_p=1/max_seqlen_q)、b4×nhead8×seq256、b1×nhead1×seq512、**seq200(非 kN0=128 整除)**、**seq130(非 kM0=32 整除→验 OOB 归零)**。

## 4. R1 — FMHA default policy 复用结论
**✅ 直接复用,零覆写。** 用 `BlockFmhaBwdPipelineDefaultPolicy` 配 `BlockFmhaBwdPipelineProblem`(BiasEnum=NO_BIAS + dummy BiasDataType),我的 SiLU pipeline 原样调用 policy 全部 `Make*/Get*` 描述子 + 5 个 GEMM + `PTFromGemm0CToGemm1A`/`SGradTFromGemm2CToGemm3A`,GEMM 间插入 dsilu/elementwise 与 policy 的中间 reg 分布**完全兼容**。无需派生 HstuBwdPolicy。R1 风险关闭。

## 5. R2 — CDNA4 占用(VGPR/AGPR/Scratch)
`amdclang++ -Rpass-analysis=kernel-resource-usage`(profile/M1-resource.md):
| 指标 | MAIN | 判据 |
|---|---|---|
| VGPRs | **248** | |
| AGPRs | **0** | MFMA 累加器落统一 VGPR 池 |
| **ScratchSize** | **0** | ✅ 无 spill(M1 硬性要求 MET)|
| VGPR/SGPR Spill | 0 / 0 | ✅ |
| **Occupancy** | **2 waves/SIMD** | ✅ 未掉到 1 wave |
| LDS | 32768 B | 含未用 LSE/D/bias 段(R6 可省)|

**CDNA4 加法模型**:occupancy=ArchVGPR+AGPR=248+0=248 → 512/248≈2 waves(编译器一致)。留 g(+~32 VGPR)**未致 spill、未掉 wave**。注:AGPR=0,R2 设计担忧的"留 g 叠加 AGPR 累加器"最坏情形本构型未出现,仍余 2 waves。**R2 在 hd64 上闭环通过。**

## 6. candidates / benchmark
- candidates.jsonl 加 `M1-silu-gate`(status pass,含数值 err + R1 结论 + VGPR/occupancy)。
- benchmark.csv 记 VGPR=248、occupancy=2(perf TFLOPS 未测——bwd harness 暂无计时,留 M8)。

## 7. 遗留 / 给后续里程碑
- **M2 mask**:dispatch causal 现 throw;需 HSTU 5 因子 mask 的 `GetTileRangeAlongY`(非平凡)+ `IsEdgeTile` + STAGE2 `set_tile_if(p,g<-0)`(代码已留注释位)。
- **M3 jagged / M4 group**:dispatch/kernel 现仅 batched;group throw。
- **M5 softmax**:现 throw;需 PRE(D)+ LSE 读取 + STAGE5 `ds=p*(dp-d)` + `get_validated_lse`(P1-1)。
- **M6 deterministic**:现 throw;POST 换 `BlockFmhaBwdConvertQGrad` reduce+convert,dq_acc 多 split。
- **M7 fp16 + hdim{96,128,256}+qk≠v**:dispatch 现 assert hd64+bf16。
- **M8 perf**:启用 ck_hstu 的 **trload kr_ktr_vr**(gfx950 专用,现用 kUseTrLoad=false 非最优)+ 砍 SiLU 未用 LDS 段(R6)+ bwd harness 加计时。
- 无未解决阻塞点;R1/R2 设计期风险均已实测关闭。
