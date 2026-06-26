# M5 softmax bwd 独立验证 + 对抗式 review 派单 (pane-2 / reviewer)

> 你是独立验证者。pane-1(coder)报告 M5 softmax(no_group)完成、对拍全过。**别信自述,独立验证**。
> 你的产出决定 lead 是否 promote。代码全在磁盘,git 基线 = `b0c08cba`(M5 改动相对它)。

## 范围(M5 = no_group:batched + jagged softmax;group=M5b 未做,不在本次范围)
改动文件(相对 b0c08cba):
- **新** `example/ck_tile/18_hstu_attention/hstu_attention_with_softmax_bwd_pipeline.hpp`(softmax MAIN,SiLU pipeline 的 fork)
- `hstu_attention_bwd_kernel.hpp`(+新 `HstuAttentionBwdDQDKDVSoftmaxKernel` + PRE `hstu_bwd_dot_do_o_kernel`)
- `hstu_attention_batched_backward_dispatch.hpp`(+`RunSoftmax`:PRE→memset→MAIN→POST;去 softmax throw)
- `example_hstu_attention_bwd.cpp`(softmax 时 fwd 开 is_training 产 LSE + 转置喂 reference + d_dev)
- SiLU 文件**不应在 diff 里**(`hstu_attention_no_softmax_bwd_pipeline.hpp`、SiLU kernel、group)——若在,立即报 lead。

## 任务 A:独立机器验证(权威闸门)
1. 干净重建:`cd /root/workspace/ck_hstu && cmake --build build --target tile_example_hstu_attention_bwd -j128 2>&1 | tee /root/workspace/hstu-bwd-impl/runs/build-M5-review.log`;确认 0 error。
2. 独立复跑套件:`python3 /root/workspace/hstu-bwd-impl/test/run_bwd_tests.py`;确认 **整体 exit 0、0 FAIL**,记录 TOTAL/PASS/SKIP(自述称 60/59/1)。
3. 抽 3-5 档**自己重跑** binary 对拍(别只信套件),含:softmax causal=0+num_target(batched)、softmax causal=0+num_target(jagged)、softmax 全组合 combo、softmax 非整除 seqlen。**全部 -attn_scale=1.0**。binary=`build/bin/tile_example_hstu_attention_bwd`。

## 任务 B:对抗式代码 review(专盯 silent-wrong)
读 `M5-dispatch.md`(原始规格)+ `M5-done.md`(coder 自述,当地图但要核)+ diff + `reference_hstu_attention_bwd.hpp` 的 kUseSoftmax 分支。逐条核:
1. **LSE 域**:fwd 存的是自然对数 `m+log(l)`?pipeline 是否 `exp2(α·log2e·S − log2e·LSE)`(= exp(α·S−LSE))?log2e 因子两处都在?`get_validated_lse`(LSE=-inf→0)有没有?
2. **掩码方向(易抄反)**:softmax STAGE2 是否对 `!IsTokenPairInsideMask` 置 **-inf**(不是置 0;SiLU 才置 0)?边界门用运行时 `IsEdgeTile`(非编译期 IsMasking),causal=0+num_target 是否仍掩(P1-1 同类)?— 自己跑 causal=0+target 对拍佐证。
3. **STAGE5**:`ds=p*(dp−D)`,D 是 per-row(i_idx)?符号/广播对不对?
4. **scale 接线**:dQ*=alpha、dK*=alpha、**dV 不乘**?softmax 路确实没用 scale_p?
5. **LSE/D 布局对齐(最大 silent-wrong 风险)**:GPU 侧 [batch,head,seq] 连续-seq;reference 侧 [batch,seq,head];harness 转置是否正确(`lse_host(b,s,h)=lse_flat[(b*H+h)*seq+s]`)?**GPU-bwd 与 reference 是否用同一份 GPU-产 LSE**?jagged 下 nhead_stride/基址(query_start)对不对?
6. **PRE kernel**:D=Σ_v O*dO 累加用 float?O/dO 定位 strides 对?jagged 用 offset?D 写的布局 == MAIN 读 D 的布局?越界(num_target/per-batch 数组、packed)?
7. **smem**:softmax 用了 SiLU 预留的 LSE/D region,GetSmemSize 是否够(没爆 LDS)?
8. **harness 边界数组**:num_targets/per-batch supplement 到正确长度(M2 抓过越界)?
9. **SiLU 零回归**:diff 是否真没碰 SiLU/group 源;套件里 M1-M4b 全 PASS?

## 产出(写 `/tmp/hstu-bwd-design/M5-review-findings.md`,报 lead)
- 任务 A 结果(build/suite/抽样对拍的实测数值 + exit)。
- 任务 B 逐条结论(GREEN / 问题),每个问题给文件:行号 + 复现命令 + 期望 vs 实际。
- 总评:可 promote / 需修(列 blocker)。
- **发现真缺陷如实列,别和稀泥**;没问题也明说"逐条核过 GREEN"。
