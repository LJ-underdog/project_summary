# M5 softmax bwd 实现派单 (pane-1 / 主 coder)

> lead 已完成:勘察 + 合并上游 fwd kStoreLSE + 基线适配。你负责 M5 softmax 路实现 + 对拍。
> 范围:**no_group(batched + jagged)softmax**。group softmax = M5b 后续,本单不做。
> 铁律:验证=对拍 CPU reference(bf16 rel≤2e-2/abs≤5e-2),**必须 -attn_scale=1.0**;SiLU 路保持逐字不变(零回归)。

## 0. 合并后的关键增量(先读,环境已变)
本分支已 merge `origin/hstu_attention_fwd`(commit 4bfb8e08),上游带来了 fwd 存 LSE 的完整基建:
- `HstuAttentionNoGroupFwdParams`(hstu_attention_params.hpp)新增:`bool is_training`、`void* lse_ptr`、`seq_stride_lse/nhead_stride_lse/batch_stride_lse`。当 `is_training && use_softmax` 时 fwd 写 LSE(**自然对数** `m+log(l)`)。
- fwd dispatch 已有 `kStoreLSE` 模板轴;instance 文件名多了 `lse_true/lse_false` 轴(已重新生成)。
- reference fwd 也支持输出 LSE(本单不需要,bwd reference 自己收 LSE 入参)。
- util 拆分:`hstu_attention_util.hpp` → `hstu_attention_host_util.hpp`(harness include 已由 lead 改好)。
- lead 已在 harness no_group fwd 段加了 `fp.is_training=false; fp.lse_ptr=nullptr;`(SiLU 基线)。M5 你要在 softmax 时把它们打开。

## 1. 数学(reference_hstu_attention_bwd.hpp 已是 oracle,kUseSoftmax=true 分支)
- LSE[sq] = log(Σ_sk exp(α·S))(fwd 存,自然对数);masked-out 的 S=-inf → exp(S-LSE)=0。
- P[sq,sk] = exp(α·S[sq,sk] − LSE[sq])
- D[sq] = Σ_v O[sq,v]·dO[sq,v](PRE 算)
- dV = Pᵀ@dO;dP = dO@Vᵀ;**dS = P·(dP − D)**;dQ = dS@K;dK = dSᵀ@Q
- scale 接线(同 SiLU):`dQ *= alpha`、`dK *= alpha`、**dV 不乘**。softmax 路 **不用 scale_p**。

## 2. MAIN softmax pipeline(新文件 `hstu_attention_with_softmax_bwd_pipeline.hpp`)
蓝本 = `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp`(FMHA 原版,已实现 softmax),
对照我们现成的 `hstu_attention_no_softmax_bwd_pipeline.hpp`(SiLU 版,FMHA 的 fork)。**做法:复制 SiLU pipeline 改 STAGE2/STAGE5,把 FMHA 的 LSE/D 加载搬回来**:

- operator() 签名:在 SiLU 版基础上**加 LSE/D 两个 dram window 入参**(参考 FMHA 版签名 line 90-112:`lse_dram_block_window_tmp`、`d_dram_block_window_tmp`),保留 `alpha`(去掉 `scale_p`)。
- LSE 加载:照搬 FMHA pipeline line 404-426(LSE: HBM→LDS→Reg,`MakeLSEDDramTileDistribution` + `MakeLSEDLdsWriteBlockDescriptor` + `MakeLSEDLdsReadBlockDescriptor`)。LDS 偏移用 `GetSmemSizeQT+OGrad+OGradT+Q`(FMHA line 411-415)。
- D 加载:照搬 FMHA line 428-450(D: HBM→Reg/LDS,偏移再 `+GetSmemSizeLSE`)。
- **STAGE2**(替换 SiLU 的 silu/dsilu 块,line 395-438):
  - 先 mask:边界 tile 把 `s_acc` 置 `-inf`(softmax 用 -inf,exp 后=0;**与 SiLU 不同——SiLU 禁 -inf,softmax 必须 -inf**)。用运行时门:
    ```
    if(mask.IsEdgeTile(seqlen_q_step, k_origin.at(number<0>{}), number<kM0>{}, number<kN0>{})) {
        set_tile_if(s_acc, -numeric<AccDataType>::infinity(),
                    [&](auto idx){ int r=seqlen_q_step+idx.at(number<0>{}); int c=k_origin.at(number<0>{})+idx.at(number<1>{});
                                   return !mask.IsTokenPairInsideMask(r,c); });
    }
    ```
  - 算 p:照搬 FMHA line 583-602 的 log2 域写法:
    ```
    auto row_lse = log2e_v<LSEDataType> * get_validated_lse(lse[i_idx]);
    p(i_j_idx) = exp2(scale * s_acc[i_j_idx] - row_lse);   // scale = alpha * log2e_v<AccDataType>
    ```
    其中 `scale` 在循环外:`const auto scale = alpha * log2e_v<AccDataType>;`(FMHA 的 raw_scale=alpha,scale=raw_scale*log2e)。`get_validated_lse`(LSE=-inf→0 防 NaN)是 ck_tile 现成函数。
  - **不要** SiLU 的 `g`,不要 scale_p。
- **STAGE5**(替换 SiLU 的 ds=dp*g,line 468-476):照 FMHA line 655-667:
  ```
  ds(i_j_idx) = p[i_j_idx] * (dp_acc[i_j_idx] - d[i_idx]);   // d 是 D 的 reg tile(per-row,i_idx)
  ```
  (无 dropout,去掉 undrop_flag 分支)
- STAGE3 里要 `load_tile(d_dram_window)` + 存 D 到 LDS(FMHA line 626-633),STAGE4 `load_tile(d_lds_read_window)` 得 `d`(FMHA line 648)。
- 其余(K/V/Q/dO 预载、5 个 GEMM、dK/dV/dQ 收尾 *=alpha)与 SiLU 版逐字相同。
- LSEDataType/DDataType 用 Problem 的(SiLU 版已 typedef,只是没用;softmax 要真用)。

## 3. PRE kernel(D=rowsum(O⊙dO),新自定义小 HIP kernel)
仿现成 POST `hstu_bwd_convert_dq_kernel`(在 batched_backward_dispatch.hpp 末尾或同处)写一个 `hstu_bwd_dot_do_o_kernel`:
- 输入 O、dO(InOutDataType,bf16),输出 D(float),布局 **[batch, head, seqlen_q] 连续 seq**(见 §5 布局约定)。
- 每 (batch, head, sq) 一行:`D = Σ_{v<hdim_v} float(O[...])*float(dO[...])`。
- batched:用 batch/nhead/seq strides 定位 O/dO。jagged:用 `seq_q_offsets_ptr[i_batch]` 定 token 基址(token-major packed dim0=1),D 也按 packed [ΣL, head] 写。
- grid/block 随意(每 thread 一行,或 block 内 reduce hdim);hdim_v=64 很小,单 thread 累加即可。
- 在 dispatch 里 **MAIN 之前** launch(memset D 不需要,全覆盖写)。

## 4. softmax bwd kernel(新 wrapper,仿 SiLU kernel)
`hstu_attention_bwd_kernel.hpp` 里现有 `HstuAttentionBwdDQDKDVKernel`(SiLU)。**新增一个 softmax kernel**(或给现有 kernel 加 `kUseSoftmax` 编译期分支——倾向新 kernel,SiLU 路零改动):
- Kargs 加 `lse_ptr`、`d_ptr`、`nhead_stride_lse`、`batch_stride_lse`(seq 连续 stride=1),去掉 scale_p(softmax 不用)。
- operator() 里照 SiLU kernel 建 Q/K/V/dO/dQ window 的同款方式,**多建 LSE window + D window**(参考 FMHA `fmha_bwd_kernel.hpp` 的 lse/d window 构造),传给 softmax pipeline。
- LSE/D 的 dram view:per (i_batch,i_nhead) 偏移 = `i_batch*batch_stride_lse + i_nhead*nhead_stride_lse`,window 沿 seq 长度 seqlen_q、stride=1。jagged:基址用 q_offset。

## 5. 布局约定(LSE/D 都用 [batch, head, seq] 连续 seq)
FMHA bwd pipeline 的 LSE/D window 沿 seq 连续(seq_stride=1)。所以:
- **bwd 侧** D 与 LSE 都按 `[batch, head, seqlen_q]`(seq 最内、连续)。`nhead_stride_lsed = seqlen_q`(jagged: max_seqlen_q 或 packed 总长按需)、`batch_stride_lsed = num_head*seqlen_q`。
- **fwd 产 LSE 时**(harness §6)把 fwd 的 lse stride 设成同布局:`seq_stride_lse=1, nhead_stride_lse=seqlen_q, batch_stride_lse=num_head*seqlen_q`,这样 fwd 直接写出 bwd 要的连续-seq 布局,**无需转置**。
- 我们 bwd_params 现有 `nhead_stride_lsed/batch_stride_lsed`(无 seq_stride,因 seq 连续=1)。够用。

## 6. harness(example_hstu_attention_bwd.cpp,no_group 段)
- softmax(use_softmax=true)时:
  1. fwd 段:`fp.is_training=true; fp.lse_ptr=lse_dev; fp.seq_stride_lse=1; fp.nhead_stride_lse=phy_seqlen_q; fp.batch_stride_lse=num_head*phy_seqlen_q;`(jagged 按 packed 调整);fwd 跑完 `lse_dev.FromDevice(lse_host)`。
  2. lse_host 喂给 reference bwd(其签名已收 `lse_batch_seq_nhead`——**注意 reference 期望的 LSE 布局**,核对 reference 是 [batch,seq,head] 还是 [batch,head,seq];若与我们 [batch,head,seq] 不一致,给 reference 单独建一个它要的布局的 lse_host_ref,或调 reference 的索引。**这一步务必对齐,否则 silent-wrong**)。
  3. 分配 `d_dev`(float,大小 = batch*head*seqlen_q,jagged 按 packed),接 `bp.d_ptr=d_dev; bp.lse_ptr=lse_dev; bp.nhead_stride_lsed=phy_seqlen_q; bp.batch_stride_lsed=num_head*phy_seqlen_q;`。
- SiLU 路保持 lead 设的 `is_training=false/lse_ptr=nullptr/d_ptr=nullptr`。

## 7. dispatch(hstu_attention_batched_backward_dispatch.hpp)
- 去掉 `kUseSoftmax` 的 throw(line ~219-221)。`if constexpr(kUseSoftmax)` 分支:选 softmax pipeline + softmax kernel;launch 顺序 **PRE(dot_do_o)→ memset dq_acc → MAIN → POST(convert_dq)**。
- SiLU 分支(else)逐字不变。
- instance:`*_softmax_true_*` 的 bwd instance 已生成(別忘了 instances_ref.hpp 聚合头要含 softmax_true);no_group_backward_bf16.cpp 的 BOOL_SWITCH_2 已 cover use_softmax,无需改。

## 8. 验证(对拍,逐档,先探后锁)
build:`cd /root/workspace/ck_hstu && cmake --build build --target tile_example_hstu_attention_bwd -j128`(BUILD_DEV=OFF 已配)。
二进制:`build/bin/tile_example_hstu_attention_bwd`。**全部 -attn_scale=1.0**。
逐档(softmax × 因子 × causal∈{0,1} × {batched,jagged}):
- no-mask、causal、window、contextual、min_full、num_target、全组合;每个跑 causal=0 和 causal=1(§7 交叉教训)。
- batched 与 jagged 各一遍(per-batch 变长、非整除 seqlen)。
期望全 PASS,bf16 舍入级。**任何一格 FAIL 如实记为缺陷,别绕过**。
日志写 `/root/workspace/hstu-bwd-impl/runs/run-M5-*.log`。

## 9. 落地产出(交回 lead)
1. 测试套件升级:`test/run_bwd_tests.py` 把 `reject-softmax` 删/改为 pass,新增 softmax × causal{0,1} × 因子 交叉案;`python3 test/run_bwd_tests.py` 整体 exit 0。
2. 写 `/tmp/hstu-bwd-design/M5-done.md`:改了哪些文件、STAGE2/5 关键代码、LSE 布局对齐结论、逐档对拍表(误差数值)、套件总数/exit、SiLU 零回归证据、遇到的坑。
3. 更新 `candidates.jsonl` 加 M5 条目(promoted,附 evidence 日志路径)。
4. **不要动 fwd 逻辑**(只在 harness 里开 is_training 产 LSE)。group softmax 不做(M5b)。

## 10. 速查
- FMHA softmax bwd pipeline(蓝本):`include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp`(LSE line 404-426,D line 428-450,p=exp2 line 583-602,ds=p*(dp-d) line 655-667)。
- SiLU pipeline(我们的,要 fork):`hstu_attention_no_softmax_bwd_pipeline.hpp`(STAGE2 line 395-438,STAGE5 line 468-476)。
- oracle:`reference_hstu_attention_bwd.hpp`(kUseSoftmax 分支)。
- PRE 蓝本(可不用,自定义更简单):`include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dot_do_o.hpp`。
- POST 现成参考:batched_backward_dispatch.hpp 末尾 `hstu_bwd_convert_dq_kernel`。
