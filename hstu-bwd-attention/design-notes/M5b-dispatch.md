# M5b group softmax 实现派单 (pane-1 / 主 coder)

> 在 M5(no_group softmax,git aced5784)+ M4(group SiLU 双 pipeline)之上,把 softmax 接到 **group** 路。
> 你写过 M4 group kernel 和 M5 softmax pipeline,这是两者合流。**铁律不变**:对拍 CPU reference,`-attn_scale=1.0`,SiLU/no_group 路逐字不变(零回归)。

## 0. 核心思路:复用,不重写
- **MAIN pipeline 不动**:`hstu_attention_with_softmax_bwd_pipeline.hpp`(M5 写的)是 mode-agnostic 的——group 是 jagged 超集,per-group 超参在 kernel 层处理,pipeline 只吃 mask/alpha/LSE/D window。**直接复用**。
- **PRE kernel 不动(先核)**:`hstu_bwd_dot_do_o_kernel` 已处理 jagged packed([head,ΣL] 连续-seq);group 的 O/D 也是同款 packed 布局 → **应可直接复用**。核一遍 group 的 D 布局/stride 与它一致即可,不一致再说。
- 要新写的就两块:**group softmax kernel** + **group dispatch 的 RunSoftmax** + **group harness 的 fwd-产-LSE 接线**。

## 1. group softmax kernel(hstu_attention_bwd_kernel.hpp)
现有:
- `HstuAttentionBwdDQDKDVGroupKernel<PipelineLocal,PipelineNoLocal,DKEpi,DVEpi>`(M4,SiLU 双 pipeline,运行时按 window>0 选 local/no-local,读 per-group 超参)。
- `HstuAttentionBwdDQDKDVSoftmaxKernel`(M5,no_group softmax,带 LSE/D window,单 pipeline)。
**新写** `HstuAttentionBwdDQDKDVGroupSoftmaxKernel`:= M4 group kernel 的 per-group 超参/jagged offset/双 pipeline 选择 **＋** M5 softmax kernel 的 LSE/D window 构造 + 调 softmax pipeline(去掉 scale_p)。
- Kargs 在 group kernel 基础上加 `lse_ptr/d_ptr/nhead_stride_lsed/batch_stride_lsed`(布局同 M5:[head, global_token] 连续-seq;group packed 下 nhead_stride=ΣL、token 基址=query_start)。
- 双 pipeline 都用 **softmax** pipeline 的 with-local / no-local 两个实例(对应 IsMasking;同 M4 的运行时 window>0 选择)。
- LSE/D window 构造照 M5 softmax kernel(per (i_batch=group 内 batch, i_nhead) 偏移)。注意 group 的 i_group=i_batch/num_batch_per_group 取 per-group 超参,但 LSE/D 是 per-(batch,head,token) 与 group 无关,按 jagged 同款定位。

## 2. group dispatch(hstu_attention_group_backward_dispatch.hpp)
- include `hstu_attention_with_softmax_bwd_pipeline.hpp`。
- `kUseSoftmax` 分支(现 :196 throw)改为:用 softmax 的 PipelineLocal/PipelineNoLocal 组 `HstuAttentionBwdDQDKDVGroupSoftmaxKernel`,走 **RunSoftmax**(PRE dot_do_o → memset dq_acc → MAIN group kernel → POST convert_dq),与 M5 no_group 的 RunSoftmax 同结构(参考 batched_backward_dispatch 的 RunSoftmax)。
- PRE/POST 的 grid/packed 尺寸用 group 的 total/packed(参考 M4 group dispatch 现有 memset/POST 的 total_dq_acc_elems、grid=max_seqlen_q 写法)。
- SiLU 分支(else)逐字不变。

## 3. group harness(example_hstu_attention_bwd.cpp,run_group_hstu_bwd 段)
现状:group 段 **跳过 GPU fwd**(SiLU,O unused,`bp.o_ptr=nullptr; bp.lse_ptr=nullptr`)。softmax 要补:
1. **跑 GPU group fwd 产 O+LSE**:填 `HstuAttentionGroupFwdParams`(它有 `is_training/lse_ptr/seq_stride_lse/nhead_stride_lse/batch_stride_lse`),softmax 时 `is_training=true`、lse 用 [head,ΣL] 连续-seq 布局(`seq_stride_lse=1, nhead_stride_lse=ΣL, batch_stride_lse=...`),调 `hstu_attention_group_forward_bf16`(merge 后 group fwd 支持 kStoreLSE),`o_dev.FromDevice`/`lse_dev.FromDevice`。参考 no_group 段 M5 的 fwd-产-LSE 写法(harness 已有 lse_dev/d_dev 布局常量 nhead_stride_lsed=phy_seqlen_q 等,group 段照搬)。
2. **LSE 转置喂 reference**:`reference_group_hstu_attention_bwd` 收 lse_host([batch,seq,head]);把 GPU LSE([head,ΣL])转置进去,同 M5 no_group 的转置(注意 group 是 packed,token=offset+s)。
3. **分配 d_dev**(group packed 大小)+ 接 `bp.o_ptr/lse_ptr/d_ptr/nhead_stride_lsed/batch_stride_lsed`。
4. SiLU(use_softmax=false)group 段保持原样(跳 fwd)。

## 4. 验证(对拍,-attn_scale=1.0,bf16)
build:`cmake --build build --target tile_example_hstu_attention_bwd -j128`。binary 同前,group 用 `-g=<num_group>` + per-group 参数(`-g_max_seqlens/-g_local_lens/-g_context_lens/-g_minfull_lens/-g_attn_scales`,见 M4 harness)。
逐档(softmax × causal{0,1} × per-group 异构):
- g=2/3/4;per-group window(混 with/without-local)、per-group context/minfull、per-group attn_scale、**全异构**(各组超参都不同,证 i_group 真索引)、per-batch num_target、causal=0+num_target(P1-1 同类)。
- 期望全 PASS,bf16 舍入级。任何 FAIL 如实记缺陷。日志 `runs/run-M5b-*.log`。

## 5. 落地产出(交回 lead)
1. 测试套件加 group softmax 交叉案(`test/run_bwd_tests.py`);`python3 run_bwd_tests.py` 整体 exit 0。
2. 写 `/tmp/hstu-bwd-design/M5b-done.md`(改了哪些文件、group softmax kernel 与 M4/M5 的复用边界、LSE/D group packed 布局、逐档对拍表、套件总数/exit、no_group+SiLU 零回归证据、坑)。
3. 更新 `candidates.jsonl` 加 `M5b-group-softmax`(promoted,附 evidence)。
4. **不动 fwd 逻辑**(只在 group harness 开 is_training)、不动 M5 no_group/SiLU 已 promoted 的文件逻辑。

## 6. 速查
- M4 group kernel + group dispatch:`hstu_attention_bwd_kernel.hpp`(GroupKernel)、`hstu_attention_group_backward_dispatch.hpp`。
- M5 softmax kernel/pipeline/RunSoftmax:`hstu_attention_bwd_kernel.hpp`(SoftmaxKernel + PRE)、`hstu_attention_with_softmax_bwd_pipeline.hpp`、`hstu_attention_batched_backward_dispatch.hpp`(RunSoftmax 模板)。
- M5 no_group harness softmax 接线:`example_hstu_attention_bwd.cpp`(run_no_group_hstu_bwd,fwd is_training + LSE 转置 + d_dev)。
- group fwd 入口:`hstu_attention_group_forward_bf16`;group fwd params:`HstuAttentionGroupFwdParams`。
- oracle:`reference_group_hstu_attention_bwd`(reference_hstu_attention_bwd.hpp,kUseSoftmax 分支)。
- 参考报告:`/tmp/hstu-bwd-design/M5-done.md`、`M4-done.md`。
