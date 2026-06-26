# M5b group softmax — 独立验证 + 对抗 review 派单 (pane-2 / reviewer)

> 你独立验证 M5b(group softmax)。pane-1 报告完成、对拍全过(10/10 sweep,套件 68/67/0/1)。**别信自述。**
> 你的产出决定 lead 是否 promote。git 基线 = `aced5784`(M5);M5b 改动 = working-tree 未提交。

## 范围 + 改动文件(相对 aced5784)
M5b = 把 softmax 接到 **group** 路,**复用** M5 的 with_softmax pipeline + PRE + POST,**新写** group-softmax kernel + group RunSoftmax + group harness 产 LSE。
- `hstu_attention_bwd_kernel.hpp`(+`HstuAttentionBwdDQDKDVGroupSoftmaxKernel` +359)
- `hstu_attention_group_backward_dispatch.hpp`(+`RunSoftmax` +127,去 softmax throw)
- `hstu_attention_bwd_params.hpp`(GroupBwdParams +`d_ptr`/`nhead_stride_lsed`)
- `example_hstu_attention_bwd.cpp`(group 段产 O+LSE +79)
- `CMakeLists.txt`(BWD target +`hstu_attention_group_forward_bf16.cpp`)
- **不应在 diff**:`hstu_attention_with_softmax_bwd_pipeline.hpp`(M5)、`hstu_attention_no_softmax_bwd_pipeline.hpp`(SiLU)、`hstu_attention_batched_backward_dispatch.hpp`(no_group)——若在,报 lead。

## 任务 A:独立机器验证(权威闸门)
1. 干净重建(防 ninja no-work):`touch` 改动源后 `cmake --build build --target tile_example_hstu_attention_bwd -j128 2>&1 | tee runs/build-M5b-review.log`;0 error。
2. 独立复跑 `python3 test/run_bwd_tests.py`;确认 exit 0/0 FAIL,记 TOTAL/PASS/SKIP(自述 68/67/1)。确认 M1–M5(60 案)全 PASS = 零回归。
3. 自抽 4-5 档 group 对拍(用 ≠ 套件的 g/seqlens,全 `-attn_scale=1.0`),必含:g=2 全异构(`-g_local_lens/-g_context_lens/-g_minfull_lens/-g_attn_scales/-targets` 各组不同)、causal=0+per-batch num_target、g=3 或 g=4。binary=`build/bin/tile_example_hstu_attention_bwd`,group 参数见 M4 harness/M5b-done.md。

## 任务 B:对抗 review(group 特有,逐条核)
读 `M5b-dispatch.md`(规格)+ `M5b-done.md`(自述,核不信)+ diff + `reference_hstu_attention_bwd.hpp` 的 group/softmax 分支。重点:
1. **group LSE/D 四方布局(最大风险,group packed 与 M5 jagged 略不同)**:GPU 侧 [head,ΣL] 连续-seq(group fwd **无 batch_stride_lse**,用 query_start;`nhead_stride_lse=ΣL`)。自己推 fwd 写 / bwd 读 / PRE 写 / reference 转置(`lse_host(0,offset+sq,h)`)四方偏移是否落同一元素。GPU-bwd 与 reference 是否同吃一份 GPU-产 LSE?
2. **group-softmax kernel**:`i_group=i_batch/num_batch_per_group` 取 per-group window/contextual/min_full 对不对?**softmax 不读 scale_p**(不碰 group_attn_scale/group_max_seqlen)——确认没误用?运行时 `window>0` 选 with-local/no-local **softmax** pipeline(不是 SiLU pipeline)?LSE/D window 构造与 M5 一致?
3. **PRE 复用 group**:`is_jagged=true` 路对 group packed 是否正确(token=offset+sq,d_base=i_nhead*ΣL+token)?免-memset 全覆盖成立?
4. **group fwd 产 LSE**:harness 填 `HstuAttentionGroupFwdParams` 字段集对不对(**无 attn_scale/batch_stride_lse 标量**,别照搬 no_group)?`is_training=use_softmax`?转置喂 reference 正确?
5. **CMakeLists**:加 `group_forward_bf16.cpp` 进 bwd target 有无重复符号/链接冲突(它已在 fwd target)?
6. **零回归**:diff 真没碰 M5/SiLU 源;套件 M1–M5 全 PASS;SiLU group 路(no_softmax group kernel)未动。
7. 边界:per-batch num_target supplement、packed 越界、causal=0+target 掩码(P1-1 同类,group 也要)。

## 产出
写 `/tmp/hstu-bwd-design/M5b-review-findings.md`:任务 A 实测(build/suite/抽样数值+exit)、任务 B 逐条 GREEN/问题(文件:行号+复现+期望vs实际)、总评(promote / 需修+blocker)。发现真缺陷如实列、立刻报 lead。
