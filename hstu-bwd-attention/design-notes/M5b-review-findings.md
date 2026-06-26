# M5b group softmax — 独立验证 + 对抗 review 结论 (pane-2 / reviewer)

**总评:✅ PROMOTE。** 任务 A(干净重建 + 独立复跑套件 + 自抽 5 档 off-suite 对拍)全过;
任务 B(7 条 group 特有对抗点)全 GREEN。无 blocker,未发现缺陷。
git 基线 `aced5784`;M5b = working-tree 未提交。日期 2026-06-08。

---

## 任务 A:独立机器验证(权威闸门)

### A1 干净重建(防 ninja no-work)
`touch` 全部改动源(含 `hstu_attention_group_backward_bf16.cpp` TU)后:
`cmake --build build --target tile_example_hstu_attention_bwd -j128`
→ **`BUILD_EXIT=0`**,`[8/11] Linking … bin/tile_example_hstu_attention_bwd` 成功。
日志 `runs/build-M5b-review.log`。CMakeLists 加 `group_forward_bf16.cpp` 进 bwd target
**未产生重复符号/链接冲突**(链接通过即铁证 → B5 GREEN)。

### A2 独立复跑套件(零回归闸门)
`python3 test/run_bwd_tests.py` → **`SUITE_EXIT=0`**
**TOTAL 68 / PASSED 67 / FAILED 0 / SKIPPED 1**(与自述 68/67/0/1 一致)。
- 8 个新 `pass-gsm-*`(M5b)全 PASS。
- 全部 M1–M5(60 案,含 6 个 no_group softmax `pass-sm-*`)仍 PASS = **零回归实测确认**。
- 1 SKIP = `skip-deterministic`(M6,符合预期);2 reject(fp16 / hdim128)按预期 reject。
日志 `/tmp/hstu-bwd-design/M5b-review-suite.log` → `runs/test-20260608-095621.log`。

### A3 自抽 5 档 off-suite 对拍(全 `-attn_scale=1.0`,seqlens/g 均 ≠ 套件)
binary=`build/bin/tile_example_hstu_attention_bwd`。全部 **[PASS] dQ/dK/dV**:

| 档 | 配置 | 结果 | 误差(max_abs vs max\|ref\|) |
|---|---|---|---|
| A1 | g=2 **全异构** causal=1 (seqlens 80,176,112,240; window/ctx/minfull/scale/target 各组异) | ✅ | dQ 2.4e-4 / .42, dV 2.0e-3 / 1.84 |
| A2 | g=2 **causal=0 + per-batch num_target**(P1-1 同类) | ✅ | bf16 级 |
| A3 | g=3 per-group window+scale causal=1 (seqlens 6 不同) | ✅ | bf16 级 |
| A4 | g=4 singleton 全异构 causal=1 | ✅ | dQ 7.6e-6 / .41, dV 4.9e-4 / 1.98 |
| A5 | g=2 全异构 **causal=0 + target** | ✅ | bf16 级 |

误差均为 bf16 舍入级(< atol)。**全异构档(A1/A3/A4)对 per-group-aware reference 通过
= `i_group` 索引真被行使**(误用恒 group0 必 FAIL)。含全异构 + causal=0+target(A2/A5)。

---

## 任务 B:对抗 review(group 特有,逐条核)

### B1 — group LSE/D 四方 packed 布局(最大风险)→ **GREEN**
group 中 `phy_seqlen_q = Σ batch_seqlen = ΣL`、`batches_for_alloc=1`
→ `nhead_stride_lsed = phy_seqlen_q = ΣL`(确属 packed 总 token 数,非 per-batch)。
四方均落 **同一元素 `[head*ΣL + token]`**(token=offset+sq):
- GPU-fwd 写:`seq_stride_lse=1, nhead_stride_lse=ΣL`(harness:347/fwd 段)。
- GPU-bwd 读:base=`i_nhead*nhead_stride_lsed + query_start`,seq stride 1(kernel:1432/1450)。
- harness 转置:`lse_host(0,s,h)=lse_flat[h*ΣL+s]`(harness fwd 段)。
- reference 读:`lse_batch_seq_nhead(0, seq_q_offsets[i_batch]+sq, i_head)`(reference:708)。
**GPU-产 LSE 同喂 GPU-bwd(lse_dev 直传)与 reference(同 lse_flat 转置副本)** — 同源。
D:reference 自算 `D[sq]=dO·O`(reference:767)、GPU 用 PRE 产 D,**二者吃同一份 O**
(GPU-fwd O → o_host;PRE 读同 o_dev),数学同,仅 bf16 舍入差。

### B2 — group-softmax kernel → **GREEN**
- `i_group=i_batch/num_batch_per_group` 取 per-group window/contextual/min_full(kernel:1411/1420-1427)✓。
- **softmax 不读 scale_p**:`group_attn_scale_ptr`/`group_max_seqlen_q_ptr` 在 Kargs 但 kernel
  从不解引用;pipeline 调用无 scale_p 形参(kernel:1546/1556),仅传 `kargs.alpha`(global)。✓
- 运行时 `if(window>0)` 选 **with-softmax** PipelineLocal / 否则 PipelineNoLocal(kernel:1536)——
  实例来自 `HstuAttentionWithSoftmaxBwdDQDKDVPipelineKRKTRVR`(dispatch:205-209),**非 SiLU pipeline**。✓
- LSE/D window 构造、`eff_min_full` clamp、mask helper 调用与 M5 softmax kernel + M4 group kernel
  逐字一致。**`window_size` 填入 mask 的 max_attn_len 槽位**与 M4 group SiLU kernel(:1184)同 —
  group 路该槽位本就是 per-group window,非 bug(no_group 才用独立 `max_attn_len` 标量)。

### B3 — PRE 复用 group(`is_jagged=true`)→ **GREEN**
`hstu_bwd_dot_do_o_kernel` jagged 分支:`token=q_start+sq`,
`o_base=token*seq_stride + i_nhead*nhead_stride`,`d_base=i_nhead*d_nhead_stride+token`(seq stride 1)
(kernel:1599-1606)——与 MAIN 读 D 偏移**逐字相同**。dispatch 传 `batch_stride=0`(packed)、
`d_nhead_stride=nhead_stride_lsed=ΣL`(dispatch:236-237)。packed 无 batch 间隙(token 连续覆盖
`[q_start, q_start+seqlen)`),**免-memset 全覆盖成立**;`sq>=seqlen_q` 越界早退。

### B4 — group fwd 产 LSE(`HstuAttentionGroupFwdParams`)→ **GREEN**
harness fwd 段填字段集:`is_training=true`(softmax)、`use_softmax=true`、`scale_s=scale_s`、
`seq_stride_lse=1`、`nhead_stride_lse=ΣL`;**无 attn_scale 标量、无 batch_stride_lse**(group
用 group_*_ptr + query_start 定位)——未照搬 no_group。字段名经编译通过验证存在。转置喂 reference
正确(见 B1)。`o_dev.FromDevice`/`lse_dev.FromDevice` 落 o_host/lse_host。

### B5 — CMakeLists 重复符号 → **GREEN**
`group_forward_bf16.cpp` 仅出现于 `BWD_INTERFACES_SRCS` 一处;`FWD_BF16_INSTANCE_SRCS` glob 是
`instances/*forward_bf16*.cpp`(子目录,不含根目录此文件)→ bwd target 内无重复 TU。**链接成功**(A1)即铁证。

### B6 — 零回归 → **GREEN**
- diff 仅触 5 文件(CMakeLists/harness/kernel.hpp/params.hpp/group_dispatch.hpp),与派单声明完全一致。
- **三个禁改文件 byte-identical 于 aced5784**:`hstu_attention_with_softmax_bwd_pipeline.hpp`(M5)、
  `hstu_attention_no_softmax_bwd_pipeline.hpp`(SiLU)、`hstu_attention_batched_backward_dispatch.hpp`
  (no_group)——`git diff --stat` 空。
- group-softmax kernel/RunSoftmax 为纯新增;SiLU group 分支(`RunSilu`/`else`)逐字不变(dispatch:321)。
- 套件 60 个 M1–M5 案全 PASS(A2)。

### B7 — 边界 → **GREEN**
- per-batch num_target supplement:A1/A2/A5 用非均匀 `-targets`(含 0)且 PASS;reference 同读。
- causal=0 + target(P1-1 同类):A2/A5 显式覆盖,PASS。
- packed 越界:PRE `sq>=seqlen_q` 早退;MAIN `i_n0>=seqlen_kv` 早退(kernel:1408)。

---

## 结论
全部任务 A 实测 + 任务 B 七条对抗点 GREEN,零回归实测,off-suite 全异构对拍通过证 i_group 真索引。
**建议 lead:promote M5b。** 无需修复,无 blocker。
