# M5b group softmax — 文档级独立 review 结论 (pane-3 / 第三方视角)

> 角色:pane-3,边写 HTML 讲义边以全新视角核 silent-wrong(coder=pane-1、reviewer=pane-2 已过)。
> 只读源码,基线 git `dc8c6b21`(相对 M5 `aced5784` 的 5 文件改动)。
> **结论先行:GREEN,无 blocker,未发现 pane-2 之外的真问题。**

---

## 我独立重核过的点(逐条 GREEN)

### 1. group LSE/D 四方 packed 布局(最大风险)— GREEN(自推偏移)
统一 `[head, ΣL]` 连续-seq(packed,`nhead_stride_lsed = ΣL = phy_seqlen_q`,seq stride 1;**无 batch 维**,group 恒 packed `batches_for_alloc=1`)。四方落同一元素 `[i_nhead·ΣL + (query_start+sq)]`:

| 方 | 代码位置 | 地址 |
|---|---|---|
| **fwd 写 LSE** | fwd_kernel:710/724/936(jagged 分支) | `i_nhead·nhead_stride_lse(=ΣL) + query_start`(`seq_stride_lse=1`)+ s |
| **bwd 读 LSE/D** | kernel:1432/1448-1453 | `i_nhead·nhead_stride_lsed(=ΣL) + query_start`(batch_offset_lsed)+ s(packed view,stride 1) |
| **PRE 写 D** | kernel jagged 分支 + dispatch:240-242 | `i_nhead·d_nhead_stride(=nhead_stride_lsed=ΣL) + token(=query_start+sq)`,`d_batch_stride=0` |
| **reference 转置读** | harness:826-827 / ref:709 | `lse_host(0, query_start+sq, h) = lse_flat[h·ΣL + s]` |

四方逐项相等。PRE 写与 MAIN 读共用 `param.nhead_stride_lsed`(dispatch:241 vs MakeKargs:282),构造上不可能错位。
**与 M5 jagged 路一字不差**(M5 jagged 也是 `batches_for_alloc=1` / packed),group 只是把 jagged 复用过来。✓

### 2. i_group 真取 per-group(非恒 group0)— GREEN
kernel:1411-1412 `i_group = i_batch / num_batch_per_group`(readfirstlane),读 `group_window/contextual/min_full[i_group]`(1413-1418)。reference:579 同式 `i_group = i_batch/num_batch_per_group`、读 per-group(585-591)。**全异构档(sweep + pane-2 A1/A3/A4)对 per-group-aware reference 通过 = i_group 真被行使**(恒 group0 必 FAIL)。✓

### 3. softmax 确无误用 scale_p — GREEN
`group_attn_scale_ptr` / `group_max_seqlen_q_ptr` 在 Kargs(1257-1258)且 dispatch 传入(257-258),但 **kernel 从不解引用** —— softmax 只读 window/contextual/min_full。pipeline 调用(1544-1558)只传 `kargs.alpha`(global),**形参无 scale_p**(softmax pipeline operator() 本就无 scale_p 形参)。scale_p 被 LSE 取代。✓

### 4. window>0 选的是 softmax 而非 SiLU pipeline — GREEN
dispatch RunSoftmax 的 `PipelineLocal/NoLocal = HstuAttentionWithSoftmaxBwdDQDKDVPipelineKRKTRVR<ProblemFor<...>>`(dispatch:201-204,**M5 softmax pipeline**,非 SiLU 的 `HstuAttentionBwdDQDKDVPipelineKRKTRVR`)。kernel:1536 `if(window_size>0)` → PipelineLocal,否则 PipelineNoLocal,二者均为 softmax 实例。`ProblemFor` 的 LSEDataType/DDataType = `TC::CompDataType`(真,dispatch:73/75)。✓

### 5. PRE group jagged 路偏移 — GREEN
dispatch RunSoftmax 传 `is_jagged=true`(232)、`o_batch_stride=0`、`d_nhead_stride=nhead_stride_lsed`、`d_batch_stride=0`(238-242)。PRE jagged 分支 `token=q_start+sq`、`o_base=token·o_seq_stride + i_nhead·o_nhead_stride`、`d_base=i_nhead·d_nhead_stride + token`——与 MAIN 读 D 逐字相同。packed token 连续覆盖 `[q_start, q_start+seqlen)`,免-memset 全覆盖成立;`if(sq>=seqlen_q) return` 越界早退。✓

### 6. CMake 无重复符号 — GREEN(自查 + 链接铁证)
`hstu_attention_group_forward_bf16.cpp` 出现两处但属**两个不同 target**:
- 行 6 `INTERFACES_SRCS` → **fwd 可执行** `tile_example_hstu_attention`(行 7-9)。
- 行 38 `BWD_INTERFACES_SRCS` → **bwd 可执行** `tile_example_hstu_attention_bwd`(行 39-41)。
两 target 各自独立链接,跨 target 不冲突。bwd target 内该 entry **仅列一次**(行 38);`FWD_BF16_INSTANCE_SRCS` glob = `instances/*forward_bf16*.cpp`(子目录的 instance TU,符号不同,非根目录此 entry)→ bwd target 内无重复 entry 符号。**干净重建链接成功(pane-2 A1)即铁证**。✓

### 7. 零回归 — GREEN(git 实测)
`git diff --stat aced5784 -- <3 禁改文件>` = **空**,即 `hstu_attention_with_softmax_bwd_pipeline.hpp`(M5)、`hstu_attention_no_softmax_bwd_pipeline.hpp`(SiLU)、`hstu_attention_batched_backward_dispatch.hpp`(no_group)**byte-identical 于 aced5784**。group-softmax kernel/RunSoftmax 为纯新增;SiLU group `RunSilu`/`else` 分支(dispatch:323)未碰。套件 60 个 M1–M5 案全 PASS(pane-2 A2)。✓

---

## 非阻塞观察(沿用 M5,非 M5b 引入)
1. **α=1 验证包络**:铁律 `-attn_scale=1.0` 使 α 缩放数值上仅以 α=1 验证;代码层与 reference 逐项对称一致,贯穿性约束。
2. **LSE 数值盲区**:GPU-bwd 与 reference 共吃同一份 GPU-产 LSE,对拍结构上无法独立验 LSE 数值;由 fwd 里程碑兜底 + 写读自洽闭合。pane-2 B1 已记录,我复核同意,无新增。

---

## 总评
**GREEN,可保持 promoted。** 7 条 group 特有点(四方偏移自推 / i_group 真索引 / softmax 不用 scale_p / window>0 选 softmax 双 pipeline / PRE group 偏移 / CMake 无重复符号 / 零回归)逐条独立重核,与 reference + M4/M5 蓝本逐字对齐,git 实测三禁改文件 byte-identical。未发现 pane-2 之外的 silent-wrong。HTML 讲义据此如实写出,无"讲不圆"之处。
