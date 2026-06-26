# M5b group softmax — 完成报告 (pane-1 / coder)

状态:**✅ 通过**。softmax + bf16 + hd64 的 **group** 路径端到端对拍全 PASS(attn_scale=1.0,softmax × causal{0,1} × g{2,3,4} × per-group 异构超参)。M5 no_group + SiLU 全路径逐字不变(零回归)。测试套件整体 exit 0。日期 2026-06-08。

## 0. 基线确认
动手前 `run_bwd_tests.py` = 60 案全绿(59 pass/1 skip,exit 0),M0–M5 无回归。

## 1. 复用边界(reuse, not rewrite)
| 组件 | 来源 | M5b 处理 |
|---|---|---|
| MAIN softmax pipeline | M5 `hstu_attention_with_softmax_bwd_pipeline.hpp` | **直接复用**(mode-agnostic:只吃 mask/alpha/LSE/D window) |
| PRE `hstu_bwd_dot_do_o_kernel` | M5 | **直接复用**(`is_jagged=true`;group packed [head,ΣL] = jagged 同款) |
| POST `hstu_bwd_convert_dq_kernel` | M1 | 直接复用 |
| per-group 超参 + 双 pipeline 选择 | M4 group kernel | 搬进新 group-softmax kernel |
| **新写** | — | group-softmax kernel + group dispatch RunSoftmax + group harness 产 LSE |

## 2. 改了哪些文件
| 文件 | 改动 |
|---|---|
| `hstu_attention_bwd_params.hpp` | `HstuAttentionGroupBwdParams` 加 `d_ptr` + `nhead_stride_lsed`(group packed LSE/D [head,ΣL]) |
| `hstu_attention_bwd_kernel.hpp` | 新增 `HstuAttentionBwdDQDKDVGroupSoftmaxKernel`(= M4 group kernel 骨架 + M5 LSE/D window + softmax pipeline 调用,去 scale_p) |
| `hstu_attention_group_backward_dispatch.hpp` | include with_softmax pipeline;新增 `RunSoftmax`(PRE→memset→MAIN→POST);`kUseSoftmax` 分支去 throw 改调 RunSoftmax;SiLU 分支逐字不变 |
| `example_hstu_attention_bwd.cpp` | group 段:softmax 时跑 GPU group fwd 产 O+LSE + LSE 转置喂 reference + 分配 o_dev/lse_dev/d_dev + bp.o/lse/d_ptr/nhead_stride_lsed;SiLU 段原样(跳 fwd) |
| `CMakeLists.txt` | BWD_INTERFACES_SRCS 加 `hstu_attention_group_forward_bf16.cpp`(harness 现调 group fwd 产 LSE,需链接其 entry) |
| **fwd 逻辑零改动**(只在 group harness 切 is_training);M5 no_group/SiLU 文件逻辑未碰 |

## 3. group-softmax kernel(M4×M5 合流点)
- per-(batch) jagged offset + 早退 `if(i_n0>=seqlen_kv) return`(M4 同款,group 恒 packed)。
- `i_group=i_batch/num_batch_per_group` 取 per-group **window/contextual/min_full**(softmax 不用 scale_p,故不读 group_attn_scale/group_max_seqlen)。
- **LSE/D 定位**(group packed):base = `i_nhead*nhead_stride_lsed + query_start`,seq stride=1(1-D packed,= M5 jagged 路一字不差)。`nhead_stride_lsed=ΣL`。
- 运行时 `if(window>0)` 选 with-local / no-local **softmax** pipeline(同 M4 的双 pipeline,但实例是 M5 softmax pipeline),各自调 `(q,k,v,do, lse_win, d_win, dq, mask, alpha, smem)`——**无 scale_p**。
- 收尾 `write_dkdv` lambda 共享(mask 无关)。

## 4. LSE/D group packed 布局
- GPU 侧统一 **[head, ΣL] 连续-seq**(packed):group fwd `seq_stride_lse=1, nhead_stride_lse=ΣL`(group 无 batch_stride_lse,fwd 用 query_start 定位);group bwd `nhead_stride_lsed=ΣL`,token 基址=query_start。fwd/bwd **共用同一 lse_dev**。
- reference 要 `[1, ΣL, head]`(`lse_batch_seq_nhead(0, offset+sq, head)`);harness 转置 `lse_host(0,s,h)=lse_flat[h*ΣL+s]`。
- **GPU-产 LSE 同时喂 GPU-bwd 与 reference**,两侧 P 一致(与 M5 同策略)。
- D(PRE 产)group packed [head,ΣL],与 MAIN 读 D 同布局;PRE `is_jagged=true` 用 token=offset+sq、d_base=i_nhead*ΣL+token。

## 5. 逐档对拍(attn_scale=1.0,bf16 阈值 rel≤2e-2/abs≤5e-2)
`runs/run-M5b-sweep.log`(**10/10 PASS**):

| 档 | 结果 |
|---|---|
| g=2 causal / g=2 causal=0 no-mask | ✅ |
| g=2 per-group window(16,0 混 with/without-local)| ✅ |
| g=2 per-group attn_scale(1.0,0.5)/ attn_scale=0(softmax 忽略,无害)| ✅ |
| g=2 **全异构**(window/context/minfull/attn_scale/num_target 各组各不同)| ✅ |
| g=2 per-batch num_target / causal=0+num_target(P1-1 同类)| ✅ |
| g=3(per-group window+attn_scale)/ g=4(1 batch/group,全异构)| ✅ |

误差 bf16 舍入级:dQ max_abs ≤ 1.4e-2(vs max|ref| ~0.3–10.75,rel ~1e-3;且 < atol 5e-2)、dV ≤ 4e-3。全异构档证明真按 i_group 取 per-group 超参。

## 6. 验收对照
- 编译 0 error(`runs/build-M5b.log`)。✅
- 对拍 PASS:softmax × causal{0,1} × g{2,3,4} × per-group 异构(window/context/minfull/attn_scale/num_target)+ 全异构 + causal=0+target。✅
- 测试套件加 8 个 M5b 交叉案;`python3 test/run_bwd_tests.py` → **TOTAL 68 / PASS 67 / FAIL 0 / SKIP 1,exit 0**(`runs/test-20260608-094807.log`)。✅
- candidates.jsonl 加 `M5b-group-softmax`(promoted)。✅
- **零回归**:group-softmax kernel/RunSoftmax 为独立新增;M5 no_group softmax + SiLU(no_group/group)pipeline/kernel/dispatch 逻辑一字未改;套件内全部 60 个 M1–M5 案仍 PASS。✅

## 7. 遇到的坑
- 唯一链接坑:group harness 现要调 `hstu_attention_group_forward_bf16` 产 LSE,但该 entry 之前未在 bwd target 的 BWD_INTERFACES_SRCS 里 → 加上(其 instances 已被 `*forward_bf16*` glob 覆盖)。
- group fwd params(`HstuAttentionGroupFwdParams`)**无 attn_scale / batch_stride_lse 标量**(group 用 group_*_ptr + packed),按其字段集填,勿照搬 no_group。

## 8. 遗留 / 给后续
- **cross-attention softmax**:no_group/group 现均仅 self;cross 待对应里程碑。
- M6 deterministic / M7 fp16+hdim / M8 perf 不变(group softmax 双 pipeline 实例化体积是已知 perf 取舍)。
- 无未解决阻塞点。
