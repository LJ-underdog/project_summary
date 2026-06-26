# M4 group 模式 — 完成报告 (pane-1 / coder)

状态:**✅ 通过**。SiLU + bf16 + hd64 的 group 路径端到端对拍全 PASS,**含 per-group 异构超参**(window/contextual/min_full/attn_scale/num_target 各组各不相同)。`reject-group-g2` 升级为 8 个 M4 pass case。测试套件整体 exit 0,no_group(batched/jagged/mask)零回归。日期 2026-06-08。

## 关键认知(DESIGN §4.7 D6)
**group = jagged 超集**:同样 dim0=1 token-major packed + cu_seqlens(M3 offset 索引**直接复用**)。差异:
- `alpha` 仍是**全局单标量**(D6,reference:515)。
- `scale_p` + mask 4 参(window/contextual/min_full/max_seqlen)变 **per-group device 指针**,kernel 内 `i_group = i_batch / num_batch_per_group` 取。
- `num_target` 仍 per-batch(`num_targets_ptr[i_batch]`)。
- `scale_p = group_attn_scale[i_group] ? : 1/group_max_seqlen_q[i_group]`。

**难点 & 解法(per-group window → kUseLocal 不能编译期定)**:同一 launch 内不同 group 的 window 可为 0 或 >0,无法把 kUseLocal 固定为编译期常量。bwd pipeline 只把 `FmhaMask::IsMasking` 烘进 Problem(`kUseLocal` 不被 pipeline 直接用,仅 mask 对象方法用),故**镜像 fwd kernel**:group kernel 同时实例化 **with-local + without-local 两条 pipeline**(同 kUseCausal),运行时按 `window>0` 选(`make_..._with_local` vs `_without_local`)。causal 仍是编译期轴(group 间统一,与 fwd 一致)。

## 改动(`example/ck_tile/18_hstu_attention/`)

### 1. params(`hstu_attention_bwd_params.hpp`)
填充 `HstuAttentionGroupBwdParams`:复用 jagged 全部输入/输出/workspace(q/k/v/o/do/dq/dk/dv ptr + seq/nhead stride + seq_q/kv_offsets_ptr + stride_dq_acc/nhead_stride_dq_acc + `total_dq_acc_elems`)+ per-group 指针 `group_{attn_scale,max_seqlen_q,window_size,contextual_seqlen,min_full_attn_seqlen}_ptr` + `num_group`、`num_batch`(→ num_batch_per_group)+ 全局 `alpha`、`num_targets_ptr`、`nhead_ratio_qk`、`kIsDeterministic`。命名对齐 `HstuAttentionGroupFwdParams`。

### 2. group kernel(`hstu_attention_bwd_kernel.hpp`,新增 `HstuAttentionBwdDQDKDVGroupKernel`)
- 模板 `<PipelineLocal, PipelineNoLocal, DKEpi, DVEpi>`;常量取自 PipelineLocal(两 pipeline 仅 Mask 不同,shape 一致),`GetSmemSize=max(两 pipeline,两 epi)`。
- Kargs 带 seq_*_offsets_ptr + 5 个 group 指针 + num_batch_per_group + num_targets_ptr + 全局 alpha + 各 stride。
- operator():jagged offset(token-major)+ per-batch seqlen + **`if(i_n0>=seqlen_kv) return`** early-exit;`i_group` 取 per-group 超参算 scale_p;构造 q/k/v/do/dq_acc 窗口(mask 无关);**运行时 `if(window>0){with_local mask + PipelineLocal} else {without_local + PipelineNoLocal}`**,各自跑完用共享 lambda `write_dkdv` 写 dk/dv。min_full 钳制复刻 reference。
- **no_group 的 `HstuAttentionBwdDQDKDVKernel` 完全未改**(零回归)。

### 3. dispatch / entry / api / cmake
- 新增 `hstu_attention_group_backward_dispatch.hpp`:`ProblemFor<Mask>` 模板 → 两 Problem/Pipeline;`RunSilu` 组 Kernel,`hipMemsetAsync(dq_acc, total_dq_acc_elems)`,grid.x = ceil(max_seqlen_q/kN0),launch + POST convert。`Run` 门控:hdim≠64→throw(M7)、deterministic→throw(M6)、softmax→throw(M5)。
- 新增 `hstu_attention_group_backward_bf16.cpp` entry:`BOOL_SWITCH_2(use_causal,use_softmax)` → `run_group_backward_dispatch`(**直接实例化,无 extern-template instance 文件**,只编 SiLU causal×{local,nolocal})。
- `hstu_attention_api.hpp` 加 `hstu_attention_group_backward_bf16` 声明。
- `CMakeLists.txt` `BWD_INTERFACES_SRCS` += group entry(CMake 自动 reconfigure 已验证编入 + 符号链接)。

### 4. harness(`example_hstu_attention_bwd.cpp`)
- 加 `get_floats_from_string` + args `-g -g_max_seqlens -g_local_lens -g_context_lens -g_minfull_lens -g_attn_scales`。
- 新增 `run_group_hstu_bwd`:校验 num_batch%num_group==0;per-group 数组(supplement 到 num_group)+ per-batch seqlen/num_target(supplement 到 num_batch);算 group_max_seqlens_q(与 reference 同一向量,scale_p fallback 一致);cu_seqlens 前缀和(per-batch seqlen 用 group_contextual[i_group]);packed [1,ΣL,H,D] 分配;**SiLU 路 O 不被用 → 跳过 GPU fwd**;offsets + 5 个 per-group 数组上传 device;喂 GPU group entry + CPU `reference_group_hstu_attention_bwd<...,kUseSoftmax,kUseCausal>`(group 恒 jagged-packed,无 kIsJagged 轴)。
- `main()`:`-g>1` → group 路,否则 no_group。

### 5. 测试套件(`test/run_bwd_tests.py`)
删 `reject-group-g2`,加 8 个 M4 pass:g2-nomask / g2-causal / g2-pergroup-window(16,0)/ g2-pergroup-attnscale(1.0,0.5)/ g2-attnscale-fallback(0,1.0)/ **g2-heterogeneous(全异构)** / g3 / g4-singleton。

## 对拍结果(bf16 阈值 rel≤2e-2/abs≤5e-2,`runs/run-bwd-M4-sweep.log` 8/8 PASS)
| 档 | 结果 |
|---|---|
| g=2 no-mask(b4,per-batch 128/200/96/160)| ✅ |
| g=2 causal | ✅ |
| g=2 **per-group window 16,0**(混 with/without-local,验运行时双 pipeline 分支)| ✅ |
| g=2 per-group attn_scale 1.0,0.5 | ✅ |
| g=2 attn_scale=0 fallback(1/group_max_seqlen_q)| ✅ |
| g=2 **全异构**(window/context/minfull/attn_scale/num_target 各组各不同)| ✅ |
| g=3(window+attn_scale per-group,含 fallback)| ✅ |
| g=4(1 batch/group,全异构)| ✅ |

误差 bf16 舍入级(abs ≤ ~4e-3,max\|ref\| ~3–11)。**全异构档若 i_group 索引/device 指针/默认 scale_p 错会暴露大误差** —— 实测舍入级,确认真按 group 取数。

## 验收对照
- 编译 0 error(`runs/build-bwd-M4.log`)。✅
- group + no-mask / causal / mask 因子;多 group(g=2/3/4)、per-group 不同 max_seqlen/window/contextual/attn_scale、per-batch num_target 对拍 PASS;**至少一档(heterogeneous / g4)per-group 超参全不同**。✅
- 测试套件 `reject-group-g2`→pass;`python3 test/run_bwd_tests.py` **TOTAL 34 / PASS 33 / FAIL 0 / SKIP 1,exit 0**(`runs/test-20260608-025738.log`)。✅
- candidates.jsonl 加 `M4-group`(pass);softmax M5 / fp16+hdim128 M7 仍正确拒绝。✅
- batched/jagged/mask 零回归(suite 内全部 no_group case 仍 PASS)。✅

## 遗留 / 给后续
- **M5 softmax**:group/no_group 现均 throw;需 PRE(D=rowsum(O·dO))+ LSE 读取 + STAGE5 `ds=p*(dp-d)`。group softmax 的 O 在 harness 现被跳过,M5 要接 group GPU fwd 产 O+LSE。
- **cross-attention**:group/jagged 现仅 self(kv offsets==q);cross 需独立 kv offsets + cross mask 构造(mask 成员 M2 已加)。
- **M6 deterministic / M7 fp16+hdim**:group dispatch 已留 throw 门;deterministic 的 dq_acc 多 split + reduce POST、fp16/hdim96-256 待接。
- **perf(M8)**:group 两条 pipeline 都实例化(代码体积↑,运行时只跑一条);grid 按 max_seqlen 开 + GetTileRangeAlongY 保守(M2 遗留)。这些是 perf,不影响正确性。
- 无未解决阻塞点。per-group 取数经"全异构 + g4 singleton"对拍交叉验证。
