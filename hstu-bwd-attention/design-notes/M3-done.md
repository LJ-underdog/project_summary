# M3 jagged(变长 packed)— 完成报告 (pane-1 / coder)

状态:**✅ 通过**。SiLU + bf16 + hd64 在 **jagged(变长)+ (no-mask 与 mask 均可)** 下端到端对拍全 PASS(attn_scale=1.0 与 default scale_p)。`reject-jagged` 升级为 8 个 M3 pass case。测试套件整体 exit 0。日期 2026-06-08。

## 设计要点(与 DESIGN §3.4 / §4.6 一致)
jagged = **dim0=1 的 token-major packed `[1, ΣL, H, D]`** + cu_seqlens(`seq_q/kv_offsets`,size num_batch+1)。
- per-(batch) base offset = `seq_*_offsets_ptr[i_batch] * seq_stride`(token-major,**复刻 fwd kernel 的 jagged 分支**);
- per-batch `seqlen_q/seqlen_kv = offsets[b+1] - offsets[b]`;
- 自注意力:`seq_kv_offsets == seq_q_offsets`(dispatch 传 `is_cross ? kv : q`,M3 只 self)。
- **同一 SiLU MAIN kernel 处理 batched/jagged**,仅靠运行时 `is_jagged` 分支选 base offset/seqlen——**不新增 kernel 实例**(符合"batched 路保持不变 + 运行时分支")。

## 改动(`example/ck_tile/18_hstu_attention/`)

### 1. kernel(`hstu_attention_bwd_kernel.hpp`)
- Kargs / MakeKargs 新增 `bool is_jagged`、`const void* seq_q_offsets_ptr`、`const void* seq_kv_offsets_ptr`。
- `operator()`:把原"batched offsets"块改为 `if(is_jagged){…}else{…}`。
  - jagged:`query_start=q_off[i_batch]`、`key_start=kv_off[i_batch]`,各 `batch_offset_* = start * kargs.stride_*`(q/do/dq_acc 用 query_start;k/v/dk/dv 用 key_start);并 **覆盖** `kargs.seqlen_q/seqlen_kv` 为 per-batch 值(后续 window/mask/OOB 全部复用,无需再分支)。
  - **early-exit**:`if(i_n0 >= kargs.seqlen_kv) return;`(grid.x 按 max 尺寸开,短 batch 的越界 KV-tile 直接退出)。batched grid 精确,该分支永不触发,**batched 行为零变化**。
- GridSize/POST convert 未改(POST 对整段连续 buffer elementwise cast,packed 同样成立)。

### 2. dispatch(`hstu_attention_batched_backward_dispatch.hpp`)
- 删除 `if(is_jagged) throw`。
- MakeKargs 传入 `is_jagged` + 两个 offsets ptr(self:kv 传 q)。
- `dq_acc` 清零字节数与 POST 元素数:jagged 用 `batch_stride_dq_acc`(=dim0 stride=ΣL*H*hdim 全量),batched 用 `num_batch*batch_stride_dq_acc`。
- grid.x 的 seqlen_kv:jagged 用 `param.max_seqlen_q`(self 时 == max_seqlen_kv),batched 用 `param.seqlen_kv`。

### 3. harness(`example_hstu_attention_bwd.cpp`)
- 新增 `-jagged` 开关;`-seqlens` 在 jagged 下接受 **per-batch 逗号列表**(supplement 到 num_batch)。
- jagged:`batches_for_alloc=1`,Q/K/V/O/dO/dQ/dK/dV/dq_acc 全部按 `[1, ΣL, H, D]` packed 分配;构造 `seq_offsets_q`(前缀和,per-batch seqlen = `seqlens[i] + num_target[i] + contextual`),self 下 kv==q;`phy_seqlen_kv=phy_seqlen_q`。
- `max_seqlen_q = max_uih + max_target + contextual`(GPU 与 reference 同源,保证 scale_p 一致)。
- offsets 上传 device,喂 **GPU fwd 参数**(产 O,SiLU 不用但保持 jagged 正确,避免 packed 上越界)、**GPU bwd 参数**、**CPU `reference_no_group_hstu_attention_bwd<…,kIsJagged=true,…>`**(用 `BOOL_SWITCH_3(is_jagged,…)`;batched 传 empty_offsets)。
- 全 buffer 对拍合法:packed 无 padding(ΣL 精确),每个 token 都被 GPU 与 reference 写到。

### 4. 测试套件(`test/run_bwd_tests.py`)
- 删 `reject-jagged`,加 8 个 `pass` case(M3):nomask-varying / causal-varying / causal-window / causal-numtarget-perbatch / 5factor-combo / single-batch / large-spread(512,32,256)/ tiny-seqlens(1,128,7)。
- 其余 reject 不变(softmax M5、group M4、fp16/hdim128 M7)。

## 对拍结果(attn_scale=1.0,bf16 阈值 rel≤2e-2/abs≤5e-2)
`runs/run-bwd-M3-sweep.log`(10/10 PASS,exit 0):

| 档 | 结果 |
|---|---|
| jagged no-mask,per-batch 128/200/96(含非整除)| ✅ |
| jagged causal | ✅ |
| jagged causal + window(b4 256/200/128/96)| ✅ |
| jagged causal + contextual / + minfull | ✅ |
| jagged causal + num_target(per-batch 8/24/16)| ✅ |
| jagged 5 因子全组合(per-batch varying)| ✅ |
| jagged single batch(seq300)| ✅ |
| jagged 大段差(512/32/256,短 batch early-exit)| ✅ |
| jagged tiny(1,128,7)| ✅ |

误差 bf16 舍入级(abs ≤ ~3e-2,max\|ref\| ~5–11);default scale_p 的 jagged 亦 PASS(dQ/dK 逐位 0)。

## 验收对照
- 编译 0 error(`runs/build-bwd-M3.log`)。✅
- jagged + no-mask / causal / 多 mask 因子 + 多组 per-batch 不同 seqlen(非整除/单 batch/段差大/tiny)对拍 PASS。✅
- 测试套件 `reject-jagged` → pass(+jagged×mask 组合);`python3 test/run_bwd_tests.py` **TOTAL 27 / PASS 26 / FAIL 0 / SKIP 1,exit 0**(`runs/test-20260608-020631.log`)。✅
- candidates.jsonl 加 `M3-jagged`(pass);其余 reject(group M4 / softmax M5 / fp16+hdim M7)仍正确拒绝。✅
- batched/mask 不回归(suite 内全部 batched/mask case 仍 PASS)。✅

## 遗留 / 给后续
- **M4 group**:reference_group_… 已就绪;需接 `HstuAttentionGroupBwdParams`(目前空)+ group dispatch(per-group window/contextual/minfull/max_seqlen/attn_scale 设备指针,`i_group=i_batch/num_batch_per_group`)。group 是 jagged 的超集(per-group 超参),M3 的 jagged offset 索引可直接复用。
- **cross-attention jagged**:kernel/harness 现仅 self(kv offsets==q);cross 需独立 kv offsets + cross mask 构造(mask 成员 M2 已加),待对应里程碑。
- perf(M8):jagged 短 batch 的 early-exit 已避免越界 KV-tile 的 atomic,但 grid 仍按 max 开;GetTileRangeAlongY 仍保守全 Q 扫描(M2 遗留)。
- 无未解决阻塞点。offset 计算 / reference jagged 分支 / packed stride 三处均经"大段差 + 非整除 + tiny"对拍交叉验证(若 offset 错会暴露大误差,实测 bf16 舍入级)。
