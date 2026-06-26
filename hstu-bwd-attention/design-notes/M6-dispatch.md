# M6 deterministic 实现派单 (pane-1 / 主 coder)

> 目标:dQ 的 **逐位可复现**(bit-reproducible)路径。现 dQ 走 atomic_add(nsplits=1,原子加顺序不定→非可复现);M6 改为 **每 KV-block 写自己的 split 槽(无 atomic)+ POST reduce over splits + convert**。
> 范围:**no_group(batched + jagged)× SiLU + softmax**。group determ = M6b 后续(group dispatch/POST/sizing 另算)。
> 铁律:对拍 CPU reference(`-attn_scale=1.0`),atomic 路 + SiLU/softmax 已 promoted 逻辑零回归。

## 0. 现状(已就位的部分)
- pipeline **已有** determ 分支(no_softmax:523 / with_softmax:561):`if constexpr(kIsDeterministic) store_tile(dq_dram_window, dq_acc) else update_tile(atomic)`。**pipeline 不用改。**
- params **已有** `num_splits` / `split_stride_dq_acc` / `kIsDeterministic`(no_group + group 都有)。
- 缺:kernel 的 dq_acc 窗口(现恒 atomic_add + 无 split 偏移)、POST reduce、dispatch 接线、entry BOOL_SWITCH。

## 1. 机制(每 KV-block 一个 split)
- grid block 沿 KV 维(i_n0 = i_tile_n*kN0)。**split_idx = i_tile_n**(= i_n0/kN0)。`num_splits = ceil(max_seqlen_kv / kN0)`。
- determ 时每个 KV-block 把它对**全体 Q 行**的 dQ 贡献 **store(非 atomic)** 到第 split_idx 份 dq_acc 副本;不同 block 写不同副本→无竞争→可复现。
- dq_acc workspace = `num_splits × (单份 dQ 全量)`;`split_stride_dq_acc = 单份元素数`(= batched: num_batch*seqlen_q*H*hdim_qk 的单份;按现有 atomic 单份布局 × num_splits)。
- POST:对每个 (token, h, d) 求 `Σ_{s<num_splits} dq_acc[s*split_stride + idx]` → convert 成 dq。

## 2. kernel(hstu_attention_bwd_kernel.hpp)
现 dq_acc 窗口(~357)恒 `memory_operation_enum::atomic_add`、偏移只含 nhead/batch。改成 **编译期按 kIsDeterministic 分叉**:
- **determ**:`memory_operation_enum::set`(plain store);base 偏移再 **+ `split_idx * split_stride_dq_acc`**(split_idx = i_tile_n,kernel 已有 i_tile_n / i_n0)。pipeline 的 `store_tile` 分支会写它。
- **atomic**(else):保持现状(atomic_add,无 split)。
- 三个 kernel(no_group SiLU `HstuAttentionBwdDQDKDVKernel`、no_group softmax `...SoftmaxKernel`)都要这个分叉(group kernel 本单不动,M6b)。建议抽个小 helper 或 `if constexpr` 各自加,保持 atomic 路 codegen 不变。
- 注意 jagged:split_stride 是「单份 packed 全量」,determ packed 下 split 副本整块堆叠。

## 3. POST reduce(优先扩展现 custom kernel,别引 FMHA 复杂度)
现 `hstu_bwd_convert_dq_kernel`(atomic 路,单份 convert)。**加一个 determ 变体**(或给它加 num_splits 参数):
- 每线程一个 (flat idx)::`float acc=0; for(s=0..num_splits-1) acc += dq_acc[s*split_stride + idx]; dq[idx]=convert(acc);`
- atomic 路保持 num_splits=1 等价(现行为不变)。
- (FMHA `BlockFmhaBwdConvertQGrad` 是 tile-window reduce 版,可参考但 custom flat-loop 更简单,与现 POST 一致。)

## 4. dispatch(hstu_attention_batched_backward_dispatch.hpp)
- 现两条 Run(RunSilu/RunSoftmax)都对 `kIsDeterministic` 编译期分叉:
  - determ:`num_splits = ceil(grid_seqlen_kv / kN0)`;dq_acc 尺寸 = 单份 × num_splits;memset 全量;kernel 用 **set 路实例**(kIsDeterministic=true 模板);POST 用 reduce 变体(传 num_splits/split_stride)。
  - atomic(else):现状不变。
- `split_stride_dq_acc` 在 dispatch 算好塞进 kargs(= 单份元素数)。
- grid 不变(仍沿 KV 维;split_idx 在 kernel 内由 i_tile_n 得)。
- 去掉 determ throw(:217 一带)。

## 5. entry(hstu_attention_no_group_backward_bf16.cpp)
现 `run_batched_backward_dispatch<..., false /*kIsDeterministic*/, ...>` 硬编 false。改成 **BOOL_SWITCH on param.kIsDeterministic**(与 use_causal/use_softmax 并列)→ determ 实例才会编出来。注意 instance 数翻倍(determ × {causal} × {softmax}),generate_instances.py 若按矩阵生成需同步加 determ 轴(no_group backward instances);若 instance 是直接实例化则改 entry 即可。**先确认 no_group backward 用的是 extern instance 还是直接实例化**(M5 softmax 那两个 `*softmax_true*` instance 是怎么来的),据此决定改 generate_instances.py 还是仅改 entry。

## 6. harness(example_hstu_attention_bwd.cpp,no_group 段)
- `-deterministic` flag 已有(`bp.kIsDeterministic`)。determ 时:
  - `num_splits = ceil(max_seqlen_kv / kN0)`(kN0 见 tile 设置,hd64 preset kN0=128;与 dispatch 一致,可在 harness 算或固定查 tile）。
  - dq_acc 设备缓冲扩到 `单份 × num_splits`;`bp.split_stride_dq_acc = 单份元素数`;`bp.num_splits = num_splits`。
  - atomic(default)保持现状(num_splits=1,split_stride=0)。

## 7. 验证(两个维度,都要)
build 后 binary 同前。**全 `-attn_scale=1.0`**。
1. **正确性**:determ dQ/dK/dV 仍对拍 reference PASS(SiLU + softmax × batched + jagged × 几个 mask 因子 × causal{0,1})。
2. **可复现性(M6 的核心)**:同一 case `-deterministic=1` **跑两次,dQ 逐位相等**(byte-identical);对照 atomic 路两次可能不等(浮点 atomic 顺序)。写个小脚本/binary 选项 dump dQ 两次比对,或在 harness 加一个 determ-repro 检查。**这是 M6 的验收点,务必显式验。**
3. 套件:`test/run_bwd_tests.py` 把 `skip-deterministic` 升级为真断言(正确性 + 可复现);加 determ × {SiLU,softmax} × {batched,jagged} 案。整体 exit 0。

## 8. 落地产出(交 lead)
1. 套件升级 + 全绿 exit 0。
2. `/tmp/hstu-bwd-design/M6-done.md`(改了哪些文件、split 机制、POST reduce、可复现性验证证据-两次 dQ byte-identical、atomic 路零回归、instance 处理、坑)。
3. `candidates.jsonl` 加 `M6-deterministic`(promoted)。
4. 不动 fwd / group(M6b)/ M5·SiLU 的 atomic 路逻辑。

## 9. 速查
- pipeline determ 分支:no_softmax:523 / with_softmax:561(已就位)。
- kernel dq_acc 窗口:hstu_attention_bwd_kernel.hpp ~357(atomic,要加 determ 分叉);POST custom kernel 同文件。
- FMHA reduce 参考:`include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_convert_dq.hpp`(BlockFmhaBwdConvertQGrad)。
- params determ 字段:bwd_params.hpp:102-106。
- dispatch determ throw:batched_backward_dispatch.hpp:~217。
- entry:hstu_attention_no_group_backward_bf16.cpp。
