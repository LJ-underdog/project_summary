# M6b group deterministic + 修 O1 实现派单 (pane-1 / 主 coder)

> 把 M6 的 deterministic dQ 扩到 **group**(M6 no_group determ × M4/M5b group 结构),并修 O1(group+determ 现静默走 atomic、throw 不可达)。完成后 determinism 覆盖**全模式**。
> 复用 M6 机制 + M5b group 结构,**复用不重写**。铁律:对拍 `-attn_scale=1.0`,no_group/atomic/SiLU/softmax 已 promoted 逻辑零回归。

## 0. 复用清单
| 组件 | 来源 | M6b |
|---|---|---|
| POST `hstu_bwd_reduce_convert_dq_kernel`(固定序 split reduce）| M6 | **直接复用**(mode-agnostic flat reduce) |
| determ 机制(set+split 偏移、固定序 reduce、bit-repro)| M6 | 照搬到 group kernel |
| group per-group 超参 + 双 pipeline + jagged offset | M4/M5b group kernel | 不动结构,只给 dq_acc 加 determ 分叉 |
| **新/改** | — | group 两 kernel dq_acc determ 分叉 + group params split 字段 + group dispatch determ 分支 + group entry BOOL_SWITCH_3 + harness group determ workspace |

## 1. group params(hstu_attention_bwd_params.hpp,`HstuAttentionGroupBwdParams`)
现有 `total_dq_acc_elems`(单份 packed ΣL·H·hdim)+ `kIsDeterministic`,**缺** `split_stride_dq_acc`(+ `num_splits` 若 dispatch 要)。加上:
- `index_t split_stride_dq_acc;`(= 单份 packed 元素数 = total_dq_acc_elems 的"单份"语义;determ 时每 split 一份)。
- `int num_splits;`(atomic=1;determ=ceil(max_seqlen_q/kN0))。
（注意:M5b 时 group 单份叫 total_dq_acc_elems;determ 时整 workspace = 单份 × num_splits。命名上 split_stride = 单份。）

## 2. group 两 kernel dq_acc determ 分叉(hstu_attention_bwd_kernel.hpp)
`HstuAttentionBwdDQDKDVGroupKernel`(SiLU)和 `HstuAttentionBwdDQDKDVGroupSoftmaxKernel`(softmax)的 dq_acc 窗口现恒 atomic。照 M6 no_group kernel 的写法(本文件已有范例,~line 361-372)给两者加**编译期分叉**:
```cpp
constexpr auto mop = kIsDeterministic ? memory_operation_enum::set : memory_operation_enum::atomic_add;
AccDataType* dq_acc_ptr = base(packed: query_start*stride_dq_acc + i_nhead*nhead_stride_dq_acc);
if constexpr(kIsDeterministic) dq_acc_ptr += i_tile_n * kargs.split_stride_dq_acc;
make_naive_tensor_view<global, mop>(dq_acc_ptr, ...);
```
- group 是 packed:单份 base = `query_start*stride_dq_acc + i_nhead*nhead_stride_dq_acc`(M5b 已有);determ 再 `+ i_tile_n*split_stride_dq_acc`。
- Kargs/MakeKargs 给两 group kernel 加 `split_stride_dq_acc`(若没有)。pipeline 的 `if constexpr(kIsDeterministic) store_tile/else update_tile` 自动写 set/atomic(group kernel 调的就是同两条 pipeline)。
- kIsDeterministic 作为 group kernel 的模板参/Problem 轴传进来(随 dispatch)。

## 3. group dispatch(hstu_attention_group_backward_dispatch.hpp)
- RunSilu / RunSoftmax 都对 `kIsDeterministic` 编译期分叉(照 M6 no_group 的 `launch_main_and_post` 思路;group 这两条可各自加,或抽个 group 版 helper):
  - determ:`num_splits = ceil(param.max_seqlen_q / kN0)`;dq_acc workspace = `total_dq_acc_elems(单份) × num_splits`;memset 全量;kernel 用 determ 实例(kIsDeterministic=true 模板);`split_stride_dq_acc = 单份(=total_dq_acc_elems)`传 kargs;POST 用 `hstu_bwd_reduce_convert_dq_kernel`(n=单份, num_splits, split_stride=单份)。
  - atomic(else):现状不变(memset 单份、atomic convert)。
- 去掉 :318-319 的 determ throw(改为真分支)。

## 4. group entry(hstu_attention_group_backward_bf16.cpp)—— 修 O1 根因
`BOOL_SWITCH_2(use_causal, use_softmax)` + 硬编 `false /*kIsDeterministic*/` → 改 **`BOOL_SWITCH_3` 加 `param.kIsDeterministic` 轴**。这样 determ 实例真编出、dispatch determ 分支真可达。**O1 消失**(group+determ 不再静默走 atomic)。
- group 是直接实例化(无 extern instance),改 entry 即可,**不用动 generate_instances.py**(与 no_group 不同)。注意编译体积:group 已是双 pipeline(with/without-local)× {causal} × {softmax};再 ×{determ} 翻倍——可接受,但留意编译时长。

## 5. harness(example_hstu_attention_bwd.cpp,run_group_hstu_bwd 段)
- determ 时:`num_splits = ceil(max_seqlen_q/128)`;group dq_acc 缓冲扩到 `单份(ΣL·H·hdim_qk) × num_splits`;`bp.split_stride_dq_acc = 单份`、`bp.num_splits = num_splits`、`bp.total_dq_acc_elems = 单份`(保持单份语义)。
- atomic(default group)保持现状。

## 6. 验证(两维 + O1,全 `-attn_scale=1.0`)
build `cmake --build build --target tile_example_hstu_attention_bwd -j128`。
1. **正确性**:group determ × {SiLU,softmax} × causal{0,1} × per-group 异构(window/context/minfull/attn_scale/num_target)对拍 reference PASS。含多 split(大 seqlen → num_splits>1)、g{2,3,4}。
2. **可复现性**:group determ 同 case 跑两次 dQ **byte-identical**(md5/cmp)。含多 split。
3. **O1 修复确认**:`-deterministic=1 -g=2` 现在**真走 determ**(可复现),不再静默 atomic;若哪天传不支持组合应有可达 throw(本单 group+determ 已实现,不需 throw)。
4. **零回归**:no_group(M6 determ + atomic)、SiLU/softmax/group atomic、全 M1–M6 套件仍全绿。
日志 `runs/run-M6b-*.log`。

## 7. 落地产出(交 lead)
1. 套件加 group determ × {SiLU,softmax} 案 + group determ repro 断言;`python3 test/run_bwd_tests.py` exit 0。
2. `/tmp/hstu-bwd-design/M6b-done.md`(改了哪些文件、group determ 复用 M6 边界、group packed split workspace、可复现性证据-两次 byte-identical、O1 修复确认、零回归、坑)。
3. `candidates.jsonl` 加 `M6b-group-deterministic`(promoted)。
4. **不动 fwd / no_group 已 promoted 逻辑 / M7**。

## 8. 速查
- M6 no_group determ 范例:hstu_attention_bwd_kernel.hpp(no_group kernel dq_acc ~361-372、POST `hstu_bwd_reduce_convert_dq_kernel`)、batched_backward_dispatch.hpp(`launch_main_and_post`)。
- group kernel/dispatch/entry:hstu_attention_bwd_kernel.hpp(两 Group kernel)、hstu_attention_group_backward_dispatch.hpp(RunSilu :94 / RunSoftmax :191 / throw :318)、hstu_attention_group_backward_bf16.cpp。
- group params:hstu_attention_bwd_params.hpp `HstuAttentionGroupBwdParams`(total_dq_acc_elems :185 / kIsDeterministic :187)。
- 参考:`M6-done.md`、`M5b-done.md`。
