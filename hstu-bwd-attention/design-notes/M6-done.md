# M6 deterministic(dQ 逐位可复现)— 完成报告 (pane-1 / coder)

状态:**✅ 通过**(两维验收全过)。no_group(batched + jagged)× SiLU + softmax 的 deterministic 路:① 对拍 reference 正确性 PASS;② **同 case 跑两次 dQ 逐位相等**(byte-identical,M6 核心)。atomic/SiLU/softmax/group 已 promoted 逻辑零回归。group determ = M6b(未做)。日期 2026-06-09。

## 0. 基线确认
动手前 `run_bwd_tests.py` = 68 案全绿(67 pass/1 skip,exit 0)。

## 1. 机制(每 KV-block 一个 split,无 atomic)
- grid block 沿 KV 维,`split_idx = i_tile_n`(= i_n0/kN0);`num_splits = ceil(grid_seqlen_kv/kN0) = grid.x`。
- determ:每个 KV-block 把它对**全体 Q 行**的 dQ 贡献 **plain store(memory_operation_enum::set)** 到第 split_idx 份 dq_acc 副本(base += `i_tile_n*split_stride_dq_acc`)。不同 block 写不同副本 → 无竞争 → 顺序无关 → 可复现。
- dq_acc workspace = `num_splits × 单份`;`split_stride_dq_acc = 单份元素数`。
- POST:固定顺序 `Σ_{s=0..num_splits-1} dq_acc[s*split_stride + idx]` → convert → dq。固定求和序 ⇒ bit-reproducible。

## 2. 改了哪些文件
| 文件 | 改动 |
|---|---|
| `hstu_attention_bwd_kernel.hpp` | SiLU + softmax kernel:Kargs/MakeKargs 加 `split_stride_dq_acc`;dq_acc 窗口改**编译期分叉**——determ `mop=set` + base `+= i_tile_n*split_stride`,atomic 保持 `atomic_add`(constexpr `mop` 三元 + `if constexpr` 偏移,镜像 FMHA)。新增 POST `hstu_bwd_reduce_convert_dq_kernel`(固定序 split 归约 + convert);原 `hstu_bwd_convert_dq_kernel`(atomic 单份)保留不变 |
| `hstu_attention_batched_backward_dispatch.hpp` | Problem 的 kIsDeterministic 由硬编 false 改为类模板轴 `kIsDeterministic`;抽出共享 `launch_main_and_post<Pipeline,Kernel>`(算 num_splits、memset 单份×num_splits、launch MAIN、POST 走 reduce/convert 变体);去掉 determ throw |
| `hstu_attention_no_group_backward_bf16.cpp` | `BOOL_SWITCH_2` → `BOOL_SWITCH_3` 加 `param.kIsDeterministic` 轴 |
| `generate_instances.py` | backward determ 轴 `[False]` → `[False, True]`,重新生成 → no_group bwd 8 instance(4 atomic + 4 determ)+ ref 8 extern |
| `example_hstu_attention_bwd.cpp` | no_group 段:determ 时 dq_acc 缓冲扩到 `单份×num_splits`,`bp.split_stride_dq_acc=单份`、`bp.num_splits=ceil(grid_seqlen_kv/128)`;atomic 保持 num_splits=1/split_stride=0 |
| pipeline | **未改**(determ `store_tile` 分支 no_softmax:523 / with_softmax:561 已就位)|
| **fwd / group / atomic 路** | 逻辑零改动 |

## 3. 关键实现点
- **kernel dq_acc 窗口**(SiLU + softmax 各一处):
  ```cpp
  constexpr auto mop = kIsDeterministic ? memory_operation_enum::set
                                        : memory_operation_enum::atomic_add;
  AccDataType* dq_acc_ptr = base + i_nhead*nhead_stride_dq_acc + batch_offset_dq_acc;
  if constexpr(kIsDeterministic) dq_acc_ptr += i_tile_n * kargs.split_stride_dq_acc;
  make_naive_tensor_view<global, mop>(dq_acc_ptr, ...);
  ```
  pipeline 的 `if constexpr(kIsDeterministic) store_tile(...) else update_tile(...)` 自动写 set/atomic。
- **POST reduce**:`float acc=0; for(s<num_splits) acc += dq_acc[s*split_stride+i]; dq[i]=convert(acc);` —— 固定 s 升序,与 block 调度无关。
- **instance 处理**:no_group backward 用 **extern template instance**(非直接实例化),故必须给 determ 轴生成 4 个新 instance + 更新 ref(否则 link error:undefined `run_batched_backward_dispatch<...,true,64>`)。改 generator 重生成解决。

## 4. 验收维度①:正确性(对拍 reference,attn_scale=1.0,bf16)
`runs/run-M6-correctness.log`(10/10 PASS):determ × {SiLU,softmax} × {batched,jagged} × causal{0,1} × {no-mask, causal, window, combo, num_target},含非整除 seq200、多 KV-block seq512。误差 bf16 舍入级(dQ max_abs ≤ 8e-3,dV ≤ 3e-2)。

## 5. 验收维度②:可复现性(M6 核心)
`runs/run-M6-repro.log`:同一 case `-deterministic=1` `-dump_grad=1` 跑两次,`cmp` dq_dev.dat:
| case(seq512 → 4 KV splits)| 两次结果 |
|---|---|
| SiLU causal+window+targets | **BYTE-IDENTICAL** ✅ |
| softmax causal | **BYTE-IDENTICAL** ✅ |
| jagged softmax +per-batch targets(seq512/300/400)| **BYTE-IDENTICAL** ✅ |

(本机 atomic 路两次恰也相等——HW 上 atomic 顺序此规模未发散;但 determ 是**构造上保证**可复现,atomic 仅偶然。M6 验收点=determ byte-identical,已满足。)

## 6. 测试套件
- `skip-deterministic` 删除(轴已接,不再是 N/A)。
- 加 7 个 M6 正确性 pass 案(determ × SiLU/softmax × batched/jagged × 因子,含 seq512 多 split)。
- 新增 **in-runner 可复现性断言** `run_repro_checks()`:对 3 个 determ case 在临时 cwd 跑两次 + `filecmp` dq_dev.dat,失败计入 exit。
- `python3 test/run_bwd_tests.py` → **TOTAL 77 / PASS 77 / FAIL 0 / SKIP 0,exit 0**(`runs/test-20260609-093735.log`)。
- **零回归**:60 个 M1–M5b(atomic/SiLU/softmax/group)案仍全 PASS。

## 7. 遇到的坑
- **CMake `file(GLOB)` 陈旧**:重生成 determ instance 后,未改 CMakeLists → glob 未重算 → 4 个 determ instance 没编进 → link error(undefined symbol)。`touch CMakeLists.txt` 触发 reconfigure 重 glob 后解决。(后续若再加 instance 需注意。)
- determ instance 必须随 entry 的 BOOL_SWITCH 轴同步生成(extern template 模型),否则链接缺符号。

## 8. 遗留 / 给后续
- **M6b group determ**:group dispatch 现 determ 仍 throw;需 group kernel dq_acc 加同款 set+split 分叉 + group POST reduce + group workspace sizing(单份=ΣL×H×hdim packed × num_splits)。本单未做(范围外)。
- M7 fp16+hdim / M8 perf 不变。
- 无未解决阻塞点、未发现缺陷。
