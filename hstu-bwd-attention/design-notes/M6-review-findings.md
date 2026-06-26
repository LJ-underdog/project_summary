# M6 deterministic — 独立验证 + 对抗 review findings (pane-2 / reviewer)

**裁决:✅ PROMOTE**(M6 范围内 = no_group batched+jagged × SiLU+softmax 的 deterministic dQ 路,正确性 + 逐位可复现 + atomic 零回归 全部独立实测通过)。
1 个非阻塞观察(group+determ 静默走 atomic、done.md 措辞不准),建议留给 M6b 处理,不阻塞 M6。

基线 git `dc8c6b21`(M5b)。M6 = working-tree 未提交 + 4 新 determ instance。审查 + 实测 2026-06-09。

---

## 范围闸门(改动文件)— GREEN
`git status` 恰 10 项,与派单清单完全一致,无越界:
- 改:`hstu_attention_bwd_kernel.hpp`、`hstu_attention_batched_backward_dispatch.hpp`、`hstu_attention_no_group_backward_bf16.cpp`、`generate_instances.py`、`example_hstu_attention_bwd.cpp`、`instances/...batched_backward_bf16_instances_ref.hpp`
- 新:4 个 `*deterministic*` instance .cpp
- **禁改文件确认未动**:两 pipeline(`*_bwd_pipeline.hpp`)、`group_backward_dispatch.hpp` — 均不在 diff。✅

---

## 任务 A:独立机器验证(权威闸门)

### A.1 干净重建 — GREEN
- `touch CMakeLists.txt` + 全部改动源 + instances/*backward*.cpp → reconfigure(`cmake -B build`,configure/generate done)→ `cmake --build --target tile_example_hstu_attention_bwd -j128`。
- **0 error**(`runs/build-M6-review.log`,仅 78 host warnings,皆既存 fwd pipeline note),`[158/159] Linking ... bin/tile_example_hstu_attention_bwd` 成功。
- 链接成功 = 4 个 determ extern-template instance(`run_batched_backward_dispatch<...,true,64>`)确实编进并解析,否则 undefined-symbol。**CMake GLOB 陈旧坑已通过 touch CMakeLists 规避。**

### A.2 套件独立复跑 — GREEN
`python3 test/run_bwd_tests.py`(独立机器,`runs/test-M6-review.log`):
- **TOTAL 77 / PASSED 77 / FAILED 0 / SKIPPED 0,exit 0**(与自述一致)。
- M1–M5b 60 案(atomic SiLU/softmax/group)全 PASS = **dispatch 重构(`launch_main_and_post`)后 atomic 路零回归**(实测佐证)。
- M6 7 正确性案(determ × SiLU/softmax × batched/jagged × 因子,含 seq512 4-split)全 PASS。
- in-runner 可复现性 3/3 byte-identical。

### A.3 可复现性独立亲验(M6 核心,未信 runner)— GREEN
自己跑(`/tmp/m6rev`,off-suite 配置),两次 dump dQ 后 `md5sum`+`cmp`:
| case | num_splits | 结果 |
|---|---|---|
| SiLU batched seq640 causal+window(48)+target(24) | 5 | md5 一致,`cmp` **BYTE-IDENTICAL** ✅ |
| softmax jagged seq=600,257,448 + targets | 多 | md5 一致,`cmp` **BYTE-IDENTICAL** ✅ |

### A.4 off-suite 正确性对拍(全 `-attn_scale=1.0 -v=1`,≠套件配置)— GREEN
| case | num_splits | 对拍 |
|---|---|---|
| determ softmax jagged **causal=0 + num_target**(512,300,400)— P1-1 交叉 | 多 | dQ/dK/dV [PASS], numeric_pass=true |
| determ SiLU batched **seq768 + context_len=32** + causal | 6 | dQ/dK/dV [PASS] |
| determ softmax batched seq256 window causal=0 | 多 | dQ/dK/dV [PASS] |
| determ SiLU jagged **非整除** 200/137/99 causal | 多 | dQ/dK/dV [PASS] |

### A.5 atomic-vs-determ 数值交叉核(我加的关键核)— GREEN
同一 case(softmax seq512 causal+window+target,4 split)分别 `-deterministic=0/1` dump dQ,bf16→fp32 比对:
**max_abs_diff(atomic vs determ) = 0.000e+00**(完全相等)。
⇒ determ 不是"自洽的错值",而是产出与已对拍 atomic **逐位相同的正确梯度**;此规模 atomic 恰也稳定(印证 done.md 诚实叙述,见 B7)。

---

## 任务 B:对抗 review(逐条)

**B1 memory_op 分叉 — GREEN.** kernel(SiLU :364 / softmax :795)`constexpr auto mop = kIsDeterministic ? memory_operation_enum::set : atomic_add`;pipeline(no_softmax:523 / with_softmax:561)`if constexpr(kIsDeterministic) store_tile else update_tile`。set↔store(plain)、atomic_add↔update(atomic)严格匹配。

**B2 split 偏移 — GREEN.** determ:`dq_acc_ptr += i_tile_n * kargs.split_stride_dq_acc`(`if constexpr(kIsDeterministic)` 内)。`split_idx = i_tile_n = i_n0/kN0`;不同 KV-block 写不同副本。num_splits=grid.x=ceil(grid_seqlen_kv/kN0)、i_tile_n∈[0,grid.x) ⇒ 偏移落在分配的 num_splits 槽内,**无越界、无重叠**。

**B3 POST 固定序 reduce — GREEN.** `hstu_bwd_reduce_convert_dq_kernel`:`for(s=0;s<num_splits;++s) acc += dq_acc[s*split_stride + i]; dq[i]=convert(acc)`。s 严格升序、与 block 调度无关 ⇒ bit-reproducible。AccDataType=float,float 累加后 convert。

**B4 num_splits / workspace — GREEN(逐项核且证明 stride 一致).**
- `num_splits = kIsDeterministic ? ceil(grid_seqlen_kv/Pipeline::kN0) : 1`;`GridSize=dim3(ceil(seqlen_kv/kN0),...)` ⇒ num_splits==grid.x。✅
- `Pipeline::kN0 = BlockFmhaShape::kN0 = BlockTile::at(1) = 128`(FmhaBlockTile=<32,**128**,...>);harness 硬编 `kN0_bwd=128` 与之一致。✅
- **memset 全量**:`single * num_splits * sizeof(float)`(atomic num_splits=1 → 与重构前等价;determ 全 workspace 清零,保证每槽未写 Q 行=0,POST 累加 0 不污染)。✅ 这是 determ 正确性的关键前置,已满足。
- **write/read split_stride 一致性(silent-corrupt 风险点,已证伪)**:kernel 写偏移用 `param.split_stride_dq_acc`(harness 设 = `single_dq_acc_elems = batches_for_alloc*phy_seqlen_q*num_head*hdim_qk`);POST 读 split_stride 用 dispatch 独算 `single`。逐式推导:batched `single=num_batch*batch_stride_dq_acc=num_batch*stride[0]=batches_for_alloc*single_dq_acc_elems`;jagged `single=batch_stride_dq_acc=stride[0]=single_dq_acc_elems`(batches_for_alloc=1)。两侧**恒相等**,无错位。A.5 diff=0 + seq512/640/768 多 split 对拍 PASS 实测兜底。✅

**B5 instance — GREEN.** generator determ 轴 `[False]→[False,True]`;重生成 8 instance(4 atomic+4 determ)+ ref 8 extern(`grep -c extern=8`)。模板序 `<InOut,kUseCausal,kUseSoftmax,kHasBias,kIsDeterministic,MaxK>` 与 entry `BOOL_SWITCH_3` 调用 `<bf16,kUseCausal,kUseSoftmax,false,kIsDeterministic,64>` 一一对位;determ instance 第 5 参=true。✅

**B6 atomic 零回归 — GREEN.** dispatch 抽 `launch_main_and_post<Pipeline,Kernel>`:atomic 走 num_splits=1 → memset `single`、走 `hstu_bwd_convert_dq_kernel`(原 kernel,未改)、kernel mop=atomic_add 无 split 偏移 ⇒ codegen/行为与重构前等价。套件 M1–M5b 60 案全 PASS 实测佐证。✅

**B7 诚实性 — GREEN.** done.md §5 明言"本机 atomic 两次恰也相等…但 determ 是**构造上保证**可复现,atomic 仅偶然"。我 A.5 实测 atomic==determ(diff=0)印证此规模 atomic 稳定,且**未**把"atomic 也稳"算作 determ 功劳;叙述诚实。

**B8 边界 — GREEN.** seq512/640/768 多 KV-block(num_splits>1 真路径)、causal=0+target(P1-1 交叉)、context_len、非整除 jagged(200/137/99)、jagged 多 split — 全对拍 PASS + 多 split repro byte-identical。

---

## 非阻塞观察(建议 M6b 处理,不阻塞 M6 promote)

**O1 — group+determ 静默走 atomic,done.md 措辞不准。**
- 实测 `-deterministic=1 -g=2 -softmax=1 ...` → 对拍 PASS、numeric_pass=true,**但走的是 atomic 路、非 deterministic**。
- 根因:group entry `hstu_attention_group_backward_bf16.cpp:17-22` 仍 `BOOL_SWITCH_2` + 硬编 `false /*kIsDeterministic*/`;group dispatch :319 的 `if constexpr(kIsDeterministic) throw "(M6)"` 因此**永不可达**。
- done.md §8 称"group dispatch 现 determ 仍 throw"——**不准确**:不会 throw,而是静默忽略 determ 请求、返回正确但非逐位可复现的梯度。
- 性质:① 梯度数值**正确**(atomic 路已对拍),非 silent-**wrong**;② 静默忽略的是 *determinism 契约*,非数值;③ 是 M6 前就存在的模式(M6 前 no_group entry 也硬编 false,dispatch throw 同样不可达;M6 已为 no_group 接 BOOL_SWITCH_3 修正之);④ group determ 明确属 M6b 范围。
- 建议:M6b 实现 group determ 时,要么 group entry 接 determ 轴,要么在 group entry/dispatch 加一个**运行时可达**的 `if(param.kIsDeterministic && is_group) throw` 守卫(对齐 HANDOFF §7"silent-wrong/silent-ignore 不可取");并修正 done.md/HANDOFF 对 group determ 现状的描述。

---

## 证据
- build:`/root/workspace/ck_hstu/runs/build-M6-review.log`(0 error,link OK)
- 套件:`/root/workspace/hstu-bwd-impl/runs/test-M6-review.log`(77/77/0/0 exit 0)
- 亲验 repro + atomic/determ 交叉:`/tmp/m6rev/`(r1*/r2*/atomic.dat/determ.dat)

## 总评
M6 范围内交付(no_group batched+jagged × SiLU+softmax 的 dQ deterministic)三维独立闭合:**重建 0-error + 套件 77/77 零回归 + 我亲跑多 split repro byte-identical + off-suite 对拍 PASS + atomic/determ 逐位一致**。对抗 8 条全 GREEN,关键 silent-corrupt(write/read split_stride 错位)风险点经静态推导 + 实测双重证伪。**裁决:PROMOTE。** O1 为非阻塞、留 M6b。
