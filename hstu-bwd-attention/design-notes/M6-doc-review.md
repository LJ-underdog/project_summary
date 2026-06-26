# M6 deterministic dQ — 文档级独立 review 结论 (pane-3 / 第三方视角)

> 角色:pane-3,边写 HTML 讲义边以全新视角核 silent-wrong(coder=pane-1、reviewer=pane-2 已过)。
> 只读源码,基线 git `c79d3296`(相对 M5b `dc8c6b21` 的改动)。
> **结论先行:GREEN,无 blocker。O1(group+determ 静默走 atomic)经我独立核实属实,已如实写入报告,非 M6 范围内缺陷(M6b 修)。**

---

## 我独立重核过的点(逐条 GREEN)

### 1. memory_op 分叉(set↔store / atomic_add↔update)— GREEN
- kernel:`constexpr auto mop = kIsDeterministic ? set : atomic_add`(SiLU :364-365、softmax :795-796)。
- pipeline:`if constexpr(kIsDeterministic) store_tile(dq_dram_window,dq_acc) else update_tile(...)`(no_softmax:523-529、with_softmax:561-567)。
- set(plain store)↔ `store_tile`、atomic_add ↔ `update_tile`(atomic)严格配对。两条 pipeline **byte-identical 于 M5b**(git diff 空),determ 分支早就在,M6 未碰。✓

### 2. split 偏移(无重叠/越界)— GREEN
determ 时 `dq_acc_ptr += i_tile_n * kargs.split_stride_dq_acc`(`if constexpr(kIsDeterministic)` 内,SiLU :370-372 / softmax :801-803)。`split_idx = i_tile_n = i_n0/kN0`,不同 KV-block 写不同副本。`i_tile_n ∈ [0, grid.x)`、`num_splits = grid.x` ⇒ 偏移恒落在分配的 num_splits 槽内,无越界、无重叠。✓

### 3. POST 固定序 reduce — GREEN
`hstu_bwd_reduce_convert_dq_kernel`(kernel:1672-1686):`for(s=0;s<num_splits;++s) acc += dq_acc[s*split_stride + i]; dq[i]=convert(acc)`。`s` 严格升序、与 block 调度无关 ⇒ bit-reproducible。`acc` 为 float,累加后才 convert。✓

### 4. num_splits == grid.x — GREEN
dispatch `launch_main_and_post`:`num_splits = kIsDeterministic ? ceil(grid_seqlen_kv/Pipeline::kN0) : 1`(:84-87);`GridSize(...,grid_seqlen_kv)` → grid.x = ceil(grid_seqlen_kv/kN0)。两者同式同输入 ⇒ num_splits == grid.x。`Pipeline::kN0=128`,harness 硬编 `kN0_bwd=128`(:301)一致。memset = `single*num_splits`(:89-93)清零全 workspace,保证未写 Q 行=0、POST 累加 0 不污染。✓

### 5. 写/读 split_stride 一致(silent-corrupt 风险点)— GREEN(自推证伪)
- **kernel 写** 用 `param.split_stride_dq_acc`(harness:479 设 = `single_dq_acc_elems`,:305-306 = `batches_for_alloc*phy_seqlen_q*num_head*hdim_qk`)。
- **POST 读** 用 dispatch 独算的 `single`(:75-79)。
- 自推两侧相等:
  - batched:`single = num_batch*batch_stride_dq_acc`;`batch_stride_dq_acc = dq_host.strides()[0] = phy_seqlen_q*num_head*hdim_qk`;batched `batches_for_alloc=num_batch` ⇒ `single = num_batch*phy_seqlen_q*num_head*hdim_qk = single_dq_acc_elems`。
  - jagged:`single = batch_stride_dq_acc = strides()[0] = phy_seqlen_q*num_head*hdim_qk`;jagged `batches_for_alloc=1` ⇒ `single_dq_acc_elems = phy_seqlen_q*num_head*hdim_qk = single`。
  两侧恒相等,无错位。pane-2 A.5 atomic-vs-determ diff=0 + seq512/640/768 多 split 对拍实测兜底。✓

### 6. atomic 路零回归(dispatch 重构后)— GREEN
`launch_main_and_post` 的 atomic 分支:`num_splits=1` → memset `single` → `hstu_bwd_convert_dq_kernel`(原 POST,未改)→ kernel `mop=atomic_add` 无 split 偏移。codegen/行为与重构前等价。两 pipeline + `group_backward_dispatch.hpp` **byte-identical 于 dc8c6b21**(git diff 空)。套件 60 个 M1–M5b 案全 PASS(pane-2 A.2)。✓

### 7. O1 描述准确性 — 确认属实(我独立核到根因)
- **根因(源码级)**:group entry `hstu_attention_group_backward_bf16.cpp:17-22` 用 `BOOL_SWITCH_2`(无 determ 轴)+ 硬编 `false /*kIsDeterministic*/` → group dispatch **只会以 `kIsDeterministic=false` 实例化**。
- 故 group dispatch `if constexpr(kIsDeterministic) throw "(M6)"`(`group_backward_dispatch.hpp:318-319)`)是**编译期裁掉的不可达分支** —— `if constexpr` 在 false 实例里整段不编出,**永不 throw**。
- 后果:`-deterministic=1 -g=2` 请求 → group entry 静默忽略 determ,走 atomic 路 → 返回**数值正确但非逐位可复现**的梯度,且**不报错**。
- **`M6-done.md §8` 称"group dispatch 现 determ 仍 throw"措辞不准**(`if constexpr` 不可达,不会 throw);以 `HANDOFF.md` M6 节 O1 修正为准。我已在报告"范围与 O1"节如实写出,未美化。
- 性质:数值正确(atomic 已对拍),非 silent-**wrong**;违背的是 *determinism 契约*(silent-ignore);属 M6b 范围。**非 M6 范围内 blocker。**

---

## 非阻塞观察(轻微,留备)
- `bp.num_splits`(harness:480)被设置,但 dispatch `launch_main_and_post` 在 determ POST 里用的是**本地独算的** `num_splits`(:84),不读 `param.num_splits`。属无害冗余(两者同式同值),非 bug。

---

## 总评
**GREEN,M6 范围(no_group batched+jagged × SiLU+softmax 的 dQ deterministic)无 blocker。** 6 条机制点(memory_op 分叉 / split 偏移 / POST 固定序 / num_splits==grid.x / split_stride 写读一致自推 / atomic 零回归)逐条独立重核,与 FMHA 蓝本 + pane-2 findings 对齐;关键 silent-corrupt 点经静态自推 + A.5 diff=0 双重证伪。O1(group+determ 静默 atomic)经我核到源码根因属实,已如实写报告并标注 done.md §8 措辞不准、以 HANDOFF 为准,留 M6b。未发现 pane-2 之外的真问题。
