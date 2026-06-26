# M6 deterministic — 独立验证 + 对抗 review 派单 (pane-2 / reviewer)

> 你独立验证 M6(dQ 逐位可复现)。pane-1 报告完成(正确性 PASS + determ 两次 byte-identical,套件 77/77/0/0)。**别信自述。**
> git 基线 = `dc8c6b21`(M5b);M6 = working-tree 未提交 + 4 新 determ instance。

## 范围 + 改动(相对 dc8c6b21)
M6 = dQ deterministic 路(no_group:batched+jagged × SiLU+softmax)。机制:每 KV-block(split_idx=i_tile_n)plain-store(非 atomic)到自己 split 副本 → POST 固定序 reduce over splits + convert → bit-reproducible。group determ = M6b(不在本单)。
改动文件:
- `hstu_attention_bwd_kernel.hpp`(SiLU+softmax kernel:dq_acc 窗口编译期分叉 set/atomic + split 偏移;新 POST `hstu_bwd_reduce_convert_dq_kernel`)
- `hstu_attention_batched_backward_dispatch.hpp`(kIsDeterministic 轴 + 抽共享 `launch_main_and_post`,**atomic 路也走重构后 helper**)
- `hstu_attention_no_group_backward_bf16.cpp`(BOOL_SWITCH_3 加 determ 轴)
- `generate_instances.py`(determ 轴 [False,True])+ 4 新 `*deterministic*` instance + ref 头
- `example_hstu_attention_bwd.cpp`(determ workspace ×num_splits)
- **不应在 diff**:两 pipeline(`*_bwd_pipeline.hpp`,determ 分支本就在)、`group_backward_dispatch.hpp`——若在,报 lead。

## 任务 A:独立机器验证(权威闸门)
1. **干净重建(必须，pane-1 踩过 CMake GLOB 陈旧坑)**:`touch CMakeLists.txt` + `touch` 改动源,`cmake -B build ...`(若 instance 没编进会 link error undefined `run_batched_backward_dispatch<...,true,64>`)再 `cmake --build build --target tile_example_hstu_attention_bwd -j128 2>&1 | tee runs/build-M6-review.log`;0 error + 链接成功(=4 determ instance 确实编进)。
2. 独立复跑 `python3 test/run_bwd_tests.py`;确认 **exit 0 / 0 FAIL / 0 SKIP**(M6 把 skip-deterministic 升级为真断言 + in-runner repro 检查),记 TOTAL/PASS(自述 77/77)。确认 M1–M5b(60 案 atomic/SiLU/softmax/group)全 PASS = **atomic 路零回归**(dispatch 重构后)。
3. **可复现性独立亲验(M6 核心，别只信 run-M6-repro.log)**:自己挑 2 个 determ case(含 seq512 多 split、softmax),`-deterministic=1` 跑两次 dump dQ,`cmp`/`md5sum` 比对 → 必须 **byte-identical**。
4. 自抽 3-4 档 determ 对拍 reference(全 `-attn_scale=1.0`,≠ 套件配置),含 jagged + causal=0+target。

## 任务 B:对抗 review(逐条核)
1. **memory_op 分叉**:determ 用 `memory_operation_enum::set`(plain store)、atomic 用 `atomic_add`?pipeline 的 `if constexpr(kIsDeterministic) store_tile else update_tile` 与之匹配?
2. **split 偏移**:determ base `+= i_tile_n * split_stride_dq_acc`?split_idx 确 = i_tile_n(KV tile 索引),不同 block 写不同副本(无重叠)?
3. **POST 固定序 reduce**:`for s=0..num_splits-1: acc += dq_acc[s*split_stride+i]` 升序固定?所以 bit-reproducible(与 block 调度无关)?
4. **num_splits / workspace**:`num_splits=ceil(grid_seqlen_kv/kN0)=grid.x`?dq_acc 缓冲扩到 单份×num_splits?memset 全量?`split_stride_dq_acc=单份元素数`?jagged packed 下单份大小对?
5. **instance**:4 个 determ instance + ref 头齐?BOOL_SWITCH_3 轴接对(causal×softmax×determ)?
6. **atomic 零回归**:dispatch 重构成 `launch_main_and_post` 后,atomic 路(num_splits=1、atomic_add、原 convert_dq）行为与重构前一致?(套件 M1–M5b PASS 佐证 + 读代码确认 atomic 分支没被 determ 改动污染)。
7. **诚实性核**:自述称"本机 atomic 两次也 byte-identical"——这是否削弱 M6 价值叙述?核心是 determ **构造上保证**可复现(固定序无 atomic),与 atomic 本机是否偶然稳定无关;确认 done.md 叙述诚实、没把"atomic 也稳"当 determ 的功劳。
8. 边界:seq512 多 KV-block(num_splits>1 真路径)、causal=0+target、非整除 seq。

## 产出
写 `/tmp/hstu-bwd-design/M6-review-findings.md`:任务 A 实测(build/suite/**亲验 repro cmp 结果**/抽样数值+exit)、任务 B 逐条 GREEN/问题、总评(promote / 需修+blocker)。发现真缺陷如实列、立刻报 lead。
