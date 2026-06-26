# Task Contract — HSTU Attention Backward (GPU)

> 依据已批准设计:`/root/workspace/hstu-b1052-report/hstu-bwd-design-20260604.html` / `/tmp/hstu-bwd-design/DESIGN.md`(双 review 过,P0=0,U1–U4 默认已确认)。

## 1. Objective
为 HSTU attention 反向实现 GPU kernel(现状仅 855 行 CPU 参考,无 GPU bwd),复用 ck_tile FMHA bwd 的 3-kernel(PRE/MAIN kr_ktr_vr/POST)+ default policy。**主目标芯片 gfx950(MI350X / CDNA4)**(本机即是),gfx942 次之。

## 2. Correctness requirements
- Oracle:`ck_hstu/.../reference_hstu_attention_bwd.hpp`(`reference_no_group_*` / `reference_group_*`)。
- 流程:同输入(固定 seed)下 GPU(fwd 产 O/LSE → bwd 产 dQ/dK/dV)对比 CPU reference,`ck_tile::check_err` 分张量报 max/mean 相对+绝对误差。
- 容差:bf16 rel≤2e-2 / abs≤5e-2;fp16 rel≤5e-3 / abs≤1e-2(按实测收紧)。
- 不变量:masked-out 位 dS=0(不污染 dK/dV);deterministic 路逐位可复现(memcmp)。
- 覆盖:激活{SiLU,softmax} × 模式{batched,jagged,group} × mask{causal/window/contextual/min_full/num_target 及组合} × dtype{bf16,fp16} × hdim{(64,64),(128,128),(128,64),(256,256)} × dQ{atomic,deterministic}。

## 3. Performance target
M8 阶段:gfx950 上 perf 不显著落后于同 hdim 的 FMHA bwd;占用率按 CDNA4 加法模型评估。MVP 阶段(M1–M7)以正确性为先,性能记录入 benchmark.csv 但不设硬门槛。

## 4. Allowed approaches
ck_tile / HIP;复用 FMHA bwd device 模板(PRE/POST/default policy/shape/enum)+ MAIN 作结构蓝本特化。镜像 fwd 的 gfx950 结构(`BUILD_HSTU_FOR_GFX95_ONLY` + `-fno-slp-vectorize` + `#ifdef __gfx950__` + fwd_setting gfx95 tile)。**不改 fwd 行为**;mask 仅新增成员(`GetTileRangeAlongY`/`IsEdgeTile`,纯加)。Problem 保留 `BiasEnum=NO_BIAS`+`BiasDataType` dummy(default policy 复用硬前提)。

## 5. Validation command
```
# 在 ck_hstu build 目录:编译 bwd target + 跑对拍(PASS/FAIL + err)
cmake --build build --target tile_example_hstu_attention_bwd -j && \
  ./build/.../tile_example_hstu_attention_bwd --bwd_v 1 <case args>   # print: [PASS|FAIL] dQ/dK/dV max/mean err
```
(M0 阶段先确保 fwd baseline 可编可跑;bwd target 随 M0 建立。)

## 6. Evaluation command
```
./build/.../tile_example_hstu_attention_bwd <case args>   # print TFLOPS / GB/s / ms
# 深度剖析(按需): omniperf profile -- <同命令>   /   rocprofv3 --stats -- <命令>
```

## 7. Promotion criteria
候选晋级:validation **PASS**(容差内)且无新增数值回退;性能候选额外要求 eval ≥ 当前最优。被否决候选在 candidates.jsonl 记 reason。

## 8. Target arch / build
- GPU_TARGETS: **gfx950**(本机 MI350X;`rocminfo` 确认 `gfx950:sramecc+:xnack-`)。
- build: `cmake -B build -DCMAKE_PREFIX_PATH=/opt/rocm -DGPU_TARGETS=gfx950 -DCMAKE_BUILD_TYPE=Release -G Ninja`(gfx95 自动加 `-fno-slp-vectorize -DBUILD_HSTU_FOR_GFX95_ONLY`)。
- ROCm: /opt/rocm(hipcc/amdclang++);cmake + ninja。
- 关键:`-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3`(沿用 fwd)。
