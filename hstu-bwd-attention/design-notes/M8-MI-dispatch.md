# M8 Stage 1 实现单 —— 测量基线 MI(coder pane 0.1,已 /clear)

新里程碑 **M8 perf**,本期 scope = **MI + B2/B3**(runtime 真赢)。设计 **`/root/workspace/hstu-bwd-impl/docs/draft-M8-perf.md` 已过 lead 闸门**——**先完整读它**(尤其顶部"★ lead 闸门裁决",实现以此为准)。基线 HEAD=`4629508f`(cross)。

## 背景(实测,scoping workflow)
MAIN dqdkdv kernel 主导 84–90% 时间、矩阵核闲 ~90%(MfmaUtil 9.9%、occupancy 10.6%)、**非 memory-bound**。瓶颈=浪费的 MAIN 迭代(GetTileRangeAlongY 现保守全扫,causal ~1.9x / window 最高 ~22x 白做)+ 占用率(占用率本期暂缓)。**harness 现无 kernel 计时 → MI 是一切 runtime perf 的前提。**

## Stage 1 = MI 测量基线(只做这个,完停报 lead)
镜像 fwd `-perf`(`example_hstu_attention_fwd.cpp:243` flag、`:664` gpu_timer):
1. harness 加 `-perf` flag(**behind flag**,不给 -perf 时走原 validation 路、253 套件零受扰)。
2. hipEvent gpu_timer 计时:**envelope(PRE+memset+MAIN+POST)+ 每-kernel**(per-kernel 必需——B2/B3 只动 MAIN,要单独归因 MAIN 加速;可用 stream_config time_kernel + warmup + repeat median)。
3. FLOPS 模型:bwd 5-GEMM `2*(2*sq*skv*hdim_qk + 3*sq*skv*hdim_v)*b*nhead`(注:忽略 elementwise=GEMM-only TFLOPS 作 tracking;memset 单独算给 ZERO_dq_acc 别混进 MAIN)。
4. benchmark.csv schema:`candidate,arch,mode,activation,dtype,hdim,kernel{envelope/PRE/MAIN/POST},metric{time_ms/TFLOPS/occ},value,date`,追加到 `/root/workspace/hstu-bwd-impl/benchmark.csv`。
5. **记录基线行**:canonical(`-prec=bf16 -b=2 -nhead=8 -seqlens=2048 -softmax=1 -causal=1` hd64)+ hd256 + window + SiLU/softmax 各档,跑 -perf 写 benchmark.csv。

## ★ Stage 1 硬检查点(完成停报 lead 亲验)
1. **MI behind-flag 零回归**:不给 -perf 时 `co_symbols.py verify` self 符号 **byte-identical**(reviewer 的 870 基线口径;MI 是 -perf 内追加,不改正常路设备码)+ 套件 **253/253 exit 0**。
2. -perf 跑通,**benchmark.csv 基线行落地**(canonical + hd256 + window),per-kernel(MAIN/PRE/POST)时间分得开。
3. 报 lead:"MI done,253/253 + co_symbols byte-identical + 基线 benchmark.csv 见 X,MAIN per-kernel 计时可用",停。

## Stage 2/3(MI 检查点放行后)
B2 GetTileRangeAlongY 紧致化 causal → B3 window;每候选过 4 gate(离线 validate_tile_range_y ALL GREEN + 253/253 + 边界 stress + MAIN 加速 vs 基线)。详见 draft 顶部安全要求。

## 纪律
- MI 必须 behind-flag(正常路 byte-identical);不 commit(lead 闭合后统一);带 FAIL/裸数据不充数。
