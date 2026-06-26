# M8 Stage 1 (MI 测量基线) — DONE, 停报 lead 亲验

HEAD 基线 `4629508f`(cross)。**未 commit**(等 lead 闭合)。

## 设计要点(为何 behind-flag 且设备码 byte-identical)
- 计时唯一原语 `hstu_attention_bwd_perf.hpp::time_op(measure,stream,fn)`:
  - `measure=false`(正常路/套件)→ **只调 `fn()` 一次**,host 行为等同裸 launch;
  - `measure=true` → cold3 warmup + nrep10 hipEvent gpu_timer,返回 per-launch 均值 ms。
- perf 字段是 **params 纯 host 输出字段**(`measure_perf` + `perf_{pre,memset,main,post}_ms`),
  **不进 MakeKargs → 不入设备 kargs → 设备符号不变**。
- dispatch(batched + group)把 PRE/memset/MAIN/POST 各自 launch 包进 `time_op`,
  内层 MAIN 仍 `stream_config{stream,false}`(单发,warmup/repeat 由 time_op 做)。
- 正常路设备码 = 改前逐位一致(下证)。

## 改面(8 文件)
1. `hstu_attention_bwd_perf.hpp`(新)— time_op。
2. `hstu_attention_bwd_params.hpp` — 两 struct 末尾追加 host-only perf 字段(默认初值)。
3. `hstu_attention_batched_backward_dispatch.hpp` — PRE/memset/MAIN/POST 包 time_op。
4. `hstu_attention_group_backward_dispatch.hpp` — 同上。
5. `example_hstu_attention_bwd.cpp` — `-perf` flag;校验后(host 梯度已落)做 envelope(整 bwd warmup+repeat)+ per-kernel(measure_perf=true 单发填字段)+ 5-GEMM FLOPS;打印 `PERF kernel=.. metric=.. value=..`。no_group + group 都接。timer.hpp include。
6. `test/run_perf_baseline.py`(新)— 跑 canonical/hd256/window × silu/softmax,解析 PERF,追加 csv。
7. `benchmark.csv` — 迁移到 10 列 schema(加 `kernel` 列,历史 3 行 kernel=n/a)+ 42 MI 基线行。
8. `runs/M8-MI-baseline-HEAD-4629508f.json`(新)— 13782 符号基线(457 obj 全集)。

## ★ Stage 1 硬检查点 —— 全过
### 1. MI behind-flag 零回归
- **co_symbols verify(全 457 bwd obj:4 entry + 448 instance + group + kentry wrapper)
  = 13782/13782 byte-identical,0 MISSING,0 DIFF,exit 0**。
  (是 reviewer 870 基线口径的**严格超集**——更强。baseline 在改前 HEAD 净 dump。)
- **套件 253/253 PASSED,0 FAILED,0 SKIPPED,exit OK**(`runs/test-20260615-093723.log`)。
- 非 -perf 跑 0 条 PERF 行(套件路零受扰)。

### 2. -perf 跑通 + benchmark.csv 基线行落地 + per-kernel 分得开
6 config 全 `numeric_pass=true`,42 行入 `benchmark.csv`(canonical + hd256 + window,silu+softmax 各档):

| config | act | MAIN ms | MAIN TFLOPS | envelope ms | PRE | memset | POST |
|---|---|---|---|---|---|---|---|
| canonical hd64 | softmax | 0.2632 | 163.2 | 0.3011 | 0.0254 | 0.0035 | 0.0047 |
| canonical hd64 | silu | 0.3327 | 129.1 | 0.3451 | 0 | 0.0034 | 0.0048 |
| hd256 | softmax | 1.0325 | 166.4 | 1.1530 | 0.0963 | 0.0075 | 0.0171 |
| hd256 | silu | 0.9709 | 176.9 | 1.0209 | 0 | 0.0072 | 0.0170 |
| window256 hd64 | softmax | 0.2992 | 143.5 | 0.3366 | 0.0251 | 0.0045 | 0.0047 |
| window256 hd64 | silu | 0.4193 | 102.4 | 0.4310 | 0 | 0.0034 | 0.0049 |

**与独立 rocprofv3 profile 互证(强信号,非臆造)**:
- canonical MAIN 0.263ms ≈ profile 266.20us;envelope 0.301ms ≈ profile 303.88us;MAIN 占 87%(profile 87.6%)。
- hd256 silu MAIN 0.971ms ≈ profile 943us。
- **SiLU 26% 异常复现**:canonical silu 0.333 / softmax 0.263 = **1.27×**(profile 335/266us)。
- **window 现无加速**(GetTileRangeAlongY 仍全扫,silu window 0.419 > canonical 0.333)= B3 待收割的浪费。

## 报 lead
**MI done,253/253 + co_symbols 13782/13782 byte-identical(超 870 口径)+ 基线 benchmark.csv
见上表/MI-baseline-* 行,MAIN per-kernel(MAIN/PRE/memset/POST/envelope)计时可用且与 profile 互证。停,候 lead 亲验放行 Stage 2(B2)。**
