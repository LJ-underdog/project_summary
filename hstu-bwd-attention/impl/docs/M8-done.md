# M8 perf — DONE (MI + B2 + B3). 本期 scope 全部完成,停报 lead。

> 本期 M8 scope(lead 闸门裁决,`docs/draft-M8-perf.md` 顶部)= **MI + B2 + B3 only**(runtime 真赢)。
> 基线 HEAD `4629508f`(cross)。**全部未 commit**(候 lead 闭合后统一)。

## 一句话
为 HSTU bwd 建测量基线(MI,behind `-perf`),并据此把 MAIN dqdkdv 的 `GetTileRangeAlongY`
从保守全扫收紧到 causal(B2)+ local/window(B3)真实 Q-row 范围,MAIN 实测加速
**causal 1.25–1.60×、window 4.7–9.8×**(诚实低于 [derived] 模型,Amdahl 归因如下)。

---

## 三个 candidate

### MI — 测量基线(enabler,behind `-perf`)
- 改面:`hstu_attention_bwd_perf.hpp`(新,`time_op` 唯一计时原语)+ bwd params 两 struct 加 host-only
  `measure_perf`/`perf_{pre,memset,main,post}_ms`(**不进 MakeKargs → 设备码不变**)+ batched/group
  dispatch 把 PRE/memset/MAIN/POST 各 launch 包 `time_op` + harness `-perf`(envelope 整 bwd warmup+repeat
  + per-kernel + 5-GEMM FLOPS,打印 `PERF kernel=.. metric=.. value=..`)。
- 驱动 `test/run_perf_baseline.py`(跑 canonical/hd256/window × silu/softmax,解析 PERF,追加 benchmark.csv 10 列 schema)。
- **零回归证据**:不给 `-perf` 时 co_symbols **13782/13782 byte-identical**(457 obj 全集,超 reviewer 870 口径)
  + 套件 **253/253**;`-perf` per-kernel 与独立 rocprofv3 profile 互证(canonical MAIN 0.263ms≈266us、
  envelope 87%≈87.6%、hd256 silu 0.971ms≈943us、**SiLU 26% 异常复现** silu/softmax=1.27×)。
- 报告:`/tmp/hstu-bwd-design/M8-MI-stage1-done.md`。基线 json `runs/M8-MI-baseline-HEAD-4629508f.json`。

### B2 — GetTileRangeAlongY 紧致化 causal(NoLocal)
- 改面:`hstu_block_masking.hpp` 两个 **NoLocal** mask(self+cross)`GetTileRangeAlongY`:
  `y_start=(ctx>0&&i_x<max_uih)?0:align_down(i_x[-diff])`,`y_end=seqlen`;非 causal(IsMasking=false,
  含 P1-1 num_target)与 WithLocal 原样不动。
- 4 gate 全过:① 离线校验器 ALL GREEN;② 套件 253/253 + 边界 stress 15/15;③ MAIN 加速(下表);
  ④ co_symbols surgical(仅 causal 实例变)。报告 `/tmp/hstu-bwd-design/M8-B2-done.md`,证据 `runs/B2-cosym-vs-HEAD.txt`。

### B3 — GetTileRangeAlongY 紧致化 local/window(WithLocal)
- 改面:`hstu_block_masking.hpp` 两个 **WithLocal** mask(self+cross)`GetTileRangeAlongY`:window 双边带
  (causal row∈[col,col+W];非causal [col-W,col+W])+ cross diff_q_kv_len 偏移 + contextual(y_start=0,
  y_end≥ctx)+ **非causal min_full 行下沉 y_start**(校验器抓到的真 bug,见下)+ target/min_full → y_end=seqlen。
- 4 gate 全过(下文)。

---

## ★ 校验器抓到 2 个真 under-tighten bug(B3,silent-wrong 被挡在 GPU 之外)
1. **非 causal + min_full**:min_full 行(row_id≥max_id−mf)attend **所有列**,在非 causal 下落在 band 下沿
   **之下**(物理起点 max_uih_len−mf),最初只在 y_end 处理 → 漏。修:非 causal WithLocal 把 y_start 下沉到
   `max_uih_len−mf`(causal 不受影响,min_full 仍需 row≥col 不低于 i_x)。
2. **cross causal 大 diff + contextual**:diff 大时 band 映射出界(y_end 被 clamp 到 0),但 contextual 行
   [0,ctx) 仍 attend → 漏。修:contextual 分支 floor `y_end≥contextual_seqlen`。
→ 修后校验器 **1,973,278 checks ALL GREEN**。**这正是离线穷举 gate 的价值**(比对拍更早、更硬地挡 silent-wrong)。

---

## benchmark.csv MAIN 加速汇总(bf16, batched, causal, seqlen=2048, per-kernel MAIN time_ms)
| config | MI 基线 | B2 | B3 | 说明 |
|---|---|---|---|---|
| canonical softmax hd64 | 0.2632 | **0.2027 (1.30×)** | 0.2030 | B2 收割 causal;B3 不动 NoLocal |
| canonical silu hd64 | 0.3327 | **0.2085 (1.60×)** | 0.2084 | |
| hd256 softmax | 1.0325 | **0.7673 (1.35×)** | 0.7740 | |
| hd256 silu | 0.9709 | **0.7762 (1.25×)** | 0.7779 | |
| window256 softmax | 0.2992(全扫) | 0.2994(B2不动) | **0.0635 (4.71×)** | B3 收割 window |
| window256 silu | 0.4193 | 0.4192 | **0.0840 (4.99×)** | |
| window64 softmax/silu | 0.2992/0.4193 | — | **0.0362/0.0468 (8.3×/9.0×)** | 窄 window 赢更多 |
| window16 softmax/silu | 0.2992/0.4193 | — | **0.0318/0.0430 (9.4×/9.8×)** | |

(window 的 "before" = MI 全扫基线;全扫 MAIN 与 window 大小无关,故各 window 档共用 window256 基线。)

## 诚实归因(Amdahl,实测 < [derived] 模型)
- B2 causal 实测 1.25–1.60×,**低于 draft [derived] ~1.9×**:收紧只砍 MAIN 内 Q-tile 循环数;每 KV-block 的
  K/V load、atomic dq_acc 写流量、kernel 启动开销 **不随之减半** → Amdahl 限制 < 2×。
- B3 window 实测 4.7–9.8×,**低于 [derived] ~22×**:同理 + 窄 window 时 MAIN 已极小(window16 MAIN 0.032ms
  < PRE 0.025ms),MAIN 不再是瓶颈,envelope 被 PRE/启动开销主导。**实测多少记多少,不吹模型数。**

## 暂缓项(本期 scope 外,draft 已裁决)
- **B4 grid widening**:scoping "grid starvation 根因" 被 critique 实测证伪(256x 超额订阅),依据作废。
- **B7 hd256 occupancy / VGPR 124-vs-248**:阻塞于先解 VGPR 矛盾 + LDS-vs-MFMA 测量。
- **B1 group TU split**(build-axis,可日后并行)、**B5 first-split-skip**、**B6 trload**(高风险高工)、
  **INV SiLU 26% 异常**(MI 已复现,根因待查)、**B8/B9/B10**。
- 真限制器 = per-block LDS/VGPR(非 grid 数);MemUnitStalled=0.024%。占用率本期不碰。

## 全证据索引
- 离线校验器:`test/validate_tile_range_y.cpp`(B2+B3 扩 cross + WithLocal + 2 tile 形状,1,973,278 checks GREEN)。
- 套件:`runs/test-20260615-110525.log`(253/253);MI `test-20260615-093723`、B2 `test-20260615-101834`。
- co_symbols:MI 基线 `runs/M8-MI-baseline-HEAD-4629508f.json`;B2 `runs/B2-cosym-vs-HEAD.txt`;
  B3 分类 `runs/B3-cosym-category.txt`(FORWARD 388 obj 0 DIFF、kentry 0 DIFF、bwd MAIN 改动合法)。
- 加速:`benchmark.csv`(`MI-baseline-*` / `B2-causal-tighten-*` / `B3-window-tighten-*` 行)+ `test/run_perf_baseline.py`。
- 各 stage 报告:`/tmp/hstu-bwd-design/M8-{MI-stage1,B2}-done.md` + 本文件。

## 改面文件(全部未 commit)
1. `hstu_attention_bwd_perf.hpp`(新,MI)
2. `hstu_attention_bwd_params.hpp`(MI,host-only perf 字段)
3. `hstu_attention_batched_backward_dispatch.hpp`(MI 计时)
4. `hstu_attention_group_backward_dispatch.hpp`(MI 计时)
5. `example_hstu_attention_bwd.cpp`(MI `-perf` + envelope + FLOPS)
6. `hstu_block_masking.hpp`(B2 NoLocal + B3 WithLocal `GetTileRangeAlongY` 收紧)
7. `test/validate_tile_range_y.cpp`(B2+B3 校验器扩展)
8. `test/run_perf_baseline.py`(新,MI 驱动)
9. `benchmark.csv`(迁 10 列 schema + MI/B2/B3 行)
