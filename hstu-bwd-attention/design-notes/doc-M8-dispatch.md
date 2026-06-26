# 派单:写 M8 讲义(perf — MI 计时基建 + B2 causal + B3 window 紧致化)

**先读硬规格:`/tmp/hstu-bwd-design/doc-series-spec.md`(全文)。本卡给 M8 专属输入。**

- **里程碑**:M8 = 性能优化首轮,scope(lead 闸门裁决)= **MI(measurement infra,测量基建)+ B2(causal 紧致化)+ B3(window/local 紧致化)only**;占用率类(B4/B7)经 profiling 证伪/低 ROI 暂缓。核心:GetTileRangeAlongY 此前保守全扫 q-tile,B2/B3 按 mask 结构收紧 MAIN kernel 的 q-loop 范围。
- **commit(行号锚定)**:`a86529dc`。`cd /root/workspace/ck_hstu && git show --stat a86529dc`(改 6 文件:`example_hstu_attention_bwd.cpp`、`hstu_attention_batched_backward_dispatch.hpp`、`hstu_attention_bwd_params.hpp`、新 `hstu_attention_bwd_perf.hpp`、`hstu_attention_group_backward_dispatch.hpp`、`hstu_block_masking.hpp`〔B2/B3 收紧主战场,+178 行〕)。
- **旧 HTML(参考叙事,行号/数值必重核)**:`/root/workspace/hstu-b1052-report/hstu-bwd-M8-perf-20260615.html`。
  - ⚠ 旧讲义/旧笔记里 CDNA4 LDS 出过错(写成 64KB,实测 **160KB** = GROUP segment;64KB 是 CDNA3)。本篇若提硬件别抄 64KB;但 M8 主题是 runtime 紧致化,占用率/LDS 仅在「暂缓的 B4/B7」一笔带过即可,别展开臆造。
- **事实来源**:`/root/workspace/hstu-bwd-impl/docs/M8-done.md`(权威,数值/scope/验证全在此)+ `/tmp/hstu-bwd-design/M8-INV-findings.md`(profiling 深挖,如要引占用率/VALU-bound 结论)+ `draft-M8-perf.md`(顶部闸门头有 scope 裁决)。加速比/对拍案数照抄并标出处。
- **输出**:`/root/workspace/hstu-b1052-report/hstu-bwd-M8-perf-20260625.html`
- **M8 讲解重点**:
  1. **profiling 立靶**(rocprofv3 实测):MAIN dqdkdv 占 wall-time **84–90%**、**矩阵核闲 ~90%**(MfmaUtil 9.9%、occupancy 10.6%)、**非 memory-bound**(MemUnitStalled 0.024%)→ 瓶颈 = 浪费的 MAIN q-tile 迭代(GetTileRangeAlongY 保守全扫)。先讲清「为什么砍 q-loop 是对的靶」。
  2. **MI 测量基建**(behind `-perf`,**device 码零改**):新 `hstu_attention_bwd_perf.hpp` `time_op`(measure=false=裸 launch;perf 纯 host 字段不进 MakeKargs)+ hipEvent envelope/per-kernel + 5-GEMM TFLOPS + `benchmark.csv` 10 列 schema + `test/run_perf_baseline.py`。强调 MI 不碰设备码(byte-identical 验证背书)。
  3. **B2 causal 紧致化**(NoLocal self+cross):MAIN **1.25–1.60×**。
  4. **B3 window/local 紧致化**(WithLocal):MAIN **4.7–9.8×**(窄窗最高,window16 实测 10.4×)。诚实 Amdahl 归因(实测<模型:只砍 q-loop,K/V load+atomic 写+启动开销不减)。
  5. **★ 离线穷举校验器** `test/validate_tile_range_y`(GetTileRangeAlongY superset 校验)在 B3 **抓到并修 2 个真 under-tighten silent-wrong**(非 causal min_full 行、cross causal 大 diff+contextual)→ 修后 **1,973,278 checks GREEN**。这是「离线 gate 比对拍更早更硬挡 silent-wrong」的硬价值,放进设计动机/教训重点讲。
- **验证(四方闭合)**:MI 设备符号级 byte-identical(FORWARD 9216/9216、no_causal-NoLocal 256/256、helper 6/6)；校验器 reverse-proof(破坏收紧→校验器 FAIL=非 vacuous)；套件 **253/253 exit 0** + 2 bug 配置 PASS;co_symbols surgical(DIFF 全落 mask kernel、0 MISSING);reference/pipeline/kernel byte-identical。
- **⑤ 遗留/边界**:暂缓项 B4(grid widening,scoping 根因证伪——实测 256× 超额订阅非饥饿)、B7(hd256 占用率/VGPR,MAIN 实为 VALU-bound 非 occupancy-bound,低 ROI)、B1(group TU split,build-axis)、B6 trload、近似 sigmoid(SiLU 26% 本质开销)。这些是「明确不做、留给后续」。
- 写完按规格 §6 回报。
