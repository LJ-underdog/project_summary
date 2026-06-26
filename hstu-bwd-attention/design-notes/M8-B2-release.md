# M8 MI → Stage 2 放行(coder pane 0.1)

**lead 亲核通过 MI**:独立 co_symbols **13782/13782 byte-identical**(457 bwd obj 全集)、套件 **253/253**、benchmark.csv 42 基线行(与 rocprofv3 互证:canonical MAIN 266us/87%、hd256 943us、SiLU 1.27× 异常、window 全扫浪费已确认)。测量基线就绪。

## 放行 Stage 2 = B2 GetTileRangeAlongY 紧致化 — **causal**(draft 顶部安全要求)
现 GetTileRangeAlongY 对 causal 仍返回保守 (0,seqlen) 全扫 → MAIN 白扫 ~48% q-tile。收紧到每 kv-tile 的真实 q-range(causal:kv-tile 只被 q≥对应位置的行 attend)。

## ★ B2 silent-wrong 硬安全要求(收紧过头=永久丢梯度、不崩溃)
- 收紧范围必须是**真实所需范围的严格 kM0-aligned superset**(start `align_down`、end `align_up`/clamp)。
- **contextual 行 q_start=0**(attend 所有 q 行,`hstu_block_masking.hpp:206/463`);**min_full_attn 全 reach 行**;**cross-attn `diff_q_kv_len` 偏移**(`:89/:159`);**non-causal NoLocal 的 num_target 行不排除**(P1-1)——causal 这步先保证 causal 几何正确,但别破坏这些既有路径。

## ★ B2 验证 4 gate(全过才报 done)
1. **离线 superset 校验器 ALL GREEN**:`test/validate_tile_range_y.cpp`(M2 exhaustive 工具)——**先扩它覆盖收紧后的 causal 逻辑(+ cross diff_q_kv_len)**,再跑,**穷举证明仍 superset**(这是挡 under-tighten 的最硬 gate,无需 GPU)。
2. **对拍套件 253/253 exit 0** + 边界 stress(causal × contextual/min_full/num_target/cross/非整除 逐项,seqlen_q≠seqlen_kv 双向)。
3. **MAIN 加速 vs MI 基线**:`-perf` 跑 canonical(causal)+ 记 benchmark.csv 新行,**MAIN ms 下降**(预期 ~1.9x;实测多少记多少,别吹)。per-kernel 归因 MAIN(非 envelope 稀释)。
4. **co_symbols**:B2 改 GetTileRangeAlongY → 设备码本就变(非 byte-identical),不要求;靠 1+2+3 兜。但**未受影响的 fwd/无关符号**不应无故变。

## 完成停报 lead 亲验,再放行 Stage 3(B3 window/local)
- 报告:改面、离线校验器 GREEN 证据、253/253 + 边界 stress、MAIN 加速数(benchmark.csv 前后对比)。
- 不 commit;带 FAIL/裸 PASS/吹速度不充数(实测加速多少记多少)。
