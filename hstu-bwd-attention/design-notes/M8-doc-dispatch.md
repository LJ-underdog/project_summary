# M8 perf 图文并茂 HTML 讲义 —— 文档(pane 0.3)

写 **M8 perf(MI+B2+B3)HTML 讲义**。已四方闭合 promoted、commit `048f0a9a`。

## 0. skill + 风格 + 纪律
- **必用 skill `html-report`**。风格对齐最新:`/root/workspace/hstu-b1052-report/hstu-bwd-cross-attention-20260615.html`、`hstu-bwd-M7c-hdim-pad-20260615.html`。
- 输出:`/root/workspace/hstu-b1052-report/hstu-bwd-M8-perf-20260615.html`。
- **无-emoji 铁则**(★/✓/✗/方块 dingbat 禁;→/⇒ 可)。

## 1. 素材(只读,数字必一致,不臆造)
- `docs/M8-done.md`、`docs/draft-M8-perf.md`(顶部闸门裁决)、`/tmp/hstu-bwd-design/M8-review-findings.md`(reviewer)、`M8-{MI-stage1,B2}-done.md`。
- `benchmark.csv`、candidates 末 3 行、HANDOFF M8 块、`git -C /root/workspace/ck_hstu show 048f0a9a --stat`。

## 2. 讲义必讲清(图文并茂)
1. **M8 是什么 + scope**:MI 测量基线 + GetTileRangeAlongY 紧致化(B2 causal、B3 window);**明确 scope=MI+B2+B3**,占用率类(B4/B7)暂缓。
2. **★ profiling 实测驱动(亮点)**:rocprofv3 显示 MAIN 主导 84–90%、矩阵核闲 90%、非带宽瓶颈 → 瓶颈是 GetTileRangeAlongY 保守全扫的浪费。配 profile 快照图。**critique 证伪了 scoping 的 "grid starvation 根因"**(实测 256x 超额订阅)——讲这个"数据驱动纠错"的过程。
3. **MI behind-flag byte-identical**:time_op measure=false=裸 launch、perf 纯 host 不进 MakeKargs → 设备码不变(helper kernel 走 time_op 仍逐位不变=铁证)。
4. **B2/B3 紧致化机制**:GetTileRangeAlongY 从 (0,seqlen) 全扫收紧到真实 q-band(causal/window + cross diff_q_kv_len + contextual/min_full 特例)。配全扫 vs 紧致带示意图。
5. **★★ 离线穷举校验器抓 2 个真 bug(最大亮点)**:非causal min_full、cross causal 大diff+contextual——对拍可能漏、校验器穷举抓到 → 修后 1,973,278 GREEN。reviewer 还做 reverse-proof(破坏收紧→校验器 FAIL=非 vacuous)。讲透"离线 gate 比对拍更硬"。
6. **加速 + 诚实 Amdahl**:causal 1.25–1.60×、window 4.7–9.8×(配 benchmark.csv 前后对比表);诚实标实测<模型(Amdahl:只砍 q-loop)、TFLOPS 是 GEMM-only tracking。
7. **四方闭合**:coder 3-candidate + reviewer 独立 2-build+reverse-proof + lead 亲核 + validator。
8. **暂缓项诚实**:B4(根因证伪)、B7/VGPR 矛盾、B1/trload/SiLU异常 —— 标清未做 + 理由。

## 3. 纪律
- 只据素材、范围诚实(只 MI+B2+B3)、加速实测多少写多少(别吹模型)、无 dingbat/外链/占位符、图 SVG/CSS 自包含。完成 pane 报路径,等文档级 review。
