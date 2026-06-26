# 文档级 review:M8 perf HTML 讲义(pane 0.2)

你刚做完 M8 对抗 review,最熟。独立文档级 review `/root/workspace/hstu-b1052-report/hstu-bwd-M8-perf-20260615.html`,只读不改文档,问题列给 lead。

## 清单(逐条 GREEN/RED + 证据,对照 docs/M8-done.md + 你的 M8-review-findings.md + candidates)
1. **数字一致**:causal 1.25–1.60×、window 4.7–9.8×、253/253、校验器 1,973,278 GREEN、commit 048f0a9a、MI co_symbols(你的口径)—— 无臆造。
2. **范围诚实**:scope=MI+B2+B3;暂缓项(B4 根因证伪/B7 VGPR 矛盾/B1/trload/SiLU 异常)标清、无夸大。
3. **技术准确**:① profiling 实测驱动 + critique 证伪 grid-starvation;② MI behind-flag byte-identical(time_op/host-only);③ B2/B3 GetTileRangeAlongY 收紧机制;④ 离线校验器抓 2 bug + reverse-proof;⑤ Amdahl 诚实(实测<模型)、TFLOPS=GEMM-only tracking 非 roofline。
4. **加速表**:benchmark.csv 前后对比数与素材一致;别把模型 [derived] 当实测。
5. **四方闭合**:coder + reviewer 独立 2-build + lead 亲核 + validator,呈现准确。
6. **HTML/排版**:可渲染、图正常、无占位符、无 dingbat/emoji、无外链。

产出 `/tmp/hstu-bwd-design/M8-doc-review.md`,逐条 GREEN/RED,RED 给位置+应改。结论可发布/需改。完成 pane 报。
