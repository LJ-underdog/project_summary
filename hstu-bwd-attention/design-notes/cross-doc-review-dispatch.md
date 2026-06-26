# 文档级 review:cross-attention HTML 讲义(pane 0.3)

pane-2 写了 cross-attention 讲义 `/root/workspace/hstu-b1052-report/hstu-bwd-cross-attention-20260615.html`。独立文档级 review,只读不改文档,问题列给 lead。

## 审查清单(逐条 GREEN/RED + 证据)
1. **数字一致**:self co_symbols **870/870** byte-identical、cross sweep **32/32**、套件 **253/253**(220 self + 33 cross)、commit `4629508f`、R1 reverse-proof err 4.70 vs 1.95e-3 —— 必与素材 `docs/cross-attn-done.md` + `/tmp/hstu-bwd-design/cross-review-findings.md` + candidates 末行一致,无臆造。
2. **★ 范围诚实(最关键)**:cross = seqlen_q≠seqlen_kv 全方向(含 kv>q)× 全模式。**target_in_kv=false、独立 dO layout 未做(R7)、非方形 tile = out-of-scope** 必须标清;不得暗示 target_in_kv 或非方形支持。
3. **技术叙述准确**:① 核心洞察(mask 钉死 self 是唯一破绽、kv-offset/grid/loop/PRE 本就 cross-ready、reference 零改)讲对;② 机制(BOOL_SWITCH + 4 if constexpr cross builder、max_seqlen_kv 纯 host 字段 device 码不变)讲对;③ Option B 决策(critique 抓 dispatch grid + 无 max_seqlen_kv → kv>q silent-wrong)讲对;④ **R1 reverse-proof**(篡改 cross→self 令 cross FAIL,load-bearing)+ R4(kv>q multi-block)讲对、别讲歪;⑤ 870 vs coder 486(kentry wrapper 盲区)若提及须准确。
4. **零回归表述**:co_symbols 870/870 byte-identical、reference+pipeline 未碰、5 文件改、cross 运行时 switch 零 instance 增长。
5. **四方闭合**:coder 3-stage + reviewer 3-binary 独立(build_review/build_m7c/build_r1)+ lead 亲核 + R1 reverse-proof,呈现准确。
6. **HTML/排版**:可渲染、图正常、无占位符;**无 dingbat/emoji**、无外链。

## 产出
写 `/tmp/hstu-bwd-design/cross-doc-review.md`,逐条 GREEN/RED + 证据,RED 给位置+应改。结论:可发布/需改。完成 pane 里一句话报。
