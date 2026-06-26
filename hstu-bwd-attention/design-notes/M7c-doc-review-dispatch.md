# 文档级 review:M7c HTML 讲义(pane 0.3)

pane-2 写了 M7c 讲义 `/root/workspace/hstu-b1052-report/hstu-bwd-M7c-hdim-pad-20260615.html`。独立文档级 review,只读不改文档,问题列给 lead。

## 审查清单(逐条 GREEN/RED + 证据)
1. **数字一致**:294/294 byte-identical、220/220 套件、batched poison 168/168、group poison 96/96、commit `17515fcc` —— 必与素材 `docs/M7c-done.md` + `/tmp/hstu-bwd-design/M7c-review-findings.md` + candidates 末行一致,无臆造。
2. **★ 范围诚实(最关键)**:M7c = 任意 hdim_qk/hdim_v∈(0,256](对称+非对称+非典范 via pad)。**真 reject hdim>256、非方形 tile out-of-scope** 必须标清;不得暗示 hdim>256 或非方形。
3. **技术叙述准确**:① 核心洞察(pad 机制"已接线但死"被喂 constexpr 0)讲对;② poison-pad 正向证 OOB(NaN 填→泄漏 FAIL)+ **两条 reverse-proof 判伪**(强制 pad=false→NaN FAIL;强制 dram-view 谓词 false→store-skip FAIL)讲对、别讲歪;③ N1(store-skip 载荷=dram-view 谓词、epilogue flag 冗余)若提及须准确;④ dq_acc store-skip = poison 盲区但代码核实 production 安全,诚实标注。
4. **零回归表述**:canonical pad=0 设备符号 294/294 byte-identical;pipeline/kernel/reference 未碰、仅 4 文件改。
5. **四方闭合**:coder 4-stage + reviewer 独立 build_review 9/9 + lead 逐 stage 亲核 + reverse-proof,呈现准确。
6. **HTML/排版**:可渲染、图正常、无占位符;**无 dingbat/emoji**、无外链。

## 产出
写 `/tmp/hstu-bwd-design/M7c-doc-review.md`,逐条 GREEN/RED + 证据,RED 给位置+应改。结论:可发布/需改。完成 pane 里一句话报。
