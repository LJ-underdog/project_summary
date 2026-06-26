# 文档级 review：M6b HTML 讲义(pane 0.2)

pane-3 写了 M6b 的 HTML 讲义 `/root/workspace/hstu-b1052-report/hstu-bwd-M6b-group-determ-20260610.html`。你做**独立文档级 review**(你熟 review,较真)。只读不改文档本身,把问题列给 lead/写作者。

## 审查清单(逐条 GREEN/RED + 证据)
1. **数字一致**:讲义里所有数字(套件 91/91、误差、case 数、commit `d4fb2884`、新增 3 锁定案等)必须与素材 `/tmp/hstu-bwd-design/M6b-done.md` + candidates.jsonl 第 12 行一致,**无臆造**。逐一核。
2. **★ 上游定性准确(最关键)**:讲义**不得**把上游 `example_hstu_attention_fwd.cpp` 的 `group_max_seqlens_q` 说成 bug。HANDOFF §6 已定性=**用法约定、非 bug**(作者澄清:`-g_max_seqlens` "can be ignored or else bigger",调用方负责 over-provision)。本次修的是 **bwd harness setup**。确认讲义口径正确、没复活那个作废的"上游 bug"措辞。
3. **根因链准确**:max_seqlen_q 低估 → PRE `dot_do_o`(grid 按 max_seqlen_q)漏算尾 token D → 垃圾 D → `dS=P*(dP−D)` 错 → 仅 softmax target 行 dQ 错(dK 稀释/dV 不含 D 不受)。判别实验(G_ref==N_ref、G_dev!=N_dev、max≥224 PASS/208 FAIL 但 grid 不变)。讲义技术叙述是否与 done.md 一致、无错。
4. **机制准确**:group determ = 复用 M6 set+split / 固定序 reduce;O1 = group entry BOOL_SWITCH_2→3 接 kIsDeterministic。
5. **诚实呈现过程**:首轮带 1 FAIL 误标 promoted 被 lead 打回 → 根因 → 修 → 对抗 formula-revert 验证。是否如实(不美化)。
6. **范围/边界**:别把 M6b 能力写超(determ 现覆盖全模式 no_group+group × SiLU/softmax)。
7. **HTML 质量**:浏览器可渲染、图(SVG/CSS)正常、无断链/占位符/溢出。

## 产出
写 `/tmp/hstu-bwd-design/M6b-doc-review.md`,逐条 GREEN/RED,RED 给具体位置 + 应改成什么。结论:可发布 / 需改。完成 pane 里一句话报。
