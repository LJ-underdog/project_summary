# M6b 图文并茂 HTML 讲义 —— 文档 pane(pane 0.3)

补 **M6b 的图文并茂 HTML 讲义**(HANDOFF §3 铁律③:解释性文档 HTML、放 `/root/workspace/hstu-b1052-report/`)。M6b 已 promoted、commit `d4fb2884`,但当时没写 HTML(只有 md 内部报告),现补上。这是个有料的里程碑(group determ + 修 O1 + 抓出并修一个 pre-existing harness bug)。

## 0. 用 skill + 看历史风格
- **必用 skill `html-report`**(`/root/.claude/skills/html-report/SKILL.md`)。
- 对齐风格:读 `/root/workspace/hstu-b1052-report/hstu-bwd-M6-deterministic-20260609.html`(M6 是 M6b 的前置,同系列直接续)+ `hstu-bwd-M5b-group-softmax-20260608.html`。
- 输出:`/root/workspace/hstu-b1052-report/hstu-bwd-M6b-group-determ-20260610.html`。

## 1. 素材(只读,别臆造,数字必须与素材一致)
- `/tmp/hstu-bwd-design/M6b-done.md`(coder:根因/修复/验证全表)。
- `/tmp/hstu-bwd-design/M6b-review-findings.md`、`M6b-fix-review-findings.md`、`M6b-fix-approval.md`(review + lead 批复)。
- candidates.jsonl 第 12 行 M6b(promoted reason,含根因/判别实验/防御)。
- HANDOFF.md M6b 块 + §6 那条"上游 fwd group_max_seqlens_q 非 bug"的定性(讲义里要呼应:harness 修的是 bwd harness setup,上游 fwd 是用法约定非 bug,别混)。
- 代码:`git -C /root/workspace/ck_hstu show d4fb2884 --stat`。

## 2. 讲义必讲清(图文并茂)
1. **M6b 是什么**:把 M6 的 deterministic dQ 机制扩到 group(M4/M5b group × M6 determ)+ 修 O1(group entry hardcode false → determ 不可达静默 atomic)。
2. **group determ 机制**:复用 M6 POST reduce + set+split 机制(每 KV-block plain-store 到自己 split 副本 base+=i_tile_n*split_stride → POST 固定升序 reduce → 构造上 bit-reproducible)。配 split/reduce 示意图(可续用 M6 文档的图风格)。
3. **O1 修复**:group entry `BOOL_SWITCH_2→BOOL_SWITCH_3`(接 kIsDeterministic 真轴),否则 group+determ 静默落 atomic、不可复现。
4. **★ 高潮:抓出并修 harness `group_max_seqlens_q` 老洞**(这段最值得图解):
   - 现象:1 个配置 dQ FAIL(0.0626),仅 softmax target 行错,dK/dV 对。
   - 判别实验:group vs no_group-jagged → `G_ref==N_ref`(reference 对)、`G_dev!=N_dev`(GPU 侧);`-g_max_seqlens` override max≥224→PASS、208→FAIL 但 grid.x 不变 → 非 grid、是 max_seqlen_q 数值。
   - 根因:harness `group_max_seqlens_q[i_grp]=...+num_targets[i_grp]` 用**组下标**索引**逐 batch** 数组 → 组内最长 packed batch≠batch[i_grp] 时低估 → PRE `dot_do_o`(grid 按 max_seqlen_q)漏算尾 token D → 垃圾 D → `dS=P*(dP−D)` 错 → dQ 错。配数据流图(max_seqlen_q 低估 → PRE 漏算 → D 垃圾 → dQ 错)。
   - 性质:atomic==determ 逐位相同 ⇒ 非 determ/库,纯 harness setup bug;库 kernel/dispatch/reference 全对。
   - 修复:公式改组内逐 batch max + HSTU_CHECK 守卫 + PRE 前置条件注释(不 memset 兜底)。
5. **过程纪律(诚实,值得讲)**:coder 首轮带 1 FAIL 标 promoted 被 lead 打回 → 根因定位 → 批准修复 → reviewer 对抗 formula-revert 证回归案有效 → 四方闭合。这段体现项目"独立复核"文化,如实呈现。
6. **教训**:又一条 P1-1 式覆盖洞(group 测试矩阵此前没覆盖"同组多 batch 异 seqlen + 长 batch 大 target + window")。
7. **数据**:套件 91/91 exit 0;新增 3 个精确触发锁定案。

## 3. 纪律
- **只据素材,不臆造**;数字/case 与 done.md 一致。
- **别把上游 fwd 说成 bug**——HANDOFF §6 已定性为用法约定非 bug,讲义要准确(修的是 bwd harness)。
- 图用 SVG/CSS 自包含单文件。
- 完成后 pane 里一句话报路径,等交叉 review。
