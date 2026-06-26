# 上游 fwd group_max_seqlens_q bug — 图文 HTML 讲义派单 (pane-3)

> 目标:写一篇**让人彻底搞懂这个 bug** 的图文 HTML。读者=想理解"为什么会错、错在哪、怎么证、怎么修"的工程师。已实测确认是真 bug。
> 用 html-report skill,风格对齐你写的 M5/M6 报告。输出:`/root/workspace/hstu-b1052-report/hstu-fwd-group-maxseqlen-bug-20260610.html`。

## 输入材料(全读)
- 实证报告(主):`/tmp/hstu-bwd-design/upstream-fwd-bug-verify.md`(三路硬证据、反双错判别、根因到行、修法)。
- issue 草稿:`/tmp/hstu-bwd-design/upstream-issue-draft.md`。
- 代码:
  - `example/ck_tile/18_hstu_attention/example_hstu_attention_fwd.cpp:850-854`(出错公式)+ L725/L776(num_targets supplement)。
  - `hstu_attention_group_forward_dispatch.hpp`:L166 GridSize / L186 mtile / L205 splitkv(max_seqlen_q 下游消费)。
  - `reference_hstu_attention_fwd.hpp` L433/L438(reference 对 max_seqlen_q 独立)。
- 关联背景:bwd 侧同款已在 M6b 修(`/tmp/hstu-bwd-design/M6b-done.md`),可对比"bwd 仅 dQ vs fwd 全局"。

## 讲义结构(图文并茂,有图有代码有数据)
1. **一句话**:group fwd 的 max_seqlen_q 被组下标错算低估 → 启动配置偏 → O/LSE 全局错(含 NaN)。default 用法即触发。
2. **背景图:group/packed 布局**:画 g=2、4 batch、组内异 seqlen 的 packed token 排布([uih|target] per batch,cu_seqlens),标出每 batch 真实 packed seqlen。
3. **max_seqlen_q 是什么、喂给谁**:画它如何驱动 GridSize / m-tile / split-kv。强调"它低估 → 整个 launch 偏,不只尾巴"。
4. **bug 本身(核心图)**:并排"正确公式 vs 错误公式";高亮 `num_targets[i_grp]` 用**组下标**索引**per-batch** 数组的错位——画一个具体例子(seqlens=100, targets=0,0,0,200, g=2):group1 应取 batch3 的 300,却取了 num_targets[1]=0 → 算成 100,漏 200。用箭头画"取错了哪个元素"。
5. **为什么全局错而非仅尾 token**:max_seqlen_q 进 GridSize/mtile/splitkv → 所有 batch 的 tile 映射/归约都偏 → 实测 b0/b1/b2 即便在网格内也错 + NaN。
6. **怎么证明是真 bug(方法论亮点)**:讲"双错相等陷阱"+ 如何排除(reference 对 max_seqlen_q 独立、byte-identical 实测)+ 三路证据表(trigger/control/真改公式)。这段教读者"怎样严谨地确认一个 bug"。
7. **修法**:per-group `max_b(seq+tgt)+ctx` patch + assert;对比错误式"replace(可低估)" vs 正确式"max(只增不减)"。
8. **影响面 + 对比 bwd**:fwd O+LSE 全局(更重)vs bwd 仅 PRE-D 的 dQ target 行;触发条件;default 派生即中。

## 要求
- 图可用 SVG/ASCII/HTML table;关键数字(87.5%/84.6%/NaN、100 vs 300)要醒目。
- 代码片段引真实行号,与磁盘一致。
- 诚实、精确,不夸大;这是给人决策"要不要报上游/怎么修"的依据。
- 只读不改源码。写完报 lead(给文件路径 + 一句话自评)。
