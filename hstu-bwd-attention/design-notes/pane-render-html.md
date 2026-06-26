# 派给 pane(角色:drafter / HTML 渲染)— 把 HSTU bwd DESIGN 渲染成图文并茂 HTML

调度模式:tmux pane。把已定稿的设计文档**忠实渲染成一份图文并茂、零基础友好的 HTML**(供用户最终 review)。**不改技术内容/不重新设计**,只做"markdown→精美 HTML + 配图"。不要派 sub-teammate。

## 输入(唯一内容源,忠实呈现)
`/tmp/hstu-bwd-design/DESIGN.md`(344 行,§0–§8,已过双 review、P0=0)。所有事实/决策/行号以它为准,**不得增删改技术结论**(措辞可为可读性微调)。

## 输出
`/root/workspace/hstu-b1052-report/hstu-bwd-design-20260604.html`(与现有报告同目录)

## 样式(与同目录报告统一)
复用 `/root/workspace/hstu-b1052-report/ck-vs-hstu-bwd-20260529.html` 的整套 `<head>`/CSS（`:root` 变量 + `.tldr/.lede/.formula/.formula-block/.note-block/.callout-grid/.callout/.compare-table/.code-block[.kw/.cm/.fn]/.svg-caption/.layout/.toc/.tag/.qa` 等）。左正文 + 右 sticky TOC。配色:clay=#D97757 标 SiLU/HSTU-特有/新写,olive=#788C5D 标 softmax/复用 FMHA,slate 文字。

## 必须的图(≥5 张内联 SVG,服务"给人看")
1. **3-kernel 架构流程图**:PRE(仅 softmax)→ MAIN(恒)→ POST(恒);标条件发射 + 数据流(SiLU 路 vs softmax 路两条线,见 DESIGN §1.4)。
2. **KV 外 / Q 内 tile 布局图**:grid 沿 seqlen_k,dk_acc/dv_acc 寄存器累加,内层扫 Q(§1.2)。
3. **七阶段双路图**:STAGE1–7,高亮**只有 STAGE2/STAGE5 分 SiLU/softmax 双路**,其余 5 GEMM 照搬;标 alpha/scale_p 落点(§2)。
4. **scale 接线图/表**:alpha(STAGE2 头 + dQ/dK 尾,dV 不吃)vs scale_p(折进 p/g,softmax 不用)——防双 scale bug(§4.7)。
5. **里程碑时间线**:M0→M8,**高亮 M1 为风险闸门**(§6)。
（可选第 6 张:复用 vs 新写 的边界示意,§7。）

## 结构(对应 DESIGN §0–§8,别漏)
- 页头(eyebrow「CK · HSTU Backward」、h1「HSTU Attention Backward — GPU 实现方案」、subtitle、audience「读者:要 review 该方案的工程师」)+ **TL;DR 卡**(总策略一句话 + 已过双 review/P0=0/闸门=M1)。
- §0 摘要 + 6 个本质差异(用 callout-grid)。
- §1 架构(图1+图2)。§2 七阶段(图3 + 双路骨架用 code-block + scale 落点)。§3 mask/三模式/索引(方案 A、masked-out 置零、三模式表)。§4 工程(文件表、problem/policy 含 P1-A 硬前提、dQ 两路含 P1-E、canonical 字段表、图4 scale 表)。§5 验证(对拍流程 + 容差 + 测试矩阵)。§6 里程碑(图5)。§7 复用 vs 新写。§8 风险与**待用户拍板 U1–U4**。
- **重点突出**:把 **§8.3 U1–U4「待用户确认」做成显眼的决策卡片**(clay 边框 + 每项"建议默认值"),因为这是 review 的核心动作点。

## 铁则
- 忠实 DESIGN.md,不臆造、不改决策;行号保留(用 `.src`/code 注释呈现)。
- 零基础友好:术语首现给一句白话(可链到/复用之前报告的术语习惯)。中文为主、字段名英文。
- 自检:HTML 标签平衡(div/p/section/svg/h2/h3/table)、TOC 锚点无悬空、SVG 文字不溢出。
- 完成写 `/tmp/hstu-bwd-design/render-done.md`:小节数、SVG 数、字节、标签平衡、U1–U4 决策卡是否到位。正文写进 HTML,不在终端长输出。
