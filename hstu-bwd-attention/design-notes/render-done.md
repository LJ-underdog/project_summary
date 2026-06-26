# HTML 渲染完成 — HSTU bwd DESIGN

## 输出
`/root/workspace/hstu-b1052-report/hstu-bwd-design-20260604.html`

## 规模
- 字节数:**86,646 bytes**
- 小节(section/h2):**9**(§0 摘要 / §1 架构 / §2 七阶段 / §3 mask / §4 工程 / §5 验证 / §6 里程碑 / §7 复用vs新写 / §8 风险)
- h3 子节:**31**
- 表格:**12**

## SVG(内联,图文并茂)— 共 6 张(要求 ≥5)
1. 图1 **3-kernel 架构流程图** — PRE(仅softmax)→MAIN(恒)→POST(恒),双数据流(SiLU clay 路省 PRE/LSE/O/D vs softmax olive 路),标条件发射规则(§1.4/§1.1)。
2. 图2 **KV 外 / Q 内 tile 布局图** — grid 沿 seqlen_k,dk_acc/dv_acc 寄存器累加,内层扫 Q,标 GetTileRangeAlongY 无条件调用实证(§1.2)。
3. 图3 **七阶段双路图** — STAGE1–7,高亮仅 STAGE2/STAGE5 分 SiLU/softmax 双路,其余 5 GEMM 灰底照搬;底部专栏标 alpha/scale_p 落点(§2)。
4. 图4 **scale 接线图** — alpha(STAGE2头+dQ/dK尾,dV不吃) vs scale_p(折进p/g,softmax不用),含「防双scale bug」红框 + host接线框(§4.7),后附固化表。
5. 图5 **里程碑时间线** — M0→M8 圆点时间轴,M1 放大为 clay 闸门节点 + 180px 详情框(验收 ScratchSize=0/VGPR + 一次性验证 policy复用/SiLU特化/留g)(§6)。
6. 图6(可选)**复用 vs 新写边界示意** — 左 olive 复用面 / 右 clay 新写 9 点(§7)。

## 结构对应(DESIGN §0–§8 全覆盖,无漏)
- 页头:eyebrow「CK · HSTU Backward」+ h1 + subtitle + audience「读者:要 review 该方案的工程师」+ **TL;DR 卡**(总策略一句话 + 三枚徽章:已过双review / P0=0 / 闸门=M1)。
- §0:摘要 + 6 本质差异(callout-grid,clay/olive 配色 + 图例)。
- §1:1.1 3-kernel(图1) / 1.2 tile(图2) / 1.3 GridSize表 / 1.4 数据流。
- §2:2.1 6步↔7stage↔5GEMM(图3+表) / 2.2 逐阶段(含 P1-1 NaN守卫) / 2.3 双路骨架(code-block,高亮 alpha/scale_p) / 2.4 留g(P1-B) / 2.5 fwd副产物契约。
- §3:3.1 方案A(GetTileRangeAlongY/IsEdgeTile,P1-C/P1-D) / 3.2 masked-out置零统一口径 / 3.3 5因子对称(D2) / 3.4 三模式表 / 3.5 hdim_qk≠hdim_v。
- §4:4.1 文件表 / 4.2 problem/policy(**P1-A 硬前提做成 gate-banner**) / 4.3 dispatch / 4.4 dQ两路(P1-E) / 4.5 instances/CMake / 4.6 canonical字段表 / 4.7 scale(图4+表)。
- §5:5.1 对拍流程 / 5.2 容差表 / 5.3 测试矩阵 / 5.4 deterministic+边界(离线超集校验)。
- §6:里程碑(图5 + 全表,M1 行 clay 高亮)。
- §7:复用vs新写(图6 概括 §7 九点)。
- §8:**8.3 U1–U4 决策卡前置置顶**(见下) / 8.1 已决D1–D7 / 8.2 建议R1–R6 / 8.4 冲突C1–C3 / 8.5 review落实表。

## U1–U4「待用户确认」决策卡 — 已到位且显眼 ✅
- 做成 `.decision-card`:**2px clay 边框 + clay 阴影 + 2×2 网格**,放在 §8 **最前**(置顶,review 核心动作点)。
- 每卡含:U编号 / 问题标题 / **clay 高亮「建议默认值」框**(U1 不支持dbias / U2 atomic / U3 SiLU全覆盖优先 / U4 nhead_ratio_qk=1) / 理由(why)。
- 上方 gate-banner 强调「双 review 认可默认、按此推进无阻塞」。

## 自检
- **标签平衡**:div 198/198、section 9/9、svg 6/6、h2 9/9、h3 31/31、table 12/12、p/nav/main 均 OK(修复了 .page 包裹 div 漏闭 1 处)。
- **TOC 锚点**:全部 href="#..." 均有对应 id,**无悬空**(脚本校验通过);左正文 + 右 sticky TOC,§8 中 8.3 排在 TOC 8.1 之上以呼应正文置顶。
- **SVG 文字**:估算最坏右边缘 ≈841 < viewBox 880,无溢出。
- **忠实度**:技术结论/行号/裁决 100% 源自 DESIGN.md(行号以 `.src`/code 注释呈现),仅措辞为可读性微调,未增删改技术结论;配色 clay=#D97757(SiLU/HSTU特有/新写)、olive=#788C5D(softmax/复用FMHA)、slate 文字,复用同目录报告整套 CSS。
