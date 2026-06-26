# 认真 review:HSTU bwd 总览讲义 — reviewer(pane 0.2)

> 对象:`/root/workspace/hstu-b1052-report/hstu-bwd-overview-20260615.html`(pane-3 加了 5 张 SVG,只读不改)。
> 对照:`docs/HANDOFF.md` + `docs/*-done.md` + `/tmp/hstu-bwd-design/*-done.md` + `M8-INV-findings.md` + `benchmark.csv` + `git`(逐 hash/逐 commit 实查)。

## 结论:**需改 1 处(RED),改后可发布。** 其余 5 条全 GREEN,逻辑清晰、图文正确、事实扎实(连上游 5204dc75 的 `*/+` 笔误都核实属实)。

---

## 逐条

### 1. 逻辑清晰 — **GREEN**
- 叙事流顺:概述(项目是什么 + kernel 结构 + 能力边界)→ 时间线(里程碑表)→ M8 perf → 方法论/纪律 → 教训 → git 链 → 后续。无跳跃/重复/矛盾。
- 顶部 `nav-card`「如何用本页」给了按需阅读路径(整体把握/时间追踪/性能/可信度),接手者能据此快速建立全局。
- 各 §与图互相呼应,scope(MI+B2+B3,B4/B7 暂缓)在 §三 banner + §七 一致重申,无前后矛盾。

### 2. ★ 图文明确(5 张新 SVG)— **GREEN**
逐张核(viewBox/溢出/重叠/标注/呼应/配色/emoji):
- **图1 bwd 三段结构**(vb 920×210):框 x40–880 内含;PRE~9%/MAIN 84–90%/POST~2% 与 profile 一致(实算 8.6/88.7/1.6%);箭头标 D/dq_acc 正确。有 caption + §一正文「见图1」。
- **图2 能力边界矩阵**(vb 920×322):左 7 轴绿 chip(模式/注意力/激活/dtype/hdim/dQ/mask)+ 右真 reject(hdim>256)/ oos(target_in_kv/非方形/dO layout)分列,与 HANDOFF 能力边界**逐字一致**;chip x≤498、右框 x560–900 不溢出。caption + §一「见图2」。
- **图3 能力累积时间线**(vb 940×220):11 节点 x=60+i·84(末 900<940),标签**上下交替**(y78/y92)避重叠,绿点=正确性里程碑/赭点=加宽/深色=perf,与里程碑序一致。caption + §二「见图3」。
- **图4 M8 加速条形**(vb 920×232):柱值 **1.30/1.60/4.71/4.99/8.3/9.0/9.4/9.8** 与 benchmark.csv **逐格一致**;轴 1×/5×/10× + 虚线刻度;图例 B2/B3。caption + §三「见图4」。(注:正文 10.4× 的问题见 RED-1,**图4 本身正确**。)
- **图5 四方闭合+证据栈**(vb 920×250):上排 coder→reviewer→lead→对抗实验(箭头连),下排 co_symbols/poison-pad/离线校验器/reverse-proof 四工具,框 x40–880 不溢出。caption + §四正文引用 + 下方 card 逐条展开。
- 全部用 `:root` 变量配色(olive/clay/gray),**无 dingbat/emoji**(仅 `→ ↑ ≠ ≫ ·` 等排版/数学符号)。

### 3. 数字/事实核对 — **GREEN(1 RED,见下)**
全部实查属实,仅 1 处臆造:
- **能力边界**:与 HANDOFF 逐字一致 ✓。**253/253** ✓(我对抗 review 时亲跑过)。
- **git 链 12 节点**:逐 hash `git cat-file` —— `418e36ec/4bfb8e08/b0c08cba/aced5784/dc8c6b21/c79d3296/d4fb2884/bf82a1d2/1ae97750/17515fcc/4629508f/048f0a9a` **全部存在** ✓。stat「10 里程碑提交」= 12 节点减 2 集成(4bfb8e08/b0c08cba)= **精确** ✓。
- **causal 1.25–1.60× / window 4.7–9.8×**:与 benchmark.csv 一致(实算 window256 4.71/4.99、win64 8.27/8.96、win16 9.40/9.75)✓。
- **§三 banner 占用率定论**:VGPR 真值 248(rocprofv3 报半值 124 = 单位假象)、VALUBusy 41% ≫ MfmaUtil 18%、occupancy 2 blocks/CU(VGPR+LDS co-limit)、B7 低 ROI(需砍 VGPR<170 + tile 重设计)、SiLU 26% = sigmoid `exp+rcp` 2× transcendental vs softmax 单 exp2 —— **全部与 `M8-INV-findings.md` 逐条吻合** ✓(非臆造)。
- **§五 上游 5204dc75 `*/+` 笔误**:`git show 5204dc75` 实查——确为真 commit「Fix the using of num_targets[]」,且 `int i_global_batch = i_grp * num_batch_per_group * i_batch;` **确是 `*` 应为 `+` 的笔误**(扁平 batch 下标错)→ 该上游修复**确实仍不对**。**claim 精确属实** ✓。
- 图1 wall-time、PRE dot_do_o / MAIN 5-GEMM / POST 结构、5-GEMM 公式(dS/dV/dP/dK/dQ)均准确。

### 4. 20 超链接有效 + 作废 doc 排除 — **GREEN**
- 20 个非锚点链接(6 早期深读 + 14 里程碑/设计)**全部指向同目录存在文件**(逐一 `[ -f ]` 验过,0 MISS)。
- `hstu-fwd-group-maxseqlen-bug`(作废)**仅在 §二注里文字提及为作废、无超链接指向**(`grep href.*maxseqlen` = 空)—— 排除正确,且说明了作废原因(用法约定,HANDOFF §6)。

### 5. 范围诚实 — **GREEN**
- **真 reject(显式 throw)= hdim>256** 与 **out-of-scope(结构性)= target_in_kv / 非方形 tile / 独立 dO layout** 在能力 banner + 图2 + §七 三处一致区分,措辞「诚实标注、不夸大成『任意』」恰当。
- M8 scope(MI+B2+B3)与暂缓(B4 依据证伪、B7 低 ROI)诚实,§七「占用率类已判低 ROI」与 INV findings 一致。

### 6. HTML 质量 — **GREEN**
- 结构完整可渲染(DOCTYPE/head/style/body 闭合,5 SVG viewBox 正常)。
- **无占位符**(TODO/TBD/XXX/lorem = 0);**无外链**(唯一非锚点 href 均为同目录 sibling html);**无 emoji/dingbat**。

---

## RED(需改)

### RED-1 — §三正文「10.4×」臆造,且与本文自身图4/范围自相矛盾(line 223)
原文:「收紧后 MAIN 加速 causal 1.25–1.60× · window **4.7–9.8×（窄窗最高 10.4×）**」。
- **benchmark.csv 中最大 window MAIN 加速 = 9.75×(silu window16)→ 9.8×;最大 envelope 比 = 8.28×。全表无任何 10.4×**(已逐行 `python` 算 + grep 确认)。
- 「10.4×」**无任何素材来源**,且**与同句的「4.7–9.8×」及图4(最高柱 9.8×)自相矛盾**。
- **应改**:删去「(窄窗最高 10.4×)」括注,或改为与图4 一致的「(窄窗 win16 最高 9.8×)」。
- 关联(非阻塞):顶部 stat「window ~10×」是 9.8× 的合理约整(带 `~`),可保留;但 §三 的「10.4×」是确指更高峰值,属错误,必须改。

---

## 给 lead 一句话
**总览讲义逻辑清晰、5 张新 SVG 几何无溢出/标注正确/与正文呼应、20 链接全有效、范围诚实、事实逐项核实属实(VALUBusy 41%/VGPR 248/SiLU 26% 均出自 M8-INV-findings,连上游 5204dc75 的 `*/+` 笔误都核实为真)。唯一硬伤:§三「窄窗最高 10.4×」是臆造数(实际峰值 9.8×,且与本文图4 自相矛盾)。改这一处后即可发布。**
