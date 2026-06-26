# HSTU bwd 里程碑讲义系列 — 统一体例规格(所有 pane 必读)

> lead(pane0)派单。你被指派写**其中一篇**里程碑讲义,把它做成与 M0 讲义**同一体例**的、自包含的图文 HTML。本文件是硬规格,违反任一条都会被 lead 退回重做。

## 0. 你的产物(一句话)
为指定里程碑写一篇 `hstu-bwd-<Mx>-<slug>-20260625.html`,放在 `/root/workspace/hstu-b1052-report/`,体例**克隆** M0 讲义 `hstu-bwd-M0-scaffolding-20260625.html`,内容讲清「这个里程碑改了什么、为什么这样改、怎么验证的」。

## 1. 第一步:读模板(必做,别跳)
1. **完整读** `/root/workspace/hstu-b1052-report/hstu-bwd-M0-scaffolding-20260625.html` —— 这是体例黄金标准。**整段 `<style>` 原样复制**(配色/字体/卡片/code-block/svg-wrap/badge/tag/statgrid/toc 全部沿用,不要自创 CSS)。
2. 结构必须含这些块(顺序同 M0):
   - `header.page-head`:eyebrow + h1(`HSTU Attention Backward — <Mx>(中文名):一句话主旨`)+ subtitle(点明本里程碑 scope + 行号锚定哪个 commit)+ audience 行。
   - `.tldr`:一句话看懂 + `.badges`(scope/状态)+ `.statgrid`(4 个关键数字,如改动文件数、对拍案数、加速比、新增实例数等,挑该里程碑最有代表性的)。
   - `① 总览`:`callout-grid`(3-4 张卡概括「改了哪几件事」)+ 一条 `note-block` 点出该里程碑**最精妙/最易错的设计点**。
   - `② 最佳阅读顺序`:一张 **SVG 依赖流图**(用 M0 的 `svg-wrap`+`viewBox` 写法,自己画该里程碑的文件/概念依赖)+ 一张 `compact` 表(逐文件:读它回答什么问题 + 重点行号)。
   - `③ 逐文件 review`:每个改动文件一个 `h3`(带 `tag-new`/`tag-mod`/`tag-reuse` 标签)+ `ul` 列要点 + 至少 1 个 `code-block` 展示关键代码片段(带 `.ttl` 文件:行号标题、`.kw/.str/.cm/.fn/.hl` 高亮)。
   - `④ 设计动机小结`:`ol` 提炼 3-5 条「为什么这么做」。
   - `⑤ 遗留 / 边界`:`compact` 表,本里程碑明确不做的、留给后续哪个 M。
   - sticky `nav.toc`:目录锚点,和 section id 对应。
3. **自包含**:单文件,内联 CSS,无外部依赖(SVG 手写、不引图片/JS/CDN)。中文 `lang="zh-CN"`,UTF-8。

## 2. 行号锚定(★最硬的铁律,错一处整篇作废)
- 每条代码引用的行号**必须锚定该里程碑自己的 commit**,用 `git show <你的commit>:<相对仓库路径>` 核对,**不是** HEAD、不是工作树。
- 仓库:`/root/workspace/ck_hstu`。例:`cd /root/workspace/ck_hstu && git show <commit>:example/ck_tile/18_hstu_attention/hstu_attention_bwd_params.hpp | sed -n '17,107p'`。
- 想知道「这个里程碑到底改了哪些文件」:`git show --stat <commit>`;想看具体改动:`git show <commit> -- <file>` 或 `git diff <commit>^ <commit> -- <file>`。
- **每个写进 HTML 的 `<code>:行号</code>` 都要回 `git show` 验证过**指向你说的那段。行号是 point-in-time 快照,锚死 commit 后就别再刷成别的值。

## 3. 反幻觉铁律(lead 会抽查,编造直接退回)
- 不确定的事实(行号、函数名、数值、加速比、对拍案数)**一律回源核**:源码用 `git show`,里程碑结论用对应 `M*-done.md` + `candidates.jsonl` 里该里程碑那条,硬件/数学口径用 `/tmp/rocm-ref/`(别臆造 CDNA4 参数)。
- **旧讲义只能当叙事/结构参考,不能当事实来源**——旧 HTML 里有已知 stale 行号和个别错(见各自派单卡的提示)。凡引用必回 `git show` 重核。
- 数值(加速比、误差、case 数)以 `candidates.jsonl` 对应条目 + `M*-done.md` 为准,照抄并标出处。

## 4. 复用旧讲义(用户明确:旧内容可参考保留)
- 你这篇对应的**旧 HTML**(见派单卡)可借鉴其叙事、SVG 思路、措辞——但**结构改造成 M0 体例**、**所有行号/数值重新核**。
- 旧讲义不删、不动(用户要保留),你只**新建**带 `-20260625` 日期的新文件。

## 5. 里程碑 → commit 映射(权威,以此为准;HANDOFF 里的旧 hash 已 stale 勿用)
| Mx | commit | 中文名 / slug |
|----|--------|---------------|
| M1 | `1b3c90b4` | SiLU MAIN 闸门 / silu-gate |
| M2 | `9d129c88` | HSTU 5 因子 mask / mask |
| M3 | `94174bd9` | jagged 变长 / jagged |
| M4 | `3573f083` | group 模式 / group |
| M4b| `180a8acb` | 修 P1-1(causal=0+target)/ p1-1-fix |
| M5 | `476bc16a` | softmax 路径 / softmax |
| M5b| `f7db567d` | group softmax / group-softmax |
| M6 | `48673726` | deterministic dQ / deterministic |
| M6b| `ecda0f06` | group deterministic / group-determ |
| M7a| `8b1fab06` | fp16 加宽 / fp16 |
| M7b| `c9fe2891` | 对称 hdim{64,96,128,256} / hdim |
| M7c| `fc13643e` | 非对称/非典范 hdim via pad / hdim-pad |
| cross | `f2f55622` | cross-attention seqlen_q≠kv / cross-attention |
| M8 | `a86529dc` | perf(MI+B2+B3)/ perf |

## 6. 完成标准 + 回报
- HTML 写完后,自查:① `<style>` 与 M0 一致;② 五大 section + tldr + toc 齐全;③ 每条行号都 `git show` 核过;④ 数值有出处;⑤ 浏览器能开(标签闭合)。可用 `python3 -c "import html.parser..."` 或简单 grep 自查 `</html>` 结尾、无半截标签。
- 写一行产出说明追加到你的派单卡末尾(或回 pane 末尾打印):**输出文件路径 + 你核过的 commit + 写了几个 section + 引用了哪些源文件**,供 lead review。
- **不要** commit、不要碰 git、不要改别的文件。只新建你那一篇 HTML。
