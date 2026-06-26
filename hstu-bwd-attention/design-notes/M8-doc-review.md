# 文档级 review:M8 perf HTML 讲义 — reviewer(pane 0.2)

> 对象:`/root/workspace/hstu-b1052-report/hstu-bwd-M8-perf-20260615.html`(只读不改)。
> 对照:`docs/M8-done.md` + 我的 `M8-review-findings.md` + `benchmark.csv` + commit `048f0a9a`(已核 = 我 review 的同一份代码,ref/两 pipeline/kernel byte-identical)。

## 结论:**可发布**(改 1 处小笔误后)。6 条:5 GREEN + 1 GREEN-with-nit。0 RED。

逐档数字、范围、技术机制全部与素材一致、无臆造、无夸大。仅 §6 有 **1 处不等号写反**(YELLOW,建议改)+ 1 处可选润色。

---

## 逐条

### 1. 数字一致 — **GREEN**
逐一核对,全部与 `benchmark.csv` / `M8-done.md` / 我的 findings 精确一致:
- causal **1.25–1.60×**、window **4.7–9.8×** ✓;套件 **253/253** ✓;校验器 **1,973,278 ALL GREEN** ✓;commit **048f0a9a** ✓(已 `git show` 核实)。
- §6 加速表逐格核 benchmark.csv:canonical sm 0.2632→0.2027(1.30×)、silu 0.3327→0.2085(1.60×)、hd256 sm 1.0325→0.7673、hd256 silu 0.9709→0.7762、window256 sm 0.2992→0.0635(4.71×)、silu 0.4193→0.0840(4.99×)、window64 0.0362/0.0468(8.3×/9.0×)、window16 0.0318/0.0430(9.4×/9.8×) —— **全中**。
- §6 TFLOPS:canonical sm 163→212(csv 163.154→211.848)✓、window16 sm **1349.75**(csv 1349.75)✓ 精确。
- §3 MI co_symbols **双口径诚实**:reviewer 自产 FORWARD 9216/9216 + no_causal-NoLocal 256/256 + helper 6/6 identical;**明确标注 coder 口径 13782/13782(457 obj 含 fwd)**—— 与我 findings 完全一致,归属清楚。
- §3 互证 rocprofv3:MAIN 0.263≈266us、envelope 87%≈87.6%、hd256 silu 0.971≈943us、SiLU 异常 1.27× —— 与 MI-stage1 一致。

### 2. 范围诚实 — **GREEN**
- scope=MI+B2+B3 在 标题/TL;DR/§1 warn-banner/§7 多处明示,且明说「本文档不把 M8 说成全面 perf 优化」。
- 暂缓项 §7 表逐项给状态+理由:**B4 依据被 critique 证伪**、**B7 阻塞于 VGPR 124-vs-248 矛盾**、INV SiLU 26%(已复现、根因待查)、B1/B5/B6/B8-10。无对成果的夸大。§7 末「这是 scope 诚实,不是遗漏」表述恰当。

### 3. 技术准确 — **GREEN**
- ① §2 profiling 实测驱动 + critique 证伪 grid-starvation:grid (16,8,2)=256 / 256 CU、16× 放大 occupancy 3.39→6.96(10.6→21.8%)、MfmaUtil 9.9→18.4%、真限制器 per-block LDS/VGPR、修正 MemUnitStalled 0.024%(非 2.4%)—— 与 draft 一致,「诚实自我证伪」框定准确。
- ② §3 MI byte-identical:time_op(measure=false→fn() 一次)、perf 字段不进 MakeKargs、dispatch 仅包 host lambda、-perf 在 FromDevice 之后 —— **与我读的源码逐条吻合**。helper 走 time_op 仍 byte-identical 的「直接铁证」框定准确。
- ③ §4 B2/B3 收紧机制 + 代码片段:`y_start=(ctx&&i_x<max_uih)?0:align_down(i_x,YTile)`、`y_end=seqlen`、非causal 原样全扫(含 P1-1)—— 与 `hstu_block_masking.hpp` 实码一致(片段是忠实简化)。silent-wrong 4 特例(contextual/min_full/cross diff/P1-1)齐全。
- ④ §5 校验器 + reverse-proof:逻辑(同 factory+oracle 穷举 superset)、2 个 bug(非causal+min_full / cross 大diff+contextual)的根因与修法、reverse-proof 两破坏的**报错文本**(`KVtile@64 attends sq=0..5 but range=[64,128)` 等)—— **与我实跑输出逐字一致**。
- ⑤ §6 Amdahl + TFLOPS:见第 4/下方 nit。

### 4. 加速表 — **GREEN(with nit,见下)**
- 表数 = benchmark.csv 前后对比,**[derived] 模型(~1.9× / ~22×)始终标 `[derived]` 且与实测分列**,§6 warn-banner 明说「实测 < [derived] 模型」,**没把模型当实测**。✓
- window「before」=全扫基线(window 大小无关,各档共用 window256 基线)说明准确。
- 图3 柱状与表数一致(1.30/1.60/4.71/4.99/8.3/9.0/9.4/9.8)。

### 5. 四方闭合 — **GREEN**
- §7 图4 呈现 coder(3-candidate,每 stage 4-gate)+ reviewer(2-build 独立 + reverse-proof + 加速分子逐档复现 + 设备码 surgical/helper 6/6)+ 校验器(离线 gate)+ lead 亲核。与实际闭合过程一致。
- **reviewer caveat 诚实写进图4 caption**(MI-only 全扫基线未独立重建,causal 1.30× 分母用 coder 记录值 + rocprof 佐证,分子精确复现)—— 与我 findings 的 caveat 完全对应,非阻塞标注得当。
- §7「9 文件(5 改 + 2 新 + benchmark.csv + 校验器扩展)」口径与 dispatch 一致(跨 ck_hstu + impl workspace)。

### 6. HTML / 排版 — **GREEN**
- 结构完整可渲染(DOCTYPE/head/style/body 闭合,SVG viewBox 正常,4 图 + 多表 + statgrid 样式齐全)。
- **无占位符**(grep TODO/TBD/XXX/lorem/FIXME/占位 = 0)。
- **无外链**(唯一 `href` 非锚点 = 同目录 sibling `hstu-bwd-cross-attention-20260615.html`,本地相对,非 external http)。
- **无 emoji/dingbat**:非 ASCII 仅 `→`(U+2192 排版箭头)、`≈ ≥ − ×`(数学符号)、`·`(middot),均技术文档常规,非象形 emoji/✓✗类 dingbat。

---

## YELLOW(建议改,非阻塞)

### N1 — §6 note 不等号写反(line 388,事实小错)
原文:「窄 window 时 MAIN 已极小(**window16 MAIN 0.032ms < PRE 0.025ms**),MAIN 不再是瓶颈」。
- **0.032 > 0.025**,不等号方向反了。window16 MAIN(0.0318)实际**大于** PRE(0.0254,canonical softmax PRE,seqlen 同 2048 故 PRE 量级一致)。
- 论点本身成立(MAIN 收到与 PRE **同量级**,不再压倒性主导),但「< PRE」是笔误。
- **应改**:`MAIN 0.032ms ≈ PRE 0.025ms`(或「已降到与 PRE 同量级」)。

### N2 — §6 window TFLOPS 可加一句界定(可选润色)
§6 note「MAIN TFLOPS 随收紧上升 … window16 softmax 收紧后 1349.75」:该值用**固定 5-GEMM(全注意力)FLOP 模型 ÷ 收紧后时间**算出,而 window 恰恰**跳过了大量 tile**,故 1349.75 实为「全注意力 FLOPs ÷ 窗口时间」,**不代表真实算力**(本质是 1/time 的 tracking 代理)。
- 文档已有总括 caveat「TFLOPS 是 tracking 不是 roofline / 非硬件利用率」,**已基本覆盖**,故仅为可选润色。
- 若要更严:对 window 档补一句「TFLOPS 因 FLOP 模型计入被跳过的 tile 而虚高,window 档的诚实指标是 MAIN ms 加速」。文档主打的正是 ms 加速,方向已对。

---

## 给 lead 一句话
**M8 HTML 讲义数字/范围/技术/闭合全部与素材精确一致、诚实标注 scope 与 caveat、排版干净无占位符/外链/emoji。唯一需改:§6 一处不等号写反(MAIN 0.032 应 ≈ 而非 < PRE 0.025);另 window TFLOPS 可选补一句界定。改 N1 后可发布。**
