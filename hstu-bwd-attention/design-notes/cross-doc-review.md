# cross-attention HTML 讲义 —— 独立文档级 review (pane 0.3)

> 审查对象:`/root/workspace/hstu-b1052-report/hstu-bwd-cross-attention-20260615.html`(503 行,3 SVG)
> 方法:只读不改文档。逐数核对 `docs/cross-attn-done.md` + `cross-review-findings.md` + `candidates.jsonl` 末行(Mcross)+ 实查 git(commit / diff stat)。
> 日期 2026-06-15。

## 总评:**可发布**(1 条 minor RED 建议改 + 1 条正向注;不阻塞)

数字全部可溯源,多项已 git 实证(commit 4629508f、5 文件 262/66、库未碰);技术五点全讲对;零回归 870/870 byte-identical 表述准确;四方闭合此次与 candidates 表述**精确一致**;HTML 无 emoji/外链/占位符可渲染。1 条 minor RED:§8 note-block 把 `target_in_kv=true` 与 `hdim>256` 并列为「真 reject」,但前者实为**结构性假设/out-of-scope(无运行时 guard throw)**,§1/§7 已正确标为「不做/out-of-scope」,§8 措辞应订正。1 条正向:§3「5 源文件」比 done.md 自身的「6 源文件」更准(与 git+reviewer 一致)。

---

## 逐条审查

### 1. 数字一致 —— **GREEN(多项 git 实证;含 1 正向)**

| 数字 | HTML 位置 | 素材 / 实证 | 判定 |
|---|---|---|---|
| self co_symbols **870/870 byte-identical** | badge/stat/§5①/§6/§8 | reviewer §2、done §20、cand;**git 实证库不在 diff** | ✓ |
| cross sweep **32/32**(双向,容差未松) | badge/stat/§6 | reviewer §6、done §29、cand | ✓ |
| 套件 **253/253**(220 self + 33 cross) | badge/stat/§6/§8 | reviewer §7、done §30、cand | ✓ |
| commit **4629508f**(基线 17515fcc) | header/§8 | **git show 实证存在**(2026-06-15) | ✓ |
| R1 reverse-proof **err 4.70 vs 1.95e-3**(dK 5.03、dV 5.25;\|ref\| 6.5/5.0/5.25) | §5② svg + §8 | reviewer §3 逐字 | ✓ |
| **870 vs coder 486**(漏 384 = 64 batched × 6 kentry wrapper;group 131/131 全) | §5① warn-banner | reviewer §2 逐字、done §20 | ✓ |
| **14/14** bit-repro byte-identical(含 2 xattn) | stat/§5③ | reviewer §7、cand | ✓ |
| 改面 **5 源文件 / 262 插 66 删**(params+6/batched+24/group+21/kernel+121/harness+156) | §3 表 | **git diff --stat 17515fcc..4629508f 逐项吻合**、reviewer §1 | ✓ |

**git 实证补强**:`git diff --stat 17515fcc..4629508f` = 恰 **5 源文件**(example_bwd 156 / batched 24 / kernel 121 / params 6 / group 21,262 ins 66 del),`reference_*` 与两条 pipeline **不在 diff** → §3/§5①/§8 的「reference/pipeline byte-identical」commit 级坐实。

**正向注 — §3「5 源文件」优于素材**:done.md §1 标题写「**6 源文件**」却只列 5 个(params/batched/group/kernel/example)。HTML §3 写「5 源文件」,与 git(5)+reviewer §1(「仅 5 源文件」)+candidates(「5 files」)一致 → HTML **更准**,已规避 done.md 的笔误。无需改。

### 2. ★ 范围诚实(最关键)—— **GREEN(无 overclaim)+ 1 minor RED(措辞精度)**
未暗示 target_in_kv / 非方形支持,三项 out-of-scope 全标:
- badge「范围:target_in_kv=false · 独立 dO layout 未做」✓
- §1 三 callout:in-scope seqlen_q≠seqlen_kv 双向(含 kv>q);out-of-scope target_in_kv==true(硬假设 `max_k_uih_len=seqlen_k`,`:53,:566`,「不做」);out-of-scope 独立 dO layout(R7)+ 非方形 tile ✓
- §7 诚实限制表 6 行(target_in_kv / R7 / 非方形 / R11 14min TU / co_symbols 基线补 kentry / contextual≤min) ✓
- §8「能力边界」+「真 reject:hdim>256、target_in_kv=true」

**minor RED — §8 note-block(行 478)措辞**:「真 reject:hdim > 256、target_in_kv=true」把两个**机制不同**的边界并列。
- `hdim>256`:**真 reject** —— HDIM_SWITCH else-throw,M7c 实测 throw(确凿)。
- `target_in_kv=true`:**非 reject** —— 是 cross mask 的**结构性假设**(`// assuming target_in_kv == false`),harness 根本不把 targets 放 KV;**无运行时 guard/throw**,若硬传会 silent-wrong 而非 reject。§1 callout / §7 表均正确标为「out-of-scope / 不做(结构性新路)」,唯 §8 误升为「真 reject」。
- 影响:不构成 scope-overclaim(全文从未声称支持 target_in_kv),但 §8 夸大了该边界的**强度**(暗示有硬 guard),且与 §1/§7 自相矛盾。
- 应改:§8 改为「真 reject:hdim>256(throw);out-of-scope:target_in_kv=true(结构性假设,无运行时 guard)」。

### 3. 技术叙述准确 —— **GREEN(5/5)**
- ① 核心洞察(mask 钉死 self 是唯一破绽;K/V DRAM view/grid/jagged offset/PRE/reference 本就 cross-ready;reference 零改):§2 对照表逐行对齐 reviewer §1 + done §0。✓
- ② 机制(dispatch BOOL_SWITCH + kernel 4 `if constexpr` cross builder `seqlen_kv→seqlen_k`;`max_seqlen_kv` 纯 host 字段、device MakeKargs 不读 → 设备码不变 = byte-identical 基石):§3 对齐 done §1 + reviewer §2/§4。✓
- ③ Option B 决策(critique 抓 dispatch grid 按 max_seqlen_q + 无 max_seqlen_kv → kv>q 时尾 KV 块没启动 → dK/dV 静默归零、dQ 漏贡献 = R4):§4 + 图(grid 修复前后)对齐 done §2/§3、reviewer §4。✓
- ④ R1 reverse-proof(篡改 8 处 cross builder seqlen_kv→seqlen_q、DRAM view 不动 → cross 案 err 4.70/5.03/5.25 灾难性 FAIL = load-bearing 非 vacuous)+ R4(kv=512>q=128 跨 4 KV 块对拍 PASS + 两次 byte-identical):§5②§5③ 讲对未讲歪,对齐 reviewer §3/§4。主工作树零污染(独立 worktree+cp,不 git checkout)亦如实。✓
- ⑤ 870 vs coder 486(kentry wrapper 盲区):§5① warn-banner 准确(384=64×6 kentry,group 131/131 全;coder 486 hash 逐一匹配干净 M7c = 基线真实只是数被低估),对齐 reviewer §2。✓

### 4. 零回归表述 —— **GREEN**
§3 表 + §5① + §8:co_symbols **870/870 byte-identical**(reviewer 自产基线,比 coder 486 更全)、`reference`+两 pipeline git diff 空、**5 文件**改、cross 是运行时 BOOL_SWITCH **零 instance 增长**(generate_instances 无 cross 轴)。`max_seqlen_kv` 纯 host → 设备码不变的机制解释准确。**git diff --stat 实证 5 文件 + 库不在 diff**。✓

### 5. 四方闭合 —— **GREEN(此次与 candidates 精确一致)**
§8 表四方与 dispatch/candidates 表述**逐项吻合**(不同于 M7b/M7c 的归并差异):
- coder 3-stage:Stage A 零回归重构(纯 false 腿等价)→ B harness 解钉 + max_seqlen_kv + dispatch grid → C 全矩阵对拍,每 stage 硬检查点 ✓(done §2)
- reviewer 3-binary 独立:build_review(cross)/ build_m7c(自产基线)/ build_r1(R1 篡改);co_symbols 870/870、sweep 32/32、套件 253/253、R1、R4 byte-identical、R2/R3 逐字 → **8 条清单全 GREEN** ✓(reviewer 结论)
- lead 亲核:scope 5 文件 / reference·pipeline 未改、co_symbols、cross 案(group 异构+batched)、套件自跑 ✓
- R1 reverse-proof:篡改 cross→self → cross FAIL(err 4.70 vs 1.95e-3)= mask switch load-bearing ✓
全部对齐 candidates「FOUR-PARTY CLOSURE (coder 3-stage + reviewer 3-binary independent + lead亲核 + R1 reverse-proof)」。✓

### 6. HTML / 排版 —— **GREEN**
- **无 dingbat/emoji**:grep 全 Unicode 符号区(★/✓/✗/方块等)**零命中** —— 状态用 PASS/FAIL/throw + status-yes/no 配色文字。✓
- 单文件可即渲染:无 `src=http`/`href=http`/`@import`/CDN/`<script`(零命中)。
- 3 SVG 开/闭各 3 配平;图1(self↔cross 数据流 + diff_q_kv_len 对齐)/图2(grid 修复前后 kv>q 尾块)/图3(R1 误差对照柱状)结构正常;图1 的 `marker#ah` 虽在引用后定义,SVG 按 id 解析 forward-ref 合法,可正常渲染。
- 无占位符(TODO/FIXME/lorem/占位/TBD 零命中)。
- 章节 s1–s8 与 h2「1…8」顺序自洽;TOC 含 8 主项 + 3 图子项,anchor 全部命中存在 id,无死链。
- 沿用 M6/M7a/M7b/M7c 同系列 ivory/slate/clay 基线,风格统一。

---

## 给 lead 的一句话
**cross-attention 讲义可发布。** commit 4629508f 与「5 文件 262/66、库未碰」经 git 实证;870/870 byte-identical、32/32、253/253、R1 err 4.70 vs 1.95e-3、870-vs-486 盲区、14/14 repro 全部与素材一致;技术五点全讲对、四方闭合此次与 candidates 精确对齐、无 emoji。**仅 1 条 minor RED 建议改**:§8 note-block 把 `target_in_kv=true` 与 `hdim>256` 并列为「真 reject」不准 —— 前者是无运行时 guard 的结构性 out-of-scope(§1/§7 已正确如此标),应订正措辞以免暗示有硬 reject、并消除与 §1/§7 的自相矛盾。另正向一提:§3「5 源文件」比 done.md 笔误的「6 源文件」更准。
