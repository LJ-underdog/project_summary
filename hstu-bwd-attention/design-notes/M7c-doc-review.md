# M7c HTML 讲义 —— 独立文档级 review (pane 0.3)

> 审查对象:`/root/workspace/hstu-b1052-report/hstu-bwd-M7c-hdim-pad-20260615.html`(544 行,3 SVG)
> 方法:只读不改文档。逐数核对 `docs/M7c-done.md` + `M7c-review-findings.md` + `candidates.jsonl` 末行(M7c)+ 实查 git(commit / diff stat)。
> 日期 2026-06-15。

## 总评:**可发布**(无阻塞 RED;2 条 minor 观察,均不阻塞)

数字全部可溯源且多项已用 git 实证(commit 17515fcc、4 文件 336/118、库未碰);范围诚实(最关键项)零夸大、四 reject/out-of-scope 标清;四个技术点(死代码激活 / poison + 双 reverse-proof / N1 归因 / dq_acc 盲区)全讲对未讲歪;零回归 294/294 byte-identical 表述准确;HTML 无 emoji/dingbat/外链/占位符可正常渲染。2 条 minor:① §8① 「M7b 170 基」与 M7b 自身套件 171 的跨文档口径(忠实镜像 done.md、且 170+50=220 自洽);② 「四方」成员归并与 candidates 略异(draft vs reverse-proof),要素齐全无失真。

---

## 逐条审查

### 1. 数字一致 —— **GREEN(多项 git 实证)**

| 数字 | HTML 位置 | 素材 / 实证 | 判定 |
|---|---|---|---|
| canonical **294/294 byte-identical** | badge/stat/§7 | done §47、review §2、cand;**git 实证库文件不在 diff** | ✓ |
| 套件 **220/220/0/0 exit0**(独立重建) | badge/stat/§8 | done §38、review §8、cand | ✓ |
| batched poison **168/168** | badge/stat/§8② | done §44、review §3、cand | ✓ |
| group poison **96/96** | badge/stat/§8② | done §45、review §3、cand | ✓ |
| commit **17515fcc**(基线 1ae97750) | header/§9 | **git show 17515fcc 实证存在**(2026-06-15,M7c) | ✓ |
| 改面 **4 文件 / 336 ins / 118 del**(shape+20/batched71/group60/harness303) | §9 | **git diff --stat 1ae97750..17515fcc 逐项吻合** | ✓ |
| co_symbols **66 obj(64 batched+2 group)/294 符号**;group **70/70**(2×35);~576 pad-true | §7 | done §20-21、review §2/§6 | ✓ |
| 2 条 reverse-proof | badge/§4 | review §3#1 §4#2、cand | ✓ |
| reviewer **9/9 GREEN** | §9 | review TL;DR、cand「9/9」 | ✓ |
| 50 poison 案 × 4 marker;真 reject hdim>256 exit-6 | §8① | done §39-40、review §8 | ✓ |
| bf16 dQ≤0.016 vs |ref|6.7–8.8;softmax~1e-4;fp16≤4e-3 | §8③ | done §46、review §8 | ✓ |

**git 实证补强**:commit `17515fcc` 我亲 `git show` 确认存在(done.md 写「未 commit」是闭合前态,闭合后已落地);`git diff --stat 1ae97750..17515fcc` = **恰 4 文件**(example_bwd.cpp 303 / batched dispatch 71 / shape.hpp 20 / group dispatch 60,336 ins 118 del),**pipeline/kernel/reference 不在 diff** → §7/§9 的「库 git diff 空」「仅 4 文件」commit 级坐实,非臆造。

**minor ① — §8① 「M7b 170 基」跨文档口径**:HTML「= M7b 170 基 + 50 poison-asserted M7c = 220」逐字镜像 done.md §39;但 M7b 讲义自身套件为 **171**(其 reject-hdim128 等)。M7c 把 base 记 170(2 旧 skip/reject 重分类,见 done §22「171→172(2 skip)」)。属源内重分类细节,HTML 忠实于 done.md 且 170+50=220 自洽,**非杜撰**;若要跨 M7b/M7c 完全一致可注一句「M7b 套件 171,其中 reject-hdim128 在 M7c 转 poison-pass / 2 案改判」。不阻塞。

### 2. ★ 范围诚实(最关键)—— **GREEN(强)**
全文一致钉死 (0,256],真 reject 与 out-of-scope 分列,无一处暗示 hdim>256 或非方形:
- subtitle「hdim>256 仍真 reject、非方形 tile out-of-scope」✓
- TL;DR badge「范围:hdim ∈ (0,256] 任意;hdim>256 真 reject」✓
- §1 三 callout:in-scope 任意(0,256](对称+非对称+非典范);真 reject hdim>256(else-throw,实测 512/300/asym-512/group-512 全 throw);out-of-scope 非方形 tile(bhdq≠bhdv)→M8 ✓
- §9 warn-banner「真 reject:hdim>256 … Out of scope:非方形 tile … **本文档不把范围夸大成「任意 hdim」**」✓✓
对齐 done §49-59、review §9、cand。**零夸大。**

### 3. 技术叙述准确 —— **GREEN(4/4)**
- ① 核心洞察(pad 机制「已接线但死」—— 喂 constexpr 0 + guard 挡门):§2 + 图1 + code,对齐 cand root insight + review §1。定性准确(机制本在 kernel/pipeline,M7c 只换 flag 来源)。✓
- ② poison 正向证 OOB + **两条 reverse-proof**:§4 + 图2 讲对未讲歪 —— #1 强制 pad_qk/pad_v=false→NaN 泄漏→`[FAIL]×3`(证 load-zero 判伪);#2 仅关 dk/dv 视图谓词 `sequence<false,false>`、loads 不动→grad 仍 PASS 但 store-skip FAIL(隔离证 store-skip 非 vacuous)。逐字对齐 review §3#1/§4#2。✓
- ③ N1(store-skip 载荷=dram-view 谓词、epilogue flag 冗余):§5 表三行(正常 PASS / 只关 epilogue flag 仍 PASS / 只关视图谓词 FAIL)精确对齐 review §4 步骤1-2 + N1。措辞「非缺陷,归因精确化」准确。✓
- ④ dq_acc store-skip = poison 盲区但 production 安全:§6 诚实标注(over-alloc 吸收 OOB 写→无法直证;同源谓词 code 核实 + reverse-proof#2 已证该谓词有效 + exact-alloc 真实 stride 契约同 M7b)。对齐 done §55-57 + review §5 + cand。✓

### 4. 零回归表述 —— **GREEN**
§7「canonical false-false leg 实例化出与 M7b 同类型 → 294/294 设备符号 byte-identical」+ §9「库 pipeline/kernel/reference git diff 空、仅 4 文件」。逐字对齐 review §1-2 + cand,且 **git diff --stat 实证仅 4 文件、库不在 diff**。byte-identity 自动成立的解释(TileFmhaBwdTraits pad 形参 index_t,false 隐式转 0 字面量,同类型)准确。✓

### 5. 四方闭合 —— **GREEN(1 处归并 minor)**
§9 图3 四盒内容均与素材吻合:
- ① coder 4-stage:Stage0 基线294符号 / Stage1 重构零回归 / Stage2 batched poison 168 / Stage3 group poison 96 / Stage4 套件220 永久化 / harness poison-pad ✓(done Stage0-4)
- ② reviewer 独立 build_review:9/9 GREEN 0RED / rm-rf 重建 220/220 / canonical 294/294 / 双向 poison+hdim=100 / REVERSE-PROOF #1+#2 / dq_acc code-audit · N1 归因 ✓(review 全条)
- ③ draft:6 路并行+critique / 逐 GEMM pad·OOB / R1-R11 红旗 / guard+switch 同 commit 铁律 / 5 must-fix / lead 闸门 ✓(cand「approved draft」+ 主线;draft 细节未独立提供但与流程一致)
- ④ lead 逐 stage 亲核:byte-identity 294/294 / batched 168 / group 96 / suite 220 / dq_acc 代码核实 / 放行 commit ✓(cand LEAD亲核 逐项)

**minor ② — 「四方」成员与 candidates 略异**:candidates four-party = `coder 4-stage + reviewer + lead + 2 reverse-proofs`;HTML 图3 四盒 = `coder + reviewer + draft + lead`(reverse-proof 并入 reviewer 盒 + 独立 §4)。要素无缺失(reverse-proof 有专节 §4 且在 reviewer 盒醒目,draft 真实贡献者),无失真;仅归并不同(与 M7a/M7b 讲义一致的处理)。
- *可选*:reviewer N3(误 `git checkout` 回退 batched dispatch、已逐 hunk 重建并 294/294 复验、coder 比对确认)是「诚实文化」一例,讲义未提。非必需(N3 是已闭环的 working-tree 事故、非 M7c 特性属性),如愿强化诚实叙事可在 reviewer 盒注一句「N3 事故已闭环」。不阻塞。

### 6. HTML / 排版 —— **GREEN**
- **无 dingbat/emoji**:grep 全 Unicode 符号区(★/✓/✗/方块等)**零命中** —— 状态用 PASS/FAIL/throw + status-yes/no 配色文字表达,守 no-emoji 铁则。✓
- 单文件可即渲染:无 `src=http`/`href=http`/`@import`/CDN/`<script`(零命中)。
- 3 SVG 开/闭各 3 配平;图1(死代码→激活)/图2(poison + 两条 reverse-proof)/图3(四方闭合)结构正常,viewBox 内无明显溢出。
- 无占位符(TODO/FIXME/lorem/占位/TBD 零命中)。
- 章节 s1–s9 与 h2「1…9」顺序自洽;TOC 含 9 主项 + 3 图子项,anchor 全部命中存在 id,无死链。
- 沿用 M6/M7a/M7b 同系列 ivory/slate/clay 基线,风格统一。

---

## 给 lead 的一句话
**M7c 讲义可发布,无阻塞 RED。** commit 17515fcc 与「4 文件 336/118、库未碰」均经 git 实证;294/294 byte-identical、220/220、batched 168/168、group 96/96、双 reverse-proof、N1 归因、dq_acc 盲区诚实标注全部数字与素材一致。仅 2 条 minor(均不改也可发):① §8① 「M7b 170 基」是 done.md 内部重分类口径(与 M7b 讲义 171 跨文档不一致,但自洽),可注一句来由;② 图3「四方」把 reverse-proof 并入 reviewer 盒、用 draft 充第四方(与 candidates 的成员表述不同但要素齐全),可选注明 N3 事故已闭环以强化诚实叙事。范围诚实、技术四点全讲对、零回归 commit 级坐实、无 emoji。
