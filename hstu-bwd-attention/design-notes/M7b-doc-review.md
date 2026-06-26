# M7b HTML 讲义 —— 独立文档级 review (pane 0.3)

> 审查对象:`/root/workspace/hstu-b1052-report/hstu-bwd-M7b-hdim-20260611.html`(546 行,5 SVG)
> 方法:只读不改文档。逐数核对素材 `M7b-done.md` + `M7b-review-findings.md` + `candidates.jsonl` 末行(M7b)+ 实查 git commit。
> 日期 2026-06-11。

## 总评:**可发布**(无阻塞 RED;3 条 minor 建议厘清,均不阻塞)

数字全部可溯源、范围诚实(最关键项)零夸大、五个技术点全讲对、零回归 byte-level 准确、HTML 无 emoji/外链/占位符可正常渲染。3 条 minor:① §9① 标题「+12 determ repro」与正文「+4 新 repro」口径混(12 是全套总数、4 是新增,171 的算术用 4);② 个别 |ref| 具体值(9.7 / 5.16)在三素材中查不到(疑出自 sweep log);③「四方」成员构成与 candidates 表述略有出入(draft vs 二进制反证),但要素齐全无失真。

---

## 逐条审查

### 1. 数字一致 —— **GREEN(2 处 minor)**

| 数字 | HTML 位置 | 素材 | 判定 |
|---|---|---|---|
| sweep **128/128** | badge/stat/§9 | done「128/128」、cand | ✓ |
| 套件 **171/171/0/0 exit0**(独立 ×2) | badge/stat/§9 | done、review「171/171 ×2」、cand | ✓ |
| 171 拆解 = 旧 106 −1 reject-hdim128 +2 guard +60 pass +4 repro | §9① 正文 | done §54 逐字(106−1+2+60+4=171) | ✓ 算术自洽 |
| **2 guard reject**(hdim=100 noncanonical、64/128 asymmetric) | §4/§9① | done §48-50、review §3 | ✓ |
| **60 M7b pass**(hdim{96,128,256}×{bf16,fp16}×10,每 hdim 3 P1-1 cross) | §9① | done §51、review §6 | ✓ |
| **12 determ repro byte-identical**(含 4 新 per-hdim,hd256 no_group+group) | stat/§9① | done §53「12 全 byte-identical」、cand | ✓(见下口径注) |
| commit **1ae97750**(基线 bf82a1d2) | header/§10 | **实查 git:存在,2026-06-11,M7b** | ✓(见下) |
| hd256 资源 **Scratch=0 / VGPR 172-184**、AGPR0/SGPR112/LDS64KB | §8 表 | done §43、review §7 逐项吻合、cand | ✓ |
| 四 tile 参数(hd64<32,128,64,…>BW 141/411/141;96 BW2=2,2,1;128 bm0=16;256 bm0=16 bn0=64) | §3 表+code | review §1 逐字符对蓝本 | ✓ |
| binary 429MB / +48 batched instance / hd64 16 instance byte-identical | §10svg/§6svg/§9③ | done §27/§64、cand | ✓ |
| mean abs err 随 hdim 升 8.6e-7→1.2e-6;max 0.0156(bf16 ULP) | §9② note | review §1「8.6e-7→1.2e-6」「0.0078/0.0156」 | ✓ |

**commit 1ae97750 —— 已实查确认**:done.md/candidates 写「未 commit / Not committed」(写于闭合前);dispatch 与 HTML 用 `1ae97750`。我 `git show 1ae97750 --stat` 证实该 commit **存在**(2026-06-11 06:55,"HSTU bwd M7b: symmetric hdim {64,96,128,256}",改面与 done 一致:新 shape.hpp + 2 dispatch + 4 entry + generate_instances + harness kN0 + 48 new instance)。即闭合后已落 commit,HTML 引用正确,非臆造。

**minor ① — §9① 标题(行 435)口径混**:`① 套件 171/171（reject-hdim128 → pass + 2 guard reject + 60 M7b pass + **12** determ repro）`。这里 `60 M7b pass` 是**新增**项,而 `12 determ repro` 是**全套件总数**(其中新增仅 **4**,见同段正文「4 个 per-hdim determ byte-identical repro」+ stat 框「12」)。并列在同一「→」拆解里易被读成「+12」,但 171 的算术是 `106−1+2+60+4`(用 4)。
- 性质:**非臆造**(4 与 12 均真实、且 dispatch item1 自身亦用「12 determ repro」措辞);标题是描述性而非等式,正文+stat 已正确。建议标题改「… + 60 M7b pass + **4** 新 determ repro(全 12 案 byte-identical)」消歧。

**minor ② — 个别 |ref| 具体值查不到**:§9② 表「bf16 SiLU multi-split |ref| **9.7 → 10.7**」「h256 combo 0.0078 vs |ref| **5.16**」。其中 10.7、0.0156、0.0078、mean 曲线均在 review/done 中可证;但 **9.7**(h64 |ref|)与 **5.16**(h256 combo |ref|)在三素材中未出现,疑出自 `runs/run-M7b-sweep.log`(未提供)。非臆造嫌疑(量级与同行自洽),但**无法据三素材核**。建议写作者标注来源 log 或 lead 抽核一眼。不阻塞。

### 2. ★ 范围诚实(最关键)—— **GREEN(强)**
全文一致钉死 symmetric,无一处暗示任意 hdim:
- subtitle「仅 symmetric,hdim_qk≠hdim_v 与非典范 hdim 是 M7c(guard 显式 throw 挡住)」✓
- TL;DR badge「范围:symmetric hdim_qk==hdim_v」✓
- §1 三 callout:in-scope symmetric{64,96,128,256};out→M7c 非对称 hdim_qk≠hdim_v;out→M7c 非典范 hdim ✓
- §4 整节 guard;§10 warn-banner「M7b 仅 symmetric … 非对称与非典范任意 hdim(需 padding)属 M7c,已由入口 guard 显式 throw 挡住。**本文档不把范围夸大成「任意 hdim」**」✓✓
对齐 done §59、review §3/残留、cand「Scope honest … guard-blocked」。**零夸大。**

### 3. 技术叙述准确 —— **GREEN(5/5)**
- ① 核心洞察(pre-M7b 硬编码 hd64 tile、MaxK 仅进符号名不选 shape → 直接加轴=silent-wrong):§2 + 图1 + formula 讲对,对齐 cand root insight + review §1。能编译/能跑/可能 PASS 的错——定性准确。✓
- ② selector`<64>`同型=hd64 零回归:§3 code(sequence/BW0/1/2/WT0/1/11 实参序/末尾 0 逐字等价 → 同类型 → byte-identical kernel,无需 if-constexpr hack),对齐 review §2。✓
- ③ guard 防非典范/非对称:§4 + 图2,规则 `qk!=v||qk!=MaxK→throw`、向上圆 silent-wrong 机理、reject 实测 throw,对齐 review §3。✓
- ④ harness kN0=(256)?64:128 修 hd256 determ:§5,bn0=64→若仍 kN0=128 则 num_splits 估值半数、与 dispatch Pipeline::kN0=64 失配→workspace 越界,对齐 review §4。机理与方向正确。✓
- ⑤ 二进制符号反证(LDS 32/48/64KB 多档→tile 真随 hdim 分化):§7 + 图4,逻辑「静默复用 hd64 则恒 32KB;多档存在=分化」讲对未讲歪,对齐 review §1🔑。✓
  - *次要*:§3/图4 把 ~48KB 同时归 hd96 与 hd128 —— review 仅证「32/48/64 三档存在」未逐 hdim 钉;但 §8 已钉 hd64=32、hd256=64,余 96/128 必落 48 档(三档两端已定,中段被逻辑唯一确定),且标「~」近似。属合理推断,非杜撰。可不改。

### 4. 零回归表述 —— **GREEN**
§9③ 表 + note 准确分类:新增 shape.hpp;改·重构(2 dispatch 取 selector+guard);改·加性(4 entry HDIM_SWITCH、generate_instances、ref.hpp 纯增、harness kN0);byte-identical(两 pipeline/bwd_kernel/reference「git diff bf82a1d2 空」+ hd64 maxk_64 16 instance 空)。逐字对齐 review §5 + done §62-67。「promoted 逻辑零碰」表述准确。✓

### 5. 四方闭合 —— **GREEN(1 处构成 minor)**
§10 图5 四盒内容均与素材吻合:
- ① coder:build 0err 429MB / sweep128 / 套件171 / stage1 检查点 / hd256 rocprof ✓(done)
- ② reviewer 独立 build_review:7/7 GREEN 0RED / tile 逐字符对蓝本 / 二进制反证 LDS 多档 / rm-rf 干净重建 171 / 库 diff 空 · Scratch=0 ✓(review 全条)
- ③ 设计稿 draft:揪出「MaxK 没选 shape」/ 定铁律 shape 先就位 / 逐 GEMM 论证无需 pad / kN0 风险预标红 / lead 闸门 ✓(cand「approved draft」+ §2/§6 主线)
- ④ lead 亲核:171 自跑 / hd64 byte-identical / guard throw 100&64/128 / hd256 softmax+combo PASS / stage1 放行 ✓(cand LEAD亲核 行逐项)

**minor ③ — 「四方」成员与 candidates 略不同**:candidates 的 four-party = `coder + reviewer + lead亲核 + 二进制反证`;HTML 图5 的四盒 = `coder + reviewer + draft + lead`(把二进制反证并入 reviewer 盒 + 独立 §7)。要素无缺失(二进制反证有专节 §7 且在 reviewer 盒醒目,draft 亦真实贡献者),无失真;仅「哪四个算一方」的归并与 cand 不同。非阻塞,如求与 cand 严格对齐可在图5 注一句「二进制反证(reviewer 主导)」。

### 6. HTML / 排版 —— **GREEN**
- **无 dingbat/emoji**:grep 全 Unicode 符号区(★/✓/✗/方块等)**零命中** —— 用 PASS/throw/status-yes(纯文字+配色)表状态,守住 skill no-emoji 铁则。✓
- 单文件可即渲染:无 `src=http`/`href=http`/`@import`/CDN/`<script`(零命中)。
- 5 SVG 开/闭各 5 配平;图1(前后对比)/图2(guard 决策)/图3(两阶段)/图4(LDS 分档柱状)/图5(闭合矩阵)结构正常,viewBox 内无明显溢出。
- 无占位符(TODO/FIXME/lorem/占位/TBD 零命中)。
- 章节 s1–s10 与 h2「1…10」顺序自洽;TOC 含 10 主项 + 5 图子项,anchor 全部命中存在 id,无死链。
- 沿用 M6/M7a 同系列 ivory/slate/clay 基线,风格统一。

---

## 给 lead 的一句话
**M7b 讲义可发布,无阻塞 RED。** commit 1ae97750 已实查存在(闭合后落地,解 done「未 commit」之惑)。仅 3 条 minor 建议厘清:① §9① 标题「+12 determ repro」改「+4 新(全 12 byte-identical)」消歧(正文/stat 已对);② §9② 的 |ref| 9.7/5.16 标来源 log 或抽核;③ 图5「四方」可注明二进制反证归 reviewer 以对齐 candidates。范围诚实、五技术点(MaxK 未选 shape 的 silent-wrong / selector<64> 同型零回归 / guard / kN0 修复 / 二进制反证)全讲对、零回归 byte-level 与 hd256 资源均准确无夸大、无 emoji。
