# M7a HTML 讲义 —— 独立文档级 review (pane 0.3)

> 审查对象:`/root/workspace/hstu-b1052-report/hstu-bwd-M7a-fp16-20260611.html`(472 行,4 SVG)
> 方法:只读不改。逐项与素材 `M7a-done.md` + `M7a-review-findings.md` + `candidates.jsonl` 末行(M7a)逐数核对。
> 日期 2026-06-11。

## 总评:**可发布**(1 条 minor RED 建议改,1 条可选 clarity 提示;均不阻塞)

数字全部可溯源、无臆造;范围诚实(最关键项)零夸大;fp16-ULP 反证 / 容差双实验 / 零回归 byte-level / 四方闭合叙述均准确;HTML 可渲染、无外链/占位符/溢出。唯一须改:§6 一处「91 个 bf16 历史案全 PASS」与 §7 自身「91 − 1」口径冲突(应为 90),非臆造(镜像自 reviewer §6 措辞),但内部不自洽,建议订正。

---

## 逐条审查

### 1. 数字一致 —— **GREEN(1 处 minor RED,见末)**
逐一核对,全部命中素材:

| 数字 | HTML 位置 | 素材 | 判定 |
|---|---|---|---|
| fp16 sweep **66/66** | header/TL;DR/§7 stat/§7① | done §2、cand「66/66」 | ✓ |
| 套件 **106/106/0/0 exit0** | TL;DR/§7 stat/§7③ | done §5、review §1 | ✓ |
| 106 拆解 = 旧 91 − 1 reject-fp16 + 14 fp16 pass + 2 repro | §7③ | done §5 逐字 | ✓(算术自洽) |
| 容差 fp16 **5e-3/1e-2** vs bf16 **2e-2/5e-2** | TL;DR/§3 code/§3 note/§5 表 | done §2、review §2、code:151-153 | ✓ |
| SiLU **rel~1e-3** / softmax **rel~1e-4** | §7② 表 | done §2 逐字 | ✓ |
| determ multi-split(seq512)err~4e-3 rel~4e-4 | §7② 表 | done §2「~4e-3 rel~4e-4」 | ✓ |
| max_abs_err **6.10352e-05=2⁻¹⁴**、dV **1.22e-4=2⁻¹³** | §4 formula+证据1 | review §2/§3 逐字 | ✓ |
| ~160× headroom | §5 表 | review §2 | ✓ |
| 实验A 放宽 2e-2/5e-2 PASS;实验B 收紧 **1e-3/5e-4** PASS(紧 5–20×) | §5 表+gate | review §2 逐字 | ✓ |
| commit **bf82a1d2** | header/§2 svg/§8 | cand 末行、HANDOFF §M7a/链 | ✓ |
| binary **422MB** | §8 svg | done §1 | ✓ |
| **524288B** cmp -s 逐位 | §8 svg | HANDOFF §M7a | ✓ |
| 8 fp16 instance / ref | §6 表/§8 svg | review §4 | ✓ |
| git diff 仅 **4** tracked 文件 + 9 库 byte-identical | §6 表/note | review §6 逐字 | ✓ |
| max|ref| 最高 **~10.9** < 65504 | §7b | done §3、review §7 | ✓ |

**minor RED — §6 行 343**:「reviewer 独立 build_review 跑全套件,**91 个 bf16 历史案(M0–M6b)全 PASS**」。
- 问题:与同文档 §7③「旧 **91 − 1** 过期 reject-fp16 + 14 fp16 + 2 repro = 106」口径冲突。旧 91 套件里有一格 `reject-fp16`(改前实为 FAIL),升级后剩 **90** 个 bf16 历史案;新 106 套件中 bf16 = 90(90+14+2=106)。故「91 全 PASS」既与 §7 自身拆解不符,也不精确(那 1 格改前并非 PASS)。
- 性质:**非臆造** —— 镜像自 reviewer §6 原话「91 个 bf16 案在 build_review 全 PASS」(reviewer §1 的「91+15+2」本身亦算术不闭合=108,coder done §5 的「91−1+14+2=106」才是自洽口径)。
- 应改:§6 那句把「91 个 bf16 历史案」→「**90 个 bf16 历史案**」或「**M0–M6b 全部 bf16 历史案**」,与 §7③ 的「91−1」一致。

### 2. ★ 范围诚实(最关键)—— **GREEN(强)**
全文一致、反复明示 hd64 + hdim_qk==hdim_v,无任一处暗示任意 hdim:
- header subtitle:「仍 hd64 + hdim_qk==hdim_v,hdim{96,128,256} 与 hdim_qk≠hdim_v 是 M7b/M7c」✓
- TL;DR badge「范围:hd64 + hdim_qk==hdim_v」✓
- §1:「不动 hdim(仍 64)、不动 hdim_qk≠hdim_v(仍相等)」+ §1 code「MaxK 仍 64」✓
- §8 warn-banner:「M7a 仍 hd64 + hdim_qk == hdim_v。hdim{96,128,256} 与 hdim_qk ≠ hdim_v 仍属 M7b/M7c,套件 reject-hdim128 仍守 dispatch 运行时 throw。**本文档不把范围夸大成「任意 hdim」**」✓
对齐 done §6 / cand「still hd64 + hdim_qk==hdim_v ... remain M7b/M7c (reject-hdim128 still guards throw)」。**零夸大。**

### 3. 技术叙述准确 —— **GREEN(1 条可选 clarity)**
- **复用而非重写**:§1 + 图1 + formula 讲对——dispatch/kernel/pipeline 本就模板化于 `InOutDataType`,`GemmAccDataType=CompDataType=float`,dtype 只活在「load Q/K/V/dO + store dQ/dK/dV」边界。对齐 review §1 type-config 对称性。✓
- **fp16 容差更紧的理由**:§3 讲对——尾数 fp16 10bit > bf16 7bit ⇒ 相对量化误差小 2³=8× ⇒ 收紧而非放水。图3 ε(bf16 2⁻⁸≈3.9e-3 / fp16 2⁻¹¹≈4.9e-4)与「多 3 bit→8×」自洽,数值正确。✓
- **fp16-ULP 反证(亮点)**:§4 逻辑讲对且未讲歪——「同输入若静默落回 bf16,误差应 ~8× 大;实测 6.10352e-05=2⁻¹⁴ 正是 fp16 量级 ⇒ 确在跑 fp16」。补强了「复用模板最易漏的失败模式(dtype 没真切、对拍假绿)」可证伪性。对齐 review §3。✓
- **容差 revert 双实验**:§5 结论对——A 放宽到 bf16 标尺仍 PASS=无隐藏误差;B 收紧 5–20× 仍 PASS=误差真小。对齐 review §2。✓
- *可选 clarity(非错)*:同一记号 `2⁻¹¹` 在 §3 标为「fp16 机器 ε」、在 §4 标为「bf16-ULP 量级」。二者实为不同维度(§3 是无量纲相对 ε,§4 是某小幅值梯度的绝对误差),且 8×(3 尾数 bit)比例两处一致、均忠实素材;但并列易让细心读者短暂困惑。可在 §4 加半句点明「此处为绝对误差量级,非 §3 的相对 ε」。不阻塞。

### 4. 零回归表述 —— **GREEN**
§6 表 + note 准确区分三类:① 4 个 tracked 文件「改·加性」(CMakeLists / example_bwd.cpp / generate_instances.py / api.hpp);② fp16 entry×2 + 8 instance + ref「新增」;③ kernel/两 pipeline/两 dispatch/reference/type_config/bool_switch「byte-identical(git diff HEAD 空)」。对齐 review §6 列的 9 个库文件 + done §4。措辞「库逻辑一行未动、改面限边界与构建层」准确。✓

### 5. 四方闭合 —— **GREEN**
§8 图4 四盒与素材一一对应:① coder(build 0err 422MB / sweep66 / 套件106 / +14pass+2repro);② reviewer 独立 build_review(7/7 GREEN / 对拍公平+容差双实验 / fp16-ULP 反证 / 9 库 byte-identical);③ WIP 本体(entry 镜像 / 8 instance+ref / 改面仅边界构建层 / 结构忠实);④ lead 亲核 3 项(git diff 仅 4 文件 / M6b 触发配置 fp16 PASS / fp16 determ 两次 cmp -s 524288B 逐位)。与 cand 末行「FOUR-PARTY CLOSURE (coder + reviewer + WIP body + lead亲核)」、HANDOFF §M7a 一致。强调 reviewer 用「独立 build_review 不复用 coder build」的呈现准确。✓

### 6. HTML 质量 —— **GREEN**
- 单文件、可立即渲染:无 `src=http`/`href=http`/`@import`/CDN/`<script`(grep 零命中)。
- 4 个 inline SVG,开/闭标签各 4 配平;图1(边界分叉)/图2(时间线)/图3(位布局)/图4(闭合矩阵)结构正常,viewBox 内坐标无明显溢出。
- 无占位符(TODO/FIXME/placeholder/lorem/占位/TBD 零命中)。
- 章节编号自洽:s1–s7 显示 1–7、s7b 显示「8」、s8 显示「9」;TOC 13 个 anchor 全部命中存在的 id(#s1..#s8,#s7b),无死链。
- 沿用 M6/M6b 同系列 ivory/slate/clay 视觉基线,serif 标题 + mono 代码,与同目录前作风格一致。

---

## 给 lead / 写作者的一句话
**M7a 讲义可发布。** 仅 1 处需改:§6(行 343)「91 个 bf16 历史案全 PASS」与 §7「91−1」口径冲突,应改 90(或「M0–M6b 全部 bf16 历史案」)——非臆造,是镜像了 reviewer 的不闭合措辞,但文档内部要自洽。另可选在 §4 半句澄清 `2⁻¹¹` 在 §3/§4 是两种不同量,避免读者混淆。范围诚实、技术亮点(fp16-ULP 反证 / 容差双实验)、零回归与四方闭合均准确无夸大。
