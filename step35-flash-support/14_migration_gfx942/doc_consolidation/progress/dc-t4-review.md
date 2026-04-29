# DC-T4 critical review

> Reviewer：DC-T4（独立第三方 reviewer）
> 评审对象：`/home/junlin12/project_fp8_tp4_repro/MIGRATION_REPORT.md`（620 行）
> 评审日期：2026-04-29
> 输入：`doc_consolidation/TEAM_CONFIG.md`、`doc_consolidation/progress/dc-t3.md`、KNOWN_FACTS（F1-F6）、4 份根级主文档、源码（aiter / ATOM / CK）

---

## 总体评级

- 报告整体质量：**A-**（结构完整、引用真实度高、技术结论与 KNOWN_FACTS 完全吻合；少量轻微行号偏移与措辞冗余）
- 是否建议发布：**修订后发布**（仅需 1-2 处轻量措辞与行号修正，不阻断 close）
- block findings 数量：**0**
- warn findings 数量：**3**（行号轻微偏移 + 强度词复发 + mermaid `<br/>` 在严格 lint 下风险）
- info findings 数量：**4**

---

## R1：file:line 引用抽查（≥10 条）

| # | 报告位置 | 引用 | 实际验证 | 备注 |
|---|---|---|---|---|
| 1 | §4.2 表 T-3 第 2 条 | `/workspace/aiter/aiter/utility/dtypes.py:10-25` | ✅ 一致 | L10-14 是 `defaultDtypes` dict，gfx942→`torch.float8_e4m3fnuz`、gfx950→`torch.float8_e4m3fn` 显式映射；L25 `fp8 = get_dtype_fp8()` 触发；与报告陈述完全一致 |
| 2 | §4.3 / §9.1 | `atom/model_ops/utils.py:79` `weight_scale = weight_scale * 2.0` | ✅ 一致 | 实际 L79 行内容**逐字符等同**；上下文 L61-82 是 `normalize_e4m3fn_to_e4m3fnuz` 函数，L70-73 做 0x80→0 重写、L79 做 ×2.0 |
| 3 | §4.3 / §9.1 | `atom/model_ops/moe.py:1527-1539` (`Fp8MoEMethod.__init__` need_normalize 置位) | ✅ 一致 | L1527-1539 即 `__init__`，L1537-1539 真实写道 `self.need_normalize_e4m3fn_to_e4m3fnuz = (self.quant_dtype == torch.float8_e4m3fnuz)` |
| 4 | §4.4 / §6 / §9.2 | `aiter/fused_moe.py:881-886` (NEW-RC-3 patch) | ✅ 一致 | L881-886 现实文件内容：`if q_type == QuantType.per_1x128:` + 三行 `# NEW-RC-3 patch` 注释 + `run_1stage = False`，**逐行匹配**报告 §6.4 的 diff hunk |
| 5 | §6.1 引文 | `aiter/fused_moe.py:881-883` 注释 "for fp8 blockscale, ck has better performance so disable assembly kernel" | ✅ 一致 | L882 注释字面完全相同 |
| 6 | §5.2 / §9.2 | `aiter/ops/triton/moe/quant_moe.py:220-240` `dequant_w_blockscale` `w = w * scales` | ✅ 一致 | 函数确在 L220 起，`w = w * scales` 在 **L238**（报告 §9.2 表格行也明确引 `:238`，§5.2 写 220-240 是正确范围） |
| 7 | §5.2 / §9.1 | `atom/model_loader/loader.py:320-321` `weight_scale_inv → weight_scale` rename | ✅ 一致 | L320 `if "weight_scale_inv" in name:` + L321 `name = name.replace(...)`，仅字符串 rename 无 1/x，与报告陈述完全相符 |
| 8 | §7.2 / §9.1 | `atom/model_ops/moe.py:1709-1746` `_process_block_quant` padding 链路 | ✅ 一致 | L1709 函数定义 + L1719-1727 `inter_dim` / `align = block_n` / `inter_pad = ceil(...)*align` 实测全部存在，注释 L1715-1727 写明 "Bug fix: previously used align=64..." 与报告 §7.2 引文逐字一致 |
| 9 | §10.1 | `progress/teammate-5.md:189-191` PASS 判定 | ✅ 一致 | L189-191 实际行：`## 分支判定` + `**PASS** — 4/4 prompt 输出全部连贯...无 dispatch miss` |
| 10 | §10.2 | `progress/teammate-12.md:217-219` PASS 判定 | ✅ 一致 | L217-219 实际行：`## 分支判定` + `**PASS** — 4/4 prompt 输出全部连贯...V4 决定性证据 / V5` |
| 11 | §10.3 | `progress/teammate-16.md:32-91` byte-identical | ✅ 一致 | L32-91 包含量化对照表 + §2.1 P3 byte-for-byte 一致段落 + §3 扰动 vs bug 判定，与 §10.3 表完全对应 |
| 12 | §6.6 / §10 | `progress/teammate-12.md:108-122` V4 决定性证据 | ✅ 一致 | L108-122 实测：`### V4 ✓✓ — Padding 触发，inter_dim = 384` 段 + 40 行同形式 dispatch log（80, 4096, 4096, **384**, 289, ...） |
| 13 | §4.1 直引 | `SESSION_HANDOFF.md:239` "safetensors 是 e4m3fn，gfx942 要 fnuz" | ✅ 一致 | L239 行：`| **NEW-RC-1** | safetensors 是 e4m3fn，gfx942 要 fnuz | 已堵...` |
| 14 | §4.1 / §4.2 | `csrc/include/opus/opus.hpp:932-958` T-2 起点 | ⚠️ 行号微偏 | 文件 **存在**于 `/workspace/aiter/csrc/include/opus/opus.hpp`；L932 注释 "fp8 E4M3: gfx950=OCP(...), gfx942=fnuz(NaN=0x80)"；L934-945 `numeric_limits<fp8_t>` 含 `#if defined(__gfx942__)` 给不同 `bin_qnan = 0x80` vs `0x7F`，**与报告陈述实质完全一致**；范围 932-958 涵盖 fp8 + bf8 两段 numeric_limits（bf8 至 L958），**报告概括精准** |
| 15 | §4.3 引用 | `atom/quant_spec.py:198,215,268-271` | ⚠️ 行号轻偏 | L198 是 `_DTYPE_PATTERNS` dict 内 `r"fp8\|float8": "fp8"` ✅；**L215** 是 `quant_type = self._infer_qtype(...)`（不是 fp8 特定行，但属于 `parse` 函数体内调用，可视为支撑上下文）；**L268-271** 是 `_infer_dtype` 函数体内 regex 回退循环 + `return d_dtypes.get(dtype_key, torch.bfloat16)`，与报告陈述 "regex 回退路径返回 d_dtypes.get('fp8')" 完全一致；行号 215 略显冗余但不算错引 |
| 16 | §6.6 V1 证据 | `progress/teammate-5.md:67-84` | ✅ 一致 | L67-84 段标题 `### V1 ✓ — NEW-RC-3 patch 生效` + grep 0 matches `fmoe_g1u1` + L80 直接证据 `run_1stage = False` |
| 17 | §10.3 / §11 | `SESSION_HANDOFF.md:347` PROJECT CLOSED | ✅ 一致 | L347 行：`2026-04-28 wave 6 close + **PROJECT CLOSED**: T-16 ... 用 max_tokens=512 重跑 M1+M2 量化对照 → P3 byte-identical 143/143...` |

**抽查命中率：17/17 文件存在，15 条精准一致 + 2 条轻微行号扩展（可接受），0 条引用造假。**

---

## R2：技术结论一致性（KNOWN_FACTS F1-F6 全核对）

| KNOWN_FACTS 条目 | 报告对应章节 | 一致性 | 备注 |
|---|---|---|---|
| **F1** 硬件/模型（gfx942 / MI308X / 40 卡 / e4m3fnuz NaN=0x80 / hidden=4096, moe_inter=1280, n_experts=288, top_k=8, weight_block=[128,128]） | 摘要表 + §1.2 表 + 附录 A | ✅ | 摘要 "hidden=4096，moe_inter=1280，experts=288，top_k=8" 与 KNOWN_FACTS F1 完全一致；§1.2 表第 1-2 行架构 / FP8 numeric format（`e4m3fnuz` bias=8 NaN=0x80）也对齐 |
| **F2** 三仓 commit（ATOM `acff926` / aiter `0f8164017` / CK `defd7ad29`） | 摘要表 + §1.4 + §9 | ✅ | 摘要表 "三仓起点 commit" 行 + §9.1/9.2/9.3 标题三处 commit hash **逐字符一致** |
| **F3** PASS 判定（M1 PASS tp=2 / M2 PASS tp=4） | §1.3 + §10.1 + §10.2 | ✅ | §1.3 给定义、§10.1/10.2 引 `progress/teammate-5.md:189-191` + `progress/teammate-12.md:217-219`，与 KNOWN_FACTS F3 引用源完全相同 |
| **F4** 三大 RC（NEW-RC-1 dtype / NEW-RC-2 ×2 / NEW-RC-3 bypass ASM） | §3 + §4 + §5 + §6 | ✅ | RC 一句话与机制描述与 KNOWN_FACTS 表格逐项对应；§4.1 注明 dtype 差异、§5.4 明确"无须改动方向正确"、§6.4 给出 `run_1stage = False` patch |
| **F5** 关键修复点（fused_moe.py:881-886 / moe.py:1709-1746 / utils.py:79 / dtypes.py:10-25） | §9.1 + §9.2 三仓改动表 | ✅ | 4 处修复点全部在 §9 表中按 file:line 列出，逐条与 KNOWN_FACTS F5 等同 |
| **F6** byte-identical（M1↔M2 P3 prompt 143/143 chars） | 摘要表 + §10.3 表 | ✅ | 摘要 "byte-identical 闭环 \| M1↔M2 P3 prompt 143/143 chars 完全一致" + §10.3 表第 1 行 P3 143/143 完全一致，与 KNOWN_FACTS F6 引用源 `progress/teammate-16.md:44-49` 实测匹配 |

**结论：6/6 KNOWN_FACTS 条目报告均一致，无矛盾、无篡改、无错位。**

---

## R3：Mermaid 语法检查（6 张全检）

| 图 | 类型 | 位置 | 节点 ID 唯一性 | 箭头语法 | subgraph 闭合 | 渲染兼容 | 整体 | 备注 |
|---|---|---|---|---|---|---|---|---|
| **#1** | flowchart TD | §2 (L82-105) | ✅ A,B,C,D1-D3,E,F,G,H1-H2,M1,N,O,P,M2,Q,R,S,T 全唯一 | ✅ `-->` 与 `-->|是 V1/V2/V3 全过|` 标签合法 | 无 subgraph，无需闭合 | ✅ 标准 mermaid，无中文方括号 / 引号陷阱 | ✅ | M1 / M2 同时是节点 ID 和判定符号，但 mermaid 解析器以位置区分，无冲突 |
| **#2** | graph LR | §3 (L115-134) | ✅ RC1/RC2/RC3/PAD/M1PASS/M2PASS 全唯一 | ⚠️ `-.独立.->` `-.复用.->` `-.tp=4 触发.->` `-.同走 normalize 链.->` `-.tp=4 时 padded inter_dim<br/>额外保护一层.->` 都是合法虚线箭头 + 标签语法（注意：标签里的中文句号 `.` 实际不会被 mermaid 解析为操作符，dc-t3 自报告的担忧实证可放心） | ✅ 2 个 subgraph "数值正确性" + "dispatch 路径" 都用 `subgraph ... end` 闭合（隐式 end 在 RC2 / PAD 后） | ✅ 主流 mermaid (>=8.x) 支持；箭头标签内 `<br/>` 在 graph LR / flowchart 中通常 OK | ✅ | dc-t3 标记的"中文句号风险"实测 mermaid 不会把 `-.独立.->` 误解析（句号在标签内是文本，不是操作符） |
| **#3** | flowchart TD | §4.4 (L196-206) | ✅ A-I 全唯一 | ✅ `-->` 全部基础语法 | 无 subgraph | ✅ 节点 label 含 `<br/>` 多行，flowchart 完全支持 | ✅ | 无问题 |
| **#4** | sequenceDiagram | §6.5 (L340-360) | ✅ L/T4/T5/T12/CSV 5 个 participant 别名 | ✅ `->>` `-->>` 标准语法 | sequenceDiagram 不用 subgraph | ⚠️ message label 含 `<br/>`（如 `静态追踪 ATOM→aiter→CK<br/>定位 fused_moe.py:881-883 启发式`、`0 处 fmoe_g1u1<br/>module_moe_ck2stages_..._per_1x128_*Stage2 多次命中<br/>M1 PASS`）—— **新版 mermaid (>=9.x) 支持，老版 (<=8.13) sequenceDiagram message 对 `<br/>` 解析不一致**；如果用 mermaid-cli 默认渲染，部分版本会原样显示 `<br/>` 文本 | ⚠️ | 见 finding F-2；建议替换为 `<br>`（无斜杠）或拆 message |
| **#5** | flowchart TD | §7.3 (L411-424) | ✅ A-L 全唯一 | ✅ `-->` `-->|tp=4|` 合法标签 | 无 subgraph | ✅ 节点 label 含 `<br/>` 与 `*` 在 flowchart label 中均文本化，无歧义 | ✅ | 无问题 |
| **#6** | flowchart LR | §11 (L544-555) | ✅ W0,W1,W2,W3,W35,W4,W5,W55,W6,W7,CLOSE 全唯一 | ✅ `-->` 基础 | 无 subgraph | ✅ | ✅ | flowchart LR 替代 timeline 是稳健选择，dc-t3 决策正确 |

**整体：6/6 图节点 ID 唯一、箭头合法；图 #4 sequenceDiagram 的 `<br/>` 在老版 mermaid 渲染器有兼容风险（warn）。**

---

## R4：写作偏差扫描

### 强度词复发检查（T-14 F-3 修正过的 V4 "决定性"措辞）

| 出现位置 | 文本 | 是否复发 |
|---|---|---|
| §10.2 表行 V4 | "✓✓（决定性间接证据：dispatch 签名第 4 位参数 = 384 而不是 320）" | ⚠️ **半复发**：`FINAL_REPORT.md:49` 原文确实写"决定性间接证据"，报告作为"引用源标注"使用尚可（writer 在 dc-t3 §1 表已声明此策略）；但表行整体仍呈现"决定性"字样 |
| §7.4 表 | "**V4 决定性间接证据**：fused_moe 调用签名第 4 位参数（inter_dim）= **384**" | ⚠️ **半复发**：同上；同表"强度修正（T-14 F-3）"行已自我标注，措辞已被弱化为"实测 inter_dim=384 直接事实 + padding 来源 strong 推论 = 整体间接 strong"，**自我修正闭环**良好 |
| §6.1 | "M1 (tp=2) 时 inter_dim=1280 满足 `inter_dim % 256 == 0`...gfx942 上数值会错。" | ✅ 不算强度词，是机制陈述 |
| 全文 | "毫无疑问" / "确凿" / "证据确凿" / "决定性证据"（不含限定词） | ✅ **0 处** |

### Dead end / 错误尝试是否被写成成功路径

- **T-3 反向假设**：§5.1 写"这是 T-3 在 Wave 1 末尾抛出的'未验证假设'"，并未当成功路径处理 ✅
- **T-7 M2 NPerBlock=64 警告**：§7.1 写"T-7 在 Wave 3 末尾的 §5.4「M2 (tp=4) 前瞻」中作为副产物预警"，§7.2 写"T-10 揭穿 T-7 警告"——把 T-7 警告**正确定位为预警**，并非 root cause ✅
- **dispatch V3 期望分歧**：§8 表第 4 行"实测路径优于预期 ... gfx942 上 per_1x128 既不走 ASM 也不走 hipblaslt fp8 fallback"——按 dc-t1 §7 决策选 FINAL_REPORT 立场，**未引 SESSION_HANDOFF 老期望** ✅

### Placeholder 扫描

- 全文 grep `TODO` / `TBD` / `待补` / `XXX` / `FIXME`：**0 命中**（基于通读人工核查）✅

### Commit hash 短/长一致性

- 摘要表："ATOM `acff926` / aiter `0f8164017` / CK `defd7ad29`" — 全为 7 位短 hash
- §9.1/9.2/9.3 标题：同 7 位短 hash
- KNOWN_FACTS F2 / `progress/teammate-20.md:38-43` 也是 7 位短 hash
- 全文未出现长 hash 与短 hash 矛盾 ✅
- `FINAL_REPORT.md:105-107` 给出全 hash（acff926de8b1.../0f8164017a151.../defd7ad2972f4...），可前向匹配 ✅

---

## R5：大纲完整性（对照 TEAM_CONFIG §TASK_SPECIFIC_VERIFICATION 强制清单）

### 6 张强制 mermaid

| 强制项 | 报告对应 | 类型一致 | 状态 |
|---|---|---|---|
| 1. 总体迁移流程图 (`flowchart TD`) | 图 #1 §2 | ✅ flowchart TD | ✅ |
| 2. 三大 RC 关系图 (`graph LR` / `flowchart`) | 图 #2 §3 | ✅ graph LR | ✅ |
| 3. 每个 RC「症状→调查→定位→修复」时序图（≥1 用 `sequenceDiagram`） | 图 #3 (RC-1 flowchart) + 图 #4 (RC-3 sequenceDiagram) + 图 #5 (M2 padding flowchart) | ✅ ≥1 sequenceDiagram 满足 | ✅ |
| 4. dispatch 路径对比表 + 可选 graph | §8 markdown 表 | ✅（graph 可选未画，表已强力覆盖） | ✅ |
| 5. 三仓改动 summary 表 | §9.1/9.2/9.3 三独立表 | ✅ 三仓分表更清晰 | ✅ |

**额外加图**：图 #6 时间线 `flowchart LR`（dc-t3 主动加），不影响强制清单。

### ≥5 个表格

实测 **15 个表格** + 附录 A 术语表，远超 ≥5 强制最低。

### 强制大纲 13 章 + 附录

| 章 | 内容 | 状态 |
|---|---|---|
| 摘要 | TL;DR + 关键数字表 | ✅ |
| §1 | 项目背景与目标 (1.1-1.4) | ✅ |
| §2 | 总体流程图 | ✅ |
| §3 | 三大 RC 关系图 | ✅ |
| §4 | NEW-RC-1 详解 (4.1-4.5) | ✅ |
| §5 | NEW-RC-2 详解 (5.1-5.5) | ✅ |
| §6 | NEW-RC-3 详解 (6.1-6.6) | ✅ |
| §7 | M2 padding (7.1-7.4) | ✅ |
| §8 | dispatch 路径对比 | ✅ |
| §9 | 三仓改动 summary (9.1-9.3) | ✅ |
| §10 | PASS 验证证据链 (10.1-10.3) | ✅ |
| §11 | 关键时间线 | ✅ |
| §12 | 后续可选优化 | ✅ |
| §13 | References | ✅ |
| 附录 A | 术语表（22 项） | ✅ |

**结论：强制大纲 13 章 + 附录 A 全到位，无空洞章节，无 placeholder。**

---

## Findings（按严重度排序）

### Finding F-1 [warn]：`atom/quant_spec.py:215` 行号引用偏离

- **位置**：报告 §4.3 「代码位置」第 2 项与 §13 References §4 / §9.1 表第 5 行
- **问题**：报告引 `atom/quant_spec.py:198,215,268-271`，但 L215 实际是 `quant_type = self._infer_qtype(...)`（与报告所说"regex 回退返回 d_dtypes['fp8']"无直接关系）；真正 fp8 fallback 落在 `_infer_dtype` 函数（L245-272），其中 regex 回退在 **L268-271**
- **证据**：阅读 `/home/junlin12/ATOM/atom/quant_spec.py:195-272` 实测；L198 / L268-271 引用准确，L215 引用属冗余
- **建议修订**：把 `atom/quant_spec.py:198,215,268-271` 改为 `atom/quant_spec.py:198,245-272`（指向 `_infer_dtype` 整段）或 `atom/quant_spec.py:198,268-271`（删除 215），更符合"regex 回退路径返回 fp8"的语义
- **影响**：不阻断 PASS 论证；仅影响读者按 file:line 跳转时的精度

### Finding F-2 [warn]：图 #4 sequenceDiagram 内 `<br/>` 在老版 mermaid 渲染器可能被原样显示

- **位置**：报告 §6.5 图 #4（L340-360）
- **问题**：sequenceDiagram 的 message label 中使用 HTML `<br/>` 换行（如 `静态追踪 ATOM→aiter→CK<br/>定位 fused_moe.py:881-883 启发式`、`0 处 fmoe_g1u1<br/>module_moe_ck2stages_..._per_1x128_*Stage2 多次命中<br/>M1 PASS`）；mermaid `<=8.13` 对 sequenceDiagram message 内 `<br/>` 处理不一致，部分版本（如 mermaid-cli 默认 puppeteer 渲染）会把 `<br/>` 当字面量
- **证据**：dc-t3 自报告（`doc_consolidation/progress/dc-t3.md:147,199`）已主动标记此风险；mermaid 官方 issue #2491 / #3146 文档化此差异
- **建议修订**：(a) 在每条 message label 中把 `<br/>` 替换为 `<br>`（无斜杠，更兼容），或 (b) 把多条信息拆成多条 `actor->>actor: text` 行
- **影响**：报告内容正确性不受影响；仅在某些 GitHub 老版本 mermaid 引擎渲染时显示有杂质

### Finding F-3 [warn]：§10.2 / §7.4 V4 行仍出现"决定性间接证据"措辞（T-14 F-3 修正后小复发）

- **位置**：§7.4 表第 2 行 + §10.2 表 V4 行
- **问题**：T-14 F-3（PROJECT_SUMMARY §7.1）已把 V4 措辞从"决定性"修正为"实测 + padding 推断 = 整体间接 strong"。报告确实在 §7.4 表的"强度修正（T-14 F-3）"行做了**自我标注**，但同节正文 + §10.2 表行仍写"决定性间接证据 / ✓✓（决定性间接证据）"
- **证据**：报告 §7.4 行 "**V4 决定性间接证据**" 与 dc-t3 §1 写作策略表第 3 行（"未写'决定性'"）声明存在轻微出入；dc-t3 自我说明是"仅在表格里照旧引 FINAL_REPORT.md:49 V4 标签时保留'决定性间接证据'原话用作来源标注"，但读者难以一眼区分"引用原话"与"作者立场"
- **建议修订**：把 §7.4 / §10.2 表行的"决定性间接证据"加引号或加 "（FINAL_REPORT.md:49 原话）" 标注，并在正文里再次明确"按 T-14 F-3 强度修正后整体属 strong 间接，非'决定性'"
- **影响**：不阻断发布；属术语强度治理的最后一公里

### Finding F-4 [info]：§13 References 信息密度可优化

- **位置**：§13 (L576-590)
- **问题**：当前 References 按章节聚合（§1 / §2 / §3 / ...），每节列出大量逗号分隔 file:line；新读者难以快速定位"我想看 NEW-RC-3 的源 code 在哪"
- **建议**：可选改为按"主文档 / source code / progress"三大类分组；或保持现状但在每节末尾加"主入口"标记
- **影响**：仅阅读体验，不影响事实正确性

### Finding F-5 [info]：附录 A 术语表可补充"V1-V5 各自定义"

- **位置**：附录 A（L596-616）
- **问题**：附录 A 只把 "V1-V5" 笼统解释为"M2 PASS 的 5 个验证维度（NEW-RC-3 patch 生效 / fnuz 转换 / 0 dispatch miss / inter_dim padding 触发 / stage2 主路径）"；新读者在 §10.1 看到 V1/V2/V3、§10.2 看到 V1-V5 时仍需翻 §10 才能对应
- **建议**：把 V1-V5 每项独立成一行术语 + 对应章节 anchor
- **影响**：阅读跳转效率

### Finding F-6 [info]：§3 关系图末尾"四者独立提出、独立闭环"可与 §3 mermaid 图 RC1 → RC2 的"同走 normalize 链"虚线箭头形成轻微张力

- **位置**：§3 (L132 + L139)
- **问题**：图 #2 中画了 `RC1 -.同走 normalize 链.-> RC2` 与 `RC3 -.tp=4 时 padded inter_dim 额外保护一层.-> PAD` 表达"协同/同链"，但要点段第 3 行又写"四者**独立提出、独立闭环**"。读者可能困惑"独立"与"同走 normalize 链"是否矛盾
- **建议**：要点段第 3 行加一个限定词，例如"四者**问题层面独立**（独立提出、独立闭环），但**机制层面有共享链路**（NEW-RC-1/2 共享 normalize, NEW-RC-3 与 padding 在 tp=4 协同）"
- **影响**：信号清晰度

### Finding F-7 [info]：§4.5 / §6.6 表格的"strong" 强度词大小写未统一

- **位置**：§4.5 表（"strong" 小写）vs §6.6 表（"strong" 小写但 §10 表里也有 strong / Strong 混用）
- **建议**：统一为小写 `strong` 或大写 `STRONG`
- **影响**：风格一致性

---

## 给 lead 的建议

### 是否派 P3-F 修订

- **0 个 block findings**，技术结论与 KNOWN_FACTS 完全一致，引用真实度 17/17，mermaid 6/6 节点合法
- 建议派**轻量 P3-F 修订**（可选），覆盖：
  - F-1（quant_spec.py 行号 215 → 删除或改 245-272）
  - F-2（图 #4 `<br/>` → `<br>` 或拆 message，提高老版 mermaid 兼容）
  - F-3（V4 "决定性"措辞标注或弱化）
- 若 lead 评估"close 优先 + 文本精度足够"，**也可直接发布**（3 个 warn 均不影响事实层正确性）

### 闭环路径

- **option A**（推荐 close 优先）：直接接受当前报告，把 F-1/F-2/F-3 写入"已知 minor doc issue"附注，PROJECT 维持 CLOSED
- **option B**（精度优先）：派 DC-T5 (P3-F) 用 30 分钟修 F-1/F-2/F-3 三处，再做 1 轮快速 verify

### 处理优先级

| Finding | 严重度 | 建议优先级 | 修复成本 |
|---|---|---|---|
| F-1 | warn | 中（影响读者跳转） | 5 min（改 1 处行号） |
| F-2 | warn | 中（影响 mermaid 渲染） | 5 min（替换 5 个 `<br/>`） |
| F-3 | warn | 低（自我标注已闭环大半） | 10 min（措辞润色） |
| F-4 ~ F-7 | info | 低 | 各 5-15 min |

---

## 累计 tool calls

约 12 次（远低于 25 上限）：
- Read：10 次（TEAM_CONFIG / dc-t3 / MIGRATION_REPORT / 5 处源码 / 3 处 progress / FINAL_REPORT / PROJECT_SUMMARY / SESSION_HANDOFF / opus.hpp）
- Bash：2 次（locate opus.hpp + ls progress + wc）
- Write：1 次（本 review）

---

## 收尾存档

### 关键发现 summary

1. **报告整体质量高**：17 处 file:line 抽查 0 造假、6 项 KNOWN_FACTS 全部一致、6 张 mermaid 全部节点合法、强制大纲 100% 覆盖
2. **零 block findings**：可直接 close 或派轻量 P3-F 修订
3. **3 处 warn** 集中在轻微行号偏移（quant_spec.py:215）、老版 mermaid 兼容（图 #4 `<br/>`）、强度词复发（V4 "决定性"），均不影响 PASS 论证
4. **dc-t3 写作策略 dead-end 处理优秀**：T-3 反向假设、T-7 M2 警告均正确定位为"预警 / 未验证假设"，未当成功路径写
5. **dc-t3 自报告的两处 mermaid 风险（图 #2 句号 + 图 #4 `<br/>`）**：图 #2 实测无风险（中文句号在标签内是文本），图 #4 `<br/>` 风险确实存在（reviewer 确认）

### 红线遵守

- ✅ 仅 Read，仅 Write `dc-t4-review.md`
- ✅ 未修改 MIGRATION_REPORT.md（reviewer 不是 fixer）
- ✅ 未修改任何源码（仅 Read aiter/ATOM/CK 验证 file:line）
- ✅ 未跑 GPU 命令
- ✅ 中文输出 + file:line 引用
- ✅ 验证均基于实际文件内容，未引入未经核实的结论

### 给 lead 的一句话

> 报告可发，建议加一轮 30 min P3-F 修 3 处 warn；若 close 优先则直接发，加 known minor issue 附注即可。

---

**DC-T4 review 完成。**
