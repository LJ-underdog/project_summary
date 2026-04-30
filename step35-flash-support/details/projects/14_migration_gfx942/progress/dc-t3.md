# DC-T3 writer progress

> 任务：item #P2-W，撰写 `/home/junlin12/project_fp8_tp4_repro/MIGRATION_REPORT.md`
> 输入：`doc_consolidation/TEAM_CONFIG.md`、`doc_consolidation/progress/dc-t1.md`、`doc_consolidation/progress/dc-t2.md`
> 输出：`MIGRATION_REPORT.md`（项目根级，与 SESSION_HANDOFF.md 平级）

---

## 完成情况

### 章节完成度（强制大纲对照）

| 章节 | 状态 |
|---|---|
| 摘要 (TL;DR) | ✅ 含关键数字表 |
| §1 项目背景与目标（含 1.1-1.4 全部子节） | ✅ |
| §2 总体迁移流程图（图 #1）| ✅ |
| §3 三大 root cause 关系图（图 #2）| ✅ |
| §4 NEW-RC-1（4.1-4.5 全节 + 图 #3）| ✅ |
| §5 NEW-RC-2（5.1-5.5 全节 + 表）| ✅ |
| §6 NEW-RC-3（6.1-6.6 全节 + 图 #4）| ✅ |
| §7 M2 padding（7.1-7.4 全节 + 图 #5）| ✅ |
| §8 dispatch 路径对比表 | ✅ |
| §9 三仓改动 summary（9.1-9.3 三表）| ✅ |
| §10 PASS 验证证据链（10.1-10.3 + byte-identical 表）| ✅ |
| §11 关键时间线（图 #6）| ✅ |
| §12 后续可选优化（P1-P5）| ✅ |
| §13 References | ✅ |
| 附录 A 术语表 | ✅ |

**全部 6 张 mermaid 图 + 全部 13 章节 + 附录均已完成，无 placeholder。**

### Mermaid 图统计（强制 6 张）

| 编号 | 类型 | 位置 | 主题 | 状态 |
|---|---|---|---|---|
| #1 | flowchart TD | §2 | 总体迁移流程（gfx950 起点 → PROJECT CLOSED）| ✅ |
| #2 | graph LR | §3 | 三大 RC + M2 padding 关系（含独立性 + tp 复用）| ✅ |
| #3 | flowchart TD | §4.4 | NEW-RC-1 转换链路（safetensors → fused_moe q_dtype=fnuz）| ✅ |
| #4 | sequenceDiagram | §6.5 | NEW-RC-3 修复时序（actors: lead, T-4, T-5, T-12, CSV）| ✅ |
| #5 | flowchart TD | §7.3 | M2 padding 链路（config → TP 切 → buffer → padding 384）| ✅ |
| #6 | flowchart LR | §11 | Wave 0 → Wave 7 时间线 | ✅ |

### Markdown 表格统计（强制 ≥5 个）

实际写出 **15 个表格**：

1. 摘要 - 关键数字表（commit / 改动行数 / PASS / byte-identical）
2. §1.2 - gfx950 vs gfx942 关键差异表
3. §1.3 - M1 / M2 PASS 标准表
4. §4.2 - NEW-RC-1 调查时序表
5. §4.5 - NEW-RC-1 验证证据 4 类表
6. §5.5 - **forward vs inverse absmax 对比表**（用户强制要求）
7. §7.1 - M2 padding 触发场景层次表
8. §7.4 - M2 padding 验证证据表
9. §8 - **dispatch 路径对比表**（gfx950 vs gfx942 各 op）
10. §9.1 - **ATOM 改动表**
11. §9.2 - **aiter 改动表**
12. §9.3 - **CK 改动表**
13. §10.1 - M1 PASS V1/V2/V3 表
14. §10.2 - M2 PASS V1-V5 表
15. §10.3 - **M1 vs M2 byte-identical 对照表**（用户强制要求）

外加 §12 的优化清单表 + 附录 A 术语表。

### 引用统计

- **file:line 引用约 90+ 处**，全部来自 dc-t1 / dc-t2 已验证的二级源
- 引用源文档分布：
  - `FINAL_REPORT.md`（约 30 处）
  - `PROJECT_SUMMARY.md`（约 20 处）
  - `SESSION_HANDOFF.md`（约 8 处）
  - `TEAM_CONFIG.md`（约 5 处）
  - `progress/teammate-{1,2,3,4,5,6,7,10,11,12,13,16}.md`（约 25 处）
  - `doc_consolidation/progress/dc-t1.md` / `dc-t2.md`（约 15 处，作为整合层引用）
- 直引原文 4 处：
  - `SESSION_HANDOFF.md:239` "safetensors 是 e4m3fn，gfx942 要 fnuz"
  - `PROJECT_SUMMARY.md:222` "gfx942 mfma 要 fnuz"
  - `aiter/fused_moe.py:881-883` 原始注释 "for fp8 blockscale, ck has better performance so disable assembly kernel"
  - `atom/model_ops/moe.py:1715-1727` 开发者 padding 注释 "Bug fix: previously used align=64..."
- 直接引用 NEW-RC-3 完整 diff hunk（含 `+`/`-` 行，照搬 dc-t2 §4 修复段）

---

## 写作策略与决策

### 1. 不一致项的取舍（按 dc-t1 §7 建议）

| 主题 | 处理 |
|---|---|
| dispatch V3 期望 | 以 `FINAL_REPORT.md:66` 为准，写"实测路径优于预期"；不引用 SESSION_HANDOFF §5.3 老期望 |
| NEW-RC-2 方向 | 以 `FINAL_REPORT.md §2.3` 为准（方向正确），把 SESSION_HANDOFF 历史风险表表述完全弃用 |
| V4 强度措辞 | 按 PROJECT_SUMMARY §7.1 F-3 修正，写"实测 + padding 推断"，**未写"决定性"**（仅在表格里照旧引 FINAL_REPORT.md:49 V4 标签时保留"决定性间接证据"原话用作来源标注，但行文中不重复"决定性"措辞）|
| 三仓 commit 描述 | 用完整 hash + commit subject |
| NEW-RC-3 patch diff | 引完整 diff hunk |

### 2. dead end 处理

- T-3 的 `weight_scale_inv` 反向假设：在 §5.1 简短提及"T-3 在 Wave 1 末尾抛出的未验证假设"，作为 NEW-RC-2 的发现起点，未展开"如果反了怎么办"
- T-7 的 M2 NPerBlock=64 stage2 miss 警告：在 §7.1 作为"触发场景"叙事起点，§7.2 写明 T-10 揭穿（"上游 padding 自动规避"），未把 T-7 警告本身当作 root cause 写
- SESSION_HANDOFF wave 0 时代脚本 / HF 下载 / cron / token：完全未引用
- M1_BASELINE_DISPATCH_PLAN 各 prompt 全文：完全未引用
- TEAM_CONFIG 初始 TODO：完全未引用

### 3. 面向新读者的处理

- 附录 A 术语表覆盖 22 项缩略语（gfx942/950、e4m3fn/fnuz、per_1x128、NPerBlock、KPack、NEW-RC、V1-V5、TP、fused_moe、fmoe_g1u1、CK 2-stage、SwigluStep 等）
- §1.2 的 gfx950 vs gfx942 关键差异表是新读者快速建立心智模型的入口
- §3 关系图用文字注释强调"四者独立提出、独立闭环"以减少新读者误判 RC 间存在因果链

### 4. 中文 + file:line 格式遵守

- 全文中文（除直引英文原文 + 代码 + commit hash + 术语外）
- 引用统一 `file:line` / `progress/teammate-N.md:行号` 格式
- 每节末尾有"引用："聚合行，方便 reviewer 抽查

---

## 给 reviewer (DC-T4) 的建议

### 抽查命中率最高的 file:line 引用

按 dc-t1 §8 提示，建议 reviewer 优先抽查这些行（命中率最高）：

1. `aiter/fused_moe.py:881-886` —— NEW-RC-3 patch 实物，可直接 grep "NEW-RC-3"
2. `atom/model_ops/moe.py:1715-1727` —— ATOM padding 开发者注释直引
3. `aiter/utility/dtypes.py:10-25` —— gfx942 → e4m3fnuz 静态映射
4. `progress/teammate-6.md:51-58` —— forward vs inverse absmax 7 数量级数据
5. `FINAL_REPORT.md:49` —— V4 inter_dim=384 决定性间接证据
6. `progress/teammate-12.md:108-122` —— M2 baseline log fused_moe 调用签名
7. `progress/teammate-16.md:32-91` —— byte-identical 143/143 数据
8. `csrc/include/opus/opus.hpp:932-958` —— T-2 主动识别 NEW-RC-1 起点

### Mermaid 渲染语法心算检查

| 图 | 节点 ID 唯一性 | 箭头语法 | subgraph 闭合 | 注 |
|---|---|---|---|---|
| #1 | ✅ A,B,C,D1-D3,E,F,G,H1-H2,M1,N,O,P,M2,Q,R,S,T 全唯一 | `-->` `-->|是 V1/V2/V3 全过|` ok | 无 subgraph | flowchart TD |
| #2 | ✅ RC1/RC2/RC3/PAD/M1PASS/M2PASS 全唯一 | `-.独立.->` `-.复用.->` 虚线箭头 ok | 2 个 subgraph 都闭合 | graph LR |
| #3 | ✅ A-I 全唯一 | `-->` ok | 无 subgraph | flowchart TD |
| #4 | ✅ L/T4/T5/T12/CSV 5 actors，无重复 | `->>` `-->>` ok | sequenceDiagram，无 subgraph | 注：节点文字含括号已用 `<br/>` 分隔避免歧义 |
| #5 | ✅ A-L 全唯一 | `-->` `-->|tp=4|` ok | 无 subgraph | flowchart TD |
| #6 | ✅ W0,W1,W2,W3,W35,W4,W5,W55,W6,W7,CLOSE 全唯一 | `-->` ok | 无 subgraph | flowchart LR（备选 timeline，但 flowchart 更稳）|

**潜在 mermaid 风险**（reviewer 可重点检查）：
- 图 #4 的 sequenceDiagram 中我用了 `<br/>` 分行节点 label，部分 mermaid 渲染器对 sequenceDiagram 的 message label 是否支持 `<br/>` 不一致；如果失败，reviewer 可建议改成多行 message
- 图 #2 的虚线箭头标签中文 + 句号语法 `-.独立.->` 是合法的，但如果 lint 器严格，建议把中文标签去掉句号

### 技术结论与 KNOWN_FACTS 一致性

已与 `TEAM_CONFIG.md §KNOWN_FACTS` F1-F6 全部对齐：

- F1 硬件/模型 → §1.2 表 + 摘要表
- F2 三仓 commit → 摘要表 + §9
- F3 PASS 判定 → §10.1 / §10.2
- F4 三大 RC → §3 关系图 + §4/§5/§6
- F5 关键修复点 → §9 三仓改动表
- F6 byte-identical → §10.3 表

### 大纲完整性

强制大纲 13 章 + 附录 A 全部齐全，无遗漏。6 张 mermaid 图全部到位，15 个表格远超强制最低 5 个。

---

## 累计 tool calls

约 5 次（远低于 25 上限）：
- Read：3 次（TEAM_CONFIG / dc-t1 / dc-t2 并行）
- Bash：1 次（ls 验目录）
- Write：2 次（MIGRATION_REPORT.md + 本 progress）

---

## 收尾存档

### 关键发现

1. **dc-t1 + dc-t2 已把所有引用做到位**，writer 真正的工作是组织叙事 + 画图 + 制表，不需要回查 4 份主文档原文
2. **6 张 mermaid + 15 张表格**全部基于 dc-t1/dc-t2 既有素材，未引入任何新事实
3. **NEW-RC-3 是唯一源码改动**这一事实非常适合在 §6 突出（用 sequenceDiagram 给 4 actor 演出）
4. **forward vs inverse 7 数量级**是项目最干净的单一证据（§5.5 表）
5. **byte-identical 143/143** 是 PASS 的最强论据（§10.3 表）

### 红线遵守

- ✅ 仅 Read，仅 Write `MIGRATION_REPORT.md` + 本 progress
- ✅ 未修改任何源文档
- ✅ 所有结论引自 dc-t1 / dc-t2 / 4 份主文档（无新结论）
- ✅ 错误尝试已省略（T-3 反向假设、T-7 M2 警告仅作为叙事起点简短提及）
- ✅ 中文输出 + file:line 引用
- ✅ 未复述项目内 typo
- ✅ 未评价 dead end teammate 判断错误

### 给 lead 的建议

- 报告体量约 600+ 行，如果反馈过长可拆分摘要为独立"executive summary"文档；当前已用 TL;DR 节代替
- §13 References 是聚合视图；如果 lead 希望 References 改成"按 file 字典序排序"或"按 doc 类型分组"，请告知，writer 可派 P2-W2 微调
- 图 #4 sequenceDiagram 的 `<br/>` 在某些 mermaid 渲染器（特别是老版 mermaid-cli）可能不识别；如果 reviewer 实测渲染失败，建议改成单行 message

---

**DC-T3 已存档，MIGRATION_REPORT.md 已写完所有 13 章 + 附录 A + 6 mermaid + 15 表格，无 placeholder，建议 lead 直接派 DC-T4 reviewer。**
