# WAVE_CLOSE — issue_wave

> Wave：fp8-tp4-repro / wave 6 / issue_wave
> 日期：2026-04-29
> 状态：✅ CLOSED
> 红线遵守：未修任何源码 / 未改任何已有 .md / 未动任何 PASS 判定 / 未派衍生 task

---

## 1. 任务回顾

| 项 | 值 |
|---|---|
| Task | P0（来源 `handoff_wave/HANDOFF_PACKET.md` §6）：写 ATOM 上游 issue 草稿（tp=8 `_load_w2` narrow size<0）|
| 范围 | 纯文档（issue 草稿 + reviewer 闭环），不实施 fix，不开 PR |
| 目标 close 的 finding | F-OPEN-1（CORRECTNESS_REPORT §4 + HANDOFF_PACKET §4.1） |

---

## 2. 子 task 完成情况

| # | task | owner | 状态 | 产物 |
|---|---|---|---|---|
| T0 | 收齐证据 | Lead | ✅ | (in-context) CORRECTNESS_REPORT §4 + corr-t1.md §3 + moe.py:2335-2364 + moe.py:2292-2333 |
| T1 | 写 TEAM_CONFIG | Lead | ✅ | `TEAM_CONFIG.md` |
| T2 | 写 ATOM_ISSUE_DRAFT v1 | Lead | ✅ | `ATOM_ISSUE_DRAFT.md` (225 行) |
| T3 | reviewer 派单 + critical review | Teammate (issue-reviewer) | ✅ | `progress/iw-reviewer.md` (评级 A-, 0 block / 1 HIGH / 4 warn / 3 info) |
| T4 | Lead 综合 finding 决策 + 修订 draft | Lead | ✅ | `ATOM_ISSUE_DRAFT.md` v2 (226 行)；3 处 Edit（HIGH-1 / warn-3 / warn-4 / info-3）|
| T5 | 写 WAVE_CLOSE | Lead | ✅ | 本文件 |

---

## 3. Reviewer finding 处理决策表

| Finding | 严重度 | Lead 决策 | 实施 |
|---|---|---|---|
| HIGH-1（§"Affected configurations" 穷举 D 表算错）| HIGH | **修** | ✅ Edit 1：删穷举表，改为 closed-form 双 symptom 条件 + 单一 concrete observed example (D=10/tp=8) |
| warn-3（Option B Cons 缺 ordering caveat）| warn | **修** | ✅ Edit 2：Option B Cons 末尾追加一句 "reverses the per-rank residual size ordering ..." |
| warn-4（inter_size=1280 自身也是 inferred）| warn | **修** | ✅ Edit 3：Caveats #1 末尾追加一句 "inter_size = 1280 is itself inferred from ..." |
| info-3（F-OPEN-1 内部 ID 暴露给上游）| info | **修** | ✅ Edit 4：Caveats #4 改用 "internal project tracker" 中性措辞 |
| warn-1（safetensors shard 1/44 vs 4/44 上游 .md 自身不一致）| warn | **接受不改** | 上游 .md 自身有出入，draft 与主报告 §4 一致即可；非本 wave 修复范围 |
| warn-2（draft 内 `# ceil split` 注释易被误读为源码注释）| warn | **接受不改** | 边际收益低；draft 已用 `# ← line 2357` 等 marker 区隔 reviewer 注解与源码 |
| info-1（§"Why this matters" bullet 3 暴露 sibling fix 列表）| info | **接受不改** | 该 bullet 为上游 maintainer 提供有用的家族 bug 上下文；删除会损失"我们做过 family-level 分析"的信号 |
| info-2（未隔离崩 expert / param name）| info | **接受不改** | Caveat #1 已隐式覆盖（含上游自验方法 print(name, ...)）|

---

## 4. 关键产出（issue file 时引用本表）

| 文件 | 路径 | 备注 |
|---|---|---|
| ATOM 上游 issue 草稿 | `issue_wave/ATOM_ISSUE_DRAFT.md` | **可直接 copy-paste 到 GitHub issue**（226 行，A1-A5 全部 PASS，reviewer A-）|
| Wave 配置 | `issue_wave/TEAM_CONFIG.md` | 红线 + 验收标准 |
| Reviewer 报告 | `issue_wave/progress/iw-reviewer.md` | 8 维度核对 + 决策建议 |
| 本 close packet | `issue_wave/WAVE_CLOSE.md` | mini-handoff |

---

## 5. 对 F-OPEN-1 的影响

| 维度 | issue_wave 之前 | issue_wave 之后 |
|---|---|---|
| 证据成熟度 | corr wave 已锁 root cause + 数学推断；缺上游可读包装 | ✅ 上游可读 issue draft 完成；可随时 file |
| 上游归一程度 | 内部知识 | ✅ 已具备一次性向 ATOM 上游 file 的形态 |
| 主线 PROJECT_SUMMARY 引用清单 | F-OPEN-1 状态：open | ✅ 可升级为 "draft ready, awaiting upstream file" |
| F-OPEN-1 是否 close | 否（内部 finding）| ⏸ **本 wave 不 close F-OPEN-1**：file 动作（开 GitHub issue）需用户授权（"创建 issue / 通知上游" 属于跨边界 risky 动作，按 Claude Code 默认确认原则）|

---

## 6. 给下任 Lead 的 next-step 选项

| 优先级 | 动作 | 触发 | 红线影响 |
|---|---|---|---|
| **由用户决策** | file `ATOM_ISSUE_DRAFT.md` 到 ATOM upstream GitHub | 用户授权（risky / shared-state action）后，由用户或 Lead 在用户许可下执行 | 跨边界（影响外部仓库）|
| P1（HANDOFF §6 第二项）| tp=8 byte-identical / multi-prompt correctness | 用户决定继续解 F-OPEN-1/2/5 | 需跑 GPU |
| P1（HANDOFF §6 第三项）| 比对 perf wave tp=8 PASS vs corr wave tp=8 FAIL 启动命令 | 想关闭 §"Caveats #4" 的开口（也是 ATOM_ISSUE_DRAFT.md 唯一未解释的 caveat）| 纯 doc，无 |
| P2/P3 | 见 HANDOFF_PACKET §6 | — | — |

**Lead 推荐**：**向用户呈交 ATOM_ISSUE_DRAFT.md 终稿，等用户决定是否 file**。本 wave 已完成 P0 task 的全部内部交付，剩余 file 动作不属于 lead 自主权限。

---

## 7. HANDOFF_PACKET 是否需要更新

- HANDOFF_PACKET.md 是 wave 5 (handoff_wave) 的 close packet，本身已 frozen（红线 R2）
- 本 wave (issue_wave) 的产出**不写回** HANDOFF_PACKET
- 下一会话若需要"从 wave 6 之后接手"，应：
  - 第一步读 HANDOFF_PACKET.md（仍是项目主入口）
  - 第二步读本 `issue_wave/WAVE_CLOSE.md`（增量入口）
  - 第三步读 `ATOM_ISSUE_DRAFT.md`（如需 file）

如未来增 wave 7+，建议在 wave 7 close packet 同时维护一个"自 HANDOFF_PACKET 以来的所有 wave close 列表"作为 super-handoff 索引，避免下任 Lead 需要逐 wave 翻找。

---

**issue_wave 至此 CLOSED。** Lead 交付物已就位，等用户决策 file / next wave。
