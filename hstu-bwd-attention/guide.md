# GUIDE — 给接手的 Claude Code:怎么读这套文档

> 你是一个**新机器上的 Claude Code 实例**,刚 clone 到这个 `hstu-bwd-attention/` 目录。
> 本文件告诉你**按什么顺序读、每份文档回答什么问题、想干某件事该翻哪里**。先把这一篇读完,再动手。

## 0. 一句话背景

为 HSTU attention 的 **backward** 从零实现 GPU kernel(gfx950/CDNA4),复用 ck_tile FMHA bwd 基建。
M0–M8 + cross-attention 全部完成(promoted、已提交)。代码在 fork,不在本库;本库 = 上下文 + 文档 + 测试 + 讲义 + 复现指针。

## 1. 固定开场三步(照做,别跳)

1. **恢复 memory** → 读 `memory/INSTALL.md`,把 `memory/*.md` 放进你的 `~/.claude/projects/<slug>/memory/`。
   这样你即使 `/clear` 也始终带着 HSTU 项目上下文(谁在做、做到哪、关键约束)。
2. **读 `HANDOFF.md`** —— **活状态的唯一权威**。重点读这几节:
   - `§0 一句话(当前)` + `§0b 本会话收尾状态` —— 现在到底在哪
   - 文末 `能力边界(现)` —— 支持什么/reject 什么/out-of-scope 什么
   - `§6 下一步候选` —— 可以接着做什么(每条都标了 ROI)
   - `§7 关键教训 / 决策` + `§3 铁律` —— 别重蹈的坑
3. **想动手跑** → 跟着 `REPRODUCTION_GUIDE.md` 走(拉码 → 构建 `-DBUILD_DEV=OFF` → 跑套件到 253/253)。

做完这三步,你就和"上一台机器的 lead"站在同一个起点了。

## 2. 各文档回答什么问题(按需翻)

| 我想知道… | 翻这里 |
|---|---|
| 现在整体到哪了、下一步能做啥、有哪些坑 | `HANDOFF.md`(活状态权威) |
| 代码在哪、哪个分支/commit、里程碑→hash 映射 | `code_location.md` |
| 怎么在新机器构建+跑测试 | `REPRODUCTION_GUIDE.md` |
| 某个里程碑**具体怎么实现的**(图文,首选) | `reports/hstu-bwd-<Mx>-*-20260625.html`(14 篇统一体例讲义) |
| 某里程碑的**实现/评审原始记录**(更深) | `design-notes/M*-done.md`、`*-review-findings.md`、`draft-*.md` |
| 整体设计/架构决策的来龙去脉 | `design-notes/DESIGN.md` |
| 哪些组合验证过、误差/加速实测值 | `impl/candidates.jsonl`(候选账本)、`impl/benchmark.csv` |
| 回归测试怎么组织的 | `impl/test/run_bwd_tests.py` + `impl/test/README.md` |
| ⚠ 哪些旧文档**已作废别采用** | `design-notes/README.md`(尤其上游 fwd "bug" 那批) |

## 3. 决策树:你想做什么

- **只想了解、不改代码** → 看 `reports/` 的 `-20260625` 讲义(M0→M8 顺序),配 `HANDOFF.md` §0/能力边界。
- **想复现验证现状** → `REPRODUCTION_GUIDE.md` 全程,跑到 `253/253 exit 0`。
- **想继续做新里程碑/优化** → `HANDOFF.md §6 下一步候选`(B1 group TU 拆分 / B6 trload / 近似 sigmoid 都是 M8 残项,ROI 已标);先读对应的 `design-notes/draft-*.md`、`M8-INV-findings.md`。动手前遵守 `HANDOFF.md §3 铁律` + skill `rocm-kernel-design`(task contract + 每候选对拍验证)。
- **想做 out-of-scope 扩展**(target_in_kv / 独立 dO layout / 非方形 tile)→ 这些需用户拍板;见 `HANDOFF.md` 能力边界 + cross-attn draft。
- **想对上游提 PR** → 尚未提;代码在 fork `hstu_attention_fwd_bwd_v2`。注意 `design-notes/README.md` 标注的"上游 fwd group_max_seqlens_q 非 bug,别误报"。

## 4. 反踩坑(高频,务必记住)

1. **commit hash 用 `code_location.md` 的映射表,别用 `HANDOFF.md` 正文里的旧 hash** —— git 历史 2026-06-25 重写过,HANDOFF 部分旧 hash 已 stale;权威映射在 `code_location.md` / `design-notes/doc-series-spec.md §5`。
2. **构建必须 `-DBUILD_DEV=OFF`** —— 否则 `-Werror -Weverything` 把新 clang 诊断当错,fwd 都编不过。
3. **对拍必须 `-attn_scale=1.0`** —— 让梯度量级有意义,别让 ref 太小巧合 PASS。
4. **硬件结论查 rocm-ref(`/tmp/rocm-ref/`,gfx950=CDNA4),别臆造** —— 例:CDNA4 每 CU LDS = **160KB**(不是 64KB,那是 CDNA3)。
5. **silent-wrong 比 throw 危险** —— 未支持组合要么对拍覆盖、要么显式 throw;新特性要 **causal × mask因子 交叉覆盖**(P1-1/M6b 两次覆盖洞都因只测对角线)。
6. **`reports/` 里有 23 篇旧版/总览**(早期日期)与 14 篇 `-20260625` 统一系列并存;**首选看 `-20260625` 那套**。旧的保留作参考,个别有已知 stale 行号。
7. **别采用 `design-notes/README.md` 点名的作废上游 "bug" 草稿**。

## 5. 想还原多 pane 工作流(可选)

上一台机器用 tmux + skill `agent-team` 跑 lead+3 teammate 的派单/review 流程(写文档、并行 review)。新机器若装了该 skill:见 `REPRODUCTION_GUIDE.md §6`。不需要的话,单实例照本 guide 也能完整接手。

---
**TL;DR**:`memory/INSTALL.md` 恢复记忆 → `HANDOFF.md` 看现状 → 要跑就 `REPRODUCTION_GUIDE.md` → 要懂某里程碑看 `reports/*-20260625.html` → hash 以 `code_location.md` 为准 → 别碰 `design-notes/README.md` 标的作废项。
