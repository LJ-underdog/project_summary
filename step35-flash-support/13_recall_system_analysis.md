# Recall 系统在本次任务中的作用分析

调查日期：2026-04-28
调查范围：本次 Step-3.5-Flash 支持任务全周期（2026-04-22 ~ 04-28）

---

## 1. Recall 系统是什么？

Recall 是一个安装在 `/root/.claude/plugins/recall/` 的 Claude Code 插件，用于**跨 session 的结构化知识 / 任务追踪**。

### 设计目标
- 自动捕捉每次 session 中验证过的发现（auto-save），不依赖人工手抄
- 把知识按"作用范围"分层，避免重复也避免污染
- 提供任务级状态追踪（status.md），把"思考过程"和"最终结论"分开

### 核心概念

| 概念 | 含义 | 存放位置 |
|------|------|----------|
| **Project** | 一个 git repo / 工作区 | `<recall-root>/<project>/` |
| **Branch** | git 分支级的知识隔离区 | `<project>/branches/<branch>/` |
| **Task** | branch 下的一次具体工作 | `<project>/branches/<branch>/tasks/<task>/` |
| **Knowledge** | 已验证的事实（[OBSERVED]/[VERIFIED]）| `knowledge/<topic>.md` |
| **Workflows** | 复用的操作步骤 | `workflows/<name>.md` |
| **Directives** | 项目级配置 + 强制规则 | `directives.md` |

### 核心命令（来自 /recall-help）

```
Setup & Info: /recall-init  /recall-status  /recall-help
Knowledge:    /recall-add <topic>  /recall-search <q>  /recall-changelog
Branches:     /branch-status  /branch-abandon  /promote [branch]
Tasks:        /task-create <name>  /task-switch <name>  /task-complete  /task-abandon
```

### 知识流转模型（promote 机制）

```
Task knowledge (verified)
        ↓ task-complete
Branch overlay (knowledge/<topic>.md)
        ↓ promote (branch merged)
Project knowledge (general patterns / archive specifics)
```

`/promote` 会把 branch 知识分类：
- **Promote**：架构模式、硬件行为、调试技术、性能模式 → 写入 project 级
- **Archive**：commit hash、WIP 状态、临时 workaround → 仅归档不晋升

### 自动行为
- 检测新 branch 自动建目录
- 按 `auto-save.auto/ask/never` 规则决定是否保存某条发现
- 检测 merge 自动触发 promote 提醒
- 根据 `stale-branch-days` 标记陈旧 branch

---

## 2. 本次 Session 的 Recall 状态

### Session startup 加载的内容

```
RECALL_BRANCH=DETACHED
RECALL_PROJECT=unknown
RECALL_ROOT=/root/.local/share/claude/recall
```

### `RECALL_BRANCH=DETACHED` 的含义

DETACHED 表示 **当前工作目录不在任何 git repo 内，或者 git HEAD 处于 detached 状态**。本次 session 的 cwd 是 `/home/hanchang`，这是用户家目录，不是 git repo（任务里也确认 `Is directory a git repo: No`）。

**后果**：
- 所有 branch 级、task 级机制都无法生效（没有 branch 可挂靠）
- `/task-create` 会直接拒绝："Tasks belong to feature branches. Create or switch to a branch first."
- `/promote` 没有 branch 可处理
- 自动 promote / 自动 branch 检测全部失效

### `RECALL_PROJECT=unknown` 的含义

启动时无法从 `git remote get-url origin` 解析出项目名，于是 fallback 到 `unknown`。
对应目录 `/root/.local/share/claude/recall/unknown/` 里有完整的 directives + 空 knowledge/workflows index。

### 实际加载的 directives

system-reminder 里直接显示了 `/root/.local/share/claude/recall/hanchang/directives.md` 的内容（**注意 project 名是 `hanchang` 而不是 `unknown`**，说明启动脚本至少把 `hanchang/` 这个 user-level 目录的 directives 也读了一份）：

```
auto-save.auto: [hardware findings, build errors, debugging root causes,
                 kernel behaviors, GPU architecture, verified inference results]
auto-save.ask:  [coding conventions, architecture decisions, new model support]
auto-save.never:[temporary workarounds, debugging session logs, speculation]
auto-save.default: auto
promotion: auto
stale-branch-days: 30
default-branch: main
confidence-min: observed
```

外加四条强制规则：必须实证、gfx950 ≠ gfx942、cd /tmp 跑 python、清 ATOM 缓存。

### `hanchang/` 项目其实有真实数据

虽然 RECALL_PROJECT=unknown，但 `/root/.local/share/claude/recall/hanchang/knowledge/` 下其实有 **11 个 topic 文件**，是过去 sessions 积累的：

```
aiter-import-fix.md            gfx950-moe-kernels.md
moe-pipeline-fix-task.md       step35flash-fp8.md
step35flash-tp-debug.md        swiglustep-gap.md
swiglustep-wiring-handoff.md   swiglustep-wiring-progress.md
tp4-longseq-bos-debug.md       triton-moe-replacement.md
index.md
```

这些都是和本次任务**直接相关**的知识。最近一次更新是 2026-04-25（V01-V07 验证 pipeline 完成那天）。

---

## 3. 本次任务实际使用的知识管理工具

| 工具 | 路径 | 作用 | 加载/写入方式 |
|------|------|------|----------------|
| **MEMORY.md** | `/root/.claude/projects/-home-hanchang/memory/MEMORY.md` | 工作原则、环境、路径、状态速查、topic 索引 | 每次 session 自动加载到 system context |
| **memory/ 子文件** | `memory/{moe-kernels,tp48-fixes,fp8-work}.md` | 主题深度笔记，被 MEMORY.md 索引 | 按需 Read |
| **project_summary/** | `/home/hanchang/project_summary/step35-flash-support/` | 12 篇结构化技术文档 + verification_pipeline 结果 | 任务结束时手动 Write，git push 到 GitHub |
| **本地工作目录** | `/home/hanchang/project_fp8_tp4/`、`project_moe_no_padding/` 等 | 单次实验的 prompt、log、临时脚本 | 实验过程中临时产物 |
| **junlin12_repos/** | `/home/hanchang/junlin12_repos/{ATOM,aiter}/` | 真正的代码 commit/push 源 | git 操作 |

### 各自分工

- **MEMORY.md** = "脑内常驻索引"：每次必读的高密度状态摘要
- **project_summary** = "对外可读的最终文档"：他人读 GitHub 就能复现
- **memory/*.md** = "中间深度笔记"：MEMORY.md 装不下的细节
- **本地工作目录** = "草稿 / 实验产物"：寿命短

---

## 4. Recall 系统 vs 实际使用工具的对比

### 功能对照表

| 能力 | Recall 系统 | 实际使用 | 谁更适合 |
|------|------------|----------|----------|
| 跨 session 状态加载 | ✓（startup 注入 directives + user.md） | ✓（MEMORY.md auto-load） | 平手，但 MEMORY.md 显示在 system prompt 里更显眼 |
| 主题知识库 | ✓（knowledge/`<topic>`.md） | ✓（memory/*.md + project_summary/） | 各有优势：recall 自动归档、project_summary 公开 |
| 跨 branch 隔离 | ✓（branches/`<name>`/） | ✗ | **Recall 独有** |
| 任务进度追踪 | ✓（task status.md） | ✗ | **Recall 独有**（实际用对话上下文代替） |
| 自动保存触发 | ✓（按 auto-save 规则） | ✗（全手工） | **Recall 独有但本次未生效** |
| Knowledge 搜索 | ✓（/recall-search） | ✗（手工 Grep） | Recall 略胜 |
| 公开可分享 | ✗（在 ~/.local/share/） | ✓（project_summary 推 GitHub） | **MEMORY/project_summary 独有** |
| 复现指南 | ✓（workflows/） | ✓（12_reproduction_guide_fp8_tp4.md） | 平手 |

### 重叠程度

`hanchang/knowledge/step35flash-fp8.md` 和 `project_summary/step35-flash-support/05_fp8_inference.md` 是**严重重叠**的——同样的 bug 描述、同样的 L904 fix。本次任务相当于把同一份知识手写了两遍。

### 互补部分

- Recall 的 **directives 强制规则**（gfx950 ≠ gfx942、cd /tmp）确实在 system prompt 里强化了，这是 MEMORY.md 同时有写、但 recall 启动时也额外注入了一次
- Recall 的 **task status.md** 概念在 MEMORY.md / project_summary 里没有对等物——本次任务的"思考过程"基本只活在对话历史里，session 结束就丢了

### 没用到的 Recall 能力

1. `/task-create` + `status.md`：把 V01-V07 当成 7 个 task 追踪
2. `/branch-status`：跨任务的 branch 概览
3. `/promote`：把 task 知识晋升到 project 级（本次连 branch 都没有）
4. `/recall-search`：例如本次想查"tp=4 BOS bug"时，实际是手工 grep MEMORY.md
5. `/recall-add --project`：自动把新发现写入正确层级
6. auto-save 自动触发：本次所有写入都是手动 Write 调用

---

## 5. 是否"用错了"？

### 直接结论：**不算错，但浪费了系统设计**

更准确地说：用户**绕开了 recall 系统**，自己用 MEMORY.md + project_summary 重新发明了一套等价但更轻量的知识管理流程，并在本次场景下其实更合适。

### Recall 该用但没用的地方

1. **没有 `/recall-init` 在某个 git repo 里**
   后果：RECALL_PROJECT=unknown，所有 branch/task 机制全部空转。如果在 `junlin12_repos/ATOM/` 或 `junlin12_repos/aiter/` 里 init，本次 V01-V07 7 次验证、tp=4 BOS bug 调试、FP8 上线就有真正的 branch/task 结构。

2. **没有用 task 系统追踪 V01-V07**
   每次 verification 实验本来就是一个 task。用 `/task-create v06-fp8-tp4-perf` + status.md 比放在对话上下文里更耐久。

3. **没有调 `/recall-add` 写入新发现**
   tp=4 长序列 BOS bug 这个根因本来完全符合 `auto-save.auto: [debugging root causes, kernel behaviors]`，应该 `/recall-add tp4-longseq-bos-debug` 触发自动写入，而不是手动维护两份（recall 一份 + MEMORY.md 一份）。

4. **没有用 `/recall-search`**
   本次多次回看历史发现时都是手工 Grep MEMORY.md 或翻 project_summary。

### 替代方案（MEMORY + project_summary）反而更合适的地方

1. **公开可分享**：project_summary 在 GitHub 上，team 其他人能直接看；recall 在 `~/.local/share/` 私人目录里，离开本机就丢。
2. **每次 session 启动就强制看见**：MEMORY.md 在 system prompt 里，没法忽略；recall 的 knowledge index 必须主动调 `/recall-status` 才看得到。
3. **用户对结构有完全控制**：MEMORY.md 的"性能速查"表格、"Topic 文件索引"表格是定制结构；recall 的模板是固定的。
4. **不依赖 git repo**：本次很多操作就是在 `/home/hanchang` 这个非 repo 目录下做的，MEMORY.md 不挑 cwd。

### 失误成本

- **重复劳动**：同一份知识写两遍（recall + project_summary）
- **状态分裂风险**：recall 的 step35flash-fp8.md 是"待启动"状态（2026-04-24 写的），但 MEMORY.md 已经显示 FP8 跑通了。两边没同步。
- **失去 task 历史**：V01-V07 的中间思考过程没有持久化，下次想回顾"为什么选了 tp=4 而不是 tp=2"只能翻 project_summary 的最终结论。

---

## 6. 推荐的工作流

### 选项 A：彻底放弃 recall，加固现有方案

如果继续用 MEMORY.md + project_summary，建议：
1. 删除 `/root/.local/share/claude/recall/hanchang/knowledge/` 避免脑裂
2. 在 MEMORY.md 加一节"任务追踪"，每个大任务有 status 行
3. 在 project_summary 里加 `tasks/` 子目录追踪 in-flight 工作

### 选项 B：完整启用 recall（推荐）

如果要真正用 recall 的 task / branch / promote 能力：

#### 一次性设置
```bash
# 在 ATOM repo 里 init
cd /home/hanchang/junlin12_repos/ATOM
/recall-init                       # 创建 ATOM 项目
# 同样在 aiter repo 里
cd /home/hanchang/junlin12_repos/aiter
/recall-init                       # 创建 aiter 项目
```

#### 每次新任务
```bash
cd /home/hanchang/junlin12_repos/ATOM
git checkout -b feat/step35-fp8-tp4
# recall 自动创建 branch 目录
/task-create fp8-blockscale-fix
# 接着开始工作，关键发现用：
/recall-add fp8-blockscale --branch       # branch 级临时知识
/recall-add gfx950-mfma-kpack32 --project # 普适规律直接到 project
```

#### 任务收尾
```bash
/task-complete                     # 总结 task 知识
git merge feat/step35-fp8-tp4 main
/promote                           # 自动晋升 branch 知识到 project
```

#### 跨 session 查找
```
/recall-status                     # 看当前 branch / task / 知识量
/recall-search "tp=4 BOS"          # 跨 branch / archive 全文搜索
/branch-status                     # 看哪些 branch 还没 promote
```

### 选项 C：混合（实际最现实）

- **Recall**：日常 in-flight 状态、task status、跨 session 个人记忆
- **MEMORY.md**：精简的高密度状态速查（几十行内）
- **project_summary**：对外发布的最终技术文档（GitHub 可读）
- **三者明确分工，禁止内容重叠**：MEMORY.md 只放"指针 + 结论"，详情链接到 recall 或 project_summary

最关键的动作：**先 `/recall-init` 在某个 git repo 里**，否则 RECALL_BRANCH=DETACHED 一直会让 recall 处于半瘫痪状态。

---

## 附录：本次 Session 的判定

| 维度 | 判定 |
|------|------|
| Recall 系统是否启用 | 部分启用（directives 加载了，branch/task 全部失效） |
| 是否符合 recall 设计意图 | 不符合（DETACHED + unknown） |
| 替代方案是否合理 | 合理且高效，但**与 recall 重复**造成浪费 |
| 主要损失 | task 中间状态没持久化、知识两份维护 |
| 主要收益 | 公开 GitHub 文档质量高、cwd 灵活不挑 git |
