# Recall 使用指南（实战版）

更新日期：2026-04-28
适用对象：本仓库 maintainer + 后续接手 Step-3.5-Flash 工作的同事
配套文件：MEMORY.md（自动加载）、project_summary/（GitHub 公开文档）、recall（私人持久化）

---

## 第一章：最快上手（3 步）

如果你只看一节，看这一节。从零到能用 recall，三条命令。

### Step 1：在某个 git repo 里 init（**必须先 cd 进去**）

```bash
cd /home/hanchang/junlin12_repos/ATOM            # 必须是 git repo
/recall-init
```

会发生什么：
- 从 `git remote get-url origin` 解析出项目名（例如 `ATOM`）
- 创建 `/root/.local/share/claude/recall/ATOM/` 目录
- 写入 `directives.md`（auto-save 规则 + 强制规则）、空的 `knowledge/index.md`、`workflows/index.md`、`user.md`
- 如果当前不在默认分支，问你 "What's the epic/goal for this branch?"，然后建 `branches/<分支名>/` 目录

> 关键陷阱：如果你在 `/home/hanchang`（家目录、非 repo）开 session，`RECALL_BRANCH=DETACHED` + `RECALL_PROJECT=unknown`，**所有 branch / task 命令全部空转**。本次 Step-3.5-Flash 任务就是这种状态。

### Step 2：开始一个新 task（每个独立子目标一个）

```bash
cd /home/hanchang/junlin12_repos/ATOM
git checkout -b feat/step35-fp8-tp4
/task-create fp8-blockscale-tp4
# 提示输入 goal 时填一句话目标，例如 "让 Step-3.5-Flash FP8 在 tp=4 跑通"
```

会发生什么：
- 在 `branches/feat-step35-fp8-tp4/tasks/fp8-blockscale-tp4/` 下建 `status.md`、`knowledge.md`、`workflows.md`
- meta.md 的 `Active Task` 字段更新

### Step 3：发现新事实立即写入

```bash
/recall-add tp4-longseq-bos-debug --branch
# 然后描述发现，例如 "tp=4 长序列触发 BOS 输出，根因 aiter glm5_bf16_tuned_gemm.csv line 45 ASM kernel ..."
```

会发生什么：
- 检查发现是 `[OBSERVED]` 还是 `[VERIFIED]`（推断/假设会被拒，让你写到 `status.md`）
- 写入 `branches/<branch>/knowledge/tp4-longseq-bos-debug.md`，更新 branch 的 index.md
- 终端回复 "Saved to branch/knowledge/tp4-longseq-bos-debug.md — <reason>."

完成。从此每次开 session 在同一个 repo 里，`/recall-status` 就能看到状态。

---

## 第二章：日常工作流（按场景）

每个场景三段式：**什么时候用 → 用什么命令 → 期望什么输出**。

### 场景 A：开始一段新工作前先看现状

**时机**：刚 cd 进 repo，准备开新 task 前。

**命令**：
```bash
/recall-status
```

**期望输出**（在 feature 分支上）：
```
Recall Status
  Project:  ATOM (/root/.local/share/claude/recall/ATOM/)
  Branch:   feat/step35-fp8-tp4 (parent: main, mode: full)
  Task:     fp8-blockscale-tp4 (status: in-progress)
  Topics:   11 project + 3 branch overlay
  Workflows: 2 project + 0 branch overlay
  Config:   auto-save: auto, confidence: observed
  Alerts:   1 stale branch, 0 merged needing promotion
```

如果看到 `Branch: DETACHED` 或 `Project: unknown` —— **立刻退到正确目录重开 session**，不要继续工作。

### 场景 B：调试中发现一个根因

**时机**：实验跑完，确认了某个 bug 的根因。

**命令**（branch 局部知识）：
```bash
/recall-add tp4-longseq-bos-debug --branch
```

**命令**（普适规律，直接写到 project 级）：
```bash
/recall-add gfx950-mfma-kpack32-constraint --project
```

**期望输出**：终端回复保存路径 + 决策原因。文件内容会带 `confidence: VERIFIED` 头，并自动追加证据引用（实验 log 路径、commit hash）。

> 决策提示：commit hash、临时 workaround、WIP 状态用 `--branch`；硬件行为、kernel bug、调试方法论、性能模式用 `--project`。

### 场景 C：需要回查"上次怎么解决的"

**时机**：想起 "tp=4 长序列 BOS bug"，但记不清细节。

**命令**：
```bash
/recall-search "tp=4 BOS"
```

**期望输出**：分组结果（按 project / branch / task / archive），含文件路径和上下文行：
```
[project] knowledge/tp4-longseq-bos-debug.md:18
  - 该 ASM kernel 对非对齐 M（实测 8209-8223）输出完全错误...
[archive: feat-step35-moe] tasks/o-proj-debug/knowledge.md:42
  - forward hook 定位到 o_proj，单独 torchrun 测 all-reduce → 正确...
```

比手工 `Grep` MEMORY.md 快，且会同时搜到 archive（已合并分支的归档）。

### 场景 D：完成一个 task

**时机**：实验闭环、验证 PASS、准备 commit。

**命令**：
```bash
/task-complete
```

**期望输出**：
- 列出本 task 所有 unsaved findings，问"是否 promote 到 branch level"
- 把 task `status.md` 标记 `Status: completed`
- meta.md 的 `Active Task: none`
- 提示 "Remaining tasks: <list>. Switch to one, or create a new task?"

如果 task 失败放弃用 `/task-abandon`，它会额外问 "Why is this task being abandoned?"，把负面知识（"approach X 不行，因为 Y"）也保留下来。

### 场景 E：分支合并后晋升知识

**时机**：`git merge feat/step35-fp8-tp4 main` 之后。

**命令**：
```bash
/promote                       # 自动检测所有已合并但未 promote 的 branch
/promote feat/step35-fp8-tp4   # 或者指定单个
```

**期望输出**：
- 自动分类：架构模式 / 硬件行为 / 调试技术 → promote；commit hash / WIP / workaround → archive
- branch 目录从 `branches/` 移到 `archive/`，meta.md 标 `Status: promoted`
- 终端回复 "Promoted N findings from '<branch>' to project level. Branch archived."

### 场景 F：定期清理（看哪些 branch 还在）

**时机**：每周 / 每个 milestone 一次。

**命令**：
```bash
/branch-status            # 列所有 branch（active / stale / merged unpromoted / archived）
/recall-changelog 14      # 最近 14 天哪些 knowledge 文件被改过
```

**期望输出**：
- `/branch-status`：分类总览，标出 stale（>30 天没 commit）和 unpromoted
- `/recall-changelog`：按时间反序的修改列表，每条带文件路径 + 第一行摘要

### 场景 G：放弃整个 branch

**时机**：方向走错，整个分支不要了。

**命令**：
```bash
git checkout feat/dead-end
/branch-abandon
```

**期望输出**：问放弃原因 → 列出 branch 内所有 finding → 问哪些要 promote 到 project 级（特别是负面教训）→ 移到 archive，meta.md 标 `Status: abandoned`。

---

## 第三章：与 MEMORY.md / project_summary 的分工

三套系统**互补不替代**，明确分工才不会脑裂。

| 维度 | MEMORY.md | recall | project_summary |
|------|-----------|--------|-----------------|
| **位置** | `/root/.claude/projects/.../memory/MEMORY.md` | `/root/.local/share/claude/recall/<project>/` | `/home/hanchang/project_summary/` |
| **加载方式** | 每次 session 自动注入 system prompt | startup 注入 directives + user.md，其余按需 | 手工 Read + GitHub 浏览 |
| **可见性** | LLM 永远看得到 | 必须主动 `/recall-status` 或 `/recall-search` | 任意 reader（GitHub 公开） |
| **结构控制** | 完全自定义 | 固定模板（status.md / knowledge.md / workflows.md） | 完全自定义 |
| **依赖 git repo** | 不依赖 | **依赖**（DETACHED 状态全瘫痪） | 不依赖（自身是 repo） |
| **跨机器同步** | 无 | 无（私人本地） | git push 同步 |
| **适合内容** | 几十行内的高密度状态速查 + topic 索引 | task 中间状态、in-flight 知识、跨 session 个人记忆 | 对外发布的最终技术文档 |
| **不适合** | 长篇细节、commit 列表 | 公开分享、文档化叙述 | 频繁更新的 in-flight 状态 |
| **更新频率** | 里程碑级（每周） | 实时（每个发现） | 任务收尾（每次） |

### 推荐分工原则

1. **MEMORY.md 只放"指针 + 结论"**：路径表、状态速查、topic 索引（指向 recall 或 project_summary）。绝不复制粘贴细节。
2. **recall 放"过程 + 中间状态"**：每个 task 的 status.md、debug 思路、未 promote 的 branch knowledge。
3. **project_summary 放"最终结论 + 复现指南"**：跑通的命令、最终性能数字、commit hash 列表、新人能跟着复现的 step-by-step。

### 重叠 = 浪费 + 脑裂风险

本次任务里 `recall/hanchang/knowledge/step35flash-fp8.md` 和 `project_summary/05_fp8_inference.md` 大量重叠，造成两边状态不同步（recall 那份停在"待启动"，project_summary 显示 FP8 跑通）。**避免方法**：recall 写完后 `/task-complete` → `/promote` → 内容定型 → 把 project_summary 写成"指向 recall + 加 GitHub 友好的叙述"，不重写细节。

---

## 第四章：本次任务的复盘

### 为什么 RECALL_BRANCH=DETACHED？

- 本次 session 的 cwd 是 `/home/hanchang`（家目录）
- `/home/hanchang` 不是 git repo（task header 也确认 `Is directory a git repo: No`）
- recall 启动脚本拿不到 `git branch --show-current`，fallback 到 DETACHED
- 同时 `git remote get-url origin` 也失败，project fallback 到 `unknown`
- 后果：`/task-create` 直接拒绝（"Tasks belong to feature branches"），`/promote`、`/branch-status` 全部空转，auto-save 没有 branch 可挂载

但 startup 仍然加载了 `/root/.local/share/claude/recall/hanchang/directives.md`（这是按 user 名而非 project 名兜底找的），所以 4 条强制规则 + auto-save 配置仍生效。

### 如果重来，关键时点应该用什么 recall 命令？

下面是本次任务的实际时间线，标注**应该但没用**的 recall 命令：

| 时点 | 实际做了什么 | 应该用 recall 做什么 |
|------|--------------|---------------------|
| 2026-04-22 任务启动 | 在 `/home/hanchang` 开 session | `cd /home/hanchang/junlin12_repos/ATOM && /recall-init` 把 cwd 切到 repo，让 RECALL_BRANCH 生效 |
| 2026-04-22 创建 feat 分支 | `git checkout -b feat/step3p5-moe-swiglustep` | 同步 `/task-create moe-pipeline-fix` 把 V01 单独追踪 |
| 2026-04-23 修完 MoE V1/V3 dispatch | 直接手写 memory/moe-kernels.md | `/recall-add gfx950-moe-v1-broken --project`（这是普适硬件行为，符合 auto-save.auto） |
| 2026-04-24 tp=2/tp=4 stage1 修完 | 手写 memory/tp48-fixes.md | `/task-complete moe-pipeline-fix` 然后 `/task-create tp-stage1-padding` |
| 2026-04-25 tp=4 长序列 BOS bug 定位 | 手写 memory/tp48-fixes.md + 后来又同步到 recall/tp4-longseq-bos-debug.md | `/recall-add tp4-longseq-bos-debug --branch` 一次完成；调试方法论那段可以单独 `/recall-add longseq-bug-bisect-methodology --project` |
| 2026-04-25 V01-V07 验证全 PASS | 手写 verification_pipeline/results/SUMMARY.md | 每个 V0x 用 `/task-create v0x-verification`，跑完 `/task-complete`，最后 `/promote` |
| 2026-04-26 FP8 tp=2/tp=4 跑通 | 手写 memory/fp8-work.md + 重复写 recall/step35flash-fp8.md | `/recall-add fp8-blockscale-quant --project`，避免两份维护 |
| 2026-04-28 任务收尾 | 手写 project_summary/01-12 | `git merge && /promote` 把 branch knowledge 自动晋升，project_summary 只写"叙述 + 指针" |

### 三条最大教训

1. **session 入口 cwd 决定一切**：必须先 cd 到 git repo 再开 session，否则 recall 半瘫痪。建议把"必须从 junlin12_repos/ 操作"这条规则扩展为"必须从 junlin12_repos/ 开 session"。
2. **同一份知识不要两边写**：要么 recall + 在 MEMORY.md 留指针，要么 project_summary + 在 MEMORY.md 留指针。不要 MEMORY.md 自身存细节再分别同步两边。
3. **task 概念被严重浪费**：V01-V07 天然就是 7 个 task，每个有自己的"中间假设 → 验证 → 结论"过程。这段历史现在只活在对话上下文里，session 一结束就丢了。

---

## 第五章：命令速查表

### Setup & Info

| 命令 | 参数 | 用途 | 返回 |
|------|------|------|------|
| `/recall-init` | — | 当前 git repo 一次性初始化 | 创建 `<root>/<project>/` + 默认 directives + 空 index |
| `/recall-status` | — | 看当前 branch / task / 知识量 | 6-7 行紧凑状态块 |
| `/recall-help` | — | 列所有命令 | 帮助文本 |

### Knowledge

| 命令 | 参数 | 用途 | 备注 |
|------|------|------|------|
| `/recall-add <topic>` | `--project` / `--branch` 控制 scope | 写入新发现 | 必须是 `[OBSERVED]` / `[VERIFIED]`，假设会被拒 |
| `/recall-search <query>` | 关键词 | 跨 project + branch + task + archive 全文搜索 | 结果按 scope 分组 |
| `/recall-changelog [days]` | 天数（默认 7） | 最近改动的 knowledge 文件 | 按时间反序 |

### Branches

| 命令 | 参数 | 用途 | 何时用 |
|------|------|------|--------|
| `/branch-status` | — | 所有 branch 总览（active / stale / merged / archived） | 周期清理 |
| `/promote [branch]` | 可选 branch 名（默认自动检测已 merge 的） | branch 知识晋升到 project 级 | merge 之后 |
| `/branch-abandon` | — | 当前 branch 整体放弃 + 抢救负面知识 | 方向走错时 |

### Tasks

| 命令 | 参数 | 用途 | 前置条件 |
|------|------|------|----------|
| `/task-create <name>` | kebab-case 名字 | 在当前 branch 下建新 task | 必须在 feature 分支（不能是 main/develop） |
| `/task-switch <name>` | 名字或部分匹配 | 切换 active task | 当前 branch 已有该 task |
| `/task-complete [name]` | 可选 task 名 | 收尾 task + 复盘知识 | 默认收尾 active task |
| `/task-abandon [name]` | 可选 task 名 | 放弃 task + 抢救负面知识 | 会问 abandon 原因 |

### 配置文件速查

| 文件 | 路径 | 改什么 |
|------|------|--------|
| `directives.md` | `<root>/<project>/directives.md` | auto-save.auto/ask/never 规则、stale-branch-days、default-branch、强制规则 |
| `user.md` | `<root>/<project>/user.md` | WORKSPACE / CONTAINER / 编辑器偏好等 key-value |
| `knowledge/index.md` | `<root>/<project>/knowledge/index.md` | topic 列表（recall-add 自动维护） |
| `branches/<name>/meta.md` | branch 元数据 | active task、parent、mode、HEAD 日期 |

### auto-save 规则（来自本机 directives）

- `auto`：hardware findings、build errors、debugging root causes、kernel behaviors、GPU architecture、verified inference results
- `ask`：coding conventions、architecture decisions、new model support
- `never`：temporary workarounds、debugging session logs、speculation

`auto` 类发现命中关键词时，agent 会直接写入；`ask` 类会先问；`never` 类绝不写入 knowledge（只能进 task 的 status.md）。

---

## 最高价值的 3 个 recall 功能

如果时间有限，先用这三个，能拿到 80% 的收益：

1. **`/recall-search <query>`** —— 跨 project / branch / task / archive 全文搜索。比手工 Grep MEMORY.md 强一个数量级，特别是项目历史长了之后回查"上次怎么修的"。本次任务里多次需要回看历史发现，每次都是手工 grep，浪费时间。

2. **`/task-create` + `/task-complete`** —— 给"思考过程"一个持久化容器。V01-V07 这种验证序列、tp=4 BOS bug 这种多假设排查，中间状态不持久化等于每次都要重新载入上下文。task 的 `status.md` 就是为这个设计的。

3. **`/recall-add <topic> --project` / `--branch`** —— 强制每条新发现立即归位。配合 auto-save.auto 的关键词，新事实落地几乎零成本。本次最大的浪费是同一个 finding 在 memory/、project_summary/、recall/ 写了三遍——用 `/recall-add` 一次写入 + project_summary 引用，能把工作量砍掉一半。

剩余功能（`/promote`、`/branch-abandon`、`/recall-changelog`）属于"用熟之后才会觉得离不开"的辅助层，初期可以不强求。
