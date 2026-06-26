# 在新机器恢复 memory

这两个文件是当时机器上 Claude Code 的持久 memory:

- `MEMORY.md` — auto-memory 索引(每次 session 加载进上下文)
- `hstu-bwd-project.md` — HSTU bwd 项目 memory(设计已批准、进度、恢复必读指针)

## 放到哪

新机器 cc 的 memory 目录是 `~/.claude/projects/<workspace-slug>/memory/`,其中 `<workspace-slug>` 由 workspace 绝对路径派生(把 `/` 换成 `-`,去掉首个 `/`)。例如 workspace = `/root/workspace` → slug = `-root-workspace`,完整路径:

```
/root/.claude/projects/-root-workspace/memory/
```

```bash
DEST=/root/.claude/projects/-root-workspace/memory   # 按你的 workspace 路径改 slug
mkdir -p "$DEST"
cp MEMORY.md hstu-bwd-project.md "$DEST"/
```

## 注意 / 需要本地适配

- `MEMORY.md` 里的环境段(home=/root、claude CLI 路径、PAT 已配在 `~/.git-credentials`、tmux/agent-team bootstrap 等)是**当时那台容器**的事实。新机器若不同(尤其 home 不是 `/root`、没配 git PAT),按实际情况修订对应行。
- **PAT**:memory 文本里**不含 token 值**(只记了"token 存在本地 ~/.git-credentials")。新机器要 push 需你自己重新配置 fine-grained PAT(user LJ-underdog,Contents:write,且把 `composable_kernel` 加进 repository access)。
- `hstu-bwd-project.md` 里指向 `HANDOFF.md` 的路径(`/root/workspace/hstu-bwd-impl/docs/HANDOFF.md`)在新机器上需先把本库 `../HANDOFF.md` + `../impl/` 落到对应位置(见 REPRODUCTION_GUIDE 第 4 节),或直接读本库根的 `HANDOFF.md`。
