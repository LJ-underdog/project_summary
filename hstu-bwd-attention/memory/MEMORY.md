# Memory

## Environment (root container)
- User home: `/root` (NOT `/home/junlin12` as some skill docs assume — substitute paths accordingly)
- claude CLI: `/root/.local/bin/claude` → symlink to `~/.local/share/claude/versions/<ver>` (native ELF installer). Upgrade via `claude update`. As of 2026-06-02 on v2.1.160 (was 2.1.76). Old versions kept in `versions/` for manual rollback (each ~235MB).
- ANTHROPIC_* + CLAUDE_CODE_* env vars are present in shell but `~/.claude/container.env` does NOT exist by default
- Working dir: `/root/workspace` (not a git repo by default)
- Platform: Ubuntu (Linux 5.15), bash; apt available
- No `gh` CLI installed; use `curl` + GitHub API for repo access
- GitHub push 已配好:`credential.helper store` + `~/.git-credentials`(chmod 600)存了一枚 fine-grained PAT(user `LJ-underdog`,带 Contents:write)。对 `github.com` 的 `git push` 会自动认证,不用再贴 token。**别把 token 值写进任何文件/memory/日志**;失效就让用户重贴。该 token 由用户主动选择本地保存(未 revoke)。

## Projects
- [HSTU bwd GPU 实现](hstu-bwd-project.md) — 设计已批准(U1–U4 默认),目标芯片 **gfx950/CDNA4**,复用 FMHA bwd 基建。**进行中:M0–M4 + P1-1 修复已完成(SiLU 全模式×全mask×bf16×hd64 对拍 PASS)。恢复必读 `/root/workspace/hstu-bwd-impl/docs/HANDOFF.md`(含环境/构建 BUILD_DEV=OFF/tmux 派单 Enter-吞坑/候选账本/后续 M5-M8)。**

## Installed Skills (under /root/.claude/skills/)
- `rocm-kernel-design/` — created 2026-06-04 as `kernel-design-rocm`, **renamed + merged 2026-06-17**. Evidence-driven ROCm kernel impl+tuning workflow (CK/ck_tile/aiter/HIP/Triton, gfx942/gfx950), adapted from MIT HAN Lab KDA off CUDA: ncu→rocprofv3/rocprof-compute, KernelWiki→rocm-ref, oracle=cos_sim + rel/abs err vs CPU/torch-fp32 ref 对拍. Enforces task contract + draft→plan gate + per-candidate validation + evidence workspace (candidates.jsonl/benchmark.csv/profile/) + GPU/git discipline + §8 cross-project lessons (coverage holes/offline exhaustive validator/byte-identity gate/anti-false-positive). SKILL.md + templates/task-contract.md. **Merged with the MoE-flavored `rocm-kernel-design` from the LJ-underdog/agent_skill repo and pushed there (commit b2b343c) — local copy is byte-identical to the repo version. Old dir name `kernel-design-rocm` deleted; HANDOFF.md still cites the old name/path.**
- `html-report/` — writes 图文并茂 HTML reports (used for the FMHA/HSTU bwd 讲义 under /root/workspace/hstu-b1052-report/).
- `agent-team/` — installed from https://github.com/LJ-underdog/agent_skill (2026-05-25)
  - SKILL.md + TEAM_INSTANCE_TEMPLATE.md + templates/ (4 wave templates + 5 role yamls)
  - Added `bootstrap.sh` + `readiness.sh` adapted for `/root` paths
  - tmux 3.4 installed via apt
  - container.env dumped from current shell env (use python3 + shlex.quote — raw `env >` breaks because ANTHROPIC_CUSTOM_HEADERS contains colon+space)
  - §9.7 roundtrip smoke test PASSED on all 3 teammate panes (2026-05-25)

## Shell Config (~/.bashrc)
- `claude` is wrapped as a shell function (NOT alias) that defaults to `IS_SANDBOX=1 ... --dangerously-skip-permissions`. Calls `command claude` internally to avoid recursion. `export -f claude` so subshells inherit. Use `command claude` or `\claude` to bypass.
- Reason for function over alias: aliases don't expand in non-interactive shells (`bash -c`, scripts).

## Key Gotchas
- When dumping shell env to a `source`-able file, MUST quote values (use `shlex.quote`). Naive `env >` fails on values containing spaces/colons.
- tmux agent-team bootstrap: 4 claude processes fork in ~3-8s each; readiness check via `tmux capture-pane | grep '❯'` works reliably
- File-based dispatch (Write prompt to file + send-keys "请读取 X 并执行") is the only reliable multi-line dispatch path — never send-keys multi-line directly
