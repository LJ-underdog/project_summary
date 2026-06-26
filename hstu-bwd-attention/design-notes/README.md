# design-notes — 阅读须知

本目录是 HSTU bwd 全程的设计/评审/派单原始记录(每里程碑的 `M*-done.md` / `*-review-findings.md` / `draft-*.md` / 讲义系列 `doc-*`)。多数是历史快照,**以 `../HANDOFF.md` 为活状态权威**。

## ⚠️ 已作废文件(勿当结论采用)

下面这几份是**早期把上游 fwd `group_max_seqlens_q` 误判为 bug** 时的草稿。后经与 fwd 代码作者核实:**那不是 bug,是 example 的用法约定**(`-g_max_seqlens` 由调用方 over-provision;详见 `../HANDOFF.md` §6 的 ℹ️ 块)。**结论:不上报上游。** 以下文件措辞过度定性为 bug,**已作废,仅作历史留存**:

- `upstream-issue-draft.md` — 上游 issue 草稿(作废)
- `upstream-fwd-bug-verify.md` / `upstream-fwd-bug-verify-dispatch.md` — "验证 bug" 派单/记录(作废)
- `fwd-bug-report-dispatch.md` — 正文写"已实测确认是真 bug",**与最终结论相反**(作废)
- `../reports/hstu-fwd-group-maxseqlen-bug-20260610.html` — 同主题讲义,HANDOFF §6 自评"措辞过度定性为 bug"(作废)

新机器接手时**不要**据这些去给上游提 issue;正确口径见 `../HANDOFF.md` §6。
