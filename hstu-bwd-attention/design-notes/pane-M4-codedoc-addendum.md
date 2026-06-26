# 补充要求(并入正在进行的 M4 文档任务)

你在为写 M4 文档而逐文件 Read 代码——**顺便做代码 review**,别只描述、要带审视的眼睛。在读每段 M4 代码时,对照 DESIGN 语义 + reference oracle + M1–M3 既有约定,主动找问题:

## review 关注点(发现就记)
- **正确性**:per-group `i_group=i_batch/num_batch_per_group` 索引边界;`scale_p` fallback(`group_attn_scale[i_group]?:1/group_max_seqlen_q[i_group]`)与 reference 是否一致;min_full 钳制公式;alpha 是否真全局(没误当 per-group);num_target per-batch 索引。
- **双 pipeline 运行时选**:with-local/without-local 选择条件 `window>0` 是否覆盖所有情形(window=0 但 causal? contextual-only?);两 pipeline 的 GetSmemSize=max 是否正确;有无某 group 落入错误分支。
- **零回归风险**:no_group kernel/dispatch 是否真未改;group 改动有没有意外影响 batched/jagged 路径或公共 header。
- **边界/隐患**:packed offset 溢出、grid 按 max_seqlen 开导致的越界 early-exit 是否对所有 group 成立、dq_acc workspace sizing(group 用 total_dq_acc_elems)是否正确、host supplement 数组长度(per-group vs per-batch)有没有越界(像 M2 的 num_targets 越界那种)。
- **一致性**:与 M3 jagged 的 offset 索引是否真复用(还是复制了一份可能漂移)、命名与 fwd group params 是否对齐、容差/scale 口径。

## 产出(在文档之外另记)
- 写一份 review 结论到 `/tmp/hstu-bwd-design/M4-review-findings.md`:按 **P0(正确性错误/必改)/ P1(风险/可疑)/ P2(可选)** 分级,每条:文件:行 + 问题 + 证据(代码/reference/对拍)+ 建议。**没发现就明确写"逐项核验通过,无 P0/P1"**(别为凑而编)。
- 文档里:若有值得读者知道的点(如双 pipeline 的取舍、保守项),用 note 写;**但 P0/P1 不要藏在文档里**,要在 findings 文件单独列,以便 lead 处置。
- 能力范围内复跑佐证:可跑 `python3 test/run_bwd_tests.py` 或单档 harness 命令确认你怀疑的点(尤其异构 group)。
- **不擅自改 kernel 代码**(除非纯文档侧);P0/P1 交 lead 决定。

铁则:诚实——发现真问题比"全绿好看"重要;但也别把设计既定的取舍(如双 pipeline 体积、Y-range 保守)误报成 bug。
