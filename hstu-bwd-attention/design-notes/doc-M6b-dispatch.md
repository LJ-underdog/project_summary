# 派单:写 M6b 讲义(group deterministic + 修 O1 + 修 harness bug)

**先读硬规格:`/tmp/hstu-bwd-design/doc-series-spec.md`(全文)。本卡给 M6b 专属输入。**

- **里程碑**:M6b = determ 扩到 group(M6 determ × M4/M5b group)+ 修 O1(group entry 硬编码 false 致 determ 不可达、静默 atomic)+ 顺带挖出并修一个 pre-existing harness bug。
- **commit(行号锚定)**:`ecda0f06`。`cd /root/workspace/ck_hstu && git show --stat ecda0f06`。
- **旧 HTML(参考叙事,行号/数值必重核)**:`/root/workspace/hstu-b1052-report/hstu-bwd-M6b-group-determ-20260610.html`。
  - ⚠ 旧讲义关于 CDNA4 LDS/occupancy 的硬件数曾出过错(64KB→实测 160KB),与 M6b 主题无关,别抄那类硬件结论;本篇只讲 group determ + O1 + harness bug。
- **事实来源**:`/tmp/hstu-bwd-design/M6b-done.md`(+ fix-review 若在)+ `candidates.jsonl` 里 `"id":"M6b-group-deterministic"` 那条(91/91 exit0)。
- **输出**:`/root/workspace/hstu-b1052-report/hstu-bwd-M6b-group-determ-20260625.html`
- **M6b 讲解重点**:
  1. group determ:复用 M6 POST reduce + determ 机制;group 两 kernel dq_acc set+split 分叉、group params 加 split_stride/num_splits、group dispatch determ 分支。
  2. **修 O1**:group entry `BOOL_SWITCH_2`→`BOOL_SWITCH_3` 接 determ 轴(此前 group+determ 静默走 atomic,不可复现)。
  3. **harness bug(诚实讲,有强教训)**:`group_max_seqlens_q` 用组下标索引 per-batch `num_targets` + 单组 uih-max → 组内多 batch 异 seqlen 时**低估 max_seqlen_q** → PRE `dot_do_o`(grid 按 max_seqlen_q)漏算最长 batch 尾 token 的 D → 垃圾 D → **仅 softmax target 行 dQ 错**(dK/dV 不受)。**库逻辑/reference 本就正确**,纯 harness setup bug。修:公式改组内 `max_b(seqlen_q[b]+num_target[b])+ctx` + `HSTU_CHECK` 守卫。
  4. **教训**:又一条 P1-1 式覆盖洞(组内多 batch 异 seqlen+长 target+window 没覆盖);"repro 全绿+自述全 PASS" 仍漏一格 correctness FAIL → **独立复核(尤其对抗 formula-revert)是关键**。放进设计动机/教训。
- 写完按规格 §6 回报。
