# 派单:写 M4 讲义(group 模式)+ 折入 M4b(修 P1-1)

**先读硬规格:`/tmp/hstu-bwd-design/doc-series-spec.md`(全文)。本卡给 M4 专属输入。本篇要同时讲 M4 与紧随的 M4b 修复(放成一个独立 section)。**

- **里程碑**:M4 = group HSTU(per-group device 指针超参,group=jagged 超集),SiLU+bf16+hd64。**M4b** = 修 P1-1(causal=0+num_target>0+window=0 静默漏掩码)。
- **commit(行号锚定)**:M4 主体 `3573f083`;M4b 修复 `180a8acb`(讲 P1-1 那段用这个 commit 的行号)。`cd /root/workspace/ck_hstu && git show --stat 3573f083` 和 `git show --stat 180a8acb`。
- **旧 HTML(参考叙事,行号/数值必重核)**:`/root/workspace/hstu-b1052-report/hstu-bwd-M4-changes-20260608.html`。
- **事实来源**:`/tmp/hstu-bwd-design/M4-done.md` + `fix-P1-1-done.md` + `candidates.jsonl` 里 `"id":"M4-group"` 与 `"id":"M4b-fix-causal0-target"` 两条。
- **输出**:`/root/workspace/hstu-b1052-report/hstu-bwd-M4-group-20260625.html`
- **M4 讲解重点**:
  1. group = jagged 超集:packed `[1,ΣL,h,d]`+cu_seqlens(复用 M3 offset 寻址)**外加** per-group 超参,索引 `i_group=i_batch/num_batch_per_group`。
  2. alpha 全局标量;scale_p + 4 mask 参数(window/contextual/min_full/max_seqlen)per-group device 指针;num_target per-batch。
  3. **专用 group kernel** `HstuAttentionBwdDQDKDVGroupKernel`:因 per-group window 无法编译期定 `kUseLocal` → **同时实例化 with-local + without-local 两 pipeline,运行时按 window>0 选**(镜像 fwd)。
  4. 新 group dispatch + group entry(直接实例化、无 extern-template 文件);harness `-g/-g_*` 系列 + per-group 数组上设备 + CPU `reference_group_hstu_attention_bwd`。no_group kernel 不动(零回归)。
- **M4b(P1-1)section 重点**:
  1. 根因:STAGE2 masked-out 置零被 `if constexpr(FmhaMask::IsMasking)` 编译期 gate,而 NoLocal 在 causal=0 时 IsMasking=false 把掩码整段删掉;但 num_target>0 使 max_uih_len<seqlen,target 区仍需掩 → **静默错梯度、不 throw**。
  2. 修法:STAGE2 置零改 gate 在**运行时** `mask.IsEdgeTile`(去掉 if-constexpr),对齐 fwd 与 reference。一行区域改在共享 SiLU pipeline → batched/jagged/group 三模式同源受益。
  3. **教训(P1-1 式覆盖洞)**:测试矩阵只测对角线(causal=1 配因子 / causal=0 不配),从没交叉 `causal=0×num_target`。→ 新特性要 causal×因子 交叉覆盖。这条很重要,放进「设计动机/教训」。
- 写完按规格 §6 回报。
