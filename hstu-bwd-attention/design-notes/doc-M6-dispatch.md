# 派单:写 M6 讲义(deterministic dQ)

**先读硬规格:`/tmp/hstu-bwd-design/doc-series-spec.md`(全文)。本卡给 M6 专属输入。**

- **里程碑**:M6 = bit-reproducible dQ,范围 no_group(batched+jagged)× SiLU+softmax。
- **commit(行号锚定)**:`48673726`。`cd /root/workspace/ck_hstu && git show --stat 48673726`。
- **旧 HTML(参考叙事,行号/数值必重核)**:`/root/workspace/hstu-b1052-report/hstu-bwd-M6-deterministic-20260609.html`。
- **事实来源**:`/tmp/hstu-bwd-design/M6-done.md`(+ review 若在)+ `candidates.jsonl` 里 `"id":"M6-deterministic"` 那条(77/77 exit0、byte-identical repro、atomic-vs-determ 逐位 diff=0)。
- **输出**:`/root/workspace/hstu-b1052-report/hstu-bwd-M6-deterministic-20260625.html`
- **M6 讲解重点**:
  1. 机制:每 KV-block(split_idx=i_tile_n)用 `memory_operation_enum::set`(非 atomic)plain-store 到自己 split 副本(base += i_tile_n*split_stride_dq_acc)→ POST 固定升序 reduce + convert → **构造上 bit-reproducible**。对比 atomic 路非确定的浮点加序。
  2. dq_acc 窗口编译期分叉(set+split / atomic_add)via constexpr mop + split_stride_dq_acc karg;新 POST `hstu_bwd_reduce_convert_dq_kernel`(固定序求和)。
  3. dispatch 抽 `launch_main_and_post` helper(num_splits/memset/MAIN/POST,kIsDeterministic 真模板轴穿进 Problem);entry BOOL_SWITCH_3 加 kIsDeterministic;generate_instances determ 轴 → 8 no_group instance(4 atomic+4 determ);harness determ workspace = single×num_splits。
  4. 两 pipeline + group dispatch 逻辑**未碰**(group determ 是 M6b)。
- **核心张力(适合 note-block)**:浮点加法不满足结合律 → atomic_add 的归约顺序不定 → 非确定;determ 用「每 split 独立写 + 固定序 reduce」换 bit 可复现,代价是 num_splits 倍 workspace。
- 写完按规格 §6 回报。
