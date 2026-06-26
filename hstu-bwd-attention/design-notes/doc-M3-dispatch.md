# 派单:写 M3 讲义(jagged 变长)

**先读硬规格:`/tmp/hstu-bwd-design/doc-series-spec.md`(全文)。本卡只给 M3 专属输入。**

- **里程碑**:M3 = jagged(变长 packed)模式,SiLU+bf16+hd64。**同一 SiLU MAIN kernel 靠运行时 `is_jagged` 分支同时handle batched + jagged,零新增 instance**——这是核心卖点。
- **commit(行号锚定它)**:`94174bd9`。先 `cd /root/workspace/ck_hstu && git show --stat 94174bd9`。
- **旧 HTML(可参考叙事,行号/数值必重核)**:`/root/workspace/hstu-b1052-report/hstu-bwd-M3-changes-20260608.html`。
- **事实来源**:`/tmp/hstu-bwd-design/M3-done.md` + `candidates.jsonl` 里 `"id":"M3-jagged"` 那条。
- **输出**:`/root/workspace/hstu-b1052-report/hstu-bwd-M3-jagged-20260625.html`
- **M3 讲解重点**:
  1. packed 布局 `[1,ΣL,H,D]`(dim0=1,token-major)+ `cu_seqlens`(前缀和);per-batch base offset = `seq_q/kv_offsets_ptr[i_batch]*seq_stride`,镜像 fwd kernel。
  2. per-batch seqlen 从 offsets 反推;OOB-kv-tile early-exit(grid.x 按 max_seqlen_q 开)。
  3. kernel Kargs/MakeKargs 加 `is_jagged` + `seq_*_offsets_ptr`;dispatch 去 jagged throw、接 offsets、按 mode 定 dq_acc memset/POST + grid。
  4. harness `-jagged`:packed alloc(batches_for_alloc=1)+ cu_seqlens 前缀和 + 喂 offsets 给 GPU fwd/bwd 和 CPU reference`<kIsJagged=true>`;seqlens 接 per-batch 逗号列表。
  5. 零新增 instance(对比 batched 复用同 kernel)——强调「运行时分支 vs 编译期实例」的取舍。
- **易错提示**:packed 偏移寻址 + early-exit 适合配 1 张 SVG(画 batched 矩形 vs jagged packed 锯齿 + offset 指针)。
- 写完按规格 §6 回报。
