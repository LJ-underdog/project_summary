# 派单:写 M5b 讲义(group softmax)

**先读硬规格:`/tmp/hstu-bwd-design/doc-series-spec.md`(全文)。本卡给 M5b 专属输入。**

- **里程碑**:M5b = softmax 落到 group 路(M4 group + M5 softmax 合流)。核心卖点 = **复用不重写**。
- **commit(行号锚定)**:`f7db567d`。`cd /root/workspace/ck_hstu && git show --stat f7db567d`。
- **旧 HTML(参考叙事,行号/数值必重核)**:`/root/workspace/hstu-b1052-report/hstu-bwd-M5b-group-softmax-20260608.html`。
- **事实来源**:`/tmp/hstu-bwd-design/M5b-done.md`(+ review 文档若在)+ `candidates.jsonl` 里 `"id":"M5b-group-softmax"` 那条。
- **输出**:`/root/workspace/hstu-b1052-report/hstu-bwd-M5b-group-softmax-20260625.html`
- **M5b 讲解重点**:
  1. **复用不重写**:M5 `with_softmax` pipeline 逐字复用(mode-agnostic,吃 mask/alpha/LSE/D window);PRE `dot_do_o` 用 `is_jagged=true` 复用(group packed `[head,ΣL]` 同 jagged);POST `convert_dq` 直接用。
  2. **新写** `HstuAttentionBwdDQDKDVGroupSoftmaxKernel` = M4 group kernel 的 per-group i_group 超参 + jagged offset + 运行时 with/without-local 双 pipeline 分支,**融合** M5 LSE/D dram window + softmax pipeline 调用(去 scale_p)。
  3. group dispatch `RunSoftmax`(PRE→memset→MAIN→POST,镜像 M5 no_group);group harness 跑 GPU group fwd(is_training=true)产 O+LSE(`[head,ΣL]` seq-连续),转置 LSE 给 reference,alloc d_dev。
  4. group params 加 `d_ptr`+`nhead_stride_lsed`;CMake 把 group fwd entry 链入 bwd target。
  5. **三个禁改文件 byte-identical**(M5 softmax pipeline / SiLU pipeline / no_group dispatch)= 零回归实测——强调「合流但不碰已 promoted 代码」。
- 写完按规格 §6 回报。
