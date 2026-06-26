# M5b 图文 HTML 报告 + 顺带文档级 review 派单 (pane-3)

> 你上一单写了 M5 的 HTML 报告(`hstu-bwd-M5-softmax-20260608.html`)。这单补 **M5b group softmax**:
> ① 写图文 HTML 讲义;② 写的过程中做一遍文档级 review(第三方视角,coder=pane-1、code-reviewer=pane-2 已过,你查漏 + 把"讲不圆处"当可疑点)。
> M5b 已 promoted、对拍全绿(git `dc8c6b21`)。

## 输入材料(全读)
- 规格:`/tmp/hstu-bwd-design/M5b-dispatch.md`
- coder 自述:`/tmp/hstu-bwd-design/M5b-done.md`
- code-reviewer 结论:`/tmp/hstu-bwd-design/M5b-review-findings.md`(含 group LSE/D 四方偏移、i_group、零回归)
- 活状态:`/root/workspace/hstu-bwd-impl/docs/HANDOFF.md` §6(M5b 节)
- 代码(git `dc8c6b21`,相对 M5 `aced5784` 的 5 文件改动):
  - `hstu_attention_bwd_kernel.hpp`(新 `HstuAttentionBwdDQDKDVGroupSoftmaxKernel` + 复用的 PRE)
  - `hstu_attention_group_backward_dispatch.hpp`(`RunSoftmax`)
  - `hstu_attention_bwd_params.hpp`(GroupBwdParams +d_ptr/nhead_stride_lsed)
  - `example_hstu_attention_bwd.cpp`(group 段产 O+LSE)
  - `CMakeLists.txt`(+group_forward_bf16.cpp)
  - **复用未改**:M5 `hstu_attention_with_softmax_bwd_pipeline.hpp`、SiLU pipeline、no_group dispatch
- oracle:`reference_hstu_attention_bwd.hpp`(group + kUseSoftmax 分支)
- 对拍数据:`runs/run-M5b-sweep.log`、`runs/test-20260608-095621.log`

## 报告要求(用 html-report skill)
- skill `/root/.claude/skills/html-report/`;**风格对齐你写的 M5 报告**(`hstu-bwd-M5-softmax-20260608.html`)与 M4 报告。
- 输出:`/root/workspace/hstu-b1052-report/hstu-bwd-M5b-group-softmax-20260608.html`。
- 内容结构建议(图文并茂):
  1. **M5b 是什么**:softmax 从 no_group(M5)扩到 group;**复用不重写**——M5 pipeline + PRE + POST 直接用,只新写 group-softmax kernel + group RunSoftmax + group harness 产 LSE。画 M4×M5 合流图。
  2. **group-softmax kernel**:per-group 超参(i_group=i_batch/num_batch_per_group 取 window/contextual/min_full)、运行时 window>0 选 with/without-local **softmax** 双 pipeline、**softmax 不读 scale_p**(被 LSE 取代)、LSE/D window。
  3. **group packed LSE/D 布局**:[head,ΣL] 连续-seq(fwd seq_stride_lse=1/nhead_stride_lse=ΣL+query_start,无 batch_stride_lse);fwd 写/bwd 读/PRE 写/reference 转置四方溯源图。与 M5 jagged 的异同。
  4. **复用边界表**:哪些直接复用(M5 pipeline/PRE/POST)、哪些新写、为什么能复用(mode-agnostic)。
  5. **验证**:三方闭合、对拍数据表(g{2,3,4} × per-group 异构,**全异构档证 i_group 真索引**)、套件 68/67/0/1、零回归(3 个禁改文件 byte-identical)。
  6. **范围与后续**:cross-attn softmax 未做;group 双 pipeline 实例化体积是 M8 perf 取舍。

## 顺带 review(写时同步,产出单独报 lead)
全新视角核:group LSE/D 四方偏移自己推一遍、i_group 真取 per-group(非恒 group0)、softmax 确无误用 scale_p、window>0 选的是 **softmax** 而非 SiLU pipeline、PRE group jagged 路偏移、CMake 无重复符号、零回归。
**产出** `/tmp/hstu-bwd-design/M5b-doc-review.md`(几行:GREEN 确认 / 或疑点+文件:行号)。发现真问题立刻停下报 lead,别只写进报告。

## 注意
- 只读不改源码;报告代码片段引真实行号。不动 fwd / 不碰 M6/M7。
