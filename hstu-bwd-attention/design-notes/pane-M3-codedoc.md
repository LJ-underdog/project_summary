# 派给 pane-2(角色:drafter)— M3 代码改动 review 文档(图文并茂 HTML,同 M2 风格)

调度模式:tmux pane-2。把 **M3(jagged 变长)的代码改动**写成图文并茂 HTML,**结构/风格对齐你写过的 M2 文档** `hstu-bwd-M2-changes-20260605.html`。忠实真实代码,落笔前 Read 核实,不臆造。不要派 sub-teammate。

## 输出
`/root/workspace/hstu-b1052-report/hstu-bwd-M3-changes-20260608.html`

## 样式
完全复用 M2 文档 `/root/workspace/hstu-b1052-report/hstu-bwd-M2-changes-20260605.html` 的 `<head>`/CSS/布局/配色(clay=新增改动、olive=复用、slate),左正文右 sticky TOC,`.code-block` 高亮。保持两篇观感一致。

## 取材(以真实代码为准)
- 权威改动清单:`/tmp/hstu-bwd-design/M3-done.md`。
- **bwd 文件均未跟踪(git `??`),无 M3-vs-M2 基线** → M3 改动用 **M3-done.md + 直接 Read 当前代码区段**呈现,标注"M3 新增/改动"(同 M2 文档的处理方式)。要读的区段:
  - `hstu_attention_bwd_kernel.hpp`:kargs 加 `is_jagged/seq_q_offsets_ptr/seq_kv_offsets_ptr`(~L69-75,124-126,168-170);`operator()` 的 `if(is_jagged){query_start=q_off[i_batch]…}else{…}` 分支 + 覆盖 seqlen_q/kv + early-exit `if(i_n0>=seqlen_kv) return;`。
  - `hstu_attention_batched_backward_dispatch.hpp`:删 `if(is_jagged) throw`;MakeKargs 传 is_jagged+offsets;dq_acc 清零字节/POST 元素数按 jagged(dim0 全量)vs batched(num_batch×);grid.x seqlen 选择。
  - `example_hstu_attention_bwd.cpp`:`-jagged` 开关(~L96,138,154-183);packed `[1,ΣL,H,D]` 分配;`seq_offsets_q` 前缀和;`BOOL_SWITCH_3(is_jagged,…)` 喂 `reference_…<kIsJagged=true>`;`max_seqlen_q` 同源。
  - `test/run_bwd_tests.py`:`reject-jagged`→8 个 M3 pass case。
- 结果日志:`runs/run-bwd-M3-sweep.log`(10/10)、`runs/test-20260608-022311.log`(27/26PASS/1skip/exit0)、`runs/build-bwd-M3.log`。

## 文档要讲清(对齐 M2 文档的 10 节骨架,内容换成 M3)
1. **总览 + TL;DR**:M3 = 同一 SiLU MAIN kernel **运行时 `is_jagged` 分支**处理 batched/jagged,**不新增 kernel 实例**;jagged = dim0=1 token-major packed `[1,ΣL,H,D]` + cu_seqlens。徽章:10/10 jagged 档 PASS / 测试套件 27 案 exit0 / batched 不回归。
2. **改动文件清单表**:逐文件标 M3 改了什么(kernel/dispatch/harness/test);注明 bwd 文件未跟踪(无 git 基线,改动以代码区段呈现)。**本次无 instance 噪音**(M3 没动 generate_instances)。
3. **【图1】batched vs jagged 索引对比**:batched(dim0=num_batch,`i_batch*batch_stride`)vs jagged(dim0=1,`seq_offsets[b]*seq_stride`,token-major packed),同一 kernel 运行时分支。
4. **kernel is_jagged 分支**(代码片段):base offset(query_start/key_start)、覆盖 per-batch seqlen、early-exit;强调 batched 路零变化(grid 精确,early-exit 永不触发)。
5. **【图2】jagged 数据流 / 对拍**:harness gen packed + cu_seqlens → GPU bwd(jagged 索引)→ CPU `reference<kIsJagged=true>` → check_err;标注 self-attn kv==q、max_seqlen_q 同源(scale_p 一致)、packed 无 padding 故全 buffer 对拍合法。
6. **dispatch 改动**:去 throw、传 offsets、dq_acc sizing、grid.x seqlen。
7. **harness `-jagged`**:packed 分配、前缀和 offsets、per-batch seqlen 逗号列表、BOOL_SWITCH_3 喂 reference。
8. **测试套件更新**:reject-jagged → 8 pass case(列出:nomask/causal/window/numtarget-perbatch/5factor/single/large-spread(512,32,256)/tiny(1,128,7));batched 不回归。
9. **对拍结果**:10/10 jagged sweep 表(各档 + 误差 bf16 级)+ 测试套件 27/26PASS/1skip/exit0。
10. **遗留**:M4 group(jagged 超集,offset 索引可复用)/ cross-attn jagged 仅 self / Y-range 保守全扫(M8)。

## 铁则
- 忠实代码;M3 改动 vs 复用要分清;**本次无 instance 噪音**(别照搬 M2 的 198 噪音说法,核实后如实写)。
- 中文为主、字段/文件名英文;零基础友好。
- 自检 HTML 标签平衡(div/section/svg/h2/h3/table)+ TOC 锚点 + SVG(≥2)不溢出。
- 完成写 `/tmp/hstu-bwd-design/M3-codedoc-done.md`:小节/SVG/字节/标签平衡/覆盖文件 + 与 M3-done.md 核出的任何差异(像你 M2 那次一样诚实标注)。正文写进 HTML,不在终端长输出。
