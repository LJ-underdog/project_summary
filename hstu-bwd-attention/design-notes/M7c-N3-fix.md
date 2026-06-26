# M7c 收尾前修 N3(coder pane 0.1)

M7c 对抗 review 结论 **GREEN 可 promote**(9 条全过 + 两条 reverse-proof 判伪)。但有一个 working-tree 事故要你修(reviewer N3):

reviewer 做 reverse-proof 时误对 **带未提交 M7c 改动的** `example/ck_tile/18_hstu_attention/hstu_attention_batched_backward_dispatch.hpp` 执行了 `git checkout`(退回 M7b),之后从它捕获的 git diff **逐 hunk 重建**了 M7c 版。它已验证重建功能/设备码等价(diff --stat 71/29-42 一致、canonical byte-identity 294/294、套件 220/220、reverse-proof 全过),但**注释/空白层面可能与你原稿有非语义差异**。其余 3 个改动文件(shape.hpp / group dispatch / harness)reviewer 从未 checkout、原封未动。

## 你的任务(只针对这 1 文件)
1. 读当前磁盘上的 `hstu_attention_batched_backward_dispatch.hpp`,**对照你 Stage1 写的权威版**(你上下文里有每处 edit):确认逻辑/结构/注释完整正确,与你意图一致。
2. 若有任何非语义差异(注释丢失/空白/措辞)或语义差异 → 用你的权威版**覆盖修正**;若完全一致 → 确认即可。
3. **重编 bwd target + 复跑套件**确认 `220/220 exit 0`(`python3 /root/workspace/hstu-bwd-impl/test/run_bwd_tests.py`)+ co_symbols verify 294/294,证明覆盖后仍 GREEN。
4. pane 里一句话报:文件已确认/已覆盖 + 220/220 + 294/294。**不 commit**(lead 闭合后统一 commit)。

只动这 1 文件,别碰其它。完成等 lead 做最终亲核 + commit。
