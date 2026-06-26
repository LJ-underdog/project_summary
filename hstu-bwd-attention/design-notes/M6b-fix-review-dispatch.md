# M6b/M5b harness 修复 — 独立复核派单 (pane-2 / reviewer)

> 背景:M6b 验收时 lead 发现一个 FAIL(group determ+softmax 异构 dQ 0.0626>0.05)。判别实验定位为 **harness bug**(非库逻辑):harness `run_group_hstu_bwd` 的 `group_max_seqlens_q` 用组下标索引 per-batch `num_targets` 数组,组内多 batch 异 seqlen 时低估 max_seqlen_q → PRE kernel(grid 按 max_seqlen_q)漏算最长 batch 尾 token 的 D(d_dev 未 memset)→ 垃圾 D → 仅 target 行 dQ 错。pane-1 已修 harness 公式 + 加 assert + PRE 注释。
> **你独立复核这个修复是否真正、彻底、无副作用。别信自述。** lead 已亲跑原 FAIL 配置确认转 PASS(dQ 7.4e-9)。

## 改动(harness/测试,非库)
- `example_hstu_attention_bwd.cpp`:`group_max_seqlens_q[i_grp]` 改为组内 `max_b(seq_lengths_q[b] + num_targets[b]) + contextual`;删 group_max_uih_seqlens_q;加 `HSTU_CHECK(max_max_seqlen_q ≥ 每 batch packed seqlen)`。
- `hstu_attention_bwd_kernel.hpp`:PRE 前置条件注释(comment-only)。
- `test/run_bwd_tests.py`:加 3 个 `pass-gtrig-{sm-atomic, sm-determ, silu-atomic}`。
- **不应改**:库 kernel/dispatch/pipeline、reference、fwd、promoted 逻辑。确认。

## 任务 A:独立机器验证
1. 干净重建 + 独立复跑 `python3 test/run_bwd_tests.py`:确认 **TOTAL 91 / PASS 91 / FAIL 0 / SKIP 0 exit 0**;M0–M6 零回归。
2. 自跑原 FAIL 配置确认 PASS:`-b=4 -nhead=4 -g=2 -seqlens=128,200,96,160 -g_local_lens=16,16 -targets=8,24,8,16 -softmax=1 -causal=1 -attn_scale=1.0`(全 dQ/dK/dV PASS)。
3. determ==atomic byte-identical + determ 两次 byte-identical(trigger 配置)亲验。

## 任务 B:对抗复核(harness-bug-fix 专属)
1. **修法正确性**:新 `group_max_seqlens_q` 公式 = 组内逐 batch `max(seqlen_q+num_target)+contextual` 对所有 group 配置都 ≥ 该组每个 batch 的真实 packed seqlen?推几个 corner(组内 batch 数不等、num_target 含 0、contextual 非 0)。
2. **回归测试有效性(关键!)**:新加的 `pass-gtrig-*` 案**是否真能复现原 bug**?——即:把 harness 临时改回旧公式(或构造旧 max_seqlen_q),这些案是否会 FAIL?**一个不能触发原 bug 的回归测试=没堵洞。** 至少论证/实测该 config 的组内最长 batch ≠ 组下标 batch 且差值落在 target 区。
3. **assert 真守卫**:`HSTU_CHECK(max_max_seqlen_q≥…)` 在喂错时是否真 abort(把 silent-wrong 变响亮失败)?故意喂小 `-g_max_seqlens` 看是否触发(应 abort,不是静默错)。
4. **无副作用**:SiLU group attn_scale=0 的 scale_p fallback(也用 group_max_seqlen_q)是否仍对?删 group_max_uih_seqlens_q 没破别处?
5. **零改库**:`git diff` 确认 reference/kernel 逻辑/dispatch/fwd/promoted 未动(PRE 仅注释)。
6. **上游备注**:done.md 是否如实记了"上游 example_hstu_attention_fwd.cpp 疑似同款 bug、建议上游报"?

## 产出
写 `/tmp/hstu-bwd-design/M6b-fix-review-findings.md`:任务 A 实测、任务 B 逐条(尤其 B2 回归测试有效性的论证/实测)、总评(M5b+M6b 可签 promote / 需修)。发现真问题立刻报 lead。
