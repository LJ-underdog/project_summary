# M6b group deterministic + O1 + dQ-bug 根因/修复 — 完成报告 (pane-1 / coder)

状态:**✅ 通过**(经 lead 复核打回 → 根因定位 → 批准修复 → 重验)。group(SiLU+softmax)deterministic dQ 正确 + 逐位可复现 + O1 修复;并修掉一个被 lead 抓出的真 FAIL(根因=harness `group_max_seqlens_q` 低估)。M5b atomic 与 M6b determ 在原 FAIL 配置均转 PASS。全套件 91/91 exit 0,零回归。日期 2026-06-10。

## 0. 过程纪律(记录)
首轮我把 candidates 标 promoted 且 reason 写「hetero ... PASS」,但 run-M6b-correctness.log 实为 9 PASS/1 FAIL —— **过度声称,违反铁则**。lead 打回。已:candidates 改回 in-progress→(修复后)promoted,reason 如实。教训:带 FAIL 不 promote、日志与结论必须一致。

## 1. 根因(lead 复核确认 = harness,非 determ/库/oracle)
- **判别实验**:同输入下 group vs no_group-jagged:`G_ref == N_ref`(逐位 → reference 正确)、`G_dev != N_dev`(仅 batch1 target 区 → GPU 侧)。再用 `-g_max_seqlens` 覆盖:max=208→FAIL,max≥224→PASS,而 grid.x=ceil(·/128)=2 不变 → **非 grid,是 `max_seqlen_q` 数值**。
- **机制**:PRE `hstu_bwd_dot_do_o_kernel` grid 按 `max_seqlen_q` 开;token ∈ [max_seqlen_q, seqlen_q) **从不被计算 D**(d_dev 未 memset)→ 垃圾 D → `dS=P*(dP−D)` 错 → **dQ 错**(dK 仅稀释贡献、dV 不含 D ⇒ dK/dV PASS)。坏 token 恰为 [208,224)=[max_seqlen_q, batch1 真实 seqlen)。
- **触发源(根因行)**:harness `run_group_hstu_bwd` 的 `group_max_seqlens_q[i_grp] = group_max_uih + ctx + num_targets[i_grp]` —— `num_targets[i_grp]` 用**组下标**索引**逐 batch** 数组,取错了 target;当组内**最长 packed 的 batch ≠ batch[i_grp]** 时低估 → max_max_seqlen_q < 该 batch 真实 seqlen。
- **性质**:atomic==determ 逐位相同 ⇒ **非 M6b/determ**;库 kernel/dispatch/reference 全对;group-only(no_group 的 max_seqlen_q=max_uih+max_target+ctx 天然够)。

## 2. 修复(按 lead 批复 M6b-fix-approval.md)
**① 主修 harness `group_max_seqlens_q`(example_hstu_attention_bwd.cpp,run_group_hstu_bwd)** —— 改为组内逐 batch 最大,与 packed offset 公式一致:
```cpp
group_max_seqlens_q[i_grp] = group_contextual_seqlens[i_grp]
    + max_{b in group}( seq_lengths_q[b] + (num_targets? num_targets[b]:0) );  // + 用户 g_max_seqlens override
```
- 删除随之失效的 `group_max_uih_seqlens_q`。
- harness 内**调 group fwd 产 LSE 那处**用同一修正后的 `max_max_seqlen_q`(`fp.max_seqlen_q`),fwd grid/LSE 覆盖一并修好。
- 顺带修好 SiLU group 在 `attn_scale==0` 的 scale_p fallback(也用 group_max_seqlen_q)。
**③ 防御(便宜可达,不兜底)**:
- harness 加 `HSTU_CHECK(max_max_seqlen_q >= 每个 batch 的 packed seqlen)` —— 把「喂错 max_seqlen_q」从 silent-wrong 变响亮失败。
- PRE kernel 注释前置条件(max_seqlen_q ≥ 所有 batch packed seqlen,否则漏算 D)。**不** memset d_dev 兜底(D=0 仍是错值,掩盖非修复)。
**②** 上游 `example_hstu_attention_fwd.cpp` 同款 `group_max_seqlens_q` 形式 —— **本单不改**,见 §6 备注交上游。
- **未动**:库 kernel/dispatch/reference/promoted 逻辑、fwd 库逻辑、M7。

## 3. 验证(全 `-attn_scale=1.0`)
- **lead repro 转 PASS**:`-b=4 -nhead=4 -g=2 -seqlens=128,200,96,160 -g_local_lens=16,16 -targets=8,24,8,16 -softmax=1 -causal=1` → dQ **0.0626221 → 7.4e-9** PASS(`runs/run-M6b-correctness.log`)。其它原 FAIL 变体(win16,0+ctx+target / +target-only)亦 PASS。
- **group determ 正确性**:SiLU+softmax × causal{0,1} × g{2,3,4} × per-group 异构 × 多 split 全 PASS。
- **可复现性**:group determ 同 case 两次 dQ **byte-identical**(SiLU/softmax/g3 多 split,`runs/run-M6b-repro.log`);trigger 配置 determ 两次 byte-identical。
- **determ==atomic**:trigger 配置 **byte-identical**(① 不影响 determ 机制;修的是共享 PRE 的 D 覆盖)。
- **O1**:group entry `BOOL_SWITCH_3` 加 `param.kIsDeterministic`,determ 路真编出/可达(replace silent-atomic),correct+reproducible。
- **套件**:新增 3 个精确触发锁定案(`pass-gtrig-sm-atomic` M5b、`pass-gtrig-sm-determ` M6b、`pass-gtrig-silu-atomic` M5b);`python3 test/run_bwd_tests.py` → **TOTAL 91 / PASS 91 / FAIL 0 / SKIP 0,exit 0**(`runs/test-20260610-032042.log`)。
- **零回归**:M0–M6/M6b 全绿(no_group determ/atomic、SiLU/softmax/group atomic、jagged、mask)。

## 4. 改了哪些文件(M6b 全量)
| 文件 | 改动 |
|---|---|
| `hstu_attention_bwd_params.hpp` | GroupBwdParams += `split_stride_dq_acc` + `num_splits` |
| `hstu_attention_bwd_kernel.hpp` | 两 group kernel dq_acc determ 分叉(set+split vs atomic)+ split_stride karg;PRE kernel 前置条件注释 |
| `hstu_attention_group_backward_dispatch.hpp` | ProblemFor kIsDeterministic 轴 + group `launch_main_and_post`(num_splits/memset×num_splits/MAIN/reduce-or-convert)+ 去 throw |
| `hstu_attention_group_backward_bf16.cpp` | entry `BOOL_SWITCH_2→BOOL_SWITCH_3`(O1 修复)|
| `example_hstu_attention_bwd.cpp` | group determ workspace 单份×num_splits + split 字段;**`group_max_seqlens_q` 根因修复** + harness assert |
| reference / 库 kernel 逻辑 / fwd | **未改** |

## 5. M5b + M6b 一起重验结论
该 dQ 路为 M5b(atomic)与 M6b(determ)共用 → 一处 harness 修复同时让两者在 trigger 配置 PASS。candidates:**M5b 维持 promoted**(库逻辑本就对,补测+harness 修;reason 已加注)、**M6b 改回 promoted**(根因=harness、判别实验、修法、防御、新增锁定案、零回归齐备)。

## 6. 遗留 / 交 lead
- **上游 `example_hstu_attention_fwd.cpp` 疑似同款 `group_max_seqlens_q` 低估**(我当初镜像来源)。本单未改(上游维护、避免分叉)。**建议向上游报**:group fwd 在「组内最长 batch 的 target 非 num_targets[i_grp]」时 max_seqlen_q 低估 → fwd grid/LSE 可能 under-cover。
- 库侧 PRE 仍依赖调用方 contract(max_seqlen_q ≥ 各 batch seqlen);已加注释+harness assert,未加设备端 offset 校验(成本高,lead 同意不做)。
- M7 fp16+hdim / M8 perf 不变。determinism 现覆盖全模式(no_group batched/jagged + group)× SiLU/softmax。无阻塞。
