# 修复 P1-1 完成报告 — causal=0 + num_target>0 静默漏掩码 (pane-1 / coder)

状态:**✅ 修复完成**。两个原 FAIL 现 PASS,batched/jagged/group 三模式同源受益,测试套件整体 exit 0,纯 no-mask 零回归且不变慢。日期 2026-06-08。

## 改了哪几行
**唯一改动**:`hstu_attention_no_softmax_bwd_pipeline.hpp` STAGE2(原 :413-430)。
- 把 masked-out 置零的外层 **`if constexpr(FmhaMask::IsMasking)`** 去掉,直接以**运行时 `if(mask.IsEdgeTile(...))`** 作为门(内部 `set_tile_if(p/g, 0, !IsTokenPairInsideMask)` 不变)。
- 加注释说明 P1-1 根因与对齐 fwd/reference 的理由。

无其它文件改动(kernel/dispatch/params/harness 未动)——这是共享 SiLU pipeline 的一处修,batched(`HstuAttentionBwdDQDKDVKernel`)/ jagged / group(`HstuAttentionBwdDQDKDVGroupKernel`)都走这条 pipeline,**一处修三模式齐受益**。

## gate 方案与理由
**选「运行时 `IsEdgeTile` 门」**(非「无条件逐像素」,也非「自定义 needs_mask bool」):
- **正确性**:NoLocal 的 `IsFullTileInsideMask`(`hstu_block_masking.hpp`)两个 causal 分支**都已正确处理 num_target**——`i_tile_bottom>=max_uih_len || i_tile_right>=max_uih_len → 非 full-inside → IsEdgeTile=true`(`max_uih_len=seqlen-num_target`)。`IsTokenPairInsideMask` 非 causal 分支(`:799`)在 clamp 区对非对角对返回 false(掩码),与 reference(`:765-800`)逐字一致。故 IsEdgeTile 门一开,target 区即被逐像素正确置零。
- **对齐 fwd/reference**:fwd 用运行时 `if(!IsTokenPairInsideMask)`(非编译期 gate),reference 无条件 `if(IsTokenPairInsideMask)`。本修使 bwd 与两者语义一致。
- **纯 no-mask 不变慢**(关键):`IsEdgeTile` 对「完全在 mask 内」的 tile 返回 false → **不做逐像素扫描**。tile 整除的纯 no-mask(causal=0 无因子,max_uih_len=seqlen)→ 所有 tile full-inside → **零 set_tile_if 扫描**,仅每 Q-tile 多一次廉价整数比较的 IsEdgeTile 调用。仅非整除边界 tile / target tile 才扫描,代价与 fwd 同。比「无条件逐像素」省掉了 no-mask 的全量扫描。

## 四个新对拍结果(attn_scale=1.0,bf16;原 FAIL→现 PASS)
`runs/run-fix-P1-1.log`(5/5 PASS,exit 0):

| 档 | 修前 | 修后 |
|---|---|---|
| A) batched `causal=0 -targets=8` | **FAIL** dQ=1.160 | ✅ dQ max_abs_err=**7.6e-6** dK=6e-8 dV=2e-3 |
| E) group `causal=0 -g=2 -targets=8,24,0,16` | **FAIL** dQ=2.180 | ✅ dQ=**2.0e-3** dK=1.6e-2 dV=1.6e-2 |
| jagged `causal=0 -targets=8,24,16`(per-batch) | (同源 bug) | ✅ dQ=3.9e-3 dK=1.6e-2 dV=1.6e-2 |
| batched `causal=0 -context_len=8`(无 target) | PASS | ✅ 仍 PASS dQ=7.6e-6 |
| 纯 no-mask `causal=0`(无因子,perf 对照)| PASS | ✅ 仍 PASS dQ=1.2e-4 |

误差均 bf16 舍入级(max|ref| ~5–6.7)。

## 测试套件新 case + 整体 exit
`test/run_bwd_tests.py` 加 4 个 M4b pass 锁定 case(把 bug 永久纳入回归):
- `pass-causal0-target-batched`、`pass-causal0-target-jagged`、`pass-causal0-target-group`、`pass-causal0-contextual-only`。

`python3 test/run_bwd_tests.py` → **TOTAL 38 / PASS 37 / FAIL 0 / SKIP 1,exit 0**(`runs/test-20260608-051136.log`)。原 34 案全绿(no_group/jagged/group/mask 零回归),M5 softmax / M7 fp16+hdim128 仍正确拒绝。

## no-mask 性能影响(一句话)
纯 no-mask(tile 整除)修后**零逐像素扫描**(IsEdgeTile 对 full-inside tile 返 false),仅每 Q-tile 多一次廉价整数比较;实测仍 PASS,无可观测变慢。

## 交叉矩阵升级(并入 addendum)
不止补 num_target 一格,把测试套件从「对角线覆盖」(causal=1×因子 / causal=0×no-mask)升级为 **causal=0×因子整列**的系统性交叉,堵同类 IsMasking 耦合洞。先用 binary 逐档探测(`runs/probe-causal0-cross.log`)再入库,确保每个 `pass` 标记反映真实对拍结果。

新增 12 个 `M4b-cross` case(均走 oracle 对拍,**全 PASS,未发现新缺陷**):
- **batched causal=0 ×**:`target`(核心 repro,修前 FAIL)、`context`、`minfull`、`window`、`context+target`、`minfull+target`、`combo(local+context+minfull+target)`。
- **jagged causal=0 ×**:per-batch `target`、`context`。
- **group causal=0 ×**:per-batch `target`(修前 FAIL)、per-group `window 16,0`(混 WithLocal/NoLocal)、per-group `context`+per-batch `target`。

逐档结论:
- 真正靠本修转 PASS:`causal=0 + num_target`(及其组合)——batched/jagged/group 三模式。
- 本就 PASS、加入作回归锁定:`causal=0 + context`(max_uih_len=seqlen 不 clamp)、`causal=0 + minfull`(window=0 时 without_local 忽略 minfull,与 reference 一致)、`causal=0 + window`(WithLocal IsMasking=true 本就掩)。
- **无新缺陷**:12 档误差全为 bf16 舍入级(max_abs ≤ ~1.6e-2,max|ref| ~2.5–7),未发现 contextual/minfull 在 causal=0 下的另一格漏洞。

套件总数:**TOTAL 46 / PASS 45 / FAIL 0 / SKIP 1,exit 0**(`runs/test-20260608-052301.log`)。原 34 案(含 M1/M2/M3/M4)零回归;M5/M7 仍正确拒绝。milestone tag 统一标 `M4b-cross` 便于追溯。

## 顺手核到 / 交 lead
- **fwd 无同类问题**:fwd SiLU pipeline 本就运行时 `if(!IsTokenPairInsideMask)`(`hstu_attention_no_softmax_fwd_pipeline.hpp:378`),causal=0+targets fwd 对拍 PASS。**未改 fwd**。
- 本 bug 的根源是 M2 期 NoLocal `IsMasking=kUseCausal` 的「非 causal ⟹ 无需掩码」假设被 num_target 证伪;现已用运行时门绕开该编译期假设,不必改 mask struct 的 `IsMasking` 定义(其仍正确驱动 early-exit `:141` 等其它编译期决策)。
- P2-1(测试包络缺 causal=0×factor 负向锁定)已由上述 4 个 M4b case 一并覆盖。
