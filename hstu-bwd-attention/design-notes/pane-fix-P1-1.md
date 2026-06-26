# 派给 pane-1(角色:coder)— 修 P1-1:causal=0 + num_target>0 静默漏掩码

调度模式:tmux pane-1。修一个已被 pane-2 review 抓到并复跑坐实的真 bug。严守 kernel-design-rocm skill:修完编译 + 对拍 + 跑回归套件,证据进 runs/。不要派 sub-teammate。

## bug(已诊断,见 `/tmp/hstu-bwd-design/M4-review-findings.md` P1-1)
- 现象:`causal=0 + num_target>0 + window=0`(no-causal NoLocal 路)下,GPU bwd **静默不掩码** → 梯度算错(batched dQ err≈1.16,group≈2.18),**无 throw**。
- 根因:`hstu_attention_no_softmax_bwd_pipeline.hpp:417` 的 STAGE2 masked-out 置零被 `if constexpr(FmhaMask::IsMasking)` 包住;而 `HstuSelfAttentionBlockMaskNoLocal::IsMasking = kUseCausal`(`hstu_block_masking.hpp:542/711`)→ causal=0 时编译掉置零。但 `num_target>0` 时 target 区按 HSTU 语义仍需掩码(reference `:671` 无条件 `if(IsTokenPairInsideMask)` 照掩,GPU 没掩)。
- **判定:这是合法配置必须修,不是守门拒绝**——已验证 **fwd 支持** `causal=0+targets`(fwd 对拍 PASS),fwd 用的是**运行时** `if(!mask.IsTokenPairInsideMask(row,col))`(`hstu_attention_no_softmax_fwd_pipeline.hpp:378`),非编译期 gate。bwd 应改成与 fwd 一致。

## 修法(与 fwd 对齐:运行时置零,不靠编译期 IsMasking)
核心:STAGE2 的 masked-out 置零要在**运行时**对"可能被掩码"的情形生效,即使编译期 `IsMasking==false`(NoLocal causal=0 但 num_target/contextual 触发掩码)。建议:
- 把 STAGE2 置零(:417-428)的 `if constexpr(FmhaMask::IsMasking)` 改为**运行时条件**:要么无条件做逐像素 `set_tile_if(p/g, 0, !IsTokenPairInsideMask)`(完全对齐 fwd 的无条件运行时检查;真无掩码时 IsTokenPairInsideMask 恒 true → 不清任何元素,仅多一次扫描,与 fwd 同代价),要么 gate 在一个运行时 bool `needs_mask = (FmhaMask::IsMasking) || num_target>0 || contextual_seqlen>0`(更省:纯 no-mask 跳过扫描)。**你选更干净的;正确性优先,但别让纯 no-mask 路径白白大幅变慢**。
- 注意 `IsEdgeTile`(=`!IsFullTileInsideMask`)在 causal=0 NoLocal 下是否能正确把 num_target 的 tile 判为 edge——若不可靠,直接退化为"逐像素 IsTokenPairInsideMask"(覆盖最稳)。
- early-exit(:141 `if constexpr(IsMasking)`)与 GetTileRangeAlongY 保守全扫已正确,causal=0 不早退即可,不必动(除非顺带优化)。
- 确认 `IsTokenPairInsideMask` 在 NoLocal causal=0 分支对 num_target/contextual 返回正确(pane-2 称 `:793-800` 非 causal 分支在 clamp 区会返 false=掩码——核一下与 reference 一致)。
- group 路同源(`hstu_attention_bwd_kernel.hpp` 的 GroupKernel 走同一 pipeline)→ 一处修,batched/jagged/group 都受益;确认三模式都修好。

## 验收(全过)
- 编译 0 error。
- **新增对拍 PASS**(attn_scale=1.0,bf16):
  - batched `causal=0 -targets=8`(原 FAIL→应 PASS)
  - group `causal=0 -g=2 -targets=8,24,0,16`(原 FAIL→应 PASS)
  - jagged `causal=0 -targets`(per-batch)
  - `causal=0 -context_len>0`(contextual-only,无 target)也确认 PASS
- **无回归**:跑 `python3 test/run_bwd_tests.py` 整体 exit 0(现有 34 案全绿)。
- **测试套件补 case 锁定**(关键,防回归漂移):新增 batched/jagged/group 的 `causal=0 + num_target`(及 contextual-only)pass case,把这个 bug 永久纳入回归。
- candidates.jsonl 加 `M4b-fix-causal0-target`(pass + 修了什么 + 复跑对照)。
- 纯 no-mask 路径(causal=0 无任何因子)仍 PASS 且未显著变慢(简述你选的 gate 方案与代价)。

## 铁则
- 与 fwd 语义对齐,不放宽容差。修最小面,别动无关逻辑。
- 如果发现 fwd 其实也有同类边界没覆盖(顺手核到),记进 done 文件交 lead,别擅自改 fwd。
- 完成写 `/tmp/hstu-bwd-design/fix-P1-1-done.md`:改了哪几行、gate 方案与理由、四个新对拍结果(原 FAIL 现 PASS)、测试套件新 case + 整体 exit、no-mask 性能影响一句话。
- progress 简洁;长 log 进文件。
