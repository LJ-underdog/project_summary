# Phase 0 补充检查结果

执行日期：2026-04-25
执行者：teammate-P0A

## 0.4b: 3771835ac 完整 diff

**Commit**: `3771835ac4536041accb2e35f6c268cdb6480818`
**Title**: `revert: remove unnecessary +2 row padding in fused_moe.py (gfx950)`
**Author**: root <root@smci350-rck-g03-f12-31> (2026-04-23)

文件触及范围（仅 1 个文件，3 处）：
- `aiter/fused_moe.py` L336-371（fused_moe_ 主路径）：删除 `moe_out_padded = torch.zeros((M+2, model_dim), ...)`，直接传 `moe_buf`，去掉末尾 `result[:M]` 切片
- `aiter/fused_moe.py` L1259-1264（fused_moe_2stages stage1 buffer）：`torch.zeros((token_num+2, topk, inter_dim), ...)` → `torch.empty((token_num, topk, inter_dim), ...)`
- `aiter/fused_moe.py` L1346-1353（stage2 a2 view）：`a2.view(token_num+2, topk, inter_dim)` → `a2.view(token_num, topk, inter_dim)`

**是否触及 ATOM moe.py inter_dim padding**：**NO**
**是否触及 aiter buffer padding**：**YES**（被 revert 删除）

Commit message 验证依据：
- canary 测试（/tmp/test_moe_canary.py、test_moe_canary_stage2.py）显示 a2[T*K+K] 与 moe_out[M] 均 pristine
- tp=2 Step-3.5-Flash 推理 4 prompts 通过（1.97s/req，无 NaN/crash）
- 保留：V1→V3 强制（block_m=128）、shuffle_weight() 预处理
- Note：PR #2551 +2 padding 仅 split-K + per_1x128 quant 路径需要

## 0.4c: ATOM moe.py L489-518 现状

文件路径：`/home/hanchang/ATOM/atom/model_ops/moe.py`

实际代码（L489-518，节选关键部分）：
```python
489        # gfx950 CK a16w16 stage2 requires inter_dim % 64 == 0.
490        # For tp=4 (inter=320) and tp=8 (inter=160), pad inter_dim up to the
491        # next multiple of 64. Zero padding is safe because fused_moe clips
492        # routed-weight contributions and zero-padded rows contribute nothing.
493        # Verified 2026-04-24: cos_sim >= 0.9999 for inter=160->192 and
494        # inter=320->384 vs torch reference.
495        w13 = layer.w13_weight.data  # [E, 2*inter, hidden]
496        w2 = layer.w2_weight.data    # [E, hidden, inter]
497        inter_dim = w2.shape[2]
...
502        align = 64 if inter_dim <= 192 else 128
503        inter_pad = (inter_dim + align - 1) // align * align
504        if inter_pad != inter_dim:
505            E, _, hidden = w13.shape
506            w13_new = torch.zeros(E, 2 * inter_pad, hidden, ...)
510            w13_new[:, :inter_dim, :] = w13[:, :inter_dim, :]       # gate
511            w13_new[:, inter_pad : inter_pad + inter_dim, :] = w13[:, inter_dim:, :]  # up
513            w2_new = torch.zeros(E, hidden, inter_pad, ...)
516            w2_new[:, :, :inter_dim] = w2
517            layer.w13_weight = atom_parameter(w13_new)
518            layer.w2_weight = atom_parameter(w2_new)
```

**结论**：inter_dim padding **存在**，未被 3771835ac 影响。
- tp=4: inter=320 → 384（128 align）
- tp=8: inter=160 → 192（64 align）
- 注释验证日期 2026-04-24，cos_sim ≥ 0.9999

## 0.4d: 决策结论

3771835ac 仅触及 `aiter/fused_moe.py` 的 stage buffer 行 padding，**未触及** ATOM moe.py 的 inter_dim 列 padding。两者属于不同维度的修复：
- aiter +2 row padding：解决 sentinel `(token_id=T, kslot=K)` scatter 越界（已确认 BF16 no-quant 路径不需要）
- ATOM inter_dim padding：解决 CK a16w16 stage2 inter_dim%64==0 对齐要求（gfx950 必须）

**V01 Exp 5 应为：P1**（保持原级别，aiter 侧 revert 不影响 ATOM padding 验证）
**V04 padding 验证基础：稳固**（ATOM L489-518 inter_dim padding 未被任何 revert 影响）

## 0.5b/0.5c: 模型加载

**BF16** (`stepfun-ai/Step-3.5-Flash`): **PASS**
```
Loading BF16 config...
OK: step3p5, hidden_size=4096
Tokenizer OK, vocab_size=128000
```

**FP8** (`stepfun-ai/Step-3.5-Flash-FP8`): **PASS**
```
Loading FP8 config...
OK: step3p5
Tokenizer OK
```

无下载阻塞，config + tokenizer 均能正常本地加载（HF cache hit）。

## 0.10: Worktree 可用性

**junlin12_repos/aiter worktree**: 状态 clean，主 worktree 在 `a2883ab37 [feat/step3p5-moe-swiglustep]`
```
/home/hanchang/junlin12_repos/aiter  a2883ab37 [feat/step3p5-moe-swiglustep]
```

**junlin12_repos/atom worktree**: 状态 clean，主 worktree 在 `ccb64621 [feat/step3p5-flash-support]`
```
/home/hanchang/junlin12_repos/atom  ccb64621 [feat/step3p5-flash-support]
```

两仓库均无未提交修改，worktree 工具可用，可作为后续专题分支并行开发的基础。
