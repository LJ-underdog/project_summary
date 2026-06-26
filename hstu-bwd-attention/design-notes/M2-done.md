# M2 HSTU mask(causal + 5 因子)— 完成报告 (pane-1 / coder)

状态:**✅ 通过**。SiLU + batched + bf16 + hd64 在 **causal + 5 因子(window/contextual/min_full/num_target)及组合** 下对拍全 PASS(attn_scale=1.0)。离线超集校验 ALL GREEN(M2 硬性前置)。测试套件整体 exit 0。日期 2026-06-05。

## 1. mask 成员实现(方案 A,纯加不改 fwd)— `hstu_block_masking.hpp`
给全部 4 个 HSTU mask struct(Cross/Self × With/NoLocal)各加两个成员:
- **`GetTileRangeAlongY(i_x, number<YTile>, number<XTile>) → (0, seqlen_q)`**:KV-block→attend Q 行范围。**首版取连续保守超集 = 全 Q 扫描**(= DESIGN §8.2-R3 的 fallback B);正确性由 STAGE2 逐像素 `IsTokenPairInsideMask` 置零保证,Y-range 收紧列为后续 perf。`CK_TILE_HOST_DEVICE`(供离线校验 host 调用)。
- **`IsEdgeTile(i_tile_top,i_tile_left, number<TileHeight>, number<TileWidth>) := !IsFullTileInsideMask(top,left, number<TileWidth>, number<TileHeight>)`**(注意 IsFullTileInsideMask 的 (top,left,Width,Height) 参数序)。

## 2. 离线校验器(M2 硬性前置,先写先跑)— `test/validate_tile_range_y.cpp`
枚举 self-attention × {causal,non-causal} × {with,no local} × seqlen{64,128,130,200,256,384} × contextual{0,6} × num_target{0,8,32} × window{0,5,64} × min_full{0,6,64} × 每个 KV tile,断言
`[y_start,y_end) ⊇ { sq : ∃ sk∈tile, IsTokenPairInsideMask(sq,sk) }`。
结果:**checks=185932,failures=0 → ALL GREEN**(`runs/validate_tile_range_y.log`)。hipcc 编 host 程序直接调 mask 成员。

## 3. dispatch / STAGE2 / kernel 改动
- **STAGE2 置零**(`hstu_attention_no_softmax_bwd_pipeline.hpp`):算完 p,g 后,`if constexpr(FmhaMask::IsMasking)` 内,对 `mask.IsEdgeTile(...)` 为真的 tile 用 `set_tile_if(p,0,pred)`、`set_tile_if(g,0,pred)`,`pred = !mask.IsTokenPairInsideMask(row,col)`。SiLU 必清(silu(0)·scale_p=0 但 dsilu(0)=0.5≠0;禁 -inf)。g=0 → STAGE5 ds=dp·g 自动为 0 → dV/dK/dQ 不受 masked 对污染(等价 reference dS=0)。
- **kernel mask 构造**(`hstu_attention_bwd_kernel.hpp`):kargs 加 `num_targets_ptr/contextual_seqlen/max_attn_len/min_full_attn_seqlen`;按 `if constexpr(FmhaMask::kUseLocal)` 调 `make_hstu_self_attention_block_mask_{with,without}_local<FmhaMask>`,`num_target=num_targets_ptr[i_batch]`,**`is_tile_in_first_split=true`(保守:关掉 IsFullTileInsideMask 快路 → 每个 edge tile 都逐像素;tile 级 first-split 优化留 perf)**,并复刻 reference 的 min_full 钳制 `eff_min_full=(seqlen_q-num_target>min_full)?min_full:(seqlen_q-num_target)`。include `hstu_block_masking.hpp`。
- **dispatch**(`hstu_attention_batched_backward_dispatch.hpp`):去掉 causal 的 throw;`RunSilu` 模板化于 Mask;`Run` 内 `BOOL_SWITCH(window_size>0, kUseLocal)` 选 `HstuBlockMasking<false,kUseCausal,kUseLocal>::Type`,MakeKargs 传 mask 标量。causal=0&window=0 → NoLocal<false>(IsMasking=false)= M1 no-mask 路径,统一到 HSTU mask(不再用 GenericAttentionMask)。
- **harness**:加 `supplement_array_by_last_element(num_targets, num_batch)`(修复下述 bug)。CLI 5 因子已同时喂 GPU dispatch 与 CPU reference。

## 4. 逐因子对拍结果(attn_scale=1.0,bf16 阈值,全 PASS)
`runs/run-bwd-M2-sweep.log`(16/16 OK):
| 档 | 结果 |
|---|---|
| causal only(b2/b4×nhead8/b3 seq128/256/200)| ✅ |
| + window(local_len 16/32, seq128/300)| ✅ |
| + contextual(context_len 8/16, seq128/192)| ✅ |
| + min_full(minfull 16/32)| ✅ |
| + num_target(targets=16;per-batch 8,24)| ✅ |
| 全组合(5 因子;b4×nhead4 seq256;seq200 非整除)| ✅ |
| no-mask 回归(causal=0)| ✅ |
误差均 bf16 舍入级(多数 dQ/dK 逐位 0,dV ≤4e-3),max\|ref\| 量级 ~2–6。

## 5. 修复的 bug
**harness 缺 `num_targets` supplement→num_batch**:`-targets=16` 在 b=2 下只上传 1 个 int,kernel 读 `num_targets_ptr[1]` 越界 → batch1 mask 全错 → num_target 档 FAIL(err≈ref 量级)。加 supplement 后立即 PASS。这也是 reference 端越界(`num_targets[1]` UB)的同源问题。

## 6. 测试套件
`test/run_bwd_tests.py`:`reject-causal` 升级为 8 个 M2 pass case(causal/window/contextual/min_full/num_target/per-batch/combo/combo-seq200)。`python3 test/run_bwd_tests.py` → **TOTAL 20 / PASS 19 / FAIL 0 / SKIP 1,exit 0**(`runs/test-20260605-063647.log`)。剩余 reject(softmax M5 / jagged M3 / group M4 / fp16+hdim128 M7)仍正确拒绝。

## 7. candidates / 遗留
- candidates.jsonl 加 `M2-mask`(pass)。
- 遗留(给后续):
  - **perf**:GetTileRangeAlongY 现为全 Q 扫描(保守);收紧为真实 Y-range + 启用 is_tile_in_first_split 的 tile 级跳过 = perf 项(M8)。当前每 edge tile 逐像素 IsTokenPairInsideMask,正确但偏慢。
  - M3 jagged → M4 group(mask 已支持 per-group 结构,需接 group dispatch)→ M5 softmax → M6 deterministic → M7 fp16/hdim。
  - cross-attention mask 成员已加(GetTileRangeAlongY/IsEdgeTile),但 bwd kernel 目前只构造 self mask;cross 路径待对应里程碑接。
- 无未解决阻塞点。
