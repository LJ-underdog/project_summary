# 派给 pane-1(角色:coder)— HSTU bwd 实现 M2(HSTU mask:causal + 5 因子)

调度模式:tmux pane-1。接续 M1(你写的)。严守 kernel-design-rocm skill:每步编译+对拍验证 + **跑回归测试套件**,证据进 `/root/workspace/hstu-bwd-impl/`。不要派 sub-teammate。

## 依据
- 设计:`/tmp/hstu-bwd-design/DESIGN.md` §3.1(方案 A 新增 mask 成员)、§3.2(masked-out 显式置 0)、§5.4(GetTileRangeAlongY 离线超集校验=**M2 硬性前置**)、§6 M2、§8.2-R3。
- M1 现状:`/tmp/hstu-bwd-design/M1-done.md`;dispatch 现 `if constexpr(causal) throw`(L204)。
- mask 源:`hstu_block_masking.hpp` —— 4 个 struct(`HstuCrossAttentionBlockMaskWithLocal`:12 / `HstuSelfAttentionBlockMaskWithLocal`:268 / `HstuCrossAttentionBlockMaskNoLocal`:503 / `HstuSelfAttentionBlockMaskNoLocal`:656),每个有 `GetTileRangeAlongX`/`IsTokenPairInsideMask`/`IsFullTileInsideMask`,**全缺 `GetTileRangeAlongY`/`IsEdgeTile`**。`make_hstu_*_block_mask_*`(801+)。`HstuBlockMasking`(788)选型。
- 5 因子:causal / window(local_len)/ contextual(context_len)/ min_full(minfull_len)/ num_target(targets)。harness CLI 已有:`-causal -local_len -context_len -minfull_len -targets -max_target`(SiLU 路 + no-group)。
- FMHA MAIN 调用点:`GetTileRangeAlongY(k_origin,kM0,kN0)`(无条件,kr_ktr_vr:160-161)、`IsEdgeTile`(:559)、early-exit(:166-173)。

## M2 目标
让 **SiLU + batched + bf16 + hd64** 路在**开启 HSTU mask** 下对拍 PASS,从 causal 起逐因子加。`reject-causal` 升级为 pass。

## 步骤(按序;离线校验先行)
1. **新增 mask 成员(方案 A,纯加不改 fwd 现有成员)**:给 4 个 mask struct 加
   - `GetTileRangeAlongY(index_t i_x, number<XTile>, number<YTile>) → (y_start,y_end)`:`GetTileRangeAlongX` 的转置;**返回必须是 `IsTokenPairInsideMask` 真值集在 Y 方向的连续保守超集**(宁多算 tile 不漏)。5 因子叠加 attend 行集可能非连续 → 首版返回 [最小行, 最大行] 连续区间(中间空 tile 由逐像素清零兜底)。
   - `IsEdgeTile(i_y,i_x,TH,TW) := !IsFullTileInsideMask(...)`(注意 IsFullTileInsideMask 的参数顺序,:237)。
2. **离线校验器(M2 硬性前置,先写先跑)**:`/root/workspace/hstu-bwd-impl/test/validate_tile_range_y.cpp`(或 .py 调一个小 host 程序)——随机/枚举 (seqlen_q,seqlen_kv,5 因子) × 每个 KV tile,断言
   `[y_start,y_end) ⊇ { sq : ∃ sk∈tile, IsTokenPairInsideMask(sq,sk) }`。
   **必须全绿**才信任 GPU mask 路径。结果记 runs/。
3. **dispatch 接 mask**:`hstu_attention_batched_backward_dispatch.hpp` 去掉 causal 的 throw,改用 `HstuBlockMasking<...>` 选型(复刻 fwd dispatch 的 `make_hstu_*_block_mask_*` 构造 + 同样的 `is_tile_in_first_split` / 最后一参三元),把 mask 传进 kernel/pipeline。
4. **STAGE2 置零**:在 `hstu_attention_no_softmax_bwd_pipeline.hpp` 的 STAGE2,对 **edge tile**(`mask.IsEdgeTile`)用 `set_tile_if` 把 `p`、`g` 的 masked-out 元素(`!mask.IsTokenPairInsideMask(row,col)`)清 0(代码已留注释位)。整块 in-mask 跳过。**SiLU 必须清(silu(0)·scale_p≠自然零,dsilu(0)=0.5);禁 -inf。** 同时接 early-exit(num_total_loop≤0 整块跳)。
5. **harness**:把 `-causal/-local_len/-context_len/-minfull_len/-targets` 同时喂给 GPU dispatch 与 CPU `reference_*_bwd`(同参数构造,保证对拍一致)。

## 验收(逐因子,全过才算 M2 通过)
- 离线校验器全绿(前置)。
- 编译 0 error。
- 对拍 PASS(attn_scale=1.0,bf16 阈值),**逐因子递进**:
  1. causal only(local/context/minfull/target=0)
  2. + window(local_len>0)
  3. + contextual(context_len>0)
  4. + min_full(minfull_len>0)
  5. + num_target(targets>0)
  6. 组合
  每档跑几个 seqlen/b/nhead(含非整除)。把通过的档记录。
- **更新测试套件** `test/run_bwd_tests.py`:`reject-causal` → `pass`;新增 window/contextual/min_full/num_target/组合 的 pass case;跑 `python3 test/run_bwd_tests.py` **整体 exit 0**。
- candidates.jsonl 加 `M2-mask`(pass/fail + 哪些因子档通过 + 离线校验结果);benchmark 可选。

## 铁则
- 不改 fwd 行为 / 不放宽容差。masked-out 必须真零(对拍会暴露污染)。
- 离线校验器**没全绿就不要信 GPU 的 PASS**(可能是 Y-range 漏算碰巧没踩到)。
- 卡住超合理尝试,如实写阻塞点 + 已试 + 怀疑方向(尤其 GetTileRangeAlongY 边界、is_tile_in_first_split、置零谓词)。
- 完成写 `/tmp/hstu-bwd-design/M2-done.md`:mask 成员实现、离线校验结果、dispatch/STAGE2 改动、逐因子对拍结果、测试套件更新后整体 exit、遗留。
- progress 简洁;长 log 进文件。
