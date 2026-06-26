# pane-2 DONE — part-mask-modes.md

完成 `/tmp/hstu-bwd-design/part-mask-modes.md`(mask / 模式 / 索引 / 参数)。

## 核心结论速览
1. **架构判定**:bwd 采用 FMHA 的 **KV 外 / Q 内**布局(grid.x 沿 seqlen_k,dk_acc/dv_acc 寄存器累加),与 CPU reference 同构。
2. **头号缺口**:HSTU mask 只有 `GetTileRangeAlongX`(fwd 方向)、`IsFullTileInsideMask`、标量 `IsTokenPairInsideMask`;**缺 `GetTileRangeAlongY` 与 `IsEdgeTile`**(bwd KV 外循环必需)。→ 决策方案 A:给四个 mask struct **新增**这两个成员(纯新增,不改 fwd);给出骨架 + 正确性铁律(返回区间须为真值集超集)+ 离线校验方案 + 方案 B fallback。
3. **masked-out 显式置 0**:SiLU 路径 `dsilu(0)=0.5≠0`,STAGE5 dS 必须按 `IsTokenPairInsideMask` 屏蔽(STAGE1/STAGE5 共用同一布尔 tile);softmax 路径靠 S=-inf 自然为 0。
4. **5 因子语义 bwd 与 fwd 完全对称**(同 mask 对象 + 同构造参数),差异仅在 tile 调度方向。
5. **三模式索引**:batch / jagged(cu_seqlens)/ group(per-group device 指针超参,`i_group=i_batch/num_batch_per_group`);hdim_qk(dQ/dK)vs hdim_v(dV)分离。
6. **参数结构**:新增 `HstuAttentionNoGroupBwdParams` + `HstuAttentionGroupBwdParams`;给出完整字段清单 + 与 FMHA `fmha_bwd_args` 对应/差异表(删 dropout/bias/dbias,mask kargs 换 HSTU 5 因子,新增 scale_p 通路)。
7. **scale**:`alpha`(QK,STAGE1 + dQ/dK 末缩放)与 `scale_p`(SiLU 输出,STAGE2/4;group 为 per-group)分清。

## 未决(需 pane-1 协同)
- §5.5:`is_tile_in_first_split` reference 恒传 true,`!is_tile_in_first_split` 分支是否在 fwd GPU 别处按 tile 重建 —— 需与 pane-1 对齐。
- `GetTileRangeAlongY` 精确边界需离线校验后逐分支收紧。

文末已列出对 pane-1(mask 谓词签名 / Y-range / scale 接线 / 收缩维)与 pane-3(新增 mask 成员 / kargs 字段名 / params struct / instances 维度 / grid)的接口假设。
