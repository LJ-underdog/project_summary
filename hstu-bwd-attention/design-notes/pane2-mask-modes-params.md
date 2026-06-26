# 派给 pane-2(持久角色:architect)— HSTU bwd GPU 方案 · 第 2 部分「mask / 模式 / 索引 / 参数」

调度模式:tmux pane-2。与 pane-1/pane-3 **并行**。先读 `/tmp/hstu-bwd-design/BRIEF.md`。不要派 sub-teammate。

## 你负责:HSTU 特有的 mask、jagged/group/batch 模式、索引、bwd 参数结构设计

产出 `/tmp/hstu-bwd-design/part-mask-modes.md`,涵盖:

1. **5 因子 HSTU mask 在 bwd 的接入**:研究 `hstu_block_masking.hpp` 的 `HstuCrossAttentionBlockMaskWithLocal`(causal/window_size/contextual_seqlen/min_full_attn_seqlen/num_target,含 `is_tile_in_first_split` 逻辑)。设计:
   - bwd 是 KV 外循环、Q 内循环,mask 如何决定 (sk-block, sq-block) 对是否要算 → **tile 级 early-exit**(对照 FMHA bwd 的 `mask.GetTileRangeAlongY` / IsEdgeTile / early return)。
   - **masked-out 显式置 0**:SiLU 路径 silu(0)·scale_p≠自然零,必须在 STAGE5 把被遮位置 dS=0(还有 STAGE2 P 置 0);给出与 mask 谓词协同的具体做法。
   - contextual / min_full_attn / num_target 这些 HSTU 业务因子在 bwd 的语义是否与 fwd 完全对称(参考 reference 的 mask 调用)。
2. **三模式 + 索引**:
   - batch(dim0=num_batch)/ jagged(dim0=1 + cu_seqlens `seq_q_offsets`)/ group(每段独立超参 `group_attn_scales/contextual/window/min_full/max_seqlen`)。
   - bwd 的 grid/索引怎么算每个 (batch,head) 的 seqlen 与基址(对照 reference 的 `tensor(0, seq_off[b]+s, h, k)` 与 FMHA bwd kernel 的 kIsGroupMode 分支)。
   - **hdim_qk ≠ hdim_v** 对 dV(用 hdim_v)/dQ,dK(用 hdim_qk)的影响。
3. **bwd 参数结构设计**:`params.hpp` 现无 bwd 字段。设计 `hstu_attention_bwd_args/kargs`:新增 `do_ptr, dq_ptr, dk_ptr, dv_ptr, lse_ptr, d_ptr, dq_acc_ptr` + 各自 stride(seq/nhead/batch)+ `alpha, scale_p, kUseSoftmax, kIsDeterministic` 等;复用 fwd 已有字段(seqlen/offsets/hdim/mask 参数)。给出字段清单 + 与 FMHA `fmha_bwd_args` 的对应/差异表。
4. **scale 语义**:`alpha`(QK,STAGE1)与 `scale_p`(SiLU 输出,默认 1/max_seqlen_q,STAGE2/5)分别在哪用、bwd 怎么传。
5. 风险/未决:group 模式 per-segment 超参在 GPU 上怎么取(常量 buffer?);target/contextual 在 bwd 的边界 case。

## 边界
只设计 mask/模式/索引/参数;算法阶段(pane-1)、文件与 codegen/测试(pane-3)别碰,但文末列出对二者的接口假设(mask 谓词签名、kargs 字段名)。

## 铁则
- 以 `hstu_block_masking.hpp` / `reference_hstu_attention_bwd.hpp`(group 版在 line ~501 起)/ `hstu_attention_params.hpp` / FMHA `fmha_bwd_kernel.hpp` 源码为准。
- markdown 到 `/tmp/hstu-bwd-design/part-mask-modes.md`;完成写 `/tmp/hstu-bwd-design/mask-modes-done.md`。
