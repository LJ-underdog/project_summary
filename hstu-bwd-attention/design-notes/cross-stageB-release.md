# cross Stage A → Stage B 放行(coder pane 0.1)

**lead 亲核通过 Stage A**:scope 仅 3 文件(batched/group dispatch + kernel mask,params/harness/pipeline/reference 未碰);独立 co_symbols verify **486/486 byte-identical 0 DIFF**(self 符号逐位不变);self 套件 **220/220 exit 0**。零回归证毕。编译预算 OK(scratch=0、max VGPR 426<512;group entry 14min TU 是瓶颈、per-hdim 拆 TU 留 M8)。

## 放行 Stage B — harness 解钉 + max_seqlen_kv + dispatch grid(draft §3C + 闸门 must-fix #1/#2)
按 draft §7 Stage B(以顶部"★闸门裁决"为准):
1. **新增 `max_seqlen_kv` 字段**进两个 bwd params 结构(NoGroup + Group,现都只有 max_seqlen_q)。
2. **dispatch grid/num_splits 接 max_seqlen_kv**(cross 路):batched `hstu_attention_batched_backward_dispatch.hpp:66-70` jagged 分支 `grid_seqlen_kv` 改 `param.max_seqlen_kv`(+`:79` GridSize、`:70` num_splits);group `hstu_attention_group_backward_dispatch.hpp:90/99` 用 `param.max_seqlen_kv`。**self 路 max_seqlen_kv==max_seqlen_q → byte-identical 不变。**
3. **harness 解钉**:CLI `-seqlens_kv`(给出且≠seqlens 即 cross;空=self 向后兼容;group per-group `-g_max_seqlens_kv` 对称);解 `is_cross_attention` 钉死、解别名 max/phy_seqlen_kv、独立 `seq_offsets_kv_dev`、按 phy_seqlen_kv 分配 K/V/dK/dV、**determ grid/workspace `:365/:367` 用 max_seqlen_kv**、reference 调用第一参翻运行时 + 喂独立 kv offsets/max_seqlen_kv。

## Stage B 验证(完再停报 lead)
- **self 零回归不破**:`-seqlens_kv` 不给(self)时,co_symbols 仍 486/486 byte-identical(max_seqlen_kv==max_seqlen_q)+ self 套件 220/220。这是 Stage B 的硬要求(加字段/改 grid 不能破 self)。
- build 0 error。pane 报:"Stage B done,self 仍 486/486 + 220/220,cross harness 就绪(单点 smoke 可选)",停。
- **不 commit**;Stage C(cross 对拍双向 + 矩阵翻转)放行后做。

## 纪律
- self 路必须保持 byte-identical(max_seqlen_kv 别名 max_seqlen_q);容差禁松;不 commit。
