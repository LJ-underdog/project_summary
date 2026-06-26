# cross Stage B → Stage C 放行(coder pane 0.1)

**lead 亲核通过 Stage B**:scope 5 文件(pipeline/reference 未碰);独立 co_symbols **486/486 byte-identical**(self 设备码不变);独立 cross smoke 双向 PASS(q>kv grid-shrink + q<kv determ multi-block kv=512>q=128 R4 路)。cross 两方向都通。

## 放行 Stage C — cross 对拍全矩阵 + 套件翻转(draft §6 + §7 Stage C)
1. **单点 ctor 逐字对齐先验**(R2/R3):取最简 cross(`xattn-jagged-qlt-kv-silu-causal1`),确认 kernel cross mask ctor 参与 reference `make_hstu_cross_attention_block_mask_*` 调用点**逐字对齐**(seqlen_q/seqlen_k 顺序、with-local 包装器 num_target 末位重排)。
2. **§6 全矩阵对拍**(`-attn_scale=1.0`,前缀 `xattn-`):
   - 方向:**q<kv 与 q>kv 双向**(抓 R1/R4/R5)。
   - 模式:no_group jagged(主)+ group(主,per-group kv 长)+ batched(uniform,两方向,纳主套件)。
   - 激活 SiLU/softmax × causal{0,1}(causal=1 验 diff_q_kv_len 对齐)× P1-1 逐配置(num_target 留 Q 侧 / contextual≤min(q,kv) / local window / minfull)× atomic/determ。
   - **非整除 + determ multi-block 按选定 tile 的 kN0**(hd256=64 否则 128,must-fix #3);**≥1 个 determ 用例 kv>q 且跨多 KV 块**(用选定 kN0 算)。
   - fp16 一两例。
3. **套件永久化**:把 cross 用例(当前 REJECT,CLI 不识别 -seqlens_kv)逐个翻 PASS 进 `/root/workspace/hstu-bwd-impl/test/run_bwd_tests.py`,记新 TOTAL;**保留既有 self 220 不动**(向后兼容,-seqlens_kv 不给=self)。

## Stage C 验证(完再停报 lead → reviewer 闭合)
- cross 对拍全 PASS(双向 + 全模式 + P1-1 + determ kv>q multi-block),容差禁松。
- self 套件仍全绿(220 + 新 cross 案);co_symbols self 仍 486/486。
- `docs/cross-attn-done.md`(全阶段证据 Stage A-C + 改面 + co_symbols + 双向对拍数 + 诚实限制:target_in_kv=false、独立 dO layout 未做 R7、group entry 14min TU=M8)。
- `candidates.jsonl` 加行(status 据实 in-progress 待 reviewer+lead 闭合)。
- **不 commit**;诚实(带 FAIL/裸 PASS 不充数,M6b 教训)。
