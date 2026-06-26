# cross-attention 图文并茂 HTML 讲义 —— 文档(pane 0.2)

你刚做完 cross-attention 对抗 review(3-binary 独立 + R1 reverse-proof + 发现 486→870 kentry 盲区),最熟。写 **cross-attention HTML 讲义**。已四方闭合 promoted、commit `4629508f`。

## 0. skill + 风格 + 纪律
- **必用 skill `html-report`**。风格对齐最新:`/root/workspace/hstu-b1052-report/hstu-bwd-M7c-hdim-pad-20260615.html` + `hstu-bwd-M7b-hdim-20260611.html`。
- 输出:`/root/workspace/hstu-b1052-report/hstu-bwd-cross-attention-20260615.html`。
- **无-emoji 铁则**:别用 ★/✓/✗/方块 dingbat(→/⇒ 可)。

## 1. 素材(只读,数字必一致,不臆造)
- `docs/cross-attn-done.md`、`docs/draft-cross-attn.md`(批准设计 + 顶部闸门裁决)、`/tmp/hstu-bwd-design/cross-review-findings.md`(你写的)、`cross-stage{A,B}-checkpoint.md`。
- candidates 末行、HANDOFF cross 块、`git -C /root/workspace/ck_hstu show 4629508f --stat`。

## 2. 讲义必讲清(图文并茂)
1. **是什么 + 范围**:bwd 处理 seqlen_q≠seqlen_kv(cross-attention),全方向(含 kv>q)× 全模式;**明确** target_in_kv=false、独立 dO layout 未做、非方形 tile = out-of-scope。
2. **★ 核心洞察 + 两大利好**:mask 钉死 self 是唯一真破绽(kv offset/grid/loop/PRE 本就 cross-ready);reference oracle 本就 cross-ready 零改;cross 是运行时 BOOL_SWITCH 零 instance 增长。配 self→cross 数据流图(独立 q/kv 序列 + diff_q_kv_len 对齐)。
3. **机制**:dispatch mask `<kIsCrossAttention>` BOOL_SWITCH + kernel 4 处 if constexpr cross builder(seqlen_kv→seqlen_k);max_seqlen_kv 纯 host 字段(device 码不变=零回归基石)。
4. **★ Option B 决策(闸门)**:critique 抓到 dispatch grid 按 max_seqlen_q 开 + 无 max_seqlen_kv 字段 → kv>q silent-wrong;裁决做全(加字段 + grid 接 max_seqlen_kv)。配 grid-sizing 前后图。
5. **★★ 怎么证没 silent-wrong(亮点)**:① self 零回归 co_symbols **870/870 byte-identical**(reviewer 自产基线,补齐 coder 漏的 384 kentry wrapper);② **R1 reverse-proof**——篡改 cross mask 回 self → cross 案灾难性 FAIL(err 4.70 vs 1.95e-3)= mask switch load-bearing;③ R4 kv>q multi-block PASS + byte-identical。讲透"判伪"逻辑。
6. **诚实限制**:target_in_kv=false、dO layout(R7)、group entry 14min TU(R11/M8)、co_symbols 基线补 kentry follow-up。
7. **数据**:sweep 32/32、套件 253/253、双向、reference 零改。
8. **四方闭合**:coder 3-stage + reviewer 3-binary 独立 + lead 亲核 + R1 reverse-proof。

## 3. 纪律
- 只据素材、范围诚实、无 dingbat/外链/占位符、图 SVG/CSS 自包含。完成 pane 报路径,等 pane-3 文档 review。
