# cross-attention 对抗式 review —— reviewer(pane 0.2,已 /clear)

独立对抗 review HSTU bwd **cross-attention(seqlen_q != seqlen_kv)**,默认怀疑、不信自述。基线 HEAD=`17515fcc`(M7c),cross 未 commit(5 源文件 working-tree 改)。素材:`docs/draft-cross-attn.md`(批准设计,顶部"★闸门裁决")、`docs/cross-attn-done.md`、`/tmp/hstu-bwd-design/cross-stage{A,B}-checkpoint.md`。

## 独立 build:`rm -rf build_review` 重配重编 `tile_example_hstu_attention_bwd`(BUILD_DEV=OFF,gfx950)。注意 group entry 单 TU ~14min(R11),整体 build 久。

## 对抗审查清单(逐条 GREEN/RED + 文件:行号)
1. **scope**:`git diff 17515fcc` 仅 5 源(params/batched+group dispatch/kernel/harness);**reference + pipeline(with/no_softmax)byte-identical**(diff 空)。确认。
2. **self 零回归(byte-level)**:你 build_review 后 `co_symbols.py verify runs/cross-stageA-baseline.json <backward .o>` → **486/486 byte-identical 0 DIFF**?self 套件 220 子集全绿?
3. **★ R1 reverse-proof(mask 钉死 self 是头号 silent-wrong,必判伪)**:临时把 kernel 的 `if constexpr(kIsCrossAttention)` cross 分支改回 self builder(或强制 dispatch BOOL_SWITCH 选 false),重编,跑一个 q≠kv cross 案 → 应 **对拍 FAIL**(self 几何 mask 错的 kv)。证 cross mask 切换是 load-bearing 非 vacuous。验完恢复。**做实验改源前先 `cp` 备份该文件,别 git checkout 带未提交改动的文件(M7c N3 教训)。**
4. **★ R4(determ grid/num_splits 用 max_seqlen_kv)**:独立跑 kv>q determ multi-block(如 seqlens_kv=512 > seqlens=128 -deterministic=1)→ 对拍 PASS + 两次 byte-identical?若 grid 仍按 max_seqlen_q,尾 KV 块 dK/dV 静默归零 → 应 FAIL。
5. **R2/R3(ctor 参序 / with-local num_target 末位重排)**:kernel cross mask ctor 调用与 reference `make_hstu_cross_attention_block_mask_*` 调用点逐字对齐?走 wrapper 非直调 ctor?
6. **双向 + 全模式**:独立跑 q<kv 与 q>kv × {no_group jagged, group, batched} × {SiLU,softmax} × causal{0,1} × P1-1(target Q 侧 / contextual / local / minfull)抽样对拍 PASS,容差未松。
7. **套件 253/253 独立复跑**:你 build_review binary 跑 `run_bwd_tests.py` → 253/253 exit 0?cross 案是真 pass 断言?self 220 不动?
8. **诚实范围**:target_in_kv=false、独立 dO layout 未做(R7)、contextual≤min(q,kv) —— 与代码/实测一致,无夸大。

## 产出
写 `/tmp/hstu-bwd-design/cross-review-findings.md`,逐条 GREEN/RED + 证据;RED 给复现配置。结论:可否 promote。完成 pane 里一句话报。
