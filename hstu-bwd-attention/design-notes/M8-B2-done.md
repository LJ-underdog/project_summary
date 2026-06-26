# M8 Stage 2 (B2 GetTileRangeAlongY 紧致化 — causal) — DONE, 停报 lead 亲验

HEAD 基线 `4629508f`(+ 未 commit 的 MI Stage1)。**未 commit**(等 lead 闭合)。

## 改了什么(2 文件)
1. `hstu_block_masking.hpp` — 把 **NoLocal** mask 的 `GetTileRangeAlongY` 从保守全扫 `(0,seqlen)` 收紧到 causal 真实 Q-row 范围(GetTileRangeAlongX 的转置):
   - **`HstuSelfAttentionBlockMaskNoLocal`**:`!IsMasking`(非 causal)→ `(0,seqlen)` 原样(编译期同旧码);causal → `y_start = (contextual_seqlen>0 && i_x<max_uih_len) ? 0 : align_down(i_x, YTile)`,`y_end=seqlen`。
   - **`HstuCrossAttentionBlockMaskNoLocal`**:同上 + cross 偏移 `y_start = align_down(max(0, i_x - diff_q_kv_len), YTile)`(contextual 时 0;非 causal 原样)。`y_end=seqlen_q`。
   - **WithLocal(window)mask 未碰** → 仍全扫,留给 B3。
2. `test/validate_tile_range_y.cpp` — 扩 exhaustive superset 校验器:加 **cross sweep(diff_q_kv_len,seqlen_q≠seqlen_k 双向)**、泛化 seqlen_q/seqlen_k、加第二 tile 形状(32×128 + 16×64)。

## 安全论证(为何收紧不漏梯度)
- 收紧 = 只少扫"对该 KV-tile 零贡献"的 Q-tile;**每个被访问 tile 内 per-pixel `IsTokenPairInsideMask` 照旧掩**,所以唯一要求 = 范围是真实所需的 **kM0(YTile)-aligned superset**。
- causal:col c 被 row≥c(self)/ row≥c−diff(cross)attend → KV-tile 最小列 i_x 决定 y_start;`align_down` 保证 ≤ 真实 min;y_end=seqlen 是安全上界。
- contextual 行 attend 全部 uih 列 → tile 触 uih 区(i_x<max_uih_len)时 y_start=0(不漏 contextual)。target 行 attend uih 列,但其行号 ≥max_uih_len≥i_x 已落在 [y_start,seqlen)。**非 causal NoLocal(P1-1 含 num_target)→ IsMasking=false → 原样 (0,seqlen) 全收,绝不排除 target 行**。cross diff_q_kv_len 已并入 y_start。

## ★ 4 gate 全过
### Gate 1 — 离线 superset 校验器 ALL GREEN(最硬的 under-tighten gate,无需 GPU)
`validate_tile_range_y`:**checks=1,290,570  failures=0 → ALL GREEN**(self+cross × causal+非causal × tile 32×128 & 16×64,穷举每 (KV-tile, sq, sk))。穷举证明收紧后仍是 superset。

### Gate 2 — 对拍 253/253 + 边界 stress
- `run_bwd_tests.py`:**TOTAL 253 PASSED 253 FAILED 0 SKIPPED 0 exit OK**(`runs/test-20260615-101834.log`)。
- 定向边界 stress **15/15 numeric_pass=true**:causal×{ctx, num_target, ctx+target, 非整除130/77, minfull(nolocal), jagged, determ}、cross causal {kv>q, kv<q, kv>q+ctx+target, jagged kv>q}、group causal {基本, +num_target}、fp16 causal+target。

### Gate 3 — MAIN 加速 vs MI 基线(per-kernel 归因,benchmark.csv 前后对比)
| config | MI 基线 MAIN ms | B2 MAIN ms | 加速 |
|---|---|---|---|
| canonical softmax hd64 | 0.2632 | 0.2027 | **1.30×** |
| canonical silu hd64 | 0.3327 | 0.2085 | **1.60×** |
| hd256 softmax | 1.0325 | 0.7673 | **1.35×** |
| hd256 silu | 0.9709 | 0.7762 | **1.25×** |
| window256 softmax | 0.2992 | 0.2994 | 1.00×(未变,WithLocal=B3) |
| window256 silu | 0.4193 | 0.4192 | 1.00×(未变,WithLocal=B3) |

实测 **1.25–1.60×**(benchmark.csv candidate=`B2-causal-tighten-*` vs `MI-baseline-*`)。**低于 draft [derived] ~1.9× 预测**——诚实归因:收紧只砍 Q-tile 循环数,每 KV-block 的 K/V load + atomic dq_acc 写流量不随之减半(Amdahl),故 <2×。window 配置如期 0 变化(B2 不碰 WithLocal)。

### Gate 4 — co_symbols(B2 改设备码,不要求 byte-identical;但验证改动 surgical)
- 全 68 个 **backward** obj 全部重编(0 stale)。
- vs MI-stage1/HEAD 基线:**512 符号 DIFF,全部落在 32 个 batched `has_causal` 实例 + 2 个 group entry**(causal NoLocal MAIN 收紧,合法);**32 个 batched `no_causal` 实例 + 2 个 no_group entry(kentry wrapper)byte-identical**;**0 个意外/无关符号变化**(96 个 forward obj 用 GetTileRangeAlongX,本就不受影响)。证据 `runs/B2-cosym-vs-HEAD.txt`。

## 报 lead
**B2 done:① 离线校验器 1,290,570 checks ALL GREEN(self+cross,穷举证 superset);② 套件 253/253 + 边界 stress 15/15;③ MAIN 加速 canonical 1.30×(softmax)/1.60×(silu)、hd256 1.25–1.35×(window 如期不变,留 B3),benchmark.csv 前后对比已落;④ 设备码改动 surgical(仅 causal NoLocal,no_causal+kentry byte-identical)。停,候 lead 亲验放行 Stage 3(B3 window/local)。** 未 commit。
