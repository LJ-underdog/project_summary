# M4 代码 review findings (pane-2 / drafter)

审视对象:M4 group 模式代码,对照 DESIGN 语义 + `reference_hstu_attention_bwd.hpp` oracle + M1–M3 既有约定。日期 2026-06-08。
方法:逐文件 Read + 与 reference 逐公式对照 + 用已编译 binary 复跑佐证可疑点。
**不擅自改 kernel**;P0/P1 交 lead 处置。

---

## P0(正确性错误 / 必改)
**无。** group 路径在其验证包络内(causal=1 × per-group mask 因子 × 多 group)与 reference 逐公式一致,8/8 sweep + 套件 34 案均 PASS。

---

## P1(风险 / 可疑 — 建议 lead 处置)

### P1-1. `causal=0 + num_target>0 + window=0` 时 GPU 静默漏掩码,与 reference 背离(ref 量级误差)
- **文件:行**
  - `hstu_block_masking.hpp:711`(及 :542 Cross 同款)— `HstuSelfAttentionBlockMaskNoLocal::IsMasking = kUseCausal;`
  - `hstu_block_masking.hpp:793-800` — 非 causal 分支 `IsTokenPairInsideMask` 返回 `(row_id != col_id) || (row==col)`,在 num_target 造成的 clamp 区(`row_id==col_id==max_id`)会**对非对角对返回 false(即应掩码)**。
  - 触发点:`hstu_attention_no_softmax_bwd_pipeline.hpp` 的 STAGE2 `if constexpr(FmhaMask::IsMasking)`。NoLocal 且 causal=0 → `IsMasking=false` → **STAGE2 逐像素置零被编译掉** → GPU 不掩码;而 `reference_hstu_attention_bwd.hpp:671` 无条件 `if(mask.IsTokenPairInsideMask(sq,sk))` → reference 掩码。两者背离。
- **问题**:`num_target>0` 时 `max_uih_len = seqlen - num_target < seqlen`,target 区 `[max_uih_len, seqlen)` 的 token 之间(非对角)按 HSTU 语义应互不可见。reference 照此掩码;GPU 因 `IsMasking=kUseCausal=false` 完全跳过掩码 → 算成全注意力 → **梯度静默算错(无 throw)**。
- **证据(本机复跑 `build/bin/tile_example_hstu_attention_bwd`)**:
  - A) no_group batched `-causal=0 -seqlens=128 -targets=8 -local_len=0` → **FAIL**:dQ max_abs_err=**1.160**(max|ref|=5.22),dK 1.173,dV 1.078;`out[16385]` 起 batch1 全错。
  - E) group `-causal=0 -g=2 -seqlens=128,200,96,160 -targets=8,24,0,16`(window=0)→ **FAIL**:dQ max_abs_err=**2.180**(max|ref|=6.72)。
  - 对照 PASS:C) `causal=1 -targets=8` → max_abs_err=0;D) `causal=0 -local_len=16 -targets=8`(走 WithLocal,`IsMasking=true`)→ max_abs_err=0;B) `causal=0 -context_len=8`(无 num_target)→ PASS(contextual 单独不触发,因 max_uih_len=seqlen)。
- **定性**:
  - **非 M4 回归** —— 根因是 M2 期 NoLocal struct 的 `IsMasking=kUseCausal` 设计假设「非 causal 的 NoLocal ⟹ 无需掩码」。该假设被 `num_target>0` 证伪(reference 证明此时仍需掩码)。M3/M4 复用同一 pipeline + mask,故同样暴露;group 因 `-causal` 为全局开关 + per-batch targets,反而更易构造到。
  - **当前测试包络外**:M1 把 causal=0 定义为「no-mask 路径」,M2/M3/M4 所有 mask 因子档均 causal=1。故此组合(causal=0 且 num_target>0)在三个里程碑里**从未被对拍覆盖**。
  - **危险点**:harness / dispatch **不拒绝**该组合,直接产出静默错误梯度,而非 throw。
- **建议(交 lead 决定,勿擅改)**:三选一 ——
  1. **守门**:dispatch 或 harness 对 `!causal && (num_target>0)`(更稳妥:`!causal && window==0 && num_target>0`)显式 throw「unsupported」,与现有 hdim/softmax throw 一致;或
  2. **修语义**:让 NoLocal 的 `IsMasking` 在 `num_target>0 || contextual>0` 时也为真(但 `IsMasking` 是编译期常量,num_target 是运行时值 → 需改成运行时门控,代价较大);或
  3. **文档化为不支持** + 加一档 reject 测试锁定行为。
- **优先级理由**:产出静默错误结果(P0 级后果),但仅限**当前未声明支持、未被任何里程碑验证**的 causal=0+target 组合 → 记 P1。若项目计划支持「非 causal HSTU + targets」,应升 P0。

---

## P2(可选 / 观察)

### P2-1. 测试包络缺 `causal=0 + 因子` 的负向锁定
group/jagged/batched 的 sweep 均「causal=0 仅配 no-mask」「因子仅配 causal=1」。建议无论 P1-1 怎么处置,都补一档断言(pass 或 reject)钉死 causal=0×factor 的预期行为,防回归漂移。

### P2-2. 双 pipeline 代码体积(已知 perf 取舍,非 bug)
`hstu_attention_group_backward_dispatch.hpp:104-117` 同时实例化 with-local/without-local 两条 pipeline 进二进制(运行时只跑一条)。这是 per-group window 无法编译期定的**既定取舍**(DESIGN §4.7),M8 perf 项,**不算缺陷**,仅提示 review 者知情。

---

## 逐项核验通过(无 P0/P1)的检查清单
对照 reference oracle / M1–M3 约定,以下均一致:
- **i_group 索引**:kernel `i_batch / num_batch_per_group`(`hstu_attention_bwd_kernel.hpp:657`)= reference `:579`,且 `readfirstlane` 提 scalar,无边界越界(harness 校验 `num_batch % num_group == 0`,`example_…bwd.cpp:577`)。
- **scale_p fallback**:kernel `group_attn_scale!=0 ? : 1/group_max_seqlen_q`(`:668-670`)= reference `:587` 逐字一致;GPU 与 ref 共用同一 `group_max_seqlens_q` 向量(harness `:616-624`)→ fallback 档 PASS 佐证。
- **min_full 钳制**:kernel `eff_min_full=(seqlen_q-num_target>min_full)?min_full:(seqlen_q-num_target)`(`:779-781`)= reference self-with-local `:624-639` 一致。
- **alpha 全局**:kernel 用单标量 `kargs.alpha`(`:787,:796`)喂两条 pipeline,未误当 per-group;= reference 单 `alpha`(D6)。
- **num_target per-batch**:kernel `num_targets_ptr[i_batch]`(`:674`)= reference `:583`(非 i_group)。
- **双 pipeline 选择覆盖**:`window>0` 选 WithLocal(`IsMasking=true` 恒掩码)、否则 NoLocal;= reference `BOOL_SWITCH_2(window_size>0,...)`。causal=0+window>0 经复跑(case D)确认 WithLocal 正确掩码。
- **GetSmemSize**:`max(PipelineLocal, PipelineNoLocal, KGradEpi, VGradEpi)`(`:628-633`),两 pipeline 同 shape,取 max 安全。
- **no_group 零回归**:`HstuAttentionBwdDQDKDVKernel` 为独立 struct,group 为新增 struct;套件内全部 M1/M2/M3(batched/jagged/mask)case 仍 PASS(34/33/1skip)→ 行为级零回归确认。
- **grid early-exit 对所有 group 成立**:grid.x 按 `max_seqlen_q`(全组最大)开,每 block `if(i_n0>=seqlen_kv) return`(`:652`);g4 singleton / large-spread 档 PASS 佐证短组越界 tile 正确退出。
- **dq_acc workspace sizing**:group 用 `total_dq_acc_elems = ΣL*H*hdim_qk`(packed 全量)memset + POST(`group_backward_dispatch.hpp:157-174`),与 packed buffer 精确匹配。
- **host supplement 长度无 M2 式越界**:5 个 per-group 数组均 supplement 到 num_group(`example_…bwd.cpp:598-602`),`group_max_seqlens_q` 直接以 num_group 构造(`:616`),num_targets/seqlens supplement 到 num_batch;device 分配用 `.size()` → 设备指针长度恰够 `[i_group]`/`[i_batch]` 索引。
- **M3 jagged offset 真复用(非漂移复制)**:group kernel 的 base offset / per-batch seqlen / early-exit 公式(`:644-653`)与 M3 jagged 同式(`query_start=offsets[i_batch]`、`seqlen=offsets[b+1]-offsets[b]`)。
- **packed offset 溢出**:offsets→`long_index_t query_start` 后再 `* stride`,长整型运算,常规规模无溢出。

---

## 复跑命令(可复现)
```
BIN=/root/workspace/ck_hstu/build/bin/tile_example_hstu_attention_bwd
# P1-1 触发(FAIL):
$BIN -prec=bf16 -hdim_qk=64 -hdim_v=64 -softmax=0 -attn_scale=1.0 -v=1 -causal=0 -b=2 -nhead=2 -seqlens=128 -targets=8         # A) batched FAIL
$BIN -prec=bf16 -hdim_qk=64 -hdim_v=64 -softmax=0 -attn_scale=1.0 -v=1 -causal=0 -b=4 -nhead=2 -g=2 -seqlens=128,200,96,160 -targets=8,24,0,16  # E) group FAIL
# 对照(PASS):
$BIN ... -causal=1 ... -targets=8                  # C) PASS
$BIN ... -causal=0 ... -local_len=16 -targets=8    # D) PASS(WithLocal IsMasking=true)
$BIN ... -causal=0 ... -context_len=8              # B) PASS(无 num_target 不触发)
```
