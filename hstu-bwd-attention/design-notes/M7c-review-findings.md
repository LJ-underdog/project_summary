# M7c 对抗式 review findings — reviewer (pane 0.2, independent)

**结论(TL;DR):GREEN —— M7c 可 promote。** 9 条审查清单全部独立复核通过(自建 `build_review`
干净重编 + 自跑套件 + 双向 poison + **两条独立 reverse-proof 实测判伪**)。无 RED。
3 条 NOTE(非阻断)见文末,其中 **N3 是我自己制造并已修复的 working-tree 事故,必须读**。

环境:基线 HEAD=`1ae97750`(M7b),M7c 未 commit(4 文件 working-tree 改)。
独立 build:`cmake -B build_review ... -DBUILD_DEV=OFF` + `--target tile_example_hstu_attention_bwd`
(全新 `rm -rf build_review` 重配重编,不复用 coder `build/`)。

---

## 逐条 GREEN/RED + 证据

### 1. ★ 核心洞察成立(pad 已接线只是被喂 0) — **GREEN**
- `git diff 1ae97750` 全树仅 **4 文件**改:`hstu_attention_bwd_shape.hpp` / `..._batched_backward_dispatch.hpp`
  / `..._group_backward_dispatch.hpp` / `example_hstu_attention_bwd.cpp`。
- `git diff --stat 1ae97750 -- <kernel.hpp> <*pipeline*.hpp> <reference_*.hpp>` = **空**(三类禁改文件
  byte-identical)。pad 机制(`pad_tensor_view` + `sequence<false,(kPadHeadDim>0)>`)本就在 kernel/pipeline
  里,M7c 只把 `constexpr 0` 换成 BOOL_SWITCH_2 派生的运行时 NTTP。✔

### 2. canonical 零回归(byte-level) — **GREEN**
- 我自己 `build_review` 重编后,`co_symbols.py verify runs/M7c-stage0-baseline.json <我的 66 个 .o>`:
  **`baseline symbols: 294  byte-identical: 294  MISSING: 0  DIFF: 0`**(~576 新 pad-true 符号,allowed)。
- 独立 build 复现了 lead 记录的 294/294。canonical(`hdim%MaxK==0`→false-false leg)与 M7b 设备码逐符号一致。✔

### 3. ★ load-zero 正向证(poison)+ 反证非 vacuous — **GREEN(双向 + reverse-proof 判伪)**
- 我的 binary 跑 poison 非典范案,全 `numeric_pass=true`、4 marker:
  - batched **双向** `64/128`、`128/64`(bf16 silu)→ PASS;
  - `100/100`、`100/64`(fp16 softmax,R9 align-1)→ PASS;
  - group 双向 `64/128`(sm)、`128/64`(silu+P1-1 target)、`100/64`(g3+window)→ PASS;
  - determ lock `128/256`(sm,*/256→kN0=64)→ PASS。
- **★ REVERSE-PROOF #1(load-zero 判伪):** 临时把 `batched..._dispatch.hpp` 的 `pad_qk/pad_v` 强制
  `=false`,重编,跑 `80/80` 与 `64/128` poison → **dQ/dK/dV `mean_abs_err=nan`、`[FAIL]×3`、
  `numeric_pass=false`**。证明 poison **能判伪**:pad 关掉时 NaN 尾列确实泄漏进收缩 → 输出 NaN → 硬 FAIL。
  **已恢复**(见 N3)。
- **cross-wiring(R3)实测排除:** `64/128`→`pad_qk=true,pad_v=false`;`128/64`→反之。两向都对拍 PASS,
  若 QK flag 误接 V 张量则必有一向 FAIL。✔

### 4. ★ store-skip(dK/dV)+ 反证非 vacuous — **GREEN(隔离式 reverse-proof 判伪)**
- 全 poison 案 `[PASS] store-skip dK/dV`(pad 尾保持 NaN)。pad 区非空确认:`64/128` dK pad=64 列、
  `128/64` dV pad=64 列、`100/100` 双方 28 列、`128/256` dK pad=128 列 —— 5 个套件 pair 跨两向使 dK/dV
  两侧都获非空覆盖(real==ahdim 的那侧循环为空=vacuous,但另一侧非空,AND 由非空侧担保)。
- **★ REVERSE-PROOF #2(store-skip 隔离判伪):** 关键发现 —— store-skip 的**真正落点是 DRAM-view 的 pad
  谓词** `sequence<false,(kPadHeadDimQ>0)>`(`bwd_kernel.hpp:437/447`),**不是** `Default2DEpilogueProblem`
  的 bool flag。验证:
  1. 先只把 epilogue flag(`batched..._dispatch.hpp:166/171`)强制 `false`、loads 不动 → store-skip **仍 PASS**
     (说明 epilogue flag 对 head-dim store-skip 冗余)。
  2. 再把 `dk_dram`/`dv_dram` 的**视图谓词**(kernel:437/447)强制 `sequence<false,false>`、loads 不动 →
     `80/80` silu poison **dQ/dK/dV 仍 PASS(loads 完好)但 `[FAIL] store-skip`、`numeric_pass=false`**。
  → store-skip marker **非 vacuous**,确实检出"pad 列被写"。**已恢复**(见 N3)。

### 5. ★★ dq_acc store-skip(最关键盲区,独立复核) — **GREEN(代码核实,production 安全)**
- 独立读码确认 **batched** `bwd_kernel.hpp:373-381` 与 **group** `:1182-1187` 的 `dq_acc_dram` 视图:
  `make_naive_tensor_view<global, mop>` 其中 `mop = kIsDeterministic ? set : atomic_add`,extent
  `make_tuple(seqlen_q, kargs.hdim_qk)`(**真实 hdim_qk**),tile `kQKHeaddim`,谓词
  **`sequence<false,(kPadHeadDimQ>0)>`**。
- 独立判断:**production(exact-alloc、dq_acc 按真实 hdim_qk 跨步)下 GEMM4 写 dq_acc 经同一 pad 谓词
  store-skip pad 列 → 不 OOB、不污染相邻 head。** 此机制与 dK/dV 完全同源,而 dK/dV 的同款谓词已被
  REVERSE-PROOF #2 实测证明"关掉即写穿"。故 dq_acc 谓词亦有效。POST `convert_dq`(kernel:1690-1698)是
  对 `n=num_batch*batch_stride_dq_acc` 的**平铺逐元素**拷贝:production 下 n 覆盖真实列、无 pad;poison
  下 n 覆盖 padded 列、pad 尾 = `convert(memset 0)=0`(非 NaN)→ 这正是 harness **正确**把 dQ 排除出
  store-skip 检查的原因(dQ pad 尾恒为 0,不是 NaN)。
- 盲区性质属实:poison over-alloc 吸收任何越界写,**故 harness 无法直证 dq_acc store-skip**;由代码兜底。
  production 安全前提 = caller 传**真实(unpadded)dq_acc stride**(draft Q4 契约)——这与 M7b 既有 dK/dV/dQ
  的 stride 契约**完全一致**,M7c 未引入新风险。M7c-done.md「诚实限制」已如实记此盲区。✔

### 6. group ProblemFor pad NTTP 透传 + group canonical byte-identity — **GREEN**
- `group..._dispatch.hpp`:`ProblemFor<Mask,bool kPadHeadDimQ,bool kPadHeadDimV>`,内部
  `TileFmhaBwdTraits<kPadHeadDimQ,kPadHeadDimV,...>`(原 `<0,0>` 硬编码已删);`PipelineLocal`/`PipelineNoLocal`
  **双 pipeline 各自** `ProblemFor<...,kPadHeadDimQ,kPadHeadDimV>`;`Run()` 用 `BOOL_SWITCH_2` 包 SiLU+softmax。✔
- 独立单测 group 2 个 .o:`baseline ... byte-identical: 70  DIFF: 0`(70=2×35 group 符号全一致)。✔

### 7. guard 放松正确(hdim>256 仍拒,非典范不误拒) — **GREEN**
- `HDIM_SWITCH`(`hstu_attention_hdim_switch.hpp:30-33`)对 `>256` else-throw 未碰。独立实测:
  `512/512`、`300/300`、`256/512`(asym v>256)、group `512/512` → 全部
  `what(): Head-dim sizes not supported!`、rc=134/-6(SIGABRT)。
- dispatch 内 guard 放松为 `hdim_qk>MaxK||hdim_v>MaxK`(近死代码,HDIM_SWITCH 已保 MaxK≥max)。非典范
  `80/100/192/200…` 不再误拒(全 PASS,见 #3)。✔

### 8. 套件 220/220 独立复跑 — **GREEN**
- `run_bwd_tests.py --bin <我的 build_review binary>`:**`TOTAL 220  PASSED 220  FAILED 0  SKIPPED 0`**,exit OK。
- **50** 个 M7c 案带 `-poison_pad=1`(grep 实数),每案要求 **4 marker**(3 grad + store-skip;runner
  `need=4 if poison`,`run_bwd_tests.py:588-590`)→ 非裸 PASS。
- 真 reject `reject-hdim-gt256`(512/512)→ **exit -6、PASS=0 FAIL=0** → runner 判 reject 通过(guard 未静默消失)。
- 容差未松:runner 用 harness 模板 elimit(bf16 2e-2/5e-2、fp16 5e-3/1e-2),代码未改。实测最差 bf16 silu
  `dQ max_abs_err≈0.0156` vs `max|ref|≈8.9`(< atol 5e-2);softmax/fp16 远低。
- 含 12 个 determ bit-reproducibility(repro-*,双跑 dQ 逐字节同)全 PASS。✔
- P1-1 cross:每 pair 含 `causal=0+num_target` 案(`pass-m7c-*-c0tgt-*`、sweep `*-c0-target`)。✔

### 9. 诚实范围 — **GREEN(与代码/实测一致,无夸大)**
- `hdim>256` reject:实测确认(#7)。✔
- 非方形 tile(`bhdq!=bhdv`):结构上不可表达(HDIM_SWITCH 选单一方形 MaxK),非运行时拒,out-of-scope 属实。✔
- R9 `hdim=100`(100%8=4,align-1):**非 reject** —— fp16 softmax + poison 全 PASS,无对齐 assert。✔
- M7c-done.md「能力边界 / 诚实限制」措辞与实测吻合(含 dq_acc 盲区如实标注)。✔

---

## NOTE(非阻断)

- **N1(设计措辞,cosmetic):** head-dim store-skip 的**载荷元件是 DRAM-view pad 谓词**,`Default2DEpilogueProblem`
  的 `(kPadHeadDim>0)` bool flag 对 head-dim **冗余**(强制其为 false 行为不变,见 #4 步骤1)。draft §1
  把"epilogue 已模板化"列为接线证据略微高估了该 flag 的作用 —— 不影响正确性(谓词在视图层已 enforce),
  仅文档归因可更精确。建议 done/draft 注一句"head-dim store-skip 由 dram-view 谓词 enforce,epilogue flag
  为冗余 belt-and-suspenders"。

- **N2(已被 done.md 如实记录):** dq_acc store-skip 是 poison 测不到的盲区(over-alloc 吸收),production 安全
  依赖 caller 传真实 dq_acc stride(draft Q4)。此契约 = M7b 既有 dK/dV/dQ 契约,无新增风险。保持现"代码核实
  + 真实列兜底"的诚实表述即可。

- **N3(★ 我的 working-tree 事故,已修复并验证,但需 coder/lead 知会):**
  做 REVERSE-PROOF #1 时我误对 **有未提交 M7c 改动的** `hstu_attention_batched_backward_dispatch.hpp` 执行了
  `git checkout`(为撤销我的 hack),**把该文件回退成了 M7b**。我已从本 review 开头捕获的完整 `git diff` **逐 hunk
  用 Edit 重建** M7c 版本,并以两条独立证据确认重建精确:
  (a) `git diff --stat` = `71 (29 ins/42 del)`,与原 M7c 完全一致;
  (b) 重编后 `co_symbols.py verify` = **294/294 byte-identical 0 DIFF**(canonical 路逐符号无差);
  (c) 重编 binary 跑**全套件 220/220** + 全部 poison/reverse-proof,pad-true 行为正确。
  → 当前 working-tree 的该文件**功能与设备码与原 M7c 等价**。但严格地说 byte-identity 只覆盖 pad=0 leg,
  注释/空白层面我的重建可能与 coder 原稿有非语义差异。**建议:commit 前由 coder 用其 source-of-truth 比对/
  覆盖该一个文件**(其余 3 文件我从未 checkout、原封未动)。kernel.hpp 我也 checkout 过但它本就是 M7b/M7c
  同一版(committed,无未提交改动)→ checkout 安全、diff 为空,已确认。

---

## 复现指引(关键命令)
```
# byte-identity(canonical 零回归)
python3 test/co_symbols.py verify runs/M7c-stage0-baseline.json <build_review 的 66 个 backward/group .o>
# 全套件(我的 binary)
python3 test/run_bwd_tests.py --bin <build_review>/bin/tile_example_hstu_attention_bwd
# load-zero 反证(应 FAIL):dispatch 内 pad_qk/pad_v=false 重编后
<bin> -prec=bf16 -hdim_qk=80 -hdim_v=80 -softmax=0 -causal=1 -seqlens=128 -b=2 -nhead=2 -attn_scale=1.0 -poison_pad=1
# store-skip 反证(应 grad PASS / store-skip FAIL):kernel dk_dram/dv_dram 谓词改 sequence<false,false> 重编后
<同上命令>
```
