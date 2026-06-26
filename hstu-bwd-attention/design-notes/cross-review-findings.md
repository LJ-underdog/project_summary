# cross-attention 对抗式 review findings — reviewer(pane 0.2)

> 独立复核 HSTU bwd **cross-attention (seqlen_q != seqlen_kv)**。默认怀疑、不信自述。
> 基线 HEAD = `17515fcc`(M7c)。cross 未 commit(5 源文件 working-tree 改)。`-attn_scale=1.0`,容差未松。
> **独立性**:reviewer 自建 3 个 binary —— ① `build_review`(working-tree cross 版,gfx950 BUILD_DEV=OFF,exit 0)② `build_m7c`(干净 M7c worktree `17515fcc`,自产零回归基线)③ `build_r1`(cross 版 + 故意把 cross mask builder 篡改回 self 几何,R1 反证)。所有对拍/套件/符号校验跑在 reviewer 自建的 `build_review`/`build_r1` 上,不复用 coder 产物(除把 coder 基线当交叉校验副本)。

## 结论:**可以 promote。** 8 条清单全 GREEN,3 条核心 silent-wrong 红旗(R1/R4/R2-R3)实测/实证锁死。1 条非阻塞 note(coder 的 stageA 基线符号数被低估,见清单2)。

---

## 逐条裁决

### 1. scope —— **GREEN**
- `git diff 17515fcc --stat`:**仅 5 源文件**(params +6 / batched dispatch +24 / group dispatch +21 / kernel +121 / harness +156),262 插 66 删。
- `reference_hstu_attention_bwd.hpp` + 两 pipeline(`with_softmax` / `no_softmax`):`git diff` **空**(byte-identical)。reference 本就 cross-ready,确认一行未改。

### 2. self 零回归(byte-level)—— **GREEN(且比自述更严)**
- reviewer 从干净 M7c worktree(`build_m7c`)**自产**设备符号基线(`/tmp/review-m7c-baseline.json`)。
- `co_symbols.py verify <我的基线> <build_review 66 obj>` → **870 baseline symbols / 870 byte-identical / 0 MISSING / 0 DIFF**(日志 `runs/review-cosym-verify.log`)。`if constexpr` 守卫零泄漏;~768 个新 `mask<true>` cross 符号 allowed。
- self 套件子集:`run_bwd_tests.py --bin build_review` 的 **220 self 案全 PASS**(253 总 − 31 xattn MATRIX − 2 xattn repro = 220,见清单7)。
- 交叉校验:coder 的 `cross-stageA-baseline.json` 486 个符号的 hash **逐一匹配**我干净 M7c build(486/486 match,0 mismatch)→ coder 基线是真 M7c,数值可信。
- ⚠ **非阻塞 note**:coder 的 stageA 基线只记了 **486** 设备符号,真实数为 **870**;漏的 384 = 64 个 batched instance × 6 个 `ck_tile::kentry<…>` kernel-entry launch wrapper(group entry 131/131 完整)。coder 的 486/486 verify 因此**没覆盖那 384 个 kentry wrapper 符号**。reviewer 的 870/870 verify 已补齐这一格 → 零回归结论不受影响,反而更强。建议:把 co_symbols dump 的对象集补上 kentry(或在 done-doc 把"486 设备符号"更正为"486(其中 batched 漏 kentry wrapper);完整 870")。

### 3. ★ R1 reverse-proof(mask 钉死 self = 头号 silent-wrong)—— **GREEN,判 load-bearing(非 vacuous)**
- 篡改:在 `build_r1` 把 kernel 4 个 kernel × 2(with/without_local)= 8 处 cross builder 的 `seqlen_kv` 槽喂成 `seqlen_q`(`diff_q_kv_len=0` → self 几何复活),DRAM view 的 16 处 `make_tuple(seqlen_kv,…)` 不动。改有未提交改动的源前用独立 worktree + `cp`(未 `git checkout`,守 M7c N3 教训);主工作树全程零污染(实测 `grep R1-SABOTAGE` 主树 = 0)。
- 同一 cross 案 `-jagged=1 -seqlens=128 -seqlens_kv=256 -softmax=0 -causal=1`:
  - **build_review(正确)**:`numeric_pass=true`,dQ/dK/dV err = 1.95e-3 / 1.22e-4 / 3.91e-3(|ref|=6.5/5.0/5.25)。
  - **build_r1(篡改)**:`numeric_pass=false`,err = **4.70 / 5.03 / 5.25**(dK/dV 量级全错,[FAIL]×3)。
- → cross mask 切换确实 load-bearing;若退回 self 几何,cross 案灾难性失败。R1 锁死。

### 4. ★ R4(determ grid/num_splits 用 max_seqlen_kv)—— **GREEN**
- 代码:batched dispatch `grid_seqlen_kv = is_jagged ? max_seqlen_kv : seqlen_kv`(原 `max_seqlen_q`);group dispatch `num_splits`/`GridSize` 用 `max_seqlen_kv`;harness determ workspace `grid_seqlen_kv_h = is_jagged ? max_seqlen_kv : phy_seqlen_kv` + group `num_splits=ceil(max_max_seqlen_kv/kN0)`。两个 bwd params 各 +1 host 字段 `max_seqlen_kv`(device MakeKargs 不读 → 设备码不变,清单2 实证)。
- 实测 kv>q multi-KV-block(kv=512 > q=128,跨 4 个 KV 块):
  - 对拍 PASS:`j-determ-qlt-multiblk-sm/silu`、`g2-determ-qlt-multiblk`、`b-qlt-determ-multiblk`(sweep 全 PASS)。
  - **两次 byte-identical**:套件 repro 段 `repro-xattn-det-qlt-multiblk` + `repro-xattn-gdet-qlt-multiblk` 均 **byte-identical**(`filecmp shallow=False` 真字节比对,dq_dev.dat ×2)。
- 若 grid 仍按 max_seqlen_q,尾 KV 块 dK/dV 会静默归零 → 应 FAIL;实测全 PASS + 可复现 → R4 修对。

### 5. R2/R3(ctor 参序 / with-local num_target 末位重排)—— **GREEN(静态逐字对齐)**
- kernel cross 调用**全走 wrapper**(`make_hstu_cross_attention_block_mask_with/without_local`),无一直调 ctor。
- with_local 参序 kernel vs reference **逐字一致**:`(true, seqlen_q, seqlen_kv, contextual, num_target, max_attn_len/window_size, eff_min_full/min_full)`;wrapper 内部把 `num_target` 重排到 ctor 末位(`hstu_block_masking.hpp:883-889`),kernel/reference 共用同一 wrapper → 重排一致。without_local 同:`(seqlen_q, seqlen_kv, contextual, num_target)`。
- `seqlen_kv` 确喂进 `seqlen_k` 槽(no_group `kargs.seqlen_kv`、group 局部 `seqlen_kv = kv_offsets[i+1]-kv_offsets[i]`,均在作用域内)。`if constexpr` 分叉,false 腿 self builder 逐字不动(清单2 byte-identity 实证)。

### 6. 双向 + 全模式抽样对拍 —— **GREEN(32/32)**
- `sweep_cross.py`(repoint build_review)**32/32 PASS**(日志 `runs/run-cross-sweep.log`):
  - 双向 q<kv(128/256)& q>kv(256/128);no_group jagged / group / batched-uniform;SiLU & softmax;causal{0,1};P1-1(Q-side target / contextual≤min / local / minfull / combo);非整除(130/200、200/130);determ multi-block;fp16(2 例)。
- 误差全在容差内(多为 1e-3~1e-2,bf16 限 2e-2/5e-2、fp16 限 5e-3/1e-2);`|ref|` 量级 5~7(softmax 案 0.05~2),`-attn_scale=1.0` → 非"ref 太小巧合 PASS"。
- 容差**未松**:harness `get_bwd_elimit` 在 diff 中**零改**(bf16 2e-2/5e-2、fp16 5e-3/1e-2 = M7c 原值)。

### 7. 套件 253/253 独立复跑 —— **GREEN**
- `run_bwd_tests.py --bin build_review` → **TOTAL 253 / PASSED 253 / FAILED 0 / SKIPPED 0 / exit 0 / RESULT OK**(日志 `runs/review-suite.log`)。
- 分解:220 self(不动)+ 31 xattn MATRIX + 2 xattn repro = 33 cross。cross 案是真 pass 断言(`numeric_pass=true` + exit 0 + 无 [FAIL]/nan)。
- bit-repro 段 14/14 byte-identical(含 2 个 xattn determ,见清单4);M-series reject 案(hdim>256、asymmetric guard 等)仍满足 reject 预期。

### 8. 诚实范围 —— **GREEN,无夸大**
- **target_in_kv == false**:mask 硬假设 `max_k_uih_len = seqlen_k; // assuming target_in_kv == false`(`hstu_block_masking.hpp:53,:566`);harness cross KV 物理长 = `seq_lengths_kv[i] + contextual`(**不加 targets**,`:281,:914`)。targets 只在 Q 侧 —— 与代码/测试一致。
- **独立 dO layout 未做(R7)**:harness dO stride 取自 `do_host.get_strides()`(与 O 同 layout),PRE 用 O stride 读 dO;独立 dO layout 留后续 —— done-doc §4 如实标注。
- **contextual ≤ min(seqlen_q, seqlen_kv)**:无硬 guard,测试守此(sweep context_len=8,min=128)—— 如实标为测试约定。

---

## 复现指针
- binaries:`/root/workspace/ck_hstu/build_review/`(cross)、`/root/workspace/ck_hstu_m7c_review/build_m7c/`(干净 M7c)、`/root/workspace/ck_hstu_r1_review/build_r1/`(R1 篡改)。
- 基线/日志(`/root/workspace/hstu-bwd-impl/runs/`):`review-cosym-verify.log`、`review-suite.log`、`run-cross-sweep.log`;`/tmp/review-m7c-baseline.json`(reviewer 870-符号基线)。
- R1 反证命令见清单3。

## 给 lead 的话
零回归(byte-identical,且补齐了 coder 漏记的 384 个 kentry wrapper)、R1/R4 silent-wrong 实测锁死、R2/R3 逐字对齐、双向全模式 32/32 + 套件 253/253 + R4 两次 byte-identical —— 全部独立复核通过。**建议 promote。** 唯一 follow-up(非阻塞):把 co_symbols 基线对象集补上 batched 的 kentry wrapper,并把 done-doc "486 设备符号" 更正为完整 870。
