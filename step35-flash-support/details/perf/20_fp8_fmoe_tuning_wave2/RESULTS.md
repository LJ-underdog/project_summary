# Wave 2 — FP8 fMoE Tuning (OPT-1) — RESULTS

> **wave**: `step35-fp8-fmoe-tuning-wave2`
> **WORK_DIR (artifact 全集)**: `/home/junlin12/step35_fp8_optim_wave2/`
> **结案日期**: 2026-05-12
> **承接**: `details/perf/16_perf_gfx950_verified/RESULTS.md §四-B`（#601 trigger 起点）
> **措辞 prefix 4 级**：【实证】 / 【实证修正】 / 【推测未实证】 / 【通用教训】

---

## TL;DR

> # 🔴 Wave 结论 — FAIL（accept-and-close）

- 【实证】**OPT-1 axis（补 aiter `tuned_fmoe.csv` stepfun-specific entries）在 stepfun-Flash-FP8 + gfx942 + aiter HEAD `315123ace` 路径上不仅无效，还有害**：27 metric 综合 **0/27 ≥5% 改善**，3 个反向退化（TTFT np=4 +10.3% / TPOT np=8 +3.1% / decode np=8 -3.1%），8/12 multi-prompt + tp≥4 cell 因上游 aiter `apply_act_and_mul` shape bug crash 不可测。
- 【实证】**工程动作 4/4 done**：tune sweep 42/42 success（wall 125.5s）、dispatch HIT 6/6 全 cfg dict 完整、ATOM bench 9 Edit hunk patch 落地（208 行 +105/-29）、18 cell V03 矩阵跑出 10/18 + 8/18 crash —— 但全部不能翻成 perf PASS。
- 【通用教训】**dispatch HIT ≠ perf 改善**：tune sweep 跑通 + dispatch HIT 6/6 验证通过都只是工程动作 done，perf 矩阵实测才是 wave PASS 唯一标准；本 wave 三者中前两者 PASS，第三者 FAIL → wave FAIL。
- 【实证】**6 Promote 全部落地**：救命级 #1（q_dtype 必须 fnuz） / OPT-1 axis 证伪 / FP8 byte-drift 物理必然 / dispatch HIT≠perf 通用教训 / aiter shape bug / bench script max_model_len bug —— 详见 §7。

---

## 1. Outcome（FAIL 显式声明）

### 1.1 一句话结论

【实证】**Wave 2 FAIL**：OPT-1（补 fmoe csv stepfun-specific entries）axis 在当前 aiter HEAD `315123ace` 上 0/27 metric 达成 ≥5% 改善阈值（GOAL "至少一档 TTFT 或 TPOT 改善 ≥5%" 严格 FAIL），且 V03 multi-prompt 路径 3 反向退化 + 8 cell 上游 bug 不可测。

来源：`WAVE_CLOSE.md §0` + archival memory `stepfun_fp8_fmoe_wave2.md` TL;DR。

### 1.2 章节级 PASS/FAIL 标注（借鉴 wave 16 disclaimer pattern）

> # 🔴 重要：本 wave 是 FAIL wave，**未来 wave 不要再投资源在 OPT-1 axis**（除非先解决 §6 推测的 ASM kernel list 缺失 + tuner 量纲两条根因）。

| Phase | 内容 | verdict |
|---|---|---|
| Phase 0 | baseline 跑通 + tooling 调研 | ✅ PASS |
| Phase 1 | 并行调查 OPT-1 fmoe tune + OPT-6 perf bench 改造 + cos-sim 路径溯源 + reviewer | ✅ PASS |
| Phase 1.5 | 修 P0-1 dispatch HIT 验证 / P0-2 GOAL 降级 byte-equal / P0-3 untuned csv 加 token={4,8} | ✅ PASS（工程） |
| Phase 2 | T7-A ATOM bench 9 Edit + T7-B aiter tune 42/42 + dispatch HIT 6/6 | ✅ PASS（工程） |
| Phase 3 v1 | 6 cell np=1 perf 矩阵 + byte-equal | 🔴 FAIL（perf 0/9 + byte-equal 0/3 GOAL 双 FAIL） |
| Phase 4 V03 | GOAL 重设删 byte-equal + V03 multi-prompt 18 cell | 🔴 FAIL（27 metric 0 改善 + 3 反向退化 + 8 cell crash） |
| Phase 5 | 接受 FAIL 结案 + rollback csv + Promote 教训 | ✅ 收尾 |

来源：`WAVE_CLOSE.md §1`。

---

## 2. 工程动作完成度（done-but-no-perf）

### 2.1 4 项 done 表

| # | 动作 | 完成度 | artifact path |
|---|---|---|---|
| 1 | aiter tune 42 stepfun shape | 【实证】42/42 success / wall **125.5s** / 0 failed | `WORK_DIR/tuned_fmoe_step35.csv`（43 行） + `WORK_DIR/logs/c01_tune_sweep.log` |
| 2 | dispatch HIT 6 stepfun key tuple | 【实证】**6/6 ✅ HIT** w/ 完整 cfg dict（kernelName1/2 / block_m / ksplit / us / err / tflops / bw / run_1stage / xbf16 / _tag） | `WORK_DIR/c01_dispatch_verify.py` + `WORK_DIR/logs/c01_dispatch_verify.log` |
| 3 | ATOM perf bench 9 Edit hunk | 【实证】9/9 done / patch **208 行 (+105/-29)** / 加 `--ignore-eos` `--num-prompts` `--enable-cudagraph` flag | `WORK_DIR/c01_atom_bench.patch`（**未 rollback** — 留供未来 wave 复用） |
| 4 | 18 cell perf 矩阵（v01 6 + V03 12） | 【实证】10/18 跑通（v01 6/6 + V03 4/12 tp=2 全部）+ 8/18 上游 aiter bug crash（V03 tp=4/8 全部） | `WORK_DIR/v01_perf_matrix.md` + `WORK_DIR/v03_perf_matrix_multiprompt.md` + `WORK_DIR/logs/v0{1,3}_*.log` × 36 |

来源：`WAVE_CLOSE.md §3` + `c01_aiter_summary.md §3-§5` + `v01_perf_matrix.md` + `v03_perf_matrix_multiprompt.md`。

### 2.2 通用教训：HIT ≠ perf

【通用教训】tune sweep 跑通 + dispatch HIT 验证通过 = **工程动作 done**，但**不等于 wave PASS**：
- tune sweep 完成 = 工程动作 done
- dispatch HIT 验证通过 = 调度路径正确（新 entry 真的命中而非 fallback）
- **但 perf 矩阵实测才是 wave PASS 唯一标准**
- 三者中**任一缺失**都不能判 wave PASS
- **反模式**：把"tune sweep 跑通 + dispatch HIT 验证通过"当成 wave PASS 信号，跳过实测 perf 矩阵直接收尾

本 wave 实证：dispatch HIT 6/6 全 cfg dict 实证替换 kernel 已生效，但 27 metric 0/27 ≥5% 改善 → tune 出来的 kernel 与 fallback heuristic（`fused_moe.py:932+`）选出的 kernel 性能相当，甚至更差。详见 §9.4 扩展段。

---

## 3. 数据矩阵（27 metric）

> 表头统一【实证】prefix；cell 内 metric 数据均来自 `WORK_DIR/logs/v0{1,3}_*.log` raw stdout 截录，不重复 prefix。

### 3.1 v01 9 metric 表（np=1，来源 `v01_perf_matrix.md §1+§2+§3`）

| TP | metric | baseline (623 行 csv) | tuning (665 行 csv) | Δ % | ≥5% 改善 |
|---|---|---|---|---|---|
| 2 | TTFT | 1652.1 ms | 1652.6 ms | +0.03% | ❌ |
| 2 | TPOT | 15.6 ms/tok | 15.5 ms/tok | -0.64% | ❌ |
| 2 | decode | 64.2 tok/s | 64.6 tok/s | +0.62% | ❌ |
| 4 | TTFT | 967.7 ms | 973.2 ms | +0.57%（慢） | ❌ |
| 4 | TPOT | 14.5 ms/tok | 14.4 ms/tok | -0.69% | ❌ |
| 4 | decode | 69.1 tok/s | 69.4 tok/s | +0.43% | ❌ |
| 8 | TTFT | 736.2 ms | 745.9 ms | +1.32%（慢） | ❌ |
| 8 | TPOT | 13.7 ms/tok | 13.6 ms/tok | -0.73% | ❌ |
| 8 | decode | 72.8 tok/s | 73.3 tok/s | +0.69% | ❌ |

**v01 9 metric 全部 |Δ| < 1.5%**，远低于 5% 改善阈值，统计上落在测量噪声 floor (±1-2%) 内 → 等同于无效。

【实证】**baseline anchor 9/9 PASS**（实测 baseline 与 TEAM_CONFIG L55-59 anchor 全部 ±5% 内）→ csv rollback + cache 清理生效，6 cell 数据可信。来源：`v01_perf_matrix.md §3` 完整对比表。

### 3.2 v03 18 metric 表（multiprompt 含 8 crash，来源 `v03_perf_matrix_multiprompt.md §1.2+§2.2`）

| cell | TP | np | state | TTFT (ms) | TPOT (ms/tok) | decode (tok/s) | chars | 启发式 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| v03-7  | 2 | 4 | baseline | 5485.8 | 18.2 | 54.7 | 1197 | PASS | ✅ |
| v03-1  | 2 | 4 | tuning | **6053.0** | 18.2 | 54.8 | 1291 | PASS | ✅ |
| v03-8  | 2 | 8 | baseline | 12086.9 | 19.4 | 51.1 | 1236 | PASS | ✅ |
| v03-2  | 2 | 8 | tuning | 12093.4 | **20.0** | **49.5** | 1253 | PASS | ✅ |
| v03-9  | 4 | 4 | baseline | — | — | — | — | — | 🔴 CRASH (a=8 vs b=16) |
| v03-3  | 4 | 4 | tuning | — | — | — | — | — | 🔴 CRASH (a=8 vs b=16) |
| v03-10 | 4 | 8 | baseline | — | — | — | — | — | 🔴 CRASH (a=8 vs b=16) |
| v03-4  | 4 | 8 | tuning | — | — | — | — | — | 🔴 CRASH (a=8 vs b=16) |
| v03-11 | 8 | 4 | baseline | — | — | — | — | — | 🔴 CRASH (a=8 vs b=32) |
| v03-5  | 8 | 4 | tuning | — | — | — | — | — | 🔴 CRASH (a=8 vs b=32) |
| v03-12 | 8 | 8 | baseline | — | — | — | — | — | 🔴 CRASH (a=8 vs b=32) |
| v03-6  | 8 | 8 | tuning | — | — | — | — | — | 🔴 CRASH (a=8 vs b=32) |

V03 Δ 表（仅可比对 tp=2 cell）：

| TP | np | TTFT Δ% | TPOT Δ% | decode Δ% | ≥5% 改善 |
|---|---|---|---|---|---|
| 2 | 4 | **+10.3%** ⚠️ 反向退化 | 0% | +0.2% | ❌ |
| 2 | 8 | +0.05% | **+3.1%** ⚠️ 反向 | **-3.1%** ⚠️ 反向 | ❌ |

【实证】**baseline csv (623 无 step35 entry) + tuning csv (665 含 42 entry) 在 tp=4/8 + np=4/8 全部触发同样 bug** → 与 stepfun shape 无关，是上游通用 bug（详见 §4）。

【实证】**Model trap 防护通过 12/12**：每 cell `Model load done:` grep 确认 stepfun-Flash-FP8 加载，无 ATOM `--model` default Qwen3-0.6B trap（`v03_perf_matrix_multiprompt.md §5`）。

### 3.3 反向退化数据点高亮

【实证】3 个**反向退化**实证数据点（V03 tp=2 path 实测，非推测）：

| # | 工况 | metric | baseline | tuning | Δ % | 性质 |
|---|---|---|---|---|---|---|
| 1 | tp=2 np=4 | TTFT | 5485.8 ms | 6053.0 ms | **+10.3%** | tuning 反向变慢 |
| 2 | tp=2 np=8 | TPOT | 19.4 ms/tok | 20.0 ms/tok | **+3.1%** | tuning 反向变慢 |
| 3 | tp=2 np=8 | decode | 51.1 tok/s | 49.5 tok/s | **-3.1%** | tuning 反向降吞吐 |

【推测未实证】tp=2 np=4 TTFT +10.3% 退化潜在原因（详见 §6 推测根因，本 wave 未做 layer dump 验证）：(a) tuning csv 某 entry 在 num_tokens=4 batch decode 路径上选了不优 tile config → 比 fallback 慢；(b) tuning entry 是 decode-batch-tuned，但 4×10240 token 一次性 prefill 走 prefill path → mismatch；(c) 同一 nohup 内连跑 jit cache 效应（每 cell 已清缓存但 ATOM 内部 state 不能 100% 排除）。

---

## 4. 救命级 #2 — 上游 aiter bug（独立 H2，不藏 appendix）

> 【实证】**未来跑 multi-prompt + tp≥4 + fp8 ck2stages 路径必须先验证此 bug 是否已 fix**，否则必崩。本 wave 实证 baseline csv 也崩 → 与 stepfun shape **无关**，是上游通用 bug。

### 4.1 bug 位置

【实证】`/workspace/aiter/aiter/fused_moe.py:1646` 行
```
apply_act_and_mul(out, valid_out.view(dtypes.fp32), activation)
```

来源：`v03_perf_matrix_multiprompt.md §6.3`。

### 4.2 触发条件

【实证】**num_prompts ≥ 4 + tp ≥ 4 + fp8 ck2stages 路径**必崩（cudagraph capture 阶段）：
- tp=4: `RuntimeError: The size of tensor a (8) must match the size of tensor b (16) at non-singleton dimension 1`
- tp=8: `RuntimeError: The size of tensor a (8) must match the size of tensor b (32) at non-singleton dimension 1`

【实证】a=8 与 num_tokens 对齐（np=8 时 a=8），b=16 (tp=4) / b=32 (tp=8) 与 inter_dim_per_tp 的 pad 后值或 tile dim 相关。具体定位本 wave 未投资源（lead+user 决策不追根因）。

【实证】**baseline csv (623 行无 step35 entry) + tuning csv (665 行含 step35 42 entry) 都崩** → 与 stepfun shape 无关，是上游通用 bug。aiter HEAD `315123ace` 上 reproducible。

### 4.3 crash trace path

【实证】完整 crash trace 在以下 8 个 log（按 cell 拆分）：
- `WORK_DIR/logs/v03_baseline_tp4_np4_full.log` / `v03_tuning_tp4_np4_full.log`
- `WORK_DIR/logs/v03_baseline_tp4_np8_full.log` / `v03_tuning_tp4_np8_full.log`
- `WORK_DIR/logs/v03_baseline_tp8_np4_full.log` / `v03_tuning_tp8_np4_full.log`
- `WORK_DIR/logs/v03_baseline_tp8_np8_full.log` / `v03_tuning_tp8_np8_full.log`

【实证】触发路径：cudagraph capture (`model_runner.py:2190 outputs[:num_tokens] = self.model(...)`) → fp8 path `module_moe_ck2stages_f8_f8_preshuffle_on_b16_*` → `ck_moe_stage1` → `apply_act_and_mul`。

### 4.4 未来跑 multi-prompt 必须先验证此 bug 是否已 fix

【实证】未来 wave 想跑 multi-prompt + tp≥4 + fp8 ck2stages 路径前，**必须**先：
1. 跑 1 cell baseline csv（无 stepfun entry）+ np=4 + tp=4 sanity check
2. 如崩同样 trace `apply_act_and_mul` shape mismatch → bug 未 fix，path blocked，不要继续
3. 如不崩 → bug 已 fix，可继续 V04 重测 12 cell

【实证】**Promote-4 副仓位 (e) 上游 GitHub issue 留 user 决策**（不在本 doc scope；建议含本 wave V03 logs 8 个 crash trace 作 issue artifact）。来源：`WAVE_CLOSE.md §5 Promote-4`。

---

## 5. 救命级 #3 — FP8 byte-drift verification anti-pattern（独立 H2）

> 【实证】**FP8 kernel 替换在 temp=0 greedy 路径上物理上必然导致 byte-drift**（不是 bug，是数值物理特性）；**未来 wave 不要用 byte-equal 作 fp8-vs-fp8 verification**（必然 FAIL）。

### 5.1 物理必然性（非 bug）

【实证】FP8 kernel 切换 → 累加顺序 / tile / splitk 变 → bf16 输出低位 bit 飘 → softmax/argmax 边界 case 翻 → 整段 decode token 序列分歧。

任何 fmoe kernel 切换（即使 input fp8 weight bit-equal、output dtype 不变）都会因累加顺序变化使 bf16 输出最低 bit 不一致；temp=0 greedy decode 在 softmax 边界（top-k 概率非常接近）会因这 1 bit 翻转选不同 token，后续 token 序列雪崩分歧。

来源：archival memory `stepfun_fp8_fmoe_wave2.md` "FP8 byte-equal 物理特性教训"段 + `WORK_DIR/v02_byte_equal_report.md §6.1` + `WAVE_CLOSE.md §4.1` (lead 自反 P0-2=β GOAL 设计错)。

### 5.2 本 wave 6 cell × 1 prompt 实证

【实证】Phase 3 v1 teammate-8 v02 6 cell × 1 prompt 全部 byte-diff，包括 first 80 chars 即飘（非 decode 末段才发散）。byte-equal 6/6 FAIL，启发式 `_check_correctness` 6/6 PASS。

来源：`v02_byte_equal_report.md §6.1`（teammate-8 完整 byte-by-byte diff 报告）。

### 5.3 不要用 byte-equal 作 fp8-vs-fp8 verification

【实证】这是设计前置约束，不是 wave 偶发 bug：
- ❌ **不要做**：fp8-vs-fp8 byte-equal 比对作为 wave PASS criterion（必然 FAIL）
- ❌ **不要做**：把 byte-equal 当 GOAL（lead Phase 1.5 P0-2=β 推荐已被实证 FAIL）
- ✅ **要做**：替代 verification 路径（详见 §5.4）

### 5.4 替代方案

替代 verification 设计建议（属设计建议，不带 prefix）：
1. **cos-sim 距离阈值**：bf16 输出 hidden state cos-sim ≥ 0.95（或 token-level logit cos-sim）作为 PASS 阈值；详 cos-sim 路径 ready 化见 §8.3
2. **启发式 PASS**：`_check_correctness` 已实证 10/10 可测 cell PASS（含 bos_spam=False / first_token_id sane / chars 在合理范围 / word_count 非异常），是低成本快速 sanity
3. **bf16 reference 距离对照**：跑 bf16 model 作 reference，fp8 输出与 bf16 reference 距离 vs baseline fp8 与 bf16 reference 距离对比，差异在阈值内即 PASS

---

## 6. 推测根因（3 条标【推测未实证】+ 1 条已实证排除）

> 【实证】lead+user (α) 决策**不追根因实证**直接结案 — 节约 30 tool calls / 1 wave；根因实证不会改变 wave 2 结案决策（即使确认是下面任一条，都是上游 aiter 的事，不是 wave 2 scope）。

### 6.1 ASM kernel list 缺失（推测）

【推测未实证】tuner 漏搜 ASM kernel 候选：
- 实证现象：tune sweep 报 warning `ASM kernel list file not exist: /workspace/aiter/hsa/gfx942/fmoe//fmoe_bf16_blockscaleFp8_g1u1_.csv`（来源 `c01_aiter_summary.md §3` + `WORK_DIR/logs/c01_tune_sweep.log`）
- 推测：tuner 只搜了 CK 候选 → 选出"CK 池里最优"，但 fused_moe.py:932+ fallback heuristic 能拿到"全空间最优含 ASM"
- **本 wave 未实证此因果链**（layer dispatch dump 缺失），仅是合理 mechanism

### 6.2 量纲错配（推测）

【推测未实证】tune cmd 量纲与生产工况错配：
- tune cmd 实证：`--batch 100 --errRatio 0.5`（来源 `c01_aiter_summary.md §3`）
- 推测：`--errRatio 0.5` isolated kernel bench 选出的 kernel 在 end-to-end 工况下（与 attention/kvcache 共享 HBM/L2）可能反而慢
- **本 wave 未实证此因果链**（无 kernel-level cache hit-rate dump 或 isolated vs e2e A/B 对照）

### 6.3 token dim dispatch 错配（推测）

【推测未实证】tune 加的 token={4,8} entry 量纲与 decode batch access pattern 不同：
- 实证：T6 P0-3 patch 给 untuned csv 加了 token={4,8} entry（来源 `WORK_DIR/progress/teammate-6.md`）
- 推测：tune 加的 token={4,8} entry 可能按 prefill 量纲 tune（4 个 prompt × 10240 token = 40960 token 一次 prefill），与 decode batch（np=4 decode 时 4 token/step）量纲 access pattern 不同 → dispatch 命中但配置不优
- **本 wave 未实证此因果链**（无 prefill vs decode 路径分别 dispatch trace + tile id dump）

### 6.4 已实证排除：dispatch HIT 6/6 ✅（互引 §2.1）

【实证】**dispatch HIT 6/6 已实证通过**（含完整 cfg dict：kernelName1/2、block_m、ksplit、us、err、tflops、bw、run_1stage、xbf16、_tag 全字段），来源 `WORK_DIR/c01_dispatch_verify.py` + `WORK_DIR/logs/c01_dispatch_verify.log`。

→ **可排除**"路径走 fallback 没命中 tuning entry"这一假设。tuning entry 真的被 dispatch 选中并执行（`_ZN5aiter59fmoe_stage1_bf16_pertokenFp8_blockscale_g1u1_32x128_3tg_pf3E` 等 kernel 实证），但**实测 perf 与 fallback 相当甚至更差** → 根因不在"是否命中"而在"命中但配置不优"（即 §6.1-6.3 三条推测之一）。

详见 §2.1 工程动作 #2 与 `c01_aiter_summary.md §5` 6 cell 完整 cfg dict 表。

### 末段红线

【实证】未来 wave 想重启 OPT-1 axis 前，**必须先 layer-dump 实证根因**（§6.1-6.3 任一），否则继续投资源是赌博。本 wave Promote-1 已显式 raise："未来 wave 不再投资源在 OPT-1（fmoe csv tuning）axis，应转向 OPT-7 / OPT-2 / OPT-3 等 axis（如 attention 优化 / cudagraph audit / sliding window）"。

---

## 7. Lessons & Promotes（6 candidate 全落地）

> 【实证修正】Promote 总数 = **6**（原 WAVE_CLOSE.md §5 列 5 个 + doc_t3 audit raise Promote-6 q_dtype dispatch key 修正）。本 doc plan 显式扩列 → user APPROVE 即视为采纳此扩列。

| # | Candidate | 措辞 | 主仓位 | 副仓位 |
|---|---|---|---|---|
| 7.1 | 救命级 #1 — q_dtype dispatch key | 【实证修正】 | 主文 §7.1 | §10 KNOWN_FACTS W1（wave2 独立编号） |
| 7.2 | OPT-1 axis 证伪 | 【实证】 | 主文 §7.2 | §10 KNOWN_FACTS W2（wave2 独立编号） |
| 7.3 | FP8 byte-drift 物理必然 | 【实证】 | 主文 §5/§7.3 | §10 KNOWN_FACTS W3（wave2 独立编号） |
| 7.4 | dispatch HIT≠perf 通用教训 | 【通用教训】 | 主文 §7.4 + appendix §9.4 | agent-team SKILL.md 反模式表 #21（独立 PR） |
| 7.5 | aiter shape bug | 【实证】 | 主文 §4 已承载 | 上游 GitHub issue（lead+user 决策） |
| 7.6 | bench script max_model_len bug | 【通用教训】 | appendix §9.5 | — |

### 7.1 救命级 #1 — q_dtype dispatch key —— `torch.float8_e4m3fnuz` not `e4m3fn`（Promote-6 → §10 W1）

【实证修正】aiter dispatch 路径上 **`q_dtype` 必须用 `torch.float8_e4m3fnuz`**（gfx942 native fnuz format），**不是** model 文件的 `e4m3fn`：

- **实证证据**：`/workspace/aiter/aiter/ops/moe_op.py:468 dtype2str_dict` 只支持 `torch.float8_e4m3fnuz`（aiter `dtypes.fp8` 的真实底层 type）
- **e4m3fn 字面理解会 KeyError**：按 untuned csv 原版 `q_dtype=torch.float8_e4m3fn` 跑 tune 立即 `KeyError: torch.float8_e4m3fn at gemm_moe_tune.py:2143`（实证 v1 fail log: `WORK_DIR/logs/c01_tune_sweep_v1_e4m3fn_FAILED.log`）
- **即使绕过 dispatch 走错路径**：用 e4m3fn 即使能 dispatch（旧 ASM 路径），会选 `asm_stage1`（**NEW-RC-3 patch f06cdcca5 已禁用**），矛盾；用 e4m3fnuz 才选 `ck_moe_stage1`（NEW-RC-3 强制路径，正确）

【实证修正】**wave2 独立 KNOWN_FACT W1 措辞要点**：「e4m3fn 是 stepfun model 实际 dtype」字面对（model 文件确实是 e4m3fn），但**dispatch key 不是** — sglang/aiter integration 在 dispatch 前 cast e4m3fn → fnuz。

→ **§10 W1 完整措辞**："dispatch key 是 `torch.float8_e4m3fnuz`（aiter dtype2str_dict 唯一支持），model weight dtype 是 `torch.float8_e4m3fn`（sglang/aiter cast 后才到 fnuz）"（注：与 parent F5 "FP8 decode TPOT 19%" 无关，独立编号；详见 §10 W# vs F# 关系节）。

来源：`c01_aiter_summary.md §2`。

### 7.2 OPT-1 axis 证伪（Promote-1 → §10 W2）

【实证】**stepfun-Flash-FP8 + gfx942 + aiter HEAD `315123ace` 路径上，补 fmoe csv stepfun-specific entries（OPT-1）无效甚至有害**：
- np=1：9 metric 0/9 ≥5% 改善，最大 +0.7%，全在 ±1.5% 噪声 floor
- np=4/8 tp=2：3 反向退化（TTFT +10.3% / TPOT +3.1% / decode -3.1%）
- np=4/8 tp=4/8：上游 aiter `apply_act_and_mul` shape bug 不可测

→ **§10 W2 完整措辞**："未来 wave 不再投资源在 OPT-1（fmoe csv tuning）axis，应转向 OPT-7（cudagraph audit）/ OPT-2（sliding window attention）/ OPT-3（fmoe 1-stage 路径选择）等 axis；除非先 layer-dump 实证 §6.1-6.3 任一根因（ASM kernel list / 量纲 / token dim dispatch）已修复"（注：与 parent F8 "aiter commit c38d0c9e6" 无关，独立编号）。

来源：`v01_perf_matrix.md` + `v03_perf_matrix_multiprompt.md` + `c01_aiter_summary.md §3` + archival memory "后续可选 axis (按 ROI 排序)"段。

### 7.3 FP8 byte-drift 物理必然（Promote-2 → §10 W3，与 §5 互引）

【实证】见 §5 完整论述。简表：

→ **§10 W3 完整措辞**："FP8 kernel 替换在 temp=0 greedy 路径**物理上必然**导致 byte-drift（不是 bug，是数值物理特性）；**未来 wave 不再用 byte-equal 作 fp8-vs-fp8 verification**（必然 FAIL）；改用 cos-sim 距离阈值 / 启发式 PASS / bf16 reference 距离对照（详 §5.4）"（注：与 parent F9 "aiter commit a2883ab37" 无关，独立编号）。

来源：`v02_byte_equal_report.md §6.1` + `WAVE_CLOSE.md §4.1` (lead 自反 P0-2=β GOAL 设计错)。

### 7.4 dispatch HIT≠perf 通用教训（Promote-3 → appendix §9.4 + agent-team SKILL.md 待独立 PR）

【通用教训】见 §2.2 完整论述 + §9.4 详写。

【通用教训】**主仓位**：本 doc appendix §9.4（与 §2.2 互引）。**副仓位**：agent-team SKILL.md 反模式表 #21（独立 PR review，doc_t2 §Promote workflow 路径建议明示"双 promote 双向一致但 Edit 路径不重叠"，跨 skill PR 应分 2 路独立 review）。

【通用教训】副仓位 PR 不在本 doc scope 范围执行（user 决策 Q6=(c) 推但分独立 PR）。

来源：archival memory "通用调试教训" + 反模式表 #20 「Self-report 即 ground truth」延伸。

### 7.5 aiter shape bug（Promote-4 → §4 已承载 + 上游 issue 留 user 决策）

【实证】见 §4 完整论述（独立 H2 救命级 #2）。

【实证】**Promote-4 副仓位 (e) 上游 GitHub issue 留 user 决策**：建议开 issue 给 ROCm/aiter，含本 wave V03 8 个 `v03_*tp{4,8}_np{4,8}_full.log` crash trace 作 artifact。**本 doc 不承载 upstream issue 起草**（lead+user 决策）。

来源：`WAVE_CLOSE.md §5 Promote-4` + `v03_perf_matrix_multiprompt.md §6.3`。

### 7.6 bench script max_model_len bug（Promote-5 → appendix §9.5）

【通用教训】见 §9.5 完整论述。简表：5 行 trivial patch（`getattr(args, "max_model_len", 0)` 改 `getattr(args, "max_model_len", None) or 0`），不到主文层级；列入 OPT-6 follow-up。当前绕过 = cmdline 显式传 `--max-model-len 11264`（合规 ATOM EngineArgs flag）。

来源：archival memory "bench script 已知 bug" + `WAVE_CLOSE.md §5 Promote-5`。

---

## 8. Future Work

### 8.1 OPT-1 重启 prerequisites

【实证】未来 wave 想重启 OPT-1 axis 前必须先解决：
1. **layer-dump 实证根因**：派 dev-debug teammate 在 stepfun load 后做 layer-by-layer dispatch trace，对每个 fmoe layer dump (selected kernel name, tile config, inter_dim, num_tokens, batch state)，对照 baseline vs tuning 双 state 各自 dispatch 选择，证实 §6.1-6.3 哪一条是真根因
2. **ASM kernel list 补**：定位 `/workspace/aiter/hsa/gfx942/fmoe/fmoe_bf16_blockscaleFp8_g1u1_.csv` 是否有上游补的版本，无则需先生成 ASM 候选列表（上游 aiter PR 或本地 build）
3. **dispatch verify 在 baseline + tuning 两 state 各跑一次**：避免本 wave 只验证 tuning state HIT 6/6，不知道 baseline state 同样 key tuple 会 fallback 到什么 kernel 作对比

### 8.2 multi-prompt + tp≥4 阻塞解除条件

【实证】见 §4.4。简表：
- aiter `apply_act_and_mul` shape bug 必须先 fix（上游 PR 或本地 patch）
- 验证方法：跑 1 cell baseline + np=4 + tp=4 sanity check，不崩则可继续 V04 重测 12 cell
- 阻塞解除前**所有 multi-prompt + tp≥4 + fp8 ck2stages 实验都被 block**

### 8.3 cos-sim 路径 ready 化

【实证】本 wave Phase 1 派 teammate 调研 10 个 cos-sim 候选路径（来源 `WORK_DIR/cos_sim_path_survey.md`），**全 PARTIAL / DOC-ONLY**，无可直接用的 ready 化路径。

【推测未实证】未来 wave 想用 cos-sim 作 fp8-vs-fp8 verification（替代物理上必然 FAIL 的 byte-equal）前需先：
1. 派 teammate 选 1-2 个最 ready 的候选路径做 e2e 实施 + 阈值标定
2. baseline cos-sim 数据收集（fp8 vs bf16 reference 的距离基线）
3. 阈值确定（cos-sim ≥ 0.95？ ≥ 0.99？需结合下游 task accuracy 对照）

---

## 9. Appendix

### 9.1 物料 path 索引

【实证】本 wave 所有持久 artifact 在 `WORK_DIR = /home/junlin12/step35_fp8_optim_wave2/`：

| 类型 | path | 用途 |
|---|---|---|
| WAVE_CLOSE | `WORK_DIR/WAVE_CLOSE.md` | 最完整收尾文档（含 cost summary + 5 Promote candidate） |
| TEAM_CONFIG | `WORK_DIR/TEAM_CONFIG.md` | wave 配置（含 baseline anchor + 量化 GOAL） |
| archival memory | `~/.claude/projects/-home-junlin12/memory/stepfun_fp8_fmoe_wave2.md` | user 校对过的 ground truth |
| aiter csv backup（baseline） | `WORK_DIR/c01_aiter_csv_backup.csv` | 623 行 / md5 `66ca8222b02fba8200fa7aea5625d8ee` |
| aiter csv tuning snapshot | `WORK_DIR/tuned_fmoe_pre_v01.csv` | 665 行（保留供未来 OPT-1 重启基线） |
| aiter csv patch（**已 rollback**，仅备份） | `WORK_DIR/c01_aiter_csv.patch` | 50 行 / +42 entry |
| ATOM bench patch（**未 rollback**） | `WORK_DIR/c01_atom_bench.patch` | 208 行 +105/-29 / 加 `--ignore-eos` `--num-prompts` `--enable-cudagraph` flag |
| perf 矩阵 v01 | `WORK_DIR/v01_perf_matrix.md` | Phase 3 v1 6 cell np=1 |
| perf 矩阵 V03 | `WORK_DIR/v03_perf_matrix_multiprompt.md` | Phase 4 V03 18 cell |
| byte-equal report | `WORK_DIR/v02_byte_equal_report.md` | Phase 3 v1 byte-equal FAIL 实证 |
| escalation | `WORK_DIR/ESCALATION.md` | E-1（v01 双 FAIL）/ E-2（Bash transient）/ E-3（V03 + aiter bug） |
| logs/ | `WORK_DIR/logs/v0{1,3}_*.log` × 36 + `c01_{tune_sweep,dispatch_verify}.log` 等 | 见 logs/INDEX.md |
| progress/ | `WORK_DIR/progress/teammate-{0..9}.md` | 10 个 teammate self-report，见 progress/INDEX.md |
| tune sweep output | `WORK_DIR/tuned_fmoe_step35.csv`（43 行） + `WORK_DIR/profile_fmoe_step35.csv`（全候选 timing） | T7-B 实证 |
| dispatch verify | `WORK_DIR/c01_dispatch_verify.py` + `WORK_DIR/logs/c01_dispatch_verify.log` | 6/6 HIT 实证 |

### 9.2 Cost summary 一句话

【实证】wave 总 tool calls **~196/200 budget**（teammate-0 ~25 + T1/T2/T4/T3 ~60 + T5/T6 ~16 + T7-A/B ~36 + T8 ~16 + T9 v1+resume ~30 + lead ~13）；wave wall time ~5-6 hours（含 Phase 0 baseline + Phase 2 tune sweep + Phase 3/4 GPU bench 矩阵）；GPU usage ~3-4 hours（tune 125s + 18 cell × ~5 min + 1 hour 调研）；估算 USD ~$50-70（per AMD LLM Gateway 30 天 ratio 推算）。详 `WAVE_CLOSE.md §7`。

### 9.3 ESCALATION 一句话引用

【实证】wave 内 3 次 ESCALATION 全实证落地：
- **E-1**（v01 双 FAIL）：teammate-8 raise 4 option，lead 选 (2) → 派 teammate-9 跑 V03 multi-prompt
- **E-2**（Bash transient outage）：lead + teammate-9 v1 同 session 触发，user 重启 session 后恢复（~10 min）
- **E-3**（V03 + aiter bug）：teammate-9 raise，lead 选 (α) 接受 wave 2 FAIL 收摊，rollback csv + write WAVE_CLOSE

详 `WORK_DIR/ESCALATION.md`。

### 9.4 dispatch HIT≠perf 教训详写（与 §7.4 互引）

【通用教训】**核心命题**：tune sweep 跑通 + dispatch HIT 验证通过 = 工程动作 done，但**不等于 wave PASS**。perf 矩阵实测才是 wave PASS 唯一标准。

【通用教训】本 wave 实证三件套：
1. **tune sweep**：T7-B 实证 42/42 success / wall 125.5s / 0 failed → 工程动作 done ✅
2. **dispatch HIT**：T7-B 实证 6/6 HIT 完整 cfg dict（kernelName1/2 / block_m / ksplit / us / err / tflops / bw / run_1stage / xbf16 / _tag）→ 调度路径正确 ✅
3. **perf 矩阵**：T8 + T9 实测 27 metric 综合 0/27 ≥5% 改善 + 3 反向退化 + 8 cell crash → wave PASS criterion FAIL 🔴

【通用教训】**反模式触发条件**：lead 看到工程动作 done + dispatch HIT 验证通过两条都 PASS 后，**容易**跳过实测 perf 矩阵直接判 wave PASS（认为"该改的都改了，HIT 也证明了"）。本 wave 实证此反模式必然踩坑：dispatch HIT 6/6 但 perf 0/27，因为 tune 出来的 kernel 与 fallback heuristic 选出的 kernel 性能相当（甚至更差，§3.3 三个反向退化数据点）。

【通用教训】**正确做法**：
- tune sweep 完成 → mark "工程动作 done"，**不**判 wave PASS
- dispatch HIT 验证 → mark "调度路径正确"，**不**判 wave PASS
- perf 矩阵实测 ≥5% 改善 → 此时才能判 wave PASS
- 三者中**任一缺失**都不能判 PASS

【通用教训】**推 agent-team SKILL.md 反模式表 #21 候选措辞**："dispatch HIT ≠ perf 改善（即使新 entry 命中也可能比 fallback heuristic 差）— 不要把 tune sweep 跑通 + dispatch HIT 验证通过当 wave PASS 信号，跳过实测 perf 矩阵直接收尾"。是反模式表 #20「Self-report 即 ground truth」延伸（PASS 不能只看工程动作完成，必须 perf 实证）。

### 9.5 bench script max_model_len bug fix path

【通用教训】**位置**：`/home/junlin12/project_summary/step35-flash-support/details/scripts/perf_correctness_bench.py:260`

【通用教训】**触发**：np × input_tokens > 16384 + 用户未显式传 `--max-model-len`

【通用教训】**现象**：`getattr(args, "max_model_len", 0)` 在 args 已 setattr 但 value=None 时返回 None vs int 比较 TypeError（注：ATOM EngineArgs 默认会 setattr `max_model_len=None`，此时 getattr 返回 None 而非 default 0）

【通用教训】**当前绕过**：cmdline 显式 `--max-model-len 11264`（合规 ATOM EngineArgs flag，不改源码）— 本 wave V03 12 cell 全用此绕过

【通用教训】**5 行 trivial patch**：
```python
# before (line 260)
if np * input_tokens > 16384 and getattr(args, "max_model_len", 0) < args.input_tokens + 1024:
# after
_max_model_len = getattr(args, "max_model_len", None) or 0
if np * input_tokens > 16384 and _max_model_len < args.input_tokens + 1024:
```

【通用教训】**列入 OPT-6 follow-up**（不到主文层级）；未来跑 multi-prompt 矩阵的 wave 可顺手修。

来源：archival memory "bench script 已知 bug" + `WAVE_CLOSE.md §5 Promote-5` + `v03_perf_matrix_multiprompt.md §0` 第 4 点。

---

## 10. KNOWN_FACTS（wave2 新增 / 独立编号 W1-W3）

> 设计说明：parent `verification_pipeline/TEAM_CONFIG_verification.md` 的 F# 表 (F1-F20) 持有 **硬件 / commit / 实测 perf 数字** 类事实；wave2 新增的 3 条属 **wave-defining 决策类**（误用→后续 wave 直接踩坑），类目不混。本节独立 W# 编号，未来 wave 检索 wave2-specific KNOWN_FACTS 来此节。parent F# 表保持不动。

| # | 事实 | 来源 | 救命级 |
|---|------|------|--------|
| **W1** | 【实证修正】FP8 在 gfx942 上 dispatch key 必须 `torch.float8_e4m3fnuz`（not `e4m3fn`）；csv 中 `q_type: fnuz` 是 dispatch fnuz format 名，不是 e4m3fn → 用 e4m3fn 直接 KeyError 或绕过走错路径（sglang/aiter 内部 cast） | doc_t3 audit / c01_aiter_summary.md §2 / wave2 phase0 baseline | ⭐⭐⭐ #1 |
| **W2** | 【实证】OPT-1 fmoe csv tuning 在 stepfun-Flash-FP8 上**全 axis 证伪** — 27 metric 0 改善 + tp=2 multi-prompt 反向退化（TTFT np=4 +10.3% / TPOT np=8 +3.1% / decode np=8 -3.1%）；dispatch HIT 6/6 ≠ perf 改善（HIT 是必要非充分） | v01_perf_matrix.md / v03_perf_matrix_multiprompt.md / wave2 WAVE_CLOSE.md | ⭐⭐ |
| **W3** | 【实证】FP8 kernel swap 必伴 byte-drift（累加顺序 / tile / splitk variance → bf16 低位飘 → softmax 边界翻 → token 序列发散）— 物理必然，**不要用 byte-equal 作 fp8-vs-fp8 verification**；替代：cos-sim ≥ threshold / top-k overlap / perplexity Δ / task accuracy | wave2 archival memory + RESULTS.md §5 | ⭐⭐⭐ #3 |

### W# vs parent F# 关系
- W1 与 parent F5（FP8 decode TPOT 19%）**无关**；W1 是 dispatch key 修正
- W2 与 parent F8（aiter commit c38d0c9e6）**无关**；W2 是 OPT-1 证伪结论
- W3 与 parent F9（aiter commit a2883ab37）**无关**；W3 是 byte-drift 物理特性
- 早期 doc-planning 阶段（TEAM_CONFIG_DOC.md / WRITE_PLAN.md / archival memory）曾以 "F5/F8/F9" 编号引用本 wave 的同一组事实，假设与 parent KNOWN_FACTS F# 表同号映射；impl 阶段 survey 实证 parent F5/F8/F9 已被无关事实占用 (FP8 decode TPOT / aiter commit)，经 user APPROVE 改用本节独立 W# 编号 — **以 W1/W2/W3 为本 wave 最终规范编号**，早期 F# 引用视为别名 obsolete

---

> **wave 关闭。** 本 doc 是 step35-fp8-fmoe-tuning-wave2 wave 的主报告（FAIL 显式声明 + 27 metric 实证 + 6 Promote 全落地 + 救命级 Top 3 全 H2 主章节承载）。
