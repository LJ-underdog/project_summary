# ★ lead 闸门裁决(2026-06-15)+ critique 解决 —— 实现以此为准,与下文冲突处以此覆盖

> 本 draft 由 M8 scoping workflow 产出,**critique verdict=needs-revision**。lead 裁决:**本期 M8 scope = MI + B2/B3 only**(runtime 真赢),其余暂缓 → critique 关于 B4/grid/VGPR 的 must-fix 对本 scope **moot**(那些候选不做)。

## 本期 scope(只做这 3 项)
1. **MI 测量基线(前提)**:harness 加 `-perf` flag(behind flag,253 套件不受扰)—— hipEvent gpu_timer **envelope(PRE+memset+MAIN+POST)+ 每-kernel**(per-kernel 必需,因 B2/B3 只动 MAIN,要单独归因 MAIN 加速)。镜像 fwd `-perf`(`example_hstu_attention_fwd.cpp:243` flag、`:664` gpu_timer)。FLOPS 模型 = 5-GEMM(注:忽略 elementwise,报 GEMM-only TFLOPS 作 tracking 非 roofline)。benchmark.csv schema:`candidate,arch,mode,activation,dtype,hdim,kernel{envelope/PRE/MAIN/POST},metric{time_ms/TFLOPS/occ},value,date`。记录基线行(canonical + hd256 + window 各档)。
2. **B2 GetTileRangeAlongY 紧致化 — causal**:现保守全扫 (0,seqlen);收紧到每 kv-tile 的真实 q-range。
3. **B3 GetTileRangeAlongY 紧致化 — local/window**(causal 通过后)。

## B2/B3 silent-wrong 安全要求(critique + M2 工具,**硬 gate**)
紧致化 = 减少 MAIN 扫的 q-tile;**收紧过头 = 永久丢 dK/dV/dQ 贡献、不崩溃**。每个收紧候选必须:
- (a) 收紧后的范围必须是**真实所需范围的严格 kM0-aligned superset**(start `align_down`、end `align_up`/clamp);
- (b) **contextual 行 q_start=0**(它们 attend 所有 q 行,`hstu_block_masking.hpp:206/463`);(c) **min_full_attn 全 reach 行**;(d) **cross-attn `diff_q_kv_len` 偏移**(`:89/:159`);(e) **non-causal NoLocal 的 num_target 行不能排除**(P1-1)。
- **★ 硬安全 gate = 离线 superset 校验器**:`test/validate_tile_range_y.cpp`(M2 遗留,exhaustive,无需 GPU)—— 收紧后**必须 ALL GREEN**(穷举证明仍 superset,这比对拍更强,直接挡 under-tighten)。先扩它覆盖 cross + 新收紧逻辑再跑。

## 每候选验证 gate(evidence-driven,rocm-kernel-design)
① 离线 superset 校验器 ALL GREEN(B2/B3);② 对拍套件 **253/253**(102 case def 参数化展开,实测 run 数,非臆造)+ 边界 stress(contextual/min_full/num_target/cross/window 逐项);③ **MAIN 加速 vs MI 基线**(per-kernel timing,记 benchmark.csv);④ co_symbols 对**未受 GetTileRangeAlongY 影响的路**仍 byte-identical(MI 是 behind-flag 应 byte-identical;B2/B3 改 mask range → 设备码本就变,不要求 byte-identity,靠 ①②③ 兜)。

## 暂缓(本期不做,critique 的 grid/VGPR must-fix 随之 moot)
- **B4 grid widening**:scoping 的 "grid starvation 根因" 被 critique 证伪(实测 256x 超额订阅),依据作废,暂缓。
- **B7 hd256 occupancy / VGPR 124-vs-248**:阻塞于先解 VGPR 矛盾 + LDS-vs-MFMA 测量,本期不做。
- B5 first-split-skip(silent-wrong 分析不足)、B6 trload(高风险高工)、B1 group TU split(build-axis,可日后并行)、INV SiLU 26% 异常、B8/B9/B10 —— 均暂缓。
- **修正记录**:真限制器是 per-block LDS/VGPR(非 grid 数);MemUnitStalled=0.024%(非 2.4%)。这些对 MI+B2/B3 scope 无影响(我们不碰占用率)。

---

# M8 Performance Plan — HSTU Backward (gfx950/MI350X)

Status: DRAFT for review. Synthesizes 6 analyses including one empirical rocprofv3 profile. Every numeric claim is tagged **[measured]**, **[derived]** (from measured + a model), or **[guess]** (estimate pending the timing harness this plan builds).

---

## 0. Goal Framing — Two Separate Perf Axes

M8 conflates two unrelated kinds of "performance." They have different metrics, different validation, and must not be traded against each other.

### Axis A — Build-time perf (dev iteration speed)
Metric: object-file compile wall-time. Validation: **byte-identical `.o`** (or identical suite pass) — these changes are required to be runtime-neutral and correctness-neutral.
- **Group TU split** — `group_backward_bf16.cpp.o` is **17,181,816 B** vs `no_group_backward_bf16.cpp.o` **10,192 B** **[measured]** (`hd256-occ-and-build`). One inline fan-out object is the single biggest build bottleneck.
- **CMake slim / dead fwd legs** — over-linking; must keep reachable fwd legs (`tile-firstsplit-and-misc`, `tile-and-build`).

### Axis B — Runtime perf (kernel TFLOPS / wall-time)
Metric: per-kernel time and GEMM-TFLOPS. Validation: speedup **AND** no-regression on the 253-case oracle suite + `co_symbols`/suite check (per `rocm-kernel-design`). Several of these are **silent-wrong** risks.
- GetTileRangeAlongY tightening, hd256 occupancy, trload pipeline, first-split skip, pad/vectorization.

> The two axes are decoupled: build-time work is safe and parallelizable; runtime work is correctness-sensitive and **blocked on measurement infra (§2)**.

---

## 1. Current-State Empirical Snapshot (from rocprofv3 profile)

All from `profile-baseline`. Canonical small config: `-prec=bf16 -b=2 -nhead=8 -seqlens=2048 -softmax=1 -causal=1`, hd64.

| Fact | Value | Tag |
|---|---|---|
| Kernel that dominates | **MAIN dqdkdv = 84–90% of bwd wall-time** across every config | [measured] |
| MAIN time (small config) | **266.20 us** (PRE 28.20 / fill 3.80 / POST 5.68 → 303.88 total, MAIN 87.6%) | [measured] |
| hd256 MAIN | **943.25 us** of 1068.65 (88.3%) | [measured] |
| MAIN occupancy | **3.39 waves/CU = 10.6%** | [measured] |
| MfmaUtil | **9.9%** (matrix cores idle >80% of the time) | [measured] |
| VALUBusy / MemUnitStalled | 22.5% / **2.4% → NOT memory-bound** | [measured] |
| VGPR | 124/thread, Scratch 0 → ~16 waves/CU headroom, **not** the limiter | [measured] |

**Root cause #1 — grid starvation [measured].** Grid is dKdV-centric: `dim3(ceil(seqlen_kv/kN0), nhead, batch)` = `(16,8,2)=256` workgroups for **256 CUs** → ~1 block/CU, no second wave for latency hiding (`hstu_attention_bwd_kernel.hpp:210-212`). Validated: a 16x-larger grid (4096 wg) lifts occupancy 3.39→**6.96 waves** (10.6→21.8%) and MfmaUtil 9.9→**18.4%**.

**Root cause #2 — LDS cap + serial Q-loop [derived].** Even with abundant workgroups, occupancy plateaus near **2 blocks/CU** = 32KB LDS/block ÷ 64KB/CU. The LDS-vs-dependency-chain split is **inferred, not counter-proven** (open question).

**Anomaly [measured]:** SiLU (softmax=0) MAIN = 335 us vs softmax 266 us at identical shape — a 26% inversion, unexplained.

**Infra gap [measured]:** the bwd harness has **no kernel-timing flag** (unlike fwd `-perf` at line 243). rocprofv3 `--kernel-trace` is the only working path today; PMC multi-pass aborts at finalization (flushes partial CSVs).

**Headline:** matrix cores sit idle ~90% of the time. The win is occupancy + cutting wasted MAIN iterations — not memory bandwidth.

---

## 2. Measurement Infrastructure (PREREQUISITE — do first)

No runtime candidate may be merged without a hipEvent baseline. rocprofv3 single-shot trace has ~few-% jitter and runs kernels once (no warmup) — adequate for ratios, **not** for tracking incremental candidate speedups.

**Plan (`measurement-infra`):** clone the fwd `-perf` path into bwd, gated behind `-perf` so the 253/253 suite is untouched (`if(measure_perf)` block appended after validation; dispatch lines 603–610 stay identical).
- **Phase 1 (recommended first):** envelope timing via `gpu_timer` wrapping PRE+memset+MAIN+POST (`timer.hpp` start/stop sync+record; warmup + repeat, `ms=dur/10`). Zero dispatch-signature edits.
- **Phase 2:** per-kernel via `launch_kernel` `time_kernel_` bool + `stream_config{stream,false}` cold3/nrepeat10 median. **Required** to attribute MAIN-only speedups (GetTileRangeAlongY/trload only move MAIN).

**FLOPS model:** bwd = 5 GEMMs ≈ 2.5x fwd loop: `2*(2*sq*skv*hdim_qk + 3*sq*skv*hdim_v)*b*nhead`. **Caveat [measured]:** ignores SiLU/softmax/dS elementwise → reports GEMM-only TFLOPS, not HW utilization. Attribute the separate `memset` launch to ZERO_dq_acc, else MAIN TFLOPS reads artificially low.

**benchmark.csv schema:** `candidate,arch,mode,activation,dtype,hdim,kernel,metric,value,date` where `kernel ∈ {envelope, PRE_dot_do_o, MAIN_dqdkdv, POST_convert_dq}` and `metric ∈ {time_ms, TFLOPS, effective_BW_GBs, occupancy_pct}`.

**Per-candidate gate (rocm-kernel-design):** (1) speedup vs recorded baseline row, (2) 253/253 suite hold, (3) `co_symbols`/suite no-regression. Open: pin the benchmark workload mix — only one C0 causal hd64 bf16 row exists today; ROI of several candidates depends entirely on the mask/seqlen/window distribution, which is not yet pinned.

---

## 3. Prioritized Candidate Table

ROI/risk are **[guess]** unless the profile supports them. "Silent-wrong" = produces incorrect numbers with no crash (suite would regress to wrong values, not abort).

| # | Candidate | Axis | Est. ROI | Risk | Effort | Silent-wrong? | Basis |
|---|---|---|---|---|---|---|---|
| **MI** | Measurement infra (§2) | enabler | unblocks all of B | low | low–med | no | blueprint exists (fwd `-perf`) |
| **B1** | Group TU split (mirror batched) | **Build** | **17.2MB→64 parallel objects** [measured]; biggest build bottleneck | **low** | **low** (~3 edits, no CMake change, byte-identical) | no | `hd256-occ-and-build` |
| **B2** | GetTileRangeAlongY tighten — **causal** | Run | **~1.9x MAIN** at 4k seqlen (~48% wasted) [derived] | med (silent-wrong) | low–med (4 constexpr fns, transpose of validated fwd GetTileRangeAlongX) | **YES** | `gettilerange-waste` |
| **B3** | GetTileRangeAlongY tighten — **local/window** | Run | **4x–22x MAIN** (w=256/8k → 21.7x, 95% wasted) [derived] | med (silent-wrong) | low–med (same fns) | **YES** | `gettilerange-waste` |
| **B4** | Grid widening / split-K over Q (raise resident wg) | Run | occupancy 10.6→21.8% **demonstrated** by 16x-grid test [measured] | med–high | med–high | no (semantics-preserving if reduction correct) | `profile-baseline` RC#1 |
| **B5** | first-split skip (hardcoded first split) | Run | moderate | med (re-run suite) | low | possible | `tile-firstsplit-and-misc` `bwd_kernel.hpp:410,857` |
| **B6** | trload `ds_read_tr16` pipeline (port SiLU/softmax into FMHA trload body) | Run | **1.2–1.5x MAIN** [guess], LDS-bound dependent | **med–high** | **high** (separate forks, hard `static_assert` against trload) | **YES** (SiLU edge masking in ping-pong body) | `trload-pipeline` |
| **B7** | hd256 occupancy 1→2 (LDS trim + `kBlockPerCu` 1→2) | Run | up to 2x on hd256 only [guess] | high (tile redesign, shared layout 253 suite depends on) | med–high | possible | `hd256-occ-and-build` |
| **B8** | PRE dot_do_o vectorize | Run | small (PRE only 9%) | low | low | no | `tile-firstsplit-and-misc` |
| **B9** | pad align-1 → vectorized loads | Run | small–med [guess] | low (byte-identical target) | low | no | `tile-firstsplit-and-misc` |
| **B10** | CMake slim / dead fwd legs | Build | build-time, modest | low (keep reachable fwd) | low | no | `tile-firstsplit-and-misc` |
| **INV** | Investigate SiLU 26% slowdown | Run | unknown | n/a | low (profiling) | no | `profile-baseline` anomaly |

**Critical silent-wrong notes for B2/B3 (`gettilerange-waste`):** the in-loop `IsTokenPairInsideMask` zeroing only fixes tiles that ARE visited; a wrongly-excluded q-tile permanently drops its dK/dV/dQ with no error. The tight range MUST: (a) stay a strict **kM0-aligned superset** (`align_down` start, `align_up`/clamp end); (b) force `q_start=0` for `contextual_seqlen` rows (they attend all q-rows, `:206/:463`); (c) handle `min_full_attn_seqlen` full-reach rows; (d) apply cross-attn `diff_q_kv_len` shift (`:89,:159`); (e) not exclude `num_target` rows in non-causal NoLocal (P1-1).

---

## 4. Recommended M8 Sequencing

Order: enabler → highest-ROI-lowest-risk → high-effort/high-risk last. Build-axis (B1, B10) runs in parallel — it is independent and runtime-neutral.

1. **MI — measurement infra (Phase 1 envelope, then Phase 2 per-kernel).** Hard gate for everything in Axis B. Record baseline rows in benchmark.csv (canonical small config + hd256 + window).
2. **B1 — group TU split** *(parallel track, build-axis).* Highest ROI for dev iteration, lowest risk, byte-identical. No dependency on MI.
3. **B2 + B3 — GetTileRangeAlongY tighten (causal then window).** Highest runtime ROI per unit effort; transpose of already-validated fwd logic. **Gate on Phase-2 MAIN-only timing** (the win is MAIN-specific) + 253/253 suite. Do causal first (simpler), then window (largest ROI but more special-cases).
4. **INV — diagnose SiLU 26% slowdown.** Cheap (profiling only); may reveal a quick MAIN win and de-risks B6's SiLU port.
5. **B4 — grid widening / split-K over Q.** Directly attacks the #1 measured root cause for small-batch/short-seq recsys shapes (grid<256). Medium effort; semantics-preserving if the reduction is correct.
6. **B9, B8 — pad vectorization, PRE vectorize.** Low-risk fills.
7. **B6 — trload pipeline.** High effort/high risk; do only after MI shows MAIN is **LDS-bound** (rocprofv3 LDS-traffic vs MFMA/VALU-busy). If MFMA-bound, the win is small — don't pay the port cost.
8. **B7 — hd256 occupancy.** Last: tile redesign touches the shared layout the suite depends on, hd256-only payoff. Note: LDS trim alone is inert without `kBlockPerCu` 1→2 (hard-pinned via launch_bounds).
9. **B10 — CMake slim** *(parallel, build-axis, opportunistic).*

**Dependency rule:** B2/B3/B4/B6/B7 each block-by MI Phase 2. B6 blocks-on INV + an LDS-vs-MFMA balance measurement.

---

## 5. Non-Goals / Deferred

- **fp16 and cross-attention runtime tuning** — only self/group bf16 was profiled; same kernel, expected similar, **unverified [measured]**. Defer dedicated tuning.
- **PRE/POST optimization beyond B8** — PRE 9%, POST 2% [measured]; negligible, not worth structural work.
- **Eliminating GEMM-only TFLOPS gap to true HW util** — FLOPS model ignores elementwise; treat reported TFLOPS as a tracking metric, not an absolute roofline claim.
- **PMC multi-pass robustness** — known to SIGABRT at finalization; stay on single-counter-group passes + `--kernel-trace`. Not worth fixing in M8.
- **Workload-balance redesign from causal tightening** — B2 makes col-0 kv-tiles heavier than col-seqlen tiles; a grid/scheduling rebalance is a possible follow-on, **deferred** until B2's realized imbalance is measured.
- **Unifying SiLU + softmax into one `kUseSoftmax` body** — desirable to avoid porting twice in B6, but a refactor in its own right; deferred unless B6 is greenlit.
- **Pinning the production workload mix** — out of scope to *decide*, but flagged: B2/B3 realized ROI (1.9x vs 22x) depends entirely on the seqlen/window/contextual/num_target distribution, which must be supplied before final ROI is claimed.