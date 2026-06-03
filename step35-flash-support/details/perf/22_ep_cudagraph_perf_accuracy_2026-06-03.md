# EP (expert-parallel) 配置 perf + 精度 + cudagraph 修复（2026-06-03）

> **来源**：W8_resume wave teammate-24/25/26/27/28/29/30。本文记录 **EP（`--enable-expert-parallel`）配置**下的实测 perf、cudagraph 可用性结论、EP 精度（弱验证）、以及 EP 与纯 TP 的关键区别。
>
> 🔴 **头号铁律（读前必看）**：本文所有 perf 数字都是 **EP 配置（inter=1280，未沿 TP 分片）**。它们 **不可** 与 REPRODUCE.md §6.2 的现有 anchor（TTFT 1665/980/747ms、TPOT 15.5/14.5/13.7ms）混读 —— **后者是纯 TP（inter 沿 TP 分片到 640/320/160）**（teammate-30 四重证据坐实）。EP 与 TP 是不同 parallelism 路径，**不同配置不可直接比**。

---

## 1. EP 配置 perf（cudagraph 态，TP8）

### 1a. decode TPOT（batch=1 单序列，短输出）— teammate-27
口径：TP8 + EP + fp8 + **cudagraph ON（无 `--enforce-eager`）** + `cudagraph_capture_sizes=[1]` + 单序列 batch=1 + `max_tokens=256` + `ignore_eos=True`（强制满 256 步）+ temperature=0；TPOT = ATOM engine 内置 `(leave-first_token)/(num_out-1)`。

| 指标（**EP, cudagraph, batch=1**） | 值 |
|---|---|
| decode TPOT | **~12.62 ms/tok** |
| decode throughput | **~77 tok/s** |
| cudagraph vs eager 加速 | **~7.8–8.2×**（eager anchor TPOT ~99–104ms，teammate-25 → cudagraph ~12.62ms）|

### 1b. prefill TTFT（长输入 10213 tok）— teammate-28
口径：TP8 + EP + fp8 + cudagraph ON（`--enable-cudagraph` + `--cudagraph-capture-sizes "[1]"`）+ `--input-tokens 10240`（实际 10213）+ `--output-tokens 256` + `--ignore-eos` + 单序列 + temperature=0；脚本 `details/scripts/perf_correctness_bench.py`（未改，只读 CLI 用）。

| 指标（**EP, cudagraph, input=10213**） | 值 |
|---|---|
| prefill TTFT | **~560–571 ms** |
| decode TPOT | **~13.5 ms/tok** |
| decode throughput | **~74 tok/s** |

> TPOT 13.5ms 恰好接近 REPRODUCE §6.2 的 **TP** anchor 13.7ms，但这是巧合性接近 —— **两者是不同 parallelism 路径（EP vs TP），不构成可比对照**。TTFT 560–571ms < TP anchor 747ms 是因本测 `gpu-mem-util`/`max-model-len`/input 口径与 anchor 档不同，非可比 regression。

### 1c. 🔴 EP 下 "nopad vs pad" = 同一路径，**不是** nopad/pad perf 对比（teammate-29 坐实）
- T27/T28 的 progress 把两次 run 标了 "nopad" / "pad"，但 **EP 下 inter=1280 ≥ 256**，`ATOM_FP8_MOE_DISABLE_PAD` 是 **no-op** → 两次 run 走的是 **同一条 inter=1280 路径**。
- 故 T27/T28 观察到的 "nopad vs pad" 微差（如 T28 TTFT 差 ~1.9%、T27 TPOT 差 0.06%）**是 run-to-run 噪声，不可解读为 nopad/pad 性能差异**。
- **正确读法**：上述 1a/1b 数字是 **EP 配置的整体 perf**（与 pad/nopad 无关）。
- 🟢 **三路径完整对比（同 commit 0526446 + 同 cudagraph 口径，2026-06-03 re-verify PASS）见 `REPRODUCE.md §6.2-纯TP-nopad`**：纯 TP nopad（inter=160）TTFT 594.0 / TPOT 14.2 / 70.6 tok/s、纯 TP pad（256）681.4 / 13.3 / 75.0、EP（inter=1280，本文路径，T54 fresh run）562.9 / 13.5 / 74.2。**本文 §1a/§1b 旧 EP 表与该三路径表口径档不同（gpu-mem/max-len/input），勿与三路径表的 EP 行混读**；纯 TP nopad/pad 与 EP 是不同 parallelism，三路径表内同口径方可比。bug/fix 细节见 `NOPAD_TP_HANDOFF.md`。

---

## 2. cudagraph 修复结论（teammate-26）

- **TP8 cudagraph 崩溃根因 = custom-allreduce IPC 不兼容**：`hipIpcGetMemHandle failed: invalid argument` @ `allocate_kv_cache` 的 `dist.barrier()`。**非 nopad 特异（pad 也崩）**；根因在 TP8 + cudagraph + custom-allreduce 三者，与 pad/nopad 无关（详 teammate-25）。
- **ATOM 原生 `simple_inference` / `EngineArgs.create_engine()` 默认全关 IPC-allreduce**（custom-allreduce OFF + quick-reduce NONE，走 RCCL）→ **去掉 `--enforce-eager` 即可用 cudagraph，无需改代码**。
- 仅 **custom-allreduce ON** 的路径（vllm-direct / 某些 plugin 配置）才会崩，需显式关 custom-allreduce。
- 实测（T27/T28）：cudagraph run 均 clean shutdown，GPU 回基线，**无 193GB/dead 占用、无需 host reset**（单序列小 KV teardown 快）。

---

## 3. EP 精度（teammate-24）— ⚠️ 弱验证，诚实记录

- ATOM 原生 e2e（**EP + cudagraph**）：4/4 prompt 输出**连贯、非 Qwen、有自然 eos**。
- **⚠️ 但 EP 精度未对 ground-truth / 参考做严格数值验证**：
  - T24 的 "pad_parity" 是 **EP-vs-EP**（同 inter=1280 路径自比）= **无意义对照**，不构成精度证明。
  - 无 EP-vs-reference（fp32 correct-ref / 已知正确输出）数值对照。
  - EP 本就**不受 ÷8 b_scale bug 影响**（inter=1280 对齐，nopad smalltile 路径根本不触发）。
- **结论**：EP 精度 = **"看着连贯"**，**非严格验证**。如需 EP 精度定论，须补 EP-vs-reference 数值对照（本 wave 未做）。

---

## 4. EP vs TP 区别 + 为何 EP 下 nopad/pad 无区别

| | **纯 TP** | **EP（`--enable-expert-parallel`）** |
|---|---|---|
| MoE 权重切分 | 沿 **TP** 切 inter 维 | 按 **expert** 切（inter 维**不切**）|
| inter_dim（gfx942 stepfun，full=1280） | tp=2→640 / tp=4→320 / tp=8→**160** | 恒 **1280**（≥256）|
| nopad smalltile 路径（inter=160, NPerBlock=32） | tp=8 **触发**（160→pad 256 或 nopad 160）| **不触发**（1280≥256，`ATOM_FP8_MOE_DISABLE_PAD` no-op）|
| ÷8 b_scale bug 暴露面 | nopad(inter=160)路径才暴露 | 不暴露（对齐路径）|

- **为何 EP nopad/pad 无区别**：EP 下 inter=1280 始终满足 `% 256 == 0`，pad/nopad 分支条件不成立 → 同一路径。
- **EP 引入时间线**（teammate-30）：ATOM 框架能力 `aaf83fa`（2025-08-12，default False）；本项目 serving `start_vllm_v5.sh`（2026-05-14）；本项目 e2e/nopad wave `dryrun_e2e_tp8.py` / V65B（**2026-05-28**，为修 `output_size=160 not divisible by block_n=128` 抄来 EP）。
- **副作用**：自 2026-05-28 起 e2e 改用 EP → inter=1280 → **nopad（inter=160 TP）路径不再被 e2e 覆盖**。REPRODUCE §6.2 的纯 TP perf（2026-05-09，EP 引入前）反而是唯一覆盖过 inter 分片（TP）路径的 anchor。

---

## 5. Caveat 汇总（必读）

1. 本文全部 perf = **EP（inter=1280）**，**不可**与 REPRODUCE §6.2 纯 TP anchor 混读。
2. T27/T28 "nopad vs pad" 标签在 EP 下是 no-op，差异 = 噪声，**非 nopad/pad 对比**。
3. EP 精度 = 连贯但**未严格对参考验证**。
4. **nopad（inter=160 TP）路径** 的 ÷8 bug / stage1-stage2 fix / 真实 perf（TP cudagraph）见 **NOPAD_TP_HANDOFF.md**（lead 整合中；当前细节散见 W8_resume/progress/teammate-19b（stage2 fix）/-22/-23（stage1）/-24（e2e）/-25（cudagraph）/-30（EP vs TP 考古））。本文不重复 nopad 细节。

---

## 6. EP 完整复现（脚本 + commit pin + 预期结果）

> 让人能照着把 EP perf / e2e 跑出来。脚本在 `details/scripts/ep_*.sh`（顶部变量化路径，按你的环境改 MODEL/ATOM_DIR/BENCH）。

### 6.1 Commit pin（3 仓 + 模型，照搬勿改）

> ℹ️ 本节 = **nopad/EP（W8）复现** pin；**pad/通用 tp8 历史复现**（pre-nopad，ATOM `969d564` + aiter `f06cdcca5` + CK `defd7ad29`）见 `REPRODUCE.md §3.1`。两套各服务各路径、均保留。

| 组件 | 版本 | 说明 |
|---|---|---|
| ATOM | `0526446`（branch `feat/step3p5-flash-support`，PR #641 + stepfun SWA per-layer kv-head fix）| nopad shared-expert 退化量化根治 + stage2 weight 预洗 + stepfun-Flash SWA per-layer kv-head workspace 修复（num_heads_kv 32→per-layer 4）。溯源：`0526446` = `e18b467`(PR #641 权威版) + stepfun SWA per-layer kv-head 补丁（原未提交，现已 commit 到 feat 顶）；更早本地 WIP `880dd46`（branch `w8-nopad-shared-expert-requant-fix`，未推 origin）已被取代 |
| aiter | `feat/step3p5-moe-swiglustep`，含 stage2 ÷8 fix（远端 `360ebdb66` 禁广播；本地等价 `57983d2f4`）| |
| ck（aiter submodule）| `e90ecddea` | nopad TP8 stage2 blockscale fix |
| 模型 | `stepfun-ai/Step-3.5-Flash-FP8` snapshot `6eebda59dd87ca5729648ec7cfed0becfceb273e` | ~90GB，fp8 blockscale |

### 6.2 env（两脚本一致）
```
HF_HOME / HF_HUB_CACHE = /workspace/hf_cache
TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR / TORCH_EXTENSIONS_DIR = /workspace/cache/*
HIP_VISIBLE_DEVICES = 0,1,2,3,4,5,6,7
unset AITER_QUICK_REDUCE_QUANTIZATION   # quick-reduce NONE/disabled
# custom-allreduce 默认 OFF（ATOM 原生 simple_inference/EngineArgs 默认全关 IPC-allreduce）
```
EP 关键开关：`--enable-expert-parallel`（EP）；cudagraph = **去掉** `--enforce-eager`；`--tensor-parallel-size 8`（TP8）。

### 6.3 两步命令
```bash
# 1) EP e2e correctness（cudagraph）
bash details/scripts/ep_e2e_cudagraph.sh

# 2) EP perf（prefill TTFT + decode TPOT，cudagraph）
bash details/scripts/ep_perf_bench.sh
```

### 6.4 预期结果（引本文已有数字）
- **e2e**（脚本 1）：Engine Core fully initialized + 44/44 shards + 4 段 Generated text，**4/4 连贯**（非 Qwen、有自然 eos）。
- **perf**（脚本 2）：prefill **TTFT ~560–571ms**；decode **TPOT ~13.5ms/tok**（短输出档 ~12.62ms）；decode throughput ~74 tok/s；cudagraph vs eager 加速 **~7.8–8.2×**。

### 6.5 🔴 R5 GPU 防泄漏注（必读）
- TP8 cudagraph 有显存泄漏史。**判泄漏看内存趋势**（`rocm-smi --showmeminfo vram`），**非** `ps -p`（进程没了不等于显存回收）。
- teardown **~3min settle** 回基线属正常，别误判残留。详 `details/TP8_THREE_BUGS.md` B3。

### 6.6 caveat
- **EP 下 nopad/pad = no-op 同路径**：inter=1280 ≥ 256，`ATOM_FP8_MOE_DISABLE_PAD` 设不设走同一路径（见 §1c / §4）。脚本里设它仅为显式，**不构成 nopad/pad 对比**。
- **EP 精度未对参考严格验证** = "看着连贯"（见 §3）。如需 EP 精度定论须补 EP-vs-reference 数值对照。

---

### 来源 progress（W8_resume/progress/）
- teammate-24：EP e2e correctness（连贯，弱验证）+ R5 无泄漏
- teammate-25：cudagraph IPC 根因调研（custom-allreduce，pad 也崩）
- teammate-26：cudagraph 修复（ATOM 原生默认关 IPC-allreduce → 去 enforce_eager 即可用）
- teammate-27：EP cudagraph decode TPOT ~12.62ms + ~7.8–8.2× vs eager
- teammate-28：EP cudagraph 长输入 TTFT ~560–571ms / TPOT ~13.5ms
- teammate-29：EP 下 nopad/pad = 同路径（`ATOM_FP8_MOE_DISABLE_PAD` no-op）坐实
- teammate-30：REPRODUCE §6.2 = 纯 TP（四重证据）+ EP 引入时间线
