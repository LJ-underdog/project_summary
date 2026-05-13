# Step-3.5-Flash FP8 性能验证结果 — gfx950 vs gfx942

**测试日期**：2026-04-29
**执行方式**：agent team（perf_gfx950_verified）
**脚本**：`perf_correctness_bench.py`（gfx950）/ `perf_bench.py`（gfx942）

---

> # 🔴 BASELINE 失效声明（2026-05-09 由 wave `tp2_verify_post_merge_wave` 加注）
>
> **本报告 §二（gfx942 参考数据）/ §三（H5 验证） / §四（对比分析） / §四-A / §四-B / §五（H1-H6 假设）/ §六（gfx950 内部对比）所引用的 "gfx942 perf_bench.py tp=2 TTFT=186ms / TPOT=5.2ms / tp=4 TTFT=110ms / TPOT=5.5ms" 已被实证为 Qwen/Qwen3-0.6B（dense, non-MoE）误归属，并非 stepfun-ai/Step-3.5-Flash-FP8 的实测值。**
>
> - 决定性证据：`details/perf/15_perf_tp2_tp4_tp8_eval/logs/tp2_run2_full.log:47,50` `Model load done: Qwen/Qwen3-0.6B`（含 tp4_run2_full.log:79 同源 8/8 行 + tp8_long_run2_full.log:144 同源 12/12 行 + tp8_launch_full.log:144 同源 6/6 行）
> - 实证 wave：L17c (`progress/teammate-L17c-baseline-audit.md`) → L18 (`teammate-L18-perf-rerun.md`) → L20 (`teammate-L20-perf-tp4-tp8.md`) → L24 (`teammate-L24-audit-data-residue.md`) → L26 (`teammate-L26-audit-perf-coverage.md`)
> - 真实 stepfun gfx942 三档 anchor（L18+L20 实测，2026-05-09）：tp=2 TTFT≈1665ms / tp=4 TTFT≈980ms（详见 `step35-flash-support/REPRODUCE.md` §6.2）
>
> **后果**：
> 1. §四 "gfx942 比 gfx950 快 2.3-3.5×" 与 §四 末尾结论 **"硬件代际反转" 论断的对比基础失效** — 比的不是同 model（gfx950 跑 stepfun MoE vs gfx942 跑 Qwen3-0.6B dense）。按真实 stepfun gfx942 1665ms 对照 gfx950 388ms，方向可能完全反转（gfx950 反而更快）。
> 2. §三 H5 "排除"论证基础失效 — "同脚本对比"前提不再成立。
> 3. §五 H1/H5/H6 verdict 全部失去依据；H6 "成立（强证据）" + 量化 87ms gap 的 anchor 数据失效。
> 4. §六 gfx950 内部对比的 "与 MEMORY 一致" 描述也是基于 Qwen3 误归属链。
>
> **本报告原结论文字保留作审计追溯**，但全部数据 / 比值 / 因果论断 **暂不可用**。需要重跑 gfx942 stepfun MoE perf 才能重验本报告所有跨硬件结论。
>
> 详见末尾 §附录-DISCLAIMER。

---

## 一、gfx950 实测结果（本次验证）

**平台**：8x MI350X (gfx950)
**commits**：ATOM `acff926d` / aiter `0f8164017` / CK `defd7ad29`
**模型**：`stepfun-ai/Step-3.5-Flash-FP8`（snapshots/6eebda59）
**测试参数**：input≈10213 tokens（target 10240 ±32）/ max_output=1024 / temperature=0 / batch=1 / runs=2（取 Run2 稳态）

| 配置 | GPU | TTFT (ms) | TPOT (ms/tok) | output_tokens | decode_throughput (tok/s) | CORRECTNESS |
|------|-----|-----------|---------------|---------------|--------------------------|-------------|
| FP8 tp=2 | 0,1 | **428.7** | **12.7** | 302 | 78.8 | PASS |
| FP8 tp=4 | 0,1,2,3 | **382.9** | **12.5** | 248 | 80.2 | PASS |

**数据来源**：
- tp=2：`logs/fp8_tp2_full.log`（teammate-1 progress L15-20）
- tp=4：`logs/fp8_tp4_full.log`（teammate-2 progress L16-18）

**正确性验证**：
- tp=2：first 80 chars = `"Hmm, the user has provided a repetitive text about a fox and dog sentence in bot"` → 非 BOS / 非 Qwen `<think>`，Step-3.5-Flash 风格确认
- tp=4：first 80 chars = `"Hmm, the user has provided a massive repetitive text block alternating between a"` → 同上

---

## 二、gfx942 参考数据（15_perf_tp2_tp4_tp8_eval）

> ⚠️ **本节全表 model 归属误标 — 实为 Qwen/Qwen3-0.6B 而非 stepfun-Flash-FP8**（顶部 🔴 BANNER 已说明）。表内数据 ~~strikethrough~~ 标识下游引用不再可信；待重跑 gfx942 stepfun MoE perf 才能重验。当前真实 stepfun gfx942 anchor 见 `step35-flash-support/REPRODUCE.md` §6.2。

**平台**：8x MI308X (gfx942)
**commits**：ATOM `acff926` / aiter `0f8164017`（含 fused_moe.py:881-886 dirty patch `run_1stage=False`）/ CK `defd7ad29`
**测试参数**：input=10265 tokens / max_output=1024 / temperature=0 / batch=1 / runs=2（取 Run2 稳态）

| 配置 | GPU | TTFT (ms) | TPOT (ms/tok) | output_tokens | decode_throughput (tok/s) |
|------|-----|-----------|---------------|---------------|--------------------------|
| FP8 tp=2 ⚠️ | 0,1 | ~~**186**~~ | ~~**5.2**~~ | ~~317~~ | ~~190.7~~ |
| FP8 tp=4 ⚠️ | 0,1,2,3 | ~~**110**~~ | ~~**5.5**~~ | ~~416~~ | ~~183.4~~ |

⚠️ 上表两行数据 = Qwen/Qwen3-0.6B 误归属（L17c/L19d 实证），并非 stepfun-Flash-FP8；保留作审计追溯，下游引用不再可信。

**数据来源（已被 audit 翻转）**：`15_perf_tp2_tp4_tp8_eval/progress/perf-t1.md`（tp=2）、`perf-t2.md`（tp=4）— 这两份 progress 顶部已被 L19b/L19d 加 🔴 误归属警告

---

## 三、H5 验证：perf_bench.py 在 gfx950 上重跑结果

> ⚠️ **本章 H5 论证基础失效（顶部 🔴 BANNER）**：本节 "同脚本对比 gfx950 vs gfx942" 的 gfx942 列数值（186 / 110ms）实为 Qwen3-0.6B，并非 stepfun-Flash-FP8。"消除脚本差异" 前提下两边跑的并非同 model，整章 "H5 排除" 论断暂不可用。下方表格 / 比值 / 排除结论 **均待重跑 gfx942 stepfun MoE perf 才能重验**。

> 目的：消除脚本差异，确认 gfx950 vs gfx942 性能差距是真实的

**gfx950（perf_bench.py，与 gfx942 完全相同脚本 + 参数）**：

| 配置 | TTFT (ms) | TPOT (ms/tok) | decode_throughput (tok/s) |
|------|-----------|---------------|--------------------------|
| FP8 tp=2 (GPU 0,1) | **388** | **12.28** | 81.4 |
| FP8 tp=4 (GPU 0,1,2,3) | **241** | **12.50** | 80.0 |

**数据来源**：teammate-3 (#301) `logs/perf_bench_tp2.log`，teammate-4 (#302) `logs/perf_bench_tp4.log`

**H5 结论（脚本差异）**：

| 维度 | gfx950 两脚本差异 | gfx950 vs gfx942（同脚本）⚠️ Qwen3 误归属 |
|------|-----------------|--------------------------|
| TPOT tp=2 | 12.7 → 12.28ms（**-3%**，可忽略）| ~~12.28 vs 5.2ms = **2.36× 慢**~~ |
| TPOT tp=4 | 12.5 → 12.50ms（**≈0%**，无差异）| ~~12.50 vs 5.5ms = **2.27× 慢**~~ |
| TTFT tp=2 | 428.7 → 388ms（**-9.5%**，小幅）| ~~388 vs 186ms = **2.09× 慢**~~ |
| TTFT tp=4 | 382.9 → 241ms（**-37%**，较大）| ~~241 vs 110ms = **2.19× 慢**~~ |

⚠️ 右列 "gfx950 vs gfx942" 全部 strikethrough：gfx942 列分母 = Qwen3-0.6B 误归属，比值无意义。gfx950 内部脚本差异部分（左列）仍有效。

~~**H5 排除**：TPOT 几乎无脚本差异，TTFT 脚本差异最大 37%，但同脚本下 gfx950 仍比 gfx942 慢 2× 以上。脚本不是主因。~~

⚠️ **H5 排除论断暂不可用**：分母错位 → "同脚本下 gfx950 比 gfx942 慢 2×" 不成立；H5 verdict 待重跑 gfx942 stepfun MoE perf 才能重验。

---

## 四、对比分析

> ⚠️ **本章对比基础失效（顶部 🔴 BANNER）**：下表 gfx942 列数据 = Qwen/Qwen3-0.6B 误归属（dense, non-MoE），与 gfx950 跑 stepfun-Flash-FP8 MoE 路径**不是同 model 对比**。"2.3-3.5× 慢" 比值 + "硬件代际反转" 结论 **均不成立**；按真实 gfx942 stepfun TTFT≈1665ms（L18 实测）对照 gfx950 388ms，方向可能完全反转。
>
> 表格保留作审计追溯，所有比值与下方结论 strikethrough，待重跑 gfx942 stepfun MoE perf 才能重验。

| 指标 | FP8 tp=2 gfx950 | FP8 tp=2 gfx942 ⚠️ Qwen3 误归属 | 比值（gfx950/gfx942）|
|------|-----------------|-----------------|---------------------|
| TTFT | 428.7ms | ~~186ms~~ | ~~**2.3× 慢**~~ |
| TPOT | 12.7ms | ~~5.2ms~~ | ~~**2.4× 慢**~~ |
| decode_throughput | 78.8 tok/s | ~~190.7 tok/s~~ | ~~**2.4× 低**~~ |

| 指标 | FP8 tp=4 gfx950 | FP8 tp=4 gfx942 ⚠️ Qwen3 误归属 | 比值（gfx950/gfx942）|
|------|-----------------|-----------------|---------------------|
| TTFT | 382.9ms | ~~110ms~~ | ~~**3.5× 慢**~~ |
| TPOT | 12.5ms | ~~5.5ms~~ | ~~**2.3× 慢**~~ |
| decode_throughput | 80.2 tok/s | ~~183.4 tok/s~~ | ~~**2.3× 低**~~ |

~~**结论：gfx942（MI308X）在该基准下比 gfx950（MI350X）快 2-3.5 倍，与硬件代际预期相反。**~~

⚠️ **上述结论暂不可用**：分母 gfx942 数据为 Qwen3 误归属，"快 2-3.5 倍 / 硬件代际反转" 论断失去依据。需重跑 gfx942 stepfun MoE perf 才能重验。

---

## 五、可能根因（假设，待验证）

> ⚠️ **本章 H 体系 verdict 全部失去依据（顶部 🔴 BANNER）**：H1 / H5 的 "已排除" + H6 的 "成立（强证据）" 全部建立在 "gfx942 baseline = stepfun-Flash-FP8 186/110ms" 前提下。该前提已被 L17c 实证为 Qwen3-0.6B 误归属 → 整套 H1-H6 verdict 暂不可用，需重跑 gfx942 stepfun MoE perf 后**重新审视所有 H 假设的 verdict**。
>
> 下表保留作审计追溯，verdict 列 strikethrough。

以下为 Lead 分析，所有条目均标注为【未验证假设】，需后续实验确认：

| ID | 状态 ⚠️ 失效 | 假设 | 验证方法 / 结论 ⚠️ 论证基础失效 |
|----|------|------|---------------|
| H1 | ~~✅ **已排除**~~ ⚠️ unreliable | ~~aiter dirty patch 差异（run_1stage=False）~~ | ~~#403/#404 实测：patch 生效（日志确认走 2-stage），但 TTFT 变化 tp=2 -1.0% / tp=4 +2.9%，均在噪声范围内。MoE 1-stage vs 2-stage 不是 prefill 瓶颈。~~ ⚠️ 排除论断需重审 |
| **H6** | ~~🔴 **成立（强证据）**~~ ⚠️ INVALIDATED-PENDING-RECOMPUTE | **bf16_tuned_gemm.csv 对 Step-3.5-Flash 形状覆盖为零** | ~~见下方 §四 H6 详细分析~~ ⚠️ "解释 100~200ms TTFT gap" 量级推算 anchor 已失效 |
| H2 | ⬜ **未验证** | **gfx950 CK kernel 调优不足**（MoE/attention kernel tuning） | 检查 `aiter/configs/tuned_fmoe.csv` 中 gfx950 条目数量；对比 gfx942 覆盖度 |
| H3 | ⬜ **未验证** | **ATOM/JIT cache 差异** | 在 gfx950 不清 cache 重跑，对比结果 |
| H4 | ⬜ **未验证** | **GPU 硬件状态** | `rocm-smi` 检查 GPU 0-3 健康；对比单卡 gemm 基准 |
| H5 | ~~✅ **已排除**~~ ⚠️ unreliable | ~~测试脚本差异~~ | ~~#301/#302：同脚本 gfx950 仍比 gfx942 慢 2.1-2.4×~~ ⚠️ "同脚本对比" gfx942 端为 Qwen3 → 排除论断失效 |

---

## 六、gfx950 内部 tp=2 vs tp=4 对比

> ⚠️ 本节 gfx950 内部对比数值本身（428.7 / 382.9 / 12.7 / 12.5ms）由 §一 实测得到，**仍有效**；但末句 "与 MEMORY 一致" 引用的 MEMORY "prefill → tp=4 快, decode → tp=2 快" 链路源自已被 L17c/L19b 翻转的旧 perf wave，需重新审视。

| 指标 | tp=2 | tp=4 | 结论 |
|------|------|------|------|
| TTFT | 428.7ms | 382.9ms | tp=4 快 **11%**（更多 GPU 加速 prefill） |
| TPOT | 12.7ms | 12.5ms | 几乎持平（decode 瓶颈不在 TP 通信） |

~~与 MEMORY 中"prefill → tp=4 快，decode → tp=2 快"的结论一致。~~

⚠️ MEMORY 引用链已被 L17c/L19b/L19d audit 翻转，"prefill → tp=4 快, decode → tp=2 快" 论断暂不可用。

---

## 四、根因全景（#701/#702 确认，2026-04-29）

> ⚠️ **本章 H6 量化推导失去 anchor（顶部 🔴 BANNER）**：§"TTFT gap 解释力"（line 156-168 附近）所引 "FP8 perf_bench tp=2 TTFT = 388ms / gfx942 baseline 显著更低，TTFT gap 100~200ms" 中 "gfx942 baseline" 实为 Qwen3-0.6B → "BF16 miss 解释 ~87ms gap" 推算 anchor 失效。代码读取部分（46.1% FLOPs / 62 次 miss / aiter dispatch 路径分析）**仍有效**，但 "解释 gap 量级" 论断 strikethrough。

> 所有结论来自代码读取（文件+行号）+ 日志实测，非推断

### 核心结论：MoE 始终走 CK kernel，bf16_tuned_gemm miss 影响的是另外 46% 的 FLOPs

**关键质疑已回答**：无论 BF16 还是 FP8 模式，routed MoE experts **始终** 走 CK fused_moe，与 bf16_tuned_gemm.csv 完全无关。

代码证据：
- BF16：`atom/model_ops/moe.py:581` `return fused_moe(...)` (UnquantizedFusedMoEMethod) → `aiter.fmoe`
- FP8：`atom/model_ops/moe.py:1873` `torch.ops.aiter.rocm_aiter_fused_moe(...)` → `aiter.fmoe_fp8_blockscale_g1u1`
- 两路均 import `from aiter.fused_moe import fused_moe`（moe.py:12），无任何 `tgemm.mm` 调用

**bf16_tuned_gemm miss 实际影响的层**（全部通过 `atom/model_ops/linear.py:393` 的 `tgemm.mm`）：

| 层类型 | 是否经过 bf16_tuned_gemm | FLOPs | 占比 |
|--------|--------------------------|-------|------|
| Attention QKV+O proj（45 层）| ✅ 是（BF16，modules_to_not_convert）| 69.73 TF | 34.6% |
| Dense MLP（layer 0-2，3 层）| ✅ 是（BF16）| 8.52 TF | 4.2% |
| Shared expert（42 MoE 层）| ✅ 是（BF16）| 13.56 TF | 6.7% |
| Router gate（42 MoE 层）| ✅ 是（BF16）| 1.02 TF | 0.5% |
| **BF16 tgemm miss 小计** | → torch.mm fallback | **92.83 TF** | **46.1%** |
| MoE routed experts（42 层）| ❌ 否，直接走 CK fused_moe | 108.47 TF | 53.9% |
| **总 GEMM** | | **201.29 TF** | 100% |

**62 次 miss 来源验证**（h1_tp2_full.log，AITER_LOG_TUNED_CONFIG=1，FP8 模型）：

| (N, K) | 来源层 | 来自 MoE？ |
|--------|--------|-----------|
| (5120, 4096) | QKV proj（64+2×8 heads × 128 / tp=2）| ❌ attn |
| (4096, 4096) | O proj | ❌ attn |
| (7168, 4096) | QKV proj sliding window（96+2×8 heads × 128 / tp=2）| ❌ attn |
| (4096, 6144) | O proj sliding window | ❌ attn |
| (11264, 4096) | Dense MLP gate_up（intermediate=11264）| ❌ dense |
| (4096, 5632) | Dense MLP down（K=11264/tp=2）| ❌ dense |
| (1280, 4096) | Shared expert gate_up | ❌ shared |
| (4096, 640) | Shared expert down | ❌ shared |
| (32, 4096) / (48, 4096) | g_proj head-wise gate | ❌ attn |
| (64448, 4096) | lm_head（vocab=128896/tp=2）| ❌ head |
| **MoE routed expert shape** | — | **无任何一条** |

### TTFT gap 解释力（#702 计算）

模型参数：hidden=4096，num_heads=64，kv_heads=8，head_dim=128，intermediate=11264，moe_inter=1280，vocab=128896

在 tp=2、M=10262 下（MI350X BF16 peak ≈ 2500 TFLOPS）：

| 路径 | per-GPU FLOPs | tuned kernel（50% eff）| torch.mm（15% eff）| Gap |
|------|--------------|----------------------|--------------------|-----|
| BF16 miss（tp=2）| 46.41 TFLOPs | 37 ms | 124 ms | **+87 ms** |
| BF16 miss（tp=4）| 23.21 TFLOPs | 19 ms | 62 ms | **+43 ms** |

~~→ **BF16 tuned_gemm miss 在 tp=2 单独可贡献 ~87 ms TTFT 额外延迟**，足以解释 gfx950 vs gfx942 的 TTFT gap 量级。即使 torch.mm 仅比 tuned 慢 1.5×，也会贡献 20-30 ms。~~

⚠️ "解释 gfx950 vs gfx942 TTFT gap" 量级推算的 anchor "gfx942 baseline" 实为 Qwen3-0.6B → 87ms / 20-30ms 推算结论暂不可用。BF16 miss 路径占 46.1% FLOPs 这一独立观察仍有效，但 "解释 gap" 论断需重新计算（按真实 stepfun gfx942 1665ms anchor 推 gap 方向甚至可能反转）。

---

## 四-A、H6 根因分析 — bf16_tuned_gemm.csv 覆盖率严重不足

> 来源：teammate-7 #501 调查（2026-04-29）
> 代码：`aiter/tuned_gemm.py:38-193`，`aiter/jit/core.py:178-296`

### 证据链

**1. 加载机制**：runtime CSV = `/tmp/aiter_configs/bf16_tuned_gemm.csv`（base + model_configs/* 合并）

**2. gfx950 tuning 来源**：

| 文件 | gfx950 条目 |
|------|------------|
| `aiter/configs/bf16_tuned_gemm.csv`（base） | **0**（完全为空）|
| `model_configs/glm5_bf16_tuned_gemm.csv` | 71 |
| `model_configs/llama*`, `dsv3*`, `qwen32B*` 等 | 合计 ~708 |
| **Step-3.5-Flash 专属文件** | **不存在** |

→ gfx950 的所有 tuning 来自其他模型；**没有针对 Step-3.5-Flash (hidden=4096) 的专属 tuning**

**3. M 值覆盖**：M ∈ {1…512（步长8）, 1024, 2048, 4096, 8192, **gap**, 16384, 32768}
→ **M=10262（实际 prefill tokens）落入 8192~16384 的空隙，无任何 tuning 条目**

**4. (N, K) 覆盖**：Step-3.5-Flash prefill 用到的关键形状：

| (N, K) | 用途 | gfx950 有 tuning？ |
|--------|------|-------------------|
| (4096, 4096) | attn O proj | ❌ 无 |
| (11264, 4096) | QKV proj（合并）| ❌ 无 |
| (7168, 4096) | MLP gate/up | ❌ 无 |
| (4096, 5632) | MLP down | ❌ 无 |
| (64448, 4096) | lm_head | ❌ 无 |
| (4096, 2048) | attn proj (tp=4) | ✓ 有，但 M 只到 512，M=10262 不命中 |

**5. 实测 miss 数**（h1 验证日志，`using torch solution:0`）：
- tp=2：**62 次**（全部 prefill GEMM fallback → torch.mm）
- tp=4：**120 次**（更多 TP 切分形状，miss 更多）

**6. decode TPOT 也受影响**：(N,K)=(4096,4096) 等形状在 gfx950 tuning 中完全不存在，M=1 的 decode 也同样 miss → **TTFT 和 TPOT 均 2× 慢均可由此解释**

### 结论（#801 修正）

**gfx950 上 Step-3.5-Flash 的所有非 MoE 的 bf16 GEMM（attention proj、MLP、lm_head）均走 torch.mm fallback。** 这对 gfx950 自身性能有影响，但 **gfx942 在本 aiter 仓库中同样没有 Step-3.5-Flash 的 BF16 tuning**：

- 所有 `*bf16_tuned_gemm.csv` 中 gfx942 条目数 = **0**（gfx942 完全没有 BF16 tuning）
- gfx942 独有的条目仅在 `dsv3_a8w8_bpreshuffle`（K=7168）和 batched GEMM，**不覆盖 Step-3.5-Flash 的 (N, K=4096) 形状**

→ **gfx942 vs gfx950 的 2× 性能差距，不能由"gfx942 有 BF16 tuning 而 gfx950 没有"来解释**

gfx942 机器可能在其本地 `/workspace/aiter/aiter/configs/` 下有额外的 Step-3.5-Flash 专属 tuning（gfx950 的 aiter repo 中没有），或者 gfx942（MI308X）在相同 torch.mm 路径下执行速度本身就比 gfx950（MI350X）快（硬件内存带宽/计算特性差异）。需要 SSH 访问 gfx942 确认其实际 CSV 文件内容。

### Gap-1 修复路径

创建 `aiter/configs/model_configs/step3p5_bf16_tuned_gemm.csv`（gfx950 专属）：
- 用 `AITER_TUNE_GEMM=1` dump 缺失形状
- Offline tune：(N,K) ∈ {(11264,4096),(4096,4096),(7168,4096),(4096,5632),(1280,4096),(64448,4096)} × M ∈ {1,128,256,512,1024,2048,4096,8192,10262,16384}
- 预期效果：BF16 GEMM miss 全部消除，TTFT 大幅下降

---

## 四-B、FP8 fmoe tuning Gap（#601 新发现）

> 来源：teammate-8 #601 调查（2026-04-29）
> 代码：`aiter/fused_moe.py:780-867`，`aiter/jit/core.py:70-160`

### FP8 fmoe dispatch key（13 元 tuple）

`fused_moe.py:802-867` 按 `(cu_num, token, model_dim, inter_dim, expert, topk, activation, dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1, doweight_stage1)` 查 tuned_fmoe.csv。

### Step-3.5-Flash 所需的 4 个 unique key

| key（简化）| tp | 状态 |
|------------|-----|------|
| (256, 16384, 4096, **640**, **289**, 9, Silu, bf16, e4m3fn, e4m3fn, per_1x128, True, False) | tp=2 prefill | **MISS** |
| (256, 16384, 4096, **640**, **288**, 8, **SwigluStep**, bf16, e4m3fn, e4m3fn, per_1x128, True, False) | tp=2 prefill layer 43-44 | **MISS** |
| (256, 1, 4096, 640, 289/288, ...) | tp=2 decode | **MISS** |
| (256, *, 4096, **384**, 288/289, ...) | tp=4（inter padded 320→384）| inter 维匹配但 expert/activation/topk 全错 → **MISS** |

### 覆盖缺口

| 维度 | Step-3.5-Flash 需要 | tuned_fmoe.csv 有 | 状态 |
|------|---------------------|-------------------|------|
| inter_dim=640 (tp=2) | 必须 | 无 | **MISS** |
| expert=288 / 289 | 必须 | 最近 257/513 | **MISS** |
| SwigluStep activation | layer 43-44 必须 | 无（仅 Silu/Gelu）| **MISS** |
| e4m3fn + per_1x128 组合 | 必须 | 部分（fnuz format，格式不同）| **MISS** |

→ **FP8 routed-expert：0 命中 tuning，全部 fallback 到 default heuristics**

### 关键区别：gfx942 同样 0 命中

tuned_fmoe.csv 中 gfx942 (cu=80) 条目也不包含上述 4 个 Step-3.5-Flash key tuple。
**→ FP8 fmoe tuning gap 对 gfx950 vs gfx942 性能差异无解释力**（两者都未 tuned）。
这是一个 gfx950 独立于 gfx942 差距之外的额外优化机会。

### Gap-2 修复路径

用 `gemm_moe_tune.py` 为 gfx950 补 4 个 Step-3.5-Flash fmoe key：
```bash
# 写入：aiter/configs/model_configs/step35flash_a8w8_blockscale_tuned_fmoe.csv
# 优先：tp=2（inter_dim=640, expert=289, Silu） + SwigluStep（layer 43-44）
# 次之：tp=4（inter_dim=384, expert=288/289）
```
预期效果：MoE prefill/decode 各 layer 走调优 kernel，TTFT 和 TPOT 进一步下降。

> **跨 wave cross-link（2026-05-13 反向加注 / 硬件 axis 澄清）**：相关 wave 详见 [`details/perf/20_fp8_fmoe_tuning_wave2/RESULTS.md`](../20_fp8_fmoe_tuning_wave2/RESULTS.md)。**硬件覆盖差异**：本节 §四-B 的 fix path 是为 **gfx950** 补 csv（L317）；wave2 测的是 **gfx942** 上 stepfun-Flash-FP8 的 OPT-1 axis（aiter `tuned_fmoe.csv` 加 stepfun-specific entry），结论 = gfx942 上全 axis 证伪 + tp=2 multi-prompt 反向退化。**未触及** §四-B 的 gfx950 fix path 推荐 — gfx950 上是否同样无效未实证。Disclaimer：本反向加注由 wave2 doc-impl 时追加，不修改 16 wave 原结论。

---

## 七、遗留问题与建议

### 已排除
- **H1**（2026-04-29）：run_1stage=False patch 无效，MoE kernel 选择不是瓶颈
- **H5**（2026-04-29）：脚本差异仅解释 TTFT ≤37%，同脚本仍差 2×

### aiter repo 全量盘点结论（#801，2026-04-29）

70 个 CSV 文件完整扫描，关键发现：

| 类别 | gfx950 | gfx942 | Step-3.5-Flash 覆盖 |
|------|--------|--------|---------------------|
| BF16 tuned_gemm（所有文件）| 779 条（其他模型形状）| **0 条** | ❌ 两机均无 |
| FP8 fmoe tuned（所有文件）| inter∈{192,256,384,512,1024}，expert∈{128,256} | inter∈{192,256,384,512,1536,2048,4096}，expert≤257 | ❌ 两机均无 inter=640/expert=288-289 |
| FP8 dense a8w8_blockscale | 覆盖 dsv3/qwen3/glm5 形状 | 仅 dsv3 rowwise（1403 条） | ❌ 两机均无 Step-3.5-Flash 形状 |

**关键修正**：之前假设"gfx942 有 BF16 tuning 而 gfx950 没有"是错误的——gfx942 在 BF16 tuning 上条目为 0，两机在 Step-3.5-Flash 形状上处于相同的未 tuned 状态。

### 根因重新评估与修复优先级

| 优先级 | 行动 | 说明 |
|--------|------|------|
| 🔴 P0 | **SSH 访问 gfx942，查看其 `/workspace/aiter/aiter/configs/`** | 确认 gfx942 是否有本地 Step-3.5-Flash 专属 CSV；若有则可直接复用 |
| 🔴 P1 | 为两机补 `step3p5_bf16_tuned_gemm.csv` | BF16 attn/dense/shared/lm_head 共 46% FLOPs 均走 torch.mm |
| 🟡 P2 | 补 `step35flash_a8w8_blockscale_tuned_fmoe.csv` | FP8 fmoe 0 命中（两机共有）|
| ⬜ P3 | 若 P0 确认两机 tuning 状态相同，则 gfx942 更快可能源于硬件差异（MI308X vs MI350X 在某些算子上的特性），需进一步 profiling | — |

### 其他注意事项
- **MEMORY.md 历史数据校正**：MEMORY 中记录的 FP8 tp=4 TTFT=86ms 是短 prompt（~20 tokens）场景，与本次 10k 输入 382.9ms 不可直接比较，两者均正确但场景不同
- **Run 抖动**：tp=4 Run1/Run2 TTFT 差异较大，建议后续测试增加 runs=3 取中位数

---

## 附录-DISCLAIMER：baseline 翻转事件链（2026-05-09 加注）

| Wave / Teammate | 文件 | 关键发现 |
|---|---|---|
| L4-data-review | `tp2_verify_post_merge_wave/baseline_data_audit.md` | 首次 raise N1 警告 "PERF baseline 跑错 model" |
| **L17c-baseline-audit** | `tp2_verify_post_merge_wave/progress/teammate-L17c-baseline-audit.md` | 决定性实证：raw log `tp2_run2_full.log:47,50` `Model load done: Qwen/Qwen3-0.6B`；186/110ms = Qwen3-0.6B dense 而非 stepfun-Flash-FP8 |
| L18-perf-rerun | `tp2_verify_post_merge_wave/progress/teammate-L18-perf-rerun.md` | 实测真实 stepfun gfx942 tp=2 ≈ 1665ms（与原 186ms 假数差距 ~9× → 方向可能反转）|
| L20-perf-tp4-tp8 | `tp2_verify_post_merge_wave/progress/teammate-L20-perf-tp4-tp8.md` | 实测真实 stepfun gfx942 tp=4 ≈ 980ms |
| L19b/L19d/L19e/L22 | `15_perf_tp2_tp4_tp8_eval/PERF_REPORT.md` 等 7 文件 | 已加 🔴 误归属 banner + 真实数据写入 `step35-flash-support/REPRODUCE.md` §6.2 / §7.13 KNOWN_FACT |
| **L24-audit-data-residue** | `tp2_verify_post_merge_wave/progress/teammate-L24-audit-data-residue.md` | 全 repo grep 找出本 16_perf_gfx950_verified 目录 8 处残留误归属（精确 file:line 表）|
| **L26-audit-perf-coverage** | `tp2_verify_post_merge_wave/progress/teammate-L26-audit-perf-coverage.md` | 标定本 RESULTS.md §二/§四/§五-六 全篇基于已死分母 → P0 |
| **L27-fix-gfx950**（本次）| 本文件 + 同目录其他文件 | 仅基于已有证据加 disclaimer banner / strikethrough，**未重跑实验**；未删除原结论文字（保审计追溯）|

**deferred path（不在本 wave 范围）**：重跑 gfx942 stepfun-Flash-FP8 perf（tp=2 / tp=4），重新计算 gfx950 vs gfx942 真实比值，重审 H1 / H5 / H6 verdict。

**真实 stepfun gfx942 anchor 当前 SoT**：`step35-flash-support/REPRODUCE.md` §6.2（tp=2/4/8 三档 TTFT/TPOT/decode_throughput/engine_init，由 wave `tp2_verify_post_merge_wave` L18+L20 实测产出于 2026-05-09）。

---

## 八、附：测试命令（可在 gfx942 复用）

```bash
# FP8 tp=2（gfx942 上替换 CUDA_VISIBLE_DEVICES 和 MODEL_PATH）
cd /tmp && \
CUDA_VISIBLE_DEVICES=0,1 \
HF_HOME=/workspace/hf_cache \
AITER_LOG_LEVEL=WARNING \
/opt/venv/bin/python \
  /path/to/perf_correctness_bench.py \
  --tp 2 \
  --model stepfun-ai/Step-3.5-Flash-FP8 \
  --input-tokens 10240 \
  --output-tokens 1024 \
  --runs 2 \
  --log-file /path/to/logs/fp8_tp2_gfx942.log \
  2>&1 | tee /path/to/logs/fp8_tp2_gfx942_full.log

# FP8 tp=4
cd /tmp && \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
HF_HOME=/workspace/hf_cache \
AITER_LOG_LEVEL=WARNING \
/opt/venv/bin/python \
  /path/to/perf_correctness_bench.py \
  --tp 4 \
  --model stepfun-ai/Step-3.5-Flash-FP8 \
  --input-tokens 10240 \
  --output-tokens 1024 \
  --runs 2 \
  --log-file /path/to/logs/fp8_tp4_gfx942.log \
  2>&1 | tee /path/to/logs/fp8_tp4_gfx942_full.log
```
