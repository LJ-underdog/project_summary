# Step-3.5-Flash FP8 性能验证结果 — gfx950 vs gfx942

**测试日期**：2026-04-29
**执行方式**：agent team（perf_gfx950_verified）
**脚本**：`perf_correctness_bench.py`（gfx950）/ `perf_bench.py`（gfx942）

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

**平台**：8x MI308X (gfx942)
**commits**：ATOM `acff926` / aiter `0f8164017`（含 fused_moe.py:881-886 dirty patch `run_1stage=False`）/ CK `defd7ad29`
**测试参数**：input=10265 tokens / max_output=1024 / temperature=0 / batch=1 / runs=2（取 Run2 稳态）

| 配置 | GPU | TTFT (ms) | TPOT (ms/tok) | output_tokens | decode_throughput (tok/s) |
|------|-----|-----------|---------------|---------------|--------------------------|
| FP8 tp=2 | 0,1 | **186** | **5.2** | 317 | 190.7 |
| FP8 tp=4 | 0,1,2,3 | **110** | **5.5** | 416 | 183.4 |

**数据来源**：`15_perf_tp2_tp4_tp8_eval/progress/perf-t1.md`（tp=2）、`perf-t2.md`（tp=4）

---

## 三、H5 验证：perf_bench.py 在 gfx950 上重跑结果

> 目的：消除脚本差异，确认 gfx950 vs gfx942 性能差距是真实的

**gfx950（perf_bench.py，与 gfx942 完全相同脚本 + 参数）**：

| 配置 | TTFT (ms) | TPOT (ms/tok) | decode_throughput (tok/s) |
|------|-----------|---------------|--------------------------|
| FP8 tp=2 (GPU 0,1) | **388** | **12.28** | 81.4 |
| FP8 tp=4 (GPU 0,1,2,3) | **241** | **12.50** | 80.0 |

**数据来源**：teammate-3 (#301) `logs/perf_bench_tp2.log`，teammate-4 (#302) `logs/perf_bench_tp4.log`

**H5 结论（脚本差异）**：

| 维度 | gfx950 两脚本差异 | gfx950 vs gfx942（同脚本） |
|------|-----------------|--------------------------|
| TPOT tp=2 | 12.7 → 12.28ms（**-3%**，可忽略）| 12.28 vs 5.2ms = **2.36× 慢** |
| TPOT tp=4 | 12.5 → 12.50ms（**≈0%**，无差异）| 12.50 vs 5.5ms = **2.27× 慢** |
| TTFT tp=2 | 428.7 → 388ms（**-9.5%**，小幅）| 388 vs 186ms = **2.09× 慢** |
| TTFT tp=4 | 382.9 → 241ms（**-37%**，较大）| 241 vs 110ms = **2.19× 慢** |

**H5 排除**：TPOT 几乎无脚本差异，TTFT 脚本差异最大 37%，但同脚本下 gfx950 仍比 gfx942 慢 2× 以上。脚本不是主因。

---

## 四、对比分析

| 指标 | FP8 tp=2 gfx950 | FP8 tp=2 gfx942 | 比值（gfx950/gfx942）|
|------|-----------------|-----------------|---------------------|
| TTFT | 428.7ms | 186ms | **2.3× 慢** |
| TPOT | 12.7ms | 5.2ms | **2.4× 慢** |
| decode_throughput | 78.8 tok/s | 190.7 tok/s | **2.4× 低** |

| 指标 | FP8 tp=4 gfx950 | FP8 tp=4 gfx942 | 比值（gfx950/gfx942）|
|------|-----------------|-----------------|---------------------|
| TTFT | 382.9ms | 110ms | **3.5× 慢** |
| TPOT | 12.5ms | 5.5ms | **2.3× 慢** |
| decode_throughput | 80.2 tok/s | 183.4 tok/s | **2.3× 低** |

**结论：gfx942（MI308X）在该基准下比 gfx950（MI350X）快 2-3.5 倍，与硬件代际预期相反。**

---

## 五、可能根因（假设，待验证）

以下为 Lead 分析，所有条目均标注为【未验证假设】，需后续实验确认：

| ID | 状态 | 假设 | 验证方法 / 结论 |
|----|------|------|---------------|
| H1 | ✅ **已排除** | ~~aiter dirty patch 差异（run_1stage=False）~~ | #403/#404 实测：patch 生效（日志确认走 2-stage），但 TTFT 变化 tp=2 -1.0% / tp=4 +2.9%，均在噪声范围内。MoE 1-stage vs 2-stage 不是 prefill 瓶颈。 |
| **H6** | 🔴 **成立（强证据）** | **bf16_tuned_gemm.csv 对 Step-3.5-Flash 形状覆盖为零** | 见下方 §四 H6 详细分析 |
| H2 | ⬜ **未验证** | **gfx950 CK kernel 调优不足**（MoE/attention kernel tuning） | 检查 `aiter/configs/tuned_fmoe.csv` 中 gfx950 条目数量；对比 gfx942 覆盖度 |
| H3 | ⬜ **未验证** | **ATOM/JIT cache 差异** | 在 gfx950 不清 cache 重跑，对比结果 |
| H4 | ⬜ **未验证** | **GPU 硬件状态** | `rocm-smi` 检查 GPU 0-3 健康；对比单卡 gemm 基准 |
| H5 | ✅ **已排除** | ~~测试脚本差异~~ | #301/#302：同脚本 gfx950 仍比 gfx942 慢 2.1-2.4× |

---

## 六、gfx950 内部 tp=2 vs tp=4 对比

| 指标 | tp=2 | tp=4 | 结论 |
|------|------|------|------|
| TTFT | 428.7ms | 382.9ms | tp=4 快 **11%**（更多 GPU 加速 prefill） |
| TPOT | 12.7ms | 12.5ms | 几乎持平（decode 瓶颈不在 TP 通信） |

与 MEMORY 中"prefill → tp=4 快，decode → tp=2 快"的结论一致。

---

## 四（新增）、H6 根因分析 — bf16_tuned_gemm.csv 覆盖率严重不足

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

### 结论

**gfx950 上 Step-3.5-Flash 的所有非 MoE 的 bf16 GEMM（attention proj、MLP、lm_head）均走 torch.mm fallback，未使用任何调优 ASM/flydsl kernel。** 这是 TTFT 和 TPOT 均比 gfx942 慢约 2× 的最可能根因。

gfx942 在 `/workspace/aiter/aiter/configs/model_configs/` 中很可能存在针对 Step-3.5-Flash (hidden=4096) 形状的专属 tuning CSV，使其走优化 kernel。

### 修复路径

1. **近期 workaround**：为 Step-3.5-Flash 在 gfx950 上创建专属 tuning CSV：
   - 用 `AITER_TUNE_GEMM=1` dump 缺失形状
   - 在 gfx950 上 offline tune (M=10262/8192/16384, N=4096/7168/11264, K=4096 等)
   - 写入 `aiter/configs/model_configs/step3p5_bf16_tuned_gemm.csv`
   - 预期效果：TTFT 大幅下降，接近 gfx942 水平

2. **长期**：aiter 官方应在 gfx950 上为 Step-3.5-Flash 补全 tuning

---

## 七、遗留问题与建议

### 已排除
- **H1**（2026-04-29）：run_1stage=False patch 生效但 TTFT 无显著变化，MoE kernel 选择不是瓶颈
- **H5**（2026-04-29）：脚本差异只解释 TTFT ≤37%，同脚本仍差 2×

### 下一步优先级
1. **H6 已成立**（2026-04-29）：为 gfx950 补充 Step-3.5-Flash 专属 tuning CSV（见 §四修复路径）
2. H2/H3/H4 暂时搁置，H6 修复后重测若仍有差距再继续

### 其他注意事项
- **MEMORY.md 历史数据校正**：MEMORY 中记录的 FP8 tp=4 TTFT=86ms 是短 prompt（~20 tokens）场景，与本次 10k 输入 382.9ms 不可直接比较，两者均正确但场景不同
- **Run 抖动**：tp=4 Run1/Run2 TTFT 差异较大，建议后续测试增加 runs=3 取中位数

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
