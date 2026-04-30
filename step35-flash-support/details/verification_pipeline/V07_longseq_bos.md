# V07 LongSeq BOS 验证计划

适用范围：tp=4 长序列（≥10k tokens）输出全 BOS bug 的修复，对应 aiter commit `a2883ab37`（在 `/home/hanchang/junlin12_repos/aiter`），以及 PROJECT_SUMMARY_longseq.md 里描述的 workaround：从 `glm5_bf16_tuned_gemm.csv` 删除 `(N=4096, K=2048, M=16384)` 那一行，把 Step-3.5-Flash o_proj 的 padded_M=16384 命中强制 fallback 到 `torch.mm`。

参考文档：
- `/home/hanchang/project_fp8_tp4/PROJECT_SUMMARY_longseq.md`
- `/home/hanchang/project_fp8_tp4/lead_progress_longseq.md`
- `/home/hanchang/project_fp8_tp4/proposed_fix_A01_v2_longseq.md`
- `/home/hanchang/project_fp8_tp4/logs/longseq_debug/*`（关键日志，见 PROJECT_SUMMARY §9）

---

## A. Code Review 结论

### A.1 修复内容确认

**修复目标**：删除 `/home/hanchang/aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv` 中的下行（修复前 line 45）：

```
gfx950,256,16384,4096,2048,False,torch.bfloat16,torch.bfloat16,False,False,asm,10,1,213.9689,_ZN5aiter24bf16gemm_bf16_tn_256x256E,0.0,1284.66,1019.32
```

**当前状态核查**（reviewer 在本 verification 写作时的实地核查，2026-04-25）：
- `Grep ",4096,2048," /home/hanchang/aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv` 命中 line 35-44，全部是 `M ∈ [1, 256]` + `bf16gemm_fp32bf16_tn_*` 系 kernel；**无 M=16384 + 256x256 的条目**。
- `Grep "bf16gemm_bf16_tn_256x256" /home/hanchang/aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv` 仅命中 line 70-71（N=6144, K=3072 / N=6144, K=6144），不在 Step-3.5-Flash o_proj 触发面上。
- 结论：`/home/hanchang/aiter` 工作树当前 HEAD 已无 buggy 行；与 PROJECT_SUMMARY 报告的"已修复"一致。

**Reviewer 注意**：本 V07 只验证 workaround 的正确性 + 副作用 + 影响面，不涉及 ASM kernel 内部修复（属 AMD aiter 团队工作，详见 §C.3）。

### A.2 删行的完整性核查

**Q1：删除该行后，是否还有其他 (N=4096, K=2048) 条目可能命中 ASM 256x256 kernel？**

实地核查结果：
- glm5_bf16_tuned_gemm.csv 中 (N=4096, K=2048) 仅出现于 line 35-44，对应 `M ∈ {1, 2, 4, 8, 16, 32, 48, 64, 128, 256}`，全部是 `bf16gemm_fp32bf16_tn_*x*_splitk_*` 这类 splitK kernel，与 `bf16gemm_bf16_tn_256x256` 不同族，非本 bug 触发面。
- `get_GEMM_A16W16_config`（`/home/hanchang/aiter/aiter/tuned_gemm.py:101-193`）的 lookup 流程：依次用 `gl=None / 0 / 1` 计算 `padded_M`，命中即返回。删除 padded_M=16384 这一行后，对 M ∈ [8193, 16384] 的查询会因为 `gl=None`（M 本值）也不在 csv 中，落入 `default_config` 路径（line 153-191）；`default_config` 在 bf16 + `bpreshuffle=False` 下 → `libtype="torch"`（line 185-187），即 fallback 到 `torch.mm`。
- 结论：删行后 (N=4096, K=2048, M ∈ [8193, 16384]) 走 torch fallback，**不存在残留命中 ASM 256x256 的路径**。

**Q2：删除是否会影响其他 caller？**

- glm5 csv 是模型专用 tuning（`AITER_CONFIG_GEMM_BF16_FILE` 在 ATOM 中默认指向该文件），仅当推理 GLM5 / Step-3.5-Flash 类模型时加载。
- 但实际上 csv lookup 的 key 是 `(cu_num, padded_M, N, K, ...)`，不含模型 ID。因此任何其它模型只要在 gfx950 上跑 (N=4096, K=2048) 的 bf16 GEMM、且加载这个 csv，都受同样 dispatch 影响。删行让所有这类 caller 都走 torch fallback —— 一致性更好，无 caller 残留 bug。

### A.3 Workaround 副作用分析

**Q3：删行的性能成本**

- 修复前 csv 给出 M=16384 时 ASM kernel 213.97 us, 1284.66 TFLOPS（cf. csv 原 line 45 的 `us` 与 `tflops` 列）。
- 修复后 (M ∈ [8193, 16384], N=4096, K=2048) 全部走 `torch.mm`（rocBLAS / hipBLASLt 默认）。
- 单算子开销估计：torch.mm 的 bf16 GEMM 在 gfx950 上对相同形状大致 2-4× ASM kernel 延迟（基于工程直觉，未实测；本 V07 实验 4 会量化）。
- **E2E 测得的延迟反而下降**（PROJECT_SUMMARY §6.3）：10021 tokens TTFT 349ms → 333ms（−4.6%），8206 tokens TTFT 347ms → 320ms（−7.8%）。可能解释：
  1. 原 ASM kernel 对非对齐 M 走慢 padding 路径（虽然结果错，但耗时反而比真正 M=16384 还短，因为 actual_M < 16384）。
  2. 或者 torch.mm 在 (N=4096, K=2048, M=10021) 这一具体形状下被 rocBLAS 路由到一个比 ASM 更适合此形状的 kernel。
- **风险点**：M 真正 = 16384（warmup 的 dummy forward 就是这个值，见 PROJECT_SUMMARY §5 里 teammate-15 提到的 16384 dummy）时性能可能下降。warmup 不计入 TTFT，所以对线上 SLA 影响不大；但仍需 § B 实验 4 量化。

**Q4：性能损失上界假设**

- 即使 torch.mm 比 ASM 慢 2×，单层 o_proj 200us → 400us（M=16384），Step-3.5-Flash 56 layer 全部 prefill 一次额外 11ms；M=10021 实际差距更小（因为 ASM 的 213us 是 M=16384 的标定值）。
- 整体 prefill 影响 < 10ms 量级，相对 333ms TTFT < 3%。**workaround 性能可接受**。

### A.4 其他 CSV 的同类风险

**Q5：其他模型 CSV 是否有同 ASM kernel 条目？**

实地核查（reviewer Grep `bf16gemm_bf16_tn_256x256` 全部 model_configs/）：

| 文件 | 命中行数 | (N, K) 分布 | 风险评估 |
|------|----------|------------|---------|
| `glm5_bf16_tuned_gemm.csv` | 2 | (6144, 3072), (6144, 6144) | 残留 2 条；N=6144 不在 Step-3.5-Flash 任何 proj 形状上（Step3p5: o_proj N=4096, qkv N=2560, mlp 走 fmoe），但其它 GLM5 变体可能命中 → **需补 verify** |
| `llama70B_bf16_tuned_gemm.csv` | 6+ | (8192, 2048) × M={1024,2048,4096,8192,16384,32768} 等 | **K=2048 同族** → 若 llama70B 在 gfx950 上跑长序列，o_proj/down_proj 命中后可能复现同 bug |
| `llama405B_bf16_tuned_gemm.csv` | 7+ | (16384, 2048) × M={512..32768}, (16384, 4608), (2304, 16384) | **K=2048 同族** → 同上风险 |
| `qwen32B_bf16_tuned_gemm.csv` | 0 | — | 不受影响 |
| `kimik2_bf16_tuned_gemm.csv` | 0 | — | 不受影响 |
| `dsv3_bf16_tuned_gemm.csv` | 0 | — | 不受影响 |
| `gptoss_bf16_tuned_gemm.csv` | 0 | — | 不受影响 |

**结论**：
- Step-3.5-Flash 的修复（删 glm5 csv 的 1 条）**只解决了 Step-3.5-Flash 的 case**。
- llama70B / llama405B 在 gfx950 上跑 bf16 长序列时仍可能踩同一 ASM kernel bug（不同 (M, N, K) 组合）。
- glm5_bf16_tuned_gemm.csv 自身仍残留 2 条 256x256 条目（N=6144），下游若有 callers 命中 (N=6144, K=3072 或 6144) 长序列也可能复现 → 实验 5 需扫描这 4 个文件并对每个残留条目做正确性 spot-check。

### A.5 tp=2 / tp=8 不受影响的代码路径验证

**Q6：为何 tp=2 / tp=8 不命中此 bug？**

代码层推断（来自 PROJECT_SUMMARY §7 与 ATOM step3p5 attention 配置）：

| TP | num_attn_heads / TP | num_kv_groups / TP | hidden_per_head | o_proj K_in (= n_q_heads/TP × head_dim) | o_proj K_out (= hidden) | 是否命中 csv (N=4096, K=2048) |
|----|---------------------|---------------------|------------------|-----------------------------------------|-------------------------|-------------------------------|
| 2  | 32                  | 8                   | 128              | 32 × 128 = 4096                         | 4096                    | K_in=4096 ≠ 2048 → 不命中 |
| 4  | 16                  | 4                   | 128              | 16 × 128 = 2048                         | 4096                    | **命中** |
| 8  | 8                   | 2                   | 128              | 8 × 128 = 1024                          | 4096                    | K_in=1024 ≠ 2048 → 不命中 |

**注意**：以上数字是依据 PROJECT_SUMMARY §3 的"K_in=hidden/tp"叙述与 §7 影响面表格反推；Step-3.5-Flash hidden=4096、num_attn_heads=64 是默认值假设。**【未验证 / 推断】**：tp=2 / tp=8 实测 (N=4096, K=2048) 不命中——需用 §B 实验 1 直接列出每个 TP 下 o_proj 的 (M, N, K) tuple 来 cross-check。

---

## B. 验证实验设计

### 实验 1：tgemm.mm 直调验证（最关键，最快）

**目标**：在不跑模型的前提下，直接调用 `tgemm.mm`（ATOM 侧 wrapper，对应 `aiter.gemm_a16w16`），覆盖 BAD 区间 + 边界，确认所有 M 都 diff < 阈值。

**脚本**（保存为 `/tmp/v07_exp1_tgemm.py`）：

```python
import os, sys
os.environ.setdefault("AITER_LOG_LEVEL", "INFO")
import torch
sys.path.insert(0, "/home/hanchang/ATOM")
from atom.model_ops.linear import tgemm

K_in, K_out = 2048, 4096  # Step-3.5-Flash tp=4 o_proj
torch.manual_seed(42)
device = "cuda"
dtype = torch.bfloat16

# 覆盖：边界（8208 OK / 8209 首个 BAD），bisect log 中所有 BAD 样点，
# E2E 实际形状（10021），padded_M 上界（16384，warmup 真值）
M_list = [4096, 8192, 8208, 8209, 8214, 8216, 8240, 8990, 9007, 9019, 10021, 12288, 16384]

w = torch.randn(K_out, K_in, dtype=dtype, device=device)  # weight: [N, K]
results = []
for M in M_list:
    x = torch.randn(M, K_in, dtype=dtype, device=device)
    ref = torch.mm(x.float(), w.float().t())  # fp32 ref
    got = tgemm.mm(x, w, otype=torch.bfloat16).float()
    diff = (got - ref).abs().max().item()
    rel = diff / max(ref.abs().max().item(), 1e-6)
    status = "PASS" if diff < 5 else "FAIL"
    results.append((M, diff, rel, status))
    print(f"M={M:6d}  max_abs_diff={diff:8.4f}  rel={rel:.4f}  {status}")

print("\n=== Summary ===")
fails = [r for r in results if r[3] == "FAIL"]
print(f"FAILED: {len(fails)} / {len(results)}")
for M, d, r, s in fails:
    print(f"  M={M}: diff={d:.4f}, rel={r:.4f}")
```

**命令**：
```bash
cd /tmp && /opt/venv/bin/python /tmp/v07_exp1_tgemm.py 2>&1 | tee /tmp/v07_exp1.log
```

**通过标准**：
- 所有 13 个 M 全部 `diff < 5`（修复后预期 diff ≈ 0；阈值 5 是 bf16 量化噪声上界）。
- 修复前对照（来自 PROJECT_SUMMARY 表 §6.3）：M=8209 diff=197.38, M=8214 diff=392, M=8216 diff=207.90 → 全部 FAIL。
- **若任何 M FAIL**：说明 csv 删行未生效或 dispatch 未走 torch fallback，需查 `AITER_LOG_TUNED_CONFIG` 日志确认 `libtype` 实际值。

**额外推荐**：在脚本顶部 `os.environ["AITER_LOG_TUNED_CONFIG"] = "1"`，捕获 dispatch 日志确认每个 M 的 libtype 是 `torch` 而非 `asm`。

### 实验 2：E2E 长序列验证（核心）

**目标**：在真实推理 pipeline 上确认 first_output_token ≠ 0，输出连贯中文。

**命令**：
```bash
cd /tmp && rm -rf /root/.cache/atom/* && \
MODEL=stepfun-ai/Step-3.5-Flash TP=4 GMU=0.7 MAX_TOKENS=10 \
CUDA_VISIBLE_DEVICES=0,1,2,3 AITER_LOG_LEVEL=WARNING \
/opt/venv/bin/python /home/hanchang/project_fp8_tp4/logs/perf_compare_10k/run_inference.py \
  2>&1 | tee /tmp/v07_exp2_e2e_10k.log
```

**通过标准**（全部满足才 PASS）：
- 进程 exit code = 0，无 traceback。
- 日志中 `num_tokens_input` = 10021（即真的跑了长序列）。
- `token_ids` 第一个元素 ≠ 0（修复后预期 = 3648，对应 "好的"）。
- `token_ids` 不全相同（diversity > 1）。
- 输出文本目检：连贯中文，不是 `<｜begin▁of▁sentence｜>` 重复。

**对照基线**（修复前 `phase0_baseline.log`）：`token_ids = [0]*10`，输出 `<｜begin▁of▁sentence｜>` ×10。

**失败处理**：若仍 [0]\*10 → 检查 ATOM 是否正确加载了修复后的 aiter wheel（`pip show aiter` + `python -c "import aiter; print(aiter.__file__)"`）。若加载的是旧 wheel，需 `pip install -e /home/hanchang/aiter` 重装。

### 实验 3：短 prompt 回归

**目标**：确保 workaround 没有把短 prompt（< 8192 tokens，原本就走 torch fallback）搞坏。

**命令**：
```bash
cd /tmp && rm -rf /root/.cache/atom/* && CUDA_VISIBLE_DEVICES=0,1,2,3 \
  AITER_LOG_LEVEL=WARNING \
  /opt/venv/bin/python -m atom.examples.simple_inference \
  --model stepfun-ai/Step-3.5-Flash --kv_cache_dtype bf16 --trust-remote-code \
  --tensor-parallel-size 4 --level 0 --temperature 0 --max-tokens 64 \
  2>&1 | tee /tmp/v07_exp3_short_tp4.log
```

（76 tokens 短 prompt 走 ATOM examples 默认 prompt set，prompt 长度参考 `simple_inference.py` 默认值。）

**通过标准**：
- 4 个 sample 全部输出连贯文本，无 BOS 重复。
- TPOT 在 baseline 范围（tp=4 BF16 baseline TPOT ≈ 15.75ms，详见 MEMORY 性能速查表）；允许 ±10% 浮动。

**对照**：与 PROJECT_SUMMARY §6.3 的 `fix_e2e_short.log` 全 PASS 一致。

### 实验 4：性能影响验证（M=16384 torch.mm vs 原 ASM）

**目标**：量化 workaround 在 M=16384 这一原本走 ASM kernel 的 worst case 上的性能损失。

**脚本**（`/tmp/v07_exp4_perf.py`）：

```python
import os, sys, time
os.environ.setdefault("AITER_LOG_LEVEL", "WARNING")
import torch
sys.path.insert(0, "/home/hanchang/ATOM")
from atom.model_ops.linear import tgemm

K_in, K_out = 2048, 4096
device = "cuda"
dtype = torch.bfloat16
torch.manual_seed(0)
w = torch.randn(K_out, K_in, dtype=dtype, device=device)

WARMUP = 10
ITERS = 200
for M in [4096, 8192, 8208, 10021, 12288, 16384]:
    x = torch.randn(M, K_in, dtype=dtype, device=device)
    # tgemm.mm（修复后路径）
    for _ in range(WARMUP):
        _ = tgemm.mm(x, w, otype=dtype)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        _ = tgemm.mm(x, w, otype=dtype)
    torch.cuda.synchronize()
    t_tgemm = (time.perf_counter() - t0) / ITERS * 1e6  # us

    # torch.mm 同形状对照（理论上 tgemm 在 fallback 下应 ≈ torch.mm）
    for _ in range(WARMUP):
        _ = torch.mm(x, w.t())
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        _ = torch.mm(x, w.t())
    torch.cuda.synchronize()
    t_torch = (time.perf_counter() - t0) / ITERS * 1e6

    ratio = t_tgemm / t_torch
    print(f"M={M:6d}  tgemm={t_tgemm:8.2f}us  torch.mm={t_torch:8.2f}us  ratio={ratio:.2f}x")
```

**命令**：
```bash
cd /tmp && /opt/venv/bin/python /tmp/v07_exp4_perf.py 2>&1 | tee /tmp/v07_exp4_perf.log
```

**通过标准**：
- 修复后 tgemm.mm 与 torch.mm 延迟比 ≤ 1.5×（fallback 路径基本就是 torch.mm，差距应 ≈ 1.0×）。
- 与"原 ASM kernel 213us @ M=16384"对照：tgemm.mm 在 M=16384 下绝对延迟 ≤ 2 × 213 = 426us 即可接受（workaround 容忍 2× 性能损失）。
- 若 ratio > 3× → 性能损失过大，需评估是否在 dispatcher 加 actual_M ≠ padded_M 的安全 check（保留 ASM kernel 给 M=16384 真值，仅对 actual_M < padded_M 走 torch）。

**注意**：实验 4 给的是 micro-benchmark 数字，与 E2E TTFT 影响（实测反而 −4.6%）不直接可比；实验 4 主要用来给 SRE 留性能档案。

### 实验 5：其他 CSV 扫描 + spot-check

**目标**：确认其他 model_configs 的 256x256 ASM 条目是否在 gfx950 上仍有同 bug（决定是否要扩展 fix）。

**5.a 扫描**：

```bash
echo "=== model_configs scan ===" > /tmp/v07_exp5_scan.log
for f in /home/hanchang/aiter/aiter/configs/model_configs/*.csv; do
  hits=$(grep -c "bf16gemm_bf16_tn_256x256" "$f" 2>/dev/null || echo 0)
  if [ "$hits" -gt 0 ]; then
    echo "FOUND $hits hit(s) in $f" | tee -a /tmp/v07_exp5_scan.log
    grep -nE ",4096,2048," "$f" | grep "256x256" >> /tmp/v07_exp5_scan.log || true
    grep -n "bf16gemm_bf16_tn_256x256" "$f" | awk -F',' '{print "  line "$1"  M="$3"  N="$4"  K="$5}' >> /tmp/v07_exp5_scan.log
  fi
done
```

**预期输出**（基于 reviewer 已扫描结果，见 §A.4 表）：
- glm5: 2 hits（N=6144），与本 bug 不同形状，但同 kernel → spot-check 必要。
- llama70B: ≥6 hits（N=8192, K=2048），同 K=2048 → 高风险。
- llama405B: ≥7 hits（N=16384, K=2048 / K=4608 / etc），同 K=2048 → 高风险。
- qwen32B / kimik2 / dsv3 / gptoss: 0 hits。

**5.b spot-check**（对每个 hit 的 (N, K) 用实验 1 的脚本验证 M ∈ {padded_M-16, padded_M-13, padded_M-1, padded_M} 的 4 个非对齐点）：

```python
# 对 llama70B 的 (N=8192, K=2048, padded_M=8192) 这条作为示例
# bisect 思路同实验 1，但形状换成下表
shapes_to_check = [
    # (csv_file, N, K, padded_M)
    ("glm5", 6144, 3072, 16384),
    ("glm5", 6144, 6144, 16384),
    ("llama70B", 8192, 2048, 8192),  # 注意：padded_M 命中要 < actual M
    ("llama70B", 8192, 2048, 16384),
    ("llama405B", 16384, 2048, 8192),
    ("llama405B", 16384, 2048, 16384),
]
# 对每条：取 M ∈ [padded_M - 16, padded_M] 范围内非对齐点，跑 tgemm vs torch.mm
```

**通过标准**：
- 5.a：所有 hit 被列出；与 §A.4 的 reviewer 扫描结果一致。
- 5.b：若任意 (N, K, M) 出现 diff > 5 → **该模型在 gfx950 上有同类 bug**，需提 issue + 扩展修复（再删一行 csv）。
- 5.b 全 PASS：说明 bug 仅限 (N=4096, K=2048)；ASM kernel 在其他 (N, K) 下对非对齐 M 处理正常。

**Reviewer 注**：spot-check 不改 csv，只读 + 跑直调验证；任何 FAIL 都登记到 §C 待办。

### 实验 6：tp=2 不受影响实验验证

**目标**：填补 PROJECT_SUMMARY §7 注明的"未做 tp=2/tp=8 ≥10k tokens 长序列直调实验确认"空白。

**命令**：
```bash
cd /tmp && rm -rf /root/.cache/atom/* && \
MODEL=stepfun-ai/Step-3.5-Flash TP=2 GMU=0.9 MAX_TOKENS=10 \
CUDA_VISIBLE_DEVICES=0,1 AITER_LOG_LEVEL=WARNING \
/opt/venv/bin/python /home/hanchang/project_fp8_tp4/logs/perf_compare_10k/run_inference.py \
  2>&1 | tee /tmp/v07_exp6_tp2_10k.log
```

**通过标准**：
- 长序列 10021 tokens 下 first_output_token ≠ 0、输出连贯。
- 验证 tp=2 路径（K_in=4096，不命中 csv 的 K=2048 entry）确实不复现 bug。

**注**：tp=8 因 GPU5 硬件阻塞跳过（MEMORY 列出此限制）。等硬件修复后补 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7,...` 的对应实验。

---

## C. 关键问题与回答

### C.1 为何 qkv_proj（K_in=4096, K_out=2560）不受影响？实验如何直接验证？

**代码层回答**（来自 PROJECT_SUMMARY §7 与 csv 实地核查）：
- qkv_proj 在 tp=4 时形状为 `[N, 4096] @ [2560, 4096].T`，即 csv lookup key 为 `(cu_num=256, padded_M=*, N=2560, K=4096)`。
- `Grep ",2560,4096," /home/hanchang/aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv` 应返回空（reviewer 未直接 grep；按 PROJECT_SUMMARY §7 "qkv_proj 规格 (N=2560, K=4096)：N 不在 csv 中，永远走 torch fallback" 的 teammate-19/teammate-20 结论采信）。
- 因此 qkv_proj 始终走 default_config → libtype="torch" → torch.mm，从不命中 ASM 256x256 kernel。

**直接验证实验**（建议加入到实验 1 的脚本扩展）：
```python
# 在 v07_exp1 脚本末尾追加
print("\n=== qkv_proj shape check ===")
K_in_qkv, K_out_qkv = 4096, 2560
w_qkv = torch.randn(K_out_qkv, K_in_qkv, dtype=dtype, device=device)
for M in [8208, 8214, 10021, 16384]:
    x = torch.randn(M, K_in_qkv, dtype=dtype, device=device)
    ref = torch.mm(x.float(), w_qkv.float().t())
    got = tgemm.mm(x, w_qkv, otype=dtype).float()
    diff = (got - ref).abs().max().item()
    print(f"qkv M={M}: diff={diff:.4f}")
```

**通过标准**：所有 M diff < 5（torch fallback 路径，即使没有修复也不应 FAIL）。

### C.2 tp=2 不受影响的验证

见实验 6。

### C.3 彻底修复方向：ASM kernel 的 non-aligned M 问题如何上报/追踪？

**当前修复定性**（PROJECT_SUMMARY §6.1）：commit `a2883ab37` 是 workaround，根因（`bf16gemm_bf16_tn_256x256` 在 actual_M ≠ padded_M 且 M 非 256/16 对齐时 boundary 处理 buggy）未修。

**上报路径建议**：
1. **AMD aiter 仓库 issue**：在 `https://github.com/ROCm/aiter` 开 issue，标题示例：
   `[gfx950][bf16][ASM] bf16gemm_bf16_tn_256x256 produces wrong output for actual_M < padded_M (non-256-aligned)`
   附 minimal repro：`M=8209, N=4096, K=2048, dtype=bf16` + 修复前 csv 行。
2. **Dispatcher 防御性 check**（中期 fix，参见 PROJECT_SUMMARY §8 第 2 条 + `proposed_fix_H18.md` 方案 A）：
   在 `/home/hanchang/aiter/aiter/tuned_gemm.py:301-304` 之前加：
   ```python
   if config["libtype"] == "asm" and config["kernelName"] in BUGGY_ASM_KERNELS:
       if M != padded_M:
           # actual_M ≠ padded_M 时不安全，强制 torch fallback
           use_torch = True
   ```
   `BUGGY_ASM_KERNELS = {"_ZN5aiter24bf16gemm_bf16_tn_256x256E"}`。这种修法比删 csv 行更通用，能保留 M=16384 真值的 ASM 加速。
3. **ASM 源码层修复**（长期 fix）：需要 AMD aiter ASM 团队修 `bf16gemm_bf16_tn_256x256` 的 last-tile mask 或 split-K 边界。本团队无 ASM 源码访问，无法本地修复。
4. **CI 保护**：把"tp=4 + 10021 tokens prefill first_token != 0"加到 ATOM CI（PROJECT_SUMMARY §8 第 5 条），防止其他 commit 又把这条 csv 行加回来。

**上报清单（建议外发）**：
- 复现脚本：实验 1 的 `/tmp/v07_exp1_tgemm.py`。
- BAD M 集合：[8209, 8223] ∪ [8225, 8239] ∪ [8990, 8991] ∪ [8993, 9007] ∪ [9009, 9019]（PROJECT_SUMMARY §4.2 步骤 12，来自 `t19_bisect_seed42.log`）。
- 影响范围：§A.4 表中标记为"K=2048 同族"的所有模型 + (N=4096, K=2048) 命中。
- 临时 workaround：删除 csv 命中行（已合入 commit `a2883ab37`）。

---

## D. 验证依赖与执行顺序

| 步骤 | 依赖 | 优先级 | 失败时是否阻断后续 |
|------|------|--------|--------------------|
| 实验 1（tgemm 直调） | 无 | P0 | 是（最快验证 fix 是否真生效） |
| 实验 5.a（CSV 扫描） | 无 | P0 | 否（只是登记） |
| 实验 2（E2E 10k tokens） | 实验 1 PASS | P0 | 是 |
| 实验 3（短 prompt 回归） | 实验 1 PASS | P0 | 是 |
| 实验 4（性能影响） | 实验 1 PASS | P1 | 否（信息性） |
| 实验 5.b（其他 CSV spot-check） | 实验 5.a 完成 | P1 | 否（FAIL 登记到 §C.3 上报） |
| 实验 6（tp=2 长序列） | 无 | P1 | 否（验证负面控制） |
| C.1 qkv 直验 | 实验 1 同时跑 | P1 | 否 |

**建议顺序**：实验 1 + 实验 5.a 并行 → 实验 2 + 实验 3 并行 → 实验 4 + 实验 5.b + 实验 6 并行。

---

## E. 通过标准汇总

| 验证项 | 通过标准 | 数据来源 | 数值阈值 |
|--------|---------|---------|---------|
| tgemm 直调 BAD 区间 | M ∈ {8209, 8214, 8216, 10021} 全部 diff < 5 | 实验 1 | < 5 (bf16) |
| tgemm 直调 OK 边界 | M=8208 diff < 5（保持原行为） | 实验 1 | < 5 |
| tgemm 直调 padded_M 真值 | M=16384 diff < 5 | 实验 1 | < 5 |
| E2E 10021 tokens | first_output_token ≠ 0 且输出连贯中文 | 实验 2 | first_token != 0 |
| E2E 短 prompt 回归 | 4 sample 全连贯，TPOT 在 baseline ±10% | 实验 3 | TPOT ≤ 17.3ms |
| 性能 M=16384 | tgemm 延迟 ≤ 2× 原 ASM (213us) → ≤ 426us | 实验 4 | ≤ 426us |
| 其他 CSV 扫描 | 列出全部 hit；登记每个文件的 (M, N, K) | 实验 5.a | 完整性 |
| 其他 CSV spot-check | 任何 hit 对应 4 个非对齐 M 全 diff < 5 | 实验 5.b | < 5 |
| tp=2 长序列 negative control | tp=2 + 10021 tokens first_token ≠ 0 | 实验 6 | first_token != 0 |
| qkv_proj 不受影响 | (N=2560, K=4096) 全 M diff < 5 | C.1 直验 | < 5 |

---

## F. 待确认问题（信息性，不阻断 PASS）

1. **AITER wheel 安装路径**：实验执行前需先确认 ATOM 加载的 aiter 是 `/home/hanchang/aiter` 工作树（而不是某个老的 site-packages）。命令：`/opt/venv/bin/python -c "import aiter; print(aiter.__file__)"`。
2. **ASM 内部 bug 假说**：是 split-K 计数器溢出？tile padding 条件？last-tile mask？无 ASM 源码访问，**【推断，未验证】**（与 PROJECT_SUMMARY §8 一致）。
3. **glm5 csv 残留 (N=6144) 两条**：是否 GLM5 / Step-3.5 有任何 caller 命中？需 grep ATOM 源码 `tgemm.mm` 调用点的 (M, N, K) 形状全集，目前未 grep。
4. **dispatcher 防御性 check（§C.3 第 2 条）**：是否值得替换 csv 删行？需要团队对齐"workaround 简单 vs 通用性更强"的取舍。本 verification 只验证当前 workaround，不实施第 2 条。
5. **llama70B / llama405B 的 ASM 256x256 命中（§A.4）**：本验证组只 spot-check 4-6 条，未做完整 sweep。若这些模型在 gfx950 上线，需要独立的 V08-Llama-LongSeq 验证组。
6. **CI long-seq 端到端校验落地**（PROJECT_SUMMARY §8 第 5 条）：本 V07 不负责 CI 接入，仅建议。

---

V07 LongSeq BOS 验证计划完成
