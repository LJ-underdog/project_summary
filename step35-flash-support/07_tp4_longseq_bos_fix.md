# Project Summary: tp4 Long-Sequence BOS Bug (tp4-longseq-bos-debug)

**日期**：2026-04-25
**状态**：已修复（commit `a2883ab37`，`/home/hanchang/junlin12_repos/aiter`）
**修复人**：Jun Lin <junlin12@amd.com>
**修复 commit message**：`fix: remove buggy ASM kernel entry for (N=4096,K=2048) bf16 GEMM on gfx950`

---

## 1. Bug 症状

在 gfx950 (8x MI350X) 上运行 ATOM + Step-3.5-Flash（bf16）时，使用 tensor parallel = 4 推理：

- 输入约 ≥ 8209 tokens 的 prompt 时，模型输出的**第一个 token 就是 BOS**（token id = 0），后续 decode token 也全是 BOS。
- 输出文本表现：`<｜begin▁of▁sentence｜>` 重复若干次。
- TTFT / TPOT 数值正常（不是 OOM 或 kernel 崩溃，prefill 跑完了，只是 logits 全坏）。
- tp=2 / tp=8 不复现该 bug（**注：此结论基于代码分析推断——tp=2/tp=8 的 o_proj K_in 不同，不命中有 bug 的 GEMM 规格；未做 tp=2/tp=8 ≥10k tokens 长序列直调实验**，详见 §7）；同一模型 tp=4 短 prompt（≤ 8208 tokens）输出完全正常。
- 现象在 BF16 与 FP8 量化下都出现（FP8 路径中 attention `o_proj` 仍然是 bf16，所以走同一 kernel）。

| 字段 | BUG case (10021 tokens) | OK case (8206 tokens) |
|------|------------------------|------------------------|
| first_output_token | 0 (BOS) | 3648 ("好的") |
| token_ids (前 3) | [0, 0, 0] | [3648, 303, 6640] |
| TTFT | 0.349s | 0.347s |
| 数据来源 | `logs/longseq_debug/phase0_baseline.log`（teammate-1）；`logs/longseq_debug/t13_bug_10021.log`（teammate-13） | `logs/longseq_debug/t13_ok_8208.log`（teammate-13）（Reviewer 核实：t13 实测 0.347s） |

## 2. 精确复现条件

- **模型**：`stepfun-ai/Step-3.5-Flash`（bf16 与 FP8 均触发；FP8 attention `o_proj` 仍走 bf16 GEMM）。
- **TP**：4（tp=2 / tp=8 不触发）。
- **输入 token 数**：≥ 8209 tokens（精确阈值见 §4-§5）。
- **配置**：`gpu-memory-utilization=0.7`、`max-num-batched-tokens=16384`、`max-num-seqs=4`、`enforce-eager`、`kv_cache_dtype=bf16`。
- **执行命令**（来自 teammate-1）：
  ```
  cd /tmp && MODEL=stepfun-ai/Step-3.5-Flash TP=4 GMU=0.7 MAX_TOKENS=10 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /opt/venv/bin/python /home/hanchang/project_fp8_tp4/logs/perf_compare_10k/run_inference.py
  ```
- **关键日志（修复前 baseline）**：`/home/hanchang/project_fp8_tp4/logs/longseq_debug/phase0_baseline.log`
  - `num_tokens_input: 10021`、`token_ids: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`。

## 3. 根因（Root Cause）

aiter 中的 ASM bf16 GEMM kernel `_ZN5aiter24bf16gemm_bf16_tn_256x256E`（符号名称，对应
`bf16gemm_bf16_tn_256x256`）在被 dispatcher 当作 `padded_M=16384` 调用、但 actual_M 落在
某些非对齐的"坏"区间时，**输出数值完全错误**（未抛 NaN，而是把结果写成与 ground truth
diff 远大于 max_abs 的值）。teammate-19 / teammate-20 把这个错误数值放回模型，Step-3.5-Flash
layer 0 attention `o_proj` 输出立即变成 NaN（teammate-15 / teammate-16），层间 residual 把 NaN
传染到全部 hidden state，最后 logits 也是 NaN，`argmax(NaN)` 在 PyTorch 下确定性返回 token 0
= BOS。

### 触发链（精确到文件 / 行号）

1. `/home/hanchang/ATOM/atom/model_ops/linear.py:393`
   `tgemm.mm(x, self.weight, bias, otype=bf16)`  — Step-3.5-Flash 的 attention `o_proj`（bf16，N=4096，K=2048）。
2. `/home/hanchang/aiter/aiter/tuned_gemm.py:552-572`
   `TunedGemm.mm` → `gemm_a16w16(...)`。
3. `/home/hanchang/aiter/aiter/tuned_gemm.py:101-193`
   `get_GEMM_A16W16_config`：用 `get_padded_m(M, N, K, gl)` 查 csv。对 (N=4096, K=2048)：
   - M ≤ 8192 → `padded_M=8192` → CSV 不命中 → fallback torch；
   - M ∈ [8193, 16384] → `padded_M=16384` → 命中 csv。
4. `/home/hanchang/aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv`
   原 line 45（修复前内容）：
   ```
   gfx950,256,16384,4096,2048,False,torch.bfloat16,torch.bfloat16,False,False,asm,10,1,213.9689,_ZN5aiter24bf16gemm_bf16_tn_256x256E,0.0,1284.66,1019.32
   ```
5. `/home/hanchang/aiter/aiter/tuned_gemm.py:301-304`
   `libtype="asm"` → `asm_gemm(...)` → `gemm_a16w16_asm` → 调用 ASM kernel
   `bf16gemm_bf16_tn_256x256`，但实际 M 是 e.g. 8214 或 10021，**不是真正的 16384**。
6. ASM kernel 在这些非对齐 M 下产生错误结果（精确坏区间见 §4 / §5）。
7. `/home/hanchang/ATOM/atom/models/step3p5.py` 中 layer 0 self_attn 的 `o_proj` 输出 NaN
   → residual + post LN → MLP → 全 layer NaN → logits NaN → argmax = 0 = BOS。

## 4. 调查过程（Investigation Timeline）

### 4.1 已排除的方向（含实验证据）

| 假说 | 实验/证据 | 排除原因 |
|------|---------|---------|
| H1 sliding window 是根因 | A01：`ATOM_STEP3P5_NO_SLIDING=1` + 10021 tokens（teammate-2，`A01_no_sliding.log`） | 关闭 sliding 后 token_ids 仍 = [0]×10 |
| H2 enforce_eager（无 cudagraph）触发 bug | B01：删除 `--enforce-eager` 跑 GMU=0.5（teammate-2，`B01_no_eager_gmu05.log`） | cudagraph 路径下 token_ids 仍 = [0]×10 |
| H3 阈值刚好 8192 = 16 × sliding_window | R00 二分（teammate-R，`sweep_R00_bisect.log` / `sweep_R00_bisect2.log`） | M=8206 OK、M=8215 BUG，翻转点在 (8206, 8215]；不是 8192 |
| H4 aiter triton fwd_prefill autotune 在 M≈8208 切换 | 阅读 `aiter/aiter/ops/mha.py`、`fwd_prefill.py`（teammate-5） | ATOM 默认走 CK 路径，CK wrapper 无 seqlen 分支；triton gfx950 单 config 无切换 |
| H5 rope cache 上限 ≈ 8192 | 阅读 `step3p5.py:365-367` 与 `aiter/rotary_embedding.py:132-140`（teammate-6） | `max_position_embeddings=262144`，cache size 远大于 8208 |
| H6-Python CK fmha_v3 在 Python 侧有 seqlen 上限 | 阅读 `mha.py:569-603` 与 `cmdGenFunc_mha_varlen_fwd`（teammate-6） | Python 侧无 seqlen assert；dispatch 仅按 dtype/feature |
| H6-CK CK kernel 内部 seqlen 限制 | 直调 `fmha_v3_varlen_fwd`（teammate-7，`fmha_v3_minrepro.log`） | M=4096..8224 输出 norm 平滑增长，无 NaN/Inf |
| H7 max_model_len 默认过小 | 显式设 `--max-model-len 32768`（teammate-8，`H7_maxlen32k_10k.log`） | 仍 token_ids=[0]×5；ATOM 实际取 hf 的 262144 |
| H8 forward_vars 预分配缓冲的"垃圾行"污染下游 | 在 `model_runner.py` self.model() 之前对所有 `forward_vars.gpu` 调 `zero_()`（teammate-11，`H8_zero_test.log`） | 全清零后 first token 仍 = 0 |
| ATOM_DEBUG_NAN2 报告的 NaN 是真有效行 NaN | OK case（8208 tokens，输出"好的，用户"）也报相同 NaN 告警（teammate-10B，`debug_nan2_ok_8208.log`） | 假阳性：`isnan().any()` 在 [16384, hidden] 全张量上跑，垃圾行 NaN 被算入 |
| sliding 层 mha_varlen_fwd 是 NaN 来源 | 用 out= 预分配 + 完整 18 参数直调（teammate-10C，`mha_varlen_proper.log`） | M ∈ {4096, 8192, 8208, 8216, 10021} 全部 has_nan=False |
| RCCL all-reduce 在 [10021, 4096] bf16 上出 NaN | tp=4 torchrun 直调 `dist.all_reduce`（teammate-17，`ar_test_tp4.log`） | M ∈ {8192, 8206, 8208, 8214, 8216, 10021, 10240, 12288} 全部 has_nan=False |
| chunked prefill / max-num-batched-tokens 触发 | C01 sweep（teammate-3）：触发点 8499 < max_num_batched_tokens=16384 | 阈值远低于 chunked 触发条件 |
| FP8 量化路径独有 | 已知 BF16 也复现（teammate-1 baseline 用 BF16） | 与量化无关 |

### 4.2 关键转折点（按时间顺序）

1. **Phase 0 baseline 复现**（teammate-1，`phase0_baseline.log`）：确认 bug 100% 复现，token_ids=[0]×10。
2. **A01/B01 同时排除两个最显眼假设**（teammate-2）：sliding window 与 enforce_eager 都不是根因。
3. **C01 长度 sweep**（teammate-3，`sweep_*.log`）：定位粗阈值在 (8200, 8500]，疑似与 8192 = 2^13 相关。
4. **R00 精确二分**（teammate-R，`sweep_R00_bisect2.log`）：把阈值收紧到 (8206, 8215]，最近边界 `8208 = 8192 + 16`，**不是 8192 本身**。
5. **代码审计排除多条路径**（teammate-5/6/7/8）：CK / triton / fmha_v3 kernel / rope cache / max_model_len 全部清零。
6. **ATOM_DEBUG_NAN2 证伪**（teammate-10B）：发现 NaN 告警是假阳性，扫除一个误导信号。
7. **回归到模型层逐层定位**（teammate-13，`t13_bug_10021.log`）：标准 runner 下 prefill 出来的 hidden_states.norm() = NaN，`argmax(NaN) = 0`，BOS 现象解释清楚 — bug 在 forward 内。
8. **Layer 0 sub-module hook**（teammate-15，`t15_bug.full.log`）：embed_tokens / inputLN clean，**layer0_attn 输出首次 NaN**。
9. **self_attn 子组件 hook**（teammate-16，`t16_bug_full.log`）：qkv_proj / q_norm / k_norm / rotary / fmha / g_proj 全 clean，**`o_proj` 4 个 rank 同时输出 NaN**。
10. **区分 GEMM vs all-reduce**（teammate-17）：all-reduce 直调测试无 NaN → NaN 来自本地 GEMM。
11. **tgemm.mm 直调出现数值偏差**（teammate-18，`t18_tgemm.log`）：M=8214 时 tgemm 与 torch.mm diff=392 > max=247，说明 tgemm 路由到的 kernel 数值错误。
12. **dispatch 阈值 + 精确 bisect**（teammate-19，`t19_bisect_seed42.log`）：定位 csv 命中 padded_M=16384 → ASM kernel `bf16gemm_bf16_tn_256x256`，并精确给出 BAD M 集合：[8209, 8223] ∪ [8225, 8239] ∪ [8990,8991] ∪ [8993,9007] ∪ [9009,9019]。
13. **修复 + 端到端验证**（teammate-20，`fix_e2e_bug10021.log`）：删 csv 行后 first_token = 3648 = "好的"，bug 消失。

## 5. 定位过程（Localization）

定位链（每一步都引用具体证据）：

1. **`model.forward()` 输出 NaN**：`logs/longseq_debug/t13_bug_10021.log` 显示
   `Prefill: n_tok=10021, hs_last_norm=nan`；OK case (n_tok=8206) 同位置 norm=120.78。
   `argmax(NaN)` 在 PyTorch 下返回 0，解释了 BOS。
2. **Layer 0 是首次出 NaN 的层**：`logs/longseq_debug/t15_bug.full.log`
   embed_tokens norm=0.96 (clean)、inputLN norm=51.42 (clean)、**layer0_attn norm=nan**；
   warmup 阶段 16384 dummy forward 全 clean → 与真实输入相关。
3. **Self_attn 中 `o_proj` 是首次出 NaN 的子组件**：`logs/longseq_debug/t16_bug_full.log`
   qkv_proj/q_norm/k_norm/rotary_emb/attn(fmha)/g_proj 全 clean，**`o_proj` 4 ranks
   同时 nan=True**。
4. **NaN 不来自 all-reduce**：`logs/longseq_debug/ar_test_tp4.log` 中
   tp=4 RCCL all-reduce 在 [10021, 4096] bf16 上 has_nan=False。又因 ATOM 调
   `all_reduce` 时从未传 `prefill_support=True`（teammate-17 grep 全 ATOM 无命中），
   n_tok=10021 → 78.3 MiB > 64 MiB 的 ca_comm decode 阈值（`custom_all_reduce.py:437-438`），
   实际走 pynccl_comm。可疑面收窄到本地 GEMM。
5. **本地 GEMM 数值错误**：`logs/longseq_debug/t18_tgemm.log` 显示 o_proj 规格
   `[N,2048] @ [4096,2048].T`、bf16，N=8214 时 tgemm 与 torch.mm diff=392 但 max 仅 247。
   qkv 规格 `[N,4096] @ [2560,4096].T` 全 N diff=0（与服务侧 qkv clean 一致）。
6. **Dispatch 命中错误 ASM kernel**：`aiter/tuned_gemm.py:101-193` + `glm5_bf16_tuned_gemm.csv`
   line 45 显示 (N=4096, K=2048) padded_M=16384 → asm kernel
   `_ZN5aiter24bf16gemm_bf16_tn_256x256E`；M ≤ 8192 不命中走 torch fallback，
   解释了为什么 M ≤ 8192 OK。
7. **精确坏区间二分**：`logs/longseq_debug/t19_bisect_seed42.log` 给出
   首个 BAD = M=8209（diff=197.38, rel=0.80），最后 OK = M=8208（diff=0），
   坏区间见上 §4.2 步骤 12。

## 6. 修复方案

### 6.1 是 Workaround 还是彻底修复？

**当前 commit `a2883ab37` 是 workaround，不是彻底修复**。

- **Workaround 性质**：从 `glm5_bf16_tuned_gemm.csv` 删除一行 tuning，让
  (M, N=4096, K=2048) bf16 GEMM 全部回退到 `torch.mm` (F.linear)，绕开有 bug 的 ASM kernel。
- **彻底修复需要**：修改 ASM kernel `bf16gemm_bf16_tn_256x256` 本身对非 256/16
  对齐 M 的 boundary 处理（属于 AMD aiter ASM 团队的工作，本次未做）；或
  在 dispatcher 加 actual_M ≠ padded_M 的安全 check（`proposed_fix_H18.md` §三 方案 A）。
- **代价**：M=16384 失去 ASM 加速，回落到 torch 路径；从 teammate-20 测得 TTFT 修复前后
  349ms → 333ms（10021 tokens）、347ms → 320ms（8206 tokens），实际未见明显回归。
  （修复前 8206 tokens TTFT 0.347s 来自 `t13_ok_8208.log`；修复后 0.320s 来自 `fix_e2e_ok8208.log`）

### 6.2 修复内容

**文件**：`/home/hanchang/aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv`

**改动**：删除原 line 45：
```
gfx950,256,16384,4096,2048,False,torch.bfloat16,torch.bfloat16,False,False,asm,10,1,213.9689,_ZN5aiter24bf16gemm_bf16_tn_256x256E,0.0,1284.66,1019.32
```

**修复前后行数**：73 → 72。
**备份**：`/home/hanchang/project_fp8_tp4/before_fix_H18.csv`。
**Git 提交**：commit `a2883ab37`，仓库 `/home/hanchang/junlin12_repos/aiter`，
作者 Jun Lin <junlin12@amd.com>。

### 6.3 修复验证（来自实验数据）

| 验证项 | 修复前 | 修复后 | 数据来源 |
|--------|--------|--------|---------|
| tgemm.mm 直调 M=8208, diff vs torch.mm | 0 | 0 | `logs/longseq_debug/fix_gemm_verify.log` |
| tgemm.mm 直调 M=8209, diff | 197.38 (rel=0.80) | 0 | 修复前 `t19_bisect_seed42.log`；修复后 `fix_gemm_verify.log` |
| tgemm.mm 直调 M=8214, diff | 392 (rel=1.59) | 0 | 同上 |
| tgemm.mm 直调 M=8216, diff | 207.90 (rel=0.84) | 0 | 同上 |
| tgemm.mm 直调 M=10021, diff | ~378–392 | 0 | 同上 |
| E2E 10021 tokens, first_output_token | 0 (BOS) | 3648 ("好的") | 修复前 `phase0_baseline.log`；修复后 `fix_e2e_bug10021.log` |
| E2E 10021 tokens, output 文本 | `<｜begin▁of▁sentence｜>` ×10 | "好的，用户给了一段重复了很多遍的关于" | 同上 |
| E2E 10021 tokens, TTFT | 0.349s | 0.333s | 同上 |
| E2E 8206 tokens 回归 | 已 OK | 仍 OK (token_ids=[3648, 303, 6640, 1621, 78040]) | `fix_e2e_ok8208.log` |
| 短 prompt 回归（4 examples） | OK | 全部 OK | `fix_e2e_short.log` |

## 7. 影响范围

**直接受影响的配置**：
- gfx950（CU=256）+ bf16 GEMM + (N=4096, K=2048) + M ∈ [8209, ~9020] 中的非对齐子集。
- Step-3.5-Flash tp=4 attention `o_proj`（K_in=hidden/tp=4096/4×2 GQA=2048，K_out=hidden=4096），
  prefill 长度 ≥ 8209 tokens 时正中靶心。

**不受影响**：
- tp=2 / tp=8 配置：`o_proj` 的 K_in 不同（4096 / 1024），不命中此 kernel（teammate-19 §5 代码分析推断；**未做 tp=2/tp=8 ≥10k tokens 长序列直调实验确认**）。
- qkv_proj 规格 (N=2560, K=4096)：N 不在 csv 中，永远走 torch fallback（teammate-18 / teammate-19）。
- M ≤ 8192：padded_M=8192 不在 csv 中，走 torch fallback。
- decode 阶段（M=1）：永远 torch fallback。
- 短 prompt 推理（< 8209 tokens）。

**潜在影响**：其他模型若有 (N=4096, K=2048) bf16 GEMM 且 M 落在 bug 区间，也会受影响 — 当前 commit 已让所有 (M, 4096, 2048) 走 torch fallback。

## 8. 后续建议

1. **上报 aiter / AMD ASM kernel 团队**：根因是 `bf16gemm_bf16_tn_256x256` 在 actual_M ≠ padded_M 且 M 非 256/16 对齐时 boundary 处理 buggy。需要 ASM 源码修复。
2. **dispatcher 加防御性 check**（proposed_fix_H18.md §三 方案 A）：在 `aiter/tuned_gemm.py:301` 之前，若 kernelName 在已知 buggy 集合且 actual_M ≠ padded_M，强制 fallback torch。比删 csv 更通用。
3. **扫描其他 tuning CSV**：`kimik2 / qwen32B / llama70B / dsv3 / llama405B / gptoss / glm5` 共 7 份 bf16 csv，是否还有同 ASM kernel 条目（teammate-20 已建议）。
4. **写 unit test**：对 (N=4096, K=2048) bf16 GEMM 在 M ∈ [8193, 16384] 全范围对 `torch.mm` 做 reference check（test_tgemm3.py 已具雏形）。
5. **CI 加 long-seq 端到端校验**：tp=4 + 10021 tokens prefill 的 first_token != 0 判断（防回归）。

【未验证 / 推断】
- ASM kernel 内部 boundary 的具体 bug（是 split-K 计数器溢出？tile padding 条件？last-tile mask？）—— 没有 ASM 源码访问，**【推断，未验证】**。
- 其他 (N, K) 组合是否也有此 kernel 的 bug：teammate-19 / teammate-20 已建议扫描，未执行。

## 9. 参考文件

### 关键日志
- `/home/hanchang/project_fp8_tp4/logs/longseq_debug/phase0_baseline.log` — bug baseline
- `/home/hanchang/project_fp8_tp4/logs/longseq_debug/sweep_R00_bisect2.log` — 阈值精确二分
- `/home/hanchang/project_fp8_tp4/logs/longseq_debug/t13_bug_10021.log` — hidden_states=NaN 证据
- `/home/hanchang/project_fp8_tp4/logs/longseq_debug/t15_bug.full.log` — layer 0 attn 首次 NaN
- `/home/hanchang/project_fp8_tp4/logs/longseq_debug/t16_bug_full.log` — o_proj 子组件 NaN
- `/home/hanchang/project_fp8_tp4/logs/longseq_debug/ar_test_tp4.log` — all-reduce 干净
- `/home/hanchang/project_fp8_tp4/logs/longseq_debug/t18_tgemm.log` — tgemm 数值偏差
- `/home/hanchang/project_fp8_tp4/logs/longseq_debug/t19_bisect_seed42.log` — 精确 BAD 边界
- `/home/hanchang/project_fp8_tp4/logs/longseq_debug/fix_gemm_verify.log` — 修复后 tgemm 直调验证
- `/home/hanchang/project_fp8_tp4/logs/longseq_debug/fix_e2e_bug10021.log` — 修复后端到端验证
- `/home/hanchang/project_fp8_tp4/logs/longseq_debug/fix_e2e_ok8208.log` — OK case 回归
- `/home/hanchang/project_fp8_tp4/logs/longseq_debug/fix_e2e_short.log` — 短 prompt 回归

### Progress 文件
- `/home/hanchang/project_fp8_tp4/progress/teammate-1.md` — Phase 0 baseline
- `/home/hanchang/project_fp8_tp4/progress/teammate-2.md` — A01/B01 (sliding/eager 排除)
- `/home/hanchang/project_fp8_tp4/progress/teammate-3.md` — C01 长度 sweep
- `/home/hanchang/project_fp8_tp4/progress/teammate-4.md` — D01/D02 代码阅读
- `/home/hanchang/project_fp8_tp4/progress/teammate-R.md` — Phase 1 review + 精确二分
- `/home/hanchang/project_fp8_tp4/progress/teammate-5.md` — flash_attn dispatch 调查
- `/home/hanchang/project_fp8_tp4/progress/teammate-6.md` — rope cache / fmha_v3 排除
- `/home/hanchang/project_fp8_tp4/progress/teammate-7.md` — fmha_v3 直调
- `/home/hanchang/project_fp8_tp4/progress/teammate-8.md` — max_model_len 排除
- `/home/hanchang/project_fp8_tp4/progress/teammate-10A.md` — mha_varlen 早期尝试
- `/home/hanchang/project_fp8_tp4/progress/teammate-10B.md` — NaN 假阳性确认
- `/home/hanchang/project_fp8_tp4/progress/teammate-10C.md` — mha_varlen_fwd 正确调用
- `/home/hanchang/project_fp8_tp4/progress/teammate-11.md` — H8 zero_() 验证
- `/home/hanchang/project_fp8_tp4/progress/teammate-13.md` — 标准 runner hidden state 对比
- `/home/hanchang/project_fp8_tp4/progress/teammate-15.md` — layer 0 sub-module 定位
- `/home/hanchang/project_fp8_tp4/progress/teammate-16.md` — self_attn 内部 NaN 定位
- `/home/hanchang/project_fp8_tp4/progress/teammate-17.md` — o_proj GEMM vs all-reduce
- `/home/hanchang/project_fp8_tp4/progress/teammate-18.md` — tgemm.mm 直调
- `/home/hanchang/project_fp8_tp4/progress/teammate-19.md` — GEMM dispatch 阈值
- `/home/hanchang/project_fp8_tp4/progress/teammate-20.md` — 修复实施与验证
- `/home/hanchang/project_fp8_tp4/proposed_fix_H18.md` — 修复方案文档

### 关键代码位置
- `/home/hanchang/ATOM/atom/model_ops/linear.py:393` — o_proj 调 tgemm.mm
- `/home/hanchang/aiter/aiter/tuned_gemm.py:101-193, 247-336, 552-572` — TunedGemm dispatch
- `/home/hanchang/aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv` — 修改目标
- `/home/hanchang/ATOM/atom/models/step3p5.py:320-474` — Step3p5Attention
- `/home/hanchang/aiter/aiter/dist/device_communicators/custom_all_reduce.py:423-442` — should_custom_ar 阈值
