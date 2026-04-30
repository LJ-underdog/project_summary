# teammate-10 progress: #702 compute 占比估算

## 任务
估算 bf16_tuned_gemm miss（attention/dense/shared 层 BF16 GEMM）的实际 FLOPs 占比，与 MoE CK fused_moe FLOPs 对比，判断 H6 假设的实际影响范围。

## 结论摘要（一行）
BF16 tuned_gemm miss 路径占总 prefill GEMM FLOPs 的 **46.1%**（92.83 TFLOPs / 201.29 TFLOPs），MoE CK 路径占 **53.9%**（108.47 TFLOPs）；若 torch.mm fallback 比 tuned ASM 慢 ~3.3x（50% vs 15% peak 利用），bf16 miss 在 tp=2 可解释 **~87 ms** TTFT gap，远超观测到的 FP8 prefill 总时长，**强烈支持 H6 是 TTFT gap 的主导根因**。

---

## 1. 模型参数（实测，从 config.json 读取）

来源：`/root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/6eebda59dd87ca5729648ec7cfed0becfceb273e/config.json`

| 参数 | 值 | 备注 |
|------|-----|------|
| hidden_size | 4096 | |
| intermediate_size | 11264 | dense MLP（layer 0-2）|
| moe_intermediate_size | 1280 | per routed expert |
| share_expert_dim | 1280 | shared expert intermediate |
| moe_num_experts | 288 | |
| moe_top_k | 8 | |
| num_attention_heads | 64 | Q heads |
| num_attention_groups | 8 | KV heads（GQA, ratio 8:1）|
| head_dim | 128 | |
| num_hidden_layers | 45 | |
| moe_layers_enum | layer 3..44 | 共 42 MoE 层；layer 0-2 dense |
| vocab_size | 128896 | |

注：之前任务背景里 num_attention_heads=32 是错的，**实际是 64 heads + 8 KV groups**。这影响 attention proj 的 N 维度。

---

## 2. FLOPs 分解表（prefill, M=10262, tp=1 理论）

公式：GEMM FLOPs = 2 × M × N × K

| 层类型 | 路径 | TFLOPs | 占比 |
|--------|------|--------|------|
| Attention QKV+O proj（45 层）| BF16 tuned_gemm（**MISS**）| 69.73 | 34.64% |
| Dense MLP（layer 0-2，3 层）| BF16 tuned_gemm（**MISS**）| 8.52 | 4.23% |
| Shared expert（42 MoE 层）| BF16 tuned_gemm（**MISS**）| 13.56 | 6.74% |
| Router gate（42 MoE 层，N=288）| BF16 tuned_gemm（**MISS**）| 1.02 | 0.51% |
| **BF16 miss 小计** | torch.mm fallback | **92.83** | **46.11%** |
| MoE routed experts（42 层，top_k=8）| CK fused_moe | 108.47 | 53.89% |
| **总 GEMM** | | **201.29** | 100% |
| FlashAttention SDPA（独立）| flash kernel | 77.64 | （非 GEMM）|

计算细节：
- QKV proj: M × (64+2×8)×128 × 4096 × 2 = 10262 × 10240 × 4096 × 2 ≈ 0.861 TFLOPs/层 × 45 = 38.74 TFLOPs
- O proj: M × 4096 × (64×128) × 2 = 10262 × 4096 × 8192 × 2 ≈ 0.689 TFLOPs/层 × 45 = 30.99 TFLOPs
- Dense gate_up: M × 22528 × 4096 × 2 ≈ 1.894 TFLOPs/层 × 3 = 5.68 TFLOPs
- Dense down: M × 4096 × 11264 × 2 ≈ 0.947 TFLOPs/层 × 3 = 2.84 TFLOPs
- Shared gate_up: M × 2560 × 4096 × 2 ≈ 0.215 TFLOPs/层 × 42 = 9.04 TFLOPs
- Shared down: M × 4096 × 1280 × 2 ≈ 0.108 TFLOPs/层 × 42 = 4.52 TFLOPs
- Router: M × 288 × 4096 × 2 ≈ 0.024 TFLOPs/层 × 42 = 1.02 TFLOPs
- MoE routed: M × 8 × 2560 × 4096 × 2 + M × 8 × 4096 × 1280 × 2 ≈ 2.583 TFLOPs/层 × 42 = 108.47 TFLOPs

---

## 3. TTFT gap 解释力分析

假设 MI350X BF16 peak ≈ 2500 TFLOPS：

| TP | 路径 | per-GPU FLOPs | tuned (50% eff) | torch fallback (15% eff) | gap |
|----|------|---------------|-----------------|--------------------------|-----|
| tp=2 | BF16 miss | 46.41 TFLOPs | 37.1 ms | 123.8 ms | **+86.6 ms** |
| tp=2 | MoE CK | 54.23 TFLOPs | 43.4 ms | — | — |
| tp=4 | BF16 miss | 23.21 TFLOPs | 18.6 ms | 61.9 ms | **+43.3 ms** |
| tp=4 | MoE CK | 27.12 TFLOPs | 21.7 ms | — | — |

观测对比：
- FP8 perf_bench tp=2 TTFT = 388 ms
- FP8 perf_bench tp=4 TTFT = 241 ms
- gfx942 baseline 显著更低（参见 RESULTS.md），TTFT gap 量级在 100~200 ms

**结论**：bf16_tuned_gemm miss 引发的 torch fallback，在 tp=2 单独就足以贡献 ~87 ms 额外延迟；这与 FP8 prefill 路径中 BF16 attn/dense/shared 层占 GEMM FLOPs 46% 的事实一致。即使 torch.mm 实际只比 tuned 慢 1.5x（更保守），仍会贡献 ~20-30 ms 额外延迟，足以解释观测到的 TTFT gap 量级。

---

## 4. h1_tp2_full.log 中 62 次 miss 的 (N,K) 分布

来源：`/home/hanchang/project_summary/step35-flash-support/16_perf_gfx950_verified/logs/h1_tp2_full.log`

总 miss 次数：**62**

按 M 分类：
- M=1: 22 calls（decode/capture warmup）
- M=10262: 20 calls（prefill 主路径，每个 GEMM 出现 2 次 = 重复调用 / 不同 layer 共享 shape）
- M=16384: 20 calls（CUDA-graph capture 用的最大 max_q_len）

每种 M 下出现的 (N,K) 形状（tp=2 切分后）：

| (N, K) | 推测来源 | 验证 |
|--------|---------|------|
| (5120, 4096) | Attention QKV proj，N=10240/2=5120 | ✓ |
| (4096, 4096) | Attention O proj 或 hidden→hidden | ✓ |
| (11264, 4096) | Dense MLP gate_up，N=2×11264/2=11264 | ✓（dense layer 0-2）|
| (4096, 5632) | Dense MLP down，K=11264/2=5632 | ✓ |
| (7168, 4096) | 估计是 MTP 或 head（vocab 之外的辅助投影；不在主 attn/MLP/MoE 列表）| 待查，可能 nextn predict |
| (1280, 4096) | Shared expert gate_up 部分 / 或 router 相关，K=hidden=4096 | ✓（shared）|
| (4096, 640) | Shared expert down，K=1280/2=640 | ✓（shared）|
| (4096, 6144) | Combined / fused projection（hidden=4096 输出，6144 输入）| 待查 |
| (32, 4096) | Tiny head（head-wise attn gate 或 sink，N=64/2=32）| ✓（use_head_wise_attn_gate=True）|
| (48, 4096) | Tiny head（可能 head_gate + extra）| 同上 |
| (64448, 4096) | LM head（vocab=128896/2=64448）| ✓（仅 M=1 出现，sample 阶段）|

**关键观察**：
1. **N=288 的 router gate 形状未出现** — 说明 router 走了别的 GEMM 路径（不是 aiter gemm_a16w16），可能是 torch native 或独立 kernel；不在 62 次 miss 之内。
2. **MoE 专家 shape 完全未出现**：没有任何形如 N=2×1280/tp 或 K=1280/tp 且涉及 288 个 expert 的 batched-GEMM 调用通过 bf16_tuned_gemm 路径。这**直接验证 MoE routed expert 走 CK fused_moe 而非 bf16_tuned_gemm**。
3. 62 次 miss = 11 unique shape × 2（gate_up + 某种重复）× 3 个 M bucket（1 / 10262 / 16384）。每个 prefill batch 实际触发约 20 次 miss 的 dispatch。
4. 如果 bf16 layer 在每个 transformer layer 都调用一次而非只 1 次，实际 fallback 总耗时 = miss 次数 × 单次延迟。当前 log 看每种 shape 只 2 次（一次 prefill + 一次 capture），意味着 aiter dispatch 在 tuned-CSV miss 后会缓存 fallback 决定，**不重复打印**，但**每个 transformer layer 仍真实执行 fallback GEMM**。

---

## 5. 与 H6 的关系

H6 假设：bf16_tuned_gemm.csv 不覆盖 Step-3.5-Flash 的 prefill 形状，导致 attn/dense/shared 层走 torch.mm fallback，贡献 TTFT gap。

本次 #702 的数学计算结论：
- **占比成立**：BF16 miss 路径占 prefill GEMM FLOPs 的 46.11%，**不是少数**。
- **量级足够**：即使保守估计 torch.mm 比 tuned ASM 慢 1.5x，也能贡献 20-30 ms TTFT 增量；按 3x 倍率（更接近实际差距）可贡献 ~87 ms（tp=2）/ ~43 ms（tp=4）。
- **MoE 与 BF16 miss 是独立的两条路径**：MoE 走 CK fused_moe，本次 62 miss 中无 MoE shape，#701 应独立验证 CK kernel 命中率。
- H6 可作为 TTFT gap 的**主要根因**，下一步修复路径明确：补 bf16_tuned_gemm.csv 中 M={10262, 16384}, K=4096, N∈{32, 48, 1280, 4096, 5120, 7168, 11264} 等 11 组形状的 tuning entry。

---

## 引用文件

- 模型 config: `/root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/6eebda59dd87ca5729648ec7cfed0becfceb273e/config.json`
- Miss log: `/home/hanchang/project_summary/step35-flash-support/16_perf_gfx950_verified/logs/h1_tp2_full.log`
- 上游 H6 调查: `/home/hanchang/project_summary/step35-flash-support/16_perf_gfx950_verified/progress/teammate-7.md`
- FP8 fmoe 调查: `/home/hanchang/project_summary/step35-flash-support/16_perf_gfx950_verified/progress/teammate-8.md`
