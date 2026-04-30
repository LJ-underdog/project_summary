# t2-inference: FP8 tp=2 推理日志 + kernel dispatch 提取

> 任务：#102
> 日期：2026-04-29
> 角色：teammate-2（推理日志）

## 运行配置
- 模型：Step-3.5-Flash-FP8（snapshot 6eebda59dd87...）
- TP=2，CUDA_VISIBLE_DEVICES=0,1
- 输入 token=502（target=512），输出 token=32
- 环境变量：`AITER_LOG_LEVEL=INFO AITER_LOG_TUNED_CONFIG=1`
- ATOM cache 已清理；GPU 0/1 运行前/后 VRAM=0
- exit_code=0；perf_bench 输出 TTFT=0.054s, TPOT=11.143ms, throughput_decode=89.74 tokens/s

## 日志文件
- `logs/fp8_tp2_dispatch_full.log`（226 行，~50KB；脚本 stdout+stderr 全量）
- `logs/fp8_tp2_dispatch.log`（仅 [PERF] 行；perf_bench 自带）

## 1. 日志中实际观察到的 kernel dispatch 信息

### 1.1 加载的 aiter JIT .so 模块（unique 12 个）
来源：`grep -E "import \[module_"` 提取
- `module_aiter_core` — 核心
- `module_custom_all_reduce` — TP all-reduce
- `module_activation` — 激活
- `module_moe_sorting` — MoE expert 排序
- `module_quant` — FP8 量化
- `module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2` — **CK 2-stage MoE（Silu）**
- `module_moe_ck2stages_f8_f8_preshuffle_on_b16_swiglustep_per_1x128_mulWeightStage2` — **CK 2-stage MoE（SwigluStep）**
- `module_sample` — 采样
- `module_rope_2c_cached_positions_fwd` — RoPE
- `module_cache` — KV cache write
- `module_custom` — 含 wv_splitk_small_fp16_bf16（skinny GEMV）
- `module_fmha_v3_varlen_fwd` — **prefill attention（varlen FMHA v3）**

注意：日志中未出现 `module_pa_*`（paged attention）或类似 decode attention 的 .so import — decode 阶段的 attention 实现可能在 `module_aiter_core` 里，或通过 ATOM 的 MLAAttentionImplDecorator/PagedAttentionImpl 转发，但没有打印对应 dispatch 行（无独立 .so JIT 加载）。

### 1.2 计数统计（关键）
| 项 | 计数 | 说明 |
|---|---|---|
| `not found tuned config` | 62 | bf16 tuned_gemm.csv 全 miss |
| 真正 `found tuned config`（去掉 not） | 0 | 完全无命中 |
| `using torch solution:0` | 58 | 大多数 BF16 GEMM 走 torch.mm |
| `using skinny solution:2` | 4 | 小 N 的 decode（M=1, N=32 或 N=48）走 aiter skinny GEMV（wv_splitk_small_fp16_bf16） |
| `using asm` | 0 | **BF16 linear 完全无 ASM** |
| `using ck` | 0 | **BF16 linear 完全无 CK** |
| `run_1stage = True` | 0 | **MoE 完全没走 1-stage / ASM** |
| `run_1stage = False` | 12 | 全部 MoE 走 2-stage |
| `using 2stage default` | 12 | 与 run_1stage=False 一一对应 |
| `using 1stage` | 0 | — |

## 2. MoE routed experts 走的是哪种 kernel？

**结论：全程走 CK 2-stage（ck_moe_stage1 + ck_moe_stage2），从未走 ASM fmoe_fp8_blockscale_g1u1。**

证据（log 行号）：

### Prefill（warmup, M=16384）— L76, L83, L97, L104
```
[aiter] run_1stage = False, ksplit = 0 q_type = QuantType.per_1x128 block_m = 64 use_nt = False, estimated_m_per_expert = 510
[aiter] [fused_moe] using 2stage default for (256, 16384, 4096, 640, 289, 9, 'ActivationType.Silu', 'torch.bfloat16', 'torch.float8_e4m3fn', 'torch.float8_e4m3fn', 'QuantType.per_1x128', True, False)
```
- 即便 `estimated_m_per_expert=510`（远大于常说的"token>32 & inter%256==0 → 走 ASM"的阈值），`run_1stage=False` 仍然 False
- inter_dim=640，640%256=128 ≠ 0 — 这正是不走 ASM 的原因（不满足 inter%256==0 条件）
- TEAM_CONFIG.md 中 KNOWN_FACTS 里"tp=2 inter=640，满足 ASM 条件"是错误的（640 mod 256=128，并不整除）

### Decode（cudagraph capture, M=1）— L148-L155
```
[aiter] run_1stage = False, ksplit = 4 q_type = QuantType.per_1x128 block_m = 16 use_nt = True, estimated_m_per_expert = 0
[aiter] [fused_moe] using 2stage default for (256, 1, 4096, 640, 289, 9, 'ActivationType.Silu', ...)
```
- decode 单 token 时 ksplit=4, block_m=16, use_nt=True

### Real prefill（502 tokens）— L196-L204
```
[aiter] run_1stage = False, ksplit = 0 ... estimated_m_per_expert = 15
[aiter] [fused_moe] using 2stage default for (256, 512, 4096, 640, 289, 9, ...)
```

### 实际加载的 CK kernel .so（L80, L85, L101, L106）
- `module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2.so`（gate up，Silu）
- `module_moe_ck2stages_f8_f8_preshuffle_on_b16_swiglustep_per_1x128_mulWeightStage2.so`（SwigluStep）

L81-L82 进一步确认 Python 端调用了 `ck_moe_stage1` / `ck_moe_stage2`：
```
[aiter] type hints mismatch, override to --> ck_moe_stage1(hidden_states, w1, w2, ..., quant_type, activation, splitk, non_temporal_load, dst_type, is_shuffled) -> None
[aiter] type hints mismatch, override to --> ck_moe_stage2(...)
```

## 3. BF16 linear 走的是什么？

**结论：tgemm.mm 走 aiter `bf16_tuned_gemm.csv` 全 miss，绝大多数回退到 `torch.mm`（torch solution:0），仅极小 N（M=1, N=32/N=48 的 decode）走 aiter skinny GEMV solution:2（wv_splitk_small_fp16_bf16）。**

### 证据（log）
| 阶段 | M | N×K（典型） | 走 | 行号 |
|---|---|---|---|---|
| Warmup (prefill) | 16384 | 5120×4096, 4096×4096, 11264×4096, 4096×5632, 7168×4096, 4096×6144, 1280×4096, 4096×640, 32×4096, 48×4096 | torch solution:0 | L65-L74, L86-L95, L107-L108 |
| LM head (M=1) | 1 | 64448×4096 | torch solution:0 | L109-L110 |
| Decode capture (M=1) | 1 | 5120×4096, 4096×4096, 11264×4096, 4096×5632, 7168×4096, 4096×6144, 1280×4096, 4096×640 | torch solution:0 | L124-L147, L156-L159 |
| Decode capture (M=1, N=32 or N=48) | 1 | 32×4096, 48×4096 | **skinny solution:2** | L130-L131, L144, L146 |
| Real prefill (502 tokens) | 502 | 5120×4096 等全部 shape | torch solution:0 | L176-L207 |

### 关键日志原文（L65 示例）
```
shape is M:16384, N:5120, K:4096 dtype='torch.bfloat16' otype='torch.bfloat16' bias=False, scaleAB=False, bpreshuffle=False,
not found tuned config in /tmp/aiter_configs/bf16_tuned_gemm.csv, will use default config! using torch solution:0
```

### 关键日志原文（L130 示例 — skinny）
```
shape is M:1, N:32, K:4096 ... using skinny solution:2
```
对应 aiter `module_custom` 中的 `wv_splitk_small_fp16_bf16(arg0, arg1, arg2, arg3, arg4) -> None`（L132-L134）—— 这是 aiter 的 split-K skinny GEMV kernel（HIP），并非 torch op。所以严格来说"BF16 linear 全走 torch.mm"是不准确的：极小输出维度的 decode GEMV 走 aiter skinny。

## 4. Attention 走的是什么 kernel？

### Prefill attention — `module_fmha_v3_varlen_fwd`（aiter ASM/HIP varlen FMHA v3）
- 仅在真实 prefill batch 触发（L178-L181，紧跟 L175 "Scheduled prefill batch: 1 reqs, 502 tokens"）
- 函数签名（L180-L181）：`fmha_v3_varlen_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, ..., is_causal, window_size_left, window_size_right, ..., how_v3_bf16_cvt, ..., q_descale, k_descale, v_descale, ...)` — 标准 aiter v3 varlen FMHA
- warmup 阶段（M=16384, dummy）未触发该 import → warmup 用了别的路径（可能是 ATOM 自行替换的 dummy attention）

### Decode attention — 未观察到独立 .so 加载，疑似走 paged attention（ATOM 装饰）
- L25/L28：`Use PagedAttentionImplDecoratorForPluginMode to decorate PagedAttentionImpl`
- L21-L24：`MLAAttentionImplDecoratorForPluginMode`、`MLASparseAttentionImplDecoratorForPluginMode`（仅装饰器注册）
- decode 期间（cudagraph capture & 实际 32 token 生成）日志中没有出现新的 attention kernel JIT import — 说明 decode attention 要么已编进 `module_aiter_core` 里（未单独打印），要么走 paged attention（pa_fwd 未触发独立 import 行）
- 没有出现 `flash_attn_varlen` / `paged_attention` 字面打印 — `AITER_LOG_TUNED_CONFIG=1` 对 attention kernel 不输出独立 dispatch 行

**警告**：本日志不足以最终断言 decode attention 用的是哪个 aiter API。需要 #101（代码追踪）从 ATOM PagedAttentionImpl 装饰器路径反推。

## 5. 意外发现 / 与 KNOWN_FACTS 的偏差

### 5.1 ⚠ 重要：tp=2 MoE 没有走 ASM（与 TEAM_CONFIG KNOWN_FACTS 矛盾）
- TEAM_CONFIG 写："tp=2 inter_dim=640（640%256==0，满足 ASM 条件）"
- **实测 640 % 256 = 128 ≠ 0**，不满足 ASM 条件，run_1stage 始终为 False
- 12 次 MoE 调用全部 `using 2stage default`
- Prefill `estimated_m_per_expert=510`（足够大），但因 inter_dim 不整除 256 仍走 CK 2-stage
- 这意味着 **#101 的代码追踪需要核对真实 ASM 触发条件**（不仅是 token 数，还有 inter%256），并且 #103 单 kernel 验证应直接用相同 shape 复现

### 5.2 BF16 linear "完全 miss tuned" 与 KNOWN_FACTS 一致
- bf16_tuned_gemm.csv miss 率 100%（62/62）
- glm5 model_configs 已包含但未命中（因为可能 shape 不匹配 tp=2 的实际输入）

### 5.3 skinny solution（GEMV）未在 KNOWN_FACTS 提及
- M=1 N=32 / N=48 走 aiter `wv_splitk_small_fp16_bf16`（HIP kernel），不是 torch.mm
- 涉及 MoE gate logits（N=288/289 路由分数？实际 N=32, N=48 应该是某种小投影 — 待 #101 确认到具体 layer）
- 4 次（warmup decode capture 各两个 shape × 2 ranks）

### 5.4 `module_moe_ck2stages_..._mulWeightStage2` 命名解读
- `f8_f8`：input fp8, weight fp8
- `preshuffle_on`：weight 已 shuffle（gfx950 always preshuffle）
- `b16`：output bf16
- `silu` / `swiglustep`：两个不同 .so，分别处理两个 expert
- `per_1x128`：blockscale per-1x128（对应 q_type=QuantType.per_1x128）
- `mulWeightStage2`：在 stage2 应用 routing weight

### 5.5 没有出现 `Capturing` 多 batch — cudagraph_capture_sizes=[1] 只 capture 了 bs=1
- 这意味着 perf_bench 输入 502 tokens 的 prefill 不走 cudagraph（只有 decode 的 bs=1 走）

### 5.6 RCCL 2.27.7 加载，custom_all_reduce 也加载
- 同时加载两套 all-reduce — 看 ATOM 配置选哪个

### 5.7 Tokenizer warning
- L4：`incorrect regex pattern ... You should set fix_mistral_regex=True` — 不影响 kernel dispatch，但可能影响 token 计数（actual_input=501 vs target=512）

## 提取命令复现
```bash
# 1. 全量 MoE dispatch
grep -E "run_1stage|fused_moe.*using|module_moe" logs/fp8_tp2_dispatch_full.log

# 2. BF16 GEMM dispatch
grep -E "torch solution|skinny solution|asm|using ck" logs/fp8_tp2_dispatch_full.log

# 3. attention 模块加载
grep -E "fmha|paged|PagedAttention|MLA" logs/fp8_tp2_dispatch_full.log
```

## 任务状态
- #102 完成 ✓
- 输出供 #201 REPORT.md 引用
- 给 #101 和 #103 的提示：
  - tp=2 inter_dim=640 不满足 inter%256==0，KNOWN_FACTS 需修正
  - decode attention kernel 名未在日志中出现，需从代码追踪
  - skinny solution:2 路径需追踪具体调用 layer（哪些 N=32/48 的 linear）
