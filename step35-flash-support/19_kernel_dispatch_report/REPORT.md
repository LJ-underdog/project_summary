# Step-3.5-Flash-FP8 Kernel Dispatch 报告

**模型**：`stepfun-ai/Step-3.5-Flash-FP8`（snapshot 6eebda59）
**平台**：gfx950（MI350X），ATOM acff926d / aiter 0f8164017
**测试配置**：FP8 tp=2（GPU 0,1）和 FP8 tp=4（GPU 0,1,2,3）
**验证方法**：代码追踪（t1-code #101）+ 推理日志（t2-inference #102，AITER_LOG_TUNED_CONFIG=1）+ 单 kernel 单元测试（t3-unit #103）
**日期**：2026-04-29

---

## 一、关键模型参数

| 参数 | 值 |
|------|-----|
| hidden_size | 4096 |
| num_attention_heads | 64（Q），8（KV，GQA）|
| head_dim | 128 |
| moe_intermediate_size | 1280（per expert，全量）|
| inter_dim（tp=2） | 1280/2 = **640** |
| inter_dim（tp=4） | 1280/4 = 320 → ATOM padding → **384** |
| 640 % 256 | **128（非零，ASM 1-stage 不触发）** |
| 384 % 256 | **128（非零，ASM 1-stage 不触发）** |
| moe_num_experts | 288 routed + 1 shared |
| moe_top_k | 8（routed）+ 1（shared，部分层）|
| Attention 类型 | ~1/4 层 full FMHA，~3/4 层 sliding window（window=512）|
| FP8 量化 | weight_block_size=[128,128]，e4m3fn，dynamic activation |
| modules_to_not_convert | attn proj、dense MLP（layer 0-2）、shared expert、lm_head（共 286 项，保持 BF16）|

---

## 二、Kernel 对照总表

| 操作 | 场景 | Kernel 类型 | 具体 Kernel / 模块 | 实验验证来源 |
|------|------|------------|-------------------|-------------|
| **MoE routed experts（FP8 blockscale）** | Prefill（任意 token 数）| **CK 2-stage** | `ck_moe_stage1_fwd` + `ck_moe_stage2_fwd` | #102 日志 L76/L83，#103 unit test |
| **MoE routed experts（FP8 blockscale）** | Decode（token=1）| **CK 2-stage** | 同上，block_m=16, ksplit=4 | #102 日志 L148-L155，#103 unit test |
| **MoE routed experts（layer 43-44 SwigluStep）** | Prefill + Decode | **CK 2-stage** | `module_moe_ck2stages_..._swiglustep_per_1x128_mulWeightStage2` | #102 JIT .so 加载 L85 |
| **Attention Prefill（full + sliding window）** | Prefill | **ASM（aiter FA v3）** | `aiter.flash_attn_varlen_func` / `fmha_v3_varlen_fwd` | #102 JIT L180-L181，#101 代码 attention_mha.py:509 |
| **Attention Decode（sliding window 层）** | Decode | **Triton（Gluon）** | `torch.ops.aiter.pa_decode_gluon` | #101 代码 attention_mha.py:585 |
| **Attention Decode（full attention 层）** | Decode | **ASM** | `aiter.pa_fwd_asm` 或 `aiter.pa_persistent_fwd`（取决于 block_size）| #101 代码 attention_mha.py:589-591 |
| **BF16 Linear（attn QKV/O proj）** | Prefill + Decode | **torch.mm** | `F.linear`（tgemm.mm CSV miss）| #102 日志 62次miss，#103 unit 5/5 miss |
| **BF16 Linear（dense MLP，layer 0-2）** | Prefill | **torch.mm** | `F.linear` | #102 日志 L86-L95 |
| **BF16 Linear（shared expert，42 层）** | Prefill + Decode | **torch.mm** | `F.linear`（少数小 N decode 见下）| #102 日志 |
| **BF16 Linear（g_proj，head-wise gate，tiny N）** | Decode（M=1, N=32 或 N=48）| **aiter Skinny GEMV** | `wv_splitk_small_fp16_bf16`（HIP kernel）| #102 日志 L130-L131 `skinny solution:2` |
| **lm_head**（vocab proj） | Prefill last token + Decode | **torch.mm** | `F.linear`（tgemm.mm CSV miss）| #102 日志 L109-L110 |

---

## 三、MoE Kernel 详细说明

### 3.1 为什么 MoE 不走 ASM（fmoe_fp8_blockscale_g1u1）？

ASM 1-stage 触发条件（`aiter/fused_moe.py:881-883`）：

```python
if q_type == QuantType.per_1x128:
    # for fp8 blockscale, ck has better performance so disable assembly kernel
    run_1stage = token > 32 and (inter_dim % 256 == 0)
```

Step-3.5-Flash 的 inter_dim：
- tp=2：1280/2 = **640**，640 % 256 = **128 ≠ 0** → `run_1stage = False`
- tp=4：320 → padding → **384**，384 % 256 = **128 ≠ 0** → `run_1stage = False`

**两种 TP 配置下，inter_dim 均不满足 256 整除条件，ASM 1-stage 路径从不触发。**

此外，代码注释明确说明："for fp8 blockscale, ck has better performance so disable assembly kernel"——即 CK 2-stage 性能优于 ASM，这是刻意设计的限制。

### 3.2 实际加载的 CK 2-stage 模块（来自 #102 日志）

```
module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2
module_moe_ck2stages_f8_f8_preshuffle_on_b16_swiglustep_per_1x128_mulWeightStage2
```

命名解读：
- `f8_f8`：输入 FP8，权重 FP8
- `preshuffle_on`：权重已 shuffle（gfx950 始终 preshuffle）
- `b16`：输出 BF16
- `silu` / `swiglustep`：layer 0-42 用 silu，layer 43-44 用 SwigluStep（clamp ±7）
- `per_1x128`：blockscale 量化（block_n=128, block_k=128）
- `mulWeightStage2`：routing weight 在 stage2 中乘入

### 3.3 Prefill vs Decode 的差异

| 参数 | Prefill（M=502 实测）| Decode（M=1）|
|------|---------------------|--------------|
| run_1stage | False | False |
| block_m | 64 | 16 |
| ksplit | 0 | 4 |
| use_nt（non-temporal load）| True | True |
| estimated_m_per_expert | ~15（502 tokens / 289 experts）| 0 |
| Kernel | CK 2-stage | CK 2-stage |

### 3.4 单 kernel 验证（#103 unit test）

直接调用 `fused_moe(quant_type=QuantType.per_1x128, inter=640, E=289, topk=9)`：

**Prefill (M=512)**：
```
[aiter] run_1stage = False, ksplit = 0, block_m = 64
[aiter] using 2stage default for (256, 512, 4096, 640, 289, 9, ..., QuantType.per_1x128, True, False)
→ 加载 module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_1x128_mulWeightStage2
→ 调用 ck_moe_stage1 + ck_moe_stage2
PASS: out.shape=[512, 4096], dtype=bfloat16
```

**Decode (M=1)**：
```
[aiter] run_1stage = False, ksplit = 4, block_m = 16
[aiter] using 2stage default for (256, 1, 4096, 640, 289, 9, ..., QuantType.per_1x128, True, False)
PASS: out.shape=[1, 4096]
```

---

## 四、Attention Kernel 详细说明

### 4.1 Prefill：统一走 aiter Flash Attention v3

```
aiter.flash_attn_varlen_func（fmha_v3_varlen_fwd）
```

来源：`atom/model_ops/attentions/attention_mha.py:509`

Full attention 和 sliding window attention 层在 prefill 阶段均调用同一个 varlen FMHA 接口，sliding window 通过 `window_size=(512, 0, 0)` 参数传入，kernel 内部处理 mask。

### 4.2 Decode：按层类型分流

```python
# attention_mha.py:130
use_triton_attn = self.sliding_window != -1 or self.head_dim != 128
```

Step-3.5-Flash head_dim=128：
- **sliding window 层**（window=512）：`sliding_window != -1` → `use_triton_attn = True`
  → `torch.ops.aiter.pa_decode_gluon`（Triton Gluon kernel）
- **full attention 层**：`sliding_window == -1` → `use_triton_attn = False`
  → `aiter.pa_fwd_asm`（block_size ≠ 1024）或 `aiter.pa_persistent_fwd`（block_size == 1024）

### 4.3 实验日志验证

#102 推理日志中，`module_fmha_v3_varlen_fwd` JIT 模块在真实 prefill batch 到来时加载（L178-L181）。Decode attention kernel 未打印独立 dispatch 行（AITER_LOG_TUNED_CONFIG=1 不覆盖 attention dispatch），通过代码追踪（#101）确认路径。

---

## 五、BF16 Linear 详细说明

### 5.1 dispatch 路径

所有 BF16 不量化层（attn proj、dense MLP、shared expert、lm_head）均通过：

```
atom/model_ops/linear.py:393
  tgemm.mm(x, weight, bias)
    → aiter/tuned_gemm.py:gemm_a16w16()
      → 查 /tmp/aiter_configs/bf16_tuned_gemm.csv
      → 全部 MISS → torch solution:0 → F.linear（torch.mm）
```

### 5.2 例外：极小 N 的 Decode GEMV

`M=1, N=32` 或 `M=1, N=48`（head-wise attention gate g_proj）在 decode 阶段命中 skinny 分支：
```
using skinny solution:2 → wv_splitk_small_fp16_bf16（aiter HIP split-K kernel）
```

这是 aiter 对小 N 大 K 的 GEMV 专项优化，并非 torch op。

### 5.3 为什么 CSV 全 miss？

- gfx950 上所有 `*bf16_tuned_gemm.csv` 中，没有 Step-3.5-Flash 使用的 (N, K=4096) 形状（如 (5120,4096)、(7168,4096)、(4096,4096)）
- 仅有 glm5/llama/dsv3/qwen 等其他模型的特定形状
- 这是 BF16 GEMM 性能瓶颈的根因（见 RESULTS.md §四）

---

## 六、tp=4 与 tp=2 的差异

| 维度 | tp=2 | tp=4 |
|------|------|------|
| inter_dim | 640（640%256=128）| 384（ATOM padding 320→384，384%256=128）|
| MoE kernel | CK 2-stage，block_m=64 | CK 2-stage，block_m 同（待实测确认）|
| attn QKV proj N | (64+2×8)×128/2 = 5120 | (64+2×8)×128/4 = 2560 |
| attn O proj N | 4096（不 TP 切）| 4096（不 TP 切）|
| dense gate_up N | 11264/2 = 5632 | 11264/4 = 2816 |
| BF16 miss 次数 | 62 | 120（N 拆分更多，形状更多）|
| BF16 dispatch | torch.mm | torch.mm |

tp=4 MoE inter_dim=384，384%256=128≠0，同样不满足 ASM 条件，仍走 CK 2-stage。

---

## 七、完整 Kernel 清单（所有 JIT .so 模块，来自 #102 推理日志）

| .so 模块 | 类型 | 用途 |
|----------|------|------|
| `module_aiter_core` | 核心库 | 基础 aiter op |
| `module_custom_all_reduce` | HIP | TP all-reduce |
| `module_activation` | HIP | 激活函数 |
| `module_moe_sorting` | HIP | MoE expert 排序（top-k dispatch）|
| `module_quant` | HIP | FP8 量化（dynamic activation）|
| `module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2` | **CK** | MoE routed expert（Silu，layer 3-42）|
| `module_moe_ck2stages_f8_f8_preshuffle_on_b16_swiglustep_per_1x128_mulWeightStage2` | **CK** | MoE routed expert（SwigluStep，layer 43-44）|
| `module_sample` | HIP | 采样（argmax/temperature）|
| `module_rope_2c_cached_positions_fwd` | HIP | RoPE 位置编码 |
| `module_cache` | HIP | KV cache 写入 |
| `module_custom`（含 `wv_splitk_small_fp16_bf16`）| **HIP** | Skinny GEMV（tiny decode BF16）|
| `module_fmha_v3_varlen_fwd` | **ASM** | Prefill attention（varlen Flash Attention v3）|

Decode attention（paged attention）未打印独立 .so 加载 log，推断已编入 `module_aiter_core` 或通过 ATOM `PagedAttentionImpl` 装饰器调用。

---

## 八、实验验证摘要

| 验证方法 | 结论 |
|---------|------|
| **#101 代码追踪** | MoE 2-stage 路径：fused_moe.py:883 `inter%256==0` 不满足；attention 各路径：attention_mha.py:509/585/589-591 |
| **#102 推理日志**（AITER_LOG_TUNED_CONFIG=1，512 input / 32 output）| `run_1stage=False` 12/12，`using 2stage default` 12/12，`torch solution:0` 58次，`skinny solution:2` 4次，JIT 模块 12 个（含 fmha_v3 + 两个 CK MoE .so）|
| **#103 单 kernel 测试**（fused_moe 直调，M=512/M=1，inter=640，E=289）| Prefill: CK 2-stage, block_m=64；Decode: CK 2-stage, block_m=16, ksplit=4；BF16 linear 5/5 全部 torch solution:0 |

---

## 九、重要修正（对之前 MEMORY/KNOWN_FACTS 的纠正）

| 之前的错误记录 | 实际正确情况 | 证据 |
|--------------|-------------|------|
| "tp=2 inter_dim=640，640%256==0，满足 ASM 条件" | **640 % 256 = 128 ≠ 0，ASM 不触发** | #102 日志 run_1stage=False 12/12；#103 unit test |
| "MoE prefill 走 ASM fmoe_fp8_blockscale_g1u1" | **全程走 CK 2-stage**（prefill 和 decode 均如此）| #102 JIT 加载的是 CK .so；#103 直调确认 |
| "TPOT 差异来自 ASM vs CK 选择" | ASM 在 Step-3.5-Flash 上从不触发，TPOT 差异原因需另查 | #103 |

---

## 十、结论

**Step-3.5-Flash-FP8 在 gfx950 上（tp=2 / tp=4）所有 MoE 层无论 prefill 还是 decode，均走 CK 2-stage kernel（ck_moe_stage1 + ck_moe_stage2），从不走 ASM fmoe_fp8_blockscale_g1u1。** 这由 inter_dim 不满足 256 整除（640%256=128，384%256=128）以及代码注释"CK has better performance"共同决定。

BF16 保留层（attention proj、dense MLP、shared expert、lm_head）因 bf16_tuned_gemm.csv 缺少 Step-3.5-Flash 专属形状，全部 fallback 到 `torch.mm`，仅极少数小 N decode GEMV 命中 aiter skinny kernel。

Attention prefill 走 aiter Flash Attention v3 ASM kernel；decode 分 sliding window（Triton Gluon）和 full attention（ASM paged attention）两路。
