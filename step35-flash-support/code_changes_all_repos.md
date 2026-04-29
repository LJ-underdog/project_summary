# Step-3.5-Flash 全栈推理支持 — 三仓库代码修改全记录

**编写日期**：2026-04-29
**覆盖范围**：ATOM / aiter / composable_kernel
**验证状态**：V01-V07 全 PASS（2026-04-26）
**平台**：8x MI350X (gfx950), ROCm

---

## 概览

| Repo | 分支 | HEAD commit | 修改 commit 数 |
|------|------|-------------|---------------|
| ATOM | `feat/step3p5-flash-support` | `ccb64621` / `acff926d` | 6 |
| aiter | `feat/step3p5-moe-swiglustep` | `0f8164017` | 9（含 revert） |
| composable_kernel | `feat/swiglustep-moe-no-quant` | `defd7ad29` | 1 |

**git push 操作路径**：`/home/hanchang/junlin12_repos/{atom,aiter,composable_kernel}`（author: Jun Lin \<junlin12@amd.com\>）

---

## 一、ATOM repo

**路径**：`/home/hanchang/junlin12_repos/atom`
**主要改动文件**：`atom/model_ops/moe.py`、`atom/models/step3p5.py`、`atom/model_engine/model_runner.py`、`atom/model_loader/loader.py`、`atom/model_ops/attentions/aiter_attention.py`

---

### ATOM-1：`ec8cbe87` — Step-3.5-Flash 基础支持 + gfx950 preshuffle 修复

**日期**：2026-04-23
**改动文件**：6 files (+938/-16)

#### 问题根因

gfx950 上 preshuffle_off（NSwizzle=0）CK kernel 在 GEMM 时输出错误结果（cos_sim ≈ -0.017）。原 ATOM 代码对 gfx950 BF16 跳过 `shuffle_weights()`，基于"gfx950 不需要 shuffle"的错误假设。

正确行为：gfx950 的 preshuffle_on（NSwizzle=1）kernel 才是正确的，必须始终 shuffle。

#### 修改内容

**`atom/model_ops/moe.py`（preshuffle fix）**：

删除 gfx950 skip-shuffle 分支，始终调用 `shuffle_weights()`：

```python
# Before: gfx950 bf16 g1u1 skipped shuffle
# After:
shuffle_weights(layer.w13_weight, layer.w2_weight)
```

注：shuffle 必须在 `process_weights_after_loading` 中做，运行时 shuffle 会 OOM。

**`atom/models/step3p5.py`（新增 860 行）**：

- `Step3p5MLP`：dense SwiGLU + 可选 gate/up clamp（`clamp(max=limit)`）
- `Step3p5MoE`：288 routed + 1 shared expert，sigmoid routing + router_bias，自定义 `_routing_function`（绕开 grouped_top_k）
- `Step3p5DecoderLayer`：mixed full/sliding attention（按 `layer_types` 配置）
- `Step3p5ForCausalLM`：含 `detect_fused_expert_format()` / `get_fused_expert_mapping()`，处理 flat `[E, I, H]` checkpoint 格式

**`atom/model_engine/model_runner.py`**：

- L66：注册架构 `"Step3p5ForCausalLM"`
- KV head 计算兼容 Step-3.5 配置字段 `num_attention_groups`（替代 `num_key_value_heads`）
- 新增 `ATOM_SAVE_LAYERS` / `ATOM_DEBUG_LOGITS` 环境变量驱动的 per-layer hidden_states dump hook（验证用）

**`atom/model_loader/loader.py`**：

fused expert detection 前置到 `packed_modules_mapping` 之前，防止 `moe.gate_proj` 被 `gate_proj → gate_up_proj` 错误匹配。

#### 验证

- cos_sim = 0.999989~0.999990（T ∈ {1,4,32,128,512}，真实权重 layer 10）
- tp=2 端到端 4 prompts 正常
- **验证文档**：`verification_pipeline/results/v01_exp1_preshuffle.md`

---

### ATOM-2：`4a8495ec` — SwigluStep per-layer wiring

**日期**：2026-04-24
**改动文件**：`atom/models/step3p5.py`（+81/-23）

#### 问题根因

Step-3.5-Flash config 中 layer 43/44 的 `swiglu_limits=[..., 7.0, 7.0]`（非零），意味着这两层 routed expert 需要 SwigluStep 激活（silu + clamp_upper=7 + clamp_both_sides=±7）。

ATOM 中 `Step3p5MoE` 已存储 `self.clamp_limit`，但从未传给 `FusedMoE`，实际仍走 plain silu。

另一约束：shared expert 在 layer 43/44 的 limit=16，与 CK kernel 硬编码的 7.0f 不同，无法共用 SwigluStep kernel，必须保持 dense path。

#### 修改内容

**新增 per-layer helpers**：

```python
def _uses_swiglustep_at_layer(config, layer_idx) -> bool:
    swiglu_limits = getattr(config, "swiglu_limits", None)
    if not swiglu_limits or layer_idx >= len(swiglu_limits):
        return False
    return bool(swiglu_limits[layer_idx])

def _fuse_shared_at_layer(config, layer_idx) -> bool:
    # SwigluStep 层不融合 shared expert（limit 不同）
    return (
        is_rocm_aiter_fusion_shared_expert_enabled()
        and not _uses_swiglustep_at_layer(config, layer_idx)
    )
```

**`Step3p5MoE.__init__`**：按层传入 `activation=ActivationType.SwigluStep`，shared expert 在 SwigluStep 层走 dense MLP：

```python
self._uses_swiglustep = self.clamp_limit is not None
self._activation = (
    ActivationType.SwigluStep if self._uses_swiglustep else ActivationType.Silu
)
self.experts = FusedMoE(..., activation=self._activation)
self._fuse_shared = _fuse_shared_at_layer(config, layer_idx)
```

#### 验证

- 12/12 cases cos_sim ≥ 0.99999（M ∈ {1,32,256} × seed × {Silu, SwigluStep}）
- 深度 clamp 场景（scale=5.0，values near ±7）验证通过
- **验证文档**：`verification_pipeline/results/v02_exp1_swiglu.md`

---

### ATOM-3：`635e59e9` — BF16 inter_dim padding（tp=4/8）

**日期**：2026-04-24
**改动文件**：`atom/model_ops/moe.py`（+32）

#### 问题根因

CK 2-stage MoE kernel（`gemm_moe_ck2stages.cu` L98）：stage1 的 N 维度 = `w1.size(1) / 2 = inter_dim`（不是 2×inter_dim）。

dispatch 规则（`gen_instances.py`）：
- `inter_dim <= 192` → NPerBlock=64，要求 `inter_dim % 64 == 0`
- `inter_dim > 192` → NPerBlock=128，要求 `inter_dim % 128 == 0`

Step-3.5-Flash（moe_intermediate_size=1280）TP 切分后：
- tp=4：inter=320，320%128=64 ≠ 0 → crash
- tp=8：inter=160，160%64=32 ≠ 0 → crash

#### 修改内容

在 `process_weights_after_loading` 中（shuffle 之前）zero-pad：

```python
inter_dim = w2.shape[2]
align = 64 if inter_dim <= 192 else 128
inter_pad = (inter_dim + align - 1) // align * align
if inter_pad != inter_dim:
    # w13: [E, 2*inter, H] → [E, 2*inter_pad, H]（gate/up 分别 pad）
    # w2:  [E, H, inter]   → [E, H, inter_pad]
    w13_new = torch.zeros(E, 2 * inter_pad, hidden, ...)
    w13_new[:, :inter_dim, :]               = w13[:, :inter_dim, :]     # gate
    w13_new[:, inter_pad:inter_pad+inter_dim, :] = w13[:, inter_dim:, :]  # up
    w2_new = torch.zeros(E, hidden, inter_pad, ...)
    w2_new[:, :, :inter_dim] = w2
```

结果：tp=8 inter=160→**192**，tp=4 inter=320→**384**。零 padding 对 stage2 输出贡献为零，无需 slice。

#### 验证

- tp=4 BF16 TTFT=81ms，tp=2 回归 TTFT=92ms
- **验证文档**：`verification_pipeline/results/v04_tp_support.md`

---

### ATOM-4：`9a67e493` — FP8 `get_fused_moe_quant_config` block_shape fix

**日期**：2026-04-24
**改动文件**：`atom/model_ops/moe.py`（+12/-7）

#### 问题根因

`Fp8MoEMethod.get_fused_moe_quant_config` 的 else 分支硬编码 `block_shape=None`，导致 per_1x128 / per_1x32 quant 路径传入错误 block_shape（影响 EP 路径的 quant config 构建）。

#### 修改内容

拆分 else 分支为显式分派：

```python
elif self.quant_type == QuantType.per_1x128:
    block_shape = [128, 128]
elif self.quant_type == QuantType.per_1x32:
    block_shape = [1, 32]
else:
    block_shape = None
return fp8_w8a8_moe_quant_config(..., block_shape=block_shape)
```

#### 验证

作为 FP8 tp=2 修复的组成部分，随 V05 验证通过。

---

### ATOM-5：`ccb64621` — FP8 tp=4 三处协调修复（**核心**）

**日期**：2026-04-25
**作者**：Jun Lin \<junlin12@amd.com\>
**改动文件**：`atom/model_ops/moe.py`（+64/-12）

#### 问题根因（三个独立问题需同时修）

| # | 问题 | 症状 |
|---|------|------|
| 1 | `create_weights` ValueError check 不支持 padded inter | 加载即 crash |
| 2 | `_process_block_quant` 无 inter_dim zero-pad | 未对齐 CK kernel |
| **3（根本）** | `_load_w13`/`_load_w2` scale TP sharding 用 floor 除法 | scale 残留初值 1.0，dequant 放大 ~5000× → gibberish |

**根因 3 详解**：per_1x128 blockscale 对 inter=1280 共 10 个 N-block。tp=4 时 `10 // 4 = 2`，第 3 个 partial block 永不被任何 rank 复制，停留在 `torch.ones()` 初值 1.0。dequant 时 scale=1.0 而非 ~0.0002，放大约 5000 倍，输出完全乱码。

tp=2（10/2=5，整除）不受影响。

#### 修改内容

**Fix1（`create_weights` L1559-1583）**：ValueError check 改为 padding-aware：

```python
padded_inter = (intermediate_size_per_partition + block_n - 1) // block_n * block_n
if padded_inter % block_n != 0:  # 始终满足，只做 sanity check
    raise ValueError(...)
```

**Fix2（`_process_block_quant` L1713-1740）**：FP8 inter_dim zero-pad（镜像 ATOM-3 的 BF16 逻辑）：

```python
inter_dim = layer.w2_weight.shape[-1]
block_n = 128 if self.quant_type == QuantType.per_1x128 else 32
align = 64 if inter_dim <= 192 else block_n
inter_pad = (inter_dim + align - 1) // align * align
if inter_pad != inter_dim:
    # zero-pad w13 [E, 2*inter, H] → [E, 2*inter_pad, H]
    # zero-pad w2  [E, H, inter]   → [E, H, inter_pad]
```

**Fix3（`_load_w13`/`_load_w2` L2298-2356，根本修复）**：floor → ceil + narrow 边界保护：

```python
# Before:
load_shard_size = loaded_weight.shape[shard_dim] // self.tp_size
# After:
load_shard_size = (loaded_weight.shape[shard_dim] + self.tp_size - 1) // self.tp_size
start = load_shard_size * tp_rank
size = min(load_shard_size, loaded_weight.shape[shard_dim] - start)
loaded_weight = loaded_weight.narrow(shard_dim, start, size)
```

#### 验证

| 配置 | TTFT | TPOT |
|------|------|------|
| FP8 tp=4 | 86ms | 13ms |
| FP8 tp=2 回归 | 78ms | 14ms |
| BF16 tp=4 回归 | 88ms | 16ms |

- **验证文档**：`verification_pipeline/results/v06_fp8_tp4.md`

---

### ATOM-6：`acff926d` — FP8 blockscale align bug fix（Bug 5，tp=8）

**日期**：2026-04-27
**作者**：Jun Lin \<junlin12@amd.com\>
**改动文件**：`atom/model_ops/moe.py`（+6/-4）

#### 问题根因

`_process_block_quant` 中的 align 计算从 BF16 路径照搬：`align = 64 if inter_dim <= 192 else block_n`。

这对 FP8 不正确：gfx950 FP8 mfma 的 KPack=32 约束导致 stage2 kernel 只支持 KPerBlock=128（`static_assert(KPerThread % KPack == 0)` 在 KPerBlock=64 时失败）。

tp=8 inter=160 → align=64 → inter_pad=**192**，但 192%128=64≠0，触发 CK kernel `device_moe_gemm_blockscale.hpp:448` 拒绝。

#### 修改内容

```python
inter_dim = layer.w2_weight.shape[-1]
block_n = 128 if self.quant_type == QuantType.per_1x128 else 32
# Before: align = 64 if inter_dim <= 192 else block_n  （从 BF16 错误复制）
# After:  always align = block_n（FP8 stage2 只支持 KPerBlock=128）
align = block_n
```

结果：tp=8 inter=160 → **256**（原为 192，修正）。

> **注**：此 commit 之前存在两个中间尝试（`270fee71` 尝试 align=64，`3696345e` revert），最终 `acff926d` 给出正确解。

#### 验证状态

fix 已提交并通过单元验证。tp=8 端到端因 GPU5 硬件异常（~700ms/tensor，iommu=pt 缺失）尚未完成 e2e 验证。

---

## 二、aiter repo

**路径**：`/home/hanchang/junlin12_repos/aiter`
**主要改动文件**：`aiter/fused_moe.py`、`aiter/dist/parallel_state.py`、`aiter/ops/triton/gluon/pa_decode_gluon.py`、`csrc/ck_gemm_moe_2stages_codegen/`、`aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv`

---

### aiter-1：`6d70f7b54` — SwigluStep enum + C++ codegen + CK submodule bump

**日期**：2026-04-23
**改动文件**：7 files（C++ enum、codegen python、pybind11、CK submodule）

#### 问题根因

原 CK stage1 入口用 `activation = !activation;` 实现 enum 反向（`Silu(0)→1, Gelu(1)→0`）。新增 `SwigluStep=3` 后 `!3=0=Gelu`，激活函数被错映射。

codegen 中 ActOP 为 bool，无法表示三元状态；str-to-enum 用 `capitalize()` 无法还原 `SwigluStep`（CamelCase）。

#### 修改内容

**`csrc/include/aiter_enum.h`**：新增枚举值：

```cpp
enum class ActivationType : int {
    No=0, Silu=0, Gelu=1, Swiglu=2,
    SwigluStep = 3,    // 新增
};
```

**`csrc/include/rocm_ops.hpp`**：pybind11 export 添加 `.value("SwigluStep", ...)`。

**`gemm_moe_ck2stages.cu`**：用显式 switch 替换 `!activation`：

```cpp
static inline int map_activation_to_ck_stage1(int activation) {
    switch(static_cast<ActivationType>(activation)) {
    case ActivationType::Silu:       return 1;
    case ActivationType::Gelu:       return 0;
    case ActivationType::SwigluStep: return 2;
    default: TORCH_CHECK(false, ...);
    }
}
// L121:
activation = map_activation_to_ck_stage1(activation);
// Stage2 直接删除 activation = !activation;（stage2 不跑 activation）
```

**`gemm_moe_ck2stages_common.py`**：ActOP 由 bool 升级为 int，新增 `ACT_OP_MAP = {"gelu":0, "silu":1, "swiglustep":2}`。

**`gen_instances.py`**：codegen 新增 swiglustep 循环，生成 preshuffle=True/False 两种实例：

```python
for (b_dtype, routed_weight, preshuffle) in itertools.product(
        b_quant_dtypes, routed_weight_l, [True, False]):
    codegen = ck_moe_2stage_gemm_codegen(..., "swiglustep", ...)
```

**`aiter/utility/dtypes.py`**：`str2ActivationType` 改为显式 mapping。

**CK submodule**：`3rdparty/composable_kernel` bump 至含 `defd7ad29`（见第三章）。

#### 验证

随 v02_exp1_swiglu.md 验证通过（V02 PASS）。

---

### aiter-2：`68fc7d48b` — gfx950 MoE pipeline 修复（V1→V3 强制 + Python 端 SwigluStep）

**日期**：2026-04-23
**改动文件**：`aiter/fused_moe.py`、`aiter/dist/parallel_state.py`、`aiter/jit/utils/moe_recipes.py`

#### 问题根因

1. **V1 CK kernel 在 inter_dim>192 时输出错误**：decode 场景 `block_m<128` 触发 V1 kernel，inter_dim>192 时（>3 次 N-tile pass）GEMM 结果错误（gfx950 tile 错位）。
2. **CustomAllreduce 在 gfx950 + tp=2 产生 NaN**：RCCL `ncclCommInitRank` hang，默认需关闭。
3. **preshuffle 推断仅对 fp4x2 生成两种模式**：swiglustep+no-quant 只编 preshuffle_off，gfx950 又需要 preshuffle_on。

#### 修改内容

**`fused_moe.py` L905-913**：V1→V3 强制：

```python
# gfx950 workaround: V1 kernel produces wrong results for inter_dim>192
if not run_1stage and inter_dim > 192 and get_gfx() == "gfx950" \
        and q_type not in (QuantType.per_1x128, QuantType.per_1x32):  # 见 aiter-5
    block_m = 128
    kernelName2 = "moe_ck2stages_gemm2_256x128x128x64_1x4_TypeCast_v3_..."
```

**`parallel_state.py` L1079**：默认关闭 CustomAllreduce：

```python
_ENABLE_CUSTOM_ALL_REDUCE = False
```

**`fused_moe.py`**：新增 Python 端 SwigluStep 参考实现：

```python
def swiglustep(x_glu, x_linear, limit: float = 7.0):
    x_glu = F.silu(x_glu)
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    return x_glu * x_linear
```

**`moe_recipes.py`**：preshuffle 推断扩展至 swiglustep+no-quant：

```python
if activation == "swiglustep" and quant_type == "no":
    return [True, False]
```

#### 验证

V01 PASS（cos_sim=0.999989~0.999990，tp=2 e2e 正常）。

---

### aiter-3：`3771835ac` — revert 不必要的 +2 row padding

**日期**：2026-04-23
**改动文件**：`aiter/fused_moe.py`（+5/-23）

#### 背景

`68fc7d48b` 中额外引入了 `moe_out_padded = torch.zeros((M+2, model_dim)...)` 和对应的 buffer/view 调整。通过 canary 实验证明：CK kernel 在 `expert_id == sentinel` 时跳过整个 block，sentinel token `(K<<24)|T` 永不触发越界 scatter，+2 padding 对正确性无贡献。

#### 修改内容

还原为直接传 `moe_buf`，a2 buffer 由 `token_num+2` 改回 `token_num`：

```python
# Before:  moe_out_padded = torch.zeros((M+2, model_dim)...)
#           a2 = torch.zeros((token_num+2, topk, inter_dim)...)
# After:    直接使用 moe_buf，a2 = torch.empty((token_num, topk, inter_dim)...)
```

#### 验证

V01 回归测试通过。

---

### aiter-4：`7ebae9afb` — sliding window mask off-by-one 修复

**日期**：2026-04-24
**改动文件**：`aiter/ops/triton/gluon/pa_decode_gluon.py`（1 行）

#### 问题根因

`paged_attention_decode_sliding_window` 的 `IS_CAUSAL=False` 分支下界多了 `+ 1`，实际有效窗口长度变为 `SLIDING_WINDOW - 1`。

`ctx < window` 不受影响；`ctx >= window` 时每步 decode 丢最早 token，误差随解码长度累积。

#### 修改内容

```python
# aiter/ops/triton/gluon/pa_decode_gluon.py L1502
- >= sequence_start_idx + query_token_idx[:, None] + 1
+ >= sequence_start_idx + query_token_idx[:, None]
```

（`decode` 时 `query_token_idx=0`，`+ 1` 即多排除了窗口边界 token）

#### 验证

| ctx_len | 修复前 cos_sim | 修复后 cos_sim |
|---------|-------------|-------------|
| 508-511 | 0.999998 | 0.999998 |
| 512 | 0.998982 **FAIL** | 0.999998 PASS |
| 513-516 | 0.998942~0.999016 FAIL | 0.999998 PASS |
| 1024 | 0.999092 FAIL | 0.999998 PASS |

tp=2 推理中"ungi"等乱码消失。**验证文档**：`verification_pipeline/results/v03_sliding_window.md`

---

### aiter-5：`7312ea166` — gfx950 分布式 allreduce/allgather 兼容性修复

**日期**：2026-04-24
**改动文件**：3 files（`communicator_pynccl.py`、`parallel_state.py`、`communication.py`）

#### 问题根因

1. `ncclCommInitRank` 在 gfx950 + world_size=8 RCCL 下 hang
2. 关闭 CustomAllreduce（`ca_comm=None`）后，`_all_gather_out_place` 仍硬 `assert ca_comm is not None`
3. IPC 自定义 allreduce 在 gfx950 hang；signal buffer 初始化在 `ca_comm=None` 时 AttributeError

#### 修改内容

**`communicator_pynccl.py`**：env var fallback：

```python
if os.environ.get("AITER_PYNCCL_SKIP", "0") == "1":
    self.available = False
    self.disabled = True
    return
```

**`parallel_state.py` L491-510**：`ca_comm=None` guard + NCCL all_gather fallback：

```python
if ca_comm is None:
    output_tensor = torch.empty((world_size,) + input_size, ...)
    torch.distributed.all_gather_into_tensor(output_tensor, input_, ...)
    return output_tensor.movedim(0, dim).reshape(...)
```

**`communication.py`**：`set_custom_all_reduce(False)`；signal buffer 初始化包在 `if ca_comm is not None:` 内。

#### 验证

tp=4 端到端运行正常，`ca_comm=None` 路径无 assert 失败。

---

### aiter-6：`c38d0c9e6` — FP8 blockscale 排除 V1→V3 强制的 guard

**日期**：2026-04-24
**改动文件**：`aiter/fused_moe.py`（+4/-1）

#### 问题根因

`aiter-2` 引入的 V1→V3 强制 `block_m=128` workaround 无条件触发。但 a8w8blkscale dispatch（per_1x128/per_1x32）只支持 `block_m=16/32/64`，传 128 触发 `TORCH_CHECK: Unsupported block_m value: 128`，FP8 模型推理失败。

#### 修改内容

```python
# Before:
if not run_1stage and inter_dim > 192 and get_gfx() == "gfx950":
# After:
if not run_1stage and inter_dim > 192 and get_gfx() == "gfx950" \
        and q_type not in (QuantType.per_1x128, QuantType.per_1x32):
```

FP8 tp=2 inter_dim=640（640%128=0），blockscale dispatch 用 block_m=64，不受 V1 bug 影响。

#### 验证

FP8 tp=2 TTFT=87ms / TPOT=14ms。**验证文档**：`verification_pipeline/results/v05_fp8_inference.md`

---

### aiter-7：`a2883ab37` — 删除 buggy ASM kernel tuning 条目

**日期**：2026-04-25
**改动文件**：`aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv`（删除 1 行）

#### 问题根因

tp=4 prefill 时 `o_proj`（N=4096, K=2048）的 bf16 GEMM 会走 tuned dispatch：

```
TunedGemm.mm → get_padded_m(M, 4096, 2048) → M∈[8193,16384] 时 padded_M=16384
→ 命中 csv L45: libtype=asm, kernel=bf16gemm_bf16_tn_256x256
→ 该 ASM kernel 对非 256 对齐的 M（如 8209-8223）输出错误数值
→ attention 输出损坏 → logits 退化 → argmax=0=BOS → 全 BOS spam
```

M≤8192 不命中该行，走 torch.mm fallback，正常。tp=2（K_in=4096 不同）不触发。

#### 修改内容

删除 `glm5_bf16_tuned_gemm.csv` 原 L45 整行：

```
gfx950,256,16384,4096,2048,...,asm,...,_ZN5aiter24bf16gemm_bf16_tn_256x256E,...
```

所有 (M, 4096, 2048) bf16 GEMM 回退 `torch.mm`。

> 根本修复（ASM kernel M boundary 处理）需 AMD aiter 团队介入，作为 workaround 删除该 tuning 行。

#### 验证

| 验证项 | 修复前 | 修复后 |
|--------|--------|--------|
| tgemm M=8209 diff vs torch | 197.38 | **0** |
| tgemm M=8214 diff | 392 | **0** |
| E2E 10021 tokens first_token | 0 (BOS) | 3648 ("好的") |
| 短 prompt 4 examples | OK | OK |

**验证文档**：`verification_pipeline/results/v07_longseq_bos.md`

---

### aiter-8/9/10：`a2247989d` + `dd4257d8f` + `0f8164017` — stage1 NPerBlock=64 blockscale kernels

**日期**：2026-04-27
**改动文件**：`csrc/ck_gemm_moe_2stages_codegen/gen_instances.py`、`gemm_moe_ck2stages_common.py`

#### 背景与目标

消除 FP8 tp=4（inter_dim=320）依赖 ATOM 端 zero-padding 的问题：若能在 CK codegen 中添加 NPerBlock=64 实例，320=64×5 可整除，无需 padding。

#### 三连提交过程

**`a2247989d`（引入 NPerBlock=64/KPerBlock=64 dispatch）**：

在 `gen_instances.py` `A8W8_blockscale_gemm1/gemm2_heuristic_dispatch` 模板新增 `if (inter_dim % 128 != 0 && inter_dim % 64 == 0)` 分支：

- Stage1 NPerBlock=64：block_m=16/32/64 三档
- Stage2 KPerBlock=64：block_m=16/32/64 三档

**`dd4257d8f`（注册 kernel instance）**：

在 `gemm_moe_ck2stages_common.py` 的两个 kernel list 各添加 entry 4/5/6（stage1 NPerBlock=64，stage2 KPerBlock=64）。

**`0f8164017`（revert stage2，保留 stage1）**：

发现 gfx950 FP8 mfma `static_assert(KPerThread % KPack == 0)` 在 KPack=32、KPerBlock=64 时失败（KPerThread=KPerBlock/WarpCount/KPack=8 < 1），stage2 KPerBlock=64 在 gfx950 FP8 不可编译。

同时发现 stage1 block_m=64 的 V3 需要 MRepeat≥4，而 MPerBlock=64+MWaves=2 只给 MRepeat=2，退回 V1：

```python
# Before: kernelInstanceGEMM1(256, 64, 64, 128, 2, 2, 3)  # V3
# After:  kernelInstanceGEMM1(256, 64, 64, 128, 1, 4, 1)  # V1
# Stage2 KPerBlock=64 entries 全部删除，注释 KPack=32 约束
```

#### 最终状态

- stage1 NPerBlock=64（block_m=16/32/64）：**保留**，支持 `inter_dim % 64 == 0` 的路径
- stage2 KPerBlock=64：**已删除**（gfx950 FP8 不支持）
- ATOM 端 padding（inter_dim=320→384）暂时保留（stage2 仍需 128 对齐）

**验证文档**：`project_summary/step35-flash-support/10_fp8_mfma_kpack32_constraint.md`

---

## 三、composable_kernel repo

**路径**：`/home/hanchang/junlin12_repos/composable_kernel`
**分支**：`feat/swiglustep-moe-no-quant`（领先 `origin/develop` 1 commit）
**base**：`fdf4bb7fc`（ROCm/rocm-libraries#6653）

---

### CK-1：`defd7ad29` — 添加 swiglustep_and_mul epilogue 分支

**日期**：2026-04-23
**作者**：Jun Lin \<junlin12@amd.com\>
**改动文件**：`include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm.hpp`（+76，全 additive）

#### 问题根因

`GridwiseMoeGemm` 的 epilogue activation 分发只有 `silu_and_mul` 和 `gelu_and_mul` 两条分支，无法支持 SwigluStep（silu + clamp）激活。

#### 修改内容

向 4 个位置（quant/non-quant × 两组 pipeline）各添加一个 `else if constexpr(ActivationOperation == Activation::swiglustep_and_mul)` 分支，算子实现完全一致：

```cpp
Silu{}(gate, gate);
gate = gate < 7.0f ? gate : 7.0f;                            // upper clamp
up   = up > -7.0f ? (up < 7.0f ? up : 7.0f) : -7.0f;       // ±7 clamp
c_thread_buf_fp32(cidx) = gate * up;
```

**关键设计约束**：7.0f 硬编码，这也是 ATOM 中 shared expert（limit=16 != 7）无法复用此 kernel 的直接原因。

#### 验证

随 aiter SwigluStep 整体验证通过（V02 PASS，cos_sim ≥ 0.99999）。

---

## 四、跨 repo 修改关系图

```
composable_kernel
  └── defd7ad29 (swiglustep_and_mul)
        ↓ submodule bump
aiter
  ├── 6d70f7b54 (SwigluStep enum + codegen)        ← 依赖 CK defd7ad29
  ├── 68fc7d48b (V1→V3 force + SwigluStep Python)  ← 修 gfx950 MoE correctness
  ├── 3771835ac (revert +2 padding)                 ← 撤销 68fc7d48b 的冗余改动
  ├── 7ebae9afb (sliding window off-by-one)         ← 独立 bug
  ├── 7312ea166 (ca_comm None guard)                ← 独立 bug，tp>1 必须
  ├── c38d0c9e6 (FP8 blockscale guard)              ← 依赖 68fc7d48b 存在
  ├── a2883ab37 (删除 buggy CSV 行)                  ← 独立 bug，tp=4 长序列
  └── 0f8164017 (stage1 NPerBlock=64)               ← 探索性，stage2 受限未完成

ATOM
  ├── ec8cbe87 (Step3p5 模型 + preshuffle fix)      ← 基础，其余 commits 依赖此
  ├── 4a8495ec (SwigluStep per-layer wiring)        ← 依赖 aiter 6d70f7b54
  ├── 635e59e9 (BF16 inter_dim padding)             ← 依赖 aiter 68fc7d48b（V3 force）
  ├── 9a67e493 (FP8 quant_config block_shape)       ← FP8 基础准备
  ├── ccb64621 (FP8 tp=4 三处修复)                  ← 依赖 635e59e9
  └── acff926d (FP8 blockscale align fix)           ← 修正 ccb64621 的遗漏
```

---

## 五、验证结果汇总

| 验证专题 | 关键修复 | 验证方法 | 关键指标 | 结论 |
|---------|---------|---------|---------|------|
| V01 preshuffle | ATOM ec8cbe8 | cos_sim（preshuffle_on/off） | preshuffle_on=0.99999 | **PASS** |
| V01 V1→V3 | aiter 68fc7d48b | inter_dim 矩阵（192/384/640） | 全 ≥0.99999 | **PASS** |
| V02 SwigluStep | ATOM 4a8495e + CK defd7ad29 | 12 cases cos_sim | ≥0.99999 | **PASS** |
| V03 sliding window | aiter 7ebae9afb | cos_sim ctx sweep | ctx≥512 从 FAIL→PASS | **PASS** |
| V04 tp=4/8 | ATOM 635e59e + aiter 7312ea166 | tp=4 e2e TTFT | 81ms | **PASS** |
| V05 FP8 tp=2 | aiter c38d0c9e6 + ATOM 9a67e49 | FP8 tp=2 e2e | TTFT=87ms | **PASS** |
| V06 FP8 tp=4 | ATOM ccb64621 | FP8 tp=4 e2e | TTFT=86ms TPOT=13ms | **PASS** |
| V07 长序列 BOS | aiter a2883ab37 | tgemm diff=0 + 10k e2e | first_token=3648 | **PASS** |

---

## 六、已知 Open Items

| 项目 | 状态 | 说明 |
|------|------|------|
| FP8 tp=8 e2e | 受阻 | GPU5 硬件异常（~700ms/tensor），需 sysadmin 修复 iommu=pt |
| ASM kernel 根本修复 | 待提 bug | `bf16gemm_bf16_tn_256x256` 在 gfx950 非对齐 M 时输出错误，需 AMD 团队修复 |
| stage2 KPerBlock=64 | 受限 | gfx950 FP8 KPack=32 约束，需 CK 上游支持或换 tile 策略 |
| FP8 tp=4 无 padding | 下一任务 | inter_dim=320 目前仍需 padding 至 384，候选方案 C（NPerBlock=64 stage2 调整）待验证 |
| V06 Exp1c (oversharding) | P0 未跑 | 验证 FP8 tp=4 rank 4-7 在 oversharding 场景不 crash |
