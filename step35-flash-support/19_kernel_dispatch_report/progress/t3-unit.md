# t3-unit — 单 kernel dispatch 验证

> 任务：#103 写并运行最小化测试脚本，直接验证各 kernel 的 dispatch 路径
> 完成时间：2026-04-29
> GPU：CUDA_VISIBLE_DEVICES=4
> 日志：`logs/unit_moe.log`、`logs/unit_tgemm.log`

---

## 1. 测试 A：FP8 blockscale fused_moe dispatch

### 实际运行的脚本（`/tmp/test_moe_dispatch.py`）

```python
import os, torch
os.environ["AITER_LOG_TUNED_CONFIG"] = "1"
os.environ["AITER_LOG_LEVEL"] = "INFO"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import aiter
from aiter.fused_moe import fused_moe
from aiter import ActivationType, QuantType  # 注意：从 aiter 顶层 import，不是 jit.utils.moe_recipes

# Step-3.5-Flash FP8 tp=2 配置
E = 289       # num_experts (含 shared)
topk = 9      # top-k (含 shared)
H = 4096      # hidden
I = 640       # inter_dim tp=2
q_dtype = torch.float8_e4m3fn

# === Test 1: Prefill (M=512) ===
M_prefill = 512
x = torch.randn(M_prefill, H, dtype=torch.bfloat16, device="cuda")
w1 = torch.randn(E, 2*I, H, dtype=torch.bfloat16, device="cuda").to(q_dtype)
w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device="cuda").to(q_dtype)
# blockscale shape：[E, N//128, K//128]（3D，不是 2D）
w1_scale = torch.ones(E, (2*I+127)//128, H//128, dtype=torch.float32, device="cuda")
w2_scale = torch.ones(E, H//128, (I+127)//128, dtype=torch.float32, device="cuda")
topk_weights = torch.ones(M_prefill, topk, dtype=torch.float32, device="cuda") / topk
topk_ids = torch.randint(0, E, (M_prefill, topk), device="cuda", dtype=torch.int32)  # int32, 不是 int64

out = fused_moe(
    hidden_states=x, w1=w1, w2=w2,
    topk_weight=topk_weights, topk_ids=topk_ids,
    w1_scale=w1_scale, w2_scale=w2_scale,
    quant_type=QuantType.per_1x128,   # 启用 blockscale 路径
    activation=ActivationType.Silu,
)

# === Test 2: Decode (M=1) ===
# 同上，仅 M_decode=1
```

运行命令：
```bash
cd /tmp && CUDA_VISIBLE_DEVICES=4 AITER_LOG_LEVEL=INFO AITER_LOG_TUNED_CONFIG=1 \
  /opt/venv/bin/python /tmp/test_moe_dispatch.py 2>&1 \
  | tee /home/hanchang/project_summary/step35-flash-support/19_kernel_dispatch_report/logs/unit_moe.log
```

### 关键 dispatch 日志

**Prefill (M=512, inter=640)**：
```
[aiter] run_1stage = False, ksplit = 0 q_type = QuantType.per_1x128 block_m = 64 use_nt = True, estimated_m_per_expert = 15
[aiter] [fused_moe] using 2stage default for (256, 512, 4096, 640, 289, 9, 'ActivationType.Silu', 'torch.bfloat16', 'torch.float8_e4m3fn', 'torch.float8_e4m3fn', 'QuantType.per_1x128', True, False)
[aiter] start build [module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_1x128_mulWeightStage2]
[aiter] type hints mismatch, override to --> ck_moe_stage1(...)
[aiter] type hints mismatch, override to --> ck_moe_stage2(...)
PASS: out.shape=torch.Size([512, 4096]), dtype=torch.bfloat16
```

**Decode (M=1, inter=640)**：
```
[aiter] run_1stage = False, ksplit = 4 q_type = QuantType.per_1x128 block_m = 16 use_nt = True, estimated_m_per_expert = 0
[aiter] [fused_moe] using 2stage default for (256, 1, 4096, 640, 289, 9, ..., 'QuantType.per_1x128', True, False)
PASS: out.shape=torch.Size([1, 4096])
```

### 结论：Prefill 与 Decode 都走 CK 2-stage（不是 ASM）

- Prefill (M=512)：`run_1stage = False`，**CK 2-stage**，block_m=64，单实验 padded_M=256
- Decode (M=1)：`run_1stage = False`，**CK 2-stage**，block_m=16，ksplit=4
- 两者使用同一个动态构建 module：`module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_1x128_mulWeightStage2`，调用 `ck_moe_stage1` + `ck_moe_stage2`
- 没有任何日志包含 `1stage` / `asm` / `fmoe_fp8_blockscale_g1u1`

### 重要修正：TEAM_CONFIG.md 中的 KNOWN_FACTS 有误

TEAM_CONFIG.md 第 39 行写：
> "MoE routed experts：prefill 走 ASM（fmoe_fp8_blockscale_g1u1，run_1stage=True when token>32 and inter%256==0）"
> "tp=2 inter_dim=640（640%256==0，满足 ASM 条件）"

实测推翻：**640 % 256 = 128（NOT 0）**。`fused_moe.py:883` 的 1stage 触发条件 `token > 32 and inter_dim % 256 == 0` 在 inter=640 时 **不成立**，因此即便 prefill 也走 2stage CK。

代码引用：`/home/hanchang/aiter/aiter/fused_moe.py:881-883`
```python
if q_type == QuantType.per_1x128:
    # for fp8 blockscale, ck has better performance so disable assembly kernel
    run_1stage = token > 32 and (inter_dim % 256 == 0)
```
注释明确说 "for fp8 blockscale, ck has better performance so disable assembly kernel"，即 ASM 路径在 blockscale 场景被刻意限制。

**对 tp=4 inter=384 的推论**：384%256=128，同样不满足 → tp=4 也是 CK 2-stage。该结论需 #102 的 tp=2/tp=4 实际推理日志再次验证。

---

## 2. 测试 B：BF16 linear tgemm dispatch

### 实际运行的脚本（`/tmp/test_tgemm_dispatch.py`）

```python
import os, torch
os.environ["AITER_LOG_TUNED_CONFIG"] = "1"
os.environ["AITER_LOG_LEVEL"] = "INFO"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from aiter.tuned_gemm import tgemm

shapes = [
    (512, 4096, 4096, "O proj prefill"),
    (1, 4096, 4096, "O proj decode"),
    (512, 5120, 4096, "QKV proj prefill"),
    (1, 5120, 4096, "QKV proj decode"),
    (512, 11264, 4096, "dense gate_up prefill"),
]

for M, N, K, desc in shapes:
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    out = tgemm.mm(x, w)
    print(f"{desc} ({M},{N},{K}): shape={out.shape}")
```

### 关键 dispatch 日志（每个 shape 都触发同一条）

```
[aiter] shape is M:512, N:4096, K:4096 dtype='torch.bfloat16' otype='torch.bfloat16' bias=False, scaleAB=False, bpreshuffle=False, not found tuned config in /tmp/aiter_configs/bf16_tuned_gemm.csv, will use default config! using torch solution:0
[aiter] shape is M:1, N:4096, K:4096 ... using torch solution:0
[aiter] shape is M:512, N:5120, K:4096 ... using torch solution:0
[aiter] shape is M:1, N:5120, K:4096 ... using torch solution:0
[aiter] shape is M:512, N:11264, K:4096 ... using torch solution:0
```

### 结论：BF16 linear 全部走 torch.mm

- 5 个 shape 全部 csv miss（包括 glm5_bf16_tuned_gemm.csv），落到 `torch solution:0`
- 验证了 KNOWN_FACTS：`aiter/tuned_gemm.py tgemm.mm → bf16_tuned_gemm.csv 全 miss → torch.mm`
- attn (O/QKV proj)、dense gate_up 三类 BF16 linear 在 prefill 与 decode 场景都走 torch op

---

## 3. 报错与解决方式

| # | 报错 | 原因 | 解决 |
|---|------|------|------|
| 1 | `ImportError: cannot import name 'dtypes' from 'aiter.utility.dtypes'` | aiter API 变化 | 删除该 import（脚本用不到） |
| 2 | `ImportError: cannot import name 'ActivationType' from 'aiter.jit.utils.moe_recipes'` | 实际位置变了 | 改为 `from aiter import ActivationType, QuantType` |
| 3 | `TypeError: fused_moe() got an unexpected keyword argument 'block_shape'` | API 没有 `block_shape` 参数；blockscale 通过 `quant_type=QuantType.per_1x128` 隐式启用（block_n/block_k 写死 128） | 把 `block_shape=[128,128]` 改为 `quant_type=QuantType.per_1x128` |
| 4 | `RuntimeError: CKPyInterface: Unsupported data type 4` | topk_ids 默认 int64，moe_sorting 仅支持 int32 | `torch.randint(..., dtype=torch.int32)` |
| 5 | scale shape 不对 | per_1x128 要求 3D `[E, N//128, K//128]` | 把 2D 改为 3D |

修正后两个测试都 PASS。

---

## 4. 总结结论

| 操作 | Prefill (M=512) | Decode (M=1) | 备注 |
|------|------|------|------|
| FP8 MoE (per_1x128, inter=640) | **CK 2-stage** (`ck_moe_stage1` + `ck_moe_stage2`)，block_m=64 | **CK 2-stage**，block_m=16, ksplit=4 | inter=640 不满足 inter%256==0，1stage ASM 路径不触发 |
| BF16 linear (O/QKV/gate_up) | **torch.mm** (csv miss) | **torch.mm** (csv miss) | bf16_tuned_gemm.csv 没有这些 shape 的 entry |

**最关键发现**：Step-3.5-Flash-FP8 tp=2 的 routed expert 即使 prefill 也走 CK 2-stage 而非 ASM 1stage，与 TEAM_CONFIG 写的 KNOWN_FACTS 不一致。需要在 REPORT.md 中纠正。
