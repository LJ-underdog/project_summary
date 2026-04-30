# V04 — TP Support 验证计划

**专题组**：V04-TP Support（Code Reviewer + Experiment Designer）
**修复来源**：04_tp_support.md
**关联 commits**：
- ATOM `635e59e`：moe.py inter_dim padding（tp=4/tp=8 fused MoE 对齐）
- aiter `7312ea166`：parallel_state.py ca_comm None guard（NCCL fallback）

**辅助文件**：
- `/home/hanchang/ATOM/atom/model_ops/moe.py` L483-525（`UnquantizedFusedMoEMethod.process_weights_after_loading`）
- `/home/hanchang/aiter/aiter/dist/parallel_state.py` L492-514（`_all_gather_out_place`）
- `/home/hanchang/aiter/aiter/dist/parallel_state.py` L582-593（参考：`all_gather` NCCL 路径）

---

## A. Code Review

### A.1 Fix 1（ATOM `moe.py` L495-518）：inter_dim padding

#### A.1.1 Align 规则（≤192 用 64，>192 用 128）的数学推导

**代码摘录**（moe.py:498-503）：
```python
# Stage1 dispatch: inter<=192 uses NPerBlock=64, inter>192 uses NPerBlock=128.
# Stage2 dispatch: inter>192 uses KPerBlock=64.
# So required alignment: 64 when inter<=192, 128 when inter>192.
align = 64 if inter_dim <= 192 else 128
inter_pad = (inter_dim + align - 1) // align * align
```

**推导分析**：
- Stage1（w13 GEMM, output dim = 2·inter）的 CK kernel dispatch 由 inter_dim 大小驱动：
  - inter ≤ 192 → 选 NPerBlock=64 的 tile
  - inter > 192 → 选 NPerBlock=128 的 tile
- Stage2（w2 GEMM, K dim = inter）：inter > 192 时 KPerBlock=64
- 综合约束：取两阶段对应 BlockSize 的 LCM 上界，简化为「stage1 NPerBlock」即可：
  - tp=8: inter=160 ≤ 192，align=64，pad → 192（192 % 64 = 0 ✓）
  - tp=4: inter=320 > 192，align=128，pad → 384（384 % 128 = 0 ✓，且 384 % 64 = 0 ✓）

**为何不能都用 align=128？**
- tp=8 inter=160：若强制 align=128，pad 到 256，多分配 256/160 ≈ 1.6× 内存与计算量
- tp=8 inter=160 → 192 仅 +20% padding，inter=160 → 256 +60% padding
- 等价的 stage1 kernel 在 inter ≤ 192 时本身就用 NPerBlock=64，没有「全部对齐 128」的必要
- 推论：选最小满足两阶段 dispatch 约束的 align，是「精确对齐」而非保守对齐

**Review 结论**：✓ 数学上自洽。**但**：
- ⚠️ 分支条件 `inter_dim <= 192` 是个「魔术值」。应在 code review 中要求 inline 注释指向 CK kernel manifest（确认 192 是 NPerBlock 切换边界，而不是经验值）

#### A.1.2 w13 与 w2 padding 方向

**代码摘录**（moe.py:505-516）：
```python
E, _, hidden = w13.shape
# pad w13: gate half [E, inter, hidden] and up half [E, inter, hidden]
w13_new = torch.zeros(E, 2 * inter_pad, hidden, dtype=w13.dtype, device=w13.device)
w13_new[:, :inter_dim, :] = w13[:, :inter_dim, :]                   # gate
w13_new[:, inter_pad : inter_pad + inter_dim, :] = w13[:, inter_dim:, :]  # up
# pad w2: [E, hidden, inter_pad]
w2_new = torch.zeros(E, hidden, inter_pad, dtype=w2.dtype, device=w2.device)
w2_new[:, :, :inter_dim] = w2
```

**张量布局回顾**：
- w13: `[E, 2*inter, hidden]`，前 `inter` 行 = gate，后 `inter` 行 = up
- w2:  `[E, hidden, inter]`，第 2 维（K 维）是 inter

**Padding 方向分析**：
| 张量 | 原 shape | Pad 后 shape | Pad 维度 | 是否「N 维」 |
|------|---------|------------|---------|------------|
| w13  | [E, 2·inter, hidden] | [E, 2·inter_pad, hidden] | dim=1（行） | ✓ 输出维（N） |
| w2   | [E, hidden, inter] | [E, hidden, inter_pad] | dim=2（列） | ✗ 输入维（K） |

- **w13 在 N 维 pad**（输出/中间维度扩张）：✓ 正确。stage1 输出 `gate, up` 多产生 padding 列，但它们将进入 stage2 的 K 维与 w2 的 0 行相乘 → 贡献为 0
- **w2 在 K 维 pad**（输入维度扩张）：✓ 正确。padded K 槽位的权重为 0，无论输入是什么，对输出的贡献为 0

**关键不变量**：gate 和 up 半段的「相对偏移」从 `inter_dim` 改为 `inter_pad`。这要求 stage1/stage2 的 fused_moe kernel 也按 `inter_pad`（而非原 `inter_dim`）解释 layout。**这是 padding 能 work 的隐含前提**：kernel 必须读取 weight tensor 的 actual shape 而非缓存的 config inter_dim。

**Review 结论**：✓ 方向正确，但需在实验 4 验证 kernel 是否的确按 weight.shape 推导 inter（而非从 config）。

#### A.1.3 Padding 部分填零，对 stage2 输出的影响

**注释声明**（moe.py:489-492）：
> Zero padding is safe because fused_moe clips routed-weight contributions and zero-padded rows contribute nothing.

**分析**：
- Stage1 输出 `(gate, up)` 中 padding 段（索引 [inter_dim, inter_pad)）：
  - gate_padding = x @ 0 = 0
  - up_padding = x @ 0 = 0
  - SiLU(0) * 0 = 0 → 进入 stage2 的中间激活 padding 段为 0
- Stage2: `out = act @ w2`，K 维 padding 槽位的 act 为 0（即使 w2 padding 槽位非 0 也不影响）
- 双重保险：act_padding=0 和 w2_padding=0，任何一个为 0 都使该项贡献为 0

**潜在风险**：
- **Numerical**：如果 stage1 后做了 RMSNorm/LayerNorm（包含 padding 段），padding 列影响均值/方差。**确认 fused MoE 不做 inter-dim normalization**（仅 SiLU·gate*up，逐元素操作）→ 安全
- **Routed-weight scaling**（topk_weights 乘到激活上）：作用于 hidden 维，不涉及 inter，安全
- **Reduce/sum 路径**：stage2 的输出回到 hidden 维，不会 sum over inter

**注释中「fused_moe clips routed-weight contributions」**：这指的是 fused_moe.py 在 K 维上的 reduction 不会把 padding K 拿去做意外的归一化。**需要在实验中确认**（建议用 cos_sim 与 torch reference 对比，目标 ≥ 0.9999，注释中已声明 2026-04-24 验证过）。

**Review 结论**：✓ 逻辑正确。但需在 experiment 中复现注释里的 cos_sim ≥ 0.9999 数据，避免「相信注释」。

---

### A.2 Fix 2（aiter `parallel_state.py` L492-514）：ca_comm None guard

#### A.2.1 None guard 逻辑完整性

**代码摘录**（parallel_state.py:492-514）：
```python
def _all_gather_out_place(self, input_: torch.Tensor, dim: int = 0) -> torch.Tensor:
    ca_comm = self.device_communicator.ca_comm
    if ca_comm is None:
        # custom all-reduce disabled (e.g. gfx950): fall back to NCCL all-gather
        world_size = self.world_size
        input_size = input_.size()
        output_tensor = torch.empty(
            (world_size,) + input_size, dtype=input_.dtype, device=input_.device
        )
        torch.distributed.all_gather_into_tensor(
            output_tensor, input_, group=self.device_group
        )
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim]
            + (world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )
        return output_tensor
    assert not ca_comm.disabled
    out = ca_comm.custom_all_gather(input_, dim)
    assert out is not None
    return out
```

**完整性 check**：
- ✓ Guard 在函数顶部（早返回），覆盖所有后续路径
- ✓ Fallback 不依赖 `ca_comm` 任何属性
- ⚠️ **未 guard 的相关调用**：`L363 ca_comm = self.device_communicator.ca_comm; if ca_comm is not None: maybe_ca_context = ca_comm.capture()` —— 这里已有 None guard。
- ⚠️ **同文件 L484 `device_communicator.fused_allreduce_rmsnorm_quant`**：未在本次 fix 范围内，但若 `device_communicator` 自身为 None 会更早 raise。

**遗漏 check**：grep 全文件 `ca_comm.` 调用，确认每处都有 guard。

#### A.2.2 NCCL fallback 路径与原 L567-576（all_gather）的一致性

**比对**：
| 步骤 | `_all_gather_out_place`（L498-510，新增） | `all_gather` NCCL 路径（L583-592，参考） |
|------|---------------------------------------|----------------------------------|
| 1. 分配 output | `torch.empty((world_size,) + input_size, ...)` | 同 ✓ |
| 2. all_gather_into_tensor | 同 ✓ | 同 ✓ |
| 3. movedim(0, dim) | 同 ✓ | 同 ✓ |
| 4. reshape | `input_size[:dim] + (world_size * input_size[dim],) + input_size[dim+1:]` | 完全相同 ✓ |

**结论**：✓ 完全一致，可视为复用 `all_gather` 的 NCCL 路径片段。

**潜在问题**：
- 代码重复（同样的 NCCL fallback 在两处）。建议（不在本次 fix 范围内）：抽出 `_nccl_all_gather(input_, dim)` 共享。
- `_all_gather_out_place` 的语义是「out-place 版本」；NCCL 路径本身就是 out-of-place，命名一致。

#### A.2.3 tp=8 未端到端验证的影响评估

| 结论 | 来源 | 可信度 |
|------|------|--------|
| inter=160 → 192 满足 192 % 64 = 0 | 代码（数学） | 推断 ✓（几何上必真） |
| stage1/stage2 kernel 接受 192 inter 不 crash | **代码推断** | ⚠️ 仅 inter=192 单算子未跑过 |
| tp=8 全模型 forward 正确 | **未实测** | ⚠️ 因 GPU5 硬件阻塞 |
| ca_comm fix 在 tp=8 路径正确 | **未实测** | ⚠️ tp=4 已经 trigger 了同一路径，但 tp=8 communication pattern 不同 |
| tp=4 BF16 4 prompts max_tokens=128 通过 | summary | ✓ 实测 |

**风险**：
- tp=8 的 ca_comm None guard 路径可能与 tp=4 不同（rank 数不同会影响 NCCL group 选择）
- inter=192 padding 在大模型 forward 中的累积误差未验证

---

## B. Experiment Design

### 实验 1：inter_dim 对齐验证（最小复现）

**目标**：在隔离环境中验证 align 规则（≤192→64，>192→128）的边界。

**方法**：直接调用 fused MoE GEMM，构造 expert weight，分别测 4 个 inter_dim：

| 配置 | tp=8 模拟 | tp=4 模拟 |
|------|----------|----------|
| 不对齐输入 | inter=160（pre-fix） | inter=320（pre-fix） |
| 对齐输出 | inter=192（post-fix） | inter=384（post-fix） |

**期望结果**：
- inter=160 → CK kernel crash 或 wrong-shape error
- inter=192 → pass，cos_sim vs torch reference ≥ 0.9999
- inter=320 → CK kernel crash
- inter=384 → pass，cos_sim ≥ 0.9999

**脚本草案**（伪码）：
```python
import torch
from aiter import fused_moe_ck

for E, inter, hidden in [(8, 160, 4096), (8, 192, 4096),
                          (8, 320, 4096), (8, 384, 4096)]:
    w13 = torch.randn(E, 2*inter, hidden, dtype=torch.bfloat16, device='cuda')
    w2  = torch.randn(E, hidden, inter, dtype=torch.bfloat16, device='cuda')
    x   = torch.randn(16, hidden, dtype=torch.bfloat16, device='cuda')
    try:
        out_ck = fused_moe_ck(x, w13, w2, ...)
        out_ref = torch_ref_moe(x, w13, w2, ...)
        cos = torch.nn.functional.cosine_similarity(
            out_ck.flatten(), out_ref.flatten(), dim=0)
        print(f"inter={inter}: PASS, cos_sim={cos:.6f}")
    except Exception as e:
        print(f"inter={inter}: CRASH, {e}")
```

**通过标准**：边界完全符合上表。

---

### 实验 2：tp=4 端到端（核心验证）

**命令**：
```bash
cd /tmp && rm -rf /root/.cache/atom/* && CUDA_VISIBLE_DEVICES=0,1,2,3 \
  AITER_LOG_LEVEL=WARNING \
  /opt/venv/bin/python -m atom.examples.simple_inference \
  --model stepfun-ai/Step-3.5-Flash --kv_cache_dtype bf16 --trust-remote-code \
  --tensor-parallel-size 4 --level 0 --temperature 0 --max-tokens 128 \
  --gpu-memory-utilization 0.7 --max-num-batched-tokens 4096 --max-num-seqs 256
```

**通过标准**：
1. 4 prompts 全部正常完成（exit code 0）
2. 无 crash、无 hang
3. 输出无乱码（人工检查 4 个回答的语义合理性）
4. 无 AITER ERROR-level log

**回归比对**：
- 与 BF16 tp=4 baseline（memory 中标注 TTFT=88ms，TPOT=15.75ms）对比
- 性能不应回退（padding 增加 384/320 = 1.2× 计算量；TPOT 允许 ≤20% 回退）

---

### 实验 3：tp=2 回归（确认不受 inter_dim padding 影响）

**目标**：tp=2 时 inter=640，640 % 128 = 0，触发 `inter_pad == inter_dim` 分支跳过 padding。

**命令**：
```bash
cd /tmp && rm -rf /root/.cache/atom/* && CUDA_VISIBLE_DEVICES=0,1 \
  AITER_LOG_LEVEL=WARNING \
  /opt/venv/bin/python -m atom.examples.simple_inference \
  --model stepfun-ai/Step-3.5-Flash --kv_cache_dtype bf16 --trust-remote-code \
  --tensor-parallel-size 2 --level 0 --temperature 0 --max-tokens 128 \
  --gpu-memory-utilization 0.9
```

**通过标准**：
- 与 fix 之前完全一致的输出（bit-exact 或 cos_sim = 1.0）
- TTFT/TPOT 不应有任何变化（memory baseline: TTFT=92ms, TPOT=17ms）
- 验证手段：在 `process_weights_after_loading` 加 print，确认 `inter_pad == inter_dim` 分支命中（或事后 grep）

---

### 实验 4：ca_comm None guard 验证

**问题**：如何在 gfx950 上 trigger None guard 路径？

**回答**：gfx950 上 custom all-reduce 默认 disabled，`ca_comm` 即为 None。任何 tp>1 的 all-gather 调用都会走新路径。

**验证方法**：

**方法 A（被动验证）**：实验 2 或 3 通过即证 None guard 不 crash。

**方法 B（主动验证）**：在 `_all_gather_out_place` 函数顶部插入 `assert ca_comm is None, "expected None on gfx950"; print("[fallback] hit NCCL path")`，运行 tp=4 simple_inference，确认 stdout 出现该 print（≥ 1 次/forward step）。

**方法 C（数值验证）**：写单元测试，直接调用 `_all_gather_out_place(input, dim=0/1/2)`，对比结果与 `torch.distributed.all_gather` 手动拼接的结果（应 bit-exact）。

**通过标准**：
- 方法 B：每个 forward step 都有 fallback 命中
- 方法 C：bit-exact match

---

### 实验 5：tp=8 代码路径验证（GPU5 异常下的最大化覆盖）

**约束**：GPU5 硬件阻塞，端到端 tp=8 不可行。

**可执行的验证**：

1. **静态检查（已完成）**：
   - inter=160 → 192，192 % 64 = 0 ✓
   - 192 % 128 = 64 ≠ 0，但此分支不要求 128 对齐（因 inter ≤ 192）

2. **替代实验：tp=8 排除 GPU5**：
   ```bash
   # 用 0,1,2,3,4,6,7 + 任一其他设备模拟 8 卡（实际只有 8 卡时不可行）
   # 可改为：CUDA_VISIBLE_DEVICES=0,1,2,3 模拟 tp=4 with inter=160
   #         （需手动 patch 模型 num_experts/inter_dim 以匹配 tp=8 shard）
   ```

3. **MoE kernel-only test on GPU 0-3**：用实验 1 的脚本，只测 inter=160 → 192 的 GEMM 正确性（不需要分布式）。
   - 通过则证明：weight padding 下 kernel 不 crash + 数值正确
   - 仍未覆盖：tp=8 communication pattern 下的 ca_comm fallback

4. **降级 tp=8 → tp=4 with 同 inter**：通过修改 expert sharding 强制每卡 inter=160（即「假 tp=8」on 4 GPUs），可覆盖 inter=160 padding 路径但不覆盖 8-rank NCCL all-gather。

**通过标准**：
- 至少完成 (1)+(3)，证明 inter=192 kernel 正确性
- 报告中明确标注 tp=8 端到端「**未实测，pending GPU5 修复**」

---

## C. 关键问题

### C.1 tp=8 GPU5 硬件问题是否已解决？如否，tp=8 端到端能否测试？

**当前状态**（来自 MEMORY.md）：
> GPU 5 硬件异常（~700ms/tensor），避免使用
> tp=8 ⚠️ GPU5 硬件阻塞

**回答**：**未解决**。tp=8 端到端**不能**用 0-7 全卡测试。

**可行替代**：
- 单算子测试（实验 1、5.3）覆盖 weight padding 正确性
- 「假 tp=8」（实验 5.4）覆盖 inter=160 路径，但牺牲 communication 覆盖
- 等待 GPU5 修复或换机器

**风险声明**：
- 任何 tp=8 的性能数据当前都不可信
- tp=8 ca_comm fix 仅靠 tp=4 实测推断（不同 rank 数下 NCCL group 行为不同）

### C.2 align=128 vs align=64 的边界条件 192 如何确认？

**当前依据**：moe.py:498-499 注释声明 CK kernel dispatch 表（NPerBlock=64 for inter≤192，NPerBlock=128 for inter>192）。

**验证方法**：

1. **Grep CK kernel manifest**：
   ```bash
   grep -rn "NPerBlock.*64\|NPerBlock.*128" /home/hanchang/aiter/csrc/ck_gemm_moe_2stages_codegen/
   ```
   找到 dispatch table，确认 192 是真实切换点。

2. **Brute-force inter sweep**：实验 1 扩展，遍历 inter ∈ {64, 128, 160, 192, 224, 256, 320, 384, 512}，记录每个值需要的最小 align：
   - 若 inter=193 已经要求 align=128，则边界确实是 192
   - 若 inter=224 仍可用 align=64，则边界 > 192（注释错误）

3. **读 CK source**：找 `device_grouped_gemm_xdl_cshuffle_two_stage` 的 instance 定义，看 inter 大小到 NPerBlock 的映射逻辑。

**通过标准**：方法 1 或 3 找到明确证据，否则方法 2 实测 sweep。

### C.3 w2（down projection）padding 后 padded 行为零，stage2 的 K-reduction 行为是否正确？

**问题**：stage2 GEMM `out[hidden] = sum_k(act[k] * w2[hidden, k])`，K 维包含 padded 槽位。

**理论分析**：
- padded K 槽位的 `w2[h, k_pad] = 0`（zero-padded）
- 即使 act[k_pad] ≠ 0（来自 stage1 的 padded gate*up），乘积 = 0
- 求和不变 → 结果数学等价

**潜在 corner case**：
- **FP8/INT8 quantization**：若 padded 0 在 quantize 时与 scale 相乘后非 0（subnormal/round-up），会引入误差。本次 fix 只针对 BF16（unquant 路径），FP8 路径在 moe.py:1701（独立的 process_weights_after_loading）。
- **Atomic accumulation**：若 stage2 用 atomicAdd 累加，浮点求和 0 项也需要正确处理（应安全，但需 confirm）。
- **Block-quantized scale**：若 scale 张量未同步 padding（仅 weight padding），block 边界对不上 → 数值错。需检查 fc2_smooth_scale 是否同步 pad（moe.py:616 注释提到该字段）。

**验证方法**：
- 实验 1 的 cos_sim ≥ 0.9999 即覆盖此问题
- 额外：构造 act 的 padded 段为 NaN，看 output 是否被污染（检测「padded 段实际未参与 reduction」的强语义）

**通过标准**：实验 1 cos_sim ≥ 0.9999 ⇒ 行为正确。

---

## D. 验证执行顺序（推荐）

| 序号 | 实验 | 估计工时 | 依赖 |
|------|------|----------|------|
| 1 | C.2 grep CK manifest 确认 192 边界 | 0.5h | 无 |
| 2 | 实验 3：tp=2 回归（最快、零风险） | 0.5h | 无 |
| 3 | 实验 1：单算子 inter sweep | 1h | 单算子测试脚本 |
| 4 | 实验 2：tp=4 端到端 | 1h | 实验 1 通过 |
| 5 | 实验 4-B：ca_comm fallback 命中验证 | 0.5h | 实验 2 |
| 6 | 实验 5：tp=8 静态 + 单算子 | 1h | 实验 1 |

**总工时**：~4.5h（不含 GPU5 修复后的 tp=8 端到端）

---

## E. 不在本次验证范围内（明确标注）

- tp=8 端到端正确性 + 性能（GPU5 阻塞）
- FP8 路径的 inter_dim padding（属于 V03 / fp8-work.md 范围；本次仅覆盖 BF16 unquant）
- Mori expert-parallel all-to-all（独立 codepath，未受本次 fix 影响）
- 长序列 ≥10k 输出全 BOS 的 open bug（属于 tp48-fixes.md 范围）

V04 TP Support 验证计划完成
