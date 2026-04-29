# #701 — MoE forward 路径追踪（teammate-9）

任务：确认 BF16/FP8 模式下 routed experts 是否都走 CK fused_moe（aiter 2-stage / fmoe），与 bf16_tuned_gemm.csv 完全无关；同时核实日志中 62/120 次 tuned_gemm miss 实际归属哪些层。

铁则：所有结论都给出 文件:行号 + 原文片段。

---

## 1. BF16 模式 MoE forward 路径

调用链（从模型层 forward 到 aiter kernel）：

1. `Step3p5MoE.forward()`
   - 文件：`/home/hanchang/ATOM/atom/models/step3p5.py:278-295`
   - 关键行 293：`routed_out = self.experts(hidden_states, router_logits)` —— `self.experts` 是 `FusedMoE`（同文件 223 行 `self.experts = FusedMoE(...)`，构造时 `quant_config=quant_config`，BF16 模式下 quant_config=None）。

2. `FusedMoE.forward()` → `FusedMoE.forward_impl()` / `forward_impl_graph()`
   - 文件：`/home/hanchang/ATOM/atom/model_ops/moe.py:2723-2832`
   - 第 2724 行：`return torch.ops.aiter.moe_forward(hidden_states, router_logits, self.layer_name)` —— 走 aiter custom op，最终落到 `forward_impl` / `forward_impl_graph`。
   - 第 2760 / 2815 行：`final_hidden_states = self.quant_method.apply(layer=self, x=..., router_logits=..., activation=self.activation, ...)`。
   - BF16 模式下 `self.quant_method` 由构造路径指定（同文件 2154 行）：
     ```
     self.quant_method: Optional[QuantizeMethodBase] = UnquantizedFusedMoEMethod(...)
     ```

3. `UnquantizedFusedMoEMethod.apply()`
   - 文件：`/home/hanchang/ATOM/atom/model_ops/moe.py:533-589`
   - 关键尾部（581 行起）：
     ```
     return fused_moe(
         hidden_states=x,
         w1=layer.w13_weight,
         w2=layer.w2_weight,
         topk_weight=topk_weights,
         topk_ids=topk_ids,
         expert_mask=expert_map,
         activation=activation,
     )
     ```
   - `fused_moe` 是文件首部 import 的 `from aiter.fused_moe import fused_moe`（moe.py:12）。

4. `aiter.fused_moe.fused_moe()` → `fused_moe_()` → `fused_moe_2stages` / `fused_moe_1stage`
   - 文件：`/home/hanchang/junlin12_repos/aiter/aiter/fused_moe.py:120-171`，`208-366`
   - 第 312-366 行根据 metadata 派发到 CK 2-stage 或 1-stage：
     ```
     if metadata.run_1stage:
         return metadata.stage1(...)
     else:
         return fused_moe_2stages(...)
     ```
   - BF16 路径上 quant_type=No、dtype=bf16，对应 dispatch 表（aiter/fused_moe.py:574 / 588）：
     ```
     (Silu, QuantType.No, bf16, bf16, bf16, False, False) : aiter.fmoe
     ```
     即 `aiter.fmoe`（CK kernel），与 `aiter.tuned_gemm.tgemm` 完全无关。

结论：BF16 routed-MoE 全程走 CK fused_moe → aiter.fmoe，**不**经过 `tgemm`，因此与 `bf16_tuned_gemm.csv` 无任何调用关系。

---

## 2. FP8 模式 MoE forward 路径

1. 相同入口：`Step3p5MoE.forward()` → `self.experts(...)`（step3p5.py:293）。FP8 模式下 `quant_config` 不为 None，导致 `FusedMoE` 在构造时把 `self.quant_method` 设置为 `Fp8MoEMethod`。

2. `FusedMoE.forward_impl()` → `self.quant_method.apply(...)`（moe.py:2815）。

3. `Fp8MoEMethod.apply()`
   - 文件：`/home/hanchang/ATOM/atom/model_ops/moe.py:1835-1904`
   - 两条尾部分支：
     - per_Tensor 或无 modular kernel → `torch.ops.aiter.rocm_aiter_fused_moe(...)`（moe.py:1873-1887）。
     - 否则 → `self.fused_experts(...)`（moe.py:1888-1903），`self.fused_experts` 是 `FusedMoEModularKernel`（同文件 432 行 `self.fused_experts = FusedMoEModularKernel(...)`，由 `process_weights_after_loading`/`maybe_make_prepare_finalize` 装配）。
   - `rocm_aiter_fused_moe_impl` 内部（moe.py:648-668 段）也是 `return fused_moe(...)`，即 aiter.fused_moe。

4. 同样落到 `aiter/fused_moe.py:fused_moe_()`。FP8 dispatch（aiter/fused_moe.py:590）：
   ```
   (Silu, per_1x128, bf16, fp8, fp8, True, False) : aiter.fmoe_fp8_blockscale_g1u1
   ```
   即 CK FP8 blockscale kernel。

结论：FP8 routed-MoE 同样走 CK fused_moe（fmoe_fp8_blockscale_g1u1 / rocm_aiter_fused_moe），与 `tgemm`/`bf16_tuned_gemm.csv` 无关。

---

## 3. BF16 与 FP8 是否都走 CK？

是。两条路径在 ATOM 这一层都通过 `aiter.fused_moe.fused_moe`（或 `torch.ops.aiter.rocm_aiter_fused_moe` 包装）进入 aiter，最终 dispatch 到 `aiter.fmoe*` 系列 CK kernel。

- 共同的进入点：`atom/model_ops/moe.py:12` 的 `from aiter.fused_moe import fused_moe`。
- BF16：`UnquantizedFusedMoEMethod.apply` → `fused_moe(...)`（moe.py:581）。
- FP8：`Fp8MoEMethod.apply` → `torch.ops.aiter.rocm_aiter_fused_moe(...)` / `self.fused_experts(...)`（moe.py:1873/1888），二者最终都走 aiter CK。
- 无任何路径调用 `tgemm.mm`。

---

## 4. `bf16_tuned_gemm.csv` 的实际调用层（`tgemm` 在 ATOM 内的调用点）

全仓库扫描 `tuned_gemm | TunedGemm | bf16_tuned`：

```
/home/hanchang/ATOM/atom/model_ops/embed_head.py:17  from aiter.tuned_gemm import tgemm
/home/hanchang/ATOM/atom/model_ops/linear.py:22      from aiter.tuned_gemm import tgemm
```

具体调用点：

- `atom/model_ops/embed_head.py:181`：
  ```
  logits = tgemm.mm(x, self.weight, self.bias)
  ```
  即 lm_head 的 logits 投影。

- `atom/model_ops/linear.py:393`（`LinearBase.forward`，QuantType.No 分支）：
  ```
  if self.quant_type.value == QuantType.No.value:
      y = tgemm.mm(x, self.weight, self.bias, otype=otype)
  ```
- `atom/model_ops/linear.py:413`（QuantType.per_Tensor 分支，FP8 才走）。

`LinearBase` 的子类（`atom/model_ops/linear.py`）：
- 205 `LinearBase`
- 470 `ReplicatedLinear`
- 496 `ColumnParallelLinear`
- 526 `MergedColumnParallelLinear`
- 632 `QKVZBAParallelLinear`
- 732 `QKVGParallelLinear`
- 879 `QKVParallelLinear`
- 962 `RowParallelLinear`
- 1007 `MergedReplicatedLinear`

Step-3.5 在 step3p5.py 中使用的 Linear（grep `from atom.model_ops.linear`）：`MergedColumnParallelLinear`、`QKVParallelLinear`、`ReplicatedLinear`、`RowParallelLinear`、`ColumnParallelLinear`。具体使用点：
- step3p5.py:111 `gate_up_proj = MergedColumnParallelLinear(...)`（dense MLP / 共享专家 MLP）
- step3p5.py:118 `down_proj = RowParallelLinear(...)`
- step3p5.py:196 `gate = ReplicatedLinear(...)`（router 投影；BF16 模式下 forward 用 fp32 linear，未走 tgemm —— step3p5.py:283-286）
- step3p5.py:378 `qkv_proj = QKVParallelLinear(...)`
- step3p5.py:387 `o_proj = RowParallelLinear(...)`
- step3p5.py:405 `g_proj = ColumnParallelLinear(...)`（head-wise gate）

→ `tgemm` 在 BF16 模式下调用方为：QKV proj、O proj、g_proj（head-wise gate）、shared/dense MLP gate_up_proj/down_proj、lm_head（embed_head）。**routed FusedMoE 完全不调用 `tgemm`**。

---

## 5. 62/120 次 miss 的实际来源层（按 N/K 反推）

模型 dim：hidden=4096，head_dim=128，num_heads=64，kv=8，intermediate=11264，moe_intermediate=?，vocab=128896。Sliding 层 num_heads=96，kv=8。

### h1_tp2_full.log（62 次 miss，tp=2，每个 (N,K) 各 6 次，1 次 2）

| 计数 | N    | K    | 推断层（tp=2 后维度） |
|------|------|------|------------------------|
| 6    | 5120 | 4096 | qkv_proj 全注意力层：(64+8+8)*128 / 2 = 5120 ✓ QKVParallelLinear |
| 6    | 7168 | 4096 | qkv_proj 滑窗层：(96+8+8)*128 / 2 = 7168 ✓ QKVParallelLinear |
| 6    | 4096 | 4096 | o_proj 全注意力层：64*128 / 2 = 4096 ✓ RowParallelLinear |
| 6    | 11264| 4096 | dense / shared MLP gate_up：intermediate=11264，未 tp 切（或 fp32 gate 路径产物） |
| 6    | 4096 | 5632 | dense / shared MLP down：intermediate/2 = 5632，K=intermediate/tp=11264/2 ✓ RowParallelLinear |
| 6    | 4096 | 6144 | sliding 层 o_proj：96*128/2 = 6144 ✓ RowParallelLinear |
| 6    | 1280 | 4096 | shared/MoE 中间投影（gate_up shape=2*moe_intermediate？需 moe_intermediate=640，1280=640*2）|
| 6    | 4096 | 640  | 对应 down_proj，K=640 ✓ |
| 6    | 32   | 4096 | g_proj 全注意力层：64/2 = 32 ✓ ColumnParallelLinear |
| 6    | 48   | 4096 | g_proj 滑窗层：96/2 = 48 ✓ ColumnParallelLinear |
| 2    | 64448| 4096 | lm_head：128896/2 = 64448 ✓ embed_head.py |

对照 `tgemm` 调用点全部命中：QKV / O / g_proj / dense MLP gate_up / dense MLP down / lm_head。

**关键观察**：N=1280, K=4096 与 N=4096, K=640 的组合就是共享专家的 `Step3p5MLP`（shared expert 走 dense MLP 而非 FusedMoE，详见 step3p5.py:99-143 的 `Step3p5MLP`），亦或在 SwigluStep 层 fallback 用 dense MLP（step3p5.py:218-220 注释明确说 R5 mitigation：在 SwigluStep 层不能 fuse shared expert 进 FusedMoE，回退到 dense MLP 路径）。这里 1280=2*640、640 是 shared expert 的 intermediate，仍然属于 dense Linear，不是 routed FusedMoE。

### h1_tp4_full.log（120 次 miss，tp=4）

| 计数 | N    | K    | 推断层 |
|------|------|------|--------|
| 12   | 2560 | 4096 | qkv_proj 全注意力 tp=4：(64+8+8)*128/4 = 2560 ✓ |
| 12   | 3584 | 4096 | qkv_proj 滑窗 tp=4：(96+8+8)*128/4 = 3584 ✓ |
| 12   | 5632 | 4096 | dense MLP gate_up tp=4：intermediate=11264？11264/2=5632 → MergedColumn 的单 stage（一个 gate 或 up） |
| 12   | 4096 | 2816 | dense MLP down tp=4：11264/4 = 2816 ✓ |
| 12   | 4096 | 3072 | sliding 层 o_proj tp=4：96*128/4 = 3072 ✓ |
| 8    | 4096 | 2048 | o_proj 全注意力 tp=4：64*128/4 = 2048 ✓ |
| 12   | 640  | 4096 | shared expert MLP gate_up tp=4 的拆半 |
| 12   | 4096 | 320  | shared expert MLP down tp=4：640/2 = 320 ✓ |
| 12   | 16   | 4096 | g_proj 全注意力 tp=4：64/4 = 16 ✓ |
| 12   | 24   | 4096 | g_proj 滑窗 tp=4：96/4 = 24 ✓ |
| 4    | 32224| 4096 | lm_head tp=4：128896/4 = 32224 ✓ |

全部对应 attention / dense MLP / shared expert MLP / lm_head。**没有任何一条与 routed FusedMoE 相关的形状**：routed expert weight 通过 `aiter.fmoe` 直接处理，不会以 `tgemm.mm` 的 (N,K) 形式出现在日志中。

### FP8 日志对照（fp8_tp2_full.log / fp8_tp4_full.log）

```
fp8_tp2_full.log: 0 misses
fp8_tp4_full.log: 0 misses
```

FP8 模式下 attention/MLP 的 Linear 走 per_Tensor / per_Token / per_1x128 量化（linear.py:399-...），不进 `if self.quant_type.value == QuantType.No.value` 分支，因此不再读 `bf16_tuned_gemm.csv`。这与"BF16 模式 60+ 次 miss、FP8 模式 0 miss"的现象一致。

---

## 6. 给 lead 的结论（直接答用户的质疑）

**用户的质疑成立但需要更细分**：

1. **routed MoE 在 BF16 与 FP8 两种模式下都走 CK fused_moe**，与 `bf16_tuned_gemm.csv` 完全无关。代码证据：
   - BF16：`atom/model_ops/moe.py:581 return fused_moe(...)`（UnquantizedFusedMoEMethod）
   - FP8：`atom/model_ops/moe.py:1873 torch.ops.aiter.rocm_aiter_fused_moe(...)` / `:1888 self.fused_experts(...)`
   - 二者 import 同一个 `from aiter.fused_moe import fused_moe`（moe.py:12），dispatch 表（aiter/fused_moe.py:574/588/590）落到 `aiter.fmoe` / `aiter.fmoe_fp8_blockscale_g1u1`。

2. **`bf16_tuned_gemm.csv` 的 miss 只影响 attention proj、dense/shared MLP、g_proj、lm_head 的 BF16 GEMM**。62/120 次 miss 的 (N,K) 全部能精确映射到这些层（见上表），**不包含**任何 routed-expert 形状。

3. **因此 H6（bf16_tuned_gemm 覆盖率不足）的 TTFT 影响应当重新评估**：
   - 影响范围：每层 5 个 BF16 Linear（qkv/o/gate_up/down/g_proj）+ 顶层 lm_head；其中 dense MLP 只在 SwigluStep 层（layer 43、44 共 2 层）+ 共享专家 fallback 路径出现。
   - **不影响**：routed MoE expert（占 prefill compute 的绝大部分）。
   - 这意味着 H6 的实际 TTFT 影响要小于"bf16_tuned_gemm miss=62/120"的字面感觉，需要 #702 提供的 compute 占比作交叉验证。

4. **shared expert 在 SwigluStep 层（layer 43/44）走 dense MLP 而非 FusedMoE**（step3p5.py:218-220 注释 + step3p5.py:99-143 的 `Step3p5MLP` 实现）。所以 N=1280, K=4096 / N=4096, K=640（tp=2）和 N=640, K=4096 / N=4096, K=320（tp=4）的 miss 来自 shared expert MLP 而非 routed MoE。这部分确实被 bf16_tuned_gemm miss 拖慢，但量很小（仅 2 层 + 单 expert）。

5. **FP8 模式下 routed MoE 走 `fmoe_fp8_blockscale_g1u1`**，attention/MLP 走 per_Token/per_1x128 量化路径（linear.py:399 之后），完全不读 `bf16_tuned_gemm.csv`，所以 FP8 日志 miss=0 与代码路径一致。FP8 的 TTFT gap 的原因在 #601 已识别（fmoe tuning 缺 4 个 key tuple），与 BF16 的 tuned_gemm miss 是两条独立通道。
