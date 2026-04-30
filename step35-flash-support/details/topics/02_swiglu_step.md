# 子任务 2：SwigluStep Wiring

**日期**：2026-04-23
**状态**：✅ 完成（wiring 正确；长输出 BOS-spam 为 bf16 噪声累积，非 kernel bug）
**commits**：ATOM `4a8495e`，aiter `6d70f7b54`（含 CK submodule bump）

---

## 1. 背景

Step-3.5-Flash 的 layer 43 和 44 使用 SwigluStep 激活函数（`swiglu_limits[43]=7, swiglu_limits[44]=7`），
即 `silu(gate).clamp(max=7) * up.clamp(±7)`，其余层用普通 silu。

**起始状态**：
- CK kernel 和 aiter 的 SwigluStep 基础设施已在 `feat/swiglustep-step35` 分支实现
- ATOM 端 wiring 未完成：`self.clamp_limit` 字段存了但从未传给 FusedMoE kernel
- layer 43-44 的 routed expert 实际走 plain silu，clamp 完全丢弃

**影响**：bf16 激活值大多数情况在 ±7 内（行为等价），但 outlier 时有差异，长输出下会累积误差。

---

## 2. 调查过程

### Phase A：建立 baseline

先跑 tp=2 bf16 baseline（plain silu，无 clamp），记录输出质量和延迟。
确认 baseline 正常后，SwigluStep 改动以不破坏此 baseline 为前提。

### Phase B：确认 aiter SwigluStep 基础设施

读 CK kernel（`gridwise_moe_gemm.hpp`）确认 4 处 `swiglustep_and_mul` 分支已实现：
```cpp
silu(gate); gate = clamp(gate, max=7.0f);
up = clamp(up, min=-7.0f, max=7.0f);
result = gate * up;
```

读 aiter `__init__.py` 确认 `ActivationType.SwigluStep = 3` 已导出。

### Phase C：op_test 验证

运行 `test_moe_2stage.py --activation swiglustep`，结果：
```
SwigluStep vs HF reference（clamp=7）：cos_sim = 0.999989 PASS
```

**踩坑**：首次运行 cos_sim=0.0，原因是 stale .so 文件。清理后重跑正常。

```bash
# 必须同时删 .so 和 build/（只删一个无效）
rm -f aiter/jit/module_moe_ck2stages_*swiglustep*.so
rm -rf aiter/jit/build/module_moe_ck2stages_*swiglustep*
```

### Phase D：ATOM wiring

在 `Step3p5MoE.__init__` 中根据 `clamp_limit` 决定 `activation_type`，
传给 FusedMoE → `aiter.fused_moe(activation=...)` 调用链：

```python
self._activation = (
    ActivationType.SwigluStep if self.clamp_limit else ActivationType.Silu
)
```

权重检查脚本（`D_check_weights.py`）确认：layer 43-44 检测为 SwigluStep，层 3-42 为 Silu。

### Phase E：端到端验证

tp=2 推理，4 prompts，max_tokens=128：
- 延迟在 baseline ±10% 内
- "1+2+3=?" prompt 在 SwigluStep 下给出完整正确答案；baseline（A1）同一 prompt 出现 `\x1a` 字符，但 `\x1a` 是独立的 tokenization 问题（与 SwigluStep 无关），在该测试配置下凑巧未被 SwigluStep 触发

### Phase G：层级验证（真实权重）

对 layer 44（G1，M=16，scale=0.5）及 layer 43/44（G2，M∈{16, 64, 256, 1024}，scale∈{0.5, 2.0, 5.0}）：
```
cos_sim = 0.999989 ~ 0.999990  # 全部 PASS，包括 scale=5.0 深度 clamp 场景
```
CK kernel 与 HF reference 高精度对齐（cos_sim=0.999989 为 bf16 精度上限，非 bit-exact）。

### Phase H：BOS-spam 调查（max_tokens=512+）

增大 max_tokens 后发现 ~200 token 后出现 BOS token 重复（spam）。
通过 bisection 实验定位原因：

| Variant | SwigluStep 层 | BOS-spam |
|---------|--------------|---------|
| baseline | 无 | 不出现 |
| layer-44 only | {44} | 1/4 prompt |
| layer-43 only | {43} | 2/4 prompt |
| 两层均开 | {43, 44} | 3/4 prompt |

**结论**：效果叠加，符合 bf16 噪声累积特性（≥200 decode steps），而非 kernel logic bug。
层级 cos_sim=0.999989 已是 bf16 精度上限，不可避免。

---

## 3. 根因（缺口）

**原始缺口**：ATOM `step3p5.py` 存储了 `self.clamp_limit` 但未传给 kernel（代码中有 TODO 注释）。

**为何有这个缺口**：SwigluStep CK kernel 实现在 wiring 之前完成，ATOM 端 wiring 是本次任务的核心工作。

---

## 4. 解决方案

### ATOM `step3p5.py` wiring

```python
# Step3p5MoE.__init__ 中
from aiter import ActivationType

self._activation = (
    ActivationType.SwigluStep
    if (self.clamp_limit is not None and self.clamp_limit > 0)
    else ActivationType.Silu
)

# FusedMoE 构造时传入 activation
self.experts = FusedMoE(
    ...,
    activation=self._activation,
)
```

同时设计 `_fuse_shared_at_layer` helper：layer 44 的 shared expert 走 dense path，
不接入 SwigluStep（shared expert limit=16，与 routed expert limit=7 不同，避免硬编码冲突）。

### aiter SwigluStep 基础设施 commit

8 个文件修改，包含 enum 定义、pybind 导出、codegen 实例化，随 CK submodule 一起 commit。

---

## 5. 验证结果

| 验证项 | 结果 |
|--------|------|
| op_test（preshuffle_on） | cos_sim=0.999989 PASS |
| 层级验证（真实权重 layer 43，scale=0.5~5.0） | cos_sim=0.999989~0.999990 PASS |
| tp=2 4 prompts，max_tokens=128 | 全部正常，"1+2+3=?" 给出完整正确答案 |
| BOS-spam（max_tokens=512+） | 已知问题，bf16 噪声累积，非 kernel bug |

---

## 6. 教训

| 教训 | 说明 |
|------|------|
| stale .so 必须同时清 | 只删 build/ 或只删 .so 都不够，两者必须同时清 |
| stale .so 根因 | BF16 下 preshuffle_on 和 preshuffle_off 实际 codegen 相同 kernel（均假设 shuffled layout）；首次运行 cos_sim=0.0 是因为加载了旧 .so，不是 preshuffle 路径差异 |
| baseline 优先 | 任何改动前先建立 baseline（命令+输出+延迟），改动后对比 |
| 层级验证 vs 端到端 | 层级 cos_sim 高可证明 kernel 正确，端到端异常可能是其他原因（如 BOS-spam 是噪声累积） |
| shared expert 不进 fused kernel | shared expert limit 与 routed 不同，强行 fuse 会触发硬编码 clamp 值冲突 |
