# Reviewer-B 审查报告：V04 / V06

**审查范围**：V04 (TP support：inter_dim padding + ca_comm None guard) 与 V06 (FP8 tp=4：scale ceil + padding 时机)。
**视角**：TP 分布式 × FP8 量化的联合验证；Agent Team 并行执行可行性；GPU5 异常环境下的覆盖度。

---

## 总体评估表

| 维度 | V04 | V06 | 备注 |
|------|-----|-----|------|
| 逻辑严密度 | B+ | A- | V04 的 align 边界仅靠注释，V06 的 ceil 算术有完整 walk-through |
| 实验可执行性 | B（GPU5 阻塞 tp=8） | A | V06 全在 tp≤4，无硬件阻塞 |
| 负向控制 | 缺（仅正向） | 有（Exp 5 revert ceil） | V04 应补 pre-fix crash 复现 |
| Agent Team 并行度 | 中（资源争用） | 高（GPU 集合可拆） | 见下文分工 |
| 风险声明完整度 | 完整 | 完整 | 两者都明确标注未覆盖项 |
| **联合一致性** | — | — | 见末段「V04 ↔ V06 联合问题」 |

---

## V04-TP Support 审查

### 逻辑问题

1. **align 分界 192 的证据强度不足（A.1.1 + C.2）**
   - 注释「inter≤192 用 NPerBlock=64」属于自述性证据，没有交叉引用 CK kernel manifest。
   - C.2 提供了三种验证方法（grep / brute-force sweep / 读 CK source），但**未指定哪一种是 gating 条件**。建议明确：方法 1（grep manifest）必须 PASS，方法 2（sweep）作为补强。
   - 风险：若 192 实际是经验值（例如 CK dispatch 在 inter=160 时也走 NPerBlock=128，但恰好 work），则 align=64 可能 silently 触发其他 path 的对齐 bug。
   - **建议补充实验**：grep `aiter/csrc/ck_gemm_moe_2stages_codegen/` 找到 dispatch table，把找到的边界数值（应为 192）作为 V04 sign-off 的强约束。

2. **w2 zero-padding 后 stage2 K-reduction 的「贡献为零」论证（A.1.3 + C.3）**
   - 理论部分（act=0 + w2=0 双重保险）数学上正确。
   - 但「fused_moe clips routed-weight contributions」这句注释**没有源头引用**——实际 clip 行为在 fused_moe.py 的哪一行？需要 reviewer 在审查时把 clip 代码路径定位出来，否则属于循环论证。
   - **建议补充**：grep `routed_weights` 在 aiter fused_moe 的处理，特别是 reduce 阶段是否对 inter 维做 sum/scale；若有任何 inter-axis aggregation（包括 LayerNorm/RMSNorm），则 padding 假设破裂。
   - C.3 提到 FP8 的 `fc2_smooth_scale` 是否同步 padding——这一点**与 V06 直接耦合**（见联合问题）。

3. **ca_comm None guard 的验证完整性（A.2.1）**
   - 「grep 全文件 `ca_comm.` 调用」是关键步骤，但目前文档只列了一处「遗漏 check」候选，**没有给出完整 grep 结果**。建议在执行 V04 时把 grep 结果附在 sign-off 表里。
   - 实验 4 的方法 A（被动监控）、B（assert + print）、C（数值单元测试）三选一中，**只有方法 C 是真验证**——A/B 只能证明「fallback 被走到」，无法证明「fallback 输出正确」。建议把方法 C 提升为 P0，A/B 作为补充 evidence。

4. **NCCL fallback 与原 `all_gather` 路径的「完全一致」结论过强（A.2.2）**
   - 表格列出 4 个步骤一致是字面比较；但需注意 `_all_gather_out_place` 的调用上下文（caller 的 stride/contiguity 假设）可能与 `all_gather` 不同。
   - **建议**：在实验 4-C 单元测试中，覆盖 `dim ∈ {0, 1, 2}` × `input.is_contiguous() ∈ {True, False}` 的所有组合。

### Agent Team 分工建议（含 GPU 资源分配方案）

V04 共 5 个实验，按 GPU 占用拆分：

| Agent | 任务 | GPU | 时长 | 依赖 |
|-------|------|-----|------|------|
| **A1（静态）** | C.2 grep CK manifest + A.2.1 grep `ca_comm.` 全调用 | 0 | 0.5h | 无 |
| **A2（tp=2 回归）** | 实验 3：tp=2 端到端 BF16 | 0,1 | 0.5h | 无 |
| **A3（单算子）** | 实验 1：inter sweep {64,128,160,192,224,256,320,384} + 实验 5.3 | 6,7（避开 GPU5） | 1h | A1 完成（确认边界值） |
| **A4（tp=4 端到端 + ca_comm 验证）** | 实验 2 + 实验 4-B + 4-C | 2,3 与 0,1 串行（与 A2 同 GPU 分时） | 1.5h | A2 完成 |

**关键并行机会**：
- A1（CPU-only grep）+ A2（GPU 0,1）+ A3（GPU 6,7）**可三路完全并行**
- A4 等 A2 释放 GPU 0,1 后，再用 0,1,2,3 跑 tp=4

**不建议并行的项**：
- 实验 4-B 修改 `_all_gather_out_place` 加 print 与实验 2/3 共享同一份源码 → **必须串行**或在 worktree 隔离
- 实验 5 的「假 tp=8」需要 patch 模型 sharding 配置，与所有其他实验冲突，**最后单独跑**

**Worktree 建议**：A1/A2 在主 worktree；A3/A4 各自创建 worktree（特别是 A4 因为要 patch ca_comm 路径）。

### 准确性风险

1. **tp=8 单算子降级验证不等价于端到端**：
   - 实验 5 的「kernel-only test on GPU 0-3」只能覆盖 inter=192 的 GEMM 正确性，**不覆盖**：(a) 8-rank NCCL all-gather，(b) ca_comm fallback 在 8-rank 下的 group 行为，(c) MoE expert 在 8 卡上 shard 的 partition 边界。
   - 文档 C.1 已经标注「不可信」，但建议在 sign-off 中加一行「tp=8 仅 partial coverage：weight padding ✓，communication ✗」。

2. **inter_dim padding 后的 weight shape 变化是否触发其他 kernel 约束**：
   - 当前实验 1 只验证「不 crash + cos_sim ≥ 0.9999」，**没有验证 K 维 stride / contiguity** 是否被 kernel 的 vectorized load 接受。
   - 建议：在实验 1 中额外打印 `w13_new.stride()` 和 `w13_new.is_contiguous()`，确保 padding 后的 layout 与 kernel 期望一致（aiter CK kernel 通常要求 row-major contiguous）。

3. **ca_comm None guard 被动 log 监控不充分**：
   - gfx950 上 `ca_comm == None` 是默认状态，所以「log 出现 fallback 命中」只能证明代码被 invoke，**不能证明 fallback 路径产生的 tensor 正确**。
   - 建议：实验 4-C（单元测试 vs torch reference）必须 run，不可以「方法 A 通过则跳过 C」。
   - 关于「主动触发 set_custom_all_reduce(True)」——在 gfx950 上这反而会 crash（因为 ca_comm 在 gfx950 上是 unsupported），所以不建议。

---

## V06-FP8 tp=4 审查

### 逻辑问题

1. **A.1 中 `expert_data.copy_(loaded_weight)` shape mismatch 的「open question」必须先回答**
   - 文档自己标注「This is a potential shape-mismatch bug to verify」，但 Exp 1b 只是 monkey-patch `narrow` 记录 `(start, size)`，**并未直接测 copy_ 的行为**。
   - 风险：若 PyTorch 在 dest 已经 narrow 到 `load_shard_size=3`、source 是 `size=1` 时**静默 broadcast**，那么 rank 3 的 padded scale 就不是「保留 1.0」而是「整段被填成 source[0] 的值」——这会改变 Exp 5 gibberish 复现的因果链。
   - **建议补充实验 1b'**：直接构造 dest=zeros(3) + source=tensor([0.5]) 调用 `dest.narrow(0,0,3).copy_(source)`，观察是 broadcast、raise、还是部分写入。

2. **Fix 2 padding 时机（`_process_block_quant` 而非 `process_weights_after_loading`）的中间状态竞态**
   - A.4 表格说明了为什么 FP8 必须在 quant 阶段 pad，但**未讨论**：从 `_load_w13` 写入 weight 到 `_process_block_quant` 调用 padding 之间，是否有任何函数读取 `w13.shape` 用作 buffer allocation 或 shape assertion？
   - 风险：若 `create_weights` 在 padded 之前被调用并 cache 了 `inter_dim=320`，后续 `kernel.dispatch(inter=384)` 与 cached metadata 不匹配。
   - **建议**：grep `inter` 在 `_load_w13`、`_load_w2`、`_process_block_quant`、`process_weights_after_loading` 的所有读取点，画出 timeline 确认 padded shape 的「可见时刻」。

3. **Exp 1c 的 narrow(dim, 3, 0) 边界**
   - A.2 提到 PyTorch `narrow(dim, 3, 0)` 返回空 tensor，但「subsequent copy_ on a non-empty `expert_data` slice would broadcast or raise」——**这个开放问题没有 gate**。
   - Exp 1c 在「tp=8 + GPU5 阻塞」环境下能否运行？答案：**Exp 1c 是离线 unit test，不需要分布式**——只需在单 GPU 上构造 `tp_size=8` 的 mock 调用。建议明确标注 Exp 1c **不依赖** GPU 资源。

4. **Exp 5 的负向控制需要更严格**
   - 「revert ceil → floor」时同时去掉了 `min(...)` clamp，这是**两个变化**。如果 gibberish 重现，无法区分是 ceil 还是 clamp 造成的。
   - **建议拆成两步**：
     - Exp 5a：只 revert ceil 保留 clamp → 看是否 crash（rank 3 应该 narrow 到 size=0，clamp 无效但 floor 算术不要求 clamp）
     - Exp 5b：只 revert clamp 保留 ceil → rank 3 调用 `narrow(dim, 9, 3)` on length-10 → 应当 raise IndexError

### Agent Team 分工建议

V06 共 5 个实验，并行度极高（无 GPU5 依赖）：

| Agent | 任务 | GPU | 时长 | 依赖 |
|-------|------|-----|------|------|
| **B1（offline）** | Exp 1b/1c：narrow cover-completeness + 极端 oversharding | 0（CPU 即可） | 0.5h | 无 |
| **B2（FP8 dump + tp=4）** | Exp 1a + Exp 2：scale dump + FP8 tp=4 端到端 | 0,1,2,3 | 1h | 修改源码（在 worktree） |
| **B3（FP8 tp=2 + perf 对比）** | Exp 3 + Exp 4：FP8 vs BF16 perf + tp=2 回归 | 6,7 | 1h | 无 |
| **B4（gibberish 复现）** | Exp 5a + 5b：拆成两步 revert | 2,3（与 B2 不同时） | 1h | B2 完成 |

**关键并行**：B1 + B2（GPU 0-3） + B3（GPU 6-7）**三路并行**。

**Exp 5 安全并行的前置条件**：
- B4 必须在**独立 worktree**修改 `moe.py`（用 `EnterWorktree` 工具），否则会污染 B2 正在跑的源码。
- 修改完成后，B4 的 worktree 跑完即可丢弃，无需 cleanup。
- Exp 5 跑完 B4 应主动 verify worktree 中的 moe.py 与 main 的 diff 仅限 ceil→floor 两行，避免误改。

**Exp 1a 的 source patch（env-gated print）安全性**：
- 只在 `os.environ.get("ATOM_DEBUG_FP8_SCALE")` 为真时执行，production 路径零 overhead。
- 但建议**用 worktree 隔离**，避免污染 B3 的并行 run（否则 B3 也会读到 patched moe.py，虽然 env 不设置时无影响，但 cache invalidation 仍会触发）。
- **关键**：`rm -rf /root/.cache/atom/*` 必须在 patch 后、运行前执行。

### 准确性风险

1. **Scale dump 方法对 production 路径的影响**：
   - env-gated in-source patch：无 env 时是纯 `if False:` branch，对 JIT 编译产物可能略有影响（行号偏移），但运行时零开销。可接受。
   - 风险：若 `process_weights_after_loading` 被 `@support_torch_compile` 间接 trace（不太可能），patch 会破坏 Dynamo——**建议先确认该函数不在 compile graph 内**（grep `@support_torch_compile` 上下游）。

2. **Exp 1c 在 GPU5 异常环境下能跑**：
   - Exp 1c 是 pure CPU/单 GPU mock test，**不需要** tp=8 真实分布式。文档应明确这一点，避免 reviewer 误以为 1c 被 GPU5 阻塞。

3. **FP8 gibberish 判断标准缺失**：
   - Exp 2/5 的 pass criteria 是「output is coherent」/「output is gibberish」，**没有量化阈值**。
   - 建议补充：
     - **Diversity 指标**：对 4 个 prompt 的 output token-id 求 unique ratio，正常应 > 0.5，gibberish 全 BOS 时 ≈ 0.01
     - **Repetition 指标**：max n-gram repetition rate，正常 < 0.3，gibberish 时 > 0.8
     - **Perplexity（可选）**：用一个 small LM 重新打分，正常 < 50，gibberish > 1000
   - 没有量化标准，「人工检查」会导致复现性差。

4. **Exp 3 perf gate `[0.15, 0.25]` 区间过窄**：
   - 文档基于 MEMORY 的 19% 数据划了 ±5% 区间，但单次测量噪声可能更大。建议改为「TPOT_fp8 < TPOT_bf16 × 0.90」（至少 10% 加速）作为 gate，19% ±5% 作为 informational。

---

## V04 ↔ V06 联合问题

### 联合问题 1：inter_dim padding 路径在 BF16 与 FP8 下的「padded shape 一致性」

V04 在 `process_weights_after_loading` 中 pad（BF16 unquant 路径）；V06 在 `_process_block_quant` 中 pad（FP8 路径）。两者**目标 shape 必须一致**（tp=4 时都是 inter_pad=384），否则 kernel 调用时根据 weight.shape 推导出的 inter 不一致，会导致：

- 同一份 model code 在 BF16/FP8 切换时的 dispatch 不同
- V04 A.1.2 中提到的「kernel 必须读取 weight tensor 的 actual shape 而非缓存的 config inter_dim」假设在 FP8 路径下也必须成立

**联合实验建议（必须做）**：
```python
# 同一 model 配置，分别加载 BF16 / FP8 checkpoint，dump:
print(f"BF16 w13.shape={w13.shape}, w2.shape={w2.shape}, inter_pad={inter_pad}")
print(f"FP8  w13.shape={w13.shape}, sc13.shape={sc13.shape}, inter_pad={inter_pad}")
```
**Pass criteria**：两者 `w13.shape[1] / 2 == w2.shape[2] == inter_pad == 384`。
此实验在 V06 A.4 已经提到（"Cross-reference test"），但**未列入 V04 计划**。建议把它**提升为 V04+V06 共享 sign-off 项**。

### 联合问题 2：FP8 路径的 fc2_smooth_scale 同步 padding（V04 C.3 提出，但 V06 未跟踪）

V04 C.3 提到「若 scale 张量未同步 padding（仅 weight padding），block 边界对不上 → 数值错」。
V06 全文聚焦「scale shard ceil 修复」，**没有覆盖** scale 是否也需要 inter 维 padding。

**核心问题**：FP8 block scale 张量 `sc13` 的 shape 在 inter_pad=384 下应是 `[E, 384/128=3, hidden/128]`，而非 `[E, 320/128=2.5 → 2, hidden/128]`。

如果 V06 的 Fix 2 只 pad 了 weight 没 pad scale，那么 scale 的 N 维只有 2 个 block，weight 的 N 维有 3 个 block —— kernel 访问越界或 silently 用错 scale。

**联合实验建议**：
```python
# 在 _process_block_quant 完成后 dump:
print(f"FP8 w13.shape={w13.shape}, sc13.shape={sc13.shape}")
# 期望: w13[1]/2 == sc13[1] * 128 ==384
```
若 sc13[1] != 3，则 V06 Fix 2 不完整，需要补 scale padding。

### 联合问题 3：ca_comm None guard 在 FP8 路径下的覆盖

V04 实验 4 只在 BF16 路径验证 ca_comm fallback；FP8 路径有独立的 all-reduce 调用点（fp8 quant 后的 reduce_scatter 等）吗？

**建议**：grep aiter `parallel_state.py` 中所有 `ca_comm.` / `_all_gather_out_place` 的 caller，列出哪些 caller 会被 FP8 路径触发。若有 FP8 专属 caller，需要在 V06 Exp 2 中也验证 None guard 命中（方法 B）。

---

## 给 Synthesizer 的关键问题

按优先级（P0 必答，P1 建议答）：

### P0
1. **align=192 边界证据**：是否必须 grep CK manifest 找到硬证据？还是允许 brute-force sweep 推断？这影响 V04 sign-off 是否「数学论证」充分。
2. **FP8 scale 是否需要同步 inter padding**（联合问题 2）？这是 V04/V06 的接缝，目前两份文档都没有明确覆盖，可能是 latent bug。
3. **Exp 5 拆成 5a/5b 是否必要**？当前的「同时 revert ceil + clamp」无法区分两个变化的贡献。
4. **Gibberish 量化阈值**（diversity / repetition）：是否要求 V06 在 Exp 2/5 中加入数值化判定？

### P1
5. **tp=8 单算子降级 + 「假 tp=8」算不算 sign-off**？还是必须等 GPU5 修复才允许 V04 close？
6. **ca_comm fallback 的数值正确性**：实验 4-C 是否 P0？（当前文档说三选一，我认为 C 必须做。）
7. **`expert_data.copy_(loaded_weight)` 在 shape mismatch 时的行为**（V06 A.1 open question）：是否需要先用一个 1 行 PyTorch script 直接验证？
8. **Worktree 隔离方案**：V04 实验 4-B（patch ca_comm 加 print）和 V06 Exp 1（patch moe.py dump scale）是否都用 EnterWorktree 工具？同时跑会冲突吗？
9. **Agent Team 启动门槛**：V04（5 实验，~4.5h，4 agents）+ V06（5 实验，~3.5h，4 agents）是否触达 30+ tool calls 阈值？我估计 V04 ~25 calls、V06 ~20 calls，**单独看都不够阈值**，但两者合并 + 联合问题 ~50 calls，**合并启用 Agent Team 较合理**。

### 资源协调
10. 若 V04 与 V06 同时 verify，建议的 GPU 切片：
   - V04-A2 / V04-A4：GPU 0,1,2,3
   - V04-A3：GPU 6,7
   - V06-B2：GPU 0,1,2,3（与 V04-A4 串行）
   - V06-B3：GPU 6,7（与 V04-A3 串行）
   - **避开 GPU 4,5**

---

Reviewer-B 完成，请读 REVIEW_B.md
