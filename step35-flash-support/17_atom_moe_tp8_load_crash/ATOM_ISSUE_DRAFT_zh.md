# [BUG] tp=8 weight load 崩溃：`_load_w2` / `_load_w13` 在 trailing rank 计算出非法 `narrow()` size（Step-3.5-Flash-FP8 / per_1x128 scale 维）

> **状态**：DRAFT — 尚未提交至上游
> **作者**：fp8-tp4-repro 项目，issue_wave（2026-04-29）
> **范围**：bug 报告 + 根因分析 + advisory 修复建议。**本草稿不实施任何源码修改。**

---

## 摘要

在 `stepfun-ai/Step-3.5-Flash-FP8` 上以 `-tp 8` 启动 OpenAI 兼容 server 时，ATOM 在 **weight load 阶段确定性崩溃**（崩溃发生在任何 inference / cudagraph / batching 之前）。根因位于 `atom/model_ops/moe.py`：`_load_w2`（line 2335-2364）与 `_load_w13`（line 2292-2333）使用基于 `ceil` 的切分，未对 trailing rank 上 `start >= loaded_weight.shape[shard_dim]` 的情况做兜底，导致传入 `torch.Tensor.narrow()` 的 `size` 参数为负。

崩溃**与** CUDAGraph capture sizes、batch size、sampling 参数、`max_tokens` **均无关**。它是 `(D, tp_size)` 的纯函数，其中 `D = loaded_weight.shape[shard_dim]`。

`_load_w13` 中存在与 `_load_w2` **完全相同**的 bug 模式，且在同一模型 `-tp 8` 下可触达。

---

## 环境

| 项 | 值 |
|------|-------|
| ATOM commit | `acff926` |
| AITER commit | `0f8164017` |
| CK（Composable Kernel）commit | `defd7ad29` |
| 模型 | `stepfun-ai/Step-3.5-Flash-FP8` |
| 硬件 | AMD Instinct MI308X UBB（8 GPU/节点，gfx942，e4m3fnuz）|
| 启动命令 | `python -m atom.entrypoints.openai_server --model stepfun-ai/Step-3.5-Flash-FP8 --kv_cache_dtype fp8 -tp 8` |
| Sampling（无关 — 从未触达）| `temperature=0`, `top_p=1`, `max_tokens=512`, `cudagraph_capture_sizes=[1,2,4]` |

`-tp 2` 与 `-tp 4` 在同一组 commit 下端到端正常运行。

---

## 复现

```bash
cd /path/to/ATOM
# 与 tp=2 / tp=4 下 PASS 的脚本完全相同
python -m atom.entrypoints.openai_server \
  --model stepfun-ai/Step-3.5-Flash-FP8 \
  --kv_cache_dtype fp8 \
  -tp 8
```

预期：server 进入 `Loaded model and starting engine ...`
实际：`Loading safetensors shards: 1/44` 阶段 `ModelRunner6/8` 与 `ModelRunner7/8` 抛 `RuntimeError`，随后 EngineCoreMgr 收到所有 rank 的 `SHUTDOWN` 信号，进程退出。

完整脚本（带 4 prompt + 串行三方 tp）见 fp8-tp4-repro 项目的 `correctness_eval/correctness_bench.py`；对本 bug 而言 prompt 内容与 sampling 参数均无影响 —— 崩溃发生在任何 prompt 被消费之前。

---

## Traceback（原文，节选）

### 症状 A — rank 6/7：`narrow()` size 为负

```
File "atom/models/step3p5.py", line 897, in load_fused_expert_weights
    weight_loader(param, loaded_weight[expert_id], name, shard_id, expert_id)
File "atom/model_ops/moe.py", line 2610, in weight_loader
    self._load_model_weight_or_group_weight_scale(...)
File "atom/model_ops/moe.py", line 2256, in _load_model_weight_or_group_weight_scale
    self._load_w2(...)
File "atom/model_ops/moe.py", line 2357, in _load_w2
    loaded_weight = loaded_weight.narrow(shard_dim, start, size)
RuntimeError: narrow(): length must be non-negative.
```

### 症状 B — rank 5：`narrow()` size=0 通过，但下游 `copy_` 形状不匹配

```
RuntimeError: The size of tensor a (2) must match the size of tensor b (0)
              at non-singleton dimension 1
```

（在同一 `_load_w2` 路径中的 `expert_data.copy_(loaded_weight)` 处触发。）

两个症状是**同一根因**在不同 rank 上的体现：rank 5 恰好命中 `start == D`（`size = 0`，`narrow` 不报错但下游 `copy_` 失败）；rank 6 与 rank 7 命中 `start > D`（`size < 0`，`narrow` 直接报错）。

完整日志：fp8-tp4-repro 项目的 `correctness_eval/logs/tp8_full.log`。

---

## 根因

相关代码（原文，行号取自 ATOM `acff926`）：

```python
# atom/model_ops/moe.py:2335-2364   _load_w2  (down_proj, RowParallel on input_dim)
def _load_w2(self, expert_data, shard_dim, loaded_weight, tp_rank, load_full=False):
    shard_size = expert_data.shape[shard_dim]
    if not load_full:
        load_shard_size = (
            loaded_weight.shape[shard_dim] + self.tp_size - 1
        ) // self.tp_size                                      # ← ceil-based split
        start = load_shard_size * tp_rank
        size  = min(load_shard_size, loaded_weight.shape[shard_dim] - start)
        loaded_weight = loaded_weight.narrow(shard_dim, start, size)   # ← line 2357
        if load_shard_size != shard_size:
            expert_data = expert_data.narrow(shard_dim, 0, load_shard_size)
    ...
    expert_data.copy_(loaded_weight)                            # ← line 2364
```

```python
# atom/model_ops/moe.py:2310-2314   _load_w13  (gate_up_proj, MergedColumnParallel on output_dim)
load_shard_size = (
    loaded_weight.shape[shard_dim] + self.tp_size - 1
) // self.tp_size                                              # ← 同一 ceil 切分公式
start = load_shard_size * tp_rank
size  = min(load_shard_size, loaded_weight.shape[shard_dim] - start)
loaded_weight = loaded_weight.narrow(shard_dim, start, size)   # ← line 2315（同一 bug）
```

记 `D = loaded_weight.shape[shard_dim]`，`C = ceil(D / tp_size) = load_shard_size`。

对 trailing rank，`start = C * tp_rank` 可能满足 `start >= D`，此时：
- 若 `start == D` → `size = 0` → `narrow` 返回空切片，但后续 `copy_` 失败（`expert_data` 在该维上非零，形状不匹配）；
- 若 `start > D` → `size = D - start < 0` → `narrow` 抛 `RuntimeError: narrow(): length must be non-negative.`

### 触发条件

最后一个 rank 上 `start_{last} = C * (tp_size - 1) > D`，即：

> `D < C * (tp_size - 1)`     （`tp_rank = tp_size - 1` 触发 `narrow` raise 的充要条件）

等价地：当 `D / tp_size` 与一个整数相差较远，导致 ceil 上调把"最后一个 rank 该做的活"提前消耗完时即触发。

`_load_w13` 现有注释（line 2306-2309）给出了一个例子：`inter=1280, tp=4 → 10 个 scale 块 / 4 = 2.5 → ceil=3`，该例在 `tp=4` 下没问题。把同一形状代入 `tp=8`：

> `D = 10`, `C = ceil(10/8) = 2`
> `rank 5: start=10 = D → size = 0`     ← 症状 B
> `rank 6: start=12 > D → size = -2`    ← 症状 A
> `rank 7: start=14 > D → size = -4`    ← 症状 A

这与观测到的症状切分（rank 5 `copy_` 形状不匹配 + rank 6/7 `narrow` 报错）**完全一致**，所以 `D = 10`（`inter_size = 1280` 的 per_1x128 scale 块数）是 Step-3.5-Flash-FP8 上最可能的触发值。

bug 是**确定性**的（无 race，无 GPU 非确定性），是 `(D, tp_size)` 的纯函数。

---

## 受影响配置

bug 表现为两个相关症状，二者的边界是最后一个 rank 上 `start == D` 还是 `start > D`：

- 症状 A（`narrow` 报错）：`start_{last} = C * (tp_size - 1) > D`     ⟺ `D < C * (tp_size - 1)`
- 症状 B（`copy_` 形状不匹配）：`start_{last} = D`，`size = 0`，但 `expert_data` 在该维非零

二者根因相同：基于 ceil 的切分给至少一个 trailing rank 分配了越界 `start`。

对 `tp_size = 2`，trailing rank 是 rank 1，`start = ceil(D/2) ≤ D` 对所有 `D ≥ 1` 成立，因此 `tp = 2` 可证安全。对 `tp_size ≥ 4`，许多小的 `D` 值会触发 —— 这正是 fp8 per-block scale 维所处的范围（例如 `inter_size / 128`，对常见 MoE expert 大小通常是个位数 `D`）。

**已观测的具体案例**：Step-3.5-Flash-FP8 在 `tp_size = 8` 下 `D = 10`（`inter_size = 1280` 的 per_1x128 scale 块数，与 `_load_w13` 注释 line 2306-2309 给出的例子吻合）—— 触发 rank 5 的症状 B 与 rank 6/7 的症状 A。

本草稿不穷举完整的 `(D, tp_size)` 触发集；上面给出的闭式条件足以驱动修复方案与单元测试参数化。

---

## 推测修复方案（advisory only — 本草稿未实施）

两个自然的方案。**两者都未实施**，二者均建议先与 ATOM 维护者讨论再 land。

### 方案 A — 当 rank 不持有任何切片时 early return

```python
# 伪代码，NOT a patch
start = load_shard_size * tp_rank
if start >= loaded_weight.shape[shard_dim]:
    # 该 rank 不持有该 expert 此 weight 的任何切片
    return  # 完全跳过 narrow + copy_
size = min(load_shard_size, loaded_weight.shape[shard_dim] - start)
```

优点：改动最小；符合物理含义（"trailing rank 没东西可加载"）；保留 line 2306-2309 注释明确论证过的 ceil 逻辑（用于 partial scale block 正确性）。
缺点：需仔细审计下游消费者是否能容忍"该 rank 跳过加载"的语义（expert_data 初值、all-reduce 语义、mxf4 dtype 路径等）。

### 方案 B — 偶数切分，余数挂到 rank 0

```python
# 伪代码，NOT a patch
base, rem = divmod(loaded_weight.shape[shard_dim], self.tp_size)
load_shard_size = base + (1 if tp_rank < rem else 0)
start = base * tp_rank + min(tp_rank, rem)
```

优点：每个 rank 都满足 `start + size <= D`，永不触达负 size 情况。
缺点：改变切分分布；line 2306-2309 注释指出 ceil 是为了确保最后一个 partial scale block 被包含 —— 方案 B 把这个 partial block 放到 rank 0 而不是 trailing rank，AITER fused MoE kernel 是否依赖该顺序需上游 review。同时该方案也**反转**了 per-rank residual size 的顺序（ceil 切法下 trailing rank 持有 partial 块；方案 B 下 rank 0 持有），任何依赖该顺序做 per-block dequant indexing 的下游消费者都需要审计。本方案需要上游 review。

### Sweep 目标

同一修复需同时落到 **`_load_w2`（line 2355-2357）** 与 **`_load_w13`（line 2313-2315）** —— 两处使用完全相同的公式。按 ATOM 项目的 "fix-then-sweep" 规则，单一 PR 应同时 patch 两处并增加一个按 `(D, tp_size)` 参数化的单元测试，覆盖 `D ∈ {tp_size-1, tp_size, tp_size+1, tp_size+rem}` 这类边界。

---

## 影响 / 范围

- **100% 阻塞**：任意用户在 `tp_size ≥ 4` 下尝试 serve 满足触发条件的 per-block scale 维的模型时均会崩溃。Step-3.5-Flash-FP8 + `tp=8` 是观测到的具体案例；同模型 + `tp=4` 之所以安全，仅因 `D=10 ≥ ceil(10/4)*3 = 9` 恰好不踩边界。
- bug 完全位于 **weight load 路径**，inference / CUDAGraph / batching / sampling 均**无涉及**，无需调查 runtime engine。
- 与 fp8-tp4-repro 项目主线 RCA 的上游症状交叉引用：Step-3.5 MoE w2/w13 在 tp 边界处的 sharding 失配 —— 三个 root cause 已分别落到 `aiter/fused_moe.py:881-886`、`atom/model_ops/moe.py:1709-1746` padding 与 `atom/model_ops/utils.py:79` weight_scale。本 `_load_w2` / `_load_w13` 的边界 case 是该家族中第四个、也是位置最上游的一个。

---

## 引用

| 来源 | 路径 |
|--------|------|
| 代码：`_load_w2` | `atom/model_ops/moe.py:2335-2364`（ATOM `acff926`）|
| 代码：`_load_w13`（同 bug 模式）| `atom/model_ops/moe.py:2292-2333` |
| 完整崩溃日志（逐 rank）| `correctness_eval/logs/tp8_full.log`（fp8-tp4-repro）|
| 正确性 wave RCA | `correctness_eval/CORRECTNESS_REPORT.md` §4（fp8-tp4-repro）|
| 逐 rank 复现笔记 | `correctness_eval/progress/corr-t1.md` §3 |
| 项目 handoff packet | `handoff_wave/HANDOFF_PACKET.md` §4.1 F-OPEN-1 |
| 三仓 commit pin | ATOM `acff926` / AITER `0f8164017` / CK `defd7ad29` |

---

## Caveats / 本草稿**不**主张的内容

1. `D = 10` 这个具体值是**推断**出来的，依据是把观测到的 rank 5/6/7 症状切分对照 ceil 切分的算术结果。它与 `inter_size = 1280` 与 per_1x128 scale 块假设一致，但具体的 tensor name 与 dtype 本草稿没有从 dump 直接提取。此外 `inter_size = 1280` 这个值本身也是推断出来的，依据是把 `_load_w13` 注释 line 2306-2309 的例子对照观测到的崩溃模式 —— 我们没有从模型 config 中提取该值。上游维护者可在 `-tp 8` 启动时于 `moe.py:2357` 之前加 `print(name, loaded_weight.shape)` 来同时确认这两个值。
2. 我们**没有**独立观测 `_load_w13` 在 Step-3.5-Flash-FP8 + `tp=8` 下崩溃 —— 因为 `_load_w2` 的崩溃在受影响 expert 上已先杀掉 loader。`_load_w13` 共享同一 bug 的论据基于代码结构完全相同，而非独立运行。
3. 我们**没有**实施或测试方案 A 或方案 B，二者均为给上游讨论用的草图。
4. 我们的内部 benchmarking 中**单独观测到**：`tp=8` 在另一种配置（`cudagraph_capture_sizes=[1]` + 单条短 prompt）下 PASS。我们目前**无法**把该 PASS 与本处描述的确定性崩溃调和；一种解释是 benchmark run 用了不同的启动路径、模型变体或 expert-loading 分支。该问题将在我们的项目跟踪中独立解决；它不阻塞上游对本 bug 的 triage，因为本草稿描述的崩溃在本 issue 的复现命令下完全确定性发生。

---

**草稿结束。**
