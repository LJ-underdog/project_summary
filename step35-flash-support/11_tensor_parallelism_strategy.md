# Transformer 张量并行（TP）策略：从原理到每个算子

> 本文档由 agent team 调研 ATOM 代码后整理，所有结论均附代码文件路径 + 行号。
> 代码基准：`/home/hanchang/ATOM/atom/`，模型：Step-3.5-Flash（MoE + GQA）。

---

## 一、TP 设计的三条核心原则

TP（Tensor Parallelism）把单层权重切到多 GPU，让每个 GPU 只持有 `1/tp` 的权重和计算量，通过尽量少的集合通信保证结果与单卡完全等价。

### 原则 1：激活函数的非线性决定切分方向

Linear 层 `y = x @ W`（`x ∈ R^{M×K}`, `W ∈ R^{K×N}`）有两种切法：

**(a) Column-Parallel（切 N / 输出维）**

```
每个 rank: y_i = x @ W_i    # W_i ∈ R^{K×(N/tp)}
y = concat(y_0, ..., y_{tp-1})  # 概念上；物理上保持分片
```

每个 rank 的 `y_i` 是最终结果的**列子集**（完整值，不是 partial sum），可以直接进激活函数。**无需通信。**

**(b) Row-Parallel（切 K / 输入维）**

```
每个 rank: y_i = x_i @ W_i    # x_i ∈ R^{M×(K/tp)}, W_i ∈ R^{(K/tp)×N}
y = sum(y_i)                   # 必须 all-reduce
```

每个 rank 的 `y_i` 是 **partial sum**，必须 all-reduce 才得到正确 `y`。

**激活函数的关键约束**：激活函数（SiLU、GELU、Softmax）是非线性的：
```
f(a + b) ≠ f(a) + f(b)
```
Row-Parallel 后 `y_i` 是 partial sum，**必须先 all-reduce 凑齐，才能进激活函数**。
Column-Parallel 后 `y_i` 是完整列子集，可直接进激活函数（element-wise 不跨列）。

> **结论：激活函数前必须用 Column-Parallel；激活函数后才能用 Row-Parallel。**

### 原则 2：Column + 激活 + Row → 只需 1 次 all-reduce

把两段 Linear + 激活的 `Linear → f → Linear` 模式：

```
W1 用 Column-Parallel → 输出列子集 hidden_i [M, I/tp]
f(hidden_i)            → element-wise，无通信
W2 用 Row-Parallel    → partial sum y_i [M, H]
all-reduce(y_i)        → 完整 y [M, H]
```

**整个 FFN/MoE block 只需 1 次 all-reduce，中间激活零通信。**

若反过来（Row + Column）：Row-Parallel 后必须 all-reduce 才能做激活 → 多一次通信 → 总通信 ≥ 2 次。

### 原则 3：独立子计算 → 按子计算维度切，零通信

某些算子内部有相互独立的子计算，直接按"独立维度"切，计算过程中不需要通信：

- **Multi-Head Attention**：head 之间完全独立（每个 head 只用自己的 Q/K/V）→ 按 head 切，attention compute 0 通信
- **MoE Expert**：不同 expert 权重独立 → EP 按 expert 切（all-to-all），TP 按 inter_dim 切（all-reduce）
- **Vocab Embedding**：词表不同块可独立查表 → 按 vocab 切

---

## 二、FMHA（Flash Multi-Head Attention）TP

### 切分方案：按 head 维度

代码：`atom/models/step3p5.py`（Step3p5Attention）+ `atom/model_ops/linear.py`

| 阶段 | 操作 | TP 类 | 通信 |
|------|------|-------|------|
| QKV projection | Column-Parallel，切 head 维（N = heads×head_dim） | `QKVParallelLinear`（L879） | 无 |
| Attention 计算 | 每 rank 独立完成其 `num_heads/tp` 个 head | — | 无 |
| O projection | Row-Parallel，切 head 维（K = heads×head_dim） | `RowParallelLinear`（L962） | **1 次 all-reduce** |

### 为什么按 head 切是自然的

Multi-head attention 的数学定义：
```
head_i = softmax(Q_i K_i^T / √d_k) V_i    # i = 0..H-1，head 之间无依赖
MHA(x) = Concat(head_0, ..., head_{H-1}) W_O
```

**head_i 的计算只依赖自己的 Q_i/K_i/V_i，不与其他 head 交互**（原则 3 的独立性）。按 head 切后：
- QKV proj = Column-Parallel（W 的 N 维 = head 维切分）→ 每 rank 独立得到自己的 Q/K/V 列子集
- Attention 计算：Softmax 在 seq 维归一化，不跨 head → 完全局部
- O proj = Row-Parallel（W 的 K 维 = head 维切分）→ partial sum → all-reduce

### GQA 的特殊处理（Step-3.5-Flash）

Step-3.5-Flash 使用 GQA（Grouped Query Attention）：Q-head 数 > KV-head 数。

```python
# step3p5.py L341-344
assert self.total_num_heads % tp_size == 0          # Q-head 严格整除
self.num_heads = self.total_num_heads // tp_size    # 每 rank Q-head 数
self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # KV-head 可复制
```

| Case | 条件 | 处理 |
|------|------|------|
| A | `kv_heads >= tp` 且整除 | 每 rank 独占 `kv_heads/tp` 个 KV-head |
| B | `kv_heads < tp` | 每 rank 1 个 KV-head，多 rank 复制（`num_kv_head_replicas = tp/kv_heads`） |

复制实现：`linear.py` L935：`shard_rank = tp_rank // num_kv_head_replicas`，多个 rank 映射到同一个 KV shard。

### KV Cache 在 TP 下的形状

每个 rank 只存自己负责的 KV-head：
```
shape: [num_blocks, page_size, num_kv_heads/tp, head_dim]
```

**Case A（正常切）**：总 KV cache 随 tp 线性下降，是 long-context 服务的内存收益。
**Case B（复制）**：每 rank 存同一份，总内存不省，但保证每 rank 有完整 KV 做 attention。

### 数据流图（tp=4）

```
hidden_states [B*S, H]  ← 全 rank 一致
        │
  ┌─────┼─────┬─────┐
rank0  rank1  rank2  rank3
  │     │     │     │
qkv_proj (Column)      ← 无通信，各 rank 算自己 heads 的 Q/K/V
  │     │     │     │
FMHA (独立)            ← 无通信，各 rank 的 KV cache 独立
  │     │     │     │
o_proj (Row) partial   ← all-reduce ★（唯一通信点）
  └─────┴─────┴─────┘
        │
   output [B*S, H]     ← 全 rank 一致
```

---

## 三、MoE TP

### 切分方案：按 inter_dim 切（Column + Row）

代码：`atom/model_ops/moe.py`

| 权重 | 全局 shape | per-rank shape | TP 类 | 代码注释 |
|------|-----------|---------------|-------|---------|
| `w13`（gate+up） | `[E, 2I, H]` | `[E, 2I/tp, H]` | MergedColumnParallel | L2302 |
| `w2`（down） | `[E, H, I]` | `[E, H, I/tp]` | RowParallel | L2345 |

`intermediate_size_per_partition = intermediate_size // tp_size`（L2118）

### 为什么必须切 inter_dim（SiLU 的决定性作用）

MoE 的两段 GEMM：
```
stage1: x[M,H] @ w13ᵀ[H, 2I] → [M, 2I] → SiLU → act[M, I]
stage2: act[M,I] @ w2ᵀ[I, H] → [M, H]
```

**方案 A（切 hidden_dim，K 维）**：
```
每 rank 算: x_k @ W13_k = partial_k [M, 2I]    # partial sum
完整值需要: all-reduce(partial_k) → [M, 2I]     ← 强制通信
然后 SiLU  → [M, I] → 再切 → stage2 → 再通信    ← 2 次通信
```
SiLU 非线性：`SiLU(a+b) ≠ SiLU(a)+SiLU(b)`，partial sum 不能直接进 SiLU。

**方案 B（切 inter_dim，N 维，当前做法）**：
```
每 rank 算: x @ W13_k = out_k [M, 2I/tp]       # 列子集，完整值
SiLU(out_k) element-wise → act_k [M, I/tp]      ← 无需通信
act_k @ W2_k = partial_k [M, H]                 # partial sum
all-reduce → [M, H]                              ← 仅 1 次通信
```

### 通信量对比

| 方案 | 通信次数 | 通信量 | 时机 |
|------|---------|--------|------|
| 切 inter_dim（当前）| 1 | `M × H` | stage2 后 |
| 切 hidden_dim | 2 | `M × 2I`（stage1 后）+ `M × H`（stage2 后）| stage1 + stage2 |

典型 MoE 扩张比 `2I ≥ H`，切 hidden_dim 通信量至少 3× 且次数翻倍。

### all-reduce 的唯一触发点

验证（`moe.py` grep 结果）：fused_moe 内部无任何 all-reduce；唯一触发点在 fused_moe 出口：
```python
# moe.py L1037-1043
if layer.reduce_results and (_tp_size > 1 or _ep_size > 1):
    _moe_result = get_tp_group().all_reduce(_moe_result, ca_fp8_quant=False)
```
stage1 → stage2 之间**零通信**。

### 数据流图（tp=2）

```
         x [M, H]              x [M, H]
             │                     │
    GPU 0                    GPU 1
  w13_0ᵀ [H, I]          w13_1ᵀ [H, I]
  ───────────────          ───────────────
  out_0 [M, 2I/2]         out_1 [M, 2I/2]   ← 列子集，非 partial
  SiLU + gate             SiLU + gate        ← element-wise，无通信
  act_0 [M, I/2]          act_1 [M, I/2]
  act_0 @ w2_0ᵀ           act_1 @ w2_1ᵀ
  partial_0 [M, H]        partial_1 [M, H]  ← partial sum
             │                     │
             └──────────┬──────────┘
                    all-reduce               ← 唯一通信（M×H）
                   final [M, H]
```

### TP 与 EP 的关系

| | TP | EP |
|--|----|----|
| 切什么 | 同一个 expert 的 weight（切 inter_dim） | 不同 expert 分到不同 GPU |
| 通信 | 1 次 all-reduce（stage2 后） | 2 次 all-to-all（dispatch + combine） |
| 适合 | expert 内 GEMM 太大 | expert 数量多，单 GPU 装不下 |

两者可组合：`tp × ep ≤ world_size`。

---

## 四、Dense FFN（非 MoE 层）TP

Step-3.5-Flash 的 layer 0-2 使用 Dense SwiGLU FFN（`Step3p5MLP`）。

代码：`atom/models/step3p5.py:98-143`

```python
self.gate_up_proj = MergedColumnParallelLinear(  # L111: 切 N=2I
    input_size=hidden_size,
    output_sizes=[intermediate_size] * 2,
)
self.down_proj = RowParallelLinear(              # L118: 切 K=I，all-reduce
    input_size=intermediate_size,
    output_size=hidden_size,
    reduce_results=True,
)
```

切分与 MoE 完全相同（Column+Row），唯一区别是没有 Expert 维度 E。

| | Dense FFN | MoE Expert |
|-|-----------|-----------|
| weight shape | `[2I, H]` / `[H, I]` | `[E, 2I, H]` / `[E, H, I]` |
| 激活函数 | SiluAndMul（可带 clamp） | Silu / SwigluStep（±7 clamp） |
| 通信 | 1 次 all-reduce | 1 次 all-reduce |

---

## 五、Embedding 与 LM Head

### Embedding：Vocab 并行

代码：`atom/model_ops/embed_head.py:105-151`（`VocabParallelEmbedding`）

**切分**：词表维度，每 rank 持有 `[vocab/tp, hidden]`。

**通信策略**：masked lookup + all-reduce
1. 用 Triton kernel 检查 token_id 是否在本 rank 的 `[vocab_start, vocab_end)` 范围
2. 在范围内则查本地表，否则输出 0
3. 所有 rank all-reduce → 每个 token 恰好一个 rank 贡献真值，其余贡献 0

通信量 = `M × H`，1 次 all-reduce。

### LM Head：Column-Parallel + all-gather

代码：`atom/model_ops/embed_head.py:154-192`（`ParallelLMHead`）

**切分**：vocab 维（输出 N 维），每 rank 持有 `[vocab/tp, hidden]`。
**通信**：各 rank 算出 `[M, vocab/tp]` logits → 1 次 all-gather → `[M, vocab]`。

**与 Embedding 共享权重**（`tie_word_embeddings=True`，`step3p5.py:758-759`）：
同一块 `[vocab/tp, hidden]` 权重，在 Embedding 用 all-reduce（input 聚合），在 LM Head 用 all-gather（output 聚合）——不同的通信方式对应不同的语义需求。

---

## 六、全算子对比表

| 算子 | 切分维度 | TP 类 | 通信操作 | 通信量 | 为什么这样切 |
|------|---------|-------|---------|-------|------------|
| Embedding | vocab（N） | `VocabParallelEmbedding` | all-reduce | M×H | 词表大，均分存储；masked lookup 后 sum |
| FMHA QKV proj | head（N=heads×d） | `QKVParallelLinear` | 无 | — | head 间独立（原则 3） |
| FMHA Attention 计算 | head | — | 无 | — | head 间无依赖 |
| FMHA O proj | head（K=heads×d） | `RowParallelLinear` | all-reduce | M×H | partial sum → all-reduce |
| Dense FFN gate+up | inter（N=2I） | `MergedColumnParallelLinear` | 无 | — | SiLU 非线性（原则 1） |
| Dense FFN down | inter（K=I） | `RowParallelLinear` | all-reduce | M×H | 与 gate+up 配对（原则 2） |
| MoE Router | 不切 | `ReplicatedLinear` | 无 | — | 输出维=num_experts，小；复制比切划算 |
| MoE w13（gate+up） | inter（N=2I） | FusedMoE 内 | 无 | — | 同 Dense FFN（原则 2） |
| MoE w2（down） | inter（K=I） | FusedMoE 内 | all-reduce | M×H | partial sum，reduce_results=True |
| LM Head | vocab（N） | `ParallelLMHead` | all-gather | M×vocab | Column 切后需要拼完整 logits |

---

## 七、完整 Transformer Layer 的 TP 数据流

基于 `Step3p5DecoderLayer.forward`（`step3p5.py:587-621`）：

```
输入: hidden_states [M, H]  ← 全 rank 一致（前一层 all-reduce 后）

──────────────── Self-Attention ────────────────

  LayerNorm（element-wise，无通信）

  qkv_proj (Column-Parallel)    → [M, (n_q+2*n_kv)/tp * head_dim]  无通信
  split q, k, v；RoPE；KV cache update                              无通信
  FMHA attention(q, k, v)       → [M, n_q/tp * head_dim]            无通信
  o_proj (Row-Parallel)         → partial [M, H]
                    ★ all-reduce #1                  通信量 = M×H

──────────────── FFN (Dense or MoE) ─────────────

  LayerNorm（element-wise，无通信）

  [layer 0-2，Dense FFN]
    gate_up_proj (Column)  → [M, 2I/tp]              无通信
    SiluAndMul             → [M, I/tp]               无通信
    down_proj (Row)        → partial [M, H]
                    ★ all-reduce #2                  通信量 = M×H

  [layer 3-44，MoE]
    router gate (Replicated)    → topk routing        无通信
    FusedMoE w13 (Column)      → [M, 2I/tp]          无通信
    SiLU/SwigluStep            → [M, I/tp]           无通信
    FusedMoE w2 (Row)          → partial [M, H]
                    ★ all-reduce #2                  通信量 = M×H
    (+shared expert，若有，额外 1 次 all-reduce)

输出: hidden_states [M, H]  ← 全 rank 一致
```

### 每层通信汇总

| 阶段 | all-reduce 次数 | 通信量 |
|------|---------------|--------|
| Attention（O proj 后） | 1 | M×H |
| FFN（down/w2 后） | 1 | M×H |
| **每层合计** | **2** | **2×M×H** |

**整模型额外通信**：
- Embedding（入口 1 次）：all-reduce，M×H
- LM Head（出口 1 次）：all-gather，M×vocab

### 关键设计观察

1. **每层固定 2 次 all-reduce**，与 hidden_size 线性，与 tp_size 弱相关（ring-allreduce BW 不随 tp 线性下降）。
2. **MoE 不引入额外通信**：FusedMoE 的 reduce_results=True 把所有 expert 的 partial sum 一次性 all-reduce。
3. **head 切 + Column/Row 配对**把层内通信压到理论最小——任何其他切法（如 QKV 用 Row）都会多至少 2 次通信。
4. **Embedding（all-reduce）vs LM Head（all-gather）**：同一权重形状 `[vocab/tp, H]`，入口 sum、出口 concat，对应 lookup 和 matmul 两种语义的通信需求。

---

## 附：关键代码索引

| 主题 | 文件 | 关键行号 |
|------|------|---------|
| Step3p5MLP（Dense FFN） | `atom/models/step3p5.py` | 98-143 |
| Step3p5Attention（GQA） | `atom/models/step3p5.py` | 303-493 |
| Step3p5DecoderLayer.forward | `atom/models/step3p5.py` | 587-621 |
| w13/w2 per-rank shape 定义 | `atom/model_ops/moe.py` | L449-478 |
| intermediate_size_per_partition | `atom/model_ops/moe.py` | L2118 |
| MoE TP all-reduce 触发点 | `atom/model_ops/moe.py` | L1037-1043 |
| _load_w13（MergedColumnParallel） | `atom/model_ops/moe.py` | L2292-2333 |
| _load_w2（RowParallel） | `atom/model_ops/moe.py` | L2335-2364 |
| QKVParallelLinear（GQA head 切） | `atom/model_ops/linear.py` | L879-959 |
| KV-head 复制逻辑 | `atom/model_ops/linear.py` | L899-908, L935-944 |
| RowParallelLinear all-reduce | `atom/model_ops/linear.py` | L465-466 |
| ColumnParallelLinear | `atom/model_ops/linear.py` | L496-523 |
| VocabParallelEmbedding | `atom/model_ops/embed_head.py` | L105-151 |
| ParallelLMHead | `atom/model_ops/embed_head.py` | L154-192 |
