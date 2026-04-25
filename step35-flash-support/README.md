# Step-3.5-Flash 全栈推理支持

## 背景

**模型**：StepFun Step-3.5-Flash（BF16）及 Step-3.5-Flash-FP8（FP8 权重量化）
**硬件**：8× AMD MI350X (gfx950)，每卡 252GB HBM
**目标**：在 ATOM 推理框架上跑通 tp=2/4/8 BF16 推理及 tp=2 FP8 推理
**起始状态**：模型完全无法跑，首次运行即 crash（MoE 输出全错）

---

## 时间线与任务依赖

```mermaid
graph LR
    A["① MoE Pipeline Fix<br/>Apr 23<br/>tp=2 bf16 跑通<br/>（所有任务的基础）"]
    A --> B["② SwigluStep<br/>Apr 23<br/>layer 43-44 clamp"]
    A --> C["③ Sliding Window<br/>Apr 23<br/>'ungi' 消除"]
    A --> D["④ TP=4/8<br/>Apr 24<br/>inter_dim padding"]
    A --> E["⑤ FP8 tp=2<br/>Apr 24<br/>blockscale dispatch fix"]
    D --> F["⑥ FP8 tp=4<br/>Apr 25<br/>scale sharding ceil fix"]
    E --> F

    style A fill:#4CAF50,color:#fff
    style B fill:#4CAF50,color:#fff
    style C fill:#4CAF50,color:#fff
    style D fill:#4CAF50,color:#fff
    style E fill:#4CAF50,color:#fff
    style F fill:#4CAF50,color:#fff
```

**说明**：① 是所有后续任务的唯一前置条件。
- ②③④⑤ 在 ① 完成后相互独立并行
- ⑥（FP8 tp=4）依赖 ④（weight padding 方案）和 ⑤（blockscale dispatch 理解）

---

## 最终状态

| 配置 | 状态 | TTFT | TPOT |
|------|------|------|------|
| tp=2 BF16 | ✅ | ~91ms | ~16ms |
| tp=4 BF16 | ✅ | ~76ms | ~15ms |
| tp=8 BF16 | ⚠️ GPU5 硬件阻塞 | — | — |
| tp=2 FP8 | ✅ | ~91ms | ~16ms |
| tp=4 FP8 | ✅ | ~93ms | ~13ms（比 BF16 快 19%） |

---

## 子任务详情

| 文档 | 内容 |
|------|------|
| [01_moe_pipeline.md](./01_moe_pipeline.md) | MoE GEMM 数值错误根因与修复（两个独立 Bug） |
| [02_swiglu_step.md](./02_swiglu_step.md) | Layer 43-44 SwigluStep 激活函数 wiring |
| [03_sliding_window.md](./03_sliding_window.md) | Sliding window attention mask off-by-one |
| [04_tp_support.md](./04_tp_support.md) | TP=4/8 MoE kernel alignment 问题与修复 |
| [05_fp8_inference.md](./05_fp8_inference.md) | FP8 block-quantized 模型推理支持（tp=2） |
| [06_fp8_tp4.md](./06_fp8_tp4.md) | FP8 tp=4：三层 bug（check/padding/scale sharding ceil） |

---

## 架构速查

```
Step-3.5-Flash 模型结构：
  45 layers，hidden=4096
  层 0-2:  Dense MLP
  层 3-44: MoE（288 routed + 1 shared expert，top-8，sigmoid 路由）
           moe_intermediate_size=1280
           层 43-44: SwigluStep activation（clamp ±7）

TP 分割后 inter_dim（moe_intermediate_size / TP）：
  tp=2 → 640    tp=4 → 320    tp=8 → 160

Attention 分布：
  ~1/4 层：full attention（FMHA）
  ~3/4 层：sliding window attention（window=512）

推理调用链（MoE）：
  ATOM Step3p5MoE.forward()
    → FusedMoE.apply()
      → aiter.rocm_aiter_fused_moe() / fused_moe()
        → CK 2-stage GEMM（stage1: gate+up projection，stage2: down projection）
```

```mermaid
graph TD
    A["ATOM<br/>Step3p5MoE.forward()"] --> B["FusedMoE.apply()"]
    B --> C["aiter.rocm_aiter_fused_moe()"]
    C --> D["fused_moe_() [fused_moe.py]<br/>block_m 选择 / quant_type 路由"]
    D --> E["CK Stage1 GEMM<br/>A[M,hidden] × B[hidden,inter_dim]<br/>含 activation（silu / swiglustep）"]
    D --> F["CK Stage2 GEMM<br/>A[M,inter_dim] × B[inter_dim,hidden]<br/>含 weighted sum"]

    style A fill:#2196F3,color:#fff
    style B fill:#2196F3,color:#fff
    style C fill:#FF9800,color:#fff
    style D fill:#FF9800,color:#fff
    style E fill:#9C27B0,color:#fff
    style F fill:#9C27B0,color:#fff
```

---

## 环境

```bash
# 平台
8× MI350X (gfx950)，ROCm

# Python（必须 cd /tmp 避免 aiter namespace 问题）
cd /tmp && /opt/venv/bin/python

# 关键路径
ATOM:  /home/hanchang/ATOM
aiter: /home/hanchang/aiter
git:   /home/hanchang/junlin12_repos/{aiter,atom}（author: Jun Lin <junlin12@amd.com>）

# 标准推理命令（tp=2 bf16）
rm -rf /root/.cache/atom/* && cd /tmp && CUDA_VISIBLE_DEVICES=0,1 \
  AITER_LOG_LEVEL=WARNING \
  python -m atom.examples.simple_inference \
  --model stepfun-ai/Step-3.5-Flash --kv_cache_dtype bf16 --trust-remote-code \
  --tensor-parallel-size 2 --level 0 --temperature 0 --max-tokens 128 \
  --max-num-batched-tokens 4096 --max-num-seqs 2048
```
