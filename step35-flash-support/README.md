# Step-3.5-Flash 全栈推理支持

## 背景

**模型**：StepFun Step-3.5-Flash（BF16）及 Step-3.5-Flash-FP8（FP8 权重量化）
**硬件**：8× AMD MI350X (gfx950)，每卡 252GB HBM
**目标**：在 ATOM 推理框架上跑通 tp=2/4/8 BF16 推理及 tp=2 FP8 推理
**起始状态**：模型完全无法跑，首次运行即 crash（MoE 输出全错）

---

## 时间线

```
2026-04-23
  ├── MoE Pipeline 修复    → tp=2 bf16 第一次正确输出
  ├── SwigluStep Wiring    → layer 43-44 激活函数对齐 HF 实现
  └── Sliding Window 修复  → "ungi" 乱码消除
2026-04-24
  ├── TP=4/8 支持          → tp=4 正常；tp=8 被 GPU5 硬件异常阻塞
  └── FP8 推理支持         → tp=2 FP8 量化模型跑通
```

---

## 最终状态

| 配置 | 状态 | TTFT | TPOT |
|------|------|------|------|
| tp=2 BF16 | ✅ | ~91ms | ~16ms |
| tp=4 BF16 | ✅ | — | — |
| tp=8 BF16 | ⚠️ GPU5 硬件阻塞 | — | — |
| tp=2 FP8 | ✅ | ~91ms | ~16ms |

---

## 子任务详情

| 文档 | 内容 |
|------|------|
| [01_moe_pipeline.md](./01_moe_pipeline.md) | MoE GEMM 数值错误根因与修复（两个独立 Bug） |
| [02_swiglu_step.md](./02_swiglu_step.md) | Layer 43-44 SwigluStep 激活函数 wiring |
| [03_sliding_window.md](./03_sliding_window.md) | Sliding window attention mask off-by-one |
| [04_tp_support.md](./04_tp_support.md) | TP=4/8 MoE kernel alignment 问题与修复 |
| [05_fp8_inference.md](./05_fp8_inference.md) | FP8 block-quantized 模型推理支持 |

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
