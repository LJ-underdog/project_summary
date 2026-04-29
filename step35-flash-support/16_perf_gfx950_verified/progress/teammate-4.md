# Teammate 4 Progress

## 接手状态
- #301 perf_bench.py FP8 tp=2 (gfx950): TTFT=388ms / TPOT=12.28ms / VRAM 归零 PASS
- H5 已基本排除：脚本差异仅 ~10% 偏差，gfx950 vs gfx942 仍差 2× 以上
- GPU 0,1 VRAM 已归零
- 任务目的：在 gfx950 上用原始 perf_bench.py 跑 tp=4，进一步验证 H5（脚本差异）在 tp=4 下的表现

## 已完成 Items

### [#302] perf_bench.py FP8 tp=4 on gfx950
**类型**：验证型
**命令**：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HOME=/root/.cache/huggingface AITER_LOG_LEVEL=INFO AITER_LOG_TUNED_CONFIG=1 \
  /opt/venv/bin/python /home/hanchang/project_summary/step35-flash-support/15_perf_tp2_tp4_tp8_eval/perf_bench.py \
  --tp 4 --model .../Step-3.5-Flash-FP8/snapshots/6eebda59... \
  --input-tokens 10240 --output-tokens 1024
```

**结论**（来自 [PERF] 行）：
- actual_input_tokens: 10262 (target=10240)
- Engine init: 71.33s
- TTFT: **241ms** (0.241s)
- TPOT: **12.504 ms/token**
- total_latency: 3.229s
- output_tokens: 240（warmup 后 measure 阶段实际生成；early EOS）
- decode_throughput: 79.97 tok/s
- wall_clock (sanity): 3.240s
- VRAM 归零: PASS（GPU[0]=0%, GPU[1]=0%, GPU[2]=0%, GPU[3]=0%）

**前后 VRAM 检查**：
- 运行前：GPU[0..3] VRAM=0%
- 运行后：GPU[0..3] VRAM=0%

**数据**：
- logs/perf_bench_tp4.log（[PERF] 行）
- logs/perf_bench_tp4_full.log（完整输出）

## 三方对比（tp=4）

| 脚本 / 平台 | TTFT | TPOT | 备注 |
|---|---|---|---|
| perf_bench.py gfx950 tp=4（本次 #302） | **241 ms** | **12.504 ms** | warmup+method A |
| perf_correctness_bench.py gfx950 tp=4（#102） | 382.9 ms | 12.5 ms | 同 GPU 0-3，长 prompt |
| perf_bench.py gfx942 tp=4 | 110 ms | 5.5 ms | 参考基线 |

**TTFT 对比**：
- gfx950 两脚本：perf_bench (241) vs perf_correctness_bench (382.9) = perf_bench 快 **37%**（差异比 tp=2 的 9.5% 更显著，可能因 tp=4 prefill 路径对 batching/warmup 更敏感）
- 同脚本 gfx950 vs gfx942：241 vs 110 = gfx950 慢 **2.19×**

**TPOT 对比**：
- gfx950 两脚本：12.504 vs 12.5 = 几乎一致（< 0.05%）
- 同脚本 gfx950 vs gfx942：12.504 vs 5.5 = gfx950 慢 **2.27×**

## H5 结论补充（tp=4）

**tp=4 同样支持排除 H5 作为主因**：
- TPOT 上脚本差异为 0（12.504 vs 12.5），完全无法解释 gfx950 vs gfx942 的 2.27× 差距
- TTFT 上脚本差异 37%（比 tp=2 的 9.5% 大），但 gfx950 与 gfx942 仍差 2.19×；即便假设全部 37% 都归因于脚本，也只能把 gfx950 perf_bench 的 241ms 进一步压低，仍远高于 gfx942 的 110ms
- 综合 tp=2 (#301) + tp=4 (#302)：脚本差异在 TPOT 上几乎为零，在 TTFT 上 ≤37%；**gfx950 vs gfx942 的 2× 差距来源于硬件/kernel/配置层面，与脚本无关**
- 下一步建议优先验证 H1（aiter dirty patch fused_moe.py:881-886 run_1stage=False）或 H2（gfx950 CK kernel 调优不足）

## 收尾存档
- tool calls 累计：~7 次
- 已完成：#302
- 关键发现：tp=4 下 perf_bench.py 与 perf_correctness_bench.py 在 TPOT 几乎无差异（12.504 vs 12.5 ms），TTFT 差 37%，但都不足以解释与 gfx942 的 2.19×/2.27× 差距 — H5 在 tp=2/tp=4 两个配置下均被排除为主因
- 给 lead 的建议（#303 执行前）：
  1. RESULTS.md 加入 perf_bench.py 列时，TTFT 差异在 tp=2/tp=4 表现不同（9.5% vs 37%），值得单独说明（可能与 prefill 阶段对 warmup/batching 敏感度有关）
  2. output_tokens=240 偏短（早 EOS），后续如需更高 TPOT 信噪比可考虑用 ignore_eos 或更长固定 max_tokens
  3. H1/H2 应作为下一阶段优先实验目标
