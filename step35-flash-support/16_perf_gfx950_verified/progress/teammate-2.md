# Teammate 2 Progress

## 接手状态
- #101 FP8 tp=2 完成：TTFT=428.7ms / TPOT=12.7ms / CORRECTNESS=PASS
- GPU 0,1 VRAM 已归零（rocm-smi 确认）
- 运行前必须 `cd /tmp &&`（aiter namespace）
- 脚本必须显式传 `--model`
- tp=4 长序列 BOS bug 已通过 aiter a2883ab37 workaround，input=10213 应稳定

## 已完成 Items

### [#102] FP8 tp=4 benchmark
**类型**：验证型
**结论**（来自命令输出）：
- Engine init: 64.63s
- Run1: TTFT=274.5ms TPOT=13.6ms total=4.45s output_tokens=309 decode_throughput=73.7 tok/s
- Run2(stable): TTFT=382.9ms TPOT=12.5ms total=3.463s output_tokens=248 decode_throughput=80.2 tok/s
- CORRECTNESS: PASS（output chars=1215, words=180, bos_spam=False）
- first 80 chars: "Hmm, the user has provided a massive repetitive text block alternating between a"（确认非 Qwen <think> 输出，是 Step-3.5-Flash 的正常分析式开头）
- VRAM 归零: PASS（GPU 0-3 全部 0%）
- ATOM commit: acff926d
- aiter commit: 0f8164017
- Input tokens: 10213（target 10240，符合 ±32）
**数据**：来自 logs/fp8_tp4_full.log（完整）和 logs/fp8_tp4.log（摘要）

## 关键观察
- **tp=4 长序列稳定**：input=10213 未触发 BOS spam bug，aiter a2883ab37 workaround 验证有效
- **TPOT 跨 TP 一致性**：FP8 tp=2（12.7ms）vs tp=4（12.5ms）几乎持平，tp=4 略快 0.2ms
- **TTFT 跨 TP 差异显著**：FP8 tp=4（382.9ms）比 tp=2（428.7ms）快约 11%（10.7%），与 MEMORY 中"prefill → tp=4 快"的结论一致
- **Decode throughput**：tp=4 stable 80.2 tok/s
- Run1 vs Run2 TTFT 抖动较大（274 vs 383ms），按规范取 Run2 (last) 作为 stable

## 收尾存档
- tool calls 累计：~9 次
- 已完成：#102
- 关键发现：
  1. FP8 tp=4 性能稳定，TTFT=382.9ms / TPOT=12.5ms / CORRECTNESS=PASS
  2. tp=4 长序列 BOS workaround 在 gfx950 上有效（aiter a2883ab37）
  3. Engine init 较慢（64.63s vs tp=2），属正常（更多 rank 初始化）
  4. Dynamo recompile_limit warning 在 warmup 触发但不影响后续测量
- 给 lead 的建议（#201 写报告需注意）：
  1. **stable 取值**：本任务 Run2 stable TTFT 比 Run1 高（382 vs 274ms），如对照 gfx942 数据请确认 gfx942 也是取 last run，避免 first vs stable 比较口径不一致
  2. **TTFT/TPOT 数据来源标注**：tp=2 和 tp=4 都用同一脚本 perf_correctness_bench.py，stable=last run，input≈10213
  3. **MEMORY 校对**：MEMORY.md 中 FP8 tp=4 记录为 "TTFT=86ms, TPOT=13ms"，本次实测 TTFT=382.9ms 差异显著（86ms 应是过时或不同测试条件下的数据），需在报告中说明本次为 input=10213 长序列场景
  4. **GPU5 状态**：本次 CUDA_VISIBLE_DEVICES=0,1,2,3 完全避开了 GPU5，结果可信
  5. **decode throughput**：建议在报告中加入 decode_throughput 列，tp=4=80.2 tok/s
