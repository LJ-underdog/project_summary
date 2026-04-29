# Teammate 1 Progress

## 接手状态
- Phase 0 #000 全部 PASS（脚本/模型/GPU 0-3 VRAM=0%/Python/API 兼容均通过）
- GPU 0,1 VRAM=0%，可直接用于 tp=2
- 运行前必须 `cd /tmp &&`（aiter namespace package 问题）
- 脚本必须显式传 `--model`，否则 EngineArgs 默认加载错误模型

## 已完成 Items

### [#101] FP8 tp=2 benchmark
**类型**：验证型
**结论**（来自命令输出）：
- Engine init: 158.12s
- Run1: TTFT=429.4ms TPOT=12.7ms total=3.37s output_tokens=233
- Run2(stable): TTFT=428.7ms TPOT=12.7ms total=4.249s output_tokens=302
- Decode throughput (stable): 78.8 tok/s
- CORRECTNESS: PASS（output chars=1422, words=213, bos_spam=False）
- first 80 chars: "Hmm, the user has provided a repetitive text about a fox and dog sentence in bot"（确认是 Step-3.5-Flash 输出，不含 <think> 标签）
- VRAM 归零: PASS（GPU[0]=0%, GPU[1]=0%）

**配置**：
- ATOM commit: acff926d
- aiter commit: 0f8164017
- Input tokens: 10213（target 10240, ±32）
- Max output tokens: 1024
- TP=2 on CUDA_VISIBLE_DEVICES=0,1
- Model: Step-3.5-Flash-FP8 (snapshots/6eebda59...)

**与 MEMORY.md 对比**：
- MEMORY 记录 FP8 tp=2 = TTFT=85ms / TPOT=13.5ms（gfx942 等参考值）
- 本次 gfx950 实测 TTFT=428.7ms / TPOT=12.7ms
- TPOT 接近（13.5ms → 12.7ms，略快 ~6%）；TTFT 显著高于历史值（428.7ms vs 85ms）— 可能与历史输入长度不同或 gfx950 prefill 差异有关，需在 RESULTS.md 中说明

**数据**：来自 logs/fp8_tp2_full.log（完整 stdout）和 logs/fp8_tp2.log（脚本日志）

## 收尾存档
- tool calls 累计：~7 次
- 已完成：#101
- 关键发现：
  1. FP8 tp=2 在 gfx950 上跑通且 CORRECTNESS=PASS，输出无 <think> 标签、无 bos_spam
  2. TPOT=12.7ms，与 MEMORY 记录的 gfx942 FP8 tp=2 (13.5ms) 持平/略快
  3. TTFT=428.7ms，明显高于 MEMORY 中 gfx942 的 85ms 参考值，需在汇总报告中标注（可能是输入长度/版本差异）
  4. Engine init 耗时 158s（safetensors 加载 ~86s + warmup ~59s），属于一次性开销
  5. Run2 后 GPU 0/1 VRAM 完全归零，#102 可立刻接手
- 给 lead 的建议（#102 tp=4 执行前）：
  1. CUDA_VISIBLE_DEVICES=0,1,2,3，注意避开 GPU 5（MEMORY 记录其硬件异常）
  2. tp=4 模型加载会更慢（~3 分钟载入 + 1 分钟 warmup），timeout 建议 ≥ 480s
  3. tp=4 长序列 (≥8209) BOS bug 已通过 aiter `a2883ab37` workaround，本次 input=10213 应当稳定；若复现 BOS spam，参考 memory/tp48-fixes.md
  4. 跑完后立即 `rocm-smi --showmemuse` 确认 4 张卡都归零
  5. RESULTS.md 中将 tp=2 vs tp=4 的 TPOT/TTFT 直接对比（MEMORY 提示 decode tp=2 性价比更高）
