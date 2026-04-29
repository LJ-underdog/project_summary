# Teammate 3 Progress

## 接手状态
- #101 FP8 tp=2 (perf_correctness_bench.py): TTFT=428.7ms / TPOT=12.7ms / CORRECTNESS=PASS
- #102 FP8 tp=4 (perf_correctness_bench.py): TTFT=382.9ms / TPOT=12.5ms / CORRECTNESS=PASS
- GPU 0-3 VRAM 已归零
- 任务目的：用原始 perf_bench.py（与 gfx942 完全相同的脚本）在 gfx950 上重跑，消除脚本差异（H5 假设）

## 已完成 Items

### [#301] perf_bench.py FP8 tp=2 on gfx950
**类型**：验证型
**结论**（来自 [PERF] 行）：
- actual_input_tokens: 10261 (target=10240)
- Engine init: 57.39s
- TTFT: 388ms (0.388s)
- TPOT: 12.283ms/token
- total_latency: 3.225s
- output_tokens: 232
- decode_throughput: 81.41 tok/s
- wall_clock (sanity): 3.238s
- VRAM 归零: PASS（GPU[0]=0%, GPU[1]=0%）

**对比**（来自 RESULTS.md）：
- 本次 perf_bench.py gfx950 tp=2: TTFT=388ms TPOT=12.28ms
- 上次 perf_correctness_bench.py gfx950 tp=2: TTFT=428.7ms TPOT=12.7ms
- gfx942 perf_bench.py tp=2: TTFT=186ms TPOT=5.2ms

**H5 结论：脚本差异 [部分影响 / 不能解释 gfx950 vs gfx942 的 2-3 倍差距]**
- gfx950 上两个脚本对比：perf_bench.py (388ms TTFT / 12.28ms TPOT) vs perf_correctness_bench.py (428.7ms / 12.7ms)
  - TTFT 差 9.5%（perf_bench 更快），TPOT 差 3.4%（接近）
  - 表明脚本本身确实有少量差异（可能 warmup/measure 方法导致），但量级很小
- 与 gfx942 相同脚本 (perf_bench.py) 比较：TTFT 388ms vs 186ms = **2.09× 慢**；TPOT 12.28 vs 5.2 = **2.36× 慢**
- 即使消除脚本差异，gfx950 仍比 gfx942 慢 2× 以上
- **H5 基本排除为主因**：脚本差异只能解释约 10% 的偏差，不能解释 2-3× 的硬件代际反转差距
- 应转向 H1（aiter dirty patch）或 H2（gfx950 CK kernel 调优不足）作为优先验证目标

**数据**：来自 logs/perf_bench_tp2.log

## 收尾存档
- tool calls 累计：~7 次
- 已完成：#301
- 关键发现：H5（脚本差异）已基本排除作为主因 — 仅 ~10% 偏差，gfx950 与 gfx942 的 2-3× 性能差距来源于其他因素
- 给 lead 的建议（#302 执行前注意）：
  1. #302 (tp=4) 用相同命令模板，CUDA_VISIBLE_DEVICES=0,1,2,3，预期 TTFT/TPOT 与 perf_correctness_bench.py 也只差 ~10%
  2. 完成 #302 后，建议优先验证 H1（aiter dirty patch fused_moe.py:881-886 run_1stage=False），这是最低成本的下一步实验
  3. 当前 perf_bench.py 输出 token 数较少（232），warmup 后 measure 时间稳定但样本短；如需更高信噪比可考虑增加 output-tokens
  4. Engine init 57s 在 cold start，若连续多次跑可考虑复用 process（当前脚本不支持）
