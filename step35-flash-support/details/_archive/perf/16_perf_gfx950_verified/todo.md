# TODO - perf_gfx950_verified

## 规范
- [~] = 进行中（@teammate-N 认领）
- [x] = 完成（附结论一行）
- [!] = 卡住，需 lead 介入
- [ ] = Pending

## Phase 0（串行，必须先跑）
- [x] #000（环境预检：全部PASS — 脚本/模型/GPU0-3 VRAM=0%/Python/API 兼容均通过）

## Phase 1（串行，tp=2 先跑）
- [x] #101 [验证] FP8 tp=2 benchmark（tp=2: TTFT=428.7ms TPOT=12.7ms CORRECTNESS=PASS）
- [x] #102 [验证] FP8 tp=4 benchmark（tp=4: TTFT=382.9ms TPOT=12.5ms CORRECTNESS=PASS）

## Phase 2（汇总）
- [x] #201 [执行] 写 RESULTS.md（gfx950 实测 + gfx942 对比 + 5 条根因假设）@Lead

## Phase 3（H5 验证 — 用 perf_bench.py 消除脚本差异）
- [x] #301 [验证] perf_bench.py FP8 tp=2（perf_bench.py tp=2: TTFT=388ms TPOT=12.28ms）
- [x] #302 [验证] perf_bench.py FP8 tp=4（perf_bench.py tp=4: TTFT=241ms TPOT=12.504ms）
- [x] #303 [执行] 更新 RESULTS.md（H5 排除 + perf_bench.py 对比列 + 根因优先级）@Lead

## Phase 4（H1 验证 — aiter run_1stage=False patch）
- [x] #401（proposed_fix_401.md 已写，Lead 审批通过）
- [x] #402 [执行] 临时加 patch（patch 已应用，run_1stage=False，缓存已清）
- [x] #403 [验证] 重跑 perf_bench.py tp=2（H1 patch tp=2: TTFT=384ms TPOT=12.346ms ≈ baseline）
- [x] #404 [验证] 重跑 perf_bench.py tp=4（H1 patch tp=4: TTFT=248ms TPOT=12.549ms ≈ baseline）
- [x] #405 [执行] 还原 patch（git diff clean，status clean，缓存已清）
- [x] #406 [执行] 更新 RESULTS.md（H1 排除 + H6 新发现）@Lead

## Phase 5（H6 验证 — bf16_tuned_gemm.csv 覆盖率）
- [x] #501 [调查] bf16 tuning 严重不覆盖 Step-3.5-Flash prefill 形状（runtime CSV /tmp/aiter_configs/bf16_tuned_gemm.csv: gfx950=779/gfx942=0；M={1..512,1024,2048,4096,8192,16384,32768}，M=10262 落空隙；(4096,4096)/(11264,4096)/(1280,4096)/(5120,4096)/(7168,4096) 等关键 (N,K) 不在 tuned 集；h1_tp2 miss=62, h1_tp4 miss=120，全部 fallback 到 torch solution:0）
- [x] #502 [执行] 更新 RESULTS.md（H6 成立 + 根因分析 + 修复路径）@Lead

## Phase 6（H6 深化 — FP8 tuning 路径调查）
- [x] #601 [调查] FP8 GEMM dispatch 路径 + gfx942/gfx950 FP8 tuning 覆盖对比（FP8 仅 routed-expert 走 fmoe；4 个 Step-3.5-Flash key tuple 全 miss：inter_dim=640/384、expert=288/289、SwigluStep、e4m3fn+per_1x128 组合无 tuning；prefill 42 MoE 层 fallback 2stage default，是 H6 之外第二个 TTFT gap）
- [x] #602 [执行] 更新 RESULTS.md（FP8 fmoe gap + 修复路径 + 根因汇总）@Lead

## Phase 7（关键质疑：MoE 是否走 CK kernel，bf16_tuned_gemm miss 是否有实际影响）
- [x] #701 [调查] BF16/FP8 routed MoE 都走 aiter.fused_moe (CK fmoe / fmoe_fp8_blockscale_g1u1)，与 bf16_tuned_gemm.csv 无关；62/120 次 miss 全部归属 attention proj / dense+shared MLP / g_proj / lm_head（详见 progress/teammate-9.md）
- [x] #702 [调查] 估算 bf16_tuned_gemm miss 的实际计算量占比 vs MoE compute（BF16 miss 占 prefill GEMM FLOPs 46.1%（92.83/201.29 TFLOPs），MoE CK 占 53.9%；torch fallback 在 tp=2 可贡献 ~87ms TTFT gap，足以解释观测；62 次 miss 全部为 attn/dense/shared/head shape，无 MoE expert shape）
- [x] #703 [执行] 汇总结论，更新 RESULTS.md（MoE→CK确认，BF16 miss 46% FLOPs，~87ms gap）@Lead

## Phase 8（aiter tuning 全量盘点）
- [x] #801 [调查] aiter configs/ 下所有 CSV 的 gfx950/gfx942 覆盖（70 CSV 全扫；gfx942 仅在 dsv3/batched/rowwise 上独占，0 命中 Step-3.5-Flash 形状；gfx950 也 0 命中 BF16 (N,K=4096) 与 fmoe (inter_dim=640, expert∈{288,289})；gfx942 vs gfx950 tuning 差异不能解释性能差异；详见 progress/teammate-11.md）

## In Progress

## Done
- [x] #801 [调查] aiter GEMM tuning 全量盘点（详见 progress/teammate-11.md）@teammate-11
- [x] #701 [调查] BF16/FP8 routed MoE 路径（CK fmoe，与 bf16_tuned_gemm.csv 无关；详见 progress/teammate-9.md）@teammate-9
- [x] #702 [调查] compute 占比估算（BF16 miss 46.1% / MoE CK 53.9%；可解释 ~87ms tp=2 TTFT gap；详见 progress/teammate-10.md）@teammate-10
- [x] #601 [调查] FP8 fmoe tuning 0 命中（详见 progress/teammate-8.md）@teammate-8
- [x] #501 [调查] bf16 tuning 不覆盖 prefill 形状（详见 progress/teammate-7.md）@teammate-7
- [x] #405 [执行] 还原 patch（git diff clean，status clean）@teammate-6
- [x] #404 [验证] H1 patch tp=4: TTFT=248ms TPOT=12.549ms（≈ baseline 241/12.50）@teammate-6
- [x] #403 [验证] H1 patch tp=2: TTFT=384ms TPOT=12.346ms（≈ baseline 388/12.28）@teammate-6
- [x] #402 [执行] H1 patch 已应用 + 缓存清除 @teammate-6
- [x] #302 [验证] perf_bench.py FP8 tp=4（perf_bench.py tp=4: TTFT=241ms TPOT=12.504ms）@teammate-4
- [x] #301 [验证] perf_bench.py FP8 tp=2（perf_bench.py tp=2: TTFT=388ms TPOT=12.28ms）@teammate-3
- [x] #201 写 RESULTS.md @Lead
- [x] #101 [验证] FP8 tp=2 benchmark（tp=2: TTFT=428.7ms TPOT=12.7ms CORRECTNESS=PASS）@teammate-1
- [x] #102 [验证] FP8 tp=4 benchmark（tp=4: TTFT=382.9ms TPOT=12.5ms CORRECTNESS=PASS）@teammate-2

## Blocked
