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

## In Progress

## Done
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
