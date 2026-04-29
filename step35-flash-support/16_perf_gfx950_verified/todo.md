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

## In Progress

## Done
- [x] #201 写 RESULTS.md @Lead
- [x] #101 [验证] FP8 tp=2 benchmark（tp=2: TTFT=428.7ms TPOT=12.7ms CORRECTNESS=PASS）@teammate-1
- [x] #102 [验证] FP8 tp=4 benchmark（tp=4: TTFT=382.9ms TPOT=12.5ms CORRECTNESS=PASS）@teammate-2

## Blocked
