# Teammate 0 Progress

## 接手状态
Phase 0 预检，无前序 teammate。

## 已完成 Items

### [#000] 环境预检
**类型**：验证型
**结论**（来自命令输出）：
- 检查1 脚本：[PASS] `/home/hanchang/project_summary/step35-flash-support/perf_correctness_bench.py` 存在，16K，shebang 为 `#!/usr/bin/env python3`，docstring 标识为 "Step-3.5-Flash FP8 标准化性能 + 正确性测试，适用 gfx950/gfx942"
- 检查2 模型路径：[PASS] `/root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/6eebda59dd87ca5729648ec7cfed0becfceb273e/config.json` 存在
- 检查3 GPU 0,1：[PASS] VRAM=0% / 0%（GPU[0] 与 GPU[1] 均 0% allocated）
- 检查4 GPU 2,3：[PASS] VRAM=0% / 0%（GPU[2] 与 GPU[3] 均 0% allocated）
- 检查5 Python：[PASS] `import atom` OK，`import transformers` OK；有 aiter NUMA balancing 提示（warning，非阻断）
- 检查6 API 兼容：[PASS] 关键行号
  - L191: `EngineArgs.add_cli_args(parser)`
  - L207: `args.cudagraph_capture_sizes = str([1])`（字符串形式，与 EngineArgs 契约一致）
  - L250: `engine_args = EngineArgs.from_cli_args(args)`
- 总结：**全部 PASS**，环境就绪可执行 Phase 1

## 收尾存档
- tool calls 累计：~9 次
- 已完成：#000
- 关键发现：
  1. 脚本 L207 显式将 `cudagraph_capture_sizes` 设为字符串 `'[1]'`，提示 EngineArgs 期望字符串而非 list；后续若要修改 capture sizes 须保持字符串形式
  2. GPU 0-3 VRAM 全 0%，tp=2（用 0,1）与 tp=4（用 0,1,2,3）均无 VRAM 残留
  3. aiter import 时打印 NUMA balancing 警告（非阻断），与 MEMORY 中环境一致
- 给 lead 的建议：
  1. todo.md L10 描述写的是 "GPU 4,6 / 0,1,2,3"，但 TASK 与本次执行检查的是 GPU 0,1（tp=2）和 GPU 0,1,2,3（tp=4）；若实际要用 4,6 需在 #101 执行前再确认 CUDA_VISIBLE_DEVICES 设定
  2. 环境无阻断，可直接派发 #101 (FP8 tp=2 benchmark)
  3. 跑 python 须 `cd /tmp &&` 前缀（aiter namespace package 问题，已在本次 check 5 中遵守）
