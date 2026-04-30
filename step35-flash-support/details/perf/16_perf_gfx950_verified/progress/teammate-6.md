# Teammate 6 Progress

## 任务范围
H1 验证（强制 CK 2-stage for per_1x128 FP8 blockscale）。
串行执行 #402 → #403 → #404 → #405。

## 已完成 Items

### [#402] patch 实施
- 文件：`/home/hanchang/junlin12_repos/aiter/aiter/fused_moe.py` L881-885
- diff（applied，已通过 git diff 验证）：
  ```
  -                # for fp8 blockscale, ck has better performance so disable assembly kernel
  -                run_1stage = token > 32 and (inter_dim % 256 == 0)
  +                # [TEMP H1 VERIFICATION PATCH — DO NOT COMMIT]
  +                # Force CK 2-stage for per_1x128 FP8 blockscale on gfx950.
  +                # Original: run_1stage = token > 32 and (inter_dim % 256 == 0)
  +                run_1stage = False
  ```
- 清缓存 `rm -rf /root/.cache/atom/*`：DONE

### [#403] tp=2 with patch（GPU 0,1）
- 命令：`perf_bench.py --tp 2 --input-tokens 10240 --output-tokens 1024`
- 实际 input=10262, output=215（method A）
- **TTFT = 384 ms**（基线 388 ms，变化 **-1.0%**，无显著变化）
- **TPOT = 12.346 ms/token**（基线 12.28 ms，变化 **+0.5%**，无显著变化）
- throughput_decode = 81.00 tokens/s
- VRAM 归零（GPU0/1）：PASS
- 日志确认 patch 生效：`run_1stage = False, ksplit = 4 q_type = QuantType.per_1x128 ... using 2stage default`
- 完整日志：`logs/h1_tp2.log` + `logs/h1_tp2_full.log`

### [#404] tp=4 with patch（GPU 0,1,2,3）
- 命令：`perf_bench.py --tp 4 --input-tokens 10240 --output-tokens 1024`
- 实际 input=10262, output=232（method A）
- **TTFT = 248 ms**（基线 241 ms，变化 **+2.9%**，在噪声范围内）
- **TPOT = 12.549 ms/token**（基线 12.50 ms，变化 **+0.4%**，无显著变化）
- throughput_decode = 79.69 tokens/s
- VRAM 归零（GPU0-3）：PASS
- 完整日志：`logs/h1_tp4.log` + `logs/h1_tp4_full.log`

### [#405] patch 还原
- `git checkout -- aiter/fused_moe.py`：DONE
- `git diff aiter/fused_moe.py` 结果：（空，clean）
- `git status` 结果：clean（无未追踪/未暂存改动）
- 再次 `rm -rf /root/.cache/atom/*`：DONE

## H1 初步结论

| 指标 | 基线 (run_1stage=token>32 ∧ inter%256==0) | H1 patch (run_1stage=False) | Δ |
|------|------|------|---|
| tp=2 TTFT | 388 ms | 384 ms | **-1.0%**（噪声内）|
| tp=2 TPOT | 12.28 ms | 12.346 ms | **+0.5%**（噪声内）|
| tp=4 TTFT | 241 ms | 248 ms | **+2.9%**（噪声内，反而略升）|
| tp=4 TPOT | 12.50 ms | 12.549 ms | **+0.4%**（噪声内）|

- TTFT 变化：**无显著变化**（tp=2 -1.0%，tp=4 +2.9%，皆在 ±5% 噪声带内）
- TPOT 变化：**大致不变**（两者皆 < 1%）
- **H1 排除**：强制 CK 2-stage 既未带来 TTFT 显著下降，也未拖慢 TPOT。说明对于当前模型/序列长度（input=10262）和 inter_dim 的组合，原 1stage assembly kernel vs CK 2stage 在 prefill 阶段性能基本等价；H1 假设（"运行时选错 1stage assembly kernel 导致 TTFT 偏慢"）**不成立**。

## 关键发现

1. **patch 生效证据**：日志中明确出现 `run_1stage = False ... using 2stage default for (..., QuantType.per_1x128, ...)`，说明 per_1x128 路径已切换到 CK 2stage，覆盖原本会走的 1stage assembly。
2. **prefill 性能瓶颈不在 MoE 1stage/2stage 选择**：两种 kernel 在 input≈10k 下产出几乎相同的 TTFT，根因应在别处（候选：attention prefill kernel、tuned_gemm 缺失（日志反复出现 "not found tuned config in bf16_tuned_gemm.csv"）、张量并行通信、layout/quantization overhead 等）。
3. **decode 路径无 regression**：TPOT 完全等价，说明 patch 哪怕保留也不会损害 decode；但既然 prefill 没收益，没必要改。
4. **token > 32 条件本身已限制了 1stage 仅用于 prefill 大 batch**：decode 阶段 token≤16 不会走 1stage，所以本 patch 对 TPOT 没有理论影响——实测确认。

## 给 Lead 的建议

1. **#406 RESULTS.md 更新**：将 H1 标记为 **排除**（rejected by experiment, 2026-04-29）。证据：tp=2/tp=4 双向都未见 TTFT 显著变化。
2. **后续根因优先级建议**（按数据指向重新排序）：
   - **H? 新增（tuned_gemm 覆盖率）**：日志反复出现 `not found tuned config in /tmp/aiter_configs/bf16_tuned_gemm.csv, will use default config! using torch solution:0`，prefill 阶段大量 `M=10262` 的 GEMM 全走 torch fallback。这可能是 prefill 真正瓶颈，建议作为新假设 H6 优先验证。
   - 其余 H2/H3/H4 维持原优先级。
3. **patch 已彻底还原**，仓库 clean，无需 lead 进一步清理。

## 收尾存档
- tool calls 累计：~12 次（含 ToolSearch 1 + Bash 6 + Read 2 + Edit 2 + Write 1）
- 关键文件路径：
  - patch 目标文件：`/home/hanchang/junlin12_repos/aiter/aiter/fused_moe.py` L881-883
  - 基线 reference：`/home/hanchang/project_summary/step35-flash-support/15_perf_tp2_tp4_tp8_eval/perf_bench.py`
  - 日志：`/home/hanchang/project_summary/step35-flash-support/16_perf_gfx950_verified/logs/h1_tp2{,_full}.log`、`h1_tp4{,_full}.log`
  - proposed_fix：`/home/hanchang/project_summary/step35-flash-support/16_perf_gfx950_verified/proposed_fix_401.md`
- 实验耗时：tp=2 ~30s 实测 + ~60s 模型加载；tp=4 类似。
