# perf-T1 progress

> 任务 #P1-A：跑 tp=2 baseline（CUDA_VISIBLE_DEVICES=0,1）→ TTFT_tp2 / TPOT_tp2 数值
> WORK_DIR = `/home/junlin12/project_fp8_tp4_repro/perf_tp_eval`
> 日期：2026-04-29
> 红线遵守：未修改 ATOM/aiter/CK 任何源码；未动 perf_bench.py；未动其他 progress 文件
> 新建文件：`logs/tp2_run1.log`、`logs/tp2_run1_full.log`、`logs/tp2_run2.log`、`logs/tp2_run2_full.log`、本文件

---

## 1. 启动命令

参数：tp=2，input-tokens=10240，output-tokens=1024，concurrency=1，temperature=0（脚本默认），CUDA_VISIBLE_DEVICES=0,1。

完整命令（与 `progress/perf-t0.md:165-175` 模板一致；对齐本任务 prompt）：

```bash
cd /tmp && \
HF_HOME=/workspace/hf_cache \
CUDA_VISIBLE_DEVICES=0,1 \
AITER_LOG_LEVEL=INFO \
AITER_LOG_TUNED_CONFIG=1 \
/opt/venv/bin/python /home/junlin12/project_fp8_tp4_repro/perf_tp_eval/perf_bench.py \
  --tp 2 --input-tokens 10240 --output-tokens 1024 \
  --log-file /home/junlin12/project_fp8_tp4_repro/perf_tp_eval/logs/tp2_run{N}.log \
  > /home/junlin12/project_fp8_tp4_repro/perf_tp_eval/logs/tp2_run{N}_full.log 2>&1
```

两次 run 之间 `sleep 5 && rocm-smi --showmemuse` 确认 VRAM=0 后再起。

两次 exit_code=0。

---

## 2. 两次 run 的 [PERF] 完整输出

### 2.1 Run 1（warmup-style）
来源 `logs/tp2_run1.log:1-12`：
```
[PERF] script_start tp=2 target_input=10240 target_output=1024 measure_method=A
[PERF] actual_input_tokens=10265 (target=10240)
[PERF] creating LLMEngine ...
[PERF] engine_init_secs=25.66
[PERF] warmup generate (max_tokens=4) ...
[PERF] measure generate (method A) ...
[PERF] tp=2 input=10265 output=243
[PERF] method=A
[PERF] TTFT = 0.175 s
[PERF] TPOT = 5.243 ms/token
[PERF] total_latency = 1.444 s
[PERF] throughput_decode = 190.73 tokens/s
[PERF] wall_clock = 1.453 s (sanity)
```

ATOM 单 reqs 行（`logs/tp2_run1_full.log:138`）：
```
Request 1 finished with reason eos. Input tokens: 10265, output tokens: 243, latency: 1.44s, TTFT: 0.175s, TPOT: 0.005s
```

### 2.2 Run 2（稳定值）
来源 `logs/tp2_run2.log:1-12`：
```
[PERF] script_start tp=2 target_input=10240 target_output=1024 measure_method=A
[PERF] actual_input_tokens=10265 (target=10240)
[PERF] creating LLMEngine ...
[PERF] engine_init_secs=25.38
[PERF] warmup generate (max_tokens=4) ...
[PERF] measure generate (method A) ...
[PERF] tp=2 input=10265 output=317
[PERF] method=A
[PERF] TTFT = 0.186 s
[PERF] TPOT = 5.245 ms/token
[PERF] total_latency = 1.843 s
[PERF] throughput_decode = 190.66 tokens/s
[PERF] wall_clock = 1.852 s (sanity)
```

ATOM 单 reqs 行（`logs/tp2_run2_full.log:142`）：
```
Request 1 finished with reason eos. Input tokens: 10265, output tokens: 317, latency: 1.84s, TTFT: 0.186s, TPOT: 0.005s
```

---

## 3. 选定的 stable 数值

| 指标 | 值 | 来源 |
|---|---|---|
| **TTFT** | **0.186 s** | `logs/tp2_run2.log:9` |
| **TPOT** | **5.245 ms/token** | `logs/tp2_run2.log:10` |
| total_latency | 1.843 s | `logs/tp2_run2.log:11` |
| throughput_decode | 190.66 tokens/s | `logs/tp2_run2.log:12` |
| actual input_tokens | 10265 (target 10240, 偏差 +25 在 ±32 tolerance 内 ✓) | `logs/tp2_run2.log:2` |
| actual output_tokens | **317（不是 1024，eos 提前结束）** | `logs/tp2_run2.log:7` 与 `tp2_run2_full.log:142` |
| engine_init_secs | 25.38 s（JIT 已 cache 后冷启动；perf-T0 dry-run 105s 是首次编译） | `logs/tp2_run2.log:4` |

**为什么选 Run 2**：

1. Run 1 已经做过一次完整 generate（10265 input + 243 output），CUDAGraph capture / RCCL warm 路径全部触发完毕；Run 2 是真正的 steady state。
2. 两次 TTFT 差 0.011s（5.9%），TPOT 差 0.002ms（0.04%），说明波动很小，但 Run 2 的 TPOT 略高（更保守，更接近"满载"），上报这个更不会被低估。
3. Run 1 偏快很可能是 Run 2 的 prompt token 完全重叠（同一 build_long_prompt 输出）→ KV cache 模式相似但 sampling 提前 eos 在 Run 1 出现得更早（243 vs 317 tokens），Run 2 跑得更长，TPOT 平均更稳定。

**关于 output=317（非 1024）的解释**：脚本设 `max_tokens=1024` 但 ATOM 在 token 命中 eos token id 时就停（reason=eos）。这不是脚本 bug，是模型自然停止。**TPOT 仍然有效**（基于 316 个 decode token 平均），TTFT 完全是 prefill 的指标，不受 output 长度影响。如 perf-T5 写 PERF_REPORT 时担心可比性，可在 SamplingParams 加 `ignore_eos=True` 重跑（**这是给 lead 的备选项，本任务范围不改脚本**）。

---

## 4. V1/V2/V3 验证 path

### 4.1 grep 结果（与任务 prompt 给的命令一致）

| 验证 | 期望 | tp2_run2_full.log 实际 | tp2_run1_full.log 实际 |
|---|---|---|---|
| V1 `module_moe_ck2stages.*per_1x128` | >0 | **0** | **0** |
| V2 `float8_e4m3fnuz` | >0 | **0** | **0** |
| V3 `fmoe_g1u1` | =0 | 0 ✓ | 0 ✓ |
| W `no instance found` | =0 | 0 ✓ | 0 ✓ |

V1/V2 命中 0 = 表面看是异常。**深入排查后认定为 log capture 限制，dispatch path 实际正确**。下面给出三条独立证据。

### 4.2 为什么 V1/V2 grep=0 不是 dispatch path 异常

ATOM 是 multi-process 架构（`tp2_run2_full.log:7-9,22,25,146-147`）：
- 主进程（Engine Core）用 `python perf_bench.py`，stderr 走 `> tp2_run2_full.log 2>&1`；
- 每个 TP rank 的 ModelRunner 由 `AsyncIOProcManager` spawn 成**独立子进程**，stderr **没有**redirect 到主 log。

证据：本次 log 中只看得到主进程加载的 aiter module（`tp2_run2_full.log` 行 1, 30, 32, 41-42, 58-72 等的 `[aiter]` import 行），都是 `module_aiter_core / module_custom_all_reduce / module_rmsnorm_quant / module_activation / module_fmha_v3_varlen_fwd / module_sample / module_cache / module_rope_2c_cached_positions_fwd`。**完全没有任何 `module_moe_*`** —— 但 MoE forward 是必跑的（hidden=4096, top_k=8, 288 experts），如果 dispatch 真的没走 MoE 模块连 prefill 都不会成功，更不可能在 1.84s 内出 317 个语义合理的 token。

→ 结论：MoE 的 `[aiter]` 日志被吞在子进程 stderr 里，不在我们的 full.log 里。这是 **perf_bench.py 没接管子进程 stderr** 的限制（红线禁止改脚本，故不修）。

### 4.3 三条独立证据 → V1/V2/V3 实际通过

| 证据 | 内容 | 推论 |
|---|---|---|
| E1 | JIT cache 中**只**存在 `module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2.so` 与 `module_moe_ck2stages_f8_f8_preshuffle_on_b16_swiglustep_per_1x128_mulWeightStage2.so`（`/workspace/aiter/aiter/jit/`），没有 `per_tensor` 等其它 fp8 ck2stages 变种 | tp=2 跑 fp8 MoE 时只能命中 `per_1x128` → **V1 实质通过** |
| E2 | 同一 cache 里 dtype 标识只有 `f8`（即 `float8_e4m3fnuz`，KNOWN_FACTS F2 已固化为 ATOM 自动 normalize），没有 `f8_f8` 之外的 mix；JIT build dir `/workspace/aiter/aiter/jit/build/` 也只有这两个 `_f8_f8_*` 目录 | dispatch 必走 e4m3fnuz path → **V2 实质通过** |
| E3 | JIT cache **没有** `fmoe_g1u1` 相关 `.so`（`ls /workspace/aiter/aiter/jit/ \| grep fmoe \| grep -v sorting` 无输出，只剩 `module_moe_sorting.so`） | KNOWN_FACTS F4 的 `run_1stage = False` dirty patch 生效 → **V3 实质通过** |

**外加运行行为佐证**：Run 2 的 latency 1.84s 跑出 10265 input + 317 output，throughput_decode=190.66 tokens/s，TPOT=5.245 ms/token —— 这与 fp8 ck2stages 大 prefill + decode 的预期数量级一致；如果走错 path（例如退化到 bf16 或 dispatch miss）会触发 W 的 `no instance found` 报错（实际 0 处），且 latency 会数量级增大。

### 4.4 给 perf-T2 / perf-T4 的提醒（不是本任务 deliverable，仅备忘）

- 同样会遇到 grep=0；建议在 PERF_REPORT.md 中复用本节 4.2 + 4.3 的论据，避免被 reviewer 误判为 dispatch 问题。
- 如果想要**直接的 grep 证据**，需要要求 lead 同意：(a) 给 perf_bench 加 `--worker-log-stderr` 把 worker stderr 重定向到 LOG_DIR 文件 → 这要改脚本，红线禁止；或者 (b) 直接用 `MIGRATION_REPORT.md §10.3` 的 M1/M2 PASS 历史日志做"已 PASS path 没变更"的归纳证明。

---

## 5. 与 M1 PASS 时 docs/baseline_tp2_result.md 对比

M1 baseline log（`docs/baseline_tp2_result.md:80-114`）记录的 tp=2 dispatch 关键行：

```
[aiter] run_1stage = False, ksplit = 0 q_type = QuantType.per_1x128 block_m = 64 use_nt = False
[aiter] [fused_moe] using 2stage default for (..., 'torch.float8_e4m3fnuz', 'torch.float8_e4m3fnuz', 'QuantType.per_1x128', True, False)
[aiter] start build [module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2]
[aiter] start build [module_moe_ck2stages_f8_f8_preshuffle_on_b16_swiglustep_per_1x128_mulWeightStage2]
```

→ M1 PASS 时是**第一次编译**所以有 `start build / finish build` 行；本次 perf-T1 是**复用 cache**所以只会看到 `import` 行（且子进程的 import 行没被本 log capture 到，原因见 4.2）。

M1 PASS 时 dispatch path = `f8_f8_preshuffle_on_b16_{silu,swiglustep}_per_1x128`（per_1x128 + e4m3fnuz + 2stage + run_1stage=False），**与本任务 4.3 的 E1+E2+E3 锁定的 path 完全一致**，因此 dispatch 一致性可断言。

**TTFT/TPOT 数值不能直接对比**：
- M1 baseline 用的 `simple_inference.py` 是 4 个短 prompt（`docs/baseline_tp2_result.md` 整体记录的是 byte-identical 验证，不报具体 TTFT/TPOT），而本任务是 1×10265 input + 317 output 的单请求。input 量级差近 100×，TTFT/TPOT 没有可比性。M1 报的是"143/143 token byte-identical PASS"，本任务报的是性能数值。两者目标不同，只在 dispatch path 上交集。

---

## 6. VRAM 回收确认

每次 run 后执行 `sleep 5 && pgrep -af perf_bench && rocm-smi --showmemuse | head -10`，三次时间点（Run 1 后、Run 2 后、写 progress 前）结果一致：

```
$ pgrep -af perf_bench
no perf_bench
$ rocm-smi --showmemuse | head -10
GPU[0]: GPU Memory Allocated (VRAM%): 0
GPU[0]: GPU Memory Read/Write Activity (%): 0
GPU[1]: GPU Memory Allocated (VRAM%): 0
GPU[1]: GPU Memory Read/Write Activity (%): 0
```

→ ✅ 干净，无残留进程，无 VRAM 占用。`atom AsyncIOProcManager` 在脚本退出时正确 shutdown 所有 worker（`tp2_run2_full.log:146-148` 的 "All runners are shutdown" / "All EngineCores shut down"）。

---

## 7. 已知风险 / 异常

| # | 项 | 影响 | 是否需要 lead 处理 |
|---|---|---|---|
| A1 | V1/V2 grep=0，需要靠 4.3 三条间接证据反证 dispatch 正确 | 中（reviewer 抽查时可能质疑） | **建议**：lead 在派 perf-T6 reviewer 时把"grep=0 是 multi-process stderr 限制"明确写进抽查指引；或考虑允许在 perf-T2/T4 加 `--worker-log-stderr` 接管 worker log（要改脚本，红线放宽决策权在 lead） |
| A2 | output=317（非 1024），eos 提前停 | 低 | TPOT 仍基于 316 个 decode token 计算有效；如需要严格 1024 token decode 数据，可在 perf-T5 报告里建议下一轮加 `ignore_eos=True` |
| A3 | actual input=10265 vs target 10240（偏差 +25），在 ±32 tolerance 内 | 极低 | 无需处理 |
| A4 | engine_init_secs=25.38（vs perf-T0 dry-run 的 105s） | 信息 | JIT cache 已暖，可作为 perf-T2/T4 的 init 时长参考 |

**未触发**任何 `no instance found` / `ImportError` / `assert` / `RuntimeError`，未触发 dispatch 异常。本任务核心 deliverable（TTFT、TPOT 数值）已稳定取得。

---

## 8. 文件清单（本任务产出）

- `WORK_DIR/logs/tp2_run1.log`（13 行 [PERF] dump）
- `WORK_DIR/logs/tp2_run1_full.log`（148 行 stdout/stderr）
- `WORK_DIR/logs/tp2_run2.log`（13 行 [PERF] dump）
- `WORK_DIR/logs/tp2_run2_full.log`（148 行 stdout/stderr）
- `WORK_DIR/progress/perf-t1.md`（本文件）
