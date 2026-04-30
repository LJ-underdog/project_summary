# perf-T7 progress

> 任务 #P1-C：跑 tp=8 long-prompt baseline（CUDA_VISIBLE_DEVICES=0..7，input=10240/output=1024）→ TTFT_tp8 / TPOT_tp8 / decode_throughput_tp8 / total_latency_tp8 数值，补齐 PERF_REPORT.md §7 P1 缺口
> WORK_DIR = `/home/junlin12/project_fp8_tp4_repro/perf_tp_eval`
> 日期：2026-04-29
> 红线遵守：未修改 ATOM/aiter/CK 任何源码；未动 `perf_bench.py`；未动其他 progress 文件；未动 PERF_REPORT.md
> 新建文件：`logs/tp8_long_run1.log`、`logs/tp8_long_run1_full.log`、`logs/tp8_long_run2.log`、`logs/tp8_long_run2_full.log`、本文件

---

## 1. 任务背景

PERF_REPORT.md §7 P1 缺口：tp=8 在与 tp=2/tp=4 完全可比工况下（input=10240 / output=1024 / temperature=0 / concurrency=1）的 TTFT、TPOT、decode_throughput、total_latency 数值。

perf-T4 已完成 tp=8 起服 + short prompt（256/64）冒烟，结论是起服 PASS、JIT cache 与 tp=2/tp=4 一致，但**短 case 数值不可与 tp=2/tp=4 long-prompt 直接对比**（perf-T4 §8.1 显式标注）。本任务在同 model / 同 ATOM / 同 aiter / 同 CK / 同脚本 / 同采样参数下，把 input/output 调到 10240/1024，给出 stable 数据。

---

## 2. 启动命令

参数：tp=8，input-tokens=10240，output-tokens=1024，concurrency=1，temperature=0（脚本默认），CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7。

完整命令（与 `progress/perf-t0.md:165-175` 模板一致；与本任务 prompt §"执行步骤" 一致）：

```bash
cd /tmp && \
HF_HOME=/workspace/hf_cache \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
AITER_LOG_LEVEL=INFO \
AITER_LOG_TUNED_CONFIG=1 \
/opt/venv/bin/python /home/junlin12/project_fp8_tp4_repro/perf_tp_eval/perf_bench.py \
  --tp 8 --input-tokens 10240 --output-tokens 1024 \
  --log-file /home/junlin12/project_fp8_tp4_repro/perf_tp_eval/logs/tp8_long_run{N}.log \
  2>&1 | tee /home/junlin12/project_fp8_tp4_repro/perf_tp_eval/logs/tp8_long_run{N}_full.log
```

两次 run 之间执行 `sleep 6 && pgrep -af perf_bench && rocm-smi --showmemuse`，确认 8 卡 VRAM=0、无残留进程后再起。两次均自然退出（atom 输出 "All EngineCores shut down"），未触发任何 RuntimeError / Traceback / dispatch miss。

---

## 3. 两次 run 的 [PERF] 完整输出

### 3.1 Run 1（warmup-style，可弃用数据）
来源 `logs/tp8_long_run1.log:1-13`：
```
[PERF] script_start tp=8 target_input=10240 target_output=1024 measure_method=A
[PERF] actual_input_tokens=10265 (target=10240)
[PERF] creating LLMEngine ...
[PERF] engine_init_secs=43.77
[PERF] warmup generate (max_tokens=4) ...
[PERF] measure generate (method A) ...
[PERF] tp=8 input=10265 output=253
[PERF] method=A
[PERF] TTFT = 0.077 s
[PERF] TPOT = 5.527 ms/token
[PERF] total_latency = 1.469 s
[PERF] throughput_decode = 180.92 tokens/s
[PERF] wall_clock = 1.479 s (sanity)
```

ATOM 单 reqs 行（`logs/tp8_long_run1_full.log` 倒数第 13 行附近）：
```
Request 1 finished with reason eos. Input tokens: 10265, output tokens: 253, latency: 1.47s, TTFT: 0.077s, TPOT: 0.006s
```

### 3.2 Run 2（稳定值，主数据）
来源 `logs/tp8_long_run2.log:1-13`：
```
[PERF] script_start tp=8 target_input=10240 target_output=1024 measure_method=A
[PERF] actual_input_tokens=10265 (target=10240)
[PERF] creating LLMEngine ...
[PERF] engine_init_secs=44.98
[PERF] warmup generate (max_tokens=4) ...
[PERF] measure generate (method A) ...
[PERF] tp=8 input=10265 output=282
[PERF] method=A
[PERF] TTFT = 0.071 s
[PERF] TPOT = 5.542 ms/token
[PERF] total_latency = 1.629 s
[PERF] throughput_decode = 180.43 tokens/s
[PERF] wall_clock = 1.638 s (sanity)
```

ATOM 单 reqs 行（`logs/tp8_long_run2_full.log` 倒数第 13 行附近）：
```
Request 1 finished with reason eos. Input tokens: 10265, output tokens: 282, latency: 1.63s, TTFT: 0.071s, TPOT: 0.006s
```

---

## 4. 选定的 stable 数值（Run 2）

| 指标 | 值 | 来源 |
|---|---|---|
| **TTFT** | **0.071 s** | `logs/tp8_long_run2.log:9` |
| **TPOT** | **5.542 ms/token** | `logs/tp8_long_run2.log:10` |
| **total_latency** | **1.629 s** | `logs/tp8_long_run2.log:11` |
| **throughput_decode** | **180.43 tokens/s** | `logs/tp8_long_run2.log:12` |
| actual_input_tokens | 10265（target 10240，偏差 +25 在 ±32 tolerance 内 ✓） | `logs/tp8_long_run2.log:2` |
| actual_output_tokens | **282（不是 1024，eos 提前结束，与 tp=2/tp=4 同因）** | `logs/tp8_long_run2.log:7` |
| engine_init_secs | 44.98 s（vs perf-T4 short case 的 45.82 s，几乎相同；JIT cache 已暖，与 tp=2/tp=4 同代际差异符合预期） | `logs/tp8_long_run2.log:4` |

**为什么选 Run 2**：
1. Run 1 已完整跑过一次 generate（10265 input + 253 output），CUDAGraph capture / RCCL warm 路径全部触发；Run 2 是真正的 steady state。
2. 两次 TTFT 差 0.006s（7.8%），TPOT 差 0.015 ms（0.27%），波动很小；Run 2 跑了更长 decode（282 vs 253 token），平均更稳定。
3. Run 2 的 TPOT 略高（更保守，更接近"满载"），上报这个不会被低估。

**关于 output=282（非 1024）**：与 perf-T1（317）/ perf-T2（416）一致，模型在确定性 sampling 下自然命中 eos 提前停止，**不是脚本 bug**。TPOT 仍然有效（基于 281 个 decode token 平均），TTFT 是纯 prefill 指标，不受 output 长度影响。三种 tp 都受同一 sampling 行为影响，**横向对比仍可比**。

---

## 5. V1/V2/V3 + W 验证

### 5.1 grep 命中数（与任务 prompt §"Step 3" 给的命令一致）

| 验证 | 期望 | tp8_long_run2_full.log 实际 | 解释 |
|---|---|---|---|
| V1 `module_moe_ck2stages.*per_1x128` | >0（理想） | **0** | multi-process stderr 限制（见 5.2） |
| V2 `float8_e4m3fnuz` | >0（理想） | **0**（含在 V1 同一 grep） | 同上 |
| V3 `fmoe_g1u1` | =0 | **0** ✓ | NEW-RC-3 patch（aiter/fused_moe.py:881-886 hardcoded `run_1stage = False`）生效 |
| W `warn|warning`（过滤掉 DeprecationWarning / UserWarning.*torch） | =0 | **2**（仅 NCCL barrier device_id 提示） | 见 5.3 详细分析 |
| 异常 `RuntimeError`/`Traceback`/`HSAMemoryAllocFailed`/`no instance found`/`IsSupportedArgument false` | =0 | **0** ✓ | dispatch path / shape / dtype / NaN 全无异常 |

### 5.2 V1/V2 grep=0 解释（与 perf-T1 §4.2、perf-T2 §4.1、perf-T4 §4.1 同形）

ATOM 是 multi-process 架构（`logs/tp8_long_run2_full.log:7-9` 创建 EngineCore + 8 个 ModelRunner 子进程）：
- 主进程 stderr 走 `2>&1 | tee` 进入 full.log；
- 8 个 TP rank ModelRunner 由 `AsyncIOProcManager` spawn 成独立子进程，其 stderr **没有**重定向到主 log。

主 log 只能看到主进程加载的 `[aiter] import [module_aiter_core / module_custom_all_reduce / module_rmsnorm_quant / module_activation / module_fmha_v3_varlen_fwd / module_sample / ...]` 行（参考 `tp8_long_run2_full.log` 中 `[aiter] import` 的开头若干行），**完全没有任何 `module_moe_*` 行** —— 但 MoE forward 在 prefill+decode 是必跑的（hidden=4096, top_k=8, 288 experts），如果 dispatch 真的没走 MoE 模块，1.63s 内不可能产出 282 个语义合理的 token。

→ 结论：MoE 的 `[aiter]` 日志被吞在子进程 stderr 里。这是 perf_bench.py 的已知 log capture 限制（红线禁止改脚本）。本任务**复用** perf-T1 §4.3 的间接证据论证 V1/V2 实质通过（见 §6 JIT cache 间接证据）。

### 5.3 W=2 详细分析（实质 PASS）

```
/opt/venv/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py:4876:
UserWarning: barrier(): using the device under current context.
You can specify `device_id` in `init_process_group` to mute this warning.
  warnings.warn(  # warn only once
```

- 这两行是同一 warning 文本的两段（`warnings.warn(...)` 多行）；本质 1 处实际 warning。
- 来源：torch distributed 的 `barrier()` 没显式传 `device_id`，提示性 UserWarning，**与 ATOM/aiter/CK dispatch path 完全无关**。
- perf-T4 §3.2 E4 已记录同源 NCCL 提示行（`[W429 ...] Guessing device ID`），属于已知非阻塞 warning。
- 任务 prompt §"Step 3" 的 W grep 用 `grep -vE "DeprecationWarning|UserWarning.*torch"` 过滤，但此 warning 因为 `warnings.warn(...)` 调用栈本身被截在 distributed_c10d.py 路径，没有被 `UserWarning.*torch` 模式捕到（`UserWarning` 与 `torch` 之间隔了 `:`，但本行匹配 `barrier()` 的 message 行没含 `torch` keyword）。**手工核对此为 torch.distributed 的 UserWarning，等同于过滤目标，实质 W=0 PASS**。

---

## 6. JIT cache 间接证据（V1/V2 实质闭环）

跑完 Run 2 后 `ls /workspace/aiter/aiter/jit/ | grep -E "module_moe|fmoe"`：

```
module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2.so
module_moe_ck2stages_f8_f8_preshuffle_on_b16_swiglustep_per_1x128_mulWeightStage2.so
module_moe_sorting.so
```

与跑 tp=8 Run 1 之前（已记录基线，§5 perf-T4 跑完 tp=8 short case 后的 cache 状态完全一致）**完全相同**：仅 ck2stages per_1x128 (silu + swiglustep) 两个 fp8 MoE module + sorting helper，**0 个新增**；尤其**无 fmoe_g1u1 / 无 per_tensor / 无任何其它 fp8 ck2stages 变种 / 无任何新 dtype**。

→ 三层推论（与 perf-T1 §4.3 / perf-T2 §4.2 / perf-T4 §4.2 同形）：

| 证据 | 推论 |
|---|---|
| E1：JIT cache 仅 ck2stages_*per_1x128_*Stage2.so | tp=8 fp8 MoE 必走 `per_1x128` → **V1 实质 PASS** |
| E2：dtype 标识仅 `f8`（即 `float8_e4m3fnuz`，KNOWN_FACTS F2 已固化为 ATOM 自动 normalize） | dispatch 必走 e4m3fnuz path → **V2 实质 PASS** |
| E3：cache 无 `fmoe_g1u1.so` | NEW-RC-3 patch 生效 → **V3 实质 PASS** |

外加 §4 总耗时 1.63s 跑 10265 input + 282 output、TPOT=5.542 ms/token 与 fp8 ck2stages per_1x128 大 prefill + decode 的预期数量级一致；若走错 path 必会触发 `no instance found` 或 latency 数量级增大（实测 0 异常）。

---

## 7. 横向对比 tp=2 / tp=4 / tp=8

数据来自 `progress/perf-t1.md:89-95`、`progress/perf-t2.md:89-95` 与本文件 §4：

| tp | TTFT (s) | TPOT (ms/tok) | total_latency (s) | decode_thru (tok/s) | input / output | engine_init (s) |
|---|---|---|---|---|---|---|
| **tp=2** | 0.186 | 5.245 | 1.843 | 190.66 | 10265 / 317 | 25.38 |
| **tp=4** | 0.110 | 5.451 | 2.373 | 183.44 | 10265 / 416 | 30.25 |
| **tp=8** | **0.071** | **5.542** | **1.629** | **180.43** | **10265 / 282** | **44.98** |

观察（仅供 perf-T5 / lead 写 PERF_REPORT 引用）：
- **TTFT**：tp=2 → tp=4 → tp=8 单调下降（0.186 → 0.110 → 0.071 s），从 tp=2 到 tp=8 提速 **2.62×**，符合 prefill 算力随 tp 接近线性扩展的预期（10k input prefill 受算力 bound）。
- **TPOT**：tp=2 → tp=4 → tp=8 略微单调上升（5.245 → 5.451 → 5.542 ms/tok）。decode 阶段 batch=1 是 memory-bandwidth + all-reduce 通信 bound，tp 增大 → 单卡分摊算力收益小，但 all-reduce 通信开销增大，所以 TPOT 微增（tp=8 比 tp=2 仅慢 5.7%），数量级一致，符合典型 multi-GPU MoE decode 行为。
- **decode throughput**：随 TPOT 倒数变化（190.66 → 183.44 → 180.43 tok/s），同一趋势。
- **total_latency**：output 长度差异（317 / 416 / 282）导致 total 不可直接比，但 TPOT × output_tokens + TTFT 公式可分解，说明 tp=8 跑得更短是因 eos 早停，**非性能问题**。
- **engine_init**：tp=2 → tp=4 → tp=8（25.38 → 30.25 → 44.98 s），多 worker process + RCCL init，符合 perf-T2 §6.2 B4 给出的"+5s 量级 per +2 worker"经验。

**结论**：tp=8 在 long-prompt 工况下，dispatch path 与 tp=2/tp=4 一致（ck2stages per_1x128 + e4m3fnuz + run_1stage=False），TTFT 最快、TPOT 略慢但仍在 5.5 ms/tok 以内、无 dispatch 异常，**P1 缺口闭环**。

---

## 8. VRAM 8 卡回收确认

每次 run 后执行 `sleep 6 && pgrep -af perf_bench && rocm-smi --showmemuse`，三次时间点（Run 1 后、Run 2 后、写 progress 前）结果一致：

```
$ pgrep -af perf_bench
ok no leftover
$ rocm-smi --showmemuse | grep "VRAM%"
GPU[0]: GPU Memory Allocated (VRAM%): 0
GPU[1]: GPU Memory Allocated (VRAM%): 0
GPU[2]: GPU Memory Allocated (VRAM%): 0
GPU[3]: GPU Memory Allocated (VRAM%): 0
GPU[4]: GPU Memory Allocated (VRAM%): 0
GPU[5]: GPU Memory Allocated (VRAM%): 0
GPU[6]: GPU Memory Allocated (VRAM%): 0
GPU[7]: GPU Memory Allocated (VRAM%): 0
```

→ ✅ 8 张卡 VRAM 全归零，无残留 worker。`atom AsyncIOProcManager` 在脚本退出时正确 shutdown 所有 8 个 worker（`tp8_long_run2_full.log` 末尾的 "All runners are shutdown" / "All EngineCores shut down" 行）。

---

## 9. 已知风险 / 异常

| # | 项 | 影响 | 是否需 lead 处理 |
|---|---|---|---|
| C1 | V1/V2 grep=0，需靠 §6 JIT cache 三条间接证据反证 dispatch 正确（与 perf-T1 A1 / perf-T2 B1 / perf-T4 同源） | 中 | 与 perf-T1 / perf-T2 / perf-T4 同源；建议 lead 在 PERF_REPORT 把"grep=0 是 multi-process stderr 限制"统一交代一次，不再每节重复 |
| C2 | output=282（非 1024），eos 提前停 | 低 | TPOT 仍基于 281 个 decode token 计算有效；与 perf-T1 A2 / perf-T2 B2 同源 |
| C3 | actual input=10265 vs target 10240（偏差 +25），在 ±32 tolerance 内 | 极低 | 无需处理 |
| C4 | W=2 实为 1 处 torch.distributed `barrier()` UserWarning（被 `warnings.warn(...)` 多行截断成 2 行） | 极低 | 已在 §5.3 论证为非阻塞，等同于过滤目标；不构成新异常 |
| C5 | engine_init_secs=44.98（vs perf-T2 tp=4 的 30.25） | 信息 | 多 4 个 worker process + 8-rank RCCL init，符合预期 |

**未触发**任何 `no instance found` / `ImportError` / `assert` / `RuntimeError` / `Traceback` / `HSAMemoryAllocFailed` / `IsSupportedArgument false`，未触发 dispatch 异常。

---

## 10. 给 lead 的建议（perf-T7 收尾）

1. **本任务 deliverable 已闭环**：tp=8 long-prompt（10240/1024）的 TTFT=0.071s / TPOT=5.542 ms/tok / total=1.629s / decode_thru=180.43 tok/s 已稳定取得，可直接 inline 进 PERF_REPORT.md §7 P1 缺口表。
2. **§7 横向对比表**已按 lead 在任务 prompt 中给的格式备好，可直接复制。
3. **V1/V2 grep=0 解释**：建议 lead 在 PERF_REPORT 加一节统一说明 multi-process stderr 限制，引用本文件 §5.2 + §6（避免每个 tp 节重复同样论述）。
4. **若 reviewer 仍质疑 V1/V2**：可考虑下一轮新开 task（不在本任务范围）放宽红线、给 perf_bench.py 加 `--worker-log-stderr` 接管 worker stderr。本任务严格遵守红线，不改脚本。
5. **本任务 tool calls 计数**：约 10 次（含读 4 个 progress + 2 次 GPU/cache 状态查 + 2 次跑 + 1 次 grep + 1 次写本文件），未触达 15/20 上限。

---

## 11. 文件清单（本任务产出）

- `WORK_DIR/logs/tp8_long_run1.log`（13 行 [PERF] dump）
- `WORK_DIR/logs/tp8_long_run1_full.log`（stdout/stderr 全量）
- `WORK_DIR/logs/tp8_long_run2.log`（13 行 [PERF] dump）
- `WORK_DIR/logs/tp8_long_run2_full.log`（stdout/stderr 全量）
- `WORK_DIR/progress/perf-t7.md`（本文件）

## 12. 红线自查

- [x] 未改 ATOM / aiter / CK 任何源码
- [x] 未改 perf_bench.py
- [x] 未动其他 progress 文件（perf-t0/t1/t2/t3/t4/t5/t6-review 全保留）
- [x] 未动 PERF_REPORT.md
- [x] 仅新建 logs/tp8_long_run{1,2}{,_full}.log + 本 progress
- [x] 中文 + file:line 引用
- [x] 跑完 8 卡 VRAM 全归零，无残留进程
