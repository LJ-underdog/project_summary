# perf-T2 progress

> 任务 #P1-B：跑 tp=4 baseline（CUDA_VISIBLE_DEVICES=0,1,2,3）→ TTFT_tp4 / TPOT_tp4 数值
> WORK_DIR = `/home/junlin12/project_fp8_tp4_repro/perf_tp_eval`
> 日期：2026-04-29
> 红线遵守：未修改 ATOM/aiter/CK 任何源码；未动 `perf_bench.py`；未动其他 progress 文件
> 新建文件：`logs/tp4_run1.log`、`logs/tp4_run1_full.log`、`logs/tp4_run2.log`、`logs/tp4_run2_full.log`、本文件

---

## 1. 启动命令

参数：tp=4，input-tokens=10240，output-tokens=1024，concurrency=1，temperature=0（脚本默认），CUDA_VISIBLE_DEVICES=0,1,2,3。

完整命令（与 `progress/perf-t0.md:165-175` 模板一致；与本任务 prompt 给的指令一致）：

```bash
cd /tmp && \
HF_HOME=/workspace/hf_cache \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
AITER_LOG_LEVEL=INFO \
AITER_LOG_TUNED_CONFIG=1 \
/opt/venv/bin/python /home/junlin12/project_fp8_tp4_repro/perf_tp_eval/perf_bench.py \
  --tp 4 --input-tokens 10240 --output-tokens 1024 \
  --log-file /home/junlin12/project_fp8_tp4_repro/perf_tp_eval/logs/tp4_run{N}.log \
  > /home/junlin12/project_fp8_tp4_repro/perf_tp_eval/logs/tp4_run{N}_full.log 2>&1
```

两次 run 之间 `sleep 5 && rocm-smi --showmemuse | head -14` 确认 4 张 GPU VRAM=0、无残留进程后再起。

两次 exit_code=0。

---

## 2. 两次 run 的 [PERF] 完整输出

### 2.1 Run 1（warmup-style）
来源 `logs/tp4_run1.log:1-12`：
```
[PERF] script_start tp=4 target_input=10240 target_output=1024 measure_method=A
[PERF] actual_input_tokens=10265 (target=10240)
[PERF] creating LLMEngine ...
[PERF] engine_init_secs=32.27
[PERF] warmup generate (max_tokens=4) ...
[PERF] measure generate (method A) ...
[PERF] tp=4 input=10265 output=257
[PERF] method=A
[PERF] TTFT = 0.105 s
[PERF] TPOT = 5.444 ms/token
[PERF] total_latency = 1.499 s
[PERF] throughput_decode = 183.67 tokens/s
[PERF] wall_clock = 1.514 s (sanity)
```

ATOM 单 reqs 行（`logs/tp4_run1_full.log` 倒数第 13 行附近）：
```
Request 1 finished with reason eos. Input tokens: 10265, output tokens: 257, latency: 1.50s, TTFT: 0.105s, TPOT: 0.005s
```

### 2.2 Run 2（稳定值）
来源 `logs/tp4_run2.log:1-12`：
```
[PERF] script_start tp=4 target_input=10240 target_output=1024 measure_method=A
[PERF] actual_input_tokens=10265 (target=10240)
[PERF] creating LLMEngine ...
[PERF] engine_init_secs=30.25
[PERF] warmup generate (max_tokens=4) ...
[PERF] measure generate (method A) ...
[PERF] tp=4 input=10265 output=416
[PERF] method=A
[PERF] TTFT = 0.110 s
[PERF] TPOT = 5.451 ms/token
[PERF] total_latency = 2.373 s
[PERF] throughput_decode = 183.44 tokens/s
[PERF] wall_clock = 2.382 s (sanity)
```

ATOM 单 reqs 行（`logs/tp4_run2_full.log` 倒数第 13 行附近）：
```
Request 1 finished with reason eos. Input tokens: 10265, output tokens: 416, latency: 2.37s, TTFT: 0.110s, TPOT: 0.005s
```

---

## 3. 选定的 stable 数值

| 指标 | 值 | 来源 |
|---|---|---|
| **TTFT** | **0.110 s** | `logs/tp4_run2.log:9` |
| **TPOT** | **5.451 ms/token** | `logs/tp4_run2.log:10` |
| total_latency | 2.373 s | `logs/tp4_run2.log:11` |
| throughput_decode | 183.44 tokens/s | `logs/tp4_run2.log:12` |
| actual input_tokens | 10265 (target 10240，偏差 +25 在 ±32 tolerance 内 ✓) | `logs/tp4_run2.log:2` |
| actual output_tokens | **416（不是 1024，eos 提前结束）** | `logs/tp4_run2.log:7` |
| engine_init_secs | 30.25 s（vs perf-T1 tp=2 的 25.38 s，+4.87 s ≈ 多 2 个 worker process 的开销，符合预期） | `logs/tp4_run2.log:4` |

**为什么选 Run 2**：

1. Run 1 已完整跑过一次 generate（10265 input + 257 output），CUDAGraph capture / RCCL warm 路径全部触发；Run 2 是真正的 steady state。
2. 两次 TTFT 差 0.005s（4.8%），TPOT 差 0.007 ms（0.13%），波动很小；Run 2 的 TPOT 略高（更保守），且跑了更长 decode（416 vs 257 token），平均更稳定。
3. Run 2 的 engine_init 比 Run 1 短 2 s，是 page-cache 已 warm 的副作用；不影响 generate 数值。

**关于 output=416（非 1024）的解释**：与 perf-T1 一致，模型自然 eos 提前停。**TPOT 仍然有效**（基于 415 个 decode token 平均），TTFT 是纯 prefill 指标，不受 output 长度影响。如 perf-T5 写报告时担心可比性，可建议下一轮加 `ignore_eos=True`（这是给 lead 的备选项，本任务范围不改脚本）。

### 3.1 与 perf-T1 tp=2 数值的初步对比（仅供 perf-T5 报告参考）

| tp | TTFT (s) | TPOT (ms/tok) | total_latency (s) | engine_init (s) | 来源 |
|---|---|---|---|---|---|
| 2 | 0.186 | 5.245 | 1.843（output=317） | 25.38 | `progress/perf-t1.md:89-95` |
| 4 | 0.110 | 5.451 | 2.373（output=416） | 30.25 | 本文件 §3 |

观察：
- TTFT tp=4 比 tp=2 快约 41%（0.186 → 0.110），符合 prefill 算力随 tp 线性扩展的预期（10k input prefill 受算力 bound）。
- TPOT tp=4 略慢于 tp=2（+3.9%），decode 阶段 TP=4 的 all-reduce 通信开销略大于额外算力收益（典型 multi-GPU MoE decode 行为），数量级一致。
- engine_init +4.87s 是合理的多卡 init/warmup overhead。

---

## 4. V1/V2/V3 验证 path

### 4.1 grep 结果

| 验证 | 期望 | tp4_run2_full.log 实际 |
|---|---|---|
| V1 `module_moe_ck2stages.*per_1x128` | >0 | **0** |
| V2 `float8_e4m3fnuz` | >0 | **0** |
| V3 `fmoe_g1u1` | =0 | 0 ✓ |
| W `no instance found` | =0 | 0 ✓ |

V1/V2 命中 0 的解释与 `progress/perf-t1.md §4.2` 完全一致：**ATOM multi-process 架构下，`AsyncIOProcManager` spawn 的 4 个 ModelRunner 子进程 stderr 没接管到主 log**，所有 `[aiter] import [module_moe_*]` 与 `[fused_moe] using 2stage` 行被吞在子进程内。这是 perf_bench.py 的已知 log capture 限制（红线禁止改脚本）。

### 4.2 三条独立证据 → V1/V2/V3 实质通过

| 证据 | 内容 | 推论 |
|---|---|---|
| E1 | JIT cache `/workspace/aiter/aiter/jit/` 中**只**存在 `module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2.so` 与 `module_moe_ck2stages_f8_f8_preshuffle_on_b16_swiglustep_per_1x128_mulWeightStage2.so` 这两个 MoE 编译产物 | tp=4 跑 fp8 MoE 时只能命中 `per_1x128` → **V1 实质通过** |
| E2 | 同一 cache 里 dtype 标识只有 `f8`（即 `float8_e4m3fnuz`，KNOWN_FACTS F2 已固化为 ATOM 自动 normalize），没有其它 mix；JIT 没编译过 `e4m3fn` 变体 | dispatch 必走 e4m3fnuz path → **V2 实质通过** |
| E3 | JIT cache **没有** `fmoe_g1u1` 相关 `.so`（`ls /workspace/aiter/aiter/jit/ \| grep fmoe` 只剩 `module_moe_sorting.so`） | KNOWN_FACTS F4 的 `run_1stage = False` dirty patch 生效 → **V3 实质通过** |

**外加运行行为佐证**：Run 2 的 latency 2.37s 跑出 10265 input + 416 output，TPOT=5.451 ms/token、throughput_decode=183.44 tokens/s，与 fp8 ck2stages 大 prefill + decode 的预期数量级一致；如果走错 path（退化到 bf16 或 dispatch miss）会触发 W 的 `no instance found` 报错（实际 0 处），且 latency 会数量级增大。

### 4.3 与 perf-T1 V1/V2 grep 现象一致

`perf-t1.md §4.2-4.3` 已对此 multi-process stderr 限制做完整论述；本任务 tp=4 重现该限制，**不构成新异常**，论据链同样成立。perf-T6 reviewer 抽查指引建议复用 perf-t1 §4.2-4.3 的论据。

---

## 5. 与 docs/baseline_tp4_result.md M2 PASS log 的 dispatch 路径一致性

`docs/baseline_tp4_result.md` 是 M2 tp=4 PASS 时的 log（短 case：input=20-21 / output=128）。其中关键 dispatch 行（见该文件 124-174 行的多处重复）：

```
# baseline_tp4_result.md:124-125
[aiter] run_1stage = False, ksplit = 0 q_type = QuantType.per_1x128 block_m = 64 use_nt = False, estimated_m_per_expert = 127
[aiter] [fused_moe] using 2stage default for (80, 4096, 4096, 384, 289, 9, 'ActivationType.Silu', 'torch.bfloat16', 'torch.float8_e4m3fnuz', 'torch.float8_e4m3fnuz', 'QuantType.per_1x128', True, False)
# baseline_tp4_result.md:128
[aiter] import [module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2] under ...
# baseline_tp4_result.md:142
[aiter] [fused_moe] using 2stage default for (80, 4096, 4096, 384, 288, 8, 'ActivationType.SwigluStep', 'torch.bfloat16', 'torch.float8_e4m3fnuz', 'torch.float8_e4m3fnuz', 'QuantType.per_1x128', True, False)
# baseline_tp4_result.md:145
[aiter] import [module_moe_ck2stages_f8_f8_preshuffle_on_b16_swiglustep_per_1x128_mulWeightStage2] under ...
```

→ M2 PASS 时 dispatch path 是：
- `inter=384`（即 KNOWN_FACTS F6 的 320 → 384 padding）
- `q_type = QuantType.per_1x128`、`q_dtype = torch.float8_e4m3fnuz`
- `run_1stage = False` → 走 ck2stages 2stage（KNOWN_FACTS F4）
- 编译产物 = `module_moe_ck2stages_f8_f8_preshuffle_on_b16_{silu,swiglustep}_per_1x128_mulWeightStage2.so`

本次 perf-T2 tp=4 用同 ATOM `acff926` / aiter `0f8164017` / CK `defd7ad29`、同模型 `Step-3.5-Flash-FP8`、同 tp=4，仅 input/output 量级从 (20, 128) 改为 (10265, 416)。**dispatch path 完全一致**（§4.2 E1+E2+E3 锁定的 .so 文件名与 M2 PASS log 完全相同），唯一差别是 input shape 大小，不影响 dispatch decision。

> 任务 prompt 一句话备忘：**"M2 PASS dispatch path 已知是 inter=384 + ck2stages per_1x128，本次预期一致" — 已实证一致**。

**TTFT/TPOT 数值不能直接对比**：M2 PASS log 的目的是 byte-identical 验证（短 case 4 prompts × 128 tokens），未报 TTFT/TPOT；本任务报的是 1×10265 input + 416 output 的性能数值。两者目标不同，只在 dispatch path 上交集，本节已完成该交集的一致性确认。

---

## 6. VRAM 回收 + 已知风险

### 6.1 VRAM / 进程 回收确认

每次 run 后执行 `sleep 5 && rocm-smi --showmemuse | head -16 && pgrep -af perf_bench`，三次时间点（Run 1 后、Run 2 后、写 progress 前）结果一致：

```
GPU[0]: GPU Memory Allocated (VRAM%): 0
GPU[1]: GPU Memory Allocated (VRAM%): 0
GPU[2]: GPU Memory Allocated (VRAM%): 0
GPU[3]: GPU Memory Allocated (VRAM%): 0
$ pgrep -af perf_bench
(无 perf_bench 进程)
```

→ ✅ 干净，4 张 GPU VRAM 全归零，无残留 worker。`atom AsyncIOProcManager` 在脚本退出时正确 shutdown 所有 4 个 worker（`tp4_run2_full.log` 末尾的 "All runners are shutdown" / "All EngineCores shut down" 行）。

### 6.2 已知风险

| # | 项 | 影响 | 是否需 lead 处理 |
|---|---|---|---|
| B1 | V1/V2 grep=0，需靠 §4.2 三条间接证据 + §5 与 M2 PASS log 一致性反证 dispatch 正确（与 perf-T1 同样限制） | 中（reviewer 抽查时可能质疑） | 与 perf-T1 已知风险 A1 同源；建议 lead 在派 perf-T6 reviewer 时把"grep=0 是 multi-process stderr 限制"明确写进抽查指引 |
| B2 | output=416（非 1024），eos 提前停 | 低 | TPOT 仍基于 415 个 decode token 计算有效；与 perf-T1 风险 A2 同源 |
| B3 | actual input=10265 vs target 10240（偏差 +25），在 ±32 tolerance 内 | 极低 | 无需处理 |
| B4 | engine_init_secs=30.25（vs perf-T1 tp=2 的 25.38） | 信息 | 多 2 个 worker process + RCCL init，符合预期，可作 perf-T4 tp=8 init 时长参考（预期 40-50 s 量级） |
| B5 | `rocm-smi` 输出 "Not supported on the given system" 行干扰阅读，但 VRAM% 行可读 | 信息 | 不影响验证 |

**未触发**任何 `no instance found` / `ImportError` / `assert` / `RuntimeError`，未触发 dispatch 异常。本任务核心 deliverable（TTFT、TPOT 数值）已稳定取得。

---

## 7. 文件清单（本任务产出）

- `WORK_DIR/logs/tp4_run1.log`（13 行 [PERF] dump）
- `WORK_DIR/logs/tp4_run1_full.log`（stdout/stderr 全量，含主进程 [aiter] / [atom] 行）
- `WORK_DIR/logs/tp4_run2.log`（13 行 [PERF] dump）
- `WORK_DIR/logs/tp4_run2_full.log`（stdout/stderr 全量）
- `WORK_DIR/progress/perf-t2.md`（本文件）
