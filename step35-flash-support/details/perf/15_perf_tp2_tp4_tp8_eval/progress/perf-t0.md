# perf-T0 progress

> 任务 #000：写 TTFT/TPOT 测量脚本骨架 + 小 dry-run 验证
> WORK_DIR = `/home/junlin12/project_fp8_tp4_repro/perf_tp_eval`
> 日期：2026-04-29
> 红线遵守：未修改 ATOM/aiter/CK 任何源码（只新建 `perf_bench.py` + `progress/perf-t0.md` + `logs/dry_run_tp2*.log`）

---

## 1. 脚本设计

### 1.1 文件位置
`WORK_DIR/perf_bench.py`

### 1.2 采用方案 A（首选），保留 B 作为 fallback

**为什么选 A**：
ATOM `acff926` 已经在 `InputOutputProcessor.postprocess` 里直接计算 ttft/tpot 并塞进返回 dict（`atom/model_engine/llm_engine.py:236-262`）：

```python
# llm_engine.py:236-244
ttft = 0.0
tpot = 0.0
if req.first_token_time > 0:
    ttft = req.first_token_time - req.arrive_time
    if req.num_completion_tokens > 1:
        tpot = (req.leave_time - req.first_token_time) / (
            req.num_completion_tokens - 1
        )
# llm_engine.py:260-261
"ttft": ttft,        # seconds
"tpot": tpot,        # seconds per token
```

时间戳由：
- `seq.arrive_time = time.time()` 在 `preprocess` 设置（`llm_engine.py:217`）
- `seq.first_token_time` 字段定义在 `atom/model_engine/sequence.py:79`，由 scheduler/engine_core 在第一次 decode 后写入
- `seq.leave_time = time.time()` 在 `postprocess` 设置（`llm_engine.py:233`）

→ 比手动两次 generate（方案 B）更准确，且只跑一次推理，避免冷启动差异。

**方案 B fallback 在脚本里保留**（`--measure-method B`），万一 `first_token_time` 在某个 tp 配置下为 0（理论不会），perf-T1/T2 可立即切 B 重跑。

### 1.3 关键代码片段

```python
# perf_bench.py:36-50  build_long_prompt 用 chat template + 二分调整 token 数
def build_long_prompt(tokenizer, target_tokens, tolerance=32):
    seed = ("...中英混合种子...")
    seed_tokens = len(tokenizer.encode(seed, add_special_tokens=False))
    repeats = max(1, (target_tokens - 30) // max(1, seed_tokens))
    # _make_prompt + while-loop 直到 |actual - target| <= tolerance
```

```python
# perf_bench.py:144-167  方案 A：1 次 warmup + 1 次 measure，直接读 ATOM 输出
warm_sp = SamplingParams(temperature=args.temperature, max_tokens=4)
_ = llm.generate([chat_prompt], warm_sp)        # warmup（触发 CUDAGraph capture / JIT）

sp = SamplingParams(temperature=args.temperature, max_tokens=args.output_tokens)
t0 = time.perf_counter()
outputs = llm.generate([chat_prompt], sp)
wall = time.perf_counter() - t0
out = outputs[0]
ttft = float(out["ttft"])                        # 来自 ATOM
tpot_s = float(out["tpot"])                      # 来自 ATOM（秒/token）
total_lat = float(out["latency"])                # 来自 ATOM
```

### 1.4 入参（argparse）

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--tp` | int | required | tensor parallel size，会覆盖 `--tensor-parallel-size` |
| `--input-tokens` | int | 10240 | 目标 input prompt token 数 |
| `--output-tokens` | int | 1024 | max_new_tokens |
| `--model` | str | （由 ATOM `EngineArgs` 提供，默认改为 `stepfun-ai/Step-3.5-Flash-FP8`） | |
| `--log-file` | str | None | 把 [PERF] 行 dump 到该文件 |
| `--measure-method` | str | A | A=用 ATOM ttft/tpot 字段；B=两次 generate fallback |
| `--temperature` | float | 0.0 | |

**坑**：原任务描述写 `from atom import LLM`，实际 ATOM `__init__.py:10-14` 只 export `LLMEngine / SamplingParams / prepare_model`，没有 `LLM`。脚本走 `simple_inference.py` 的 `EngineArgs.from_cli_args(args).create_engine()` 路径（参考 `simple_inference.py:6-7,49-50`）。

**坑**：`--method` 与 ATOM `EngineArgs` 冲突，已改名为 `--measure-method`。

---

## 2. Dry-run 命令 + 实际输出

### 2.1 命令

```bash
cd /tmp && \
HF_HOME=/workspace/hf_cache \
CUDA_VISIBLE_DEVICES=0,1 \
AITER_LOG_LEVEL=INFO \
AITER_LOG_TUNED_CONFIG=1 \
/opt/venv/bin/python /home/junlin12/project_fp8_tp4_repro/perf_tp_eval/perf_bench.py \
  --tp 2 --input-tokens 256 --output-tokens 32 \
  --log-file /home/junlin12/project_fp8_tp4_repro/perf_tp_eval/logs/dry_run_tp2.log \
  > /home/junlin12/project_fp8_tp4_repro/perf_tp_eval/logs/dry_run_tp2_full.log 2>&1
```

### 2.2 结果：通过（exit_code=0）

第一次失败：`argparse.ArgumentError: argument --method: conflicting option string: --method`
→ 修：把 `--method` 改名为 `--measure-method`（见 perf_bench.py:103）
→ 第二次：通过。

### 2.3 关键输出（来自 `logs/dry_run_tp2_full.log`）

```
[PERF] script_start tp=2 target_input=256 target_output=32 measure_method=A
[PERF] actual_input_tokens=269 (target=256)
[PERF] engine_init_secs=105.55
[PERF] warmup generate (max_tokens=4) ...
[atom] Request 0 finished with reason max_tokens. Input tokens: 269, output tokens: 4, latency: 0.26s, TTFT: 0.240s, TPOT: 0.007s
[PERF] measure generate (method A) ...
[atom] Request 1 finished with reason max_tokens. Input tokens: 269, output tokens: 32, latency: 0.15s, TTFT: 0.038s, TPOT: 0.003s
[PERF] tp=2 input=269 output=32
[PERF] method=A
[PERF] TTFT = 0.038 s
[PERF] TPOT = 3.489 ms/token
[PERF] total_latency = 0.146 s
[PERF] throughput_decode = 286.58 tokens/s
[PERF] wall_clock = 0.147 s (sanity)
```

数值仅证脚本 ok，**绝不可作为正式 baseline**（input 才 256，output 才 32，prefill 没触发大尺寸 ck2stages 路径）。

---

## 3. V1/V2/V3 验证 path 的 dry-run 状态

| 验证 | 期望 | dry-run 实际 | 解释 |
|---|---|---|---|
| V1: `module_moe_ck2stages_..._per_1x128_*` 命中 | ≥1 | **0**（`logs/dry_run_tp2_full.log` 全文 grep） | dry-run 才 256 input + 4/32 output，MoE 路径未必打印 LOG_TUNED_CONFIG。**perf-T1/T2 在 10k input 跑里必须重检** |
| V2: `q_dtype=torch.float8_e4m3fnuz` | ≥1 | **0**（同上） | 同 V1，aiter LOG_TUNED_CONFIG 在 ck2stages 大 prefill 才打 |
| V3: `aiter.fmoe_g1u1` 0 出现 | =0 | **0** ✓ | 这个 negative check 在 dry-run 也满足 |

**给 perf-T1/T2 的强提醒**：跑 10k input 后**必须**在自己的 log 里 grep `module_moe_ck2stages.*per_1x128`、`float8_e4m3fnuz`、`fmoe_g1u1`，与 KNOWN_FACTS F2/F4 对齐，否则不能算 PASS。

---

## 4. VRAM / 进程 释放确认

```bash
# 跑完 ~3 秒后
$ rocm-smi --showmemuse | head -6
GPU[0]: GPU Memory Allocated (VRAM%): 0
GPU[1]: GPU Memory Allocated (VRAM%): 0
$ pgrep -af perf_bench
(无输出)
```

→ ✅ 干净。脚本 `try/finally` 中的 `llm.close()`（perf_bench.py:200-203）调用了 `core_mgr.close()`（`atom/model_engine/llm_engine.py:73-76`），engine cores 5 秒内全部退出。

---

## 5. 给 perf-T1 / perf-T2 / perf-T4 的使用说明

### 5.1 命令模板（替换 tp / CUDA_VISIBLE_DEVICES）

```bash
# 通用模板
cd /tmp && \
HF_HOME=/workspace/hf_cache \
CUDA_VISIBLE_DEVICES=<gpu_list> \
AITER_LOG_LEVEL=INFO \
AITER_LOG_TUNED_CONFIG=1 \
/opt/venv/bin/python /home/junlin12/project_fp8_tp4_repro/perf_tp_eval/perf_bench.py \
  --tp <N> --input-tokens 10240 --output-tokens 1024 \
  --log-file /home/junlin12/project_fp8_tp4_repro/perf_tp_eval/logs/<run_id>.log \
  2>&1 | tee /home/junlin12/project_fp8_tp4_repro/perf_tp_eval/logs/<run_id>_full.log
```

### 5.2 各 teammate 的具体值

| teammate | tp | CUDA_VISIBLE_DEVICES | run_id 建议 |
|---|---|---|---|
| perf-T1 | 2 | `0,1` | `tp2_run1`、`tp2_run2` |
| perf-T2 | 4 | `0,1,2,3` | `tp4_run1`、`tp4_run2` |
| perf-T4 | 8 | `0,1,2,3,4,5,6,7` | `tp8_baseline` |

**强制要求**：每个 tp 至少跑 2 次取后一次稳定值（`tp*_run2`），warmup 已在脚本内做 1 次（max_tokens=4），但 engine init 只算一次冷启动。

### 5.3 收尾步骤（每次跑完）

```bash
# 1. 等脚本自然退出
# 2. 验进程清干净
pgrep -af perf_bench || echo "ok no leftover"
# 3. 验 VRAM 释放
rocm-smi --showmemuse | head -6
# 4. 在 log 里抽查关键行
grep -E "module_moe_ck2stages.*per_1x128|float8_e4m3fnuz|fmoe_g1u1" logs/<run_id>_full.log | wc -l
grep "^\[PERF\]" logs/<run_id>.log
```

### 5.4 解读输出

脚本的 `[PERF]` 行就是要报给 lead 的最终数值，**TTFT 单位是秒**、**TPOT 单位是毫秒/token**。

---

## 6. 已知风险

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R1 | dry-run input 太小，V1/V2 path 没打印，方案 A 在真实 10k input 下是否仍然返回非 0 ttft/tpot 未实测 | 中 | perf-T1 跑 tp=2/10k 时**先看 [PERF] 行有没有数值**，若 ttft=0 立即切 `--measure-method B` 重跑 |
| R2 | engine_init 105 秒（首次有 JIT 编译 module_rmsnorm_quant 48.9s） | 低 | 后续 tp=4/8 切第一次仍可能慢，给 timeout 预算 600s+ |
| R3 | `actual_input_tokens=269` vs target=256 偏差 13（>tolerance 32 内） | 极低 | 大 input 时偏差会被稀释，仍在 ±10 token 范围内可忽略；脚本会打印 actual 值便于对账 |
| R4 | 方案 A 的 ttft 包含 `arrive → first_token_time`（含 prefill+第一个 decode），与 vLLM 定义一致；但若上游需要"纯 prefill 耗时"得另算 | 低 | 在 PERF_REPORT 里说明定义即可 |
| R5 | `--cudagraph-capture-sizes` 写死 `[1]`（perf_bench.py:117），如果 perf-T4 tp=8 启动失败可去掉或调成 `[1,2]` | 低 | perf-T4 失败时优先试 `--cudagraph-capture-sizes "[1,2,4,8]"` |
| R6 | dry-run 没跑 tp=4/tp=8，多卡 NCCL/RCCL init 行为未在本脚本验证 | 中 | perf-T1（tp=2）先跑通，再放 perf-T2（tp=4）；perf-T4 单独排队 |

---

## 7. 文件清单（本任务产出）

- `WORK_DIR/perf_bench.py`（211 行，新建）
- `WORK_DIR/progress/perf-t0.md`（本文件）
- `WORK_DIR/logs/dry_run_tp2.log`（13 行 [PERF] dump）
- `WORK_DIR/logs/dry_run_tp2_full.log`（157 行 stdout/stderr 全量）
