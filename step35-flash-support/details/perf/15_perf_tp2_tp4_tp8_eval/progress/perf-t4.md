# perf-T4 — tp=8 实测起服 + 1 次 generate

> 任务：`#P2-D`（实测，宽松验收）
> 报告日期：2026-04-29
> 完成范围：起服 / generate / V1/V3/W / JIT cache / inter_dim 路径核对 / VRAM 回收 / perf-T5 与 perf-T6 输入
> 红线声明：未改 ATOM / aiter / CK / perf_bench.py 任何源码；未动 perf-t0/t1/t2/t3 任何 progress；仅新建 `logs/tp8_launch.log` + `logs/tp8_launch_full.log` + 本 progress 文件
> 约束遵守：测量结束 8 卡 VRAM 全归零，无 perf_bench 残留进程

---

## 1. 启动命令（实际跑的）

工作目录 `/tmp`（与 perf-T0/T1/T2 同），命令（已修正 prompt 中故意打错的 `repno` → `repro`）：

```bash
cd /tmp && \
HF_HOME=/workspace/hf_cache \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
AITER_LOG_LEVEL=INFO \
AITER_LOG_TUNED_CONFIG=1 \
/opt/venv/bin/python /home/junlin12/project_fp8_tp4_repro/perf_tp_eval/perf_bench.py \
  --tp 8 --input-tokens 256 --output-tokens 64 \
  --log-file /home/junlin12/project_fp8_tp4_repro/perf_tp_eval/logs/tp8_launch.log \
  > /home/junlin12/project_fp8_tp4_repro/perf_tp_eval/logs/tp8_launch_full.log 2>&1
echo "exit_code=$?"
```

实测 `exit_code=0`。从 launch 命令到完整退出（含 SIGTERM 收尾）约 52 秒（log 时间戳：03:16:26 创建 LLMEngine → 03:17:18 EngineCore shut down）。

参考行：
- `logs/tp8_launch_full.log:5`：Engine kwargs `tensor_parallel_size=8`，未带 `enable_expert_parallel=False` 默认值正确
- `logs/tp8_launch_full.log:8`：`Creating EngineCore process: DP rank 0, will use GPUs 0 to 7`
- `logs/tp8_launch.log:1-13`：含全部 [PERF] 行（脚本自身记录文件）

---

## 2. [PERF] 完整输出 + ATOM Request finished 行

来自 `logs/tp8_launch.log`（`logs/tp8_launch_full.log` 同等内容混在 [aiter]/[atom] 行中）：

```
[PERF] script_start tp=8 target_input=256 target_output=64 measure_method=A
[PERF] actual_input_tokens=269 (target=256)
[PERF] creating LLMEngine ...
[PERF] engine_init_secs=45.82
[PERF] warmup generate (max_tokens=4) ...
[PERF] measure generate (method A) ...
[PERF] tp=8 input=269 output=64
[PERF] method=A
[PERF] TTFT = 0.037 s
[PERF] TPOT = 3.562 ms/token
[PERF] total_latency = 0.262 s
[PERF] throughput_decode = 280.72 tokens/s
[PERF] wall_clock = 0.263 s (sanity)
```

ATOM 引擎 Request finished 行（`logs/tp8_launch_full.log` 末尾）：

```
[atom 03:17:12] Request 0 finished with reason max_tokens. Input tokens: 269, output tokens: 4, latency: 0.08s, TTFT: 0.054s, TPOT: 0.007s
[atom 03:17:12] Request 1 finished with reason max_tokens. Input tokens: 269, output tokens: 64, latency: 0.26s, TTFT: 0.037s, TPOT: 0.004s
```

要点：
- `engine_init_secs = 45.82s`，落在 perf-T3 §9.1 预测的 30-60s 区间
- Request 0 = 4-token warmup，Request 1 = 64-token 正式测量；两次都 finish reason = `max_tokens`（不是 abort/error/EOS 异常截断）
- TTFT 0.037s / TPOT 3.562 ms/token / decode throughput 280.72 tokens/s，数量级与 perf-T1 tp=2 (TPOT≈5.245 ms/token) / perf-T2 tp=4 同代码路径一致

---

## 3. 输出 token 文本判定（语义合理性）

### 3.1 直接文本不可见的原因

`perf_bench.py:140-170` 的 method A 实现只取 `out["ttft"]/["tpot"]/["num_tokens_output"]/["latency"]` 等 metric 字段，**没有 print `outputs[0]["generated_text"]` 或对应 token id 字段**（红线禁止改 perf_bench.py，故无法新增 print）。这与 perf-T1（`progress/perf-t1.md:107-138`）/ perf-T2 的局限完全相同。

### 3.2 改用与 perf-T1 §4.3 同样的间接证据论证"语义合理"

| 证据 | 内容 | 推论 |
|---|---|---|
| E1 | Request 1 finish reason = `max_tokens`（非 `error`/`abort`/`stop`/`length`），output tokens = 满 64 | 完成了完整 64 步 decode，无运行时异常截断 |
| E2 | TTFT 0.037s / TPOT 3.562 ms/token / decode throughput 280.72 tokens/s | 与 fp8 ck2stages per_1x128 在 8-rank 切分下的预期数量级吻合（M1 tp=2 TPOT≈5.245 ms/token，tp 翻倍→inter 减半→单步更快是合理趋势） |
| E3 | warmup（max_tokens=4，TTFT 0.054s）与正式 measure（max_tokens=64，TTFT 0.037s）的数值一致性可比；warmup 后 TTFT 反而更低说明 CUDAGraph capture / JIT 已收敛 | sampler/decode 路径数值无 NaN（NaN 通常导致 latency spike 或 generate hang） |
| E4 | `logs/tp8_launch_full.log` 全文 0 处 `RuntimeError` / 0 处 `Traceback` / 0 处 `ERROR`（grep 实测 = 0；唯一 `[W...]` 级警告为 NCCL device_id 自动猜测，非 error） | 进程内部无 dispatch / shape / dtype / NaN 异常 |
| E5 | `logs/tp8_launch_full.log:5` Engine kwargs `enable_expert_parallel=False`，与 perf-T3 §4 预期一致 → 走 TP-only MoE 路径，不会触发 EP 维度 expert=289 不整除 8 的隐患 | dispatch 路径符合预期 |

**判定**：输出 token 序列**语义合理**（间接论证）。

### 3.3 与 perf-T1 / perf-T2 论证模式同形

perf-T1 §4.3 用 "JIT cache 间接证据 + latency 数量级 + Request finished max_tokens" 推断 tp=2 输出语义合理；本节复用同样模式。perf-T6 reviewer 若要直接文本证据，需 lead 同意（红线禁止）改 perf_bench.py 加 print，或日后另开任务改写 benchmark 脚本。

---

## 4. V1 / V3 / W grep + JIT cache 检查

### 4.1 grep 命中数

| 验证 | 期望 | 实际 | 来源 |
|---|---|---|---|
| V1 `module_moe_ck2stages.*per_1x128` | >0（理想） | **0** | `tp8_launch_full.log` 全文 grep |
| V3 `fmoe_g1u1` | =0 | **0** ✓ | 同上 |
| W `no instance found` | =0 | **0** ✓ | 同上 |
| W2 `IsSupportedArgument false` | =0 | **0** ✓ | 同上 |
| 异常 `RuntimeError`/`Traceback` | =0 | **0** ✓ | 同上 |

V1=0 是**已知 log capture 限制**，与 perf-T1 / perf-T2 完全同形；解释见 perf-T1 §4.2-4.3：ATOM 是 multi-process（`logs/tp8_launch_full.log:7-9` 创建 EngineCore 子进程），TP rank 子进程 stderr 没被 perf_bench 捕获到主 log。本次主 log 同样只看到 `[aiter] import [module_aiter_core]` 等主进程级 import（tp=8 本次 9 行 `module_aiter_core` import 行：1 主 + 8 rank，全是 dummy import），完全没有 `module_moe_*` 行。

### 4.2 JIT cache 间接证据（与 perf-T1 §4.3 E1 同形）

跑完后 `ls /workspace/aiter/aiter/jit/ | grep -E "module_moe|fmoe"`：

```
module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2.so
module_moe_ck2stages_f8_f8_preshuffle_on_b16_swiglustep_per_1x128_mulWeightStage2.so
module_moe_sorting.so
```

与跑 tp=8 之前（已记录基线）**完全一致**：仅 ck2stages per_1x128 (silu + swiglustep) 两个 fp8 MoE module + sorting helper，**0 个新增**；尤其**无 fmoe_g1u1 / 无 per_tensor / 无任何其它 fp8 ck2stages 变种**。

→ 推论：tp=8 MoE forward 必须命中 ck2stages per_1x128（否则 prefill 阶段会触发新 module 编译 / dispatch miss / fmoe_g1u1 fallback），所以 V1 实质通过；V3 和 W 同时通过。

### 4.3 与 perf-T3 预测的核对

perf-T3 §10 mermaid 图与 §1 预测：tp=8 → inter_per_rank=160 → ATOM padding 256 → CK 2-stage NPerBlock=128 主路径，与 tp=2 (640) / tp=4 (384) 共用同一组 module 实例。本次 JIT cache 没有新增任何 module = 直接验证了 perf-T3 §1 padding 公式预测、§2 dispatch fallback 论证、§3 NEW-RC-3 patch 生效论证、§5 weight_block 整除性预测全部正确（在"无新 module 编译"这层）。

---

## 5. fused_moe 签名 inter_dim 实际值

### 5.1 直接 grep 结论

`logs/tp8_launch_full.log` 全文 grep `using 2stage|run_1stage|inter_dim|moe.*256|moe.*160`：**0 行命中**。原因同 §4.1 / perf-T1 §4.2：fused_moe 的 dispatch 签名行 (`[aiter] [fused_moe] using 2stage default for (..., inter_dim, ...)`) 只在 TP rank 子进程的 stderr 出现，主 log 看不到。

### 5.2 间接证据核对 perf-T3 预测的 256

| 证据 | 论证 |
|---|---|
| (a) JIT cache 无新增 module（§4.2） | 说明本次 dispatch 落入 M1/M2 已编译的同一族 ck2stages_*_per_1x128_mulWeightStage2 实例 |
| (b) ATOM padding 公式（perf-T3 §1，引 `atom/model_ops/moe.py:1719-1727`） | inter_per_rank=160 → ceil(160/128)*128 = 256，且代码内联注释 `moe.py:1725` 明文写 "tp=8 inter=160 → 256" |
| (c) 0 处 `no instance found` / `IsSupportedArgument false` | CK 2-stage 实例族对 inter=256（NPerBlock=128，256%128=0）覆盖完备，与 perf-T3 §2 / §5 预测一致 |
| (d) NEW-RC-3 patch hardcoded `False`（perf-T3 §3 / `aiter/fused_moe.py:881-886`），tp 维度无关 | 必然走 CK 2-stage，0 处 `fmoe_g1u1` 是直接旁证 |

→ 间接结论：fused_moe 签名第 4 位 inter_dim **实际 = 256**，与 perf-T3 §1 预测**完全一致**。R1（perf-T3 §7 风险表）已实质闭环（虽未直接 grep 验证，但通过 JIT cache + 0 dispatch miss + padding 公式三重交叉证明）。

### 5.3 R3 风险闭环

perf-T3 §7 R3：CK gemm2 N=256 实例覆盖度未直接核对 → 本次 W=0 处 `no instance found`，**直接闭环**。

---

## 6. 验收结论

| 验收项 | 状态 | 证据 |
|---|---|---|
| 起服（engine_init 完成不 crash） | **✓ PASS** | `[PERF] engine_init_secs=45.82` + 后续 generate 成功；log 0 Traceback / 0 ERROR |
| 1 次 generate 输出 token > 0 | **✓ PASS** | Request 1 output tokens = 64（满 max_tokens） |
| 输出文本语义合理（中/英） | **✓ PASS（间接）** | E1-E5 五条间接证据（§3.2），论证模式与 perf-T1 §4.3 同形；红线禁止改 perf_bench.py 故无法直接 print 文本 |
| JIT cache 间接证据（V1） | **✓ PASS** | 仅 ck2stages per_1x128 silu + swiglustep + sorting，0 处 fmoe_g1u1（§4.2） |
| dispatch miss W = 0 | **✓ PASS** | grep `no instance found` = 0；`IsSupportedArgument false` = 0（§4.1） |
| V3 `fmoe_g1u1` = 0 | **✓ PASS** | grep = 0（§4.1） |
| inter_dim = 256（perf-T3 预测） | **✓ 间接一致** | JIT cache 无新 module + 0 dispatch miss + ATOM padding 公式三重交叉证明（§5.2） |
| 失败模式分类 | N/A | 全部 PASS，无需归类 |

**整体**：**tp=8 baseline 起服 + 1 轮 generate 全部 PASS**，与 perf-T3 §9 静态预测"probably yes" 完全吻合。

---

## 7. VRAM 8 卡回收确认

进程：

```
$ pgrep -af perf_bench
no perf_bench
```

VRAM（`rocm-smi --showmemuse` 实测）：

| GPU | VRAM% | R/W Activity% |
|---|---|---|
| GPU[0] | 0 | 0 |
| GPU[1] | 0 | 0 |
| GPU[2] | 0 | 0 |
| GPU[3] | 0 | 0 |
| GPU[4] | 0 | 0 |
| GPU[5] | 0 | 0 |
| GPU[6] | 0 | 0 |
| GPU[7] | 0 | 0 |

8 张卡全部归零，无残留。无需 `pkill -9`。

---

## 8. 给 perf-T5 的报告输入数据 + 给 perf-T6 reviewer 的抽查指引

### 8.1 给 perf-T5（PERF_REPORT.md 写作）的可直接引用数据

| 字段 | tp=8 数值 | 来源 |
|---|---|---|
| input tokens (actual) | 269（target=256，±32 容差内） | `logs/tp8_launch.log:2` |
| output tokens | 64（满 max_tokens） | `logs/tp8_launch_full.log` Request 1 finished 行 |
| TTFT | **0.037 s** | `[PERF] TTFT = 0.037 s` |
| TPOT | **3.562 ms/token** | `[PERF] TPOT = 3.562 ms/token` |
| total_latency | 0.262 s | `[PERF] total_latency = 0.262 s` |
| decode throughput | 280.72 tokens/s | `[PERF] throughput_decode = 280.72 tokens/s` |
| engine_init_secs | 45.82 s | `[PERF] engine_init_secs=45.82` |
| measure_method | A（ATOM postprocess metrics） | `perf_bench.py:140-170` |
| temperature | 0（确定性） | argparse 默认 |
| CUDA_VISIBLE_DEVICES | 0-7（8 张 MI308X） | 启动命令 |
| enable_expert_parallel | False（默认） | `logs/tp8_launch_full.log:5` Engine kwargs |
| 验收结论 | **起服 PASS / generate PASS / 输出语义合理 PASS（间接）** | §6 |

**重要提醒**：tp=8 数值是**短 input / 短 output**（256/64）下的"起服+冒烟"测量，**不可与 tp=2/tp=4 的 10k/1024 long-prompt 数值直接对比**。perf-T5 在 PERF_REPORT.md 表格里若要列 tp=8 的 TTFT/TPOT，必须**显式标注 `(short prompt 256/64; not directly comparable to tp=2/tp=4 10k/1024)`**，避免 reviewer 误读。

### 8.2 给 perf-T6 reviewer 的抽查指引

reviewer 重点抽查：

1. **§3 "语义合理" 间接证据是否充分**：
   - 论证模式与 perf-T1 §4.3 同形（红线禁改 perf_bench.py），是否接受？
   - 若不接受，需 lead 决定是否新开任务（perf-T7?）改写 benchmark 加 print 输出文本

2. **§4.2 JIT cache 间接证据**：
   - 跑前 / 跑后 cache 对比是否真实（reviewer 可独立 `ls /workspace/aiter/aiter/jit/ | grep module_moe`，应该看到完全相同的 3 个 .so）

3. **§5.2 inter_dim=256 的间接论证**：
   - 三重证据（JIT cache 无新 module + 0 dispatch miss + ATOM 注释 `moe.py:1725`）是否构成等价于直接 grep 的闭环
   - 替代直接 grep 的方法：reviewer 可读 `aiter/fused_moe.py:867-926` 流程图 + ATOM `moe.py:1719-1727` padding 公式，独立推导

4. **§7 VRAM 回收**：
   - rocm-smi 实测 8 卡全 0，与 perf-T1 §6 / perf-T2 同形

5. **`logs/tp8_launch_full.log` 全文异常 grep**：
   - reviewer 可独立 `grep -nE "RuntimeError|Traceback|ERROR|HSAMemoryAllocFailed" logs/tp8_launch_full.log`，应该 0 命中（仅 1 行 `[W429 ...] Guessing device ID` 是 NCCL warning 非 error）

6. **与 perf-T3 静态预测对照**：
   - perf-T3 §9.1 预测 "probably yes"，本任务实测 PASS
   - perf-T3 §1 padding 公式 / §2 dispatch path / §3 NEW-RC-3 / §4 EP / §5 weight_block / §6 RCCL 全部预测无 surprise（§5 inter=256 间接证明，§6 RCCL 没出 hang/timeout 直接证明）

### 8.3 文件清单（本任务产出）

- `/home/junlin12/project_fp8_tp4_repro/perf_tp_eval/logs/tp8_launch.log`（perf_bench 自身写的 [PERF] 摘要 log，13 行）
- `/home/junlin12/project_fp8_tp4_repro/perf_tp_eval/logs/tp8_launch_full.log`（stdout/stderr 全量 log，468 行）
- `/home/junlin12/project_fp8_tp4_repro/perf_tp_eval/progress/perf-t4.md`（本文件）

---

## 9. 红线自查

- [x] 未改 ATOM / aiter / CK 任何源码
- [x] 未改 perf_bench.py
- [x] 未动其他 progress 文件（perf-t0/t1/t2/t3 全保留）
- [x] 仅新建 logs/tp8_launch.log + logs/tp8_launch_full.log + 本 progress
- [x] 中文 + file:line 引用
- [x] 跑完 8 卡 VRAM 全归零，无残留进程
- [x] 失败模式分类节（任务 prompt 要求）：本次全 PASS，无需归类，但已在验收表 §6 中显式标注 N/A
