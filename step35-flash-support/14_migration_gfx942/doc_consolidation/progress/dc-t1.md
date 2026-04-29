# DC-T1 主文档调研

> 任务范围：通读 5 份主文档（SESSION_HANDOFF / FINAL_REPORT / PROJECT_SUMMARY / TEAM_CONFIG / M1_BASELINE_DISPATCH_PLAN），提取与「Step-3.5-Flash-FP8 从 gfx950 迁移到 gfx942(MI308X)」迁移主题相关的权威结论 + 证据 file:line。
>
> 红线：所有引用经实际 Read 确认；KNOWN_FACTS 已有内容不复述。

---

## §1 总体迁移结论（What & Why）

### 1.1 起点状态（gfx950 上的状况）

- 参考指南是为 gfx950（MI350X，8 张卡）写的；本机 gfx942（MI308X，40 张卡）→ `TEAM_CONFIG.md:30-37`（差异表）
- gfx950 路径里几个关键 ASM kernel：
  - `bf16gemm_bf16_tn_256x256` — gfx950 独有，gfx942 dispatcher 不路由 → `PROJECT_SUMMARY.md:68`
  - ASM `fmoe_fp8_blockscale_g1u1` — gfx950 独有；gfx942 上 `aiter.fmoe_g1u1` 签名不带 block shape 参数 → `PROJECT_SUMMARY.md:69`
- gfx950 上"全 BOS"是 dispatcher 路由到该 ASM 的现象，gfx942 不会重现 → `TEAM_CONFIG.md:32`、`TEAM_CONFIG.md:178`

### 1.2 目标状态（gfx942 PASS 标准）

- M1（首要）：tp=2 配置 4/4 prompt 输出连贯（无 BOS / 无乱码 / 无 crash） → `TEAM_CONFIG.md:21`
- M2（M1 通过后）：tp=4 配置同标准 → `TEAM_CONFIG.md:22`
- 不要求性能数值；不跑 BF16 模型 → `TEAM_CONFIG.md:23-24`、`TEAM_CONFIG.md:177`
- 完整 PASS 准则：4/4 prompt 连贯 + 无全 BOS + 无乱码 + 无 ValueError + 无 crash + 无 hang → `TEAM_CONFIG.md:132-149`

### 1.3 整体策略

- 串行升级：先 tp=2 把环境/编译/fix 跑通，再扩 tp=4 → `TEAM_CONFIG.md:25`
- 三仓 commit 锁定：ATOM `acff926` / aiter `0f8164017` / CK `defd7ad29` → `PROJECT_SUMMARY.md:103-107`、`SESSION_HANDOFF.md:225-229`
- 仅 1 处源码 dirty：`aiter/fused_moe.py`（NEW-RC-3 patch） → `PROJECT_SUMMARY.md:109`、`SESSION_HANDOFF.md:231`

### 1.4 项目结果

- M1 PASS（2026-04-28，T-5） → `FINAL_REPORT.md:15-31`
- M2 PASS（2026-04-28，T-12） → `FINAL_REPORT.md:33-50`
- PROJECT CLOSED 标记 → `SESSION_HANDOFF.md:3-4`、`SESSION_HANDOFF.md:347`

---

## §2 三大 root cause 权威表述

### NEW-RC-1（FP8 dtype：e4m3fn → e4m3fnuz）

- **症状**：safetensors 是 OCP `e4m3fn`（bias=7、NaN=0x7f/0xff），gfx942 mfma 要 `e4m3fnuz`（bias=8、NaN=0x80）；若不转换会差 factor 2 + NaN 解释错位 → `PROJECT_SUMMARY.md:222`、`SESSION_HANDOFF.md:239`
- **root cause 表述（直引）**：
  > "safetensors 是 e4m3fn，gfx942 要 fnuz" — `SESSION_HANDOFF.md:239`
  > "gfx942 mfma 要 fnuz" — `PROJECT_SUMMARY.md:222`
- **修复点**（4 段链路）：
  1. aiter `aiter/utility/dtypes.py:10-25` 静态把 `d_dtypes["fp8"]` 锁到 arch 对应类型（gfx942 → `torch.float8_e4m3fnuz`） → `PROJECT_SUMMARY.md:190`
  2. ATOM `quant_spec.py:198,215,268-271` regex 回退路径返回 fnuz dtype → `PROJECT_SUMMARY.md:191`
  3. ATOM `model_ops/moe.py:1531,1537-1539` 据此置 `need_normalize_e4m3fn_to_e4m3fnuz=True` → `PROJECT_SUMMARY.md:191`
  4. ATOM `model_ops/utils.py:61-82` 把 0x80（fn 的 +0、fnuz 的 NaN）改写为 0；`weight_scale = weight_scale * 2.0` → `PROJECT_SUMMARY.md:192`
- **闭环证据**：
  - M2 baseline log 中 fused_moe 调用签名出现 `'torch.float8_e4m3fnuz'` ×40 处 → `PROJECT_SUMMARY.md:222`、`FINAL_REPORT.md:47`
  - T-11 静态 trace 反证强度 strong → `PROJECT_SUMMARY.md:222`

### NEW-RC-2（weight_scale * 2.0 方向）

- **症状**：`weight_scale_inv` 字段名误导（看起来像 inverse scale），若 utils.py 方向反 4× 偏离 → 输出乱码 → `PROJECT_SUMMARY.md:223`
- **root cause 表述（直引）**：
  > "safetensors 字段是 `weight_scale_inv`（inverse scale = 1/forward），但 `atom/model_ops/utils.py:79` 做 `weight_scale = weight_scale * 2.0`（假设 forward）" — `PROJECT_SUMMARY.md:223`
  >
  > 实际命名沿袭 DeepSeek-V3 历史（"inv" = 1/amax 缩放因子，不是 dequant 时取倒数）→ `FINAL_REPORT.md:84`
- **修复点**：保持 `atom/model_ops/utils.py:79` 的 `weight_scale * 2.0`（方向正确，无需改）；`atom/model_loader/loader.py:320-321` 仅做字符串 rename `weight_scale_inv → weight_scale`，不取 1/x → `FINAL_REPORT.md:87-88`
- **闭环证据**：
  - T-6 静态 fp32 dequant：layer 3 expert 0 第一个 (128,128) block，forward 解释 absmax=0.093（正常 bf16 量级）vs inverse 解释 absmax=2.15e6（爆炸 7 个数量级） → `FINAL_REPORT.md:91-95`、`PROJECT_SUMMARY.md:223`
  - 下游 `aiter/aiter/ops/triton/moe/quant_moe.py:238` 的 `w = w * scales` 确认 forward 语义 → `FINAL_REPORT.md:89`
  - M1 + M2 PASS 反证 → `PROJECT_SUMMARY.md:223`

### NEW-RC-3（per_1x128 ASM bypass）

- **症状**：per_1x128 prefill 在未 patch 时被 dispatch 到 ASM `aiter.fmoe_g1u1`，该 kernel 签名 `(fc1_scale, fc2_scale)` 不带 block shape 参数，gfx942 上数值会错 → `PROJECT_SUMMARY.md:198`、`FINAL_REPORT.md:64`
- **触发矩阵**：tp=2 / tp=8 等"对齐良好"的 inter_dim 反而中招；tp=4 因 padded inter=384 不被 256 整除反而安全 → `PROJECT_SUMMARY.md:198`
- **root cause 表述（直引）**：
  > "per_1x128 prefill 走 ASM `fmoe_g1u1`，签名不带 block shape" — `PROJECT_SUMMARY.md:224`
- **修复点**：`/workspace/aiter/aiter/fused_moe.py:881-886`，把原 heuristic `run_1stage = token > 32 and (inter_dim % 256 == 0)` 强制改为 `run_1stage = False` → `FINAL_REPORT.md:62-64`、`PROJECT_SUMMARY.md:115-138`、`SESSION_HANDOFF.md:151-167`
- **闭环证据**：
  - M1 实测 0 处 `fmoe_g1u1`，全部走 `module_moe_ck2stages_f8_..._per_1x128_*Stage2` → `FINAL_REPORT.md:28`、`PROJECT_SUMMARY.md:224`
  - M2 实测同样 0 处 `fmoe_g1u1`，8 行命中 → `FINAL_REPORT.md:46`、`PROJECT_SUMMARY.md:224`
  - 旁路风险审计（仍存的 `fused_moe.py:932`）：tuned_fmoe.csv 348 条 per_1x128 行 `run_1stage` 列全为 0，patch 100% 生效 → `PROJECT_SUMMARY.md:138`、`SESSION_HANDOFF.md:171-174`、`FINAL_REPORT.md:125`

---

## §3 M2 tp=4 padding（inter_dim 320→384）

### 3.1 为什么 tp=4 会触发 padding

- tp=4 时 inter_dim=1280/4=**320** → `FINAL_REPORT.md:72`
- T-7 静态预测：M2 tp=4 应触发 NPerBlock=64 stage2 路径，而 `a8w8_gemm2_blockscale_kernels_list` 缺该 instance（因 fp8 mfma KPack=32 限制 → KPerBlock 必须 ≥128，与 K=320 不整除冲突） → `FINAL_REPORT.md:74`
- 不修，则会 dispatch fail；修复路径：ATOM 入口处 zero-pad 把 320 推到 384

### 3.2 修复在哪

- ATOM `_process_block_quant`：`/home/junlin12/ATOM/atom/model_ops/moe.py:1709-1746`，对 per_1x128 路径无条件 `align = block_n = 128` 并 `inter_pad = ceil(inter_dim/128)*128` → `FINAL_REPORT.md:72`
- 开发者注释（`moe.py:1715-1727`）已写明动机：
  > "Bug fix: previously used align=64 for inter<=192 (copied from BF16 path), but 192%128=64!=0 → stage2 kernel dispatch fails. Correct: always align to block_n. tp=8 inter=160 → 256; tp=4 inter=320 → 384." — `FINAL_REPORT.md:76`
- T-10 发现这是不改源码自动规避方案 → `FINAL_REPORT.md:112`

### 3.3 验证证据

- T-12 V4 决定性证据：M2 baseline 中 fused_moe 调用签名第 4 位参数（inter_dim） = **384**（不是 320） → `FINAL_REPORT.md:49`、`FINAL_REPORT.md:78`
- T-13 padding strong 闭合 + KPack 跨架构推论 medium-偏-strong → `FINAL_REPORT.md:80`、`FINAL_REPORT.md:115`
- V5（stage2 走 NPerBlock=128 主路径）：K=384%128=0，无 IsSupportedArgument throw → `FINAL_REPORT.md:50`
- 注：T-14 F-3 review 修正 V4 措辞 = "实测 inter_dim=384 是直接事实 + padding 来源是 T-13 §1 strong 推论" = 整体间接 strong（不是直接 strong），因为 fused_moe 本身不打 `_process_block_quant called` 日志 → `PROJECT_SUMMARY.md:252`

---

## §4 dispatch 路径（gfx950 vs gfx942）

| 维度 | gfx950 路径 | gfx942 路径（本项目实测） | 来源 |
|---|---|---|---|
| FP8 numeric format | `e4m3fn`（OCP）| **`e4m3fnuz`**（NaN=0x80）— aiter dtypes.py 静态映射 + ATOM normalize | `PROJECT_SUMMARY.md:66`、`FINAL_REPORT.md:219` |
| BF16 GEMM 256x256 | ASM `bf16gemm_bf16_tn_256x256` | **不路由**（dispatcher fallback `default_config["libtype"] = "hipblaslt"` at `aiter/tuned_gemm.py:162`）| `PROJECT_SUMMARY.md:67-68`、`PROJECT_SUMMARY.md:152` |
| FP8 MoE per_1x128 prefill | ASM `fmoe_fp8_blockscale_g1u1`（带 block shape） | **CK 2-stage blockscale**（`module_moe_ck2stages_f8_f8_preshuffle_on_b16_{silu\|swiglustep}_per_1x128_mulWeightStage2`），由 NEW-RC-3 patch 强制 | `FINAL_REPORT.md:62-66`、`PROJECT_SUMMARY.md:69` |
| 实测路径优于预期 | — | gfx942 上 per_1x128 既不走 ASM 也不走 hipblaslt fp8 fallback；T-9 F-4 + T-11 已澄清这是合理结果，不是 V3 验收失败 | `FINAL_REPORT.md:66`、`SESSION_HANDOFF.md:204` |
| stage2 NPerBlock 选择 | — | 因 ATOM padding inter→384，K=384%128=0，走 NPerBlock=128 主路径，不触 NPerBlock=64 stage2 缺失 | `FINAL_REPORT.md:74`、`FINAL_REPORT.md:80` |
| block_m heuristic | — | 强制 2-stage 后 block_m=64/16，M1 inter=640 + M2 inter=384 实测 0 dispatch miss | `SESSION_HANDOFF.md:242`、`PROJECT_SUMMARY.md:225` |
| CK SwigluStep | — | gfx942 编译路径有效（无 arch guard） | `PROJECT_SUMMARY.md:158-166` |

---

## §5 三仓改动 summary

### 5.1 ATOM（commit `acff926`，唯一 dirty 已闭环）

| 文件 | 变更 | 为什么 | 来源 |
|---|---|---|---|
| `atom/model_ops/moe.py:1709-1746`（`_process_block_quant`）| `align = block_n=128`，`inter_pad = ceil(inter_dim/128)*128` | M2 tp=4 inter_dim 320→384 padding，规避 NPerBlock=64 stage2 缺失 | `FINAL_REPORT.md:72-76`、`FINAL_REPORT.md:214` |
| `atom/model_ops/moe.py:2310-2312`（`_load_w13`）/ `:2352-2354`（`_load_w2`）| ceil 整除 `(x + tp_size - 1) // tp_size` | FP8 scale 加载，确保最后 partial scale block 被包括 | `PROJECT_SUMMARY.md:181-182` |
| `atom/model_ops/moe.py:1531,1537-1539` | `need_normalize_e4m3fn_to_e4m3fnuz=True` 由 quant_spec 回退路径触发 | NEW-RC-1 自动 normalize 链路 | `PROJECT_SUMMARY.md:191` |
| `atom/model_ops/utils.py:61-82`（含 `:79` `weight_scale * 2.0`） | 0x80→0 + scale ×2.0 补偿 | NEW-RC-1 + NEW-RC-2 数值补偿（方向正确） | `PROJECT_SUMMARY.md:192`、`FINAL_REPORT.md:88` |
| `atom/quant_spec.py:198,215,268-271` | regex 回退返回 `d_dtypes["fp8"]` | 触发 normalize 路径 | `PROJECT_SUMMARY.md:191` |
| `atom/model_loader/loader.py:320-321` | `weight_scale_inv → weight_scale` 仅 rename，不取 1/x | NEW-RC-2 命名陷阱处理（forward 语义） | `FINAL_REPORT.md:87` |

### 5.2 aiter（commit `0f8164017`，含 NEW-RC-3 patch — 唯一 dirty 文件）

| 文件 | 变更 | 为什么 | 来源 |
|---|---|---|---|
| `aiter/fused_moe.py:881-886` | `run_1stage = False`（强制覆盖原 heuristic） | NEW-RC-3：阻止 per_1x128 prefill 路由 ASM `fmoe_g1u1` | `SESSION_HANDOFF.md:151-167`、`PROJECT_SUMMARY.md:115-138`、`FINAL_REPORT.md:218` |
| `aiter/utility/dtypes.py:10-25` | gfx942 → `torch.float8_e4m3fnuz` 静态映射 | NEW-RC-1 dtype 锁定（import 时生效） | `PROJECT_SUMMARY.md:190`、`FINAL_REPORT.md:219` |
| `aiter/ops/triton/moe/quant_moe.py:238` | `w = w * scales`（仅引用，未改） | 确认 dequant 是 forward 语义（NEW-RC-2 旁证） | `FINAL_REPORT.md:220` |
| `aiter/configs/tuned_fmoe.csv` | per_1x128 行 348 条 `run_1stage` 列全为 0（仅检查未改） | 确认 NEW-RC-3 patch 不被 csv 旁路 | `PROJECT_SUMMARY.md:138`、`FINAL_REPORT.md:222` |
| `aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv` | 已删 `bf16gemm_bf16_tn_256x256`（commit a2883ab37） | gfx950 ASM 移除，gfx942 dispatcher 不路由 | `PROJECT_SUMMARY.md:152`、`PROJECT_SUMMARY.md:298` |

### 5.3 CK（commit `defd7ad29`，submodule 分支 `swiglustep_and_mul`）

| 文件 | 变更 | 为什么 | 来源 |
|---|---|---|---|
| `include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm.hpp` | 4 段 `swiglustep_and_mul` branch 插入（hardcoded 7.0f clamp） | gfx942 走 CK 2-stage SwigluStep 路径需要的 activation 实现；无 arch guard，gfx942 编译有效 | `PROJECT_SUMMARY.md:107`、`PROJECT_SUMMARY.md:162`、`FINAL_REPORT.md:224` |
| `include/ck/tensor_operation/gpu/device/impl/device_moe_gemm_blockscale.hpp:425-473` | `IsSupportedArgument` 检查（仅引用，未改） | M2 实测 0 throw（V5 闭环） | `FINAL_REPORT.md:225` |

---

## §6 PASS 验证证据链

### 6.1 M1 PASS 证据（V1/V2/V3 + log 数值）

| 项 | 值 | 来源 |
|---|---|---|
| 判定 | **PASS** | `FINAL_REPORT.md:19`（指向 `progress/teammate-5.md:189-191`） |
| 4/4 prompt 输出连贯 | ✓（中英文均 OK，无乱码 / 无 BPE 异常 / 无 token repetition） | `FINAL_REPORT.md:20` |
| 原始 log | `docs/baseline_tp2_result.md` | `FINAL_REPORT.md:21` |
| TTFT / TPOT | 36.32s（含 JIT）/ 0.018 s/token | `FINAL_REPORT.md:23-24` |
| weight load / warmup / cudagraph | 52s / 301s / 188.9s | `FINAL_REPORT.md:25-27` |
| **V1**：NEW-RC-3 patch 生效（per_1x128 走 CK 2-stage） | ≥10 行命中 + 0 处 `fmoe_g1u1` | `FINAL_REPORT.md:28` |
| **V2**：fnuz 转换发生 | fused_moe `q_dtype=torch.float8_e4m3fnuz`，T-11 static trace strong | `FINAL_REPORT.md:29` |
| **V3**：dispatch 路径命中 0 miss | 130 行 dispatch 日志 + 0 `no instance found` | `FINAL_REPORT.md:30` |

### 6.2 M2 PASS 证据（V1-V5 + byte-identical 143/143）

| 项 | 值 | 来源 |
|---|---|---|
| 判定 | **PASS** | `FINAL_REPORT.md:36`（指向 `progress/teammate-12.md:217-219`） |
| 4/4 prompt 输出连贯 | ✓ | `FINAL_REPORT.md:37` |
| 原始 log | `docs/baseline_tp4_result.md` | `FINAL_REPORT.md:38` |
| JIT 增量编译 | 0s（全部复用 M1 cache）；TTFT −99% | `FINAL_REPORT.md:40,45` |
| **V1**：NEW-RC-3 patch 生效 | 8 行命中，0 `fmoe_g1u1` | `FINAL_REPORT.md:46` |
| **V2**：fnuz 转换 | `fnuz` 40 matches | `FINAL_REPORT.md:47` |
| **V3**：dispatch 命中 0 miss | 40+ hit / 0 miss | `FINAL_REPORT.md:48` |
| **V4**：ATOM padding 触发 inter_dim 320→384 | dispatch 签名第 4 位参数 = 384（不是 320） | `FINAL_REPORT.md:49` |
| **V5**：stage2 走 NPerBlock=128 主路径 | K=384%128=0，无 IsSupportedArgument throw | `FINAL_REPORT.md:50` |
| **byte-identical 闭环（T-16）** | Prompt #3 (1+2+3=?) M1↔M2 byte-for-byte 完全一致 143/143 chars；25 primes 完全正确；中文增肌 10kg 论点完全一致 | `FINAL_REPORT.md:163-168`、`SESSION_HANDOFF.md:347` |
| F-1 论据强度升级 | "数值漂移变 bug" 假设正式证否，PASS 论据从端到端定性 → 量化对照 + correctness verified | `FINAL_REPORT.md:170` |

---

## §7 主文档之间的不一致 / 互补关系

> 重要：以下记录差异，让 writer 选权威版本。

| 主题 | 文档 A 说法 | 文档 B 说法 | 取舍建议 |
|---|---|---|---|
| **dispatch V3 期望** | `SESSION_HANDOFF.md:204` 老版"应观察 hipblaslt fp8 fallback"；后用括号修正 "M1 实测路径优于预期，直接走 CK 2-stage 未触 hipblaslt" | `FINAL_REPORT.md:66` 明确"路径优于 SESSION_HANDOFF §5.3 原 V3 期望" | **以 FINAL_REPORT.md:66 为准**；SESSION_HANDOFF §5 是历史 wave 0 快照（顶部 `:15` 已声明"已过期，仅作历史参考"） |
| **NEW-RC-2 方向** | `SESSION_HANDOFF.md:240` 历史风险表把"weight_scale * 2.0 假设 forward"列为风险 | `FINAL_REPORT.md:82-95` + `PROJECT_SUMMARY.md:223` 已闭环：方向**正确**（forward 语义） | **以 FINAL_REPORT.md §2.3 为准**；SESSION_HANDOFF 同行尾部已注明"实测闭环 ✓ T-6 ratio=2.0 正确" |
| **V4 强度措辞** | `FINAL_REPORT.md:49` 标 "✓✓ 决定性证据" | `PROJECT_SUMMARY.md:252`（T-14 F-3）修正为"实测 inter_dim=384 是直接事实 + padding 来源是 T-13 §1 strong 推论 = 整体间接 strong（不是直接 strong）" | **以 PROJECT_SUMMARY §7.1 F-3 为准**；writer 在引用 V4 时不要写"决定性"，写"实测 + padding 推断" |
| **三仓 commit 描述位置** | `SESSION_HANDOFF.md:225-229` 仅短 hash | `PROJECT_SUMMARY.md:103-107` 含完整 hash + commit subject | **以 PROJECT_SUMMARY §4.3 为准**（信息更全） |
| **NEW-RC-3 patch diff** | `SESSION_HANDOFF.md:151-167` 只贴 patch 后内容 | `PROJECT_SUMMARY.md:117-132` 含完整 diff hunk（删除前 + 新增后） | **以 PROJECT_SUMMARY §4.4 为准**（diff 更直观） |
| **M1_BASELINE_DISPATCH_PLAN 时序** | 文档本身定位 "wave 1 派单计划（一次性）" → `M1_BASELINE_DISPATCH_PLAN.md:393` | 实际只覆盖 M1 阶段 + Wave 1/2 的 T-5..T-9 派单 | **历史价值文档**：writer 引用时取其设计原则（speculative execution / reviewer 必要性 / 4 维度审查），不要照抄 §3-§7 的具体 prompt 全文 |
| **TEAM_CONFIG.md 旧版与新版** | 项目根的 `TEAM_CONFIG.md` 是初始化时模板（含初始 TODO 列表 + ENVIRONMENT） | doc_consolidation 子任务的 `TEAM_CONFIG.md` 是新版 | 两者**不冲突**，是不同任务的 config；writer 引用迁移结论用根目录版 |

---

## §8 给 writer 的建议

### 8.1 可直接照搬（含原文位置）

| 内容 | 出处 | 备注 |
|---|---|---|
| §1 项目目标段（M1/M2 定义、PASS 标准） | `TEAM_CONFIG.md:21-26`、`TEAM_CONFIG.md:132-149` | 一字未改即可 |
| §2.1 NEW-RC-3 patch 完整 diff hunk | `PROJECT_SUMMARY.md:117-132` | diff 比 SESSION_HANDOFF 完整 |
| §2.2 ATOM padding 注释（开发者亲笔） | `FINAL_REPORT.md:76`（引自 ATOM `moe.py:1715-1727`） | "Bug fix: previously used align=64..." |
| §2.3 weight_scale_inv 命名陷阱解释 | `FINAL_REPORT.md:82-95` | 含 7 数量级数值对比，非常 self-explanatory |
| §3 软件栈表（commit + commit subject） | `PROJECT_SUMMARY.md:103-107` | 完整 hash + commit message |
| §4 PASS 数据表（M1 / M2） | `FINAL_REPORT.md:15-50` | 表格完整含 V1-V5 |
| §5 byte-identical 闭环数据 | `FINAL_REPORT.md:160-170` | T-16 量化对照表已写好 |

### 8.2 需要重新组织（说明原因）

| 内容 | 原因 | 重组建议 |
|---|---|---|
| **gfx950 → gfx942 整体迁移叙事** | 4 份主文档都没有"step-by-step 迁移流程"叙事，都是 closure 后的状态总结 | writer 需基于 §4 dispatch 路径对比表 + DC-T2 progress 时序，编一条"从初始 crash 假设 → Phase 1 调查（5+1 项）→ Phase 2 fix → M1 PASS → M2 padding 自动救急 → M2 PASS"主线 |
| **三大 RC 关系图** | 4 份主文档没有显式的 RC 间依赖图（独立 / 顺序 / 触发关系） | writer 需用 mermaid `graph LR`：NEW-RC-1（dtype）独立；NEW-RC-2（scale）独立但与 RC-1 同走 normalize 链；NEW-RC-3（ASM bypass）独立；M2 padding 是 NEW-RC-3 修复后的派生需求，不是新 RC |
| **dispatch 路径对比** | 散落在 SESSION_HANDOFF §7、FINAL_REPORT §2.1、PROJECT_SUMMARY §4.1 | 已在本文档 §4 整合为单表，writer 直接引用本文档 §4 即可 |
| **三仓改动 summary** | 散落在 PROJECT_SUMMARY §4.3/§4.4/§5、FINAL_REPORT References | 已在本文档 §5 整合，writer 直接引用 |
| **wave 时序** | 散落在 FINAL_REPORT §3 表 + PROJECT_SUMMARY §2 表 + SESSION_HANDOFF 末尾 wave close 行 | 让 DC-T2 在 progress mining 时整合时间线，writer 用 mermaid `gantt` 或 `timeline` |

### 8.3 dead end / 可省略

- `SESSION_HANDOFF.md` §1-§9（wave 0 时代 quick check 脚本、HF 下载 / cron 监控 / token 配置等）— 与迁移技术结论无关，writer 可全部省略 → 顶部 `:15` 自承"已过期"
- `M1_BASELINE_DISPATCH_PLAN.md` §3-§7 各 teammate prompt 全文 — 历史派单 artifact，writer 不需要引用具体 prompt 内容
- `TEAM_CONFIG.md` §初始 TODO List（`:184-235`）— 初始化模板，与最终结论无关
- `SESSION_HANDOFF.md` §11 文件索引 — writer 用自己的 References 节即可

---

## 收尾存档

### Tool calls 累计

- 7 次（包括 1 次 ls 验目录）
- Read：5 次（5 份主文档）
- Bash：1 次
- Write：1 次（本文档）

### 关键发现

1. **4 份主文档 PASS 判定完全一致**（M1 PASS T-5 + M2 PASS T-12），无矛盾
2. **NEW-RC-3 是唯一源码 patch**；NEW-RC-1 / NEW-RC-2 / M2 padding 都是利用已有 ATOM 自动 normalize/padding 链路（"不修代码"或"早已修过"）
3. **M2 tp=4 的 inter_dim 320→384 padding 是 ATOM 已踩坑修过的副作用**，不是本项目新引入的修复（开发者注释明确）
4. **V3 验收路径**比预期好：实测既不走 ASM 也不走 hipblaslt fp8 fallback，直接走 CK 2-stage（writer 不要照抄 SESSION_HANDOFF §5.3 的旧 V3 期望）
5. **T-14 F-3 修正 V4 措辞强度**（去掉"决定性"），writer 引用时务必同步
6. **T-16 byte-identical 143/143** 是项目最强证据，PASS 论据强度从定性 → 量化 + correctness

### 给 lead 的建议

- writer 阶段建议先看本文档 §7（不一致表）+ §8.1（可照搬清单），节省至少 1 wave 的 fact 核对时间
- §8.2 重组建议中的"gfx950→gfx942 step-by-step 叙事"需要 DC-T2 的时序结果合并；建议 writer 先读 DC-T2 的 progress mining 输出再动笔
- mermaid 图表种类（TEAM_CONFIG §TASK_SPECIFIC_VERIFICATION 强制清单 5 种）的素材已在本文档 §4（dispatch 对比）+ §5（三仓改动）就绪；NEW-RC 关系图需 writer 自己设计
- Reviewer（DC-T4）抽查 file:line 时建议从本文档 §2 / §3 / §6 的引用入手，命中率最高

