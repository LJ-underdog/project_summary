# Step-3.5-Flash 验证 Pipeline — 总体执行计划

**生成日期**：2026-04-25
**验证范围**：project_summary/step35-flash-support/01~07（V01-V07）
**编制方**：Master Reviewer
**输入文件**：V01_moe.md / V02_swiglu.md / V03_sliding_window.md / V04_tp_support.md / V05_fp8_inference.md / V06_fp8_tp4.md / V07_longseq_bos.md

---

## 0. 执行总览

### 0.1 验证依赖图

```
V01-MoE (Fix1: shuffle_weights gfx950, Fix2: V1->V3 block_m=128)
  ├─ V02-SwigluStep (依赖 V01：activation 走 fused_moe 同 dispatch 路径)
  ├─ V03-SlidingWindow (依赖 V01：端到端验证用 fused_moe；kernel 修复独立)
  └─ V04-TP Support (依赖 V01：tp=4 inter_dim padding 走 V01 fix 后的 V3 kernel)
        ├─ V05-FP8 Inference (依赖 V04：blockscale guard 与 V01 Fix2 互斥分支)
        │     └─ V06-FP8 tp=4 (依赖 V04 + V05：FP8 padding 与 scale shard 都依赖 TP 框架)
        └─ V07-LongSeq BOS (依赖 V04：tp=4 路径打通后才能跑 10k；与 V06 可并行)
```

**关键串行点**：V01 必须先于其他所有专题（任何 fused_moe 路径正确性破坏会污染所有 e2e 实验结果）。

### 0.2 时间估算

| 专题 | P0 实验数 | 预计工时 | 是否需 GPU |
|------|----------|---------|-----------|
| V01-MoE | 4 (实验 1/2/3/4) | 4-5h | 是（gfx950）|
| V02-SwigluStep | 4 (实验 1/2/3/5) | 5-6h | 是 |
| V03-SlidingWindow | 3 (实验 1/2/3) | 3-4h | 是 |
| V04-TP Support | 4 (C.2/实验 1/2/3) | 4-5h | 是 |
| V05-FP8 Inference | 3 (实验 1/2/3) | 3h | 是 |
| V06-FP8 tp=4 | 4 (Exp 1/2/4/5) | 4-5h | 是 |
| V07-LongSeq BOS | 3 (实验 1/2/3) + 5.a | 3-4h | 是 |
| **总计** | **25 P0** | **26-32h** | — |

### 0.3 并行执行建议

- **第一波（无依赖）**：V01 实验 4（静态 grep）+ V07 实验 5.a（CSV 扫描）
- **第二波（V01 内部并行）**：V01 实验 1 + 实验 2 同时跑
- **第三波（V01 通过后）**：V02/V03/V04 三个专题可并行启动各自的 op_test 实验
- **第四波（V04 通过后）**：V05 + V06 + V07 端到端可在不同 GPU 集合上并行
  - tp=2 实验可用 GPU 0,1
  - tp=4 实验可用 GPU 0,1,2,3（避开 GPU5）

---

## 1. 跨专题待确认问题

### 高优先（影响其他专题的结论）

#### 1.1 [V01 ↔ V04] Buffer padding revert 与 inter_dim padding 的关系

- **现象**：V01 §A 末尾把 commit 3771835ac 描述为「buffer padding 全部 revert」，并在 §E.3 标注 ATOM L489-518 inter_dim padding 仍然存在。V04 又把 inter_dim padding（commit 635e59e）作为 tp=4/tp=8 的核心修复来验证。
- **关键问题**：3771835ac revert 的「buffer padding」具体是 aiter 内 stage 中间 buffer 的 K/N 对齐 padding，与 ATOM 侧 `process_weights_after_loading` 中的 weight inter_dim padding 是两个独立的修改。**V01 §E.3 已明确指出此张力，V04 没有引用 V01 的这个发现**。
- **行动**：执行 V01 实验 5（canary）前，先 `git show 3771835ac` 读 diff 范围，确认 revert 的实际触及面（aiter or ATOM？哪几个文件？哪几行？）。若 revert 包含了 ATOM 侧 inter_dim padding，则 V04 的整个 padding 验证基础有问题。
- **优先级**：阻断 V01 实验 5、V04 实验 2/3。

#### 1.2 [V07] 其他模型 CSV 中存在同 ASM kernel 条目

- **V07 §A.4 表**：llama70B（≥6 hits）、llama405B（≥7 hits）的 CSV 文件中存在同一 buggy ASM kernel `bf16gemm_bf16_tn_256x256`，K=2048 同族，长序列下可能复现同 bug。
- **当前修复范围**：commit `a2883ab37` 仅删除了 glm5_bf16_tuned_gemm.csv 中的 1 行，未触及其他模型。
- **行动**：V07 实验 5.b 必须执行；任何 spot-check FAIL 都需登记到 §C.3 上报清单。
- **优先级**：不阻断 V07 PASS，但影响其他模型在 gfx950 上的可用性。建议提交 issue 给 AMD aiter 团队（V07 §C.3）。

#### 1.3 [V06] 最后 rank 的 `copy_` mis-broadcast 风险

- **V06 §A.1 末尾标注**：`expert_data.narrow(dim, 0, load_shard_size)` 用未 clamp 的 `load_shard_size` 作 destination，源 `loaded_weight` narrow 后是 clamped `size`。当 `size < load_shard_size` 时（即 `N % tp != 0` 且为最后 rank），`copy_` 的 src/dst shape 不匹配，PyTorch 的行为不确定（broadcast 还是 raise？）。
- **行动**：V06 Exp 1c（extreme oversharding）必须执行，验证 rank 3 / rank 4-7 不 crash 且 scale 值无 1.0 残留。
- **优先级**：阻断 V06 PASS（直接影响 Fix 3 的正确性结论）。

#### 1.4 [V01 ↔ V05] FP8 per_Token 是否走 V3 kernel fix

- **V01 §E.4**：fused_moe.py L906 `q_type not in (per_1x128, per_1x32)` 没有排除 `per_Token`，意味着 FP8 per_Token 也会进入 block_m=128 V3 强制路径。
- **V05 §A.1**：guard 集合一致结论是基于「blockscale 路径」，未明确讨论 per_Token 路径的 V3 kernel 是否注册。
- **行动**：V05 实验 2 跑通时，用 `AITER_LOG_LEVEL=INFO` 抓取 q_type 实际值（V05 §C.3 已建议）；若 q_type=per_Token 则需补一个 V3 kernel 名查询的实验。
- **优先级**：FP8 tp=2 已 PASS（MEMORY），间接说明无问题；但缺直接证据。

### 中优先（专题内待确认）

#### 1.5 [V02] BOS-spam 「噪声累积」证据强度不足

- V02 §A.3 明确指出 bisection 表（spam 率随 SwigluStep 层数单调上升）同样符合「kernel 数值偏差在某些 outlier 输入下」假说，**无法区分**。
- 实验 6（cos_sim(t) 衰减曲线）能定量回答此问题，但被标为可选。建议 reviewer 把实验 6 升级为 P0/P1 边缘的「条件 P0」：若实验 4 完美复现 summary，可降级到 P1；若实验 4 出现异常（如 C0 也有 spam），实验 6 升 P0。

#### 1.6 [V03] SLIDING_WINDOW 实际值未确认

- V03 §E 说明：实验前必须从 Step-3.5-Flash config 确认实际 W；本计划假设 W=512。
- 行动：V03 执行前 `grep -i sliding /root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash/snapshots/ab446a3.../config.json`，校准 ctx sweep 边界。

#### 1.7 [V04 ↔ V06] BF16 与 FP8 padding 时机差异未交叉验证

- V06 §A.4 表格已明确：BF16 在 `process_weights_after_loading` 后 padding，FP8 在 `_process_block_quant` 中 padding。
- V04 验证 BF16 路径，V06 验证 FP8 路径，但都没有「dump w13.shape, sc13.shape, inter_pad 后比对两路径 inter_pad=384 一致」的 cross-reference 实验（V06 §A.4 已建议但未列入 Exp）。
- 行动：在 V06 Sign-off 前补一个简短 cross-check（5 行 print），同时跑 BF16/FP8 各一次确认 inter_pad 一致。

#### 1.8 [V01 ↔ V02 ↔ V05] kernel JIT cache 命中验证

- V01 §E.2 / V02 实验 1 / V05 §C.3 都涉及「确认实际命中的 kernel 名」。
- 建议统一在执行前一次性 `ls /root/.cache/aiter/*` + `AITER_LOG_LEVEL=INFO` 抓 dispatch 日志，把 kernel 名记录到一份 dispatch_log.md。

---

## 2. 专题验证计划索引

| 专题 | 文件 | P0 实验 | 关键依赖 | 状态 |
|------|------|---------|---------|------|
| V01 | V01_moe.md | 1, 2, 3, 4 | 无 | 待执行 |
| V02 | V02_swiglu.md | 1, 2, 3, 5 | V01 PASS | 待执行 |
| V03 | V03_sliding_window.md | 1, 2, 3 | V01 PASS | 待执行 |
| V04 | V04_tp_support.md | C.2, 1, 2, 3 | V01 PASS | 待执行 |
| V05 | V05_fp8_inference.md | 1, 2, 3 | V04 PASS | 待执行 |
| V06 | V06_fp8_tp4.md | Exp 1, 2, 4, 5 | V04 + V05 PASS | 待执行 |
| V07 | V07_longseq_bos.md | 1, 2, 3, 5.a | V04 PASS | 待执行 |

---

## 3. P0 实验汇总（阻断性验证）

### V01-MoE P0 实验

#### V01.1 preshuffle on/off 对比（Fix 1 必要性）

```bash
cd /tmp && /opt/venv/bin/python /home/hanchang/aiter/op_tests/test_moe_2stage.py \
  --dtype bf16 --token 32 --model_dim 7168 --inter_dim 320 \
  --E 256 --topk 8 --use_g1u1 --preshuffle off 2>&1 | tee /tmp/v01_exp1_off.log
# 重复 --preshuffle on
```

通过标准：preshuffle_off cos_sim < 0.01 + preshuffle_on cos_sim ≥ 0.9999。
**风险**：V01 §E.1 标注 `--preshuffle` CLI 参数可能未注册，可能需写临时 driver。

#### V01.2 V1 vs V3 kernel inter_dim 边界矩阵（Fix 2 必要性）

矩阵：(inter_dim, block_m) ∈ {(192, 64), (256, 64), (256, 128), (320, 128), (640, 128)}
通过标准：inter=192/V1 PASS，inter=256/V1 FAIL，其余 PASS（cos_sim ≥ 0.9999）。
**风险**：V01 §E.2 标注 kernelName2 是否已编译进 JIT cache 待确认。

#### V01.3 端到端 tp=2 + tp=4 BF16

```bash
cd /tmp && rm -rf /root/.cache/atom/* && CUDA_VISIBLE_DEVICES=0,1 \
  /opt/venv/bin/python -m atom.examples.simple_inference \
  --model stepfun-ai/Step-3.5-Flash --kv_cache_dtype bf16 --trust-remote-code \
  --tensor-parallel-size 2 --level 0 --temperature 0 --max-tokens 128
# 重复 tp=4 with CUDA_VISIBLE_DEVICES=0,1,2,3
```

通过标准：exit 0 + 无乱码 + 无 BOS spam（人工目检）。

#### V01.4 静态代码核查（gfx942 回归）

- `grep -n "gfx950" /home/hanchang/ATOM/atom/model_ops/moe.py` 确认无残留 skip。
- `grep -n "get_gfx" /home/hanchang/aiter/aiter/fused_moe.py` 确认 L906 仅 gfx950 守卫。
- `git show ec8cbe8 68fc7d48b` 确认 diff 仅触及目标行。

---

### V02-SwigluStep P0 实验

#### V02.1 op_test SwigluStep kernel 正确性

```bash
cd /tmp && python -m aiter.test_moe_2stage \
    --activation swiglustep --preshuffle 1 \
    --M 32 --N 1024 --K 7168 --E 288 --topk 8 --seed 0 --dtype bf16
```

矩阵：M ∈ {1, 8, 16, 32, 64, 256, 1024}, preshuffle ∈ {0, 1}, seed ∈ {0, 1, 42}。
通过标准：cos_sim ≥ 0.99998，max_abs_err ≤ 0.05。
**风险**：V02 §B 列的实验前清理命令路径（`aiter/jit/build/...`）需要确认实际 stale .so 路径。

#### V02.2 真实权重层级验证（layer 43-44 + scale sweep）

矩阵：layer ∈ {43, 44}, M ∈ {16, 64, 256, 1024}, scale ∈ {0.5, 2.0, 5.0, 8.0}。
通过标准：scale ≤ 5.0 cos_sim ≥ 0.99998；scale=8.0 cos_sim ≥ 0.9999。
**风险**：依赖 `D_check_weights.py`（PROJECT_SUMMARY phase G 提及），V02 未给具体路径。

#### V02.3 端到端 max_tokens=128 (regression baseline)

通过标准：4 prompt 全部合理 + TTFT/TPOT 在 ±10%。

#### V02.5 CK kernel `_activation` 透传验证

三方法：日志注入 / codegen artifact 检查 / weight check 脚本。
通过标准：layer 43-44 路径为 SwigluStep。

---

### V03-SlidingWindow P0 实验

#### V03.1 ctx sweep cos_sim 验证

ctx ∈ {256, 511, 512, 513, 514, 1024, 4096}，对比修复前后 cos_sim。
通过标准：ctx=512 修复前 0.998982 / 修复后 ≥ 0.999998；ctx ≥ 513 修复前 < 0.999 / 修复后 ≥ 0.999998。
**风险**：V03 没有给具体复现脚本（草案级），实际执行需要先实现脚本。

#### V03.2 decode 专项（T=1, long context）

直接调用 `pa_decode_gluon` kernel，构造特征 KV 验证窗口最早 token 被纳入。
通过标准：修复后能检出特征值贡献，修复前为 0。

#### V03.3 端到端推理（去掉 ATOM_STEP3P5_NO_SLIDING workaround）

ctx ≈ {600, 2048, 10000} 输入，建议 tp=2 跑（避免与 V07 BOS bug 混淆）。
通过标准：无 "ungi" 乱码，输出连贯。
**注**：ctx=10000 与 V07 重叠，建议同时跑 tp=2 + tp=4 对比。

---

### V04-TP Support P0 实验

#### V04.C2 grep CK manifest 确认 align=192 边界

```bash
grep -rn "NPerBlock.*64\|NPerBlock.*128" /home/hanchang/aiter/csrc/ck_gemm_moe_2stages_codegen/
```

通过标准：找到 dispatch table，确认 192 是真实切换点。

#### V04.1 单算子 inter_dim 对齐验证

inter ∈ {160, 192, 320, 384}，verify CK kernel 接受 padded shape。
通过标准：inter=160/320 crash，inter=192/384 PASS cos_sim ≥ 0.9999。

#### V04.2 tp=4 端到端

通过标准：4 prompts PASS + TTFT/TPOT 与 baseline 对比 ≤ 20% 回退。

#### V04.3 tp=2 回归

通过标准：bit-exact 或 cos_sim=1.0 + TTFT/TPOT 不变。
**风险**：bit-exact 假设过严，浮点 reduction order 不同可能导致 ULP 级差异；建议放宽为 cos_sim ≥ 0.99999。

---

### V05-FP8 Inference P0 实验

#### V05.1 Crash 复现验证（Fix 1 必要性）

回退 q_type guard，预期 FP8 forward 在 fused_moe 2stage dispatch crash 或 gibberish。
通过标准：观察到 crash / dispatch error；恢复 guard 后实验 2 通过。
**注**：实验需修改源码并复原，必须 git checkout 还原。

#### V05.2 FP8 tp=2 端到端（核心）

```bash
cd /tmp && rm -rf /root/.cache/atom/* && CUDA_VISIBLE_DEVICES=0,1 \
  /opt/venv/bin/python -m atom.examples.simple_inference \
  --model stepfun-ai/Step-3.5-Flash-FP8 --trust-remote-code \
  --tensor-parallel-size 2 --level 0 --temperature 0 --max-tokens 1000
```

通过标准：4 prompts EOS + TTFT < 150ms + TPOT < 25ms + 无 gibberish。
**风险**：V05 §C.1 标注 FP8 模型缓存需要预检（首次运行会下载）。

#### V05.3 BF16 tp=2 回归

byte-level diff 与历史 baseline 一致（temperature=0）。
**风险**：byte-level diff 过严，建议放宽为 cos_sim 或 token-id 序列一致。

---

### V06-FP8 tp=4 P0 实验

#### V06.Exp 1 Scale loading 验证（包含 1b/1c 变体）

- 1a: 注入 dump 验证修复前后 sc13 值（无 1.0 残留）
- 1b: 离线 cover-completeness 断言（monkey-patch narrow）
- 1c: extreme oversharding（N=4, tp=8，rank 4-7 接收空切片不 crash）

通过标准：所有 scale 值在 [1e-5, 1e-1] 范围；ranks 4-7 不 crash。
**关键**：1c 涉及跨专题问题 1.3（mis-broadcast 风险）。

#### V06.Exp 2 FP8 tp=4 端到端（核心）

通过标准：输出连贯 + TTFT < 200ms + TPOT < 20ms + 无 ValueError / 无 shape mismatch。

#### V06.Exp 4 FP8 tp=2 回归（Fix 3 应为 no-op）

通过标准：TTFT ≈ 85ms + TPOT ≈ 13.5ms + 无新 error。

#### V06.Exp 5 Gibberish 复现（Fix 3 必要性 negative-control）

回退 ceil→floor，预期 output gibberish + Exp 1 dump 显示 1.0 残留。
**注**：必须严格还原。

---

### V07-LongSeq BOS P0 实验

#### V07.1 tgemm.mm 直调验证（最快）

```bash
cd /tmp && /opt/venv/bin/python /tmp/v07_exp1_tgemm.py
```

M ∈ {4096, 8192, 8208, 8209, 8214, 8216, 8240, 8990, 9007, 9019, 10021, 12288, 16384}。
通过标准：所有 M diff < 5（修复后预期 ≈ 0）。
**风险**：脚本需要先创建（V07 已给完整代码）。

#### V07.2 E2E 长序列验证

ctx=10021，verify first_output_token ≠ 0，输出 "好的"。
通过标准：first_token ≠ 0 + diversity > 1 + 输出连贯。

#### V07.3 短 prompt 回归

tp=4 短 prompt 不应受 workaround 影响。
通过标准：4 sample 连贯 + TPOT 在 ±10%。

#### V07.5.a CSV 扫描

枚举所有 model_configs 中 256x256 ASM kernel 条目。
通过标准：扫描完整 + 与 §A.4 表一致。

---

## 4. 验证质量评估

### 4.1 Master Review 发现的问题

| 专题 | 问题类型 | 描述 | 严重度 |
|------|---------|------|--------|
| V01 | 命令完整性 | 实验 1/2 的 `--preshuffle` / `--block_size_M` CLI 参数可能未注册（V01 §E.1）；可能需写临时 driver | 中 |
| V01 | 命令完整性 | 实验 2 的 monkey-patch `get_block_size_M` 实操路径未给完整代码 | 中 |
| V01 | 标准量化 | 实验 3「人工目检 + grep `<s><s><s>`」不完全量化，建议加 token-id 多样性阈值 | 低 |
| V02 | 命令完整性 | 实验 2 依赖 `D_check_weights.py`，V02 未给具体路径 | 中 |
| V02 | 命令完整性 | 实验 4 的 `ATOM_DISABLE_SWIGLUSTEP_LAYERS` env 不存在（V02 §E.3 已标注）；需先加临时开关 | 高 |
| V02 | 标准量化 | 实验 1 清理 stale .so 命令路径需确认 | 低 |
| V03 | 命令完整性 | 实验 1/2 没有给具体复现脚本，仅描述配置 | 高 |
| V03 | 命令完整性 | SLIDING_WINDOW=512 是假设值，未从模型 config 确认 | 中 |
| V04 | 命令完整性 | 实验 1 单算子 driver 是伪码 | 中 |
| V04 | 标准量化 | 实验 3 「bit-exact」过严，建议放宽到 cos_sim ≥ 0.99999 | 中 |
| V04 | 覆盖性 | tp=8 端到端因 GPU5 阻塞，实验 5 给的 4 个降级方案需挑一个落地 | 高 |
| V05 | 命令完整性 | FP8 模型缓存预检命令已给（§C.1），但首次下载可能阻塞 | 低 |
| V05 | 标准量化 | 实验 3 byte-level diff 过严，建议放宽 | 中 |
| V06 | 标准量化 | Exp 3 性能比 [15%, 25%] 区间过严（设备/驱动波动可能超 ±5%）| 低 |
| V06 | 跨专题 | A.1 末尾的 mis-broadcast 风险（问题 1.3）必须 Exp 1c 验证 | 高 |
| V07 | 覆盖性 | 实验 5.b 只给思路未给完整脚本 | 中 |
| V07 | 跨专题 | llama70B/llama405B 同 ASM kernel 影响（问题 1.2）超出 V07 范围，需开 V08 | 高 |

### 4.2 各专题计划质量评分

| 专题 | 命令完整性 | 标准量化 | 依赖覆盖 | 总评 |
|------|----------|---------|---------|------|
| V01 | 7/10 | 9/10 | 9/10 | A- |
| V02 | 6/10 | 8/10 | 9/10 | B+ |
| V03 | 5/10 | 9/10 | 8/10 | B |
| V04 | 7/10 | 8/10 | 9/10 | A- |
| V05 | 8/10 | 7/10 | 9/10 | A- |
| V06 | 9/10 | 8/10 | 9/10 | A |
| V07 | 9/10 | 9/10 | 8/10 | A |

**整体评估**：所有专题 P0 实验的「该测什么」都明确，但「具体怎么测」在 V03/V04/V02 上需要补脚本。V06/V07 是质量最高的两份计划。

---

## 5. 风险与局限

### 5.1 硬件限制

- **tp=8 端到端不可测**：GPU5 硬件异常（~700ms/tensor），影响 V01/V04/V07 的 tp=8 验证。所有 tp=8 结论目前只能停留在「单算子 + 代码推断」级别。
- **gfx942 不可测**：本机仅 gfx950，所有 gfx942 回归只能做代码静态核查，影响 V01 实验 4 的覆盖率。

### 5.2 跨专题已知 open bug

- **tp=4 长序列 ≥10k 输出全 BOS**：V07 主修复目标。MEMORY 标注为 open bug；V01/V03/V05/V06 的 tp=4 实验都需避免 prompt 长度 ≥10k，否则会被 V07 bug 污染结果（V06 §C.1 已明确避开此组合）。

### 5.3 验证盲区（无法证伪）

- **kernel JIT cache 一致性**：所有 op_test 实验都依赖 CK kernel 是否真编译进 JIT cache。建议执行前统一一次 dispatch_log dump，避免假性 PASS / FAIL。
- **noise 累积假说**（V02 BOS-spam）：实验 6 是定量验证，但被列为可选；若不跑，结论只是「方向正确」。
- **FP8 per_Token 路径**：V01 §E.4 标注未排除，目前依赖 V05 端到端间接证明。
- **3771835ac revert 实际范围**（问题 1.1）：V01 §E.3 已标注张力，必须先读 commit diff 才能确认 V01 实验 5 与 V04 padding 验证的关系。

### 5.4 工程限制

- **spawn 子进程 monkey-patch 不传播**：影响 V06 Exp 1 dump（V06 §C.2 已给替代方案：env-gated 源码 patch）；V02 实验 5 方法 A 同样受影响。
- **必须 `cd /tmp &&`**：所有 python 命令前缀（aiter namespace package 问题）。各专题命令已遵守。
- **GPU5 排除**：CUDA_VISIBLE_DEVICES 不能用 5。

---

## 6. 执行 Checklist

### Phase 0 — 预检（执行任何实验前）

- [ ] **0.1** 确认 aiter 工作树 commit：`cd /home/hanchang/aiter && git log -1 --format='%H %s'` 应为 `c38d0c9e6` 或更新（含 V05 Fix 1）
- [ ] **0.2** 确认 ATOM 工作树 commit：`cd /home/hanchang/ATOM && git log -1 --format='%H %s'` 应为 `ccb64621` 或更新（含 V06 Fix）
- [ ] **0.3** 确认 aiter 安装路径：`/opt/venv/bin/python -c "import aiter; print(aiter.__file__)"` 必须指向 `/home/hanchang/aiter/aiter/__init__.py`
- [ ] **0.4** 跨专题问题 1.1：`cd /home/hanchang/aiter && git show 3771835ac --stat` 确认 revert 范围
- [ ] **0.5** 模型缓存预检：`ls ~/.cache/huggingface/hub/ | grep -i flash`（BF16 + FP8 都要有）
- [ ] **0.6** Step-3.5-Flash SLIDING_WINDOW 校准（V03 前置）
- [ ] **0.7** 检查 GPU 可用性：`rocm-smi`，确认 GPU 0-4, 6, 7 正常，GPU 5 标注异常

### Phase 1 — V01 MoE（必须最先）

- [ ] **1.1** V01 实验 4：静态 grep（5 分钟，可与 0.x 并行）
- [ ] **1.2** V01 实验 1：preshuffle on/off 对比
- [ ] **1.3** V01 实验 2：V1/V3 inter_dim 边界矩阵（5 个 case）
- [ ] **1.4** V01 实验 3：端到端 tp=2 + tp=4 BF16
- [ ] **1.5** V01 实验 5（P1）：buffer padding canary（依赖 0.4 结论）

### Phase 2 — V02 / V03 / V04 并行（V01 PASS 后）

V02:
- [ ] **2.1** V02 实验 1：op_test SwigluStep kernel
- [ ] **2.2** V02 实验 5：CK kernel activation 透传验证
- [ ] **2.3** V02 实验 2：层级验证 layer 43-44
- [ ] **2.4** V02 实验 3：端到端 max_tokens=128
- [ ] **2.5** V02 实验 4（P1，需先加 env 开关）：BOS-spam 复现
- [ ] **2.6** V02 实验 7（P1）：shared expert 反向

V03:
- [ ] **3.1** V03 实验 1：ctx sweep cos_sim
- [ ] **3.2** V03 实验 4：短 ctx regression
- [ ] **3.3** V03 实验 2：decode 专项
- [ ] **3.4** V03 实验 3：端到端去掉 workaround

V04:
- [ ] **4.1** V04 §C.2：grep CK manifest 确认 192 边界
- [ ] **4.2** V04 实验 3：tp=2 回归（最快）
- [ ] **4.3** V04 实验 1：单算子 inter sweep
- [ ] **4.4** V04 实验 2：tp=4 端到端
- [ ] **4.5** V04 实验 4-B：ca_comm fallback 命中验证
- [ ] **4.6** V04 实验 5：tp=8 静态 + 单算子降级

### Phase 3 — V05 / V06 / V07 并行（V04 PASS 后）

V05:
- [ ] **5.1** V05 §C.1：FP8 模型缓存预检
- [ ] **5.2** V05 实验 2：FP8 tp=2 端到端
- [ ] **5.3** V05 实验 3：BF16 tp=2 回归
- [ ] **5.4** V05 实验 1：Crash 复现（修源码 + 复原）
- [ ] **5.5** V05 实验 5（P1）：block_shape 区分性

V06:
- [ ] **6.1** V06 Exp 1a：scale dump（含 dispatch_log）
- [ ] **6.2** V06 Exp 1b：cover-completeness 断言
- [ ] **6.3** V06 Exp 1c：extreme oversharding（跨专题问题 1.3 验证）
- [ ] **6.4** V06 Exp 2：FP8 tp=4 端到端
- [ ] **6.5** V06 Exp 4：FP8 tp=2 回归
- [ ] **6.6** V06 Exp 5：gibberish negative control（修源码 + 复原）
- [ ] **6.7** V06 Exp 3（P1）：FP8 vs BF16 性能对比
- [ ] **6.8** V06 跨专题 1.7：BF16/FP8 inter_pad 一致性 cross-check

V07:
- [ ] **7.1** V07 实验 5.a：CSV 扫描（无 GPU 依赖，最早执行）
- [ ] **7.2** V07 实验 1：tgemm 直调（含 C.1 qkv 直验扩展）
- [ ] **7.3** V07 实验 2：E2E 10k tokens
- [ ] **7.4** V07 实验 3：短 prompt 回归
- [ ] **7.5** V07 实验 6（P1）：tp=2 长序列 negative control
- [ ] **7.6** V07 实验 4（P1）：性能影响
- [ ] **7.7** V07 实验 5.b（P1）：其他 CSV spot-check（任何 FAIL 登记到 §C.3）

### Phase 4 — 总结（所有 Phase PASS 后）

- [ ] **8.1** 汇总每个专题的 PASS/FAIL 矩阵
- [ ] **8.2** 整理跨专题待确认问题（§1）的实验答案
- [ ] **8.3** 提交 V07 §C.3 的 ASM kernel issue 给 AMD aiter 团队
- [ ] **8.4** 评估是否需要 V08 专题（llama70B/llama405B + ASM kernel）
- [ ] **8.5** 更新 MEMORY.md（修复后状态、新发现的 open bug）

---

## 7. 关键文件路径速查

| 用途 | 路径 |
|------|------|
| ATOM moe.py | `/home/hanchang/ATOM/atom/model_ops/moe.py` |
| ATOM step3p5.py | `/home/hanchang/ATOM/atom/models/step3p5.py` |
| aiter fused_moe.py | `/home/hanchang/aiter/aiter/fused_moe.py` |
| aiter parallel_state.py | `/home/hanchang/aiter/aiter/dist/parallel_state.py` |
| aiter pa_decode_gluon.py | `/home/hanchang/aiter/aiter/ops/triton/gluon/pa_decode_gluon.py` |
| aiter tuned_gemm.py | `/home/hanchang/aiter/aiter/tuned_gemm.py` |
| aiter glm5 csv | `/home/hanchang/aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv` |
| op_test (MoE 2stage) | `/home/hanchang/aiter/op_tests/test_moe_2stage.py` |
| simple_inference | `/home/hanchang/ATOM/atom/examples/simple_inference.py` |
| 长序列 driver | `/home/hanchang/project_fp8_tp4/logs/perf_compare_10k/run_inference.py` |
| BF16 模型权重 | `/root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash/snapshots/ab446a3.../` |
| FP8 模型权重 | `~/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash-FP8/`（待预检） |
| PR 仓库（commit/push 必须从这里） | `/home/hanchang/junlin12_repos/` |

---

## 8. 总结

**总体可执行性**：所有 7 个专题计划逻辑严密，P0 实验有清晰的通过标准。主要执行障碍是 V02/V03/V04 的部分实验脚本仅给草案，需在执行前补完整代码（约 0.5-1h 工程量/专题）。

**最高风险**：跨专题问题 1.1（buffer padding revert 范围）必须在 Phase 0 解决，否则会污染 V01 实验 5 + V04 padding 验证。

**最高价值**：V06 Exp 1c（extreme oversharding，问题 1.3）+ V07 实验 5.b（其他 CSV spot-check，问题 1.2），两者揭示了潜在的二阶 bug，必须执行。

**总工时**：约 26-32h（不含跨专题问题 1.1 调查的 2-3h 与脚本补完的 3-5h）。建议预算 5 个工作日。

---

Master Pipeline 编制完成
