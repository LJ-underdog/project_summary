# 验证 Pipeline Review 最终报告

生成日期：2026-04-25
Reviewer：A（kernel 正确性 / V01-V03）、B（TP 分布式 / V04+V06）、C（FP8 + MASTER / V05+V07+整体）
Synthesizer：整合输出

---

## 执行摘要

整体 pipeline 设计合理、依赖图清晰，所有 7 个专题的「该测什么」明确，**总体评分 7.5/10**，**建议「补完脚本与 Phase 0 增强后执行」**。三组 Reviewer 共同指出的最关键问题集中在：(1) **缺乏全局 noise floor，cos_sim 阈值跨专题没有统一基线**（A/B/C 均提及）；(2) **修改工作树缺少隔离机制**（V01-Exp5 / V02-Exp4 / V05-Exp1 / V06-Exp5 全部直接动主仓库）；(3) **多个关键 negative-control 被错放在 P1**（V07-Exp6 tp=2、V07-Exp5b llama spot-check、V02-Exp6 noise 累积）。Phase 2 / Phase 3 GPU 资源调度方案三方有冲突，需 Synthesizer 裁决（见 §2.1）。建议先完成 4-6h 的 Phase 0 增强 + 脚本补完，再启动 Agent Team 执行；预计总工时可从 MASTER 估的 26-32h 压缩到 ~12h。

---

## 1. 逻辑合理性改进

### 1.1 必须修正（执行前需解决）

| # | 专题 | 问题 | 来源 Reviewer | 修正方案 |
|---|------|------|-------------|---------|
| L1 | 全局 | cos_sim 阈值无 noise floor 基线（V01=0.9999/V02=0.99998/V03=0.999998 三数量级跳） | A (S1) + B (V06 §3) + C (4.3) | Phase 0 新增 V00：固定 seed × 5 次同 op 跑 cos_sim 取 worst，作为各 op 的 baseline，阈值 = baseline + ε |
| L2 | V01/V02/V05/V06 | 修改工作树缺隔离（V01-Exp5 注释 ATOM padding、V02-Exp4 加 env switch、V05-Exp1 改 aiter fused_moe、V06-Exp5 改 ATOM moe.py） | A (S3) + B (V04/V06 §Worktree) + C (1.1.2) | 强制规则：所有「修源码」实验必须 `EnterWorktree` 或在 `/home/hanchang/junlin12_repos/` clone 中执行；执行前 `git status --porcelain` 必须为空 |
| L3 | 跨专题 | commit 3771835ac 的 revert 实际范围未确认（V01 §E.3 标注，V04 未引用） | C (3.2.3) + MASTER §1.1 | Phase 0 新增 0.4a/b/c/d 四个 sub-step，必须看到完整 diff 并判断是否触及 ATOM moe.py L489-518；若是则阻断 V04 启动 |
| L4 | V01 Exp 2 | kernelName2 是否真命中未直接验证（fallback 假性 PASS 风险） | A (V01 L3) + MASTER §1.8 | 强制 `AITER_LOG_LEVEL=INFO` + grep 实际命中 kernel 名，写入 dispatch_log.md |
| L5 | V03 | SLIDING_WINDOW=512 是假设值，未实测 | A (V03 L2) + MASTER §1.6 | Phase 0 新增 0.6：`grep -i sliding /root/.cache/huggingface/.../config.json` |
| L6 | V06 Exp 5 | ceil→floor + 去 clamp 是两个变化，因果不可分 | B (V06 §1.4) | 拆为 5a（只 revert ceil 保留 clamp）+ 5b（只 revert clamp 保留 ceil） |
| L7 | V06 Exp 1 | `expert_data.copy_(loaded_weight)` shape mismatch 行为未直接测 | B (V06 §1.1) | 补 Exp 1b'：`dest.narrow(0,0,3).copy_(tensor([0.5]))` 单 1 行验证 broadcast/raise/部分写入 |
| L8 | V04 + V06 | FP8 scale 是否需同步 inter padding（联合问题 2） | B (联合问题 2) | 必须做 cross-check：dump `sc13.shape`，验 `w13[1]/2 == sc13[1]*128 == 384`；若 sc13[1]=2 则 V06 Fix 2 不完整 |
| L9 | V05 Exp 1 | JIT cache 污染未清理（仅清 `/root/.cache/atom/*`，未清 `/root/.cache/aiter/*`） | C (1.3.2) | 升级清理命令 `rm -rf /root/.cache/atom/* /root/.cache/aiter/* ~/.aiter_cache/*`，并在 Exp 1 完成后再清一次 |

### 1.2 建议改进（执行中注意）

| # | 专题 | 问题 | 来源 | 建议 |
|---|------|------|------|------|
| L10 | V01 Exp 1 | 缺反例对照组（移除 ATOM gfx950 skip 的对照） | A (V01 L1) | 在 Exp 5 之外补反例 |
| L11 | V01 Exp 2 | 边界 192-256 之间未细化（200/224/240） | A (V01 L2) | 加 3 个 case 明确 PASS 上界 |
| L12 | V02 Exp 4 | BOS-spam ±20% 容差等价于 50%-90% 范围，几乎不可证伪 | A (V02 L1) | 改用卡方检验 (p<0.05) |
| L13 | V02 Exp 5 | weight check 脚本路径未给 (`D_check_weights.py`) | A (V02 L3) | 执行前 Glob 验证 |
| L14 | V02 Exp 7 | "无论结果如何都 PASS" 等于不验证 | A (V02 L4) | 量化：必须 cos_sim 或 BLEU 劣化 ≥ X% 才算「shared 反向防御必要」 |
| L15 | V03 Exp 2 | 特征 KV 贡献量级未给阈值 | A (V03 L3) | 给定 σ：`output_norm_diff > σ` |
| L16 | V03 Exp 3 | 长 prompt（10000）会撞 V07 已知 tp=4 BOS bug | A (V03 风险) | 强制 tp=2，把 tp=4 长序列移到 V07 |
| L17 | V04 Exp 4-A/B | 「被动监控」「assert+print」不能证明 fallback 输出正确，只方法 C（数值单元测试）是真验证 | B (V04 §3) | 方法 C 升级为 P0，A/B 作为补充 |
| L18 | V05 §A.1 | guard 黑名单可扩展性差，未来新 q_type 默认错配 | C (1.1.1) | 加 import-time assert：`QuantType.per_1x*` 全部在 guard tuple 内 |
| L19 | V05 Exp 5 | 「无差异 = PASS」可能掩盖 silent fallback | C (1.1.3) | 必须配合 dispatch 日志验证 block_shape 实际下游值 |
| L20 | V07 Exp 4 | 性能微 bench 缺 M=16384 真实权重正确性 spot-check | C (2.1.1) | 加一个 case：用真实 o_proj weight 跑 M=16384 vs fp32 ref |

---

## 2. Agent Team 效率优化方案

### 2.1 GPU 资源分配总图（Phase 2 + Phase 3）

**约束**：8 GPU 中 GPU 5 硬件异常必须排除，可用 7 张（0,1,2,3,4,6,7）。

#### Phase 2（V02 + V03 + V04 并行）

```
时间窗 T+0~30min:
  GPU 0,1,2,3   → V04-A4 (tp=4 端到端 + ca_comm)        关键路径
  GPU 4         → V04-A2 (tp=2 回归)
  GPU 6         → V03-A   (W 验证 + ctx sweep)
  GPU 7         → V02-A   (op_test SwigluStep)
  CPU only      → V04-A1  (CK manifest grep)             与上述并行
  CPU only      → V03 W 数值确认                          与上述并行

时间窗 T+30~60min:
  GPU 0,1,2,3   → V04 inter sweep (释放后) + V04 实验 5 单算子降级
  GPU 4,6       → V03-B (decode 专项) + V03-C (端到端 tp=2)
  GPU 7         → V02-B (layer 43-44 验证)

时间窗 T+60~90min:
  GPU 0,1,2,3   → 释放
  GPU 4,6       → V02-D (端到端 + shared 反向)
  GPU 7         → V02-E (BOS-spam + noise 累积，关键路径瓶颈，4-6h)
```

#### Phase 3（V05 + V06 + V07 并行）

```
时间窗 T+0~30min:
  GPU 0,1       → V05-B  (FP8 tp=2 端到端)
  GPU 2,3       → V05-C  (BF16 tp=2 回归)
  GPU 4,6       → V06-B3 (FP8 tp=2 + perf)
  GPU 7         → V07-A (tgemm 直调) + V05-D (模型预检)
  CPU only      → V07-B (CSV 扫描)

时间窗 T+30~60min:
  GPU 0,1,2,3   → V06-B2 (FP8 tp=4 端到端 + scale dump)   关键路径
  GPU 4         → V07-D (tp=2 negative control，10k)      ⚠️ 升 P0
  GPU 6         → V05-D 转 Exp 5 (block_shape 验证)
  GPU 7         → V07-F (perf bench)

时间窗 T+60~90min:
  GPU 0,1,2,3   → V07-C (E2E 10k tokens)
  GPU 6         → V06-B4 (gibberish 复现 5a/5b)            独占 worktree
  GPU 7         → V07-G (其他 CSV spot-check) ⚠️ 升 P0/V08

时间窗 T+90~120min:
  GPU 0,1       → V05-A (Exp 1 crash 复现，独占 aiter 源码) 必须最后
```

**裁决**（三方分歧）：
- Reviewer-B 建议 V06-B2 用 GPU 0,1,2,3，C 建议用 GPU 2,3,4,6 → **采纳 B 方案**（连续 4 卡 NCCL 性能更好；GPU 4 留给 V07-D negative control）
- Reviewer-C 把 V07 Exp 2 推迟到 Phase 3 末段，A 未涉及 → **采纳 C 方案**，但前移 30min（T+60min）

### 2.2 每个专题的 Agent 分工方案

#### V01-MoE（来源：A）
- **V01-A**（CPU）：实验 4 静态 grep — 15 min — 无依赖
- **V01-B**（GPU 0,1）：实验 1 preshuffle 对照 — 30 min — 无依赖
- **V01-C**（GPU 0,1）：实验 2.a-2.e + 边界细化 200/224/240 — 60-90 min — 无依赖
- **V01-D**（GPU 0,1,2,3）：实验 3 端到端 tp=2 + tp=4 — 40 min — 依赖 B+C PASS
- **V01-E**（worktree + GPU 0,1）：实验 5 padding canary — 30 min — 依赖 D PASS
- 节省：4h → 2.5h（37%）

#### V02-SwigluStep（来源：A）
- **V02-A**（GPU 7）：实验 1 op_test 矩阵 7×2×3=42 case — 60 min — 依赖 V01 PASS
- **V02-B**（GPU 6）：实验 2 layer×M×scale=2×4×4=32 case — 90 min — 依赖 V01
- **V02-C**（GPU 7）：实验 5 三方法 wiring 透传 — 30 min — 依赖 V01
- **V02-D**（GPU 0,1）：实验 3 + 实验 7 共享 driver — 60 min — 依赖 A+C PASS
- **V02-E**（GPU 7）：实验 4 + 实验 6 强制配对（关键路径瓶颈） — 4-6h — 依赖 D PASS
- 节省：10h → 7h（30%）

#### V03-SlidingWindow（来源：A）
- **V03-A**（GPU 6）：W 数值确认 + 实验 1+4 合并 ctx sweep — 45 min — 无
- **V03-B**（GPU 6）：实验 2 decode 专项 — 60 min — 依赖 A
- **V03-C**（GPU 4）：实验 3 buggy/fixed/reference 三方对照 — 60 min — 依赖 A
- 节省：3h → 1.5h（50%）

#### V04-TP（来源：B）
- **V04-A1**（CPU）：CK manifest grep + ca_comm grep — 0.5h — 无
- **V04-A2**（GPU 4）：实验 3 tp=2 BF16 回归 — 0.5h — 无
- **V04-A3**（GPU 6,7）：实验 1 inter sweep + 实验 5.3 — 1h — 依赖 A1
- **V04-A4**（GPU 0,1,2,3）：实验 2 tp=4 + 实验 4-B/4-C — 1.5h — 依赖 A2 释放
- 节省：4.5h → 2h（55%）

#### V05-FP8 Inference（来源：C）
- **V05-A**（GPU 0,1，**独占 aiter 源码 + worktree**）：实验 1 crash 复现 — 30 min — 必须最后跑
- **V05-B**（GPU 0,1）：实验 2 FP8 tp=2 — 20 min — 与 C/D 并行
- **V05-C**（GPU 2,3）：实验 3 BF16 tp=2 回归 — 20 min
- **V05-D**（GPU 6）：模型预检 + 实验 5 block_shape — 25 min
- 节省：3h → 75 min（58%）

#### V06-FP8 tp=4（来源：B）
- **V06-B1**（CPU/GPU 0）：Exp 1b/1c offline cover-completeness — 0.5h
- **V06-B2**（GPU 0,1,2,3，**worktree 隔离**）：Exp 1a + Exp 2 端到端 + scale dump — 1h
- **V06-B3**（GPU 4,6）：Exp 3 perf + Exp 4 tp=2 回归 — 1h
- **V06-B4**（GPU 0,1 / 后段，**独占 ATOM moe.py worktree**）：Exp 5a + 5b — 1h — 依赖 B2
- 节省：4.5h → 2h（55%）

#### V07-LongSeq BOS（来源：C）
- **V07-A**（GPU 7）：Exp 1 tgemm 13 个 M — T+0
- **V07-B**（CPU）：Exp 5.a CSV 扫描 — T+0
- **V07-C**（GPU 0,1,2,3）：Exp 2 E2E 10k tp=4 — T+0
- **V07-D**（GPU 4,6）：Exp 6 tp=2 negative control（**升 P0**） — T+0
- **V07-E**（GPU 0,1,2,3）：Exp 3 短 prompt 回归 — T+30min（C 后）
- **V07-F**（GPU 7）：Exp 4 perf bench — T+30min
- **V07-G**（GPU 7）：Exp 5.b 其他 CSV spot-check — T+45min
- 节省：3h → 60 min（67%）

### 2.3 整体并行效率提升估算

| 方案 | 总工时 | 说明 |
|------|--------|------|
| 纯串行 | 26-32h | 当前 MASTER 估算 |
| 专题间并行（无 agent team）| ~18h | Phase 2/3 三专题并行，各专题内串行 |
| Agent Team 优化 | **~12h** | 专题内并行 + GPU 跨专题并发 + V02-E 提前启动 |
| 含 Phase 0 增强 + 脚本补完 | **~16h** | 加 4h 前置工作 |

**关键路径**：V02-E（BOS-spam + noise 累积量化）4-6h 是瓶颈，建议 V01 PASS 后立即提前 dispatch（不等 V02-A/B/C/D），允许 V02-E 跨 Phase 2+3 跑。

---

## 3. 验证准确性改进

### 3.1 需要量化的主观标准

| 专题 | 当前标准 | 建议量化标准 | 来源 |
|------|---------|------------|------|
| V01 Exp 3 | 人工目检无乱码 | (1) `len(set(token_ids[1:10])) >= 5` (2) `<s>` 计数 ≤ 1 (3) BLEU ≥ 0.85 vs BF16 ref | A (S2) |
| V02 Exp 3/4 | 人工目检无乱码 | repetition rate < 0.3 + diversity > 0.5 | A (S2) |
| V03 Exp 3 | 无 "ungi" 乱码 pattern | token-level repetition rate < 阈值 | A (V03 风险) |
| V04 Exp 1 | 不 crash + cos_sim ≥ 0.9999 | 加打印 `w13_new.stride()` + `is_contiguous()` | B (V04 §2) |
| V05 Exp 5 | "无差异 = PASS" | 必须配 `AITER_LOG_LEVEL=INFO` 显示 `block_shape` 实际下游值 | C (1.1.3) |
| V06 Exp 2/5 | "coherent" / "gibberish" | diversity ratio > 0.5 / < 0.05；max n-gram repetition < 0.3 / > 0.8；perplexity（可选） | B (V06 §3) |
| V07 Exp 1 | diff < 5 | strict: diff < 1；loose: diff < 50（区别于 buggy 197+） | C (2.3.1) |
| V07 Exp 2 | first_token ≠ 0 + diversity > 1 | (1) `first_token == 3648`(已知值) (2) `len(set(token_ids)) >= 5` (3) `0 not in token_ids[1:]` | C (2.3.2) |
| V02 Exp 4 | spam 命中率 ±20% | 卡方检验 p < 0.05 | A (V02 L1) |
| V02 Exp 6 | "导数有上界" | `dcos/dt < 1e-4/step` 具体数值阈值 | A (V02 §准确性) |

### 3.2 优先级调整（P1 → P0）

| 实验 | 原优先级 | 建议 | 理由 | 来源 |
|------|---------|------|------|------|
| V07 Exp 6（tp=2 long-seq negative control）| P1 | **P0** | 唯一直接验证 ASM kernel 影响面假设；FAIL 推翻整个 §A.5 表 + 影响 V05 FP8 tp=2 结论 | C (2.1.2) |
| V07 Exp 5.b（其他 CSV spot-check）| P1 | **P0**（或拆 V08）| llama70B/llama405B 上线时复发 BOS bug 的唯一定量证据 | C (2.1.3) + MASTER §1.2 |
| V02 Exp 6（cos_sim 衰减曲线）| P2 可选 | **条件 P0** | Exp 4 「噪声累积」结论强烈依赖它；不跑则 Exp 4 PASS 不能反推根因 | A (V02 L2) |
| V04 Exp 4-C（数值单元测试 vs torch ref）| 三选一 | **必做 P0** | A/B 只能证明 fallback 被触发，C 才证明输出正确 | B (V04 §3) + (V04 准确性 3) |
| V01 Exp 5（buffer padding canary）| P1 | **条件 P0** | 若 Phase 0 0.4d 显示 ATOM padding 也被 revert，则升 P0 | C (3.2.3) + MASTER §1.1 |
| V06 Exp 1c（extreme oversharding）| P0 | 维持 P0 | 已正确 | B + MASTER §1.3 |
| V05 Exp 5（block_shape 区分性）| P1 | 维持 P1，**通过标准升级** | 必须配 dispatch 日志 | C (1.1.3) |

### 3.3 通过标准调整建议

| 专题 | 当前标准 | 建议调整 | 严格化/放宽 |
|------|---------|---------|-----------|
| V04 Exp 3 | bit-exact tp=2 | cos_sim ≥ 0.99999 + token-id 序列一致 | **放宽**（浮点 reduction 不确定性） |
| V05 Exp 3 | byte-level diff vs baseline | cos_sim 或 token-id 序列 | **放宽**（同上）|
| V07 Exp 1 | diff < 5（统一） | strict<1 / loose<50 双档 | **严格化**（buggy 197+，5 过宽） |
| V06 Exp 3 | TPOT 加速 [15%, 25%] | TPOT_fp8 < TPOT_bf16 × 0.90 作为 gate；19% ±5% 作 informational | **放宽**（设备波动） |
| V01 Exp 1 cos_sim ≥ 0.9999 | 经验值 | 待 V00 noise floor 跑出后定 | 待定 |
| V02 Exp 1 cos_sim ≥ 0.99998 | 经验值 | 待 V00 noise floor + SwigluStep clamp 噪声评估 | 可能需放宽 |
| V03 Exp 1 cos_sim ≥ 0.999998 | 经验值（已有 buggy 0.998 vs fixed 0.999998 实测） | 维持 | — |
| V01 Exp 3 cos_sim < 0.01 判 FAIL | abs ≤ 0.01 | 维持 | — |

---

## 4. 新增发现（三组 Reviewer 共性问题）

按提及频次排序：

### N1. cos_sim 阈值无统一 noise floor（A + B + C，3 票）
- **A** 提出 V00 全局 noise floor 实验
- **B** 在 V06 §3 指出 gibberish 量化阈值缺失
- **C** 在 4.3 要求出「数值精度阈值表」
- **裁决**：Phase 0 强制新增 V00 noise floor 实验，作为所有 cos_sim 阈值的标定基线。

### N2. 修改工作树缺隔离（A + B + C，3 票）
- **A** S3：V01-Exp5 / V02-Exp4 / V03-checkout 直接动主仓库
- **B** V04/V06：实验 4-B 和 V06-Exp1 必须用 EnterWorktree
- **C** 1.1.2：V05-Exp1 + V06-Exp5 需 git status 守卫 + 互斥锁
- **裁决**：Phase 0 强制制定「源码修改协议」：必须 worktree + git status pre-check + 备份 + diff 验证还原。

### N3. JIT cache 状态未在执行前确认（A + B + C，3 票）
- **A** S4：V01 担心 kernelName2、V02 担心 stale .so、V03 checkout 触发重编译
- **B** V06 §1：env-gated patch 行号偏移可能影响 JIT
- **C** 1.3.2：V05-Exp1 仅清 atom cache，未清 aiter cache
- **裁决**：Phase 0 加 0.8 全局 JIT cache 健康检查 + 统一清理命令模板：`rm -rf /root/.cache/atom/* /root/.cache/aiter/* ~/.aiter_cache/*`。

### N4. 端到端实验在 V01/V02/V03 重复（A 提出，B/C 隐含）
- **A** "相互依赖" 部分指出 V01-Exp3 已隐式 verify V02 wiring
- **裁决**：把「端到端 tp=2 短 prompt」提到 pipeline 顶层共享一次，三专题在 e2e 实验合并节省 ~60min。

### N5. 跨专题 V04 ↔ V06 inter_pad 一致性未列入实验（B 强调，MASTER 提及）
- **B** 联合问题 1：BF16 与 FP8 的 padded shape 必须一致（都=384）
- **MASTER** §1.7 已识别但未列入 Exp
- **裁决**：必须做 cross-check 实验，作为 V06 sign-off gate；同时检查 sc13 是否需要同步 padding（联合问题 2）。

---

## 5. 最终执行建议

### Phase 0 修订（预检新增项）

原 0.1-0.7 保留。新增：

- [ ] **0.4a/b/c/d**（升级 0.4）：`git show 3771835ac --stat` + 看 aiter diff + 看 ATOM moe.py diff + 若触及 L489-518 阻断 V04
- [ ] **0.5a/b/c**（拆分 0.5）：BF16 模型 + FP8 模型 + FP8 模型 dry-run 加载
- [ ] **0.6**（升级）：grep config.json 确认 SLIDING_WINDOW 实际值
- [ ] **0.8**（新增）：JIT cache 健康检查 — `ls /root/.cache/aiter/* | wc -l` + 统一清理命令测试
- [ ] **0.9**（新增）：V00 全局 noise floor 实验 — 固定 seed × 5 次同 op，输出 MoE/SwigluStep/paged_decode 的 cos_sim worst-case，作为各专题阈值标定输入
- [ ] **0.10**（新增）：源码修改协议落地 — 检查 EnterWorktree 工具可用性，准备 worktree 模板路径
- [ ] **0.11**（新增）：aiter logger INFO 级别可用性 — `python -c "import aiter; print(aiter.logger.level, aiter.logger.handlers)"`

### 需要在执行前补写的脚本

| 专题 | 缺失脚本 | 优先级 | 预计工时 |
|------|---------|--------|---------|
| V01 Exp 1 | `--preshuffle` CLI 注册或临时 driver | P0 | 0.5h |
| V01 Exp 2 | monkey-patch `get_block_size_M` 完整代码 | P0 | 0.3h |
| V02 Exp 1 | stale .so 路径确认 + 清理命令绝对化 | P0 | 0.1h |
| V02 Exp 2 | `D_check_weights.py` 路径确认或重写 | P0 | 0.5h |
| V02 Exp 4 | `ATOM_DISABLE_SWIGLUSTEP_LAYERS` env 加临时开关 | P0 | 1h |
| V03 Exp 1/2 | 完整复现脚本（当前仅描述） | P0 | 1.5h |
| V04 Exp 1 | 单算子 driver（当前伪码） | P0 | 0.5h |
| V06 Exp 1b' | dest.copy_(src) shape mismatch 1 行验证 | P0 | 0.1h |
| V06 cross-check | BF16/FP8 inter_pad + sc13.shape dump | P0 | 0.3h |
| V07 Exp 5.b | 其他 CSV spot-check 完整脚本 | P1 → **P0** | 0.5h |
| V07 Exp 1 | `/tmp/v07_exp1_tgemm.py`（V07 已给完整代码，仅需落地） | P0 | 0.1h |

**合计**：~5.4h 脚本补完工作量。

### 不建议用 Agent Team 的场景

1. **V05 Exp 1 crash 复现**：必须独占 aiter 源码，全局排他锁；其他 V05/V06 agent 必须暂停
2. **V06 Exp 5a/5b gibberish 复现**：必须独占 ATOM moe.py worktree；与 V05 Exp 1 不冲突（不同源文件），但与 V04 修改 ca_comm 冲突
3. **Phase 0 noise floor 实验（V00）**：所有专题阈值的标定输入，必须串行先于所有 Phase 1-3 实验
4. **跨专题问题 1.1（3771835ac diff 调查）**：人工 review 决策点，不适合 agent 自动判断
5. **V02-E（BOS-spam + noise 累积，4-6h）**：单一长任务，agent 拆分无收益；只在调度上提前启动

---

## 6. 最终评分

| 专题 | 逻辑合理性 | Agent 效率 | 验证准确性 | 最终评分 | 是否可执行 |
|------|----------|-----------|---------|---------|----------|
| V01 | B+（实验 4/5 弱） | A（4 路并行可省 37%） | B+（kernelName2 + 主观目检） | **B+** | 补 driver 后执行 |
| V02 | B（Exp 4 不可证伪、Exp 6 应升 P0） | B+（瓶颈 Exp 4+6 4-6h） | B（cos_sim 阈值过严 + ±20% 过宽） | **B** | 补 env 开关 + Exp 6 升级后执行 |
| V03 | A-（结构最干净） | A（合并实验 1+4 + 50% 节省） | A-（cos_sim 区分度 1000×） | **A-** | 补复现脚本后执行 |
| V04 | B+（align=192 证据弱、tp=8 不可测） | B+（A1+A2+A3 三路并行） | B（bit-exact 过严、ca_comm 数值未验） | **B+** | 补单算子 driver + Exp 4-C 升 P0 后执行 |
| V05 | A-（guard 完整，可扩展性弱） | B+（Exp 1 必须独占串行） | B（log level + JIT cache + Exp 5 标准弱） | **B+** | 加源码修改协议后执行 |
| V06 | A-（Fix 1 open question + Exp 5 双变化） | A（B1+B2+B3 三路并行） | A-（缺 gibberish 量化）| **A-** | 补 Exp 1b' + Exp 5 拆分后执行 |
| V07 | A（CSV 实地核查充分） | A（4 阶段并行 67% 节省） | A-（diff<5 缺来源 + first_token 标准弱） | **A** | Exp 6/5.b 升 P0 后执行 |

**Overall Pipeline 评分**：**7.5/10**

具体扣分：
- 逻辑合理性 -1（cross-专题 inter_pad / scale padding 一致性未列入实验）
- Agent 效率 -0.5（Phase 2/3 GPU 调度三方有冲突，需 Synthesizer 强裁决）
- 验证准确性 -1（noise floor 缺失 + gibberish 量化缺失 + 多个 P1 应升 P0）

**建议**：**[补完脚本 + Phase 0 增强后执行]**

理由：
1. 逻辑骨架已可执行，但 N1（noise floor）+ N2（worktree 隔离）+ N3（JIT cache）三个共性问题不解决会污染所有结论
2. V07-Exp6 / V07-Exp5.b / V02-Exp6 必须升 P0，否则覆盖面有结构性缺口
3. ~5.4h 脚本补完 + ~3h Phase 0 增强（含 V00 noise floor）即可启动；总投入 ~16h，比 MASTER 原估 26-32h 节省一半
4. Agent Team 适用：本任务 >50 tool calls、可分解、多假设并发，符合 SKILL 启动条件（参考 `/home/hanchang/agent_skill/.claude/skills/agent-team/SKILL.md`）

---

## 附：三方分歧与 Synthesizer 裁决记录

| 分歧点 | A 立场 | B 立场 | C 立场 | 裁决 |
|--------|--------|--------|--------|------|
| V02-Exp 6 优先级 | 升 P0/P1 边缘 | 未涉及 | 条件 P0 | **采纳：条件 P0**（若 Exp 4 完美则 P1，否则 P0） |
| V06-B2 GPU 分配 | 未涉及 | GPU 0,1,2,3 | GPU 2,3,4,6 | **采纳 B**（连续 4 卡 NCCL 性能更好） |
| V04-Exp 4 三方法选择 | 未涉及 | 必须做方法 C | 未涉及 | **采纳 B**：方法 C 升 P0 |
| V07-Exp 2 时间窗 | 未涉及 | 未涉及 | Phase 3 末段 | **微调 C**：T+60min 而非末段（提前 30min） |
| V08 是否独立专题 | 未涉及 | 未涉及 | 升 P0 或开 V08 | **裁决：先 V07-Exp5.b 升 P0，跑后再决定是否开 V08** |

---

报告完成。
