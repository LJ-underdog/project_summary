# Step-3.5-Flash 验证 Pipeline — 新 Session 快速入门

> 5 分钟看完，立刻能开始执行。详细内容见末尾「文档位置」表。

---

## 1. 你在做什么

一句话：**验证 Step-3.5-Flash 在 gfx950（8x MI350X）上的 7 个修复是否正确、完整、稳定**。

7 个修复分布在两个仓库（ATOM + aiter），覆盖：MoE shuffle / V1→V3 kernel 升级 / SwigluStep 接线 / inter_dim padding / sliding window off-by-one / FP8 scale ceil / FP8 scale loading / ASM kernel CSV / ca_comm fallback。验证目标是产出 7 份 PASS/FAIL 矩阵 + 跨专题一致性结论。

---

## 2. 当前代码状态（验证起点）

| 仓库 | 路径（本地验证用） | 当前 commit | 包含修复 |
|------|------------------|-----------|---------|
| ATOM | `/home/hanchang/ATOM` | `ccb64621` | Fix1: shuffle_weights gfx950 / Fix2: V1→V3 block_m=128 / Fix3: SwigluStep wiring / Fix4: inter_dim padding / Fix5: FP8 scale ceil |
| aiter | `/home/hanchang/aiter` | `c38d0c9e6`（或更新）| Fix1: sliding window off-by-one / Fix2: ca_comm None guard / Fix3: blockscale q_type guard / Fix4: tuned_gemm CSV |
| junlin12_repos/aiter | `/home/hanchang/junlin12_repos/aiter` | `a2883ab37` | Fix4 上游 PR 用途，**验证不要用这个仓库** |

**铁则**：验证执行用 `/home/hanchang/ATOM` 和 `/home/hanchang/aiter`；只有 git commit/push 时才切到 `junlin12_repos/`。

---

## 3. 验证依赖图（不能跳）

```
V01-MoE  (preshuffle gfx950 + V3 kernel)        ← 必须最先；任何 fused_moe 错误污染所有下游
   ├─ V02-SwigluStep    (依赖 V01：activation 走同 dispatch)
   ├─ V03-SlidingWindow (依赖 V01：e2e 用 fused_moe)
   └─ V04-TP Support    (依赖 V01：tp=4 inter_dim padding 走 V3)
         ├─ V05-FP8 Inference (依赖 V04：blockscale guard 与 V01 Fix2 互斥)
         │      └─ V06-FP8 tp=4 (依赖 V04 + V05：FP8 padding + scale shard)
         └─ V07-LongSeq BOS   (依赖 V04：tp=4 通了才能跑 10k；与 V06 可并行)
```

**严格串行点**：V01 必须先全部 PASS。

---

## 4. Phase 0 — 必须先做的 5 件事

| 步骤 | 做什么 | 命令/操作 | 阻断什么 |
|------|--------|----------|---------|
| 0.1 | commit 校验 | `cd /home/hanchang/ATOM && git log -1 --format='%H %s'` 和 aiter 同样 | 所有 |
| 0.2 | aiter 安装路径校验 | `cd /tmp && /opt/venv/bin/python -c "import aiter; print(aiter.__file__)"` 必须指向 `/home/hanchang/aiter/aiter/__init__.py` | 所有 |
| 0.3 | 3771835ac 范围调查 | `cd /home/hanchang/aiter && git show 3771835ac --stat`；若触及 `ATOM/atom/model_ops/moe.py` L489-518，则**阻断 V04** | V01 Exp5 / V04 |
| 0.4 | SLIDING_WINDOW 实测 | `grep -i sliding /root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash/snapshots/ab446a3*/config.json` | V03 |
| 0.5 | V00 noise floor | 固定 seed × 5 次同 op 跑 cos_sim worst-case，标定各专题阈值（**所有 cos_sim 阈值的基线**）| 所有 cos_sim 判断 |

**第一条命令**（一键完成 Phase 0）：

```bash
bash /home/hanchang/project_fp8_tp4/verification_pipeline/phase0_preflight.sh
```

> 该脚本若不存在，先按上面 5 步手动执行；执行结果先汇总成「Phase 0 报告」再进入 Phase 1。

---

## 5. 最快启动路径（只有 2h 时怎么办）

按顺序执行：
1. Phase 0 全部 5 步（30 min）
2. **V01 实验 4 静态 grep**（CPU only，5 min）
3. **V01 实验 1 + 实验 2**（GPU 0,1，60 min）
4. **V01 实验 3 端到端 tp=2**（GPU 0,1，30 min）

V01 全 PASS 后，再决定下一步分发哪些专题。

---

## 6. Agent Team 分工总览

适合 Agent Team（>50 tool calls，可分解，多 GPU 并发）。Skill 在 `/home/hanchang/agent_skill/.claude/skills/agent-team/SKILL.md`。

| 专题 | Agent 数 | GPU 分配 | 关键路径用时 | 备注 |
|------|---------|---------|-------------|------|
| V01-MoE | 5（A/B/C/D/E）| GPU 0,1,2,3 | ~2.5h | E 必须 worktree |
| V02-SwigluStep | 5（A/B/C/D/E）| GPU 6,7（+0,1）| ~7h | E（BOS-spam + noise）是全局瓶颈，V01 PASS 后立即提前 dispatch |
| V03-SlidingWindow | 3（A/B/C）| GPU 4,6 | ~1.5h | 端到端强制 tp=2，避开 V07 BOS bug |
| V04-TP | 4（A1-A4）| GPU 0,1,2,3 + 4 | ~2h | A4 占满 4 卡 NCCL；Exp 4-C 升 P0 |
| V05-FP8 | 4（A/B/C/D）| GPU 0,1,2,3,6 | ~75 min | A（crash 复现）必须独占 aiter 源码 + 最后跑 |
| V06-FP8 tp=4 | 4（B1-B4）| GPU 0,1,2,3 + 4,6 | ~2h | B2 用连续 4 卡；B4 独占 ATOM moe.py worktree |
| V07-LongSeq | 7（A-G）| GPU 0,1,2,3 + 4,6,7 | ~60 min | Exp 6 / Exp 5.b 升 P0 |

GPU 资源：**GPU 0-4, 6, 7 可用，GPU 5 硬件异常禁用**。详细时间窗见 `PIPELINE_REVIEW_FINAL.md §2.1`。

总工时估算：纯串行 26-32h → Agent Team 优化后 ~12h（含 Phase 0 增强后 ~16h）。

---

## 7. 最需要注意的坑（来自 PIPELINE_REVIEW_FINAL）

### N1. cos_sim 阈值无 noise floor 基线
当前阈值 V01=0.9999 / V02=0.99998 / V03=0.999998 跨 3 数量级。**必须先跑 Phase 0.5 V00 实验**取 worst-case，再用 baseline + ε 作判定。否则任何 PASS/FAIL 结论都不可信。

### N2. 修改源码的实验必须 worktree 隔离
涉及实验：V01-Exp5（注释 ATOM padding）/ V02-Exp4（加 env switch）/ V05-Exp1（改 aiter fused_moe）/ V06-Exp5（改 ATOM moe.py）。
强制规则：用 `EnterWorktree`，执行前 `git status --porcelain` 必须为空，结束后 diff 确认还原。

### N3. JIT cache 必须统一清理
每次切换 commit / 修改 kernel 源码后：
```bash
rm -rf /root/.cache/atom/* /root/.cache/aiter/* ~/.aiter_cache/*
```
所有 op_test 实验执行前必须 `AITER_LOG_LEVEL=INFO` 抓取实际命中的 kernel 名（避免 fallback 假性 PASS）。

### 必须从 P1 升 P0 的实验
1. **V07 Exp 6**（tp=2 long-seq negative control）— 唯一直接验证 ASM kernel 影响面
2. **V07 Exp 5.b**（其他 CSV spot-check）— llama70B / llama405B 复发 BOS 的唯一证据
3. **V02 Exp 6**（cos_sim 衰减曲线）— Exp 4「噪声累积」结论的根因证据
4. **V04 Exp 4-C**（数值单元测试 vs torch ref）— A/B 只能证 fallback 触发，C 才证输出正确
5. **V01 Exp 5**（buffer padding canary）— 若 Phase 0.3 显示 ATOM padding 也被 revert 则升 P0

### 其他坑
- **必须 `cd /tmp &&`**：所有 python 命令前缀（aiter namespace package）
- **CUDA_VISIBLE_DEVICES 不能含 5**
- **tp=4 长 prompt ≥10k 会撞 V07 已知 BOS bug**：V01/V03/V05/V06 的 tp=4 实验必须避开此组合
- **V06 Exp 5 必须拆 5a/5b**：ceil→floor + 去 clamp 是两个变化，因果不可分
- **V04 ↔ V06 cross-check**：必须 dump `w13.shape` + `sc13.shape`，验 `inter_pad=384` 一致

---

## 8. 所有文档位置

| 文档 | 路径 |
|------|------|
| 7 个专题验证计划 | `/home/hanchang/project_fp8_tp4/verification_pipeline/V0[1-7]_*.md` |
| 总执行计划（依赖图 + Checklist）| `/home/hanchang/project_summary/step35-flash-support/verification_pipeline/MASTER_PIPELINE.md` |
| Review 最终报告（裁决记录 + 评分）| `/home/hanchang/project_summary/step35-flash-support/verification_pipeline/PIPELINE_REVIEW_FINAL.md` |
| 三份独立 Review | `/home/hanchang/project_fp8_tp4/verification_pipeline/REVIEW_[A-C].md` |
| TEAM_CONFIG（agent team 配置）| `/home/hanchang/project_fp8_tp4/verification_pipeline/TEAM_CONFIG_verification.md`（待生成）|
| Phase 0 预检脚本 | `/home/hanchang/project_fp8_tp4/verification_pipeline/phase0_preflight.sh`（待生成）|
| Agent Team Skill | `/home/hanchang/agent_skill/.claude/skills/agent-team/SKILL.md` |
| GPU 资源 | GPU 0-4, 6, 7 可用；GPU 5 硬件异常禁用 |

---

## 9. 第一条命令

```bash
bash /home/hanchang/project_fp8_tp4/verification_pipeline/phase0_preflight.sh
```

若脚本未生成，按 §4 表格手工执行 0.1-0.5 五步，结果汇总成 Phase 0 报告，再进入 V01。
