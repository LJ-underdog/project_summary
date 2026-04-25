# Team Config: step35-flash-verification

# 继承自：agent_skill/.claude/skills/agent-team/SKILL.md
# 创建日期：2026-04-25
# 状态：[READY]
# 上游计划：project_summary/step35-flash-support/verification_pipeline/MASTER_PIPELINE.md
# 评审报告：project_summary/step35-flash-support/verification_pipeline/PIPELINE_REVIEW_FINAL.md
# 验证范围：V01-MoE / V02-SwigluStep / V03-SlidingWindow / V04-TP / V05-FP8 / V06-FP8 tp=4 / V07-LongSeq BOS

---

## 基础配置

```
PROJECT:    step35-flash-verification
WORK_DIR:   /home/hanchang/project_fp8_tp4/verification_pipeline/
DOC_DIR:    /home/hanchang/project_fp8_tp4/verification_pipeline/results/
LOG_DIR:    /home/hanchang/project_fp8_tp4/verification_pipeline/results/logs/
CODE_ROOTS:
  - /home/hanchang/ATOM/atom/model_ops/moe.py            # MoE forward + padding + shuffle_weights
  - /home/hanchang/ATOM/atom/models/step3p5.py            # attention forward + sliding-window flag
  - /home/hanchang/aiter/aiter/fused_moe.py               # MoE 2-stage dispatch / V1↔V3 切换 / q_type guard
  - /home/hanchang/aiter/aiter/ops/triton/gluon/pa_decode_gluon.py  # decode 路径
  - /home/hanchang/aiter/aiter/tuned_gemm.py              # ASM kernel dispatch
  - /home/hanchang/aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv  # 已删 buggy 行
  # 辅助：
  - /home/hanchang/aiter/aiter/dist/parallel_state.py
  - /home/hanchang/aiter/op_tests/test_moe_2stage.py
  - /home/hanchang/ATOM/atom/examples/simple_inference.py
  - /home/hanchang/project_fp8_tp4/logs/perf_compare_10k/run_inference.py
GOAL:       验证 Step-3.5-Flash 在 gfx950 上的 7 个修复（V01-V07）是否正确、完整、稳定；
            目标产出：每专题 PASS/FAIL 矩阵 + 跨专题 1.1/1.3/1.7 实验答案 + 新发现的 open bug 登记
```

---

## ENVIRONMENT

```
# python 前置（必须）
cd /tmp && /opt/venv/bin/python ...
原因：aiter 在 cwd != /tmp 时被识别为 namespace package，import 路径错乱

# GPU 分配
GPU 5 硬件异常禁用（~700ms/tensor）
可用：GPU 0,1,2,3,4,6,7（共 7 张）

# 各配置对应 CUDA_VISIBLE_DEVICES
tp=2:           CUDA_VISIBLE_DEVICES=0,1
tp=4:           CUDA_VISIBLE_DEVICES=0,1,2,3
单 GPU 测试:    CUDA_VISIBLE_DEVICES=0
（tp=8 因 GPU5 异常不可端到端测，仅静态 + 单算子降级）

# 缓存清理（任何代码改动后必须）
rm -rf /root/.cache/atom/* /root/.cache/aiter/* ~/.aiter_cache/*
（注意：仅清 atom cache 不够；REVIEW C-1.3.2 已确认 V05-Exp1 旧脚本漏清 aiter cache）

# 路径
ATOM:               /home/hanchang/ATOM
aiter:              /home/hanchang/aiter
PR 仓库:            /home/hanchang/junlin12_repos/{aiter,atom}（commit/push 必须从此处）
验证结果日志:        /home/hanchang/project_fp8_tp4/verification_pipeline/results/
op_tests:           /home/hanchang/aiter/op_tests/
长序列 driver:       /home/hanchang/project_fp8_tp4/logs/perf_compare_10k/run_inference.py

# 模型路径
BF16: stepfun-ai/Step-3.5-Flash
FP8:  stepfun-ai/Step-3.5-Flash-FP8
HF cache: /root/.cache/huggingface/hub/
BF16 snapshot: /root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash/snapshots/ab446a3.../

# Logger 级别（验证 kernel 命中必须）
AITER_LOG_LEVEL=INFO  # 抓 dispatch 日志（V01 Exp 2 / V05 Exp 5 必需）
AITER_LOG_LEVEL=WARNING  # 端到端推理默认（不被日志淹没）

# 常用推理命令（tp=2 BF16）
cd /tmp && rm -rf /root/.cache/atom/* && CUDA_VISIBLE_DEVICES=0,1 \
  AITER_LOG_LEVEL=WARNING \
  /opt/venv/bin/python -m atom.examples.simple_inference \
  --model stepfun-ai/Step-3.5-Flash --kv_cache_dtype bf16 --trust-remote-code \
  --tensor-parallel-size 2 --level 0 --temperature 0 --max-tokens 128 \
  --max-num-batched-tokens 4096 --max-num-seqs 2048

# 常用推理命令（tp=4 FP8）
cd /tmp && rm -rf /root/.cache/atom/* && CUDA_VISIBLE_DEVICES=0,1,2,3 \
  AITER_LOG_LEVEL=WARNING \
  /opt/venv/bin/python -m atom.examples.simple_inference \
  --model stepfun-ai/Step-3.5-Flash-FP8 --trust-remote-code \
  --tensor-parallel-size 4 --level 0 --temperature 0 --max-tokens 128 \
  --max-num-batched-tokens 4096 --max-num-seqs 2048

# 长序列 driver（10k tokens）
MODEL="stepfun-ai/Step-3.5-Flash" TP=4 GMU=0.7 MAX_TOKENS=10 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 AITER_LOG_LEVEL=WARNING \
  /opt/venv/bin/python /home/hanchang/project_fp8_tp4/logs/perf_compare_10k/run_inference.py
（硬编码：--enforce-eager L29、--max-num-batched-tokens 16384、--max-num-seqs 4）

# 长时运行
nohup sh -c '...' > {LOG_DIR}/xxx.log 2>&1 &
while kill -0 $PID 2>/dev/null; do sleep 20; echo "running..."; done
```

---

## CONSTRAINTS

- 不能修改 `@support_torch_compile` 装饰的模型文件（破坏 Dynamo tracing）
- 任何修改 ATOM/aiter 源码的实验必须遵循「源码修改协议」（见末节）
- 端到端实验前必须 `rm -rf /root/.cache/atom/* /root/.cache/aiter/* ~/.aiter_cache/*`
- 不得用 GPU 5（硬件异常）
- 长序列实验（≥10k）只在 V07 跑，V01/V03/V05/V06 的 tp=4 端到端实验 prompt 长度 < 10k（避免被 V07 BOS bug 污染）
- V05 Exp 1（aiter 源码 crash 复现）+ V06 Exp 5a/5b（ATOM moe.py gibberish 复现）必须串行 + worktree 独占
- 执行型 item 必须等 lead 明确批准
- git push 从 `/home/hanchang/junlin12_repos/{aiter,atom}` 操作（author: Jun Lin <junlin12@amd.com>，SSH key id_ed25519_junlin12）
- 不得在主仓库 `/home/hanchang/aiter` 或 `/home/hanchang/ATOM` 直接 push
- 输出文本只能中文或英文，禁止其他语言

---

## KNOWN_FACTS（已验证，无需重验）

| # | 事实 | 来源 |
|---|------|------|
| F1  | BF16 tp=2 端到端正常：TTFT=92ms, TPOT=17ms, gmu=0.9 | MEMORY.md |
| F2  | BF16 tp=4 端到端正常（短 prompt）：TTFT=88ms, TPOT=15.75ms, gmu=0.7 | MEMORY.md |
| F3  | FP8 tp=2 端到端正常：TTFT=85ms, TPOT=13.5ms, gmu=0.7 | MEMORY.md |
| F4  | FP8 tp=4 端到端正常（短 prompt）：TTFT=93ms, TPOT=12.75ms, gmu=0.7 | MEMORY.md |
| F5  | FP8 decode TPOT 比 BF16 同 tp=4 快 19% | fp8-work.md |
| F6  | tp=8 端到端不可测（GPU 5 硬件异常） | MEMORY.md |
| F7  | tp=4 + ≥10k tokens → 输出全 BOS（V07 bug 主目标，已通过删 csv 修复） | tp48-fixes.md |
| F8  | aiter commit `c38d0c9e6` = FP8 blockscale 修复（含 V05 Fix 1 q_type guard） | MEMORY.md |
| F9  | aiter commit `a2883ab37` = 删 glm5_bf16_tuned_gemm.csv L45 buggy ASM kernel | TEAM_CONFIG_longseq_debug.md |
| F10 | aiter commit `7ebae9afb` = sliding window decode off-by-one 修复 | moe-kernels.md |
| F11 | aiter commit `7312ea166` = 分布式修复 | MEMORY.md |
| F12 | aiter commit `68fc7d48b` = MoE V1→V3 fix（block_m=128，inter_dim≥256） | MEMORY.md |
| F13 | ATOM commit `ec8cbe8` = preshuffle_on 在 gfx950 始终执行 | MEMORY.md |
| F14 | ATOM commit `4a8495e` = SwigluStep wiring | MEMORY.md |
| F15 | ATOM commit `635e59e` = tp4/8 inter_dim padding | MEMORY.md |
| F16 | ATOM commit `ccb64621` = FP8 tp=4 修复 | MEMORY.md |
| F17 | gfx950 上 shuffle_weights() 始终执行（preshuffle_on 路径） | ATOM CLAUDE.md |
| F18 | Step-3.5-Flash 约 3/4 层 sliding window | step35flash_debug.md |
| F19 | aiter namespace package 问题：python 前必须 cd /tmp | MEMORY.md |
| F20 | run_inference.py 硬编码 --enforce-eager + --max-num-batched-tokens 16384 | TEAM_CONFIG_longseq_debug.md |

---

## BASELINE（验证通过时的标准）

各专题通过门槛（详细见 PIPELINE_REVIEW_FINAL §3.3 的标准调整）：

| 专题 | 关键 PASS 标准 |
|------|-------------|
| V01 Exp 1 | preshuffle_off cos_sim < 0.01；preshuffle_on cos_sim ≥ 0.9999 |
| V01 Exp 2 | inter=192/V1 PASS；inter=256/V1 FAIL；其余 PASS（cos_sim ≥ 0.9999）；**实际命中 kernel 名记入 dispatch_log** |
| V01 Exp 3 | exit 0 + len(set(token_ids[1:10])) ≥ 5 + `<s>` 计数 ≤ 1 + BLEU ≥ 0.85 vs BF16 ref |
| V02 Exp 1 | cos_sim ≥ 0.99998（待 V00 noise floor 后定）；max_abs_err ≤ 0.05 |
| V02 Exp 2 | scale ≤ 5.0 cos_sim ≥ 0.99998；scale=8.0 cos_sim ≥ 0.9999 |
| V02 Exp 3 | 4 prompt 全部合理 + repetition rate < 0.3 + diversity > 0.5 + TTFT/TPOT ±10% |
| V02 Exp 4 | spam 命中分布卡方检验 p < 0.05（不是 ±20%） |
| V02 Exp 6 | dcos/dt < 1e-4/step（条件 P0）|
| V03 Exp 1 | ctx=512 修复前 0.998982 / 修复后 ≥ 0.999998；ctx ≥ 513 修复后 ≥ 0.999998 |
| V03 Exp 2 | 修复后能检出特征 KV 贡献；output_norm_diff > σ |
| V03 Exp 3 | tp=2 跑（避开 V07 bug）；无 "ungi" 乱码；token-level repetition < 阈值 |
| V04 Exp 3 | cos_sim ≥ 0.99999 + token-id 序列一致（**放宽 from bit-exact**） |
| V04 Exp 4-C | 数值单元测试 vs torch ref（**升 P0**） |
| V05 Exp 2 | 4 prompts EOS + TTFT < 150ms + TPOT < 25ms + 无 gibberish |
| V05 Exp 3 | cos_sim 或 token-id 一致（**放宽 from byte-level**） |
| V05 Exp 5 | 配 `AITER_LOG_LEVEL=INFO` 显示 block_shape 实际下游值（不只看输出无差异） |
| V06 Exp 2 | 输出连贯 + TTFT < 200ms + TPOT < 20ms + diversity > 0.5 + max n-gram repetition < 0.3 |
| V06 Exp 3 | TPOT_fp8 < TPOT_bf16 × 0.90 作 gate；19% ±5% 作 informational |
| V06 Exp 5a/5b | gibberish: diversity < 0.05 + max n-gram repetition > 0.8 |
| V07 Exp 1 | strict diff < 1；loose diff < 50（buggy 197+ 区分）|
| V07 Exp 2 | first_token == 3648（已知值）+ len(set(token_ids)) ≥ 5 + 0 not in token_ids[1:] |
| V07 Exp 6 | tp=2 长序列正常（**升 P0**）|

---

## 验证顺序（依赖图）

```
V00 (Phase 0 noise floor) ─── 强制阻塞所有阈值依赖
   │
V01 (MoE，必须最先) ─────── Phase 1
  ├─ V02 (SwigluStep)        ┐
  ├─ V03 (SlidingWindow)     ├── Phase 2 三专题并行
  └─ V04 (TP)                ┘
        ├─ V05 (FP8)         ┐
        ├─ V06 (FP8 tp=4)    ├── Phase 3 三专题并行
        └─ V07 (LongSeq BOS) ┘
```

**关键串行点**：
- V00 noise floor → 所有 cos_sim 阈值
- V01 → 其他所有专题（fused_moe 路径污染所有 e2e）
- V04 → V05/V06/V07（tp 框架是它们的前置）
- V05 Exp 1 / V06 Exp 5a/5b（独占源码 worktree）必须末尾跑

---

## Phase 0 TODO List

```markdown
- [ ] **0.1** 确认 aiter 工作树 commit ≥ c38d0c9e6
      `cd /home/hanchang/aiter && git log -1 --format='%H %s'`
- [ ] **0.2** 确认 ATOM 工作树 commit ≥ ccb64621
      `cd /home/hanchang/ATOM && git log -1 --format='%H %s'`
- [ ] **0.3** 确认 aiter 安装路径
      `cd /tmp && /opt/venv/bin/python -c "import aiter; print(aiter.__file__)"`
      期望：`/home/hanchang/aiter/aiter/__init__.py`
- [ ] **0.4a** `cd /home/hanchang/aiter && git show 3771835ac --stat`
- [ ] **0.4b** 看 aiter diff 完整内容
- [ ] **0.4c** 看 ATOM 侧是否同 commit 触及 moe.py L489-518
- [ ] **0.4d** 若 0.4c=YES → V01 Exp 5 升 P0，V04 padding 验证基础需重新论证
- [ ] **0.5a** `ls ~/.cache/huggingface/hub/ | grep -i flash`（BF16 + FP8 都要有）
- [ ] **0.5b** FP8 模型 dry-run 加载
- [ ] **0.5c** BF16 模型 dry-run 加载
- [ ] **0.6** Step-3.5-Flash SLIDING_WINDOW 实测：
      `grep -i sliding /root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash/snapshots/ab446a3.../config.json`
- [ ] **0.7** GPU 可用性：`rocm-smi`，确认 0,1,2,3,4,6,7 正常，5 异常标注
- [ ] **0.8** JIT cache 健康检查：
      `ls /root/.cache/aiter/* | wc -l` + 测试统一清理命令
      `rm -rf /root/.cache/atom/* /root/.cache/aiter/* ~/.aiter_cache/*`
- [ ] **0.9** V00 全局 noise floor（**关键阻塞**）：
      固定 seed × 5 次同 op 跑 cos_sim 取 worst，作为各 op 阈值标定
      覆盖 op：fused_moe / SwigluStep / paged_decode
      产出：`results/v00_noise_floor.md`
- [ ] **0.10** 源码修改协议落地：
      检查 `EnterWorktree` 工具可用；准备 worktree 模板路径
      `/home/hanchang/junlin12_repos/aiter` 与 `/home/hanchang/junlin12_repos/atom` 各开一个 worktree
- [ ] **0.11** aiter logger INFO 级别可用性：
      `cd /tmp && /opt/venv/bin/python -c "import aiter; print(aiter.logger.level, aiter.logger.handlers)"`
```

预计 Phase 0 工时：3-4h。

---

## Phase 1-3 TODO List（按 MASTER_PIPELINE §6 改写）

### Phase 1 — V01 MoE（必须最先，串行）

```markdown
- [ ] **1.1** V01-A：实验 4 静态 grep（CPU，无依赖）
- [ ] **1.2** V01-B：实验 1 preshuffle on/off 对比（GPU 0,1）
- [ ] **1.3** V01-C：实验 2.a-2.e + 边界细化 200/224/240（GPU 0,1）
- [ ] **1.4** V01-D：实验 3 端到端 tp=2 + tp=4 BF16（GPU 0,1,2,3，依赖 1.2/1.3 PASS）
- [ ] **1.5** V01-E：实验 5 buffer padding canary（worktree + GPU 0,1，条件 P0/P1，依赖 0.4d）
```

### Phase 2 — V02 / V03 / V04 并行（V01 PASS 后）

```markdown
V02:
- [ ] **2.1** V02-A：实验 1 op_test SwigluStep 7×2×3=42 case（GPU 7）
- [ ] **2.2** V02-C：实验 5 三方法 wiring 透传（GPU 7）
- [ ] **2.3** V02-B：实验 2 layer×M×scale=2×4×4=32 case（GPU 6）
- [ ] **2.4** V02-D：实验 3 + 实验 7 共享 driver（GPU 0,1，依赖 2.1/2.2 PASS）
- [ ] **2.5** V02-E：实验 4 + 实验 6 强制配对（GPU 7，关键路径瓶颈 4-6h，依赖 2.4 PASS；可与 V03/V04 并发）
      注：实验 4 需先加 ATOM_DISABLE_SWIGLUSTEP_LAYERS env 开关（脚本工时 1h）

V03:
- [ ] **3.1** V03-A：W 数值确认 + 实验 1+4 合并 ctx sweep（GPU 6）
- [ ] **3.2** V03-B：实验 2 decode 专项（GPU 6，依赖 3.1）
- [ ] **3.3** V03-C：实验 3 buggy/fixed/reference 三方对照（GPU 4，依赖 3.1）

V04:
- [ ] **4.1** V04-A1：CK manifest grep + ca_comm grep（CPU）
- [ ] **4.2** V04-A2：实验 3 tp=2 BF16 回归（GPU 4）
- [ ] **4.3** V04-A3：实验 1 inter sweep + 实验 5.3（GPU 6,7，依赖 4.1）
- [ ] **4.4** V04-A4：实验 2 tp=4 + 实验 4-B/4-C（GPU 0,1,2,3，依赖 4.2 释放）
      注：4-C（数值单元测试 vs torch ref）升 P0
```

### Phase 3 — V05 / V06 / V07 并行（V04 PASS 后）

```markdown
V05:
- [ ] **5.1** V05-D 模型预检（GPU 6）+ V05-B 实验 2 FP8 tp=2（GPU 0,1）
- [ ] **5.2** V05-C：实验 3 BF16 tp=2 回归（GPU 2,3）
- [ ] **5.3** V05-D 转：实验 5 block_shape 区分性（GPU 6，配 AITER_LOG_LEVEL=INFO）
- [ ] **5.4** V05-A：实验 1 crash 复现（GPU 0,1，**独占 aiter 源码 worktree**，必须 Phase 3 末段）

V06:
- [ ] **6.1** V06-B1：Exp 1b/1c offline cover-completeness（CPU/GPU 0）
- [ ] **6.2** V06-B3：Exp 3 perf + Exp 4 tp=2 回归（GPU 4,6）
- [ ] **6.3** V06-B2：Exp 1a + Exp 2 端到端 + scale dump（GPU 0,1,2,3，**worktree 隔离**）
- [ ] **6.4** V06-B4：Exp 5a + 5b gibberish 复现（GPU 0,1，**独占 ATOM moe.py worktree**，依赖 6.3）
      注：5 拆 5a（只 revert ceil 保留 clamp）+ 5b（只 revert clamp 保留 ceil）
- [ ] **6.5** V06 cross-check：BF16/FP8 inter_pad + sc13.shape dump（联合问题 1.7+L8）

V07:
- [ ] **7.1** V07-A：Exp 1 tgemm 直调 13 个 M（GPU 7，T+0）
- [ ] **7.2** V07-B：Exp 5.a CSV 扫描（CPU，T+0）
- [ ] **7.3** V07-C：Exp 2 E2E 10k tp=4（GPU 0,1,2,3，T+0）
- [ ] **7.4** V07-D：Exp 6 tp=2 negative control（GPU 4,6，**升 P0**，T+0）
- [ ] **7.5** V07-E：Exp 3 短 prompt 回归（GPU 0,1,2,3，T+30min）
- [ ] **7.6** V07-F：Exp 4 perf bench（GPU 7，T+30min）
- [ ] **7.7** V07-G：Exp 5.b 其他 CSV spot-check（GPU 7，**升 P0/V08**，T+45min）
```

### Phase 4 — 总结

```markdown
- [ ] **8.1** 汇总每个专题 PASS/FAIL 矩阵 → results/SUMMARY.md
- [ ] **8.2** 跨专题问题 1.1/1.3/1.7 实验答案
- [ ] **8.3** 提交 V07 §C.3 ASM kernel issue 给 AMD aiter 团队
- [ ] **8.4** 评估是否开 V08（llama70B/llama405B + ASM kernel）
- [ ] **8.5** 更新 MEMORY.md
```

---

## Agent 分工方案（Phase 2 + Phase 3）

引用自 PIPELINE_REVIEW_FINAL §2.1，已采纳 Synthesizer 裁决。

### Phase 2 GPU 分配

```
时间窗 T+0~30min:
  GPU 0,1,2,3   → V04-A4 (tp=4 端到端 + ca_comm)        关键路径
  GPU 4         → V04-A2 (tp=2 回归)
  GPU 6         → V03-A   (W 验证 + ctx sweep)
  GPU 7         → V02-A   (op_test SwigluStep)
  CPU only      → V04-A1  (CK manifest grep)
  CPU only      → V03 W 数值确认

时间窗 T+30~60min:
  GPU 0,1,2,3   → V04 inter sweep + V04 实验 5 单算子降级
  GPU 4,6       → V03-B (decode) + V03-C (端到端 tp=2)
  GPU 7         → V02-B (layer 43-44)

时间窗 T+60~90min:
  GPU 4,6       → V02-D (端到端 + shared 反向)
  GPU 7         → V02-E (BOS-spam + noise 累积，瓶颈 4-6h)
```

### Phase 3 GPU 分配

```
时间窗 T+0~30min:
  GPU 0,1       → V05-B  (FP8 tp=2 端到端)
  GPU 2,3       → V05-C  (BF16 tp=2 回归)
  GPU 4,6       → V06-B3 (FP8 tp=2 + perf)
  GPU 7         → V07-A (tgemm 直调) + V05-D (模型预检)
  CPU only      → V07-B (CSV 扫描)

时间窗 T+30~60min:
  GPU 0,1,2,3   → V06-B2 (FP8 tp=4 端到端 + scale dump)   关键路径
  GPU 4         → V07-D (tp=2 negative control，10k)      升 P0
  GPU 6         → V05-D 转 Exp 5 (block_shape 验证)
  GPU 7         → V07-F (perf bench)

时间窗 T+60~90min:
  GPU 0,1,2,3   → V07-C (E2E 10k tokens)
  GPU 6         → V06-B4 (gibberish 复现 5a/5b)            独占 worktree
  GPU 7         → V07-G (其他 CSV spot-check) 升 P0/V08

时间窗 T+90~120min:
  GPU 0,1       → V05-A (Exp 1 crash 复现，独占 aiter 源码) 必须最后
```

裁决：
- V06-B2 用 GPU 0,1,2,3（连续 4 卡 NCCL 性能更好；GPU 4 留给 V07-D negative control）
- V07 Exp 2 在 T+60min（提前 30min）

---

## Teammate Prompt 模板

每个 teammate session 启动时，lead 应按以下结构发 prompt：

```
你是 step35-flash-verification 的 [{TEAMMATE_ID}]，负责 [{专题/Exp 编号}]。

## 环境（必读，不可省略）
- 工作目录：先 `cd /tmp` 再跑 python（aiter namespace package 问题）
- GPU：CUDA_VISIBLE_DEVICES={{X,Y,...}}（不得用 5）
- Logger：AITER_LOG_LEVEL={{INFO|WARNING}}
- 缓存清理（代码改动后）：
  rm -rf /root/.cache/atom/* /root/.cache/aiter/* ~/.aiter_cache/*
- 日志输出：tee 到 /home/hanchang/project_fp8_tp4/verification_pipeline/results/logs/{{exp}}.log

## 你的任务
[{Exp 描述，含完整命令、矩阵、迭代条件}]

## 已知事实（无需重验，直接采信）
- 引用 KNOWN_FACTS § F1-F20 中相关条目
- 上游 PASS 结果：{{V01-D PASS / V04-A2 PASS / ...}}

## 通过标准（量化）
- 引用 BASELINE 表中本 Exp 对应行
- 关键阈值：{{cos_sim ≥ 0.9999 / TPOT < 25ms / first_token == 3648}}
- 失败处理：FAIL 立即停下并回报，不要尝试自行修源码

## 记录格式（必须）
results/{{专题}}_{{Exp}}.md，包含：
1. 命令完整 dump（含 env vars + CUDA_VISIBLE_DEVICES）
2. 实际命中 kernel 名（grep AITER_LOG_LEVEL=INFO 输出）
3. 量化指标（cos_sim/TPOT/diversity/...）
4. 通过/失败结论 + 与 BASELINE 对比
5. 任何意外 warning / dispatch 路径偏移

## 源码修改（若涉及）
强制走「源码修改协议」（见 TEAM_CONFIG 末节），任何遗漏直接判 FAIL。

## Context 自检
< 15 tool calls 正常；接近 20 立即收尾、写中间结果回报 lead。
```

---

## 源码修改协议（必读，强制）

任何涉及修改 `/home/hanchang/ATOM` 或 `/home/hanchang/aiter` 源码的实验
（V01-Exp5 / V02-Exp4 / V05-Exp1 / V06-Exp1a / V06-Exp5a / V06-Exp5b / V04-Exp4-C 等）
必须严格执行：

1. **执行前检查**
   ```bash
   cd /home/hanchang/{aiter|ATOM}
   git status --porcelain  # 必须为空，否则中止
   ```

2. **备份 diff（即使为空）**
   ```bash
   git diff > /tmp/before_patch_{exp_id}.diff
   ```

3. **使用 Edit 工具修改**（禁用 sed/awk/echo >>）
   - 必须先 Read 该文件
   - 修改范围最小化，注释新增行 `# [V0X-EXP-Y revertable patch]`

4. **实验结束立即还原**
   ```bash
   cd /home/hanchang/{aiter|ATOM}
   git checkout -- {modified_file}
   git status --porcelain  # 必须再次为空
   git diff                # 必须为空
   ```

5. **更优做法（推荐）**：用 `EnterWorktree` 在隔离 worktree 中操作
   - 首选目标路径：`/home/hanchang/junlin12_repos/{aiter,atom}`
   - 实验完成后 ExitWorktree

6. **互斥锁**
   - V05 Exp 1（aiter fused_moe.py）与 V06 Exp 1a/5a/5b（ATOM moe.py）不冲突 → 可并行
   - V05 Exp 1 与 V04 修改 ca_comm 冲突 → 必须串行
   - V06 Exp 5a 与 5b 必须串行（同文件不同 patch）

7. **JIT cache 一致性**
   - 改 aiter 源码后必须 `rm -rf /root/.cache/aiter/* ~/.aiter_cache/*`
   - 改 ATOM 后必须 `rm -rf /root/.cache/atom/*`
   - 还原后再清一次

违反协议任何一条 → 实验结果作废，重跑。

---

## 不建议用 Agent Team 的场景（参考 PIPELINE_REVIEW_FINAL §5）

1. **V05 Exp 1 crash 复现**：必须独占 aiter 源码，全局排他锁
2. **V06 Exp 5a/5b gibberish 复现**：必须独占 ATOM moe.py worktree
3. **Phase 0 V00 noise floor**：所有阈值标定输入，必须串行先于所有 Phase 1-3
4. **跨专题问题 1.1（3771835ac diff 调查）**：人工 review 决策点
5. **V02-E（BOS-spam + noise 累积，4-6h）**：单一长任务，agent 拆分无收益；只在调度上提前启动

---

## Promotion Candidates

（执行过程中追加：值得抽象成通用 SKILL/CONSTRAINT 的发现）

- [ ] 待填
