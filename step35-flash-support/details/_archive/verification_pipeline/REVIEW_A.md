# Reviewer-A 审查报告：V01 / V02 / V03

> 审查角色：Reviewer-A，专注 kernel 正确性 / cos_sim 类实验
> 审查日期：2026-04-25
> 审查范围：`V01_moe.md` / `V02_swiglu.md` / `V03_sliding_window.md`

---

## 总体评估

| 专题 | 逻辑合理性 | 效率优化空间 | 准确性风险 | 综合评分 |
|------|----------|------------|---------|---------|
| V01-MoE         | B+（实验 4/5 偏弱） | 高（实验 1+2 可批量并行） | 中（阈值合理，但 kernelName2 未直接验证） | **B+** |
| V02-SwigluStep  | B（实验 4 假说不可证伪、实验 6 工程量未量化） | 中（实验 1/2/5/7 可并行） | 中-高（cos_sim 阈值过严 + BOS-spam 通过标准 ±20% 太宽） | **B**  |
| V03-SlidingWindow | A-（结构最干净、边界论证最充分） | 高（实验 1+4 可合并到一个 sweep） | 低（cos_sim 0.999998 阈值合理；buggy 复现明确） | **A-** |

---

## V01-MoE 审查

### 逻辑问题

**L1. 实验 1 缺少「反例对照」组，无法独立证明 Fix 1 必要性**
- 实验 1 仅复现 preshuffle_off vs on 的 cos_sim 差异，但未在「保留 gfx950 skip」与「移除 skip」两种 ATOM 版本下重跑端到端，逻辑上只能证明 *kernel 路径差异*，不能直接证明 *ATOM commit `ec8cbe8` 的必要性*。建议：在实验 5 之外，再补一个「ATOM 侧把 `if get_gfx() == "gfx950": pass` 临时还原 + 跑实验 3」的反例，否则修复必要性靠间接推理。

**L2. 实验 2 的边界覆盖不对称**
- 表格只测了 192（PASS）/ 256（FAIL/PASS），缺少 192–256 之间细粒度边界（例如 200/224/240）。如果真实边界并非 192 而是 224，结论"inter_dim>192 时触发 V1 bug"会被误固化。建议加 inter_dim ∈ {200, 224, 240} 三个 V1 路径 case，明确「PASS 上界」。

**L3. 实验 2 未验证 kernelName2 真的命中**
- E.2 已经登记了"kernelName2 是否编译进 CK JIT cache"的疑问，但实验 2.c 只看 cos_sim，无法区分「V3 kernel 真的被调用」vs「fallback 到别的路径但碰巧 PASS」。必须在 2.c/2.d/2.e 的执行步骤里强制 `AITER_LOG_LEVEL=INFO` 并 grep 实际命中的 kernel 名称。

**L4. 实验 4 是「代码静态核查」而非真正的回归验证**
- gfx942 不可用情况下降级到 grep，但 grep 只能证明 *显式 gfx950 守卫不存在*，不能证明 shuffle_weights 在其它架构上不引入数值差异。应在 D 节标注此实验为「弱保证」，不能作为 P0。

**L5. 实验 5 修改工作树但缺少 `git stash` / 隔离机制**
- 直接修改 `/home/hanchang/ATOM/atom/model_ops/moe.py` 注释 L504-L518 是高风险动作。一旦后续实验 abort，未还原会污染主仓库。强烈建议改用 `EnterWorktree` 或在 `/home/hanchang/junlin12_repos/` 的隔离 clone 中执行。

### Agent Team 分工建议

| Agent | 负责实验 | 预计时长 | 依赖 |
|-------|---------|----------|------|
| **V01-A**（kernel 静态核查） | 实验 4（grep + git diff） | 15 min | 无（先做） |
| **V01-B**（preshuffle 对照） | 实验 1 (off + on，2 个 case) | 30 min | 无 |
| **V01-C**（V1/V3 边界 sweep） | 实验 2.a–2.e（5 个 case） + 补加 200/224/240 边界 | 60–90 min | 无 |
| **V01-D**（端到端） | 实验 3（tp=2 + tp=4，2 个 case） | 40 min | V01-B + V01-C 全 PASS |
| **V01-E**（canary） | 实验 5 | 30 min | V01-D PASS |

**节省时间估算**：串行执行 ~4h；3 路并行（A+B+C）+ D + E ≈ 2.5h，节省 ~37%。

**批量打包建议**：实验 2.a-2.e + 实验 1 都基于同一个 `test_moe_2stage.py` driver，建议 V01-B 和 V01-C 合并为单 agent 跑一个参数化 driver（pytest parametrize 风格），共用 driver 编写成本。

### 准确性风险

| 风险 | 严重度 | 说明 | 修正建议 |
|------|--------|------|----------|
| cos_sim ≥ 0.9999 阈值对 BF16 MoE 是否过严 | 中 | 文档未给"BF16 MoE 上限"基线。若上限本来就只能到 0.9998，实验 2.d/2.e 会假性 FAIL | 实验 1 之前先跑一次「同输入、不同 seed」获取 cos_sim 噪声基线 |
| `cos_sim < 0.01` 当作"FAIL"判定 | 低 | -0.006 与 0.005 差异巨大但都满足 abs ≤ 0.01；正确判定 | 保留 |
| 实验 3 "人工目检无乱码" 主观 | 中 | 不同 reviewer 对"乱码"标准不同 | 量化：(1) 输出 token 多样性 entropy ≥ 阈值；(2) `<s>` token 数 ≤ 1；(3) 与 BF16 reference 的 BLEU ≥ 0.85 |
| kernelName2 fallback 可能假性 PASS | 高 | 见 L3 | 强制日志验证 |
| 实验 5 padding revert 范围模糊 | 高 | E.3 已登记，但实验 5 命令并未指明到底 revert ATOM padding 还是 aiter buffer padding | 在实验 5 标题下明确「revert 对象 = ATOM moe.py L504-L518」并把 aiter 侧 buffer padding 的 canary 拆为独立子实验 |

---

## V02-SwigluStep 审查

### 逻辑问题

**L1. 实验 4 的"BOS-spam 命中率"假说不可证伪**
- 表格用 "≈3/12 / ≈6/12 / ≈9/12" 配合 "±20%" 容差，等价于命中率范围 50%~90%，几乎任何随机扰动都满足。需要做卡方检验（C0 vs C3 spam 计数），p < 0.05 才算单调相关。

**L2. 实验 6 列为可选，但实验 4 的"噪声累积"结论强烈依赖它**
- 实验 4 通过即声称"bf16 噪声累积"；但 A.3 自己承认 *仅靠效应叠加无法区分噪声累积 vs outlier kernel bug*。实验 6 才是定量证据。逻辑上实验 6 应升级为 **必做**，否则实验 4 通过不能反推根因。

**L3. 实验 5 的"weight check 脚本"路径未给出**
- 提到"复用 summary 提到的 `D_check_weights.py`"但未指定文件位置，可能不存在。需在执行前先 `Glob` 确认或写明 fallback。

**L4. 实验 7 反向实验的失败语义不闭合**
- 通过标准 4.b 写"若反向实验输出仍正常 → 决策可能过度防御，但**不必移除**"。这等于"无论反向实验结果如何都 PASS"，等于不验证。应改为：「反向实验必须在某个量化指标（如 cos_sim 或 BLEU）上明确劣化 ≥ X%；若无劣化，则将 `_fuse_shared_at_layer` 标记为 *未必要的防御* 并降级到独立专题再评估」。

**L5. 实验 1 的清理 `.so` 命令带通配符且路径不完整**
- `rm -f aiter/jit/module_moe_ck2stages_*swiglustep*.so` 是相对路径，且 cwd 由 `cd /tmp` 变到 /tmp 后会失效。需改为绝对路径 `rm -f /home/hanchang/aiter/aiter/jit/module_moe_ck2stages_*swiglustep*.so`。

### Agent Team 分工建议

| Agent | 负责实验 | 预计时长 | 依赖 |
|-------|---------|----------|------|
| **V02-A**（kernel op_test） | 实验 1（M × preshuffle × seed = 7×2×3 = 42 case，可参数化） | 60 min | V01 PASS |
| **V02-B**（层级精度） | 实验 2（layer × M × scale = 2×4×4 = 32 case） | 90 min | V01 PASS |
| **V02-C**（wiring 透传） | 实验 5 三种方法 | 30 min | V01 PASS |
| **V02-D**（短输出 e2e + shared 反向） | 实验 3 + 实验 7（共享同一个 driver，仅 env 不同） | 60 min | V02-A + V02-C PASS |
| **V02-E**（BOS-spam 复现 + 噪声累积量化） | 实验 4 + 实验 6（强制配对） | 4–6h | V02-D PASS |

**节省时间估算**：串行 ~10h；A/B/C 并行 + D + E = ~7h，节省 ~30%。

**批量打包**：实验 1 与实验 2 共用 op-test 框架，可由同一 agent 用一个 parametrize 文件跑完，但矩阵规模总计 74 case，单 agent 时长可能 >2h，分两个 agent 更稳妥。

### 准确性风险

| 风险 | 严重度 | 说明 | 修正建议 |
|------|--------|------|----------|
| cos_sim ≥ 0.99998 阈值过严 | 高 | 比 V01 的 0.9999 严两个数量级；SwigluStep 含 clamp（不可微，离散点会引入额外噪声）。E.5 已自承"未验证 0.999989 是否为 bf16 上限" | 必须在实验 1 之前做 baseline noise floor（同 seed 跑 5 次取最差），再据此设定阈值 |
| BOS-spam 通过标准 ±20% 太宽 | 高 | 见 L1 | 用统计检验代替百分比 |
| 实验 2 scale=8.0 阈值放宽到 0.9999 主观 | 中 | 没有论证为什么是 0.9999 而不是 0.999 | 同样以 noise floor 为基线，给定 ±N×σ |
| 实验 6 缺少明确通过/失败界 | 中 | 只说"导数有上界"未给数值 | 给出 dcos/dt < 1e-4/step 这样的具体阈值 |
| 实验 7 BLEU/cos_sim 未量化 | 中 | 见 L4 | 同 L4 修复 |

---

## V03-SlidingWindow 审查

### 逻辑问题

**L1. 实验 1 与实验 4 高度重复**
- 两者都是同一脚本扩展 ctx 列表（D 节也承认）。建议合并为「实验 1+4 合并 ctx sweep」，单 agent 跑 ctx ∈ {64, 128, 256, 511, 512, 513, 514, 1024, 4096}。

**L2. SLIDING_WINDOW = 512 是假设值，未在实验前验证**
- E 节已登记，但实验 1 直接用 512 设计 ctx 边界（512 / 513 / 514）。若实际 W ≠ 512（例如 W=4096），整张表都需重做。建议在 D 节"执行顺序"前面加一步 P0：`grep -i sliding /root/.cache/huggingface/.../config.json` 确认 W 数值。

**L3. 实验 2 "构造特征 KV" 设计可行但缺少 reference 对比**
- 通过标准只看「输出 head 是否包含特征值」，但没规定特征值贡献量级；如果 softmax weight = 1e-6，肉眼 / grep 都难判别。建议给定下界：`output_norm_with_feature - output_norm_without_feature > σ` 的具体 σ。

**L4. checkout `7ebae9afb^` 复现 buggy 状态需要保证 ATOM/aiter 其它修复不被同时回退**
- aiter 在该 commit 之前可能还没有 `68fc7d48b` MoE fix，回退后 V01 路径会同时被破坏，cos_sim 退化无法归因到 sliding window。必须确认 `7ebae9afb` 的父 commit 是否已包含 V01/V02 fixes，或在 cherry-pick 模式下只回退 sliding window hunk。

### Agent Team 分工建议

| Agent | 负责实验 | 预计时长 | 依赖 |
|-------|---------|----------|------|
| **V03-A**（W 数值确认 + ctx sweep） | 前置 W 验证 + 实验 1 + 实验 4（合并） | 45 min | 无 |
| **V03-B**（decode 专项） | 实验 2（构造 KV 特征） | 60 min | V03-A PASS |
| **V03-C**（端到端） | 实验 3（buggy / fixed / reference 三方） | 60 min | V03-A PASS（V01/V02 不强依赖） |

**节省时间估算**：串行 ~3h；V03-A 后并行 B+C ≈ 1.5h，节省 ~50%。

**批量打包**：实验 1 + 实验 4 必须合并（已论证）。实验 3 的三方对照可作为单 driver 三次 invocation。

### 准确性风险

| 风险 | 严重度 | 说明 | 修正建议 |
|------|--------|------|----------|
| cos_sim 0.999998 阈值合理性 | 低 | 已有 buggy 0.998982 vs fixed 0.999998 的实测对照，区分度 1000× | 保留 |
| 实验 3 "ungi" 乱码 pattern 仅作为定性证据 | 中 | 未量化 spam 频次 | 给出 token-level repetition rate 阈值 |
| ctx=512 边界判定（W==ctx）未细化到 W±1 | 低 | 当前已包含 511/512/513，覆盖足够 | 保留 |
| 实验 3 长 prompt（10000 tokens）会撞上 V01 已知的 tp=4 全 BOS bug | 高 | 计划自己提到了"必要时降到 tp=2"但未硬规定 | 实验 3 强制 tp=2，把 tp=4 长序列另列为已知 open bug 不验证 |

---

## 跨 V01/V02/V03 发现

### 共性问题

**S1. cos_sim 阈值缺乏统一的 noise floor 基线**
- V01 用 0.9999、V02 用 0.99998、V03 用 0.999998。三个数量级跳跃没有论证。三个专题都暗示阈值是"经验值"。建议在 Synthesizer 阶段引入「全局 noise floor 实验」：固定 seed × 5 次同一 op 的 cos_sim，取最坏值作为各 op 的基线，阈值 = baseline + ε。

**S2. "人工目检无乱码" 主观判定贯穿三专题**
- V01 实验 3、V02 实验 3/4、V03 实验 3 都依赖人工 review。建议统一改为：(a) repetition rate 阈值；(b) 与 reference 的 BLEU/ROUGE；(c) `<s>` 重复 token 计数。

**S3. 修改工作树缺少隔离机制**
- V01 实验 5（注释 ATOM padding）、V02 实验 4（可能加临时 env switch）、V03 checkout 旧 commit，都直接动 `/home/hanchang/aiter` 或 `/home/hanchang/ATOM`。违反 MEMORY.md 要求（commit 必须从 junlin12_repos）。建议统一约定使用 worktree。

**S4. CK kernel JIT cache 状态未在执行前确认**
- V01 担心 kernelName2 未编译；V02 担心 stale `.so`；V03 不涉及 JIT 但 checkout 会触发重编译。Synthesizer 应在 pipeline 开头加一步「JIT cache 健康检查 / 清理」。

### 相互依赖

| 依赖 | 描述 | 当前粒度问题 |
|------|------|-------------|
| V02 → V01 | V02 整体声明依赖 "V01 PASS" | 粒度过粗。V02 实验 1（kernel op_test）只依赖 Fix 2（block_m=128 强制），不依赖 Fix 1（preshuffle）。可让 V02-A 在 V01-C（实验 2）PASS 后即启动，无需等 V01 全部完成 |
| V03 → V01/V02 | V03 实验 3 端到端共享 ATOM stack | V03 强制 tp=2 后基本独立。但若 ATOM 上 V02 wiring 有 bug，V03 实验 3 长 prompt 会受 SwigluStep 路径误差污染 sliding window 判定 |
| V01 实验 3 → V02 wiring | V01 端到端 tp=2 推理也会触发 layer 43-44 SwigluStep 路径 | V01 实验 3 实际上 *已经* 隐式 verifying V02 wiring。重复测试可压缩 |

**优化建议**：把"端到端 tp=2 短 prompt"提到 pipeline 顶层共享一次，由 Synthesizer 统一收集证据。三个专题在端到端实验上合并节省 ~60min。

---

## 给 Synthesizer 的关键问题清单

| 优先级 | 问题 | 影响范围 | 推荐处理 |
|--------|------|---------|----------|
| **P0** | 全局 noise floor 实验缺失，三个专题阈值无统一基线 | V01/V02/V03 | 在 pipeline 入口加 V00 实验：固定 seed × 5 次同一 MoE / SwigluStep / paged_decode 的 cos_sim 取最差值 |
| **P0** | 修改工作树（V01-exp5 / V02-exp4 / V03-checkout）缺隔离 | V01/V02/V03 | 统一要求使用 EnterWorktree 或 junlin12_repos clone |
| **P0** | V03 SLIDING_WINDOW = 512 是假设，未实测 | V03 全部实验 | pipeline 第一步实证 W 值 |
| **P0** | V01 实验 2 kernelName2 命中未直接验证（fallback 假性 PASS 风险） | V01 实验 2.c-2.e | 强制 AITER_LOG_LEVEL=INFO + grep |
| **P1** | V02 实验 4 BOS-spam 通过标准过宽（±20%） | V02 长输出结论 | 改用统计检验；并把实验 6 升级为必做 |
| **P1** | V02 实验 1 cos_sim 0.99998 阈值依据不明 | V02 全部精度判定 | noise floor 之后再设定 |
| **P1** | V01 实验 4 grep-only "回归验证" 不构成 P0 证据 | V01 gfx942 风险评估 | 降级为 P2 + 标注「弱保证」 |
| **P1** | V01 实验 3 已隐式覆盖 V02 wiring，工作重复 | V01/V02 | 合并端到端实验由 Synthesizer 统一调度 |
| **P2** | V02 实验 5 weight check 脚本路径未确认 | V02 实验 5 | 执行前 Glob 验证 |
| **P2** | V02 实验 7 反向实验通过条件不闭合 | V02 设计决策辩护 | 量化劣化阈值 |
| **P2** | V03 实验 2 特征贡献量级未给阈值 | V03 单元证据 | 给定 σ 数值 |
| **P2** | V01 实验 2 边界 192-256 之间未细化 | V01 边界结论 | 加 200/224/240 三 case |

---

## 附：Agent Team 总览（跨专题）

| 阶段 | 并行 Agent 数 | 总耗时 | 串行基线 | 节省 |
|------|--------------|--------|----------|------|
| 阶段 0（noise floor + JIT 健康检查） | 1 | 30 min | — | — |
| 阶段 1（V01-A/B/C + V03-A 同时起） | 4 | 90 min | — | — |
| 阶段 2（V01-D + V02-A/B/C/D + V03-B/C） | 7 | 120 min | — | — |
| 阶段 3（V01-E + V02-E） | 2 | 360 min（V02-E 慢） | — | — |
| **合计** | — | **~10h** | **~17h** | **~40%** |

V02-E（BOS-spam + 噪声累积量化）是关键路径瓶颈，建议 Synthesizer 评估是否提前启动或拆分。
