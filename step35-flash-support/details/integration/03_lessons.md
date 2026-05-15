# 跨 patch 教训 (vllm/ATOM 集成层)

> 此文档为 **step35-flash-support 系列 patch 文档之一**, 完整索引见 [README.md](./README.md)。兄弟 doc: [01_patch_swiglustep.md](./01_patch_swiglustep.md) | [02_patch_swa_perlayer.md](./02_patch_swa_perlayer.md)。
>
> 本 doc 提取 Patch A + Patch B 调研 + apply + verify 全过程暴露的 5 条**跨 patch 共同教训**, 编号承袭 agent-team SKILL.md 反模式表 (#20 / #22) 和 USER_REPORT 中的 promotion candidates。供后续 wave 派单 + reviewer prompt 设计 + 文档化流程参考。

---

## 教训 1 — Self-report ≠ Ground truth (反模式 #20 实例)

### 现象 (Patch B T52)

T52 progress L11 自报 `patch_state: ROLLED_BACK`, L178-179 写"已 cp 回 .bak.t52", 但**实地文件系统状态不一致**:

- 后续 T54 启动前 cp 的 `bak.t54` md5 = `5289f0d6c7f6a2364f5f6b8025c352c9` (**patched**, 含 T52 init=0 修正)
- T55 audit 实地 grep `swa_max_num_heads_kv` 字段, 仍命中 (在文件中)
- T55 audit prompt 直接基于 T52 self-report, audit 时立即挑出 fact mismatch 并标 blocker

### Root cause (两种诚实可能, progress 链不足以单独锁定)

1. **T52 self-report 错误**: 实际未执行 cp 回 bak, 但按 prompt 红线"失败 patch 必回滚"惯性写"已回滚"
2. **T52 真回滚 + 未编号动作重 apply**: 但 lead 派单流通常都进 progress, 此假设缺 evidence 支持

### 防御对策

1. **每个 patch teammate 完成后, lead 必须自跑 `md5sum + grep` 实测**, 不依赖 teammate self-report 决定 patch state
2. **Reviewer prompt 加红线**: "actor 报 ROLLED_BACK 必须附 `md5sum` 输出 + `diff` 验证, 缺即视为未实证"
3. **跨 wave 状态指针** (USER_REPORT.md "跨 session 指针" 节) 必须**实地 md5 验证后**写入, 不引用 teammate self-report

### 影响范围

不只 SWA patch — 任何 source-modify teammate 都可能踩。Patch A 因为 T32 实测 mtime + .bak 双备份, 单次 forward apply 没踩; 但若 patch 流程更长, 同样高危。

---

## 教训 2 — Dispatch enum 加白 ≠ 数值正确性

### 现象 (Patch A vs wave G-3(i) NaN)

Patch A T33 verify 5/5 PASS (含 96 次 SwigluStep dispatch + JIT build CK kernel 84.6s 成功) → **dispatch 路径完全 work**。
但同 wave 后续 (T63 → T67) 发现 vllm tp=8 路径有 NaN, **NaN root cause 与 SwigluStep 解耦** (T67 实证 enforce_eager 模式无 NaN, 输出语义正确)。

### Lesson

**不能用"dispatch path work" 推断 "数值 path work"**:

| 维度 | T33 实测 | 是否覆盖? |
|---|---|---|
| backend 选择 (Oracle gate) | ✅ 96 次命中 | dispatch |
| dispatch elif 路由 | ✅ 0 ValueError | dispatch |
| CK kernel JIT build | ✅ 84.6s 成功 | dispatch |
| MoE expert 数学 forward | ❌ T33 未跑 inference | **数值** (T67 enforce_eager 间接证明 OK) |
| 整 model 输出语义 | ❌ T33 仅 startup | **数值** (T67 间接证明) |

T33 的 5 维 grep PASS 全是 **dispatch / startup 维度**; 数值维度需要跑 inference + 比对参考输出 (sanity prompts 或 logprobs 比对)。

### 防御对策

1. **patch verify 阶段** (派单 prompt) 必须显式拆分 "dispatch verify" + "numerical verify" 两个 phase, 不能合并
2. **派 verify teammate 时**, 红线写"startup PASS ≠ inference 数值正确, 必须额外跑 ≥1 个 sanity prompt 比对参考输出"
3. **如果 verify 仅 startup PASS** (如 T33 当时), 文档化时**显式标注**"数值正确性未在本 patch 范围内验证, 仅由后续 T67 enforce_eager 间接证实"

---

## 教训 3 — Caveat-stripping 防御 (反模式 #22)

### 现象

Patch A + Patch B **均**与 wave G-3(i) NaN 解耦, 这个 caveat 在以下位置都出现:

- DOC1-A progress §"unverified_assumptions_carried_forward"
- DOC1-B progress §4 强标注
- 01_patch_swiglustep.md §"Caveat / 已知局限" 🔴 强标注
- 02_patch_swa_perlayer.md 顶部 CAVEAT 强标注 + §"Caveat" 表格 + §"3-axis 隔离结论" 引用
- README.md §"共同 caveat"
- USER_REPORT.md §"3-axis 隔离结论" L93-97

如果**任何一处** silent strip 这个 caveat, 后续接手人就可能误以为"Patch B 解了 NaN" → 再也不查 inductor/cudagraph 真因 → 重复踩坑。

### Lesson

反模式 #22 (Plan 假设 silent 升格 / caveat-stripping) 在**多 doc 综合任务**上风险尤其大:

- 调查 teammate 标 caveat 时诚实
- synth teammate 把多份 progress 拼成 doc 时, 可能为"措辞简洁"strip caveat
- reviewer 在 isolated context 不查外部 doc, 看不出 caveat 是否被 strip
- 接手人读 doc 时, 看不到 caveat = 默认"patch 解决了所有相关问题"

### 防御对策

1. **synth teammate prompt 加红线**: "caveat 措辞必须**逐字保留**, 不得改写为'仅适用 X 路径'等弱化措辞"
2. **每份 doc 顶部 banner 区域**留 CAVEAT 强标注 (本 doc 集 02 已示范)
3. **README.md 集中重复一次共同 caveat** (本 doc 集已做, §"共同 caveat" 节)
4. **reviewer 5 维 GPA 加 caveat-preservation 维度** (或者扩 Plan Adherence): 比对源 progress + doc 草稿, 确认 caveat 措辞**逐字保留**
5. **lead 在 wave 收尾**, 最后一道 grep 检查: `grep -ri "解耦\|不解决\|与.*无关" doc_dir/` 看 caveat 命中数与源头一致

---

## 教训 4 — 多次 patch-revert-restore 的 .bak audit trail 价值

### 现象 (Patch B 5 个 .bak 文件)

| .bak | md5 | 含义 | 价值 |
|---|---|---|---|
| `bak.t50` | `d49ef0...` | T50 apply 前 (clean HEAD) | rollback 锚点 |
| `bak.t51` | `d49ef0...` | T51 dump 前 (与 t50 同) | 防 dump 改坏的兜底 |
| `bak.t52` | `d49ef0...` | T52 apply 前 (与 t50 同) | 防 init=0 修正版改坏的兜底 |
| `bak.t54` | `5289f0...` | T54 启动前 (patched 状态) | **关键**: 反推 T52 self-report 与实地状态矛盾的 ground truth |
| `bak.t65` | `5289f0...` | T65 control 前 (patched 状态) | 防 git checkout HEAD 后无法 restore |

### Lesson

**每次 patch / revert / restore 动作都打 .bak, mtime 自然递增**, 是 wave 状态机的 audit trail:

- pre-patch md5 一致 (`d49ef0...` 三份) → 实证 T50→T51→T52 之间 file 内容不变
- post-patch md5 一致 (`5289f0...` 两份) → 实证 T54→T65 之间 file 内容不变
- 三方 md5 自动还原 wave 状态变化时点 → 反推 self-report 错误 (教训 1)

DOC1-B 调研时**仅用 6 个 md5 + 1 个 diff 命令**就还原了整 wave 的 patch 状态变化序列。

### 防御对策

1. **任何 source-modify teammate prompt** 必须红线: "改 file 前先 `cp file file.bak.tNN` (NN = teammate id), 不得 in-place 改"
2. **bak 命名严格 `<file>.bak.t<NN>`**, 不要用 `.bak.20260514` 等时间戳 (难按 teammate 索引)
3. **wave 收尾前** lead 跑一次 `ls -la <files>.bak.* | sort` + `md5sum` 表格, 留在 WAVE_CLOSE.md 作为 pause-and-resume 锚点
4. **bak 不删**, 即使 md5 重复也保留 — 多余成本几乎为零, 但 audit 价值高

---

## 教训 5 — vllm v1 inductor / cudagraph 与 ROCm fp8 的相互作用是 wave G-3(i) **真 P0**

### 现象

整个 wave G-3(i) 历经 19 个 teammate, ~230 tool calls, 排除 MTP (T63) + 排除 SWA Path-1 patch (T65) 之后, 才在 T67 + T71 双证锁定 P0:

- **T67**: vllm Python API + `enforce_eager=True` → 0 NaN, 输出语义正确 (`"List primes within 50:" → " 2, 3, 5, 7, 11, 13, ..."`)
- **T71**: vllm serve HTTP + `--enforce-eager` → 3 探针全 200, logprobs=1 路径 64 token 全有限实数, 服务端 log 0 NaN

排除 fp8 model code / aiter MoE kernel / OpenAI server 包装层 3 个反向假设后, P0 = **vllm v1 inductor compile / cudagraph 优化路径在 ROCm + fp8 + step3p5 配置下引入 NaN**。

### 推断 (未实证, 长期 fix 需 vllm/PyTorch 团队介入)

| 可能机制 | 假设 | 验证路径 |
|---|---|---|
| torch.compile fusion + fp8 dequant | inductor 把 fp8 quant scale 路径融合到 fused kernel, scale broadcast 时溢出 (e4m3fn vs e4m3fnuz dtype 在 ROCm 平台错配) | `TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_TRACE_GRAPH=1` dump fused kernel 源码 |
| cudagraph capture + dynamic shapes | capture 期间 NaN guard 被绕过, replay 阶段 NaN 在 attention KV / softmax 数值边界爆发 | 区分 "compile + no graph" vs "compile + graph" 跑 (USER_REPORT.md "长期 fix 路径 A" 矩阵) |
| aiter `fmha_v3_varlen_fwd` 在 capture context 下的 q/k/v_descale | T67 第 3 轮 log 8 worker 全报 `[aiter] type hints mismatch, override to --> fmha_v3_varlen_fwd(... q_descale, k_descale, v_descale ...)`; 这些 descale 张量在 cudagraph capture 模式下 buffer 可能未正确 init 为 finite | per-layer dump fp8 scale 张量 |

### Lesson

**早期 wave 的"集成路径 patch"是必要的 unblock 条件, 但不是最终 P0 fix**:

- Patch A 解了 vllm dispatch enum gate
- Patch B 解了 SWA workspace shape (tp=2 路径)
- 两者一起让 wave 能"跑起来"; 但跑起来后, 数值层在 inductor/cudagraph 路径仍 NaN
- **如果调查停在"patch 应用了, 看起来 work" 阶段, 永远抓不到真 P0**

### 防御对策

1. wave-planning 阶段必须显式区分 **"集成 unblock patch"** (dispatch / shape / API 兼容) vs **"数值正确性 fix"** (model output 语义对齐); 两者**串联**而非"前者 PASS 即整体 PASS"
2. 用户 directive "确保 vllm 输出正确" 必须**翻译成数值 verify 任务** (sanity prompts + logprobs 验证), 不能止步于 startup PASS
3. 工具链层 (vllm v1 / inductor / cudagraph / torch.compile) 应在静态调研排除 model 算法层后**早跳** (USER_REPORT.md promotion candidate 4); 本 wave 漏跑 enforce_eager 早 1-2 phase

### 跨 session 指针

详见 `USER_REPORT.md`:
- §"用户立即可用的 fix workaround" — 方案 A (Python API) + 方案 B (HTTP serve) 双路径 workaround
- §"长期 fix 方向" — 路径 A (区分 compile vs cudagraph) / 路径 B (per-kernel dump) / 路径 C (vllm GitHub issue 模板)
- §"P0 实证 evidence chain" — 配置矩阵 + 3-axis 隔离 + 反方向假设证伪

---

## 附录 — 编号映射表 (本 doc 教训 ↔ agent-team SKILL.md 反模式 ↔ USER_REPORT promotion candidates)

| 本 doc 教训 | agent-team SKILL.md 反模式 # | USER_REPORT.md promotion candidate |
|---|---|---|
| 1 (self-report ≠ ground truth) | #20 (self-report-as-ground-truth) | (与 PC 5 "no-show artifact" 同源) |
| 2 (dispatch ≠ 数值) | (新; 候选 promotion) | (与 PC 4 "工具链层早跳" 互补) |
| 3 (caveat-stripping) | #22 (Plan 假设 silent 升格) | PC 3 (Phase 9 USER_REPORT silent strip tp=8 caveat) |
| 4 (.bak audit trail) | (新; 候选 promotion) | (与 PC 7 "reviewer timing race" 同 family — 都是 ground truth 锚点) |
| 5 (集成 unblock vs 数值正确性) | (新; 候选 promotion) | PC 4 (静态调研 EXCLUDED 后应跳工具链层) |

---

_Promoted from `project_step35_vllm_repro/` (wave DOC-1) at 2026-05-15. 原 4 份 doc 在该 wave WORK_DIR 的 progress/teammate-DOC1-{A,B,C,D,E,F}.md 有完整审计 trail。_
