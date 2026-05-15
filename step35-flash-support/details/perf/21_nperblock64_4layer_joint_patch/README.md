# 21_nperblock64_4layer_joint_patch — 导航 README

## 一句话 atomic story

【实证】stepfun-Flash-FP8 tp=4（inter_dim=320）走 CK MoE block-scale GEMM 的 `NPerBlock=64` dispatch path 触发 4 层耦合 scale layout mismatch；4 层 joint patch（CK 3 层 105+/15- in `gridwise_moe_gemm_blockscale.hpp` + caller 1 层 `w1_quant_blk_n` dispatch-aware）落地后 → bit-exact correctness（max_err=0.0）+ kernel-level 快 15.80% + model-level PERF_NEUTRAL（HIGH confidence）；硬件 = MI308X (gfx942)。

## 路径声明

- 本 doc 路径：`/home/junlin12/project_summary/step35-flash-support/details/perf/21_nperblock64_4layer_joint_patch/`
- 创建 ts：2026-05-15
- 状态：
  - **doc 写作**：COMPLETE（reviewer 阶段 + user 确认前不 push）
  - **wave 实验结论**：PASS（dual-superior on this fixture，多 caveat 限定）
- 项目：step35-flash-support / perf 分类 / NN=21（接续 20 wave2 entry，NN_max=20+1）

## ⚠️ Hardware axis 显式声明

【实证】本 entry 实测硬件 = **MI308X (gfx942)** / 4 卡组合。
- **不引用 gfx950 entry (`16_perf_gfx950_verified`) 数据**
- **不与 gfx950 entry 做 cross-link 推断**
- 防 wave2「hardware axis 错配」反模式（entry 20 README 已 commit `126a3b4` clarify）

## 子文件索引

| 文件 | 职责 |
|------|------|
| `README.md` | 本导航文件 |
| `RESULTS.md` | 主报告（12 章：TL;DR / Outcome / 工程动作 + 4 层 patch verbatim / 数据矩阵 / Root Cause / Cross-wave 自洽 / 9× discrepancy / F-A 破 caveat / Incident / Lessons / Caveat / Production Verdict / Appendix）|
| `progress/INDEX.md` | 三 wave + 上游 + candidate verify + 本整合 wave 的 artifacts 路径 stub（不复制原文，与 entry 20 / 16 子目录格式平行）|

（注：未生成 `TEAM_CONFIG.md` / `WAVE_CLOSE.md` / `logs/INDEX.md` stub —— 本 entry 是三 wave 整合产出，原 wave artifacts 仍在 `/home/junlin12/m1quad_*_wave/` 各自路径下；本 entry 通过 §12.1 物料索引 + §12.3 Cross-link + `progress/INDEX.md` 直接 link 引用，不复制。子目录可发现性 stub 已在 `progress/INDEX.md` 补全，对齐 entry 20 / entry 16 模式。）

## entry 21 内容索引（指向 RESULTS.md 各节）

| Topic | RESULTS.md 节 |
|-------|--------------|
| TL;DR + 5 项核心 verdict | §TL;DR |
| Wave PASS 显式声明 + 章节 PASS/FAIL 标注 | §1 Outcome |
| 4 层 patch 落地表 + verbatim diff snippet | §2.1 / §2.2 |
| 三层 perf 缺一不可通用教训 | §2.3 |
| Correctness 表 (192/320/448 bit-exact + 384 control) | §3.1 |
| Kernel-level perf 表 (pad 77.118us / nopad 64.932us / -15.80%) | §3.2 |
| Model-level e2e perf F-A 表 (TTFT 973.2 vs 965.4 / Δ ≤ 0.81%) | §3.3 |
| Dispatch path 实证 (.so mtime + 文件名 per_1x128) | §3.4 |
| Root Cause (4 层耦合 + 触发条件 + mechanism) | §4 |
| Joint 必要性实证 (candidate A/B FAIL) | §4.4 |
| Cross-wave 自洽 (~20× amortize 与 F3 一致) | §5 |
| 9× discrepancy 解析 + baseline 1 EXCLUDED 决策 | §6 |
| F-A 终极实证如何破 C1/C4/C7 caveat | §7 |
| 3 个 incident 透明度记录 + Patch state restore 验证 | §8 |
| Lessons & Promotes (本 atomic story 衍生 6 candidate) | §9 |
| Caveat (joint-fix 7 + kernel 14 + model F-A 后剩余 + §22 红线) | §10 |
| Production Verdict + 适用边界 + Follow-up wave 建议 | §11 |
| 物料 path 索引 + 三仓 SHA + Cross-link + 命令速查 | §12 |

## 上游 wave artifacts cross-link

本 entry 整合三 wave + 一 partial-root-cause wave 的 artifacts：

| Wave | 路径 | 作用 |
|------|------|------|
| 上游 partial root cause | `/home/junlin12/m1ppp_padding_explanation_wave/` | H_alt_1 partial mechanism 提出（PADDING_EXPLAINED.md §3）|
| Joint-fix correctness | `/home/junlin12/m1quad_joint_fix_wave/` | 4 层 patch 落地 + 三点 NPerBlock=64 bit-exact |
| Kernel-level perf | `/home/junlin12/m1quad_perf_compare_wave/` | pad vs nopad kernel timing (15.80% delta) |
| Model-level e2e perf | `/home/junlin12/m1quad_model_perf_wave/` | F-A 同 HEAD CK patch on/off → PERF_NEUTRAL HIGH |
| 本 project_summary 整合 | `/home/junlin12/m1quad_project_summary_wave/` | reader teammate 4 个 progress + recon + synth + reviewer |
| Candidate 方案 verify | `/home/junlin12/m1ppp_ck_bscale_fix_verify_wave/` | candidate A/B FAIL → joint 必要性反证 |

## §22 反模式防御 红线（lead 显式列出，本 entry 已遵守）

1. **保留 caveat 措辞**：joint-fix 7 + kernel 14 + model F-A 后剩余 verbatim 全收 → §10 三节
2. **Production verdict 严格 deployment context 限定**：仅 stepfun-Flash-FP8 + tp=4 + gfx942 + NPerBlock=64 path → §11.1
3. **Hardware axis (gfx942) 显式标记**：本文件顶部 + RESULTS §10.4 末段 + §12.3 反向 cross-link 段
4. **数据 verbatim**：所有 metric / commit SHA / log path 从 reader teammate progress copy-paste（patch diff 从 reader-joint-fix verbatim 摘 + synth 阶段实地 git diff --stat 核对 105+/15- 一致）
5. **外部引用 grep 实证**：entry 16 / 20 / 18 已 ls 实证存在；entry 20 RESULTS.md / README.md 已 Read 实证结构与措辞参考
6. **代码 patch snippet copy-paste**：4 层 verbatim 摘自 reader-joint-fix progress §2，与 patch 文件一致

## Quick Start — 未来 wave 该读这 4 节

未来 wave 接手 NPerBlock=64 path / fp8 fmoe scale layout / fmoe 4 层 joint patch 类任务前，**必读** RESULTS.md 以下 4 节：

1. **§4 Root Cause (4 层耦合)**：避免再走单层修 candidate A/B 已证伪 FAIL 路
2. **§5 Cross-wave 自洽 (~20× amortize)**：kernel-level perf 收益 e2e amortize 数学校验，不要直接外推
3. **§7 F-A 终极实证如何破 caveat**：同 HEAD 唯一变量 stash/pop 是高 ROI 设计模板，未来 model-level perf wave 复用
4. **§10 Caveat 节 + §11.2 适用边界**：production rollout 前必须确认 deployment context 在 §11.1 限定内，否则走 §11.3 follow-up wave 验证
