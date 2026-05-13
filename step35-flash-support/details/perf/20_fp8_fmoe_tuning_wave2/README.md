# 20_fp8_fmoe_tuning_wave2 — 导航 README

## 路径声明

- 本 doc 路径：`/home/junlin12/project_summary/step35-flash-support/details/perf/20_fp8_fmoe_tuning_wave2/`
- 创建 ts：2026-05-12
- 状态：
  - **doc 写作**：COMPLETE
  - **wave 实验结论**：FAIL（工程动作 4/4 done，但 OPT-1 axis 性能改善 0/N → wave-level 证伪）
- 项目：step35-flash-support / perf 分类 / NN=20（接续 17 README pattern + 16 §四-B causal 起点）

## 子文件索引

| 文件 | 职责 |
|------|------|
| `README.md` | 本导航文件 |
| `RESULTS.md` | 主报告（TL;DR / Outcome / 数据矩阵 / aiter bug / byte-drift / 推测根因 / 6 Promote / Future Work / Appendix） |
| `TEAM_CONFIG.md` | stub link → wave2_archive |
| `WAVE_CLOSE.md` | stub link → wave2_archive |
| `progress/INDEX.md` | wave2_archive/progress/ 11 teammate 文件路径索引 |
| `logs/INDEX.md` | wave2_archive/logs/ 主要 log 类别索引（raw 不复制） |

## Quick Start — 未来 wave 该读这 3 节（救命级）

未来 wave 接手 fp8 fmoe / OPT-1 / multi-prompt 类任务前，**必读**主报告以下 3 节：

1. **救命级 #1 — q_dtype dispatch key 必须 fnuz** → `RESULTS.md §7.1`
   - 一句话：`torch.float8_e4m3fnuz` not `e4m3fn`，否则 dispatch KeyError 或走错路径
2. **救命级 #2 — 上游 aiter bug**：multi-prompt + tp≥4 + fp8 ck2stages 必崩 → `RESULTS.md §4`
   - 一句话：`/workspace/aiter/aiter/fused_moe.py:1646` `apply_act_and_mul` shape 不兼容；np≥4 + tp≥4 必触发；跑 multi-prompt 前先 verify bug 是否 fix
3. **救命级 #3 — FP8 byte-drift verification anti-pattern** → `RESULTS.md §5`
   - 一句话：fp8 累加顺序变 → bf16 低位飘 → softmax 边界翻；**不要**用 byte-equal 作 fp8-vs-fp8 verification；改用 cos-sim / 启发式 PASS / bf16 reference 距离

## Wave2 Archive Cross-link

完整工件区根路径（含 raw csv / patch / log / progress / TEAM_CONFIG / WAVE_CLOSE / archival memory）：

- **root**：`/home/junlin12/step35_fp8_optim_wave2/`
- TEAM_CONFIG：`/home/junlin12/step35_fp8_optim_wave2/TEAM_CONFIG.md`
- WAVE_CLOSE：`/home/junlin12/step35_fp8_optim_wave2/WAVE_CLOSE.md`
- progress 11 teammate：`/home/junlin12/step35_fp8_optim_wave2/progress/`
- logs：`/home/junlin12/step35_fp8_optim_wave2/logs/`
- doc planning（本 doc 的策划工件）：`/home/junlin12/step35_fp8_optim_wave2/doc_planning/`

本 project_summary doc 不复制 wave2_archive 任何原文，仅 link 引用。

## 6 Promote 落点速查表

| Promote | 一行简述 | 主仓位 |
|---------|---------|--------|
| **Promote-1** | OPT-1 axis 证伪（0/N 改善），未来重启需先 layer-dump 实证根因 | `RESULTS.md §7.2` + KNOWN_FACTS F8 |
| **Promote-2** | FP8 byte-equal 物理必然 drift，verification 改 cos-sim/启发式 | `RESULTS.md §5` + `§7.3` + KNOWN_FACTS F9 |
| **Promote-3** | dispatch HIT ≠ perf；HIT 6/6 ≠ 实际加速 | `RESULTS.md §9.4` (appendix) + agent-team SKILL.md 反模式表 #21 |
| **Promote-4** | aiter `apply_act_and_mul` shape bug，np≥4+tp≥4+fp8 ck2stages 必崩 | `RESULTS.md §4` + 上游 GitHub issue（user 决策） |
| **Promote-5** | bench script `max_model_len` bug（5 行 trivial patch） | `RESULTS.md §7.6` (appendix) |
| **Promote-6** | q_dtype dispatch key 修正（doc_t3 raise 新候选）；KNOWN_FACTS F5 加注 | `RESULTS.md §7.1` + KNOWN_FACTS F5 修订 |

## 上游 cross-link

本 wave 是 16 wave 的 causal 续作 + KNOWN_FACTS 的实证修订源：

- 入向 causal 起点：`/home/junlin12/project_summary/step35-flash-support/details/perf/16_perf_gfx950_verified/RESULTS.md §四-B (#601 trigger)`
- 入向 disclaimer pattern：`/home/junlin12/project_summary/step35-flash-support/details/perf/16_perf_gfx950_verified/RESULTS.md §附录-DISCLAIMER`
- 入向 KNOWN_FACTS 落点：F5 (q_dtype dispatch key) / F8 (OPT-1 axis 证伪) / F9 (FP8 byte-drift)
- 反向 link（待 user 决策 Q4）：是否在 16/RESULTS.md §四-B 末加 "→ 详见 20_fp8_fmoe_tuning_wave2/"
