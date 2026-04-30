# 17_atom_moe_tp8_load_crash — ATOM MoE tp=8 load_w2 / load_w13 narrow size<0 issue draft

> 日期：2026-04-29
> 来源 wave：fp8-tp4-repro / issue_wave（wave 6）
> 状态：✅ 内部 wave CLOSED（draft + reviewer 全部就绪）；⏸ **尚未 file 到 ATOM upstream**（GitHub issue 动作未授权）
> 红线遵守：未修任何源码 / 未改任何 PASS 判定

---

## 1. 这是什么

ATOM 在 `stepfun-ai/Step-3.5-Flash-FP8` + `-tp 8` 下 **weight load 阶段确定性崩溃** 的 bug 报告草稿。根因位于 `atom/model_ops/moe.py`：`_load_w2`（line 2335-2364）与 `_load_w13`（line 2292-2333）使用基于 `ceil` 的切分，未对 trailing rank 上 `start >= D` 做兜底，导致传入 `torch.Tensor.narrow()` 的 `size` 参数为负。

**bug 性质**：
- 确定性（无 race / 无 GPU 非确定性）
- 与 inference / cudagraph / batching / sampling **无关**
- 是 `(D, tp_size)` 的纯函数
- 在 `_load_w2` 与 `_load_w13` 两处都存在同一模式（fix-then-sweep target）

**触发例子**：`D = 10`（`inter_size = 1280` 的 per_1x128 scale 块数），`tp_size = 8` → rank 5 命中 `size = 0` 导致 `copy_` 形状不匹配，rank 6/7 命中 `size < 0` 导致 `narrow` 报错。

---

## 2. 文件清单

| 文件 | 用途 | 行数 |
|---|---|---|
| `ATOM_ISSUE_DRAFT_zh.md` | **中文版主交付**（推荐先读）| 226 |
| `ATOM_ISSUE_DRAFT.md` | 英文版（可直接 copy-paste 到 GitHub issue）| 226 |
| `TEAM_CONFIG.md` | issue_wave 配置 + 红线 + 验收标准 |  -  |
| `WAVE_CLOSE.md` | issue_wave 关闭报告 + reviewer finding 处理决策表 |  -  |
| `progress/iw-reviewer.md` | reviewer 报告（评级 A-，0 block / 1 HIGH / 4 warn / 3 info；HIGH-1 已修）| 168 |
| `README.md` | 本文件 |  -  |

---

## 3. 与前序 wave 的关系

| 前序 wave | 路径 | 与本 wave 的关系 |
|---|---|---|
| 15_perf_tp2_tp4_tp8_eval | `../15_perf_tp2_tp4_tp8_eval/` | 报 tp=8 PASS（短 prompt + cudagraph_capture_sizes=[1]）；本 wave §"Caveats #4" 标注该 PASS 与本崩溃**无法调和**，是开口 sub-question |
| 16_perf_gfx950_verified | `../16_perf_gfx950_verified/` | 与本 bug 无直接交集（cross-arch 数据） |
| fp8-tp4-repro 主线 | （另一仓）| 主线已落 3 个 sibling fix（`aiter/fused_moe.py:881-886` / `atom/model_ops/moe.py:1709-1746` padding / `atom/model_ops/utils.py:79`）；本 wave 暴露的 `_load_w2/w13` 边界 case 是该家族中第四个、最上游的一个 |

---

## 4. 是否已 file 到 ATOM upstream

**否。** 本目录是 issue **草稿**，存档于本仓供：
- 项目内 PROJECT_SUMMARY 的 evidence 引用
- 内部转给 ATOM 维护者（Slack / 邮件 / 内部 PR review）的素材
- 未来如决定公开 file，可直接以 `ATOM_ISSUE_DRAFT.md` 作为 issue body 提交到 https://github.com/ROCm/ATOM/issues

是否公开 file 不在本 wave 的范围内，等用户/team 决策。

---

## 5. 推测修复（advisory only — 未实施）

详见 `ATOM_ISSUE_DRAFT_zh.md` §"推测修复方案"。两个并列方案：
- **方案 A**：`_load_w2`/`_load_w13` 在 `start >= D` 时 early return（最小改动，需 audit 下游消费者）
- **方案 B**：偶数切分 + 余数挂到 rank 0（永不触达负 size，但反转 partial scale block 的 per-rank 顺序，需上游 review）

两者均**未实施**，等 ATOM 上游决策。

---

## 6. Caveats（draft 已诚实标注的不确定性）

1. `D = 10` 与 `inter_size = 1280` 均为**推断**（来自 `_load_w13` 注释 line 2306-2309 的例子 + 观测到的 rank 5/6/7 症状切分），未从模型 config / dump 直接提取。上游可在 `moe.py:2357` 前加 `print(name, loaded_weight.shape)` 自验。
2. `_load_w13` 在本模型 + tp=8 下的崩溃**未独立观测**（`_load_w2` 先崩；论据基于代码结构完全相同）。
3. 两个推测修复方案均未实施。
4. perf wave tp=8 PASS 与本崩溃无法调和（见上文 §3 表）。

---

**End of README.**
