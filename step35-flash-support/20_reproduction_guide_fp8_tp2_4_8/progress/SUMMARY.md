# progress/SUMMARY.md — wave 13 (repro_guide_wave) 各 teammate 一行摘要

> 本文件是 **索引**，不复制 progress 全文。
> 若需具体 teammate 报告，请回查源项目对应 wave 的 `progress/teammate-*.md`。
> 源项目根：`/home/junlin12/project_fp8_tp4_repro/repro_guide_wave/`（promote 时点；后续可能迁移）
> 沿用格式：参照 `18_fp8_tp8_root_cause_and_fix/progress/SUMMARY.md`
> 状态：Phase 1 + Phase 1.1 + Phase 2 已闭环（9 teammate）；Phase 3 STAGING + commit 已完成，待 #LEAD-T13 push

---

## Phase 1（调研 + 起草） — `repro_guide_wave/progress/`

| teammate | 角色 | 任务 / 关键发现 / 产出 |
|---|---|---|
| teammate-perf | 调研 throughput 入口 | **任务**：调查 ATOM/vLLM offline-throughput 入口 + 三档命令模板。**关键发现**：ATOM **无** offline-throughput 入口（`benchmark_serving.py` 是 online HTTP，`profile_offline.py` 仅 trace 不报 tok/s）；ATOM `generate()` 输出 dict 已含 `latency / num_tokens_input / num_tokens_output / ttft / tpot`，无须自测。**产出**：推荐方案 A — 基于 `correctness_bench.py` 加 ~30-40 行写 `throughput_bench.py`（time.time 包 generate + 累加 num_tokens_output）；三档参考 (128/128/64)/(1024/1024/64)/(2048/1024/32)，须 `--ignore-eos`。详见 `progress/teammate-perf.md` |
| teammate-draft | 起草 GUIDE + 校验脚本 | **任务**：起草 `REPRODUCTION_GUIDE.md` + 校验/复用 `throughput_bench.py`。**关键发现**：既有 `correctness_eval/throughput_bench.py` 默认参数 `num_prompts=32 / input_len=4096 / output_len=512 / ignore_eos=True` 与 #DRAFT 任务 byte-byte 匹配，复用避免新风险；三仓 commit 实测对齐 TEAM_CONFIG（ATOM 969d564 / aiter f06cdcca5 / CK defd7ad29），原起草 aiter+CK branch 名是猜测需校正。**产出**：在 lead 起草版上 patch 7 处 — REPRO_ROOT 抽象 / aiter `feat/step3p5-moe-swiglustep` / CK `feat/swiglustep-moe-no-quant` 校正 / §1 第 4 步 promote 路径 / §7 9-run for-loop / §8.1 NEW-RC-3 落地路径 / §9 commit 表同步。Caveats C1-C6 记给 #FRESH-VERIFY。详见 `progress/teammate-draft.md` |

## Phase 1.1（并行 review + 优化） — `repro_guide_wave/progress/`

| teammate | 角色 | 任务 / 关键发现 / 产出 |
|---|---|---|
| teammate-review-cmd | GUIDE 命令 review | **任务**：critical review GUIDE 命令 copy-paste 可执行性。**关键发现**：3 BLOCK + 5 WARN + 4 INFO；BLOCK = F1（promote 包缺 `correctness_eval/` + `reverify_wave/outputs/`）/ F2（本机 vs 外部 reader 路径分叉未提示）/ F3（缺 `hf auth login`）；WARN = brace expansion / `declare -A` bash 兼容性 / aiter build chain prereq / commit reachability fallback / snapshot revision pin。**产出**：最低修复集 F1+F2+F3 + 强烈合并修 F1+F12（promote 同时附 NEW-RC-3 patch 文本）。详见 `progress/teammate-review-cmd.md` |
| teammate-review-script | throughput_bench.py review | **任务**：critical review 脚本（CLI / ATOM API / JSON schema / determinism / edge cases）。**关键发现**：0 BLOCK / 5 WARN（F1 ignore_eos default=True 语义陷阱 / F2 determinism 仅 temp=0 + ATOM 无 seed / F4 cudagraph 边界 / F5 input_len=16384×tp<8 OOM near-certain / F10 ATOM 内部 round-trip drift / F12 prefill_/decode_tps 命名误导）+ 6 INFO；ATOM API 用法全合法（generate dict schema 与 `llm_engine.py:253-262` 实测一致）。**产出**：ranked P0-P8 修复清单；最高 ROI = P0（GUIDE §7 跳过 16384×tp<8 OOM + 注释 prefill/decode_tps 实际语义） + P1（命令模板移除 `--ignore-eos`）。详见 `progress/teammate-review-script.md` |
| teammate-review-content | GUIDE 内容/锚点 review | **任务**：critical review 章节完整性 / caveats 充分性 / §6.1 锚点 vs reverify outputs 严格对照。**关键发现**：1 BLOCK（F3 TP8 D<tp_size 双层 fix 历史/§8.7 兜底缺失，commit 969d564 隐式带入但 reader 不知 fix 存在）+ 4 WARN（F1 license / F2 §8 known-issue 矩阵 / F4 port cleanup+§8.6 / F7 `HF_HUB_ENABLE_HF_TRANSFER=0` 反直觉 / F9 §6.1 量化容差缺失）+ 5+ INFO；§6.1 锚点对照 — P0/P3 OK，P1 tp=8 引号差异需放宽，P2 tp=8 reasoning 路径不同需放宽，P3 近义词应扩到 8 词。**产出**：最少 publish-ready 集合 Rank 1-5（约 60 行 patch）；6 个已知坑覆盖矩阵（K1 NEW-RC-3 ✅ / K2 16K OOM ✅ / K3 TP8 fix ❌ / K4 port ⚠ / K5 HF transfer ⚠ / K6 fp8 dispatch ❌）。详见 `progress/teammate-review-content.md` |
| teammate-integrate | findings 整合 + patch | **任务**：整合 3 份 review（cmd / script / content）→ patch GUIDE + 脚本。**关键发现**：4 BLOCK 全修 + 9 WARN 采纳 + 2 INFO 采纳 / 5 WARN + 13 INFO 拒绝（理由全记）；冲突取舍 6 项（OOM 16384 修 GUIDE 端 / ignore_eos 仅改 GUIDE 模板 / brace+declare 合并修 / HF login §3+§8.8 双修 / TP8 fix 区分 ATOM 侧 D<tp_size vs aiter 侧 NEW-RC-3 / per-run cleanup 三处加）。**产出**：GUIDE 净增 ≈145 行（§0.5 license / §1 路径分叉+reachability / §3 hf auth login / §4 HF_TRANSFER caveat / §6 cleanup + §6.1.1 量化容差 / §7 OOM SKIP + 7-run 矩阵 + bash compat + variance/命名 caveat / §8.0 矩阵 + §8.6/§8.7/§8.8 新增），脚本净增 4 行（actual_avg_input_len + variance NOTE + 命名澄清）。备份到 `before_integrate.{REPRODUCTION_GUIDE.md, throughput_bench.py}`。详见 `progress/teammate-integrate.md` |
| teammate-hfcache-patch | §4 micro-patch | **任务**：让本机 reader 复用 `/workspace/hf_cache` 中 stepfun-ai/Step-3.5-Flash-FP8 snapshot 避免重下 ~90 GB。**关键发现**：本机实测 195 GB（含历史 blob，nominal 90 GB）/ 1 个 revision 子目录；shard 数阈值 44 比"总大小阈值"更严格。**产出**：§4 拆为两段 — §4.1 先检测（`ls snapshots/` + shard ≥44 阈值，命中 SKIP）/ §4.2 fallback `snapshot_download`（`cache_dir='$HF_HOME/hub'` 与 §4.1 路径口径对齐）；保留 `HF_HUB_ENABLE_HF_TRANSFER=0` 反直觉说明；§4 行数 ~25 → ~55（净增 ~30）；外部 fresh reader 行为与原 §4 等价。详见 `progress/teammate-hfcache-patch.md` |

---

## Phase 2（fresh 验证 + 修订） — pending

| teammate | 角色 | 状态 |
|---|---|---|
| teammate-fresh-verify | fresh dir 真跑 GUIDE | **任务**：fresh dir (`/tmp/repro_guide_fresh_20260430_031900`) 走 GUIDE §1 路径 (A) (`export REPRO_ROOT=/home/junlin12/project_fp8_tp4_repro` 跳 clone)，10 GPU run = accuracy 3 (tp=2/4/8) + throughput 7 (tp=2 in {4096,8192} / tp=4 in {4096,8192} / tp=8 in {4096,8192,16384})，OOM SKIP 2 (tp=2/4 × 16384)；严格串行 (pkill+sleep 3 间隔)。**关键发现**：12/12 acc prompt PASS（含 tp=8 P2 长路径变体 + tp=2 P1 finish flip）；7/7 thr exit 0；total_tps 单调（4096: 4749→6533→8756；8192: 5222→7636→10744；16384(tp8): 11982）。**6 deviations ranked**：D-B1 (BLOCK) `Engine Core fully initialized` 锚点不存在 / D-B2 (BLOCK) §6.1.1 P1 finish_reason 单点结论 / D-W1 (WARN) §6.1.1 P2 ntoken 表与 §6.1 文字内冲 / D-W2 (WARN) acc output 路径触红线 R5 / D-W3 (WARN) acc/thr mkdir 不对称 / D-I2 (INFO) aiter type hints mismatch 噪音。**红线**：R5 fresh dir 隔离 ✓ / R6 GPU 串行 ✓ / R7 sampling 容差 12/12 PASS ✓。详见 `progress/teammate-fresh-verify.md` |
| teammate-revise | 据 findings 修订 GUIDE | **任务**：按 fresh-verify §5 ranked 清单 patch GUIDE。**关键发现**：6/6 deviations 全修 — D-B1 (BLOCK，§5/§6/§7/§8.7 共 4 处锚点改 `Loading safetensors shards 44/44` + `[OK] dumped JSON`) / D-B2 (BLOCK，§6.1.1 P1 finish_reason 改 `eos | max_tokens` + 脚注) / D-W1 (WARN，§6.1.1 P2 tp=8 ntoken 加容差 + §6.1 跨路径脚注) / D-W2 (WARN，§6 `--output-json` 改 `repro_guide_wave/outputs/acc_tp${tp}.json` + 隔离说明) / D-W3 (WARN，§6 mkdir 归一到 `repro_guide_wave/`) / D-I2 (INFO，§8 加 known-noise 备注)。**产出**：GUIDE 净增 ~35 行（28 898 → 28 933 bytes）；备份 `before_revise_GUIDE.md`；5/5 自验 PASS。详见 `progress/teammate-revise.md` |

---

## Phase 3（lead 收尾 + promote） — pending

| 节点 | owner | 状态 |
|---|---|---|
| #LEAD-T12 决策 | Lead | ✅ 直接 #PROMOTE，不派第二轮（patch 全 wording-only / 命令逻辑未变） |
| #PROMOTE 准备 STAGING + git commit | teammate-promote | ✅ 本次 #PROMOTE 完成；4 文件齐 / git commit 但不 push |
| #LEAD-T13 push 到 LJ-underdog/project_summary main | Lead | pending |
| #LEAD-T14 写 WAVE_CLOSE.md | Lead | ✅ Lead 写完，6 章节，已含入 STAGING |

---

**End of SUMMARY — 2026-04-30 / wave 13 / teammate-promote（Phase 1+1.1+2 闭环 9 teammate；STAGING 4 文件齐；待 #LEAD-T13 push）**
