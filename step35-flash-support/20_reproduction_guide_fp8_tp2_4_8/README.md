# 20_reproduction_guide_fp8_tp2_4_8 — Step-3.5-Flash-FP8 在 tp=2/4/8 三档的端到端复现指南（accuracy + throughput）

> 日期：2026-04-30
> 来源 wave：fp8-tp4-repro / repro_guide_wave (wave 13)
> 状态：⏳ staging（README 提前归位；REPRODUCTION_GUIDE.md 等 fresh-verify 完成后再随 #PROMOTE 拷入）
> 红线遵守：本目录仅整理文档（无源码改写、无 GPU 跑、无其他 wave 文档改）

---

## 1. 这是什么

一份 **step-by-step reproduction guide**，让一个**已有 ROCm docker + 8×MI308X 节点**的新接手者，通过 git clone 三仓 + 按文档命令一步步执行，在 tp=2/4/8 三档分别得到：

1. **Accuracy**：`correctness_eval/correctness_bench.py` 4 prompt 全部 coherent / 与 `reverify_wave/outputs/tp{2,4,8}.json` 同质量（P2 给 6、P3 中文"不现实/不健康/不可能"近义词、P0 introduce-myself、P1 prime numbers）
2. **Throughput**：`correctness_eval/throughput_bench.py`（基于 `time.time()` 包 ATOM offline `generate` 的 ~150 行脚本）三档 tok/s + req/s 数据

并由一个 **reviewer teammate 在 fresh working dir 真跑一遍** 验证文档可落地（fresh-verify 完成后会更新本 README 状态为 ✅）。

**主交付**：`REPRODUCTION_GUIDE.md`（含 §0 Prerequisites / §0.5 Licensing / §1 三仓 clone + 精确 commit / §2 aiter build / §3 HF auth + cache / §4 env vars / §5 sanity / §6 accuracy 三档 + §6.1 输出锚点对照 / §7 throughput 三档 + OOM 矩阵 / §8 known-issue + cleanup + TP8 D<tp_size 历史背景）

---

## 2. 文件清单（promote 完成后最终形态）

| 文件 | 用途 | 备注 |
|---|---|---|
| `REPRODUCTION_GUIDE.md` | **主交付**（user-facing，~600 行端到端步骤）| 等 fresh-verify + #REVISE 完成后由 #PROMOTE 拷入 |
| `WAVE_CLOSE.md` | wave 13 关键交付摘要 | #LEAD-T14 产出 |
| `progress/SUMMARY.md` | 各 teammate 一行摘要（不拷全部 progress 文件，仅索引）| stage-summary teammate 产出 |
| `README.md` | 本文件 | ✅ 已就位 |

> 当前状态：本 README 提前归位（wave 13 #PROMOTE 前期 staging）；`REPRODUCTION_GUIDE.md` 仍在 `repro_guide_wave/` 内供 fresh-verify teammate 使用，确认 publish-ready 后再拷入本目录。

---

## 3. 与 17 / 18 的关系

| topic | wave | 阶段 | 状态 |
|---|---|---|---|
| `17_atom_moe_tp8_load_crash/` | issue_wave (wave 6) | crash 现象 + 推测修复 A/B 草稿 | ⏸ 仅 issue draft |
| `18_fp8_tp8_root_cause_and_fix/` | fix_wave (wave 7-8) + doc_wave (wave 9) + topdoc_update_wave (wave 10) + promote_wave (wave 11) | tp=8 双层根因 + 双层 fix（ATOM `969d564`）+ 三档 verify + user-facing 主文档 | ✅ 已 push 上游 |
| `20_reproduction_guide_fp8_tp2_4_8/` (本 topic) | repro_guide_wave (wave 13) | 把 18 的 fix 转化为**外部接手者可执行的端到端命令脚本**（accuracy + throughput）| ⏳ staging |

读者推荐顺序：

- 想知道**为什么 tp=8 之前会崩 + 怎么修的** → 先读 18
- 想**自己在新机器上从零跑通 tp=2/4/8 三档** → 直接读 19（本目录 `REPRODUCTION_GUIDE.md`）；§8.7 会把 18 的 D<tp_size 修复历史背景作 caveat 引用

19 = 18 修复落定 + reverify_wave (wave 12) 三档 PASS 后，把整套复现路径**外化**给非项目成员。

---

## 4. 三仓上游 commit（reproduction guide 钉住的版本）

| 仓 | commit | 远程分支 |
|---|---|---|
| **ATOM** `github.com/ROCm/ATOM` | `969d564` "fix(moe): handle D < tp_size in fp8 _load_w13/_load_w2" | `feat/step3p5-flash-support` |
| **aiter** `github.com/ROCm/aiter` | `f06cdcca5` "fix(moe): force per_1x128 fp8 blockscale to CK 2-stage on gfx942" (NEW-RC-3) | `feat/step3p5-moe-swiglustep` |
| **CK** `github.com/ROCm/composable_kernel`（aiter `3rdparty/` 子模块）| `defd7ad29` "Add swiglustep_and_mul branches to gridwise_moe_gemm" | `feat/swiglustep-moe-no-quant` |

三仓全部 pushed（wave 13 起始时实测，详见 `REPRODUCTION_GUIDE.md` §1）。

---

## 5. 引用源（项目内文档原始路径，仅 promote 学习用）

| 源文档 | 来源 wave | 用途 |
|---|---|---|
| `repro_guide_wave/REPRODUCTION_GUIDE.md` | repro_guide_wave (wave 13) | 主交付（本目录同名文件 promote 时从此处 byte-id 拷贝） |
| `repro_guide_wave/throughput_bench.py` | repro_guide_wave (wave 13) | throughput 入口脚本（基于 `correctness_bench.py` 加 `time.time()` 包 generate） |
| `repro_guide_wave/progress/teammate-perf.md` | repro_guide_wave (wave 13 #PERF) | ATOM 无 offline-throughput 入口的调研结论 + 三档 (input/output/num) 配置依据 |
| `repro_guide_wave/progress/teammate-draft.md` | repro_guide_wave (wave 13 #DRAFT) | REPRODUCTION_GUIDE.md 起草过程 + 7 处 patch 记录 |
| `repro_guide_wave/progress/teammate-review-{script,cmd,content}.md` | repro_guide_wave (wave 13 Phase 1.1) | 三路 critical review（脚本 / 命令可执行性 / 章节完整性）findings |
| `repro_guide_wave/progress/teammate-integrate.md` | repro_guide_wave (wave 13) | 4 BLOCK 全修 + 9 WARN + 2 INFO 整合记录（净增 ~145 行） |
| `repro_guide_wave/progress/teammate-hfcache-patch.md` | repro_guide_wave (wave 13) | §4 本机 reader `/workspace/hf_cache` 复用 micro-patch |
| `reverify_wave/outputs/tp{2,4,8}.json` | reverify_wave (wave 12) | accuracy 锚点（4 prompt × 3 tp 档 PASS 输出） |
| `18_fp8_tp8_root_cause_and_fix/TP8_ROOT_CAUSE_AND_FIX.md` | promote_wave (wave 11) | §8.7 caveat 引用：tp=8 D<tp_size 修复历史 |

---

**End of README.**
