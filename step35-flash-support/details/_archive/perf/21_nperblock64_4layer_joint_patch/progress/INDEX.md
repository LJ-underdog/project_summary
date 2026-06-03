# progress/ INDEX (引用三 wave artifacts，不复制)

本 entry 21 是 **三 wave 整合产出**，原 wave artifacts 仍在 `/home/junlin12/m1quad_*_wave/` 各自路径下；本 doc 不复制原文，仅列路径 + 一句话定性，作为子目录可发现性 stub（与 entry 20 `progress/INDEX.md` 平行）。

| Wave | 路径 | 一句话定性 |
|------|------|----------|
| Joint-fix correctness | `/home/junlin12/m1quad_joint_fix_wave/` | 4 层 patch 落地（CK 105+/15- + caller 5 行）+ 三点 NPerBlock=64 (192/320/448) bit-exact 实证；含 `WAVE_CLOSE.md` / `IMPL_VERIFY_REPORT_v2.md` / `CROSS_CONFIG_VERIFY.md` / `PATCH_SPEC.md` / `patches/applied_ck.patch` / `progress/` |
| Kernel-level perf | `/home/junlin12/m1quad_perf_compare_wave/` | pad (NPerBlock=128) vs nopad (NPerBlock=64+4 层 patch) kernel timing；avg(median) 77.118 vs 64.932 us → 15.80% delta；含 `PERF_COMPARE.md` / `PERF_{PAD,NOPAD}_RESULT.md` / `scripts/perf_kernel_{pad,nopad}.py` |
| Model-level e2e perf | `/home/junlin12/m1quad_model_perf_wave/` | F-A 同 HEAD `f06cdcca5` 唯一变量 CK patch on/off → PERF_NEUTRAL HIGH；含 `WAVE_CLOSE.md` / `PERF_PAD_CURRENT_MODEL_RESULT.md` / `MODEL_PERF_COMPARE.md` / `BASELINE_RECON.md` / `PERF_NOPAD_MODEL_RESULT.md` / `progress/` |
| 上游 partial root cause | `/home/junlin12/m1ppp_padding_explanation_wave/` | H_alt_1 partial mechanism 提出（`PADDING_EXPLAINED.md §3`），sanity coverage 仅 83% |
| Candidate verify | `/home/junlin12/m1ppp_ck_bscale_fix_verify_wave/` | candidate A (仅 L1) / B (仅 caller reshape) FAIL → joint 必要性反证（`M1PPP_FIX_VERIFY_RESULTS.md`）|
| 本 entry 整合 wave | `/home/junlin12/m1quad_project_summary_wave/` | reader 4 个 progress (joint-fix / perf-compare / model-perf / cross-cut) + recon + synth + reviewer + synth-v2（本 doc） |

详细分工 / TEAM_CONFIG / phase-by-phase teammate progress 见各 wave 自身 `TEAM_CONFIG.md` + `progress/` 目录；本 INDEX 仅作 entry 21 子目录可发现性入口（与 entry 20 / entry 16 子目录格式平行）。
