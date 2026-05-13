# logs/ INDEX (引用 wave2_archive，不复制)

raw log 全部在 wave2_archive：`/home/junlin12/step35_fp8_optim_wave2/logs/`

本 doc **不复制** raw log（多个百 KB 级），仅列主要类别 + 引根目录。

## 主要 log 类别

| 类别 | 路径前缀 | 内容 |
|------|---------|------|
| Phase 0 baseline | `phase0_baseline_tp2*.log` | wave 启动 baseline（tp=2） |
| c01 dispatch + tune | `c01_dispatch_verify.log` / `c01_tune_sweep*.log` / `c01_tune_sweep_v1_e4m3fn_FAILED.log` | dispatch verify + tuning sweep（含 e4m3fn FAILED 实证） |
| v01 baseline / tuning | `v01_baseline_tp{2,4,8}*.log` / `v01_tuning_tp{2,4,8}*.log` | v01 9 metric np=1 矩阵 raw |
| v03 baseline / tuning | `v03_baseline_tp{2,4,8}_np{4,8}*.log` / `v03_tuning_tp{2,4,8}_np{4,8}*.log` | v03 18 metric multiprompt 矩阵 raw（含 aiter bug crash trace） |

`*_full.log` 为完整 stderr/stdout（含 crash trace），裸 `*.log` 为 summary 摘录。aiter `apply_act_and_mul` crash trace 在 `v03_*tp{4,8}_np{4,8}_full.log` 中（详见主报告 `RESULTS.md §4`）。
