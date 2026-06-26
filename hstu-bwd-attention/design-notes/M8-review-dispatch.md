# M8 perf 对抗式 review —— reviewer(pane 0.2,已 /clear)

独立对抗 review HSTU bwd **M8 perf(MI + B2 + B3)**,默认怀疑、不信自述。基线 HEAD=`4629508f`(cross),M8 未 commit(9 文件 working-tree 改)。素材:`docs/draft-M8-perf.md`(顶部闸门裁决:scope=MI+B2+B3)、`docs/M8-done.md`、`/tmp/hstu-bwd-design/M8-{MI-stage1,B2}-done.md`。

## 独立 build:`rm -rf build_review` 重配重编(BUILD_DEV=OFF,gfx950);group entry ~14min TU,build 久。

## 对抗审查清单(逐条 GREEN/RED + 证据)
1. **scope**:`git diff 4629508f --stat` = 9 文件(perf.hpp 新 / params / batched+group dispatch / harness / block_masking / validator / run_perf_baseline / benchmark.csv);**reference + pipeline byte-identical**。
2. **★ MI behind-flag 零回归(byte-level)**:不给 `-perf` 时,你 build_review 后 `co_symbols.py verify` self 设备符号 byte-identical?(coder 报 13782/13782;你自产基线复核)。套件 253/253 独立复跑?perf 字段确不进 MakeKargs?
3. **★★ 离线 superset 校验器(B2/B3 silent-wrong 最硬 gate)**:独立编译+跑 `test/validate_tile_range_y.cpp` → **ALL GREEN**(coder 报 1,973,278 checks)?**反证**:临时把某收紧 y_start 改激进(如去掉 contextual 的 y_start=0,或去掉 min_full 下沉)→ 校验器应 **报 failure**(证它能判伪、非 vacuous)。验完恢复(独立 worktree/cp,别 git checkout 带改动文件,守 N3 教训)。注:host 编译该校验器需 `-D__HIP_PLATFORM_AMD__` + 可能要解 ck_tile math redefinition(查 coder 怎么编的 / runs 日志)。
4. **B2/B3 对拍正确(under-tighten 经验验)**:独立跑 causal/window × {contextual, min_full, num_target, cross kv>q/kv<q, 非整除} 对拍 PASS;尤其**校验器抓的 2 个 bug 配置**(非causal window+min_full、cross causal 大 diff window+contextual)对拍 PASS。
5. **加速实证 + 诚实**:-perf 跑 causal + window 各档,MAIN ms vs MI 基线 → causal 1.25–1.60×、window 4.7–9.8× 复现?Amdahl 归因(实测<模型)诚实?容差未松、TFLOPS 是 GEMM-only tracking 不当 roofline?
6. **co_symbols surgical**:DIFF 仅落 GetTileRangeAlongY 的 bwd MAIN 实例;fwd(用 GetTileRangeAlongX)+ kentry + no-affected 路 byte-identical?
7. **套件 253/253 独立复跑**(build_review binary)。
8. **暂缓项诚实**:B4(grid 根因证伪)、B7/VGPR、B1/B5/B6/INV 如实标暂缓,无夸大本期成果。

## 产出
写 `/tmp/hstu-bwd-design/M8-review-findings.md`,逐条 GREEN/RED + 证据;RED 给复现。结论:可否 promote。完成 pane 报。
