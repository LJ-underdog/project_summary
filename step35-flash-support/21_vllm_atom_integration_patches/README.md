# 21_vllm_atom_integration_patches

step35-flash-support 系列第 21 篇 — vllm + ATOM 双 source-level patch 文档化。

## 任务背景

step3p5 model 跑 vllm + ATOM 集成栈时, 有 2 个 source-level patch 应用; 本文档记录 patch 完整证据链 (motivation / change / verification / caveat / timeline) 并诚实标注与 wave G-3(i) NaN 问题的解耦关系。

## 文档地图

- [00_overview.md](./00_overview.md) — 任务背景 + 2 patch 一览表 + 共同 caveat
- [01_patch_swiglustep.md](./01_patch_swiglustep.md) — Patch A: vllm `rocm_aiter_fused_moe.py` SwigluStep enum 加白
- [02_patch_swa_perlayer.md](./02_patch_swa_perlayer.md) — Patch B: ATOM `attention.py` SWA workspace per-layer KV head 修复
- [03_lessons.md](./03_lessons.md) — 5 教训 (self-report / dispatch≠数值 / caveat-strip 防御 / .bak audit / 真 P0 root cause)

Reviewer GPA 4.4/5 PASS (DOC1-D isolated review, 0 P0 / 2 P1 非阻塞)。
