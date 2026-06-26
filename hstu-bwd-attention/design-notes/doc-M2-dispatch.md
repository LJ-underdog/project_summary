# 派单:写 M2 讲义(HSTU 5 因子 mask)

**先读硬规格:`/tmp/hstu-bwd-design/doc-series-spec.md`(全文)。本卡只给 M2 专属输入。**

- **里程碑**:M2 = causal + HSTU 5 因子 mask(window / contextual / min_full / num_target + 组合),SiLU+batched+bf16+hd64。
- **commit(行号锚定它)**:`9d129c88`。先 `cd /root/workspace/ck_hstu && git show --stat 9d129c88`。
- **旧 HTML(可参考叙事,行号/数值必重核)**:`/root/workspace/hstu-b1052-report/hstu-bwd-M2-changes-20260605.html`。
  - ⚠ 已知:旧讲义把后来 P1-1 bug 的构造**夸成了特性**(stale,见 HANDOFF YELLOW)。你写 M2 时**不要**把 causal=0+num_target 的处理说成已正确——M2 时点它其实有 P1-1 静默 bug,M4b 才修。诚实写「M2 时点的 mask 覆盖范围」,P1-1 留给 M4b 讲义。
- **事实来源**:`/tmp/hstu-bwd-design/M2-done.md` + `candidates.jsonl` 里 `"id":"M2-mask"` 那条(含 185932 checks 离线校验器 GREEN、逐因子对拍 PASS、harness num_targets supplement bug 修复)。
- **输出**:`/root/workspace/hstu-b1052-report/hstu-bwd-M2-mask-20260625.html`
- **M2 讲解重点**:
  1. 4 个 HSTU mask 结构体新增成员 `GetTileRangeAlongY`(保守 (0,seqlen) 超集 + fallback)+ `IsEdgeTile`(=`!IsFullTileInsideMask`)——**纯加,不改 fwd**。
  2. pipeline STAGE2:edge-tile `set_tile_if(p,g<-0, !IsTokenPairInsideMask)`。
  3. kernel 构造 HSTU mask(`is_tile_in_first_split=true` 保守)via `make_hstu_self_*`;dispatch 按 `(kUseCausal, kUseLocal=window>0)` 选 mask 类型。
  4. **离线 superset 校验器** `test/validate_tile_range_y.cpp`(185932 checks GREEN)——为什么需要离线穷举 gate(对拍可能漏的 silent-wrong)。
  5. harness num_targets supplement 到 num_batch 的 OOB bug(挖到并修)。
- **易错提示**:5 因子 mask 的几何(causal/window band/contextual/min_full/num_target)适合配 1-2 张 SVG 示意。
- 写完按规格 §6 回报。
