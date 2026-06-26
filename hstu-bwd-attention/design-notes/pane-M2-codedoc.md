# 派给 pane-2(角色:drafter)— M2 代码改动 review 文档(图文并茂 HTML)

调度模式:tmux pane-2。把 **M2(HSTU mask)的代码改动**写成图文并茂、可 review 的 HTML。**忠实真实代码/diff**,落笔前用 git + Read 核实,不臆造。不要派 sub-teammate。

## 输出
`/root/workspace/hstu-b1052-report/hstu-bwd-M2-changes-20260605.html`(与其它报告同目录)

## 样式
复用 `/root/workspace/hstu-b1052-report/hstu-bwd-impl-status-20260604.html` 的整套 `<head>`/CSS + 左正文右 sticky TOC。配色 clay=新增/改动、olive=复用、slate 文字。代码用 `.code-block`(可加简单高亮)。

## 取材(以真实代码为准)
代码库 `/root/workspace/ck_hstu`(git 仓库)。M2 背景报告:`/tmp/hstu-bwd-design/M2-done.md`(权威改动清单)。
- **mask 成员改动(已跟踪文件,可 git diff)**:`git diff example/ck_tile/18_hstu_attention/hstu_block_masking.hpp`(看 `GetTileRangeAlongY`/`IsEdgeTile` 在 4 个 struct 的新增)。也 `git diff hstu_attention_api.hpp CMakeLists.txt generate_instances.py`。
- **bwd 文件是新增(未跟踪,无 git 基线)** → M2 在其中的改动用 **M2-done.md + 直接 Read 当前代码区段**呈现,标注"M2 新增/改动":
  - `hstu_attention_no_softmax_bwd_pipeline.hpp` 的 **STAGE2 置零**(`if constexpr(FmhaMask::IsMasking)` + `set_tile_if(p/g,0, !IsTokenPairInsideMask)`)
  - `hstu_attention_bwd_kernel.hpp` 的 **mask 构造**(kargs 加 num_targets/contextual/max_attn_len/min_full;`make_hstu_self_attention_block_mask_{with,without}_local`;`is_tile_in_first_split=true`;min_full 钳制)
  - `hstu_attention_batched_backward_dispatch.hpp` 的 **去 causal throw + `BOOL_SWITCH(window>0)` 选 HstuBlockMasking**
  - `example_hstu_attention_bwd.cpp` 的 **num_targets supplement 修复**
- **离线校验器**:`test/validate_tile_range_y.cpp`(新增)+ 结果 `runs/validate_tile_range_y.log`(185932 checks ALL GREEN)。
- **测试套件更新**:`test/run_bwd_tests.py`(reject-causal→8 个 M2 pass case);结果 20/19PASS/1skip/exit0。

## 文档要讲清(核心:M2 改了哪些 code、为什么、怎么连)
1. **总览 + TL;DR**:M2 = 复用现成 mask 语义(`IsTokenPairInsideMask` 5 因子)+ 补 bwd 方向 helper + 接线;**没重新实现 mask**(强调这点)。徽章:16/16 因子档对拍 PASS / 离线校验 185932 全绿 / 测试套件 20 案 exit0。
2. **改动文件清单表**:逻辑改动文件(mask/pipeline/kernel/dispatch/harness/validator/test)+ 每个的角色与 M2 改了什么。**单列一行诚实标注**:`generate_instances.py` 重生成导致 ~190 个 fwd instance 文件被改写(每个仅 2 行,**非 M2 逻辑、属再生成噪音**,提醒 review 者别被 git status 的 202 文件吓到)。
3. **【图1】mask 复用 vs 新增**:olive=复用(`IsTokenPairInsideMask` 5 因子 / `GetTileRangeAlongX` / `IsFullTileInsideMask`)vs clay=M2 新增(`GetTileRangeAlongY` / `IsEdgeTile`,4 个 struct)。
4. **mask 成员新增(git diff 呈现)**:`GetTileRangeAlongY` 首版=保守全 Q 扫描 `(0,seqlen_q)`(正确性靠逐像素置零,紧致化=M8 perf);`IsEdgeTile=!IsFullTileInsideMask`(注意参数序)。贴关键 diff 片段。
5. **【图2】masked-out 置零数据流**:STAGE2 算完 p,g → edge tile 上 `set_tile_if(p,g←0, !IsTokenPairInsideMask)` → g=0 使 STAGE5 `ds=dp·g=0` → dV/dK/dQ 不被污染。强调 SiLU 必须真清(silu(0)·scale_p=0 但 **dsilu(0)=0.5≠0**,禁 -inf)。贴 STAGE2 代码片段。
6. **dispatch 接线**:去 causal throw → `BOOL_SWITCH(window>0)` 选 `HstuBlockMasking<false,causal,local>` → kernel 构造 mask(num_target per-batch、is_tile_in_first_split=true 保守、min_full 钳制)。贴片段。
7. **离线校验器**:作用(断言 Y-range ⊇ 真值集)、覆盖枚举、185932 全绿——为什么这是信任 GPU mask 的前置。
8. **抓修的 bug**:harness `num_targets` 未 supplement 到 num_batch → 越界读 `num_targets_ptr[1]` → batch1 mask 全错 → num_target 档 FAIL;补 supplement 后 PASS(同源问题 reference 端也有)。这是 M2 的一个真实发现,值得单独讲。
9. **验证结果**:16/16 逐因子对拍表(causal/window/contextual/min_full/num_target/组合,误差 bf16 级)+ 测试套件 20 案 exit0 + 离线校验。
10. **遗留**:Y-range 现保守全扫(perf,M8 紧致化);cross mask 成员已加但 bwd 暂只构造 self。

## 铁则
- 忠实代码/diff;区分"M2 逻辑改动"与"instance 再生成噪音"(后者必须诚实标注,别混入"成果")。
- 中文为主、字段/文件名英文;零基础友好(关键术语一句白话)。
- 自检 HTML 标签平衡(div/section/svg/h2/h3/table)+ TOC 锚点 + SVG(≥2)不溢出。
- 完成写 `/tmp/hstu-bwd-design/M2-codedoc-done.md`:小节/SVG/字节/标签平衡/覆盖的改动文件。正文写进 HTML,不在终端长输出。
