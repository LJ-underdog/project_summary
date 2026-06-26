# M2 代码改动 review 文档 — 完成 (pane-2 / drafter)

输出:`/root/workspace/hstu-b1052-report/hstu-bwd-M2-changes-20260605.html`
状态:**完成**。忠实代码/diff,落笔前用 git diff + Read 核实,未派 sub-teammate。

## 自检
- **字节**:53697 bytes
- **标签平衡**:div 111/111 · section 10/10 · svg 3/3 · table 2/2 · tr 19/19 · nav/main/head/body/style 全 1/1 · ul/tbody/thead 全配对 —— 全部平衡
- **TOC 锚点**:10 个 href 全部命中 section id(s1–s10),0 缺失
- **SVG**:3 张(要求 ≥2),均 viewBox + width=100% 防溢出
- **代码转义**:42 处 &lt;/&gt;/&amp; 转义,grep 无 raw 模板尖括号泄漏
- **样式**:完整复用 impl-status HTML 的 head/CSS + 左正文右 sticky TOC;clay=新增/改动、olive=复用、slate 文字;.code-block 带 kw/cm/fn/hl/str 高亮 + 新增 .ttl 片段标题、.tag-noise 噪音徽章

## 小节(10 节,3 SVG)
1. 总览:M2 没重写 mask,只补 helper+置零+接线;统计卡 + 4 件事 + 「202 文件噪音」预防针
2. 改动文件清单表:逐文件标注 M2 逻辑 vs 噪音;**诚实单列** instances/*forward* ×198 = 再生成噪音
3. 【图1 SVG】mask 复用(IsTokenPairInsideMask 5因子/GetTileRangeAlongX/IsFullTileInsideMask) vs M2 新增(GetTileRangeAlongY/IsEdgeTile ×4 struct)
4. mask 成员新增 **git diff 呈现**(SelfNoLocal 真实片段)+ 3 个设计要点(保守全扫/参数序/HOST_DEVICE)
5. 【图2 SVG】masked-out 置零数据流 + STAGE2 真实代码;SiLU 必清(silu(0)=0 但 dsilu(0)=0.5≠0 禁 -inf)
6. 【图3 SVG】dispatch(去 throw+BOOL_SWITCH)+ kernel(num_target per-batch/first_split=true/eff_min_full 钳制)真实片段
7. 离线校验器:断言式 + 枚举覆盖 + 185932/0 全绿(信任前置)
8. 抓修的 bug:harness num_targets 漏 supplement→越界→batch1 全错;含修复代码 + reference 同源说明
9. 验证结果:sweep 因子表(忠实标 15 OK 行/0 FAIL,非套用 16/16)+ 测试套件 20/19/1skip/exit0 + 离线校验
10. 遗留:Y-range 保守(M8 perf)/cross 成员已加但只构造 self/覆盖面边界

## 覆盖的改动文件(均核实)
- `hstu_block_masking.hpp`(git diff,4 struct ×2 成员)— 唯一含 M2 mask 逻辑的已跟踪文件
- `hstu_attention_no_softmax_bwd_pipeline.hpp`(STAGE2 ~L408–430,Read)
- `hstu_attention_bwd_kernel.hpp`(mask 构造 ~L331–360,Read)
- `hstu_attention_batched_backward_dispatch.hpp`(Run ~L195–223,Read)
- `example_hstu_attention_bwd.cpp`(supplement fn ~L72–82+L157–162,Read)
- `test/validate_tile_range_y.cpp`(断言 ~L41–63)+ `runs/validate_tile_range_y.log`(185932/0)
- `test/run_bwd_tests.py` + `runs/test-20260605-063647.log`(TOTAL 20/PASS 19/SKIP 1/exit0)
- `runs/run-bwd-M2-sweep.log`(grep -c '^OK'=15,0 FAIL)

## 核实出的与背景报告的差异(忠实代码为准)
- 噪音文件数:实测 **198**(git diff --stat instances/ = 198 files/+198/−198,每文件恰 1 行注释路径),非 ~190;噪音是 **1 行**(comment path `composable_kernel/…`→`ck_hstu/…`),非 2 行
- git status 总改动 = 202(=198 fwd instance + 4 非 instance:mask/cmake/gen/api),与背景报告「202 文件」吻合
- sweep 日志实为 **15 条 OK 行**(14 掩码 + 1 no-mask 回归),文档忠实标注此数,未直接套用 M2-done 的「16/16」措辞
- api.hpp/CMakeLists.txt/generate_instances.py 经核实为 **bwd 脚手架(M0 期)**,非 M2 mask 逻辑 → 文档据实归类(generate_instances.py 标为 198 噪音之源)
