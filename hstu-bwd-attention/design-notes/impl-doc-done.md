# impl-doc 完成 (pane-3 / drafter)

状态:**✅ 完成**。HSTU bwd 实现现状 HTML 已落,图文并茂、忠实当前真实代码(落笔前逐文件 Read)。

## 输出
`/root/workspace/hstu-b1052-report/hstu-bwd-impl-status-20260604.html`
- 字节:**52231 B**
- 样式:复用 design HTML 整套 <head>/CSS + 左正文右 sticky TOC + clay(新写)/olive(复用) 配色

## 小节 (10 节,与派单一一对应)
1. 总览 — M0 脚手架 + M1 闸门已通过;TL;DR 三徽章(M1 PASS / R1 零覆写 / R2 VGPR248·AGPR0·Scratch0·occ2)+ 4 枚 stat
2. 文件清单表 — 每文件路径/行数/角色/新写改动/复用 FMHA(11 行)
3. 【图1 SVG】include/依赖关系
4. 【图2 SVG】运行时调用链/数据流(GPU vs CPU 双列)
5. 【图3 SVG】MAIN 七阶段 + 代码佐证 code-block
6. 复用 vs 新写 — 6 callout 卡 + 为何 pipeline 是改写
7. 关键工程决策 — 4 决策卡(①ck_tile 版本 ②自写 kernel ③float dq_acc+atomic+POST ④NO_BIAS dummy)+ R1/R2 gate-banner
8. 覆盖面 vs TODO 表 — 13 行(✅6 / ⏳7,标里程碑 M2-M8)
9. 验证现状 — oracle/数值结果/6-case 稳定性/R2 资源 statgrid
10. 构建/运行 — cmake + 运行命令(强调 -attn_scale=1.0)

## SVG 数:3(图1 依赖图 / 图2 调用链 / 图3 七阶段),均 viewBox + text 不溢出

## 自检(全 OK)
- 标签平衡:div 93/93 · section 10/10 · svg 3/3 · h2 10/10 · h3 9/9 · table 2/2 · thead/tbody 2/2 · tr 26/26
- TOC 锚点:#s1–#s10 与 section id 一一对应
- 无 stray replacement char(已修一处 §7 乱码)

## 覆盖的文件(逐个 Read 后落笔,行数实测 wc -l)
bwd_params.hpp(119) · no_softmax_bwd_pipeline.hpp(518) · bwd_kernel.hpp(376) · batched_backward_dispatch.hpp(233) · no_group_backward_bf16.cpp(33) · example_hstu_attention_bwd.cpp(483) · api.hpp(22) · reference_hstu_attention_bwd.hpp(855) · instances/..._ref.hpp(42)+4 实例 · generate_instances.py(bwd 分支) · CMakeLists.txt(bwd target)
背景报告:M0-done.md / M1-done.md(数值+R1/R2 结论引自此)

## 铁则遵守
- 忠实当前代码;区分"已实现"(M0+M1 真实路径)与"TODO"(M2-M8 throw 门控)
- 中文为主、字段/文件名英文;关键术语配白话(闸门/KV-resident/dq_acc/atomic/OOB)
- 正文全写进 HTML,终端无长输出
