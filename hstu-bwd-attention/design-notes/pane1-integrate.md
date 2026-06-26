# 派给 pane-1(角色:integrator)— 整合 HSTU bwd GPU 设计三件套为单一方案

调度模式:tmux pane-1。把三份设计整合成一份连贯、自洽、无矛盾的实现方案文档。不要派 sub-teammate。

## 输入(全部 Read)
- `/tmp/hstu-bwd-design/BRIEF.md`(共享基线)
- `/tmp/hstu-bwd-design/part-algo.md`(算法&pipeline,你自己写的)
- `/tmp/hstu-bwd-design/part-mask-modes.md`(mask/模式/参数)
- `/tmp/hstu-bwd-design/part-engineering.md`(工程落地&验证)
- 三份 done 概要(`*-done.md`)里列了各自的"未决"和"接口假设"

## 输出
`/tmp/hstu-bwd-design/DESIGN.md` —— 一份完整、可交付 review 的 HSTU bwd GPU 实现方案。

## 整合要求(重点是消解跨组矛盾,不是简单拼接)

1. **统一结构**(建议章节):
   - §0 摘要 + 目标芯片(gfx942/gfx950)+ 复用 FMHA bwd 的总策略一句话
   - §1 总体架构:3-kernel(PRE/MAIN/POST)+ KV 外/Q 内 + grid + 数据流
   - §2 算法与七阶段(SiLU/softmax 双路)+ 关键骨架伪代码(取 part-algo)
   - §3 mask / 三模式 / 索引 / 参数结构(取 part-mask-modes)
   - §4 工程落地:文件结构(新建 vs 复用 FMHA 映射表)、problem/policy、dispatch、instances、dQ 写回、CMake(取 part-engineering)
   - §5 正确性验证:CPU reference 对拍流程、容差、测试矩阵、deterministic
   - §6 分阶段里程碑 M0–Mx + 每阶段验收(以 M1 batched+SiLU+no-mask 为风险闸门)
   - §7 复用 vs 新写总清单
   - §8 风险与未决问题(分"已决/建议"与"真未决,需用户拍板")

2. **必须逐条裁决这些跨组点(给单一一致结论 + 理由;真无法定的标为"需用户/实测拍板"):**
   - **mask Y 方向能力缺口**:pane-2 指出 HSTU mask 缺 `GetTileRangeAlongY`/`IsEdgeTile`(bwd 必需),方案 A=新增成员(纯加不改 fwd)。确认采纳 A,并和 pane-1 的"STAGE 置零依赖 block-tile mask 谓词"对齐成一套谓词接口。
   - **`is_tile_in_first_split`**:reference 恒传 true,`!first_split` 分支在 bwd 怎么处理(pane-1 R?/pane-2 §5.5)——给统一结论或标未决。
   - **SiLU 留 g(dsilu 因子)vs 留整张 S**:pane-1 选留 g 以与 FMHA 寄存器同形;pane-3 要在真实 tile 上验 VGPR 压力(R1)。统一表述为"设计选 g,M1 阶段实测验证"。
   - **masked-out 显式置零**:STAGE1/STAGE5 共用同一布尔 mask tile(`IsTokenPairInsideMask`),SiLU 必清 g(dsilu(0)=0.5),禁 -inf。确认口径一致。
   - **scale 接线**:`alpha`(STAGE2 头 + dQ/dK 收尾,dV 不吃)+ `scale_p`(折进 p 与 g);softmax 用 exp(S−LSE) 自然对数域,**别在 exp2 再乘 scale**(高危 bug)。固化成一节"scale 语义与接线表"。
   - **group 逐段 alpha/scale_p/mask 超参取数**:device 指针 + `i_group` 索引,统一归口。
   - **kargs 字段名 / problem 模板形参 / mask 谓词签名**:三方接口假设若有出入,统一成一套命名(给最终字段表)。

3. **标注矛盾**:若三份之间有事实/决策冲突(而非仅措辞),在 §8 显式列出"冲突点 X:partA 说…partB 说…裁决…"。

## 铁则
- 这是**设计方案**,不是最终 kernel;保留关键接口/伪代码,别灌满整份实现。
- 事实/行号以源码为准(reference + FMHA + HSTU fwd);不臆造。
- markdown 写到 `/tmp/hstu-bwd-design/DESIGN.md`;progress 简洁;完成写 `/tmp/hstu-bwd-design/integrate-done.md`(裁决了哪些跨组点、剩几个真未决、字数)。
