# 派给 pane-1(持久角色:architect)— HSTU bwd GPU 方案 · 第 1 部分「核心算法 & pipeline 设计」

调度模式:tmux pane-1。与 pane-2/pane-3 **并行**各设计一部分。先读 `/tmp/hstu-bwd-design/BRIEF.md`(共享基线)。不要派 sub-teammate。

## 你负责:把 reference 的反向数学映射成 GPU pipeline(复用 FMHA bwd kr_ktr_vr 7-stage 体系)

产出 `/tmp/hstu-bwd-design/part-algo.md`,涵盖:

1. **3-kernel 总体结构决策**:沿用 FMHA 的 PRE(dot_do_o)→ MAIN(dq_dk_dv kr_ktr_vr)→ POST(convert_dq)?论证哪些直接复用、哪些要 HSTU 特化。特别说明:**SiLU 路径不需要 PRE(无 D)**,只有 softmax 路径才跑 dot_do_o——是否做成编译期分支(kUseSoftmax)跳过 PRE。
2. **MAIN pipeline 七阶段的 HSTU 改造**(逐阶段对照 FMHA):
   - STAGE1 Q@K → S:用 `alpha` 替代 1/√d(注意 alpha 在哪一步乘进去,reference 是 S=alpha·Q·Kᵀ)。
   - STAGE2 重算激活:**双路**——SiLU(`P=silu(S)·scale_p`,无 LSE,需保留 S 供 STAGE5 的 dsilu)vs Softmax(`P=exp(S−LSE)`,LSE 来自 fwd)。讨论 S 是否要在寄存器/LDS 留一份给 dsilu。
   - STAGE3 dV += Pᵀ@dO、STAGE4 dP=dO@Vᵀ:基本同 FMHA(确认 hdim_v 维度)。
   - STAGE5 dS:**双路**——SiLU `dS=dP·scale_p·dsilu(S)`(需 S);Softmax `dS=P·(dP−D)`(需 D,来自 PRE)。**masked-out 必须显式置 0**(SiLU silu(0)≠自然零)——讨论在哪一步、如何与 mask 协同置零。
   - STAGE6 dK += dSᵀ@Q(×alpha)、STAGE7 dQ=dS@K(×alpha) + 写回。确认 alpha 在收尾乘(类似 FMHA raw_scale)。
3. **fwd 副产物契约**:softmax 路径需 fwd 存 LSE(kStoreLSE 接线柱)+ O(给 PRE 算 D);SiLU 路径都不需要。明确每条路径 bwd 的输入集合 {Q,K,V,dO,(O,LSE,D)}。
4. **关键骨架伪代码**:给出 MAIN pipeline `operator()` 的双路骨架(只写改造点,引用 FMHA 行号说明"此处同 FMHA / 此处替换")。给出 PRE 是否跳过的编译期逻辑。
5. **复用 vs 新写清单**(算法层面)+ 风险点(如:S 重算的寄存器压力、SiLU 留 S 的额外 LDS/VGPR、双路模板膨胀)。

## 边界
只设计算法与 pipeline 阶段;mask/索引/模式留给 pane-2,工程落地(文件/codegen/CMake/测试)留给 pane-3。但要和它们接口自洽(mask 如何喂进 STAGE2/5、params 如何提供 alpha/scale_p)——在文末列出对 pane-2/pane-3 的接口假设。

## 铁则
- 数学/行号以 reference + FMHA 源码为准(已 Read 复核)。聚焦"方案+接口+关键伪代码",别写满整份 kernel。
- markdown 写到 `/tmp/hstu-bwd-design/part-algo.md`;progress 简洁;完成写 `/tmp/hstu-bwd-design/algo-done.md`(一行:覆盖了哪些、关键决策、未决问题)。
