# pane-3 (architect) —— 工程落地 & 验证 完成

产出:`/tmp/hstu-bwd-design/part-engineering.md`

## 覆盖项(对照任务清单 1-9 全覆盖)
1. ✅ 文件/目录结构 + 「新建 vs 复用 FMHA」映射表(§1)
2. ✅ problem/traits/tile_setting(`HstuAttentionBwdPipelineProblem` + 9 元 BlockTile + policy 继承)(§2)
3. ✅ 三套 backward dispatch + 实例化矩阵收敛(384→MVP 48)(§3)
4. ✅ dQ 写回两路:atomicAdd vs deterministic split-workspace + convert_dq;dq_acc 分配/stride(§4)
5. ✅ params bwd 字段扩展(§5)
6. ✅ kernel 包装 + 3-kernel GridSize/launch 顺序(§6)
7. ✅ instances + codegen(扩展 `generate_instances.py`)(§7)
8. ✅ CMake 新 bwd target + CLI 增量(§8)
9. ✅ 测试验证(oracle 对拍流程、bf16/fp16 容差、测试矩阵、deterministic 逐位、边界)(§9)
   ✅ 分阶段里程碑 M0-M8 + 验收标准(§10)
   ✅ 风险/未决 8 条(§11)
   ✅ 对 pane-1/pane-2 依赖假设 + 对外保证(§12)

## 关键工程决策
- 走 HSTU 自有风格(自写 dispatch + `generate_instances.py`),**不**搬 FMHA `codegen/ops/fmha_bwd.py`。
- PRE/POST/policy/shape/enum **零成本复用** FMHA;**仅 MAIN** 必须 HSTU 特化(SiLU 双路 + 5 因子 mask + scale_p + jagged/group 索引)。
- dQ 默认 atomicAdd(float dq_acc + POST convert-only),deterministic 为显式可选(split-workspace 逐位可复现)。
- M1(batched+SiLU+no-mask+bf16+atomicAdd+hdim64 端到端对拍)是**风险闸门**:验证 FMHA MAIN/policy 能否被 HSTU 特化复用。

## 源码核实要点
- HSTU 目录平铺无 codegen/;fwd dispatch 入口 `run_<mode>_forward_causal_softmax_bias_dropout_dispatch<...>` 已读。
- FMHA 3-kernel GridSize 核实:MAIN=(ceil(seqlen_k,kN0),nhead,batch);PRE/POST 沿 seqlen_q。
- `TileFmhaBwdShape` 9 元 BlockTile 语义、`BlockFmhaBwdConvertQGrad` Reduce+Convert、`BlockFmhaBwdOGradDotO` D 公式 已读。
- oracle 签名(no_group/group)已核实并写入 §9.1。
