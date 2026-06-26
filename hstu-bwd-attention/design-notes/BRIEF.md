# DESIGN-BRIEF —— HSTU bwd GPU 实现方案(所有设计 teammate 共享基线)

目标:为 **HSTU attention backward 设计一套 GPU kernel 实现方案**(当前只有 855 行 CPU 参考、无 GPU kernel)。**参考 CPU reference 的语义,复用 ck_tile FMHA bwd 的 GPU 基建**。产出是"设计方案",不是写最终 kernel 代码(可给关键骨架伪代码/接口)。

## 唯一权威来源(落笔前各自 Read)
- **HSTU bwd 语义(oracle)**:`/root/workspace/ck_hstu/example/ck_tile/18_hstu_attention/reference_hstu_attention_bwd.hpp`(855 行,顶部 21-58 行有完整公式注释;`reference_no_group_...` + `reference_group_...` 两套)
- **HSTU fwd 可复用基建**(同目录):`hstu_attention_params.hpp`、`hstu_attention_pipeline_problem.hpp`、`hstu_attention_traits.hpp`、`hstu_attention_tile_setting_define.hpp`、`hstu_block_masking.hpp`、`hstu_attention_{jagged,group,batched}_forward_dispatch.hpp`、`hstu_attention_{no,with}_softmax_fwd_pipeline.hpp`、`generate_instances.py`、`CMakeLists.txt`
- **FMHA bwd 可复用 GPU 基建**(`/root/ck/include/ck_tile/ops/fmha/`):
  - `pipeline/block_fmha_bwd_dot_do_o.hpp`(PRE:D=rowsum(O⊙dO),95 行)
  - `pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp`(MAIN:5 GEMM/7 stage,779 行)
  - `pipeline/block_fmha_bwd_convert_dq.hpp`(POST:deterministic dQ 归约)
  - `pipeline/block_fmha_bwd_pipeline_{problem,default_policy,enum}.hpp`
  - `kernel/fmha_bwd_kernel.hpp`(3 个 kernel 包装 + GridSize)
- **我们已产出的讲解文档**(`/root/workspace/hstu-b1052-report/`):`bwd-main-pipeline-7stages`、`bwd-5-gemms-deepdive`、`bwd-saved-tensors`、`ck-vs-hstu-bwd`(可作背景,但事实以源码为准)

## HSTU bwd 数学(来自 reference 顶部注释,已核实)
```
S[sq,sk]  = alpha * Q[sq]·K[sk]ᵀ           (masked-in), 否则 0 或 -inf
P[sq,sk]  = silu(S)*scale_p                 (kUseSoftmax=false, 默认 SiLU 路径)
          = softmax_row(S)                  (kUseSoftmax=true)
dV[sk,k]  = Σ_sq P[sq,sk]·dO[sq,k]          = Pᵀ @ dO
dP[sq,sk] = Σ_k  dO[sq,k]·V[sk,k]           = dO @ Vᵀ
-- SiLU 路径: dsilu(x)=σ(x)(1+x(1-σ(x)));  dS = dP*scale_p*dsilu(S)   (masked-out 显式置 0!silu(0)≠自然零)
-- Softmax 路径: D[sq]=rowsum(O⊙dO);  P=exp(S-LSE)(LSE 来自 fwd);  dS = P*(dP - D)
dQ[sq,k]  = alpha * Σ_sk dS[sq,sk]·K[sk,k]  = alpha * dS  @ K
dK[sk,k]  = alpha * Σ_sq dS[sq,sk]·Q[sq,k]  = alpha * dSᵀ @ Q
```
关键参数:`alpha`(QK 缩放,对应 FMHA 的 1/√d 位置)、`scale_p`(SiLU 输出缩放,默认 1/max_seqlen_q)、`attn_scale`、`contextual_seqlen`、`window_size`、`min_full_attn_seqlen`、`num_target`;模板 `kIsJagged`/`kUseSoftmax`/`kIsGroupMode`。布局 **bshd**,`hdim_qk` 可 ≠ `hdim_v`。

## HSTU vs FMHA bwd 的关键差异(决定能复用多少)
1. **激活默认 SiLU 而非 softmax**:STAGE2 不读 LSE、必须重算 S 取 dsilu;STAGE5 的 dS 公式两路不同;masked-out 必须**显式置 0**(不能靠 exp(-∞))。
2. **两套 scale**:`alpha`(QK)+ `scale_p`(SiLU 输出),FMHA 只有 1/√d。
3. **5 因子 HSTU mask**:`HstuCrossAttentionBlockMaskWithLocal`(causal/window/contextual_seqlen/min_full_attn_seqlen/num_target),影响 tile 范围与 early-exit。
4. **jagged/group/batch 三模式 + bshd + cu_seqlens**;group 模式每段独立超参。
5. **hdim_qk ≠ hdim_v**。
6. softmax 路径才需要 LSE(fwd 侧 `kStoreLSE` 接线柱已预留)+ D(PRE pass)。

## 现状缺口(要设计补齐的)
- `params.hpp` **没有 bwd 字段**(dq/dk/dv/do/d/dq_acc 指针、stride、bwd 专用参数)。
- 无 bwd pipeline / kernel / dispatch / instances / CMake target。

## 复用策略主线(供参考,设计可调整)
3-kernel 结构沿用 FMHA:**PRE**(dot_do_o,仅 softmax 路径算 D)→ **MAIN**(dq_dk_dv kr_ktr_vr,适配 SiLU/softmax 双路 + HSTU mask + jagged/group)→ **POST**(convert_dq,deterministic 时归约 dQ)。policy/problem 尽量继承 FMHA bwd,差异点(激活、mask、scale_p、索引)做 HSTU 特化。

## 输出形式约定
每个设计 teammate 产出一份 **markdown 设计文档**到 `/tmp/hstu-bwd-design/part-<X>.md`,结构清晰、含:决策 + 理由 + 关键接口/伪代码骨架 + 复用 vs 新写清单 + 风险/未决问题。**别写满整份 kernel**,聚焦"方案与接口"。行号/事实以源码为准。
