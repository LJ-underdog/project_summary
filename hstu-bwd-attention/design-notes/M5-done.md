# M5 softmax bwd — 完成报告 (pane-1 / coder)

状态:**✅ 通过**。softmax + bf16 + hd64 的 **no_group(batched + jagged)** 路径端到端对拍全 PASS(attn_scale=1.0,softmax × causal{0,1} × 5 因子 × {batched,jagged})。SiLU 路逐字不变(零回归)。测试套件整体 exit 0。group softmax = M5b 未做(本单范围外)。日期 2026-06-08。

## 0. 基线确认
合并 fwd kStoreLSE 后,先跑 `run_bwd_tests.py` → 46 案全绿(45 pass/1 skip,exit 0),M0–M4b 无回归,方才动手。

## 1. 改了哪些文件
| 文件 | 改动 |
|---|---|
| **新** `hstu_attention_with_softmax_bwd_pipeline.hpp` | softmax MAIN pipeline(SiLU pipeline 的 fork) |
| `hstu_attention_bwd_kernel.hpp` | 新增 `HstuAttentionBwdDQDKDVSoftmaxKernel`(带 LSE/D window)+ PRE `hstu_bwd_dot_do_o_kernel`(D=rowsum(O⊙dO)) |
| `hstu_attention_batched_backward_dispatch.hpp` | include softmax pipeline;新增 `RunSoftmax`(PRE→memset→MAIN→POST);`kUseSoftmax` 分支去 throw 改调 RunSoftmax |
| `example_hstu_attention_bwd.cpp` | softmax 时 fwd 开 `is_training/lse_ptr`(连续-seq 布局)+ 取回 LSE 转置喂 reference + 分配 d_dev + bp.lse/d_ptr/stride_lsed |
| **fwd 逻辑零改动**(只在 harness 切 `is_training`);group softmax 未碰 |

## 2. STAGE2/5 关键代码(数学 = reference kUseSoftmax 分支)
- 域确认:fwd 用**自然 exp**(`__expf`)、`pcomp*=scale_s`、`lse=m+log(l)` ⇒ **LSE = log(Σexp(α·S)) 自然对数**,与 reference `P=exp(α·S−LSE)` 一致。
- **STAGE2**(替 SiLU 的 silu/dsilu):
  ```cpp
  // 运行时门(非编译期 IsMasking;沿用 P1-1 教训,causal=0+num_target 也掩)
  if(mask.IsEdgeTile(seqlen_q_step, k_origin..., kM0, kN0))
      set_tile_if(s_acc, -numeric<AccDataType>::infinity(),
                  [&](idx){ return !mask.IsTokenPairInsideMask(row,col); }); // softmax 用 -inf
  const auto scale = alpha * log2e_v<AccDataType>;
  auto row_lse = log2e_v<LSEDataType> * get_validated_lse(lse[i_idx]); // -inf→0 防 NaN
  p(i_j_idx) = exp2(scale * s_acc[i_j_idx] - row_lse);   // = exp(α·S − LSE)
  ```
  (`set_tile_if(-inf)` 与 SiLU 的 `set_tile_if(0)` 相反——softmax 必须 -inf 使 exp→0;SiLU 禁 -inf。)
- **STAGE5**(替 `ds=dp*g`):`ds(i_j_idx) = p[i_j_idx] * (dp_acc[i_j_idx] - d[i_idx]);`(无 dropout,去 undrop 分支)
- 其余(K/V/Q/dO 预载、5 GEMM、收尾 `dq*=alpha`/`dk*=alpha`/dV 不乘)与 SiLU 逐字相同。LSE/D 的 HBM→LDS→Reg 用 policy `MakeLSEDDramTileDistribution`/`MakeLSEDLds{Write,Read}BlockDescriptor`,LDS 偏移用 FMHA 同款(`QT+OGrad+OGradT+Q` 起 LSE,`+LSE` 起 D)。**这些 region SiLU 已在 GetSmemSize 预留 → smem 总量不变。**

## 3. PRE kernel(D=rowsum(O⊙dO))
`hstu_bwd_dot_do_o_kernel`:每线程一行 (i_batch,i_nhead,sq),`D=Σ_v float(O)*float(dO)`,hdim_v=64 单线程累加。O/dO 用 [batch,seq,head,hdim] strides 定位(jagged 用 `seq_q_offsets[i_batch]` 定 token 基址);D 写 **[batch,head,seq] 连续-seq**(`nhead_stride_lsed/batch_stride_lsed`,seq stride=1)——正是 MAIN 读 D 的布局。dispatch 在 MAIN 前 launch(全覆盖写,免 memset)。

## 4. LSE 布局对齐结论(防 silent-wrong,关键)
- **GPU 侧**(fwd 产、bwd 读)统一 **[batch,head,seq] 连续-seq**:fwd `seq_stride_lse=1, nhead_stride_lse=phy_seqlen_q, batch_stride_lse=num_head*phy_seqlen_q`;bwd `nhead_stride_lsed=phy_seqlen_q, batch_stride_lsed=num_head*phy_seqlen_q`,seq 1-D packed。jagged 时 nhead_stride=ΣL、batch 基址=query_start。fwd 与 bwd **共用同一 lse_dev**,无转置。
- **reference 侧**要 **[batch,seq,head]**(`lse_batch_seq_nhead(i_batch,sq,i_head)`,jagged `(0,offset+sq,head)`)。harness 把 GPU LSE 取回后**转置**进 `lse_host`:`lse_host(b,s,h)=lse_flat[(b*num_head+h)*phy_seqlen_q+s]`。
- **GPU bwd 与 reference 用的是同一份 GPU-产 LSE**(reference 不依赖自身 fwd),保证两侧 P 一致。这是对拍能精确闭合的根因。

## 5. 逐档对拍(attn_scale=1.0,bf16 阈值 rel≤2e-2/abs≤5e-2)
`runs/run-M5-sweep.log`(**21/21 PASS**):

| 维度 | 覆盖 | 结果 |
|---|---|---|
| batched × causal{0,1} | no-mask/causal/window/context/minfull/num_target/combo | ✅ 全过 |
| batched 非整除 | seq200, b3×nhead4 | ✅ |
| jagged × causal{0,1} | no-mask/causal/window/num_target(per-batch)/combo + tiny(1,128,7) | ✅ |
| causal=0 + num_target | batched & jagged(沿用 P1-1 修复,softmax 同样运行时掩码)| ✅ |

误差 bf16 舍入级:dQ/dK max_abs ≤ 2.4e-4、dV ≤ 2e-3(softmax 梯度量级本身 ~0.08–2)。

## 6. 验收对照
- 编译 0 error(`runs/build-M5.log`)。✅
- 对拍 PASS:softmax × {no-mask,causal,window,contextual,min_full,num_target,全组合} × causal{0,1} × {batched,jagged},含非整除/per-batch 变长。✅
- 测试套件 `reject-softmax` 删除,新增 15 个 M5 pass 交叉案;`python3 test/run_bwd_tests.py` → **TOTAL 60 / PASS 59 / FAIL 0 / SKIP 1,exit 0**(`runs/test-20260608-063912.log`)。✅
- candidates.jsonl 加 `M5-softmax`(promoted)。✅
- **SiLU 零回归**:softmax pipeline/kernel 为独立新增;SiLU 的 no_softmax pipeline、SiLU kernel、group kernel 一字未改;套件内全部 M1/M2/M3/M4/M4b(44 案)仍 PASS。✅
- fp16+hdim(M7)、deterministic(M6)仍正确拒绝/跳过。

## 7. 遇到的坑
- 唯一编译错:dispatch 在 `ck_tile` namespace 外,PRE launch 里 `long_index_t` 需写 `ck_tile::long_index_t`(已修)。
- LSE 双布局(GPU [b,h,s] vs reference [b,s,h])是最大 silent-wrong 风险点——靠"GPU LSE 同时喂 GPU-bwd 与 reference + 显式转置"闭合,对拍逐位级误差佐证布局/域均正确。

## 8. 遗留 / 给后续
- **M5b group softmax**:group dispatch/kernel 现仍仅 SiLU;接 group 需 group kernel 双 pipeline 也走 softmax + PRE 按 group packed 布局产 D + fwd group 产 LSE。本单未做(范围外)。
- **cross-attention softmax**:no_group 现仅 self;cross 待对应里程碑。
- M6 deterministic / M7 fp16+hdim / M8 perf 不变。
- 无未解决阻塞点。
