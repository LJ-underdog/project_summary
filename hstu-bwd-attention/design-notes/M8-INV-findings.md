# M8 investigation — VGPR 124-vs-248 + SiLU 26% 异常(纯 profiling,不改库码,不 commit)

基线 HEAD=`048f0a9a`。GPU gfx950/CDNA4。canonical config = `-b=2 -nhead=8 -seqlens=2048 -causal=1` hd64 bf16。
证据:`profile/M8-INV-kernel-metadata.txt`、`profile/pmc_hd64`(softmax)、`profile/pmc_big`(大 config)、
`profile/pmc_silu`/`pmc_inst_{silu,smx}`(SiLU vs softmax)。

---

## 任务 1:VGPR 124-vs-248 矛盾 —— 定论 **真值 248**(rocprofv3 报的是半值)

**三源交叉核实(同一个 canonical MAIN kernel):**
| 来源 | VGPR | 说明 |
|---|---|---|
| kernel descriptor `.vgpr_count`(code object metadata,**权威**)| **248**(NoLocal/WithLocal self)/ 250(cross)| llvm-readelf --notes;运行时按此分配 |
| 编译器 `-Rpass-analysis=kernel-resource-usage` | **248** | `profile/M1-resource.md`,与 descriptor 一致 |
| rocprofv3 v1.3.0 `VGPR_Count`(kernel-trace/pmc)| **124** | = 248/2 **整除**;v3 的 VGPR_Count 列报的是半值(granule/单位约定),**非另一个 kernel** |

→ **真实 archVGPR = 248**(AGPR=0,Scratch=0 无 spill)。rocprofv3 的 124 是 v3 的报数单位假象(248/2),
占用率计算一律用 **248**。矛盾消除。

### MAIN occupancy 真限制器
CDNA4 512-VGPR/SIMD 池,block=256 线程=4 waves(wave64)。
- **VGPR**:512/248 = 2.06 → **2 waves/SIMD = 8 waves/CU = 2 blocks/CU**(编译器 occupancy 报告=2,吻合)。
- **LDS**:32768 B/block,CU 64KB → **2 blocks/CU**。
- → **VGPR 与 LDS 在 2 blocks/CU 处共同(co-)限制**。

**实测占用率(MeanOccupancyPerCU,rocprofv3 PMC):**
| config | grid blocks | MAIN occ (waves/CU) | % | 限制器 |
|---|---|---|---|---|
| canonical 小(`pmc_hd64`)| 256(=1/CU)| **3.39**(10.6%)| 256 wg / ~256 CU | **GRID-limited**(供不应求,~1 block/CU)|
| 大 config(`pmc_big`,4096 wg)| 4096(超额订阅)| **6.96**(21.8%)| ≈2 blocks/CU(8 waves 的 87%)| **资源 ceiling(VGPR+LDS co-limit 2 blocks)** |

→ **小 config 占用率被 grid 卡(1 block/CU);大 config 触到 2-block/CU 资源天花板。**
两者 **MfmaUtil 仍只 9.9% / 18.4%**,**矩阵核闲 80–90%**。大 config **VALUBusy=41% >> MfmaUtil=18%** →
**MAIN 偏 VALU/SFU-bound,不是 MFMA-bound、也不是单纯 occupancy-bound**。

### B7(occupancy 1→2 / 提占用率)可行性 + 代价
- **2→3+ blocks/CU 必须同时**砍 VGPR ≤170(512/3)**且** LDS ≤21KB(64/3)。**只砍一个无效**(另一个仍卡 2)。
  - LDS:32KB 含未用 LSE/D/bias 段(`M1-resource.md` R6 标过可省),但**单砍 LDS inert**;且改共享布局有 253-suite 回归风险。
  - VGPR:248→170 = 砍 31%,对 5-GEMM 寄存器重的 kernel **极难无 spill**,基本要 tile 重设计(动共享布局)。
- **B7 上限有限**:从 grid-limited(小)到 2-block ceiling(大)占用率翻倍(3.39→6.96),MfmaUtil 也才 9.9%→18.4%;
  再到 3 block 预计 MfmaUtil 仅 ~25%。**真瓶颈是 VALU/SFU 占用 + 依赖链/串行 Q-loop,不是 occupancy。**
- **建议**:B7(纯提 occupancy)**ROI 低、代价高(tile 重设计)**,**不优先**。更高 ROI 的方向是降 VALU/transcendental
  工作量或提升 MFMA/VALU overlap(见任务 2)。小 config 的 grid 不足是真,但 B4 已被大 config 超额订阅证伪,
  不普适。

---

## 任务 2:SiLU 26% 异常 —— 根因 = **2× transcendental(sigmoid 的 exp+rcp vs softmax 单 exp2)**

MI 复现:同 shape SiLU MAIN 0.333ms vs softmax 0.263ms = **1.27×**。两者 **VGPR/LDS/AGPR/Scratch/SQ_WAVES 完全相同**
→ 非资源/occupancy 静态差异。

**指令计数(`pmc_inst_{silu,smx}`,MAIN,per-SE aggregate):**
| 计数器 | softmax | SiLU | SiLU/smx |
|---|---|---|---|
| SQ_INSTS_MFMA | 1.950e6 | 1.950e6 | **1.000**(矩阵工作完全相同)|
| SQ_INSTS_VALU | 1.569e7 | 1.700e7 | 1.084 |
| **SQ_INSTS_VALU_TRANS_F32** | 5.58e5 | 1.115e6 | **1.998(整 2×)** |
| SQ_INSTS_LDS | 3.16e6 | 2.96e6 | 0.934 |

**运行时利用率(`pmc_hd64` vs `pmc_silu`,小 config):** SiLU 的 occupancy 2.17 vs 3.39、MfmaUtil 6.6 vs 9.9、
VALUBusy 15.4 vs 22.5 —— **利用率全更低、但耗时更长** → SiLU 在更长的 transcendental(SFU 低吞吐)依赖链上 stall 更多,
有效占用率被拖低。

**根因(代码核实 `hstu_attention_no_softmax_bwd_pipeline.hpp:365-409`):**
- SiLU:`f_sigmoid(x)=rcpf(1+expf(-x))` = **expf + rcpf = 2 个 TRANS_F32**;`sig` 已 CSE(只算一次,silu 与 dsilu 共用,
  **不是重复计算 bug**)。
- softmax:`p=exp2(...)` = **1 个 TRANS_F32**。
- → 每元素 SiLU 2 trans vs softmax 1 trans,正好 2.0×。SFU 低吞吐 + 长延迟 → SiLU 26% 慢,**符合预期,非缺陷**。

### 是否有便宜 quick win?
- **没有便宜的 win**:sigmoid=1/(1+e^−x) 本质需要 exp **和** reciprocal 两个超越运算;`sig` 已 CSE。
- 候选(均**非 cheap**,都改库设备码、要走 4-gate,本期不做):
  1. **近似 sigmoid 单超越式**(如 tanh/多项式)→ 改数值精度,有对拍风险;
  2. 用更高 ILP / occupancy 隐藏 SFU 延迟 → 受 2-block ceiling 限,且 SiLU 已近 ceiling;
  3. 大 config(超额订阅)下 SFU 延迟更易被隐藏,SiLU 罚分预计缩小(未测,可后续验证)。
- **结论**:SiLU 26% 是 transcendental 工作量本质 2× 的结果,**不是 bug、无便宜快赢**;若要追 SiLU,需近似式(精度权衡)或
  更大的 MAIN 重构,优先级低。

---

## 给 lead 的决策输入(总结)
1. **VGPR 真值 248**(rocprofv3 报半值 124);occupancy ceiling = **2 blocks/CU,VGPR+LDS co-limit**;小 config grid-limited、大 config 触 ceiling。
2. **B7 不优先**:提 occupancy 需同砍 VGPR(<170,极难无 spill,要 tile 重设计)+ LDS(<21KB,单砍 inert),且 MAIN 实为 VALU/SFU-bound(VALUBusy 41% >> MfmaUtil 18%),occupancy 上限收益有限。
3. **SiLU 26% 根因 = sigmoid 2× transcendental(exp+rcp)vs softmax 单 exp2**,已 CSE 非 bug,无便宜快赢。
4. 纯 investigation,**未改库码、未 commit**。证据见 `profile/M8-INV-*` + `profile/pmc_{hd64,big,silu,inst_*}`。
