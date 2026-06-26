# pane-2 REVIEW + OPTIMIZE — HSTU bwd 实现现状文档

对象:`/root/workspace/hstu-b1052-report/hstu-bwd-impl-status-20260604.html`(pane-3 写)
方式:独立 review,对照真实代码逐条核验,复跑 M1 验证命令。P0 直接最小修复;P1/P2 视情况改。
结论:**文档总体高度准确**,核心技术事实(七阶段/调用链/复用边界/数值/资源)全部与代码一致且 M1 命令**逐位复现**。发现 1 处 P0(行数)、3 处 P1(门控措辞不精确)、1 处 P2(图1箭头语义),已全部在 HTML 内最小修复。

---

## 第一部分:REVIEW 逐条判定

### 1. 文件清单准确性 — ✅(修 1 处 P0)
`wc -l` 实测,8 个主文件行数与文档**完全一致**:
| 文件 | 文档 | 实测 |
|---|---|---|
| bwd_params.hpp | 119 | 119 ✅ |
| no_softmax_bwd_pipeline.hpp | 518 | 518 ✅ |
| bwd_kernel.hpp | 376 | 376 ✅ |
| batched_backward_dispatch.hpp | 233 | 233 ✅ |
| no_group_backward_bf16.cpp | 33 | 33 ✅ |
| example_hstu_attention_bwd.cpp | 483 | 483 ✅ |
| api.hpp | 22 | 22 ✅ |
| reference_hstu_attention_bwd.hpp | 855 | 855 ✅ |

- **❌→修复(P0)**:instance `.cpp` 文档写「~12」,实测每个 **18 行**(单条 explicit template instantiation,见 `..._no_causal_softmax_false_no_bias_atomic_maxk_64.cpp`);ref 头 42 行正确。已改 `~12/42` → `~18/42`。
- 角色 / 「新写 vs 改动 vs 复用」/「复用了哪些 FMHA」分类:逐行核对 include 与内容,**准确**。reference 归「复用」(早于本工作存在)正确。

### 2. include / 依赖图(图1)— ✅⚠️(P2 已加图注)
实测 include 关系:
- `entry.cpp` → `instances_ref.hpp`(L11)✅;`instances_ref.hpp` → `dispatch.hpp`(L10)✅
- `dispatch.hpp` → `pipeline.hpp`+`kernel.hpp`+`bwd_params.hpp`+`ck_tile/ops/fmha.hpp`+`epilogue.hpp`(L11/12/19/20/21)✅
- `pipeline.hpp` → `block_fmha_bwd_pipeline_default_policy.hpp`(L8)+`block_attention_bias_enum.hpp`(L7)✅
- `GenericAttentionMask`/`TileFmhaBwdShape`/`BlockFmhaBwdPipelineProblem`/`BlockDropoutBwd` 经 `fmha.hpp`、`Default2DEpilogue` 经 `epilogue.hpp` 拉入 ✅
- 所有节点与复用边界正确。
- **⚠️(P2)**:图1 是**自顶向下的「引用/构建流」**,而非逐条严格 `#include` 方向——少数箭头(`params→dispatch`、`dispatch→policy 簇`、`pipeline→kernel`)按「被谁使用/组合」画。其中 default policy 实为 **pipeline** include(非 dispatch),kernel 把 pipeline/epilogue 作**模板参数**组合(kernel.hpp 不 include pipeline)。已在图注补一句澄清箭头语义,避免误读;SVG 几何未动(低风险)。
- 小注(未改):图1 框内 `BlockFmhaBwdProblem` 为 `BlockFmhaBwdPipelineProblem` 的排版缩写,正文用全名,可接受。

### 3. 运行时调用链(图2)— ✅
逐段对 dispatch/kernel/harness 核验,**全部一致**:
- harness `main`→GPU fwd `hstu_attention_no_group_forward_bf16`(产 O)→`hstu_attention_no_group_backward_bf16`→`BOOL_SWITCH_2(causal,softmax)`→`run_batched_backward_dispatch::Run`→(jagged/determ/softmax/causal 门控)→`RunSilu` ✅
- `RunSilu`:`scale_p = attn_scale ? attn_scale : 1/max_seqlen_q`(dispatch L122-124)✅ → ① `hipMemsetAsync(dq_acc=0)`(L165)→ ② launch MAIN `HstuAttentionBwdDQDKDVKernel`(atomic_add 写 float dq_acc;`Default2DEpilogue` 写 dk/dv,kernel L355-356)→ ③ launch POST `hstu_bwd_convert_dq_kernel`(L180)✅
- grid = `(ceil(seqlen_kv/kN0), nhead, batch)`(kernel `GridSize` L177-181)✅
- CPU `reference_no_group_hstu_attention_bwd` → `check_err` 三张量,`rtol=2e-2/atol=5e-2`(harness L107)、`exit = numeric_pass?0:-2`(L482)✅
- `dq_acc` 作用(float+atomic 累加避 bf16 精损/多 block 冲突,POST cast)描述准确(kernel atomic_add window L302-303 + POST elementwise cast L365-374)✅
- GPU/CPU 边界标注正确。

### 4. 七阶段(图3)+ 代码佐证 — ✅
对 `no_softmax_bwd_pipeline.hpp` 逐 STAGE 核验:
- gemm_0..4 ↔ STAGE1/3/4/6/7 映射(`GetQKBlockGemm`/`GetPTOGradTBlockGemm`/`GetOGradVBlockGemm`/`GetSGradTQTBlockGemm`/`GetSGradKTBlockGemm`,L119-123)✅
- STAGE2(L395-412):`s=alpha*s_acc`;`sig=f_sigmoid(s)`;`silu=s*sig`;`dsilu=sig*(1+s*(1-sig))`;`p=silu*scale_p`(→dV);`g=scale_p*dsilu`(→dS)——与图/code-block **逐字一致** ✅
- `f_sigmoid` = `rcpf(1+expf(-x))`(L366-372,fp32 分支用 `__builtin_amdgcn_rcpf`+`__expf`)✅
- STAGE5(L442-450):`ds = dp_acc * g`,g 已带 scale_p ✅
- 收尾:`dq_acc*=alpha`(L495)、atomic `update_tile` / determ `store_tile`(L497-504)、`dk_acc*=alpha`、dV 不乘(L512)✅
- code-block 行号标注(STAGE2~L395 / STAGE5~L442 / 收尾~L494/L512)**准确**。
- softmax 路 `ds=p·(dp−D)` 标 M5 TODO,与 reference L413 一致 ✅

### 5. 复用 vs 新写 — ✅
- pipeline 标「改写(不直接 include)」准确:以 kr_ktr_vr 为蓝本重写,但原样调全部 policy `Make*/Get*` + 5 GEMM + `PTFromGemm0CToGemm1A`/`SGradTFromGemm2CToGemm3A`(L430/L458)✅
- 「复用·直接 include」:default policy / mask / epilogue / dropout / problem / shape / enum,与 dispatch L74-118 的 `using` 一致 ✅
- 新写:params/dispatch/kernel/entry/harness 分类正确 ✅

### 6. 关键工程决策 4 条 — ✅
对 M1-done.md + 代码核验,4 条全部相符:
- ① ck_hstu 自带较新 ck_tile(Traits/Shape/Problem 签名 + `kUseTrLoad` + smem 首参)——dispatch L74-105、pipeline `operator()` smem 在末参但 problem 带 `kUseTrLoad`(L104)✅
- ② 自写 kernel 因双 scale(`alpha`+`scale_p`)——kargs L74-75、MakeKargs 传两标量 ✅
- ③ float dq_acc + atomic + POST(atomic 路 nsplits=1、dq_acc 与 dq 同布局、convert 纯 elementwise)——dispatch L162-188 + kernel POST ✅
- ④ 保留 `BiasEnum=NO_BIAS` + dummy `BiasDataType`——dispatch L76/L91、pipeline L49-52 dummy typedef ✅

### 7. 覆盖面 / TODO 表 — ✅(修 3 处 P1)
实测 dispatch `Run`(L191-216)门控:
- `if constexpr`:deterministic(L196)、softmax(L200)、causal(L204)→ throw ✅
- **运行时 `if`**:jagged(`if(param.is_jagged) throw` L193)、非 hd64(`if(hdim!=64) throw` L211)
- group:dispatch/entry **无 group 路径**(`GroupBwdParams` 空 struct;entry 仅 `no_group`;harness 仅 no-group)——**并非 throw**
- fp16:**入口/harness 仅 bf16**,无 fp16 entry

→ 文档原措辞把以上一律说成「编译期 if constexpr 门控成 throw」不精确:
- **⚠️→修复(P1)**:§1 段落重写,区分 `if constexpr` throw(causal/softmax/determ)、运行时 `if` throw(jagged/非 hd64)、group 无入口、fp16 仅 bf16。
- **⚠️→修复(P1)**:§8 表 group 行「group → throw」→「暂无 group dispatch 入口(非 throw,harness 仅 no-group)」。
- **⚠️→修复(P1)**:§8 表 fp16 行「assert hd64+bf16」→「非 hd64 → 运行时 throw;fp16 则入口/harness 仅 bf16」(代码是 `throw` 非 `assert`)。
- 其余行(batched/SiLU/no-mask/bf16/hd64/atomic ✅;causal/jagged/softmax/determ throw)均准确。

### 8. 数值 / 资源数 — ✅(亲自复跑)
**复跑** `tile_example_hstu_attention_bwd -prec=bf16 -b=2 -nhead=2 -hdim_qk=64 -hdim_v=64 -seqlens=128 -softmax=0 -causal=0 -attn_scale=1.0 -v=1`:
```
  dQ: max_abs_err=0.00012207 mean_abs_err=3.72529e-09 (max|ref|=5.125)
  dK: max_abs_err=0          mean_abs_err=0           (max|ref|=5.21875)
  dV: max_abs_err=0.00390625 mean_abs_err=2.94298e-07 (max|ref|=4.625)
[PASS] dQ   [PASS] dK   [PASS] dV    numeric_pass=true    EXIT=0
```
→ 与文档 §9 数值**逐位一致**,PASS + exit 0 ✅。GPU = gfx950 / MI350X(rocminfo 确认)。
- R1(default policy 零覆写复用)结论与 pipeline 实现一致 ✅
- R2(VGPR248/AGPR0/Scratch0/occ2,LDS 32768)取自 M1-done.md §5,与文档一致 ✅(本次未重测资源,无 GPU 侧 spill 反证)
- reference「6 步推导」核实:Step1 S/P→Step2 dV→Step3 dP→Step4 dS→Step5 dQ→Step6 dK(ref L237-461),与 §2「S→P→dV/dP/dS→dQ/dK,6 步」一致 ✅
- 附:§10 CMake 旗标(`-fno-slp-vectorize`/`-DBUILD_HSTU_FOR_GFX95_ONLY`/`CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3`)、`EXCLUDE_FROM_ALL`、GLOB `*backward*.cpp`+`*forward_bf16*.cpp`、`generate_instances.py` 的 `create_backward_instances(_ref)` 均核实存在 ✅

---

## 第二部分:OPTIMIZE
本文档可读性、零基础友好、配色(clay 新写 / olive 复用)、图例已相当完善,未做大改;仅在修事实的同时顺手提升精确度:
- §1 门控段落改写后,读者能一眼区分「四种不同的未实现拦截方式」,比原「一律 if constexpr throw」更贴合代码、也更有 review 价值。
- 图1 图注补「箭头=引用/构建流,非严格 #include 方向」一句,消除「为什么 pipeline 指向 kernel / params 指向 dispatch」的潜在困惑。
- 未改动:正文叙述流畅、术语首现均有白话解释、三张 SVG 清晰且有图例,无需返工。

---

## 改动清单(均在 HTML 内最小修复)
1. (P0)§2 表 instance 行数 `~12/42` → `~18/42`。
2. (P1)§1 门控段落重写:区分 if constexpr throw / 运行时 if throw / group 无入口 / fp16 仅 bf16。
3. (P1)§8 表 group 行:「group → throw」→「暂无 group dispatch 入口(非 throw)」。
4. (P1)§8 表 fp16 行:「assert hd64+bf16」→「非 hd64 运行时 throw;fp16 入口/harness 仅 bf16」。
5. (P2)§3 图1 图注:补箭头语义澄清。

## 复跑验证
M1 闸门命令复跑 **PASS + exit 0**,三梯度误差与文档逐位一致(见上 §8)。

## 标签平衡 / 锚点(改后独立复核)
- div 93/93、section 10/10、svg 3/3、g 42/42、table 2/2、thead 2/2、tbody 2/2、tr 26/26 —— **全平衡**。
- h2=10(对 10 个 section)、h3=9。
- TOC 锚点 `#s1..#s10` 与 `id="s1..s10"` **一一对应**,无悬空。
- 3 个 SVG 内容坐标均 < viewBox 宽 980,且 `.svg-wrap` 带 `overflow-x:auto`,无溢出。

无未解决阻塞点。文档可作为 HSTU bwd M0+M1 现状的准确 review 材料。
