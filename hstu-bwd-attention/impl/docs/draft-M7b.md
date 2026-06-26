# draft-M7b — hdim{96,128,256} symmetric(hdim_qk==hdim_v)bwd 设计稿

> 状态: **draft 闸门稿,等 lead 审批,未写实现码 / 未改库 / 未 build。**
> 基线 HEAD=`bf82a1d2`(M7a),能力边界 = SiLU+softmax 全模式 × 全 mask × causal{0,1} ×
> bf16+fp16 × **hd64** × atomic+determ;套件 106/106。本单只解禁 **symmetric hdim∈{64,96,128,256}**;
> **hdim_qk≠hdim_v 留 M7c**。对拍铁律 `-attn_scale=1.0`。

---

## 核心发现(决定整单形态)

**现 bwd dispatch 把 hd64 tile shape 写死,`MaxK` 是穿透但未被用来选 shape。**
`hstu_attention_batched_backward_dispatch.hpp:43-63` 和 group `:44-63` 都硬编码:
```
FmhaBlockTile = sequence<32,128,64,32,64,32,32,64,64>  // = bm0,bn0,bk0,bk1,bk2,bk3,bk4,bhdq,bhdv (hd64)
BlockWarps0=<1,4,1> BlockWarps1=<4,1,1> BlockWarps2=<1,4,1> WarpTile0=<16,16,32> WarpTile1=<16,16,16>
```
`MaxK` 仅出现在模板签名 + instance 文件名/extern,**不参与 shape 构造**。
→ 若现在直接把 `generate_instances` 的 headdim 轴加 96/128/256,只会生成 4 份**用同一 hd64 tile**
的 kernel(MaxK 值不同但 FmhaBwdShape 相同)——**silent-wrong**。
**∴ M7b 的第一要务 = 让 tile shape 随 MaxK 选;headdim 轴必须在 shape 选择就位之后才加。**

---

## 1. HDIM_SWITCH 复用

- 宏在 `hstu_attention_hdim_switch.hpp`,签名 `HDIM_SWITCH(HDIM_1, HDIM_2, CONST_NAME, ...)`,
  运行时 `hdim_qk/hdim_v` → 编译期常量 `MaxK`: `≤64→64, ≤96→96, ≤128→128, ≤256→256, else throw`。
- fwd(`hstu_attention_no_group_forward_bf16.cpp:17`)已用它把 MaxK 喂给 `run<...,MaxK>()`。
  **bwd 可直接复用同一宏**(已 include 进两个 dispatch)。
- symmetric 映射: hdim 64→MaxK64, 96→96, 128→128, 256→256(精确值,`hdim==MaxK`)。
- **改法**: 两个 entry(`hstu_attention_{no_group,group}_backward_{bf16,fp16}.cpp`)把现在硬编码的
  `64` 换成 `HDIM_SWITCH(param.hdim_qk, param.hdim_v, MaxK, [&]{ run_..._dispatch<...,MaxK>(...) })`,
  套在现有 `BOOL_SWITCH_3` 内层(镜像 fwd 的 BOOL_SWITCH_3→HDIM_SWITCH 嵌套)。

## 2. head-dim padding(最大风险点)—— 结论:**精确 symmetric 不需要 pad**

蓝本 = FMHA bwd codegen `example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`
`KernelComponentFactoryGfx9.get_dq_dk_dv_tiles("fp16"/"bf16", tr_load="")`(gfx950 非 trload 继承之):

| hdim | bm0 | bn0 | bk0 | bk1 | bk2 | bk3 | bk4 | bhdq | bhdv | warps0 | warps1 | warps2 | wtile0 | wtile1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64  | 32 | 128 | 64  | 32 | 64  | 32 | 32 | 64  | 64  | 1,4,1 | 4,1,1 | 1,4,1 | 16,16,32 | 16,16,16 |
| 96  | 32 | 128 | 96  | 32 | 96  | 32 | 32 | 96  | 96  | 1,4,1 | 4,1,1 | **2,2,1** | 16,16,32 | 16,16,16 |
| 128 | **16** | 128 | 128 | 16 | 128 | 16 | 32 | 128 | 128 | 1,4,1 | 4,1,1 | 1,4,1 | 16,16,32 | 16,16,16 |
| 256 | **16** | **64** | 256 | 16 | 256 | 16 | 32 | 256 | 256 | 1,4,1 | 4,1,1 | 1,4,1 | 16,16,32 | 16,16,16 |

(hd64 行与现 dispatch 硬编码**逐字相同** → 验证 shape selector 对 MaxK=64 必须产出完全一致的类型。)

**逐 GEMM 的 hdim 出现点 + 是否需 pad(精确 symmetric,hdim==bhdq==MaxK):**
- GEMM0 `S = Q·Kᵀ`:沿 hdim_qk 收缩,unroll = bk0(64/96/128/256 = 全 %wk0(32)==0)。
- GEMM1 `dK = dSᵀ·Q`:输出 N=hdim_qk,tile bhdq;hdim_qk==bhdq → 整除。
- GEMM2 `dP = dO·Vᵀ`:沿 hdim_v 收缩,unroll bk2(=hdim,%32==0)。
- GEMM3 `dV = Pᵀ·dO`:输出 N=hdim_v,tile bhdv;hdim_v==bhdv → 整除。
- GEMM4 `dQ = dS·K`:输出 N=hdim_qk,Gemm4WarpTile=`<wm0,wn0,min(wk0,bk4)>`=`<16,16,min(32,32)>`
  =`<16,16,32>`=WarpTile0(**四个 hdim 的 wk0=32 且 bk4=32,故 warp_tile2==warp_tile0 恒成立**
  → 现 dispatch 第 10 槽复用 WarpTile0 对全 hdim 仍正确)。
- load(Q/K/V/O/dO)+ dq_acc + convert/reduce POST:沿 hdim 维 = hdim_qk/hdim_v(harness 已按
  hdim 分配 buffer,`stride_dq_acc=dq_host.strides`,见 §3)。

**关键判定**: HDIM_SWITCH 选的 `MaxK == bhdq == bhdv`,且**目标只取精确 {64,96,128,256}**,
故 `hdim_qk % bhdq == 0` 恒成立 → **dpad=dvpad=0**(与 hd64 现状一致)。**精确 symmetric 不触发
head-dim padding。** (fwd 用运行时 `pad_headdim_qk=!(hdim_qk%kQKHeaddim==0)` BOOL_SWITCH 兜任意
hdim;那是为 **非典范 hdim**(如 80→tile96 需补 80→96)准备的,**不在 M7b 范围**。)

**🟥 silent-wrong 防线(必做)**: HDIM_SWITCH 对 `64<hdim_qk≤96` 之类**非典范值**(如 80)会静默选
MaxK=96,而 dpad=0 要求 `hdim%96==0` → 80 失配 → 静默错。**∴ dispatch `Run()` 入口加 guard**:
`if (hdim_qk != hdim_v || hdim_qk ∉ {64,96,128,256}) throw`(替换现 3 处 `!=64` throw)。
任意非典范 hdim 的 padding 支持 = **Option B(见 §4 末),本单不做、显式 throw 挡住。**

## 3. 改面清单

1. **新增 shape selector**(新文件,如 `hstu_attention_bwd_shape.hpp`):
   `template<index_t MaxK> struct HstuBwdShape;` 对 64/96/128/256 各一特化,导出 `using Type =
   TileFmhaBwdShape<...>`(按 §2 表)。**MaxK=64 特化必须与现硬编码逐字等价**(零回归基石,§4)。
2. **两个 dispatch**(`hstu_attention_batched_backward_dispatch.hpp`、`..._group_backward_dispatch.hpp`):
   删掉硬编码 FmhaBlockTile/WarpTile/BlockWarps,改 `using FmhaBwdShape = HstuBwdShape<MaxK>::Type;`;
   解 3 处 `hdim!=64` throw(`batched:390-391, 406-407`;`group:321-322`)→ 换成 §2 的典范值 guard。
   `kPadHeadDimQ/V` 仍恒 0(精确 symmetric)。
3. **两个 entry × {bf16,fp16}**(4 文件):hardcode `64` → `HDIM_SWITCH(...MaxK...)`(§1)。
4. **`generate_instances.py`**:`BWD_HEADDIMS_M0 = [64]` → `[64,96,128,256]`。
   - instance 数: no_group(batched)= causal{2}×softmax{2}×determ{2}=8 combos × dtype{2} ×
     **hdim{4}** = **64** 个 .cpp(现 16)+ 2 ref.hpp(ref 内含 4 hdim extern)。**+48 个重型 bwd TU**。
   - group 不走 instance 文件(entry 内直接实例化)→ 不增 .cpp,但 **group entry TU 体积 ×4 hdim**。
   - **🟥 编译时间风险**: bwd kernel 是 5-GEMM 重模板,+48 TU 显著拖长;group 2 个 entry TU 各
     涨 4×(单 TU 内 4 hdim × 8 combos 实例化)→ 可能成为最慢 TU。**缓解**: 增量落地(§7,先 hd128)
     + 若 group entry TU 过慢,考虑把 group 也拆成 per-hdim instance 文件(本单可选,记风险)。
5. **harness `example_hstu_attention_bwd.cpp`**:
   - q/k/v/o/do/dq/dk/dv/dq_acc 已按 `hdim_qk/hdim_v` 分配(`:240-263, :305-306`)→ buffer **无需改**。
   - **🟥 `kN0_bwd` 硬编码 = 128(`:301`)用于 determ workspace `num_splits` 求值**。dispatch 的
     `num_splits` 用 `Pipeline::kN0`(=bn0):hd64/96/128 bn0=128 ✓,**hd256 bn0=64** → dispatch
     split 数 = 2× harness 估值 → **determ workspace 写越界 / reduce 错**。**必改**: harness
     `kN0_bwd` 随 hdim 取(256→64,else 128),两处(no_group `:301`、group `:771` 附近)同步。
   - 加 `-hdim_qk/-hdim_v` 已有(默认 64),测试用例显式传值即可。
6. **CMake**:见 §5。
7. **reference**(`reference_hstu_attention_bwd.hpp`):CPU 通用 hdim 循环,**不改**(只读,铁律禁改 reference 逻辑)。

## 4. 是否动 promoted pipeline/kernel —— **不动,零回归可保全**

- promoted 的 `hstu_attention_{no_softmax,with_softmax}_bwd_pipeline.hpp` / `..._bwd_kernel.hpp` /
  group kernel **全部按 `Problem`(内含 `FmhaBwdShape`)模板化**,shape 是注入的 → **M7b 不需要碰它们**。
  M7a 的「零碰库逻辑」纪律在 M7b **延续**。
- **🟩 hd64 byte-identical 策略**: shape selector 的 `MaxK==64` 特化产出与现硬编码**完全相同的类型**
  → MaxK=64 路实例化出**逐位相同**的 kernel。改动只在「shape 从哪来」,不在 shape 内容、不在 pipeline。
  → 不需要 `if constexpr(MaxK==64)` 走旧路这种 hack;**selector 同型即同码**。
- **验证零回归**: (a) 重生成 instance 后,hd64 的 16 个 bwd instance 与基线 `bf82a1d2` 对应文件除
  「新增 96/128/256 行」外内容不变;(b) 跑现 106 套件 + determ byte-identical repro 全绿、误差与
  基线同量级(§6)。
- **Option B(非典范任意 hdim,M7b 不做)**: 若未来要支持任意 hdim≤256,照抄 fwd 的运行时
  `pad_headdim_qk/v` BOOL_SWITCH → `TileFmhaBwdTraits<kPadHeadDimQ,kPadHeadDimV,...>` + epilogue
  `(kPadHeadDim>0)`。代价 = instance 再 ×(2~4)pad 组合 + 需独立验 pad-load 的 OOB 数值。**留 M7c/后续。**

## 5. fwd over-link 瘦身(§5)

- 现状(M7a):CMake `file(GLOB FWD_BF16_INSTANCE_SRCS instances/*forward_bf16*)` +
  `*forward_fp16*` → **全部 288 个 fwd instance**(4 hdim × 3 mode × 2 dtype × 多 causal/softmax/
  storelse 组合)拉进 bwd target。hd64-only 时代 96/128/256 fwd 是**死重**。
- **M7b 后**:bwd 真用 4 hdim → fwd 的 4 hdim instance **不再是死重**(harness 跑 fwd 产 O+LSE 需要)。
  故 §5 的主要浪费**随 M7b 自动消解**;glob 维持现状即正确,**不建议在本单收窄 glob**
  (收窄需同步裁 fwd entry 的 `*_instances_ref.hpp` extern 引用,否则 link 未定义符号 → 高风险)。
- **残留浪费**(可记 M8,本单不动):fwd 的 bias/dropout 变体 instance 永不被 bwd harness 调用却仍编译。
  裁它需精确匹配 fwd entry 实际引用集 —— 风险/收益不划算,**留 perf 单**。
- 结论入 draft: **CMake 仅按 §3.4 加 bwd instance glob 自然涵盖(已是 `*backward*.cpp` 通配),
  fwd glob 不动。** 唯一 CMake 动作:确认 4 hdim fwd instance 已在 glob 内(已在)。

## 6. 测试矩阵设计

每 hdim ∈ {64,96,128,256} 覆盖(`sweep_M7b.py` 仿 `sweep_fp16.py`,全 `-attn_scale=1.0`):
- {SiLU,softmax} × {bf16,fp16} × causal{0,1} × {batched,jagged,group} × {atomic,determ}。
- 代表性 mask:nomask / causal / window / combo(5 因子)/ **causal=0+num_target(P1-1,每 hdim 必cover)**。
- 非整除 seqlen(200/130)、单 batch、tiny、group 多 batch 异 seqlen。
- determ byte-identical repro:每 hdim ≥1(尤其 **hd256**,验 §3.5 的 kN0=64 split 修复)。
- **🟥 容差是否随 hdim 变**: dQ/dK 沿 hdim 收缩(GEMM4/GEMM1),hdim 越大累加项越多 → abs err 可能略增;
  dV/dP 沿 seqlen 收缩,与 hdim 无关。**策略**: **沿用现 bf16(rtol2e-2/atol5e-2)、fp16(rtol5e-3/
  atol1e-2)不预松**;若 hd256 某案 FAIL,先判数值真错 vs 容差,**如实报,别为凑 PASS 调容差**(M7a 纪律)。
- 套件:把 `reject-hdim128` 升级为 `pass-hdim128-*`(及补 hdim96/256 案),`run_bwd_tests.py` 新 TOTAL。

## 7. 增量实现顺序建议

1. **shape selector + hd64 零回归证明**(还不加新 hdim):重构 dispatch 取 selector<64>,跑 106 套件 +
   repro 全绿、误差同量级、hd64 instance 内容不变 → 锁定「重构本身零回归」。
2. **hd128 symmetric**(2 的幂、最常用、warps 同 hd64 结构):batched SiLU 先通对拍 → 全 no_group
   (jagged+softmax+determ)→ group;每步对拍 `-attn_scale=1.0`。
3. **hd96**(非 2 幂,验 warps2=`<2,2,1>` 分支)+ **hd256**(bm0=16/bn0=64,验 §3.5 harness kN0=64
   修复 + 寄存器压力)。
4. **fp16 × 全 hdim**(M7a 已证 fp16 复用同码路,预期顺滑)。
5. 收尾:`sweep_M7b.py` 全表 + 套件升级 + bf16/fp16 零回归 + 每 hdim determ repro + done.md/candidates。

## 风险红旗汇总

- 🟥 **shape 必须随 MaxK 选**(否则加 headdim 轴 = silent-wrong);selector<64> 须与现硬编码同型(零回归)。
- 🟥 **harness `kN0_bwd=128` 对 hd256(bn0=64)失配** → determ workspace 越界;必随 hdim 改。
- 🟥 **非典范 hdim(如 80)经 HDIM_SWITCH 静默选大 tile + dpad=0** → silent-wrong;Run() 入口加典范值 guard。
- 🟧 **编译时间**:+48 bwd TU + group entry TU ×4;增量落地,group 过慢则考虑拆 instance 文件。
- 🟧 **hd256 寄存器压力**(bn0=64/大 bk):gfx950 上验 occupancy/spill,M1 的 VGPR 加法模型口径复核。
- 🟩 **不动 promoted pipeline/kernel**;fwd glob 不动(M7b 后 4 hdim fwd 不再是死重)。

> draft 就绪。**未写实现码、未改库、未 build。** 等 lead 审 §1-7 + 风险红旗后再派实现单。
> 若 lead 认为 group entry TU 编译时间不可接受,可在实现单决定「group 也拆 per-hdim instance」或
> 「先只上 no_group hdim,group 留下一档」——本 draft 已把两种路径的取舍点标出。
