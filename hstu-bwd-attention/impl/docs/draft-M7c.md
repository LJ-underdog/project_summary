# docs/draft-M7c.md — HSTU bwd Milestone M7c: asymmetric + non-canonical hdim via head-dim padding

> 状态:**draft 闸门稿,等 lead/用户审批,未写实现码 / 未改库 / 未 build。** 基线 HEAD=`1ae97750`(M7b)。
> 来源:M7c 设计 workflow(6 路并行分析 + 综合 + 完整性 critique)产出,**lead 已据 critique must-fix 修正**(见文末「critique 解决记录」)。对拍铁律 `-attn_scale=1.0`。

## (0) 范围 — 一句话 + 边界

**范围:** 让 HSTU **bwd** 接受 `hdim_qk != hdim_v` 与非典范 head dim(如 80/48/100/200),办法 = **激活已接线但当前钉死(dead)的 head-dim pad 机制**,完全镜像 fwd 现有做法 —— 在 **per-MaxK dispatch body 内用运行时 BOOL_SWITCH**,**不**新增 instance 轴。

**In scope**
- 放松两个 bwd guard-throw(`hstu_attention_batched_backward_dispatch.hpp:378` softmax、`:397` SiLU;`hstu_attention_group_backward_dispatch.hpp:309`)—— **行号已 lead 核实**。
- 把硬编码 `constexpr kPadHeadDimQ/V = 0`(`batched:123-124, 233-234`;`group:139-140, 208-209`;group `ProblemFor` 别名 `:74`)改成运行时派生 bool,经 `BOOL_SWITCH_2` 喂入。
- pad 谓词用 **fwd 的取模式**(`batched_forward_dispatch.hpp:68-70`)。
- 修 harness `dq_acc` determ workspace sizing(`example_hstu_attention_bwd.cpp:303, 771`)按**选定的 MaxK** 而非 raw `hdim_qk` 取 kN0。
- 扩测试矩阵 + 加 `-poison_pad` OOB 正向证明(测试在 **`/root/workspace/hstu-bwd-impl/test/run_bwd_tests.py`**,见 §6)。

**Out of scope / 不移植(诚实边界)**
- `hdim > 256` —— 仍由 `hstu_attention_hdim_switch.hpp:31-33` else-throw 拒绝(结构性,不碰)。
- **非方形 tile**(`bhdq != bhdv`)。HSTU + FMHA 都只证方形:`HstuBwdShape<MaxK>` 令 `bhdq==bhdv==MaxK`(`hstu_attention_bwd_shape.hpp:36/61/86/114`),`fmha_bwd.py:430-438` 每个 preset 也 `bhdq==bhdv`。两个 hdim 都 round-up 到**同一个**方形 MaxK = ≥max(hdim_qk,hdim_v) 的最小典范 tile(`hdim_switch.hpp:9-34`),我们 pad **进**这个方形 tile,不引入非方形 tile。
- FMHA 的 **8-aligned vs 1-aligned pad 区分**(`fmha_bwd.py:294-295` `dpad/dvpad∈{0,8,1}`)**不移植**。HSTU bwd 用纯 **bool** pad → active 时对齐降到 1(scalar),和 fwd 一致、足够。见 §8 R9 的 `hdim=100` 注意。
- `reference_hstu_attention_bwd.hpp` 与 promoted-symmetric byte-identical 路:**不碰**(见 §4)。

---

## (1) 核心洞察 & 复用 fwd

**bwd 的 pad 机制结构上已完整,只是死的(dead)。** 四个 bwd kernel(no_group SiLU+softmax、group SiLU+softmax)的每个 DRAM view、每个 LDS window、每个 epilogue **都已经按 fwd 同款方式 honor `kPadHeadDimQ`/`kPadHeadDimV`** —— 只是被喂 `constexpr 0` 且被 guard-throw 挡在门外。

接线已活且正确的证据:
- DRAM view 用**真实** `hdim_qk`/`hdim_v` extent + tile 维 `kQKHeaddim`/`kVHeaddim` + `sequence<false,(kPadHeadDimQ>0)>`:`hstu_attention_bwd_kernel.hpp:307-345`(Q/K/V/dO)、`:373-385`(dq_acc)、`:429-453`(dk/dv);softmax `:745-867`;group `:1136-1204`。
- pipeline 对齐在 pad 时降 1、否则走 policy:`no_softmax_bwd_pipeline.hpp:75-94`;`with_softmax_bwd_pipeline.hpp:68-82`。GEMM static_assert 只要 `kQKHeaddim>=kK0`/`kVHeaddim>=kK2`(`:356-359`),**不**要求 `kQKHeaddim==kVHeaddim` → 方形 pad 合法。
- epilogue 已按 `(kPadHeadDimQ>0)`/`(kPadHeadDimV>0)` 模板化:`batched_backward_dispatch.hpp:162-171, 270-273`;group `:154,156,222,224`。

**复用 fwd 的精确范式**(`batched_forward_dispatch.hpp:68-95`):
```cpp
const bool pad_headdim_qk = !(param.hdim_qk % kQKHeaddim == 0);
const bool pad_headdim_v  = !(param.hdim_v  % kN1 == 0);   // bwd 用 kVHeaddim,见 §3A 注
BOOL_SWITCH_3(pad_seqlen_k,..., pad_headdim_qk,kPadHeadDimQK, pad_headdim_v,kPadHeadDimV, [&]{
    using HstuTraits = HstuAttentionFwdTraits<...,kPadHeadDimQK,kPadHeadDimV,...>;
});
```
fwd 的 `.cpp` instance **无** pad 轴 —— pad 在 MaxK-templated body 内运行时解析。bwd 的 `RunSilu`/`RunSoftmax` body 占同一槽。`BOOL_SWITCH_2` 已存在(`hstu_attention_bool_switch.hpp:20`),bwd 用 2-way 即可(不需 `pad_seqlen_k`,见下)。

**seqlen 自动受保护;head-dim 不受保护。** fwd 注释(`forward_dispatch.hpp:71-73`):seqlen_q 非最快维,`buffer_load/store` 自处理其 OOB。head-dim 是连续/最快维 —— 越界会读/写到相邻 head 的内存。这正是 pad 存在的全部理由,也是核心 silent-wrong 陷阱(§2)。

---

## (2) 逐 GEMM padding 计划 + OOB / pad-value / silent-wrong 分析

pad 模型(fwd,`tensor_view.hpp:543-582`):`pad = ceil(real/tile)*tile - real`;右 pad 是**逐元素 validity 谓词、非 clamp** —— OOB head-dim **load 零替代**、OOB head-dim **store 跳过**。故 padded 收缩列贡献恰好 0;padded 输出列从不写。**正确性依赖 OOB load 返回恰好 0 —— 这是整个里程碑赖以成立的不变式。**

两个独立 flag,各自跟踪自己那组 —— **绝不交叉接线**(否则静默毒化梯度):
- **QK 组**(pad flag `kPadHeadDimQ`,tile `kQKHeaddim`):Q、K、dQ、dK、dQ_acc —— 全 `hdim_qk` 宽。
- **V 组**(pad flag `kPadHeadDimV`,tile `kVHeaddim`):V、dO、dV、O —— 全 `hdim_v` 宽。
- dk epilogue → `(kPadHeadDimQ>0)`,dv epilogue → `(kPadHeadDimV>0)`(`batched_backward_dispatch.hpp:162-171`)。**两 flag 混用 = 把张量 round 到错 tile = 静默毒化梯度。** ⚠ SILENT-WRONG。

| GEMM | 角色 | pad 轴 | 机制 | OOB 风险 |
|---|---|---|---|---|
| GEMM0 | S=Q·Kᵀ | 收缩=hdim_qk | padded 列 load 0→加 0 | 安全 iff load-zero 成立 |
| GEMM1 | dV+=Pᵀ·dO | 输出 N1=hdim_v | epilogue 跳过 padded 列 store | ⚠ store 必须跳过,否则写进下一 head |
| GEMM2 | dP=dO·Vᵀ | 收缩=hdim_v | padded 列 load 0→加 0 | 安全 iff load-zero 成立 |
| GEMM3 | dK+=dSᵀ·Q | 输出 N=hdim_qk | epilogue store-skip | ⚠ store-skip |
| GEMM4 | dQ_acc+=dS·K | 输出/accum=hdim_qk | dq_acc 写掩码 | ⚠ atomic/写必须掩 padded lane |
| dsilu / PRE(`dot_do_o`)/ POST(`convert_dq`) | 逐元素 | n/a | 按**真实** hdim 循环 | pad 无关,不改 |

**silent-wrong 陷阱(pad!=0 build 时每条都要查):**
1. **load-zero 不变式** —— 任一 padded head-dim load 若返垃圾而非 0,GEMM0/GEMM2 加非零→污染 dK/dV/dQ;在 bf16 5e-2 容差下可能蒙混过关。只能由 §6 poison-pad 证。
2. **store-skip 不变式** —— GEMM1/GEMM3/GEMM4 epilogue 写过真实 hdim 会污染相邻 head。bwd dK/dV/dQ epilogue 必须吃**被 switch 的** bool,而非残留 `constexpr 0`(⚠ R4,§8)。
3. **交叉接线** —— QK flag 用到 V 张量(或反之),refactor 时极易写反。
4. **pad 谓词必须取模、非 `hdim != MaxK`** —— fwd 用 `!(hdim%tile==0)`。典范 hdim 下两者一致(故 no-pad 路无论哪种都安全),但取模是前向兼容的正确形式。

PRE/POST 确认 pad 无关:`bwd_kernel.hpp:1677` `dot_do_o` 循环 `v<hdim_v`;`convert_dq`/`reduce_convert_dq` 按真实 hdim_qk 元素数迭代。不改。

---

## (3) 改面清单

**A. dispatch —— 派生 pad + BOOL_SWITCH(核心改)**
- `hstu_attention_batched_backward_dispatch.hpp`:在 `Run()`(MaxK-templated body)算 `pad_qk = !(hdim_qk % kQKHeaddim == 0)`、`pad_v = !(hdim_v % kVHeaddim == 0)`(tile 维从 `HstuBwdShape<MaxK>` 读;**注:bwd 用 `kVHeaddim`,有意区别于 fwd 的 `kN1`**——fwd 的 V-tile head-dim 字段命名不同,bwd 的方形 tile 里就是 `kVHeaddim==bhdv==MaxK`)。把现有 `RunSilu<Mask>`/`RunSoftmax<Mask>` 包进 `BOOL_SWITCH_2(pad_qk,kPadHeadDimQ, pad_v,kPadHeadDimV, [&]{...})`,**在 `Run()` 里 hoist 一次**(fwd 风格)让 SiLU+softmax 共用,免 4× 重复;把两个 NTTP 作模板参穿进 `RunSilu`/`RunSoftmax`;**删** `:123-124` 与 `:233-234` 的 `constexpr 0`。
- `hstu_attention_group_backward_dispatch.hpp`:同,但更难 —— `Run()`(`:303-319`)直接调 `RunSoftmax/RunSilu` 无 BOOL_SWITCH 脚手架,且 `ProblemFor` 别名在 `:74` 把 `TileFmhaBwdTraits<0,0,...>` **写死**。**决策(§7 Stage3)**:弃用 `ProblemFor` 的硬编码 `<0,0>`,在 pad switch 内建 Problem(镜像 no_group),并嵌在已有 Local/NoLocal 切换下(pad 嵌套深一层)。删 `:139-140, 208-209` 的 `constexpr 0`。

**B. 放松 guard**(行号已核实)
- `batched_backward_dispatch.hpp:378`(softmax)与 `:397`(SiLU):把 `if(hdim_qk!=hdim_v||hdim_qk!=MaxK) throw` 换成 `if(hdim_qk>MaxK||hdim_v>MaxK) throw`(近乎死代码——`HDIM_SWITCH` 已保证 `MaxK>=max(hdim_qk,hdim_v)`;真正的 `hdim>256` 拒绝留在 `hdim_switch.hpp:31-33`)。保留 `assert(hdim_qk>0&&hdim_v>0)`。
- `group_backward_dispatch.hpp:309` 同。
- ⚠ **guard 放松与 pad switch 必须同一 commit 落地。** 放松 guard 而无 switch → 非典范 hdim 以 `kPadHeadDimQ/V=0` 运行 → head-dim OOB(最危险的顺序陷阱,R1)。

**C. traits/pipeline/epilogue honor pad** —— 无需改(§1 已 honor)。仅需核对 Traits **和**两个 epilogue 吃的是**同一个被 switch 的常量**(⚠ R4 残留 constexpr 陷阱)。

**D. generate_instances.py** —— **无 pad 轴、无新文件。** pad 是 per-symbol 内层 switch,非模板/instance 轴。现集 = 2 dtype×2 causal×2 softmax×2 determ×4 maxk = 64 batched `.cpp`,模板无 pad 字段,**不动**。(FMHA 把 pad 烤进文件名只因它没有 MaxK-template dispatch 层;HSTU 有那层。)加 codegen 轴会把文件数 4×(→256)且零收益。

**E. harness(`example_hstu_attention_bwd.cpp`)**
- ⚠ **修 `kN0_bwd`(`:303` 与 `:771`)**:现 `(hdim_qk==256)?64:128`,按 **raw** hdim_qk。非典范 hdim 桶到 MaxK=256(如 200)时 `hdim_qk!=256`→选 128,但 `HstuBwdShape<256>` 用 `kN0=64` → `num_splits` 减半 → determ `dq_acc` workspace **欠分配** → determ kernel 写越界 + POST reduce 读垃圾。**改:按选定 MaxK 取**(理想 `kN0 = HstuBwdShape<MaxK>::kN0`)。典范 hdim==MaxK 时与旧行为完全一致(canonical workspace 不变)。
- **poison-pad 改造(比设计初稿想的大,critique #3)**:`q/k/v/o/do`(`:240-248`)**和** 梯度输出 `dq/dk/dv`(`:255-259`)当前都按 **exact hdim** 分配。要 poison-pad 必须把**这些全部** over-allocate 到 MaxK:① 输入尾 head-dim NaN 填充(证 load-zero,§6.1);② 输出 `dq/dk/dv` 尾 pre-poison(证 store-skip,§6.4);③ 调整所有 stride/offset;④ reference 比对**只读真实 hdim 列**。group 路同改(`:708+`)。这是非平凡的 alloc+stride+compare 改造,非仅输入 over-alloc。
- 加 `-poison_pad` flag。

**F. 测试驱动(`/root/workspace/hstu-bwd-impl/test/run_bwd_tests.py` —— 真驱动,非 ck_hstu/scripts)**
- 该驱动结构:`_COMMON`/`_SM`/`_GB`/`_DET`/`_FP16` 等参数列表(`:39-56`)+ case 行(`:80+`)+ runner。当前 **171/171 exit 0**。M7c:① 按 §6 加 (hdim_qk,hdim_v) pair × dtype × 子矩阵;② 把 M7b 的 hdim guard-reject 案里属于 M7c 的(asymmetric/非典范)翻成 pass;③ **保留一个真 reject**(hdim>256)防 guard 静默消失;④ 加 poison-pad 案。**不要再引用任何 `scripts/run_bwd_tests.py` 路径或编造行号**(综合稿幻觉,已纠)。

**G. 不碰(byte-identical 契约)** —— `reference_hstu_attention_bwd.hpp`(oracle);promoted-symmetric 路靠 §4 的 false-false leg 保 byte-identical,不靠改 kernel/pipeline 源。

---

## (4) 零回归策略(pad=0 byte-identical 于 M7b)

**主张:** 对 exact-canonical hdim(`hdim%MaxK==0`),新 switch 选 `kPadHeadDimQ=kPadHeadDimV=0`,false-false leg 实例化出与 M7b **同一** Traits/Problem/Pipeline/Kernel 类型。

**为何成立:**
1. pad 只作**真值门**用 —— `kPadHeadDim>0` 给 `pad_tensor_view`、`kPadHeadDim?1:policy` 给对齐;`0` 复现 pre-M7c 展开,pad 从不参与算术。
2. MaxK/`HstuBwdShape` tile 选择与 pad **正交**,`hdim==MaxK` 时不变;`HstuBwdShape<64>` byte-identical 于 pre-M7b preset(M7b 已证)。
3. pad **非** codegen 轴(§3D)→ 64 个 instance `.cpp` 及其 MAIN kernel 符号不变;每 TU 只**新增** pad-true 符号。

**byte-identity 的要求(critique #4 已校正其严重性):**
- **modulo 谓词**(`!(hdim%tile==0)`)而非 `hdim!=MaxK` —— 这条是**真要求**(fwd 已证形式,`forward_dispatch.hpp:69-70`)。
- **false leg 的类型恒等是自动的,非陷阱**:`TileFmhaBwdTraits` 的 pad 形参是 `index_t`(`tile_fmha_traits.hpp:40-48`,static_assert ∈{0,8,1}),`BOOL_SWITCH` 给 `constexpr bool`;`bool false` 作 `index_t` NTTP **隐式转换为字面量 0**,故 `TileFmhaBwdTraits<false,false,...>` **本就是** `<0,0,...>` 同一类型 —— byte-identity 自动保住(fwd 也是直接把 raw bool 传进 traits)。写成 `COND?1:0` 是无害的防御性写法、**非**载荷性 silent-wrong 陷阱(此处从初稿的"会静默破坏零回归"**降级为 cosmetic**)。

**验证闸门(检查点,§7):** Stage0 先抓 M7b 64 个 instance 的 object-hash/反汇编**基线**(现 CI 只有 171/171 功能态);refactor 后(guard 放松 + switch 加,但只跑典范 hdim)diff 重生成的 false-false 符号 vs 基线,须 byte-identical;再跑 171/171 + hd64/96/128/256。绿了才进 pad-true。

---

## (5) instance / 编译时间预算

- **新 `.cpp`:0。** pad 是运行时 switch。
- **每 TU 成本(预期、非实测):** 64 个 batched TU(及 group 等价)各多编最多 **4× pipeline/kernel 实例化**(2×2 padQ/padV)于 `RunSilu`+`RunSoftmax`。对象码总量大致与 256-file codegen 拆分相当,但 build 并行度/增量重编粒度更利于少而肥的 TU。**(此为预期,非实测结论;Stage2 实测后回填。)**
- **预算吃紧的缓解:** canonical-symmetric 调用方只会命中 false-false,true leg 对常见路是纯码体积。可后续用 build flag 关 true-leg 编译。M7c 不需要;若二进制体积回归记 M8。
- **hd256 继承:** 桶到 MaxK=256 的 asymmetric(如 200/128)继承 `bn0=64` 翻倍的 determ split 数 + dq_acc workspace —— 由 §3E harness 修处理,不加 instance。

---

## (6) 测试矩阵 + 正向 OOB 正确性证明

基:克隆 M7b 生成法,跑 **(hdim_qk, hdim_v) pair** × {bf16,fp16} × M7b 子矩阵(SiLU/softmax、batched/jagged/group、causal{0,1}、atomic/determ、P1-1 causal=0+num_target)。symmetric-canonical pair 精简到 ~6 行;`*/256`、`48/*`、`100/*` 跑全行。

**pair:**
- asymmetric-canonical:`64/128`、`128/64`、`96/256`、`128/256`。
- 非典范-symmetric:`80/80`、`48/48`、`192/192`、`100/100`。
- asymmetric+非典范:`80/128`、`100/64`、`48/96`、`192/256`。
- **两个方向** `64/128` 与 `128/64` —— 证 QK/V flag 不交叉接线、dV 不被错 MaxK 截断。
- **determ lock:** `128/256` + `-deterministic` —— 证 §3E 的 `kN0_bwd` MaxK 重 key(修前会欠分配 dq_acc→reduce 越界)。
- **保留一个真 reject**(hdim>256,如 `512/512`)防 guard 静默消失。

**正向 OOB 正确性证明(别靠 bf16 容差 —— 5e-2 会掩 OOB 垃圾):**
1. **POISON-PAD(首要、最强):** harness over-alloc 输入到 MaxK 并 NaN 填 head-dim 尾(`-poison_pad`)。任一 padded load 没零替代 → 泄漏传播 → 输出 NaN/inf → 硬 FAIL。这**肯定性地**证 OOB 归零。(harness 现按 exact hdim 分配,§3E,故今天 OOB 静默读相邻有效内存。)
2. **差分 oracle:** 非典范 hdim vs `hdim==MaxK` 且尾零填 的运行比,真实 hdim 列梯度须吻合。
3. **保持容差** bf16 2e-2/5e-2、fp16 5e-3/1e-2(`example_hstu_attention_bwd.cpp:146-153`)。**禁松**(松了掩 OOB)。用 `attn_scale=1.0` 让 OOB 误差若存在则超 5e-2。
4. **epilogue store-pad 检查:** 输出缓冲尾 pre-poison,验 dq/dk/dv 写不碰它(无相邻 head 污染)。
5. **fp16 + hdim=100:** `100%8=4` 强制 scalar(align-1)load;含 `softmax+100+fp16` 压最差对齐路(见 §8 R9)。

**回归锁:** `171/171` M7b 全套须对 refactor 后的 false-false instance 全 PASS。

---

## (7) 分阶段实现顺序 —— 零回归 refactor 后设 HARD CHECKPOINT

**Stage 0 — 基线抓取(改之前)。** build M7b,抓 64 batched + group instance 的 object-hash/反汇编 —— §4 byte-identity 参照。

**Stage 1 — 仅 refactor(典范行为不变)。**
- batched:hoist `BOOL_SWITCH_2(pad_qk,pad_v)` 进 `Run()`,NTTP 穿进 `RunSilu`/`RunSoftmax`,删 `:123-124,233-234` 的 `constexpr 0`;放松 guard `:378/:397` 为 `>MaxK`;modulo 谓词派生 pad。**先不跑任何非典范 case。**

**★ HARD CHECKPOINT(闸门 —— 不绿不前进):**
1. 重生成的 false-false 符号 **byte-identical** 于 Stage0 基线;
2. `171/171` + hd64/96/128/256 PASS 不变;
3. grep 确认无残留 `<0,0>`/`constexpr 0`(Traits 与两 epilogue 同读 switched 常量)。

绿了才跑 pad!=0。

**Stage 2 — 激活 batched pad(true legs)。** build pad!=0 变体(首次编译 pad-true descriptor/LDS layout,潜在 static_assert 风险 R7,尤其 hd96 `<2,2,1>` warps、hd128 `bm0=16`)。harness `kN0_bwd` 重 key + `-poison_pad`。跑 §6 batched pair 含 poison-pad + `128/256` determ。

**Stage 3 — group dispatch。** 同款 switch;解 `ProblemFor` `<0,0>` 别名(`:74`,在 pad switch 内建 Problem,嵌 Local/NoLocal 下)。group canonical 先过 byte-identity,再 group 非典范(`-g` 传 >1 且 num_batch 整除,critique #gap4)。

**Stage 4 — 全矩阵 + sign-off。** §6 全矩阵 bf16+fp16、全模式、poison-pad on、真 reject 保留。文档化 hdim=100 状态(§8 R9)。

---

## (8) 红旗风险汇总

| # | 级别 | 风险 | 缓解 |
|---|---|---|---|
| R1 | ⚠ SILENT-WRONG | guard 放松而无 pad switch → 非典范 hdim 跑 pad=0 → head-dim(最快维)OOB,不像 seqlen 自动保护 | guard+switch 同一 commit(§3B);poison-pad(§6.1) |
| R2 | ⚠ SILENT-WRONG | load-zero 不变式 —— padded 收缩列(GEMM0/2)须返恰好 0,垃圾静默污染 dK/dV/dQ | poison-pad NaN 填;首个 pad!=0 build 必先跑它 |
| R3 | ⚠ SILENT-WRONG | 交叉接线 QK flag 用到 V(dk↔dv epilogue) | 测 `64/128` 与 `128/64`;review flag→组映射 |
| R4 | ⚠ SILENT-WRONG | epilogue 残留 stale constexpr —— Traits 拿 switched bool 但 epilogue 仍读 `(constexpr 0>0)` → 未 pad 尾部分垃圾 | checkpoint grep 残留 `<0,0>`/`constexpr 0` |
| R5 | ⚠ OOB-WRITE/REGRESSION | harness `kN0_bwd` 按 raw hdim_qk(`:303/:771`),MaxK=256 经非典范 hdim 欠分配 determ dq_acc | 按选定 MaxK 重 key;`128/256` determ lock 测 |
| R6 | (降级)cosmetic | false leg → 字面 0:**自动成立**(bool→index_t NTTP),非陷阱 | §4;`?1:0` 仅防御性,非必需 |
| R7 | COMPILE | pad!=0 从未编译过 —— 潜在 static_assert/descriptor/LDS-size 失败(hd96 `<2,2,1>`、hd128 `bm0=16`、hd256 `bn0=64`) | Stage2 每 MaxK 专门 pad-true build,四档都编过+数值过才算 done |
| R8 | PERF | pad active 强制对齐 1(scalar load)。正确性可接受;确保 false leg 保宽对齐 policy 让典范路无向量化回归 | §4 false leg 对齐落 policy;checkpoint 验 |
| R9 | SCOPE/SILENT-WRONG | hdim=100(`100%8=4`)bool pad→align1 *应*可行,但 `MaxVectorSize=8` 路在 odd remainder + align1 从未跑过 | 含 `100/100`、`100/64` fp16+softmax(§6.5)。若 vector-load 对齐 assert 触发且 bool pad 满足不了,**100 记为 documented reject**(诚实边界),不强松 |
| R10 | SCOPE | 非方形 tile(`bhdq!=bhdv`)不支持 —— 仅方形 MaxK + 独立 pad;若未来需 qk/v round 到不同 tile,那是 M8 | HDIM_SWITCH 选一个方形 MaxK≥max;明确标 |
| R11 | STRUCTURAL | group dispatch 被遗忘 —— `Run()` 无 switch 脚手架、`ProblemFor` 烤 `<0,0>`,易只发 batched 而 group 静默拒绝/错 pad | 专设 Stage3;group 进测试矩阵(`-g`) |

**实现期待解的开放问题:**
- Q1:CI 是否存 64 instance 的 byte-identical 基线,还是只有 171/171 功能态?→ Stage0 抓。
- Q2:全 MaxK(尤其 hd96 `<2,2,1>`)pad!=0 能否干净编译?→ 仅真 pad!=0 build 能证(Stage2)。
- Q3:group `ProblemFor` —— 模板化 pad 别名 vs switch 内建 Problem?→ 倾向内建(镜像 no_group),Stage3。
- Q4:production(非 harness)调用方是否传 unpadded 输出 stride(真实 hdim)?harness 传(`:256-266`);声称 integration-ready 前须确认 production 契约一致。

---

## critique 解决记录(lead,对完整性 critique 的 5 must-fix)
1. **幻觉测试驱动** → 已纠:真驱动 = `/root/workspace/hstu-bwd-impl/test/run_bwd_tests.py`(独立 workspace、171/171),删所有 `scripts/run_bwd_tests.py` 及编造行号(§3F/§6)。
2. **guard 行号** → lead 实测确认 batched `:378`(softmax)/`:397`(SiLU)、group `:309` **正确**,保留。
3. **poison-pad harness 范围** → §3E 扩明:输入(`:240-248`)+ 输出 dq/dk/dv(`:255-259`)都要 over-alloc 到 MaxK + stride/compare 改造,非仅输入。
4. **bool→index_t 非陷阱** → §4/R6 降级为 cosmetic(`<false,false>` 本就 == `<0,0>`);保留 modulo-谓词这条真要求。
5. **无 asymmetric 测试先例** → 明确:现有无任何 `hdim_qk!=hdim_v` 端到端测试(旧 hdim96_hdim64 脚本是每轮 symmetric + 仅 fwd),M7c **从零引入**该覆盖。
- 另:§5 对象码体积措辞改为"预期非实测"。
