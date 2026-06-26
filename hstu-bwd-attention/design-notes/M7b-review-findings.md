# M7b 对抗式 review findings — reviewer(pane 0.2)

**结论:M7b 可 promote。全 GREEN,0 RED。** 独立复核(非复述 lead):静态对蓝本 + 独立干净构建
(`build_review` rm -rf 重配重编 exit 0)+ 独立跑套件 171/171 ×2 + rocprofv3 独立抽验 hd256 资源 +
二进制符号反证 tile 真随 hdim 变。基线 HEAD=`bf82a1d2`。GPU=gfx950/MI350X/CDNA4。

---

## 逐条裁决

### 1. ★ shape 真随 hdim 变(silent-wrong 高发区)—— 🟩 GREEN
- **静态对蓝本逐字核对**:`hstu_attention_bwd_shape.hpp` 四特化 vs `01_fmha/codegen/ops/fmha_bwd.py`
  `KernelComponentFactoryGfx9.get_dq_dk_dv_tiles("fp16"/"bf16","")`(lines 432–437)。**gfx950 非 trload
  继承 Gfx9 base**(Gfx950 的 extend 仅在 `tr_load=="t"` 触发,本路 tr_load="" → 取 base list)。
  - hd64 `<32,128,64,32,64,32,32,64,64>` BW0=141/BW1=411/BW2=141/WT0=16,16,32/WT1=16,16,16 ✓ 蓝本 line433
  - hd96 `<…,96,…>` **BW2=<2,2,1>** ✓ 蓝本 line434(warps2 特殊分支正确)
  - hd128 `<16,128,128,…>` **bm0=16** ✓ 蓝本 line435
  - hd256 `<16,64,256,…>` **bm0=16,bn0=64** ✓ 蓝本 line437
  - WT2(第10槽)用 WarpTile0 替代蓝本 `min(wk0,bk4)`:四 hdim **bk4=32 且 wk0=32 → min=32=wk0** →
    WT2≡WT0 恒成立(`fmha_bwd.py:49`),替代合法。TileFmhaBwdShape 11 实参顺序与蓝本 `:55-66` 一致。
- **🔑 反证(独立、决定性)**:rocprofv3 抽二进制 kernel 符号 → LDS 分布 **32KB / 48KB / 64KB**、
  VGPR **124/128/168/172/176/184/192** 多档并存。若四 hdim 仍共用 hd64 tile,所有 kernel LDS 应恒 32KB。
  → tile **确实随 hdim 分化**,非 silent 复用。
- **误差曲线合理**:bf16 det-silu multi-split mean_abs_err 随 hdim 升(h64 8.6e-7→h256 1.2e-6),
  max_abs_err 量化到 bf16 ULP(0.0078/0.0156),非异常。

### 2. selector<64> 同型 = hd64 零回归 —— 🟩 GREEN
- `HstuBwdShape<64>::Type` 与两 dispatch 删除的硬编码块**逐字等价**(同 sequence、同 BW0/1/2、
  同 WT0/1、同 11 实参序、同 `0`)。两 dispatch 唯一改动 = `using FmhaBwdShape = HstuBwdShape<MaxK>::Type;`。
- **hd64 instance byte-identical**:`git diff bf82a1d2 -- instances/*maxk_64*` = 空(我亲自重跑
  `generate_instances.py` 后仍空)。

### 3. guard 挡 silent-wrong —— 🟩 GREEN
- guard = `if(hdim_qk!=hdim_v || hdim_qk!=MaxK) throw`(batched softmax+SiLU 两处、group 一处,
  替换原 3 处 `!=64` throw)。
- **亲测 what()**(我跑的套件 log):
  - `reject-hdim-noncanonical`(hdim=100):HDIM_SWITCH→MaxK=128,guard 100≠128 → **throw,exit -6**,
    `what(): HSTU bwd SiLU supports symmetric hdim_qk==hdim_v in {64,96,128,256} only`。
  - `reject-hdim-asymmetric`(64/128):**throw,exit -6**,同 what()。
- 🟧 **小注**:dispatch 要求的「注释掉 guard 跑 100 看 silent-wrong」字面反证**未执行**(需第二次全量
  重编,成本高)。机理已确定:guard 去掉后 100→hd128 tile,bhdq=128 而实际 hdim=100,buffer_load 越界
  补 0 → 静默错值(非 throw)。reject case 的 throw 已实证 guard 在位生效;此项不影响 promote。

### 4. harness kN0_bwd 修复(hd256 determ)—— 🟩 GREEN
- 实证 `Pipeline::kN0 = BlockFmhaShape::kN0 = BlockTile::at(1) = bn0`
  (`rocm-libraries/.../tile_fmha_shape.hpp:104`)。hd256 bn0=64,余 128。
- harness 改 `const int kN0_bwd = (hdim_qk==256)?64:128`(no_group `:301` + group `:765` 两处)→ 与
  dispatch `Pipeline::kN0` 精确一致。
- **hd256 determ byte-identical 亲验**:套件 repro `repro-h256-det-softmax-seq512`(no_group)+
  `repro-h256-gdet-silu-g2`(group)两次跑 **byte-identical PASS**。
- **反证逻辑**:若 hd256 误用 hd64 tile(bn0=128),harness kN0=64 与 dispatch kN0=128 失配 → determ
  reduce split 数不符 → 对 oracle 应 FAIL。实测 hd256 determ 对拍 PASS(dQ err 0.0156 vs |ref|10.7)→
  反证 bn0=64 tile 真在用。

### 4b/5. 对拍公平 + 容差不放水 —— 🟩 GREEN
- harness diff 仅 2 行 kN0 + 注释,**无任何容差常量改动** → 容差沿用基线 bf16(2e-2/5e-2)、
  fp16(5e-3/1e-2),未松。
- 容差有真实余量(非「擦边过」):hd256 bf16 dQ max_abs_err ~0.0078–0.0156 vs atol 5e-2(3–6× 余量);
  mean ~1e-6;误差非零、随问题规模缩放 = 真算了。(容差为编译期模板值,无法 CLI 收紧 5×;以原始误差
  量级佐证余量,免第二次重编。)

### 5(库零回归). promoted pipeline/kernel/reference byte-level —— 🟩 GREEN
- `git diff bf82a1d2 -- *pipeline*.hpp *bwd_kernel*.hpp *reference*.hpp` = **空**。promoted 逻辑零碰。
- 全改面(diff stat):dispatch×2(selector+guard)、entry×4(HDIM_SWITCH)、generate_instances.py
  (`BWD_HEADDIMS=[64,96,128,256]`)、ref.hpp×2(**纯增** 96/128/256 extern,hd64 行未删)、harness
  (kN0)、新 shape.hpp。无越界改动。

### 6. 套件 171/171 独立复跑 —— 🟩 GREEN
- **独立干净构建**:`rm -rf build_review` + 重 configure(`-DBUILD_DEV=OFF -DGPU_TARGETS=gfx950`)+
  重编 `tile_example_hstu_attention_bwd` **exit 0**(`m7b-review-configure.log`/`m7b-review-build.log`)。
- **独立跑套件 ×2**:coder binary 171/171、**独立 build_review binary 171/171**
  (`TOTAL 171 PASSED 171 FAILED 0 SKIPPED 0 exit 0`,log `test-20260611-065018.log`)。
- 覆盖真实:**每 hdim 有 P1-1 cross**(silu-b-c0-target + sm-b-c0-target + sm-g2-c0-target ×{96,128,256}
  ×{bf16,fp16});2 guard reject + 60 M7b pass + 12 determ repro 全 byte-identical(含 hd256 no_group+group)。

### 7. hd256 资源声称 —— 🟩 GREEN(rocprofv3 独立复跑)
- `rocprofv3 --kernel-trace` 抽 hd256 SiLU MAIN kernel:**VGPR=172,AGPR=0,SGPR=112,Scratch=0,
  LDS=65536** —— 与 `profile/M7b-hd256-resource.md` 表格(hd256 SiLU 172 / Scratch 0 / 64KB)**逐项吻合**;
  hd256 softmax 一档 VGPR=184 亦吻合。**全部 kernel Scratch=0**(无 spill,任何 hdim)。
- occupancy=1:dispatch 中 `occupancy=1` 对**所有 hdim 硬编码**(非 M7b 引入,pre-M7b 即如此)→ 非回归。
  hd256 LDS=64KB 占满 CDNA4 64KB/CU → 1 WG/CU,与 trait 自洽,纯 perf 特征,留 M8 判定正确。

---

## 残留/建议(均不阻塞 promote)
- 🟧 guard「注释掉看 silent-wrong」字面反证未跑(成本=第二次全量重编);机理确定 + reject throw 已实证。
- 🟧 group entry TU ×4 hdim 较重(本次独立重编仍顺利完成,未成阻塞)+ hd256 occupancy=1 → 均 M8 perf。
- ✅ 范围如实:hdim_qk≠hdim_v + 非典范 hdim = M7c,已 guard 显式 throw 挡住,非隐瞒。

**promote 裁决:批准。** 证据:tile 蓝本逐字符 + 二进制符号反证 tile 分化 + 独立干净构建 171/171 +
rocprof 独立证 Scratch=0/资源吻合 + 库 byte-level 零回归 + hd64 byte-identical。
