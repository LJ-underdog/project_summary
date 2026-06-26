# M7b done — symmetric hdim{96,128,256} bwd(coder,等 reviewer+lead 闭合)

范围:解禁 **symmetric hdim ∈ {64,96,128,256}(hdim_qk==hdim_v)** 的 HSTU bwd。基线 HEAD=`bf82a1d2`(M7a)。
全程对拍 `-attn_scale=1.0`,**未动 reference / promoted pipeline / kernel 逻辑**,**未 commit**。
诚实纪律:**无任何 FAIL;status 据实 = in-progress 待闭合,非自标 promoted。**

---

## stage 1(重构零回归)—— 已经 lead 亲核放行
shape selector + dispatch 取 `HstuBwdShape<MaxK>::Type` + 3 处 throw 换典范 guard + harness kN0_bwd 随 hdim。
lead 亲验:selector<64> 同型、hd64 instance byte-identical、全套件 106/106 exit 0。详见
`M7b-stage1-checkpoint.md`。

## stage 2(加 hdim)—— 本轮完成

### 改面(8 改 + 1 新 + 50 instance)
- **新** `hstu_attention_bwd_shape.hpp`:`HstuBwdShape<MaxK>` 特化 64/96/128/256(蓝本 = FMHA bwd
  codegen gfx9 fp16/bf16 非 trload tile;<64> 与原硬编码逐字等价)。
- **2 dispatch**:取 `HstuBwdShape<MaxK>::Type`;guard = `hdim_qk!=hdim_v || hdim_qk!=MaxK`(symmetric
  + 精确典范值,挡非典范 silent-wrong)。
- **4 entry**(no_group/group × bf16/fp16):hardcode 64 → `HDIM_SWITCH(hdim_qk,hdim_v,MaxK,…)`。
- **generate_instances.py**:`BWD_HEADDIMS_M0 = [64,96,128,256]` → 64 batched bwd .cpp(16/maxk)+ 2 ref。
  **hd64 maxk_64 instance 与基线 `bf82a1d2` byte-identical**(git diff 空)。
- **harness**:两处 `kN0_bwd = (hdim_qk==256)?64:128`(hd256 tile bn0=64,修 determ workspace 失配)。

### 构建:0 error
`runs/build-M7b-stage2.log`(362 步,binary 429MB 新鲜重建)。group entry TU(单 TU 内 4 hdim×8 combo
实例化)是较重 TU 但构建顺利完成,**未成阻塞瓶颈** → 暂不拆 per-hdim instance(lead 授权的拆分留作
M8 可选优化,本单不需)。

### 对拍 sweep:128/128 PASS, 0 FAIL(`runs/run-M7b-sweep.log`)
每 hdim{64,96,128,256} × dtype{bf16,fp16} × 16 代表案 = 128:
- SiLU/softmax × causal{0,1} × {nomask,combo(5因子)} × {batched,jagged,group(hetero)}。
- **P1-1 cross(causal=0+num_target)每 hdim 覆盖**(batched silu/softmax + group softmax)。
- 非整除 seqlen(200)、determ multi-split(seq512;hd256 走 kN0=64)、group determ。
- 误差随 hdim 增大如预测,**容差未松**(沿用 bf16 2e-2/5e-2、fp16 5e-3/1e-2):
  - hd64 bf16 dQ≤~2e-3;hd128 bf16 dQ≤~0.016;**hd256 bf16 dQ≤~0.03 vs |ref|~10.6**(< atol 5e-2)。
  - softmax 路全 hdim dQ err ~1e-4 量级。fp16 路全 hdim dQ≤~4e-3。
  - per-hdim/dtype:h64 32P/0F,h96 32P/0F,h128 32P/0F,h256 32P/0F。

### hd256 寄存器/occupancy(lead 必做 #1)—— 无 spill,可行
`profile/M7b-hd256-resource.md`(rocprofv3 --kernel-trace):
- **Scratch=0(无溢出)**;VGPR 124(hd64)→172/184(hd256),< 256;AGPR 0。
- LDS 32KB→64KB(hd256 用满 CDNA4 64KB/CU → occupancy 1 WG/CU,与 occupancy=1 trait 一致;
  纯 perf 特征,非正确性问题,留 M8)。**hd256 保留交付,无需降级。**

### 套件升级:171/171 PASS exit 0(`runs/test-20260611-062413.log` 等)
- `reject-hdim128` 删 → hdim128/256 进 pass;**新增 2 个 guard reject 案**锁 silent-wrong:
  - `reject-hdim-noncanonical`(hdim=100→MaxK128 但 hdim≠MaxK)→ 我们的 guard throw(what() 已核)。
  - `reject-hdim-asymmetric`(hdim_qk=64≠hdim_v=128)→ guard throw(asymmetric 是 M7c)。
- 新增 **60 个 M7b pass 案**(hdim{96,128,256}×{bf16,fp16}×10 代表案,每 hdim 含 P1-1 cross)。
- 新增 **4 个 per-hdim determ byte-identical repro**(h96/h128/h256 + h256 group;**hd256 验 kN0=64 split**)。
- **12 个 determ repro 全 byte-identical**;TOTAL 171 / PASS 171 / FAIL 0 / SKIP 0 / exit 0。
  (= 旧 106 − 1 reject-hdim128 + 2 guard reject + 60 M7b pass + 4 M7b repro。)

## 完成度 / 缺口(如实)
- ✅ 能力边界扩展:SiLU+softmax 全模式(batched/jagged/group)× 全 5 mask × causal{0,1} × bf16+fp16 ×
  **hdim∈{64,96,128,256} symmetric** × atomic+determ。hd64 零回归(byte-identical + 106 子集全绿)。
- ⚠️ 范围限定(非缺口隐瞒):**hdim_qk≠hdim_v + 非典范任意 hdim(pad 路)= M7c**,已用 guard 显式 throw 挡住。
- ⚠️ hd256 occupancy=1(LDS-bound),perf 特征留 M8;group entry TU 较重(拆分留 M8 可选)。
- 盲区继承 M5/M5b:LSE 数值两侧共用 GPU fwd 产出,靠 fwd 里程碑兜底(hdim 未改变此结构)。

## 产物
- `hstu_attention_bwd_shape.hpp`(新);8 改源文件;50 instance(48 新 + 2 ref;hd64 byte-identical)。
- `runs/build-M7b-stage1.log`/`build-M7b-stage2.log`(0 err)、`runs/run-M7b-sweep.log`(128/128)、
  `runs/test-20260611-062413.log`(171/171)、`profile/M7b-hd256-resource.md`、`test/sweep_M7b.py`(新)、
  `test/run_bwd_tests.py`(改)。
- `candidates.jsonl` 加 M7b 行(status=in-progress)。**未 commit**;等 reviewer 对抗 review + lead 闭合。
