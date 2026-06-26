# M7c done — asymmetric + non-canonical hdim via head-dim padding (coder, await reviewer+lead closure)

范围:让 HSTU bwd 接受 `hdim_qk != hdim_v` 与非典范 head dim(48/80/100/192/200…),办法 = **激活已接线但
死的 head-dim pad 机制**(运行时 `BOOL_SWITCH_2`,镜像 fwd),**不新增 instance**。基线 M7b `1ae97750`。
对拍 `-attn_scale=1.0`,容差禁松。**改面仅 4 文件;pipeline/kernel/reference 未碰。未 commit。**
status 据实 = **in-progress 待 reviewer+lead 闭合,非自标 promoted。**

## 改面(4 文件,`git diff --stat 1ae97750`)
- `hstu_attention_bwd_shape.hpp`(+20):每 MaxK 特化加 `kQKHeaddim/kVHeaddim`(方形=MaxK,pad 取模谓词)
  + `kN0`(bn0;hd256=64 余 128;harness determ 用)。
- `hstu_attention_batched_backward_dispatch.hpp`(±71):`RunSilu/RunSoftmax` 加 `bool kPadHeadDimQ/V`
  NTTP、删 `constexpr 0`;`Run()` 取模派生 pad + hoist `BOOL_SWITCH_2` + guard 放松 `>MaxK`。
- `hstu_attention_group_backward_dispatch.hpp`(±60):同上,且 **`ProblemFor` 弃用硬编码 `<0,0>` 改 pad
  NTTP**(Local/NoLocal 双 pipeline 各穿透)。
- `example_hstu_attention_bwd.cpp`(±303):harness poison-pad(no_group+group)—— over-alloc 到 MaxK、
  输入 pad 尾 NaN、输出预填 NaN、reference 喂真实 hdim 抽取副本、比对抽真实列、store-skip dK/dV 检查、
  `kN0_bwd`/`single_dq_acc_elems` 按选定 MaxK。`-poison_pad` flag。**off → byte-identical M7b。**

## 分阶段证据(Stage 0–4,均经 lead 亲核放行)
- **Stage 0** 基线:`co_symbols.py` 抽 64 batched instance + 2 group entry .o 的 gfx950 设备符号 →
  294 符号 sha256 基线(`runs/M7c-stage0-baseline.json`)。
- **Stage 1** batched refactor 零回归:**294/294 byte-identical 0 DIFF**;套件 171/171→172(2 skip)exit 0;
  无残留 `<0,0>`。**R7(pad!=0 首编)提前退役**(BOOL_SWITCH_2 编全 4 leg × 4 MaxK 干净)。
- **Stage 2** batched poison:harness 改造 + sweep **168/168 PASS 全 store-skip=PASS**;**R9 hdim=100
  (align-1)跑通 → 非 documented reject**。
- **Stage 3** group:dispatch + harness 镜像;**group canonical 294/294 byte-identical**;group poison
  sweep **96/96 PASS 全 store-skip=PASS**。
- **Stage 4** 收尾:套件永久化 M7c 覆盖。

## OOB 正向证明(poison-pad,非裸 PASS)
- **load-zero**:输入 pad 尾 NaN;若 masked load 泄漏 → NaN 传播 → 输出 NaN → check_err 硬 FAIL。
  全 sweep 真实列有限且对拍 PASS = OOB 归零证。
- **store-skip(dK/dV)**:输出 pad 尾预填 NaN;epilogue honor `kPadHeadDim>0` 跳写 → pad 尾保持 NaN。
  全 store-skip=PASS。**dQ 排除**(经 convert_dq 全量写 pad=0 by design;其 load-zero 由真实列证)。
- 非平凡:asymmetric pair 的 pad 区非空(16–192 NaN 列),检查可判伪。

## 全矩阵结果(最终 binary)
- **套件 `run_bwd_tests.py`:TOTAL 220 / PASSED 220 / FAILED 0 / SKIPPED 0 / exit 0**
  (= M7b 170 基 + 50 个 M7c poison-asserted pass;2 个旧 skip 转 poison pass;真 reject `hdim>256` 保留)。
  50 个 M7c 案带 `-poison_pad=1`,**套件本身用 poison 硬证 OOB**(runner 对 poison 案要求 4 marker:
  3 grad + store-skip)。5 pair(64/128,128/64,100/100,80/128,128/256)× {bf16,fp16} ×
  {no_group silu-c1 / no_group sm-c0-target(P1-1) / no_group determ(*/256→kN0=64)/ group sm-c1 /
  group silu-c0-target(P1-1)}。
- **batched poison sweep**:168/168 PASS(`runs/run-M7c-stage2-sweep.log`),12 pair × {bf16,fp16} × 7 配置。
- **group poison sweep**:96/96 PASS(`runs/run-M7c-stage3-group-sweep.log`),g{2,3,4} × 6 pair × 8 配置。
- 误差容差未松:bf16 最大 dQ≤0.016 vs |ref|~6.7–8.8(< atol 5e-2);softmax/fp16 远低。
- **canonical byte-identity**:294/294(`runs/M7c-stage3-byteident.txt`)。

## 能力边界(现)
SiLU+softmax × **batched/jagged/group** × 全 5 mask × causal{0,1} × bf16+fp16 ×
**hdim_qk/hdim_v ∈ (0,256] 任意(对称+非对称+非典范,经 head-dim pad)** × atomic+determ。
真 reject:**hdim>256**(`hstu_attention_hdim_switch.hpp` else-throw)。

## 诚实限制 / 范围外
- **dq_acc store-skip 测试盲区**:poison over-alloc 吸收 OOB 写,故 GEMM4 的 dq_acc store-skip
  本 harness 无法直证 → 由 lead 代码核实(`bwd_kernel.hpp:373-381` `sequence<false,(kPadHeadDimQ>0)>`
  + mop set/atomic_add,production exact-alloc 无 OOB 写)+ 真实列正确性兜底。**非 bug,是测试盲区。**
- **R9 hdim=100**(100%8=4,align-1):**跑通**(fp16 softmax + poison 全 PASS,无对齐 assert)→ 非 reject。
- **非方形 tile**(bhdq≠bhdv)、**hdim>256**:out-of-scope(方形 MaxK + 独立 pad;前者属 M8 若需)。
- LSE 数值盲区(M5/M5b 继承):两侧共用 GPU fwd LSE,fwd 里程碑兜底;hdim 未改变此结构。

## 产物
- 源:`hstu_attention_bwd_shape.hpp`、`hstu_attention_{batched,group}_backward_dispatch.hpp`、
  `example_hstu_attention_bwd.cpp`。
- 测试:`test/co_symbols.py`(新,byte-identity 工具)、`test/sweep_M7c.py`(新)、
  `test/sweep_M7c_group.py`(新)、`test/run_bwd_tests.py`(+50 poison 案 + runner poison-aware)。
- 日志:`runs/build-M7c-stage{1,2,3}.log`、`runs/M7c-stage0-baseline.json`、
  `runs/M7c-stage{1,3}-byteident.txt`、`runs/run-M7c-stage2-sweep.log`、
  `runs/run-M7c-stage3-group-sweep.log`、`runs/M7c-stage4-suite.log`。
- 阶段报告:`/tmp/hstu-bwd-design/M7c-stage{1,2,3}-*.md`。
- `candidates.jsonl` 加 M7c(status=in-progress)。**未 commit;等 reviewer 对抗+文档 review → 四方闭合。**
