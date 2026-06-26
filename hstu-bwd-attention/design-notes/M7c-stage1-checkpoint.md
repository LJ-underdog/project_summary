# M7c Stage 1 硬检查点 —— refactor 零回归(coder,停下等 lead 亲验放行 Stage 2)

Stage 1 范围(按实现单 + draft §7):仅 refactor —— 激活 batched dispatch 已接线但死的 pad 机制
(运行时 `BOOL_SWITCH_2` 镜像 fwd)+ 放松 guard + harness `kN0_bwd` 按选定 MaxK。**未碰 group(Stage 3)、
未加 poison-pad(Stage 2)、未跑任何非典范 case 做断言。** 基线 HEAD=`1ae97750`(M7b)。**未 commit。**

## 改面(Stage 1)
- `hstu_attention_bwd_shape.hpp`:每 MaxK 特化加 `kQKHeaddim/kVHeaddim`(方形=MaxK,pad 取模谓词用)
  + `kN0`(bn0;hd256=64,余 128;harness determ 用)。
- `hstu_attention_batched_backward_dispatch.hpp`:`RunSilu`/`RunSoftmax` 加 `bool kPadHeadDimQ,kPadHeadDimV`
  NTTP、删 `constexpr 0`;`Run()` 取模派生 `pad_qk=!(hdim_qk%kQKHeaddim==0)`、`pad_v=!(hdim_v%kVHeaddim==0)`,
  hoist `BOOL_SWITCH_2(pad_qk,pad_v)` 一次包住 SiLU+softmax;guard 放松为 `hdim_qk/hdim_v>MaxK||<=0 → throw`
  (hdim>256 的真 reject 仍在 `hdim_switch.hpp`)。**guard 放松与 pad switch 同一改动落地(R1)。**
- `example_hstu_attention_bwd.cpp`:加 `bwd_selected_maxk`/`bwd_kN0_for`,两处 `kN0_bwd` 改按**选定 MaxK**
  (修 R5:非典范 hdim 桶到 256→kN0=64 的 determ workspace 欠分配)。

## 构建:0 error —— 且 R7 提前退役
`runs/build-M7c-stage1.log`(71 步增量,binary 429→**445MB**)。**注**:`BOOL_SWITCH_2` 在编译期实例化
全部 4 个 pad leg,故 Stage 1 build **已编译 pad-true 路** → **R7(pad!=0 首次编译风险)提前退役**:
4 pad 组合 × 4 MaxK(含 hd96 `<2,2,1>`、hd128 bm0=16、hd256 bn0=64)**全部干净编过**,无 static_assert/
descriptor/LDS 失败。(binary 增大即 pad-true 符号已进二进制的实证。)

## ★ 三项检查点证据

### 证据 1 — false-false 设备符号 byte-identical(`runs/M7c-stage1-byteident.txt`)
工具 `test/co_symbols.py`(llvm-objdump --offloading 抽 gfx950 code object → 反汇编 → 每符号取指令
编码流 sha256,地址列剥除故重排不影响)。Stage0 基线 = 64 batched bwd instance + 2 group entry .o 的
**294 个 pad=0 设备符号**(`runs/M7c-stage0-baseline.json`)。
```
baseline symbols: 294   byte-identical: 294   MISSING: 0   DIFF: 0
(new pad-true symbols in refactored objects: ~384, allowed)
```
→ **典范(pad=0)路设备码逐位不变**;新增 ~384 个 pad-true 符号(允许)。

### 证据 2 — 套件 exit 0,典范全绿(`runs/M7c-stage1-suite.log`)
```
TOTAL 172   PASSED 170   FAILED 0   SKIPPED 2   exit 0
```
- **169 个 M7b 典范案(全 hd64/96/128/256 symmetric + 12 determ byte-identical repro)全 PASS 不变**。
- +1 新 `reject-hdim-gt256`(hdim=512)PASS = 结构性 reject 仍生效。
- **2 个 SKIP**:原 M7b guard-reject 案(`hdim=100` 非典范、`64/128` asymmetric)—— guard 现按设计放松,
  其 `reject` 期望已失效。**它们当前确实跑出 3/3 PASS,但我【不】采信**:harness 尚无 poison-pad,
  OOB head-dim load 会读到相邻有效内存、bf16 5e-2 容差可能掩盖垃圾(draft R2/§6.1)。故标 SKIP,
  真验证留 Stage 2 poison-pad → Stage 4 转 pass-asym-*。**诚实:不拿"裸 PASS"充数。**

### 证据 3 — 无残留 stale constexpr(R4)
`grep` batched dispatch:无 `kPadHeadDim=0`/`TileFmhaBwdTraits<0`/`Epilogue<...,0>`;Traits(:128,:236)
与全部 4 个 epilogue(:166/:171 silu、:270/:272 softmax)**同读被 switch 的 NTTP** `kPadHeadDimQ/V`。

## 结论
**Stage 1 refactor 零回归证毕**(典范设备码 byte-identical + 典范套件全绿 + 无 stale constexpr),
且 R7 编译风险已提前退役。**停,等 lead 亲核放行 Stage 2**(harness poison-pad 改造 + 跑非典范 batched
pair 含 poison-pad NaN 硬证 OOB 归零 + `128/256` determ lock)。
未触 group(Stage 3)、未 commit。
