# M7c Stage 3 报告 —— group dispatch pad + poison(coder,停下等 lead 放行 Stage 4)

Stage 3 范围:group dispatch 镜像 batched 的 pad switch + 解 `ProblemFor` `<0,0>` 写死 + harness group
poison-pad,验 group canonical byte-identity + group 非典范 poison sweep + 套件零回归。**未 commit。**

## 改面
- `hstu_attention_group_backward_dispatch.hpp`:
  - `ProblemFor` 加 `bool kPadHeadDimQ,kPadHeadDimV` 模板参,`TileFmhaBwdTraits<0,0,...>` → `<kPadHeadDimQ,kPadHeadDimV,...>`(**弃用硬编码 `<0,0>`**)。
  - `RunSilu`/`RunSoftmax` 加 `<bool kPadHeadDimQ,bool kPadHeadDimV>` NTTP、删 `constexpr 0`、
    `ProblemFor<Mask,kPadHeadDimQ,kPadHeadDimV>`(Local/NoLocal 双 pipeline 各穿透,pad 深一层)。
  - `Run()`:取模派生 `pad_qk/pad_v`(读 `HstuBwdShape<MaxK>::kQKHeaddim/kVHeaddim`)+ `BOOL_SWITCH_2`
    包 RunSilu/RunSoftmax + guard 放松为 `hdim>MaxK||<=0 → throw`(与 pad switch 同改动落地,R1)。
- `example_hstu_attention_bwd.cpp` group 路(`run_group_hstu_bwd`):镜像 no_group poison —— over-alloc
  到 ahdim、输入 pad 尾 NaN、输出预填 NaN、`single_dq_acc_elems` 用 ahdim_qk、reference 喂真实 hdim
  抽取副本、比对抽真实列、store-skip dK/dV NaN 检查。off → ahdim==real → byte-identical。

## 构建:0 error(`runs/build-M7c-stage3.log`,binary 457MB,group pad legs 已进二进制)

## ★ 三项验证

### 1. group canonical byte-identity(`runs/M7c-stage3-byteident.txt`)
`co_symbols.py verify` vs Stage0 基线(64 batched instance + 2 group entry .o):
```
baseline symbols: 294   byte-identical: 294   MISSING: 0   DIFF: 0
(new pad-true symbols: ~576, allowed)
```
→ group(+batched)canonical pad=0 设备符号**逐位不变**(`ProblemFor<Mask,false,false>`==旧 `<0,0>` 同型)。

### 2. group 非典范 poison sweep:96/96 PASS,0 FAIL,全 store-skip=PASS(`runs/run-M7c-stage3-group-sweep.log`)
g{2,3,4} × 6 pair(64/128,128/64,80/80,100/64,48/96,192/256)× {bf16,fp16} × 8 配置:
g2 silu/sm c1、**g2 silu/sm c0-target(P1-1 cross)**、g2 silu-hetero(全 per-group 异构)、g3 sm c1、
g4 silu c1(singleton)、**g2 det-sm c1(group determ lock,*/256 走 kN0=64)**。全 `-poison_pad=1`:
- **load-zero**:全真实列梯度有限 + 对拍 PASS(NaN 输入未泄漏);容差未松。
- **store-skip**:全 store-skip=PASS(dK/dV pad 尾保持 NaN;asymmetric pad 区非空)。
- **每组合含 P1-1 cross**(g2 c0-target)+ group determ + per-group 异构。

### 3. canonical 套件零回归(`runs/M7c-stage3-suite.log`)
`TOTAL 172  PASSED 170  FAILED 0  SKIPPED 2  exit 0`(poison-off;group canonical 在内不变)。

## 能力边界(现,全模式)
SiLU+softmax × **batched/jagged/group** × 全 5 mask × causal{0,1} × bf16+fp16 ×
**hdim_qk/hdim_v ∈ (0,256] 任意(对称+非对称+非典范,经 head-dim pad)** × atomic+determ。
真 reject:hdim>256(HDIM_SWITCH else-throw)。

## 诚实限制(续 Stage 2)
poison over-alloc 吸收 OOB 写,GEMM4 dq_acc store-skip 由 lead 已代码核实(`bwd_kernel.hpp:373-381`
`sequence<false,(kPadHeadDimQ>0)>` + mop set/atomic_add,production 无 OOB 写)+ 真实列正确兜底。

**Stage 3 证毕,停,等 lead 亲核放行 Stage 4**(§6 全矩阵收尾 + 套件 2 个 SKIP 转 pass-asym +
真 reject 保留 + done.md + candidates)。未 commit。
