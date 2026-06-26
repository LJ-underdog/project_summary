# cross-attention Stage A 零回归证毕 — 硬检查点报告(待 lead 亲验)

基线 HEAD = `17515fcc`(M7c)。`-attn_scale=1.0`。**未动 reference / promoted self 逻辑 / harness / params。**

## 改面(Stage A,纯 false 腿等价重构,is_cross_attention 仍全 false)
`git diff --stat`(3 文件):
- `hstu_attention_batched_backward_dispatch.hpp`(+18/-4):`HstuBlockMasking<false>` → `BOOL_SWITCH(param.is_cross_attention, kIsCrossAttention)` 包 `<kIsCrossAttention>`,**嵌在 use_local switch 内**(未提到 pad/local 之上)。
- `hstu_attention_group_backward_dispatch.hpp`(+15/-...):RunSilu + RunSoftmax 各把 mask typedef + 下游双 pipeline/kernel/launch 包进 `BOOL_SWITCH(param.is_cross_attention, kIsCrossAttention)`。
- `hstu_attention_bwd_kernel.hpp`(+121/-...):4 处 mask 构造加 `if constexpr(FmhaMask/LocalMask/NoLocalMask::kIsCrossAttention)` 分叉 `make_hstu_cross_attention_block_mask_*`(cross 把 `seqlen_kv` 喂进 `seqlen_k` 槽,经 wrapper 保参序 R2/R3)else 现有 self builder 逐字。no_group SiLU(:404/414)+ softmax(:828/834)+ group SiLU(:1221/1231)+ group softmax(:1591/1602)。
- **未加 `max_seqlen_kv` 字段**(`git diff | grep -c max_seqlen_kv` = 0)、**未解钉 harness**、**未跑任何 cross case**(is_cross_attention 默认 false,套件全 self)。

## 1. co_symbols 字节同一性(★零回归核心证据)
基线在当前 HEAD `17515fcc` 净构建后抽取:`runs/cross-stageA-baseline.json`(66 obj = 64 batched instance + 2 group entry,**486 设备符号**)。
Stage A 改动 + 重建后 verify:
```
baseline symbols: 486  byte-identical: 486  MISSING: 0  DIFF: 0
(new pad-true/cross symbols in refactored objects: ~256, allowed)  exit 0
```
→ **所有 self 符号逐字节同一,0 DIFF = if constexpr 守卫无泄漏。** 新 `mask<true>`(cross)是全新 mangled 符号、允许。
日志:`runs/cross-stageA-cosym-verify.log`。

## 2. self 回归套件
```
python3 test/run_bwd_tests.py  →  TOTAL 220  PASSED 220  FAILED 0  SKIPPED 0  exit 0
```
(含 repro byte-identical 12 例)。日志:`runs/cross-stageA-suite.log` / `runs/test-20260615-063223.log`。

## 3. 编译时长 + 寄存器预算(§5 group 4-腿 / R11)
**寄存器(group bf16 entry obj,259 kernels,llvm-readelf metadata):**
- **scratch(private_segment_fixed_size)= 0,全部 259 kernels → 零 spill。**
- cross hd64 ~248–254 VGPR(与 self ~252 持平);cross hd96 ~383–385(self 384 持平);max VGPR = 426(hd256),max SGPR = 106。**全 < 512(CDNA4 上限),cross 与 self 寄存器持平、无 spill 回归。**
- self 符号已字节同一(见 §1)→ self 寄存器用量未变,cross 仅新增、不挤占。

**编译时长:**
- 全 bwd target 重建(改 kernel 头 → ~69 TU 重编 + link):**14m37s wall**(`runs/cross-stageA-build.time` 仅记 exit;实测 `time`)。
- **group bf16 entry 单 TU 独立编译 = 14m9s**(`{local,nolocal}×{cross,self}` 4 腿 × 全 hdim/causal/softmax/determ 一 TU 内)。**此 TU = 全 build 关键路径**(14m9 TU ≈ 14m37 全程)。
- 结论:能编过、寄存器/scratch 无回归,但 **group entry TU 已成 build 瓶颈(R11 兑现成本)**;per-hdim 拆 TU 是既有 M8 候选,本里程碑不做,记入风险。

## 裁决请求
Stage A 零回归证毕:co_symbols 486/486 byte-identical + 套件 220/220 + 编译预算见上(scratch=0、VGPR<512、group entry TU 14m9s 为关键路径)。**停,等 lead 亲验放行 Stage B。**
