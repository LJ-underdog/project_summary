# M7b stage 1 检查点 —— 重构零回归证明(coder,等 lead 亲验放行 stage 2)

stage 1 范围(按实现单):shape selector + 两 dispatch 重构取 `HstuBwdShape<MaxK>::Type` +
解 3 处 hd64 throw 换典范值 guard + harness `kN0_bwd` 随 hdim。**未加新 hdim 到 generate_instances**
(`BWD_HEADDIMS_M0` 仍 `[64]`)。基线 HEAD=`bf82a1d2`。**未 commit。**

改面(4 文件):
- 新增 `hstu_attention_bwd_shape.hpp`(selector,64/96/128/256 特化)
- `hstu_attention_batched_backward_dispatch.hpp`(取 selector + 2 guard)
- `hstu_attention_group_backward_dispatch.hpp`(取 selector + 1 guard)
- `example_hstu_attention_bwd.cpp`(两处 `kN0_bwd` 随 hdim:256→64,else 128)

---

## 证据 #1:selector<64> 与原硬编码逐字等价

stage-1 diff 显示两 dispatch **删除**的硬编码块:
```
FmhaBlockTile = sequence<32,128,64,32,64,32,32,64,64>
BlockWarps0=<1,4,1> BlockWarps1=<4,1,1> BlockWarps2=<1,4,1> WarpTile0=<16,16,32> WarpTile1=<16,16,16>
FmhaBwdShape = TileFmhaBwdShape<FmhaBlockTile, BW0,WT0, BW1,WT1, BW0,WT0, BW1,WT1, BW2,WT0, 0>
```
`hstu_attention_bwd_shape.hpp` 的 `HstuBwdShape<64>::Type` **逐字相同**(同 sequence 值、同
TileFmhaBwdShape 11 个类型实参顺序、同 `0` kMaxSeqLenQ)→ 类型恒等 → MaxK=64 实例化出同一 kernel。
(96 的 BlockWarps2=<2,2,1>、128/256 的 bm0=16、256 的 bn0=64 等差异仅在各自特化,不影响 <64>。)

## 证据 #2:hd64 bwd instance 与基线 byte-identical
```
$ git diff --stat bf82a1d2 -- 'example/ck_tile/18_hstu_attention/instances/*backward*'
(空)
```
stage 1 未重生成 instance(BWD_HEADDIMS 仍 [64]),16 个 hd64 bwd instance + 2 ref **未被触碰**。
(重构在 dispatch 头文件,instance 仅 `extern template ...<...,64>` 引用,内容不变。)

## 证据 #3:106 套件 + determ byte-identical repro 全绿,误差同量级
```
TOTAL 106   PASSED 106   FAILED 0   SKIPPED 0   exit 0
(含 7 个 determ repro 全部 byte-identical,含 2 个 M7a fp16 repro)
log: runs/test-20260611-053558.log
```
误差抽样(与 hd64 基线同量级,远低于容差):
- bf16 SiLU `pass-basic-attnscale1`  dQ err 1.2e-4 vs |ref| 5.13
- bf16 softmax `pass-sm-b-c1-causal`  dQ err 7.3e-9 vs |ref| 0.246
- bf16 SiLU combo `pass-mask-combo`   dQ err 7.6e-6 vs |ref| 4.91
- bf16 group determ softmax window    dQ err 3.8e-6 vs |ref| 0.660

构建:`runs/build-M7b-stage1.log` 0 error(增量 23 步,binary 新鲜重建)。

---

**结论:重构本身零回归证毕(类型恒等 + instance 未变 + 106/106 + repro byte-identical + 误差同量级)。**
guard 现为 `hdim_qk != hdim_v || hdim_qk != MaxK`(symmetric + 精确典范值;stage 1 entry 仍硬编码
MaxK=64,故非 64 hdim 仍被 guard 挡;stage 2 entry 换 HDIM_SWITCH 后同一 guard 自然放行典范多 hdim)。

**停,等 lead 亲验放行 stage 2(加 96/128/256 到 generate_instances + entry HDIM_SWITCH + 对拍 sweep)。**
