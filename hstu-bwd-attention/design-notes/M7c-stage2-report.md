# M7c Stage 2 报告 —— batched poison-pad 验真(coder,停下等 lead,之后 Stage 3 group)

Stage 2 范围:harness poison-pad 改造(no_group/batched)+ 跑非典范 (hdim_qk,hdim_v) pair 含 poison-pad
NaN 硬证 OOB 归零/store-skip + 128/256 determ lock,每 pair 含 P1-1 cross,**容差禁松**。
**group 未碰(Stage 3)。未 commit。** 基线 M7b `1ae97750` + Stage1 refactor。

## harness 改造(`example_hstu_attention_bwd.cpp`,no_group,全程 `poison_pad` 门控,off=旧行为)
- 新 `-poison_pad` flag。`ahdim_{qk,v} = poison ? sel_maxk : real`;GPU 端 host/device 张量按 ahdim
  over-alloc(stride/alloc/upload 随张量 shape 自动变 padded);`single_dq_acc_elems` 用 ahdim_qk。
- 输入 q/k/v/dO 的 pad 尾列 NaN 填(证 load-zero:任一 OOB load 泄漏 → NaN → 输出 NaN → 硬 FAIL)。
- 输出 dq/dk/dv device 预填 NaN 上传(证 store-skip)。
- CPU **reference 喂真实 hdim 抽取副本**(`extract_real`),输出比对也抽真实列 → reference 不被污染。
- store-skip 检查:**dK/dV** pad 尾必须保持 NaN(epilogue honor kPadHeadDim>0 跳写)。**dQ 排除**
  (经 POST convert_dq 全量写,pad 尾=convert(0)=0 by design,无害;dQ load-zero 由真实列正确性证)。
- **off 路 byte-identical**:canonical 套件 **172:170 PASS + 2 SKIP exit 0**(poison-off 不变)。

## R7 早退(Stage1 已发生)+ R9 解决
- Stage1 BOOL_SWITCH_2 已编译全 pad leg → 本阶段构建仅 harness 重编(`runs/build-M7c-stage2.log` 0 err)。
- **R9(hdim=100,100%8=4,align-1):跑通**(fp16 softmax + poison 全 PASS,无对齐 assert,无 NaN)→
  **hdim=100 不是 documented reject**,bool-pad align-1 正常处理奇余数。

## poison-pad sweep:168/168 PASS,0 FAIL,全 store-skip=PASS(`runs/run-M7c-stage2-sweep.log`)
12 pair × {bf16,fp16} × 7 配置(silu-c1 / sm-c1 / **silu-c0-target(P1-1)** / **sm-c0-target(P1-1)** /
silu-c1-combo(5因子,seq200) / sm-jagged-c1 / **det-sm-c1(determ lock)**),全 `-poison_pad=1`:
- pair:asymmetric-canonical `64/128,128/64,96/256,128/256`;非典范-symmetric `80/80,48/48,192/192,100/100`;
  asym+非典范 `80/128,100/64,48/96,192/256`;**含两方向 64/128 与 128/64**(证 QK/V flag 不交叉接线)。
- **load-zero 证**:全部真实列梯度有限且对拍 PASS(NaN 输入未泄漏)。容差未松(bf16 2e-2/5e-2、
  fp16 5e-3/1e-2);hd256 桶 pair 误差最大 dQ≤0.016 vs |ref|~6.7(< atol 5e-2)。
- **store-skip 证**:全部 store-skip=PASS(dK/dV pad 尾保持 NaN;asymmetric pair pad 区非空,16–192 列,
  故检查非平凡)。
- **determ lock**:`*/256` pair 的 det-sm-c1 PASS(hd256 kN0=64 split + poison 同时成立)。
- **P1-1 cross**:每 pair 的 c0-target(causal=0+num_target)PASS。

## 非平凡性(诚实)
poison 检查可判伪:asymmetric pair 的输入 pad 列确含 NaN、输出 pad 尾确预填 NaN(非空 pad 区);
若 mask 失效,GEMM0/2 读 NaN → 输出 NaN → check_err FAIL;若 epilogue 误写 → store-skip FAIL。
两者皆 PASS = 机制真证 OOB 归零 + 跳写,非 vacuous。

## 缺口/限制(如实)
- poison over-alloc 吸收 OOB **写**:故 GEMM4 的 dq_acc store-skip(非典范 exact-alloc 下防越界)
  本 harness 无法直证,靠真实列正确 + 代码审计兜底(已注明)。
- group 路 poison/非典范 = **Stage 3**(group dispatch 尚未 refactor,现仍 guard 拒 asymmetric)。

## 产物
`runs/build-M7c-stage2.log`(0 err)、`runs/run-M7c-stage2-sweep.log`(168/168)、
`runs/M7c-stage2-suite.log`(172/170/0/2)、`test/sweep_M7c.py`(新)、`example_hstu_attention_bwd.cpp`(poison harness)。

**Stage 2 证毕,停,等 lead 放行 Stage 3(group dispatch pad switch + group canonical byte-identity + group 非典范 poison)。**
