# M7c Stage 2 → Stage 3 放行(coder pane 0.1)

**lead 亲核通过 Stage 2**:独立复跑 poison 非典范案(双方向 64/128↔128/64 + 100/100 + 100/64)全 PASS + store-skip;canonical 套件 172/170/0/2 exit 0(poison-off 零回归);**dq_acc store-skip 我已代码核实**(`hstu_attention_bwd_kernel.hpp:373-381` `sequence<false,(kPadHeadDimQ>0)>` + mop set/atomic_add → production 无 OOB 写,你标的"poison 测不到"=测试盲区非 bug,结论:dQ 路 production 安全)。R9(hdim=100)/R7 已退役。

## 放行 Stage 3 — group dispatch
按 draft §3A(group)+ §7 Stage 3:
- `hstu_attention_group_backward_dispatch.hpp`:镜像 batched 的 pad switch —— 取模派生 pad_qk/pad_v、`BOOL_SWITCH_2` 包 RunSilu/RunSoftmax、删 `:139-140,208-209` 的 `constexpr 0`、放松 guard `:309` 为 `>MaxK`。
- **解 `ProblemFor` `:74` 的 `<0,0>` 写死**:在 pad switch 内建 Problem(镜像 no_group),嵌在已有 Local/NoLocal 切换下(pad 深一层)。
- harness group 路(`:708+`/`:771`)同 batched 做 poison-pad over-alloc(若 group 已共用 no_group 的 poison 基建则复用)。

## 验证(同 Stage 1/2 标准)
1. **group canonical byte-identity**:group 的 pad=0 设备符号 byte-identical 于基线(co_symbols verify;group entry .o 在基线 294 符号集里)。
2. **group 非典范 poison sweep**:g{2,3,4} × asymmetric/非典范 pair × {bf16,fp16} × {SiLU,softmax} × **每组合 P1-1 cross** + group determ,全 `-poison_pad=1` 硬证 OOB 归零/store-skip,容差禁松。
3. **canonical 套件不回归**(172 全绿,group canonical 在内)。

## 纪律
- guard 放松与 pad switch 同 commit(R1);容差禁松;带 FAIL 不 promote;诚实记限制。
- **Stage 3 完再停报我**(group byte-identity + group poison sweep + 套件三证据),lead 亲核放行 Stage 4(全矩阵收尾 + 把套件 2 个 SKIP 转 pass-asym + done.md)。
- 不 commit。
