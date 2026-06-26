# M0 脚手架 — 完成报告 (pane-1 / coder)

状态:**✅ 通过**。HSTU bwd 端到端编译通过 + gfx950 launch 不崩 + 对拍 harness 跑通,exit 0。
日期 2026-06-04。严格遵守 kernel-design-rocm skill(每步编译验证,证据进 `/root/workspace/hstu-bwd-impl/`)。

## 1. 新建/改动文件(全在 `ck_hstu/example/ck_tile/18_hstu_attention/`)

新建:
- `hstu_attention_bwd_params.hpp` — `HstuAttentionNoGroupBwdParams`(batch+jagged,字段按 DESIGN §4.6:复用 fwd 输入 + do_ptr/lse_ptr + dq/dk/dv + 各 stride + d_ptr/dq_acc + alpha/attn_scale + nhead_ratio_qk + kIsDeterministic);`HstuAttentionGroupBwdParams` 留空 struct(TODO M4)。
- `hstu_attention_bwd_kernel.hpp` — 3-kernel(PRE/MAIN/POST)结构文档 + TODO M1/M5/M6 标记(M0 不接 FMHA MAIN,只占位)。
- `hstu_attention_batched_backward_dispatch.hpp` — `run_batched_backward_dispatch<InOutDataType,kUseCausal,kUseSoftmax,kHasBias,kIsDeterministic,MaxK>`,镜像 fwd dispatch 模板签名 + `BUILD_HSTU_FOR_GFX95_ONLY` 占位分支;**M0 body = hipMemsetAsync 把 dQ/dK/dV 置 0**(jagged 路 throw,M3 再做)。
- `hstu_attention_no_group_backward_bf16.cpp` — bf16 入口,`BOOL_SWITCH_2(causal,softmax)` → dispatch;用 extern-template ref header(镜像 fwd)。
- `example_hstu_attention_bwd.cpp` — 对拍 harness:gen Q/K/V/dO → GPU fwd 产 O → GPU bwd 产 dQ/dK/dV → `reference_no_group_hstu_attention_bwd` 产 dQ*/dK*/dV* → 显式打印每张量 max/mean abs err + `ck_tile::check_err` 三张量 PASS/FAIL。M0 batched+bf16+SiLU only(jagged/group/softmax/fp16 留 TODO)。
- `instances/hstu_attention_batched_backward_bf16_*.cpp`(4 个)+ `..._instances_ref.hpp` — generate_instances.py 生成。

改动:
- `hstu_attention_api.hpp` — 加 `extern void hstu_attention_no_group_backward_bf16(...)` 声明 + include bwd_params(未改 fwd 行为)。
- `generate_instances.py` — 加 `create_backward_instances` + `create_backward_instances_ref`(M0 轴:batched×bf16×{causal,softmax 各 2}×bias=false×determ=false×maxk=64 = 4 instance)。
- `CMakeLists.txt` — 加 `tile_example_hstu_attention_bwd` target(EXCLUDE_FROM_ALL;GLOB `*backward*.cpp` + harness 需要的 fwd bf16 entry/instances;gfx95 加 `-fno-slp-vectorize -DBUILD_HSTU_FOR_GFX95_ONLY -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3`)。

**未碰 fwd 行为**;mask 这阶段未碰(M0 用 causal=0 / no-mask)。

## 2. 编译 — 0 error

```
cmake -B build -DBUILD_DEV=OFF -DGPU_TARGETS=gfx950
cmake --build build --target tile_example_hstu_attention_bwd -j$(nproc)
```
结果:**0 error**,`bin/tile_example_hstu_attention_bwd` 链接成功。
证据:`runs/build-bwd-M0.log`(`grep -c error: = 0`)、`runs/cmake-bwd-M0.log`。

修过的 2 个编译/链接问题(均已解决):
1. harness dump lambda 用了 `decltype(t)::value_type`(HostTensor 无此成员)→ 改用显式 `InOutDataType`。
2. 链接缺 `hstu_attention_no_group_forward_bf16`(harness 要先跑 GPU fwd 产 O)→ CMake bwd target 追加 fwd bf16 entry + `instances/*forward_bf16*.cpp`。

## 3. 运行 — exit 0 + 三梯度 err

命令(M0 验收 case):
```
./build/bin/tile_example_hstu_attention_bwd -prec=bf16 -b=2 -nhead=2 \
   -hdim_qk=64 -hdim_v=64 -seqlens=128 -softmax=0 -causal=0 -v=1
```
输出(`runs/run-bwd-M0.log`,exit 0):
```
  dQ: max_abs_err=0.0400391 mean_abs_err=0.00716305 (max|ref|=0.0400391)
  dK: max_abs_err=0.0407715 mean_abs_err=0.0071176  (max|ref|=0.0407715)
  dV: max_abs_err=0.0361328 mean_abs_err=0.0069616  (max|ref|=0.0361328)
[PASS] dQ   [PASS] dK   [PASS] dV
```
关键观察:**每张量 max_abs_err 恰等于 max|ref|** → 证明 GPU bwd 输出全 0、reference 产真实梯度、对拍管线正确连通。默认 case 因 scale_p=1/max_seqlen_q≈1/134 使 ref 梯度极小(~0.04),恰落在 bf16 atol=5e-2 内,故 check_err 报 PASS(并非数值正确,只是 ref 量级小于阈值)。

为消除"PASS 是巧合"的疑虑,补跑 `-attn_scale=1.0`(放大 scale_p → ref 梯度量级 ~5):
```
  dQ: max_abs_err=5.125    (max|ref|=5.125)
  dK: max_abs_err=5.21875  (max|ref|=5.21875)
  dV: max_abs_err=4.625    (max|ref|=4.625)
[FAIL] dQ   [FAIL] dK   [FAIL] dV   (96% wrong values)
```
→ 与 DESIGN §6 M0 预期(全 0 输出 → 对拍 FAIL/大误差)完全一致;exit 仍 0。证据:`runs/run-bwd-M0-attnscale.log`。

## 4. M0 验收对照(DESIGN §6 / pane-M0 §具体交付)
| 验收项 | 结果 |
|---|---|
| 编译 0 error | ✅ |
| 程序 exit 0 | ✅(两 case 均 exit 0)|
| 打印三梯度 err | ✅(显式 max/mean abs err + check_err)|
| dispatch→launch→harness 通 | ✅(GPU fwd→O→GPU bwd→ref→check_err)|
| bwd 输出全 0(M0 stub)| ✅(err==\|ref\| 佐证)|
| params/3-kernel 壳/dispatch/CMake target/instances 分支/CLI | ✅ 全部交付 |

## 5. candidates.jsonl
已追加 `M0-scaffold`(parent `C0-baseline`,status `pass`,evidence 指向三个 log)。

## 6. 遗留 / 给 M1 的接力点(均已在代码 TODO 标注)
- **M1 接 MAIN**:`hstu_attention_batched_backward_dispatch.hpp` 的 memset 换成 `[PRE if softmax]→MAIN(kr_ktr_vr)→POST(convert)`;harness `main()` 把 exit code 改成 `numeric_pass?0:-2`;harness 分配 float `dq_acc` workspace(M0 置 nullptr)。
- **M1 关键验证**(闸门):FMHA policy 复用(R1)、留 g 的 VGPR 按 **CDNA4 加法占用模型**验 ScratchSize=0 不掉 wave(R2)、平凡 `GetTileRangeAlongY→(0,seqlen_q)`(P1-D)、保留 `BiasEnum=NO_BIAS`+`BiasDataType` dummy(P1-A)。
- **M3** jagged(dispatch 现 throw)、**M4** group(GroupBwdParams 空 struct)、**M5** softmax(PRE+LSE,harness 现 SiLU only)、**M7** fp16(harness/entry 现 bf16 only)。
- 无未解决阻塞点。
