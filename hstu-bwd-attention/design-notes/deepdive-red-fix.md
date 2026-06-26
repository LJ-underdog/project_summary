# 修 2 个深读讲义的 MFMA 形状 RED(coder pane 0.1)

审计发现 2 个老深读讲义把 **fp32 的 16×16×16 MFMA 形状误标成 f16/bf16**。权威来源:`/root/workspace/ck_hstu/include/ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp`(16×16×16 只在 fp32 `float,float,float`;f16 `half_t,half_t,float` 是 32×32×8/32×32×16/16×16×32)+ FMHA codegen `example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`。先**自己核实**这两处真值再改。

## RED-4:`hstu-b1052-report/bwd-5-gemms-deepdive-20260602.html` §5.1
- 错:"CDNA3(gfx942):32×32×8  **16×16×16** ← 原生 f16 MFMA";且 §5.2/§5.3 图复用了这个错的 f16 WarpTile。
- 改:16×16×16 是 **fp32** 形状,非 f16。f16 原生形状 = **32×32×8 / 32×32×16 / 16×16×32**(gfx950 bwd 实际用 16×16×32,见 `hstu_attention_bwd_shape.hpp` WarpTile0)。把 §5.1 文字 + §5.2/§5.3 图里被误用的 f16 WarpTile 标注一并改对(用 16×16×32 作 gfx950 f16 示例)。**核 SVG 渲染不溢出。**

## RED-5:`hstu-b1052-report/fmha-bwd-intro-20260528.html` §3.6 perf 表
- 错:bf16/fp16 "default" 标题下列的是 **fp32 presets**(hdim64 kM0=16/kN0=64/kK0=64/kK4=16;MFMA 16×16×16)。
- 改(`fmha_bwd.py:432-437` fp16/bf16 `get_dq_dk_dv_tiles`):hdim64 = **bm0=32, bn0=128, bk0=64, bk4=32**;hdim128 = **bm0=16, bn0=128, bk0=128, bk4=32**;MFMA warp tile = **16×16×32**(非 16×16×16)。hdim32 行两 dtype 都对、不动。**先核 fmha_bwd.py 实值再填**(别照抄我这里、以源为准)。

## 纪律
- 只改这 2 文件、只改 MFMA/preset 错处;数字以**源码/fmha_bwd.py 实值**为准(自核);其余内容/链接/无 dingbat 保持。改完核 SVG/表渲染 + 一句话报改了啥。不 commit(这些是 report,本就不进 git)。
