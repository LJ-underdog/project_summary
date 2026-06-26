# 派给 pane-1(角色:coder)— HSTU bwd 实现 M0 脚手架

调度模式:tmux pane-1。这是真实代码任务。先读设计与现状,再动手。不要派 sub-teammate。**严格遵守 kernel-design-rocm skill:每次有意义改动后编译验证,证据记 `/root/workspace/hstu-bwd-impl/`。**

## 背景与依据
- 设计(权威):`/tmp/hstu-bwd-design/DESIGN.md`(gfx950 优先;§1 架构、§4 工程落地、§4.6 字段表、§6 M0 验收)。
- 实现 workspace:`/root/workspace/hstu-bwd-impl/`(docs/task-contract.md、docs/draft.md、candidates.jsonl、benchmark.csv、runs/、src/)。
- 代码目标目录:`/root/workspace/ck_hstu/example/ck_tile/18_hstu_attention/`(在此新增 bwd 文件,**镜像 fwd 命名**)。
- 复用:`/root/ck/include/ck_tile/ops/fmha/`(PRE `block_fmha_bwd_dot_do_o.hpp` / MAIN `block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp` / POST `block_fmha_bwd_convert_dq.hpp` / `block_fmha_bwd_pipeline_default_policy.hpp` / `kernel/fmha_bwd_kernel.hpp` / `tile_fmha_shape.hpp`)。
- baseline 已通:`cd /root/workspace/ck_hstu && cmake --build build --target tile_example_hstu_attention -j`(已配 `-DBUILD_DEV=OFF -DGPU_TARGETS=gfx950`)。
- 参考 fwd 文件作模板:`example_hstu_attention_fwd.cpp`(对拍 harness 写法、CLI、check_err)、`hstu_attention_params.hpp`、`hstu_attention_{batched}_forward_dispatch.hpp`、`hstu_attention_api.hpp`、`CMakeLists.txt`。
- oracle:`reference_hstu_attention_bwd.hpp`(`reference_no_group_hstu_attention_bwd<...>::Run(...)`,签名见 DESIGN §5.1)。

## M0 目标(只做脚手架,**不写 MAIN 真实计算**——那是 M1)
让"HSTU bwd"端到端**编译通过 + 在 gfx950 launch 不崩 + 对拍 harness 跑通**(此阶段 bwd 输出可为全 0,故对拍预期 FAIL/大误差,**这是正常的**,M0 验收看的是"管线通"不是"数值对")。

### 具体交付
1. **bwd params**:在 `hstu_attention_params.hpp`(或新建 `hstu_attention_bwd_params.hpp`)加 `HstuAttentionNoGroupBwdParams`(batch+jagged),字段按 DESIGN §4.6(复用 fwd 输入 + do_ptr/dq/dk/dv + stride + lse_ptr/d_ptr/dq_acc_ptr + alpha/attn_scale + kIsDeterministic)。group 版可先留空 struct(M4 再填)。
2. **kernel 包装**:M0 先做**最小可 launch 的 stub** —— 一个把 dQ/dK/dV 置 0 的简单 kernel(或直接 hipMemset 输出),**不接 FMHA MAIN**(M1 才接)。目的是打通 dispatch→launch→harness。文件 `hstu_attention_bwd_kernel.hpp` 留好将来接 3-kernel 的结构(注释标 TODO M1/M5/M6)。
3. **dispatch**:`hstu_attention_batched_backward_dispatch.hpp` + `hstu_attention_no_group_backward_bf16.cpp` 实例(M0 只需 bf16+batched+SiLU 一条),镜像 fwd dispatch 结构(`BUILD_HSTU_FOR_GFX95_ONLY` 分支占位)。
4. **API 接缝**:`hstu_attention_api.hpp` 加 bwd 入口声明 + 一个 `hstu_attention_no_group_backward_bf16(params, stream)`。
5. **对拍 harness**:`example_hstu_attention_bwd.cpp` —— 镜像 fwd:解析 CLI(复用 fwd 参数 + `-v`)、gen Q/K/V/dO、**GPU 跑 fwd 得 O(+LSE)** 再 **GPU 跑 bwd 得 dQ/dK/dV**、调 `reference_no_group_hstu_attention_bwd` 得 dQ*/dK*/dV*、`ck_tile::check_err` 分别比对三者并打印(M0 预期不过,打印 err 即可)。
6. **CMake**:加 `tile_example_hstu_attention_bwd` target(EXCLUDE_FROM_ALL,GLOB bwd instances,gfx95 加 `-fno-slp-vectorize -DBUILD_HSTU_FOR_GFX95_ONLY`)。
7. **generate_instances.py**:加 `create_backward_instances`(M0 先只生成 bf16 batched SiLU no-causal 一个,够编译即可)。

### 编译/验证(每步做,记证据)
- 构建:`cd /root/workspace/ck_hstu && cmake -B build -DBUILD_DEV=OFF -DGPU_TARGETS=gfx950 >/dev/null && cmake --build build --target tile_example_hstu_attention_bwd -j$(nproc) 2>&1 | tee /root/workspace/hstu-bwd-impl/runs/build-bwd-M0.log`
- 跑:`./build/bin/tile_example_hstu_attention_bwd -prec=bf16 -b=2 -nhead=2 -hdim_qk=64 -hdim_v=64 -seqlens=128 -softmax=0 -causal=0 -v=1 2>&1 | tee runs/run-bwd-M0.log`
- 验收:**编译 0 error + 程序 exit 0 + 打印出三个梯度的 err(数值大没关系)**。
- 把 M0 候选写进 `candidates.jsonl`(id `M0-scaffold`,status pass/fail + evidence 指向 log)。

## 铁则
- 不改 fwd 行为;mask 这阶段不碰(M0 用 causal=0 / no-mask)。
- 遇到编译错就修到通(这是 M0 的主要工作);卡住超过合理尝试就在报告里如实写阻塞点 + 已试方案,别假装通过。
- 完成写 `/tmp/hstu-bwd-design/M0-done.md`:建了哪些文件、build 是否 0 error、run 是否 exit 0 + 三梯度 err 数值、candidates.jsonl 是否更新、遇到并解决/未解决的问题。
- progress 简洁;长输出进 log 文件不刷屏。
