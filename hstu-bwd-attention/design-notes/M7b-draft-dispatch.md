# M7b hdim{96,128,256} —— draft 闸门阶段(coder pane 0.1)

你刚完成 M7a fp16(已 promoted `bf82a1d2`,四方闭合)。下一里程碑 **M7b:支持 hdim{96,128,256}(symmetric,hdim_qk==hdim_v)**,解禁现有 hd64-only throw。**hdim_qk≠hdim_v 不在本单(留 M7c)。**

**本阶段只出 draft 设计稿,不写实现码。** 写完停下等 lead 审批闸门,批准后再开工。全程对拍铁律 `-attn_scale=1.0`。

## 0. 背景(只读)
- 基线现 HEAD=`bf82a1d2`(M7a),能力边界 = SiLU+softmax 全模式 × 全 mask × causal{0,1} × **bf16+fp16** × **hd64**;atomic+determ。套件 106/106。
- hd64 throw 位置(本单要解禁):
  - `hstu_attention_batched_backward_dispatch.hpp:390-391`(M5 softmax)、`:406-407`(M1/M2 SiLU)
  - `hstu_attention_group_backward_dispatch.hpp:321-322`(group)
- **MaxK 已是模板轴**(dispatch `…, MaxK>::Run(param,stream)` 已穿透);entry 现 hardcode `64`。
- **fwd 蓝本**:`hstu_attention_no_group_forward_bf16.cpp:17` 用 `HDIM_SWITCH(param.hdim_qk, param.hdim_v, MaxK, [&]{ run<…,MaxK>() })` 把运行时 hdim 映射到编译期 MaxK。fwd 已有 maxk{64,96,128,256} instance,镜像它。
- harness 已有 `-hdim_qk/-hdim_v` 参数(默认 64)。

## 1. draft 必须回答(写到 `/root/workspace/hstu-bwd-impl/docs/draft-M7b.md`)
1. **HDIM_SWITCH 复用**:fwd 的 `HDIM_SWITCH` 宏在哪(`hstu_attention_bool_switch.hpp`?)、覆盖哪些 (hdim_qk,hdim_v)→MaxK 映射?bwd 能否直接复用同一宏?symmetric 下 hdim∈{64,96,128,256} 各映射到哪个 MaxK tile?
2. **head-dim padding(最大风险点)**:现 dispatch 注释「hdim64: head-dim padding never needed」。非 64 hdim(尤其 96 非 2 的幂、非 tile 整除)在 5 个 GEMM(dQ=dS·K, dK=dSᵀ·Q, dV=Pᵀ·dO, dP=dO·Vᵀ, dS…)里**哪些维度需要 pad?pad value?** Q/K/V/O/dO load、dq_acc、convert/reduce POST 各处 hdim 出现点逐一列出。fwd 怎么处理 hdim padding 的(照抄)?这是 silent-wrong 高发区,draft 要逐 GEMM 列清。
3. **改面清单**:entry(no_group/group × bf16/fp16,HDIM_SWITCH 替 hardcode 64)、`generate_instances.py`(headdim 轴 [64,96,128,256],instance 数 = 现 8 × 4 hdim × 2 dtype?算清总量 + 编译时间预估,M7a 已暴露 fwd over-link 拖慢)、dispatch(解 3 处 throw + padding 逻辑)、kernel/pipeline(若 padding 需改——指明是否动 promoted pipeline,**动 promoted pipeline 要特别标红**,因 byte-identical 零回归是铁律)、CMake。
4. **是否动 promoted pipeline/kernel**:M7a 没碰库逻辑。M7b 若 padding 必须改 `no_softmax_bwd_pipeline`/`with_softmax_bwd_pipeline`/`bwd_kernel`,**会破坏 hd64 的 byte-identical 零回归保证** → draft 要论证:能否用编译期 `if constexpr(MaxK==64)` 走原路、非 64 才走 pad 路,使 hd64 指令流不变?给出零回归保全策略。
5. **顺带瘦身**:M7a 的 CMake `*forward_fp16*` glob 把 maxk 96/128/256 fwd 全拉进 bwd target(hd64 时是死重)。M7b 既然要用多 hdim,重新界定 fwd instance 链接范围(按需而非全 glob),记入 draft。
6. **测试矩阵设计**:每 hdim × {SiLU,softmax} × {bf16,fp16} × causal{0,1} × 代表性 mask × {batched,jagged,group} × atomic/determ。对拍容差是否随 hdim 变(更大 hdim → 更多累加 → 误差略增?)。**P1-1 cross(causal=0+target)必须每 hdim 覆盖**(别重蹈覆盖洞)。
7. **增量实现顺序建议**:如 hdim128(2 的幂、最常用)symmetric 先通 → 96/256 → 收尾。每步对拍。

## 2. 产出
- `docs/draft-M7b.md`(上述 7 点),pane 里一句话提示 draft 就绪。**不写实现码、不改库文件、不 build。** 等 lead 审 draft 后再派实现单。
- 若发现 M7b 比预想复杂(如 padding 必须重写 promoted pipeline、零回归无法保全),**在 draft 里如实标红风险**,供 lead 决定是否拆更细或调顺序。
