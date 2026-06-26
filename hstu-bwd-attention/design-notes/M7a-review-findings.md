# M7a fp16 WIP —— 对抗式 review findings (reviewer / pane 0.2)

**结论(先说): 全 7 条 GREEN,无阻塞 RED。** M7a fp16(纯 dtype 加宽,hd64 复用 bf16 模板路)
在**独立干净重建**(`build_review`,从 HEAD=`d4fb2884` 工作树)下:
- 自建独立套件运行 **TOTAL 106 / PASS 106 / FAIL 0 / SKIP 0 exit 0**(`/tmp/hstu-bwd-design/test-20260611-041316.log`)
  = 91 bf16(零回归)+ 15 新 fp16 案 + 2 fp16 determ byte-identical repro。
- 容差 revert 实验做完并已 revert,代码树恢复 pristine(仅 4 文件 diff + 新 fp16 文件)。

**M7a 可在对拍闭合后 promote —— 而对拍此刻已闭合(106/106)。** 仅一条非阻塞建议(§7 极端 stress)。

---

## 逐条审查(GREEN/RED + 证据)

### 1. 对拍公平性(最关键)—— **GREEN**
harness 全程模板化在 `InOutDataType` 上;`-prec=fp16` → `InOutDataType=fp16_t`,GPU 与 reference 同走 fp16 I/O、同 float 内算:
- host tensor q/k/v/o/do/dq/dk/dv 全为 `InOutDataType`(`example_hstu_attention_bwd.cpp:240-265, 706-729`)。
- reference `reference_{no_group,group}_hstu_attention_bwd<InOutDataType, GemmAccDataType,...>`(`:511, 952`),I/O 全 `InOutDataType`(`reference_hstu_attention_bwd.hpp:66-74`),内算 `GemmAccDataType`。
- `HstuAttentionFwdTypeConfig<fp16_t>` 与 `<bf16_t>` **GemmAccDataType=float、CompDataType=float 完全对称**(`hstu_attention_fwd_type_config.hpp:16-17, 26-27`)。
- reference 甚至把中间 P 显式 round 回 `InOutDataType`(`reference_hstu_attention_bwd.hpp:319`)、q/k 读为 `InOutDataType` 再转 float 做点积(`:253-267`)—— 即 reference **忠实建模 fp16 I/O 边界**,与 GPU kernel 一致。
- 比对 `check_err(dq_host<fp16>, dq_host_ref<fp16>, rtol, atol)`(`:567-571, 1004-1008`),两侧同 dtype。
**无 silent-unfair:不存在一侧 fp16 一侧 bf16。**

### 2. 容差是否放水 —— **GREEN(revert 实验已做)**
`get_bwd_elimit<fp16_t>` = rtol **5e-3** / atol **1e-2**(`:151-153`),比 bf16 的 2e-2/5e-2 **更紧**(尾数 fp16 10bit > bf16 7bit,物理合理收紧,方向正确,不是放水)。
实测错误量级 **远低于** 容差,绝非贴边过线:
- 典型 fp16 softmax causal: dQ/dK max_abs_err=**6.10352e-05**(=2⁻¹⁴)、dV=1.22e-4(=2⁻¹³),vs atol 1e-2 → **~160× headroom**。

**Revert 实验(均干净 relink,已全部 revert)**:
- **实验 A**(放宽到 bf16 容差 2e-2/5e-2):仍 PASS → 放宽不改变判定 = **无被掩盖的隐藏误差**。
- **实验 B**(收紧到 1e-3/5e-4,比所选紧 5–20×):**仍 PASS** → 误差是真小,所选 5e-3/1e-2 留巨大余量。
- 实验后 `cp` 回 `.orig`,grep 确认恢复 5e-3/1e-2,`git diff` 仅 4 文件、无容差改动。

### 3. fp16 路真在跑 fp16(非静默落回 bf16)—— **GREEN**
- `main()` 真按 `-prec` 实例化 `run_*<fp16_t>`(`:1051-1062`),非 hardcode bf16。
- 三个 harness fwd/bwd 调用点均有真实 `else if(fp16) → *_fp16(...)` 分支(`:387-390, 485-488, 849-852, 933-936`)。
- `hstu_attention_no_group_backward_fp16.cpp` 与 bf16 版**逐字镜像**(仅 `bf16_t→fp16_t`),`BOOL_SWITCH_3(causal, softmax, determ)` fan-out 与 bf16 **完全一致**。
- group entry `hstu_attention_group_backward_fp16.cpp:18` 用 **BOOL_SWITCH_3**(含 kIsDeterministic 真轴),**无 O1 式 hardcode-false** 隐患。
- **铁证(经验证)**:fp16 误差为 fp16-ULP 量级(2⁻¹⁴/2⁻¹³)。若静默落回 bf16,误差应大 ~8×(bf16-ULP 量级)。实测正是 fp16-scale → **确在跑 fp16**。

### 4. instance 完整性 —— **GREEN**
- 8 个 fp16 no_group bwd instance = causal{T,F} × softmax{T,F} × determ{atomic,determ} = **4 atomic + 4 determ**(`ls instances/*backward_fp16*.cpp` = 8;bf16 同为 8)。
- `instances/hstu_attention_batched_backward_fp16_instances_ref.hpp` 含 **8 个 extern template**,与 8 个 instance 定义**一一对应**(检查全 8 行 axis tuple 匹配)。
- instance 内容确为 fp16 实例化(`template void run_batched_backward_dispatch<ck_tile::fp16_t, ...>`);与 bf16 counterpart diff 仅 `half.hpp` vs `bfloat16.hpp` + dtype token —— 正确。
- `generate_instances.py` `BWD_DTYPES_M0 = ["fp16","bf16"]`,与 fwd 生成器 dtype 轴一致。
- **group 确为 direct-instantiation 无 instance 文件**(8 个 fp16 bwd instance 全是 `batched`;group entry 直接调 `run_group_backward_dispatch`)—— 与 bf16 group 一致。

### 5. CMake —— **GREEN**
`CMakeLists.txt` diff:新增 `FWD_FP16_INSTANCE_SRCS` glob(144 个 fp16 fwd instance)+ `*_forward_fp16.cpp` 两 entry + `*_backward_fp16.cpp` 两 entry,全部加入 bwd target 的 `target_sources`。
- 纯加性,bf16 链接路不变。
- **实证**:`build_review` 全新 configure+build **exit 0**(`m7a-review-build.log`,18 分钟,链接成功);harness 跑 GPU fwd 产 O/LSE → fp16 对拍全 PASS,证明 fp16 fwd 已正确链入。

### 6. bf16 零回归(byte-level)—— **GREEN**
- `git diff --name-only HEAD` **仅 4 文件**:CMakeLists.txt / example_hstu_attention_bwd.cpp / generate_instances.py / hstu_attention_api.hpp。
- 库/kernel/pipeline/dispatch/reference(bwd_kernel / no_softmax_pipeline / with_softmax_pipeline / batched_dispatch / group_dispatch / bwd_params / reference / fwd_type_config / bool_switch)**全部 byte-identical 于 HEAD**(`git diff --name-only HEAD -- <9 文件>` 为空)。
- 4 改文件对 bf16 **纯加性**:CMake 加 glob/源;example main() bf16 走原 `else` 分支逻辑等价;api.hpp 加 extern;generate_instances 加 dtype 轴(bf16 instance 文件未变 = git status 未列为 modified)。
- **实证**:91 个 bf16 案在 `build_review` 全 PASS;bf16 SiLU/group-softmax spot-check `numeric_pass=true`。

### 7. fp16 溢出风险 —— **GREEN(+ 非阻塞建议)**
论证:输入 N(0,1)/U(-1,1),alpha=0.125,hd64 → S~O(8)·0.125 小;softmax exp/LSE/D 全 float 内算(CompDataType=float),仅 I/O 边界 round fp16;梯度量级实测 max|ref|≤~10,远低 fp16 max 65504。两侧同 fp16,即便溢出也对称。
**实测无 inf/nan**:
- 大 seqlen=1024 softmax:PASS,误差仍 6e-5。
- contextual=8 + targets=16 + window=32 + causal,seqlen=256:PASS。
- determ 多 split(seq512)、jagged+per-batch target:PASS,最大误差 dQ 3.9e-3(=2⁻⁸,rel~3.8e-4,仍 < rtol 5e-3,determ 跨多 split 累加更多 round 仍合格)。
**建议(非阻塞)**:套件可选加一个极端 stress(超大 seqlen + 大 attn_scale 或人为大 dO)进一步压 fp16 上界,但不阻塞 promote。

---

## 旁证 / 已被 coder 闭合的点
- 旧 `reject-fp16` 案(原期望 harness 拒绝 `-prec=fp16`)已被 coder 正确**升级为 pass-fp16-* 块**(`run_bwd_tests.py:308` 注释 + `:389-428`),否则会变 mismatch。覆盖矩阵良好:SiLU/softmax × batched/jagged/group × causal{0,1} × determ × **P1-1 cross(causal=0+target)** × 非 tile 整除 seqlen × group 异构;另含 2 个 fp16 determ byte-identical repro(`:454-458`)。**P1-1 那一格已覆盖,没重蹈 M6b 覆盖洞。**

## 复现要点(供 lead/coder)
- 独立 build:`cmake -B build_review ... -DBUILD_DEV=OFF` + `--target tile_example_hstu_attention_bwd`(干净,与 coder 的 `build/` 不冲突)。
- 独立套件:`python3 run_bwd_tests.py --bin <build_review>/bin/tile_example_hstu_attention_bwd` → 106/106 exit 0。
- 容差实验日志:`m7a-relink-A.log`(放宽)/`m7a-relink-B.log`(收紧),原文件备份 `example_bwd.cpp.orig`;现已 revert。

**最终:M7a 三方/对拍闭合达成,可 promote。无 silent-wrong,无放水,无 bf16 回归。**
