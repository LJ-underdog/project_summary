# 派给 pane-1(角色:coder)— HSTU bwd 实现 M4(group 模式)

调度模式:tmux pane-1。接续 M3(jagged,你刚做的,代码在手)。严守 kernel-design-rocm skill:每步编译+对拍+跑回归测试套件,证据进 `/root/workspace/hstu-bwd-impl/`。不要派 sub-teammate。

## 关键认知
**group = jagged 的超集**:同样 dim0=1 token-major packed + cu_seqlens(M3 的 offset 索引**直接复用**),**额外**是每个 group 有独立超参(window/contextual/min_full/max_seqlen/attn_scale),按 `i_group = i_batch / num_batch_per_group` 取。`alpha` 仍是**全局单标量**(DESIGN §4.7 D6);**`scale_p` 与 mask 4 参变成 per-group(device 指针)**。

## 先读
- 设计:`DESIGN.md` §3.4(group 行)、§4.7 D6(alpha 全局 / scale_p+mask per-group / i_group 索引)、§4.6(GroupBwdParams 字段)。
- M3:`/tmp/hstu-bwd-design/M3-done.md`(jagged offset 索引,M4 复用)。
- oracle:`reference_hstu_attention_bwd.hpp:501` `reference_group_hstu_attention_bwd::Run(is_cross, q,k,v,lse,o,do, dq,dk,dv, num_batch, num_batch_per_group, alpha, seq_q_offsets, seq_kv_offsets, num_targets, group_max_seqlens_q, group_contextual_seqlens, group_window_sizes, group_min_full_attn_seqlens, group_attn_scales)`(全 vector)。
- 模板参照:`hstu_attention_group_forward_dispatch.hpp`(per-group 取数:`param.group_*_ptr` + `num_batch/num_group` + `num_batch_per_group`),fwd `HstuAttentionGroupFwdParams`(group_*_ptr 字段),fwd harness 的 `-g -g_max_seqlens -g_local_lens -g_context_lens -g_minfull_lens -g_attn_scales` 处理。
- 现状:`HstuAttentionGroupBwdParams` 是空 struct(`hstu_attention_bwd_params.hpp:111`);dispatch/harness/entry 现仅 no_group;测试套件 `reject-group-g2`。

## M4 目标
**SiLU + bf16 + hd64 的 group 路径**端到端对拍 PASS;`reject-group-g2` 升级为 pass。

## 要做的
1. **`HstuAttentionGroupBwdParams`**(填充):复用 jagged 全部输入/输出/workspace 字段(q/k/v/o/do/dq/dk/dv ptr + stride + seq_q/kv_offsets_ptr)+ **per-group device 指针**:`group_window_size_ptr / group_contextual_seqlen_ptr / group_min_full_attn_seqlen_ptr / group_max_seqlen_q_ptr / group_attn_scale_ptr` + `num_group`、`num_batch_per_group` + 全局 `alpha`、`num_targets_ptr`、`kIsDeterministic`。命名对齐 fwd `HstuAttentionGroupFwdParams`。
2. **group dispatch**(`hstu_attention_group_backward_dispatch.hpp`,镜像 fwd group dispatch + M3 jagged 索引):kernel 内 `i_group = i_batch / num_batch_per_group` 取 per-group 超参;jagged offset 索引复用 M3。
3. **kernel per-group 取数**:把现在的标量 `scale_p` 与 mask 4 参(window/contextual/min_full/max_seqlen)改为"group 模式下按 `i_group` 从 device 指针读"(no_group 路保持标量,用 `if constexpr`/模板轴 `kUseGroup` 区分,不回归)。`scale_p` per-group:`group_attn_scale[i_group] ? group_attn_scale[i_group] : 1/group_max_seqlen_q[i_group]`。`alpha` 仍全局。mask 构造用 per-group window/contextual/min_full + per-batch num_target。
4. **entry + harness**:加 group 入口(`hstu_attention_group_backward_bf16`?或在现 entry 加 group 分支);harness 加 `-g`(num_group)+ `-g_*` per-group 超参列表,packed 输入(group 也是 dim0=1 packed),offsets + per-group 数组上传 device,喂 GPU group dispatch 与 CPU `reference_group_...<kIsJagged=true>`(group 恒 jagged-packed)。
5. **测试套件**:`reject-group-g2` → pass(+ group×{mask 因子, 多 group, per-group 不同超参}组合);跑 `python3 test/run_bwd_tests.py` **整体 exit 0**;no_group(batched+jagged)不回归。

## 验收(全过)
- 编译 0 error(log→runs/build-bwd-M4.log)。
- 对拍 PASS(attn_scale 由 per-group 决定,bf16 阈值):group + no-mask / causal / mask 因子;多 group(g=2,3,4)、per-group 不同 max_seqlen/window/contextual/attn_scale、per-batch num_target。
- 测试套件整体 exit 0;`reject-group-g2`→pass;其余 reject(softmax M5 / fp16+hdim M7)仍正确拒绝。
- candidates.jsonl 加 `M4-group`(pass + 覆盖 case)。
- batched/jagged/mask 不回归。

## 铁则
- 不改 fwd 行为 / 不放宽容差。per-group 取数错(i_group 索引、device 指针、默认 scale_p)会让对拍大误差暴露——别靠巧合;**至少跑一档 per-group 超参各不相同**确保真按 group 取。
- no_group 路径零回归(测试套件全部 batched/jagged case 仍 PASS)。
- 卡住超合理尝试,如实写阻塞点 + 已试 + 怀疑方向。
- 完成写 `/tmp/hstu-bwd-design/M4-done.md`:params/dispatch/kernel/entry/harness 改动、对拍结果(group 各档,含 per-group 异构超参)、测试套件整体 exit、遗留。
- progress 简洁;长 log 进文件。
