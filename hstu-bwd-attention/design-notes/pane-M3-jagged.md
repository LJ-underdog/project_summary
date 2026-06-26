# 派给 pane-1(角色:coder)— HSTU bwd 实现 M3(jagged 变长模式)

调度模式:tmux pane-1。接续已完成的 M0–M2(代码在磁盘,稳定)。严守 kernel-design-rocm skill:每步编译+对拍+**跑回归测试套件**,证据进 `/root/workspace/hstu-bwd-impl/`。不要派 sub-teammate。

## 先读(当前真实代码 + 背景)
- 进度:`/tmp/hstu-bwd-design/M1-done.md`、`M2-done.md`(M2 遗留明确:jagged 现仅 batched;harness 无 `-jagged` 开关;dispatch jagged 路径需接)。
- 设计:`DESIGN.md` §3.4(三模式索引:jagged = dim0=1 + cu_seqlens,base offset = `seqstart[b]*seq_stride`,seqlen = `seqstart[b+1]-seqstart[b]`)、§4.6 字段表(`seq_q_offsets_ptr/seq_kv_offsets_ptr`、`is_jagged`)。
- 代码目录 `/root/workspace/ck_hstu/example/ck_tile/18_hstu_attention/`,逐个 Read:
  - `hstu_attention_batched_backward_dispatch.hpp`(现 `if(param.is_jagged) throw`)
  - `hstu_attention_bwd_kernel.hpp`(batched 索引;需加 jagged base offset)
  - `hstu_attention_bwd_params.hpp`(`HstuAttentionNoGroupBwdParams` 已含 jagged 字段?核对 seq_*_offsets_ptr/is_jagged)
  - `example_hstu_attention_bwd.cpp`(harness;需加 `-jagged` 开关 + 生成 jagged 输入 + 喂 offsets 给 GPU 与 CPU reference)
- **模板参照**:fwd 的 jagged 路径 `hstu_attention_jagged_forward_dispatch.hpp`(怎么处理 cu_seqlens / dim0=1 / offset),fwd harness 里 jagged 输入怎么 gen(`example_hstu_attention_fwd.cpp` 的 `-jagged` 分支)。
- **oracle**:`reference_hstu_attention_bwd.hpp` 的 `reference_no_group_hstu_attention_bwd<...,kIsJagged,...>`(kIsJagged=true 时 dim0=1、用 seq_q_offsets/seq_kv_offsets;签名见 DESIGN §5.1)。

## M3 目标
让 **SiLU + bf16 + hd64 + (no-mask 与 mask 均可) 的 jagged 路径**端到端对拍 PASS;`reject-jagged` 升级为 pass。

## 要做的
1. **harness `-jagged` 开关**:镜像 fwd —— jagged 时 Q/K/V/dO/O 按 `[1, ΣL, H, D]` packed 分配,构造 `seq_q_offsets`(前缀和,size num_batch+1),per-batch seqlen 可不同;dO 同布局;`dq/dk/dv/dq_acc` 同 packed 布局。把 offsets 同时喂 **GPU dispatch** 与 **CPU `reference_...<kIsJagged=true>`**(reference 的 jagged 分支需 dim0=1 + offsets)。
2. **dispatch**:去掉 `if(param.is_jagged) throw`;`is_jagged` 分支走 jagged 索引(传 seq_*_offsets_ptr;grid.z 仍 = num_batch)。
3. **kernel**:per-(batch,head) base offset 与 seqlen 由 `seq_q_offsets_ptr[i_batch]`/`[i_batch+1]` 求(dim0=1,token-major);batched 路保持不变(`if constexpr`/运行时分支)。对齐 FMHA group 分支的 `seqstart_*_ptr` 用法。注意 hdim_qk(dQ/dK)vs hdim_v(dV)。
4. **early-exit / 非整除**:jagged 各段长度差异大 + 非 tile 整除,确保 OOB 归零(M1 已验 batched 的 OOB,jagged 要再验)。

## 验收(全过才算 M3 通过)
- 编译 0 error(log → runs/build-bwd-M3.log)。
- 对拍 PASS(attn_scale=1.0,bf16):
  - jagged + no-mask;jagged + causal;jagged + 几个 mask 因子。
  - 多组 per-batch 不同 seqlen(含非整除、单 batch、段长差异大)。
- **更新测试套件** `test/run_bwd_tests.py`:`reject-jagged` → pass(加 `-jagged` 相关 case + jagged×mask 组合);跑 `python3 test/run_bwd_tests.py` **整体 exit 0**。
- candidates.jsonl 加 `M3-jagged`(pass + 覆盖的 case);保留其它 reject(group M4 / softmax M5 / fp16+hdim M7)仍正确拒绝。

## 铁则
- 不改 fwd 行为 / 不放宽容差。jagged offset 索引错会让对拍大误差暴露——别靠巧合。
- batched 路径不能回归(测试套件里 batched/mask case 必须仍 PASS)。
- 卡住超合理尝试,如实写阻塞点 + 已试 + 怀疑方向(尤其 offset 计算、reference jagged 分支参数、packed 布局 stride)。
- 完成写 `/tmp/hstu-bwd-design/M3-done.md`:harness/dispatch/kernel 改动、对拍结果(jagged 各档)、测试套件更新后整体 exit、遗留。
- progress 简洁;长 log 进文件。
