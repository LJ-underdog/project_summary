# cross-attention Stage B done — harness 解钉 + max_seqlen_kv + dispatch grid(待 lead 亲核)

基线 HEAD = `17515fcc`(M7c)。`-attn_scale=1.0`。**未 commit。** reference/pipeline 一行未改。

## 改面(`git diff --stat`,5 文件)
- `hstu_attention_bwd_params.hpp`(+6):两个 bwd params 结构各 **新增 1 字段 `max_seqlen_kv`**(NoGroup + Group,紧邻 max_seqlen_q;注明 self 路 == max_seqlen_q)。**纯 host 字段,kernel MakeKargs/device 不读 → device 码不变。**
- `hstu_attention_batched_backward_dispatch.hpp`:`launch_main_and_post` 的 `grid_seqlen_kv` jagged 分支 `max_seqlen_q`→`max_seqlen_kv`(num_splits/:79 GridSize 随之);batched 非 jagged 仍用 seqlen_kv 标量。(Stage A 的 BOOL_SWITCH 不变)
- `hstu_attention_group_backward_dispatch.hpp`:`launch_main_and_post` num_splits(:90)+ GridSize(:99)`max_seqlen_q`→`max_seqlen_kv`。(Stage A 的 BOOL_SWITCH 不变)
- `hstu_attention_bwd_kernel.hpp`:**Stage A 改动,本阶段未动**(diff 仍在因同一工作树)。
- `example_hstu_attention_bwd.cpp`(+156/-...):见下。

## harness 解钉(no_group + group 对称)
- **CLI**:新增 `-seqlens_kv`(同 -seqlens 格式;空=self 向后兼容;给出且≠seqlens=cross)、`-max_seqlen_kv`(uih kv 上限 override)、`-g_max_seqlens_kv`(group per-group)。
- **解钉**:`is_cross_attention` 由 `-seqlens_kv` 派生(不再钉 false)。
- **独立 kv**:独立 `seq_lengths_kv` / `max_seqlen_kv` / `seq_offsets_kv` 向量 + **独立 `seq_offsets_kv_dev` 设备 buffer**,喂 fp/bp.seq_kv_offsets_ptr(不再别名 seq_offsets_q_dev)。
- **target_in_kv==false**:cross KV 物理长 = kv_uih + contextual(**无 targets**);Q 物理长 = uih + targets + contextual(不变)。镜像 fwd harness(:344-345/:377-382)。
- **determ grid/workspace**:no_group `grid_seqlen_kv_h`(:365)+ group num_splits 改按 **max_seqlen_kv**(R4)。
- **K/V/dK/dV 分配**:本就按 phy_seqlen_kv,现 phy_seqlen_kv 真正独立(cross 时 != phy_seqlen_q)。
- **reference 调用**:第一参 is_cross_attention 运行时;喂独立 seq_offsets_kv + max_seqlen_kv。**reference 签名已就绪,零改。**
- self 路:max_seqlen_kv==max_seqlen_q、kv offsets 内容==q offsets → 数值不变。

## Stage B 验证
1. **build 0 error**(`tile_example_hstu_attention_bwd`,exit 0)。
2. **self 零回归不破**(硬要求):
   - co_symbols verify vs Stage A 基线:**486/486 byte-identical,0 DIFF,0 MISSING**(加字段/改 grid 是 host 码,device 不变)。日志 `runs/cross-stageB-cosym-verify.log`。
   - self 套件 **220/220 exit 0**。日志 `runs/cross-stageB-suite.log`。
3. **cross harness 就绪 — 3 smoke 全 PASS**(早期信号,非 Stage C 全矩阵):
   - `q<kv` jagged SiLU causal=1:dQ/dK/dV PASS(max_abs_err 2e-3/1e-4/4e-3)。
   - `q>kv` jagged SiLU causal=1(max_seqlen_kv<max_seqlen_q grid-shrink 路):PASS(err ~5e-7/0/0)。
   - `q<kv` determ multi-block softmax causal=1(grid/num_splits over max_seqlen_kv,R4 路):PASS。

## 裁决请求
Stage B done:build 0 error,self 仍 **486/486 byte-identical + 220/220**,cross harness 就绪(双向 + determ multi-block 3 smoke PASS)。**停,等 lead 亲核放行 Stage C(cross 对拍双向 + §6 矩阵翻转 + 单点 mask ctor 逐字对齐验证)。不 commit。**

> 注:smoke 已意外覆盖部分 Stage C 信号(双向 + R4),但 §6 全矩阵(P1-1 逐配置 / group / 非整除 / batched / fp16 / 单点 ctor 对齐验证)仍待 Stage C 系统跑,不在此 over-claim。
