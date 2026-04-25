# V04 TP Support 验证

## C.2 CK manifest grep

inter_dim=192 切换边界证据（来自代码，非推断）：

- `/home/hanchang/aiter/aiter/fused_moe.py:900-910`
  - L900-903 注释明确："gfx950 workaround: V1 CK kernel produces wrong results for inter_dim>192 (memory corruption / incorrect computation for both preshuffle_on and preshuffle_off paths). Force block_m=128 to select the correct V3 stage1 kernel."
  - L906-908 条件：`if not run_1stage and inter_dim > 192 and get_gfx() == "gfx950" and q_type not in (QuantType.per_1x128, QuantType.per_1x32): block_m = 128`
  - L910 强制 V3 stage2 kernel：`kernelName2 = "moe_ck2stages_gemm2_256x128x128x64_1x4_TypeCast_v3_..."`

- `/home/hanchang/ATOM/atom/model_ops/moe.py:498-502`
  - L498-499："Stage1 dispatch: inter<=192 uses NPerBlock=64, inter>192 uses NPerBlock=128. Stage2 dispatch: inter>192 uses KPerBlock=64."
  - L502 align 计算：`align = 64 if inter_dim <= 192 else 128`

结论：inter_dim=192 是 V1/V3 切换边界，由 aiter `fused_moe.py:906` + ATOM `moe.py:502` 共同决定。CK codegen 目录 `/home/hanchang/aiter/csrc/ck_gemm_moe_2stages_codegen/` 存在（含 `gemm_moe_ck2stages.cu/.h/_common.cuh`），但具体 NPerBlock/192 字面值由 Python 侧 dispatch 决定。

## ca_comm fallback 存在性

`/home/hanchang/aiter/aiter/dist/parallel_state.py`：
- L238: `ca_comm: Optional[Any]  # Custom allreduce communicator`
- L363-365: `ca_comm = self.device_communicator.ca_comm; if ca_comm is not None: maybe_ca_context = ca_comm.capture()`
- L492-501: `_all_gather_out_place`：`ca_comm = self.device_communicator.ca_comm; if ca_comm is None: torch.distributed.all_gather_into_tensor(...)`（fallback 到原生 NCCL/RCCL）

ATOM `/home/hanchang/ATOM/atom/model_ops/moe.py`：未引用 ca_comm（依赖 aiter 的 parallel_state 抽象）。

结论：ca_comm fallback 机制在 aiter parallel_state.py 存在（ca_comm is None → torch.distributed 路径）。

## Exp3 tp=2 回归（tp=1 单卡 OOM，已跳过）

跑 tp=2（GPU 4,6），日志：`/home/hanchang/project_fp8_tp4/verification_pipeline/results/logs/v04_exp3_tp2.log`

- Request 2: TTFT=0.092s TPOT=0.018s
- Request 0: TTFT=0.092s TPOT=0.018s
- Request 1: TTFT=0.092s TPOT=0.018s
- Request 3: TTFT=0.092s TPOT=0.018s

TTFT=92ms TPOT=18ms（vs BF16 tp=2 基线 TTFT=85ms TPOT=18ms，±20% 内）。
输出正常无乱码（中英文 prompt 均生成合理 completion）。

结论：**PASS**

注：tp=1（GPU 4 单卡）首次尝试 OOM 失败（torch 误报 GPU 0 251GB），单卡显存不足以装下完整模型权重，已跳过；改用 tp=2 跑 GPU 4,6 即成功。本节实际只覆盖 tp=2。

## Exp1 inter_dim padding 确认

ATOM `/home/hanchang/ATOM/atom/model_ops/moe.py` L489-518：

- L489-494 注释："gfx950 CK a16w16 stage2 requires inter_dim % 64 == 0. For tp=4 (inter=320) and tp=8 (inter=160), pad inter_dim up to the next multiple of 64. Verified 2026-04-24: cos_sim >= 0.9999 for inter=160->192 and inter=320->384."
- L502: `align = 64 if inter_dim <= 192 else 128`
- L503: `inter_pad = (inter_dim + align - 1) // align * align`
- L504-518: 实际 zero-pad w13/w2 张量

inter_dim ∈ {192, 384} 在 padding 后路径下正确性已由 V01-Exp2 覆盖（cos_sim >= 0.9999 已记录于代码注释）。

## Exp2 tp=4（未跑）

V04-A2 已 PASS，但 tool calls 已接近预算（~10），且 Exp2 涉及 4 卡冷启动 + ~3 分钟运行时间，按预算约束**未运行**。BF16 tp=4 基线（TTFT=84ms TPOT=18ms）已在 V01-Exp3 验证。

结论：**未跑**

## Exp2 tp=4 BF16 e2e 验证（2026-04-25 补跑）

GPU 0,1,2,3，日志：`/home/hanchang/project_fp8_tp4/verification_pipeline/results/logs/v04_exp2_tp4.log`

命令：`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m atom.examples.simple_inference --model stepfun-ai/Step-3.5-Flash --kv_cache_dtype bf16 --trust-remote-code --tensor-parallel-size 4 --level 0 --temperature 0 --max-tokens 128 --max-num-batched-tokens 4096 --max-num-seqs 2048`

请求实测：
- Request 2 (eos):        TTFT=0.081s TPOT=0.015s (in=19, out=60)
- Request 0 (max_tokens): TTFT=0.082s TPOT=0.017s (in=16, out=128)
- Request 1 (max_tokens): TTFT=0.081s TPOT=0.017s (in=20, out=128)
- Request 3 (max_tokens): TTFT=0.081s TPOT=0.017s (in=21, out=128)

均值：**TTFT=81.25ms, TPOT=16.5ms**

通过标准对比：
- TTFT 81.25ms < 106ms ✅
- TPOT 16.5ms  < 21ms  ✅
- 无 crash，干净退出 ✅
- 无 BOS-spam，4 个 prompt（中英文混合）均生成合理 completion ✅

结论：**PASS**

## V04 总结

| 验证项 | 状态 | 备注 |
|--------|------|------|
| C.2 CK manifest grep | PASS | inter_dim=192 V1/V3 切换边界由代码 dispatch 确认 |
| ca_comm fallback 存在性 | PASS | aiter parallel_state.py 实现 ca_comm is None 分支 |
| Exp1 inter_dim padding 确认 | PASS | ATOM moe.py L489-518 padding 逻辑（160→192, 320→384） |
| Exp3 tp=2 回归 | PASS | TTFT=92ms TPOT=18ms（tp=1 因单卡 OOM 跳过） |
| Exp2 tp=4 BF16 e2e | PASS | TTFT=81.25ms TPOT=16.5ms，无 crash 无 BOS-spam |
