# V00 Noise Floor — 基准测定结果

测试时间：2026-04-25
GPU: gfx950 (GPU 0, MI350X)
CUDA_VISIBLE_DEVICES: 0
脚本：`/tmp/v00_noise_floor.py`
日志：`/home/hanchang/project_fp8_tp4/verification_pipeline/results/logs/v00_noise_floor.log`

## 测试设计

固定 seed 下连续运行同一 op 5 次，计算每次输出与 reference 输出的 cos_sim，取 5 次中最坏值作为噪声基准。
- Test 1 GEMM：`A @ B`，A=(32,7168) B=(7168,2048) BF16
- Test 2 fused_moe：aiter `fused_moe`，BF16，`preshuffle_off`，dims 接近 Step-3.5-Flash MoE
  - M=128, K(hidden)=4096, N(intermediate)=1536, E=128, topk=8
  - 走的 kernel：`module_moe_ck2stages_b16_b16_preshuffle_off_b16_silu_no_mulWeightStage2`
- Test 3 RMSNorm：`torch.nn.functional.rms_norm`，hidden=7168 BF16

> 注：fused_moe BF16 `preshuffle_on` 路径在 gfx950 上有已知 correctness bug（见 `memory/moe-kernels.md`），本基准只覆盖 `preshuffle_off`。preshuffle_on 路径不可作为 noise floor 参考——任何使用该路径的对比必须独立度量。

## 测试结果

| Op | 5次运行最坏 cos_sim | 噪声幅度 (1 - cos_sim) | 建议安全阈值 (10× 噪声) |
|----|------------------|---------|------------|
| gemm_bf16        | 0.9999999404 | 5.96e-08 | 0.999999 |
| fused_moe_bf16 (preshuffle_off) | 0.9999981523 | 1.85e-06 | 0.999982 |
| rms_norm_bf16    | 0.9999999404 | 5.96e-08 | 0.999999 |

5 次内 GEMM/RMSNorm 完全 deterministic（cos_sim 抖动只在 BF16 单 ULP 量级，源自 ref 与 out cast 顺序），fused_moe 5 次有轻微抖动（最大差 ~1.5e-06），属正常 atomic-add reduction 噪声。

## 结论：各验证专题 cos_sim 阈值确认

| 阈值 | 评估 | 说明 |
|------|------|------|
| `cos_sim ≥ 0.9999`     | **合理（偏宽松）** | 比 fused_moe noise floor (1.85e-06) 宽 ~54×，能稳定通过同算子重复，适合做 BF16 vs FP8 cross-impl 对比 |
| `cos_sim ≥ 0.99998`    | **合理** | 仍高于 fused_moe noise floor（1 - 0.99998 = 2e-05，比 noise 大 ~10×），落在 10× margin 上，能区分实质性 numerical 差异 |
| `cos_sim ≥ 0.999998`   | **偏严** | 已逼近 fused_moe noise floor 的 1× margin，对 MoE 类算子的 batch 抖动不友好；非 MoE op（GEMM/RMSNorm）可用 |
| `cos_sim ≥ 0.9999999`  | **过严** | 接近所有 op 的 ULP 量级噪声，会产生 false negative |

**PASS/FAIL 标准建议（不变）：**
- GEMM / RMSNorm / Attention 类（无 atomic reduction）：`cos_sim ≥ 0.99998` 合理
- fused_moe / 其它含 atomic-add reduction 的算子：`cos_sim ≥ 0.9999` 合理（保守）；`≥ 0.99998` 也可接受但对 batch 顺序敏感
- 端到端 logits 对比：`cos_sim ≥ 0.999` 即可，因为多层累积放大 noise

理由：noise floor 是单算子 5 次重复抖动；实际验证的是不同 impl（BF16 vs FP8、preshuffle_on vs off）的 functional equivalence，阈值留出 ~10× margin 既能 catch 真 bug 又能避开 atomic reduction 不确定性。

## 已知限制

- 只测 5 次，未覆盖长尾分布；如需更严格分析需扩到 100+ 次并看 P99
- fused_moe 只测了 BF16 `preshuffle_off`，未测 FP8 / preshuffle_on（preshuffle_on 在 gfx950 BF16 已知 broken）
- 单 GPU 测量，多卡 all-reduce 引入的额外 noise 需独立基准（建议后续 V01 / V02 补充）
