# Draft — HSTU bwd GPU 实现计划(skill 闸门)

> 闸门文档:本文件存在后方可动 kernel 代码。基于已批准 DESIGN(gfx950 优先)。

## 1. Baseline 行为与验证方法
- 现状:HSTU **fwd** 有完整 GPU kernel(`tile_example_hstu_attention` target,gfx950 路径齐全);**bwd 无 GPU 实现**,只有 855 行 CPU reference。
- baseline 动作:先在本机 gfx950 **构建并跑通现有 fwd example**,确认工具链 + 取得"fwd 产 O/LSE"能力(bwd 对拍 harness 依赖它)。
- 验证 = 对拍 CPU reference(task-contract §5)。

## 2. 主要风险与未知(对应 DESIGN §8.2)
- **R1** FMHA default policy 能否直接复用(M1 一次性验)。
- **R2** 留 g 的 VGPR/occupancy —— **gfx950 是 CDNA4 加法模型(ArchVGPR+AGPR),风险高于 gfx942**;M1 按加法验 `ScratchSize=0` 且不掉 wave;溢出则 g 暂存 LDS(CDNA4 LDS 更大,代价低)。
- **R3** `GetTileRangeAlongY` 5 因子叠加非连续 → 连续保守超集 + 离线校验(M2 前置)。
- gfx950 tile 用 CDNA4 K-doubled MFMA(32×32×16/16×16×32),镜像 fwd_setting gfx95。

## 3. 候选方向(= 里程碑,排序;期望值 vs 风险)
- **M0 脚手架**(低风险,必做先行):bwd params struct、3 kernel 空壳、dispatch、CMake bwd target、instances bwd 分支、CLI、对拍 harness。验收:编译过 + launch 不崩 + 全 0 输出 + fwd baseline 跑通。
- **M1 端到端闸门**(最高价值/最高风险):batched+SiLU+no-mask+bf16+atomic+hdim64,MAIN(5 GEMM + dsilu + masked-out 0 + scale_p)+ 平凡 GetTileRangeAlongY + 保留 BiasEnum dummy + float dq_acc+POST convert。**一次性验 R1+R2(按 CDNA4 加法占用)**。验收:对拍过 bf16 阈值 + ScratchSize=0 + 不掉 wave。
- M2 mask 因子 → M3 jagged → M4 group → M5 softmax(PRE+LSE)→ M6 deterministic → M7 多 dtype/hdim → M8 perf。(详见 DESIGN §6)

## 4. 头几个具体步骤(M0)
1. baseline:`cmake` 配置 + 构建现有 fwd example(gfx950),跑一个 fwd case 确认 O/LSE 正常。
2. 在 `18_hstu_attention/` 起 bwd 文件骨架(params struct、3 kernel thin 包装、batched backward dispatch 空壳)。
3. CMake 加 `tile_example_hstu_attention_bwd` target(EXCLUDE_FROM_ALL)+ gfx95 flags。
4. 写对拍 harness(`example_hstu_attention_bwd.cpp`:gen 输入 → GPU fwd+bwd → CPU reference → check_err)。
5. 编译通过 + 全 0 kernel launch 不崩。

## 5. 精确 validation / evaluation 命令
见 task-contract §5/§6。M0 先用 fwd target 验证工具链;bwd target 随 M0 建立后接入 `--bwd_v 1`。

## 6. 晋级/否决一个候选需要的证据
- 晋级:validation PASS(分张量 err 截图入 runs/)+(性能候选)benchmark.csv 数字 + 无回退。
- 否决:记 candidates.jsonl 的 reason(对拍 FAIL 的 err / 编译错 / occupancy 掉 wave 等)。
- M1 额外:`--save-temps`/资源用量确认 ScratchSize=0、VGPR 数(按 CDNA4 加法算 occupancy)。

## 7. 第一动作
**建立 baseline:配置 cmake(gfx950)+ 构建现有 fwd example + 跑通一个 case。** 然后进 M0 脚手架。
