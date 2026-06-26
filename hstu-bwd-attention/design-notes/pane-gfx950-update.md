# 派给 pane — 把 HSTU bwd 方案更新为 gfx950/CDNA4 优先(DESIGN.md + HTML 都改)

调度模式:tmux pane。用户批准方案、U1–U4 全用默认,并指明**当前机器是 gfx950(MI350/CDNA4)**,需重点关注 CDNA4 差异。更新两份文档。不要派 sub-teammate。

## 要改的两个文件
1. `/tmp/hstu-bwd-design/DESIGN.md`(方案源)
2. `/root/workspace/hstu-b1052-report/hstu-bwd-design-20260604.html`(图文版,改完保持标签平衡 + TOC 锚点不破;新增内容配既有 CSS 类与 clay/olive/slate 配色)

## 已核实的 gfx950/CDNA4 事实(对到 rocm-ref + fwd 源码,照此写,不臆造)
- **MFMA 仍适用、wave 仍 64**(CDNA4)→ **算法/七阶段不变**,变的是 tile/occupancy/build。
- **占用率模型变了(最关键)**:CDNA3(gfx942)= 512 ArchVGPR + 512 AGPR 独立文件,occupancy = `max(ArchVGPR,AGPR)`;**CDNA4(gfx950)= 统一 512 寄存器池,occupancy = ArchVGPR + AGPR(加法)**(rocm-ref `occupancy-register-pressure.md` AGPR occupancy rule 行 / glossary AGPR、Register File 条)。
- **MFMA tile 翻倍**:CDNA4 新增 K-doubled 形状 `32×32×16` / `16×16×32`(f16/bf16,2× 吞吐 vs CDNA3 的 32×32×8/16×16×16)(rocm-ref `mfma-register-layout.md` CDNA4 表 :42-64)。
- **LDS 更宽裕**:CDNA4 LDS 比 CDNA3(64KB)大、**64 banks**(CDNA3 32)+ 新 `ds_read_tr` 转置读变体(B64_TR_B4/B8/B16、B96_TR_B6)(rocm-ref `lds-bank-conflicts.md`)。注:rocm-ref 内部 CDNA4 LDS 容量 128 vs 160KB 不一致 → **只写"更大",不给具体数**。
- **fwd 已有成套 gfx950 路径**(bwd 必须镜像):`BUILD_HSTU_FOR_GFX95_ONLY` 宏(全 dispatch 文件 + CMake)、CMake `-fno-slp-vectorize`(改善 gfx950 pipelining)、`hstu_attention_fwd_pipeline_policy.hpp` 的 `#ifdef __gfx950__` 设备分支(3 处)、`hstu_attention_fwd_setting.hpp` 的 gfx95 专用 tile setting。

## 具体改动

### 1. 目标芯片改为 gfx950 优先(§0 + 通篇口径)
- §0 摘要:目标芯片从"gfx942/gfx950"改为"**主目标 gfx950(MI350/CDNA4),gfx942 次之**"。
- HTML 页头 subtitle / TL;DR 同步。

### 2. 新增小节「§gfx950/CDNA4 注意事项」(建议放 §4 工程末或 §8 前,HTML 同步加 + TOC 加条目)
要点:
- **占用率加法(升级 R2/P1-B)**:留 g 的 +~32 ArchVGPR 在 gfx950 上**叠加**到 MFMA 的 AGPR 累加器(dk_acc/dv_acc/dq_acc 在 AGPR)之上(CDNA4 加法模型),**比当前按 gfx942 `max()` 的估计风险更高**。→ M1 的 VGPR 验收必须按 **CDNA4 加法模型**算 occupancy;若掉 wave,优先用 g 暂存 LDS 的 fallback(CDNA4 LDS 更大 + 64 banks + ds_read_tr 利好,代价更低)。
- **tile/MFMA presets**:gfx950 tile_setting 用 **K-doubled MFMA(32×32×16 / 16×16×32)**,别照搬 gfx942 的 32×32×8/16×16×16;走 gfx95 分支,镜像 `hstu_attention_fwd_setting.hpp` 的 gfx95 路径。
- **build/结构镜像 fwd**:bwd dispatch/setting 加 `BUILD_HSTU_FOR_GFX95_ONLY` 分支、CMake 对 gfx95 加 `-fno-slp-vectorize`、policy/pipeline 内需要处加 `#ifdef __gfx950__`。
- 不变项:wave64、MFMA 适用、算法/七阶段双路、scale 接线、复用边界 —— 这些 gfx942/gfx950 一致。

### 3. 调整受影响的既有条目
- **§2.4 / §8.2-R2 / §8 P1-B**:占用估计补"gfx950 为加法模型,风险更高;M1 按加法验"。
- **§4.2 tile presets**:从"照搬 FMHA gfx942/gfx950 预设"改为"gfx950 用 CDNA4 K-doubled MFMA tile,镜像 fwd_setting gfx95 分支"。
- **§4.5 CMake**:补 gfx95 `-fno-slp-vectorize`、`BUILD_HSTU_FOR_GFX95_ONLY`。
- **§6 M1 验收**:VGPR/occupancy 判据标注"按 CDNA4 加法模型"。
- **§4.1 文件结构**:bwd setting 文件要有 gfx95 分支(对齐 fwd_setting)。

### 4. 锁定 U1–U4
§8.3 决策卡 + HTML 决策卡:把"待用户确认"改为「**✅ 用户已确认(2026-06-04):全用默认**」,保留每项默认值说明。

## 铁则
- 只做 gfx950 相关更新 + 锁 U1–U4,不改其它技术结论。gfx950 事实严格按上面(对 rocm-ref/fwd 源码),不臆造容量/周期具体数。
- 两文件都改;HTML 改完自检标签平衡(div/section/svg/h2/h3/table)+ TOC 锚点 + 新增可配 1 张小 SVG/表说明 CDNA3 vs CDNA4 占用差异(可选)。
- 完成写 `/tmp/hstu-bwd-design/gfx950-done.md`:改了哪些节、HTML 标签平衡、U1–U4 是否标为已确认。正文写进文件,不在终端长输出。
