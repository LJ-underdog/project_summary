# gfx950 / CDNA4 更新完成 — DESIGN.md + HTML 双文件

用户已批准方案、U1–U4 全用默认,并指定**当前机器 gfx950(MI350/CDNA4)、重点关注 CDNA4 差异**。两文件均已更新,只做 gfx950 相关改动 + 锁 U1–U4,未改其它技术结论;gfx950 事实严格对到 rocm-ref / fwd 源码,未臆造容量/周期具体数。

## 一、改了哪些节

### `/tmp/hstu-bwd-design/DESIGN.md`
- **§0 摘要**:目标芯片 → 「**主目标 gfx950(MI350/CDNA4),gfx942(CDNA3)次之**」,指向 §4.8。
- **§2.4(留 g / P1-B)**:补占用率模型按芯片分叉 —— CDNA3 `max(ArchVGPR,AGPR)` vs **CDNA4 `ArchVGPR+AGPR`(加法)**,+32 ArchVGPR 叠加 AGPR 累加器,gfx950 风险更高;M1 按加法模型验;fallback(g 暂存 LDS)在 CDNA4 代价更低。
- **§4.1 文件结构**:`hstu_attention_bwd_setting.hpp` 必须含 gfx95 专用 tile setting 分支(对齐 fwd_setting);dispatch 镜像 `BUILD_HSTU_FOR_GFX95_ONLY`。
- **§4.2 tile preset**:从「照搬 gfx942/gfx950 预设」改为按芯片分叉 —— gfx950 用 **CDNA4 K-doubled MFMA(32×32×16 / 16×16×32)**,镜像 fwd_setting gfx95 分支。
- **§4.5 CMake**:补 gfx95 target `-fno-slp-vectorize` + `BUILD_HSTU_FOR_GFX95_ONLY`。
- **§4.8(新增小节)gfx950 / CDNA4 注意事项**:①占用率加法模型(升级 R2/P1-B,M1 按加法验) ②K-doubled MFMA tile ③build 镜像 fwd(宏 + `-fno-slp-vectorize` + `#ifdef __gfx950__` 3 处) ④LDS 更宽裕(64 banks + `ds_read_tr`,只取定性"更大") + 不变项清单。
- **§6 M1 验收**:标注主目标 gfx950 + gfx95 tile preset;VGPR/occupancy 判据「**按 CDNA4 加法模型 `occupancy=ArchVGPR+AGPR`**」。
- **§8.2-R2**:补 gfx950 加法模型、风险高于 gfx942、按加法验、LDS fallback 代价更低。
- **§8.3 决策卡**:标题改「**✅ 用户已确认 2026-06-04:全用默认**」;U1–U4 每项前缀 `建议默认`→`✅ 已确认`;新增"已批准 + 指定 gfx950"前言。
- **§8.5-P1-B 行 + 结尾拍板状态**:补 CDNA4 加法模型;结尾改「✅ U1–U4 已全部确认,无待确认项 + 指定 gfx950」。

### `/root/workspace/hstu-b1052-report/hstu-bwd-design-20260604.html`
- **页头日期行 + subtitle 区 + §0 lede**:目标芯片改 gfx950 优先(clay 高亮),链到 §4.8。
- **TL;DR 卡**:新增一句 gfx950 主目标 + CDNA4 差异概述(链 §4.8);徽章加「U1–U4 ✅ 用户已确认」、闸门徽章补「按 CDNA4 占用模型验」。
- **§2.4 note-block(P1-B)**:同 DESIGN,补加法模型对比 + fallback。
- **§4.1**:callout-grid 后加 clay note-block(setting gfx95 分支 + 宏镜像)。
- **§4.2 tile 行**:改按芯片分叉(gfx950 K-doubled MFMA,clay 高亮)。
- **§4.5 CMake**:补 gfx95 `-fno-slp-vectorize` + 宏(clay 高亮)。
- **§4.8(新增小节 + 图7 SVG)**:lede + **图 7「CDNA3 vs CDNA4 占用率模型对比」SVG**(左 olive max 模型 / 右 clay 加法模型,可视化 +32 叠加) + 4 张 callout(占用/preset/build/LDS) + 不变项 olive note。
- **§6 M1 表行 + M1 闸门 SVG 验收文字**:补主目标 gfx950 + CDNA4 加法模型判据。
- **§8.2-R2 表行 / §8.5-P1-B 表行**:同 DESIGN。
- **§8.3 决策卡**:h3 标题 + gate-banner 改「✅ 用户已确认(2026-06-04)」并指定 gfx950;4 张卡 `.rec` 全部 `建议默认`→`✅ 已确认`;结尾 note 改已确认状态(olive)。
- **TOC**:新增 `4.8 gfx950 / CDNA4(图7)★` 条目;`8.3` 标签改 `✅已确认`。

## 二、HTML 自检(全过)
- **标签平衡**:div 218/218、section 9/9、svg **7/7**、h2 9/9、h3 32/32、table 12/12、p 13/13、nav 1/1、main 1/1 —— 全 OK。
- **TOC 锚点**:全部 href="#…" 均有对应 id,**无悬空**(脚本校验);新 §4.8 锚点 `s4-cdna4` 已接入正文 + TOC + 多处交叉引用。
- **新增图 7 SVG 文字**:最坏右边缘 ≈778 < viewBox 880,**无溢出**。
- **配色**:新增内容复用既有 CSS 类(note-block/callout/decision-card/gate-banner/svg-caption),clay=#D97757 标 CDNA4 新写/风险、olive=#788C5D 标不变项/复用、slate 文字 —— 与全文一致。
- 字节:95,427(原 86,646)。gfx950/CDNA4 提及:HTML 25 处、DESIGN 18 处。

## 三、U1–U4 状态
✅ **全部标为「用户已确认(2026-06-04),全用默认」**(DESIGN §8.3 标题 + 4 项前缀 + 结尾;HTML h3 + gate-banner + 4 张决策卡 `.rec` + 结尾 note + TL;DR 徽章 + TOC)。保留每项默认值说明不变。

## 四、铁则遵守
- 仅 gfx950 相关更新 + 锁 U1–U4,未改算法/七阶段/scale/复用边界等其它技术结论。
- gfx950 事实严格按交付清单(占用加法模型、K-doubled MFMA 32×32×16/16×16×32、64 banks、ds_read_tr、-fno-slp-vectorize、BUILD_HSTU_FOR_GFX95_ONLY、#ifdef __gfx950__);LDS 容量按要求**只写"更大"不固化数值**(并注明 rocm-ref 内部记述不一致)。
