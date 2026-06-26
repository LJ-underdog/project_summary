# M7b 图文并茂 HTML 讲义 —— 文档(pane 0.2)

你刚做完 M7b 对抗 review,技术细节最熟。现写 **M7b 的图文并茂 HTML 讲义**(HANDOFF §3 铁律③)。M7b 已四方闭合 promoted、commit `1ae97750`。

## 0. 用 skill + 看历史风格
- **必用 skill `html-report`**(`/root/.claude/skills/html-report/SKILL.md`)。
- 风格对齐同系列最新两篇:`/root/workspace/hstu-b1052-report/hstu-bwd-M7a-fp16-20260611.html` + `hstu-bwd-M6b-group-determ-20260610.html`。
- 输出:`/root/workspace/hstu-b1052-report/hstu-bwd-M7b-hdim-20260611.html`。
- **排版纪律(M6b doc-review 提的)**:skill 无-emoji 铁则——别用 ★/✓/✗ dingbat,用纯文字(箭头 →/⇒ 可留)。

## 1. 素材(只读,数字必与素材一致,不臆造)
- `/tmp/hstu-bwd-design/M7b-done.md`、`M7b-review-findings.md`(你自己写的)、`M7b-stage1-checkpoint.md`、`M7b-draft.md`(批准设计)。
- candidates.jsonl 末行 M7b、HANDOFF M7b 块。
- 代码:`git -C /root/workspace/ck_hstu show 1ae97750 --stat` + `hstu_attention_bwd_shape.hpp`。
- profile:`/root/workspace/hstu-bwd-impl/profile/M7b-hd256-resource.md`。

## 2. 讲义必讲清(图文并茂)
1. **M7b 是什么 + 范围**:symmetric hdim{64,96,128,256}(hdim_qk==hdim_v);**明确** hdim_qk≠hdim_v / 非典范 hdim = M7c(guard 挡),别夸大成任意 hdim。
2. **★ 核心洞察(最值得图解)**:pre-M7b dispatch 硬编码 hd64 tile、MaxK 穿透但没用来选 shape → 直接加 headdim 轴 = 静默复用 hd64 tile = silent-wrong。配"MaxK→shape selector"前后对比图。
3. **shape selector**:`HstuBwdShape<MaxK>` 四特化(蓝本 FMHA bwd codegen gfx9 非 trload);**<64> 与原硬编码逐字等价=hd64 零回归基石**。配 4 hdim 的 tile 参数表(bm0/bn0/.../warps2 差异:96→warps2<2,2,1>、128/256→bm0=16、256→bn0=64)。
4. **guard 防 silent-wrong**:非典范 hdim(80/100)经 HDIM_SWITCH 静默选大 tile+dpad=0 会错 → 入口 guard `hdim_qk!=hdim_v||hdim_qk!=MaxK` throw。
5. **harness kN0 修复**:hd256 bn0=64,`kN0_bwd=(hdim==256)?64:128`,否则 determ workspace num_splits 失配越界。
6. **分阶段 + stage1 硬检查点**:先证重构零回归(selector<64> 同型/hd64 byte-identical/106 套件)再加 hdim。
7. **★ 二进制符号反证(亮点)**:rocprofv3 抽 kernel 符号 LDS 32/48/64KB 多档并存 → 证 tile 真随 hdim 分化(若静默复用 hd64,LDS 应恒 32KB)。讲透这个"怎么证明 silent-wrong 没发生"。
8. **hd256 资源**:Scratch=0 无 spill、VGPR172-184<256;occupancy=1(LDS 占满 64KB)=已知 perf 特征留 M8,非正确性问题。
9. **四方闭合**:coder + reviewer 独立 build_review + lead 亲核 + 二进制反证。
10. **数据**:sweep 128/128、套件 171/171(reject-hdim128→pass + 2 guard reject + 60 pass + 12 determ repro byte-identical)、库 byte-level 零回归。

## 3. 纪律
- 只据素材、不臆造;范围诚实(symmetric only);无 dingbat/占位符;图 SVG/CSS 自包含单文件。
- 完成后 pane 里一句话报路径,等 pane-3 文档级 review。
