# 文档级 review:M7b HTML 讲义(pane 0.3)

pane-2 写了 M7b 讲义 `/root/workspace/hstu-b1052-report/hstu-bwd-M7b-hdim-20260611.html`。你做独立文档级 review,只读不改文档,问题列给 lead。

## 审查清单(逐条 GREEN/RED + 证据)
1. **数字一致**:128/128 sweep、171/171 套件(reject-hdim128→pass + 2 guard reject + 60 pass + 12 determ repro byte-identical)、commit `1ae97750`、hd256 资源(Scratch=0/VGPR172-184)——必与素材 `/tmp/hstu-bwd-design/M7b-done.md` + `M7b-review-findings.md` + candidates 末行一致,无臆造。
2. **★ 范围诚实(最关键)**:M7b = symmetric hdim{64,96,128,256}(hdim_qk==hdim_v)。**不得**暗示任意 hdim;hdim_qk≠hdim_v + 非典范 hdim = M7c(guard 挡)必须标清。
3. **技术叙述准确**:① 核心洞察(pre-M7b 硬编码 hd64 tile、MaxK 没用来选 shape → 直接加轴=silent-wrong)讲对;② shape selector <64> 同型=hd64 零回归;③ guard 防非典范/asymmetric;④ harness kN0=(256)?64:128 修 hd256 determ;⑤ 二进制符号反证(LDS 32/48/64KB 多档 → tile 真随 hdim 分化)讲对、别讲歪。
4. **零回归表述**:库/pipeline/kernel/reference byte-identical,只改 dispatch/entry/instance/generator/harness/新 shape.hpp。
5. **四方闭合**:coder + reviewer 独立 build_review + lead 亲核 + 二进制反证,呈现准确。
6. **HTML/排版**:可渲染、图正常、无占位符;**无 dingbat/emoji**(★/✓/✗,skill 无-emoji 铁则)、无外链。

## 产出
写 `/tmp/hstu-bwd-design/M7b-doc-review.md`,逐条 GREEN/RED + 证据,RED 给位置+应改。结论:可发布/需改。完成 pane 里一句话报。
