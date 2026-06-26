# 文档级 review：M7a HTML 讲义(pane 0.3)

pane-2 写了 M7a 的 HTML 讲义 `/root/workspace/hstu-b1052-report/hstu-bwd-M7a-fp16-20260611.html`。你做**独立文档级 review**。只读不改文档本身,把问题列给 lead/写作者。

## 审查清单(逐条 GREEN/RED + 证据)
1. **数字一致**:所有数字(sweep 66/66、套件 106/106、误差量级 SiLU rel~1e-3 / softmax rel~1e-4、容差 fp16 5e-3/1e-2 vs bf16 2e-2/5e-2、commit `bf82a1d2`)必须与素材 `/tmp/hstu-bwd-design/M7a-done.md` + `M7a-review-findings.md` + candidates.jsonl 末行一致,**无臆造**。逐一核。
2. **★ 范围诚实(最关键)**:M7a 只做 **fp16 dtype 加宽、hd64、hdim_qk==hdim_v**。讲义**不得**暗示支持任意 hdim 或 hdim_qk≠hdim_v;必须明确 hdim{96,128,256} 与 hdim_qk≠hdim_v 是 M7b/M7c。确认无夸大。
3. **技术叙述准确**:
   - "复用而非重写"=dispatch/kernel/pipeline 本就模板化于 InOutDataType,fp16 只在边界加 dtype;
   - fp16 容差更紧的理由=尾数 10bit>bf16 7bit;
   - **fp16-ULP(2⁻¹⁴)vs bf16-ULP(~8×)反证"真在跑 fp16"** 的逻辑是否讲对(这是亮点,别讲歪);
   - 容差 revert 双实验(放宽仍过=无隐藏误差、收紧仍过=误差真小)结论对不对。
4. **零回归表述**:库/kernel/pipeline/dispatch/reference byte-identical;只改 entry/instance/harness/generator/CMake。
5. **四方闭合**:coder + reviewer 独立 build_review + WIP 本体 + lead 亲核(3 项),呈现准确。
6. **HTML 质量**:可渲染、图正常、无占位符/溢出。

## 产出
写 `/tmp/hstu-bwd-design/M7a-doc-review.md`,逐条 GREEN/RED,RED 给位置 + 应改。结论:可发布 / 需改。完成 pane 里一句话报。
