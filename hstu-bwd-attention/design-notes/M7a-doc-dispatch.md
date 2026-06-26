# M7a fp16 图文并茂 HTML 讲义 —— 文档 pane(pane 0.2)

你刚做完 M7a 对抗式 review(`M7a-review-findings.md`),最熟这批改动。现在补 **M7a 的图文并茂 HTML 讲义**(HANDOFF §3 铁律③:解释性文档 HTML、放 `/root/workspace/hstu-b1052-report/`)。M7a 已四方闭合 promoted、commit `bf82a1d2`,但漏了 HTML 文档,补上。

## 0. 用 skill + 看历史风格
- **必用 skill `html-report`**(`/root/.claude/skills/html-report/SKILL.md`)。
- 对齐历史风格:读 `/root/workspace/hstu-b1052-report/hstu-bwd-M6-deterministic-20260609.html` 和 `hstu-bwd-M5-softmax-20260608.html`(同系列,图文/配色/结构参照)。
- 输出:`/root/workspace/hstu-b1052-report/hstu-bwd-M7a-fp16-20260611.html`。

## 1. 素材(只读,别臆造)
- `/tmp/hstu-bwd-design/M7a-done.md`(coder:构建/sweep 66/66/套件 106/106/误差量级)。
- `/tmp/hstu-bwd-design/M7a-review-findings.md`(你自己写的:7 条 GREEN/容差 revert 双实验/fp16-ULP 反证)。
- candidates.jsonl 末行 M7a(四方闭合 reason)。
- HANDOFF.md M7a 块 + 能力边界。
- 代码:`git -C /root/workspace/ck_hstu show bf82a1d2 --stat` + 看 diff(4 改 + fp16 entry/instance)。

## 2. 讲义必讲清(图文并茂,面向"想懂 M7a 怎么做对的"读者)
1. **M7a 是什么**:fp16 dtype 加宽(纯 dtype 轴,hd64,hdim_qk==hdim_v);为何能"复用而非重写"——dispatch/kernel/pipeline 本就模板化于 `InOutDataType`,fp16 只在边界加 dtype。配一张"模板码路共享、仅 dtype 边界分叉"的示意图。
2. **来源故事**:上轮 session 留的未提交孤儿 WIP → lead 恢复后 stash 立干净 M6b 边界 → 评估通过复用(可画时间线/流程图)。
3. **为何 fp16 比 bf16 容差更紧**:fp16 尾数 10bit > bf16 7bit(存储更准);elimit rtol5e-3/atol1e-2 vs bf16 2e-2/5e-2。配 bf16 vs fp16 位布局对比图(指数/尾数位)。
4. **怎么验"真在跑 fp16 而非静默落回 bf16"**:fp16-ULP(2⁻¹⁴)误差量级 vs bf16-ULP(~8×)的反证逻辑——这是亮点,讲透。
5. **容差没放水的证明**:revert 双实验(放宽到 bf16 容差仍过=无隐藏误差;收紧 5-20× 仍过=误差真小)。配实验结果表。
6. **零回归如何保证**:库/kernel/pipeline/dispatch/reference byte-identical 于基线;只改 entry/instance/harness/generator/CMake。
7. **四方闭合**:coder + reviewer 独立 build_review + WIP 本体 + lead 亲核(三项独立证据)。配闭合矩阵图。
8. **能力边界 & 范围**:扩到 bf16+fp16 全模式;明确 hd64-only、hdim_qk≠hdim_v 是 M7b/M7c(诚实标范围,不夸大)。
9. **数据**:sweep 66/66、套件 106/106、误差量级表(SiLU rel~1e-3 / softmax rel~1e-4)。

## 3. 纪律
- **只据素材,不臆造数字**;误差/case 数与 done.md/review 一致。
- 范围诚实(hd64-only 别写成"任意 hdim")。
- 图用 SVG/CSS(html-report skill 风格),自包含单文件。
- 完成后 pane 里一句话报路径,等 pane-3 文档级 review。
