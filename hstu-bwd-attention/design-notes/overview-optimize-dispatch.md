# 优化 HSTU bwd 总览讲义 —— 加图 + 逻辑清晰(pane 0.3)

对象:`/root/workspace/hstu-b1052-report/hstu-bwd-overview-20260615.html`(lead 已写初版,纯文字+表,**缺 SVG 图**)。任务:**优化使其逻辑清晰、图文明确**(图文并茂),对齐系列视觉。**只优化这一个文件**。

## 0. 用 skill + 纪律
- **必用 skill `html-report`**;视觉对齐 `hstu-bwd-M8-perf-20260615.html` / `hstu-bwd-cross-attention-20260615.html`(同 ivory/slate/clay 调色已在初版 :root)。
- **无-emoji 铁则**(★/✓/✗/方块 dingbat 禁;→/⇒/×/≈ 数学箭头可)。
- **自包含单文件**、无外链(锚点 + 同目录 sibling html 链接除外)。

## 1. 必做:加清晰 SVG 图(现版 0 张图,这是主要短板)
按内容加 4–5 张 inline SVG(viewBox 不溢出、配色用 :root 变量):
1. **里程碑能力累积时间线**:M0 脚手架 → M1 SiLU 闸门 → M2 mask → M3 jagged → M4 group → M5/M5b softmax → M6/M6b determ → M7a fp16 / M7b hdim / M7c pad → cross → M8 perf;直观показ 能力逐里程碑叠加(可横向 timeline + 每节点标新增轴)。
2. **bwd kernel 结构**:PRE `dot_do_o`(D=rowsum(O·dO))→ MAIN 5-GEMM(dS/dV/dP/dK/dQ)→ POST convert/reduce;标 MAIN 占 84–90% wall-time。
3. **最终能力边界矩阵**:轴 = {模式 batched/jagged/group} × {self/cross} × {SiLU/softmax} × {bf16/fp16} × {hdim 对称/非对称/非典范} × {atomic/determ},直观показ全覆盖 + 真 reject(hdim>256)/out-of-scope(target_in_kv/非方形)。
4. **四方闭合 + 证据栈**:coder→reviewer(独立重建)→lead 亲核→对抗实验;底座工具 co_symbols byte-identity / poison-pad / 离线 superset 校验器 / reverse-proof。
5. **M8 perf 加速条形图**:causal 1.25–1.60×、window 4.7–9.8×(各 window 档)。

## 2. 逻辑清晰(优化结构,别改事实)
- 检查叙事流:概述→能力边界→时间线→性能→方法论→教训→git 链→后续,是否顺。可加一句「如何用本页」(给接手者的导航)。
- **不得改任何数字/commit/结论**(以 `docs/HANDOFF.md` + 各 `docs/*-done.md` + `benchmark.csv` 为准,核对后保持一致);**20 个内部链接必须全部保留且仍有效**(`ls` 对应文件)。
- 表格/figure 配文要呼应(图文明确:每图有 caption + 正文引用)。

## 3. 产出
- 原地优化 `hstu-bwd-overview-20260615.html`(覆盖)。完成 pane 里一句话报:加了哪几张图 + 逻辑调整点 + 链接/数字未动核对。
- 之后 lead 会派**另一个 pane 认真独立 review**(逻辑清晰度 + 图文正确 + 数字/链接核对),故你优化时务必自洽、不引入错。
