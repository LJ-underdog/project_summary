# M7c 图文并茂 HTML 讲义 —— 文档(pane 0.2)

你刚做完 M7c 对抗 review(含两条 reverse-proof + dq_acc 分析 + N1 store-skip 谓词落点),技术理解最深。写 **M7c HTML 讲义**(HANDOFF §3 铁律③)。M7c 已四方闭合 promoted、commit `17515fcc`。

## 0. 用 skill + 风格 + 纪律
- **必用 skill `html-report`**(`/root/.claude/skills/html-report/SKILL.md`)。
- 风格对齐同系列最新:`/root/workspace/hstu-b1052-report/hstu-bwd-M7b-hdim-20260611.html` + `hstu-bwd-M7a-fp16-20260611.html`。
- 输出:`/root/workspace/hstu-b1052-report/hstu-bwd-M7c-hdim-pad-20260615.html`。
- **排版纪律**:skill 无-emoji 铁则——别用 ★/✓/✗/方块 dingbat,纯文字(→/⇒ 箭头可留)。

## 1. 素材(只读,数字必与素材一致,不臆造)
- `docs/M7c-done.md`、`docs/draft-M7c.md`(批准设计)、`/tmp/hstu-bwd-design/M7c-review-findings.md`(你写的)、`M7c-stage{1,2,3}-*.md`。
- candidates.jsonl 末行 M7c、HANDOFF M7c 块。`git -C /root/workspace/ck_hstu show 17515fcc --stat`。

## 2. 讲义必讲清(图文并茂)
1. **M7c 是什么 + 范围**:接受 hdim_qk≠hdim_v + 非典范 hdim(48/80/100/200…)via head-dim padding;**明确** 真 reject hdim>256、非方形 tile out-of-scope。能力边界:任意 hdim_qk/hdim_v∈(0,256]。
2. **★ 核心洞察(最值得图解)**:pre-M7c bwd 的 pad 机制**已接线但死**(每个 DRAM view/LDS/epilogue 都 honor kPadHeadDimQ/V,只是被喂 constexpr 0 + guard 挡)。M7c = 运行时 BOOL_SWITCH_2 打开它,**不新增 instance**。配"死代码→激活"前后对比图。
3. **pad 模型**:右 pad = 逐元素 validity 谓词(非 clamp);OOB load 零替代、OOB store 跳过。逐 GEMM(GEMM0/2 收缩列 load 0;GEMM1/3/4 输出列 store-skip)。
4. **★★ 怎么证 OOB 真归零(亮点,别讲虚)**:poison-pad —— 输入 pad 尾 NaN 填(证 load-zero:泄漏→NaN→硬 FAIL)+ 输出尾预填 NaN(证 store-skip dK/dV)。**两条 reverse-proof 判伪**(强制 pad=false→NaN FAIL;强制 dram-view 谓词 sequence<false,false>→store-skip FAIL)证检查非 vacuous。这是 M7c 最强的工程论证,讲透。
5. **N1 精确归因(你 review 的发现)**:head-dim store-skip 的**载荷元件是 DRAM-view pad 谓词**(`sequence<false,(kPadHeadDimQ>0)>`),epilogue 的 bool flag 对 head-dim 冗余。
6. **dq_acc store-skip production 安全**:poison over-alloc 吸收 OOB 写=测试盲区,但代码核实(同 dK/dV 谓词,reverse-proof 已证该谓词有效)→ production exact-alloc 无 OOB。诚实标盲区。
7. **零回归**:canonical pad=0 设备符号 294/294 byte-identical(co_symbols.py 工具);库 pipeline/kernel/reference 未碰。
8. **四方闭合**:coder 4-stage + reviewer 独立 build_review 9/9 + lead 逐 stage 亲核 + reverse-proof。
9. **数据**:batched poison 168/168、group poison 96/96、套件 220/220、R9 hdim=100 跑通。

## 3. 纪律
- 只据素材、不臆造、范围诚实(任意 hdim≤256 但 hdim>256 reject、非方形 out);无 dingbat/外链/占位符;图 SVG/CSS 自包含。
- 完成 pane 里一句话报路径,等 pane-3 文档级 review。
