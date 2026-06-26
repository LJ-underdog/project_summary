# M7c 对抗式 review —— reviewer(pane 0.2,已 /clear)

你是 HSTU bwd 项目独立 reviewer(较真、对抗式,**默认怀疑、不信自述**)。coder 刚完成 **M7c:asymmetric hdim_qk≠hdim_v + 非典范 hdim(via head-dim padding)**,停在闭合检查点。独立对抗 review,只读不改库代码,GREEN/RED + 证据列给 lead。

## 0. 背景(只读)
- 基线 HEAD=`1ae97750`(M7b)。M7c 未 commit。素材:
  - `docs/draft-M7c.md`(批准设计,含逐 GEMM pad 分析 + 风险 R1–R11)、`docs/M7c-done.md`(coder 自述)、`/tmp/hstu-bwd-design/M7c-stage{1,2,3}-*.md`。
  - 改面:`git -C /root/workspace/ck_hstu diff 1ae97750 -- example/ck_tile/18_hstu_attention/`(4 文件:shape + batched/group dispatch + harness)。
- lead 已逐 stage 亲核(byte-identity 294/294、batched poison 168/168、group poison 96/96、suite 220/220、dq_acc store-skip 代码核实)。**你要独立复核、找洞,别复述 lead。**
- 铁律:对拍 CPU reference,`-attn_scale=1.0`;bf16 2e-2/5e-2、fp16 5e-3/1e-2。

## 1. 对抗审查清单(逐条 GREEN/RED + 文件:行号)
1. **★ 核心洞察成立**:pad 机制是否真"已接线只是被喂 0"?即激活后只是开 pad，没改库逻辑?`git diff` 确认 pipeline/kernel/reference **byte-identical**(仅 4 文件改)。
2. **canonical 零回归(byte-level)**:独立 `python3 test/co_symbols.py verify runs/M7c-stage0-baseline.json <build_review 的 backward .o>` → 294/294 0 DIFF?(你自己干净 `build_review` 重建后跑,不复用 coder build。)
3. **★ load-zero 正向证(poison)**:独立跑几个 poison 非典范案(batched + group、含双方向 64/128↔128/64、hdim=100),确认 NaN 输入不泄漏(输出有限、对拍 PASS)。**反证**:临时把某处 pad flag 强制成 false(或喂一个 guard 之外的非典范 hdim 走 pad=0)看是否 NaN/FAIL → 证 poison 能判伪、非 vacuous。验完恢复。
4. **★ store-skip(dK/dV)**:poison 输出 pad 尾 NaN 是否真保持(epilogue 跳写)?asymmetric pair pad 区非空确认。
5. **★★ dq_acc store-skip(最关键盲区,lead 已代码核实,你独立复核)**:`hstu_attention_bwd_kernel.hpp:373-381` dq_acc DRAM view 是否真带 `sequence<false,(kPadHeadDimQ>0)>` + mop set/atomic_add?即 **production(dq_acc 按真实 hdim_qk 跨步、exact-alloc)下 GEMM4 写 dq_acc 是否 store-skip pad 列、不 OOB**?这是 poison(over-alloc)测不到的点,必须靠代码。给独立判断:production 安全 or 有 OOB 风险。
6. **group ProblemFor**:`group_backward_dispatch.hpp:74` 的 `<0,0>` 是否真改成 pad NTTP 透传(Local/NoLocal 双 pipeline 都透)?group canonical byte-identity 在内?
7. **guard 放松正确**:`>MaxK` 放松后,hdim>256 仍被 `hdim_switch.hpp` else-throw 拒(独测 hdim=512 throw);非典范不再误拒。
8. **套件 220/220 独立复跑**:你 build_review binary 跑 `run_bwd_tests.py` → 220/220 exit 0?50 个 poison 案是否真带 poison 断言(非裸 PASS)?每 pair 含 P1-1 cross?容差未松?
9. **诚实范围**:hdim>256 reject、非方形 tile out-of-scope、R9 hdim=100 非 reject —— 与代码/实测一致,无夸大。

## 2. 产出
写 `/tmp/hstu-bwd-design/M7c-review-findings.md`,逐条 GREEN/RED + 证据行号;RED 给可复现配置。结论:M7c 可否 promote。**发现 RED 必给复现。** 完成 pane 里一句话报。
独立 build:`cmake -B build_review ... -DBUILD_DEV=OFF` + `--target tile_example_hstu_attention_bwd`(与 coder build 不冲突)。
