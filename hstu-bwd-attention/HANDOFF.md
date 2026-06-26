# HANDOFF — HSTU bwd GPU 实现(更新 2026-06-15)

> 给 /clear 后的 lead(pane 0.0)恢复用。先读本文件 + `candidates.jsonl` + 最近的 `M*-done.md`。**活的状态以本文件为准。**

## 0. 一句话(当前)
为 HSTU attention **backward** 实现 GPU kernel(原仅 855 行 CPU 参考,无 GPU bwd),复用 ck_tile FMHA bwd 基建,**目标芯片 gfx950 / MI350X / CDNA4(本机就是)**。**M0–M8 + cross-attention 全部 promoted、git 已提交(HEAD `048f0a9a`)。** 能力边界 = SiLU+softmax × **batched/jagged/group × self+cross(seqlen_q≠seqlen_kv 全方向)** × 全 5 因子 mask × causal{0,1} × **bf16+fp16** × **任意 hdim_qk/hdim_v∈(0,256](对称+非对称+非典范 via head-dim pad)** × atomic+deterministic。对拍套件 **253/253 exit 0**。perf(M8):MAIN causal 1.6× / window ~10×。真 reject:hdim>256。out-of-scope:target_in_kv、非方形 tile、独立 dO layout。

## 0b. 本会话(2026-06-15)收尾状态 + OPEN 项 —— 恢复必读
**2026-06-16 恢复 session(纯核对+文档,无代码改动)**:① 复跑回归套件 **253/253 exit 0**(`runs/test-resume-20260616.log`)= HEAD `048f0a9a` 基线干净。② 查实并消除 HANDOFF 自相矛盾:**cross-attention softmax 早已实现+测过**(旧候选条目"现仅 self"是 cross 前残留,已删去重)。③ **RED-4 查证为误报关闭**(详见下方 OPEN 块):HTML 本就对,反是 HANDOFF 旧 RED-4 描述错;已修 HANDOFF + 刷新 HTML 两处 stale 行号引用。④ **应用户要求仔细 review 总览 HTML 链接的全部 20 篇讲义**(20 个并行 reviewer agent,每条 finding 核到源码/rocm-ref/benchmark/git,带反误报铁律)→ 找到 **4 个真 RED 已全部 lead 亲核+修复**:(a) `ck-vs-hstu` §1.5 表 "HSTU softmax scale 数=2(α+内置1/√d)" 臆造→改 1(α);(b) `M5-softmax` "α 仅以 α=1 验证" 误读(attn_scale≠alpha,实际 α=0.125 已验)→更正;(c) `M7b` "CDNA4 LDS=64KB→hd256 LDS-bound" 硬件数错→**rocminfo 实测 GROUP segment=160KB**,occupancy=1 是硬编码 trait 非 LDS-bound,已改 HTML+本 INV 笔记 ②④;(d) `cross-attn` SVG caption diff_q_kv_len 公式多减 num_target→改正。其余 16 篇仅 YELLOW(多为 point-in-time stale 行号)。⑤ 应要求收尾:修 deepdive 图2 SVG viewBox 溢出(320→366);**仅刷新 floating 基础篇**(deepdive/fmha-intro 的 policy getter 行号 39/73/...→32/66/104/138/176 + codegen 行号、ck-qkv footer GridSize)的失效引用——**pinned-commit 里程碑篇(M5/M6/M6b/M7a/M7b/M7c/cross,行号锚定各自 commit、reviewer 已验对)的行号一律不动**(刷到 HEAD 反会破坏快照)。完整 reviewer 输出见本会话。工作树仍干净、HEAD 不变(HTML 不进 git)。
**本会话(2026-06-15)完成**:M7a fp16(`bf82a1d2`)→ M7b symmetric hdim(`1ae97750`)→ M7c 非对称/非典范 hdim via pad(`17515fcc`)→ cross-attention(`4629508f`)→ M8 perf MI+B2+B3(`048f0a9a`),均四方闭合。补全 HTML 讲义至全系列 + 写**总览索引** `hstu-b1052-report/hstu-bwd-overview-20260615.html`(衔接 20 篇,带 5 SVG,经优化+review)。
**文档审计**(Workflow 21-agent 审 20 篇超链接):5 RED + 36 YELLOW。已修:M3 §9②计数、M5b §5③"6→15"、M6b §2 行号、fmha-intro §3.6(fp32 标注)、ck-vs-hstu+saved-tensors 加过时横幅、总览 §三 10.4×→9.8×。全报告 `/tmp/linked-audit-report.md`。
**⚠ OPEN(下次收尾,非阻塞,均文档非代码)**:
- **RED-4 = 误报,已查证关闭(2026-06-16 lead 核 rocm-ref + dispatcher)**:HTML §5.1 "CDNA3 原生 f16 MFMA = 32×32×8 + 16×16×16 / CDNA4 新增 32×32×16 + 16×16×32" **本就正确,勿改**。审计 agent 把 `warp_gemm_dispatcher.hpp:36` 的 **fp32** 16×16×16 误当成 f16 那条,漏看 line 56 的**真 f16** 16×16×16(`v_mfma_f32_16x16x16_f16`,#else 分支=gfx9/CDNA 可用)。权威依据 rocm-ref `topics/mfma-register-layout.md:31-32`(CDNA3 f16=32×32×8+16×16×16)/46,61-62(CDNA4 K-doubled=32×32×16+16×16×32)。HANDOFF 旧 RED-4 描述"f16 真值=32×32×8/16×16×32"**本身是错的(漏了 16×16×16)**。已顺手刷新 HTML 两处 stale 行号引用(25-34→40-57、26/28/30/32→41/43/47/57)并加"fp32-vs-f16 16×16×16 勿混"提示防再误判。**教训:audit workflow 综合产物的 RED 判定也会幻觉,lead 必须回 rocm-ref/源码核事实(这次差点"修对成错")。**
- 36 个 YELLOW(老文档 stale 行号/point-in-time 快照,如 M2 把后来 P1-1 bug 的构造夸成特性、M4 记 M6b 修掉的 buggy harness)—— 多数是历史快照、可不动;详见审计报告。
- **HTML 讲义在 `/root/workspace/hstu-b1052-report/`,不进 git**(改了不需 commit)。
**git/工作树**:HEAD `048f0a9a`,ck_hstu 工作树**干净**(所有里程碑已提交);test/benchmark 在 `hstu-bwd-impl/`(非 git)。
**已推送到 fork(2026-06-25)**:remote `fork` = `https://github.com/LJ-underdog/composable_kernel.git`(ROCm/composable_kernel 的 fork,默认分支 develop);14 个提交(M0–M8+cross,HEAD `048f0a9a`)已推到 fork 分支 **`hstu_attention_fwd_bwd`**(本地同名分支已 tracking `fork/hstu_attention_fwd_bwd`)。`origin` 仍指向上游 ROCm(无写权限)。**坑**:推送用的 fine-grained PAT 必须把 `composable_kernel` 加进 repository access + Contents:write,否则 git push 403(只 fork 在 scope 里不够,要本仓库)。浏览:https://github.com/LJ-underdog/composable_kernel/tree/hstu_attention_fwd_bwd 。尚未对上游提 PR。
**下一步候选**:M8 剩余(B1 group entry TU 拆分治 14min build / B6 trload / 近似 sigmoid;占用率 B4/B7 已判低 ROI 见 M8 块)、target_in_kv、独立 dO layout、非方形 tile;或上游 fwd `5204dc75` 的 `*`/`+` bug(lead 裁定"先放着")。

## 1. 环境(关键,别重新踩坑)
- GPU:`rocminfo` → **gfx950 (MI350X, CDNA4)**;ROCm `/opt/rocm`(hipcc/amdclang++);cmake + ninja。
- 代码仓库:`/root/workspace/ck_hstu`(git 仓库,完整 CK fork);HSTU 目录 `example/ck_tile/18_hstu_attention/`。
- **构建必须 `-DBUILD_DEV=OFF`**(否则 `-Werror -Weverything` 把新 clang 诊断当错,fwd 都编不过):
  ```
  cd /root/workspace/ck_hstu
  cmake -B build -G Ninja -DCMAKE_PREFIX_PATH=/opt/rocm -DGPU_TARGETS=gfx950 -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ -DBUILD_DEV=OFF
  cmake --build build --target tile_example_hstu_attention_bwd -j$(nproc)   # bwd
  cmake --build build --target tile_example_hstu_attention     -j$(nproc)   # fwd(oracle 产 O 用)
  ```
- gfx950 专属:fwd/bwd 都走 `BUILD_HSTU_FOR_GFX95_ONLY` + `-fno-slp-vectorize` + `#ifdef __gfx950__`(CMake 自动加)。`-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3`。

## 2. 多 pane 团队(tmux session `claudeteam`)
- pane 0.0 = lead(你);0.1/0.2/0.3 = teammate。
- **派单**:Write prompt 到 `/tmp/...md` → `tmux send-keys -t claudeteam:0.N "请读取 <file> 并严格按其执行" Enter`。
- **!!! 大坑(踩过两次)**:pane **忙时 send-keys 的 Enter 会被吞**,文字只进输入缓冲不提交。**可靠流程**:发完**回读** `tmux capture-pane -p -t claudeteam:0.N` 确认输入框已清空 + 开始处理;若仍卡缓冲,等 pane **真空闲**(底部无 spinner、cost 冻结)再补 `tmux send-keys -t ... Enter`,再回读确认。
- 角色习惯:pane-1 = 主 coder(M0–M4 都它写,代码上下文最熟);pane-2 = 文档/review(写了 M2/M3/M4 代码改动文档 + review,很较真);pane-3 = 文档/review 备用。pane 重时先 `/clear` 再派(代码在磁盘,给指针即可)。

## 3. skill 与 workspace(纪律)
- skill `rocm-kernel-design`(`/root/.claude/skills/rocm-kernel-design/`,旧名 `kernel-design-rocm`,2026-06-17 改名+合并):task contract + draft 闸门 + **每候选对拍验证** + 证据 workspace。
- 实现 workspace `/root/workspace/hstu-bwd-impl/`:
  - `docs/task-contract.md`、`docs/draft.md`、本 `docs/HANDOFF.md`
  - `candidates.jsonl`(候选账本,promoted/pass/fail)、`benchmark.csv`、`runs/`(所有编译/对拍/测试日志)、`test/`(回归套件)
- **铁律**:① 验证 = 对拍 CPU reference(`reference_hstu_attention_bwd.hpp`,bf16 rel≤2e-2/abs≤5e-2);② 硬件结论查 **rocm-ref**(`/tmp/rocm-ref/`,gfx950=CDNA4 口径)不臆造;③ 解释性文档写 **HTML 图文并茂、派非 lead pane**(放 `/root/workspace/hstu-b1052-report/`);④ 对拍**必须用 `-attn_scale=1.0`**(梯度量级有意义,别让 ref 太小巧合 PASS)。

## 4. 回归测试(每步必跑)
```
python3 /root/workspace/hstu-bwd-impl/test/run_bwd_tests.py    # 整体 exit 0 = 全达预期
```
- 当前 ~47 case(pass 基线 + reject 未实现 + skip determ)。新里程碑落地后:把对应 `reject-*` 升级为 pass,并**对每个 mask 因子跑 causal∈{0,1} 交叉**(见第 7 节教训)。
- 离线 mask 校验器:`test/validate_tile_range_y.cpp`(GetTileRangeAlongY 超集,185932 checks GREEN)。

## 5. 已完成(候选账本均 promoted)
- **C0** gfx950 fwd baseline 跑通(13 TFLOPS)。
- **M0** 脚手架(bwd params/dispatch/kernel/harness/CMake target/instances)。
- **M1** SiLU MAIN 闸门:**R1(FMHA default policy 零覆写复用)+ R2(VGPR248/AGPR0/Scratch0/occ2,CDNA4 加法模型)实测关闭**。
- **M2** HSTU 5 因子 mask(causal/window/contextual/min_full/num_target + 组合);新增 mask 成员 `GetTileRangeAlongY`/`IsEdgeTile`(保守全扫)。
- **M3** jagged 变长(packed [1,ΣL,H,D] + cu_seqlens;同 kernel 运行时 is_jagged 分支)。
- **M4** group(per-group device 指针超参 `i_group=i_batch/num_batch_per_group`;**per-group window 无法编译期定 → 同时实例化 with/without-local 两 pipeline 运行时选**;alpha 全局、scale_p+mask per-group、num_target per-batch)。
- **M4b** 修 P1-1:`causal=0+num_target>0` 静默漏掩码 →STAGE2 去 `if constexpr(IsMasking)` 改运行时 `if(IsEdgeTile)`(对齐 fwd),三模式同源受益。

**能力边界**:✅ SiLU 全模式 × 全 mask × bf16 × hd64 × atomic。

## 6. 进行中(/clear 时)—— M5 已完成 promoted;下一步 M5b group softmax

### ✅ M5 softmax(no_group=batched+jagged)已 promoted(2026-06-08,git `aced5784`)
- pane-1 实现 + pane-2 独立验证(/clear 干净重建+复跑套件 60/59/0/1 exit 0+自抽 5 档对拍+对抗 review 9 条 GREEN)+ lead 亲核 STAGE2/5。三方闭合。
- 报告:`/tmp/hstu-bwd-design/M5-done.md`(coder)、`M5-review-findings.md`(reviewer)、`M5-dispatch.md`(规格)。
- 新 `hstu_attention_with_softmax_bwd_pipeline.hpp`:STAGE2 边界 `-inf` 掩(运行时 IsEdgeTile,causal=0+target 也掩)+`p=exp2(α·log2e·S − log2e·get_validated_lse(LSE))`;STAGE5 `ds=p*(dp−D)`;dq/dk*=alpha,dV 不乘。LSE/D 用 SiLU policy 已预留的 LDS region(GetSmemSize 不变)。
- PRE `hstu_bwd_dot_do_o_kernel`(D=rowsum(O⊙dO),float,[batch,head,seq] 连续-seq,batched+jagged);dispatch `RunSoftmax`=PRE→memset→MAIN→POST。
- **LSE 布局**:GPU 侧 [batch,head,seq] 连续-seq(fwd 用可配 lse stride 直接写;bwd+PRE 读;reference 读转置后的 host 副本)。GPU-bwd 与 reference 同吃一份 GPU-产 LSE。
- **已知盲区**(reviewer 指出):对拍无法独立验 LSE *数值*(两侧共用同一份 GPU LSE),靠代码审计 + 未改的 fwd 里程碑 LSE 验证兜底。
- 套件:`reject-softmax` 删,+16 M5 交叉案 → 60 案。`runs/test-20260608-065351.log`、`runs/run-M5-sweep.log`。

### ✅ M5b group softmax 已 promoted(2026-06-08,git `dc8c6b21`)
- **四方闭合**(pane-1 实现 + pane-2 代码 review:干净重建+套件 68/67/0/1+5 档 off-suite+B1–B7 GREEN + pane-3 文档级 review:7 条 group 点自推 GREEN + lead 亲核 group kernel)。
- 文档:`M5b-done.md`/`M5b-review-findings.md`/`M5b-doc-review.md`/`M5b-dispatch.md` + HTML `hstu-bwd-M5b-group-softmax-20260608.html`。
- 非阻塞观察(M5/M5b 共有,已裁决):① α 接线已在 α=0.125(=1/√64 默认)验证,"仅 α=1" 是 attn_scale(scale_p,softmax 不用)与 alpha(scale_s)的误读,非缺口;② LSE 数值盲区由 fwd 里程碑兜底 + 写读自洽闭合。
- 复用不重写:M5 `with_softmax` pipeline + PRE `dot_do_o` + POST `convert_dq` 直接用;新写 `HstuAttentionBwdDQDKDVGroupSoftmaxKernel`(M4 group 双 pipeline 骨架 × M5 LSE/D window,去 scale_p)+ group `RunSoftmax` + group harness 产 LSE。
- group params 加 `d_ptr`/`nhead_stride_lsed`;CMake bwd target 加 `group_forward_bf16.cpp`(harness 调 group fwd 产 LSE)。
- LSE/D group packed `[head,ΣL]` 连续-seq(fwd seq_stride_lse=1/nhead_stride_lse=ΣL+query_start;无 batch_stride_lse)。
- **三个禁改文件 byte-identical 于 aced5784**(M5 softmax pipeline/SiLU pipeline/no_group dispatch)= 零回归实测。套件 60 个 M1–M5 仍 PASS。

### ✅ M6 deterministic dQ 已 promoted(2026-06-09,git `c79d3296`)
- 范围:**no_group(batched+jagged)× SiLU+softmax**。机制:每 KV-block(split_idx=i_tile_n)plain-store(`memory_operation_enum::set`)到自己 split 副本(base += i_tile_n*split_stride_dq_acc)→ POST `hstu_bwd_reduce_convert_dq_kernel` 固定升序 reduce + convert → **构造上 bit-reproducible**。
- 改:kernel dq_acc 窗口编译期分叉(set+split / atomic_add);dispatch 抽 `launch_main_and_post` + kIsDeterministic 真模板轴;entry BOOL_SWITCH_3;generate_instances determ 轴(8 no_group bwd instance:4 atomic+4 determ);harness determ workspace×num_splits。**两 pipeline + group dispatch 逻辑未碰**。
- 四方闭合(pane-1 + pane-2 独立验证:干净重建+套件 77/77/0/0+**亲跑 multi-split repro byte-identical**+off-suite 对拍+**A.5 atomic-vs-determ 逐位 diff=0**+B1–B8 GREEN + pane-3 文档级 review 7 条 GREEN + lead 亲核 memory_op/split/POST)。
- 文档:`M6-done.md`/`M6-review-findings.md`/`M6-doc-review.md`/`M6-dispatch.md` + HTML `hstu-bwd-M6-deterministic-20260609.html`。
- 无害冗余(doc-review 提,非阻塞):`bp.num_splits`(harness)被设但 dispatch determ POST 用本地独算值。
- 套件:`skip-deterministic` 升级为真断言 + in-runner repro 检查 → **77/77/0/0 exit 0**;M1–M5b 60 案 atomic 零回归(dispatch 重构后)。
- O1 已在 M6b 修复(group entry BOOL_SWITCH_3 接 determ 轴)。

### ✅ M6b group deterministic + 修 O1 + 修 harness bug 已 promoted(2026-06-10,git `d4fb2884`)
- **group determ**:复用 M6 POST reduce + determ 机制;group 两 kernel dq_acc set+split 分叉、group params 加 split_stride/num_splits、group dispatch determ 分支、**group entry BOOL_SWITCH_3 修 O1**(group+determ 不再静默 atomic,真可复现)。
- **顺带挖出并修一个 pre-existing harness bug(lead M6b 验收拦截)**:harness `run_group_hstu_bwd` 的 `group_max_seqlens_q` 用组下标索引 per-batch `num_targets` + 单组 uih-max → 组内多 batch 异 seqlen 时**低估 max_seqlen_q** → PRE `dot_do_o`(grid 按 max_seqlen_q、d_dev 未 memset)漏算最长 batch 尾 token 的 D → 垃圾 D → **仅 softmax target 行 dQ 错**(dK/dV 不受;atomic+determ 都中)。**库逻辑/reference 本就正确**,纯 harness setup bug。修:公式改组内 `max_b(seqlen_q[b]+num_target[b])+ctx`(构造上恒不低估)+ `HSTU_CHECK` 守卫(喂错响亮 abort)+ PRE 前置注释(不 memset 兜底)。
- **四方闭合 + lead 拦截**:pane-1 首轮**过度声称**(带 1 FAIL 标 promoted,已纠正,接受批评)→ lead 亲跑原 FAIL 配置 0.0626→**7.4e-9 PASS** → pane-2 **对抗 formula-revert** 证 2 个 softmax 回归案真锁洞(改回旧式即 FAIL)、silu 案空挡(已改诚实 note)、零改库 byte-identical。
- 套件升级:加 `pass-gtrig-{sm-atomic,sm-determ,silu-atomic}` → **TOTAL 91/PASS 91/FAIL 0/SKIP 0 exit 0**;M0–M6 零回归。
- 文档:`M6b-done.md`/`M6b-review-findings.md`/`M6b-fix-review-findings.md`/`M6b-dispatch.md`/`M6b-fix-approval.md`。
- **教训(又一条 P1-1 式覆盖洞)**:group 测试矩阵此前没覆盖"同组多 batch 异 seqlen + 长 batch 大 target + window";"repro 全绿 + 自述全 PASS" 仍漏了 correctness 一格 FAIL,**独立复核(尤其对抗 formula-revert 验回归案有效性)是关键**。

### ℹ️ 上游 fwd group_max_seqlens_q —— 调查后结论:**非 bug,不报上游(2026-06-10 与作者核实后定性)**
- 我们曾把 `example_hstu_attention_fwd.cpp:851` 的 `group_max_seqlens_q = group_max_uih + ctx + num_targets[i_grp]` 当成"低估 bug",并实测到触发配置(`-g=2 -seqlens=100 -targets=0,0,0,200`,不传 `-g_max_seqlens`)→ O/LSE 大面积错。
- **但作者(fwd 代码作者)澄清 + 复核确认:这不是 bug,是 example 的用法约定**:`-seqlens`=uih;`-g_max_seqlens` 文档明确"can be ignored, **or else bigger**",即**调用方/脚本负责保证 `max_seqlen_q ≥ 每 batch physical seqlen`**(Meta dump 的真实工作流里作者按 physical 反算 uih 并 over-provision `-g_max_seqlens`,"更大不影响 accuracy")。默认推导只是 best-effort;组内 target 异构时传足够大的 `-g_max_seqlens` 即正确(传 300 实测全对)。**kernel 正确,正确用法下无误。**
- 结论:**不报上游**。相关草稿/讲义已作废、勿采用:`/tmp/hstu-bwd-design/upstream-issue-draft.md`、`upstream-fwd-bug-verify.md`、HTML `hstu-fwd-group-maxseqlen-bug-20260610.html`(措辞过度定性为 bug)。
- 教训(给 lead 自己):对"上游/他人代码"下 bug 定性前,先回到代码 + 找作者/owner 核实用法约定,别凭对拍 FAIL 就判 bug——这次差点误报。

### ✅ M7a fp16 dtype 加宽已 promoted(2026-06-11,git `bf82a1d2`)
- **来源**:上轮某 session 留下未提交、未测试的 M7a fp16 WIP(纯 dtype 加宽,hd64,fp16 复用 bf16 同模板码路);本轮 lead 恢复后**先 stash 立干净 M6b 边界 → 评估通过 → 复用而非重写**。
- **机制**:dispatch/kernel/pipeline 本就模板化于 `InOutDataType`,fp16 仅在边界加 dtype。改:`example_hstu_attention_bwd.cpp` 运行时 `-prec` 选 fp16_t/bf16_t(no_group+group fwd/bwd wiring)+ `get_bwd_elimit<fp16_t>` rtol5e-3/atol1e-2(**比 bf16 紧,fp16 尾数 10bit**);新 no_group/group fp16 bwd entry(bf16 逐字镜像);`generate_instances.py` dtype 轴 → 8 batched fp16 instance + ref.hpp;`api.hpp` 2 fp16 extern;CMake glob fp16 fwd/bwd entry+instance 入 bwd target。
- **四方闭合**(coder + reviewer 独立 build_review + WIP 本体 + lead 亲核):构建 0 error;fp16 sweep **66/66 PASS**;套件升级 **106/106 exit 0**(reject-fp16 → 14 fp16 pass + 2 fp16 determ byte-identical repro);reviewer 7/7 GREEN(含**容差 revert 双实验** + fp16-ULP 量级反证确跑 fp16 + 9 库文件 byte-identical);**lead 亲核**:库 byte-identical 零回归、M6b 老洞触发配置 fp16 PASS、fp16 determ dQ 两次 byte-identical。
- 文档:`M7a-done.md`(coder)、`M7a-review-findings.md`(reviewer)、派单 `M7a-eval-dispatch-{coder,reviewer}.md`。
- **非阻塞观察**:CMake 过度链接整套 fwd fp16 matrix(maxk 96/128/256 在 hd64 是死重)→ 构建提速可瘦身(M7b 顺带处理)。
- **库/kernel/pipeline/dispatch/reference byte-identical 于 d4fb2884**(`git diff` 仅 4 文件 + fp16 新文件)= 零回归实证。

### ✅ M7b symmetric hdim{64,96,128,256} 已 promoted(2026-06-11,git `1ae97750`)
- **范围**:解禁 symmetric hdim ∈ {64,96,128,256}(hdim_qk==hdim_v)。**hdim_qk≠hdim_v + 非典范任意 hdim(pad 路)= M7c**,已 guard 显式 throw 挡住。
- **核心洞察(draft 抓到)**:pre-M7b dispatch **硬编码 hd64 tile shape,MaxK 穿透但没用来选 shape** → 直接加 headdim 轴会静默复用 hd64 tile = silent-wrong。修法:新 `hstu_attention_bwd_shape.hpp` 把 TileFmhaBwdShape 做成 `HstuBwdShape<MaxK>` 编译期函数(蓝本=FMHA bwd codegen gfx9 非 trload tile;**<64> 与原硬编码逐字等价=hd64 零回归基石**)。
- **改面**:新 shape.hpp;2 dispatch 取 selector + 解 3 处 hd64 throw 换典范值 guard(`hdim_qk!=hdim_v||hdim_qk!=MaxK`);4 entry HDIM_SWITCH;generate_instances headdim 轴 [64,96,128,256](48 新 instance,hd64 maxk_64 byte-identical);**harness `kN0_bwd=(hdim==256)?64:128`**(hd256 tile bn0=64,修 determ workspace num_splits 失配)。**未动 reference/promoted pipeline/kernel 逻辑(git diff 空)**。
- **分阶段 + stage1 硬检查点**:stage1=纯重构(selector+guard+kN0,不加新 hdim)先证零回归 → lead 亲核(selector<64> 同型/hd64 instance byte-identical/106 套件)→ 放行 stage2 加 hdim。
- **四方闭合**(coder + reviewer 独立 rm-rf build_review + lead 亲核 + 二进制符号反证):sweep **128/128 PASS**(hdim{64,96,128,256}×{bf16,fp16}×模式×P1-1×determ);hd256 rocprofv3 **Scratch=0 无 spill**(VGPR172-184;occupancy=1 LDS-bound=M8 perf);套件 **171/171 exit 0**(reject-hdim128→pass + 2 guard reject〔非典范 hdim=100、asymmetric 64/128〕+ 60 M7b pass + 4 determ repro,12 repro byte-identical);reviewer **二进制符号反证**(LDS 32/48/64KB 多档 → tile 确随 hdim 分化,非静默复用)；lead 亲核套件 171/171 自跑 + hd64 byte-identical + guard throw + hd256 PASS。
- 文档:`M7b-done.md`/`M7b-review-findings.md`/`M7b-stage1-checkpoint.md`/`M7b-draft.md` + 派单。HTML 讲义:`hstu-bwd-M7b-hdim-20260611.html`(**写作中**)。

### ✅ M7c asymmetric + 非典范 hdim(head-dim padding)已 promoted(2026-06-15,git `17515fcc`)
- **范围**:接受 `hdim_qk≠hdim_v` + 非典范 hdim(48/80/100/192/200…),办法 = **激活已接线但死的 head-dim pad 机制**(运行时 `BOOL_SWITCH_2` 镜像 fwd),**不新增 instance**。真 reject:hdim>256。
- **核心洞察**:pre-M7c bwd 的 pad 路结构上已完整(每个 DRAM view/LDS window/epilogue 都 honor kPadHeadDimQ/V),只是被喂 `constexpr 0` + guard-throw 挡着。M7c 把它打开。
- **设计经 workflow**:M7c 设计用 ultracode 设计 workflow(6 路并行分析 + 综合 + 完整性 critique)产出 draft → lead 据 critique 5 must-fix 修正(尤其纠正综合 agent 幻觉的测试驱动路径)→ 闸门通过 → 实现走 pane(混合方案:设计用 workflow 广度,实现用 pane 有状态)。draft:`docs/draft-M7c.md`。
- **改面(4 文件,pipeline/kernel/reference byte-identical)**:`hstu_attention_bwd_shape.hpp`(每 MaxK kQKHeaddim/kVHeaddim/kN0)、`batched`+`group` dispatch(pad NTTP + 取模派生 + BOOL_SWITCH_2 + guard 放松 `>MaxK`;group `ProblemFor` 弃 `<0,0>` 改 pad NTTP)、harness `-poison_pad`(over-alloc MaxK + 输入尾 NaN 证 load-zero + 输出尾预填 NaN 证 store-skip + reference 喂真实 hdim;poison off=byte-identical)。
- **分阶段 + 硬检查点**:Stage0 基线→Stage1 refactor(★checkpoint:294/294 byte-identical + 172 套件)→Stage2 batched poison→Stage3 group→Stage4 收尾;每 stage lead 亲核放行。
- **四方闭合**(coder 4-stage + reviewer 独立 build_review 9/9 GREEN + lead 逐 stage 亲核 + **2 条 reverse-proof**):canonical pad=0 设备符号 **294/294 byte-identical**;batched poison **168/168** + group poison **96/96**(全 store-skip=PASS,**poison NaN 正向硬证 OOB 归零/跳写**,非 bf16 容差);套件 **220/220 exit 0**(50 个 poison-asserted 案,真 reject hdim>256 保留)。reviewer **reverse-proof 判伪**:强制 pad=false→NaN FAIL、强制 dram-view 谓词 false→store-skip FAIL,证检查非 vacuous。**dq_acc store-skip 代码核实 production 安全**(`bwd_kernel.hpp:373-381/1182-1187` `sequence<false,(kPadHeadDimQ>0)>` + mop,同 dK/dV 谓词)。
- 文档:`M7c-done.md`/`M7c-review-findings.md`/`M7c-stage{1,2,3}-*.md`/`draft-M7c.md`。HTML 讲义:`hstu-bwd-M7c-hdim-pad-*.html`(**写作中**)。
- 新工具:`test/co_symbols.py`(gfx950 设备符号 byte-identity 校验,milestone 零回归利器)。
- **教训**:① reviewer 做 reverse-proof 改源后误 `git checkout` 带未提交改动的文件 → 退回 + 逐 hunk 重建(N3),coder 已确认 verbatim 一致。**改有未提交改动的文件做实验前应先备份/用独立副本,别 git checkout。** ② 综合 agent 会幻觉路径/行号(把测试驱动写成不存在的 `scripts/run_bwd_tests.py`),critique + lead 权威知识纠正——**workflow 综合产物必须 lead 核事实再用**。

### ✅ cross-attention(seqlen_q != seqlen_kv)已 promoted(2026-06-15,git `4629508f`)
- **范围**:bwd 正确处理 q/kv 独立序列(seqlen_q≠seqlen_kv),**全方向**(含 jagged/group 的 kv>q)× 全模式 × SiLU/softmax × causal{0,1} × 5-mask × atomic/determ × bf16/fp16。**target_in_kv=false**(targets 只 Q 侧,out-of-scope)。
- **机制**:镜像 fwd —— 钉死的 `HstuBlockMasking<false /*cross*/>` 改运行时 `BOOL_SWITCH(param.is_cross_attention)` 选 `kIsCrossAttention` 轴 + kernel 4 处 mask 构造 `if constexpr` 分叉到 `make_hstu_cross_attention_block_mask_*`(`seqlen_kv`→`seqlen_k` 槽)。**cross 是运行时 switch,非 instance 轴(零 instance 增长)**。
- **两大利好**:① **reference oracle 本就 cross-ready,一行未改**;② cross 运行时 switch 零 instance 增长。
- **设计经 workflow**(6 路+critique)→ lead 闸门(critique 抓到真问题:dispatch grid 按 max_seqlen_q 开、无 max_seqlen_kv 字段 → kv>q silent-wrong)→ **裁决 Option B 做全(加 max_seqlen_kv 字段)** → 实现走 pane 3-stage。draft:`docs/draft-cross-attn.md`。
- **改面(5 文件,reference + 两 pipeline byte-identical)**:params +`max_seqlen_kv`(**纯 host 字段,device MakeKargs 不读 → 设备码不变**)、batched/group dispatch(mask BOOL_SWITCH + jagged/group grid+num_splits 按 max_seqlen_kv for cross)、kernel(4 if constexpr cross 分叉)、harness(`-seqlens_kv` + 独立 kv offsets + determ grid by max_seqlen_kv + reference cross 调用)。
- **四方闭合**(coder 3-stage + reviewer **3-binary 独立**〔build_review/build_m7c/build_r1〕+ lead 亲核 + R1 reverse-proof):self 零回归 **co_symbols 870/870 byte-identical**(reviewer 自产 M7c 基线)、self 套件 220 不动;cross sweep **32/32**(双向×全模式×P1-1×determ kv>q multi-block×fp16,容差未松);套件 **253/253 exit 0**(220 self + 33 cross,14 bit-repro)。**R1 reverse-proof**:篡改 cross mask 回 self → cross 案 FAIL(err 4.70 vs 1.95e-3)= mask switch load-bearing。**R4**:kv=512>q=128 multi-block PASS + byte-identical repro。
- 文档:`cross-attn-done.md`/`cross-review-findings.md`/`cross-stage{A,B}-checkpoint.md`/`draft-cross-attn.md`。HTML 讲义:`hstu-bwd-cross-attention-*.html`(**写作中**)。
- **教训**:① reviewer 发现 **co_symbols 基线对象集漏了 batched 的 384 个 `kentry` launch-wrapper 符号**(coder 记 486、完整应 870)→ **co_symbols dump 对象集要含 kentry wrapper**,否则 byte-identity gate 漏覆盖一格(本次 reviewer 870/870 补齐)。② reviewer 做 R1 篡改实验用**独立 worktree + cp**(不 git checkout 带改动文件,守 M7c N3 教训)。

### ✅ M8 perf(MI + B2 + B3)已 promoted(2026-06-15,git `048f0a9a`)
- **scope(lead 闸门裁决)= MI + B2 + B3 only**(runtime 真赢);占用率类(B4/B7)暂缓。设计经 scoping workflow(6 路含 rocprofv3 实测 + critique)→ critique **证伪 scoping 的 "grid starvation 根因"**(实测 256x 超额订阅、非饥饿)→ lead 裁决聚焦 GetTileRangeAlongY 浪费。draft:`docs/draft-M8-perf.md`(顶部闸门头)。
- **profiling 实测**(rocprofv3):MAIN dqdkdv 主导 **84–90%** wall-time、**矩阵核闲 ~90%**(MfmaUtil 9.9%、occupancy 10.6%)、**非 memory-bound**(MemUnitStalled 0.024%)。瓶颈 = 浪费的 MAIN q-tile 迭代(GetTileRangeAlongY 保守全扫)。
- **MI 测量基线**(behind `-perf`,device 码不变):新 `hstu_attention_bwd_perf.hpp` `time_op`(measure=false=裸 launch、perf 纯 host 字段不进 MakeKargs)+ hipEvent envelope/per-kernel + 5-GEMM TFLOPS(GEMM-only tracking 非 roofline)+ `benchmark.csv` 10 列 schema + `test/run_perf_baseline.py`。计时与 rocprofv3 互证。
- **B2 causal 紧致化**(NoLocal self+cross):MAIN **1.25–1.60×**。**B3 window/local 紧致化**(WithLocal):MAIN **4.7–9.8×**(窄窗最高,window16 实测 10.4×)。诚实 Amdahl 归因(实测<模型:只砍 q-loop、K/V load+atomic 写+启动开销不减)。
- **★ 离线穷举校验器 `validate_tile_range_y` 在 B3 抓到并修 2 个真 under-tighten silent-wrong**(非causal min_full 行、cross causal 大 diff+contextual)→ 修后 1,973,278 checks GREEN。**这是离线 gate 比对拍更早更硬挡 silent-wrong 的硬价值。**
- **四方闭合**(coder 3-candidate + reviewer **独立 2-build + 校验器 reverse-proof** + lead 亲核 + validator):MI 设备符号级 byte-identical(FORWARD 9216/9216、no_causal-NoLocal 256/256、mask-无关 helper 6/6——helper 走 time_op 包裹仍逐位不变=MI 不碰设备码铁证);校验器 reverse-proof(破坏收紧→校验器 FAIL=非 vacuous);套件 253/253 + 2 bug 配置 PASS;加速分子精确复现;co_symbols surgical(1280 DIFF 全落 mask kernel、0 MISSING);reference/pipeline/kernel byte-identical。
- 文档:`M8-done.md`/`M8-review-findings.md`/`M8-{MI-stage1,B2}-done.md`/`draft-M8-perf.md`。HTML 讲义待写。
- **暂缓(scope 外)**:B4 grid widening(根因证伪)、B1 group TU split(build-axis,14min 单 TU 仍是 build 瓶颈)、B5 first-split、B6 trload、B8/B9/B10。
- **✅ INV investigation 已完成(2026-06-15,纯 profiling 未改码)**:① **VGPR 124-vs-248 解决 = 真值 248**(rocprofv3 v1.3.0 报半值 124=248/2 单位假象;descriptor+编译器都 248)。② occupancy 限制器 = **VGPR 单卡 2 blocks/CU**(512/248=2.06)。**[更正 2026-06-16:CDNA4/gfx950 每 CU LDS = 160 KB(rocminfo GROUP segment 实测,非 64KB——64KB 是 CDNA3)**,hd64 的 32KB/WG → LDS 容 5 blocks,故 LDS **不**参与限制,VGPR 是唯一约束;早先"32KB÷64KB 共同卡"用错了 CDNA3 的 64KB]。小 config grid-limited、大 config 触天花板。③ **关键:MAIN 是 VALU/SFU-bound(VALUBusy 41% >> MfmaUtil 18%),非 MFMA-bound 非纯 occupancy-bound** → 矩阵核闲 90% 是超越运算+依赖链所致。④ **B7(占用率)判定不值得**:2→3 block 需砍 VGPR≤170(5-GEMM 极难无 spill=tile 重设计);**[更正:LDS 不是 3-block 障碍——160KB/CU 下 3 blocks 只需 LDS≤53KB,32KB 已满足;早先"LDS≤21KB"基于错的 64KB]**。天花板收益有限,只剩 VGPR 这道极难的坎。⑤ **SiLU 26% 异常根因 = sigmoid 2 个超越运算(exp+rcp)vs softmax 1 个(exp2)**,`sig` 已 CSE 非 bug,无便宜快赢(需近似 sigmoid 精度权衡)。证据 `profile/M8-INV-*`、`profile/pmc_*`。findings `/tmp/hstu-bwd-design/M8-INV-findings.md`。
- **战略结论**:高价值 runtime 浪费(GetTileRangeAlongY)已被 B2/B3 收割;占用率非瓶颈(MAIN VALU-bound)、B7 低 ROI;SiLU 本质开销。**剩余 perf 仅 B1(安全 build win)或 B6 trload(深、不确定)或近似 sigmoid(精度权衡)。**
- **教训**:① 离线穷举 superset 校验器是收紧类优化的 silent-wrong 硬 gate(抓到对拍可能漏的 2 bug);② reviewer 别同时跑 >1 个 -j128 全量 build(503GB RAM 也 OOM,maxk_256 TU 单个吃几 GB);③ perf 模型 [derived] 数会高估(Amdahl),实测多少记多少。

### git 里程碑链
`418e36ec`(M0–M4b)→ `4bfb8e08`(merge fwd LSE)→ `b0c08cba`(集成)→ `aced5784`(M5)→ `dc8c6b21`(M5b)→ `c79d3296`(M6)→ `d4fb2884`(M6b)→ `bf82a1d2`(M7a fp16)→ `1ae97750`(M7b symmetric hdim)→ `17515fcc`(M7c asym/非典范 hdim via pad)→ `4629508f`(cross-attention)→ `048f0a9a`(M8 perf MI+B2+B3)。

### 能力边界(现)
SiLU + softmax **全模式(batched/jagged/group)** × **self + cross-attention(seqlen_q≠seqlen_kv 全方向)** × 全 5 因子 mask × causal{0,1} × **bf16 + fp16** × **hdim_qk/hdim_v ∈ (0,256] 任意(对称+非对称+非典范 via pad)**;dQ **atomic + deterministic(全模式)**。真 reject:hdim>256。out-of-scope:target_in_kv、非方形 tile。

### 下一步候选(panes 空闲)——已去重 2026-06-16(基线复跑 253/253 exit 0)
> **注**:cross-attention softmax **早已实现并测过**(下面旧条目"cross-softmax 现仅 self"是 cross 里程碑之前的残留,已删)。lead 2026-06-16 核实:batched dispatch `RunSoftmax` 被 `BOOL_SWITCH(is_cross_attention)` 包(`:408-414`)、group 同(`:229/292`),套件有 cross+softmax 实测案(`j-*-sm-*`/`g2-*-sm-*`/`g2-het`/`b-qgt-sm-c1`/`g2-qgt-sm-c1-fp16`)。功能上**无此缺口**。
- **M8 perf 残项(均低 ROI,INV 已判,见 M8 块)**:① **B1 group entry TU 14min**(cross 后 {local,nolocal}×{cross,self} 4 腿)拆 per-hdim/per-mode instance = 唯一安全 build win;② B6 trload(深、不确定);③ 近似 sigmoid(精度权衡,SiLU 26% 本质开销根因);④ pad-true align-1 标量 load;⑤ bwd harness 计时;⑥ co_symbols 基线补 kentry wrapper(已在 cross 教训记)。占用率类 B4/B7 已证伪/低 ROI。
- **out-of-scope(需用户拍板才做)**:target_in_kv(cross targets 在 KV 侧)、独立 dO layout(cross R7)、非方形 tile(bhdq≠bhdv)。
- 上游 fwd group_max_seqlens_q —— **已定性非 bug,不上报**(见 §6 ℹ️ 块,与作者核实过)。
- (可选,doc-reviewer 提)α≠1 补测 —— lead 判定非必需(现 α=1/sqrt(64)=0.125 已在所有对拍验证 α 接线)。
- M6/M7/M8 各路彼此耦合弱 → 真要并行可配 git worktree(独立 build 目录)。

---
（以下为 M5 之前的历史记录,保留)
- **交叉矩阵补充**(M4b-cross)已完成、lead 亲验:`run_bwd_tests.py` = 46/45-pass/0-fail/1-skip exit 0(`runs/test-20260608-053548.log`)。M4b-fix-causal0-target promoted。
- **2026-06-08:合并上游 fwd kStoreLSE + 开工 M5**:
  - 上游 `origin/hstu_attention_fwd` 已实现 fwd 存 LSE(`is_training`+`lse_ptr`+`*_stride_lse`,自然对数 `m+log(l)`)。已 merge 进本分支。
  - **git 里程碑**:`418e36ec` = M0–M4b SiLU bwd(合并前快照);`4bfb8e08` = merge upstream fwd LSE;`b0c08cba` = 合并后集成适配(harness util include→host_util、SiLU 设 is_training=false、重生成 instance 带 lse 轴)。**bwd target 合并后编译 0 error**。
  - 集成解冲突:6 个 fwd dispatch 的 `kStoreLSE` stub 取上游版;`generate_instances.py`/`api.hpp`/`block_masking.hpp` 自动合并保住 bwd 改动。util 拆成 `hstu_attention_host_util.hpp`。FwdParams 去了 splitkv 字段(我们没用)。
  - **M5 已派 pane-1**(范围 = no_group:batched+jagged;group softmax = M5b 后续)。派单全文 `/tmp/hstu-bwd-design/M5-dispatch.md`(含 FMHA 蓝本行号、STAGE2 `-inf`掩+`p=exp2(α·log2e·S−log2e·LSE)`、STAGE5 `ds=p*(dp−D)`、PRE kernel D=rowsum(O⊙dO)、LSE/D 用 `[batch,head,seq]`连续布局、dispatch 去 throw、对拍清单)。
  - pane-1 第一步:复跑 `run_bwd_tests.py` 确认基线零回归 → 再实现 M5。恢复时:`cat /tmp/hstu-bwd-design/M5-dispatch.md` + `/tmp/hstu-bwd-design/M5-done.md`(若已写)+ 读 pane-1 末尾 + `python3 test/run_bwd_tests.py`。
  - **M5 关键风险**(lead 已在派单标注,验收重点):① LSE 布局 fwd 产出 vs reference 期望必须对齐(否则 silent-wrong);② softmax 用 `-inf` 掩(SiLU 是置 0,别抄错);③ `get_validated_lse` 防 LSE=-inf→NaN;④ dV 不乘 alpha。

## 7. 关键教训 / 决策(别重蹈)
- **测试覆盖洞**:bug P1-1 漏测因为矩阵只测"对角线"(causal=1 配因子 / causal=0 不配因子),从没交叉 `causal=0×num_target`。→ **新特性要 causal×因子 交叉覆盖;bwd 测试配置应从 fwd 支持的配置派生**。
- **silent-wrong 比 throw 危险**:未支持组合若不 throw 会静默算错。要么对拍覆盖,要么显式 throw。
- **edge-case host 数组越界**(M2 抓过):`num_targets` 等 per-batch/per-group 数组必须 supplement 到正确长度,否则 `[i_batch]` 越界 → 静默错。
- scale 接线(防双 scale bug):`alpha`(STAGE2 头 + dQ/dK 收尾,**dV 不乘**)+ `scale_p`(折进 p/g)。softmax 路(M5)用 `exp(S−LSE)` 别再乘 scale,且需 `get_validated_lse`(LSE=−inf→0)防 NaN。
- masked-out:SiLU 必须**显式置 0**(`dsilu(0)=0.5≠0`,禁 -inf)。

## 8. 后续要做(按优先级)
1. **核验交叉矩阵补充**(pane-1 在跑)→ 跑 `run_bwd_tests.py` 全绿则把它 promote;有新 FAIL 则处置。
2. **给用户 review**:`hstu-bwd-M4-changes-20260608.html`(M4 文档,已写好)+ 本轮 P1-1 修复 + 交叉矩阵。用户在等 M4 文档的 review。
3. **M5 softmax 路**:需 PRE kernel(`D=rowsum(O⊙dO)`,复用 FMHA `block_fmha_bwd_dot_do_o`)+ fwd 存/读 LSE(kStoreLSE)+ STAGE5 `ds=p*(dp−D)` + `get_validated_lse` NaN 守卫;harness 现 SiLU 跳过 GPU fwd,M5 要接 group/no_group GPU fwd 产 O+LSE。dispatch 现 softmax→throw。
4. **M6 deterministic**(dq_acc 多 split + POST reduce+convert,复用 FMHA convert_dq 的 reduce 路)。
5. **M7 fp16 + hdim{96,128,256} + hdim_qk≠hdim_v**(现 assert hd64+bf16)。
6. **M8 perf**:GetTileRangeAlongY 紧致化(现保守全扫)+ tile 级 first-split 跳过 + gfx950 trload pipeline + 砍 SiLU 未用 LDS 段 + group 双 pipeline 体积 + bwd harness 加计时。
7. **cross-attention**(现仅 self,kv offsets==q;mask cross 成员 M2 已加)。

## 9. 待办杂项(不阻塞)
- git 里 ~203 个 fwd instance 是 **M2 期遗留的未提交噪音**(generate_instances 重生成,1 行注释路径)→ 建议找时机把 bwd 工作 commit,立 milestone diff 边界。
- P4 setup 文件没拿到:`github.amd.com/DDEle/dotfiles/.../reference_p4_setup.md` 需 SSO,无凭证 → 等用户贴内容/拉到 `/tmp/p4setup.md` 再装。

## 10. 关键路径速查
- 设计(权威):`/tmp/hstu-bwd-design/DESIGN.md` + HTML `hstu-bwd-design-20260604.html`(gfx950 优先版)。
- 各里程碑报告:`/tmp/hstu-bwd-design/M{0,1,2,3,4}-done.md`、`fix-P1-1-done.md`、`M4-review-findings.md`。
- bwd 源码:`/root/workspace/ck_hstu/example/ck_tile/18_hstu_attention/hstu_attention_{bwd_params,no_softmax_bwd_pipeline,bwd_kernel,batched_backward_dispatch,group_backward_dispatch,no_group_backward_bf16,group_backward_bf16}.* + example_hstu_attention_bwd.cpp`。
- oracle:同目录 `reference_hstu_attention_bwd.hpp`。
- 报告 HTML:`/root/workspace/hstu-b1052-report/*.html`。
- rocm-ref:`/tmp/rocm-ref/`(INDEX.md 路由)。
