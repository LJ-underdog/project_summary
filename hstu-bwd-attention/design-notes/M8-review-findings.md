# M8 perf 对抗式 review findings — reviewer (pane 0.2)

> 独立对抗 review,默认怀疑、不信自述。基线 HEAD=`4629508f`(cross),M8 未 commit。
> 独立 build:`build_review`(MI+B2+B3,rm-rf 重配重编,BUILD_DEV=OFF gfx950,0 error)
> + **自产 HEAD 基线** `/tmp/ck_head/build_head`(git worktree @ 4629508f,clean,0 error)。
> 所有实验在隔离副本 `/tmp/m8-review/` 进行,**working tree 全程未碰**(守 N3:不 git checkout 带改动文件)。

## 结论:**可以 PROMOTE**。8 条全 GREEN,0 RED,无 silent-wrong 风险。
唯一 caveat(非阻塞,见 #5):MI-only 全扫基线未独立重建(需第 3 次全量 build);causal 1.30× 的
分母(0.263)用 coder 记录值,但已被独立 rocprofv3(266us)佐证,且**收紧后的分子被我逐档独立精确复现**。

---

## ★ 构建事故说明(我方,非代码缺陷)
首轮我同时跑 3 个全量 build(-j128×3 ≈ 1004 clang 进程)→ **OOM kill**(`clang frontend command failed: Killed`)
打在 maxk_256 forward instance 上。**纯并行度过高,与 M8 改动无关**。清进程后 **顺序 -j32 重建,build_review
+ build_head 均 0 error 干净通过**。教训记:这机器 503GB RAM,maxk_256 TU 单个吃几 GB,别 >1 全量 build 并行。

---

## 逐条对抗审查(GREEN/RED + 证据)

### 1. scope — **GREEN**
- `git diff 4629508f --name-only`(ck_hstu 仓)= 5 改 + 1 新(`hstu_attention_bwd_perf.hpp`);另 3 个在
  impl workspace(`test/validate_tile_range_y.cpp`、`test/run_perf_baseline.py`、`benchmark.csv`)= 共 9。
- **reference + 两 pipeline + kernel byte-identical 于 4629508f**(`git diff --quiet` 实测):
  `reference_hstu_attention_bwd.hpp` / `hstu_attention_{no_softmax,with_softmax}_bwd_pipeline.hpp` /
  `hstu_attention_bwd_kernel.hpp` 全部 BYTE-IDENTICAL。

### 2. ★ MI behind-flag 零回归(byte-level)— **GREEN**
- **代码级**:`grep` 确认 `perf_*`/`measure_perf` 字段**从不出现在任何 MakeKargs/kargs 构造**(host-only,
  追加在 params struct 末尾带默认值)。`time_op(measure=false)` = **只调 `fn()` 一次**,host 行为 == 裸 launch。
  dispatch 仅把每个 launch 包进 `time_op` lambda(纯 host),MakeKargs 不变。harness `-perf` 整块 gated
  在 `if(measure_perf)` 且**在校验落 host 之后**(梯度已 FromDevice)→ 套件路完全不受扰。
- **设备符号级(自产 HEAD 基线复核)**:`co_symbols dump` 我自己的 build_head(361 obj/10854 sym)→ verify
  build_review。**所有"MI 可能动但 B2/B3 不动"的路全部 byte-identical**:
  - FORWARD **9216/9216 identical, 0 DIFF, 0 MISSING**
  - BWD **no_causal NoLocal**(全注意力)**256/256 identical**
  - **mask-无关 helper kernel**(PRE `dot_do_o` / POST `convert_dq` / `reduce_convert_dq`)**6/6 identical**
    —— 这些 kernel 的 launch 现已走 `time_op` 包裹,**设备码仍逐位不变 = MI host 包裹不碰设备码的直接铁证**。
- MI = 唯一非 B2/B3 改动,且 B2/B3 只动 mask 设备码 → 上述 byte-identity 等价证明 **MI 单独 = 零设备改动**。
- 套件 **253/253**(非 -perf 路,见 #7)独立复跑通过,0 条 PERF 行污染。
- **note**:未能逐字复现 coder 的 "13782/13782" —— 我基线 10854 sym/361 obj = **bwd target 完整全集**;
  coder 457 obj 含 fwd target(多 96 个 fwd-only instance,M8 不碰、本就 0-diff)。**我的分解证明更细更强**。

### 3. ★★ 离线 superset 校验器(B2/B3 最硬 gate)— **GREEN**
- **独立编译 + 跑当前(working-tree)校验器**:`hipcc -std=c++17 -D__HIP_PLATFORM_AMD__ -I<hstu> -I<include>`
  (无需解 math redefinition,直接过)→ **checks=1,973,278  failures=0  ALL GREEN**(精确匹配 coder)。
- **校验器对生产忠实(Explore 映射核实)**:与 GPU 用**同一组 factory**(`make_hstu_*`)构造 mask、同一
  `eff_min_full = min(mf, sq-ntgt)` clamp、`is_tile_in_first_split=true`、`win` 喂进 `max_attn_len_` 槽、
  oracle = 生产同款 `IsTokenPairInsideMask`。tightening 算术对 W/ctx/ntgt **单调**,under-tighten 风险方向
  (小 window/对齐/cross diff 双向/target+minfull+contextual 在场)校验器全覆盖。
- **★ reverse-proof 判伪(隔离副本,非 git checkout)**:
  - 破坏 #1:删 self causal NoLocal 的 contextual `y_start=0` carve-out → 校验器 **FAIL**
    (`KVtile@64 attends sq=0..5 but range=[64,128)`,contextual 行被漏)。
  - 破坏 #2:删非 causal WithLocal 的 min_full `y_start` 下沉 → 校验器 **FAIL**
    (`nocausal,local ... mf=64: KVtile@64 attends sq=32.. but range=[48,128)`)。
  → 校验器**能判伪、非 vacuous**,且这正是 coder 声称它抓到的 2 个 bug 类。

### 4. B2/B3 对拍正确(under-tighten 经验验)— **GREEN**(全部 `-attn_scale=1.0`)
- 套件 **253/253**(#7)。
- **校验器抓的 2 个 bug 配置**对拍 PASS:
  - 非causal window16 + minfull64 + target(softmax & SiLU)→ numeric_pass=true。
  - cross causal kv512>q128 + ctx6 + window16(WithLocal),及 cross causal kv512>q128 + ctx6(NoLocal)→ PASS(dV err=0)。
- 额外 stress 全 PASS:**window256 大窗 seq2048**(校验器 win 仅扫到 64,故这是窗口收紧的独立运行时确认)、
  self causal contextual、self causal minfull-NoLocal、**P1-1 非causal+target NoLocal**、jagged window causal、
  jagged 非causal window+minfull+target、group g2 window+ctx causal、determ window causal、fp16 cross window+ctx。
- 容差**未松**(标准 bf16 rel2e-2/abs5e-2、fp16 5e-3/1e-2),误差量级正常(多在 1e-4~1e-2)。

### 5. 加速实证 + 诚实 — **GREEN**(+ 1 caveat)
- **收紧后 MAIN per-kernel 时间在我自己的二进制上逐档精确复现**(`-perf`):
  | config | coder benchmark.csv | 我复现 | |
  |---|---|---|---|
  | B2 causal canonical softmax | 0.2027 | **0.2028** | ✓ |
  | B3 window256 causal softmax | 0.0635 | **0.0635** | ✓ |
  | B3 window16 causal softmax | 0.0318 | **0.0318** | ✓ |
- **窗口加速在我的二进制上自证**:build_review 上 **非causal NoLocal 路仍全扫(未被 B2/B3 改)**= 0.205ms,
  可当我自测的全扫基线下界;window256=0.0635 → **3.2×**,window16=0.0318 → **6.4×**(对 WithLocal 全扫基线
  0.299 则 4.7×/9.4×)。**窗口收紧实打实大赢**。
- **诚实性**:Amdahl 归因(实测 1.25–1.60× / 4.7–9.8× < [derived] 1.9× / 22×)正确——收紧只砍 Q-tile 循环
  数,K/V load + atomic dq_acc 写流量 + 启动开销不随之减;窄窗时 MAIN 已 < PRE,envelope 被启动开销主导。
  TFLOPS 明确标注 **GEMM-only tracking 非 roofline**,memset 单独计时不混入 MAIN。容差未松。
- **caveat(非阻塞)**:我没重建 MI-only 全扫 build 来独立测 causal 全扫基线(0.263);它是 coder 记录值,
  但**已被独立 rocprofv3 佐证**(266us ≈ 0.263)。因不同 mask 类型 per-pixel 掩码开销不同(非causal NoLocal
  0.205 < causal NoLocal 0.263 < causal WithLocal 0.299,内部自洽),我的 0.205 非 causal 全扫的有效代理,
  故 causal 1.30× 的分母用记录值。**收紧后的分子已被我精确独立复现**,结论可信。

### 6. co_symbols surgical — **GREEN**
- 全 1280 个 DIFF **全部落在 mask kernel**:no_causal WithLocal 256(B3 非causal窗口+min_full下沉)、
  has_causal 512(B2 causal NoLocal + B3 causal WithLocal)、group entry WithLocal 512(B2/B3 group mask)。
- **0 MISSING**(无设备码被删)。**FORWARD 9216/9216、no_causal NoLocal 256/256、helper 6/6 全 identical**。
- note:coder B2-done 说"no_causal byte-identical"是**B2 单阶段**口径(B2 只动 NoLocal causal);叠加 B3 后
  no_causal WithLocal 合法改变(min_full 下沉本就是非causal专属修复)。**最终态与 B3 scope 完全一致,无矛盾**。

### 7. 套件 253/253 独立复跑 — **GREEN**
- `run_bwd_tests.py --bin build_review/...` → **TOTAL 253  PASSED 253  FAILED 0  SKIPPED 0  RESULT: OK**。
- 含 14 个 deterministic **byte-identical** bit-repro(repro-det/gdet/fp16/h96/h128/h256/xattn 全 PASS)。
- log:`runs/test-20260615-120406.log`。

### 8. 暂缓项诚实 — **GREEN**
- M8-done 如实标 **B4 grid widening**(scoping "grid starvation 根因" 被 critique 实测证伪 256x 超额订阅,
  依据作废)、**B7 hd256 occ/VGPR**(阻塞于 VGPR 矛盾+LDS-vs-MFMA 测量)、**B1/B5/B6/INV/B8-10** 全暂缓带理由。
- **SiLU 26% 异常(INV)**:MI 已复现(silu/softmax=1.27×,profile 335/266us),根因诚实留待后续,无夸大。
- 无对本期成果的过度声称。

---

## 证据索引(全部我方独立产出)
- build:`build_review`(MI+B2+B3)、`/tmp/ck_head/build_head`(HEAD baseline)。
- 校验器:`/tmp/m8-review/validate`(1,973,278 GREEN)、reverse-proof `validate_broken{1,2}`(均 FAIL on 预期配置)。
- co_symbols:`/tmp/m8-review/head_base.json`(自产 HEAD 基线)、`review_dump.json`、`categorize_cosym.py`。
- 对拍:套件 `runs/test-20260615-120406.log` + 上文 stress 批次(终端实录)。
- 加速:`-perf` 终端实录(收紧档逐一匹配 benchmark.csv)。
- working tree:`git status` 全程仅 6 个 M8 文件,HEAD 仍 4629508f,**未受 review 污染**。

## 给 lead 的一句话
**M8(MI+B2+B3)correctness 闭环铁证(校验器 GREEN + reverse-proof 判伪 + 套件 253/253 + bug 配置 + 大窗对拍),
零回归铁证(forward/no_causal-NoLocal/helper 设备码 byte-identical + 0 MISSING),加速收紧分子精确复现且诚实归因。
建议 PROMOTE。唯一 caveat:MI-only 全扫基线没独立重建(rocprof 已佐证),不阻塞。**
