# M7c 实现单 —— draft 已批准(coder pane 0.1)

你刚做完 M7b(commit `1ae97750`)。下一里程碑 **M7c:asymmetric hdim_qk≠hdim_v + 非典范 hdim(via head-dim padding)**。设计稿 **`/root/workspace/hstu-bwd-impl/docs/draft-M7c.md` 已过 lead 闸门**——**先完整读它**,它是权威计划(含逐 GEMM pad 分析、改面行号、零回归策略、风险红旗 R1–R11)。本单只给执行纪律 + 硬检查点,细节以 draft 为准。全程 `-attn_scale=1.0`,**不动 reference/promoted pipeline/kernel 逻辑**。

## 核心认知(draft §1)
bwd 的 pad 机制**已接线但死**(被喂 `constexpr 0` + guard 挡)。M7c = **激活**它(运行时 BOOL_SWITCH_2 镜像 fwd),**不新增 instance**(pad 非 codegen 轴)。

## ⛔ 分阶段 + Stage1 后硬检查点(必停报 lead)

**Stage 0 — 基线抓取(改之前)**:build M7b,抓 64 batched + group instance 的 object-hash/反汇编基线(§4 byte-identity 参照)。落 `runs/M7c-stage0-baseline.txt`。

**Stage 1 — 仅 refactor(典范行为不变,先不跑任何非典范 case)**:
- batched dispatch:`Run()` 内 modulo 派生 `pad_qk/pad_v`(tile 维从 `HstuBwdShape<MaxK>` 读,V 用 `kVHeaddim` 不是 fwd 的 kN1,见 draft §3A 注)→ hoist `BOOL_SWITCH_2(pad_qk,kPadHeadDimQ, pad_v,kPadHeadDimV)` 包住 RunSilu/RunSoftmax → NTTP 穿进去 → 删 `:123-124,233-234` 的 `constexpr 0`。
- 放松 guard `:378`(softmax)/`:397`(SiLU)为 `if(hdim_qk>MaxK||hdim_v>MaxK) throw`。**⚠ R1:guard 放松必须与 pad switch 同一改动落地,别分两步**(否则非典范跑 pad=0 → OOB 静默错)。
- harness `kN0_bwd`(`:303/:771`)改按选定 MaxK(`HstuBwdShape<MaxK>::kN0`),修 hd256/桶到256 的 determ workspace 欠分配(R5)。

**★ HARD CHECKPOINT(完成 Stage1 后停,报 lead 亲验,别自行进 Stage2):**
1. 重生成的 **false-false 符号 byte-identical** 于 Stage0 基线(贴 diff 证据);
2. **171/171 + hd64/96/128/256 全 PASS 不变**(`python3 /root/workspace/hstu-bwd-impl/test/run_bwd_tests.py`);
3. grep 确认无残留 `<0,0>`/`constexpr 0`(Traits 与两 epilogue 同读 switched 常量,R4)。
报 lead:"Stage1 零回归证毕,三项证据见 X",**停**。lead 亲核放行才进 Stage2。

**Stage 2 — 激活 batched pad(true legs)**:首次编 pad!=0(潜在 static_assert R7:hd96 `<2,2,1>`、hd128 bm0=16、hd256 bn0=64,四档都要编过)。harness poison-pad 改造(**over-alloc 输入 q/k/v/o/do `:240-248` 和输出 dq/dk/dv `:255-259` 到 MaxK + stride/compare 改,见 draft §3E**)。跑 §6 batched pair + poison-pad + `128/256` determ。

**Stage 3 — group dispatch**:解 `ProblemFor` `:74` 的 `<0,0>` 写死(pad switch 内建 Problem,嵌 Local/NoLocal)。group canonical 先过 byte-identity,再 group 非典范。

**Stage 4 — 全矩阵 + sign-off**:§6 全矩阵 bf16+fp16 全模式、poison-pad on、保留真 reject(hdim>256)。

## 纪律(draft §2/§8 红旗)
- **正向证 OOB**:load-zero/store-skip 靠 **poison-pad NaN 填**硬证(别靠 bf16 容差掩),§6.1/6.4。
- **容差禁松**(bf16 2e-2/5e-2、fp16 5e-3/1e-2)。
- **每 (hdim_qk,hdim_v) 含 P1-1 cross**(causal=0+num_target),别重蹈覆盖洞。
- **hdim=100**:若 vector-load 对齐 assert 触发且 bool pad 满足不了 → **如实记为 documented reject**(诚实边界),别强松(R9)。
- **诚实**:带任何 FAIL 不标 promoted;日志数字与结论一致。

## 产出
- 每阶段日志落 `runs/M7c-*.log`;checkpoint 三证据;`profile/`(若需);新套件 TOTAL;`docs/M7c-done.md`;candidates 加行(status 据实)。
- **不 commit**(lead 闭合后统一 commit)。Stage1 checkpoint 必停。
