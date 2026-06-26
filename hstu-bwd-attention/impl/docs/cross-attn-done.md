# cross-attention (seqlen_q != seqlen_kv) — DONE (Stage A–C), 待 reviewer + lead 闭合

> 状态:**实现完成,自验全绿,等独立 reviewer 复核 + lead 闭合。未 commit。**
> 基线 HEAD = `17515fcc`(M7c)。对拍铁律 `-attn_scale=1.0`,容差禁松。
> 设计稿:`docs/draft-cross-attn.md`(lead 闸门通过)。派单:`/tmp/hstu-bwd-design/cross-{impl,stageB-release,stageC-release}-dispatch.md`。

## 0. 一句话
让 HSTU **bwd** 正确处理 **seqlen_q != seqlen_kv**(cross-attention),办法 = 镜像 fwd 的 cross 范式:把钉死的 `HstuBlockMasking<false>` 改成运行时 `BOOL_SWITCH(is_cross_attention)` 选的 `kIsCrossAttention` 轴,kernel 4 处 mask 构造 `if constexpr` 分叉到 `make_hstu_cross_attention_block_mask_*`(`seqlen_kv` 喂 `seqlen_k` 槽)。**cross 是运行时 switch,不是 instance 轴**(零 instance 增长)。reference 本就 cross-ready,一行未改。

## 1. 改面(6 源文件 + 2 test 文件;reference / pipeline 零改)
- `hstu_attention_bwd_params.hpp`:两个 bwd params 各 +1 字段 `max_seqlen_kv`(纯 host,device MakeKargs 不读)。
- `hstu_attention_batched_backward_dispatch.hpp`:mask typedef 外包 `BOOL_SWITCH(param.is_cross_attention, kIsCrossAttention)`(嵌 use_local switch 内,不提到 pad/local 之上);`launch_main_and_post` jagged grid/num_splits 用 `max_seqlen_kv`。
- `hstu_attention_group_backward_dispatch.hpp`:RunSilu + RunSoftmax 各包 `BOOL_SWITCH(is_cross_attention)`;grid/num_splits 用 `max_seqlen_kv`。
- `hstu_attention_bwd_kernel.hpp`:4 处 mask 构造(no_group SiLU/softmax + group SiLU/softmax)加 `if constexpr(...::kIsCrossAttention)` cross builder(`seqlen_kv`→`seqlen_k`)else self 逐字。
- `example_hstu_attention_bwd.cpp`:CLI `-seqlens_kv`/`-max_seqlen_kv`/`-g_max_seqlens_kv`;解钉 `is_cross_attention`;独立 `seq_offsets_kv` 向量 + `seq_offsets_kv_dev` 设备 buffer;**target_in_kv==false**(cross KV = kv_uih+contextual,无 targets);determ grid/workspace 用 `max_seqlen_kv`;reference 喂独立 kv offsets/max。镜像 fwd harness(`example_hstu_attention_fwd.cpp:262-389`)。
- `test/sweep_cross.py`(新,32 案全矩阵对拍)、`test/run_bwd_tests.py`(+31 cross MATRIX +2 cross determ-repro)。

## 2. 分阶段 + 硬检查点(每阶段 lead 亲核放行)
**Stage A — 零回归重构**(纯 false 腿等价,is_cross_attention 仍全 false):dispatch 5 处 `<false>`→BOOL_SWITCH + kernel 4 处 if constexpr。
- co_symbols verify vs M7c HEAD 净构建基线(66 obj / **486 设备符号**):**486/486 byte-identical,0 DIFF,0 MISSING**(+~256 新 cross 符号 allowed)。**[reviewer 更正]** 该基线对象集漏了 batched 的 384 个 `kentry` launch-wrapper 符号(真实完整数=870);reviewer 自产 870-符号基线复核 **870/870 byte-identical**,零回归结论更强。follow-up(非阻塞):co_symbols dump 对象集补 kentry wrapper→ if constexpr 无泄漏。
- self 套件 **220/220 exit 0**。编译预算:group entry obj scratch=0、max VGPR 426<512;group bf16 entry 单 TU 14m9s = build 关键路径(R11,per-hdim 拆 TU 留 M8)。

**Stage B — harness 解钉 + max_seqlen_kv + dispatch grid**:
- build 0 error;co_symbols 仍 **486/486 byte-identical**(加字段/改 grid 是 host 码,device 不变);self 套件 **220/220**。
- cross smoke 双向 PASS(q<kv、q>kv grid-shrink、q<kv determ multi-block kv=512 R4 路)。

**Stage C — cross 对拍全矩阵 + 套件翻转**:
- **单点 ctor 逐字对齐(R2/R3)**:kernel `make_hstu_cross_attention_block_mask_with_local(true, seqlen_q, seqlen_kv, contextual, num_target, max_attn_len, eff_min_full)` 与 reference 调用点逐字一致(参序 + with-local 包装器 num_target 末位重排,均走 wrapper 不直调 ctor);without_local 同。
- **§6 全矩阵对拍 `test/sweep_cross.py`:32/32 PASS**(双向 × {no_group jagged, group, batched-uniform} × SiLU/softmax × causal{0,1} × P1-1〔Q-side target / contextual≤min(q,kv) / local / minfull〕× atomic/determ + 非整除 + determ kv>q multi-KV-block〔R4〕+ fp16),`-attn_scale=1.0`,**容差未松**。日志 `runs/run-cross-sweep.log`。
- **套件永久化**:`run_bwd_tests.py` +31 cross MATRIX(expect=pass)+2 cross determ-repro → **TOTAL 253 PASSED 253 FAILED 0 SKIPPED 0 exit 0**(self 220 不动:208 MATRIX + 12 repro;cross 33:31 MATRIX + 2 repro)。日志 `runs/cross-stageC-suite.log`。
- self co_symbols 仍 **486/486 byte-identical**(Stage C 仅改 test 文件,binary 未变)。

## 3. 抓陷阱证据(draft §2/§8 红旗)
- **R1**(mask 钉死 self,头号 silent-wrong):双向 seqlen_q!=seqlen_kv 显式测,32 案全过。
- **R2/R3**(ctor 参序 / with-local num_target 重排):只走 `make_hstu_cross_attention_block_mask_*` wrapper;对拍逐字对齐验证 PASS。
- **R4**(determ grid/num_splits 用 max_seqlen_kv):`j/g2/b-determ ... kv=512>q=128 multi-block` 对拍 PASS + 2 repro byte-identical。
- **R8**(scale_p 分母 = 1/max_seqlen_q):cross 仍用 Q-side 分母,softmax 案对拍 PASS。
- causal=1 双向过 → `diff_q_kv_len` 对齐正确。

## 4. 诚实限制(out-of-scope / 已知风险)
- **target_in_kv == false**:targets 只在 Q 侧;targets-in-KV 是结构性新路,**不做**(`hstu_block_masking.hpp:53,:566` 硬假设)。测试均 targets 留 Q 侧。
- **独立 dO layout 未做(R7)**:测试 dO 与 O 同 layout 规避;PRE 用 O stride 读 dO,独立 dO layout 留后续。
- **group entry 14min TU(R11)**:cross 子腿令 group entry = `{local,nolocal}×{cross,self}` 4 腿,单 TU 14m9s = build 瓶颈;per-hdim 拆 TU 留 M8。
- contextual 须 ≤ min(seqlen_q, seqlen_kv)(测试守此)。

## 5. 关键路径
- 源:`/root/workspace/ck_hstu/example/ck_tile/18_hstu_attention/`(上列 6 文件)。
- 测试:`test/sweep_cross.py`、`test/run_bwd_tests.py`、`test/co_symbols.py`。
- 日志:`runs/cross-stage{A,B,C}-*.log`、`runs/run-cross-sweep.log`、`runs/cross-stageA-baseline.json`。
- 各阶段检查点报告:`/tmp/hstu-bwd-design/cross-stage{A,B}-checkpoint.md` + 本文件。
