# M6b/M5b dQ bug — lead 批复(根因已确认=harness)

判别实验干净、根因到行漂亮。结论:**库逻辑(kernel/dispatch/reference)正确,bug 在测试 harness 的 group_max_seqlens_q 低估 → PRE 漏算最长 batch 的 [max_seqlen_q, seqlen) 的 D(且 d_dev 未 memset)→ 垃圾 D → 仅 target 行 dQ 错**。批准如下:

## ① 主修 harness group_max_seqlens_q —— 批准
改成组内逐 batch 最大,与 packed offset 公式一致:
```
group_max_seqlens_q[i_grp] = group_contextual_seqlens[i_grp]
    + max_{b in group i_grp}( seq_lengths_q[b] + (num_targets? num_targets[b]:0) );
```
- 确保 harness 里**调 group fwd 产 LSE 那处**也用修正后的 max_seqlen_q(否则 fwd LSE 也会同样 under-cover)。
- 顺带修好的 SiLU group attn_scale=0 的 scale_p fallback(也用 group_max_seqlen_q)一并受益,OK。

## ② 上游 fwd example(example_hstu_attention_fwd.cpp)同款 bug —— 本里程碑**不改**
- 若你说的"fwd group harness"指**我们 bwd harness 内调 group fwd 的那次**,已被 ① 覆盖(用修正值)。
- 若指**上游 example_hstu_attention_fwd.cpp**:那是上游维护的代码,改它=scope 蔓延 + 与上游分叉风险。**本单只在 M6b-done.md 里如实记一笔"上游 fwd group harness 疑似同款 group_max_seqlens_q 低估,建议上游报"**,不动它。

## ③ 库侧防御 —— 要,但用便宜可达守卫,别过度
- **harness 加 assert**(host 有逐 batch seqlen,便宜):`assert max_max_seqlen_q >= 每个 batch 的 packed seqlen`,把"喂错 max_seqlen_q"从 silent-wrong 变成响亮失败。
- **PRE/dispatch 加注释**:文档化前置条件"max_seqlen_q 必须 ≥ 所有 batch 的 packed seqlen,否则 PRE 漏算 D"。
- 设备端 offset 的 host 校验(要拷回 offsets)成本高,**不做**——注释 + harness assert 足够。
- **不要**用 memset d_dev "兜底":真 token 的 D=0 仍是错值(只是从垃圾变确定性错),那是掩盖不是修复。别做。

## 验证(修后)
1. 该 FAIL 配置转 PASS:`-b=4 -nhead=4 -g=2 -seqlens=128,200,96,160 -g_local_lens=16,16 -targets=8,24,8,16 -softmax=1 -causal=1 -attn_scale=1.0`。
2. **把这个精确触发配置加进测试套件**(nbpg>1 + 组内异 seqlen + 长 batch 大 target + window),堵 P1-1 式覆盖洞。再加一个 SiLU 版同触发(确认 SiLU group 也被 ① 修好/本就因 minfull 没踩)。
3. 全套件零回归(M0–M6 全绿)。
4. M6b determ==atomic 仍 byte-identical(① 不影响 determ 机制)。
5. 这同时让 M5b atomic 与 M6b determ 在该配置都 PASS。

## 收尾
- 修 + 验完,写 M6b-done.md(含:根因=harness max_seqlen_q、判别实验、① 修法、③ 防御、新增套件案、零回归、M5b/M6b 一起重验结果、上游 fwd 同款 bug 备注)。
- candidates:M5b 重验后维持 promoted(库逻辑本就对,补测+harness 修)、M6b 改回 promoted。
- ctx 你已 75%,注意余量:进展随手落盘 M6b-done.md;若接近上限先落盘再告诉我,可 /clear 续做(代码在磁盘)。
- 改完报 lead,我亲核 + 派 pane-2 复核再签。
