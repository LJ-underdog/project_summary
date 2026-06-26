# M6b FAIL — lead 裁决与指令 (给 pane-1)

## 你是对的,我的假设错了
- atomic 与 determ **逐位相同**(0.0626221 一字不差)证明这**不是 determ 专属 bug**。我把「hetero 含 minfull(PASS)」与「hetero 无 minfull(FAIL)」两个不同配置误比了。你纠正得对,根因定位扎实(非首 batch target 区 dQ;dK 共用 dS 且 PASS ⇒ 仅 dQ 输出路径)。candidates 你已正确改回 in-progress。

## 裁决:方案 A(修),不签方案 B
silent-wrong 不能留在 promoted 代码里(铁律 §7「silent-wrong 比 throw 危险」)。**授权你动 M5b promoted 代码做正确性修复。**

## 步骤(先回我,等批再改代码)
1. **判别实验(先做)**:把触发配置里的 batch1 单独当 no_group jagged softmax(同 seqlen=224、window=16、target=24)跑,比对四方:
   - group-GPU-dQ(batch1 那段) vs no_group-GPU-dQ vs reference_group(batch1) vs reference_no_group。
   - 目的:判定是 **GPU group-softmax kernel** 错,还是 **reference_group** 错(这决定改哪边;reference 是 oracle,动它要极谨慎并交叉验证)。
2. **回报**:判别实验结果 + 根因到具体代码行(GPU kernel 的 target-row dQ 映射,或 reference_group 的 target-row dQ)+ 修法。**等我批准再改代码**(尤其若指向 reference)。
3. **修后验证**:
   - 该 FAIL 配置转 PASS(`-b=4 -nhead=4 -g=2 -seqlens=128,200,96,160 -g_local_lens=16,16 -targets=8,24,8,16 -softmax=1 -causal=1 -attn_scale=1.0`)。
   - 把这个**精确触发配置**(nbpg>1 + 同组异 seqlen + window>0 + num_target>0)**加进测试套件**,堵覆盖洞(P1-1 教训)。
   - 全套件零回归 + M6b determ==atomic 仍 byte-identical。
4. 此修同时修复 M5b atomic 与 M6b determ(共用 dQ 路);修完 **M5b + M6b 一起重验**,我再裁决 promote(并派 pane-2 复核)。

## 注意
- ctx 已 72%,注意余量;若接近上限先告诉我,我可让你把进展落盘后 /clear 续做。
- 不擅自改 reference;不签 B。
