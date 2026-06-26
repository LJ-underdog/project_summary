# 实证上游 fwd group_max_seqlens_q 是否真 bug (pane-2)

> 目标:**实测确认或证伪**上游 `example_hstu_attention_fwd.cpp:850-851` 的 `group_max_seqlens_q` 低估是否真产出错值。**报上游之前必须有硬证据,不能凭读代码。**
> 背景:bwd 侧已确认同款公式低估 → PRE 漏算 D → dQ 错。fwd 侧假设:同样低估 → fwd grid/O/LSE under-cover 长 batch 尾 token。要证明这导致 fwd **O 或 LSE 数值错**。

## 关键:别掉进"双错相等"陷阱
bwd 的 LSE 盲区教训:若 GPU 和 reference **都吃同一个错的 max_seqlen_q**,可能双错相等而 PASS,掩盖 bug。所以必须**判别**:
- 查清 fwd reference(`reference_hstu_attention_fwd`)是否也用 group_max_seqlens_q / max_seqlen_q —— 若 ref 独立按真实 per-batch seqlen 算,则 GPU under-cover → GPU≠ref → corrpair FAIL(能检出);若 ref 也用同一低估值,则可能双错相等(检不出,需别的手段)。
- 真正的判别:**把触发配置的长 batch 单独当 no_group / 单 batch 跑**(no_group 的 max_seqlen_q 天然够),得到"正确 O/LSE 金标准",再和 group 路的 GPU O/LSE 逐元素比对那个长 batch 段。group≠no_group(在长 batch 尾 token)= 确认 bug。

## 步骤
1. 建 fwd:`cd /root/workspace/ck_hstu && cmake --build build --target tile_example_hstu_attention -j128`(0 error)。
2. 看 fwd example 的 group 参数与 `-training`(产 LSE)用法(`example_hstu_attention_fwd.cpp` arg parser);确认 fwd dispatch 的 grid 是否按 max_seqlen_q 开(grep group_forward_dispatch 的 GridSize/max_seqlen_q)。
3. **构造触发配置**(组内多 batch 异 seqlen、长 batch 带较大 target,使 group_max_seqlens_q 低估):参考 bwd 触发 `g=2, b=4, seqlens=128,200,96,160, targets=8,24,8,16, local=16,16`,翻成 fwd example 的等价参数;**务必加 -training=1 -softmax=1** 同时验 O 与 LSE。
4. 跑该配置 + 一个对照(同数据但 uniform seqlen / 单 batch per group,不触发)。看 fwd corrpair(O、LSE)是否 FAIL。
5. **判别实验**(决定性):把触发组里那个长 batch(seqlen 224)单独当 no_group(或单-batch-group)fwd 跑 → 金标准 O/LSE;与 group 路 GPU 输出的对应段逐元素 cmp。
6. (可选加强)临时把 fwd harness 的 group_max_seqlens_q 改成正确公式(组内 max_b(seq+tgt)+ctx)重跑触发配置 —— 若由 FAIL/不一致转 PASS/一致,则**因果坐实**(改完务必还原,别留改动)。

## 产出
写 `/tmp/hstu-bwd-design/upstream-fwd-bug-verify.md`:
- 结论:**确认是 bug** / **证伪(非 bug,说明为何 fwd 不受影响)** / **检不出(双错相等,需上游侧验)**。
- 硬证据:触发配置命令 + corrpair 实测数值 + 判别实验(group-GPU vs no_group-金标准)逐元素差 + (若做了)正确公式重跑结果。
- 若确认:精确根因(fwd 哪条路 under-cover:grid?LSE 写?O 写?)+ 触发条件 + 建议修法。
- 只读不改 promoted/库;临时改 harness 验因果后**必还原**(git status 干净)。发现意外立刻报 lead。
