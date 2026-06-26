# 实证:上游 fwd `group_max_seqlens_q` 是否真 bug (pane-2)

> 任务:报上游前**实测**确认/证伪 `example_hstu_attention_fwd.cpp:850-851` 的 `group_max_seqlens_q` 低估是否真产出错值。**不凭读代码下结论。**
> 日期 2026-06-10。fwd target 独立重建(`runs/build-fwd-pane2.log` EXIT=0,0 error)。

## 结论:**确认是真 bug(且远比 bwd 严重)**

低估的 `group_max_seqlens_q` → `max_seqlen_q` 偏小 → GPU fwd 产出 **O 87.5% 错(含 NaN/inf)、LSE 84.6% 错**;喂正确值即全对。**软证据(读代码)+ 三路独立硬证据**坐实因果,无"双错相等"。**建议向上游报。**

---

## 关键防陷阱:fwd reference 对 `max_seqlen_q` **独立**(softmax 路)
读 `reference_hstu_attention_fwd.hpp`(group 版,L348+):
- L433 `seqlen_q = seq_q_offsets[i_batch+1]-seq_q_offsets[i_batch]` —— 每 batch 按 **offset 真实 seqlen** 算 O/LSE。
- L438/L139 `max_seqlen_q=group_max_seqlens_q[i_group]` 仅用于:① `scale_p=1/max_seqlen_q`(**SiLU attn_scale=0 专属,softmax 不用**);② mask 张量维度。
→ **softmax 下 reference 的 O/LSE 与低估值无关**,故 corrpair FAIL 能真正检出 GPU under-cover(非双错相等)。
**实测坐实**:trigger 与 control 两次跑的 **`output_host.dat` / `lse_host.dat` BYTE-IDENTICAL**(`cmp` 通过)→ reference 确与 `max_seqlen_q` 无关。

## GPU 侧:`max_seqlen_q` 驱动启动配置
`hstu_attention_group_forward_dispatch.hpp`:L166 `GridSize(...,param.max_seqlen_q,...)`、L186 `get_hstu_attention_fwd_mtile(...,max_seqlen_q)`、L205 `shall_use_splitkv(...,max_seqlen_q)`。低估 → grid/mtile/splitkv 选择全偏。

---

## 硬证据

### 触发配置(softmax+training,同验 O 与 LSE)
```
-prec=bf16 -hdim_qk=64 -hdim_v=64 -nhead=4 -b=4 -g=2 -softmax=1 -training=1 -causal=1 -v=1 \
-g_context_lens=0,0 -g_local_lens=0,0 -g_minfull_lens=0,0 -g_attn_scales=1.0,1.0 \
-seqlens=100 -targets=0,0,0,200
```
组内异 seqlen、长 batch(batch3)带大 target。旧公式:`group_max_seqlens_q[g]=group_max_uih(100)+ctx(0)+num_targets[g]`,而 `num_targets[0]=num_targets[1]=0`(**组下标错取**)→ `[100,100]`,`max_seqlen_q=100`。但 batch3 真实 seqlen=uih100+tgt200=**300**。低估 200。

### 实验 1:trigger(无 override)→ **FAIL(灾难级)**
```
O  : max err = inf,   134467 errs, 87.54% wrong   (含 NaN)
LSE: max err = 5.056,   2030 errs, 84.58% wrong
```
逐 token 区间(numpy,bf16=u16<<16)显示**全 batch 皆错**(非仅 batch3 尾):
| region | O max_abs | LSE max_abs |
|---|---|---|
| b0 [0,100) | 0.965 | 2.76 |
| b1 | 0.940 | 2.59 |
| b2 | 0.951 | 2.64 |
| b3-head | **nan** | 2.68 |
| b3-tail | 0.408 | 5.06 |

### 实验 2:control = 同配置 + `-g_max_seqlens=300,300`(强制覆盖 batch3)→ **PASS**
```
O  : max err = 0.000977 (bf16 ULP), LSE: 0 errs
```
逐区间 O max_abs ≤ 0.0039、LSE max_abs = 0.0（全对）。**唯一变量是 `max_seqlen_q` 100→300** → 因果锁定。

### 实验 3:mild trigger(`-seqlens=256 -targets=0,0,0,200`,短 batch 满 tile 覆盖)→ 仍**全错**
O 64.2% wrong(max 0.925)、LSE 58.1% wrong(max 5.78);b0/b1/b2 即便完全在 grid 内仍错 → **低估腐蚀全局输出**,非局部尾 token。

### 实验 4(决定性·实际改公式):临时把 fwd harness 公式改成正确 per-batch max 重建重跑 trigger(**无 override**)→ **PASS**
正确式:`group_max_seqlens_q[g]=ctx[g]+max_{b∈g}(seq_lengths_q[b]+num_targets[b])`。
```
extreme trigger: O max err = 0.0009765625, 5558 errs, 3.61849%   <- 与实验2 override 结果 BYTE 级一致
mild    trigger: O max err = 0.001464844,  12304 errs, 3.93%      (bf16 噪声,PASS)
```
extreme 数字 `0.0009765625 / 5558 / 3.61849%` **与 override control 完全相同** → 正确公式产出的 `max_seqlen_q` 与 override 等价。**改完已还原,`git status` 干净。**

> 三路独立证据(override 翻转 / 真改公式翻转 / reference byte-identical 独立)互证 → **确认 bug**。

---

## 根因 + 触发条件 + 建议修法

- **根因**:`example_hstu_attention_fwd.cpp:850-851`(及 853-854 的 kv 侧)
  ```cpp
  group_max_seqlens_q[i_grp] =
      group_max_uih_seqlens_q[i_grp] + group_contextual_seqlens[i_grp] + num_targets[i_grp];
  ```
  `group_max_uih_seqlens_q[i_grp]` 是组内**逐 batch uih 的 max**,但 `num_targets[i_grp]` 用**组下标**索引**逐 batch** `num_targets[]`(L725/L776 supplement 到 num_batch 长)→ 取了第 i_grp 个 batch 的 target,而非"组内最长 packed batch"的 target。两者来自不同 batch 时 → `max_seqlen_q` < 该组某 batch 真实 packed seqlen。
- **触发条件**:`g>1` 且组内 batch 数 >1、组内 seqlen 异构、**组内最长(uih+target)的 batch ≠ 第 i_grp 个 batch**(典型:高下标 batch 带大 target)。default 派生(不传 `-g_max_seqlens`)即触发——上游真实用法。
- **下游危害**:`max_seqlen_q` 喂 GridSize/mtile/splitkv → O 与 LSE 全局错(本测 87.5%/84.6%,含 NaN)。比 bwd 更严重(bwd 仅 PRE-D 的 dQ target 行;fwd 直接腐蚀主输出 O+LSE)。
- **建议修法**(与 bwd M6b 同源):per-batch 取 max,与 offset 公式一致
  ```cpp
  int gmax_q = 0, gmax_kv = 0;
  for (b in group i_grp) {
      int tgt = num_targets.empty()?0:num_targets[b];
      gmax_q  = max(gmax_q,  seq_lengths_q[b]  + tgt);
      gmax_kv = max(gmax_kv, seq_lengths_kv[b] + tgt);
  }
  // 若支持 -g_max_seqlens override:再 max 上 (override + 该组代表 target)
  group_max_seqlens_q[i_grp]  = gmax_q  + group_contextual_seqlens[i_grp];
  group_max_seqlens_kv[i_grp] = gmax_kv + group_contextual_seqlens[i_grp];
  ```
  另建议:加 `assert(max_max_seqlen_q >= 每 batch packed seqlen)`(同 bwd M6b),把 silent-wrong 变响亮失败。

## 纪律
- 只读 promoted/库;**仅临时改 fwd harness 公式验因果,已 `cp` 还原 + 重建,`git status` 干净**(M6b 已由 lead commit `d4fb2884`,工作树 clean)。
- 备份 `example_fwd.cpp.bak` 已删。fwd 二进制已从还原源重建。
