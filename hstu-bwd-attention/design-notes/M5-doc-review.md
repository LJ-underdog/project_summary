# M5 softmax bwd — 文档级独立 review 结论 (pane-3 / 第三方视角)

> 角色:pane-3,边写 HTML 讲义边以全新视角核 silent-wrong(coder=pane-1、reviewer=pane-2 已过)。
> **只读源码,基线 = 磁盘当前态(aced5784 附近)。结论先行:GREEN,无 blocker,未发现 pane-2 之外的真问题。**

---

## 我独立重核过的点(逐条 GREEN)

### 1. 数学 ↔ 代码 ↔ reference 三方一致 — GREEN
- log2 域恒等式自己推过:`p = exp2(scale·s_acc − log2e·lse)`,`scale = α·log2e`
  ⇒ `exp2(log2e·(α·s − lse)) = e^(α·s − lse)` = reference `P=exp(α·S − LSE)`(reference:309)。✓
  - pipeline:410(scale)/464(row_lse=log2e·validated_lse)/467(p)。
- `get_validated_lse`(-inf→0)pipeline:413-416 无条件应用;reference 用 `lse_sq==-inf → P 整行=0`(301-305)分支。两者对 fully-masked 行都给 P=0,等价。✓
- STAGE5 `ds = p·(dp − d[i_idx])`(pipeline:512,d per-row)= reference `dS = P·(dP − D)`(413)。✓
- D 定义一致:PRE `D=Σ_v O·dO`(kernel:1259-1264)= reference `D=dO·O`(408-411)。✓
  且两侧 **同吃 GPU-fwd 产的 O**(harness:385 `o_dev.FromDevice(o_host)` 后同一 o_host 喂 reference)。✓
- scale 接线:dQ*=α(pipeline:559)、dK*=α(576)、dV 不乘(578);softmax 不传 scale_p(operator() 签名无 scale_p)。= reference dQ/dK *alpha(439/442)、dV 无。✓

### 2. 掩码方向 — GREEN
STAGE2 对 `!IsTokenPairInsideMask` 置 **−inf**(pipeline:457,exp2→0),与 SiLU 置 0 相反——正确。
运行时门 `mask.IsEdgeTile`(449,非编译期 IsMasking),causal=0+num_target 也逐 tile 掩(沿用 P1-1)。
row/col 计算(453-455)与 reference `mask.IsTokenPairInsideMask(sq,sk)` 同序。✓

### 3. LSE/D 四方布局 — GREEN(我自己推了偏移,四方落同一元素)
统一 `[batch,head,seq]` 连续-seq(head stride=`phy_seqlen_q`,seq stride=1):

| 方 | 代码位置 | base 地址 |
|---|---|---|
| **fwd 写 LSE** | fwd_kernel:724/758/936 | `i_nhead·phy_seqlen_q + {query_start \| i_batch·num_head·phy_seqlen_q}` + s(seq_stride_lse=1,harness:368) |
| **bwd 读 LSE/D** | bwd_kernel:683/697/717-722 | `i_nhead·nhead_stride_lsed(=phy_seqlen_q) + {query_start \| i_batch·batch_stride_lsed}` + s(packed view,stride 1) |
| **PRE 写 D** | bwd_kernel:1245/1253 | `i_nhead·d_nhead_stride(=nhead_stride_lsed) + {token=q_start+sq \| i_batch·d_batch_stride + sq}` |
| **reference 转置读** | harness:392-398 / ref:297-299 | `lse_host(b,s,h)=lse_flat[(b·num_head+h)·phy_seqlen_q + s]`;ref 读 (i_batch,sq,h) / (0,q_start+sq,h) |

四方地址逐项相等(`nhead_stride_lsed==phy_seqlen_q==fwd nhead_stride_lse`,`batch_stride_lsed==num_head·phy_seqlen_q`)。
PRE 写与 bwd 读用**同一对** `param.{nhead,batch}_stride_lsed`(dispatch:291-292 vs MakeKargs:332/340),构造上不可能错位。✓

### 4. PRE 免-memset 全覆盖 — GREEN
jagged:cu_seqlens 把 [0,ΣL) 精确划分,每 (head,token) 恰写一次;`if(sq>=seqlen_q) return`(1256)防跨批越界。
batched:grid 覆盖 num_batch·num_head·max_seqlen_q,全覆盖。✓

### 5. smem 复用 — GREEN
softmax ds_lds 偏移 = QT+OGrad+OGradT+Q+**LSE+D**(pipeline:367-372),SiLU 的 GetSmemSize 已预留 LSE/D 空洞 → 总量不变,无越界。✓

---

## 非阻塞观察(留 lead 评估,非 M5 缺陷)

1. **alpha 缩放在 bwd 全程仅以 α=1 验证**(铁律 `-attn_scale=1.0`)。softmax 的 `scale=α·log2e`、`dQ*=α`、`dK*=α` 与 reference 的 `S*=α`、`dQ/dK*α` 在**代码层逐项对称一致**,但数值上未被非 1 的 α 触发。这是 **bwd 系列贯穿性约束(SiLU 路同样如此)**,非 M5 引入。建议(若 lead 认可)后续补一档 α≠1 的对拍(GPU 与 reference 共用同一 α,可验证缩放接线而不触 LSE 盲区)。
2. **LSE 数值盲区**:GPU-bwd 与 reference 共吃同一份 GPU-产 LSE,对拍**结构上无法**独立验证 LSE 数值正确性(全错也会双错相等)。pane-2 已记录并由 fwd 里程碑兜底 + 写读布局自洽闭合——我复核同意,无新增。
3. **PRE 假定 O、dO 同 strides**(dispatch:284-292 只传 O strides)。本 harness O/dO shape 逐字相同恒成立;cross/异布局 dO 时需显式传 dO strides。pane-2 已提,代码已有注释(1205-1207 + dispatch 284-287)。

---

## 总评

**GREEN,可保持 promoted。** 三方数学/布局/scale/掩码/smem 我逐条独立重核(含自推四方偏移),与 reference + FMHA 蓝本逐字对齐,未发现 pane-2 之外的 silent-wrong。仅 1 项非阻塞建议(α≠1 补测),其余均为已记录且已闭合的盲区。HTML 讲义已据此如实写出,无"讲不圆"之处。
