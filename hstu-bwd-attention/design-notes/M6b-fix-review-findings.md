# M6b/M5b harness 修复 — 独立复核结论 (pane-2 / reviewer)

> 复核对象:`group_max_seqlens_q` 低估 → PRE 漏算长 batch 尾 token 的 D → 仅 target 行 dQ 错。
> 方法:独立机器复跑 + 对抗实验(亲手把公式改回旧式实测回归案是否真触发)。**不信自述。**
> 日期 2026-06-10。binary 全程独立 touch+重编(`runs/build-M6b-review-pane2.log` EXIT=0)。

## 总评:**可签 promote(M5b 维持 / M6b 转 promoted)** — 1 条非阻塞发现须告知 lead

修法**正确、彻底、无副作用、零改库**,实测全部成立。bug 的真实面(softmax PRE-D)**被两个 softmax 回归案有效锁定**(改回旧公式即 FAIL)。
唯一发现:第三个回归案 `pass-gtrig-silu-atomic` 是**空挡(vacuous)** —— 改回旧公式它仍 PASS,锁不住本 bug;其 note「locks ... grid/scale_p」**夸大**。非阻塞(softmax 两案已堵洞),但建议改 note。详见 B2。

---

## 任务 A:独立机器验证(全 `-attn_scale=1.0`)

| 项 | 结果 |
|---|---|
| A1 干净重建 | 强制重编 5 改动 TU + relink,EXIT=0,0 error |
| A1 全套件(初次,我的干净 binary) | **TOTAL 91 / PASS 91 / FAIL 0 / SKIP 0 exit 0**(`runs/test-20260610-033522.log`)|
| A1 全套件(对抗实验后**还原 binary** 再跑) | **91/91/0/0 exit 0**(`runs/test-pane2-final-*.log`)= 终态磁盘已认证 |
| A2 亲跑原 FAIL 配置(直接) | dQ/dK/dV **全 PASS**;dQ max_abs_err=**7.39e-09**(原 0.0626) |
| A3 determ 两次 byte-identical | `cmp det1 det2` → **BYTE-IDENTICAL**(可复现)|
| A3 determ==atomic byte-identical | `cmp atomic det1` → **BYTE-IDENTICAL**(印证 done.md §3)|
| repro 套件 6 案(含 group determ 多 split)| 全 byte-identical |

零回归:M0–M6b 全绿(no_group determ/atomic、SiLU/softmax/group atomic+determ、jagged、mask)。

---

## 任务 B:对抗复核

### B1 修法正确性 — GREEN
新公式 `group_max_seqlens_q[g] = max_{b∈g}(seq_lengths_q[b]+num_targets[b]) + ctx[g]`。
offset 环里真实 `batch_seqlen = seq_lengths_q[b]+num_targets[b]+ctx[g]`。
∴ `group_max_seqlens_q[g] ≥ batch_seqlen ∀ b∈g`(ctx 组内统一相加,max 取在同一 per-batch 项上)——**构造上恒成立**。
- corner:组内 batch 数恒等(`num_batch_per_group=num_batch/num_group` 固定,无不等组);`num_target=0` 两侧公式同走 0;`ctx≠0` 统一加,均安全。
- override:旧式是**替换**(可低估),新式是 `max(...)`(**只增不减**)——更安全。

### B2 回归测试有效性 —— **关键,实测做了「改回旧公式」对抗实验**
我亲手把 harness 公式改回旧式(`packed=seq_lengths_q[b]` only + 末尾 `+num_targets[i_grp]` 组下标),分两步实测:

**Exp1(旧公式 + assert 在位)**:触发配置 → **loud abort**
```
what(): 'max_max_seqlen_q >= batch_seqlen' failed: group max_seqlen_q under-covers ...
```
**Exp2(旧公式 + assert 注释掉 = 还原原始 silent-wrong 态)**:三个回归案逐个亲跑 ——

| 回归案 | 旧公式下结果 | 有效锁定? |
|---|---|---|
| `pass-gtrig-sm-atomic` | dQ **FAIL** max_abs_err=**0.0958** | ✅ 真能复现原 bug |
| `pass-gtrig-sm-determ` | dQ **FAIL** max_abs_err=**0.0958** | ✅ 真能复现原 bug |
| `pass-gtrig-silu-atomic` | dQ **PASS** (0.0039) | ❌ **空挡 — 锁不住** |

**根因(为何 SiLU 案无法触发本 bug)**——三条路都不通,实测逐一证伪:
1. **PRE-D**:SiLU 路**无** `dot_do_o` PRE-D kernel,under-cover 无 D 可坏(bug 的唯一发源)。
2. **MAIN grid**:grid 按 kM0=128 瓦化,`max_seqlen_q` 208 vs 224 → `ceil(/128)=2` **同 2 tile**,长 batch 尾行仍被 MAIN 覆盖(per-block early-exit 按真实 seqlen,非 max)。
3. **scale_p**:即便喂 `-g_attn_scales=0,0` 走 `1/group_max_seqlen_q` fallback,该向量**GPU 与 reference 同吃一份**(`example_*_bwd.cpp:964` 同传 reference),错的 scale **两侧抵消** → 实测 dQ err=**0** 仍 PASS。

> **结论**:本 bug 本质 **softmax-PRE-D 专属**,已被 `sm-atomic`+`sm-determ` **有效锁定**(改回旧式即 FAIL)。
> `silu-atomic` 既非 PRE-D、grid 瓦化无效、scale_p 抵消 → **结构上不可能**检出本 bug,是冗余覆盖而非 lock。其 note 「locks group_max_seqlens_q fix for grid/scale_p」**夸大**,会给假信心。
> **非阻塞**(洞已堵),但**建议**:改 note 为「group SiLU hetero 通用覆盖(本 bug 仅 softmax-PRE-D 可检,见 sm-* 两案)」,或干脆删其 lock 声称。

### B3 assert 真守卫 — GREEN(但与派单设想的触发路不同,见下)
- Exp1 已证:**公式回归 → assert 真 abort**(把 silent-wrong 变响亮失败)。这是 assert 的真实价值:守 harness 内部公式一致性。
- 派单设想的「喂小 `-g_max_seqlens` → 应 abort」**实测不 abort**:`-g_max_seqlens=8,8` → 仍 **PASS、结果正确**。
  原因:override 现为 `max(...)`**只增不减**,小值被真实 per-batch max 压住,无法低估 → 无需 assert 也不会 silent-wrong。**这是更强的安全姿态,不是漏洞**(从"靠 assert 兜"升级为"根本构造不出低估")。assert 退化为防"未来有人改坏公式"的护栏,Exp1 证其可达。

### B4 无副作用 — GREEN
- `group_max_uih_seqlens_q` 在 bwd harness **彻底删除**;全仓残留仅在**上游 `example_hstu_attention_fwd.cpp`**(其自有局部变量,独立 TU,不受影响)。
- SiLU group `attn_scale=0` 的 scale_p fallback:用同一修正后向量,GPU/ref 同吃,实测仍对(B2 路 3)。
- 套件 91/91 绿,无连带破坏。

### B5 零改库 — GREEN
- vs M6 `c79d3296` **byte-identical**:`reference_hstu_attention_bwd.hpp`、`hstu_attention_with_softmax_bwd_pipeline.hpp`、`hstu_attention_bwd_pipeline.hpp`(SiLU)、`hstu_attention_bwd_dispatch.hpp`(no_group)、`example_hstu_attention_fwd.cpp`、`hstu_attention_fwd_kernel.hpp`。
- 5 个改动文件 = **M6b group-determ 特性**(params 加 split 字段 / 两 group kernel determ 分叉 / group dispatch `launch_main_and_post`+kIsDeterministic 轴 / group bf16 `BOOL_SWITCH_3` 修 O1)+ **harness 公式修复** + PRE **仅注释**(diff 证)。
- **额外安全核**:determ split workspace 一致性 —— harness 硬编 `kN0_bwd=128`,dispatch 用 `Pipeline::kN0`。实证 `kN0 = BlockTile::at(1)`,而 `FmhaBlockTile = sequence<32,128,...>` → **kN0==128 一致**(no_group M6 promoted 同款 pattern)→ **无 split 越界**。多 split group determ 结果正确且 byte-reproducible 旁证一致。

### B6 上游备注 — GREEN
done.md §6 如实记「上游 `example_hstu_attention_fwd.cpp` 疑似同款 `group_max_seqlens_q` 低估、建议上游报」。
亲核 `example_hstu_attention_fwd.cpp:850-851`:`group_max_uih_seqlens_q[i_grp] + ctx + num_targets[i_grp]` —— **同款 group-下标 target 混 group-max uih 模式**,确为同类。备注准确且恰当地用"疑似/建议"措辞、未越界改上游。

---

## 给 lead 的一句话
M6b harness 修复**可签**:公式恒正确、零改库、determ 可复现且==atomic、套件 91/91 还原态再认证。**1 条非阻塞**:`pass-gtrig-silu-atomic` 是空挡回归案(改回旧公式仍 PASS,结构上检不出本 bug),note 夸大,建议改措辞——但 bug 已由 `sm-atomic`/`sm-determ` 两案有效锁定,**不阻塞 promote**。
