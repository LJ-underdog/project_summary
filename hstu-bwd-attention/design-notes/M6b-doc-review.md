# M6b HTML 讲义 —— 独立文档级 review（pane 0.2 / reviewer）

**审查对象**:`/root/workspace/hstu-b1052-report/hstu-bwd-M6b-group-determ-20260610.html`
**素材基准**:`M6b-done.md` + `candidates.jsonl` 第 12 行 + `HANDOFF.md §6`(上游定性)+ `M6b-fix-review-findings.md`(reviewer 数字)+ `git d4fb2884`。
**方法**:逐数核对源 + 抽查源码行号 + 渲染/链接自检。**不信自述。**

## 结论:**可发布(GREEN)。** 全 7 条 GREEN,无 RED、无臆造。仅 1 条 trivial 非阻塞排版 nit(可选)。

---

## 逐条结论

### 1. 数字一致 —— **GREEN(逐一核过,含两个 done.md 外的数字均已溯源)**
| 数字 | HTML | 源 | 判定 |
|---|---|---|---|
| commit | `d4fb2884` | HANDOFF/candidates/done | ✓ |
| 套件 | 91/91/0/0 exit 0 | done §3 / candidates | ✓ |
| 原 FAIL→PASS | 0.0626221 → 7.4e-9 | done §3 | ✓ |
| reviewer 独立机器 | 7.39e-09 | **fix-review-findings A2** | ✓ 溯源到 |
| 触发配置 | `-b=4 -nhead=4 -g=2 -seqlens=128,200,96,160 -g_local_lens=16,16 -targets=8,24,8,16 -softmax=1 -causal=1` | done §3(+`-attn_scale=1.0` 见 done 抬头铁律) | ✓ |
| max 208→FAIL / ≥224→PASS / grid.x=2 | 同 | done §1 | ✓ |
| 坏 token [208,224) | 同 | done §1 | ✓ |
| 3 锁定案名 | sm-atomic/sm-determ/silu-atomic | done §3 | ✓ |
| lock 实测 | sm-atomic 0.0958 FAIL / sm-determ 0.0958 FAIL / silu-atomic 0.0039 PASS | **fix-review-findings B2** | ✓ 溯源到 |
| 改 5 文件 / +186−114 | 同 | done §4(5 文件)+ **git show d4fb2884 = "5 files changed, 186 insertions(+), 114 deletions(-)"** | ✓ git 实证 |
| B3/B5/B6 引用 | 同 | fix-review-findings 真有这些标号 | ✓ |

**两个 done.md 里没有的数字(`+186/−114`、reviewer 的 `7.39e-09`/`0.0958`/`0.0039`)我都追到了一手出处(git stat + fix-review-findings),非臆造。**

### 2. ★ 上游定性准确(最关键)—— **GREEN(不仅没复活 bug 措辞,还正确记录了撤回)**
- §5 表:本 repo **bwd harness** = `真 harness setup bug`(红,正确——这是我们自己要修的);上游 **fwd example** = `非 bug`(绿)+「2026-06-10 与 fwd 作者核实后定性」+「不改、不报上游」。**完全对齐 HANDOFF §6。**
- warn-banner 给出非-bug 理由(`-g_max_seqlens` "can be ignored, or else bigger"、调用方负责 over-provision、"kernel 正确,正确用法下无误"),与 HANDOFF §6 一字不差的口径。
- clay note **主动记录差点误报**:曾建议报上游 → 核实后撤回 → 以 HANDOFF §6 为准 →「早期过度定性为 bug 的草稿/HTML 已作废」+ 教训。
- **关键裁断**:done.md §6(coder 旧报告)其实**仍写着「建议向上游报」「上游疑似低估」**——那是上游再定性之前的措辞。HTML **正确地用了更晚的 HANDOFF §6 结论覆盖 done §6**,没有照抄 done 的过时建议。**这是文档作者做对了,是加分项,不是问题。**

### 3. 根因链准确 —— **GREEN**
§4 + 图2:max_seqlen_q 低估 → PRE `dot_do_o`(grid 按 max_seqlen_q、d_dev 未 memset)漏算 [max_seqlen_q, 真实 seqlen) 尾 token D → 垃圾 D → `dS=P·(dP−D)` 错 → 仅 softmax target 行 dQ 错;dK 稀释/dV 不含 D 不受。判别实验三刀(G_ref==N_ref、G_dev!=N_dev、max≥224 PASS/208 FAIL 而 grid 不变)。**与 done §1 完全一致。** "仅 dQ"三因(dS 经 K 进 dQ / dV 不含 D / dK 仅稀释)亦与 done 括注一致。

### 4. 机制准确 —— **GREEN(抽查源码行号属实)**
- group determ = 复用 M6 `set+split`(`base += i_tile_n·split_stride`)+ 固定序 `Σ_s` reduce(直接复用 M6 `hstu_bwd_reduce_convert_dq_kernel`)。§2 述及。
- O1 = group entry `BOOL_SWITCH_2→3` 接 `param.kIsDeterministic`。**实查 `hstu_attention_group_backward_bf16.cpp:20-27` = BOOL_SWITCH_3 + kIsDeterministic,属实。**
- params += `split_stride_dq_acc`/`num_splits`:**实查 `hstu_attention_bwd_params.hpp:102-103/186-188`,属实。**
- "去掉那条永不可达的 determ throw":**实查——`"not implemented yet (M6)"` 已不存在(grep 空);`launch_main_and_post` 在 dispatch L83,属实。dispatch L310 仍有一条 throw,但那是 hdim 校验守卫(M7b/M7c 未支持 hdim 的 runtime reject),与 determ 无关——HTML 的"删 determ throw"声称准确,未误指。**

### 5. 诚实呈现过程 —— **GREEN(如实,且转述了 reviewer 全部细微发现)**
§6 完整呈现:coder 首轮 9 PASS/1 FAIL 误标 promoted → lead 打回 → 根因 → 修 → reviewer 对抗 formula-revert。**未美化**。还忠实转述了 reviewer 的两个细处:
- `silu-atomic` 是**空挡(vacuous)**回归案(改回旧公式仍 PASS,结构上检不出本 bug),其 note「locks grid/scale_p」**夸大**,reviewer 建议改措辞(非阻塞)。—— 对应 fix-review-findings B2,转述准确。
- 守卫触发路**与派单设想不同**(override 现为 `max(...)` 只增不减,小值被压住,根本构造不出低估;assert 退化为防"未来改坏公式"的护栏,Exp1 证可达)。—— 对应 B3,转述准确。
这种"把对自己不利/不完美的细节也写进讲义"正是诚实呈现。

### 6. 范围/边界 —— **GREEN(无超范围)**
§1 note:「determinism 自此覆盖全模式」明确限定 = no_group(batched+jagged,M6)+ group(M6b),均跨 SiLU+softmax。**与 done §6 一致,未夸大到 fp16/其它 hdim。** dQ-only(dK/dV 无跨 block 累加问题)也讲清。

### 7. HTML 质量 —— **GREEN**
- 自检:489 行;3 个 `<svg>` 全闭合;**外链 0**(无 CDN/http/@import);TOC 7 个 anchor 与 7 个 section id 全对应;跨文档链接 `hstu-bwd-M6-deterministic-20260609.html` **目标文件存在**;无 TODO/占位符/lorem;无文本溢出风险(沿用 M6 同款 `.svg-wrap overflow-x` + `max-width` 容器)。
- 单文件自包含、`<style>` 内联、图全 inline SVG——可双击直接渲染。

---

## 唯一非阻塞 NIT(可选,不阻塞发布)
- **`★ / ✓ / ✗` dingbat 符号 7 处**(§4/§6 标题用 ★;§6 lock 表用 ✓/✗)。skill 铁则「无 emoji」;同系列 M6 讲义这三个符号用了 **0** 次。这些是单色 dingbat 非彩色 emoji,渲染无害,但为系列一致可考虑替成纯文字(如 ★→「重点」、✓/✗→「有效/空挡」)。`→`/`⇒` 箭头是标准技术排版,保留无妨。**纯排版偏好,不影响内容正确性。**

## 一句话给 lead/写作者
**M6b 讲义内容全部属实(数字溯源齐、源码行号抽查属实)、上游口径正确(非 bug + 撤回记录,完全对齐 HANDOFF §6)、过程诚实(连 reviewer 的空挡案/夸大 note/守卫触发路差异都如实转述)——可直接发布;唯一可选 nit 是 ★/✓/✗ dingbat 与系列一致性,改不改都行。**
