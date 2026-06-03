# iw-reviewer — ATOM_ISSUE_DRAFT review

> 日期：2026-04-29
> Reviewer：teammate (issue-reviewer)
> 红线：未修任何源码 / 未改 ATOM_ISSUE_DRAFT.md / 未动其他 .md

---

## TL;DR

- **整体评级：A-**
- **block: 0** / **HIGH: 1** / **warn: 3** / **info: 3**
- 一句话推荐 Lead 决策：draft 主体（traceback / 代码引用 / 数学推断 / D=10 推断 / advisory fix 双 option / caveats）全部 PASS 且与上游 maintainer 视角对齐；唯一 HIGH 是 §"Affected configurations" 表格里若干 D 值算错（特别是 tp=8 的 "9..15" 上界、tp=4 列），需要 Lead 在 file 前用一句话修订该表（建议把表格简化成"举触发示例 D=10/tp=8"+"通用条件公式"，避免穷举出错）；其它都是 warn/info 加分项。

---

## C1 traceback 准确性 — **PASS（带 1 warn）**

逐条核对：

| draft 元素 | 上游源 | 是否一致 |
|---|---|---|
| `step3p5.py", line 897, in load_fused_expert_weights` | corr-t1.md L59 = "line 897"；ATOM step3p5.py L897 实测 = `weight_loader(param, loaded_weight[expert_id], name, shard_id, expert_id)` | ✓ |
| `moe.py", line 2610, in weight_loader` | corr-t1.md L61 ✓ / CORRECTNESS_REPORT §4 L124 ✓ | ✓ |
| `moe.py", line 2256, in _load_model_weight_or_group_weight_scale` | corr-t1.md L63 ✓ / §4 L125 ✓ | ✓ |
| `moe.py", line 2357, in _load_w2` | corr-t1.md L65 ✓ / §4 L126 ✓ | ✓ |
| `RuntimeError: narrow(): length must be non-negative.` | corr-t1.md L67 ✓ / §4 L128 ✓ | ✓ |
| Symptom B 错误信息 `(2) must match (0) at non-singleton dim 1` | §4 L132 ✓ | ✓ |
| 崩溃阶段 `Loading safetensors shards: 1/44` | §4 L121 ✓ | ✓ |
| rank 5/6/7 分配 | §4 "rank6 / rank7 崩溃，rank5 切到 size=0" L161 ✓ | ✓ |

无虚构 stack frame。

**warn-1（C1）**：corr-t1.md L56 写崩溃发生在 "safetensors 加载到 4/44 shard"，CORRECTNESS_REPORT §4 L121 写 "1/44"，draft "Reproduction" 段也写 "1/44"。两个上游 .md 数字本身有出入但 draft 与主报告 §4 一致；建议 Lead 知悉但不阻塞 file。

---

## C2 代码引用准确性 — **PASS（带 1 warn）**

draft §"Root cause" 引的 `_load_w2` 代码块与 ATOM `acff926` 实测 moe.py:2335-2364 逐行核对：

- 函数签名 ✓（`def _load_w2(self, expert_data, shard_dim, loaded_weight, tp_rank, load_full=False):`）
- `shard_size = expert_data.shape[shard_dim]` ✓
- `if not load_full:` ✓
- ceil 切分公式 ✓
- `start = load_shard_size * tp_rank` ✓
- `size = min(load_shard_size, loaded_weight.shape[shard_dim] - start)` ✓
- `narrow` line 标注 `← line 2357` ✓
- `if load_shard_size != shard_size: expert_data = expert_data.narrow(shard_dim, 0, load_shard_size)` ✓
- `expert_data.copy_(loaded_weight)` ← line 2364 ✓（注：实际 line 2364 在 `if dtype != fp4x2` 分支下；draft 简化呈现，非误引）

draft §"Root cause" 引的 `_load_w13` 代码块（L2310-2314）：
- 公式与 moe.py L2310-2315 一字不差 ✓
- 注释 line 2306-2309 引用 `inter=1280, tp=4 → 10 blocks` 与实测注释一致 ✓
- draft 标注 "line 2315 (same bug)" ✓ 与实测 narrow 行号一致

**warn-2（C2）**：draft §"Root cause" `_load_w2` 代码块的注释写 "ceil split"，实际 moe.py L2351 注释是 `Use ceil (same reason as _load_w13: partial last scale block).`。draft 自加注释属"reviewer 注解"，但用单 `#` 与代码混排可能让上游 maintainer 误以为是源码注释。建议 Lead 把 draft 内的解释性 `# ceil split` / `# ceil 切分` / `# same ceil split` 这三处统一改成 `# ← ceil-based split` 或在代码块外文字描述（非阻塞）。

---

## C3 数学推断正确性 — **PASS**

| 命题 | 验算 | 结论 |
|---|---|---|
| 触发条件 `D < C * (tp_size-1)` 对 last rank narrow raise | last rank 的 `start = C*(tp_size-1)`，narrow raise 当且仅当 `start > D` ⟺ `D < C*(tp_size-1)` | ✓ 正确 |
| tp=2 永远安全 | tp=2 → trailing rank=1, start=ceil(D/2)*1=ceil(D/2)；ceil(D/2)≤D 对所有 D≥1 成立（D=1: 1≤1; D=2: 1≤2; …） | ✓ 正确 |
| D=10, tp=8 → C=2, rank5 start=10, rank6 start=12, rank7 start=14 | 2*5=10=D → size=0；2*6=12>10 → size=-2；2*7=14>10 → size=-4 | ✓ 完全正确，且与观测 symptom 完全匹配 |
| symptom 拆分（rank5 copy_ mismatch / rank6,7 narrow raise） | size=0 时 narrow 不 raise 但 copy_ 拒绝（expert_data 在该 dim 不为 0）；size<0 时 narrow 直接 raise | ✓ 物理图像与 PyTorch 行为一致 |

数学段无错误。

---

## C4 D=10 推断合理性 — **PASS**

- moe.py L2306-2309 注释明确给出 `per_1x128 with inter=1280 and tp=4: 10 blocks`，所以 "D = 10 = inter_size 1280 / 128 (per_1x128 scale block 数)" 这条推断**直接由源码注释支撑**，不是凭空。
- "D=10 在 tp=8 重现 rank5 size=0 + rank6/7 size<0 完美对齐 symptom" 是强一致信号，但由于 draft 没有从 dump 直接 print(name, loaded_weight.shape) 作 ground truth，**严格说仍是 inferred 而非 confirmed**。
- draft §Caveats #1 已诚实标注："The exact value `D = 10` is **inferred** ..." 并给出上游 maintainer 自验方法（在 line 2357 前加 print）✓

**结论**：推断逻辑严谨，已正确标注为 inferred，C4 PASS 无 finding。

---

## C5 advisory fix 严谨性 — **PASS（带 1 warn）**

- §"Proposed fix" 顶部写 "advisory only — not implemented in this draft"，"Neither has been implemented and either should be discussed with ATOM maintainers before landing" ✓ 极清晰
- Option A / B 各自代码块标注 `# Pseudocode, NOT a patch` ✓
- Option A trade-off：minimal change / 但需 audit 下游消费者 ✓
- Option B trade-off：never reach negative / 但改变 split 分布、与 L2306-2309 注释 "ceil 是为包含 last partial scale block" 的设计意图冲突 ✓
- "Sweep target" 段明确指出需同时 patch `_load_w2` (L2355-2357) 与 `_load_w13` (L2313-2315) ✓ 与 ATOM CLAUDE.md 的 "fix-then-sweep" rule 对齐

**warn-3（C5）**：Option B 段写 "Pros: every rank always gets `start + size <= D`, no negative-size case is reachable" — 严格说还应附 caveat "rank0 拿到 base+1, trailing rank 拿到 base，而 ceil 切法是反过来（trailing rank 容易拿到 0），下游 kernel 若期望 'rank0 是 full block / trailing rank 是 partial block' 这种顺序约定也会被反转"。当前 Cons 段只说"partial block 位置反转"，未点明可能影响 kernel 内 dispatch 顺序。建议 Lead 在 Cons 末尾追加一句"this also reverses the per-rank residual size ordering, which AITER fused MoE kernel may rely on for per-block dequant indexing"。非阻塞。

---

## C6 issue 体裁 — **PASS（带 1 info）**

- **长度**：实测 225 行 / A5 上限 250 ✓ 紧凑
- **toC**：Title / Summary / Environment / Reproduction / Traceback / Root cause / Affected configurations / Proposed fix / Why this matters / References / Caveats — 完全符合 GitHub issue 标准结构
- **5 min 上游 maintainer 体验自测**：
  - 文件:行号 → Title 即写 `_load_w2` (line 2335-2364) / `_load_w13` (line 2292-2333) ✓
  - 触发条件 → §"Trigger condition" 1 个公式 + 1 行文字 ✓
  - advisory fix → §"Proposed fix" 两 option 并列 ✓
  - **A1 验收成立**

**info-1（C6）**：§"Why this matters / scope" 第 3 个 bullet（"Cross-references the upstream symptom of the fp8-tp4-repro project's main RCA ..."）一气列了三个 sibling fix（`aiter/fused_moe.py:881-886`、`atom/model_ops/moe.py:1709-1746`、`atom/model_ops/utils.py:79`），上游 maintainer 视角看这是 fp8-tp4-repro 项目内部上下文，对 file issue 帮助小，可能让 issue 看起来"我们已经在自己 fork 改了一堆"。建议 Lead 评估：要么删除该 bullet（信息密度不高）、要么改成 "(Internal note: this is the 4th and most upstream sibling of three sharding fixes already integrated downstream in this project; not relevant to upstream patch scope)"。非阻塞 / Lead 可酌情。

---

## C7 reference 路径正确性 — **PASS**

| draft 引用 | 实测核对 | 结论 |
|---|---|---|
| `atom/model_ops/moe.py:2335-2364` (_load_w2) | moe.py 实测 L2335-2364 ✓ | ✓ |
| `atom/model_ops/moe.py:2292-2333` (_load_w13) | 实测 L2292-2333 ✓ | ✓ |
| `correctness_eval/logs/tp8_full.log` | 文件存在（19358 bytes, 2026-04-29 05:22）✓ | ✓ |
| `correctness_eval/CORRECTNESS_REPORT.md §4` | 文件存在；§4 即 "tp=8 崩溃 RCA" ✓ | ✓ |
| `correctness_eval/progress/corr-t1.md §3` | 文件存在；§3 即 "tp=8 启动 + 失败" ✓ | ✓ |
| `handoff_wave/HANDOFF_PACKET.md §4.1 F-OPEN-1` | 文件存在；§4.1 起于 line 78，F-OPEN-1 是 §4.1 第一个子节（line 80）✓ | ✓ |
| ATOM commit `acff926` | 实测 `git rev-parse HEAD` = `acff926de8b1699101962116470066d4e3c78b0e` ✓ 与 TEAM_CONFIG / CORRECTNESS_REPORT 一致 | ✓ |
| AITER `0f8164017` / CK `defd7ad29` | aiter / ck 仓不在 reviewer 当前可访问目录直接核对，但与 TEAM_CONFIG §1 + CORRECTNESS_REPORT 顶 + handoff_wave 全部 .md 一致 ✓ | ✓（间接证据） |

C7 全部 PASS。

---

## C8 caveats 完整性 — **PASS（带 1 warn + 2 info）**

draft §Caveats 4 条：
1. D=10 inferred 而非 confirmed ✓（C4 已确认）
2. `_load_w13` crash 未独立观测（_load_w2 先 crash 杀进程）✓ 诚实
3. Option A/B 都未实施 / 未测试 ✓
4. perf wave tp=8 PASS 与本 crash 不能 reconcile，标记 F-OPEN-1 sub-question ✓

**HIGH-1（C8 → 实际是 C3 的反弹）**：§"Affected configurations" 表格里穷举 D 值有错（这是本次 review 唯一升 HIGH 的 finding）：

  - tp=8 行写 `1..7, 9..15`：实测 D=15 时 C=ceil(15/8)=2, rank7 start=14<15, size=1 → **不 crash**；类似 D=14 时 size=0 也只是 copy_ mismatch (Symptom B) 不是 narrow raise。准确的"narrow raise" D 集合是 {1..7, 9..13}（D=14 是 Symptom B；D=15 完全 OK；D=8/16 整除全 OK）。
  - tp=4 行写 `1..3, 5..7, 9..11, 13..15`：实测 D=13 时 C=ceil(13/4)=4, rank3 start=12<13, size=1 → **不 crash**；D=14 size=2 也 OK；D=15 size=3 也 OK。准确的"tp=4 narrow raise" D 集合是 {1..3, 5..6, 9..10}（不含 13..15，且 D=7,11,12 也 OK）。
  - 该表如被上游 maintainer 当作"我们已经穷举测过 narrow 触发集"看，会发现"D=15 反而 OK"立刻怀疑整个 RCA 的严谨度，对 issue 可信度伤害较大。
  - **建议 Lead 修订该表**：要么删除穷举 D 列、改为 "for `tp_size` in {4, 8, ...} the trigger is satisfied by many small D values; concrete observed example: tp=8, D=10"；要么把"crash"列严格定义为"narrow raise OR copy_ mismatch"两类一起列。

**warn-4（C8）**：Caveats 没有显式说"我们没有真正打开 model checkpoint 看 inter_size 是否真的是 1280"。draft §"Trigger condition" 段说"`D = 10` (the per_1x128 scale block count for `inter_size = 1280`)"，但 inter_size=1280 这个数本身在 draft 内没有 ground truth 引用（既不在 corr-t1.md 也不在 CORRECTNESS_REPORT 内显式出现，而是从 moe.py L2306-2309 的**注释例子**反推）。建议 Caveat #1 末尾追加 "Furthermore, `inter_size = 1280` for Step-3.5-Flash-FP8 is itself inferred from matching the L2306-2309 comment example to the observed crash pattern; we have not extracted it from the model config." 非阻塞。

**info-2（C8）**：draft 没说"crash 发生在哪个 expert / 哪个 param name"。从 §"Caveats" #1 暗示需要上游打 print(name, ...) 自验，已经隐式覆盖；但加一条 Caveat "we did not isolate which `name` (e.g. `moe.gate_proj.weight_scale_inv`) hit the crash first; it could be either the weight or its scale tensor" 会让上游 5 min 阅读体验更顺畅。非阻塞。

**info-3（C8）**：draft Caveat #4 引用 "F-OPEN-1's open sub-question" 但 issue 上游 maintainer 不知道 F-OPEN-1 是什么内部编号。建议把这句改成"we will resolve this independently in our project tracker; this is not a blocker for triaging the upstream bug"，避免暴露内部 task ID。非阻塞。

---

## Lead 决策建议

### 接受标注（caveat 添 1 句即可，不重写段落）
- warn-1（safetensors shard 1/44 vs 4/44 不一致）：知悉，不动 draft 即可
- warn-2（draft 内 `# ceil split` 注释易被误读为源码注释）：可选，1 处文字润色
- warn-3（Option B Cons 缺 ordering caveat）：可选，1 句追加
- warn-4（inter_size=1280 自身也是 inferred）：可选，1 句追加到 Caveat #1
- info-1 / info-2 / info-3：纯加分项，Lead 自由

### 修订 draft（**建议 Lead 在 file 前修**）
- **HIGH-1（§"Affected configurations" 穷举 D 表算错）**：必须修，否则上游 maintainer 抽样核算就会立即怀疑整体 RCA。建议替换方案：删掉穷举 D 表，留通用条件公式 + "concrete observed: tp=8, D=10" + 一句 "the boundary between Symptom A (narrow raise) and Symptom B (copy_ mismatch) is `start == D` vs `start > D`; both are the same root cause"。

### 派补证据 task（**不需要**）
- 本 review 没有 raise 任何"需要新跑数据才能闭环"的 finding。draft 现有证据链（CORRECTNESS_REPORT §4 + corr-t1.md §3 + moe.py L2335-2364 实测）已足够支撑 issue file，**无需新跑 GPU**。

---

**End of review.** draft 主体素质优良（A-）；仅需 Lead 在 file 前 1 处必修（HIGH-1 affected configs 表）+ 至多 4 处可选润色。
