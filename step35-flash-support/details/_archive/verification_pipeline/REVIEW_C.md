# Reviewer-C 审查报告：V05 / V07 / MASTER_PIPELINE

审查者：Reviewer-C（FP8 dispatch 正确性 / ASM kernel workaround / pipeline 合理性）
审查日期：2026-04-25
审查对象：V05_fp8_inference.md、V07_longseq_bos.md、MASTER_PIPELINE.md

---

## 0. 总体评估表

| 文档 | 逻辑严密度 | Agent Team 适配性 | 准确性风险 | 总评 |
|------|-----------|-------------------|-----------|------|
| V05 | A-（guard 集合完整，但可扩展性、Fix 2 反证薄弱） | B+（Exp 1 必须串行，Exp 2/3 可拆 GPU） | B（log level、JIT cache 污染、Exp 5「无差异」结论模糊） | B+ |
| V07 | A（CSV 实地核查充分，损益评估合理） | A-（Exp 1/5.a 完美并行，Exp 2/3 GPU 冲突需切分） | A-（diff<5 阈值缺定量来源；first_token!=0 标准过弱） | A |
| MASTER | A-（依赖图清晰，跨专题问题 1.1-1.7 已识别） | B（Phase 2 三专题 GPU 总和 8，恰好满；调度策略未给） | B+（Phase 0 缺 FP8 模型预检；MEMORY 更新未自动化） | A- |

**总结论**：三份文档均可执行，但 MASTER 的 Phase 2 GPU 调度、V07 的优先级排序、V05 的 Fix 2 反证设计需补强。下文给出具体修订建议。

---

## 1. V05-FP8 Inference 审查

### 1.1 逻辑问题

#### 1.1.1 q_type guard 可扩展性（A.1 节）

**问题**：当前 guard 写法 `q_type not in (per_1x128, per_1x32)` 是「黑名单包含式」—— 任何未来新增的 blockscale q_type（如 `per_1x64`、`mx_e4m3` 等）默认会被推入 V3 强制 block_m=128 的路径，再次复发本 bug。

**V05 §A.1 已识别此风险**，但仅在「PR description 中标注同步点」—— 这是社会工程而非工程兜底。

**建议**：
- 在 V05 §A.1 末尾追加实验 7：**guard 反向覆盖断言**。在 `aiter/fused_moe.py` 中加 import-time assert，检查 `QuantType` 中所有以 `per_1x` 开头的成员是否都在 guard tuple 内；若发现新增枚举，立刻 raise 或 warn。该 assert 是 0 成本的回归保护。
- 或在 V05 P0 实验中加一项「QuantType enum 静态扫描」（Exp 0：grep `class QuantType` + diff against guard tuple），归入 V01 实验 4 同类的静态核查。

#### 1.1.2 Exp 1（crash 复现，临改源码）的安全性

**问题**：V05 §B 实验 1 的步骤 1 直接编辑 `/home/hanchang/aiter/aiter/fused_moe.py`，但步骤 4「复原」仅说「git checkout 该文件」。若实验过程中：
- aiter 仓库恰好处于 dirty 状态（其他 uncommitted 修改），checkout 会丢失。
- 实验 agent 进程被 kill 而未执行复原步骤。
- 多个 agent 并发写同一文件。

**建议**：
- 在 Exp 1 步骤 1 前强制：`cd /home/hanchang/aiter && git status --porcelain aiter/fused_moe.py` 必须为空，否则 abort。
- 用 `cp aiter/fused_moe.py aiter/fused_moe.py.v05_backup` 显式保留备份；复原阶段对比 `diff -q` 确认 byte-identical。
- Exp 1 必须独占运行（agent 加 `wait_for_other_agents` 同步点）；其他实验在 Exp 1 期间禁止读 `fused_moe.py`。

#### 1.1.3 Fix 2「无直接影响」反证不足（A.3 节）

**问题**：V05 §A.3 推断「block_shape=None 在 TP 路径下可能下游 inferred，无差异」，并把这种「无差异」结论作为可接受输出（§B 实验 5 通过标准）。但「无差异」可能源于：
- (a) Fix 2 真无影响（下游 wrapper 自己根据 quant_type 推断）—— 此时 Fix 2 是冗余但安全。
- (b) 下游有 silent fallback 路径，吃掉了 None 但走了一条数值次优 kernel —— 此时 Fix 2 是 must-have，但用 cos_sim 看不出。
- (c) None 被传到下游后某分支根本未触发（如 fused_moe_quant_config 在 TP 路径里被 cache，第二次调用前已被覆盖）。

V05 实验 5 没有区分这三种情况。

**建议**：
- 在实验 5 中加一步：`AITER_LOG_LEVEL=INFO` + 在 fused_moe 入口 print 实际 `block_shape` 形参值。若 None 被某处转换成 `[128,128]`（情况 a），日志会显示；若 None 一路传到 kernel（情况 b/c），需进一步在 weight scale 校验处加 print。
- 通过标准升级：必须能在日志中看到 `block_shape` 的实际下游值，而不仅是「输出是否相同」。

### 1.2 Agent Team 分工方案（含 GPU 分配）

#### V05 最优 Agent 拆分

| Agent | 任务 | GPU | 是否可并行 | 预计时长 |
|-------|------|-----|----------|---------|
| **agent-V05-A**（独占源码） | Exp 1（修源码 + 跑 Exp 2 命令观察 crash + 复原 + 验证 git status 干净） | GPU 0,1 | **不可并行**（独占 fused_moe.py） | 30 min |
| **agent-V05-B** | Exp 2（FP8 tp=2 端到端核心） | GPU 0,1 | 与 agent-V05-C 串行（同卡）；与 agent-V05-D 可并行 | 20 min |
| **agent-V05-C** | Exp 3（BF16 tp=2 回归） | GPU 2,3 | 与 agent-V05-B 并行（不同 GPU） | 20 min |
| **agent-V05-D** | C.1 模型缓存预检 + Exp 5 block_shape 区分性（需改 ATOM moe.py） | GPU 6,7 | 与 B/C 并行（不同 GPU + 不同源文件） | 25 min |

**关键设计点**：
1. **Exp 1 必须最后跑**：因为修了 aiter 源码后即使复原，JIT cache 已被污染（见 1.3 节）。Exp 2/3 必须在 Exp 1 之前完成，确保用的是 clean 源码。
2. **Exp 2 与 Exp 3 可并行（不同 GPU 集）**：Exp 2 用 0,1，Exp 3 用 2,3。两者不修源码、不冲突。
3. **Exp 5 修 ATOM 而非 aiter，且独占 ATOM moe.py**：与 Exp 1 的 aiter 修改不冲突，但与 V06 的 ATOM moe.py 修改冲突（必须互斥）。

#### 推荐执行顺序

```
T+0    : agent-V05-D (C.1 预检) + agent-V05-B (Exp 2) + agent-V05-C (Exp 3) 三方并行
T+25min: agent-V05-D 转入 Exp 5（独占 ATOM moe.py）
T+45min: agent-V05-B/C 完成 → agent-V05-A 启动 Exp 1（独占 aiter fused_moe.py）
T+75min: 全部完成
```

总计约 75 分钟（vs 串行的 ~150 分钟）。

### 1.3 准确性风险

#### 1.3.1 AITER_LOG_LEVEL=INFO 的覆盖性

**问题**：V05 §C.3 依赖 `AITER_LOG_LEVEL=INFO` 抓 dispatch 日志。需确认 aiter logger 是否真的有 INFO 级别 + 是否在 fused_moe.py L921-923 实际调用 `aiter.logger.info(...)`。

**风险**：若 aiter 用的是 Python 标准 logging 但默认 handler 配置为 WARNING，INFO 消息会被静默丢弃。

**建议**：在执行 Exp 2 前，先用 `python -c "import aiter; print(aiter.logger.level, aiter.logger.handlers)"` 确认 logger 配置；若 handler 缺失，在测试脚本顶部显式 `logging.basicConfig(level=logging.INFO)`。

#### 1.3.2 Exp 1 的 JIT cache 污染

**问题**：V05 §B 实验 1 的步骤 2 仅 `rm -rf /root/.cache/atom/*`，但**未清理 aiter 自己的 JIT 缓存**（通常在 `/root/.cache/aiter/` 或 `~/.aiter_cache/`）。修源码后跑出 crash，JIT cache 会保留 buggy kernel；复原源码后再跑 Exp 2，JIT cache 可能仍命中 buggy 版本。

**建议**：Exp 1 步骤 2 升级为：
```
rm -rf /root/.cache/atom/* /root/.cache/aiter/* ~/.aiter_cache/* 2>/dev/null
```
并在 Exp 1 完成后再次清理一次，确保 Exp 2 用的是 clean cache。

#### 1.3.3 Exp 4（cos_sim>0.995）的 router topk 一致性

V05 §B 实验 4 已注意到 router topk 不一致问题。但「expert-mask 后再比较」的具体实现未给。BF16 与 FP8 的 router 即使 temperature=0 + 同 input embedding，因 expert weight 量化误差，topk 可能在 boundary case 选不同 expert，导致 cos_sim 假性偏低。

**建议**：把实验 4 降级为 P2（信息性），通过标准放宽：「cos_sim > 0.9（含 router 偏差）或 cos_sim > 0.99（强制对齐 router）」。

---

## 2. V07-LongSeq BOS 审查

### 2.1 逻辑问题

#### 2.1.1 Exp 4 副作用测试缺正确性维度

**问题**：V07 §B 实验 4 测的是延迟（tgemm vs torch.mm us 对比），但**没有测 M=16384 这一 worst case 下的功能正确性**。当前修复后 (N=4096, K=2048, M=16384) 走 `torch.mm` —— 我们假设 torch.mm 在所有 M 下都正确，但这本身需要数值验证。

实验 1 的 M_list 包含了 16384，似乎覆盖了正确性。但仔细看实验 1 的 `w` 是 `randn` 随机权重，与实际模型 o_proj 权重的 condition number 不同。在 randn 权重下 torch.mm diff < 5，不能完全保证真实 o_proj 在 M=16384 下也正确。

**建议**：
- 实验 4 加一步「正确性 spot-check」：用真实模型 o_proj 权重（从 `/root/.cache/huggingface/.../snapshots/.../` 加载某层 weight）跑 M=16384 的 tgemm.mm，与 fp32 ref 对比。
- 或在实验 1 中追加一个真实权重 case：`w_real = load_real_o_proj_weight(layer=0)` 跑 M=16384。

#### 2.1.2 Exp 6（tp=2 negative control）应升 P0

**问题**：V07 §A.5 表格断言「tp=2 K_in=4096 ≠ 2048 → 不命中」，并打了【未验证 / 推断】标记。**实验 6 是这个推断的唯一直接证据**，但被列为 P1。

如果实验 6 失败（tp=2 也复现 bug），则：
- §A.5 表格的 K_in 推算有误，整个 ASM kernel 影响面分析需重做。
- V05 的 FP8 tp=2 端到端结论可能也受影响（FP8 tp=2 长 prompt 是否走类似路径？）。
- MASTER §1.4（FP8 per_Token V3 kernel）的间接证据更弱。

**建议**：**Exp 6 升级为 P0**。理由：它不是「负面控制锦上添花」，而是「证伪当前修复 scope 假设」的关键实验。若 PASS 才能锁定 Fix 范围只到 tp=4。

#### 2.1.3 其他 CSV spot-check（5.b）应升 P0

**问题**：V07 §A.4 已列出 llama70B / llama405B 在同一 ASM kernel 上有 ≥6 / ≥7 处命中。这些模型如果上线，长序列下会复发 BOS bug。**5.b 是这一风险的唯一定量验证**。

**当前定级 P1 的理由**：「不阻断 V07 PASS」。但 V07 PASS 的范围只是 Step-3.5-Flash —— 项目层面（gfx950 上 bf16 GEMM 整体可用性）这个检查至关重要。

**建议**：
- 在 MASTER 中**增设独立专题 V08-Llama-LongSeq**（V07 §F.5 已建议），把 5.b 移到 V08。
- 或保留在 V07 但升 P0，并在 V07 通过标准里加一条「llama70B/llama405B spot-check 全 PASS 或已登记到上报清单」。

### 2.2 Agent Team 分工方案（含 GPU 分配）

#### V07 最优拆分（6 个 agent，4 阶段）

| Agent | 任务 | GPU 需求 | 阶段 | 可与谁并行 |
|-------|------|---------|------|-----------|
| **agent-V07-A** | Exp 1（tgemm 直调，13 个 M） | 1 GPU | T+0 | B（无 GPU）+ C（4 GPU）+ D（2 GPU） |
| **agent-V07-B** | Exp 5.a（CSV 扫描，无 GPU） | 0 GPU | T+0 | A + C + D |
| **agent-V07-C** | Exp 2（E2E 10k tokens, tp=4） | 4 GPU (0,1,2,3) | T+0 | A + B + D |
| **agent-V07-D** | Exp 6（tp=2 negative control, 10k） | 2 GPU (4,6) | T+0 | A + B + C（避开 GPU 5） |
| **agent-V07-E** | Exp 3（短 prompt 回归，tp=4） | 4 GPU (0,1,2,3) | T+30min（C 完成后） | F（不同卡） |
| **agent-V07-F** | Exp 4（性能 micro-bench） | 1 GPU (7) | T+30min | E |
| **agent-V07-G** | Exp 5.b（其他 CSV spot-check） | 1 GPU (7) | T+45min（F 后） | — |

#### GPU 占用图（避开 GPU 5）

```
Time      GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
T+0       C     C     C     C     D     skip  D     A
T+30min   E     E     E     E     -     skip  -     F
T+45min   -     -     -     -     -     skip  -     G
T+60min   完成
```

**说明**：
- T+0 时 A（GPU 7）+ B（无 GPU）+ C（GPU 0-3）+ D（GPU 4,6）四方并行，**8 GPU 中 7 个被使用**（GPU 5 skip，GPU 7 给 A）。
- E2E 实验（C/E）连续占用 GPU 0-3，必须串行（同卡）。
- Exp 4 / Exp 5.b（F/G）只需单 GPU，挤在 GPU 7 上串行。

总耗时约 60 分钟（vs 串行的 ~3 小时）。

### 2.3 准确性风险

#### 2.3.1 diff < 5 阈值的来源

**问题**：V07 §B 实验 1 通过标准「diff < 5」缺乏定量来源。给的解释是「bf16 量化噪声上界」，但：
- BF16 GEMM 在 M=16384, K=2048 下的理论最大累积误差是 `O(K * eps_bf16 * max(|x|*|w|))`。对 randn 输入，max(|x|)≈4, max(|w|)≈4，eps_bf16≈2^-8。理论 max abs err ≈ 2048 * 2^-8 * 16 = 128。所以 diff < 128 才是宽松上界，diff < 5 是**比理论紧 25 倍**的标准。
- 修复前 diff = 197-392，比理论上界还高 1.5-3 倍 —— 说明 buggy kernel 是「数量级错」而非「精度边界」。
- 修复后 torch.mm 的 diff 实测可能远小于 5（≈ 0.1-1），所以 5 是个**过宽**的判定阈值。

**建议**：把 diff < 5 拆成两档：
- **strict**：diff < 1（torch.mm 期望 diff）
- **loose**：diff < 50（明确区别于 buggy 路径的 197+）

若实测 strict FAIL 但 loose PASS，说明 fallback 走了某个非标准 BF16 GEMM 实现，需进一步追踪 dispatch。

#### 2.3.2 first_token ≠ 0 标准过弱

**问题**：V07 §B 实验 2 通过标准之一「first_token ≠ 0」可被假阳性满足：
- 假设 buggy kernel 让 first_token = 1（不等于 0）但 token 2-10 全是垃圾，实验照样 PASS。
- diversity > 1 也只要求两个 token 不同，不能排除 [3648, 0, 0, 0, 0, 0, 0, 0, 0, 0]。

**建议**：通过标准升级为：
1. `first_token == 3648`（具体 ID，对应 "好的"，已知正确值）—— 但这要求训练后输出确定。
2. `len(set(token_ids)) >= 5`（10 个 token 至少 5 个不同）。
3. `0 not in token_ids[1:]`（前 10 token 中除首位外不应再出现 BOS）。

#### 2.3.3 dispatcher 防御性 check 方案 A 与 csv 删行的取舍

V07 §C.3 第 2 条提出 `BUGGY_ASM_KERNELS` 黑名单方案。这个方案比 csv 删行更通用（保留 M=16384 真值的 ASM 加速），但 V07 仅作为「待确认问题 4」未列入 P0/P1。

**建议**：在 MASTER §6 Phase 4 checklist 里加 8.6「评估 dispatcher 防御性 check 是否替代 csv 删行」，并明确触发条件：若 V07 实验 4 显示 torch.mm 在 M=16384 比 ASM 慢 > 1.5×，则启动方案 A 设计。

---

## 3. MASTER_PIPELINE 整体审查

### 3.1 GPU 资源分配方案（Phase 2 + Phase 3）

#### Phase 2 GPU 调度（V02 + V03 + V04 并行）

**总需求**：V02（op_test 1-2 GPU）+ V03（kernel test 1-2 GPU）+ V04 tp=4（4 GPU）+ V04 tp=2 回归（2 GPU）= 最高同时占用 9 GPU，超出 7 可用（避开 GPU5）。

**冲突解决方案**：

| 时间窗 | V02 agent | V03 agent | V04 agent | 总占用 |
|--------|-----------|-----------|-----------|--------|
| T+0~30min | op_test M=32~256 (GPU 7) | ctx sweep cos_sim (GPU 6) | tp=4 端到端 (GPU 0,1,2,3) + tp=2 回归 (GPU 4) | 7/8 |
| T+30~60min | layer 验证 (GPU 7) | decode kernel (GPU 6) | inter sweep (GPU 4) + tp=8 单算子降级 (GPU 0,1,2,3) | 7/8 |
| T+60~90min | E2E max_tokens=128 (GPU 0,1) | E2E 去 workaround (GPU 2,3) | ca_comm fallback (GPU 4) | 5/8 |

**说明**：
- V04 tp=4 端到端是关键路径，独占 GPU 0-3 一整段。
- V02 / V03 op_test 用单 GPU（6 或 7），与 V04 不抢资源。
- T+60min 后 V04 端到端释放 GPU 0-3，V02/V03 的 E2E 才能上。

#### Phase 3 GPU 调度（V05 + V06 + V07 并行）

| 时间窗 | V05 agent | V06 agent | V07 agent | 总占用 |
|--------|-----------|-----------|-----------|--------|
| T+0~30min | FP8 tp=2 端到端 (GPU 0,1) | FP8 tp=4 端到端 (GPU 2,3,4,6) | Exp 5.a CSV 扫描（无 GPU）+ Exp 1 tgemm (GPU 7) | 7/8 |
| T+30~60min | BF16 tp=2 回归 (GPU 0,1) | scale dump Exp 1a/b (GPU 2,3,4,6) | Exp 6 tp=2 长序列 (GPU 7 单卡跑不动 → 推迟) | 5/8 |
| T+60~90min | Exp 1 crash 复现 (GPU 0,1，**独占 aiter 源码**) | Exp 5 gibberish 复现 (GPU 2,3,4,6，**独占 ATOM 源码**) | Exp 2 E2E 10k (GPU ?) | 冲突 |

**Phase 3 关键冲突**：
- V05 Exp 1 修 aiter `fused_moe.py`；V06 Exp 5 修 ATOM `moe.py`；V07 不修源码。三者源码层面无冲突，可并行。
- V07 Exp 2 需要 4 GPU（tp=4），但 V06 占了 GPU 2,3,4,6 → V07 Exp 2 必须等 V06 完成或共享时间窗。

**修订建议**：把 V07 Exp 2 移到 Phase 3 的最末段（T+90min~120min），或与 V06 Exp 2 合并成同一个 tp=4 端到端测试（FP8 + BF16 各跑一次）。

### 3.2 关键 checklist 遗漏

#### 3.2.1 Phase 0 缺 FP8 模型预检

MASTER Phase 0 checklist 包含 0.5 模型缓存预检（`grep -i flash`），但**没有显式检查 FP8 模型可加载**。建议拆分：

```
- [ ] 0.5a BF16 模型缓存：ls ~/.cache/huggingface/hub/ | grep -i flash$
- [ ] 0.5b FP8 模型缓存：ls ~/.cache/huggingface/hub/ | grep -i flash-fp8
- [ ] 0.5c FP8 模型 dry-run：CUDA_VISIBLE_DEVICES=0 python -c "from atom.examples.simple_inference import load_model; load_model('stepfun-ai/Step-3.5-Flash-FP8')"（或类似最小加载脚本）
```

理由：若 FP8 权重未下载，Phase 3 V05/V06 启动时会阻塞首次下载（10-30 min），破坏 Phase 3 并行调度。

#### 3.2.2 Phase 4 checklist 8.5（更新 MEMORY.md）应自动化

MASTER §6 Phase 4 的 8.5 是手动 checklist 项。但 MEMORY.md 已经被 agent-team skill 管理（recall-add），**应该在每个 Phase 通过后自动追加更新条目**，而不是等 Phase 4 统一手动写。

**建议**：在每个 Phase 末尾插入子项：
```
- [ ] X.last  通过 recall-add 写入「{专题} 验证 PASS/FAIL，关键发现」到 MEMORY 对应 topic 文件
```
例如 V05 通过后 → recall-add 写入 `memory/fp8-work.md` 的「§ V05 验证状态（2026-04-25）」。

#### 3.2.3 跨专题问题 1.1 的 action 需扩展

MASTER §6 Phase 0 的 0.4 写的是 `git show 3771835ac --stat`，**只看文件列表**，看不到具体 diff。

**建议**：升级为：
```
- [ ] 0.4a git show 3771835ac --stat  # 文件列表
- [ ] 0.4b git show 3771835ac -- aiter/ | head -200  # aiter 侧 diff
- [ ] 0.4c git show 3771835ac -- ATOM/atom/model_ops/moe.py  # ATOM 侧 diff（确认是否触及 inter_dim padding L489-518）
- [ ] 0.4d 若 0.4c 显示 ATOM padding 也被 revert：阻断 V04 启动，重新评估 V04 §A 假设
```

#### 3.2.4 Phase 4 8.3 应产出 issue 草稿

MASTER 8.3 写「提交 V07 §C.3 的 ASM kernel issue 给 AMD aiter 团队」—— 这是个动作但没有产出物模板。

**建议**：增加 8.3a 子项：「在 `/home/hanchang/project_fp8_tp4/verification_pipeline/issue_drafts/asm_kernel_bug.md` 落地 issue 草稿」，模板包含：
- 标题（V07 §C.3 已给）
- 复现脚本路径（V07 Exp 1 的 `/tmp/v07_exp1_tgemm.py`）
- BAD M 集合（V07 §C.3 已列出）
- 影响模型范围（V07 §A.4 表 + V08 spot-check 结果）
- 临时 workaround commit（`a2883ab37`）

### 3.3 优先级调整建议（哪些 P1 应升 P0）

| 当前 P1 | 建议 | 理由 |
|---------|------|------|
| **V07 Exp 6**（tp=2 negative control） | **升 P0** | 是 V07 §A.5 表格的唯一直接验证；FAIL 会推翻整个 ASM kernel 影响面分析（见 2.1.2） |
| **V07 Exp 5.b**（其他 CSV spot-check） | **升 P0 或独立到 V08** | 关系到其他模型在 gfx950 上的可用性；不验证则 llama 系列上线时复发（见 2.1.3） |
| **V01 Exp 5**（buffer padding canary） | 维持 P1，但增加 Phase 0 阻断条件 | 如 3.2.3 所述，依赖 0.4 结论；若 0.4d 触发则升 P0 |
| **V06 Exp 1c**（extreme oversharding） | 维持 P0（已是） | 验证跨专题问题 1.3 的 mis-broadcast 风险 |
| **V05 Exp 5**（block_shape 区分性） | 维持 P1，但通过标准升级 | 如 1.1.3 所述，「无差异」结论必须配合 dispatch 日志验证 |
| **V02 Exp 6**（cos_sim 衰减曲线） | 条件 P0 | 若 V02 Exp 4 完美复现则 P1；若有异常则 P0（MASTER §1.5 已建议） |

---

## 4. 给 Synthesizer 的关键问题

### 4.1 优先级冲突需裁决

1. **V07 Exp 6 / Exp 5.b 是否升 P0？**（Reviewer-C 强烈建议升）
   - 升 P0 → Phase 3 时间增加 30-60min，但锁定影响面。
   - 维持 P1 → 节省时间，但留下「tp=2/llama 可能受影响」的盲区。

2. **V08-Llama-LongSeq 是否独立专题？**
   - 独立 → MASTER 需扩展到 8 个专题，工时增 4-6h。
   - 不独立 → V07 §F.5 标注的 5-7 个 hit 永远停留在「待确认」状态。

### 4.2 GPU 调度策略需确认

3. **Phase 2 / Phase 3 的 GPU 分配方案（见 3.1）是否被各专题 agent 接受？**
   - V04 端到端独占 GPU 0-3 是否阻断 V02/V03 的 E2E 实验？
   - V07 Exp 2 推迟到 Phase 3 末段是否影响整体进度？

4. **V05/V06 的源码独占性如何调度？**
   - V05 Exp 1（独占 aiter）+ V06 Exp 5（独占 ATOM）是否需要全局锁机制？
   - 各 agent 修改前后必须执行的 git status 检查是否纳入 SKILL 标准流程？

### 4.3 通过标准量化需统一

5. **diff < 5 / first_token ≠ 0 / cos_sim ≥ 0.995 这些数值阈值的定量来源**
   - 当前各专题阈值各自给定，缺统一定标。
   - 建议 Synthesizer 出一份「数值精度阈值表」（见 2.3.1 / 2.3.2），列出每个标准的理论上界与经验下界。

6. **「无差异」类结论（V05 Exp 5、V06 Exp 4）的可接受性**
   - 当前允许「无差异 = PASS」，但 1.1.3 指出可能掩盖 silent fallback。
   - 是否要求所有「无差异」结论必须配 dispatch 日志佐证？

### 4.4 自动化与 traceability

7. **MEMORY.md 更新的自动化触发点**（见 3.2.2）
   - 是否在每个 Phase 末段自动调用 recall-add？
   - 如何避免重复写入或冲突？

8. **issue 草稿（V07 → AMD）的产出位置与格式**（见 3.2.4）
   - 是否在 `verification_pipeline/issue_drafts/` 下集中？
   - 草稿是否需要在 Phase 4 完成时自动生成（基于 V07 测试结果）？

---

## 5. 审查总结

**三份文档总体合格**，可作为执行依据。但执行前必须解决以下 5 个高优先项：

1. **MASTER Phase 0 扩展**（3.2.1 + 3.2.3）：FP8 模型 dry-run + 3771835ac diff 详查 —— 否则 V04 / V05 启动可能 stall。
2. **V07 Exp 6 升 P0**（2.1.2）：tp=2 negative control 是影响面假设的关键证据。
3. **V07 Exp 5.b 决策**（2.1.3）：升 P0 或开 V08。
4. **GPU 调度方案落地**（3.1）：Phase 2 三专题 + Phase 3 三专题的具体卡分配，需各专题 agent 团队接受。
5. **V05 Exp 1 与源码独占协议**（1.1.2 + 1.3.2）：JIT cache 清理 + git status 守卫 + agent 互斥锁。

**不阻断但应处理的优化项**：
- diff/cos_sim 阈值统一定标（4.3）
- MEMORY.md 自动更新（3.2.2）
- issue 草稿模板（3.2.4）
- V05 guard 反向覆盖断言（1.1.1）

---

Reviewer-C 审查报告完成。
