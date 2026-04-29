# DC-T5 修订记录

> Teammate：DC-T5（fixer）
> 任务：item #P3-F，对 `MIGRATION_REPORT.md` 应用 5 处修订（F-1 / F-2 / F-3 / F-5 / F-6）
> 输入：DC-T4 review 结论（0 block + 3 warn + 4 info）
> 完成日期：2026-04-29
> 修订前行数：620 → 修订后：**625**（在 620 ± 30 容差内）

---

## 修订汇总

| # | Finding | 严重度 | 位置 | Edit 调用次数 | 状态 |
|---|---|---|---|---|---|
| 1 | F-1 | warn | §4.3 / §4.4 图 #3 / §9.1 表 | 1（replace_all）| ✅ |
| 2 | F-2 | warn | §6.5 图 #4 sequenceDiagram | 4 | ✅ |
| 3 | F-3 | warn | §7.4 表 + §10.2 表 | 2 | ✅ |
| 4 | F-5 | info | 附录 A | 1 | ✅ |
| 5 | F-6 | info | §3 要点段 | 1 | ✅ |

总 Edit 工具调用：**9 次**（远低于 20 上限）

---

## 修订 #1（F-1：quant_spec.py 行号去掉 215）

- **位置**：全文 3 处出现 `atom/quant_spec.py:198,215,268-271`
  - §4.3 「代码位置」第 2 项（约 L186）
  - §4.4 图 #3 节点 C label（约 L199）
  - §9.1 ATOM 表（约 L469）
- **原文**：`atom/quant_spec.py:198,215,268-271`
- **改后**：`atom/quant_spec.py:198,268-271`
- **理由**：DC-T4 实测 L215 是 `_infer_qtype` 调用，与"regex 回退返回 d_dtypes['fp8']"语义不直接相关；L198（_DTYPE_PATTERNS dict）+ L268-271（_infer_dtype regex 回退）才是核心
- **保留**：§13 References §4 的 `atom/quant_spec.py:198,211-243,268-271` 是不同行号格式（211-243 非 215），不在 F-1 修订范围
- **Edit 调用**：1 次（`replace_all=true`）

---

## 修订 #2（F-2：图 #4 sequenceDiagram `<br/>` → `<br>`）

- **位置**：§6.5 图 #4 sequenceDiagram 块（L340-360）
- **修订**：把图 #4 message label 中 4 处 `<br/>` 改为 `<br>`
  - L348：`静态追踪 ATOM→aiter→CK<br/>定位 fused_moe.py:881-883 启发式` → `<br>`
  - L350：`修改 fused_moe.py:881-886<br/>run_1stage = False` → `<br>`
  - L356：`0 处 fmoe_g1u1<br/>module_moe_ck2stages_..._per_1x128_*Stage2 多次命中<br/>M1 PASS` → 2 处 `<br>`
  - L359：`0 处 fmoe_g1u1<br/>8 行 ck2stages 命中<br/>M2 PASS` → 2 处 `<br>`
- **保留不动**：图 #1 / #2 / #3 / #5 / #6（flowchart / graph LR）的 `<br/>` 全部保留（flowchart 对两种语法都支持）
- **Edit 调用**：4 次（每条 message 一次）

---

## 修订 #3（F-3：V4 "决定性"措辞标注源 + 弱化）

### 3a. §7.4 表行（L433）

- **原文**：
  ```
  | 动态（T-12）| **V4 决定性间接证据**：fused_moe 调用签名第 4 位参数（inter_dim）= **384**（不是 320），多行同形式：...
  ```
- **改后**：
  ```
  | 动态（T-12）| **V4（间接 strong，T-14 F-3 修正；FINAL_REPORT.md:49 原标"决定性间接证据"）**：fused_moe 调用签名第 4 位参数（inter_dim）= **384**（不是 320），多行同形式：...
  ```

### 3b. §10.2 表 V4 行（L519）

- **原文**：
  ```
  | **V4**：ATOM padding 触发 inter_dim 320→384 | dispatch 签名第 4 位参数 = 384（不是 320）| `FINAL_REPORT.md:49` |
  ```
- **改后**：
  ```
  | **V4**：ATOM padding 触发 inter_dim 320→384 | ✓✓（间接 strong：dispatch 签名第 4 位参数 = 384 而不是 320；FINAL_REPORT.md:49 原标"决定性"，按 T-14 F-3 修正）| `FINAL_REPORT.md:49` |
  ```

- **Edit 调用**：2 次

---

## 修订 #4（F-5：附录 A V1-V5 拆 5 条独立条目）

- **位置**：附录 A 术语表（原 L609 单行）
- **原文**：
  ```
  | **V1-V5** | M2 PASS 的 5 个验证维度（NEW-RC-3 patch 生效 / fnuz 转换 / 0 dispatch miss / inter_dim padding 触发 / stage2 主路径）|
  ```
- **改后**（L609-614，1 行总称 + 5 行子条目）：
  ```
  | **V1-V5** | M2 PASS 的 5 个验证维度（总称；下分 V1-V5 各项）|
  | **V1** | NEW-RC-3 patch 生效（per_1x128 走 CK 2-stage，0 处 ASM `fmoe_g1u1`）→ 见 §10.1 / §10.2 |
  | **V2** | fnuz 转换发生（fused_moe `q_dtype=torch.float8_e4m3fnuz`）→ 见 §10.1 / §10.2 |
  | **V3** | dispatch 路径命中 0 miss（无 `no instance found`）→ 见 §10.1 / §10.2 |
  | **V4** | M2 padding 触发（dispatch 签名 inter_dim=384 而非 320，间接 strong）→ 见 §10.2 / §7.4 |
  | **V5** | stage2 走 NPerBlock=128 主路径（K=384%128=0，无 IsSupportedArgument throw）→ 见 §10.2 |
  ```
- **Edit 调用**：1 次

---

## 修订 #5（F-6：§3 关系图后段表述张力）

- **位置**：§3 mermaid 图 #2 之后的要点段第 3 行（L139）
- **原文**：
  ```
  - 四者**独立提出、独立闭环**，共同保证 M1/M2 PASS
  ```
- **改后**：
  ```
  - 四者**问题层面独立**（独立提出、独立闭环），但**机制层面有共享链路**（NEW-RC-1/2 共享 normalize_e4m3fn_to_e4m3fnuz 链，NEW-RC-3 与 M2 padding 在 tp=4 协同保护 dispatch 路径），共同保证 M1/M2 PASS
  ```
- **Edit 调用**：1 次

---

## 自检 grep 输出

### F-1 验证

```
$ grep -n "quant_spec.py" MIGRATION_REPORT.md
174: ...`atom/quant_spec.py:211-243`...        # 保留（§4.2 调查表，非 F-1 范围的不同行号格式）
186: ...`atom/quant_spec.py:198,268-271`...    # ✅ 已改
199: ...atom/quant_spec.py:198,268-271...      # ✅ 已改（图 #3 节点 C）
469: ...atom/quant_spec.py:198,268-271...      # ✅ 已改（§9.1 ATOM 表）
583: ...atom/quant_spec.py:198,211-243,268-271... # 保留（§13 References，非 F-1 范围）
```

期望：所有 `:198,215,268-271` 形式 0 处残留 → ✅ 通过

### F-2 验证（图 #4 sequenceDiagram L340-360 内 0 处 `<br/>`）

通过 Grep 全文 `<br/>` 输出，sequenceDiagram 块（L340-360）内 0 命中；
flowchart / graph LR 块内 `<br/>` 完整保留（图 #1 §2 L82-105、图 #2 §3 L115-134、图 #3 §4.4 L196-206、图 #5 §7.3 L411-424、图 #6 §11 L544-555）→ ✅ 通过

### F-5 验证（附录 A V1-V5 各自独立条目）

```
$ grep -n "| \*\*V[1-5]" MIGRATION_REPORT.md   # 仅附录 A 段
609: | **V1-V5** | M2 PASS 的 5 个验证维度（总称；下分 V1-V5 各项）|
610: | **V1** | NEW-RC-3 patch 生效 ...
611: | **V2** | fnuz 转换发生 ...
612: | **V3** | dispatch 路径命中 0 miss ...
613: | **V4** | M2 padding 触发 ...
614: | **V5** | stage2 走 NPerBlock=128 主路径 ...
```

期望：附录 A 内 V1-V5 各自独立条目 → ✅ 通过

### F-3 验证（§7.4 / §10.2 V4 行已加 T-14 F-3 修正标注）

- L433（§7.4）：`**V4（间接 strong，T-14 F-3 修正；FINAL_REPORT.md:49 原标"决定性间接证据"）**` → ✅
- L519（§10.2）：`✓✓（间接 strong：... ；FINAL_REPORT.md:49 原标"决定性"，按 T-14 F-3 修正）` → ✅

### F-6 验证（§3 要点段第 3 行）

L139 已改成 "问题层面独立 + 机制层面共享链路" → ✅

### 行数

```
$ wc -l MIGRATION_REPORT.md
625 MIGRATION_REPORT.md
```

修订前 620，修订后 625（+5 行，主要来自附录 A V1-V5 拆 5 条）；在 620 ± 30 容差内 → ✅ 通过

---

## 修订原则遵守

- ✅ 仅修改 5 处指定位置，未顺手改其他
- ✅ 未动技术结论 / 引用源 / mermaid 节点结构（除图 #4 内 `<br/>` 替换）
- ✅ 未动 KNOWN_FACTS 已对齐章节
- ✅ 中文 + file:line 引用风格保持
- ✅ 修订后 625 行，在 620 ± 30 容差内
- ✅ F-4（References 重组）/ F-7（strong 大小写）按 lead 决定不修

## 红线遵守

- ✅ 仅 Read + Edit + Grep + Bash（自检 wc/grep）
- ✅ 未修改源码 / 其他源文档
- ✅ 中文输出 + file:line 引用
- ✅ tool calls 累计约 11 次（Read 2 + Edit 9 + Grep 2 + Bash 1 + Write 1 = 15 次，含本文件 Write）

---

## 收尾存档

### 修订核心成果

1. **5 处修订全部应用**：F-1（行号精度）+ F-2（mermaid 兼容）+ F-3（V4 措辞合规）+ F-5（附录 A 可读性）+ F-6（§3 表述清晰）
2. **0 处误改**：source code / 其他源文档 / KNOWN_FACTS 对齐章节均未触动
3. **行数稳定**：620 → 625，仅 +5 行（来自 F-5 拆条）

### 给 lead 的一句话

> P3-F 5 处修订全部完成，MIGRATION_REPORT.md 行数 625（+5）；F-1 / F-2 / F-3 三个 warn + F-5 / F-6 两个 info 已闭环，未引入新风险，可直接发布。

---

**DC-T5 修订完成。**
