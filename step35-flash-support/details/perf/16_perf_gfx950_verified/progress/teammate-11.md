# #801 aiter GEMM tuning 全量盘点

调查范围：`/home/hanchang/junlin12_repos/aiter/aiter/configs/` 下所有 CSV 文件（共 70 个，其中 35 个 *_tuned_*.csv）。

## 文件总览（仅列出含 gfx950 / gfx942 或 cu_num 条目的 CSV）

注：dense GEMM CSV 第一列是 `gfx`（直接含 "gfx950"/"gfx942" 字符串）；fmoe CSV 第一列是 `cu_num`，**256 ≈ gfx950（MI355X）, 80 ≈ gfx942（MI300X）**。

### Dense GEMM CSV（按 `gfx` 列）

| 文件 | gfx950 | gfx942 | 总行数 |
|------|--------|--------|--------|
| a4w4_blockscale_tuned_gemm.csv | 1470 | 0 | 1471 |
| a8w8_blockscale_bpreshuffle_tuned_gemm.csv | 58 | 0 | 59 |
| a8w8_blockscale_tuned_gemm.csv | 5 | 0 | 6 |
| a8w8_bpreshuffle_tuned_gemm.csv | 481 | 69 | 551 |
| a8w8_tuned_batched_gemm.csv | 0 | 26 | 27 |
| a8w8_tuned_gemm.csv | 556 | 26 | 583 |
| bf16_tuned_batched_gemm.csv | 0 | 26 | 27 |
| bf16_tuned_gemm.csv | 0 | 0 | 1 (header only) |
| model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_dsv3.csv | 720 | 0 | 721 |
| model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_qwen3.5_397b.csv | 553 | 0 | 554 |
| model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_qwen3_235b.csv | 87 | 0 | 88 |
| model_configs/a8w8_blockscale_tuned_gemm_ds_v3.csv | 1020 | 0 | 1021 |
| model_configs/a8w8_blockscale_tuned_gemm_qwen3_235b.csv | 128 | 0 | 129 |
| model_configs/a8w8_blockscale_tuned_gemm_qwen3_5_397b_a13b.csv | 100 | 0 | 101 |
| model_configs/dsv3_a4w4_blockscale_tuned_gemm.csv | 30 | 0 | 31 |
| model_configs/dsv3_a8w8_bpreshuffle_tuned_gemm.csv | 131 | 1403 | 1535 |
| model_configs/dsv3_bf16_tuned_gemm.csv | 58 | 0 | 59 |
| model_configs/glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv | 80 | 0 | 81 |
| model_configs/glm5_bf16_tuned_gemm.csv | 71 | 0 | 72 |
| model_configs/gptoss_bf16_tuned_gemm.csv | 57 | 0 | 58 |
| model_configs/kimik2_bf16_tuned_gemm.csv | 125 | 0 | 126 |
| model_configs/llama405B_bf16_tuned_gemm.csv | 156 | 0 | 157 |
| model_configs/llama70B_bf16_tuned_gemm.csv | 156 | 0 | 157 |
| model_configs/qwen32B_bf16_tuned_gemm.csv | 156 | 0 | 157 |

### Fmoe CSV（按 cu_num 列）

| 文件 | cu80 (gfx942) | cu256 (gfx950) | 总行数 |
|------|---------------|----------------|--------|
| tuned_fmoe.csv | 947 | 751 | 1699 |
| model_configs/a8w8_blockscale_tuned_fmoe_ds_v3.csv | 0 | 16 | 16 |
| model_configs/a8w8_blockscale_tuned_fmoe_glm5.csv | 0 | 16 | 16 |
| model_configs/a8w8_blockscale_tuned_fmoe_minimax-m2_5.csv | 0 | 32 | 32 |
| model_configs/a8w8_blockscale_tuned_fmoe_qwen3_235b.csv | 0 | 32 | 32 |
| model_configs/a8w8_blockscale_tuned_fmoe_qwen3_5_397b.csv | 0 | 16 | 16 |
| model_configs/dsv3_fp4_tuned_fmoe.csv | 0 | 49 | 49 |
| model_configs/gptoss_fp8fp4_tuned_fmoe.csv | 0 | 14 | 14 |
| model_configs/kimik2_fp4_tuned_fmoe.csv | 0 | 128 | 128 |
| model_configs/kimik2_fp8fp4_tuned_fmoe.csv | 0 | 64 | 64 |
| model_configs/minimax_m25_fp4_tuned_fmoe.csv | 0 | 64 | 64 |
| model_configs/qwen3_5_397b_fp4_tuned_fmoe.csv | 0 | 32 | 32 |

---

## BF16 tuning（bf16_tuned_gemm.csv 系列）

### 关键观察
- **基础 `bf16_tuned_gemm.csv` 是空文件**（仅 header）。所有 BF16 tuning 都在 model_configs/ 下分模型。
- **没有 Step-3.5-Flash 专属的 bf16 CSV。**
- 现存所有 BF16 tuned 文件 **gfx 列均只含 gfx950（无 gfx942）**。

### gfx950 BF16 (N,K) 覆盖（按文件汇总）

| 文件 | (N,K) 形状 | M 值 |
|------|------------|------|
| dsv3_bf16_tuned_gemm.csv | (256,7168), (2112,7168), (16160,7168), (3072,1536), (7168,2048) | 1..256 |
| glm5_bf16_tuned_gemm.csv | (128,6144), (256,6144), (2624,6144), (38720,6144), (4096,2048), (6144,3072), (6144,4096), (6144,6144) | 1..256, 16384 |
| gptoss_bf16_tuned_gemm.csv | (128,2880), (2560,2880), (2880,2048), (2880,4096), (5120,2880) | 1..256 |
| kimik2_bf16_tuned_gemm.csv | (1024,7168), (2112,7168), (384,7168), (4096,512), (7168,512) | 1..512 (步长 8) |
| llama405B_bf16_tuned_gemm.csv | 12 个形状（多 K=16384, K=4096 等） | 1..32768 |
| llama70B_bf16_tuned_gemm.csv | 13 个形状（K=8192 主） | 1..32768 |
| qwen32B_bf16_tuned_gemm.csv | 12 个形状（K=5120 主） | 1..32768 |

### gfx942 BF16 覆盖
**完全没有。** 所有 *_bf16_tuned_gemm.csv 都只有 gfx950。

### Step-3.5-Flash 关键 BF16 形状是否覆盖

针对 N ∈ {5120, 7168, 4096, 11264, 1280, 64448}, K=4096 系统搜索所有 *bf16*tuned*.csv：

| Step-3.5-Flash 形状 | 是否在任何 BF16 CSV | 备注 |
|---------------------|--------------------|------|
| (5120, 4096) | ❌ 无 | qwen32B 有 (5120,*)，但 K≠4096 |
| (7168, 4096) | ❌ 无 | dsv3 仅有 (7168,2048) |
| (4096, 4096) | ❌ 无 | glm5 有 (4096,2048) |
| (11264, 4096) | ❌ 无 | — |
| (1280, 4096) | ❌ 无 | llama70B 有 (1280,8192) |
| (64448, 4096) | ❌ 无 | — |

M 维度：M=10262 也不在任何 BF16 文件的覆盖列表里（只有 llama/qwen 文件 M 跨度到 32768，但 (N,K) 不匹配）。

**结论：Step-3.5-Flash 所有 BF16 prefill 关键形状在 aiter 中 0 命中**，与 #501 / #702 结论一致。

---

## FP8 fmoe tuning（tuned_fmoe.csv 系列）

### gfx950 (cu_num=256) 覆盖（tuned_fmoe.csv 主文件）
unique (inter_dim, expert) 仅 5 组：
- (192, 128), (256, 256), (384, 128), (512, 256), (1024, 128)

### gfx942 (cu_num=80) 覆盖（tuned_fmoe.csv 主文件）
unique (inter_dim, expert) 8 组：
- (192, 128), (256, 256), (256, 257), (384, 128), (512, 256), (1536, 8), (2048, 33), (4096, 8)

### 模型专属 fmoe CSV
- glm5 fmoe：仅 (inter_dim=256, expert=257)
- ds_v3 fmoe：少数 dsv3 形状
- qwen / minimax / kimik2 / gptoss 各自的形状
- **没有 stepfun / step3 / step35 的 fmoe CSV。**

### Step-3.5-Flash 关键 fmoe key tuple 是否覆盖

针对 inter_dim ∈ {640, 384}, expert ∈ {288, 289} 系统搜索所有 *fmoe*.csv：

| Step-3.5-Flash 关键 tuple | 是否覆盖 |
|---------------------------|----------|
| inter_dim=640, expert=288 | ❌ 任何 cu_num、任何 CSV 都没有 |
| inter_dim=640, expert=289 | ❌ 同上 |
| inter_dim=384, expert=288 | ❌ 同上（(384,128) 有，但 expert 不匹配） |
| inter_dim=384, expert=289 | ❌ 同上 |

`inter_dim=640` 在所有 fmoe CSV 中 0 出现；`expert=288/289` 在所有 fmoe CSV 中 0 出现。

**结论：Step-3.5-Flash 的 routed-expert 形状在 aiter fmoe 中完全无 tuning，与 #601 结论一致。**

---

## FP8 dense GEMM tuning（a8w8_blockscale 系列）

### a8w8_blockscale_bpreshuffle_tuned_gemm.csv（基础）
- gfx950: 58 条；gfx942: 0
- (N,K) 形状：10 组，全是 dsv3 风格的 K=7168/1536（如 (1536,7168), (4608,7168), (7168,2048) 等）

### a8w8_blockscale_tuned_gemm.csv（基础非 bpreshuffle）
- gfx950: 5 条；gfx942: 0
- 仅 5 行：包含 (512,7168), (1024,4096), (4096,1280)

### model_configs/glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv
- gfx950: 80；gfx942: 0
- (N,K)：(128,6144), (2624,6144), (3072,6144), (3584,512), (6144,1536) — GLM 形状

### model_configs/dsv3_a8w8_bpreshuffle_tuned_gemm.csv
- **gfx942: 1403, gfx950: 131** —— 唯一一个 gfx942 显著多于 gfx950 的文件
- gfx942 覆盖 43 组 (N,K)；gfx950 仅 8 组 (N,K)（且 gfx950 都是 gfx942 子集）

### model_configs/a8w8_blockscale_tuned_gemm_ds_v3.csv
- gfx950: 1020；gfx942: 0
- 34 组 (N,K)，包含 (7168,4096), (4096,7168) 等 dsv3 dense 形状

### Step-3.5-Flash FP8 dense 关键形状（N=*, K=4096）覆盖
搜索所有 *a8w8*tuned*.csv：

| 形状 | 命中文件 |
|------|----------|
| (5120, 4096) | qwen3.5_397b bpreshuffle (16 entries) |
| (4096, 4096) | qwen3_235b bpreshuffle/non-bpreshuffle (各 16) |
| (1280, 4096) | qwen3_235b bpreshuffle/non-bpreshuffle (各 16) |
| (7168, 4096) | dsv3 ds_v3 (34) + dsv3_bpreshuffle (48) |
| (11264, 4096) | ❌ 无 |
| (64448, 4096) | ❌ 无 |

注意：FP8 dense GEMM **走的是 a8w8_blockscale_bpreshuffle 而非 _bpreshuffle 后缀**（参见 ATOM 调用路径）；上面命中的形状即便存在，也是 qwen / dsv3 专属 CSV，且 ATOM 端 dispatch 是否会复用其他模型的 tuning 表需另查。base `a8w8_blockscale_bpreshuffle_tuned_gemm.csv` 中没有 Step-3.5-Flash 任何关键 (N,4096) 形状。

---

## gfx942 vs gfx950 全局对比

| 类别 | gfx942 在哪些文件多 | gfx950 在哪些文件多 |
|------|---------------------|---------------------|
| dense FP8 a8w8 (rowwise) | a8w8_tuned_gemm (gfx942=26 vs gfx950=556) — gfx950 多 | gfx950 远多 |
| dense FP8 a8w8 bpreshuffle | a8w8_bpreshuffle_tuned_gemm (g942=69, g950=481) — gfx950 多 | gfx950 多 |
| dense FP8 a8w8 batched | a8w8_tuned_batched_gemm: g942=26, g950=0 — **gfx942 唯一** | — |
| dense BF16 batched | bf16_tuned_batched_gemm: g942=26, g950=0 — **gfx942 唯一** | — |
| dsv3 FP8 bpreshuffle | dsv3_a8w8_bpreshuffle: g942=1403 vs g950=131 — **gfx942 显著多** | — |
| 其余所有 model_configs（包括所有 BF16 模型 CSV、所有 a8w8_blockscale 模型 CSV、所有 fp4） | 0 | 全部 gfx950 独占 |

### gfx942 独有 / 显著占优的 tuning 是否能解释 Step-3.5-Flash 性能差异？

1. **bf16_tuned_batched_gemm.csv（仅 gfx942 26 条）**：batched GEMM，与 Step-3.5-Flash prefill/decode 单 GEMM 路径无关，不解释。
2. **a8w8_tuned_batched_gemm.csv（仅 gfx942 26 条）**：同上，且 Step-3.5-Flash 用的是 a8w8_blockscale 而非 a8w8 rowwise，不解释。
3. **dsv3_a8w8_bpreshuffle_tuned_gemm.csv（gfx942 多 1272 条）**：DeepSeek-V3 专属 K=7168/1536 等形状，不属于 Step-3.5-Flash 的 (N,K=4096) 形状集，不解释。
4. **a8w8_tuned_gemm.csv / a8w8_bpreshuffle_tuned_gemm.csv 中的 gfx942 条目**：N 多在 1280/8192/5120 区间（K=8192/1024/5120），同样不覆盖 Step-3.5-Flash (N, K=4096) 形状。

---

## 结论

1. **gfx942 没有任何能直接覆盖 Step-3.5-Flash 关键 (N,4096) 形状的 tuning。** gfx942 仅在 dsv3 / a8w8 batched / a8w8 rowwise 上有独立 tuning，全部不命中 Step-3.5-Flash。
2. **gfx950 也不覆盖 Step-3.5-Flash：**
   - BF16：所有 *bf16_tuned_gemm.csv 中 0 命中 (N, K=4096) 关键形状；M=10262 在任何文件都无 entry。
   - FP8 fmoe：inter_dim=640、expert∈{288,289} 在任何 fmoe CSV、任何 cu_num 下都为 0。
   - FP8 dense：base `a8w8_blockscale_bpreshuffle_tuned_gemm.csv` 仅 dsv3 形状；qwen3 系列偶有命中但属于其他模型专属 CSV。
3. **gfx942 vs gfx950 性能差异不能由 "gfx942 有 gfx950 没有的 tuning" 来解释**——gfx942 仅在 dsv3/batched/rowwise 上多，没有任何条目覆盖 Step-3.5-Flash 形状。
4. **真正的根因仍是 #501/#601/#702 的结论**：Step-3.5-Flash 在 gfx950 上 BF16 与 FP8 fmoe 都 0 tuning 命中，全部 fallback 到 default 路径，而非 gfx950 缺少与 gfx942 平行的 tuning 表。
5. **修复方向**：为 Step-3.5-Flash 单独 tune 一份 `step35_bf16_tuned_gemm.csv` 与 `a8w8_blockscale_tuned_fmoe_step35.csv`，覆盖 (N∈{5120,7168,4096,11264,1280,64448}, K=4096) × M={1,10262,16384,...} 与 (inter_dim=640, expert∈{288,289}) 的 token 扫描表，比尝试复用 dsv3/qwen3 tuning 更可靠。

---

## 数据出处（全部来自实际文件）
- 文件清单：`find /home/hanchang/junlin12_repos/aiter/aiter/configs -name "*.csv"` 共 70 个
- 计数方法：`awk -F',' '$1==X' file | wc -l`（dense GEMM 用 gfx 字符串匹配；fmoe 用 cu_num 数值匹配）
- 形状提取：`awk -F',' '{print $4","$5}' | sort -u`（dense）/`$4,$5` 为 inter_dim/expert（fmoe）
