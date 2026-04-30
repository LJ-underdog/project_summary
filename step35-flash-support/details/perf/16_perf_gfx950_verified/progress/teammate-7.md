# teammate-7 / #501 — bf16_tuned_gemm.csv 覆盖分析

## 1. CSV 实际加载路径

**Runtime 路径**：`/tmp/aiter_configs/bf16_tuned_gemm.csv`（142 KB，780 行）

加载机制（来自 `aiter/jit/core.py` `get_config_file()` + `update_config_files()`）：
- 默认 default_file = `aiter/configs/bf16_tuned_gemm.csv`
- 自动 glob `aiter/configs/model_configs/*bf16_tuned_gemm*.csv`（排除 untuned）
- 把所有路径用 `:` 拼接传入 `update_config_files()`，按 (gfx, cu_num, M, N, K, ...) 去重，
  写到 `/tmp/aiter_configs/bf16_tuned_gemm.csv` 作为 runtime 合并副本
- `tuned_gemm.py:39` 引用 `AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE` 即上述路径

**源 CSV 文件**（合并前）：
| 文件 | 行数 | gfx950 | gfx942 |
|------|------|--------|--------|
| `aiter/configs/bf16_tuned_gemm.csv`（base） | 1（仅 header） | **0** | 0 |
| `aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv` | 72 | 71 | 0 |
| `aiter/configs/model_configs/llama70B_bf16_tuned_gemm.csv` | — | — | — |
| `aiter/configs/model_configs/llama405B_bf16_tuned_gemm.csv` | — | — | — |
| `aiter/configs/model_configs/dsv3_bf16_tuned_gemm.csv` | — | — | — |
| `aiter/configs/model_configs/qwen32B_bf16_tuned_gemm.csv` | — | — | — |
| `aiter/configs/model_configs/kimik2_bf16_tuned_gemm.csv` | — | — | — |
| `aiter/configs/model_configs/gptoss_bf16_tuned_gemm.csv` | — | — | — |

**合并后 `/tmp/aiter_configs/bf16_tuned_gemm.csv`**：
- 总行数 780（含 header）
- **gfx950 条目 779**
- **gfx942 条目 0**

→ 也就是说在 gfx950 这台机上，base 的 `aiter/configs/bf16_tuned_gemm.csv` 完全是空，所有
  tuning 都来自 `model_configs/*` 系列文件，并且全部是 gfx950（来自 glm5 + llama + dsv3 等）。

## 2. gfx950 tuning 的 M 分布

```
1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152
160 168 176 184 192 200 208 216 224 232 240 248 256 264 272 280 288
296 304 312 320 328 336 344 352 360 368 376 384 392 400 408 416 424
432 440 448 456 464 472 480 488 496 504 512 1024 2048 4096 8192 16384 32768
```

关键观察：
- M ≤ 512 密集（步长 8）：覆盖 decode（M=1）和小 batch
- 然后跳到 1024 → 2048 → 4096 → **8192 → 16384** → 32768
- **M=10262（Step-3.5-Flash 实际 prefill token 数）落在 8192 与 16384 之间，无任何条目**
- 即使 M 命中 8192 或 16384，N/K 组合也未必匹配（见下）

## 3. gfx950 tuning 的 (N, K) 覆盖

CSV 共 58 个 unique (N, K) 组合，主要服务于：
- glm5（N=2624, 6144, 4096, 2048, 3072 等）
- llama70B/405B（N=8192, 28672, 1024 等）
- DSv3、qwen32B、kimik2、gptoss

**Step-3.5-Flash prefill 实际 GEMM (N, K) 形状（从 h1 验证日志提取）**：

tp=2（h1_tp2_full.log，62 次 miss）：
| (N, K) | 用途 |
|--------|------|
| (4096, 4096) | attn O proj（hidden=4096） |
| (11264, 4096) | qkv proj（合并） |
| (1280, 4096) | head_size 相关 |
| (5120, 4096) | head_size 相关 |
| (7168, 4096) | MLP gate/up 拆 |
| (32, 4096), (48, 4096) | router/小投影 |
| (4096, 5632) | MLP down |
| (4096, 6144) | MLP down |
| (4096, 640) | MoE expert |
| (64448, 4096) | lm_head |

tp=4（h1_tp4_full.log，120 次 miss，更多 N 拆分）：
- 额外 (N,K)：(2560, 4096), (3584, 4096), (16, 4096), (24, 4096), (5632, 4096),
  (640, 4096), (4096, 2048), (4096, 2816), (4096, 3072), (4096, 320), (32224, 4096) 等

**这些 (N, K) 组合在 gfx950 tuning 中是否存在**：
- (4096, 4096) → ✗ 不在 58 个 unique (N,K) 列表中
- (4096, 2048) → ✓ 存在，但只 tuned 到 M ≤ 512（M=10262 不命中）
- (1280, 4096) → ✗ 不存在
- (5120, 4096) → 列表只有 (5120, 1280/25600/2880/3200/5120/640/6400)，无 K=4096
- (7168, 4096) → 列表只有 (7168, 2048/512/8192)，无 K=4096
- (4096, 5632), (4096, 6144), (4096, 640) → ✗
- (11264, 4096), (32, 4096), (48, 4096) → ✗
- 总体：**Step-3.5-Flash 用到的 prefill (N, K) 组合几乎全部不在 gfx950 tuned 集合内**

## 4. 日志实测 miss 计数

`/tmp/aiter_configs/bf16_tuned_gemm.csv` 中 fallback 到 `torch solution:0` 的次数：

| 日志 | 次数 |
|------|------|
| h1_tp2_full.log | 62 |
| h1_tp4_full.log | 120 |

每条都是 `using torch solution:0`，调用 `torch_gemm()` → `F.linear`（即 torch.mm）。
M=10262 的形状基本全部命中此 fallback。

注意 M=16384 的 miss 也大量出现（带 BOS workaround / aiter 内部 padding），同样无 tuning。

## 5. H6 初步结论

**H6 假设成立**：gfx950 的 bf16 tuning 集合对 Step-3.5-Flash 的 prefill 形状覆盖严重不足。

具体证据链：
1. 加载逻辑无误（`/tmp/aiter_configs/bf16_tuned_gemm.csv` 即 runtime 合并副本，路径与日志一致）
2. base `bf16_tuned_gemm.csv` 在 gfx950 上完全为空（0 条），所有 tuning 来自模型专属
   `model_configs/*_bf16_tuned_gemm.csv`，未包含 Step-3.5-Flash 专属的 tuning 文件
3. 现有 gfx950 tuning 的 M 上限 32768 但分布稀疏（…8192, 16384, 32768），M=10262 落入空隙
4. 即使 M 命中 padded 8192/16384，(N, K) 组合也几乎无一匹配 Step-3.5-Flash hidden=4096 的 layout
5. 日志直接证明：tp=2 有 62 次、tp=4 有 120 次 prefill GEMM 落到 `torch solution:0`
   （F.linear / torch.mm fallback 路径，绕过 ASM/flydsl 优化 kernel）

**是否能解释 TTFT ~2× 差距**：高度可能。
- gfx950 上 prefill 主要 GEMM 全部走 torch.mm，未走调优 ASM/flydsl
- 对比 glm5 已 tuned 的形状（如 16384,6144,3072 → asm bf16gemm_bf16_tn_256x256，
  1401 TFLOPS），未 tuned 走 torch.mm 的算力差几个 X 是常见事
- 但严格证明 2× 差距需要直接对比 tuned vs untorched 的两次 prefill latency
  （建议下一步：用 `AITER_TUNE_GEMM=1` dump 缺失形状 → 在 gfx950 上 offline tune 后
  再跑一次 perf_bench.py，看 TTFT 是否回落到 gfx942 水平）

## 6. 关键文件路径

- runtime CSV：`/tmp/aiter_configs/bf16_tuned_gemm.csv`
- 加载代码：`/home/hanchang/junlin12_repos/aiter/aiter/tuned_gemm.py:38-193`（fallback 在
  L188-191）
- 合并代码：`/home/hanchang/junlin12_repos/aiter/aiter/jit/core.py:178-296`（CSV merge to
  /tmp）
- 源 CSV（base 空）：`/home/hanchang/junlin12_repos/aiter/aiter/configs/bf16_tuned_gemm.csv`
- 模型 tuning：`/home/hanchang/junlin12_repos/aiter/aiter/configs/model_configs/*bf16_tuned_gemm*.csv`
- H1 验证日志（含 miss）：`/home/hanchang/project_summary/step35-flash-support/16_perf_gfx950_verified/logs/h1_tp{2,4}_full.log`

## 7. 给 #502 的输入

- H6 已有强证据成立
- RESULTS.md 应记录：覆盖率差距是 prefill 慢的最可能根因
- 推荐后续验证（不在本单据范围）：
  1. `AITER_TUNE_GEMM=1` 跑一次 prefill，把缺失 (M=10262, N, K) dump 出来
  2. offline 用 aiter tune 工具补 gfx950 tuning（专门为 Step-3.5-Flash hidden=4096）
  3. 重跑 perf_bench.py，对比 TTFT 是否回落
