# teammate-8 / #601 — FP8 GEMM dispatch & gfx942/gfx950 FP8 tuning 覆盖

## 1. Step-3.5-Flash-FP8 量化范围（来自 HF config）

**配置文件**：`/root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/6eebda59dd87ca5729648ec7cfed0becfceb273e/config.json`

`quantization_config` (line 313-608)：
- `quant_method = "fp8"`，`activation_scheme = "dynamic"`，`fmt = "e4m3"`
- `weight_block_size = [128, 128]` → blockscale per_1x128
- `modules_to_not_convert` 共 286 项

### 走 BF16 GEMM（保留 bf16，不量化）的层
| 类别 | 路径 | 数量 |
|------|------|------|
| Embedding / norm / lm_head | `lm_head`, `model.embed_tokens`, `model.norm` | 3 |
| Layer 0/1/2 全部 attn + dense MLP | `self_attn.{g,qkv,o}_proj`, `mlp.{gate_up,down}_proj` | 5 × 3 = 15 |
| Layer 3-44 attn projections | `self_attn.{g,qkv,o}_proj` | 3 × 42 = 126 |
| Layer 3-44 router (`moe.gate`) | `moe.gate` | 42 |
| Layer 3-44 shared expert | `share_expert.{gate_up,down}_proj` | 2 × 42 = 84 |
| MTP layers 45/46/47 | attn + mlp | 5 × 3 = 15 |

### 走 FP8 blockscale（per_1x128 e4m3fn）的层
- **唯一 FP8 量化**：layer 3-44 的 `moe.experts`（routed experts，FusedMoE kernel）
- 即：**288 个 routed experts × 42 层**（含 layer 43/44 SwigluStep）

### ATOM 端代码确认
- `step3p5.py:230` `Step3p5MoE.experts = FusedMoE(..., quant_config=quant_config, ...)` → 走 fmoe FP8 dispatch
- `step3p5.py:200` router gate `quant_config=None` → BF16
- `step3p5.py:563` shared expert `Step3p5MLP(quant_config=quant_config)` 但 modules_to_not_convert 把 `share_expert.*` 全部排除 → BF16
- `step3p5.py:384/391` qkv_proj/o_proj 同样被 modules_to_not_convert 排除 → BF16

## 2. FP8 GEMM tuning 文件

### aiter `configs/` 下 FP8 相关 CSV
| 文件 | 行数 | 用途 |
|------|------|------|
| `tuned_fmoe.csv` | 1700 | **routed-expert FusedMoE 主 tuning** |
| `untuned_fmoe.csv` | 13 | fmoe 待 tune 输入 |
| `a8w8_blockscale_tuned_gemm.csv` | 6 | dense FP8 blockscale GEMM（仅 5 条 + header） |
| `a8w8_blockscale_bpreshuffle_tuned_gemm.csv` | 59 | dense FP8 blockscale + preshuffle |
| `a8w8_bpreshuffle_tuned_gemm.csv` | 551 | per-tensor/token FP8 |
| `a8w8_tuned_gemm.csv` | 583 | per-tensor/token FP8 |
| `model_configs/a8w8_blockscale_tuned_fmoe_*.csv` | 17~33 each | 模型特定 fmoe（dsv3/glm5/qwen3） |
| `model_configs/glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv` | 81 | glm5 dense FP8 |

**Step-3.5-Flash 没有任何专属 FP8 tuning CSV**（只有 dsv3/glm5/qwen3.5_397b/minimax-m2_5/qwen3_235b/gptoss/kimik2 有 model-specific 文件）。

### Runtime merged 文件
- `/tmp/aiter_configs/tuned_fmoe.csv` 共 2162 行（含 model_configs 合并）
  - **cu=256（gfx950，CDNA4）**：1214 行
  - **cu=80（gfx942/MI300X）**：947 行

### dispatch 加载路径（aiter/jit/core.py:80-98）
```
AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE → a8w8_bpreshuffle_tuned_gemm.csv
AITER_CONFIG_GEMM_A8W8_BLOCKSCALE  → a8w8_blockscale_tuned_gemm.csv
AITER_CONFIG_FMOE                  → tuned_fmoe.csv
AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE → a8w8_blockscale_bpreshuffle_tuned_gemm.csv
AITER_CONFIG_GEMM_BF16             → bf16_tuned_gemm.csv  ← H6 已分析
```

## 3. FP8 routed-expert dispatch 逻辑（aiter/fused_moe.py）

`fused_moe.py:802-867` `cu_num = get_cu_num()` → key 包含 `(cu_num, token, model_dim, inter_dim, expert, topk, activation, dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1, doweight_stage1)`，按此 13 元 tuple 在 `cfg_2stages`(=tuned_fmoe.csv) 查找。

未找到（cfg is None）→ **使用 default heuristics**（line 867-：`block_m`/`run_1stage`/`kernelName1=""`/`kernelName2=""` 全靠启发式选择）。

## 4. H1 日志中的 FP8 fmoe miss 实证

来源：`logs/h1_tp2_full.log` 与 `h1_tp4_full.log`，过滤 `using 2stage default`。

### tp=2（cu=256, inter_dim=640 = moe_intermediate_size 1280 / tp=2）
8 处 miss，4 个 unique key（token=1 decode + token=16384 prefill warmup × Silu(routed) / SwigluStep(layer 43-44)）：
```
(256, 16384, 4096, 640, 289, 9, ActivationType.Silu,        bf16, e4m3fn, e4m3fn, per_1x128, True, False)
(256, 16384, 4096, 640, 288, 8, ActivationType.SwigluStep,  bf16, e4m3fn, e4m3fn, per_1x128, True, False)
(256, 1,     4096, 640, 289, 9, ActivationType.Silu,        bf16, e4m3fn, e4m3fn, per_1x128, True, False)
(256, 1,     4096, 640, 288, 8, ActivationType.SwigluStep,  bf16, e4m3fn, e4m3fn, per_1x128, True, False)
```
（expert=289 = 288 routed + 1 fused shared at non-SwigluStep layers；expert=288 + topk=8 = SwigluStep layers 43-44，shared 拆出去；这与 step3p5.py:220-225 `_fuse_shared_at_layer` 完全一致）

### tp=4（cu=256, inter_dim=384 = padded from 320）
16 处 miss，4 个 unique key（同结构，inter_dim=384 instead of 640）。

### tuned_fmoe.csv (cu=256, per_1x128) 实际覆盖
- `inter_dim ∈ {256, 384, 512, 768, 1536}`
- `expert ∈ {16, 128, 256, 257, 513}`
- 总共 460 条 per_1x128 行（含 cu=80 的 gfx942 条目）

### gap 分析（每条都缺）
| 维度 | Step-3.5-Flash 需要 | tuned_fmoe.csv 覆盖 | 状态 |
|------|---------------------|----------------------|------|
| `cu_num=256, inter_dim=640` (tp=2) | 必须 | 无 | **MISS** |
| `cu_num=256, inter_dim=384` (tp=4 padded) | 必须 | 有 inter_dim=384 行 | inter 维匹配，但下面字段全错 |
| `cu_num=256, expert=288 / 289` | 必须 | 无（最接近 257/513） | **MISS** |
| `cu_num=256, ActivationType.SwigluStep` | layer 43-44 必须 | 无（只有 Silu/Gelu） | **MISS** |
| `cu_num=256, q_dtype=float8_e4m3fn + per_1x128 + topk=8/9` | 必须 | 无 | **MISS** |

→ FP8 routed-expert **0 命中 tuning**，全部 fallback 到 2stage default heuristics。

## 5. gfx942 vs gfx950 FP8 tuning 覆盖对比

| CSV (per_1x128 only) | gfx942 (cu=80) | gfx950 (cu=256) |
|----------------------|----------------|------------------|
| tuned_fmoe.csv 总行 | 947 | 1214（runtime 合并后） |
| tuned_fmoe per_1x128 行 | 部分（fnuz format） | 部分（fn format）|
| a8w8_bpreshuffle_tuned_gemm.csv | 69 | 481 |
| a8w8_tuned_gemm.csv | 26 | 556 |
| a8w8_blockscale_bpreshuffle_tuned_gemm.csv | 0（仅 cu=256 entries） | 58 |

但**没有任何条目同时满足 (cu, inter_dim=640|320, expert=288|289, SwigluStep, e4m3fn, per_1x128)**，无论 gfx942 还是 gfx950。
即使切到 gfx942 (cu=80)，FP8 fmoe 仍然 miss 全部 4 个 Step-3.5-Flash 特征 tuple。

## 6. BF16 GEMM 范围（H6 miss 来源）

H1 日志 `not found tuned config in /tmp/aiter_configs/bf16_tuned_gemm.csv` 的 N 维：
- N=11264, 4096, 5120 → dense MLP gate_up_proj / down_proj（layer 0-2）
  - gate_up [N=2*intermediate=22528 → ColumnParallel/tp=2 → 11264] ✓
  - down [N=hidden=4096] ✓
- N=7168, 4096, 5120 → attn qkv_proj (Q+K+V merged) / o_proj (各层 attn)
- N=1280, 640 → share_expert gate_up / down (`share_expert_dim=1280`)
- N=32, 48 → router 相关 (g_proj head-wise gate, num_heads=32 or 48)
- N=64448 → lm_head (vocab=128896 / tp=2)
- M=10262 → prefill seq_len 10000+ tokens

这些**全部是 BF16 modules_to_not_convert 中的层**，与 H6 一致。

## 7. 结论

### 7.1 H6（bf16 tuning 缺失）能解释多少 TTFT gap？
H6 已确认：bf16_tuned_gemm.csv runtime 仅含 glm5 的 779 条 gfx950 entries，**全部 5 类 BF16 GEMM 形状（attn projections + dense MLP + shared expert + router gate + lm_head）+ M=10262 prefill 形状均 fallback 到 torch solution:0**。
prefill TTFT 的主要时间花在：
- **第一阶段**：7 个 attn projection GEMM × 45 layers + 2 个 dense MLP × 3 layers + 2 个 shared expert × 42 layers + 2 个 lm_head GEMM
- 全 BF16 fallback；H6 是 prefill TTFT 主要 gap 来源。

### 7.2 FP8 routed-expert 还有额外 tuning gap（本次新发现）
**FP8 fmoe 0 命中 tuning**，全部 4 类 key tuple miss：
- 每层 routed-expert prefill (token≈10262) 用 default heuristics 选 kernel
- `block_m=64`、`run_1stage=False`、`ksplit=0`、`use_nt=False`，`kernelName1/kernelName2=""`（线索：未指定具体 ck kernel name → 走 dispatcher 默认路径）
- 这是 **prefill TTFT 的第二个 gap**：42 个 MoE 层 × token=10262 的 fmoe 都没 tuning

### 7.3 修复建议
1. 用 `gemm_moe_tune.py` (`AITER_CSRC_DIR/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`) 为 Step-3.5-Flash 4 个 unique fmoe key 做 tuning，写入 `aiter/configs/model_configs/step35flash_a8w8_blockscale_tuned_fmoe.csv`
2. 同时补 `step35flash_bf16_tuned_gemm.csv`（参考 H6 #501 的 N/K 列表 + M={1,128,256,512,1024,2048,4096,8192,10262,16384}）
3. tp=2 优先（inter_dim=640 + expert=289 + SwigluStep），tp=4 次之

### 7.4 H6 + #601 综合
| Gap | 影响阶段 | 修复来源 |
|-----|----------|----------|
| H6: BF16 GEMM 全 miss → torch fallback | prefill TTFT 主要 | bf16 tuning |
| #601: FP8 fmoe 全 miss → default heuristics | prefill TTFT 次要 + decode TPOT 部分 | fmoe tuning |
| Decode TPOT 已经 12.3-12.5 ms 接近合理（routed expert token=1 也 miss tuning，但 default heuristics 在 token=1 表现尚可） | — | — |

## 8. 文件路径速查
- HF config: `/root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/6eebda59dd87ca5729648ec7cfed0becfceb273e/config.json:313-608`
- ATOM Step3p5: `/home/hanchang/ATOM/atom/models/step3p5.py:200,220-235,384-411,560-565`
- ATOM tp4 padding: `/home/hanchang/ATOM/atom/model_ops/moe.py:1565`
- aiter dispatch: `/home/hanchang/junlin12_repos/aiter/aiter/fused_moe.py:780-867`
- aiter config paths: `/home/hanchang/junlin12_repos/aiter/aiter/jit/core.py:70-160`
- runtime merged tuning: `/tmp/aiter_configs/{tuned_fmoe.csv, bf16_tuned_gemm.csv}`
- log evidence: `/home/hanchang/project_summary/step35-flash-support/16_perf_gfx950_verified/logs/h1_tp{2,4}_full.log` (lines 77-78, 84-85, 98-99, 105-106, 147-148 etc.)
