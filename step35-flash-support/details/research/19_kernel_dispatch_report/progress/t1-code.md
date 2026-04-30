# t1-code 进度 — #101 代码追踪

任务范围：Step-3.5-Flash-FP8（FP8 tp=2/tp=4）推理涉及的所有操作，找到最终 kernel
（torch / CK / ASM / triton / gluon）函数名，纯代码阅读，结论附文件:行号。

模型关键参数（来自 `/root/.cache/huggingface/.../Step-3.5-Flash/.../config.json`）：
- hidden_size=4096, head_dim=128, num_attention_heads=64, num_attention_groups=8 (GQA)
- moe_intermediate_size=1280；tp=2 时 inter_dim=640，tp=4 时 inter_dim=320（实际有 padding 到 384，见 § MoE）
- 部分层 `sliding_window=512`（"sliding_attention"），其他层 "full_attention"

---

## 操作对照表（核心输出）

| 操作 | 场景 | Kernel 类型 | 具体 Kernel 名 | 触发条件 | 代码来源 |
|------|------|------------|---------------|---------|---------|
| MoE routed experts | prefill, q_type=per_1x128, token>32 且 inter%256==0 | ASM 1-stage | `aiter.fmoe_fp8_blockscale_g1u1` | gfx950 + per_1x128 + g1u1 | fused_moe.py:590, 881-883, 951-968, 468-474 |
| MoE routed experts | prefill, q_type=per_1x128, token>32 但 inter%256!=0 | CK 2-stage | `aiter.ck_moe_stage1_fwd` + `aiter.ck_moe_stage2_fwd` | run_1stage=False 且 (q_type=per_1x128 且 doweight_stage1) 或 q_dtype_w∈{fp8,…} | fused_moe.py:881-889, 1074-1115, 1737-1786, 1094-1100 |
| MoE routed experts | decode (token=1) | CK 2-stage | `aiter.ck_moe_stage1_fwd` + `aiter.ck_moe_stage2_fwd` | run_1stage=False（token≤32） | 同上 |
| Attention prefill | full + sliding 都走同一函数 | ASM (FA) | `aiter.flash_attn_varlen_func` | ctx.is_prefill | attention_mha.py:497-525, 581-582 |
| Attention decode | sliding_window 层 | Triton (Gluon) | `torch.ops.aiter.pa_decode_gluon` | use_triton_attn=True（sliding_window≠-1 或 head_dim≠128） | attention_mha.py:130, 367-440, 584-585 |
| Attention decode | full_attention 层 (block_size=1024) | ASM | `aiter.pa_persistent_fwd` | block_size==1024 且非 triton 路径 | attention_mha.py:467-495, 587-590 |
| Attention decode | full_attention 层 (block_size≠1024) | ASM | `aiter.pa_fwd_asm` | 非 triton 且 block_size≠1024 | attention_mha.py:444-465, 591 |
| BF16 linear (attn proj / dense MLP / shared expert) | tuned 命中 → asm | ASM | `aiter.gemm_a16w16_asm`（kernelName 由 csv 决定） | tuned_df 命中 libtype="asm" | tuned_gemm.py:300-304, 433-446 |
| BF16 linear | tuned 命中 → hipblaslt | hipBLASLt | `aiter.hipb_mm` | tuned 配置 libtype="hipblaslt" | tuned_gemm.py:373-392 |
| BF16 linear | 默认 skinny shape (M 小、N≤2·cu_num) | ASM/HIP | `aiter.wvSpltK` 或 `aiter.LLMM1` 或 `aiter.wv_splitk_small_fp16_bf16` | 命中 `is_skinny_default_shape` | tuned_gemm.py:76-97, 180-184, 339-370 |
| BF16 linear | 完全没 tuning 且非 skinny | torch | `F.linear` | default config libtype="torch" | tuned_gemm.py:185-187, 395-430 |
| lm_head | all | 复用 BF16 linear pipeline | `tgemm.mm` → `gemm_a16w16` → 上述任一分支 | 见上 | embed_head.py:181, tuned_gemm.py:552-572, 246-336 |

---

## 各操作详细追踪

### 操作 1 — MoE Routed Experts (FP8 blockscale per_1x128)

**入口**：`fused_experts(...)` → `metadata.stage1(...)`（fused_moe.py:312-337）。
metadata 由 `get_2stage_cfgs(...)`（L729-）返回，dispatch 决策位于 L867-1138。

**run_1stage 启发式**（fused_moe.py:867-889）：
```
867:    if cfg is None or int(os.environ.get("AITER_BYPASS_TUNE_CONFIG", "0")):
868:        ksplit = 0
...
880:        ) in fused_moe_1stage_dict[get_gfx()]:
881:            if q_type == QuantType.per_1x128:
882:                # for fp8 blockscale, ck has better performance so disable assembly kernel
883:                run_1stage = token > 32 and (inter_dim % 256 == 0)
```

**`fused_moe_1stage_dict["gfx950"]` 中针对 per_1x128 + bf16 + fp8 + fp8 + g1u1**（fused_moe.py:590）：
```
590:        (ActivationType.Silu,   QuantType.per_1x128,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_fp8_blockscale_g1u1,
```

**1-stage 走向 ASM kernel**（fused_moe.py:468-474, 496-512）：
```
468:        if quant_type == QuantType.per_1x128:
469:            fmoe_func = functools.partial(
470:                aiter.fmoe_fp8_blockscale_g1u1,
471:                fc_scale_blkn=128,
472:                fc_scale_blkk=128,
473:                block_size_M=block_size_M,
474:            )
...
496:        fmoe_func(...)
```

**Step-3.5-Flash 的 inter_dim**：moe_intermediate_size=1280 → 单 expert 行宽。tp=2 时 1280/2=640；tp=4 时 1280/4=320，但代码做 padding（参见 ATOM padding 路径，本任务不展开），运行配置中 inter_dim 实际取 640（tp=2）和 384（tp=4 padding 后）。
- 640 % 256 = 128 ≠ 0 → **run_1stage=False**（即使 prefill）→ 走 2-stage CK
- 384 % 256 = 128 ≠ 0 → **run_1stage=False** → 走 2-stage CK

⇒ 在 Step-3.5-Flash-FP8 上，因 inter_dim 不是 256 的倍数，**MoE 实际不会进入 ASM 1-stage `fmoe_fp8_blockscale_g1u1`**，**全部走 2-stage CK**（无论 prefill 还是 decode）。

**2-stage 分支**（fused_moe.py:1074-1115）：
```
1074:    if (kernelName1 and "ck2stages" in kernelName1) or (
1075:        not kernelName1
1076:        and (
1077:            (q_type == QuantType.per_1x128 and doweight_stage1)
1078:            or q_dtype_w in [...dtypes.fp8...]
...
1094:            stage2_func = functools.partial(
1095:                aiter.ck_moe_stage2_fwd,
1096:                kernelName=kernelName2,
...
1102:            functools.partial(
1103:                ck_moe_stage1,
1104:                kernelName=kernelName1,
```

`ck_moe_stage1` 内部调 `aiter.ck_moe_stage1_fwd`（fused_moe.py:1767）。
`stage2_func` 直接是 `aiter.ck_moe_stage2_fwd`。

---

### 操作 2 — Attention Prefill

**dispatch**（attention_mha.py:577-591）：
```
577:    def dispatch_backend(self, fwd_ctx: ForwardContext):
578:
579:        ctx = fwd_ctx.context
580:
581:        if ctx.is_prefill:
582:            return self.prefill_attention
```

**prefill_attention**（L497-525）：
```
497:    @mark_trace(prefix="prefill_attention", torch_compile=False)
498:    def prefill_attention(
...
504:        sliding_window = (
505:            (self.sliding_window, 0, 0)
506:            if self.sliding_window is not None
507:            else (-1, -1, 0)
508:        )
509:        o = aiter.flash_attn_varlen_func(
510:            q, k, v,
...
521:            window_size=sliding_window,
522:            sink_ptr=self.sinks,
523:        )
```

prefill 分支不区分 sliding/full layer，统一走 ASM Flash Attention varlen，sliding 通过 window_size 参数控制。

---

### 操作 3 — Attention Decode

**use_triton_attn 判定**（attention_mha.py:130-131）：
```
130:        use_triton_attn = self.sliding_window != -1 or self.head_dim != 128
131:        self.use_triton_attn = use_triton_attn
```

Step-3.5-Flash head_dim=128。
- sliding_attention 层：sliding_window=512 ≠ -1 → use_triton_attn=True
- full_attention 层：sliding_window=-1 → use_triton_attn=False

**dispatch decode**（attention_mha.py:583-591）：
```
583:        else:
584:            if self.use_triton_attn:
585:                return self.paged_attention_triton
586:            else:
587:                # Only use pa persistent when block_size == 1024
588:                atom_config = get_current_atom_config()
589:                if atom_config.kv_cache_block_size == 1024:
590:                    return self.paged_attention_persistent_asm
591:                return self.paged_attention_asm
```

**paged_attention_triton** → `torch.ops.aiter.pa_decode_gluon`（attention_mha.py:418-440）。
内部使用 Triton Gluon kernel，对 sliding window 做特殊分区（L387-389：`max_context_partition_num=1, context_partition_size=128`）。

**paged_attention_asm** → `aiter.pa_fwd_asm`（attention_mha.py:450-463）。
**paged_attention_persistent_asm** → `aiter.pa_persistent_fwd`（attention_mha.py:474-493）。

---

### 操作 4 — BF16 Linear（attn qkv/o proj、dense MLP、shared expert）

**入口**：`tgemm.mm(x, w, bias)`（tuned_gemm.py:552-572）→ `gemm_a16w16(...)`（L246-336）。

**dispatch 优先级**（L272-321）：
1. 查 tuned config（cu_num, padded_M, N, K, dtype, bpreshuffle 命中 csv）
2. tuned 命中 `libtype="flydsl"` → `flydsl_gemm`（gfx942 路径，gfx950 一般不走）
3. tuned 命中 `libtype="asm"`（且非 gfx12）→ `asm_gemm` → `aiter.gemm_a16w16_asm`（L433-446）
4. 否则 `solMap[libtype]` 调度（L519-525）：
   ```
   519:    solMap = {
   520:        "torch": torch_gemm,
   521:        "hipblaslt": hipb_gemm,
   522:        "skinny": skinny_gemm,
   523:        "asm": asm_gemm,
   524:        "triton": triton_gemm,
   525:    }
   ```

**默认 fallback**（无 tuned config）：
- bpreshuffle=True 且 gfx950 + bf16 + N%64==0 + K%64==0 → `libtype="asm"`，kernelName=None（让 ASM 走默认）（L160-176）
- skinny shape → `skinny_gemm`（L180-184）→ 调用 `wvSpltK / LLMM1 / wv_splitk_small_fp16_bf16`（L353-367）
- 都不满足 → `libtype="torch"` → `F.linear`（L186-187, L429）

**bpreshuffle 来源**（L256-258）：
```
256:    bpreshuffle = False
257:    if hasattr(B, "is_shuffled") and B.is_shuffled is True:
258:        bpreshuffle = True
```
ATOM 在 gfx950 下对所有 BF16 weight 调用 `shuffle_weights()`（CLAUDE.md/MEMORY 规则确认），所以 BF16 linear 主路径是 `bpreshuffle=True`，命中 csv 时多为 ASM/hipblaslt，未命中时进入 ASM 默认 fallback（kernelName=None）。

---

### 操作 5 — lm_head (vocab projection)

**`ParallelLMHead.forward`**（embed_head.py:172-192）：
```
172:    def forward(self, x: torch.Tensor):
...
178:            if context.is_prefill and not context.is_draft:
179:                last_indices = attn_metadata.cu_seqlens_q[1:] - 1
180:                x = x[last_indices].contiguous()
181:        logits = tgemm.mm(x, self.weight, self.bias)
182:        if self.tp_size > 1:
...
184:            logits = tensor_model_parallel_all_gather(logits, use_custom=use_custom)
```

走与 BF16 Linear 完全相同的 dispatch（`tgemm.mm` → `gemm_a16w16`），所以最终 kernel 由 csv tuning 决定：典型 prefill last-token (M 极小) + N=vocab/tp 形状 → 命中 skinny 或 hipblaslt；decode 同理。lm_head weight 默认未 shuffled（embedding 路径不走 shuffle_weights），所以 bpreshuffle=False。

---

### 操作 6 — MoE routed experts 在 decode (token=1) 的路径

token=1 时进入 `get_2stage_cfgs`，run_1stage=False（fused_moe.py:881-889 中 per_1x128 要求 token>32）。流程进入 fused_moe.py:339-366 → `fused_moe_2stages` → `metadata.stage1(...)`（L1281）+ `metadata.stage2(...)`。

stage1 在 per_1x128 + fp8 + g1u1 + 命中 L1074-1086 条件 → `ck_moe_stage1` → `aiter.ck_moe_stage1_fwd`。
stage2 → `aiter.ck_moe_stage2_fwd`。

注意：metadata.stage1 不是 `asm_stage1`（asm_stage1 仅在 L1123-1135 的 fallback 分支返回，但本配置先命中 L1074 分支）。验证：fused_moe.py:1077 `(q_type == QuantType.per_1x128 and doweight_stage1)` — Step-3.5-Flash 的 sigmoid router + norm_expert_weight=True 路径下 doweight_stage1 视实际配置决定。若 doweight_stage1=False，分支条件由 `q_dtype_w in [..., dtypes.fp8]` 满足（L1078-1085），仍命中 CK 2-stage 路径。

---

## 备注 / 不确定项

- ATOM 实际的 `doweight_stage1` 取值（影响 stage1 是 CK 还是 ASM）需在 t2 实测日志中确认 `[fused_moe] using 2stage` 日志行的 kernelName1 是否包含 `ck2stages` 或 `moe_stage1_g1u1`（asm 路径）。
- BF16 linear 是否实际命中 csv tuned config，需 t2 看 `aiter.tuned_gemm` 的 `logger.info("...is tuned on cu_num...")` 日志。
- shared expert（dense MLP）虽然走 BF16，但实际 weight 是否 shuffled 取决于 ATOM 的 ColumnParallelLinear/RowParallelLinear 实现路径，本次未追到端点。
