# DC-T2 progress 时间线挖掘

> 输入：`/home/junlin12/project_fp8_tp4_repro/progress/teammate-{1,2,3,4,5,6,7,9-review,10,11,12,13,14-review,16,17,18,19,20}.md`（18 份，缺 8/15）
> 输出策略：按 RC 组织事件流；只保留最终被采纳的有效路径，dead end 省略

---

## §1 项目阶段总览

| 阶段 | 主线 | 涉及 teammate | 阶段产出 |
|---|---|---|---|
| **Wave 0**（接手 / Phase 0） | lead 完成 ATOM/aiter clone、commit checkout、aiter editable install；HF 下载 ~59% | lead（无 progress） | 三仓 commit 锁定（ATOM `acff926` / aiter `0f8164017` / CK `defd7ad29`） |
| **Wave 1 / Phase 1（静态预查）** | 并行调查 4 类静态风险，证明 ATOM/aiter 侧 fix 已落地，提前识别 **NEW-RC-1**（fp8 fnuz vs ocp） | T-1（#A03/#A04）、T-2（#A01/#A02）、T-3（#A05） | 三大候选 root cause 静态收敛：BOS bug 不会触发、CK SwigluStep 支持 gfx942、fnuz 转换链已闭环 |
| **Wave 2 / Phase 2（NEW-RC-3 patch）** | lead 通过 Explore agent 静态定位 NEW-RC-3（per_1x128 prefill 误走 ASM `fmoe_g1u1`）；T-4 应用 patch | T-4（#C02） | `aiter/fused_moe.py:881-886` 单一 hunk patch（注释 + `run_1stage = False`） |
| **Wave 3 / Phase 3（M1 baseline tp=2）** | T-5 跑 baseline；T-6/T-7 静态预研 NEW-RC-2 方向 + dispatch coverage | T-5（#004）、T-6（NEW-RC-2 静态）、T-7（dispatch coverage） | M1 PASS（4/4 prompt 连贯，0 dispatch miss）；T-7 抛出 M2 stage2 NPerBlock=64 风险 |
| **Wave 3.5（reviewer + closeout）** | T-9 critical review，识别 V2 闭环 gap（F-1 block）+ HF 引文存疑（F-2）+ M2 警告强度（F-3）+ V3 重定义（F-4） | T-9（review） | 9 findings；F-1 block 待补 |
| **Wave 4（F-1 静态补强）** | T-11 用 ATOM `moe.py:2150-2170` if 链 + 三 process 分支无条件 normalize 静态闭合 V2 | T-11 | F-1 反证强度判 strong；F-2 判（b）推断措辞；F-4 给出文字建议 |
| **Wave 5（M2 tp=4）** | T-10 深度验证 T-7 警告 → 发现 ATOM padding 自动规避；T-12 跑 M2 实测 PASS；T-13 静态交叉证据 + supersede T-7 §5.4 | T-10、T-12、T-13 | M2 PASS（V4 `inter_dim=384` 决定性间接证据）；M2 padding 触发强度 strong；KPack 根因 medium-偏-strong |
| **Wave 5.5（reviewer）** | T-14 critical review M2 wave，raise F-1 block（M1↔M2 prompt 输出"小分化"被淡化）+ V4/V5 措辞夸张 | T-14（review） | 7 findings；F-1 待量化补强 |
| **Wave 6（量化补强 + 文档同步）** | T-16 用 max_tokens=512 重跑 M1+M2，量化对照证否"数值漂移变 bug"；T-17 把 T-14 findings 同步到 PROJECT_SUMMARY/FINAL_REPORT/SESSION_HANDOFF | T-16、T-17 | P3 byte-identical 143/143 chars + P2 双方 25 prime 全对，F-1 闭环；wave 5 close + PROJECT CLOSED |
| **Wave 7（doc-tidy）** | T-18 文档完整性交叉审查；T-19 新会话可读性梳理；T-20 follow-up readiness 一次性整理 | T-18、T-19、T-20 | 项目正式 close；T-20 产出新会话 quick-start 包 |

---

## §2 NEW-RC-1（FP8 dtype：gfx942 必须用 e4m3fnuz）事件流

### 发现
- **谁**：T-2 在调查 #A02 CK SwigluStep arch 支持时主动识别（Wave 1）
- **场景**：审 `csrc/include/opus/opus.hpp` L932-958 时发现 fp8 numeric_limits 用 `#if defined(__gfx942__)` 给出不同 `bin_max/bin_lowest/bin_qnan`
- **症状**：注释明示"fp8 E4M3: gfx950=OCP(ieee-like, NaN=0x7F), gfx942=fnuz(NaN=0x80)"
- 来源：`progress/teammate-2.md:88-89` —— "csrc/include/opus/opus.hpp L932 标明 gfx942 用 fp8 fnuz 而 gfx950 用 ocp e4m3fn — bias 不同 → 这是 fp8 路径的高风险点"

### 调查（仅最终采纳路径）
- T-3 接手 #A05，从「模型权重格式 / aiter 类型映射 / ATOM 转换路径 / 转换路径触发完整性」四方面挖
- 关键证据 1（模型侧）：`safetensors.safe_open` 实测 `model.layers.3.moe.{down,gate,up}_proj.weight` dtype 全为 `torch.float8_e4m3fn`（OCP，非 fnuz）→ 来源 `progress/teammate-3.md:28-37`
- 关键证据 2（aiter 侧）：`/workspace/aiter/aiter/utility/dtypes.py` L10-25 在 import 时按 GPU arch 静态绑定 `defaultDtypes = {"gfx942": {"fp8": torch.float8_e4m3fnuz}, "gfx950": {"fp8": torch.float8_e4m3fn}}` → 来源 `progress/teammate-3.md:47-58`
- 关键证据 3（ATOM 侧）：`atom/quant_spec.py` L211-243 `_infer_dtype` regex 回退路径 `_DTYPE_PATTERNS = {r"fp8|float8": "fp8"}` → `d_dtypes.get("fp8")` → gfx942 上为 `torch.float8_e4m3fnuz` → 来源 `progress/teammate-3.md:71-79`

### 定位（root cause）
- ATOM `Fp8MoEMethod.__init__`（`atom/model_ops/moe.py:1527-1539`）置位 `need_normalize_e4m3fn_to_e4m3fnuz = (self.quant_dtype == torch.float8_e4m3fnuz)`
- 触发条件：模型 ckpt 是 e4m3fn，但 gfx942 上 ATOM 的 `quant_dtype` 已被 aiter 静态映射成 `e4m3fnuz`，需要在 weight 加载阶段做 bit-pattern 重写 + scale 翻倍补偿
- 来源：`progress/teammate-3.md:81-89`

### 修复（无须新增改动 —— 链路已存在）
- `aiter/utility/dtypes.py:10-25`：gfx942 → e4m3fnuz 静态映射（aiter 既有）
- `atom/model_ops/utils.py:61-82` `normalize_e4m3fn_to_e4m3fnuz`：bit-pattern `-128` → 0 重写 + scale `* 2.0`（ATOM 既有）
- 触发链：`Fp8MoEMethod.process_weights_after_loading` → `_process_block_quant`（L1701→L1709）→ `_normalize_weights_and_scales`（L1683-1699）→ utils.py:79
- 来源：`progress/teammate-3.md:91-117,141-150`

### 验证
- 静态：T-11 用 `atom/model_ops/moe.py:2150-2170` if 链证明 fp8 path **必且仅**走 `Fp8MoEMethod`（`compressed-tensors` 分支被本模型 config `quant_method: "fp8"` 排除），三个 `_process_*` 分支无条件调 `_normalize_weights_and_scales`，反证强度 **strong** → 来源 `progress/teammate-11.md:14-101`
- 动态间接证据（M1）：log L92 fused_moe 调用签名 `q_dtype_a / q_dtype_w == 'torch.float8_e4m3fnuz'`，多处命中 → 来源 `progress/teammate-5.md:90-95`
- 动态间接证据（M2）：grep `fnuz` → **40 matches**；fused_moe 签名同上 → 来源 `progress/teammate-12.md:84-92`
- 反证：若 normalize 未发生，e4m3fn bits 被当 fnuz 解释会差 2× / 7 数量级（对应 NEW-RC-2 inverse 解释爆炸），M1+M2 PASS 输出连贯互斥这种失败模式

---

## §3 NEW-RC-2（weight_scale * 2.0 方向）事件流

### 发现
- **谁**：T-3 在 #A05 §4「转换路径触发完整性」收尾时识别（Wave 1）
- **场景**：T-3 看到 `utils.py:79` `weight_scale = weight_scale * 2.0` 假设 `weight_scale` 是 forward scale，但 stepfun 权重命名是 `weight_scale_inv`（inverse 暗示）
- **原始问题表述**：T-3 §4 第 4 项【未验证假设】明确提示 — 若 inverse 解释 dequant 用 `w / scale`，那 `* 2.0` 应改 `/ 2.0` → 来源 `progress/teammate-3.md:156`

### 调查（仅最终采纳路径）
- T-6 在 baseline 跑前从权重数值 + 上下游代码静态推断方向（Wave 3 并行）
- 四条独立证据链（`progress/teammate-6.md:20-26`）：
  1. **代码上下文**：读 `ATOM/atom/model_ops/utils.py:61-82` 公式
  2. **loader rename 链**：`ATOM/atom/model_loader/loader.py:320-321` 直接把 `weight_scale_inv` 重命名为 `weight_scale`，**不做 1/x** → ATOM 全程不取 scale 倒数
  3. **下游 dequant 公式**：`/workspace/aiter/aiter/ops/triton/moe/quant_moe.py:220-240` `dequant_w_blockscale` 明确 `w = w * scales`（forward 语义）
  4. **safetensors 数值实测 + 本地 fp32 dequant**

### 定位（root cause —— 不是 root cause，而是反证 utils.py:79 方向**正确**）
- 实测层 3 expert 0 第一个 (128,128) block，按两种语义算 fp32 dequant：
  - **Forward (w * scale)**：block absmax = 0.0933，mean|.| = 0.0181（符合 bf16 LLM absmax 0.05–0.5 区间）
  - **Inverse (w / scale)**：block absmax = 2.15e+06，mean|.| = 4.17e+05（爆炸 7 个数量级）
- 来源：`progress/teammate-6.md:51-58`
- 结论：stepfun 的 `weight_scale_inv` **数值上是 forward scale**（命名沿袭 DeepSeek-V3 习惯，"inv" 指 1/amax 缩放因子，不是 dequant 时再做倒数）

### 修复
- **无须修复**：`utils.py:79` 现状 `weight_scale * 2.0` 方向正确
- 来源：`progress/teammate-6.md:140` —— "结论：utils.py:79 的 `weight_scale = weight_scale * 2.0` 方向 ✅ 正确，ratio 应是 ×2（保持现状，无须改 /2）"

### 验证
- 静态：T-6 §3.2 fp32 dequant 实测（最强证据，7 数量级差距不可能误判）
- 动态：M1 PASS + M2 PASS（若方向反，4 个 prompt 必然乱码）
- 来源：`progress/teammate-5.md:189-191` + `progress/teammate-12.md:217-219`

---

## §4 NEW-RC-3（per_1x128 prefill ASM bypass）事件流

### 发现
- **谁**：lead 通过 Explore agent 静态追踪 ATOM→aiter→CK 调用链定位（Wave 2）
- **场景**：审 `aiter/fused_moe.py:881-883` 门控逻辑：
  ```python
  if q_type == QuantType.per_1x128:
      # for fp8 blockscale, ck has better performance so disable assembly kernel
      run_1stage = token > 32 and (inter_dim % 256 == 0)
  ```
- **症状**：M1 (tp=2) inter_dim=1280，prefill (token>32) 满足 `run_1stage=True` → 走 ASM `fmoe_g1u1`（gfx942 dispatch 表 row 582 把 `(Silu, per_1x128, bf16, fp8, fp8, isG1U1=True, doweight_stage1=False)` 映射到此），但该 kernel 签名 `(fc1_scale, fc2_scale)` 不带 block shape 参数，**注释明说 "ck has better performance so disable assembly kernel" 但代码并未真禁用**
- 来源：`progress/teammate-4.md:5-18`

### 调查（仅最终采纳路径）
- T-4 验证 patch 是否会被旁路：grep `run_1stage` 全部赋值/读位置（fused_moe.py 行 312/624/871/883/885/887/889/906/913/922/932/...）
- 关键风险点 L932：`run_1stage = cfg.get("run_1stage", False)`（tuned-config 路径会绕过启发式）
- 检查 `aiter/configs/tuned_fmoe.csv`：`awk '$11 == "QuantType.per_1x128" && $22 == "1"'` → **0 行**（348 条 per_1x128 行 `run_1stage` 列全为 0）
- 来源：`progress/teammate-4.md:68-101`

### 定位（root cause）
- `aiter/fused_moe.py:881-883` 的 `run_1stage = token > 32 and (inter_dim % 256 == 0)` 是 root cause —— per_1x128 prefill 在 inter_dim%256==0 时被路由到 ASM `fmoe_g1u1`，该 kernel 不是 blockscale 实现（无 block shape 参数）

### 修复
- T-4 应用 single hunk patch（`progress/teammate-4.md:42-59`）：
  ```diff
  --- a/aiter/fused_moe.py
  +++ b/aiter/fused_moe.py
  @@ -880,7 +880,10 @@
              if q_type == QuantType.per_1x128:
                  # for fp8 blockscale, ck has better performance so disable assembly kernel
  -                run_1stage = token > 32 and (inter_dim % 256 == 0)
  +                # NEW-RC-3 patch (2026-04-28): force CK blockscale path on gfx942 to avoid
  +                # routing per_1x128 prefill to ASM fmoe_g1u1 which lacks block shape param
  +                # original: run_1stage = token > 32 and (inter_dim % 256 == 0)
  +                run_1stage = False
  ```
- grep 验证：`grep -n "NEW-RC-3" /workspace/aiter/aiter/fused_moe.py` → 1 行（L883）
- 来源：`progress/teammate-4.md:62-65`

### 验证
- M1 实测（T-5）：grep `aiter\.fmoe_g1u1` → **0 matches**；grep `module_moe_ck2stages_f8.*per_1x128` → 多次命中；log L80 直接命中 `[aiter] run_1stage = False, ksplit = 0 q_type = QuantType.per_1x128 block_m = 64` → 来源 `progress/teammate-5.md:67-84`
- M2 实测（T-12）：grep `module_moe_ck2stages_f8.*per_1x128` → 8 行命中；grep `fmoe_g1u1` → **0 matches**；多处 `run_1stage = False` → 来源 `progress/teammate-12.md:69-80`
- 静态覆盖（T-7）：M1 inter=640 + per_1x128 + 强制 2-stage CK 后，prefill (block_m=64) / decode (block_m=16) 共 4 类 (gemm, case) 调用全部命中 `a8w8_gemm{1,2}_blockscale_kernels_list` → 差集 = ∅ → 来源 `progress/teammate-7.md:200-211`

---

## §5 M2 tp=4 padding（inter_dim 320→384）事件流

### 发现
- **谁**：T-7 在 dispatch coverage 静态预查的 §5.4「M2 (tp=4) 前瞻」中作为副产物预警（Wave 3）
- **场景**：T-7 主任务是 M1 dispatch 覆盖；末尾顺手算了 M2 风险
- **预警内容**：`inter_dim_tp4 = 1280/4 = 320`，320 % 128 = 64 ≠ 0，但 320 % 64 == 0 → 走 NPerBlock=64 子路径；**stage2 NPerBlock=64 路径在 `a8w8_gemm2_blockscale_kernels_list` 里没有**（注释 L326-328 说 KPerBlock=64 在 gfx950 不支持，KPack=32 限制）→ M2 stage2 可能 miss
- 来源：`progress/teammate-7.md:233-239`

### 调查（仅最终采纳路径）
- T-10 接 Phase 4 风险深度验证（Wave 5），**深入挖到 ATOM padding 层**（T-7 漏看的部分）
- 调用链证据（`progress/teammate-10.md:22-31`）：
  | 层 | 文件 / 函数 | 计算 | tp=4 实际值 |
  |----|------------|------|------------|
  | 模型原始 | config.json `moe_intermediate_size` | — | 1280 |
  | TP 切分 | `intermediate_size_per_partition = 1280 / tp_size` | tp=4 | **320** |
  | Fp8MoEMethod.create_weights | `moe.py:1541-1614` | unpadded buffer = 320 |
  | **process_weights_after_loading → _process_block_quant** | **moe.py:1709-1746** | `align = block_n = 128`；`inter_pad = ceil(320/128)*128 = 384`；zero-pad w13/w2 | **w2.shape[-1] = 384** |
  | fused_moe heuristic | `inter_dim = w2.shape[-1]` | dispatch 看到 384 | **384** |

### 定位（root cause）
- T-7 警告的 NPerBlock=64 stage2 缺失**确实是事实**（且根因 = FP8 mfma KPack=32 强制 KPerThread%32==0），但该路径在 M2 实测中**根本不被触发**
- ATOM `_process_block_quant`（`atom/model_ops/moe.py:1709-1746`）的 zero-pad 把 inter_dim 320→384，让 fused_moe 永远走 NPerBlock=128 主路径
- ATOM 注释（L1715-1727）显式说明此为 stage2 KPerBlock=128 约束的 padding 解决方案，曾踩过 `align=64` 旧 bug 已改为 `align=block_n` → 来源 `progress/teammate-10.md:34-50`
- T-13 §1.3 强调 **TP 切分发生在 padding 之前**（buffer 用 320 分配，padding 在加载完才执行），这是 T-7 漏看的关键 → 来源 `progress/teammate-13.md:42-52`

### 修复
- **无须修复**：现有 ATOM padding 机制已自动规避
- T-10 推荐 **Option 0（不改任何代码）**：`progress/teammate-10.md:238-244`

### 验证
- 静态（T-13）：唯一入口论证强度 strong（`progress/teammate-13.md:62-71`）
- 动态（T-12）M2 实测**决定性间接证据**：fused_moe 调用签名第 4 位参数（inter_dim）= **384**（不是 320），多行同形式：
  ```
  [aiter] [fused_moe] using 2stage default for (80, 4096, 4096, 384, 289, 9, 'ActivationType.Silu', ..., 'QuantType.per_1x128', True, False)
  ```
  来源：`progress/teammate-12.md:108-122`
- 反证：若 padding 未触发，K=320%128=64≠0 应触发 CK `IsSupportedArgument false` → 抛 `"wrong! device_gemm with the specified compilation parameters does not support this GEMM problem"`，实测 0 matches → 来源 `progress/teammate-12.md:124-135`

---

## §6 其他次要问题

### 6.1 BOS bug（gfx942 上不会触发）
- **风险描述**：旧 BOS workaround 删除 `bf16gemm_bf16_tn_256x256` CSV entry
- **T-2 验证**：CSV grep 全部 `gfx950,256,...` 开头（cu_num=256）；gfx942 dispatcher `cu_num=get_cu_num()` 不匹配 → fallback 到 `default_config["libtype"]="hipblaslt"`（`tuned_gemm.py:162`）→ **gfx942 上永远不会触发该 ASM kernel**
- 来源：`progress/teammate-2.md:11-30`

### 6.2 CK SwigluStep 支持 gfx942
- **风险描述**：CK commit defd7ad29 加入 swiglustep_and_mul 4 段 branch，是否限定 gfx950
- **T-2 验证**：4 段全部位于 `if constexpr(ActivationOperation == Activation::swiglustep_and_mul)` 内，**无任何 arch guard**，与同文件 silu_and_mul/gelu_and_mul 平级；父模板 `gridwise_moe_gemm.hpp` 整体在 `__gfx9__` 守卫下编译，gfx942 满足
- 来源：`progress/teammate-2.md:46-67`

### 6.3 ATOM `align = block_n` fix 已落地（FP8 blockscale）
- **T-1 验证**：`atom/model_ops/moe.py:1726` `align = block_n`，含 inline 注释解释 "Bug fix: previously used align=64 for inter<=192 (copied from BF16 path), but 192%128=64!=0 → stage2 kernel dispatch fails"
- 来源：`progress/teammate-1.md:19-23`

### 6.4 ATOM `_load_w13/_load_w2` ceil 整除 fix 已落地
- **T-1 验证**：`atom/model_ops/moe.py:2310-2312, 2352-2354` ceil pattern `+ self.tp_size - 1`，inline 注释说明若用 floor 整除 per_1x128 inter=1280 tp=4 时第 3 个 scale block 不被复制，导致 ~5000× dequant error
- 来源：`progress/teammate-1.md:36-44`

### 6.5 block_m heuristic 覆盖（M1 inter=640 全部命中）
- **T-7 静态枚举**：NEW-RC-3 patch 后 per_1x128 路径下 block_m 唯一可能值 {64, 16}；M1 4 类 (gemm, case) 全命中 codegen heuristic_dispatch 模板
- 来源：`progress/teammate-7.md:200-211`

### 6.6 KPack=32 跨架构推论强度（gfx950 → gfx942）
- **T-13 §3 判定**：medium-偏-strong
  - strong 一面：FP8 mfma 指令族 KPack=32 是 CDNA3+ 硬件常量；blockscale 模板 `static_assert(KPerThread % KPack == 0)` 在 KPerBlock=64 + 标准 wave 配置下数学上不可能满足；非 blockscale 路径横向对比（BF16/int8 有 KPerBlock=64 / fp8 blockscale 独缺）支持硬件约束假说
  - medium 削弱：注释字面只提 gfx950；非 blockscale fp8 stage2 KPerBlock=64 在 gfx942 codebase 还活着（反例）
- 来源：`progress/teammate-13.md:96-157`

### 6.7 F-1 量化补强（T-16 证否「TP 数值漂移变 bug」假设）
- **场景**：T-14 F-1 raise M1↔M2 prompt 输出"小分化"被淡化，担心数值漂移变 bug
- **T-16 量化对照**（`max_tokens=512` 重跑）：
  - **P3（1+2+3=?）M1↔M2 byte-for-byte 完全一致 143/143 chars**（最强单一证据）
  - P2（list primes）双方均给出正确的 25 个 prime 完整列表（无错漏）
  - P1 身份描述完全一致（"Step, large language model developed by StepFun"）
  - P4 核心论点完全一致（10kg 不可能 / 合理目标 1-2kg 肌肉）
  - first-divergence char idx：P1=14, P2=51, P4=33（在 think 段早期 high-entropy token 翻牌）
- 来源：`progress/teammate-16.md:32-91`

### 6.8 JIT cache 路径
- M1 编译累计 282.5 s（`module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2` 81.7s + `_swiglustep_..._Stage2` 82.2s 等）
- M2 增量编译 = **0 s**（M1 已编完所有 module，全部 import 命中）
- 来源：`progress/teammate-5.md:46-51`、`progress/teammate-12.md:54`

### 6.9 HF auth 双 token
- HF token 文件需放两处：`~/.cache/huggingface/token` + `/workspace/hf_cache/token`
- 用户：`junlin12amd`
- 来源：`progress/teammate-20.md:55-56`

---

## §7 关键时间节点表

| 时间 / wave | teammate | 事件 | 影响 |
|---|---|---|---|
| Wave 1 | T-1 | 验证 ATOM `align = block_n` + ceil 整除两处 fix 已落地（acff926） | 排除 #A03/#A04 是 baseline 失败根因 |
| Wave 1 | T-2 | 验证 BOS bug 在 gfx942 不触发 + CK SwigluStep 支持 gfx942；**主动识别 NEW-RC-1**（fp8 fnuz vs ocp） | 排除 #A01/#A02；催生 Wave 1 #A05 task |
| Wave 1 | T-3 | 完成 NEW-RC-1 转换链条静态闭环（aiter dtypes → ATOM quant_spec → moe.py 触发 → utils.py 重写）；**抛出 NEW-RC-2 风险** | NEW-RC-1 候选解除；新风险 NEW-RC-2 进入 Wave 3 调查 |
| Wave 2 | lead + T-4 | lead 静态定位 NEW-RC-3；T-4 应用 `fused_moe.py:881-886` patch | per_1x128 prefill 改走 CK 2-stage |
| Wave 3 | T-5 | M1 baseline tp=2 PASS（4/4 prompt 连贯，0 dispatch miss，0 fmoe_g1u1） | M1 闭环；NEW-RC-3 patch 实测验证 |
| Wave 3 | T-6 | NEW-RC-2 fp32 dequant 静态实测（forward absmax=0.093 vs inverse=2.15e6） | NEW-RC-2 方向正确，无须修复 |
| Wave 3 | T-7 | M1 dispatch 覆盖差集=∅；末尾预警 M2 stage2 NPerBlock=64 风险 | 新风险进入 Wave 5 调查 |
| Wave 3.5 | T-9 | review raise F-1 V2 闭环 gap + F-2 HF 引文存疑 + F-3 M2 警告高估 + F-4 V3 重定义 | F-1 block 推动 Wave 4 静态补强 |
| Wave 4 | T-11 | F-1 静态闭环（`moe.py:2150-2170` if 链 + 三 process 分支无条件 normalize）→ strong | NEW-RC-1 论据强度升 strong |
| Wave 5 | T-10 | M2 padding 调用链深挖，发现 ATOM `_process_block_quant` 已自动 320→384 | T-7 警告被 supersede；推荐 Option 0 |
| Wave 5 | T-12 | M2 baseline tp=4 PASS（V4 inter_dim=384 决定性间接证据） | M2 闭环；T-10 假设实测验证 |
| Wave 5 | T-13 | M2 padding 必触发 strong + KPack 根因 medium-strong + T-7 §5.4 supersession | NEW-RC-3 / KPack 根因强度量化 |
| Wave 5.5 | T-14 | review raise F-1 prompt 输出"小分化"淡化（block）+ V4/V5 措辞夸张 | F-1 block 推动 Wave 6 量化补强 |
| Wave 6 | T-16 | max_tokens=512 重跑 M1+M2，P3 byte-identical 143/143 + 4 项 correctness 全对 | F-1 闭环，证否"数值漂移变 bug" |
| Wave 6 | T-17 | T-14 7 findings 同步到 PROJECT_SUMMARY/FINAL_REPORT/SESSION_HANDOFF | wave 5/6 close + PROJECT CLOSED |
| Wave 7 | T-18, T-19, T-20 | doc-tidy（一致性 / 可读性 / quick-start 包） | 项目正式 close 文档化 |

---

## §8 给 writer 的建议

### 8.1 适合 sequenceDiagram 的事件流（推荐 1 个，节省图表预算）
- **NEW-RC-3 修复链路**：lead Explore → T-4 patch（含 L932 风险检查）→ T-5 实测 0 fmoe_g1u1 → T-12 M2 复用同 patch → 全程"一行 patch + 双 baseline 复验"是项目最戏剧化的环节，sequenceDiagram 能把 lead/T-4/T-5/T-12 四个 actor 的交互画出来

### 8.2 适合 flowchart 的事件流
- **NEW-RC-1 转换链路**（`safetensors e4m3fn → aiter dtypes.py 静态映射 fnuz → ATOM quant_spec _infer_dtype → Fp8MoEMethod.__init__ need_normalize=True → process_weights_after_loading → _process_block_quant → _normalize_weights_and_scales → utils.py:79 (NaN 重写 + scale ×2) → fused_moe q_dtype=fnuz`）：节点多、单向数据流，flowchart TD 最自然
- **M2 padding 链路**（`config.json moe_intermediate_size=1280 → TP 切 320 → create_weights buffer=320 → loader 加载 fp8 ckpt → process_weights_after_loading → _process_block_quant align=128 → inter_pad=ceil(320/128)*128=384 → zero-pad w13/w2 → shuffle_weights → fused_moe.apply 看到 inter_dim=384 → NPerBlock=128 主路径`）：层次清晰，flowchart TD 表达切分点 + padding 时机
- **总体迁移流程**（gfx950 → gfx942 从 0 到 PASS）：flowchart TD，三大 RC 节点平行 + Wave 时序竖向

### 8.3 适合 graph LR 的关系图
- **三大 RC 关系图**：NEW-RC-1/2/3 之间相互独立（不同 root cause、不同修复点），但共同决定 fp8 forward 数值正确性；M2 padding 与三大 RC 并列，是 tp 升级时遇到的 dispatch 兼容性问题。可用 graph LR 画"四个独立 root cause → 共同导致 PASS"的并列结构

### 8.4 适合表格的内容（推荐合并）
- **三仓改动 summary 表**（来源 §1.3 三仓 commit 表 + §5.1 禁区清单）
- **dispatch 路径对比表**（gfx950 vs gfx942 各 op 路由差异）：可整合 T-2 §A01 BOS dispatcher 路径 + T-7 §3.3 fused_moe_1stage_dict + T-7 §4.3 codegen heuristic_dispatch 表
- **M1 vs M2 对照表**（来源 `progress/teammate-12.md:223-235`）：inter_dim / block_m / q_dtype / dispatch miss / JIT 增量编译 / PASS 结果

### 8.5 整体时间线的"高潮点"
1. **T-2 主动识别 NEW-RC-1**（Wave 1）——非分配的 task，由 reviewer 顺手发现，是项目早期最关键的一次"扩大调查范围"决策
2. **lead Explore agent 静态定位 NEW-RC-3**（Wave 2）——从一行注释 "ck has better performance so disable assembly kernel" 但代码并未禁用，反查到整条 ASM 错路由
3. **T-6 fp32 dequant 实测 7 数量级差距**（Wave 3）——"数值实测 + 横向对比"是项目里最强的单一静态证据
4. **T-10 揭穿 T-7 警告**（Wave 5）——"上游 padding 自动规避下游缺失 instance"是项目最优雅的 root cause 位移
5. **T-12 M2 实测 inter_dim=384 决定性证据**（Wave 5）——T-10 静态预测被实测一锤定音
6. **T-16 P3 byte-identical 143/143**（Wave 6）——彻底证否"TP 数值漂移变 bug"假设的最干净证据

---

## 收尾存档

- tool calls 累计：~10 次（远低于 15 上限）
- 已完成：18 份 progress 全部读完，按 RC 组织事件流，dead end 已省略
- 关键发现：
  1. **三大 RC 独立**：NEW-RC-1（dtype 映射 + scale ×2 自动）、NEW-RC-2（验证现状方向正确，无须修）、NEW-RC-3（一行 patch）三者无依赖
  2. **M2 padding 与三大 RC 并列**：是 tp 升档触发的 dispatch 兼容问题，由 ATOM 既有 padding 机制自动规避
  3. **修复点极少**：实际只有 1 处源码改动（`aiter/fused_moe.py:881-886` NEW-RC-3 patch），其他均靠既有代码 / 数学反证
  4. **PASS 强度演化**：M1 PASS（端到端定性）→ T-9 raise V2 gap → T-11 静态强 → M2 PASS → T-14 raise prompt "小分化" → T-16 byte-identical 量化闭环
- 给 writer 的建议（详见 §8）：
  - 1 个 sequenceDiagram 给 NEW-RC-3
  - 2-3 个 flowchart 给 NEW-RC-1 转换链路 + M2 padding 链路 + 总体迁移流程
  - 1 个 graph LR 给三大 RC + M2 padding 关系
  - 3 个表格：三仓改动 summary / dispatch 路径对比 / M1 vs M2 对照
- 红线遵守：
  - 未修改任何源文档（仅 Read，仅 Write 本 progress）
  - 所有引用真实存在的 file:line（已抽查 T-2 §A01 / T-3 §4 / T-4 patch / T-6 §3.2 / T-7 §5.4 / T-10 §1 / T-12 V4 / T-13 §1 / T-16 §2.1 全部有效）
  - 未复述 KNOWN_FACTS（重复内容仅用于事件流叙事必要的串联）
  - 错误尝试已省略（T-3 提的 `weight_scale_inv` 反向假设、T-7 M2 stage2 miss 警告作为"被 supersede"事件保留必要信息）
  - 中文输出
