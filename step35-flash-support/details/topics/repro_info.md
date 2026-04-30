# 复现信息收集 (Step-3.5-Flash FP8 tp=4)

收集日期：2026-04-28
信息来源：实测 git/文件系统/已验证 V06 报告

---

## 1. 代码版本

### ATOM
- 仓库：`/home/hanchang/junlin12_repos/atom`（**git push 必须从此处**）
- 分支：`feat/step3p5-flash-support`
- HEAD（关键 FP8 tp=4 修复）：`ccb64621` — `fix: support FP8 block-quantized inference at tp=4 (inter_dim=320)`
- 近期相关 commits（按新→旧）：
  - `acff926d` fix(moe): correct FP8 blockscale inter_dim padding align for all tp configs
  - `3696345e` revert(moe): restore FP8 blockscale inter_dim padding align logic
  - `270fee71` fix(moe): use align=64 for FP8 blockscale to remove inter_dim=320 padding
  - `ccb64621` **fix: support FP8 block-quantized inference at tp=4 (inter_dim=320)** ← Fix 3 (floor→ceil)
  - `841dc4ee` fix: pass correct block_shape in Fp8MoEMethod.get_fused_moe_quant_config
  - `26585d42` fix: pad inter_dim in UnquantizedFusedMoEMethod for gfx950 tp=4/8
  - `c732b993` feat: add Step-3.5-Flash support and fix MoE weight shuffling on gfx950

### aiter
- 仓库：`/home/hanchang/junlin12_repos/aiter`（**git push 必须从此处**）
- 分支：`feat/step3p5-moe-swiglustep`
- HEAD：`0f8164017` feat(moe): add stage1 NPerBlock=64 blockscale kernels (stage2 pending)
- 关键 commits：
  - `a2883ab37` **fix: remove buggy ASM kernel entry for (N=4096,K=2048) bf16 GEMM on gfx950** ← BOS workaround
  - `0f1015685` fix: exclude blockscale quant types from gfx950 block_m=128 override
  - `169a00879` fix: gfx950 distributed allreduce/allgather compatibility
  - `d01d31dee` chore: bump CK submodule to rebased swiglustep commit
  - `40a402f0c` feat: add ActivationType.SwigluStep enum + codegen + CK submodule bump
  - `825ed208c` fix: CK 2-stage MoE pipeline correctness on gfx950 + SwigluStep Python support

### 运行时仓库（被实际 import）
- `/home/hanchang/aiter` HEAD：`80fb59782` (merge: bring local main into feat/step3p5-moe-swiglustep)
  - **注意**：此 HEAD 不包含 `a2883ab37`，CSV 中仍残留 buggy ASM 行（见 §4）
- `/home/hanchang/ATOM`（无显式分支信息收集，CLAUDE.md 显示其为 ATOM 工作树）

### CK submodule
- 路径：`/home/hanchang/aiter/3rdparty/composable_kernel`
- HEAD：`defd7ad29` Add swiglustep_and_mul branches to gridwise_moe_gemm (4 paths, hardcoded 7.0f clamp)
- 上游基线：`fdf4bb7fc` [rocm-libraries] ROCm/rocm-libraries#6653

---

## 2. 安装命令

### Python 环境
- Python：`/opt/venv/bin/python`，3.12.3
- ROCm：`/opt/rocm`（gfx950 target，8x MI350X）

### aiter
- 配置文件：`setup.py`、`requirements.txt`、`pyproject.toml`（无 `*.sh` 安装脚本）
- 关键环境变量（`setup.py` 提取）：
  - `BUILD_TARGET=auto`（默认）
  - `PREBUILD_KERNELS=0`（默认；JIT 编译）
  - `PRETUNE_MODULES=""`（默认）
  - `ENABLE_CK=1`（默认；启用 CK 编译）
- requirements：`pandas, pytest, psutil, matplotlib, pyyaml, einops, pybind11>=3.0.1, ninja, flydsl==0.1.4`
- 安装命令（推断标准流程）：
  ```bash
  cd /home/hanchang/aiter
  /opt/venv/bin/pip install -r requirements.txt
  /opt/venv/bin/pip install -e .
  ```

### ATOM
- 配置文件：`pyproject.toml`（无 `setup.py`、`requirements.txt`、`*.sh`）
- 包名：`atom`，依赖：`pybind11, transformers==5.2.0, zmq, xxhash, fastapi, psutil, protobuf, uvicorn, aiohttp, datasets, openpyxl, tqdm`
- 安装命令（来自 `/home/hanchang/ATOM/CLAUDE.md`）：
  ```bash
  cd /home/hanchang/ATOM
  /opt/venv/bin/pip install -e .
  ```

---

## 3. 关键 workaround

### Workaround 1：BOS bug ASM kernel CSV 行删除
- 根因（commit `a2883ab37` 描述）：tuning entry `M=16384,N=4096,K=2048,bf16,asm,bf16gemm_bf16_tn_256x256` 导致 dispatcher 对 M ∈ [8193, 16384] 全部选 `_ZN5aiter24bf16gemm_bf16_tn_256x256E`（通过 padded_M=16384）。该 ASM kernel 对非 256 对齐 M（如 8209-8223）输出错误（diff ≈ 392 vs ref_max ≈ 247），导致 Step-3.5-Flash o_proj 在 tp=4 长序列 prefill（M ≥ 8209）产生 all-BOS。
- 受影响文件：`/home/hanchang/aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv`
- **现状（实测）**：`/home/hanchang/aiter` 当前 HEAD `80fb59782` 不包含 `a2883ab37`，CSV 仍含 N=4096,K=2048 行（line 35-44，splitk_clean variant；以及上游被 dispatcher 用作 padded_M 选取的 `bf16gemm_bf16_tn_256x256` 行）。**复现 tp=4 长序列推理前必须删除该行**。
- 修复方法：checkout junlin12_repos/aiter `a2883ab37`，或手动删除 `glm5_bf16_tuned_gemm.csv` 中触发 `_ZN5aiter24bf16gemm_bf16_tn_256x256E` 的 N=4096,K=2048 entry。

### Workaround 2：FP8 tp=4 scale shard ceil 整除（已合入 ATOM `ccb64621`）
- 根因（V06 报告）：FP8 blockscale loader 对 inter_dim=1280, tp=4 用 floor 整除，得 `load_shard_size = 10 // 4 = 2`，scale block [8,9] 永远未被任何 rank 加载，残留 `torch.ones()` 默认值 → gibberish。
- 修复位置：`/home/hanchang/ATOM/atom/model_ops/moe.py`
  - `_load_w13` L2305-2307（gate/up shard）
  - `_load_w2` L2347-2349（down shard）
- 修复代码：`load_shard_size = (loaded_weight.shape[shard_dim] + self.tp_size - 1) // self.tp_size`
- 已通过 V06 Exp2 验证（tp=4 端到端 PASS）。

### Workaround 3：GPU5 硬件异常
- GPU5 单 tensor 操作 ~700ms（正常 <1ms）。**避免使用 CUDA_VISIBLE_DEVICES 包含 5**。
- tp=4 推荐 GPU 0,1,2,3；tp=2 用 4,6（V06 Exp4 配置）。

### Workaround 4：缓存清理
- 修改 ATOM 代码后必须清缓存（避免 stale CUDAGraph 编译产物）：
  ```bash
  rm -rf /root/.cache/atom/*
  ```

### Workaround 5：cwd 必须不在 aiter 仓库内
- 来自 MEMORY：`运行 python 必须先 cd /tmp &&`（否则 aiter 被识别为 namespace package）。

---

## 4. 模型权重路径

- BF16：`/root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash/snapshots/ab446a3de5e171ea341227e24bb1f090e1b771f7/`
- FP8：`/root/.cache/huggingface/hub/models--stepfun-ai--Step-3.5-Flash-FP8/snapshots/6eebda59dd87ca5729648ec7cfed0becfceb273e/`

模型加载使用 HF id：`stepfun-ai/Step-3.5-Flash-FP8` 与 `--trust-remote-code`。

---

## 5. 运行命令

### FP8 tp=4 端到端（V06 Exp2，已验证 PASS）
```bash
cd /tmp && \
CUDA_VISIBLE_DEVICES=0,1,2,3 ATOM_LOG_LEVEL=WARNING AITER_LOG_LEVEL=WARNING \
  /opt/venv/bin/python -m atom.examples.simple_inference \
  --model stepfun-ai/Step-3.5-Flash-FP8 --trust-remote-code \
  --tensor-parallel-size 4 --level 0 --temperature 0 --max-tokens 128 \
  --max-num-batched-tokens 4096 --max-num-seqs 2048
```

### FP8 tp=2 回归（V06 Exp4，已验证 PASS）
```bash
cd /tmp && \
CUDA_VISIBLE_DEVICES=4,6 ATOM_LOG_LEVEL=WARNING \
  /opt/venv/bin/python -m atom.examples.simple_inference \
  --model stepfun-ai/Step-3.5-Flash-FP8 --trust-remote-code \
  --tensor-parallel-size 2 --level 0 --temperature 0 --max-tokens 128 \
  --max-num-batched-tokens 4096 --max-num-seqs 2048
```

### simple_inference.py 内置 prompts
4 个固定 prompts（`/home/hanchang/ATOM/atom/examples/simple_inference.py:38-43`）：
- "introduce yourself"
- "list all prime numbers within 100"
- "1+2+3=?"
- "如何在一个月内增肌10公斤"

均通过 `tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)` 包装。

---

## 6. 期望输出（V06 Exp2 实测）

| 指标 | 实测值 | 通过标准 |
|------|--------|---------|
| TTFT | 86 ms | < 200 ms |
| TPOT | 12-13 ms | < 20 ms |
| 输出连贯性 | 4/4 正常 | 无 gibberish |
| 无 BOS-spam | 是 | `<s>` ≤ 1 |
| 无 crash/ValueError/shape mismatch | 是 | — |

示例输出（"introduce yourself"）：
> "Hmm, the user simply asked me to introduce myself. This is a straightforward request..."（在 max_tokens=128 处截断）

性能对比（来自 MEMORY）：
- FP8 tp=4 vs BF16 tp=4：decode TPOT 13ms vs 17ms（FP8 快 24%）
- FP8 tp=4 vs FP8 tp=2：tp=2 TTFT=78/85ms、TPOT=14/13.5ms；tp=2 在 decode 性价比更优

---

## 7. 复现步骤总结

1. 确保 `/home/hanchang/aiter` checkout 到包含 `a2883ab37` 的版本，或手动删 `glm5_bf16_tuned_gemm.csv` 中 N=4096,K=2048 行（**否则 tp=4 长序列 prefill BOS-spam**）。
2. 确保 `/home/hanchang/ATOM` 包含 commit `ccb64621`（FP8 tp=4 ceil fix），即 `_load_w13` L2305 与 `_load_w2` L2347 使用 ceil 整除。
3. 安装：`cd /home/hanchang/aiter && pip install -e .`；`cd /home/hanchang/ATOM && pip install -e .`。
4. 清缓存：`rm -rf /root/.cache/atom/*`。
5. `cd /tmp` 后运行 §5 中的 FP8 tp=4 命令（GPU 0,1,2,3，避开 GPU5）。
6. 检查输出：4 个 prompt 全部连贯，TTFT≈86ms、TPOT≈13ms。

---

## 8. 风险与未覆盖项

- V06 Exp2 用的是默认 prompts（短输入），未在该报告中显式跑长序列（≥8209 tokens）端到端 BOS 验证；该项由 V07 覆盖（详见 `verification_pipeline/results/v07_longseq_bos.md`）。
- tp=8 因 GPU5 硬件异常仍处于阻塞状态。
- aiter HEAD `0f8164017` 中的 NPerBlock=64 stage1 kernel 为 pending（"stage2 pending"），与 ccb64621 的 align=64 路径相关，复现时若 inter_dim=320 路径走 NPerBlock=64 dispatch 需注意 stage2 是否就绪。
