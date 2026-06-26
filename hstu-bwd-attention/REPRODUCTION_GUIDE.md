# 复现指南 — 新机器上接手 HSTU bwd

> 目标:在另一台 gfx950 机器的 Claude Code 上,快速恢复到当前工作进度(M0–M8 + cross 全 promoted,14 篇里程碑讲义全部成稿)。

## 0. 前置:硬件 / 环境

- **GPU 必须 gfx950 / MI350X / CDNA4**(本项目 kernel 走 `BUILD_HSTU_FOR_GFX95_ONLY` + `#ifdef __gfx950__`,其它芯片编不出/跑不对)。`rocminfo` 确认。
- ROCm 装在 `/opt/rocm`(hipcc / amdclang++);需 cmake + ninja。
- 容器内 home 一般是 `/root`(若不同,把下面路径里的 `/root` 替换掉)。

## 1. 恢复 memory(让新 cc 有上下文)

见 `memory/INSTALL.md`。一句话:把 `memory/MEMORY.md` 和 `memory/hstu-bwd-project.md` 放进新机器的 `~/.claude/projects/<your-workspace-slug>/memory/`,新 cc 启动就会带上 HSTU 项目上下文。

## 2. 拉代码(实际 kernel 在 fork,不在本库)

见 `code_location.md`。

```bash
git clone --branch hstu_attention_fwd_bwd_v2 \
  https://github.com/LJ-underdog/composable_kernel.git /root/workspace/ck_hstu
cd /root/workspace/ck_hstu && git log --oneline -1   # a86529dc
```

## 3. 构建(★必须 BUILD_DEV=OFF)

不加 `-DBUILD_DEV=OFF` 会被 `-Werror -Weverything` 拦住(新 clang 诊断当错,fwd 都编不过)。

```bash
cd /root/workspace/ck_hstu
cmake -B build -G Ninja -DCMAKE_PREFIX_PATH=/opt/rocm -DGPU_TARGETS=gfx950 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
      -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ -DBUILD_DEV=OFF
cmake --build build --target tile_example_hstu_attention_bwd -j$(nproc)   # bwd
cmake --build build --target tile_example_hstu_attention     -j$(nproc)   # fwd(对拍产 O/LSE 用)
```

> gfx950 专属开关(CMake 自动加):`-fno-slp-vectorize`、`-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3`。
> ⚠ 构建很吃内存:别同时跑多个 `-j128` 全量 build(maxk_256 单 TU 吃几 GB,503GB RAM 也能 OOM)。group entry 单 TU 约 14min。

## 4. 跑回归套件(验证基线)

测试套件在本库 `impl/test/`(**不在 fork 的 git 里**,务必从这里取)。它会调用上面 build 出的可执行文件对拍 CPU reference。

> **路径约定**:下文及 §5 出现的裸 `test/...` 都指**第 4 节 `cp` 之后的 workspace 内相对路径**(`/root/workspace/hstu-bwd-impl/test/`)。在本 bundle 里这些文件位于 `impl/test/`。

```bash
# 把 impl/test 放到一个 workspace(脚本里路径假设 /root/workspace/hstu-bwd-impl/)
mkdir -p /root/workspace/hstu-bwd-impl && cp -r impl/* /root/workspace/hstu-bwd-impl/
cd /root/workspace/hstu-bwd-impl
python3 test/run_bwd_tests.py    # 期望 253/253 exit 0
```

- 脚本默认路径(`run_bwd_tests.py` 顶部):`DEFAULT_BIN=/root/workspace/ck_hstu/build/bin/tile_example_hstu_attention_bwd`、`DEFAULT_BUILD_DIR=/root/workspace/ck_hstu`、`LOG_DIR=/root/workspace/hstu-bwd-impl/runs`。**照第 2/3 节的布局 clone+build 即与默认值吻合,无需改文件**。布局不同就用 CLI 覆盖:`python3 test/run_bwd_tests.py --bin <path> --build-dir <dir> --log-dir <dir>`(比改脚本干净)。
- 离线 mask 校验器:`test/validate_tile_range_y.cpp`(GetTileRangeAlongY superset,~1.97M checks GREEN);收紧类优化的 silent-wrong 硬 gate。**需单独编译**(独立 host 程序,不依赖 GPU):`g++ -std=c++17 -O2 test/validate_tile_range_y.cpp -o /tmp/vtry && /tmp/vtry`(对达成 253/253 非必需,仅在改 GetTileRangeAlongY 类收紧时跑)。
- co_symbols 零回归校验:`test/co_symbols.py`(gfx950 设备符号 byte-identity)。

## 5. 验证铁律(改代码前必读)

1. **验证 = 对拍 CPU reference**(`reference_hstu_attention_bwd.hpp`,bf16 rel≤2e-2/abs≤5e-2;fp16 rel5e-3/atol1e-2)。
2. 对拍**必须用 `-attn_scale=1.0`**(梯度量级有意义,别让 ref 太小巧合 PASS)。
3. 硬件结论查 rocm-ref(`/tmp/rocm-ref/`,gfx950=CDNA4 口径),**别臆造**(踩过坑:CDNA4 每 CU LDS = **160KB**,不是 64KB;64KB 是 CDNA3)。
4. silent-wrong 比 throw 危险:未支持组合要么对拍覆盖、要么显式 throw。
5. 新特性要 **causal × mask因子 交叉覆盖**(P1-1 / M6b 两次覆盖洞都因只测"对角线")。

## 6. 多 pane 团队(可选,文档/review 工作流)

skill `agent-team`(`/root/.claude/skills/agent-team/`):
```bash
WAVE=<name> bash /root/.claude/skills/agent-team/bootstrap.sh   # 建 tmux claudeteam,4 pane
bash /root/.claude/skills/agent-team/readiness.sh               # 等 pane 就绪
```
- 派单:Write prompt 到 `/tmp/*.md` → `tmux send-keys -t claudeteam:0.N "请读取 <file> 并严格按其执行" Enter`。
- **大坑**:pane 忙时 Enter 会被吞,甚至刚起的 pane 首个 Enter 也可能被吞 → 发完**回读** `tmux capture-pane -p -t claudeteam:0.N` 确认 ctx 在涨/输入框已清空;没提交就补发裸 Enter。
- 新起的 pane 会先停在 "trust this folder" 页,需先发一个 Enter 选 "1. Yes"。

## 7. 当前进度快照(写于 2026-06-26)

- **代码**:M0–M8 + cross-attention 全部 promoted、已提交、工作树干净(a86529dc)。
- **讲义系列**(克隆 M0 体例的统一图文 HTML,在 `reports/` 的 `-20260625` 篇):M0–M8 共 **14 篇里程碑全部成稿**(2026-06-26 完成,均经 lead review 抽查行号)。派单卡在 `design-notes/doc-*-dispatch.md`,硬规格 `design-notes/doc-series-spec.md`(供新机器复用体例)。
- 活状态以 `HANDOFF.md` 为准。
