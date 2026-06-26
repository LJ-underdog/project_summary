# HSTU Attention Backward — GPU kernel 实现(gfx950 / CDNA4)

## 背景

- **任务**:为 HSTU(Hierarchical Sequential Transduction Units)attention 的 **backward** 实现 GPU kernel。起始状态:仓库只有 855 行 CPU 参考(`reference_hstu_attention_bwd.hpp`),**无任何 GPU bwd**。
- **硬件**:AMD MI350X / **gfx950 / CDNA4**(kernel 走 gfx950-only 路径)。
- **基建**:复用 ROCm composable_kernel 的 ck_tile FMHA bwd 流水线 + codegen。
- **代码所在**:fork `LJ-underdog/composable_kernel`,分支 `hstu_attention_fwd_bwd_v2` @ `a86529dc`(详见 `code_location.md`)。**本 summary 库不含代码,只含上下文/文档/测试/讲义 + 复现指针。**

## 最终能力边界(a86529dc)

| 维度 | 支持 |
|---|---|
| 激活/归一 | SiLU + softmax |
| 模式 | batched / jagged / group |
| attention | self + **cross**(seqlen_q≠seqlen_kv 全方向) |
| mask | 全 5 因子(causal/window/contextual/min_full/num_target)× causal{0,1} |
| dtype | bf16 + fp16 |
| head dim | hdim_qk/hdim_v ∈ (0,256] 任意(对称 + 非对称 + 非典范 via head-dim pad) |
| dQ 归约 | atomic + deterministic(全模式) |

- 对拍套件:**253/253 exit 0**;perf(M8):MAIN causal ~1.3–1.6×(silu 峰值 1.6×/softmax 1.3×)/ window ~4.7–9.8×(窄窗最高,window16 实测 10.4×)。
- **真 reject**:hdim>256。**out-of-scope(需拍板)**:target_in_kv、独立 dO layout、非方形 tile。
- **尚未对上游 ROCm 提 PR。**

## 里程碑链

`M0 脚手架` → `M1 SiLU闸门` → `M2 5因子mask` → `M3 jagged` → `M4 group`(+`M4b` 修 P1-1)→ `M5 softmax` → `M5b group-softmax` → `M6 determ dQ` → `M6b group-determ`(+修 O1+harness bug)→ `M7a fp16` → `M7b 对称hdim` → `M7c 非对称/非典范hdim(pad)` → `cross-attention` → `M8 perf(MI+B2+B3)`。

每个里程碑都"四方闭合"(coder 实现 + reviewer 独立验证 + lead 亲核 + 反误报/byte-identity gate)。commit→里程碑映射见 `code_location.md`。

## 目录导览

| 路径 | 内容 |
|---|---|
| `HANDOFF.md` | **活状态文档,恢复必读**(含环境/构建/候选账本/教训/下一步) |
| `code_location.md` | fork 仓库 + 分支 + 里程碑 commit 映射 + 克隆方法 |
| `REPRODUCTION_GUIDE.md` | 新机器从零接手:环境→拉码→构建(BUILD_DEV=OFF)→跑套件→验证铁律 |
| `memory/` | 当时 cc 的持久 memory(MEMORY.md + 项目 memory)+ `INSTALL.md`(放回 ~/.claude) |
| `design-notes/` | 设计 + 每里程碑 done/review/draft/findings(`/tmp/hstu-bwd-design` 的 .md;含讲义系列硬规格 `doc-series-spec.md` 与派单卡) |
| `impl/` | 回归套件 `test/`(+离线校验器 validate_tile_range_y、co_symbols)+ `candidates.jsonl` 候选账本 + `benchmark.csv` + `docs/`(**不含 GB 级 build log**) |
| `reports/` | 37 篇图文 HTML(浏览器直接开):**14 篇 `-20260625` 统一里程碑系列**(M0–M8+cross,克隆 M0 体例,首选看这套)+ 23 篇旧版/总览/基础篇(早期日期,保留参考) |

## 快速接手(TL;DR)

```bash
# 1. 恢复 memory(见 memory/INSTALL.md)
# 2. 拉代码
git clone --branch hstu_attention_fwd_bwd_v2 \
  https://github.com/LJ-underdog/composable_kernel.git /root/workspace/ck_hstu
# 3. 构建 + 跑套件(见 REPRODUCTION_GUIDE.md,务必 -DBUILD_DEV=OFF)
# 4. 读 HANDOFF.md 接着干
```

## 进度快照(2026-06-26)

- **代码**:M0–M8 + cross 全 promoted、已提交、工作树干净。
- **讲义系列**(克隆 M0 体例的统一 HTML,在 `reports/` 的 `-20260625` 篇):**14 篇里程碑全部成稿**(M0–M8 + cross,M4b 折在 M4 内),均已 lead review 抽查行号通过。派单卡/硬规格留在 `design-notes/doc-*-dispatch.md` / `doc-series-spec.md`(供新机器复用体例)。
- **下一步候选**(见 HANDOFF §6):M8 perf 残项(B1 group TU 拆分/B6 trload/近似 sigmoid,均低 ROI)、out-of-scope 扩展(target_in_kv 等,需拍板)、对上游提 PR。
