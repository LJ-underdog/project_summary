# 代码位置 + 里程碑 commit 链

> 实际 kernel/host 代码**不在本 summary 库**,在 ROCm/composable_kernel 的 fork 里。本文件是权威指针。

## Fork 仓库

- **Fork**:`https://github.com/LJ-underdog/composable_kernel`(ROCm/composable_kernel 的 fork,上游默认分支 develop)
- **推荐分支(与讲义 commit 映射对得上)**:`hstu_attention_fwd_bwd_v2` @ **`a86529dc`**(2026-06-25 重写出的干净里程碑历史,新 hash)
- 旧分支:`hstu_attention_fwd_bwd` @ `048f0a9a`(**代码树与 a86529dc 逐字节一致**,tree hash 都是 `bfb00452…`;只是 commit 历史/hash 不同。能用,但里程碑 hash 对不上讲义映射表)
- 上游 `origin` = `https://github.com/ROCm/composable_kernel`(无写权限)

HSTU 代码目录:`example/ck_tile/18_hstu_attention/`

## 克隆(新机器)

```bash
git clone --branch hstu_attention_fwd_bwd_v2 \
  https://github.com/LJ-underdog/composable_kernel.git ck_hstu
cd ck_hstu && git log --oneline -1    # 应为 a86529dc
```

> push 需要你自己的 fine-grained PAT(user LJ-underdog,Contents:write,**且 `composable_kernel` 本仓库必须在 repository access 里**,只加 fork 不够否则 403)。本 summary 库不含任何 token。

## 里程碑 → commit 映射(权威,doc-series-spec §5)

| Mx | commit | 中文名 / slug |
|----|--------|---------------|
| M1 | `1b3c90b4` | SiLU MAIN 闸门 / silu-gate |
| M2 | `9d129c88` | HSTU 5 因子 mask / mask |
| M3 | `94174bd9` | jagged 变长 / jagged |
| M4 | `3573f083` | group 模式 / group |
| M4b| `180a8acb` | 修 P1-1(causal=0+target)/ p1-1-fix |
| M5 | `476bc16a` | softmax 路径 / softmax |
| M5b| `f7db567d` | group softmax / group-softmax |
| M6 | `48673726` | deterministic dQ / deterministic |
| M6b| `ecda0f06` | group deterministic / group-determ |
| M7a| `8b1fab06` | fp16 加宽 / fp16 |
| M7b| `c9fe2891` | 对称 hdim{64,96,128,256} / hdim |
| M7c| `fc13643e` | 非对称/非典范 hdim via pad / hdim-pad |
| cross | `f2f55622` | cross-attention seqlen_q≠kv / cross-attention |
| M8 | `a86529dc` | perf(MI+B2+B3)/ perf(= 分支顶端) |

> 想看某里程碑改了啥:`git show --stat <commit>`;看具体 diff:`git show <commit> -- <file>` 或 `git diff <commit>^ <commit>`。讲义 HTML 里的行号都锚定各自 commit,用 `git show <commit>:<path>` 复核。

## 能力边界(a86529dc 现状)

SiLU + softmax × 全模式(batched/jagged/group)× self + cross-attention(seqlen_q≠seqlen_kv 全方向)× 全 5 因子 mask × causal{0,1} × bf16 + fp16 × hdim_qk/hdim_v ∈ (0,256] 任意(对称+非对称+非典范 via head-dim pad)× dQ atomic + deterministic(全模式)。

- 对拍套件:**253/253 exit 0**
- perf(M8):MAIN causal 1.6× / window ~10×
- **真 reject**:hdim>256
- **out-of-scope(未做,需拍板)**:target_in_kv、独立 dO layout、非方形 tile
- 尚未对上游 ROCm 提 PR
