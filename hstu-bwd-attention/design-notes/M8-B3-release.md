# M8 B2 → Stage 3 放行(coder pane 0.1)

**lead 亲核通过 B2**:套件 253/253 + 针对性 causal 边界对拍(contextual+target、cross kv>q)PASS;MAIN 加速实证 canonical silu 0.2088ms vs 基线 0.3327ms=**1.59×**(对上你报的 1.60×);co_symbols surgical(仅 causal 实例变)。离线校验器你已跑 1.29M GREEN。

## 放行 Stage 3 = B3 GetTileRangeAlongY 紧致化 — **local/window**(draft 顶部安全要求)
现 WithLocal(window>0)mask 的 GetTileRangeAlongY 仍保守全扫 → window 配置 0 加速(B2 表已证 window256 = 1.00×)。窄 window 浪费最大(scoping [derived] 最高 ~22x,实际 Amdahl 后会小、实测多少记多少)。收紧到 window 真实 Q-row 带(local:col c 只被 [c, c+window)(self)/含 diff_q_kv_len(cross)的行 attend)。

## ★ B3 silent-wrong 硬安全要求(同 B2)
- kM0-aligned superset(start align_down、end align_up/clamp)。
- **contextual 行 q_start=0**;**min_full_attn 全 reach**;**cross diff_q_kv_len 偏移**;window 的**双边带**(下界 c、上界 c+window)都要含;causal+window 组合(window 内且 causal)取交集的安全超集。
- **non-causal 路** WithLocal 也要正确(window 不依赖 causal)。

## ★ B3 验证 4 gate(全过才报 done)
1. **离线 superset 校验器 ALL GREEN**:`test/validate_tile_range_y.cpp` —— **先扩它覆盖 WithLocal 收紧逻辑(window × causal{0,1} × cross × contextual/minfull)**,再跑穷举,证仍 superset。这是挡 under-tighten 最硬 gate。
2. **对拍套件 253/253 exit 0** + 边界 stress(window × causal{0,1} × contextual/minfull/target/cross/非整除 逐项,双向 kv)。
3. **MAIN 加速 vs 基线**:-perf 跑 window 配置(窄/中/宽 window 各档)+ 记 benchmark.csv,MAIN ms 下降(实测多少记多少,Amdahl 诚实归因)。
4. **co_symbols surgical**:仅 WithLocal(has_local)实例设备码变;no_local + fwd byte-identical。

## 完成停报 lead 亲验(这是本期 M8 最后一个 candidate)
- 报告:改面、离线校验器 GREEN、253/253 + 边界、window MAIN 加速(各 window 档 benchmark.csv 前后对比)、co_symbols surgical。
- 之后写 `docs/M8-done.md`(MI + B2 + B3 全证据 + benchmark.csv 加速汇总 + 诚实归因 Amdahl + 暂缓项 B4/B7/B1/trload)+ candidates 加行。
- 不 commit;实测加速多少记多少,别吹模型数。
