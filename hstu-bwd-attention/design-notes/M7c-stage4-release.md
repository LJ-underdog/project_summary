# M7c Stage 3 → Stage 4 放行(coder pane 0.1)

**lead 亲核通过 Stage 3**:改面仅 4 文件(group/batched dispatch + shape + harness,**pipeline/kernel/reference 未碰**);独立 co_symbols verify **294/294 byte-identical 0 diff**(group canonical pad=0 逐位不变);group poison 实跑(g2 asym 64/128 + g2 determ 128/256)PASS + numeric_pass=true;canonical 套件 **172/170/0/2 exit 0**。

## 放行 Stage 4 — 全矩阵收尾 + sign-off(draft §7 Stage 4)
1. **套件永久化 M7c 覆盖**:把现 2 个 SKIP(`hdim=100` 非典范、`64/128` asymmetric)**转成真 pass 断言**(带 `-poison_pad=1` 跑,这样套件本身就用 poison 硬证 OOB,而非裸 PASS)。再补一组代表性 **pass-asym/pass-noncanon** 案进 `test/run_bwd_tests.py`:no_group + group × {bf16,fp16} × 代表 pair(含双方向 + */256 determ + 每案 P1-1 cross)。
2. **保留真 reject**:`reject-hdim-gt256`(hdim=512)留着,防 guard 静默消失。
3. **全矩阵复跑**:`sweep_M7c.py`(batched)+ group sweep 全 PASS 存档;新套件 TOTAL 记清(应为 旧 + M7c pass 案,0 FAIL 0 SKIP 或仅保留必要 skip,exit 0)。
4. **能力边界**:SiLU+softmax × batched/jagged/group × 全 5 mask × causal{0,1} × bf16+fp16 × hdim_qk/hdim_v∈(0,256] 任意(对称+非对称+非典范 via pad)× atomic+determ;真 reject hdim>256。
5. **done.md**(`docs/M7c-done.md`):全阶段证据汇总(Stage0–4)、改面、byte-identity、batched+group poison sweep 数、新套件 TOTAL、**诚实限制**(dq_acc store-skip 由代码核实 production 安全+真实列兜底;R9 hdim=100 跑通非 reject;非方形 tile / hdim>256 = out-of-scope)。
6. **candidates.jsonl** 加 M7c 行(status 据实:全绿=in-progress 待 reviewer+lead 闭合,**别自标 promoted**)。

## 纪律
- 容差禁松;带任何 FAIL/裸 PASS 不充数;日志数字与结论一致(M6b 教训)。
- **不 commit**(lead 闭合后统一 commit 立里程碑)。
- 完成后 pane 里一句话报概要 + 新套件 TOTAL,停,等 lead 派 reviewer 做对抗+文档 review → 四方闭合。
