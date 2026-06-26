# M7a fp16 WIP 评估 — coder done (2026-06-11)

任务: 把上轮 stash 恢复的 **M7a fp16 WIP**(纯 dtype 加宽,hd64,fp16 复用 bf16 同模板码路)
**建成 + 对拍 + 补套件** 验真。非重写。全程 `-attn_scale=1.0`。基线 HEAD=`d4fb2884`(M6b)。

诚实纪律先行: **下面没有任何 FAIL 被标 promoted。** 套件那一格"FAIL"是 reject-fp16 过期期望(fp16
已实现 → 该 reject 案应升级为 pass,已升级),不是 bf16 回归。详见 §4。

---

## 1. 构建(BUILD_DEV=OFF, gfx950)—— 0 error
- 先 `cmake -B build ...`(BUILD_DEV=OFF 完整配置行)重 re-glob 新增 fp16 instance/entry(`file(GLOB)` 仅配置期求值)。
- `cmake --build build --target tile_example_hstu_attention_bwd -j` → **[121/122] / [122/122] Linking … exit 0**。
- 证据: `grep -cE "error:" runs/build-M7a.log` = **0**;binary 新鲜重建
  `build/bin/tile_example_hstu_attention_bwd` 422MB @ 2026-06-11 04:08。
- 仅 host-side 78 warnings(既有 CK 模板告警,与 M7a 无关)。
- WIP 改面(已逐一读核,结构忠实): CMakeLists(+fp16 fwd/bwd entry+instance glob)、
  example_hstu_attention_bwd.cpp(runtime `-prec` 选 fp16_t/bf16_t + 4 处 fwd/bwd wiring +
  `get_bwd_elimit<fp16_t>` rtol5e-3/atol1e-2)、generate_instances.py(dtype 轴加 fp16)、
  api.hpp(2 fp16 bwd extern);新文件: fp16 no_group/group entry ×2、8 batched fp16 instance、
  fp16 instances_ref.hpp。entry 与 bf16 版逐字镜像,仅 `bf16_t→fp16_t`。

## 2. fp16 对拍 sweep —— 66/66 PASS, 0 FAIL
脚本 `test/sweep_fp16.py`,全 `-prec=fp16 -attn_scale=1.0`,fp16 elimit rtol5e-3/atol1e-2(**比 bf16 紧,没放水**)。
日志 `runs/run-M7a-sweep.log`。覆盖:

| 维度 | 覆盖 | 结果 |
|---|---|---|
| 模式 | batched / jagged(per-batch 异 seqlen)/ group(g=2,3,4) | 全 PASS |
| activation×causal | SiLU/softmax × causal{0,1} | 全 PASS |
| mask 5 因子 | window/contextual/min_full/num_target 各自+组合 | 全 PASS |
| determ | no_group(SiLU/softmax,multi-split)+ group | 全 PASS |
| 非整除 seqlen | 200/130/512/96/32/1/7 | 全 PASS |
| 单 batch / tiny | b1-seq512 / jagged 1,7 | 全 PASS |
| 关键交叉 | causal=0+num_target(P1-1 老洞,batched/jagged/group×SiLU/softmax) | 全 PASS |
| M6b 老洞触发 | group 同组多 batch 异 seqlen + 长 batch 大 target + window(gtrig-sm/silu) | 全 PASS |

误差量级(max_abs_err vs max\|ref\|,代表性):
- SiLU 路: err ~5e-4 … 4e-3, \|ref\| ~2 … 11 → rel ~1e-3 量级,远低于 atol1e-2。
- softmax 路: err ~3e-5 … 2e-4, \|ref\| ~0.08 … 2.3 → rel ~1e-4 量级。
- determ multi-split(seq512): err ~4e-3, \|ref\| ~10 → rel ~4e-4。
全表见 `runs/run-M7a-sweep.log`(每案 dQ/dK/dV err+\|ref\| 全列)。

## 3. fp16 数值风险自检 —— 无溢出,未触边界
- fp16 max=65504。sweep 全程 max\|ref\| 最高仅 **~10.9**(silu-j-spread),输入 O(1)(harness 默认
  uniform/可选 normal),S/exp/LSE 内部累加是 **float**(fp16 仅 I/O)→ 量级安全。
- **所有 softmax 配置(指数路)均 PASS,无一例 fp16 溢出 FAIL**。
- 结论: 当前测试包络内 **无 fp16 已知边界**。(若未来喂极端大 magnitude 输入需另测;现 contract 内无。)

## 4. bf16 零回归 —— 真零回归
`python3 test/run_bwd_tests.py`(改套件前)→ TOTAL 91 / PASS 90 / **FAIL 1** / exit 1。
**唯一那格 FAIL = `reject-fp16` 期望过期**,不是 bf16 算错:
```
CASE reject-fp16 (expect=reject) -> FAIL
EXIT 0  PASS=3 FAIL=0
REASON REGRESSION/false-positive: path now produces all-PASS -> upgrade this case to expect='pass'
```
即:该案设计为"fp16 未实现 → harness guard 应拒绝";M7a 实现了 fp16,它现在真算对(全 PASS),
runner 按设计提示"升级为 pass"。**所有 M0–M6b bf16 PASS/REPRO 案(89+repro)全绿**,bf16 码路未被 WIP 碰坏。
(三个禁改 pipeline / dispatch 逻辑 WIP 未改;改面只在 entry/instance/harness/generator/CMake。)

## 5. 补 fp16 测试套件 —— 完成,全 pass 断言
- `reject-fp16` 删除 → 升级为 `pass-fp16-*` 块(`test/run_bwd_tests.py`)。
- 新增 **14 个 fp16 pass 案**: no_group batched/jagged × SiLU/softmax × causal{0,1} ×
  (combo / num_target / 非整除) + group(SiLU hetero / softmax c1 / softmax c0+target)+
  determ(no_group SiLU multi-split、no_group softmax jagged+target、group softmax)。
- 新增 **2 个 fp16 determ byte-identical 复现案**(`repro-fp16-det-softmax-seq512`、
  `repro-fp16-gdet-silu-g2`;`-prec=fp16` 末值覆盖前缀 bf16,已验末值生效)。
- 改后全套: **TOTAL 106 / PASSED 106 / FAILED 0 / SKIPPED 0 / exit 0**
  (= 旧 91 − 1 reject-fp16 + 14 fp16 pass + 2 fp16 repro)。
- 日志 `runs/test-20260611-041239.log`。所有 fp16 案均 **pass 断言**(exit0+3×PASS),不再是 reject。

## 6. 完成度 / 缺口(如实)
- ✅ 构建 0 error;fp16 sweep 66/66;bf16 真零回归;新套件 106/106 exit 0;fp16 determ 逐位复现。
- ✅ 能力边界扩展: SiLU+softmax 全模式 × 全 5 mask × causal{0,1} × **bf16+fp16** × hd64 × atomic+determ。
- ⚠️ 范围限定(WIP 本就如此,非缺口隐瞒): **仍 hd64、hdim_qk==hdim_v**;hdim{96,128,256} 与
  hdim_qk≠hdim_v 仍属 M7b/M7c(`reject-hdim128` 仍在套件守 throw)。
- ⚠️ 盲区继承 M5/M5b: LSE 数值由 GPU fwd 产、对拍两侧共用同一份(无法独立验 LSE 数值),
  靠 fwd 里程碑兜底——fp16 未改变此结构。
- 状态据实: **全绿,待 lead + reviewer 闭合(in-progress)**,非自行 promoted。

## 7. 产物清单
- `runs/build-M7a.log`(0 error)、`runs/run-M7a-sweep.log`(66 案全表)、`runs/test-20260611-041239.log`(106/106)。
- `test/sweep_fp16.py`(新)、`test/run_bwd_tests.py`(改: reject-fp16→14 fp16 pass + 2 fp16 repro)。
- `candidates.jsonl` 加一行 M7a(status: in-progress)。
- **未 commit;未动 reference / 库 kernel 逻辑 / promoted 路。**
