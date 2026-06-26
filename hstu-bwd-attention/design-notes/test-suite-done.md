# pane-2 — HSTU bwd 自动化回归测试套件 完成报告

状态:**✅ 完成,实跑全绿(整体 exit 0)**。日期 2026-06-05,GPU = gfx950 / MI350X。

## 交付物
- **runner**:`/root/workspace/hstu-bwd-impl/test/run_bwd_tests.py`(纯 python3 + subprocess,无第三方依赖)
- **README**:`/root/workspace/hstu-bwd-impl/test/README.md`(用法 / 判定规则 / 矩阵 / 里程碑落地后如何升级 case / ctest 接法)
- **运行日志**:`/root/workspace/hstu-bwd-impl/runs/test-<stamp>.log`(每 case 完整输出 + CMD + EXIT + NOTE + REASON + 汇总)
- ctest:**未改 CMakeLists**(遵守「不动 kernel/构建配置」),ctest `add_test` 接法写在 README,需要时一行接入。

## 测试矩阵:13 条(pass 6 / reject 6 / skip 1)
**pass 基线(M1 已验)**:basic-attnscale1、b4-nhead8-seq256、b1-nhead1-seq512、seq200(非 kN0=128 整除)、seq130(非 kM0=32 整除→OOB 归零)、default-attn_scale。
**reject**:causal(M2)、softmax(M5)、jagged(M3)、group-g2(M4)、fp16(M7)、hdim128(M7)。
**skip/N-A**:deterministic(M6)。

## 判定规则
- pass:`exit==0 且 恰 3 个 [PASS] 且 0 个 [FAIL]`(正则**按出现次数**计——三张量在同一行)。
- reject:`exit!=0 且 非 all-3-PASS`。若 reject 居然 exit0+3PASS → 判 FAIL 且提示「该路径可能已实现→升级为 pass」(同时抓假阳 + 忘记升级测试)。
- skip:仅报告 N/A,不影响整体 exit。
- 整体:任一 FAIL → exit 1;否则 0。构建失败 3;二进制缺失/过滤为空 4;每 case 超时(默认 120s)判 FAIL。

## 实跑汇总(本次,整体 exit 0)
```
matrix : 13 cases (pass=6 reject=6 skip=1) timeout=120s
[PASS] pass-basic-attnscale1     exit=0    P/F=3/0   dQ err=1.2e-4 dK=0 dV=3.9e-3 (max|ref|~5)
[PASS] pass-b4-nhead8-seq256     exit=0    P/F=3/0
[PASS] pass-b1-nhead1-seq512     exit=0    P/F=3/0
[PASS] pass-seq200-non-kN0-128   exit=0    P/F=3/0
[PASS] pass-seq130-non-kM0-32    exit=0    P/F=3/0
[PASS] pass-default-attn_scale   exit=0    P/F=3/0
[PASS] reject-causal   M2  exit=-6  (SIGABRT)  what(): causal/mask path not implemented (M2)
[PASS] reject-softmax  M5  exit=-6  (SIGABRT)  what(): softmax path not implemented (M5)
[PASS] reject-jagged   M3  exit=255           no such arg:jagged / Failed to parse
[PASS] reject-group-g2 M4  exit=255           no such arg:g / Failed to parse
[PASS] reject-fp16     M7  exit=253           M0 bwd harness only supports -prec=bf16
[PASS] reject-hdim128  M7  exit=-6  (SIGABRT)  what(): supports hdim_qk=hdim_v=64 only (M7)
[N/A ] skip-deterministic M6 exit=0 P/F=3/0   (不断言)
================================================================================
TOTAL 13   PASSED 12   FAILED 0   SKIPPED 1   RESULT: OK
OVERALL_EXIT=0
```

## 关键发现(实测,非臆造——已据此校正矩阵)
1. **harness 无 `-jagged` / `-g`(group)开关**:传未知 flag → 解析失败 exit 255(`no such arg`)。当前 jagged/group 的拒绝是**CLI-parse 层**(非 dispatch 路径拒绝),仍满足 reject 判据(兜底「不会悄悄算出结果还报 PASS」)。README 已标注:M3/M4 给 harness 加真开关后,更新这两条 args 成路径级拒绝,再随实现升级为 pass。
2. **`-deterministic=1` 实际 exit0 + PASS**:entry 把 `kIsDeterministic` 硬编码为 false、未把该 CLI 轴接到 dispatch 模板,所以走 atomic 路并通过 → **不能**作为 reject,标 **skip/N-A**(对齐任务「可跳过/标 N/A」)。M6 接通模板轴后再改。
3. reject 判定**不会把 abort/throw 误判为 pass**:SIGABRT 在 Python subprocess 下 returncode=-6、parse=255、guard=253,均 `!=0` 且 0 个 PASS → 正确拒绝。已用注入式 dry-run 验证:伪造一个「reject 却返回 exit0+3PASS」→ runner 正确判 FAIL 并给出升级提示。

## 鲁棒性(已实测)
- `--bin` 缺失/不可执行 → 友好报错 + exit 4。
- `--filter <substr>` 子集运行(实测 `--filter reject` = 6 条全绿)。
- `--build` 可选先 `cmake --build ... --target tile_example_hstu_attention_bwd`。
- `[PASS]/[FAIL]` 正则精确匹配避免子串误判;每 case 超时保护。
- 未改 fwd/bwd kernel、未放宽容差(容差在 harness 内,runner 只读判定)。

## 用法一句话
`python3 test/run_bwd_tests.py`(默认二进制路径,跑全 13 条;`--build` 先编、`--filter` 过滤、`--bin/--timeout/--log-dir` 可调),整体 exit 0=全达预期。
