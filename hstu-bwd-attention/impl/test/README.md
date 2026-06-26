# HSTU bwd 回归测试套件

自动化 runner，作为 HSTU attention backward 每个里程碑（M2–M8）的回归闸门。
驱动 harness 二进制 `tile_example_hstu_attention_bwd` 跑一个固定测试矩阵，逐 case
断言「应 PASS 的真 PASS / 未实现的被正确拒绝」，任一不达预期则整体非 0 退出（CI 友好）。

纯 `python3 + subprocess`，无第三方依赖。

## 怎么跑

```bash
# 默认二进制路径 = /root/workspace/ck_hstu/build/bin/tile_example_hstu_attention_bwd
python3 test/run_bwd_tests.py

# 先 build 再测（cmake --build 该 bwd target）
python3 test/run_bwd_tests.py --build

# 只跑名字含某子串的 case
python3 test/run_bwd_tests.py --filter pass
python3 test/run_bwd_tests.py --filter reject-causal

# 指定二进制 / 超时 / 日志目录
python3 test/run_bwd_tests.py --bin /path/to/binary --timeout 180 --log-dir /tmp/logs
```

每次运行把完整逐 case 输出写到带时间戳的日志：
`/root/workspace/hstu-bwd-impl/runs/test-<YYYYmmdd-HHMMSS>.log`。

## 判定规则

每条 case 标 `expect ∈ {pass, reject, skip}`：

| expect | 通过判据 |
|---|---|
| **pass** | `exit==0` **且** stdout 恰好 3 个 `[PASS]`、0 个 `[FAIL]` |
| **reject** | `exit!=0` **且 非** all-3-PASS（即未实现路径被正确拒绝） |
| **skip** | 不断言，仅报告 `N/A`（不影响整体 exit） |

- `[PASS]`/`[FAIL]` 用正则**精确匹配**计数（三张量在同一行，按出现次数而非行数计）。
- reject 若**居然** `exit==0` 且 all-PASS → 判 FAIL，并提示「该路径可能已实现 → 把此 case 升级为 `expect="pass"`」。这能同时抓住「假 PASS」和「里程碑落地后忘记升级测试」。
- 超时（默认 120s）→ 判 FAIL。
- 整体退出码：有任一 FAIL → 1；仅有 skip/全 PASS → 0；构建失败 → 3；二进制缺失/过滤为空 → 4。

## 当前测试矩阵（13 条：pass 6 / reject 6 / skip 1）

实测基线（gfx950 / MI350X，2026-06-05；reject 的当前拒绝机制见下）：

**PASS 基线（M1 已验，SiLU·no-mask·bf16·hd64·atomic）**
- `pass-basic-attnscale1`：b2 nhead2 seq128 `-attn_scale=1.0`
- `pass-b4-nhead8-seq256`、`pass-b1-nhead1-seq512`
- `pass-seq200-non-kN0-128`（seq 非 kN0=128 整除）
- `pass-seq130-non-kM0-32`（seq 非 kM0=32 整除 → 验 OOB buffer_load 归零）
- `pass-default-attn_scale`（默认 `scale_p=1/max_seqlen_q`，梯度极小仍须 PASS）

> 基线默认带 `-attn_scale=1.0` 把梯度量级放大到 ~5，才能真正检验数值正确（默认 scale 梯度 ~0.04，PASS 区分力弱，仅保留 1 条覆盖默认路径）。

**REJECT（未实现路径，不得出现 all-PASS）—— 注意「当前拒绝机制」各不相同**
| case | 里程碑 | 当前拒绝机制 | exit |
|---|---|---|---|
| `reject-causal` | M2 | dispatch `if constexpr` throw → SIGABRT | -6 |
| `reject-softmax` | M5 | dispatch `if constexpr` throw → SIGABRT | -6 |
| `reject-hdim128` | M7 | dispatch 运行时 throw（非 hd64）→ SIGABRT | -6 |
| `reject-fp16` | M7 | harness guard「only supports -prec=bf16」 | 253 |
| `reject-jagged` | M3 | **CLI-parse reject**（harness 尚无 `-jagged` flag） | 255 |
| `reject-group-g2` | M4 | **CLI-parse reject**（harness 尚无 `-g`/group flag；`GroupBwdParams` 空 struct、无入口） | 255 |

> ⚠️ `reject-jagged` / `reject-group-g2` 现阶段是**因未知 CLI flag 而解析失败**被拒，而非 dispatch 路径拒绝（harness 还没有这两个开关）。它们仍满足 reject 判据（非 0 退出、无 all-PASS），起到「不会悄悄算出 jagged/group 结果还报 PASS」的兜底作用。等 M3/M4 给 harness 加上 `-jagged`/`-g` 开关后，应同时更新这两条 case 的 args，使其变成真正的「路径级拒绝」，再随实现落地升级为 `pass`。

**SKIP / N-A**
- `skip-deterministic`（M6）：harness 有 `-deterministic` flag，但 **entry 把 `kIsDeterministic` 硬编码为 false**、未把该 CLI 轴接到 dispatch 模板，故 `-deterministic=1` 实际仍走 atomic 路并 PASS。它**不能**作为有意义的 reject，标 N/A。M6 把该模板轴接通后再改为 pass/reject。

## 新里程碑落地后如何升级 case

里程碑 Mx 实现并对拍通过后：
1. 找到对应 `reject-*`（或 `skip-deterministic`）条目，把 `expect` 由 `"reject"`/`"skip"` 改为 `"pass"`；
2. 校正其 `args`（如 M3 给 harness 加了真正的 `-jagged` 开关、M5 的 `-softmax=1` 现已实现），并按需带 `-attn_scale=1.0`；
3. `note` 改成实现说明；
4. 跑 `python3 test/run_bwd_tests.py --filter <name>` 确认该 case 变绿，再跑全量。

矩阵集中在 `run_bwd_tests.py` 顶部的 `MATRIX` 列表，一处维护。

## ctest 接法（可选，未改 CMakeLists）

为遵守「只加 test/、不动 kernel/构建配置」，本套件**未**修改 `CMakeLists.txt`。
若要纳入 ctest，可在 `example/ck_tile/18_hstu_attention/CMakeLists.txt` 的 bwd target
（`tile_example_hstu_attention_bwd`，`EXCLUDE_FROM_ALL`）之后追加：

```cmake
add_test(
  NAME hstu_bwd_regression
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../../../hstu-bwd-impl/test/run_bwd_tests.py
          --bin $<TARGET_FILE:tile_example_hstu_attention_bwd>)
```

注意：bwd target 是 `EXCLUDE_FROM_ALL`，ctest 前须先显式
`cmake --build build --target tile_example_hstu_attention_bwd`，否则二进制不存在
（runner 会给出友好报错并以 exit 4 退出）。直接用 `--build` 让 runner 自己 build 也可。
