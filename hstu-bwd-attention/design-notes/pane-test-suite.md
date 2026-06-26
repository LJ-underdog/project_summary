# 派给 pane-2(角色:coder)— 写 HSTU bwd 自动化回归测试套件

调度模式:tmux pane-2。你刚深审过 bwd 代码,最清楚 harness CLI 与 dispatch 的 throw 门控。现在写一个**自动化测试 runner**,作为后续每个里程碑(M2–M8)的回归闸门。不要派 sub-teammate。

## 背景
- 被测二进制:`/root/workspace/ck_hstu/build/bin/tile_example_hstu_attention_bwd`(harness)。CLI 用 `-key=value`;关键参数:`-prec`(bf16/fp16)、`-b -nhead -hdim_qk -hdim_v -seqlens -softmax -causal -attn_scale -v -jagged -g`(完整见 `./tile_example_hstu_attention_bwd -?`)。
- harness 行为:对拍 CPU reference,打印每张量 `dQ:/dK:/dV: max_abs_err=...`、`[PASS]/[FAIL]`、末行 `numeric_pass=...`,**exit 0=PASS / -2=数值 FAIL**;未实现路径(dispatch throw)→ 非 0 退出(uncaught throw → abort)。
- 当前实现状态(你 review 已确认):
  - **应 PASS**:batched × SiLU(`-softmax=0`)× no-mask(`-causal=0`)× **bf16** × hdim_qk=hdim_v=64 × atomic。
  - **预期被拒(未实现,应非 0 退出且不得出现假 [PASS])**:`-causal=1`(M2)、`-softmax=1`(M5)、`-jagged=1`(M3)、`-g>1`(M4 group,无入口)、`-prec=fp16`(M7,入口仅 bf16)、`hdim≠64`(M7,运行时 throw)、deterministic(M6,harness 暂无开关→可跳过/标 N/A)。

## 交付
1. **测试 runner**:`/root/workspace/hstu-bwd-impl/test/run_bwd_tests.py`(纯 python3 + subprocess,无第三方依赖)。
   - 内置**测试矩阵**:每条 `{name, args(list), expect: "pass"|"reject"}`。
     - **pass 基线**(对齐 M1 已验,全用有意义梯度量级即带 `-attn_scale=1.0`,另含默认 scale 1 条):
       - 基本:`-b=2 -nhead=2 -hdim_qk=64 -hdim_v=64 -seqlens=128 -softmax=0 -causal=0 -attn_scale=1.0`
       - 变体:`b=4,nhead=8,seqlens=256`;`b=1,nhead=1,seqlens=512`;**非整除** `seqlens=200`、`seqlens=130`;默认 attn_scale(去掉 `-attn_scale`)各 1 条。
     - **reject 项**(应非 0 退出、且输出**不得**三个全 [PASS]):`-causal=1`、`-softmax=1`、`-jagged=1`、`-g=2`、`-prec=fp16`、`-hdim_qk=128 -hdim_v=128`。
   - 每 case:`subprocess.run` 跑二进制,**超时**(如 120s),捕获 stdout/exit code。
     - expect=pass:判定 = `exit==0 且 stdout 含 3 个 [PASS] 且无 [FAIL]`。
     - expect=reject:判定 = `exit!=0 且 NOT(三个 [PASS])`(即正确拒绝;若它居然 PASS 了 = 回归/假阳,标 FAIL 并提示"该路径已实现?需把此 case 升级为 pass")。
   - **汇总**:逐 case 打印 `PASS/FAIL + name + exit + 关键 err`;末尾统计 `total/passed/failed`;**任一不达预期 → 整体 exit 非 0**(CI 友好)。
   - 把完整结果写时间戳日志到 `/root/workspace/hstu-bwd-impl/runs/test-<stamp>.log`(stamp 用 `subprocess date +%Y%m%d-%H%M%S` 或 python 传入,**注意本环境 python 无 Date.now 限制不影响 subprocess `date`**)。
   - 参数:`--bin <path>`(默认上面路径)、`--build`(可选,先 `cmake --build ... --target tile_example_hstu_attention_bwd`)、`--filter <substr>`。
2. **README/用法**:`/root/workspace/hstu-bwd-impl/test/README.md` —— 怎么跑、如何在新里程碑落地后把对应 reject case 升级为 pass、判定规则。
3. (可选,做了更好)`ctest` 集成:在 `18_hstu_attention/CMakeLists.txt` 的 bwd target 后加 `add_test(NAME hstu_bwd_regression COMMAND python3 .../run_bwd_tests.py)`(EXCLUDE_FROM_ALL 体系下注意 test 依赖 target 已 build;不确定就只在 README 写 ctest 接法,不强加)。

## 自验(必须真跑)
- 实跑 `python3 test/run_bwd_tests.py`,**确认 pass 基线全过、reject 项全部被正确拒绝**,整体 exit 0。
- 故意制造一次"应 reject 却 pass"不现实(代码就是 throw),但要确认 reject 判定逻辑不会把 abort/throw 误判为 pass。
- 把本次运行日志留在 runs/。

## 铁则
- runner 要稳:超时保护、二进制不存在时友好报错、解析 [PASS]/[FAIL] 用精确匹配(避免子串误判)。
- 不改 fwd/bwd kernel 代码(只加 test/ 与可选 CMake test);不放宽容差(容差在 harness 内,runner 只读判定)。
- 完成写 `/tmp/hstu-bwd-design/test-suite-done.md`:runner 路径、矩阵条数(pass/reject)、**实跑汇总结果(各 case 判定 + 整体 exit)**、是否加了 ctest、用法一句话。
- progress 简洁;长输出进 runs/ 日志。
