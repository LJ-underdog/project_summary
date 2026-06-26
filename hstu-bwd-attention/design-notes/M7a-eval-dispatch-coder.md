# M7a fp16 WIP 评估 —— coder(pane 0.1)

你是 HSTU bwd 项目主 coder。上轮某 session 留下一批**未提交的 M7a fp16 WIP**(纯 dtype 加宽:fp16 复用 bf16 同一条模板码路,hdim 仍 64),已从 stash 恢复进工作树。lead 已亲读确认结构忠实、容差更紧(没放水)。**现在的任务是把它"建成 + 对拍 + 补套件"验真,不是重写。** 全程 `-attn_scale=1.0`。

## 0. 背景必读(按序,只读)
1. `/root/workspace/hstu-bwd-impl/docs/HANDOFF.md`(环境/构建 BUILD_DEV=OFF/铁律/能力边界 M0–M6b)。
2. WIP 改了什么:`git -C /root/workspace/ck_hstu status` + `git -C /root/workspace/ck_hstu diff HEAD -- example/ck_tile/18_hstu_attention/`(4 改 + fp16 entry×2 + 8 instance + ref.hpp + harness fp16 elimit/-prec 选型)。
3. 现基线:HEAD=`d4fb2884`(M6b),干净 bf16 套件 **91/91 exit 0** 已由 lead 复核。

## 1. 构建(BUILD_DEV=OFF,gfx950)
```
cd /root/workspace/ck_hstu
cmake --build build --target tile_example_hstu_attention_bwd -j$(nproc) 2>&1 | tee /root/workspace/hstu-bwd-impl/runs/build-M7a.log
```
若 CMake 没自动 re-glob 到新 fp16 instance/entry,先 `cmake -B build ...`(完整配置行见 HANDOFF §1)再 build。**0 error 才往下**;有 error 把首条贴出来分析(别强行往下)。

## 2. fp16 对拍 sweep(核心,全 `-attn_scale=1.0 -prec=fp16`)
镜像 M5/M5b 已验过的 bf16 配置,但 `-prec=fp16`,覆盖:
- **模式** batched / jagged(`-jagged`,per-batch 逗号 seqlen)/ group(`-g=2,3,4`)
- **softmax** {0=SiLU, 1=softmax} × **causal** {0,1}
- **mask 5 因子** 各自 + 组合(window `-g_local_lens`/contextual/min_full/num_target `-targets`)
- **determ** `-deterministic=1`(no_group + group)
- **非整除 seqlen**(如 200/300/512/96)、单 batch、tiny(1/7)
- **关键交叉**:`causal=0 + num_target>0`(P1-1 老洞)、group 同组多 batch 异 seqlen + 长 batch 大 target(M6b harness 老洞触发配置)
逐配置记 PASS/FAIL + dQ/dK/dV max_abs_err vs max|ref|,落 `runs/run-M7a-sweep.log`。**fp16 容差 rtol5e-3/atol1e-2 已设;若有 FAIL,先判是数值真错还是容差问题——别动容差去凑 PASS,如实报。**

## 3. fp16 数值风险自检(fp16 指数位仅 5bit,max~65504)
- softmax 路:S/exp/LSE 是否可能 fp16 溢出?(注:内部累加是 float,fp16 仅 I/O;但确认 O/dO/Q/K/V 量级在 fp16 范围内,FillNormal{0,1} 应安全)。
- 若某 softmax 配置因 fp16 溢出 FAIL,**如实记录为 fp16 已知边界**,不要硬调。

## 4. bf16 零回归(必须)
跑全套确认 bf16 路没被 WIP 碰坏:
```
python3 /root/workspace/hstu-bwd-impl/test/run_bwd_tests.py 2>&1 | tail -5
```
预期仍 **91/91 exit 0**。

## 5. 补 fp16 测试套件(收尾)
套件现全 `-prec=bf16`、无 fp16 案。**新增 fp16 案**:镜像每模式的代表性 bf16 案到 fp16(no_group/group × SiLU/softmax × causal{0,1} × 几个 mask 因子 + determ + ≥2 个 fp16 determ byte-identical 复现案)。再跑一次,记新 TOTAL。确保 fp16 案是 **pass 断言**(不再是 reject)。改完 `runs/test-<ts>.log`。

## 6. 产出(写 `/tmp/hstu-bwd-design/M7a-done.md`)
- 构建 0 error 证据;sweep PASS/FAIL 全表(err vs |ref|);bf16 零回归证据;新套件 TOTAL/PASS/FAIL/SKIP exit;fp16 已知边界(若有)。
- **诚实纪律**:带任何 FAIL 不许标 promoted;日志数字与结论必须一致(M6b 首轮过度声称被打回的教训)。完成度/缺口如实列。
- 更新 `/root/workspace/hstu-bwd-impl/candidates.jsonl` 加一行 M7a(status 据实:全绿=in-progress 待 lead+reviewer 闭合;有缺口=如实)。

完成后在 pane 里一句话汇报结果概要,等 lead 裁决闭合。**不要 commit,不要动 reference/库 kernel 逻辑/promoted 路。**
