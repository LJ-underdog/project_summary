# M3 代码改动 review 文档 — 完成 (pane-2 / drafter)

输出:`/root/workspace/hstu-b1052-report/hstu-bwd-M3-changes-20260608.html`
状态:**完成**。忠实当前源码,落笔前 Read 核实每个区段,未派 sub-teammate。结构/风格完全对齐 M2 文档(同 head/CSS/布局/配色/10 节骨架)。

## 自检
- **字节**:50132 bytes
- **标签平衡**:div 86/86 · section 10/10 · svg 2/2 · table 3/3 · tr 28/28 · nav/main/head/body/style 全 1/1 · ul/tbody/thead 全配对 —— 全部平衡
- **TOC 锚点**:10 个 href 全部命中 section id(s1–s10),0 缺失
- **SVG**:2 张(要求 ≥2),viewBox + width=100% 防溢出
- **代码转义**:grep 无 raw 模板尖括号泄漏
- **样式**:逐字复用 M2 文档 head/CSS(含 .ttl/.tag-noise),左正文右 sticky TOC,clay=新增改动/olive=复用/slate

## 小节(10 节,2 SVG)对齐 M2 骨架
1. 总览+TL;DR:M3=同一 SiLU MAIN kernel 运行时 is_jagged 分支,不新增实例;徽章 10/10 PASS·27 案 exit0·batched 不回归·无 instance 噪音
2. 改动文件清单表:kernel/dispatch/harness/test 标 M3 改动;bwd 文件未跟踪说明;**遗留噪音行**诚实标注
3. 【图1 SVG】batched(i_batch*batch_stride,等长含 padding) vs jagged(offsets[b]*stride,token-major packed [1,ΣL,H,D]),同一 kernel
4. kernel is_jagged 分支:kargs 3 字段 + base offset 分支 + 覆盖 seqlen + early-exit;3 个 callout(覆盖seqlen/early-exit/batched 零变化)
5. 【图2 SVG】jagged 数据流/对拍:harness packed+cu_seqlens → GPU bwd(jagged)+CPU reference<kIsJagged=true> → check_err;3 个同源 callout
6. dispatch:MakeKargs 传 offsets + dq_acc sizing + grid.x seqlen + Run 无 jagged throw
7. harness -jagged:cu_seqlens 前缀和 + batches_for_alloc + BOOL_SWITCH_3 喂 reference
8. 测试套件:8 个 jagged pass case 表 + 意图(大段差/非整除/tiny 能抓 offset/stride bug)
9. 对拍结果:10/10 jagged sweep 表(忠实日志计数:10 numeric_pass=true/0 FAIL/0 非零 EXIT)+ 套件 27/26/1skip/exit0 + 编译 0 error:
10. 遗留:M4 group(jagged 超集,offset 可复用)/ cross-attn 仅 self / perf M8

## 覆盖的改动文件(均 Read 核实)
- `hstu_attention_bwd_kernel.hpp`(kargs L69–75/124–126/168–170;分支+early-exit L247–280)
- `hstu_attention_batched_backward_dispatch.hpp`(MakeKargs L126–137;dq_acc/grid L171–186;Run L209–214)
- `example_hstu_attention_bwd.cpp`(-jagged L138/154–203;BOOL_SWITCH_3 reference L430–462)
- `test/run_bwd_tests.py`(8 个 jagged pass case)
- 日志:`runs/run-bwd-M3-sweep.log`(10 小节/10 numeric_pass=true/0 FAIL)、`runs/test-20260608-022311.log`(TOTAL 27/PASS 26/SKIP 1/exit0)、`runs/build-bwd-M3.log`(0 error:)

## 与 M3-done.md 核出的差异 / 澄清(忠实代码为准)
- **关键诚实点(铁则要求):本次确实无 instance 噪音**。核实 `instances/*` 与 `generate_instances.py` 的 mtime=06-04,而 M3 改的 kernel/dispatch/harness mtime=06-08 —— M3 一个 instance 没动。`git status` 仍显示的 ~203 个 instance 修改是 **M2 期遗留**(里程碑间未 commit),文档已据实标为"遗留噪音"而非照搬 M2 的"198 噪音"说法。
- **build log "error" 澄清**:`grep -c 'error:'` = 0;grep 出的 2 处 "error" 是源码 `check_err(... "dQ error" ...)` 字符串被编译器告警时打印,非编译错误。文档据实写"0 error:"。
- **sweep 日志格式**:M3 sweep 用 `===== 小节 ===== + numeric_pass=true + EXIT=0`(非 M2 的 `^OK` 行)。文档按真实格式计数:10 小节/10 numeric_pass=true/0 FAIL。M3-done 表为 9 行分组,实际日志 10 个独立小节(把 contextual/minfull 拆开),文档列全 10 档。
- **default scale_p jagged**:M3-done 称亦 PASS,但 sweep 日志 10 档均 attn_scale=1.0;文档忠实标注此点,未把 default-scale 算进 10 档。
- **测试日志文件名**:任务 §3 引 022311,M3-done §验收引 020631,两者均存在且结果一致(27/26/1skip);文档引 022311 并注明同结果。
