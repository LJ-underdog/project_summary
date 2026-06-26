# M4 代码改动 review 文档 — 完成 (pane-2 / drafter)

输出:`/root/workspace/hstu-b1052-report/hstu-bwd-M4-changes-20260608.html`(53145 bytes)
代码 review findings:`/tmp/hstu-bwd-design/M4-review-findings.md`(addendum 要求,单独记)
状态:**完成**。忠实当前源码,落笔前 Read 核实每个区段,未派 sub-teammate。结构/风格逐字对齐 M3 文档(同 head/CSS/布局/配色/10 节骨架),M2/M3/M4 三篇观感一致。

## 补充要求(pane-M4-codedoc-addendum)已并入
边写文档边做代码 review,findings 单独落 `/tmp/hstu-bwd-design/M4-review-findings.md`(P0/P1/P2 分级)。
- **发现 1 条 P1**(复跑佐证,非 M4 回归):`causal=0 + num_target>0 + window=0` 时 GPU 静默漏掩码,与 reference 背离(ref 量级误差)。根因 = M2 期 NoLocal struct `IsMasking=kUseCausal` 的设计假设被 num_target 证伪;STAGE2 `if constexpr(IsMasking)` 被编译掉而 reference 无条件掩码。
  - 复跑证据:A) batched `-causal=0 -targets=8` → FAIL(dQ max_abs_err=1.160);E) group `-causal=0 -g=2 -targets=8,24,0,16` → FAIL(2.180)。对照 C/D/B 均 PASS,触发条件锁定为 NoLocal(window=0)+causal=0+num_target>0。
  - 当前**未被任何里程碑对拍覆盖**(三里程碑 mask 因子均 causal=1);harness/dispatch 不拒绝 → 静默错误。建议(交 lead):守门 throw / 改运行时门控 / 文档化不支持 + 加 reject 测试。**未擅自改 kernel**。
- **P0:无**。group 在其验证包络内(causal=1 × per-group 因子 × 多 group)与 reference 逐公式一致。
- **P2**:① 补 causal=0×factor 负向锁定测试;② 双 pipeline 体积为既定 perf 取舍非 bug。
- 逐项核验通过(无 P0/P1):i_group 索引/scale_p fallback/min_full 钳制/alpha 全局/num_target per-batch/双 pipeline 选择/GetSmemSize/no_group 零回归/grid early-exit/dq_acc sizing/host supplement 长度(无 M2 式越界)/M3 offset 真复用/packed 溢出 —— 均与 reference 一致,详见 findings。
- **文档侧**:§10 加一条中性「验证包络」口径说明(mask 因子均 causal=1 对拍;causal=0 为 no-mask 路径,叠因子属未验证区间、已留档交 lead),P1 细节不入文档(按铁则在 findings 单列)。

## 自检
- **字节**:52661 bytes
- **标签平衡**:div 73/73 · section 10/10 · svg 2/2 · table 3/3 · tr 28/28 · ul/tbody/thead/nav/main/head/body/style/**p** 全配对 —— 全部平衡(初版有 1 处 gate-banner 多余 </p>,已修)
- **TOC 锚点**:10 个 href 全部命中 section id(s1–s10),0 缺失
- **SVG**:2 张(要求 ≥2),viewBox + width=100% 防溢出
- **代码转义**:grep 无 raw 模板尖括号泄漏
- **样式**:逐字复用 M3 head/CSS(含 .ttl/.tag-noise),左正文右 sticky TOC,clay=新增改动/olive=复用/slate

## 小节(10 节,2 SVG)对齐 M3 骨架
1. 总览+TL;DR:group=jagged 超集 + per-group device-ptr 超参(i_group 取) + 运行时双 pipeline 选;alpha 全局/num_target per-batch;徽章 8/8 PASS(含全异构+g4)·34 案 exit0·no_group 零回归·无 instance 噪音
2. 改动文件清单表:params/kernel(新 GroupKernel)/dispatch(新)/entry(新)/api/cmake/harness/test;遗留噪音行 + mtime 证据
3. 【图1 SVG】group 取数模型:i_batch→i_group=i_batch/num_batch_per_group,per-group 设备数组,三种粒度(alpha 全局/scale_p+mask per-group/num_target per-batch)
4. 【图2 SVG】运行时双 pipeline(M4 核心难点):per-group window 不能编译期定→实例化 with/without-local 两 pipeline,kernel if(window>0) 选,共享 write_dkdv;贴 kernel L775-798 + dispatch L98-117
5. per-group scale_p/mask 取数:i_group readfirstlane + 5 设备指针 + scale_p fallback + num_target per-batch;贴 L643-675
6. params 填充 + dispatch 门控/grid/POST + entry BOOL_SWITCH_2 直接实例化 + CMake
7. harness -g:per-group/per-batch supplement + group_max_seqlens_q + cu_seqlens + reference_group(BOOL_SWITCH_2,无 kIsJagged 轴)+ SiLU 跳过 GPU fwd
8. 测试套件:8 group pass case 表;强调全异构/g4 最值钱(取数 bug 放大)
9. 对拍结果:8/8 sweep 表(贴真实 hetero/g4 ARGS 与误差)+ 套件 34/33/1skip/exit0 + 编译 0 error:
10. 遗留:M5 softmax(group O 跳过,需接 group fwd 产 O+LSE)/ cross 仅 self / 双 pipeline 体积 perf M8 / Y-range 保守

## 覆盖的改动文件(均 Read 核实)
- `hstu_attention_bwd_params.hpp`(GroupBwdParams L116+)
- `hstu_attention_bwd_kernel.hpp`(GroupKernel L462+;i_group/取数 L643-675;双 pipeline L775-798;write_dkdv L754-773;no_group kernel 未改)
- `hstu_attention_group_backward_dispatch.hpp`(两 Problem/Pipeline L98-117;门控/grid/POST L157-198)
- `hstu_attention_group_backward_bf16.cpp`(entry BOOL_SWITCH_2 直接实例化)
- `hstu_attention_api.hpp`(group decl)、`CMakeLists.txt`(BWD_INTERFACES_SRCS += group entry)
- `example_hstu_attention_bwd.cpp`(run_group_hstu_bwd L560+;6 个 -g* 参数;supplement/cu_seqlens L600-639;reference_group L800-826)
- `test/run_bwd_tests.py`(8 group pass case L145-176)
- 日志:`runs/run-bwd-M4-sweep.log`(8 小节/8 numeric_pass=true/0 FAIL)、`runs/test-20260608-032623.log`(34/33/1skip/exit0)、`runs/build-bwd-M4.log`(0 error:,链接成功)

## 与 M4-done.md 核出的差异 / 澄清(忠实代码为准)
- **关键诚实点(铁则要求):本次确实无 instance 噪音** —— 核实 `generate_instances.py` 与 `instances/*` mtime=06-04(M2 期),M4 改的 kernel/dispatch/entry mtime=06-08;`ls instances/ | grep group.*backward` 为空(无 group instance 文件);group entry 用 BOOL_SWITCH_2 在 .cpp 直接实例化,不走 instance 文件机制。`git status` 仍显示的 ~203 instance 修改是 M2 期遗留,文档据实标"遗留噪音"。
- **build log "error:"**:`grep -c 'error:'`=0(M4 build 用 28MB 日志,grep 'error' 计数省略,直接核 'error:'=0);末行 `[8/11] Linking ... bin/tile_example_hstu_attention_bwd` 成功。文档据实写"0 error:"。
- **测试日志文件名**:任务 §20 引 032623,M4-done §验收引 025738,两者均存在且结果一致(34/33/1skip);文档引 032623 并注明同结果。
- **套件 reject 组成**:M4-done §验收提"softmax M5 / fp16+hdim128 M7 仍正确拒绝";实测 reject 为 softmax + hdim128(fp16),deterministic 为 1 SKIP。文档据实列(2 reject + 1 skip,共 34=6+9+8+8 pass +2 reject +1 skip... 注:6+9+8+8=31 pass + ? 经核 PASS=33,故 reject 项亦计入 33 之外;文档按日志 TOTAL 34/PASS 33/SKIP 1 如实陈述,未强行拆分各里程碑精确计数以免臆造)。
- **sweep 误差**:文档引真实 hetero 档(max_abs_err≤3.9e-3,max|ref|3.5-7.1)与 g4 档(部分 dQ 逐位 0),取自日志原值,未套用 M4-done 概数。
