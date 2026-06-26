# M7a fp16 WIP 对抗式 review —— reviewer(pane 0.2)

你是 HSTU bwd 项目的独立 reviewer(较真、对抗式)。上轮某 session 留下一批**未提交、未测试、来源不明的 M7a fp16 WIP**(纯 dtype 加宽:fp16 复用 bf16 模板码路,hdim 仍 64),已恢复进工作树。coder(pane 0.1)在并行建+对拍+补套件。**你的职责是独立审代码 + 验对拍公平性,默认怀疑,找洞。** 不信任 WIP 作者的自述。

## 0. 背景(只读)
1. `/root/workspace/hstu-bwd-impl/docs/HANDOFF.md`(铁律 §3:验证=对拍 CPU reference bf16 rel≤2e-2/abs≤5e-2;能力边界)。
2. WIP diff:`git -C /root/workspace/ck_hstu diff HEAD -- example/ck_tile/18_hstu_attention/` + 新文件 `hstu_attention_{no_group,group}_backward_fp16.cpp`、`instances/*fp16*`。
3. 基线 HEAD=`d4fb2884`(M6b),bf16 套件 91/91。

## 1. 对抗审查清单(逐条给 GREEN/RED + 证据行号)
1. **对拍公平性(最关键)**:harness reference 是否也走 fp16 I/O?(`run_*_hstu_bwd` 用 `InOutDataType` host tensor + `reference_hstu_attention_bwd`)。确认 fp16 时 GPU 与 reference **同 dtype I/O、同 float 内算**——否则一侧 fp16 一侧 bf16 是 silent-unfair。
2. **容差是否放水**:`get_bwd_elimit<fp16_t>` = rtol5e-3/atol1e-2(比 bf16 紧)。确认这是 fp16 物理(尾数 10bit>bf16 7bit)合理收紧,**不是**被调来掩盖误差;反向验:故意把 fp16 容差调到 bf16 的 2e-2/5e-2,是否仍 PASS(若放宽后才过=有隐藏误差,RED)。验完改回。
3. **fp16 路真的在跑 fp16**:`main()` 的 `-prec=fp16` 分支是否真实例化 `run_*<fp16_t>`,而非静默落回 bf16。`BOOL_SWITCH_3` fan-out 是否与 bf16 一致(causal×softmax×determ,O1 式 hardcode 隐患复查 group entry)。
4. **instance 完整性**:8 个 fp16 no_group instance = 4 atomic + 4 determ × causal×softmax,与 `generate_instances.py` dtype 轴 `["fp16","bf16"]` 一致;ref.hpp extern template 与 instance 定义一一对应(漏一个=链接错或落回)。group 是 direct-instantiation 无 instance 文件(确认)。
5. **CMake**:fp16 fwd instance(`*forward_fp16*`)+ fp16 entry 是否真链入 bwd target(harness 跑 GPU fwd 产 O/LSE 需要)。
6. **bf16 零回归(byte-level)**:WIP 改的共享文件(CMakeLists/example_bwd.cpp/generate_instances.py/api.hpp)对 bf16 行为是否纯加性?reference/库 kernel/pipeline/dispatch **应 byte-identical 于 d4fb2884**(`git diff HEAD -- <那些库文件>` 应为空)。确认 WIP 没碰库逻辑。
7. **fp16 溢出风险**:softmax exp/LSE/大梯度在 fp16 I/O 边界是否可能 inf/nan?给出你判断的安全性论证或反例配置(交 coder 实测)。

## 2. 产出(写 `/tmp/hstu-bwd-design/M7a-review-findings.md`)
逐条 GREEN/RED + 文件:行号证据;容差 revert 实验结果;任何 silent-wrong 隐患。**发现 RED 必须给可复现配置**。结论:M7a 是否可在对拍闭合后 promote,还是有阻塞洞。

干净重建你自己的 build 目录或复用都行;若与 coder 抢同一 build 冲突,用独立 `-B build_review`。完成后 pane 里一句话报结论,等 lead 汇总。
