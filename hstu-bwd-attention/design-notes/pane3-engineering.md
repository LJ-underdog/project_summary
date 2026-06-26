# 派给 pane-3(持久角色:architect)— HSTU bwd GPU 方案 · 第 3 部分「工程落地 & 验证」

调度模式:tmux pane-3。与 pane-1/pane-2 **并行**。先读 `/tmp/hstu-bwd-design/BRIEF.md`。不要派 sub-teammate。

## 你负责:文件/目录结构、problem/traits/policy、dispatch、instances/codegen、dQ 写回、CMake、测试与验证、分阶段里程碑

产出 `/tmp/hstu-bwd-design/part-engineering.md`,涵盖:

1. **文件/目录结构**:在 `18_hstu_attention/` 下新增哪些 bwd 文件(镜像 fwd 命名):如 `hstu_attention_bwd_pipeline_problem.hpp`、`hstu_attention_bwd_pipeline_policy.hpp`(或直接继承 FMHA 的 `block_fmha_bwd_pipeline_default_policy.hpp`)、`hstu_attention_{no,with}_softmax_bwd_pipeline.hpp`、`hstu_attention_bwd_kernel.hpp`(或复用 FMHA `fmha_bwd_kernel.hpp`)、`hstu_attention_{jagged,group,batched}_backward_dispatch.hpp`、`hstu_attention_bwd_dot_do_o_*`、`*_convert_dq_*`。给一张"新建 vs 复用 FMHA"文件映射表。
2. **problem/traits/tile_setting**:bwd 的 `HstuAttentionBwdPipelineProblem`(模板形参:dtypes、kIsJagged/kUseSoftmax/kIsGroupMode/kIsDeterministic、mask 类型、kPad*、hdim_qk/hdim_v)。tile 设置(kM0/kN0/kK0..kK4)如何从 `hstu_attention_tile_setting_define.hpp` 扩展或借 FMHA 预设(gfx942/gfx950)。
3. **dispatch**:三套 backward dispatch(jagged/group/batched),与 fwd dispatch 对称;模板实例化矩阵(dtype × causal × softmax × bias × maxk …)如何收敛规模。
4. **dQ 写回两条路**:atomicAdd(默认)vs deterministic split-workspace + `convert_dq`;`dq_acc` workspace 的分配与 stride;与 params 的 `kIsDeterministic` 联动。
5. **instances + codegen**:`generate_instances.py` 如何扩展生成 bwd instance(.cpp);命名规范;编译规模/时间评估。
6. **CMake**:新增 target、与 fwd 的关系;example 主程序加 bwd 路径与 CLI 参数。
7. **测试与验证(重点)**:**以 `reference_no_group_/reference_group_hstu_attention_bwd` 为 oracle** 做数值对拍;设计:随机输入 → fwd(GPU)产 O/LSE → bwd(GPU)产 dQ/dK/dV → 对比 CPU reference;bf16/fp16 容差(相对/绝对、max/mean err);覆盖 SiLU/softmax × jagged/group/batch × 各 mask 因子 的测试矩阵;deterministic 路验逐位可复现。
8. **分阶段里程碑(MVP→完整)**:建议先 **batched + SiLU + no-mask + bf16 + atomicAdd** 打通端到端对拍,再逐步加 mask 因子 / jagged / group / softmax 路 / deterministic / 多 dtype / 多 maxk。每阶段验收标准。
9. 风险/未决:模板组合爆炸、编译时间、workspace 显存、policy 直接复用 FMHA 的可行性。

## 边界
只设计工程落地与验证;算法(pane-1)、mask/参数语义(pane-2)别重复设计,但要消费它们的接口(problem 模板形参、kargs 字段、mask 类型)——文末列出对二者的依赖假设。

## 铁则
- 参考 fwd 现有文件结构(`hstu_attention_*_forward_dispatch.hpp`、`generate_instances.py`、`CMakeLists.txt`)与 FMHA bwd(`fmha_bwd_kernel.hpp`、codegen `example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`)。
- markdown 到 `/tmp/hstu-bwd-design/part-engineering.md`;完成写 `/tmp/hstu-bwd-design/engineering-done.md`。
