# M7a fp16 bwd 实现派单 (pane-1 / 主 coder)

> 放宽"bwd 仅 bf16"限制,加 **fp16** dtype。范围:fp16 覆盖现有全部 bf16 路径(no_group batched/jagged + group;SiLU + softmax + atomic/determ)。hdim 变体 / hdim_qk≠hdim_v = M7b/M7c 后续,本单不做(保持 hd64)。
> 性质:**最小增量**——dispatch/kernel/pipeline/reference 本就按 `InOutDataType` 模板化,fp16 主要是"补 instance + entry + harness 接线",非新逻辑。
> 铁律:对拍 CPU reference(fp16 阈值用现成 `get_bwd_elimit<fp16_t>`,line ~151);bf16 已 promoted 路径**零回归**。

## 0. 现状(survey 已做,给你省时)
- harness `example_hstu_attention_bwd.cpp` **已按 `InOutDataType` 模板化**(q_host/fill/对拍全泛型);`get_bwd_elimit<ck_tile::fp16_t>` 已存在(:151)。缺的只是:`prec=fp16` 时实例化 `run_*_hstu_bwd<fp16_t>` + 调 fp16 fwd/bwd entry + 去掉 bf16-only throw(:390/:486)。
- **bwd entry 仅 bf16**(api.hpp:19-21:`hstu_attention_no_group_backward_bf16`/`..._group_backward_bf16`)。**fwd 已有 fp16 entry**(`hstu_attention_no_group_forward_fp16.cpp`/`..._group_forward_fp16.cpp`)——直接照它的样子做 bwd fp16 entry。
- `generate_instances.py` fwd 已 `for dtype in ["fp16","bf16"]`(:95);**bwd instance 目前 bf16-only**,要给 backward 生成也加 dtype 轴。
- dispatch `run_*_backward_dispatch<InOutDataType,...>` 已模板化,fp16 实例化即可。

## 1. bwd fp16 entry(仿 fwd fp16 entry)
- 新 `hstu_attention_no_group_backward_fp16.cpp` + `hstu_attention_group_backward_fp16.cpp`:与对应 `_bf16.cpp` 一字不差,只把 `ck_tile::bf16_t` → `ck_tile::fp16_t`、函数名 `_bf16`→`_fp16`、include 的 instances_ref 若按 dtype 命名则同步。
- `hstu_attention_api.hpp`:加 `extern ... hstu_attention_no_group_backward_fp16(...)` + `..._group_backward_fp16(...)`。
- `CMakeLists.txt`:BWD entry/instance 源若按 dtype 列,加 fp16 的(参照 fwd fp16 怎么进 build)。

## 2. generate_instances.py:backward 加 dtype 轴
- `create_backward_instances` / `create_backward_instances_ref` 现只生成 bf16;改成 `for dtype in ["fp16","bf16"]`(对齐 fwd 的 :95 写法),文件名 `_{dtype}_` 段、`HSTU_DTYPE_MAP`/`HSTU_DTYPE_STR_MAP` 复用。重生成 → no_group bwd 出 fp16+bf16 两套(× causal × softmax × determ)+ ref。
- group 是直接实例化(无 extern instance),只靠 §1 的 fp16 entry。
- **注意编译体积**:bwd instance 翻倍(dtype 轴)。可接受,留意编译时长。

## 3. harness(example_hstu_attention_bwd.cpp)
- `prec` arg 描述更新(去掉"fp16 not in M0")。
- main() 里按 `prec` 分发:`bf16 → run_*_hstu_bwd<ck_tile::bf16_t>`、`fp16 → run_*_hstu_bwd<ck_tile::fp16_t>`(现应只有 bf16 分支)。
- fwd 调用(:387-390):fp16 时调 `hstu_attention_no_group_forward_fp16` / group 的 fp16 fwd(softmax 产 LSE 路);去掉 throw。
- bwd 调用(:483-486):fp16 时调 `hstu_attention_no_group_backward_fp16` / `..._group_backward_fp16`;去掉 throw。
- group 段同理接 fp16 fwd/bwd。
- fp16 数值范围:fp16 动态范围小,若 FillUniform[-1,1] + attn_scale=1.0 下 fp16 溢出/精度不够,可酌情(但**别为了过对拍缩小输入**——铁律 §3④;用 `get_bwd_elimit<fp16_t>` 的既定阈值,过不了如实报)。

## 4. 验证(对拍,-attn_scale=1.0)
build:`cmake --build build --target tile_example_hstu_attention_bwd -j$(nproc)`。
- fp16 × {SiLU,softmax} × {batched,jagged,group} × causal{0,1} × 几个 mask 因子 × {atomic,determ} 代表性对拍,阈值 `get_bwd_elimit<fp16_t>`。
- **bf16 零回归**:全套件仍 91/91(fp16 是新增轴,bf16 案不动)。
- `reject-fp16` 测试案(现期望 reject)→ 升级为 fp16 pass 案。
- 日志 `runs/run-M7a-*.log`。

## 5. 落地产出(交 lead,**别自标 promoted——等 lead 独立验证 + pane-2 复核**)
1. 套件加 fp16 案 + `reject-fp16` 升级;`python3 test/run_bwd_tests.py` exit 0。
2. `/tmp/hstu-bwd-design/M7a-done.md`(改了哪些文件、fp16 复用边界、对拍逐档数值、bf16 零回归证据、fp16 精度注意点、坑)。
3. candidates 加 `M7a-fp16`,**状态先写 in-progress**,reason 如实(含对拍结果);promote 由 lead 裁决。
4. 不动 hdim(M7b)、不动 fwd 逻辑(只 harness 调 fp16 fwd entry)。

## 6. 速查
- fwd fp16 entry 参照:`hstu_attention_no_group_forward_fp16.cpp` / `hstu_attention_group_forward_fp16.cpp`。
- bwd bf16 entry(要镜像):`hstu_attention_no_group_backward_bf16.cpp` / `hstu_attention_group_backward_bf16.cpp`。
- generate_instances fwd dtype 轴:`generate_instances.py:95` + `HSTU_DTYPE_MAP`(:77)。
- fp16 阈值:`get_bwd_elimit<fp16_t>`(example_hstu_attention_bwd.cpp:151)。
- 参考报告:`M6-done.md`(determ 轴怎么加 instance)、`M5b-done.md`(group entry)。
